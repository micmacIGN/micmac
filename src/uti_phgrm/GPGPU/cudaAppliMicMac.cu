#include "GpGpu/cudaAppliMicMac.cuh"
#include "GpGpu/cudaTextureTools.cuh"

static __constant__ paramMicMacGpGpu cH;

// ATTENTION : erreur de compilation avec l'option cudaReadModeNormalizedFloat et l'utilisation de la fonction tex2DLayered
texture< pixel,	cudaTextureType2D >			TexS_MaskTer;
texture< float,	cudaTextureType2DLayered >	TexL_Images;
TexFloat2Layered							TexL_Proj_01;
TexFloat2Layered							TexL_Proj_02;

template<int TexSel> __device__ __host__ TexFloat2Layered TexFloat2L();

template<> __device__ __host__ TexFloat2Layered TexFloat2L<1>() { return TexL_Proj_01; };
template<> __device__ __host__ TexFloat2Layered TexFloat2L<2>() { return TexL_Proj_02; };

//------------------------------------------------------------------------------------------

extern "C" textureReference& getMask(){	return TexS_MaskTer;}
extern "C" textureReference& getImage(){ return TexL_Images;}
extern "C" textureReference& getProjection(){return TexL_Proj_01;}

extern "C" void CopyParamTodevice( paramMicMacGpGpu param )
{
	checkCudaErrors(cudaMemcpyToSymbol(cH, &param, sizeof(paramMicMacGpGpu)));
}

template<int TexSel> __global__ void correlationKernel( float *dev_NbImgOk, float* cachVig, uint2 nbActThrd)
{
	__shared__ float cacheImg[ BLOCKDIM ][ BLOCKDIM ];

	// Coordonnées du terrain global avec bordure // __umul24!!!! A voir
	const uint2 ptHTer = make_uint2(blockIdx) * nbActThrd + make_uint2(threadIdx);
	
	// Si le processus est hors du terrain, nous sortons du kernel
	if (oSE(ptHTer,cH.dimTer)) return;

#if (SAMPLETERR == 1)
	const float2 ptProj = tex2DLayeredPt(TexFloat2L<TexSel>(),ptHTer,cH.dimSTer,blockIdx.z);
#else
	const float2 ptProj = tex2DLayeredPt(TexFloat2L<TexSel>(),ptHTer,cH.dimSTer,cH.sampTer,blockIdx.z);
#endif

	if (oI(ptProj,0))
	{
		cacheImg[threadIdx.y][threadIdx.x]  = cH.badVig;
		return;
	}
 	else
#if		INTERPOLA == NEAREST
		cacheImg[threadIdx.y][threadIdx.x] = tex2DLayered( TexL_Images, (((int)ptProj.x )+ 0.5f) / (float)cH.dimImg.x, (((int)(ptProj.y) )+ 0.5f) / (float)cH.dimImg.y,(int)(blockIdx.z % cH.nbImages));
#elif	INTERPOLA == LINEARINTER
		cacheImg[threadIdx.y][threadIdx.x] = tex2DLayeredPt( TexL_Images, ptProj, cH.dimImg, (int)(blockIdx.z % cH.nbImages));
#elif	INTERPOLA == BICUBIC
		cacheImg[threadIdx.y][threadIdx.x] = tex2DFastBicubic<float,float>(TexL_Images, ptProj.x, ptProj.y, cH.dimImg,(int)(blockIdx.z % cH.nbImages));
#endif
		
	__syncthreads();

	const int2 ptTer = make_int2(ptHTer) - make_int2(cH.rVig);
	// Nous traitons uniquement les points du terrain du bloque ou Si le processus est hors du terrain global, nous sortons du kernel
	if (oSE(threadIdx, nbActThrd + cH.rVig) || oI(threadIdx , cH.rVig) || oSE( ptTer, cH.rDiTer) || oI(ptTer, 0))
		return;

	if(tex2D(TexS_MaskTer, ptTer.x, ptTer.y) == 0) return;

	const short2 c0	= make_short2(threadIdx) - cH.rVig;
	const short2 c1	= make_short2(threadIdx) + cH.rVig;
	 
	// Intialisation des valeurs de calcul 
	float aSV = 0.0f, aSVV	= 0.0f;
	short2 pt;
	
	#pragma unroll // ATTENTION PRAGMA FAIT AUGMENTER LA quantité MEMOIRE des registres!!!
	for (pt.y = c0.y ; pt.y <= c1.y; pt.y++)
		#pragma unroll
		for (pt.x = c0.x ; pt.x <= c1.x; pt.x++)
		{	
			const float val = cacheImg[pt.y][pt.x];	// Valeur de l'image

			if (val ==  cH.badVig) return;

			aSV  += val;		// Somme des valeurs de l'image cte 
			aSVV += (val*val);	// Somme des carrés des vals image cte
		}

#ifdef FLOATMATH
		aSV	 = fdividef(aSV,(float)cH.sizeVig );
		aSVV = fdividef(aSVV,(float)cH.sizeVig );
		aSVV -=	(aSV * aSV);
#else
		aSV	 /=	cH.sizeVig;
		aSVV /=	cH.sizeVig;
		aSVV -=	(aSV * aSV);
#endif
	
	if ( aSVV <= cH.mAhEpsilon) return;

	aSVV =	rsqrtf(aSVV); // racine carre inverse

	const uint pitchCache = blockIdx.z * cH.sizeCach + ptTer.x * cH.dimVig.x;
	const uint pitchCachY = ptTer.y * cH.dimVig.y ;
	#pragma unroll
	for ( pt.y = c0.y ; pt.y <= c1.y; pt.y++)
	{
		const int _py	= (pitchCachY + (pt.y - c0.y))* cH.dimCach.x;
		#pragma unroll
		for ( pt.x = c0.x ; pt.x <= c1.x; pt.x++)		
			cachVig[ pitchCache + _py  + (pt.x - c0.x)] = (cacheImg[pt.y][pt.x] -aSV)*aSVV;

	}	

	const int ZPitch = (blockIdx.z / cH.nbImages) * cH.rSiTer;

	atomicAdd( &dev_NbImgOk[ZPitch + to1D(ptTer,cH.rDiTer)], 1.0f);
};

extern "C" void	 KernelCorrelation(dim3 blocks, dim3 threads, float *dev_NbImgOk, float* cachVig, uint2 nbActThrd)
{
	correlationKernel<1><<<blocks, threads>>>( dev_NbImgOk, cachVig, nbActThrd);
	getLastCudaError("Basic Correlation kernel failed");
}

// Calcul "rapide"  de la multi-correlation en utilisant la formule de Huygens	///
__global__ void multiCorrelationKernel(float *dTCost, float* cacheVign, float * dev_NbImgOk, uint2 nbActThr)
{
	__shared__ float aSV [ SBLOCKDIM ][ SBLOCKDIM ];		// Somme des valeurs
	__shared__ float aSVV[ SBLOCKDIM ][ SBLOCKDIM ];		// Somme des carrés des valeurs
	__shared__ float resu[ SBLOCKDIM/2 ][ SBLOCKDIM/2 ];	// resultat
	__shared__ ushort nbIm[ SBLOCKDIM/2 ][ SBLOCKDIM/2 ];	// nombre d'images correcte

	// coordonnées des threads
	const uint2 t = make_uint2(threadIdx);

	if ( threadIdx.z == 0)
	{
		aSV [t.y][t.x]		= 0.0f;
		aSVV[t.y][t.x]		= 0.0f;
		resu[t.y/2][t.x/2]	= 0.0f;
		nbIm[t.y/2][t.x/2]	= 0;
	}
	
	__syncthreads();

 	if ( oSE( t, nbActThr))	return; // si le thread est inactif, il sort

	// Coordonnées 2D du cache vignette
	const uint2 ptCach = make_uint2(blockIdx) * nbActThr  + t;
	
	// Si le thread est en dehors du cache
	if ( oSE(ptCach, cH.dimCach))	return;
	
	const uint2	ptTer	= ptCach / cH.dimVig;						// Coordonnées 2D du terrain

	if(tex2D(TexS_MaskTer, ptTer.x, ptTer.y) == 0) return;

	const uint	iTer	= blockIdx.z * cH.rSiTer + to1D(ptTer, cH.rDiTer);	// Coordonnées 1D dans le terrain
	const bool	mThrd	= t.x % cH.dimVig.x == 0 &&  t.y % cH.dimVig.y == 0 && threadIdx.z == 0;
	const uint2 thTer	= t / cH.dimVig;									// Coordonnées 2D du terrain dans le repere des threads
	
	if (mThrd)
		nbIm[thTer.y][thTer.x] = (ushort)dev_NbImgOk[iTer];

	__syncthreads();

	if (nbIm[thTer.y][thTer.x] < 2) return;
	
	const uint sizLayer = (blockIdx.z * cH.nbImages + threadIdx.z) * cH.sizeCach;	// Taille du cache vignette pour une image

	const uint2 cc		= ptTer * cH.dimVig;										// coordonnées 2D 1er pixel de la vignette
	const int iCC		= sizLayer + to1D( cc, cH.dimCach );						// coordonnées 1D 1er pixel de la vignette

	if (cacheVign[iCC] == cH.DefaultVal) return;									// sortir si la vignette incorrecte
	
	const uint iCach	= sizLayer + to1D( ptCach, cH.dimCach );					// Coordonnées 1D du cache vignette
	const float val		= cacheVign[iCach]; 

	atomicAdd( &(aSV[t.y][t.x]), val);
	atomicAdd(&(aSVV[t.y][t.x]), val * val);
	__syncthreads();

	if ( threadIdx.z != 0) return;

#ifdef FLOATMATH
	atomicAdd(&(resu[thTer.y][thTer.x]),aSVV[t.y][t.x] - fdividef(aSV[t.y][t.x] * aSV[t.y][t.x],(float)nbIm[thTer.y][thTer.x])); 
#else
	atomicAdd(&(resu[thTer.y][thTer.x]),aSVV[t.y][t.x] - ((aSV[t.y][t.x] * aSV[t.y][t.x])/ nbIm[thTer.y][thTer.x])); 
#endif
	
	if ( !mThrd ) return;
	__syncthreads();

	// Normalisation pour le ramener a un equivalent de 1-Correl 
#ifdef FLOATMATH
	const float cost = fdividef( resu[thTer.y][thTer.x], (float)( nbIm[thTer.y][thTer.x] -1.0f) * (cH.sizeVig));
#else
	const float cost = resu[thTer.y][thTer.x]/ (( nbIm[thTer.y][thTer.x] -1.0f) * ((float)cH.sizeVig));
#endif

	dTCost[iTer] = 1.0f - max (-1.0, min(1.0f,1.0f - cost));
}

extern "C" void KernelmultiCorrelation(dim3 blocks, dim3 threads, float *dTCost, float* cacheVign, float * dev_NbImgOk, uint2 nbActThr)
{
	multiCorrelationKernel<<<blocks, threads>>>(dTCost, cacheVign, dev_NbImgOk, nbActThr);
	getLastCudaError("Multi-Correlation kernel failed");

}