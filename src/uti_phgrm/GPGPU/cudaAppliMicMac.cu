#include "GpGpu/cudaAppliMicMac.cuh"
#include "GpGpu/cudaTextureTools.cuh"

// ATTENTION : erreur de compilation avec l'option cudaReadModeNormalizedFloat et l'utilisation de la fonction tex2DLayered
texture< pixel,	cudaTextureType2D >			TexS_MaskTer;
texture< float2,cudaTextureType2DLayered >	TexL_Proj;
texture< float,	cudaTextureType2DLayered >	TexL_Images;

ImageCuda<pixel>			mask;
ImageLayeredCuda<float>		LayeredImages;
ImageLayeredCuda<float2>	LayeredProjection;

CuData3D<float>	volumeCost;	// volume des couts   
CuData3D<float>	volumeCach;	// volume des calculs intermédiaires
CuData3D<float>	volumeNIOk;	// nombre d'image correct pour une vignette

static __constant__ paramGPU cH;
paramGPU h;

//------------------------------------------------------------------------------------------
extern "C" void SetMask(pixel* dataMask, uint2 dimMask)
{
	mask.InitImage(dimMask,dataMask);

	cudaBindTextureToArray(TexS_MaskTer,mask.GetCudaArray());
}

extern "C" void allocMemory(void)
{

	volumeCost.Realloc(h.rDiTer,h.ZInter);
	volumeCach.Realloc(h.dimCach, h.nbImages * h.ZInter);
	volumeNIOk.Realloc(h.rDiTer,h.ZInter);

	// Texture des projections
	TexL_Proj.addressMode[0]	= cudaAddressModeClamp;
	TexL_Proj.addressMode[1]	= cudaAddressModeClamp;	
	TexL_Proj.filterMode		= cudaFilterModeLinear; //cudaFilterModePoint cudaFilterModeLinear
	TexL_Proj.normalized		= true;

}

extern "C" paramGPU updateSizeBlock( uint2 ter0, uint2 ter1, uint Zinter )
{

	uint oldSizeTer = h.sizeTer;

	h.ZInter	= Zinter;
	h.ptMask0	= make_int2(ter0);
	h.ptMask1	= make_int2(ter1);
	h.pUTer0.x	= (int)ter0.x - (int)h.rVig.x;
	h.pUTer0.y	= (int)ter0.y - (int)h.rVig.y;
	h.pUTer1.x	= (int)ter1.x + (int)h.rVig.x;
	h.pUTer1.y	= (int)ter1.y + (int)h.rVig.y;
	h.rDiTer	= make_uint2(ter1.x - ter0.x, ter1.y - ter0.y);
	h.dimTer	= make_uint2(h.pUTer1.x - h.pUTer0.x, h.pUTer1.y - h.pUTer0.y);
	h.dimSTer	= iDivUp(h.dimTer,h.sampTer);	// Dimension du bloque terrain sous echantilloné
	h.sizeTer	= size(h.dimTer);				// Taille du bloque terrain
	h.sizeSTer  = size(h.dimSTer);				// Taille du bloque terrain sous echantilloné
	h.rSiTer	= size(h.rDiTer);
	h.dimCach	= h.rDiTer * h.dimVig;
	h.sizeCach	= size(h.dimCach);
	h.restTer	= h.dimSTer * h.sampTer - h.dimTer;

	checkCudaErrors(cudaMemcpyToSymbol(cH, &h, sizeof(paramGPU)));

	if (oldSizeTer < h.sizeTer)
		allocMemory();

	return h;
}

static void correlOptionsGPU( uint2 ter0, uint2 ter1, uint2 dV,uint2 dRV, uint2 dI, float mAhEpsilon, uint samplingZ, int uvINTDef, uint nLayer, uint interZ )
{

	float uvDef;
	memset(&uvDef,uvINTDef,sizeof(float));

	h.nbImages	= nLayer;
	h.dimVig	= dV;							// Dimension de la vignette
	h.dimImg	= dI;							// Dimension des images
	h.rVig		= dRV;							// Rayon de la vignette
	h.sizeVig	= size(dV);						// Taille de la vignette en pixel 
	h.sampTer	= samplingZ;					// Pas echantillonage du terrain
	h.DefaultVal= uvDef;						// UV Terrain incorrect
	h.IntDefault	= uvINTDef;
	h.badVig	= -4.0f;
	h.mAhEpsilon= mAhEpsilon;

	updateSizeBlock( ter0, ter1, interZ );
}

extern "C" void imagesToLayers(float *fdataImg1D, uint2 dimImage, int nbLayer)
{

	LayeredImages.SetDimension(dimImage,nbLayer);
	LayeredImages.AllocMemory();
	LayeredImages.copyHostToDevice(fdataImg1D);

	// Lié à la texture
	TexL_Images.addressMode[0]	= cudaAddressModeWrap;
    TexL_Images.addressMode[1]	= cudaAddressModeWrap;
    TexL_Images.filterMode		= cudaFilterModePoint; //cudaFilterModeLinear cudaFilterModePoint
    TexL_Images.normalized		= true;
	
	checkCudaErrors( cudaBindTextureToArray(TexL_Images,LayeredImages.GetCudaArray()) );

};

extern "C" void  allocMemoryTabProj(uint2 dimTer, int nbLayer)
{
	LayeredProjection.DeallocMemory();
	LayeredProjection.SetDimension(dimTer,nbLayer);
	LayeredProjection.AllocMemory();
}

extern "C" void  CopyProjToLayers(float2 *h_TabProj)
{
	LayeredProjection.copyHostToDevice(h_TabProj);
};

__global__ void correlationKernel( float *dev_NbImgOk, float* cachVig, uint2 nbActThrd )
{
	__shared__ float cacheImg[ BLOCKDIM ][ BLOCKDIM ];

	// Coordonnées du terrain global avec bordure // __umul24!!!! A voir
	const uint2 ptHTer = make_uint2(blockIdx) * nbActThrd + make_uint2(threadIdx);
	
	// Si le processus est hors du terrain, nous sortons du kernel
	if (oSE(ptHTer,cH.dimTer)) return;

#if (SAMPLETERR == 1)
	const float2 ptProj = tex2DLayeredPt(TexL_Proj,ptHTer,cH.dimSTer,blockIdx.z);
#else
	const float2 ptProj = tex2DLayeredPt(TexL_Proj,ptHTer,cH.dimSTer,cH.sampTer,blockIdx.z);
#endif
	
	if (oI(ptProj,0))
	{
		cacheImg[threadIdx.y][threadIdx.x]  = cH.badVig;
		return;
	}
 	else
		//cacheImg[threadIdx.y][threadIdx.x] = tex2DFastBicubic<float,float>(TexL_Images, ptProj.x, ptProj.y, cH.dimImg,(int)(blockIdx.z % cH.nbImages));
		//cacheImg[threadIdx.y][threadIdx.x] = tex2DLayeredPt( TexL_Images, ptProj, cH.dimImg, (int)(blockIdx.z % cH.nbImages));
		cacheImg[threadIdx.y][threadIdx.x] = tex2DLayered( TexL_Images, (((int)ptProj.x )+ 0.5f) / (float)cH.dimImg.x, (((int)(ptProj.y) )+ 0.5f) / (float)cH.dimImg.y,(int)(blockIdx.z % cH.nbImages));
 
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
	
	aSV	 /=	cH.sizeVig;
	aSVV /=	cH.sizeVig;
	aSVV -=	(aSV * aSV);
	
	if ( aSVV <= cH.mAhEpsilon) return;

	aSVV =	sqrt(aSVV);

	const uint pitchCache = blockIdx.z * cH.sizeCach + ptTer.x * cH.dimVig.x;
	const uint pitchCachY = ptTer.y * cH.dimVig.y ;
	#pragma unroll
	for ( pt.y = c0.y ; pt.y <= c1.y; pt.y++)
	{
		const int _py	= (pitchCachY + (pt.y - c0.y))* cH.dimCach.x;
		#pragma unroll
		for ( pt.x = c0.x ; pt.x <= c1.x; pt.x++)					
			cachVig[ pitchCache + _py  + (pt.x - c0.x)] = (cacheImg[pt.y][pt.x] -aSV)/aSVV;
	}	

	const int ZPitch = (blockIdx.z / cH.nbImages) * cH.rSiTer;

	atomicAdd( &dev_NbImgOk[ZPitch + to1D(ptTer,cH.rDiTer)], 1.0f);
};

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

	atomicAdd(&(resu[thTer.y][thTer.x]),aSVV[t.y][t.x] - ((aSV[t.y][t.x] * aSV[t.y][t.x])/ nbIm[thTer.y][thTer.x])); 

	if ( !mThrd ) return;
	__syncthreads();

	// Normalisation pour le ramener a un equivalent de 1-Correl 
	const float cost = resu[thTer.y][thTer.x]/ (( nbIm[thTer.y][thTer.x] -1.0f) * ((float)cH.sizeVig));

	dTCost[iTer] = 1.0f - max (-1.0, min(1.0f,1.0f - cost));
}

extern "C" paramGPU Init_Correlation_GPU(  uint2 ter0, uint2 ter1, int nbLayer , uint2 dRVig , uint2 dimImg, float mAhEpsilon, uint samplingZ, int uvINTDef , uint interZ)
{
	correlOptionsGPU( ter0, ter1, dRVig * 2 + 1,dRVig, dimImg,mAhEpsilon, samplingZ, uvINTDef,nbLayer, interZ);
	allocMemory();

	return h;
}

extern "C" void basic_Correlation_GPU( float* h_TabCost,  int nbLayer, uint interZ ){

 	volumeCost.SetDimension(h.rDiTer,interZ);
	volumeCach.SetDimension(h.dimCach,nbLayer * interZ);
	volumeNIOk.SetDimension(h.rDiTer,interZ);

	//----------------------------------------------------------------------------
	
	volumeCost.Memset(h.IntDefault);
	volumeCach.Memset(h.IntDefault);
	volumeNIOk.Memset(0);
 	checkCudaErrors( cudaBindTextureToArray(TexL_Proj,LayeredProjection.GetCudaArray()) );

	// --------------- calcul de dimension du kernel de correlation --------------

	dim3	threads( BLOCKDIM, BLOCKDIM, 1);
	uint2	thd2D		= make_uint2(threads);
	uint2	actiThsCo	= thd2D - 2 * h.dimVig;
	uint2	block2D		= iDivUp(h.dimTer,actiThsCo);
	dim3	blocks(block2D.x , block2D.y, nbLayer * interZ);


	//-------------	calcul de dimension du kernel de multi-correlation ------------

	uint2	actiThs		= SBLOCKDIM - make_uint2( SBLOCKDIM % h.dimVig.x, SBLOCKDIM % h.dimVig.y);
	dim3	threads_mC(SBLOCKDIM, SBLOCKDIM, nbLayer);
	uint2	block2D_mC	= iDivUp(h.dimCach,actiThs);
	dim3	blocks_mC(block2D_mC.x,block2D_mC.y,interZ);

	//-----------------------  KERNEL  Correlation  -------------------------------
	
	correlationKernel<<<blocks, threads>>>( volumeNIOk.pData(), volumeCach.pData(), actiThsCo);
	getLastCudaError("Basic Correlation kernel failed");
	
	//-------------------  KERNEL  Multi Correlation  ------------------------------

    multiCorrelationKernel<<<blocks_mC, threads_mC>>>( volumeCost.pData(), volumeCach.pData(), volumeNIOk.pData(), actiThs);
    getLastCudaError("Multi-Correlation kernel failed");

	//----------------------------------------------------------------------------

	checkCudaErrors( cudaUnbindTexture(TexL_Proj) );
	volumeCost.CopyDevicetoHost(h_TabCost);
	
	//----------------------------------------------------------------------------
	//checkCudaErrors( cudaMemcpy( h_TabCost, dev_NbImgOk, costMemSize, cudaMemcpyDeviceToHost) );
	//checkCudaErrors( cudaMemcpy( host_Cache, dev_Cache,	  cac_MemSize, cudaMemcpyDeviceToHost) );
	//GpGpuTools::OutputArray(h_TabCost,h.rDiTer,11.0f,h.DefaultVal);
	//----------------------------------------------------------------------------
}

extern "C" void freeGpuMemory()
{
	checkCudaErrors( cudaUnbindTexture(TexL_Images) );	
	checkCudaErrors( cudaUnbindTexture(TexS_MaskTer) );	

	volumeCach.Dealloc();
	volumeCost.Dealloc();
	volumeNIOk.Dealloc();

	mask.DeallocMemory();
	LayeredImages.DeallocMemory();
	LayeredProjection.DeallocMemory();
}
