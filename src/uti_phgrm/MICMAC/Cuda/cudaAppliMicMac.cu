#include "gpu/cudaAppliMicMac.cuh"

#include <iostream>
#include <string>
using namespace std;

#ifdef _WIN32
  #include <windows.h>
  #include <Lmcons.h>
#endif

#ifdef _DEBUG
	#define   BLOCKDIM	16
	#define   SBLOCKDIM 10
#else
	#define   BLOCKDIM	32
	#define   SBLOCKDIM 16
#endif

//------------------------------------------------------------------------------------------
// Non utilisé
texture<float, cudaTextureType2D, cudaReadModeNormalizedFloat> refTex_Image;
texture<bool, cudaTextureType2D, cudaReadModeNormalizedFloat> refTex_Cache;
texture<float2, cudaTextureType2D, cudaReadModeNormalizedFloat> refTex_Project;
cudaArray* dev_Img;				// Tableau des valeurs de l'image
cudaArray* dev_CubeProjImg;		// Declaration du cube de projection pour le device
cudaArray* dev_ArrayProjImg;	// Declaration du tableau de projection pour le device
//------------------------------------------------------------------------------------------

//------------------------------------------------------------------------------------------
// ATTENTION : erreur de compilation avec l'option cudaReadModeNormalizedFloat
// et l'utilisation de la fonction tex2DLayered
texture<float2,	cudaTextureType2DLayered > TexLay_Proj;
texture<float,	cudaTextureType2DLayered > refTex_ImagesLayered;
cudaArray* dev_ImgLd;	//
cudaArray* dev_ProjLr;		//

//------------------------------------------------------------------------------------------
float*	host_SimpCor;
float*	host_Cache;
float*	dev_SimpCor;
float*	dev_Cost;
float*	dev_Cache;
float*	dev_NbImgOk;

paramGPU h;

extern "C" void allocMemory(void)
{

	if (dev_NbImgOk	!= NULL) checkCudaErrors( cudaFree(dev_NbImgOk));
	if (dev_SimpCor != NULL) checkCudaErrors( cudaFree(dev_SimpCor));
	if (dev_Cache	!= NULL) checkCudaErrors( cudaFree(dev_Cache));
	if (dev_Cost	!= NULL) checkCudaErrors( cudaFree(dev_Cost));


	int sCorMemSize = h.sizeTer * sizeof(float);
	int costMemSize = h.rSiTer	* sizeof(float);
	int nBI_MemSize = h.rSiTer	* sizeof(float);
	int cac_MemSize = h.sizeCach* sizeof(float)* h.nLayer;
	
	// Allocation mémoire
	host_SimpCor	= (float*)	malloc(sCorMemSize);
	host_Cache		= (float*)	malloc(cac_MemSize);
	
	checkCudaErrors( cudaMalloc((void **) &dev_SimpCor	, sCorMemSize) );	
	checkCudaErrors( cudaMalloc((void **) &dev_Cache	, cac_MemSize ) );
	checkCudaErrors( cudaMalloc((void **) &dev_NbImgOk	, nBI_MemSize ) );
	checkCudaErrors( cudaMalloc((void **) &dev_Cost		, costMemSize ) );

	// Texture des projections
	TexLay_Proj.addressMode[0]	= cudaAddressModeClamp;
	TexLay_Proj.addressMode[1]	= cudaAddressModeClamp;	
	TexLay_Proj.filterMode		= cudaFilterModePoint; //cudaFilterModePoint cudaFilterModeLinear
	TexLay_Proj.normalized		= true;

}

extern "C" paramGPU updateSizeBlock(  uint2 ter0, uint2 ter1 )
{

	uint oldSizeTer = h.sizeTer;

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
	
	checkCudaErrors(cudaMemcpyToSymbol(cRDiTer, &h.rDiTer, sizeof(uint2)));
	checkCudaErrors(cudaMemcpyToSymbol(cSDimTer, &h.dimSTer, sizeof(uint2)));
	checkCudaErrors(cudaMemcpyToSymbol(cDimTer, &h.dimTer, sizeof(uint2)));
	checkCudaErrors(cudaMemcpyToSymbol(cSizeTer, &h.sizeTer, sizeof(uint)));
	checkCudaErrors(cudaMemcpyToSymbol(cSizeSTer, &h.sizeSTer, sizeof(uint)));

	checkCudaErrors(cudaMemcpyToSymbol(cDimCach, &h.dimCach, sizeof(uint2)));
	checkCudaErrors(cudaMemcpyToSymbol(cSizeCach, &h.sizeCach, sizeof(uint)));

	if (oldSizeTer < h.sizeTer)
		allocMemory();

	return h;
}

static void correlOptionsGPU( uint2 ter0, uint2 ter1, uint2 dV,uint2 dRV, uint2 dI, float mAhEpsilon, uint samplingZ, float uvDef, uint nLayer )
{

	h.nLayer	= nLayer;
	h.dimVig	= dV;							// Dimension de la vignette
	h.dimImg	= dI;							// Dimension des images
	h.rVig		= dRV;							// Rayon de la vignette
	h.sizeVig	= size(dV);						// Taille de la vignette en pixel 
	h.sampTer	= samplingZ;					// Pas echantillonage du terrain
	h.UVDefValue= uvDef;						// UV Terrain incorrect
	float badVi	= -4.0f;

	checkCudaErrors(cudaMemcpyToSymbol(cRVig, &dRV, sizeof(uint2)));
	checkCudaErrors(cudaMemcpyToSymbol(cDimVig, &h.dimVig, sizeof(uint2)));
	checkCudaErrors(cudaMemcpyToSymbol(cDimImg, &dI, sizeof(uint2)));
	checkCudaErrors(cudaMemcpyToSymbol(cMAhEpsilon, &mAhEpsilon, sizeof(float)));
	checkCudaErrors(cudaMemcpyToSymbol(cSizeVig, &h.sizeVig, sizeof(uint)));
	checkCudaErrors(cudaMemcpyToSymbol(cSampTer, &h.sampTer, sizeof(uint)));
	checkCudaErrors(cudaMemcpyToSymbol(cUVDefValue, &h.UVDefValue, sizeof(float)));
	checkCudaErrors(cudaMemcpyToSymbol(cBadVignet, &badVi, sizeof(float)));
	
	updateSizeBlock( ter0, ter1 );
}

extern "C" void imagesToLayers(float *fdataImg1D, uint2 dimImage, int nbLayer)
{
	cudaExtent sizeImgsLay = make_cudaExtent( dimImage.x, dimImage.y, nbLayer );

	// Définition du format des canaux d'images
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

	// Allocation memoire GPU du tableau des calques d'images
	checkCudaErrors( cudaMalloc3DArray(&dev_ImgLd,&channelDesc,sizeImgsLay,cudaArrayLayered) );

	// Déclaration des parametres de copie 3D
	cudaMemcpy3DParms	p	= { 0 };
	cudaPitchedPtr		pit = make_cudaPitchedPtr(fdataImg1D, sizeImgsLay.width * sizeof(float), sizeImgsLay.width, sizeImgsLay.height);

	p.dstArray	= dev_ImgLd;		// Pointeur du tableau de destination
	p.srcPtr	= pit;						// Pitch
	p.extent	= sizeImgsLay;				// Taille du cube
	p.kind		= cudaMemcpyHostToDevice;	// Type de copie

	// Copie des images du Host vers le Device
	checkCudaErrors( cudaMemcpy3D(&p) );

	// Lié à la texture
	refTex_ImagesLayered.addressMode[0]	= cudaAddressModeWrap;
    refTex_ImagesLayered.addressMode[1]	= cudaAddressModeWrap;
    refTex_ImagesLayered.filterMode		= cudaFilterModePoint; //cudaFilterModeLinear cudaFilterModePoint
    refTex_ImagesLayered.normalized		= true;
	checkCudaErrors( cudaBindTextureToArray(refTex_ImagesLayered,dev_ImgLd) );

};

extern "C" void  projectionsToLayers(float *h_TabProj, uint2 dimTer, int nbLayer)
{
	// Définition du format des canaux d'images
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float2>();

	// Taille du tableau des calques 
	cudaExtent siz_PL = make_cudaExtent( dimTer.x, dimTer.y, nbLayer);

	// Allocation memoire GPU du tableau des calques de projections
	
	//checkCudaErrors( cudaMalloc3DArray(&dev_ProjLr,&channelDesc,siz_PL,cudaArrayLayered ));
	cudaError_t eC =  cudaMalloc3DArray(&dev_ProjLr,&channelDesc,siz_PL,cudaArrayLayered );
	if (eC != cudaSuccess)
	{
		std::cout << "Dimension du tableau des Images : " << h.dimImg.x << ","<< h.dimImg.x << "," << nbLayer  << "\n";
		std::cout << "Dimension du tableau des projections : " << dimTer.x << ","<< dimTer.x << "," << nbLayer  << "\n";
		
	}

	// Déclaration des parametres de copie 3D
	cudaMemcpy3DParms p = { 0 };

	p.dstArray	= dev_ProjLr;			// Pointeur du tableau de destination
	p.srcPtr	= make_cudaPitchedPtr(h_TabProj, siz_PL.width * sizeof(float2), siz_PL.width, siz_PL.height);
	p.extent	= siz_PL;
	p.kind		= cudaMemcpyHostToDevice;	// Type de copie

	// Copie des projections du Host vers le Device
	checkCudaErrors( cudaMemcpy3D(&p) );

};

__device__  inline float2 simpleProjection( uint2 size, uint2 ssize/*, uint2 sizeImg*/ ,uint2 coord, int L)
{
	const float2 cf = make_float2(ssize) * make_float2(coord) / make_float2(size) ;
	const int2	 a	= make_int2(cf);
	const float2 uva = (make_float2(a) + 0.5f) / (make_float2(ssize));
	const float2 uvb = (make_float2(a+1) + 0.5f) / (make_float2(ssize));
	float2 ra, rb, Iaa;

	ra	= tex2DLayered( TexLay_Proj, uva.x, uva.y, L);
	rb	= tex2DLayered( TexLay_Proj, uvb.x, uva.y, L);
	if (ra.x < 0.0f || ra.y < 0.0f || rb.x < 0.0f || rb.y < 0.0f)
		return make_float2(cBadVignet);

	Iaa	= ((float)(a.x + 1.0f) - cf.x) * ra + (cf.x - (float)(a.x)) * rb;
	ra	= tex2DLayered( TexLay_Proj, uva.x, uvb.y, L);
	rb	= tex2DLayered( TexLay_Proj, uvb.x, uvb.y, L);

	if (ra.x < 0.0f || ra.y < 0.0f || rb.x < 0.0f || rb.y < 0.0f)
		return make_float2(cBadVignet);

	ra	= ((float)(a.x+ 1.0f) - cf.x) * ra + (cf.x - (float)(a.x)) * rb;
	ra = ((float)(a.y+ 1.0f) - cf.y) * Iaa + (cf.y - (float)(a.y)) * ra;
	/*ra = (ra + 0.5f) / (make_float2(sizeImg));*/

	return ra;
}

__global__ void correlationKernel( float *dev_NbImgOk, float* cachVig/*, float *siCor*/, uint2 nbActThrd ) //__global__ void correlationKernel( int *dev_NbImgOk, float* cachVig)
{
	__shared__ float cacheImg[ BLOCKDIM ][ BLOCKDIM ];

	// Coordonnées du terrain global avec bordure
	const uint2 ptHTer = make_uint2(blockIdx) * nbActThrd + make_uint2(threadIdx);
	
	// Si le processus est hors du terrain, nous sortons du kernel
 	if ( ptHTer.x >= cDimTer.x || ptHTer.y >= cDimTer.y) return;

	//float2 PtTProj = tex2DLayered(TexLay_Proj, ((float)ghTer.x / (float)cDimTer.x * (float)cSDimTer.x + 0.5f) /(float)cSDimTer.x, ((float)ghTer.y/ (float)cDimTer.y * (float)cSDimTer.y + 0.5f) /(float)cSDimTer.y ,blockIdx.z) ;
	//const float2 PtTProj = simpleProjection( cDimTer, cSDimTer/*, cDimImg*/, ptHTer, blockIdx.z);
	const float2 PtTProj = tex2DLayered(TexLay_Proj, ((float)ptHTer.x  + 0.5f) /(float)cDimTer.x, ((float)ptHTer.y + 0.5f) /(float)cDimTer.y ,blockIdx.z) ;
	
	const int2 ptTer	= make_int2(ptHTer) - make_int2(cRVig);
	const int2 caVig	= ptTer * make_int2(cDimVig);
	const int  iC		= blockIdx.z * cSizeCach + caVig.y * cDimCach.x + caVig.x;

	if ( PtTProj.x == cUVDefValue || PtTProj.y == cUVDefValue )
	{
		cacheImg[threadIdx.y][threadIdx.x]  = cBadVignet;
		if (!(caVig.x >= cDimCach.x || caVig.y >= cDimCach.y || caVig.x <0 || caVig.y < 0 ))
			cachVig[iC]		= cBadVignet;
		//if (blockIdx.z	== iDI) siCor[iTer2] = 2*cBadVignet; 
		return;
	}
 	else
		// !!! ATTENTION Modification pour simplification du debug !!!!
		//cacheImg[threadIdx.y][threadIdx.x] = tex2DLayered( refTex_ImagesLayered, (PtTProj.x + 0.5f) / (float)cDimImg.x, (PtTProj.y + 0.5f) / (float)cDimImg.y,blockIdx.z);
		cacheImg[threadIdx.y][threadIdx.x] = tex2DLayered( refTex_ImagesLayered, (((int)PtTProj.x )+ 0.5f) / (float)cDimImg.x, (((int)(PtTProj.y) )+ 0.5f) / (float)cDimImg.y,blockIdx.z);

	__syncthreads();

	// Nous traitons uniquement les points du terrain du bloque ou Si le processus est hors du terrain global, nous sortons du kernel
	if ((threadIdx.x >= (nbActThrd.x + cRVig.x))||(threadIdx.y >= (nbActThrd.y + cRVig.y) || (threadIdx.x < cRVig.x) || (threadIdx.y < cRVig.y)) || ( ptTer.x >= cRDiTer.x) || (ptTer.y >= cRDiTer.y) || (ptTer.x < 0) || (ptTer.y < 0) )
		return;
	
	const short2 c0	= make_short2(threadIdx.x - ((short)(cRVig.x)),threadIdx.y - ((short)(cRVig.y)));
	const short2 c1	= make_short2(threadIdx.x + ((short)(cRVig.x)),threadIdx.y + ((short)(cRVig.y)));

	// Si le parcours de la vignette est hors du terrain, nous sortons!!! Sinon crash GPU!!!!
// 	if ( (c1.x >= blockDim.x) || (c1.y >= blockDim.y) || (c0.x < 0) || (c0.y < 0) )	//if (blockIdx.z == iDI) siCor[iTer] = 3*cBadVignet; // ## z ##
// 	{
// 		cachVig[iC] = cBadVignet;
// 		return;
// 	}

	// Intialisation des valeurs de calcul 
	float aSV = 0.0f, aSVV	= 0.0f;
	
	#pragma unroll // ATTENTION PRAGMA FAIT PETER LA MEMOIRE des registres!!!
	for (short y = c0.y ; y <= c1.y; y++)
	{
		#pragma unroll
		for (short x = c0.x ; x <= c1.x; x++)
		{	
			const float val = cacheImg[y][x];	// Valeur de l'image

			if (val ==  cBadVignet)
			{
				cachVig[iC] = cBadVignet; 
				return;
			}
			aSV  += val;		// Somme des valeurs de l'image cte 
			aSVV += (val*val);	// Somme des carrés des vals image cte
		}
	}

	aSV	 /=	cSizeVig;
	aSVV /=	cSizeVig;
	aSVV -=	(aSV * aSV);
	
	if ( aSVV <= cMAhEpsilon) //
	{
		cachVig[iC] = cBadVignet;
		return;
	}

	aSVV =	sqrt(aSVV);

	#pragma unroll
	for (short y = c0.y ; y <= c1.y; y++)
	{
		const int _cy	= ptTer.y * cDimVig.y + (y - c0.y);
		#pragma unroll
		for (short x = c0.x ; x <= c1.x; x++)					
// 			if (cacheImg[y][x] == cBadVignet)
// 			{
// 				cachVig[iC] = cBadVignet;
// 				return;
// 			}
// 			const int _cx	= ter.x * cDimVig.x + (x - c0.x);
// 			const int _iC   = (blockIdx.z * cSizeCach) + _cy * cDimCach.x + _cx;
			cachVig[(blockIdx.z * cSizeCach) + _cy * cDimCach.x + ptTer.x * cDimVig.x + (x - c0.x)] = (cacheImg[y][x] -aSV)/aSVV;
		
	}

//  	if (blockIdx.z	== iDI)
// 		siCor[iTer] = (1.0f + cachVig[iC]) / 2.0f; //== 0.0f ? -9 * cBadVignet : cachVig[iC] ; // ## ¤ ##

	// Coordonnées 1D du terrain
	//const int iTer	= (cRDiTer.x * ter.y) + ter.x; // ne sert pas 
	// Nombre d'images correctes
	//atomicAdd( &dev_NbImgOk[iTer], 1.0f);
	atomicAdd( &dev_NbImgOk[(cRDiTer.x * ptTer.y) + ptTer.x], 1.0f);
};

// ---------------------------------------------------------------------------
// Calcul "rapide"  de la multi-correlation en utilisant la formule de Huygens
// ---------------------------------------------------------------------------
__global__ void multiCorrelationKernel(float *dTCost, float* cacheVign, float * dev_NbImgOk, uint2 nbActThr)
{
	__shared__ float aSV [ SBLOCKDIM ][ SBLOCKDIM ];
	__shared__ float aSVV[ SBLOCKDIM ][ SBLOCKDIM ];
	__shared__ float resu[ SBLOCKDIM/2 ][ SBLOCKDIM/2 ];

	// coordonnées des threads
	const uint2 t = make_uint2(threadIdx);

	if ( threadIdx.z == 0)
	{
		aSV [t.y][t.x]		= 0.0f;
		aSVV[t.y][t.x]		= 0.0f;
		resu[t.y/2][t.x/2]	= 0.0f;
	}
	
	__syncthreads();

	// si le thread est inactif, il sort
 	if ( t.x >=  nbActThr.x || t.y >=  nbActThr.y )
 		return;

	// Coordonnées 2D du cache vignette
	const uint2 cCach = make_uint2(blockIdx) * nbActThr  + t;
	
	// Si le thread est en dehors du cache
	if ( cCach.x >= cDimCach.x || cCach.y >= cDimCach.y )
		return;
	
	const uint pitCachLayer = threadIdx.z * cSizeCach;

	// Coordonnées 1D du cache vignette
	const uint iCach	= pitCachLayer + cCach.y * cDimCach.x + cCach.x ;
	
	// Coordonnées 2D du terrain 
	const uint2 coorTer		= cCach / cDimVig;
	
	// coordonnées central de la vignette
	const uint2 cc = coorTer * cDimVig;
	const int iCC = pitCachLayer + cc.y * cDimCach.x + cc.x;

	// Coordonnées 1D dans le terrain
	const int iTer		= coorTer.y * cRDiTer.x  + coorTer.x;
	const bool mainThread	= t.x % cDimVig.x == 0 &&  t.y% cDimVig.y == 0 && threadIdx.z == 0;
	const float aNbImOk = dev_NbImgOk[iTer];

	if (aNbImOk < 2)
	{
		if (mainThread) dTCost[iTer] = -1000.0f;
		return;
	}

	float val = (cacheVign[iCC] != cBadVignet) ? cacheVign[iCach] : 0.0f; // sortir si bad vignette

	// Coordonnées 2D du terrain dans le repere des threads
	const int2 coorTTer = make_int2(t.x / ((int)(cDimVig.x )), t.y / ((int)(cDimVig.x )));

	atomicAdd( &(aSV[t.y][t.x]), val);
	__syncthreads();

	const float VV = val * val;
	atomicAdd(&(aSVV[t.y][t.x]), VV);
	__syncthreads();

	if ( threadIdx.z != 0) return;

	atomicAdd(&(resu[coorTTer.y][coorTTer.x]),aSVV[t.y][t.x] - ((aSV[t.y][t.x] * aSV[t.y][t.x])/ aNbImOk)); 
	__syncthreads();

	if ( !mainThread ) return;

	// Normalisation pour le ramener a un equivalent de 1-Correl 
	const float cost = resu[coorTTer.y][coorTTer.x]/ (( aNbImOk -1.0f) * ((float)cSizeVig));

	dTCost[iTer] = 1.0f - max (-1.0, min(1.0f,1.0f - cost));
}

extern "C" paramGPU Init_Correlation_GPU(  uint2 ter0, uint2 ter1, int nbLayer , uint2 dRVig , uint2 dimImg, float mAhEpsilon, uint samplingZ, float uvDef )
{
	dev_NbImgOk		= NULL;
	dev_SimpCor		= NULL;
	dev_Cache		= NULL;
	dev_Cost		= NULL;

	correlOptionsGPU( ter0, ter1, dRVig * 2 + 1,dRVig, dimImg,mAhEpsilon, samplingZ, uvDef,nbLayer);
	allocMemory();

	return h;
}

extern "C" void basic_Correlation_GPU( float* h_TabCost,  int nbLayer ){
	
	//////////////////////////////////////////////////////////////////////////
	 
	//int sCorMemSize = h.sizeTer  * sizeof(float);
	int nBI_MemSize = h.rSiTer	 * sizeof(float);
	int cac_MemSize = h.sizeCach * sizeof(float) * nbLayer;
	int costMemSize = h.rSiTer	 * sizeof(float);

	//////////////////////////////////////////////////////////////////////////

	//checkCudaErrors( cudaMemset( dev_SimpCor,	0, sCorMemSize ));
	checkCudaErrors( cudaMemset( dev_Cost,		0, costMemSize ));
	checkCudaErrors( cudaMemset( dev_Cache,		0, cac_MemSize ));
	checkCudaErrors( cudaMemset( dev_NbImgOk,	0, nBI_MemSize ));
	checkCudaErrors( cudaBindTextureToArray(TexLay_Proj,dev_ProjLr) );

	//////////////////////////////////////////////////////////////////////////

	dim3 threads( BLOCKDIM, BLOCKDIM, 1);
	uint2 actiThsCo = make_uint2(threads.x - 2 *((int)(h.dimVig.x)), threads.y - 2 * ((int)(h.dimVig.y)));
	dim3 blocks(iDivUp((int)(h.dimTer.x),actiThsCo.x) , iDivUp((int)(h.dimTer.y), actiThsCo.y), nbLayer);
	
	uint2 actiThs = make_uint2(SBLOCKDIM - SBLOCKDIM % ((int)h.dimVig.x), SBLOCKDIM - SBLOCKDIM % ((int)h.dimVig.y));
	dim3 threads_mC(SBLOCKDIM, SBLOCKDIM, nbLayer);
	dim3 blocks_mC(iDivUp((int)(h.dimCach.x), actiThs.x) , iDivUp((int)(h.dimCach.y), actiThs.y));

	////////////////////--  KERNEL  Correlation  --//////////////////////////
	
	correlationKernel<<<blocks, threads>>>( dev_NbImgOk, dev_Cache /*, dev_SimpCor*/, actiThsCo);
	getLastCudaError("Basic Correlation kernel failed");
	//cudaDeviceSynchronize();
	
	//////////////////--  KERNEL  Multi Correlation  --///////////////////////

   	multiCorrelationKernel<<<blocks_mC, threads_mC>>>( dev_Cost, dev_Cache, dev_NbImgOk, actiThs);
   	getLastCudaError("Multi-Correlation kernel failed");

	//////////////////////////////////////////////////////////////////////////

	checkCudaErrors( cudaUnbindTexture(TexLay_Proj) );
	checkCudaErrors( cudaMemcpy( h_TabCost, dev_Cost, costMemSize, cudaMemcpyDeviceToHost) );
	
	//cudaDeviceSynchronize();
	//checkCudaErrors( cudaMemcpy( h_TabCost, dev_NbImgOk, costMemSize, cudaMemcpyDeviceToHost) );
	//checkCudaErrors( cudaMemcpy( host_SimpCor, dev_SimpCor, sCorMemSize, cudaMemcpyDeviceToHost) );
	//checkCudaErrors( cudaMemcpy( host_Cache, dev_Cache,	  cac_MemSize, cudaMemcpyDeviceToHost) );

	//////////////////////////////////////////////////////////////////////////
/*
	if(0)
	{
		//////////////////////////////////////////////////////////////////////////
		if (0)
		{
			//for (uint idii = 0 ; idii < 4 ; idii++)
			uint idii = 1;
			{
				std::cout << "CACHE IMAGE : " << idii << " --------------------------------\n";
				for (uint j = 0 ; j < h.dimCach.y / h.dimVig.y ; j++)
				{
					for (uint i = 0; i < h.dimCach.x / h.dimVig.x ; i++)
					{
						float off	= 10.0f;
						int ii = i * h.dimVig.x + h.rVig.x;
						int jj = j * h.dimVig.y + h.rVig.y;

						int id		= (idii * h.sizeCach + jj * h.dimCach.x + ii );
						float out	= host_Cache[id];
						out			= floor(out*off)/off;

						int bad = -4;

						std::string S2 = "   ";
						std::string ES = "";
						std::string S1 = "  ";

						std::string valS;
						stringstream sValS (stringstream::in | stringstream::out);
						sValS << abs(out);
						long sizeV = sValS.str().length();
						if (sizeV == 3) ES = ES + " ";
						else if (sizeV == 2) ES = ES + "  ";
						else if (sizeV == 1) ES = ES + "   ";

						if (out == bad)
							std::cout << S1 << "!" + S2;
						else if (out == -1000.0f)
							std::cout << S1 << "." << S2;
						else if (out == 2*bad)
							std::cout << S1 << "s" << S2;
						else if (out == 3*bad)
							std::cout << S1 << "z" << S2;
						else if (out == 4*bad)
							std::cout << S1 << "s" << S2;
						else if (out == 5*bad)
							std::cout << S1 << "v" << S2;
						else if (out == 6*bad)
							std::cout << S1 << "e" << S2;
						else if (out == 7*bad)
							std::cout << S1 << "c" << S2;
						else if (out == 8*bad)
							std::cout << S1 << "?" << S2;
						else if (out == 9*bad)
							std::cout << S1 << "¤" << S2;
						else if (out == 0.0f)
							std::cout << S1 << "0" << S2;
						else if ( out < 0.0f)
							std::cout <<  out << ES;				
						else 
							std::cout << S1 << out << ES;

					}
					std::cout << "\n";	
				}
				std::cout << "------------------------------------------\n";
			}
		}

		if (0)
		{
			uint idImage = 0;

			uint2 dimCach = h.dimTer * h.dimVig;

			float* imageCache	= new float[h.sizeTer * h.sizeVig];
			for (uint j = 0; j < dimCach.y; j++)
				for (uint i = 0; i < dimCach.x ; i++)
				{
					int id = (j * dimCach.x + i );
					imageCache[id] = host_Cache[idImage * size(dimCach) + id]/7.0f + 3.5f;
				}

				TCHAR name [ UNLEN + 1 ];
				DWORD size = UNLEN + 1;
				GetUserName( (TCHAR*)name, &size );

				std::string suname = name;

				std::string fileImaCache = "C:\\Users\\" + suname + "\\Pictures\\imageCache.pgm";

				std::cout << suname << "\n";
				// save PGM
				if (sdkSavePGM<float>(fileImaCache.c_str(), imageCache, dimCach.x,dimCach.y))
					std::cout <<"success save image" << "\n";
				else
					std::cout <<"Failed save image" << "\n";

				delete[] imageCache;
		
			float* image	= new float[h.rSiTer];
			for (uint j = 0; j < h.rDiTer.y ; j++)
				for (uint i = 0; i < h.rDiTer.x ; i++)
				{
					int id = (j * h.rDiTer.x + i );
					if (host_SimpCor[id] == -8)
					{
						image[id] = 0;
					} 
					else
					{
						image[id] = host_SimpCor[id]/500.f;	
						//image[id] = host_SimpCor[id]/2.0f;	
					}
					
				}

			TCHAR name [ UNLEN + 1 ];
			DWORD size = UNLEN + 1;
			GetUserName( (TCHAR*)name, &size );

			std::string suname = name;
			std::string fileImage = "C:\\Users\\" + suname + "\\Pictures\\image.pgm";

			// save PGM
			if (sdkSavePGM<float>(fileImage.c_str(), image, h.rDiTer.x,h.rDiTer.y))
				std::cout <<"success save image" << "\n";
			else
				std::cout <<"Failed save image" << "\n";

			delete[] image;
		}
		

		if(0)
		{

			for (uint j = 0 ; j < h.dimTer.y; j+= h.sampTer)
			{
				for (uint i = 0; i < h.dimTer.x ; i+= h.sampTer)
				{
					float off = 10000.0f;
					int id = (j * h.dimTer.x + i );
					float out = host_SimpCor[id];
					std::cout << floor(out*off)/off << " ";
				}
				std::cout << "\n";	
			}
			std::cout << "------------------------------------------\n";
		}
		if (0)
		{
			for (uint j = 0 ; j < h.rDiTer.y; j++)
			{
				for (uint i = 0; i < h.rDiTer.x ; i++)
				{
					float off = 10.0f;
					int id = (j * h.rDiTer.x + i );
					float out = h_TabCost[id];
					if (out < 10)
						std::cout << out << "  ";
					else
						std::cout << out << " ";
				}
				std::cout << "\n";	
			}
			std::cout << "------------------------------------------\n";

		}

		if (0)
		{

			for (uint j = 0 ; j < h.rDiTer.y; j+= h.sampTer)
			{
				for (uint i = 0; i < h.rDiTer.x ; i+= h.sampTer)
				{
					float off = 1.0f;

					int id = (j * h.rDiTer.x + i );
					float out = h_TabCost[id];
					if (out == -1000)
						std::cout << ".  ";
					else if (out >= 10 )
						std::cout << floor(out*off)/off  << " ";
					else
						std::cout << floor(out*off)/off  << "  ";
				}

				std::cout << "\n";	
			}

			std::cout << "------------------------------------------\n";
		}


		//if (0)
		

		{
			int bad = -4;
			for (uint j = 0 ; j < h.rDiTer.y; j+= h.sampTer)
			{
				for (uint i = 0; i < h.rDiTer.x ; i+= h.sampTer)
				{

					float off = 100.0f;
					int id = (j * h.rDiTer.x + i );

					std::string S2 = "    ";
					std::string ES = "";
					std::string S1 = " ";

					//float out = host_SimpCor[id];// 500.0f;
					float out = h_TabCost[id];
					out = floor(out*off)/off ;

					std::string valS;
					stringstream sValS (stringstream::in | stringstream::out);
					sValS << abs(out);
					long sizeV = sValS.str().length();

					if (sizeV == 5) ES = ES + "";
					else if (sizeV == 4) ES = ES + " ";
					else if (sizeV == 3) ES = ES + "  ";
					else if (sizeV == 2) ES = ES + "   ";
					else if (sizeV == 1) ES = ES + "    ";

					if (out == bad)
						std::cout << S1 << "!" + S2;
					else if (out == -1000.0f)
						std::cout << S1 << "." << S2;
					else if (out == 2*bad)
						std::cout << S1 << "s" << S2;
					else if (out == 3*bad)
						std::cout << S1 << "z" << S2;
					else if (out == 4*bad)
						std::cout << S1 << "s" << S2;
					else if (out == 5*bad)
						std::cout << S1 << "v" << S2;
					else if (out == 6*bad)
						std::cout << S1 << "e" << S2;
					else if (out == 7*bad)
						std::cout << S1 << "c" << S2;
					else if (out == 8*bad)
						std::cout << S1 << "?" << S2;
					else if (out == 9*bad)
						std::cout << S1 << "¤" << S2;
					else if (out == 0.0f)
						std::cout << S1 << "0" << S2;
					else if ( out < 0.0f)
						std::cout << out << ES;				
					else 
						std::cout << S1 << out << ES;

				//////////////////////////////////////////////////////////////////////////
// 					else if ( out < 0.0f && out > -1.0f)
// 					{
// 						std::cout << " " << out << ES;
// 						//std::cout << "|\\|";
// 					}
// 					else if ( out > 0.0f && out < 1.0f)
// 						std::cout << S1 << out << ES;
// 						//std::cout << " *" << S1;
// 					else
// 						std::cout << S1 << "H" << S2;

				}
				std::cout << "\n";	
			}
			std::cout << "------------------------------------------\n";
		}	
	}
	*/
}

extern "C" void freeGpuMemory()
{
	checkCudaErrors( cudaUnbindTexture(refTex_Image) );
	checkCudaErrors( cudaUnbindTexture(refTex_ImagesLayered) );
	checkCudaErrors( cudaFreeArray(dev_Img) );	
	checkCudaErrors( cudaFreeArray(dev_CubeProjImg) );
	checkCudaErrors( cudaFreeArray(dev_ArrayProjImg) );

	if(dev_ImgLd	!= NULL) checkCudaErrors( cudaFreeArray( dev_ImgLd) );
	if(dev_ProjLr	!= NULL) checkCudaErrors( cudaFreeArray( dev_ProjLr) );
	if(dev_NbImgOk	!= NULL) checkCudaErrors( cudaFree( dev_NbImgOk));
	if(dev_SimpCor	!= NULL) checkCudaErrors( cudaFree( dev_SimpCor));
	if(dev_Cache	!= NULL) checkCudaErrors( cudaFree( dev_Cache));
	if(dev_Cost		!= NULL) checkCudaErrors( cudaFree( dev_Cost));

	dev_NbImgOk	= NULL;
	dev_SimpCor = NULL;
	dev_Cache	= NULL;
	dev_ImgLd	= NULL;
	dev_Cost	= NULL;

	// DEBUG 

	free(host_SimpCor); 
	free(host_Cache);
}

extern "C" void  FreeLayers()
{
	checkCudaErrors( cudaFreeArray(dev_ImgLd));

};

extern "C" void  projToDevice(float* aProj,  int sXImg, int sYImg)
{
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float2>();

	// Allocation mémoire du tableau cuda
	checkCudaErrors( cudaMallocArray(&dev_ArrayProjImg,&channelDesc,sYImg,sXImg) );

	// Copie des données du Host dans le tableau Cuda
	checkCudaErrors( cudaMemcpy2DToArray(dev_ArrayProjImg,0,0,aProj, sYImg*sizeof(float2),sYImg*sizeof(float2), sXImg, cudaMemcpyHostToDevice) );

	// Lier la texture au tableau Cuda
	checkCudaErrors( cudaBindTextureToArray(refTex_Project,dev_ArrayProjImg) );

}

extern "C" void cubeProjToDevice(float* cubeProjPIm, cudaExtent dimCube)
{

	// Format des canaux 
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindFloat);
			
	// Taille du cube
	cudaExtent sizeCube = dimCube;
			
	// Allocation memoire GPU du cube de projection
	checkCudaErrors( cudaMalloc3DArray(&dev_CubeProjImg,&channelDesc,sizeCube) );

	// Déclaration des parametres de copie 3D
	cudaMemcpy3DParms p = { 0 };
			
	p.dstArray	= dev_CubeProjImg;			// Pointeur du tableau de destination
	p.srcPtr	= make_cudaPitchedPtr(cubeProjPIm, dimCube.width * 2 * sizeof(float), dimCube.width, dimCube.height);
	p.extent	= dimCube;					// Taille du cube
	p.kind		= cudaMemcpyHostToDevice;	// Type de copie

	// Copie du cube de projection du Host vers le Device
	checkCudaErrors( cudaMemcpy3D(&p) );
		
}

extern "C" void  imageToDevice(float** aDataIm,  int sXImg, int sYImg)
{
	float *dataImg1D	= new float[sXImg*sYImg];
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();

	// TACHE : changer la structuration des donnees pour le stockage des images 
	// Tableau 2D  --->> tableau linéaire
	for (int i = 0; i < sXImg ; i++)
		for (int j = 0; j < sYImg ; j++)
			dataImg1D[i*sYImg+j] = aDataIm[j][i];

	// Allocation mémoire du tableau cuda
	checkCudaErrors( cudaMallocArray(&dev_Img,&channelDesc,sYImg,sXImg) );

	// Copie des données du Host dans le tableau Cuda
	checkCudaErrors( cudaMemcpy2DToArray(dev_Img,0,0,dataImg1D, sYImg*sizeof(float),sYImg*sizeof(float), sXImg, cudaMemcpyHostToDevice) );

	// Lier la texture au tableau Cuda
	checkCudaErrors( cudaBindTextureToArray(refTex_Image,dev_Img) );

	delete dataImg1D;

}
