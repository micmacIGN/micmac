#include "GpGpu/cudaAppliMicMac.cuh"
#include "GpGpu/helper_math_extented.cuh"


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

/* Non utilisé
texture<float, cudaTextureType2D, cudaReadModeNormalizedFloat> refTex_Image;
texture<bool, cudaTextureType2D, cudaReadModeNormalizedFloat> refTex_Cache;
texture<float2, cudaTextureType2D, cudaReadModeNormalizedFloat> refTex_Project;
cudaArray* dev_Img;				// Tableau des valeurs de l'image
cudaArray* dev_CubeProjImg;		// Declaration du cube de projection pour le device
cudaArray* dev_ArrayProjImg;	// Declaration du tableau de projection pour le device
*/

//------------------------------------------------------------------------------------------
// ATTENTION : erreur de compilation avec l'option cudaReadModeNormalizedFloat
// et l'utilisation de la fonction tex2DLayered
texture< bool,	cudaTextureType2D >			TexMaskTer;
texture< float2,cudaTextureType2DLayered >	TexLay_Proj;
texture< float,	cudaTextureType2DLayered >	refTex_ImagesLayered;
cudaArray* dev_ImgLd;		//
cudaArray* dev_ProjLr;		//
cudaArray* dev_MaskTer;		//

//------------------------------------------------------------------------------------------
//float*	host_SimpCor;
//float*	dev_SimpCor;
float*	host_Cache;
float*	dev_Cost;
float*	dev_Cache;
float*	dev_NbImgOk;

paramGPU h;
static __constant__ paramGPU cH;

extern "C" void allocMemory(void)
{
	//if (dev_SimpCor != NULL) checkCudaErrors( cudaFree(dev_SimpCor));
	//int sCorMemSize = h.sizeTer * sizeof(float);
	//host_SimpCor	= (float*)	malloc(sCorMemSize);
	//checkCudaErrors( cudaMalloc((void **) &dev_SimpCor	, sCorMemSize) );
	//host_Cache		= (float*)	malloc(cac_MemSize);

	if (dev_NbImgOk	!= NULL) checkCudaErrors( cudaFree(dev_NbImgOk));
	if (dev_Cache	!= NULL) checkCudaErrors( cudaFree(dev_Cache));
	if (dev_Cost	!= NULL) checkCudaErrors( cudaFree(dev_Cost));

	int costMemSize = h.rSiTer	* sizeof(float);
	int nBI_MemSize = h.rSiTer	* sizeof(float);
	int cac_MemSize = h.sizeCach* sizeof(float)* h.nLayer;
	
	// Allocation mémoire
	checkCudaErrors( cudaMalloc((void **) &dev_Cache	, cac_MemSize ) );
	checkCudaErrors( cudaMalloc((void **) &dev_NbImgOk	, nBI_MemSize ) );
	checkCudaErrors( cudaMalloc((void **) &dev_Cost		, costMemSize ) );

	// Texture des projections
	TexLay_Proj.addressMode[0]	= cudaAddressModeClamp;
	TexLay_Proj.addressMode[1]	= cudaAddressModeClamp;	
	TexLay_Proj.filterMode		= cudaFilterModePoint; //cudaFilterModePoint cudaFilterModeLinear
	TexLay_Proj.normalized		= true;

}

extern "C" paramGPU updateSizeBlock( uint2 ter0, uint2 ter1 )
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
	
	checkCudaErrors(cudaMemcpyToSymbol(cH, &h, sizeof(paramGPU)));

	if (oldSizeTer < h.sizeTer)
		allocMemory();

	return h;
}

static void correlOptionsGPU( uint2 ter0, uint2 ter1, uint2 dV,uint2 dRV, uint2 dI, float mAhEpsilon, uint samplingZ, int uvINTDef, uint nLayer )
{

	float uvDef;
	memset(&uvDef,uvINTDef,sizeof(float));

	h.nLayer	= nLayer;
	h.dimVig	= dV;							// Dimension de la vignette
	h.dimImg	= dI;							// Dimension des images
	h.rVig		= dRV;							// Rayon de la vignette
	h.sizeVig	= size(dV);						// Taille de la vignette en pixel 
	h.sampTer	= samplingZ;					// Pas echantillonage du terrain
	h.UVDefValue= uvDef;						// UV Terrain incorrect
	h.UVIntDef	= uvINTDef;
	h.badVig	= -4.0f;
	h.mAhEpsilon= mAhEpsilon;

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

	p.dstArray	= dev_ImgLd;				// Pointeur du tableau de destination
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

extern "C" void  allocMemoryTabProj(uint2 dimTer, int nbLayer)
{

	// Définition du format des canaux d'images
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float2>();

	// Taille du tableau des calques 
	cudaExtent siz_PL = make_cudaExtent( dimTer.x, dimTer.y, nbLayer);

	// Allocation memoire GPU du tableau des calques de projections
	if (dev_ProjLr != NULL) cudaFreeArray(dev_ProjLr);

	checkCudaErrors( cudaMalloc3DArray(&dev_ProjLr,&channelDesc,siz_PL,cudaArrayLayered ));

}

extern "C" void  CopyProjToLayers(float *h_TabProj, uint2 dimTer, int nbLayer)
{
	cudaExtent siz_PL = make_cudaExtent( dimTer.x, dimTer.y, nbLayer);

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
	const float2 cf		= make_float2(ssize) * make_float2(coord) / make_float2(size) ;
	const int2	 a		= make_int2(cf);
	const float2 uva	= (make_float2(a) + 0.5f) / (make_float2(ssize));
	const float2 uvb	= (make_float2(a+1) + 0.5f) / (make_float2(ssize));
	float2 ra, rb, Iaa;

	ra	= tex2DLayered( TexLay_Proj, uva.x, uva.y, L);
	rb	= tex2DLayered( TexLay_Proj, uvb.x, uva.y, L);
	if (ra.x < 0.0f || ra.y < 0.0f || rb.x < 0.0f || rb.y < 0.0f)
		return make_float2(cH.badVig);

	Iaa	= ((float)(a.x + 1.0f) - cf.x) * ra + (cf.x - (float)(a.x)) * rb;
	ra	= tex2DLayered( TexLay_Proj, uva.x, uvb.y, L);
	rb	= tex2DLayered( TexLay_Proj, uvb.x, uvb.y, L);

	if (ra.x < 0.0f || ra.y < 0.0f || rb.x < 0.0f || rb.y < 0.0f)
		return make_float2(cH.badVig);

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
	if (oSE(ptHTer,cH.dimTer)) return;

	//float2 PtTProj = tex2DLayered(TexLay_Proj, ((float)ghTer.x / (float)cDimTer.x * (float)cSDimTer.x + 0.5f) /(float)cSDimTer.x, ((float)ghTer.y/ (float)cDimTer.y * (float)cSDimTer.y + 0.5f) /(float)cSDimTer.y ,blockIdx.z) ;
	//const float2 PtTProj = simpleProjection( cDimTer, cSDimTer/*, cDimImg*/, ptHTer, blockIdx.z);
	const float2 PtTProj = tex2DLayered(TexLay_Proj, ((float)ptHTer.x  + 0.5f) /(float)cH.dimTer.x, ((float)ptHTer.y + 0.5f) /(float)cH.dimTer.y ,blockIdx.z) ;

	if (oEq(PtTProj, cH.UVDefValue))
	{
		cacheImg[threadIdx.y][threadIdx.x]  = cH.badVig;
		return;
	}
 	else
		// !!! ATTENTION Modification pour simplification du debug !!!!
		//cacheImg[threadIdx.y][threadIdx.x] = tex2DLayered( refTex_ImagesLayered, (PtTProj.x + 0.5f) / (float)cDimImg.x, (PtTProj.y + 0.5f) / (float)cDimImg.y,blockIdx.z);
		cacheImg[threadIdx.y][threadIdx.x] = tex2DLayered( refTex_ImagesLayered, (((int)PtTProj.x )+ 0.5f) / (float)cH.dimImg.x, (((int)(PtTProj.y) )+ 0.5f) / (float)cH.dimImg.y,blockIdx.z);

	__syncthreads();

	const int2 ptTer = make_int2(ptHTer) - make_int2(cH.rVig);

	// Nous traitons uniquement les points du terrain du bloque ou Si le processus est hors du terrain global, nous sortons du kernel
	if ( oSE(threadIdx, nbActThrd + cH.rVig) || oI(threadIdx , cH.rVig) || oSE( ptTer, cH.rDiTer) || oI(ptTer, 0))
		return;
	
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

	#pragma unroll
	for ( pt.y = c0.y ; pt.y <= c1.y; pt.y++)
	{
		const int _cy	= ptTer.y * cH.dimVig.y + (pt.y - c0.y);
		#pragma unroll
		for ( pt.x = c0.x ; pt.x <= c1.x; pt.x++)					
			cachVig[(blockIdx.z * cH.sizeCach) + _cy * cH.dimCach.x + ptTer.x * cH.dimVig.x + (pt.x - c0.x)] = (cacheImg[pt.y][pt.x] -aSV)/aSVV;
	}	

	atomicAdd( &dev_NbImgOk[to1D(ptTer,cH.rDiTer)], 1.0f);
};

///////////////////////////////////////////////////////////////////////////////////
//																				///
// Calcul "rapide"  de la multi-correlation en utilisant la formule de Huygens	///
//																				///
///////////////////////////////////////////////////////////////////////////////////

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

 	if ( oSE( t, nbActThr))	return; // si le thread est inactif, il sort

	// Coordonnées 2D du cache vignette
	const uint2 ptCach = make_uint2(blockIdx) * nbActThr  + t;
	
	// Si le thread est en dehors du cache
	if ( oSE(ptCach, cH.dimCach))	return;
	
	const uint2	ptTer	= ptCach / cH.dimVig;					// Coordonnées 2D du terrain 
	const uint	iTer	= to1D(ptTer, cH.rDiTer);				// Coordonnées 1D dans le terrain
	const bool	mThrd	= t.x % cH.dimVig.x == 0 &&  t.y % cH.dimVig.y == 0 && threadIdx.z == 0;
	const float aNbImOk = dev_NbImgOk[iTer];					// Nombre vignettes correctes

	if (aNbImOk < 2) return;
	
	const uint sizLayer = threadIdx.z * cH.sizeCach;			// Taille du cache vignette pour une image
	const uint iCach	= sizLayer + to1D( ptCach, cH.dimCach );	// Coordonnées 1D du cache vignette
	const uint2 cc		= ptTer * cH.dimVig;						// coordonnées 2D 1er pixel de la vignette
	const int iCC		= sizLayer + to1D( cc, cH.dimCach );	// coordonnées 1D 1er pixel de la vignette
	
	const float val = (cacheVign[iCC] != cH.UVDefValue) ? cacheVign[iCach] : 0.0f; // sortir si bad vignette

	atomicAdd( &(aSV[t.y][t.x]), val);

	atomicAdd(&(aSVV[t.y][t.x]), val * val);
	__syncthreads();

	if ( threadIdx.z != 0) return;

	const uint2 thTer = t / cH.dimVig;	// Coordonnées 2D du terrain dans le repere des threads
	
	atomicAdd(&(resu[thTer.y][thTer.x]),aSVV[t.y][t.x] - ((aSV[t.y][t.x] * aSV[t.y][t.x])/ aNbImOk)); 

	if ( !mThrd ) return;
	__syncthreads();

	// Normalisation pour le ramener a un equivalent de 1-Correl 
	const float cost = resu[thTer.y][thTer.x]/ (( aNbImOk -1.0f) * ((float)cH.sizeVig));

	dTCost[iTer] = 1.0f - max (-1.0, min(1.0f,1.0f - cost));
}

extern "C" paramGPU Init_Correlation_GPU(  uint2 ter0, uint2 ter1, int nbLayer , uint2 dRVig , uint2 dimImg, float mAhEpsilon, uint samplingZ, int uvINTDef )
{
	dev_NbImgOk		= NULL;
	dev_Cache		= NULL;
	dev_Cost		= NULL;
	dev_ProjLr		= NULL;

	correlOptionsGPU( ter0, ter1, dRVig * 2 + 1,dRVig, dimImg,mAhEpsilon, samplingZ, uvINTDef,nbLayer);
	allocMemory();

	return h;
}

extern "C" void basic_Correlation_GPU( float* h_TabCost,  int nbLayer ){

	//////////////////////////////////////////////////////////////////////////
	//int sCorMemSize = h.sizeTer  * sizeof(float);
	//checkCudaErrors( cudaMemset( dev_SimpCor,	0, sCorMemSize ));
	////////////////////////////////////////////////////////////////////////// 
	
	int nBI_MemSize = h.rSiTer	 * sizeof(float);
	int cac_MemSize = h.sizeCach * sizeof(float) * nbLayer;
	int costMemSize = h.rSiTer	 * sizeof(float);

	//////////////////////////////////////////////////////////////////////////
	 
	checkCudaErrors( cudaMemset( dev_Cost,	h.UVIntDef, costMemSize ));
	checkCudaErrors( cudaMemset( dev_Cache,	h.UVIntDef, cac_MemSize ));
	checkCudaErrors( cudaMemset( dev_NbImgOk,0, nBI_MemSize ));
	checkCudaErrors( cudaBindTextureToArray(TexLay_Proj,dev_ProjLr) );

	//////////////////////////////////////////////////////////////////////////

	dim3 threads( BLOCKDIM, BLOCKDIM, 1);
	uint2 actiThsCo = make_uint2(threads.x - 2 *((int)(h.dimVig.x)), threads.y - 2 * ((int)(h.dimVig.y)));
	dim3 blocks(iDivUp((int)(h.dimTer.x),actiThsCo.x) , iDivUp((int)(h.dimTer.y), actiThsCo.y), nbLayer);
	
	uint2 actiThs = make_uint2(SBLOCKDIM - SBLOCKDIM % ((int)h.dimVig.x), SBLOCKDIM - SBLOCKDIM % ((int)h.dimVig.y));
	dim3 threads_mC(SBLOCKDIM, SBLOCKDIM, nbLayer);
	dim3 blocks_mC(iDivUp((int)(h.dimCach.x), actiThs.x) , iDivUp((int)(h.dimCach.y), actiThs.y));

	////////////////////--  KERNEL  Correlation  --//////////////////////////
	
	correlationKernel<<<blocks, threads>>>( dev_NbImgOk, dev_Cache, actiThsCo);
	getLastCudaError("Basic Correlation kernel failed");
	
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

	//GpGpuTools::OutputArray(h_TabCost,h.rDiTer,100.0f,h.UVDefValue);

	//////////////////////////////////////////////////////////////////////////

}

extern "C" void freeGpuMemory()
{
	//checkCudaErrors( cudaUnbindTexture(refTex_Image) );
	//checkCudaErrors( cudaFreeArray(dev_Img) );
	//checkCudaErrors( cudaFreeArray(dev_CubeProjImg) );
	//checkCudaErrors( cudaFreeArray(dev_ArrayProjImg) );

	//if(dev_SimpCor	!= NULL) checkCudaErrors( cudaFree( dev_SimpCor));
	
	checkCudaErrors( cudaUnbindTexture(refTex_ImagesLayered) );	

	if(dev_ImgLd	!= NULL) checkCudaErrors( cudaFreeArray( dev_ImgLd) );
	if(dev_ProjLr	!= NULL) checkCudaErrors( cudaFreeArray( dev_ProjLr) );
	if(dev_NbImgOk	!= NULL) checkCudaErrors( cudaFree( dev_NbImgOk));
	if(dev_Cache	!= NULL) checkCudaErrors( cudaFree( dev_Cache));
	if(dev_Cost		!= NULL) checkCudaErrors( cudaFree( dev_Cost));

	dev_NbImgOk	= NULL;
	dev_Cache	= NULL;
	dev_ImgLd	= NULL;
	dev_Cost	= NULL;

	// DEBUG
	//dev_SimpCor = NULL;
	//free(host_SimpCor); 
	//free(host_Cache);
}

extern "C" void  projToDevice(cudaArray_t *dev_ArrayProjImg,texture<float2, cudaTextureType2D, cudaReadModeNormalizedFloat> refTex_Project, float* aProj,  int sXImg, int sYImg)
{
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float2>();

	// Allocation mémoire du tableau cuda
	checkCudaErrors( cudaMallocArray(dev_ArrayProjImg,&channelDesc,sYImg,sXImg) );

	// Copie des données du Host dans le tableau Cuda
	checkCudaErrors( cudaMemcpy2DToArray(*dev_ArrayProjImg,0,0,aProj, sYImg*sizeof(float2),sYImg*sizeof(float2), sXImg, cudaMemcpyHostToDevice) );

	// Lier la texture au tableau Cuda
	checkCudaErrors( cudaBindTextureToArray(refTex_Project,*dev_ArrayProjImg) );

}

extern "C" void cubeProjToDevice(cudaArray_t *dev_CubeProjImg,float* cubeProjPIm, cudaExtent dimCube)
{

	// Format des canaux 
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindFloat);
			
	// Taille du cube
	cudaExtent sizeCube = dimCube;
			
	// Allocation memoire GPU du cube de projection
	checkCudaErrors( cudaMalloc3DArray(dev_CubeProjImg,&channelDesc,sizeCube) );

	// Déclaration des parametres de copie 3D
	cudaMemcpy3DParms p = { 0 };
			
	p.dstArray	= *dev_CubeProjImg;			// Pointeur du tableau de destination
	p.srcPtr	= make_cudaPitchedPtr(cubeProjPIm, dimCube.width * 2 * sizeof(float), dimCube.width, dimCube.height);
	p.extent	= dimCube;					// Taille du cube
	p.kind		= cudaMemcpyHostToDevice;	// Type de copie

	// Copie du cube de projection du Host vers le Device
	checkCudaErrors( cudaMemcpy3D(&p) );
		
}

extern "C" void  imageToDevice(cudaArray_t *dev_Img, texture<float, cudaTextureType2D, cudaReadModeNormalizedFloat> refTex_Image, float** aDataIm,  int sXImg, int sYImg)
{
	float *dataImg1D	= new float[sXImg*sYImg];
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();

	// TACHE : changer la structuration des donnees pour le stockage des images 
	// Tableau 2D  --->> tableau linéaire
	for (int i = 0; i < sXImg ; i++)
		for (int j = 0; j < sYImg ; j++)
			dataImg1D[i*sYImg+j] = aDataIm[j][i];

	// Allocation mémoire du tableau cuda
	checkCudaErrors( cudaMallocArray(dev_Img,&channelDesc,sYImg,sXImg) );

	// Copie des données du Host dans le tableau Cuda
	checkCudaErrors( cudaMemcpy2DToArray(*dev_Img,0,0,dataImg1D, sYImg*sizeof(float),sYImg*sizeof(float), sXImg, cudaMemcpyHostToDevice) );

	// Lier la texture au tableau Cuda
	checkCudaErrors( cudaBindTextureToArray(refTex_Image,*dev_Img) );

	delete dataImg1D;

}
