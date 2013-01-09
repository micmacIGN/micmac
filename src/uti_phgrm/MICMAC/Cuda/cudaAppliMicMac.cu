#include "gpu/cudaAppliMicMac.cuh"

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
	TexLay_Proj.filterMode		= cudaFilterModePoint; //cudaFilterModePoint 
	TexLay_Proj.normalized		= true;

}

extern "C" paramGPU updateSizeBlock( int x0, int x1, int y0, int y1 )
{

	uint oldSizeTer = h.sizeTer;

	h.pUTer0.x	= x0 - (int)h.rVig.x;
	h.pUTer0.y	= y0 - (int)h.rVig.y;
	h.pUTer1.x	= x1 + (int)h.rVig.x;
	h.pUTer1.y	= y1 + (int)h.rVig.y;
	
	h.rDiTer	= make_uint2(x1 - x0, y1 - y0);
	h.dimTer	= make_uint2(h.pUTer1.x - h.pUTer0.x, h.pUTer1.y - h.pUTer0.y);
	h.dimSTer	= iDivUp(h.dimTer,h.sampTer);	// Dimension du bloque terrain sous echantilloné
	h.sizeTer	= size(h.dimTer);				// Taille du bloque terrain
	h.sizeSTer  = size(h.dimSTer);				// Taille du bloque terrain sous echantilloné
	h.rSiTer	= size(h.rDiTer);

	h.dimCach	= h.rDiTer * h.dimVig;
	h.sizeCach	= size(h.dimCach);
	
	checkCudaErrors(cudaMemcpyToSymbol(cRDiTer, &h.rDiTer, sizeof(uint2)));
	checkCudaErrors(cudaMemcpyToSymbol(cSDimTer, &h.dimSTer, sizeof(int2)));
	checkCudaErrors(cudaMemcpyToSymbol(cDimTer, &h.dimTer, sizeof(int2)));
	checkCudaErrors(cudaMemcpyToSymbol(cSizeTer, &h.sizeTer, sizeof(uint)));
	checkCudaErrors(cudaMemcpyToSymbol(cSizeSTer, &h.sizeSTer, sizeof(uint)));

	checkCudaErrors(cudaMemcpyToSymbol(cDimCach, &h.dimCach, sizeof(uint2)));
	checkCudaErrors(cudaMemcpyToSymbol(cSizeCach, &h.sizeCach, sizeof(uint)));

	if (oldSizeTer < h.sizeTer)
		allocMemory();

	return h;
}

static void correlOptionsGPU(int x0, int x1, int y0, int y1, uint2 dV,uint2 dRV, uint2 dI, float mAhEpsilon, uint samplingZ, float uvDef, uint nLayer )
{

	h.nLayer	= nLayer;
	h.dimVig	= dV;							// Dimension de la vignette
	h.dimImg	= dI;							// Dimension des images
	h.rVig		= dRV;							// Rayon de la vignette
	h.sizeVig	= size(dV);						// Taille de la vignette en pixel 
	h.sampTer	= samplingZ;					// Pas echantillonage du terrain
	h.UVDefValue= uvDef;						// UV Terrain incorrect
	int badVi	= -4;

	checkCudaErrors(cudaMemcpyToSymbol(cRVig, &dRV, sizeof(uint2)));
	checkCudaErrors(cudaMemcpyToSymbol(cDimVig, &h.dimVig, sizeof(uint2)));
	checkCudaErrors(cudaMemcpyToSymbol(cDimImg, &dI, sizeof(uint2)));
	checkCudaErrors(cudaMemcpyToSymbol(cMAhEpsilon, &mAhEpsilon, sizeof(float)));
	checkCudaErrors(cudaMemcpyToSymbol(cSizeVig, &h.sizeVig, sizeof(uint)));
	checkCudaErrors(cudaMemcpyToSymbol(cSampTer, &h.sampTer, sizeof(uint)));
	checkCudaErrors(cudaMemcpyToSymbol(cUVDefValue, &h.UVDefValue, sizeof(float)));
	checkCudaErrors(cudaMemcpyToSymbol(cBadVignet, &badVi, sizeof(int)));
	
	updateSizeBlock( x0, x1, y0, y1 );
}

extern "C" void imagesToLayers(float *fdataImg1D, uint2 dimTer, int nbLayer)
{
	cudaExtent sizeImgsLay = make_cudaExtent( dimTer.x, dimTer.y, nbLayer );

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
    refTex_ImagesLayered.filterMode		= cudaFilterModeLinear; //cudaFilterModeLinear cudaFilterModePoint
    refTex_ImagesLayered.normalized		= true;
	checkCudaErrors( cudaBindTextureToArray(refTex_ImagesLayered,dev_ImgLd) );

};

extern "C" void  projectionsToLayers(float *h_TabProj, uint2 dimTer, int nbLayer)
{
	// Définition du format des canaux d'images
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float2>();

	// Taille du tableau des calques 
	cudaExtent siz_PL = make_cudaExtent( dimTer.x, dimTer.y, nbLayer);

	// Allocation memoire GPU du tableau des calques d'images
	checkCudaErrors( cudaMalloc3DArray(&dev_ProjLr,&channelDesc,siz_PL,cudaArrayLayered ));

	// Déclaration des parametres de copie 3D
	cudaMemcpy3DParms p = { 0 };

	p.dstArray	= dev_ProjLr;			// Pointeur du tableau de destination
	p.srcPtr	= make_cudaPitchedPtr(h_TabProj, siz_PL.width * sizeof(float2), siz_PL.width, siz_PL.height);
	p.extent	= siz_PL;
	p.kind		= cudaMemcpyHostToDevice;	// Type de copie

	// Copie des projections du Host vers le Device
	checkCudaErrors( cudaMemcpy3D(&p) );

};

__device__  inline float2 simpleProjection( uint2 size, uint2 ssize, uint2 sizeImg ,int2 coord, int L)
{
	const float bad = -1.0f;
	const float2 cf = make_float2(ssize) * make_float2(coord) / make_float2(size) ;
	const int2	 a	= make_int2(cf);
	const float2 uva = (make_float2(a) + 0.5f) / (make_float2(ssize));
	const float2 uvb = (make_float2(a+1) + 0.5f) / (make_float2(ssize));
	float2 ra, rb, Iaa;

	ra	= tex2DLayered( TexLay_Proj, uva.x, uva.y, L);
	rb	= tex2DLayered( TexLay_Proj, uvb.x, uva.y, L);
	if (ra.x < 0.0f || ra.y < 0.0f || rb.x < 0.0f || rb.y < 0.0f)
		return make_float2(bad);

	Iaa	= ((float)(a.x + 1.0f) - cf.x) * ra + (cf.x - (float)(a.x)) * rb;
	ra	= tex2DLayered( TexLay_Proj, uva.x, uvb.y, L);
	rb	= tex2DLayered( TexLay_Proj, uvb.x, uvb.y, L);

	if (ra.x < 0.0f || ra.y < 0.0f || rb.x < 0.0f || rb.y < 0.0f)
		return make_float2(bad);

	ra	= ((float)(a.x+ 1.0f) - cf.x) * ra + (cf.x - (float)(a.x)) * rb;
	
	ra = ((float)(a.y+ 1.0f) - cf.y) * Iaa + (cf.y - (float)(a.y)) * ra;
	ra = (ra + 0.5f) / (make_float2(sizeImg));

	return ra;
}

__global__ void correlationKernel( float *dev_NbImgOk, float* cachVig, float *siCor ) //__global__ void correlationKernel( int *dev_NbImgOk, float* cachVig)
{
	__shared__ float cacheImg[ BLOCKDIM ][ BLOCKDIM ];
	const int outMask	= -1;
	const int iDI		= 0;

	// Se placer dans l'espace terrain

	const int DT_x	= (blockDim.x - 2 * ((int)cRVig.x));
	const int Te_x	= blockIdx.x * DT_x + threadIdx.x - (int)cRVig.x;
	 if (Te_x < 0) return;

	const int2	coorTer = make_int2(blockIdx.x * (blockDim.x - 2 * ((int)cRVig.x)) + threadIdx.x - ((int)cRVig.x ),   blockIdx.y * (blockDim.y - 2 * ((int)cRVig.y)) + threadIdx.y - ((int)cRVig.y)  );
	const int	iTer	= coorTer.y * cDimTer.x + coorTer.x;
	
	// Si le processus est hors du terrain, nous sortons du kernel
	if ( coorTer.x >= cDimTer.x || coorTer.y >= cDimTer.y || coorTer.x < 0 || coorTer.y < 0) 
		return;

	//const float2 PtTProj = simpleProjection( cDimTer, cSDimTer, cDimImg, coorTer, blockIdx.z);
	const float2 PtTProj = tex2DLayered(TexLay_Proj, ((float)coorTer.x / cDimTer.x * cSDimTer.x + 0.5f) /(float)cSDimTer.x, ((float)coorTer.y/ cDimTer.y * cSDimTer.y + 0.5f) /(float)cSDimTer.y ,blockIdx.z) ;
	//const float2 PtTProj = tex2DLayered(TexLay_Proj, (float)coorTer.x / cDimTer.x , (float)coorTer.y/ cDimTer.y,blockIdx.z) ;

	const int cx	= cRVig.x + coorTer.x * cDimVig.x;
	const int cy	= cRVig.y + coorTer.y * cDimVig.y;
	const uint iC   = (blockIdx.z * cSizeCach) + cy * cDimCach.x + cx;

	if (cx >= cDimCach.x || cy >= cDimCach.y || cx <0 || cy < 0 )
		return;

	if ( PtTProj.x == outMask ||  PtTProj.y == outMask )
	{
		cacheImg[threadIdx.y][threadIdx.x]  = cBadVignet;		
		cachVig[iC]		= cBadVignet; // ## - ##
		if (blockIdx.z	== iDI) siCor[iTer] = 2*cBadVignet; // ## . ##
		return;
	}
	else
		cacheImg[threadIdx.y][threadIdx.x] = tex2DLayered( refTex_ImagesLayered, PtTProj.x, PtTProj.y,blockIdx.z);

	__syncthreads();

	// Intialisation des valeurs de calcul 
	float		aSV	= 0.0f;
	float	   aSVV	= 0.0f;
	const int2 c0	= make_int2(threadIdx.x - (int)cRVig.x,threadIdx.y - (int)cRVig.y);
	const int2 c1	= make_int2(threadIdx.x + cRVig.x,threadIdx.y + cRVig.y);

	if ( c1.x >= blockDim.x || c1.y >= blockDim.y || c0.x < 0 || c0.y < 0 )
	{
		if (blockIdx.z == iDI) siCor[iTer] = 3*cBadVignet; // ## z ##
		return;
	}

	//#pragma unroll
	for (int y = c0.y ; y <= c1.y; y++)
	{
		//#pragma unroll
		for (int x = c0.x ; x <= c1.x; x++)
		{	
			const float val = cacheImg[y][x];	// Valeur de l'image

			if (val ==  cBadVignet)
			{
				cachVig[iC] = cBadVignet; 
				if (blockIdx.z == iDI) siCor[iTer] = 5*cBadVignet; // ## v ##
				return;
			}
			aSV  += val;		// Somme des valeurs de l'image cte 
			aSVV += (val*val);	// Somme des carrés des vals image cte
		}
	}
	
	aSV	 /=	cSizeVig;
	aSVV /=	cSizeVig;
	aSVV -=	(aSV * aSV);
	
	if ( aSVV <= cMAhEpsilon)
	{
		cachVig[iC] = cBadVignet;
		if (blockIdx.z == iDI) siCor[iTer] = 6*cBadVignet; // ## e ##
		return;
	}

	aSVV =	sqrt(aSVV);

	#pragma unroll
	for (int y = c0.y ; y <= c1.y; y++)
	{
		
		const uint _cy	= coorTer.y * cDimVig.y + (y - c0.y);

		#pragma unroll
		for (int x = c0.x ; x <= c1.x; x++)
		{			

			if (cacheImg[y][x]  ==  cBadVignet) // A priori Inutile
			{
				cachVig[iC] = cBadVignet;
				if (blockIdx.z == iDI) 	siCor[iTer] = 7*cBadVignet; // ## c ##
				return;
			}			
			const uint _cx	= coorTer.x * cDimVig.x + (x - c0.x);
			const uint _iC   = (blockIdx.z * cSizeCach) + _cy * cDimCach.x + _cx;

			cachVig[_iC] = (cacheImg[y][x] -aSV)/aSVV;
			//cachVig[_iC] = (y - c0.y) * cDimVig.x + x - c0.x;
			//cachVig[_iC]	= 3 * cBadVignet;
			//cachVig[iC]		= 4 * cBadVignet;
		}
	}

	/*if (blockIdx.z == iDI)
	{
		const uint _cx	= cRVig.x + coorTer.x * cDimVig.x ;
		const uint _cy	= cRVig.y + coorTer.y * cDimVig.y;
		const uint _iC   = (blockIdx.z * cSizeCach) + _cy * cDimCach.x + _cx;

		siCor[iTer] = (cachVig[_iC] == 0.0f) ? 9*cBadVignet : cachVig[_iC]; // ## ¤ ##
	}
*/

	// Nombre d'images correctes
	atomicAdd( &dev_NbImgOk[iTer], 1.0f);

};

// ---------------------------------------------------------------------------
// Calcul "rapide"  de la multi-correlation en utilisant la formule de Huygens
// ---------------------------------------------------------------------------
__global__ void multiCorrelationKernel(float *dTCost, float* cacheVign, float * dev_NbImgOk, uint2 actThr)
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
	if ( t.x >=  actThr.x || t.y >=  actThr.y )
		return;

	// Coordonnées 2D du cache vignette
	const uint2 cCach = make_uint2(blockIdx.x * actThr.x  + t.x, blockIdx.y * actThr.y  + t.y);
	
	// Si le thread est en dehors du cache
	if ( cCach.x >= cDimCach.x || cCach.y >= cDimCach.y )
		return;
	
	const uint pitCachLayer = threadIdx.z * cSizeCach;

	// Coordonnées 1D du cache vignette
	const uint iCach	= pitCachLayer + cCach.y * cDimCach.x + cCach.x ;
	
	// Coordonnées 2D du terrain 
	const uint2 coorTer		= cCach / cDimVig;
	
	// coordonnées central de la vignette
	const uint cx	= cRVig.x + coorTer.x * cDimVig.x;
	const uint cy	= cRVig.y + coorTer.y * cDimVig.y;
	const uint iCC	= pitCachLayer + cy * cDimCach.x + cx;

	// Coordonnées 1D dans le terrain
	const uint iTer		= coorTer.y * cRDiTer.x  + coorTer.x;

	if (cacheVign[iCC] == cBadVignet)
	{
		dTCost[iTer] = -1000.0f;
		return;
	}
	
	const bool mainThread	= mdlo(t.x,cDimVig.x) == 0 && mdlo(t.y,cDimVig.y) == 0 && threadIdx.z == 0;

	const uint aNbImOk		= dev_NbImgOk[iTer];

	if ( aNbImOk < 2)
	{
		if (mainThread)
			dTCost[iTer] = -1000.0f;
		return;
	}

	// Coordonnées 2D du terrain dans le repere des threads
	const uint2 coorTTer	= t / cDimVig;

	if (mainThread)
		dTCost[iTer] = aNbImOk;
	return;

	const float val	= cacheVign[iCach];

	__syncthreads();

	atomicAdd( &aSV[t.y][t.x], val);
	__syncthreads();

	atomicAdd( &aSVV[t.y][t.x], val * val);
	__syncthreads();

	atomicAdd(&resu[coorTTer.y][coorTTer.x],aSVV[t.y][t.x] - ( aSV[t.y][t.x] * aSV[t.y][t.x] / aNbImOk)); 
	__syncthreads();

	if ( !mainThread ) return;

	// Normalisation pour le ramener a un equivalent de 1-Correl 
	const float cost = resu[coorTTer.y][coorTTer.x] / (( aNbImOk-1) * cSizeVig);

	dTCost[iTer] = 1.0f - max (-1.0, min(1.0f,1.0f - cost));

}

extern "C" paramGPU Init_Correlation_GPU( int x0, int x1, int y0, int y1, int nbLayer , uint2 dRVig , uint2 dimImg, float mAhEpsilon, uint samplingZ, float uvDef )
{
	dev_NbImgOk		= NULL;
	dev_SimpCor		= NULL;
	dev_Cache		= NULL;
	dev_Cost		= NULL;

	correlOptionsGPU(x0, x1, y0, y1, dRVig * 2 + 1,dRVig, dimImg,mAhEpsilon, samplingZ, uvDef,nbLayer);
	allocMemory();

	return h;
}

extern "C" void basic_Correlation_GPU( float* h_TabCost,  int nbLayer ){
	
	//////////////////////////////////////////////////////////////////////////
	 
	int sCorMemSize = h.sizeTer  * sizeof(float);
	int nBI_MemSize = h.rSiTer	 * sizeof(float);
	int cac_MemSize = h.sizeCach * sizeof(float) * nbLayer;
	int costMemSize = h.rSiTer	 * sizeof(float);

	//////////////////////////////////////////////////////////////////////////

	checkCudaErrors( cudaMemset( dev_SimpCor,	0, sCorMemSize ));
	checkCudaErrors( cudaMemset( dev_Cost,		0, costMemSize ));
	checkCudaErrors( cudaMemset( dev_Cache,		0, cac_MemSize ));
	checkCudaErrors( cudaMemset( dev_NbImgOk,	0, nBI_MemSize ));
	checkCudaErrors( cudaBindTextureToArray(TexLay_Proj,dev_ProjLr) );

	//////////////////////////////////////////////////////////////////////////

	dim3 threads( BLOCKDIM, BLOCKDIM, 1);
	dim3 blocks(iDivUp(h.dimTer.x,threads.x - 2 *((int)h.dimVig.x)) , iDivUp(h.dimTer.y,threads.y - 2 * ((int)h.dimVig.y)), nbLayer);
	uint2 actiThs = make_uint2(SBLOCKDIM - SBLOCKDIM % ((int)h.dimVig.x), SBLOCKDIM - SBLOCKDIM % ((int)h.dimVig.y));

	dim3 threads_mC(SBLOCKDIM, SBLOCKDIM, nbLayer);
	dim3 blocks_mC(iDivUp(h.dimCach.x, actiThs.x) , iDivUp(h.dimCach.y, actiThs.y));

	////////////////////--  KERNEL  Correlation  --//////////////////////////
	
	correlationKernel<<<blocks, threads>>>( dev_NbImgOk, dev_Cache , dev_SimpCor);
	getLastCudaError("Basic Correlation kernel failed");
	
	//////////////////--  KERNEL  Multi Correlation  --///////////////////////

// 	multiCorrelationKernel<<<blocks_mC, threads_mC>>>( dev_Cost, dev_Cache, dev_NbImgOk,actiThs);
// 	getLastCudaError("Multi-Correlation kernel failed");

	//////////////////////////////////////////////////////////////////////////

	checkCudaErrors( cudaUnbindTexture(TexLay_Proj) );
	
	//checkCudaErrors( cudaMemcpy( h_TabCost, dev_Cost, costMemSize, cudaMemcpyDeviceToHost) );
	checkCudaErrors( cudaMemcpy( h_TabCost, dev_NbImgOk, costMemSize, cudaMemcpyDeviceToHost) );

	//checkCudaErrors( cudaMemcpy( host_SimpCor,	dev_SimpCor, sCorMemSize, cudaMemcpyDeviceToHost) );
	//checkCudaErrors( cudaMemcpy( host_Cache,	dev_Cache,	  cac_MemSize, cudaMemcpyDeviceToHost) );
	
	//////////////////////////////////////////////////////////////////////////

	//if(0)
	{
		//////////////////////////////////////////////////////////////////////////
		if (0)
		{

			int idii = 1;
			for (uint j = 0 ; j < h.dimCach.y ; j++)
			{
				for (uint i = 0; i < h.dimCach.x / 3 ; i++)
				{
					float off	= 1.0f;
					int id		= (idii * h.sizeCach + j * h.dimCach.x + i );
					float outC	= host_Cache[id];
					int bad = -4;
					if (outC == bad)
						std::cout << " *  ";
					else if (outC == 2*bad)
						std::cout << " -  ";
					else if (outC == 3*bad)
						std::cout << " +  ";
					else if (outC == 4*bad)
						std::cout << " O  ";
					else if (outC < 0 )
						std::cout << floor(outC*off)/off<< "  ";
					else if (outC >= 1e5 )
						std::cout << " !  ";
					else if (outC >= 10 )
						std::cout << " " << floor(outC*off)/off<< " ";
					else
						std::cout << " " << floor(outC*off)/off<< "  ";

				}
				std::cout << "\n";	
			}
			std::cout << "------------------------------------------\n";
		}

		//////////////////////////////////////////////////////////////////////////
/*

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

				std::string fileImaCache = "C:\\Users\\gchoqueux\\Pictures\\imageCache.pgm";
				// save PGM
				if (sdkSavePGM<float>(fileImaCache.c_str(), imageCache, dimCach.x,dimCach.y))
					std::cout <<"success save image" << "\n";
				else
					std::cout <<"Failed save image" << "\n";

				delete[] imageCache;
		
			float* image	= new float[h.sizeTer];
			for (uint j = 0; j < h.dimTer.y ; j++)
				for (uint i = 0; i < h.dimTer.x ; i++)
				{
					int id = (j * h.dimTer.x + i );
					//image[id] = host_SimpCor[id]/500.f;	
					image[id] = host_SimpCor[id]/2.0f;	
				}

			std::string file = "C:\\Users\\gchoqueux\\Pictures\\image.pgm";
			// save PGM
			if (sdkSavePGM<float>(file.c_str(), image, h.dimTer.x,h.dimTer.y))
				std::cout <<"success save image" << "\n";
			else
				std::cout <<"Failed save image" << "\n";

			delete[] image;
		}


		//////////////////////////////////////////////////////////////////////////

		/*for (uint j = 0 ; j < h.dimTer.y; j+= h.sampTer)
		{
			for (uint i = 0; i < h.dimTer.x ; i+= h.sampTer)
			{
				float off = 10.0f;
				int id = (j * h.dimTer.x + i );
				float out = host_SimpCor[id];
				std::cout << host_SimpCor[id] << " ";
			}
			std::cout << "\n";	
		}

		std::cout << "------------------------------------------\n";
*/
		//////////////////////////////////////////////////////////////////////////

		if (0)
		{
			for (uint j = 0 ; j < h.rDiTer.y; j++)
			{
				for (uint i = 0; i < h.rDiTer.x ; i++)
				{
					float off = 10.0f;
					int id = (j * h.rDiTer.x + i );
					float out = h_TabCost[id];
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

		if (0)
		{

			int bad = -4;
			for (uint j = 0 ; j < h.rDiTer.y; j+= h.sampTer)
			{
				for (uint i = 0; i < h.rDiTer.x ; i+= h.sampTer)
				{
					//float off = 10000000.0f;
					int id = (j * h.rDiTer.x + i );

					float out = h_TabCost[id];

					if (out == bad)
						std::cout << " ! ";
					else if (out == -1000.0f)
						std::cout << " n ";
					else if (out == 2*bad)
						std::cout << " . ";
					else if (out == 3*bad)
						std::cout << " z ";
					else if (out == 4*bad)
						std::cout << " s ";
					else if (out == 5*bad)
						std::cout << " v ";
					else if (out == 6*bad)
						std::cout << " e ";
					else if (out == 7*bad)
						std::cout << " c ";
					else if (out == 8*bad)
						std::cout << " ? ";
					else if (out == 9*bad)
						std::cout << " ¤ ";
					else if (out == 0.0f)
						std::cout << " 0 ";
					else if ( out < 0.0f && out > -1.0f)
					{
						//std::cout << floor(out*off)/off  << " ";
						std::cout << "|\\|";
					}
					else if ( out < 1.0f  && out > 0.0f)
						//std::cout << " "  <<  floor(out*off)/off  << " ";
						std::cout << "|/|";
					else
						//std::cout << floor(out*off)/off  << " ";
						std::cout << " * ";
				}

				std::cout << "\n";	
			}

			std::cout << "------------------------------------------\n";
		}
		
	}

	/*if(0)
	{

		float minCache =  1e10;
		float maxCache = -1e10;
		int step = 3;
		for (uint j = 0; j < h.dimTer.y * h.dimVig.y ; j+=step)
		{
			for (uint i = 0; i < h.dimTer.x * h.dimVig.x ; i+=step)
			{
				int id = (j * h.dimTer.x * h.dimVig.x + i );
				float c = host_Cache[id];

				if ( c < minCache || c > maxCache )
				{
					minCache = min( minCache, c);
					maxCache = max( maxCache, c);
					if(c!=0.0f)
					std::cout << minCache << " / " << maxCache << std::endl;
				}
			}
		}
	}*/
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
