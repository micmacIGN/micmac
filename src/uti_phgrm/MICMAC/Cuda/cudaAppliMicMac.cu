#include "gpu/cudaAppliMicMac.cuh"

#ifdef _DEBUG
	#define   BLOCKDIM	8
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
cudaArray* dev_ImagesLayered;	//
cudaArray* dev_ProjLayered;		//

//------------------------------------------------------------------------------------------
//float*	host_Corr_Out;
float*	host_Cache;
int*	host_NbImgOk;
float*	dev_Corr_Out;
float*	dev_Cache;
int*	dev_NbImgOk;
paramGPU h;

extern "C" void allocMemory(void)
{

	if (dev_NbImgOk != NULL) checkCudaErrors( cudaFree(dev_NbImgOk));
	if (dev_Corr_Out != NULL) checkCudaErrors( cudaFree(dev_Corr_Out));
	if (dev_Cache != NULL) checkCudaErrors( cudaFree(dev_Cache));

	int out_MemSize = h.sizeTer * sizeof(float);
	int nBI_MemSize = h.sizeTer	* sizeof(int);
	int cac_MemSize = h.sizeCach* sizeof(float)* h.nLayer;

	// Allocation mémoire
	//host_Corr_Out	= (float*)	malloc(out_MemSize);
	//host_Cache		= (float*)	malloc(cac_MemSize);
	//host_NbImgOk	= (int*)	malloc(nBI_MemSize);

	checkCudaErrors( cudaMalloc((void **) &dev_Corr_Out, out_MemSize) );	
	checkCudaErrors( cudaMalloc((void **) &dev_Cache, cac_MemSize ) );
	checkCudaErrors( cudaMalloc((void **) &dev_NbImgOk, nBI_MemSize ) );

	// Texture des projections
	TexLay_Proj.addressMode[0]	= cudaAddressModeClamp;
	TexLay_Proj.addressMode[1]	= cudaAddressModeClamp;	
	TexLay_Proj.filterMode		= cudaFilterModePoint; //cudaFilterModePoint 
	TexLay_Proj.normalized		= true;

}

extern "C" paramGPU updateSizeBlock( int x0, int x1, int y0, int y1 )
{

	int oldSizeTer = h.sizeTer;

	h.pUTer0.x	= x0 - h.rVig.x;
	h.pUTer0.y	= y0 - h.rVig.y;
	h.pUTer1.x	= x1 + h.rVig.x;
	h.pUTer1.y	= y1 + h.rVig.y;

	h.dimTer	= make_uint2(h.pUTer1.x - h.pUTer0.x, h.pUTer1.y - h.pUTer0.y);
	h.dimSTer	= iDivUp(h.dimTer,h.sampTer);	// Dimension du bloque terrain sous echantilloné
	h.sizeTer	= size(h.dimTer);				// Taille du bloque terrain
	h.sizeSTer  = size(h.dimSTer);				// Taille du bloque terrain sous echantilloné
	h.dimCach	= h.dimTer * h.dimVig;
	h.sizeCach	= size(h.dimCach);

	checkCudaErrors(cudaMemcpyToSymbol(cSDimTer, &h.dimSTer, sizeof(uint2)));
	checkCudaErrors(cudaMemcpyToSymbol(cDimTer, &h.dimTer, sizeof(uint2)));
	checkCudaErrors(cudaMemcpyToSymbol(cSizeTer, &h.sizeTer, sizeof(uint)));
	checkCudaErrors(cudaMemcpyToSymbol(cSizeSTer, &h.sizeSTer, sizeof(uint)));
	checkCudaErrors(cudaMemcpyToSymbol(cDimCach, &h.dimCach, sizeof(uint2)));
	checkCudaErrors(cudaMemcpyToSymbol(cSizeCach, &h.sizeCach, sizeof(uint)));

	if (oldSizeTer != h.sizeTer)
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

	checkCudaErrors(cudaMemcpyToSymbol(cRVig, &dRV, sizeof(uint2)));
	checkCudaErrors(cudaMemcpyToSymbol(cDimVig, &h.dimVig, sizeof(uint2)));
	checkCudaErrors(cudaMemcpyToSymbol(cDimImg, &dI, sizeof(uint2)));
	checkCudaErrors(cudaMemcpyToSymbol(cMAhEpsilon, &mAhEpsilon, sizeof(float)));
	checkCudaErrors(cudaMemcpyToSymbol(cSizeVig, &h.sizeVig, sizeof(uint)));
	checkCudaErrors(cudaMemcpyToSymbol(cSampTer, &h.sampTer, sizeof(uint)));
	checkCudaErrors(cudaMemcpyToSymbol(cUVDefValue, &h.UVDefValue, sizeof(float)));
	
	updateSizeBlock( x0, x1, y0, y1 );
}

extern "C" void imagesToLayers(float *fdataImg1D, uint2 dimTer, int nbLayer)
{
	cudaExtent sizeImgsLay = make_cudaExtent( dimTer.x, dimTer.y, nbLayer );

	// Définition du format des canaux d'images
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

	// Allocation memoire GPU du tableau des calques d'images
	checkCudaErrors( cudaMalloc3DArray(&dev_ImagesLayered,&channelDesc,sizeImgsLay,cudaArrayLayered) );

	// Déclaration des parametres de copie 3D
	cudaMemcpy3DParms	p	= { 0 };
	cudaPitchedPtr		pit = make_cudaPitchedPtr(fdataImg1D, sizeImgsLay.width * sizeof(float), sizeImgsLay.width, sizeImgsLay.height);

	p.dstArray	= dev_ImagesLayered;		// Pointeur du tableau de destination
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
	checkCudaErrors( cudaBindTextureToArray(refTex_ImagesLayered,dev_ImagesLayered) );

};

extern "C" void  projectionsToLayers(float *h_TabProj, uint2 dimTer, int nbLayer)
{
	// Définition du format des canaux d'images
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float2>();

	// Taille du tableau des calques 
	cudaExtent siz_PL = make_cudaExtent( dimTer.x, dimTer.y, nbLayer);

	// Allocation memoire GPU du tableau des calques d'images
	checkCudaErrors( cudaMalloc3DArray(&dev_ProjLayered,&channelDesc,siz_PL,cudaArrayLayered ));

	// Déclaration des parametres de copie 3D
	cudaMemcpy3DParms p = { 0 };

	p.dstArray	= dev_ProjLayered;			// Pointeur du tableau de destination
	p.srcPtr	= make_cudaPitchedPtr(h_TabProj, siz_PL.width * sizeof(float2), siz_PL.width, siz_PL.height);
	p.extent	= siz_PL;
	p.kind		= cudaMemcpyHostToDevice;	// Type de copie

	// Copie des projections du Host vers le Device
	checkCudaErrors( cudaMemcpy3D(&p) );

};

__device__  inline float2 simpleProjection( uint2 size, uint2 ssize, uint2 sizeImg ,uint2 coord, int L)
{

	const float2 cf = make_float2(ssize) * make_float2(coord) / make_float2(size) ;
	const int2	 a	= make_int2(cf);
	const float2 uva = (make_float2(a) + 0.5f) / (make_float2(ssize));
	const float2 uvb = (make_float2(a+1) + 0.5f) / (make_float2(ssize));
	float2 ra, rb, Iaa;

	ra	= tex2DLayered( TexLay_Proj, uva.x, uva.y, L);
	rb	= tex2DLayered( TexLay_Proj, uvb.x, uva.y, L);
	if (ra.x < 0.0f || ra.y < 0.0f || rb.x < 0.0f || rb.y < 0.0f)
		return make_float2(-1.0f,-1.0f);

	Iaa	= ((float)(a.x + 1.0f) - cf.x) * ra + (cf.x - (float)(a.x)) * rb;
	ra	= tex2DLayered( TexLay_Proj, uva.x, uvb.y, L);
	rb	= tex2DLayered( TexLay_Proj, uvb.x, uvb.y, L);

	if (ra.x < 0.0f || ra.y < 0.0f || rb.x < 0.0f || rb.y < 0.0f)
		return make_float2(-1.0f,-1.0f);

	ra	= ((float)(a.x+ 1.0f) - cf.x) * ra + (cf.x - (float)(a.x)) * rb;
	
	ra = ((float)(a.y+ 1.0f) - cf.y) * Iaa + (cf.y - (float)(a.y)) * ra;
	ra = (ra + 0.5f) / (make_float2(sizeImg));

	return ra;
}

__global__ void correlationKernel( int *dev_NbImgOk, float* cache, float *dest )
//__global__ void correlationKernel( int *dev_NbImgOk, float* cache)
{
	__shared__ float cacheImg[ BLOCKDIM ][ BLOCKDIM ];

	// Se placer dans l'espace terrain
	const uint2	coorTer = make_uint2(blockIdx) * (make_uint2(blockDim) - 2 * cRVig) + make_uint2(threadIdx) - cRVig;
	const uint	iTer	= coorTer.y * cDimTer.x + coorTer.x;

	// Si le processus est hors du terrain, nous sortons du kernel
	if ( coorTer.x >= cDimTer.x || coorTer.y >= cDimTer.y || coorTer.x < 0 || coorTer.y < 0) 
		return;

	const float2 PtTProj = simpleProjection( cDimTer, cSDimTer, cDimImg, coorTer, blockIdx.z);

	if ( PtTProj.x < 0.0f ||  PtTProj.y < 0.0f )
	{
		cacheImg[threadIdx.y][threadIdx.x]  = -1.0f;
		//dest[iTer] = -1.0f;
		return;
	}
	else
	{
		cacheImg[threadIdx.y][threadIdx.x] = tex2DLayered( refTex_ImagesLayered, PtTProj.x, PtTProj.y,blockIdx.z);
		//dest[iTer] = cacheImg[threadIdx.y][threadIdx.x] ;
	}

	__syncthreads();


	// Intialisation des valeurs de calcul 
	float		aSV	= 0.0f;
	float	   aSVV	= 0.0f;
	const int2 c0	= (threadIdx - cRVig);
	const int2 c1	= (threadIdx + cRVig);

	if ( c1.x >= blockDim.x || c1.y >= blockDim.y || c0.x < 0 || c0.y < 0 )
	{
		//if (threadIdx.z == 0)
			//dest[iTer] = cacheImg[threadIdx.y][threadIdx.x] ;
		return;
	}
	else
	{
		#pragma unroll
		for (int y = c0.y ; y <= c1.y; y++)
		{
			#pragma unroll
			for (int x = c0.x ; x <= c1.x; x++)
			{	
				const float val = cacheImg[y][x];	// Valeur de l'image

				if (val < 0.0f) return;
				aSV  += val;			// Somme des valeurs de l'image cte 
				aSVV += val*val;		// Somme des carrés des vals image cte
			}
		}
	}

	aSV 		/=	cSizeVig;
	aSVV 		/=	cSizeVig;
	aSVV		-=	aSV * aSV;
	
	if ( aSVV <= cMAhEpsilon)
		return;
	
	const uint iC   = blockIdx.z * cSizeCach + coorTer.y * cDimVig.y * cDimCach.x + coorTer.x * cDimVig.x;
	aSVV =	sqrt(aSVV);

	#pragma unroll
	for (int y = c0.y ; y <= c1.y; y++)
	{
		const uint pCach = cDimCach.x * (y - c0.y);
		#pragma unroll
		for (int x = c0.x ; x <= c1.x; x++)
		{
			if (cacheImg[y][x] < 0.0f) return;
			cache[iC + pCach  +  x - c0.x] = (cacheImg[y][x] -aSV)/aSVV;
		}
	}

	// Nombre d'images correctes
	atomicAdd( &dev_NbImgOk[iTer], 1);
};

// ---------------------------------------------------------------------------
// Calcul "rapide"  de la multi-correlation en utilisant la formule de Huygens
// ---------------------------------------------------------------------------
__global__ void multiCorrelationKernel(float *dest, float* cache, int * dev_NbImgOk)
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

	// Threads utilisées dans le bloque
	const uint2 activThds = make_uint2(blockDim.x - blockDim.x % cDimVig.x, blockDim.y - blockDim.y % cDimVig.y);

	// si le thread est inactif, il sort
	if ( t.x >=  activThds.x || t.y >=  activThds.y )
		return;

	// Coordonnées 3D du cache
	const uint2 coordCach		= make_uint2(blockIdx.x * activThds.x  + t.x, blockIdx.y * activThds.y  + t.y);

	// Si le thread est en dehors du cache
	if ( coordCach.x >=  cDimTer.x * cDimVig.x || coordCach.y >=  cDimTer.y * cDimVig.y )
		return;
	
	// Coordonnées 1D du cache
	const unsigned int iCach	= threadIdx.z * cSizeTer * cSizeVig + coordCach.y * cDimTer.x * cDimVig.x + coordCach.x ;

	// Coordonnées 2D du terrain 
	const uint2 coordTer		= coordCach / cDimVig;
	
	// Coordonnées 1D dans le cache
	const unsigned int iTer		= coordTer.y * cDimTer.x  + coordTer.x;

	// Coordonnées 2D du terrain dans le repere des threads
	const uint2 coorTTer	= t / cDimVig;

	const bool mainThread	= (t.x % cDimVig.x)== 0 && (t.y % cDimVig.y) == 0 && threadIdx.z == 0;
	
	const uint aNbImOk		= dev_NbImgOk[iTer];

	if ( aNbImOk < 2)
	{
		if (mainThread)
			dest[iTer] = -1.0f;
		return;
	}

	const float val = cache[iCach];

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

	dest[iTer] = 1.0f - max (-1.0, min(1.0,1.0f - cost));

}

extern "C" paramGPU Init_Correlation_GPU( int x0, int x1, int y0, int y1, int nbLayer , uint2 dRVig , uint2 dimImg, float mAhEpsilon, uint samplingZ, float uvDef )
{
	dev_NbImgOk		= NULL;
	dev_Corr_Out	= NULL;
	dev_Cache		= NULL;

	correlOptionsGPU(x0, x1, y0, y1, dRVig * 2 + 1,dRVig, dimImg,mAhEpsilon, samplingZ, uvDef,nbLayer);
	allocMemory();

	return h;
}

extern "C" void basic_Correlation_GPU( float* h_TabCorre,  int nbLayer ){
	
	int out_MemSize = h.sizeTer  * sizeof(float);
	int nBI_MemSize = h.sizeTer	 * sizeof(int);
	int cac_MemSize = h.sizeCach * sizeof(float) * nbLayer;

	checkCudaErrors( cudaMemset( dev_Corr_Out, 0, out_MemSize ));
	checkCudaErrors( cudaMemset( dev_Cache, 0, cac_MemSize ));
	checkCudaErrors( cudaMemset( dev_NbImgOk, 0, nBI_MemSize ));
	checkCudaErrors( cudaBindTextureToArray(TexLay_Proj,dev_ProjLayered) );
	
	//------------   Kernel correlation   ----------------
	
	dim3 threads( BLOCKDIM, BLOCKDIM, 1);
	//dim3 blocks(iDivUp(h.dimTer.x,threads.x) , iDivUp(h.dimTer.y,threads.y), nbLayer);
	dim3 blocks(iDivUp(h.dimTer.x,threads.x - 2 * h.dimVig.x) , iDivUp(h.dimTer.y,threads.y - 2 * h.dimVig.y), nbLayer);

	correlationKernel<<<blocks, threads>>>( dev_NbImgOk, dev_Cache , dev_Corr_Out);
	getLastCudaError("Basic Correlation kernel failed");
	
	//checkCudaErrors( cudaDeviceSynchronize() );

	//---------- Kernel multi-correlation -----------------

	int actiThs_X = SBLOCKDIM - SBLOCKDIM % h.dimVig.x;
	int actiThs_Y = SBLOCKDIM - SBLOCKDIM % h.dimVig.y;

	dim3 threads_mC(SBLOCKDIM, SBLOCKDIM, nbLayer);
	dim3 blocks_mC(iDivUp(h.dimTer.x * h.dimVig.x  ,actiThs_X) , iDivUp(h.dimTer.y * h.dimVig.y ,actiThs_Y));

	multiCorrelationKernel<<<blocks_mC, threads_mC>>>( dev_Corr_Out, dev_Cache, dev_NbImgOk);
	getLastCudaError("Multi-Correlation kernel failed");

	checkCudaErrors( cudaUnbindTexture(TexLay_Proj) );
	//checkCudaErrors( cudaMemcpy( host_Corr_Out,	dev_Corr_Out, out_MemSize, cudaMemcpyDeviceToHost) );
	checkCudaErrors( cudaMemcpy( h_TabCorre, dev_Corr_Out, out_MemSize, cudaMemcpyDeviceToHost) );
	//checkCudaErrors( cudaMemcpy( host_NbImgOk,	dev_NbImgOk,  nBI_MemSize, cudaMemcpyDeviceToHost) );
	//checkCudaErrors( cudaMemcpy( host_Cache,	dev_Cache,	  cac_MemSize, cudaMemcpyDeviceToHost) );
	//--------------------------------------------------------

	if(0)
	{
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
					//image[id] = h_TabCorre[id]/500.f;	
					image[id] = h_TabCorre[id]/2.0f;	
				}

			std::string file = "C:\\Users\\gchoqueux\\Pictures\\image.pgm";
			// save PGM
			if (sdkSavePGM<float>(file.c_str(), image, h.dimTer.x,h.dimTer.y))
				std::cout <<"success save image" << "\n";
			else
				std::cout <<"Failed save image" << "\n";

			delete[] image;
		}


		for (uint j = 0; j < h.dimTer.y ; j+= h.sampTer)
		{
			for (uint i = 0; i < h.dimTer.x ; i+= h.sampTer)
			{
				int id = (j * h.dimTer.x + i );
				//image[id] = h_TabCorre[id]/500.f;	
				if (h_TabCorre[id]!=-1.0f)
				{
					float off = 2.0f;
					std::cout << floor(h_TabCorre[id]*off)/off  << " ";
					//std::cout << h_TabCorre[id]  << " ";

				}
				else
					std::cout << "  .  ";
			}

				std::cout << "\n";	
		}

		std::cout << "------------------------------------------\n";	
		
	}
	if(0)
	{

		float minCache =  1000000000000.0f;
		float maxCache = -1000000000000.0f;
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
				//float c = host_NbImgOk[id];
				//std::cout << c << " ";
			}
			//std::cout << std::endl; 
		}
	}

}

extern "C" void freeGpuMemory()
{
	checkCudaErrors( cudaUnbindTexture(refTex_Image) );
	checkCudaErrors( cudaUnbindTexture(refTex_ImagesLayered) );
	checkCudaErrors( cudaFreeArray(dev_Img) );
	if(dev_ImagesLayered != NULL) checkCudaErrors( cudaFreeArray(dev_ImagesLayered) );
	checkCudaErrors( cudaFreeArray(dev_CubeProjImg) );
	checkCudaErrors( cudaFreeArray(dev_ArrayProjImg) );
	if(dev_ProjLayered != NULL)  checkCudaErrors( cudaFreeArray(dev_ProjLayered) );
	if(dev_NbImgOk != NULL) checkCudaErrors( cudaFree(dev_NbImgOk));
	if(dev_Corr_Out != NULL) checkCudaErrors( cudaFree(dev_Corr_Out));
	if(dev_Cache != NULL) checkCudaErrors( cudaFree(dev_Cache));
	//free(host_Corr_Out);
	//free(host_NbImgOk); 
	//free(host_Cache);
}

extern "C" void  FreeLayers()
{
	checkCudaErrors( cudaFreeArray(dev_ImagesLayered));

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
