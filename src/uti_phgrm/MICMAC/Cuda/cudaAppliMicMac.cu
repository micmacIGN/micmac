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
float*	host_Corr_Out;
float*	host_Cache;
int*	host_NbImgOk;
float*	dev_Corr_Out;
float*	dev_Cache;
int*	dev_NbImgOk;
paramGPU h;

static void correlOptionsGPU(uint2 dTer, uint2 dV,uint2 dRV, uint2 dI, float mAhEpsilon, uint samplingZ, float uvDef )
{

	h.DimTer	= dTer;							// Dimension du bloque terrain
	h.SDimTer	= iDivUp(h.DimTer,samplingZ);	// Dimension du bloque terrain sous echantilloné
	h.DimVig	= dV;							// Dimension de la vignette
	h.DimImg	= dI;							// Dimension des images
	h.RVig		= dRV;							// Rayon de la vignette
	h.SizeVig	= size(dV);						// Taille de la vignette en pixel 
	h.SizeTer	= size(dTer);					// Taille du bloque terrain
	h.SizeSTer  = size(h.SDimTer);				// Taille du bloque terrain sous echantilloné
	h.SampTer	= samplingZ;					// Pas echantillonage du terrain
	h.UVDefValue= uvDef;						// UV Terrain incorrect

	checkCudaErrors(cudaMemcpyToSymbol(cDimVig, &dV, sizeof(uint2)));
	checkCudaErrors(cudaMemcpyToSymbol(cSDimTer, &h.SDimTer, sizeof(uint2)));
	checkCudaErrors(cudaMemcpyToSymbol(cRVig, &dRV, sizeof(uint2)));
	checkCudaErrors(cudaMemcpyToSymbol(cDimTer, &dTer, sizeof(uint2)));
	checkCudaErrors(cudaMemcpyToSymbol(cDimImg, &dI, sizeof(uint2)));
	checkCudaErrors(cudaMemcpyToSymbol(cMAhEpsilon, &mAhEpsilon, sizeof(float)));
	checkCudaErrors(cudaMemcpyToSymbol(cSizeVig, &h.SizeVig, sizeof(uint)));
	checkCudaErrors(cudaMemcpyToSymbol(cSizeTer, &h.SizeTer, sizeof(uint)));
	checkCudaErrors(cudaMemcpyToSymbol(cSizeSTer, &h.SizeSTer, sizeof(uint)));
	checkCudaErrors(cudaMemcpyToSymbol(cSampTer, &h.SampTer, sizeof(uint)));
	checkCudaErrors(cudaMemcpyToSymbol(cUVDefValue, &h.UVDefValue, sizeof(float)));

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
    refTex_ImagesLayered.filterMode		= cudaFilterModePoint;
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

__global__ void correlationKernel( int *dev_NbImgOk, float* cache)
{
	__shared__ float cacheImg[ BLOCKDIM ][ BLOCKDIM ];

	// Se placer dans l'espace terrain
	const uint2	coorTer = make_uint2(blockIdx) * make_uint2(blockDim.x,blockDim.y) + make_uint2(threadIdx);
	const uint	iTer	= coorTer.y * cDimTer.x + coorTer.x;

	// Si le processus est hors du terrain, nous sortons du kernel
	if ( coorTer.x >= cDimTer.x || coorTer.y >= cDimTer.y) 
		return;

	const float2 PtTProj	= simpleProjection( cDimTer, cDimTer/5, cDimImg, coorTer, blockIdx.z);
	
	if ( PtTProj.x < 0.0f ||  PtTProj.y < 0.0f )
	{
		cacheImg[threadIdx.x][threadIdx.y]  = 0.0f;
		return;
	}
	else
		cacheImg[threadIdx.x][threadIdx.y] = tex2DLayered( refTex_ImagesLayered, PtTProj.x, PtTProj.y,blockIdx.z);
	
	__syncthreads();

	// Intialisation des valeurs de calcul 
	float		aSV	= 0.0f;
	float	   aSVV	= 0.0f;
	const uint	x0	= threadIdx.x - cRVig.x;
	const uint	x1	= threadIdx.x + cRVig.x;
	const uint	y0	= threadIdx.y - cRVig.y;
	const uint	y1	= threadIdx.y + cRVig.y;

	if ((x1 >= blockDim.x )||(y1 >= blockDim.y )||(x0 < 0)||(y0 < 0)) 
		return;
	else
	{
		#pragma unroll
		for (int y = y0 ; y <= y1; y++)
		{
			#pragma unroll
			for (int x = x0 ; x <= x1; x++)
			{	
				const float val = cacheImg[y][x];	// Valeur de l'image
				aSV  += val;				// Somme des valeurs de l'image cte 
				aSVV += val*val;			// Somme des carrés des vals image cte
			}
		}
	}

	aSV 		/=	cSizeVig;
	aSVV 		/=	cSizeVig;
	aSVV		-=	aSV * aSV;
	
	if ( aSVV <= cMAhEpsilon) return;

	const uint iCach = ( iTer + blockIdx.z * cSizeTer) * cSizeVig; 

	aSVV =	sqrt(aSVV);

	#pragma unroll
	for (int y = y0 ; y <= y1; y++)
	{
		const uint pitchV = cDimVig.x *  ( y - y0); 
		#pragma unroll
		for (int x = x0 ; x <= x1; x++)	
			cache[iCach + pitchV  +  x - x0] = (cacheImg[y][x] -aSV)/aSVV;
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

	// nombres de threads utilisées dans le bloques
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

extern "C" paramGPU Init_Correlation_GPU( uint2 dimTer, int nbLayer , uint2 dRVig , uint2 dimImg, float mAhEpsilon, uint samplingZ, float uvDef )
{

	correlOptionsGPU(dimTer,dRVig * 2 + 1,dRVig, dimImg,mAhEpsilon, samplingZ, uvDef);
	int out_MemSize = h.SizeTer * sizeof(float);
	int nBI_MemSize = h.SizeTer * sizeof(int);
	int cac_MemSize = out_MemSize * nbLayer * h.SizeVig;

	// Allocation mémoire
	host_Corr_Out = (float*)	malloc(out_MemSize);
	//host_Cache		= (float*)	malloc(cac_MemSize);
	//host_NbImgOk	= (int*)	malloc(nBI_MemSize);

	checkCudaErrors( cudaMalloc((void **) &dev_Corr_Out, out_MemSize) );	
	checkCudaErrors( cudaMalloc((void **) &dev_Cache, cac_MemSize ) );
	checkCudaErrors( cudaMalloc((void **) &dev_NbImgOk, nBI_MemSize ) );

	// Texture des projections
	TexLay_Proj.addressMode[0]	= cudaAddressModeClamp;
    TexLay_Proj.addressMode[1]	= cudaAddressModeClamp;	
    TexLay_Proj.filterMode		= cudaFilterModeLinear; //cudaFilterModePoint 
    TexLay_Proj.normalized		= true;

	return h;
}

extern "C" void basic_Correlation_GPU( float* h_TabCorre,  int nbLayer ){

	int out_MemSize = h.SizeTer * sizeof(float);
	int nBI_MemSize = h.SizeTer * sizeof(int);
	int cac_MemSize = out_MemSize * nbLayer * h.SizeVig;

	checkCudaErrors( cudaMemset( dev_Corr_Out, 0, out_MemSize ));
	checkCudaErrors( cudaMemset( dev_Cache, 0, cac_MemSize ));
	checkCudaErrors( cudaMemset( dev_NbImgOk, 0, nBI_MemSize ));
	checkCudaErrors( cudaBindTextureToArray(TexLay_Proj,dev_ProjLayered) );


	//------------   Kernel correlation   ----------------
		dim3 threads(BLOCKDIM, BLOCKDIM, 1);
		dim3 blocks(iDivUp(h.DimTer.x,threads.x) , iDivUp(h.DimTer.y,threads.y), nbLayer);
		
		correlationKernel<<<blocks, threads>>>( dev_NbImgOk, dev_Cache);
		getLastCudaError("Basic Correlation kernel failed");
		//checkCudaErrors( cudaDeviceSynchronize() );

	//---------- Kernel multi-correlation -----------------
	{
		int actiThs_X = SBLOCKDIM - SBLOCKDIM % h.DimVig.x;
		int actiThs_Y = SBLOCKDIM - SBLOCKDIM % h.DimVig.x;

		dim3 threads_mC(SBLOCKDIM, SBLOCKDIM, nbLayer);
		dim3 blocks_mC(iDivUp(h.DimTer.x * h.DimVig.x  ,actiThs_X) , iDivUp(h.DimTer.y * h.DimVig.y ,actiThs_Y));

		multiCorrelationKernel<<<blocks_mC, threads_mC>>>( dev_Corr_Out, dev_Cache, dev_NbImgOk);
		getLastCudaError("Multi-Correlation kernel failed");
	}

	//checkCudaErrors( cudaDeviceSynchronize() );
	checkCudaErrors( cudaUnbindTexture(TexLay_Proj) );
	checkCudaErrors( cudaMemcpy( host_Corr_Out,	dev_Corr_Out, out_MemSize, cudaMemcpyDeviceToHost) );
	//checkCudaErrors( cudaMemcpy( host_NbImgOk,	dev_NbImgOk,  nBI_MemSize, cudaMemcpyDeviceToHost) );
	//checkCudaErrors( cudaMemcpy( host_Cache,	dev_Cache,	  cac_MemSize, cudaMemcpyDeviceToHost) );
	//--------------------------------------------------------

	//if(0)
	{
		int step = 2;
		std::cout << " --------------------  size ter (x,y) : " << iDivUp(h.DimTer.x, step) << ", " << iDivUp(h.DimTer.y, step) << std::endl;
		for (uint j = 0; j < h.DimTer.y ; j+=step)
		{
			std::cout << "       "; 
			for (uint i = 0; i < h.DimTer.x ; i+=step)
			{
				int id = (j * h.DimTer.x + i );
				float c = host_Corr_Out[id];
				if( c > 0.0f)
					std::cout << floor(c*1000)/1000 << " ";
				else if( c == -1.0f)			
					std::cout << " .  ";
				else
					std::cout << " -  ";

			}
			std::cout << std::endl; 
		}
	}

	if(0)
	{

		float minCache =  1000000000000.0f;
		float maxCache = -1000000000000.0f;
		int step = 1;
		std::cout << "Taille du cache (x,y) : ..??" << std::endl;
		for (uint j = 0; j < h.DimTer.y * h.DimVig.y ; j+=step)
		{
			for (uint i = 0; i < h.DimTer.x * h.DimVig.x ; i+=step)
			{
				int id = (j * h.DimTer.x * h.DimVig.x + i );
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
	//--------------------------------------------------------
	

}

extern "C" void freeGpuMemory()
{
	checkCudaErrors( cudaUnbindTexture(refTex_Image) );
	checkCudaErrors( cudaUnbindTexture(refTex_ImagesLayered) );
	checkCudaErrors( cudaFreeArray(dev_Img) );
	checkCudaErrors( cudaFreeArray(dev_ImagesLayered) );
	checkCudaErrors( cudaFreeArray(dev_CubeProjImg) );
	checkCudaErrors( cudaFreeArray(dev_ArrayProjImg) );
	checkCudaErrors( cudaFreeArray(dev_ProjLayered) );
	checkCudaErrors( cudaFree(dev_NbImgOk));
	checkCudaErrors( cudaFree(dev_Corr_Out));
	checkCudaErrors( cudaFree(dev_Cache));
	free(host_Corr_Out);
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
