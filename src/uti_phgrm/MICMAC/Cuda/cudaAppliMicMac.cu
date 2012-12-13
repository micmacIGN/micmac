#include "cudaAppliMicMac.cuh"

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

extern "C" void imagesToLayers(float *fdataImg1D, uint2 dimImg, int nbLayer)
{
	cudaExtent sizeImgsLay = make_cudaExtent( dimImg.x, dimImg.y, nbLayer );

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

__device__  float2 simpleProjection(texture<float2, cudaTextureType2DLayered, cudaReadModeElementType> texRef, uint2 size, uint2 ssize, uint2 sizeImg ,uint2 coord, int L)
{
	const float2 cf = make_float2(ssize) * make_float2(coord) / make_float2(size) ;
	const int2	 a	= make_int2(cf); 
	const int2	 b	= a + 1; 
	const float2 uva = (make_float2(a) + 0.5f) / (make_float2(ssize));
	const float2 uvb = (make_float2(b) + 0.5f) / (make_float2(ssize));
	
	float2 ra, rb, Iaa, Iba, result;

	ra	= tex2DLayered( texRef, uva.x, uva.y, L);
	rb	= tex2DLayered( texRef, uvb.x, uva.y, L);

	if (ra.x < 0.0f || ra.y < 0.0f || rb.x < 0.0f || rb.y < 0.0f)
		return make_float2(-1.0f,-1.0f);

	Iaa	= ((float)(b.x) - cf.x) * ra + (cf.x - (float)(a.x)) * rb;
	ra	= tex2DLayered( texRef, uva.x, uvb.y, L);
	rb	= tex2DLayered( texRef, uvb.x, uvb.y, L);

	if (ra.x < 0.0f || ra.y < 0.0f || rb.x < 0.0f || rb.y < 0.0f)
		return make_float2(-1.0f,-1.0f);

	Iba	= ((float)(b.x) - cf.x) * ra + (cf.x - (float)(a.x)) * rb;

	result = ((float)(b.y) - cf.y) * Iaa + (cf.y - (float)(a.y)) * Iba;
	
	return (result + 0.5f) / (make_float2(sizeImg));
}

__global__ void correlationKernel(float* dest, int *dev_NbImgOk, float* cache, uint2 dimTer, uint2 sDimTer, uint2 rVig, uint2 dimImg, float mAhEpsilon )
{
	__shared__ float cacheImg[ BLOCKDIM ][ BLOCKDIM ];

	// coordonnées des threads
	const unsigned int tx		= threadIdx.x;
	const unsigned int ty		= threadIdx.y;

	// Se placer dans l'espace terrain
	const uint2 coorTer	= make_uint2( blockIdx.x * blockDim.x  + tx, blockIdx.y * blockDim.y  + ty);
	const int L			= blockIdx.z;
	const int iTer		= coorTer.y * dimTer.x + coorTer.x;

	// Si le processus est hors du terrain, nous sortons du kernel
	if ( coorTer.x >= dimTer.x || coorTer.y >= dimTer.y) 
		return;

	// Coordonnées de textures dans l'image
	const float2 uv	= simpleProjection( TexLay_Proj, dimTer,sDimTer, dimImg,coorTer, L);

	// Valeur de l'image
	if ( uv.x < 0.0f ||  uv.y < 0.0f )
		return;
	else
		cacheImg[tx][ty] = tex2DLayered( refTex_ImagesLayered, uv.x, uv.y,L);
	
	__syncthreads();

	// Intialisation des valeurs de calcul 
	float		aSV	= 0.0f;
	float	   aSVV	= 0.0f;
	const int	x0	= tx - rVig.x;
	const int	x1	= tx + rVig.x;
	const int	y0	= ty - rVig.y;
	const int	y1	= ty + rVig.y;

	if ((x1 >= blockDim.x )|(y1 >= blockDim.y )|(x0 < 0)|(y0 < 0)) 
		return;
	else
	{
		#pragma unroll
		for (int y = y0 ; y <= y1; y++)
		{
			#pragma unroll
			for (int x = x0 ; x <= x1; x++)
			{	
				float val = cacheImg[y][x];	// Valeur de l'image
				aSV  += val;				// Somme des valeurs de l'image cte 
				aSVV += val*val;			// Somme des carrés des vals image cte
			}
		}
	}



	uint2 dimVig = make_uint2( 2 * rVig.x + 1, 2 * rVig.y + 1);

	int size_Vign	 = size(dimVig);
	aSV 		/=	size_Vign;
	aSVV 		/=	size_Vign;
	aSVV		-=	aSV * aSV;
	
	if ( aSVV <= mAhEpsilon ) return;

	int iCach = ( iTer + L * size(dimTer)) * size_Vign; 

	aSVV =	sqrt(aSVV);

	#pragma unroll
	for (int y = y0 ; y <= y1; y++)
	{
		int pitchV = dimVig.x *  ( y - y0); 
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
__global__ void multiCorrelationKernel(float *dest, float* cache, int * dev_NbImgOk, uint2 dimTer, uint2 rVig, uint2 dimImg)
{

	__shared__ float aSV [ SBLOCKDIM ][ SBLOCKDIM ];
	__shared__ float aSVV[ SBLOCKDIM ][ SBLOCKDIM ];
	__shared__ float resu[ SBLOCKDIM/2 ][ SBLOCKDIM/2 ];

	// dimensions des vignettes
	const uint2 dimVig = make_uint2(2 * rVig.x + 1, 2 * rVig.y + 1);

	// coordonnées des threads
	const uint2 t = make_uint2(threadIdx.x,threadIdx.y);
	const unsigned int tz		= threadIdx.z;

	aSV [t.y][t.x]		= 0.0f;
	aSVV[t.y][t.x]		= 0.0f;
	resu[t.y/2][t.x/2]	= 0.0f;
	__syncthreads();

	// nombres de threads utilisées dans le bloques
	const unsigned int actiThs_X = blockDim.x - blockDim.x % dimVig.x;
	const unsigned int actiThs_Y = blockDim.y - blockDim.y % dimVig.y;

	// si le thread est inactif, il sort
	if ( t.x >=  actiThs_X || t.y >=  actiThs_Y )
		return;
	
	// taille de la vignette et du terrain
	const unsigned int size_Vign = size(dimVig);
	const unsigned int size_Terr = size(dimTer);

	// Coordonnées 3D du cache
	const uint2 cC = make_uint2(blockIdx.x * actiThs_X  + t.x,blockIdx.y * actiThs_Y  + t.y);
	const unsigned int l		= threadIdx.z;

	// Si le thread est en dehors du cache
	if ( cC.x >=  dimTer.x * dimVig.x || cC.y >=  dimTer.y * dimVig.y )
		return;

	// Coordonnées 1D du cache
	const unsigned int iCach	= l * size_Terr * size_Vign + cC.y * dimTer.x * dimVig.x + cC.x ;

	// Coordonnées 2D du terrain 
	const uint2 coordTer = cC / dimVig;

	// Coordonnées 1D dans le cache
	const unsigned int iTer	= coordTer.y * dimTer.x  + coordTer.x;

	// Coordonnées 2D du terrain dans le repere des threads
	const uint2 tT = t / dimVig;

	bool mainThread = (t.x % dimVig.x)== 0 && (t.y % dimVig.y) == 0 && tz == 0;

	int aNbImOk = dev_NbImgOk[iTer];

	if ( aNbImOk < 2)
	{
		if (mainThread)
			dest[iTer] = -1.0f;
		return;
	}

	float val = cache[iCach];

	__syncthreads();

	atomicAdd( &aSV[t.y][t.x], val);
	
	__syncthreads(); //printf ("aSV : %d, %4.2f, val :%4.2f | ", tz ,aSV[t.y][tx], val);

	atomicAdd( &aSVV[t.y][t.x], val * val);

	__syncthreads();

	atomicAdd(&resu[tT.y][tT.x],aSVV[tT.y][tT.x] - ( aSV[tT.y][tT.x] * aSV[tT.y][tT.x] / aNbImOk)); 

	__syncthreads();

	if ( !mainThread ) return;

	// Normalisation pour le ramener a un equivalent de 1-Correl 
	float cost = resu[tT.y][tT.x] / (( aNbImOk-1) * size_Vign);

	dest[iTer] = 1.0f - max (-1.0, min(1.0,1.0f - cost));

}

extern "C" void Init_Correlation_GPU( int sTer_X, int sTer_Y, int nbLayer , int rxVig, int ryVig )
{
	int svX = ( rxVig * 2 + 1 );
	int svY = ( ryVig * 2 + 1 );
	int out_MemSize = sTer_X * sTer_Y * sizeof(float);
	int nBI_MemSize = sTer_X * sTer_Y * sizeof(int);
	int cac_MemSize = out_MemSize * nbLayer * svX * svY;

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
    TexLay_Proj.filterMode		= cudaFilterModePoint; //cudaFilterModePoint //cudaFilterModeLinear
    TexLay_Proj.normalized		= true;
}

extern "C" void basic_Correlation_GPU(  float* h_TabCorre, uint2 dTer, uint2 sdTer, int nbLayer , uint2 dRVig, uint2 dImg, float espi)
{

	uint2 sv = make_uint2( dRVig.x * 2 + 1, dRVig.y * 2 + 1);

	int out_MemSize = size(dTer) * sizeof(float);
	int nBI_MemSize = size(dTer) * sizeof(int);
	int cac_MemSize = out_MemSize * nbLayer * size(sv);

	checkCudaErrors( cudaMemset( dev_Corr_Out, 0, out_MemSize ));
	checkCudaErrors( cudaMemset( dev_Cache, 0, cac_MemSize ));
	checkCudaErrors( cudaMemset( dev_NbImgOk, 0, nBI_MemSize ));

	checkCudaErrors( cudaBindTextureToArray(TexLay_Proj,dev_ProjLayered) );


	//------------   Kernel correlation   ----------------
		dim3 threads(BLOCKDIM, BLOCKDIM, 1);
		dim3 blocks(iDivUp(dTer.x,threads.x) , iDivUp(dTer.y,threads.y), nbLayer);
		
		correlationKernel<<<blocks, threads>>>( dev_Corr_Out, dev_NbImgOk, dev_Cache, dTer, sdTer, dRVig, dImg, espi);
		getLastCudaError("Basic Correlation kernel failed");
		//checkCudaErrors( cudaDeviceSynchronize() );

	//---------- Kernel multi-correlation -----------------
	{

		uint2 actiThs = SBLOCKDIM - make_uint2(SBLOCKDIM % sv.x, SBLOCKDIM % sv.y);

		dim3 threads_mC(SBLOCKDIM, SBLOCKDIM, nbLayer);
		dim3 blocks_mC(iDivUp(dTer.x * sv.x  ,actiThs.x) , iDivUp(dTer.y * sv.y ,actiThs.y));

		multiCorrelationKernel<<<blocks_mC, threads_mC>>>( dev_Corr_Out, dev_Cache, dev_NbImgOk, dTer, dRVig, dImg);
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
		int step = 3;
		std::cout << " --------------------  size ter (x,y) : " << iDivUp(dTer.x, step) << ", " << iDivUp(dTer.y, step) << std::endl;
		for (int j = 0; j < dTer.y ; j+=step)
		{
			std::cout << "       "; 
			for (int i = 0; i < dTer.x ; i+=step)
			{
				int id = (j * dTer.x  + i );
				float c = host_Corr_Out[id];
				if( c > 0.0f)
					//std::cout << floor(c*10000)/10000 << " ";
					std::cout << c << " ";
				else
					std::cout << " .  ";
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
		for (int j = 0; j < dTer.y * sv.y ; j+=step)
		{
			for (int i = 0; i < dTer.x * sv.x ; i+=step)
			{
				int id = (j * dTer.x * sv.x + i );
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