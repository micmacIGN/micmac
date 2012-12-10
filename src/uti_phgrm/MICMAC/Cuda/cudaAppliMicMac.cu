#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>

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

extern "C" void imagesToLayers(float *fdataImg1D, int sxImg, int syImg, int nbLayer)
{
	cudaExtent sizeImgsLay = make_cudaExtent( sxImg, syImg, nbLayer );

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

	if(0)
		for (int x = 0 ; x <= sxImg * syImg; x++)
			std::cout << fdataImg1D[x] << " \n";

	
};

extern "C" void  projectionsToLayers(float *h_TabProj, int sTer_X, int sTer_Y, int nbLayer)
{
	// Définition du format des canaux d'images
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float2>();

	// Taille du tableau des calques 
	cudaExtent siz_PL = make_cudaExtent( sTer_X, sTer_Y, nbLayer);

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

__global__ void correlationKernel(int *dev_NbImgOk, float* cache, int sTer_X, int sTer_Y, int rxVig, int ryVig, int sxImg, int syImg, float mAhEpsilon )
{
	__shared__ float cacheImg[ BLOCKDIM ][ BLOCKDIM ];

	// coordonnées des threads
	const unsigned short tx		= threadIdx.x;
	const unsigned short ty		= threadIdx.y;

	// Se placer dans l'espace terrain
	const int X	= blockIdx.x * blockDim.x  + tx;
	const int Y	= blockIdx.y * blockDim.y  + ty;
	const int L	= blockIdx.z;
	const int iTer = Y * sTer_X + X;

	// Si le processus est hors du terrain, nous sortons du kernel
	if ( X >= sTer_X || Y >= sTer_Y) 
		return;

	// Decalage dans la memoire partagée de la vignette
	int spiX = threadIdx.x ;
	int spiY = threadIdx.y ;

	float uTer = ((float)X + 0.5f) / ((float) sTer_X);
	float vTer = ((float)Y + 0.5f) / ((float) sTer_Y);

	// Les coordonnées de projections dans l'image
	const float2 PtTProj	= tex2DLayered( TexLay_Proj, uTer, vTer, L);

	if ( PtTProj.x < 0.0f ||  PtTProj.y < 0.0f ||  PtTProj.x > sxImg || PtTProj.y > syImg )
	{
	
		return;
	}
	else
	{
		float uImg = (PtTProj.x+0.5f) / (float) sxImg;
		float vImg = (PtTProj.y+0.5f) / (float) syImg;
		cacheImg[spiX][spiY] = tex2DLayered( refTex_ImagesLayered, uImg, vImg,L);
	}

	__syncthreads();

	// Intialisation des valeurs de calcul 
	float		aSV	= 0.0f;
	float	   aSVV	= 0.0f;
	const int	x0	= spiX - rxVig;
	const int	x1	= spiX + rxVig;
	const int	y0	= spiY - ryVig;
	const int	y1	= spiY + ryVig;

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

	int siCaX	 = 2 * rxVig + 1;
	int siCaY	 = 2 * ryVig + 1 ;
	int size_Vign	 = siCaX * siCaY;
	aSV 		/=	size_Vign;
	aSVV 		/=	size_Vign;
	aSVV		-=	aSV * aSV;
	
	if ( aSVV <= mAhEpsilon ) return;

	int iCach = ( iTer + L * sTer_X * sTer_Y) * size_Vign; 

	aSVV =	sqrt(aSVV);

	#pragma unroll
	for (int y = y0 ; y <= y1; y++)
	{
		int pitchV = siCaX *  ( y - y0); 
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
__global__ void multiCorrelationKernel(float *dest, float* cache, int * dev_NbImgOk, int sTer_X, int sTer_Y, int rxVig, int ryVig, int sxImg, int syImg)
{

	__shared__ float aSV [ SBLOCKDIM ][ SBLOCKDIM ];
	__shared__ float aSVV[ SBLOCKDIM ][ SBLOCKDIM ];
	__shared__ float resu[ SBLOCKDIM/2 ][ SBLOCKDIM/2 ];

	// dimensions des vignettes
	const unsigned short svX	= 2 * rxVig + 1;
	const unsigned short svY	= 2 * ryVig + 1;

	// coordonnées des threads
	const unsigned short tx		= threadIdx.x;
	const unsigned short ty		= threadIdx.y;
/*

	aSV [ty][tx]		= 0.0f;
	aSVV[ty][tx]		= 0.0f;
	resu[ty/2][tx/2]	= 0.0f;
	__syncthreads();*/

	// nombres de threads utilisées dans le bloques
	const unsigned short actiThs_X = blockDim.x - blockDim.x % svX;
	const unsigned short actiThs_Y = blockDim.y - blockDim.y % svY;

	// taille de la vignette et du terrain
	const unsigned short size_Vign = svX * svY;
	const unsigned short size_Terr = sTer_X * sTer_Y;

	// si le thread est inactif, il sort
	if ( tx >=  actiThs_X || ty >=  actiThs_Y )
		return;

	// Coordonnées 3D du cache
	const unsigned short x		= blockIdx.x * actiThs_X  + tx;
	const unsigned short y		= blockIdx.y * actiThs_Y  + ty;
	const unsigned short l		= threadIdx.y;

	// Si le thread est en dehors du terrain
	if ( x >=  sTer_X * svX || y >=  sTer_Y * svY )
		return;

	// Coordonnées 1D du cache
	const unsigned short iCach	= l * size_Terr * size_Vign + y * sTer_X * svX + x ;

	// Coordonnées 2D du terrain 
	const unsigned short X		= x / svX;
	const unsigned short Y		= y / svY;

	// Coordonnées 1D dans le cache
	const unsigned short iTer	= Y * sTer_X  + X;

	// Coordonnées 2D du terrain dans le repere des threads
	const unsigned short tT_X	= tx / svX; 
	const unsigned short tT_Y	= ty / svY;

	bool mainThread = (tx % svX)== 0 && (ty % svY) == 0;

	int aNbImOk = dev_NbImgOk[iTer];

	if ( aNbImOk < 2)
	{
		if (mainThread)
			dest[iTer] = -1.0f;
		return; 
	}

	float val = cache[iCach];

	if ( mainThread ) 
		resu[tT_Y][tT_X] = 0.0f;

	__syncthreads();


	atomicAdd( &aSV[ty][tx], val);

	__syncthreads();

	atomicAdd( &aSVV[ty][tx], val * val);

	__syncthreads();

	atomicAdd(&resu[tT_Y][tT_X],aSVV[ty][tx] - ( aSV[ty][tx] * aSV[ty][tx] / aNbImOk)); 

	__syncthreads();

	if ( !mainThread ) return;

	__syncthreads();

	// Normalisation pour le ramener a un equivalent de 1-Correl 
	float cost = resu[tT_Y][tT_X] / (( aNbImOk-1) * size_Vign);

	float aCorrel = 1.0f - cost;
	aCorrel = max (-1.0, min(1.0,aCorrel));
	
	float temp = 1.0f - aCorrel;

	dest[iTer] = 1.0f - aCorrel;
}

static int iDivUp(int a, int b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
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
	host_Cache		= (float*)	malloc(cac_MemSize);
	host_NbImgOk	= (int*)	malloc(nBI_MemSize);

	checkCudaErrors( cudaMalloc((void **) &dev_Corr_Out, out_MemSize) );	
	checkCudaErrors( cudaMalloc((void **) &dev_Cache, cac_MemSize ) );
	checkCudaErrors( cudaMalloc((void **) &dev_NbImgOk, nBI_MemSize ) );

	// Texture des projections
	TexLay_Proj.addressMode[0]	= cudaAddressModeWrap;
    TexLay_Proj.addressMode[1]	= cudaAddressModeWrap;	
    TexLay_Proj.filterMode		= cudaFilterModePoint;
    TexLay_Proj.normalized		= true;
}

extern "C" void basic_Correlation_GPU( float* h_TabCorre, int sTer_X, int sTer_Y, int nbLayer , int rxVig, int ryVig , int sxImg, int syImg, float mAhEpsilon ){

	int svX = ( rxVig * 2 + 1 );
	int svY = ( ryVig * 2 + 1 );
	int out_MemSize = sTer_X * sTer_Y * sizeof(float);
	int nBI_MemSize = sTer_X * sTer_Y * sizeof(int);
	int cac_MemSize = out_MemSize * nbLayer * svX * svY;

	checkCudaErrors( cudaMemset( dev_Corr_Out, 0, out_MemSize ));
	checkCudaErrors( cudaMemset( dev_Cache, 0, cac_MemSize ));
	checkCudaErrors( cudaMemset( dev_NbImgOk, 0, nBI_MemSize ));

	checkCudaErrors( cudaBindTextureToArray(TexLay_Proj,dev_ProjLayered) );


	//------------   Kernel correlation   ----------------
		dim3 threads(BLOCKDIM, BLOCKDIM, 1);
		dim3 blocks(iDivUp(sTer_X,threads.x) , iDivUp(sTer_Y,threads.y), nbLayer);
		
		correlationKernel<<<blocks, threads>>>( dev_NbImgOk, dev_Cache, sTer_X, sTer_Y, rxVig, ryVig, sxImg, syImg, mAhEpsilon);
		getLastCudaError("Basic Correlation kernel failed");
		//checkCudaErrors( cudaDeviceSynchronize() );

	//---------- Kernel multi-correlation -----------------
	{
		int actiThs_X = SBLOCKDIM - SBLOCKDIM % svX;
		int actiThs_Y = SBLOCKDIM - SBLOCKDIM % svY;

		dim3 threads_mC(SBLOCKDIM, SBLOCKDIM, nbLayer);
		dim3 blocks_mC(iDivUp(sTer_X * svX  ,actiThs_X) , iDivUp(sTer_Y * svY ,actiThs_Y));

		//multiCorrelationKernel<<<blocks_mC, threads_mC>>>( dev_Corr_Out, dev_Cache, dev_NbImgOk, sTer_X, sTer_Y, rxVig, ryVig, sxImg, syImg );
		//getLastCudaError("Multi-Correlation kernel failed");
	}

	//checkCudaErrors( cudaDeviceSynchronize() );
	checkCudaErrors( cudaUnbindTexture(TexLay_Proj) );
	checkCudaErrors( cudaMemcpy( host_Corr_Out,	dev_Corr_Out, out_MemSize, cudaMemcpyDeviceToHost) );
	//checkCudaErrors( cudaMemcpy( host_NbImgOk,	dev_NbImgOk,  nBI_MemSize, cudaMemcpyDeviceToHost) );
	checkCudaErrors( cudaMemcpy( host_Cache,	dev_Cache,	  cac_MemSize, cudaMemcpyDeviceToHost) );
	//--------------------------------------------------------

	//if(0)
	{
		int step = 1;
		std::cout << "Taille du terrain (x,y) : " << iDivUp(sTer_X, step) << ", " << iDivUp(sTer_Y, step) << std::endl;
		for (int j = 0; j < sTer_Y ; j+=step)
		{
			for (int i = 0; i < sTer_X ; i+=step)
			{
				int id = (j * sTer_X  + i );
				float c = host_Corr_Out[id];
				//std::cout << floor(c*10)/10 << " ";
				//float c = host_NbImgOk[id];
				std::cout << c << " ";
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
		for (int j = 0; j < sTer_Y * svY ; j+=step)
		{
			for (int i = 0; i < sTer_X * svX ; i+=step)
			{
				int id = (j * sTer_X * svX + i );
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
	free(host_NbImgOk); 
	free(host_Cache);
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