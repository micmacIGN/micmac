#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <helper_math.h>
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


__device__  inline float2 simpleProjection( uint2 size, uint2 ssize, uint2 sizeImg ,uint2 coord, int L)
{

	const float2 cf = make_float2(ssize) * make_float2(coord) / make_float2(size) ;

	const int2	 a	= make_int2(cf); 
	const int2	 b	= a + 1; 
	const float2 uva = (make_float2(a) + 0.5f) / (make_float2(ssize));
	const float2 uvb = (make_float2(b) + 0.5f) / (make_float2(ssize));
	float2 ra, rb, Iaa, result;

	ra	= tex2DLayered( TexLay_Proj, uva.x, uva.y, L);
	rb	= tex2DLayered( TexLay_Proj, uvb.x, uva.y, L);
	if (ra.x < 0.0f || ra.y < 0.0f || rb.x < 0.0f || rb.y < 0.0f)
		return make_float2(-1.0f,-1.0f);

	Iaa	= ((float)(b.x) - cf.x) * ra + (cf.x - (float)(a.x)) * rb;
	ra	= tex2DLayered( TexLay_Proj, uva.x, uvb.y, L);
	rb	= tex2DLayered( TexLay_Proj, uvb.x, uvb.y, L);

	if (ra.x < 0.0f || ra.y < 0.0f || rb.x < 0.0f || rb.y < 0.0f)
		return make_float2(-1.0f,-1.0f);

	result	= ((float)(b.x) - cf.x) * ra + (cf.x - (float)(a.x)) * rb;
	
	result = ((float)(b.y) - cf.y) * Iaa + (cf.y - (float)(a.y)) * result;
	result = (result + 0.5f) / (make_float2(sizeImg));

	return result;
}

__global__ void correlationKernel(float* dest, int *dev_NbImgOk, float* cache, int sTer_X, int sTer_Y, int rxVig, int ryVig, int sxImg, int syImg, float mAhEpsilon )
{
	__shared__ float cacheImg[ BLOCKDIM ][ BLOCKDIM ];

	// Se placer dans l'espace terrain
	const int X	= blockIdx.x * blockDim.x  + threadIdx.x;
	const int Y	= blockIdx.y * blockDim.y  + threadIdx.y;
	const int L	= blockIdx.z;
	const int iTer = Y * sTer_X + X;

	// Si le processus est hors du terrain, nous sortons du kernel
	if ( X >= sTer_X || Y >= sTer_Y) 
		return;

	const float2 PtTProj	= simpleProjection(  make_uint2(sTer_X,sTer_Y), make_uint2(sTer_X /5 ,sTer_Y /5), make_uint2(sxImg,syImg), make_uint2(X,Y), L);
	
	if ( PtTProj.x < 0.0f ||  PtTProj.y < 0.0f )
		return;
	else
		cacheImg[threadIdx.x][threadIdx.y] = tex2DLayered( refTex_ImagesLayered, PtTProj.x, PtTProj.y,L);
	
	__syncthreads();

	// Intialisation des valeurs de calcul 
	float		aSV	= 0.0f;
	float	   aSVV	= 0.0f;
	const int	x0	= threadIdx.x - rxVig;
	const int	x1	= threadIdx.x + rxVig;
	const int	y0	= threadIdx.y - ryVig;
	const int	y1	= threadIdx.y + ryVig;

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
	const unsigned int svX	= 2 * rxVig + 1;
	const unsigned int svY	= 2 * ryVig + 1;

	// coordonnées des threads
	const unsigned int tx		= threadIdx.x;
	const unsigned int ty		= threadIdx.y;
	const unsigned int tz		= threadIdx.z;

	aSV [ty][tx]		= 0.0f;
	aSVV[ty][tx]		= 0.0f;
	resu[ty/2][tx/2]	= 0.0f;
	__syncthreads();

	// nombres de threads utilisées dans le bloques
	const unsigned int actiThs_X = blockDim.x - blockDim.x % svX;
	const unsigned int actiThs_Y = blockDim.y - blockDim.y % svY;

	// si le thread est inactif, il sort
	if ( tx >=  actiThs_X || ty >=  actiThs_Y )
		return;
	
	// taille de la vignette et du terrain
	const unsigned int size_Vign = svX * svY;
	const unsigned int size_Terr = sTer_X * sTer_Y;

	// Coordonnées 3D du cache
	const unsigned int x		= blockIdx.x * actiThs_X  + tx;
	const unsigned int y		= blockIdx.y * actiThs_Y  + ty;
	const unsigned int l		= threadIdx.z;

	// Si le thread est en dehors du cache
	if ( x >=  sTer_X * svX || y >=  sTer_Y * svY )
		return;

	// Coordonnées 1D du cache
	const unsigned int iCach	= l * size_Terr * size_Vign + y * sTer_X * svX + x ;

	// Coordonnées 2D du terrain 
	const unsigned int X		= x / svX;
	const unsigned int Y		= y / svY;

	// Coordonnées 1D dans le cache
	const unsigned int iTer	= Y * sTer_X  + X;

	// Coordonnées 2D du terrain dans le repere des threads
	const unsigned int tT_X	= tx / svX; 
	const unsigned int tT_Y	= ty / svY;

	bool mainThread = (tx % svX)== 0 && (ty % svY) == 0 && tz == 0;

	int aNbImOk = dev_NbImgOk[iTer];

	if ( aNbImOk < 2)
	{
		if (mainThread)
			dest[iTer] = -1.0f;
		return;
	}

	float val = cache[iCach];

	__syncthreads();

	atomicAdd( &aSV[ty][tx], val);

	//printf ("aSV : %d, %4.2f, val :%4.2f | ", tz ,aSV[ty][tx], val);
	
	__syncthreads();

	atomicAdd( &aSVV[ty][tx], val * val);

	__syncthreads();

	atomicAdd(&resu[tT_Y][tT_X],aSVV[ty][tx] - ( aSV[ty][tx] * aSV[ty][tx] / aNbImOk)); 

	__syncthreads();

	if ( !mainThread ) return;

	// Normalisation pour le ramener a un equivalent de 1-Correl 
	float cost = resu[tT_Y][tT_X] / (( aNbImOk-1) * size_Vign);

	dest[iTer] = 1.0f - max (-1.0, min(1.0,1.0f - cost));

}
/*

static int iDivUp(int a, int b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}*/

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
    TexLay_Proj.filterMode		= cudaFilterModeLinear; //cudaFilterModePoint 
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
		
		correlationKernel<<<blocks, threads>>>( dev_Corr_Out, dev_NbImgOk, dev_Cache, sTer_X, sTer_Y, rxVig, ryVig, sxImg, syImg, mAhEpsilon);
		getLastCudaError("Basic Correlation kernel failed");
		//checkCudaErrors( cudaDeviceSynchronize() );

	//---------- Kernel multi-correlation -----------------
	{
		int actiThs_X = SBLOCKDIM - SBLOCKDIM % svX;
		int actiThs_Y = SBLOCKDIM - SBLOCKDIM % svY;

		dim3 threads_mC(SBLOCKDIM, SBLOCKDIM, nbLayer);
		dim3 blocks_mC(iDivUp(sTer_X * svX  ,actiThs_X) , iDivUp(sTer_Y * svY ,actiThs_Y));

		multiCorrelationKernel<<<blocks_mC, threads_mC>>>( dev_Corr_Out, dev_Cache, dev_NbImgOk, sTer_X, sTer_Y, rxVig, ryVig, sxImg, syImg );
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
		std::cout << " --------------------  size ter (x,y) : " << iDivUp(sTer_X, step) << ", " << iDivUp(sTer_Y, step) << std::endl;
		for (int j = 0; j < sTer_Y ; j+=step)
		{
			std::cout << "       "; 
			for (int i = 0; i < sTer_X ; i+=step)
			{
				int id = (j * sTer_X  + i );
				float c = host_Corr_Out[id];
				if( c > 0.0f)
					std::cout << floor(c*1000)/1000 << " ";
				else if( c == -1.0f)			
					std::cout << " .  ";
				else
					std::cout << " -  ";

				//float c = host_NbImgOk[id];
				//std::cout << c << " ";
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