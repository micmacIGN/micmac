#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#define   BLOCKDIM 32


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
//

texture<float2,	cudaTextureType2DLayered > TexLay_Proj;
texture<float,	cudaTextureType2DLayered > refTex_ImagesLayered;
cudaArray* dev_ImagesLayered;	//
cudaArray* dev_ProjLayered;		//

//------------------------------------------------------------------------------------------

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

extern "C" void  projectionsToLayers(float *h_TabProj, int sxTer, int syTer, int nbLayer)
{
	// Définition du format des canaux d'images
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float2>();

	// Taille du tableau des calques 
	cudaExtent siz_PL = make_cudaExtent( sxTer, syTer, nbLayer);

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

__global__ void correlationKernel(float *dest, float* cache, int sxTer, int syTer, int sxVig, int syVig, int sxImg, int syImg, float mAhEpsilon )
{
	__shared__ float cacheImg[ BLOCKDIM ][ BLOCKDIM ];

	// Se placer dans l'espace terrain
	const int X	= blockIdx.x * blockDim.x  + threadIdx.x;
	const int Y	= blockIdx.y * blockDim.y  + threadIdx.y;
	const int L	= blockIdx.z;
	const int iTer = Y * sxTer + X;

	// Si le processus est hors du terrain, nous sortons du kernel
	if ( X >= sxTer || Y >= syTer) 
		return;

	// Decalage dans la memoire partagée de la vignette
	int spiX = threadIdx.x ;
	int spiY = threadIdx.y ;

	float uTer = ((float)X + 0.5f) / ((float) sxTer);
	float vTer = ((float)Y + 0.5f) / ((float) syTer);

	// Les coordonnées de projections dans l'image
	const float2 PtTProj	= tex2DLayered( TexLay_Proj, uTer, vTer, L);

	if ( PtTProj.x < 0.0f ||  PtTProj.y < 0.0f ||  PtTProj.x > sxImg || PtTProj.y > syImg )
	{
		return;
	}
	else
	{
		float uImg = ((float)PtTProj.x+0.5f) / (float) sxImg;
		float vImg = ((float)PtTProj.y+0.5f) / (float) syImg;
		cacheImg[spiX][spiY] = tex2DLayered( refTex_ImagesLayered, uImg, vImg,L);
	}
	__syncthreads();

	// Intialisation des valeurs de calcul 
	float		aSV	= 0.0f;
	float	   aSVV	= 0.0f;
	const int	x0	= spiX - sxVig;
	const int	x1	= spiX + sxVig;
	const int	y0	= spiY - syVig;
	const int	y1	= spiY + syVig;

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

	int siCaX	 = 2 * sxVig + 1;
	int siCaY	 = 2 * syVig + 1 ;
	int dimVign	 = siCaX * siCaY;
	aSV 		/=	dimVign;
	aSVV 		/=	dimVign;
	aSVV		-=	aSV * aSV;
	
	if ( aSVV <= mAhEpsilon ) return;

	// Nombre d'images correctes
	atomicAdd( &dest[iTer], 1);

	__syncthreads();

	int aNbImOk = dest[iTer];
	dest[iTer]  = 0.0f;

	if (aNbImOk < 2) return;
	
	int iCTer = ( iTer + L * sxTer * syTer) * dimVign; 

	aSVV =	sqrt(aSVV);

	#pragma unroll
	for (int y = y0 ; y <= y1; y++)
	{
		int pitchV = siCaX *  ( y - y0); 
		#pragma unroll
		for (int x = x0 ; x <= x1; x++)	
			cache[iCTer + pitchV  +  x - x0] = (cacheImg[y][x] -aSV)/aSVV;
	}

	// ---------------------------------------------------------------------------
	// Calcul "rapide"  de la multi-correlation en utilisant la formule de Huygens
	// ---------------------------------------------------------------------------

	int size = 6;

	int iSV		= ( iTer + (size - 2 ) * sxTer * syTer) * dimVign;
	int iSVV	= ( iTer + (size - 1 ) * sxTer * syTer) * dimVign;

	#pragma unroll
	for (int v = 0 ; v < dimVign; v++)
	{
			float aSV = atomicAdd( &cache[iSV + v], cache[iCTer + v] );
			atomicAdd( &cache[iSVV + v] , aSV * aSV );
	}

	float result = 0.0f;

	#pragma unroll
	for (int v = 0 ; v < dimVign; v++)		
			result += (cache[iSVV + v] - ( cache[iSV + v] * cache[iSV + v] / aNbImOk));

	// Normalisation pour le ramener a un equivalent de 1-Correl 
	float cost = result / (( aNbImOk-1) * dimVign);



	float aCorrel = 1.0f - cost;
	aCorrel = max (-1.0, min(1.0,aCorrel));
	dest[iTer] = 1.0f - aCorrel;

};

static int iDivUp(int a, int b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

extern "C" void basic_Correlation_GPU( float* h_TabCorre, int sxTer, int syTer, int nbLayer , int sxVig, int syVig , int sxImg, int syImg, float mAhEpsilon ){

	int out_MemSize = sxTer * syTer * sizeof(float);
	int cac_MemSize = out_MemSize * ( nbLayer + 2 ) * ( sxVig * 2 + 1 ) * ( sxVig * 2 + 1 );

	float* host_Correl_Out;
	float* dev_Correl_Out;
	float* dev_Cache;
	
	// Allocation mémoire
	host_Correl_Out = (float *) malloc(out_MemSize);
	
	checkCudaErrors( cudaMalloc((void **) &dev_Correl_Out, out_MemSize) );	
	checkCudaErrors( cudaMalloc((void **) &dev_Cache, cac_MemSize ) );

	checkCudaErrors( cudaMemset( dev_Correl_Out, 0, out_MemSize ));
	checkCudaErrors( cudaMemset( dev_Cache, 0, cac_MemSize ));

	// Parametres de la texture
    
	TexLay_Proj.addressMode[0]	= cudaAddressModeWrap;
    TexLay_Proj.addressMode[1]	= cudaAddressModeWrap;	
    TexLay_Proj.filterMode		= cudaFilterModePoint;
    TexLay_Proj.normalized		= true;

	checkCudaErrors( cudaBindTextureToArray(TexLay_Proj,dev_ProjLayered) );

	// KERNEL
	dim3 threads(BLOCKDIM, BLOCKDIM, 1);
	dim3 blocks(iDivUp(sxTer,threads.x) , iDivUp(syTer,threads.y), nbLayer);

	correlationKernel<<<blocks, threads>>>( dev_Correl_Out, dev_Cache, sxTer, syTer, sxVig, syVig, sxImg, syImg, mAhEpsilon );
	getLastCudaError("Basic Correlation kernel failed");
	checkCudaErrors( cudaDeviceSynchronize() );

	checkCudaErrors( cudaUnbindTexture(TexLay_Proj) );
	checkCudaErrors( cudaMemcpy(host_Correl_Out, dev_Correl_Out, out_MemSize, cudaMemcpyDeviceToHost) );
	
	//if(0)
	{
		int step = 2;
		std::cout << "Taille du terrain (x,y) : " << iDivUp(sxTer, step) << ", " << iDivUp(syTer, step) << std::endl;
		for (int j = 0; j < syTer ; j+=step)
		{
			for (int i = 0; i < sxTer ; i+=step)
			{
				int id = (j * sxTer  + i );
				float c = host_Correl_Out[id];
				std::cout << floor(c*10)/10 << " ";
			}
			std::cout << std::endl; 
		}
		std::cout << "---------------------------------------------------------" << std::endl;
	}
	
	cudaFree(dev_Cache);
	cudaFree(dev_Correl_Out);
	free(host_Correl_Out);
}

extern "C" void freeImagesTexture()
{
	checkCudaErrors( cudaUnbindTexture(refTex_Image) );
	checkCudaErrors( cudaUnbindTexture(refTex_ImagesLayered) );
	checkCudaErrors( cudaFreeArray(dev_Img) );
	checkCudaErrors( cudaFreeArray(dev_ImagesLayered) );
}

extern "C" void freeProjections()
{
	checkCudaErrors( cudaUnbindTexture(refTex_Project) );
	checkCudaErrors( cudaUnbindTexture(TexLay_Proj) );
	checkCudaErrors( cudaFreeArray(dev_CubeProjImg) );
	checkCudaErrors( cudaFreeArray(dev_ArrayProjImg) );
	checkCudaErrors( cudaFreeArray(dev_ProjLayered) );
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