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

__global__ void correlationKernel(float *dest, int sxTer, int syTer, int idLayer, int sxVig, int syVig, int sxImg, int syImg, float mAhEpsilon )
{

	__shared__ float cacheImg[ BLOCKDIM ][ BLOCKDIM ];

	float noC = -1.0f;

	// Se placer dans l'espace terrain
	const int X	= blockIdx.x * blockDim.x  + threadIdx.x;
	const int Y	= blockIdx.y * blockDim.y  + threadIdx.y;

	// Si le processus est hors du terrain, nous sortons du kernel
	if ( X >= sxTer || Y >= syTer) 
	{
		return;
	}
	// Decalage dans la memoire partagée de la vignette
	int spiX = threadIdx.x ;
	int spiY = threadIdx.y ;

	float uTer = ((float)X + 0.5f) / ((float) sxTer);
	float vTer = ((float)Y + 0.5f) / ((float) syTer);

	// Calcul de toute les valeurs de l'image et mise en cache
	const float2 PtTProj	= tex2DLayered( TexLay_Proj, uTer, vTer, idLayer);

	//dest[ Y * sxTer + X ] = PtTProj.x;
	//return;

	if ( PtTProj.x < 0.0f ||  PtTProj.y < 0.0f ||  PtTProj.x > sxImg || PtTProj.y > syImg )
	{
		cacheImg[spiX][spiY] = noC;
		//return;
	}
	else
	{

		float uImg = ((float)PtTProj.x+0.5f) / (float) sxImg;
		float vImg = ((float)PtTProj.y+0.5f) / (float) syImg;
		cacheImg[spiX][spiY] = tex2DLayered( refTex_ImagesLayered, uImg, vImg, idLayer);
	
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
	{
		dest[ Y * sxTer + X ] = noC;
		//return;
	}
	else
	{
		#pragma unroll
		for (int x = x0 ; x <= x1; x++)
		{
			#pragma unroll
			for (int y = y0 ; y <= y1; y++)
			{	
				float val = cacheImg[x][y];	// Valeur de l'image
				aSV  += val;				// Somme des valeurs de l'image cte 
				aSVV += val*val;			// Somme des carrés des vals image cte
			}
		}
	}

	if(aSVV > 0 && aSV > 0)
		dest[ Y * sxTer + X ] = aSV/aSVV;
	else
		dest[ Y * sxTer + X ] = noC;
	return;
	/*
	// Je dois recuperer le nombre de vignettes OK
	

	aSV 		/=	dimVign;
	aSVV 		/=	dimVign;
	aSVV		-=	aSV * aSV;
	float saSVV	 =	sqrt(aSVV);


	if ( aSVV <= mAhEpsilon ) return;

	int cI = atomicAdd( &aNbImOk, 1);

	__syncthreads();

	#pragma unroll
	for (int i = 0 ; i <= dimVign; i++)
		cacheCorrel[pitchCo + i] = (cacheCorrel[pitchCo + i]-aSV)/saSVV;

	

	// Si plus 1 image correcte
	// Calcul "rapide"  de la multi-correlation en utilisant la formule de Huygens

	if (aNbImOk < 2 ) return;

#pragma unroll
	for (int i = 0 ; i <= dimVign; i++)
	{
		 
		float aSV = atomicAdd( &cache__aSV[i], cacheCorrel[pitchCo + i] );
		atomicAdd( &cache_aSVV[i] , aSV * aSV );

	}	
	
	int id = blockDim.x * threadIdx.y + threadIdx.x;
	anEC2[id] = threadIdx.z;

	__syncthreads();

	if ( threadIdx.z != anEC2[id])
		return;

	anEC2[id] = 0;

#pragma unroll	
	for (int i = 0 ; i <= dimVign; i++)
		anEC2[id] += (cache_aSVV[i] - cache__aSV[i] * cache__aSV[i] /aNbImOk); // Additionner l'ecart type inter imagettes
	
	// Normalisation pour le ramener a un equivalent de 1-Correl 
	float aCost			= anEC2[id] / (( aNbImOk-1) * dimVign);
	dest[ Y * sxTer + X ]	= 1 - max(-1.0,min(1.0,1-aCost));
	*/
};

static int iDivUp(int a, int b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

extern "C" void basic_Correlation_GPU( float* h_TabCorre, int sxTer, int syTer, int nbLayer , int sxVig, int syVig , int sxImg, int syImg, float mAhEpsilon ){

	int mem_size = sxTer * syTer * sizeof(float);
	float* host_Correl_Out = (float *) malloc(mem_size);
	float* dev_Correl_Out;
	checkCudaErrors( cudaMalloc((void **) &dev_Correl_Out, mem_size) );	

	// Parametres de la texture
    TexLay_Proj.addressMode[0]	= cudaAddressModeWrap;
    TexLay_Proj.addressMode[1]	= cudaAddressModeWrap;	
    TexLay_Proj.filterMode		= cudaFilterModePoint; //cudaFilterModePoint cudaFilterModeLinear
    TexLay_Proj.normalized		= true;
	checkCudaErrors( cudaBindTextureToArray(TexLay_Proj,dev_ProjLayered) );

	// KERNEL
	dim3 threads(BLOCKDIM, BLOCKDIM);
	dim3 blocks(iDivUp(sxTer,threads.x) , iDivUp(syTer,threads.y));

	correlationKernel<<<blocks, threads>>>( dev_Correl_Out, sxTer, syTer, 0, sxVig, syVig, sxImg, syImg, mAhEpsilon );
	getLastCudaError("Basic Correlation kernel failed");
	checkCudaErrors( cudaDeviceSynchronize() );


	checkCudaErrors( cudaUnbindTexture(TexLay_Proj) );
	checkCudaErrors( cudaMemcpy(host_Correl_Out, dev_Correl_Out, mem_size, cudaMemcpyDeviceToHost) );
	checkCudaErrors( cudaDeviceSynchronize() );
	if(0)
	{
		int step = 2;
		std::cout << "Taille du terrain (x,y) : " << iDivUp(sxTer, step) << ", " << iDivUp(syTer, step) << std::endl;
		for (int j = 0; j < syTer ; j+=step)
		{
			for (int i = 0; i < sxTer ; i+=step)
			{
				int id = (j * sxTer  + i );
				float c = host_Correl_Out[id];
				if (c > 0.0f) 
					std::cout << floor(c*1000)/10 << " ";
				else
					std::cout << c << " ";
					
			}
			std::cout << std::endl; 
		}
		std::cout << "---------------------------------------------------------" << std::endl;
	}
	
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