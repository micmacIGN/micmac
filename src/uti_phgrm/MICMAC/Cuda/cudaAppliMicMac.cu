#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>

texture<float, cudaTextureType2D, cudaReadModeNormalizedFloat> refTex_Image;
texture<float2, cudaTextureType2D, cudaReadModeNormalizedFloat> refTex_Project;


// ATTENTION : erreur de compilation avec l'option cudaReadModeNormalizedFloat et l'utilisation de la fonction tex2DLayered
texture<float2,	cudaTextureType2DLayered>	refTex_ProjectsLayered;
texture<float,	cudaTextureType2DLayered>	refTex_ImagesLayered;

cudaArray* dev_Img;				// Tableau des valeurs de l'image
cudaArray* dev_CubeProjImg;		// Declaration du cube de projection pour le device
cudaArray* dev_ArrayProjImg;	// Declaration du tableau de projection pour le device

cudaArray* dev_ImagesLayered;	//
cudaArray* dev_ProjLayered; //

__constant__ int dev_ListImgs[32];

extern "C" void imagesToLayers(float *fdataImg1D, int sx, int sy, int sz)
{

		cudaExtent sizeImgsLay = make_cudaExtent( sx, sy, sz );

		// Définition du format des canaux d'images
		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

		// Allocation memoire GPU du tableau des calques d'images
		checkCudaErrors( cudaMalloc3DArray(&dev_ImagesLayered,&channelDesc,sizeImgsLay,cudaArrayLayered) );

		// Déclaration des parametres de copie 3D
		cudaMemcpy3DParms p = { 0 };

		p.dstArray	= dev_ImagesLayered;		// Pointeur du tableau de destination
		p.srcPtr	= make_cudaPitchedPtr(fdataImg1D, sizeImgsLay.width * sizeof(float), sizeImgsLay.width, sizeImgsLay.height);	
		p.extent	= sizeImgsLay;				// Taille du cube
		p.kind		= cudaMemcpyHostToDevice;	// Type de copie

		// Copie des images du Host vers le Device
		checkCudaErrors( cudaMemcpy3D(&p) );

		// Lié à la texture
		checkCudaErrors( cudaBindTextureToArray(refTex_ImagesLayered,dev_ImagesLayered) );


};
#define   BLOCKDIM 32



extern "C" void  projectionsToLayers(float *h_TabProj, int sx, int sy, int sz)
{
	// Définition du format des canaux d'images
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float2>();

	// Taille du tableau des calques 
	cudaExtent siz_PL = make_cudaExtent( sx, sy, sz);

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

extern __shared__ float cacheCorrel[];

__global__ void correlationKernel(float *dest, int sx, int sy, int winX, int winY, int sXI, int sYI )
{


	// Se placer dans l'espace terrain
	const int X		= blockIdx.x * blockDim.x  + threadIdx.x;
	const int Y		= blockIdx.y * blockDim.y  + threadIdx.y;
	const int Z		= blockIdx.z * blockDim.z  + threadIdx.z;
	const int idL	= dev_ListImgs[Z];

	// Definir la zone de la fenetre/vignette
	const int x0	= X - winX;
	const int x1	= X + winX;
	const int y0	= Y - winY;
	const int y1	= Y + winY;

	// Intialisation des valeurs de calcul 
	float aSV	= 0.0f;
	float aSVV	= 0.0f;

	// nombre de pixel dans la vignette
	int dimVign	=	(winX * 2 + 1 ) * ( winY * 2 + 1 );
	
	// Decalage dans la memoire partagée de la vignette
	int piX		= dimVign * threadIdx.x;
	int piY		= dimVign * threadIdx.y * blockDim.x;
	int	piZ		= dimVign * threadIdx.z * blockDim.y *  blockDim.x;
	int pitchCo	= dimVign * ( piX + piY + piZ );

	// Balayage des points de la vignettes
	#pragma unroll
	for (int x = x0 ; x <= x1; x++)
	{
		#pragma unroll
		for (int y = y0 ; y <= y1; y++)
		{
			const float  u		= (X / (float) sx)*2.0f-1.0f;
			const float	 v		= (Y / (float) sy)*2.0f-1.0f;

			// Projection dans l'image
			const float2 pTProj	= tex2DLayered( refTex_ProjectsLayered, u, v, Z);

			// Sortir si la projection est hors de l'image
			if ((pTProj.x <0.0f)|(pTProj.y <0.0f)) return;

			// Projection dans l'image en coordonnées de la texture GPU 
			const float ui = (pTProj.x / (float) sXI)*2.0f-1.0f;
			const float vi = (pTProj.y / (float) sYI)*2.0f-1.0f;
			
			// Valeur de l'image
			float val = tex2DLayered( refTex_ImagesLayered, ui, vi, idL);

			// Calcul du décalage dans la vignette
			int pitchVi	=	x * (winX * 2 + 1 ) + y;
			int i		= pitchCo +  pitchVi;
			 
			cacheCorrel[ i ] = val; // Mis en cache de la valeur de l'image
			aSV  += val;			// Somme des valeurs de l'image cte 
			aSVV += val*val;		// Somme des carrés des vals image cte

		}
	}
	__syncthreads();

	aSV 		/=	dimVign;
	aSVV 		/=	dimVign;
	aSVV		-=	aSV * aSV;
	float saSVV	 =	sqrt(aSVV);	

	// si aSVV > mAhEpsilon

	float result = 0;

	#pragma unroll
	for (int i = 0 ; i <= dimVign; i++)
	{
		cacheCorrel[pitchCo + i] = (cacheCorrel[pitchCo + i]-aSV)/saSVV;
		result += cacheCorrel[pitchCo + i];
	}

	dest[sx * sy * Z + Y * sx + X ] = result / dimVign;

	//Si plus 1 image correcte, Calcul "rapide"  de la multi-correlation en utilisant la formule de Huygens
	// Pour chaque pixel de la vignette	: 0	< aKV	< mNbPtsWFixe
/*	
	
	#pragma unroll
	for (int i = 0 ; i <= dimVign; i++)
	{
		 
		float aSVG	= 0;
		float aSVVG	= 0;

		//Pour chaque image correcte  	: 0 < aKIm	< aNbImOk 	// maj des stat 1 et 2
		float aVt	= aVVals[aKIm][aKV];
										aSV 		+= aV;
										aSVV 		+= QSquare(aV);
								
								anEC2 += (aSVV-QSquare(aSV)/aNbImOk); // Additionner l'ecart type inter imagettes
	
	}				

*/							


};

extern "C" void correlation( float* h_TabCorre, int* listImgProj, int sx, int sy, int sz , int winX, int winY , int sXI, int sYI ){

	int mem_size = sx * sy * sz;
	
	float* host_Correl_Out = (float *) malloc(mem_size);
	float* dev_Correl_Out;

	checkCudaErrors(  cudaMalloc((void **) &dev_Correl_Out, mem_size) );

	// liste des images projetées copies de Host vers device
	checkCudaErrors( cudaMemcpyToSymbol(dev_ListImgs, listImgProj, sizeof(int)*sz));

	dim3 threads(BLOCKDIM / winX , BLOCKDIM / winY, sz);
	dim3 blocks(sx / threads.x , sy /  threads.y, sz / threads.y);

	// set texture parameters
    refTex_ProjectsLayered.filterMode		= cudaFilterModePoint;
    refTex_ProjectsLayered.normalized		= true;  // access with normalized texture coordinates

	// Lié à la texture
	checkCudaErrors( cudaBindTextureToArray(refTex_ProjectsLayered,dev_ProjLayered) );

	// Lancer la fonction de Kernel GPU pour calculer la correlation
	correlationKernel<<<blocks, threads>>>( dev_Correl_Out, sx, sy, winX, winY, sXI, sYI );

    checkCudaErrors( cudaUnbindTexture(refTex_ProjectsLayered) );
	checkCudaErrors( cudaMemcpy(host_Correl_Out, dev_Correl_Out, mem_size, cudaMemcpyDeviceToHost) );
	
	if(0)
	{
		float result = 0.0f;

		for ( int l = 0 ; l < sz; l++)
		{
			for ( int i = 0 ; i < sx; i++)
			{
				for ( int j = 0 ; j < sy; j++)
				{
					result = floor(host_Correl_Out[ l * sx * sy + i * sy + j ]/100.0f);
					if (result > 0.0f)
						std::cout <<  result << " ";
				}
				if (result != 0.0f) std::cout << std::endl;
			}

			std::cout << "---------------------------------------------------------" << std::endl;
		}
	}
	
	cudaFree(dev_Correl_Out);
	free(host_Correl_Out);
}



extern "C" void freeTexture()
{
	checkCudaErrors( cudaUnbindTexture(refTex_Image) );
	checkCudaErrors( cudaUnbindTexture(refTex_Project) );
	checkCudaErrors( cudaUnbindTexture(refTex_ImagesLayered) );
	checkCudaErrors( cudaFreeArray(dev_Img) );
	checkCudaErrors( cudaFreeArray(dev_CubeProjImg) );
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

}