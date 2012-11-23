#include <cuda_runtime.h>

texture<float, 1, cudaReadModeNormalizedFloat> refDeTex;

cudaArray* dev_Img;
// Declaration du cube de projection pour le device
cudaArray* dev_CubeProjImg;
	
// Creation des tableaux de resultats de corrélation
/*

> Cube des corrélations
	- channel : somme des valeurs, somme des valeur


*/

extern "C" void  imageToDevice(float** aDataIm,  int sXImg, int sYImg)
{
	float *dataImg1D	= new float[sXImg*sYImg];
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();


	// TACHE : changer la structuration des donnees pour le stockage des images 
	// Tableau 2D  --->> tableau linéaire
	for (int i = 0; i < sXImg ; i++)
		for (int j = 0; j < sYImg ; j++)
			dataImg1D[i*sYImg+j] = aDataIm[j][i];

	cudaError_t cudaERROR;

	// Allocation mémoire du tableau cuda
	cudaERROR = cudaMallocArray(&dev_Img,&channelDesc,sYImg,sXImg);

	// Copie des données du Host dans le tableau Cuda
	cudaERROR = cudaMemcpy2DToArray(dev_Img,0,0,dataImg1D, sYImg*sizeof(float),sYImg*sizeof(float), sXImg, cudaMemcpyHostToDevice);

	// Lier la texture au tableau Cuda
	cudaERROR = cudaBindTextureToArray(refDeTex,dev_Img);

}

extern "C" void cubeProjToDevice(float* cubeProjPIm, cudaExtent dimCube)
{


		// Variable erreur cuda
		cudaError_t cudaERROR;

		// Format des canaux 
		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindFloat);
		//cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
			
		// Taille du cube
		cudaExtent sizeCube = dimCube;
			
		// Allocation memoire GPU du cube de projection
		cudaERROR = cudaMalloc3DArray(&dev_CubeProjImg,&channelDesc,sizeCube);

		// Déclaration des parametres de copie 3D
		cudaMemcpy3DParms p = { 0 };
			
		// Pointeur du tableau de destination
		p.dstArray	= dev_CubeProjImg;
		// Pas du cube
		p.srcPtr	= make_cudaPitchedPtr(cubeProjPIm, dimCube.width * 2 * sizeof(float), dimCube.width, dimCube.height);
		// Taille du cube
		p.extent	= dimCube;
		// Type de copie
		p.kind		= cudaMemcpyHostToDevice;

		// Copie du cube de projection du Host vers le Device
		cudaERROR	= cudaMemcpy3D(&p);
		// Sortie console : Statut de la copie 3D
		
}

extern "C" void correlation(){

	


}

extern "C" void freeTexture()
{
	cudaUnbindTexture(refDeTex);
	cudaFreeArray(dev_Img);
	cudaFreeArray(dev_CubeProjImg);
}