#pragma once

#include "GpGpu/helper_math_extented.cuh"
#include <cuda_runtime.h>
#include <helper_math.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <sstream>     // for ostringstream
#include <string>
#include <iostream>
using namespace std;

#ifdef _WIN32
	#include <Lmcons.h>
#else
	#include <unistd.h>
	#include <sys/types.h>
	#include <pwd.h>


	#include <cmath>
	
#endif

#define DISPLAYOUTPUT

class GpGpuTools
{

public:
	
	GpGpuTools(){};
	
	~GpGpuTools(){};
	
	//					Convert array 2D to linear array
	static void			Memcpy2Dto1D(float** dataImage2D, float* dataImage1D, uint2 dimDest, uint2 dimSource);

	//					Save to file image PGM
	template <class T>
	static bool			Array1DtoImageFile(T* dataImage,const char* fileName, uint2 dimImage);

	//					Save to file image PGM
	template <class T>
	static bool			Array1DtoImageFile(T* dataImage,const char* fileName, uint2 dimImage, float factor );

	//					Retourne la dossier image de l'utilisateur
	static std::string	GetImagesFolder();

	//					Divise toutes les valeurs du tableau par un facteur
	template <class T>  
	static T* 			DivideArray(T* data, uint2 dimImage, float factor);

	//					Sortie console d'un tableau
	template <class T>	
	static void			OutputArray(T* data, uint2 dim, float offset = 1.0f, float defaut = 0.0f, float sample = 1.0f, float factor = 1.0f);

	//					Sortie console formater d'une valeur
	template <class T>
	static void			OutputValue(T value, float offset = 1.0f, float defaut = 0.0f, float factor = 1.0f);

	//					Retour chariot
	static void			OutputReturn(char * out = "");

};

template <class T>
void GpGpuTools::OutputValue( T value, float offset, float defaut, float factor)
{
#ifndef DISPLAYOUTPUT
	return;
#endif

	std::string S2	= "    ";
	std::string ES	= "";
	std::string S1	= " ";

	float outO	= (float)value*factor;
	float out	= floor(outO*offset)/offset ;

	std::string valS;
	stringstream sValS (stringstream::in | stringstream::out);

	sValS << abs(out);
	long sizeV = (long)sValS.str().length();

	if (sizeV == 5) ES = ES + "";
	else if (sizeV == 4) ES = ES + " ";
	else if (sizeV == 3) ES = ES + "  ";
	else if (sizeV == 2) ES = ES + "   ";
	else if (sizeV == 1) ES = ES + "    ";

	if (outO == 0.0f)
		std::cout << S1 << "0" << S2;
	else if (outO == defaut)
		std::cout << S1 << "!" + S2;
	else if (outO == -1000.0f)
		std::cout << S1 << "." << S2;
	else if (outO == 2*defaut)
		std::cout << S1 << "s" << S2;
	else if (outO == 3*defaut)
		std::cout << S1 << "z" << S2;
	else if (outO == 4*defaut)
		std::cout << S1 << "s" << S2;
	else if (outO == 5*defaut)
		std::cout << S1 << "v" << S2;
	else if (outO == 6*defaut)
		std::cout << S1 << "e" << S2;
	else if (outO == 7*defaut)
		std::cout << S1 << "c" << S2;
	else if (outO == 8*defaut)
		std::cout << S1 << "?" << S2;
	else if (outO == 9*defaut)
		std::cout << S1 << "¤" << S2;
	else if ( outO < 0.0f)
		std::cout << out << ES;				
	else 
		std::cout << S1 << out << ES;

}

template <class T>
void GpGpuTools::OutputArray( T* data, uint2 dim, float offset, float defaut, float sample, float factor )
{

#ifndef DISPLAYOUTPUT
	return;
#endif

	uint2 p;

	for (p.y = 0 ; p.y < dim.y; p.y+= (int)sample)
	{
		for (p.x = 0; p.x < dim.x ; p.x+= (int)sample)

			OutputValue(data[to1D(p,dim)],offset,defaut,factor);

		std::cout << "\n";	
	}
	std::cout << "------------------------------------------\n";
}	


template <class T>
T* GpGpuTools::DivideArray( T* data, uint2 dim, float factor )
{
	if (factor == 0) return NULL;

	int sizeData = size(dim);

	T* image = new T[sizeData];

	for (int i = 0; i < sizeData ; i++)
		image[i] = data[i] * (T)factor;

	return image;

}

template <class T>
bool GpGpuTools::Array1DtoImageFile( T* dataImage,const char* fileName, uint2 dimImage )
{
	std::string pathfileImage = std::string(GetImagesFolder()) + std::string(fileName);

	std::cout << pathfileImage << "\n";
	return sdkSavePGM<T>(pathfileImage.c_str(), dataImage, dimImage.x,dimImage.y);
}

template <class T>
bool GpGpuTools::Array1DtoImageFile(T* dataImage,const char* fileName, uint2 dimImage, float factor)
{
	T* image = DivideArray(dataImage, dimImage, factor);

	bool r = Array1DtoImageFile( image, fileName, dimImage );

	delete[] image;

	return r;
}

template <class T> 
class ImageCuda
{

public:

	ImageCuda();
	ImageCuda(textureReference textRef);
	~ImageCuda();
	
	void		InitImage(uint2 dimension, T* data);
	uint2		GetDimension();
	uint2		SetDimension(uint2 dimension);
	uint2		SetDimension(int dimX,int dimY);
	uint2		SetDimension(uint dimX,uint dimY);
	void		SetTexRef(textureReference texRef);
	cudaArray*	GetCudaArray();
	void		AllocMemory();
	void		DeallocMemory();
	void		BindTexture();
	void		copyHostToDevice(T* data);

private:

	uint2				_dimension;
	cudaArray*			_cudaArray;
	

};

template <class T>
void ImageCuda<T>::DeallocMemory()
{
	if (_cudaArray !=NULL) checkCudaErrors( cudaFreeArray( _cudaArray) );
	_cudaArray = NULL;
}

template <class T>
ImageCuda<T>::~ImageCuda()
{
	//DeallocMemory();
}

template <class T>
void ImageCuda<T>::copyHostToDevice( T* data )
{
	// Copie des données du Host dans le tableau Cuda
	//checkCudaErrors( cudaMemcpy2DToArray(_cudaArray,0,0,data, _dimension.y*sizeof(T),_dimension.y*sizeof(T), _dimension.x, cudaMemcpyHostToDevice) );
	checkCudaErrors(cudaMemcpyToArray(_cudaArray, 0, 0, data, sizeof(T)*size(_dimension), cudaMemcpyHostToDevice));
	//checkCudaErrors(cudaMemcpyToArray(array, 0, 0, data, Bpp*sizeof(Pixel)*iw*ih, cudaMemcpyHostToDevice));
}

template <class T>
cudaArray* ImageCuda<T>::GetCudaArray()
{

	return _cudaArray;

}
template <class T>
ImageCuda<T>::ImageCuda()
{
	_cudaArray = NULL;
}

template <class T>
ImageCuda<T>::ImageCuda( textureReference textRef )
{
	SetTexRef(textRef);
}

template <class T>
void ImageCuda<T>::InitImage(uint2 dimension, T* data)
{
	SetDimension(dimension);
	AllocMemory();
	copyHostToDevice(data);
}

template <class T>
uint2 ImageCuda<T>::GetDimension()
{
	return _dimension;

}
template <class T>
uint2 ImageCuda<T>::SetDimension( uint2 dimension )
{
	_dimension = dimension;
	return _dimension;
}
template <class T>
uint2 ImageCuda<T>::SetDimension( uint dimX,uint dimY )
{
	return SetDimension(make_uint2(dimX,dimY));
}
template <class T>
uint2 ImageCuda<T>::SetDimension( int dimX,int dimY )
{
	return SetDimension((uint)dimX,(uint)dimY);
}

template <class T>
void ImageCuda<T>::SetTexRef( textureReference texRef )
{
	//_textureRef = texRef;
}

template <class T>
void ImageCuda<T>::AllocMemory()
{
	
	cudaChannelFormatDesc channelDesc =  cudaCreateChannelDesc<T>() ;
	// Allocation mémoire du tableau cuda
	checkCudaErrors( cudaMallocArray(&_cudaArray,&channelDesc,_dimension.x,_dimension.y) );

}



