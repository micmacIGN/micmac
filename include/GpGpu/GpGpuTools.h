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

typedef unsigned char pixel;

class GpGpuTools
{

public:
	
	GpGpuTools(){};
	
	~GpGpuTools(){};
	
	//					Convertir array 2D en tableau linéaire
	template <class T>
	static void			Memcpy2Dto1D(T** dataImage2D, T* dataImage1D, uint2 dimDest, uint2 dimSource);

	//					Sauvegarder tableau de valeur dans un fichier PGN
	template <class T>
	static bool			Array1DtoImageFile(T* dataImage,const char* fileName, uint2 dimImage);

	//					Sauvegarder tableau de valeur (multiplier par un facteur) dans un fichier PGN
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

	//
	static void			OutputGpu();

};

template <class T>
void GpGpuTools::Memcpy2Dto1D( T** dataImage2D, T* dataImage1D, uint2 dimDest, uint2 dimSource )
{
	for (uint j = 0; j < dimSource.y ; j++)
		memcpy(  dataImage1D + dimDest.x * j , dataImage2D[j],  dimSource.x * sizeof(T));		
}

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

//-----------------------------------------------------------------------------------------------
//									CLASS IMAGE CUDA
//-----------------------------------------------------------------------------------------------

class struct2D
{
public:

	struct2D(){};
	~struct2D(){};
	uint2		GetDimension();
	uint2		SetDimension(uint2 dimension);
	uint2		SetDimension(int dimX,int dimY);
	uint2		SetDimension(uint dimX,uint dimY);

private:

	uint2		_dimension;

};

class struct2DLayered : public struct2D
{

public:

	struct2DLayered(){};
	~struct2DLayered(){};
	uint	GetNbLayer();
	void	SetNbLayer(uint nbLayer);
	void	SetDimension(uint2 dimension, uint nbLayer);
	void	SetDimension(uint3 dimension);

private:

	uint _nbLayers;
};

class CCudaArray
{

public:

	CCudaArray();
	~CCudaArray();
	cudaArray*	GetCudaArray();
	cudaArray_t*GetCudaArray_t();
	void		DeallocMemory();

private:

	cudaArray*	_cudaArray;
	
};

template <class T> 
class CuData3D : public struct2DLayered
{

public:

	CuData3D();
	~CuData3D(){};
	virtual void	Malloc() = 0;
	virtual void	Memset(int val) = 0;
	virtual void	Dealloc() = 0;
	void			dataNULL();
	uint			SizeByt();
	T*				pData();
	void			Realloc(uint2 dim, uint l);
	bool			isNULL();
	T**				ppData();
	
private:

	T*	_data;
};

template <class T>
T** CuData3D<T>::ppData()
{
	return &_data;
}

template <class T>
CuData3D<T>::CuData3D()
{
	dataNULL();
}

template <class T>
bool CuData3D<T>::isNULL()
{
	return (_data == NULL);
}

template <class T>
void CuData3D<T>::Realloc( uint2 dim, uint l )
{
	Dealloc();
	SetDimension(dim,l);
	Malloc();
}

template <class T>
void CuData3D<T>::dataNULL()
{
	_data = NULL;
}

template <class T>
T* CuData3D<T>::pData()
{
	return _data;
}

template <class T>
uint CuData3D<T>::SizeByt()
{
	return struct2DLayered::GetNbLayer()*size(struct2D::GetDimension())* sizeof(T);
}

template <class T> 
class CuHostData3D : public CuData3D<T>
{
public:
	CuHostData3D(){};
	~CuHostData3D(){};
	void Dealloc();
	void Malloc();
	void Memset(int val);
	
};

template <class T>
void CuHostData3D<T>::Memset( int val )
{
	memset(pData(),val,SizeByt());
}

template <class T>
void CuHostData3D<T>::Malloc()
{
	cudaMallocHost(ppData(),SizeByt());
}

template <class T>
void CuHostData3D<T>::Dealloc()
{
	cudaFreeHost(pData());
}

template <class T> 
class CuDeviceData3D : public CuData3D<T> 
{

public:

	CuDeviceData3D();
	~CuDeviceData3D(){};
	void Dealloc();
	void Malloc();
	void Memset(int val);
	void CopyDevicetoHost(T* hostData);

};

template <class T>
void CuDeviceData3D<T>::CopyDevicetoHost( T* hostData )
{
	
	checkCudaErrors( cudaMemcpy( hostData, pData(), SizeByt(), cudaMemcpyDeviceToHost) );
}

template <class T>
void CuDeviceData3D<T>::Memset( int val )
{

	checkCudaErrors( cudaMemset( pData(), val, SizeByt()));
}

template <class T>
CuDeviceData3D<T>::CuDeviceData3D()
{
	dataNULL();
}

template <class T>
void CuDeviceData3D<T>::Malloc()
{
	checkCudaErrors( cudaMalloc((void **)ppData(), SizeByt()));
}

template <class T>
void CuDeviceData3D<T>::Dealloc()
{
	if (isNULL()) checkCudaErrors( cudaFree(pData()));
	dataNULL();
}

template <class T> 
class ImageCuda : public CCudaArray, public struct2D
{

public:

	ImageCuda();
	~ImageCuda();
	
	void		InitImage(uint2 dimension, T* data);
	void		AllocMemory();
	void		copyHostToDevice(T* data);

};


template <class T>
ImageCuda<T>::~ImageCuda()
{
	//DeallocMemory();
}

template <class T>
void ImageCuda<T>::copyHostToDevice( T* data )
{
	// Copie des données du Host dans le tableau Cuda
	checkCudaErrors(cudaMemcpyToArray(CCudaArray::GetCudaArray(), 0, 0, data, sizeof(T)*size(GetDimension()), cudaMemcpyHostToDevice));
}


template <class T>
ImageCuda<T>::ImageCuda()
{

}

template <class T>
void ImageCuda<T>::InitImage(uint2 dimension, T* data)
{
	SetDimension(dimension);
	AllocMemory();
	copyHostToDevice(data);
}

template <class T>
void ImageCuda<T>::AllocMemory()
{
	cudaChannelFormatDesc channelDesc =  cudaCreateChannelDesc<T>() ;
	// Allocation mémoire du tableau cuda
	checkCudaErrors( cudaMallocArray(CCudaArray::GetCudaArray_t(),&channelDesc,GetDimension().x,GetDimension().y) );
}

//-----------------------------------------------------------------------------------------------
//									CLASS IMAGE LAYARED CUDA
//-----------------------------------------------------------------------------------------------

template <class T> 
class ImageLayeredCuda : public struct2DLayered, public CCudaArray 
{
public:
	ImageLayeredCuda(){};
	~ImageLayeredCuda(){};

	void	AllocMemory();
	void	copyHostToDevice(T* data);

};

template <class T>
void ImageLayeredCuda<T>::copyHostToDevice( T* data )
{

	cudaExtent sizeImagesLayared = make_cudaExtent( struct2D::GetDimension().x, struct2D::GetDimension().y, struct2DLayered::GetNbLayer());

	// Déclaration des parametres de copie 3D
	cudaMemcpy3DParms	p		= { 0 };
	cudaPitchedPtr		pitch	= make_cudaPitchedPtr(data, sizeImagesLayared.width * sizeof(T), sizeImagesLayared.width, sizeImagesLayared.height);

	p.dstArray	= CCudaArray::GetCudaArray();	// Pointeur du tableau de destination
	p.srcPtr	= pitch;						// Pitch
	p.extent	= sizeImagesLayared;			// Taille du cube
	p.kind		= cudaMemcpyHostToDevice;		// Type de copie

	// Copie des images du Host vers le Device
	checkCudaErrors( cudaMemcpy3D(&p) );

}

template <class T>
void ImageLayeredCuda<T>::AllocMemory()
{
	cudaExtent sizeImagesLayared = make_cudaExtent( struct2D::GetDimension().x, struct2D::GetDimension().y,struct2DLayered::GetNbLayer());

	// Définition du format des canaux d'images	
	cudaChannelFormatDesc channelDesc =	cudaCreateChannelDesc<T>();

	// Allocation memoire GPU du tableau des calques d'images
	checkCudaErrors( cudaMalloc3DArray(CCudaArray::GetCudaArray_t(),&channelDesc,sizeImagesLayared,cudaArrayLayered) );

}





