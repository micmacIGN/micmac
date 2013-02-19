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
#define TexFloat2Layered texture<float2,cudaTextureType2DLayered>

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
	uint		GetSize();

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
	uint	GetSize();

private:

	uint _nbLayers;
};

template <class T> 
class CData 
{

public:
	CData();
	~CData(){};
	virtual void	Malloc()		= 0;
	virtual void	Memset(int val) = 0;
	virtual void	Dealloc()		= 0;

	void	dataNULL();
	bool	isNULL();
	T*		pData();
	T**		ppData();
	uint	GetSizeofMalloc();
	void	SetSizeofMalloc(uint sizeofmalloc);
private:

	
	T*		_data;
	uint	_sizeofMalloc;

};

template <class T>
void CData<T>::SetSizeofMalloc( uint sizeofmalloc )
{
	_sizeofMalloc = sizeofmalloc;
}

template <class T>
uint CData<T>::GetSizeofMalloc()
{
	return _sizeofMalloc;
}

template <class T>
CData<T>::CData()
{
	dataNULL();
}

template <class T>
T** CData<T>::ppData()
{
	return &_data;
}
template <class T>
bool CData<T>::isNULL()
{
	return (_data == NULL);
}

template <class T>
void CData<T>::dataNULL()
{
	_data = NULL;
}

template <class T>
T* CData<T>::pData()
{
	return _data;
}

template <class T> 
class CData2D : public struct2D, public CData<T>
{
public:
	CData2D(){};
	~CData2D(){};
	virtual void	Malloc() = 0;
	virtual void	Memset(int val) = 0;
	virtual void	Dealloc() = 0;

	void			Malloc(uint2 dim);
	void			Realloc(uint2 dim);
	uint			Sizeof();
};

template <class T>
uint CData2D<T>::Sizeof()
{
	return sizeof(T) * struct2D::GetSize();
}

template <class T>
void CData2D<T>::Realloc( uint2 dim )
{
	Dealloc();
	Malloc(dim);
}

template <class T>
void CData2D<T>::Malloc( uint2 dim )
{
	SetDimension(dim);
	Malloc();
}

template <class T> 
class CData3D : public struct2DLayered, public CData<T>
{

public:

	CData3D(){};
	~CData3D(){};
	virtual void	Malloc() = 0;
	virtual void	Memset(int val) = 0;
	virtual void	Dealloc() = 0;
	
	void			Malloc(uint2 dim, uint l);
	void			Realloc(uint2 dim, uint l);
 	uint			Sizeof();

};

template <class T>
void CData3D<T>::Malloc( uint2 dim, uint l )
{
	SetDimension(dim,l);
	Malloc();
}

template <class T>
void CData3D<T>::Realloc( uint2 dim, uint l )
{
	Dealloc();
	Malloc(dim,l);
}

template <class T>
uint CData3D<T>::Sizeof()
{
	return GetSize() * sizeof(T);
}

template <class T> 
class CuHostData3D : public CData3D<T>
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
	if (CData<T>::GetSizeofMalloc() < CData3D<T>::Sizeof())
		std::cout << "Allocation trop petite !!!" << "\n";
	memset(CData3D<T>::pData(),val,CData3D<T>::Sizeof());
}

template <class T>
void CuHostData3D<T>::Malloc()
{
	SetSizeofMalloc(CData3D<T>::Sizeof());
	cudaMallocHost(CData3D<T>::ppData(),CData3D<T>::Sizeof());
}

template <class T>
void CuHostData3D<T>::Dealloc()
{
	cudaFreeHost(CData3D<T>::pData());
}

template <class T> 
class CuDeviceData3D : public CData3D<T> 
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
	
	checkCudaErrors( cudaMemcpy( hostData, CData3D<T>::pData(), CData3D<T>::Sizeof(), cudaMemcpyDeviceToHost) );
}

template <class T>
void CuDeviceData3D<T>::Memset( int val )
{
	if (CData<T>::GetSizeofMalloc() < CData3D<T>::Sizeof())
		std::cout << "Allocation trop petite !!!" << "\n";

	cudaError_t cuER = cudaMemset( CData3D<T>::pData(), val, CData3D<T>::Sizeof());

	if (cuER != cudaSuccess)
	{
		checkCudaErrors( cuER );
		std::cout << "Pointeur de donnees : " << CData3D<T>::pData() << "\n";
		std::cout << "Taille des donnees  : " << CData3D<T>::Sizeof() << "\n";
	}
}

template <class T>
CuDeviceData3D<T>::CuDeviceData3D()
{
  CData3D<T>::dataNULL();
}

template <class T>
void CuDeviceData3D<T>::Malloc()
{
	SetSizeofMalloc(CData3D<T>::Sizeof());
	checkCudaErrors( cudaMalloc((void **)CData3D<T>::ppData(), CData3D<T>::Sizeof()));
}

template <class T>
void CuDeviceData3D<T>::Dealloc()
{
	if (CData3D<T>::isNULL()) checkCudaErrors( cudaFree(CData3D<T>::pData()));
	CData3D<T>::dataNULL();
}

class AImageCuda : public CData<cudaArray>
{
public:
	AImageCuda(){};
	~AImageCuda(){};
	void		bindTexture(textureReference& texRef);
	cudaArray*	GetCudaArray();
	void		Dealloc();
	void		Memset(int val);

};

template <class T> 
class ImageCuda : public CData2D<cudaArray>, public AImageCuda
{

public:

	ImageCuda(){};
	~ImageCuda(){};
	
	void	InitImage(uint2 dimension, T* data);
	void	Malloc();
	void	copyHostToDevice(T* data);
	void	Memset(int val){AImageCuda::Memset(val);};
	void	Dealloc(){AImageCuda::Dealloc();};

};

template <class T>
void ImageCuda<T>::copyHostToDevice( T* data )
{
	// Copie des données du Host dans le tableau Cuda
	checkCudaErrors(cudaMemcpyToArray(AImageCuda::pData(), 0, 0, data, sizeof(T)*size(GetDimension()), cudaMemcpyHostToDevice));
}

template <class T>
void ImageCuda<T>::InitImage(uint2 dimension, T* data)
{
	SetDimension(dimension);
	Malloc();
	copyHostToDevice(data);
}

template <class T>
void ImageCuda<T>::Malloc()
{
	cudaChannelFormatDesc channelDesc =  cudaCreateChannelDesc<T>() ;
	// Allocation mémoire du tableau cuda
	checkCudaErrors( cudaMallocArray(AImageCuda::ppData(),&channelDesc,GetDimension().x,GetDimension().y) );
}

//-----------------------------------------------------------------------------------------------
//									CLASS IMAGE LAYARED CUDA
//-----------------------------------------------------------------------------------------------

template <class T> 
class ImageLayeredCuda : public CData3D<cudaArray>, public AImageCuda
{

public:

	ImageLayeredCuda(){};
	~ImageLayeredCuda(){};
	void	Malloc();
	void	Memset(int val){AImageCuda::Memset(val);};
	void	Dealloc(){AImageCuda::Dealloc();};
	void	copyHostToDevice(T* data);

};

template <class T>
void ImageLayeredCuda<T>::copyHostToDevice( T* data )
{
	cudaExtent sizeImagesLayared = make_cudaExtent( CData3D::GetDimension().x, CData3D::GetDimension().y, CData3D::GetNbLayer());

	// Déclaration des parametres de copie 3D
	cudaMemcpy3DParms	p		= { 0 };
	cudaPitchedPtr		pitch	= make_cudaPitchedPtr(data, sizeImagesLayared.width * sizeof(T), sizeImagesLayared.width, sizeImagesLayared.height);

	p.dstArray	= AImageCuda::GetCudaArray();	// Pointeur du tableau de destination
	p.srcPtr	= pitch;					// Pitch
	p.extent	= sizeImagesLayared;		// Taille du cube
	p.kind		= cudaMemcpyHostToDevice;	// Type de copie

	// Copie des images du Host vers le Device
	checkCudaErrors( cudaMemcpy3D(&p) );
}

template <class T>
void ImageLayeredCuda<T>::Malloc()
{

	cudaExtent sizeImagesLayared = make_cudaExtent( CData3D::GetDimension().x, CData3D::GetDimension().y,CData3D::GetNbLayer());

	// Définition du format des canaux d'images	
	cudaChannelFormatDesc channelDesc =	cudaCreateChannelDesc<T>();

	// Allocation memoire GPU du tableau des calques d'images
	checkCudaErrors( cudaMalloc3DArray(AImageCuda::ppData(),&channelDesc,sizeImagesLayared,cudaArrayLayered) );

}
