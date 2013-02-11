#include "GpGpu/GpGpuTools.h"

std::string GpGpuTools::GetImagesFolder()
{

#ifdef _WIN32

	TCHAR name [ UNLEN + 1 ];
	DWORD size = UNLEN + 1;
	GetUserName( (TCHAR*)name, &size );

	std::string suname = name;
	std::string ImagesFolder = "C:\\Users\\" + suname + "\\Pictures\\";
#else
	struct passwd *pw = getpwuid(getuid());

	const char *homedir = pw->pw_dir;

	std::string ImagesFolder = std::string(homedir) + "/Images/";

#endif

	return ImagesFolder;
}


void GpGpuTools::OutputReturn( char * out )
{

#ifndef DISPLAYOUTPUT
	return;
#else
	
	std::cout << std::string(out) << "\n";

#endif

}

void GpGpuTools::OutputGpu()
{
#if (ELISE_windows)
	Sleep(10);
	std::cout << "\b\\" << std::flush;
	Sleep(10);
	std::cout << "\b|" << std::flush;
	Sleep(10);
	std::cout << "\b/" << std::flush;
	Sleep(10);
	std::cout << "\b-" << std::flush;
#endif
}

uint2 struct2D::GetDimension()
{
	return _dimension;
}

uint2 struct2D::SetDimension( uint2 dimension )
{
	_dimension = dimension;
	return _dimension;
}

uint2 struct2D::SetDimension( uint dimX,uint dimY )
{
	return SetDimension(make_uint2(dimX,dimY));
}

uint2 struct2D::SetDimension( int dimX,int dimY )
{
	return SetDimension((uint)dimX,(uint)dimY);
}

uint struct2D::GetSize()
{
	return size(_dimension);
}

void struct2DLayered::SetDimension( uint3 dimension )
{
	SetDimension(make_uint2(dimension),dimension.z);
}

void struct2DLayered::SetDimension( uint2 dimension, uint nbLayer )
{
	struct2D::SetDimension(dimension);
	SetNbLayer(nbLayer);
}

void struct2DLayered::SetNbLayer( uint nbLayer )
{
	_nbLayers = nbLayer;
}

uint struct2DLayered::GetNbLayer()
{
	return _nbLayers;

}

uint struct2DLayered::GetSize()
{
	return struct2D::GetSize() * GetNbLayer();
}

cudaArray_t* CCudaArray::GetCudaArray_t()
{

	return &_cudaArray;
}

void CCudaArray::Dealloc()
{
	if (_cudaArray !=NULL) checkCudaErrors( cudaFreeArray( _cudaArray) );
	_cudaArray = NULL;
}

cudaArray* CCudaArray::GetCudaArray()
{
	return _cudaArray;
}

CCudaArray::CCudaArray()
{
	_cudaArray = NULL;
}

CCudaArray::~CCudaArray()
{

}