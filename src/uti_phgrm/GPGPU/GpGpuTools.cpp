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

float GpGpuTools::fValue( float value,float factor )
{
	return value * factor;
}

float GpGpuTools::fValue( float2 value,float factor )
{
	return (float)value.x * factor;
}

std::string GpGpuTools::toStr( uint2 tt )
{
	stringstream sValS (stringstream::in | stringstream::out);

	sValS << "(" << tt.x << "," << tt.y << ")";

	return sValS.str();
}

void GpGpuTools::OutputInfoGpuMemory()
{
	size_t free;  
	size_t total;  
	cudaMemGetInfo(&free, &total);
	cout << "Free memory video : " << (float)free / pow(2.0f,20) << "/" << (float)total / pow(2.0f,20) << "mb" << endl;
}

std::string CGObject::Name()
{
  return _name;
}

void CGObject::SetName( std::string name )
{
	_name = name;
}

std::string CGObject::Type()
{
	return _type;
}

void CGObject::SetType( std::string type )
{
	_type = type;
}

CGObject::CGObject()
{
	SetName("NO_NAME");
	SetType("NO_TYPE");
	ClassTemplate("NO_CLASS_TEMPLATE");
}

CGObject::~CGObject()
{

}

std::string CGObject::ClassTemplate()
{
	return _classTemplate;
}

void CGObject::ClassTemplate( std::string classTemplate )
{
	_classTemplate = classTemplate;
}

std::string CGObject::Id()
{
	return " NAME : " + Name() + ", " + "TYPE : " + Type() + "/" + ClassTemplate() + "\n";
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

void struct2D::Output()
{
	std::cout << "Dimension : " << GpGpuTools::toStr(_dimension) << "\n";
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

struct2DLayered::struct2DLayered()
{
    SetDimension(make_uint2(0),0);
}

uint struct2DLayered::GetNbLayer()
{
	return _nbLayers;

}

uint struct2DLayered::GetSize()
{
	return struct2D::GetSize() * GetNbLayer();
}

void struct2DLayered::Output()
{
	struct2D::Output();
	std::cout << "Nombre de calques : " << GetNbLayer() << "\n";
}

bool  AImageCuda::bindTexture( textureReference& texRef )
{
	cudaChannelFormatDesc desc;
	bool bCha	= CData::ErrorOutput(cudaGetChannelDesc(&desc, GetCudaArray()),"Bind Texture / cudaGetChannelDesc");
	bool bBind	= CData::ErrorOutput(cudaBindTextureToArray(&texRef,GetCudaArray(),&desc),"Bind Texture / Bind");
	return bCha && bBind;
}

cudaArray* AImageCuda::GetCudaArray()
{
	return CData<cudaArray>::pData();
}

bool AImageCuda::Dealloc()
{
	cudaError_t erC = cudaSuccess;
	SubMemoryOc(GetSizeofMalloc());
	SetSizeofMalloc(0);
	if (!CData<cudaArray>::isNULL()) erC = cudaFreeArray( CData<cudaArray>::pData());
	CData<cudaArray>::dataNULL();
	return erC == cudaSuccess ? true : false;
}

bool AImageCuda::Memset( int val )
{
	std::cout << "PAS DE MEMSET POUR CUDA ARRAY" << "\n";
	return true;
}
