#include "GpGpu/GpGpuTools.h"

void GpGpuTools::SetParamterTexture(textureReference &textRef)
{
    textRef.addressMode[0]	= cudaAddressModeBorder;
    textRef.addressMode[1]	= cudaAddressModeBorder;
    textRef.filterMode		= cudaFilterModeLinear; //cudaFilterModePoint cudaFilterModeLinear
    textRef.normalized		= false;
}

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

const char *GpGpuTools::conca(const char *texte, int t)
{
    stringstream sValS (stringstream::in | stringstream::out);

    sValS << texte << t;

    return sValS.str().c_str();
}

void GpGpuTools::OutputInfoGpuMemory()
{
	size_t free;  
	size_t total;  
    checkCudaErrors( cudaMemGetInfo(&free, &total));
    cout << "Memoire video       : " << (float)free / pow(2.0f,20) << " / " << (float)total / pow(2.0f,20) << "Mo" << endl;
}

std::string CGObject::Name()
{
  return _name;
}

void CGObject::SetName( std::string name )
{
    _name = name;
}

void CGObject::SetName(string name, int id)
{
    SetName(GpGpuTools::conca(name.c_str(),id));
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
    return " NAME : " + Name() + ", " + "TYPE : " + Type() + "/" + ClassTemplate();
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
    std::cout << "Dimension 2D        : " << GpGpuTools::toStr(_dimension) << "\n";
}

void struct2DLayered::SetDimension( uint3 dimension )
{
    SetDimension(make_uint2(dimension),dimension.z);
}

uint3 struct2DLayered::GetDimension3D()
{
    return make_uint3(GetDimension().x,GetDimension().y,GetNbLayer());
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
    std::cout << "Nombre de calques   : " << GetNbLayer() << "\n";
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

bool AImageCuda::Memset( int val )
{
	std::cout << "PAS DE MEMSET POUR CUDA ARRAY" << "\n";
    return true;
}

bool AImageCuda::abDealloc()
{
    return (cudaFreeArray( CData<cudaArray>::pData()) == cudaSuccess) ? true : false;
}
