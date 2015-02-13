#include "GpGpu/GpGpu_Object.h"

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
	if(name.size())
        SetName(GpGpuTools::conca(name.c_str(),id));
    else
        SetName("NO_NAME");
}

std::string CGObject::Type()
{
    return _type;
}

void CGObject::SetType(string type )
{
    _type = type;
}

CGObject::CGObject()
{
    SetName("NO_NAME");
    SetType("NO_TYPE");
    ClassTemplate("NO_CLASS_TEMPLATE");
}

CGObject::~CGObject(){}

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
#ifdef NOCUDA_X11
	return " NAME : " + Name() + ", " + "TYPE : " + Type();
#else
	return " NAME : " + Name() + ", " + "TYPE : " + Type() + "<" + ClassTemplate() + ">";
#endif
}

struct2D::struct2D():
    _m_maxsize(0),
    _m_maxdimension(make_uint2(0,0))
{}

uint2 struct2D::GetDimension()
{
    return _dimension;
}

void struct2D::SetMaxSize(uint size)
{
    _m_maxsize = size;
}

uint2 struct2D::GetMaxDimension()
{
    return _m_maxdimension;
}

void struct2D::SetMaxDimension(uint2 dim)
{
    _m_maxdimension = dim;
}

void struct2D::RefreshMaxSize()
{
    uint size = GetSize();

    if(_m_maxsize < size) SetMaxSize(size);
}

void struct2D::RefreshMaxDim()
{
    if(oI(_m_maxdimension,GetDimension()))
        _m_maxdimension = GetDimension();
}

uint2 struct2D::SetDimension( uint2 dimension )
{
    _dimension = dimension;
    RefreshMaxSize();
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
    std::cout << "Structure 2D : \n";
    std::cout << "Dimension 2D        : " << GpGpuTools::toStr(_dimension) << "\n";
}

uint2 struct2D::SetDimensionOnly(uint2 dimension)
{
    _dimension = dimension;
    return _dimension;
}

uint struct2D::GetMaxSize()
{
    return _m_maxsize;
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
    struct2D::SetDimensionOnly(dimension);
    SetNbLayer(nbLayer);
    struct2DLayered::RefreshMaxSize();
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

void struct2DLayered::RefreshMaxSize()
{
    uint size = struct2DLayered::GetSize();
    if(struct2D::GetMaxSize() < size) struct2D::SetMaxSize(size);
    struct2D::RefreshMaxDim();
}


