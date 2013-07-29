#include "GpGpu/GpGpuObject.h"

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
