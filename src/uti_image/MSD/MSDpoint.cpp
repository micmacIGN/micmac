#include "StdAfx.h"
#include "MSDpoint.h"



MSDPoint::MSDPoint(Pt2df P, float size, float angle, float response)
{
    m_Point.x=P.x;
    m_Point.y=P.y;
    m_size=size;
    addAngle(angle);
    m_response=response;
}

MSDPoint::MSDPoint(float x, float y, float size, float angle, float response)
{
    m_Point.x=x;
    m_Point.y=y;
    m_size=size;
    addAngle(angle);
    m_response=response;
}

MSDPoint::MSDPoint(){
    addAngle(0.0);
}

MSDPoint::~MSDPoint(){}

void MSDPoint::undist(CamStenope * aCalib)
{
    Pt2dr ptUndist=aCalib->DistDirecte(Pt2dr(m_Point.x,m_Point.y));
    m_Point=Pt2df(ptUndist.x,ptUndist.y);
}


// cast a MSD point to a Digeo Point

//comparaison scale = aPt.getScale()ou scale = aPt.getSize(): more tiepoint with getSize than getScale but precense of a proportion of outlier.
// with getScale: less tie point, but no outlier!!

// there are generally more than one angle for the orientation of MSD points, which is a bad sign
// compute one descriptor per angle
template <class tData, class tComp>
DigeoPoint ToDigeo(MSDPoint & aPt,DescriptorExtractor<tData, tComp> & aDesc){
    DigeoPoint DP;

    for (auto & angle : aPt.getAngles()){
      // 0.0 angle is default constructor of MSD pt
      if (angle!=0.0){
    REAL8 scale(aPt.getScale());
    DP.x=aPt.getPoint().x;
    DP.y=aPt.getPoint().y;
    REAL8  descriptor[DIGEO_DESCRIPTOR_SIZE];
    aDesc.describe(DP.x,DP.y,scale,angle,descriptor);
    aDesc.normalize_and_truncate(descriptor);
    DP.addDescriptor(angle,descriptor);
        }
    }

    return DP;
}

// Cast a vector of msd point into a vector of DigeoPoint
template <class tData, class tComp>
std::vector<DigeoPoint> ToDigeo(std::vector<MSDPoint> & aVMSD,Im2D<tData, tComp> & Image){

    std::vector<DigeoPoint> aVDigeo;
    DescriptorExtractor<tData, tComp> Desc=DescriptorExtractor<tData, tComp>(Image);
    for (auto & MSD: aVMSD){
     DigeoPoint DP=ToDigeo(MSD,Desc);
     aVDigeo.push_back(DP);
    }
    return aVDigeo;
}

template DigeoPoint ToDigeo(MSDPoint & aPt,DescriptorExtractor<U_INT1,INT> & aDesc);
template DigeoPoint ToDigeo(MSDPoint & aPt,DescriptorExtractor<U_INT2,INT> & aDesc);
template DigeoPoint ToDigeo(MSDPoint & aPt,DescriptorExtractor<REAL4,REAL8> & aDesc);
template std::vector<DigeoPoint> ToDigeo(std::vector<MSDPoint> & aVMSD,Im2D<U_INT1,INT> & Image);
template std::vector<DigeoPoint> ToDigeo(std::vector<MSDPoint> & aVMSD,Im2D<U_INT2,INT> & Image);
template std::vector<DigeoPoint> ToDigeo(std::vector<MSDPoint> & aVMSD,Im2D<REAL4,REAL8> & Image);
