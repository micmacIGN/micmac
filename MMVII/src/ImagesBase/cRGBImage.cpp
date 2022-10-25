
#include "MMVII_Image2D.h"

namespace MMVII
{

const cPt3di cRGBImage::Red(255,0,0);
const cPt3di cRGBImage::Green(0,255,0);
const cPt3di cRGBImage::Blue(0,0,255);
const cPt3di cRGBImage::Yellow(255,255,0);
const cPt3di cRGBImage::Magenta(255,0,255);
const cPt3di cRGBImage::Cyan(0,255,255);

//template <class Type> void SetGrayPix(cRGBImage&,const cPt2di & aPix,const cDataIm2D<Type> & aIm,const double & aMul=1.0);
/// Do it for all pix; 
//template <class Type> void SetGrayPix(cRGBImage&,const cDataIm2D<Type> & aIm,const double & aMul=1.0);
//template <class Type> cRGBImage  RGBImFromGray(const cDataIm2D<Type> & aIm,const double & aMul=1.0);

typename cRGBImage::tIm1C cRGBImage::ImR() {return mImR;}
typename cRGBImage::tIm1C cRGBImage::ImG() {return mImG;}
typename cRGBImage::tIm1C cRGBImage::ImB() {return mImB;}

cRGBImage::cRGBImage(const cPt2di & aSz) :
    mImR (aSz),
    mImG (aSz),
    mImB (aSz)
{
}
cRGBImage::cRGBImage(const cPt2di & aSz,const cPt3di & aCoul) :
   cRGBImage (aSz)
{
    for (const auto & aPix : mImR.DIm())
        SetRGBPix(aPix,aCoul);
}


void cRGBImage::SetRGBPix(const cPt2di & aPix,int aR,int aG,int aB)
{
    mImR.DIm().SetVTruncIfInside(aPix,aR);
    mImG.DIm().SetVTruncIfInside(aPix,aG);
    mImB.DIm().SetVTruncIfInside(aPix,aB);
}

void cRGBImage::SetRGBPix(const cPt2di & aPix,const cPt3di & aCoul)
{
     SetRGBPix(aPix,aCoul.x(),aCoul.y(),aCoul.z());
}

cPt3di cRGBImage::GetRGBPix(const cPt2di & aPix) const
{
    return cPt3di(mImR.DIm().GetV(aPix),mImG.DIm().GetV(aPix),mImB.DIm().GetV(aPix));
}

void cRGBImage::SetGrayPix(const cPt2di & aPix,int aGray)
{
     SetRGBPix(aPix,aGray,aGray,aGray);
}

void cRGBImage::ToFile(const std::string & aName)
{
    mImR.DIm().ToFile(aName,mImG.DIm(),mImB.DIm());
}

void cRGBImage::SetRGBPixWithAlpha(const cPt2di & aPix,const cPt3di &aCoul,const cPt3dr & aAlpha)
{
      cPt3di aCurC = GetRGBPix(aPix); 

      cPt3di aMix
             (
                 round_ni(aCurC.x()*aAlpha.x() + aCoul.x()*(1.0-aAlpha.x())),
                 round_ni(aCurC.y()*aAlpha.y() + aCoul.y()*(1.0-aAlpha.y())),
                 round_ni(aCurC.z()*aAlpha.z() + aCoul.z()*(1.0-aAlpha.z()))
             );
      SetRGBPix(aPix,aMix);
}

void cRGBImage::SetRGBrectWithAlpha(const cPt2di & aC,int aSzW,const cPt3di & aCoul,const double & aAlpha)
{
    for (const auto & aPix  :  cRect2::BoxWindow(aC,aSzW))
        SetRGBPixWithAlpha(aPix,aCoul,cPt3dr(aAlpha,aAlpha,aAlpha));
}

    ///  ===========  Manipulation from gray images ========================

template <class Type> void SetGrayPix(cRGBImage& aRGBIm,const cPt2di & aPix,const cDataIm2D<Type> & aGrayIm,const double & aMul)
{
    aRGBIm.SetGrayPix(aPix,round_ni(aMul*aGrayIm.GetV(aPix)));
}

template <class Type> void SetGrayPix(cRGBImage& aRGBIm,const cDataIm2D<Type> & aGrayIm,const double & aMul)
{
    for (const auto & aPix : aRGBIm.ImR().DIm())
        SetGrayPix(aRGBIm,aPix,aGrayIm,aMul);
}


template <class Type> cRGBImage  RGBImFromGray(const cDataIm2D<Type> & aGrayIm,const double & aMul)
{
   cRGBImage aRes(aGrayIm.Sz());

   SetGrayPix(aRes,aGrayIm,aMul);

   return aRes;
}

template  void SetGrayPix(cRGBImage&,const cPt2di & aPix,const cDataIm2D<tREAL4> & aIm,const double &);
template  void SetGrayPix(cRGBImage&,const cDataIm2D<tREAL4> & aIm,const double & aMul=1.0);
template  cRGBImage  RGBImFromGray(const cDataIm2D<tREAL4> & aGrayIm,const double & aMul =1.0);
};
