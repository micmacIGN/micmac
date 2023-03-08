#include "MMVII_Image2D.h"
#include "MMVII_Tpl_Images.h"
#include "MMVII_Geom2D.h"


namespace MMVII
{

const cPt3di cRGBImage::Red(255,0,0);
const cPt3di cRGBImage::Green(0,255,0);
const cPt3di cRGBImage::Blue(0,0,255);
const cPt3di cRGBImage::Yellow(255,255,0);
const cPt3di cRGBImage::Magenta(255,0,255);
const cPt3di cRGBImage::Cyan(0,255,255);
const cPt3di cRGBImage::Orange(255,128,0);
const cPt3di cRGBImage::White(255,255,255);

//template <class Type> void SetGrayPix(cRGBImage&,const cPt2di & aPix,const cDataIm2D<Type> & aIm,const double & aMul=1.0);
/// Do it for all pix; 
//template <class Type> void SetGrayPix(cRGBImage&,const cDataIm2D<Type> & aIm,const double & aMul=1.0);
//template <class Type> cRGBImage  RGBImFromGray(const cDataIm2D<Type> & aIm,const double & aMul=1.0);
//
//

void cRGBImage::AssertZ1() const
{
    MMVII_INTERNAL_ASSERT_tiny(mZoom==1,"RGBImage:: AssertZ1");
}

const cRect2& cRGBImage::BoxZ1() const { return mBoxZ1;}


typename cRGBImage::tIm1C cRGBImage::ImR() {return mImR;}
typename cRGBImage::tIm1C cRGBImage::ImG() {return mImG;}
typename cRGBImage::tIm1C cRGBImage::ImB() {return mImB;}

cRGBImage::cRGBImage(const cPt2di & aSz,int aZoom) :
    mSz1         (aSz),
    mBoxZ1       (cRect2(cPt2di(0,0),mSz1)),
    mSzz         (aSz*aZoom),
    mZoom        (aZoom),
    mRZoom       (aZoom),
    mOffsetZoom  (cPt2dr(0.5,0.5) * (mRZoom-1)),
    mImR         (mSzz),
    mImG         (mSzz),
    mImB         (mSzz)
{
}
cRGBImage::cRGBImage(const cPt2di & aSz,const cPt3di & aCoul,int aZoom) :
   cRGBImage (aSz,aZoom)
{
    for (const auto & aPix : mBoxZ1)
        SetRGBPix(aPix,aCoul);
}

cPt2dr cRGBImage::PointToRPix(const cPt2dr & aPt) const
{
    // if we want that [-0.5, 0.5[ is mapped to {-0.5,1,2 .. Zoom-0.5}
    //
    // if zoom =1  Ofst=0, so [-0.5,0.5]  correspond to Pix(0,0)
    // if zoom =3  Ofst=1, so  [-0.5,0.5]  correspon to 0,1,2

    return aPt * mRZoom + mOffsetZoom;
}

cPt2di cRGBImage::PointToPix(const cPt2dr & aPt) const
{
	return ToI(PointToRPix(aPt));
}


void cRGBImage::SetRGBPoint(const cPt2dr & aPoint,const cPt3di & aCoul)
{
       RawSetPoint(PointToPix(aPoint),aCoul.x(),aCoul.y(),aCoul.z());
}

void cRGBImage::RawSetPoint(const cPt2di & aPixZ,int aR,int aG,int aB)
{
       mImR.DIm().SetVTruncIfInside(aPixZ,aR);
       mImG.DIm().SetVTruncIfInside(aPixZ,aG);
       mImB.DIm().SetVTruncIfInside(aPixZ,aB);
}

void cRGBImage::RawSetPoint(const cPt2di & aPixZ,const cPt3di & aCoul)
{
    RawSetPoint(aPixZ,aCoul.x(),aCoul.y(),aCoul.z());
}


void cRGBImage::SetRGBPix(const cPt2di & aPix1,int aR,int aG,int aB)
{
    cRect2 aBoxZ(aPix1*mZoom,(aPix1+cPt2di(1,1))*mZoom);

    for (const auto & aPixz : aBoxZ)
    {
        RawSetPoint(aPixz,aR,aG,aB);
	    /*
       mImR.DIm().SetVTruncIfInside(aPixz,aR);
       mImG.DIm().SetVTruncIfInside(aPixz,aG);
       mImB.DIm().SetVTruncIfInside(aPixz,aB);
       */
    }
}

void cRGBImage::SetRGBPix(const cPt2di & aPix,const cPt3di & aCoul)
{
     SetRGBPix(aPix,aCoul.x(),aCoul.y(),aCoul.z());
}

cPt3di cRGBImage::GetRGBPix(const cPt2di & aPix) const
{
    AssertZ1();
    return LikeZ1_RGBPix(aPix);
}

cPt3di cRGBImage::LikeZ1_RGBPix(const cPt2di & aPix) const
{
    return cPt3di(mImR.DIm().GetV(aPix),mImG.DIm().GetV(aPix),mImB.DIm().GetV(aPix));
}


void cRGBImage::SetGrayPix(const cPt2di & aPix,int aGray)
{
     SetRGBPix(aPix,aGray,aGray,aGray);
}

    // ========  Method requiring read of images, only work when zoom==1

cPt3di cRGBImage::GetRGBPixBL(const cPt2dr & aPix) const
{  
    AssertZ1();
    return cPt3di(mImR.DIm().GetVBL(aPix),mImG.DIm().GetVBL(aPix),mImB.DIm().GetVBL(aPix));
}


bool cRGBImage::InsideBL(const cPt2dr & aPix) const
{
    AssertZ1();
    return mImR.DIm().InsideBL(aPix);
}


void cRGBImage::SetRGBPixWithAlpha(const cPt2di & aPix,const cPt3di &aCoul,const cPt3dr & aAlpha)
{
    AssertZ1();
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
    AssertZ1();
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
    for (const auto & aPix : aRGBIm.BoxZ1())
        SetGrayPix(aRGBIm,aPix,aGrayIm,aMul);
}


template <class Type> cRGBImage  RGBImFromGray(const cDataIm2D<Type> & aGrayIm,const double & aMul,int aZoom)
{
   cRGBImage aRes(aGrayIm.Sz(),aZoom);

   SetGrayPix(aRes,aGrayIm,aMul);

   return aRes;
}

    // ==================   FILE  EXPORT/EXPORT ====================
    
               //  Creation/Read from file

cRGBImage cRGBImage::FromFile(const std::string& aName,const cBox2di & aBox,int aZoom)
{
     cRGBImage aRes(aBox.Sz(),aZoom);
     aRes.Read(cDataFileIm2D::Create(aName,false),aBox.P0());

     return aRes;
}

cRGBImage cRGBImage::FromFile(const std::string& aName,int aZoom)
{
     cRect2 aRect = cDataFileIm2D::Create(aName,false);
     return FromFile(aName,aRect,aZoom);
}

void cRGBImage::Read(const cDataFileIm2D & aDFI,const cPt2di & aP0File,double aDyn,const cRect2& aArgRect)
{
    // Default value for empty box will not work her
    cRect2 aRect = aArgRect;
    if (aRect.IsEmpty())
       aRect = cRect2(cPt2di(0,0),mSz1);

    cPt2di aP0Z = aRect.P0() * mZoom;
    cRect2 aRect1Z (aP0Z,aP0Z+aRect.Sz());

    // In a first step we transfere data at the good origine (P0Z) but not
    // taking into account the zoom
    if (aDFI.NbChannel()==3)
        mImR.DIm().Read(aDFI,mImG.DIm(),mImB.DIm(),aP0File,aDyn,aRect1Z);
    else
    {
        mImR.DIm().Read(aDFI,aP0File,aDyn,aRect1Z);
	cRect2 aRectZ(aP0Z,aRect.Sz());
        RectCopyIn(mImG.DIm(),mImR.DIm(),aRect1Z);
        RectCopyIn(mImB.DIm(),mImR.DIm(),aRect1Z);
    }

     ReplicateForZoom(aRect1Z);
}


void cRGBImage::ReplicateForZoom(const cRect2 & aRect1Z)
{
    // nothing to do
    if (mZoom==1) return;

    cPt2di aP0Z = aRect1Z.P0();
    cPt2di aP0Z1 = aP0Z / mZoom;
    cPt2di aSz1 = aRect1Z.Sz();

    // Make a replication  of rectangle beging at P0Z, to avoid
    // sides-effect begin by higher value
    for (int aX=aSz1.x() -1 ; aX>=0; aX--)
    {
       for (int aY=aSz1.y() -1 ;  aY>=0 ; aY--)
       {
	   cPt2di aPLoc1(aX,aY);
	   cPt3di aCol = LikeZ1_RGBPix(aP0Z+aPLoc1);
	   SetRGBPix(aP0Z1+aPLoc1,aCol);
       }
    }
}


void cRGBImage::Read(const std::string & aName,const cPt2di & aP0,double aDyn,const cRect2& aRect) 
{
     Read(cDataFileIm2D::Create(aName,false),aP0,aDyn,aRect);
}

               //  file  create/write

void cRGBImage::ToFile(const std::string & aName)
{
    mImR.DIm().ToFile(aName,mImG.DIm(),mImB.DIm());
}

void cRGBImage::Write(const cDataFileIm2D & aDFI,const cPt2di & aP0,double aDyn,const cRect2& aRect) const
{
    AssertZ1();
    mImR.DIm().Write(aDFI,mImG.DIm(),mImB.DIm(),aP0,aDyn,aRect);
}

void cRGBImage::Write(const std::string & aName,const cPt2di & aP0,double aDyn,const cRect2& aRect) const
{
    AssertZ1();
     Write(cDataFileIm2D::Create(aName,false),aP0,aDyn,aRect);
}

void cRGBImage::DrawEllipse(const cPt3di& aCoul,const cPt2dr & aCenter,tREAL8 aGA,tREAL8 aSA,tREAL8 aTeta)
{
    cPt2dr aCenterLoc = PointToRPix(aCenter);

    std::vector<cPt2di> aVPts;
    GetPts_Ellipse(aVPts,aCenterLoc,aGA*mRZoom,aSA*mRZoom,aTeta,true);
    for (const auto & aPix : aVPts)
    {
         RawSetPoint(aPix,aCoul);
    }
}

void cRGBImage::DrawCircle(const cPt3di& aCoul,const cPt2dr & aCenter,tREAL8  aRay)
{
	DrawEllipse(aCoul,aCenter,aRay,aRay,0.0);
}

void cRGBImage:: DrawLine(const cPt2dr & aP1,const cPt2dr & aP2,const cPt3di & aCoul)
{
    std::vector<cPt2di> aVPts;
    GetPts_Line(aVPts,PointToRPix(aP1),PointToRPix(aP2));
    for (const auto & aPix : aVPts)
    {
         RawSetPoint(aPix,aCoul);
    }
}

std::vector<cPt3di>  cRGBImage::LutVisuLabRand(int aNbLab)
{
    int aNbByC = round_up(std::sqrt(aNbLab));
    std::vector<cPt2di> aV2;
    for (int aK=0 ; aK<aNbLab ; aK++)
    {
         int aR = ((aK % aNbByC) * 255) / (aNbByC-1);
         int aG = ((aK / aNbByC) * 255) / (aNbByC-1);

	 aV2.push_back(cPt2di(aR,aG));
    }
    aV2 = RandomOrder(aV2);

    std::vector<cPt3di> aRes;
    for (int aK=0 ; aK<aNbLab ; aK++)
        aRes.push_back(cPt3di(aV2.at(aK).x(),aV2.at(aK).y(),aK));

    return aRes;
}

#if (0)


#endif
template  void SetGrayPix(cRGBImage&,const cPt2di & aPix,const cDataIm2D<tREAL4> & aIm,const double &);
template  void SetGrayPix(cRGBImage&,const cDataIm2D<tREAL4> & aIm,const double & aMul);
template  cRGBImage  RGBImFromGray(const cDataIm2D<tREAL4> & aGrayIm,const double & aMul,int aZoom);
template  cRGBImage  RGBImFromGray(const cDataIm2D<tU_INT1> & aGrayIm,const double & aMul,int aZoom);
};
