#include "MMVII_Sensor.h"
#include "MMVII_2Include_Serial_Tpl.h"


/**
   \file SensorBases.cpp

   \brief base classes used in all sensors
*/


namespace MMVII
{

/* ******************************************************* */
/*                                                         */
/*                    cPixelDomain                         */
/*                    cDataPixelDomain                     */
/*                                                         */
/* ******************************************************* */

      //  ============ cDataPixelDomain  ================

cDataPixelDomain::cDataPixelDomain(const cPt2di &aSz) :
     mSz  (aSz)
{
}

const cPt2di & cDataPixelDomain::Sz() const {return mSz;}

void cDataPixelDomain::AddData(const cAuxAr2007 & anAux)
{
    MMVII::AddData(cAuxAr2007("NbPix",anAux),mSz);
}

      //  ============ cPixelDomain  ================

cPixelDomain::cPixelDomain(cDataPixelDomain * aDPD) :
     cDataBoundedSet<tREAL8,2>(cBox2dr(cPt2dr(0,0),ToR(aDPD->Sz()))),
     mDPD  (aDPD)
{
}

const cPt2di & cPixelDomain::Sz() const {return mDPD->Sz();}

tREAL8 cPixelDomain::DegreeVisibility(const cPt2dr & aP) const
{
   cBox2dr aBox(cPt2dr(0,0),ToR(Sz()));
   return aBox.Insideness(aP);
}


/* ******************************************************* */
/*                                                         */
/*                   cSensorImage                          */
/*                                                         */
/* ******************************************************* */

cSensorImage::cSensorImage(const std::string & aNameImage)  :
     mNameImage               (aNameImage),
     mEqColinearity           (nullptr),
     mEqCIsInit               (false)
{
}

cSensorImage::~cSensorImage()
{
}


const cPt3dr * cSensorImage::CenterOfPC() const  {return nullptr;} // By default, we are not a central perpective
								    //
cCalculator<double> * cSensorImage::SetAndGetEqColinearity(bool WithDerives,int aSzBuf,bool ReUse)
{
    if (! mEqCIsInit)
    {
       MMVII_INTERNAL_ASSERT_tiny(WithDerives,"SetAndGetEqColinearity  w/o derivate to implement");
       mEqCIsInit = true;
       mEqColinearity  = CreateEqColinearity(WithDerives,aSzBuf,ReUse);
    }

    return mEqColinearity;
}

cCalculator<double> * cSensorImage::GetEqColinearity()
{
    MMVII_INTERNAL_ASSERT_tiny(mEqCIsInit,"GetEqColinearity Eq not init");

    return mEqColinearity;
}



void cSensorImage::SetNameImage(const std::string & aNameImage)
{
     mNameImage = aNameImage;
}

const std::string & cSensorImage::NameImage() const {return mNameImage;}

double  cSensorImage::SqResidual(const cPair2D3D & aPair) const
{
     return SqN2(aPair.mP2-Ground2Image(aPair.mP3));
}

double  cSensorImage::AvgSqResidual(const cSet2D3D & aSet) const
{
     double aSum = 0;
     for (const auto & aPair : aSet.Pairs() )
     {
         aSum +=  SqResidual(aPair);
     }
     return std::sqrt(aSum/aSet.Pairs().size());
}

double cSensorImage::DegreeVisibilityOnImFrame(const cPt2dr & aP) const 
{
     return PixelDomain().DegreeVisibility(aP);
}

std::vector<cPt2dr>  cSensorImage::PtsSampledOnSensor(int aNbByDim,tREAL8 aEpsMarginRel)  const 
{
    std::vector<cPt2dr> aRes;
    tREAL8 aEps =  aNbByDim * aEpsMarginRel;

    for (int aKx=0 ; aKx<=aNbByDim ; aKx++)
    {
        for (int aKy=0 ; aKy<=aNbByDim ; aKy++)
	{
            aRes.push_back(  MulCByC(ToR(Sz()) , cPt2dr(aKx+aEps,aKy+aEps)/tREAL8(aNbByDim+2*aEps)) );
	}
    }

    return aRes;
}


     // method that by default generate errors, 

cPt3dr cSensorImage::Ground2ImageAndDepth(const cPt3dr &) const
{
    MMVII_INTERNAL_ERROR("No cSensorImage::Ground2ImageAndDepth");
    return cPt3dr::Dummy();
}

cPt3dr cSensorImage::ImageAndDepth2Ground(const cPt3dr &) const
{
    MMVII_INTERNAL_ERROR("No cSensorImage::ImageAndDepth2Ground");
    return cPt3dr::Dummy();
}

cCalculator<double> * cSensorImage::CreateEqColinearity(bool WithDerives,int aSzBuf,bool ReUse)
{
    MMVII_INTERNAL_ERROR("cSensorImage::CreateEqColinearity not implemanted");
    return nullptr;
}

void cSensorImage::PutUknowsInSetInterval()
{
    MMVII_INTERNAL_ERROR("cSensorImage::PutUknowsInSetInterval not implemanted");
}

void cSensorImage::PushOwnObsColinearity( std::vector<double> &,const cPt3dr&) 
{
    MMVII_INTERNAL_ERROR("cSensorImage::PushOwnObsColinearity not implemanted");
}

void cSensorImage::ToFile(const std::string &) const 
{
    MMVII_INTERNAL_ERROR("cSensorImage::ToFile not implemanted");
}


cPt2dr cSensorImage::GetIntervalZ() const
{
    MMVII_INTERNAL_ERROR("cSensorImage::GetIntervalZ not implemanted");
    return cPt2dr::Dummy();
}



/*
double cSensorImage::RobustAvResidualOfProp(const cSet2D3D &,double aProp) const
{
    cWeightAv<tREAL8>  aWAvg;
    for (const auto & aPair : aSet.Pairs() )
    {
         tREAL8 aResidual = std::sqrt(SqResidual(aPair));
         tREAL8 aWeight = aSigma/(aSigma+aResidual);
	 aWAvg.Add(aWeight,aResidual);
    }
    return aWAvg.Average();
}
*/



std::string cSensorImage::PrefixName() { return "Ori"; }



std::string  cSensorImage::NameOri_From_PrefixAndImage(const std::string & aPrefix,const std::string & aNameImage)
{ 
    return PrefixName() + "-" + aPrefix + "-" + aNameImage + "." + GlobTaggedNameDefSerial(); 
}
std::string cSensorImage::NameOriStd() const 
{ 
    std::string aRes =   NameOri_From_PrefixAndImage(V_PrefixName(),mNameImage);
    return  aRes;
}


cPt3dr cSensorImage::ImageAndDepth2Ground(const cPt2dr & aP2,const double & aDepth) const 
{
    return ImageAndDepth2Ground(cPt3dr(aP2.x(),aP2.y(),aDepth));
}

bool   cSensorImage::HasImageAndDepth() const {return false;}
bool   cSensorImage::HasIntervalZ() const {return false;}

cPt3dr  cSensorImage::EpsDiffGround2Im(const cPt3dr &) const 
{
    MMVII_INTERNAL_ERROR("EspDiffGround2Im has not been defined for sensor class : " + V_PrefixName());
    return cPt3dr::Dummy();
}

tProjImAndGrad  cSensorImage::DiffG2IByFiniteDiff(const cPt3dr & aPt) const
{
     tProjImAndGrad aRes;
     aRes.mPIJ = Ground2Image(aPt);

     cPt3dr aEpsXYZ = EpsDiffGround2Im(aPt);

     for (size_t aKCoord=0 ; aKCoord<3 ; aKCoord++)
     {
          tREAL8 aEps = aEpsXYZ[aKCoord];
	  cPt3dr aPPlus =  aPt + cPt3dr::P1Coord(aKCoord,aEps);
	  cPt3dr aPMinus = aPt + cPt3dr::P1Coord(aKCoord,-aEps);

	  cPt2dr aGradK = (Ground2Image(aPPlus)-Ground2Image(aPMinus)) / (2*aEps);

	  aRes.mGradI[aKCoord] = aGradK.x();
	  aRes.mGradJ[aKCoord] = aGradK.y();
     }
     return aRes;
}

tProjImAndGrad  cSensorImage::DiffGround2Im(const cPt3dr & aPt) const
{
	return DiffG2IByFiniteDiff(aPt);
}





const cPt2di & cSensorImage::Sz() const {return PixelDomain().Sz();}

tPt2dr cSensorImage::RandomVisiblePIm() const 
{
      bool IsOk=false;
      tPt2dr aRes;
      while (! IsOk)
      {
           aRes = MulCByC(  tPt2dr::PRand()  , ToR(Sz())  );
	   IsOk = IsVisibleOnImFrame(aRes) ;
      }
       
      return aRes;
}

tPt2dr  cSensorImage::RelativePosition(const tPt2dr & aPt) const
{
   return DivCByC(aPt,ToR(Sz()));
}


tPt3dr cSensorImage::RandomVisiblePGround(const cSensorImage & other,int aNbTestMax,bool * isOk ) const
{

    if (isOk!=nullptr ) *isOk= false;

    for (int aKTest=0 ; aKTest< aNbTestMax ; aKTest++)
    {
       tPt2dr aPIm1 = this->RandomVisiblePIm();
       tPt2dr aPIm2 = other.RandomVisiblePIm();

       tPt3dr aResult = PInterBundle(cHomogCpleIm(aPIm1,aPIm2),other);

// StdOut() << aPIm1 << aPIm2 << aResult << aResult << std::endl;
       if ( this->IsVisible(aResult)  && other.IsVisible(aResult))
       {
           if (isOk!=nullptr) *isOk= true;
	   return aResult;
       }
    }

    if (isOk==nullptr )
    {
        MMVII_INTERNAL_ERROR("Cannot compute RandomVisiblePGround");
    }
    return tPt3dr(0.0,0.0,0.0);
}

cHomogCpleIm cSensorImage::RandomVisibleCple(const cSensorImage & other,int aNbTestMax,bool * isOk) const
{
    tPt3dr aPGr = RandomVisiblePGround(other,aNbTestMax,isOk);
    return cHomogCpleIm(this->Ground2Image(aPGr),other.Ground2Image(aPGr));
}


cPt3dr cSensorImage::RandomVisiblePGround(tREAL8 aDepMin,tREAL8 aDepMax)
{
     cPt2dr aPIm   = RandomVisiblePIm();
     tREAL8 aDepth = RandInInterval(aDepMin,aDepMax);

     // MPD : big bug, but never catched as it was random simul ...
     // return  Ground2ImageAndDepth(cPt3dr(aPIm.x(),aPIm.y(),aDepth));
     return  ImageAndDepth2Ground(cPt3dr(aPIm.x(),aPIm.y(),aDepth));
}



cSet2D3D  cSensorImage::SyntheticsCorresp3D2D (int aNbByDim,std::vector<double> & aVecDepth,bool IsDepthOrZ,tREAL8 aEpsMarginRel) const
{
    cSet2D3D aResult;

    std::vector<cPt2dr>  aVPts =  PtsSampledOnSensor(aNbByDim,aEpsMarginRel);

    for (const auto & aPIm : aVPts)
    {
        for (const auto & aDepth : aVecDepth)
        {
             cPt3dr aP = IsDepthOrZ                                           ? 
		           ImageAndDepth2Ground(aPIm,aDepth)                  :
		           ImageAndZ2Ground(cPt3dr(aPIm.x(),aPIm.y(),aDepth)) ;
	     aResult.AddPair(aPIm,aP);
	}
    }

    return aResult;
}
         ///  call variant with vector, depth regularly spaced
cSet2D3D  cSensorImage::SyntheticsCorresp3D2D (int aNbByDim,int aNbDepts,double aD0,double aD1,bool IsDepthOrZ,tREAL8 aEpsMarginRel) const
{
   std::vector<tREAL8> aVDepth;


   for (int aKD=0 ; aKD < aNbDepts; aKD++)
   {
        tREAL8 aW = (aKD+aEpsMarginRel) / (aNbDepts-1+2*aEpsMarginRel);
        if (IsDepthOrZ)
	{
	     //  Case depth we make some log regular spacing
            aVDepth.push_back(aD0 * pow(aD1/aD0,aW));
	}
	else
	{
	     //  Case z we make basic regular spacing
             aVDepth.push_back(  (aD0* (1-aW)) + aD1 * aW);
	}
   }

   return SyntheticsCorresp3D2D(aNbByDim,aVDepth,IsDepthOrZ,aEpsMarginRel);
}

bool cSensorImage::IsVisible(const cPt3dr & aP3) const  { return DegreeVisibility(aP3) > 0; }
bool cSensorImage::IsVisibleOnImFrame(const cPt2dr & aP2) const  { return DegreeVisibilityOnImFrame(aP2) > 0;}
bool cSensorImage:: PairIsVisible(const cPair2D3D & aPair) const
{
	return IsVisible(aPair.mP3) && IsVisibleOnImFrame(aPair.mP2) ;
}

cPt3dr cSensorImage::Image2PlaneInter(const cPlane3D & aPlane,const cPt2dr & aPIm) const
{
    return aPlane.Inter(Image2Bundle(aPIm));
}

cPt2dr cSensorImage::Image2LocalPlaneInter(const cPlane3D & aPlane,const cPt2dr &aPIm) const
{
  return Proj(aPlane.ToLocCoord(Image2PlaneInter(aPlane,aPIm)));
}

cEllipse cSensorImage::EllipseIm2Plane(const cPlane3D & aPlane,const cEllipse & anElIm,int aNbTeta) const
{
   cEllipse_Estimate aEEst(Image2LocalPlaneInter(aPlane,anElIm.Center()));
   for (int aKTeta =0 ; aKTeta < aNbTeta ; aKTeta++)
   {
       cPt2dr  aPIm    = anElIm.PtOfTeta(aKTeta* ((2*M_PI)/aNbTeta));  // pt on image ellipse
       cPt2dr  aPPlane = Image2LocalPlaneInter(aPlane,aPIm);

       aEEst.AddPt(aPPlane);
    }
    return aEEst.Compute() ;
}

const cPt3dr *  cSensorImage::CenterOfFootPrint() const
{
   return nullptr;
}

tREAL8 cSensorImage::PixResInterBundle(const cHomogCpleIm & aCple,const cSensorImage & other) const
{
   cPt3dr aP3d = PInterBundle(aCple,other);

   return (Norm2(Ground2Image(aP3d)-aCple.mP1)+Norm2(other.Ground2Image(aP3d)-aCple.mP2)) / 2.0;
}

cPt3dr cSensorImage::PInterBundle(const cHomogCpleIm & aCple,const cSensorImage & other) const
{
     tSeg3dr aSeg1 = this->Image2Bundle(aCple.mP1);
     tSeg3dr aSeg2 = other.Image2Bundle(aCple.mP2);

     return BundleInters(aSeg1,aSeg2);
}

cPt3dr cSensorImage::Ground2ImageAndZ(const cPt3dr & aPGround) const 
{
    cPt2dr aPIm = Ground2Image(aPGround);

    return cPt3dr(aPIm.x(),aPIm.y(),aPGround.z());
}

cPt3dr cSensorImage::ImageAndZ2Ground(const cPt3dr & aPImZ) const 
{
    tSeg3dr  aBundle =  Image2Bundle(cPt2dr(aPImZ.x(),aPImZ.y()));
    
    return BundleFixZ(aBundle,aPImZ.z());
}

            // ===========   coordinate systems  ==========================
	  
bool  cSensorImage::HasCoordinateSystem() const 
{
	return mNameSysCo.has_value();
}

const  std::string & cSensorImage::GetCoordinateSystem() const 
{
    MMVII_INTERNAL_ASSERT_tiny(HasCoordinateSystem(),"No coord system for" + NameImage());
    return mNameSysCo.value();
}

void cSensorImage::SetCoordinateSystem(const std::string& aSysCo) 
{
    if (aSysCo != MMVII_NONE)
       mNameSysCo = aSysCo;
}
std::optional<std::string> &  cSensorImage::OptCoordinateSystem() { return mNameSysCo; }

void cSensorImage::TransferateCoordSys(const cSensorImage & aSI)
{
    if (aSI.HasCoordinateSystem())
       SetCoordinateSystem(aSI.GetCoordinateSystem());
}
const std::string cSensorImage::TagCoordSys = "CoordinateSys";

bool  cSensorImage::IsSensorCamPC() const  { return false; }
const cSensorCamPC * cSensorImage::GetSensorCamPC() const
{
    MMVII_INTERNAL_ERROR("impossible required cast to cSensorCamPC");
    return nullptr;
}
cSensorCamPC * cSensorImage::GetSensorCamPC() 
{
    MMVII_INTERNAL_ERROR("impossible required cast to cSensorCamPC");
    return nullptr;
}

cSensorCamPC * cSensorImage::UserGetSensorCamPC() 
{
   if (!IsSensorCamPC())
   {
      MMVII_UnclasseUsEr("Camera " +  NameImage() + " was not central perspective");
   }
   return GetSensorCamPC();
}


const cSensorCamPC * cSensorImage::UserGetSensorCamPC() const
{
    return const_cast<cSensorImage*>(this)->UserGetSensorCamPC();
}



/* ******************************************************* */
/*                                                         */
/*                   cSIMap_Ground2ImageAndProf            */
/*                                                         */
/* ******************************************************* */

cSIMap_Ground2ImageAndProf::cSIMap_Ground2ImageAndProf(cSensorImage * aSens)  :
    mSI  (aSens)
{
}

cPt3dr cSIMap_Ground2ImageAndProf::Value(const cPt3dr & aPt) const
{
	return mSI->Ground2ImageAndDepth(aPt);
}

cPt3dr cSIMap_Ground2ImageAndProf::Inverse(const cPt3dr & aPt) const
{
	return mSI->ImageAndDepth2Ground(aPt);
}

/* ******************************************************* */
/*                                                         */
/*                   cSetVisibility                        */
/*                                                         */
/* ******************************************************* */

cSetVisibility::cSetVisibility(cSensorImage * aSens,double aBorder) :
            cDataBoundedSet<tREAL8,3> (cBox3dr::BigBox()),
            mSens                     (aSens),
	    mBorder                   (aBorder)
{}

tREAL8 cSetVisibility::Insideness(const cPt3dr & aP) const 
{
    return mSens->DegreeVisibility(aP) - mBorder;
}



}; // MMVII






