#include "MMVII_Sensor.h"
#include "MMVII_PCSens.h"
#include "MMVII_Radiom.h"
#include "MMVII_Stringifier.h"
#include "MMVII_2Include_Serial_Tpl.h"


namespace MMVII
{

/* ================================================== */
/*                cCalibRadiomSensor                  */
/* ================================================== */

cCalibRadiomSensor::cCalibRadiomSensor()
{
}


void  cCalibRadiomSensor::ToFileIfFirstime(const std::string & aNameFile) const
{
    MMVII::ToFileIfFirstime(this,aNameFile);
}

cCalibRadiomSensor * cCalibRadiomSensor::FromFile(const std::string & aNameFile)
{
   if (starts_with(FileOfPath(aNameFile),PrefixCalRadRad))
      return cRadialCRS::FromFile(aNameFile);

   MMVII_UsersErrror(eTyUEr::eUnClassedError,"Cannot determine radiom-file mode for :" + aNameFile);
   return nullptr;
}

cCalibRadiomSensor::~cCalibRadiomSensor() 
{
}

/*
int cCalibRadiomSensor::NbParamRad() const 
{
   MMVII_INTERNAL_ERROR("No default NbParamRad");
   return -1;
}
*/

/* ============================================= */
/*                  cDataRadialCRS               */
/* ============================================= */

cDataRadialCRS::cDataRadialCRS
(
      const cPt2dr & aCenter,
      size_t aDegRad,
      const cPt2di & aSzPix,
      const std::string & aNameCal,
      bool   WithCsteAdd,
      int    aDegPol
) :
   mNameCal      (aNameCal),
   mCenter       (aCenter),
   mCoeffRad     (aDegRad,0.0),
   mWithAddCste  (WithCsteAdd),
   mCste2Add     (0.0),
   mDegPol       (aDegPol),
   mSzPix        (aSzPix)
{
   if (mDegPol>0)
      mCoeffPol.resize(VDesc_RadiomCPI(mDegPol,aDegRad).size(),0.0);
}
//
// === defaut constructor for serialization =============
cDataRadialCRS::cDataRadialCRS() :
    cDataRadialCRS(cPt2dr(-1e10,-1e10),0,cPt2di(0,0),"NONE")
{
}

   
void  cDataRadialCRS::AddData(const cAuxAr2007 & anAux)
{
     MMVII::AddData(cAuxAr2007("Name",anAux)      ,mNameCal);
     MMVII::AddData(cAuxAr2007("Center",anAux)    ,mCenter);
     MMVII::AddData(cAuxAr2007("CoeffRad",anAux)  ,mCoeffRad);
     MMVII::AddData(cAuxAr2007("SzPix",anAux)     ,mSzPix);

     MMVII::AddData(cAuxAr2007("WithCsteAdd",anAux)      ,mWithAddCste);
     if (mWithAddCste)
     {
          MMVII::AddData(cAuxAr2007("CsteAdd",anAux)     ,mCste2Add);
     }
     MMVII::AddData(cAuxAr2007("DegPol",anAux)      ,mDegPol);
     if (mDegPol>0)
     {
          MMVII::AddData(cAuxAr2007("CoeffPol",anAux)     ,mCoeffPol);
     }

}
   
void AddData(const cAuxAr2007 & anAux,cDataRadialCRS & aDataRadCRS)
{
    aDataRadCRS.AddData(anAux);
}


/* ================================================== */
/*                    cRadialCRS                      */
/* ================================================== */

cRadialCRS::cRadialCRS (const cDataRadialCRS & aData) :
    cCalibRadiomSensor   (),
    cDataRadialCRS       (aData),
    mCalcFF              (nullptr)
{
     if (mSzPix.x() >0)
     {
         cBox2dr aBoxIm(ToR(mSzPix));
         mScaleNor = aBoxIm.DistMax2Corners(mCenter);
	 mVObs = std::vector<tREAL8>({0,0,mCenter.x(),mCenter.y(),mScaleNor});

	 mCalcFF = EqRadiomCalibRadSensor(mCoeffRad.size(),false,1,mWithAddCste,mDegPol);
     }
}

cRadialCRS::cRadialCRS(const cPt2dr & aCenter,size_t aDegRad,const cPt2di & aSzPix,const std::string & aNameCal,bool WithCste,int aDegPol) :
    cRadialCRS(cDataRadialCRS(aCenter,aDegRad,aSzPix,aNameCal,WithCste,aDegPol))
{
}

cRadialCRS::~cRadialCRS() 
{
     delete mCalcFF;
}


cRadialCRS * cRadialCRS::FromFile(const std::string & aNameFile)
{
   return RemanentObjectFromFile<cRadialCRS,cDataRadialCRS>(aNameFile);
}



void  cRadialCRS::ToFile(const std::string & aNameFile) const
{
      SaveInFile(static_cast<const cDataRadialCRS&>(*this),aNameFile);
}

std::vector<double>& cRadialCRS::CoeffRad() {return mCoeffRad;}

const std::vector<tREAL8> & cRadialCRS::VObs(const cPt2dr & aPix ) const
{
    mVObs[0] = aPix.x();
    mVObs[1] = aPix.y();

    return mVObs;
}

/*
tREAL8  cRadialCRS::CorrectRadiom(const tREAL8& aRadiom,const cPt2dr & aPt) const
{
      cPt2dr  aCC = AddMul_CC(aPt);
      return (aRadiom - aCC.x()) / aCC.y();
}
*/

cPt2dr  cRadialCRS::AddMul_CC(const cPt2dr & aPt) const
{
      std::vector<tREAL8>  aVUk;
      if (mWithAddCste)
         aVUk.push_back(mCste2Add);
      AppendIn(aVUk,mCoeffRad);

      if (mDegPol>0)
      {
	      // StdOut() << "YuyuymCoeffPolmCoeffPol " << mCoeffPol << std::endl;
          AppendIn(aVUk,mCoeffPol);
      }
      auto aVR =  mCalcFF->DoOneEval(aVUk,VObs(aPt));
      return cPt2dr(aVR.at(0),aVR.at(1));
}

void cRadialCRS::PutUknowsInSetInterval()
{
   if (mWithAddCste)
      mSetInterv->AddOneInterv(mCste2Add);
   mSetInterv->AddOneInterv(mCoeffRad);
   if (mCoeffPol.size())
      mSetInterv->AddOneInterv(mCoeffPol);
}

const std::string & cRadialCRS::NameCal() const
{
   return mNameCal;
}

int cRadialCRS::NbParamRad() const  
{
    return  mCoeffRad.size();
}

bool cRadialCRS::WithCste() const  
{
    return  mWithAddCste;
}

int  cRadialCRS::DegPol() const
{
    return mDegPol;
}



const std::vector<tREAL8>& cRadialCRS::CoeffRad() const 
{
    return  mCoeffRad;
}

tREAL8 & cRadialCRS::Cste2Add() {return mCste2Add;}



/* ================================================== */
/*                  cCalibRadiomIma                   */
/* ================================================== */

cCalibRadiomIma::cCalibRadiomIma()
{
}

cCalibRadiomIma::~cCalibRadiomIma() {}


int  cCalibRadiomIma::IndDegree(const cPt2di & aDegree,bool SVP) const
{
    const std::vector<cDescOneFuncDist> & aVDesc =  VDesc();

    for (size_t aK=0 ; aK<aVDesc.size() ; aK++)
    {
        if (aVDesc[aK].mDegMon == aDegree)
           return aK+IndUk0();
    }
    MMVII_INTERNAL_ASSERT_tiny(SVP,"Cannot find in cCalibRadiomIma::IndDegree");

    return -1;
}

int  cCalibRadiomIma::IndCste() const
{
	return IndDegree(cPt2di(0,0));
}

/* ================================================== */
/*                  cCalRadIm_Pol                     */
/* ================================================== */


cCalRadIm_Pol::cCalRadIm_Pol(cCalibRadiomSensor * aCalSens,int  aDegree,const std::string & aNameIm) :
      mCalibSens       (aCalSens),
      mDegree          (aDegree),
      mNameIm          (aNameIm),
      mImaOwnCorr      (nullptr),
      mImaEqual        (nullptr),
      mImaStab         (nullptr)
{
     if (mDegree >= 0)
     {
         // Initialize with constant-1 polynom
         mCoeffPol.resize(VDesc_RadiomCPI(mDegree).size(),0.0);
         mCoeffPol.at(0) = 1.0;
	 mNameCalib = mCalibSens->NameCal();
         // mImaOwnCorr =  EqRadiomCalibPolIma(mDegree,false,1);
         // mImaEqual = EqRadiomEqualisation(mCalibSens->NbParamRad(),mDegree,true,1);
	 PostInit();

	 // StdOut() << "CPPP=" << mCoeffPol  << " D=" << mDegree << std::endl;
     }
}

void cCalRadIm_Pol::PostInit()
{
    mImaOwnCorr = EqRadiomCalibPolIma(mDegree,false,1);
    mImaEqual   = EqRadiomEqualisation (mCalibSens->NbParamRad(),mDegree,true,1,mCalibSens->WithCste(),mCalibSens->DegPol());
    mImaStab    = EqRadiomStabilization(mCalibSens->NbParamRad(),mDegree,true,1,mCalibSens->WithCste(),mCalibSens->DegPol());
}

cCalRadIm_Pol::cCalRadIm_Pol()  :
    cCalRadIm_Pol(nullptr,-1,"")
{
}

cCalRadIm_Pol::~cCalRadIm_Pol() 
{
    delete mImaOwnCorr;
    delete mImaEqual;
    delete mImaStab;
}

const std::string & cCalRadIm_Pol::NameIm() const {return mNameIm;}

tREAL8  cCalRadIm_Pol::ImageOwnDivisor(const cPt2dr & aPt) const
{
    return mImaOwnCorr->DoOneEval(mCoeffPol,mCalibSens->VObs(aPt)).at(0);
}

tREAL8  cCalRadIm_Pol::ImageCorrec(tREAL8 aGray,const cPt2dr & aPt) const
{
    cPt2dr  aAddMul = mCalibSens->AddMul_CC(aPt);
    return  (aGray/ ImageOwnDivisor(aPt) - aAddMul.x()) / aAddMul.y();
}

cPt3dr  cCalRadIm_Pol::ImageCorrec(const cPt3dr & aRGB,const cPt2dr & aPt) const
{
    cPt2dr  aAddMul = mCalibSens->AddMul_CC(aPt);
    return (aRGB/ImageOwnDivisor(aPt)-cPt3dr::PCste(aAddMul.x())) /  aAddMul.y();
}

void  cCalRadIm_Pol::AddData(const cAuxAr2007 & anAux)
{
    MMVII::AddData(cAuxAr2007("NameCalib",anAux) ,mNameCalib);
    MMVII::AddData(cAuxAr2007("Degree",anAux)    ,mDegree);
    MMVII::AddData(cAuxAr2007("NameIm",anAux) ,   mNameIm);
    MMVII::AddData(cAuxAr2007("CoeffPol",anAux) , mCoeffPol);
}

void AddData(const cAuxAr2007 & anAux,cCalRadIm_Pol & aCalRadIm_Pol)
{
    aCalRadIm_Pol.AddData(anAux);
}

cCalRadIm_Pol * cCalRadIm_Pol::FromFile(const std::string & aName)
{
     cCalRadIm_Pol *  aRes = new cCalRadIm_Pol();
     ReadFromFile(*aRes,aName);
     aRes->mCalibSens = cCalibRadiomSensor::FromFile(DirOfPath(aName) + aRes->mNameCalib + "."+GlobTaggedNameDefSerial());
     // aRes->mImaOwnCorr =  EqRadiomCalibPolIma(aRes->mDegree,false,1);
     // aRes->mImaEqual = EqRadiomEqualisation(aRes->mCalibSens->NbParamRad(),aRes->mDegree,true,1);
     aRes->PostInit();

     return aRes; 
}

cCalibRadiomSensor &  cCalRadIm_Pol::CalibSens() {return *mCalibSens;}

void cCalRadIm_Pol::PutUknowsInSetInterval() 
{
   mSetInterv->AddOneInterv(mCoeffPol);
}

std::vector<double> &  cCalRadIm_Pol::Params() 
{
    return mCoeffPol;
}

const std::vector<cDescOneFuncDist> & cCalRadIm_Pol::VDesc()  const
{
	return VDesc_RadiomCPI(mDegree);
}


void  cCalRadIm_Pol::ToFile(const std::string & aNameFile) const 
{
    SaveInFile(*this,aNameFile);
    std::string aNameCalib = DirOfPath(aNameFile) + mNameCalib + "." + GlobTaggedNameDefSerial();
    mCalibSens->ToFileIfFirstime(aNameCalib);
}

NS_SymbolicDerivative::cCalculator<double> * cCalRadIm_Pol::ImaEqual()
{
     return mImaEqual;
}
NS_SymbolicDerivative::cCalculator<double> * cCalRadIm_Pol::ImaStab()
{
     return mImaStab;
}

int cCalRadIm_Pol::MaxDegree() const 
{
    return mDegree;
}


};
