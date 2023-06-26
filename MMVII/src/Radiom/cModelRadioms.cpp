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
    MMVII::ToFileIfFirstime(*this,aNameFile);
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

int cCalibRadiomSensor::NbParamRad() const 
{
   MMVII_INTERNAL_ERROR("No default NbParamRad");
   return -1;
}

/* ============================================= */
/*                  cDataRadialCRS               */
/* ============================================= */

cDataRadialCRS::cDataRadialCRS(const cPt2dr & aCenter,size_t aDegRad,const cPt2di & aSzPix,const std::string & aNameCal) :
   mNameCal  (aNameCal),
   mCenter   (aCenter),
   mCoeffRad (aDegRad,0.0),
   mSzPix    (aSzPix)
{
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

	 mCalcFF = EqRadiomCalibRadSensor(mCoeffRad.size(),false,1);
     }
}
cRadialCRS::cRadialCRS(const cPt2dr & aCenter,size_t aDegRad,const cPt2di & aSzPix,const std::string & aNameCal) :
    cRadialCRS(cDataRadialCRS(aCenter,aDegRad,aSzPix,aNameCal))
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

tREAL8  cRadialCRS::FlatField(const cPt2dr & aPt) const
{
      return mCalcFF->DoOneEval(mCoeffRad,VObs(aPt)).at(0);
}

void cRadialCRS::PutUknowsInSetInterval()
{
   mSetInterv->AddOneInterv(mCoeffRad);
}

const std::string & cRadialCRS::NameCal() const
{
   return mNameCal;
}

int cRadialCRS::NbParamRad() const  
{
    return  mCoeffRad.size();
}

const std::vector<tREAL8>& cRadialCRS::CoeffRad() const 
{
    return  mCoeffRad;
}



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
      mImaEqual        (nullptr)
{
     if (mDegree >= 0)
     {
         // Initialize with constant-1 polynom
         mCoeffPol.resize(VDesc_RadiomCPI(mDegree).size(),0.0);
         mCoeffPol.at(0) = 1.0;
         mImaOwnCorr =  EqRadiomCalibPolIma(mDegree,false,1);
         mImaEqual = EqRadiomEqualisation(mCalibSens->NbParamRad(),mDegree,true,1);
	 mNameCalib = mCalibSens->NameCal();

	 StdOut() << "CPPP=" << mCoeffPol  << " D=" << mDegree << "\n";
     }
}

cCalRadIm_Pol::cCalRadIm_Pol()  :
    cCalRadIm_Pol(nullptr,-1,"")
{
}

cCalRadIm_Pol::~cCalRadIm_Pol() 
{
    delete mImaOwnCorr;
    delete mImaEqual;
}

const std::string & cCalRadIm_Pol::NameIm() const {return mNameIm;}

tREAL8  cCalRadIm_Pol::ImageOwnDivisor(const cPt2dr & aPt) const
{
    return mImaOwnCorr->DoOneEval(mCoeffPol,mCalibSens->VObs(aPt)).at(0);
}

tREAL8  cCalRadIm_Pol::ImageCorrec(tREAL8 aGray,const cPt2dr & aPt) const
{
    return aGray / (ImageOwnDivisor(aPt) * mCalibSens->FlatField(aPt));
}

cPt3dr  cCalRadIm_Pol::ImageCorrec(const cPt3dr & aRGB,const cPt2dr & aPt) const
{
    return aRGB / (ImageOwnDivisor(aPt) * mCalibSens->FlatField(aPt));
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
     aRes->mCalibSens = cCalibRadiomSensor::FromFile(DirOfPath(aName) + aRes->mNameCalib + ".xml");
     aRes->mImaOwnCorr =  EqRadiomCalibPolIma(aRes->mDegree,false,1);
     aRes->mImaEqual = EqRadiomEqualisation(aRes->mCalibSens->NbParamRad(),aRes->mDegree,true,1);

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
    std::string aNameCalib = DirOfPath(aNameFile) + mNameCalib + ".xml";
    mCalibSens->ToFileIfFirstime(aNameCalib);
}

NS_SymbolicDerivative::cCalculator<double> * cCalRadIm_Pol::ImaEqual()
{
     return mImaEqual;
}

int cCalRadIm_Pol::MaxDegree() const 
{
    return mDegree;
}


};
