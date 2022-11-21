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

cCalibRadiomSensor::cCalibRadiomSensor(const std::string & aNameCal) :
    mNameCal  (aNameCal)
{
}

const std::string & cCalibRadiomSensor::NameCal() const {return mNameCal;}

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
/*
*/


/* ================================================== */
/*                    cRadialCRS                      */
/* ================================================== */

cRadialCRS::cRadialCRS(const cPt2dr & aCenter,size_t aDegRad,const cPt2di & aSzPix,const std::string & aNameCal) :
    cCalibRadiomSensor   (aNameCal),
    mCenter              (aCenter),
    mCoeffRad            (aDegRad,0.0),
    mSzPix               (aSzPix),
    mScaleNor            (-1.0)
{
     if (mSzPix.x() >0)
     {
         cBox2dr aBoxIm(ToR(mSzPix));
         mScaleNor = Square(aBoxIm.DistMax2Corners(mCenter));
     }
}

cRadialCRS::cRadialCRS() :
    cRadialCRS(cPt2dr(-1e10,-1e10),0,cPt2di(0,0),"NONE")
{
}

void  cRadialCRS::AddData(const cAuxAr2007 & anAux)
{
     MMVII::AddData(cAuxAr2007("Name",anAux)      ,mNameCal);
     MMVII::AddData(cAuxAr2007("Center",anAux)    ,mCenter);
     MMVII::AddData(cAuxAr2007("CoeffRad",anAux)  ,mCoeffRad);
     MMVII::AddData(cAuxAr2007("SzPix",anAux)     ,mSzPix);
     MMVII::AddData(cAuxAr2007("ScaleNorm",anAux) ,mScaleNor);
}

void AddData(const cAuxAr2007 & anAux,cRadialCRS & aRCRS)
{
    aRCRS.AddData(anAux);
}

void  cRadialCRS::ToFile(const std::string & aNameFile) const
{
      SaveInFile(const_cast<cRadialCRS&>(*this),aNameFile);
}

cRadialCRS * cRadialCRS::FromFile(const std::string & aNameFile)
{
   return RemanentObjectFromFile<cRadialCRS,cRadialCRS>(aNameFile);
}

std::vector<double>& cRadialCRS::CoeffRad() {return mCoeffRad;}


tREAL8  cRadialCRS::NormalizedRho2(const cPt2dr & aPt) const
{
      return SqN2(ToR(aPt)-mCenter) / mScaleNor;
}

tREAL8  cRadialCRS::FlatField(const cPt2dr & aPt) const
{
      tREAL8  aRho2 = NormalizedRho2(aPt);
      tREAL8 aSum = 1.0;
      tREAL8 aPowRho2 = 1.0;

      for (const auto & aCoeff : mCoeffRad)
      {
           aPowRho2 *= aRho2;
           aSum += aCoeff * aPowRho2;
      }
      return aSum;
}

/* ================================================== */
/*                  cCalibRadiomIma                   */
/* ================================================== */

cCalibRadiomIma::cCalibRadiomIma(const std::string & aNameIm) :
   mNameIm  (aNameIm)
{
}
const std::string & cCalibRadiomIma::NameIm() const {return mNameIm;}

cCalibRadiomIma::~cCalibRadiomIma() {}

/* ================================================== */
/*                  cCalRadIm_Cst                     */
/* ================================================== */


cCalRadIm_Cst::cCalRadIm_Cst(cCalibRadiomSensor * aCalSens,const std::string & aNameIm) :
      cCalibRadiomIma  (aNameIm),
      mCalibSens       (aCalSens),
      mDivIm           (1.0)
{
}

tREAL8  cCalRadIm_Cst::ImageCorrec(const cPt2dr & aPt) const
{
    return mDivIm * mCalibSens->FlatField(aPt);
}

cCalRadIm_Cst * cCalRadIm_Cst::FromFile(const std::string & aName)
{
     cCalRadIm_Cst *  aRes = new cCalRadIm_Cst(nullptr,"NONE");
     ReadFromFile(*aRes,aName);
     aRes->mCalibSens = cCalibRadiomSensor::FromFile(DirOfPath(aName) + aRes->mTmpCalib + ".xml");
     aRes->mTmpCalib = "";

     return aRes; 
}

tREAL8 & cCalRadIm_Cst::DivIm() {return mDivIm;}
const tREAL8 & cCalRadIm_Cst::DivIm() const {return mDivIm;}
cCalibRadiomSensor &  cCalRadIm_Cst::CalibSens() {return *mCalibSens;}

void  cCalRadIm_Cst::AddData(const cAuxAr2007 & anAux)
{
    MMVII::AddData(cAuxAr2007("DivIm",anAux) ,mDivIm);
    MMVII::AddData(cAuxAr2007("NameIm",anAux) ,mNameIm);

    if (!anAux.Input())
       mTmpCalib = mCalibSens->NameCal();
    MMVII::AddData(cAuxAr2007("NameCal",anAux) ,mTmpCalib);
}

void AddData(const cAuxAr2007 & anAux,cCalRadIm_Cst & aCRI_Cst)
{
    aCRI_Cst.AddData(anAux);
}

void  cCalRadIm_Cst::ToFile(const std::string & aNameFile) const 
{
    SaveInFile(*this,aNameFile);
    std::string aNameCalib = DirOfPath(aNameFile) + mTmpCalib + ".xml";
    mCalibSens->ToFileIfFirstime(aNameCalib);

    mTmpCalib = "";
}

};
