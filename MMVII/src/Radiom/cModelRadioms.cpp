#include "MMVII_Sensor.h"
#include "MMVII_PCSens.h"
#include "MMVII_Radiom.h"
#include "MMVII_Stringifier.h"
#include "MMVII_2Include_Serial_Tpl.h"


namespace MMVII
{

	/*
class cPreProcessRadiom
{
    public :
        cPreProcessRadiom(const cPerspCamIntrCalib &);
        const std::vector<tREAL8> & VObs(const cPt2dr & ) const;

    private :
        const cPerspCamIntrCalib *          mCal;
        mutable std::vector<tREAL8>         mVObs;
};

cPreProcessRadiom::cPreProcessRadiom(const cPerspCamIntrCalib & aCal) :
     mCal  (&aCal),
     mVObs ( {0.0,0.0,mCal->PP().x(),mCal->PP().y(),Norm2(mCal->SzPix())}  )
{
}


const std::vector<tREAL8> & cPreProcessRadiom::VObs(const cPt2dr & aPix ) const
{
    mVObs[0] = aPix.x();
    mVObs[1] = aPix.y();

    return mVObs;
}
*/

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
    cDataRadialCRS       (aData)
{
     if (mSzPix.x() >0)
     {
         cBox2dr aBoxIm(ToR(mSzPix));
         mScaleNor = Square(aBoxIm.DistMax2Corners(mCenter));
	 mVObs = std::vector<tREAL8>({0,0,mCenter.x(),mCenter.y(),mScaleNor});

	 mCalcFF = EqRadiomCalibRadSensor(mCoeffRad.size(),false,1);
     }
}
cRadialCRS::cRadialCRS(const cPt2dr & aCenter,size_t aDegRad,const cPt2di & aSzPix,const std::string & aNameCal) :
    cRadialCRS(cDataRadialCRS(aCenter,aDegRad,aSzPix,aNameCal))
{
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



/* ================================================== */
/*                  cCalibRadiomIma                   */
/* ================================================== */

cCalibRadiomIma::cCalibRadiomIma()
{
}

cCalibRadiomIma::~cCalibRadiomIma() {}

/* ================================================== */
/*                  cCalRadIm_Pol                     */
/* ================================================== */


cCalRadIm_Pol::cCalRadIm_Pol(cCalibRadiomSensor * aCalSens,int  aDegree,const std::string & aNameIm) :
      mCalibSens       (aCalSens),
      mDegree          (aDegree),
      mNameIm          (aNameIm)
{
     if (mDegree >= 0)
     {
         // Initialize with constant-1 polynom
         mCoeffPol.resize(RadiomCPI_NbParam(mDegree),0.0);
         mCoeffPol.at(0) = 1.0;
         mImaCorr =  EqRadiomCalibPolIma(mDegree,false,1);
	 mNameCalib = mCalibSens->NameCal();
     }
}

cCalRadIm_Pol::cCalRadIm_Pol()  :
    cCalRadIm_Pol(nullptr,-1,"")
{
}

const std::string & cCalRadIm_Pol::NameIm() const {return mNameIm;}

tREAL8  cCalRadIm_Pol::ImageOwnCorrec(const cPt2dr & aPt) const
{
    return mImaCorr->DoOneEval(mCoeffPol,mCalibSens->VObs(aPt)).at(0);
}

tREAL8  cCalRadIm_Pol::ImageCorrec(const cPt2dr & aPt) const
{
    return ImageOwnCorrec(aPt) * mCalibSens->FlatField(aPt);
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
     aRes->mImaCorr =  EqRadiomCalibPolIma(aRes->mDegree,false,1);

     return aRes; 
}

cCalibRadiomSensor &  cCalRadIm_Pol::CalibSens() {return *mCalibSens;}

void cCalRadIm_Pol::PutUknowsInSetInterval() 
{
   mSetInterv->AddOneInterv(mCoeffPol);
}




void  cCalRadIm_Pol::ToFile(const std::string & aNameFile) const 
{
    SaveInFile(*this,aNameFile);
    std::string aNameCalib = DirOfPath(aNameFile) + mNameCalib + ".xml";
    mCalibSens->ToFileIfFirstime(aNameCalib);
}

#if (0)
#endif


};
