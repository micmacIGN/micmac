#include "MMVII_Sensor.h"
#include "MMVII_PCSens.h"
#include "MMVII_2Include_Serial_Tpl.h"

namespace MMVII
{

/*
  Vexcel example 4 first test 
      30975 pixel ~ 124 mm 
      Sz =  124/30975  =  4.003228 ~ 4  micron  
      NbPix = 26460 17004
      SzMm  = 105.840 x 68.016
      Focal = 123.9 
*/

/*  ******************************************************* */
/*                                                          */
/*                     cElemCamDataBase                     */
/*                                                          */
/*  ******************************************************* */

void cElemCamDataBase::AddData(const cAuxAr2007 & anAux)
{
    MMVII::AddData(cAuxAr2007("Name",anAux),mName);
    MMVII::AddData(cAuxAr2007("SzPix_micron",anAux),mSzPixel_Micron);
    MMVII::AddData(cAuxAr2007("SzSensor_mm",anAux),mSzSensor_Mm);
    MMVII::AddData(cAuxAr2007("NbPixels",anAux),mNbPixels);
}
void AddData(const cAuxAr2007 & anAux,cElemCamDataBase & anElem)
{
    anElem.AddData(anAux);
}

/*  ******************************************************* */
/*                                                          */
/*                     cCamDataBase                         */
/*                                                          */
/*  ******************************************************* */

void cCamDataBase::AddData(const cAuxAr2007 & anAux)
{
    MMVII::AddData(cAuxAr2007("CamDataBase",anAux),mMap);
}

void AddData(const cAuxAr2007 & anAux,cCamDataBase & aBase)
{
    aBase.AddData(anAux);
}

const std::map<std::string,cElemCamDataBase>  &  cCamDataBase::Map() const {return mMap;}
std::map<std::string,cElemCamDataBase>  &        cCamDataBase::Map()       {return mMap;}

/*  ******************************************************* */
/*                                                          */
/*                     cPhotogrammetricProject              */
/*                                                          */
/*  ******************************************************* */
static const std::string TheNameDBCam = "CameraDataBase.xml";

bool  cPhotogrammetricProject::OneTestMakeCamDataBase(const std::string & aDir,cCamDataBase & aDB,bool ForceNew)
{
  std::string aName = aDir + TheNameDBCam;
  if (! ExistFile(aName))
  {
     if (! ForceNew)
        return false;

     // If force new, create a entry just to have a template for editing
     cElemCamDataBase anElem;
     anElem.mName = "UltraCam Eagle Mark 3";
     anElem.mSzPixel_Micron = cPt2dr(4,4);
     anElem.mSzSensor_Mm = cPt2dr(105.840,68.016);
     anElem.mNbPixels = cPt2di(26460,17004);

     cCamDataBase aDataB;
     aDataB.Map()[anElem.mName] = anElem;
     SaveInFile(aDataB,aName);
  }

  cCamDataBase aDBNew;
  ReadFromFile(aDBNew,aName);

  for (const auto & [aName,aCam] : aDBNew.Map())
  {
        aDB.Map()[aName] = aCam;
  }

  return true;
}

void cPhotogrammetricProject::MakeCamDataBase()
{
    OneTestMakeCamDataBase(cMMVII_Appli::DirRessourcesMMVII(),mCamDataBase,true);
}

const cElemCamDataBase *  cPhotogrammetricProject::GetCamFromNameCam(const std::string& aNameCam,bool SVP) const
{
    auto anIter = mCamDataBase.Map().find(aNameCam);
    if (anIter == mCamDataBase.Map().end())
    {
        if (! SVP)
        {
            MMVII_UnclasseUsEr("Cannot get camera in data base for " + aNameCam);
        }
        return nullptr;
    } 

    return &(anIter->second);
}


cPerspCamIntrCalib * cPhotogrammetricProject::GetCalibInit
                     (
                          const std::string& aNameIm,
                          eProjPC aProj,
                          const cPt3di & aDeg, 
                          const cPt2dr &  aPPRel,
                          bool SVP
                     )
{
    static std::map<std::string,cPerspCamIntrCalib *> TheMapRes;

    // extract metadata : focal+name of camera
    cMetaDataImage  aMTD = GetMetaData(aNameIm);
    // if already exist return it
    std::string  anIdent = aMTD.InternalCalibGeomIdent();
    cPerspCamIntrCalib  * & aRes = TheMapRes[anIdent];
    if (aRes!=nullptr) 
       return aRes;

    // extract Camera from Data Base
    const cElemCamDataBase * aCam = GetCamFromNameCam(aMTD.CameraName(),SVP);
    if (aCam==nullptr)
    {
       return nullptr;
    }

    // [1] extract number of pixel
    cPt2di aNbPix  =   aMTD.NbPixels(true);  // if number of pixel was set by user or is in meta-data it has the prioriy
    if (aNbPix.x() <=0)
    {
        // if the file exist  read the pixel from image
        if (ExistFile(aNameIm))
        {
            cDataFileIm2D aDF2 = cDataFileIm2D::Create(aNameIm,false);
            aNbPix =  aDF2.Sz();
        }
        else
        {
            // else try to get the number of pixel from the camera
            if ((aCam==nullptr) || (aCam->mNbPixels.x()<=0))
                MMVII_UnclasseUsEr("Cannot compute number of pixel for " + aNameIm);
             aNbPix = aCam->mNbPixels;
        }
    }

    // [2] extract the focal length in pixel
    tREAL8 aFocPix = -1;

    tREAL8 aFoc35 = aMTD.FocalMMEqui35(true);
    if (aFoc35>0)
    {
       // in fact I am not sure of foc equi 35, but I think it is suited for a 24x36 ..
       aFocPix =  NormInf(aNbPix) * (aFoc35 / 35.0);
    }
    else
    {
       if ((aCam==nullptr) || (aCam->mSzPixel_Micron.x()<=0))
          MMVII_UnclasseUsEr("Cannot compute focal in pixel for : " + aNameIm);
       cPt2dr aSzPMicron = aCam->mSzPixel_Micron;
       if (aSzPMicron.x() != aSzPMicron.y())
          MMVII_UnclasseUsEr("Non squared pixel non handled for now");
    
       aFocPix = (aMTD.FocalMM() / aSzPMicron.x()) * 1000.0;
    }


    cDataPerspCamIntrCalib  aDataPCIC(anIdent, aProj, aDeg,aFocPix,aNbPix,aPPRel);
    aRes = new cPerspCamIntrCalib(aDataPCIC);

    cMMVII_Appli::AddObj2DelAtEnd(aRes);

    return aRes;
}

/* ************************************************************* */
/*                                                               */
/*                   cAppli_CreateCalib                          */
/*                                                               */
/* ************************************************************* */

class cAppli_CreateCalib : public cMMVII_Appli
{
     public :
        cAppli_CreateCalib(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;
        // std::vector<std::string>  Samples() const override;
     private :

        cPhotogrammetricProject   mPhProj;
        std::string               mSpecIm;
        eProjPC                   mProj;
        cPt3di                    mDegree;
        cPt2dr                    mPPRel;
};


cAppli_CreateCalib::cAppli_CreateCalib(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli  (aVArgs,aSpec),
   mPhProj       (*this),
   mProj         (eProjPC::eStenope),
   mDegree       (3,1,1),
   mPPRel        (0.5,0.5)
{
}

cCollecSpecArg2007 & cAppli_CreateCalib::ArgObl(cCollecSpecArg2007 & anArgObl)
{
    return anArgObl
              <<  Arg2007(mSpecIm ,"Name of Input File",{{eTA2007::MPatFile,"0"},{eTA2007::FileDirProj}})
              <<  mPhProj.DPOrient().ArgDirOutMand()
           ;
}

cCollecSpecArg2007 & cAppli_CreateCalib::ArgOpt(cCollecSpecArg2007 & anArgObl)
{
  return      anArgObl
            << AOpt2007(mProj,"Proj","Projection mode ",{{eTA2007::HDV},AC_ListVal<eProjPC>()})
            << AOpt2007(mDegree,"Degree","Degree for distorsion param",{{eTA2007::HDV}})
            // << AOpt2007(mNameBloc,"NameBloc","Set the name of the bloc ",{{eTA2007::HDV}})
    ;

}

int cAppli_CreateCalib::Exe() 
{
    mPhProj.FinishInit();

    for (const auto & aNameIm : VectMainSet(0))
    {
        cPerspCamIntrCalib * aCalib = mPhProj.GetCalibInit(aNameIm,mProj,mDegree,mPPRel);
        mPhProj.SaveCalibPC(*aCalib);
    }

    return EXIT_SUCCESS;
}

tMMVII_UnikPApli Alloc_CreateCalib(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_CreateCalib(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_CreateCalib
(
     "OriCreateCalib",
      Alloc_CreateCalib,
      "Create initial internal calibration",
      {eApF::SysCo,eApF::Ori},
      {eApDT::Ori},
      {eApDT::Ori},
      __FILE__
);

};


