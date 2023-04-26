#include "MMVII_nums.h"
#include "MMVII_Error.h"
#include "MMVII_Sensor.h"
#include "MMVII_2Include_Serial_Tpl.h"

/**
   \file cMetaDataImages.cpp

   \brief file for handling meta data
*/


namespace MMVII
{

//  =========  These class are used to indicate information missing (or wrong) on metadata or other stuff
class cOneTryCAI;
class cOneTranslAttrIm;
class cOneCalculMetaDataProject;
class cGlobCalculMetaDataProject;

/**  Define a try to associate a name to another .
 *     For a given name "N" if , if N match pattern then pattern
 *     substitution is used to compute  mValue.
 *
 *     For example :
 *         Pat =  IM_([0-9]*).tif
 *         Value = Stuf_$1
 *         N = IM_128.tif
 *
 *       the value computed is  Stuf_128
 */

class cOneTryCAI
{
     public :
        cOneTryCAI();


        cOneTryCAI(const std::string & aPat,const std::string & aValue);

        std::string                  mPat;
	tNameSelector                mSel;
        std::string                  mValue;
	std::optional<std::string>   mPatDir;

};

/**    Define the rule for associting a value to name :
 *
 *        - return the firt value computed by a try in  VTry for which the name match
 *        - return Default if none
 */

class cOneTranslAttrIm
{
     public :
	  cOneTranslAttrIm();

          std::string Translate(const std::string & aName) const;

	  eMTDIm                   mMode;
          std::vector<cOneTryCAI>  mVTries;
};

/**   Define the value computed for all possible  enums
 */
class cCalculMetaDataProject
{
     public :
	 cCalculMetaDataProject();
         std::string Translate(const std::string &,eMTDIm ) const;

         std::vector<cOneTranslAttrIm>  mTranslators;

	 static void  GenerateSample(const std::string & aNameFile);
	 static const std::string  NameStdFile;
};

// class cCalculMetaDataProject
 
class cGlobCalculMetaDataProject
{
     public :
         std::string Translate(const std::string &,eMTDIm ) const;
         void AddDir(const std::string& aDir);
         void      SetReal(tREAL8 & aVal,const std::string &,eMTDIm ) const;
         void      SetName(std::string & aVal,const std::string &,eMTDIm ) const;
     private :
	 std::vector<cCalculMetaDataProject>  mTranslators;
};

/* ******************************************* */
/*                                             */
/*          cGlobCalculMetaDataProject         */
/*                                             */
/* ******************************************* */

void cGlobCalculMetaDataProject::AddDir(const std::string& aDir)
{
     std::string aNameF = aDir + cCalculMetaDataProject::NameStdFile;


     if (ExistFile(aNameF))
     {
         cCalculMetaDataProject aCalc;
         ReadFromFile(aCalc,aNameF);
	 mTranslators.push_back(aCalc);
     }
}

std::string cGlobCalculMetaDataProject::Translate(const std::string & aName,eMTDIm aMode) const
{
    for (const auto & aTr : mTranslators)
    {
        std::string aRes = aTr.Translate(aName,aMode);
	if (aRes != MMVII_NONE)
           return aRes;
    }
    return MMVII_NONE;
}

void     cGlobCalculMetaDataProject::SetReal(tREAL8 & aVal,const std::string & aNameIm,eMTDIm aMode) const
{
    // already set by a more important rule
    if (aVal !=-1) return;

    std::string aTr = Translate(aNameIm,aMode);

    if (aTr !=MMVII_NONE)  
        aVal =  cStrIO<double>::FromStr(aTr);
}

void  cGlobCalculMetaDataProject::SetName(std::string & aVal,const std::string & aNameIm,eMTDIm aMode) const
{
    // already set by a more important rule
    if (aVal !="") return;

    std::string aTr = Translate(aNameIm,aMode);

    if (aTr !=MMVII_NONE)  
        aVal =  aTr;
}

/* ******************************************* */
/*                                             */
/*                cOneTryCAI                   */
/*                                             */
/* ******************************************* */

cOneTryCAI::cOneTryCAI(const std::string & aPat,const std::string & aValue) :
     mPat     (aPat),
     mSel     (AllocRegex(aPat)),
     mValue   (aValue)
{
}

cOneTryCAI::cOneTryCAI() :
	cOneTryCAI("","")
{
}

void AddData(const cAuxAr2007 & anAux,cOneTryCAI & aTry)
{
     AddData(cAuxAr2007("Pat",anAux),aTry.mPat);

     if (anAux.Input())
     {
         aTry.mSel = AllocRegex(aTry.mPat);
     }
     AddData(cAuxAr2007("Val",anAux),aTry.mValue);
}

/* ******************************************* */
/*                                             */
/*             cOneTranslAttrIm                */
/*                                             */
/* ******************************************* */

cOneTranslAttrIm::cOneTranslAttrIm():
    mMode (eMTDIm::eNbVals)
{
}

std::string cOneTranslAttrIm::Translate(const std::string & aName) const
{
    for (const auto & aTry : mVTries)
    {
        if (aTry.mSel.Match(aName))
	{
            std::string aTransfo = ReplacePattern(aTry.mPat,aTry.mValue,aName);
	    if (aTransfo != MMVII_NONE)
               return aTransfo;
	}
    }
    return MMVII_NONE;
}

void AddData(const cAuxAr2007 & anAux,cOneTranslAttrIm & aTransl)
{
      //  cAuxAr2007 anAux("Translat",anAuxParam);

      EnumAddData(anAux,aTransl.mMode,"Mode");
      AddData(anAux,aTransl.mVTries);
}

/* ******************************************* */
/*                                             */
/*         cCalculMetaDataProject              */
/*                                             */
/* ******************************************* */


cCalculMetaDataProject:: cCalculMetaDataProject()
{
}

void AddData(const cAuxAr2007 & anAuxParam,cCalculMetaDataProject & aCalc)
{
	cAuxAr2007 anAux("MetaData",anAuxParam);
	AddData(anAux,aCalc.mTranslators);
}

std::string cCalculMetaDataProject::Translate(const std::string & aName,eMTDIm  aMode) const
{
    for (const auto & aTransl : mTranslators)
    {
         if (aTransl.mMode==aMode)
            return aTransl.Translate(aName);
    }
    return MMVII_NONE;
}


void cCalculMetaDataProject::GenerateSample(const std::string & aNameFile)
{
   if (ExistFile(aNameFile))
      return;

   cCalculMetaDataProject aRes;

   for (size_t aILab = 0 ; aILab< size_t(eMTDIm::eNbVals) ; aILab++ )
   {
	  cOneTranslAttrIm aCAI;
	  aCAI.mMode    = (eMTDIm) aILab;

	  cOneTryCAI  aTry;
	  aTry.mPat = "XXXXXXXXXX.*XXXXXXX";
	  for (auto aV : {"1","2"})
	  {
              if (aCAI.mMode == eMTDIm::eFocalmm)
              {
                 aTry.mValue = aV;
                 aCAI.mVTries.push_back(aTry);
              }
	  }
	  aRes.mTranslators.push_back(aCAI);
   }

   SaveInFile(aRes,aNameFile);
}

const std::string  cCalculMetaDataProject::NameStdFile = "CalcMTD.xml";


/* ******************************************* */
/*                                             */
/*         cMetaDataImage                      */
/*                                             */
/* ******************************************* */

tREAL8  cMetaDataImage::Aperture() const
{
   MMVII_INTERNAL_ASSERT_User(mAperture>0,eTyUEr::eNoAperture,"Aperture is not init for " + mNameImage);
   return mAperture;
}

tREAL8  cMetaDataImage::FocalMM() const
{
   MMVII_INTERNAL_ASSERT_User(mFocalMM>0,eTyUEr::eNoFocale,"Focale is not init for " + mNameImage);
   return mFocalMM;
}

tREAL8  cMetaDataImage::FocalMMEqui35() const
{
    MMVII_INTERNAL_ASSERT_User(mFocalMMEqui35>0,eTyUEr::eNoFocaleEqui35,"FocaleEqui35 is not init for " + mNameImage);
   return mFocalMMEqui35;
}

const std::string&  cMetaDataImage::CameraName() const
{
    MMVII_INTERNAL_ASSERT_User(mCameraName!="",eTyUEr::eNoCameraName,"Camera Name is not init for " + mNameImage);
    return mCameraName;
}



cMetaDataImage::cMetaDataImage(const std::string & aDir,const std::string & aNameIm,const cGlobCalculMetaDataProject * aGlobCalc) :
   cMetaDataImage()
{
    mNameImage    = aNameIm;

    aGlobCalc->SetReal(mAperture,aNameIm,eMTDIm::eAperture);
    aGlobCalc->SetReal(mFocalMM,aNameIm,eMTDIm::eFocalmm);
    aGlobCalc->SetName(mCameraName,aNameIm,eMTDIm::eModeleCam);
    aGlobCalc->SetName(mAdditionalName,aNameIm,eMTDIm::eAdditionalName);
}

cMetaDataImage::cMetaDataImage() :
    mCameraName       (""),
    mAdditionalName   (""),
    mAperture         (-1),
    mFocalMM          (-1),
    mFocalMMEqui35    (-1)
{
}

std::string  cMetaDataImage::InternalCalibGeomIdent() const
{
    std::string  aRes = "CalibIntr";
    aRes = aRes + "_Cam"+ ToStandardStringIdent(CameraName());  // replace " " by "_" , refuse special characters
    if (mAdditionalName!="")
    {
        aRes = aRes + "_Add"+ mAdditionalName;  // replace " " by "_" , refuse special characters
    }
    aRes = aRes + "_Foc"+ToStr(FocalMM());


    return aRes;
}


/* ******************************************* */
/*                                             */
/*         cMetaDataImage                      */
/*                                             */
/* ******************************************* */

cMetaDataImage cPhotogrammetricProject::GetMetaData(const std::string & aFullNameIm) const
{
   std::string aDir,aNameIm;
   SplitDirAndFile(aDir,aNameIm,aFullNameIm,false);
   static std::map<std::string,cMetaDataImage> aMap;
   auto  anIt = aMap.find(aNameIm);
   if (anIt== aMap.end())
   {
        if (mGlobCalcMTD==nullptr)
	{
           mGlobCalcMTD = new cGlobCalculMetaDataProject;
	   mGlobCalcMTD->AddDir(mDPMetaData.FullDirIn());
	   mGlobCalcMTD->AddDir(mAppli.DirProfileUsage());
	}

	// mDPMetaData.FullDirOut()
        aMap[aNameIm] = cMetaDataImage(aDir,aNameIm,mGlobCalcMTD);
   }

   return aMap[aNameIm];
}

void cPhotogrammetricProject::DeleteMTD()
{
    delete mGlobCalcMTD;
}

void cPhotogrammetricProject::GenerateSampleCalcMTD()
{
     cCalculMetaDataProject::GenerateSample( mDPMetaData.FullDirIn()+cCalculMetaDataProject::NameStdFile);
}

/* ******************************************* */
/*                                             */
/*         cAppli_EditCalcMetaDataImage        */
/*                                             */
/* ******************************************* */

class cAppli_EditCalcMetaDataImage : public cMMVII_Appli
{
     public :

        cAppli_EditCalcMetaDataImage(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);

     private :
        int Exe() override;
	
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override;

        cPhotogrammetricProject  mPhProj;
	eMTDIm                   mTypeMTDIM;
};

cAppli_EditCalcMetaDataImage::cAppli_EditCalcMetaDataImage(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
     cMMVII_Appli (aVArgs,aSpec),
     mPhProj      (*this)
{
}     

cCollecSpecArg2007 & cAppli_EditCalcMetaDataImage::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
  return      anArgObl
          <<  mPhProj.DPMetaData().ArgDirInMand()
	  <<  Arg2007(mTypeMTDIM ,"Type of meta-data",{AC_ListVal<eMTDIm>()})

   ;
}

cCollecSpecArg2007 & cAppli_EditCalcMetaDataImage::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
    return    anArgOpt
	    /*
           << AOpt2007(mNbTriplets,"NbTriplets","Number max of triplet tested in Ransac",{eTA2007::HDV})
           << AOpt2007(mNbIterBundle,"NbIterBund","Number of bundle iteration, after ransac init",{eTA2007::HDV})
           << AOpt2007(mShowBundle,"ShowBundle","Show detail of bundle results",{eTA2007::HDV})
	   */
    ;
}

int cAppli_EditCalcMetaDataImage::Exe() 
{
    mPhProj.FinishInit();

    return EXIT_SUCCESS;
}


    /* ==================================================== */
    /*                                                      */
    /*               MMVII                                  */
    /*                                                      */
    /* ==================================================== */


tMMVII_UnikPApli Alloc_EditCalcMetaDataImage(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_EditCalcMetaDataImage(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_EditCalcMetaDataImage
(
     "EditCalcMTDI",
      Alloc_EditCalcMetaDataImage,
      "Edit the calculator of Meta-Data images",
      {eApF::Project},
      {eApDT::Xml},
      {eApDT::Xml},
      __FILE__
);
/*
*/



}; // MMVII

