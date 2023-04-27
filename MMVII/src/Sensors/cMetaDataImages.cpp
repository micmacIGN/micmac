#include "MMVII_nums.h"
#include "MMVII_Error.h"
#include "MMVII_Sensor.h"
#include "MMVII_2Include_Serial_Tpl.h"
#include "MMVII_util_tpl.h"

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
	// std::optional<std::string>   mPatDir;

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

          std::string Translate(const std::string & aName,bool ForTest) const;

	  eMTDIm                   mMode;
          std::list<cOneTryCAI>  mVTries;
};

/**   Define the value computed for all possible  enums
 */
class cCalculMetaDataProject
{
     public :
	 cCalculMetaDataProject();
         std::string Translate(const std::string &,eMTDIm,bool ForTest ) const;

         std::vector<cOneTranslAttrIm>  mTranslators;

	 static void  GenerateSample(const std::string & aNameFile);
	 static const std::string  NameStdFile;
};

// class cCalculMetaDataProject
 
class cGlobCalculMetaDataProject
{
     public :
         std::string Translate(const std::string &,eMTDIm,bool ForTest = false ) const;
         void   AddDir(const std::string& aDir);
         void   SetReal(tREAL8 & aVal,const std::string &,eMTDIm ) const;
         void   SetName(std::string & aVal,const std::string &,eMTDIm ) const;

	 cCalculMetaDataProject * CMDPOfName(const std::string &);

     private :
	 std::list<cCalculMetaDataProject>               mTranslators;
	 std::map<std::string,const cCalculMetaDataProject *>  mMapDir2T;

};

cCalculMetaDataProject * cGlobCalculMetaDataProject::CMDPOfName(const std::string & aName)
{
   const cCalculMetaDataProject * aRes = mMapDir2T[aName];
   MMVII_INTERNAL_ASSERT_tiny(aRes!=nullptr,"cGlobCalculMetaDataProject CMDPOfName");

   return const_cast<cCalculMetaDataProject *>(aRes);
}
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
	 mMapDir2T[aDir] = &(mTranslators.back()) ;
     }
}

std::string cGlobCalculMetaDataProject::Translate(const std::string & aName,eMTDIm aMode,bool ForTest) const
{
    for (const auto & aTr : mTranslators)
    {
        if (ForTest)
	{

             const std::string * aV = FindByVal(mMapDir2T,&aTr,false);
	     StdOut() << "============= Try with dir " << *aV << " =================\n";
	}
        std::string aRes = aTr.Translate(aName,aMode,ForTest);
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

std::string cOneTranslAttrIm::Translate(const std::string & aName,bool ForTest) const
{

    for (const auto & aTry : mVTries)
    {
        if (ForTest)
            StdOut()  <<  "     *  with with pattern [" << aTry.mPat <<"] ,";
        if (aTry.mSel.Match(aName))
	{
            std::string aTransfo = ReplacePattern(aTry.mPat,aTry.mValue,aName);
	    if (aTransfo != MMVII_NONE)
	    {
               if (ForTest)
                 StdOut()  <<  " match and got : [" << aTransfo  << "]\n" ;
               return aTransfo;
	    }
	    else
	    {
                if (ForTest)
                   StdOut()  <<  " match but got " << MMVII_NONE << "\n";
	    }
	}
        if (ForTest)
           StdOut()  <<  "    no match\n";
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

std::string cCalculMetaDataProject::Translate(const std::string & aName,eMTDIm  aMode,bool ForTest) const
{
    for (const auto & aTransl : mTranslators)
    {
         if (aTransl.mMode==aMode)
         {
            if (ForTest)
                StdOut()  <<  "   -> found section for : " << E2Str(aMode) << "\n";
            return aTransl.Translate(aName,ForTest);
         }
    }
    if (ForTest)
       StdOut()  <<  "   -> Did not find section for : " << E2Str(aMode) << "\n";
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
    aGlobCalc->SetName(mCameraName,aNameIm,eMTDIm::eModelCam);
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

cGlobCalculMetaDataProject * cPhotogrammetricProject::InitGlobCalcMTD() const
{
    if (mGlobCalcMTD==nullptr)
    {
           mGlobCalcMTD = new cGlobCalculMetaDataProject;
	   mGlobCalcMTD->AddDir(mDPMetaData.FullDirIn());
	   mGlobCalcMTD->AddDir(mAppli.DirProfileUsage());
    }
    return mGlobCalcMTD;
}

cCalculMetaDataProject * cPhotogrammetricProject::CMDPOfName(const std::string & aName)
{
    InitGlobCalcMTD();
    return mGlobCalcMTD->CMDPOfName(aName);
}

cMetaDataImage cPhotogrammetricProject::GetMetaData(const std::string & aFullNameIm) const
{
   std::string aDir,aNameIm;
   SplitDirAndFile(aDir,aNameIm,aFullNameIm,false);
   static std::map<std::string,cMetaDataImage> aMap;
   auto  anIt = aMap.find(aNameIm);
   if (anIt== aMap.end())
   {
        InitGlobCalcMTD();
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

        cPhotogrammetricProject     mPhProj;
	eMTDIm                      mTypeMTDIM;
	cCalculMetaDataProject*     mCalcProj;
	cGlobCalculMetaDataProject* mCalcGlob;
	std::string                 mNameImTest;
	std::vector<std::string>    mModif;
};

cAppli_EditCalcMetaDataImage::cAppli_EditCalcMetaDataImage(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
     cMMVII_Appli (aVArgs,aSpec),
     mPhProj      (*this)
{
}     

cCollecSpecArg2007 & cAppli_EditCalcMetaDataImage::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
   return     anArgObl
          <<  mPhProj.DPMetaData().ArgDirInMand()
	  <<  Arg2007(mTypeMTDIM ,"Type of meta-data",{AC_ListVal<eMTDIm>()})
   ;
}

cCollecSpecArg2007 & cAppli_EditCalcMetaDataImage::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
    return    anArgOpt
          <<  mPhProj.DPMetaData().ArgDirOutOpt()
          << AOpt2007(mNameImTest,"ImTest","Im for testing rules")
          << AOpt2007(mModif,"Modif","Modification [Pat,Subst,Rank], Rank: {at(0)... ,-1 front,High back,at(0),-2 replace }",
			  {{eTA2007::ISizeV,"[3,3]"}})
	    /*
           << AOpt2007(mNbTriplets,"NbTriplets","Number max of triplet tested in Ransac",{eTA2007::HDV})
           << AOpt2007(mNbIterBundle,"NbIterBund","Number of bundle iteration, after ransac init",{eTA2007::HDV})
           << AOpt2007(mShowBundle,"ShowBundle","Show detail of bundle results",{eTA2007::HDV})
	   */
    ;
}

int cAppli_EditCalcMetaDataImage::Exe() 
{
    mPhProj.DPMetaData().SetDirOutInIfNotInit();

    mPhProj.FinishInit();
    mCalcGlob = mPhProj.InitGlobCalcMTD();
    mCalcProj = mCalcGlob->CMDPOfName(mPhProj.DPMetaData().FullDirIn());

    if (IsInit(&mModif))
    {
         cOneTryCAI aTry(mModif[0],mModif[1]);
	 int  aRank = cStrIO<int>::FromStr(mModif[2]);

	 if (aRank>=0)
	 {
	 }
	 else if (aRank==-1)
	 {
             //mVTries.push_front(aTry);
	 }
	 else if (aRank==-2)
	 {

             //mVTries.push_front(aTry);
	 }
    }

    if (IsInit(&mNameImTest))
    {
         mCalcGlob->Translate(mNameImTest,mTypeMTDIM,true);
    }


	   // mGlobCalcMTD->AddDir(mDPMetaData.FullDirIn());


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

}; // MMVII

