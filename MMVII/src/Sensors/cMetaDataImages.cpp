#include "MMVII_nums.h"
#include "MMVII_Error.h"
#include "MMVII_Sensor.h"
#include "MMVII_2Include_Serial_Tpl.h"
#include "MMVII_util_tpl.h"
#include "MMVII_PCSens.h"

/**
   \file cMetaDataImages.cpp

   \brief file for handling meta data
*/


namespace MMVII
{


//  =========  These class are used to indicate information missing (or wrong) on metadata or other stuff

class cOneTryCAI;                 ///< contains a pair Pattern => Transformation
class cOneTranslAttrIm;           ///< contains the set of pair for a given type
class cOneCalculMetaDataProject;  ///< contains all the translation for all type, contained in a single file
class cGlobCalculMetaDataProject; ///< contains all the files

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
        cOneTryCAI();  ///< Defaut cstr, required for serialization
        cOneTryCAI(const std::string & aPat,const std::string & aValue);

        std::string                  mPat;    ///< Pattern for selecting and translatting
	tNameSelector                mSel;    ///<  Computation of Pattern
        std::string                  mValue;  ///<  Value computed 

};

/**    Define the rule for associting a value to name :
 *
 *        - return the firt value computed by a try in  VTry for which the name match
 *        - return Default if none
 */

class cOneTranslAttrIm
{
     public :
	  cOneTranslAttrIm(); ///< Defaut cstr, required for serialization
            
	  /// Compute association 
          std::string Translate(const std::string & aName,bool ForTest) const;

	  eMTDIm                   mMode;  ///< Type as Focal, Apertur ...
          std::list<cOneTryCAI>  mVTries;  ///< List of associations
};

/**   Define the value computed for all possible  enums
 */
class cCalculMetaDataProject
{
     public :
	 cCalculMetaDataProject(); ///< Defaut cstr, required for serialization
	 /// Translate for a given name and type
         std::string Translate(const std::string &,eMTDIm,bool ForTest ) const;

	 /// Return the translator of the corresponding type
	 cOneTranslAttrIm * GetTransOfType(eMTDIm) const;

	 ///  Generate a sample of calculator , for user to modify it
	 static void  GenerateSample(const std::string & aNameFile);
         ///  Default name used to save computations
	 static const std::string  NameStdFile();

	 /// All translator for different type
         std::vector<cOneTranslAttrIm>  mTranslators;
};

// class cCalculMetaDataProject
 
class cGlobCalculMetaDataProject
{
     public :
         std::string Translate(const std::string &,eMTDIm,bool ForTest = false ) const;
         void   AddDir(const std::string& aDir);
         void   SetReal(tREAL8 & aVal,const std::string &,eMTDIm ) const;
         void   SetName(std::string & aVal,const std::string &,eMTDIm ) const;
         void   SetPt2dr(cPt2dr & aVal,const std::string &,eMTDIm ) const;
         void   SetPt2di(cPt2di & aVal,const std::string &,eMTDIm ) const;

	 cCalculMetaDataProject * CMDPOfName(const std::string &);

     private :
	 std::list<cCalculMetaDataProject>               mTranslators;
	 std::map<std::string,const cCalculMetaDataProject *>  mMapDir2T;

};

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

static const std::string TheNameSHOW_MTD = "TheNameSHOW_MTD";

std::string cOneTranslAttrIm::Translate(const std::string & aName,bool ForTest) const
{

    for (const auto & aTry : mVTries)
    {
        if (aName==TheNameSHOW_MTD)
	{
           StdOut()  <<  "     *  Rule [" << aTry.mPat <<"] => [" << aTry.mValue << "]" << std::endl;
	}
	else
	{
            if (ForTest)
                StdOut()  <<  "     *  with with pattern [" << aTry.mPat <<"] ,";
            if (aTry.mSel.Match(aName))
	    {
                std::string aTransfo = ReplacePattern(aTry.mPat,aTry.mValue,aName);
	        if (aTransfo != MMVII_NONE)
	        {
                   if (ForTest)
                     StdOut()  <<  " match and got : [" << aTransfo  << "]" << std::endl ;
// StdOut() << "TTrrRanfooo= " << aTransfo << " P=" << aTry.mPat << " V=" << aTry.mValue << " N=" << aName<< std::endl;
                   return aTransfo;
	        }
	        else
	        {
                    if (ForTest)
                       StdOut()  <<  " match but got " << MMVII_NONE << std::endl;
	        }
	    }
            if (ForTest)
               StdOut()  <<  "    no match" << std::endl;
	}
    }
    return MMVII_NONE;
}

void AddData(const cAuxAr2007 & anAux,cOneTranslAttrIm & aTransl)
{
      //  cAuxAr2007 anAux("Translat",anAuxParam);

      EnumAddData(anAux,aTransl.mMode,"Mode");
      AddData(cAuxAr2007("Tries",anAux),aTransl.mVTries);
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
    cOneTranslAttrIm * aTransl = GetTransOfType(aMode);
    if (aTransl)
    {
       if (ForTest || (aName==TheNameSHOW_MTD))
           StdOut()  <<  "   -> found section for : " << E2Str(aMode) << std::endl;
       return aTransl->Translate(aName,ForTest);
    }
	/*
    for (const auto & aTransl : mTranslators)
    {
         if (aTransl.mMode==aMode)
         {
            if (ForTest || (aName==TheNameSHOW_MTD))
                StdOut()  <<  "   -> found section for : " << E2Str(aMode) << std::endl;
            return aTransl.Translate(aName,ForTest);
         }
    }
    */
    if ((ForTest) ||  (aName==TheNameSHOW_MTD))
       StdOut()  <<  "   -> Did not find section for : " << E2Str(aMode) << std::endl;
    return MMVII_NONE;
}

cOneTranslAttrIm * cCalculMetaDataProject::GetTransOfType(eMTDIm aMode) const
{
    for (const auto & aTransl : mTranslators)
    {
        if (aTransl.mMode==aMode)
        {
           return const_cast<cOneTranslAttrIm*>(&aTransl);
        }
    }
    return nullptr;
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

const std::string  cCalculMetaDataProject::NameStdFile()
{
     return "CalcMTD." + cMMVII_Appli::CurrentAppli().TaggedNameDefSerial();
}

/* ******************************************* */
/*                                             */
/*          cGlobCalculMetaDataProject         */
/*                                             */
/* ******************************************* */

void cGlobCalculMetaDataProject::AddDir(const std::string& aDir)
{
     std::string aNameF = aDir + cCalculMetaDataProject::NameStdFile();

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
        if ((ForTest) ||  (aName==TheNameSHOW_MTD))
	{

             const std::string * aV = FindByVal(mMapDir2T,&aTr,false);
	     StdOut() << "============= Try with dir " << *aV << " =================" << std::endl;
	}
        std::string aRes = aTr.Translate(aName,aMode,ForTest);
	// StdOut()  << " WwwttTttt " << aRes << std::endl;
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

void  cGlobCalculMetaDataProject::SetPt2dr(cPt2dr & aVal,const std::string & aNameIm,eMTDIm aMode) const
{
    // already set by a more important rule
    if (aVal.x() >=0) return;

    std::string aTr = Translate(aNameIm,aMode);


    if (aTr !=MMVII_NONE)  
        aVal =  cStrIO<cPt2dr>::FromStr(aTr);
}




cCalculMetaDataProject * cGlobCalculMetaDataProject::CMDPOfName(const std::string & aName)
{
   const cCalculMetaDataProject * aRes = mMapDir2T[aName];
   MMVII_INTERNAL_ASSERT_tiny(aRes!=nullptr,"cGlobCalculMetaDataProject CMDPOfName");

   return const_cast<cCalculMetaDataProject *>(aRes);
}


/* ******************************************* */
/*                                             */
/*         cMetaDataImage                      */
/*                                             */
/* ******************************************* */

tREAL8  cMetaDataImage::Aperture(bool SVP) const
{
   MMVII_INTERNAL_ASSERT_User((mAperture>0) || SVP ,eTyUEr::eNoAperture,"Aperture is not init for " + mNameImage);
   return mAperture;
}

tREAL8  cMetaDataImage::FocalMM(bool SVP) const
{
   MMVII_INTERNAL_ASSERT_User((mFocalMM>0) || SVP ,eTyUEr::eNoFocale,"Focale is not init for " + mNameImage);
   return mFocalMM;
}


tREAL8  cMetaDataImage::FocalMMEqui35(bool SVP) const
{
    MMVII_INTERNAL_ASSERT_User((mFocalMMEqui35>0) || SVP ,eTyUEr::eNoFocaleEqui35,"FocaleEqui35 is not init for " + mNameImage);
   return mFocalMMEqui35;
}


tREAL8  cMetaDataImage::FocalPixel(bool SVP) const
{
   MMVII_INTERNAL_ASSERT_User((mFocalPixel>0) || SVP ,eTyUEr::eUnClassedError,"Focal Pixel is not init for " + mNameImage);
   return mFocalPixel;
}

cPt2dr  cMetaDataImage::PPPixel(bool SVP) const
{
   MMVII_INTERNAL_ASSERT_User((mPPPixel.x()>0) || SVP ,eTyUEr::eUnClassedError,"Principal Point Pixel is not init for " + mNameImage);
   return mPPPixel;
}




cPt2di  cMetaDataImage::NbPixels(bool SVP) const
{
    MMVII_INTERNAL_ASSERT_User((mNbPixel.x()>0) || SVP ,eTyUEr::eNoNumberPixel,"Number pixel is not init for " + mNameImage);

    return mNbPixel;
}


const std::string&  cMetaDataImage::CameraName(bool SVP) const
{
    MMVII_INTERNAL_ASSERT_User((mCameraName!="") || SVP ,eTyUEr::eNoCameraName,"Camera Name is not init for " + mNameImage);
    return mCameraName;
}



cMetaDataImage::cMetaDataImage(const std::string & aDir,const std::string & aNameIm,const cGlobCalculMetaDataProject * aGlobCalc) :
   cMetaDataImage()
{
    mNameImage    = aNameIm;

    aGlobCalc->SetPt2dr(mPPPixel,aNameIm,eMTDIm::ePPPix);
    aGlobCalc->SetReal(mFocalPixel,aNameIm,eMTDIm::eFocalPix);

    aGlobCalc->SetReal(mAperture,aNameIm,eMTDIm::eAperture);
    aGlobCalc->SetReal(mFocalMM,aNameIm,eMTDIm::eFocalmm);
    aGlobCalc->SetName(mCameraName,aNameIm,eMTDIm::eModelCam);
    aGlobCalc->SetName(mAdditionalName,aNameIm,eMTDIm::eAdditionalName);

    /// StdOut()  <<  "cMetaDataImagecMetaDataImage " << mNameImage << " " << mAdditionalName << "\n" ; 
}

cMetaDataImage::cMetaDataImage() :
    mCameraName       (""),
    mAdditionalName   (""),
    mAperture         (-1),
    mFocalMM          (-1),
    mFocalMMEqui35    (-1),
    mFocalPixel       (-1),
    mPPPixel          (-1,-1),
    mNbPixel          (-1,-1)
{
}

std::string  cMetaDataImage::InternalCalibGeomIdent() const
{
    std::string  aRes = cPerspCamIntrCalib::SharedCalibPrefixName();
    aRes = aRes + "_Cam"+ ToStandardStringIdent(CameraName());  // replace " " by "_" , refuse special characters
    if (mAdditionalName!="")
    {
        aRes = aRes + "_Add"+ mAdditionalName;  // replace " " by "_" , refuse special characters
    }
    aRes = aRes + "_Foc"+ToStr(round_ni(FocalMM()*1000));

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
     cCalculMetaDataProject::GenerateSample( mDPMetaData.FullDirIn()+cCalculMetaDataProject::NameStdFile());
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
        virtual std::vector<std::string>  Samples() const; ///< For help, gives samples of "good" use


        cPhotogrammetricProject     mPhProj;
	eMTDIm                      mTypeMTDIM;

	bool                        mShow;
	bool                        mSave;
	std::string                 mNameImTest;
	std::vector<std::string>    mModif;

	cGlobCalculMetaDataProject* mCalcGlob;
	cCalculMetaDataProject*     mCalcProj;
};

cAppli_EditCalcMetaDataImage::cAppli_EditCalcMetaDataImage(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
     cMMVII_Appli (aVArgs,aSpec),
     mPhProj      (*this),
     mShow        (false),
     mSave        (false),
     mCalcGlob    (nullptr),
     mCalcProj    (nullptr)
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
          << AOpt2007(mShow,"Show","Show all rules",{eTA2007::HDV})
          << AOpt2007(mSave,"Save","Save result in a new file",{eTA2007::HDV})
          << AOpt2007(mModif,"Modif","Modification [Pat,Subst,Rank], Rank: {at(0)... ,-1 front,High back,at(0),-2 replace }",
			  {{eTA2007::ISizeV,"[3,3]"}})
	    /*
           << AOpt2007(mNbTriplets,"NbTriplets","Number max of triplet tested in Ransac",{eTA2007::HDV})
           << AOpt2007(mNbIterBundle,"NbIterBund","Number of bundle iteration, after ransac init",{eTA2007::HDV})
           << AOpt2007(mShowBundle,"ShowBundle","Show detail of bundle results",{eTA2007::HDV})
	   */
    ;
}

std::vector<std::string>  cAppli_EditCalcMetaDataImage::Samples() const
{
    return {
               "MMVII EditCalcMTDI Std ModelCam ImTest=043_0136.JPG  Modif=[.*.JPG,\"NIKON D5600\",0] Save=1",
               "MMVII EditCalcMTDI Std Focalmm ImTest=043_0136.JPG  Modif=[.*.JPG,24,0] Save=1",
               "MMVII EditCalcMTDI Std AdditionalName ImTest=043_0136.JPG  Modif=[\"(.*)_.*\",\"\\$1\",0] Save=1"
           };
}


int cAppli_EditCalcMetaDataImage::Exe() 
{
    mPhProj.DPMetaData().SetDirOutInIfNotInit();

    mPhProj.FinishInit();
    mCalcGlob = mPhProj.InitGlobCalcMTD();
    mCalcProj = mCalcGlob->CMDPOfName(mPhProj.DPMetaData().FullDirIn());

    // If we made some modification 
    if (IsInit(&mModif))
    {
	 cOneTranslAttrIm *   aTransPtr = mCalcProj->GetTransOfType(mTypeMTDIM);
	 //  if section for type did not exist we add one
	 if (aTransPtr==nullptr)
	 {
             cOneTranslAttrIm aTrans;
             aTrans.mMode = mTypeMTDIM;
	     mCalcProj->mTranslators.push_back(aTrans);
             aTransPtr = mCalcProj->GetTransOfType(mTypeMTDIM);
             MMVII_INTERNAL_ASSERT_tiny(aTransPtr!=nullptr,"KthElem");
	 }

         cOneTryCAI aTry(mModif.at(0),mModif.at(1));
	 int  aRank = cStrIO<int>::FromStr(mModif.at(2));

         std::list<cOneTryCAI> &  aVTries = aTransPtr->mVTries;

	 if (aRank>=0)
	 {
            if (aRank<int(aVTries.size()))
                *KthElem(aVTries,aRank) = aTry;
	    else
                aVTries.push_back(aTry);
	 }
	 else if (aRank==-1)
	 {
              aVTries.push_front(aTry);
             //mVTries.push_front(aTry);
	 }
	 else if (aRank==-2)
	 {
              aVTries.clear();
              aVTries.push_front(aTry);
	 }
	 else
	 {
              MMVII_UnclasseUsEr("Bad rank for Modif");
	 }
    }

    if (mShow)
    {
        StdOut() << std::endl;
        StdOut() << "********************************************" << std::endl;
        StdOut() << "*********       SHOW ALL RULES    **********" << std::endl;
        StdOut() << "********************************************\n" << std::endl;
        mCalcGlob->Translate(TheNameSHOW_MTD,mTypeMTDIM,false);
        StdOut() << "********************************************\n" << std::endl;
    }
    if (IsInit(&mNameImTest))
    {
        StdOut() << std::endl;
	StdOut() << "********************************************" << std::endl;
	StdOut() << "*********       TEST TRANSLATE    **********" << std::endl;
	StdOut() << "********************************************\n" << std::endl;
        std::string aRes =mCalcGlob->Translate(mNameImTest,mTypeMTDIM,true);
	StdOut() << std::endl;
	StdOut() << "    Result of computation=[" << aRes << "]\n" << std::endl;
	StdOut() << "********************************************" << std::endl;
    }

    if (mSave)
    {
       MakeBckUp(mPhProj.DPMetaData().FullDirIn(),cCalculMetaDataProject::NameStdFile(),5);

       SaveInFile(*mCalcProj,mPhProj.DPMetaData().FullDirIn() + cCalculMetaDataProject::NameStdFile());
       // mCalcProj
    }


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

