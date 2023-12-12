#include "MMVII_PCSens.h"
#include "MMVII_MMV1Compat.h"
#include "MMVII_DeclareCste.h"
#include "MMVII_BundleAdj.h"
#include "MMVII_2Include_Serial_Tpl.h"


/**
   \file cConvCalib.cpp  testgit

   \brief file for conversion between calibration (change format, change model) and tests
*/


namespace MMVII
{


   /* ********************************************************** */
   /*                                                            */
   /*                 cAppli_ImportOri                           */
   /*                                                            */
   /* ********************************************************** */

class cAppli_ImportOri : public cMMVII_Appli
{
     public :
        cAppli_ImportOri(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;

        std::vector<std::string>  Samples() const override;
     private :

	cPhotogrammetricProject  mPhProj;

	// Mandatory Arg
	std::string              mNameFile;
	std::string              mFormat;


	// Optionall Arg
	int                      mL0;
	int                      mLLast;
	char                     mComment;
	eTyUnitAngle             mAngleUnit;
	std::string              mRepIJK;
	bool                     mRepIJDir;
	std::vector<std::string> mChgName;
	std::vector<std::string> mChgName2;  // for ex, with IMU, we want to export as image init & imu measure
	std::string              mNameDicoName;
	std::string              mFileSaveIm;
	size_t                   mNbMinTieP;
};

cAppli_ImportOri::cAppli_ImportOri(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli  (aVArgs,aSpec),
   mPhProj       (*this),
   mL0           (0),
   mLLast        (-1),
   mComment      ('#'),
   mAngleUnit    (eTyUnitAngle::eUA_radian),
   mRepIJK       ("ijk"),
   mRepIJDir     (false),
   mNbMinTieP    (1)
{
}

cCollecSpecArg2007 & cAppli_ImportOri::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
    return anArgObl
	      <<  Arg2007(mNameFile ,"Name of Input File")
	      <<  Arg2007(mFormat   ,"Format of file as for ex \"SNSXYZSS\" ")
              <<  mPhProj.DPOrient().ArgDirInMand("Ori folder to extract calibration")
              <<  mPhProj.DPOrient().ArgDirOutMand()
           ;
}

cCollecSpecArg2007 & cAppli_ImportOri::ArgOpt(cCollecSpecArg2007 & anArgObl) 
{
    
    return anArgObl
       << AOpt2007(mL0,"NumL0","Num of first line to read",{eTA2007::HDV})
       << AOpt2007(mLLast,"NumLast","Num of last line to read (-1 if at end of file)",{eTA2007::HDV})
       << AOpt2007(mComment,"Com","Carac for commentary",{eTA2007::HDV})
       << AOpt2007(mAngleUnit,"AngU","Unity for angles",{{eTA2007::HDV},{AC_ListVal<eTyUnitAngle>()}})
       << AOpt2007(mRepIJK,"Rep","Repair coded (relative  to MMVII convention)  ",{{eTA2007::HDV}})
       << AOpt2007(mRepIJDir,"KIsUp","Corespond to repair \"i-j-k\" ",{{eTA2007::HDV}})
       << AOpt2007(mChgName,"ChgN","Change name [Pat,Name], for ex \"[(.*),\\$0.tif]\"  add postfix \"tif\" ",{{eTA2007::ISizeV,"[2,2]"}})
       << AOpt2007(mChgName2,"ChgN2","Change name [Pat,Name], for ex \"[(.*),\\$0.IMU]\"  add postfix \"IMU\" ",{{eTA2007::ISizeV,"[2,2]"}})
       << AOpt2007(mNameDicoName,"DicName","Dictionnary for changing names of images ")
       << AOpt2007(mFileSaveIm,"FileSaveIm","File for saving all names of images ")
       << mPhProj.DPMulTieP().ArgDirInOpt("TiePF","TieP for filtering on number")
       << AOpt2007(mNbMinTieP,"NbMinTieP","Number mininmal of tie point (when TiePF is init)",{eTA2007::HDV} )
    ;
}

std::vector<std::string>  cAppli_ImportOri::Samples() const
{
   return 
   {
        "MMVII ImportOri trajectographie_tif.opk NSSXYZWPKS Calib InitUP AngU=degree KIsUp=true ChgN=[\".*\",\"Traj_\\$0\"]",
        "MMVII ImportOri trajectographie_tif.opk NSSXYZWPKS Calib001 InitUP AngU=degree KIsUp=true DicName=DicoVol.xml"
   };
}

// "NSSXYZWPKS"

int cAppli_ImportOri::Exe()
{
    mPhProj.FinishInit();
    std::vector<std::vector<std::string>> aVNames;
    std::vector<std::vector<double>> aVNums;
    std::vector<cPt3dr> aVXYZ,aVWKP;

    if (mRepIJDir)
       mRepIJK = "i-j-k";

    cRotation3D<tREAL8>  aRotAfter = cRotation3D<tREAL8>::RotFromCanonicalAxes(mRepIJK);

    // MMVII_INTERNAL_ASSERT_tiny(CptSameOccur(mFormat,"XYZN")==1,"Bad format vs NXYZ");

    ReadFilesStruct
    (
        mNameFile, mFormat,
        mL0, mLLast, mComment,
        aVNames,aVXYZ,aVWKP,aVNums,
	false
    );

    tREAL8  aAngDiv = AngleInRad(mAngleUnit);

    std::string mSeparator = "@";
    std::map<std::string,std::string>  aDicoChName;
    bool withDico = false;

    if (IsInit(&mNameDicoName))
    {
        withDico = true;
	ReadFromFile(aDicoChName,mNameDicoName);
    }

    tNameSet aSetIm;
    bool  WithName2 = IsInit(&mChgName2);

    for (size_t aK=0 ; aK<aVXYZ.size() ; aK++)
    {
         std::string aNameIm = aVNames.at(aK).at(0);
	 for (size_t aKName=1 ; aKName<aVNames.at(aK).size() ; aKName++)
             aNameIm = aNameIm + mSeparator + aVNames.at(aK).at(aKName);

	 ChgName(mChgName,aNameIm);

         if (withDico)
         {
            auto anIt = aDicoChName.find(aNameIm);
            if (anIt == aDicoChName.end())
            {
                MMVII_UnclasseUsEr("Cannot find name in dico for : " + aNameIm);
            }
	    aNameIm = anIt->second;
         }
         std::string aNameIm2 = aNameIm;
         ChgName(mChgName2,aNameIm2);

	 // StdOut() << "aNameImaNameIm=" << aNameIm << "\n";
	 /*
	 bool Add2Set = mPhProj.HasNbMinMultiTiePoints(aNameIm,mNbMinTieP,true);
	 if ( mPhProj.DPMulTieP().DirInIsInit())
	 {
              cVecTiePMul aVPM(aNameIm);
	      mPhProj.ReadMultipleTieP(aVPM,aNameIm,true);
	      Add2Set = (aVPM.mVecTPM.size() >= mNbMinTieP);
	 }
	 */

	 if (mPhProj.HasNbMinMultiTiePoints(aNameIm,mNbMinTieP,true))
	 {
	    aSetIm.Add(aNameIm);
            if (WithName2)
	       aSetIm.Add(aNameIm2);
	 }
            

	 // Orient may be non init, by passing NONE, if we want to generate only the name

	 if (mPhProj.DPOrient().DirInIsInit())
	 {
	     cPt3dr aCenter = aVXYZ.at(aK);
	     cPt3dr aWPK = aVWKP.at(aK) / aAngDiv ;

	     cPerspCamIntrCalib *  aCalib = mPhProj.InternalCalibFromStdName(aNameIm);

	 
	     cRotation3D<tREAL8>  aRot =  cRotation3D<tREAL8>::RotFromWPK(aWPK);
	     aRot = aRot * aRotAfter;

	     cIsometry3D aPose(aCenter,aRot);
	     cSensorCamPC  aCam(aNameIm,aPose,aCalib);
	     mPhProj.SaveCamPC(aCam);
             if (WithName2)
             {
                 cSensorCamPC aCam2(aNameIm2,aPose,nullptr);
                 mPhProj.SaveCamPC(aCam2);  
             }
	 }
    }

    if (IsInit(&mFileSaveIm))
       SaveInFile(aSetIm,mFileSaveIm);

    return EXIT_SUCCESS;
}




tMMVII_UnikPApli Alloc_ImportOri(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_ImportOri(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_ImportOri
(
     "ImportOri",
      Alloc_ImportOri,
      "Import/Convert basic Orient file in MMVII format",
      {eApF::Ori},
      {eApDT::Ori},
      {eApDT::Ori},
      __FILE__
);


}; // MMVII

