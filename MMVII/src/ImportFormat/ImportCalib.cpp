#include "MMVII_PCSens.h"
#include "MMVII_MMV1Compat.h"
#include "MMVII_DeclareCste.h"
#include "MMVII_BundleAdj.h"

/**
   \file cConvCalib.cpp  testgit

   \brief file for conversion between calibration (change format, change model) and tests
*/


namespace MMVII
{

   /* ********************************************************** */
   /*                                                            */
   /*                 cAppli_V2ImportCalib                       */
   /*                                                            */
   /* ********************************************************** */

class cAppli_V2ImportCalib : public cMMVII_Appli
{
     public :
        cAppli_V2ImportCalib(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;
     private :

	cPhotogrammetricProject  mPhProj;

	// Mandatory Arg
	std::string              mExtFolder;
	std::string              mExtCalib;

	std::vector<std::string>  Samples() const override;
};

cAppli_V2ImportCalib::cAppli_V2ImportCalib(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli  (aVArgs,aSpec),
   mPhProj       (*this)
{
}

cCollecSpecArg2007 & cAppli_V2ImportCalib::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
    return anArgObl
	      <<  Arg2007(mExtFolder ,"Folder of external project where the calib is to be imported")
	      <<  Arg2007(mExtCalib ,"Calib to import")
              <<  mPhProj.DPOrient().ArgDirOutMand()
           ;
}

cCollecSpecArg2007 & cAppli_V2ImportCalib::ArgOpt(cCollecSpecArg2007 & anArgObl) 
{
    
    return anArgObl
       //  << AOpt2007(mNameGCP,"NameGCP","Name of GCP set")
       //  << AOpt2007(mNbDigName,"NbDigName","Number of digit for name, if fixed size required (only if int)")
       //  << AOpt2007(mL0,"NumL0","Num of first line to read",{eTA2007::HDV})
       //  << AOpt2007(mLLast,"NumLast","Num of last line to read (-1 if at end of file)",{eTA2007::HDV})
       //  << AOpt2007(mPatternTransfo,"PatName","Pattern for transforming name (first sub-expr)")
    ;
}


int cAppli_V2ImportCalib::Exe()
{
    mPhProj.FinishInit();

    std::string aDir  =  mExtFolder + mPhProj.DPOrient().DirLocOfMode() + mExtCalib + StringDirSeparator() ;

    tNameSelector aSel =   AllocRegex(cPerspCamIntrCalib::SharedCalibPrefixName()+".*");

    std::vector<std::string>   aLCalib = GetFilesFromDir(aDir,aSel);

    for (const auto & aNameCal : aLCalib)
    {
        CopyFile(aDir+aNameCal,mPhProj.DPOrient().FullDirOut()+aNameCal);
    }

   

    return EXIT_SUCCESS;
}


std::vector<std::string>  cAppli_V2ImportCalib::Samples() const
{
	return {"MMVII V2ImportCalib ../../Pannel/ BA_725 CalibInit725"};
}


tMMVII_UnikPApli Alloc_V2ImportCalib(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_V2ImportCalib(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_V2ImportCalib
(
     "V2ImportCalib",
      Alloc_V2ImportCalib,
      "Import a calibration from another MMVII project (just copy files)",
      {eApF::Ori},
      {eApDT::Ori},
      {eApDT::Ori},
      __FILE__
);


}; // MMVII

