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
   /*                 cAppli_ImportGCP                           */
   /*                                                            */
   /* ********************************************************** */

class cAppli_ImportGCP : public cMMVII_Appli
{
     public :
        cAppli_ImportGCP(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);
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
	std::string              mNameGCP;
	int                      mL0;
	int                      mLLast;
	int                      mComment;
	int                      mNbDigName;
	std::string              mPatternTransfo;        
};

cAppli_ImportGCP::cAppli_ImportGCP(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli  (aVArgs,aSpec),
   mPhProj       (*this),
   mL0           (0),
   mLLast        (-1),
   mComment      (-1)
{
}

cCollecSpecArg2007 & cAppli_ImportGCP::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
    return anArgObl
	      <<  Arg2007(mNameFile ,"Name of Input File")
	      <<  Arg2007(mFormat   ,"Format of file as for ex \"SNSXYZSS\" ")
              << mPhProj.DPPointsMeasures().ArgDirOutMand()
           ;
}

cCollecSpecArg2007 & cAppli_ImportGCP::ArgOpt(cCollecSpecArg2007 & anArgObl) 
{
    return anArgObl
       << AOpt2007(mNameGCP,"NameGCP","Name of GCP set")
       << AOpt2007(mNbDigName,"NbDigName","Number of digit for name, if fixed size required (only if int)")
       << AOpt2007(mL0,"NumL0","Num of first line to read",{eTA2007::HDV})
       << AOpt2007(mLLast,"NumLast","Num of last line to read (-1 if at end of file)",{eTA2007::HDV})
       << AOpt2007(mPatternTransfo,"PatName","Pattern for transforming name (first sub-expr)")
       << mPhProj.ArgChSys(true)  // true =>  default init with None
    ;
}


int cAppli_ImportGCP::Exe()
{
    mPhProj.FinishInit();
    std::vector<std::vector<std::string>> aVNames;
    std::vector<std::vector<double>> aVNums;
    std::vector<cPt3dr> aVXYZ,aVWKP;


    MMVII_INTERNAL_ASSERT_tiny(CptSameOccur(mFormat,"XYZN")==1,"Bad format vs NXYZ");

    ReadFilesStruct
    (
        mNameFile, mFormat,
        mL0, mLLast, mComment,
        aVNames,aVXYZ,aVWKP,aVNums
    );


    if (! IsInit(&mNameGCP))
    {
       mNameGCP = FileOfPath(mNameFile,false);
       if (IsPrefixed(mNameGCP))
         mNameGCP = LastPrefix(mNameGCP);
    }

   cChangSysCoordV2 & aChSys = mPhProj.ChSys();

    cSetMesGCP aSetM(mNameGCP);
    for (size_t aK=0 ; aK<aVXYZ.size() ; aK++)
    {
         std::string aName = aVNames.at(aK).at(0);
	 if (IsInit(&mPatternTransfo))
		 aName = PatternKthSubExpr(mPatternTransfo,1,aName);
	 if (IsInit(&mNbDigName))
            aName =   ToStr(cStrIO<int>::FromStr(aName),mNbDigName);

         aSetM.AddMeasure(cMes1GCP(aChSys.Value(aVXYZ[aK]),aName,1.0));
    }

    mPhProj.SaveGCP(aSetM);
    mPhProj.SaveCurSysCoGCP(aChSys.SysTarget());
   

    // delete aChSys;

    return EXIT_SUCCESS;
}


std::vector<std::string>  cAppli_ImportGCP::Samples() const
{
   return 
   {
       "MMVII ImportGCP  2023-10-06_15h31PolarModule.coo  NXYZ Std  NumL0=14 NumLast=34  PatName=\"P\\.(.*)\" NbDigName=4",
       "MMVII ImportGCP  Pannel5mm.obc  NXYZ Std NbDigName=4 ChSys=[LocalPannel]"
   };
}



tMMVII_UnikPApli Alloc_ImportGCP(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_ImportGCP(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_ImportGCP
(
     "ImportGCP",
      Alloc_ImportGCP,
      "Import/Convert basic GCP file in MMVII format",
      {eApF::GCP},
      {eApDT::GCP},
      {eApDT::GCP},
      __FILE__
);


}; // MMVII

