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

class cAppli_ImportTiePMul : public cMMVII_Appli
{
     public :
        cAppli_ImportTiePMul(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);
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
	int                        mL0;
	int                        mLLast;
	int                        mComment;
	std::vector<std::string>   mPatternTransfo;        
};

cAppli_ImportTiePMul::cAppli_ImportTiePMul(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli  (aVArgs,aSpec),
   mPhProj       (*this),
   mL0           (0),
   mLLast        (-1),
   mComment      (-1)
{
}

cCollecSpecArg2007 & cAppli_ImportTiePMul::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
    return anArgObl
	      <<  Arg2007(mNameFile ,"Name of Input File")
	      <<  Arg2007(mFormat   ,"Format of file as for ex \"SNSXYZSS\" ")
              <<  mPhProj.DPMulTieP().ArgDirOutMand()
           ;
}

cCollecSpecArg2007 & cAppli_ImportTiePMul::ArgOpt(cCollecSpecArg2007 & anArgObl) 
{
    return anArgObl
       << AOpt2007(mL0,"NumL0","Num of first line to read",{eTA2007::HDV})
       << AOpt2007(mLLast,"NumLast","Num of last line to read (-1 if at end of file)",{eTA2007::HDV})
       << AOpt2007(mPatternTransfo,"PatName","Pattern for transforming name (first sub-expr)",{{eTA2007::ISizeV,"[2,2]"}})
/*
       << AOpt2007(mNameGCP,"NameGCP","Name of GCP set")
       << AOpt2007(mNbDigName,"NbDigName","Number of digit for name, if fixed size required (only if int)")
       << mPhProj.ArgChSys(true)  // true =>  default init with None
*/
    ;
}


int cAppli_ImportTiePMul::Exe()
{
    mPhProj.FinishInit();
    std::vector<std::vector<std::string>> aVNames;
    std::vector<std::vector<double>> aVNums;
    std::vector<cPt3dr> aVXYZ,aVWKP;


    MMVII_INTERNAL_ASSERT_tiny(CptSameOccur(mFormat,"XYNF")==1,"Bad format vs NXY");

    ReadFilesStruct
    (
        mNameFile, mFormat,
        mL0, mLLast, mComment,
        aVNames,aVXYZ,aVWKP,aVNums,
        false
    );

    std::map<std::string,cVecTiePMul*>  mMapRes;

    for (size_t aK=0 ; aK<aVXYZ.size() ; aK++)
    {
         std::string aNameI   = aVNames.at(aK).at(0);
         if (IsInit(&mPatternTransfo))
            aNameI = ReplacePattern(mPatternTransfo.at(0),mPatternTransfo.at(1),aNameI);

         if (mMapRes.find(aNameI) == mMapRes.end())
         {
             mMapRes[aNameI] = new cVecTiePMul(aNameI);
             StdOut() << "III = " << aNameI << "\n";
         }
         cPt2dr aP2 = Proj(aVXYZ.at(aK));
         int aInd = round_ni(aVNums.at(aK).at(0));
         mMapRes[aNameI]->mVecTPM.push_back(cTiePMul(aP2,aInd));
StdOut() << aNameI << " " << aInd << "\n";
    }
/*

    mPhProj.SaveGCP(aSetM);
    mPhProj.SaveCurSysCoGCP(aChSys.SysTarget());
*/
   


    return EXIT_SUCCESS;
}


std::vector<std::string>  cAppli_ImportTiePMul::Samples() const
{
   return 
   {
          "MMVII ImportTiePMul External-Data/Liaisons.MES NIXY toto NumL0=1 PatName=[\".*\",\"\\$0.tif\"]"
   };
}



tMMVII_UnikPApli Alloc_ImportTiePMu(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_ImportTiePMul(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_ImportTiePMul
(
     "ImportTiePMul",
      Alloc_ImportTiePMu,
      "Import/Convert basic TieP mult file in MMVII format",
      {eApF::TieP},
      {eApDT::TieP},
      {eApDT::TieP},
      __FILE__
);
#if (0)
#endif

}; // MMVII

