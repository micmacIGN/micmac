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
#if (0)

   /* ********************************************************** */
   /*                                                            */
   /*                 cAppli_ImportGCP                           */
   /*                                                            */
   /* ********************************************************** */

class cAppli_ImportMesImGCP : public cMMVII_Appli
{
     public :
        cAppli_ImportMesImGCP(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);
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
};

cAppli_ImportMesImGCP::cAppli_ImportMesImGCP(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli  (aVArgs,aSpec),
   mPhProj       (*this),
   mL0           (0),
   mLLast        (-1),
   mComment      (-1)
{
}

cCollecSpecArg2007 & cAppli_ImportMesImGCP::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
    return anArgObl
	      <<  Arg2007(mNameFile ,"Name of Input File")
	      <<  Arg2007(mFormat   ,"Format of file as for ex \"SNSXYZSS\" ")
              <<  mPhProj.DPPointsMeasures().ArgDirOutMand()
           ;
}

cCollecSpecArg2007 & cAppli_ImportMesImGCP::ArgOpt(cCollecSpecArg2007 & anArgObl) 
{
    return anArgObl
       << AOpt2007(mL0,"NumL0","Num of first line to read",{eTA2007::HDV})
       << AOpt2007(mLLast,"NumLast","Num of last line to read (-1 if at end of file)",{eTA2007::HDV})
    ;
}


int cAppli_ImportTiePMul::Exe()
{
    mPhProj.FinishInit();
    std::vector<std::vector<std::string>> aVNames;
    std::vector<std::vector<double>> aVNums;
    std::vector<cPt3dr> aVXYZ,aVWKP;


    MMVII_INTERNAL_ASSERT_tiny(CptSameOccur(mFormat,"XYNI")==1,"Bad format vs NIXY");

    ReadFilesStruct
    (
        mNameFile, mFormat,
        mL0, mLLast, mComment,
        aVNames,aVXYZ,aVWKP,aVNums,
        false
    );

    size_t  aRankI_InF = mFormat.find('I');
    size_t  aRankP_InF = mFormat.find('N');
    size_t  aRankP = (aRankP_InF<aRankI_InF) ? 0 : 1;
    size_t  aRankI = 1 - aRankP;

    std::map<std::string,cVecTiePMul*>  mMapRes;

    for (size_t aK=0 ; aK<aVXYZ.size() ; aK++)
    {
         bool PIsSel = true;
         std::string aNamePt = aVNames.at(aK).at(aRankP);
         if (IsInit(&mPatPt))
         {
             if (MatchRegex(aNamePt,mPatPt.at(0)))
             {
                   aNamePt=ReplacePattern(mPatPt.at(0),mPatPt.at(1),aNamePt);
             }
             else
                PIsSel = false;
         }

         if (PIsSel)
         {
             std::string aNameI   = aVNames.at(aK).at(aRankI);
             if (IsInit(&mPatIm))
                aNameI = ReplacePattern(mPatIm.at(0),mPatIm.at(1),aNameI);

             if (mMapRes.find(aNameI) == mMapRes.end())
             {
                 mMapRes[aNameI] = new cVecTiePMul(aNameI);
             }
             cPt2dr aP2 = Proj(aVXYZ.at(aK));
             int aInd = cStrIO<int>::FromStr(aNamePt);
             mMapRes[aNameI]->mVecTPM.push_back(cTiePMul(aP2,aInd));
         }
    }

    tNameSet aSetSave;
    for (const auto & [aName,aVecMTp] : mMapRes)
    {
        mPhProj.SaveMultipleTieP(*aVecMTp,aName);
        if (aVecMTp->mVecTPM.size() > mNbMinPt)
           aSetSave.Add(aName);

        delete aVecMTp;
    }
    SaveInFile(aSetSave,mFileSelIm+"."+GlobTaggedNameDefSerial());

    return EXIT_SUCCESS;
}


std::vector<std::string>  cAppli_ImportTiePMul::Samples() const
{
   return 
   {
          "MMVII ImportTiePMul External-Data/Liaisons.MES NIXY toto NumL0=1 PatIm=[\".*\",\"\\$0.tif\"] PatPt=[\"(MES_)(.*)\",\"\\$2\"]"
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
#endif

}; // MMVII

