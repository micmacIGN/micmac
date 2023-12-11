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
   /*                 cAppli_ImportGCP                           */
   /*                                                            */
   /* ********************************************************** */

class cAppli_ImportTiePMul : public cMMVII_Appli
{
     public :
        cAppli_ImportTiePMul(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec,bool ModeTieP);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;

        std::vector<std::string>  Samples() const override;
     private :

        bool                     mModeTieP;
	cPhotogrammetricProject  mPhProj;

	// Mandatory Arg
	std::string              mNameFile;
	std::string              mFormat;


	// Optionall Arg
	int                        mL0;
	int                        mLLast;
	int                        mComment;
	std::vector<std::string>   mPatIm;        
	std::vector<std::string>   mPatPt;        
        size_t                     mNbMinPt;
        std::string                mFileSelIm;
        bool                       mNumByConseq;
};

cAppli_ImportTiePMul::cAppli_ImportTiePMul(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec,bool ModeTieP) :
   cMMVII_Appli  (aVArgs,aSpec),
   mModeTieP     (ModeTieP),
   mPhProj       (*this),
   mL0           (0),
   mLLast        (-1),
   mComment      (-1),
   mNbMinPt      (0),
   mFileSelIm    ("ImagesWithTieP"),
   mNumByConseq  (false)
{
}

cCollecSpecArg2007 & cAppli_ImportTiePMul::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
    cCollecSpecArg2007 &
          aRes  =  anArgObl
	      <<  Arg2007(mNameFile ,"Name of Input File")
	      <<  Arg2007(mFormat   ,"Format of file as for ex \"SNSXYZSS\" ")
           ;
 
   if (mModeTieP)
      return      aRes
              <<  mPhProj.DPMulTieP().ArgDirOutMand();
   else
      return      aRes
              <<  mPhProj.DPPointsMeasures().ArgDirOutMand();
}

cCollecSpecArg2007 & cAppli_ImportTiePMul::ArgOpt(cCollecSpecArg2007 & anArgObl) 
{
    cCollecSpecArg2007 &
      aRes =   anArgObl
            << AOpt2007(mL0,"NumL0","Num of first line to read",{eTA2007::HDV})
            << AOpt2007(mLLast,"NumLast","Num of last line to read (-1 if at end of file)",{eTA2007::HDV})
            << AOpt2007(mPatIm,"PatIm","Pattern for transforming name [Pat,Replace]",{{eTA2007::ISizeV,"[2,2]"}})
            << AOpt2007(mPatPt,"PatPt","Pattern for transforming/select pt [Pat,Replace] ",{{eTA2007::ISizeV,"[2,2]"}})
   ;
   if (mModeTieP)
      return      aRes
               << AOpt2007(mNbMinPt,"NbMinPt","Number minimal of point for selected image",{eTA2007::HDV})
               << AOpt2007(mFileSelIm,"FileSel","Name for file saving selected image ",{eTA2007::HDV})
               << AOpt2007(mNumByConseq,"NumByConseq","Numeratation comes implicitely of consecutive ident",{eTA2007::HDV})
            ;
   else
      return      aRes
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

    std::map<std::string,cVecTiePMul*>     mMapTieP;
    std::map<std::string,cSetMesPtOf1Im*>  mMapGCP;


    std::string  aLastNamePt = "";
    int aInd = 0;
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

             cPt2dr aP2 = Proj(aVXYZ.at(aK));

             if (mModeTieP)
             {
                 if (mNumByConseq)
                 {
                     if (aNamePt != aLastNamePt)
                     {
                          aInd++;
                          aLastNamePt = aNamePt;
                     }
                 }
                 else
                 {
                     aInd = cStrIO<int>::FromStr(aNamePt);
                 }

                 if (mMapTieP.find(aNameI) == mMapTieP.end())
                 {
                     mMapTieP[aNameI] = new cVecTiePMul(aNameI);
                 }
                 mMapTieP[aNameI]->mVecTPM.push_back(cTiePMul(aP2,aInd));
             }
             else
             {
                 if (mMapGCP.find(aNameI) == mMapGCP.end())
                 {
                     mMapGCP[aNameI] = new cSetMesPtOf1Im(aNameI);
                 }
                 mMapGCP[aNameI]->AddMeasure(cMesIm1Pt(aP2,aNamePt,1.0));
             }
         }
    }

    if (mModeTieP)
    {
       tNameSet aSetSave;
       for (const auto & [aName,aVecMTp] : mMapTieP)
       {
          mPhProj.SaveMultipleTieP(*aVecMTp,aName);
          if (aVecMTp->mVecTPM.size() > mNbMinPt)
             aSetSave.Add(aName);

          delete aVecMTp;
        }
        SaveInFile(aSetSave,mFileSelIm+"."+GlobTaggedNameDefSerial());
     }
     else
     {
          for (const auto & [aName,aSetMesIm] : mMapGCP)
          {
              mPhProj.SaveMeasureIm(*aSetMesIm);
              delete aSetMesIm;
          }
     }


      return EXIT_SUCCESS;
}


std::vector<std::string>  cAppli_ImportTiePMul::Samples() const
{
   if (mModeTieP)
      return 
      {
          "MMVII ImportTiePMul External-Data/Liaisons.MES NIXY Vexcell NumL0=1 PatIm=[\".*\",\"\\$0.tif\"] PatPt=[\"(MES_)(.*)\",\"\\$2\"]"
      };

    return {};
}

/*********************************************************************/
/*                                                                   */
/*                       ImportTiePMul                               */
/*                                                                   */
/*********************************************************************/

tMMVII_UnikPApli Alloc_ImportTiePMul(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_ImportTiePMul(aVArgs,aSpec,true));
}

cSpecMMVII_Appli  TheSpec_ImportTiePMul
(
     "ImportTiePMul",
      Alloc_ImportTiePMul,
      "Import/Convert basic TieP mult file in MMVII format",
      {eApF::TieP},
      {eApDT::TieP},
      {eApDT::TieP},
      __FILE__
);

/*********************************************************************/
/*                                                                   */
/*                       ImportTiePMul                               */
/*                                                                   */
/*********************************************************************/

tMMVII_UnikPApli Alloc_ImportMesImGCP(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_ImportTiePMul(aVArgs,aSpec,false));
}

cSpecMMVII_Appli  TheSpec_ImportMesImGCP
(
     "ImportMesImGCP",
      Alloc_ImportMesImGCP,
      "Import/Convert basic Mes Im GCP MMVII format",
      {eApF::GCP},
      {eApDT::GCP},
      {eApDT::GCP},
      __FILE__
);





}; // MMVII

