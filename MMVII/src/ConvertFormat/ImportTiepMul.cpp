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

        bool                     mModeTieP;  // Mode TieP vs Mode XY
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

	std::string                mImFilter;
        tNameSet                   mSetFilterIm;
	bool                       mWithImFilter;
	cMMVII_Ofs *               mFiltFile;

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
   mNumByConseq  (false),
   mWithImFilter (false),
   mFiltFile     (nullptr)
{
}

cCollecSpecArg2007 & cAppli_ImportTiePMul::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
    cCollecSpecArg2007 &
          aRes  =  anArgObl
	      <<  Arg2007(mNameFile ,"Name of Input File")
	      <<  Arg2007(mFormat   ,"Format of file as for ex \"SINSXYSS\" ")
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
            << AOpt2007(mImFilter,"ImFilter","File/Pattern for selecting images")
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

//          POS=5217769,0001
//     Nbeellem=999999

int cAppli_ImportTiePMul::Exe()
{
   mPhProj.FinishInit();
   MMVII_INTERNAL_ASSERT_tiny(CptSameOccur(mFormat,"XYNI")==1,"Bad format vs NIXY");

   cReadFilesStruct aRFS(mNameFile,mFormat,mL0,mLLast, mComment);

   if (IsInit(&mImFilter))
   {
       mWithImFilter = true;
       mSetFilterIm = SetNameFromString(mImFilter,true);
       mFiltFile = new cMMVII_Ofs("Filtered_" + mNameFile,eFileModeOut::CreateText);
       aRFS.SetMemoLinesInit();
   }

   aRFS.Read();
   const std::vector<std::string> & aVNIm = aRFS.VNameIm();
   const std::vector<std::string> & aVNPt = aRFS.VNamePt();
   const std::vector<cPt3dr> & aVXYZ = aRFS.VXYZ();


   std::map<std::string,cVecTiePMul*>     mMapTieP;
   std::map<std::string,cSetMesPtOf1Im*>  mMapGCP;


   std::string  aLastNamePt = "";
   int aInd = 0;
   // StdOut()  << "Nbeellem=" << aVXYZ.size() << " " << aRFS.NbRead() << "\n";
   for (size_t aK=0 ; aK<aVXYZ.size() ; aK++)
   {
	 // Read name point, and eventually select + transformate it
         bool PIsSel = true;
         std::string aNamePt = aVNPt.at(aK);
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
             std::string aNameI   = aVNIm.at(aK);
             if (IsInit(&mPatIm))
                aNameI = ReplacePattern(mPatIm.at(0),mPatIm.at(1),aNameI);


             cPt2dr aP2 = Proj(aVXYZ.at(aK));
             bool  ImIsSel = true;
             if (mWithImFilter)
                 ImIsSel =  mSetFilterIm.Match(aNameI);

            if (ImIsSel)
            {
                if (mFiltFile )
                    mFiltFile->Ofs() <<  aRFS.VLinesInit().at(aK) << "\n";

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

    delete mFiltFile;

    return EXIT_SUCCESS;
}


std::vector<std::string>  cAppli_ImportTiePMul::Samples() const
{
   if (mModeTieP)
      return 
      {
          "MMVII ImportTiePMul External-Data/Liaisons.MES NIXY Vexcell NumL0=1 PatIm=[\".*\",\"\\$&.tif\"] PatPt=[\"(MES_)(.*)\",\"\\$2\"]"
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
/*                       ImportMesImGCP                              */
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

