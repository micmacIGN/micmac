#include "include/MMVII_all.h"
#include "include/MMVII_2Include_Serial_Tpl.h"
#include "LearnDM.h"
//#include "include/MMVII_Tpl_Images.h"

namespace MMVII
{

class cAppliCalcHistoCarac : public cAppliLearningMatch
{
     public :
        cAppliCalcHistoCarac(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);


     private :
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;

        void AddOneFile(const std::string&,int aKFile,int aNbFile);

         // -- Mandatory args ----
        std::string       mPatHom0;
        std::string       mNameResult;

         // -- Optionnal args ----
        std::string       mPatShowSep;
        bool              mWithCr;

         // -- Internal variables ----
        cStatAllVecCarac* mStats;

};


cAppliCalcHistoCarac::cAppliCalcHistoCarac(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cAppliLearningMatch  (aVArgs,aSpec),
   mWithCr              (true),
   mStats               (nullptr)
{
}


cCollecSpecArg2007 & cAppliCalcHistoCarac::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
 return
      anArgObl
          <<   Arg2007(mPatHom0,"Name of input(s) file(s)",{{eTA2007::MPatFile,"0"}})
          <<   Arg2007(mNameResult,"Name used for results")
   ;
}

cCollecSpecArg2007 & cAppliCalcHistoCarac::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return anArgOpt
          << AOpt2007(mPatShowSep, "ShowSep","Pattern for show separation")
/*
          << AOpt2007(mSzTile, "TileSz","Size of tile for spliting computation",{eTA2007::HDV})
          << AOpt2007(mOverlap,"TileOL","Overlap of tile to limit sides effects",{eTA2007::HDV})
          << AOpt2007(mPatShowCarac,"PSC","Pattern for Showing Caracteristics")
          << AOpt2007(mNb2Select,"Nb2S","Number of point to select, def=all in masq")
          << AOpt2007(mSaveImFilter,"SIF","Save Image Filter",{eTA2007::HDV,eTA2007::Tuning})
          << AOpt2007(mFlagRand,"FlagRand","Images to randomizes, bit of flag [0-3]",{eTA2007::HDV,eTA2007::Tuning})
          // << AOpt2007(mSaveImFilter,"SIF","Save Image Filter",{eTA2007::HDV})
*/
   ;
}

void cAppliCalcHistoCarac::AddOneFile(const std::string& aStr0,int aKFile,int aNbFile)
{
    StdOut() << "****** "   << aStr0  << " : " << aKFile << "/" << aNbFile <<  "   *******\n"; 
    cFileVecCaracMatch aFCV0(HomFromHom0(aStr0,0));
    mStats->AddOneFile(0,aFCV0);
    StdOut() << "   -------------------------------\n";

    cFileVecCaracMatch aFCV1(HomFromHom0(aStr0,1));
    mStats->AddOneFile(1,aFCV1);
    StdOut() << "   -------------------------------\n";


    cFileVecCaracMatch aFCV2(HomFromHom0(aStr0,2));
    mStats->AddOneFile(2,aFCV2);
    StdOut() << "   -------------------------------\n";

   if (mWithCr)
   {
      mStats->AddCr(aFCV0,aFCV1,true);
      mStats->AddCr(aFCV0,aFCV2,false);
      StdOut() << "   ------------ DONE CR-------------------\n";
   }

   if (IsInit(&mPatShowSep))
   {
      mStats->ShowSepar(mPatShowSep,StdOut());
   }

}

int  cAppliCalcHistoCarac::Exe()
{
   SetNamesProject("",mNameResult);
   CreateDirectories(DirVisu(),true);
   CreateDirectories(DirResult(),true);

   mStats = new cStatAllVecCarac(mWithCr);
   int aKFile=0;
   int aNbFile = MainSet0().size();
   for (const auto & aStr : ToVect(MainSet0()))
   {
       AddOneFile(aStr,aKFile,aNbFile);
       aKFile++;
   }
   mStats->SaveHisto(250,DirVisu());
   if (mWithCr)
   {
      mStats->SaveCr(10,DirVisu());
   }
   mStats->MakeCumul();

   {
       cMultipleOfs  aMulOfs(NameReport());
       aMulOfs << "COM=[" << CommandOfMain() << "]\n\n";
       mStats->ShowSepar(".*",aMulOfs);
   }

   mStats->PackForSave();
   SaveInFile(*mStats,FileHisto1Carac(false));


   delete mStats;
   return EXIT_SUCCESS;
}





/* =============================================== */
/*                                                 */
/*                       ::                        */
/*                                                 */
/* =============================================== */

tMMVII_UnikPApli Alloc_CalcHistoCarac(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppliCalcHistoCarac(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpecCalcHistoCarac
(
     "DM2CalcHistoCarac",
      Alloc_CalcHistoCarac,
      "Compute and save histogramm on single caracteristics",
      {eApF::Match},
      {eApDT::FileSys},
      {eApDT::FileSys},
      __FILE__
);



};
