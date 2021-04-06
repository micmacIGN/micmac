#include "include/MMVII_all.h"
#include "Formulas_CamStenope.h"

namespace MMVII
{
  // Put all the stuff that dont vocate to be exported in namespce
namespace NS_GenerateCode
{

class cAppli ;  // class for main application

/*  ============================================== */
/*                                                 */
/*             cAppli                              */
/*                                                 */
/*  ============================================== */

class cAppli : public cMMVII_Appli
{
     public :

       // =========== Declaration ========
                 // --- Method to be a MMVII application
        cAppli(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli &);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override;
        // cAppliBenchAnswer BenchAnswer() const override ; ///< Has it a bench, default : no
        // int  ExecuteBench(cParamExeBench &) override ;

     private :
       // =========== Data ========
            // Mandatory args
};


/*  ============================================== */
/*                                                 */
/*              cAppli                             */
/*                                                 */
/*  ============================================== */

cAppli::cAppli
(
    const std::vector<std::string> &  aVArgs,
    const cSpecMMVII_Appli &          aSpec
)  :
   cMMVII_Appli(aVArgs,aSpec)
{
}


cCollecSpecArg2007 & cAppli::ArgObl(cCollecSpecArg2007 & anArgObl)
{
   
   return 
      anArgObl  ;
/*
         << Arg2007(mModeMatch,"Matching mode",{AC_ListVal<eModeEpipMatch>()})
         << Arg2007(mNameIm1,"Name Input Image1",{eTA2007::FileImage})
         << Arg2007(mNameIm2,"Name Input Image1",{eTA2007::FileImage})
   ;
*/
}

cCollecSpecArg2007 & cAppli::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return
      anArgOpt;
/*
         << AOpt2007(mSzTile,"SzTile","Size of tiling used to split computation",{eTA2007::HDV})
         << AOpt2007(mOutPx,CurOP_Out,"Name of Out file, def=Px_+$Im1")
         // -- Tuning
         << AOpt2007(mDoPyram,"DoPyram","Compute the pyramid",{eTA2007::HDV,eTA2007::Tuning})
         << AOpt2007(mDoClip,"DoClip","Compute the clip of images",{eTA2007::HDV,eTA2007::Tuning})
         << AOpt2007(mDoMatch,"DoMatch","Do the matching",{eTA2007::HDV,eTA2007::Tuning})
         << AOpt2007(mDoPurge,"DoPurge","Do we purge the result ?",{eTA2007::HDV,eTA2007::Tuning})
   ;
*/
}


int cAppli::Exe()
{
   return EXIT_SUCCESS;
}

/*
cAppliBenchAnswer cAppli::BenchAnswer() const 
{
   return cAppliBenchAnswer(true,1.0);
}

int  cAppli::ExecuteBench(cParamExeBench & aParam) 
{

   return EXIT_SUCCESS;
}
*/

/*  ============================================= */
/*       ALLOCATION                               */
/*  ============================================= */

tMMVII_UnikPApli Alloc_GenCode(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
   cMMVIIUnivDist<4,3,2> aDist;
   std::cout << aDist.NameModel()  << aDist.OkMonome(true,1,2) << "\n";


   return tMMVII_UnikPApli(new cAppli(aVArgs,aSpec));
}

};

cSpecMMVII_Appli  TheSpecGenSymbDer
(
     "GenCodeSymDer",
      NS_GenerateCode::Alloc_GenCode,
      "Generation of code for symbolic derivatives",
      {eApF::ManMMVII},
      {eApDT::ToDef},
      {eApDT::ToDef},
      __FILE__
);

};

