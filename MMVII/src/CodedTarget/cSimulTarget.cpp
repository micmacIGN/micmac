#include "CodedTarget.h"
#include "include/MMVII_2Include_Serial_Tpl.h"
#include "include/MMVII_Tpl_Images.h"


// Test git branch

namespace MMVII
{

namespace  cNS_CodedTarget
{


/*  *********************************************************** */
/*                                                              */
/*              cAppliSimulCodeTarget                           */
/*                                                              */
/*  *********************************************************** */


class cAppliSimulCodeTarget : public cMMVII_Appli
{
     public :
        cAppliSimulCodeTarget(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);

     private :
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;

	std::string mNameIm;
	std::string mNameTarget;
};


/* *************************************************** */
/*                                                     */
/*              cAppliSimulCodeTarget                  */
/*                                                     */
/* *************************************************** */

cAppliSimulCodeTarget::cAppliSimulCodeTarget(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli  (aVArgs,aSpec)
{
}

cCollecSpecArg2007 & cAppliSimulCodeTarget::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
   return  anArgObl
             <<   Arg2007(mNameIm,"Name of image (first one)",{{eTA2007::MPatFile,"0"},eTA2007::FileImage})
             <<   Arg2007(mNameTarget,"Name of target file")
   ;
}


cCollecSpecArg2007 & cAppliSimulCodeTarget::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return 
	        anArgOpt
   ;
}


int  cAppliSimulCodeTarget::Exe()
{
	/*
   mTestedFilters = SubOfPat<eDCTFilters>(mPatF,true);
   StdOut()  << " IIIIm=" << APBI_NameIm()   << "\n";

   if (RunMultiSet(0,0))  // If a pattern was used, run in // by a recall to itself  0->Param 0->Set
      return ResultMultiSet();

   mPCT.InitFromFile(mNameTarget);
   APBI_ExecAll();  // run the parse file
   */


   return EXIT_SUCCESS;
}
#if (0)
#endif
};


/* =============================================== */
/*                                                 */
/*                       ::                        */
/*                                                 */
/* =============================================== */
/*
using namespace  cNS_CodedTarget;

tMMVII_UnikPApli Alloc_ExtractCodedTarget(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppliExtractCodeTarget(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpecExtractCodedTarget
(
     "CodedTargetExtract",
      Alloc_ExtractCodedTarget,
      "Extract coded target from images",
      {eApF::CodedTarget,eApF::ImProc},
      {eApDT::Image,eApDT::Xml},
      {eApDT::Xml},
      __FILE__
);
*/


};
