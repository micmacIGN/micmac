#include "CodedTarget.h"
#include "include/MMVII_2Include_Serial_Tpl.h"


namespace MMVII
{

namespace  cNS_CodedTarget
{


/*  *********************************************************** */
/*                                                              */
/*             cAppliExtractCodeTarget                             */
/*                                                              */
/*  *********************************************************** */

class cAppliExtractCodeTarget : public cMMVII_Appli
{
     public :
        cAppliExtractCodeTarget(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);

     private :


        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;

	std::string mNameIm;
	std::string mNameTarget;

	cParamCodedTarget  mPCT;
};


/* *************************************************** */
/*                                                     */
/*              cAppliExtractCodeTarget                   */
/*                                                     */
/* *************************************************** */


cAppliExtractCodeTarget::cAppliExtractCodeTarget(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli  (aVArgs,aSpec)
{
}

cCollecSpecArg2007 & cAppliExtractCodeTarget::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
 return
      anArgObl
          <<   Arg2007(mNameIm,"Name of input file",{{eTA2007::MPatFile,"0"}})
          <<   Arg2007(mNameTarget,"Name of target file",{{eTA2007::MPatFile,"0"}})
   ;
}

cCollecSpecArg2007 & cAppliExtractCodeTarget::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return anArgOpt
          //<< AOpt2007(mPCT.NbRedond(), "Redund","Number of repetition inside a circle",{eTA2007::HDV})
          //<< AOpt2007(mPCT.NbCircle(), "NbC","Number of circles",{eTA2007::HDV})
   ;
}


int  cAppliExtractCodeTarget::Exe()
{
   if (RunMultiSet(0,0))  // If a pattern was used, run in // by a recall to itself  0->Param 0->Set
      return ResultMultiSet();

   mPCT.InitFromFile(mNameTarget);
	/*

   for (int aNum=0 ; aNum<mPCT.NbCodeAvalaible() ; aNum+=mPerGen)
   {
      cCodesOf1Target aCodes = mPCT.CodesOfNum(aNum);
      aCodes.Show();
      tImTarget aImT= mPCT.MakeIm(aCodes);
      
      std::string aName = "Target_" + ToStr(aNum) + ".tif";
      aImT.DIm().ToFile(aName);
      // FakeUseIt(aCodes);
   }

   SaveInFile(mPCT,"Target_Spec.xml");
   */

   return EXIT_SUCCESS;
}
};


/* =============================================== */
/*                                                 */
/*                       ::                        */
/*                                                 */
/* =============================================== */
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


};
