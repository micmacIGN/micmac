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
/*             cAppliExtractCodeTarget                          */
/*                                                              */
/*  *********************************************************** */

class cAppliExtractCodeTarget : public cMMVII_Appli,
	                        public cAppliParseBoxIm<tREAL4>
{
     public :
        cAppliExtractCodeTarget(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);

     private :
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;

	int ExeOnParsedBox() override;

	void TestFilters();

	std::string mNameTarget;

	cParamCodedTarget  mPCT;
	std::vector<int>   mTestDistSym;
};


/* *************************************************** */
/*                                                     */
/*              cAppliExtractCodeTarget                   */
/*                                                     */
/* *************************************************** */

cAppliExtractCodeTarget::cAppliExtractCodeTarget(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli  (aVArgs,aSpec),
   cAppliParseBoxIm<tREAL4>(*this,true,cPt2di(5000,5000),cPt2di(300,300)), // static_cast<cMMVII_Appli & >(*this))
   mTestDistSym   ({4,8,12})
{
}

cCollecSpecArg2007 & cAppliExtractCodeTarget::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
   // Standard use, we put args of  cAppliParseBoxIm first
   return
         APBI_ArgObl(anArgObl)
             <<   Arg2007(mNameTarget,"Name of target file")
   ;
}
/* But we could also put them at the end
   return
         APBI_ArgObl(anArgObl <<   Arg2007(mNameTarget,"Name of target file"))
   ;
*/

cCollecSpecArg2007 & cAppliExtractCodeTarget::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return APBI_ArgOpt
	  (
	        anArgOpt
                    << AOpt2007(mTestDistSym, "TestDistSym","Dist for testing symetric filter",{eTA2007::HDV,eTA2007::Tuning})
	  );
   ;
}




void  cAppliExtractCodeTarget::TestFilters()
{
     tDataIm &  aDIm = APBI_DIm();

     StdOut() << "SZ "  <<  aDIm.Sz() << " Im=" << APBI_NameIm() << "\n";

     cImGrad<tREAL4>  aImG = Deriche(aDIm,1.0);

     for (const auto & aDist :  mTestDistSym)
     {
          StdOut() << "DDDD " << aDist << "\n";

          cIm2D<tREAL4>  aImBin = ImBinarity(aDIm,aDist/1.5,aDist,1.0);
	  std::string aName = "TestBin_" + ToStr(aDist) + "_" + Prefix(APBI_NameIm()) + ".tif";
	  aImBin.DIm().ToFile(aName);
	  StdOut() << "Done Bin\n";
	  /*
          cIm2D<tREAL4>  aImSym = ImSymetricity(aDIm,aDist/1.5,aDist,1.0);
	  std::string aName = "TestSym_" + ToStr(aDist) + "_" + Prefix(mNameIm) + ".tif";
	  aImSym.DIm().ToFile(aName);
	  StdOut() << "Done Sym\n";

          cIm2D<tREAL4>  aImStar = ImStarity(aImG,aDist/1.5,aDist,1.0);
	  aName = "TestStar_" + ToStr(aDist) + "_" + Prefix(mNameIm) + ".tif";
	  aImStar.DIm().ToFile(aName);
	  StdOut() << "Done Star\n";

          cIm2D<tREAL4>  aImMixte =   aImSym + aImStar * 2.0;
	  aName = "TestMixte_" + ToStr(aDist) + "_" + Prefix(mNameIm) + ".tif";
	  aImMixte.DIm().ToFile(aName);
	  */
     }

}

int cAppliExtractCodeTarget::ExeOnParsedBox()
{
   if (APBI_TestMode())
   {
       TestFilters();
   }
   else
   {
   }

   return EXIT_SUCCESS;
}

int  cAppliExtractCodeTarget::Exe()
{
   StdOut()  << " IIIIm=" << APBI_NameIm() << "\n";

   if (RunMultiSet(0,0))  // If a pattern was used, run in // by a recall to itself  0->Param 0->Set
      return ResultMultiSet();

   mPCT.InitFromFile(mNameTarget);
   APBI_ExecAll();  // run the parse file


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
