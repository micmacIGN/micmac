#include "CodedTarget.h"
#include "include/MMVII_2Include_Serial_Tpl.h"
#include "include/MMVII_Tpl_Images.h"


// Test git branch

namespace MMVII
{

template<class TypeEl> class  cAppliParseBoxIm
{
    public :
    protected :
        typedef cIm2D<TypeEl>      tIm;
        typedef cDataIm2D<TypeEl>  tDataIm;

	cAppliParseBoxIm(cMMVII_Appli & anAppli,bool IsGray) :
            mBoxTest  (cBox2di::Empty()),
	    mDFI2d    (cDataFileIm2D::Empty()),
	    mIsGray   (IsGray),
            mAppli    (anAppli),
	    mIm       (cPt2di(1,1))
	{
	}

	~cAppliParseBoxIm()
	{
	}

        cCollecSpecArg2007 & APBI_ArgObl(cCollecSpecArg2007 & anArgObl) 
        {
           return
               anArgObl
                   <<   Arg2007(mNameIm,"Name of input file",{{eTA2007::MPatFile,"0"}})
           ;
        }
        cCollecSpecArg2007 & APBI_ArgOpt(cCollecSpecArg2007 & anArgOpt)
        {
                 return anArgOpt
                         << AOpt2007(mBoxTest, "TestBox","Box for testing before runing all",{eTA2007::Tuning})
                  ;
	}

	void APBI_PostInit()
	{
            mDFI2d = cDataFileIm2D::Create(mNameIm,mIsGray);
	}

	tDataIm & APBI_LoadI(const cBox2di & aBox)
	{
            mDFI2d.AssertNotEmpty();
            DIm().Resize(aBox.Sz());
	    DIm().Read(mDFI2d,aBox.P0());

	    return DIm();
	}

	bool APBI_TestMode() const
	{
              return IsInit(&mBoxTest);
	}

	tDataIm & APBI_LoadTestBox() {return APBI_LoadI(mBoxTest);}


	std::string   mNameIm;  // Name of image to parse
	cBox2di       mBoxTest; // Box for quick testing, in case we dont parse all image

    private :
	cAppliParseBoxIm(const cAppliParseBoxIm &) = delete;
	tDataIm & DIm() {return mIm.DIm();}

	cDataFileIm2D  mDFI2d;
	bool           mIsGray;
        cMMVII_Appli & mAppli;
	tIm            mIm;
};


namespace  cNS_CodedTarget
{


/*  *********************************************************** */
/*                                                              */
/*             cAppliExtractCodeTarget                             */
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
   cAppliParseBoxIm<tREAL4>(*this,true), // static_cast<cMMVII_Appli & >(*this))
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
     tDataIm &  aDIm = APBI_LoadTestBox();

     StdOut() << "SZ "  <<  aDIm.Sz() << " Im=" << mNameIm << "\n";

     cImGrad<tREAL4>  aImG = Deriche(aDIm,1.0);

     for (const auto & aDist :  mTestDistSym)
     {
          StdOut() << "DDDD " << aDist << "\n";

          cIm2D<tREAL4>  aImBin = ImBinarity(aDIm,aDist/1.5,aDist,1.0);
	  std::string aName = "TestBin_" + ToStr(aDist) + "_" + Prefix(mNameIm) + ".tif";
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

int  cAppliExtractCodeTarget::Exe()
{
   StdOut()  << " IIIIm=" << mNameIm << "\n";

   if (RunMultiSet(0,0))  // If a pattern was used, run in // by a recall to itself  0->Param 0->Set
      return ResultMultiSet();

   mPCT.InitFromFile(mNameTarget);
   APBI_PostInit();

   StdOut() << "TEST " << APBI_TestMode() << "\n";

   if (APBI_TestMode())
   {
       TestFilters();
   }
   else
   {
   }

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
