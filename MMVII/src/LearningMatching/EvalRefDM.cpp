#include "include/MMVII_all.h"
#include "LearnDM.h"


namespace MMVII
{




class cAppliDMEvalRef : public cAppliLearningMatch
{
     public :

        typedef cIm2D<tREAL4>              tImPx;
        typedef cDataIm2D<tREAL4>          tDataImPx;


        typedef cIm2D<tU_INT1>             tImMasq;
        typedef cDataIm2D<tU_INT1>         tDataImMasq;

        cAppliDMEvalRef(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);

     private :
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;

	// -------------- Mandatory args -------------------
	std::string   mNameI1;
	std::string   mNamePx;

	// -------------- Optionnal args -------------------
	double  mDyn;
	double  mMaxDif;
	std::vector<double>  mValues;
	// -------------- Optionnal args -------------------
};

cAppliDMEvalRef::cAppliDMEvalRef(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cAppliLearningMatch  (aVArgs,aSpec),
   mDyn                 (20),
   mMaxDif              (50),
   mValues              ({0.5,1.0,2.0,4.0})
{
}


cCollecSpecArg2007 & cAppliDMEvalRef::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
 return
      anArgObl
          <<   Arg2007(mNameI1,"Name of first image of ref")
          <<   Arg2007(mNamePx,"Name of Px")
   ;
}

cCollecSpecArg2007 & cAppliDMEvalRef::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return anArgOpt
          // << AOpt2007(mStepZ, "StepZ","Step for paralax",{eTA2007::HDV})
   ;
}

int  cAppliDMEvalRef::Exe()
{
    tImPx   aImPxRef   = tImPx::FromFile(Px1FromIm1(mNameI1));
    tImPx   aImPxCalc   = tImPx::FromFile(mNamePx);
    tImMasq aImMasq = tImMasq::FromFile(Masq1FromIm1(mNameI1));

    tDataImPx &   aDRef  = aImPxRef.DIm();
    tDataImPx &   aDCalc = aImPxCalc.DIm();
    tDataImMasq & aDMasq = aImMasq.DIm();

    int aNbDif = round_ni(mDyn*mMaxDif);
    cHistoCumul<double,double> aHC(aNbDif+1);

    for (const auto & aP : aDMasq)
    {
        if (aDMasq.GetV(aP))
	{
            double aDif = std::abs(aDRef.GetV(aP)-aDCalc.GetV(aP));
	    int aIDif = round_ni(aDif*mDyn);
	    aIDif = std::min(aIDif,aNbDif);
	    aHC.AddV(aIDif,1.0);
	}
    }
    aHC.MakeCumul();


    for (const auto &  aV : mValues)
    {
        StdOut() << " V=" << aV
		 << " Bad=" << aHC.PercBads(aV*mDyn)
		 << " ErTh=" << aHC.AvergBounded(aV*mDyn) / mDyn
		 << " ErApod=" << aHC.ApodAverg(aV*mDyn) / mDyn
		 << "\n";
    }
    return EXIT_SUCCESS;
}


/* =============================================== */
/*                                                 */
/*                       ::                        */
/*                                                 */
/* =============================================== */

tMMVII_UnikPApli Alloc_DMEvalRef(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppliDMEvalRef(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpecDMEvalRef
(
     "DM5StatMatch",
      Alloc_DMEvalRef,
      "Make some evaluation of dense Match with a reference",
      {eApF::Match},
      {eApDT::Image},
      {eApDT::ToDef},
      __FILE__
);



};
