#include "include/MMVII_all.h"
#include "include/MMVII_2Include_Serial_Tpl.h"
#include "LearnDM.h"
//#include "include/MMVII_Tpl_Images.h"



namespace MMVII
{



class cAppliDensifyRefMatch : public cAppliLearningMatch
{
     public :
        typedef cIm2D<tU_INT1>             tImMasq;
        typedef cIm2D<tREAL4>              tImPx;
        typedef cDataIm2D<tU_INT1>         tDataImMasq;
        typedef cDataIm2D<tREAL4>          tDataImPx;

        cAppliDensifyRefMatch(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);

     private :
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;
	// std::vector<std::string>  Samples() const  override;

    
        std::string  mIm1;

           // --- Optionnal ----

};

cAppliDensifyRefMatch::cAppliDensifyRefMatch(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cAppliLearningMatch  (aVArgs,aSpec)
{
}


cCollecSpecArg2007 & cAppliDensifyRefMatch::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
 return
      anArgObl
          <<   Arg2007(mIm1,"Name of input(s) file(s), Im1",{{eTA2007::MPatFile,"0"}})
   ;
}

cCollecSpecArg2007 & cAppliDensifyRefMatch::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return anArgOpt
          // << AOpt2007(mSzTile, "TileSz","Size of tile for spliting computation",{eTA2007::HDV})
          // << AOpt2007(mSaveImFilter,"SIF","Save Image Filter",{eTA2007::HDV,eTA2007::Tuning})
          // << AOpt2007(mCutsParam,"CutParam","Interval Pax + Line of cuts[PxMin,PxMax,Y0,Y1,....]",{{eTA2007::ISizeV,"[3,10000]"}})
          // << AOpt2007(mSaveImFilter,"SIF","Save Image Filter",{eTA2007::HDV})
   ;
}

/*
std::vector<std::string>  cAppliDensifyRefMatch::Samples() const
{
    return std::vector<std::string>
           (
               {
                   "MMVII DM2CalcHistoCarac DMTrain_MDLB2014-Vintage-perfect_Box0Std_LDHAime0.dmp Test",
                   "MMVII DM2CalcHistoCarac DMTrain_MDLB2014.*LDHAime0.dmp AllMDLB2014"
               }
          );

}
*/

int  cAppliDensifyRefMatch::Exe()
{
   // If a multiple pattern, run in // by recall
   if (RunMultiSet(0,0))
      return ResultMultiSet();

   return EXIT_SUCCESS;
}



/* =============================================== */
/*                                                 */
/*                       ::                        */
/*                                                 */
/* =============================================== */

tMMVII_UnikPApli Alloc_DensifyRefMatch(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppliDensifyRefMatch(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpecExtractLearnVecDM
(
     "DM01DensifyRefMatch",
      Alloc_DensifyRefMatch,
      "Create dense map using a sparse one (LIDAR) with or without images",
      {eApF::Match},
      {eApDT::Image},
      {eApDT::Image},
      __FILE__
);



};
