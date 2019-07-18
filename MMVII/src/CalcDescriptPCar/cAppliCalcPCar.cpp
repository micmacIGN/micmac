#include "include/MMVII_all.h"
#include "include/MMVII_Tpl_Images.h"

namespace MMVII
{


/// 

/**
     Later this class will evolve to offer the same service than MMV1 command.

     For now it is pretty basic, the scale is integer, the algorithm is
     "gaussian filter" + "decimation".

     The purpose was to test visually gaussian pyramid that will be used in
     descriptors (Aime ...)
*/

class cAppliCalcDescPCar : public cMMVII_Appli
{
     public :
        cAppliCalcDescPCar(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;
     private :
        std::string mNameIn;  ///< Input image name
        std::string mNameOut; ///< Output image name
        int         mScale;   ///< Reduction factor
        double      mDilate;    ///< "Dilatation" of Gaussian Kernel, 1.0 means Sigma =  1/2 reduction
        bool        mForceGray;  ///< Impose gray image
};

cCollecSpecArg2007 & cAppliCalcDescPCar::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
 return
      anArgObl
          <<   Arg2007(mNameIn,"Name of input file",{})
          <<   Arg2007(mScale,"Scaling factor",{})
   ;
}

cCollecSpecArg2007 & cAppliCalcDescPCar::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return anArgOpt
             << AOpt2007(mDilate,"Dilate","Dilatation od gaussian",{eTA2007::HDV})
             << AOpt2007(mForceGray,"FG","Force gray image",{eTA2007::HDV})
             << AOpt2007(mNameOut,CurOP_Out,"Name of out file, Def=\"Scaled_\"+Input",{})
   ;
}

int cAppliCalcDescPCar::Exe() 
{
   cGaussianPyramid<tREAL4>  aGP(cGP_Params(cPt2di(400,700),5,5,3));
   
   return EXIT_SUCCESS;
}

cAppliCalcDescPCar:: cAppliCalcDescPCar(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec) :
  cMMVII_Appli (aVArgs,aSpec),
  mDilate      (1.0),
  mForceGray   (false)
{
}

tMMVII_UnikPApli Alloc_CalcDescPCar(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppliCalcDescPCar(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpecCalcDescPCar
(
     "TieP-PCar",
      Alloc_CalcDescPCar,
      "Compute caracteristic points and descriptors, using Aime method",
      {eApF::TieP,eApF::ImProc},
      {eApDT::Image},
      {eApDT::TieP},
      __FILE__
);




};
