#include "MMVII_Ptxd.h"
#include "MMVII_SysSurR.h"
#include "MMVII_Sensor.h"
#include "MMVII_PCSens.h"
#include "MMVII_Tpl_Images.h"
#include "MMVII_BundleAdj.h"


namespace MMVII
{


/* ==================================================== */
/*                                                      */
/*              cAppli_StackIm                          */
/*                                                      */
/* ==================================================== */

class cAppli_StackIm : public cMMVII_Appli
{
     public :

        typedef cIm2D<tREAL4> tIm;
        typedef cDataIm2D<tREAL4> tDIm;

        cAppli_StackIm(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli &);
	int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override;

     private :

	std::string mSpecImIn;
};

cAppli_StackIm::cAppli_StackIm(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec):
	cMMVII_Appli   (aVArgs,aSpec)
{
}

cCollecSpecArg2007 & cAppli_StackIm::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
      return anArgObl
              << Arg2007(mSpecImIn,"Pattern/file for images",{{eTA2007::MPatFile,"0"},{eTA2007::FileDirProj}})
/*
              <<  mPhProj.DPPointsMeasures().ArgDirInMand()
              <<  mPhProj.DPOrient().ArgDirOutMand()
*/
           ;
}

cCollecSpecArg2007 & cAppli_StackIm::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{

    return anArgOpt
/*
	       << AOpt2007(mDegDist,"DegDist","Degree of distorsion, if model different of linear one")
	       << AOpt2007(mShow,"ShowNP","Show possible names of param for distorsion",{eTA2007::Tuning,eTA2007::HDV})
	       << AOpt2007(mPatParFrozen,"PatFrozen","Pattern for frozen parameters",{eTA2007::PatParamCalib})
	       << AOpt2007(mValFixPP,"ValPPRel","Fix value of PP in relative to image size ([0.5,0.5] for middle)")
	       << AOpt2007(mMedianCalib,"MedianCalib","Export for a median calib for multiple images")
*/
           ;
}


int cAppli_StackIm::Exe()
{
     //for (const auto &  aNameIm : VectMainSet(0))

    auto aVecIm = VectMainSet(0);

    tIm aIm0 = tIm::FromFile(aVecIm.at(0));
    tDIm & aDIm0 = aIm0.DIm();
    aDIm0.InitCste(0);

    for (const auto & aNameIm : aVecIm)
    {
         tIm aImK = tIm::FromFile(aNameIm);

	 AddIn(aDIm0,aImK.DIm());
    }
    DivCsteIn(aDIm0,tREAL4(aVecIm.size()));

    aDIm0.ToFile("Stack.tif");

    return EXIT_SUCCESS;
};




/* ==================================================== */
/*                                                      */
/*                                                      */
/*                                                      */
/* ==================================================== */


tMMVII_UnikPApli Alloc_StackIm(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_StackIm(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_StackIm
(
     "ImageStack",
      Alloc_StackIm,
      "Stack a serie of images",
      {eApF::ImProc},
      {eApDT::Image},
      {eApDT::Image},
      __FILE__
);



}; // MMVII




