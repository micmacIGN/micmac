#include "MMVII_DeclareCste.h"
#include "MMVII_Tpl_Images.h"

namespace MMVII
{


/* ========================== */
/*          cDataIm2D         */
/* ========================== */

///  A Class to compute scaling of one image

/**
     Later this class will evolve to offer the same service than MMV1 command.

     For now it is pretty basic, the scale is integer, the algorithm is
     "gaussian filter" + "decimation".

     The purpose was to test visually gaussian pyramid that will be used in
     descriptors (Aime ...)
*/

class cAppli_ScaleImage : public cMMVII_Appli
{
     public :
        cAppli_ScaleImage(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);
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

cCollecSpecArg2007 & cAppli_ScaleImage::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
 return
      anArgObl
          <<   Arg2007(mNameIn,"Name of input file",{})
          <<   Arg2007(mScale,"Scaling factor",{})
   ;
}

cCollecSpecArg2007 & cAppli_ScaleImage::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return anArgOpt
             << AOpt2007(mDilate,"Dilate","Dilatation od gaussian",{eTA2007::HDV})
             << AOpt2007(mForceGray,"FG","Force gray image",{eTA2007::HDV})
             << AOpt2007(mNameOut,CurOP_Out,"Name of out file, Def=\"Scaled_\"+Input",{})
   ;
}

int cAppli_ScaleImage::Exe() 
{
   if (!IsInit(&mNameOut))
     mNameOut = "Scaled_"+ Prefix(mNameIn) + ".tif";
   cDataFileIm2D aFileIn= cDataFileIm2D::Create(mNameIn,mForceGray);
   cIm2D<tREAL4> aImIn(aFileIn.Sz());
   aImIn.Read(aFileIn,cPt2di(0,0));

   cIm2D<tREAL4> aImOut = aImIn.GaussDeZoom(mScale,3,mDilate);
   cDataFileIm2D aFileOut = cDataFileIm2D::Create(mNameOut,aFileIn.Type(),aImOut.DIm().Sz(),1);
   aImOut.Write(aFileOut,cPt2di(0,0));

   
   
   return EXIT_SUCCESS;
}

cAppli_ScaleImage:: cAppli_ScaleImage(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec) :
  cMMVII_Appli (aVArgs,aSpec),
  mDilate      (1.0),
  mForceGray   (false)
{
}

tMMVII_UnikPApli Alloc_ScaleImage(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_ScaleImage(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpecScaleImage
(
     "ImageScale",
      Alloc_ScaleImage,
      "Down scale an image, basic 4 now",
      {eApF::ImProc},
      {eApDT::Image},
      {eApDT::Image},
      __FILE__
);




};
