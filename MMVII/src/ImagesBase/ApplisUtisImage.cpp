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
        cAppli_ScaleImage(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec,bool isBasic);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;
     private :
        std::string mNameIn;  ///< Input image name
        std::string mNameOut; ///< Output image name
        std::string mPrefOut; ///< for output image
        tREAL8      mScale;   ///< Reduction factor
        double      mDilate;    ///< "Dilatation" of Gaussian Kernel, 1.0 means Sigma =  1/2 reduction
        bool        mForceGray;  ///< Impose gray image
	bool        mIsBasic;
	tREAL8      mSzSinC;
};

cCollecSpecArg2007 & cAppli_ScaleImage::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
 return
      anArgObl
          <<   Arg2007(mNameIn,"Name of input file",{{eTA2007::MPatFile,"0"},eTA2007::FileImage})
          <<   Arg2007(mScale,"Scaling factor",{})
   ;
}

cCollecSpecArg2007 & cAppli_ScaleImage::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   anArgOpt
             << AOpt2007(mDilate,"Dilate","Dilatation od gaussian",{eTA2007::HDV})
             << AOpt2007(mForceGray,"FG","Force gray image",{eTA2007::HDV})
             << AOpt2007(mPrefOut,"PrefOut","Prefix for output image",{eTA2007::HDV})
             << AOpt2007(mNameOut,CurOP_Out,"Name of out file, Def=\"$PrefOut\"+Input",{})
   ;

   if (! mIsBasic)
      anArgOpt
             << AOpt2007(mSzSinC,"SzSinC","Sz of SinCardinal (default bicubic) for image elarging",{eTA2007::HDV})
      ;

   return anArgOpt;
}

int cAppli_ScaleImage::Exe() 
{
   if (RunMultiSet(0,0))
   {
       return ResultMultiSet();
   }

   if (!IsInit(&mNameOut))
     mNameOut = mPrefOut + Prefix(FileOfPath(mNameIn)) + ".tif";
   cDataFileIm2D aFileIn= cDataFileIm2D::Create(mNameIn,false);

   cIm2D<tREAL4> aImIn(aFileIn.Sz());
   aImIn.Read(aFileIn,cPt2di(0,0));

   cIm2D<tREAL4> aImOut(cPt2di(1,1));

   if (mIsBasic)
       aImOut = aImIn.GaussDeZoom(mScale,3,mDilate);
   else 
       aImOut = aImIn.Scale(mScale,mScale,mSzSinC,mDilate);


   cDataFileIm2D aFileOut = cDataFileIm2D::Create(mNameOut,aFileIn.Type(),aImOut.DIm().Sz(),1);
   aImOut.Write(aFileOut,cPt2di(0,0));

   
   
   return EXIT_SUCCESS;
}

cAppli_ScaleImage:: cAppli_ScaleImage(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec,bool isBasic) :
  cMMVII_Appli (aVArgs,aSpec),
  mPrefOut     ("Scaled_"),
  mDilate      (1.0),
  mForceGray   (false),
  mIsBasic     (isBasic),
  mSzSinC      (-1)
{
}

  //====================  Basic =======================================================
  
tMMVII_UnikPApli Alloc_ScaleImage_Basic(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_ScaleImage(aVArgs,aSpec,true));
}

cSpecMMVII_Appli  TheSpecScaleImage_Basic
(
     "ImageScale_Basic",
      Alloc_ScaleImage_Basic,
      "Down scale an image, basic Gauss-Filter + integer decimation, for backwrad compatibility",
      {eApF::ImProc},
      {eApDT::Image},
      {eApDT::Image},
      __FILE__
);

  //====================  Basic =======================================================
  
tMMVII_UnikPApli Alloc_ScaleImage_Std(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_ScaleImage(aVArgs,aSpec,false));
}

cSpecMMVII_Appli  TheSpecScaleImage_Std
(
     "ImageScale_Std",
      Alloc_ScaleImage_Std,
      "Down scale an image, basic Gauss-Filter + integer decimation, for backwrad compatibility",
      {eApF::ImProc},
      {eApDT::Image},
      {eApDT::Image},
      __FILE__
);








};
