#include "include/MMVII_all.h"
#include "include/MMVII_Tpl_Images.h"

namespace MMVII
{


/* ========================== */
/*          cDataIm2D         */
/* ========================== */


class cAppli_ScaleImage : public cMMVII_Appli
{
     public :
        cAppli_ScaleImage(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) ;
     private :
        std::string mNameIn;
        std::string mNameOut;
        int         mScale;
        double      mDilate;
        bool        mForceGray;
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
  mDilate      (DefStdDevRessample),
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
