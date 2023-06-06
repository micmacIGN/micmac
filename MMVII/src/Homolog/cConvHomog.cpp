#include "MMVII_PCSens.h"
#include "MMVII_MMV1Compat.h"
#include "MMVII_DeclareCste.h"
#include "MMVII_BundleAdj.h"

/**
   \file cConvCalib.cpp  testgit

   \brief file for conversion between calibration (change format, change model) and tests
*/


namespace MMVII
{




   /* ********************************************************** */
   /*                                                            */
   /*                 cAppli_OriConvV1V2                         */
   /*                                                            */
   /* ********************************************************** */

class cAppli_HomolConvert : public cMMVII_Appli
{
     public :
        cAppli_HomolConvert(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;
     private :
	std::string              mSpecIm;
	eFormatExtern            mFormat;
	cPhotogrammetricProject  mPhProj;
};

cAppli_HomolConvert::cAppli_HomolConvert(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli(aVArgs,aSpec),
   mPhProj (*this)
{
}

cCollecSpecArg2007 & cAppli_HomolConvert::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
    return anArgObl
	   <<  Arg2007(mSpecIm,"Pattern/file for images",{{eTA2007::MPatFile,"0"},{eTA2007::FileDirProj}})
	   <<  Arg2007(mFormat,"Format specification" ,{AC_ListVal<eFormatExtern>()})
	   <<  mPhProj.DPHomol().ArgDirOutMand()
     ;
}

cCollecSpecArg2007 & cAppli_HomolConvert::ArgOpt(cCollecSpecArg2007 & anArgObl) 
{
    
    return anArgObl
           ;
}


int cAppli_HomolConvert::Exe()
{
    return EXIT_SUCCESS;
}


tMMVII_UnikPApli Alloc_HomolConvert(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_HomolConvert(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_HomolConv
(
     "HomolConvert",
      Alloc_HomolConvert,
      "Convert homologous point",
      {eApF::TieP},
      {eApDT::TieP},
      {eApDT::TieP},
      __FILE__
);


}; // MMVII

