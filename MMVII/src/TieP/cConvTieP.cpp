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

class cAppli_TiePConvert : public cMMVII_Appli
{
     public :
        cAppli_TiePConvert(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;
     private :
	std::string              mSpecIm;
	eFormatExtern            mFormat;
	cPhotogrammetricProject  mPhProj;
	std::string              mExtV1;
	double                   mDownScale;
};

cAppli_TiePConvert::cAppli_TiePConvert(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli  (aVArgs,aSpec),
   mPhProj       (*this),
   mDownScale    (1.0)
{
}

cCollecSpecArg2007 & cAppli_TiePConvert::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
    return anArgObl
	   <<  Arg2007(mSpecIm,"Pattern/file for images",{{eTA2007::MPatFile,"0"},{eTA2007::FileDirProj}})
	   <<  Arg2007(mFormat,"Format specification" ,{AC_ListVal<eFormatExtern>()})
	   <<  mPhProj.DPTieP().ArgDirOutMand()
     ;
}

cCollecSpecArg2007 & cAppli_TiePConvert::ArgOpt(cCollecSpecArg2007 & anArgObl) 
{
    
    return anArgObl
	     << AOpt2007(mExtV1,"V1Ext","Extension of V1 input file if !) Homol/")
	     << AOpt2007(mDownScale,"DS","Downscale, if want to adapt to smaller images",{eTA2007::HDV})
           ;
}


int cAppli_TiePConvert::Exe()
{
   mPhProj.FinishInit();
   cInterfImportHom * aIIH = nullptr;
    
   if (mFormat==eFormatExtern::eMMV1)
   {
       aIIH = cInterfImportHom::CreateImportV1(DirProject(),mExtV1);
   }
   else
   {
       MMVII_UnclasseUsEr("Only mmv1 suported now for tie point import");
   }

   auto aVName = VectMainSet(0);


   for (size_t aK1=0; aK1<aVName.size() ; aK1++)
   {
      for (size_t aK2=aK1+1; aK2<aVName.size() ; aK2++)
      {
           std::string aN1 = aVName[aK1];
           std::string aN2 = aVName[aK2];
	   OrderMinMax(aN1,aN2);  //  prefer to fix arbitrary order
           bool Exist12 = aIIH->HasHom(aN1,aN2);
           bool Exist21 = aIIH->HasHom(aN2,aN1);
           if (Exist12 || Exist21)
           {
               std::vector<std::string> aV12({aN1,aN2});
	       cSetMultipleTiePoints aSMTP(aV12,aIIH);

	       const auto & aSetP = aSMTP.Pts();
	       if (aSetP.empty())
	       {
	       }
	       else if (aSetP.size()==1)
	       {
                   const auto & aVP = aSetP.begin()->second;
                   cSetHomogCpleIm  aSetH;
		   for (size_t  aKP=0 ; aKP<aVP.size() ; aKP+=2)
		   {
                       aSetH.Add(cHomogCpleIm(aVP.at(aKP)/mDownScale,aVP.at(aKP+1)/mDownScale));
		   }

		   mPhProj.SaveHomol(aSetH,aN1,aN2);
	       }
	       else
	       {
	       }

           }
      }
   }

   delete aIIH;
   return EXIT_SUCCESS;
}


tMMVII_UnikPApli Alloc_TiePConvert(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_TiePConvert(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_TiePConv
(
     "TiePConvert",
      Alloc_TiePConvert,
      "Convert homologous point",
      {eApF::TieP},
      {eApDT::TieP},
      {eApDT::TieP},
      __FILE__
);


}; // MMVII

