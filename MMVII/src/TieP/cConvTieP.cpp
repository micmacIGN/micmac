#include "MMVII_PCSens.h"
#include "MMVII_DeclareCste.h"
#include "MMVII_BundleAdj.h"
#include "MMVII_DeclareAllCmd.h"

#include "TieP.h"

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
        std::string              mNameDSByIm;  ///< Name of File, for computing downscale by image
	bool                     mTimingTest;
	std::string              mPatNameDebug;
	std::string              mPostV1;
	bool                     mGenerateMTP;
};

cAppli_TiePConvert::cAppli_TiePConvert(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli  (aVArgs,aSpec),
   mPhProj       (*this),
   mDownScale    (1.0),
   mTimingTest   (false),
   mPatNameDebug ("ZkW@@M"),
   mPostV1       ("txt"),
   mGenerateMTP  (true)
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
	     << AOpt2007(mPostV1,"Post","V1 postifx like txt or dat",{eTA2007::HDV})
	     << AOpt2007(mDownScale,"DS","Downscale, if want to adapt to smaller images",{eTA2007::HDV})
	     << AOpt2007(mNameDSByIm,"UpSByIm","",{eTA2007::HDV})
	     << AOpt2007(mPatNameDebug,"PND","Pattern names for debuging",{eTA2007::Tuning})
	     << AOpt2007(mGenerateMTP,"GenMTP","Generate multiple tie point after conversion",{eTA2007::HDV})
           ;
}


int cAppli_TiePConvert::Exe()
{
   mPhProj.FinishInit();
   cInterfImportHom * aIIH = nullptr;
    
   if (mFormat==eFormatExtern::eMMV1)
   {
       aIIH = cInterfImportHom::CreateImportV1(DirProject(),mExtV1,mPostV1);
   }
   else
   {
       MMVII_UnclasseUsEr("Only mmv1 suported now for tie point import");
   }

   auto aVName = VectMainSet(0);

   tNameSelector  aRegDebug =  AllocRegex(mPatNameDebug);

   cComputeAssociation aCompUpS;
   if (IsInit(& mNameDSByIm))
      aCompUpS = cComputeAssociation::FromFile(mNameDSByIm);

   int aNbExist = 0;
   // parse all pair, in only one sense, as we will merge "AB" with "BA" when both exist
   int aFreqMsg = std::max(1,std::min(100,int(aVName.size())/10));
   for (size_t aK1=0; aK1<aVName.size() ; aK1++)
   {
      if (IsInit(&mPatNameDebug) && (aK1%aFreqMsg==0))
         StdOut() << "Still " << aVName.size() - aK1 << " to process " << std::endl;
      for (size_t aK2=aK1+1; aK2<aVName.size() ; aK2++)
      {
           std::string aN1 = aVName[aK1];
           std::string aN2 = aVName[aK2];
	   OrderMinMax(aN1,aN2);  //  prefer to fix arbitrary order

           bool aDebug = aRegDebug.Match(aN1) && aRegDebug.Match(aN2);

           bool Exist12 = aIIH->HasHom(aN1,aN2);
           bool Exist21 = aIIH->HasHom(aN2,aN1);

	   if (aDebug)  StdOut() << aN1 << "/" << aN2 << " Exist :" << Exist12 << "/" << Exist21 << std::endl;

           if (Exist12 || Exist21) // if one of both exist : create
           {
               aNbExist++;
               std::vector<std::string> aV12({aN1,aN2});
	       // structure to make the fusion
	       cComputeMergeMulTieP aSMTP(aV12,aIIH);

	       const auto & aSetP = aSMTP.Pts(); // get the map config -> points
	       if (aSetP.empty()) // no coherent  pair no config
	       {
	       }
	       else if (aSetP.size()==1)  // some coherent pair
	       {
                   tREAL8 aDS1 = mDownScale;
                   tREAL8 aDS2 = mDownScale;
                   if (IsInit(& mNameDSByIm))
                   {
                      aDS1 = 1.0 / cStrIO<double>::FromStr(aCompUpS.Translate(aN1));
                      aDS2 = 1.0 / cStrIO<double>::FromStr(aCompUpS.Translate(aN2));
   // cComputeAssociation aCompUpS;
                   }
                   const auto & aVP = aSetP.begin()->second.mVPIm;
                   cSetHomogCpleIm  aSetH;
		   // point ar stored "A1 A2 B1 B2 C1 C2 ..."
		   for (size_t  aKP=0 ; aKP<aVP.size() ; aKP+=2)
		   {
                       aSetH.Add(cHomogCpleIm(aVP.at(aKP)/aDS1,aVP.at(aKP+1)/aDS2));
		   }

		   mPhProj.SaveHomol(aSetH,aN1,aN2);


		   if (mTimingTest && (aK1==0)  && (aK2==1))
		   {
                      // on meramptah test :  TIME SAVE 1.38784 TIME READ 0.295475

                      double aT0 = SecFromT0();
                      for (int aTime=0 ;aTime<100; aTime++)
                          mPhProj.SaveHomol(aSetH,aN1,aN2);
                      double aT1 = SecFromT0();
		      StdOut() << "TIME SAVE " << aT1-aT0 << std::endl;

                      for (int aTime=0 ;aTime<100; aTime++)
		           mPhProj.ReadHomol(aSetH,aN1,aN2,mPhProj.DPTieP().FullDirOut());
                      double aT2 = SecFromT0();
		      StdOut() << "TIME READ " << aT2-aT1 << std::endl;
		   }
	       }
	       else  // this case should never happen
	       {
                   MMVII_INTERNAL_ERROR("Incoherence in tie points merge");
	       }

           }
      }
   }

   if (aNbExist==0)
   {
       MMVII_UnclasseUsEr("No file found (if MMV1 : maybe change dat/txt)");
   }

   delete aIIH;

   if (mGenerateMTP)
   {
      cParamCallSys aPCS(cMMVII_Appli::FullBin(),TheSpec_ToTiePMul.Name(),mSpecIm,mPhProj.DPTieP().DirOut());
      ExtSysCall(aPCS,false);
   }
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

