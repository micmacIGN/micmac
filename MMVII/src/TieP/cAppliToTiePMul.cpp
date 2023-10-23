#include "MMVII_PCSens.h"
#include "MMVII_MMV1Compat.h"
#include "MMVII_DeclareCste.h"
#include "MMVII_BundleAdj.h"
#include "MMVII_2Include_Serial_Tpl.h"
#include "MMVII_TplHeap.h"

#include "TieP.h"


/**
   \file cConvCalib.cpp  testgit

   \brief file for conversion between calibration (change format, change model) and tests
*/


namespace MMVII
{


/* **************************************** */
/*                                          */
/*          cMMVII_ImportHom                */
/*                                          */
/* **************************************** */

class cMMVII_ImportHom : public cInterfImportHom
{
      public : 
          void GetHom(cSetHomogCpleIm &,const std::string & aNameIm1,const std::string & aNameIm2) const override;
          bool HasHom(const std::string & aNameIm1,const std::string & aNameIm2) const override;
          cMMVII_ImportHom(cPhotogrammetricProject & aPhProj);
      private : 
	  cPhotogrammetricProject & mPhProj;
};

cMMVII_ImportHom::cMMVII_ImportHom(cPhotogrammetricProject & aPhProj) :
   mPhProj  (aPhProj)
{
}


bool cMMVII_ImportHom::HasHom(const std::string & aNameIm1,const std::string & aNameIm2) const
{
   return ExistFile(mPhProj.NameTiePIn(aNameIm1,aNameIm2));
}

void cMMVII_ImportHom::GetHom(cSetHomogCpleIm & aCple,const std::string & aNameIm1,const std::string & aNameIm2) const
{
    mPhProj.ReadHomol(aCple,aNameIm1,aNameIm2);
}

   /* ********************************************************** */
   /*                                                            */
   /*                 cAppli_OriConvV1V2                         */
   /*                                                            */
   /* ********************************************************** */

class cAppli_ToTiePMul : public cMMVII_Appli
{
     public :
        cAppli_ToTiePMul(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;

     private :
	cPhotogrammetricProject  mPhProj;
	std::string              mSpecIm;
	bool                     mExportConfiXml;
};

cAppli_ToTiePMul::cAppli_ToTiePMul(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli     (aVArgs,aSpec),
   mPhProj          (*this),
   mExportConfiXml  (false)
{
}

cCollecSpecArg2007 & cAppli_ToTiePMul::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
    return anArgObl
	   <<  Arg2007(mSpecIm,"Pattern/file for images",{{eTA2007::MPatFile,"0"},{eTA2007::FileDirProj}})
	   <<  mPhProj.DPTieP().ArgDirInMand()
     ;
}

cCollecSpecArg2007 & cAppli_ToTiePMul::ArgOpt(cCollecSpecArg2007 & anArgObl) 
{
    
    return anArgObl
	   <<  mPhProj.DPMulTieP().ArgDirInOpt("TestReload","Temporay for testing re-merge")
	   << AOpt2007(mExportConfiXml,"ExpConfXml","Export also configuration of multiple point in xml, for info",{eTA2007::HDV})
	     // << AOpt2007(mDownScale,"DS","Downscale, if want to adapt to smaller images",{eTA2007::HDV})
    ;
}



int cAppli_ToTiePMul::Exe()
{
   // we save the output with the same name that input
   mPhProj.DPMulTieP().SetDirOut(mPhProj.DPTieP().DirIn());

   mPhProj.FinishInit();
   cMMVII_ImportHom aIHH(mPhProj);
    
   auto aVName = VectMainSet(0);
   size_t aNbIm = aVName.size();
   StdOut()  << "BEGINING MERGE" << std::endl;
   cComputeMergeMulTieP aSMTP(aVName,&aIHH); // compute the mergin in multiple tie points

   const auto & aPtsGlob = aSMTP.Pts() ;

   StdOut()  << "BEGINING LOG-CONFIG" << std::endl;
   // count the number of point per image
   std::vector<size_t> aVecCptIm(aNbIm,0);
   for (const auto & aPair : aPtsGlob)
   {
        const auto & aConfig  = aPair.first;
        const auto & aPts     = aPair.second;
	int aNbImConf =  aConfig.size();
	int aNbPts = aPts.size() / aNbImConf;
	MMVII_INTERNAL_ASSERT_tiny((aPts.size() % aNbImConf)==0,"Incohernec in PtMul size");
	for (const auto & aKIm : aConfig)
             aVecCptIm.at(aKIm) += aNbPts;
   }

   // init the vector of vector of PMul
   std::vector< cVecTiePMul> aVVPm;
   for (const auto& aName : aVName)
       aVVPm.push_back(cVecTiePMul(aName));

   // give the proper size to VVPM
   for (size_t aKIm=0 ; aKIm<aNbIm ; aKIm++)
   {
      aVVPm.at(aKIm).mVecTPM.resize(aVecCptIm.at(aKIm));
      aVecCptIm.at(aKIm) = 0;
   }

   size_t anIdGlob = 0;  // Id to give number to points
   cGlobConfLogMTP  aConfGlob(aVName,aPtsGlob.size());
   size_t aKConf = 0;

   StdOut()  << "BEGINING FORMAT BY POINT IDENT" << std::endl;
   for (const auto & aPair : aPtsGlob)  
   {
        const auto & aConfig  = aPair.first;  // config of image
        const auto & aPts     = aPair.second;
	int aNbImConf =  aConfig.size();
	int aNbPts = aPts.size() / aNbImConf;
	MMVII_INTERNAL_ASSERT_tiny((aPts.size() % aNbImConf)==0,"Incohernec in PtMul size");

	aConfGlob.KthConf(aKConf).SetIndIm(aConfig);
	aConfGlob.KthConf(aKConf).SetNbPts(aNbPts);
	aConfGlob.KthConf(aKConf).SetIdP0(anIdGlob);

	int aIndP = 0;
	for (int aKPt=0 ; aKPt<aNbPts ; aKPt++)
	{
	     for (const auto & aKIm : aConfig)
	     {
                 cTiePMul aTPM(aPts.at(aIndP++),anIdGlob) ;
                 aVVPm.at(aKIm).mVecTPM.at(aVecCptIm.at(aKIm) ++) = aTPM;
	     }
             anIdGlob++;
	}

	aKConf++;
   }

   StdOut()  << "BEGINING SAVE" << std::endl;
   //  save the multiple tie point of each image
   for (const auto & aVPm : aVVPm)
   {
       mPhProj.SaveMultipleTieP(aVPm,aVPm.mNameIm );
   }

   SaveInFile(aConfGlob,mPhProj.NameConfigMTPOut());

   if (mExportConfiXml)
       SaveInFile(aConfGlob,mPhProj.NameConfigMTPOut("xml"));


   if (0)
   {
      cGlobConfLogMTP  aConf2;
      ReadFromFile(aConf2,mPhProj.NameConfigMTPOut());
      SaveInFile(aConf2,"tata.xml");
   }

   if (mPhProj.DPMulTieP().DirInIsInit())
   {
       StdOut()  << "BEGINING RELOAD" << std::endl;
       cComputeMergeMulTieP * aCM =  AllocStdFromMTP(aVName,mPhProj);
       StdOut()  << "BEGINING TEST INTEGRITY" << std::endl;
       aSMTP.TestEq(*aCM);
       delete aCM;
   }
   StdOut()  << "END" << std::endl;

   return EXIT_SUCCESS;
}

tMMVII_UnikPApli Alloc_ToTiePMul(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_ToTiePMul(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_ToTiePMul
(
     "TieP2PMul",
      Alloc_ToTiePMul,
      "Convert TieP from by-pair  to multiple",
      {eApF::TieP},
      {eApDT::TieP},
      {eApDT::TieP},
      __FILE__
);


}; // MMVII

