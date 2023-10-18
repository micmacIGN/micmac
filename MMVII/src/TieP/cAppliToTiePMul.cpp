#include "MMVII_PCSens.h"
#include "MMVII_MMV1Compat.h"
#include "MMVII_DeclareCste.h"
#include "MMVII_BundleAdj.h"
#include "MMVII_2Include_Serial_Tpl.h"

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

/* **************************************** */
/*                                          */
/*               cTiePMul                   */
/*                                          */
/* **************************************** */

class cTiePMul
{
    public :
        cTiePMul(cPt2dr  aPt,int anIndex);
        cTiePMul();

        cPt2dr mPt;
        int    mId;
};

cTiePMul::cTiePMul(cPt2dr  aPt,int anIndex) :
   mPt    (aPt) ,
   mId    (anIndex)
{
}

cTiePMul::cTiePMul() :
     cTiePMul(cPt2dr::Dummy(),-1)
{
}

void AddData(const cAuxAr2007 & anAux,cTiePMul & aPMul)
{
    AddData(cAuxAr2007("Pt",anAux),aPMul.mPt);
    AddData(cAuxAr2007("Id",anAux),aPMul.mId);
}

/* **************************************** */
/*                                          */
/*               cVecTiePMul                */
/*                                          */
/* **************************************** */

class   cVecTiePMul
{
      public :
          cVecTiePMul(const std::string & anIm);

	  std::string           mNameIm;
	  std::vector<cTiePMul> mVecTPM;
};

cVecTiePMul::cVecTiePMul(const std::string & anIm) :
   mNameIm (anIm)
{
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
};

cAppli_ToTiePMul::cAppli_ToTiePMul(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli  (aVArgs,aSpec),
   mPhProj       (*this)
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
	     // << AOpt2007(mExtV1,"V1Ext","Extension of V1 input file if !) Homol/")
	     // << AOpt2007(mDownScale,"DS","Downscale, if want to adapt to smaller images",{eTA2007::HDV})
           ;
}




int cAppli_ToTiePMul::Exe()
{
   mPhProj.FinishInit();
   cMMVII_ImportHom aIHH(mPhProj);
    
   auto aVName = VectMainSet(0);
   size_t aNbIm = aVName.size();
   cSetMultipleTiePoints aSMTP(aVName,&aIHH);

   const auto & aPtsGlob = aSMTP.Pts() ;

   // count the number of point per image
   std::vector<size_t> aVecCpt(aNbIm,0);
   for (const auto & aPair : aPtsGlob)
   {
        const auto & aConfig  = aPair.first;
        const auto & aPts     = aPair.second;
	int aNbImConf =  aConfig.size();
	int aNbPts = aPts.size() / aNbImConf;
	for (const auto & aKIm : aConfig)
             aVecCpt.at(aKIm) += aNbPts;
   }

   // init the vector of vector of PMul
   std::vector< cVecTiePMul> aVVPm;
   for (const auto& aName : aVName)
       aVVPm.push_back(aName);

   // give the proper size to VVPM
   for (size_t aKIm=0 ; aKIm<aNbIm ; aKIm++)
   {
      aVVPm.at(aKIm).mVecTPM.resize(aVecCpt.at(aKIm));
      aVecCpt.at(aKIm) = 0;
   }

   size_t anIdGlob = 0;
   for (const auto & aPair : aPtsGlob)
   {
        const auto & aConfig  = aPair.first;
        const auto & aPts     = aPair.second;
	int aNbImConf =  aConfig.size();
	int aNbPts = aPts.size() / aNbImConf;
	MMVII_INTERNAL_ASSERT_tiny((aPts.size() % aNbImConf)==0,"Incohernec in PtMul size");

	int aIndP = 0;
	for (int aKPt=0 ; aKPt<aNbPts ; aKPt++)
	{
	     for (const auto & aKIm : aConfig)
	     {
                 cTiePMul aTPM(aPts.at(aIndP++),anIdGlob) ;
                 aVVPm.at(aKIm).mVecTPM.at(aVecCpt.at(aKIm) ++) = aTPM;
	     }
             anIdGlob++;
	}
   }

   for (const auto & aVPm : aVVPm)
   {
       SaveInFile(aVPm.mVecTPM,"PMUL-"+aVPm.mNameIm +".csv");
   }

   StdOut () << "VVVV " << aVecCpt << "\n";

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

