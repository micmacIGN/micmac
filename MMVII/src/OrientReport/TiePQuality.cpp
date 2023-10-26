#include "MMVII_Ptxd.h"
#include "cMMVII_Appli.h"
#include "MMVII_Geom3D.h"
#include "MMVII_PCSens.h"
#include "MMVII_Tpl_Images.h"


/**
   \file GCPQuality.cpp


 */

namespace MMVII
{

class cAppli_TiePReport : public cMMVII_Appli
{
     public :

        cAppli_TiePReport(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli &);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override;

     private :
             cPhotogrammetricProject  mPhProj;
             std::vector<int>         mPropStat;
	     std::string              mSpecImIn;   ///  Pattern of xml file
             std::vector<std::string> mSetNames;

             cComputeMergeMulTieP *   mCMTP;
             std::string              mPrefixCSV;
             std::string              mPrefixCSVIma;

	     void MakeStatByImage();
};

cAppli_TiePReport::cAppli_TiePReport
(
     const std::vector<std::string> &  aVArgs,
     const cSpecMMVII_Appli & aSpec
) :
     cMMVII_Appli  (aVArgs,aSpec),
     mPhProj       (*this),
     mPropStat     ({50,75}),
     mCMTP         (nullptr)
{
}

cCollecSpecArg2007 & cAppli_TiePReport::ArgObl(cCollecSpecArg2007 & anArgObl)
{
      return     anArgObl
              << Arg2007(mSpecImIn,"Pattern/file for images",{{eTA2007::MPatFile,"0"},{eTA2007::FileDirProj}})
              << mPhProj.DPMulTieP().ArgDirInMand()
              << mPhProj.DPOrient().ArgDirInMand()
      ;
}


cCollecSpecArg2007 & cAppli_TiePReport::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{

    return   anArgOpt
          << AOpt2007(mPropStat,"Perc","Percentil for stat exp",{eTA2007::HDV})
    ;
}

cPt2dr  KthPt(const tPairTiePMult & aPair, int aKIm,int aKPt)
{
   // size_t aMult = Multiplicity(aPair);

   return aPair.second.mVPIm.at(aKIm+Multiplicity(aPair)*aKPt);
}


void cAppli_TiePReport::MakeStatByImage()
{
   InitReport(mPrefixCSVIma,"csv",false);
   AddStdHeaderStatCSV(mPrefixCSVIma,"Image",mPropStat,{"AvgX","AvgY"});

   const auto & aVII  = mCMTP->IndexeOfImages();
   for (size_t aKImGlob = 0 ; aKImGlob<mSetNames.size() ; aKImGlob++)
   {
       const auto & aLInd = aVII.at(aKImGlob);
       cSensorImage * aSensI = mCMTP->VSensors().at(aKImGlob);
       // a litle check on indexe for these complexe structures
       MMVII_INTERNAL_ASSERT_tiny(mSetNames[aKImGlob]==aSensI->NameImage(),"Chek names in Appli_TiePReport::MakeStatByImage");

       cWeightAv<tREAL8,cPt2dr>  aAvg2d;
       cStdStatRes               aStat;

       for (const auto& aPairKC : aLInd)
       {
           size_t aKImLoc =  aPairKC.first;
           const auto & aConfig = aPairKC.second->first;
	   // a litle check on indexe for these complexe structures
           MMVII_INTERNAL_ASSERT_tiny(aKImGlob==(size_t)aConfig.at(aKImLoc),"Check nums in cAppli_TiePReport::MakeStatByImage");

           const auto & aVal = aPairKC.second->second;
	   size_t aNbP = NbPtsMul(*aPairKC.second);
           // size_t aMult = Multiplicity(*aPairKC.second);

	   for (size_t aKp=0 ; aKp<aNbP ; aKp++)
	   {
               // cPt2dr aPt  = aVal.mVPIm.at(aKImLoc + aMult*aKp);
               cPt2dr aPt =  KthPt(*aPairKC.second,aKImLoc,aKp);
               cPt3dr aPGr = aVal.mVPGround.at(aKp);
               cPt2dr aResidu = aSensI->Ground2Image(aPGr) - aPt;

               aStat.Add(Norm2(aResidu));
	       aAvg2d.Add(1.0,aResidu);
	   }

       }

       AddStdStatCSV
       (
          mPrefixCSVIma,mSetNames[aKImGlob],aStat,mPropStat,
          {ToStr(aAvg2d.Average().x()),ToStr(aAvg2d.Average().y())}
       );
       StdOut() << mSetNames[aKImGlob]  << " Avg=" << aStat.Avg() << "\n";
   }
}


int cAppli_TiePReport::Exe()
{
   mPhProj.FinishInit();

   mSetNames = VectMainSet(0);
   mCMTP = AllocStdFromMTP(mSetNames,mPhProj,true,true,true);
   mCMTP->SetPGround();

   mPrefixCSV =  "_Ori-"+  mPhProj.DPOrient().DirIn() +  "_Mes-"+  mPhProj.DPMulTieP().DirIn() ;
   mPrefixCSVIma =  "ByImages" + mPrefixCSV;

   MakeStatByImage();

   delete mCMTP;
   return EXIT_SUCCESS;
}


/* ==================================================== */

tMMVII_UnikPApli Alloc_TiePReport(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_TiePReport(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_TiePReport
(
     "ReportTieP",
      Alloc_TiePReport,
      "Reports on TieP projection",
      {eApF::TieP,eApF::Ori},
      {eApDT::TieP,eApDT::Orient},
      {eApDT::Image,eApDT::Xml},
      __FILE__
);


}; // MMVII

