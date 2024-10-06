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

class cAppli_SegImReport : public cMMVII_Appli
{
     public :

        cAppli_SegImReport(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli &);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override;

     private :
             cPhotogrammetricProject  mPhProj;
             std::vector<int>         mPropStat;
	     std::string              mSpecImIn;   ///  Pattern of xml file
             std::vector<std::string> mSetNames;
	     std::string               mFolder2;

             std::string              mPrefixCSV;
             std::string              mPrefixCSVIma;

	     void AddOneImage(const std::string & aNameIma);
};

cAppli_SegImReport::cAppli_SegImReport
(
     const std::vector<std::string> &  aVArgs,
     const cSpecMMVII_Appli & aSpec
) :
     cMMVII_Appli  (aVArgs,aSpec),
     mPhProj       (*this),
     mPropStat     ({50,75})
{
}

cCollecSpecArg2007 & cAppli_SegImReport::ArgObl(cCollecSpecArg2007 & anArgObl)
{
      return     anArgObl
              << Arg2007(mSpecImIn,"Pattern/file for images",{{eTA2007::MPatFile,"0"},{eTA2007::FileDirProj}})
              << mPhProj.DPPointsMeasures().ArgDirInMand()
	      << mPhProj.DPPointsMeasures().ArgDirInMand("Folder for refernce data",&mFolder2)
              << mPhProj.DPOrient().ArgDirInMand()
      ;
}


cCollecSpecArg2007 & cAppli_SegImReport::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{

    return   anArgOpt
          << AOpt2007(mPropStat,"Perc","Percentil for stat exp",{eTA2007::HDV})
    ;
}

// 671_0090.tif

void cAppli_SegImReport::AddOneImage(const std::string & aNameIma)
{
     cPerspCamIntrCalib *  aCal= mPhProj.InternalCalibFromImage(aNameIma);


     bool hasL1 = mPhProj.HasFileLines(aNameIma);
     bool hasL2 = mPhProj.HasFileLinesFolder(mFolder2,aNameIma);

     if (hasL1 && hasL2)
     {
        cLinesAntiParal1Im  aVL1 = mPhProj.ReadLines(aNameIma);
        cLinesAntiParal1Im  aVL2 = mPhProj.ReadLinesFolder(mFolder2,aNameIma);


	if ((aVL1.mLines.size()==1) && (aVL2.mLines.size()==1))
	{
            cOneLineAntiParal aL1 = aVL1.mLines.at(0);
	    tSegComp2dr aSeg1 (aL1.mSeg.P1(),aL1.mSeg.P2());

            cOneLineAntiParal aL2 = aVL2.mLines.at(0);
	    tSegComp2dr aSeg2 (aL2.mSeg.P1(),aL2.mSeg.P2());

	    if (1)
               aSeg1 = tSegComp2dr(aCal->Undist(aSeg1.P1()), aCal->Undist(aSeg1.P2()));
	    if (1)
               aSeg2 = tSegComp2dr(aCal->Undist(aSeg2.P1()), aCal->Undist(aSeg2.P2()));


	    StdOut()  << " DDDD=" << aNameIma << " " << aSeg1.Dist(aSeg2.P1())  << " " << aSeg1.Dist(aSeg2.P2()) << "\n";

	}
     }
     else
     {
	     StdOut() << "MISSING " << aNameIma  << " Im1=" << hasL1 << " Im2=" << hasL2 << " F=" << aCal->F()  << "\n";
     }

//	          cLinesAntiParal1Im  ReadLines(const std::string & aNameIm) const;

	/*
   InitReport(mPrefixCSVIma,"csv",false);
   AddStdHeaderStatCSV(mPrefixCSVIma,"Image",mPropStat,{"AvgX","AvgY"});

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
           // Unused in mode release
           [[maybe_unused]] const auto & aConfig = aPairKC.second->first;
	   // a litle check on indexe for these complexe structures
           MMVII_INTERNAL_ASSERT_tiny(aKImGlob==(size_t)aConfig.at(aKImLoc),"Check nums in cAppli_SegImReport::MakeStatByImage");

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
       StdOut() << mSetNames[aKImGlob]  << " Avg=" << aStat.Avg() << std::endl;
   }
   */
}


int cAppli_SegImReport::Exe()
{
   mPhProj.FinishInit();
   mSetNames = VectMainSet(0);

   /*

   mPrefixCSV =  "_Ori-"+  mPhProj.DPOrient().DirIn() +  "_Mes-"+  mPhProj.DPMulTieP().DirIn() ;
   mPrefixCSVIma =  "ByImages" + mPrefixCSV;
   */

   for (const auto & aNameIma : mSetNames)
       AddOneImage(aNameIma);

   return EXIT_SUCCESS;
}


/* ==================================================== */

tMMVII_UnikPApli Alloc_SegImReport(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_SegImReport(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_SegImReport
(
     "ReportSegIm",
      Alloc_SegImReport,
      "Reports on SegImage comparison",
      {eApF::TieP,eApF::Ori},
      {eApDT::TieP,eApDT::Orient},
      {eApDT::Image,eApDT::Xml},
      __FILE__
);


}; // MMVII

