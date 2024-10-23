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

     std::string aStrDist = "XXXXXXX";
     int aNb1 = 0;
     int aNb2 = 0;
     cLinesAntiParal1Im  aVL1 ;
     cLinesAntiParal1Im  aVL2 ;
     if (hasL1)
     {
        aVL1 = mPhProj.ReadLines(aNameIma);
        aNb1 = aVL1.mLines.size();
     }
     if (hasL2)
     {
        aVL2 = mPhProj.ReadLinesFolder(mFolder2,aNameIma);
        aNb2 = aVL2.mLines.size();
     }


     if ((aNb1==1) && (aNb2==1))
     {
        cOneLineAntiParal aL1 = aVL1.mLines.at(0);
	cSegment2DCompiled<tREAL8> aSeg1 (aL1.mSeg.P1(),aL1.mSeg.P2());

        cOneLineAntiParal aL2 = aVL2.mLines.at(0);
	cSegment2DCompiled<tREAL8> aSeg2 (aL2.mSeg.P1(),aL2.mSeg.P2());

	if (1)
           aSeg1 = tSegComp2dr(aCal->Undist(aSeg1.P1()), aCal->Undist(aSeg1.P2()));
	if (1)
           aSeg2 = tSegComp2dr(aCal->Undist(aSeg2.P1()), aCal->Undist(aSeg2.P2()));

        tREAL8 aDist = (aSeg1.Dist(aSeg2.P1()) +  aSeg1.Dist(aSeg2.P2())) / 2.0;
        aStrDist = ToStr(aDist);
     }

     AddOneReportCSV(mPrefixCSV,{aNameIma,ToStr(aNb1),ToStr(aNb2),aStrDist});
}


int cAppli_SegImReport::Exe()
{
   mPhProj.FinishInit();
   mSetNames = VectMainSet(0);

   std::string aN1 =  mPhProj.DPPointsMeasures().DirIn() ;
   std::string aN2 =  mFolder2;

   mPrefixCSV = "CmpLines_"+  aN1  + "_" + aN2 ;
   InitReport(mPrefixCSV,"csv",false);
   AddHeaderReportCSV(mPrefixCSV,{"Image","Nb " + aN1 ,"Nb " + aN2,"Dist"});


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

