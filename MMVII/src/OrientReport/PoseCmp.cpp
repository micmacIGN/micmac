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

class cAppli_PoseCmp : public cMMVII_Appli
{
     public :

        cAppli_PoseCmp(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli &);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override;

     private :
             cPhotogrammetricProject  mPhProj;
             std::vector<int>         mPropStat;
	     std::string              mSpecImIn;   ///  Pattern of xml file
             std::vector<std::string> mSetNames;

	     std::string              mOri2;
	     std::vector<std::string> mPatBand;

             // std::string              mPrefixCSV;
             // std::string              mPrefixCSVIma;

	     void MakeStatByImage();
};

cAppli_PoseCmp::cAppli_PoseCmp
(
     const std::vector<std::string> &  aVArgs,
     const cSpecMMVII_Appli & aSpec
) :
     cMMVII_Appli  (aVArgs,aSpec),
     mPhProj       (*this),
     mPropStat     ({50,75})
{
}

cCollecSpecArg2007 & cAppli_PoseCmp::ArgObl(cCollecSpecArg2007 & anArgObl)
{
      return     anArgObl
              << Arg2007(mSpecImIn,"Pattern/file for images",{{eTA2007::MPatFile,"0"},{eTA2007::FileDirProj}})
              << mPhProj.DPOrient().ArgDirInMand()
              << Arg2007(mOri2,"Second orientation folder",{{eTA2007::Input},{eTA2007::Orient}})
      ;
}


cCollecSpecArg2007 & cAppli_PoseCmp::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{

    return   anArgOpt
          << AOpt2007(mPropStat,"Perc","Percentil for stat exp",{eTA2007::HDV})
          << AOpt2007(mPatBand,"PatBand","Pattern for band [Patter,Subts]")
    ;
}



int cAppli_PoseCmp::Exe()
{
   mPhProj.FinishInit();

   mSetNames = VectMainSet(0);

   cDirsPhProj &  aDirOri2 = *mPhProj.NewDPIn(eTA2007::Orient,mOri2);



   std::vector<cSensorCamPC*> aVCam1;
   std::vector<cSensorCamPC*> aVCam2;

   std::string aLastBand = "Xy#@Z-4lj";

   cPt3dr aLastWPK;
   cWeightAv<tREAL8>  aAvgDif;
   cWeightAv<tREAL8>  aAvgRelDif;
   cWeightAv<tREAL8>  aAvgBandRelDif;

   for (size_t aK=0 ; aK<mSetNames.size() ; aK++)
   {
        std::string aName = mSetNames[aK];
        cSensorCamPC * aCam1 = mPhProj.ReadCamPC(aName,true);
        cSensorCamPC * aCam2 = mPhProj.ReadCamPC(aDirOri2,aName,true);

	aVCam1.push_back(aCam1);
	aVCam2.push_back(aCam2);

	auto aP2In1 = aCam1->RelativePose(*aCam2);
	cPt3dr aWPK = aP2In1.Rot().ToWPK() ;
	aAvgDif.Add(1.0,Norm2(aWPK));

	if (aK!=0)
           aAvgRelDif.Add(1.0,Norm2(aWPK-aLastWPK));

        if (IsInit(&mPatBand))
	{
            std::string aNameBand = ReplacePattern(mPatBand.at(0),mPatBand.at(1),aName);
	    bool aNewBand = (aNameBand!=aLastBand);
	    aLastBand = aNameBand;

	    if (aNewBand)
               StdOut() << "NB=" << aName << std::endl;
	    else
               aAvgBandRelDif.Add(1.0,Norm2(aWPK-aLastWPK));
	}

	// StdOut() <<  "Im=" << aName << " Tr="  << aP2In1.Tr()  << " WPK=" << aP2In1.Rot().ToWPK() << std::endl;
	aLastWPK = aWPK;
   }

   // mPrefixCSV =  "_Ori-"+  mPhProj.DPOrient().DirIn() +  "_Mes-"+  mPhProj.DPMulTieP().DirIn() ;
   //
   StdOut() << "AVG DIFF=" << aAvgDif.Average() << std::endl;
   StdOut() << "AVG REL DIFF=" << aAvgRelDif.Average() << std::endl;
   if (aAvgBandRelDif.SW() > 0)
      StdOut() << "AVG BAND REL DIFF=" << aAvgBandRelDif.Average() << std::endl;

   return EXIT_SUCCESS;
}


/* ==================================================== */

tMMVII_UnikPApli Alloc_PoseCmp(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_PoseCmp(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_PoseCmpReport
(
     "ReportPoseCmp",
      Alloc_PoseCmp,
      "Reports on pose comparison",
      {eApF::Ori},
      {eApDT::Orient},
      {eApDT::Csv},
      __FILE__
);


}; // MMVII

