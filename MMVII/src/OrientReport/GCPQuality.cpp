#include "MMVII_Ptxd.h"
#include "cMMVII_Appli.h"
#include "MMVII_Geom3D.h"
#include "MMVII_PCSens.h"


/**
   \file GCPQuality.cpp


 */

namespace MMVII
{


/* ==================================================== */
/*                                                      */
/*          cAppli_CalibratedSpaceResection             */
/*                                                      */
/* ==================================================== */

class cAppli_CGPReport : public cMMVII_Appli
{
     public :

        cAppli_CGPReport(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli &);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override;

     private :
        void MakeOneIm(const std::string & aNameIm);

        std::string              mSpecImIn;   ///  Pattern of xml file
        cPhotogrammetricProject  mPhProj;
        cSetMesImGCP             mSetMes;
};

cAppli_CGPReport::cAppli_CGPReport
(
     const std::vector<std::string> &  aVArgs,
     const cSpecMMVII_Appli & aSpec
) :
     cMMVII_Appli  (aVArgs,aSpec),
     mPhProj       (*this)
{
}



cCollecSpecArg2007 & cAppli_CGPReport::ArgObl(cCollecSpecArg2007 & anArgObl)
{
      return anArgObl
              << Arg2007(mSpecImIn,"Pattern/file for images",{{eTA2007::MPatFile,"0"},{eTA2007::FileDirProj}})
              <<  mPhProj.DPPointsMeasures().ArgDirInMand()
              <<  mPhProj.DPOrient().ArgDirInMand()
           ;
}

cCollecSpecArg2007 & cAppli_CGPReport::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{

    return    anArgOpt
	   //  << AOpt2007(mNbTriplets,"NbTriplets","Number max of triplet tested in Ransac",{eTA2007::HDV})
    ;
}


//================================================


void cAppli_CGPReport::MakeOneIm(const std::string & aNameIm)
{
    cSet2D3D aSet32;
    mSetMes.ExtractMes1Im(aSet32,aNameIm);
    cSensorImage*  aCam = mPhProj.LoadSensor(aNameIm,false);

    StdOut() << " aNameImaNameIm " << aNameIm  << " " << aSet32.NbPair() << " Cam=" << aCam << "\n";

    if (1)
    {
         cRGBImage aIma =  cRGBImage::FromFile(aNameIm);

	 for (const auto & aPair : aSet32.Pairs())
	 {
             aIma.DrawCircle(cRGBImage::Green,aPair.mP2,8.0);

	     cPt2dr aProj = aCam->Ground2Image(aPair.mP3);
	     cPt2dr  aVec = (aPair.mP2-aProj);

             aIma.DrawLine(aPair.mP2,aPair.mP2+aVec*800.0,cRGBImage::Red,2.0);
	 }

         aIma.ToFile("ZOOM-"+aNameIm+".tif");
    }

}

int cAppli_CGPReport::Exe()
{
   mPhProj.FinishInit();
   mPhProj.LoadGCP(mSetMes);
   for (const auto & aNameIm : VectMainSet(0))
   {
        mPhProj.LoadIm(mSetMes,aNameIm);
   }

   for (const auto & aNameIm : VectMainSet(0))
   {
        MakeOneIm(aNameIm);
   }



        //MakeOneIm(aNameIm);

    return EXIT_SUCCESS;
}                                       

/* ==================================================== */
/*                                                      */
/*               MMVII                                  */
/*                                                      */
/* ==================================================== */


tMMVII_UnikPApli Alloc_CGPReport(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_CGPReport(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_CGPReport
(
     "ReportGCP",
      Alloc_CGPReport,
      "Reports on GCP projection",
      {eApF::GCP,eApF::Ori},
      {eApDT::GCP,eApDT::Orient},
      {eApDT::Image,eApDT::Xml},
      __FILE__
);





}; // MMVII

