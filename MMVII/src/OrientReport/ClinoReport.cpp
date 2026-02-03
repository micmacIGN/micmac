#include "MMVII_BlocRig.h"

#include "MMVII_Ptxd.h"
#include "cMMVII_Appli.h"
#include "MMVII_Geom3D.h"
#include "MMVII_PCSens.h"
#include "MMVII_Tpl_Images.h"
#include "MMVII_2Include_Serial_Tpl.h"

#include "MMVII_Clino.h"
#include "MMVII_HeuristikOpt.h"


namespace MMVII
{


/* ==================================================== */
/*                                                      */
/*                  cAppli_CernInitRep                  */
/*                                                      */
/* ==================================================== */

class cAppli_ClinoReport : public cMMVII_Appli
{
     public :

        cAppli_ClinoReport(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli &);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override;

        void ProcessOneCam(cSensorCamPC *);
	//std::vector<std::string>  Samples() const override;

     private :
        cPhotogrammetricProject      mPhProj;
        std::string                  mSpecIm;
        cSetMeasureClino             mMesClino;
        bool                         mFirst;
        std::string                  mIdentRepIndiv;
        std::vector<cWeightAv<tREAL8>> mSumDif;
};

cCollecSpecArg2007 & cAppli_ClinoReport::ArgObl(cCollecSpecArg2007 & anArgObl)
{
      return anArgObl
             <<  Arg2007(mSpecIm,"Pattern/file for images", {{eTA2007::MPatFile,"0"},{eTA2007::FileDirProj}}  )
             <<  mPhProj.DPOrient().ArgDirInMand()
             <<  mPhProj.DPMeasuresClino().ArgDirInMand()
             <<  mPhProj.DPClinoMeters().ArgDirInMand()
           ;
}

cCollecSpecArg2007 & cAppli_ClinoReport::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{

    return      anArgOpt
             // << AOpt2007(mTestAlreadyV,"TestAlreadyV","If repair is already verticalized, for test",{{eTA2007::HDV}})
    ;
}

cAppli_ClinoReport::cAppli_ClinoReport
(
     const std::vector<std::string> &  aVArgs,
     const cSpecMMVII_Appli & aSpec
) :
     cMMVII_Appli    (aVArgs,aSpec),
     mPhProj         (*this),
     mFirst          (true),
     mIdentRepIndiv  ("Indiv")
{
}


void cAppli_ClinoReport::ProcessOneCam(cSensorCamPC * aCam)
{
   cPerspCamIntrCalib *  aCalib = aCam->InternalCalib();
   const  cOneMesureClino * aMes =  mMesClino.ClinoDeprecatedMeasureOfImage(aCam->NameImage()); 
   if (aMes==nullptr)
      return;

   // first time to know  the number of column
   if (mFirst)
   {
       mFirst = false;

       std::vector<std::string> aMsg{"Image"};
       for (int aK=0 ; aK<2 ; aK++)
           for (const auto& aNameC : mMesClino.NamesClino())
               aMsg.push_back( ((aK==0) ? "Dif-" : "Val-") + aNameC);

       InitReportCSV(mIdentRepIndiv,"csv",false,aMsg);
       mFirst = false;
       mSumDif.resize(aMes->Angles().size());
   }
   
//    cCalibSetClino aSetC = mPhProj.ReadSetClino(*aCalib,aMes->NamesClino());


   // StdOut() << " * NI=" << aCam->NameImage() << " A=" << aMes->Angles() << "\n";
   cCalibSetClino aSetC = mPhProj.ReadSetClino(*aCalib,mMesClino.NamesClino());


   cPt3dr aVertInLoc =  cPt3dr(0,0,-1);
   cPt3dr aVertInCam =  aCam->Vec_W2L(aVertInLoc);

   std::vector<std::string> aVDif{aCam->NameImage()};
   std::vector<std::string> aVValues;
   for (size_t aK=0 ; aK<aMes->Angles().size() ; aK++)
   {
       cPt2dr aNeed = Proj(aSetC.ClinosCal().at(aK).CamToClino(aVertInCam));
       tREAL8 aDif = std::abs(Teta(aNeed)- aMes->Angles().at(aK));

       // StdOut() << Teta(aNeed) - aMes->Angles().at(aK)  << " " << aDif << "\n";

       aVDif.push_back(ToStr(aDif));
       aVValues.push_back(ToStr(aMes->Angles().at(aK)));

       mSumDif.at(aK).Add(1.0,aDif);
   }
   AddOneReportCSV(mIdentRepIndiv,Append(aVDif,aVValues));
}


int cAppli_ClinoReport::Exe()
{
    mPhProj.FinishInit();  // the final construction of  photogrammetric project manager can only be done now

    mMesClino = mPhProj.ReadMeasureClino();

    for (const auto & aNameIm : VectMainSet(0)) 
    {
         ProcessOneCam(mPhProj.ReadCamPC(aNameIm,DelAuto::Yes));
    }
    for (int aKTime = 0 ; aKTime<2 ; aKTime++)
    {
       std::vector<std::string> aMsg {(aKTime==0) ? "AvgDif Rad"  : "AvgDif DMgon"};
       tREAL8 aMul = (aKTime==0) ? 1.0 : (400.0/(2*M_PI) * 1e3 * 10);
       for (const auto & aAvg : mSumDif)
           aMsg.push_back(ToStr(aMul*aAvg.Average()));

       AddOneReportCSV(mIdentRepIndiv,aMsg);
    }

    return EXIT_SUCCESS;
}                                       

/* ==================================================== */
/*                                                      */
/*               MMVII                                  */
/*                                                      */
/* ==================================================== */

tMMVII_UnikPApli Alloc_ClinoReport(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_ClinoReport(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_ClinoReport
(
      "ReportClino",
      Alloc_ClinoReport,
      "Report on Clino Calibration&Measures",
      {eApF::Ori},
      {eApDT::Orient}, 
      {eApDT::Xml}, 
      __FILE__
);

}; // MMVII

