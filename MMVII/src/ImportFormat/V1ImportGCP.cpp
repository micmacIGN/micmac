#include "MMVII_PCSens.h"
#include "MMVII_MMV1Compat.h"  // TO SEE
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
   /*                 cAppli_ConvertV1V2_GCPIM                   */
   /*                                                            */
   /* ********************************************************** */

class cAppli_ConvertV1V2_GCPIM : public cMMVII_Appli
{
     public :
        cAppli_ConvertV1V2_GCPIM(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;

     private :
	cPhotogrammetricProject  mPhProj;
	std::string              mNameGCP;
	std::string              mNameIm;
};

cAppli_ConvertV1V2_GCPIM::cAppli_ConvertV1V2_GCPIM(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli  (aVArgs,aSpec),
   mPhProj       (*this)
{
}

cCollecSpecArg2007 & cAppli_ConvertV1V2_GCPIM::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
    return anArgObl
	      <<  Arg2007(mNameIm  ,"Name of V1-image-measure file (\""+MMVII_NONE +"\" if none !)",{eTA2007::FileTagged})
	      <<  Arg2007(mNameGCP ,"Name of V1-GCP file (\""+MMVII_NONE +"\")if none !)",{eTA2007::FileTagged})
              <<  mPhProj.DPPointsMeasures().ArgDirOutMand()
           ;
}

cCollecSpecArg2007 & cAppli_ConvertV1V2_GCPIM::ArgOpt(cCollecSpecArg2007 & anArgOpt) 
{
    return   anArgOpt
          << mPhProj.DPOrient().ArgDirInOpt()
    ;
}

int cAppli_ConvertV1V2_GCPIM::Exe()
{
    mPhProj.FinishInit();

    cSetMesGCP  aMesGCP;
    bool  useGCP=false;
    if (mNameGCP != MMVII_NONE)
    {
        aMesGCP = ImportMesGCPV1(mNameGCP,"FromV1-"+LastPrefix(mNameGCP));
        useGCP = true;
	mPhProj.SaveGCP(aMesGCP);
        // std::string aNameOut = mPhProj.DPPointsMeasures().FullDirOut() + cSetMesGCP::ThePrefixFiles + "_" + mNameGCP;
        // aMesGCP.ToFile(aNameOut);
    }

    std::list<cSetMesPtOf1Im> aLMesIm;

    bool  useIm=false;
    if (mNameIm != MMVII_NONE)
    {
        useIm = true;
        ImportMesImV1(aLMesIm,mNameIm);
        for (const auto & aMes1Im : aLMesIm)
             mPhProj.SaveMeasureIm(aMes1Im);
    }

    if (useGCP && useIm && mPhProj.DPOrient().DirInIsInit())
    {
          cSetMesImGCP aMesImGCP;
          aMesImGCP.AddMes3D(aMesGCP);

          for (const auto & aMesIm : aLMesIm)
          {
               std::string aNameIm =  aMesIm.NameIm();
               cSensorCamPC * aCamPC =  mPhProj.ReadCamPC(aNameIm,true);
               aMesImGCP.AddMes2D(aMesIm,aCamPC);
               // StdOut() << " Nom Im " << aNameIm << " " << aCamPC->InternalCalib()->F() << std::endl;
          }

          StdOut() << "AVSS RESIDUAL " <<  aMesImGCP.AvgSqResidual() << std::endl;
          int aK=0;
          for (const auto & aMesIm : aLMesIm)
          {
               cSet2D3D aSet32;
               aMesImGCP.ExtractMes1Im(aSet32,aMesIm.NameIm());
               cSensorImage * aSens =  aMesImGCP.VSens().at(aK);

               StdOut() << " Im=" << aMesIm.NameIm()  << " AvgRes=" << aSens->AvgSqResidual(aSet32) << std::endl;

               aK++;
          }
    }
    

    return EXIT_SUCCESS;
}

tMMVII_UnikPApli Alloc_ConvertV1V2_GCPIM(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_ConvertV1V2_GCPIM(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_ConvertV1V2_GCPIM
(
     "V1ConvertGCPIm",
      Alloc_ConvertV1V2_GCPIM,
      "Convert Im&GCP measure from V1 to VII format",
      {eApF::GCP},
      {eApDT::GCP},
      {eApDT::GCP},
      __FILE__
);


}; // MMVII

