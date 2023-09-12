#include "MMVII_PCSens.h"
#include "MMVII_MMV1Compat.h"
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
	      <<  Arg2007(mNameIm  ,"Name of V1-image-measure file (\""+MMVII_NONE +"\" if none !)")
	      <<  Arg2007(mNameGCP ,"Name of V1-GCP file (\""+MMVII_NONE +"\")if none !)")
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
        aMesGCP = ImportMesGCPV1(mNameGCP,"FromV1-"+Prefix(mNameGCP));
        useGCP = true;
        std::string aNameOut = mPhProj.DPPointsMeasures().FullDirOut() + cSetMesGCP::ThePrefixFiles + "_" + mNameGCP;
        aMesGCP.ToFile(aNameOut);
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
               // StdOut() << " Nom Im " << aNameIm << " " << aCamPC->InternalCalib()->F() << "\n";
          }

          StdOut() << "AVSS RESIDUAL " <<  aMesImGCP.AvgSqResidual() << "\n";
          int aK=0;
          for (const auto & aMesIm : aLMesIm)
          {
               cSet2D3D aSet32;
               aMesImGCP.ExtractMes1Im(aSet32,aMesIm.NameIm());
               cSensorImage * aSens =  aMesImGCP.VSens().at(aK);

               StdOut() << " Im=" << aMesIm.NameIm()  << " AvgRes=" << aSens->AvgSqResidual(aSet32) << "\n";

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
     "ConvertGCPIm",
      Alloc_ConvertV1V2_GCPIM,
      "Convert Im&GCP measure from V1 to VII format",
      {eApF::GCP},
      {eApDT::GCP},
      {eApDT::GCP},
      __FILE__
);



   /* ********************************************************** */
   /*                                                            */
   /*                 cAppli_ImportGCP                           */
   /*                                                            */
   /* ********************************************************** */

class cAppli_ImportGCP : public cMMVII_Appli
{
     public :
        cAppli_ImportGCP(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;
     private :

	cPhotogrammetricProject  mPhProj;

	// Mandatory Arg
	std::string              mNameFile;
	std::string              mFormat;


	// Optionall Arg
	std::string      mNameGCP;
	int              mL0;
	int              mLLast;
	int              mComment;
	int              mNbDigName;

        std::string mNameOut; 
};

cAppli_ImportGCP::cAppli_ImportGCP(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli  (aVArgs,aSpec),
   mPhProj       (*this),
   mL0           (0),
   mLLast        (-1),
   mComment      (-1)
{
}

cCollecSpecArg2007 & cAppli_ImportGCP::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
    return anArgObl
	      <<  Arg2007(mNameFile ,"Name of Input File")
	      <<  Arg2007(mFormat   ,"Format of file as for ex \"SNSXYZSS\" ")
           ;
}

cCollecSpecArg2007 & cAppli_ImportGCP::ArgOpt(cCollecSpecArg2007 & anArgObl) 
{
    
    return anArgObl
       << AOpt2007(mNameGCP,"NameGCP","Name of GCP set")
       << AOpt2007(mNameOut,"Out","Name of output file, def=\"NameGCP+xml/json\"")
       << AOpt2007(mNbDigName,"NbDigName","Number of digit for name, if fixed size required (only if int)")
       << mPhProj.DPPointsMeasures().ArgDirOutOpt()
    ;
}


int cAppli_ImportGCP::Exe()
{
    mPhProj.FinishInit();
    std::vector<std::vector<std::string>> aVNames;
    std::vector<std::vector<double>> aVNums;
    std::vector<cPt3dr> aVXYZ,aVWKP;


    MMVII_INTERNAL_ASSERT_tiny(CptSameOccur(mFormat,"XYZN")==1,"Bad format vs NXYZ");

    ReadFilesStruct
    (
        mNameFile, mFormat,
        mL0, mLLast, mComment,
        aVNames,aVXYZ,aVWKP,aVNums
    );


    if (! IsInit(&mNameGCP))
    {
       mNameGCP = FileOfPath(mNameFile,false);
       if (IsPrefixed(mNameGCP))
         mNameGCP = LastPrefix(mNameGCP);
    }

    if (!IsInit(&mNameOut))
       mNameOut = mNameGCP;

    mNameOut  = mNameOut + "." +  NameDefSerial();

    if (mPhProj.DPPointsMeasures().DirOutIsInit())
    {
        mNameOut = mPhProj.DPPointsMeasures().FullDirOut() + cSetMesGCP::ThePrefixFiles + "_" + mNameOut;
    }

    if ((mNameOut==mNameFile) && (!IsInit(&mNameOut)))
    {
       MMVII_UnclasseUsEr("Default out would overwrite input file");
    }

    cSetMesGCP aSetM(mNameGCP);
    for (size_t aK=0 ; aK<aVXYZ.size() ; aK++)
    {
         std::string aName = aVNames.at(aK).at(0);
	 if (IsInit(&mNbDigName))
            aName =   ToStr(cStrIO<int>::FromStr(aName),mNbDigName);

         aSetM.AddMeasure(cMes1GCP(aVXYZ[aK],aName,1.0));
    }

    aSetM.ToFile(mNameOut);

    return EXIT_SUCCESS;
}




tMMVII_UnikPApli Alloc_ImportGCP(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_ImportGCP(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_ImportGCP
(
     "ImportGCP",
      Alloc_ImportGCP,
      "Import basic GCP file in MicMac format",
      {eApF::GCP},
      {eApDT::GCP},
      {eApDT::GCP},
      __FILE__
);


}; // MMVII

