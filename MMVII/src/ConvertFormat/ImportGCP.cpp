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
	eTypeSerial      mTypeS;
	int              mNbDigName;

        std::string mNameOut; 
};

cAppli_ImportGCP::cAppli_ImportGCP(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli  (aVArgs,aSpec),
   mPhProj       (*this),
   mL0           (0),
   mLLast        (-1),
   mTypeS        (eTypeSerial::exml)
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
       << AOpt2007(mNameOut,"Out","Name of output file, def=\"NameGCP+xml\"")
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

    mNameOut  = mNameOut + "." +  E2Str(mTypeS);

    // StdOut() << "ggggg " << mPhProj.DPPointsMeasures().DirOutIsInit() << "\n";
    // StdOut() << "ggggg " << mPhProj.DPPointsMeasures().FullDirOut() << "\n";
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

