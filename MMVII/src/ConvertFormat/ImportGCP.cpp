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
       <<  mPhProj.DPPointsMeasures().ArgDirOutOpt()
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


    cSetMesGCP aSetM(mNameGCP);
    for (size_t aK=0 ; aK<aVXYZ.size() ; aK++)
    {
         aSetM.AddMeasure(cMes1GCP(aVXYZ[aK],aVNames.at(aK).at(0),1.0));
    }

    if (! IsInit(&mNameGCP))
    {
       mNameGCP = FileOfPath(mNameFile,false);
       if (IsPrefixed(mNameGCP))
         mNameGCP = LastPrefix(mNameGCP);
    }

    if (!IsInit(&mNameOut))
       mNameOut = mNameGCP;

    mNameOut  = mNameOut + "." +  E2Str(mTypeS);

    if (mPhProj.DPPointsMeasures().DirOutIsInit())
    {
        mNameOut = mPhProj.DPPointsMeasures().DirOut() + mNameOut;
    }

    if ((mNameOut==mNameFile) && (!IsInit(&mNameOut)))
    {
       MMVII_UnclasseUsEr("Default out would overwrite input file");
    }

    aSetM.ToFile(mNameOut);

    return EXIT_SUCCESS;
}


#if (0)





tMMVII_UnikPApli Alloc_OriConvV1V2(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_OriConvV1V2(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_OriConvV1V2
(
     "OriConvV1V2",
      Alloc_OriConvV1V2,
      "Convert orientation of MMV1  to MMVII",
      {eApF::Ori},
      {eApDT::Orient},
      {eApDT::Orient},
      __FILE__
);
#endif


}; // MMVII

