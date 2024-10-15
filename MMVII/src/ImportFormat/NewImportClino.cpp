#include "MMVII_PCSens.h"
#include "MMVII_MMV1Compat.h"
#include "MMVII_DeclareCste.h"
#include "MMVII_BundleAdj.h"
#include "MMVII_2Include_Serial_Tpl.h"

#include "MMVII_ReadFileStruct.h"

#include "MMVII_util_tpl.h"


/**
   \file ImportClino.cpp

   \brief file for conversion from raw calibration file to  MMVII format

*/


namespace MMVII
{

#if (0)


   /* ********************************************************** */
   /*                                                            */
   /*                 cAppli_ImportClino                         */
   /*                                                            */
   /* ********************************************************** */

class cAppli_ImportClino : public cMMVII_Appli
{
     public :
        cAppli_ImportClino(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;

        std::vector<std::string>  Samples() const override;
     private :

	cPhotogrammetricProject  mPhProj;

	// Mandatory Arg
	std::string              mNameFile;
	std::string              mFormat;

	// Optionall Arg
	cNRFS_ParamRead            mParamRead;

	//   Format specif
	std::string              mNameFieldIm;
	std::string              mNameFieldAngle;
	std::string              mNameSigma;
	std::string              mSpecFormatMand;
	std::string              mSpecFormatTot;

};

cAppli_ImportClino::cAppli_ImportClino(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli    (aVArgs,aSpec),
   mPhProj         (*this),
   mNameFieldIm    ("Im"),
   mNameFieldAngle ("A"),
   mNameSigma      ("S"),
   mSpecFormatMand (mNameFieldIm+mNameFieldAngle+"*" + mNameSigma+"*"),
   mSpecFormatTot  (cNewReadFilesStruct::MakeSpecTot(mSpecFormatMand,""))
{
	// std::map<std::string,int>  aMap{{"2",2}};
}

cCollecSpecArg2007 & cAppli_ImportClino::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
    return anArgObl
	      <<  Arg2007(mNameFile ,"Name of Input File",{eTA2007::FileAny})
	      // <<  Arg2007(mFormat   ,"Format of file as for in spec :  \"" + mSpecFormatTot + "\"")
	      <<  Arg2007(mFormat   ,cNewReadFilesStruct::MsgFormat(mSpecFormatTot))
              <<  mPhProj.DPClinoMeters().ArgDirOutMand()
           ;
}

cCollecSpecArg2007 & cAppli_ImportClino::ArgOpt(cCollecSpecArg2007 & anArgOpt) 
{
    mParamRead.AddArgOpt(anArgOpt);
    return      anArgOpt
    ;
}


int cAppli_ImportClino::Exe()
{
    mPhProj.FinishInit();

    cNewReadFilesStruct aNRFS;
    aNRFS.SetPatternAddType({"^$",mNameFieldAngle,"^$"});
    aNRFS.SetFormat(mFormat,mSpecFormatMand,mSpecFormatTot);
    aNRFS.ReadFile(mNameFile,mParamRead);

    int aNbAngle = aNRFS.ArrityField(mNameFieldAngle);
    int aNbSigma = aNRFS.ArrityField(mNameSigma);

    if ((aNbSigma!=0) && (aNbSigma!=aNbAngle))
    {
        MMVII_UnclasseUsEr("Nb sigma must equal nb angle or 0");
    }


    for (size_t aKL=0 ; aKL<aNRFS.NbLineRead() ; aKL++)
    {
         std::string aNameIm =  aNRFS.GetValue<std::string>(mNameFieldIm,aKL);

	 StdOut() << "II=" << aNameIm << "\n";

    }

    return EXIT_SUCCESS;
}

std::vector<std::string>  cAppli_ImportClino::Samples() const
{
   return 
   {
          "MMVII ImportM32 verif_1B.txt SjiXYZ XingB NumL0=13 NumLast=30 NameIm=SPOT_1B.tif"
   };
}


tMMVII_UnikPApli Alloc_ImportClino(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_ImportClino(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_ImportClino
(
     "ImportClino",
      Alloc_ImportClino,
      "Import/Convert file of clinometers from raw to MMVII format",
      {eApF::Lines},
      {eApDT::Lines},
      {eApDT::Lines},
      __FILE__
);
#endif

}; // MMVII

