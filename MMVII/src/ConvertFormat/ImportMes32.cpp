#include "MMVII_PCSens.h"
#include "MMVII_MMV1Compat.h"
#include "MMVII_DeclareCste.h"
#include "MMVII_BundleAdj.h"
#include "MMVII_2Include_Serial_Tpl.h"


/**
   \file cConvCalib.cpp  testgit

   \brief file for conversion between calibration (change format, change model) and tests
*/


namespace MMVII
{

   /* ********************************************************** */
   /*                                                            */
   /*                 cAppli_ImportM32                           */
   /*                                                            */
   /* ********************************************************** */

class cAppli_ImportM32 : public cMMVII_Appli
{
     public :
        cAppli_ImportM32(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);
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
	int                        mL0;
	int                        mLLast;
	int                        mComment;
	std::string                mNameGCP;
	std::string                mNameImage;
};

cAppli_ImportM32::cAppli_ImportM32(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli  (aVArgs,aSpec),
   mPhProj       (*this),
   mL0           (0),
   mLLast        (-1),
   mComment      (-1),
   mNameGCP      ("Measure3D"),
   mNameImage    (MMVII_NONE)
{
}

cCollecSpecArg2007 & cAppli_ImportM32::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
    return anArgObl
	      <<  Arg2007(mNameFile ,"Name of Input File")
	      <<  Arg2007(mFormat   ,"Format of file as for ex \"SijXYZS\" ")
              <<  mPhProj.DPPointsMeasures().ArgDirOutMand()
           ;
}

cCollecSpecArg2007 & cAppli_ImportM32::ArgOpt(cCollecSpecArg2007 & anArgObl) 
{
    return anArgObl
       << AOpt2007(mL0,"NumL0","Num of first line to read",{eTA2007::HDV})
       << AOpt2007(mLLast,"NumLast","Num of last line to read (-1 if at end of file)",{eTA2007::HDV})
       << AOpt2007(mComment,"Comment","Carac for comment")
       << AOpt2007(mNameGCP,"NameGCP","Name for set of GCP",{eTA2007::HDV})
       << AOpt2007(mNameImage,"NameIm","Name for Image",{eTA2007::HDV})
    ;
}


int cAppli_ImportM32::Exe()
{
	// end init & check
    mPhProj.FinishInit();
    MMVII_INTERNAL_ASSERT_tiny(CptSameOccur(mFormat,"XYZij")==1,"Bad format vs NIXY");

       //  read file
    cReadFilesStruct aRFS(mNameFile, mFormat, mL0, mLLast, mComment);
    aRFS.Read();
       // create structur to import in MMVII representation
    cSetMesGCP aSetGCP(mNameGCP);
    cSetMesPtOf1Im aSetIm(mNameImage);

    for (int aKObj=0 ; aKObj<aRFS.NbRead() ; aKObj++)
    {
         const cPt2dr & aP2 = aRFS.Vij().at(aKObj);
         const cPt3dr & aP3 = aRFS.VXYZ().at(aKObj);

	 std::string aNamePt = std::string("Pt_") + ToStr(aKObj);
	 cMes1GCP aMesGCP(aP3,aNamePt,1.0);
	 cMesIm1Pt aMesIm(aP2,aNamePt,1.0);

	 aSetGCP.AddMeasure(aMesGCP);
         aSetIm.AddMeasure(aMesIm);

    }

    mPhProj.SaveGCP(aSetGCP);
    mPhProj.SaveMeasureIm(aSetIm);

    return EXIT_SUCCESS;
}

std::vector<std::string>  cAppli_ImportM32::Samples() const
{
   return 
   {
          "MMVII toto"
   };
}


tMMVII_UnikPApli Alloc_ImportM32(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_ImportM32(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_ImportM32
(
     "ImportM32",
      Alloc_ImportM32,
      "Import/Convert Set of 3d-2d corresspondances",
      {eApF::GCP},
      {eApDT::GCP},
      {eApDT::GCP},
      __FILE__
);
#if (0)
#endif

}; // MMVII

