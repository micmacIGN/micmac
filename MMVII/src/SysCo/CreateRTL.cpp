#include "MMVII_PCSens.h"
#include "MMVII_MMV1Compat.h"
#include "MMVII_DeclareCste.h"
#include "MMVII_BundleAdj.h"
#include "MMVII_Matrix.h"

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

class cAppli_CreateRTL : public cMMVII_Appli
{
     public :
        cAppli_CreateRTL(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;

        std::vector<std::string>  Samples() const override;
     private :

	cPhotogrammetricProject  mPhProj;

	// Mandatory Arg
	std::string              mSpecIm;
	std::string              mSysIn;
	std::string              mSysOut;

	// Optionall Arg
	cPt3dr           mCenter;
};

cAppli_CreateRTL::cAppli_CreateRTL(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli  (aVArgs,aSpec),
   mPhProj       (*this)
{
}

cCollecSpecArg2007 & cAppli_CreateRTL::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
    return anArgObl
	      <<  Arg2007(mSpecIm ,"Name of Input File",{{eTA2007::MPatFile,"0"},{eTA2007::FileDirProj}})
	      <<  Arg2007(mSysIn  ,"Input coordinate system")
	      <<  Arg2007(mSysOut ,"Output coordinate system")
           ;
}

cCollecSpecArg2007 & cAppli_CreateRTL::ArgOpt(cCollecSpecArg2007 & anArgObl) 
{
    
    return      anArgObl
            <<  mPhProj.DPOrient().ArgDirInOpt()
            <<  mPhProj.DPPointsMeasures().ArgDirInOpt()
            << AOpt2007(mCenter,"Center","Center of RTL Measures",{{eTA2007::HDV}})
            // << AOpt2007(mNameBloc,"NameBloc","Set the name of the bloc ",{{eTA2007::HDV}})
    ;
}


int cAppli_CreateRTL::Exe()
{
    mPhProj.FinishInit();

    cWeightAv<tREAL8,cPt3dr> aAvgSens;
    if (mPhProj.DPOrient().DirInIsInit())
    {
        for (const auto & aNameIm : VectMainSet(0))
        {
	    cSensorImage* aSI = mPhProj. LoadSensor(aNameIm);
	    aAvgSens.Add(1.0,aSI->PseudoCenterOfProj());
	}
    }


    cSetMesImGCP aMesIm;
    cWeightAv<tREAL8,cPt3dr> aAvgGCP;
    if (mPhProj.DPPointsMeasures().DirInIsInit())
    {
	    mPhProj.LoadGCP(aMesIm);
	    for (const auto & aGCP : aMesIm.MesGCP())
                aAvgGCP.Add(1,aGCP.mPt);
    }

    //mPhProj.SaveGCP(aSetM);

    return EXIT_SUCCESS;
}


std::vector<std::string>  cAppli_CreateRTL::Samples() const
{
   return {"MMVII SysCoCreateRTL "};
}



tMMVII_UnikPApli Alloc_CreateRTL(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_CreateRTL(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_CreateRTL
(
     "SysCoCreateRTL",
      Alloc_CreateRTL,
      "Create RTL (local tangent repair)",
      {eApF::SysCo},
      {eApDT::GCP,eApDT::Ori,eApDT::SysCo},
      {eApDT::GCP,eApDT::Ori,eApDT::SysCo},
      __FILE__
);


}; // MMVII

