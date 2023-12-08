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

class cAppli_ChSysCoOri : public cMMVII_Appli
{
     public :
        cAppli_ChSysCoOri(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;

        std::vector<std::string>  Samples() const override;
     private :

	cPhotogrammetricProject  mPhProj;

	// Mandatory Arg
	std::string              mSpecIm;
	std::string              mNameSysIn;
	std::string              mNameSysOut;

	// Optionall Arg
	cPt3dr           mOrigin;
	//CM: unused:  tREAL8           mZ0;
        tREAL8           mEpsDer ;
};

cAppli_ChSysCoOri::cAppli_ChSysCoOri(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli  (aVArgs,aSpec),
   mPhProj       (*this),
   mEpsDer       (200.0)
{
}

cCollecSpecArg2007 & cAppli_ChSysCoOri::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
    return anArgObl
	      <<  Arg2007(mSpecIm ,"Name of Input File",{{eTA2007::MPatFile,"0"},{eTA2007::FileDirProj}})
	      <<  Arg2007(mNameSysOut ,"Output coordinate system")
              <<  mPhProj.DPOrient().ArgDirInMand()
              <<  mPhProj.DPOrient().ArgDirOutMand()
           ;
}

cCollecSpecArg2007 & cAppli_ChSysCoOri::ArgOpt(cCollecSpecArg2007 & anArgObl) 
{
    
    return      anArgObl
            << AOpt2007(mEpsDer,"EpsDer","Epislon 4 computing derivative",{{eTA2007::HDV}})
            << AOpt2007(mNameSysIn,"SysIn","Input system coordindate (def Ori)")
            // << AOpt2007(mNameBloc,"NameBloc","Set the name of the bloc ",{{eTA2007::HDV}})
    ;
}


int cAppli_ChSysCoOri::Exe()
{
    mPhProj.FinishInit();

    tPtrSysCo aSysIn =  IsInit(&aSysIn) ? mPhProj.ReadSysCo(mNameSysIn) :  mPhProj.CurSysCoOri();
    tPtrSysCo aSysOut = mPhProj.ReadSysCo(mNameSysOut);

    cChangSysCoordV2  aChSys(aSysIn,aSysOut,mEpsDer);


    int aNbRem = VectMainSet(0).size();
    for (const auto & aNameIm : VectMainSet(0))
    {
        cSensorImage* aSI = mPhProj.LoadSensor(aNameIm);
        cSensorImage* aSO =  aSI->SensorChangSys(aChSys);
        mPhProj.SaveSensor(*aSO);
        delete aSO;
        aNbRem--;
        if ((aNbRem%50)==0)
           StdOut() << "Remain:  " << aNbRem << std::endl;
    }

    mPhProj.SaveCurSysCoOri(aSysOut);

    return EXIT_SUCCESS;
}


std::vector<std::string>  cAppli_ChSysCoOri::Samples() const
{
   return {"MMVII SysCoCreateRTL "};
}



tMMVII_UnikPApli Alloc_ChSysCo(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_ChSysCoOri(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_ChSysCo
(
     "OriChSysCo",
      Alloc_ChSysCo,
      "Chang coord system of an orientation",
      {eApF::SysCo,eApF::Ori},
      {eApDT::Ori,eApDT::SysCo},
      {eApDT::Ori,eApDT::SysCo},
      __FILE__
);


}; // MMVII

