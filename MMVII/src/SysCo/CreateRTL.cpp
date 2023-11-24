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
	std::string              mNameSysIn;
	std::string              mNameSysOut;

	// Optionall Arg
	cPt3dr           mOrigin;
	tREAL8           mZ0;
        tREAL8           mEpsDer ;
};

cAppli_CreateRTL::cAppli_CreateRTL(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli  (aVArgs,aSpec),
   mPhProj       (*this),
   mEpsDer       (50.0)
{
}

cCollecSpecArg2007 & cAppli_CreateRTL::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
    return anArgObl
	      <<  Arg2007(mSpecIm ,"Name of Input File",{{eTA2007::MPatFile,"0"},{eTA2007::FileDirProj}})
	      <<  Arg2007(mNameSysIn  ,"Input coordinate system")
	      <<  Arg2007(mNameSysOut ,"Output coordinate system")
           ;
}

cCollecSpecArg2007 & cAppli_CreateRTL::ArgOpt(cCollecSpecArg2007 & anArgObl) 
{
    
    return      anArgObl
            <<  mPhProj.DPOrient().ArgDirInOpt()
            <<  mPhProj.DPOrient().ArgDirOutOpt()
            <<  mPhProj.DPPointsMeasures().ArgDirInOpt()
            << AOpt2007(mOrigin,"Origin","Force origin of RTL Measures",{{eTA2007::HDV}})
            << AOpt2007(mZ0,"Z0","Force altitute of RTL Measures",{{eTA2007::HDV}})
            << AOpt2007(mEpsDer,"EpsDer","Epislon 4 computing derivative",{{eTA2007::HDV}})
            // << AOpt2007(mNameBloc,"NameBloc","Set the name of the bloc ",{{eTA2007::HDV}})
    ;
}


int cAppli_CreateRTL::Exe()
{
    mPhProj.FinishInit();

    cWeightAv<tREAL8,cPt3dr> aAvgSens;
    bool isInitSens  =false;
    bool isInitGCP  =false;
    if (mPhProj.DPOrient().DirInIsInit())
    {
        for (const auto & aNameIm : VectMainSet(0))
        {
	    cSensorImage* aSI = mPhProj. LoadSensor(aNameIm);
	    aAvgSens.Add(1.0,aSI->PseudoCenterOfProj());
	    isInitSens = true;
	}
    }


    cSetMesImGCP aMesIm;
    cWeightAv<tREAL8,cPt3dr> aAvgGCP;
    if (mPhProj.DPPointsMeasures().DirInIsInit())
    {
	mPhProj.LoadGCP(aMesIm);
	for (const auto & aGCP : aMesIm.MesGCP())
        {
            aAvgGCP.Add(1,aGCP.mPt);
	    isInitGCP = true;
	}
    }

    if (! IsInit(&mOrigin))
    {
        MMVII_INTERNAL_ASSERT_User(isInitSens||isInitGCP,eTyUEr::eUnClassedError,"No data for init center");

	const cWeightAv<tREAL8,cPt3dr>	 &  aAvgXY  = isInitSens ? aAvgSens : aAvgGCP ;
	const cWeightAv<tREAL8,cPt3dr>	 &  aAvgZ   = isInitGCP ?  aAvgGCP  : aAvgSens;

	mOrigin.x() = aAvgXY.Average().x();
	mOrigin.y() = aAvgXY.Average().y();
	mOrigin.z() =  aAvgZ.Average().z();
    }

    if (IsInit(&mZ0))
       mOrigin.z() = mZ0;

    tPtrSysCo aSysRTL = mPhProj.CreateSysCoRTL(mOrigin,mNameSysIn);
    mPhProj.SaveSysCo(aSysRTL,mNameSysOut);


    tPtrSysCo aSysIn = mPhProj.ReadSysCo(mNameSysIn);
    cChangSysCoordV2  aChSys(aSysIn,aSysRTL,mEpsDer);

    if (mPhProj.DPOrient().DirOutIsInit())
    {
        int aCpt=0;
        for (const auto & aNameIm : VectMainSet(0))
        {
	    aCpt++;
	    cSensorImage* aSIn  = mPhProj.LoadSensor(aNameIm);
	    cSensorImage* aSOut = aSIn->ChangSys(aChSys);

	    mPhProj.SaveSensor(*aSOut);

	    // a bit slow now, probably because way work ChSys by sys-call on proj4
	    if (aCpt%50==0)
	       StdOut () << " Remain  " << VectMainSet(0).size() - aCpt  << "\n";
	    delete aSOut;
	}
    }

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

