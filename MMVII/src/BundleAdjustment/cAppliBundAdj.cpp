#include "BundleAdjustment.h"

/**
   \file cAppliBundAdj.cpp

*/


namespace MMVII
{

/* ************************************************************************ */
/*                                                                          */
/*                            cMMVII_BundleAdj                              */
/*                                                                          */
/* ************************************************************************ */

cMMVII_BundleAdj::cMMVII_BundleAdj(cPhotogrammetricProject * aPhp) :
    mPhProj    (aPhp),
    mPhaseAdd  (true),
    mSys       (nullptr),
    mR8_Sys    (nullptr),
    mMesGCP    (nullptr)
{
}

cMMVII_BundleAdj::~cMMVII_BundleAdj() 
{
    delete mSys;
    delete mMesGCP;
    DeleteAllAndClear(mGCP_UK);
}


void cMMVII_BundleAdj::AssertPhaseAdd() 
{
    MMVII_INTERNAL_ASSERT_tiny(mPhaseAdd,"Mix Add and Use of unknown in cMMVII_BundleAdj");
}
void cMMVII_BundleAdj::AssertPhp() 
{
    MMVII_INTERNAL_ASSERT_tiny(mPhProj,"No cPhotogrammetricProject");
}

void cMMVII_BundleAdj::AssertPhpAndPhaseAdd() 
{
	AssertPhaseAdd();
	AssertPhp();
}


void cMMVII_BundleAdj::InitIteration()
{
    mPhaseAdd = false;

    InitItereGCP();
    mR8_Sys = new cResolSysNonLinear<tREAL8>(eModeSSR::eSSR_LsqNormSparse,mSetIntervUK.GetVUnKnowns());

    mSys =  mR8_Sys;
}


void cMMVII_BundleAdj::OneIteration()
{
    if (mPhaseAdd)
    {
        InitIteration();
    }

    if (mPatParamFrozenCalib !="")
    {
        for (const  auto & aPtrCal : mVPCIC)
	{
            mR8_Sys->SetFrozenFromPat(*aPtrCal,mPatParamFrozenCalib,true);
	}
    }

    OneItere_GCP();

    // StdOut() << "SYS=" << mR8_Sys->GetNbObs() << " " <<  mR8_Sys->NbVar() << std::endl;

    const auto & aVectSol = mSys->R_SolveUpdateReset();
    mSetIntervUK.SetVUnKnowns(aVectSol);
}

//================================================================

void cMMVII_BundleAdj::AddCalib(cPerspCamIntrCalib * aCalib)  
{
    AssertPhaseAdd();
    if (! aCalib->UkIsInit())
    {
	  mVPCIC.push_back(aCalib);
	  mSetIntervUK.AddOneObj(aCalib);
    }
}

void cMMVII_BundleAdj::AddSensor(cSensorImage* aSI)
{
    AssertPhaseAdd();
    MMVII_INTERNAL_ASSERT_tiny (!aSI->UkIsInit(),"Multiple add of cam : " + aSI->NameImage());
    mSetIntervUK.AddOneObj(aSI);
    mVSIm.push_back(aSI);

    auto anEq = aSI->EqColinearity(true,10,true);  // WithDer, SzBuf, ReUse
    mVEqCol.push_back(anEq);
}


void cMMVII_BundleAdj::AddCamPC(cSensorCamPC * aCamPC)
{
    AddSensor(aCamPC);
    mVSCPC.push_back(aCamPC);

    AddCalib(aCamPC->InternalCalib());
}

void  cMMVII_BundleAdj::AddCam(const std::string & aNameIm)
{
    AssertPhpAndPhaseAdd();

    cSensorImage * aNewS;
    cSensorCamPC * aSPC;

    mPhProj->LoadSensor(aNameIm,aNewS,aSPC,false);  // false -> NoSVP
    if (aSPC)
       AddCamPC(aSPC);

}
const std::vector<cSensorImage *> &  cMMVII_BundleAdj::VSIm() const  {return mVSIm;}
const std::vector<cSensorCamPC *> &  cMMVII_BundleAdj::VSCPC() const {return mVSCPC;}


void cMMVII_BundleAdj::SetParamFrozenCalib(const std::string & aPattern)
{    
    mPatParamFrozenCalib = aPattern;
}


   /* ********************************************************** */
   /*                                                            */
   /*                 cAppliBundlAdj                             */
   /*                                                            */
   /* ********************************************************** */

class cAppliBundlAdj : public cMMVII_Appli
{
     public :
        cAppliBundlAdj(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;
     private :

	std::string               mSpecImIn;

	std::string               mDataDir;  /// Default Data dir for all

	cPhotogrammetricProject   mPhProj;
	cMMVII_BundleAdj          mBA;

	std::string               mGCPDir;  ///  GCP Data Dir if != mDataDir
	std::vector<double>       mGCPW;
	int                       mNbIter;

	std::string               mPatParamFrozCalib;
};

cAppliBundlAdj::cAppliBundlAdj(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli  (aVArgs,aSpec),
   mDataDir      ("Std"),
   mPhProj       (*this),
   mBA           (&mPhProj),
   mNbIter       (10)
{
}

cCollecSpecArg2007 & cAppliBundlAdj::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
    return anArgObl
              << Arg2007(mSpecImIn,"Pattern/file for images",{{eTA2007::MPatFile,"0"},{eTA2007::FileDirProj}})
              <<  mPhProj.DPOrient().ArgDirInMand()
	      <<  mPhProj.DPOrient().ArgDirOutMand()
           ;
}

cCollecSpecArg2007 & cAppliBundlAdj::ArgOpt(cCollecSpecArg2007 & anArgOpt) 
{
    
    return anArgOpt
               << AOpt2007(mDataDir,"DataDir","Defautl data directories ",{eTA2007::HDV})
	       << AOpt2007(mNbIter,"NbIter","Number of iterations",{eTA2007::HDV})

               << AOpt2007(mGCPDir,"GCPDir","Dir for GCP if != DataDir")
               << AOpt2007(mGCPW,"GCPW","Weithing of GCP if any [SigmaG,SigmaI], SG=0 fix, SG<0 schurr elim, SG>0",{{eTA2007::ISizeV,"[2,2]"}})
	       << AOpt2007(mPatParamFrozCalib,"PPFzCal","Pattern for freezing internal calibration parameters")
           ;
}

int cAppliBundlAdj::Exe()
{
    bool  MeasureAdded = false; 

    SetIfNotInit(mGCPDir,mDataDir);
    mPhProj.DPPointsMeasures().SetDirIn(mGCPDir);

    mPhProj.FinishInit();

    for (const auto &  aNameIm : VectMainSet(0))
    {
         mBA.AddCam(aNameIm);
    }

    if (IsInit(&mPatParamFrozCalib))
    {
        mBA.SetParamFrozenCalib(mPatParamFrozCalib);
    }
	   

    if (IsInit(&mGCPW))
    {
        MeasureAdded = true;
        cSetMesImGCP * aFullMesGCP = new cSetMesImGCP;
	mPhProj.LoadGCP(*aFullMesGCP);

        for (const auto  & aSens : mBA.VSIm())
        {
             mPhProj.LoadIm(*aFullMesGCP,*aSens);
        }
	cSetMesImGCP * aMesGCP = aFullMesGCP->FilterNonEmptyMeasure();
	delete aFullMesGCP;

	mBA.AddGCP(mGCPW,aMesGCP);
    }

    MMVII_INTERNAL_ASSERT_tiny(MeasureAdded,"Not any measure added");

    for (int aKIter=0 ; aKIter<mNbIter ; aKIter++)
    {
        mBA.OneIteration();
    }

    for (auto & aCamPC : mBA.VSCPC())
	mPhProj.SaveCamPC(*aCamPC);


    return EXIT_SUCCESS;
}


tMMVII_UnikPApli Alloc_BundlAdj(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppliBundlAdj(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_OriBundlAdj
(
     "OriBundleAdj",
      Alloc_BundlAdj,
      "Bundle adjusment between images, using several observations/constraint",
      {eApF::Ori},
      {eApDT::Orient},
      {eApDT::Orient},
      __FILE__
);

/*
*/

}; // MMVII

