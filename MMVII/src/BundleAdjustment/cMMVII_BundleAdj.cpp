#include "BundleAdjustment.h"
#include "MMVII_util_tpl.h"

/**
   \file cAppliBundAdj.cpp

*/


namespace MMVII
{

/* ************************************************************************ */
/*                                                                          */
/*                        cStdWeighterResidual                              */
/*                                                                          */
/* ************************************************************************ */

cStdWeighterResidual::cStdWeighterResidual(tREAL8 aSGlob,tREAL8 aSigAtt,tREAL8 aThr,tREAL8 aExp) :
   mWGlob       (1/Square(aSGlob)),
   mWithAtt     (aSigAtt>0),
   mSig2Att     (Square(aSigAtt)),
   mWithThr     (aThr>0),
   mSig2Thrs    (Square(aThr)),
   mExpS2       (aExp/2.0)
{
}

cStdWeighterResidual::cStdWeighterResidual(const std::vector<tREAL8> & aVect,int aK0) :
    cStdWeighterResidual
    (
        aVect.at(aK0),
	GetDef(aVect,aK0+1,-1.0),
	GetDef(aVect,aK0+2,-1.0),
	GetDef(aVect,aK0+3, 1.0)
    )
{
}

cStdWeighterResidual::cStdWeighterResidual() :
    cStdWeighterResidual({1.0},0)
{
}


tREAL8  cStdWeighterResidual::SingleWOfResidual(const std::vector<tREAL8> & aVResidual) const
{
   tREAL8 aSumSquare = 0;     

   for (auto & aResidual : aVResidual)
       aSumSquare += Square(aResidual);

   if (mWithThr && (aSumSquare > mSig2Thrs))
      return 0.0;

   if (!mWithAtt)
      return  mWGlob;

   return  mWGlob /  (1.0 + std::pow(aSumSquare/mSig2Att,mExpS2) ); 
}

tREAL8  cStdWeighterResidual::SingleWOfResidual(const cPt2dr & aPt) const
{
     return SingleWOfResidual(aPt.ToStdVector());
}

std::vector<tREAL8> cStdWeighterResidual::WeightOfResidual(const tStdVect & aVResidual) const
{
   return std::vector<tREAL8>( aVResidual.size() , SingleWOfResidual(aVResidual) );
}



/* ************************************************************************ */
/*                                                                          */
/*                            cMMVII_BundleAdj                              */
/*                                                                          */
/* ************************************************************************ */

cMMVII_BundleAdj::cMMVII_BundleAdj(cPhotogrammetricProject * aPhp) :
    mPhProj           (aPhp),
    mPhaseAdd         (true),
    mSys              (nullptr),
    mR8_Sys           (nullptr),
    mMesGCP           (nullptr),
    mSigmaGCP         (-1),
    mMTP              (nullptr),
    mSigmaViscAngles  (-1.0),
    mSigmaViscCenter  (-1.0)
{
}

cMMVII_BundleAdj::~cMMVII_BundleAdj() 
{
    delete mSys;
    delete mMesGCP;
    delete mMTP;
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
    AddPoseViscosity();

    OneItere_GCP();
    OneItere_TieP();

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
    AddCalib(aCamPC->InternalCalib());
}

void  cMMVII_BundleAdj::AddCam(const std::string & aNameIm)
{
    AssertPhpAndPhaseAdd();

    cSensorImage * aNewS;
    cSensorCamPC * aSPC;

    mPhProj->LoadSensor(aNameIm,aNewS,aSPC,false);  // false -> NoSVP
    AddSensor(aNewS);

    mVSCPC.push_back(aSPC);  // eventually nullptr, for example with push-broom
    if (aSPC)
       AddCamPC(aSPC);
}
const std::vector<cSensorImage *> &  cMMVII_BundleAdj::VSIm() const  {return mVSIm;}
const std::vector<cSensorCamPC *> &  cMMVII_BundleAdj::VSCPC() const {return mVSCPC;}


void cMMVII_BundleAdj::SetParamFrozenCalib(const std::string & aPattern)
{    
    mPatParamFrozenCalib = aPattern;
}

}; // MMVII

