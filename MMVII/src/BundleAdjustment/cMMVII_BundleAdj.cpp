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
    mBlRig            (nullptr),
    mSigmaViscAngles  (-1.0),
    mSigmaViscCenter  (-1.0)
{
}

cMMVII_BundleAdj::~cMMVII_BundleAdj() 
{
    mSetIntervUK.SIUK_Reset();
    delete mSys;
    delete mMesGCP;
    delete mMTP;
    delete mBlRig;
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
    // if it's first step, alloc ressources
    if (mPhaseAdd)
    {
        InitIteration();
    }


    // ================================================
    //  [1]   Add "Hard" constraint 
    // ================================================

    // if necessary, fix frozen parameters of internal calibration
    if (mPatParamFrozenCalib !="")
    {
        for (const  auto & aPtrCal : mVPCIC)
	{
            mR8_Sys->SetFrozenFromPat(*aPtrCal,mPatParamFrozenCalib,true);
	}
    }

    // if necessary, fix frozen centers of external calibration
    if (mPatFrozenCenter !="")
    {
        tNameSelector aSel =   AllocRegex(mPatFrozenCenter);
        for (const auto & aPtrCam : mVSCPC)
        {
            if ((aPtrCam != nullptr)  && aSel.Match(aPtrCam->NameImage()))
	    {
                 mR8_Sys->SetFrozenVarCurVal(*aPtrCam,aPtrCam->Center());
	    }
        }
    }

    if (mBlRig) // RIGIDBLOC
    {
        mBlRig->SetFrozenVar(*mR8_Sys);
    }

    // ================================================
    //  [2]   Add "Soft" constraint 
    // ================================================

    // if necessary, add some "viscosity" on poses 
    AddPoseViscosity();


    // ================================================
    //  [3]   Add compensation measures
    // ================================================


    OneItere_GCP();   // add GCP informations
    OneItere_TieP();  // ad tie-points information
		      //

    if (mBlRig)
    {
        mBlRig->AddRigidityEquation(*mR8_Sys);
    }
    // StdOut() << "SYS=" << mR8_Sys->GetNbObs() << " " <<  mR8_Sys->NbVar() << std::endl;

    const auto & aVectSol = mSys->R_SolveUpdateReset();
    mSetIntervUK.SetVUnKnowns(aVectSol);

    StdOut() << "---------------------------" << std::endl;
}


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

void cMMVII_BundleAdj::SetFrozenCenters(const std::string & aPattern)
{    
    mPatFrozenCenter = aPattern;
}

    /* ---------------------------------------- */
    /*            AddViscosity                  */
    /* ---------------------------------------- */

void cMMVII_BundleAdj::AddPoseViscosity()
{
     //  parse all centra
     for (auto aPcPtr : mVSCPC)
     {
         if (aPcPtr!=nullptr)
         {
            if (mSigmaViscCenter>0)
            {
               mR8_Sys->AddEqFixCurVar(*aPcPtr,aPcPtr->Center(),Square(1/mSigmaViscCenter));
            }
            if (mSigmaViscAngles>0)
            {
               mR8_Sys->AddEqFixCurVar(*aPcPtr,aPcPtr->Omega(),Square(1/mSigmaViscAngles));
            }
         }
     }
}


void cMMVII_BundleAdj::SetViscosity(const tREAL8& aViscTr,const tREAL8& aViscAngle)
{
    mSigmaViscCenter = aViscTr;
    mSigmaViscAngles = aViscAngle;
}


    /* ---------------------------------------- */
    /*            Rigid Bloc                    */
    /* ---------------------------------------- */

void cMMVII_BundleAdj::AddBlocRig(const std::vector<double>& aSigma,const std::vector<double>& aSigmaRat)  // RIGIDBLOC
{
    AssertPhpAndPhaseAdd();
    mBlRig = new cBA_BlocRig(*mPhProj,aSigma,aSigmaRat);

    mBlRig->AddToSys(mSetIntervUK);
}
void cMMVII_BundleAdj::AddCamBlocRig(const std::string & aNameIm) // RIGIDBLOC
{
    cSensorCamPC * aCam = mPhProj->ReadCamPC(aNameIm,/*ToDel*/true,/*SVP*/false);
    if (aCam == nullptr) 
       return;

    mBlRig->AddCam(aCam);
}
void cMMVII_BundleAdj::SaveBlocRigid()
{
    if (mBlRig  && mPhProj->DPRigBloc().DirOutIsInit())  // RIGIDBLOC
    {
       mBlRig->Save();
    }
}



}; // MMVII

