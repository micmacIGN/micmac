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
    mFolderRefCam     (""),
    mSigmaTrRefCam    (-1.0),
    mSigmaRotRefCam   (-1.0),
    mPatternRef       (".*"),
    mDirRefCam        (nullptr),
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
    CompileSharedIntrinsicParams(true);
    mPhaseAdd = false;

    InitItereGCP();
    mR8_Sys = new cResolSysNonLinear<tREAL8>(eModeSSR::eSSR_LsqNormSparse,mSetIntervUK.GetVUnKnowns());

    mSys =  mR8_Sys;
    CompileSharedIntrinsicParams(false);
}


void cMMVII_BundleAdj::OneIteration(tREAL8 aLVM)
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

    // Add constriant betweenn reference and pose
    AddConstrainteRefPose();


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

    const auto & aVectSol = mSys->R_SolveUpdateReset(aLVM);
    mSetIntervUK.SetVUnKnowns(aVectSol);

    StdOut() << "---------------------------" << std::endl;
}


void cMMVII_BundleAdj::AddCalib(cPerspCamIntrCalib * aCalib)  
{
    AssertPhaseAdd();
    if (aCalib==nullptr)  return;
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

    aSI->SetAndGetEqColinearity(true,10,true);  // WithDer, SzBuf, ReUse
}


void cMMVII_BundleAdj::AddCamPC(cSensorCamPC * aCamPC)
{
    AddCalib(aCamPC->InternalCalib());
}

void cMMVII_BundleAdj::AddReferencePoses(const std::vector<std::string> & aVec)
{
     MMVII_INTERNAL_ASSERT_tiny(mVSCPC.empty(),"Must Add Ref Pose before any cam");
     AssertPhpAndPhaseAdd();

     mFolderRefCam = aVec.at(0);
     mDirRefCam  = mPhProj->NewDPIn(eTA2007::Orient,mFolderRefCam);

     mSigmaTrRefCam = cStrIO<tREAL8>::FromStr(aVec.at(1));
     if (aVec.size() > 2)
        mSigmaRotRefCam = cStrIO<tREAL8>::FromStr(aVec.at(2));

     if (aVec.size() > 3)
        mPatternRef =  aVec.at(3);
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

    //  process the reference cameras
    if (mDirRefCam)
    {
        // if PC Cam dont exist push 0 to be coherent between "mVCamRefPose" and   "mVSCPC"
        if (aSPC==nullptr)
	{
            mVCamRefPoses.push_back(aSPC);
	}
	else
	{
            const std::string & aNameImage = aSPC->NameImage();
            cSensorCamPC * aCamRef  = mPhProj->ReadCamPC(*mDirRefCam,aNameImage,true);
            mVCamRefPoses.push_back(aCamRef);
	}
    }
}
const std::vector<cSensorImage *> &  cMMVII_BundleAdj::VSIm() const  {return mVSIm;}
const std::vector<cSensorCamPC *> &  cMMVII_BundleAdj::VSCPC() const {return mVSCPC;}

    /* ---------------------------------------- */
    /*            Frozen/Shared                 */
    /* ---------------------------------------- */

void cMMVII_BundleAdj::SetParamFrozenCalib(const std::string & aPattern)
{    
    mPatParamFrozenCalib = aPattern;
}

void cMMVII_BundleAdj::SetFrozenCenters(const std::string & aPattern)
{    
    mPatFrozenCenter = aPattern;
}

void cMMVII_BundleAdj::SetSharedIntrinsicParams(const std::vector<std::string> & aVParams)
{
    mVPatShared = aVParams;
}

typedef std::tuple<int,std::string,tREAL8 *> tISRP;

void cMMVII_BundleAdj::CompileSharedIntrinsicParams(bool ForAvg)
{
    MMVII_INTERNAL_ASSERT_tiny((mVPatShared.size()%2)==0,"Expected even size for shared intrinsic params");
    bool  Show = ForAvg;

    // Parse the pair Pattern Name Cam / Pattern Name Params
    for (size_t aKPat=0 ; aKPat<mVPatShared.size() ; aKPat+=2)
    {
        std::map<std::string,std::vector<int>> aMapSharedIndexes; // store the shared index of a given param name
        std::map<std::string,std::vector<std::string>> aMapNames; // store the sharing as name, for show
        std::map<std::string,std::vector<tISRP>> aMapValues; // store the sharing of adress for averaging
        // Parse the calib and select those which name match the pattern name cam
        for (auto  aPtrCal : mVPCIC)
        {
            if (MatchRegex(aPtrCal->Name(),mVPatShared[aKPat]))
            {
                // Extract information on parameter macthing the pattern of params
                cGetAdrInfoParam<tREAL8>  aGIP(mVPatShared[aKPat+1],*aPtrCal);  
                for (size_t aKParam=0 ; aKParam<aGIP.VAdrs().size() ; aKParam++)
                {
                    tREAL8 * aAdr                 = aGIP.VAdrs().at(aKParam);
                    const std::string & aNameP    = aGIP.VNames().at(aKParam);
                    cObjWithUnkowns<tREAL8>* aObj = aGIP.VObjs().at(aKParam);

                    size_t  aNum = aObj->IndOfVal(aAdr);
                    aMapSharedIndexes[aNameP].push_back(aNum);
                    aMapNames[aNameP].push_back(aPtrCal->Name());

                    aMapValues[aNameP].push_back(tISRP(aNum,aPtrCal->Name(),aAdr));
                }
            }
        }
        if (Show)
        {
                StdOut()  << "=========== Shared params for" 
                          << " PatName ={" << mVPatShared[aKPat] << "}" 
                          << " PatCal={" << mVPatShared[aKPat+1] << "}"
                          << " ============ " << std::endl;
        }
        for (const auto & [aNamePar,aVTuple] : aMapValues)
        {
            std::vector<int>  aVIndEqui;
            tREAL8  aSum = 0.0;
            for (const auto & [aNum,aNameCam,anAdr] : aVTuple)
            {
                aSum += *anAdr;
                aVIndEqui.push_back(aNum);
            }
            aSum /= aVTuple.size();
            if (Show)
               StdOut()  <<  "   * "  << aNamePar  << " : " << aSum << std::endl;
            if (ForAvg)
            {
                for (const auto & [aNum,aNameCam,anAdr] : aVTuple)
                {
                    //*anAdr = aSum;  => dont undertand why it apparently slow down the convergence ??
                    StdOut()  <<  "      - "  << aNameCam <<  " : " << *anAdr << std::endl;
                }
            }
            else
            {
               mSys->SetShared(aVIndEqui);
            }
        }
/*
        if (ForAvg)
        {
           if (Show)
           {
                StdOut()  << "=========== Shared params for" 
                          << " PatName ={" << mVPatShared[aKPat] << "}" 
                          << " PatCal={" << mVPatShared[aKPat+1] << "}"
                          << " ============ " << std::endl;
                for (const auto & [aNamePar,aVNameCam] : aMapNames)
                {
                      StdOut()  <<  "   * "  << aNamePar << std::endl;
                      for (const auto &  aNameCam : aVNameCam )
                          StdOut()  <<  "      - "  << aNameCam << std::endl;
                }
                StdOut()  << "==========================================================" << std::endl;
           }
        }
        else
        {
             for (const auto & [aNamePar,aVIndexes] : aMapSharedIndexes)
             {
                 mSys->SetShared(aVIndexes);
             }
        }
*/
    }
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
    /*             Reference Pose               */
    /* ---------------------------------------- */

void cMMVII_BundleAdj::AddConstrainteRefPose()
{
   if (!mDirRefCam)
      return;

   for (size_t aKC=0 ; aKC<mVSCPC.size() ; aKC++)
   {
        cSensorCamPC * aCam = mVSCPC[aKC];
        cSensorCamPC * aCamRef =  mVCamRefPoses[aKC];
        if ((aCam!=nullptr) && (aCamRef!=nullptr))
           AddConstrainteRefPose(*aCam,*aCamRef);
   }
}

void cMMVII_BundleAdj::AddConstrainteRefPose(cSensorCamPC & aCam,cSensorCamPC & aCamRef)
{
     if (! MatchRegex(aCam.NameImage(),mPatternRef))
        return;
     // mR8_Sys
     if (mSigmaTrRefCam > 0)
     {
        mR8_Sys->AddEqFixNewVal(aCam,aCam.Center(),aCamRef.Center(),Square(1/mSigmaTrRefCam));
     }
     
     if (mSigmaRotRefCam>0)
     {
         cPt3dr aWTarget = aCam.Pose_WU().ValAxiatorFixRot(aCamRef.Pose().Rot());
         mR8_Sys->AddEqFixNewVal(aCam,aCam.Omega(),aWTarget,Square(1/mSigmaRotRefCam));
     }

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

