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


/* ************************************************************* */
/*                                                               */
/*                       cCorresp32_BA                           */
/*                                                               */
/* ************************************************************* */


cCorresp32_BA::cCorresp32_BA
(
     cSensorImage       *    aSensor,
     const cSet2D3D &        aSetCorresp
) :
    mSensor        (aSensor),
    mFGC           (true),  // By default we dont optimize position of GCP, they are fix
    mCFix          (false), // By default we use fix point for GCP (contrary only interesting 4 bench)
    mSetCorresp    (aSetCorresp),
    mSzBuf         (100),
    mEqColinearity (mSensor->SetAndGetEqColinearity(true,mSzBuf,false))
{

    for (auto & anObj : mSensor->GetAllUK())
        mSetInterv.AddOneObj(anObj); // #DOC-AddOneObj
    //   mSetInterv.AddOneObj(m CamPC); // #DOC-AddOneObj
    //   mSetInterv.AddOneObj(m Calib);  // #DOC-AddOneObj

    cDenseVect<double> aVUk = mSetInterv.GetVUnKnowns();  // #DOC-GetVUnKnowns
    mSys = new cResolSysNonLinear<double>(eModeSSR::eSSR_LsqDense,aVUk);
}

cResolSysNonLinear<double> & cCorresp32_BA::Sys() {return *mSys;}

void cCorresp32_BA::SetFrozenVarOfPattern(const std::string & aPat)
{
    mSys->UnfrozeAll();
    mSys->SetFrozenFromPat(*mSensor,aPat,true);
}

cCorresp32_BA::~cCorresp32_BA()
{
    delete mEqColinearity;
    delete mSys;
}

     // ==============   Iteration to  ================

void cCorresp32_BA::OneIteration()
{
     //PushErrorEigenErrorLevel(eLevelCheck::Warning);  // still the same problem with eigen excessive error policy ...

     if (mCFix)
     {
        //The fix center will apply only with Perspective central camera
        const cPt3dr * aC = mSensor->CenterOfPC();
	if (aC)
           mSys->SetFrozenVarCurVal(*mSensor,*aC); //  #DOC-FixVar
     }
     //  Three temporary unknowns for x-y-z of the 3d point
     std::vector<int> aVIndGround{-1,-2,-3};

     // Fill indexe Glob in the same order as in cEqColinearityCamPPC::VNamesUnknowns()
     std::vector<int> aVIndGlob = aVIndGround;
     // m CamPC->PushIndexes(aVIndGlob); // #DOC-PushIndex
     // m Calib->PushIndexes(aVIndGlob); // #DOC-PushIndex

     for (auto & anObj : mSensor->GetAllUK())
        anObj->PushIndexes(aVIndGlob); // #DOC-PushIndex

     for (const auto & aCorresp : mSetCorresp.Pairs())
     {
         if (mSensor->PairIsVisible(aCorresp))
	 {
            // structure for points substistion, in mode test they are free
            cSetIORSNL_SameTmp<tREAL8>   aStrSubst
                                         (
                                            aCorresp.mP3.ToStdVector() , // we have 3 temporary unknowns with initial value
					    // #DOC-FrozTmp   If mFGC we indicate that temporary is frozen
                                             (mFGC ? aVIndGround : std::vector<int>())
                                         );

            if (! mFGC)
            {
               for (const auto & anInd : aVIndGround)
                  aStrSubst.AddFixCurVarTmp(anInd,1.0);
            }

            // "observation" of equation  : PTIm (real obs) + Cur-Rotation (Rot = Axiator*CurRot : to avoid guimbal-lock)
            std::vector<double> aVObs = aCorresp.mP2.ToStdVector(); //  Add X-Im, Y-Im in obs

	    mSensor->PushOwnObsColinearity(aVObs); // For PC cam dd all matrix coeff og current rot

            mSys->AddEq2Subst(aStrSubst,mEqColinearity,aVIndGlob,aVObs);
            mSys->AddObsWithTmpUK(aStrSubst);
	 }
     }

     const auto & aVectSol = mSys->SolveUpdateReset();
     mSetInterv.SetVUnKnowns(aVectSol);  // #DOC-SetUnknown

     // PopErrorEigenErrorLevel();
}



}; // MMVII

