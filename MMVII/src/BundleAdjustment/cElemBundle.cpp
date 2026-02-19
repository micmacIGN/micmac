#include "MMVII_PCSens.h"
#include "MMVII_DeclareCste.h"
#include "MMVII_BundleAdj.h"

/**
   \file cConvCalib.cpp  testgit

   \brief file for conversion between calibration (change format, change model) and tests
*/


namespace MMVII
{

class cElemBA
{
   public :
       cElemBA(const std::vector<tPoseR>& aVPose);
       ~cElemBA();
   private :
        cElemBA(const cElemBA &) = delete;

        //void Add

        std::vector<tPoseR>                mCurPose;

       int                                mSzBuf;       ///<  Sz Buf for calculator
       cCalculator<double> *              mEqElemCam1;  ///< Colinearity equation
       cCalculator<double> *              mEqElemCam2;
      // cCalculator<double> *              mEqElemCamN;
       cSetInterUK_MultipeObj<double>     mSetInterv;   ///< coordinator for autom numbering
       cResolSysNonLinear<double> *       mSys;   ///< Solver
       cP3dNormWithUK                     mTr1;
       cRotWithUK                         mRot1;
       std::vector<cPoseWithUK*>          mPoseN;
};

cElemBA::cElemBA(const std::vector<tPoseR>& aVPose) :
    mCurPose    (aVPose),
    mSzBuf      (1),
    mEqElemCam1 (EqBundleElem_Cam1(true,mSzBuf,true)),
    mEqElemCam2 (EqBundleElem_Cam2(true,mSzBuf,true)),
  //  mEqElemCamN (nullptr),
    mSetInterv  (),
    mSys        (nullptr),
    mTr1        (aVPose.at(1).Tr(),"BAElem","Base1"),
    mRot1         (aVPose.at(1).Rot())

{
    MMVII_INTERNAL_ASSERT_always(mCurPose.at(0).DistPose(tPoseR::Identity(),1.0)==0,"Pose0!=Id in cElemBA");
    mSetInterv.AddOneObj(&mTr1);
    mSetInterv.AddOneObj(&mRot1);

    for (size_t aKP=2 ; aKP<aVPose.size() ; aKP++)
    {
        mPoseN.push_back(new cPoseWithUK(aVPose.at(aKP)));
        mSetInterv.AddOneObj(mPoseN.back());
    }

    mSys = new cResolSysNonLinear<double>(eModeSSR::eSSR_LsqDense,mSetInterv.GetVUnKnowns());
}

cElemBA::~cElemBA()
{
    DeleteAllAndClear(mPoseN);
    delete mSys;
}

void BenchElemBA()
{
    tPoseR aP0 = tPoseR::Identity();
    tPoseR aP1 = tPoseR::RandomIsom3D(1.0);

    std::vector<tPoseR> aVPose{aP0,aP1};
    cElemBA aBA(aVPose);

    StdOut() << "BenchElemBABenchElemBABenchElemBABenchElemBA\n";
}

#if (0)


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
    {
        mSetInterv.AddOneObj(anObj); // #DOC-AddOneObj
    }
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
        //  StdOut() << "WWWWW " << aCorresp.mWeight << "\n";
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

            mSensor->PushOwnObsColinearity(aVObs,aCorresp.mP3); // For PC cam dd all matrix coeff og current rot

            // StdOut() << "WWWWWWWWWWWWW=" << aCorresp.mWeight << "\n";
            cResidualWeighter<tREAL8> aWeighter(aCorresp.mWeight);
            mSys->AddEq2Subst(aStrSubst,mEqColinearity,aVIndGlob,aVObs,aWeighter);
            mSys->AddObsWithTmpUK(aStrSubst);
         }
         else
         { 
               // StdOut() << "Hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh " <<  aCorresp.mP2 << "\n";
         }
     }

     const auto & aVectSol = mSys->SolveUpdateReset();
     mSetInterv.SetVUnKnowns(aVectSol);  // #DOC-SetUnknown

     // PopErrorEigenErrorLevel();
}
#endif



}; // MMVII

