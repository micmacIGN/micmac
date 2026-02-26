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
       cElemBA(eModResBund,const std::vector<tPoseR>& aVPose);
       ~cElemBA();

       void AddHomBundle_Cam1Cam2(const cPt3dr & aDirB0,const cPt3dr & aDirB1,tREAL8 aW,tREAL8 aEpsilon=1e-6);

       void OneIter(tREAL8 aLVM,const std::vector<tPoseR> * aVRef);
       cResolSysNonLinear<double> *  Sys();   ///< Accesor
   private :

       void AddHomBundle_Cam1(cSetIORSNL_SameTmp<tREAL8> &,const cPt3dr & aDirB0,tREAL8  aWeight);
       void AddHomBundle_Cam2(cSetIORSNL_SameTmp<tREAL8> &,const cPt3dr & aDirB1,tREAL8  aWeight);
       void AddHomBundle_Cam12(const cPt3dr & aDirB1,const cPt3dr & aDirB2,tREAL8  aWeight);

       tSeg3dr  Bundle(int aKCam,const cPt3dr &) const;


        cElemBA(const cElemBA &) = delete;

        //void Add

       eModResBund                        mMode;
       bool                               isMode12;
       std::vector<tPoseR>                mCurPose;
       tPoseR                             mPoseRef;
       tREAL8                             mScaleRef;
       int                                mSzBuf;       ///<  Sz Buf for calculator
       cCalculator<double> *              mEqElemCam1;  ///< Colinearity equation
       cCalculator<double> *              mEqElemCam2;
       cCalculator<double> *              mEqElemCam12;

      // cCalculator<double> *              mEqElemCamN;
       cSetInterUK_MultipeObj<double>     mSetInterv;   ///< coordinator for autom numbering
       cResolSysNonLinear<double> *       mSys;   ///< Solver
       cLeasSqtAA<tREAL8> *               mSystAA;  ///< Pointer to dense solver
      // cLinearOverCstrSys<tREAL8> *       mLinSys;
       cP3dNormWithUK                     mTr2;   ///< Unknown normaized trace for unit
       cRotWithUK                         mRot2;
       std::vector<cPoseWithUK*>          mPoseN;

       cWeightAv<tREAL8>                  mRes1;
       cWeightAv<tREAL8>                  mRes2;

};

cElemBA::cElemBA(eModResBund aMode,const std::vector<tPoseR>& aVPose) :
    mMode       (aMode),
    isMode12    (! ModResBund_IsModeGen(mMode)),
    mCurPose    (aVPose),
    mSzBuf      (1),
    mEqElemCam1 (EqBundleElem_Cam1(mMode,true,mSzBuf,true)),
    mEqElemCam2 (EqBundleElem_Cam2(mMode,true,mSzBuf,true)),
    mEqElemCam12 (EqBundleElem_Cam12(mMode,true,mSzBuf,true)),
  //  mEqElemCamN (nullptr),
    mSetInterv  (),
    mSys        (nullptr),
    mTr2        (cPt3dr(0,0,1),"BAElem","Base1"),
    mRot2       (tRotR::Identity())

{
    mPoseRef = mCurPose.at(0);
    mScaleRef = Norm2(mCurPose.at(0).Tr() -mCurPose.at(1).Tr());

    for (auto & aPose : mCurPose)
    {
        // PoseRef  C1 ->W
       aPose = (mPoseRef.MapInverse()*aPose).ScaleTr(1.0/mScaleRef);
       // aPose = (aPose*mPoseRef.MapInverse()).ScaleTr(1.0/mScaleRef);
    }
    mTr2.SetPNorm(mCurPose.at(1).Tr());
    mRot2.SetRot(mCurPose.at(1).Rot());


    MMVII_INTERNAL_ASSERT_always(mCurPose.at(0).DistPose(tPoseR::Identity(),1.0)<1e-7,"Pose0!=Id in cElemBA");
    MMVII_INTERNAL_ASSERT_always((Norm2(mCurPose.at(1).Tr())-1.0)<1e-8,"Norma base in cElemBA");

    mSetInterv.AddOneObj(&mTr2);
    mSetInterv.AddOneObj(&mRot2);

    for (size_t aKP=2 ; aKP<mCurPose.size() ; aKP++)
    {
        mPoseN.push_back(new cPoseWithUK(mCurPose.at(aKP)));
        mSetInterv.AddOneObj(mPoseN.back());
    }

    mSys = new cResolSysNonLinear<double>(eModeSSR::eSSR_LsqDense,mSetInterv.GetVUnKnowns());
    mSystAA = mSys->SysLinear()->Get_tAA();
}

cElemBA::~cElemBA()
{
    DeleteAllAndClear(mPoseN);
    delete mSys;
}

cResolSysNonLinear<double> *  cElemBA::Sys() {return mSys;}



void cElemBA::AddHomBundle_Cam1
     (
           cSetIORSNL_SameTmp<tREAL8> & aStrSubst,
           const cPt3dr & aDirB1,
           tREAL8         aWeight
      )
{
    std::vector<int> aVIndGlob = {-1,-2,-3};
    std::vector<double> aVObs = aDirB1.ToStdVector();

    mSys->R_AddEq2Subst(aStrSubst,mEqElemCam1,aVIndGlob,aVObs,aWeight);

    for (int aK=0 ; aK<3 ; aK++)
        mRes1.Add(1.0,std::abs(mEqElemCam1->ValComp(0,aK)));

  //  StdOut() << " R1 " << mRes1.Average() << "\n";
}

void cElemBA::AddHomBundle_Cam2
     (
           cSetIORSNL_SameTmp<tREAL8> & aStrSubst,
           const cPt3dr & aDirB2,
           tREAL8         aWeight
      )
{
    std::vector<int> aVIndGlob = {-1,-2,-3};
    std::vector<double> aVObs = aDirB2.ToStdVector();
    mTr2.AddIdexesAndObs(aVIndGlob,aVObs);
    // false dont transpose, we use Cam->Word
    mRot2.AddIdexesAndObs(aVIndGlob,aVObs,  false);
    mSys->R_AddEq2Subst(aStrSubst,mEqElemCam2,aVIndGlob,aVObs,aWeight);

    for (int aK=0 ; aK<3 ; aK++)
        mRes2.Add(1.0,std::abs(mEqElemCam2->ValComp(0,aK)));
}

void cElemBA::AddHomBundle_Cam12(const cPt3dr & aDirB1,const cPt3dr & aDirB2,tREAL8  aWeight)
{
      if (mMode==eModResBund::eLinDet12)
      {
       // As the
       //  [B , Uu , R(Id+W) u2] = [R'B , R'u1 , u2 + W^u2] = [R'(B+A da + B db), R'u1 ,u2 + W ^u2]
       //   [R'B,R'u1,::u2] + [R'(A da + B db),R'u1,u2] +  [R'B,R'u1,  W^u2]
       //   [B,u1,Ru2]   + [A da+ B db,u1,R u2] 

          cPt3dr aU02 = mRot2.Rot().Value(aDirB2);
          cPt3dr aN0 = aDirB1 ^ aU02;
          cPt3dr aB0 =  mTr2.RawPNorm();

          cDenseVect<tREAL8> aVect(5);

          aVect(0) = Scal(aN0, mTr2.U());
          aVect(1) = Scal(aN0, mTr2.V());
          tREAL8 aRes = Scal(aB0,aN0);

          cPt3dr aUP1 = mRot2.Rot().Inverse(aDirB1);
          cPt3dr aBP  = mRot2.Rot().Inverse(aB0);

          cPt3dr aScalW =   (aBP*Scal(aUP1,aDirB2) - aUP1 *Scal(aBP,aDirB2) ) * -1.0;

          aVect(2) = aScalW.x();
          aVect(3) = aScalW.y();
          aVect(4) = aScalW.z();
          if (0)
          {
             mSys->AddObservationLinear(aWeight,aVect,-aRes);
          }
          else
          {
            // Work now with correction on "IO_UnKnowns"
             mSystAA->PublicAddObservation(aWeight,aVect,-aRes);
          }
          mRes1.Add(1.0,std::abs(aRes));
          mRes2.Add(1.0,std::abs(aRes));
         return;
      }
      std::vector<int> aVIndGlob ;
      std::vector<double> aVObs = Append(aDirB1.ToStdVector(),aDirB2.ToStdVector());
      mTr2.AddIdexesAndObs(aVIndGlob,aVObs);
      mRot2.AddIdexesAndObs(aVIndGlob,aVObs,  false);

      mSys->R_CalcAndAddObs(mEqElemCam12,aVIndGlob,aVObs,aWeight);

      for (size_t aK=0 ; aK<mEqElemCam12->NbElem() ; aK++)
      {
          tREAL8 aRes = std::abs(mEqElemCam12->ValComp(0,aK));

         /* StdOut()  << "CCC " << mEqElemCam12->ValComp(0,aK) << " DDD " ;
          for (int aD=0 ; aD<5 ; aD++)
              StdOut()  << " " <<  mEqElemCam12->DerComp(0,aK,aD) ;
          StdOut()<< "\n";*/

          mRes1.Add(1.0,aRes);
          mRes2.Add(1.0,aRes);
      }

}

tSeg3dr  cElemBA::Bundle(int aKPose,const cPt3dr & aDirBundle) const
{
    const tPoseR & aPose = mCurPose.at(aKPose);
    return tSeg3dr(mPoseRef.Value(aPose.Tr())/mScaleRef,mPoseRef.Value(aPose.Value(aDirBundle))/mScaleRef);
}


void cElemBA::AddHomBundle_Cam1Cam2(const cPt3dr & aDirB1,const cPt3dr & aDirB2,tREAL8 aW,tREAL8 aEpsilon)
{   
    cPt3dr aDirLoc1 = mPoseRef.Rot().Inverse(aDirB1);
    cPt3dr aDirLoc2 = mPoseRef.Rot().Inverse(aDirB2);

   // StdOut()  << "HHHH " << Norm2(aSeg1.V12()) << " " << Norm2(aSeg2.V12()) << "\n";

    if (NormInf(aDirLoc1 ^ aDirLoc2) < aEpsilon)
        return;


    if (isMode12)
    {
       AddHomBundle_Cam12(aDirB1,aDirB2,aW);
    }
    else
    {
        tSeg3dr aSeg1 = Bundle(0,aDirB1);
        tSeg3dr aSeg2 = Bundle(1,aDirB2);

       cPt3dr aPGround = BundleInters(aSeg1,aSeg2,0.5);

       tSegComp3dr aSC1(aSeg1);
       tSegComp3dr aSC2(aSeg2);

      // StdOut() << " DDD " << aSC1.Dist(aPGround) << " " << aSC2.Dist(aPGround) << "\n"; // getchar();
       cSetIORSNL_SameTmp<tREAL8>  aStrSubst(aPGround.ToStdVector(),{});

       AddHomBundle_Cam1(aStrSubst, aDirB1,aW);
       AddHomBundle_Cam2(aStrSubst, aDirB2,aW);
       mSys->R_AddObsWithTmpUK(aStrSubst);
    }
}

void cElemBA::OneIter(tREAL8 aLVM,const std::vector<tPoseR> * aVRef)
{

    // Do the computation
    const auto & aVectSol =  mSys->SolveUpdateReset(aLVM);
    mSetInterv.SetVUnKnowns(aVectSol);

    // Transferate the result to poses (as they

    mCurPose.at(1) = tPoseR(mTr2.GetPNorm(),mRot2.Rot());
    for (size_t aKP=2 ; aKP<mCurPose.size() ; aKP++)
        mCurPose.at(aKP) = mPoseN.at(aKP-2)->Pose();

    StdOut() // << "  POUT " << mTr2.RawPNorm()
             << " Res=" << mRes1.Average() << " " << mRes2.Average() ;

    {
       tPoseR aPoseRef = (mPoseRef.MapInverse()*aVRef->at(1)).ScaleTr(1.0/mScaleRef);

       StdOut()   << " DeltaTr=" << 1e4*Norm2( mCurPose.at(1).Tr()-aPoseRef.Tr())
                << " DeltaRot=" << 1e4*mCurPose.at(1).Rot().Dist(aPoseRef.Rot()) ;
        StdOut()        << "\n";
    }


   // StdOut()  << "UUUUUUUUUUUUUUppdattte  mCuuuuuurPoosse\n";

    mRes1.Reset();
    mRes2.Reset();
}

/* *************************************************** */
/*                                                     */
/*                 cParamBenchElemBA                   */
/*                                                     */
/* *************************************************** */

class cParamBenchElemBA
{
    public :
      cParamBenchElemBA() ;

       eModResBund    mMode;
       int            mNbIter;
       cPt2dr         mSigTrRot;
       tREAL8         mLVM;
       int            mNbSamples;

       //  ---- Parameters for generating bundles
       cPt3dr         mCenterGP;  // centre of ground points
       tREAL8         mRayGP;     // Ray of Groun Point Sphere
       tREAL8         mDistAvoid; // Distance min to center of poses in
};

cParamBenchElemBA::cParamBenchElemBA() :
   mMode             (eModResBund::eNbVals),
   mNbIter           (10),
   mSigTrRot         (0.1,0.1),
   mLVM              (1e-5),
   mNbSamples        (100),
   mCenterGP         (0,0,10.0),
   mRayGP            (5.0),
   mDistAvoid        (0.5)
{
}

// cPt3dr(0,0,10),5.0,{0,1},0.5
/* *************************************************** */
/*                                                     */
/*                 cBenchElemBA                        */
/*                                                     */
/* *************************************************** */


class cGenPoseBenchElemBA
{
     public :
         cGenPoseBenchElemBA(size_t aNbPose,const cParamBenchElemBA&);

         ///  Generate bundle perfect in ground truh
         std::vector<cPt3dr> GenBundle(const cPt3dr &,tREAL8,const std::vector<int>&,tREAL8) const;

         std::vector<cPt3dr> GenBundle(const std::vector<int>&) const;

         size_t              mNbPose;
         std::vector<tPoseR> mVPoseGT;    //<  Ground truth poses
         std::vector<tPoseR> mVPosePert;  //< Perturbated

         cParamBenchElemBA  mParam;

     private :
         /// return the minimal distance  of center (GT&Pert) to "Pt"
         tREAL8  DMinCenter(const cPt3dr &,const std::vector<int> &aVIndexe) const;

         /// return a point in a sphere that is enough far of both GT & Pert
         cPt3dr RandomPt(const cPt3dr&,tREAL8,const std::vector<int>&,tREAL8) const;

};

cGenPoseBenchElemBA::cGenPoseBenchElemBA(size_t aNbPose,const cParamBenchElemBA& aParamBA) :
    mNbPose (aNbPose),
    mParam  (aParamBA)
{
    // used to be a parameter, but could not solve the internal normalistion in
    bool NormInit = true;
    std::vector<int> aVIndexes;
    for (size_t aK=0 ; aK<aNbPose ; aK++)
    {
        //  -- generate random rot assuring far enough of previous -----------
        tRotR  aRotGT = tRotR::RandomRot();
        cPt3dr aCenterGT = RandomPt(cPt3dr(0,0,0),1.0,aVIndexes,0.5);
        tPoseR aPoseGT(aCenterGT,aRotGT);

        // --- generate smal perturbation then perturbated pose
        tPoseR aPert = tPoseR(cPt3dr::PRandInSphere()*aParamBA.mSigTrRot.x(),tRotR::RandomRot(aParamBA.mSigTrRot.y()));
        tPoseR  aPosePert = aPoseGT * aPert;

        // --  assure that pose are normalized
        if (NormInit)
        {
           if (aK==0)  // First pose is identity
           {
               aPoseGT =   tPoseR::Identity();
               aPosePert = tPoseR::Identity();
           }
           if (aK==1)  // distance C0-C1 is 1.0
           {
               aPoseGT.Tr()   = VUnit(aPoseGT.Tr());
               aPosePert.Tr() = VUnit(aPosePert.Tr());
           }
        }

        mVPoseGT.push_back(aPoseGT);
        mVPosePert.push_back(aPosePert);
        aVIndexes.push_back(aK);
    }
}

tREAL8  cGenPoseBenchElemBA::DMinCenter(const cPt3dr & aPt,const std::vector<int>& aVIndexe) const
{
    tREAL8 aDMin =1e10;

    for (const auto & anIndexe : aVIndexe)
    {
       UpdateMin(aDMin,Norm2(aPt-mVPoseGT.at(anIndexe).Tr()));
       UpdateMin(aDMin,Norm2(aPt-mVPosePert.at(anIndexe).Tr()));
    }

    return aDMin;
}

cPt3dr cGenPoseBenchElemBA::RandomPt(const cPt3dr& aCenter,tREAL8 aRay,const std::vector<int>& aVIndexe,tREAL8 aDMin) const
{
    for (int aK=0 ;aK<1e5 ; aK++)
    {
        cPt3dr aPt = aCenter + cPt3dr::PRandInSphere()*aRay;
        if (DMinCenter(aPt,aVIndexe)>aDMin)
            return aPt;
    }

    MMVII_INTERNAL_ERROR("cBenchElemBA::RandomPt");
    return cPt3dr(0,0,0);
}


std::vector<cPt3dr>
    cGenPoseBenchElemBA::GenBundle(const cPt3dr & aC,tREAL8 aRay,const std::vector<int>& aVInd,tREAL8 aDMin) const
{
    std::vector<cPt3dr>  aRes;

   cPt3dr  aPGround = RandomPt(aC,aRay,aVInd,aDMin);

   for (const auto anInd : aVInd )
   {
       const tPoseR & aPose = mVPoseGT.at(anInd);
       cPt3dr aPLoc = aPose.Inverse(aPGround);
       aRes.push_back(VUnit(aPLoc));
   }

    return aRes;
}



void BenchElemBA()
{

}

/* *************************************************** */
/*                                                     */
/*                 cAppliTestElemBundle                */
/*                                                     */
/* *************************************************** */




class cAppliTestElemBundle : public cMMVII_Appli,
                             public cParamBenchElemBA
{
     public :
        cAppliTestElemBundle(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);
        ~cAppliTestElemBundle();

        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override;
        int Exe() override;

     private :
};




cAppliTestElemBundle::cAppliTestElemBundle(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
    cMMVII_Appli      (aVArgs,aSpec)
{

}

cAppliTestElemBundle::~cAppliTestElemBundle()
{

}

cCollecSpecArg2007 & cAppliTestElemBundle::ArgObl(cCollecSpecArg2007 & anArgObl)
{
      return    anArgObl
            <<  Arg2007(mMode,"Mode of bundle compens", {AC_ListVal<eModResBund>()})
      ;
}

cCollecSpecArg2007 & cAppliTestElemBundle::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{

    return anArgOpt
         << AOpt2007(mNbIter,"NbIter","Number of iteration",{eTA2007::HDV})
         << AOpt2007(mSigTrRot,"SigTR","Sigma Noise Tr/Rot",{eTA2007::HDV})
         << AOpt2007(mLVM,"LVM","Levenberg/Markard parameter",{eTA2007::HDV})
         << AOpt2007(mNbSamples,"NbS","Number of samples",{eTA2007::HDV})
        ;
}


int cAppliTestElemBundle::Exe()
{
    cGenPoseBenchElemBA aBench2(2,*this);

    cElemBA aBA(mMode,aBench2.mVPosePert);

    std::vector<std::vector<cPt3dr>> aVVBund;
    for (int aK=0 ; aK<mNbSamples ; aK++)
    {
        std::vector<cPt3dr> aVBund= aBench2.GenBundle(cPt3dr(0,0,10),5.0,{0,1},0.5);
        aVVBund.push_back(aVBund);
    }

    for (int aKIter= 0 ; aKIter<mNbIter ; aKIter++)
    {
        for (const auto & aVBund : aVVBund)
        {
           aBA.AddHomBundle_Cam1Cam2(aVBund.at(0),aVBund.at(1),1.0);
        }
        aBA.OneIter(mLVM,&aBench2.mVPoseGT);

    }

    return EXIT_SUCCESS;
}


tMMVII_UnikPApli Alloc_TestElemBundle(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
      return tMMVII_UnikPApli(new cAppliTestElemBundle(aVArgs,aSpec));
}


cSpecMMVII_Appli  TheSpecAppliTestElemBundle
(
     "TestElemBundle",
      Alloc_TestElemBundle,
      "Internal, and possibly temporary, test application for elementary bundles",
      {},
      {},
      {},
      __FILE__
);

}; // MMVII

