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
   private :

       void AddHomBundle_Cam1(cSetIORSNL_SameTmp<tREAL8> &,const cPt3dr & aDirB0,tREAL8  aWeight);
       void AddHomBundle_Cam2(cSetIORSNL_SameTmp<tREAL8> &,const cPt3dr & aDirB1,tREAL8  aWeight);

       tSeg3dr  Bundle(int aKCam,const cPt3dr &) const;


        cElemBA(const cElemBA &) = delete;

        //void Add

       eModResBund                        mMode;
       std::vector<tPoseR>                mCurPose;
       int                                mSzBuf;       ///<  Sz Buf for calculator
       cCalculator<double> *              mEqElemCam1;  ///< Colinearity equation
       cCalculator<double> *              mEqElemCam2;
      // cCalculator<double> *              mEqElemCamN;
       cSetInterUK_MultipeObj<double>     mSetInterv;   ///< coordinator for autom numbering
       cResolSysNonLinear<double> *       mSys;   ///< Solver
       cP3dNormWithUK                     mTr2;   ///< Unknown normaized trace for unit
       cRotWithUK                         mRot2;
       std::vector<cPoseWithUK*>          mPoseN;

       cWeightAv<tREAL8>                  mRes1;
       cWeightAv<tREAL8>                  mRes2;

};

cElemBA::cElemBA(eModResBund aMode,const std::vector<tPoseR>& aVPose) :
    mMode       (aMode),
    mCurPose    (aVPose),
    mSzBuf      (1),
    mEqElemCam1 (EqBundleElem_Cam1(mMode,true,mSzBuf,true)),
    mEqElemCam2 (EqBundleElem_Cam2(mMode,true,mSzBuf,true)),
  //  mEqElemCamN (nullptr),
    mSetInterv  (),
    mSys        (nullptr),
    mTr2        (aVPose.at(1).Tr(),"BAElem","Base1"),
    mRot2       (aVPose.at(1).Rot())

{
    MMVII_INTERNAL_ASSERT_always(mCurPose.at(0).DistPose(tPoseR::Identity(),1.0)==0,"Pose0!=Id in cElemBA");
    MMVII_INTERNAL_ASSERT_always((Norm2(mCurPose.at(1).Tr())-1.0)<1e-8,"Norma base in cElemBA");

    mSetInterv.AddOneObj(&mTr2);
    mSetInterv.AddOneObj(&mRot2);

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

tSeg3dr  cElemBA::Bundle(int aKPose,const cPt3dr & aDirBundle) const
{
    const tPoseR & aPose = mCurPose.at(aKPose);
    return tSeg3dr(aPose.Tr(),aPose.Value(aDirBundle));
}

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

    // StdOut() << " R2 " << mRes2.Average() << "\n";
}


void cElemBA::AddHomBundle_Cam1Cam2(const cPt3dr & aDirB1,const cPt3dr & aDirB2,tREAL8 aW,tREAL8 aEpsilon)
{
    tSeg3dr aSeg1 = Bundle(0,aDirB1);
    tSeg3dr aSeg2 = Bundle(1,aDirB2);

    if (NormInf(aSeg1.V12() ^ aSeg2.V12()) < aEpsilon)
        return;
    cPt3dr aPGround = BundleInters(aSeg1,aSeg2,0.5);

    cSetIORSNL_SameTmp<tREAL8>  aStrSubst(aPGround.ToStdVector(),{});

    AddHomBundle_Cam1(aStrSubst,aDirB1,aW);
    AddHomBundle_Cam2(aStrSubst,aDirB2,aW);

    mSys->R_AddObsWithTmpUK(aStrSubst);
}

void cElemBA::OneIter(tREAL8 aLVM,const std::vector<tPoseR> * aVRef)
{
    StdOut() << "  PIN " << mTr2.RawPNorm() << "\n";
    const auto & aVectSol =  mSys->SolveUpdateReset(aLVM);
    mSetInterv.SetVUnKnowns(aVectSol);

    StdOut() << "  POUT " << mTr2.RawPNorm()
             << " Res=" << mRes1.Average() << " " << mRes2.Average()
             << " DeltaTr=" << 1e4*Norm2( mTr2.GetPNorm()-aVRef->at(1).Tr())
             << "\n";

    mCurPose.at(1) = tPoseR(mTr2.GetPNorm(),mRot2.Rot());
    for (size_t aKP=2 ; aKP<mCurPose.size() ; aKP++)
        mCurPose.at(aKP) = mPoseN.at(aKP-2)->Pose();

   // StdOut()  << "UUUUUUUUUUUUUUppdattte  mCuuuuuurPoosse\n";

    mRes1.Reset();
    mRes2.Reset();
}

/* *************************************************** */
/*                                                     */
/*                 cBenchElemBA                        */
/*                                                     */
/* *************************************************** */


class cBenchElemBA
{
     public :
         cBenchElemBA(size_t aNbPose,cPt2dr aSIgTrRot);

         size_t              mNbPose;
         std::vector<tPoseR> mVPoseGT;    //<  Ground truth poses
         std::vector<tPoseR> mVPosePert;  //< Perturbation

         /// return the minimal distance  of center (GT&Pert) to "Pt"
         tREAL8  DMinCenter(const cPt3dr &,const std::vector<int> &aVIndexe) const;

         /// return a point in a sphere that is enough far of both GT & Pert
         cPt3dr RandomPt(const cPt3dr&,tREAL8,const std::vector<int>&,tREAL8) const;

         ///  Generate bundle perfect in ground truh
         std::vector<cPt3dr> GenBundle(const cPt3dr &,tREAL8,const std::vector<int>&,tREAL8) const;
};

cBenchElemBA::cBenchElemBA(size_t aNbPose,cPt2dr aSIgTrRot) :
    mNbPose (aNbPose)
{
    for (size_t aK=0 ; aK<aNbPose ; aK++)
    {
        tPoseR aPoseGT =  tPoseR::RandomIsom3D(1.0);
        tPoseR aPert = tPoseR(cPt3dr::PRandInSphere()*aSIgTrRot.x(),tRotR::RandomRot(aSIgTrRot.y()));
        tPoseR  aPosePert = aPoseGT * aPert;
        if (aK==0)
        {
            aPoseGT =   tPoseR::Identity();
            aPosePert = tPoseR::Identity();
        }
        if (aK==1)
        {
             aPoseGT.Tr()   = VUnit(aPoseGT.Tr());
             aPosePert.Tr() = VUnit(aPosePert.Tr());
        }

        StdOut()  << "GT " << aPoseGT.Tr() << " PERT " << aPosePert.Tr() << "\n";
        mVPoseGT.push_back(aPoseGT);
        mVPosePert.push_back(aPosePert);
    }
}

tREAL8  cBenchElemBA::DMinCenter(const cPt3dr & aPt,const std::vector<int>& aVIndexe) const
{
    tREAL8 aDMin =1e10;

    for (const auto & anIndexe : aVIndexe)
    {
       UpdateMin(aDMin,Norm2(aPt-mVPoseGT.at(anIndexe).Tr()));
       UpdateMin(aDMin,Norm2(aPt-mVPosePert.at(anIndexe).Tr()));
    }

    return aDMin;
}

cPt3dr cBenchElemBA::RandomPt(const cPt3dr& aCenter,tREAL8 aRay,const std::vector<int>& aVIndexe,tREAL8 aDMin) const
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
    cBenchElemBA::GenBundle(const cPt3dr & aC,tREAL8 aRay,const std::vector<int>& aVInd,tREAL8 aDMin) const
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



class cAppliTestElemBundle : public cMMVII_Appli
{
     public :
        cAppliTestElemBundle(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);
        ~cAppliTestElemBundle();

        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override;
        int Exe() override;

     private :

        eModResBund    mMode;
        int            mNbIter;
        cPt2dr         mSigTrRot;
        tREAL8         mLVM;
        int            mNbSamples;
};




cAppliTestElemBundle::cAppliTestElemBundle(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
    cMMVII_Appli      (aVArgs,aSpec),
    mNbIter           (10),
    mSigTrRot         (0.1,0.1),
    mLVM              (1e-5),
    mNbSamples        (1000)
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



         /*  << AOpt2007(mShowSteps,"ShowSteps","Show detail of computation steps by steps",{eTA2007::HDV})
           << AOpt2007(mZoomImL,"ZoomImL","Zoom for images of line",{eTA2007::HDV})
           << AOpt2007(mRelThrsCumulLow,"ThrCumLow","Low Thresold relative for cumul in histo",{eTA2007::HDV})
           << AOpt2007(mRelThrsCumulHigh,"ThrCumHigh","Low Thresold relative for cumul in histo",{eTA2007::HDV})
               << mPhProj.DPPointsMeasures().ArgDirInOpt("","Folder for ground truth measure")
                       */
            ;
}


int cAppliTestElemBundle::Exe()
{
    cBenchElemBA aBench2(2,mSigTrRot);

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
           // std::vector<cPt3dr> aVBund= aBench2.GenBundle(cPt3dr(0,0,10),5.0,{0,1},0.5);
           aBA.AddHomBundle_Cam1Cam2(aVBund.at(0),aVBund.at(1),1.0);
        }
        aBA.OneIter(mLVM,&aBench2.mVPoseGT);
    }
    StdOut() << "BenchElemBABenchElemBABenchElemBABenchElemBA\n"; getchar();

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

