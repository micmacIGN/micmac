#include "MMVII_PCSens.h"
#include "MMVII_DeclareCste.h"
#include "MMVII_BundleAdj.h"
#include "MMVII_PoseRel.h"

/**
   \file cConvCalib.cpp  testgit

   \brief file for conversion between calibration (change format, change model) and tests
*/


namespace MMVII
{


cElemBA::cElemBA(eModResBund aMode,const std::vector<tPoseR>& aVPose) :
    mMode        (aMode),
    isModeCoplan (! ModResBund_IsModeGen(mMode)),
    mCurPose     (aVPose),
    mSzBuf       (1),
    mEqElemCam1  (EqBundleElem_Cam1(mMode,true,mSzBuf,true)),
    mEqElemCam2  (EqBundleElem_Cam2(mMode,true,mSzBuf,true)),
    mEqElemCamN  (EqBundleElem_CamN(mMode,true,mSzBuf,true)),
    mEqElemCam12 (EqBundleElem_Cam12(mMode,true,mSzBuf,true)),
  //  mEqElemCamN (nullptr),
    mSetInterv   (new   cSetInterUK_MultipeObj<double> ),
    mSys         (nullptr),
    mTr2         (cPt3dr(0,0,1),"BAElem","Base1"),
    mRot2        (tRotR::Identity()),
    mSCCam12     (true),
    mIndCamGen   (mSCCam12 ? 2 : 0)

{

    //--- Check that pose complies with normalization pre-requirement
    MMVII_INTERNAL_ASSERT_always(mCurPose.at(0).DistPose(tPoseR::Identity(),1.0)<1e-7,"Pose0!=Id in cElemBA");
    MMVII_INTERNAL_ASSERT_always((Norm2(mCurPose.at(1).Tr())-1.0)<1e-8,"Norma base in cElemBA");

    //  --------  Initialize the unknowns ----------------------
    if (mSCCam12)
    {
       mTr2.SetPNorm(mCurPose.at(1).Tr());
       mRot2.SetRot(mCurPose.at(1).Rot());

       mSetInterv->AddOneObj(&mTr2);
       mSetInterv->AddOneObj(&mRot2);
    }
    for (size_t aKP=mIndCamGen ; aKP<mCurPose.size() ; aKP++)
    {
        mPoseN.push_back(new cPoseWithUK(mCurPose.at(aKP)));
        mSetInterv->AddOneObj(mPoseN.back());
    }

    //  ------------------ Create the non linear system -------------------------
    mSys = new cResolSysNonLinear<double>(eModeSSR::eSSR_LsqDense,mSetInterv->GetVUnKnowns());
    mSystAA = mSys->SysLinear()->Get_tAA();
    // in this case, as we short-cut the standard system, we supress this warning
    if (mMode==eModResBund::eLinDet12)
        mSys->SetUseWarningNotEnoughObs(false);
}

cElemBA::~cElemBA()
{
    //  WARN  & TRICKY : I have made pointer of mSetInterv, because it must be destroyed before mPoseN
    // in destuctor of mSetInterv, some clearing method on the object are applied ....
    delete mSetInterv;

    DeleteAllAndClear(mPoseN);
    delete mSys;
}

cResolSysNonLinear<double> *  cElemBA::Sys() {return mSys;}
const std::vector<tPoseR>  &  cElemBA::CurPose() const {return mCurPose;}
tREAL8 cElemBA::AvgRes1() const {return mRes1.Average();}
tREAL8 cElemBA::AvgRes2() const {return mRes2.Average();}
tREAL8 cElemBA::AvgResN() const {return mResN.Average();}


tREAL8 cElemBA::AddEquationColinearity_Cam1
     (
           cSetIORSNL_SameTmp<tREAL8> & aStrSubst,
           const cPt3dr & aDirB1,
           tREAL8         aWeight
      )
{
    std::vector<int> aVIndGlob = {-1,-2,-3};  // index of unknwon, only temporary Ground point
    std::vector<double> aVObs = aDirB1.ToStdVector();  // vector of observation the bundle

    // add the in the subs-struct the equation forcing point to belong to bundle
    mSys->R_AddEq2Subst(aStrSubst,mEqElemCam1,aVIndGlob,aVObs,aWeight);

    // accumulate residual for cam1
    tREAL8 aSumR=0.0;
    for (size_t aK=0 ; aK<mEqElemCam1->NbElem() ; aK++)
    {
        tREAL8 aRes = std::abs(mEqElemCam1->ValComp(0,aK));
        mRes1.Add(1.0,aRes);
        aSumR += Square(aRes);
    }
    return std::sqrt(aSumR) ;
}

// mEqElemCam12->NbElem()

tREAL8  cElemBA::AddEquationColinearity_Cam2
     (
           cSetIORSNL_SameTmp<tREAL8> & aStrSubst,
           const cPt3dr & aDirB2,
           tREAL8         aWeight
      )
{
    std::vector<int> aVIndGlob = {-1,-2,-3}; //< Index UK, begin with 3D point
    std::vector<double> aVObs = aDirB2.ToStdVector(); // Obs : begin bundle
    mTr2.AddIdexesAndObs(aVIndGlob,aVObs);  // Add Obs & UK for unitary translation
    // false dont transpose, we use Cam->Word
    mRot2.AddIdexesAndObs(aVIndGlob,aVObs,  false); // Add Ob & Uk for rotation

    // Add in sub-stru equation forcing point to belong to bundle, taking into account the unknown pose
    mSys->R_AddEq2Subst(aStrSubst,mEqElemCam2,aVIndGlob,aVObs,aWeight);

    // accumulate residual for cam2
    tREAL8 aSumR = 0;
    for (size_t aK=0 ; aK<mEqElemCam2->NbElem() ; aK++)
    {
        tREAL8 aRes = std::abs(mEqElemCam2->ValComp(0,aK));
        mRes2.Add(1.0,aRes);
        aSumR += Square(aRes);
    }
    return std::sqrt(aSumR);
}

tREAL8 cElemBA::AddEquationColinearity_CamN
     (
        size_t aIndC,
        cSetIORSNL_SameTmp<tREAL8> & aStrSubst,
        const cPt3dr & aDirB,
        tREAL8  aWeight
     )
{
    std::vector<int> aVIndGlob = {-1,-2,-3}; //< Index UK, begin with 3D point
    std::vector<double> aVObs = aDirB.ToStdVector(); // Obs : begin bundle

    // false dont transpose, we use Cam->Word
    mPoseN.at(aIndC-mIndCamGen)->AddIdexesAndObs(aVIndGlob,aVObs,false);
    // Add in  equation forcing point to belong to bundle, taking into account the unknown pose
    mSys->R_AddEq2Subst(aStrSubst,mEqElemCamN,aVIndGlob,aVObs,aWeight);

    tREAL8 aSumRes = 0.0;
    for (size_t aK=0 ; aK<mEqElemCamN->NbElem() ; aK++)
    {
        tREAL8 aRes = std::abs(mEqElemCamN->ValComp(0,aK));
        aSumRes += Square(aRes);
        mResN.Add(1.0,aRes);
    }
    return std::sqrt(aSumRes);
}


tREAL8 cElemBA::AddEquationCoplanarity(const cPt3dr & aDirB1,const cPt3dr & aDirB2,tREAL8  aWeight)
{
      // case where use an "handrafted" linearization (for fun ? and efficiency ?)
      if (mMode==eModResBund::eLinDet12)
      {
       // !! The equation of doc is "wrong" because it assumed rotation is coded (Id+W)*R0, and in MMVII we
       // use instead R0(Id+W) so, the computation is briefly describe bellow with R' R = Id

       //  [B , Uu , R(Id+W) u2] = [R'B , R'u1 , u2 + W^u2] = [R'(B+A da + B db), R'u1 ,u2 + W ^u2]
       //   [R'B,R'u1,u2] + [R'(A da + B db),R'u1,u2] +  [R'B,R'u1,  W^u2]
       //   [B,u1,Ru2]   + [A da+ B db,u1,R u2] + R'B.(R'u1 ^(W^u2))

          cPt3dr aU02 = mRot2.Rot().Value(aDirB2); // R u2
          cPt3dr aN0 = aDirB1 ^ aU02;              // u1 ^R u2
          cPt3dr aB0 =  mTr2.RawPNorm();

          cDenseVect<tREAL8> aVect(5);

          aVect(0) = Scal(aN0, mTr2.U());  // [A ,u1,R u2]
          aVect(1) = Scal(aN0, mTr2.V());  // [B,u1,R u2]
          tREAL8 aRes = Scal(aB0,aN0);   // [B,u1,Ru2]

          cPt3dr aUP1 = mRot2.Rot().Inverse(aDirB1); // R' u1
          cPt3dr aBP  = mRot2.Rot().Inverse(aB0);  // R' B

          //  R'B.(R'u1 ^(W^u2)) = R'B (R'u1. W u2- R'u1. u2 W) =   (R'u1.u2 R'B -
          cPt3dr aScalW =   (aBP*Scal(aUP1,aDirB2) - aUP1 *Scal(aBP,aDirB2) ) * -1.0;

          aVect(2) = aScalW.x();
          aVect(3) = aScalW.y();
          aVect(4) = aScalW.z();
          if (0)
          {
            // old version due to a bug in IO_UnKnowns (vector not modified after on update)
             mSys->AddObservationLinear(aWeight,aVect,-aRes);
          }
          else
          {
            // Work now with correction on "IO_UnKnowns"
             mSystAA->PublicAddObservation(aWeight,aVect,-aRes);
          }
          mRes1.Add(1.0,std::abs(aRes));
          mRes2.Add(1.0,std::abs(aRes));
          return std::abs(aRes);
      }

      // geneal case , work with several formulas for residual
      std::vector<int> aVIndGlob ;// indexe of unknowns
      // obe begin with two bundles
      std::vector<double> aVObs = Append(aDirB1.ToStdVector(),aDirB2.ToStdVector());
      mTr2.AddIdexesAndObs(aVIndGlob,aVObs); // add obs & unknowns of translation
      mRot2.AddIdexesAndObs(aVIndGlob,aVObs,  false); // add obs & unknowns of rotation

      // Add equation in subst struct
      mSys->R_CalcAndAddObs(mEqElemCam12,aVIndGlob,aVObs,aWeight);

      // compute residual
      tREAL8 aSumRes = 0.0;
      for (size_t aK=0 ; aK<mEqElemCam12->NbElem() ; aK++)
      {
          tREAL8 aRes = std::abs(mEqElemCam12->ValComp(0,aK));

          mRes1.Add(1.0,aRes);
          mRes2.Add(1.0,aRes);
          aSumRes += Square(aRes);
      }
      return std::sqrt(aSumRes);

}



tSeg3dr  cElemBA::Bundle(int aKPose,const cPt3dr & aDirBundle) const
{
    const tPoseR & aPose = mCurPose.at(aKPose);

    // bundle , first point is center, second is position of bundle in the pose
    return tSeg3dr(aPose.Tr(),aPose.Value(aDirBundle));
}


void cElemBA::AddHomBundle_Cam1Cam2(const cPt3dr & aDirB1,const cPt3dr & aDirB2,tREAL8 aW,tREAL8 aEpsilon,tREAL8 aNoise)
{   
    // compute the 2 bundle in common repai
    tSeg3dr aSeg1 = Bundle(0,aDirB1);
    tSeg3dr aSeg2 = Bundle(1,aDirB2);

    // Test if bundles are almost paralell
    if (NormInf(aSeg1.V12() ^ aSeg2.V12()) < aEpsilon)
        return;

    if (isModeCoplan)  // Case coplanarity, juste add the equation
    {
       AddEquationCoplanarity(aDirB1,aDirB2,aW);
    }
    else
    {
        // case colinearity
       cPt3dr aPGround = BundleInters(aSeg1,aSeg2,0.5); // estimate 3D point by bundle intersection

       if (aNoise>0)
           aPGround = aPGround + cPt3dr::PRandInSphere()*aNoise;

       // create a structure where 3D point will be used, the schurr-substitued
       cSetIORSNL_SameTmp<tREAL8>  aStrSubst(aPGround.ToStdVector(),{});

       AddEquationColinearity_Cam1(aStrSubst, aDirB1,aW);  // add colinearity for cam1 in str subsr
       AddEquationColinearity_Cam2(aStrSubst, aDirB2,aW);  // add colinarity for cam2  in str subst
       mSys->R_AddObsWithTmpUK(aStrSubst);  // will add the set of equation after schurr eliminate

       if (0)
       {
          tSegComp3dr aSC1(aSeg1);
          tSegComp3dr aSC2(aSeg2);

           StdOut() << " DDD " << aSC1.Dist(aPGround) << " " << aSC2.Dist(aPGround) << "\n"; // getchar();
       }
    }
}

/*

*/
tREAL8 cElemBA::AddHom_NCam(const std::vector<int> &aVNumCams,const cPt3dr * aVDirB,const cPt3dr &aPGr,tREAL8 aW)
{
    cSetIORSNL_SameTmp<tREAL8>  aStrSubst(aPGr.ToStdVector(),{});

    tREAL8 aSumR=0.0;
    for (size_t aKInd=0 ; aKInd<aVNumCams.size() ; aKInd++)
    {
        size_t aNumCam = aVNumCams.at(aKInd);
        const cPt3dr & aDirB = aVDirB[aKInd];
        tREAL8 aRes=0;
        if (aNumCam==0)
        {
            aRes =AddEquationColinearity_Cam1(aStrSubst, aDirB,aW);  // add colinearity for cam1 in str subsr
        }
        else if (aNumCam==1)
        {
            aRes = AddEquationColinearity_Cam2(aStrSubst, aDirB,aW);  // add colinarity for cam2  in str subst
        }
        else
        {
             aRes= AddEquationColinearity_CamN(aNumCam,aStrSubst,aDirB,aW);
        }
        aSumR += Square(aRes);
    }

    mSys->R_AddObsWithTmpUK(aStrSubst);  // will add the set of equation after schurr eliminate


   // int aNbObs = 2 * aVNumCams.size();
   // tREAL8 aRatio = tREAL8(aNbObs) / (aNbObs-3.0);

    return std::sqrt(aSumR) ;
}

std::pair<tREAL8,cPt3dr> AnglesInterBundles
                         (
                               const std::vector<tPoseR> & aVPose,
                               const cPt3dr * aDirBdund,
                               tREAL8 aEpsilon
                          )
{
   std::vector<tSeg3dr> aVSeg;

   cVarPts<3> aVar;
   size_t aNbC = aVPose.size();

   for (size_t aKInd=0 ; aKInd<aNbC ; aKInd++)
   {
       const tPoseR & aPose = aVPose.at(aKInd);
       const cPt3dr & aC = aPose.Tr();
       cPt3dr aDir = aPose.Rot().Value(aDirBdund[aKInd]);
       aVar.Add(aDir);

       aVSeg.push_back(tSeg3dr(aC,aC+aDir));
   }
   if (aVar.StdDev() < aEpsilon)
   {
       return std::pair<tREAL8,cPt3dr>(-1,cPt3dr(0,0,0));
   }

   cPt3dr aPGround = BundleInters(aVSeg);
   cWeightAv<tREAL8> aAvAng;
   for (const auto & aSeg : aVSeg)
   {
       cPt3dr  aVec = aPGround-aSeg.P1();
       cPt3dr aDiff = aVec ^ aSeg.V12();
       tREAL8 aD1 = Norm2(aDiff);
       tREAL8 aD2 = Norm2(aVec);
       aAvAng.Add(1.0,aD1/(aD1+aD2+1e-8));
   }

   int aNbObs = aNbC*2;
   tREAL8 aRatio = tREAL8(aNbObs) / (aNbObs-3.0);

   return std::pair<tREAL8,cPt3dr>(aRatio*aAvAng.Average(),aPGround);
}


std::pair<tREAL8,cPt3dr> cElemBA::InterBundles
                         (
                               const std::vector<int> &aVNumCams,
                               const cPt3dr * aDirBdund,
                               tREAL8 aEpsilon
                          ) const
{
   int aNbC = aVNumCams.size();

   std::vector<tPoseR> aVPoses;
   for (int aKInd=0 ; aKInd<aNbC ; aKInd++)
     aVPoses.push_back(mCurPose.at(aVNumCams.at(aKInd)));

   return AnglesInterBundles(aVPoses,aDirBdund,aEpsilon);
}


void cElemBA::OneIter(tREAL8 aLVM)
{
    // --------- Do the computation
    const auto & aVectSol =  mSys->SolveUpdateReset(aLVM);
    // Update the current unknowns
    mSetInterv->SetVUnKnowns(aVectSol);

    // -------- Transferate the result to poses (as they have differnt struct in uknowns)
    if (mSCCam12)
    {
       mCurPose.at(1) = tPoseR(mTr2.GetPNorm(),mRot2.Rot());
    }
    for (size_t aKP=mIndCamGen ; aKP<mCurPose.size() ; aKP++)
        mCurPose.at(aKP) = mPoseN.at(aKP-mIndCamGen)->Pose();

    // Reset averages
    mRes1.Reset();
    mRes2.Reset();
    mResN.Reset();
}

/* *************************************************** */
/*                                                     */
/*                 cParamBenchElemBA                   */
/*                                                     */
/* *************************************************** */

/**  Class for parametrization of Bench on Elementary bundle adjusment */
class cParamBenchElemBA
{
    public :
      cParamBenchElemBA() ;

       eModResBund    mMode;      ///< mode of bundle
       int            mNbIter;    ///< number of iteration
       cPt2dr         mSigTrRot;  ///< Sigma on trans&rotation for simulation
       tREAL8         mLVM;       ///< Levenberg-Markard parameters
       int            mNbSamples; ///< Number of point

       //  ---- Parameters for generating points for bundles
       cPt3dr         mCenterGP;    ///< centre of ground points
       tREAL8         mRayGP;       ///< Ray of Groun Point Sphere
       tREAL8         mDistAvoid;   ///< Distance min to center of poses in
       tREAL8         mNoiseInterB; ///< Possible noise added in bundle intersection
};

cParamBenchElemBA::cParamBenchElemBA() :
   mMode             (eModResBund::eNbVals),
   mNbIter           (10),
   mSigTrRot         (0.1,0.1),
   mLVM              (1e-5),
   mNbSamples        (100),
   mCenterGP         (0,0,10.0),
   mRayGP            (5.0),
   mDistAvoid        (0.5),
   mNoiseInterB      (0)
{
}

/* *************************************************** */
/*                                                     */
/*                 cBenchElemBA                        */
/*                                                     */
/* *************************************************** */


class cGenPoseBenchElemBA
{
     public :
         cGenPoseBenchElemBA(size_t aNbPose,const cParamBenchElemBA&);

         ///  Generate Nb Sample of Set of Bundles corresp to Config; Geom of Ground Point from mParam
         std::vector<std::vector<cPt3dr>>  GenBundle(const std::vector<int>& aConfig,int aNbSample) const;

         ///  Error Tr/Rot between a sol and ground truth
         cPt2dr GroundTruthError(const std::vector<tPoseR> &) const;

          const  std::vector<tPoseR>& VPosePert() const;
     private :

         ///  Generate bundle perfect in ground truh
         std::vector<cPt3dr> GenBundle(const cPt3dr &,tREAL8,const std::vector<int>&,tREAL8) const;

         /// return the minimal distance  of center (GT&Pert) to "Pt"
         tREAL8  DMinCenter(const cPt3dr &,const std::vector<int> &aVIndexe) const;

         /// return a point in a sphere that is enough far of both GT & Pert
         cPt3dr RandomPt(const cPt3dr&,tREAL8,const std::vector<int>&,tREAL8) const;


         size_t              mNbPose;     ///< Number of pose
         cParamBenchElemBA   mParam;      ///< Parameter
         std::vector<tPoseR> mVPoseGT;    ///< Ground truth poses
         std::vector<tPoseR> mVPosePert;  ///< Perturbated poses


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

const  std::vector<tPoseR>& cGenPoseBenchElemBA::VPosePert() const {return mVPosePert;}

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


 std::vector<std::vector<cPt3dr>> cGenPoseBenchElemBA::GenBundle(const std::vector<int>& aConfig,int aNb) const
{
     std::vector<std::vector<cPt3dr>> aVVBund;
     for (int aK=0 ; aK<aNb ; aK++)
     {
         std::vector<cPt3dr> aVBund= GenBundle(mParam.mCenterGP,mParam.mRayGP,aConfig,mParam.mDistAvoid);
         aVVBund.push_back(aVBund);
     }

     return aVVBund;
}

cPt2dr cGenPoseBenchElemBA::GroundTruthError(const std::vector<tPoseR> & aVPose) const
{
    MMVII_INTERNAL_ASSERT_always(mNbPose==aVPose.size(),"cGenPoseBenchElemBA::GroundTruthError");

    cPt2dr aRes(0,0);

    for (size_t aKP=0 ; aKP<mNbPose; aKP++)
    {
        aRes.x() += Norm2(aVPose.at(aKP).Tr()-mVPoseGT.at(aKP).Tr());
        aRes.y() += aVPose.at(aKP).Rot().Dist(mVPoseGT.at(aKP).Rot());
    }

    return aRes;
}

void BenchElemBA(eModResBund aMode)
{
   cParamBenchElemBA aParam;
   if (aMode == eModResBund::eProduct)
       aParam.mSigTrRot = cPt2dr(0.03,0.01);
   if ((aMode == eModResBund::eDist12) || (aMode == eModResBund::eAngle))
       aParam.mSigTrRot = cPt2dr(0.05,0.02);
   if (aMode==eModResBund::eAng12)
       aParam.mSigTrRot = cPt2dr(0.01,0.005);

   cGenPoseBenchElemBA aBench2(2,aParam);

   cElemBA aBA(aMode,aBench2.VPosePert());

   std::vector<std::vector<cPt3dr>> aVVBund =  aBench2.GenBundle({0,1},100);


   for (int aKIter= 0 ; aKIter<40 ; aKIter++)
   {
       for (const auto & aVBund : aVVBund)
       {
          aBA.AddHomBundle_Cam1Cam2(aVBund.at(0),aVBund.at(1),1.0,1e-5,0);
       }
       aBA.OneIter(1e-5);
       cPt2dr aRes =  aBench2.GroundTruthError(aBA.CurPose());
       if (Norm2(aRes) < 1e-8)
           return;

   }
   StdOut()   << " GT/Diff " << E2Str(aMode) << " " << aBench2.GroundTruthError(aBA.CurPose()) << "\n";
   MMVII_INTERNAL_ASSERT_bench(false,"Did not reach convergence in Elem Bundle Adj");
}


void BenchElemBA()
{
   for (int aNbTest = 0 ; aNbTest<10 ; aNbTest++)
   {
       for (int aKMode=0 ; aKMode<(int)eModResBund::eNbVals ; aKMode++)
       {
             BenchElemBA(eModResBund(aKMode));
       }
   }
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
         << AOpt2007(mNoiseInterB,"NIB","Noise in bundle intersection",{eTA2007::HDV})
        ;
}


int cAppliTestElemBundle::Exe()
{
    cGenPoseBenchElemBA aBench2(2,*this);

    cElemBA aBA(mMode,aBench2.VPosePert());

    std::vector<std::vector<cPt3dr>> aVVBund =  aBench2.GenBundle({0,1},mNbSamples);


    for (int aKIter= 0 ; aKIter<mNbIter ; aKIter++)
    {
        for (const auto & aVBund : aVVBund)
        {
           aBA.AddHomBundle_Cam1Cam2(aVBund.at(0),aVBund.at(1),1.0,1e-5,mNoiseInterB);
        }
        tREAL8 aRes1 = aBA.AvgRes1() ;
        tREAL8 aRes2 = aBA.AvgRes2() ;
        aBA.OneIter(mLVM);

        StdOut()  << " RESIDUAL " << aRes1 << " " << aRes2
                   << " GT/Diff " << aBench2.GroundTruthError(aBA.CurPose())
                   << "\n";
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

