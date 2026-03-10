#include "MMVII_TplHeap.h"

#include "MMVII_PoseRel.h"
#include "MMVII_Tpl_Images.h"
#include "MMVII_HeuristikOpt.h"
#include "MMVII_DeclareAllCmd.h"

namespace MMVII
{

enum class eModePE2I
           {
              eRansac,
              ePlane1,
              ePlane2,
              eNbVals
           };

///  Basic class to store the pose between 2 images
class cCdtPoseRel2Im
{
  public :
      cCdtPoseRel2Im(const tPoseR&,eModePE2I,tREAL8 aScore,const std::string& aMsg);
      cCdtPoseRel2Im();

      tPoseR                 mPose;  ///< The pose itself
      eModePE2I              mMode;
      tREAL8                 mScore; ///< The score /residual : the smaller the better
      tREAL8                 mScore0; ///< The score before BA
      std::string            mMsg;   ///< Message for tuning
};

class cCdtFinalPoseRel2Im
{
    public :
       std::string            mIm1;
       std::string            mIm2;
       tPoseR                 mPose;
       std::string            mMsg;
       tREAL8                 mScorePix;
       std::optional<cPt2dr>  mScorePixGT;
};

void AddData(const  cAuxAr2007 & anAux,cCdtFinalPoseRel2Im & aCdt)
{
   MMVII::AddData(cAuxAr2007("Im1",anAux),aCdt.mIm1);
   MMVII::AddData(cAuxAr2007("Im2",anAux),aCdt.mIm2);
   MMVII::AddData(cAuxAr2007("Pose",anAux),aCdt.mPose);
   MMVII::AddData(cAuxAr2007("PixRnkScore",anAux),aCdt.mScorePix);
   MMVII::AddData(cAuxAr2007("Origin",anAux),aCdt.mMsg);
   MMVII::AddOptData(anAux,"GTDisTrRot",aCdt.mScorePixGT);
}



cCdtPoseRel2Im::cCdtPoseRel2Im(const tPoseR& aPose,eModePE2I aMode,tREAL8 aScore,const std::string& aMsg) :
    mPose   (aPose),
    mMode   (aMode),
    mScore  (aScore),
    mScore0 (aScore),
    mMsg    (aMsg)
{
}

cCdtPoseRel2Im::cCdtPoseRel2Im() :
    cCdtPoseRel2Im(tPoseR::Identity(),eModePE2I::eNbVals,1e20,MMVII_NONE)
{
}


///  Comparison of cCdtPoseRel2Im for use in heap
class  cCmp_cCdtPoseRel2Im
{
     public :
         /// In Heap we need to take out the worst candidate , thats why use >
         bool operator ()(const cCdtPoseRel2Im & aS1,const cCdtPoseRel2Im & aS2) const
         {
             return aS1.mScore > aS2.mScore;
         }
};

/** Class for computing pose estimation between 2 images using
 *  different algorithm . In this first raw, it's quite basic to go fast
 *  for Toronto's echeance ... and test :
 *
 *     # global essential matrix in L1 mod
 *     # ransac essential matrix
 *     # ransac  planary scene
 *     # locliazed ransac for planar scene
 *     # (deprecated)  heurisitk/combinatorial
 *
 *  Initial evaluation of solution is made on rank weighted residual :
 *  Let R1 R2  ... RN be the decreasing residual , we compute
 *      Avg (k Rk)
 *   Heuristik, but "universal" formula to un-weight outlayer.
 *
 *  Witt this formula, we select N best solution with this criteria, and
 *  make bundle on it. For weighted bundle we need a "guess" of sigma, it comes
 *  for the best ranked average.
 *
 *  Small detail, for now we select 3 vector of best sol :
 *
 *  - 1 for mat-ess
 *  - 1 for first sol of planar
 *  - 1 for second sol of planar (when we have 2, its the second accorind to rank weight)
 *
 *  It happens that when we have ground-truth , untill now the second solution on residual is also the
 *  worst on GT. But not reliable in long term with 100% planar scene.
 *
 *   The N better result
 *   It will evolve later with some refinement coded in MM-V1 :
 *      # use some ransac on essential matrix
 *      # use patches on planary scenes
 *      # make a "small&quick" bundle adj on each tested solution
 *      # maybe some special treatment of quasi-co-centric (co-centric) image
 */


/** Class for doinf all the test, not an application */

class cEstimatePosRel2Im :  public cOptimizeRotAndVUnit // Herit for combinatorial opt
{
   public :
    /// Tyope for storing N Best candidate
    typedef cKBestValue<cCdtPoseRel2Im,cCmp_cCdtPoseRel2Im> tCmpSol;

     cEstimatePosRel2Im
     (
         cPerspCamIntrCalib & aCalib1,
         cPerspCamIntrCalib & aCalib12,
         const cSetHomogCpleIm  &    aSetHomFull,
         const    cSetHomogCpleIm &     aSetHomAvg,
         const    cSetHomogCpleIm &     aSetHomSmall,
         int                  aNbKBestSol,
         cTimerSegm *          =nullptr
     );

     void GenerateAllSolution();
     void MakeBundleAdjustment(int aNbIter);
     cCdtFinalPoseRel2Im MakeDecision(bool Show);


     /// Free linear system
     ~cEstimatePosRel2Im();

     /// This method is called with posibly different algorithm & parameters
     void TestNewSol(const tPoseR&,const std::string & aMsg,eModePE2I aMode);
     void TestNewSol(cCdtPoseRel2Im);

     /// Show the solution
     void ShowSol(const tPoseR&,const std::string & aMsg);

     /// Initialize de ground truth
     void SetGT(const tPoseR&);

  private :
        //  --- Avoid unwated copies --------------------------------
    cEstimatePosRel2Im(const cEstimatePosRel2Im&) = delete;
    cEstimatePosRel2Im& operator = (const cEstimatePosRel2Im&) = delete;

    tCmpSol & KBS(eModePE2I) ;


    void GenerateMatEssSolutions();

    /// Use the essentiall matrix algorithm, on all points, using L1 metric
    void  EstimPose_By_MatEssL1Glob();

    /// Make NbTest try of Ransac with NbPts point
    void  EstimPose_By_MatEssRansac(int aNbPts,int aNbTest);

    /// Test the combinatorial/heuristik opt,  unused for now (slow and not efficient)
    void EstimateHeuristik();

    ///  Put in VRes the residual corresponding to Pose, using mVDir1/mVDir2
    void EstimateResidual(std::vector<tREAL8>& aVRes,const tPoseR& aPose,const cSetHomogCpleDir& aSetDir) const;

    ///  Score of residual, using ranking weighting
    tREAL8 RnkW_ScorePose (const tPoseR&,const cSetHomogCpleDir&) const;


    /// RnkW_ScorePose interfaced for cOptimizeRotAndVUnit
    tREAL8 ScoreRotAndVect (const tRotR&,const cPt3dr &) const override;

    ///  Elementary score, +- equiv to angle for now
    tREAL8 Score2Bundle(const cPt3dr & aP1,const cPt3dr & aP2,const tPoseR& aPose) const;

    /// return weighting of given score
    tREAL8  WeightOfScore(tREAL8 aScore) const;

    /// Refine pose by bundle adjustement
    tPoseR BA_RefineOnePose(const tPoseR &,const cSetHomogCpleDir &);

    /// Test with planary model on a set of dir
    void OneTestPlane(const cSetHomogCpleDir&,const std::string & aMsg);
    /// Use above, with index to extract dir
    void OneTestPlane(const std::vector<size_t>&,const std::string & aMsg);
    /// Make all the planar scene hypothesis
    void TestPlanar();


        //  --------  Data for camera -------------------
    cPerspCamIntrCalib &         mCalib1;    ///< Calib of first image
    tREAL8                       mFoc1;      ///< Foc first image
    cPerspCamIntrCalib &         mCalib2;    ///< Calib second image
    tREAL8                       mFoc2;      ///< Foc second image
    tREAL8                       mFocMoy;    ///< merging of 2 focal

       //  --------  Data  tie points -------------------

    cSetHomogCpleIm              mSetCpleHom;   ///< set of homologous point
    const std::vector<cHomogCpleIm> & mSetH;    ///< vecto of cpler extract from above
    cIndexHomOnP1                mIndexP1;      ///< spatial index on set of cple
    cSetHomogCpleDir             mSetSmallCpleDir;   ///<  set of 3D dir budnles pairs
    cSetHomogCpleDir             mSetAvgCpleDir;   ///<  set of 3D dir budnles pair
    cSetHomogCpleDir             mSetFullCpleDir;   ///<  set of 3D dir budnles pair

    const std::vector<cPt3dr>&   mVDir1;        ///< Dir bundle first image
    const std::vector<cPt3dr>&   mVDir2;        ///< Dir bundel second image
    int                          mNbPtsTot;


        // ----------- Data for optimization -----------------
    cLinearOverCstrSys<tREAL8>*  mSysL1;    ///< L1 syst system for mat ess
    cLinearOverCstrSys<tREAL8>*  mSysL2;    ///< L2 syst for mat ess

    int                          mKMaxME;   ///< Store result of direction  in [0-] used for Mat Ess
    cCmp_cCdtPoseRel2Im          mCmpSol;     ///< Compare Cdt for K Best
    int                          mNbKBestS;
    std::vector<tCmpSol*>        mVKBS;
  //  tCmpSol                      mKBSPlane1; ///< struct for K standard best sol
  //  tCmpSol                      mKBSPlane2; ///< struct for K best, for 2nd cdt of planar test
  //  tCmpSol                      mKBSMatEss; ///< struct for K best, for 2nd cdt of planar test

    cTimerSegm *                 mTimeSegm;    ///< Struct for time info

    tREAL8                       mBestScoreWR;  ///< Store the best score of weighted rank
    tREAL8                       mEpsBundle;    ///< Value to avoid // bundle in comp
    tREAL8                       mLVMBundle;    ///< Levenberg markard  parameter
    tPoseR                       mGTPose;       ///< Pose use for ground truh, if any
    bool                         mWithGT;       ///< Do we have a ground truth
};

/* -------------------------------------------------- */
/*        Constructor, Destructor, Accessor, Modier   */
/* -------------------------------------------------- */

cEstimatePosRel2Im::cEstimatePosRel2Im
(
    cPerspCamIntrCalib & aCalib1,
    cPerspCamIntrCalib & aCalib2,
    const cSetHomogCpleIm &     aSetHomFull,
    const cSetHomogCpleIm &     aSetHomAvg,
    const cSetHomogCpleIm &     aSetHomSmall,
    int                  aNbKBestSol,
    cTimerSegm *         aTimeSegm
) :
   cOptimizeRotAndVUnit(5,4,false),
   mCalib1     (aCalib1),
   mFoc1       (mCalib1.F()),
   mCalib2     (aCalib2),
   mFoc2       (mCalib2.F()),
   mFocMoy     ((mFoc1+mFoc2)/2.0),
   mSetCpleHom (aSetHomSmall),
   mSetH       (mSetCpleHom.SetH()),
   mIndexP1    (mSetCpleHom,3),
   mSetSmallCpleDir (mSetCpleHom,mCalib1,mCalib2),
   mSetAvgCpleDir (aSetHomAvg,mCalib1,mCalib2),
   mSetFullCpleDir (aSetHomFull,mCalib1,mCalib2),
   mVDir1      (mSetSmallCpleDir.VDir1()),
   mVDir2      (mSetSmallCpleDir.VDir2()),
   mNbPtsTot   (mVDir1.size()),
   mSysL1      (AllocL1_Barrodale<tREAL8>(9)),
   mSysL2      (new cLeasSqtAA<tREAL8>(9)),
   mKMaxME     (MatEss_GetKMax(mSetSmallCpleDir, 1e-6)),
   mNbKBestS   (aNbKBestSol),
   mVKBS       (),
   mTimeSegm   (aTimeSegm),
   mBestScoreWR (1e10),
   mEpsBundle  (1e-4),
   mLVMBundle  (1e-5),
   mWithGT     (false)
{
    for (int aKM=0 ; aKM<(int)eModePE2I::eNbVals ; aKM++)
        mVKBS.push_back(new tCmpSol(mCmpSol,mNbKBestS));

}

cEstimatePosRel2Im::~cEstimatePosRel2Im()
{
    delete mSysL1;
    delete mSysL2;
    DeleteAllAndClear(mVKBS);
}

void cEstimatePosRel2Im::SetGT(const tPoseR& aPose)
{
    mWithGT = true;
    mGTPose = aPose;
}

cEstimatePosRel2Im::tCmpSol & cEstimatePosRel2Im::KBS(eModePE2I aMode) {return *mVKBS.at(int(aMode));}

    /* -------------------------------------------------- */
    /*         Scoring & Bundle intersection              */
    /* -------------------------------------------------- */


tREAL8 cEstimatePosRel2Im::Score2Bundle(const cPt3dr & aP1,const cPt3dr & aP2,const tPoseR& aPose) const
{
    tSeg3dr aSeg1(cPt3dr(0.0,0.0,0.0),aP1);
    tSeg3dr aSeg2( aPose.Tr(),aPose.Value(aP2));
    cPt3dr aCoeffI;
    BundleInters(aCoeffI,aSeg1,aSeg2);
    int aNbBadSign = (aCoeffI.x()<0) + (aCoeffI.y()<0);
    return (std::abs(aCoeffI.z()) +aNbBadSign) / (1.0+Norm2(aCoeffI)); //+- angle
}

void cEstimatePosRel2Im::EstimateResidual(std::vector<tREAL8>& aVRes,const tPoseR& aPose,const cSetHomogCpleDir& aSetDir) const
{
    aVRes.clear();
    const auto & aVD1 = aSetDir.VDir1();
    const auto & aVD2 = aSetDir.VDir2();

    for (size_t aKpt=0 ; aKpt<aVD1.size() ; aKpt++)
    {
        tREAL8 aRes = Score2Bundle(aVD1.at(aKpt),aVD2.at(aKpt),aPose);
        aVRes.push_back(aRes);
    }
}

tREAL8  cEstimatePosRel2Im::WeightOfScore(tREAL8 aScore) const
{
   // we consider that the best ranking weighted "mBestScoreWR" is a robust estimator
    // of sigma, by the way as it is quite optimistic we multiply by 4
   return 1.0 / (1.0 + Square(aScore/(4.0*mBestScoreWR)));
}

tREAL8 cEstimatePosRel2Im::RnkW_ScorePose (const tPoseR& aPose,const cSetHomogCpleDir& aSetDir) const
{
    std::vector<tREAL8> aVRes;
    EstimateResidual(aVRes,aPose,aSetDir);

   //  return Average(aVRes); not significative gain in time & worst in result
    return RankWeigthedAverage(aVRes,1.0,false); // 1.0 Exp , false no cos transfo
}

tREAL8  cEstimatePosRel2Im::ScoreRotAndVect (const tRotR& aRot,const cPt3dr & aTr) const
{
    return RnkW_ScorePose(tPoseR(aTr,aRot),mSetSmallCpleDir);
}

   /* -------------------------------------------------- */
   /*    Combinatoriale/Heuristik => Deprecated          */
   /*  At the time being : slow & unefficient            */
   /* -------------------------------------------------- */

void cEstimatePosRel2Im::EstimateHeuristik()
{
     StdOut() << "BEGIN HEURISTIK \n";
      auto [aCost,aPair] =  ComputeSolInit(1.0,0.01/mFocMoy,4,10.0/mFocMoy);

      ShowSol(tPoseR(aPair.second,aPair.first),"Heuristitk");
}

    /* -------------------------------------------------- */
    /*         Handling Planar Case                       */
    /* -------------------------------------------------- */


void cEstimatePosRel2Im::OneTestPlane(const cSetHomogCpleDir& aSetH,const std::string & aMsg)
{   
   cPS_CompPose aPS(aSetH,false,mKMaxME,nullptr,1e-8); // create the planar solution

   std::vector<cCdtPoseRel2Im>  aVCdt; // create Cdt to order them on RnkW_ScorePose
   for (const auto & aP :  aPS.Sols().Elements())
   {
       tPoseR aPose = aP.PoseRel();
       tREAL8 aScore = RnkW_ScorePose(aPose,mSetSmallCpleDir);
       aVCdt.push_back(cCdtPoseRel2Im(aPose,eModePE2I::eNbVals,aScore,aMsg+ToStr(aSetH.VDir1().size())));
   }
   std::sort(aVCdt.begin(),aVCdt.end(),[](const auto & aC1,const auto & aC2){return aC1.mScore<aC2.mScore;});

   for (size_t aKC=0 ; aKC<aVCdt.size() ; aKC++)
   {
       cCdtPoseRel2Im aCdt = aVCdt.at(aKC);
       aCdt.mMode = (aKC==0) ? eModePE2I::ePlane1 : eModePE2I::ePlane2;
       TestNewSol(aCdt);
   }
}

void cEstimatePosRel2Im::OneTestPlane(const std::vector<size_t>& aVInd,const std::string & aMsg)
{
    std::vector<cPt3dr> aVDir1,aVDir2;

    for (const auto & anInd : aVInd)
    {
        aVDir1.push_back(mVDir1.at(anInd));
        aVDir2.push_back(mVDir2.at(anInd));
    }

    cSetHomogCpleDir aSet(aVDir1,aVDir2);
    OneTestPlane(aSet,aMsg);
}


void cEstimatePosRel2Im::TestPlanar()
{
    cAutoTimerSegm aTSMakeIm(mTimeSegm,"Planar");

    // Test full candidates
    OneTestPlane(mSetSmallCpleDir,"GlobPlane");

    // Test subset that are localized on neihgbours
    {
        // Estimate the typical ray for having 1 point in a neighbour
        cPt2di aSz = mCalib1.SzPix();
        tREAL8 aSurfPerPix = aSz.x()*aSz.y()/ mVDir1.size(); // S = Pi R^2
        tREAL8 aR0 = std::sqrt(aSurfPerPix/M_PI);

        for (int aNbTestLoc=0 ; aNbTestLoc<200 ; aNbTestLoc++)
        {
           // random point
           int  aKPt = RandUnif_N(mVDir1.size());
           cPt2dr aC = mSetCpleHom.KthHom(aKPt).mP1;

           // Random Ray and number of point in neighbourd
           tREAL8 aRay = aR0 * RandInInterval(0.5,1.5);
           int aNbTarget = std::min((int)mVDir1.size(),RandUnif_M_N(4,12));

           // get at least aNbTarget number of point
           std::vector<size_t> aVInd;
           while ((int)aVInd.size()<aNbTarget)
           {
               mIndexP1.GetIndexInNeighbourhhod(aVInd,aC,aRay);
               aRay *= 1.5;
           }
           // select a random subset of 4 point
           std::vector<int> aSet4Ind = RandSet(4,aVInd.size());
           std::vector<size_t> aVInd4;// (aVInd.begin(),aVInd.begin()+4);
           for (auto anInd : aSet4Ind)
              aVInd4.push_back(aVInd.at(anInd));

           OneTestPlane(aVInd4,"PlaneNeigh");
        }
    }

    // Test small subset completely random

    for (int aNbPts = 4 ; aNbPts<std::min(mNbPtsTot,20) ; aNbPts++)
    {
        std::vector<cSetIExtension> aVecSetInd;
        int aNbTest = 40.0 / std::sqrt(aNbPts-3);
        GenRanQsubCardKAmongN(aVecSetInd,aNbTest,aNbPts,mVDir1.size());

        for (const auto & aSet : aVecSetInd )
            OneTestPlane(aSet.mElems,"PlaneRand");
    }
}

/* -------------------------------------------------- */
/*       Essential  Matrix Stuff                      */
/* -------------------------------------------------- */


void  cEstimatePosRel2Im::EstimPose_By_MatEssL1Glob()
{
    // Estimate the essential matrix on all points
    mSysL1->PublicReset();
    cMatEssential aMatEL1(mSetSmallCpleDir,*mSysL1,mKMaxME);

    tPoseR aRes  = aMatEL1.ComputePose(mSetSmallCpleDir);
    TestNewSol(aRes,"L1GME",eModePE2I::eRansac);
}

void  cEstimatePosRel2Im::EstimPose_By_MatEssRansac(int aNbPts,int aNbTest)
{
    // Generate a random NbTest subset of Indexes with cardinality=NbPts
    std::vector<cSetIExtension> aVecSetInd;
    GenRanQsubCardKAmongN(aVecSetInd,aNbTest,aNbPts,mVDir1.size());

    // Parse all subset and Make a L2 estimation
    for (const auto & aSetInd : aVecSetInd)
    {
         //  Convert the subset of indexes in subset of bundles
         mSysL2->PublicReset();
         std::vector<cPt3dr> aSubDir1,aSubDir2;
         for (size_t anInd : aSetInd.mElems)
         {
             aSubDir1.push_back(mVDir1.at(anInd));
             aSubDir2.push_back(mVDir2.at(anInd));
         }
         cSetHomogCpleDir aSubD12(aSubDir1,aSubDir2);

         // Make the pose estimation by essential matrix on the subset of bundles
         cMatEssential aMatEL2(aSubD12,*mSysL2,mKMaxME);
         tPoseR aPose = aMatEL2.ComputePose(aSubD12);

         TestNewSol(aPose,"L2Ransac "+ToStr(aNbPts),eModePE2I::eRansac);
    }
}

void cEstimatePosRel2Im::GenerateMatEssSolutions()
{
    cAutoTimerSegm aTSMakeIm(mTimeSegm,"MatEss");

    EstimPose_By_MatEssL1Glob(); // Try L1 on all points (deceiving ... ;((

    // The "existenstial question" is "is it preferable to have very small
    // subset (with higher prob of no outlayer) or bigger one with more
    // stable estimation ?  As we canot decide, we try several cardinal,
    // and for each make several test

    int aNbPtsMax = std::max(30, mNbPtsTot/2);
    aNbPtsMax = std::min(50,std::min(aNbPtsMax,mNbPtsTot));

    for (int aNbPt=8 ; aNbPt<aNbPtsMax ; aNbPt++)
    {
        EstimPose_By_MatEssRansac(aNbPt,20);
    }
}

/* -------------------------------------------------- */
/*        Bundle Adjustement                          */
/* -------------------------------------------------- */

tPoseR cEstimatePosRel2Im::BA_RefineOnePose(const tPoseR & aPose,const cSetHomogCpleDir & aSetDir)
{
    std::vector<tPoseR> aVPose{tPoseR::Identity(),aPose};

    cElemBA aBA(eModResBund::eLinDet12,aVPose);

    const auto & aVD1 = aSetDir.VDir1();
    const auto & aVD2 = aSetDir.VDir2();

    tREAL8 aRes1=0,aRes2=0;
    for (int aKIter= 0 ; aKIter<1 ; aKIter++)
    {
        for (size_t aKP=0 ; aKP<aVD1.size() ; aKP++)
        {
           tREAL8 aScore = Score2Bundle(aVD1.at(aKP),aVD2.at(aKP),aBA.CurPose().at(1));
           tREAL8 aW = WeightOfScore(aScore);
           aBA.AddHomBundle_Cam1Cam2(aVD1.at(aKP),aVD2.at(aKP),aW,mEpsBundle,0);
        }
        aRes1 = aBA.AvgRes1() ;
        aRes2 = aBA.AvgRes2() ;
        aBA.OneIter(mLVMBundle);
   }

   FakeUseIt(aRes1);FakeUseIt(aRes2);


   return aBA.CurPose().at(1);
}

void cEstimatePosRel2Im::MakeBundleAdjustment(int aNbIter)
{
    {
       cAutoTimerSegm aTSMakeIm(mTimeSegm,"BA");

       for (int aKIter=0 ; aKIter<aNbIter ; aKIter++)
       {
           tREAL8 aBestRnkIter = mBestScoreWR;
           for (auto & aKB : mVKBS)
           {
                 for (auto & aCdt : aKB->Elements())
                 {
                     tPoseR aPose = BA_RefineOnePose(aCdt.mPose,mSetAvgCpleDir);
                     tREAL8 aScore = RnkW_ScorePose(aPose,mSetAvgCpleDir);
                     if (aScore < aCdt.mScore)
                     {
                        UpdateMin(aBestRnkIter,aScore);
                        aCdt.mPose = aPose;
                        aCdt.mScore = aScore;
                     }
                }
            }
            // StdOut() << " BEST " << mBestScoreWR << " => " << aBestRnkIter << "\n";
            UpdateMin(mBestScoreWR,aBestRnkIter);
       }
    }
}

/* -------------------------------------------------- */
/*          Global Methods                            */
/* -------------------------------------------------- */

cCdtFinalPoseRel2Im cEstimatePosRel2Im::MakeDecision(bool Show)
{
    // Parse all method and extract best score
    cCdtPoseRel2Im aBest(tPoseR::Identity(),eModePE2I::eNbVals,1e10,"NONE");
    for (auto & aKB : mVKBS)
    {
        std::vector<cCdtPoseRel2Im> aVSol = aKB->Elements();
        std::sort(aVSol.begin(),aVSol.end(),[](const auto &aC1,const auto & aC2){return aC1.mScore<aC2.mScore;});
        for (const auto & aCdt : aVSol)
        {
            if (aCdt.mScore< aBest.mScore)
                aBest = aCdt;
            if (Show)
            {
               StdOut() << "M=" << aCdt.mMsg
                        << " Sc=" << aCdt.mScore *mFocMoy
                        << " Sc0=" << aCdt.mScore0 *mFocMoy;
               if (mWithGT)
                  StdOut() << " DistGT=" << aCdt.mPose.DistPose(mGTPose,1.0) *mFocMoy;
               StdOut()   << "\n";
            }
        }
        if (Show)
           StdOut() << "------------------=======================------------------\n";
    }

    cCdtFinalPoseRel2Im aRes;

    aRes.mScorePix = aBest.mScore * mFocMoy;
    aRes.mMsg      = aBest.mMsg;
    aRes.mPose     = BA_RefineOnePose(aBest.mPose,mSetFullCpleDir);
    if (mWithGT)
    {
        tREAL8 aDTr = Norm2(aRes.mPose.Tr()-mGTPose.Tr());
        tREAL8 aDRot = aRes.mPose.Rot().Dist(mGTPose.Rot());
        aRes.mScorePixGT = cPt2dr(aDTr,aDRot)*mFocMoy;
    }

    return aRes;
}


void cEstimatePosRel2Im::TestNewSol(cCdtPoseRel2Im aCdt)
{
    UpdateMin(mBestScoreWR,aCdt.mScore);
    KBS(aCdt.mMode).Push(aCdt);
}

void cEstimatePosRel2Im::TestNewSol(const tPoseR& aPose,const std::string & aMsg,eModePE2I aMode)
{
    TestNewSol(cCdtPoseRel2Im(aPose,aMode,RnkW_ScorePose(aPose,mSetSmallCpleDir),aMsg));
}

void cEstimatePosRel2Im::ShowSol(const tPoseR& aPose,const std::string & aMsg)
{
    std::vector<tREAL8> aVRes;
    EstimateResidual(aVRes,aPose,mSetFullCpleDir);


    StdOut()  << " MSG:" << aMsg << " ";
    StdOut() << " ResRnk=" << RankWeigthedAverage(aVRes,1.0,false)*mFocMoy;
    if (mWithGT)
    {
        StdOut() << " [GT " <<  aPose.DistPose(mGTPose,1.0) * mFocMoy << "] " ;
    }
    for (const auto aProp : {0.5,0.75,0.9,1.001})
    {
        StdOut() << " [P="<< aProp << " R=" << Cst_KthVal(aVRes,aProp)*mFocMoy << "]";
    }
    StdOut() << "\n";
}



void cEstimatePosRel2Im::GenerateAllSolution()
{
    GenerateMatEssSolutions();
    TestPlanar();
    //  EstimateHeuristik();   // far too long, we supress for now
}

   /* ********************************************************** */
   /*                                                            */
   /*                 cAppli_OriRel2Im                           */
   /*                                                            */
   /* ********************************************************** */

class cAppli_OriRelTripletsOfIm : public cMMVII_Appli
{
     public :

        typedef std::pair<int,cCdtFinalPoseRel2Im> tRes1Pair;
        static const int RESULT_NO_POSE = 2007;

        typedef cIsometry3D<tREAL8>  tPose;

        cAppli_OriRelTripletsOfIm(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec,int aMode);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;
         std::vector<std::string>  Samples() const override;

     private :

         tRes1Pair  EstimatePose2IM(const std::string& aIm1,const std::string& aIm2);
         int DoPairsOf1Im();
         int DoAllPairs();

        /// Randomize the poistion of a point, while maintaining it inside camera
        cPt2dr  RandomizePt(const cPt2dr&,const cSensorCamPC&) const;

        int                       mModeCompute;
        cPhotogrammetricProject   mPhProj;
        std::string               mIm1;
        std::string               mIm2;
        std::string               mFileAllPairs;
        cSetHomogCpleIm           mCpleHFull;
        int                       mNbBig;
        cSetHomogCpleIm           mCpleHAvg;
        int                       mNbAvg;
        cSetHomogCpleIm           mCpleHSmall;
        int                       mNbSmall;





         cPerspCamIntrCalib        * mCalib1;
         cPerspCamIntrCalib        * mCalib2;
         cEstimatePosRel2Im  *     mEstimatePose;
         bool                      mShow;
         int                       mNbMinHom;
         int                       mNbSolInit;
         int                       mNbIterBA;
         int                       mNbSimulPt;
         std::vector<double>       mParamOutLayer;

         bool                      mUseOri4GT;
         std::string               mFolderOriGT;
         tPoseR                    mGTPose;
         cSensorCamPC *            mPC1GT;
         cSensorCamPC *            mPC2GT;

         cTimerSegm *              mTimeSegm ;
         std::vector<const tNamePair *> mVecPairs;

};

cAppli_OriRelTripletsOfIm::cAppli_OriRelTripletsOfIm(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec,int aMode) :
    cMMVII_Appli  (aVArgs,aSpec),
    mModeCompute  (aMode),
    mPhProj       (*this),
    mNbBig        (2000),
    mNbAvg        (500),
    mNbSmall      (150),
    mCalib1       (nullptr),
    mCalib2       (nullptr),
    mEstimatePose (nullptr),
    mShow         (aMode==0),
    mNbMinHom     (10),
    mNbSolInit    (10),
    mNbIterBA     (5),
    mNbSimulPt    (0),
    mUseOri4GT    (false),
    mGTPose       (tPoseR::Identity()),
    mPC1GT        (nullptr),
    mPC2GT        (nullptr)
{
}

// mEstimatePose

cCollecSpecArg2007 & cAppli_OriRelTripletsOfIm::ArgObl(cCollecSpecArg2007 & anArgObl)
{
     if (mModeCompute==0)
     {
         anArgObl
              << Arg2007(mIm1,"name first image",{eTA2007::FileImage})
              << Arg2007(mIm2,"name second image",{eTA2007::FileImage})
         ;
     }
     if (mModeCompute==1)
     {
         anArgObl << Arg2007(mIm1,"name first image",{eTA2007::FileImage})
                  << mPhProj.DPOriRel().ArgDirInMand()
          ;
     }
     if (mModeCompute==2)
     {
         anArgObl    << mPhProj.DPOriRel().ArgDirInMand() ;
     }

     anArgObl <<  mPhProj.DPOrient().ArgDirInMand("Input orientation for calibration")  ;

  /*   if (mModeCompute!=0)
     {
         anArgObl << mPhProj.DPOriRel().ArgDirOutMand();
     }*/

     return anArgObl;
}

cCollecSpecArg2007 & cAppli_OriRelTripletsOfIm::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return       anArgOpt
            <<  mPhProj.DPTieP().ArgDirInOpt()
            <<  mPhProj.DPGndPt2D().ArgDirInOpt()
            <<  mPhProj.DPMulTieP().ArgDirInOpt()

            <<  AOpt2007(mNbMinHom,"NbMinHom","Number minimal of homologous point required",{eTA2007::HDV})
            <<  AOpt2007(mNbSolInit,"NbSol0","Number of solution initial (before BA)",{eTA2007::HDV})
            <<  AOpt2007(mNbIterBA,"NbIterBA","Number of iteration in Bundle/Adj",{eTA2007::HDV})

            <<  AOpt2007(mUseOri4GT,"UseOriGT","Set if orientation contains also exterior as a ground truth",{eTA2007::HDV})
            <<  AOpt2007(mFolderOriGT,"OriGT","If ground truth ori != calib")
            <<  AOpt2007(mNbSimulPt,"NbSimulPt","Number os fimulation point, if any",{eTA2007::HDV})
            <<  AOpt2007(mParamOutLayer,"OutLayers","Param for generating outlayers [Nb,Sigma]",{{eTA2007::ISizeV,"[2,2]"}})
            <<  AOpt2007(mShow,"Show","Show messages",{eTA2007::HDV})
   ;
}


cPt2dr  cAppli_OriRelTripletsOfIm::RandomizePt(const cPt2dr& aP0,const cSensorCamPC& aCam) const
{
    cPt2dr aMargin(50,50);
    cPt2dr aRes = aP0 + cPt2dr::PRandC() * mParamOutLayer.at(1);
    cBox2dr aBox(aMargin,ToR(aCam.SzPix()) -aMargin);

    return aBox.Proj(aRes);
}

std::vector<std::string>  cAppli_OriRelTripletsOfIm::Samples() const
{
    if (mModeCompute==0)
    {
        return {
           std::string("MMVII OriPoseEstimRel2Im 043_1012_CalibInit.tif 043_1015_CalibInit.tif")
              + " BA1-CalibInit_311 InObjMesInstr=Pannel OriGT=BA1-CalibInit_311 NbSol0=5 OutLayers=[5,100]"
        };
    }
    return {};
}




int cAppli_OriRelTripletsOfIm::Exe()
{
    mTimeSegm = mShow ? new cTimerSegm(this) : nullptr ;
    mPhProj.FinishInit();




    if (mModeCompute==0)
    {
       EstimatePose2IM(mIm1,mIm2);
    }
    else if ((mModeCompute==1) || (mModeCompute==2))
    {
        mPhProj.DPOriRel().SetDirOutInIfNotInit();
        tNameRel  aSetOfPair = RelNameFromFile (mPhProj.OriRel_NamePairsOfAllImages(true));

        aSetOfPair.PutInVect(mVecPairs,true);
        if (mModeCompute==1)
        {
            return DoPairsOf1Im();
        }
        else
        {
            return DoAllPairs();
        }
    }

    delete mTimeSegm;
    return EXIT_SUCCESS;
}

int cAppli_OriRelTripletsOfIm::DoAllPairs()
{
    // ======= Extract, as a set, all the first images ====================
    tNameSet aSetN1;
    for (const auto & aPair : mVecPairs )
    {
        aSetN1.Add(aPair->V1());
    }
    std::vector<const std::string *> aVecStr;
    aSetN1.PutInVect(aVecStr,true);

    // =========== Parse these images to generate a list of command ============
    std::list<cParamCallSys> aListCom;
    for (const auto aPtrStr : aVecStr)
    {
        cParamCallSys aParam(cMMVII_Appli::FullBin(),TheSpec_OriRelPairsOf1m.Name(),*aPtrStr);

        for (size_t aKP=2 ; aKP<mArgv.size() ; aKP++)
        {
             aParam.AddArgs(mArgv[aKP]);
        }
        aListCom.push_back(aParam);
       // StdOut() << aParam.Com() << "\n";
    }
    ExeComParal(aListCom);
    //StdOut() <<  mArgv << "\n";

    return EXIT_SUCCESS;
}



int cAppli_OriRelTripletsOfIm::DoPairsOf1Im()
{
    std::vector<cCdtFinalPoseRel2Im> aVecRes;
    for (const auto & aPair : mVecPairs )
    {
        if (aPair->V1() == mIm1)
        {
            tRes1Pair aRes = EstimatePose2IM(aPair->V1(), aPair->V2());
            if (aRes.first == EXIT_SUCCESS)
            {
                aRes.second.mIm1 = mIm1;
                aRes.second.mIm2 = aPair->V2();
                aVecRes.push_back(aRes.second);

                SaveInFile(aRes.second, mPhProj.OriRel_NameOriPair2Images(mIm1,mIm2,false));
            }
            else if (aRes.first== RESULT_NO_POSE)
            {
            }
            else
            {
                delete mTimeSegm;
                return aRes.first;
            }
        }
    }

    SaveInFile(aVecRes, mPhProj.OriRel_NameOriAllPairsOf1Image(mIm1,false));

    // StdOut() << "SAVE IN=" << mPhProj.NamePairsOriRel(mIm1,false) << "\n";
    return EXIT_SUCCESS;
}

cAppli_OriRelTripletsOfIm::tRes1Pair
        cAppli_OriRelTripletsOfIm::EstimatePose2IM(const std::string& aIm1,const std::string& aIm2)
{
    mIm1 = aIm1;
    mIm2 = aIm2;
    OrderMinMax(mIm1,mIm2);
    mCpleHFull = cSetHomogCpleIm();


     // If we have a ground truth we initialize it
     // Also in SimulaFolderOri
     mUseOri4GT = mUseOri4GT ||  IsInit(&mFolderOriGT);
     if (mUseOri4GT || mNbSimulPt || IsInit(&mParamOutLayer) )
     {
       //  std::string aFolderOri =  IsInit(&mFolderOriGT) ? mFolderOriGT
         std::string aOriGT = ValWithDef(mFolderOriGT,mPhProj.DPOrient().DirIn());
         mPC1GT = mPhProj.ReadCamPCFromFolder(aOriGT,mIm1,true);
         mPC2GT = mPhProj.ReadCamPCFromFolder(aOriGT,mIm2,true);
         mGTPose = mPC1GT->Norm1RelativePose(*mPC2GT);
     }

     // as initialisation are both optional, used to check that one at least is used
     int aNbInit=0;
     mPhProj.ReadHomolMultiSrce(aNbInit,mCpleHFull,mIm1,mIm2);

     if (mNbSimulPt)  // case we add simulation
     {
         aNbInit++;
         for (int aK=0 ; aK<mNbSimulPt ; aK++)
              mCpleHFull.Add(mPC1GT->RandomVisibleCple(*mPC2GT)) ;
     }

     // We accept both initialization, maybe this can be usefull ??
     MMVII_INTERNAL_ASSERT_User_UndefE(aNbInit!=0,"None Homologouos init");

     if ((int)mCpleHFull.NbH()<mNbMinHom)
     {
         return tRes1Pair(RESULT_NO_POSE,cCdtFinalPoseRel2Im());
     }


     // eventualy generates outlayers
     if (IsInit(&mParamOutLayer))
     {
         for (int aK=0 ; aK<mParamOutLayer.at(0) ; aK++)
         {
             cHomogCpleIm aCple = mPC1GT->RandomVisibleCple(*mPC2GT);
             aCple.mP1 = RandomizePt(aCple.mP1,*mPC1GT);
             aCple.mP2 = RandomizePt(aCple.mP2,*mPC2GT);

            mCpleHFull.Add(aCple) ;
         }
     }

     {
         cAutoTimerSegm aATS(mTimeSegm,"Select");
         cSetHomogCpleIm aSetBig = mCpleHFull.SelectRandom(mNbBig);

         mCpleHAvg = aSetBig.SelectOnSpatialCriteria(mNbAvg);
         mCpleHSmall = mCpleHAvg.SelectOnSpatialCriteria(mNbSmall);

         if (mShow)
         {
            StdOut() << "NBPts, Full: " << mCpleHFull.NbH()
                     << " Avg:" << mCpleHAvg.NbH()
                     << " Small:" << mCpleHSmall.NbH() << "\n";
         }
     }

//     cEstimatePosRel2Im  *     mEstimatePose;

     mCalib1 =  mPhProj.InternalCalibFromImage(mIm1);
     mCalib2 =  mPhProj.InternalCalibFromImage(mIm2);

     mEstimatePose = new cEstimatePosRel2Im(*mCalib1,*mCalib2,mCpleHFull,mCpleHAvg,mCpleHSmall,mNbSolInit,mTimeSegm);
     if (mUseOri4GT)
         mEstimatePose->SetGT(mGTPose);

     mEstimatePose->GenerateAllSolution();
     mEstimatePose->MakeBundleAdjustment(mNbIterBA);

     cCdtFinalPoseRel2Im aCdt = mEstimatePose->MakeDecision(mShow);

     if (mUseOri4GT && mShow)
         mEstimatePose->ShowSol(mGTPose,"GroundTruh");

     if (mShow)
         StdOut() << "NbPts=" << mCpleHFull.NbH() << "\n";
     delete mEstimatePose;

     return tRes1Pair(EXIT_SUCCESS,aCdt);
}



/* ====================================================== */
/*               OriPoseEstimRel2Im                       */
/* ====================================================== */

tMMVII_UnikPApli Alloc_OriRel2Im(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_OriRelTripletsOfIm(aVArgs,aSpec,0));
}

cSpecMMVII_Appli  TheSpec_OriRel2Im
(
     "OriPoseEstimRel2Im",
      Alloc_OriRel2Im,
      "Estimate relative orientation of 2 images testing various different algorithms & configs",
      {eApF::Ori},
      {eApDT::TieP},
      {eApDT::Orient},
      __FILE__
);

/* ====================================================== */
/*               OriPoseEstimRelPairsOf1Im                */
/* ====================================================== */

tMMVII_UnikPApli Alloc_OriRelPairsOf1m(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_OriRelTripletsOfIm(aVArgs,aSpec,1));
}

cSpecMMVII_Appli  TheSpec_OriRelPairsOf1m
(
     "OriPoseEstimRelPairsOf1Im",
      Alloc_OriRelPairsOf1m,
      "Estimate relative orientation for all pairs of 1 image",
      {eApF::Ori},
      {eApDT::TieP},
      {eApDT::Orient},
      __FILE__
);

/* ====================================================== */
/*               OriPoseEstimRelPairsOf1Im                */
/* ====================================================== */

tMMVII_UnikPApli Alloc_OriRelAllPairs(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_OriRelTripletsOfIm(aVArgs,aSpec,2));
}

cSpecMMVII_Appli  TheSpec_OriRelAllPairs
(
     "OriPoseEstimRelAllPairs",
      Alloc_OriRelAllPairs,
      "Estimate relative orientation for all pairs of a file",
      {eApF::Ori},
      {eApDT::TieP},
      {eApDT::Orient},
      __FILE__
);

}; // MMVII




