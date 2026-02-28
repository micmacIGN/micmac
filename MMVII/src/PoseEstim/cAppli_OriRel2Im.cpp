#include "MMVII_TplHeap.h"

#include "MMVII_PoseRel.h"
#include "MMVII_Tpl_Images.h"
#include "MMVII_HeuristikOpt.h"

namespace MMVII
{

///  Basic class to store the pose between 2 images
class cCdtPoseRel2Im
{
  public :
      tPoseR       mPose;  ///< The pose itself
      tREAL8       mScore; ///< The score /residual : the smaller the better
      std::string  mMsg;   ///< Message for tuning
};

///  Comparison of cCdtPoseRel2Im for use in heap
class  cCmp_cCdtPoseRel2Im
{
     public :
         bool operator ()(const cCdtPoseRel2Im & aS1,const cCdtPoseRel2Im & aS2) const
         {
             return aS1.mScore > aS2.mScore;
         }
};

/** Class for computing pose estimation between 2 images using
 *  different algorithm . In this first raw, it's quite basic to go fast
 *  for Toronto's echeance ... and test :
 *
 *     # global essential matrix in L1 mode
 *     # global planary scene
 *
 *   It will evolve later with some refinement coded in MM-V1 :
 *      # use some ransac on essential matrix
 *      # use patches on planary scenes
 *      # make a "small&quick" bundle adj on each tested solution
 *      # maybe some special treatment of quasi-co-centric (co-centric) image
 */

class cEstimatePosRel2Im :  public cOptimizeRotAndVUnit // Herit for combinatorial opt
{
   public :
    /// Tyope for storing N Best candidate
    typedef cKBestValue<cCdtPoseRel2Im,cCmp_cCdtPoseRel2Im> tCmpSol;

     cEstimatePosRel2Im
     (
         cPerspCamIntrCalib & aCalib1,
         cPerspCamIntrCalib & aCalib12,
         cSetHomogCpleIm      aSetHom,
         int                  aNbKBestSol,
         cTimerSegm *          =nullptr
     );

     void DoAllCompute();

     /// Free linear system
     ~cEstimatePosRel2Im();

     /// This method is called with posibly different algorithm & parameters
     void TestNewSol(const tPoseR&,const std::string & aMsg);

     void ShowSol(const tPoseR&,const std::string & aMsg);

     void SetGT(const tPoseR&);

  private :
        //  --- Avoid unwated copies --------------------------------
    cEstimatePosRel2Im(const cEstimatePosRel2Im&) = delete;
    cEstimatePosRel2Im& operator = (const cEstimatePosRel2Im&) = delete;

    /// Use the essentiall matrix algorithm, on all points, using L1 metric
    void  EstimPose_By_MatEssL1Glob();

    /// Make NbTest try of Ransac with NbPts point
    void  EstimPose_By_MatEssRansac(int aNbPts,int aNbTest);

    /// Test the combinatorial/heuristik opt,  unused for now (slow and not efficient)
    void EstimateHeuristik();

    ///  Put in VRes the residual corresponding to Pose, using mVDir1/mVDir2
    void EstimateResidual(std::vector<tREAL8>& aVRes,const tPoseR& aPose) const;

    ///  Score of residual, using ranking weighting
    tREAL8 RnkW_ScorePose (const tPoseR&) const;


    /// RnkW_ScorePose interfaced for cOptimizeRotAndVUnit
    tREAL8 ScoreRotAndVect (const tRotR&,const cPt3dr &) const override;

    ///  Elementary score, +- equiv to angle for now
    tREAL8 Score2Bundle(const cPt3dr & aP1,const cPt3dr & aP2,const tPoseR& aPose) const;

    /// return weighting of given score
    tREAL8  WeightOfScore(tREAL8 aScore) const;

    ///
    tPoseR RefineOnePose(const cCdtPoseRel2Im & );

    ///
    void TestPlane(const cSetHomogCpleDir&);

        //  --------  Data for camera -------------------
    cPerspCamIntrCalib &         mCalib1;    ///< Calib of first image
    tREAL8                       mFoc1;
    cPerspCamIntrCalib &         mCalib2;
    tREAL8                       mFoc2;
    tREAL8                       mFocMoy;

       //  --------  Data  tie points -------------------

    cSetHomogCpleIm              mSetCpleHom;
    cSetHomogCpleDir             mSetCpleDir;
    const std::vector<cPt3dr>&   mVDir1;
    const std::vector<cPt3dr>&   mVDir2;

        // ----------- Data for optimization -----------------
    cLinearOverCstrSys<tREAL8>*  mSysL1;
    cLinearOverCstrSys<tREAL8>*  mSysL2;

    int                          mKMaxME;   ///< Store result of direction  in [0-] used for Mat Ess
    cCmp_cCdtPoseRel2Im          mCmpSol;
    tCmpSol                      mKBestSols;
    cTimerSegm *                 mTimeSegm;

    tREAL8                       mBestScoreWR;  ///< Store the best score of weighted rank
    tREAL8                       mEpsBundle;    ///< Value to avoid // bundle in comp
    tREAL8                       mLVMBundle;    ///< Levenberg markard  parameter
    tPoseR                       mGTPose;
    bool                         mWithGT;
};

cEstimatePosRel2Im::cEstimatePosRel2Im
(
    cPerspCamIntrCalib & aCalib1,
    cPerspCamIntrCalib & aCalib2,
    cSetHomogCpleIm      aSetHom,
    int                  aNbKBestSol,
    cTimerSegm *         aTimeSegm
) :
   cOptimizeRotAndVUnit(5,4,false),
   mCalib1     (aCalib1),
   mFoc1       (mCalib1.F()),
   mCalib2     (aCalib2),
   mFoc2       (mCalib2.F()),
   mFocMoy     ((mFoc1+mFoc2)/2.0),
   mSetCpleHom (aSetHom),
   mSetCpleDir (mSetCpleHom,mCalib1,mCalib2),
   mVDir1      (mSetCpleDir.VDir1()),
   mVDir2      (mSetCpleDir.VDir2()),
   mSysL1      (AllocL1_Barrodale<tREAL8>(9)),
   mSysL2      (new cLeasSqtAA<tREAL8>(9)),
   mKMaxME     (MatEss_GetKMax(mSetCpleDir, 1e-6)),
   mKBestSols  (mCmpSol,aNbKBestSol),
   mTimeSegm   (aTimeSegm),
   mBestScoreWR (1e10),
   mEpsBundle  (1e-4),
   mLVMBundle  (1e-5),
   mWithGT     (false)

{
}

void cEstimatePosRel2Im::SetGT(const tPoseR& aPose)
{
    mWithGT = true;
    mGTPose = aPose;
}

void cEstimatePosRel2Im::TestPlane(const cSetHomogCpleDir& aSetH)
{
   cPS_CompPose aPS(aSetH);

   for (const auto & aP :  aPS.Sols().Elements())
   {
       StdOut() << " PLANE " << RnkW_ScorePose(aP.PoseRel()) * mFocMoy
                << " DGT "   <<  aP.PoseRel().DistPose(mGTPose,1.0) * mFocMoy
                << "\n";
    }
   StdOut() << "NBSPL=" << aPS.Sols().Elements().size() << "\n";
}

void cEstimatePosRel2Im::DoAllCompute()
{
    TestPlane(mSetCpleDir);
    {
       cAutoTimerSegm aTSMakeIm(mTimeSegm,"L1MatEss");
      //  EstimateHeuristik();   // far too long, we supress for now
       EstimPose_By_MatEssL1Glob();
    }

    {
       cAutoTimerSegm aTSMakeIm(mTimeSegm,"Ransac");

       int aNbPtsMax = std::max(30, (int)mVDir1.size()/2);
       aNbPtsMax = std::min(100,std::min(aNbPtsMax,(int)mVDir1.size()));

       for (int aNbPt=11 ; aNbPt<aNbPtsMax ; aNbPt++)
       {
           EstimPose_By_MatEssRansac(aNbPt,20);
       }
    }


    StdOut() << " BEST SC=" << mBestScoreWR  * mFocMoy << "\n";

    {
       for (int aKIter=0 ; aKIter<1 ; aKIter++)
       {
          cAutoTimerSegm aTSMakeIm(mTimeSegm,"BA");

          std::vector<cCdtPoseRel2Im> aVSol = mKBestSols.Elements();
          for (const auto & aCdt : mKBestSols.Elements())
          {
             ShowSol(aCdt.mPose,aCdt.mMsg);
             RefineOnePose(aCdt);
          }
       }
    }
}


cEstimatePosRel2Im::~cEstimatePosRel2Im()
{
    delete mSysL1;
    delete mSysL2;
}

tREAL8 cEstimatePosRel2Im::Score2Bundle(const cPt3dr & aP1,const cPt3dr & aP2,const tPoseR& aPose) const
{
    tSeg3dr aSeg1(cPt3dr(0.0,0.0,0.0),aP1);
    tSeg3dr aSeg2( aPose.Tr(),aPose.Value(aP2));
    cPt3dr aCoeffI;
    BundleInters(aCoeffI,aSeg1,aSeg2);
    return  std::abs(aCoeffI.z()) / (1.0+Norm2(aCoeffI)); //+- angle
}

void cEstimatePosRel2Im::EstimateResidual(std::vector<tREAL8>& aVRes,const tPoseR& aPose) const
{
    aVRes.clear();
    for (size_t aKpt=0 ; aKpt<mVDir1.size() ; aKpt++)
    {
        tREAL8 aRes = Score2Bundle(mVDir1.at(aKpt),mVDir2.at(aKpt),aPose);
        aVRes.push_back(aRes);
    }
}

tREAL8  cEstimatePosRel2Im::WeightOfScore(tREAL8 aScore) const
{
   // we consider that the best ranking weighted "mBestScoreWR" is an estimator
    // of sigma, by the way as it is quite optimistic we multiply by 4
   return 1.0 / (1.0 + Square(aScore/(4.0*mBestScoreWR)));
}

tPoseR cEstimatePosRel2Im::RefineOnePose(const cCdtPoseRel2Im & aCdt)
{
    std::vector<tPoseR> aVPose{tPoseR::Identity(),aCdt.mPose};

    cElemBA aBA(eModResBund::eLinDet12,aVPose);

    tREAL8 aRes1=0,aRes2=0;
    for (int aKIter= 0 ; aKIter<5 ; aKIter++)
    {
        for (size_t aKP=0 ; aKP<mVDir1.size() ; aKP++)
        {
           tREAL8 aScore = Score2Bundle(mVDir1.at(aKP),mVDir2.at(aKP),aBA.CurPose().at(1));
           tREAL8 aW = WeightOfScore(aScore);
           aBA.AddHomBundle_Cam1Cam2(mVDir1.at(aKP),mVDir2.at(aKP),aW,mEpsBundle,0);
        }
        aRes1 = aBA.AvgRes1() ;
        aRes2 = aBA.AvgRes2() ;
        aBA.OneIter(mLVMBundle);
   }

   FakeUseIt(aRes1);FakeUseIt(aRes2);

   ShowSol(aBA.CurPose().at(1),"BA");

   return aBA.CurPose().at(1);
}


tREAL8 cEstimatePosRel2Im::RnkW_ScorePose (const tPoseR& aPose) const
{
    std::vector<tREAL8> aVRes;
    EstimateResidual(aVRes,aPose);

    return RankWeigthedAverage(aVRes,1.0,false); // 1.0 Exp , false no cos transfo
}

tREAL8  cEstimatePosRel2Im::ScoreRotAndVect (const tRotR& aRot,const cPt3dr & aTr) const
{
    return RnkW_ScorePose(tPoseR(aTr,aRot));
}

void cEstimatePosRel2Im::EstimateHeuristik()
{
     StdOut() << "BEGIN HEURISTIK \n";
      auto [aCost,aPair] =  ComputeSolInit(1.0,0.01/mFocMoy,4,10.0/mFocMoy);

      ShowSol(tPoseR(aPair.second,aPair.first),"Heuristitk");
}


void cEstimatePosRel2Im::TestNewSol(const tPoseR& aPose,const std::string & aMsg)
{
    cCdtPoseRel2Im aCdt;
    aCdt.mScore =   RnkW_ScorePose(aPose);
    aCdt.mMsg = aMsg;
    aCdt.mPose = aPose;

    UpdateMin(mBestScoreWR,aCdt.mScore);
    mKBestSols.Push(aCdt);
}

void cEstimatePosRel2Im::ShowSol(const tPoseR& aPose,const std::string & aMsg)
{
    std::vector<tREAL8> aVRes;
    EstimateResidual(aVRes,aPose);


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

void  cEstimatePosRel2Im::EstimPose_By_MatEssL1Glob()
{
    mSysL1->PublicReset();
    cMatEssential aMatEL1(mSetCpleDir,*mSysL1,mKMaxME);

    tPoseR aRes  = aMatEL1.ComputePose(mSetCpleDir);
    TestNewSol(aRes,"L1GME");
}

void  cEstimatePosRel2Im::EstimPose_By_MatEssRansac(int aNbPts,int aNbTest)
{
    std::vector<cSetIExtension> aVecSetInd;
    GenRanQsubCardKAmongN(aVecSetInd,aNbTest,aNbPts,mVDir1.size());

    for (const auto & aSetInd : aVecSetInd)
    {
         mSysL2->PublicReset();
         std::vector<cPt3dr> aSubDir1,aSubDir2;
         for (size_t anInd : aSetInd.mElems)
         {
             aSubDir1.push_back(mVDir1.at(anInd));
             aSubDir2.push_back(mVDir2.at(anInd));
         }
         cSetHomogCpleDir aSubD12(aSubDir1,aSubDir2);

         cMatEssential aMatEL2(aSubD12,*mSysL2,mKMaxME);
         tPoseR aPose = aMatEL2.ComputePose(aSubD12);

         TestNewSol(aPose,"L2Ransac "+ToStr(aNbPts));
    }

}



   /* ********************************************************** */
   /*                                                            */
   /*                 cAppli_OriRel2Im                           */
   /*                                                            */
   /* ********************************************************** */

class cAppli_OriRel2Im : public cMMVII_Appli
{
     public :
	typedef cIsometry3D<tREAL8>  tPose;

        cAppli_OriRel2Im(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;
     private :
        tPose  EstimPose_By_MatEssL1Glob();

        cPt2dr  RandomizePt(const cPt2dr&,const cSensorCamPC&) const;

        cPhotogrammetricProject   mPhProj;
        std::string               mIm1;
        std::string               mIm2;
        cSetHomogCpleIm           mCpleH;


         cPerspCamIntrCalib        * mCalib1;
         cPerspCamIntrCalib        * mCalib2;
         cEstimatePosRel2Im  *     mEstimatePose;

         int                       mNbSolInit;
         int                       mNbSimulPt;
         std::vector<double>       mParamOutLayer;


         tREAL8                    mFocM;
         cSetHomogCpleDir          * mCpleDir;
         int                       mKMaxME;


         bool                      mUseOri4GT;
         std::string               mFolderOriGT;
         tPoseR                    mGTPose;
         cSensorCamPC *            mPC1GT;
         cSensorCamPC *            mPC2GT;


         cTimerSegm                mTimeSegm;

};

cAppli_OriRel2Im::cAppli_OriRel2Im(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
    cMMVII_Appli  (aVArgs,aSpec),
    mPhProj       (*this),
    mCalib1       (nullptr),
    mCalib2       (nullptr),
    mEstimatePose (nullptr),
    mNbSolInit    (20),
    mNbSimulPt    (0),
    mUseOri4GT    (false),
    mGTPose       (tPoseR::Identity()),
    mPC1GT        (nullptr),
    mPC2GT        (nullptr),
    mTimeSegm     (this)
{
}

// mEstimatePose

cCollecSpecArg2007 & cAppli_OriRel2Im::ArgObl(cCollecSpecArg2007 & anArgObl)
{
    return anArgObl
              << Arg2007(mIm1,"name first image",{eTA2007::FileImage})
              << Arg2007(mIm2,"name second image",{eTA2007::FileImage})
              <<  mPhProj.DPOrient().ArgDirInMand("Input orientation for calibration")

           ;
}

cCollecSpecArg2007 & cAppli_OriRel2Im::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return       anArgOpt
            <<  mPhProj.DPTieP().ArgDirInOpt()
            <<  mPhProj.DPGndPt2D().ArgDirInOpt()
            <<  AOpt2007(mUseOri4GT,"UseOriGT","Set if orientation contains also exterior as a ground truth",{eTA2007::HDV})
            <<  AOpt2007(mFolderOriGT,"OriGT","If ground truth ori != calib")
            <<  AOpt2007(mNbSimulPt,"NbSimulPt","Number os fimulation point, if any",{eTA2007::HDV})
            <<  AOpt2007(mNbSolInit,"NbSol0","Number of solution initial (before BA)",{eTA2007::HDV})
            <<  AOpt2007(mParamOutLayer,"OutLayers","Param for generating outlayers [Nb,Sigma]",{{eTA2007::ISizeV,"[2,2]"}})



   ;
}

/*
cAppli_OriRel2Im::tPose  cAppli_OriRel2Im::EstimPose_By_MatEssL1Glob()
{
     cLinearOverCstrSys<tREAL8> *  aSysL1 = AllocL1_Barrodale<tREAL8>(9);
     cMatEssential aMatEL1(*mCpleDir,*aSysL1,mKMaxME);

     StdOut()  <<  "CCCCC= " << aMatEL1.AvgCost(*mCpleDir,0.05)  * mFocM << std::endl;
     StdOut()  <<  "CCCCC= " << aMatEL1.AvgCost(*mCpleDir,5.0/mFocM) * mFocM << std::endl;
     StdOut()  <<  "MED= " << aMatEL1.KthCost(*mCpleDir,0.5) * mFocM << std::endl;
     StdOut()  <<  "P90= " << aMatEL1.KthCost(*mCpleDir,0.9) * mFocM << std::endl;
     StdOut()  <<  "MAX= " << aMatEL1.KthCost(*mCpleDir,1.1) * mFocM << std::endl;

     aMatEL1.Show(*mCpleDir);

     tPose aRes  = aMatEL1.ComputePose(*mCpleDir);
     delete aSysL1;


     if (mUseOri4GT)
     {
         StdOut() << VUnit(mGTPose.Tr())  << aRes.Tr() << std::endl;

         mGTPose.Rot().Mat().Show();
         StdOut() << "=============================" << std::endl;
         aRes.Rot().Mat().Show();
     }

     return aRes;
}*/


cPt2dr  cAppli_OriRel2Im::RandomizePt(const cPt2dr& aP0,const cSensorCamPC& aCam) const
{
    cPt2dr aMargin(50,50);
    cPt2dr aRes = aP0 + cPt2dr::PRandC() * mParamOutLayer.at(1);
    cBox2dr aBox(aMargin,ToR(aCam.SzPix()) -aMargin);

    return aBox.Proj(aRes);
}


int cAppli_OriRel2Im::Exe()
{
     mPhProj.FinishInit();
     OrderMinMax(mIm1,mIm2);

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

     if (mPhProj.DPTieP().DirInIsInit())
     {
         aNbInit++;
         mPhProj.ReadHomol(mCpleH,mIm1,mIm2);
     }
     if (mPhProj.DPGndPt2D().DirInIsInit())
     {
         aNbInit++;
         cSetMesPtOf1Im  aSetM1 = mPhProj.LoadMeasureIm(mIm1);
         cSetMesPtOf1Im  aSetM2 = mPhProj.LoadMeasureIm(mIm2);
         mCpleH.AddPairSet(aSetM1,aSetM2);
     }
     if (mNbSimulPt)
     {
         aNbInit++;
         for (int aK=0 ; aK<mNbSimulPt ; aK++)
               mCpleH.Add(mPC1GT->RandomVisibleCple(*mPC2GT)) ;

     }

     // eventualy generates outlayers
     if (IsInit(&mParamOutLayer))
     {
         for (int aK=0 ; aK<mParamOutLayer.at(0) ; aK++)
         {
             cHomogCpleIm aCple = mPC1GT->RandomVisibleCple(*mPC2GT);
             aCple.mP1 = RandomizePt(aCple.mP1,*mPC1GT);
             aCple.mP2 = RandomizePt(aCple.mP2,*mPC2GT);

            mCpleH.Add(aCple) ;
         }
     }


     // We accept both initialization, maybe this can be usefull ??
     MMVII_INTERNAL_ASSERT_User_UndefE(aNbInit!=0,"None Homologouos init");

//     cEstimatePosRel2Im  *     mEstimatePose;

     mCalib1 =  mPhProj.InternalCalibFromImage(mIm1);
     mCalib2 =  mPhProj.InternalCalibFromImage(mIm2);

    mEstimatePose = new cEstimatePosRel2Im(*mCalib1,*mCalib2,mCpleH,mNbSolInit,&mTimeSegm);
    if (mUseOri4GT)
        mEstimatePose->SetGT(mGTPose);
    mEstimatePose->DoAllCompute();

    if (mUseOri4GT)
        mEstimatePose->ShowSol(mGTPose,"GT");
    StdOut() << " ================= OLD ===================\n";


     delete mEstimatePose;
     return EXIT_SUCCESS;
}


tMMVII_UnikPApli Alloc_OriRel2Im(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_OriRel2Im(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_OriRel2Im
(
     "OriPoseEstimRel2Im",
      Alloc_OriRel2Im,
      "Estimate relative orientation with different algorithms",
      {eApF::Ori},
      {eApDT::TieP},
      {eApDT::Orient},
      __FILE__
);



}; // MMVII




