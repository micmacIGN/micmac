#include "MMVII_PoseRel.h"
#include "MMVII_Tpl_Images.h"
#include "MMVII_TplHeap.h"
#include "MMVII_HeuristikOpt.h"

namespace MMVII
{

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


class cCdtPoseRel
{
  public :
      tPoseR       mPose;
      tREAL8       mScore;
      std::string  mMsg;
};

class  cCmp_cCdtPoseRel
{
     public :
         bool operator ()(const cCdtPoseRel & aS1,const cCdtPoseRel & aS2) const
         {
             return aS1.mScore > aS2.mScore;
         }
};

class cEstimatePosRel2Im :  public cOptimizeRotAndVUnit
{
   public :
    typedef cKBestValue<cCdtPoseRel,cCmp_cCdtPoseRel> tCmpSol;

     cEstimatePosRel2Im
     (
         cPerspCamIntrCalib & aCalib1,
         cPerspCamIntrCalib & aCalib12,
         cSetHomogCpleIm      aSetHom,
         int                  aNbKBestSol
     );

     ~cEstimatePosRel2Im();
     // This method is called with posibly different algorithm & parameters
     void TestNewSol(const tPoseR&,const std::string & aMsg);

     void ShowSol(const tPoseR&,const std::string & aMsg);

  private :
    cEstimatePosRel2Im(const cEstimatePosRel2Im&) = delete;

    // Use the essentiall matrix algorithm, on all points, using L1 metric
    void  EstimPose_By_MatEssL1Glob();

    // Make one try of Ransac with aNb
    void  EstimPose_By_MatEssRansac(int aNbPts,int aNbTest);

    // use some heuris
    void EstimateHeuristik();

    //  Estimate a vector of angular residual
    void EstimateResidual(std::vector<tREAL8>& aVRes,const tPoseR& aPose) const;

    tREAL8 ScoreRotAndVect (const tRotR&,const cPt3dr &) const override;

  //  tREAL8  ScoreOfPose(const tPoseR&) const;

    cPerspCamIntrCalib &         mCalib1;
    tREAL8                       mFoc1;
    cPerspCamIntrCalib &         mCalib2;
    tREAL8                       mFoc2;
    tREAL8                       mFocMoy;
    cSetHomogCpleIm              mSetCpleHom;
    cSetHomogCpleDir             mSetCpleDir;
    const std::vector<cPt3dr>&   mVDir1;
    const std::vector<cPt3dr>&   mVDir2;
    cLinearOverCstrSys<tREAL8>*  mSysL1;
    cLinearOverCstrSys<tREAL8>*  mSysL2;

    int                          mKMaxME;
    cCmp_cCdtPoseRel             mCmpSol;
    tCmpSol                      mKBestSols;
};

cEstimatePosRel2Im::cEstimatePosRel2Im
(
    cPerspCamIntrCalib & aCalib1,
    cPerspCamIntrCalib & aCalib2,
    cSetHomogCpleIm      aSetHom,
    int                  aNbKBestSol
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
   mKBestSols  (mCmpSol,aNbKBestSol)
{

  //  EstimateHeuristik();   // far too long, we supress for now
    EstimPose_By_MatEssL1Glob();

    int aNbPtsMax = std::max(30, (int)mVDir1.size()/2);
    aNbPtsMax = std::min(100,std::min(aNbPtsMax,(int)mVDir1.size()));

    for (int aNbPt=11 ; aNbPt<aNbPtsMax ; aNbPt++)
    {
        EstimPose_By_MatEssRansac(aNbPt,20);
    }

    for (const auto & aCdt : mKBestSols.Elements())
    {
        ShowSol(aCdt.mPose,aCdt.mMsg);
    }
}


cEstimatePosRel2Im::~cEstimatePosRel2Im()
{
    delete mSysL1;
    delete mSysL2;
}

void cEstimatePosRel2Im::EstimateResidual(std::vector<tREAL8>& aVRes,const tPoseR& aPose) const
{
    cPt3dr aC2 = aPose.Tr();
    tRotR  aRot = aPose.Rot();

    aVRes.clear();
    for (size_t aKpt=0 ; aKpt<mVDir1.size() ; aKpt++)
    {
        tSeg3dr aSeg1(cPt3dr(0.0,0.0,0.0),mVDir1.at(aKpt));
        tSeg3dr aSeg2(aC2,aC2+aRot.Value(mVDir2.at(aKpt)));
        cPt3dr aCoeffI;
        BundleInters(aCoeffI,aSeg1,aSeg2);
        tREAL8 aRes = std::abs(aCoeffI.z()) / (1.0+Norm2(aCoeffI));
        aVRes.push_back(aRes);
    }
}

tREAL8  cEstimatePosRel2Im::ScoreRotAndVect (const tRotR& aRot,const cPt3dr & aTr) const
{
    std::vector<tREAL8> aVRes;
    EstimateResidual(aVRes,tPoseR(aTr,aRot));

    return RankWeigthedAverage(aVRes,1.0,false);
}
void cEstimatePosRel2Im::EstimateHeuristik()
{
    StdOut() << "BEGIN HEURISTIK \n";
      auto [aCost,aPair] =  ComputeSolInit(1.0,0.01/mFocMoy,4,10.0/mFocMoy);

      ShowSol(tPoseR(aPair.second,aPair.first),"Heuristitk");
}


void cEstimatePosRel2Im::TestNewSol(const tPoseR& aPose,const std::string & aMsg)
{
    std::vector<tREAL8> aVRes;
    EstimateResidual(aVRes,aPose);

    cCdtPoseRel aCdt;
    aCdt.mScore =    RankWeigthedAverage(aVRes,1.0,false) ;
    aCdt.mMsg = aMsg;
    aCdt.mPose = aPose;

    mKBestSols.Push(aCdt);
}

void cEstimatePosRel2Im::ShowSol(const tPoseR& aPose,const std::string & aMsg)
{
    std::vector<tREAL8> aVRes;
    EstimateResidual(aVRes,aPose);

    StdOut()  << " MSG:" << aMsg << " ";
    StdOut() << " ResRnk=" << RankWeigthedAverage(aVRes,1.0,false)*mFocMoy;
    for (const auto aProp : {0.5,0.75,0.9,0.95,0.99,1.001})
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
         bool                      mUseOri4GT;
         int                       mNbSimulPt;
         std::vector<double>       mParamOutLayer;


         tREAL8                    mFocM;
         cSetHomogCpleDir          * mCpleDir;
         int                       mKMaxME;
         tPoseR                    mGTPose;
         cSensorCamPC *            mPC1GT;
         cSensorCamPC *            mPC2GT;
};

cAppli_OriRel2Im::cAppli_OriRel2Im(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
    cMMVII_Appli  (aVArgs,aSpec),
    mPhProj       (*this),
    mCalib1       (nullptr),
    mCalib2       (nullptr),
    mEstimatePose (nullptr),
    mUseOri4GT    (false),
    mNbSimulPt    (0),
    mGTPose       (tPoseR::Identity()),
    mPC1GT        (nullptr),
    mPC2GT        (nullptr)
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
            <<  AOpt2007(mUseOri4GT,"OriGT","Set if orientation contains also exterior as a ground truth",{eTA2007::HDV})
             << AOpt2007(mNbSimulPt,"NbSimulPt","Number os fimulation point, if any",{eTA2007::HDV})
             << AOpt2007(mParamOutLayer,"OutLayers","Param for generating outlayers [Nb,Sigma]",{{eTA2007::ISizeV,"[2,2]"}})



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
     // Also in Simul
     if (mUseOri4GT || mNbSimulPt || IsInit(&mParamOutLayer))
     {
         mPC1GT = mPhProj.ReadCamPC(mIm1,true);
         mPC2GT = mPhProj.ReadCamPC(mIm2,true);
         mGTPose = mPC1GT->RelativePose(*mPC2GT);
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

    mEstimatePose = new cEstimatePosRel2Im(*mCalib1,*mCalib2,mCpleH,20);

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




