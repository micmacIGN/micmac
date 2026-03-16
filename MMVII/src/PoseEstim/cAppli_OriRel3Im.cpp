#include "MMVII_TplHeap.h"

#include "MMVII_PoseRel.h"
#include "MMVII_Tpl_Images.h"
#include "MMVII_HeuristikOpt.h"
#include "MMVII_DeclareAllCmd.h"
#include "MMVII_Geom3D.h"
#include "MMVII_2Include_Serial_Tpl.h"

namespace MMVII
{

/**  Filter a set of homologous point on residual criteria assuming the images have been oriented.
 If there is several poses, the residual is the average weight by the inverse of the median
 (we have to find a formula ...)
*/

void FilterHomOnPropResidualAng
     (
         cSetHomogCpleIm & aSetH, // Homologous point to filter
         const std::vector<tPoseR> aVPoseRel2to1, // vector of pose, often 1
         const cPerspCamIntrCalib &aCam1,   // calibration of first image
         const cPerspCamIntrCalib &aCam2,   // calibraytion of second images
         tREAL8 aProp                       // proportion
     )
{
   size_t aNbPose = aVPoseRel2to1.size();
   std::vector<std::vector<tREAL8>> aVVA(aNbPose);

   // [1] Compute in VVA, for each pose, the vector of residual
   {
       // 1.1 make the pose interfaced for AnglesInterBundles (add Identity for first image)
      std::vector<std::vector<tPoseR>> aVVPoses;
      for (const auto & aPose : aVPoseRel2to1)
      {
         std::vector<tPoseR> aVP{tPoseR::Identity(),aPose};
         aVVPoses.push_back(aVP);
      }

      // 1.2 Compute the residual
      for (size_t aKH=0 ; aKH<aSetH.NbH() ; aKH++)
      {
          const cHomogCpleIm & aCple = aSetH.KthHom(aKH);
          cPt3dr aDir1 = aCam1.DirBundle(aCple.mP1);
          cPt3dr aDir2 = aCam1.DirBundle(aCple.mP2);
          std::vector aVDir{aDir1,aDir2};

          for (size_t aKP = 0 ; aKP<aNbPose ; aKP++ )
          {
             auto [anAng,aPG] = AnglesInterBundles(aVVPoses.at(aKP),aVDir.data(),1e-6);
             aVVA.at(aKP).push_back(anAng);
          }
      }
   }

   // [2] Compute in VA, for each cple, a weighted average of residual
   std::vector<tREAL8> aVAngles;
   {
      // 2.1  Compute the median for each pose in aVMedA
      std::vector<tREAL8> aVMedA;
      tREAL8 aSumMedA = 0.0;
      for (const auto & aVA : aVVA)
      {
          tREAL8 aMedA = ConstMediane  (aVA);
          aVMedA.push_back(aMedA);
          aSumMedA += aMedA;
      }

      // 2.2  Compute the weight Average, the formula is expected to
      //   - if the poses are ~ equivalent, they are probably all +-  good residual, and it will
      //     supress outlayr
      //   - else if on pose is much better, it will dominate the other
      for (size_t aKH=0 ; aKH<aSetH.NbH() ; aKH++)
      {
          cWeightAv<tREAL8> aAvgA;
          for (size_t aKA=0 ; aKA<aNbPose ; aKA++)
              aAvgA.Add(aSumMedA/aVMedA.at(aKA),aVVA.at(aKA).at(aKH));
           aVAngles.push_back(aAvgA.Average());
      }
   }

   // [3] Finally filter homologous over threshold
   tREAL8 aThresh =  Cst_KthVal(aVAngles,aProp);
   cSetHomogCpleIm aNewSetH;
   for (size_t aKH=0 ; aKH<aSetH.NbH() ; aKH++)
   {
       if (aVAngles.at(aKH)<= aThresh)
          aNewSetH.Add(aSetH.KthHom(aKH));
   }
   aSetH = aNewSetH;
}


/* ******************************************** */
/*                                              */
/*            cOneSolOriTriplet                 */
/*                                              */
/* ******************************************** */

class cDataSolOriTriplet
{
   public :
       cDataSolOriTriplet();

       std::vector<std::string> mVNames;
       tPoseR                mP01;
       tPoseR                mP02;
       tREAL8                mScore;  ///<  Score, Rank-Weighted
       std::optional<cPt2dr> mDistGT; ///< Distance to ground truth
};

cDataSolOriTriplet::cDataSolOriTriplet() :
    mScore   (1e8)
{
}

class cOneSolOriTriplet : public cMemCheck,
                          public cDataSolOriTriplet
{
      public :
         cOneSolOriTriplet(const std::vector<tPoseR> & aVPose);
         void SetPoses(tREAL8 aFocM,const std::vector<std::string>& aVN);

         cElemBA  mEBA;

       //  tREAL8   mScore;  ///<  Score, Rank-Weighted
       //  std::optional<cPt2dr> mDistGT; ///< Distance to ground truth

      private :
         cOneSolOriTriplet(const cOneSolOriTriplet & ) = delete;
};

cOneSolOriTriplet::cOneSolOriTriplet(const std::vector<tPoseR> & aVPose) :
    mEBA     (eModResBund::eAngle,aVPose)
  //,
   // mScore   (1e8),
   // mDistGT  (cPt2dr(-1.0,-1.0))
{
}

void cOneSolOriTriplet::SetPoses(tREAL8 aFocM,const std::vector<std::string>& aVN)
{
    mVNames = aVN;
    mP01 = mEBA.CurPose().at(1);
    mP02 = mEBA.CurPose().at(2);
    mScore *= aFocM;

    if (mDistGT.has_value())
    {
        mDistGT = mDistGT.value()*aFocM;
    }
}

void AddData(const  cAuxAr2007 & anAux,cDataSolOriTriplet & aDSOT)
{
   MMVII::StdContAddData(cAuxAr2007("Names",anAux),aDSOT.mVNames);
   MMVII::AddData(cAuxAr2007("Pose01",anAux),aDSOT.mP01);
   MMVII::AddData(cAuxAr2007("Pose02",anAux),aDSOT.mP02);
   MMVII::AddData(cAuxAr2007("ScorePix",anAux),aDSOT.mScore);
   MMVII::AddOptData(anAux,"GTDisTrRot",aDSOT.mDistGT);
}


/* ******************************************** */
/*                                              */
/*            cOriTriplets                      */
/*                                              */
/* ******************************************** */

class cOriTriplets : public cMemCheck
{
   public :
            cOriTriplets
            (
                 std::vector<std::string>,
                 const cPhotogrammetricProject &,
                 const cPt2dr& aRanTrR,
                 const std::vector<tPoseR> & aVPoseRef,
                 bool Show,
                 cTimerSegm  *
             );

            ~cOriTriplets();
            typedef std::vector<tPoseR>* tPtrVPoses;

            const cOneSolOriTriplet * BestSol() const;

   private :

            cOriTriplets(const cOriTriplets &) = delete;

            /// Compute Dir bundles in PGrounds, retunr triple point bundles
            const std::vector<cPt3dr> * ComputeDirBundles(cComputeMergeMulTieP * );
            void TestSol(const tPoseR& aPose0,const tPoseR&aPose1,const tPoseR& aPose2);

            std::vector<tPoseR> *  ReadPoseRel(int aK1,int aK2);

            void ComputeLambdaFrom3P(const std::vector<tPoseR> &,const cPt3dr& ,const cPt3dr&,const cPt3dr&);

            void OneIterSol(cComputeMergeMulTieP&,cOneSolOriTriplet &);

            std::vector<std::string>           mVNames;
            const cPhotogrammetricProject&     mPhProj;
            std::vector<cPerspCamIntrCalib*>   mVCalibs;

            cComputeMergeMulTieP *             mTiepMFull;
            cComputeMergeMulTieP *             mTiepMAvg;
            cComputeMergeMulTieP *             mTiepMSmall;


            std::map<cPt2di,tPtrVPoses>        mMapPoses;
            tPtrVPoses                         mVPoses01;
            tPtrVPoses                         mVPoses02;
            tPtrVPoses                         mVPoses12;

            std::vector<tREAL8>                mVLambda2;
            tREAL8                             mFocMoy;
            tREAL8                             mBestScore;
            cOneSolOriTriplet *                mBestSol;
            std::vector<cOneSolOriTriplet*>    mVSols;
            cPt2dr                             mRandTrR;
            const std::vector<cPt3dr> *        mVP3;
            std::vector<tPoseR>                mVPosesRef;
            bool                               mWithPoseRef;
            bool                               mShow;
            cTimerSegm*                        mTimeSegm;
};



cOriTriplets::cOriTriplets
(
        std::vector<std::string> aVNames,
        const cPhotogrammetricProject & aPhProj,
        const cPt2dr & aRanTrR,
        const std::vector<tPoseR> & aVPoseRef,
        bool                        Show,
        cTimerSegm  *               aTimeS

) :
    mVNames    (aVNames),
    mPhProj    (aPhProj),
    mVPoses01  (ReadPoseRel(0,1)),
    mVPoses02  (ReadPoseRel(0,2)),
    mVPoses12  (ReadPoseRel(1,2)),
    mFocMoy    (0.0),
    mBestScore (1e10),
    mBestSol   (nullptr),
    mRandTrR   (aRanTrR),
    mVP3       (nullptr),
    mVPosesRef (aVPoseRef),
    mWithPoseRef (mVPosesRef.size()==3),
    mShow        (Show),
    mTimeSegm    (aTimeS)
{
    //  ---------------- Read internal calibs & compute avg foc ------------------
    for (const auto & aName :  mVNames)
    {
        mVCalibs.push_back(mPhProj.InternalCalibFromImage(aName));
        mFocMoy += mVCalibs.back()->F();
    }
    mFocMoy /= mVCalibs.size();

    cAutoTimerSegm aTSFH(mTimeSegm,"FilterHom");
    // ----- read homologous points by pair, and filter them of pose residual -------------
    cMemoryInterfImportHom aMIIH;
    for (size_t aK1=0 ; aK1< mVNames.size() ; aK1++)
    {
        const std::string & aN1 = mVNames.at(aK1);
        for (size_t aK2=aK1+1 ; aK2<mVNames.size() ; aK2++)
        {
            const std::string & aN2 = mVNames.at(aK2);
            cSetHomogCpleIm aSetH;

            int aNbInit;
            mPhProj.ReadHomolMultiSrce(aNbInit,aSetH,aN1,aN2);
            MMVII_INTERNAL_ASSERT_always(aNbInit!=0,"No source for homologous point");

            FilterHomOnPropResidualAng(aSetH,*mMapPoses[cPt2di(aK1,aK2)], *mVCalibs.at(aK1),*mVCalibs.at(aK2),0.85);

            MMVII_INTERNAL_ASSERT_always(aNbInit,"No Data for homol in cOriTriplets\n");
            aMIIH.Add(aSetH,aN1,aN2);
        }
    }

    cAutoTimerSegm aTS_RH(mTimeSegm,"ReduceHom");

    mTiepMFull = new cComputeMergeMulTieP(aVNames,&aMIIH); // full tie point
    mTiepMAvg    = new cComputeMergeMulTieP(*mTiepMFull,2000); // filter 2000 on random
    mTiepMSmall    = new cComputeMergeMulTieP(500,*mTiepMAvg);  // filter 500 on spatial homogeneity


    cAutoTimerSegm aTS_DB(mTimeSegm,"DirBund");

    mVP3 = ComputeDirBundles(mTiepMAvg);
    ComputeDirBundles(mTiepMSmall);
    ComputeDirBundles(mTiepMFull);

    // StdOut() << "VP3===" << mVP3 << "\n";
    if ((mVP3==nullptr) || (mVP3->size()/3 < 5))
    {
        return;
    }

    MMVII_INTERNAL_ASSERT_always((mVP3->size()%3)==0,"Bad size for aVP3 in cOriTriplets ");

    cAutoTimerSegm aTS_IS(mTimeSegm,"InitSol");

    for (const auto & aPose01 : *mVPoses01)
    {
        for (const auto & aPose02 : *mVPoses02)
        {
            TestSol(tPoseR::Identity(),aPose01,aPose02);
           /* TestSol(mVPosesRef.at(0),mVPosesRef.at(1),mVPosesRef.at(2));
            FakeUseIt(aPose01);FakeUseIt(aPose02);*/
        }
    }


    cAutoTimerSegm aTS_BA(mTimeSegm,"BundleAdj");
    for (int aNbIter=0 ; aNbIter<4 ; aNbIter++)
    {
        cWhichMin<cOneSolOriTriplet*,tREAL8> aWMin;
        for (auto & aPtrSol : mVSols)
        {
            OneIterSol(*mTiepMSmall,*aPtrSol);
            aWMin.Add(aPtrSol,aPtrSol->mScore);
        }
        mBestSol = aWMin.IndexExtre();
        mBestScore = mBestSol->mScore;
        if (mShow)
           StdOut() << "------------------------------------\n";
    }

    for (int aNbIter=0 ; aNbIter<2 ; aNbIter++)
        OneIterSol(*mTiepMSmall,*mBestSol);
    for (int aNbIter=0 ; aNbIter<2 ; aNbIter++)
        OneIterSol(*mTiepMAvg,*mBestSol);

    OneIterSol(*mTiepMFull,*mBestSol);
    mBestSol->SetPoses(mFocMoy,aVNames);
}

cOriTriplets::~cOriTriplets()
{
    delete mTiepMFull;
    delete mTiepMAvg;
    delete mTiepMSmall;

    DeleteAllAndClear(mVSols);

    for (auto [aP,aVect] : mMapPoses)
        delete aVect;
}

const cOneSolOriTriplet * cOriTriplets::BestSol() const
{
    return mBestSol;
}


const std::vector<cPt3dr> * cOriTriplets::ComputeDirBundles(cComputeMergeMulTieP * aMTP )
{
    const std::vector<cPt3dr> * aVP3 = nullptr;
    for ( auto & [aConf,aPts] : aMTP->Pts())
    {
        size_t aNbPts = aPts.mVPIm.size();
        aPts.mVPGround.reserve(aNbPts);
        int aNbIm = aConf.size();
        MMVII_INTERNAL_ASSERT_always((aNbPts%aNbIm)==0,"Bad size in cComputeMergeMulTieP");

        for (size_t aKPts =0 ; aKPts <aNbPts ; aKPts++)
        {
            int aKIm = aKPts%aNbIm;
            aPts.mVPGround.push_back(mVCalibs.at(aKIm)->DirBundle(aPts.mVPIm.at(aKPts)));
        }
        if (aConf.size()==3)
            aVP3 = & aPts.mVPGround;
    }
    return aVP3;
}



std::vector<tPoseR>*  cOriTriplets::ReadPoseRel(int aK1,int aK2)
{
    std::string aName =  mPhProj.OriRel_NameOriPair2Images(mVNames.at(aK1),mVNames.at(aK2),true);
    cCdtFinalPoseRel2Im aSetCdt;
    ReadFromFile(aSetCdt,aName);
    //std::vector<tPoseR>*
    tPtrVPoses aResult = new std::vector<tPoseR> ;

    for (const auto & aCdt : aSetCdt.mVCdt)
        aResult->push_back(aCdt.mPose);

    mMapPoses[cPt2di(aK1,aK2)] = aResult;

    return aResult;
}


void cOriTriplets::ComputeLambdaFrom3P
    (
            const std::vector<tPoseR> & aVPose,
            const cPt3dr& aP0,
            const cPt3dr& aP1,
            const cPt3dr & aP2
    )
{
     tSeg3dr aBund0(cPt3dr(0,0,0),aP0);

     tSeg3dr aBund1(aVPose.at(1).Tr(),aVPose.at(1).Value(aP1));
     cPt3dr aCoeff;

     cPt3dr aPGround = BundleInters(aCoeff,aBund0,aBund1);

     // X TR2 + Y R2(P2) = PGround
     tSeg3dr aBund2(cPt3dr(0,0,0),aVPose.at(2).Tr());

     tSeg3dr aBund3(aPGround,aPGround+aVPose.at(2).Rot().Value(aP2));

     aPGround = BundleInters(aCoeff,aBund2,aBund3);

     mVLambda2.push_back(aCoeff.x());
    // StdOut() << " CooEfff=" << aCoeff  << "\n";
}

void cOriTriplets::TestSol
     (
            const tPoseR& aPose0,
            const tPoseR&aPose1,
            const tPoseR& aPose2
     )
{
    std::vector<tPoseR> aVPose{aPose0,aPose1,aPose2};
    // Eventually randomize
    if (IsNotNull(mRandTrR))
    {
        for (auto & aPose : aVPose)
            aPose = aPose * tPoseR(cPt3dr::PRandC()*mRandTrR.x(),tRotR::RandomRot(mRandTrR.y()));
    }

    NormalizePosesRef(aVPose);


    // compute all the possible lambdas
    for (size_t aKPt=0 ; aKPt<mVP3->size() ; aKPt+=3 )
    {
         ComputeLambdaFrom3P(aVPose,mVP3->at(aKPt),mVP3->at(aKPt+1),mVP3->at(aKPt+2));
    }

    // estimate lambda and set the value of Pose(2)
    tREAL8 aL2 = NonConstMediane(mVLambda2);
    aVPose.at(2) = tPoseR(aVPose.at(2).Tr()*aL2,aVPose.at(2).Rot());

    // Add a new solution
    mVSols.push_back(new cOneSolOriTriplet(aVPose));
    cOneSolOriTriplet & aSol = * mVSols.back();
    cElemBA & anEBA = aSol.mEBA;

    // ---- compute the vector of residual ------------------------------
    std::vector<tREAL8> aVRes;
    for ( auto & [aConf,aPts] : mTiepMSmall->Pts())
    {
        size_t aNbPts = aPts.mVPIm.size();
        int aNbIm = aConf.size();
        MMVII_INTERNAL_ASSERT_always((aNbPts%aNbIm)==0,"Bad size in cComputeMergeMulTieP");

        for (size_t aKPts =0 ; aKPts <aNbPts ; aKPts+=aNbIm)
        {
            const cPt3dr* aPtrPts = aPts.mVPGround.data()+aKPts;
            auto [aRes,aPGr] = anEBA.InterBundles(aConf,aPtrPts,1e-6);
            if (aRes>=0)
            {
               aVRes.push_back(aRes);
            }
        }
    }
    // memorize the residual of thi sol, and update best residus
    aSol.mScore =  RankWeigthedAverage(aVRes,1.0,false);
    UpdateMin(mBestScore,aSol.mScore );
}



void cOriTriplets::OneIterSol(cComputeMergeMulTieP& aMTP,cOneSolOriTriplet & aSol)
{


  cElemBA & anEBA = aSol.mEBA;
  cWeightAv<tREAL8> aResWAvg;

  for ( auto & [aConf,aPts] : aMTP.Pts())
  {
      size_t aNbPts = aPts.mVPIm.size();
      int aNbIm = aConf.size();

      for (size_t aKPts =0 ; aKPts <aNbPts ; aKPts+=aNbIm)
      {
          const cPt3dr* aPtrPts = aPts.mVPGround.data()+aKPts;
          auto [aRes1,aPGr] = anEBA.InterBundles(aConf,aPtrPts,1e-6);
          tREAL8 aW = 1.0 / (1.0 + Square(aRes1/(4.0*mBestScore)));
          tREAL8 aRes2 = anEBA.AddHom_NCam(aConf,aPtrPts,aPGr,aW);
          aResWAvg.Add(aW,aRes2);
      }
  }

  anEBA.OneIter(1e-5);
  aSol.mScore = aResWAvg.Average();

  if (mShow)
  {
      StdOut() << "OneIterSol ;; "
              << " Best:"  << mBestScore * mFocMoy
              << " Avg:"  << aResWAvg.Average() * mFocMoy;
  }
  if (mWithPoseRef)
  {
          std::vector<tPoseR> aVP = anEBA.CurPose();
          aSol.mDistGT = cPt2dr(0,0);
          cPt2dr & aDGT = aSol.mDistGT.value();
          for (size_t aKP=0 ; aKP<mVPosesRef.size() ; aKP++)
          {
              aDGT.x() += Norm2( aVP.at(aKP).Tr()-mVPosesRef.at(aKP).Tr());
              aDGT.y() += aVP.at(aKP).Rot().Dist(mVPosesRef.at(aKP).Rot());
          }
          aDGT =  (aDGT / 2.0);
          if (mShow)
             StdOut()  << " DRef=" << aDGT  * mFocMoy ;
  }
  if (mShow)
      StdOut() << "\n";
}



   /* ********************************************************** */
   /*                                                            */
   /*                 cAppli_OriRel2Im                           */
   /*                                                            */
   /* ********************************************************** */

class cAppli_OriRelTripletsOfIm : public cMMVII_Appli
{
     public :

        static const int RESULT_NO_POSE = 2007;

        typedef cIsometry3D<tREAL8>  tPose;

        cAppli_OriRelTripletsOfIm(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec,int aMode);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;
       //  std::vector<std::string>  Samples() const override;

     private :
        cOriTriplets * Do1Triplet(const std::vector<std::string> & a3Names);

        void DoTripletOf1Image();
        void DoAllTriplet();


        int                       mModeCompute;
        cPhotogrammetricProject   mPhProj;
        std::string               mIm1;
        std::string               mIm2;
        std::string               mIm3;
        cPt2dr                    mRanTrR;
        bool                      mUseOri4GT;
        std::string               mFolderOriGT;
        bool                      mShow;
        cTimerSegm  *             mTimeSegm ;

};

cAppli_OriRelTripletsOfIm::cAppli_OriRelTripletsOfIm
(
      const std::vector<std::string> & aVArgs,
      const cSpecMMVII_Appli & aSpec,
      int aMode
) :
    cMMVII_Appli  (aVArgs,aSpec),
    mModeCompute  (aMode),
    mPhProj       (*this),
    mRanTrR       (0,0),
    mUseOri4GT    (false),
    mShow         (mModeCompute==0)

{
}


cCollecSpecArg2007 & cAppli_OriRelTripletsOfIm::ArgObl(cCollecSpecArg2007 & anArgObl)
{
     if (mModeCompute==0)
     {
         anArgObl
              << Arg2007(mIm1,"name first image",{eTA2007::FileImage})
              << Arg2007(mIm2,"name second image",{eTA2007::FileImage})
              << Arg2007(mIm3,"name second image",{eTA2007::FileImage})
         ;
     }
     if (mModeCompute==1)
     {
         anArgObl << Arg2007(mIm1,"name first image",{eTA2007::FileImage})

          ;
     }
     if (mModeCompute==2)
     {
        // anArgObl     ;
     }

     anArgObl  <<  mPhProj.DPOriRel().ArgDirInMand()
               <<  mPhProj.DPOrient().ArgDirInMand("Input orientation for calibration")  ;


     return anArgObl;
}

cCollecSpecArg2007 & cAppli_OriRelTripletsOfIm::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return       anArgOpt
            <<  mPhProj.DPTieP().ArgDirInOpt()
            <<  mPhProj.DPGndPt2D().ArgDirInOpt()
            <<  mPhProj.DPMulTieP().ArgDirInOpt()
            <<  AOpt2007(mRanTrR,"RanTrR","Random for Trans&Rot (test & tune)",{eTA2007::HDV})
            <<  AOpt2007(mShow,"Show","Show details of result",{eTA2007::HDV})
            <<  AOpt2007(mUseOri4GT,"UseOriGT","Set if orientation contains also exterior as a ground truth",{eTA2007::HDV})
            <<  AOpt2007(mFolderOriGT,"OriGT","If ground truth ori != calib")
   ;
}

cOriTriplets * cAppli_OriRelTripletsOfIm::Do1Triplet(const std::vector<std::string> & a3Names)
{

    std::vector<tPoseR>      aVPoseRef;
    mUseOri4GT = mUseOri4GT ||  IsInit(&mFolderOriGT);

    if (mUseOri4GT)
    {
        std::string aOriGT = ValWithDef(mFolderOriGT,mPhProj.DPOrient().DirIn());
        for (const auto & aNameIm : a3Names)
        {
            aVPoseRef.push_back(mPhProj.ReadCamPCFromFolder(aOriGT,aNameIm,true)->Pose());
        }

        NormalizePosesRef(aVPoseRef);
    }

    cOriTriplets * anOri3 = new cOriTriplets(a3Names,mPhProj,mRanTrR,aVPoseRef,mShow,mTimeSegm);

    return anOri3;
}

void cAppli_OriRelTripletsOfIm::DoAllTriplet()
{
    tNameSet aSetN1;
    ReadFromFile(aSetN1,mPhProj.OriRel_NameAllImages(true));
    std::vector<std::string > aVecStr = ToVect(aSetN1);

    // =========== Parse these images to generate a list of command ============
    std::list<cParamCallSys> aListCom;
    for (const auto& aName : aVecStr)
    {
        cParamCallSys aParam(cMMVII_Appli::FullBin(),TheSpec_OriRelTripletsOf1m.Name(),aName);

        for (size_t aKP=2 ; aKP<mArgv.size() ; aKP++)
        {
             aParam.AddArgs(mArgv[aKP]);
        }
        aListCom.push_back(aParam);
        //StdOut() << aParam.Com() << "\n";
    }
    ExeComParal(aListCom);
    //StdOut() <<  mArgv << "\n";

}

void cAppli_OriRelTripletsOfIm::DoTripletOf1Image()
{
    std::string aName3 =  mPhProj.OriRel_NameOriAllTripletsOf1Image(mIm1,true);
    cExtSet<cTripletName> aSet3;
    ReadFromFile (aSet3,aName3);
    std::vector<const cTripletName *> aV3;
    aSet3.PutInVect(aV3,false);

    std::vector<cDataSolOriTriplet> aVData;
    for (const auto & a3 : aV3)
    {
        std::vector<std::string> aVN(a3->mNames.begin(),a3->mNames.end());
        if (mShow)
           StdOut() <<  "DoTriplet " << aVN << "\n";

        cOriTriplets * anOri3 =Do1Triplet(aVN);
        const cOneSolOriTriplet* aBSol = anOri3->BestSol();
        if (aBSol!=nullptr)
            aVData.push_back(*aBSol);

        delete anOri3;
    }
    SaveInFile(aVData,"toto.xml");
}

int cAppli_OriRelTripletsOfIm::Exe()
{
    mPhProj.FinishInit();
    mTimeSegm    = mShow ? (new cTimerSegm(this)) : nullptr ;

    if (mModeCompute==0)
    {
        cOriTriplets * anOri3 =Do1Triplet({mIm1,mIm2,mIm3});

        delete anOri3;
    }
    else if (mModeCompute==1)
    {
        DoTripletOf1Image();
    }
    else if (mModeCompute==2)
    {
        DoAllTriplet();
    }


    delete mTimeSegm;
    return EXIT_SUCCESS;
}


/* ====================================================== */
/*               OriPoseEstimRel3Im                       */
/* ====================================================== */

tMMVII_UnikPApli Alloc_OriRel3Im(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_OriRelTripletsOfIm(aVArgs,aSpec,0));
}

cSpecMMVII_Appli  TheSpec_OriRel3Im
(
     "OriPoseEstimRel3Im",
      Alloc_OriRel3Im,
      "Estimate relative orientation of 3 images testing various different algorithms & configs",
      {eApF::Ori},
      {eApDT::TieP},
      {eApDT::Orient},
      __FILE__
);

/* ====================================================== */
/*               OriPoseEstimRelTripletssOf1Im            */
/* ====================================================== */

tMMVII_UnikPApli Alloc_OriRelTripletsOf1m(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_OriRelTripletsOfIm(aVArgs,aSpec,1));
}

cSpecMMVII_Appli  TheSpec_OriRelTripletsOf1m
(
     "OriPoseEstimRelTripletssOf1Im",
      Alloc_OriRelTripletsOf1m,
      "Estimate relative orientation of all triplets of 1 image",
      {eApF::Ori},
      {eApDT::TieP,eApDT::Orient},
      {eApDT::Orient},
      __FILE__
);

/* ====================================================== */
/*               OriPoseEstimRelPairsOf1Im                */
/* ====================================================== */

tMMVII_UnikPApli Alloc_OriRelAllTriplets(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_OriRelTripletsOfIm(aVArgs,aSpec,2));
}

cSpecMMVII_Appli  TheSpec_OriRelAllTriplets
(
     "OriPoseEstimRelAllTriplets",
      Alloc_OriRelAllTriplets,
      "Estimate relative orientation for all triplets of a file",
      {eApF::Ori},
      {eApDT::TieP},
      {eApDT::Orient},
      __FILE__
);

}; // MMVII




