#include "MMVII_TplHeap.h"

#include "MMVII_PoseRel.h"
#include "MMVII_Tpl_Images.h"
#include "MMVII_HeuristikOpt.h"
#include "MMVII_DeclareAllCmd.h"
#include "MMVII_Geom3D.h"

namespace MMVII
{

class cMemoryInterfImportHom : public cInterfImportHom
{
      public :
           void GetHom(cSetHomogCpleIm &,const std::string & aNameIm1,const std::string & aNameIm2) const override;
           bool HasHom(const std::string & aNameIm1,const std::string & aNameIm2) const  override;

           void Add(const cSetHomogCpleIm &,const std::string & aNameIm1,const std::string & aNameIm2)     ;
      private :
             std::map<tSS,cSetHomogCpleIm> mMapN2Cple;
};


bool cMemoryInterfImportHom::HasHom(const std::string & aN1,const std::string & aN2) const
{
    //return BoolFind(mMapN2Cple,tSS(aN1,aN2));

    return mMapN2Cple.find(tSS(aN1,aN2)) != mMapN2Cple.end();
}


void cMemoryInterfImportHom::GetHom(cSetHomogCpleIm & aCple,const std::string & aN1,const std::string & aN2) const
{
    const auto & anIter = mMapN2Cple.find(tSS(aN1,aN2));
    aCple = anIter->second;
}

void cMemoryInterfImportHom::Add(const cSetHomogCpleIm & aCple,const std::string & aN1,const std::string & aN2)
{
     mMapN2Cple[tSS(aN1,aN2)] = aCple;
}


class cOneSolOriTriplet
{
      public :
         cOneSolOriTriplet(const std::vector<tPoseR> & aVPose);

         cElemBA  mEBA;
         tREAL8   mScRnkW;  ///<  Score, Rank-Weighted
};

cOneSolOriTriplet::cOneSolOriTriplet(const std::vector<tPoseR> & aVPose) :
    mEBA     (eModResBund::eAngle,aVPose),
    mScRnkW  (1e8)
{
}



class cOriTriplets
{
   public :
            cOriTriplets(std::vector<std::string>, const cPhotogrammetricProject &,const cPt2dr& aRanTrR);

            ~cOriTriplets();
   private :

            void TestSol(const tPoseR& aPose0,const tPoseR&aPose1,const tPoseR& aPose2);

            std::vector<tPoseR>  ReadPoseRel(int aK1,int aK2);

            void ComputeLambdaFrom3P(const std::vector<tPoseR> &,const cPt3dr& ,const cPt3dr&,const cPt3dr&);

            void OneIterSol(cOneSolOriTriplet &);

            std::vector<std::string>           mVNames;
            const cPhotogrammetricProject&     mPhProj;
            std::vector<cPerspCamIntrCalib*>   mVCalibs;
            cComputeMergeMulTieP *             mTiepMul;
            std::vector<tPoseR>                mVPoses1;
            std::vector<tPoseR>                mVPoses2;

            std::vector<tREAL8>                mVLambda2;
            tREAL8                             mFocMoy;
            tREAL8                             mBestWRank;
            std::vector<cOneSolOriTriplet*>    mVSols;
            cPt2dr                             mRandTrR;
            std::vector<cPt3dr> *              mVP3;


};


cOriTriplets::cOriTriplets
(
        std::vector<std::string> aVNames,
        const cPhotogrammetricProject & aPhProj,
        const cPt2dr & aRanTrR
) :
    mVNames    (aVNames),
    mPhProj    (aPhProj),
    mVPoses1   (ReadPoseRel(0,1)),
    mVPoses2   (ReadPoseRel(0,2)),
    mFocMoy    (0.0),
    mBestWRank (1e10),
    mRandTrR   (aRanTrR)
{
    for (const auto & aName :  mVNames)
    {
        mVCalibs.push_back(mPhProj.InternalCalibFromImage(aName));
        mFocMoy += mVCalibs.back()->F();
    }
    mFocMoy /= mVCalibs.size();

    cMemoryInterfImportHom aMIIH;

    for (size_t aK1=0 ; aK1< mVNames.size() ; aK1++)
    {
        const std::string & aN1 = mVNames.at(aK1);
        for (size_t aK2=aK1+1 ; aK2<mVNames.size() ; aK2++)
        {
            const std::string & aN2 = mVNames.at(aK2);
            cSetHomogCpleIm aSet;
            int aNbInit;
            mPhProj.ReadHomolMultiSrce(aNbInit,aSet,aN1,aN2);
            MMVII_INTERNAL_ASSERT_always(aNbInit,"No Data for homol in cOriTriplets\n");
            aMIIH.Add(aSet,aN1,aN2);
        }
    }

    mTiepMul = new cComputeMergeMulTieP(aVNames,&aMIIH);

    for ( auto & [aConf,aPts] : mTiepMul->Pts())
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
            mVP3 = & aPts.mVPGround;
        StdOut() << "CCCC " << aConf << " " << aPts.mVPIm.size() << "\n";
    }

    MMVII_INTERNAL_ASSERT_always((mVP3->size()%3)==0,"Bad size for aVP3 in cOriTriplets ");

    for (const auto & aPose1 : mVPoses1)
    {
        for (const auto & aPose2 : mVPoses2)
        {
            StdOut() << "ttTTTTSooolll \n";
              TestSol(tPoseR::Identity(),aPose1,aPose2);
        }
    }


    /*
    for (size_t aKPt=0 ; aKPt<aVP3->size() ; aKPt+=3 )
    {
         ComputeLambdaFrom3P(aVP3->at(aKPt),aVP3->at(aKPt+1),aVP3->at(aKPt+2));
    }
    tREAL8 aL2 = NonConstMediane(mVLambda2);
    mPose2 = tPoseR(mPose2.Tr()*aL2,mPose2.Rot());

    TestSol(tPoseR::Identity(),mPose1,mPose2);

    for (int aNbIter=0 ; aNbIter<5 ; aNbIter++)
        for (auto & aPtrSol : mVSols)
            OneIterSol(*aPtrSol);
    }
    */
}




std::vector<tPoseR>  cOriTriplets::ReadPoseRel(int aK1,int aK2)
{
    std::string aName =  mPhProj.OriRel_NameOriPair2Images(mVNames.at(aK1),mVNames.at(aK2),true);

    cCdtFinalPoseRel2Im aSetCdt;
    ReadFromFile(aSetCdt,aName);

    std::vector<tPoseR> aResult;
    for (const auto & aCdt : aSetCdt.mVCdt)
        aResult.push_back(aCdt.mPose);

    StdOut() << "NBBBBBSoo " << aResult.size() << "\n";
    return aResult;
}


void cOriTriplets::ComputeLambdaFrom3P(const std::vector<tPoseR> & aVPose,const cPt3dr& aP0,const cPt3dr& aP1,const cPt3dr & aP2)
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

void cOriTriplets::TestSol(const tPoseR& aPose0,const tPoseR&aPose1,const tPoseR& aPose2)
{
    std::vector<tPoseR> aVPose{aPose0,aPose1,aPose2};
    // Eventually randomize
    if (IsNotNull(mRandTrR))
    {
        for (auto & aPose : aVPose)
            aPose = aPose * tPoseR(cPt3dr::PRandC()*mRandTrR.x(),tRotR::RandomRot(mRandTrR.y()));
    }

    // Normalize
    for (auto & aPose : aVPose)
    {
        //  aP0 : C0->W  aP1 :C1->W   ##  aP0-1 * aP1 :  C1->W->C0
        aPose = aVPose.at(0).MapInverse() * aPose;
    }
    tREAL8 aScale = Norm2(aVPose.at(1).Tr());
    for (auto & aPose : aVPose)
    {
        aPose = tPoseR(aPose.Tr()/aScale,aPose.Rot());
    }


    for (size_t aKPt=0 ; aKPt<mVP3->size() ; aKPt+=3 )
    {
         ComputeLambdaFrom3P(aVPose,mVP3->at(aKPt),mVP3->at(aKPt+1),mVP3->at(aKPt+2));
    }
    tREAL8 aL2 = NonConstMediane(mVLambda2);
    aVPose.at(2) = tPoseR(aVPose.at(2).Tr()*aL2,aVPose.at(2).Rot());

    mVSols.push_back(new cOneSolOriTriplet(aVPose));

    cOneSolOriTriplet & aSol = * mVSols.back();
    cElemBA & anEBA = aSol.mEBA;

    std::vector<tREAL8> aVRes;
    cBoundVals<tREAL8> aBounds;
    for ( auto & [aConf,aPts] : mTiepMul->Pts())
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
               aBounds.Add(aRes);
            }
        }
    }
    aSol.mScRnkW =  RankWeigthedAverage(aVRes,1.0,false);
    UpdateMin(mBestWRank,aSol.mScRnkW );
    StdOut() << " RESS " << aSol.mScRnkW  * mFocMoy  << "\n";

    for (int aKIter=0 ; aKIter<6 ; aKIter++)
        OneIterSol(* mVSols.back());
}


void cOriTriplets::OneIterSol(cOneSolOriTriplet & aSol)
{
  cElemBA & anEBA = aSol.mEBA;
  cWeightAv<tREAL8> aResWAvg;
  for ( auto & [aConf,aPts] : mTiepMul->Pts())
  {
      size_t aNbPts = aPts.mVPIm.size();
      int aNbIm = aConf.size();

      for (size_t aKPts =0 ; aKPts <aNbPts ; aKPts+=aNbIm)
      {
          const cPt3dr* aPtrPts = aPts.mVPGround.data()+aKPts;
          auto [aRes1,aPGr] = anEBA.InterBundles(aConf,aPtrPts,1e-6);
          tREAL8 aW = 1.0 / (1.0 + Square(aRes1/(4.0*mBestWRank)));
          tREAL8 aRes2 = anEBA.AddHom_NCam(aConf,aPtrPts,aPGr,aW);
          aResWAvg.Add(aW,aRes2);
      }
  }
  anEBA.OneIter(0.0);

  StdOut() << "OneIterSol ;; "
           << " Rnk:"  << mBestWRank * mFocMoy
           << " Avg:"  << aResWAvg.Average() * mFocMoy
           << " RATIO=" <<   (mBestWRank / aResWAvg.Average())
           << "\n";
}


cOriTriplets::~cOriTriplets()
{
    delete mTiepMul;
    DeleteAllAndClear(mVSols);
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


        int                       mModeCompute;
        cPhotogrammetricProject   mPhProj;
        std::string               mIm1;
        std::string               mIm2;
        std::string               mIm3;
        cPt2dr                    mRanTrR;
};

cAppli_OriRelTripletsOfIm::cAppli_OriRelTripletsOfIm(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec,int aMode) :
    cMMVII_Appli  (aVArgs,aSpec),
    mModeCompute  (aMode),
    mPhProj       (*this),
    mRanTrR       (0,0)
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
              << Arg2007(mIm3,"name second image",{eTA2007::FileImage})
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

   ;
}

int cAppli_OriRelTripletsOfIm::Exe()
{
    mPhProj.FinishInit();

    cOriTriplets anOri3({mIm1,mIm2,mIm3},mPhProj,mRanTrR);


  //  delete mTimeSegm;
    return EXIT_SUCCESS;
}


/* ====================================================== */
/*               OriPoseEstimRel2Im                       */
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


}; // MMVII




