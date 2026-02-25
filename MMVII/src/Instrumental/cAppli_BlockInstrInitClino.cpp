#include "MMVII_InstrumentalBlock.h"
#include "cMMVII_Appli.h"
#include "MMVII_2Include_Serial_Tpl.h"
#include "MMVII_HeuristikOpt.h"


/**
  \file cInstrumentalBloc.cpp


  \brief This file contains the core implemantation of Block of rigid instrument
 
*/

namespace MMVII
{

#if (0)
class cCdtInitV
{
     public :
        tREAL8 SqDist(const cCdtInitV&) const;
        cPt4dr mQuat;
        cPt3dr mVecOnSph;
        tREAL8 mScore;
        bool   mFree;
};


tREAL8 cCdtInitV::SqDist(const cCdtInitV& aCdt) const
{
    return SqN2(mQuat-aCdt.mQuat) + SqN2(mVecOnSph-aCdt.mVecOnSph);
}


/** Extract the best "Free" Candidate, and marq all its neighboor in Ray aDistMin as non free */
cCdtInitV * ExtractMin(std::vector<cCdtInitV> &  aVCdt,tREAL8 aDistMin,tREAL8 * aScMaxNeigh=nullptr)
{
    cCdtInitV * aRes = nullptr;
    tREAL8      aScMin = 1e20;

    // extract the minimal cdt that is still free
    for (auto & aCdt : aVCdt)
    {
        if (aCdt.mFree && (aCdt.mScore < aScMin))
        {
            aRes = &aCdt;
            aScMin = aCdt.mScore;
        }
    }

    // if we got it, marks its neighbours as not free

    if (aRes)
    {
        if (aScMaxNeigh)
            *aScMaxNeigh = aScMin;
        tREAL8 aD2 = Square(aDistMin);
        for (auto & aCdt : aVCdt)
            if (aCdt.mFree && (aCdt.SqDist(*aRes) < aD2))
            {
                aCdt.mFree = 0;
                if (aScMaxNeigh)
                    *aScMaxNeigh = std::max(*aScMaxNeigh,aCdt.mScore);
            }
    }

    return aRes;
}



cOptimizeRotAndVUnit::cOptimizeRotAndVUnit(int aNbSamleRot,int aNbSampleSphere,bool isSignAmb) :
    mNbSampleRot    (aNbSamleRot),
    mNbSampleSphere (aNbSampleSphere),
    mIsSignAmb      (isSignAmb),
    mCurV0          (cPt3dr(1,0,0),"","")
{

}

std::pair<tRotR,cPt3dr>  cOptimizeRotAndVUnit::P5toRotP3(const cPtxd<tREAL8,5>& aPt) const
{
    cPt3dr aWPK(aPt.x(),aPt.y(),aPt.z());
    cPt2dr aDVert(aPt.t(),aPt.u());

    return std::pair<tRotR,cPt3dr>(mCurRot0*tRotR::RotFromWPK(aWPK), mCurV0.GetPNorm(aDVert));
}

cPt1dr  cOptimizeRotAndVUnit::Value(const cPtxd<tREAL8,5>& aPt5) const
{
   auto [aRot,aVec] = P5toRotP3(aPt5);
   return cPt1dr(ScoreRotAndVect(aRot,aVec));
}
std::pair<tREAL8,std::pair<tRotR,cPt3dr>>
           cOptimizeRotAndVUnit::ComputeSolInit(tREAL8 aDistNeigh,tREAL8 anEpsilon,int aNbTest,tREAL8 aDeltaSc)
{
    //  Create object for sampling Rot & Sphere
    cSampleQuat     aSQuat(mNbSampleRot,true); // true => sampling for rotation
    // mIsSignAmb : projective, if V and -V are equivalent
    cSampleSphere3D aSSph(mNbSampleSphere,mIsSignAmb);


    std::vector<cCdtInitV> mVCdts; // will store the score, for extracting multiple max loc
    int aNbCdt = aSQuat.NbRot()*aSSph.NbSamples();
    mVCdts.reserve(aNbCdt);

    //  Parse the rotation and the sphere and store the score in mVCdts
    for (int aKQ=0 ; aKQ<(int)aSQuat.NbRot() ; aKQ++)
    {
        cPt4dr aQuat =   aSQuat.KthQuat(aKQ);
        tRotR aRot(Quat2MatrRot(aQuat),false);
        for (int aKV=0 ; aKV<aSSph.NbSamples() ; aKV++)
        {
            cPt3dr aVecOnSph = aSSph.KthPt(aKV);
            tREAL8 aScore = ScoreRotAndVect(aRot,aVecOnSph);

            cCdtInitV aCdt;
            aCdt.mVecOnSph = aVecOnSph;
            aCdt.mQuat = aQuat;
            aCdt.mScore = aScore;
            aCdt.mFree  = true;
            mVCdts.push_back(aCdt);
        }
    }

    cCdtInitV * aCdtMin =  ExtractMin(mVCdts,aDistNeigh); // Extract the best candidate
    MMVII_INTERNAL_ASSERT_always(aCdtMin!=nullptr,"No cdt in InitVerticalAnd2OthogClino");
    tREAL8 aThSc0 =  2.0* aCdtMin->mScore + aDeltaSc;

    tREAL8 aBestScore = 1e10;
    tRotR aRotOpt;
    cPt3dr aVectOpt;
    while (aNbTest>0)
    {
        mCurRot0 = tRotR (Quat2MatrRot(aCdtMin->mQuat),false);
        mCurV0.SetPNorm(aCdtMin->mVecOnSph);

        cOptimByStep<5> anOptim(*this,true,1.0);
        tREAL8 aStep0 = 1.0 /(mNbSampleRot+mNbSampleSphere);
        auto [aScOpt,aPt5Opt] = anOptim.Optim(cPtxd<tREAL8,5>::PCste(0),aStep0,anEpsilon,1/sqrt(2.0));
        if (aScOpt<aBestScore)
        {
            aBestScore = aScOpt;
            auto aPair = P5toRotP3(aPt5Opt);
            aRotOpt = aPair.first;
            aVectOpt = aPair.second;
        }
        aCdtMin =  ExtractMin(mVCdts,aDistNeigh);

        if (aCdtMin==nullptr)
        {
            aNbTest=0;
        }
        else
        {
           if (aCdtMin->mScore>aThSc0)
              aNbTest=0;
        }
        aNbTest--;
    }

    return std::pair<tREAL8,tSol>(aBestScore,tSol(aRotOpt,aVectOpt));
//    StdOut()  << " --VVV=" << aVectOpt << " D1=" << aRotOpt.AxeI() << " " << aRotOpt.AxeJ() << "\n";
}
#endif

/* *************************************************************** */
/*                                                                 */
/*               cAppli_BlockInstrInitClino                          */
/*                                                                 */
/* *************************************************************** */
class cAppli_BlockInstrInitClino;
class cABIII_Optim2CV ;




class cAppli_BlockInstrInitClino : public cMMVII_Appli,
                                   public cDataMapping<tREAL8,2,1>,
                                   public cDataMapping<tREAL8,3,1>

        //                           public cDataMapping<tREAL8,5,1>

{
     public :
        friend cABIII_Optim2CV;


        cAppli_BlockInstrInitClino(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli &);
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override;
        int Exe() override;
        // std::vector<std::string>  Samples() const ;
     private :
        tREAL8 Ang4Show(tREAL8) const;

        // =============  Methods for individual clino ====================

           /// Calibrate a single clino independantly of others
        void Process1ClinoIndep(int aK);
           /// Score of calibration "aDir"  for clino "aKClino"
        tREAL8 ScoreDirClino(const cPt3dr& aDir,size_t aKClino) const;
           /// Interface to  ScoreDirClino for cDataMapping  R2 -> R
        cPt1dr  Value(const cPt2dr&) const override ;




        // =============  Methods for orthohonal clino ====================

        /// Calibrate a pair of clino forced to be orthog
        void Process2OrthogClino(const cPt2di &);
        /// Score of pair of orthog clino coded as a rotation
        tREAL8 ValueOf2OthogClino(const tRotR& aR) const;
        /// Score of an WPK, with signature for cDataMapping
        cPt1dr  Value(const cPt3dr&) const override ;

        // =============  Methods for 2 orthogonal clino + Vertical ====================

        void InitVerticalAnd2OthogClino(const cPt2di & aK1K2);

        cPt3dr  P2toP3(const cPt2dr&) const;
        tRotR   P3toRot(const cPt3dr&) const;

        cPhotogrammetricProject   mPhProj;

        // -------------------------- Mandatory parameters ---------------------------
        std::string               mSpecImIn;  //< images

        // ------------- Optionnal parameters ----------------
        std::string               mNameBloc;     //< name of the bloc inside the
        int                       mNbSamp1Cl;     //< Number of sample of sphere for 1 clino (in each face of cube)
        cPt2di                    mNbSVert;      //< Nb Sample Vert&Rot, in case we guess vertical
        std::vector<cPt2di>       mPairOrthog;   //< vector of orthog if we change (more for test)
        eTyUnitAngle              mUnitShow;     //< Unity for print result
        bool                      mReSetSigma;   //< do we use sigma

        //  ----------
        cIrbComp_Block *          mBlock;        //< Bloc : Calib + Data of time stamp
        cIrbCal_Block *           mCalBlock;     //< Calibration of above block
        cIrbCal_ClinoSet *        mSetClinos ;   //< Set of clino of above calib
        const int                 mNbClino;      //< Number of clinos



        int                       mKCurClino;     //< Index of cur clino, fix by DoOneClino
        int                       mK1CurC;        //< Case we have 2 clino, Index of Clin1
        int                       mK2CurC;        //<                      , Index of Clin2

        tRotR                     mAxes2Orthog;    //< code 2 pair of orthog clino as a rotation
        tRotR                     mAxes1Clino;     //< code 1 Clino and its 2 orthog compl or code for Vertical
        // tRotR                     mAxesVertical;   //< code vertical and its 2 orthog compl

        std::vector<int>          mNumPoseInstr;  //< Num cams used for pose estimation of instrument
        std::vector<bool>         mInPairOrthog;  //< Is the clino used in a pair (unused 4 now)

        cWeightAv<tREAL8,tREAL8>  mAvgClinIndep;
        std::vector<tREAL8>       mScoreClinoIndep;
        std::vector<cPt3dr>       mDirClinoIndep;

        cWeightAv<tREAL8,tREAL8>  mAvgClinGlob;
        std::vector<tREAL8>       mScoreClinoGlob;
        std::vector<cPt3dr>       mDirClinoGlob;

        cPt3dr                    mVertApriori;


};



cAppli_BlockInstrInitClino::cAppli_BlockInstrInitClino
(
        const std::vector<std::string> &  aVArgs,
        const cSpecMMVII_Appli & aSpec
) :
    cMMVII_Appli (aVArgs,aSpec),
    mPhProj      (*this),
    mNameBloc    (cIrbCal_Block::theDefaultName),
    mNbSamp1Cl   (100),
    mNbSVert     (7,7),
    mUnitShow    (eTyUnitAngle::eUA_DMgon),
    mReSetSigma  (true),
    mBlock       (nullptr),
    mCalBlock    (nullptr),
    mSetClinos   (nullptr),
    mNbClino     (-1),
    mKCurClino   (-1),
    mVertApriori (0,0,1)
{
}

cCollecSpecArg2007 & cAppli_BlockInstrInitClino::ArgObl(cCollecSpecArg2007 & anArgObl)
{
     return anArgObl
             <<  Arg2007(mSpecImIn,"Pattern/file for images", {{eTA2007::MPatFile,"0"},{eTA2007::FileDirProj}}  )
             <<  mPhProj.DPBlockInstr().ArgDirInMand()
             <<  mPhProj.DPOrient().ArgDirInMand()
             <<  mPhProj.DPMeasuresClino().ArgDirInMand()
             <<  mPhProj.DPBlockInstr().ArgDirOutMand()
     ;
}

cCollecSpecArg2007 & cAppli_BlockInstrInitClino::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
	return anArgOpt
            << AOpt2007(mNameBloc,"NameBloc","Name of bloc to calib ",{{eTA2007::HDV}})
            << AOpt2007(mNbSamp1Cl,"NbSS","Number of sample on the sphere ",{{eTA2007::HDV}})
            << AOpt2007(mNumPoseInstr,"NPI","Num of cams used  for estimate pose of intsrument")
            << AOpt2007(mPairOrthog,"PairOrthog","Pair of orthogonal camera (if reset)")
            << AOpt2007(mUnitShow,"USA","Unity Show Angles",{AC_ListVal<eTyUnitAngle>(),{eTA2007::HDV}})
            << AOpt2007(mReSetSigma,"ResetSigma","Do we use sigma",{{eTA2007::HDV}})
            << AOpt2007(mVertApriori,"Vert0","A priori vertical => we compte both clino & vert")
            << AOpt2007(mNbSVert,"NbSV","Number of sample, when both Orthog & Vert, x=>quat, y=>Spher")
    ;
}


tREAL8 cAppli_BlockInstrInitClino::Ang4Show(tREAL8 anAng) const
{

   return AngleFromRad(anAng,mUnitShow);
}

// ========================  5D optimisation ==========================

class cABIII_Optim2CV : public cOptimizeRotAndVUnit
{
  public :
    cABIII_Optim2CV(cAppli_BlockInstrInitClino &,int aNbSamleRot,int aNbSampleSphere);
  private :
    tREAL8 ScoreRotAndVect (const tRotR&,const cPt3dr &) const override;
    cAppli_BlockInstrInitClino & mAppli;
};

cABIII_Optim2CV::cABIII_Optim2CV(cAppli_BlockInstrInitClino & anAppli,int aNbSamleRot,int aNbSampleSphere) :
    cOptimizeRotAndVUnit(aNbSamleRot,aNbSampleSphere,true),
    mAppli (anAppli)
{
}

tREAL8 cABIII_Optim2CV::ScoreRotAndVect (const tRotR& aR2Clin,const cPt3dr & aVertical) const
{
    mAppli.mBlock->SetVerticalCste(VUnit(aVertical));
    return mAppli.ValueOf2OthogClino(aR2Clin);
}


void cAppli_BlockInstrInitClino::InitVerticalAnd2OthogClino(const cPt2di & aK1K2)
{
    mK1CurC = aK1K2.x();
    mK2CurC = aK1K2.y();

    cABIII_Optim2CV anOpt(*this,mNbSVert.x(),mNbSVert.y());
    auto [aSc,aPair] = anOpt.ComputeSolInit(1.0,1e-8,10,0.05);
    auto [aRot,aV] = aPair;

            mBlock->SetVerticalCste(aV);
    StdOut()  << " SS=" << Ang4Show(aSc) <<  " VV=" << aV << " C1="<< aRot.AxeI() << " C2=" << aRot.AxeJ() << "\n";
}

// ========================  3D optimisation ==========================


tREAL8 cAppli_BlockInstrInitClino::ValueOf2OthogClino(const tRotR& aR) const
{
    tREAL8 aRes1 = ScoreDirClino(aR.AxeI(),mK1CurC);
    tREAL8 aRes2 = ScoreDirClino(aR.AxeJ(),mK2CurC);

    return (aRes1+aRes2)/2.0;
}

cPt1dr  cAppli_BlockInstrInitClino::Value(const cPt3dr& aWPK) const
{
    return cPt1dr(ValueOf2OthogClino(P3toRot(aWPK)));
}


tRotR  cAppli_BlockInstrInitClino::P3toRot(const cPt3dr& aWPK) const
{
    return mAxes2Orthog * tRotR::RotFromWPK(aWPK);
}

void cAppli_BlockInstrInitClino::Process2OrthogClino(const cPt2di & aK1K2)
{
    mK1CurC = aK1K2.x();
    mK2CurC = aK1K2.y();
    mInPairOrthog.at(mK1CurC) = true;
    mInPairOrthog.at(mK2CurC) = true;

    cPt3dr aP1 = mDirClinoIndep.at(mK1CurC);
    cPt3dr aP2 = mDirClinoIndep.at(mK2CurC);

    auto [aQ1,aQ2] = OrthogonalizePair(aP1,aP2);
    cPt3dr aQ3 = aQ1 ^ aQ2;
    mAxes2Orthog = tRotR(aQ1,aQ2,aQ3,false);

    //tREAL8 anAng = std::abs(AbsAngleTrnk(aP1,aP2)-M_PI/2.0) ;

    cOptimByStep<3> anOptim(*this,true,1.0);
    auto [aScMin,aPt3Min] = anOptim.Optim(cPt3dr(0,0,0),2.0/mNbSamp1Cl,1e-8,1/sqrt(2.0));

    mAxes2Orthog = P3toRot(aPt3Min);

     mScoreClinoGlob.at(mK1CurC) =    mScoreClinoGlob.at(mK2CurC) = aScMin;
     mDirClinoGlob.at(mK1CurC) = mAxes2Orthog.AxeI();
     mDirClinoGlob.at(mK2CurC) = mAxes2Orthog.AxeJ();
}

            // ========================  2D optimisation ==========================



tREAL8 cAppli_BlockInstrInitClino::ScoreDirClino(const cPt3dr& aDir,size_t aKClino) const
{
        return mBlock->ScoreDirClino(aDir,aKClino);
}

cPt3dr  cAppli_BlockInstrInitClino::P2toP3(const cPt2dr& aP2) const
{
        return VUnit(mAxes1Clino.Value(cPt3dr(1.0,aP2.x(),aP2.y())));
}

cPt1dr  cAppli_BlockInstrInitClino::Value(const cPt2dr&aP2) const
{
   return cPt1dr( ScoreDirClino(P2toP3(aP2),mKCurClino));
}


void cAppli_BlockInstrInitClino::Process1ClinoIndep(int aK)
{
    mKCurClino = aK;
    cSampleSphere3D aSSph(mNbSamp1Cl);

    //==== [1]  Compute an initial solution by parsing the sphere discretization =============
    cWhichMin<cPt3dr,tREAL8> aWMin(cPt3dr(0,0,1),1e10);

    for (int aKPt=0 ; aKPt<aSSph.NbSamples(); aKPt++)
    {
        cPt3dr aPt =  aSSph.KthPt(aKPt);
        aWMin.Add(aPt,ScoreDirClino(aPt,mKCurClino));
    }

    //======[2]  refine the solution ============================

    //   Let I be the initial solution, and J,K complentary base, we will store the solution (a,b) as
    // I + aJ + b K ; J and K are computed as orthogonal base
    mAxes1Clino = tRotR::CompleteRON(aWMin.IndexExtre());


    tREAL8 aScMin = 1e10;
    for (auto aStep : {0.8,0.6,0.4,0.3})
    {
       cOptimByStep<2> anOptim(*this,true,1.0,2);
       auto [aScLocMin,aPt2Min] = anOptim.Optim(cPt2dr(0,0),2.0/mNbSamp1Cl,1e-6,aStep);
       aScMin = aScLocMin;
       mAxes1Clino = tRotR::CompleteRON(P2toP3(aPt2Min));
    }

    mAvgClinIndep.Add(1.0,aScMin);
    mScoreClinoIndep.at(mKCurClino) = aScMin;
    mDirClinoIndep.at(mKCurClino) = mAxes1Clino.AxeI();
}




int cAppli_BlockInstrInitClino::Exe()
{
    mPhProj.FinishInit();

    // read an existing bloc from std folder
    mBlock = new cIrbComp_Block(mPhProj,mNameBloc);
    mCalBlock  = & mBlock->CalBlock();
    mSetClinos = & mCalBlock->SetClinos();
    const_cast<int&>(mNbClino) = mSetClinos->NbClino() ;

    mScoreClinoIndep.resize(mNbClino);
    mDirClinoIndep.resize(mNbClino);
    mInPairOrthog.resize(mNbClino,false);

    if (IsInit(&mNumPoseInstr))
    {
       mCalBlock->SetCams().SetNumPoseInstr(mNumPoseInstr);
    }


    //  add all the camera
    for (const auto & aNameIm :  VectMainSet(0))
    {
       mBlock->AddImagePose(aNameIm);
    }

    mBlock->SetVerticalCste(cPt3dr(0,0,1));
  //  StdOut() << "DEFFFF " <<  mBlock->OriSysCo().Def() << "\n";
            // << " "<<   mBlock->OriSysCo().<< "\n";


    mBlock->ComputePoseInstrument();
    mBlock->SetClinoValues();

    if (!IsInit(&mPairOrthog))
    {
        for (const auto & [aPair,aCstr] : mCalBlock->CstrOrthog())
        {
            int aK1 = mSetClinos->IndexClinoFromName(aPair.V1(),true);
            int aK2 = mSetClinos->IndexClinoFromName(aPair.V2(),true);

            if ((aK1>=0) && (aK2>=0))
                mPairOrthog.push_back(cPt2di(aK1,aK2));
        }
    }

    if (IsInit(&mVertApriori))
    {
        for (const auto aPt: mPairOrthog)
        {
           InitVerticalAnd2OthogClino(aPt);
        }
    }


    for (int aKC=0 ; aKC<(int)mSetClinos->NbClino() ; aKC++)
        Process1ClinoIndep(aKC);

    mScoreClinoGlob = mScoreClinoIndep;
    mDirClinoGlob   = mDirClinoIndep;

   // mPhProj.SaveRigBoI(mBlock->CalBlock());

    for (const auto aPt: mPairOrthog)
    {
        Process2OrthogClino(aPt);
    }

    StdOut() <<  " AVG ;  INDEP=" <<  Ang4Show(mAvgClinIndep.Average()) ;
    if (! mPairOrthog.empty())
       StdOut() <<  " ; ORTHOG=" << Ang4Show(AvgElem(mScoreClinoGlob)) ;
    StdOut() << "\n\n";


    // ========================= print the result  (Reports 2 add) ===========================================
    for (int aKC=0 ; aKC<mNbClino ; aKC++)
    {
        tREAL8 aScore = mScoreClinoGlob.at(aKC);
        std::string aName = mSetClinos->VNames().at(aKC);
        if (mReSetSigma)
        {
            auto & aDescInstr = mCalBlock->AddSigma_Indiv(aName,eTyInstr::eClino);
            aDescInstr.SetSigma(cIrb_SigmaInstr(0.0,1.0,0.0,aScore));
        }
        mSetClinos->KthClino(aKC).SetPNorm(mDirClinoGlob.at(aKC));

        StdOut() << " K=" << aKC << " N=" << aName
                 << " Score=" << Ang4Show(aScore)
                 << " Dir=" <<  mDirClinoGlob.at(aKC)
                 << "\n";
    }

    if (! mPairOrthog.empty())
    {
         StdOut() <<  "   =============== Orthogonality ===================\n";
         for (const auto & aPair : mPairOrthog)
         {
             int aK1 = aPair.x();
             int aK2 = aPair.y();
             tREAL8 aDelta1 =  mScoreClinoGlob.at(aK1) -  mScoreClinoIndep.at(aK1);
             tREAL8 aDelta2 =  mScoreClinoGlob.at(aK2) -  mScoreClinoIndep.at(aK2);

             tREAL8 anAng = std::abs(AbsAngleTrnk( mDirClinoIndep.at(aK1),mDirClinoIndep.at(aK2))-M_PI/2.0) ;
             StdOut() << " * Pair=" << aPair
                      << " OrthoAPrior=" <<  Ang4Show(anAng)
                      << " Dif:Indep/Orthog=" << Ang4Show(aDelta1) << " " << Ang4Show(aDelta2) << "\n";
         }
    }

    mPhProj.SaveRigBoI(*mCalBlock);


    delete mBlock;
    return EXIT_SUCCESS;
}

    /* ==================================================== */
    /*                                                      */
    /*               MMVII                                  */
    /*                                                      */
    /* ==================================================== */


tMMVII_UnikPApli Alloc_BlockInstrInitClino(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_BlockInstrInitClino(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_BlockInstrInitClino
(
     "BlockInstrInitClino",
      Alloc_BlockInstrInitClino,
      "Init  camera poses inside a block of instrument",
      {eApF::BlockInstr,eApF::Ori,eApF::Clino},
      {eApDT::BlockInstr,eApDT::Ori,eApDT::Clino},
      {eApDT::BlockInstr},
      __FILE__
);


};

