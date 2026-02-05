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


/* *************************************************************** */
/*                                                                 */
/*               cAppli_BlockInstrInitClino                          */
/*                                                                 */
/* *************************************************************** */

class cAppli_BlockInstrInitClino : public cMMVII_Appli,
                                   public cDataMapping<tREAL8,2,1>,
                                   public cDataMapping<tREAL8,3,1>
{
     public :

        cAppli_BlockInstrInitClino(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli &);
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override;
        int Exe() override;
        // std::vector<std::string>  Samples() const ;
     private :
        tREAL8 ScoreDirClino(const cPt3dr& aDir,size_t aKClino) const;
        tREAL8 Ang4Show(tREAL8) const;

        // force a pair of clino to be orthog
        void OrthogPair(const cPt2di &);
        void DoOneClino(int aK);

        cPt1dr  Value(const cPt2dr&) const override ;

        tREAL8 ValueOfRot(const tRotR& aR) const;
        cPt1dr  Value(const cPt3dr&) const override ;


        cPt3dr  P2toP3(const cPt2dr&) const;
        tRotR   P3toRot(const cPt3dr&) const;

        cPhotogrammetricProject   mPhProj;
        std::string               mSpecImIn;

        // ------------- Optionnal parameters ----------------
        std::string               mNameBloc;  //< name of the bloc inside the
        int                       mNbSS;
        std::vector<cPt2di>       mPairOrthog;  //< vector of orthog if we change (more for test)
        eTyUnitAngle              mUnitShow;    //< Unity for print result
        bool                      mReSetSigma;    //< do we use sigma

        //  ----------
        cIrbComp_Block *          mBlock;
        cIrbCal_Block *           mCalBlock;
        cIrbCal_ClinoSet *        mSetClinos ;          //< Accessors
        int                       mNbClino;



        //bool                      mAvgSigma;  //< Do we average sigma of pairs
        int                       mKCurClino;
        int                       mK1CurC;
        int                       mK2CurC;
        cPt2di                    mK1K2CurC;
        tRotR                     mAxesCur;
        std::vector<int>          mNumPoseInstr;  //< Num cams used for pose estimation
        std::vector<bool>         mInPairOrthog;  //< Is the clino used in a pair

        cWeightAv<tREAL8,tREAL8>  mAvgClinIndep;
        std::vector<tREAL8>       mScoreClinoIndep;
        std::vector<cPt3dr>       mDirClinoIndep;

        cWeightAv<tREAL8,tREAL8>  mAvgClinGlob;
        std::vector<tREAL8>       mScoreClinoGlob;
        std::vector<cPt3dr>       mDirClinoGlob;


};



cAppli_BlockInstrInitClino::cAppli_BlockInstrInitClino(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec) :
    cMMVII_Appli (aVArgs,aSpec),
    mPhProj      (*this),

    mNameBloc    (cIrbCal_Block::theDefaultName),
    mNbSS        (100),
    mUnitShow    (eTyUnitAngle::eUA_DMgon),
    mReSetSigma  (true),
    mBlock       (nullptr),
    mCalBlock    (nullptr),
    mSetClinos   (nullptr),
    mKCurClino   (-1)
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
            << AOpt2007(mNbSS,"NbSS","Number of sample on the sphere ",{{eTA2007::HDV}})
            << AOpt2007(mNumPoseInstr,"NPI","Num of cams used  for estimate pose of intsrument")
            << AOpt2007(mPairOrthog,"PairOrthog","Pair of orthogonal camera (if reset)")
            << AOpt2007(mUnitShow,"USA","Unity Show Angles",{AC_ListVal<eTyUnitAngle>(),{eTA2007::HDV}})
            << AOpt2007(mReSetSigma,"ResetSigma","Do we use sigma",{{eTA2007::HDV}})

    ;
}

tREAL8 cAppli_BlockInstrInitClino::ScoreDirClino(const cPt3dr& aDir,size_t aKClino) const
{
    return mBlock->ScoreDirClino(aDir,aKClino);
}

tREAL8 cAppli_BlockInstrInitClino::Ang4Show(tREAL8 anAng) const
{
   return AngleFromRad(anAng,mUnitShow);
}
  // ========================  3D optimisation ==========================

/*
tREAL8 cAppli_BlockInstrInitClino::ValueOfRot(const tRotR& aR) const
{

}
*/

cPt1dr  cAppli_BlockInstrInitClino::Value(const cPt3dr& aWPK) const
{
    tRotR aR = P3toRot(aWPK);
    tREAL8 aRes1 = ScoreDirClino(aR.AxeI(),mK1K2CurC.x());
    tREAL8 aRes2 = ScoreDirClino(aR.AxeJ(),mK1K2CurC.y());

    return cPt1dr((aRes1+aRes2)/2.0);
}




tRotR  cAppli_BlockInstrInitClino::P3toRot(const cPt3dr& aWPK) const
{
    return mAxesCur * tRotR::RotFromWPK(aWPK);
}

void cAppli_BlockInstrInitClino::OrthogPair(const cPt2di & aK1K2)
{
    mK1K2CurC = aK1K2;
    mK1CurC = mK1K2CurC.x();
    mK2CurC = mK1K2CurC.y();
    mInPairOrthog.at(mK1CurC) = true;
    mInPairOrthog.at(mK2CurC) = true;

    cPt3dr aP1 = mDirClinoIndep.at(mK1CurC);
    cPt3dr aP2 = mDirClinoIndep.at(mK2CurC);
    //tREAL8 aSc1 = mScoreClinoIndep.at(mK1CurC);
    //tREAL8 aSc2 = mScoreClinoIndep.at(mK2CurC);


    auto [aQ1,aQ2] = OrthogonalizePair(aP1,aP2);
    cPt3dr aQ3 = aQ1 ^ aQ2;
    mAxesCur = tRotR(aQ1,aQ2,aQ3,false);

    //tREAL8 anAng = std::abs(AbsAngleTrnk(aP1,aP2)-M_PI/2.0) ;

    cOptimByStep<3> anOptim(*this,true,1.0);
    auto [aScMin,aPt3Min] = anOptim.Optim(cPt3dr(0,0,0),2.0/mNbSS,1e-8,1/sqrt(2.0));

    mAxesCur = P3toRot(aPt3Min);

     mScoreClinoGlob.at(mK1CurC) =    mScoreClinoGlob.at(mK2CurC) = aScMin;
     mDirClinoGlob.at(mK1CurC) = mAxesCur.AxeI();
     mDirClinoGlob.at(mK2CurC) = mAxesCur.AxeJ();
}

            // ========================  2D optimisation ==========================

cPt3dr  cAppli_BlockInstrInitClino::P2toP3(const cPt2dr& aP2) const
{
    return VUnit(mAxesCur.Value(cPt3dr(1.0,aP2.x(),aP2.y())));
}


cPt1dr  cAppli_BlockInstrInitClino::Value(const cPt2dr&aP2) const
{
   return cPt1dr( ScoreDirClino(P2toP3(aP2),mKCurClino));
}


void cAppli_BlockInstrInitClino::DoOneClino(int aK)
{
    mKCurClino = aK;
    cSampleSphere3D aSSph(mNbSS);

    cWhichMin<cPt3dr,tREAL8> aWMin(cPt3dr(0,0,1),1e10);

  //  StdOut()  << "NBSSSS=" << aSSph.NbSamples() << "\n";
    for (int aKPt=0 ; aKPt<aSSph.NbSamples(); aKPt++)
    {
        cPt3dr aPt =  aSSph.KthPt(aKPt);
        aWMin.Add(aPt,ScoreDirClino(aPt,mKCurClino));
    }
    mAxesCur = tRotR::CompleteRON(aWMin.IndexExtre());

    tREAL8 aSc1 = aWMin.ValExtre();

    cOptimByStep<2> anOptim(*this,true,1.0,2);
    auto [aScMin,aPt2Min] = anOptim.Optim(cPt2dr(0,0),2.0/mNbSS,1e-7,0.7);
    mAxesCur = tRotR::CompleteRON(P2toP3(aPt2Min));

    tREAL8 aSc2 = aScMin;

    if (NeverHappens())
    {
        StdOut() << " K=" << mKCurClino
                 << " V=" << Ang4Show(aWMin.ValExtre())
                 << " VV=" << Ang4Show(aScMin)
                 << " P=" << aWMin.IndexExtre() << " "<< mAxesCur.AxeI() << "\n";
    }

    for (int aK=0 ; aK<100 ; aK++)
    {
        int aDivStep= 4;
        tREAL8 aStep = aScMin * RandInInterval(1.0-1.0/aDivStep,1.0) /aDivStep;
        cPt2dr anOffset = cPt2dr::PRandC() * aStep;
        int aMulStep = 3;
        int aNbStep = aDivStep * aMulStep;
        cRect2 aBox = cRect2::BoxWindow(aNbStep);

        cWhichMin<cPt2dr,tREAL8> aMinNeigh(cPt2dr(0,0),Value(cPt2dr(0,0)).x());
        for (const auto aPix : aBox)
        {
             cPt2dr aDelta = anOffset + ToR(aPix) * aStep;
             aMinNeigh.Add(aDelta,Value(aDelta).x());
        }
        if (IsNotNull(aMinNeigh.IndexExtre()))
        {
             aScMin = aMinNeigh.ValExtre();
             mAxesCur = tRotR::CompleteRON(P2toP3(aMinNeigh.IndexExtre()));
            // StdOut()  << "SCCCC= " << Ang4Show(aScMin) << "\n";
        }
    }

    tREAL8 aSc3 = aScMin;
    if (NeverHappens())
    {
        StdOut() << " SCORE CLINO "
                 << " A0=" <<   Ang4Show(aSc1)
                 << " Opt1=" << Ang4Show(aSc2)
                 << " Opt2=" << Ang4Show(aSc3)
              << "\n";
    }

    mAvgClinIndep.Add(1.0,aSc3);
    mScoreClinoIndep.at(mKCurClino) = aSc3;
    mDirClinoIndep.at(mKCurClino) = mAxesCur.AxeI();
}




int cAppli_BlockInstrInitClino::Exe()
{
    mPhProj.FinishInit();

    // read an existing bloc from std folder
    mBlock = new cIrbComp_Block(mPhProj,mNameBloc);
    mCalBlock  = & mBlock->CalBlock();
    mSetClinos = & mCalBlock->SetClinos();
    mNbClino = mSetClinos->NbClino() ;

    mScoreClinoIndep.resize(mNbClino);
    mDirClinoIndep.resize(mNbClino);
    mInPairOrthog.resize(mNbClino,false);

    if (IsInit(&mNumPoseInstr))
       mCalBlock->SetCams().SetNumPoseInstr(mNumPoseInstr);


    //  add all the camera
    for (const auto & aNameIm :  VectMainSet(0))
    {
       mBlock->AddImagePose(aNameIm);
    }

    mBlock->ComputePoseInstrument();
    mBlock->SetClinoValues();

    for (int aKC=0 ; aKC<(int)mSetClinos->NbClino() ; aKC++)
        DoOneClino(aKC);

    mScoreClinoGlob = mScoreClinoIndep;
    mDirClinoGlob   = mDirClinoIndep;

    mPhProj.SaveRigBoI(mBlock->CalBlock());

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

    for (const auto aPt: mPairOrthog)
    {
        OrthogPair(aPt);
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

