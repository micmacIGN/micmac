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
        void DoOneClino();

        cPt1dr  Value(const cPt2dr&) const override ;
        cPt1dr  Value(const cPt3dr&) const override ;

        cPt3dr  P2toP3(const cPt2dr&) const;
        tRotR   P3toRot(const cPt3dr&) const;

        cPhotogrammetricProject   mPhProj;
        std::string               mSpecImIn;
        cIrbComp_Block *          mBlock;
        cIrbCal_Block *           mCalBlock;
        cIrbCal_ClinoSet *        mSetClinos ;          //< Accessors
        std::string               mNameBloc;  //< name of the bloc inside the
        //bool                      mAvgSigma;  //< Do we average sigma of pairs
        int                       mKCurClino;
        cPt2di                    mK1K2CurC;
        int                       mNbSS;
        tRotR                     mAxesCur;
        std::vector<int>          mNumPoseInstr;
        cWeightAv<tREAL8,tREAL8>  mAvgClinIndep;
        std::vector<tREAL8>       mScoreClino;
        std::vector<cPt3dr>       mDirClinoIndep;
        std::vector<cPt2di>       mPairOrthog;
        eTyUnitAngle              mUnity;
};



cAppli_BlockInstrInitClino::cAppli_BlockInstrInitClino(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec) :
    cMMVII_Appli (aVArgs,aSpec),
    mPhProj      (*this),
    mBlock       (nullptr),
    mCalBlock    (nullptr),
    mSetClinos   (nullptr),
    mNameBloc    (cIrbCal_Block::theDefaultName),
  //  mAvgSigma    (true),
    mKCurClino   (-1),
    mNbSS        (100),
    mUnity       (eTyUnitAngle::eUA_DMgon)
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
            << AOpt2007(mPairOrthog,"PairOrthog","Num of cams used  for estimate pose of intsrument")
            << AOpt2007(mUnity,"USA","Unity Show Angles",{AC_ListVal<eTyUnitAngle>(),{eTA2007::HDV}})

        ;
}

tREAL8 cAppli_BlockInstrInitClino::ScoreDirClino(const cPt3dr& aDir,size_t aKClino) const
{
    return mBlock->ScoreDirClino(aDir,aKClino);
}

tREAL8 cAppli_BlockInstrInitClino::Ang4Show(tREAL8 anAng) const
{
   return AngleFromRad(anAng,mUnity);
}
  // ========================  3D optimisation ==========================

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
    cPt3dr aP1 = mDirClinoIndep.at(mK1K2CurC.x());
    cPt3dr aP2 = mDirClinoIndep.at(mK1K2CurC.y());
    tREAL8 aSc1 = mScoreClino.at(mK1K2CurC.x());
    tREAL8 aSc2 = mScoreClino.at(mK1K2CurC.y());


    auto [aQ1,aQ2] = OrthogonalizePair(aP1,aP2);
    cPt3dr aQ3 = aQ1 ^ aQ2;
    mAxesCur = tRotR(aQ1,aQ2,aQ3,false);

    tREAL8 anAng = std::abs(AbsAngleTrnk(aP1,aP2)-M_PI/2.0) ;
    StdOut() << " * ANGL-Orthog =" <<   Ang4Show(anAng)   << "\n";

    cOptimByStep<3> anOptim(*this,true,1.0);
    auto [aScMin,aPt3Min] = anOptim.Optim(cPt3dr(0,0,0),2.0/mNbSS,1e-8,1/sqrt(2.0));

    StdOut() << " SC-ORTHO=" << Ang4Show(aScMin)
             << " SC-INDIV=" << Ang4Show((aSc1+aSc2)/2.0)
            << "\n";
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


void cAppli_BlockInstrInitClino::DoOneClino()
{
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

    if (0)
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

     StdOut() << " SCORE CLINO "
              << " A0=" <<   Ang4Show(aSc1)
              << " Opt1=" << Ang4Show(aSc2)
              << " Opt2=" << Ang4Show(aSc3)
              << "\n";
      mAvgClinIndep.Add(1.0,aSc3);
      mScoreClino.push_back(aSc3);
      mDirClinoIndep.push_back(mAxesCur.AxeI());
}




int cAppli_BlockInstrInitClino::Exe()
{
    mPhProj.FinishInit();

    // read an existing bloc from std folder
    mBlock = new cIrbComp_Block(mPhProj,mNameBloc);
    mCalBlock  = & mBlock->CalBlock();
    mSetClinos = & mCalBlock->SetClinos();

    if (IsInit(&mNumPoseInstr))
       mCalBlock->SetCams().SetNumPoseInstr(mNumPoseInstr);


    //  add all the camera
    for (const auto & aNameIm :  VectMainSet(0))
    {
       mBlock->AddImagePose(aNameIm);
    }

    mBlock->ComputePoseInstrument();
    mBlock->SetClinoValues();

    for (mKCurClino=0 ; mKCurClino<(int)mSetClinos->NbClino() ; mKCurClino++)
        DoOneClino();

    mPhProj.SaveRigBoI(mBlock->CalBlock());

    if (!IsInit(&mPairOrthog))
    {
        for (const auto & [aPair,aCstr] : mCalBlock->CstrOrthog())
        {
            int aK1 = mSetClinos->IndexClinoFromName(aPair.V1());
            int aK2 = mSetClinos->IndexClinoFromName(aPair.V2());

            if ((aK1>=0) && (aK2>=0))
                mPairOrthog.push_back(cPt2di(aK1,aK2));

            StdOut()  <<  "KKKK "  << aK1 << " " << aK2 << "\n";
        }
    }

    for (const auto aPt: mPairOrthog)
    {
        OrthogPair(aPt);
    }

    StdOut() <<  " AVG CLINO INDEP=" <<  Ang4Show(mAvgClinIndep.Average()) << "\n";

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

