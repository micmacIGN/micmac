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
                                   public cDataMapping<tREAL8,2,1>
{
     public :

        cAppli_BlockInstrInitClino(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli &);
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override;
        int Exe() override;
        // std::vector<std::string>  Samples() const ;
     private :
        tREAL8 ScoreDirClino(const cPt3dr& aDir,size_t aKClino) const;

        cPt1dr  Value(const cPt2dr&) const override ;
        cPt3dr  P2toP3(const cPt2dr&) const;

        void DoOneClino();

        cPhotogrammetricProject   mPhProj;
        std::string               mSpecImIn;
        cIrbComp_Block *          mBlock;
        cIrbCal_Block *           mCalBlock;
        cIrbCal_ClinoSet *        mSetClinos ;          //< Accessors
        std::string               mNameBloc;  //< name of the bloc inside the
        //bool                      mAvgSigma;  //< Do we average sigma of pairs
        int                       mKCurClino;
        int                       mNbSS;
        tRotR                     mAxesCur;
        cPt3dr                    mU;
        cPt3dr                    mV;
        cPt3dr                    mW;

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
    mNbSS        (100)
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
//          << AOpt2007(mAvgSigma,"AvgSigma","Do we average the sigma init",{{eTA2007::HDV}})
        ;
}


tREAL8 cAppli_BlockInstrInitClino::ScoreDirClino(const cPt3dr& aDir,size_t aKClino) const
{
    return mBlock->ScoreDirClino(aDir,aKClino);
}


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

    StdOut()  << "NBSSSS=" << aSSph.NbSamples() << "\n";
    for (int aKPt=0 ; aKPt<aSSph.NbSamples(); aKPt++)
    {
        cPt3dr aPt =  aSSph.KthPt(aKPt);
        aWMin.Add(aPt,ScoreDirClino(aPt,mKCurClino));
    }
    mAxesCur = tRotR::CompleteRON(aWMin.IndexExtre());

    cOptimByStep<2> anOptim(*this,true,1.0,2);
    auto [aScMin,aPt2Min] = anOptim.Optim(cPt2dr(0,0),2.0/mNbSS,1e-7,0.7);
    mAxesCur = tRotR::CompleteRON(P2toP3(aPt2Min));

     StdOut() << " K=" << mKCurClino
              << " V=" << AngleFromRad(aWMin.ValExtre(),eTyUnitAngle::eUA_DMgon)
              << " VV=" << AngleFromRad(aScMin,eTyUnitAngle::eUA_DMgon)
              << " P=" << aWMin.IndexExtre() << " "<< mAxesCur.AxeI() << "\n";

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
             StdOut()  << "SCCCC= " << AngleFromRad(aScMin,eTyUnitAngle::eUA_DMgon) << "\n";
        }
    }



}

int cAppli_BlockInstrInitClino::Exe()
{
    mPhProj.FinishInit();

    // read an existing bloc from std folder
    mBlock = new cIrbComp_Block(mPhProj,mNameBloc);
    mCalBlock  = & mBlock->CalBlock();
    mSetClinos = & mCalBlock->SetClinos();


    //  add all the camera
    for (const auto & aNameIm :  VectMainSet(0))
    {
// StdOut() << "IiIImmm=" << aNameIm << "\n";
       mBlock->AddImagePose(aNameIm);
    }

    mBlock->ComputePoseInstrument();
    mBlock->SetClinoValues();

    for (mKCurClino=0 ; mKCurClino<(int)mSetClinos->NbClino() ; mKCurClino++)
        DoOneClino();

    mPhProj.SaveRigBoI(mBlock->CalBlock());

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

