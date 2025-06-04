#include "MMVII_BundleAdj.h"
#include "../Graphs/ArboTriplets.h"

/**
   \file HierarchSfm.cpp

   \brief Hierarchical Structure-from-Motion Using Triplet Graphs
          (tree hierarchy from the graph, then BA along the tree)

*/

namespace MMVII
{


/* ********************************************************** */
/*                                                            */
/*                          cMiniBA                           */
/*                                                            */
/* ********************************************************** */

class cMiniBA
{
    public:
        cMiniBA();
        ~cMiniBA();

    private:

};

/* ********************************************************** */
/*                                                            */
/*                     cAppli_HierarchSfm                     */
/*                                                            */
/* ********************************************************** */

class cAppli_HierarchSfm : public cMMVII_Appli
{
    public:
        cAppli_HierarchSfm(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);


        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;

    private:
        cPhotogrammetricProject   mPhProj;
        std::string               mPatImIn;
        //int                       mNbMaxClust;
        //tREAL8                    mDistClust;
        //std::vector<tREAL8>       mLevelRand;
        std::vector<tREAL8>       mWeigthEdge3;
        bool                      mDoCheck;
        tREAL8                    mWBalance;
        std::vector<tREAL8>       mViscPose;      ///< regularization on poses in BA
        tREAL8                    mSigmaTPt;
        tREAL8                    mFacElim;
        int                       mNbIterBA;

};

cAppli_HierarchSfm::cAppli_HierarchSfm(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
    cMMVII_Appli (aVArgs,aSpec),
    mPhProj      (*this),
    mDoCheck     (true),
    mWBalance    (1.0),
    mViscPose    ({-1,-1}),
    mSigmaTPt    (1.0),
    mFacElim     (10.0),
    mNbIterBA    (2)
{}

cCollecSpecArg2007 & cAppli_HierarchSfm::ArgObl(cCollecSpecArg2007 & anArgObl)
{
    return anArgObl
           << Arg2007(mPatImIn,"Pattern/file for images",{{eTA2007::MPatFile,"0"},{eTA2007::FileDirProj}})
           <<  mPhProj.DPOrient().ArgDirInMand("Input calibration folder")
           <<  mPhProj.DPOriTriplets().ArgDirInMand("Input relative motions")
        ;
}

cCollecSpecArg2007 & cAppli_HierarchSfm::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
    return    anArgOpt
           <<  mPhProj.DPMulTieP().ArgDirInOpt("","Input features")
           <<  mPhProj.DPOrient().ArgDirInOpt("","Ground truth input orientation directory")
           <<  mPhProj.DPOrient().ArgDirOutOpt("","Global orientation output directory")
           <<  mPhProj.DPOriTriplets().ArgDirOutOpt("","Directory for dmp-save of triplet (for faster read later)")
           <<  AOpt2007(mViscPose,"ViscPose","Regularization on poses for BA: [SigmaTr,SigmaRot]",{eTA2007::HDV})
           <<  AOpt2007(mSigmaTPt,"SigmaTPt","Sigma for tie-points",{eTA2007::HDV})
           <<  AOpt2007(mFacElim,"FacElim","Outlier threshold=(FacElim*SigmaTPt)",{eTA2007::HDV})
           <<  AOpt2007(mNbIterBA,"NbIterBA","Number of iteration in BA refinement",{eTA2007::HDV})
        ;
}

int cAppli_HierarchSfm::Exe()
{
    mPhProj.FinishInit();

    cAutoTimerSegm  aATS(TimeSegm(),"Read motions");
    cTripletSet *  a3Set =  mPhProj.ReadTriplets();

    if (mPhProj.DPOriTriplets().DirOutIsInit())
    {
        mPhProj.SaveTriplets(*a3Set,false);
        delete a3Set;
        return EXIT_SUCCESS;
    }


    TimeSegm().SetIndex("cMakeArboTriplet");
    cMakeArboTriplet  aMk3(*a3Set,mDoCheck,mWBalance,mPhProj,*this);
    if (IsInit(&mViscPose))
        aMk3.ViscPose() = mViscPose;
    if (IsInit(&mSigmaTPt))
        aMk3.SigmaTPt() = mSigmaTPt;
    if (IsInit(&mFacElim))
        aMk3.FacElim()= mFacElim;
    if (IsInit(&mNbIterBA))
        aMk3.NbIterBA() = mNbIterBA;
    if (mPhProj.DPMulTieP().DirInIsInit())
    {
        aMk3.TPFolder() = mPhProj.DPMulTieP().DirIn() ;
        std::vector<std::string> aSetIm = VectMainSet(0);
        aMk3.InitTPtsStruct(mPhProj.DPMulTieP().DirIn(),aSetIm);
    }
    else
        MMVII_INTERNAL_ASSERT_always(mPhProj.DPMulTieP().DirInIsInit(),"Input features not initialised");


    TimeSegm().SetIndex("MakeGraphPose");
    aMk3.MakeGraphPose();
    StdOut() << "MakeGraphPose DONE" << std::endl;

    TimeSegm().SetIndex("InitialiseCalibs");
    aMk3.InitialiseCalibs();
    StdOut() << "InitialiseCalibs DONE" << std::endl;

    TimeSegm().SetIndex("PoseRef");
    aMk3.DoPoseRef();
    StdOut() << "DoPoseRef DONE" << std::endl;

    TimeSegm().SetIndex("MakeCnxTriplet");
    aMk3.MakeCnxTriplet();
    StdOut() << "MakeCnxTriplet DONE" << std::endl;

    TimeSegm().SetIndex("TripletWeighting");
    aMk3.MakeWeightingGraphTriplet();
    StdOut() << "MakeWeightingGraphTriplet DONE" << std::endl;

    TimeSegm().SetIndex("ComputeArbor");
    aMk3.ComputeArbor();
    StdOut() << "ComputeArbor DONE" << std::endl;

    if (mPhProj.DPOrient().DirOutIsInit())
    {
        StdOut() << " ========== Output Global Orientation  ========== " << std::endl;
        aMk3.SaveGlobSol();
    }

    aMk3.ShowStat();

    delete a3Set;

    return EXIT_SUCCESS;
}

tMMVII_UnikPApli Alloc_HierarchSfm(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
    return tMMVII_UnikPApli(new cAppli_HierarchSfm(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_HierarchSfm
    (
        "___HierarchSfm",
        Alloc_HierarchSfm,
        "Construct global orientation from a graph of relative motions",
        {eApF::Ori},
        {eApDT::Ori},
        {eApDT::Orient},
        __FILE__
        );

}; // namespace MMVII
