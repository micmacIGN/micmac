#include "MMVII_BundleAdj.h"
#include "../Graphs/ArboTriplets.h"

/**
   \file HierarchSfm.cpp

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
        //int                       mNbMaxClust;
        //tREAL8                    mDistClust;
        //std::vector<tREAL8>       mLevelRand;
        std::vector<tREAL8>       mWeigthEdge3;
        bool                      mDoCheck;
        tREAL8                    mWBalance;
        //bool                      mPerfectData;

};

cAppli_HierarchSfm::cAppli_HierarchSfm(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
    cMMVII_Appli (aVArgs,aSpec),
    mPhProj      (*this),
    mDoCheck     (true),
    mWBalance    (1.0)
{}

cCollecSpecArg2007 & cAppli_HierarchSfm::ArgObl(cCollecSpecArg2007 & anArgObl)
{
    return anArgObl
           <<  mPhProj.DPOriTriplets().ArgDirInMand("Input relative motions")
        ;
}

cCollecSpecArg2007 & cAppli_HierarchSfm::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
    return    anArgOpt
           <<  mPhProj.DPOrient().ArgDirInOpt("","Ground truth input orientation directory")
           <<  mPhProj.DPOrient().ArgDirOutOpt("","Global orientation output directory")
           <<  mPhProj.DPOriTriplets().ArgDirOutOpt("","Directory for dmp-save of triplet (for faster read later)")
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

    cMakeArboTriplet  aMk3(*a3Set,mDoCheck,mWBalance,*this);
    /*if (IsInit(&mLevelRand))
        aMk3.SetRand(mLevelRand);
    if (IsInit(&mWeigthEdge3))
        aMk3.WeigthEdge3() = mWeigthEdge3;
    if (IsInit(&mPerfectData))
        aMk3.PerfectData() = true;
    if (mPhProj.IsOriInDirInit())
    {
        aMk3.PerfectOri() = true;
    }*/

    TimeSegm().SetIndex("MakeGraphPose");
    aMk3.MakeGraphPose(mPhProj);

    TimeSegm().SetIndex("PoseRef");
    aMk3.DoPoseRef();

    TimeSegm().SetIndex("MakeCnxTriplet");
    aMk3.MakeCnxTriplet();

    TimeSegm().SetIndex("TripletWeighting");
    aMk3.MakeWeightingGraphTriplet();

    TimeSegm().SetIndex("ComputeArbor");
    aMk3.ComputeArbor();

    if (mPhProj.DPOrient().DirOutIsInit())
    {
        StdOut() << " ========== Output Global Orientation  ========== " << std::endl;
        aMk3.SaveGlobSol(mPhProj);
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
