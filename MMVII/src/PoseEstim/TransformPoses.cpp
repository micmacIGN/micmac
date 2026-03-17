#include "cMMVII_Appli.h"
#include "MMVII_Sensor.h"
#include "MMVII_PCSens.h"
#include "MMVII_Geom3D.h"

namespace MMVII
{

typedef cIsometry3D<tREAL8> tPoseR;

/* ********************************************************** */
/*                                                            */
/*                     cAppli_TransformPoses                  */
/*                                                            */
/* ********************************************************** */

class cAppli_TransformPoses : public cMMVII_Appli
{
    public:
        cAppli_TransformPoses(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);

        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;

    private:
        cPhotogrammetricProject mPhProj;
        std::string             mPatImFrame1;
        std::string             mDirOriFrame2;
        std::string             mPatImFrame2;
        std::string             mDirOut;
        bool                    mCSVReport;
};

cAppli_TransformPoses::cAppli_TransformPoses(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
    cMMVII_Appli (aVArgs,aSpec),
    mPhProj      (*this)
{
}

cCollecSpecArg2007 & cAppli_TransformPoses::ArgObl(cCollecSpecArg2007 & anArgObl)
{
    return anArgObl
           << Arg2007(mPatImFrame1,"Pattern/file of images",{{eTA2007::MPatFile,"0"},{eTA2007::FileDirProj}})
           << mPhProj.DPOrient().ArgDirInMand("Orientation of images in frame 1")
           << Arg2007(mDirOriFrame2,"Orientation of images in frame 2",{{eTA2007::Orient}})
        ;
}

cCollecSpecArg2007 & cAppli_TransformPoses::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
    return    anArgOpt
           << mPhProj.DPOrient().ArgDirOutOpt("","Name of output orientation folder")
           << AOpt2007(mPatImFrame2,"PatImFrame2","Pattern of images in frame 2",
                       {{eTA2007::MPatFile,"1"},{eTA2007::HDV}})
           << AOpt2007(mCSVReport,"CSV","Save results to CSV file",{eTA2007::HDV})
        ;
}

int cAppli_TransformPoses::Exe()
{
    mPhProj.FinishInit();

    mDirOut = mDirOriFrame2 + "_in_" + mPhProj.DPOrient().DirIn();

    if (! mPhProj.DPOrient().DirOutIsInit())
        mPhProj.DPOrient().SetDirOut(mDirOut);
    if (! IsInit(&mPatImFrame2))
        mPatImFrame2 = ".*";

    std::string aReportErrors="ErrorOnPoses_" + mPhProj.DPOrient().DirIn()
                                + "_" + mDirOriFrame2
                                + "_Im" + ToStr(VectMainSet(0).size());
    if (mCSVReport)
        InitReportCSV(aReportErrors,"csv",true,{"Image","ErrorTr","ErrorRot"});

    std::vector<tPoseR> aVPoseFrame1;
    std::vector<tPoseR> aVPoseFrame2;

    tNameSelector  aSel = AllocRegex(mPatImFrame2);



    // get corresponding images in two sets of orientation
    for (auto aImName : VectMainSet(0))
    {
        if (aSel.Match(aImName))
        {
            StdOut() << aImName << std::endl;

            // image in frame1
            cSensorCamPC * aCamCur = mPhProj.ReadCamPC(aImName,true);
            //aVPoseFrame1.push_back(aCamCur->Pose());

            // corresponding image in frame2
            cSensorImage* aSI = mPhProj.ReadSensorFromFolder(mDirOriFrame2,aImName,true);
            //aVPoseFrame2.push_back( aSI->GetSensorCamPC()->Pose() );

            // if sensor exists
            if (aSI!=nullptr)
            {
                aVPoseFrame1.push_back(aCamCur->Pose());
                aVPoseFrame2.push_back( aSI->GetSensorCamPC()->Pose() );
            }
        }
    }

    //  transformation from frame1 to frame2
    auto [aRes,aSim] = EstimateSimTransfertFromPoses(aVPoseFrame2,aVPoseFrame1);

    // transform images in frame1 to frame2
    double aErrTotal=0;
    double aErrRot=0;
    double aErrTr=0;
    int aK=0;
    for (auto aImName : VectMainSet(0))
    {
        cSensorCamPC * aCamInFrame1 = mPhProj.ReadCamPC(aImName,true);
        tPoseR aPoseInFrame2 = TransfoPose(aSim,aCamInFrame1->Pose());

        // stats over corresponding images
        if (aSel.Match(aImName))
        {
            double aErrTrCur  = Norm2(aVPoseFrame2[aK].Tr() - aPoseInFrame2.Tr());
            double aErrRotCut = aVPoseFrame2[aK].Rot().Dist(aPoseInFrame2.Rot());

            aErrTotal += aVPoseFrame2[aK].DistPose(aPoseInFrame2,1.0);
            aErrRot   += aErrRotCut;
            aErrTr    += aErrTrCur;

            aK++;

            if (mCSVReport)
                AddOneReportCSV(aReportErrors,{aImName,ToStr(aErrTrCur),ToStr(aErrRotCut)});

        }

        // save
        cSensorCamPC * aCamInFrame2 = new cSensorCamPC(aImName,aPoseInFrame2,aCamInFrame1->InternalCalib());

        if (mPhProj.DPOrient().DirOutIsInit())
            mPhProj.SaveCamPC(*aCamInFrame2);

        delete aCamInFrame2;
    }

    StdOut() << "==== Mean alignment errors ====\n"
             << "#images evaluated:\t" << aK << "\n"
             << "total:\t" << aErrTotal/aK << "\n"
             << "tr:\t" << aErrTr/aK << "\n"
             << "rot:\t" << aErrRot/aK << std::endl;


    return EXIT_SUCCESS;
}

tMMVII_UnikPApli Alloc_TransformPoses(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
    return tMMVII_UnikPApli(new cAppli_TransformPoses(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_TransformPoses
    (
        "___TransformPoses",
        Alloc_TransformPoses,
        "Align one set of poses with another",
        {eApF::Ori},
        {eApDT::Ori},
        {eApDT::Orient},
        __FILE__
        );

} // namespace MMVII
