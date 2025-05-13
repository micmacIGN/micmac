#include "VisPoseAndStructure.h"

/**
   \file VisPoseAndStructure.cpp

   \brief Visualise a set of poses and the sparse 3D structure

*/

namespace MMVII
{




cAppli_VisuPoseStr3D::cAppli_VisuPoseStr3D(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
    cMMVII_Appli (aVArgs,aSpec),
    mPhProj      (*this),
    mErrProjMax  (20.0),
    mCamScale    (1.0),
    mOutfile     ("VisSFM_${ori}_${features}.ply"),
    mBinary      (true)
{
}

cCollecSpecArg2007 & cAppli_VisuPoseStr3D::ArgObl(cCollecSpecArg2007 & anArgObl)
{
    return anArgObl
           << Arg2007(mPatImIn,"Pattern/file for images",{{eTA2007::MPatFile,"0"},{eTA2007::FileDirProj}})
           << mPhProj.DPOrient().ArgDirInMand("Input orientation (Sfm)")
           << mPhProj.DPMulTieP().ArgDirInMand("Input features")
        ;
}

cCollecSpecArg2007 & cAppli_VisuPoseStr3D::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
    return    anArgOpt
           << AOpt2007(mErrProjMax,"ErrMax","Outlier threshold",{eTA2007::HDV})
           << AOpt2007(mCamScale,"CamScale","Scale camera frustum",{eTA2007::HDV})
           << AOpt2007(mOutfile,"Outfile","Output filename",{eTA2007::HDV})
           << AOpt2007(mBinary,"Bin","Output in binary format",{eTA2007::HDV})
        ;
}

int cAppli_VisuPoseStr3D::Exe()
{
    mPhProj.FinishInit();
    mOutfile = ("VisSFM_"+mPhProj.DPOrient().DirIn()+"_"+mPhProj.DPMulTieP().DirIn()+".ply");


    // vector of all image names belonging to this tree level
    std::vector<std::string> aVNames;
    // cameras
    //std::vector<cSensorCamPC *> aVCams ;
    std::vector<cSensorImage *> aVSens ;

    for (const auto & aIm : VectMainSet(0))
    {
        aVNames.push_back(aIm);
        aVSens.push_back(mPhProj.ReadSensor(aIm,true));

    }
    // sort images alphbetically (and aVSens accordingly) for AllocStdFromMTPFromFolder
    Sort2VectFirstOne(aVNames,aVSens);

   /*  for (size_t aK=0; aK<aVCams.size(); aK++)
        aVSens.push_back(aVCams.at(aK));


*/

    // read the tie points corresponding to your image set
    cComputeMergeMulTieP * aTPts = AllocStdFromMTPFromFolder(
                mPhProj.DPMulTieP().DirIn(),aVNames,mPhProj,true,false,true);

    // intersect in 3d
    for (auto & aPair : aTPts->Pts())
        MakePGround(aPair,aVSens);


    WritePly(aTPts,aVSens);

    delete aTPts;

    return EXIT_SUCCESS;
}

tMMVII_UnikPApli Alloc_VisuPoseStr3D(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
    return tMMVII_UnikPApli(new cAppli_VisuPoseStr3D(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_VisuPoseStr3D
    (
        "___VisSfm",
        Alloc_VisuPoseStr3D,
        "Create PLY with poses and 3D structure",
        {eApF::Ori},
        {eApDT::Ori},
        {eApDT::Orient},
        __FILE__
        );

}; // namespace MMVII
