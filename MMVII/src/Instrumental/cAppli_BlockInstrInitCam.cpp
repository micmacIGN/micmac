#include "MMVII_InstrumentalBlock.h"

#include "cMMVII_Appli.h"
#include "MMVII_2Include_Serial_Tpl.h"




/**
  \file cInstrumentalBloc.cpp


  \brief This file contains the core implemantation of Block of rigid instrument
 
*/

namespace MMVII
{

/* *************************************************************** */
/*                                                                 */
/*               cAppli_BlockInstrInitCam                          */
/*                                                                 */
/* *************************************************************** */

class cAppli_BlockInstrInitCam : public cMMVII_Appli
{
     public :

        cAppli_BlockInstrInitCam(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli &);
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override;
        int Exe() override;
        // std::vector<std::string>  Samples() const ;

     private :
        cPhotogrammetricProject   mPhProj;
        std::string               mSpecImIn;
        cIrbComp_Block *           mBlock;
        std::string               mNameBloc;
};


cAppli_BlockInstrInitCam::cAppli_BlockInstrInitCam(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec) :
    cMMVII_Appli (aVArgs,aSpec),
    mPhProj      (*this),
    mBlock       (nullptr),
    mNameBloc    (cIrbCal_Block::theDefaultName)
{
}

cCollecSpecArg2007 & cAppli_BlockInstrInitCam::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
     return anArgObl
             <<  Arg2007(mSpecImIn,"Pattern/file for images", {{eTA2007::MPatFile,"0"},{eTA2007::FileDirProj}}  )
             <<  mPhProj.DPBlockInstr().ArgDirInMand()
             <<  mPhProj.DPOrient().ArgDirInMand()
             <<  mPhProj.DPBlockInstr().ArgDirOutMand()
     ;
}

cCollecSpecArg2007 & cAppli_BlockInstrInitCam::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
	return anArgOpt
            << AOpt2007(mNameBloc,"NameBloc","Name of bloc to calib ",{{eTA2007::HDV}})
        ;
}

int cAppli_BlockInstrInitCam::Exe()
{
    mPhProj.FinishInit();

    // read an existing bloc from std folder
    mBlock = new cIrbComp_Block(mPhProj,mNameBloc);
    size_t aNbCam = mBlock->NbCams();

    //  add all the camera
    for (const auto & aNameIm :  VectMainSet(0))
    {
       mBlock->AddImagePose(aNameIm);
    }

    //  initialize the sigma of each pair of cams & compute the score of each cam
    std::vector<tREAL8>  aVScoreCam(aNbCam,0.0); // Cumultated score
    for (size_t aKC1=0 ; aKC1<aNbCam ; aKC1++)
    {
        for (size_t aKC2=0 ; aKC2<mBlock->NbCams(); aKC2++)
        {
            if (aKC1!=aKC2)
            {
               const auto & aSg_Pose_Sigm = mBlock->ComputeCalibCamsInit(aKC1,aKC2); // Pose + sigma
               tREAL8 aSigG = std::get<tREAL8>(aSg_Pose_Sigm);
               aVScoreCam.at(aKC1) += aSigG;
               aVScoreCam.at(aKC2) += aSigG;
               mBlock->CalBlock().AddSigma
               (
                   mBlock->CalBlock().SetCams().KthCam(aKC1).NameCal(),
                   eTyInstr::eCamera,
                   mBlock->CalBlock().SetCams().KthCam(aKC2).NameCal(),
                   eTyInstr::eCamera,
                   std::get<cIrb_SigmaInstr>(aSg_Pose_Sigm)
               );
            }
        }
    }

    // compute num of master
    cWhichMin<size_t,tREAL8> aMinK;
    for (size_t aKC1 = 0 ; aKC1<aNbCam ; aKC1++)
        aMinK.Add(aKC1,aVScoreCam.at(aKC1));
    int aNumMaster = aMinK.IndexExtre();

    mBlock->CalBlock().SetCams().SetNumMaster(aNumMaster);
    StdOut() << " NUM-MASTER " << aNumMaster << "\n";

    //  initialise the relative pose in the ARBITRARY coord syst of master cam
    for (size_t aKC1=0 ; aKC1<aNbCam ; aKC1++)
    {
        const auto & aSg_Pose_Sigm = mBlock->ComputeCalibCamsInit(aNumMaster,aKC1);
        mBlock->CalBlock().SetCams().KthCam(aKC1).SetPose(std::get<tPoseR>(aSg_Pose_Sigm));
    }

    mPhProj.SaveRigBoI(mBlock->CalBlock());
    delete mBlock;

    return EXIT_SUCCESS;
}

    /* ==================================================== */
    /*                                                      */
    /*               MMVII                                  */
    /*                                                      */
    /* ==================================================== */


tMMVII_UnikPApli Alloc_BlockInstrInitCam(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_BlockInstrInitCam(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_BlockInstrInitCam
(
     "BlockInstrInitCam",
      Alloc_BlockInstrInitCam,
      "Init  camera poses inside a block of instrument",
      {eApF::BlockInstr,eApF::Ori},
      {eApDT::BlockInstr,eApDT::Ori},
      {eApDT::BlockInstr},
      __FILE__
);


};

