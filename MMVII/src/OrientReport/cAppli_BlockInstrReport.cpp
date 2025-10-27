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
/*               cAppli_BlockInstrReport                          */
/*                                                                 */
/* *************************************************************** */

class  cAppli_BlockInstrReport : public cMMVII_Appli
{
     public :

        cAppli_BlockInstrReport(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli &);
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override;
        int Exe() override;
        // std::vector<std::string>  Samples() const ;

     private :
        cPhotogrammetricProject   mPhProj;
        std::string               mSpecImIn;
        cIrbComp_Block *          mBlock;
        const cIrbCal_Block*      mCalBlock;
        const cIrbCal_CamSet*     mCamCal;
        int                       mNbCams;
        std::string               mNameBloc;

        void DoReportOrient();
};


cAppli_BlockInstrReport::cAppli_BlockInstrReport(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec) :
    cMMVII_Appli (aVArgs,aSpec),
    mPhProj      (*this),
    mBlock       (nullptr),
    mCalBlock    (nullptr),
    mCamCal      (nullptr),
    mNameBloc    (cIrbCal_Block::theDefaultName)
{
}


cCollecSpecArg2007 & cAppli_BlockInstrReport::ArgObl(cCollecSpecArg2007 & anArgObl)
{
     return anArgObl
             <<  Arg2007(mSpecImIn,"Pattern/file for images", {{eTA2007::MPatFile,"0"},{eTA2007::FileDirProj}}  )
             <<  mPhProj.DPBlockInstr().ArgDirInMand()
             <<  mPhProj.DPOrient().ArgDirInMand()
     ;
}

cCollecSpecArg2007 & cAppli_BlockInstrReport::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
	return anArgOpt
            << AOpt2007(mNameBloc,"NameBloc","Name of bloc to calib ",{{eTA2007::HDV}})
        ;
}

void cAppli_BlockInstrReport::DoReportOrient()
{
   std::map<tNamePair,cIrb_SigmaInstr>  aMoyPair;

   for (const auto & [aTimeS,aDataTS] : mBlock->DataTS())
   {
      // const auto & aVCam = aDataTS.SetCams().VCompPoses();
       for (int aK1=0 ; aK1< mNbCams ;aK1++)
       {
            for (int aK2=aK1+1 ; aK2< mNbCams ;aK2++)
            {
                tPoseR aPoseComp = aDataTS.SetCams().PoseRel(aK1,aK2);
                tPoseR aPoseCal = mCamCal->PoseRel(aK1,aK2);
                tREAL8 aDTr = Norm2(aPoseComp.Tr()- aPoseCal.Tr());
                tREAL8 aDRot = aPoseComp.Rot().Dist(aPoseCal.Rot());
                std::string aN1 = mCamCal->KthCam(aK1).NameCal();
                std::string aN2 = mCamCal->KthCam(aK2).NameCal();

                tNamePair aPair(aN1,aN2);
                aMoyPair[aPair].AddNewSigma(cIrb_SigmaInstr(1.0,aDTr,aDRot));
            }
       }
   }

   for (const auto & [aPt,aSigma] : aMoyPair)
       StdOut() << aSigma.SigmaTr() << " " << aSigma.SigmaRot() << "\n";
}



int cAppli_BlockInstrReport::Exe()
{
    mPhProj.FinishInit();

    // read an existing bloc from std folder
    mBlock = new cIrbComp_Block(mPhProj,mNameBloc);
    mCalBlock = & mBlock->CalBlock();
    mCamCal = & mCalBlock->SetCams();
    mNbCams = mCamCal->NbCams();

    // size_t aNbCam = mBlock->NbCams();

    //  add all the camera
    for (const auto & aNameIm :  VectMainSet(0))
    {
       mBlock->AddImagePose(aNameIm);
    }

    DoReportOrient();


    delete mBlock;

    return EXIT_SUCCESS;
}


    /* ==================================================== */
    /*                                                      */
    /*               MMVII                                  */
    /*                                                      */
    /* ==================================================== */


tMMVII_UnikPApli Alloc_BlockInstrReport(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_BlockInstrReport(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_BlockInstrReport
(
     "BlockInstrReport",
      Alloc_BlockInstrReport,
      "Make a report on a block of instrument confronted to data",
      {eApF::BlockInstr,eApF::Ori},
      {eApDT::BlockInstr,eApDT::Ori},
      {eApDT::Csv},
      __FILE__
);


};

