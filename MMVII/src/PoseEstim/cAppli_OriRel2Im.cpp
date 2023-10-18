#include "MMVII_PoseRel.h"
#include "MMVII_Tpl_Images.h"


namespace MMVII
{


   /* ********************************************************** */
   /*                                                            */
   /*                 cAppli_OriRel2Im                           */
   /*                                                            */
   /* ********************************************************** */

class cAppli_OriRel2Im : public cMMVII_Appli
{
     public :
	typedef cIsometry3D<tREAL8>  tPose;

        cAppli_OriRel2Im(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;
     private :
        tPose  EstimPose_By_MatEssL1Glob();

        cPhotogrammetricProject   mPhProj;
	std::string               mIm1;
	std::string               mIm2;
	cSetHomogCpleIm           mCpleH;
	bool                      mUseOri4GT;
	cPerspCamIntrCalib        * mCalib1;
	cPerspCamIntrCalib        * mCalib2;
	tREAL8                    mFocM;
        cSetHomogCpleDir          * mCpleDir;
	int                       mKMaxME;
	tPoseR                    mGTPose;
};

cAppli_OriRel2Im::cAppli_OriRel2Im(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
    cMMVII_Appli (aVArgs,aSpec),
    mPhProj      (*this),
    mUseOri4GT   (false),
    mGTPose      (tPoseR::Identity())
{
}

cCollecSpecArg2007 & cAppli_OriRel2Im::ArgObl(cCollecSpecArg2007 & anArgObl)
{
    return anArgObl
              << Arg2007(mIm1,"name first image")
              << Arg2007(mIm2,"name second image")
              <<  mPhProj.DPOrient().ArgDirInMand("Input orientation for calibration")
              <<  mPhProj.DPTieP().ArgDirInMand()
           ;
}

cCollecSpecArg2007 & cAppli_OriRel2Im::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return    anArgOpt
          << AOpt2007(mUseOri4GT,"OriGT","Set if orientation contains also exterior as a ground truth",{eTA2007::HDV})
   ;
}

cAppli_OriRel2Im::tPose  cAppli_OriRel2Im::EstimPose_By_MatEssL1Glob()
{
     cLinearOverCstrSys<tREAL8> *  aSysL1 = AllocL1_Barrodale<tREAL8>(9);
     cMatEssential aMatEL1(*mCpleDir,*aSysL1,mKMaxME);

     StdOut()  <<  "CCCCC= " << aMatEL1.AvgCost(*mCpleDir,0.05)  * mFocM << "\n";
     StdOut()  <<  "CCCCC= " << aMatEL1.AvgCost(*mCpleDir,5.0/mFocM) * mFocM << "\n";
     StdOut()  <<  "MED= " << aMatEL1.KthCost(*mCpleDir,0.5) * mFocM << "\n";
     StdOut()  <<  "P90= " << aMatEL1.KthCost(*mCpleDir,0.9) * mFocM << "\n";
     StdOut()  <<  "MAX= " << aMatEL1.KthCost(*mCpleDir,1.1) * mFocM << "\n";
     tPose aRes  = aMatEL1.ComputePose(*mCpleDir);
     delete aSysL1;


     if (mUseOri4GT)
     {
         StdOut() << VUnit(mGTPose.Tr())  << aRes.Tr() << "\n";

	 mGTPose.Rot().Mat().Show();
	 StdOut() << "=============================\n";
	 aRes.Rot().Mat().Show();
     }

     return aRes;
}

int cAppli_OriRel2Im::Exe()
{
     mPhProj.FinishInit();

     OrderMinMax(mIm1,mIm2);
     mPhProj.ReadHomol(mCpleH,mIm1,mIm2);

     mCalib1 =  mPhProj.InternalCalibFromImage(mIm1);
     mCalib2 =  mPhProj.InternalCalibFromImage(mIm2);
     mCpleDir = new cSetHomogCpleDir(mCpleH,*mCalib1,*mCalib2);

     if (mUseOri4GT)
     {
         cSensorCamPC *  aPC1 = mPhProj.ReadCamPC(mIm1,true);
         cSensorCamPC *  aPC2 = mPhProj.ReadCamPC(mIm2,true);

	 mGTPose = aPC1->RelativePose(*aPC2);
     }

     mFocM = (mCalib1->F()+mCalib2->F()) /2.0;

     mKMaxME =  MatEss_GetKMax(*mCpleDir, 1e-6);
     EstimPose_By_MatEssL1Glob();

     delete mCpleDir;

     return EXIT_SUCCESS;
}


tMMVII_UnikPApli Alloc_OriRel2Im(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_OriRel2Im(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_OriRel2Im
(
     "OriPoseEstimRel2Im",
      Alloc_OriRel2Im,
      "Estimate relative orientation with different algorithms",
      {eApF::Ori},
      {eApDT::TieP},
      {eApDT::Orient},
      __FILE__
);



}; // MMVII




