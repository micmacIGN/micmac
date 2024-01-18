#include "BundleAdjustment.h"

/**
   \file cAppliBundAdj.cpp

*/

/*
    Track info on bundle adj/push broom in V1 :

  * mm3d Campari

       * [Name=PdsGBRot] REAL :: {Weighting of the global rotation constraint (Generic bundle Def=0.002)}
       * [Name=PdsGBId] REAL :: {Weighting of the global deformation constraint (Generic bundle Def=0.0)}
       * [Name=PdsGBIter] REAL :: {Weighting of the change of the global rotation constraint between iterations (Generic bundle Def=1e-6)}

  *  micmac/src/uti_phgrm/CPP_Campari.cpp [POS=0486,0016] 
    
          Apero ... "Apero-Compense.xml" ...
             +  std::string(" +PdsGBRot=") + ToString(aPdsGBRot) + " "

  * micmac/include/XML_MicMac/Apero-Compense.xml 

       <ContrCamGenInc>
             <PatternApply> .*  </PatternApply>
             <PdsAttachToId>   ${PdsGBId}     </PdsAttachToId>
             <PdsAttachToLast> ${PdsGBIter}    </PdsAttachToLast>
             <PdsAttachRGLob>  ${PdsGBRot}    </PdsAttachRGLob>
        </ContrCamGenInc>


  *  micmac/src/uti_phgrm/Apero/cPosePolynGenCam.cpp 

       cPolynBGC3M2D_Formelle * aPF = aGPC->PolyF();

       if (aCCIG.PdsAttachRGLob().IsInit())
          aPF->AddEqRotGlob(aCCIG.PdsAttachRGLob().Val()*aGPC->SomPM());


   * micmac/src/uti_phgrm/Apero/cGenPoseCam.cpp
   * micmac/src/uti_phgrm/Apero/BundleGen.h       

*/


namespace MMVII
{

   /* ********************************************************** */
   /*                                                            */
   /*                 cAppliBundlAdj                             */
   /*                                                            */
   /* ********************************************************** */

class cAppliBundlAdj : public cMMVII_Appli
{
     public :
        cAppliBundlAdj(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;
     private :

	std::string               mSpecImIn;

	std::string               mDataDir;  /// Default Data dir for all

	cPhotogrammetricProject   mPhProj;
	cMMVII_BundleAdj          mBA;

	std::vector<double>       mGCPW;
        std::string               mGCPFilter;  // pattern to filter names of GCP
	std::vector<double>       mTiePWeight;
	std::vector<double>       mBRSigma; // RIGIDBLOC
	std::vector<double>       mBRSigma_Rat; // RIGIDBLOC
        std::vector<std::string>  mParamRefOri;

	int                       mNbIter;
	std::string               mPatParamFrozCalib;
	std::string               mPatFrosenCenters;
	std::string               mPatFrosenOrient;
	std::vector<tREAL8>       mViscPose;
        tREAL8                    mLVM;  // Levenberk Markard
        std::vector<std::string>  mVSharedIP;  // Vector for shared intrinsic param
};

cAppliBundlAdj::cAppliBundlAdj(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli  (aVArgs,aSpec),
   mDataDir      ("Std"),
   mPhProj       (*this),
   mBA           (&mPhProj),
   mGCPFilter    (""),
   mNbIter       (10),
   mLVM          (0.0)
{
}

cCollecSpecArg2007 & cAppliBundlAdj::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
    return anArgObl
              << Arg2007(mSpecImIn,"Pattern/file for images",{{eTA2007::MPatFile,"0"},{eTA2007::FileDirProj}})
              <<  mPhProj.DPOrient().ArgDirInMand()
	      <<  mPhProj.DPOrient().ArgDirOutMand()
           ;
}


cCollecSpecArg2007 & cAppliBundlAdj::ArgOpt(cCollecSpecArg2007 & anArgOpt) 
{
    
    return 
          anArgOpt
      << AOpt2007(mDataDir,"DataDir","Defautl data directories ",{eTA2007::HDV})
      << AOpt2007(mNbIter,"NbIter","Number of iterations",{eTA2007::HDV})
      << mPhProj.DPPointsMeasures().ArgDirInOpt("GCPDir","Dir for GCP if != DataDir")
      << mPhProj.DPMulTieP().ArgDirInOpt("TPDir","Dir for Tie Points if != DataDir")
      << mPhProj.DPRigBloc().ArgDirInOpt("BRDirIn","Dir for Bloc Rigid if != DataDir") //  RIGIDBLOC
      << mPhProj.DPRigBloc().ArgDirOutOpt() //  RIGIDBLOC
      << AOpt2007
         (
            mGCPW,
            "GCPW",
            "GCP Weight [SigG,SigI,SigAt?=-1,Thrs?=-1,Exp?=1], SG=0 fix, SG<0 schurr elim, SG>0",
            {{eTA2007::ISizeV,"[2,5]"}}
         )
      << AOpt2007(mGCPFilter,"GCPFilter","Pattern to filter GCP from their name")
      << mPhProj.DPPointsMeasures().ArgDirOutOpt("GCPDirOut","Dir for output GCP")
      << AOpt2007(mTiePWeight,"TiePWeight","Tie point weighting [Sig0,SigAtt?=-1,Thrs?=-1,Exp?=1]",{{eTA2007::ISizeV,"[1,4]"}})
      << AOpt2007(mPatParamFrozCalib,"PPFzCal","Pattern for freezing internal calibration parameters")
      << AOpt2007(mPatFrosenCenters,"PatFzCenters","Pattern of images for freezing center of poses")
      << AOpt2007(mPatFrosenOrient,"PatFzOrient","Pattern of images for freezing orientation of poses")
      << AOpt2007(mViscPose,"PoseVisc","Sigma viscosity on pose [SigmaCenter,SigmaRot]",{{eTA2007::ISizeV,"[2,2]"}})
      << AOpt2007(mLVM,"LVM","Levenberg–Marquardt parameter (to have better conditionning of least squares)",{eTA2007::HDV})
      << AOpt2007(mBRSigma,"BRW","Bloc Rigid Weighting [SigmaCenter,SigmaRot]",{{eTA2007::ISizeV,"[2,2]"}})  // RIGIDBLOC
      << AOpt2007(mBRSigma_Rat,"BRW_Rat","Rattachment fo Bloc Rigid Weighting [SigmaCenter,SigmaRot]",{{eTA2007::ISizeV,"[2,2]"}})  // RIGIDBLOC
      << AOpt2007(mParamRefOri,"RefOri","Reference orientation [Ori,SimgaTr,SigmaRot?,PatApply?]",{{eTA2007::ISizeV,"[2,4]"}})  
      << AOpt2007(mVSharedIP,"SharedIP","Shared intrinc parmaters [Pat1Cam,Pat1Par,Pat2Cam...] ",{{eTA2007::ISizeV,"[2,20]"}})    // ]]
    ;
}

int cAppliBundlAdj::Exe()
{
    bool  MeasureAdded = false; 

    mPhProj.DPPointsMeasures().SetDirInIfNoInit(mDataDir);
    mPhProj.DPMulTieP().SetDirInIfNoInit(mDataDir);
    mPhProj.DPRigBloc().SetDirInIfNoInit(mDataDir); //  RIGIDBLOC

    mPhProj.FinishInit();


    if (IsInit(&mParamRefOri))
         mBA.AddReferencePoses(mParamRefOri);

    for (const auto &  aNameIm : VectMainSet(0))
    {
         mBA.AddCam(aNameIm);
    }

    if (IsInit(&mPatParamFrozCalib))
    {
        mBA.SetParamFrozenCalib(mPatParamFrozCalib);
    }

    if (IsInit(&mPatFrosenCenters))
    {
        mBA.SetFrozenCenters(mPatFrosenCenters);
    }

    if (IsInit(&mPatFrosenOrient))
    {
        mBA.SetFrozenOrients(mPatFrosenOrient);
    }

    if (IsInit(&mViscPose))
    {
        mBA.SetViscosity(mViscPose.at(0),mViscPose.at(1));
    }

    if (IsInit(&mVSharedIP))
    {
        mBA.SetSharedIntrinsicParams(mVSharedIP);
    }
	   

    if (IsInit(&mGCPW))
    {
        //  load the GCP
        MeasureAdded = true;
        cSetMesImGCP  aFullMesGCP; 
	mPhProj.LoadGCP(aFullMesGCP,"",mGCPFilter);

        //if (IsInit(&mGCPFilter))
        //    aFullMesGCP = aFullMesGCP.Filter(mGCPFilter);

        for (const auto  & aSens : mBA.VSIm())
        {
             mPhProj.LoadIm(aFullMesGCP,aSens->NameImage(),aSens,true);
        }
	cSetMesImGCP * aMesGCP = aFullMesGCP.FilterNonEmptyMeasure(0);

	cStdWeighterResidual aWeighter(mGCPW,1);
	mBA.AddGCP(mGCPW.at(0),aWeighter,aMesGCP);
    }

    if (IsInit(&mTiePWeight))
    {
        MeasureAdded = true;
	cStdWeighterResidual aWeighter(mTiePWeight,0);
	mBA.AddMTieP(AllocStdFromMTP(VectMainSet(0),mPhProj,false,true,false),aWeighter);
    }

    if (IsInit(&mBRSigma)) // RIGIDBLOC
    { 
        mBA.AddBlocRig(mBRSigma,mBRSigma_Rat);
        for (const auto &  aNameIm : VectMainSet(0))
            mBA.AddCamBlocRig(aNameIm);
    }

    MMVII_INTERNAL_ASSERT_User(MeasureAdded,eTyUEr::eUnClassedError,"Not any measure added");

    for (int aKIter=0 ; aKIter<mNbIter ; aKIter++)
    {
        mBA.OneIteration(mLVM);
    }

    for (auto & aCamPC : mBA.VSCPC())
        mPhProj.SaveCamPC(*aCamPC);

    mPhProj.CpSysIn2Out(true,true);

    mBA.SaveBlocRigid();  // RIGIDBLOC
    mBA.Save_newGCP();

    return EXIT_SUCCESS;
}


tMMVII_UnikPApli Alloc_BundlAdj(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppliBundlAdj(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_OriBundlAdj
(
     "OriBundleAdj",
      Alloc_BundlAdj,
      "Bundle adjusment between images, using several observations/constraint",
      {eApF::Ori},
      {eApDT::Orient},
      {eApDT::Orient},
      __FILE__
);

/*
*/

}; // MMVII

