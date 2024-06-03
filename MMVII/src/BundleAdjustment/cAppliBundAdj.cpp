#include "BundleAdjustment.h"

/**
   \file cAppliBundAdj.cpp

*/
#include "../Topo/Topo.h"

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

        std::vector<tREAL8>  ConvParamStandard(const std::vector<std::string> &,size_t aSzMin,size_t aSzMax) ;
        /// New Method for multiple GCP : each 
        void  AddOneSetGCP(const std::vector<std::string> & aParam);
        void  AddOneSetTieP(const std::vector<std::string> & aParam);

	std::string               mSpecImIn;

	std::string               mDataDir;  /// Default Data dir for all

	cPhotogrammetricProject   mPhProj;
	cMMVII_BundleAdj          mBA;

	std::vector<std::string>  mGCPW;
	std::vector<std::vector<std::string>>  mAddGCPW; // In case there is multiple GCP Set
        std::string               mGCPFilter;  // pattern to filter names of GCP
        std::string               mGCPFilterAdd;  // pattern to filter GCP by additional info
	std::vector<std::string>  mTiePWeight;
	std::vector<std::vector<std::string>>  mAddTieP; // In case there is multiple GCP Set
	std::vector<double>       mBRSigma; // RIGIDBLOC
	std::vector<double>       mBRSigma_Rat; // RIGIDBLOC
        std::vector<std::string>  mParamRefOri;

	int                       mNbIter;

	std::string               mPatParamFrozCalib;
	std::string               mPatFrosenCenters;
	std::string               mPatFrosenOrient;
	std::vector<tREAL8>       mViscPose;
        tREAL8                    mLVM;  ///< Levenberk Markard
        bool                      mMeasureAdded ;
        std::vector<std::string>  mVSharedIP;  ///< Vector for shared intrinsic param
};

cAppliBundlAdj::cAppliBundlAdj(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli    (aVArgs,aSpec),
   mDataDir        ("Std"),
   mPhProj         (*this),
   mBA             (&mPhProj),
   mGCPFilter      (""),
   mGCPFilterAdd   (""),
   mNbIter         (10),
   mLVM            (0.0),
   mMeasureAdded   (false)
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
      << AOpt2007(mDataDir,"DataDir","Default data directories ",{eTA2007::HDV})
      << AOpt2007(mNbIter,"NbIter","Number of iterations",{eTA2007::HDV})
      << mPhProj.DPPointsMeasures().ArgDirInOpt("GCPDir","Dir for GCP if != DataDir")
      << mPhProj.DPMulTieP().ArgDirInOpt("TPDir","Dir for Tie Points if != DataDir")
      << mPhProj.DPRigBloc().ArgDirInOpt("BRDirIn","Dir for Bloc Rigid if != DataDir") //  RIGIDBLOC
      << mPhProj.DPRigBloc().ArgDirOutOpt() //  RIGIDBLOC
      << mPhProj.DPTopoMes().ArgDirInOpt("TopoDirIn","Dir for Topo measures") //  TOPO
      << mPhProj.DPTopoMes().ArgDirOutOpt() //  TOPO
      << AOpt2007
         (
            mGCPW,
            "GCPW",
            "GCP Weight [SigG,SigI,SigAt?=-1,Thrs?=-1,Exp?=1], SG=0 fix, SG<0 schurr elim, SG>0",
            {{eTA2007::ISizeV,"[2,5]"}}
         )
      << AOpt2007(mAddGCPW,"AddGCPW","For additional GPW, [[Folder,SigG...],[Folder,...]] ")
      << AOpt2007(mGCPFilter,"GCPFilter","Pattern to filter GCP by name")
      << AOpt2007(mGCPFilterAdd,"GCPFilterAdd","Pattern to filter GCP by additional info")
      << mPhProj.DPPointsMeasures().ArgDirOutOpt("GCPDirOut","Dir for output GCP")
      << AOpt2007(mTiePWeight,"TiePWeight","Tie point weighting [Sig0,SigAtt?=-1,Thrs?=-1,Exp?=1]",{{eTA2007::ISizeV,"[1,4]"}})
      << AOpt2007(mAddTieP,"AddTieP","For additional TieP, [[Folder,SigG...],[Folder,...]] ")
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



std::vector<tREAL8>  cAppliBundlAdj::ConvParamStandard(const std::vector<std::string> & aVParStd,size_t aSzMin,size_t aSzMax)
{
    if ((aVParStd.size() <aSzMin) || (aVParStd.size() >aSzMax))
    {
        MMVII_UnclasseUsEr("Bad size of AddOneSetGCP, exp in [3,6] got : " + ToStr(aVParStd.size()));
    }
    mMeasureAdded = true;  // to avoid message corresponding to trivial error

    std::vector<tREAL8>  aRes;  // then weight must be converted from string to double
    for (size_t aK=1 ; aK<aVParStd.size() ; aK++)
        aRes.push_back(cStrIO<double>::FromStr(aVParStd.at(aK)));

    return aRes;
}

// VParam standar is done from  Folder +  weight of size [2,5]
void  cAppliBundlAdj::AddOneSetGCP(const std::vector<std::string> & aVParStd)
{
    std::string aFolder = aVParStd.at(0);  // folder
    std::vector<tREAL8>  aGCPW = ConvParamStandard(aVParStd,3,6);
/*
    if ((aVParStd.size() <3) || (aVParStd.size() >6))
    {
        MMVII_UnclasseUsEr("Bad size of AddOneSetGCP, exp in [3,6] got : " + ToStr(aVParStd.size()));
    }
    mMeasureAdded = true;  // to avoid message corresponding to trivial error

    //  convert the lo level aVParStd in more structured 
    std::vector<tREAL8>  aGCPW;  // then weight must be converted from string to double
    for (size_t aK=1 ; aK<aVParStd.size() ; aK++)
        aGCPW.push_back(cStrIO<double>::FromStr(aVParStd.at(aK)));
*/
    
    //  load the GCP
    cSetMesImGCP  aFullMesGCP; 
    mPhProj.LoadGCPFromFolder(aFolder,aFullMesGCP,mBA.getTopo(),"",mGCPFilter,mGCPFilterAdd);

    for (const auto  & aSens : mBA.VSIm())
    {
        // Load the images measure + init sens 
        mPhProj.LoadImFromFolder(aFolder,aFullMesGCP,aSens->NameImage(),aSens,SVP::Yes);
    }
    cSetMesImGCP * aMesGCP = aFullMesGCP.FilterNonEmptyMeasure(0);

    cStdWeighterResidual aWeighter(aGCPW,1);
    mBA.AddGCP(aFolder,aGCPW.at(0),aWeighter,aMesGCP);
}

void  cAppliBundlAdj::AddOneSetTieP(const std::vector<std::string> & aVParStd)
{
    std::string aFolder = aVParStd.at(0);  // folder
    std::vector<tREAL8>  aTiePW = ConvParamStandard(aVParStd,3,6);
    cStdWeighterResidual aWeighter(aTiePW,0);
    mBA.AddMTieP(aFolder,AllocStdFromMTPFromFolder(aFolder,VectMainSet(0),mPhProj,false,true,false),aWeighter);
}


int cAppliBundlAdj::Exe()
{
/*
{
    StdOut() << "TESTT  mAddGCPWmAddGCPW\n";
    StdOut() << mAddGCPW  << mAddGCPW.size() << "\n";
    for (const auto& aGCP : mAddGCPW)
         StdOut() << " * " << aGCP  << aGCP.size() << "\n";
        


    getchar();
}
*/

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
	   

    if (IsInit(&mGCPW))  // Add if any first the standard GCP weighting
    {
        std::vector<std::string>  aVParamStdGCP{mPhProj.DPPointsMeasures().DirIn()};
        AppendIn(aVParamStdGCP,mGCPW);
        AddOneSetGCP(aVParamStdGCP);
    }
    // Add  the potential suplementary GCP
    for (const auto& aGCP : mAddGCPW)
        AddOneSetGCP(aGCP);

    if (IsInit(&mTiePWeight))
    {
        std::vector<std::string>  aVParamTieP{mPhProj.DPMulTieP().DirIn()};
        AppendIn(aVParamTieP,mTiePWeight);
        AddOneSetTieP(aVParamTieP);
    }
    // Add  the potential suplementary TieP
    for (const auto& aTieP : mAddTieP)
        AddOneSetTieP(aTieP);


    if (IsInit(&mBRSigma)) // RIGIDBLOC
    { 
        mBA.AddBlocRig(mBRSigma,mBRSigma_Rat);
        for (const auto &  aNameIm : VectMainSet(0))
            mBA.AddCamBlocRig(aNameIm);
    }

    if (mPhProj.DPTopoMes().DirInIsInit())
    {
        mBA.AddTopo();
    }

    MMVII_INTERNAL_ASSERT_User(mMeasureAdded,eTyUEr::eUnClassedError,"Not any measure added");

    for (int aKIter=0 ; aKIter<mNbIter ; aKIter++)
    {
        mBA.OneIteration(mLVM);
    }

    for (auto & aSI : mBA.VSIm())
        mPhProj.SaveSensor(*aSI);
	    /*
    for (auto & aCamPC : mBA.VSCPC())
        mPhProj.SaveCamPC(*aCamPC);
	*/

    mPhProj.CpSysIn2Out(true,true);

    mBA.SaveBlocRigid();  // RIGIDBLOC
    mBA.Save_newGCP();
    mBA.SaveTopo(); // just for debug for now

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

