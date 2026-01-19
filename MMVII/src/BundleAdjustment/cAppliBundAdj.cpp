#include "BundleAdjustment.h"

/**
   \file cAppliBundAdj.cpp

*/
#include "MMVII_Topo.h"

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

        void  AddOneSetGCP3D(const std::string & aFolderIn, const std::string &aFolderOut, tREAL8 aWFactor); // aFolderOut="" if no out
        void  AddOneSetGCP2D(const std::vector<std::string> & aVParStd);
        void  AddOneSetTieP(const std::vector<std::string> & aParam);

	std::string               mSpecImIn;

	std::string               mDataDir;  /// Default Data dir for all

	cPhotogrammetricProject   mPhProj;
	cMMVII_BundleAdj          mBA;

        std::vector<std::vector<std::string>>  mGCP3D; // gcp ground coords with sigma factor and optional output dir
        std::vector<std::vector<std::string>>  mGCP2D; // gcp image coords with weight
        std::string               mGCPFilter;  // pattern to filter names of GCP
        std::string               mGCPFilterAdd;  // pattern to filter GCP by additional info
	std::vector<std::string>  mTiePWeight;
	std::vector<std::vector<std::string>>  mAddTieP; // In case there is multiple GCP Set
	std::vector<double>       mBRSigma; // RIGIDBLOC
	std::vector<double>       mBRSigma_Rat; // RIGIDBLOC
        std::vector<std::string>  mParamRefOri;  // Force Poses to be +- equals to this reference

        std::vector<std::vector<std::string>>  mParamLidarPhgr; // parameters for lidar photogra/lidar via triangulation
        std::vector<std::vector<std::string>>  mParamLidarPhoto; // parameters for lidar photogra/lidar via rasterization

	int                       mNbIter;

	std::string               mPatParamFrozCalib;
	std::vector<std::vector<std::string>>  mVVParFreeCalib;
	std::string               mPatFrosenCenters;
	std::string               mPatFrosenOrient;
    std::string               mPatFrosenClino;
	std::vector<tREAL8>       mViscPose;
        tREAL8                    mLVM;  ///< Levenberk Markard
        bool                      mMeasureAdded ;
        std::vector<std::string>  mVSharedIP;  ///< Vector for shared intrinsic param
        bool                      mShow_Cond; ///< compute and show system condition number
        std::vector<std::string>  mParamShow_UK_UC;
        std::string               mPostFixReport;
        std::vector<std::string>  mParamLine;
        std::vector<std::vector<std::string>> mParamBOI;  //< Param for bloc of instrum
         std::vector<std::vector<std::string>> mParamBOIClino;
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
      << mPhProj.DPMulTieP().ArgDirInOpt("TPDir","Dir for Tie Points if != DataDir")
      << mPhProj.DPRigBloc().ArgDirInOpt("BRDirIn","Dir for Bloc Rigid if != DataDir") //  RIGIDBLOC
      << mPhProj.DPRigBloc().ArgDirOutOpt() //  RIGIDBLOC
      << mPhProj.DPTopoMes().ArgDirInOpt("TopoDirIn","Dir for Topo measures") //  TOPO
      << mPhProj.DPTopoMes().ArgDirOutOpt("TopoDirOut","Dir for Topo measures output") //  TOPO
      << mPhProj.DPClinoMeters().ArgDirInOpt("ClinoDirIn","Dir for Clino if != DataDir") //  CLINOBLOC
      << mPhProj.DPClinoMeters().ArgDirOutOpt("ClinoDirOut","Dir for Clino if != DataDir") //  CLINOBLOC
      << mPhProj.DPMeasuresClino().ArgDirInOpt()
      << AOpt2007 ( mGCP3D, "GCP3D", "GCP ground coords and sigma factor, SG=0 fix, SG<0 schurr elim, SG>0 and optional output dir [[Folder,SigG,FOut?],...]]")
      << AOpt2007 ( mGCP2D, "GCP2D", "GCP image coords and sigma factor and optional attenuation, threshold and exponent [[Folder,SigI,SigAt?=-1,Thrs?=-1,Exp?=1]...]")
      << AOpt2007(mGCPFilter,"GCPFilter","Pattern to filter GCP by name")
      << AOpt2007(mGCPFilterAdd,"GCPFilterAdd","Pattern to filter GCP by additional info")
      << AOpt2007(mTiePWeight,"TiePWeight","Tie point weighting [Sig0,SigAtt?=-1,Thrs?=-1,Exp?=1]",{{eTA2007::ISizeV,"[1,4]"}})
      << AOpt2007(mAddTieP,"AddTieP","For additional TieP, [[Folder,SigG...],[Folder,...]] ")
      << AOpt2007(mParamLidarPhgr,"LidarPhotogra","Paramaters for adj Lidar/Phgr via triangulation [[Mode,Ply,Sigma,Interp?,Perturbate?,NbPtsPerPatch=32]*]")
      << AOpt2007(mParamLidarPhoto,"LidarPhoto","Paramaters for adj Lidar/Phgr via rasterisation [[Mode,Ply,Sigma,Interp?,Perturbate?,NbPtsPerPatch=32]*]")
      << AOpt2007(mPatParamFrozCalib,"PPFzCal","Pattern for freezing internal calibration parameters")
      << AOpt2007(mVVParFreeCalib,"PPFreeCal","Pattern for free internal calibration parameters [[PatCal1,PatParam1],[PatCal2,PatParam2] ...] ")
      << AOpt2007(mPatFrosenCenters,"PatFzCenters","Pattern of images for freezing center of poses")
      << AOpt2007(mPatFrosenOrient,"PatFzOrient","Pattern of images for freezing orientation of poses")
      << AOpt2007(mPatFrosenClino,"PatFzClino","Pattern of clinometers for freezing boresight")
      << AOpt2007(mViscPose,"PoseVisc","Sigma viscosity on pose [SigmaCenter,SigmaRot]",{{eTA2007::ISizeV,"[2,2]"}})
      << AOpt2007(mLVM,"LVM","Levenberg–Marquardt parameter (to have better conditioning of least squares)",{eTA2007::HDV})
      << AOpt2007(mBRSigma,"BRW","Bloc Rigid Weighting [SigmaCenter,SigmaRot]",{{eTA2007::ISizeV,"[2,2]"}})  // RIGIDBLOC
      << AOpt2007(mBRSigma_Rat,"BRW_Rat","Rattachment fo Bloc Rigid Weighting [SigmaCenter,SigmaRot]",{{eTA2007::ISizeV,"[2,2]"}})  // RIGIDBLOC

      << AOpt2007(mParamRefOri,"RefOri","Reference orientation [Ori,SimgaTr,SigmaRot?,PatApply?]",{{eTA2007::ISizeV,"[2,4]"}})
      << AOpt2007(mVSharedIP,"SharedIP","Shared intrinc parmaters [Pat1Cam,Pat1Par,Pat2Cam...] ",{{eTA2007::ISizeV,"[2,20]"}})

      << AOpt2007(mShow_Cond,"Cond","Compute and show system condition number")
      << AOpt2007(mParamShow_UK_UC,"UC_UK","Param for uncertainty & Show names of unknowns (tuning)")
      << AOpt2007(mPostFixReport,NameParamPostFixReport(),CommentParamPostFixReport())
      << AOpt2007(mParamLine,"AdjLine3D","Parameter for line Adjustment [SigmaIm,NbPtsSampl]",{{eTA2007::ISizeV,"[2,2]"}})

      << AOpt2007
         (
             mParamBOI,
             "BOI",
             "Bloc of Instr [[Bloc?,RelSigTrPair?=1.0,RelSigRotPair?=1.0,SaveSig?=1],[GjTr?,GjRot?],[RelSigTrCur,RelSigRotCur]?]",
             {{eTA2007::ISizeV,"[2,3]"}}
          )

      << AOpt2007
         (
             mParamBOIClino,
             "ClinpBOI",
             "Clino parameter [[Bloc?,RelSigmaAngle?,RelCstrOrthog?][VertFree?,OkNewTs?][DegFree0?,DegFree1?...]]",
             {{eTA2007::ISizeV,"[1,3]"}}
          )

      << mPhProj.DPBlockInstr().ArgDirInOpt()
      << mPhProj.DPBlockInstr().ArgDirOutOpt()


    ;
}



std::vector<tREAL8>  cAppliBundlAdj::ConvParamStandard(const std::vector<std::string> & aVParStd,size_t aSzMin,size_t aSzMax)
{
    if ((aVParStd.size() <aSzMin) || (aVParStd.size() >aSzMax))
    {
        MMVII_UnclasseUsEr("Bad size of AddOneSetGCP/TieP, exp in ["+ToStr(aSzMin)+","+ToStr(aSzMax)+"] got : " + ToStr(aVParStd.size()));
    }
    mMeasureAdded = true;  // to avoid message corresponding to trivial error

    std::vector<tREAL8>  aRes;  // then weight must be converted from string to double
    for (size_t aK=1 ; aK<aVParStd.size() ; aK++)
        aRes.push_back(cStrIO<double>::FromStr(aVParStd.at(aK)));

    return aRes;
}

void  cAppliBundlAdj::AddOneSetGCP3D(const std::string & aFolderIn, const std::string & aFolderOut, tREAL8 aWFactor)
{
    cSetMesGndPt  aFullMesGCP;
    cMes3DDirInfo * aMesDirInfo = cMes3DDirInfo::addMes3DDirInfo(mBA.getGCP(), aFolderIn, aFolderOut, aWFactor);
    mPhProj.LoadGCP3DFromFolder(aFolderIn, aFullMesGCP, aMesDirInfo, "", mGCPFilter, mGCPFilterAdd);
    auto aFullMes3D = aFullMesGCP.ExtractSetGCP("???");
    mBA.AddGCP3D(aMesDirInfo,aFullMes3D);
}


// VParam standar is done from  Folder +  weight of size [1,4]
void  cAppliBundlAdj::AddOneSetGCP2D(const std::vector<std::string> & aVParStd)
{
    std::string aFolderIn = aVParStd.at(0);  // folder
    std::vector<tREAL8>  aGCPW = ConvParamStandard(aVParStd,2,5);
    cMes2DDirInfo * aMesDirInfo = cMes2DDirInfo::addMes2DDirInfo(mBA.getGCP() ,aFolderIn, cStdWeighterResidual(aGCPW,0));
    for (const auto  & aSens : mBA.VSIm())
    {
        // Load the images measure + init sens
        mPhProj.LoadImFromFolder(aFolderIn,mBA.getGCP().getMesGCP(),aMesDirInfo,aSens->NameImage(),aSens,SVP::Yes);
    }
}


void  cAppliBundlAdj::AddOneSetTieP(const std::vector<std::string> & aVParStd)
{
    std::string aFolder = aVParStd.at(0);  // folder
    std::vector<tREAL8>  aTiePW = ConvParamStandard(aVParStd,2,5);
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

    //   ========== [0]   initialisation of def values  =============================
    mPhProj.DPMulTieP().SetDirInIfNoInit(mDataDir);
    mPhProj.DPRigBloc().SetDirInIfNoInit(mDataDir); //  RIGIDBLOC

    mPhProj.FinishInit();

    {
        std::string aReportDir = mPhProj.DPOrient().DirIn() + "_" + mPhProj.DPOrient().DirOut();
        if (IsInit(& mPostFixReport))
           aReportDir += "_" + mPostFixReport;
        SetReportSubDir(aReportDir);
    }


    if (IsInit(&mParamRefOri))
         mBA.AddReferencePoses(mParamRefOri);

    //   ========== [1]   Read unkowns of bundle  =============================
    for (const auto &  aNameIm : VectMainSet(0))
    {
         mBA.AddCam(aNameIm);
    }



    if (IsInit(&mPatParamFrozCalib))
    {
        mBA.SetParamFrozenCalib(mPatParamFrozCalib);
    }
    if (IsInit(&mVVParFreeCalib))
    {
        mBA.SetParamFreeCalib(mVVParFreeCalib);
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

    for (const auto& aVStrGCP : mGCP3D)
    {
        // expected: [Folder,SigG,FOut?]
        if ((aVStrGCP.size() <2) || (aVStrGCP.size() >3))
        {
            MMVII_UnclasseUsEr("Bad size of GCP3D, exp in [2,3] got : " + ToStr(aVStrGCP.size()));
        }
        AddOneSetGCP3D(aVStrGCP[0],aVStrGCP.size()>2?aVStrGCP[2]:"",cStrIO<double>::FromStr(aVStrGCP[1]));
    }

    if (mPhProj.DPTopoMes().DirInIsInit())
    {
        mBA.AddTopo();
    }

    for (const auto& aVStrGCP : mGCP2D)
    {
        // expected: [Folder,SigI,SigAt?=-1,Thrs?=-1,Exp?=1]
        AddOneSetGCP2D(aVStrGCP);
    }


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

    if (IsInit(&mParamLine))
    {
        mBA.AddLineAdjust(mParamLine);
    }

    if (IsInit(&mParamBOI))
    {
       mBA.AddBlockInstr(mParamBOI);
    }

    if (IsInit(&mParamBOIClino))
    {
       mBA.AddClinoBlokcInstr(mParamBOIClino);
    }
    
    if (mPhProj.DPClinoMeters().DirInIsInit())
    {
        mBA.AddClinoBloc();
    }

    if (IsInit(&mPatFrosenClino))
    {
        mBA.SetFrozenClinos(mPatFrosenClino);
    }
    
    for (const auto & aParam : mParamLidarPhgr)
    {
        MMVII_INTERNAL_ASSERT_User(aParam.size()>=3,eTyUEr::eUnClassedError,"Not enough parameters for LidarPhotogra");
        mMeasureAdded = true;
        mBA.Add1AdjLidarPhotogra(aParam);
    }

    for (const auto & aParam : mParamLidarPhoto)
    {
        MMVII_INTERNAL_ASSERT_User(aParam.size()>=3,eTyUEr::eUnClassedError,"Not enough parameters for LidarPhoto");
        mMeasureAdded = true;
        mBA.Add1AdjLidarPhoto(aParam);
    }


    MMVII_INTERNAL_ASSERT_User(mMeasureAdded,eTyUEr::eUnClassedError,"Not any measure added");

   if (IsInit(&mParamShow_UK_UC))
      mBA.Set_UC_UK(mParamShow_UK_UC);

    //   ========== [2]   Make Iteration =============================
    for (int aKIter=0 ; aKIter<mNbIter ; aKIter++)
    {
        bool isLastIter =  (aKIter==(mNbIter-1)) ;
        mBA.OneIteration(mLVM,isLastIter,mShow_Cond);
    }

    //   ========== [3]   Save resulst =============================
    for (auto & aSI : mBA.VSIm())
        mPhProj.SaveSensor(*aSI);
	    /*
    for (auto & aCamPC : mBA.VSCPC())
        mPhProj.SaveCamPC(*aCamPC);
	*/

    mPhProj.CpSysCoIn2Out(true,true);

    mBA.SaveBlocRigid();  // RIGIDBLOC
    mBA.Save_newGCP3D();
    mBA.SaveTopo(); // just for debug for now
    mBA.SaveClino();

    mBA.SaveBlockInstr();

    if (IsInit(&mParamShow_UK_UC))
    {
        mBA.ShowUKNames(mParamShow_UK_UC,mPostFixReport,this);
    }

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

