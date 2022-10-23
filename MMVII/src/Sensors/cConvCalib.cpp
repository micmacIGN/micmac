#include "MMVII_PCSens.h"
#include "MMVII_MMV1Compat.h"
#include "MMVII_DeclareCste.h"

/**
   \file cConvCalib.cpp

   \brief file for conversion between calibration (change format, change model) and tests
*/


namespace MMVII
{


static const std::string ThePatOriV1 = "Orientation-(.*)\\.xml";
static std::string V1NameOri2NameImage(const std::string & aNameOri) {return ReplacePattern(ThePatOriV1,"$1",aNameOri);}


/**  Class for otimizing a model of camera  using 3d-2d correspondance and bundle adjustment.  Typically these
 *   corresponance will be synthetic ones coming from another camera. It can be used in, two scenario :
 *
 *    -(1) primary test/bench  on functionnality to do BA
 *    -(2)
 *        (2.a)   conversion between calibration (format/model ...)
 *        (2.b)   comparison of calibrations (to come)
 *
 *    In first case we create artifcially difficult conditions (randomize the initial pose, let free the perspective center).
 *
 *    In the second case,  we use as much information we have : init with identity, and froze the position center
 *
 */

class cCentralPerspConversion
{
    public :
         typedef cIsometry3D<tREAL8>   tPose;
         cCentralPerspConversion
         (
              cPerspCamIntrCalib * ,
              const cSet2D3D &,
              const tPose & aPoseInit = tPose::Identity(),
              bool    HardConstrOnGCP=true , // do we fix GCP,  false make sense in test mode
              bool    CenterFix=true        // do we fix centre of projection,  false make sense in test mode
         );
         ~cCentralPerspConversion();

         static cCentralPerspConversion *  AllocV1Converter(const std::string & aFullName,bool HCG,bool  CenterFix);
         static cPerspCamIntrCalib *       AllocCalibV1(const std::string & aFullName);
         static cSensorCamPC *             AllocSensorPCV1(const std::string& aNameIm,const std::string & aFullName);

         void OneIteration();


	 void ResetUk() 
	 {
		 // MMVII_WARGNING("cCentralPerspConversion ResetUk");
		 mSetInterv.Reset();
	 }
         const cSet2D3D  & SetCorresp() const {return   mSetCorresp;}

         const cSensorCamPC  &       CamPC() const {return mCamPC;}
         cPerspCamIntrCalib *    Calib() {return mCalib;}

    private :
         tPose                              mPoseInit;
         bool                               mHCG; // HardConstrOnGCP
         bool                               mCFix; // HardConstrOnGCP

         cPerspCamIntrCalib *               mCalib;
         cSensorCamPC                       mCamPC;
         cSet2D3D                           mSetCorresp;
         int                                mSzBuf;
         cCalculator<double> *              mEqColinearity;
         cSetInterUK_MultipeObj<double>     mSetInterv;
         cResolSysNonLinear<double> *       mSys;
};

     // ==============  constructor & destructor ================

cCentralPerspConversion::cCentralPerspConversion
(
     cPerspCamIntrCalib *    aCalib,
     const cSet2D3D &        aSetCorresp,
     const tPose &           aPoseInit ,
     bool                    HardConstrOnGCP,
     bool                    CenterFix
) :
    mPoseInit      (aPoseInit),
    mHCG           (HardConstrOnGCP),
    mCFix          (CenterFix),
    mCalib         (aCalib),
    mCamPC         ("NONE",mPoseInit,mCalib),
    mSetCorresp    (aSetCorresp),
    mSzBuf         (100),
    mEqColinearity (mCalib->EqColinearity(true,mSzBuf))
{
    mSetInterv.AddOneObj(&mCamPC);
    mSetInterv.AddOneObj(mCalib);

    mSys = new cResolSysNonLinear<double>(eModeSSR::eSSR_LsqDense,mSetInterv.GetVUnKnowns());
}

cCentralPerspConversion::~cCentralPerspConversion()
{
    delete mEqColinearity;
    delete mSys;
}

     // ==============   Iteration to  ================

void cCentralPerspConversion::OneIteration()
{
     if (mCFix)
     {
        mSys->SetFrozenVar(mCamPC,mCamPC.Center());
     }
     std::vector<int> aVIndGround{-1,-2,-3};

     // Fill indexe Glob in the same order as in cEqColinearityCamPPC::VNamesUnknowns()
     std::vector<int> aVIndGlob = aVIndGround;
     mCamPC.FillIndexes(aVIndGlob);
     mCalib->FillIndexes(aVIndGlob);

     for (const auto & aCorresp : mSetCorresp.Pairs())
     {
         // structure for points substistion, in mode test
         cSetIORSNL_SameTmp<tREAL8>   aStrSubst
                                      (
                                         aCorresp.mP3.ToStdVector() ,
                                          (mHCG ? aVIndGround : std::vector<int>())
                                      );

         if (! mHCG)
         {
            for (const auto & anInd : aVIndGround)
               aStrSubst.AddFixCurVarTmp(anInd,1.0);
         }

         // "observation" of equation  : PTIm (real obs) + Cur-Rotation to avoid guimbal-lock
         std::vector<double> aVObs = aCorresp.mP2.ToStdVector();
         mCamPC.Pose().Rot().Mat().PushByCol(aVObs);

         mSys->AddEq2Subst(aStrSubst,mEqColinearity,aVIndGlob,aVObs);
         mSys->AddObsWithTmpUK(aStrSubst);
     }

     const auto & aVectSol = mSys->SolveUpdateReset();
     mSetInterv.SetVUnKnowns(aVectSol);
}


     // ==============  conversion for V1 ================

cCentralPerspConversion *  cCentralPerspConversion::AllocV1Converter(const std::string & aFullName,bool HCG,bool  CenterFix)
{
     bool  isForTest = (!HCG) ||  (! CenterFix);
     cExportV1StenopeCalInterne  aExp(true,aFullName,15);

     cIsometry3D<tREAL8>   aPose0 = cIsometry3D<tREAL8>::Identity();
     // in mode test perturbate internal et external parameters
     if (isForTest)
     {
         aExp.mFoc *=  (1.0 + 0.05*RandUnif_C());
         aExp.mPP = MulCByC(aExp.mPP,  cPt2dr(1,1)+cPt2dr::PRandC()*0.05);
         aPose0 = cIsometry3D<tREAL8>
                         (
                              cPt3dr::PRandC() * (CenterFix ? 0.0 : 0.1),
                              cRotation3D<tREAL8>::RandomRot(0.05)
                         );
     }

     std::string aNameCam = LastPrefix(FileOfPath(aFullName,false));
     aNameCam =  ReplacePattern("AutoCal_(.*)","$1",aNameCam);
     cDataPerspCamIntrCalib aDataCalib(cPerspCamIntrCalib::PrefixName() +aNameCam,aExp.eProj,cPt3di(3,1,1),aExp.mFoc,aExp.mSzCam);
     aDataCalib.PushInformation("Converted from MMV1");
     cPerspCamIntrCalib * aCalib = cPerspCamIntrCalib::Alloc(aDataCalib);

     return new cCentralPerspConversion(aCalib,aExp.mCorresp,aPose0,HCG,CenterFix);
}

cPerspCamIntrCalib * cCentralPerspConversion::AllocCalibV1(const std::string & aFullName)
{ 
     static std::map<std::string,cPerspCamIntrCalib *> TheMap;
     cPerspCamIntrCalib * & aPersp = TheMap[aFullName];

     if (aPersp==0)
     {
         cCentralPerspConversion * aConvertor = cCentralPerspConversion::AllocV1Converter(aFullName,true,true);

         for (int aK=0 ; aK<10 ; aK++)
         {
            aConvertor->OneIteration();
         }

         aPersp = aConvertor->Calib();
	 cMMVII_Appli::AddObj2DelAtEnd(aPersp);
         aConvertor->ResetUk();
	 delete aConvertor;
     }

     return aPersp;
}

cSensorCamPC * cCentralPerspConversion::AllocSensorPCV1(const std::string & aNameIm,const std::string & aFullName)
{
     cExportV1StenopeCalInterne  aExp(false,aFullName,0); // Alloc w/o  3d-2d correspondance

     std::string aNameCal = DirOfPath(aFullName,false) + FileOfPath(aExp.mNameCalib,false);
     cPerspCamIntrCalib * aCalib =  cCentralPerspConversion::AllocCalibV1(aNameCal);

     return new cSensorCamPC(aNameIm,aExp.mPose,aCalib);
}



   /* ********************************************************** */
   /*                                                            */
   /*               BENCH PART                                   */
   /*                                                            */
   /* ********************************************************** */



void BenchCentralePerspective_ImportCalibV1(cParamExeBench & aParam,const std::string & aName,bool HCG,bool  CenterFix,double aAccuracy)
{
static int aCpt=0; aCpt++;

     std::string aFullName = cMMVII_Appli::CurrentAppli().InputDirTestMMVII() + "Ori-MMV1" +  StringDirSeparator() + aName;


     cCentralPerspConversion * aConv =   cCentralPerspConversion::AllocV1Converter(aFullName,HCG,CenterFix);

     const cSensorCamPC  &       aCamPC =  aConv->CamPC() ;
     cPerspCamIntrCalib *        aCalib =  aConv->Calib() ;

     double aResidual  = 10;
     for (int aK=0 ; aK<20 ; aK++)
     {
        aConv->OneIteration();
        aResidual  = aCamPC.AvgResidual(aConv->SetCorresp());

        if (aResidual<aAccuracy)
        {
            // create a new file , to avoid reading in map in "FromFile"
	    std::string aNameTmp = cMMVII_Appli::CurrentAppli().TmpDirTestMMVII() + "TestCalib_" + ToStr(aCpt) + ".xml";
	    aCalib->ToFile(aNameTmp);

	    cPerspCamIntrCalib *  aCam2 = cPerspCamIntrCalib::FromFile(aNameTmp);
            cSensorCamPC          aSensor2 ("NONE",aConv->CamPC().Pose(),aCam2) ;
	    double aR2 = aSensor2.AvgResidual(aConv->SetCorresp());
           
	    //  Accuracy must be as good as initial camera, but small diff possible due to string conv
            MMVII_INTERNAL_ASSERT_bench(aR2< aAccuracy+1e-5  ,"Reload camera  xml");
	    // diff of accuracy should be tiny
            MMVII_INTERNAL_ASSERT_bench(std::abs(aR2-aResidual)< 1e-10  ,"Reload camera  xml");

	    delete aConv;
	    //delete aCam2; dont delete -> create by FromFile, autom delete at end
	    delete aCalib; // create at hand  -> delete
            return;
        }
     }
     StdOut() << "---------------RR=" <<  aResidual  << "\n";
     MMVII_INTERNAL_ASSERT_bench(false ,"No convergence in BenchCentralePerspective_ImportV1");
}



void BenchPoseImportV1(const std::string & aNameOriV1,double anAccuracy)
{
     std::string aFullName = cMMVII_Appli::CurrentAppli().InputDirTestMMVII() + "Ori-MMV1" +  StringDirSeparator() + aNameOriV1;

     // AllocSensorPCV1

     cExportV1StenopeCalInterne  aExp(false,aFullName,10); 
     cSensorCamPC  *aPC  =  cCentralPerspConversion::AllocSensorPCV1(V1NameOri2NameImage(aNameOriV1),aFullName);
     double aResidual  =  aPC->AvgResidual(aExp.mCorresp);


     for (const auto & aCorresp : aExp.mCorresp.Pairs())
     {
         cPt3dr aPGround = aCorresp.mP3;
	 cPt3dr aPImAndD = aPC->Ground2ImageAndDepth(aPGround);
	 cPt3dr aPG2 = aPC->ImageAndDepth2Ground(aPImAndD);

	 double aDist = Norm2(aPGround - aPG2);
          MMVII_INTERNAL_ASSERT_bench(aDist<1e-5 ,"I&Depth inversion");
     }

     MMVII_INTERNAL_ASSERT_bench(aResidual<anAccuracy ,"No convergence in BenchCentralePerspective_ImportV1");

     std::string aNameTmp = cMMVII_Appli::CurrentAppli().TmpDirTestMMVII() +  aPC->NameOriStd();  // "ccTestOri.xml";
     aPC->ToFile(aNameTmp);


     cSensorCamPC  *aPC2  =  cSensorCamPC::FromFile(aNameTmp);
     double aR2 = aPC2->AvgResidual(aExp.mCorresp) ;
     MMVII_INTERNAL_ASSERT_bench(aR2<anAccuracy ,"No Conv in Reimport cam");


     delete aPC;
     delete aPC2;

}

void BenchCentralePerspective_ImportV1(cParamExeBench & aParam)
{
    BenchPoseImportV1("Orientation-_DSC8385.tif.xml" ,1e-5);
    BenchPoseImportV1("Orientation-Img0937.tif.xml"  ,1e-5);

    for (int aK=0 ; aK<3 ; aK++)
    {
        BenchCentralePerspective_ImportCalibV1(aParam,"AutoCal_Foc-60000_Cam-NIKON_D810.xml",false,false,1e-5);
        BenchCentralePerspective_ImportCalibV1(aParam,"AutoCal_Foc-60000_Cam-NIKON_D810.xml",false,true ,1e-5);
        BenchCentralePerspective_ImportCalibV1(aParam,"AutoCal_Foc-60000_Cam-NIKON_D810.xml",true ,false,1e-5);
        BenchCentralePerspective_ImportCalibV1(aParam,"AutoCal_Foc-60000_Cam-NIKON_D810.xml",true ,true ,1e-5);

        BenchCentralePerspective_ImportCalibV1(aParam,"AutoCal_Foc-11500_Cam-imx477imx477-1.xml",false,false,1e-3);
    }
}

    // ===============================================================================================
    // ===============================================================================================
    // ===============================================================================================


cPhotogrammetricProject::cPhotogrammetricProject(cMMVII_Appli & anAppli) :
    mAppli  (anAppli)
{
}

cPhotogrammetricProject::~cPhotogrammetricProject() 
{
    DeleteAllAndClear(mLCam2Del);
}

tPtrArg2007 cPhotogrammetricProject::OriInMand() {return  Arg2007(mOriIn ,"Input Orientation",{eTA2007::Orient,eTA2007::Input });}
tPtrArg2007 cPhotogrammetricProject:: OriOutMand() {return Arg2007(mOriOut,"Outot Orientation",{eTA2007::Orient,eTA2007::Output});}
tPtrArg2007 cPhotogrammetricProject::OriInOpt(){return AOpt2007(mOriIn,"InOri","Input Orientation",{eTA2007::Orient,eTA2007::Input});}

void cPhotogrammetricProject::FinishInit() 
{
    mFullOriOut  = mAppli.DirProject() + MMVIIDirOrient + mOriOut + StringDirSeparator();
    mFullOriIn   = mAppli.DirProject() + MMVIIDirOrient + mOriIn  + StringDirSeparator();

    if (mAppli.IsInit(&mOriOut))
    {
        CreateDirectories(mFullOriOut,true);
    }
}

void cPhotogrammetricProject::SaveCamPC(const cSensorCamPC & aCamPC) const
{
    aCamPC.ToFile(mFullOriOut + aCamPC.NameOriStd());
}

cSensorCamPC * cPhotogrammetricProject::AllocCamPC(const std::string & aNameIm,bool ToDelete)
{
    std::string aNameCam  = mFullOriIn + cSensorCamPC::NameOri_From_Image(aNameIm);
    cSensorCamPC * aCamPC =  cSensorCamPC::FromFile(aNameCam);

    if (ToDelete)
       mLCam2Del.push_back(aCamPC);

    return aCamPC;
}
/*
*/

   /* ********************************************************** */
   /*                                                            */
   /*                 cAppli_OriConvV1V2                         */
   /*                                                            */
   /* ********************************************************** */

class cAppli_OriConvV1V2 : public cMMVII_Appli
{
     public :
        cAppli_OriConvV1V2(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;
     private :
	std::string              mDirMMV1;
	cPhotogrammetricProject  mPhProj;
};

cAppli_OriConvV1V2::cAppli_OriConvV1V2(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli(aVArgs,aSpec),
   mPhProj (*this)
{
}

cCollecSpecArg2007 & cAppli_OriConvV1V2::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
    return anArgObl
	      <<  Arg2007(mDirMMV1 ,"Input Orientation for MMV1 Files")
	      <<  mPhProj.OriOutMand()
           ;
}

cCollecSpecArg2007 & cAppli_OriConvV1V2::ArgOpt(cCollecSpecArg2007 & anArgObl) 
{
    
    return anArgObl
           ;
}


int cAppli_OriConvV1V2::Exe()
{
    mPhProj.FinishInit();

    //std::string aPatV1 = "Orientation-(.*)\\.xml";
    std::vector<std::string> aListOriV1 = GetFilesFromDir(mDirMMV1,AllocRegex(ThePatOriV1),true);

    for (const auto & aNameOri : aListOriV1)
    {
        std::string aNameIm = V1NameOri2NameImage(aNameOri); // ReplacePattern(ThePatOriV1,"$1",aName);
        cSensorCamPC * aPC =  cCentralPerspConversion::AllocSensorPCV1(aNameIm,mDirMMV1+aNameOri);

	mPhProj.SaveCamPC(*aPC);

	/*
        StdOut() << "N="  << aNameOri << " " << aPC->Center() 
		<< " CalN=" << aPC->InternalCalib()->Name() 
		<< " ### [" <<   aNameIm << "]\n";
		*/


	delete aPC;
    }


    return EXIT_SUCCESS;
}


tMMVII_UnikPApli Alloc_OriConvV1V2(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_OriConvV1V2(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_OriConvV1V2
(
     "OriConvV1V2",
      Alloc_OriConvV1V2,
      "Convert orientation of MMV1  to MMVII",
      {eApF::Ori},
      {eApDT::Orient},
      {eApDT::Orient},
      __FILE__
);


}; // MMVII

