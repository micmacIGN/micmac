#include "MMVII_PCSens.h"
#include "MMVII_MMV1Compat.h"
#include "MMVII_DeclareCste.h"
#include "MMVII_BundleAdj.h"
//#include <set>

/**
   \file cAppliBundAdj.cpp

*/


namespace MMVII
{

/* ************************************************************* */
/*                                                               */
/*                                                               */
/*                                                               */
/* ************************************************************* */

/**   Class for representing a Pt of R3 in bundle adj, when it is considered as 
 *   unknown. 
 *      +  we have the exact value and uncertainty of the point
 *      -  it add (potentially many)  unknowns
 *      -  it take more place in  memory
 */	

class cPt3dr_UK :  public cObjWithUnkowns<tREAL8>,
	           public cMemCheck
{
      public :
	      cPt3dr_UK(const cPt3dr &);
	      ~cPt3dr_UK();
	      void PutUknowsInSetInterval() override;
	      const cPt3dr & Pt() const ;
      private :
	      cPt3dr_UK(const cPt3dr_UK&) = delete;
	      cPt3dr mPt;
};

class cMMVII_BundleAdj
{
     public :
          cMMVII_BundleAdj(cPhotogrammetricProject *);
          ~cMMVII_BundleAdj();

           // ======================== Add object ========================
	  void  AddCalib(cPerspCamIntrCalib *);  /// add  if not exist 
	  void  AddCamPC(cSensorCamPC *);  /// add, error id already exist
	  void  AddCam(const std::string & aNameIm);  /// add from name, require PhP exist
						      

	  ///  =======  Add GCP, can be measure or measure & object
	  void AddGCP(const  std::vector<double>&, cSetMesImGCP *);

	  /// One iteration : add all measure + constraint + Least Square Solve/Udpate/Init 
	  void OneIteration();

	  const std::vector<cSensorImage *> &  VSIm() const ;  ///< Accessor
	  const std::vector<cSensorCamPC *> &  VSCPC() const;   ///< Accessor
     private :

	  //============== Methods =============================
          cMMVII_BundleAdj(const cMMVII_BundleAdj &) = delete;
	  void  AddSensor(cSensorImage *);  /// add, error id already exist

          void AssertPhaseAdd() ;  /// Assert we are in phase add object (no iteration has been donne)
          void AssertPhp() ;             /// Assert we use a Photogram Project (class can be use w/o it)
          void AssertPhpAndPhaseAdd() ;  /// Assert both
	  void InitIteration();          /// Called at first iteration -> Init things and set we are non longer in Phase Add
          void InitItereGCP();           /// GCP Init => create UK
          void OneItere_GCP();           /// One iteraion of adding GCP measures

	  ///  One It for 1 pack of GCP (4 now 1 pack allowed, but this may change)
	  void OneItere_OnePackGCP(const cSetMesImGCP *,const std::vector<double> & aVW);


	  //============== Data =============================
          cPhotogrammetricProject * mPhProj;

	  bool  mPhaseAdd;  ///< check that we dont mix add & use of unknowns

	  cREAL8_RSNL *                 mSys;    /// base class can be 8 or 16 bytes
	  cResolSysNonLinear<tREAL8> *  mR8_Sys;  /// Real object, will disapear when fully interfaced for mSys

	  // ===================  Object to be adjusted ==================
	 
	  std::vector<cPerspCamIntrCalib *>  mVPCIC;     ///< vector of all internal calibration 4 easy parse
	  std::vector<cSensorCamPC *>        mVSCPC;      ///< vector of perspectiv  cameras
	  std::vector<cSensorImage *>        mVSIm;       ///< vector of sensor image (PC+RPC ...)
	  std::vector<cCalculator<double> *> mVEqCol;       ///< vector of sensor image (PC+RPC ...)

	  cSetInterUK_MultipeObj<tREAL8>    mSetIntervUK;


	  // ===================  Object to be adjusted ==================
	  cSetMesImGCP *           mMesGCP;
	  cSetMesImGCP             mNewGCP;
	  std::vector<double>      mWeightGCP;
	  std::vector<cPt3dr_UK*>  mGCP_UK;
};

//================================================================

cMMVII_BundleAdj::cMMVII_BundleAdj(cPhotogrammetricProject * aPhp) :
    mPhProj    (aPhp),
    mPhaseAdd  (true),
    mSys       (nullptr),
    mR8_Sys    (nullptr),
    mMesGCP    (nullptr)
{
}

cMMVII_BundleAdj::~cMMVII_BundleAdj() 
{
    delete mSys;
    delete mMesGCP;
    DeleteAllAndClear(mGCP_UK);
}


void cMMVII_BundleAdj::AssertPhaseAdd() 
{
    MMVII_INTERNAL_ASSERT_tiny(mPhaseAdd,"Mix Add and Use of unknown in cMMVII_BundleAdj");
}
void cMMVII_BundleAdj::AssertPhp() 
{
    MMVII_INTERNAL_ASSERT_tiny(mPhProj,"No cPhotogrammetricProject");
}

void cMMVII_BundleAdj::AssertPhpAndPhaseAdd() 
{
	AssertPhaseAdd();
	AssertPhp();
}


void cMMVII_BundleAdj::InitIteration()
{
    mPhaseAdd = false;

    InitItereGCP();
    mR8_Sys = new cResolSysNonLinear<tREAL8>(eModeSSR::eSSR_LsqNormSparse,mSetIntervUK.GetVUnKnowns());

    mSys =  mR8_Sys;
}


void cMMVII_BundleAdj::OneIteration()
{
    if (mPhaseAdd)
    {
        InitIteration();
    }

    OneItere_GCP();

    // StdOut() << "SYS=" << mR8_Sys->GetNbObs() << " " <<  mR8_Sys->NbVar() << "\n";

    const auto & aVectSol = mSys->R_SolveUpdateReset();
    mSetIntervUK.SetVUnKnowns(aVectSol);
}

//================================================================

void cMMVII_BundleAdj::AddCalib(cPerspCamIntrCalib * aCalib)  
{
    AssertPhaseAdd();
    if (! aCalib->UkIsInit())
    {
	  mVPCIC.push_back(aCalib);
	  mSetIntervUK.AddOneObj(aCalib);
    }
}

void cMMVII_BundleAdj::AddSensor(cSensorImage* aSI)
{
    AssertPhaseAdd();
    MMVII_INTERNAL_ASSERT_tiny (!aSI->UkIsInit(),"Multiple add of cam : " + aSI->NameImage());
    mSetIntervUK.AddOneObj(aSI);
    mVSIm.push_back(aSI);

    auto anEq = aSI->EqColinearity(true,10,true);  // WithDer, SzBuf, ReUse
    mVEqCol.push_back(anEq);
}


void cMMVII_BundleAdj::AddCamPC(cSensorCamPC * aCamPC)
{
    AddSensor(aCamPC);
    mVSCPC.push_back(aCamPC);

    AddCalib(aCamPC->InternalCalib());
}

void  cMMVII_BundleAdj::AddCam(const std::string & aNameIm)
{
    AssertPhpAndPhaseAdd();

    cSensorImage * aNewS;
    cSensorCamPC * aSPC;

    mPhProj->LoadSensor(aNameIm,aNewS,aSPC,false);  // false -> NoSVP
    if (aSPC)
       AddCamPC(aSPC);

}
const std::vector<cSensorImage *> &  cMMVII_BundleAdj::VSIm() const  {return mVSIm;}
const std::vector<cSensorCamPC *> &  cMMVII_BundleAdj::VSCPC() const {return mVSCPC;}


/* -------------------------------------------------------------- */
/*                cMMVII_BundleAdj::GCP                           */
/* -------------------------------------------------------------- */

void cMMVII_BundleAdj::AddGCP(const  std::vector<double>& aWeightGCP, cSetMesImGCP *  aMesGCP)
{
    mMesGCP = aMesGCP;
    mWeightGCP = aWeightGCP;

    if (1)
    {
        StdOut()<<  "MESIM=" << mMesGCP->MesImOfPt().size() << " MesGCP=" << mMesGCP->MesGCP().size()  << "\n";
    }
}

void cMMVII_BundleAdj::InitItereGCP()
{
    if ((mMesGCP!=nullptr) && (mWeightGCP[0] > 0))
    {

        for (const auto & aGCP : mMesGCP->MesGCP())
	{
              cPt3dr_UK * aPtrUK = new cPt3dr_UK(aGCP.mPt);
              mGCP_UK.push_back(aPtrUK);
	      mSetIntervUK.AddOneObj(aPtrUK);
	}
    }
}

void cMMVII_BundleAdj::OneItere_OnePackGCP(const cSetMesImGCP * aSet,const std::vector<double> & aVW)
{
    if (aSet==nullptr) return;
    //   W>0  obs is an unknown "like others"
    //   W=0 , obs is fix , use schurr subst and fix the variables
    //   W<0 , obs is substitued
     const std::vector<cMes1GCP> &      aVMesGCP = aSet->MesGCP();
     const std::vector<cMultipleImPt> & aVMesIm  = aSet->MesImOfPt() ;
     const std::vector<cSensorImage*> & aVSens   = aSet->VSens() ;

    // StdOut() << "GCP " << aVMesGCP.size() << " " << aVMesIm.size() << " " << aVSens.size() << "\n";

     size_t aNbGCP = aVMesGCP.size();

     tREAL8 aSigmaGCP = aVW[0];
     bool  aGcpUk = (aSigmaGCP>0);  // are GCP unknowns
     bool  aGcpFix = (aSigmaGCP==0);  // is GCP just an obervation
     tREAL8 aWeightGround =   aGcpFix ? 1.0 : (1.0/Square(aSigmaGCP)) ;  // standard formula, avoid 1/0
     tREAL8 aWeightImage =   (1.0/Square(aVW[1])) ;  // standard formula

     if (1)
     {
        StdOut() << "  Res;  Gcp0: " << aSet->AvgSqResidual() ;
        if (aGcpUk)
        {
            mNewGCP = *aSet;
	    for (size_t aK=0 ; aK< aNbGCP ; aK++)
                mNewGCP.MesGCP()[aK].mPt = mGCP_UK[aK]->Pt();
            StdOut() << "  GcpNew: " << mNewGCP.AvgSqResidual() ;
        }
        StdOut() << "\n";
     }

    // MMVII_INTERNAL_ASSERT_tiny(!aGcpUk,"Dont handle GCP UK 4 Now");

     //  Three temporary unknowns for x-y-z of the 3d point
     std::vector<int> aVIndGround = {-1,-2,-3};
     std::vector<int> aVIndFix = (aGcpFix ? aVIndGround : std::vector<int>());

     //  Parse all GCP
     for (size_t aKp=0 ; aKp < aNbGCP ; aKp++)
     {
           const cPt3dr & aPGr = aVMesGCP.at(aKp).mPt;
           cPt3dr_UK * aPtrGcpUk =  aGcpUk ? mGCP_UK[aKp] : nullptr;
	   cSetIORSNL_SameTmp<tREAL8>  aStrSubst(aPGr.ToStdVector(),aVIndFix);

	   const std::vector<cPt2dr> & aVPIm  = aVMesIm.at(aKp).VMeasures();
	   const std::vector<int> &  aVIndIm  = aVMesIm.at(aKp).VImages();

	   // Parse all image having a measure with this GCP
           for (size_t aKIm=0 ; aKIm<aVPIm.size() ; aKIm++)
           {
               int aIndIm = aVIndIm.at(aKIm);
               cSensorImage * aSens = aVSens.at(aIndIm);
               const cPt2dr & aPIm = aVPIm.at(aKIm);

	       // compute indexe of unknown, if GCp are !UK we have fix index for temporary
               std::vector<int> aVIndGlob = aGcpUk ? (std::vector<int>()) : aVIndGround;
               if (aGcpUk)  // if GCP are UK, we have to fill with its index
               {
                    aPtrGcpUk->PushIndexes(aVIndGlob);
               }
	       //  Add index of sensor (Pose+Calib for PC Cam)
               for (auto & anObj : aSens->GetAllUK())
                  anObj->PushIndexes(aVIndGlob);

	       // Do something only if GCP is visible 
               if (aSens->IsVisibleOnImFrame(aPIm) && (aSens->IsVisible(aPGr)))
               {
	             cCalculator<double> * anEqColin =  mVEqCol.at(aIndIm);
                     // the "obs" are made of 2 point and, possibily, current rotation (for PC cams)
                     std::vector<double> aVObs = aPIm.ToStdVector();
		     aSens->PushOwnObsColinearity(aVObs);

		     if (aGcpUk)  // Case Uknown, we just add the equation
		     {
		        mSys->R_CalcAndAddObs(anEqColin,aVIndGlob,aVObs,aWeightImage);
		     }
		     else  // Case to subst by schur compl,we accumulate in aStrSubst
		     {
                        mSys->R_AddEq2Subst(aStrSubst,anEqColin,aVIndGlob,aVObs,aWeightImage);
		     }
               }
	    }

	    if (! aGcpUk) // case  subst we now can make schurr commpl and subst
	    {
                if (! aGcpFix)  // if GCP is not hard fix, we must add obs on ground
		{
                    for (auto & aIndGr : aVIndGround)
                        aStrSubst.AddFixCurVarTmp(aIndGr,aWeightGround);
		}
                mSys->R_AddObsWithTmpUK(aStrSubst);  // finnaly add obs accummulated
	    }
	    else
	    {
                 //  Add observation fixing GCP
                 mR8_Sys->AddEqFixCurVar(*aPtrGcpUk,aPtrGcpUk->Pt(),aWeightGround);
	    }
    }
}


void cMMVII_BundleAdj::OneItere_GCP()
{
	OneItere_OnePackGCP(mMesGCP,mWeightGCP);
}

    /* ---------------------------------------- */
    /*            cPt3dr_UK                     */
    /* ---------------------------------------- */

cPt3dr_UK::cPt3dr_UK(const cPt3dr & aPt) :
    mPt  (aPt)
{
}

void cPt3dr_UK::PutUknowsInSetInterval() 
{
    mSetInterv->AddOneInterv(mPt);
}

const cPt3dr & cPt3dr_UK::Pt() const {return mPt;}

cPt3dr_UK::~cPt3dr_UK()
{
	Reset();
}

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

	std::string               mGCPDir;  ///  GCP Data Dir if != mDataDir
	std::vector<double>       mGCPW;
	int                       mNbIter;
};

cAppliBundlAdj::cAppliBundlAdj(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli  (aVArgs,aSpec),
   mDataDir      ("Std"),
   mPhProj       (*this),
   mBA           (&mPhProj),
   mNbIter       (10)
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
    
    return anArgOpt
               << AOpt2007(mDataDir,"DataDir","Defautl data directories ",{eTA2007::HDV})
	       << AOpt2007(mNbIter,"NbIter","Number of iterations",{eTA2007::HDV})

               << AOpt2007(mGCPDir,"GCPDir","Dir for GCP if != DataDir")
               << AOpt2007(mGCPW,"GCPW","Weithing of GCP if any [SigmaG,SigmaI], SG=0 fix, SG<0 schurr elim, SG>0",{{eTA2007::ISizeV,"[2,2]"}})
           ;
}

int cAppliBundlAdj::Exe()
{
    bool  MeasureAdded = false; 

    SetIfNotInit(mGCPDir,mDataDir);
    mPhProj.DPPointsMeasures().SetDirIn(mGCPDir);

    mPhProj.FinishInit();

    for (const auto &  aNameIm : VectMainSet(0))
    {
         mBA.AddCam(aNameIm);
    }

    if (IsInit(&mGCPW))
    {
        MeasureAdded = true;
        cSetMesImGCP * aFullMesGCP = new cSetMesImGCP;
	mPhProj.LoadGCP(*aFullMesGCP);

        for (const auto  & aSens : mBA.VSIm())
        {
             mPhProj.LoadIm(*aFullMesGCP,*aSens);
        }
	cSetMesImGCP * aMesGCP = aFullMesGCP->FilterNonEmptyMeasure();
	delete aFullMesGCP;

	mBA.AddGCP(mGCPW,aMesGCP);
    }

    MMVII_INTERNAL_ASSERT_tiny(MeasureAdded,"Not any measure added");

    for (int aKIter=0 ; aKIter<mNbIter ; aKIter++)
    {
        mBA.OneIteration();
    }

    for (auto & aCamPC : mBA.VSCPC())
	mPhProj.SaveCamPC(*aCamPC);


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

