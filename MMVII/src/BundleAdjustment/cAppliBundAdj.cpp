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

class cMMVII_BundleAdj
{
     public :
          cMMVII_BundleAdj(cPhotogrammetricProject *);
          ~cMMVII_BundleAdj();

           // ======================== Add object ========================
	  void  AddCalib(cPerspCamIntrCalib *);  /// add  if not exist 
	  void  AddCamPC(cSensorCamPC *);  /// add, error id already exist
	  void  AddCam(const std::string & aNameIm);  /// add from name, require PhP exist
						      

	  //  =======  Add GCP, can be measure or measure & object
	  void AddGCP(const  std::vector<double>&, cSetMesImGCP *);
	  const std::vector<cSensorImage *> &  VSIm() const ;  ///< Accessor

	  void OneIteration();
     private :

	  //============== Methods =============================
          cMMVII_BundleAdj(const cMMVII_BundleAdj &) = delete;
          void AssertPhaseAdd() ;
          void AssertPhp() ;
          void AssertPhpAndPhaseAdd() ;
	  void InitIteration();
          void OneItere_GCP();



	  //============== Data =============================
          cPhotogrammetricProject * mPhProj;
	  cSetInterUK_MultipeObj<tREAL8>  mSetUK;  ///< set of unknowns 


	  bool  mPhaseAdd;  ///< check that we dont mix add & use of unknowns

	  cREAL8_RSNL *                 mSys;
	  cResolSysNonLinear<tREAL8> *  mR8_Sys;

	  // ===================  Object to be adjusted ==================
	 
	  std::vector<cPerspCamIntrCalib *>  mVPCIC;     ///< vector of all internal calibration 4 easy parse
	  std::vector<cSensorCamPC *>        mSCPC;      ///< vector of perspectiv  cameras
	  std::vector<cSensorImage *>        mVSIm;       ///< vector of sensor image (PC+RPC ...)

	  cSetInterUK_MultipeObj<tREAL8>    mSetIntervUK;


	  // ===================  Object to be adjusted ==================
	  cSetMesImGCP *       mMesGCP;
	  std::vector<double>  mWeightGCP;
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
    mR8_Sys = new cResolSysNonLinear<tREAL8>(eModeSSR::eSSR_LsqDense,mSetIntervUK.GetVUnKnowns());

    mSys = mR8_Sys;
}


void cMMVII_BundleAdj::OneIteration()
{
    if (mPhaseAdd)
    {
        InitIteration();
    }

    OneItere_GCP();
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
void cMMVII_BundleAdj::AddCamPC(cSensorCamPC * aCamPC)
{
    AssertPhaseAdd();
    MMVII_INTERNAL_ASSERT_tiny (!aCamPC->UkIsInit(),"Multiple add of cam : " + aCamPC->NameImage());

    mSetIntervUK.AddOneObj(aCamPC);
    mSCPC.push_back(aCamPC);
    mVSIm.push_back(aCamPC);

    AddCalib(aCamPC->InternalCalib());
}

void  cMMVII_BundleAdj::AddCam(const std::string & aNameIm)
{
    AssertPhpAndPhaseAdd();
    cSensorImage * aNewS = nullptr;

    // Try extract a PC Cam
    {
       cSensorCamPC * aCamPC = mPhProj->AllocCamPC(aNameIm,true,true);  // true 2 Delet,  true =SVP
       if (aCamPC)
       {
           aNewS = aCamPC;
           AddCamPC(aCamPC);
       }
    }

    // No camera succed
    if (aNewS== nullptr)
    {
       MMVII_UsersErrror(eTyUEr::eUnClassedError,"Cannot get a valid camera for image" +  aNameIm);
    }

    auto anEq = aNewS->EqColinearity(true,10,true);  // WithDer, SzBuf, ReUse
    StdOut() << "EQQQ= " << (void *) anEq << "\n";
    // cMMVII_Appli::AddObj2DelAtEnd(anEq);
}
const std::vector<cSensorImage *> &  cMMVII_BundleAdj::VSIm() const {return mVSIm;}


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

void cMMVII_BundleAdj::OneItere_GCP()
{
}


   /* ********************************************************** */
   /*                                                            */
   /*                 cAppliBundlAdj                         */
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
};

cAppliBundlAdj::cAppliBundlAdj(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli  (aVArgs,aSpec),
   mDataDir      ("Std"),
   mPhProj       (*this),
   mBA           (&mPhProj)
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

               << AOpt2007(mGCPDir,"GCPDir","Dir for GCP if != DataDir")
               << AOpt2007(mGCPW,"GCPW","Weithing of GCP if any [SigmaG,SigmaI], SG=0 fix, SG<0 schurr elim, SG>0",{{eTA2007::ISizeV,"[2,2]"}})
           ;
}

int cAppliBundlAdj::Exe()
{
    SetIfNotInit(mGCPDir,mDataDir);
    mPhProj.DPPointsMeasures().SetDirIn(mGCPDir);

    mPhProj.FinishInit();

    for (const auto &  aNameIm : VectMainSet(0))
    {
         mBA.AddCam(aNameIm);
    }

    if (IsInit(&mGCPW))
    {
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

    mBA.OneIteration();
    // mBA.OneIteration();
    // mBA.OneIteration();

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

