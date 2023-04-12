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


	  void  AddCalib(cPerspCamIntrCalib *);  /// check if not exist and add
	  void  AddCamPC(cSensorCamPC *);

     private :

	  //============== Methods =============================
          cMMVII_BundleAdj(const cMMVII_BundleAdj &) = delete;
          void AssertPhaseAdd() ;
          void AssertPhp() ;
          void AssertPhpAndPhaseAdd() ;


	  //============== Data =============================
          cPhotogrammetricProject * mPhProj;
	  cSetInterUK_MultipeObj<tREAL8>  mSetUK;  ///< set of unknowns 


	  bool  mPhaseAdd;  ///< check that we dont mix add & use of unknowns

	  // ===================  Object to be adjusted ==================
	 
	  std::vector<cPerspCamIntrCalib *>  mVPCIC;     ///< vector of all internal calibration 4 easy parse
	  // std::set<cPerspCamIntrCalib *>     mSetPCIC;   ///< Internal calib a set to avoid multipl add

	  std::vector<cSensorCamPC *>        mSCPC;      ///< vector of perspectiv  cameras
	  std::vector<cSensorImage *>        mSIm;       ///< vector of sensor image (PC+RPC ...)


	  cSetInterUK_MultipeObj<tREAL8>    mSetIntervUK;

};

cMMVII_BundleAdj::cMMVII_BundleAdj(cPhotogrammetricProject * aPhp) :
    mPhProj    (aPhp),
    mPhaseAdd  (true)
{
}

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
    // MMVII_INTERNAL_ASSERT_tiny (!aCamPC->UkIsInit(),"Multiple add of cam : " + aCamPC->Name());
    {
    }


	/*
    AssertPhaseAdd();
    mSCPC.push_back(aCamPC);
    if (mSetPCIC.find(aCalib) == mSetPCIC.end())
    {
          mSetPCIC.insert(aCalib);
	  mVPCIC.push_back(aCalib);
	  mSetIntervUK.AddOneObj(aCalib);
    }
    */
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


   /* ********************************************************** */
   /*                                                            */
   /*                 cAppliBundlAdj                         */
   /*                                                            */
   /* ********************************************************** */

template <class Type>  class cAppliBundlAdj : public cMMVII_Appli
{
     public :
        cAppliBundlAdj(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;
     private :
	cPhotogrammetricProject  mPhProj;
};

template <class Type> cAppliBundlAdj<Type>::cAppliBundlAdj(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli(aVArgs,aSpec),
   mPhProj (*this)
{
}

template <class Type> cCollecSpecArg2007 & cAppliBundlAdj<Type>::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
    return anArgObl
	      <<  mPhProj.DPOrient().ArgDirOutMand()
           ;
}

template <class Type> cCollecSpecArg2007 & cAppliBundlAdj<Type>::ArgOpt(cCollecSpecArg2007 & anArgObl) 
{
    
    return anArgObl
           ;
}


template <class Type> int cAppliBundlAdj<Type>::Exe()
{
    mPhProj.FinishInit();

    return EXIT_SUCCESS;
}


tMMVII_UnikPApli Alloc_BundlAdj(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppliBundlAdj<tREAL8>(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_BundlAdj
(
     "BundleAdj",
      Alloc_BundlAdj,
      "Bundle adjusment",
      {eApF::Ori},
      {eApDT::Orient},
      {eApDT::Orient},
      __FILE__
);


}; // MMVII

