#include "MMVII_PCSens.h"
#include "MMVII_MMV1Compat.h"
#include "MMVII_DeclareCste.h"
#include "MMVII_BundleAdj.h"
#include <set>

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

template <class Type> class cMMVII_BundleAdj
{
     public :
          cMMVII_BundleAdj(cPhotogrammetricProject &);


	  /// check if not exist and add
	  void  AddCalib(cPerspCamIntrCalib *);  
	  void  AddCamPC(cSensorCamPC *);

     private :

          cMMVII_BundleAdj(const cMMVII_BundleAdj &) = delete;

	  cPhotogrammetricProject  & mPhProj;
	  cSetInterUK_MultipeObj<tREAL8>  mSetUK;  ///< set of unknowns 

          bool  InPhaseAdd() ;



	  // ===================  Object to be adjusted ==================
	 
	  std::vector<cPerspCamIntrCalib *>  mVPCIC;     ///< vector of all internal calibration 4 easy parse
	  std::set<cPerspCamIntrCalib *>     mSetPCIC;   ///< Internal calib a set to avoid multipl add

	  std::vector<cSensorCamPC *>        mSCPC;      ///< vector of perspectiv  cameras
	  std::vector<cSensorImage *>        mSIm;       ///< vector of sensor image (PC+RPC ...)


	  cSetInterUK_MultipeObj<tREAL8>    mSetIntervUK;

};

template <class Type> void cMMVII_BundleAdj<Type>::AddCalib(cPerspCamIntrCalib * aCalib)  
{
    if (mSetPCIC.find(aCalib) == mSetPCIC.end())
    {
          mSetPCIC.insert(aCalib);
	  mVPCIC.push_back(aCalib);
	  mSetIntervUK.mSetIntervUK(aCalib);
    }
}

template class cMMVII_BundleAdj<tREAL8>;
template class cMMVII_BundleAdj<tREAL16>;


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

