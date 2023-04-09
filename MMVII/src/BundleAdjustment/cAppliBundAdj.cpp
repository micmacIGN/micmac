#include "MMVII_PCSens.h"
#include "MMVII_MMV1Compat.h"
#include "MMVII_DeclareCste.h"
#include "MMVII_BundleAdj.h"

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

};

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
	cPhotogrammetricProject  mPhProj;
};

cAppliBundlAdj::cAppliBundlAdj(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli(aVArgs,aSpec),
   mPhProj (*this)
{
}

cCollecSpecArg2007 & cAppliBundlAdj::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
    return anArgObl
	      <<  mPhProj.DPOrient().ArgDirOutMand()
           ;
}

cCollecSpecArg2007 & cAppliBundlAdj::ArgOpt(cCollecSpecArg2007 & anArgObl) 
{
    
    return anArgObl
           ;
}


int cAppliBundlAdj::Exe()
{
    mPhProj.FinishInit();

    return EXIT_SUCCESS;
}


tMMVII_UnikPApli Alloc_BundlAdj(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppliBundlAdj(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_BundlAdj
(
     "OriConvV1V2",
      Alloc_BundlAdj,
      "Convert orientation of MMV1  to MMVII",
      {eApF::Ori},
      {eApDT::Orient},
      {eApDT::Orient},
      __FILE__
);


}; // MMVII

