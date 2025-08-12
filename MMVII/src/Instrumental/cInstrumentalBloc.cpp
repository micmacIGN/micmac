#include "MMVII_Ptxd.h"
#include "cMMVII_Appli.h"
#include "MMVII_Geom3D.h"
#include "MMVII_PCSens.h"
#include "MMVII_2Include_Serial_Tpl.h"



/**
  \file cInstrumentalBloc.cpp


  \brief This file contains the core implemantation of Block of rigid instrument
 
*/

namespace MMVII
{
class cCamInRBoI;
class cRigidBlockOfInstrument;

/// class for representing a camera embeded in a "Rigid Block of Instrument"
class cCamInRBoI
{
    public :
        cCamInRBoI(const std::string & aNameCal);
	const std::string & NameCal() const;
	void AddData(const  cAuxAr2007 & anAux);
    private :
        std::string   mNameCal;        ///< name of calibration associated to
	bool          mIsInit;         ///< was the pose in the block computed ?
        tPoseR        mPoseInBlock;    ///< Position in the block
	tREAL8        mSigmaC;         ///< sigma on center
	tREAL8        mSigmaR;         ///< sigma on orientation
        bool          mSelIsPat;       ///< indicate if selector is pattern/file
	std::string   mImSelect;       ///< selector, indicate if an image belongs  to the block
};




class cRigidBlockOfInstrument
{
     public :
	static const std::string  theDefaultName;  /// in most application there is only one block

        void AddCam(const std::string & aNameCalib,bool SVP=false);

        cRigidBlockOfInstrument(const std::string& aName);
     private :
	cCamInRBoI * CamFromName(const std::string&);
        std::vector<cCamInRBoI>  mCams;
};


/* *************************************************************** */
/*                                                                 */
/*                        cCamInRBoI                               */
/*                                                                 */
/* *************************************************************** */

cCamInRBoI::cCamInRBoI(const std::string & aNameCal) :
     mNameCal       (aNameCal),
     mIsInit        (false),
     mPoseInBlock   (tPoseR::Identity()),
     mSigmaC        (-1),
     mSigmaR        (-1),
     mSelIsPat      (true),
     mImSelect      (".*")
{
}

const std::string & cCamInRBoI::NameCal() const { return mNameCal; }

void cCamInRBoI::AddData(const  cAuxAr2007 & anAux)
{
      MMVII::AddData(cAuxAr2007("NameCalib",anAux),mNameCal);
      MMVII::AddData(cAuxAr2007("IsInit",anAux),mIsInit);
      MMVII::AddData(cAuxAr2007("Pose",anAux),mPoseInBlock);
      MMVII::AddData(cAuxAr2007("SigmaC",anAux),mSigmaC);
      MMVII::AddData(cAuxAr2007("SigmaR",anAux),mSigmaR);
      MMVII::AddData(cAuxAr2007("SelIsPat",anAux),mSelIsPat);
      MMVII::AddData(cAuxAr2007("ImSelect",anAux),mImSelect);
}

void AddData(const  cAuxAr2007 & anAux,cCamInRBoI & aCam)
{
    aCam.AddData(anAux);
}


/* *************************************************************** */
/*                                                                 */
/*                        cRigidBlockOfInstrument                  */
/*                                                                 */
/* *************************************************************** */

const std::string  cRigidBlockOfInstrument::theDefaultName = "TheBlock";  /// in most application there is only one block

void cRigidBlockOfInstrument::AddCam(const std::string & aNameCalib,bool SVP)
{
   if (CamFromName(aNameCalib))
   {
       MMVII_INTERNAL_ASSERT_strong(SVP,"cRigidBlockOfInstrument::AddCam, cal already exist for " + aNameCalib);
   }
   else
   {
      mCams.push_back(cCamInRBoI(aNameCalib));
   }
}

cCamInRBoI * cRigidBlockOfInstrument::CamFromName(const std::string& aName)
{
    for (auto&  aCam : mCams)
        if (aCam.NameCal() == aName)
           return & aCam;
    return nullptr;
}

/* *************************************************************** */
/*                                                                 */
/*                        cRigidBlockOfInstrument                  */
/*                                                                 */
/* *************************************************************** */

class cAppli_EditBlockInstr : public cMMVII_Appli
{
     public :

        cAppli_EditBlockInstr(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli &);
        int Exe() override;
         cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override;
         cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override;

        // std::vector<std::string>  Samples() const override;

     private :
        cPhotogrammetricProject  mPhProj;
};

cAppli_EditBlockInstr::cAppli_EditBlockInstr(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec) :
    cMMVII_Appli (aVArgs,aSpec),
    mPhProj      (*this)
{
}

cCollecSpecArg2007 & cAppli_EditBlockInstr::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
     return anArgObl
             <<  mPhProj.DPBlockInstr().ArgDirInMand()
     ;

}

cCollecSpecArg2007 & cAppli_EditBlockInstr::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
	return anArgOpt;
}


int cAppli_EditBlockInstr::Exe() 
{
    mPhProj.DPBlockInstr().SetDirOutInIfNotInit();

    mPhProj.FinishInit();

    return EXIT_SUCCESS;
}


    /* ==================================================== */
    /*                                                      */
    /*               MMVII                                  */
    /*                                                      */
    /* ==================================================== */


tMMVII_UnikPApli Alloc_EditBlockInstr(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_EditBlockInstr(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_EditBlockInstr
(
     "EditBlockInstr",
      Alloc_EditBlockInstr,
      "Create/Edit a block of instruments",
      {eApF::Project},
      {eApDT::Xml},
      {eApDT::Xml},
      __FILE__
);



};

