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

class cOneCamInRBoI;
class cCamsInRBoI;
class cClinoInRBoI;
class cRigidBlockOfInstrument;


/// class for representing a camera embeded in a "Rigid Block of Instrument"
class cOneCamInRBoI
{
    public :
        cOneCamInRBoI();  /// required for serialisation 
        cOneCamInRBoI(const std::string & aNameCal);
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
void AddData(const  cAuxAr2007 & anAux,cOneCamInRBoI & aCam);

class cCamsInRBoI
{
     public :
         cCamsInRBoI();
         cCamsInRBoI(const std::string & aPatTimeStamp);
         void AddCam(const std::string & aNameCalib,bool SVP=false);
	 void AddData(const  cAuxAr2007 & anAux);
     private :
         cOneCamInRBoI * CamFromName(const std::string& aName);

         std::string                 mPatTimeStamp;
	 std::vector<cOneCamInRBoI>  mCams;
};
void AddData(const  cAuxAr2007 & anAux,cCamsInRBoI & aCam);

class cRigidBlockOfInstrument
{
     public :
	static const std::string  theDefaultName;  /// in most application there is only one block


        cRigidBlockOfInstrument(const std::string& aName);
	void AddData(const  cAuxAr2007 & anAux);
        void AddCam(const std::string & aNameCalib,bool SVP=false);

	const std::string & NameBloc() const; //< Accessor 
     private :
	cOneCamInRBoI * CamFromName(const std::string&);
	std::string              mNameBloc;
// 	cCamsInRBoI              mCams;
        std::vector<cOneCamInRBoI>  mCams;

};

/* *************************************************************** */
/*                                                                 */
/*                        cOneCamInRBoI                            */
/*                                                                 */
/* *************************************************************** */

cOneCamInRBoI::cOneCamInRBoI(const std::string & aNameCal) :
     mNameCal       (aNameCal),
     mIsInit        (false),
     mPoseInBlock   (tPoseR::Identity()),
     mSigmaC        (-1),
     mSigmaR        (-1),
     mSelIsPat      (true),
     mImSelect      (".*")
{
}

cOneCamInRBoI::cOneCamInRBoI()  :
    cOneCamInRBoI(MMVII_NONE)
{
}

const std::string & cOneCamInRBoI::NameCal() const { return mNameCal; }

void cOneCamInRBoI::AddData(const  cAuxAr2007 & anAux)
{
      MMVII::AddData(cAuxAr2007("NameCalib",anAux),mNameCal);
      MMVII::AddData(cAuxAr2007("IsInit",anAux),mIsInit);
      MMVII::AddData(cAuxAr2007("Pose",anAux),mPoseInBlock);
      MMVII::AddData(cAuxAr2007("SigmaC",anAux),mSigmaC);
      MMVII::AddData(cAuxAr2007("SigmaR",anAux),mSigmaR);
      MMVII::AddData(cAuxAr2007("SelIsPat",anAux),mSelIsPat);
      MMVII::AddData(cAuxAr2007("ImSelect",anAux),mImSelect);
}

void AddData(const  cAuxAr2007 & anAux,cOneCamInRBoI & aCam)
{
    aCam.AddData(anAux);
}

/* *************************************************************** */
/*                                                                 */
/*                        cCamsInRBoI                              */
/*                                                                 */
/* *************************************************************** */

cCamsInRBoI::cCamsInRBoI(const std::string & aPatTimeStamp) :
     mPatTimeStamp (aPatTimeStamp)
{
}

cCamsInRBoI::cCamsInRBoI() :
	cCamsInRBoI("")
{
}

void cCamsInRBoI::AddCam(const std::string & aNameCalib,bool SVP)
{
   if (CamFromName(aNameCalib))
   {
       MMVII_INTERNAL_ASSERT_strong(SVP,"cRigidBlockOfInstrument::AddCam, cal already exist for " + aNameCalib);
   }
   else
   {
      mCams.push_back(cOneCamInRBoI(aNameCalib));
   }
}

cOneCamInRBoI * cCamsInRBoI::CamFromName(const std::string& aName)
{
    for (auto&  aCam : mCams)
        if (aCam.NameCal() == aName)
           return & aCam;
    return nullptr;
}

void  cCamsInRBoI::AddData(const  cAuxAr2007 & anAux)
{
    MMVII::AddData(cAuxAr2007("PatTimeStamp",anAux),mPatTimeStamp);
    MMVII::StdContAddData(cAuxAr2007("Cams",anAux),mCams);
}
void AddData(const  cAuxAr2007 & anAux,cCamsInRBoI & aCams)
{
    aCams.AddData(anAux);
}

/* *************************************************************** */
/*                                                                 */
/*                        cRigidBlockOfInstrument                  */
/*                                                                 */
/* *************************************************************** */

const std::string  cRigidBlockOfInstrument::theDefaultName = "TheBlock";  /// in most application there is only one block

cRigidBlockOfInstrument::cRigidBlockOfInstrument(const std::string& aName) :
     mNameBloc (aName)
{
}


void cRigidBlockOfInstrument::AddCam(const std::string & aNameCalib,bool SVP)
{
   if (CamFromName(aNameCalib))
   {
       MMVII_INTERNAL_ASSERT_strong(SVP,"cRigidBlockOfInstrument::AddCam, cal already exist for " + aNameCalib);
   }
   else
   {
      mCams.push_back(cOneCamInRBoI(aNameCalib));
   }
}

cOneCamInRBoI * cRigidBlockOfInstrument::CamFromName(const std::string& aName)
{
    for (auto&  aCam : mCams)
        if (aCam.NameCal() == aName)
           return & aCam;
    return nullptr;
}

void  cRigidBlockOfInstrument::AddData(const  cAuxAr2007 & anAux)
{
     StdContAddData(cAuxAr2007("Cams",anAux),mCams);
}
void AddData(const  cAuxAr2007 & anAux,cRigidBlockOfInstrument & aRBoI)
{
    aRBoI.AddData(anAux);
}


const std::string & cRigidBlockOfInstrument::NameBloc() const {return mNameBloc;}

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
        std::string              mNameBloc;
        std::string              mPatIm4Cam;
};

cAppli_EditBlockInstr::cAppli_EditBlockInstr(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec) :
    cMMVII_Appli (aVArgs,aSpec),
    mPhProj      (*this),
    mNameBloc    (cRigidBlockOfInstrument::theDefaultName)
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
	return anArgOpt
            << AOpt2007(mNameBloc,"NameBloc","Set the name of the bloc ",{{eTA2007::HDV}})
            << AOpt2007(mPatIm4Cam,"PatIm4Cam","Pattern of images for adding their cam in the bloc")
            << mPhProj.DPBlockInstr().ArgDirOutOpt()
        ;
}


int cAppli_EditBlockInstr::Exe() 
{
    mPhProj.DPBlockInstr().SetDirOutInIfNotInit();
    mPhProj.FinishInit();

    cRigidBlockOfInstrument *  aBlock = mPhProj.ReadRigBoI(mNameBloc,SVP::Yes);



    if (IsInit(&mPatIm4Cam))
    {
        auto aVNameIm = ToVect(SetNameFromString(mPatIm4Cam,true));
        for (const auto & aNameIm : aVNameIm)
        {
            std::string aNameCal = mPhProj.StdNameCalibOfImage(aNameIm);
	    aBlock->AddCam(aNameCal,SVP::Yes);
        }
    }

    mPhProj.SaveRigBoI(*aBlock);

    delete aBlock;
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

/* *************************************************************** */
/*                                                                 */
/*               cPhotogrammetricProject                           */
/*                                                                 */
/* *************************************************************** */
// cRigidBlockOfInstrument  ReadRigBoI(const std::string &) const;

std::string   cPhotogrammetricProject::NameRigBoI(const std::string & aName,bool isIn) const
{
    return DPBlockInstr().FullDirInOut(isIn) + aName + "." + GlobTaggedNameDefSerial();
}

cRigidBlockOfInstrument *  cPhotogrammetricProject::ReadRigBoI(const std::string & aName,bool SVP) const
{
    std::string aFullName  = NameRigBoI(aName,IO::In);
    cRigidBlockOfInstrument * aRes = new cRigidBlockOfInstrument(aName);

    if (! ExistFile(aFullName))  // if it doesnt exist and we are OK, it return a new empty bloc
    {
        MMVII_INTERNAL_ASSERT_User_UndefE(SVP,"cRigidBlockOfInstrument file dont exist");
    }
    else
    {
        ReadFromFile(*aRes,aFullName);
    }


    return aRes;
}

void   cPhotogrammetricProject::SaveRigBoI(const cRigidBlockOfInstrument & aBloc) const
{
      SaveInFile(aBloc,NameRigBoI(aBloc.NameBloc(),IO::Out));
}


};

