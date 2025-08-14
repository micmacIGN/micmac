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

class cOneCamInRBoI;            // on cam in a bloc
class cCamsInRBoI;              // set of cam in a bloc
class cOneClinoInRBoI;          // one clino in a bloc
class cClinosInRBoI;          // set of  clino in a bloc
class cRigidBlockOfInstrument;  //  bloc of rigid instrument
class cAppli_EditBlockInstr;    // appli of "edtiting" the bloc, "friend" of some classes


/// class for representing a camera embeded in a "Rigid Block of Instrument"
class cOneCamInRBoI
{
    public :
        cOneCamInRBoI();  //< required for serialisation 
        /// "real" constructor
        cOneCamInRBoI(const std::string & aNameCal,const std::string & aTimeStamp,const std::string & aPatImSel);
	const std::string & NameCal() const; //< Accessor
	void AddData(const  cAuxAr2007 & anAux); //< Serializer
    private :
        std::string   mNameCal;        ///< name of calibration associated to
        std::string   mPatTimeStamp;   //< use to extract time stamp from a name
        bool          mSelIsPat;       ///< indicate if selector is pattern/file
	std::string   mImSelect;       ///< selector, indicate if an image belongs  to the block
	bool          mIsInit;         ///< was the pose in the block computed ?
        tPoseR        mPoseInBlock;    ///< Position in the block  +- boresight
	tREAL8        mSigmaC;         ///< sigma on center
	tREAL8        mSigmaR;         ///< sigma on orientation
};
/// public interface to serialization
void AddData(const  cAuxAr2007 & anAux,cOneCamInRBoI & aCam);


///  class for representing one clino embeded in a "Rigid Block of Instrument""
class cOneClinoInRBoI
{
    public :
        cOneClinoInRBoI();  //< required for serialisation
        cOneClinoInRBoI(const std::string & aName); //< "Real" constructor
        const std::string & Name() const;  //< accessor 
        void AddData(const  cAuxAr2007 & anAux); //< serializer
    private :
        std::string   mName;           //< name of the clino
	bool          mIsInit;         //< was values computed ?
        tRotR         mOrientInBloc;    //< Position in the block
        tREAL8        mSigmaR;         //< sigma on orientation
};
void AddData(const  cAuxAr2007 & anAux,cOneClinoInRBoI & aClino);


///  class for representing the set of cameras embedded in a bloc
class cCamsInRBoI
{
     public :
         friend cAppli_EditBlockInstr;

         cCamsInRBoI();

	 void AddData(const  cAuxAr2007 & anAux);
     private :
         void AddCam
              (
                   const std::string & aNameCalib,
                   const std::string& aTimeStamp,
                   const std::string & aPatImSel,
                   bool SVP=false
              );
         cOneCamInRBoI * CamFromName(const std::string& aName);

	 std::vector<cOneCamInRBoI>  mVCams;          //< set of cameras
};
void AddData(const  cAuxAr2007 & anAux,cCamsInRBoI & aCam);

///  class for representing a set of clino
class cClinosInRBoI
{
     public :
         friend cAppli_EditBlockInstr;
         cClinosInRBoI();

	 void AddData(const  cAuxAr2007 & anAux);
     private :
         std::vector<cOneClinoInRBoI> mClinos;
   
};


class cRigidBlockOfInstrument
{
     public :
	static const std::string  theDefaultName;  /// in most application there is only one block
        cRigidBlockOfInstrument(const std::string& aName);
	void AddData(const  cAuxAr2007 & anAux);

	cCamsInRBoI &  SetCams() ;            //< Accessors
	const std::string & NameBloc() const; //< Accessor 
     private :
	std::string              mNameBloc;
 	cCamsInRBoI              mSetCams;

};
void AddData(const  cAuxAr2007 & anAux,cRigidBlockOfInstrument & aRBoI);

/* *************************************************************** */
/*                                                                 */
/*                        cOneCamInRBoI                            */
/*                                                                 */
/* *************************************************************** */

cOneCamInRBoI::cOneCamInRBoI(const std::string & aNameCal,const std::string & aTimeStamp,const std::string & aPatImSel) :
     mNameCal       (aNameCal),
     mPatTimeStamp  (aTimeStamp),
     mSelIsPat      (true),
     mImSelect      (aPatImSel),
     mIsInit        (false),
     mPoseInBlock   (tPoseR::Identity()),
     mSigmaC        (-1),
     mSigmaR        (-1)
{
}

cOneCamInRBoI::cOneCamInRBoI()  :
    cOneCamInRBoI(MMVII_NONE,MMVII_NONE,MMVII_NONE)
{
}

const std::string & cOneCamInRBoI::NameCal() const { return mNameCal; }

void cOneCamInRBoI::AddData(const  cAuxAr2007 & anAux)
{
      MMVII::AddData(cAuxAr2007("NameCalib",anAux),mNameCal);
      MMVII::AddData(cAuxAr2007("PatTimeStamp",anAux),mPatTimeStamp);
      MMVII::AddData(cAuxAr2007("SelIsPat",anAux),mSelIsPat);
      MMVII::AddData(cAuxAr2007("ImSelect",anAux),mImSelect);
      MMVII::AddData(cAuxAr2007("IsInit",anAux),mIsInit);
      MMVII::AddData(cAuxAr2007("Pose",anAux),mPoseInBlock);
      MMVII::AddData(cAuxAr2007("SigmaC",anAux),mSigmaC);
      MMVII::AddData(cAuxAr2007("SigmaR",anAux),mSigmaR);
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

cCamsInRBoI::cCamsInRBoI() 
{
}

void cCamsInRBoI::AddCam
     (
         const std::string & aNameCalib,
	 const std::string & aTimeStamp,
	 const std::string & aPatImSel,
	 bool SVP
     )
{
   cOneCamInRBoI * aCam = CamFromName(aNameCalib);
   cOneCamInRBoI aNewCam (aNameCalib,aTimeStamp,aPatImSel);
   if (aCam)
   {
       MMVII_INTERNAL_ASSERT_strong(SVP,"cRigidBlockOfInstrument::AddCam, cal already exist for " + aNameCalib);
       *aCam = aNewCam;
   }
   else
   {
      mVCams.push_back(aNewCam);
   }
}

cOneCamInRBoI * cCamsInRBoI::CamFromName(const std::string& aName)
{
    for (auto&  aCam : mVCams)
        if (aCam.NameCal() == aName)
           return & aCam;
    return nullptr;
}

void  cCamsInRBoI::AddData(const  cAuxAr2007 & anAux)
{
    MMVII::StdContAddData(cAuxAr2007("Cams",anAux),mVCams);
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


void  cRigidBlockOfInstrument::AddData(const  cAuxAr2007 & anAux)
{
    MMVII::AddData(cAuxAr2007("Cams",anAux),mSetCams);	
}
void AddData(const  cAuxAr2007 & anAux,cRigidBlockOfInstrument & aRBoI)
{
    aRBoI.AddData(anAux);
}


const std::string & cRigidBlockOfInstrument::NameBloc() const {return mNameBloc;}
cCamsInRBoI &  cRigidBlockOfInstrument::SetCams() {return mSetCams;}

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
        cPhotogrammetricProject   mPhProj;
        std::string               mNameBloc;
	std::vector<std::string>  mVPatsIm4Cam;
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
            << AOpt2007(mVPatsIm4Cam,"PatsIm4Cam","Pattern images []",{{eTA2007::ISizeV,"[1,3]"}})
            << mPhProj.DPBlockInstr().ArgDirOutOpt()
        ;
}


int cAppli_EditBlockInstr::Exe() 
{
    mPhProj.DPBlockInstr().SetDirOutInIfNotInit();
    mPhProj.FinishInit();

    cRigidBlockOfInstrument *  aBlock = mPhProj.ReadRigBoI(mNameBloc,SVP::Yes);



    if (IsInit(&mVPatsIm4Cam))
    {
        std::string aPatSelOnDisk = mVPatsIm4Cam.at(0);
        std::string aPatTimeStamp = GetDef(mVPatsIm4Cam,1,aPatSelOnDisk);
        std::string aPatSelIm = GetDef(mVPatsIm4Cam,2,aPatTimeStamp);

        auto aVNameIm = ToVect(SetNameFromString(aPatSelOnDisk,true));
        for (const auto & aNameIm : aVNameIm)
        {
            std::string aNameCal = mPhProj.StdNameCalibOfImage(aNameIm);
	    aBlock->SetCams().AddCam(aNameCal,aPatTimeStamp,aPatSelIm,SVP::Yes);
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

