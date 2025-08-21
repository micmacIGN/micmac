#include "MMVII_Ptxd.h"
#include "cMMVII_Appli.h"
#include "MMVII_Geom3D.h"
#include "MMVII_PCSens.h"
#include "MMVII_Clino.h"
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
class cClinosInRBoI;            // set of  clino in a bloc
class cRigidBlockOfInstrument;  //  bloc of rigid instrument
class cAppli_EditBlockInstr;    // appli of "edtiting" the bloc, "friend" of some classes
class   cComputedRBOI;          // RBOI for computation 
class   cOneTS_CRBOI;           // Data for one time stamp of RBOI


/// class for representing a camera embeded in a "Rigid Block of Instrument"
class cOneCamInRBoI : public cMemCheck
{
    public :
        cOneCamInRBoI();  //< required for serialisation 
        /// "real" constructor
        cOneCamInRBoI(const std::string & aNameCal,const std::string & aTimeStamp,const std::string & aPatImSel);
	const std::string & NameCal() const; //< Accessor
	void AddData(const  cAuxAr2007 & anAux); //< Serializer
    private :
        std::string   mNameCal;        ///< "full" name of calibration associated to, like  "CalibIntr_CamNIKON_D5600_Add043_Foc24000"
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
class cOneClinoInRBoI : public cMemCheck
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
class cCamsInRBoI : public cMemCheck
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
         cOneCamInRBoI * CamFromName(const std::string& aName,bool SVP=false);

	 std::vector<cOneCamInRBoI>  mVCams;          //< set of cameras
};
void AddData(const  cAuxAr2007 & anAux,cCamsInRBoI & aCam);

///  class for representing a set of clino
class cClinosInRBoI : public cMemCheck
{
     public :
         friend cAppli_EditBlockInstr;
         cClinosInRBoI();

	 void AddData(const  cAuxAr2007 & anAux);
     private :
         cOneClinoInRBoI * ClinoFromName(const std::string& aName);
         void AddClino(const std::string &,bool SVP=false);

         std::vector<cOneClinoInRBoI> mVClinos;
};


///  class for representing  the structure/calibration of instruments possibly used 
class cRigidBlockOfInstrument : public cMemCheck
{
     public :
	static const std::string  theDefaultName;  /// in most application there is only one block
        cRigidBlockOfInstrument(const std::string& aName=MMVII_NONE);
	void AddData(const  cAuxAr2007 & anAux);

	cCamsInRBoI &   SetCams() ;            //< Accessors
	cClinosInRBoI & SetClinos() ;            //< Accessors
	const std::string & NameBloc() const; //< Accessor 
     private :
	std::string              mNameBloc;   //<  Name of the bloc
 	cCamsInRBoI              mSetCams;    //<  Cameras used in the bloc
        cClinosInRBoI            mSetClinos;  //<  Clinos used in the bloc
};
void AddData(const  cAuxAr2007 & anAux,cRigidBlockOfInstrument & aRBoI);

///  class for storing one time stamp in cComputedRBOI

class   cOneTS_CRBOI : public cMemCheck
{
    public :
    private :
};

///  class for using a rigid bloc in computation (calibration/compensation)
class   cComputedRBOI : public cMemCheck
{
    public :
       cComputedRBOI(const cRigidBlockOfInstrument &) ;
       cComputedRBOI(const std::string & aNameFile);
       cComputedRBOI(const cPhotogrammetricProject& ,const std::string & aNameBloc);

       void AddCamera(const std::string & );

    private :
         cComputedRBOI(const cComputedRBOI & ) = delete;

         cOneTS_CRBOI &  DataOfTimeS();

         cRigidBlockOfInstrument             mRBOI;
         const cPhotogrammetricProject *     mPhProj;
         std::map<std::string,cOneTS_CRBOI>  mDataTS;
};

cComputedRBOI::cComputedRBOI(const cRigidBlockOfInstrument & aRBOI) :
   mRBOI   (aRBOI),
   mPhProj (nullptr)
{
}

cComputedRBOI::cComputedRBOI(const std::string & aNameFile) :
    cComputedRBOI(SimpleCopyObjectFromFile<cRigidBlockOfInstrument>(aNameFile))
{
}

cComputedRBOI::cComputedRBOI(const cPhotogrammetricProject& aPhProj,const std::string & aNameBloc) :
    cComputedRBOI  (aPhProj.NameRigBoI(aNameBloc,true))
{
    mPhProj   = &aPhProj;
}


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
   cOneCamInRBoI * aCam = CamFromName(aNameCalib,SVP::Yes);
   cOneCamInRBoI aNewCam (aNameCalib,aTimeStamp,aPatImSel);
   // in case already exist, we may ovewrite (multiple edit)
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

cOneCamInRBoI * cCamsInRBoI::CamFromName(const std::string& aName,bool SVP)
{
    for (auto&  aCam : mVCams)
        if (aCam.NameCal() == aName)
           return & aCam;
    MMVII_INTERNAL_ASSERT_strong(SVP,"Cannot get calib for camera " + aName);
    return nullptr;
}

void  cCamsInRBoI::AddData(const  cAuxAr2007 & anAux)
{
    MMVII::StdContAddData(cAuxAr2007("Set_Cams",anAux),mVCams);
}
void AddData(const  cAuxAr2007 & anAux,cCamsInRBoI & aCams)
{
    aCams.AddData(anAux);
}

/* *************************************************************** */
/*                                                                 */
/*                        cOneClinoInRBoI                          */
/*                                                                 */
/* *************************************************************** */

cOneClinoInRBoI::cOneClinoInRBoI(const std::string & aName) :
   mName         (aName),
   mIsInit       (false),
   mOrientInBloc (tRotR::Identity()),
   mSigmaR       (-1)
{
}

cOneClinoInRBoI::cOneClinoInRBoI() :
   cOneClinoInRBoI (MMVII_NONE)
{
}

void cOneClinoInRBoI::AddData(const  cAuxAr2007 & anAux)
{
      MMVII::AddData(cAuxAr2007("Name",anAux),mName);
      MMVII::AddData(cAuxAr2007("IsInit",anAux),mIsInit);
      MMVII::AddData(cAuxAr2007("OrientInBloc",anAux),mOrientInBloc);
      MMVII::AddData(cAuxAr2007("SigmaR",anAux),mSigmaR);
}
void AddData(const  cAuxAr2007 & anAux,cOneClinoInRBoI & aClino)
{
    aClino.AddData(anAux);
}

const std::string & cOneClinoInRBoI::Name() const {return mName;}


/* *************************************************************** */
/*                                                                 */
/*                        cClinosInRBoI                            */
/*                                                                 */
/* *************************************************************** */

cClinosInRBoI::cClinosInRBoI()
{
}

void cClinosInRBoI::AddData(const  cAuxAr2007 & anAux)
{
     MMVII::StdContAddData(cAuxAr2007("Set_Clinos",anAux),mVClinos);
}

void AddData(const  cAuxAr2007 & anAux,cClinosInRBoI & aSetClino)
{
    aSetClino.AddData(anAux);
}


cOneClinoInRBoI * cClinosInRBoI::ClinoFromName(const std::string& aName)
{
    for (auto&  aClino : mVClinos)
        if (aClino.Name() == aName)
           return & aClino;
    return nullptr;
}

void cClinosInRBoI::AddClino(const std::string & aName,bool SVP)
{
   cOneClinoInRBoI * aClino = ClinoFromName(aName);
   cOneClinoInRBoI aNewClino (aName);
   if (aClino)
   {
       MMVII_INTERNAL_ASSERT_strong(SVP,"cRigidBlockOfInstrument::AddClino, cal already exist for " + aName);
       *aClino = aNewClino;
   }
   else
   {
      mVClinos.push_back(aNewClino);
   }
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
    MMVII::AddData(cAuxAr2007("Clinos",anAux),mSetClinos);	
}
void AddData(const  cAuxAr2007 & anAux,cRigidBlockOfInstrument & aRBoI)
{
    aRBoI.AddData(anAux);
}


const std::string & cRigidBlockOfInstrument::NameBloc() const {return mNameBloc;}
cCamsInRBoI &  cRigidBlockOfInstrument::SetCams() {return mSetCams;}
cClinosInRBoI &  cRigidBlockOfInstrument::SetClinos() {return mSetClinos;}

/* *************************************************************** */
/*                                                                 */
/*                        cAppli_EditBlockInstr                    */
/*                                                                 */
/* *************************************************************** */

class cAppli_EditBlockInstr : public cMMVII_Appli
{
     public :

        cAppli_EditBlockInstr(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli &);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override;

        std::vector<std::string>  Samples() const ;

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


std::vector<std::string>  cAppli_EditBlockInstr::Samples() const 
{
   return 
   {
       "MMVII EditBlockInstr Bl0  PatsIm4Cam='[.*_(.*).tif]' InMeasureClino=MesClin_043",
       "MMVII EditBlockInstr Bl0  PatsIm4Cam='[.*tif,.*_(.*).tif,Fils-100.xml]' InMeasureClino=MesClin_043"
   };
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
            << AOpt2007(mVPatsIm4Cam,"PatsIm4Cam","Pattern images [PatSelOnDisk,PatTimeStamp?,PatSelInBlock?]",{{eTA2007::ISizeV,"[1,3]"}})
            << mPhProj.DPBlockInstr().ArgDirOutOpt()
            << mPhProj.DPMeasuresClino().ArgDirInOpt()
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
    if (mPhProj.DPMeasuresClino().DirInIsInit())
    {
         cSetMeasureClino aMesClin =  mPhProj.ReadMeasureClino();
         for (const auto & aName : aMesClin.NamesClino())
         {
             aBlock->SetClinos().AddClino(aName,SVP::Yes);
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
     "BlockInstrEdit",
      Alloc_EditBlockInstr,
      "Create/Edit a block of instruments",
      {eApF::BlockInstr},
      {eApDT::BlockInstr},
      {eApDT::BlockInstr},
      __FILE__
);

/* *************************************************************** */
/*                                                                 */
/*               cAppli_BlockInstrInitCam                          */
/*                                                                 */
/* *************************************************************** */

class cAppli_BlockInstrInitCam : public cMMVII_Appli
{
     public :

        cAppli_BlockInstrInitCam(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli &);
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override;
        int Exe() override;
        // std::vector<std::string>  Samples() const ;

     private :
        cPhotogrammetricProject   mPhProj;
        std::string               mSpecImIn;
        cComputedRBOI *           mCRBOI;
        std::string               mNameBloc;
};


cAppli_BlockInstrInitCam::cAppli_BlockInstrInitCam(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec) :
    cMMVII_Appli (aVArgs,aSpec),
    mPhProj      (*this),
    mCRBOI       (nullptr),
    mNameBloc    (cRigidBlockOfInstrument::theDefaultName)
{
}

cCollecSpecArg2007 & cAppli_BlockInstrInitCam::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
     return anArgObl
             <<  Arg2007(mSpecImIn,"Pattern/file for images", {{eTA2007::MPatFile,"0"},{eTA2007::FileDirProj}}  )
             <<  mPhProj.DPBlockInstr().ArgDirInMand()
     ;
}

cCollecSpecArg2007 & cAppli_BlockInstrInitCam::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
	return anArgOpt
            << AOpt2007(mNameBloc,"NameBloc","Name of bloc to calib ",{{eTA2007::HDV}})
        ;
}

int cAppli_BlockInstrInitCam::Exe()
{
    mPhProj.FinishInit();

    mCRBOI = new cComputedRBOI(mPhProj,mNameBloc);

    for (const auto & aNameIm :  VectMainSet(0))
    {
       StdOut() << " NameIm= " << aNameIm << "\n";
    }



    delete mCRBOI;
    return EXIT_SUCCESS;
}

    /* ==================================================== */
    /*                                                      */
    /*               MMVII                                  */
    /*                                                      */
    /* ==================================================== */


tMMVII_UnikPApli Alloc_BlockInstrInitCam(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_BlockInstrInitCam(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_BlockInstrInitCam
(
     "BlockInstrInitCam",
      Alloc_BlockInstrInitCam,
      "Init  camera poses inside a block of instrument",
      {eApF::BlockInstr,eApF::Ori},
      {eApDT::BlockInstr,eApDT::Ori},
      {eApDT::BlockInstr},
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

