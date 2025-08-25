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

/*
   As they are many "small" classes , with  repetitive "pattern" for they role, a systmatism naming has been adopted.
 There is two set of classes :
    - those used for storing the structure of the block of instrument and its calibration, their names begin by 
      "cIrbCal_"
    - those used for computation (initial calibration, adjustment), their names begin by "cIrbComp_"

   For "Cal" & "Comp" we have :
     - a class for storing the global block ("cIrbCal_Block" & "cIrbComp_Block") 
     - 2 classes for each type of instrument (Camera,Clino,GNSS ...). For example, for cameras we have
        *   cIrbCal_Cam1 and cIrbComp_Cam1 for representing a single camera in  Cal/Comp
        *   cIrbCal_CamSet  for set of cIrbCal_Cam1, cIrbComp_CamSet for set of cIrbComp_Cam1
     - identically for Clinometers  we have cIrbCal_ClinoSet, ... cIrbComp_Clino1
     - and the same for GNSS, IMU (to come later)

   For Comp we have one more class  "cIrbComp_TimeS" that contains all the data relative to one "TimeStamp". So:
     - cIrbCal_Block contains a cIrbCal_CamSet, cIrbCal_ClinoSet ....
     - cIrbComp_Block contains  set of "cIrbComp_TimeS", each "cIrbComp_TimeS" containing  a cIrbComp_CamSet ...
*/


class cIrbCal_Cam1;      // on cam in a calib-bloc
class cIrbCal_CamSet;    // set of cam in a calib-bloc
class cIrbCal_Clino1;    // one clino in a calib-bloc
class cIrbCal_ClinoSet;  // set of  clino in a calib-bloc
class cIrbCal_Block;     // calib bloc of rigid instrument



class   cIrbComp_Cam1;     // one came in a compute-bloc
class   cIrbComp_CamSet;   // set of cam in a compute-bloc
class   cIrbComp_TimeS;    // time-stamp for a compute bloc
class   cIrbComp_Block;    // compute bloc of rigid instrument


class cAppli_EditBlockInstr;    // appli of "edtiting" the bloc, "friend" of some classes


/* ************************************************************ */
/*                                                              */
/*        Classes for represnting calibration of IRB            */
/*                                                              */
/* ************************************************************ */

/*
   cIrbCal_Block  :
      - Name  of the bloc
      - cIrbCal_CamSet
         *   cIrbCal_Cam1  :
              - Name of intrinsic calibration 
              - Boresight Pose + sigma 
              - Function Name->Time stamp
      -  cIrbCal_ClinoSet
         *   cIrbCal_Clino1 
              - Boresight rotation + sigma
*/


/// class for representing a camera embeded in a "Rigid Block of Instrument"
class cIrbCal_Cam1 : public cMemCheck
{
    public :
        cIrbCal_Cam1();  //< required for serialisation 
        /// "real" constructor
        cIrbCal_Cam1(int aNum,const std::string & aNameCal,const std::string & aTimeStamp,const std::string & aPatImSel);
	const std::string & NameCal() const; //< Accessor
        int Num() const; //< Accesor
	void AddData(const  cAuxAr2007 & anAux); //< Serializer

        /// Compute the time stamp identifier associated to a name of image
        std::string  TimeStamp(const std::string & aNameImage ) const;

        /// Indicate if the 
        bool ImageIsInBlock (const std::string & ) const;

    private :
        int           mNum;
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
void AddData(const  cAuxAr2007 & anAux,cIrbCal_Cam1 & aCam);


///  class for representing one clino embeded in a "Rigid Block of Instrument""
class cIrbCal_Clino1 : public cMemCheck
{
    public :
        cIrbCal_Clino1();  //< required for serialisation
        cIrbCal_Clino1(const std::string & aName); //< "Real" constructor
        const std::string & Name() const;  //< accessor 
        void AddData(const  cAuxAr2007 & anAux); //< serializer
    private :
        std::string   mName;           //< name of the clino
	bool          mIsInit;         //< was values computed ?
        tRotR         mOrientInBloc;    //< Position in the block
        tREAL8        mSigmaR;         //< sigma on orientation
};
void AddData(const  cAuxAr2007 & anAux,cIrbCal_Clino1 & aClino);


///  class for representing the set of cameras embedded in a bloc
class cIrbCal_CamSet : public cMemCheck
{
     public :
         friend cAppli_EditBlockInstr;

         cIrbCal_CamSet();

	 void AddData(const  cAuxAr2007 & anAux);
         cIrbCal_Cam1 * CamFromName(const std::string& aName,bool SVP=false);

         size_t  NbCams() const;    //< Number of cameras
         int     NumMaster() const; //< Accessor
     private :
         void AddCam
              (
                   const std::string &  aNameCalib,
                   const std::string&   aPatTimeStamp,  
                   const std::string &  aPatImSel,
                   bool SVP=false
              );

         int                        mNumMaster;      //< num of "master" image
	 std::vector<cIrbCal_Cam1>  mVCams;          //< set of cameras
};
void AddData(const  cAuxAr2007 & anAux,cIrbCal_CamSet & aCam);

///  class for representing a set of clino
class cIrbCal_ClinoSet : public cMemCheck
{
     public :
         friend cAppli_EditBlockInstr;
         cIrbCal_ClinoSet();

	 void AddData(const  cAuxAr2007 & anAux);
     private :
         cIrbCal_Clino1 * ClinoFromName(const std::string& aName);
         void AddClino(const std::string &,bool SVP=false);

         std::vector<cIrbCal_Clino1> mVClinos;
};


///  class for representing  the structure/calibration of instruments possibly used 
class cIrbCal_Block : public cMemCheck
{
     public :
        friend cIrbComp_Block;

	static const std::string  theDefaultName;  /// in most application there is only one block
        cIrbCal_Block(const std::string& aName=MMVII_NONE);
	void AddData(const  cAuxAr2007 & anAux);

	cIrbCal_CamSet &         SetCams() ;            //< Accessors
	const cIrbCal_CamSet &   SetCams() const ;            //< Accessors
	cIrbCal_ClinoSet &       SetClinos() ;            //< Accessors
	const std::string &       NameBloc() const; //< Accessor 
     private :
	std::string              mNameBloc;   //<  Name of the bloc
 	cIrbCal_CamSet              mSetCams;    //<  Cameras used in the bloc
        cIrbCal_ClinoSet            mSetClinos;  //<  Clinos used in the bloc
};
void AddData(const  cAuxAr2007 & anAux,cIrbCal_Block & aRBoI);

/* ************************************************************ */
/*                                                              */
/*        Classes for computation with  IRB                     */
/*                                                              */
/* ************************************************************ */

///  class for storing poses in a cIrbComp_Cam1
class cIrbComp_Cam1 : public cMemCheck
{
     public :
         cIrbComp_Cam1();
         void Init(const tPoseR&,const std::string & aName);

         /// Compute Pose of CamB relatively to CamA=this
         tPoseR PosBInSysA(const cIrbComp_Cam1 & aCamB) const;

         bool IsInit() const ; //< Accessors 

     private :
         bool        mIsInit;
         tPoseR      mPoseInW;  /// C2W
         std::string mNameIm;
};

class cIrbComp_CamSet : public cMemCheck
{
    public :
         friend cIrbComp_Block;
         cIrbComp_CamSet(const cIrbComp_Block &);

          /// pose of K2th  relatively to K1th
          tPoseR PoseRel(size_t aK1,size_t aK2) const;
          bool   HasPoseRel(size_t aK1,size_t aK2) const;

    private :
         cIrbComp_CamSet(const cIrbComp_CamSet &) = delete;

         void AddImagePose(int anIndex,const tPoseR& aPose,const std::string & aNameIm);
         const cIrbComp_Block &      mBlock;
         std::vector<cIrbComp_Cam1>  mVCompPoses;
};

///  class for storing one time stamp in cIrbComp_Block
class   cIrbComp_TimeS : public cMemCheck
{
    public :
         friend cIrbComp_Block;

         cIrbComp_TimeS (const cIrbComp_Block &);
         const cIrbComp_CamSet & SetCams() const; //< Accessor
    private :
         cIrbComp_TimeS(const cIrbComp_TimeS&) = delete;
         const cIrbComp_Block &            mCompBlock;
         cIrbComp_CamSet                   mSetCams;
};

///  class for using a rigid bloc in computation (calibration/compensation)
//   cIrbComp_Block
class   cIrbComp_Block : public cMemCheck
{
    public :
       
       cIrbComp_Block(const cIrbCal_Block &) ;
       cIrbComp_Block(const std::string & aNameFile);
       cIrbComp_Block(const cPhotogrammetricProject& ,const std::string & aNameBloc);

       void AddImagePose(const std::string &,bool okImNotInBloc=false);
       void AddImagePose(const tPoseR&,const std::string &,bool okImNotInBloc=false);

       
       const cIrbCal_CamSet &  SetOfCalibCams() const ; //< Accessor of Accessor

  
        void ComputeCalibCamsInit(int aK1,int aK2) const;

    private :
       /// non copiable, too "dangerous"
       cIrbComp_Block(const cIrbComp_Block & ) = delete;
       const cPhotogrammetricProject &     PhProj();  //< Accessor, test !=0

       /**  return the data for time stamps (cams, clino ...)  corresponding to TS, possibly init it*/
       cIrbComp_TimeS &  DataOfTimeS(const std::string & aTS);

       cIrbCal_Block             mRBOI;
       const cPhotogrammetricProject *     mPhProj;
       std::map<std::string,cIrbComp_TimeS>  mDataTS;
};





/* *************************************************************** */
/*                                                                 */
/*                        cIrbComp_Cam1                           */
/*                                                                 */
/* *************************************************************** */

cIrbComp_Cam1::cIrbComp_Cam1() :
  mIsInit  (false),
  mPoseInW (tPoseR::RandomIsom3D(10)),
  mNameIm  ()
{
}

void cIrbComp_Cam1::Init(const tPoseR& aPoseInW,const std::string & aNameIm)
{
    if (mIsInit)
    {
        MMVII_INTERNAL_ERROR("Multiple init in cIrbComp_Cam1 for : " + aNameIm);
    }
    mIsInit    = true;
    mPoseInW   = aPoseInW;
    mNameIm    = aNameIm;
}

bool cIrbComp_Cam1::IsInit() const {return mIsInit;}

tPoseR cIrbComp_Cam1::PosBInSysA(const cIrbComp_Cam1 & aCamB) const
{
    MMVII_INTERNAL_ASSERT_tiny(mIsInit&&(aCamB.mIsInit),"cIrbComp_Cam1::PosBInSysA no init");
    //      (A->W) -1  * (B->W) 
    return mPoseInW.MapInverse() * aCamB.mPoseInW;
}


/* *************************************************************** */
/*                                                                 */
/*                        cIrbComp_CamSet                         */
/*                                                                 */
/* *************************************************************** */

cIrbComp_CamSet::cIrbComp_CamSet(const cIrbComp_Block & aCompBlock) :
    mBlock          (aCompBlock),
    mVCompPoses     (aCompBlock.SetOfCalibCams().NbCams())
{
}
void cIrbComp_CamSet::AddImagePose(int anIndex,const tPoseR& aPose,const std::string & aNameIm)
{
   mVCompPoses.at(anIndex).Init(aPose,aNameIm);
}

bool   cIrbComp_CamSet::HasPoseRel(size_t aK1,size_t aK2) const
{
    return mVCompPoses.at(aK1).IsInit() && mVCompPoses.at(aK2).IsInit() ;
}

tPoseR cIrbComp_CamSet::PoseRel(size_t aK1,size_t aK2) const
{
   return mVCompPoses.at(aK1).PosBInSysA(mVCompPoses.at(aK2));
}


/* *************************************************************** */
/*                                                                 */
/*                        cIrbComp_TimeS                             */
/*                                                                 */
/* *************************************************************** */

cIrbComp_TimeS::cIrbComp_TimeS (const cIrbComp_Block & aCompBlock) :
    mCompBlock (aCompBlock),
    mSetCams   (aCompBlock)
{
}

const cIrbComp_CamSet & cIrbComp_TimeS::SetCams() const {return mSetCams;}

/* *************************************************************** */
/*                                                                 */
/*                        cIrbComp_Block                            */
/*                                                                 */
/* *************************************************************** */


cIrbComp_Block::cIrbComp_Block(const cIrbCal_Block & aRBOI) :
   mRBOI   (aRBOI),
   mPhProj (nullptr)
{
}

cIrbComp_Block::cIrbComp_Block(const std::string & aNameFile) :
    cIrbComp_Block(SimpleCopyObjectFromFile<cIrbCal_Block>(aNameFile))
{
}

cIrbComp_Block::cIrbComp_Block(const cPhotogrammetricProject& aPhProj,const std::string & aNameBloc) :
    cIrbComp_Block  (aPhProj.NameRigBoI(aNameBloc,true))
{
    mPhProj   = &aPhProj;
}

const cIrbCal_CamSet &  cIrbComp_Block::SetOfCalibCams() const { return mRBOI.SetCams(); }

const cPhotogrammetricProject & cIrbComp_Block::PhProj()
{
    MMVII_INTERNAL_ASSERT_strong(mPhProj,"No PhProj for cIrbComp_Block");
    return *mPhProj;
}

cIrbComp_TimeS &  cIrbComp_Block::DataOfTimeS(const std::string & aTS)
{
    // possibly add an empty cIrbComp_TimeS if noting at aTS
    mDataTS.emplace(aTS,*this);

    // extract result mDataTS[aTS]  that should exist now
    auto  anIter = mDataTS.find(aTS);
    MMVII_INTERNAL_ASSERT_tiny(anIter!=mDataTS.end(),"cIrbComp_Block::DataOfTimeS");
    return anIter->second;
}

void cIrbComp_Block::AddImagePose(const tPoseR& aPose,const std::string & aNameIm,bool okImNotInBloc)
{
    // extract the name of the calibration 
    std::string aNameCal = PhProj().StdNameCalibOfImage(aNameIm);

    // extract the specification of the camera in the block
    cIrbCal_Cam1 *  aCInRBoI = mRBOI.mSetCams.CamFromName(aNameCal,okImNotInBloc);
    if (aCInRBoI==nullptr)
       return;

    // if the image does not belong to this block
    if (!aCInRBoI->ImageIsInBlock(aNameIm))
    {
        return;
    }

    // extract time stamp
    std::string aTimeS = aCInRBoI->TimeStamp(aNameIm);
    // cIrbComp_TimeS &  cIrbComp_Block::DataOfTimeS(const std::string & aTS)
    cIrbComp_TimeS &  aDataTS =  DataOfTimeS(aTimeS);

    // StdOut() << " III=" << aNameIm << " CCC=" << aNameCal << " Ptr=" << aTimeS << "\n";
    aDataTS.mSetCams.AddImagePose(aCInRBoI->Num(),aPose,aNameIm);
}

void cIrbComp_Block::AddImagePose(const std::string & aNameIm,bool  okImNotInBloc)
{
    bool hasPose;
    tPoseR aPose = PhProj().ReadPoseCamPC(aNameIm,&hasPose);
    if (hasPose)
    {
         AddImagePose(aPose,aNameIm,okImNotInBloc);
    }
}

void cIrbComp_Block::ComputeCalibCamsInit(int aKC1,int aKC2) const
{
   std::vector<tPoseR> aVPose;
   std::vector<std::string> aVTS;
   for (const auto & [aName,aDataTS] :  mDataTS)
   {
       const cIrbComp_CamSet & aSetC = aDataTS.SetCams();
        if (aSetC.HasPoseRel(aKC1,aKC2))
        {
            tPoseR aPose = aSetC.PoseRel(aKC1,aKC2);
            aVPose.push_back(aPose);
            aVTS.push_back(aName);
        }
   }
   std::vector<tREAL8>  aVDistTr;
   std::vector<tREAL8>  aVDistRot;
   for (size_t aKP1 =0 ; aKP1<aVPose.size() ; aKP1++)
   {
       for (size_t aKP2 =aKP1+1 ; aKP2<aVPose.size() ; aKP2++)
       {
            aVDistTr.push_back(Norm2(aVPose.at(aKP1).Tr()-aVPose.at(aKP2).Tr()));
            aVDistRot.push_back(aVPose.at(aKP1).Rot().Dist(aVPose.at(aKP2).Rot()));
       }
   }
   tREAL8 aMedTr  =  NonConstMediane(aVDistTr);
   tREAL8 aMedRot =  NonConstMediane(aVDistRot);
   
   StdOut() << "Med Tr=" << NonConstMediane(aVDistTr) << " Rot=" << NonConstMediane(aVDistRot) << "\n";

   int    aK1Min    = -1;
   tREAL8 aMinDGlob = 1e10;
   tREAL8 aMinDTr   = 1e10;
   tREAL8 aMinDRot  = 1e10;

   for (size_t aKP1 =0 ; aKP1<aVPose.size() ; aKP1++)
   {
       tREAL8 aSumDGlob = 0.0;
       tREAL8 aSumDTr   = 0.0;
       tREAL8 aSumDRot  = 0.0;
       for (size_t aKP2 =0 ; aKP2<aVPose.size() ; aKP2++)
       {
            tREAL8 aDTr = Norm2(aVPose.at(aKP1).Tr()-aVPose.at(aKP2).Tr());
            tREAL8 aDRot = aVPose.at(aKP1).Rot().Dist(aVPose.at(aKP2).Rot());
            aSumDTr   += aDTr;
            aSumDRot  += aDRot;
            aSumDGlob += aDTr + aDRot * (aMedTr/aMedRot);
       }
       // StdOut() << "SOM=" << aSumDGlob/aVPose.size() << "\n";
       if (aSumDGlob<aMinDGlob )
       {
           aK1Min    = aKP1;
           aMinDGlob = aSumDGlob ;
           aMinDTr   = aSumDTr   ;
           aMinDRot  = aSumDRot  ;
       }
   }
   aMinDGlob /= aVPose.size() ;
   aMinDTr   /= aVPose.size() ;
   aMinDRot  /= aVPose.size() ;
   StdOut() << " TS="  << aVTS.at(aK1Min) << " DTr=" <<  aMinDTr << " DRot=" << aMinDRot << "\n";
}

/* *************************************************************** */
/*                                                                 */
/*                        cIrbCal_Cam1                            */
/*                                                                 */
/* *************************************************************** */

cIrbCal_Cam1::cIrbCal_Cam1(int aNum,const std::string & aNameCal,const std::string & aTimeStamp,const std::string & aPatImSel) :
     mNum           (aNum),
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

cIrbCal_Cam1::cIrbCal_Cam1()  :
    cIrbCal_Cam1(-1,MMVII_NONE,MMVII_NONE,MMVII_NONE)
{
}

std::string  cIrbCal_Cam1::TimeStamp(const std::string & aNameImage) const
{
  return  ReplacePattern(mPatTimeStamp,"$1",aNameImage);
}

bool  cIrbCal_Cam1::ImageIsInBlock(const std::string & aNameImage) const
{
  return  MatchRegex(aNameImage,mImSelect);
}


const std::string & cIrbCal_Cam1::NameCal() const { return mNameCal; }
int cIrbCal_Cam1::Num() const {return mNum;}

void cIrbCal_Cam1::AddData(const  cAuxAr2007 & anAux)
{
      MMVII::AddData(cAuxAr2007("Num",anAux),mNum);
      MMVII::AddData(cAuxAr2007("NameCalib",anAux),mNameCal);
      MMVII::AddData(cAuxAr2007("PatTimeStamp",anAux),mPatTimeStamp);
      MMVII::AddData(cAuxAr2007("SelIsPat",anAux),mSelIsPat);
      MMVII::AddData(cAuxAr2007("ImSelect",anAux),mImSelect);
      MMVII::AddData(cAuxAr2007("IsInit",anAux),mIsInit);
      MMVII::AddData(cAuxAr2007("Pose",anAux),mPoseInBlock);
      MMVII::AddData(cAuxAr2007("SigmaC",anAux),mSigmaC);
      MMVII::AddData(cAuxAr2007("SigmaR",anAux),mSigmaR);
}

void AddData(const  cAuxAr2007 & anAux,cIrbCal_Cam1 & aCam)
{
    aCam.AddData(anAux);
}

/* *************************************************************** */
/*                                                                 */
/*                        cIrbCal_CamSet                           */
/*                                                                 */
/* *************************************************************** */

cIrbCal_CamSet::cIrbCal_CamSet()  :
    mNumMaster (0)
{
}

void  cIrbCal_CamSet::AddData(const  cAuxAr2007 & anAux)
{
    MMVII::AddData(cAuxAr2007("NumMaster",anAux),mNumMaster);
    MMVII::StdContAddData(cAuxAr2007("Set_Cams",anAux),mVCams);
    
    // check the coherence of num master
    MMVII_INTERNAL_ASSERT_strong
    (
         (mNumMaster>=0) && (mNumMaster<(int)mVCams.size()),
         "Bad num master for cIrbCal_CamSet"
    );
}
void AddData(const  cAuxAr2007 & anAux,cIrbCal_CamSet & aCams)
{
    aCams.AddData(anAux);
}


void cIrbCal_CamSet::AddCam
     (
         const std::string & aNameCalib,
	 const std::string & aTimeStamp,
	 const std::string & aPatImSel,
	 bool SVP
     )
{
   cIrbCal_Cam1 * aCam = CamFromName(aNameCalib,SVP::Yes);
   int aNum = aCam ?  aCam->Num() : int(mVCams.size()) ;
   cIrbCal_Cam1 aNewCam (aNum,aNameCalib,aTimeStamp,aPatImSel);
   // in case already exist, we may ovewrite (multiple edit)
   if (aCam)
   {
       MMVII_INTERNAL_ASSERT_strong(SVP,"cIrbCal_Block::AddCam, cal already exist for " + aNameCalib);
       *aCam = aNewCam;
   }
   else
   {
      mVCams.push_back(aNewCam);
   }
}

size_t  cIrbCal_CamSet::NbCams() const { return  mVCams.size();}
int     cIrbCal_CamSet::NumMaster() const {return mNumMaster;}


cIrbCal_Cam1 * cIrbCal_CamSet::CamFromName(const std::string& aName,bool SVP)
{
    for (auto&  aCam : mVCams)
        if (aCam.NameCal() == aName)
           return & aCam;
    MMVII_INTERNAL_ASSERT_strong(SVP,"Cannot get calib for camera " + aName);
    return nullptr;
}





/* *************************************************************** */
/*                                                                 */
/*                        cIrbCal_Clino1                          */
/*                                                                 */
/* *************************************************************** */

cIrbCal_Clino1::cIrbCal_Clino1(const std::string & aName) :
   mName         (aName),
   mIsInit       (false),
   mOrientInBloc (tRotR::Identity()),
   mSigmaR       (-1)
{
}

cIrbCal_Clino1::cIrbCal_Clino1() :
   cIrbCal_Clino1 (MMVII_NONE)
{
}

void cIrbCal_Clino1::AddData(const  cAuxAr2007 & anAux)
{
      MMVII::AddData(cAuxAr2007("Name",anAux),mName);
      MMVII::AddData(cAuxAr2007("IsInit",anAux),mIsInit);
      MMVII::AddData(cAuxAr2007("OrientInBloc",anAux),mOrientInBloc);
      MMVII::AddData(cAuxAr2007("SigmaR",anAux),mSigmaR);
}
void AddData(const  cAuxAr2007 & anAux,cIrbCal_Clino1 & aClino)
{
    aClino.AddData(anAux);
}

const std::string & cIrbCal_Clino1::Name() const {return mName;}


/* *************************************************************** */
/*                                                                 */
/*                        cIrbCal_ClinoSet                            */
/*                                                                 */
/* *************************************************************** */

cIrbCal_ClinoSet::cIrbCal_ClinoSet()
{
}

void cIrbCal_ClinoSet::AddData(const  cAuxAr2007 & anAux)
{
     MMVII::StdContAddData(cAuxAr2007("Set_Clinos",anAux),mVClinos);
}

void AddData(const  cAuxAr2007 & anAux,cIrbCal_ClinoSet & aSetClino)
{
    aSetClino.AddData(anAux);
}


cIrbCal_Clino1 * cIrbCal_ClinoSet::ClinoFromName(const std::string& aName)
{
    for (auto&  aClino : mVClinos)
        if (aClino.Name() == aName)
           return & aClino;
    return nullptr;
}

void cIrbCal_ClinoSet::AddClino(const std::string & aName,bool SVP)
{
   cIrbCal_Clino1 * aClino = ClinoFromName(aName);
   cIrbCal_Clino1 aNewClino (aName);
   if (aClino)
   {
       MMVII_INTERNAL_ASSERT_strong(SVP,"cIrbCal_Block::AddClino, cal already exist for " + aName);
       *aClino = aNewClino;
   }
   else
   {
      mVClinos.push_back(aNewClino);
   }
}

/* *************************************************************** */
/*                                                                 */
/*                        cIrbCal_Block                  */
/*                                                                 */
/* *************************************************************** */

const std::string  cIrbCal_Block::theDefaultName = "TheBlock";  /// in most application there is only one block

cIrbCal_Block::cIrbCal_Block(const std::string& aName) :
     mNameBloc (aName)
{
}


void  cIrbCal_Block::AddData(const  cAuxAr2007 & anAux)
{
    MMVII::AddData(cAuxAr2007("Cams",anAux),mSetCams);	
    MMVII::AddData(cAuxAr2007("Clinos",anAux),mSetClinos);	
}
void AddData(const  cAuxAr2007 & anAux,cIrbCal_Block & aRBoI)
{
    aRBoI.AddData(anAux);
}


const std::string &      cIrbCal_Block::NameBloc() const {return mNameBloc;}
cIrbCal_CamSet &        cIrbCal_Block::SetCams() {return mSetCams;}
const cIrbCal_CamSet &  cIrbCal_Block::SetCams() const {return mSetCams;}
cIrbCal_ClinoSet &      cIrbCal_Block::SetClinos() {return mSetClinos;}

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
    mNameBloc    (cIrbCal_Block::theDefaultName)
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

    cIrbCal_Block *  aBlock = mPhProj.ReadRigBoI(mNameBloc,SVP::Yes);

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
        cIrbComp_Block *           mCRBOI;
        std::string               mNameBloc;
};


cAppli_BlockInstrInitCam::cAppli_BlockInstrInitCam(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec) :
    cMMVII_Appli (aVArgs,aSpec),
    mPhProj      (*this),
    mCRBOI       (nullptr),
    mNameBloc    (cIrbCal_Block::theDefaultName)
{
}

cCollecSpecArg2007 & cAppli_BlockInstrInitCam::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
     return anArgObl
             <<  Arg2007(mSpecImIn,"Pattern/file for images", {{eTA2007::MPatFile,"0"},{eTA2007::FileDirProj}}  )
             <<  mPhProj.DPBlockInstr().ArgDirInMand()
             <<  mPhProj.DPOrient().ArgDirInMand()
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

    mCRBOI = new cIrbComp_Block(mPhProj,mNameBloc);

    for (const auto & aNameIm :  VectMainSet(0))
    {
       mCRBOI->AddImagePose(aNameIm);
       // mCRBOI->AddImagePose(aNameIm);
    }
    mCRBOI->ComputeCalibCamsInit(0,1);

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



// cIrbCal_Block  ReadRigBoI(const std::string &) const;

std::string   cPhotogrammetricProject::NameRigBoI(const std::string & aName,bool isIn) const
{
    return DPBlockInstr().FullDirInOut(isIn) + aName + "." + GlobTaggedNameDefSerial();
}

cIrbCal_Block *  cPhotogrammetricProject::ReadRigBoI(const std::string & aName,bool SVP) const
{
    std::string aFullName  = NameRigBoI(aName,IO::In);
    cIrbCal_Block * aRes = new cIrbCal_Block(aName);

    if (! ExistFile(aFullName))  // if it doesnt exist and we are OK, it return a new empty bloc
    {
        MMVII_INTERNAL_ASSERT_User_UndefE(SVP,"cIrbCal_Block file dont exist");
    }
    else
    {
        ReadFromFile(*aRes,aFullName);
    }


    return aRes;
}

void   cPhotogrammetricProject::SaveRigBoI(const cIrbCal_Block & aBloc) const
{
      SaveInFile(aBloc,NameRigBoI(aBloc.NameBloc(),IO::Out));
}


};

