#ifndef  _MMVII_INSTRUMENTAL_BLOCK_
#define  _MMVII_INSTRUMENTAL_BLOCK_

#include "MMVII_Ptxd.h"
#include "MMVII_Geom3D.h"
#include "MMVII_PCSens.h"
#include "MMVII_Clino.h"



namespace MMVII
{
/** \file  MMVII_InstrumentalBlock.h
    \brief Declaration of all classes for block rigid
*/

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


class cIrbCal_Cam1;      // one cam in a calib-bloc
class cIrbCal_CamSet;    // set of cam in a calib-bloc
class cIrbCal_Clino1;    // one clino in a calib-bloc
class cIrbCal_ClinoSet;  // set of  clino in a calib-bloc
class cIrbCal_Block;     // calib bloc of rigid instrument

class cIrb_SigmaPoseRel;   // "helper" class for storing  sigmas of rel poses


class   cIrbComp_Cam1;     // one cam in a compute-bloc
class   cIrbComp_CamSet;   // set of cam in a compute-bloc
class   cIrbComp_TimeS;    // time-stamp for a compute bloc
class   cIrbComp_Block;    // compute bloc of rigid instrument


class cAppli_EditBlockInstr;    // appli of "edtiting" the bloc, "friend" of some classes
class cAppli_BlockInstrInitCam; // appli for computing initial value of poses in block


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


/// class for representing one camera embeded in a "Rigid Block of Instrument"
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

        /// Indicate if the  image belongs to the block
        bool ImageIsInBlock (const std::string & ) const;

        /**  modify the pose, separate from constructor because must be done in calib init, after block creation */
        void SetPose(const tPoseR & aPose);

    private :
        int           mNum;
        std::string   mNameCal;        ///< "full" name of calibration associated to, like  "CalibIntr_CamNIKON_D5600_Add043_Foc24000"
        std::string   mPatTimeStamp;   //< use to extract time stamp from a name
        bool          mSelIsPat;       ///< indicate if selector is pattern/file
        std::string   mImSelect;       ///< selector, indicate if an image belongs  to the block
        bool          mIsInit;         ///< was the pose in the block computed ?
        tPoseR        mPoseInBlock;    ///< Position in the block  +- boresight
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

class cIrb_SigmaPoseRel
{
    public :
           cIrb_SigmaPoseRel();
           cIrb_SigmaPoseRel(int aK1,int aK2,tREAL8 aSigmaTr,tREAL8 aSigmaRot);

	   int    mK1;
	   int    mK2;
	   tREAL8 mSigmaTr;
	   tREAL8 mSigmaRot;
};
void AddData(const  cAuxAr2007 & anAux,cIrb_SigmaPoseRel & aSigmaPR);


///  class for representing the set of cameras embedded in a bloc
class cIrbCal_CamSet : public cMemCheck
{
     public :
         friend cAppli_EditBlockInstr;
         friend cAppli_BlockInstrInitCam;

         cIrbCal_CamSet(); //< constructor, ok for serial

         void AddData(const  cAuxAr2007 & anAux); //< serialization
         cIrbCal_Cam1 * CamFromNameCalib(const std::string& aName,bool SVP=false);

         size_t  NbCams() const;    //< Number of cameras
         int     NumMaster() const; //< Accessor
         cIrbCal_Cam1 & KthCam(size_t aK);
         const cIrbCal_Cam1 & KthCam(size_t aK) const;
     private :
         void SetSigma(const cIrb_SigmaPoseRel&) ;       //< reset the camea
         void AddCam
              (
                   const std::string &  aNameCalib,
                   const std::string&   aPatTimeStamp,  
                   const std::string &  aPatImSel,
                   bool OkAlreadyExist =false
              );

         int                             mNumMaster;      //< num of "master" image
         std::vector<cIrbCal_Cam1>       mVCams;          //< set of cameras
         std::vector<cIrb_SigmaPoseRel>  mVSigmas;        //< sigmas of pairs
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

         std::vector<cIrbCal_Clino1> mVClinos; //< set of clinos
};


///  class for representing  the structure/calibration of instruments possibly used 
class cIrbCal_Block : public cMemCheck
{
     public :
        friend cIrbComp_Block;

        static const std::string  theDefaultName;  /// in most application there is only one block
        cIrbCal_Block(const std::string& aName=theDefaultName);
        void AddData(const  cAuxAr2007 & anAux);

        cIrbCal_CamSet &         SetCams() ;            //< Accessors
        const cIrbCal_CamSet &   SetCams() const ;      //< Accessors
        cIrbCal_ClinoSet &       SetClinos() ;          //< Accessors
        const std::string &       NameBloc() const;     //< Accessor
     private :
        std::string                 mNameBloc;   //<  Name of the bloc
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
         const cIrbComp_CamSet & SetCams() const;  //< Accessor
	 const cIrbComp_Block & CompBlock() const; //< Accessor
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
       size_t  NbCams() const ;                         //< Accessor of Accessor of ...
       const cIrbCal_Block & CalBlock() const ; //< Accessor
       cIrbCal_Block & CalBlock()  ; //< Accessor

       std::pair<tPoseR,cIrb_SigmaPoseRel> ComputeCalibCamsInit(int aK1,int aK2) const;
    private :
       /// non copiable, too "dangerous"
       cIrbComp_Block(const cIrbComp_Block & ) = delete;
       const cPhotogrammetricProject &     PhProj();  //< Accessor, test !=0

       /**  return the data for time stamps (cams, clino ...)  corresponding to TS, possibly init it*/
       cIrbComp_TimeS &  DataOfTimeS(const std::string & aTS);

       cIrbCal_Block                         mCalBlock;
       const cPhotogrammetricProject *       mPhProj;
       std::map<std::string,cIrbComp_TimeS>  mDataTS;
};


};

#endif  //  _MMVII_INSTRUMENTAL_BLOCK_

