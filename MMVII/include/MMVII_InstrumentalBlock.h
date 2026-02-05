#ifndef  _MMVII_INSTRUMENTAL_BLOCK_
#define  _MMVII_INSTRUMENTAL_BLOCK_

#include "MMVII_Ptxd.h"
#include "MMVII_Geom3D.h"
#include "MMVII_PCSens.h"
#include "MMVII_Clino.h"
#include "MMVII_Matrix.h"
#include <tuple>
//#include <memory>


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

class cIrb_SigmaInstr;   // "helper" class for storing  sigmas of rel poses


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
      - Name  of the blocno
      - cIrbCal_CamSet
         *   cIrbCal_Cam1  :
              - Name of intrinsic calibration
              - Boresight Pose + sigma
              - Function Name->Time stamp
      -  cIrbCal_ClinoSet
         *   cIrbCal_Clino1
              - Boresight rotation + sigma
*/


  /*  ============   Classes for camera calibration ====================== */

/// class for representing one camera embeded in a "Rigid Block of Instrument"
class cIrbCal_Cam1  : public cMemCheck
{
    public :
        cIrbCal_Cam1();  //< required by serialisation
        ~cIrbCal_Cam1(); //< destructor do nothing sing use of shared_ptr
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

        tPoseR PoseInBlock() const; //< The pose contained in mPoseInBlock
        tPoseR PosBInSysA(const cIrbCal_Cam1 & aCamB) const;  //< Relative pose of B to A

        cPoseWithUK&  PoseUKInBlock(); //< Accessor
        bool          IsInit() const;  //< Was the pose initialzed (computed from Ptr on mPoseInBlock)
        void UnInit();  //< Set un-initialized

        /// return Intr Calib after eventually creating it
        cPerspCamIntrCalib *  IntrCalib(const cPhotogrammetricProject *);
        /// return Cam in bloc after eventually creating it
        cSensorCamPC *        CamInBloc(const cPhotogrammetricProject *);

    private :
       // cIrbCal_Cam1(const cIrbCal_Cam1&);
        int           mNum;
        std::string   mNameCal;        ///< "full" name of calibration associated to, like  "CalibIntr_CamNIKON_D5600_Add043_Foc24000"
        std::string   mPatTimeStamp;   ///< use to extract time stamp from a name
        bool          mSelIsPat;       ///< indicate if selector is pattern/file
        std::string   mImSelect;       ///< selector, indicate if an image belongs  to the block
        bool          mIsInit;         ///< was the pose in the block computed ?
        std::shared_ptr<cPoseWithUK>  mPoseInBlock;    ///< Position in the block  +- boresight
        cPerspCamIntrCalib *          mIntrCalib;
        cSensorCamPC *                mCamInBloc;
};

/// standard interface to serialization
void AddData(const  cAuxAr2007 & anAux,cIrbCal_Cam1 & aCam);



///  class for representing the set of cameras embedded in a bloc
class cIrbCal_CamSet  : public cMemCheck
{
     public :
         cIrbCal_CamSet(cIrbCal_Block* = nullptr); ///< default constructor, required by serial

         void AddData(const  cAuxAr2007 & anAux); ///< serialization
         /// Acces to pointer on an existing camera from its name, 0 if none and OkNone
         cIrbCal_Cam1 * CamFromNameCalib(const std::string& aName,bool OkNone=false);
         /// Acces to index of an existing camera from its name, 0 if none and OkNone
         int IndexCamFromNameCalib(const std::string& aName,bool OkNone=false);

         cSensorCamPC *     CamInBloc(const cPhotogrammetricProject *,const std::string & aNameIm);

         size_t  NbCams() const;                       ///< Number of cameras
         int     NumMaster() const;                    ///< Accessor
         void    SetNumMaster(int);                    ///< Modifier
         cIrbCal_Cam1 & KthCam(size_t aK);             ///< Accessor
         const cIrbCal_Cam1 & KthCam(size_t aK) const; ///< Accessor
         tPoseR PoseRel(size_t aK1,size_t aK2) const;   ///< Compute pose relative of K2 to K1
         std::vector<cIrbCal_Cam1> &      VCams();      ///< Accessor
         cIrbCal_Cam1 &                   MasterCam();  ///< Accessor inside vector

         void SetNumPoseInstr (const std::vector<int> & aVNums); ///< Modifier
         /// more than an accessor, as  it correct the special case [], [-1] ...
         std::vector<int>  NumPoseInstr() const;
         /// fails if several NumPoseInstr
         cIrbCal_Cam1 * SingleCamPoseInstr(bool OkNot1=false) ;
         void AddCam
              (
                   const std::string &  aNameCalib,
                   const std::string&   aPatTimeStamp,
                   const std::string &  aPatImSel,
                   bool OkAlreadyExist =false
              );
    private :

         int                             mNumMaster;      ///< num of "master" image
         /**  num of pose to use for estimate pose of intrument
          *   special cases :  [] -> all ,  [-1]  master */
         std::vector<int>                mNumsPoseInstr;
         std::vector<cIrbCal_Cam1>       mVCams;          ///< set of camerascIrbCal_Block
         cIrbCal_Block *                 mCalBlock;       ///< link to global calibration block
};
/// Standard interface to serialization
void AddData(const  cAuxAr2007 & anAux,cIrbCal_CamSet & aCam);


   /*  ============   Classes for clinometers calibration ====================== */

/**  class for representing one clino-calibration embeded in a "Rigid Block of Instrument";
 *   The calibration is made from  :
 *      # a normal vector indicating the direction of the clino
 *      # a polynomial correction
*/
class cIrbCal_Clino1   : public cMemCheck
{
    public :
        cIrbCal_Clino1();  ///< required for serialisation
        cIrbCal_Clino1(const std::string & aName); ///< "Real" constructor
        ~cIrbCal_Clino1(); ///< Destructor do nothin now, since destuctor

        const std::string & Name() const;  ///< accessor
        void AddData(const  cAuxAr2007 & anAux); ///< serializer

        void SetPNorm(const cPt3dr & aTr); ///< Modify value, eventually allocate

        cP3dNormWithUK&  CurPNorm();  ///< Accessor to  unknown 1-Norm point
        cVectorUK &      PolCorr();   ///< Accessor to polynom of correction
        bool          IsInit() const;  ///< Was it initiated ?
//        void UnInit();
    private :
        std::string                       mName;             ///< name of the clino
        bool                              mIsInit;           ///< was values computed ?
        std::shared_ptr<cP3dNormWithUK>   mTrInBlock;        ///< Orientation of the clino
        std::shared_ptr<cVectorUK>        mPolCorr;          ///< Polynom of correction of the angle
};
/// Standard interface to serialization
void AddData(const  cAuxAr2007 & anAux,cIrbCal_Clino1 & aClino);

///  class for representing a set of clino
class cIrbCal_ClinoSet  : public cMemCheck
{
     public :

         cIrbCal_ClinoSet(cIrbCal_Block* = nullptr);   ///< default required by serialization
         void AddData(const  cAuxAr2007 & anAux);   ///< Serialization
         std::vector<std::string> VNames() const;      /// Accessor to vector of names
         /// Accessor to number of clinometers
         size_t NbClino() const;
         /// Accesor to clino inside vector
         cIrbCal_Clino1 &  KthClino(int aK);
         /// Acces to pointer of an existing clinometers from its name, 0 if none and OkNone
         cIrbCal_Clino1 * ClinoFromName(const std::string& aName,bool OkNone=false);
         /// Acces to index of a clinometer from its name
         int  IndexClinoFromName(const std::string& aName,bool OkNone=false) const;
         /// Add a clinometer
         void AddClino(const std::string &,tREAL8 aSigma,bool SVP=false);
      private :

         std::vector<cIrbCal_Clino1>  mVClinos;    ///< set of clinos in the block
         cIrbCal_Block *              mCalBlock;  ///< link the global block

};
void AddData(const  cAuxAr2007 & anAux,cIrbCal_ClinoSet & aClino);


/// class for storing a relative orientation extern constraint ( for ex orthoganility of clino) between 2 instrument

class cIrb_CstrRelRot
{
   public :
      cIrb_CstrRelRot(const tRotR & aRot,const tREAL8 & aSigma);
      cIrb_CstrRelRot();
      void AddData(const  cAuxAr2007 & anAux);
   private :
      tRotR  mOri;
      tREAL8 mSigma;
};

void AddData(const  cAuxAr2007 & anAux,cIrb_CstrRelRot & aSigma);

/// class for storing orthogonal constraint (to generalize with a given angle)
class cIrb_CstrOrthog
{
   public :
      cIrb_CstrOrthog(const tREAL8 & aSigma);
      cIrb_CstrOrthog();
      void AddData(const  cAuxAr2007 & anAux);
      tREAL8 Sigma() const;
   private :
      tREAL8 mSigma;
};
void AddData(const  cAuxAr2007 & anAux,cIrb_CstrOrthog & aSigma);

/** Class for storing the sigma, of an instrument relatively to the block, or between them
 */
class cIrb_SigmaInstr
{
   public :
      cIrb_SigmaInstr();
      cIrb_SigmaInstr(tREAL8 aWTr,tREAL8 aWRot,tREAL8 aSigTr,tREAL8 aSigRot);
      void  AddNewSigma (const cIrb_SigmaInstr&);

      void AddData(const  cAuxAr2007 & anAux);


      tREAL8 SigmaTr() const;
      tREAL8 SigmaRot() const;

   private :
      tWArr mAvgSigTr;
      tWArr mAvgSigRot;
};
void AddData(const  cAuxAr2007 & anAux,cIrb_SigmaInstr & aSigma);


class cIrb_Desc1Intsr
{
    public :
       cIrb_Desc1Intsr();

       cIrb_Desc1Intsr (eTyInstr,const std::string & );

       void AddData(const  cAuxAr2007 & anAux);
       void  AddNewSigma (const cIrb_SigmaInstr&);
       void SetSigma(const cIrb_SigmaInstr&);
       void ResetSigma();

       const cIrb_SigmaInstr & Sigma() const;  //< Accessor
       eTyInstr             Type() const;      //< Accessor
       const std::string &  NameInstr() const; //< Accessor

    private :
       eTyInstr         mType;
       std::string      mNameInstr;
       cIrb_SigmaInstr  mSigma;
};
void AddData(const  cAuxAr2007 & anAux,cIrb_Desc1Intsr & aDesc);



///  class for representing  the structure/calibration of instruments possibly used
class cIrbCal_Block  : public cMemCheck
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

        void AddSigma(std::string aN1,eTyInstr aType1,std::string aN2, eTyInstr aType2, const cIrb_SigmaInstr &);

         const std::map<tNamePair,cIrb_SigmaInstr> & SigmaPair()  const;
         const std::map<std::string,cIrb_Desc1Intsr> & DescrIndiv() const;
         void SetSigmaPair( const  std::map<tNamePair,cIrb_SigmaInstr> & );
         void SetSigmaIndiv( const  std::map<tNamePair,cIrb_SigmaInstr> & );

         const cIrb_Desc1Intsr &  DescrIndiv(const std::string &) const;
          cIrb_Desc1Intsr &  NC_DescrIndiv(const std::string &) ;


         void AvgPairSigma(); //< Set all sigma of pairs to global average (in the same type)
         void AvgIndivSigma();  //< Set all sigma of object ir global average
         void AvgSigma();
         cIrb_Desc1Intsr &  AddSigma_Indiv(std::string aN1,eTyInstr aType1);

         void AddCstrRelRot(std::string aN1,std::string aN2,tREAL8 aSigma,tRotR aRot);
         void AddCstrRelOrthog(std::string aN1,std::string aN2,tREAL8 aSigma);

         const std::map<tNamePair,cIrb_CstrOrthog> &  CstrOrthog() const;

         void ShowDescr(eTyInstr) const;

     private :

         void AvgPairSigma (eTyInstr,eTyInstr); //< Avg sigma for pairs having the corresponding type
         void AvgIndivSigma(eTyInstr);


         cIrbCal_Block(const cIrbCal_Block&) = delete;


         void  AddSigma_Indiv(std::string aN1,eTyInstr aType1, const cIrb_SigmaInstr &);
         std::string                   mNameBloc;   //<  Name of the bloc
         cIrbCal_CamSet                mSetCams;    //<  Cameras used in the bloc
         cIrbCal_ClinoSet              mSetClinos;  //<  Clinos used in the bloc

         //  Sigma
         std::map<tNamePair,cIrb_SigmaInstr>   mSigmaPair;     //<  Sigmas between pair of instr
         std::map<std::string,cIrb_Desc1Intsr> mDescrIndiv;      //<  Sigmas of each instrument

         //  A priori external constraint
         std::map<tNamePair,cIrb_CstrRelRot>   mCstrRelRot;
         std::map<tNamePair,cIrb_CstrOrthog>   mCstrOrthog;



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
         ~cIrbComp_Cam1();

         void Init(cSensorCamPC *,bool Adopt);

         /// Compute Pose of CamB relatively to CamA=this
         tPoseR PosBInSysA(const cIrbComp_Cam1 & aCamB) const;

         bool IsInit() const ;  //< Is  mCamPC set ?
         cSensorCamPC * CamPC() const; //< Accessor
         tPoseR  Pose() const;  //< Accessor 2 mCamPC
         std::string NameIm() const; //<  Accessor 2 mCamPC


     private :

         cSensorCamPC * mCamPC;
         bool           mAdoptCam; // If the cam is adpoted it must be destroyed by this
};

class cIrbComp_CamSet  : public cMemCheck
{
    public :
         friend cIrbComp_Block;
         cIrbComp_CamSet(const cIrbComp_Block &);

          /// pose of K2th  relatively to K1th
          tPoseR PoseRel(size_t aK1,size_t aK2) const;

          // A relative pose can be estimated iff both poses are init
          bool   HasPoseRel(size_t aK1,size_t aK2) const;

          const std::vector<cIrbComp_Cam1> &  VCompPoses() const;  ///<  Accessor
          cIrbComp_Cam1 & KthCam(int aK) ;

          cSensorCamPC * SingleCamPoseInstr(bool OkNot1=false) const ;

    private :
         cIrbComp_CamSet(const cIrbComp_CamSet &) = delete;

         void AddImagePose(int anIndex,cSensorCamPC * aCamPC,bool Adopt);
         const cIrbComp_Block &      mBlock;
         std::vector<cIrbComp_Cam1>  mVCompPoses;
};


class cIrbComp_Clino1 : public cMemCheck
{
   public :
        cIrbComp_Clino1(tREAL8 anAngle);
        tREAL8 Angle() const;
   private :
        tREAL8 mAngle;
};

class cIrbComp_ClinoSet : public cMemCheck
{
   public :
      cIrbComp_ClinoSet();
      void SetClinoValues(const cOneMesureClino&);
      const cIrbComp_Clino1 & KthMeasure(int aK) const;
      size_t NbMeasure() const;

   private :
       std::vector<cIrbComp_Clino1>  mVCompClinos;
};

///  class for storing one time stamp in cIrbComp_Block
class   cIrbComp_TimeS : public cMemCheck
{
    public :
         friend cIrbComp_Block;

         cIrbComp_TimeS (const cIrbComp_Block &);
         const cIrbComp_CamSet & SetCams() const;  //< Accessor
          cIrbComp_CamSet & SetCams();  //< Accessor
          const cIrbComp_ClinoSet & SetClino() const;

         const cIrbComp_Block & CompBlock() const; //< Accessor
         const cIrbCal_Block & CalBlock() const; //< Accessor or Accessor
         // cIrbComp_Block & CompBlock() ; //< Accessor

         // if not SVP and cannot compute : error
         void ComputePoseInstrument(const std::vector<int> & aVNumCam,bool SVP = false);
         void SetClinoValues(const cOneMesureClino&);

         tREAL8 ScoreDirClino(const cPt3dr& aDir,size_t aKClino) const;

    private :
         cIrbComp_TimeS(const cIrbComp_TimeS&) = delete;
         const cIrbComp_Block &            mCompBlock;
         cIrbComp_CamSet                   mSetCams;
         cIrbComp_ClinoSet                 mSetClino;

         /** Not sure which role will play the notion of "pose of the instrument"*/
         bool                              mPoseInstrIsInit;
         tPoseR                            mPoseInstr;
};

///  class for using a rigid bloc in computation (calibration/compensation)
//   cIrbComp_Block
class   cIrbComp_Block : public cMemCheck
{
    public :
       typedef  std::tuple<tREAL8,tPoseR,cIrb_SigmaInstr>  tResCompCal;
       typedef  std::map<std::string,cIrbComp_TimeS>       tContTimeS;

       //   =================  Constructors =========================================

       /// "fundamuntal" constructor, creat from a calibration bloc
       cIrbComp_Block( cIrbCal_Block *,bool CalIsAdopted = true) ;
       /// read calib from file with "absolute name" and call fundamuntal constructor
       cIrbComp_Block(const std::string & aNameFile);
       /// read calib from name of block in standdar MMVI file and call fundamental constructot
       cIrbComp_Block(const cPhotogrammetricProject& ,const std::string & aNameBloc);

       ~cIrbComp_Block();

       //   =================  Accessors =========================================
       const cIrbCal_CamSet &  SetOfCalibCams() const ; //< Accessor of Accessor
       size_t  NbCams() const ;                         //< Accessor of Accessor of ...
       const cIrbCal_Block & CalBlock() const ; //< Accessor
       cIrbCal_Block & CalBlock()  ; //< Accessor

       const tContTimeS & DataTS() const ; //< Accessor
       tContTimeS & DataTS(); //< Accessor

       //  compute pose of instrument for all time stamp
       void ComputePoseInstrument(bool SVP = false);


       // Add all image of vect
       void AddImagesPoses(const std::vector<std::string> &,bool okImNotInBloc=false,bool Adopt=false);

       // Add an image if orientation exist (via PhProj)
       void AddImagePose(const std::string &,bool okImNotInBloc=false,bool usePosOfCalib=false);
       // Add an image with given pose
       void AddImagePose(cSensorCamPC * aCamPC,bool okImNotInBloc=false,bool usePosOfCalib=false);

       // for a given pair K1/K2 the 'best' relative pose and its sigma
       tResCompCal ComputeCalibCamsInit(int aK1,int aK2) const;

       //
       void SetClinoValues(const cSetMeasureClino&,bool OkNewTimeS=false );
       /// call previous by using std measure on phproj
       void SetClinoValues(bool OkNewTimeS=false);

       /// return the average of score of all clinos loaded
       tREAL8 ScoreDirClino(const cPt3dr& aDir,size_t aKClino) const;

    private :
       /// non copiable, too "dangerous"
       cIrbComp_Block(const cIrbComp_Block & ) = delete;
       const cPhotogrammetricProject &     PhProj();  //< Accessor, test !=0

       /**  return the data for time stamps (cams, clino ...)  corresponding to TS, possibly init it*/
       cIrbComp_TimeS &  DataOfTimeS(const std::string & aTS);
       // return
       ///cIrbComp_TimeS *  PtrDataOfTimeS(const std::string & aTS);


       cIrbCal_Block *                       mCalBlock;
       bool                                  mCalIsAdopted;
       const cPhotogrammetricProject *       mPhProj;
       tContTimeS                            mDataTS;
};


};

#endif  //  _MMVII_INSTRUMENTAL_BLOCK_

