#ifndef  _MMVII_BLOC_RIG_H_
#define  _MMVII_BLOC_RIG_H_

//  RIGIDBLOC  : all file

#include <vector>
#include "MMVII_AllClassDeclare.h"
#include "MMVII_util_tpl.h"
#include "MMVII_PCSens.h"



namespace MMVII
{

/** \file MMVII_BlocRig.h
    \brief Class for bloc rigid : initializ and impose constraint
 *  
*/

class cSetSensSameId;     //  one set of sensor (at same time, or same camera)
class cBlocMatrixSensor;  //  matricial organization of sensor
class cDataBlocCam;       //  data required for read/write bloc state
class cBlocOfCamera;      //  the bloc of camera itself


/**  store a set of "cSensorCamPC" sharing the same identifier, can be ident of time or ident  of camera, this
 * class is a an helper for implementation of set aqcuired at same time or of same camera */

class cSetSensSameId
{
      public :
         friend class cBlocMatrixSensor;

         ///  Construct with a number of sensor + the id common to all
         cSetSensSameId(size_t aNbCam,const std::string & anIdSync);
         void Resize(size_t); ///<  Extend size with nullptr

         const std::vector<cSensorCamPC*>&   VCams() const;  ///< Accessor
         const std::string & Id() const;  ///< Accessor
      private :
         /* Identifier common to same sensors like  "Im_T01_CamA.JPG","Im_T01_CamB.JPG"  => "T01"
          *                                   or    "Im_T01_CamA.JPG","Im_T02_CamA.JPG"  => "CamA" */
         std::string                 mId;
         std::vector<cSensorCamPC*>  mVCams;     ///< vector of camera sharing same  id
};

/** class to represent the "matricial" organization of bloc of sensor for example with a
 *  bloc of 2 camera "A" and "B" acquired at 5 different times, we can  will organize
 *
 *        A1  A2 A3 A4 A5
 *        B1  B2 B3 B4 B5
 *
 *  The class is organize to alloxw a dynamic creation, we permanently maintain the matrix structure
 *
 * */

class cBlocMatrixSensor
{
      public :

           size_t NbSet() const;
           size_t NbInSet() const;
           const std::string & NameKthSet(size_t) const;
           // const std::string & NameKthInSet(size_t) const;

           /// return the num of the set associated to a string (possibly new)
           size_t NumStringCreate(const std::string &) ;
           /// return the num of an "existing" set associated to a string (-1 if dont exist)
           int NumStringExist(const std::string &,bool SVP=false) const ;

           /// Add a sensor in the matrix at given "Num Of Set" and given "Num inside Set"
           void AddNew(cSensorCamPC*,size_t aNumSet,size_t aNumInSet);

           /// Creator
           cBlocMatrixSensor();
           /// Show the structure, tuning process
           void ShowMatrix() const;

           /// extract the camera for a given "Num Of Set" and a given "Num inside set"
           cSensorCamPC* &  GetCam(size_t aNumSet,size_t aNumInSet);
           cSensorCamPC*  GetCam(size_t aNumSet,size_t aNumInSet) const;

           const cSetSensSameId &  KthSet(size_t aKth) const;
      private :
           size_t                        mMaxSzSet;  ///< max number of element in Set
           t2MapStrInt                   mMapInt2Id; ///< For string/int conversion : Bijective map  SyncInd <--> int
           std::vector<cSetSensSameId>   mMatrix;    /// the matrix itself
};

/** Class for  representing all the data required for creating and saving the bloc structure  */
class cDataBlocCam
{
      public :
           friend class cBlocOfCamera;

           typedef std::map<std::string,tPoseR> tMapStrPose;

           cDataBlocCam(const std::string & aPattern,size_t aKPatBloc,size_t aKPatSync,const std::string & aName);
           /// for serialization
           cDataBlocCam();

           /** Compute the index of a sensor inside a bloc, pose must have
               same index "iff" they are correpond to a position in abloc */
           std::string  CalculIdBloc(cSensorCamPC * ) const ;
           /** Compute the synchronisation index of a sensor, pose must have
               same index "iff" they are acquired at same time */
           std::string  CalculIdSync(cSensorCamPC * ) const ;
           /// it may happen that processing cannot be made
           bool  CanProcess(cSensorCamPC * ) const ;
           ///  Local function of serialization, access to private member
           void AddData(const  cAuxAr2007 & anAux);
      private :
           std::string            mName;       ///< Identifier , will be usefull if/when we have several bloc
           std::string            mPattern;    ///< Regular expression for extracting "BlocId/SyncId"
           size_t                 mKPatBloc;   ///< Num of sub-expression that contain the CamId
           size_t                 mKPatSync;   ///< Num of sub-expression that contain the sync-id
           std::string            mMaster;     ///<  Name of master cam
           tMapStrPose            mMapPoseInBloc;    ///<  Map  IdCam -> PoseRel 
};

///  Global function with standard interface  required for serialization => just call member
void AddData(const  cAuxAr2007 & anAux,cDataBlocCam & aBloc) ;


/**   Class for manipulating  concretely the bloc of camera  , it contains essentially the 
 */
class cBlocOfCamera : public cMemCheck
{
      public :
	   typedef std::map<std::string,cPoseWithUK> tMapStrPoseUK;
           bool AddSensor(cSensorCamPC *);

           void ShowByBloc() const;
           void ShowBySync() const;
           cBlocOfCamera(const std::string & aPattern,size_t aKBloc,size_t aKSync,const std::string & aName);

           size_t  NbInBloc() const;
           size_t  NbSync() const;

           int NumInBloc(const std::string &,bool SVP=false)  const;

           const std::string & NameKthInBloc(size_t) const;
           const std::string & NameKthSync(size_t) const;

           cSensorCamPC *  CamKSyncKInBl(size_t aKInBloc,size_t aKSync) const;

           tPoseR  EstimatePoseRel1Cple(size_t aKB1,size_t aKB2,cMMVII_Appli * anAppli,const std::string & aReportGlob);
           void    StatAllCples (cMMVII_Appli * anAppli);

           void EstimateBlocInit(size_t aKMaster);
           /// Standard interface to write an object
           void ToFile(const std::string &) const;
           /// Standard interface to create an object
           static cBlocOfCamera * FromFile(const std::string &) ;

           const std::string & Name() const; // Accessor to name

	   /// return the pose which is (generally arbitrarily) considered as "master"
	   cPoseWithUK & MasterPoseInBl()  ;
	   /// return index of master
	   size_t  IndexMaster() const  ;
	   /// return name of master
	   const std::string &  NameMaster() const  ;

	   ///  Acces  to unknowns pose of bloc
           tMapStrPoseUK& MapStrPoseUK();

           cPoseWithUK &  PoseOfIdBloc(size_t);

      private :
           cBlocOfCamera();
	   ///  Transformate the state from init to computable
	   void  Set4Compute();

	   bool                          mForInit;          ///<  Some behaviur are != in init mode and in calc mod
           cDataBlocCam                  mData;             ///< Data part,
           cBlocMatrixSensor             mMatSyncBloc;      ///< Matrix [IdSync][IdBloc]  -->  cSensorCamPC*
           cBlocMatrixSensor             mMatBlocSync;      ///< Matrix [IdBloc][IdSync]  -->  cSensorCamPC*
	   tMapStrPoseUK                 mMapPoseInBlUK;    ///< Map Name-> Unknown Pose
	   std::vector<cPoseWithUK*>     mVecPUK;           ///< point to mMapPoseInBlUK for direct access from int
};



};

#endif  //  _MMVII_BLOC_RIG_H_
