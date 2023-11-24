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
class cCalibBlocCam;       //  data required for read/write bloc state
class cBlocOfCamera;      //  the bloc of camera itself

/*   Global organzization :
 *   Supose 
 *       - we have a rigid bloc 3 a camera : {a,b,c},  
 *       - we have a image acquire at $5$ times 
 *
 *    Let note {Ima1,Ima2... Imb5,Imc5}   the different images
 *
 *   "cSetSensSameId"  represent one set of sensor as a vector, for example [Ima2,Imb2,Imc2]
 *
 *   "cBlocMatrixSensor"  represent the set of all images organized as a matrix.
 *   For example , cBlocOfCamera contain two cBlocMatrixSensor 
 *
 *        mMatSyncBloc -> [ [Ima1,Imb1,Imc1]  ...   [Ima5,Imb5,Imc5] ]  , for reprsenting camera acquired at same time in a bloc
 *        mMatBlocSync -> [ [Ima1,Ima2,Ima3,Ima4,Ima5] .... [..Imc5] ]  , for representing a same position across time
 *
 *   For easy creation of the structure, the matrix contains  mMapInt2Id that compute correspondance between string and 
 *   number ("a"<=>0, "b"<=>1 , ...).
 *
 *    "cCalibBlocCam" contains the data  that allow to construct & save the bloc , essantially :
 *
 *        - the regular expression that allow to compute bloc&time identifier from a name
 *        - the result of bloc calibration  (with translation Tr, and rotation Rot) :
 *             "a" ->  Tra  Rota
 *             "b" ->  Trb  Rotb
 *
 *    "cBlocOfCamera" is the main structure for representing a bloc a camera, it contains :
 *
 *       - the two matrices mMatSyncBloc and mMatBlocSync mentionned above
 *       - a  cCalibBlocCam that allow to compute the matrices and store the initial value of bloc-calib
 *       - a set of unkown poses that can be used in the bundle adustment.
 *
 *    The 3 main operation that we want to do with rigid bloc :
 *
 *        * compute initial value of bloc calibration from a set of existing pose, this is
 *          done by method EstimateBlocInit
 *
 *        * refine by global optimization the value of a bloc 
 *
 *        * use a calibrated bloc as a constraint
 */


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
           /// Number of identifier in the bloc, may exist even when bloc is empty
           size_t SetCapacity() const;

	   ///  number of set
           size_t NbSet() const;  
	   /// size of each set
           size_t NbInSet() const;
	   /// name of kth set  (acces to "Id()" of  Kth "cSetSensSameId")
           const std::string & NameKthSet(size_t aKth) const;

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

           /// extract the camera for a given "Num Of Set" and a given "Num inside set", reference so can fill the matrix
           cSensorCamPC* &  GetCam(size_t aNumSet,size_t aNumInSet);
	   /// const version of above
           cSensorCamPC*  GetCam(size_t aNumSet,size_t aNumInSet) const;

	   /// Acces to Kth set
           const cSetSensSameId &  KthSet(size_t aKth) const;
      private :
           size_t                        mMaxSzSet;  ///< max number of element in Set
           t2MapStrInt                   mMapInt2Id; ///< For string/int conversion : Bijective map  SyncInd <--> int
           std::vector<cSetSensSameId>   mMatrix;    /// the matrix itself
};

/** Class for  representing all the data required for creating and saving the bloc structure  */
class cCalibBlocCam
{
      public :
           /// dont use friend generally, but "cCalibBlocCam" is just a helper for "cBlocOfCamera"
           friend class cBlocOfCamera;

	   typedef std::map<std::string,cPoseWithUK> tMapStrPoseUK;

	   /// contructor allow to compute identifier time/bloc + Name of the bloc
           cCalibBlocCam(const std::string & aPattern,size_t aKPatBloc,size_t aKPatSync,const std::string & aName);
           /// defalut constructor for serialization
           cCalibBlocCam();

           /** Compute the index of a sensor inside a bloc, pose have same index "iff" they are correpond to a position in abloc */
           std::string  CalculIdBloc(cSensorCamPC * ) const ;
           /** Compute the synchronisation index of a sensor, pose  have same index "iff" they are acquired at same time */
           std::string  CalculIdSync(cSensorCamPC * ) const ;
           /// it may happen that processing cannot be made  (different bloc, or camera outside the bloc)
           bool  CanProcess(cSensorCamPC * ) const ;
           ///  Local function for serialization, access to private member
           void AddData(const  cAuxAr2007 & anAux);
      private :
           std::string       mName;            ///< Identifier , will be usefull if/when we have several bloc
           std::string       mPattern;         ///< Regular expression for extracting "BlocId/SyncId"
           size_t            mKPatBloc;        ///< Num of sub-expression that contain the CamId
           size_t            mKPatSync;        ///< Num of sub-expression that contain the sync-id
           std::string       mMaster;          ///<  Name of master cam
           tMapStrPoseUK     mMapPoseUKInBloc; ///<  Map  IdCam -> PoseRel 
};

///  Global function with standard interface  required for serialization => just call member
void AddData(const  cAuxAr2007 & anAux,cCalibBlocCam & aBloc) ;


/**   Class for manipulating  concretely the bloc of camera  , it contains essentially the 
 */
class cBlocOfCamera : public cMemCheck
{
      public :
           typedef std::map<std::string,tPoseR> tMapStrPose;
	   typedef std::map<std::string,cPoseWithUK> tMapStrPoseUK;

	   /// Add a new sensor to the matrixes
           bool AddSensor(cSensorCamPC *);

           void ShowByBloc() const;  /// show matrixial organozation by bloc
           void ShowBySync() const;  /// show matricial organization by time

	   /// construct a bloc, empty for now
           cBlocOfCamera(const std::string & aPattern,size_t aKBloc,size_t aKSync,const std::string & aName);
	   /// destructor (do nothing at time being)
           ~cBlocOfCamera();

           size_t  NbInBloc() const;  ///< Number of camera in a bloc
           size_t  NbSync() const;    ///< Number of synchronization
           // Number of NbInBloc pre-allocated, can be < NbInBloc, when not all poses added
           size_t BlocCapacity() const;

	   /// For of bloc id, return its number;- if dont exist : -1 if SVP, error else
           int NumInBloc(const std::string &,bool SVP=false)  const;

	   /// name of Kth in bloc
           const std::string & NameKthInBloc(size_t) const;
	   /// name of  Kth sync
           const std::string & NameKthSync(size_t) const;

	   ///  return sensor at given KInBloc/KSync, can be nullptr if no cam at this position
           cSensorCamPC *  CamKSyncKInBl(size_t aKInBloc,size_t aKSync) const;

	   /** Given to number in a bloc : compute the average relative orientation, 
	    * optionnaly out in report :
	           - the detail of each result if Appli is given
		   - the std deviation
           */
           tPoseR  EstimatePoseRel1Cple(size_t aKB1,size_t aKB2,cMMVII_Appli * anAppli,const std::string & aReportGlob);

	   /// Do the stat for all pair by calling EstimatePoseRel1Cple 
           void    StatAllCples (cMMVII_Appli * anAppli);
           /// Make an estimation all pose relative to given master, and save it internally
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

	   ///  Acces  to unknowns pose of a given id
           cPoseWithUK &  PoseUKOfNumBloc(size_t);
	   ///  Acces  to unknowns pose of a given id
           const tPoseR  &  PoseInitOfNumBloc(size_t) const;
	   ///  Acces  to unknowns pose of a given id
           cPoseWithUK &  PoseUKOfIdBloc(const std::string&) ;

	   /// A function, that illustrate the read/write possibility once object is serialized
	   void TestReadWrite(bool OmitDel) const;

      private :
           cBlocOfCamera();
	   ///  Transformate the state from init to computable
	   void  Set4Compute();

	   bool                          mForInit;          ///<  Some behaviur are != in init mode and in calc mod
           cCalibBlocCam                  mData;             ///< Data part,
           cBlocMatrixSensor             mMatSyncBloc;      ///< Matrix [IdSync][IdBloc]  -->  cSensorCamPC*
           cBlocMatrixSensor             mMatBlocSync;      ///< Matrix [IdBloc][IdSync]  -->  cSensorCamPC*
	   tMapStrPose                   mMapPoseInit;    ///< Map Name-> Unknown Pose
	   //std::vector<cPoseWithUK*>     mVecPUK;           ///< point to mMapPoseInBlUK for direct access from int
};



};

#endif  //  _MMVII_BLOC_RIG_H_
