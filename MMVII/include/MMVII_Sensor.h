#ifndef  _MMVII_SENSOR_H_
#define  _MMVII_SENSOR_H_

#include <set>

#include "SymbDer/SymbDer_Common.h"
#include "MMVII_Mappings.h"
#include "MMVII_MeasuresIm.h"
#include "MMVII_Geom2D.h"
#include "MMVII_Geom3D.h"
#include "MMVII_DeclareCste.h"

using namespace NS_SymbolicDerivative;

namespace MMVII
{

/** \file MMVII_Sensor.h
    \brief Interface class for sensors

   Most probably this will evolve a lot, with several reengenering 
  phases. 

   For now, it will essentially contain interfaces to existing MMV1 cameras.

*/

class  cSensorImage;
class  cDataPixelDomain ;
class  cPixelDomain;
class  cPerspCamIntrCalib;
class  cSensorCamPC;
class  cPhotogrammetricProject;
class  cSIMap_Ground2ImageAndProf ;
class  cCalculMetaDataProject;
class  cGlobCalculMetaDataProject;

/**  helper for cPixelDomain, as the cPixelDomain must be serialisable we must separate the
 * minimal data for description, with def contructor from the more "sophisticated" object  */
class cDataPixelDomain 
{
      public :
           cDataPixelDomain(const cPt2di &aSz);

           const cPt2di & Sz() const;
	   virtual void AddData(const cAuxAr2007 & anAux);
      protected :
           cPt2di     mSz;
};


/**  base-class  4 definition of validity domaine in image space  */
class cPixelDomain :  public cDataBoundedSet<tREAL8,2>
{
	public :
		cPixelDomain(cDataPixelDomain *);
                const cPt2di & Sz() const;
	private :
		cDataPixelDomain * mDPD;
};


/*  base-class  4 all image sensor */

class cSensorImage  :   public cObj2DelAtEnd,
                        public cObjWithUnkowns<tREAL8>
{
     public :

          cSensorImage(const std::string & aNameImage);

	  /// create a sensor in a new coordinate system
	  virtual cSensorImage * SensorChangSys(cDataInvertibleMapping<tREAL8,3> &) const = 0;

          virtual const cPixelDomain & PixelDomain() const = 0;
          const cPt2di & Sz() const;
	  ///  Generate a point ground visible by sensor
          cPt3dr RandomVisiblePGround(tREAL8 aDepMin,tREAL8 aDepMax);
	  ///  Generate a point on sensor
          cPt2dr RandomVisiblePIm() const ;
	  ///  Position in [0 1]^ 2
          cPt2dr RelativePosition(const cPt2dr &) const ;

	  /// Generate a random point visible on 2 image , algo : generate 2 random point and comppute bundle inter
	  cPt3dr RandomVisiblePGround(const cSensorImage &,int aNbTestMax=10000,bool * OK =nullptr ) const;

	  /// reproject RandomVisiblePGround
	  cHomogCpleIm RandomVisibleCple(const cSensorImage &,int aNbTestMax=10000,bool * OK =nullptr ) const;


	 // =================   Image <-->  Ground  mappings  ===========================
	 
         ///  The most fundamental method, theoretically should be sufficient
         virtual tSeg3dr  Image2Bundle(const cPt2dr &) const =0;
	 /// Basic method  GroundCoordinate ->  image coordinate of projection
         virtual cPt2dr Ground2Image(const cPt3dr &) const = 0;
	 ///  add the the depth (to see if have a default with bundle+Gr2Ima)
         virtual cPt3dr Ground2ImageAndDepth(const cPt3dr &) const = 0;
         /// Invert of Ground2ImageAndDepth
         virtual cPt3dr ImageAndDepth2Ground(const cPt3dr &) const = 0;
	 /// Facility for calling ImageeAndDepth2Ground(const cPt3dr &)
         cPt3dr ImageAndDepth2Ground(const cPt2dr &,const double & ) const;


	 /// Compute 3D point by bundle intersection
	  cPt3dr PInterBundle(const cHomogCpleIm & aCple,const cSensorImage &) const;

	 // =================   Visibility/validity   ===========================

	 /// Indicate how much a point belongs to sensor visibilty domain
         virtual double DegreeVisibility(const cPt3dr &) const =0 ;
	 /// Indicacte how much a 2 D points belongs to definition of image frame
         virtual double DegreeVisibilityOnImFrame(const cPt2dr &) const =0 ;
	 /// Indicate if a point belongs to sensor visibilty domain (threshold DegreeVisibility)
         bool IsVisible(const cPt3dr &) const  ;
	 /// Indicacte how a 2 D points belongs to definition of image frame (threshold DegreeVisibilityOnImFrame)
         bool IsVisibleOnImFrame(const cPt2dr &) const  ;
	 /// 2d & 3d are visible
         bool PairIsVisible(const cPair2D3D &) const  ;

	 // =================   Generation of points & correspondance   ===========================

	 /// return a set point regulary sampled (+/-) on sensor, take care of frontier
         virtual std::vector<cPt2dr>  PtsSampledOnSensor(int aNbByDim)  const = 0;
	 ///  return artificial/synthetic correspondance , with vector of depth
	 cSet2D3D  SyntheticsCorresp3D2D (int aNbByDim,std::vector<double> & aVecDepth) const;
	 ///  call variant with vector, depth regularly spaced
	 cSet2D3D  SyntheticsCorresp3D2D (int aNbByDim,int aNbDepts,double aD0,double aD1) const;


	 // =================   Residual   ===========================
         double SqResidual(const cPair2D3D &) const;  ///< residual Proj(P3)-P2 , squared for efficiency
         double AvgSqResidual(const cSet2D3D &) const;  ///< avereage on all pairs, sqrt of square
	 /// "robust" average using a weigt in sigma/(sigma+R)  , sigma being a prop of residual
         double RobustAvResidualOfProp(const cSet2D3D &,double aProp) const;  


         cPt3dr Image2PlaneInter(const cPlane3D & aPlane,const cPt2dr &) const;
         cPt2dr Image2LocalPlaneInter(const cPlane3D & aPlane,const cPt2dr &) const;
         cEllipse EllipseIm2Plane(const cPlane3D & aPlane,const cEllipse & aEllipseIm,int aNbTeta) const;

	 // --------------------   methods to compute names of an orientation --------------------

	 const std::string & NameImage() const;   ///< accessor
	 void SetNameImage(const std::string &);  ///< used when reading from file

	 static std::string  PrefixName() ;  ///<  Prefix to all file for ori of 1 images
	 ///  Compute the name from the specific prefix of subtype and image name, need a static access in creation
	 static  std::string  NameOri_From_PrefixAndImage(const std::string & aPrefix,const std::string & aNameImage);
	 ///  Simply return the name
	 std::string NameOriStd() const ;
	 ///  Prefix of the subtype
	 virtual std::string  V_PrefixName() const = 0  ;
	 /// method for saving oblet
	 virtual void ToFile(const std::string &) const = 0;

	 // --------------------   methods used in bundle adjustment  --------------------
	
	 ///  For stenope camera return center, for other best approx
	 virtual cPt3dr  PseudoCenterOfProj() const = 0;
	 ///  For stenope camera return center, for other nullptr
	 virtual const cPt3dr * CenterOfPC() const = 0;
	 // Create if new, and memorize the colinearity equation
         cCalculator<double> * SetAndGetEqColinearity(bool WithDerives,int aSzBuf,bool ReUse);

	 // suppose it was init (and assert it)
         cCalculator<double> * GetEqColinearity();

	 /// If the camera has its own "obs/cste" (like curent rot for PC-Cam) that's the place to say it
	 virtual  void PushOwnObsColinearity( std::vector<double> &) = 0;

     private :
	 /// Return the calculator, adapted to the type, for computing colinearity equation
         virtual cCalculator<double> * CreateEqColinearity(bool WithDerives,int aSzBuf,bool ReUse) = 0;

	 std::string                                   mNameImage;
         cCalculator<double> *                         mEqColinearity;  // memo equation, can be nullptr (for pure pose)
	 bool                                          mEqCIsInit;      // memo if init of "mEqColinearity" was done
	 // static std::map<std::string,cSensorImage*>  mDicoSensor;
	 // static int                                  mNum;
};

/**  Interfac to make sensor a 3d-mapping, using Ground2ImageAndDepth function */
class cSIMap_Ground2ImageAndProf : public cDataInvertibleMapping<tREAL8,3>
{
       public :
           cSIMap_Ground2ImageAndProf(cSensorImage *);
	   cPt3dr Value(const cPt3dr &) const;   ///< 3D ->  2D+depth
	   cPt3dr Inverse(const cPt3dr &) const;  ///< 2D+depth -> 3D
       private :
           cSensorImage * mSI;
};

/** represent the set of visible point of the camera */
class cSetVisibility : public cDataBoundedSet<tREAL8,3>
{
    public :
        cSetVisibility(cSensorImage * aSens,double aBorder = 0) ;
        tREAL8 Insideness(const tPt & aP) const;
    private :
        cSensorImage * mSens;
	double         mBorder;
};

/**  class for storing meta-data as stored in xif part*/
class cMetaDataImage
{
      public :
          tREAL8  Aperture(bool SVP=false) const;
          tREAL8  FocalMM(bool SVP=false) const;
          tREAL8  FocalMMEqui35(bool SVP=false) const;
          cPt2di  NbPixels(bool SVP=false) const;
	  const std::string&  CameraName(bool SVP=false) const;  // return NONE if 

          /// case where it was fixed directly by user to import accurately an existing cal
          tREAL8  FocalPixel(bool SVP=false) const;
          /// idem, FocalPixel
          cPt2dr  PPPixel(bool SVP=false) const;
          //  generate an identifier specific to data 
	  std::string InternalCalibGeomIdent() const;

          cMetaDataImage(const std::string & aDir,const std::string & aNameIm,const cGlobCalculMetaDataProject * aCalc);
          cMetaDataImage();
      private :

          std::string    mCameraName;
          std::string    mAdditionalName;
          tREAL8         mAperture;
          tREAL8         mFocalMM;
          tREAL8         mFocalMMEqui35;
          tREAL8         mFocalPixel;
          cPt2dr         mPPPixel;
          cPt2di         mNbPixel;
          std::string    mNameImage;
};

class cElemCamDataBase
{
     public :
        void AddData(const cAuxAr2007 & anAux);
        std::string  mName;
        cPt2dr       mSzPixel_Micron;   /// May be not square
        cPt2dr       mSzSensor_Mm;  /// Physical size
        cPt2di       mNbPixels;  /// Number of Pixels

        // As it is redundant check and complete
        void  Finish();
};
void AddData(const cAuxAr2007 & anAux,cElemCamDataBase &);

class cCamDataBase
{
   public :
        void AddData(const cAuxAr2007 & anAux);
        const std::map<std::string,cElemCamDataBase>  & Map() const;
        std::map<std::string,cElemCamDataBase>  & Map() ;
   private :
       std::map<std::string,cElemCamDataBase>  mMap;
};
void AddData(const cAuxAr2007 & anAux,cCamDataBase &);



/**   Class for sharind code related to management the folder for one kind (=Ori,Homol,Radiom...) of "objects"
 *    Used by cPhotogrammetricProject
 */

class cDirsPhProj : public cMemCheck
{
     public :

          cDirsPhProj(eTA2007 aMode,cPhotogrammetricProject & aPhp);
          void Finish();

	  /// Input Orientation as mandatory paramaters
          tPtrArg2007     ArgDirInMand(const std::string & aMes="") ;  
	  /// Input Orientation as optional paramaters
          tPtrArg2007     ArgDirInOpt(const std::string & aNameVar="",const std::string & aMesg="",bool WithHDV=false) ;   

          tPtrArg2007  ArgDirInputOptWithDef(const std::string & aDef,const std::string & aNameVar="",const std::string & aMesg="") ;   
									    //
	  /// Output Orientation as mandatory paramaters
          tPtrArg2007     ArgDirOutMand(const std::string & aMes="");  
	  /// Output Orientation as optional paramaters
          tPtrArg2007     ArgDirOutOpt(const std::string & aNameVar="",const std::string & aMesg="",bool WithHDV=false) ;   
	  /// Output Orientation as optional paramaters  with DEF VALUE
          tPtrArg2007  ArgDirOutOptWithDef(const std::string & aDef,const std::string & aNameVar="",const std::string & aMesg="") ;   


          void  SetDirIn(const std::string&) ; ///< Modifier, use in case many out params were saved in a xml,like with MeshImageDevlp
          void  SetDirInIfNoInit(const std::string&) ; ///< do it only if not init in argopt
          const std::string & DirIn() const;   ///< Accessor
          const std::string & DirOut() const;   ///< Accessor
          const std::string & FullDirIn() const;   ///< Accessor
          const std::string & FullDirOut() const;   ///< Accessor
          const std::string & FullDirInOut(bool In) const;   ///< Facility FullDirIn/FullDirInOut

          const std::string & DirLocOfMode() const;   ///< Accessor

	  void SetDirOut(const std::string & aDirOut); ///< Modifier, for example to have defaut value to input
	  bool  DirInIsInit() const;
	  bool  DirOutIsInit() const;
          void  AssertDirInIsInit() const;
          void  AssertDirOutIsInit() const;

	  void SetDirOutInIfNotInit(); ///< If Dir Out is not init, set it to same value than In
     private :
          cDirsPhProj(const cDirsPhProj &) = delete;

          eTA2007                   mMode;
          cPhotogrammetricProject&  mPhp;
          cMMVII_Appli &            mAppli;

          const std::string         mPrefix;
          std::string               mDirLocOfMode;  // For ex "MMVII-PhgrProj/Orient/"  "MMVII-PhgrProj/Radiom/"   ...

          std::string               mDirIn;
          std::string               mDirOut;
          std::string               mFullDirIn;
          std::string               mFullDirOut;
          bool                      mPurgeOut;
};


/** Class to facilitate the management of orientations (and others ?) in a photogrammetric
 * application.  Offers facilities for :
 *        * readr/write an orient to file
 *        * read/write parameters from line command on a standardized way
 *        * eventually, deallocation of cameras
 *
 */


class cPhotogrammetricProject
{
      public :

	      
	 //===================================================================
         //==============   CONSTRUCTION & DESTRUCTION   =====================
	 //===================================================================

	        /// constructor : will memorize application
          cPhotogrammetricProject(cMMVII_Appli &);
                /// some initialisation can be done only once Appli is itself init, method must be called in mAppli.Exe()
          void FinishInit() ;
	        /// destructor  ,  some object delegates their destruction to this
          ~cPhotogrammetricProject();

	 //========================== ACCESSOR ===============================

          cMMVII_Appli &  Appli(); ///< Accessor
	  const std::string & TaggedNameDefSerial() const; /// short to Appli.Nam...
	  const std::string & VectNameDefSerial() const; /// short to Appli.Nam...
	  cDirsPhProj &   DPOrient(); ///< Accessor
	  cDirsPhProj &   DPRadiomData(); ///< Accessor
	  cDirsPhProj &   DPRadiomModel(); ///< Accessor
	  cDirsPhProj &   DPMeshDev(); ///< Accessor
	  cDirsPhProj &   DPMask(); ///< Accessor
	  cDirsPhProj &   DPPointsMeasures(); ///< Accessor
	  cDirsPhProj &   DPMetaData();    ///<  Accessor
	  cDirsPhProj &   DPTieP();    ///<  Accessor
	  cDirsPhProj &   DPMulTieP();    ///<  Accessor
	  cDirsPhProj &   DPRigBloc();    ///<  Accessor  // RIGIDBLOC
				    
	  const cDirsPhProj &   DPOrient() const; ///< Accessor
	  const cDirsPhProj &   DPRadiomData() const; ///< Accessor
	  const cDirsPhProj &   DPRadiomModel() const; ///< Accessor
	  const cDirsPhProj &   DPMeshDev() const; ///< Accessor
	  const cDirsPhProj &   DPMask() const; ///< Accessor
	  const cDirsPhProj &   DPPointsMeasures() const; ///< Accessor
	  const cDirsPhProj &   DPMetaData() const;    ///<  Accessor
	  const cDirsPhProj &   DPTieP() const;    ///<  Accessor
	  const cDirsPhProj &   DPMulTieP() const;    ///<  Accessor
	  const cDirsPhProj &   DPRigBloc() const;    ///<  Accessor  // RIGIDBLOC


	  // Sometime we need several dir of the same type, like "ReportPoseCmp", or RefPose in bundle
	  cDirsPhProj * NewDPIn(eTA2007 aType,const std::string & aDirIn);

	  const std::string &   DirPhp() const;   ///< Accessor
	  const std::string &   DirVisu() const;   ///< Accessor
	  const std::string &   DirSysCo() const;   ///< Accessor
          tPtrArg2007           ArgChSys(bool DefaultUndefined=false);

	 //===================================================================
         //==================   ORIENTATION      =============================
	 //===================================================================
	 
               //  Read/Write
          void SaveSensor(const cSensorImage &) const; ///< Save camera using OutPut-orientation
          void SaveCamPC(const cSensorCamPC &) const; ///< Save camera using OutPut-orientation
	  void SaveCalibPC(const  cPerspCamIntrCalib & aCalib) const;  ///< Save calibration using  OutPut-orientation

	  cSensorCamPC * ReadCamPC(const std::string &,bool ToDeleteAutom,bool SVP=false) const; ///< Create Camera using Input orientation
	  cSensorCamPC * ReadCamPC(const cDirsPhProj&,const std::string &,bool ToDeleteAutom,bool SVP=false) const; ///< Create Camera using Input orientation


	  /// Load a sensor, try different type (will add RPC , and others ?) use autom delete (dont need to delete it)
	  void LoadSensor(const std::string &NameIm,cSensorImage* &,cSensorCamPC * &,bool SVP=false);

	  /// return the generic sensor, use autom delete (dont need to delete it)
	  cSensorImage* LoadSensor(const std::string  &aNameIm,bool SVP=false);


	      // Internal Calibration  

	  std::string  StdNameCalibOfImage(const std::string aNameIm) const;
	  std::string  FullDirCalibIn() const;
	  std::string  FullDirCalibOut() const;
	  
	  /// read Pose file  and extract the name of internal  calibration
          cPerspCamIntrCalib *  InternalCalibFromImage(const std::string &aNameIm) const;
	  ///  compute the standard name of calibration before reading it
	  cPerspCamIntrCalib *  InternalCalibFromStdName (const std::string aNameIm) const;

	 //===================================================================
         //==================   RADIOMETRY       =============================
	 //===================================================================

	       //  ------------  Create data --------------------
	      
	  /** Create a new radial-radiom calib adapted to image; geometric calibration & meta data must
	   * be accessible; if already exist with same name return existing; add in Obj2DelAtEnd, as many
	   * image will share it */
	  cRadialCRS * CreateNewRadialCRS(size_t aDegree,const std::string& aNameIm,bool WithCste=false,int aDegPol=-1);
               //     --------   Save Data ---------------------
          void SaveRadiomData(const cImageRadiomData &) const; ///< Save camera using OutPut-orientation
          void SaveCalibRad(const cCalibRadiomIma &) const; ///< Save radiom-calib using OutPut-orientation

               //     --------   Read Data ---------------------
	  cImageRadiomData * ReadRadiomData(const std::string &) const; ///< Read radiometric data for 1 image
	  cCalibRadiomIma *  ReadCalibRadiomIma(const std::string &) const; ///< Read radiom calib of 1 image

               //     --------   Standard names of files ---------------------
	  ///  Name of radiometric calibration with a  radial model , hypothesis : depends of internal calibration
	  ///  +  metadata (aperture)
          std::string NameCalibRadiomSensor(const cPerspCamIntrCalib &,const cMetaDataImage &) const;
	  ///  Call NameCalibRadiomSensor with parameters computed from image
          std::string NameCalibRSOfImage(const std::string &) const;

	 //===================================================================
         //==================    MASKS           =============================
	 //===================================================================

	  /// return full name including sub dirs
          std::string NameMaskOfImage(const std::string & aNameImage) const;
	  /// Does the image has an existing mask : Dir is init + file exist
          bool  ImageHasMask(const std::string & aNameImage) const;

	  cIm2D<tU_INT1>  MaskWithDef(const std::string & aNameImage,const cBox2di & aBox,bool DefVal) const;
	  
	 //===================================================================
         //==================    PointsMeasures  =============================
	 //===================================================================

	  void SaveMeasureIm(const cSetMesPtOf1Im & aSetM) const;
          /// return from Std Dir, can be out in case of reload
	  cSetMesPtOf1Im LoadMeasureIm(const std::string &,bool InDir=true) const;
	  void LoadGCP(cSetMesImGCP&,const std::string & aPatFiltrFile="",const std::string & aFiltrNameGCP="") const;
	  // if SVP && file doesnt exist, do nothing
	  void LoadIm(cSetMesImGCP&,const std::string & aNameIm,cSensorImage * =nullptr,bool SVP=false) const;
	  void LoadIm(cSetMesImGCP&,cSensorImage & ) const;
	  void SaveGCP(const cSetMesGCP&);

	  /// Name of the file, usefull if we need to test existence before doing anything
	  std::string NameMeasureGCPIm(const std::string & aNameIm,bool isIn) const;



	  /// Pattern for GCP file, if "" return default  = "cSetMesGCP::ThePrefixFiles.*.xml"
	  std::string GCPPattern(const std::string & aArgPatFiltr) const;
          std::vector<std::string>  ListFileGCP(const std::string & aArgPatFiltr) const;
	  void CpGCPPattern(const std::string& aDirIn,const std::string & aDirOut,const std::string & aArgPatFiltr="") const;
	  void CpGCP() const;

	  /// For a givgen image, return 3D-2D corresp, using LoadGCP&LoadIm
	  cSet2D3D  LoadSet32(const std::string & aNameIm) const;

          void SaveAndFilterAttrEll(const cSetMesPtOf1Im &  aSetM,const std::list<std::string> & ToRem)   const ;
	  
	 //===================================================================
         //==================   META-DATA       ==============================
	 //===================================================================

	  /// Return metadata while maintaining a map for assuring that read only once for a given image
          cMetaDataImage GetMetaData(const std::string &) const;

	  /// Compute de Meta Data Calculator from FullDirIn & Usage, create only once
	  cGlobCalculMetaDataProject*  InitGlobCalcMTD() const;
	  /// Internal,  to document later ...
	  cCalculMetaDataProject * CMDPOfName(const std::string &);

          /// Create calib w/o distorsion from paramameters
          cPerspCamIntrCalib * GetCalibInit(const std::string& aName,eProjPC aTypeProj,const cPt3di & aDeg,
                                            cPt2dr   aPP=cPt2dr(0.5,0.5), bool SVP=false);

          ///  Extract Camera specif from data base, given name of camera
          const cElemCamDataBase * GetCamFromNameCam(const std::string& aNameCam,bool SVP=false) const;


	 //===================================================================
         //==================   HOMOLOGOUS Points  ===========================
	 //===================================================================
	 
	 void  SaveHomol(const cSetHomogCpleIm &,
			 const std::string & aNameIm1 ,
			 const std::string & aNameIm2,const std::string & aDir="") const;

	 void  ReadHomol(cSetHomogCpleIm &,
			 const std::string & aNameIm1 ,
			 const std::string & aNameIm2,const std::string & aDir="") const;

	 std::string NameTiePIn(const std::string & aNameIm1,const std::string & aNameIm2,const std::string & aDir="") const;

	 //===================================================================
         //==================   Multiple Tie-Points  =========================
	 //===================================================================

	 //  for config we read/write by default in dmp, but allow to write a copy in xml/json .. for inspection
	 //  and/or external software
	 static std::string NameConfigMTP(const std::string &  anExt ="dmp");
	 std::string NameConfigMTPIn() const;
	 std::string NameConfigMTPOut(const std::string &  anExt ="dmp") const;

	 std::string NameMultipleTieP(const std::string &) const;
	 void  SaveMultipleTieP(const cVecTiePMul&,const std::string &) const;
	 void  ReadMultipleTieP(cVecTiePMul&,const std::string &,bool SVP =false ) const;
	 bool HasNbMinMultiTiePoints(const std::string & aNameIm,size_t aNbMin,bool AcceptNoDirIn =false) const;

	 //===================================================================
         //==================    Coord Sys           =========================
	 //===================================================================

                  //  ======== [1]  Sysco saved in "MMVII-PhgrProj/SysCo" 
	 void  SaveSysCo(tPtrSysCo,const std::string&,bool OnlyIfNew=false) const;
	 tPtrSysCo ReadSysCo(const std::string &aName,bool SVP=false) const;
	 tPtrSysCo CreateSysCoRTL(const cPt3dr & aOrig,const std::string &aName,bool SVP=false) const;
	 std::string  FullNameSysCo(const std::string &aName,bool SVP=false) const;
	 // return  identity if Vec not init
	 cChangSysCoordV2  ChangSys(const std::vector<std::string> &,tREAL8 aEpsDif=0.1);

                  //  ======== [1]  Sysco saved in "MMVII-PhgrProj/Ori/"  or "MMVII-PhgrProj/PointsMeasure//"
         std::string  NameCurSysCo(const cDirsPhProj &,bool IsIn) const;
         tPtrSysCo  CurSysCo(const cDirsPhProj &,bool SVP=false) const;
         tPtrSysCo  CurSysCoOri(bool SVP=false) const;
         tPtrSysCo  CurSysCoGCP(bool SVP=false) const;
         void SaveCurSysCo(const cDirsPhProj &,tPtrSysCo) const ;
         void SaveCurSysCoOri(tPtrSysCo) const ;
         void SaveCurSysCoGCP(tPtrSysCo) const ;
         void CpSysIn2Out(bool OriIn,bool OriOut) const;  // bool : Ori/GCP   do it only if exist, else no error

         const cChangSysCoordV2 & ChSys() const;
         cChangSysCoordV2 & ChSys() ;
         bool  ChSysIsInit() const;
         void  AssertChSysIsInit() const;

	 //===================================================================
         //==================   Rigid Bloc           =========================
	 //===================================================================
	 
	         // RIGIDBLOC
	 std::list<cBlocOfCamera *> ReadBlocCams() const;
	 void   SaveBlocCamera(const cBlocOfCamera &) const;

         //==================   Camera Data Base     =========================

         void MakeCamDataBase();
         bool OneTestMakeCamDataBase(const std::string & aDir,cCamDataBase &,bool ForceNew);

      private :
          cPhotogrammetricProject(const cPhotogrammetricProject &) = delete;

	  /// Void generate an empty file for computing MTD that can be uses as pattern
	  void GenerateSampleCalcMTD();
	  void DeleteMTD();

          cMMVII_Appli &  mAppli;
          std::string     mFolderProject;

	  std::string     mDirPhp;
	  std::string     mDirVisu;

	  std::string     mDirSysCo;
          std::vector<std::string>   mNameChSys;
          cChangSysCoordV2          mChSys;

	  cDirsPhProj     mDPOrient;
	  cDirsPhProj     mDPRadiomData;
	  cDirsPhProj     mDPRadiomModel;
	  cDirsPhProj     mDPMeshDev;
	  cDirsPhProj     mDPMask;
	  cDirsPhProj     mDPPointsMeasures;  ///<  For GCP measures  Image + Grounds
	  cDirsPhProj     mDPTieP;            ///<  For Homologous point
	  cDirsPhProj     mDPMulTieP;         ///<  For multiple Homologous point
	  cDirsPhProj     mDPMetaData;
	  cDirsPhProj     mDPRigBloc;         // RIGIDBLOC
					      //

	  std::vector<cDirsPhProj*> mDirAdded;
	  mutable cGlobCalculMetaDataProject *  mGlobCalcMTD;

          cCamDataBase   mCamDataBase;

};
void SaveAndFilterAttrEll(const cPhotogrammetricProject & aPhp,const cSetMesPtOf1Im &  aSetM,const std::set<std::string> & ToRem);

};

#endif  //  _MMVII_SENSOR_H_
