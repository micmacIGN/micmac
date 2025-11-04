#ifndef  _MMVII_SENSOR_H_
#define  _MMVII_SENSOR_H_

#include <set>

#include "SymbDer/SymbDer_Common.h"
#include "MMVII_Mappings.h"
#include "MMVII_MeasuresIm.h"
#include "MMVII_Geom2D.h"
#include "MMVII_Geom3D.h"
#include "MMVII_DeclareCste.h"
#include "MMVII_SysCo.h"


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
class  cBA_Topo;
class  cBA_GCP;
class  cTripletSet;
class  cStaticLidar;

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
		// probably to virtualize later
		tREAL8 DegreeVisibility(const cPt2dr & aP) const;
	private :
		cDataPixelDomain * mDPD;
};



          // type to store the result of jacobian of proj R3->R2, stores P(G)  dP/dX  dP/dY dP/dZ
class tProjImAndGrad
{
      public :
          cPt2dr   mPIJ;
          cPt3dr   mGradI;
          cPt3dr   mGradJ;
};

/*  base-class  4 all image sensor */
class cSensorImage  :   public cObj2DelAtEnd,
                        public cObjWithUnkowns<tREAL8>
{
     public :
          cSensorImage(const std::string & aNameImage);

          //  Allocators  , return nulltr if dont exist

	  virtual ~cSensorImage();

	  /// create a sensor in a new coordinate system, default error
	  virtual cSensorImage * SensorChangSys(const std::string & aDir, cChangeSysCo &) const ;

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
	 
         /*  The most fundamental method, theoretically should be sufficient, when meaning full (ie non orthocentric) 
            orientation must go from sensor to the scene */
        
         virtual tSeg3dr  Image2Bundle(const cPt2dr &) const =0;
	 /// Basic method  GroundCoordinate ->  image coordinate of projection
         virtual cPt2dr Ground2Image(const cPt3dr &) const = 0;

	 ///  Coordinate system, default is undef "LocalNONE"
         // virtual std::string  CoordinateSystem() const;

	 /// Can we manipulate  Image & Depth -> 3d, default false, true for Central Persp
	 virtual bool  HasImageAndDepth() const;
	 ///  add the the depth (to see if have a default with bundle+Gr2Ima), default error
         virtual cPt3dr Ground2ImageAndDepth(const cPt3dr &) const ;
         /// Invert of Ground2ImageAndDepth, default error
         virtual cPt3dr ImageAndDepth2Ground(const cPt3dr &) const ;
	 /// Facility for calling ImageeAndDepth2Ground(const cPt3dr &)
         cPt3dr ImageAndDepth2Ground(const cPt2dr &,const double & ) const;

	 ///  add the the Z, not virtual just a facility using Ground2Image
         cPt3dr Ground2ImageAndZ(const cPt3dr &) const ;
         /// Invert of Ground2ImageAndZ, default use bundle, RPC for ex have a specialize method
         virtual cPt3dr ImageAndZ2Ground(const cPt3dr &) const ;
	 ///  Does it know its Z-validity interval
	 virtual bool  HasIntervalZ() const;
	 /// Default error
	 virtual cPt2dr GetIntervalZ() const;

	 ///  return GSD for a given point, in plane fronto // (use Ground2ImageAndDepth)
         tREAL8 Gen_GroundSamplingDistance(const cPt3dr &) const ;
	 ///  return GSD for a given point, in plane X,Y // (use Ground2ImageAndZ) 
         tREAL8 Horiz_GroundSamplingDistance(const cPt3dr &) const ;

	 /// Compute 3D point by bundle intersection
	  cPt3dr PInterBundle(const cHomogCpleIm & aCple,const cSensorImage &) const;
          /// average residual in pixel of bundle intersection
	  tREAL8 PixResInterBundle(const cHomogCpleIm & aCple,const cSensorImage &) const;


	         //  -------------------  Jacobian ---------------------------------
		 
	  /**  Compute the gradient of the projection function  R3->R2  Ground->Image, 
	   * def use finite diff;  result component I and J of differential */
	  virtual tProjImAndGrad  DiffGround2Im(const cPt3dr &) const;

	  /** For test purpose, we may wish to compute jacobian by finite difference even if DiffGround2Im was overloaded */
	  tProjImAndGrad  DiffG2IByFiniteDiff(const cPt3dr &) const;

	  /// Epislon-value for computing finite difference, def => ERROR
	  virtual cPt3dr  EpsDiffGround2Im(const cPt3dr &) const ;

	 // =================   Visibility/validity   ===========================

	 /// Indicate how much a point belongs to sensor visibilty domain
         virtual double DegreeVisibility(const cPt3dr &) const =0 ;
	 /// Indicacte how much a 2 D points belongs to definition of image frame, defautlt use PixelDomain
         virtual double DegreeVisibilityOnImFrame(const cPt2dr &) const ;
	 /// Indicate if a point belongs to sensor visibilty domain (threshold DegreeVisibility)
         bool IsVisible(const cPt3dr &) const  ;
	 /// Indicacte how a 2 D points belongs to definition of image frame (threshold DegreeVisibilityOnImFrame)
         bool IsVisibleOnImFrame(const cPt2dr &) const  ;
	 /// 2d & 3d are visible
         bool PairIsVisible(const cPair2D3D &) const  ;

	 // =================   Generation of points & correspondance   ===========================

	 /// return a set point regulary sampled (+/-) on sensor, take care of frontier, default is as simple grid
         virtual std::vector<cPt2dr>  PtsSampledOnSensor(int aNbByDim,tREAL8 aEpsMargRel=0.05)  const ;
	 ///  return artificial/synthetic correspondance , with vector of depth / Z
	 cSet2D3D  SyntheticsCorresp3D2D (int aNbByDim,std::vector<double> & aVecDepth,bool IsDepthOrZ,tREAL8 aEpsMargRel=0.05) const;
	 ///  call variant with vector, depth regularly spaced  depth / Z
	 cSet2D3D  SyntheticsCorresp3D2D (int aNbByDim,int aNbDepts,double aD0,double aD1,bool IsDepthOrZ,tREAL8 aEpsMargRel=0.05) const;


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
	 /// method for saving oblet, def error
	 virtual void ToFile(const std::string &) const ;

	 // --------------------   Method for computing center of projection, or approximation  --------------------
	
	 ///  For stenope camera return center, for other best approx
	 virtual cPt3dr  PseudoCenterOfProj() const = 0;
	 ///  For stenope camera return center, for other nullptr, default return null ptr, its not a stenope
	 virtual const cPt3dr * CenterOfPC() const ;

	 /// Return if any the center of footprint 
	 virtual  const cPt3dr *  CenterOfFootPrint() const;


         //  method used in push-broom perturbation model

          /// Extract the pose a line of sensor, meaningfull for push-broom , but can be used in other  context
          tPoseR   GetPoseLineSensor(tREAL8 aXorY,bool LineIsX,int aNbSample,bool*IsOk=0,std::vector<double>*aResCD=0) const;
          /**  Once computed the Orientation of line, compute the differential of proj relatively to WPK
               EpsXYZ and EpsWPK are used for computing derivative with relative differences */
           cDenseMatrix<tREAL8> CalcDiffProjRot(const cPt3dr & aPt,const tPoseR &,const cPt3dr & aEpsXYZ,const tREAL8 & aEpsWPK) const;

	       //  =============   Method used in bundle adjsutment , default error for 3 virtual =============

	 /// Return the calculator, adapted to the type, for computing colinearity equation, def=error
         virtual cCalculator<double> * CreateEqColinearity(bool WithDerives,int aSzBuf,bool ReUse);
	 ///  cObjWithUnkowns , default error
	 void PutUknowsInSetInterval() override;
	 /// If the camera has its own "obs/cste" (like curent rot for PC-Cam) that's the place to say it
	 virtual  void PushOwnObsColinearity(std::vector<double> &,const cPt3dr & aPGround) ;
	 
	 // Create if new, and memorize the colinearity equation
         cCalculator<double> * SetAndGetEqColinearity(bool WithDerives,int aSzBuf,bool ReUse);
	 // suppose it was init (and assert it)
         cCalculator<double> * GetEqColinearity();

	 // Acces to coordinate system of the sensor if exist

	 bool  HasCoordinateSystem() const ; ///<  Was the coordinate system provided ?
	 const  std::string & GetCoordinateSystem() const ; ///< Get it if exist (and error else ...)
	 void SetCoordinateSystem(const std::string&) ; ///< Fix the coordinate system
         std::optional<std::string> &  OptCoordinateSystem() ; ///< redondant with 3 above, but usefull for serialization
	 void TransferateCoordSys(const cSensorImage & aSI); ///< Transferat coordinate sys  aSI if it has any
         static const std::string TagCoordSys;
	 						   
         //  Cast to possible heriting class
         virtual bool  IsSensorCamPC() const  ;
         virtual const cSensorCamPC * GetSensorCamPC() const;
         virtual cSensorCamPC * GetSensorCamPC() ;
         /// generete a user error 
         const cSensorCamPC * UserGetSensorCamPC() const;
         cSensorCamPC * UserGetSensorCamPC() ;

     private :
          cSensorImage(const cSensorImage &) = delete;

	 std::string                                   mNameImage;
         cCalculator<double> *                         mEqColinearity;  // memo equation, can be nullptr (for pure pose)
	 bool                                          mEqCIsInit;      // memo if init of "mEqColinearity" was done
	 /// coordinate system, use optionnal (instead of def value as NONE) as it is an opportunity to test back compatibility
	 std::optional<std::string>                    mNameSysCo;  
								
	 // static std::map<std::string,cSensorImage*>  mDicoSensor;
	 // static int                                  mNum;
};


/**  Interfac to make sensor a 3d-mapping, using Ground2ImageAndDepth function */
class cSIMap_Ground2ImageAndProf : public tIMap_R3
{
       public :
           cSIMap_Ground2ImageAndProf(cSensorImage *);
	   cPt3dr Value(const cPt3dr &) const;   ///< 3D ->  2D+depth
	   cPt3dr Inverse(const cPt3dr &) const;  ///< 2D+depth -> 3D
       private :
           cSensorImage * mSI;
};

/**  Interfac to make sensor a 3d-mapping, using Ground2ImageAndZ function */
class cSIMap_Ground2ImageAndZ : public tIMap_R3
{
       public :
           cSIMap_Ground2ImageAndZ(cSensorImage *);
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

	  /// Input Orientation as mandatory paramaters , def 4 dest : mDirIn
          tPtrArg2007     ArgDirInMand(const std::string & aMes="");
          tPtrArg2007     ArgDirInMand(const std::string & aMes,std::string * aDest) ;  
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
	  bool  DirInIsInit() const;  ///< Was initialized and not set to NONE
	  bool  DirInIsNONE() const;  ///< Was initialized && Set to NONE
	  bool  DirOutIsInit() const;
          void  AssertDirInIsInit() const;
          void  AssertDirOutIsInit() const;
          bool CheckDirExists(bool In, bool DoError=false) const;

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
          cDirsPhProj &   DPOriTriplets(); ///< Accessor
	  cDirsPhProj &   DPRadiomData(); ///< Accessor
	  cDirsPhProj &   DPRadiomModel(); ///< Accessor
	  cDirsPhProj &   DPMeshDev(); ///< Accessor
	  cDirsPhProj &   DPMask(); ///< Accessor
	  cDirsPhProj &   DPGndPt3D(); ///< Accessor
	  cDirsPhProj &   DPGndPt2D(); ///< Accessor
	  cDirsPhProj &   DPMetaData();    ///<  Accessor
	  cDirsPhProj &   DPTieP();    ///<  Accessor
	  cDirsPhProj &   DPMulTieP();    ///<  Accessor
	  cDirsPhProj &   DPBlockInstr();    ///<  Accessor  // RIGIDBLOC
	  cDirsPhProj &   DPRigBloc();    ///<  Accessor  // RIGIDBLOC
	  cDirsPhProj &   DPClinoMeters();    ///<  Accessor  // RIGIDBLOC
	  cDirsPhProj &   DPTopoMes();    ///<  Accessor  // TOPO
	  cDirsPhProj &   DPMeasuresClino();    ///<  Accessor  // RIGIDBLOC
      cDirsPhProj &   DPStaticLidar();    ///<  Accessor  // STATIC LIDAR
				    
	  const cDirsPhProj &   DPOrient() const; ///< Accessor
      const cDirsPhProj &   DPOriTriplets() const; ///< Accessor
	  const cDirsPhProj &   DPRadiomData() const; ///< Accessor
	  const cDirsPhProj &   DPRadiomModel() const; ///< Accessor
	  const cDirsPhProj &   DPMeshDev() const; ///< Accessor
	  const cDirsPhProj &   DPMask() const; ///< Accessor
	  const cDirsPhProj &   DPGndPt3D() const; ///< Accessor
	  const cDirsPhProj &   DPGndPt2D() const; ///< Accessor
	  const cDirsPhProj &   DPMetaData() const;    ///<  Accessor
	  const cDirsPhProj &   DPTieP() const;    ///<  Accessor
	  const cDirsPhProj &   DPMulTieP() const;    ///<  Accessor
	  const cDirsPhProj &   DPBlockInstr()const;    ///<  Accessor  
	  const cDirsPhProj &   DPRigBloc() const;    ///<  Accessor  
	  const cDirsPhProj &   DPClinoMeters() const;    ///<  Accessor 
	  const cDirsPhProj &   DPTopoMes() const;    ///<  Accessor  
	  const cDirsPhProj &   DPMeasuresClino() const;    ///<  Accessor
          const cDirsPhProj &   DPStaticLidar() const;    ///<  Accessor


	  // Sometime we need several dir of the same type, like "ReportPoseCmp", or RefPose in bundle
	  cDirsPhProj * NewDPIn(eTA2007 aType,const std::string & aDirIn);

	  const std::string &   DirPhp() const;   ///< Accessor
	  const std::string &   DirVisu() const;   ///< Accessor
	  const std::string &   DirVisuAppli() const;   ///< Accessor
	  const std::string &   DirSysCo() const;   ///< Accessor
          tPtrArg2007           ArgChSys(bool DefaultUndefined=false);
	  /// To fix the "cur" sys co, its In,Out, or InOut, if both and diff use ArgChSys
          tPtrArg2007           ArgSysCo();

          //===================================================================
          //==================   ORIENTATION      =============================
          //===================================================================

          const std::string &   DirImportInitOri() const;   ///< Accessor
          bool IsOriInDirInit() const;
               //  Read/Write
          void SaveSensor(const cSensorImage &) const; ///< Save camera using OutPut-orientation
          void SaveCamPC(const cSensorCamPC &) const; ///< Save camera using OutPut-orientation
	  void SaveCalibPC(const  cPerspCamIntrCalib & aCalib) const;  ///< Save calibration using  OutPut-orientation

	  cSensorCamPC * ReadCamPC(const std::string &,bool ToDeleteAutom,bool SVP=false) const; ///< Create Camera using Input orientation
	  cSensorCamPC * ReadCamPC(const cDirsPhProj&,const std::string &,bool ToDeleteAutom,bool SVP=false) const; ///< Create Camera using Input orientation

           /// sometime we are only  interested by the pose of the camera
           tPoseR ReadPoseCamPC(const std::string & aNameIm,bool * SVP=nullptr) const;


	  /// Load a sensor, try different type (will add RPC , and others ?) use autom delete (dont need to delete it)
	  void ReadSensor(const std::string &NameIm,cSensorImage* &,cSensorCamPC * &,bool ToDeleteAutom,bool SVP=false) const;
	 
	  /// return the generic sensor, use autom delete (dont need to delete it)
	  cSensorImage* ReadSensor(const std::string  &aNameIm,bool ToDeleteAutom,bool SVP=false) const;
	  /// same as "ReadSensor" but do it from another folder than the standard input one
	  cSensorImage* ReadSensorFromFolder(const std::string  & aFolder,const std::string  &aNameIm,bool ToDeleteAutom,bool SVP=false) const;


	      // Internal Calibration  

	  std::string  StdNameCalibOfImage(const std::string aNameIm) const;
	  std::string  FullDirCalibIn() const;
	  std::string  FullDirCalibOut() const;
	  
	  /// read Pose file  and extract the name of internal  calibration
          cPerspCamIntrCalib *  InternalCalibFromImage(const std::string &aNameIm) const;
	  ///  compute the standard name of calibration before reading it
	  cPerspCamIntrCalib *  InternalCalibFromStdName (const std::string aNameIm,bool isRemanent=true) const;

    //===================================================================
    //==================   ORIENTATION OF TRIPLETS    ==================
    //===================================================================
    void SaveTriplets(const cTripletSet&,bool useXmlraterThanDmp=true) const;
    cTripletSet * ReadTriplets() const;

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

	  // read masq of image, if OkNoMasq accept masq dont exist, return a masq full or empty if file dont exist (depend DefVal)
	  cIm2D<tU_INT1>  MaskWithDef(const std::string & aNameImage,const cBox2di & aBox,bool DefVal,bool OkNoMasq=true) const;
	  // read masq , generate error if dont exist
	  cIm2D<tU_INT1>  MaskOfImage(const std::string & aNameImage,const cBox2di & aBox) const;
	  
	 //===================================================================
         //==================    PointsMeasures  =============================
	 //===================================================================

	  void SaveMeasureIm(const cSetMesPtOf1Im & aSetM) const;
	  ///  Does the measure exist
	  bool HasMeasureIm(const std::string & aNameIm,bool InDir=true) const;
          /// Does it exist for a specific folder
          bool HasMeasureImFolder(const std::string & aFolder,const std::string & aNameIma) const;

          /// return from Std Dir, can be out in case of reload
	  cSetMesPtOf1Im LoadMeasureIm(const std::string &,bool InDir=true) const;

	  /// Load the measure image from a specified folder, usefull when multiple folder
	  cSetMesPtOf1Im LoadMeasureImFromFolder(const std::string & aFolder,const std::string &) const;

         void LoadGCP3D(cSetMesGndPt&, cMes3DDirInfo *aMesDirInfo=nullptr, const std::string & aPatFiltrFile="", const std::string & aFiltrNameGCP="",
                      const std::string & aFiltrAdditionalInfoGCP="") const;
	 ///  For reading GCP from folder potentially != of standard input measures, can add missing points from topo obs
         void LoadGCP3DFromFolder(const std::string & aFolder, cSetMesGndPt&,
                                  cMes3DDirInfo * aMesDirInfo,
                                const std::string & aPatFiltrFile="", const std::string & aFiltrNameGCP="",
                                const std::string & aFiltrAdditionalInfoGCP="" ) const;
          // if SVP && file doesnt exist, do nothing
	  void LoadIm(cSetMesGndPt&,const std::string & aNameIm,cMes2DDirInfo * aMesDirInfo=nullptr, cSensorImage * =nullptr,bool SVP=false) const;
          ///  When dont read from the standard input 
	  void LoadImFromFolder(const std::string & aFolder, cSetMesGndPt&, cMes2DDirInfo *aMesDirInfo, const std::string & aNameIm,
                                cSensorImage * =nullptr, bool SVP=false) const;
	  void LoadIm(cSetMesGndPt&, cMes2DDirInfo *aMesDirInfo, cSensorImage & ) const;

	  void SaveGCP3D(const cSetMesGnd3D&aMGCP3D, const std::string &aDefaultOutName="", bool aDoAddCurSysCo=false) const; // default out name for measures without cMes3DDirInfo
	  cSetMesGnd3D LoadGCP3DFromFolder(const std::string &) const;
	  cSetMesGnd3D LoadGCP3D() const;

	  /// Name of the file, usefull if we need to test existence before doing anything
	  std::string NameMeasureGCPIm(const std::string & aNameIm,bool isIn) const;


	  /// Pattern for GCP file, if "" return default  = "cSetMesGCP::ThePrefixFiles.*.xml"
	  std::string GCPPattern(const std::string & aArgPatFiltr) const;
          std::vector<std::string>  ListFileGCP(const std::string & aArgPatFiltr) const;
	  void CpGCPPattern(const std::string& aDirIn,const std::string & aDirOut,const std::string & aArgPatFiltr="") const;
	  void CpGCP() const;
	  void CpMeasureIm() const;

	  /// For a givgen image, return 3D-2D corresp, using LoadGCP&LoadIm
	  cSet2D3D  LoadSet32(const std::string & aNameIm) const;

          void SaveAndFilterAttrEll(const cSetMesPtOf1Im &  aSetM,const std::list<std::string> & ToRem)   const ;


	      // ---------------  Segment in/out ----------------------------------------
	  std::string  NameFileLines(const std::string & aNameIm,bool isIn) const;
	  bool         HasFileLines(const std::string & aNameIm)  const;  ///<  Does exist the file with lines ?
	  bool         HasFileLinesFolder(const std::string &aFolder,const std::string & aNameIm)  const; ///< Idem with spec folder
	  void         SaveLines(const cLinesAntiParal1Im &) const;
	  cLinesAntiParal1Im  ReadLines(const std::string & aNameIm) const; ///< Read lines from std folder
	  cLinesAntiParal1Im  ReadLinesFolder(const std::string &aFolder,const std::string & aNameIm) const; ///< Idem with spec folder



	  
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
                                            cPt2dr   aPP=cPt2dr(0.5,0.5), bool SVP=false,bool IsFraser=true);

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
	 void  ReadMultipleTieP(cVecTiePMul&,const std::string &aNameIm,bool SVP =false ) const;
         ///  When dont read from the standard input 
	 void  ReadMultipleTiePFromFolder(const std::string & aFolder,cVecTiePMul&,const std::string &,bool SVP =false ) const;
	 bool HasNbMinMultiTiePoints(const std::string & aNameIm,size_t aNbMin,bool AcceptNoDirIn =false) const;

	 //===================================================================
         //==================    Coord Sys           =========================
	 //===================================================================

         //  ======== [1]  Sysco saved in "MMVII-PhgrProj/SysCo"
         void  SaveSysCo(tPtrSysCo,const std::string&,bool OnlyIfNew=false) const;
         tPtrSysCo ReadSysCo(const std::string &aName, bool aDebug=false) const;


         tPtrSysCo CreateSysCoRTL(const cPt3dr & aOrig,const std::string &aNameRef,bool SVP=false) const;
	 std::string  FullNameSysCo(const std::string &aName,bool SVP=false) const;

         // return  identity if Vec not init
	 cChangeSysCo ChangSysCo(const std::vector<std::string> &,tREAL8 aEpsDif=0.1);
	 // Return idenitity if aS1==aS2
	 cChangeSysCo ChangSysCo(const std::string aS1,const std::string aS2) const;

                  //  ======== [1]  Sysco saved in "MMVII-PhgrProj/Ori/"  or "MMVII-PhgrProj/PointsMeasure//"
         std::string  NameCurSysCo(const cDirsPhProj &,bool IsIn) const;
         tPtrSysCo  CurSysCo(const cDirsPhProj &,bool SVP=false, bool IsIn=true) const;
         tPtrSysCo  CurSysCoOri(bool SVP=false, bool IsIn=true) const;
         tPtrSysCo  CurSysCoGCP(bool SVP=false, bool IsIn=true) const;
         void SaveCurSysCo(const cDirsPhProj &,tPtrSysCo) const ;
         void SaveCurSysCoOri(tPtrSysCo) const ;
         void SaveCurSysCoGCP(tPtrSysCo) const ;
         void SaveStdCurSysCo(bool IsOri) const; /// save the Cur Sysco in Orient/GCP
         void CpSysCoIn2Out(bool OriIn,bool OriOut) const;  // bool : Ori/GCP   do it only if exist, else no error
         std::string  getDirSysCo() const { return mDirSysCo; }

         const cChangeSysCo & ChSysCo() const;
         cChangeSysCo & ChSysCo() ;
         bool  ChSysCoIsInit() const;
         void  AssertChSysCoIsInit() const;

         cSysCo & SysCo() ;
         const cSysCo & SysCo() const ;
         bool  SysCoIsInit() const;
         void  AssertSysCoIsInit() const;

         /// If ChSys.Target() is not init, set it to RTL, considering center is expressed in SysOrigin
	 void InitSysCoRTLIfNotReady(const cPt3dr & aCenter);

	 //===================================================================
         //==================   Clinometers           ========================
	 //===================================================================
	
	 /// Standard name for clino file using DPClinoMeters, in or out
	 std::string NameFileClino(const std::string &aNameCam ,bool Input, const std::string aClinoName) const;
	 /// Save clinometer calib in santdard out folder of DPClinoMeters
	 void SaveClino(const cCalibSetClino &) const;
	 /// Is there  clinometer in santdard input folder of DPClinoMeters ?
	 bool HasClinoCalib(const cPerspCamIntrCalib &, const std::string aClinoName) const;
	 /**  Read the clinometers calib in standard input folder of DPClinoMeters, create a dyn objec because
	  *  probably "cCalibSetClino" will evolve in a not copiable object*/
	 cOneCalibClino * GetClino(const cPerspCamIntrCalib &, const std::string aClinoName) const;
	 void  ReadGetClino(cOneCalibClino&,const cPerspCamIntrCalib &, const std::string aClinoName) const;
	 cCalibSetClino  ReadSetClino(const cPerspCamIntrCalib &, const std::vector<std::string> &aClinoName) const;

	 /// Standard name for file of measures clino 
	 std::string NameFileMeasuresClino(bool Input,const std::string & aNameFile="" ) const;
	 void SaveMeasureClino(const cSetMeasureClino &) const;
	 void ReadMeasureClino(cSetMeasureClino &,const std::string * aPatSel=nullptr) const;
	 cSetMeasureClino ReadMeasureClino(const std::string * aPatSel=nullptr) const;

	 //===================================================================
         //==================   Rigid Bloc           =========================
	 //===================================================================
	 
	         // RIGIDBLOC
	 std::list<cBlocOfCamera *> ReadBlocCams() const;
	 cBlocOfCamera *            ReadUnikBlocCam() const;
	 void   SaveBlocCamera(const cBlocOfCamera &) const;
              
	       //  New formalisation
	 std::string   NameRigBoI(const std::string &,bool isIn) const;
	 /// read a new bloc from existing name, if SVP and dont exist return block empty, else error
	 cIrbCal_Block*  ReadRigBoI(const std::string &,bool SVP=false) const; 
	 void   SaveRigBoI(const cIrbCal_Block &) const;
     std::vector<std::string>  ListBlockExisting() const;

         //===================================================================
         //==================   Topo Mes           =========================
	 //===================================================================

	         // TOPO
     std::vector<std::string> ReadTopoMes() const;
     void   SaveTopoMes(const cBA_Topo & aBATopo) const;


     //===================================================================
     //==================   Static Lidar         =========================
     //===================================================================

     cStaticLidar * ReadStaticLidar(const cDirsPhProj& aDP,const std::string &aScanName, bool ToDeleteAutom) const; ///< Create Static Lidar

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
	  std::string     mDirVisuAppli;

	  std::string     mDirSysCo;        /// Folder where are stored System of coordinates
          std::string     mNameCurSysCo;      /// Data where we store the system In Or Out if given in std args
          tPtrSysCo       mCurSysCo;          /// Global coord sys
	  std::string     mDirImportInitOri; /// Folder where are stored INIT-al  ORI-entation
          std::vector<std::string>   mNameChSysCo;
          cChangeSysCo          mChSysCo;

	  cDirsPhProj     mDPOrient;
          cDirsPhProj     mDPOriTriplets;         ///> For triplets of images
	  cDirsPhProj     mDPRadiomData;
	  cDirsPhProj     mDPRadiomModel;
	  cDirsPhProj     mDPMeshDev;
	  cDirsPhProj     mDPMask;
	  cDirsPhProj     mDPGndPt3D;         ///<  For ground point, measures/coords in image frame
	  cDirsPhProj     mDPGndPt2D;         ///<  For ground point, measures/coords in ground frame
	  cDirsPhProj     mDPTieP;            ///<  For Homologous point
	  cDirsPhProj     mDPMulTieP;         ///<  For multiple Homologous point
	  cDirsPhProj     mDPMetaData;
	  cDirsPhProj     mDPBlockInstr;       // RIGIDBLOC
	  cDirsPhProj     mDPRigBloc;         // RIGIDBLOC
          cDirsPhProj     mDPClinoMeters;      // +-  resulta of clino calib (boresight)
          cDirsPhProj     mDPMeasuresClino;     // measure (angles) of clino
          cDirsPhProj     mDPTopoMes;         // Topo
          cDirsPhProj     mDPStaticLidar;         // Static Lidar
 					      //

	  std::vector<cDirsPhProj*> mDirAdded;
	  mutable cGlobCalculMetaDataProject *  mGlobCalcMTD;

          cCamDataBase   mCamDataBase;

};
void SaveAndFilterAttrEll(const cPhotogrammetricProject & aPhp,const cSetMesPtOf1Im &  aSetM,const std::set<std::string> & ToRem);

};

#endif  //  _MMVII_SENSOR_H_
