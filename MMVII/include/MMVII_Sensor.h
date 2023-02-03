#ifndef  _MMVII_SENSOR_H_
#define  _MMVII_SENSOR_H_

#include "MMVII_Mappings.h"

namespace MMVII
{

/** \file MMVII_Sensor.h
    \brief Interface class for sensors

   Most probably this will evolve a lot, with several reengenering 
  phases. 

   For now, it will essentially contain interfaces to existing MMV1 cameras.

*/

struct cPair2D3D;
struct cSet2D3D;
class  cSensorImage;
class  cDataPixelDomain ;
class  cPixelDomain;
class  cSensorCamPC;
class  cPhotogrammetricProject;
class  cSIMap_Ground2ImageAndProf ;


/** class for representing  a 3D point paired with it 2d image projection */
 
struct  cPair2D3D
{
     public :
          cPair2D3D(const cPt2dr &,const cPt3dr &);
          cPt2dr mP2;
          cPt3dr mP3;
};
 
struct  cWeightedPair2D3D : public cPair2D3D
{
     public :
          cWeightedPair2D3D(const cPair2D3D&,double aWeight=1.0);
          cWeightedPair2D3D(const cPt2dr&,const cPt3dr&,double aWeight=1.0);

	  double mWeight;
};



/**  class for representing  set of pairs 2-3  */
struct cSet2D3D
{
     public :
         typedef cWeightedPair2D3D                tPair;
         typedef std::vector<tPair>   tCont2D3D;

         void AddPair(const tPair &);
         void AddPair(const cPt2dr&,const cPt3dr&,double aWeight=1.0);

         const tCont2D3D &  Pairs() const;
         void  Clear() ;

	 /// compute  weighted centroid
	 cWeightedPair2D3D  Centroid() const;

	 /// subsract a pair to all
	 void Substract(const cPair2D3D&);
     private :
        tCont2D3D  mPairs;
};

/*  base-class  4 all image sensor */

class cSensorImage  :  public cObjWithUnkowns<tREAL8>
{
     public :

         cSensorImage(const std::string & aNameImage);

	 /// Basic method  GroundCoordinate ->  image coordinate of projection
         virtual cPt2dr Ground2Image(const cPt3dr &) const = 0;
	 /// Indicate if a point belongs to sensor visibilty domain
         virtual double Visibility(const cPt3dr &) const =0 ;

	 /// Indicacte how a 2 D points belongs to definition of image frame
         virtual double VisibilityOnImFrame(const cPt2dr &) const =0 ;

	 ///  add the distance to bundle,to see if have a default with bundle+Gr2Ima
         //virtual cPt3dr Ground2ImageAndDist(const cPt3dr &) const = 0;
	 
	 ///  add the the depth (to see if have a default with bundle+Gr2Ima)
         virtual cPt3dr Ground2ImageAndDepth(const cPt3dr &) const = 0;
         /// Invert of Ground2ImageAndDepth
         virtual cPt3dr ImageAndDepth2Ground(const cPt3dr &) const = 0;
	 /// Facility for calling ImageeAndDepth2Ground(const cPt3dr &)
         cPt3dr ImageAndDepth2Ground(const cPt2dr &,const double & ) const;

	 /// return a set point regulary sampled (+/-) on sensor, take care of frontier
         virtual std::vector<cPt2dr>  PtsSampledOnSensor(int aNbByDim)  const = 0;

	 ///  return artificial/synthetic correspondance , with vector of depth
	 cSet2D3D  SyntheticsCorresp3D2D (int aNbByDim,std::vector<double> & aVecDepth) const;
	 ///  call variant with vector, depth regularly spaced
	 cSet2D3D  SyntheticsCorresp3D2D (int aNbByDim,int aNbDepts,double aD0,double aD1) const;


         double SqResidual(const cPair2D3D &) const;  ///< residual Proj(P3)-P2 , squared for efficiency
         double AvgResidual(const cSet2D3D &) const;  ///< avereage on all pairs, not squared

	 const std::string & NameImage() const;   ///< accessor
	 void SetNameImage(const std::string &);  ///< used when reading from file

	 // --------------------   methods to compute names of an orientation --------------------
	 static std::string  PrefixName() ;  ///<  Prefix to all file for ori of 1 images
	 ///  Compute the name from the specific prefix of subtype and image name, need a static access in creation
	 static  std::string  NameOri_From_PrefixAndImage(const std::string & aPrefix,const std::string & aNameImage);
	 ///  Simply return the name
	 std::string NameOriStd() const ;
	 ///  Prefix of the subtype
	 virtual std::string  V_PrefixName() const = 0  ;

     private :
	 std::string                                    mNameImage;
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
class cMedaDataImage
{
      public :
          tREAL8  Aperture() const;
          tREAL8  FocalMM() const;
          tREAL8  FocalMMEqui35() const;
          cMedaDataImage(const std::string & aNameIm);
          cMedaDataImage();
      private :

          std::string    mCameraName;
          tREAL8         mAperture;
          tREAL8         mFocalMM;
          tREAL8         mFocalMMEqui35;
          std::string    mNameImage;
};


/**   Class for sharind code related to management the folder for one kind (=Ori,Homol,Radiom...) of "objects"
 *    Used by cPhotogrammetricProject
 */

class cDirsPhProj
{
     public :
          cDirsPhProj(eTA2007 aMode,cPhotogrammetricProject & aPhp);
          void Finish();

          tPtrArg2007     ArgDirInMand(const std::string & aMes="") ;  ///< Input Orientation as mandatory paramaters
          tPtrArg2007     ArgDirInOpt() ;   ///< Input Orientation as optional paramaters
          tPtrArg2007     ArgDirOutMand();  ///< Output Orientation as mandatory paramaters
          tPtrArg2007     ArgDirOutOpt() ;   ///< Input Orientation as optional paramaters

          void  SetDirIn(const std::string&) ; ///< Modifier, use in case many out params were saved in a xml,like with MeshImageDevlp
          const std::string & DirIn() const;   ///< Accessor
          const std::string & DirOut() const;   ///< Accessor
          const std::string & FullDirIn() const;   ///< Accessor
          const std::string & FullDirOut() const;   ///< Accessor

	  bool  DirInIsInit() const;
	  bool  DirOutIsInit() const;
          void  AssertDirInIsInit() const;
          void  AssertDirOutIsInit() const;

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
                /// some initialisation can be done only once Appli is itself init, method must be calles in mAppli.Exe()
          void FinishInit() ;
	        /// destructor  ,  some object delegates their destruction to this
          ~cPhotogrammetricProject();

	 //========================== ACCESSOR ===============================

          cMMVII_Appli &  Appli(); ///< Accessor
	  cDirsPhProj &   DPOrient(); ///< Accessor
	  cDirsPhProj &   DPRadiom(); ///< Accessor
	  cDirsPhProj &   DPMeshDev(); ///< Accessor
	  const cDirsPhProj &   DPOrient() const; ///< Accessor
	  const cDirsPhProj &   DPRadiom() const; ///< Accessor
	  const cDirsPhProj &   DPMeshDev() const; ///< Accessor

	 //===================================================================
         //==================   ORIENTATION      =============================
	 //===================================================================
	 
               //  Read/Write
          void SaveCamPC(const cSensorCamPC &) const; ///< Save camera using OutPut-orientation
	  cSensorCamPC * AllocCamPC(const std::string &,bool ToDelete); ///< Create Camera using Input orientation
	  /// For now read PC and extract it (later use xif as in MMV1 in case PC does not exist)
          cPerspCamIntrCalib *  AllocCalib(const std::string &);


	 //===================================================================
         //==================   RADIOMETRY       =============================
	 //===================================================================

               //     --------   Save Data ---------------------
          void SaveRadiomData(const cImageRadiomData &) const; ///< Save camera using OutPut-orientation
          void SaveCalibRad(const cCalibRadiomIma &) const; ///< Save radiom-calib using OutPut-orientation

               //     --------   Read Data ---------------------
	  cImageRadiomData * AllocRadiomData(const std::string &) const; ///< Read radiometric data for 1 image
	  cCalibRadiomIma * AllocCalibRadiomIma(const std::string &) const; ///< Read radiom calib of 1 image

	  ///  Name of radiometric calibration with a  radial model , hypothesis : depends of internal calibration
	  ///  +  metadata (aperture)
          std::string NameCalibRadiomSensor(const cPerspCamIntrCalib &,const cMedaDataImage &) const;

	 //===================================================================
         //==================   META-DATA       ==============================
	 //===================================================================
          cMedaDataImage GetMetaData(const std::string &) const;


      private :
          cPhotogrammetricProject(const cPhotogrammetricProject &) = delete;

          cMMVII_Appli &  mAppli;
          std::string     mFolderProject;

	  cDirsPhProj     mDPOrient;
	  cDirsPhProj     mDPRadiom;
	  cDirsPhProj     mDPMeshDev;

	  std::list<cSensorCamPC*>  mLCam2Del; 
};



};

#endif  //  _MMVII_SENSOR_H_
