#ifndef  _MMVII_BUNDLEADJUSTMENT_H_
#define  _MMVII_BUNDLEADJUSTMENT_H_

#include "MMVII_PCSens.h"
#include "MMVII_MMV1Compat.h"
#include "MMVII_DeclareCste.h"
#include "MMVII_BundleAdj.h"
#include "MMVII_Clino.h"
#include "MMVII_SysSurR.h"
#include "MMVII_StaticLidar.h"

using namespace NS_SymbolicDerivative;
namespace MMVII
{

class cBA_Topo;
class cMMVII_BundleAdj;
class cBA_LidarPhotogra;
class cBA_TieP;
class cBA_GCP;
class cBA_Clino;
class cBA_BlocRig;

class cUK_Line3D_4BA;
class cBA_BlockInstr;

/**  "Standard" weighting classes, used the following formula
 *
 *    W(R) =
 *          0 if R>Thrs
 *          1/Sigma0^2  * (1/(1+ (R/SigmaAtt)^Exp)) 
 *
 */

class cStdWeighterResidual : public cResidualWeighter<tREAL8>
{
     public :
         // aThr<0 => dont use
	 cStdWeighterResidual(tREAL8 aSGlob,tREAL8 aSigAtt,tREAL8 aThr,tREAL8 aExp);
	 cStdWeighterResidual(const std::vector<tREAL8> & aVect,int aK0);
	 cStdWeighterResidual();

	 tREAL8   SingleWOfResidual(const tStdVect &) const ;
	 tREAL8   SingleWOfResidual(const cPt2dr &) const ;

	 tStdVect WeightOfResidual(const tStdVect &) const override;

     private :
       // W(R) =  1/Sigma0^2  * (1/(1+ (R/SigmaAtt)^Exp))  ;  W(R) =0 if R>Thrs
         tREAL8   mWGlob;
	 bool     mWithAtt;
	 tREAL8   mSig2Att;
	 bool     mWithThr;
	 tREAL8   mSig2Thrs;
	 tREAL8   mExpS2;
};

// RIGIDBLOC
class cBA_BlocRig
{
     public :

        cBA_BlocRig(const cPhotogrammetricProject &,const std::vector<double> & aSigma,const std::vector<double> & aSigmRat);
	~cBA_BlocRig();
        void    AddCam(cSensorCamPC * aCam);

        // The system must be aware of all the unknowns
        void  AddToSys(cSetInterUK_MultipeObj<tREAL8> &);

        // fix the variable that are frozen
        void SetFrozenVar(cResolSysNonLinear<tREAL8> &)  ;

        //  Do the kernel job : add rigidity constraint to the system
        void AddRigidityEquation(cResolSysNonLinear<tREAL8> &);

	void Save();
     private :

        // do the job for one bloc
        void OneBlAddRigidityEquation(cBlocOfCamera&,cResolSysNonLinear<tREAL8> &);
        // do the job for one pair of poses, return residual x=Pt y=Rot  z=Ok
        cPt3dr OnePairAddRigidityEquation(size_t aKSync,size_t aKBl1,size_t aKBl2,cBlocOfCamera&,cResolSysNonLinear<tREAL8> &);

	const cPhotogrammetricProject &mPhProj;
        std::list<cBlocOfCamera *>   mBlocs;
        std::vector<double>          mSigma;
        std::vector<double>          mWeight;
        bool                         mAllPair;  // Do we use all pair or only pair with master
        cCalculator<double> *        mEqBlUK;
        std::vector<double>          mSigmaRat;
        std::vector<double>          mWeightRat;
        cCalculator<double> *        mEqRatt;

};



class cClinoMes1Cam : public cMemCheck
{
     // Object with clino measures for one camera
     
     public :
          cClinoMes1Cam
          (
              const cSensorCamPC * aCam,                    // camera-> only the pose is useful
              const std::vector<std::string> &aVClinoName,  // vector of clinometer names
              const std::vector<tREAL8> & aVAngles,         // vector of angles of all clinometer
              const std::vector<tREAL8> & aVWeights,        // vector of weights of all clinometer
              const cPt3dr &   aVerticAbs = {0,0,-1}        // position of vertical in current "absolut" system
          );

          // return the camera
          const cSensorCamPC * Cam() const {return mCam;};

          // push observations : the camera orientation and clinometer measures
          void pushClinoObs(std::vector<double> & aVObs, const std::string aClinoName);

          // push unknowns : the camera orientation axiator
          void pushIndex(std::vector<int> & aVInd);

          // push weights
          void pushClinoWeights(std::vector<double> & aVWeights, const std::string aClinoName);

          // return the clinometers measures
          const std::map<std::string, double> VDir() const {return mVDir;};   

	

     private :
          const cSensorCamPC *   mCam;                 // camera , memorization seems useless
          std::map<std::string, double>     mVDir;     // map : clinometer name, measured position of needle, computed from angles
          std::map<std::string, double>     mWeights;  // map :  clinometer name, weights on clino measures
	     cPt3dr                  mVertInLoc;          // Vertical in  camera system, this is the only information usefull of camera orientation
};


class cClinoWithUK :  public cObjWithUnkowns<tREAL8>, public cMemCheck
{
     
     // Object to compute orientation of clinometer in a camera repere
     // mRot is the rotation matrix
     // mOmega is the axiator, the computed value by least squares to get a new rotation
     
     public :

          // Constructor
	     cClinoWithUK();

          // Constructor with the initial solution and clino name
          cClinoWithUK(
               tRotR aRot,                        // initial solution : relative rotation between clino and camera
               const std::string & aNameClino     // clino name
          ); 

          // Fundamental methos :  the object put it sets on unknowns intervals  in the glob struct
          void PutUknowsInSetInterval() override;

          // update mRot with mOmega
          void OnUpdate() override; 

          void FillGetAdrInfoParam(cGetAdrInfoParam<tREAL8> &) override;

          // push index of unknowns
          void pushIndex(std::vector<int> & aVInd) const;

          // get rotation between one camera and the clinometer
          tRotR Rot() const {return mRot;};

          // get vector axiator, the unknowns of the least squares
          cPt3dr Omega() const {return mOmega;}

          // get the clino name
          const std::string NameClino() const {return mNameClino;}; 
	
     private :
         const std::string    mNameClino;    // Name of the clino
         tRotR          mRot;                // Rotation between one camera and the clinometer. It is also the initial solution
         cPt3dr         mOmega;              // Vector axiator, the unknowns of the least squares
};


// CLINOBLOC
class cBA_Clino : public cMemCheck
{
     // Object to compute Bundle Adjustment on clinometers
     public :

          // Constructor for ClinoBench (set manually clino observations)
          cBA_Clino(
               const cPhotogrammetricProject *aPhProj // photogrammetric project 
          );

          // Add equation with aMeasure observations for one clinometer
          cPt2dr addOneClinoEquation(cResolSysNonLinear<tREAL8> & aSys, cClinoMes1Cam & aMeasure, const std::string aClinoName);

          // Add equations on two boresight matrix and their initial values
          cPt2dr addOneRotEquation(cResolSysNonLinear<tREAL8> & aSys, const std::string aClino1, const std::string aClino2);

          // Add all equations with all measures
          void addEquations(cResolSysNonLinear<tREAL8> & aSys);

          // Add all clino with unknowns to the system
          void AddToSys(cSetInterUK_MultipeObj<tREAL8> & aSet);

          // Froze boresight matrix of clinos described by aPatFrozenClino
          void SetFrozenVar(cResolSysNonLinear<tREAL8> & aSys, const std::string aPatFrozenClino);

          // Push observations for clino formula : initial values of Boresight matrix (9 values), and the vertical in local repere (3 values)
          void pushClinoObs(std::vector<double> & aVObs, const cPt3dr & aCamTr, const std::string aClinoName);

          // Push observations for rot formula : values of two Boresight matrix (2*9 values) and initial relative orientation between these two matrix
          void pushRotObs(std::vector<double> & aVObs, const std::string aClino1, const std::string aClino2);

          // Push index of all clino unknowns for clino formula
          void pushClinoIndex(std::vector<int> & aVInd, const std::string aClinoName);

          // Push index of all clino unknowns for rot formula
          void pushRotIndex(std::vector<int> & aVInd, const std::string aClino1, const std::string aClino2);

          // Push weights for rot formula
          void pushRotWeights(std::vector<double> & aVWeights);

          // Save relative orientation between clinos and reference camera
          void Save();

          // Add a clino observation
          void addClinoMes1Cam(const cClinoMes1Cam & aClinoMes1Cam);

          // Add a cClinoWithUK object
          void addClinoWithUK(const std::string & aClinoName, tRotR & aRot);

          // Get all relative rotations in cClinosWithUK objects. Used in BenchClino only
          std::vector<tRotR>  ClinosWithUKRot() const;

          // Display residuals
          void printRes() const; 

          // Set vector with clino names
          void setVNamesClino(std::vector<std::string> aVNamesClino){mVNamesClino=aVNamesClino;};

          // Add a initial rotation for a clino
          void addInitRotClino(std::string aClinoName, tRotR aRot){mInitRotClino[aClinoName]=aRot;};

          
     private :

          // Read initial boresight matrices computed by ClinoInit
          void readMeasures();                    

	     const cPhotogrammetricProject * mPhProj;               // Photogrammetric project
                                                                 // file to have the same names than in initial solutions file
          std::vector<cClinoMes1Cam>  mVMeasures;                // observations for one image and one clino
          std::vector<std::string> mVNamesClino;                 // clino names
          cCalculator<double> *        mEqBlUK;                  // calculator for clino formula
          cCalculator<double> *        mEqBlUKRot;               // calculator for rot formula
          std::vector<double>          mWeight;                  // weights
          std::map<std::string, cClinoWithUK>    mClinosWithUK;  // map with {clino name, cClinoWithUK object}
          cPt2dr                        mClinoRes;               // Residuals for clino formula
          cPt2dr                        mRotRes;                 // Residuals for rot formula
          std::map<std::string, tRotR>    mInitRotClino;         // map with {clino name, initial rotation}
          std::string                   mCameraName;              // name of the camera
          
};



// class to record data specific to a measurement directory : In/out name, w factor
class cMes3DDirInfo
{
public:
    static cMes3DDirInfo* addMes3DDirInfo(cBA_GCP &aBA_GCP, const std::string & aDirNameIn,
                                            const std::string & aDirNameOut, tREAL8 aSGlob);
    std::string mDirNameIn;
    std::string mDirNameOut;
    tREAL8 mSGlob; // factor, shurred or fixed
protected:
    cMes3DDirInfo(const std::string &aDirNameIn, const std::string &aDirNameOut, tREAL8 aSGlob);
};

// class to record data specific to a measurement directory : In name, weighter
class cMes2DDirInfo
{
public:
    static cMes2DDirInfo* addMes2DDirInfo(cBA_GCP &aBA_GCP, const std::string & aDirNameIn,
                                        const cStdWeighterResidual & aStdWeighterResidual);
    std::string mDirNameIn;
    cStdWeighterResidual mWeighter;
protected:
    cMes2DDirInfo(const std::string &aDirNameIn, const cStdWeighterResidual &aWeighter);
};




class cBA_GCP
{
    friend class cMMVII_BundleAdj;
     public :
	          // - - - - - - - - GCP  - - - - - - - - - - -
          cBA_GCP();
          ~cBA_GCP();
          cBA_GCP(cBA_GCP const&) = delete;
          cBA_GCP& operator=(cBA_GCP const&) = delete;

          void AddGCP3D(cMes3DDirInfo * aMesDirInfo, cSetMesGnd3D &aSetMesGnd3D, bool verbose);
          void AddMes2D(cSetMesPtOf1Im &, cMes2DDirInfo * aMesDirInfo, cSensorImage*, eLevelCheck OnNonExistP=eLevelCheck::Warning);
          const cSetMesGndPt & getMesGCP() const {return mMesGCP;}
          cSetMesGndPt & getMesGCP() {return mMesGCP;}
          std::vector<cMes2DDirInfo*> mAllMes2DDirInfo;
          std::vector<cMes3DDirInfo*> mAllMes3DDirInfo;
          const std::vector<cPt3dr_UK*>  & getGCP_UK() const { return mGCP_UK; }
    protected:
          cSetMesGndPt             mMesGCP; //< initial
          cSetMesGndPt             mNewGCP; //< set of gcp after adjust
          std::vector<cPt3dr_UK*>  mGCP_UK; //< as many elements as mMesGCP, nullptr for shurred points

};

class cBA_TieP
{
     public :
       cBA_TieP(const std::string & aName,cComputeMergeMulTieP*,const cStdWeighterResidual &aRes);
       ~cBA_TieP();
       
       std::string              mName;   // Name of folder 
       cComputeMergeMulTieP *   mMTP;
       cStdWeighterResidual     mTieP_Weighter;
};


/** "Helper" class for cBA_LidarPhotogra : for a given patch in one image, will store all the data on the points*/
class cData1ImLidPhgr
{
     public :
        size_t mKIm;  ///< num of images where the patch is seen
        std::vector<std::pair<tREAL8,cPt2dr>> mVGr; ///< pair of radiometry/gradient, in image,  for each point of the patch
};


enum class cBA_LidarPhotogra_Type
{
    Triangulation,
    Rastersization
};


/**  Class for doing the adjsment between Lidar & Photogra, prototype for now, what will most certainly will need
     to evolve is the weighting policy.
 */

class cBA_LidarPhotogra
{
    public :
       /// constructor, take the global bundle struct + one vector of param
       cBA_LidarPhotogra(cPhotogrammetricProject *aPhProj, cMMVII_BundleAdj&, const std::vector<std::string> & aParam, cBA_LidarPhotogra_Type aType);
       /// destuctor, free interopaltor, calculator ....
       ~cBA_LidarPhotogra();

       /// add observation
       void AddObs();

    private :
       /**  Add observation for 1 Patch of point */
       void Add1Patch(tREAL8 aW,const std::vector<cPt3dr> & aPatch);

       /// Method for adding observations with radiometric differences as similatity criterion
       void AddPatchDifRad(tREAL8 aW,const std::vector<cPt3dr> & aPatch,const std::vector<cData1ImLidPhgr> &aVData) ;

       /// Method for adding observations with Census Coeff as similatity criterion
       void AddPatchCensus(tREAL8 aW,const std::vector<cPt3dr> & aPatch,const std::vector<cData1ImLidPhgr> &aVData) ;

       /// Method for adding observations with Normalized Centred Coefficent Correlation as similatity criterion
       void AddPatchCorrel(tREAL8 aW,const std::vector<cPt3dr> & aPatch,const std::vector<cData1ImLidPhgr> &aVData) ;

       void SetVUkVObs
       (
            const cPt3dr&           aPGround,
            std::vector<int> *      aVIndUk,
            std::vector<tREAL8> &   aVObs,
            const cData1ImLidPhgr & aData,
            int                     aKPt
       );

       cPhotogrammetricProject *      mPhProj;         // Photogrammetric project
       cMMVII_BundleAdj&              mBA;             ///< The global bundle adj structure
       eImatchCrit                    mModeSim;        ///< type of similarity used
       cTriangulation3D<tREAL4> *     mTri;            ///< Triangulation, in fact used only for points
       cStaticLidar *                 mLidarData;      ///< Raster representations of lidar
       cDiffInterpolator1D *          mInterp;         ///< Interpolator, used to extract  Value & Grad of images
       cCalculator<double>  *         mEqLidPhgr;      ///< Calculator used for constrain the pose from image obs
       std::vector<cSensorCamPC *>    mVCam;           ///< Vector of central perspective camera
       std::vector<cIm2D<tU_INT1>>    mVIms;           ///< Vector of images associated to each cam
       cWeightAv<tREAL8,tREAL8>       mLastResidual;   ///< Accumulate the radiometric residual
       std::list<std::vector<int>>    mLPatchesI;      ///< set of patches as index in Tri, consituted by 3D points in a lidar scan
       std::list<std::vector<cPt2di>> mLPatchesP;      ///< set of patches as px in raster, consituted by 3D points in a lidar scan
       bool                           mPertRad;        ///< do we pertubate the radiometry (simulation & test)
       size_t                         mNbPointByPatch; ///< (approximate) required number of point /patch
       double                         mWeight;          ///< weight for observations
       size_t                         mNbUsedPoints;   ///< number of lidar used points
       size_t                         mNbUsedObs;      ///< number of lidar obs used
};


struct  cBundleBlocNamedVar
{
    public :
       std::string mType;
       std::string mIdObj;
       int                      mIndVar0;
       std::vector<std::string> mNamesVar;
       std::vector<bool>        mActivVar;
};


class cMMVII_BundleAdj
{
     public :
          cMMVII_BundleAdj(cPhotogrammetricProject *);
          ~cMMVII_BundleAdj();

           // ======================== Add object ========================
          void  AddCalib(cPerspCamIntrCalib *);  /// add  if not exist
          void  AddCamPC(cSensorCamPC *);  /// add, error id already exist
          void  AddCam(const std::string & aNameIm);  /// add from name, require PhP exist
	  void  AddReferencePoses(const std::vector<std::string> &);  ///  [Fofder,SigmGCP,SigmaRot ?]

	  void AddBlocRig(const std::vector<double>& aSigma,const std::vector<double>&  aSigmRat ); // RIGIDBLOC
	  void AddCamBlocRig(const std::string & aCam); // RIGIDBLOC
          void AddTopo(); // TOPO
          cBA_Topo* getTopo() { return mTopo;}

          // Add clino bloc to compute relative orientation between clino and a camera
          void AddClinoBloc();
          void AddClinoBloc(cBA_Clino * aBAClino);

          bool AddTopo(const std::string & aTopoFilePath); // TOPO
          ///  =======  Add GCP, can be measure or measure & object
          void AddGCP3D(cMes3DDirInfo * aMesDirInfo, cSetMesGnd3D &aSetMesGnd3D, bool verbose=true);
          void AddGCP2D(cMes2DDirInfo * aMesDirInfo, cSetMesPtOf1Im & aSetMesIm, cSensorImage* aSens, eLevelCheck aOnNonExistGCP=eLevelCheck::Warning, bool verbose=true);
          cBA_GCP& getGCP() { return mGCP;}

          ///  ============  Add Lidar/Photogra ===============          void AddLineAdjust(const std::vector<std::string> &);

          void Add1AdjLidarPhotogra(const std::vector<std::string> &);
          void Add1AdjLidarPhoto(const std::vector<std::string> &);

	  ///  ============  Add multiple tie point ============
	  void AddMTieP(const std::string & aName,cComputeMergeMulTieP  * aMTP,const cStdWeighterResidual & aWIm);

          /// One iteration : add all measure + constraint + Least Square Solve/Udpate/Init
          void OneIteration(tREAL8 aLVM=0.0, bool isLastIter=false, bool doShowCond=false);

          const std::vector<cSensorImage *> &  VSIm() const ;  ///< Accessor
          const std::vector<cSensorCamPC *> &  VSCPC() const;   ///< Accessor
								//

          bool CheckGCPConstraints() const; //< test if free points have enough observations
	  //  =========  control object free/frozen ===================

	  void SetParamFrozenCalib(const std::string & aPattern);
	  void SetParamFreeCalib(const std::vector<std::vector<std::string>> & aPattern);
	  void SetViscosity(const tREAL8& aViscTr,const tREAL8& aViscAngle);
	  void SetFrozenCenters(const std::string & aPattern);
	  void SetFrozenOrients(const std::string & aPattern);
       void SetFrozenClinos(const std::string & aPattern);
          void SetSharedIntrinsicParams(const std::vector<std::string> &);
           

          void AddPoseViscosity();
          void AddConstrainteRefPose();
          void AddConstrainteRefPose(cSensorCamPC & aCam,cSensorCamPC & aCamRef);

          //  ----------------  Line adjustment -------------------------------------
          void AddLineAdjust(const std::vector<std::string> &);
          void DeleteLineAdjust();
          void IterAdjustOnLine();

          //  ----------------  Block of instrument (new version) -------------------------------------
          void AddBlockInstr(const std::vector<std::vector<std::string>> &);
          void AddClinoBlokcInstr(const std::vector<std::vector<std::string>> &);

          void SetHardGaugeBlockInstr(); //< if "hard" gauge must be done outside equation
          void IterOneBlockInstr();
          // 0 None , 1 Empirical , 2 by covariance
          void SaveBlockInstr();
          void DeleteBlockInstr();



          void SaveBlocRigid();
          void Save_newGCP3D();
          void SaveTopo();

          void Set_UC_UK(const std::vector<std::string> & aParam);
          void ShowUKNames(const std::vector<std::string> & aParam, const std::string &aSuffix, cMMVII_Appli* =nullptr) ;
          // Save results of clino bundle adjustment
          void SaveClino();
          void  AddBenchSensor(cSensorCamPC *); // Add sensor, used in Bench Clino

          void setVerbose(bool aVerbose){mVerbose=aVerbose;}; // Print or not residuals
          
          cResolSysNonLinear<tREAL8> *  Sys();  /// Real object, will disapear when fully interfaced for mSys

          cSetInterUK_MultipeObj<tREAL8> &   SetIntervUK();

          cPhotogrammetricProject  &PhProj();

     private :

          //============== Methods =============================
          cMMVII_BundleAdj(const cMMVII_BundleAdj &) = delete;
          void  AddSensor(cSensorImage *);  /// add, error id already exist

          void AssertPhaseAdd() ;  /// Assert we are in phase add object (no iteration has been donne)
          void AssertPhp() ;             /// Assert we use a Photogram Project (class can be use w/o it)
          void AssertPhpAndPhaseAdd() ;  /// Assert both
          void InitIteration();          /// Called at first iteration -> Init things and set we are non longer in Phase Add
          void InitItereGCP();           /// GCP Init => create UK
          void InitItereTopo();          /// Topo Init => create UK
          void OneItere_GCP();           /// One iteraion of adding GCP measures

	  void OneItere_TieP();   /// Iteration on tie points
	  void OneItere_TieP(const cBA_TieP&);   /// Iteration on tie points

          void CompileSharedIntrinsicParams(bool ForAvg);


          //============== Data =============================
          cPhotogrammetricProject * mPhProj;

          bool  mPhaseAdd;  ///< check that we dont mix add & use of unknowns

          cREAL8_RSNL *                 mSys;    /// base class can be 8 or 16 bytes
          cResolSysNonLinear<tREAL8> *  mR8_Sys;  /// Real object, will disapear when fully interfaced for mSys
          // ===================  Object to be adjusted ==================

          std::vector<cPerspCamIntrCalib *>  mVPCIC;     ///< vector of all internal calibration 4 easy parse
          std::vector<cSensorCamPC *>        mVSCPC;      ///< vector of perspectiv  cameras
          std::vector<cSensorImage *>        mVSIm;       ///< vector of sensor image (PC+RPC ...)
          //  std::vector<cCalculator<double> *> mVEqCol;     ///< vector of co-linearity equation -> replace by direct access

          cSetInterUK_MultipeObj<tREAL8>    mSetIntervUK;

	  // ================= Frozen/UnFrozen

	  std::string  mPatParamFrozenCalib;  /// Pattern for name of paramater of internal calibration
	  /// Pattern for parameters that are "finally" free,
          std::vector<std::vector<std::string>> mPatternFreeCalib;
	  std::string  mPatFrozenCenter;      /// Pattern for name of pose with frozen centers
	  std::string  mPatFrozenOrient;      /// Pattern for name of pose with frozen centers
          std::string  mPatFrozenClinos;      /// Pattern for name of clino with frozen boresight

          std::vector<std::string>  mVPatShared;

          // ===================  Information to use ==================
	     
	          // - - - - - - - - GCP  - - - - - - - - - - -
          cBA_GCP        mGCP;

	         // - - - - - - - - MTP  - - - - - - - - - - -
          std::vector<cBA_TieP*>   mVTieP;

                 // - - - - - - -   Bloc Rigid - - - - - - - -
	  cBA_BlocRig*              mBlRig;  // RIGIDBLOC
          cBA_Clino*              mBlClino;  // CLINOBLOC
          cBA_Topo*              mTopo;  // TOPO

          std::vector<cBA_LidarPhotogra*>  mVBA_Lidar;

	         // - - - - - - -   Reference poses- - - - - - - -
          std::vector<cSensorCamPC *>        mVCamRefPoses;      ///< vector of reference  poses if they exist
	  std::string                        mFolderRefCam;
	  tREAL8                             mSigmaTrRefCam;
	  tREAL8                             mSigmaRotRefCam;
          std::string                        mPatternRef;
	  bool                               mDoRefCam;
          cDirsPhProj*                       mDirRefCam;
          // ===================  "Viscosity"  ==================

	  tREAL8   mSigmaViscAngles;  ///< "viscosity"  for angles
	  tREAL8   mSigmaViscCenter;  ///< "viscosity"  for centers
				      //
	  int      mNbIter;    /// counter of iteration, at least for debug
          bool     mVerbose; // print residuals

          std::vector<cBundleBlocNamedVar>  mVBBNamedV;

          bool                      mShow_UC_UK;
          bool                      mCompute_Uncert;
          std::vector<std::string>  mParam_UC_UK;
          std::vector<int>          mIndCompUC;
          cResult_UC_SUR<tREAL8>*   mRUCSUR;
          std::vector<cUK_Line3D_4BA*>           mVecLineAdjust;
          std::vector<cBA_BlockInstr *>          mVecBlockInstrAdj;
};


};

#endif // _MMVII_BUNDLEADJUSTMENT_H_
