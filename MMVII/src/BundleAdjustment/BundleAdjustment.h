#ifndef  _MMVII_BUNDLEADJUSTMENT_H_
#define  _MMVII_BUNDLEADJUSTMENT_H_

#include "MMVII_PCSens.h"
#include "MMVII_MMV1Compat.h"
#include "MMVII_DeclareCste.h"
#include "MMVII_BundleAdj.h"
#include "MMVII_Clino.h"
#include "MMVII_SysSurR.h"

using namespace NS_SymbolicDerivative;
namespace MMVII
{

class cBA_Topo;

/**   Class for representing a Pt of R3 in bundle adj, when it is considered as
 *   unknown.
 *      +  we have the exact value and uncertainty of the point is covariance is used
 *      -  it add (potentially many)  unknowns and then  it take more place in  memory & time
 */

/*
template <const int Dim>  class cPtxdr_UK :  public cObjWithUnkowns<tREAL8>,
                                             public cMemCheck
{
   public :
      typedef cPtxd<tREAL8,Dim>  tPt;

      cPtxdr_UK(const tPt &);
      ~cPtxdr_UK();
      void PutUknowsInSetInterval() override;
      const tPt & Pt() const ;
   private :
      cPtxdr_UK(const cPtxdr_UK&) = delete;
      tPt mPt;
};

typedef cPtxdr_UK<3> cPt3dr_UK ;
*/

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
              const cSensorCamPC * aCam,                     // camera-> only the pose is useful
              const std::vector<std::string> &aVClinoName, // vector of clinometer names
              const std::vector<tREAL8> & aVAngles,    // vector of angles of all clinometer
              const cPt3dr &   aVerticAbs = {0,0,-1}   // position of vertical in current "absolut" system
          );

          const cSensorCamPC * Cam() const {return mCam;} ;           // return the camera
          void pushObs(std::vector<double> & aVObs) const; // push observations : the camera orientation and clinometer measures
          void pushWeights(std::vector<double> & aVWeights) const; // push weights
          const std::map<std::string, double> VDir() const {return mVDir;};   // return the clinometers measures

	

     private :
          const cSensorCamPC *          mCam;  ///< camera , memorization seems useless
          std::map<std::string, double>     mVDir; ///<  map : clinometer name, measured position of needle, computed from angles
          std::map<std::string, double>     mWeights; ///< map :  clinometer name, weights on clino measures
	     cPt3dr                  mVertInLoc;  ///<  Vertical in  camera system, this  is the only information usefull of camera orientation
};


class cClinoWithUK :  public cObjWithUnkowns<tREAL8>, public cMemCheck
{
     
     // Object to compute orientation of clinometer in a camera repere
     // mRot is the rotation matrix
     // mOmega is the axiator, the computed value by least squares to get a new rotation
     
     public :
	     cClinoWithUK();
          cClinoWithUK(tRotR aRot, const std::string & aNameClino);
          void PutUknowsInSetInterval() override; 
          void OnUpdate() override;          // update mRot with mOmega
          void GetAdrInfoParam(cGetAdrInfoParam<tREAL8> &) override;
          void pushIndex(std::vector<int> & aVInd) const;
          tRotR Rot() const {return mRot;};
          cPt3dr Omega() const {return mOmega;};
          const std::string NameClino() const {return mNameClino;};
	
     private :
         const std::string    mNameClino;          // Name of the clino
         tRotR          mRot;                // Rotation between one camera and the clinometer. It is also the initial solution
         cPt3dr         mOmega;              // Vector axiator, the unknowns of the least squares
};


// CLINOBLOC
class cBA_Clino : public cMemCheck
{
     // Object to compute Bundle Adjustment on clinometers
     public :

          cBA_Clino(const cPhotogrammetricProject *aPhProj, cCalibSetClino *aCalibSetClino);
          cBA_Clino(const cPhotogrammetricProject *, const std::string & aNameClino, const std::string & aFormat, const std::vector<std::string> & aPrePost);
          ~cBA_Clino();
          cPt2dr addOneEquation(cResolSysNonLinear<tREAL8> & aSys, cClinoMes1Cam & aMeasure);
          void addEquations(cResolSysNonLinear<tREAL8> & aSys);
          void AddToSys(cSetInterUK_MultipeObj<tREAL8> & aSet);
          void pushObs(std::vector<double> & aVObs, const cPt3dr & aCamTr) const; // push observations : initial values of Boresight matrix (2*9 values), and the vertical in local repere (3 values)
          void pushIndex(std::vector<int> & aVInd) const;
          void Save() const;
          void addClinoMes1Cam(const cClinoMes1Cam & aClinoMes1Cam);
          void addClinoWithUK(const std::string & aClinoName, tRotR & aRot);
          std::vector<tRotR>  ClinosWithUKRot() const;
          void setCalibSetClino(cCalibSetClino* aCalibSetClino);
          void printRes() const; // Display residuals


          
     private :

          void readMeasures();                    //  Get initial boresight matrices, computed by ClinoInit

        
	     const cPhotogrammetricProject * mPhProj;
          const std::string mNameClino;
          const std::string mFormat;
          const std::vector<std::string> mPrePost;
          std::vector<cClinoMes1Cam>  mVMeasures;
          std::vector<std::string> mVNamesClino;
          cCalculator<double> *        mEqBlUK;
          std::vector<double>          mWeight;
          std::map<std::string, cClinoWithUK>    mClinosWithUK;
          cCalibSetClino               *mCalibSetClino;
          cPt2dr                        mRes; // Residuals
          
};








class cBA_GCP
{
     public :
	          // - - - - - - - - GCP  - - - - - - - - - - -
          cBA_GCP();
          ~cBA_GCP();

          std::string              mName;   // Name of folder 
          cSetMesImGCP *           mMesGCP;
          cSetMesImGCP             mNewGCP; // set of gcp after adjust
	  tREAL8                   mSigmaGCP;
          cStdWeighterResidual     mGCPIm_Weighter;
          std::vector<cPt3dr_UK*>  mGCP_UK;
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


/**  
 */

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

       void AddClinoBloc(const std::string aNameClino, const std::string aFormat, std::vector<std::string> aPrePost);

          bool AddTopo(const std::string & aTopoFilePath); // TOPO
          ///  =======  Add GCP, can be measure or measure & object
          void AddGCP(const std::string & aName, tREAL8 aSigmaGCP, const  cStdWeighterResidual& aWeightIm, cSetMesImGCP *, bool verbose=true);
          std::vector<cBA_GCP*> & getVGCP() { return mVGCP;}

	  ///  ============  Add multiple tie point ============
	  void AddMTieP(const std::string & aName,cComputeMergeMulTieP  * aMTP,const cStdWeighterResidual & aWIm);

          /// One iteration : add all measure + constraint + Least Square Solve/Udpate/Init
          void OneIteration(tREAL8 aLVM=0.0);
          void OneIterationTopoOnly(tREAL8 aLVM=0.0, bool verbose=false); //< if no images

          const std::vector<cSensorImage *> &  VSIm() const ;  ///< Accessor
          const std::vector<cSensorCamPC *> &  VSCPC() const;   ///< Accessor
								//

	  //  =========  control object free/frozen ===================

	  void SetParamFrozenCalib(const std::string & aPattern);
	  void SetViscosity(const tREAL8& aViscTr,const tREAL8& aViscAngle);
	  void SetFrozenCenters(const std::string & aPattern);
	  void SetFrozenOrients(const std::string & aPattern);
          void SetSharedIntrinsicParams(const std::vector<std::string> &);
           

	  void AddPoseViscosity();
	  void AddConstrainteRefPose();
          void AddConstrainteRefPose(cSensorCamPC & aCam,cSensorCamPC & aCamRef);


	  void SaveBlocRigid();
          void Save_newGCP();
          void SaveTopo();
          void SaveClino();

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
          void OneItere_GCP(bool verbose=true);           /// One iteraion of adding GCP measures

	  void OneItere_TieP();   /// Iteration on tie points
	  void OneItere_TieP(const cBA_TieP&);   /// Iteration on tie points

          ///  One It for 1 pack of GCP (4 now 1 pack allowed, but this may change)
          void OneItere_OnePackGCP(cBA_GCP &, bool verbose=true);

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
	  std::string  mPatFrozenCenter;      /// Pattern for name of pose with frozen centers
	  std::string  mPatFrozenOrient;      /// Pattern for name of pose with frozen centers

          std::vector<std::string>  mVPatShared;

          // ===================  Information to use ==================
	     
	          // - - - - - - - - GCP  - - - - - - - - - - -
          std::vector<cBA_GCP*>        mVGCP;
          //  cSetMesImGCP *           mMesGCP;
          //  cSetMesImGCP             mNewGCP; // set of gcp after adjust
	  //  tREAL8                   mSigmaGCP;
          //  cStdWeighterResidual     mGCPIm_Weighter;
          //  std::vector<cPt3dr_UK*>  mGCP_UK;

	         // - - - - - - - - MTP  - - - - - - - - - - -
	  // cComputeMergeMulTieP *   mMTP;
          // cStdWeighterResidual     mTieP_Weighter;
          std::vector<cBA_TieP*>   mVTieP;

                 // - - - - - - -   Bloc Rigid - - - - - - - -
	  cBA_BlocRig*              mBlRig;  // RIGIDBLOC
       cBA_Clino*              mBlClino;  // CLINOBLOC
          cBA_Topo*              mTopo;  // TOPO

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
};


};

#endif // _MMVII_BUNDLEADJUSTMENT_H_
