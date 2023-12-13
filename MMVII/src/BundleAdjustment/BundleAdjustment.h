#include "MMVII_PCSens.h"
#include "MMVII_MMV1Compat.h"
#include "MMVII_DeclareCste.h"
#include "MMVII_BundleAdj.h"

using namespace NS_SymbolicDerivative;
namespace MMVII
{

/**   Class for representing a Pt of R3 in bundle adj, when it is considered as
 *   unknown.
 *      +  we have the exact value and uncertainty of the point is covariance is used
 *      -  it add (potentially many)  unknowns and then  it take more place in  memory & time
 */

class cPt3dr_UK :  public cObjWithUnkowns<tREAL8>,
                   public cMemCheck
{
      public :
              cPt3dr_UK(const cPt3dr &);
              ~cPt3dr_UK();
              void PutUknowsInSetInterval() override;
              const cPt3dr & Pt() const ;
      private :
              cPt3dr_UK(const cPt3dr_UK&) = delete;
              cPt3dr mPt;
};


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

          ///  =======  Add GCP, can be measure or measure & object
          void AddGCP(tREAL8 aSigmaGCP,const  cStdWeighterResidual& aWeightIm, cSetMesImGCP *);

	  ///  ============  Add multiple tie point ============
	  void AddMTieP(cComputeMergeMulTieP  * aMTP,const cStdWeighterResidual & aWIm);

          /// One iteration : add all measure + constraint + Least Square Solve/Udpate/Init
          void OneIteration();

          const std::vector<cSensorImage *> &  VSIm() const ;  ///< Accessor
          const std::vector<cSensorCamPC *> &  VSCPC() const;   ///< Accessor
								//

	  //  =========  control object free/frozen ===================

	  void SetParamFrozenCalib(const std::string & aPattern);
	  void SetViscosity(const tREAL8& aViscTr,const tREAL8& aViscAngle);
	  void SetFrozenCenters(const std::string & aPattern);

	  void AddPoseViscosity();
	  void AddConstrainteRefPose();
          void AddConstrainteRefPose(cSensorCamPC & aCam,cSensorCamPC & aCamRef);


	  void SaveBlocRigid();

     private :

          //============== Methods =============================
          cMMVII_BundleAdj(const cMMVII_BundleAdj &) = delete;
          void  AddSensor(cSensorImage *);  /// add, error id already exist

          void AssertPhaseAdd() ;  /// Assert we are in phase add object (no iteration has been donne)
          void AssertPhp() ;             /// Assert we use a Photogram Project (class can be use w/o it)
          void AssertPhpAndPhaseAdd() ;  /// Assert both
          void InitIteration();          /// Called at first iteration -> Init things and set we are non longer in Phase Add
          void InitItereGCP();           /// GCP Init => create UK
          void OneItere_GCP();           /// One iteraion of adding GCP measures

	  void OneItere_TieP();   /// Iteration on tie points

          ///  One It for 1 pack of GCP (4 now 1 pack allowed, but this may change)
          void OneItere_OnePackGCP(const cSetMesImGCP *);


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

          // ===================  Information to use ==================
	     
	          // - - - - - - - - GCP  - - - - - - - - - - -
          cSetMesImGCP *           mMesGCP;
          cSetMesImGCP             mNewGCP; // set of gcp after adjust
	  tREAL8                   mSigmaGCP;
          cStdWeighterResidual     mGCPIm_Weighter;
          std::vector<cPt3dr_UK*>  mGCP_UK;

	         // - - - - - - - - MTP  - - - - - - - - - - -
	  cComputeMergeMulTieP *   mMTP;
          cStdWeighterResidual     mTieP_Weighter;

                 // - - - - - - -   Bloc Rigid - - - - - - - -
	  cBA_BlocRig*              mBlRig;  // RIGIDBLOC
	  
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
};


};

