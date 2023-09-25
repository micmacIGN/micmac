#include "MMVII_PCSens.h"
#include "MMVII_MMV1Compat.h"
#include "MMVII_DeclareCste.h"
#include "MMVII_BundleAdj.h"

namespace MMVII
{

/**   Class for representing a Pt of R3 in bundle adj, when it is considered as
 *   unknown.
 *      +  we have the exact value and uncertainty of the point
 *      -  it add (potentially many)  unknowns
 *      -  it take more place in  memory
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


class cMMVII_BundleAdj
{
     public :
          cMMVII_BundleAdj(cPhotogrammetricProject *);
          ~cMMVII_BundleAdj();

           // ======================== Add object ========================
          void  AddCalib(cPerspCamIntrCalib *);  /// add  if not exist
          void  AddCamPC(cSensorCamPC *);  /// add, error id already exist
          void  AddCam(const std::string & aNameIm);  /// add from name, require PhP exist


          ///  =======  Add GCP, can be measure or measure & object
          void AddGCP(const  std::vector<double>&, cSetMesImGCP *);

          /// One iteration : add all measure + constraint + Least Square Solve/Udpate/Init
          void OneIteration();

          const std::vector<cSensorImage *> &  VSIm() const ;  ///< Accessor
          const std::vector<cSensorCamPC *> &  VSCPC() const;   ///< Accessor
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

          ///  One It for 1 pack of GCP (4 now 1 pack allowed, but this may change)
          void OneItere_OnePackGCP(const cSetMesImGCP *,const std::vector<double> & aVW);


          //============== Data =============================
          cPhotogrammetricProject * mPhProj;

          bool  mPhaseAdd;  ///< check that we dont mix add & use of unknowns

          cREAL8_RSNL *                 mSys;    /// base class can be 8 or 16 bytes
          cResolSysNonLinear<tREAL8> *  mR8_Sys;  /// Real object, will disapear when fully interfaced for mSys
          // ===================  Object to be adjusted ==================

          std::vector<cPerspCamIntrCalib *>  mVPCIC;     ///< vector of all internal calibration 4 easy parse
          std::vector<cSensorCamPC *>        mVSCPC;      ///< vector of perspectiv  cameras
          std::vector<cSensorImage *>        mVSIm;       ///< vector of sensor image (PC+RPC ...)
          std::vector<cCalculator<double> *> mVEqCol;       ///< vector of sensor image (PC+RPC ...)

          cSetInterUK_MultipeObj<tREAL8>    mSetIntervUK;


          // ===================  Object to be adjusted ==================
          cSetMesImGCP *           mMesGCP;
          cSetMesImGCP             mNewGCP;
          std::vector<double>      mWeightGCP;
          std::vector<cPt3dr_UK*>  mGCP_UK;
};


};

