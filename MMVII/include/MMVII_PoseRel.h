#ifndef  _MMVII_POSE_REL_H_
#define  _MMVII_POSE_REL_H_

#include "MMVII_PCSens.h"
#include "MMVII_TplHeap.h"

namespace MMVII
{

class cSetHomogCpleDir;  // store tie point as pairs of homologous directions
class cMatEssential;     // class for representing a matt ess, double rep :  vector 9 + matrix 3x3
class cCamSimul ;        //  Class for simulating position of set camera having visibility domain in common

// Vect being "a-plat" representation of matrix M, put in Vect the constraint P1 M P2
void SetVectMatEss(cDenseVect<tREAL8> & aVect,const cPt3dr& aP1,const cPt3dr& aP2);
// transformate linear rep in a matrix
cDenseMatrix<tREAL8> Vect2MatEss(const cDenseVect<tREAL8> & aSol);
//  Add in Sys all the constraint relative to direction of Sys
void  MatEssAddEquations(const cSetHomogCpleDir & aSetD,cLinearOverCstrSys<tREAL8> & aSys);


/**   Class for storing homologous point as unitary direction of bundles (one made
 *     its no longer needed to access to the internal calibration + the coordonates are bounded).
 *
 *     Optionnaly , for normalization, the point can be turned from a rotation
 */

class cSetHomogCpleDir : public cMemCheck
{
     public :
        typedef cRotation3D<tREAL8> tRot;

        ///  Create from image homologue + internal calibration
        cSetHomogCpleDir(const cSetHomogCpleIm &,const cPerspCamIntrCalib &,const cPerspCamIntrCalib &);

        cSetHomogCpleDir(const std::vector<cPt3dr>&,const std::vector<cPt3dr>&);

        /// make both normalization so that bundles are +- centered onK
        void NormalizeRot();

        const std::vector<cPt3dr>& VDir1() const; ///< Accessor
        const std::vector<cPt3dr>& VDir2() const; ///< Accessor
        void Show() const;


        /// 4 bench :  Randomize the rotation 
        void RandomizeRot();
        ///  4 bench : Put in current data an outlayer, in Dir1 or Dir2 (randomly), at random position
        void GenerateRandomOutLayer(double aAmpl);

     private :
        /// make one normalization so that bundles are +- centered onK
        void  NormalizeRot(tRot&,std::vector<cPt3dr> &);
        /// Transormate bundle and accuumlate to memorize transformation
        void  AddRot(const tRot&,tRot&,std::vector<cPt3dr> &);

        std::vector<cPt3dr>  mVDir1;
        std::vector<cPt3dr>  mVDir2;

        tRot                 mR1ToInit;
        tRot                 mR2ToInit;
};

/** Class to represent the result of essential matrix computation,
 * essentially 3x3 matrix and an eq
 */

class cMatEssential
{
    public :
        typedef cDenseMatrix<tREAL8> tMat;
        typedef cIsometry3D<tREAL8>  tPose;

        /** Constructor use a set of 3D homologous dir, a solver (L1/L2) , and the number, in [0,8] of the variable
            that is arbirtrarily fixed */
        cMatEssential(const cSetHomogCpleDir &,cLinearOverCstrSys<tREAL8> & aSys,int aKFix);

        ///  Sigma attenuates big error  E*S / (E+S)  => ~E in 0  , bound to S at infty, if S<=0  return E
        tREAL8  Cost(const  cPt3dr & aP1,const  cPt3dr &aP2,const tREAL8 & aSigma) const;

        /// No more than average of Cost
        tREAL8  AvgCost(const  cSetHomogCpleDir &,const tREAL8 & aSigma) const;

        /// No more than
        tREAL8  KthCost(const  cSetHomogCpleDir &,tREAL8 aProp) const;

	/// Return the essential matrix
         const tMat& Mat() const {return mMat;}

        void Show(const cSetHomogCpleDir &) const;

        cMatEssential(const  tMat & aMat);

         tPose ComputePose(const cSetHomogCpleDir & aHoms,const tPose * aRes= nullptr) const;

    private :
        tMat mMat; /// The Ess  matrix itself
};

/**  For essential matrix as the equations are purely linear,  it is necessary to add
 *  some arbitray constraint to make the system solvable. 
 *
 *  => Make an estimation of this direction
 */

int   MatEss_GetKMax(const cSetHomogCpleDir & aSetD,tREAL8 aWeightStab,bool Show=false);

/** used in case for epipolar geometry, knoing base in the repair E1 (generally (1,0,0) and
 *  2 rotation going from initial repairs to the other , compute the relative rotation; quite
 *  basic, but opportunity to fix convention */

tPoseR  PoseRelFrom2RotAndBase(const cPt3dr &,const tRotR& aR2E1,const tRotR & aR2E2);


/**  Class for computing "elementary" bundle adjustment. It is optimized for initial pose
 *   estimation. Compared with other class of BA :
 *      - it take only direction of bundles (coming from camera calibration)
 *      - the parametrisation of unknowns is optimized for few poses
*/


class cElemBA
{
   public :
        /// Constructor : Mode of Compensation + Vector of poses
        cElemBA(eModResBund,const std::vector<tPoseR>& aVPose);
        /// Free allocated memory
       ~cElemBA();

       /**  Add one Obs of 2 direction between Cam1 & Cam2 , Noise is used only in test mode for verification that
            even if bundle intersection is un-accurate, we converge to the good sollution */
       void AddHomBundle_Cam1Cam2(const cPt3dr & aDirB0,const cPt3dr & aDirB1,tREAL8 aW,tREAL8 aEpsilon=1e-6, tREAL8 aNoise=0);

       /// Return Residual + Intersection
       std::pair<tREAL8,cPt3dr> InterBundles(const std::vector<int> &aVNumCams,const cPt3dr * aDirBdund,tREAL8 aEpsilon=1e-6) const;
       /// Add  Obs for N cam, "compagnion" of InterBundles
       tREAL8 AddHom_NCam(const std::vector<int> &,const cPt3dr * ,const cPt3dr &aPGr,tREAL8 aW) ;


       void OneIter(tREAL8 aLVM);  ///< Iterate one obs have been added
       cResolSysNonLinear<double> *  Sys();   ///< Accesor
       const std::vector<tPoseR>  &  CurPose() const; ///< Acessor

       tREAL8 AvgRes1() const; /// Average of residual of Cam1
       tREAL8 AvgRes2() const;  /// Average of residual of Cam2
       tREAL8 AvgResN() const;  /// Average of residual of CamN

   private :
       /// Add obs of Bundle for cam1, to put in Substiution structe - Colinearity
       tREAL8 AddEquationColinearity_Cam1(cSetIORSNL_SameTmp<tREAL8> &,const cPt3dr & aDirB0,tREAL8  aWeight);

       /// Add obs of Bundle for cam2, to put in Substiution structe  - Colinearity
       tREAL8 AddEquationColinearity_Cam2(cSetIORSNL_SameTmp<tREAL8> &,const cPt3dr & aDirB1,tREAL8  aWeight);

       /// Add obs of Bundle for cam2, to put in Substiution structe  - Colinearity
       tREAL8 AddEquationColinearity_CamN(size_t aIndC,cSetIORSNL_SameTmp<tREAL8> &,const cPt3dr & aDirB1,tREAL8  aWeight);

       /// Add obs for Cam 1 & 2, no point computed/No Schuur -> Coplanarity
       tREAL8 AddEquationCoplanarity(const cPt3dr & aDirB1,const cPt3dr & aDirB2,tREAL8  aWeight);

       /// Compute bundle given camera an local direction
       tSeg3dr  Bundle(int aKCam,const cPt3dr &) const;


        cElemBA(const cElemBA &) = delete;

        //void Add

       eModResBund                        mMode;  ///< Mode of equation
       bool                               isModeCoplan; ///< is it one mode of co-planarity
       std::vector<tPoseR>                mCurPose;      ///< Vector of current pose (init then updated)
       int                                mSzBuf;       ///<  Sz Buf for calculator
       cCalculator<double> *              mEqElemCam1;  ///< Colinearity equation - Cam1
       cCalculator<double> *              mEqElemCam2;   ///< Colinearity equation - Cam2
       cCalculator<double> *              mEqElemCamN;   ///< Colinerity equation - Cam N
       cCalculator<double> *              mEqElemCam12;   ///< Co-planrarity equatipn

       cSetInterUK_MultipeObj<double>     mSetInterv;   ///< coordinator for autom numbering
       cResolSysNonLinear<double> *       mSys;         ///< Solver
       cLeasSqtAA<tREAL8> *               mSystAA;      ///< Pointer to dense least square solve (short cut for linDet12 case)
       cP3dNormWithUK                     mTr2;         ///< Unknown  trans for cam2 force to unity
       cRotWithUK                         mRot2;        ///< Unknown rotation for cam2
       std::vector<cPoseWithUK*>          mPoseN;       ///< unkown pose over 2 (to come)

       cWeightAv<tREAL8>                  mRes1;        ///< Average of residual Cam1
       cWeightAv<tREAL8>                  mRes2;        ///< Average of residual Cam2
       cWeightAv<tREAL8>                  mResN;        ///< Average of residual Cam > 2

       /// do we use special case for cam 1(fixed) and 2 (Unit base)
       bool                               mSCCam12;
       /// Index of first camera "not special" , depend of mSCCam12
       size_t                             mIndCamGen;

};


class cPSC_PB ; // Planar Scene Param Bench
class cPSC_Selec ; // Planar Scene Param Selection

//class cPSC_Sol ; // Planar Scene Param Selection
class cPSC_Sol
{
   public :
    cPSC_Sol(const tPoseR&,tREAL8 aPNeg1,tREAL8 aPNeg2);
    tREAL8 Score() const;
     const tPoseR & PoseRel() const;
   private :
      tPoseR   mPoseRel;
      tREAL8   mPropNeg1;  //< Number of coordinate <0 (bad size of bundle)
      tREAL8   mPropNeg2;
};
class  cCmpcPSC_Sol_OnScore
{
     public :
         bool operator ()(const cPSC_Sol & aS1,const cPSC_Sol & aS2) const
         {
             return aS1.Score() > aS2.Score();
         }
};


///  Mode of execution of functions that wan be used both for Run-Time en Test
enum class eModeExec
{
    eRunTime,  //< Run time will do just internal test
    eBench,    //< Bench will make "intensive" test of correctness (on generally "perfect" data)
    eTest      //< will do Bench and  generate some output (message, images ...) to help comprehension
};
class  cCmpcPSC_Sol_OnScore;

class cPS_CompPose
{
   public :
        cPS_CompPose
        (
                const cSetHomogCpleDir &,
                bool isL1=false,
                int aKMax=8,
                const  cPSC_PB * = nullptr,
                tREAL8  aLVM = 0.0 ///< Levenberg-Markard for Homog Estimate
        );
       ~cPS_CompPose();

        typedef std::pair<cPSC_PB,cSetHomogCpleDir>  tResSimul;
        typedef cKBestValue<cPSC_Sol,cCmpcPSC_Sol_OnScore> tCmpSol;

        /** generate pair 3D of direction that can correpond to coherent camera, can be more
         "extremate" than with camera, can simulate for example two side of plane.
         */
         static tResSimul SimulateDirAny(tREAL8 aMinDistViewPoints,tREAL8 aDistPlane,bool isSameSide,eModeExec); // Steep of plane, may be adjusted


         /** generate pair of direction the correspond to camera already in epipolar config, more for
             debugin than for check */
         static cSetHomogCpleDir SimulateDirEpip
                                 (
                                      tREAL8 aZ,           // Altitude of both camera
                                      tREAL8 aSteepPlane,  // Z = X * Steep
                                      tREAL8 aRho,         // Circle of random plane
                                      bool   BaseIsXP1     // is the base (1,0,0) or (-1,0,0) ?
                                 );

         const tCmpSol & Sols() const;
   private :

        ///  Compute the 3D homog matrix
        static cDenseMatrix<tREAL8> ComputeMatHom3D
          (const cSetHomogCpleDir &,bool isL1,int aKMax,const  cPSC_PB *,tREAL8 aLVM);

        bool TestOneHypoth
             (
                cResulSVDDecomp<tREAL8>& aSvdH,
                const cPt3dr&ABL,
                int SignB,
                const cPt3dr & aSignD,
                const cPSC_Selec & aSel
              );

        /// generate a point of view by randomizing Theta
       // static    cPt3dr RandPointOfView(const cPt2dr & aRhoZ);
        static    cPt3dr RandPointOfView(int aSign,tREAL8 aDistMinPlane);


        /// Utilitary : transferate "Pt" in "Vect" at offset "Index" muliplied by "Mul"
        static void SetPt( cDenseVect<tREAL8>& aVect,size_t aIndex,const cPt3dr& aPt,tREAL8 aMul);


        const cSetHomogCpleDir &   mCpleDir;
        const  cPSC_PB *           mCurParam;
        eModeExec                  mMode;
        const std::vector<cPt3dr>* mCurV1 ;
        const std::vector<cPt3dr>* mCurV2 ;
        size_t                     mCurNbPts;

        int                        mKMax;
        cDenseMatrix<tREAL8>       mMatH ; ///<  Homography matrix


        // parameters extract of homog matrix , used to compute polynome
        tREAL8                     mHM_Det;
        tREAL8                     mHM_Tr2;
        tREAL8                     mHM_Tr4;

        size_t                     mNbSolTested; ///< number of sol we try
        tCmpSol *                  mCmpSol;      ///< will a maximum of 2 best solution
};

enum class eModePE2I
           {
              eRansac,
              ePlane1,
              ePlane2,
              eNbVals
           };


///  Basic class to store the pose between 2 images
class cCdtPoseRel2Im
{
  public :
      cCdtPoseRel2Im(const tPoseR&,eModePE2I,tREAL8 aScore,const std::string& aMsg);
      cCdtPoseRel2Im();

      tPoseR                 mPose;  ///< The pose itself
      eModePE2I              mMode;
      tREAL8                 mScore; ///< The score /residual : the smaller the better
      tREAL8                 mScore0; ///< The score before BA
      std::string            mMsg;   ///< Message for tuning
      std::optional<cPt2dr>  mScorePixGT;
};

class cCdtFinalPoseRel2Im
{
    public :
       std::string                 mIm1;
       std::string                 mIm2;
       std::vector<cCdtPoseRel2Im> mVCdt;
};
void AddData(const  cAuxAr2007 & anAux,cCdtFinalPoseRel2Im & aCdt);


};
#endif // _MMVII_POSE_REL_H_

