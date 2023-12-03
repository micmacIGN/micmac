#ifndef  _MMVII_POSE_REL_H_
#define  _MMVII_POSE_REL_H_

#include "MMVII_PCSens.h"

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

         tPose ComputePose(const cSetHomogCpleDir & aHoms,tPose * aRes= nullptr) const;

    private :
        tMat mMat; /// The Ess  matrix itself
};

/**  For essential matrix as the equations are purely linear,  it is necessary to add
 *  some arbitray constraint to make the system solvable. 
 *
 *  => Make an estimation of this direction
 */

int   MatEss_GetKMax(const cSetHomogCpleDir & aSetD,tREAL8 aWeightStab,bool Show=false);



};
#endif // _MMVII_POSE_REL_H_

