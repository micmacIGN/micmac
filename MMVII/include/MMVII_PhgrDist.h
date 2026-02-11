#ifndef  _MMVII_PHGR_DIST_H_
#define  _MMVII_PHGR_DIST_H_

#include "MMVII_Mappings.h"

namespace MMVII
{
/**  Indicate the nature of each coefficient of the distorsion
*/
enum class eTypeFuncDist
           {
              eRad,   ///<  Coefficient for radial distorsion 
              eDecX,  ///< Coefficient for decentric distorsion x mean derived by Cx (it has X and Y components)
              eDecY,  ///< Coefficient for decentric distorsion y mean derived by Cy (it has X and Y components)
              eMonX,  ///< Coefficient for a monom in X, of the polynomial distorsion
              eMonY,  ///< Coefficient for a monom in Y, of the polynomial distorsion
              eMonom,   ///< Coefficient for a monom in X or Y, used for example in Radiom where there is no reason to sep X/Y
              eNbVals ///< Tag for number of value
           };

/**  This class store a complete description of each parameters of the distorsion,
     it is used for computing the formula, the vector of name and (later) automatize
     conversion, print understandable values...

     It's rather bad design with the same classe coding different things, and some fields
     used or not according to the other, but as it is an internal/final classes, quick and dirty
     is exceptionnaly allowed...
*/
enum class eModeDistMonom
{
     eModeFraser,  // dx = b1 x + b2 y
     eModeSysCyl,  // dx = ax    dy = by
     eModeStd      // no special case
};


class cDescOneFuncDist
{
    public :
      cDescOneFuncDist(eTypeFuncDist aType,const cPt2di aDeg,eModeDistMonom aModeMonom);
      /// Majorarion of norms of jacobian , used in simulation
      double MajNormJacOfRho(double aRho) const;

      eTypeFuncDist mType;       ///< Type of dist (Rad, DecX ....)
      std::string   mName;       ///< Name : used as id for code gen & prety print
      std::string   mLongName;   ///< explicit long name
      cPt2di        mDegMon;     ///< Degree for a polynomial
      int           mNum;        ///< Order for radial and decentric
      int           mDegTot;     ///< Total degree of the polynome
};

std::vector<cDescOneFuncDist>   DescDist(const cPt3di & aDeg,bool isFraserMode);

std::vector<cDescOneFuncDist>   Polyn2DDescDist(int aDegree);

const std::vector<cDescOneFuncDist> & VDesc_RadiomCPI(int aDegree,int aDRadElim=-1);



/**  Class for generating distorsion random BUT invertible on certain domain,
    used for checking functionnalities on distorsion implying inversio
*/
class cRandInvertibleDist
{
   public :
       typedef NS_SymbolicDerivative::cCalculator<double> tCalc;
       cRandInvertibleDist
       (
              const cPt3di & aDeg,   ///< Degree of the distortio,
              double aRhoMax,        ///< Radius on which it must be invertible
              double aProbaNotNul,   ///< Probability for a coefficient to be not 0
              double aTargetSomJac,   ///< Target majoration of jacobian
              bool   isFraserMode
       );
       cDataNxNMapCalcSymbDer<double,2> *  MapDerSymb();
       const std::vector<double> & VParam() const;  ///< Accessor to parameters
       ~cRandInvertibleDist();
       tCalc & EqVal();
       tCalc & EqDer();
   private :
       cRandInvertibleDist(const cRandInvertibleDist &) = delete;

       double                          mRhoMax;  
       cPt3di                          mDeg;     ///< Degree of dist
       std::vector<cDescOneFuncDist>   mVecDesc; ///< Vector of descriptor
       tCalc *                         mEqVal;   ///< Calculator for values only
       tCalc *                         mEqDer;   ///< Calculator for values and derivatoves
       int                             mNbParam; ///< Number of parameters
       std::vector<double>             mVParam;  ///< Computed parameters
       bool                            mIsFraserMode; ///<  Std Mode / SIA Mode
};


// ==========================  utility to generate names  ===================

std::vector<std::string>  NamesP3(const std::string& aPref) ;  /// x y z
std::vector<std::string>  NamesP2(const std::string& aPref) ;  /// x y z
std::vector<std::string>  NamesMatr(const std::string& aPref,const cPt2di & aSz);  /// m00  m10  m20  m01 ...
std::vector<std::string>  NamesObsP3Norm(const std::string& aPref) ;  /// x y z


//  aPref+aK0  .... aPref+aK0-1
std::vector<std::string>  VectNames(const std::string& aPref,int aK0,int aK1) ;  /// x y z

///  Vector of name for a pose NameC->  center of proj,   NameOmega-> vector rotation
std::vector<std::string>  NamesPose(const std::string& NameC ,const std::string&  NameOmega);


std::vector<std::string>  NamesIntr(const std::string& aPref);  //  F  PPx   PPy



     //  ====  photogrametric equations ==============================
     
           // .............   Equation implying only distorsions  .............
     
/**  Allocate a calculator computing the distorsion  , used when distorsion is fixed and x,y are parameters (we need
     derivate to x,y for example in iterative inversion)
       UK=x,y  Obs=K1,K2 .....   K1 r +K2 R^3 ...
 */
NS_SymbolicDerivative::cCalculator<double> * EqDist(const cPt3di & aDeg,bool WithDerive,int aSzBuf,bool isFraserMode);

/** Allocate a calculator computing the base familly of a distorsion  UK=x,y Obs=K1,K2 .....  Kr, K2 R^3 , idem previous
    but does not return the sum, but the series of each independant function, used for least-square  feating with a
    given familly of func (for example in approximat inversion by least square)
 */
NS_SymbolicDerivative::cCalculator<double> * EqBaseFuncDist(const cPt3di & aDeg,int aSzBuf,bool isFraserMode);

/** Alloc a map corresponding to  distorsions :   create a map interface to an EqDist */
cDataNxNMapCalcSymbDer<double,2> * NewMapOfDist(const cPt3di & aDeg,const std::vector<double> & aVObs,int aSzBuf,bool isFraserMode);

           // .............   Equation implying only projection  .............
	 
///  For computing central perspectives projections     R3->R2
NS_SymbolicDerivative::cCalculator<double> * EqCPProjDir(eProjPC  aType,bool WithDerive,int aSzBuf);
///  For computing projections "inverse"   R2->R3 , return in fact direction of  bundle
NS_SymbolicDerivative::cCalculator<double> * EqCPProjInv(eProjPC  aType,bool WithDerive,int aSzBuf);

NS_SymbolicDerivative::cCalculator<double> * EqDistPol2D(int  aDeg,bool WithDerive,int aSzBuf,bool ReUse); // PUSHB
NS_SymbolicDerivative::cCalculator<double> * EqColinearityCamGen(int  aDeg,bool WithDerive,int aSzBuf,bool ReUse); // PUSHB
NS_SymbolicDerivative::cCalculator<double> * RPC_Proj(bool WithDerive,int aSzBuf,bool ReUse); // PUSHB

           // .............   Equation colinearity , imply external parameter, Projectiion, distorsion, foc+PP .............
enum class eTypeEqCol
           {
                ePt,
                eLine
           };

NS_SymbolicDerivative::cCalculator<double> * EqColinearityCamPPC(eProjPC  aType,const cPt3di & aDeg,bool WithDerive,int aSzBuf,bool ReUse,bool isFraserMode,eTypeEqCol);


          
           // .............   Equation radiometry .............
NS_SymbolicDerivative::cCalculator<double> * EqRadiomVignettageLinear(int aNbDeg,bool WithDerive,int aSzBuf,bool WithCste,int aDegPolSens);
NS_SymbolicDerivative::cCalculator<double> * EqRadiomCalibRadSensor(int aNbDeg,bool WithDerive,int aSzBuf,bool WithCste,int aDegPolSens);
NS_SymbolicDerivative::cCalculator<double> * EqRadiomCalibPolIma(int aNbDeg,bool WithDerive,int aSzBuf);
NS_SymbolicDerivative::cCalculator<double> * EqRadiomEqualisation(int aDegSens,int aDegIm,bool WithDerive,int aSzBuf,bool WithCste,int aDegPolSens);
NS_SymbolicDerivative::cCalculator<double> * EqRadiomStabilization(int aDegSens,int aDegIm,bool WithDerive,int aSzBuf,bool WithCste,int aDegPolSens);

           // .............   Equation on rigid bloc .............
NS_SymbolicDerivative::cCalculator<double> * EqBlocRig(bool WithDerive,int aSzBuf,bool Reuse);  // RIGIDBLOC
NS_SymbolicDerivative::cCalculator<double> * EqBlocRig_RatE(bool WithDerive,int aSzBuf,bool Reuse);  // RIGIDBLOC
NS_SymbolicDerivative::cCalculator<double> * EqBlocRig_Clino(bool WithDerive,int aSzBuf,bool ReUse);  // RIGIDBLOC
NS_SymbolicDerivative::cCalculator<double> * EqBlocRig_Orthog(bool WithDerive,int aSzBuf,bool ReUse);  // RIGIDBLOC


NS_SymbolicDerivative::cCalculator<double> * Old_EqClinoBloc(bool WithDerive,int aSzBuf,bool Reuse);  // CLINOBLOC
NS_SymbolicDerivative::cCalculator<double> * Old_EqClinoRot(bool WithDerive,int aSzBuf,bool Reuse);  // CLINOBLOC





     //  ====   equations used in tuto/bench/ devpt of surface  ==============================

           // .............   Equation implying 2D distance conservation .............

/// Calc for conservation of dist, Uk={x1,y1,x2,y2} Obs={D12} , let pk=(xk,yk)  Residual :  D(p1,p2)/d12 -1 
NS_SymbolicDerivative::cCalculator<double> * EqConsDist(bool WithDerive,int aSzBuf);
/// Ratio of dist, Uk={x1,y1,x2,y2,x3,y3} Obs={D12,D13,D23} ,  3 Residuals as D(p1,p2)/D12 - D(p1,p3)/D13
NS_SymbolicDerivative::cCalculator<double> * EqConsRatioDist(bool WithDerive,int aSzBuf);

// .............   Equation implying 3D distance .............

/// Calc for dist, Uk={x1,y1,z1,x2,y2,z2} Obs={D12} , let pk=(xk,yk,zk)  Residual :  D(p1,p2) - d12
NS_SymbolicDerivative::cCalculator<double> * EqDist3D(bool WithDerive,int aSzBuf);

/// Calc for parametrizes dist, Uk={d,x1,y1,z1,x2,y2,z2} Obs={} , let pk=(xk,yk,zk)  Residual :  D(p1,p2) - d
NS_SymbolicDerivative::cCalculator<double> * EqDist3DParam(bool WithDerive,int aSzBuf);

// .............   Equation implying topo subframe .............

/// Calc for parametrizes dist, Uk={a,b,c,x1,y1,z1,x2,y2,z2} Obs={r00, r01, r02, r10, r11, r12, r20, r21, r22, dx,dy,dz},
/// let pk=(xk,yk,zk), R=(r00..r22)  Residual :  R(p2-p2) - {dx,dy,dz}
NS_SymbolicDerivative::cCalculator<double> * EqTopoSubFrame(bool WithDerive,int aSzBuf);

/// Sum of square of unknown, to test non linear constraints
NS_SymbolicDerivative::cCalculator<double> * EqSumSquare(int aNb,bool WithDerive,int aSzBuf,bool ReUse);

// .............   Equation for topo stations .............
/// topo obs from a station: , Uk={pose_origin, pt_to} Obs={r00, r01, r02, r10, r11, r12, r20, r21, r22, val},
NS_SymbolicDerivative::cCalculator<double> * EqTopoHz(bool WithDerive,int aSzBuf);
NS_SymbolicDerivative::cCalculator<double> * EqTopoZen(bool WithDerive,int aSzBuf);
NS_SymbolicDerivative::cCalculator<double> * EqTopoDist(bool WithDerive,int aSzBuf);
NS_SymbolicDerivative::cCalculator<double> * EqTopoDX(bool WithDerive,int aSzBuf);
NS_SymbolicDerivative::cCalculator<double> * EqTopoDY(bool WithDerive,int aSzBuf);
NS_SymbolicDerivative::cCalculator<double> * EqTopoDZ(bool WithDerive,int aSzBuf);
NS_SymbolicDerivative::cCalculator<double> * EqTopoDH(bool WithDerive,int aSzBuf);

           // .............   Equation implying 2D distance conservation .............
	   
/// Equation used to optimize homothetic transform between model and image (used as a tutorial for deformable model)
NS_SymbolicDerivative::cCalculator<double> * EqDeformImHomotethy(bool WithDerive,int aSzBuf);
/// Variant of "EqDeformImHomotethy", case where we use linear approximation
NS_SymbolicDerivative::cCalculator<double> * EqDeformImLinearGradHomotethy(bool WithDerive,int aSzBuf);

           // .............   Covariance propagation  .............

/// For propag cov in network 
NS_SymbolicDerivative::cCalculator<double> * EqNetworkConsDistProgCov(bool WithDerive,int aSzBuf,const cPt2di& aSzN);
/// For propag in network  W/O covariance (i.e propagate directly the points)
NS_SymbolicDerivative::cCalculator<double> * EqNetworkConsDistFixPoints(bool WithDerive,int aSzBuf,const cPt2di& aSzN,bool WithSimUK);
///  idem, but more adapted to real case (as in surface devlopment)
NS_SymbolicDerivative::cCalculator<double> * EqNetworkConsDistFixPoints(bool WithDerive,int aSzBuf,int aNbPts);

           // .............   Registration Lidar/Image   .............
NS_SymbolicDerivative::cCalculator<double> * EqEqLidarImPonct(bool WithDerive,int aSzBuf);
NS_SymbolicDerivative::cCalculator<double> * EqEqLidarImCensus(bool WithDerive,int aSzBuf);
NS_SymbolicDerivative::cCalculator<double> * EqEqLidarImCorrel(bool WithDerive,int aSzBuf);


};

#endif  //  _MMVII_PHGR_DIST_H_
