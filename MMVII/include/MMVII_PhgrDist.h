#ifndef  _MMVII_PHGR_DIST_H_
#define  _MMVII_PHGR_DIST_H_


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
              eNbVals ///< Tag for number of value
           };

/**  This class store a complete description of each parameters of the distorsion,
     it is used for computing the formula, the vector of name and (later) automatize
     conversion, print understandable values ....

     It's rather bad design with the same classe coding different things, and some fields
     used of not according to the others, but as it internal/final classes, quick and dirty
     exceptionnaly allowed ...
*/
class cDescOneFuncDist
{
    public :
      cDescOneFuncDist(eTypeFuncDist aType,const cPt2di aDeg);
      /// Majorarion of norms of jacobian , used in simulation
      double MajNormJacOfRho(double aRho) const;

      eTypeFuncDist mType;   ///< Type of dist (Rad, DecX ....)
      std::string   mName;   ///< Name : used as id for code gen & prety print
      cPt2di        mDegMon;  ///< Degree for a polynomial
      int           mNum;     ///< Order for radial and decentric
      int           mDegTot;  ///< Total degree of the polynome
};


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
              double aTargetSomJac   ///< Target majoration of jacobian
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
       tCalc *                         mEqVal;   ///< Calculator for values only
       tCalc *                         mEqDer;   ///< Calculator for values and derivatoves
       int                             mNbParam; ///< Number of parameters
       std::vector<double>             mVParam;  ///< Computed parameters
       std::vector<cDescOneFuncDist>   mVecDesc; ///< Vector of descriptor
};





///  Allocate a calculator computing the distorsion
NS_SymbolicDerivative::cCalculator<double> * EqDist(const cPt3di & aDeg,bool WithDerive,int aSzBuf);
/// Allocate a calculator computing the base familly of a distorsion
NS_SymbolicDerivative::cCalculator<double> * EqBaseFuncDist(const cPt3di & aDeg,int aSzBuf);
/// Alloc a map corresponding to  distorsions
cDataNxNMapCalcSymbDer<double,2> * NewMapOfDist(const cPt3di & aDeg,const std::vector<double> & aVObs,int aSzBuf);

std::vector<cDescOneFuncDist>   DescDist(const cPt3di & aDeg);

/// Calc for conservation of dist, Uk={x1,y1,x2,y2} Obs={D12} , let pk=(xk,yk)  Residual :  D(p1,p2)/d12 -1
NS_SymbolicDerivative::cCalculator<double> * EqConsDist(bool WithDerive,int aSzBuf);
/// Ratio of dist, Uk={x1,y1,x2,y2,x3,y3} Obs={D12,D13,D23} ,  3 Residuals as D(p1,p2)/D12 - D(p1,p3)/D13
NS_SymbolicDerivative::cCalculator<double> * EqConsRatioDist(bool WithDerive,int aSzBuf);

/// Equation used to optimize homothetic transform between model and image (used as a tutorial for deformable model)
NS_SymbolicDerivative::cCalculator<double> * EqDeformImHomotethy(bool WithDerive,int aSzBuf);


/// For propag cov in network 
NS_SymbolicDerivative::cCalculator<double> * EqNetworkConsDistProgCov(bool WithDerive,int aSzBuf,const cPt2di& aSzN);
/// For propag in network  W/O covariance (i.e propagate directly the points)
NS_SymbolicDerivative::cCalculator<double> * EqNetworkConsDistFixPoints(bool WithDerive,int aSzBuf,const cPt2di& aSzN,bool WithSimUK);





};

#endif  //  _MMVII_PHGR_DIST_H_
