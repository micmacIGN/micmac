#include "include/MMVII_all.h"
#include "include/MMVII_Tpl_Images.h"


// ==========  3 variable used for debuging  , will disappear
//
using namespace NS_SymbolicDerivative;
using namespace MMVII;

namespace MMVII
{
/* ************************************************************ */
/*                                                              */
/*                  BENCH                                       */
/*                                                              */
/* ************************************************************ */

/*   To check some correctness  on cResolSysNonLinear, we will do the following stuff
     which is more or less a simulation of triangulation
 
     #  create a network for which we have approximate coordinate  (except few point for 
        which they are exact) and exact mesure of distances between pair of points

     # we try to recover the coordinates using compensation on distances


     The network is made of [-N,N] x [-N,N],  as the preservation of distance would not be sufficient for
     uniqueness of solution, some arbitrary constraint are added on "frozen" points  (X0=0,Y0=0 and X1=0)

Classes :
     # cPNetwork       represent one point of the network 
     # cMainNetwork   represent the network  itself
*/
namespace NS_Bench_RSNL
{
// #define  DEBUG_RSNL true


template <class Type>  class  cMainNetwork; 
template <class Type>  class  cPNetwork;

// This class is used  only in covariance propagation

template <class Type>  class  cPNetwork
{
      public :
            typedef cMainNetwork<Type> tNetW;
            typedef cPtxd<Type,2>       tPt;

             /// Create a point with its number, grid position and the network itself
	    cPNetwork(int aNumPt,const cPt2di & aPTh,tNetW &);

            /**  Cur point ,  for "standard" point just access to the unknown NumX,NumY
                 for "schurs" points, as they are not stored, make an estimation with neighbourhood
            */
	    tPt  PCur() const;  
	    const tPt &  TheorPt() const;  ///< Acessor
	    const tPt &  PosInit() const;  ///< Acessor

            /// Compute initial guess : add some noise+some systematism to "real" position
            void MakePosInit(const double & aMulAmpl);

	    /// Are the two point linked  (will their distances be an observation compensed)
	    bool Linked(const cPNetwork<Type> & aP2) const;

            int            mNumPt;     ///< Num in vector
            cPt2di         mInd;       ///< Index in the grid
            tPt            mTheorPt;  ///< Theoreticall position; used to compute distances and check accuracy recovered
	    const tNetW *  mNetW;    ///<  link to the network itself
            tPt            mPosInit; ///< initial position : pertubation of theoretical one
	    bool           mFrozenX; ///< is abscisse of this point frozen
	    bool           mFrozenY;  ///< is this point frozen
	    bool           mSchurrPoint;   ///< is it a temporay point (point not computed, for testing schur complement)
	    int            mNumX;    ///< Num of x unknown
	    int            mNumY;    ///< Num of y unknown

	    std::list<int> mLinked;   ///< list of linked points, if Tmp/UK the links start from tmp, if Uk/Uk order does not matters
};

class cParamMainNW
{
    public :
       cParamMainNW();

       double mAmplGrid2Real;   // Amplitude of random differerence between real position and regular grid
       double mAmplReal2Init;  // Amplitude of random and syst differerence  betwen real an init position
};

template <class Type>  class  cMainNetwork
{
	public :

          typedef cPtxd<Type,2>             tPt;
          typedef cPNetwork<Type>           tPNet;
          typedef tPNet *                   tPNetPtr;
          typedef cResolSysNonLinear<Type>  tSys;
          typedef NS_SymbolicDerivative::cCalculator<tREAL8>  tCalc;


          cMainNetwork(eModeSSR aMode,cRect2,bool WithSchurr,const cParamMainNW &,cParamSparseNormalLstSq * = nullptr);
          /// Do real construction that cannot be made in constructor do to call to virtual funcs
          virtual void PostInit();
          virtual ~cMainNetwork();

          //int   N() const;
          bool WithSchur()  const;
          int&  Num() ;
	  Type  NetSz() const {return Norm2(mBoxInd.Sz());}

          /// If we use this iteration for covariance calculation , we dont add constraint, and dont solve
	  Type DoOneIterationCompensation(double  aWeigthGauge,bool WithCalcReset);

	  Type CalcResidual();
	  void AddGaugeConstraint(Type aWeight);


          /// Access to CurSol of mSys
	  const Type & CurSol(int aK) const;

          /// Acces to a point from its number
	  const tPNet & PNet(int aK) const {return mVPts.at(aK);}

          /// Acces to a point from pixel value
	  tPNet & PNetOfGrid(const cPt2di  & aP) {return *(PNetPtrOfGrid(aP));}
          /// Is a Pixel in the grid [-N,N]^2
          bool IsInGrid(const cPt2di  & aP)  const
          {
               // return (std::abs(aP.x())<=mN) && (std::abs(aP.y())<=mN) ;
               return mBoxInd.Inside(aP);
          }

	  ///  Compute the geometry of an index using internal parameters => global simi + some random value
	  virtual tPt  ComputeInd2Geom(const cPt2di & anInd) const;
	  ///  Compute the geometry in case of cov propag
	  //  tPt  CovPropInd2Geom(const cPt2di & anInd) const;

	  /**  Classically for the gauge fixing the direction by fixing some specific var, we must take precaution
               i.e if P0=(0,0) is fixed  and P1=(1,0),  if we fix   x1=Cste for the gauge, the 
               system will be degenerated as the distance  on P1 are unsentive to y1, so we must
               fix x1=Cste or y1=Cste depending if AxeXIsHoriz
          */
	  bool  AxeXIsHoriz() const;

          const cSim2D<Type> &  SimInd2G() const;  ///<Accessor
          const cParamMainNW &  ParamNW() const;   ///<Accessor
	  tSys * Sys();


	protected :
          /// Acces to reference of a adress if point from pixel value
	  tPNetPtr & PNetPtrOfGrid(const cPt2di  & aP) {return mMatrP[aP.y()-mBoxInd.P0().y()][aP.x()-mBoxInd.P0().x()];}

          eModeSSR mModeSSR;             ///< Mode for allocating Sys Over Constrained
          cParamSparseNormalLstSq * mParamLSQ; ///< Additional parameter for allocating sparse
	  cRect2 mBoxInd;                ///< rectangle of the network
          int   mX_SzM;                  ///<  1+2*aN  = Sz of Matrix of point
          int   mY_SzM;                  ///<  1+2*aN  = Sz of Matrix of point
	  bool  mWithSchur;            ///< Do we test Schurr complement
          cParamMainNW  mParamNW;
	  int   mNum;                  ///< Current num of unknown
	  std::vector<tPNet>  mVPts;   ///< Vector of point of unknowns coordinate
          tPNet ***           mMatrP;  ///< Indexed matrice of points, give basic spatial indexing
	  tSys *              mSys;    ///< Sys for solving non linear equations 
	  tCalc *             mCalcD;  ///< Equation that compute distance & derivate/points corrd

	  /**  Similitude transforming the index in the geometry, use it to make the test more general, and also
	      to test co-variance tranfsert with geometric change  */
          cSim2D<Type>        mSimInd2G;  

          cBox2dr     mBoxPts;  /// Box englobing Theor + Init
};
}; // namespace NS_Bench_RSNL
}; // namespace MMVII
