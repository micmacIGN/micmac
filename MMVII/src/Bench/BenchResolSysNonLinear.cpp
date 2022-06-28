#include "include/MMVII_all.h"
#include "include/MMVII_Tpl_Images.h"


// ==========  3 variable used for debuging  , will disappear
//
static constexpr double THE_AMPL_P = 0.1; // Coefficient of amplitude of the pertubation
static constexpr double    THE_TETA_SIM = 0;//  M_PI/2; // Sign of similitude for position init
static constexpr int    THE_NB_ITER  = 20; // Sign of similitude for position init

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
     # cBenchNetwork   represent the network  itself
*/
namespace NB_Bench_RSNL
{

   /* ======================================== */
   /*         HEADER                           */
   /* ======================================== */

template <class Type>  class  cBenchNetwork;
template <class Type>  class  cPNetwork;

template <class Type>  class  cPNetwork
{
      public :
            typedef cBenchNetwork<Type> tNetW;
            typedef cPtxd<Type,2>       tPt;

             /// Create a point with its number, grid position and the network itself
	    cPNetwork(int aNumPt,const cPt2di & aPTh,tNetW &);

            /**  Cur point ,  for "standard" point just access to the unknown NumX,NumY
                 for "schurs" points, as they are not stored, make an estimation with neighbourhood
            */
	    tPt  PCur() const;  
	    const tPt &  TheorPt() const;  ///< Acessor

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

template <class Type>  class  cBenchNetwork
{
	public :
          typedef cPtxd<Type,2>             tPt;
          typedef cPNetwork<Type>           tPNet;
          typedef tPNet *                   tPNetPtr;
          typedef cResolSysNonLinear<Type>  tSys;
          typedef NS_SymbolicDerivative::cCalculator<tREAL8>  tCalc;

          cBenchNetwork(eModeSSR aMode,int aN,bool WithSchurr,cParamSparseNormalLstSq * = nullptr);
          ~cBenchNetwork();

          int   N() const;
          bool WithSchur()  const;
          int&  Num() ;


	  Type OneItereCompensation();

          /// Access to CurSol of mSys
	  const Type & CurSol(int aK) const;

          /// Acces to a point from its number
	  const tPNet & PNet(int aK) const {return mVPts.at(aK);}

          /// Acces to a point from pixel value
	  tPNet & PNetOfGrid(const cPt2di  & aP) {return *(PNetPtrOfGrid(aP));}
          /// Is a Pixel in the grid [-N,N]^2
          bool IsInGrid(const cPt2di  & aP)  const
          {
               return (std::abs(aP.x())<=mN) && (std::abs(aP.y())<=mN) ;
          }

	  ///  Compute the geometry of an index using internal parameters
	  tPt  Ind2Geom(const cPt2di & anInd) const;

	  ///  Classically for the gauge fixing the direction by fixing a var we must take precaution
	  bool  XIsHoriz() const;
	private :
          /// Acces to reference of a adress if point from pixel value
	  tPNetPtr & PNetPtrOfGrid(const cPt2di  & aP) {return mMatrP[aP.y()+mN][aP.x()+mN];}
	  int   mN;                    ///< Size of network is  [-N,N]x[-N,N]
          int   mSzM;                  ///<  1+2*aN  = Sz of Matrix of point
	  bool  mWithSchur;            ///< Do we test Schurr complement
	  int   mNum;                  ///< Current num of unknown
	  std::vector<tPNet>  mVPts;   ///< Vector of point of unknowns coordinate
          tPNet ***           mMatrP;  ///< Indexed matrice of points, give basic spatial indexing
	  tSys *              mSys;    ///< Sys for solving non linear equations 
	  tCalc *             mCalcD;  ///< Equation that compute distance & derivate/points corrd
          cRect2              mBoxPix; ///< Box of pixel containing the points

	  /**  Similitude transforming the index in the geometry, use it to make the test more general, and also
	      to test co-variance tranfsert with geometric change  */
	   cSim2D<Type>        mSimInd2G;  
};

/* ======================================== */
/*                                          */
/*              cBenchNetwork               */
/*                                          */
/* ======================================== */

template <class Type> cBenchNetwork<Type>::cBenchNetwork
                      (
		            eModeSSR aMode,
                            int aN,
			    bool WithSchurr,
			    cParamSparseNormalLstSq * aParam
		      ) :
    mN         (aN),
    mSzM       (1+2*aN),
    mWithSchur (WithSchurr),
    mNum       (0),
    mMatrP     (cMemManager::AllocMat<tPNetPtr>(mSzM,mSzM)),
    mBoxPix    (cRect2::BoxWindow(mN)),
    // For now I dont understand why it doese not work with too big angle ?
     mSimInd2G  (cSim2D<Type>::RandomSimInv(5.0,3.0,1)) 
    // mSimInd2G  (tPt::PRandC()*Type(50.0),FromPolar(Type(1.0 + RandUnif_0_1()*2),Type(RandUnif_C()*0.5)))
    // mSimInd2G  (tPt(0,0),FromPolar(Type(1.0),Type(THE_TETA_SIM)))
{
     //  StdOut() <<  "ZZZZ  " << mSimInd2G.Tr() << " " <<mSimInd2G.Sc() <<  "XISH " << XIsHoriz() << "\n";

     // generate in VPix a regular grid, put them in random order for testing more config in matrix
     std::vector<cPt2di> aVPix;
     for (const auto& aPix: mBoxPix)
         aVPix.push_back(aPix);
     aVPix = RandomOrder(aVPix);

     std::vector<Type> aVCoord0; // initial coordinates for creating unknowns
     // Initiate Pts of Networks in mVPts,
     for (const auto& aPix: aVPix)
     {
         tPNet aPNet(mVPts.size(),aPix,*this);
         mVPts.push_back(aPNet);

	 if (! aPNet.mSchurrPoint)
	 {
             aVCoord0.push_back(aPNet.mPosInit.x());
             aVCoord0.push_back(aPNet.mPosInit.y());
	 }
     }
     for (auto & aPNet : mVPts)
         PNetPtrOfGrid(aPNet.mInd) = & aPNet;
         
     
     // Initiate system "mSys" for solving
     if ((aMode==eModeSSR::eSSR_LsqNormSparse)  && (aParam!=nullptr))
     {
         // LEASTSQ:CONSTRUCTOR , case Normal sparse, create first the least square
	 cLeasSq<Type>*  aLeasSQ =  cLeasSq<Type>::AllocSparseNormalLstSq(aVCoord0.size(),*aParam);
         mSys = new tSys(aLeasSQ,cDenseVect<Type>(aVCoord0));
     }
     else
     {
         // BASIC:CONSTRUCTOR other, just give the mode
         mSys = new tSys(aMode,cDenseVect<Type>(aVCoord0));
     }

     // compute links between Pts of Networks,
     for (const auto& aPix: mBoxPix)
     {
	  tPNet &  aPN1 = PNetOfGrid(aPix);
          for (const auto& aNeigh: cRect2::BoxWindow(aPix,1))
          {
              if (IsInGrid(aNeigh))
              {
	          tPNet &  aPN2 = PNetOfGrid(aNeigh);
                  // Test on num to do it only one way
                  if ((aPN1.mNumPt>aPN2.mNumPt) && aPN1.Linked(aPN2))
	          {
                       // create the links, be careful that for Tmp unknown all the links start from Tmp
		       // this will make easier the regrouping of equation concerning the same tmp
		       // the logic of this lines of code take use the fact that K1 and K2 cannot be both Tmp
		
                       if (aPN1.mSchurrPoint)  // K1 is Tmp and not K2, save K1->K2
                          aPN1.mLinked.push_back(aPN2.mNumPt);
		       else if (aPN2.mSchurrPoint) // K2 is Tmp and not K1, save K2->K2
                          aPN2.mLinked.push_back(aPN1.mNumPt);
		       else // None Tmp, does not matters which way it is stored
                          aPN1.mLinked.push_back(aPN2.mNumPt);  
	          }
             }
          }
     }
     // create the "functor" that will compute values and derivates
     mCalcD =  EqConsDist(true,1);
}

template <class Type> bool  cBenchNetwork<Type>::XIsHoriz() const
{
     const tPt& aSc = mSimInd2G.Sc();

     return std::abs(aSc.x()) > std::abs(aSc.y());
}

template <class Type> cPtxd<Type,2>  cBenchNetwork<Type>::Ind2Geom(const cPt2di & anInd) const
{
    return mSimInd2G.Value(tPt(anInd.x(),anInd.y()));
}

template <class Type> cBenchNetwork<Type>::~cBenchNetwork()
{
    cMemManager::FreeMat<tPNetPtr>(mMatrP,mSzM);
    delete mSys;
    delete mCalcD;
}

template <class Type> int   cBenchNetwork<Type>::N() const {return mN;}
template <class Type> bool  cBenchNetwork<Type>::WithSchur()  const {return mWithSchur;}
template <class Type> int&  cBenchNetwork<Type>::Num() {return mNum;}

template <class Type> Type cBenchNetwork<Type>::OneItereCompensation()
{
     Type aWeightFix=100.0; // arbitray weight for fixing the 3 variable X0,Y0,X1 (the "jauge")

     Type  aSomEc = 0;
     Type  aNbEc = 0;
     //  Compute dist to sol + add constraint for fixed var
     for (const auto & aPN : mVPts)
     {
        // Add distance between theoreticall value and curent
        if (! aPN.mSchurrPoint)
        {
            aNbEc++;
            aSomEc += Norm2(aPN.PCur() -aPN.TheorPt());
	    // StdOut() << aPN.PCur() - aPN.TheorPt() << aPN.mInd << "\n";
        }
        // EQ:FIXVAR
	// Fix X and Y for the two given points
	if (aPN.mFrozenY) // If Y is frozenn add equation fixing Y to its theoreticall value
           mSys->AddEqFixVar(aPN.mNumY,aPN.TheorPt().y(),aWeightFix);
	if (aPN.mFrozenX)   // If X is frozenn add equation fixing X to its theoreticall value
           mSys->AddEqFixVar(aPN.mNumX,aPN.TheorPt().x(),aWeightFix);
     }
     
     //  Add observation on distances

     for (const auto & aPN1 : mVPts)
     {
         // If PN1 is a temporary unknown we will use schurr complement
         if (aPN1.mSchurrPoint)
	 {
            // SCHURR:CALC
            cSetIORSNL_SameTmp<Type> aSetIO; // structure to grouping all equation relative to PN1
	    cPtxd<Type,2> aP1= aPN1.PCur(); // current value, required for linearization
            std::vector<Type> aVTmp{aP1.x(),aP1.y()};  // vectors of temporary
	    // Parse all obsevation on PN1
            for (const auto & aI2 : aPN1.mLinked)
            {
                const tPNet & aPN2 = mVPts.at(aI2);
	        //std::vector<int> aVIndMixt{aPN2.mNumX,aPN2.mNumY,-1,-1};  // Compute index of unknowns for this equation
	        std::vector<int> aVIndMixt{-1,-1,aPN2.mNumX,aPN2.mNumY};  // Compute index of unknowns for this equation
                std::vector<Type> aVObs{Type(Norm2(aPN1.TheorPt()-aPN2.TheorPt()))}; // compute observations=target distance
                // Add eq in aSetIO, using CalcD intantiated with VInd,aVTmp,aVObs
		mSys->AddEq2Subst(aSetIO,mCalcD,aVIndMixt,aVTmp,aVObs);
	    }
	    mSys->AddObsWithTmpUK(aSetIO);
	 }
	 else
	 {
               // BASIC:CALC Simpler case no temporary unknown, just add equation 1 by 1
               for (const auto & aI2 : aPN1.mLinked)
               {
                    const tPNet & aPN2 = mVPts.at(aI2);
	            std::vector<int> aVInd{aPN1.mNumX,aPN1.mNumY,aPN2.mNumX,aPN2.mNumY};  // Compute index of unknowns
                    std::vector<Type> aVObs{Type(Norm2(aPN1.TheorPt()-aPN2.TheorPt()))};  // compute observations=target distance
                    // Add eq  using CalcD intantiated with VInd and aVObs
	            mSys->CalcAndAddObs(mCalcD,aVInd,aVObs);
	       }
	 }
     }

     mSys->SolveUpdateReset();
     return aSomEc / aNbEc ;
}
template <class Type> const Type & cBenchNetwork<Type>::CurSol(int aK) const
{
    return mSys->CurSol(aK);
}

/* ======================================== */
/*                                          */
/*              cPNetwork                   */
/*                                          */
/* ======================================== */

template <class Type> cPNetwork<Type>::cPNetwork(int aNumPt,const cPt2di & anInd,cBenchNetwork<Type> & aNet) :
     mNumPt    (aNumPt),
     mInd      (anInd),
     // mTheorPt (tPt(mInd.x(),mInd.y())),
     // mTheorPt (tPt(mInd.x(),mInd.y()) + tPt(10,5)),
     // mTheorPt  (tPt(mInd.x(),mInd.y())* tPt(1,-1) + tPt(10,5) ), //+  tPt::PRandC()*Type(0.005)),
     mTheorPt  (aNet.Ind2Geom(mInd)),
     mNetW     (&aNet),
	//  Tricky set cPt2di(-1,0)) to avoid interact with schurr points
     mFrozenX  (( mInd==cPt2di(0,0)  ) ||  (  aNet.XIsHoriz() ? (mInd==cPt2di(0,1)) : (mInd==cPt2di(-1,0)))),
     mFrozenY  ( mInd==cPt2di(0,0)  ), // fix origin
     mSchurrPoint    (aNet.WithSchur() && (mInd.x()==1)),  // If test schur complement, Line x=1 will be temporary
     mNumX     (-1),
     mNumY     (-1)
{
     //  To assess the correctness of our code , we must prove that we are able to recover the "real" position from
     //  a pertubated one;  the perturbation must be sufficiently complicated to be sure that position is not recovered 
     //  by "chance" , but also not to big to be sure that the gradient descent will work
     //  The pertubation is the a mix of sytematism and random, all is being mulitplied by some amplitude (aAmplP)
     
     {
        double aAmplP = THE_AMPL_P; // Coefficient of amplitude of the pertubation
	Type aSysX = -mInd.x() +  mInd.y()/2.0 +std::abs(mInd.y());  // sytematism on X : Linear + non linear abs
        Type aSysY  =  mInd.y() + 4*Square(mInd.x()/Type(aNet.N())); // sytematism on U : Linear + quadratic term

        mPosInit.x() =  mTheorPt.x() + aAmplP*(aSysX + 2*RandUnif_C());;
        mPosInit.y() =  mTheorPt.y() + aAmplP*(aSysY + 2*RandUnif_C() );
     }

     //  To fix globally the network (gauge) 3 coordinate are frozen, for these one the pertubation if void
     //  so that recover the good position
     if (mFrozenX)
       mPosInit.x() = mTheorPt.x();
     if (mFrozenY)
       mPosInit.y() = mTheorPt.y();


     if (!mSchurrPoint)
     {
        mNumX = aNet.Num()++;
        mNumY = aNet.Num()++;
     }
}
template <class Type> cPtxd<Type,2>  cPNetwork<Type>::PCur() const
{
	// For standard unknown, read the cur solution of the system
    if (!mSchurrPoint)
	return cPtxd<Type,2>(mNetW->CurSol(mNumX),mNetW->CurSol(mNumY));

    /*  For temporary unknown we must compute the "best guess" as we do by bundle intersection.
     
        If it was the real triangulation problem, we would compute the best circle intersection
	with all the linked points, but it's a bit complicated and we just want to check software
	not resolve the "real" problem.

	An alternative would be to use the theoreticall value, but it's too much cheating and btw
	may be a bad idea for linearization if too far from current solution.

	As an easy solution we take the midle of PCur(x-1,y) and PCur(x+1,y).
       */

    int aNbPts = 0;
    cPtxd<Type,2> aSomP(0,0);
    for (const auto & aI2 : mLinked)
    {
           const  cPNetwork<Type> & aPN2 = mNetW->PNet(aI2);
	   if (mInd.y() == aPN2.mInd.y())
	   {
               aSomP +=  aPN2.PCur();
               aNbPts++;
           }
    }
    MMVII_INTERNAL_ASSERT_bench((aNbPts==2),"Bad hypothesis for network");

    return aSomP / (Type) aNbPts;
}

template <class Type> const cPtxd<Type,2> &  cPNetwork<Type>::TheorPt() const
{
	return mTheorPt;
}

template <class Type> bool cPNetwork<Type>::Linked(const cPNetwork<Type> & aP2) const
{
   // Precaution, a poinnt is not linked yo itself
   if (mInd== aP2.mInd) 
      return false;

   //  If two temporay point, they are not observable
   if (mSchurrPoint && aP2.mSchurrPoint)
      return false;

    //  else point are linked is they are same column, or neighbooring colums
    return NormInf(mInd-aP2.mInd) <=1;
}


/* ======================================== */
/*                                          */
/*              ::                          */
/*                                          */
/* ======================================== */

/** Make on test with different parameter, check that after 10 iteration we are sufficiently close
    to "real" network
*/
template<class Type> void  TplOneBenchSSRNL
                           (
                               eModeSSR aMode,
                               int aNb,
                               bool WithSchurr,
                               cParamSparseNormalLstSq * aParam=nullptr
                           )
{
     Type aPrec = tElemNumTrait<Type>::Accuracy() ;
     cBenchNetwork<Type> aBN(aMode,aNb,WithSchurr,aParam);
     double anEc =100;
     for (int aK=0 ; aK < THE_NB_ITER ; aK++)
     {
         anEc = aBN.OneItereCompensation();
         // StdOut() << "  ECc== " << anEc /aPrec<< "\n";
     }
     // StdOut() << "Fin-ECc== " << anEc  / aPrec << " Nb=" << aNb << "\n";
     // getchar();
     MMVII_INTERNAL_ASSERT_bench(anEc<aPrec,"Error in Network-SSRNL Bench");
}

void  OneBenchSSRNL(eModeSSR aMode,int aNb,bool WithSchurr,cParamSparseNormalLstSq * aParam=nullptr)
{
    TplOneBenchSSRNL<tREAL8>(aMode,aNb,WithSchurr,aParam);
    TplOneBenchSSRNL<tREAL16>(aMode,aNb,WithSchurr,aParam);
    TplOneBenchSSRNL<tREAL4>(aMode,aNb,WithSchurr,aParam);
}

};

using namespace NB_Bench_RSNL;

void BenchSSRNL(cParamExeBench & aParam)
{
     if (! aParam.NewBench("SSRNL")) return;

     OneBenchSSRNL(eModeSSR::eSSR_LsqDense ,1,false);
     OneBenchSSRNL(eModeSSR::eSSR_LsqDense ,2,false);

     // Basic test, test the 3 mode of matrix , with and w/o schurr subst
     for (const auto &  aNb : {3,4,5})
     {
        cParamSparseNormalLstSq aParamSq(3.0,4,9);
	// w/o schurr
        OneBenchSSRNL(eModeSSR::eSSR_LsqNormSparse,aNb,false,&aParamSq);
        OneBenchSSRNL(eModeSSR::eSSR_LsqSparseGC,aNb,false);
        OneBenchSSRNL(eModeSSR::eSSR_LsqDense ,aNb,false);

	// with schurr
         OneBenchSSRNL(eModeSSR::eSSR_LsqNormSparse,aNb,true ,&aParamSq);
         OneBenchSSRNL(eModeSSR::eSSR_LsqDense ,aNb,true);
         OneBenchSSRNL(eModeSSR::eSSR_LsqSparseGC,aNb,true);
     }


     // test  normal sparse matrix with many parameters
     for (int aK=0 ; aK<20 ; aK++)
     {
        int aNb = 3+ RandUnif_N(3);
	int aNbVar = 2 * Square(2*aNb+1);
        cParamSparseNormalLstSq aParamSq(3.0,RandUnif_N(3),RandUnif_N(10));

	// add random subset of dense variable
	for (const auto & aI:  RandSet(aNbVar/10,aNbVar))
           aParamSq.mVecIndDense.push_back(size_t(aI));

        TplOneBenchSSRNL<tREAL8>(eModeSSR::eSSR_LsqNormSparse,aNb,false,&aParamSq); // w/o schurr
        TplOneBenchSSRNL<tREAL8>(eModeSSR::eSSR_LsqNormSparse,aNb,true ,&aParamSq); // with schurr
     }


     aParam.EndBench();
}


};
