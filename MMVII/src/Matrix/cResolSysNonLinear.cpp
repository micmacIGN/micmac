#include "include/MMVII_all.h"
#include "include/MMVII_Tpl_Images.h"


using namespace NS_SymbolicDerivative;
using namespace MMVII;

namespace MMVII
{

template <class Type> class cResidualWeighter
{
       public :
	    typedef std::vector<Type>     tStdVect;

            cResidualWeighter();
	    virtual tStdVect WeightOfResidual(const tStdVect &) const;
       private :
            
};

template <class Type> class cResolSysNonLinear
{
      public :
          typedef NS_SymbolicDerivative::cCalculator<Type>  tCalc;
	  typedef cLinearOverCstrSys<Type>                  tSysSR;
	  typedef cDenseVect<Type>                          tDVect;
	  typedef cSparseVect<Type>                         tSVect;
	  typedef std::vector<Type>                         tStdVect;
	  typedef std::vector<int>                          tVectInd;
	  typedef cResolSysNonLinear<Type>                  tRSNL;
          typedef cInputOutputRSNL<Type>                    tIO_TSNL;
	  typedef cResidualWeighter<Type>                   tResidualW;

	  cResolSysNonLinear(eModeSSR,const tDVect & aInitSol);
	  ~cResolSysNonLinear();
	  
	  /// Accessor
          const tDVect  &    CurGlobSol() const;
	  /// Value of a given num var
          const Type  &    CurSol(int aNumV) const;
	  
	  /// Solve solution,  update the current solution, Reset the least square system
          const tDVect  &    SolveUpdateReset() ;

	  /// Add 1 equation fixing variable
	  void   AddEqFixVar(const int & aNumV,const Type & aVal,const Type& aWeight);
	  /// Add equation to fix value 
	  void   AddEqFixCurVar(const int & aNumV,const Type& aWeight);


	  /// Basic Add 1 equation , no bufferistion, no schur complement
	  void   CalcAndAddObs(tCalc *,const tVectInd &,const tStdVect& aVObs,const tResidualW & = tResidualW());

	  ///  Add 1 equation in structure aSetIO , relatively basic 4 now because don't use parallelism
	  void  AddEq2Subst (cSetIORSNL_SameTmp<Type> & aSetIO,tCalc *,const tVectInd &,const tStdVect& aVTmp,
			     const tStdVect& aVObs,const tResidualW & = tResidualW());
	  void  AddObsWithTmpUK (const cSetIORSNL_SameTmp<Type> & aSetIO);
      private :
	  cResolSysNonLinear(const tRSNL & ) = delete;

	  /// Add observations as computed by CalcVal
	  void   AddObs(const std::vector<tIO_TSNL>&);

	  /** Bases function of calculating derivatives, dont modify the system as is
	      to avoid in case  of schur complement */
	  void   CalcVal(tCalc *,std::vector<tIO_TSNL>&,bool WithDer,const tResidualW & );

	  int        mNbVar;       ///< Number of variable, facility
          tDVect     mCurGlobSol;  ///< Curent solution
          tSysSR*    mSys;         ///< Sys to solve equations, equation are concerning the differences with current solution
};

/* ************************************************************ */
/*                                                              */
/*                cInputOutputRSNL                              */
/*                                                              */
/* ************************************************************ */

template <class Type>  cInputOutputRSNL<Type>::cInputOutputRSNL(const tVectInd& aVInd,const tStdVect & aVObs):
     mVInd  (aVInd),
     mObs   (aVObs)
{
}

template <class Type>  cInputOutputRSNL<Type>::cInputOutputRSNL(const tVectInd& aVInd,const tStdVect & aVTmp,const tStdVect & aVObs):
	cInputOutputRSNL<Type>(aVInd,aVObs)
{
	mTmpUK = aVTmp;
}

template <class Type> Type cInputOutputRSNL<Type>::WeightOfKthResisual(int aK) const
{
   switch (mWeights.size())
   {
	   case 0 :  return 1.0;
	   case 1 :  return mWeights[0];
	   default  : return mWeights.at(aK);
   }
}
template <class Type> size_t cInputOutputRSNL<Type>::NbUkTot() const
{
	return mVInd.size() + mTmpUK.size();
}

template <class Type> bool cInputOutputRSNL<Type>::IsOk() const
{
     if (mVals.size() !=mDers.size()) 
        return false;

     if (mVals.empty())
        return false;

     {
         size_t aNbUk = NbUkTot();
         for (const auto & aDer : mDers)
             if (aDer.size() != aNbUk)
                return false;
     }

     {
         size_t aSzW =  mWeights.size();
         if ((aSzW>1) && (aSzW!= mVals.size()))
            return false;
     }
     return true;
}


/* ************************************************************ */
/*                                                              */
/*                cSetIORSNL_SameTmp                            */
/*                                                              */
/* ************************************************************ */

template <class Type> cSetIORSNL_SameTmp<Type>::cSetIORSNL_SameTmp() :
	mOk (false),
	mNbEq (0)
{
}

template <class Type> void cSetIORSNL_SameTmp<Type>::AddOneEq(const tIO_OneEq & anIO)
{
    if (!mVEq.empty())
    {
         MMVII_INTERNAL_ASSERT_tiny
         (
             (anIO.mTmpUK.size()==mVEq.back().mTmpUK.size()),
	     "Variable size of temporaries"
         );
    }
    MMVII_INTERNAL_ASSERT_tiny(anIO.IsOk(),"Bad size for cInputOutputRSNL");

    mVEq.push_back(anIO);
    mNbEq += anIO.mVals.size();
    // A priori there is no use to less or equal equation, this doesnt give any constraint
    if (mNbEq > anIO.mTmpUK.size())
    {
        mOk = true; 
    }
}


template <class Type> 
    const std::vector<cInputOutputRSNL<Type> >& 
          cSetIORSNL_SameTmp<Type>::AllEq() const
{
     return mVEq;
}

template <class Type> void cSetIORSNL_SameTmp<Type>::AssertOk() const
{
      MMVII_INTERNAL_ASSERT_tiny(mOk,"Not enough eq to use tmp unknowns");
}

template <class Type> size_t cSetIORSNL_SameTmp<Type>::NbTmpUk() const
{
    return mVEq.at(0).mTmpUK.size();
}

/* ************************************************************ */
/*                                                              */
/*                cResidualWeighter                             */
/*                                                              */
/* ************************************************************ */

template <class Type>  cResidualWeighter<Type>::cResidualWeighter()
{
}

template <class Type>  std::vector<Type>  cResidualWeighter<Type>::WeightOfResidual(const tStdVect & aVResidual) const
{
	return tStdVect(aVResidual.size(),1.0);
}


/* ************************************************************ */
/*                                                              */
/*                cResolSysNonLinear                            */
/*                                                              */
/* ************************************************************ */

template <class Type> cResolSysNonLinear<Type>::cResolSysNonLinear(eModeSSR aMode,const tDVect & aInitSol) :
    mNbVar      (aInitSol.Sz()),
    mCurGlobSol (aInitSol.Dup()),
    // mSys        (new cLeasSqtAA<Type>(mNbVar))
    mSys        (cLinearOverCstrSys<Type>::AllocSSR(aMode,mNbVar))
{
}

template <class Type> void   cResolSysNonLinear<Type>::AddEqFixVar(const int & aNumV,const Type & aVal,const Type& aWeight)
{
     tSVect aSV;
     aSV.AddIV(aNumV,1.0);
     mSys->AddObservation(aWeight,aSV,aVal);
}


template <class Type> const cDenseVect<Type> & cResolSysNonLinear<Type>::CurGlobSol() const 
{
    return mCurGlobSol;
}
template <class Type> const Type & cResolSysNonLinear<Type>::CurSol(int aNumV) const
{
    return mCurGlobSol(aNumV);
}

template <class Type> const cDenseVect<Type> & cResolSysNonLinear<Type>::SolveUpdateReset() 
{
    // mCurGlobSol += mSys->Solve();
    mCurGlobSol += mSys->SparseSolve();
    mSys->Reset();

    return mCurGlobSol;
}


template <class Type> void   cResolSysNonLinear<Type>::AddEqFixCurVar(const int & aNumV,const Type& aWeight)
{
     AddEqFixVar(aNumV,mCurGlobSol(aNumV),aWeight);
}

template <class Type> void cResolSysNonLinear<Type>::CalcAndAddObs
                           (
                                  tCalc * aCalcVal,
			          const tVectInd & aVInd,
				  const tStdVect& aVObs,
				  const tResidualW & aWeigther
                            )
{
    std::vector<tIO_TSNL> aVIO(1,tIO_TSNL(aVInd,aVObs));

    CalcVal(aCalcVal,aVIO,true,aWeigther);
    AddObs(aVIO);
}


template <class Type> cResolSysNonLinear<Type>::~cResolSysNonLinear()
{
    delete mSys;
}


template <class Type> void cResolSysNonLinear<Type>::AddObs ( const std::vector<tIO_TSNL>& aVIO)
{
      // Parse all the linearized equation
      for (const auto & aIO : aVIO)
      {
	  // check we dont use temporary value
          MMVII_INTERNAL_ASSERT_tiny(aIO.mTmpUK.empty(),"Cannot use tmp uk w/o Schurr complement");

	  // parse all values
	  for (size_t aKVal=0 ; aKVal<aIO.mVals.size() ; aKVal++)
	  {
	      Type aW = aIO.WeightOfKthResisual(aKVal);
	      if (aW>0)
	      {
	         tSVect aSV;
		 const tStdVect & aVDer = aIO.mDers[aKVal];
	         for (size_t aKUk=0 ; aKUk<aIO.mVInd.size() ; aKUk++)
                 {
                     aSV.AddIV(aIO.mVInd[aKUk],aVDer[aKUk]);
	         }
		 // Note the minus sign :  F(X0+dx) = F(X0) + Gx.dx   =>   Gx.dx = -F(X0)
	         mSys->AddObservation(aW,aSV,-aIO.mVals[aKVal]);
	      }
	  }

      }
}


template <class Type> void   cResolSysNonLinear<Type>::AddEq2Subst 
                             (
			          cSetIORSNL_SameTmp<Type> & aSetIO,tCalc * aCalc,const tVectInd & aVInd,const tStdVect& aVTmp,
			          const tStdVect& aVObs,const tResidualW & aWeighter
			     )
{
    std::vector<tIO_TSNL> aVIO(1,tIO_TSNL(aVInd,aVTmp,aVObs));
    CalcVal(aCalc,aVIO,true,aWeighter);

    aSetIO.AddOneEq(aVIO.at(0));
}
			     
template <class Type> void cResolSysNonLinear<Type>::AddObsWithTmpUK (const cSetIORSNL_SameTmp<Type> & aSetIO)
{
    mSys->AddObsWithTmpUK(aSetIO);
}

template <class Type> void   cResolSysNonLinear<Type>::CalcVal
                             (
			          tCalc * aCalcVal,
				  std::vector<tIO_TSNL>& aVIO,
				  bool WithDer,
				  const tResidualW & aWeighter
                              )
{
      MMVII_INTERNAL_ASSERT_tiny(aCalcVal->NbInBuf()==0,"Buff not empty");

      // Put input data
      for (const auto & aIO : aVIO)
      {
          tStdVect aVCoord;
	  // transferate global coordinates
	  for (const auto & anInd : aIO.mVInd)
              aVCoord.push_back(mCurGlobSol(anInd));
	  // transferate potential temporary coordinates
	  for (const  auto & aVal : aIO.mTmpUK)
              aVCoord.push_back(aVal);
	  //  Add equation in buffer
          aCalcVal->PushNewEvals(aVCoord,aIO.mObs);
      }
      // Make the computation
      aCalcVal->EvalAndClear();

      // Put output data
      size_t aNbEl = aCalcVal->NbElem();
      size_t aNbUk = aCalcVal->NbUk();
      // Parse all equation computed
      for (int aNumPush=0 ; aNumPush<int(aVIO.size()) ; aNumPush++)
      {
           auto & aIO = aVIO.at(aNumPush);
	   aIO.mVals = tStdVect(aNbEl);
	   if (WithDer)
	       aIO.mDers = std::vector(aNbEl,tStdVect( aNbUk));  // initialize vector to good size
	   // parse different values of each equation
           for (size_t aKEl=0; aKEl<aNbEl  ; aKEl++)
	   {
               aIO.mVals.at(aKEl) = aCalcVal->ValComp(aNumPush,aKEl);
	       if (WithDer)
	       {
	            // parse  all unknowns
	            for (size_t aKUk =0 ; aKUk<aNbUk ; aKUk++)
		    {
                        aIO.mDers.at(aKEl).at(aKUk) = aCalcVal->DerComp(aNumPush,aKEl,aKUk);
		    }
               }
	   }
           aIO.mWeights = aWeighter.WeightOfResidual(aIO.mVals);
      }
}

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

	    cPNetwork(const cPt2di & aPTh,tNetW &);

	    cPtxd<Type,2>  PCur() const;  ///< Acessor
	    cPtxd<Type,2>  PTh() const;  ///< Acessor

	    /// Are the two point linked  (will their distances be an observation compensed)
	    bool Linked(const cPNetwork<Type> & aP2) const;

            cPt2di         mPosTh;  // Theoreticall position; used to compute distances and check accuracy recovered
	    const tNetW *  mNetW;    //  link to the network itself
            cPtxd<Type,2>  mPosInit; // initial position : pertubation of theoretical one
	    bool           mFrozen;  // is this point frozen
	    bool           mFrozenX; // is abscisse of this point frozen
	    bool           mTmpUk;   // is it a temporay point (point not computed, for testing schur complement)
	    int            mNumX;    // Num of x unknown
	    int            mNumY;    // Num of y unknown

	    std::list<int> mLinked;   // if Tmp/UK the links start from tmp, if Uk/Uk does not matters
};

template <class Type>  class  cBenchNetwork
{
	public :
          typedef cPNetwork<Type>           tPNet;
          typedef cResolSysNonLinear<Type>  tSys;
          typedef NS_SymbolicDerivative::cCalculator<Type>  tCalc;

          cBenchNetwork(eModeSSR aMode,int aN,bool WithSchurr);
          ~cBenchNetwork();

          int   N() const;
          bool WithSchur()  const;
          int&  Num() ;


	  Type OneItereCompensation();

	  const Type & CurSol(int aK) const;
	  const tPNet & PNet(int aK) const {return mVPts.at(aK);}

	private :
	  int   mN;                    ///< Size of network is  [-N,N]x[-N,N]
	  bool  mWithSchur;            ///< Do we test Schurr complement
	  int   mNum;                  ///< Current num of unknown
	  std::vector<tPNet>  mVPts;   ///< Vector of point of unknowns coordinate
	  std::list<cPt2di>   mListCple;  ///< List of pair of point that "interact"
	  tSys *              mSys;    ///< Sys for solving non linear equations 
	  tCalc *             mCalcD;  ///< Equation that compute distance & derivate/points corrd
};

/* ======================================== */
/*                                          */
/*              cBenchNetwork               */
/*                                          */
/* ======================================== */

template <class Type> cBenchNetwork<Type>::cBenchNetwork(eModeSSR aMode,int aN,bool WithSchurr) :
    mN         (aN),
    mWithSchur (WithSchurr),
    mNum       (0)
{
     // Initiate Pts of Networks,
     std::vector<Type> aVCoord0;
     for (const auto& aPix: cRect2::BoxWindow(mN))
     {
         tPNet aP(aPix,*this);
         mVPts.push_back(aP);
	 if (! aP.mTmpUk)
	 {
             aVCoord0.push_back(aP.mPosInit.x());
             aVCoord0.push_back(aP.mPosInit.y());
	 }
     }
     // Initiate system for solving
     mSys = new tSys(aMode,cDenseVect<Type>(aVCoord0));

     // Initiate Links between Pts of Networks,
     for (size_t aK1=0 ;aK1<mVPts.size() ; aK1++)
     {
         for (size_t aK2=aK1+1 ;aK2<mVPts.size() ; aK2++)
	 {
             if (mVPts[aK1].Linked(mVPts[aK2]))
	     {
                if (mVPts[aK1].mTmpUk)
                    mVPts[aK1].mLinked.push_back(aK2);
		else if (mVPts[aK2].mTmpUk)
                    mVPts[aK2].mLinked.push_back(aK1);
		else 
                    mVPts[aK1].mLinked.push_back(aK2);  // None Tmp, does not matters which way it is stored
                 mListCple.push_back(cPt2di(aK1,aK2));
	     }
	 }
     }

     mCalcD =  EqConsDist(true,1);
}

template <class Type> cBenchNetwork<Type>::~cBenchNetwork()
{
    delete mSys;
    delete mCalcD;
}

template <class Type> int   cBenchNetwork<Type>::N() const {return mN;}
template <class Type> bool  cBenchNetwork<Type>::WithSchur()  const {return mWithSchur;}
template <class Type> int&  cBenchNetwork<Type>::Num() {return mNum;}

template <class Type> Type cBenchNetwork<Type>::OneItereCompensation()
{
     Type aWeightFix=100.0;

     Type  aSomEc = 0;
     Type  aNbEc = 0;
     //  Compute dist to sol + add constraint for fixed var
     for (const auto & aPN : mVPts)
     {
        if (! aPN.mTmpUk)
        {
            aNbEc++;
            aSomEc += Norm2(aPN.PCur() -aPN.PTh());
        }
	// Fix X and Y for given points
	if (aPN.mFrozenX)
           mSys->AddEqFixVar(aPN.mNumX,aPN.PTh().x(),aWeightFix);
	if (aPN.mFrozen)
           mSys->AddEqFixVar(aPN.mNumY,aPN.PTh().y(),aWeightFix);
     }
     
     //  Add observation on distances

     for (const auto & aPN1 : mVPts)
     {
         if (aPN1.mTmpUk)
	 {
            cSetIORSNL_SameTmp<Type> aSetIO;
	    cPtxd<Type,2> aP1= aPN1.PCur();
            for (const auto & aI2 : aPN1.mLinked)
            {
                const tPNet & aPN2 = mVPts.at(aI2);
	        std::vector<int> aVInd{aPN2.mNumX,aPN2.mNumY};  // Compute index of unknowns
                std::vector<Type> aVTmp{aP1.x(),aP1.y()};  // compute observations
                std::vector<Type> aVObs{Norm2(aPN1.PTh()-aPN2.PTh())};  // compute observations
		mSys->AddEq2Subst(aSetIO,mCalcD,aVInd,aVTmp,aVObs);
	    }
	    //  StdOut()  << "Id: " << aPN1.mPosTh << " NL:" << aPN1.mLinked.size() << "\n";
	    mSys->AddObsWithTmpUK(aSetIO);
	 }
	 else
	 {
               for (const auto & aI2 : aPN1.mLinked)
               {
                    const tPNet & aPN2 = mVPts.at(aI2);
	            std::vector<int> aVInd{aPN1.mNumX,aPN1.mNumY,aPN2.mNumX,aPN2.mNumY};  // Compute index of unknowns
                    std::vector<Type> aVObs{Norm2(aPN1.PTh()-aPN2.PTh())};  // compute observations

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

template <class Type> cPNetwork<Type>::cPNetwork(const cPt2di & aPTh,cBenchNetwork<Type> & aNet) :
     mPosTh    (aPTh),
     mNetW     (&aNet),
     mFrozen   (mPosTh==cPt2di(0,0)),  // fix origin
     mFrozenX  (mFrozen|| (mPosTh==cPt2di(0,1))),  //fix orientation
     mTmpUk    (aNet.WithSchur() && (mPosTh.x()==1)),  // If test schur complement, Line x=1 will be temporary
     mNumX     (-1),
     mNumY     (-1)
{
     double aAmplP = 0.1;
     // Pertubate position with global movtmt + random mvmt
     mPosInit.x() =  mPosTh.x() + aAmplP*(-mPosTh.x() +  mPosTh.y()/2.0 +std::abs(mPosTh.y()) +2*RandUnif_C());;
     mPosInit.y() =  mPosTh.y() + aAmplP*(mPosTh.y() + 4*Square(mPosTh.x()/Type(aNet.N())) + RandUnif_C() *0.2);

     if (mFrozen)
       mPosInit.y() = mPosTh.y();
     if (mFrozenX)
       mPosInit.x() = mPosTh.x();

     if (!mTmpUk)
     {
        mNumX = aNet.Num()++;
        mNumY = aNet.Num()++;
     }
}
template <class Type> cPtxd<Type,2>  cPNetwork<Type>::PCur() const
{
	// For standard unknown, read the cur solution of the system
    if (!mTmpUk)
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
	   if (mPosTh.y() == aPN2.mPosTh.y())
	   {
               aSomP +=  aPN2.PCur();
               aNbPts++;
           }
    }
    MMVII_INTERNAL_ASSERT_bench((aNbPts==2),"Bad hypothesis for network");

    return aSomP / (Type) aNbPts;
}

template <class Type> cPtxd<Type,2>  cPNetwork<Type>::PTh() const
{
	return cPtxd<Type,2>(mPosTh.x(),mPosTh.y());
}

template <class Type> bool cPNetwork<Type>::Linked(const cPNetwork<Type> & aP2) const
{
   // Precaution, a poinnt is not linked yo itself
   if (mPosTh== aP2.mPosTh) 
      return false;

   //  Normal case, no temp unknown, link point regarding the 8-connexion
   if ((!mTmpUk) && (!aP2.mTmpUk))
      return NormInf(mPosTh-aP2.mPosTh) <=1;

   //  If two temporay point, they are not observable
   if (mTmpUk && aP2.mTmpUk)
      return false;
   
   // when conecting temporary to rest of network : reinforce the connexion
   return    (std::abs(mPosTh.x()-aP2.mPosTh.x()) <=1)
          && (std::abs(mPosTh.y()-aP2.mPosTh.y()) <=2) ;
}

template class cPNetwork<tREAL8>;
template class cBenchNetwork<tREAL8>;

/* ======================================== */
/*                                          */
/*              ::                          */
/*                                          */
/* ======================================== */

void  OneBenchSSRNL(eModeSSR aMode,int aNb,bool WithSchurr)
{
     cBenchNetwork<tREAL8> aBN(aMode,aNb,WithSchurr);
     double anEc =100;
     for (int aK=0 ; aK < 8 ; aK++)
     {
         anEc = aBN.OneItereCompensation();
     }
     MMVII_INTERNAL_ASSERT_bench(anEc<1e-5,"Error in Network-SSRNL Bench");
}


};

using namespace NB_Bench_RSNL;

void BenchSSRNL(cParamExeBench & aParam)
{
     if (! aParam.NewBench("SSRNL")) return;


     OneBenchSSRNL(eModeSSR::eSSR_LsqSparseGC,10,true);
     OneBenchSSRNL(eModeSSR::eSSR_LsqDense ,10,true);

     OneBenchSSRNL(eModeSSR::eSSR_LsqSparseGC,10,false);
     OneBenchSSRNL(eModeSSR::eSSR_LsqDense ,10,false);
     OneBenchSSRNL(eModeSSR::eSSR_LsqNormSparse,10,false);


     aParam.EndBench();
}


/* ************************************************************ */
/*                                                              */
/*                  INSTANTIATION                               */
/*                                                              */
/* ************************************************************ */

#define INSTANTIATE_RESOLSYSNL(TYPE)\
template class  cInputOutputRSNL<TYPE>;\
template class  cSetIORSNL_SameTmp<TYPE>;\
template class  cResidualWeighter<TYPE>;\
template class  cResolSysNonLinear<TYPE>;

INSTANTIATE_RESOLSYSNL(tREAL4)
INSTANTIATE_RESOLSYSNL(tREAL8)
INSTANTIATE_RESOLSYSNL(tREAL16)


};
