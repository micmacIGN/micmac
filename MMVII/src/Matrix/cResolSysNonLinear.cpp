#include "include/MMVII_all.h"
#include "include/MMVII_Tpl_Images.h"


using namespace NS_SymbolicDerivative;
using namespace MMVII;

namespace MMVII
{

/**  class for computing weight of residuals
     size out must equal size in or be equals 1 (means all value equal)
 */
template <class Type> class cResidualWeighter
{
      public :
	  typedef std::vector<Type>  tStdVect;

	  /// Defaut return Weight, independant of input
	  virtual tStdVect  ComputeWeith(const tStdVect &) const {return mWeight;}
	  /// Constructor with constant
	  cResidualWeighter(const Type & aW=1.0) :  mWeight ({aW}) {}
      private :
	  std::vector<Type>  mWeight;
};

/**  class for communinication  input and ouptut of equations in 
*   cResolSysNonLinear
 */
template <class Type> class cInputOutputRSNL
{
     public :
	  typedef std::vector<Type>  tStdVect;
	  typedef std::vector<int>   tVectInd;

	  tVectInd   mVInd;    ///<  index of unknown in the system
	  tStdVect   mTmpUK;   ///< possible value of temporary unknown,that would be eliminated by schur complement
	  tStdVect   mObs;     ///< Observation (i.e constants)

	  tStdVect                mVals;  ///< values of fctr, i.e. residuals
	  std::vector<tStdVect>   mDers;  ///< derivate of fctr
};

template <class Type> class cResolSysNonLinear
{
      public :
          typedef NS_SymbolicDerivative::cCalculator<Type>  tCalc;
	  typedef cSysSurResolu<Type>                       tSysSR;
	  typedef cDenseVect<Type>                          tDVect;
	  typedef cSparseVect<Type>                         tSVect;
	  typedef std::vector<Type>                         tStdVect;
	  typedef std::vector<int>                          tVectInd;
	  typedef cResolSysNonLinear<Type>                  tRSNL;
          typedef cResidualWeighter<Type>                   tResW;
          typedef cInputOutputRSNL<Type>                    tIO_TSNL;

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
	  void   CalcAndAddObs(tCalc *,const tVectInd &,const tStdVect& aVObs,tResW *  aWeighter);

      private :
	  cResolSysNonLinear(const tRSNL & ) = delete;

	  /// Add observations as computed by CalcVal
	  void   AddObs(const std::vector<tIO_TSNL>&,tResW *  aWeighter);

	  /** Bases function of calculating derivatives, dont modify the system as is
	      to avoid in case  of schur complement */
	  void   CalcVal(tCalc *,std::vector<tIO_TSNL>&,bool WithDer);

	  int        mNbVar;       ///< Number of variable, facility
          tDVect     mCurGlobSol;  ///< Curent solution
          tSysSR*    mSys;         ///< Sys to solve equations, equation are concerning the differences with current solution
};

/* ************************************************************ */
/*                                                              */
/*                cResolSysNonLinear                            */
/*                                                              */
/* ************************************************************ */

template <class Type> cResolSysNonLinear<Type>::cResolSysNonLinear(eModeSSR aMode,const tDVect & aInitSol) :
    mNbVar      (aInitSol.Sz()),
    mCurGlobSol (aInitSol.Dup()),
    // mSys        (new cLeasSqtAA<Type>(mNbVar))
    mSys        (cSysSurResolu<Type>::AllocSSR(aMode,mNbVar))
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
				  tResW *  aWeighter
                            )
{
    std::vector<tIO_TSNL> aVIO(1);
    aVIO[0].mVInd = aVInd;
    aVIO[0].mObs = aVObs;

    CalcVal(aCalcVal,aVIO,true);
    AddObs(aVIO,aWeighter);
}


template <class Type> cResolSysNonLinear<Type>::~cResolSysNonLinear()
{
    delete mSys;
}


template <class Type> void cResolSysNonLinear<Type>::AddObs
                           (
                               const std::vector<tIO_TSNL>& aVIO,
			       tResW *  aWeighter
                           )
{
      // Parse all the linearized equation
      for (const auto & aIO : aVIO)
      {
          tStdVect  aVW = aWeighter->ComputeWeith(aIO.mVals);
	  // check size of weight
          MMVII_INTERNAL_ASSERT_tiny((aVW.size()==1)||(aVW.size()==aIO.mVals.size()),"Bad size for weighting");
	  // check we dont use temporary value
          MMVII_INTERNAL_ASSERT_tiny(aIO.mTmpUK.empty(),"Cannot use tmp uk w/o Schurr complement");

	  // parse all values
	  for (size_t aKVal=0 ; aKVal<aIO.mVals.size() ; aKVal++)
	  {
              size_t aKW = std::min(aKVal,aVW.size()-1);
	      Type aW=aVW.at(aKW);
	      if (aW>0)
	      {
	         tSVect aSV;
		 const tStdVect & aVDer = aIO.mDers[aKVal];
	         for (size_t aKUk=0 ; aKUk<aIO.mVInd.size() ; aKUk++)
                 {
                     aSV.AddIV(aIO.mVInd[aKUk],aVDer[aKUk]);
	         }
		 //  F(X0+dx) = F(X0) + Gx.dx   =>   Gx.dx = -F(X0)
	         mSys->AddObservation(aW,aSV,-aIO.mVals[aKVal]);
	      }
	  }

      }
}



template <class Type> void   cResolSysNonLinear<Type>::CalcVal
                             (
			          tCalc * aCalcVal,
				  std::vector<tIO_TSNL>& aVIO,
				  bool WithDer
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
           auto & aIO = aVIO[aNumPush];
	   aIO.mVals = tStdVect(aNbEl);
	   if (WithDer)
	       aIO.mDers = std::vector(aNbEl,tStdVect( aNbUk));  // initialize vector to good size
	   // parse different values of each equation
           for (size_t aKEl=0; aKEl<aNbEl  ; aKEl++)
	   {
               aIO.mVals[aKEl] = aCalcVal->ValComp(aNumPush,aKEl);
	       if (WithDer)
	       {
	            // parse  all unknowns
	            for (size_t aKUk =0 ; aKUk<aNbUk ; aKUk++)
		    {
                        aIO.mDers[aKEl][aKUk] = aCalcVal->DerComp(aNumPush,aKEl,aKUk);
		    }
               }
	   }
      }
}
/*
template <class Type> 
    void   cResolSysNonLinear<Type>::CalcVal
                        (
                             tCalc * aCalcVal,
                             const tStdVect & aVObs,
                             const tResW * aRWeighter
                        )
{
      MMVII_INTERNAL_ASSERT_strong(aCalcVal->NbInBuf()==0,"Buff not empty");
      MMVII_INTERNAL_ASSERT_strong(aCalcVal->NbUk()==mCurPts.size(),"Bad size in cResolSysNonLinear::CalcVal");

      aCalcVal->PushNewEvals(mCurPts,aVObs);
      aCalcVal->EvalAndClear();

      tStdVect   aRes;
      for (size_t aK=0; aK<aCalcVal->NbElem()  ; aK++)
          aRes.push_back(aCalcVal->ValComp(0,aK));

      if (aRWeighter)
      {
          tStdVect   aVW = aRWeighter->ComputeWeith(aRes);
          MMVII_INTERNAL_ASSERT_strong((aVW.size()==1)||(aVW.size()==aRes.size()) ,"Bad size for weight");
          for (size_t aK=0; aK<aCalcVal->NbElem()  ; aK++)
	  {
                int aKW = std::min(aK,aVW.size()-1);
		Type aW = aVW.at(aKW);
		if (aW)
		{
		}
	  }
      }

      return aRes;
}
*/


/*
           mCalcVal->PushNewEvals(aVUk,mVObs);
       }
       mCalcVal->EvalAndClear();
       for (tU_INT4 aK=aK0 ; aK<aK1 ; aK++)
       {
           tPtOut aPRes;
           for (int aD=0 ; aD<DimOut ; aD++)
           {
               aPRes[aD] = mCalcVal->ValComp(aK-aK0,aD);
           }
           aRes.push_back(aPRes);
	   */

/* ************************************************************ */
/*                                                              */
/*                  BENCH                                       */
/*                                                              */
/* ************************************************************ */

/*   To check some correctness  on cResolSysNonLinear, we will do the following stuff
     which more or less a simulation of triangulation
 
     #  create a network for which we have approximate coordinate  (except few point for 
        which they are exact) and exact mesure of distances between pair of points

     # we try to recover the coordinates using compensation on distances


     The network is made of [-N,N] x [-N,N]
 
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

	    cPtxd<Type,2>  PCur() const;
	    cPtxd<Type,2>  PTh() const;

	    bool Linked(const cPNetwork<Type> & aP2) const;

            cPt2di         mPosTh;
	    const tNetW *  mNetW;
            cPtxd<Type,2>  mPosInit;
	    bool           mFrozen;
	    bool           mFrozenX;
	    bool           mTmpUk;
	    int            mNumX;
	    int            mNumY;
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

	private :
	  int   mN;
	  bool  mWithSchur;
	  int   mNum;
	  std::vector<tPNet>  mVPts;
	  std::list<cPt2di>   mListCple;
	  tSys *              mSys;
	  tCalc *             mCalcD;
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

     for (const auto & aICpl : mListCple)
     {
          const tPNet & aPN1 = mVPts.at(aICpl.x());
          const tPNet & aPN2 = mVPts.at(aICpl.y());
	  // Basic case, no schur compl, just add obse
	  if ((!aPN1.mTmpUk) && (!aPN2.mTmpUk))
	  {
	      std::vector<int> aVInd{aPN1.mNumX,aPN1.mNumY,aPN2.mNumX,aPN2.mNumY};  // Compute index of unknowns
              std::vector<Type> aVObs{Norm2(aPN1.PTh()-aPN2.PTh())};  // compute observations
	      cResidualWeighter<Type> aWeighter;  // basic weighter 

	      mSys->CalcAndAddObs(mCalcD,aVInd,aVObs,&aWeighter);
	      //  StdOut() << "DDDD " << aDistObs  << aPN1.PTh() << aPN2.PTh() << "\n";
	      // void   CalcAndAddObs(tCalc *,const tVectInd &,const tStdVect& aVObs,tResW *  aWeighter);
	  }
	  else
	  {
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
	return cPtxd<Type,2>(mNetW->CurSol(mNumX),mNetW->CurSol(mNumY));
}

template <class Type> cPtxd<Type,2>  cPNetwork<Type>::PTh() const
{
	return cPtxd<Type,2>(mPosTh.x(),mPosTh.y());
}

template <class Type> bool cPNetwork<Type>::Linked(const cPNetwork<Type> & aP2) const
{
   if (mPosTh== aP2.mPosTh) 
      return false;

   if ((!mTmpUk) && (!aP2.mTmpUk))
      return NormInf(mPosTh-aP2.mPosTh) <=1;

   if (mTmpUk && aP2.mTmpUk)
      return false;
   
   return    (std::abs(mPosTh.x()-aP2.mPosTh.x()) <=1)
          && (std::abs(mPosTh.y()-aP2.mPosTh.y()) <=2) ;
}

template class cPNetwork<tREAL8>;
template class cBenchNetwork<tREAL8>;

void  OneBenchSSRNL(eModeSSR aMode,int aNb)
{
     cBenchNetwork<tREAL8> aBN(aMode,aNb,false);
     double anEc =100;
     for (int aK=0 ; aK < 8 ; aK++)
     {
         anEc = aBN.OneItereCompensation();
	 //StdOut() << "EEEE=" << anEc << "\n";
     }
     StdOut() << "EEEE=" << anEc << "\n";
     MMVII_INTERNAL_ASSERT_bench(anEc<1e-5,"Error in Network-SSRNL Bench");
}


};

using namespace NB_Bench_RSNL;

void BenchSSRNL(cParamExeBench & aParam)
{
     if (! aParam.NewBench("SSRNL")) return;


     // OneBenchSSRNL(4);
     OneBenchSSRNL(eModeSSR::eSSR_LsqDense ,10);
     OneBenchSSRNL(eModeSSR::eSSR_LsqSparse,10);

     aParam.EndBench();
}


/* ************************************************************ */
/*                                                              */
/*                  INSTANTIATION                               */
/*                                                              */
/* ************************************************************ */

#define INSTANTIATE_RESOLSYSNL(TYPE)\
template class  cResidualWeighter<TYPE>;\
template class  cInputOutputRSNL<TYPE>;\
template class  cResolSysNonLinear<TYPE>;

// INSTANTIATE_RESOLSYSNL(tREAL4)
INSTANTIATE_RESOLSYSNL(tREAL8)
// INSTANTIATE_RESOLSYSNL(tREAL16)


};
