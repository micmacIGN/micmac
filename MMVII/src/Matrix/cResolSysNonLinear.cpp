#include "include/MMVII_all.h"
#include "include/MMVII_Tpl_Images.h"


using namespace NS_SymbolicDerivative;
using namespace MMVII;

namespace MMVII
{

/* ************************************************************ */
/*                                                              */
/*                cInputOutputRSNL                              */
/*                                                              */
/* ************************************************************ */

template <class Type>  cInputOutputRSNL<Type>::cInputOutputRSNL(const tVectInd& aVInd,const tStdVect & aVObs):
     cInputOutputRSNL<Type>(aVInd,{},aVObs)
{
}

template <class Type>  cInputOutputRSNL<Type>::cInputOutputRSNL(const tVectInd& aVInd,const tStdVect & aVTmp,const tStdVect & aVObs):
     mVTmpUK    (aVTmp),
     mVIndGlob  (aVInd),
     mVObs      (aVObs)
{

    //  Check consistency on temporary indexes
    int aNbInd2Subst=0;
    for (const auto & anInd : aVInd)
    {
        if (anInd<0) 
        {
            MMVII_INTERNAL_ASSERT_tiny(anInd==RSL_INDEX_SUBST_TMP,"IndTmp bas val");
            aNbInd2Subst ++;
        }
        else
        {
           mVIndUk.push_back(anInd);
        }
    }
    MMVII_INTERNAL_ASSERT_tiny(aNbInd2Subst==(int)mVTmpUK.size(),"Size Tmp/subst in  cInputOutputRSNL");
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
	return mVIndGlob.size() ;
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
    // there would be no meaning to have variable size of tmp
    if (!mVEq.empty())
    {
         MMVII_INTERNAL_ASSERT_tiny
         (
             (anIO.mVTmpUK.size()==mVEq.back().mVTmpUK.size()),
	     "Variable size of temporaries"
         );
    }
    MMVII_INTERNAL_ASSERT_tiny(anIO.IsOk(),"Bad size for cInputOutputRSNL");

    mVEq.push_back(anIO);
    mNbEq += anIO.mVals.size();
    // A priori there is no use to less or equal equation, this doesnt give any constraint
    if (mNbEq > anIO.mVTmpUK.size())
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
    return mVEq.at(0).mVTmpUK.size();
}

/* ************************************************************ */
/*                                                              */
/*                cResidualWeighter                             */
/*                                                              */
/* ************************************************************ */

template <class Type>  cResidualWeighter<Type>::cResidualWeighter(const Type & aVal) :
    mVal (aVal)
{
}

template <class Type>  std::vector<Type>  cResidualWeighter<Type>::WeightOfResidual(const tStdVect & aVResidual) const
{
	return tStdVect(aVResidual.size(),mVal);
}


/* ************************************************************ */
/*                                                              */
/*                cResolSysNonLinear                            */
/*                                                              */
/* ************************************************************ */

template <class Type> cResolSysNonLinear<Type>::cResolSysNonLinear(tLinearSysSR * aSys,const tDVect & aInitSol) :
    mNbVar      (aInitSol.Sz()),
    mCurGlobSol (aInitSol.Dup()),
    mSysLinear        (aSys)
{
}

template <class Type> cResolSysNonLinear<Type>::cResolSysNonLinear(eModeSSR aMode,const tDVect & aInitSol) :
    cResolSysNonLinear<Type>  (cLinearOverCstrSys<Type>::AllocSSR(aMode,aInitSol.Sz()),aInitSol)
{
}

template <class Type> void   cResolSysNonLinear<Type>::AddEqFixVar(const int & aNumV,const Type & aVal,const Type& aWeight)
{
// StdOut() << "VAEFC " << aNumV << " " << CurSol().
     tSVect aSV;
     aSV.AddIV(aNumV,1.0);
     // Dont forget that the linear system compute the difference with current solution ...
     mSysLinear->AddObservation(aWeight,aSV,aVal-CurSol(aNumV));
     // mSys->AddObservation(aWeight,aSV,CurSol(aNumV)+aVal);
}


template <class Type> const cDenseVect<Type> & cResolSysNonLinear<Type>::CurGlobSol() const 
{
    return mCurGlobSol;
}
template <class Type> const Type & cResolSysNonLinear<Type>::CurSol(int aNumV) const
{
    return mCurGlobSol(aNumV);
}
template <class Type> void cResolSysNonLinear<Type>::SetCurSol(int aNumV,const Type & aVal) 
{
    mCurGlobSol(aNumV) = aVal;
}

template <class Type> const cDenseVect<Type> & cResolSysNonLinear<Type>::SolveUpdateReset() 
{
    mCurGlobSol += mSysLinear->Solve();
    //  mCurGlobSol += mSysLinear->SparseSolve();
    mSysLinear->Reset();

    return mCurGlobSol;
}

template <class Type> cLinearOverCstrSys<Type> * cResolSysNonLinear<Type>::SysLinear() 
{
    return mSysLinear;
}

template <class Type> int cResolSysNonLinear<Type>::NbVar() const {return mNbVar;}


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
    std::vector<tIO_RSNL> aVIO(1,tIO_RSNL(aVInd,aVObs));

    CalcVal(aCalcVal,aVIO,true,aWeigther);
    AddObs(aVIO);
}


template <class Type> cResolSysNonLinear<Type>::~cResolSysNonLinear()
{
    delete mSysLinear;
}


template <class Type> void cResolSysNonLinear<Type>::AddObs ( const std::vector<tIO_RSNL>& aVIO)
{
      // Parse all the linearized equation
      for (const auto & aIO : aVIO)
      {
	  // check we dont use temporary value
          MMVII_INTERNAL_ASSERT_tiny(aIO.mVTmpUK.empty(),"Cannot use tmp uk w/o Schurr complement");

	  // parse all values
	  for (size_t aKVal=0 ; aKVal<aIO.mVals.size() ; aKVal++)
	  {
	      Type aW = aIO.WeightOfKthResisual(aKVal);
	      if (aW>0)
	      {
	         tSVect aSV;
		 const tStdVect & aVDer = aIO.mDers[aKVal];
	         for (size_t aKUk=0 ; aKUk<aIO.mVIndUk.size() ; aKUk++)
                 {
                     aSV.AddIV(aIO.mVIndUk[aKUk],aVDer[aKUk]);
	         }
		 // Note the minus sign :  F(X0+dx) = F(X0) + Gx.dx   =>   Gx.dx = -F(X0)
	         mSysLinear->AddObservation(aW,aSV,-aIO.mVals[aKVal]);
	      }
	  }

      }
}


template <class Type> void   cResolSysNonLinear<Type>::AddEq2Subst 
                             (
			          tSetIO_ST & aSetIO,tCalc * aCalc,const tVectInd & aVInd,const tStdVect& aVTmp,
			          const tStdVect& aVObs,const tResidualW & aWeighter
			     )
{
    std::vector<tIO_RSNL> aVIO(1,tIO_RSNL(aVInd,aVTmp,aVObs));
    CalcVal(aCalc,aVIO,true,aWeighter);

    aSetIO.AddOneEq(aVIO.at(0));
}
			     
template <class Type> void cResolSysNonLinear<Type>::AddObsWithTmpUK (const tSetIO_ST & aSetIO)
{
    mSysLinear->AddObsWithTmpUK(aSetIO);
}

template <class Type> void   cResolSysNonLinear<Type>::CalcVal
                             (
			          tCalc * aCalcVal,
				  std::vector<tIO_RSNL>& aVIO,
				  bool WithDer,
				  const tResidualW & aWeighter
                              )
{
      MMVII_INTERNAL_ASSERT_tiny(aCalcVal->NbInBuf()==0,"Buff not empty");

      // Put input data
      for (const auto & aIO : aVIO)
      {
          tStdCalcVect aVCoord;
	  // transferate global coordinates
          size_t anIndTmp=0;
	  for (const auto & anInd : aIO.mVIndGlob)
          {
              if (anInd >=0)
                 aVCoord.push_back(mCurGlobSol(anInd));
              else
              {
                  aVCoord.push_back(aIO.mVTmpUK.at(anIndTmp++));
              }
          }
          // Make a type converstion to calc type
          tStdCalcVect aVObs;
          for (const auto & aObs : aIO.mVObs)
              aVObs.push_back(aObs);
	  // transferate potential temporary coordinates
	  //  Add equation in buffer
          aCalcVal->PushNewEvals(aVCoord,aVObs);
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
#if (0)

#endif


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
