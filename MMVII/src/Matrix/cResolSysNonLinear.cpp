
#include "MMVII_Tpl_Images.h"

#include "MMVII_SysSurR.h"

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
     mGlobVInd  (aVInd),
     mVObs      (aVObs),
     mNbTmpUk   (0)
{

    //  Check consistency on temporary indexes
    for (const auto & anInd : aVInd)
    {
        if (cSetIORSNL_SameTmp<Type>::IsIndTmp(anInd))
        {
	    mNbTmpUk++;
        }
    }
    // MMVII_INTERNAL_ASSERT_tiny(mNbTmpUk==mVTmpUk.size(),"Size Tmp/subst in  cInputOutputRSNL");
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
	return mGlobVInd.size() ;
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

template <class Type> cSetIORSNL_SameTmp<Type>::cSetIORSNL_SameTmp
                      (
		           const tStdVect & aValTmpUk,
                           const tVectInd & aVFix,
		           const tStdVect & aValFix
                      ) :
	mOk                (false),
	mNbTmpUk           (aValTmpUk.size()),
	mValTmpUk          (aValTmpUk),
	mVarTmpIsFrozen    (mNbTmpUk,false),
	mValueFrozenVarTmp (mNbTmpUk,-283971), // random val
	mNbEq              (0),
	mSetIndTmpUk       (mNbTmpUk)
{
    MMVII_INTERNAL_ASSERT_tiny((aVFix.size()==aValFix.size()) || aValFix.empty(),"Bad size for fix var tmp");

    for (size_t aKInd=0 ; aKInd<aVFix.size() ; aKInd++)
    {
        int anIndFix = aVFix[aKInd];
	Type aVal = aValFix.empty() ? Val1TmpUk(anIndFix)  : aValFix.at(aKInd);

	// Need to fix the var that will be elimined, need to do it now because line after will change
        AddFixVarTmp(anIndFix,aVal,1.0); 
        mVarTmpIsFrozen.at(ToIndTmp(anIndFix)) = true;
        mValueFrozenVarTmp.at(ToIndTmp(anIndFix)) = aVal;
    }
}

template <class Type> size_t cSetIORSNL_SameTmp<Type>::ToIndTmp(int anInd) { return -(anInd+1); }
template <class Type> bool   cSetIORSNL_SameTmp<Type>::IsIndTmp(int anInd) { return anInd<0; }
template <class Type> size_t cSetIORSNL_SameTmp<Type>::NbTmpUk() const { return mNbTmpUk; }
template <class Type> const std::vector<Type> & cSetIORSNL_SameTmp<Type>::ValTmpUk() const { return mValTmpUk; }
template <class Type> Type  cSetIORSNL_SameTmp<Type>::Val1TmpUk(int aInd) const { return mValTmpUk.at(ToIndTmp(aInd));}



template <class Type> void cSetIORSNL_SameTmp<Type>::AddOneEq(const tIO_OneEq & anIO_In)
{
    mVEq.push_back(anIO_In);
    tIO_OneEq & anIO = mVEq.back();

    MMVII_INTERNAL_ASSERT_tiny(anIO.IsOk(),"Bad size for cInputOutputRSNL");

    // for (const auto & anInd : anIO.mGlobVInd)
    for (size_t aKInd=0 ; aKInd<anIO.mGlobVInd.size() ;aKInd++)
    {
        int anIndSigned = anIO.mGlobVInd[aKInd];
        if (IsIndTmp(anIndSigned))
	{
           size_t aIndPos = ToIndTmp(anIndSigned);
           mSetIndTmpUk.AddInd(aIndPos); // add it to the computed list of indexes
           if (mVarTmpIsFrozen.at(aIndPos))
           {
              Type aDeltaVar = mValueFrozenVarTmp.at(aIndPos) - mValTmpUk.at(aIndPos);
              for (size_t aKEq=0 ; aKEq<anIO.mVals.size() ; aKEq++)
              {
                   Type & aVDer = anIO.mDers.at(aKEq).at(aKInd);
		   anIO.mVals[aKEq]  +=  aVDer * aDeltaVar;
		   aVDer = 0;
              }
           }
	}
    }

    mNbEq += anIO.mVals.size();
    if 
    (
            (mNbEq > mNbTmpUk)  // A priori there is no use to less or equal equation, this doesnt give any constraint
	 && ( mSetIndTmpUk.NbElem()== mNbTmpUk)  // we are sure to have good index, because we cannot add oustide
    )
    {
        mOk = true; 
    }
}

template <class Type> void   cSetIORSNL_SameTmp<Type>::AddFixVarTmp (int aInd,const Type& aVal,const Type& aWeight)
{
     MMVII_INTERNAL_ASSERT_tiny
     (
	 cSetIORSNL_SameTmp<Type>::IsIndTmp(aInd),
	 "Non tempo index in AddFixVarTmp"
     );

     // tVectInd aVInd{anInd};

     cInputOutputRSNL<Type> aIO({aInd},{});
     aIO.mWeights.push_back(aWeight);
     aIO.mDers.push_back({1.0});
     Type aDVal = Val1TmpUk(aInd)-aVal;
     aIO.mVals.push_back({aDVal});

     AddOneEq(aIO);
}

template <class Type> void   cSetIORSNL_SameTmp<Type>::AddFixCurVarTmp (int aInd,const Type& aWeight)
{
     AddFixVarTmp(aInd,Val1TmpUk(aInd),aWeight); 
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

      // =====    constructors / destructors ================

template <class Type> cResolSysNonLinear<Type>::cResolSysNonLinear(tLinearSysSR * aSys,const tDVect & aInitSol) :
    mNbVar          (aInitSol.Sz()),
    mCurGlobSol     (aInitSol.Dup()),
    mSysLinear      (aSys),
    mInPhaseAddEq   (false),
    mVarIsFrozen    (mNbVar,false),
    mValueFrozenVar (mNbVar,-1)
{
}

template <class Type> cResolSysNonLinear<Type>::cResolSysNonLinear(eModeSSR aMode,const tDVect & aInitSol) :
    cResolSysNonLinear<Type>  (cLinearOverCstrSys<Type>::AllocSSR(aMode,aInitSol.Sz()),aInitSol)
{
}

template <class Type> cResolSysNonLinear<Type>::~cResolSysNonLinear()
{
    delete mSysLinear;
}

      // =============  miscelaneous accessors    ================
     
template <class Type> cLinearOverCstrSys<Type> * cResolSysNonLinear<Type>::SysLinear() 
{
    mInPhaseAddEq = true;  // cautious, if user requires this access he may modify
    return mSysLinear;
}

template <class Type> int cResolSysNonLinear<Type>::NbVar() const {return mNbVar;}

      // =====    handling of frozen vars ================

template <class Type> void cResolSysNonLinear<Type>::SetFrozenVar(int aK,const  Type & aVal)
{
    AssertNotInEquation();
    mVarIsFrozen.at(aK) = true;
    mValueFrozenVar.at(aK) = aVal;
}

template <class Type> void cResolSysNonLinear<Type>::SetFrozenVarCurVal(int aK)
{
	SetFrozenVar(aK,CurSol(aK));
}

template <class Type> void cResolSysNonLinear<Type>::SetFrozenVar(tObjWUk & anObj,const  Type & aVal)
{
	SetFrozenVar(anObj.IndOfVal(&aVal),aVal);
}
template <class Type> void cResolSysNonLinear<Type>::SetFrozenVar(tObjWUk & anObj,const  Type * anAdr,size_t aNb)
{
	for (size_t aK=0 ; aK<aNb ; aK++)
            SetFrozenVar(anObj,*(anAdr+aK));
}
template <class Type> void cResolSysNonLinear<Type>::SetFrozenVar(tObjWUk & anObj,const tStdVect &  aVect)
{
            SetFrozenVar(anObj,aVect.data(),aVect.size());
}
template <class Type> void cResolSysNonLinear<Type>::SetFrozenVar(tObjWUk & anObj,const cPtxd<Type,3> &  aPt)
{
            SetFrozenVar(anObj,aPt.PtRawData(),3);
}
template <class Type> void cResolSysNonLinear<Type>::SetFrozenVar(tObjWUk & anObj,const cPtxd<Type,2> &  aPt)
{
            SetFrozenVar(anObj,aPt.PtRawData(),2);
}

/*
           void  SetFrozenVar(tObjWUk & anObj,tStdVect &);  ///< indicate it var must be frozen /unfrozen
           void  SetFrozenVar(tObjWUk & anObj,cPtxd<Type,3> &);  ///< indicate it var must be frozen /unfrozen
           void  SetFrozenVar(tObjWUk & anObj,cPtxd<Type,2> &);  ///< indicate it var must be frozen /unfrozen
*/

template <class Type> void cResolSysNonLinear<Type>::SetUnFrozen(int aK)
{
    AssertNotInEquation();
    mVarIsFrozen.at(aK) = false;
}

template <class Type> void cResolSysNonLinear<Type>::UnfrozeAll()
{
    AssertNotInEquation();
    for (int aK=0 ; aK<mNbVar ; aK++)
        mVarIsFrozen[aK] = false;
}

template <class Type> bool cResolSysNonLinear<Type>::VarIsFrozen(int aK) const
{
     return mVarIsFrozen.at(aK);
}

template <class Type> void cResolSysNonLinear<Type>::AssertNotInEquation() const
{
    if (mInPhaseAddEq)
       MMVII_INTERNAL_ERROR("Operation forbiden while adding equations");
}

template <class Type> void   cResolSysNonLinear<Type>::AddEqFixVar(const int & aNumV,const Type & aVal,const Type& aWeight)
{
     tSVect aSV;
     aSV.AddIV(aNumV,1.0);
     // Dont forget that the linear system compute the difference with current solution ...
     mSysLinear->AddObservation(aWeight,aSV,aVal-CurSol(aNumV));
}




template <class Type> void   cResolSysNonLinear<Type>::AddEqFixCurVar(const int & aNumV,const Type& aWeight)
{
     AddEqFixVar(aNumV,mCurGlobSol(aNumV),aWeight);
}

template <class Type> void  cResolSysNonLinear<Type>::ModifyFrozenVar (tIO_RSNL& aIO)
{
    for (size_t aKVar=0 ; aKVar<aIO.mGlobVInd.size() ; aKVar++)
    {
         int aIndGlob = aIO.mGlobVInd[aKVar];
	 if ( (! cSetIORSNL_SameTmp<Type>::IsIndTmp(aIndGlob)) && (mVarIsFrozen.at(aIndGlob)))
	 {
              Type aDeltaVar = mValueFrozenVar.at(aIndGlob) - mCurGlobSol(aIndGlob);
              for (size_t aKEq=0 ; aKEq<aIO.mVals.size() ; aKEq++)
	      {
		   //  ..  Der* (X-X0) + Val  ..
                   Type & aVDer = aIO.mDers.at(aKEq).at(aKVar);
		   aIO.mVals[aKEq]  +=  aVDer * aDeltaVar;
		   aVDer = 0;
	      }
	 }
    }
}


      // =============  access to current solution    ================

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
    AssertNotInEquation();
    mCurGlobSol(aNumV) = aVal;
}


     /*     =========================    adding equations ==================================
 
       # Global :
	      CalcVal  -> *  fondamental methods, 
	                  *  generate the computation of formal derivative and put the result
	                     in  vector<cInputOutputRSNL>,  
	                  *  used in case schur an
			  *  it's a vector to allow later, a possible use in parallel

       # Case Non Schurr :
	      CalcAndAddObs ->   * calc CalcVal to compute derivative then AddObs
				
	      AddObs  ->         *  send cInputOutputRSNL this in linear system

       # Case Schurr :

              AddEq2Subst   ->   *   CalcVal to compute derivative  then 
	                             push the result in a structure cSetIORSNL_SameTmp

              AddObsWithTmpUK -> *  sens a structure cSetIORSNL_SameTmp to the linear system
	                            to make the schurr elimination

       # Remark : using schurr and paralelizing formal derivatives will require additionnal methods.
                  (with probably more sophisticated/complex protocols)

     */

template <class Type> void   cResolSysNonLinear<Type>::CalcVal
                             (
			          tCalc * aCalcVal,
				  std::vector<tIO_RSNL>& aVIO,
				  const tStdVect & aValTmpUk,
				  bool WithDer,
				  const tResidualW & aWeighter
                              )
{
      mInPhaseAddEq = true;
      MMVII_INTERNAL_ASSERT_tiny(aCalcVal->NbInBuf()==0,"Buff not empty");

      // Put input data
      for (const auto & aIO : aVIO)
      {
          tStdCalcVect aVCoord;
	  // transferate global coordinates
          //  size_t anIndTmp=0;
	  for (const auto & anInd : aIO.mGlobVInd)
          {
              if (cSetIORSNL_SameTmp<Type>::IsIndTmp(anInd))
	      {
                  aVCoord.push_back(aValTmpUk.at(cSetIORSNL_SameTmp<Type>::ToIndTmp(anInd)));
	      }
              else
              {
                 aVCoord.push_back(mCurGlobSol(anInd));
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
	   ModifyFrozenVar(aIO);
      }
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

    CalcVal(aCalcVal,aVIO,{},true,aWeigther);
    AddObs(aVIO);
}


template <class Type> void cResolSysNonLinear<Type>::AddObs ( const std::vector<tIO_RSNL>& aVIO)
{
      mInPhaseAddEq = true;
      // Parse all the linearized equation
      for (const auto & aIO : aVIO)
      {
	  // check we dont use temporary value
          MMVII_INTERNAL_ASSERT_tiny(aIO.mNbTmpUk==0,"Cannot use tmp uk w/o Schurr complement");

	  // parse all values
	  for (size_t aKVal=0 ; aKVal<aIO.mVals.size() ; aKVal++)
	  {
	      Type aW = aIO.WeightOfKthResisual(aKVal);
	      if (aW>0)
	      {
	         tSVect aSV;
		 const tStdVect & aVDer = aIO.mDers[aKVal];
	         for (size_t aKUk=0 ; aKUk<aIO.mGlobVInd.size() ; aKUk++)
                 {
                     aSV.AddIV(aIO.mGlobVInd[aKUk],aVDer[aKUk]);
	         }
		 // Note the minus sign :  F(X0+dx) = F(X0) + Gx.dx   =>   Gx.dx = -F(X0)
	         mSysLinear->AddObservation(aW,aSV,-aIO.mVals[aKVal]);
	      }
	  }

      }
}


template <class Type> void   cResolSysNonLinear<Type>::AddEq2Subst 
                             (
			          tSetIO_ST & aSetIO,tCalc * aCalc,const tVectInd & aVInd, 
			          const tStdVect& aVObs,const tResidualW & aWeighter
			     )
{
    std::vector<tIO_RSNL> aVIO(1,tIO_RSNL(aVInd,aVObs));
    CalcVal(aCalc,aVIO,aSetIO.ValTmpUk(),true,aWeighter);

    aSetIO.AddOneEq(aVIO.at(0));
}
			     
template <class Type> void cResolSysNonLinear<Type>::AddObsWithTmpUK (const tSetIO_ST & aSetIO)
{
    mSysLinear->AddObsWithTmpUK(aSetIO);
}



            //  =========    resolving ==========================

template <class Type> const cDenseVect<Type> & cResolSysNonLinear<Type>::SolveUpdateReset() 
{
    mInPhaseAddEq = false;
    // for var frozen, they are not involved in any equation, we must fix their value other way
    for (int aK=0 ; aK<mNbVar ; aK++)
        if (mVarIsFrozen[aK])
           AddEqFixVar(aK,mValueFrozenVar[aK],1.0);

    mCurGlobSol += mSysLinear->Solve();     //  mCurGlobSol += mSysLinear->SparseSolve();
    mSysLinear->Reset();

    return mCurGlobSol;
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
