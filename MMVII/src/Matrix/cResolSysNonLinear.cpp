#include "MMVII_Tpl_Images.h"
#include "MMVII_SysSurR.h"
#include "LinearConstraint.h"


using namespace NS_SymbolicDerivative;
using namespace MMVII;

namespace MMVII
{

/* ************************************************************ */
/*                                                              */
/*                cREAL8_RSNL                                   */
/*                                                              */
/* ************************************************************ */

cREAL8_RSNL::~cREAL8_RSNL()
{
}

cREAL8_RSNL::cREAL8_RSNL(int aNbVar) :
    mNbVar          (aNbVar),
    mInPhaseAddEq   (false),
    mVarIsFrozen    (mNbVar,false),
    mNbIter         (0),
    mCurMaxEquiv    (0),
    mEquivNum       (aNbVar,TheLabelNoEquiv)
{
}

void cREAL8_RSNL::SetUnFrozen(int aK)
{
    AssertNotInEquation();
    mVarIsFrozen.at(aK) = false;
}
void cREAL8_RSNL::UnfrozeAll()
{
    AssertNotInEquation();
    for (int aK=0 ; aK<mNbVar ; aK++)
        mVarIsFrozen[aK] = false;
}

bool cREAL8_RSNL::VarIsFrozen(int aK) const
{
     return mVarIsFrozen.at(aK);
}

void cREAL8_RSNL::AssertNotInEquation() const
{
    if (mInPhaseAddEq)
       MMVII_INTERNAL_ERROR("Operation forbiden while adding equations");
}

int cREAL8_RSNL::CountFreeVariables() const
{
     return std::count(mVarIsFrozen.begin(), mVarIsFrozen.end(), false);
}

void cREAL8_RSNL::SetPhaseEq()
{
    if (mInPhaseAddEq) return;

    InitConstraint();
    mInPhaseAddEq = true;
}

void cREAL8_RSNL::SetShared(const std::vector<int> &  aVUk)
{
// StdOut() << "cREAL8_RSNL::SetShared " << mEquivNum.size() << " " << 
     for (const auto & aIUK : aVUk)
        mEquivNum.at(aIUK) = mCurMaxEquiv;
     mCurMaxEquiv++;
}

void   cREAL8_RSNL::SetUnShared(const std::vector<int>  & aVUk)
{
     for (const auto & aIUK : aVUk)
        mEquivNum.at(aIUK) = TheLabelNoEquiv;
}

void cREAL8_RSNL::SetAllUnShared()
{
     for (auto & anEq : mEquivNum)
         anEq = TheLabelNoEquiv;
     mCurMaxEquiv = 0;
}


/* ************************************************************ */
/*                                                              */
/*                cResolSysNonLinear                            */
/*                                                              */
/* ************************************************************ */

template <class Type>  void  cResolSysNonLinear<Type>::InitConstraint()
{
    mLinearConstr->Reset();
    //  Add the constraint specific to Frozen-Var
    for (int aKV=0 ; aKV<mNbVar ; aKV++)
    {
        if (mVarIsFrozen.at(aKV))
	{
           mLinearConstr->Add1ConstrFrozenVar(aKV,mValueFrozenVar.at(aKV),&mCurGlobSol);
	}
    }

    // Add the constraint specific to shared unknowns
    {
         std::map<int,std::vector<int>> aMapEq;
         for (int aKV=0 ; aKV<mNbVar ; aKV++)
         {
              if (mEquivNum.at(aKV)>=0)
                 aMapEq[mEquivNum.at(aKV)].push_back(aKV);
         }
         // For X1,X2, ..., Xk shared, we add the constraint X1=X2, X1=X3, ... X1=Xk
         // And fix  the value to average
         for (const auto & [anEqui,aVInd] : aMapEq)
         {
             Type aSumV = 0 ;
             for (size_t aKInd=0 ; aKInd<aVInd.size() ; aKInd++)
             {
                 aSumV += mCurGlobSol(aVInd.at(aKInd));
                 if (aKInd)
                 {
                     cSparseVect<Type>  aLinC;
                     aLinC.AddIV(aVInd.at(0),1.0);
                     aLinC.AddIV(aVInd.at(aKInd),-1.0);
                     mLinearConstr->Add1Constr(aLinC,0.0,&mCurGlobSol);
                 }
             }
             // setting to the average is "better" at the first iteration,  after it's useless, but no harm ...
             aSumV /= aVInd.size();
/*   DONT UNDERSTAND WHY  !!!! But this does not work 
             for (size_t aKInd=0 ; aKInd<aVInd.size() ; aKInd++)
             {
                 mCurGlobSol(aVInd.at(aKInd)) = aSumV;
             }
*/
         }
    }

    // Add the general constraint 
    for (size_t aKC=0 ; aKC<mVCstrCstePart.size() ; aKC++)
    {
        mLinearConstr->Add1Constr(mVCstrLinearPart.at(aKC),mVCstrCstePart.at(aKC),&mCurGlobSol);
    }
    mLinearConstr->Compile(false);
}

template <class Type>  void   cResolSysNonLinear<Type>::AddConstr(const tSVect & aVect,const Type & aCste,bool OnlyIfFirstIter)
{
    if (OnlyIfFirstIter && (mNbIter!=0)) return;

    mVCstrLinearPart.push_back(aVect.Dup());
    mVCstrCstePart.push_back(aCste);
}



// template <class Type>  void  cResolSysNonLinear<Type>::I

      // =====    constructors / destructors ================

template <class Type> cResolSysNonLinear<Type>::cResolSysNonLinear(tLinearSysSR * aSys,const tDVect & aInitSol) :
    cREAL8_RSNL     (aInitSol.Sz()),
    // mNbVar          (aInitSol.Sz()),
    mCurGlobSol     (aInitSol.Dup()),
    mSysLinear      (aSys),
    // mVarIsFrozen    (mNbVar,false),
    mValueFrozenVar (mNbVar,-1),
    lastNbObs       (0),
    currNbObs       (0),
    mLinearConstr   (new cSetLinearConstraint<Type>(mNbVar))
{
}

template <class Type> cResolSysNonLinear<Type>::cResolSysNonLinear(eModeSSR aMode,const tDVect & aInitSol) :
    cResolSysNonLinear<Type>  (cLinearOverCstrSys<Type>::AllocSSR(aMode,aInitSol.Sz()),aInitSol)
{
}

template <class Type> cResolSysNonLinear<Type>::~cResolSysNonLinear()
{
    delete mSysLinear;
    delete mLinearConstr;
}

      // =============  miscelaneous accessors    ================
     
template <class Type> cLinearOverCstrSys<Type> * cResolSysNonLinear<Type>::SysLinear() 
{
    SetPhaseEq(); // cautious, if user requires this access he may modify
    return mSysLinear;
}

template <class Type> int cResolSysNonLinear<Type>::NbVar() const {return mNbVar;}
template <class Type> int cResolSysNonLinear<Type>::R_NbVar() const {return NbVar();}

      // =====    handling of frozen vars ================

template <class Type> void cResolSysNonLinear<Type>::SetFrozenVar(int aK,const  Type & aVal)
{
    AssertNotInEquation();
    mVarIsFrozen.at(aK) = true;
    mValueFrozenVar.at(aK) = aVal;
}
template <class Type> void cResolSysNonLinear<Type>::R_SetFrozenVar(int aK,const  tREAL8 & aVal)
{
	SetFrozenVar(aK,aVal);
}



template <class Type> void cResolSysNonLinear<Type>::SetFrozenVarCurVal(int aK)
{
	SetFrozenVar(aK,CurSol(aK));
}

template <class Type> void cResolSysNonLinear<Type>::SetFrozenVarCurVal(tObjWUk & anObj,const  Type & aVal)
{
	// SetFrozenVar(anObj.IndOfVal(&aVal),aVal);
	SetFrozenVarCurVal(anObj.IndOfVal(&aVal));
}


template <class Type> void cResolSysNonLinear<Type>::SetFrozenVarCurVal(tObjWUk & anObj,const  Type * anAdr,size_t aNb)
{
	for (size_t aK=0 ; aK<aNb ; aK++)
            SetFrozenVarCurVal(anObj,*(anAdr+aK));
}
template <class Type> void cResolSysNonLinear<Type>::SetFrozenVarCurVal(tObjWUk & anObj,const tStdVect &  aVect)
{
            SetFrozenVarCurVal(anObj,aVect.data(),aVect.size());
}
template <class Type> void cResolSysNonLinear<Type>::SetFrozenVarCurVal(tObjWUk & anObj,const cPtxd<Type,3> &  aPt)
{
            SetFrozenVarCurVal(anObj,aPt.PtRawData(),3);
}
template <class Type> void cResolSysNonLinear<Type>::SetFrozenVarCurVal(tObjWUk & anObj,const cPtxd<Type,2> &  aPt)
{
            SetFrozenVarCurVal(anObj,aPt.PtRawData(),2);
}

template <class Type> void  cResolSysNonLinear<Type>::SetFrozenAllCurrentValues(tObjWUk & anObj)
{
     for (int aK=anObj.IndUk0() ; aK<anObj.IndUk1() ; aK++)
         SetFrozenVarCurVal(aK);
}

template <class Type> 
    void  cResolSysNonLinear<Type>::SetFrozenFromPat(tObjWUk & anObjGlob,const std::string& aPat, bool Frozen)
{
      cGetAdrInfoParam<Type> aGIAP(aPat,anObjGlob);
      for (size_t aK=0 ;aK<aGIAP.VAdrs().size() ; aK++)
      {
          Type * anAdr =aGIAP.VAdrs()[aK];
	  tObjWUk * anObjPtr  = aGIAP.VObjs()[aK];
	  //  StdOut() << "Aaaa " << *anAdr << " NN=" << aGIAP.VNames() [aK] << " " << anObjPtr->IndOfVal(anAdr) << std::endl;
          if (Frozen)
	  {
             SetFrozenVarCurVal(*anObjPtr,*anAdr);
	  }
	  else
	  {
             SetUnFrozenVar(*anObjPtr,*anAdr);
	  }
      }
}


template <class Type> void cResolSysNonLinear<Type>::SetUnFrozenVar(tObjWUk & anObj,const  Type & aVal)
{
       SetUnFrozen(anObj.IndOfVal(&aVal));
}




template <class Type> void   cResolSysNonLinear<Type>::AddEqFixVar(const int & aNumV,const Type & aVal,const Type& aWeight)
{
     tSVect aSV;
     aSV.AddIV(aNumV,1.0);
     // Dont forget that the linear system compute the difference with current solution ...
     mSysLinear->PublicAddObservation(aWeight,aSV,aVal-CurSol(aNumV));
}

template <class Type> void   cResolSysNonLinear<Type>::R_AddEqFixVar(const int & aNumV,const tREAL8 & aVal,const tREAL8& aWeight)
{
	AddEqFixVar(aNumV,aVal,aWeight);
}




template <class Type> int  cResolSysNonLinear<Type>::GetNbObs() const
{
    return currNbObs?currNbObs:lastNbObs;
}

//   ==================================  Fix var with a given weight =====================================

template <class Type> void   cResolSysNonLinear<Type>::AddEqFixCurVar(const int & aNumV,const Type& aWeight)
{
     AddEqFixVar(aNumV,mCurGlobSol(aNumV),aWeight);
}

template <class Type> void   cResolSysNonLinear<Type>::R_AddEqFixCurVar(const int & aNumV,const tREAL8& aWeight)
{
	AddEqFixCurVar(aNumV,aWeight);
}

template <class Type> void   cResolSysNonLinear<Type>::AddEqFixCurVar(const tObjWUk & anObj,const  Type & aVal,const Type& aWeight)
{
     size_t aNumV = anObj.IndOfVal(&aVal);
     AddEqFixVar(aNumV,mCurGlobSol(aNumV),aWeight);
}
template <class Type> void   cResolSysNonLinear<Type>::AddEqFixNewVal(const tObjWUk & anObj,const  Type & aVal,const Type & aNewVal,const Type& aWeight)
{
     size_t aNumV = anObj.IndOfVal(&aVal);
     AddEqFixVar(aNumV,aNewVal,aWeight);
}


template <class Type> void   cResolSysNonLinear<Type>::AddEqFixCurVar(const tObjWUk & anObj,const  Type * aVal,size_t aNb,const Type& aWeight)
{
     for (size_t aK=0 ; aK<aNb ; aK++)
         AddEqFixCurVar(anObj,*(aVal+aK),aWeight);
}

template <class Type> void   cResolSysNonLinear<Type>::AddEqFixNewVal(const tObjWUk & anObj,const  Type * aVal,const  Type * aNewVal,size_t aNb,const Type& aWeight)
{
     for (size_t aK=0 ; aK<aNb ; aK++)
         AddEqFixNewVal(anObj,*(aVal+aK),*(aNewVal+aK),aWeight);
}



template <class Type> void   cResolSysNonLinear<Type>::AddEqFixCurVar(const tObjWUk & anObj,const  cPtxd<Type,3> & aPt,const Type& aWeight)
{
     AddEqFixCurVar(anObj,aPt.PtRawData(),3,aWeight);
}


template <class Type> void   cResolSysNonLinear<Type>::AddEqFixNewVal(const tObjWUk & anObj,const  cPtxd<Type,3> & aPt,const  cPtxd<Type,3> & aNewPt,const Type& aWeight)
{
     AddEqFixNewVal(anObj,aPt.PtRawData(),aNewPt.PtRawData(),3,aWeight);
}


template <class Type> void  cResolSysNonLinear<Type>::ModifyFrozenVar (tIO_RSNL& aIO)
{
          // CHANGE HERE
#if (WithNewLinearCstr)
    mLinearConstr->SubstituteInOutRSNL(aIO);
#else
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
#endif
}

template <class Type> void  cResolSysNonLinear<Type>::AddObservationLinear
                            (
                                 const Type& aWeight,
                                 const cSparseVect<Type> & aCoeff,
                                 const Type &  aRHS
                            )
{
     SetPhaseEq(); 
     Type  aNewRHS    = aRHS;
     cSparseVect<Type> aNewCoeff;

#if (WithNewLinearCstr)
      for (const auto & aPair :aCoeff)
      {
          aNewRHS -=  mCurGlobSol(aPair.mInd) * aPair.mVal;
          aNewCoeff.AddIV(aPair);
      }
      mLinearConstr->SubstituteInSparseLinearEquation(aNewCoeff,aNewRHS);
#else
     for (const auto & aPair :aCoeff)
     {
          // CHANGE HERE
          if (mVarIsFrozen.at(aPair.mInd))
          {
              // if freeze => transfert value in contant
              aNewRHS -= mValueFrozenVar.at(aPair.mInd) *  aPair.mVal;
          }
          else
          {
              // else tranfert current value as we are computing the difference to it
              aNewRHS -=  mCurGlobSol(aPair.mInd) * aPair.mVal;
              aNewCoeff.AddIV(aPair);
          }
     }
#endif
     currNbObs++;  ///  Check JMM
     mSysLinear->PublicAddObservation(aWeight,aNewCoeff,aNewRHS);
}

     
template <class Type> void  cResolSysNonLinear<Type>::AddObservationLinear
                            (
                                 const Type& aWeight,
                                 const cDenseVect<Type> & aCoeff,
                                 const Type &  aRHS
                            )
{
     SetPhaseEq(); 
     Type  aNewRHS    = aRHS;
     cDenseVect<Type> aNewCoeff = aCoeff.Dup();
#if (WithNewLinearCstr)
     for (int aK=0 ; aK<mNbVar ; aK++)
         aNewRHS -=  mCurGlobSol(aK) * aCoeff(aK);  // -A' X0'
      mLinearConstr->SubstituteInDenseLinearEquation(aNewCoeff,aNewRHS);
#else

     //   AX-B =  (A' X' + AiXi-B) = A' (X'-X0')  + A' X0' +AiXi -B
     //   B=> B -AiXi -A' X0'
     for (int aK=0 ; aK<mNbVar ; aK++)
     {
          // CHANGE HERE
          if (mVarIsFrozen.at(aK))
          {
              aNewRHS -= mValueFrozenVar.at(aK) * aCoeff(aK); //  -AiXi
              aNewCoeff(aK)=0;
          }
          else
          {
              aNewRHS -=  mCurGlobSol(aK) * aCoeff(aK);  // -A' X0'
          }
     }
#endif
     currNbObs++;  ///  Check JMM
     mSysLinear->PublicAddObservation(aWeight,aNewCoeff,aNewRHS);
}





      // =============  access to current solution    ================

template <class Type> const cDenseVect<Type> & cResolSysNonLinear<Type>::CurGlobSol() const 
{
    return mCurGlobSol;
}
template <class Type> cDenseVect<tREAL8>  cResolSysNonLinear<Type>::R_CurGlobSol() const 
{
	return Convert((tREAL8*)nullptr,CurGlobSol());
}
// partial specialization not so important here, but to test
template <> cDenseVect<tREAL8>  cResolSysNonLinear<tREAL8>::R_CurGlobSol() const 
{
	return mCurGlobSol;
}




template <class Type> const Type & cResolSysNonLinear<Type>::CurSol(int aNumV) const
{
    return mCurGlobSol(aNumV);
}
template<class Type>tREAL8 cResolSysNonLinear<Type>::R_CurSol(int aNumV)const{return tREAL8(CurSol(aNumV));}



template <class Type> void cResolSysNonLinear<Type>::SetCurSol(int aNumV,const Type & aVal) 
{
    AssertNotInEquation();
    mCurGlobSol(aNumV) = aVal;
}

template <class Type> void cResolSysNonLinear<Type>::R_SetCurSol(int aNumV,const tREAL8 & aVal) 
{
	SetCurSol(aNumV,aVal);
}

     /*     =========================    adding equations ==================================
 
       # Global :
	      CalcVal  -> *  fondamental methods, 
	                  *  generate the computation of formal derivative and put the result
	                     in  vector<cInputOutputRSNL>,  
	                  *  used in case schur an
			  *  it's a vector to allow later, a possible use in parallel

       # Case Non Schur :
	      CalcAndAddObs ->   * calc CalcVal to compute derivative then AddObs
				
	      AddObs  ->         *  send cInputOutputRSNL this in linear system

       # Case Schur :

              AddEq2Subst   ->   *   CalcVal to compute derivative  then 
	                             push the result in a structure cSetIORSNL_SameTmp

              AddObsWithTmpUK -> *  sens a structure cSetIORSNL_SameTmp to the linear system
	                            to make the schur elimination

       # Remark : using schur and paralelizing formal derivatives will require additionnal methods.
                  (with probably more sophisticated/complex protocols)

     */

template <class Type> void   cResolSysNonLinear<Type>::CalcVal
                             (
			          tCalc * aCalcVal,
				  std::vector<tIO_RSNL>& aVIO,
				  const tStdVect & aValTmpUk,
				  bool WithDer,
				  const tResidualW & aWeighter,
                                  bool  ForConstraint
                              )
{
      // This test is always true 4 now, which I(MPD)  was not sure
      //  The possibility of having several comes from potential paralellization
      //  MMVII_INTERNAL_ASSERT_tiny(aVIO.size()==1,"CalcValCalcVal");
     
      if (!ForConstraint)
          SetPhaseEq(); 
      MMVII_INTERNAL_ASSERT_tiny(aCalcVal->NbInBuf()==0,"Buff not empty");

      // Usefull only to test correcness of DoOneEval
      bool  TestOneVal = aVIO.size()==1;
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
	  if (TestOneVal)
             aCalcVal->DoOneEval(aVCoord,aVObs);
	  else
             aCalcVal->PushNewEvals(aVCoord,aVObs);
      }
      // Make the computation
      if (!TestOneVal)
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
	   //  StdOut() << "HHHhUuHH  " << aIO.mVals  << " " << aIO.mWeights << std::endl;
           if (! ForConstraint)
	      ModifyFrozenVar(aIO);
      }
}

template <class Type> void cResolSysNonLinear<Type>::AddNonLinearConstr
                           (
                                  tCalc * aCalcVal,
			          const tVectInd & aVInd,
				  const tStdVect& aVObs,
                                  bool  OnlyIfFirst
                           )
{
    std::vector<tIO_RSNL> aVIO(1,tIO_RSNL(aVInd,aVObs));
    CalcVal(aCalcVal,aVIO,{},true,tResidualW(),true);

    // Parse all the linearized equation
    for (const auto & aIO : aVIO)
    {
	// check we dont use temporary value
        MMVII_INTERNAL_ASSERT_tiny(aIO.mNbTmpUk==0,"Cannot use tmp uk w/o Schur complement");
	// parse all values
	for (size_t aKVal=0 ; aKVal<aIO.mVals.size() ; aKVal++)
	{
	    tSVect aSV;
            const tStdVect & aVDer = aIO.mDers[aKVal];
	    for (size_t aKUk=0 ; aKUk<aIO.mGlobVInd.size() ; aKUk++)
            {
                aSV.AddIV(aIO.mGlobVInd[aKUk],aVDer[aKUk]);
	    }
            AddConstr(aSV,-aIO.mVals[aKVal],OnlyIfFirst);
	}
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
    CalcVal(aCalcVal,aVIO,{},true,aWeigther,false);
    AddObs(aVIO);
}

template <class Type>  void   cResolSysNonLinear<Type>::R_CalcAndAddObs
                              (
			            tCalc * aCalcVal,
                                    const tVectInd & aVInd,
				    const tR_Up::tStdVect& aR_VObs,
				    const tR_Up::tResidualW & aR_PtrRW
                               ) 
{
   tStdVect aVObs;
   cREAL8_RWAdapt<Type> aRW(&aR_PtrRW) ;
   CalcAndAddObs ( aCalcVal, aVInd, Convert(aVObs,aR_VObs), aRW);
}
template <>  void   cResolSysNonLinear<tREAL8>::R_CalcAndAddObs
                              (
			            tCalc * aCalcVal,
                                    const tVectInd & aVInd,
				    const tR_Up::tStdVect& aR_VObs,
				    const tR_Up::tResidualW & aRW
                               ) 
{
	CalcAndAddObs(aCalcVal,aVInd,aR_VObs, aRW);
}





template <class Type> void cResolSysNonLinear<Type>::AddObs(const std::vector<tIO_RSNL>& aVIO)
{
      SetPhaseEq(); 
      // Parse all the linearized equation
      for (const auto & aIO : aVIO)
      {
	  currNbObs += aIO.mVals.size();
	  // check we dont use temporary value
          MMVII_INTERNAL_ASSERT_tiny(aIO.mNbTmpUk==0,"Cannot use tmp uk w/o Schur complement");
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
	         mSysLinear->PublicAddObservation(aW,aSV,-aIO.mVals[aKVal]);
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
    CalcVal(aCalc,aVIO,aSetIO.ValTmpUk(),true,aWeighter,false);

    aSetIO.AddOneEq(aVIO.at(0));
}


template <class Type> void   cResolSysNonLinear<Type>::R_AddEq2Subst 
                             (
			          tR_Up::tSetIO_ST  & aSetIO,tCalc * aCalc,const tVectInd & aVInd,
                                  const tR_Up::tStdVect& aR_VObs,const tR_Up::tResidualW & aWeighter
                             )
{
    
    std::vector<tIO_RSNL> aVIO(1,tIO_RSNL(aVInd,VecConvert<Type,tREAL8>(aR_VObs)));
    CalcVal(aCalc,aVIO,VecConvert<Type,tREAL8>(aSetIO.ValTmpUk()),true,cREAL8_RWAdapt<Type>(&aWeighter),false);

    cInputOutputRSNL<tREAL8>  aRIO (aVInd,aR_VObs);
    ConvertVWD(aRIO,aVIO.at(0));
    aSetIO.AddOneEq(aRIO);
}

template <> void   cResolSysNonLinear<tREAL8>::R_AddEq2Subst 
                             (
			          tR_Up::tSetIO_ST  & aSetIO,tCalc * aCalc,const tVectInd & aVInd,
                                  const tR_Up::tStdVect& aVObs,const tR_Up::tResidualW & aWeighter
                             )
{
    AddEq2Subst(aSetIO,aCalc,aVInd,aVObs,aWeighter);
}


			     
template <class Type> void cResolSysNonLinear<Type>::AddObsWithTmpUK (const tSetIO_ST & aSetIO)
{
    currNbObs += aSetIO.NbRedundacy();
    mSysLinear->AddObsWithTmpUK(aSetIO);
}

template <class Type> void cResolSysNonLinear<Type>::R_AddObsWithTmpUK (const tR_Up::tSetIO_ST & aR_SetIO)
{
    cSetIORSNL_SameTmp<Type> aSetIO(false,aR_SetIO);
    AddObsWithTmpUK(aSetIO);
    // mSysLinear->AddObsWithTmpUK(aSetIO);
}

template <> void cResolSysNonLinear<tREAL8>::R_AddObsWithTmpUK (const tR_Up::tSetIO_ST & aSetIO)
{
    AddObsWithTmpUK(aSetIO);
}



            //  =========    resolving ==========================

template <class Type> const cDenseVect<Type> & cResolSysNonLinear<Type>::SolveUpdateReset(const Type & aLVM) 
{
    if (mNbVar>currNbObs)
    {
           //StdOut()  << "currNbObscurrNbObs " << currNbObs  << " RRRRR=" << currNbObs - mNbVar << std::endl;
        MMVII_DEV_WARNING("Not enough obs for var ");
    }
    lastNbObs = currNbObs;
    mInPhaseAddEq = false;
    // for var frozen, they are not involved in any equation, we must fix their value other way

#if (WithNewLinearCstr)
    mLinearConstr->AddConstraint2Sys(*mSysLinear);
#else
    for (int aK=0 ; aK<mNbVar ; aK++)
    {
        // CHANGE HERE
        if (mVarIsFrozen[aK])
           AddEqFixVar(aK,mValueFrozenVar[aK],1.0);
    }
#endif

    for (int aK=0 ; aK<mNbVar ; aK++)
    {
        if (aLVM>0)
        {
           AddEqFixVar(aK,CurSol(aK),mSysLinear->LVMW(aK)*aLVM);
        }
    }

    mCurGlobSol += mSysLinear->Solve();     //  mCurGlobSol += mSysLinear->SparseSolve();
    mSysLinear->Reset();
    currNbObs = 0;

    mNbIter++;
    return mCurGlobSol;
}

template <class Type> cDenseVect<tREAL8>  cResolSysNonLinear<Type>::R_SolveUpdateReset(const tREAL8 & aLVM) 
{
	return Convert((tREAL8*)nullptr,SolveUpdateReset(aLVM));
}



/* ************************************************************ */
/*                                                              */
/*                  INSTANTIATION                               */
/*                                                              */
/* ************************************************************ */

// template class  cInputOutputRSNL<TYPE>;
// template class  cSetIORSNL_SameTmp<TYPE>;

#define INSTANTIATE_RESOLSYSNL(TYPE)\
template class  cResolSysNonLinear<TYPE>;

INSTANTIATE_RESOLSYSNL(tREAL4)
INSTANTIATE_RESOLSYSNL(tREAL8)
INSTANTIATE_RESOLSYSNL(tREAL16)


};
