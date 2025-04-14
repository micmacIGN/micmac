
#include "MMVII_Tpl_Images.h"
#include "MMVII_SysSurR.h"

// test git cred again again ... vi 

namespace MMVII
{

/* *********************************** */
/*                                     */
/*            cBufSchurSubst          */
/*                                     */
/* *********************************** */

template <class Type> 
    cBufSchurSubst<Type>::cBufSchurSubst(size_t aNbVar) :
         mNbVar     (aNbVar),
         mSetInd    (aNbVar),
	 mSysRed    (1),
	 mL         (1,1),
	 mtB        (1,1),
	 mtB_LInv   (1,1),
	 mLInv_B    (1,1),
	 mB         (1,1),
	 mtB_LInv_B (1,1),
         mM11       (1,1),
	 mA         (1),
	 mC1        (1),
         mtB_LInv_A (1)
{
}

template <class Type> const  std::vector<size_t> & cBufSchurSubst<Type>::VIndexUsed() const
{
   return mSetInd.mVIndOcc;
}


template <class Type> const cDenseMatrix<Type> & cBufSchurSubst<Type>::tAASubst() const
{
     return mM11;
}

template <class Type> const cDenseVect<Type> & cBufSchurSubst<Type>::tARhsSubst() const
{
     return mC1;
}

template <class Type> 
    void cBufSchurSubst<Type>::CompileSubst(const tSetEq & aSetSetEq)
{
     aSetSetEq.AssertOk();

     //  Compute all the index used in aSetSetEq
     mSetInd.Clear();
     mNbTmp = aSetSetEq.NbTmpUk();
     for (const auto & anEq : aSetSetEq.AllEq())
     {
         for (const auto & anInd : anEq.mGlobVInd)
	 {
             if (!cSetIORSNL_SameTmp<Type>::IsIndTmp(anInd))
	     {
                 mSetInd.AddInd(anInd);
	     }
	 }
     }
     // mSetInd.SortInd();
     mSetInd.MakeInvertIndex();

     mNbUk = mSetInd.mVIndOcc.size();
     if (mNbUk==0) return; // nothing to substitute

     mNbUkTot = mNbUk + mNbTmp;

     // Adjust size, initialize of mSysRed
     if (mSysRed.NbVar() != int(mNbUkTot))
     {
         mSysRed = cLeasSqtAA<Type>(mNbUkTot);
     }
     else
     {
         mSysRed.PublicReset();
     }

     //  Compute the reduced  least square system
     for (const auto & aSetEq : aSetSetEq.AllEq())
     {
         for (size_t aKEq=0 ; aKEq<aSetEq.mVals.size() ; aKEq++)
	 {
              mSV.Reset();
	      const std::vector<Type> & aVDer = aSetEq.mDers.at(aKEq);

              for (size_t aKGlob=0 ; aKGlob<aSetEq.mGlobVInd.size() ; aKGlob++)
              {
                   const Type  & aDer = aVDer.at(aKGlob);
                   int aInd = aSetEq.mGlobVInd[aKGlob];
                   if ( cSetIORSNL_SameTmp<Type>::IsIndTmp(aInd))
                       mSV.AddIV(cSetIORSNL_SameTmp<Type>::ToIndTmp(aInd),aDer);
                   else
                       mSV.AddIV(mNbTmp+mSetInd.mVInvertInd.at(aInd),aDer);
                     
              }

	      mSysRed.PublicAddObservation(aSetEq.WeightOfKthResisual(aKEq),mSV,-aSetEq.mVals.at(aKEq));
	 }
      }

     //  extract normal matrix, vector, symetrise
      cDenseMatrix<Type> & atAA    =  mSysRed.tAA();
      cDenseVect<Type> & atARhs  =  mSysRed.tARhs();
      atAA.SelfSymetrizeBottom();
      cPt2di aSzTmp(mNbTmp,mNbTmp);

      //  Extract 4 bloc matrices and 2 bloc vectors
      mL.ResizeAndCropIn(cPt2di(0,0),aSzTmp,atAA);
      mM11.ResizeAndCropIn(aSzTmp,cPt2di(mNbUkTot,mNbUkTot),atAA);
      mtB.ResizeAndCropIn(cPt2di(0,mNbTmp),cPt2di(mNbTmp,mNbUkTot),atAA);
      mB.ResizeAndCropIn(cPt2di(mNbTmp,0),cPt2di(mNbUkTot,mNbTmp),atAA);
      mA.ResizeAndCropIn(0,mNbTmp,atARhs);
      mC1.ResizeAndCropIn(mNbTmp,mNbUkTot,atARhs);



      // compute tB*L-1 in  mtB_LInv
      mtB_LInv.Resize(cPt2di(mNbTmp,mNbUk));
      mLInv_B.Resize(cPt2di(mNbUk,mNbTmp));
      mL.SolveIn(mLInv_B,mB,eTyEigenDec::eTED_LLDT);   // mLInv_B = L-1 B
      mLInv_B.TransposeIn(mtB_LInv);                   // mtB_LInv = tB L-1 as L=tL
  

      // compute tB*L-1*B  in  mtB_mLInv_B
      mtB_LInv_B.Resize(cPt2di(mNbUk,mNbUk)) ;
      mtB_LInv_B.MatMulInPlace(mtB_LInv,mB); //  ============  TO OPTIM MATR SYM

      // compute tB*L-1*A in  mtB_mLInv_A
      mtB_LInv_A.Resize(mNbUk);
      mtB_LInv.MulColInPlace(mtB_LInv_A,mA);

      //   substract vec and matr to have formula of doc
      mM11 -= mtB_LInv_B;
      mC1 -= mtB_LInv_A;

}




/* *********************************** */
/*                                     */
/*            cLeasSqtAA               */
/*                                     */
/* *********************************** */

template<class Type>  cLeasSqtAA<Type>::cLeasSqtAA(int aNbVar):
   cLeasSq<Type>   (aNbVar),
   mtAA            (aNbVar,aNbVar,eModeInitImage::eMIA_Null),
   mtARhs          (aNbVar,eModeInitImage::eMIA_Null),
   mBSC            (nullptr)
{
}


template<class Type>  cLeasSqtAA<Type>  cLeasSqtAA<Type>::Dup() const
{
     cLeasSqtAA<Type>  aRes(this->NbVar());

     mtAA.DIm().DupIn(aRes.mtAA.DIm());
     mtARhs.DIm().DupIn(aRes.mtARhs.DIm());

     return aRes;
}


template<class Type>  cLeasSqtAA<Type>::~cLeasSqtAA()
{
    delete mBSC;
}


template<class Type> void  cLeasSqtAA<Type>::SpecificAddObservation
                           (
                               const Type& aWeight,
                               const cDenseVect<Type> & aCoeff,
                               const Type &  aRHS
                           ) 
{
    mtAA.Weighted_Add_tAA(aWeight,aCoeff,true);
    WeightedAddIn(mtARhs.DIm(),aWeight*aRHS,aCoeff.DIm());
}

template<class Type> void  cLeasSqtAA<Type>::SpecificAddObservation
                           (
                               const Type& aWeight,
                               const cSparseVect<Type> & aCoeff,
                               const Type &  aRHS
                           ) 
{
    mtAA.Weighted_Add_tAA(aWeight,aCoeff,true);
    mtARhs.WeightedAddIn(aWeight*aRHS,aCoeff);
}



template<class Type> void  cLeasSqtAA<Type>::SpecificReset()
{
   mtAA.DIm().InitNull();
   mtARhs.DIm().InitNull();
}

template<class Type> void  cLeasSqtAA<Type>::SpecificAddObsWithTmpUK(const cSetIORSNL_SameTmp<Type>& aSetSetEq) 
{
    if (mBSC==nullptr)
         mBSC = new cBufSchurSubst<Type>(this->NbVar());
    mBSC->CompileSubst(aSetSetEq);

    const std::vector<size_t> &  aVI = mBSC->VIndexUsed();
    const cDenseMatrix<Type> & atAAS =  mBSC->tAASubst() ;
    const cDenseVect<Type> &   atARhsS = mBSC->tARhsSubst() ;

    for (size_t aI =0 ; aI<aVI.size() ; aI++)
    {
         size_t aX = aVI[aI];
         for (size_t aJ =0 ; aJ<aVI.size() ; aJ++)
         {
             size_t aY = aVI[aJ];
             mtAA.AddElem(aX,aY,atAAS.GetElem(aI,aJ)); 
         }
         mtARhs(aX) += atARhsS(aI);
    }
} 

template<class Type> cDenseVect<Type> cLeasSqtAA<Type>::SpecificSolve()
{
   mtAA.SelfSymetrizeBottom();
   return mtAA.SolveColumn(mtARhs,eTyEigenDec::eTED_LLDT);
}

template<class Type> const cDenseMatrix<Type> & cLeasSqtAA<Type>::tAA () const {return mtAA;}
template<class Type> const cDenseVect<Type>   & cLeasSqtAA<Type>::tARhs () const {return mtARhs;}
template<class Type> cDenseMatrix<Type> & cLeasSqtAA<Type>::tAA ()   {return mtAA;}
template<class Type> cDenseVect<Type>   & cLeasSqtAA<Type>::tARhs () {return mtARhs;}

template<class Type> cDenseVect<Type> cLeasSqtAA<Type>::SpecificSparseSolve()
{
   const  cDataIm2D<Type> & aDIm = mtAA.DIm();
   std::vector<cEigenTriplet<Type> > aVCoeff;            // list of non-zeros coefficients
   for (const auto & aPix : aDIm)
   {
       const Type & aVal = aDIm.GetV(aPix);
       if ((aVal != 0.0)  && (aPix.x()>=aPix.y()))
       {
           cEigenTriplet<Type>  aTri(aPix.x(),aPix.y(),aVal);
           aVCoeff.push_back(aTri);
       }
   }

   return EigenSolveCholeskyarseFromV3(aVCoeff,mtARhs);
}

template<class Type> cDenseMatrix<Type> cLeasSqtAA<Type>::V_tAA() const
{
     cDenseMatrix<Type> aRes = mtAA.Dup();
     aRes.SelfSymetrizeBottom();
     return aRes;
}

template<class Type> cDenseVect<Type> cLeasSqtAA<Type>::V_tARhs() const
{
	return mtARhs;
}
template<class Type> bool cLeasSqtAA<Type>::Acces2NormalEq() const
{
	return true;
}

template<class Type> void cLeasSqtAA<Type>::AddCov
                          (const cDenseMatrix<Type> & aMat,const cDenseVect<Type>& aVect,const std::vector<int> &aVInd)
{
    for (int aKx = 0 ; aKx<int(aVInd.size()) ; aKx++)
    {
        mtARhs(aVInd[aKx]) += aVect(aKx);
        for (int aKy = 0 ; aKy<int(aVInd.size()) ; aKy++)
        {
             // Only triangular sup used
	     if (aVInd[aKx] >= aVInd[aKy])
	        mtAA.AddElem(aVInd[aKx],aVInd[aKy],aMat.GetElem(aKx,aKy));
        }
    }
}

template<class Type> cDenseMatrix<Type> cLeasSqtAA<Type>::tAA_Solve(const cDenseMatrix<Type> & aMat) const
{
    cDenseMatrix<Type> & atAA = const_cast<cLeasSqtAA<Type>* >(this) ->mtAA;
    atAA.SelfSymetrizeBottom();
    return mtAA.Solve(aMat);
}



/* *********************************** */
/*                                     */
/*            cLeasSq                  */
/*                                     */
/* *********************************** */


template<class Type>  cLeasSq<Type>::cLeasSq(int aNbVar):
     cLinearOverCstrSys<Type> (aNbVar)
{
}

template<class Type> Type  cLeasSq<Type>::ResidualOf1Eq
                             (
                                 const cDenseVect<Type> & aVect,
                                 const Type& aWeight,
                                 const cDenseVect<Type> & aCoeff,
                                 const Type &  aRHS
                             ) const
{
   return aWeight * Square(aVect.DotProduct(aCoeff)-aRHS);
}


template<class Type> Type  cLeasSq<Type>::ResidualOf1Eq
                             (
                                 const cDenseVect<Type> & aVect,
                                 const Type& aWeight,
                                 const cSparseVect<Type> & aSparseCoeff,
                                 const Type &  aRHS
                             ) const
{
   return aWeight * Square(aSparseCoeff.DotProduct(aVect)-aRHS);
   // return 0.0;
}
/*
*/





template<class Type> cLeasSq<Type> * cLeasSq<Type>::AllocDenseLstSq(int aNbVar)
{
	return new cLeasSqtAA<Type>(aNbVar);
}

/* *********************************** */
/*                                     */
/*         cLinearOverCstrSys          */
/*                                     */
/* *********************************** */



template<class Type> cLinearOverCstrSys<Type>::cLinearOverCstrSys(int aNbVar) :
   mNbVar           (aNbVar),
   mLVMW            (aNbVar,eModeInitImage::eMIA_Null),
   mSumWCoeffRHS    (aNbVar,eModeInitImage::eMIA_Null),
   mSumWRHS2        (0.0),
   mSumW            (0.0),
   mLastSumWRHS2    (0.0),
   mLastResComp     (false),
   mLastResidual    (0.0),
   mSchurrWasUsed   (false)
{
}

template<class Type> void cLinearOverCstrSys<Type>::AddWRHS(Type aW,Type aRHS)
{
   mSumWRHS2 += aW*Square(aRHS);
   // mSumW     += aW;
   mSumW        += 1;
}


template<class Type> void cLinearOverCstrSys<Type>::PublicReset()
{
     SpecificReset();

     mLVMW.DIm().InitNull();
     mSumWCoeffRHS.DIm().InitNull();
     mSumWRHS2 = 0 ;
     mSumW     = 0 ;
     mSchurrWasUsed = false;
}

template<class Type> cDenseVect<Type> cLinearOverCstrSys<Type>::PublicSolve()
{
     cDenseVect<Type> aSol =  SpecificSolve();
/*
StdOut() << "PSol, W=" << mSumW 
         << " RW2=" << mSumWRHS2 
         << " Scal=" <<  mSumWCoeffRHS.DotProduct(aSol) 
         << "\n";
*/

     //mLastResidual = mSumWRHS2 - mSumWCoeffRHS.DotProduct(aSol);
     mLastResidual = mSumWRHS2 - mSumWCoeffRHS.DotProduct(aSol);
     mLastSumWRHS2 = mSumWRHS2;
     mLastSumW     = mSumW;
     return aSol;
}

template<class Type> Type cLinearOverCstrSys<Type>::VarOfSol(const cDenseVect<Type> & aSol)  const
{
    return  (mSumWRHS2 - mSumWCoeffRHS.DotProduct(aSol)) / mSumW;
}

template<class Type> Type cLinearOverCstrSys<Type>::VarLastSol() const
{
    return mLastSumWRHS2 / mLastSumW;
}


template<class Type> Type cLinearOverCstrSys<Type>::VarCurSol()  const
{
   return std::max( Type(0.0),mLastResidual / mLastSumW);
}



template<class Type> cLinearOverCstrSys<Type>::~cLinearOverCstrSys()
{
}

template<class Type> int cLinearOverCstrSys<Type>::NbVar() const
{
   return mNbVar;
}

template<class Type> Type cLinearOverCstrSys<Type>::LVMW(int aK) const
{
    if (false && (aK==0))
	StdOut() << "========== LVMINIT=" << mNbVar << " " << mLVMW(aK) << "\n";
   return mLVMW(aK);
}

template<class Type> void cLinearOverCstrSys<Type>::AddObsFixVar(const Type& aWeight,int aIndVal,const Type & aVal)
{
   cSparseVect<Type> aSpV;
   aSpV.AddIV(aIndVal,1.0);
   // static cIndSV<Type> & aIV = aSpV.IV()[0];
   // aIV.mInd  = aIndVal;
   // aIV.mVal  = 1.0;
   
   PublicAddObservation(aWeight,aSpV,aVal);
}

template<class Type> Type cLinearOverCstrSys<Type>::ResidualOf1Eq
                             (
                                 const cDenseVect<Type> & aVect,
                                 const Type& aWeight,
                                 const cSparseVect<Type> & aSparseCoeff,
                                 const Type &  aRHS
                             ) const
{
      // default method, generate a dense vector, not very efficient but OK ...
      cDenseVect<Type>  aDenseCoeff(aSparseCoeff,aVect.Sz());

      return ResidualOf1Eq(aVect,aWeight,aDenseCoeff,aRHS);
}


template<class Type> 
    void cLinearOverCstrSys<Type>::SpecificAddObs_UsingCast2Sparse
         (
              const Type& aWeight,
              const cDenseVect<Type> & aCoeff,
              const Type &  aRHS
         ) 
{
    SpecificAddObservation(aWeight,cSparseVect(aCoeff),aRHS);
}



template<class Type> void cLinearOverCstrSys<Type>::PublicAddObservation (const Type& aWeight,const cDenseVect<Type> & aCoeff,const Type &  aRHS)
{
     // No harm to do this optimization , even if done elsewhere
     if (aWeight==0)  return;

     AddWRHS(aWeight,aRHS);
     mSumWCoeffRHS.WeightedAddIn(aWeight*aRHS,aCoeff);

     SpecificAddObservation(aWeight,aCoeff,aRHS);
     for (int aKV=0 ; aKV<mNbVar ; aKV++)
        mLVMW(aKV) += aWeight * Square(aCoeff(aKV));
}
template<class Type> void cLinearOverCstrSys<Type>::PublicAddObservation (const Type& aWeight,const cSparseVect<Type> & aCoeff,const Type &  aRHS)
{
     // No harm to do this optimization , even if done elsewhere
     if (aWeight==0)  return;

     AddWRHS(aWeight,aRHS);
     mSumWCoeffRHS.WeightedAddIn(aWeight*aRHS,aCoeff);

     SpecificAddObservation(aWeight,aCoeff,aRHS);
     for (const auto & aPair : aCoeff)
        mLVMW(aPair.mInd) += aWeight * Square(aPair.mVal);
}

template<class Type> void cLinearOverCstrSys<Type>::AddObsFixVar(const Type& aWeight,const cSparseVect<Type> & aVVarVals)
{
    for (const auto & aP : aVVarVals)
        AddObsFixVar(aWeight,aP.mInd,aP.mVal);
}

template<class Type> void cLinearOverCstrSys<Type>::AddObsFixVar (const Type& aWeight,const cDenseVect<Type>  &  aVRHS)
{
    MMVII_INTERNAL_ASSERT_medium(aVRHS.Sz() == mNbVar,"cLinearOverCstrSys<Type>::AddObsFixVar");
    for (int aK=0 ; aK<mNbVar ; aK++)
        AddObsFixVar(aWeight,aK,aVRHS(aK));
}

template<class Type> cDenseVect<Type> cLinearOverCstrSys<Type>::SpecificSparseSolve()
{
   return this->SpecificSolve();
}

template<class Type> cDenseVect<Type> cLinearOverCstrSys<Type>::PublicSparseSolve()
{
   return SpecificSparseSolve();
}

template<class Type> cLinearOverCstrSys<Type> * cLinearOverCstrSys<Type>::AllocSSR(eModeSSR aMode,int aNbVar)
{
     switch (aMode)
     {
	     case eModeSSR::eSSR_LsqDense  :      return cLeasSq<Type>::AllocDenseLstSq (aNbVar);
	     case eModeSSR::eSSR_LsqNormSparse :  return cLeasSq<Type>::AllocSparseNormalLstSq(aNbVar,cParamSparseNormalLstSq());
	     case eModeSSR::eSSR_LsqSparseGC :    return cLeasSq<Type>::AllocSparseGCLstSq(aNbVar);
	     case eModeSSR::eSSR_L1Barrodale :    return AllocL1_Barrodale<Type>(aNbVar);
             
             default :;
     }

     MMVII_INTERNAL_ERROR("Bad enumerated valure for AllocSSR");
     return nullptr;
}

template<class Type> void cLinearOverCstrSys<Type>::SpecificAddObsWithTmpUK(const cSetIORSNL_SameTmp<Type>&)
{
	MMVII_INTERNAL_ERROR("Used AddObsWithTmpK unsupported");
}

template<class Type> void cLinearOverCstrSys<Type>::PublicAddObsWithTmpUK(const cSetIORSNL_SameTmp<Type>& aSetSetEq)
{
     SpecificAddObsWithTmpUK(aSetSetEq);
     mSchurrWasUsed = true;

     for (const auto & aSetEq : aSetSetEq.AllEq())
     {
         // For example parse the two equation on i,j residual
         for (size_t aKEq=0 ; aKEq<aSetEq.mVals.size() ; aKEq++)
         {
                 const std::vector<Type> & aVDer = aSetEq.mDers.at(aKEq);
                 Type aWeight = aSetEq.WeightOfKthResisual(aKEq);

                 Type aVal = aSetEq.mVals.at(aKEq);
                 AddWRHS(aWeight,aVal);
                 cSparseVect<Type>  aVNonTmp;

                 for (size_t aKGlob=0 ; aKGlob<aSetEq.mGlobVInd.size() ; aKGlob++)
                 {
                     int aInd = aSetEq.mGlobVInd[aKGlob];
                     if (!cSetIORSNL_SameTmp<Type>::IsIndTmp(aInd))
		     {
                         Type aDer = aVDer.at(aKGlob);
                         mLVMW(aInd) += aWeight * Square(aDer);
                         aVNonTmp.AddIV(aInd,aDer);
		     }
                 }
                 mSumWCoeffRHS.WeightedAddIn(aWeight*(-aVal),aVNonTmp);
                 // Note the minus sign because we have a taylor expansion we need to annulate
         }
     }
}

template<class Type> cDenseMatrix<Type> cLinearOverCstrSys<Type>::tAA_Solve(const cDenseMatrix<Type> & aMat) const
{
    MMVII_INTERNAL_ERROR("No acces to tAA_Solve for this class");
    return cDenseMatrix<Type> (0);
}


template<class Type> cDenseMatrix<Type> cLinearOverCstrSys<Type>::V_tAA() const
{
	MMVII_INTERNAL_ERROR("No acces to tAA for this class");
    return cDenseMatrix<Type> (0);
	//return *((cDenseMatrix<Type> *)nullptr);   // clang: binding dereferenced null pointer to reference has undefined behavior
}

template<class Type> cDenseVect<Type> cLinearOverCstrSys<Type>::V_tARhs() const
{
	MMVII_INTERNAL_ERROR("No acces to tARhs for this class");
    return cDenseVect<Type> (0);
	//return *((cDenseVect<Type> *)nullptr);   // clang: binding dereferenced null pointer to reference has undefined behavior
}

template <class Type> void cLinearOverCstrSys<Type>::AddCov
                          (const cDenseMatrix<Type> &,const cDenseVect<Type>& ,const std::vector<int> &aVInd)
{
	MMVII_INTERNAL_ERROR("No AddCov for this class");
}



template<class Type> bool cLinearOverCstrSys<Type>::Acces2NormalEq() const
{
	return false;
}




/* ===================================================== */
/* ===================================================== */
/* ===================================================== */

#define INSTANTIATE_LEASTSQ_TAA(Type)\
template class cBufSchurSubst<Type>;\
template  class  cLeasSqtAA<Type>;\
template  class  cLeasSq<Type>;\
template  class  cLinearOverCstrSys<Type>;


INSTANTIATE_LEASTSQ_TAA(tREAL4)
INSTANTIATE_LEASTSQ_TAA(tREAL8)
INSTANTIATE_LEASTSQ_TAA(tREAL16)


};


