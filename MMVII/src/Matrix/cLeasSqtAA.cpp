
#include "MMVII_Tpl_Images.h"
#include "MMVII_SysSurR.h"

// test git cred again again ... vi 

namespace MMVII
{

/* *********************************** */
/*                                     */
/*            cBufSchurrSubst          */
/*                                     */
/* *********************************** */

template <class Type> 
    cBufSchurrSubst<Type>::cBufSchurrSubst(size_t aNbVar) :
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

template <class Type> const  std::vector<size_t> & cBufSchurrSubst<Type>::VIndexUsed() const
{
   return mSetInd.mVIndOcc;
}


template <class Type> const cDenseMatrix<Type> & cBufSchurrSubst<Type>::tAASubst() const
{
     return mM11;
}

template <class Type> const cDenseVect<Type> & cBufSchurrSubst<Type>::tARhsSubst() const
{
     return mC1;
}

template <class Type> 
    void cBufSchurrSubst<Type>::CompileSubst(const tSetEq & aSetSetEq)
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
     mNbUkTot = mNbUk + mNbTmp;

     // Adjust size, initialize of mSysRed
     if (mSysRed.NbVar() != int(mNbUkTot))
     {
         mSysRed = cLeasSqtAA<Type>(mNbUkTot);
     }
     else
     {
         mSysRed.Reset();
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

	      mSysRed.AddObservation(aSetEq.WeightOfKthResisual(aKEq),mSV,-aSetEq.mVals.at(aKEq));
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

template<class Type>  cLeasSqtAA<Type>::~cLeasSqtAA()
{
    delete mBSC;
}


template<class Type> void  cLeasSqtAA<Type>::AddObservation
                           (
                               const Type& aWeight,
                               const cDenseVect<Type> & aCoeff,
                               const Type &  aRHS
                           ) 
{
    mtAA.Weighted_Add_tAA(aWeight,aCoeff,true);
    WeightedAddIn(mtARhs.DIm(),aWeight*aRHS,aCoeff.DIm());
}

template<class Type> void  cLeasSqtAA<Type>::AddObservation
                           (
                               const Type& aWeight,
                               const cSparseVect<Type> & aCoeff,
                               const Type &  aRHS
                           ) 
{
    mtAA.Weighted_Add_tAA(aWeight,aCoeff,true);
    mtARhs.WeightedAddIn(aWeight*aRHS,aCoeff);
}



template<class Type> void  cLeasSqtAA<Type>::Reset()
{
   mtAA.DIm().InitNull();
   mtARhs.DIm().InitNull();
}

template<class Type> void  cLeasSqtAA<Type>::AddObsWithTmpUK(const cSetIORSNL_SameTmp<Type>& aSetSetEq) 
{
    if (mBSC==nullptr)
         mBSC = new cBufSchurrSubst<Type>(this->NbVar());
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

template<class Type> cDenseVect<Type> cLeasSqtAA<Type>::Solve()
{
   mtAA.SelfSymetrizeBottom();
   return mtAA.SolveColumn(mtARhs,eTyEigenDec::eTED_LLDT);
}

template<class Type> const cDenseMatrix<Type> & cLeasSqtAA<Type>::tAA () const {return mtAA;}
template<class Type> const cDenseVect<Type>   & cLeasSqtAA<Type>::tARhs () const {return mtARhs;}
template<class Type> cDenseMatrix<Type> & cLeasSqtAA<Type>::tAA ()   {return mtAA;}
template<class Type> cDenseVect<Type>   & cLeasSqtAA<Type>::tARhs () {return mtARhs;}

template<class Type> cDenseVect<Type> cLeasSqtAA<Type>::SparseSolve()
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



/* *********************************** */
/*                                     */
/*            cLeasSq                  */
/*                                     */
/* *********************************** */


template<class Type>  cLeasSq<Type>::cLeasSq(int aNbVar):
     cLinearOverCstrSys<Type> (aNbVar)
{
}

template<class Type> Type  cLeasSq<Type>::Residual
                             (
                                 const cDenseVect<Type> & aVect,
                                 const Type& aWeight,
                                 const cDenseVect<Type> & aCoeff,
                                 const Type &  aRHS
                             ) const
{
   return aWeight * Square(aVect.DotProduct(aCoeff)-aRHS);
}

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
   mNbVar (aNbVar)
{
}

template<class Type> cLinearOverCstrSys<Type>::~cLinearOverCstrSys()
{
}

template<class Type> int cLinearOverCstrSys<Type>::NbVar() const
{
   return mNbVar;
}

template<class Type> void cLinearOverCstrSys<Type>::AddObsFixVar(const Type& aWeight,int aIndVal,const Type & aVal)
{
   cSparseVect<Type> aSpV;
   aSpV.AddIV(aIndVal,1.0);
   // static cIndSV<Type> & aIV = aSpV.IV()[0];
   // aIV.mInd  = aIndVal;
   // aIV.mVal  = 1.0;
   
   AddObservation(aWeight,aSpV,aVal);
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

template<class Type> cDenseVect<Type> cLinearOverCstrSys<Type>::SparseSolve()
{
     return Solve();
}

template<class Type> cLinearOverCstrSys<Type> * cLinearOverCstrSys<Type>::AllocSSR(eModeSSR aMode,int aNbVar)
{
     switch (aMode)
     {
	     case eModeSSR::eSSR_LsqDense  :  return cLeasSq<Type>::AllocDenseLstSq (aNbVar);
	     case eModeSSR::eSSR_LsqNormSparse :  return cLeasSq<Type>::AllocSparseNormalLstSq(aNbVar,cParamSparseNormalLstSq());
	     case eModeSSR::eSSR_LsqSparseGC :  return cLeasSq<Type>::AllocSparseGCLstSq(aNbVar);
             
             default :;
     }

     MMVII_INTERNAL_ERROR("Bad enumerated valure for AllocSSR");
     return nullptr;
}

template<class Type> void cLinearOverCstrSys<Type>::AddObsWithTmpUK(const cSetIORSNL_SameTmp<Type>&)
{
	MMVII_INTERNAL_ERROR("Used AddObsWithTmpK unsupported");
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
template class cBufSchurrSubst<Type>;\
template  class  cLeasSqtAA<Type>;\
template  class  cLeasSq<Type>;\
template  class  cLinearOverCstrSys<Type>;


INSTANTIATE_LEASTSQ_TAA(tREAL4)
INSTANTIATE_LEASTSQ_TAA(tREAL8)
INSTANTIATE_LEASTSQ_TAA(tREAL16)


};


