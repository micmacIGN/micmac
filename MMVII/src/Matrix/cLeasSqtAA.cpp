#include "include/MMVII_all.h"
#include "include/MMVII_Tpl_Images.h"

// test git cred again again ... vi 

namespace MMVII
{


/*
     We use notation of MMV1 doc   L= Lamda

     (L      B   .. )     (X)   (A )
     (tB     M11 .. )  *  (Y) = (C1)   =>     (M11- tB L-1 B    ...)  (Y) = C1 - tB L-1 A
       ...                (Z)    ..           (                 ...)  (Z)
*/

template <class Type> class  cBufSchurrSubst
{
     public :
          typedef cSetIORSNL_SameTmp<Type>  tSetEq;

          cBufSchurrSubst(size_t aNbVar);
	  void CompileSubst(const tSetEq &);

	  /// return normal matrix after schurr substitution ie : M11- tB L-1 B  
          const cDenseMatrix<Type> & tAASubst() const;
	  ///  return normal vector after schurr subst  ie : C1 - tB L-1 A
          const cDenseVect<Type> & tARhsSubst() const;
     private :

	  size_t            mNbVar;
	  std::vector<int>  mNumComp;
	  cSetIntDyn        mSetInd;
	  cLeasSqtAA<Type>  mSysRed;
	  cSparseVect<Type> mSV;
	  size_t            mNbTmp;
	  size_t            mNbUk;
	  size_t            mNbUkTot;

          cDenseMatrix<Type> mL;
          cDenseMatrix<Type> mLInv;
          cDenseMatrix<Type> mtB;
          cDenseMatrix<Type> mtB_LInv;
          cDenseMatrix<Type> mB;
          cDenseMatrix<Type> mtB_LInv_B;
          cDenseMatrix<Type> mM11;

          cDenseVect<Type>   mA;
          cDenseVect<Type>   mC1;
          cDenseVect<Type>   mtB_LInv_A;
};

template <class Type> 
    cBufSchurrSubst<Type>::cBufSchurrSubst(size_t aNbVar) :
         mNbVar     (aNbVar),
         mNumComp   (aNbVar,-1),
         mSetInd    (aNbVar),
	 mSysRed    (1),
	 mL         (1,1),
	 mLInv      (1,1),
	 mtB        (1,1),
	 mtB_LInv   (1,1),
	 mB         (1,1),
	 mtB_LInv_B (1,1),
         mM11       (1,1),
	 mA         (1),
	 mC1        (1),
         mtB_LInv_A (1)
{
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
         for (const auto & anInd : anEq.mVInd)
             mSetInd.AddInd(anInd);
     }
     mNbUk = mSetInd.mVIndOcc.size();
     mNbUkTot = mNbUk + mNbTmp;

StdOut() << " NbTmp:" <<  mNbTmp << " Uk:"<< mNbUk << " Tot:" << mNbUkTot << "\n";

     // Compute invert index  [0 NbVar[ ->  [0,NbUk[
     for (size_t aK=0; aK<mSetInd.mVIndOcc.size() ;aK++)
     {
          mNumComp.at(mSetInd.mVIndOcc[aK]) = aK;
     }

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
         const std::vector<int> & aVI =   aSetEq.mVInd;
	 size_t aNbI = aVI.size();
         for (size_t aKEq=0 ; aKEq<aSetEq.mVals.size() ; aKEq++)
	 {
              mSV.Reset();
	      const std::vector<Type> & aVDer = aSetEq.mDers.at(aKEq);

	      // fill sparse vector with  "real" unknown
              for (size_t aKV=0 ; aKV< aNbI ; aKV++)
	      {
                  mSV.AddIV(mNbTmp+mNumComp.at(aVI[aKV]),aVDer.at(aKV));
	      }

	      // fill sparse vector with  temporary unknown
              for (size_t  aKV=aNbI ; aKV<aVDer.size() ; aKV++)
	      {
                  mSV.AddIV((aKV-aNbI),aVDer.at(aKV));
	      }
	      // fill reduced normal equation
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


      // compute L-1 in  mLInv
      mLInv.Resize(aSzTmp);
      mLInv.InverseInPlace(mL);  //  ============  TO OPTIM MATR SYM

      // compute tB*L-1 in  mtB_mLInv
      mtB_LInv.Resize(cPt2di(mNbTmp,mNbUk));
StdOut() << "tBSZ: " << mtB_LInv.Sz() <<  cPt2di(mNbTmp,mNbUk) << "\n";
      mtB_LInv.MatMulInPlace(mtB,mLInv);

      // compute tB*L-1*B  in  mtB_mLInv_B
      mtB_LInv_B.Resize(cPt2di(mNbTmp,mNbUk)) ;
      mtB_LInv_B.MatMulInPlace(mtB_LInv,mB); //  ============  TO OPTIM MATR SYM

StdOut() << "SCHURRR " << __LINE__ << "\n";
StdOut() << mtB_LInv.Sz() << " " <<  mtB_LInv_A.Sz() << " " << mA.Sz() << "\n";
      // compute tB*L-1*A in  mtB_mLInv_A
      mtB_LInv_A.Resize(mNbUk);
      mtB_LInv.MulColInPlace(mtB_LInv_A,mA);

StdOut() << "SCHURRR " << __LINE__ << "\n";
      //   substract vec and matr to have formula of doc
      mM11 -= mtB_LInv_B;
      mC1 -= mtB_LInv_A;

}


template class cBufSchurrSubst<tREAL8>;


/* *********************************** */
/*                                     */
/*            cLeasSqtAA               */
/*                                     */
/* *********************************** */

template<class Type>  cLeasSqtAA<Type>::cLeasSqtAA(int aNbVar):
   cLeasSq<Type>   (aNbVar),
   mtAA            (aNbVar,aNbVar,eModeInitImage::eMIA_Null),
   mtARhs          (aNbVar,eModeInitImage::eMIA_Null)
{
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
    static cBufSchurrSubst<Type> *  mBSCS = new cBufSchurrSubst<Type>(this->NbVar());
    StdOut() << "hhhhhhhhhhhhhhhh\n"; 
    mBSCS->CompileSubst(aSetSetEq);
    
    StdOut() << "GGGGgggggggggggg\n"; getchar();
} 

template<class Type> cDenseVect<Type> cLeasSqtAA<Type>::Solve()
{
   mtAA.SelfSymetrizeBottom();
   return mtAA.Solve(mtARhs,eTyEigenDec::eTED_LLDT);
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
	     case eModeSSR::eSSR_LsqNormSparse :  return cLeasSq<Type>::AllocSparseNormalLstSq(aNbVar);
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



/* ===================================================== */
/* ===================================================== */
/* ===================================================== */

#define INSTANTIATE_LEASTSQ_TAA(Type)\
template  class  cLeasSqtAA<Type>;\
template  class  cLeasSq<Type>;\
template  class  cLinearOverCstrSys<Type>;


INSTANTIATE_LEASTSQ_TAA(tREAL4)
INSTANTIATE_LEASTSQ_TAA(tREAL8)
INSTANTIATE_LEASTSQ_TAA(tREAL16)


};


