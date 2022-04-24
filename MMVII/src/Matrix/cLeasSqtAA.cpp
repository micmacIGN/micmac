#include "include/MMVII_all.h"
#include "include/MMVII_Tpl_Images.h"


namespace MMVII
{

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


template<class Type> cDenseVect<Type> cLeasSqtAA<Type>::Solve()
{
   mtAA.SelfSymetrizeBottom();
   return mtAA.Solve(mtARhs,eTyEigenDec::eTED_LLDT);
}

template<class Type> const cDenseMatrix<Type> & cLeasSqtAA<Type>::tAA () const {return mtAA;}
template<class Type> const cDenseVect<Type>   & cLeasSqtAA<Type>::tARhs () const {return mtARhs;}

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

template<class Type> void cLinearOverCstrSys<Type>::AddObsWithTmpK(const cSetIORSNL_SameTmp<Type>&)
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


