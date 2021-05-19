#include "include/MMVII_all.h"

namespace MMVII
{

/* ============================================= */
/*      cLeastSqComputeMaps<Type>                */
/* ============================================= */

template <class Type,const int  DimIn,const int DimOut> 
     cLeastSqComputeMaps<Type,DimIn,DimOut>::cLeastSqComputeMaps(size_t aNbFunc) :
        mNbFunc  (aNbFunc),
        mLSQ     (aNbFunc),
        mCoeffs  (aNbFunc),
        mBufPOut (aNbFunc)
{
}

template <class Type,const int  DimIn,const int DimOut> 
     void cLeastSqComputeMaps<Type,DimIn,DimOut>::AddObs
            (const tPtIn & aPtIn,const tPtOut & aValue,const tPtOut & aPds)
{
    ComputeValFuncsBase(mBufPOut,aPtIn);
    cDataIm1D<Type> & aDIm = mCoeffs.DIm();

    for (size_t aD=0 ; aD<DimOut ; aD++)
    {
       for (size_t aKFunc=0 ; aKFunc<mNbFunc ; aKFunc++)
       {
           aDIm.SetV(aKFunc,mBufPOut.at(aKFunc)[aD]);
       }
       mLSQ.AddObservation(aPds[aD],mCoeffs,aValue[aD]);
    }
}

template <class Type,const int  DimIn,const int DimOut> 
     void cLeastSqComputeMaps<Type,DimIn,DimOut>::AddObs
            (const tPtIn & aPtIn,const tPtOut & aValue,const Type & aPds)
{
     AddObs(aPtIn,aValue,tPtOut::PCste(aPds));
}

template <class Type,const int  DimIn,const int DimOut> 
     void cLeastSqComputeMaps<Type,DimIn,DimOut>::AddObs
            (const tPtIn & aPtIn,const tPtOut & aValue)
{
     AddObs(aPtIn,aValue,1.0);
}


/* ===================================================== */
/* =====              INSTANTIATION                ===== */
/* ===================================================== */

#define INSTANTIATE_LSQMAP(TYPE,DIMIN,DIMOUT)\
template class cLeastSqComputeMaps<TYPE,DIMIN,DIMOUT>;

INSTANTIATE_LSQMAP(tREAL8,3,2)

/*
#define INSTANTIATE_OPMulMatVect(T1,T2)\
template  cDenseVect<T1> operator * (const cDenseVect<T1> & aVL,const cUnOptDenseMatrix<T2>& aMat);\
template  cDenseVect<T1> operator * (const cUnOptDenseMatrix<T2>& aVC,const cDenseVect<T1> & aMat);\
template  cDenseVect<T1> operator * (const cDenseVect<T1> & aVL,const cDenseMatrix<T2>& aMat);\
template  cDenseVect<T1> operator * (const cDenseMatrix<T2>& aVC,const cDenseVect<T1> & aMat);\


#define INSTANTIATE_DENSE_MATRICES(Type)\
template  class  cUnOptDenseMatrix<Type>;\
template  class  cDenseMatrix<Type>;\
template  cDenseMatrix<Type> operator * (const cDenseMatrix<Type> &,const cDenseMatrix<Type>&);\
template  cUnOptDenseMatrix<Type> operator * (const cUnOptDenseMatrix<Type> &,const cUnOptDenseMatrix<Type>&);\
INSTANTIATE_OPMulMatVect(Type,Type)\


INSTANTIATE_DENSE_MATRICES(tREAL4)
INSTANTIATE_DENSE_MATRICES(tREAL8)
INSTANTIATE_DENSE_MATRICES(tREAL16)
*/


};
