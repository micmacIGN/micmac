#include "include/MMVII_all.h"
#include "include/MMVII_Tpl_Images.h"

// https://liris.cnrs.fr/page-membre/remi-ratajczak

namespace MMVII
{

/* ========================== */
/*       cSysSurResolu        */
/* ========================== */

template<class Type> cSysSurResolu<Type>::cSysSurResolu(int aNbVar) :
   mNbVar (aNbVar)
{
}

template<class Type> cSysSurResolu<Type>::~cSysSurResolu()
{
}

template<class Type> int cSysSurResolu<Type>::NbVar() const
{
   return mNbVar;
}

template<class Type> void cSysSurResolu<Type>::AddObsFixVar(const Type& aWeight,int aIndVal,const Type & aVal)
{
   cSparseVect<Type> aSpV;
   aSpV.AddIV(aIndVal,1.0);
   // static cIndSV<Type> & aIV = aSpV.IV()[0];
   // aIV.mInd  = aIndVal;
   // aIV.mVal  = 1.0;
   
   AddObservation(aWeight,aSpV,aVal);
}

template<class Type> void cSysSurResolu<Type>::AddObsFixVar(const Type& aWeight,const cSparseVect<Type> & aVVarVals)
{
    for (const auto & aP : aVVarVals)
        AddObsFixVar(aWeight,aP.mInd,aP.mVal);
}

template<class Type> void cSysSurResolu<Type>::AddObsFixVar (const Type& aWeight,const cDenseVect<Type>  &  aVRHS)
{
    MMVII_INTERNAL_ASSERT_medium(aVRHS.Sz() == mNbVar,"cSysSurResolu<Type>::AddObsFixVar");
    for (int aK=0 ; aK<mNbVar ; aK++)
        AddObsFixVar(aWeight,aK,aVRHS(aK));
}
/*
            ///  Fix value of curent variable, N variable
            ///  Fix value of curent variable, All variable
       virtual void AddObservation
*/

/* ======================= */
/*       cLeasSq           */
/* ======================= */


template<class Type>  cLeasSq<Type>::cLeasSq(int aNbVar):
     cSysSurResolu<Type> (aNbVar)
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


/* ========================== */
/*       cLeasSqtAA           */
/* ========================== */



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



/*
       /// Compute a solution
       cDenseVect<Type>  Solve() override;
*/



/* ===================================================== */
/* ===================================================== */
/* ===================================================== */

#define INSTANTIATE_LEASTSQ_TAA(Type)\
template  class  cLeasSqtAA<Type>;\
template  class  cLeasSq<Type>;\
template  class  cSysSurResolu<Type>;


INSTANTIATE_LEASTSQ_TAA(tREAL4)
INSTANTIATE_LEASTSQ_TAA(tREAL8)
INSTANTIATE_LEASTSQ_TAA(tREAL16)


};


/* ========================== */
/*          cMatrix           */
/* ========================== */

