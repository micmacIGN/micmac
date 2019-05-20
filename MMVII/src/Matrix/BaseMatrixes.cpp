#include "include/MMVII_all.h"

namespace MMVII
{


/* ========================== */
/*          cDenseVect        */
/* ========================== */

template <class Type> cDenseVect<Type>::cDenseVect(int aSz,Type * aDataLin) :
   mIm  (aSz,aDataLin)
{
}

/* ========================== */
/*          cDataMatrix       */
/* ========================== */
cDataMatrix::cDataMatrix(int aX,int aY) :
   cRect2(cPt2di(0,0),cPt2di(aX,aY))
{
}

template <class Type> void cDataMatrix::TplCheckSizeCol(const cDenseVect<Type> & aV) const
{
   MMVII_INTERNAL_ASSERT_strong(Sz().y()== aV.Sz(),"Bad size for column multiplication")
}
template <class Type> void cDataMatrix::TplCheckSizeLine(const cDenseVect<Type> & aV) const
{
   MMVII_INTERNAL_ASSERT_strong(Sz().x()== aV.Sz(),"Bad size for column multiplication")
}

         // Mul Col

template <class Type> void cDataMatrix::TplMulCol(cDenseVect<Type> & aVOut,const cDenseVect<Type> & aVIn)  const
{
   TplCheckSizeCol(aVOut);
   TplCheckSizeLine(aVIn);

   if (&aVOut==&aVIn) // Will see later if we handle this case
   {
       MMVII_INTERNAL_ASSERT_strong(false,"Aliasing in TplMulCol")
   }

   for (int aY=0 ; aY<Sz().y() ; aY++)
   {
       aVOut(aY) = TplMulCol(aY,aVIn);
   }
}

template <class Type> tMatrElem cDataMatrix::TplMulCol(int aY,const cDenseVect<Type> & aVIn)  const
{
    tMatrElem aRes = 0.0;
    for (int aX=0 ; aX<Sz().x() ; aX++)
        aRes += aVIn(aX) * GetElem(aX,aY);

    return aRes;
}

         // Mul Line

template <class Type> void cDataMatrix::TplMulLine(cDenseVect<Type> & aVOut,const cDenseVect<Type> & aVIn)  const
{
   TplCheckSizeLine(aVOut);
   TplCheckSizeCol(aVIn);

   if (&aVOut==&aVIn) // Will see later if we handle this case
   {
       MMVII_INTERNAL_ASSERT_strong(false,"Aliasing in TplMulCol")
   }

   for (int aX=0 ; aX<Sz().x() ; aX++)
   {
// std::cout << "rrrrrrr\n";
       aVOut(aX) = TplMulLine(aX,aVIn);
// std::cout << "uuuu\n";
   }
}

template <class Type> tMatrElem cDataMatrix::TplMulLine(int aX,const cDenseVect<Type> & aVIn)  const
{
    tMatrElem aRes = 0.0;
    for (int aY=0 ; aY<Sz().y() ; aY++)
        aRes += aVIn(aY) * GetElem(aX,aY);

    return aRes;
}



     // Virtuals tREAL4

void cDataMatrix::MulCol(cDenseVect<tREAL4> & aOut,const cDenseVect<tREAL4> & aIn) const
{
    TplMulCol(aOut,aIn);
}
tMatrElem cDataMatrix::MulCol(int aY,const cDenseVect<tREAL4> & aIn) const
{
    return TplMulCol(aY,aIn);
}
void cDataMatrix::MulLine(cDenseVect<tREAL4> & aOut,const cDenseVect<tREAL4> & aIn) const
{
    TplMulLine(aOut,aIn);
}
tMatrElem cDataMatrix::MulLine(int aX,const cDenseVect<tREAL4> & aIn) const
{
    return TplMulLine(aX,aIn);
}


     // Virtuals tREAL8
void cDataMatrix::MulCol(cDenseVect<tREAL8> & aOut,const cDenseVect<tREAL8> & aIn) const
{
    TplMulCol(aOut,aIn);
}
tMatrElem cDataMatrix::MulCol( int aY,const cDenseVect<tREAL8> & aIn) const
{
    return TplMulCol(aY,aIn);
}
void cDataMatrix::MulLine(cDenseVect<tREAL8> & aOut,const cDenseVect<tREAL8> & aIn) const
{
    TplMulLine(aOut,aIn);
}
tMatrElem cDataMatrix::MulLine(int aX,const cDenseVect<tREAL8> & aIn) const
{
    return TplMulLine(aX,aIn);
}

     // Virtuals tREAL16
void cDataMatrix::MulCol(cDenseVect<tREAL16> & aOut,const cDenseVect<tREAL16> & aIn) const
{
    TplMulCol(aOut,aIn);
}
tMatrElem cDataMatrix::MulCol(int aY,const cDenseVect<tREAL16> & aIn) const
{
    return TplMulCol(aY,aIn);
}
void cDataMatrix::MulLine(cDenseVect<tREAL16> & aOut,const cDenseVect<tREAL16> & aIn) const
{
    TplMulLine(aOut,aIn);
}
tMatrElem cDataMatrix::MulLine(int aX,const cDenseVect<tREAL16> & aIn) const
{
    return TplMulLine(aX,aIn);
}

/* ===================================================== */
/* ===================================================== */
/* ===================================================== */

#define INSTANTIATE_BASE_MATRICES(Type)\
template  class  cDenseVect<Type>;\


INSTANTIATE_BASE_MATRICES(tREAL4)
INSTANTIATE_BASE_MATRICES(tREAL8)
INSTANTIATE_BASE_MATRICES(tREAL16)



};
