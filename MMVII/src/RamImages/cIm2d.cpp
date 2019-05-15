#include "include/MMVII_all.h"

namespace MMVII
{


/* ========================== */
/*          cDataIm2D         */
/* ========================== */


template <class Type>  cDataIm2D<Type>::cDataIm2D(const cPt2di & aP0,const cPt2di & aP1,Type * aRawDataLin,eModeInitImage aModeInit) : 
    cDataTypedIm<Type,2> (aP0,aP1,aRawDataLin,aModeInit)
{
    mRawData2D = cMemManager::Alloc<tVal*>(cRectObj<2>::Sz().y()) -Y0();
    for (int aY=Y0() ; aY<Y1() ; aY++)
        mRawData2D[aY] = tBI::mRawDataLin + (aY-Y0()) * SzX() - X0();
}

template <class Type>  cDataIm2D<Type>::~cDataIm2D()
{
   cMemManager::Free(mRawData2D+Y0());
}

template <class Type> int     cDataIm2D<Type>::VI_GetV(const cPt2di& aP)  const
{
   return GetV(aP);
}
template <class Type> double  cDataIm2D<Type>::VD_GetV(const cPt2di& aP)  const 
{
   return GetV(aP);
}

template <class Type> void  cDataIm2D<Type>::VI_SetV(const cPt2di& aP,const int & aV)
{
   SetVTrunc(aP,aV);
}
template <class Type> void  cDataIm2D<Type>::VD_SetV(const cPt2di& aP,const double & aV)
{
   SetVTrunc(aP,aV);
}


/* ========================== */
/*          cIm2D         */
/* ========================== */

template <class Type>  cIm2D<Type>::cIm2D(const cPt2di & aP0,const cPt2di & aP1,Type * aRawDataLin,eModeInitImage aModeInit) :
   mSPtr(new cDataIm2D<Type>(aP0,aP1,aRawDataLin,aModeInit)),
   mPIm (mSPtr.get())
{
}

template <class Type>  cIm2D<Type>::cIm2D(const cPt2di & aSz,Type * aRawDataLin,eModeInitImage aModeInit) :
   cIm2D<Type> (cPt2di(0,0),aSz,aRawDataLin,aModeInit)
{
}

template <class Type>  cIm2D<Type> cIm2D<Type>::FromFile(const std::string & aName)
{
   cDataFileIm2D  aFileIm = cDataFileIm2D::Create(aName);
   cIm2D<Type> aRes(aFileIm.Sz());
   aRes.Read(aFileIm,cPt2di(0,0));

   return aRes;
}

template <class Type>  cIm2D<Type>  cIm2D<Type>::Dup() const
{
   cIm2D<Type> aRes(DIm().P0(),DIm().P1());
   DIm().DupIn(aRes.DIm());
   return aRes;
}



#define INSTANTIATE_IM2D(Type)\
template  class cIm2D<Type>;\
template  class cDataIm2D<Type>;

INSTANTIATE_IM2D(tINT1)
INSTANTIATE_IM2D(tINT2)
INSTANTIATE_IM2D(tINT4)

INSTANTIATE_IM2D(tU_INT1)
INSTANTIATE_IM2D(tU_INT2)
INSTANTIATE_IM2D(tU_INT4)


INSTANTIATE_IM2D(tREAL4)
INSTANTIATE_IM2D(tREAL8)
INSTANTIATE_IM2D(tREAL16)



};
