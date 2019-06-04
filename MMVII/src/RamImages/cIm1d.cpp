#include "include/MMVII_all.h"

namespace MMVII
{


/* ========================== */
/*          cDataIm2D         */
/* ========================== */

template <class Type>  void cDataIm1D<Type>::PostInit()
{
    mRawData1D = tBI::mRawDataLin - X0();
}

template <class Type>  cDataIm1D<Type>::cDataIm1D
                       (
                            const cPt1di & aP0,
                            const cPt1di & aP1,
                            Type * aRawDataLin,
                            eModeInitImage aModeInit
                       ) : 
    cDataTypedIm<Type,1> (aP0,aP1,aRawDataLin,aModeInit)
{
    PostInit();
}

template <class Type>  void cDataIm1D<Type>::Resize ( const cPt1di & aP0, const cPt1di & aP1, eModeInitImage aModeInit)  
{
   cDataTypedIm<Type,1>::Resize(aP0,aP1,aModeInit);
   PostInit();
}

template <class Type>  cDataIm1D<Type>::~cDataIm1D()
{
}

template <class Type> int     cDataIm1D<Type>::VI_GetV(const cPt1di& aP)  const
{
   return GetV(aP);
}
template <class Type> double  cDataIm1D<Type>::VD_GetV(const cPt1di& aP)  const 
{
   return GetV(aP);
}

template <class Type> void  cDataIm1D<Type>::VI_SetV(const cPt1di& aP,const int & aV)
{
   SetVTrunc(aP,aV);
}
template <class Type> void  cDataIm1D<Type>::VD_SetV(const cPt1di& aP,const double & aV)
{
   SetVTrunc(aP,aV);
}


/* ========================== */
/*          cIm1D         */
/* ========================== */

template <class Type>  cIm1D<Type>::cIm1D(const int & aP0,const int & aP1,Type * aRawDataLin,eModeInitImage aModeInit) :
   mSPtr(new cDataIm1D<Type>(cPt1di(aP0),cPt1di(aP1),aRawDataLin,aModeInit)),
   mPIm (mSPtr.get())
{
}

template <class Type>  cIm1D<Type>::cIm1D(const int & aSz,Type * aRawDataLin,eModeInitImage aModeInit) :
   cIm1D<Type> (0,aSz,aRawDataLin,aModeInit)
{
}

template <class Type>  cIm1D<Type>  cIm1D<Type>::Dup() const
{
   cIm1D<Type> aRes(DIm().P0().x(),DIm().P1().x());
   DIm().DupIn(aRes.DIm());
   return aRes;
}

#define INSTANTIATE_IM1D(Type)\
template  class cIm1D<Type>;\
template  class cDataIm1D<Type>;

INSTANTIATE_IM1D(tINT1)
INSTANTIATE_IM1D(tINT2)
INSTANTIATE_IM1D(tINT4)

INSTANTIATE_IM1D(tU_INT1)
INSTANTIATE_IM1D(tU_INT2)
INSTANTIATE_IM1D(tU_INT4)


INSTANTIATE_IM1D(tREAL4)
INSTANTIATE_IM1D(tREAL8)
INSTANTIATE_IM1D(tREAL16)



};
