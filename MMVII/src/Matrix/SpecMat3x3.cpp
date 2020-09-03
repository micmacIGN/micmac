#include "include/MMVII_all.h"
#include "include/MMVII_Tpl_Images.h"

namespace MMVII
{

template <class Type,const int aSz> inline void AssertMnxn(cDenseMatrix<Type> & aM)
{
  MMVII_INTERNAL_ASSERT_tiny(aM.Sz()==cPt2di(aSz,aSz),"Bad size for matrix");
}

template <class Type> inline void AssertM3x3(cDenseMatrix<Type> & aM)
{
   AssertMnxn<Type,3>(aM);
}


};



