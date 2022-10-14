#include "include/MMVII_all.h"

namespace MMVII
{

template<class T> cPtxd<T,3>  PFromNumAxe(int aNum)
{
   static const cDenseMatrix<T> anId3x3(3,3,eModeInitImage::eMIA_MatrixId);
   return cPtxd<T,3>::Col(anId3x3,aNum);
}

template<class T> cDenseMatrix<T> MatFromCols(const cPtxd<T,3>&aP0,const cPtxd<T,3>&aP1,const cPtxd<T,3>&aP2)
{
   cDenseMatrix<T> aRes(3,3);

   SetCol(aRes,0,aP0);
   SetCol(aRes,1,aP1);
   SetCol(aRes,2,aP2);

   return aRes;
}

template<class T> cDenseMatrix<T> MatFromLines(const cPtxd<T,3>&aP0,const cPtxd<T,3>&aP1,const cPtxd<T,3>&aP2)
{
   cDenseMatrix<T> aRes(3,3);

   SetLine(0,aRes,aP0);
   SetLine(1,aRes,aP1);
   SetLine(2,aRes,aP2);

   return aRes;
}

/*
    (X1)   (X2)      Y1*Z2 - Z1*Y2     ( 0   -Z1    Y1)   (X2) 
    (Y1) ^ (Y2) =    Z1*X2 - X1*Z2  =  ( Z1    0   -X1) * (Y2)
    (Z1)   (Z2)      X1*Y2 - Y1*X2     (-Y1    X1    0)   (Z2)
 
*/

template<class T> cDenseMatrix<T> MatProdVect(const cPtxd<T,3>& W)
{
	return MatFromLines<T>
               (
	          cPtxd<T,3>(  0    , -W.z() ,  W.y() ),
	          cPtxd<T,3>( W.z() ,   0    , -W.x() ),
	          cPtxd<T,3>(-W.y() ,  W.x() ,   0    )
	       );
}


/*
template <class T>  cPtxd<T,3> operator ^ (const cPtxd<T,3> & aP1,const cPtxd<T,3> & aP2)
{
   return cPtxd<T,3>
          (
               aP1.y() * aP2.z() -aP1.z()*aP2.y(),
               aP1.z() * aP2.x() -aP1.x()*aP2.z(),
               aP1.x() * aP2.y() -aP1.y()*aP2.x()
          );
}
*/

template<class T> cPtxd<T,3>  VOrthog(const cPtxd<T,3> & aP)
{
   // we make a vect product with any vector, just avoid one too colinear  to P
   // test I and J, as P cannot be colinear to both, its sufficient 
   // (i.e : we are sur to maintain the biggest of x, y and z)
   if (std::abs(aP.x()) > std::abs(aP.y()))
      return cPtxd<T,3>( aP.z(), 0, -aP.x());

  return cPtxd<T,3>(0,aP.z(),-aP.y());
}



/* ========================== */
/*          ::                */
/* ========================== */

//template cPtxd<int,3>  operator ^ (const cPtxd<int,3> & aP1,const cPtxd<int,3> & aP2);
//template cPtxd<TYPE,3>  operator ^ (const cPtxd<TYPE,3> & aP1,const cPtxd<TYPE,3> & aP2);

#define MACRO_INSTATIATE_PTXD(TYPE)\
template cDenseMatrix<TYPE> MatProdVect(const cPtxd<TYPE,3>& W);\
template cDenseMatrix<TYPE> MatFromCols(const cPtxd<TYPE,3>&,const cPtxd<TYPE,3>&,const cPtxd<TYPE,3>&);\
template cDenseMatrix<TYPE> MatFromLines(const cPtxd<TYPE,3>&,const cPtxd<TYPE,3>&,const cPtxd<TYPE,3>&);\
template cPtxd<TYPE,3>  PFromNumAxe(int aNum);\
template cPtxd<TYPE,3>  VOrthog(const cPtxd<TYPE,3> & aP);


MACRO_INSTATIATE_PTXD(tREAL4)
MACRO_INSTATIATE_PTXD(tREAL8)
MACRO_INSTATIATE_PTXD(tREAL16)



};
