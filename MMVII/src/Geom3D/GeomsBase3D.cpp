#include "MMVII_Matrix.h"
#include "MMVII_Geom3D.h"

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

template <class T>  T  TetraReg (const cPtxd<T,3> & aP1,const cPtxd<T,3> & aP2,const cPtxd<T,3> & aP3,const T& FactEps)
{
     cPtxd<T,3> aCDG = (aP1+aP2+aP3) /static_cast<T>(3.0);

     T aSqDist = (SqN2(aCDG) + SqN2(aCDG-aP1) + SqN2(aCDG-aP2) + SqN2(aCDG-aP3)) / static_cast<T>(4.0);
     T aCoeffNorm = std::pow(aSqDist,3.0/2.0);  // 1/2 for D2  3->volume

     return Determinant(aP1,aP2,aP3) / (aCoeffNorm +  std::numeric_limits<T>::epsilon()*FactEps);
}

template <class T>  T  TetraReg (const cPtxd<T,3> & aP1,const cPtxd<T,3> & aP2,const cPtxd<T,3> & aP3,const cPtxd<T,3> & aP4,const T& FactEps)
{
    return TetraReg(aP2-aP1,aP3-aP1,aP4-aP1,FactEps);
}

template <class T>  T  Determinant (const cPtxd<T,3> & aP1,const cPtxd<T,3> & aP2,const cPtxd<T,3> & aP3)
{
	return Scal(aP1,aP2^aP3);
}

template<class Type>  cTriangle<Type,3> RandomTriang(Type aAmpl)
{
      return cTriangle<Type,3>(cPtxd<Type,3>::PRandC()*aAmpl,cPtxd<Type,3>::PRandC()*aAmpl,cPtxd<Type,3>::PRandC()*aAmpl);
}

template<class Type>  cTriangle<Type,3> RandomTriangRegul(Type aRegulMin,Type aAmpl)
{
    for (;;)
    {
        cTriangle<Type,3> aT = RandomTriang(aAmpl);
	if (aT.Regularity()> aRegulMin)
           return aT;
    }
    return RandomTriang(static_cast<Type>(0.0)); // Not sur its mandatory to have a return here
}

template<class Type>  cTriangle<Type,3> RandomTetraTriangRegul(Type aRegulMin,Type aAmpl)
{
    for (;;)
    {
        cTriangle<Type,3> aT = RandomTriang(aAmpl);
	if (TetraReg(aT.Pt(0),aT.Pt(1),aT.Pt(2)) > aRegulMin)
           return aT;
    }
    return RandomTriang(static_cast<Type>(0.0)); // Not sur its mandatory to have a return here
}




/* ========================== */
/*          ::                */
/* ========================== */

//template cPtxd<int,3>  operator ^ (const cPtxd<int,3> & aP1,const cPtxd<int,3> & aP2);
//template cPtxd<TYPE,3>  operator ^ (const cPtxd<TYPE,3> & aP1,const cPtxd<TYPE,3> & aP2);

#define MACRO_INSTATIATE_PTXD(TYPE)\
template  cTriangle<TYPE,3> RandomTriang(TYPE aRegulMin);\
template  cTriangle<TYPE,3> RandomTriangRegul(TYPE aRegulMin,TYPE aAmpl);\
template  cTriangle<TYPE,3> RandomTetraTriangRegul(TYPE aRegulMin,TYPE aAmpl);\
template TYPE  Determinant (const cPtxd<TYPE,3> & aP1,const cPtxd<TYPE,3> & aP2,const cPtxd<TYPE,3> & aP3);\
template TYPE  TetraReg (const cPtxd<TYPE,3> & aP1,const cPtxd<TYPE,3> & aP2,const cPtxd<TYPE,3> & aP3,const TYPE&);\
template TYPE  TetraReg (const cPtxd<TYPE,3> & aP1,const cPtxd<TYPE,3> & aP2,const cPtxd<TYPE,3> & aP3,const cPtxd<TYPE,3> & aP4,const TYPE&);\
template cDenseMatrix<TYPE> MatProdVect(const cPtxd<TYPE,3>& W);\
template cDenseMatrix<TYPE> MatFromCols(const cPtxd<TYPE,3>&,const cPtxd<TYPE,3>&,const cPtxd<TYPE,3>&);\
template cDenseMatrix<TYPE> MatFromLines(const cPtxd<TYPE,3>&,const cPtxd<TYPE,3>&,const cPtxd<TYPE,3>&);\
template cPtxd<TYPE,3>  PFromNumAxe(int aNum);\
template cPtxd<TYPE,3>  VOrthog(const cPtxd<TYPE,3> & aP);


MACRO_INSTATIATE_PTXD(tREAL4)
MACRO_INSTATIATE_PTXD(tREAL8)
MACRO_INSTATIATE_PTXD(tREAL16)



};
