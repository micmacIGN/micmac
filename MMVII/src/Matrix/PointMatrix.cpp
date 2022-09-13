#include "include/MMVII_all.h"

namespace MMVII
{

template <class Type,const int Dim> cPtxd<Type,Dim> cPtxd<Type,Dim>::Col(const cDenseMatrix<Type>& aMat,int aX) 
{
   cPtxd<Type,Dim> aRes;
   CHECK_SZMAT_COL(aMat,aRes);
   for (int aY=0 ; aY<Dim ; aY++)
        aRes.mCoords[aY]  = aMat.GetElem(aX,aY);

   return aRes;
}

template <class Type,const int Dim> cPtxd<Type,Dim> cPtxd<Type,Dim>::Line(int aY,const cDenseMatrix<Type>& aMat) 
{
   cPtxd<Type,Dim> aRes;
   CHECK_SZMAT_LINE(aMat,aRes);
   for (int aX=0 ; aX<Dim ; aX++)
        aRes.mCoords[aX]  = aMat.GetElem(aX,aY);

   return aRes;
}

template <class Type,const int Dim> cPtxd<Type,Dim> cPtxd<Type,Dim>::FromVect(const cDenseVect<Type>& aV) 
{
   cPtxd<Type,Dim> aRes;
   CHECK_SZPT_VECT(aV,aRes);
   for (int aK=0 ; aK<Dim ; aK++)
        aRes.mCoords[aK]  = aV(aK);

   return aRes;
}

/*
template  cPtxd<TYPE,DIM> cPtxd<TYPE,DIM>::FromStdVector(const std::vector<TYPE>& aV);\
template <class Type,const int Dim> cPtxd<Type,Dim> cPtxd<Type,Dim>::FromStdVector(const std::vector<Type>& aV) 
{
   cPtxd<Type,Dim> aRes;
   MMVII_INTERNAL_ASSERT_tiny(aV.size()==Dim,"Bad size in Vec/Pt");
   for (int aK=0 ; aK<Dim ; aK++)
        aRes.mCoords[aK]  = aV.at(aK);

   return aRes;
}
*/

template <class Type,const int Dim> cDenseVect<Type> cPtxd<Type,Dim>::ToVect() const
{
   cDenseVect<Type> aRes(Dim);
   for (int aK=0 ; aK<Dim ; aK++)
        aRes(aK) = mCoords[aK];

   return aRes;
}

template <class Type,const int Dim> std::vector<Type> cPtxd<Type,Dim>::ToStdVector() const
{
   std::vector<Type> aRes;
   for (int aK=0 ; aK<Dim ; aK++)
        aRes.push_back(mCoords[aK]);

   return aRes;
}



// X => Col ;;   Y => Line

template <class Type,const int Dim> 
   void GetCol(cPtxd<Type,Dim> & aPt,const cDenseMatrix<Type> & aMat,int aX)
{
   aPt = cPtxd<Type,Dim>::Col(aMat,aX);
}
template <class Type,const int Dim> 
   void SetCol(cDenseMatrix<Type> & aMat,int aX,const cPtxd<Type,Dim> & aPt)
{
   CHECK_SZMAT_COL(aMat,aPt);
   for (int aY=0 ; aY<aMat.Sz().y() ; aY++)
        aMat.SetElem(aX,aY,aPt[aY]);
}

template <class Type,const int Dim> 
   void GetLine(cPtxd<Type,Dim> & aPt,int aY,const cDenseMatrix<Type> & aMat)
{
   aPt = cPtxd<Type,Dim>::Line(aY,aMat);
}
template <class Type,const int Dim> 
   void SetLine(int aY,cDenseMatrix<Type> & aMat,const cPtxd<Type,Dim> & aPt)
{
   CHECK_SZMAT_LINE(aMat,aPt);
   for (int aX=0 ; aX<aMat.Sz().x() ; aX++)
        aMat.SetElem(aX,aY,aPt[aX]);
}


template <class Type,const int DimOut,const int DimIn> 
void MulCol(cPtxd<Type,DimOut>&aPOut,const cDenseMatrix<Type>&aMat,const cPtxd<Type,DimIn>&aPIn)
{
   CHECK_SZMAT_COL(aMat,aPOut);
   CHECK_SZMAT_LINE(aMat,aPIn);
   for (int aY=0 ; aY<aMat.Sz().y() ; aY++)
   {
       Type aSom = 0.0;
       for (int aX=0 ; aX<aMat.Sz().x() ; aX++)
       {
           aSom += aMat.GetElem(aX,aY) * aPIn[aX];
       }
       aPOut[aY] = aSom;
   }
}
template <class Type,const int Dim> 
   cPtxd<Type,Dim> operator * (const cDenseMatrix<Type> & aMat,const cPtxd<Type,Dim> & aPIn)
{
    cPtxd<Type,Dim> aRes;
    MulCol(aRes,aMat,aPIn);
    return aRes;
}

template <class Type,const int DimOut,const int DimIn> 
void MulLine(cPtxd<Type,DimOut>&aPOut,const cPtxd<Type,DimIn>&aPIn,const cDenseMatrix<Type>&aMat)
{
   CHECK_SZMAT_LINE(aMat,aPOut);
   CHECK_SZMAT_COL(aMat,aPIn);
   for (int aX=0 ; aX<aMat.Sz().x() ; aX++)
   {
       Type aSom = 0.0;
       for (int aY=0 ; aY<aMat.Sz().y() ; aY++)
       {
           aSom += aMat.GetElem(aX,aY) * aPIn[aY];
       }
       aPOut[aX] = aSom;
   }
}

template <class Type,const int Dim> 
   cPtxd<Type,Dim> operator * (const cPtxd<Type,Dim> & aPIn,const cDenseMatrix<Type> & aMat)
{
    cPtxd<Type,Dim> aRes;
    MulLine(aRes,aPIn,aMat);
    return aRes;
}

template<class Type,const int Dim>
    cPtxd<Type,Dim> SolveCol(const cDenseMatrix<Type>& aMat,const cPtxd<Type,Dim>& aPCol)
{
    cDenseVect<Type> aVCol = aPCol.ToVect();
    cDenseVect<Type> aVRes = aMat.SolveColumn(aVCol);
    return  cPtxd<Type,Dim>::FromVect(aVRes);
}
template<class Type,const int Dim>
    cPtxd<Type,Dim> SolveLine(const cPtxd<Type,Dim>& aPLine,const cDenseMatrix<Type>& aMat)
{
    cDenseVect<Type> aVLine = aPLine.ToVect();
    cDenseVect<Type> aVRes = aMat.SolveLine(aVLine);
    return  cPtxd<Type,Dim>::FromVect(aVRes);
}



#define INSTANT_MUL_MATVECT(TYPE,DIMOUT,DIMIN)\
template void MulCol(cPtxd<TYPE,DIMOUT>&,const cDenseMatrix<TYPE>&,const cPtxd<TYPE,DIMIN>&);\
template void MulLine(cPtxd<TYPE,DIMOUT>&,const cPtxd<TYPE,DIMIN>&,const cDenseMatrix<TYPE>&);

#define INSTANT_PT_MAT_int_DIM(DIM)\
template  cDenseVect<int> cPtxd<int,DIM>::ToVect() const;\
template  cPtxd<int,DIM> cPtxd<int,DIM>::FromVect(const cDenseVect<int>& aV);

INSTANT_PT_MAT_int_DIM(1)
INSTANT_PT_MAT_int_DIM(2)
INSTANT_PT_MAT_int_DIM(3)
INSTANT_PT_MAT_int_DIM(4)


#define INSTANT_PT_MAT_TYPE_DIM(TYPE,DIM)\
template cPtxd<TYPE,DIM> SolveCol(const cDenseMatrix<TYPE>&,const cPtxd<TYPE,DIM>&);\
template cPtxd<TYPE,DIM> SolveLine(const cPtxd<TYPE,DIM>&,const cDenseMatrix<TYPE>&);\
template  cDenseVect<TYPE> cPtxd<TYPE,DIM>::ToVect() const;\
template  std::vector<TYPE> cPtxd<TYPE,DIM>::ToStdVector() const;\
template  cPtxd<TYPE,DIM> cPtxd<TYPE,DIM>::FromVect(const cDenseVect<TYPE>& aV);\
template  cPtxd<TYPE,DIM> cPtxd<TYPE,DIM>::Col(const cDenseMatrix<TYPE> &,int);\
template  cPtxd<TYPE,DIM> cPtxd<TYPE,DIM>::Line(int,const cDenseMatrix<TYPE> &);\
INSTANT_MUL_MATVECT(TYPE,DIM,1);\
INSTANT_MUL_MATVECT(TYPE,DIM,2);\
INSTANT_MUL_MATVECT(TYPE,DIM,3);\
INSTANT_MUL_MATVECT(TYPE,DIM,4);\
template   cPtxd<TYPE,DIM> operator * (const cDenseMatrix<TYPE> &,const cPtxd<TYPE,DIM> &);\
template   cPtxd<TYPE,DIM> operator * (const cPtxd<TYPE,DIM> &,const cDenseMatrix<TYPE> &);\
template  void GetCol (cPtxd<TYPE,DIM> & aPt,const cDenseMatrix<TYPE> & aMat,int aCol);\
template  void SetCol (cDenseMatrix<TYPE> & aMat,int aX,const cPtxd<TYPE,DIM> & aPt);\
template  void GetLine(cPtxd<TYPE,DIM> & ,int,const cDenseMatrix<TYPE> & aMat);\
template  void SetLine(int,cDenseMatrix<TYPE> & ,const cPtxd<TYPE,DIM> & );


#define INSTANT_PT_MAT_TYPE(TYPE)\
INSTANT_PT_MAT_TYPE_DIM(TYPE,1)\
INSTANT_PT_MAT_TYPE_DIM(TYPE,2)\
INSTANT_PT_MAT_TYPE_DIM(TYPE,3)\
INSTANT_PT_MAT_TYPE_DIM(TYPE,4)


INSTANT_PT_MAT_TYPE(tREAL4)
INSTANT_PT_MAT_TYPE(tREAL8)
INSTANT_PT_MAT_TYPE(tREAL16)



};



