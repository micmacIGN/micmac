#include "include/MMVII_all.h"


#include "MMVII_EigenWrap.h"


using namespace Eigen;

namespace MMVII
{

/* ============================================= */
/*          cDenseMatrix<Type>                   */
/* ============================================= */



template <class Type>  void  cDenseMatrix<Type>::Show() const
{
    cConst_EigenMatWrap<Type> aMapThis(*this);
    StdOut() << aMapThis.EW() << "\n";
}


   //  ========= Mul and Inverse ========


template <class Type> void cDenseMatrix<Type>::MatMulInPlace(const tDM & aM1,const tDM & aM2)
{
    tMat::CheckSizeMulInPlace(aM1,aM2);
    cNC_EigenMatWrap<Type> aMapThis(*this);
    cConst_EigenMatWrap<Type> aMap1(aM1);
    cConst_EigenMatWrap<Type> aMap2(aM2);

    aMapThis.EW()  = aMap1.EW() * aMap2.EW();
}

template <class Type>  void  cDenseMatrix<Type>::InverseInPlace(const tDM & aM)
{
   tMat::CheckSquare(*this);
   DIm().AssertSameArea(aM.DIm());

   cConst_EigenMatWrap<Type> aMapThis(aM);
   cNC_EigenMatWrap<Type> aMapRes(*this);

   aMapRes.EW() = aMapThis.EW().inverse();
}

template <class Type> cDenseMatrix<Type>  cDenseMatrix<Type>::Inverse() const
{
   cDenseMatrix<Type> aRes(Sz().x(),Sz().y());
   aRes.InverseInPlace(*this);

   return aRes;
}


template <class Type> Type cDenseMatrix<Type>::Det() const
{
    this->CheckSquare(*this);
    if (Sz().x() ==1) 
    {
       return GetElem(0,0);
    }
    if (Sz().x() ==2)
    {
       return GetElem(0,0) *GetElem(1,1) -GetElem(0,1) *GetElem(1,0);
    }
    if (Sz().x() ==2)
    {
       return 
           GetElem(0,0) *(GetElem(1,1)*GetElem(2,2) -GetElem(1,2) *GetElem(2,1))
        +  GetElem(0,1) *(GetElem(1,2)*GetElem(2,0) -GetElem(1,0) *GetElem(2,2))
        +  GetElem(0,2) *(GetElem(1,0)*GetElem(2,1) -GetElem(1,1) *GetElem(2,0));
    }


    cConst_EigenMatWrap<Type> aMapThis(*this);
    return aMapThis.EW().determinant();
}


// ===============  Line ================

/**  Version where T1 != T2, cannot use Eigen, so do it by hand */
template <class T1,class T2> static
      void  TplMulLine(cDenseVect<T2> &aVRes,const cDenseMatrix<T1> & aMat, const cDenseVect<T2> & aVIn)
{
   aMat.TplCheckSizeYandX(aVIn,aVRes);

   for (int aX=0 ; aX<aMat.Sz().x() ; aX++)
   {
       typename tMergeF<T1,T2>::tMax aRes = 0.0; // Create a temporary having max accuracy  of T1/T2
       for (int aY=0 ; aY<aMat.Sz().y() ; aY++)
           aRes += aVIn(aY) * aMat.GetElem(aX,aY);
       aVRes(aX) =  aRes; // TplMulLineElem(aX,aVIn);
   }
}

/** Version with same type, use eigen */

template <class T> static  
        void  TplMulLine(cDenseVect<T> & aVRes,const cDenseMatrix<T> & aMat, const cDenseVect<T> & aVIn)
{
   aMat.TplCheckSizeYandX(aVIn,aVRes);

   cConst_EigenMatWrap<T> aMapM(aMat);
   cConst_EigenLineVectWrap<T> aMapI(aVIn);
   cNC_EigenLineVectWrap<T> aMapR(aVRes);

   aMapR.EW() = aMapI.EW() * aMapM.EW();
}

/** Mul Col X with line vector VIn */

template <class T1,class T2> static  typename tMergeF<T1,T2>::tMax TplMulLineElem(int aX,const cDenseMatrix<T1> & aMat, const cDenseVect<T2> & aVIn)
{
   aMat.TplCheckSizeY(aVIn);
   typename tMergeF<T1,T2>::tMax aRes = 0.0;
   for (int aY=0 ; aY<aMat.Sz().y() ; aY++)
       aRes += aVIn(aY) * aMat.GetElem(aX,aY);
   return aRes;
}


/*  MulLineInPlace / MulLineElem for REAL4, REAL8, REAL16 : call template functions */

template <class Type> void  cDenseMatrix<Type>::MulLineInPlace(tDV &aVRes,const tDV &aVIn) const 
{
   TplMulLine(aVRes,*this,aVIn);
}
template <class Type> Type  cDenseMatrix<Type>::MulLineElem(int  aX,const tDV & aVIn) const 
{
   return TplMulLineElem(aX,*this,aVIn);
}


// ===============  Column ================

    /* I cannot make eigen wrapper work on different type of matrix (float/double, ...) so
        create two template, on specialized with same type, using eigen, posibly more efficient ? (use //isation)
        and other using hand craft mult
    */


template <class T1,class T2> static
         void  TplMulCol(cDenseVect<T2> &aVRes,const cDenseMatrix<T1> & aMat, const cDenseVect<T2> & aVIn)
{
   aMat.TplCheckSizeYandX(aVRes,aVIn);
   /*  A conserver, verif Merge Type
       StdOut() << "Txxxxxx "  
                 << " " << E2Str(tNumTrait<T1>::TyNum())  
                 << " " << E2Str(tNumTrait<T2>::TyNum())  
                 << " => " << E2Str(tNumTrait<typename tMergeF<T1,T2>::tMax>::TyNum())  
                 << "\n";
   */

   for (int aY= 0 ; aY <aMat.Sz().y() ; aY++)
   {
       typename tMergeF<T1,T2>::tMax aRes = 0.0;
       for (int aX=0 ; aX<aMat.Sz().x() ; aX++)
           aRes += aVIn(aX) * aMat.GetElem(aX,aY);
       aVRes(aY) = aRes;
   }
}

    /** Mul Col,  With same type , use eigen */

template <class T> static  
        void  TplMulCol(cDenseVect<T> & aVRes,const cDenseMatrix<T> & aMat, const cDenseVect<T> & aVIn)
{
   aMat.TplCheckSizeYandX(aVRes,aVIn);
   
   cConst_EigenMatWrap<T>  aMapM(aMat);
   cConst_EigenColVectWrap<T> aMapI(aVIn);
   cNC_EigenColVectWrap<T> aMapR(aVRes);
   aMapR.EW() =  aMapM.EW() * aMapI.EW();
}

   /** Mul Line aY with col vector VIn */
template <class T1,class T2> static typename tMergeF<T1,T2>::tMax TplMulColElem(int aY,const cDenseMatrix<T1> & aMat, const cDenseVect<T2> & aVIn)
{
   aMat.TplCheckSizeX(aVIn);
   typename tMergeF<T1,T2>::tMax aRes = 0.0;
   for (int aX=0 ; aX<aMat.Sz().x() ; aX++)
       aRes += aVIn(aX) * aMat.GetElem(aX,aY);
   return aRes;
}


template <class Type> void  cDenseMatrix<Type>::MulColInPlace(tDV &aVRes,const tDV &aVIn) const 
{
   TplMulCol(aVRes,*this,aVIn);
}
template <class Type> Type  cDenseMatrix<Type>::MulColElem(int  aY,const tDV & aVIn) const 
{
   return TplMulColElem(aY,*this,aVIn);
}


/* ===================================================== */
/* =====              INSTANTIATION                ===== */
/* ===================================================== */


#define INSTANTIATE_DENSE_MATRICES(Type)\
template  class  cUnOptDenseMatrix<Type>;\
template  class  cDenseMatrix<Type>;\



INSTANTIATE_DENSE_MATRICES(tREAL4)
INSTANTIATE_DENSE_MATRICES(tREAL8)
INSTANTIATE_DENSE_MATRICES(tREAL16)




};
