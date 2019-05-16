#include "include/MMVII_all.h"


#include "MMVII_EigenWrap.h"


using namespace Eigen;

namespace MMVII
{


/* ============================================= */
/*      cDenseMatrix<Type>                   */
/* ============================================= */

template <class Type> cDenseMatrix<Type>::cDenseMatrix(int aX,int aY,eModeInitImage aMode) :
                             cUnOptDenseMatrix<Type>(aX,aY,aMode)
{
}

template <class Type> cDenseMatrix<Type>::cDenseMatrix(tIm anIm) :
                             cUnOptDenseMatrix<Type>(anIm)
{
}

template <class Type> cDenseMatrix<Type>::cDenseMatrix(int aXY,eModeInitImage aMode) :
                             cDenseMatrix<Type>(aXY,aXY,aMode)
{
}

template <class Type> cDenseMatrix<Type>  cDenseMatrix<Type>::Dup() const
{
    return  cDenseMatrix<Type>(Im().Dup());
}

template <class Type> cDenseMatrix<Type> cDenseMatrix<Type>::Diag(const cDenseVect<Type> & aV)
{
    cDenseMatrix<Type> aRes(aV.Sz(),eModeInitImage::eMIA_Null);
    for (int aK=0 ; aK<aV.Sz(); aK++)
        aRes.SetElem(aK,aK,aV(aK));

    return aRes;
}



template <class Type>  void  cDenseMatrix<Type>::Show() const
{
    cConst_EigenMatWrap<Type> aMapThis(*this);
    std::cout << aMapThis.EW() << "\n";
}



   //  ========= Mul and Inverse ========


template <class Type> void cDenseMatrix<Type>::MatMulInPlace(const tDM & aM1,const tDM & aM2)
{
    cNC_EigenMatWrap<Type> aMapThis(*this);
    cConst_EigenMatWrap<Type> aMap1(aM1);
    cConst_EigenMatWrap<Type> aMap2(aM2);

    aMapThis.EW()  = aMap1.EW() * aMap2.EW();
}

template <class Type> cDenseMatrix<Type>  cDenseMatrix<Type>::Inverse() const
{
   cMatrix::CheckSquare(*this);
   cDenseMatrix<Type> aRes(Sz().x(),Sz().y());

   cConst_EigenMatWrap<Type> aMapThis(*this);
   cNC_EigenMatWrap<Type> aMapRes(aRes);

   aMapRes.EW() = aMapThis.EW().inverse();

   return aRes;
}

/** Iterative inverse seem to be useless with eigen.  In the best case it divide by two the residual
    I let it hower just in case I have doubt.
*/
template <class Type> cDenseMatrix<Type>  cDenseMatrix<Type>::Inverse(double Eps,int aNbIter) const
{
   MMVII_INTERNAL_ASSERT_strong(false,"Inverse iterative not usefull");
   bool DoPertubate = false;
   //  A A' = I + E      A A-1 = I     A(A'-A-1)=E    A-1= A'- A-1 E
   //  A-1 ~ A'(I-E) = A' (2*I- AA')

   const cDenseMatrix<Type> & A = *this;
   cDenseMatrix<Type> Ap = A.Inverse();
   int aNb = Sz().x();

   {
      cDenseMatrix<Type> AAp = A * Ap;
      std::cout << "D000 " << AAp.DIm().L2Dist(cDenseMatrix<Type>(aNb,eModeInitImage::eMIA_MatrixId).DIm()) << "\n";

   // Test that pertubate the inverse, it's only for bench purpose, to be sure that
   // iterative algoritm work.
      if (DoPertubate)
      {
          for (int aX=0 ; aX<aNb ; aX++)
          {
             for (int aY=0 ; aY<aNb ; aY++)
             {
                  Ap.SetElem(aX,aY,Ap.GetElem(aX,aY)+ RandUnif_0_1()*1e-2/aNb);
             }
          }
      }
   }

   int aK = 0;
   while (aK<aNbIter)
   {
       cDenseMatrix<Type> AAp = A * Ap;
       cDenseMatrix<Type> ImE(aNb);
       double aSomEr = 0.0;
       for (int aX=0 ; aX<aNb ; aX++)
       {
          for (int aY=0 ; aY<aNb ; aY++)
          {
              Type VId = (aX==aY);
              const Type & aVAAp = AAp.GetElem(aX,aY);
              aSomEr += Square(aVAAp-VId);
              ImE.SetElem(aX,aY,2*VId-aVAAp);
          }
       }
       aSomEr = std::sqrt(aSomEr/R8Square(aNb));
       std::cout << "SOMM EE " << aSomEr << "\n";
       Ap = Ap * ImE;
       aK++;
   }
   getchar();
   return Ap;
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

template <class T1,class T2> static tMatrElem TplMulLineElem(int aX,const cDenseMatrix<T1> & aMat, const cDenseVect<T2> & aVIn)
{
   aMat.TplCheckSizeY(aVIn);
   typename tMergeF<T1,T2>::tMax aRes = 0.0;
   for (int aY=0 ; aY<aMat.Sz().y() ; aY++)
       aRes += aVIn(aY) * aMat.GetElem(aX,aY);
   return aRes;
}


/*  MulLineInPlace / MulLineElem for REAL4, REAL8, REAL16 : call template functions */

template <class Type> void  cDenseMatrix<Type>::MulLineInPlace(cDenseVect<tREAL4> &aVRes,const cDenseVect<tREAL4> &aVIn) const 
{
   TplMulLine(aVRes,*this,aVIn);
}
template <class Type> tMatrElem  cDenseMatrix<Type>::MulLineElem(int  aX,const cDenseVect<tREAL4> & aVIn) const 
{
   return TplMulLineElem(aX,*this,aVIn);
}

template <class Type> void  cDenseMatrix<Type>::MulLineInPlace(cDenseVect<tREAL8> &aVRes,const cDenseVect<tREAL8> &aVIn) const 
{
   TplMulLine(aVRes,*this,aVIn);
}
template <class Type> tMatrElem  cDenseMatrix<Type>::MulLineElem(int  aX,const cDenseVect<tREAL8> & aVIn) const 
{
   return TplMulLineElem(aX,*this,aVIn);
}
template <class Type> void  cDenseMatrix<Type>::MulLineInPlace(cDenseVect<tREAL16> &aVRes,const cDenseVect<tREAL16> &aVIn) const 
{
   TplMulLine(aVRes,*this,aVIn);
}
template <class Type> tMatrElem  cDenseMatrix<Type>::MulLineElem(int  aX,const cDenseVect<tREAL16> & aVIn) const 
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
       std::cout << "Txxxxxx "  
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
template <class T1,class T2> static tMatrElem TplMulColElem(int aY,const cDenseMatrix<T1> & aMat, const cDenseVect<T2> & aVIn)
{
   aMat.TplCheckSizeX(aVIn);
   typename tMergeF<T1,T2>::tMax aRes = 0.0;
   for (int aX=0 ; aX<aMat.Sz().x() ; aX++)
       aRes += aVIn(aX) * aMat.GetElem(aX,aY);
   return aRes;
}


template <class Type> void  cDenseMatrix<Type>::MulColInPlace(cDenseVect<tREAL4> &aVRes,const cDenseVect<tREAL4> &aVIn) const 
{
   TplMulCol(aVRes,*this,aVIn);
}
template <class Type> tMatrElem  cDenseMatrix<Type>::MulColElem(int  aY,const cDenseVect<tREAL4> & aVIn) const 
{
   return TplMulColElem(aY,*this,aVIn);
}

template <class Type> void  cDenseMatrix<Type>::MulColInPlace(cDenseVect<tREAL8> &aVRes,const cDenseVect<tREAL8> &aVIn) const 
{
   TplMulCol(aVRes,*this,aVIn);
}
template <class Type> tMatrElem  cDenseMatrix<Type>::MulColElem(int  aY,const cDenseVect<tREAL8> & aVIn) const 
{
   return TplMulColElem(aY,*this,aVIn);
}


template <class Type> void  cDenseMatrix<Type>::MulColInPlace(cDenseVect<tREAL16> &aVRes,const cDenseVect<tREAL16> &aVIn) const 
{
   TplMulCol(aVRes,*this,aVIn);
}
template <class Type> tMatrElem  cDenseMatrix<Type>::MulColElem(int  aY,const cDenseVect<tREAL16> & aVIn) const 
{
   return TplMulColElem(aY,*this,aVIn);
}

template <class T1,class T2> cDenseVect<T1> operator * (const cDenseVect<T1> & aVL,const cDenseMatrix<T2>& aMat)
{
   return aMat.MulLine(aVL);
}
template <class T1,class T2> cDenseVect<T1> operator * (const cDenseMatrix<T2>& aMat,const cDenseVect<T1> & aVC)
{
   return aMat.MulCol(aVC);
}

template <class Type> cDenseMatrix<Type> operator * (const cDenseMatrix<Type> & aM1,const cDenseMatrix<Type>& aM2)
{
   cDenseMatrix<Type> aRes(aM2.Sz().x(),aM1.Sz().y());
   aRes.MatMulInPlace(aM1,aM2);
   return aRes;
}


/* ================================================= */
/*        cUnOptDenseMatrix                          */
/* ================================================= */

template <class Type> cUnOptDenseMatrix<Type>::cUnOptDenseMatrix(tIm anIm) :
                           cMatrix(anIm.DIm().Sz().x(),anIm.DIm().Sz().y()),
                           mIm(anIm)
{
   MMVII_INTERNAL_ASSERT_strong(anIm.DIm().P0()==cPt2di(0,0),"Init Matrix P0!= 0,0");
}

template <class Type> cUnOptDenseMatrix<Type>::cUnOptDenseMatrix(int aX,int aY,eModeInitImage aMode) :
      cUnOptDenseMatrix<Type>(tIm(cPt2di(aX,aY),nullptr,aMode))
{
}

template <class Type> tMatrElem cUnOptDenseMatrix<Type>::V_GetElem(int aX,int  aY) const 
{
    return GetElem(aX,aY);
}

template <class Type> void cUnOptDenseMatrix<Type>::V_SetElem(int aX,int  aY,const tMatrElem & aV) 
{
    SetElem(aX,aY,aV);
}


template <class Type> eTyNums cUnOptDenseMatrix<Type>::TypeNum() const 
{
    return tNumTrait<Type>::TyNum();
}

     
template <class T1,class T2> cDenseVect<T1> operator * (const cDenseVect<T1> & aVL,const cUnOptDenseMatrix<T2>& aMat)
{
   return aMat.MulLine(aVL);
}
template <class T1,class T2> cDenseVect<T1> operator * (const cUnOptDenseMatrix<T2>& aMat,const cDenseVect<T1> & aVC)
{
   return aMat.MulCol(aVC);
}

template <class Type> cUnOptDenseMatrix<Type> operator * (const cUnOptDenseMatrix<Type> & aM1,const cUnOptDenseMatrix<Type>& aM2)
{
   cUnOptDenseMatrix<Type> aRes(aM2.Sz().x(),aM1.Sz().y());
   aRes.MatMulInPlace(aM1,aM2);
   return aRes;
}


/* ===================================================== */
/* =====              INSTANTIATION                ===== */
/* ===================================================== */

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
INSTANTIATE_OPMulMatVect(Type,tREAL4)\
INSTANTIATE_OPMulMatVect(Type,tREAL8)\
INSTANTIATE_OPMulMatVect(Type,tREAL16)\


INSTANTIATE_DENSE_MATRICES(tREAL4)
INSTANTIATE_DENSE_MATRICES(tREAL8)
INSTANTIATE_DENSE_MATRICES(tREAL16)




};
