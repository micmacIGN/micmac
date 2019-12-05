#include "include/MMVII_all.h"


#include "MMVII_EigenWrap.h"


using namespace Eigen;

namespace MMVII
{
/*  Static => see note on BaseMatrixes.cpp
*/


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


template <class Type> cDenseMatrix<Type>  cDenseMatrix<Type>::RandomSquareRegMatrix
                    (
                        int    aSz,
                        bool   IsSym,
                        double AmplAcc,
                        double aCondMinAccept
                    )
{
    cDenseMatrix<Type>  aRes(aSz,eModeInitImage::eMIA_RandCenter);
    cResulSymEigenValue<Type> aRSEV(1);
    if (IsSym)
    {
       aRes.SelfSymetrize();
       aRSEV = aRes.SymEigenValue();
    }
    else
    {
       // Still need to write non self adjoint interface to Eigen ....
       MMVII_INTERNAL_ASSERT_always(false,"RandomSquareRegMatrix do not handle unsym");
    }

    const cDenseVect<Type> &  aVEV =  aRSEV.EigenValues();

    // Set global amplitude 
    {
       Type aSom = 0;  // Average of eigen value
       for (int aK=0 ; aK<aSz  ; aK++)
       {
          aSom += std::abs(aVEV(aK));
       }
       aSom /= aSz;
       if (aSom<AmplAcc)
       {
          if (aSom==0)  // Case 0 , put any value non degenerate
          {
             for (int aK=0 ; aK<aSz  ; aK++)
             {
                aRSEV.SetKthEigenValue(aK,(1+aK)*(HeadOrTail() ? -1 : 1) );
             }
          }
          else // else multiply to have the given amplitude
          {
             double aMul = AmplAcc/aSom;
             for (int aK=0 ; aK<aSz  ; aK++)
             {
                aRSEV.SetKthEigenValue(aK,aMul*aVEV(aK));
             }
          }
       }
    }

    // Set conditionning
    {
       cWhitchMinMax<int,Type> aIMM(0,std::abs(aVEV(0)));
       for (int aK=0 ; aK<aSz  ; aK++)
       {
          aIMM.Add(aK,std::abs(aVEV(aK)));
       }
       double aCond = aIMM.Min().Val() / aIMM.Max().Val() ;
       if (aCond <aCondMinAccept)
       {
            //  (ToAdd + VMin) / (Vmax +ToAdd) = Cond : simplify by supresse VMin 
            //  ToAdd = Cond (VMax+ ToAdd)   =>  ToAdd = Cond * VMax (1-Cond)  + Some precuatio,

            Type AbsToAdd = (1.01 * aIMM.Max().Val() * aCondMinAccept) / (1-aCondMinAccept);
            for (int aK=0 ; aK<aSz  ; aK++)
            {
               Type ToAdd = AbsToAdd * SignSupEq0(aVEV(aK));
               aRSEV.SetKthEigenValue(aK,aVEV(aK) + ToAdd);
            }
       }
    }

    return aRSEV.OriMatr();
}




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

template <class Type> cDenseMatrix<Type>  cDenseMatrix<Type>::Inverse() const
{
   tMat::CheckSquare(*this);
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
      StdOut() << "D000 " << AAp.DIm().L2Dist(cDenseMatrix<Type>(aNb,eModeInitImage::eMIA_MatrixId).DIm()) << "\n";

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
       StdOut() << "SOMM EE " << aSomEr << "\n";
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

// ===============  Add tAB tAA  ================

template <class TM,class TV> 
   static void TplAdd_tAB(cDenseMatrix<TM> & aMat,const cDenseVect<TV> & aCol,const cDenseVect<TV> & aLine)
{
    aMat.TplCheckSizeY(aCol);
    aMat.TplCheckSizeX(aLine);
    for (int aY=0 ; aY<aMat.Sz().y() ; aY++)
    {
        TV  aVY = aCol(aY);
        const TV * aVX =   aLine.DIm().RawDataLin();
        TM * aLineMatrix = aMat.DIm().GetLine(aY);
        for (int aNbX=aMat.Sz().x() ; aNbX ;  aNbX--)
        {
           *(aLineMatrix++)  +=   aVY *  *(aVX++);
        }
    }
}

template <class TM,class TV> 
   static void TplWeightedAdd_tAA(cDenseMatrix<TM> & aMat,const TM &aWeight,const cDenseVect<TV> & aV,bool OnlySup)
{
    aMat.TplCheckSizeY(aV);
    aMat.TplCheckSizeX(aV);
    for (int aY=0 ; aY<aMat.Sz().y() ; aY++)
    {
        TV  aVY = aV(aY) * aWeight;
        int aX0 = OnlySup ? aY : 0;
        const TV * aVX =   aV.DIm().RawDataLin() + aX0;
        TM * aLineMatrix = aMat.DIm().GetLine(aY) + aX0;
        for (int aNbX=aMat.Sz().x() -aX0 ; aNbX ;  aNbX--)
        {
           *(aLineMatrix++)  +=   aVY *  *(aVX++);
        }
    }
}
template <class TM,class TV> 
   static void TplAdd_tAA(cDenseMatrix<TM> & aMat,const cDenseVect<TV> & aV,bool OnlySup)
{
   TplWeightedAdd_tAA(aMat,TM(1.0),aV,OnlySup);
/*
    aMat.TplCheckSizeY(aV);
    aMat.TplCheckSizeX(aV);
    for (int aY=0 ; aY<aMat.Sz().y() ; aY++)
    {
        TV  aVY = aV(aY);
        int aX0 = OnlySup ? aY : 0;
        const TV * aVX =   aV.DIm().RawDataLin() + aX0;
        TM * aLineMatrix = aMat.DIm().GetLine(aY) + aX0;
        for (int aNbX=aMat.Sz().x() -aX0 ; aNbX ;  aNbX--)
        {
           *(aLineMatrix++)  +=   aVY *  *(aVX++);
        }
    }
*/
}

template <class TM,class TV> 
   static void TplSub_tAA(cDenseMatrix<TM> & aMat,const cDenseVect<TV> & aV,bool OnlySup)
{
   TplWeightedAdd_tAA(aMat,TM(-1.0),aV,OnlySup);
/*
    aMat.TplCheckSizeY(aV);
    aMat.TplCheckSizeX(aV);
    for (int aY=0 ; aY<aMat.Sz().y() ; aY++)
    {
        TV  aVY = aV(aY);
        int aX0 = OnlySup ? aY : 0;
        const TV * aVX =   aV.DIm().RawDataLin() + aX0;
        TM * aLineMatrix = aMat.DIm().GetLine(aY) + aX0;
        for (int aNbX=aMat.Sz().x() -aX0 ; aNbX ;  aNbX--)
        {
           *(aLineMatrix++)  -=   aVY *  *(aVX++);
        }
    }
*/
}



template <class Type> void cDenseMatrix<Type>::Add_tAB(const tDV & aCol,const tDV & aLine) 
{
   TplAdd_tAB(*this,aCol,aLine);
}
template <class Type> void cDenseMatrix<Type>::Add_tAA(const tDV & aCol,bool OnlySup)
{
   TplAdd_tAA(*this,aCol,OnlySup);
}
template <class Type> void cDenseMatrix<Type>::Sub_tAA(const tDV & aCol,bool OnlySup)
{
   TplSub_tAA(*this,aCol,OnlySup);
}


template <class Type> void cDenseMatrix<Type>::Weighted_Add_tAA(Type aWeight,const tDV & aColLine,bool OnlySup)
{
   TplWeightedAdd_tAA(*this,aWeight,aColLine,OnlySup);
}
/*
        void  Weighted_Add_tAA(const tDV & aColLine,bool OnlySup=true) override;
*/


template <class Type>  void  cDenseMatrix<Type>::Weighted_Add_tAA(Type aWeight,const tSpV & aSparseV,bool OnlySup)
{  
   tMat::CheckSquare(*this);
   tMat::TplCheckX(aSparseV);
   const typename cSparseVect<Type>::tCont & aIV =  aSparseV.IV();
   int aNb  = aIV.size();

   if (OnlySup)
   {
      for (int aKY=0 ; aKY<aNb ; aKY++)
      {
         Type aVy = aIV[aKY].mVal * aWeight;
         int aY = aIV[aKY].mInd;
         Type  * aLineMatrix = DIm().GetLine(aY);
    
         for (int aKX=  0 ; aKX<aNb ; aKX++)
         {
             int aX = aIV[aKX].mInd;
             if (aX>=aY)
                aLineMatrix[aX] +=  aVy * aIV[aKX].mVal;
         }
      }
   }
   else
   {
      for (int aKY=0 ; aKY<aNb ; aKY++)
      {
         Type aVy = aIV[aKY].mVal * aWeight;
         Type  * aLineMatrix = DIm().GetLine(aIV[aKY].mInd);
    
         for (int aKX=  0 ; aKX<aNb ; aKX++)
         {
             aLineMatrix[aIV[aKX].mInd] +=  aVy * aIV[aKX].mVal;
         }
      }
   }

}


/* ================================================= */
/*        cUnOptDenseMatrix                          */
/* ================================================= */

template <class Type> cUnOptDenseMatrix<Type>::cUnOptDenseMatrix(tIm anIm) :
                           cMatrix<Type>(anIm.DIm().Sz().x(),anIm.DIm().Sz().y()),
                           mIm(anIm)
{
   MMVII_INTERNAL_ASSERT_strong(anIm.DIm().P0()==cPt2di(0,0),"Init Matrix P0!= 0,0");
}

template <class Type> cUnOptDenseMatrix<Type>::cUnOptDenseMatrix(int aX,int aY,eModeInitImage aMode) :
      cUnOptDenseMatrix<Type>(tIm(cPt2di(aX,aY),nullptr,aMode))
{
}

template <class Type> Type cUnOptDenseMatrix<Type>::V_GetElem(int aX,int  aY) const 
{
    return GetElem(aX,aY);
}

template <class Type> void cUnOptDenseMatrix<Type>::V_SetElem(int aX,int  aY,const Type & aV) 
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
INSTANTIATE_OPMulMatVect(Type,Type)\


INSTANTIATE_DENSE_MATRICES(tREAL4)
INSTANTIATE_DENSE_MATRICES(tREAL8)
INSTANTIATE_DENSE_MATRICES(tREAL16)




};
