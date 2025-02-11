#include "MMVII_Matrix.h"

namespace MMVII
{
/*  Static => see note on BaseMatrixes.cpp
*/


/* ============================================= */
/*      cDenseMatrix<Type>                       */
/* ============================================= */

template <class Type> cDenseMatrix<Type>::cDenseMatrix(int aX,int aY,eModeInitImage aMode) :
                             cUnOptDenseMatrix<Type>(aX,aY,aMode)
{
}

template <class Type> cDenseMatrix<Type>::cDenseMatrix(tIm anIm) :
                             cUnOptDenseMatrix<Type>(anIm)
{
}


template <class Type> cDenseMatrix<Type>  cDenseMatrix<Type>::Identity(int aSz)
{
	return cDenseMatrix(aSz,eModeInitImage::eMIA_MatrixId);
}

template <class Type> cDenseMatrix<Type>::cDenseMatrix(int aXY,eModeInitImage aMode) :
                             cDenseMatrix<Type>(aXY,aXY,aMode)
{
}

template <class Type> cDenseMatrix<Type>  cDenseMatrix<Type>::Dup() const
{
    return  cDenseMatrix<Type>(Im().Dup());
}

template <class Type> cDenseMatrix<Type>  cDenseMatrix<Type>::ExtendSquareMat(int aNewSz,eModeInitImage aMode)
{
    this->CheckSquare(*this);
    MMVII_INTERNAL_ASSERT_medium(aNewSz>=Sz().x(),"ExtendSquareMat cannot reduce size");

    tDM aRes(aNewSz,aMode);

    for (const auto & aPix : DIm())
    {
          aRes.DIm().SetV(aPix,DIm().GetV(aPix));
    }
    return aRes;
}

template <class Type> cDenseMatrix<Type>  cDenseMatrix<Type>::ExtendSquareMatId(int aNewSz)
{
     return  ExtendSquareMat(aNewSz,eModeInitImage::eMIA_MatrixId);
}

template <class Type> cDenseMatrix<Type>  cDenseMatrix<Type>::ExtendSquareMatNull(int aNewSz)
{
     return  ExtendSquareMat(aNewSz,eModeInitImage::eMIA_Null);
}

// eMIA_Null,


template <class Type> cDenseMatrix<Type> cDenseMatrix<Type>::Diag(const cDenseVect<Type> & aV)
{
    cDenseMatrix<Type> aRes(aV.Sz(),eModeInitImage::eMIA_Null);
    for (int aK=0 ; aK<aV.Sz(); aK++)
        aRes.SetElem(aK,aK,aV(aK));

    return aRes;
}

template <class Type> cDenseMatrix<Type> cDenseMatrix<Type>::FromLines(const std::vector<tDV> & aVV)
{
    cDenseMatrix<Type>  aRes(aVV.at(0).Sz(),aVV.size());
    for (int aY=0 ; aY<int(aVV.size()) ; aY++)
       aRes.WriteLine(aY,aVV.at(aY));

    return aRes;
}

template <class Type> cDenseMatrix<Type> cDenseMatrix<Type>::MatLine(const tDV & aV) {return FromLines({aV});}


template <class Type> cDenseMatrix<Type> cDenseMatrix<Type>::FromCols(const std::vector<tDV> & aVV)
{
    cDenseMatrix<Type>  aRes(aVV.size(),aVV.at(0).Sz());
    for (int aX=0 ; aX<int(aVV.size()) ; aX++)
       aRes.WriteCol(aX,aVV.at(aX));

    return aRes;
}

template <class Type> cDenseMatrix<Type> cDenseMatrix<Type>::MatCol(const tDV & aV) {return FromCols({aV});}

//tDM SubMatrix(const cPt2di & aSz) const

template <class Type> cDenseMatrix<Type> cDenseMatrix<Type>::SubMatrix(const cPt2di & aP0,const cPt2di & aP1) const
{
    cPt2di aSz = aP1 -aP0;
    cDenseMatrix<Type>   aRes(aSz.x(),aSz.y());

    for (const auto & aPix : aRes.DIm())
        aRes.SetElem(aPix.x(),aPix.y(),GetElem(aPix+aP0));

    return aRes;
}

template <class Type> cDenseMatrix<Type> cDenseMatrix<Type>::SubMatrix(const cPt2di & aSz) const
{
    return SubMatrix(cPt2di(0,0),aSz);
}



template <class Type> cDenseMatrix<Type>  cDenseMatrix<Type>::ClosestOrthog() const
{
    this->CheckSquare(*this);
    cResulSVDDecomp<Type> aSVD  =   SVD();

    // OriMatr :  return mMatU * cDenseMatrix<Type>::Diag(mSingularValues) * mMatV.Transpose();
    cDenseVect<Type> aVP = aSVD.SingularValues().Dup(); 

    Type aSignGlob = 1;
    for (int aK=0 ; aK<aVP.Sz(); aK++)
    {
        Type aSign = (aVP(aK)>=0)  ? 1 : - 1;
        aVP(aK) = aSign;
	aSignGlob *= aSign;
    }

    return aSVD.MatU() * cDenseMatrix<Type>::Diag(aVP) * aSVD.MatV().Transpose();
}

template <class Type> Type  cDenseMatrix<Type>::L2Dist(const cDenseMatrix<Type> & aV,bool isAvg) const
{
   return DIm().L2Dist(aV.DIm(),isAvg);
}


template <class Type> Type  cDenseMatrix<Type>::SqL2Dist(const cDenseMatrix<Type> & aV,bool isAvg) const
{
   return DIm().SqL2Dist(aV.DIm(),isAvg);
}


template <class Type> void cDenseMatrix<Type>::PushByLine(std::vector<Type> & aRes) const
{
    for (const auto & aPix : DIm())
    {
          aRes.push_back(GetElem(aPix));
    }
}

template <class Type> void cDenseMatrix<Type>::PushByCol(std::vector<Type> & aRes) const
{
    for (int aX=0 ; aX<Sz().x() ; aX++)
        for (int aY=0 ; aY<Sz().y() ; aY++)
            aRes.push_back(GetElem(aX,aY));
}


template <class Type> cResulSVDDecomp<Type>  cDenseMatrix<Type>::RandomSquareRegSVD
                    (
                        const cPt2di&aSz,
                        bool   IsSym,
                        double AmplAcc,
                        double aCondMinAccept
                    )
{
    cDenseMatrix<Type>  aMatRand(aSz.x(),aSz.y(),eModeInitImage::eMIA_RandCenter);

    // For simplicity purpose, do SVD and not jacobi, even when sym
    if (IsSym)
    {
       aMatRand.SelfSymetrize();
    }
    cResulSVDDecomp<Type> aSVDD = aMatRand.SVD();
    cDenseVect<Type>  aVDiag = aSVDD.SingularValues();
    int aNb = aVDiag.Sz();

    // Set global amplitude  to avoid almost all avoid values
    {
       Type aSom = 0;  // Average of eigen value
       for (int aK=0 ; aK<aNb  ; aK++)
       {
          aSom += std::abs(aVDiag(aK));
       }
       aSom /= aNb;
       if (aSom<AmplAcc)
       {
          if (aSom==0)  // Case 0 , put any value non degenerate
          {
             for (int aK=0 ; aK<aNb  ; aK++)
             {
                aVDiag(aK) = (1+aK)*(HeadOrTail() ? -1 : 1);
             }
          }
          else // else multiply to have the given amplitude
          {
             double aMul = AmplAcc/aSom;
             for (int aK=0 ; aK<aNb  ; aK++)
             {
                aVDiag(aK) *= aMul;
             }
          }
       }
    }

    // Set conditioning
    {
       // Compute max & min of all ABS values (which one get it is of no interest)
       cWhichMinMax<int,Type> aIMM(0,std::abs(aVDiag(0)));
       for (int aK=0 ; aK<aNb  ; aK++)
       {
          aIMM.Add(aK,std::abs(aVDiag(aK)));
       }
       double aCond = aIMM.Min().ValExtre() / aIMM.Max().ValExtre() ;
       // if conditioning is too low
       if (aCond <aCondMinAccept)
       {
            //  (ToAdd + VMin) / (Vmax +ToAdd) = Cond : simplify by supresse VMin 
            //  ToAdd = Cond (VMax+ ToAdd)   =>  ToAdd = Cond * VMax (1-Cond)  + Some precuatio,

            // Compute a value  to add to all, in worst case 
            //  Max =  Max (1+ C/(1-C))
            //  Min =  Max * C/(1-C)           
            //  and cond is equal to C !
            Type AbsToAdd = (1.01 * aIMM.Max().ValExtre() * aCondMinAccept) / (1-aCondMinAccept);
            for (int aK=0 ; aK<aNb  ; aK++)
            {
               aVDiag(aK) += AbsToAdd * SignSupEq0(aVDiag(aK)); // 1 or -1
               // aRSEV.SetKthEigenValue(aK,aVEV(aK) + ToAdd);
            }
       }
    }

    return aSVDD;
}

template <class Type> cDenseMatrix<Type>  cDenseMatrix<Type>::RandomSquareRegMatrix
                    (
                        const cPt2di&aSz,
                        bool   IsSym,
                        double AmplAcc,
                        double aCondMinAccept
                    )
{
    cResulSVDDecomp<Type>  aSVDD = RandomSquareRegSVD(aSz,IsSym,AmplAcc,aCondMinAccept);
    return aSVDD.OriMatr();
/*
StdOut() << "HHhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh" << std::endl;
    return RandomSquareRegSVD(aSz,IsSym,AmplAcc,aCondMinAccept).OriMatr();
*/
}

template <class Type> cResulSVDDecomp<Type>  
    cDenseMatrix<Type>::RandomSquareRankDefSVD(const cPt2di & aSz,int aSzKer)
{
    cResulSVDDecomp<Type>  aSVDD = RandomSquareRegSVD(aSz,false,1e-2,1e-3);
    cDenseVect<Type>  aVDiag = aSVDD.SingularValues();
    std::vector<int>  aVInd0 = RandSet(aSzKer,aVDiag.Sz());

    for (const auto & aInd0 : aVInd0)
        aVDiag(aInd0) = 0;

    return aSVDD;
}

template<class Type> cDenseMatrix<Type> 
    cDenseMatrix<Type>::RandomSquareRankDefMatrix(const cPt2di & aSz,int aSzK)
{
    cResulSVDDecomp<Type>  aSVDD = RandomSquareRankDefSVD(aSz,aSzK);
    return aSVDD.OriMatr();
}

template<class Type> cDenseMatrix<Type> cDenseMatrix<Type>::RandomOrthogMatrix(const int aSz)
{
    tDM  aMat(aSz,eModeInitImage::eMIA_RandCenter);
    aMat.SelfSymetrize();
    cResulSymEigenValue<Type>   aSE = aMat.SymEigenValue();
    return aSE.EigenVectors();
}



template<class Type> cDenseVect<Type> cDenseMatrix<Type>::Kernel(Type * aVp) const
{
    /*   U D tV K  =>  tV K = Dk     => K = V Dk */

    cResulSVDDecomp<Type> aSVDD = SVD();
    cDenseVect<Type>  aVDiag = aSVDD.SingularValues();

    cWhichMin<int,Type> aWMin(0,std::abs(aVDiag(0)));
    for (int aK=1 ; aK<aVDiag.Sz() ; aK++)
        aWMin.Add(aK,std::abs(aVDiag(aK)));

    if (aVp) 
       *aVp = aVDiag(aWMin.IndexExtre());
    
    return aSVDD.MatV().ReadCol(aWMin.IndexExtre());
}

template<class Type> cDenseVect<Type> cDenseMatrix<Type>::EigenVect(const Type & aVal,Type * aVp) const
{
    this->CheckSquare(*this);
    cDenseMatrix<Type> aM = Dup();
    for (int aK=0 ; aK<Sz().x() ; aK++)
        aM.SetElem(aK,aK,aM.GetElem(aK,aK) - aVal);
     
    return aM.Kernel(aVp);
}


// template <class Type> cResulSVDDecomp<Type>  
// tDM RandomSquareRankDefMatrix(const cPt2di & aSz,int aSzK);






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
      StdOut() << "D000 " << AAp.DIm().L2Dist(cDenseMatrix<Type>(aNb,eModeInitImage::eMIA_MatrixId).DIm()) << std::endl;

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
       StdOut() << "SOMM EE " << aSomEr << std::endl;
       Ap = Ap * ImE;
       aK++;
   }
   getchar();
   return Ap;
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

template <class Type> Type Bilinear (const cDenseVect<Type> & aV1,const cDenseMatrix<Type> & aMat,const cDenseVect<Type>& aV2)
{
   return aV1.DotProduct(aMat*aV2);
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

template <class Type> cDenseVect<Type>   cDenseMatrix<Type>::Random1LineCombination() const
{
     return (cDenseMatrix<Type>(Sz().y(),1,eModeInitImage::eMIA_RandCenter) * (*this)).ReadLine(0);
}

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


template <class Type>  Type  cDenseMatrix<Type>:: DotProduct_Col(int aX,const tSpV & aVec) const
{
    Type aRes = 0.0;
    for (const auto & aPairIV : aVec)
       aRes += GetElem(aX,aPairIV.mInd) * aPairIV.mVal;

   return aRes;
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

template <class Type> void cUnOptDenseMatrix<Type>::ResizeAndCropIn
                           (
                                const cPt2di & aP0,
                                const cPt2di & aP1,
                                const cUnOptDenseMatrix<Type> & aM2
                           )
{
     //aDIm.ResizeI(aP1-aP0);
     Resize(aP1-aP0);
     tDIm & aDIm =   mIm.DIm();

     aDIm.CropIn(aP0,aM2.DIm());
}

template <class Type> void cUnOptDenseMatrix<Type>::Resize(const cPt2di & aSz)
{
     static_cast<cRect2 &>(*this) = cRect2(aSz);
     DIm().Resize(aSz);
}

template <class Type> cDenseMatrix<Type> cDenseMatrix<Type>::Crop(const cPt2di & aP0,const cPt2di & aP1) const
{
    cDenseMatrix<Type> aRes = cDenseMatrix<Type>(aP1-aP0);
    aRes.ResizeAndCropIn(aP0,aP1,*this);
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
template  Type Bilinear(const cDenseVect<Type>&,const cDenseMatrix<Type>&,const cDenseVect<Type>&);\
template  class  cUnOptDenseMatrix<Type>;\
template  class  cDenseMatrix<Type>;\
template  cDenseMatrix<Type> operator * (const cDenseMatrix<Type> &,const cDenseMatrix<Type>&);\
template  cUnOptDenseMatrix<Type> operator * (const cUnOptDenseMatrix<Type> &,const cUnOptDenseMatrix<Type>&);\
INSTANTIATE_OPMulMatVect(Type,Type)\


INSTANTIATE_DENSE_MATRICES(tREAL4)
INSTANTIATE_DENSE_MATRICES(tREAL8)
INSTANTIATE_DENSE_MATRICES(tREAL16)




};
