#include "MMVII_Tpl_Images.h"
#include "MMVII_SysSurR.h"


namespace MMVII
{
template <class Type> void  NormalizeProdDiagPos(cDenseMatrix<Type> &aM1,cDenseMatrix<Type> & aM2 ,bool TestOn1);

/* ============================================= */
/*      cResulSVDDecomp    <Type>                */
/* ============================================= */
template <class Type> cResulSVDDecomp<Type>::cResulSVDDecomp(int aNb) :
   mSingularValues (aNb),
   mMatU           (aNb,aNb),
   mMatV           (aNb,aNb)
{
}

template <class Type> const cDenseVect<Type> & cResulSVDDecomp<Type>::SingularValues() const
{
   return mSingularValues;
}

template <class Type> const cDenseMatrix<Type> & cResulSVDDecomp<Type>::MatU() const
{
   return mMatU;
}
template <class Type> const cDenseMatrix<Type> & cResulSVDDecomp<Type>::MatV() const
{
   return mMatV;
}

template <class Type> cDenseMatrix<Type>  cResulSVDDecomp<Type>::OriMatr() const
{
  return mMatU * cDenseMatrix<Type>::Diag(mSingularValues) * mMatV.Transpose();
}




/* ============================================= */
/*      cResulSymEigenValue<Type>                */
/* ============================================= */

template <class Type> 
   cResulSymEigenValue<Type>::cResulSymEigenValue(int aNb) :
        mEigenValues(aNb),
        mEigenVectors(aNb,aNb)
{
}


template <class Type> const cDenseVect<Type>   &  cResulSymEigenValue<Type>::EigenValues() const
{
  return mEigenValues;
}

template <class Type> const cDenseMatrix<Type>   &  cResulSymEigenValue<Type>::EigenVectors() const
{
  return mEigenVectors;
}

template <class Type> cDenseMatrix<Type>  cResulSymEigenValue<Type>::OriMatr() const
{
  return mEigenVectors * cDenseMatrix<Type>::Diag(mEigenValues) * mEigenVectors.Transpose();
}

template <class Type> void  cResulSymEigenValue<Type>::SetKthEigenValue(int aK,const Type & aVal)
{
   mEigenValues(aK) = aVal;
}

template <class Type> Type  cResulSymEigenValue<Type>::Cond(Type aDef) const
{
   cWhichMinMax<int,Type> aIMM(0,std::abs(mEigenValues(0)));
   for (int aK=1 ; aK<mEigenValues.Sz()  ; aK++)
   {
          aIMM.Add(aK,std::abs(mEigenValues(aK)));
   }
   if (aIMM.Max().ValExtre() == Type(0.0))
   {
       MMVII_INTERNAL_ASSERT_strong(aDef>=0,"Conditioning of null eigen value without default");
       return aDef;
   }
   return  aIMM.Max().ValExtre() / aIMM.Min().ValExtre() ;
}




/* ============================================= */
/*      cResulQR_Decomp<Type>                    */
/* ============================================= */


template <class Type> 
   cResulQR_Decomp<Type>::cResulQR_Decomp(const tDM & aQ,const tDM& aR) :
        mQ_Matrix(aQ),
        mR_Matrix(aR)
{
}

template <class Type> 
   cResulQR_Decomp<Type>::cResulQR_Decomp(int aSzX,int aSzY) :
       cResulQR_Decomp<Type>(tDM(aSzY,aSzY),tDM(aSzX,aSzY))
{
}



template <class Type> 
       const cDenseMatrix<Type> &  cResulQR_Decomp<Type>::Q_Matrix() const
{
   return mQ_Matrix;
}
template <class Type> 
        const cDenseMatrix<Type> &  cResulQR_Decomp<Type>::R_Matrix() const
{
   return mR_Matrix;
}
template <class Type> 
       cDenseMatrix<Type> &  cResulQR_Decomp<Type>::Q_Matrix() 
{
   return mQ_Matrix;
}
template <class Type> 
        cDenseMatrix<Type> &  cResulQR_Decomp<Type>::R_Matrix() 
{
   return mR_Matrix;
}





template <class Type> 
    cDenseMatrix<Type>  cResulQR_Decomp<Type>::OriMatr() const
{
    return mQ_Matrix * mR_Matrix;
}

/* ============================================= */
/*      cResulRQ_Decomp<Type>                    */
/* ============================================= */

template <class Type> 
   cResulRQ_Decomp<Type>::cResulRQ_Decomp(const tDM& aR,const tDM & aQ) :
     cResulQR_Decomp<Type>(aR,aQ)
{
}

template <class Type>  cDenseMatrix<Type> cResulRQ_Decomp<Type>::OriMatr() const
{
    return this->mR_Matrix * this->mQ_Matrix;
}


/* ============================================= */
/*      cDenseMatrix<Type>                       */
/* ============================================= */


template <class Type> cResulRQ_Decomp<Type>  cDenseMatrix<Type>::RQ_Decomposition() const
{
    cMatrix<Type>::CheckSquare(*this);

       // std::pair<ElMatrix<double>, ElMatrix<double> >  aQR = QRDecomp(InvertLine(aM0).transpose());
    cDenseMatrix<Type> aM = LineInverse();
    aM.SelfTransposeIn();

    cResulQR_Decomp<Type> aRes = aM.QR_Decomposition();

  // ElMatrix<double> aQ2 = InvertLine(aQ.transpose());
    cDenseMatrix<Type> &aQ = aRes.mQ_Matrix;
    aQ.SelfTransposeIn();
    aQ.SelfLineInverse();

  // ElMatrix<double> aR2 = InvertLine(InvertCol(aR.transpose()));
    cDenseMatrix<Type> &aR = aRes.mR_Matrix;
    aR.SelfTransposeIn();
    aR.SelfColInverse();
    aR.SelfLineInverse();

    NormalizeProdDiagPos(aR,aQ,true);
   // NormalizeProdDiagPos(aRes.mQ_Matrix,aRes.mR_Matrix,false);

    return cResulRQ_Decomp<Type>(aQ,aR);
}

template <class Type> cDenseMatrix<Type>  cDenseMatrix<Type>::Solve(const tDM & aMat,eTyEigenDec aTED) const
{
    tDM aRes(aMat.Sz().x(),aMat.Sz().y());
    SolveIn(aRes,aMat,aTED);
    return aRes;
}


/* ============================================= */
/*         cStrStat2<Type>                       */
/* ============================================= */

 
template <class Type> cStrStat2<Type>::cStrStat2(int aSz) :
    mSz    (aSz),
    mPds   (0),
    mMoy   (aSz    , eModeInitImage::eMIA_Null),
    mMoyMulVE   (aSz    , eModeInitImage::eMIA_Null),
    mTmp   (aSz    , eModeInitImage::eMIA_Null),
    mCov   (aSz,aSz, eModeInitImage::eMIA_Null),
    mEigen (1)
{
}

template <class Type> double              cStrStat2<Type>::Pds() const {return mPds;}
template <class Type> const cDenseVect<Type>  & cStrStat2<Type>::Moy() const {return mMoy;}
template <class Type> const cDenseMatrix<Type>& cStrStat2<Type>::Cov() const {return mCov;}
template <class Type> cDenseMatrix<Type>& cStrStat2<Type>::Cov() {return mCov;}
template <class Type> void cStrStat2<Type>::WeightedAdd(const cDenseVect<Type> & aV,const Type & aW)
{
    mPds += aW;
    AddMulIn(mMoy.DIm(),aV.DIm(),aW);
    mCov.WeightedAdd_tAA(aV,aW,true);
}

template <class Type> void cStrStat2<Type>::Add(const cDenseVect<Type> & aV)
{
    WeightedAdd(aV,1.0);
}

template <class Type> void cStrStat2<Type>::Normalise(bool CenteredAlso)
{
   DivCsteIn(mMoy.DIm(),mPds);
   DivCsteIn(mCov.DIm(),mPds);
   if (CenteredAlso)
      mCov.Sub_tAA(mMoy);
   mCov.SelfSymetrizeBottom();
}

template <class Type> const cResulSymEigenValue<Type> & cStrStat2<Type>::DoEigen()
{
    mCov.SelfSymetrizeBottom();
    mEigen = mCov.SymEigenValue();
    mEigen.EigenVectors().MulLineInPlace(mMoyMulVE,mMoy);
    return mEigen;
}

template <class Type>  void cStrStat2<Type>::ToNormalizedCoord(cDenseVect<Type>  & aV1,const cDenseVect<Type>  & aV2) const
{
    DiffImageInPlace(mTmp.DIm(),aV2.DIm(),mMoy.DIm());
    // mCov.MulColInPlace(aV1,mTmp);
    // mEigen.EigenVectors().MulColInPlace(aV1,mTmp);
    mEigen.EigenVectors().MulLineInPlace(aV1,mTmp);
}

template <class Type>  double cStrStat2<Type>::KthNormalizedCoord(int aX,const cDenseVect<Type>  & aV2) const
{
  return mEigen.EigenVectors().MulLineElem(aX,aV2) -mMoyMulVE(aX);
}

/* =============================================== */
/*       cElemDecompQuad / cDecSumSqLinear         */
/* =============================================== */

template <class Type>  cElemDecompQuad<Type>::cElemDecompQuad(const Type& aW,const cDenseVect<Type> & aV,const Type & aCste):
    mW      (aW),
    mCoeff  (aV),
    mCste   (aCste)
{
}

template <class Type>  cDecSumSqLinear<Type>::cDecSumSqLinear() :
                         mNbVar (-1)
{
}

template <class Type>  void cDecSumSqLinear<Type>::Set
                         (const cDenseVect<Type> & aX0,const cDenseMatrix<Type> & aMat,const cDenseVect<Type> & aVecB) 

{
   cDenseVect<Type> aVect = aVecB + aMat * aX0;
   mNbVar =aMat.Sz().x() ;
   cMatrix<Type>::CheckSquare(aMat);
  //       tXAX-2BX   
  //     = t(X-S) A (X-S) 
  //     = (X-S) tR D  R (X-S) 
  //     = (X-S) tR L  L  R (X-S)     with L=sqrt(LD)
  //      Som( ||li (RiX-RiS)

    cDenseVect<Type> aSol =  aMat.SolveColumn(aVect,eTyEigenDec::eTED_LLDT);

    cResulSymEigenValue<Type> aRSEV = aMat.SymEigenValue() ;
    const cDenseVect<Type>   &  aVEVal = aRSEV.EigenValues() ;
    cDenseMatrix<Type>   aMEVect = aRSEV.EigenVectors().Transpose();
    cDenseVect<Type>     aVCste =  aMEVect * aSol;


    int aDim = aSol.Sz();

    for (int aK=0 ; aK<aDim ; aK++)
    {
          Type anEV = aVEVal(aK);
          if (anEV>0)
          {
               tElem aEDQ(anEV,aMEVect.ReadLine(aK),aVCste(aK));
               mVElems.push_back(aEDQ);
          }
    }
}

template <class Type> const std::vector<cElemDecompQuad<Type> > & cDecSumSqLinear<Type>::VElems() const
{
    return mVElems;
}

template <class Type> cLeasSqtAA<Type> cDecSumSqLinear<Type>::OriSys() const
{
    cLeasSqtAA<Type> aResult(mNbVar);

    for (const auto & anEl : mVElems)
        aResult.PublicAddObservation(anEl.mW,anEl.mCoeff,anEl.mCste);

    return aResult;
}




/* ===================================================== */
/* =====              INSTANTIATION                ===== */
/* ===================================================== */


#define INSTANTIATE_ORTHOG_DENSE_MATRICES(Type)\
template  class  cResulSVDDecomp<Type>;\
template  class  cResulSymEigenValue<Type>;\
template  class  cElemDecompQuad<Type>;\
template  class  cDecSumSqLinear<Type>;\
template  class  cStrStat2<Type>;\
template  class  cResulQR_Decomp<Type>;\
template  class  cResulRQ_Decomp<Type>;\
template  class  cDenseMatrix<Type>;


INSTANTIATE_ORTHOG_DENSE_MATRICES(tREAL4)
INSTANTIATE_ORTHOG_DENSE_MATRICES(tREAL8)
INSTANTIATE_ORTHOG_DENSE_MATRICES(tREAL16)


};
