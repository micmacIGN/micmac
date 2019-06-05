#include "include/MMVII_all.h"
#include "include/MMVII_Tpl_Images.h"

#include "MMVII_EigenWrap.h"
#include "ExternalInclude/Eigen/Eigenvalues" 
#include "ExternalInclude/Eigen/Householder"  // HouseholderQR.h"

using namespace Eigen;

namespace MMVII
{

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

/* ============================================= */
/*      cResulQR_Decomp<Type>                    */
/* ============================================= */


template <class Type> 
   cResulQR_Decomp<Type>::cResulQR_Decomp(int aSzX,int aSzY) :
        mQ_Matrix(aSzY,aSzY),
        mR_Matrix(aSzX,aSzY)
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
    cDenseMatrix<Type>  cResulQR_Decomp<Type>::OriMatr() const
{
    return mQ_Matrix * mR_Matrix;
}


/* ============================================= */
/*      cDenseMatrix<Type>                       */
/* ============================================= */

template <class Type> cResulQR_Decomp<Type>  cDenseMatrix<Type>::QR_Decomposition() const
{
    cResulQR_Decomp<Type> aRes(Sz().x(),Sz().y());
    tConst_EW aWrap(*this);
    // cDenseMatrix<Type> aM(2,2);
    HouseholderQR<typename tNC_EW::tEigenMat > qr(aWrap.EW());

    tNC_EW aWrapQ(aRes.mQ_Matrix);
    aWrapQ.EW() = qr.householderQ();

    tNC_EW aWrapR(aRes.mR_Matrix);
    aWrapR.EW() = qr.matrixQR();

    aRes.mR_Matrix.SelfTriangSup();

   return aRes;
}



template <class Type> cResulSymEigenValue<Type>  cDenseMatrix<Type>::SymEigenValue() const
{
    cMatrix<Type>::CheckSquare(*this);
    int aNb = Sz().x();
    cResulSymEigenValue<Type> aRes(aNb);

    tConst_EW aWrap(*this);
    SelfAdjointEigenSolver<typename tConst_EW::tEigenMat > es(aWrap.EW());

    tNC_EW  aWrapEVect(aRes.mEigenVectors);
    aWrapEVect.EW() =  es.eigenvectors();
    
    cNC_EigenColVectWrap<Type>  aWrapEVal(aRes.mEigenValues);
    aWrapEVal.EW() =  es.eigenvalues();

    return aRes;
}


template <class Type> double  cDenseMatrix<Type>::Unitarity() const
{
     cDenseMatrix<Type> atMM = Transpose() * (*this);
     cDenseMatrix<Type> aId(atMM.Sz().x(),eModeInitImage::eMIA_MatrixId);
     return aId.DIm().L2Dist(atMM.DIm());
   
}

/*
void TestSSS()
{
   cDenseMatrix<double> aM(2,2);
   cNC_EigenMatWrap<double> aWrap(aM);

   SelfAdjointEigenSolver<Matrix<double,Dynamic,Dynamic,RowMajor> > es(aWrap.EW());


   aWrap.EW() =  es.eigenvalues();
   aWrap.EW() =  es.eigenvectors();
   // SelfAdjointEigenSolver<cNC_EigenMatWrap<double>::tEigenWrap> es(aWrap.EW());
}mEVal
*/


/*
SelfAdjointEigenSolver<Matrix4f> es;
Matrix4f X = Matrix4f::Random(4,4);
Matrix4f A = X + X.transpose();
es.compute(A);
cout << "The eigenvalues of A are: " << es.eigenvalues().transpose() << endl;
es.compute(A + Matrix4f::Identity(4,4)); // re-use es to compute eigenvalues of A+I
cout << "The eigenvalues of A+I are: " << es.eigenvalues().transpose() << endl;
*/

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

template <class Type> const double              cStrStat2<Type>::Pds() const {return mPds;}
template <class Type> const cDenseVect<Type>  & cStrStat2<Type>::Moy() const {return mMoy;}
template <class Type> const cDenseMatrix<Type>& cStrStat2<Type>::Cov() const {return mCov;}
template <class Type> cDenseMatrix<Type>& cStrStat2<Type>::Cov() {return mCov;}
template <class Type> void cStrStat2<Type>::Add(const cDenseVect<Type> & aV)
{
    mPds ++;
    AddIn(mMoy.DIm(),aV.DIm());
    mCov.Add_tAA(aV,true);
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


/* ===================================================== */
/* =====              INSTANTIATION                ===== */
/* ===================================================== */


#define INSTANTIATE_ORTHOG_DENSE_MATRICES(Type)\
template  class  cStrStat2<Type>;\
template  class  cDenseMatrix<Type>;\
template  class  cResulSymEigenValue<Type>;\
template  class  cResulQR_Decomp<Type>;\

INSTANTIATE_ORTHOG_DENSE_MATRICES(tREAL4)
INSTANTIATE_ORTHOG_DENSE_MATRICES(tREAL8)
INSTANTIATE_ORTHOG_DENSE_MATRICES(tREAL16)


};
