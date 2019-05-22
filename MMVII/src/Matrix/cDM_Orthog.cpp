#include "include/MMVII_all.h"


#include "MMVII_EigenWrap.h"
#include "ExternalInclude/Eigen/Eigenvalues" 

using namespace Eigen;

namespace MMVII
{

/* ============================================= */
/*      cResulSymEigenValue<Type>                */
/* ============================================= */

template <class Type> 
   cResulSymEigenValue<Type>::cResulSymEigenValue(int aNb) :
        mEVal(aNb),
        mEVect(aNb,aNb)
{
}


/* ============================================= */
/*      cDenseMatrix<Type>                       */
/* ============================================= */


template <class Type> cResulSymEigenValue<Type>  cDenseMatrix<Type>::SymEigenValue() const
{
    cMatrix::CheckSquare(*this);
    int aNb = Sz().x();
    cResulSymEigenValue<Type> aRes(aNb);

    tConst_EW aWrap(*this);
    SelfAdjointEigenSolver<typename tConst_EW::tEigenMat > es(aWrap.EW());

    tNC_EW  aWrapEVect(aRes.mEVect);
    aWrapEVect.EW() =  es.eigenvectors();
    
    cNC_EigenColVectWrap<Type>  aWrapEVal(aRes.mEVal);
    aWrapEVal.EW() =  es.eigenvalues();

    return aRes;
}


template <class Type> double  cDenseMatrix<Type>::Unitarity() const
{
     cDenseMatrix<Type> atMM = Transpose() * (*this);
     cDenseMatrix<Type> aId(atMM.Sz().x(),eModeInitImage::eMIA_MatrixId);
     return aId.DIm().L2Dist(atMM.DIm());
   
}

void TestSSS()
{
   cDenseMatrix<double> aM(2,2);
   cNC_EigenMatWrap<double> aWrap(aM);

   SelfAdjointEigenSolver<Matrix<double,Dynamic,Dynamic,RowMajor> > es(aWrap.EW());


   aWrap.EW() =  es.eigenvalues();
   aWrap.EW() =  es.eigenvectors();
   // SelfAdjointEigenSolver<cNC_EigenMatWrap<double>::tEigenWrap> es(aWrap.EW());
}


/*
SelfAdjointEigenSolver<Matrix4f> es;
Matrix4f X = Matrix4f::Random(4,4);
Matrix4f A = X + X.transpose();
es.compute(A);
cout << "The eigenvalues of A are: " << es.eigenvalues().transpose() << endl;
es.compute(A + Matrix4f::Identity(4,4)); // re-use es to compute eigenvalues of A+I
cout << "The eigenvalues of A+I are: " << es.eigenvalues().transpose() << endl;
*/


/* ===================================================== */
/* =====              INSTANTIATION                ===== */
/* ===================================================== */


#define INSTANTIATE_ORTHOG_DENSE_MATRICES(Type)\
template  class  cDenseMatrix<Type>;\
template  class  cResulSymEigenValue<Type>;\

INSTANTIATE_ORTHOG_DENSE_MATRICES(tREAL4)
INSTANTIATE_ORTHOG_DENSE_MATRICES(tREAL8)
INSTANTIATE_ORTHOG_DENSE_MATRICES(tREAL16)


};
