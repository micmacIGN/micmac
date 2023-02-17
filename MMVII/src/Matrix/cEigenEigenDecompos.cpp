#include "MMVII_Tpl_Images.h"

#include "MMVII_EigenWrap.h"

using namespace Eigen;

namespace MMVII
{
	// (A + i*B)^-1 = (A + B*A^-1*B)^-1 - i*(B + A*B^-1*A)^-1 


template <class Type> cResulEigenDecomp<Type>::cResulEigenDecomp(int aN):
    mEigenVec_R  (aN),
    mEigenVec_I  (aN),
    mEigenVal_R  (aN),
    mEigenVal_I  (aN)
{
}

template <class Type>   cResulEigenDecomp<Type>  cDenseMatrix<Type>::Eigen_Decomposition() const
{
   this->CheckSquare(*this);  // this for scope, method is static
   int aNb = Sz().x();

   cResulEigenDecomp<Type> aResult(aNb);

   tConst_EW aWrap(*this);


   Eigen::EigenSolver<typename tNC_EW::tEigenMat> eigensolver(aWrap.EW());

   auto aEigVecs = eigensolver.eigenvectors();
   auto aEigVals = eigensolver.eigenvalues();

   for (int aX=0 ; aX<aNb ; aX++)
   {
        aResult.mEigenVal_R(aX) = aEigVals(aX).real();
        aResult.mEigenVal_I(aX) = aEigVals(aX).imag();
        for (int aY=0 ; aY<aNb ; aY++)
	{
            aResult.mEigenVec_R.SetElem(aX,aY,aEigVecs(aY,aX).real());
            aResult.mEigenVec_I.SetElem(aX,aY,aEigVecs(aY,aX).imag());
	}
   }

   return aResult;
}


template <class Type> void Tpl_Bench_EigenDecompos(cParamExeBench & aParam)
{
    for (int aK=0 ; aK<100 ; aK++)
    {
	 int aNb = 1 +  RandUnif_N(10);
	 bool IsSym = ((aK%2)==0);
	 cDenseMatrix<Type>  aMat = cDenseMatrix<Type>::RandomSquareRegMatrix(cPt2di(aNb,aNb),IsSym,0,0);
         cResulEigenDecomp<Type> aEigDec  = aMat.Eigen_Decomposition();

	 if (IsSym)
	 {
             // is it a real decompos
             MMVII_INTERNAL_ASSERT_bench(aEigDec.mEigenVal_I.L2Norm()<1e-6,"not real eigen values");
             MMVII_INTERNAL_ASSERT_bench(aEigDec.mEigenVec_I.DIm().L2Norm()<1e-6,"not real eigen vectors");

	     // is eigen vector matrix orthogonal
	     cDenseMatrix<Type>  aEigVec = aEigDec.mEigenVec_R;
             MMVII_INTERNAL_ASSERT_bench(aEigVec.Unitarity()<1e-6,"not real eigen vectors");

	     // chek recompose is ok 
	     cDenseMatrix<Type> aRecomp = aEigVec *  cDenseMatrix<Type>::Diag(aEigDec.mEigenVal_R) * aEigVec.Inverse();
	     cDenseMatrix<Type> aCheck = aMat - aRecomp;
             MMVII_INTERNAL_ASSERT_bench(aCheck.DIm().L2Norm()<1e-6,"not real eigen vectors");
	 }
         else
         {
	     cDenseMatrix<Type>  A = aEigDec.mEigenVec_R;
	     cDenseMatrix<Type>  B  = aEigDec.mEigenVec_I;
	     cDenseMatrix<Type>  D =  cDenseMatrix<Type>::Diag(aEigDec.mEigenVal_R);
	     cDenseMatrix<Type>  E =  cDenseMatrix<Type>::Diag(aEigDec.mEigenVal_I);
             //   (A+iB) (D+iE) (A+iB) -1 =  Mat 
             //   (A+iB) (D+iE)           =  Mat (A+iB)
	     //   AD -BE + i(BD+AE)  = Mat A  + i Mat B
	     cDenseMatrix<Type> aCheckR = A*D-B*E - aMat * A;
	     cDenseMatrix<Type> aCheckI = B*D + A*E - aMat*B;
             MMVII_INTERNAL_ASSERT_bench(aCheckR.DIm().L2Norm()<1e-6,"eigen vectors recompose (real part)");
             MMVII_INTERNAL_ASSERT_bench(aCheckI.DIm().L2Norm()<1e-6,"eigen vectors recompose (imag part)");
         }
    }
}

void Bench_EigenDecompos(cParamExeBench & aParam)
{
     Tpl_Bench_EigenDecompos<tREAL8>(aParam);
}


template class cResulEigenDecomp<tREAL8>;
template class cDenseMatrix<tREAL8>;


};
