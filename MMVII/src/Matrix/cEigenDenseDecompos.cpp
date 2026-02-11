// CM: g++ reports  "result: may be used uninitialized" with Eigen 3.4.0
#if defined(__GNUC__) && !defined(__clang__)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif

#include "MMVII_Tpl_Images.h"
#include "MMVII_SysSurR.h"

#include "MMVII_EigenWrap.h"
#include "Eigen/Eigenvalues"
#include "Eigen/Householder"  // HouseholderQR.h"
// #include "Eigen/Cholesky"  // HouseholderQR.h"

#if defined(__GNUC__) && !defined(__clang__)
#  pragma GCC diagnostic pop
#endif
using namespace Eigen;

namespace MMVII
{

template <class Type> void  NormalizeProdDiagPos(cDenseMatrix<Type> &aM1,cDenseMatrix<Type> & aM2 ,bool TestOn1);


template <class Type> cResulSVDDecomp<Type> cDenseMatrix<Type>::SVD(bool PremMatDirect) const
{
   this->CheckSquare(*this);  // this for scope, method is static
   int aNb = Sz().x();
   cResulSVDDecomp<Type> aRes(aNb);

   tConst_EW aWrap(*this);
   JacobiSVD<typename tNC_EW::tEigenMat > aJacSVD(aWrap.EW(),ComputeThinU | ComputeThinV);

   cNC_EigenMatWrap<Type> aWrap_U(aRes.mMatU);
   aWrap_U.EW() = aJacSVD.matrixU();
   Type aMulMatOrhog = 1.0;
   if (PremMatDirect && ( aRes.mMatU.Det()<0.0))
   {
       aMulMatOrhog = -1.0;
       aRes.mMatU.DIm() *= aMulMatOrhog;
   }

   cNC_EigenMatWrap<Type> aWrap_V(aRes.mMatV);
   aWrap_V.EW() = aJacSVD.matrixV();

   cNC_EigenColVectWrap<Type>  aWrapSVal(aRes.mSingularValues);
   aWrapSVal.EW() =  aJacSVD.singularValues();

   if (aMulMatOrhog<0)
   {
       aRes.mMatV.DIm() *= aMulMatOrhog;
   }

   return aRes;
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

    //  extract the Q matrix
    {
        tNC_EW aWrapQ(aRes.mQ_Matrix);
        aWrapQ.EW() = qr.householderQ();
    }

    //  extract the R matrix
    {
       tNC_EW aWrapR(aRes.mR_Matrix);
       aWrapR.EW() = qr.matrixQR();

       aRes.mR_Matrix.SelfTriangSup(); // make the image triangular sup (maybe exist  residual of eigen)
    }
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

template <class Type>  void cDenseMatrix<Type>::SolveIn(tDM & aRes,const tDM & aMat,eTyEigenDec aTED) const
{
    tMat::CheckSquare(*this);
    tMat::CheckSizeMul(*this,aMat);

    MMVII_INTERNAL_ASSERT_medium(aRes.Sz() == aMat.Sz(),"SolveIn : Bad Sz");
    // tDM aRes(aMat.Sz().x(),aMat.Sz().y());

    tConst_EW aWThis(*this);
    tConst_EW aWMat(aMat);
    tNC_EW aWRes(aRes);

    // aWRes.EW() = aWThis.EW().colPivHouseholderQr().solve(aWMat.EW());
    if (aTED == eTyEigenDec::eTED_PHQR)
    {
       // aWRes.EW() = aWThis.EW().colPivHouseholderQr().solve(aWMat.EW());
// -       aWRes.EW() = aWThis.EW().colPivHouseholderQr().solve(aWMat.EW());
        if (EigenDoTestSuccess())
        {
           auto solver = aWThis.EW().colPivHouseholderQr();
           aWRes.EW() = solver.solve(aWMat.EW());
           if (solver.info()!=Eigen::Success)
           {
               ON_EIGEN_NO_SUCC("SolveIn(eTED_PHQR)");
           }
        }
        else
        {
           aWRes.EW() = aWThis.EW().colPivHouseholderQr().solve(aWMat.EW());
        }
    }
    else if (aTED == eTyEigenDec::eTED_LLDT)
    {
       if (EigenDoTestSuccess())
       {
          auto solver = aWThis.EW().ldlt();
          aWRes.EW() = solver.solve(aWMat.EW());
          if (solver.info()!=Eigen::Success)
          {
              ON_EIGEN_NO_SUCC("SolveIn(eTED_LLDT)");
          }
       }
       else
       {
          aWRes.EW() = aWThis.EW().ldlt().solve(aWMat.EW());
       }
    }
    else
    {
        MMVII_INTERNAL_ASSERT_always(false,"Unkown type eigen decomposition");
    }

    // return aRes;
}

template <class Type> cDenseVect<Type>  cDenseMatrix<Type>::SolveColumn(const tDV & aVect,eTyEigenDec aTED) const
{
    tMat::CheckSquare(*this);
    tMat::TplCheckSizeX(aVect.Sz());

    tDV aRes(aVect.Sz());

    tConst_EW aWThis(*this);
    cConst_EigenColVectWrap<Type> aWVect(aVect);
    cNC_EigenColVectWrap<Type> aWRes(aRes);

    if (aTED == eTyEigenDec::eTED_PHQR)
    {
       if (EigenDoTestSuccess())
       {
          auto solver = aWThis.EW().colPivHouseholderQr();
          aWRes.EW() = solver.solve(aWVect.EW());
          if (solver.info()!=Eigen::Success)
          {
             ON_EIGEN_NO_SUCC("SolveColVect(eTED_PHQR)");
          }
       }
       else
       {
          aWRes.EW() = aWThis.EW().colPivHouseholderQr().solve(aWVect.EW());
       }
    }
    else if (aTED == eTyEigenDec::eTED_LLDT)
    {
       if (EigenDoTestSuccess())
       {
          auto solver = aWThis.EW().ldlt();
          aWRes.EW() = solver.solve(aWVect.EW());
          if (solver.info()!=Eigen::Success)
          {
              ON_EIGEN_NO_SUCC("SolveColVect(eTED_LLDT)");
          }
       }
       else
       {
          aWRes.EW() = aWThis.EW().ldlt().solve(aWVect.EW());
       }
    }
    else
    {
        MMVII_INTERNAL_ASSERT_always(false,"Unkown type eigen decomposition");
    }

    return aRes;
}

template <class Type> cDenseVect<Type>  cDenseMatrix<Type>::SolveLine(const tDV & aVect,eTyEigenDec aTED) const
{
    tMat::CheckSquare(*this);
    tMat::TplCheckSizeY(aVect.Sz());

    tDV aRes(aVect.Sz());

    cConst_EigenTransposeMatWrap aWThis(*this);
    cConst_EigenColVectWrap<Type> aWVect(aVect);
    cNC_EigenColVectWrap<Type> aWRes(aRes);

    if (aTED == eTyEigenDec::eTED_PHQR)
    {
       if (EigenDoTestSuccess())
       {
          auto solver = aWThis.EW().colPivHouseholderQr();
          aWRes.EW() = solver.solve(aWVect.EW());
          if (solver.info()!=Eigen::Success)
          {
             ON_EIGEN_NO_SUCC("SolveLine(eTED_PHQR)");
          }
       }
       else
       {
           aWRes.EW() = aWThis.EW().colPivHouseholderQr().solve(aWVect.EW());
       }
    }
    else if (aTED == eTyEigenDec::eTED_LLDT)
    {
       if (EigenDoTestSuccess())
       {
           auto solver = aWThis.EW().ldlt();
           aWRes.EW() = solver.solve(aWVect.EW());
           if (solver.info()!=Eigen::Success)
           {
               ON_EIGEN_NO_SUCC("SolveLine(eTED_LLDT)");
           }
       }
       else
       {
              aWRes.EW() = aWThis.EW().ldlt().solve(aWVect.EW());
       }
    }
    else
    {
        MMVII_INTERNAL_ASSERT_always(false,"Unkown type eigen decomposition");
    }

    return aRes;
}



/* ===================================================== */
/* =====              INSTANTIATION                ===== */
/* ===================================================== */

#define INSTANTIATE_ORTHOG_DENSE_MATRICES(Type)\
template  class  cResulSVDDecomp<Type>;\
template  class  cDenseMatrix<Type>;\
template  class  cResulSymEigenValue<Type>;\
template  class  cResulQR_Decomp<Type>;\

INSTANTIATE_ORTHOG_DENSE_MATRICES(tREAL4)
INSTANTIATE_ORTHOG_DENSE_MATRICES(tREAL8)
INSTANTIATE_ORTHOG_DENSE_MATRICES(tREAL16)


};
