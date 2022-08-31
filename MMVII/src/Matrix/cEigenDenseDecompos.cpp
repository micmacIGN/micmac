#include "include/MMVII_all.h"
#include "include/MMVII_Tpl_Images.h"

#include "MMVII_EigenWrap.h"
#include "ExternalInclude/Eigen/Eigenvalues" 
#include "ExternalInclude/Eigen/Householder"  // HouseholderQR.h"
// #include "ExternalInclude/Eigen/Cholesky"  // HouseholderQR.h"

using namespace Eigen;

namespace MMVII
{


template <class Type> cResulSVDDecomp<Type> cDenseMatrix<Type>::SVD() const
{
   this->CheckSquare(*this);  // this for scope, method is static
   int aNb = Sz().x();
   cResulSVDDecomp<Type> aRes(aNb);

   tConst_EW aWrap(*this);
   JacobiSVD<typename tNC_EW::tEigenMat > aJacSVD(aWrap.EW(),ComputeThinU | ComputeThinV);

   cNC_EigenMatWrap<Type> aWrap_U(aRes.mMatU);
   aWrap_U.EW() = aJacSVD.matrixU();

   cNC_EigenMatWrap<Type> aWrap_V(aRes.mMatV);
   aWrap_V.EW() = aJacSVD.matrixV();

   cNC_EigenColVectWrap<Type>  aWrapSVal(aRes.mSingularValues);
   aWrapSVal.EW() =  aJacSVD.singularValues();

   return aRes;
}

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
   cWhitchMinMax<int,Type> aIMM(0,std::abs(mEigenValues(0)));
   for (int aK=1 ; aK<mEigenValues.Sz()  ; aK++)
   {
          aIMM.Add(aK,std::abs(mEigenValues(aK)));
   }
   if (aIMM.Max().ValExtre() == Type(0.0))
   {
       MMVII_INTERNAL_ASSERT_strong(aDef>=0,"Conditionning of null eigen value without default");
       return aDef;
   }
   return  aIMM.Min().ValExtre() / aIMM.Max().ValExtre() ;
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
template <class Type> cDenseMatrix<Type>  cDenseMatrix<Type>::Solve(const tDM & aMat,eTyEigenDec aTED) const
{
    tDM aRes(aMat.Sz().x(),aMat.Sz().y());
    SolveIn(aRes,aMat,aTED);
    return aRes;
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





/*
*/
/*
   tDM  Solve(const tDM &) const;
        tDV  Solve(const tDV &) const;



   Matrix3f A;
   Vector3f b;
   A << 1,2,3,  4,5,6,  7,8,10;
   b << 3, 3, 4;
   cout << "Here is the matrix A:\n" << A << endl;
   cout << "Here is the vector b:\n" << b << endl;
   Vector3f x = A.colPivHouseholderQr().solve(b);
   cout << "The solution is:\n" << x << endl;
*/


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
        aResult.AddObservation(anEl.mW,anEl.mCoeff,anEl.mCste);

    return aResult;
}




/* ===================================================== */
/* =====              INSTANTIATION                ===== */
/* ===================================================== */


#define INSTANTIATE_ORTHOG_DENSE_MATRICES(Type)\
template  class  cElemDecompQuad<Type>;\
template  class  cDecSumSqLinear<Type>;\
template  class  cResulSVDDecomp<Type>;\
template  class  cStrStat2<Type>;\
template  class  cDenseMatrix<Type>;\
template  class  cResulSymEigenValue<Type>;\
template  class  cResulQR_Decomp<Type>;

INSTANTIATE_ORTHOG_DENSE_MATRICES(tREAL4)
INSTANTIATE_ORTHOG_DENSE_MATRICES(tREAL8)
INSTANTIATE_ORTHOG_DENSE_MATRICES(tREAL16)


};
