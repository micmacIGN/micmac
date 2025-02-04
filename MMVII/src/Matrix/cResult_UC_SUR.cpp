#include "MMVII_Tpl_Images.h"
#include "MMVII_SysSurR.h"
#include "LinearConstraint.h"


using namespace NS_SymbolicDerivative;
using namespace MMVII;

namespace MMVII
{

/* ************************************************************ */
/*                                                              */
/*                cResultSUR                                    */
/*                                                              */
/* ************************************************************ */

template <class Type>  
  cResult_UC_SUR<Type>::cResult_UC_SUR
  (
       tRSNL *aRSNL,
       bool  addAllVar,
       bool computNormalM,
       const std::vector<int>  & aVIndUC2Compute,
       const std::vector<cSparseVect<Type>> &  aVLinearCstr
  ) :
      mCompiled           (false),
      mRSNL               (aRSNL),
      mDim                (mRSNL->NbVar()),
      mSysL               (mRSNL->SysLinear()),
      mDebug              (false),
      mNormalM_Compute    (computNormalM),
      mVectCombLin        (aVLinearCstr),
      mVectSol            (1),
      mMatSols            (1),
      mNormalMatrix       (1),
      mGlobUncertMatrix   (1)
{
   if (addAllVar)
   {
       for (int aK=0; aK<mDim ; aK++)
           mIndexUC_2Compute.Add(aK);
   }
   else 
   {
       for (const auto & aK : aVIndUC2Compute)
           mIndexUC_2Compute.Add(aK);
   }
}

template <class Type> Type  cResult_UC_SUR<Type>::FUV() const {return mFUV;}

// template <class Type> void cResult_UC_SUR<Type>::SetDoComputeNormalMatrix(bool doIt ) {mNormalM_Compute=doIt;}
template <class Type>  cDenseMatrix<Type>  cResult_UC_SUR<Type>::NormalMatrix() const
{
      MMVII_INTERNAL_ASSERT_tiny(mNormalM_Compute&&mCompiled,"NormalMatrix not computed");
      return mNormalMatrix;
}



std::pair<int,int>  IntervalAfter(const std::pair<int,int> & aInt0, int aSz)
{
     return std::pair<int,int> (aInt0.second,aInt0.second+aSz);
}



template <class Type>  Type  cResult_UC_SUR<Type>::UK_VarCovarEstimate(int aK1,int aK2) const
{
    aK1 = mIndexUC_2Compute.Obj2I(aK1) + mIndEndSol;

    return mMatSols.GetElem(aK1,aK2) * mFUV ;
}

template <class Type>  Type  cResult_UC_SUR<Type>::CombLin_VarCovarEstimate(int aK1,int aK2) const
{
   return  mMatSols.DotProduct_Col(mIndEndUC+aK1,mVectCombLin.at(aK2)) * mFUV;
}


template <class Type>  void  cResult_UC_SUR<Type>::Compile()
{
   mNbObs          =  mRSNL->GetCurNbObs();
   mNbCstr         =  mRSNL->GetNbLinearConstraints();
   mRatioDOF       =  (mNbObs+mNbCstr)/double(mNbObs-(mDim-mNbCstr)) ;

   mInd0            = 0;
   mIndEndSol       = 1;
   mIndEndUC        = mIndEndSol + mIndexUC_2Compute.size();
   mIndEndCombLin   = mIndEndUC  + mVectCombLin.size();


   cDenseMatrix<Type> aM2Solve(mIndEndCombLin,mDim,eModeInitImage::eMIA_Null);
   // Put RHS for computing solution to system
   aM2Solve.WriteCol(0,mSysL->V_tARhs());

   // Put (partial) identity matrix to compute var on unknowns
   for (int aK = mIndEndSol ; aK<mIndEndUC ; aK++)
   {
        aM2Solve.SetElem(aK, *mIndexUC_2Compute.I2Obj(aK-mIndEndSol)   ,1.0);
   }

   // Put linear combination for computing var/covar on them
   for (int aK = mIndEndUC ; aK<mIndEndCombLin ; aK++)
   {
        aM2Solve.WriteCol(aK,mVectCombLin.at(aK-mIndEndUC));
   }

   mMatSols = mSysL->tAA_Solve(aM2Solve);
   mVectSol = mMatSols.ReadCol(0) ;// + mCurGlobSol ;


   mVarianceCur    =  mSysL->VarOfSol(mVectSol);
   mFUV = mRatioDOF * mVarianceCur;

   if (mNormalM_Compute)
   {
      mNormalMatrix =  mSysL->V_tAA();
   }
   
   mCompiled       = true;
}


#define INSTANTIATE_RESULT_UC_SUR(TYPE)\
template class  cResult_UC_SUR<TYPE>;

INSTANTIATE_RESULT_UC_SUR(tREAL4)
INSTANTIATE_RESULT_UC_SUR(tREAL8)
INSTANTIATE_RESULT_UC_SUR(tREAL16)


};
