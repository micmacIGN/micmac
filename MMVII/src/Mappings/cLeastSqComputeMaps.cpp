#include "SymbDer/SymbDer_Common.h"

#include "MMVII_PhgrDist.h"

using namespace NS_SymbolicDerivative;

namespace MMVII
{

/* ============================================= */
/*      cLeastSqComputeMaps<Type>                */
/* ============================================= */

template <class Type,const int  DimIn,const int DimOut> 
     cLeastSqComputeMaps<Type,DimIn,DimOut>::cLeastSqComputeMaps(size_t aNbFunc) :
        mNbFunc  (aNbFunc),
        mLSQ     (aNbFunc),
        mCoeffs  (aNbFunc),
        mBufPOut (aNbFunc)
{
}

template <class Type,const int  DimIn,const int DimOut> 
    cLeastSqComputeMaps<Type,DimIn,DimOut>::~cLeastSqComputeMaps()
{
}
template <class Type,const int  DimIn,const int DimOut> 
    void  cLeastSqComputeMaps<Type,DimIn,DimOut>::ComputeSolNotClear(std::vector<Type>& aRes)
{
     cDenseVect<Type> aVD =  mLSQ.Solve();
     aVD.DIm().DupInVect(aRes);
}

template <class Type,const int  DimIn,const int DimOut> 
    void  cLeastSqComputeMaps<Type,DimIn,DimOut>::ComputeSol(std::vector<Type>& aRes)
{
    ComputeSolNotClear(aRes);
    mLSQ.Reset();
}


template <class Type,const int  DimIn,const int DimOut> 
     void cLeastSqComputeMaps<Type,DimIn,DimOut>::AddObs
            (const tPtIn & aPtIn,const tPtOut & aValue,const tPtOut & aPds)
{
    ComputeValFuncsBase(mBufPOut,aPtIn);
    cDataIm1D<Type> & aDIm = mCoeffs.DIm();

    for (size_t aD=0 ; aD<DimOut ; aD++)
    {
       for (size_t aKFunc=0 ; aKFunc<mNbFunc ; aKFunc++)
       {
           aDIm.SetV(aKFunc,mBufPOut.at(aKFunc)[aD]);
       }
       mLSQ.AddObservation(aPds[aD],mCoeffs,aValue[aD]);
    }
}

template <class Type,const int  DimIn,const int DimOut> 
     void cLeastSqComputeMaps<Type,DimIn,DimOut>::AddObs
            (const tPtIn & aPtIn,const tPtOut & aValue,const Type & aPds)
{
     AddObs(aPtIn,aValue,tPtOut::PCste(aPds));
}

template <class Type,const int  DimIn,const int DimOut> 
     void cLeastSqComputeMaps<Type,DimIn,DimOut>::AddObs
            (const tPtIn & aPtIn,const tPtOut & aValue)
{
     AddObs(aPtIn,aValue,1.0);
}

/* ============================================= */
/*      cLeastSqCompMapCalcSymb<Type>            */
/* ============================================= */


template <class Type,const int  DimIn,const int DimOut> 
     cLeastSqCompMapCalcSymb<Type,DimIn,DimOut>::cLeastSqCompMapCalcSymb(tCalc * aCalc) :
       cLeastSqComputeMaps<Type,DimIn,DimOut>(size_t(aCalc->NbElem()/DimOut)),
       // cLeastSqComputeMaps<Type,DimIn,DimOut>(mCalc->NbElem()/DimOut),
       // cLeastSqComputeMaps<Type,DimIn,DimOut>(TTTTT(mCalc->NbElem(),DimOut)),
       mCalc (aCalc),
       mVUk  (DimIn),
       mVObs (0)
{
    MMVII_INTERNAL_ASSERT_strong(!mCalc->WithDer(),"LeastSqSymb with Derivatives");
    MMVII_INTERNAL_ASSERT_strong(DimIn==mCalc->NbUk(),"LeastSqSymb incoh DimIn/NbUk");
    MMVII_INTERNAL_ASSERT_strong(0==mCalc->NbObs(),"LeastSqSymb NoObs required");
    MMVII_INTERNAL_ASSERT_strong(this->NbFunc()*DimOut==mCalc->NbElem(),"LeastSqSymb incoh elem/Dim/Func");
}

template <class Type,const int  DimIn,const int DimOut> 
  void  cLeastSqCompMapCalcSymb<Type,DimIn,DimOut>::ComputeValFuncsBase(tVecOut & aResPt,const tPtIn & aPt) 
{
    MMVII_INTERNAL_ASSERT_medium(mCalc->NbInBuf()==0,"Buf should be empty");
    for (size_t aD=0 ; aD<DimIn; aD++)
       mVUk.at(aD) = aPt[aD];
    mCalc->PushNewEvals(mVUk,mVObs);
    const std::vector<std::vector<Type> *> & aVAllVal = mCalc->EvalAndClear();
    std::vector<Type>  & aResVal = *(aVAllVal[0]);

    size_t aKVal=0;
    for (size_t aKDim=0 ; aKDim<DimOut ; aKDim++)
    {
        for (auto & aP : aResPt)
        {
            aP[aKDim] = aResVal[aKVal++];
        }
    }
    MMVII_INTERNAL_ASSERT_tiny(aKVal==aResVal.size(),"Size in ComputeValFuncsBase");
}

template <class Type,const int  DimIn,const int DimOut> 
     cLeastSqCompMapCalcSymb<Type,DimIn,DimOut>::~cLeastSqCompMapCalcSymb()
{
}

/* ===================================================== */
/* =====              TEST                         ===== */
/* ===================================================== */

/*
*/

void BenchLeastSqMap(cParamExeBench & aParam)
{
    cPt3di aDeg(3,1,1);
   // const std::vector<cDescOneFuncDist>  & aVecD =  DescDist(aDeg);

    for (int aKTest=0 ; aKTest<100 ; aKTest++)
    {
       // ======== 1 Generate a random distorsion
           // 1-1 Distorsion 
       double aRhoMax = 5 * (0.01 +  RandUnif_0_1());
       double aProbaNotNul = 0.1 + (0.9 *RandUnif_0_1());  // No special added value to have many 0
       double aTargetSomJac = 2.0 * RandUnif_0_1();  // No problem if not invertible, but not to chaotic either
       cRandInvertibleDist aRID(aDeg,aRhoMax,aProbaNotNul,aTargetSomJac) ;
       
           //1-2 Make a Map of it
       int aNbParam = aRID.EqVal().NbObs();
       cDataMapCalcSymbDer<double,2,2> aMapDist(&aRID.EqVal(),&aRID.EqDer(),aRID.VParam(),false);

       // ======== 2  Compunte the least square estimation of MapDist
           // 2-1  initialise data for compunting

       int aNbPts = aNbParam * 10; // Over fit
       cCalculator<double> * anEqBase = EqBaseFuncDist(aDeg,aNbPts);  // Calculator for base of func
       cLeastSqCompMapCalcSymb<double,2,2> aLsqSymb(anEqBase);
       

           // 2-2  Fill the least sq with obs
       for (int aKPts=0 ; aKPts<aNbPts ; aKPts++)
       {
           cPt2dr aPIn = cPt2dr::PRandInSphere() * aRhoMax ;
           // Substract PIn because base of func is Map-Id
           cPt2dr aTarget = aMapDist.Value(aPIn)-aPIn;
           aLsqSymb.AddObs(aPIn,aTarget);  // 
       }

           // 2-3  compute solution
       std::vector<double> aParamCalc;
       aLsqSymb.ComputeSol(aParamCalc);
       cCalculator<double> * aCalcEqVal = EqDist(aDeg,false,aNbPts);  // Calculator for Vals  
       cCalculator<double> * aCalcEqDer = EqDist(aDeg,true ,aNbPts);   // Calculator for Der
       cDataMapCalcSymbDer<double,2,2> aMapCalc(aCalcEqVal,aCalcEqDer,aParamCalc,false);  // Map from computed vals

          // 3 Test solution
       
       for (int aKPts=0 ; aKPts<aNbPts ; aKPts++)
       {
           cPt2dr aPIn = cPt2dr::PRandInSphere() * aRhoMax ;
           cPt2dr aPOut1 = aMapDist.Value(aPIn);
           cPt2dr aPOut2 = aMapCalc.Value(aPIn);
           double aDif = Norm2(aPOut1 - aPOut2);
           MMVII_INTERNAL_ASSERT_bench(aDif<1e-5,"Dist by lsq");

       }

       delete aCalcEqVal;
       delete aCalcEqDer;
       delete anEqBase;
    }
}

/* ===================================================== */
/* =====              INSTANTIATION                ===== */
/* ===================================================== */

#define INSTANTIATE_LSQMAP(TYPE,DIMIN,DIMOUT)\
template class cLeastSqComputeMaps<TYPE,DIMIN,DIMOUT>;\
template class cLeastSqCompMapCalcSymb<TYPE,DIMIN,DIMOUT>;

INSTANTIATE_LSQMAP(tREAL8,3,2)
INSTANTIATE_LSQMAP(tREAL8,2,2)
INSTANTIATE_LSQMAP(tREAL8,3,3)






};
