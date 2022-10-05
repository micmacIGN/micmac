#include "TrianguRSNL.h"


namespace MMVII
{

/** \file BenchNetwRSNL.cpp
    \brief test the non linear system solver with triangulation on a network of point

    Make one test with different parameter, check that after given number iteration we are sufficiently close
    to "real" network

   [0]   void BenchSSRNL(cParamExeBench & aParam)  :
         global function, declared in header make  call to OneBenchSSRNL with various
         parameters  (size of network, type of underlying matrix, parameters for these matrices)

   [1]   void  OneBenchSSRNL(eModeSSR aMode,int aNb,bool WithSchurr,cParamSparseNormalLstSq * aParam=nullptr)
         for a given set of parameters , test the different template instanciation (REAL on 4,8,16 bytes)
          
    
   [2] template<class Type> void  TplOneBenchSSRNL(....)
        do the job for a  given type and given set of parameters, 
                *   construct the network, 
                *  iterate gauss newton, 
                *  check we reach the theoreticall solution
*/


/* ======================================== */
/*                                          */
/*              ::                          */
/*                                          */
/* ======================================== */
using namespace NS_Bench_RSNL;
static constexpr int    THE_NB_ITER  = 10;



template<class Type> void  TplOneBenchSSRNL
                           (
                               eModeSSR aMode,
                               cRect2 aRect,
                               bool WithSchurr,
                               cParamSparseNormalLstSq * aParam=nullptr,
			       const std::vector<Type> &  aWeightSetSchur ={0.0,0.0,0.0,0.0}
                           )
{
     static int aCpt=0 ; aCpt++;  // use to make jauge constant along one iteration, to check the correctnes of hard constraints
     cParamMainNW aParamNW;
     Type aPrec = tElemNumTrait<Type>::Accuracy() ;
     cMainNetwork <Type> aBN(aMode,aRect,WithSchurr,aParamNW,aParam,aWeightSetSchur);
     aBN.PostInit();
     double anEc =100;
     for (int aK=0 ; aK < THE_NB_ITER ; aK++)
     {
         double aWGauge = (aCpt%2) ? -1 : 100; // alternate "hard" constraint and soft, to test more ..
         anEc = aBN.DoOneIterationCompensation(aWGauge,true);
     }
     if (anEc>aPrec)
     {
           StdOut() << "Fin-ECc== " << anEc /aPrec   << "\n";
           MMVII_INTERNAL_ASSERT_bench(anEc<aPrec,"Error in Network-SSRNL Bench");
     }
}
template<class Type> void  TplOneBenchSSRNL
                           (
                               eModeSSR aMode,
                               int aNb,
                               bool WithSchurr,
                               cParamSparseNormalLstSq * aParam=nullptr,
			       const std::vector<Type> &  aWeightSetSchur = {0.0,0.0,0.0,0.0}
			   )
{
	TplOneBenchSSRNL<Type>(aMode,cRect2::BoxWindow(aNb),WithSchurr,aParam,aWeightSetSchur);
}

void  OneBenchSSRNL(eModeSSR aMode,int aNb,bool WithSchurr,cParamSparseNormalLstSq * aParam=nullptr)
{
    TplOneBenchSSRNL<tREAL8>(aMode,cBox2di(cPt2di(0,0),cPt2di(2,2)),false,aParam);

    TplOneBenchSSRNL<tREAL8>(aMode,aNb,WithSchurr,aParam);
    TplOneBenchSSRNL<tREAL16>(aMode,aNb,WithSchurr,aParam);
    TplOneBenchSSRNL<tREAL4>(aMode,aNb,WithSchurr,aParam);
}



void BenchSSRNL(cParamExeBench & aParam)
{
     if (! aParam.NewBench("SSRNL")) return;
/*
     for (int aK=0 ; aK<10 ; aK++)
     {
         cParamMainNW aParamNW;
         cMainNetwork <tREAL8> aNet(eModeSSR::eSSR_LsqDense,2,false,aParamNW);

	 aNet.TestCov();
     }
*/

     // Test with non centered netowrk 
     TplOneBenchSSRNL<tREAL8>(eModeSSR::eSSR_LsqDense,cBox2di(cPt2di(0,0),cPt2di(2,2)),false);


     // some minimal test
     OneBenchSSRNL(eModeSSR::eSSR_LsqDense ,1,false);
     OneBenchSSRNL(eModeSSR::eSSR_LsqDense ,2,false);

     cParamSparseNormalLstSq aParamSq(3.0,4,9);
     // Basic test, test the 3 mode of matrix , with and w/o schurr subst, with different size
     for (const auto &  aNb : {3,4,5})
     {
	// w/o schurr
        OneBenchSSRNL(eModeSSR::eSSR_LsqNormSparse,aNb,false,&aParamSq);
        OneBenchSSRNL(eModeSSR::eSSR_LsqSparseGC,aNb,false);
        OneBenchSSRNL(eModeSSR::eSSR_LsqDense ,aNb,false);

	// with schurr
         OneBenchSSRNL(eModeSSR::eSSR_LsqNormSparse,aNb,true ,&aParamSq);
         OneBenchSSRNL(eModeSSR::eSSR_LsqDense ,aNb,true);
         OneBenchSSRNL(eModeSSR::eSSR_LsqSparseGC,aNb,true);

         //OneBenchSSRNL(eModeSSR::eSSR_LsqSparseGC,aNb,true);
     }
      //  soft constraint on temporary , add low cost to current (slow but do not avoid real value)
     TplOneBenchSSRNL<tREAL8>(eModeSSR::eSSR_LsqNormSparse,3,true ,&aParamSq,{1,1,0.1,0.1});
     TplOneBenchSSRNL<tREAL8>(eModeSSR::eSSR_LsqDense     ,3,true ,nullptr  ,{1,1,0.1,0.1});
     TplOneBenchSSRNL<tREAL8>(eModeSSR::eSSR_LsqSparseGC  ,3,true ,nullptr  ,{1,1,0.1,0.1});

     // mix hard & soft constraint on temporary
     TplOneBenchSSRNL<tREAL8>(eModeSSR::eSSR_LsqNormSparse,3,true ,&aParamSq,{-1,1,0,0});
     TplOneBenchSSRNL<tREAL8>(eModeSSR::eSSR_LsqDense     ,3,true ,nullptr  ,{1,-1,0,0});
     TplOneBenchSSRNL<tREAL8>(eModeSSR::eSSR_LsqSparseGC  ,3,true ,nullptr  ,{-1,-1,0,0});

     // test  normal sparse matrix with many parameters handling starsity
     for (int aK=0 ; aK<20 ; aK++)
     {
        int aNb = 3+ RandUnif_N(3);
	int aNbVar = 2 * Square(2*aNb+1);
        cParamSparseNormalLstSq aParamSq(3.0,RandUnif_N(3),RandUnif_N(10));

	// add random subset of dense variable
	for (const auto & aI:  RandSet(aNbVar/10,aNbVar))
           aParamSq.mVecIndDense.push_back(size_t(aI));

        TplOneBenchSSRNL<tREAL8>(eModeSSR::eSSR_LsqNormSparse,aNb,false,&aParamSq); // w/o schurr
        TplOneBenchSSRNL<tREAL8>(eModeSSR::eSSR_LsqNormSparse,aNb,true ,&aParamSq); // with schurr
     }


     aParam.EndBench();
}


};
