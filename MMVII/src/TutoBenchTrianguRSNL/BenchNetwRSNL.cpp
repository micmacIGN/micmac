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
                               cParamSparseNormalLstSq * aParam=nullptr
                           )
{
     Type aPrec = tElemNumTrait<Type>::Accuracy() ;
     cMainNetwork <Type> aBN(aMode,aRect,WithSchurr,aParam);
     double anEc =100;
     for (int aK=0 ; aK < THE_NB_ITER ; aK++)
     {
         anEc = aBN.OneItereCompensation(false);
         // StdOut() << "  ECc== " << anEc /aPrec<< "\n";
     }
     // StdOut() << "Fin-ECc== " << anEc  / aPrec << " Nb=" << aNb << "\n";
     // getchar();
     MMVII_INTERNAL_ASSERT_bench(anEc<aPrec,"Error in Network-SSRNL Bench");
}
template<class Type> void  TplOneBenchSSRNL
                           (
                               eModeSSR aMode,
                               int aNb,
                               bool WithSchurr,
                               cParamSparseNormalLstSq * aParam=nullptr
			   )
{
	TplOneBenchSSRNL<Type>(aMode,cRect2::BoxWindow(aNb),WithSchurr,aParam);
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

     {
         cMainNetwork <tREAL8> aNet(eModeSSR::eSSR_LsqDense,5,false);

	 aNet.TestCov();
     }

     // Test with non centered netowrk 
     TplOneBenchSSRNL<tREAL8>(eModeSSR::eSSR_LsqDense,cBox2di(cPt2di(0,0),cPt2di(2,2)),false);


     // some minimal test
     OneBenchSSRNL(eModeSSR::eSSR_LsqDense ,1,false);
     OneBenchSSRNL(eModeSSR::eSSR_LsqDense ,2,false);

     // Basic test, test the 3 mode of matrix , with and w/o schurr subst, with different size
     for (const auto &  aNb : {3,4,5})
     {
        cParamSparseNormalLstSq aParamSq(3.0,4,9);
	// w/o schurr
        OneBenchSSRNL(eModeSSR::eSSR_LsqNormSparse,aNb,false,&aParamSq);
        OneBenchSSRNL(eModeSSR::eSSR_LsqSparseGC,aNb,false);
        OneBenchSSRNL(eModeSSR::eSSR_LsqDense ,aNb,false);

	// with schurr
         OneBenchSSRNL(eModeSSR::eSSR_LsqNormSparse,aNb,true ,&aParamSq);
         OneBenchSSRNL(eModeSSR::eSSR_LsqDense ,aNb,true);
         OneBenchSSRNL(eModeSSR::eSSR_LsqSparseGC,aNb,true);
     }


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
