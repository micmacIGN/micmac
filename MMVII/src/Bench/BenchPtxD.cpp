#include "include/MMVII_all.h"

namespace MMVII
{


template <const int Dim> void BenchTabGrow(const int aDMax)
{
     const std::vector<std::vector<cPtxd<int,Dim>>> &  aRes = TabGrowNeigh<Dim>(aDMax);

     MMVII_INTERNAL_ASSERT_bench((int(aRes.size())>aDMax),"SZ BenchTabGrow");

     for (int aD=0 ; aD<= aDMax ; aD++)
     {
          //  Test good number of points
          int aNbTh =  (aD>0)  ? round_ni(pow(1+2*aD,Dim)-pow(-1+2*aD,Dim))  : 1;
          MMVII_INTERNAL_ASSERT_bench((aNbTh==int(aRes.at(aD).size())),"Dist in BenchTabGrow");
 
          for (const auto & aP :  aRes.at(aD))
          {
              // Test good inf-norm
              MMVII_INTERNAL_ASSERT_bench((aD==NormInf(aP)),"Dist in BenchTabGrow");

              // Test all diff
              int aNbEq = 0;
              for (const auto & aP2 :  aRes.at(aD))
                  if (aP==aP2)
                     aNbEq++;
              MMVII_INTERNAL_ASSERT_bench(aNbEq==1,"Diff in BenchTabGrow");
          }
     }
}

void  Bench_cPt2dr()
{
   for (const int aDist : {0,1,2,3,6,2,0} )
   {
       BenchTabGrow<1>(aDist);
       BenchTabGrow<2>(aDist);
       BenchTabGrow<3>(aDist);
   }
   {
      const std::vector<cPt3di> & aV31 = AllocNeighbourhood<3>(1);
      MMVII_INTERNAL_ASSERT_bench((&AllocNeighbourhood<3>(2)==&AllocNeighbourhood<3>(2)),"Alloc Neighbour");
      MMVII_INTERNAL_ASSERT_bench((&AllocNeighbourhood<3>(1)==&aV31),"Alloc Neighbour");
      MMVII_INTERNAL_ASSERT_bench((&AllocNeighbourhood<3>(2)!=&aV31),"Alloc Neighbour");
      MMVII_INTERNAL_ASSERT_bench((aV31.size()==6),"Alloc Neighbour");
      MMVII_INTERNAL_ASSERT_bench((AllocNeighbourhood<3>(2).size()==18),"Alloc Neighbour");
      MMVII_INTERNAL_ASSERT_bench((AllocNeighbourhood<3>(3).size()==26),"Alloc Neighbour");
   }
   // Bench Polar function is correct on some test values
   MMVII_INTERNAL_ASSERT_bench(Norm2(cPt2dr(1,1)-FromPolar(sqrt(2.0),M_PI/4.0))<1e-5,"cPt2r Bench");
   MMVII_INTERNAL_ASSERT_bench(Norm2(cPt2dr(-2,0)-FromPolar(2,-M_PI))<1e-5,"cPt2r Bench");
   MMVII_INTERNAL_ASSERT_bench(Norm2(cPt2dr(0,2)-FromPolar(2,M_PI/2.0))<1e-5,"cPt2r Bench");
   MMVII_INTERNAL_ASSERT_bench(Norm2(cPt2dr(0,-2)-FromPolar(2,3*M_PI/2.0))<1e-5,"cPt2r Bench");
   MMVII_INTERNAL_ASSERT_bench(Norm2(cPt2dr(0,-2)-FromPolar(2,7*M_PI/2.0))<1e-5,"cPt2r Bench");
   for (int aK=0 ; aK< 100; aK++)
   {

        double aRho1 = 1e-5 + RandUnif_0_1();
        double aTeta1 =  100.0 * RandUnif_C();
        cPt2dr aP1 = FromPolar(aRho1,aTeta1);

        // Bench Rho
        MMVII_INTERNAL_ASSERT_bench(std::abs(aRho1 - Norm2(aP1))<1e-5,"cPt2r Bench");

        cPt2dr aP1Bis = FromPolar(ToPolar(aP1));
        // Bench FromPolar and ToPolar are invert
        MMVII_INTERNAL_ASSERT_bench(Norm2(aP1-aP1Bis)<1e-5,"cPt2r Bench");

        double aRho2 = 1e-5 + RandUnif_0_1();
        double aTeta2 =  100.0 * RandUnif_C();
        cPt2dr aP2 = FromPolar(aRho2,aTeta2);
        cPt2dr aP1m2 = aP1 * aP2;
        cPt2dr aQ1m2 = FromPolar(aRho1*aRho2,aTeta1+aTeta2);
        // Bench mul complex vis rho-teta
        MMVII_INTERNAL_ASSERT_bench(Norm2(aP1m2 - aQ1m2)<1e-5,"cPt2r Bench");

        cPt2dr aP1d2 = aP1 / aP2;
        cPt2dr aQ1d2 = FromPolar(aRho1/aRho2,aTeta1-aTeta2);
        // Bench div complex vis rho-teta
        MMVII_INTERNAL_ASSERT_bench(Norm2(aP1d2 - aQ1d2)<1e-5,"cPt2r Bench");

        //  StdOut() << "CcccMul " << aP1d2 - aQ1d2 << "\n";
   }
}

template <class TypePt> void  TestDist(TypePt aPt,double aN1,double aN2,double aNInf)
{
   MMVII_INTERNAL_ASSERT_bench(std::abs(Norm1(aPt)-aN1)<1e-10,"Norm2");
   MMVII_INTERNAL_ASSERT_bench(std::abs(NormK(aPt,1.0)-aN1)<1e-10,"Norm2");
   MMVII_INTERNAL_ASSERT_bench(std::abs(Norm2(aPt)-aN2)<1e-10,"Norm2");
   MMVII_INTERNAL_ASSERT_bench(std::abs(NormK(aPt,2.0)-aN2)<1e-10,"Norm2");
   MMVII_INTERNAL_ASSERT_bench(std::abs(NormInf(aPt)-aNInf)<1e-10,"Norm2");

   // As it only an approx a inf, low threshold
   // std::cout << "DIFF= " <<  NormK(aPt,100.0)-aNInf << "\n";
   MMVII_INTERNAL_ASSERT_bench(std::abs(NormK(aPt,100.0)-aNInf)<1e-5,"Norm2");


   MMVII_INTERNAL_ASSERT_bench(std::abs(Square(Norm2(aPt))- Scal(aPt,aPt))<1e-5,"Norm2");

}

void  Bench_0000_Ptxd(cParamExeBench & aParam)
{
    if (! aParam.NewBench("Ptxd")) return;
    
    {
        TestDist(cPt1dr (-8),8,8,8);
        TestDist(cPt2dr (-3,4),7,5,4);
        TestDist(cPt3dr (-3,0,4),7,5,4);
        TestDist(cPtxd<float,4> (1,1,1,2),5,sqrt(7),2);
    }
    
    Bench_cPt2dr();
    cPt1dr aA1(1);
    cPt2dr aA2(1,2);
    cPt3dr aA3(1,2,3);
    // Test x,y,z
    MMVII_INTERNAL_ASSERT_bench(aA1.x()==1,"Bench_0000_Ptxd");
    MMVII_INTERNAL_ASSERT_bench((aA2.x()==1)&&(aA2.y()==2),"Bench_0000_Ptxd");
    MMVII_INTERNAL_ASSERT_bench((aA3.x()==1)&&(aA3.y()==2)&&(aA3.z()==3),"Bench_0000_Ptxd");

    MMVII_INTERNAL_ASSERT_bench(aA1==cPt1dr(1),"Bench_0000_Ptxd");
    MMVII_INTERNAL_ASSERT_bench(aA1!=cPt1dr(1.1),"Bench_0000_Ptxd");
    MMVII_INTERNAL_ASSERT_bench(aA1==cPt1dr(8)+cPt1dr(-7),"Bench_0000_Ptxd");
    MMVII_INTERNAL_ASSERT_bench(aA1==cPt1dr(6)-cPt1dr(5),"Bench_0000_Ptxd");
    MMVII_INTERNAL_ASSERT_bench((aA1*2.0)==cPt1dr(2),"Bench_0000_Ptxd");
    MMVII_INTERNAL_ASSERT_bench((2.5*aA1)==cPt1dr(2.5),"Bench_0000_Ptxd");


    MMVII_INTERNAL_ASSERT_bench(aA2==cPt2dr(2,1)+cPt2dr(-1,1),"Bench_0000_Ptxd");
    MMVII_INTERNAL_ASSERT_bench(aA2==cPt2dr(2,3)-cPt2dr(1,1),"Bench_0000_Ptxd");
    MMVII_INTERNAL_ASSERT_bench(aA2!=cPt2dr(1,1),"Bench_0000_Ptxd");
    MMVII_INTERNAL_ASSERT_bench(aA2!=cPt2dr(2,2),"Bench_0000_Ptxd");
    MMVII_INTERNAL_ASSERT_bench((aA2*2.0)==cPt2dr(2,4),"Bench_0000_Ptxd");
    MMVII_INTERNAL_ASSERT_bench((2.5*aA2)==cPt2dr(2.5,5.0),"Bench_0000_Ptxd");


    MMVII_INTERNAL_ASSERT_bench(aA3==cPt3dr(10,10,10)+cPt3dr(-9,-8,-7),"Bench_0000_Ptxd");
    MMVII_INTERNAL_ASSERT_bench(aA3==cPt3dr(10,10,10)-cPt3dr(9,8,7),"Bench_0000_Ptxd");
    MMVII_INTERNAL_ASSERT_bench(aA3!=cPt3dr(1.1,2,3),"Bench_0000_Ptxd");
    MMVII_INTERNAL_ASSERT_bench(aA3!=cPt3dr(1,2.1,3),"Bench_0000_Ptxd");
    MMVII_INTERNAL_ASSERT_bench(aA3!=cPt3dr(1,2,3.1),"Bench_0000_Ptxd");
    MMVII_INTERNAL_ASSERT_bench(aA3*3.0==cPt3dr(3,6,9),"Bench_0000_Ptxd");
    MMVII_INTERNAL_ASSERT_bench(3.0*aA3==cPt3dr(3,6,9),"Bench_0000_Ptxd");

    MMVII_INTERNAL_ASSERT_bench(cPt2dr(1,1) == cPt2dr(1,1),"Bench_0000_Ptxd");
    MMVII_INTERNAL_ASSERT_bench(cPt2dr(1,1) != cPt2dr(1,2),"Bench_0000_Ptxd");
    MMVII_INTERNAL_ASSERT_bench(cPt2dr(1,1) != cPt2dr(2,1),"Bench_0000_Ptxd");

    cRect2 aR(cPt2di(10,20),cPt2di(50,40));
    MMVII_INTERNAL_ASSERT_bench(aR.Proj(cPt2di(11,22)) == cPt2di(11,22),"Bench_0000_Ptxd");
    MMVII_INTERNAL_ASSERT_bench(aR.Proj(cPt2di(1,2)) == cPt2di(10,20),"Bench_0000_Ptxd");
    MMVII_INTERNAL_ASSERT_bench(aR.Proj(cPt2di(1,22)) == cPt2di(10,22),"Bench_0000_Ptxd");
    MMVII_INTERNAL_ASSERT_bench(aR.Proj(cPt2di(11,19)) == cPt2di(11,20),"Bench_0000_Ptxd");
    MMVII_INTERNAL_ASSERT_bench(aR.Proj(cPt2di(100,100)) == cPt2di(49,39),"Bench_0000_Ptxd");



    aParam.EndBench();
}

};

