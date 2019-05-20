#include "include/MMVII_all.h"

namespace MMVII
{


void  Bench_0000_Ptxd()
{
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

    std::cout << cPt1dr(3) << "\n";


    cRect2 aR(cPt2di(10,20),cPt2di(50,40));
    MMVII_INTERNAL_ASSERT_bench(aR.Proj(cPt2di(11,22)) == cPt2di(11,22),"Bench_0000_Ptxd");
    MMVII_INTERNAL_ASSERT_bench(aR.Proj(cPt2di(1,2)) == cPt2di(10,20),"Bench_0000_Ptxd");
    MMVII_INTERNAL_ASSERT_bench(aR.Proj(cPt2di(1,22)) == cPt2di(10,22),"Bench_0000_Ptxd");
    MMVII_INTERNAL_ASSERT_bench(aR.Proj(cPt2di(11,19)) == cPt2di(11,20),"Bench_0000_Ptxd");
    MMVII_INTERNAL_ASSERT_bench(aR.Proj(cPt2di(100,100)) == cPt2di(49,39),"Bench_0000_Ptxd");

    StdOut() << "done Bench_0000_Ptxd \n";
}

};

