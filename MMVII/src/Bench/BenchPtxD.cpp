#include "include/MMVII_all.h"

namespace MMVII
{


void  Bench_0000_Ptxd()
{
    MMVII_INTERNAL_ASSERT_bench(cPt2dr(1,1) == cPt2dr(1,1),"Bench_0000_Ptxd");
    MMVII_INTERNAL_ASSERT_bench(cPt2dr(1,1) != cPt2dr(1,2),"Bench_0000_Ptxd");
    MMVII_INTERNAL_ASSERT_bench(cPt2dr(1,1) != cPt2dr(2,1),"Bench_0000_Ptxd");

    StdOut() << "done Bench_0000_Ptxd \n";
}

};

