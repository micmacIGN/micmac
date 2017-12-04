#include "include/MMVII_all.h"

namespace MMVII
{



/// Bench that aModb is the mathematicall definition
void BenchMod(int A,int B,int aModb)
{
     MMVII_INTERNAL_ASSERT_bench(aModb>=0,"BenchMod-1");
     MMVII_INTERNAL_ASSERT_bench(aModb<std::abs(B),"BenchMod-2");

     int AmB = A - aModb;  // AmB => A multiple de B
     MMVII_INTERNAL_ASSERT_bench((AmB/B)*B == AmB ,"BenchMod-3");
}


void Bench_Nums()
{
   std::cout << "Bench_NumsBench_NumsBench_NumsBench_Nums\n";
   MMVII_INTERNAL_ASSERT_bench (sizeof(tREAL8)==8,"Bench round_up");
   MMVII_INTERNAL_ASSERT_bench (sizeof(tINT4)==4 ,"Bench round_up");
   MMVII_INTERNAL_ASSERT_bench (sizeof(tINT8)==8 ,"Bench round_up");
   /// Bench modulo

   for (int A=-20 ; A<=20 ; A++)
   {
      for (int B=-20 ; B<=20 ; B++)
      {
         if (B!=0)
         {
            // BenchMod(A,B,mod(A,B));
            // BenchMod(A,B,mod_gen(A,B));
            double aRatio = double(A) / double(B);

            int rup = round_up(aRatio);
            MMVII_INTERNAL_ASSERT_bench ((rup>=aRatio) &&((rup-1)<aRatio),"Bench round_up");
            int ruup = round_Uup(aRatio);
            MMVII_INTERNAL_ASSERT_bench ((ruup>aRatio) &&((ruup-1)<=aRatio),"Bench round_up");
            
            int rd = round_down(aRatio);
            MMVII_INTERNAL_ASSERT_bench ((rd<=aRatio) &&((rd+1)>aRatio),"Bench round_up");
            int rdd = round_Ddown(aRatio);
            MMVII_INTERNAL_ASSERT_bench ((rdd<aRatio) &&((rd+1)>=aRatio),"Bench round_up");

            int ri = round_ni(aRatio);
            MMVII_INTERNAL_ASSERT_bench ((ri<=aRatio+0.5) &&(ri>aRatio-0.5),"Bench round_up");

            BenchMod(A,B,mod_gen(A,B));
            if (B>0)
               BenchMod(A,B,mod(A,B));
         }
      }
   }
}

};

