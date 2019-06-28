#include "include/MMVII_all.h"
#include <boost/math/special_functions/fpclassify.hpp>

namespace MMVII
{

tINT4 HCF(tINT4 a,tINT4 b)
{
   a = std::abs(a);
   b = std::abs(b);

   OrderMinMax(a,b);
   while (a!=0)
   {
      tINT4 aInit = a;
      a = b%a;
      b = aInit;
   }
   return b;
}

template <class Type> const tNumTrait<Type>   tNumTrait<Type>::TheOnlyOne;
// const tNumTrait<tINT1>   tNumTrait<tINT1>::TheOnlyOne;
/*
template <> const tNumTrait<tINT2>   tNumTrait<tINT2>::TheOnlyOne;
template <> const tNumTrait<tINT4>   tNumTrait<tINT4>::TheOnlyOne;
template <> const tNumTrait<tINT8>   tNumTrait<tINT8>::TheOnlyOne;

template <> const tNumTrait<tU_INT1>   tNumTrait<tU_INT1>::TheOnlyOne;
template <> const tNumTrait<tU_INT2>   tNumTrait<tU_INT2>::TheOnlyOne;
template <> const tNumTrait<tU_INT4>   tNumTrait<tU_INT4>::TheOnlyOne;

template <> const tNumTrait<tREAL4>   tNumTrait<tREAL4>::TheOnlyOne;
template <> const tNumTrait<tREAL8>   tNumTrait<tREAL8>::TheOnlyOne;
template <> const tNumTrait<tREAL16>  tNumTrait<tREAL16>::TheOnlyOne;
*/


static const cVirtualTypeNum & SwitchFromEnum(eTyNums aTy)
{
   switch (aTy)
   {
      case eTyNums::eTN_INT1 : return tNumTrait<tINT1>::TheOnlyOne;
      case eTyNums::eTN_INT2 : return tNumTrait<tINT2>::TheOnlyOne;
      case eTyNums::eTN_INT4 : return tNumTrait<tINT4>::TheOnlyOne;
      case eTyNums::eTN_INT8 : return tNumTrait<tINT8>::TheOnlyOne;

      case eTyNums::eTN_U_INT1 : return tNumTrait<tU_INT1>::TheOnlyOne;
      case eTyNums::eTN_U_INT2 : return tNumTrait<tU_INT2>::TheOnlyOne;
      case eTyNums::eTN_U_INT4 : return tNumTrait<tU_INT4>::TheOnlyOne;

      case eTyNums::eTN_REAL4 :  return tNumTrait<tREAL4>::TheOnlyOne;
      case eTyNums::eTN_REAL8 :  return tNumTrait<tREAL8>::TheOnlyOne;
      case eTyNums::eTN_REAL16 : return tNumTrait<tREAL16>::TheOnlyOne;

      default : ;
   }
   MMVII_INTERNAL_ASSERT_strong(false,"Uknown value in cVirtualTypeNum::FromEnum");
   return tNumTrait<tINT1>::TheOnlyOne;
}

const cVirtualTypeNum & cVirtualTypeNum::FromEnum(eTyNums aTy)
{
    static std::vector<const cVirtualTypeNum *> aV;
    if (aV.empty())
    {
        for (int aK=0 ; aK<int(eTyNums::eNbVals) ; aK++)
        {
            aV.push_back(&SwitchFromEnum(eTyNums(aK)));
        }
    }
    return *(M_VectorAt(aV,int(aTy)));
}



template <class Type> void TplBenchTraits()
{
    //typename tNumTrait<Type>::tBase aVal=0;
    StdOut()  << E2Str(tNumTrait<Type>::TyNum() )
              << " Max=" << tNumTrait<Type>::MaxValue() 
              << " Min=" <<  tNumTrait<Type>::MinValue() 
              << " IsInt=" <<  tNumTrait<Type>::IsInt() 
              << "\n";

}

void BenchTraits()
{
   TplBenchTraits<tU_INT1>();
   TplBenchTraits<tU_INT2>();
   TplBenchTraits<tINT1>();
   TplBenchTraits<tINT2>();
   TplBenchTraits<tINT4>();
   TplBenchTraits<tREAL4>();
   for (int aK=0 ; aK<int(eTyNums::eNbVals) ; aK++)
   {
       const cVirtualTypeNum & aVTN =  cVirtualTypeNum::FromEnum(eTyNums(aK));
       MMVII_INTERNAL_ASSERT_bench (int(aVTN.V_TyNum())==aK,"Bench cVirtualTypeNum::FromEnum");
   }
   // getchar();
}





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
   BenchTraits(); 

   StdOut() << "Bench_NumsBench_NumsBench_NumsBench_Nums\n";
   MMVII_INTERNAL_ASSERT_bench (sizeof(tREAL4)==4,"Bench size tREAL4");
   MMVII_INTERNAL_ASSERT_bench (sizeof(tREAL8)==8,"Bench size tREAL8");

   MMVII_INTERNAL_ASSERT_bench (sizeof(tREAL16)==16,"Bench size tREAL16");

   MMVII_INTERNAL_ASSERT_bench (sizeof( tINT1)==1,"Bench size tINT1");
   MMVII_INTERNAL_ASSERT_bench (sizeof( tINT2)==2,"Bench size tINT2");
   MMVII_INTERNAL_ASSERT_bench (sizeof( tINT4)==4,"Bench size tINT4");
   // MMVII_INTERNAL_ASSERT_bench (sizeof( tINT8)==8,"Bench round_up");
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

            {
                double aFrac = FracPart(aRatio);
                MMVII_INTERNAL_ASSERT_bench ((aFrac>=0) &&( aFrac<1),"Bench Frac");
                double I  = aRatio - aFrac;
                MMVII_INTERNAL_ASSERT_bench(round_ni(I)==I,"Bench Frac");
            }
         }
      }
   }
}

template <class Type> Type  Mediane(std::vector<Type> & aV)
{
   std::sort(aV.begin(),aV.end());
   return aV.at(aV.size()/2);
}

template  double Mediane(std::vector<double> &);

};

