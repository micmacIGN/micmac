#include "V1VII.h"
#if (MMVII_KEEP_LIBRARY_MMV1)
#include "ext_stl/numeric.h"
#endif

#include "MMVII_nums.h"
#include "MMVII_Bench.h"


namespace MMVII
{

/**  A set of method for extracting the Kth value (for ex median) of a set of value w/o sorting them.
  Statistically the time is in O(N)  instead of O(N log N). Principle :

      - find a pivot value "Piv"  (here the average)
      - permutate the tab with first values bellow that Piv followed by values over Piv
      - while doing that count how many values are bellow/over Piv
      - the we know if the kth must be found in below/lower part
      - recursively search it in the according part

    If we are very lucky, each time we split in exactly two, and the cost is N + N/2 + .. 
   More realistically, supose each time we split in a=3/4  then the cost is N +aN + a^2N = N/(1-a), still
   linear, so far so good.

  By the way in worst degenerate case it can turn to be in O(N^2) . This is the case for example if
  the set of values follow an exponential law  "V V/2  V/4 V/8 ...." then each time the split
  is done with subset of size "1/N-1" ... 
   
       #  TODO add a modfication so that if the cost appears to be too high (for example over "5+3N"),
          then we switch to method base on sort

*/

/*
   Central method :  permutate the values so that [V(0),V(Kth-1)[ are < to "[V(Kth),V(Sz)[
   Use the recursive approach described above
*/

template <class TVal> void SplitArrounKthValue(TVal * Data,int aNb,int aKth)
{
   // Three case were there is nothing to do , the split is done
   if (aNb==0) return;
   if (aKth<=0) return;
   if (aKth>=aNb) return;

   // Compute the pivot value
   TVal aPivot = Average(Data,aNb);

   // Now split arround the pivot
   int aK0 =0;  //  index a value < Pivot
   int aK1 = aNb-1; // index of value >=Pivot
   while (aK0 < aK1)
   {
        while ((aK0<aK1) && (Data[aK0] <  aPivot)) aK0++;  // while value < Pivot , good  : increase
        while ((aK0<aK1) && (Data[aK1] >= aPivot)) aK1--;  // while value >= Pivot, good : decrease

        // we have met the point where potentially have two values 
        if (aK0 < aK1) 
        {
           std::swap(Data[aK0],Data[aK1]);
        }
   }
   // A little coherence checki,g
   MMVII_INTERNAL_ASSERT_tiny(aK0==aK1,"Verif in SplitArrounKthValue");

   // if we are here, all the value are equals
   if  (aK0==0)
   {
       return;
   }

   // some degenerate case, all the value are equals , but (for numerical uncertainty) they are not
   // exactly equal to the pivot (average), this may create an infinite loop: bad ...
   {
      int aNbV0 = 0;
      TVal aV0 = Data[0];
      for (int aKv=0 ; aKv<aNb ; aKv++)
      {
          aNbV0 += (Data[aKv] == aV0);
      }
      if (aNbV0==aNb)
      {
          return;
      }
   }

   // if  K0=K1= Kth, we are done value are splited arroun Kth vals
   if (aK0 == aKth)  return;


   if (aK0 < aKth) 
      // case the Kth value is above the pivot so search it in this top-interval
      SplitArrounKthValue(Data+aK0,aNb-aK0,aKth-aK0);
   else   
      // case the Kth value is bellow the pivot so search it in this bottom-interval
      SplitArrounKthValue(Data,aK0,aKth);
}


template <class TVal> TVal KthVal(TVal * Data,int aNb,int aKth)
{
   MMVII_INTERNAL_ASSERT_tiny(aKth>=0 && (aKth<=aNb-1),"KthVal");

   AssertTabValueOk(Data,aNb);
   SplitArrounKthValue(Data,aNb,aKth);
   // After SplitArrounKthValue, we know that value in [0,Kth[ are bellow Kth value
   // KTh value is in [Kth,Nb[, and that value in [0,Kth[ are bello but dont know where
   return MinTab(Data+aKth,aNb-aKth);
}
template <class TVal> TVal KthVal(std::vector<TVal> & aV,int aKth)
{
    return KthVal(aV.data(),(int)aV.size(),aKth);
}

template <class TVal> TVal KthValProp(std::vector<TVal> & aV,double aProp)
{
    int aKTh = round_down(aV.size()*aProp);
    aKTh = std::max(0,std::min((int(aV.size())-1),aKTh));

    return KthVal(aV.data(),(int)aV.size(),aKTh);
}

/*
template void SplitArrounKthValue(double * Data,int aNb,int aKth);
template  double KthVal(double * Data,int aNb,int aKth);
template double KthVal(std::vector<double> & aV,int aKth);
template double KthValProp(std::vector<double> & aV,double aProp);
*/


double NC_KthVal(std::vector<double> & aV, double aProportion)
{
   return KthValProp(aV,aProportion);
}

// template <class TVal> TVal KthVal(std::vector<TVal> & aV,int aKth)
double  IKthVal(std::vector<double> & aV, int aK)
{
     return KthVal(aV,aK);
}



double Cst_KthVal(const std::vector<double> & aV, double aProportion)
{
     std::vector<double> aDup= aV;
     return NC_KthVal(aDup,aProportion);
}


void BenchKTHVal(cParamExeBench & aParam)
{
   if (! aParam.NewBench("KthValue")) return;

    for (int aKTest=0 ; aKTest<500 ; aKTest++)
    {
        int aSz =  1+RandUnif_N(10);
        std::vector<double> aVV = VRandUnif_0_1(aSz);
        int aNbDup =  RandUnif_N(3);
        for (int aKD=0 ; aKD< aNbDup ; aKD++)
            aVV.at(RandUnif_N(aSz)) = aVV.at(RandUnif_N(aSz));

        int aKTh = RandUnif_N(aSz);
        double  aVK = IKthVal(aVV,aKTh);
        int aNbInfStr = 0;
        int aNbInfEq = 0;
        for  (const auto & aV : aVV)
        {
             aNbInfStr += (aV < aVK);
             aNbInfEq +=  (aV <= aVK);
        }
        // StdOut() <<  aNbInfStr  << " " << aKTh << " " <<   aNbInfEq  <<  " Of " << aSz << "\n";
        MMVII_INTERNAL_ASSERT_bench(aNbInfStr<=aKTh,"BenchKTHVal Inf Str");
        MMVII_INTERNAL_ASSERT_bench(aKTh<aNbInfEq,"BenchKTHVal Inf Eq");
/*
        StdOut() << "VVVV " 
                 << " P0=" << Cst_KthVal(aVV,0.0) 
                 << " PE=" << Cst_KthVal(aVV,0.05) 
                 << " I0=" << IKthVal(aVV,0) 
                 << "\n";
*/
    }
    aParam.EndBench();
}

/* ***************************************** */

#if (MMVII_KEEP_LIBRARY_MMV1)
template <class Type>  
    std::vector<Type>  V1RealRoots(const std::vector<Type> &  aVCoef, Type aTol,int aNbMaxIter)
{
    ElPolynome<Type>  aV1Pol((char*)0,aVCoef.size()-1);
    for (size_t aK=0 ; aK<aVCoef.size() ; aK++)
	    aV1Pol[aK] = aVCoef[aK];

    std::vector<Type>  aVRoots;
    RealRootsOfRealPolynome(aVRoots,aV1Pol,aTol,aNbMaxIter);

    return aVRoots;
}
#else // MMVII_KEEP_LIBRARY_MMV1
template <class Type>  
    std::vector<Type>  V1RealRoots(const std::vector<Type> &  aVCoef, Type aTol,int aNbMaxIter)
{
    MMVII_INTERNAL_ERROR("No V1RealRoots");
    return std::vector<Type>();
}
#endif // MMVII_KEEP_LIBRARY_MMV1

#define INST_V1ROOTS(TYPE)\
template std::vector<TYPE>  V1RealRoots(const std::vector<TYPE> &  aVCoef, TYPE aTol,int aNbMaxIter);

INST_V1ROOTS(tREAL4)
INST_V1ROOTS(tREAL8)
INST_V1ROOTS(tREAL16)




};
