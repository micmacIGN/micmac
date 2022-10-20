#include "TreeDist.h"

#include "MMVII_Bench.h"


namespace MMVII
{
void BenchFastTreeDist(cParamExeBench & aParam)
{
   if (! aParam.NewBench("FastTreeDist")) return;

   {
      int aNb = 1<<std::min(4,aParam.Level()) ;
      if (aParam.Level()>4) aNb += aParam.Level()-4;
      aNb = std::min(20,aNb);
   
      NS_MMVII_FastTreeDist::AllBenchFastTreeDist(aParam.Show(),aNb);
   }
   aParam.EndBench();
}
};

