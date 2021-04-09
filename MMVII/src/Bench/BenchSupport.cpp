#include "include/MMVII_all.h"
#include <cmath>

/** \file BenchSupport.cpp
    \brief Support librairy bench

    Bench for stuf that are supposed to be general but that I cant find to my
    taste in stl . 
*/


namespace MMVII
{

void BenchSTL_Support(cParamExeBench & aParam)
{
   if (! aParam.NewBench("STL_Support")) return;

   // Test eraseif
   {
        int aNb=20;
        std::vector<int>  aVI;
        for (int aK=0 ; aK<aNb ; aK++) // contains all int
           aVI.push_back(aK);
        erase_if(aVI,[](const int &i) {return (i%2)!=0;});  // suppress odd

        MMVII_INTERNAL_ASSERT_bench((int)aVI.size()==(aNb/2),"erase_if");

        for (int aK=0 ; aK<int(aVI.size()) ; aK++) // contains all int
            MMVII_INTERNAL_ASSERT_bench(aVI.at(aK)==(2*aK),"erase_if");
   }
   aParam.EndBench();
}

};

