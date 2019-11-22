#include "include/MMVII_all.h"

namespace MMVII
{

tREAL8 CubAppGaussVal(const tREAL8& aV)
{
   tREAL8 aAbsV = std::abs(aV);
   if (aAbsV>1.0) return 0.0;

   tREAL8 aAV2 = Square(aAbsV);

   return 1.0 + 2.0*aAbsV*aAV2 - 3.0*aAV2;
}



};

