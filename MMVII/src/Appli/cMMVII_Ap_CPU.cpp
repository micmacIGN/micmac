#include "include/MMVII_all.h"

namespace MMVII
{

cMMVII_Ap_CPU::cMMVII_Ap_CPU() :
   mT0          (std::chrono::system_clock::now())     ,
   mPid         (mmvii_GetPId())   ,
   mNbProcSys   (mmvii_NbProcSys())
{
}

double cMMVII_Ap_CPU::SecFromT0() const
{
    tTime aT1 = std::chrono::system_clock::now();
    std::chrono::duration<double, std::milli> fp_ms = aT1-mT0;
    return fp_ms.count()/1000.0;

}


};

