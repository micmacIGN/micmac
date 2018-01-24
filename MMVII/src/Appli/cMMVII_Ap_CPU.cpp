#include "include/MMVII_all.h"

namespace MMVII
{

cMMVII_Ap_CPU::cMMVII_Ap_CPU() :
   mT0          (std::chrono::system_clock::now())     ,
   mPid         (mmvii_GetPId())   ,
   mNbProcSys   (mmvii_NbProcSys())
{
}



};

