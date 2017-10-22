#include "include/MMVII_all.h"

namespace MMVII
{

extern int mmvii_GetPId();
extern int mmvii_NbProcSys();


cMMVII_Ap_CPU::cMMVII_Ap_CPU() :
   mPid         (mmvii_GetPId()),
   mNbProcSys   (mmvii_NbProcSys())
{
}



};

