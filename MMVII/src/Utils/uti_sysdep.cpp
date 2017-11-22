#include "include/MMVII_all.h"

namespace MMVII
{

int SysCall(const std::string & aCom, bool SVP) 
{
   int aResult = system(aCom.c_str());
   if (aResult != EXIT_SUCCESS)
   {
      MMVII_INTERNAL_ASSERT_always(SVP,"Syscall for ["+aCom+"]");
   }
   return aResult;
}

#if   (THE_MACRO_MMVII_SYS==MMVII_SYS_L)
const std::string TheMMVII_SysName = "Gnu/Linux";
int mmvii_NbProcSys()
{
    return sysconf (_SC_NPROCESSORS_CONF);
}
int mmvii_GetPId()
{
    return getpid();
}


#elif (THE_MACRO_MMVII_SYS==MMVII_SYS_W)
const std::string TheMMVII_SysName = "Bill's shit";
int mmvii_NbProcSys()
{
    SYSTEM_INFO sysinfo;
    GetSystemInfo( &sysinfo );
    return sysinfo.dwNumberOfProcessors;
}
int mmvii_GetPId()
{
    return _getpid();
}
#else
const std::string TheMMVII_SysName = "Steve's shit";
int mmvii_GetPId()
{
    MMVII_INTERNAL_ASSERT_always(false,"mmvii_GetPId on "+TheSysName);
    return -1;
}
int mmvii_NbProcSys()
{
    MMVII_INTERNAL_ASSERT_always(false,"mmvii_NbProcSys on "+TheSysName);
    return -1;
}
#endif

};

