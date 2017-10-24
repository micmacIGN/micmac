#include "include/MMVII_all.h"

#define SYS_L 0  // Linux
#define SYS_W 1  // Window
#define SYS_A 2  // Apple

#ifdef __linux__
const std::string TheSysName = "Linux";
#include <unistd.h>
#define SYS SYS_L
#endif

#ifdef _WIN32
const std::string TheSysName = "Bill's shit";
#define SYS SYS_W
#endif

#ifdef __APPLE__
const std::string TheSysName = "Steve's shit";
#define SYS SYS_A
#endif



namespace MMVII
{

#if   (SYS==SYS_L)
int mmvii_NbProcSys()
{
    return sysconf (_SC_NPROCESSORS_CONF);
}
int mmvii_GetPId()
{
    return getpid();
}
#elif (SYS==SYS_W)
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

