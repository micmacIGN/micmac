#ifndef  _MMVII_Sys_H_
#define  _MMVII_Sys_H_

#ifdef __linux__
#include <unistd.h>
#endif

#include "MMVII_AllClassDeclare.h"

namespace MMVII
{


/** \file MMVII_Sys.h
    \brief Contains system & harwdare specificities
*/



// Use enum were it works are  they are "better" C++
// Use macro when required (TheSYS==eSYS::Linux) do not work in #if

#define MMVII_SYS_L 6  // Gnu/Linux, first perfect number ;-)
#define MMVII_SYS_A 666  // Apple  , Evil's number ;-(
#define MMVII_SYS_W 2610  // Window , Evil's number in non standard hexadecimal , some system like to do it hard  ...

enum class eSYS
{
    GnuLinux=MMVII_SYS_L,
    MacOs=MMVII_SYS_A,
    Windows=MMVII_SYS_W
};

#ifdef __linux__
const eSYS TheSYS = eSYS::GnuLinux;
#define THE_MACRO_MMVII_SYS MMVII_SYS_L
#endif

#ifdef _WIN32
const eSYS TheSYS = eSYS::Windows;
#define THE_MACRO_MMVII_SYS MMVII_SYS_W
#endif

#ifdef __APPLE__
const eSYS TheSYS = eSYS::MacOs;
#define THE_MACRO_MMVII_SYS MMVII_SYS_A
#endif

int mmvii_NbProcSys();
int mmvii_GetPId();

int GlobSysCall(const std::string &, bool SVP=false); ///< call system, if SVP=false error not EXIT_SUCCESS
///  Execucte the command in parallel by generating a makefile
int GlobParalSysCallByMkF(const std::string & aNameMkF,const std::list<std::string> & aListCom,int aNbProcess,bool SVP=false);



};

#endif  //  _MMVII_Sys_H_
