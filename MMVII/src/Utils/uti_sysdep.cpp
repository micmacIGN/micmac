#include "cMMVII_Appli.h"
#include "MMVII_DeclareCste.h"
#include "MMVII_MMV1Compat.h"
#include "MMVII_2Include_Serial_Tpl.h"
#include <unistd.h>

namespace MMVII
{
bool NeverHappens() {return false;}


cMMVII_Warning::cMMVII_Warning(const std::string & aMes,int aLine,const std::string &  aFile) :
   mCpt     (0),
   mMes     (aMes),
   mLine    (aLine),
   mFile    (aFile)
{
}

cMMVII_Warning::~cMMVII_Warning()
{
    if (mCpt==0) return;

    // At this step StdOut() may have be destroyed
    if (cMMVII_Appli::WithWarnings())
       std::cout << "##   - Nb Warning "<< mCpt << ", for :[" << mMes<<"]\n";
}

void cMMVII_Warning::Activate()
{
   mCpt++;

   if (mCpt!=1) 
      return;
   if (cMMVII_Appli::WithWarnings())
   {
      StdOut() << "   - MVII Warning at line " <<  mLine << " of " << mFile << "\n";
      StdOut() << "   - " << mMes << "\n";
   }
}



/*
typedef std::pair<int,std::string> tLFile;
void MMVII_Warning(const std::string & aMes,int aLine,const std::string &  aFile)
{
    tLFile aLF(aLine,aFile);
    static std::set<
}
*/


int GlobSysCall(const std::string & aCom, bool SVP) 
{
   int aResult = system(aCom.c_str());
   if (aResult != EXIT_SUCCESS)
   {
      MMVII_INTERNAL_ASSERT_always(SVP,"Syscall for ["+aCom+"]");
   }
   return aResult;
}


int GlobParalSysCallByMkF(const std::string & aNameMkF,const std::list<std::string> & aListCom,int aNbProcess,bool SVP)
{
   //RemoveFile(const  std::string & aFile,bool SVP)

   cMMVII_Ofs  aOfs(aNameMkF,false);
   int aNumTask=0;
   std::string aStrAllTask = "all : ";
   for (const auto & aNameCom : aListCom)
   {
       std::string aNameTask = "Task_" + ToStr(aNumTask);
       aStrAllTask += BLANK  + aNameTask;
       aOfs.Ofs() << aNameTask << " :\n";
       aOfs.Ofs() << "\t" << aNameCom << "\n";
       aNumTask++;
   }
   aOfs.Ofs() << aStrAllTask << "\n";
   aOfs.Ofs() << "\t\n";
   aOfs.Ofs().close();

   std::string aComMake = "make all -f " +  aNameMkF + " -j" + ToStr(aNbProcess);

   return GlobSysCall(aComMake,false);
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
    MMVII_INTERNAL_ASSERT_always(false,"mmvii_GetPId on "+TheMMVII_SysName);
    return -1;
}
int mmvii_NbProcSys()
{
    MMVII_INTERNAL_ASSERT_always(false,"mmvii_NbProcSys on "+TheMMVII_SysName);
    return -1;
}
#endif

};

