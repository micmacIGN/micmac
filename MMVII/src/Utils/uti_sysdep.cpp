#include "cMMVII_Appli.h"
#include "MMVII_Sys.h"
#include "MMVII_DeclareCste.h"
#include "MMVII_MMV1Compat.h"
#include "MMVII_2Include_Serial_Tpl.h"

#include <filesystem>
#include <string>
#include <mutex>
#include <thread>
#include <atomic>
#include <deque>


#if   (THE_MACRO_MMVII_SYS==MMVII_SYS_L)  // Linux
#  include <unistd.h>
#elif (THE_MACRO_MMVII_SYS==MMVII_SYS_W)  // Windows
#  include <windows.h>
#  include <process.h>
#else  // Max OS X
#  include <unistd.h>
#  include <mach-o/dyld.h>
#endif

namespace fs = std::filesystem;

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
      StdOut() << "   - MVII Warning at line " <<  mLine << " of " << mFile << std::endl;
      StdOut() << "   - " << mMes << std::endl;
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


int GlobSysCall(const cParamCallSys & aCom, bool SVP)
{
   int aResult = system(aCom.Com().c_str());
   if (aResult != EXIT_SUCCESS)
   {
      MMVII_INTERNAL_ASSERT_always(SVP,"Syscall for ["+aCom.Com()+"]");
   }
   return aResult;
}

class cMMVIIMultiProcess
{
public:
    cMMVIIMultiProcess() {}
    int Exec(const std::list<cParamCallSys> & aListCom,int aNbProcess,bool SVP,bool Silence)
    {
      for (const auto& aCom : aListCom)
	     mQueueCom.push_back(aCom.Com());
      StdOut().flush();
      std::cerr.flush();
      mError = 0;
      std::vector<std::thread> aThreadList;
      for (int i = 0; i < aNbProcess; ++i)
          aThreadList.emplace_back(std::thread(&cMMVIIMultiProcess::ExecLoop, this, i, SVP, Silence));
      for (auto& t : aThreadList)
          t.join();
      return mError;
    }

private:
    void ExecLoop(int nThread, bool SVP, bool Silence)
    {
      while (true) {
          std::string aCom;
          {
              std::lock_guard<std::mutex> lock(mMutex_QueueCom);
              if (mQueueCom.empty() || mError)
                  return;
              aCom = mQueueCom.front();
              mQueueCom.pop_front();
          }
          if (! Silence)
          {
              std::lock_guard<std::mutex> lock(mMutex_cout);
              std::cout << aCom << "\n";
              std::cout.flush();
          }
          int aResult = system(aCom.c_str());
          if (aResult!= EXIT_SUCCESS)
          {
              mError = aResult;              // mError is shared std::atomic, will be /=0 if any thread doesn't return EXIT_SUCCESS
              MMVII_INTERNAL_ASSERT_always(SVP,"Syscall for ["+aCom+"]");  // May be not thread safe ...
          }
      }
    }

    std::deque<std::string> mQueueCom;
    std::atomic<int> mError;
    std::mutex mMutex_QueueCom;
    std::mutex mMutex_cout;
};


int GlobParalSysCallByMkF(const std::string & aNameMkF,const std::list<cParamCallSys> & aListCom,int aNbProcess,bool SVP,bool Silence)
{
   //CM: Keep creating the Makefile, maybe useful for debugging ?
   cMMVII_Ofs  aOfs(aNameMkF, eFileModeOut::CreateText);
   int aNumTask=0;
   std::string aStrAllTask = "all : ";
   for (const auto & aNameCom : aListCom)
   {
       std::string aNameTask = "Task_" + ToStr(aNumTask);
       aStrAllTask += BLANK  + aNameTask;
       aOfs.Ofs() << aNameTask << " :\n";
       aOfs.Ofs() << (Silence ? "\t@" : "\t") << aNameCom.Com() << "\n";
       aNumTask++;
   }
   aOfs.Ofs() << aStrAllTask << "\n";
   aOfs.Ofs() << "\t\n";
   aOfs.Ofs().close();

   cMMVIIMultiProcess e;
   return e.Exec(aListCom, aNbProcess, SVP, Silence);
}

static fs::path MMVII_RawSelfExecName();


std::string MMVII_CanonicalRootDirFromExec()
{
   auto selfExec = MMVII_RawSelfExecName();
   if (selfExec.empty())
        MMVII_INTERNAL_ERROR("Can't find file name of this process !");
   return fs::canonical(selfExec).parent_path().parent_path().generic_string();
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

static fs::path MMVII_RawSelfExecName()
{
    char buf[4096];

    *buf = 0;
    ssize_t result = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (result >= 0 && (size_t)result < sizeof(buf) - 1)
        buf[result] = 0;
    else
        buf[0] = 0;
    return fs::path(buf);
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


fs::path MMVII_RawSelfExecName()
{
    wchar_t buffer[MAX_PATH];
    *buffer = L'0';
    DWORD size = GetModuleFileNameW(nullptr, buffer,(DWORD)sizeof(buffer));
    if (size <0 || size == (DWORD)sizeof(buffer))
        *buffer = L'0';
    return fs::path(buffer);
}

#else

const std::string TheMMVII_SysName = "Steve's shit";
int mmvii_GetPId()
{
    return getpid();
}
int mmvii_NbProcSys()
{
    return sysconf(_SC_NPROCESSORS_ONLN);
}

fs::path MMVII_RawSelfExecName()
{
// Ch.M: Not tested
    fs::path path;

    uint32_t size = 0;
    _NSGetExecutablePath(nullptr, &size);
    char *buffer = new char[size + 1];
    if (_NSGetExecutablePath(buffer, &size) >= 0) {
        buffer[size] = '\0';
        path = buffer;
    }
    delete[] buffer;
    return path;
}

#endif
} // Namespace MMVII

