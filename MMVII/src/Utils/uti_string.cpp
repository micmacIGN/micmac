#include "include/MMVII_all.h"

#include <boost/filesystem.hpp>
using namespace boost::filesystem;

namespace MMVII
{

// cGestObjetEmpruntable<cCarLookUpTable>   cCarLookUpTable::msGOE;

void  cCarLookUpTable::Init(const std::string& aStr,char aC)
{
    MMVII_INTERNAL_ASSERT_medium(!mInit,"Multiple init of  cCarLookUpTable");
    mInit= true;
    for (const char * aPtr = aStr.c_str() ; *aPtr ; aPtr++)
        mTable[*aPtr] = aC;
    mIns = aStr;
}

void  cCarLookUpTable::UnInit()
{
    MMVII_INTERNAL_ASSERT_medium(mInit,"Multiple Uninit of  cCarLookUpTable");
    mInit= false;
    for (const char * aPtr = mIns.c_str() ; *aPtr ; aPtr++)
        mTable[*aPtr] = 0;
    mIns = "";
}

cCarLookUpTable::cCarLookUpTable() :
     mInit(false)
{
    MEM_RAZ(&mTable,1);
    // MEM_RAZ(mTable,1); =>  sizeof(*mTable) == 1 !!!! 
    // std::cout << "DDddrrrr= " << &mTable << " ;; " << &(*mTable) << "\n";
}

std::string DirCur()
{
   return  "." + path::preferred_separator;
}

std::string DirOfPath(const std::string & aPath)
{
   std::string aDir,aFile;
   SplitDirAndFile(aDir,aFile,aPath);
   return aDir;
}
   
std::string FileOfPath(const std::string & aPath)
{
   std::string aDir,aFile;
   SplitDirAndFile(aDir,aFile,aPath);
   return aFile;
}

std::string UpDir(const std::string & aDir,int aNb)
{
   std::string aRes = aDir;
   for (int aK=0 ; aK<aNb ; aK++)
   {
      aRes += std::string("..") +  path::preferred_separator;
   }
   return aRes;
}

bool ExistFile(const std::string & aName)
{
   path aPath(aName);
   return exists(aPath);
}

bool SplitDirAndFile(std::string & aDir,std::string & aFile,const std::string & aDirAndFile,bool ErrorNonExist)
{
   path aPath(aDirAndFile);
   if (! exists(aPath))
   {
       MMVII_INTERNAL_ASSERT_always(!ErrorNonExist,"File non existing in SplitDirAndFile");
       return false;
   }

   if (is_directory(aPath))
   {
       aDir = aDirAndFile;
       aFile = "";
   }
   else
   {
       aFile = aPath.filename().c_str();
       aPath.remove_filename();
       aDir = aPath.c_str();
       if (aDir.empty())
       {
          aDir = DirCur();
       }
       else if (aDir.back() != '/')
       {
           aDir += path::preferred_separator;
       }
       // path& remove_filename();
   }  
   return true;
}



};

