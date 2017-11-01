#include "include/MMVII_all.h"

#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>

using namespace boost::filesystem;

namespace MMVII
{

// Prouve la pertinence du warning sur  mTable[*aPtr] = aC;

static_assert( int(char(255)) != 255,"Bad char assert");
static_assert( int(char(-1)) == -1,"Bad char assert");

// cGestObjetEmpruntable<cCarLookUpTable>   cCarLookUpTable::msGOE;

void  cCarLookUpTable::Init(const std::string& aStr,char aC)
{
    MMVII_INTERNAL_ASSERT_medium(!mInit,"Multiple init of  cCarLookUpTable");
    mInit= true;
    for (const char * aPtr = aStr.c_str() ; *aPtr ; aPtr++)
        mTable[*aPtr] = aC;  // Laisse le warning, il faudra le regler !!!
        // mTable[size_t(*aPtr)] = aC;
    mIns = aStr;
}

void  cCarLookUpTable::UnInit()
{
    MMVII_INTERNAL_ASSERT_medium(mInit,"Multiple Uninit of  cCarLookUpTable");
    mInit= false;
    for (const char * aPtr = mIns.c_str() ; *aPtr ; aPtr++)
        mTable[*aPtr] = 0;  // Laisse le warning, il faudra le regler !!!
    mIns = "";
}

cCarLookUpTable::cCarLookUpTable() :
     mInit(false)
{
    MEM_RAZ(&mTable,1);
    // MEM_RAZ(mTable,1); =>  sizeof(*mTable) == 1 !!!! 
    // std::cout << "DDddrrrr= " << &mTable << " ;; " << &(*mTable) << "\n";
}

std::vector<std::string>  SplitString(const std::string & aStr,const std::string & aSpace)
{
    std::vector<std::string> aRes;
    cMMVII_Appli::TheAppli().SplitString(aRes,aStr,aSpace);
    return  aRes;
}


void  SplitStringArround(std::string & aBefore,std::string & aAfter,const std::string & aStr,char aCharSep,bool SVP,bool PrivPref)
{
    std::string aStrSep(1,aCharSep);
    std::vector<std::string> aVStr;
    cMMVII_Appli::TheAppli().SplitString(aVStr,aStr,aStrSep);

    int aNbSplit = aVStr.size();

    if (aNbSplit==2)
    {
        aBefore = aVStr[0];
        aAfter = aVStr[1];
        return;
    }

    if (! SVP)
    {
       MMVII_INTERNAL_ASSERT_always
       (
            false,
              std::string("Cannot split string just in two arround [")+aCharSep
            + std::string("] nb got=") + ToS(int(aVStr.size()))
            + std::string(" ,input=" ) + aStr
       );
    }
/*
    if (aNbSplit==0)
    {
       aBefore="";
       aAfter="";
       return;
    }

    if (aNbSplit==1)
    {
        aBefore =    PrivPref ?  aVStr[0] : "";
        aAfter  = (!PrivPref) ?  aVStr[0] : "";
        return;
    }
*/
     
    aBefore = "";
    aAfter = "";
    int aKSplit = PrivPref ?  (aNbSplit-1) :  1;

    std::string * aCur = &aBefore;
    for (int aK=0 ; aK<aNbSplit ; aK++)
    {
       if (aK==aKSplit) aCur = &aAfter;

       if ((aK!=0) && (aK!= aKSplit)) 
          *aCur +=  aCharSep;
       *aCur +=  aVStr[aK];
    }
}

std::string Prefix(const std::string & aStr,char aSep,bool SVP,bool PrivPref)
{
    std::string aBefore,aAfter;
    SplitStringArround(aBefore,aAfter,aStr,aSep,SVP,PrivPref);
    return aBefore;
}

std::string Postfix(const std::string & aStr,char aSep,bool SVP,bool PrivPref)
{
    std::string aBefore,aAfter;
    SplitStringArround(aBefore,aAfter,aStr,aSep,SVP,PrivPref);
    return aAfter;
}


bool UCaseEqual(const std::string & aStr1 ,const std::string & aStr2)
{
   return boost::iequals(aStr1,aStr2);
}


    /***********************************************/
    /*                                             */
    /*        Dir/Files-names utils                */
    /*                                             */
    /***********************************************/

std::string DirCur()
{
// std::cout << "DDDDCCCC=[" <<  "." + path::preferred_separator << "]\n";
   return  std::string(".") + path::preferred_separator;
}

std::string DirOfPath(const std::string & aPath,bool ErrorNonExist)
{
   std::string aDir,aFile;
   SplitDirAndFile(aDir,aFile,aPath,ErrorNonExist);
   return aDir;
}
   
std::string FileOfPath(const std::string & aPath,bool ErrorNonExist)
{
   std::string aDir,aFile;
   SplitDirAndFile(aDir,aFile,aPath,ErrorNonExist);
   return aFile;
}

std::string UpDir(const std::string & aDir,int aNb)
{
   std::string aRes = aDir;
   for (int aK=0 ; aK<aNb ; aK++)
   {
      // aRes += std::string("..") +  path::preferred_separator;
      // aRes += ".." +  path::preferred_separator;
      aRes = aRes + std::string("..") +  path::preferred_separator;
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
   bool aResult = true;
   if (! exists(aPath))
   {
       MMVII_INTERNAL_ASSERT_always(!ErrorNonExist,"File non existing in SplitDirAndFile");
       //return false;
       aResult = false;
   }

   if (is_directory(aPath))
   {
       aDir = aDirAndFile;
       aFile = "";
   }
   // This case is not handled as I want , File=".", I want ""
   else if ( (! aDirAndFile.empty()) &&  (aDirAndFile.back()== path::preferred_separator))
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
   return aResult;
}


std::string  Quote(const std::string & aStr)
{
   if (aStr.empty() || aStr[0]!='"')
     return std::string("\"") + aStr + '"';

   return aStr;
}


};

