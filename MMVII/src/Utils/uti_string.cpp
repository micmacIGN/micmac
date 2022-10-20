

/** \file uti_string.cpp
    \brief Implementation of utilitary services


    Use std::filesystem and home made stuff for :

      - split names
      - separate directories from files
      - parse directories

*/


#include <filesystem>

#include "MMVII_util.h"
#include "cMMVII_Appli.h"

#include <boost/algorithm/string.hpp>


using namespace std::filesystem;


namespace MMVII
{


char ToHexacode(int aK)
{
    MMVII_INTERNAL_ASSERT_tiny((aK>=0)&&(aK<16),"ToHexacode");
    return (aK<10) ? ('0'+aK) : ('A'+(aK-10));
}

int  FromHexaCode(char aC)
{
   if ((aC>='0')&&(aC<='9')) return aC-'0';
   MMVII_INTERNAL_ASSERT_tiny((aC>='A')&&(aC<='F'),"FromHexacode");
   return 10 + (aC-'A');
}

// Prouve la pertinence du warning sur  mTable[*aPtr] = aC;

// static_assert( int(char(255)) != 255,"Bad char assert");
// static_assert( int(char(-1)) == -1,"Bad char assert");

// cGestObjetEmpruntable<cCarLookUpTable>   cCarLookUpTable::msGOE;

void  cCarLookUpTable::Init(const std::string& aStr,char aC)
{
    MMVII_INTERNAL_ASSERT_medium(!mInit,"Multiple init of  cCarLookUpTable");
    mInit= true;
    for (const char * aPtr = aStr.c_str() ; *aPtr ; aPtr++)
        mUTable[int(*aPtr)] = aC;  // Laisse le warning, il faudra le regler !!!
        // mTable[size_t(*aPtr)] = aC;
    mIns = aStr;
}

void  cCarLookUpTable::UnInit()
{
    MMVII_INTERNAL_ASSERT_medium(mInit,"Multiple Uninit of  cCarLookUpTable");
    mInit= false;
    for (const char * aPtr = mIns.c_str() ; *aPtr ; aPtr++)
        mUTable[int(*aPtr)] = 0;  // Laisse le warning, il faudra le regler !!!
    mIns = "";
}

cCarLookUpTable::cCarLookUpTable() :
     mUTable (mDTable-  std::numeric_limits<char>::min()),
     mInit(false)
{
    MEM_RAZ(&mDTable,1);
}

std::vector<std::string>  SplitString(const std::string & aStr,const std::string & aSpace)
{
    std::vector<std::string> aRes;
    cMMVII_Appli::CurrentAppli().SplitString(aRes,aStr,aSpace);
    return  aRes;
}


void  SplitStringArround(std::string & aBefore,std::string & aAfter,const std::string & aStr,char aCharSep,bool SVP,bool PrivPref)
{
    std::string aStrSep(1,aCharSep);
    std::vector<std::string> aVStr;
    cMMVII_Appli::CurrentAppli().SplitString(aVStr,aStr,aStrSep);

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

std::string LastPrefix(const std::string & aStr,char aSep)
{
    return Prefix(aStr,aSep,true,true);
}

std::string Postfix(const std::string & aStr,char aSep,bool SVP,bool PrivPref)
{
    std::string aBefore,aAfter;
    SplitStringArround(aBefore,aAfter,aStr,aSep,SVP,PrivPref);
    return aAfter;
}

std::string LastPostfix(const std::string & aStr,char aSep)
{
    return Postfix(aStr,aSep,true,true);
}

std::string ChgPostix(const std::string & aPath,const std::string & aPost)
{
    return LastPrefix(aPath,'.') + "." + aPost;
}

bool UCaseEqual(const std::string & aStr1 ,const std::string & aStr2)
{
   return boost::iequals(aStr1,aStr2);
}

bool UCaseBegin(const char * aBegin,const char * aStr)
{
   while (*aBegin)
   {
      if (tolower(*aBegin) != tolower(*aStr))
         return false;
      aBegin++;
      aStr++;
   }
   return true;
}

bool CaseSBegin(const char * aBegin,const char * aStr)
{
   while (*aBegin)
   {
      if (*aBegin != *aStr)
         return false;
      aBegin++;
      aStr++;
   }
   return true;
}




    /* =========================================== */
    /*                                             */
    /*        Dir/Files-names utils                */
    /*                                             */
    /* =========================================== */

char CharDirSeparator()
{
   return  path::preferred_separator;
}

const std::string & StringDirSeparator()
{
   static std::string aRes{path::preferred_separator};
   
   return  aRes;
}

std::string DirCur()
{
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

std::string AbsoluteName(const std::string & aName)
{
     return absolute(aName).c_str();
}

std::string AddBefore(const std::string & aPath,const std::string & ToAdd)
{
   return DirOfPath(aPath,false) + ToAdd + FileOfPath(aPath,false);
}




/*
  It was a test of using Boost for Up Dir,but untill now I am not 100% ok
  with the results:
        [.] => []
        [/a/b/c] => [/a/b]
        [a/b/c] => [a/b]
        [a] => []

std::string BUD(const std::string & aDir)
{
   path aPath(aDir);
   aPath = aPath.parent_path();
   std:: cout << "BUDDDDDD [" << aDir << "] => [" <<  aPath.c_str() << "]\n";
   return aPath.c_str();
}
*/
std::string OneUpStd(const std::string & aDir)
{
   const char * aC = aDir.c_str();
   int aL = strlen(aC);

   // Supress all the finishing  /
   while ((aL>0) && (aC[aL-1] == path::preferred_separator)) 
          aL--;

   // Supress all the not /
   while ((aL>0) && (aC[aL-1]!= path::preferred_separator))
       aL--;

   int aL0 = aL;
   // Supress all the  /
   while ((aL>0) && (aC[aL-1] == path::preferred_separator)) 
         aL--;

    // Add the last /
    if (aL0!=aL) 
        aL++;
    return  aDir.substr(0,aL);
}


std::string OneUpDir(const std::string & aDir)
{
   std::string aRes = OneUpStd(aDir);
   if (aRes!="") return aRes;
   return  aDir + std::string("..") +  path::preferred_separator;
}

/** Basic but seems to work untill now
*/
std::string UpDir(const std::string & aDir,int aNb)
{
   std::string aRes = aDir;
   for (int aK=0 ; aK<aNb ; aK++)
   {
      // aRes += std::string("..") +  path::preferred_separator;
      // aRes += ".." +  path::preferred_separator;
      // aRes = aRes + std::string("..") +  path::preferred_separator;
      aRes = OneUpDir(aRes);
   }
   return aRes;
}

bool ExistFile(const std::string & aName)
{
   path aPath(aName);
   return exists(aPath);
}

uintmax_t SizeFile(const std::string & aName)
{
    path aPath(aName);
    return file_size(aPath);
}

bool IsDirectory(const std::string & aName)
{
    path aPath(aName);
    return is_directory(aPath);
}


void MakeNameDir(std::string & aDir)
{
   if (aDir.back() != path::preferred_separator)
   {
      aDir += path::preferred_separator;
   }
}

bool SplitDirAndFile(std::string & aDir,std::string & aFile,const std::string & aDirAndFile,bool ErrorNonExist)
{
/*
if (0)
{
   static int aCpt=0; aCpt++;
   std:: cout << "SplitDirAndFile " << aCpt  << " " << __FILE__ << "\n";
   getchar();
}
*/

   path aPath(aDirAndFile);
   bool aResult = true;
   if (! exists(aPath))
   {
       if (ErrorNonExist)
       {
           MMVII_INTERNAL_ASSERT_always(false,"File non existing in SplitDirAndFile for ["+  aDirAndFile +"]");
       }
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
       else 
       {
          MakeNameDir(aDir);
       }

/*if (aDir.back() != path::preferred_separator)
       {
           aDir += path::preferred_separator;
       }     
*/
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

void SkeepWhite(const char * & aC)
{
    while (isspace(*aC)) 
         aC++;
}


bool CreateDirectories(const std::string & aDir,bool SVP)
{
    bool Ok = std::filesystem::create_directories(aDir);

    if ((! Ok) && (!SVP))
    {
        // There is something I dont understand with boost on error with create_directories,
        // for me it works but it return false, to solve later ....
	// Ch. M.: My understanrdfing is :
	//   - if directory is created, return true
	//   - if directory not created because already existing, return false
	//   - if directory not created because of error, exception is throwed
	//          (use create_directories(aDir, errorCode) to have the noexcept version)
        if (ExistFile(aDir))
        {
            MMVII_INTERNAL_ASSERT_Unresolved(false,"Cannot create directory for arg " + aDir);
        }
        else
        {
            MMVII_UsersErrror(eTyUEr::eCreateDir,"Cannot create directory for arg " + aDir);
        }
    }
    return Ok;
}

bool RemoveRecurs(const  std::string & aDir,bool ReMkDir,bool SVP)
{
    std::filesystem::remove_all(aDir);
    if (ReMkDir)
    {
        bool aRes = CreateDirectories(aDir,SVP);
        return aRes;
    }
    return true;
}

bool RemoveFile(const  std::string & aFile,bool SVP)
{
   bool Ok = std::filesystem::remove(aFile);
   MMVII_INTERNAL_ASSERT_User(  Ok||SVP  , eTyUEr::eRemoveFile,"Cannot remove file for arg " + aFile);
   return Ok;
}

/** remove a pattern of file */

bool  RemovePatternFile(const  std::string & aPat,bool SVP)
{
    tNameSet aSet = SetNameFromString(aPat,true);
    std::vector<const std::string *> aVS;
    aSet.PutInVect(aVS,false);
    std::string aDir = DirOfPath(aPat,false);

    for (const auto & aS : aVS)
    {
        if (!RemoveFile(aDir+*aS,SVP))
           return false;
    }
    return true;
}


void RenameFiles(const std::string & anOldName, const std::string & aNewName)
{
    std::filesystem::rename(anOldName,aNewName);
}


/** copy a file on another , use std::filesystem
*/

void CopyFile(const std::string & aName,const std::string & aDest)
{
   std::filesystem::copy_file(aName,aDest,std::filesystem::copy_options::overwrite_existing);
}

void ActionDir(const std::string & aName,eModeCreateDir aMode)
{
   switch(aMode)
   {
      case eModeCreateDir::DoNoting :
      break;

      case eModeCreateDir::CreateIfNew :
           CreateDirectories(aName,false);
      break;

      case eModeCreateDir::CreatePurge :
           RemoveRecurs(aName,true,false);
      break;

      case eModeCreateDir::ErrorIfExist :
           MMVII_INTERNAL_ASSERT_strong(!ExistFile(aName),"File was not expected to exist:" + aName);
           CreateDirectories(aName,false);
      break;

      case eModeCreateDir::eNbVals : break;  // Because warning
   }
}



    /* =========================================== */
    /*                                             */
    /*        Get Files from dir                   */
    /*                                             */
    /* =========================================== */


/**
   Implementation of GetFilesFromDir, by adresse use std::filesystem
*/


void GetFilesFromDir(std::vector<std::string> & aRes,const std::string & aDir,const tNameSelector &  aNS,bool OnlyRegular)
{
   for (directory_iterator itr(aDir); itr!=directory_iterator(); ++itr)
   {
      std::string aName ( itr->path().filename().c_str());
      if ( ( (!OnlyRegular) || is_regular_file(itr->status())) &&  aNS.Match(aName))
         aRes.push_back(aName);
   }
}

/**
   Implementation of GetFilesFromDir, by value , use GetFilesFromDir by adress
*/
std::vector<std::string> GetFilesFromDir(const std::string & aDir,const tNameSelector &  aNS,bool OnlyRegular)
{
    std::vector<std::string> aRes;
    GetFilesFromDir(aRes,aDir,aNS,OnlyRegular);
 
    return aRes;
}

std::vector<std::string> GetSubDirFromDir(const std::string & aDir,const tNameSelector &  aNS)
{
    // std::vector<std::string> aR0 = GetFilesFromDir(aDir,aNS,false);
    std::vector<std::string> aRes;
    for (const auto & aN : GetFilesFromDir(aDir,aNS,false))
       if (IsDirectory(aDir+aN))
          aRes.push_back(aN);
    return aRes;
}

/**
   Implementation of RecGetFilesFromDir, by adress, use std::filesystem
*/
void RecGetFilesFromDir( std::vector<std::string> & aRes, const std::string & aDir,tNameSelector  aNS,int aLevMin, int aLevMax)
{
    for (recursive_directory_iterator itr(aDir); itr!=        recursive_directory_iterator(); ++itr)
    {
        int aLev = itr.depth();
        if ((aLev>=aLevMin) && (aLev<aLevMax))
        {
           std::string aName(itr->path().c_str());
           if ( is_regular_file(itr->status()) &&  aNS.Match(aName))
              aRes.push_back(aName);
        }
    }
}

/**
   Implementation of RecGetFilesFromDir, by value , use RecGetFilesFromDir by adress
*/

std::vector<std::string> RecGetFilesFromDir(const std::string & aDir,tNameSelector aNS,int aLevMin, int aLevMax)
{
    std::vector<std::string> aRes;
    RecGetFilesFromDir(aRes,aDir,aNS,aLevMin,aLevMax);
 
    return aRes;
}


/*
std::vector<std::string>  GetFilesFromDirAndER(const std::string & aDir,const std::string & aRegEx)
{
}
*/
    /* =========================================== */
    /*                                             */
    /*        Dir/Files-names utils                */
    /*                                             */
    /* =========================================== */


/*
void TestBooostIter()
{

   std:: cout <<  std::filesystem::absolute("./MMVII") << '\n';
   std:: cout <<  std::filesystem::absolute("MMVII") << '\n';
   std:: cout <<  std::filesystem::absolute("./") << '\n';
getchar();

for (        recursive_directory_iterator itr("./"); itr!=        recursive_directory_iterator(); ++itr)
{
    std:: cout  <<  itr->path().c_str() << " " << itr->path().filename() << ' '; // display filename only
    std:: cout << itr.level()  << " ";
    if (is_regular_file(itr->status())) std:: cout << " [" << file_size(itr->path()) << ']';
    std:: cout << '\n';
}
}
*/
std::string replaceFirstOccurrence
            (
                const std::string& s,
                const std::string& toReplace,
                const std::string& replaceWith,
                bool  SVP
            )
{
    std::size_t pos = s.find(toReplace);
    if (pos == std::string::npos)
    {
        if (!SVP)
        {
           StdOut() << "REPLACE ["<< toReplace << "] by : [" << replaceWith << "\n";
           StdOut() << "in [" << s << "]\n";
           MMVII_INTERNAL_ASSERT_always(false,"Cannot make subs");
        }
        return "";
    }
    std::string aDup = s;
    return aDup.replace(pos, toReplace.length(), replaceWith);
}

};

