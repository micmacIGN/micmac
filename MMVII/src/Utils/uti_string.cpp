

/** \file uti_string.cpp
    \brief Implementation of utilitary services


    Use std::filesystem and home made stuff for :

      - split names
      - separate directories from files
      - parse directories

*/

#include <filesystem>

#include "MMVII_Sys.h"
#include "MMVII_util.h"
#include "cMMVII_Appli.h"


namespace fs=std::filesystem;

namespace MMVII
{

std::vector<std::string >  AddPostFix(const std::vector<std::string>  & aV,const std::string  & aPost)
{
    std::vector<std::string > aRes;
    for (const auto & aStr :aV)
        aRes.push_back(aStr+aPost);
    return aRes;
}

/* ************************************************* */
/*                                                   */
/*                cCarLookUpTable                    */
/*                                                   */
/* ************************************************* */


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

std::string  cCarLookUpTable::Translate(const std::string & aStr) const
{
     std::string aRes;

     for (char aC : aStr)
     {	
         char aTr = mUTable[int(aC)];
	 if (aTr!=0)
            aRes.push_back(aTr);
     }

     return aRes;
}


void cCarLookUpTable::InitId(char aC1,char aC2)
{
   mReUsable = false;
   for (char aC = aC1 ;aC<=aC2 ; aC++)
       mUTable[int(aC)] =aC;
}

void cCarLookUpTable::Chg1C(char aC1,char aC2)
{
   mReUsable = false;
   mUTable[int(aC1)] =aC2;
}

void  cCarLookUpTable::UnInit()
{
    MMVII_INTERNAL_ASSERT_medium(mInit,"Not init of  cCarLookUpTable");
    MMVII_INTERNAL_ASSERT_medium(mReUsable,"Not ReUsable");
    mInit= false;
    for (const char * aPtr = mIns.c_str() ; *aPtr ; aPtr++)
        mUTable[int(*aPtr)] = 0;  // Laisse le warning, il faudra le regler !!!
    mIns = "";
}

cCarLookUpTable::cCarLookUpTable() :
     mUTable (mDTable-  std::numeric_limits<char>::min()),
     mInit(false),
     mReUsable  (true)
{
    MEM_RAZ(&mDTable,1);
}

void cCarLookUpTable::InitIdGlob()
{
    InitId(std::numeric_limits<char>::min(),std::numeric_limits<char>::max()-1);
}


/* ************************************************* */
/*                                                   */
/*                     MMVII                         */
/*                                                   */
/* ************************************************* */

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
    return std::equal(
        aStr1.begin(), aStr1.end(), aStr2.begin(), aStr2.end(),
        [](unsigned char a, unsigned char b) { return std::tolower(a) == std::tolower(b); }
    );
}

std::string ToLower(const std::string &  aStr)
{
    std::string s;
    std::transform(
        aStr.cbegin(),aStr.cend(),std::back_inserter(s),
        [](unsigned char c) { return std::tolower(c); }
        );
    return s;
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

bool UCaseMember(const std::vector<std::string> & aVec,const std::string & aName)
{
    for (const auto &  aTest : aVec)
        if (UCaseEqual(aTest,aName))
            return true;
    return false;
}

const std::string & StrWDef(const std::string & aValue,const std::string & aDef)
{
        return  (aValue!="") ? aValue : aDef;
}

std::string  ToStandardStringIdent(const std::string & aStr)
{
    static cCarLookUpTable  aLUT;
    bool isFirst= true;
    if (isFirst)
    {
       isFirst = false;
       aLUT.InitId('0','9');
       aLUT.InitId('a','z');
       aLUT.InitId('A','Z');
       aLUT.Chg1C(' ','_');
       aLUT.Chg1C('-','-');
    }

    return aLUT.Translate(aStr);
}



    /* =========================================== */
    /*                                             */
    /*        Dir/Files-names utils                */
    /*                                             */
    /* =========================================== */

const std::string & StringDirSeparator()
{
//    static std::string aRes{fs::path::preferred_separator};
// CM: We'll always use '/' as directory separator, even on Windows.
// It's supported and MicMac pattern matching will work better and identically on all systems
    static std::string aRes{"/"};

    return  aRes;
}

std::string DirCur()
{
    return  "." + StringDirSeparator();
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
   return fs::absolute(aName).generic_string();
}

std::string AddBefore(const std::string & aPath,const std::string & ToAdd)
{
   return DirOfPath(aPath,false) + ToAdd + FileOfPath(aPath,false);
}

static bool EndWithDirectorySeparator(const std::string& aName)
{
#if (THE_MACRO_MMVII_SYS==MMVII_SYS_W)
   return aName.back() == '/' || aName.back() == '\\';
#else
   return aName.back() == '/';
#endif
}


std::string UpDir(const std::string & aDir)
{
   fs::path parent_path(aDir);
   if (EndWithDirectorySeparator(aDir))
       parent_path = parent_path.parent_path(); // remove trailing '/' , considered as a directory path
   parent_path = parent_path.parent_path();
   std::string updir = parent_path.generic_string();
   MakeNameDir(updir);
   return updir;
}


bool ExistFile(const std::string & aName)
{
   return fs::exists(aName);
}

uintmax_t SizeFile(const std::string & aName)
{
   return fs::file_size(aName);
}

bool IsDirectory(const std::string & aName)
{
   return fs::is_directory(aName);
}

void MakeNameDir(std::string & aDir)
{
   if (! EndWithDirectorySeparator(aDir))
   {
      aDir += StringDirSeparator();
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

   fs::path aPath(aDirAndFile);
   bool aResult = true;
   if (! fs::exists(aPath))
   {
       if (ErrorNonExist)
       {
           MMVII_INTERNAL_ASSERT_always(false,"File non existing in SplitDirAndFile for ["+  aDirAndFile +"]");
       }
       //return false;
       aResult = false;
   }

   if (fs::is_directory(aPath))
   {
       aDir = aDirAndFile;
       aFile = "";
   }
   // This case is not handled as I want , File=".", I want ""
   else if ( (! aDirAndFile.empty()) && EndWithDirectorySeparator(aDirAndFile))
   {
       aDir = aDirAndFile;
       aFile = "";
   }
   else
   {
       aFile = aPath.filename().generic_string().c_str();
       aPath.remove_filename();
       aDir = aPath.generic_string().c_str();
       if (aDir.empty())
       {
          aDir = DirCur();
       }
       else 
       {
          MakeNameDir(aDir);
       }

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
    bool Ok = fs::create_directories(aDir);

    if ((! Ok) && (!SVP))
    {
        // There is something I dont understand with std::filesystem on error with create_directories,
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
            MMVII_UserError(eTyUEr::eCreateDir,"Cannot create directory for arg " + aDir);
        }
    }
    return Ok;
}

bool RemoveRecurs(const  std::string & aDir,bool ReMkDir,bool SVP)
{
    fs::remove_all(aDir);
    if (ReMkDir)
    {
        bool aRes = CreateDirectories(aDir,SVP);
        return aRes;
    }
    return true;
}

bool RemoveFile(const  std::string & aFile,bool SVP)
{
   bool Ok = fs::remove(aFile);
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
    fs::rename(anOldName,aNewName);
}


/** copy a file on another , use std::filesystem
*/

void CopyFile(const std::string & aName,const std::string & aDest)
{
   fs::copy_file(aName,aDest,fs::copy_options::overwrite_existing);
}

void CopyPatternFile(const std::string & aDirIn,const std::string & aPattern,const std::string & aDirOut)
{
   std::vector<std::string> aListFile =  GetFilesFromDir(aDirIn,AllocRegex(aPattern));
   for (const auto  & aNameFile : aListFile)
   {
       CopyFile(aDirIn+aNameFile,aDirOut+aNameFile);
   }
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


void  MakeBckUp(const std::string & aDir,const std::string & aNameFile,int aNbDig)
{
    std::string aPattern = "BckUp_([0-9]*)_" + aNameFile;
    tNameSelector aSel =  AllocRegex(aPattern);

    std::vector<std::string> aVS = GetFilesFromDir(aDir,aSel);

    int aIMax = -1;
    for (const auto & aNameFile : aVS)
    {
        std::string  aStrNum = ReplacePattern(aPattern,"$1",aNameFile);
        UpdateMax(aIMax,cStrIO<int>::FromStr(aStrNum));
    }
    std::string aNewName = "BckUp_" + ToStr(aIMax+1,aNbDig) + "_" + aNameFile;

    CopyFile( aDir+aNameFile , aDir+aNewName);
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
    MMVII_INTERNAL_ASSERT_User(IsDirectory(aDir), eTyUEr::eOpenFile, aDir+" is not a directory!");
    for (fs::directory_iterator itr(aDir); itr!=fs::directory_iterator(); ++itr)
   {
      std::string aName ( itr->path().filename().generic_string().c_str());
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
    MMVII_INTERNAL_ASSERT_User(IsDirectory(aDir), eTyUEr::eOpenFile, aDir+" is not a directory!");
    for (fs::recursive_directory_iterator itr(aDir); itr!=fs::recursive_directory_iterator(); ++itr)
    {
        int aLev = itr.depth();
        if ((aLev>=aLevMin) && (aLev<aLevMax))
        {
           std::string aName(itr->path().generic_string());
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

bool starts_with(const std::string & aFullStr,const std::string & aPrefix)
{
    auto anItFull  = aFullStr.begin();
    auto anItPref = aPrefix.begin();

    while ((anItFull!=aFullStr.end()) && (anItPref!=aPrefix.end()))
    {
         if (*anItFull!=*anItPref) 
            return false;
	 anItFull++;
	 anItPref++;
    }

    return anItPref==aPrefix.end();
}

bool ends_with(const std::string & aFullStr,const std::string & aEnding)
{
    if (aFullStr.size() < aEnding.size())
        return false;
    auto it = aEnding.begin();
    return std::all_of(std::next(aFullStr.begin(),aFullStr.size()-aEnding.size()), aFullStr.end(),
                       [&it](const char& c) { return c == *(it++);});

}


bool contains(const std::string & aFullStr,char aC)
{
	return aFullStr.find(aC) != std::string::npos;
}

bool IsPrefixed(const std::string & aStr,char aSep)
{
	return contains(aStr,aSep);
	// return aStr.find(aSep) != std::string::npos;
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
           StdOut() << "REPLACE ["<< toReplace << "] by : [" << replaceWith << std::endl;
           StdOut() << "in [" << s << "]" << std::endl;
           MMVII_INTERNAL_ASSERT_always(false,"Cannot make subs");
        }
        return "";
    }
    std::string aDup = s;
    return aDup.replace(pos, toReplace.length(), replaceWith);
}

};

