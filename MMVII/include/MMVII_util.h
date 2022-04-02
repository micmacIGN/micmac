#ifndef  _MMVII_Util_H_
#define  _MMVII_Util_H_

namespace MMVII
{

/** \file MMVII_util.h
    \brief Utilitaries for non image related services

    Utilitarries for :
      * string manipulation
      * directory parsing
      * safe file services
*/


class cMMVII_Ofs;
class cMMVII_Ifs;
class cCarLookUpTable;


// ===============================================

/// Indicate if an int is a regular caracter
inline bool IsChar(int aV)
{
   return (aV>=std::numeric_limits<char>::min())&&(aV<std::numeric_limits<char>::max());
}

/**
     Class to tabulate a fonction on char
     For now, due to limited accassor, can only be used in binary mode, done to be reusable.
*/
class cCarLookUpTable
{
     public : 
         void Init(const std::string&,char aC);
         void UnInit(); 
         cCarLookUpTable ();  

         inline char Val(const int & aV) const
         {
             MMVII_INTERNAL_ASSERT_tiny(IsChar(aV),"cCarLookUpTable::Val()");
             return mUTable[aV];
         }
     private :
         // static cGestObjetEmpruntable<cCarLookUpTable>   msGOE;

         char          mDTable[256]; ///< Reserve RAM for the LUT 
         char *        mUTable;      ///< To handle that sometimes char are signed
         std::string   mIns;         ///< Memorize char !=0 to  reset
         bool          mInit;        ///< Is it initialize
};

// Indicate if all "word" of list are in KeyList, use aSpace to separate word
// Si aMes=="SVP"=> No Error just return false, else aMes is error message
bool  CheckIntersect(const std::string & aMes,const std::string & aKeyList,const std::string & aList,const std::string & aSpace);
std::string  Quote(const std::string &);  ///<  Assure a string is between quote, do nothing when begins by "


//  String spliting, post fix, prefix etc ...
            // Just an interface that use cMMVII_Appli::TheAppli()
std::vector<std::string> SplitString(const std::string & aStr,const std::string & aSpace);

// Si PrivPref  "a" => (aaa,)  (a.b.c)  => (a.b,c)
void  SplitStringArround(std::string & aBefore,std::string & aAfter,const std::string & aStr,char aSep,bool SVP=false,bool PrivPref=true);
std::string Prefix(const std::string & aStr,char aSep='.',bool SVP=false,bool PrivPref=true);
std::string LastPrefix(const std::string & aStr,char aSep='.'); ///< No error:  a=> ""  a.b.c => "a.b"
std::string Postfix(const std::string & aStr,char aSep='.',bool SVP=false,bool PrivPref=true);
std::string LastPostfix(const std::string & aStr,char aSep='.'); ///< No error:  a=> ""  a.b.c => "c"


// Direcytory and files names, Rely on boost
void MakeNameDir(std::string & aDir); ///< Add a '/', or equiv, to make a name of directory
bool ExistFile(const std::string & aName);
uintmax_t SizeFile(const std::string & aName);
bool SplitDirAndFile(std::string & aDir,std::string & aFile,const std::string & aDirAndFile,bool ErroNonExist=true);
std::string DirCur(); // as "./" on Unix
std::string DirOfPath(const std::string & aPath,bool ErroNonExist=true);
std::string FileOfPath(const std::string & aPath,bool ErroNonExist=true);


std::string OneUpStd(const std::string & aDir);  ///< Try to supress the as a/B => a/
std::string OneUpDir(const std::string & aDir);  ///< If OneUpStd fail add /../
std::string UpDir(const std::string & aDir,int aNb);

// std::string AbsoluteName(const std::string &); ///< Get absolute name of path; rather pwd than unalias, no good
bool UCaseEqual(const std::string & ,const std::string & ); ///< Case unsensitive equality
bool UCaseBegin(const char * aBegin,const char * aStr); ///< Is aBegin the case UN-sensitive premisse of aStr ?
bool CreateDirectories(const std::string & aDir,bool SVP); ///< Create dir, recurs ?
bool RemoveRecurs(const  std::string & aDir,bool ReMkDir,bool SVP); ///< Purge recursively the directory
bool RemoveFile(const  std::string & aDir,bool SVP); ///< Remove file
void RenameFiles(const std::string & anOldName, const std::string & aNewName); ///< Move/Rename
void CopyFile(const std::string & aName,const std::string & aDest);
bool  RemovePatternFile(const  std::string & aPat,bool SVP); ///< Remove all file corresponding to pattern
void ActionDir(const std::string &,eModeCreateDir);

std::string AddBefore(const std::string & aPath,const std::string & ToAdd); // A/B/C.tif,@  =>  A/B/@C.tif
std::string ChgPostix(const std::string & aPath,const std::string & aPost); // A/B/C.png,tif  =>  A/B/C.tif



bool CaseSBegin(const char * aBegin,const char * aStr); ///< Is aBegin the case SENS-itive premisse of aStr ?
void SkeepWhite(const char * & aC);
char CharDirSeparator();
const std::string & StringDirSeparator();
bool IsDirectory(const std::string & aName);



/// Create a selector associated to a regular expression, by convention return Cste-true selector if string=""
tNameSelector  AllocRegex(const std::string& aRegEx);

/// Exract name of files located in the directory, by return value
std::vector<std::string>  GetFilesFromDir(const std::string & aDir,const tNameSelector& ,bool OnlyRegular=true);
std::vector<std::string>  GetSubDirFromDir(const std::string & aDir,const tNameSelector&);
/// Exract name of files, by ref
void GetFilesFromDir(std::vector<std::string>&,const std::string & aDir,const tNameSelector &,bool OnlyRegular=true);
/// Recursively exract name of files located in the directory, by return value
void RecGetFilesFromDir( std::vector<std::string> & aRes, const std::string & aDir, tNameSelector  aNS,int aLevMin, int aLevMax);
/// Recursively exract name of files, by return value
std::vector<std::string> RecGetFilesFromDir(const std::string & aDir,tNameSelector  aNS,int aLevMin, int aLevMax);

char ToHexacode(int aK);
int  FromHexaCode(char aC);


std::string replaceFirstOccurrence(const std::string& s,const std::string& toRep,const std::string& Rep,bool SVP=false);



/*=============================================*/
/*                                             */
/*            FILES                            */
/*                                             */
/*=============================================*/

/// Secured ofstream
/**
   This class offer do not offer musch more service than std::ofstream, but
   try to offer them from a more secured way. The Write(const Type & ) are 
   typed ; it calss VoidWrite which check the number of byte written (if
   enough debug)

   No need for close() as it done automatically at destroy in std::ofstream.
*/
class cMMVII_Ofs : public cMemCheck
{
    public :
        cMMVII_Ofs(const std::string & aName,bool aModeAppend);
        std::ofstream & Ofs() ;
        const std::string &   Name() const;

        void Write(const tU_INT2 & aVal)    ;
        void Write(const int & aVal)    ;
        void Write(const double & aVal) ;
        void Write(const size_t & aVal) ;
        void Write(const std::string & aVal) ;
   
        ///  Ok for basic type (int, cPtd2r ...), not any composed type ( std::string ...)
        template <class Type> void TplDump(const Type & aVal) {VoidWrite(&aVal,sizeof(aVal));}
        void VoidWrite(const void * aPtr,size_t aNb);
    private :

        std::ofstream  mOfs;
        std::string    mName;
        // bool           mModeAppend; Unsused warning CLANG
};

/// Secured ifstream
/**
   This class is the homologous of cMMVII_Ofs, for input
*/

class cMMVII_Ifs : public cMemCheck
{
    public :
        cMMVII_Ifs(const std::string & aName);
        std::ifstream & Ifs() ;
        const std::string &   Name() const;

        void Read(int & aVal)    ;
        void Read(tU_INT2 & aVal)    ;
        void Read(double & aVal) ;
        void Read(size_t & aVal) ;
        void Read(std::string & aVal) ;

        /// Maybe more convenient as it does require declaration of auxiliary variable
        template<class Type> Type TplRead() {Type aVal; Read(aVal); return aVal;}
        void VoidRead(void * aPtr,size_t aNb);
    private :

         std::ifstream  mIfs;
         std::string   mName;
};

class cMultipleOfs  : public  std::ostream
{
    public :
        cMultipleOfs(std::ostream & aOfs) :
            mOfsCreated(nullptr)
        {
           Add(aOfs);
        }
        cMultipleOfs(const std::string & aS,bool ModeAppend = false)
        {
             mOfsCreated = new cMMVII_Ofs(aS,ModeAppend);
             Add(mOfsCreated->Ofs());
        }
        ~cMultipleOfs()
        {
            delete mOfsCreated;
        }

        void Add(std::ostream & aOfs) {mVOfs.push_back(&aOfs);}
        void Clear() {mVOfs.clear();}

        // template <class Type> cMultipleOfs & operator << (Type & aVal);
        template <class Type> cMultipleOfs & ShowCont (const Type & aCont,const std::string & aGram)
        {
             *this << aGram[0];
             int aK=0;
             for (const auto & aVal : aCont) 
             {
                 if (aK!=0)  *this << aGram[1];
                 *this << aVal;
                 aK++;
             }
             *this << aGram[2];
             return *this;
        }
        // General specialized for vector
        template <class Type> cMultipleOfs & operator << (const std::vector<Type> & aVal)
        {
             return ShowCont(aVal,"[,]");
        }
        // General version
        template <class Type> cMultipleOfs & operator << (const Type & aVal)
        {
             for (const auto & Ofs :  mVOfs)
                 *Ofs << aVal;
             return *this;
        }
    private :
        
        cMultipleOfs(const cMultipleOfs &) = delete;
        cMMVII_Ofs *                mOfsCreated;
        std::vector<std::ostream *> mVOfs;
};

/// For now I have problem with cMultipleOfs << std::endl , tag end of line to come back on it later
#define ENDL "\n"


/** Probably job could be done with boost, but I am not sure I could handle
    precisely output format
*/

class cMMVII_Duration
{
     public :

        cMMVII_Duration();  // Null
        static cMMVII_Duration FromSecond(double aNbSec,eTyUnitTime=eTyUnitTime::eNbVals);
        void Normalise(eTyUnitTime);

        std::string ToDaisyStr(std::string * aFormat=0,bool Full=false) const;

     public :
        std::string  ToString(char aSep,int aNbDigFrac,std::string * aFormat,bool Full) const;

        tINT8 mNbDay;
        tINT8 mNbHour;
        tINT8 mNbMin;
        tINT8 mNbSec;
        tREAL8 mFrac;   // in second
};


};

#endif  //  _MMVII_Util_H_
