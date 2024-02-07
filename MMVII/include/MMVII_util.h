#ifndef  _MMVII_Util_H_
#define  _MMVII_Util_H_

#include <numeric>
#include <sstream>

#include "MMVII_nums.h"
#include "MMVII_memory.h"


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
	 void InitId(char aC1,char aC2);
	 void Chg1C(char aC1,char aC2);



         inline char Val(const int & aV) const
         {
             MMVII_INTERNAL_ASSERT_tiny(IsChar(aV),"cCarLookUpTable::Val()");
             return mUTable[aV];
         }

	 /// make a translation using lut, dont maintain null char
	 std::string  Translate(const std::string &) const;
     private :
         // static cGestObjetEmpruntable<cCarLookUpTable>   msGOE;

         char          mDTable[256]; ///< Reserve RAM for the LUT 
         char *        mUTable;      ///< To handle that sometimes char are signed
         std::string   mIns;         ///< Memorize char !=0 to  reset
         bool          mInit;        ///< Is it initialize
         bool          mReUsable;    ///< If InitId of Chg1C used -> no longer reusable
};

// Indicate if all "word" of list are in KeyList, use aSpace to separate word
// Si aMes=="SVP"=> No Error just return false, else aMes is error message
bool  CheckIntersect(const std::string & aMes,const std::string & aKeyList,const std::string & aList,const std::string & aSpace);
std::string  Quote(const std::string &);  ///<  Assure a string is between quote, do nothing when begins by "
///   Transformate a string to make it a standard identifier (+- a C++ accpetable symbol) , rather conservative & basic
///  " " -> "_"  , accept letter digit "-", refuse all other
std::string  ToStandardStringIdent(const std::string &);  


//  String spliting, post fix, prefix etc ...
            // Just an interface that use cMMVII_Appli::TheAppli()
std::vector<std::string> SplitString(const std::string & aStr,const std::string & aSpace);

// Si PrivPref  "a" => (aaa,)  (a.b.c)  => (a.b,c)
void  SplitStringArround(std::string & aBefore,std::string & aAfter,const std::string & aStr,char aSep,bool SVP=false,bool PrivPref=true);
bool IsPrefixed(const std::string & aStr,char aSep='.');
std::string Prefix(const std::string & aStr,char aSep='.',bool SVP=false,bool PrivPref=true);
std::string LastPrefix(const std::string & aStr,char aSep='.'); ///< No error:  a=> ""  a.b.c => "a.b"
std::string Postfix(const std::string & aStr,char aSep='.',bool SVP=false,bool PrivPref=true);
std::string LastPostfix(const std::string & aStr,char aSep='.'); ///< No error:  a=> ""  a.b.c => "c"

bool starts_with(const std::string & aFullStr,const std::string & aBegining); /// as c++20  std::string.starts_with
bool ends_with(const std::string & aFullStr,const std::string & aEnding); /// as c++20  std::string.ends_with
bool contains(const std::string & aFullStr,const std::string & aEnding); /// as c++23  std::string.contains TO IMPLEMENT

// Direcytory and files names, Rely on std::filesystem
void MakeNameDir(std::string & aDir); ///< Add a '/', or equiv, to make a name of directory
bool ExistFile(const std::string & aName);
uintmax_t SizeFile(const std::string & aName);
bool SplitDirAndFile(std::string & aDir,std::string & aFile,const std::string & aDirAndFile,bool ErroNonExist=true);
std::string DirCur(); // as "./" on Unix
std::string DirOfPath(const std::string & aPath,bool ErroNonExist=true);
std::string FileOfPath(const std::string & aPath,bool ErroNonExist=true);
std::string UpDir(const std::string & aDir);

// std::string AbsoluteName(const std::string &); ///< Get absolute name of path; rather pwd than unalias, no good
std::string ToLower(const std::string &  aStr);  ///< return lower case version
bool UCaseEqual(const std::string & ,const std::string & ); ///< Case unsensitive equality
bool UCaseBegin(const char * aBegin,const char * aStr); ///< Is aBegin the case UN-sensitive premisse of aStr ?
bool UCaseMember(const std::vector<std::string> & aVec,const std::string & aName); ///< is Name meber of vec, insensitive way
bool CreateDirectories(const std::string & aDir,bool SVP); ///< Create dir, recurs ?
bool RemoveRecurs(const  std::string & aDir,bool ReMkDir,bool SVP); ///< Purge recursively the directory
bool RemoveFile(const  std::string & aDir,bool SVP); ///< Remove file
void RenameFiles(const std::string & anOldName, const std::string & aNewName); ///< Move/Rename
void CopyFile(const std::string & aName,const std::string & aDest);
void CopyPatternFile(const std::string & aDirIn,const std::string & aPattern,const std::string & aDirOut);
bool  RemovePatternFile(const  std::string & aPat,bool SVP); ///< Remove all file corresponding to pattern
void ActionDir(const std::string &,eModeCreateDir);
/// Generate a Back-Up by creating a copy with a new num
void  MakeBckUp(const std::string & aDir,const std::string & aNameFile,int aNbDig);




std::string AddBefore(const std::string & aPath,const std::string & ToAdd); // A/B/C.tif,@  =>  A/B/@C.tif
std::string AddAfter(const std::string & aPath,const std::string & ToAdd); // A/B/C.tif,@  =>  A/B/@C.tif
std::string ChgPostix(const std::string & aPath,const std::string & aPost); // A/B/C.png,tif  =>  A/B/C.tif
std::string AddAfterAndChgPost(const std::string & aPath,const std::string & ToAdd,const std::string & aPost); // A/B/C.tif,@  =>  A/B/@C.tif

const std::string & StrWDef(const std::string & aValue,const std::string & aDef); ///< Return value if !="", else default



bool CaseSBegin(const char * aBegin,const char * aStr); ///< Is aBegin the case SENS-itive premisse of aStr ?
void SkeepWhite(const char * & aC);
const std::string & StringDirSeparator();
bool IsDirectory(const std::string & aName);



/// Create a selector associated to a regular expression, by convention return Cste-true selector if string=""
tNameSelector  AllocRegex(const std::string& aRegEx);
/// Indicate if name match patter, uses AllocRegex
bool  MatchRegex(const std::string& aName,const std::string& aPat);
/// replace a pattern :  yy(.*)zz , A$1 , yytotozz  => Atoto
std::string ReplacePattern(const std::string & aPattern,const std::string & aSubst,const std::string & aString);

///  return the kth parenthesis sub-expr :  yy(.*)zz , 1 , yytotozz  => toto
std::string PatternKthSubExpr(const std::string & aPattern,int aKThSub,const std::string & aString);



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
enum class eFileModeOut {
    CreateText,
    CreateBinary,
    AppendText,
    AppendBinary,
};

enum class eFileModeIn {
    Text,
    Binary
};

class cMMVII_Ofs : public cMemCheck
{
    public :
        cMMVII_Ofs(const std::string & aName, eFileModeOut aMode);
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
        cMMVII_Ifs(const std::string & aName, eFileModeIn aMode);
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

class cMultipleOfs
{
    public :
        cMultipleOfs(std::ostream & aOfs) :
            mOfsCreated(nullptr)
        {
           Add(aOfs);
        }
        cMultipleOfs(const std::string & aS,eFileModeOut aMode)
        {
             mOfsCreated = new cMMVII_Ofs(aS,aMode);
             Add(mOfsCreated->Ofs());
        }
        ~cMultipleOfs()
        {
            delete mOfsCreated;
        }

        void Add(std::ostream & aOfs) {mVOfs.push_back(&aOfs);}
        void Clear() {mVOfs.clear();}

        cMultipleOfs& flush() {
            for (const auto & Ofs :  mVOfs)
                Ofs->flush();
            return *this;
        }

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
        template <class T1,class T2> cMultipleOfs & operator << (const std::pair<T1,T2> &aPair)
        {
            *this  << "{" << aPair.first  << "," << aPair.second << "}";
            return *this;
        }
        // General version
        template <class Type> cMultipleOfs & operator << (const Type & aVal)
        {
             for (const auto & Ofs :  mVOfs)
                 *Ofs << aVal;
             return *this;
        }

        // CM: This will allow standard stream manipulator to work with cMultipleOgs (std::endl, ...)
        cMultipleOfs& operator<<(std::ostream&(*f)(std::ostream&))
        {
             for (const auto & Ofs :  mVOfs)
                 *Ofs << f;
             return *this;
        }

    private :
        
        cMultipleOfs(const cMultipleOfs &) = delete;
        cMMVII_Ofs *                mOfsCreated;
        std::vector<std::ostream *> mVOfs;
};



/** Probably job could be done with std::duration, but I am not sure I could handle
    precisely output format
*/

class cMMVII_Duration
{
     public :

        cMMVII_Duration();  // Null
        static cMMVII_Duration FromSecond(double aNbSec,eTyUnitTime=eTyUnitTime::eNbVals);
        void Normalise(eTyUnitTime);

        std::string ToDaisyStr(std::string * aFormat=nullptr,bool Full=false) const;

     public :
        std::string  ToString(char aSep,int aNbDigFrac,std::string * aFormat,bool Full) const;

        tINT8 mNbDay;
        tINT8 mNbHour;
        tINT8 mNbMin;
        tINT8 mNbSec;
        tREAL8 mFrac;   // in second
};

/** Class for storing set of int , can be use econmically for sparse
    big set, if recycledwith Clear() */
 
class cSetIntDyn
{
     public :
          cSetIntDyn(size_t aNb);  ///< Create empty set
          cSetIntDyn(size_t aNb,const std::vector<size_t> & AddInit);  ///< Add initial values
          void AddInd(size_t aK);  ///< Add an element, adpat sizeof vector
          void Clear();
          void AddIndFixe(size_t aK)  ///< Add an element, assume sizeof vector of
          {
               if (!mOccupied.at(aK))
               {
                   mOccupied[aK] = true;
                   mVIndOcc.push_back(aK);
               }
          }
          size_t NbElem() const {return mVIndOcc.size();}
	  /// Generally order is of no importance, but if it has, can sort it in increasing order
	  void SortInd();
	  ///   Actualize  mVInvertInd, do the SortInd() that is required
	  void MakeInvertIndex();

          std::vector<bool>    mOccupied;     ///< direct acces to the belonging  [0 1 0 0 1 0 1 0]
          std::vector<size_t>  mVIndOcc;      ///< list  of element               [1 4 6]
          std::vector<int   >  mVInvertInd;   ///< if created, give for an index its rank [ _ 0 _ _  1 _ 2 _]
};

class cRandSubSetGenerator
{
      public :
          cRandSubSetGenerator(size_t aNb);

          void NewSubset(std::vector<size_t> & aRes,size_t aCard);

      private :
          size_t      mNb;
          cSetIntDyn mSetDyn;  /// use to accelerate computation
};


/** Class for representing a set of int simply as a vector of int,
    usefull for K among N when N is big 
    make no control of duplicate,  just an interface to vector<int> as a set
*/

class cSetIExtension
{
    public :
         cSetIExtension();
         cSetIExtension(const std::vector<size_t>&);
         static cSetIExtension EmptySet();
         void AddElem(size_t);

         std::vector<size_t>  mElems;
};

/** Generate Q subset of cardinal K [0,N], all different,  if Q too big, truncated */
void GenRanQsubCardKAmongN(std::vector<cSetIExtension> & aRes,int aQ,int aK,int aN);

class cParamRansac
{   
    public :

        int  NbTestOfErrAdm(int aNbSample) const;
        cParamRansac(double  aProba1Err,double anErrAdm);
    private :
        double mProba1Err;
        double mErrAdm;
};





};

#endif  //  _MMVII_Util_H_
