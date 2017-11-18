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

inline bool IsChar(int aV)
{
   return (aV>=std::numeric_limits<char>::min())&&(aV<std::numeric_limits<char>::max());
}

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

         char          mDTable[256]; ///< 
         char *        mUTable;    ///<
         std::string   mIns;
         bool          mInit;
};

// Indicate if all "word" of list are in KeyList, use aSpace to separate word
// Si aMes=="SVP"=> No Error just return false, else aMes is error message
bool  CheckIntersect(const std::string & aMes,const std::string & aKeyList,const std::string & aList,const std::string & aSpace);
std::string  Quote(const std::string &);  // Assure a string is between quote, do nothing is begins by "


//  String spliting, post fix, prefix etc ...
            // Just an interface that use cMMVII_Appli::TheAppli()
std::vector<std::string> SplitString(const std::string & aStr,const std::string & aSpace);

// Si PrivPref  "a" => (aaa,)  (a.b.c)  => (a.b,c)
void  SplitStringArround(std::string & aBefore,std::string & aAfter,const std::string & aStr,char aSep,bool SVP=false,bool PrivPref=true);
std::string Prefix(const std::string & aStr,char aSep='.',bool SVP=false,bool PrivPref=true);
std::string Postfix(const std::string & aStr,char aSep='.',bool SVP=false,bool PrivPref=true);


// Direcytory and files names, Rely on boost
void MakeNameDir(std::string & aDir); ///< Add a '/', or equiv, to make a name of directory
bool ExistFile(const std::string & aName);
bool SplitDirAndFile(std::string & aDir,std::string & aFile,const std::string & aDirAndFile,bool ErroNonExist=true);
std::string DirCur(); // as "./" on Unix
std::string DirOfPath(const std::string & aPath,bool ErroNonExist=true);
std::string FileOfPath(const std::string & aPath,bool ErroNonExist=true);
std::string UpDir(const std::string & aDir,int aNb=1);
// std::string AbsoluteName(const std::string &); ///< Get absolute name of path; rather pwd than unalias, no good
bool UCaseEqual(const std::string & ,const std::string & ); ///< Case unsensitive equality
bool UCaseBegin(const char * aBegin,const char * aStr); ///< Is aBegin the case UN-sensitive premisse of aStr ?
bool CreateDirectories(const std::string & aDir,bool SVP); ///< Create dir, recurs ?
bool CaseSBegin(const char * aBegin,const char * aStr); ///< Is aBegin the case SENS-itive premisse of aStr ?
void SkeepWhite(const char * & aC);


/// Create a selector associated to a regular expression, by convention return Cste-true selector if string=""
tNameSelector  BoostAllocRegex(const std::string& aRegEx);

/// Exract name of files located in the directory, by return value
std::vector<std::string>  GetFilesFromDir(const std::string & aDir,tNameSelector );
/// Exract name of files, by ref
void GetFilesFromDir(std::vector<std::string>&,const std::string & aDir,tNameSelector);
/// Recursively exract name of files located in the directory, by return value
void RecGetFilesFromDir( std::vector<std::string> & aRes, const std::string & aDir, tNameSelector  aNS,int aLevMin, int aLevMax);
/// Recursively exract name of files, by return value
std::vector<std::string> RecGetFilesFromDir(const std::string & aDir,tNameSelector  aNS,int aLevMin, int aLevMax);




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
*/
class cMMVII_Ofs : public cMemCheck
{
    public :
        cMMVII_Ofs(const std::string & aName);
        std::ofstream & Ofs() ;
        const std::string &   Name() const;

        void Write(const int & aVal)    ;
        void Write(const double & aVal) ;
        void Write(const size_t & aVal) ;
        void Write(const std::string & aVal) ;
   
        ///  Ok for basic type (int, cPtd2r ...), not any composed type ( std::string ...)
        template <class Type> void TplDump(const Type & aVal) {VoidWrite(&aVal,sizeof(aVal));}
    private :
        void VoidWrite(const void * aPtr,size_t aNb);

         std::ofstream  mOfs;
         std::string   mName;
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
        void Read(double & aVal) ;
        void Read(size_t & aVal) ;
        void Read(std::string & aVal) ;

        /// Maybe more convenient as it does require declaration of auxiliary variable
        template<class Type> Type TplRead() {Type aVal; Read(aVal); return aVal;}
    private :
        void VoidRead(void * aPtr,size_t aNb);

         std::ifstream  mIfs;
         std::string   mName;
};


};

#endif  //  _MMVII_Util_H_
