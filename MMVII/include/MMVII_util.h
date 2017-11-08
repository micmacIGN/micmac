#ifndef  _MMVII_Util_H_
#define  _MMVII_Util_H_

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


class cCarLookUpTable
{
     public : 
         void Init(const std::string&,char aC);
         void UnInit(); 
         cCarLookUpTable ();  

         inline char Val(const int & aV) const
         {
             MMVII_INTERNAL_ASSERT_tiny((aV>=0) && (aV<256),"cCarLookUpTable::Val()");
             return mTable[aV];
         }
     private :
         // static cGestObjetEmpruntable<cCarLookUpTable>   msGOE;

         char          mTable[256];
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
bool ExistFile(const std::string & aName);
bool SplitDirAndFile(std::string & aDir,std::string & aFile,const std::string & aDirAndFile,bool ErroNonExist=true);
std::string DirCur(); // as "./" on Unix
std::string DirOfPath(const std::string & aPath,bool ErroNonExist=true);
std::string FileOfPath(const std::string & aPath,bool ErroNonExist=true);
std::string UpDir(const std::string & aDir,int aNb=1);
bool UCaseEqual(const std::string & ,const std::string & ); ///< Case unsensitive equality
bool UCaseBegin(const char * aBegin,const char * aStr); ///< Is aBegin the case unsensitive premisse of aStr ?



/// Generique Interface for all operation on name filtering
class cNameSelector : public cMemCheck
{
    public :
        virtual bool Match(const std::string &) const = 0;
        virtual ~cNameSelector();
};
/// Interface class to Regex

/** This class is a purely interface class to regex.
    I separate from the implementation :
       * because I am not 100% sure I will rely on boost
       * I don't want to slow dow all compliation witrh boost (or other) header 
       * I think it's good policy that class show the minimum required in header
*/

class cInterfRegex : public cNameSelector
{
    public :
        cInterfRegex(const std::string &);
        virtual ~cInterfRegex();
        static cInterfRegex * BoostAlloc();
        const std::string & Name() const;
    private :
        std::string mName;
};

/// Exract name of files, by return value
std::vector<std::string>  GetFilesFromDir(const std::string & aDir,cNameSelector &);
/// Exract name of files, by ref
void GetFilesFromDir(std::vector<std::string>&,const std::string & aDir,cNameSelector &);
/// Recursively exract name of files, by return value
void RecGetFilesFromDir( std::vector<std::string> & aRes, const std::string & aDir, cNameSelector & aNS,int aLevMin, int aLevMax);
/// Recursively exract name of files, by return value
std::vector<std::string> RecGetFilesFromDir(const std::string & aDir,cNameSelector & aNS,int aLevMin, int aLevMax);



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




#endif  //  _MMVII_Util_H_
