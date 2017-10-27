#ifndef  _MMVII_Util_H_
#define  _MMVII_Util_H_

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

// Rely on boost
bool ExistFile(const std::string & aName);
bool SplitDirAndFile(std::string & aDir,std::string & aFile,const std::string & aDirAndFile,bool ErroNonExist=true);
std::string DirCur(); // as "./" on Unix
std::string DirOfPath(const std::string & aPath,bool ErroNonExist=true);
std::string FileOfPath(const std::string & aPath,bool ErroNonExist=true);
std::string UpDir(const std::string & aDir,int aNb=1);


std::string  Quote(const std::string &);  // Assure a string is between quote, do nothing is begins by "




#endif  //  _MMVII_Util_H_
