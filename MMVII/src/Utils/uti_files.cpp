#include "MMVII_Error.h"
#include "MMVII_util.h"

namespace MMVII
{

template<class Type> std::unique_ptr<Type>  NNfs(const std::string & aNameFile,const std::string & aMode,const std::string & aMes)
{
    std::unique_ptr<Type> aRes (new Type(aNameFile));

    MMVII_INTERNAL_ASSERT_User
    (
          aRes->good(),
          eTyUEr::eOpenFile,
          "Cannot open file : "  + aNameFile + " ,Mode=" + aMode+  " ,context=" + aMes
    );

    return aRes;
}


std::unique_ptr<std::ifstream>  NNIfs(const std::string & aNameFile,const std::string aMes)
{
   return NNfs<std::ifstream>(aNameFile,"READ",aMes);
}
std::unique_ptr<std::ofstream>  NNOfs(const std::string & aNameFile,const std::string aMes)
{
   return NNfs<std::ofstream>(aNameFile,"WRITE",aMes);
}

/*=============================================*/
/*                                             */
/*            cMMVII_Ofs                       */
/*                                             */
/*=============================================*/


cMMVII_Ofs::cMMVII_Ofs(const std::string & aName,bool ModeAppend) :
   mOfs  (aName,ModeAppend ? std::ios_base::app : std::ios_base::out),
   mName (aName)
{
    MMVII_INTERNAL_ASSERT_User
    (
         mOfs.good(),
         eTyUEr::eOpenFile,
         "Cannot open file : "  + mName + " in mode write"
    );
}

const std::string &   cMMVII_Ofs::Name() const
{
   return mName;
}

std::ofstream & cMMVII_Ofs::Ofs() 
{
#if (The_MMVII_DebugLevel>=The_MMVII_DebugLevel_InternalError_tiny)
   if (!mOfs.good())
   {
       MMVII_UsersErrror(eTyUEr::eWriteFile,"Bad file for "+mName);
   }
#endif
   return mOfs;
}


void cMMVII_Ofs::VoidWrite(const void * aPtr,size_t aNb) 
{ 
#if (The_MMVII_DebugLevel>=The_MMVII_DebugLevel_InternalError_tiny)
   std::streampos aPos0 = mOfs.tellp();
#endif
   mOfs.write(static_cast<const char *>(aPtr),aNb); 
#if (The_MMVII_DebugLevel>=The_MMVII_DebugLevel_InternalError_tiny)
    bool Ok = mOfs.tellp() == (aPos0+std::streampos(aNb));
    if (!Ok)
    {
       MMVII_INTERNAL_ASSERT_tiny
       (
           false,
           std::string("Error in write for file ") + mName
       );
    }
#endif
}

void cMMVII_Ofs::Write(const tU_INT2 & aVal){ VoidWrite(&aVal,sizeof(aVal)); }
void cMMVII_Ofs::Write(const int & aVal)    { VoidWrite(&aVal,sizeof(aVal)); }
void cMMVII_Ofs::Write(const double & aVal) { VoidWrite(&aVal,sizeof(aVal)); }
void cMMVII_Ofs::Write(const size_t & aVal) { VoidWrite(&aVal,sizeof(aVal)); }


void cMMVII_Ofs::Write(const std::string & aVal) 
{ 
   size_t aSz = aVal.size();
   Write(aSz);
   VoidWrite(aVal.c_str(),aSz);
}

/*=============================================*/
/*                                             */
/*            cMMVII_Ifs                       */
/*                                             */
/*=============================================*/


cMMVII_Ifs::cMMVII_Ifs(const std::string & aName) :
   mIfs  (aName),
   mName (aName)
{
    MMVII_INTERNAL_ASSERT_User
    (
        mIfs.good(),
        eTyUEr::eOpenFile,
        "Cannot open file : "  + mName + " in mode read"
    );
}

const std::string &   cMMVII_Ifs::Name() const
{
   return mName;
}

std::ifstream & cMMVII_Ifs::Ifs() 
{
#if (The_MMVII_DebugLevel>=The_MMVII_DebugLevel_InternalError_tiny)
   if (!mIfs.good())
   {
       MMVII_UsersErrror(eTyUEr::eReadFile,"Bad file for "+mName);
   }
#endif
   return mIfs;
}


void cMMVII_Ifs::VoidRead(void * aPtr,size_t aNb) 
{ 
#if (The_MMVII_DebugLevel>=The_MMVII_DebugLevel_InternalError_tiny)
   std::streampos aPos0 = mIfs.tellg();
#endif
   mIfs.read(static_cast<char *>(aPtr),aNb); 
#if (The_MMVII_DebugLevel>=The_MMVII_DebugLevel_InternalError_tiny)
    bool Ok = mIfs.tellg() == (aPos0+std::streampos(aNb));
    if (!Ok)
    {
       MMVII_INTERNAL_ASSERT_tiny
       (
           false,
           std::string("Error in read for file ") + mName
       );
    }
#endif
}

void cMMVII_Ifs::Read(tU_INT2 & aVal) { VoidRead(&aVal,sizeof(aVal)); }
void cMMVII_Ifs::Read(int & aVal)     { VoidRead(&aVal,sizeof(aVal)); }
void cMMVII_Ifs::Read(double & aVal)  { VoidRead(&aVal,sizeof(aVal)); }
void cMMVII_Ifs::Read(size_t & aVal)  { VoidRead(&aVal,sizeof(aVal)); }

void cMMVII_Ifs::Read(std::string & aVal )
{ 
   size_t aSz = TplRead<size_t>();
   aVal.resize(aSz);
   VoidRead(const_cast<char *>(aVal.c_str()),aSz);
}

/** Lox level read of file containing nums in fixed format */

void  ReadFilesNum(const std::string & aFormat,std::vector<std::vector<double>> & aVRes,const std::string & aNameFile)
{
    aVRes.clear();
    if (! ExistFile(aNameFile))
    {
       MMVII_UsersErrror(eTyUEr::eOpenFile,std::string("For file ") + aNameFile);
    }
    std::ifstream infile(aNameFile);

    std::string line;
    while (std::getline(infile, line))
    {
        std::istringstream iss(line);
        std::vector<double> aLD;
        for (const auto & aCar : aFormat)
        {
            if (aCar=='F')
            {
               tREAL8 aNum;
               iss >> aNum;
               aLD.push_back(aNum);
            }
            else if (aCar=='S')
            {
               std::string anAtom;
               iss >> anAtom;
            }
            else
            {
                 MMVII_UsersErrror(eTyUEr::eUnClassedError,"Bad string format");
            }
        }
        aVRes.push_back(aLD);
    }
}


};

