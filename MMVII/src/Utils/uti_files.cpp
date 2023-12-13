#include "MMVII_Error.h"
#include "MMVII_util.h"
#include "MMVII_Ptxd.h"
#include "cMMVII_Appli.h"

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

static std::ios_base::openmode StdFileMode(eFileModeOut aMode)
{
    switch (aMode) {
    case eFileModeOut::CreateText   : return std::ios_base::out | std::ios_base::trunc;
    case eFileModeOut::CreateBinary : return std::ios_base::out | std::ios_base::trunc | std::ios_base::binary;
    case eFileModeOut::AppendText   : return std::ios_base::out | std::ios_base::app;
    case eFileModeOut::AppendBinary : return std::ios_base::out | std::ios_base::app | std::ios_base::binary;
    }
    return std::ios_base::out | std::ios_base::trunc;
}

static std::ios_base::openmode StdFileMode(eFileModeIn aMode)
{
    switch (aMode) {
    case eFileModeIn::Text: return std::ios_base::in;
    case eFileModeIn::Binary: return std::ios_base::in | std::ios_base::binary;
    }
    return std::ios_base::in;
}

/*=============================================*/
/*                                             */
/*            cMMVII_Ofs                       */
/*                                             */
/*=============================================*/


cMMVII_Ofs::cMMVII_Ofs(const std::string & aName,eFileModeOut aMode) :
   mOfs  (aName,StdFileMode(aMode)),
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


cMMVII_Ifs::cMMVII_Ifs(const std::string & aName, eFileModeIn aMode) :
   mIfs  (aName, StdFileMode(aMode)),
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

/** Low level read of file containing nums in fixed format */

static std::string  CurFile;  /// global var to get context, not proud of that
static int          CurLine;  /// global var to get context, not proud of that
template<class Type> inline Type GetV(std::istringstream & iss)
{
    Type aNum;
    iss >> aNum;
    if ( iss.rdstate())
    {
       MMVII_UnclasseUsEr("Bad reading at line  " + ToStr(CurLine) + " of file [" + CurFile + "] , rdstate=" + ToStr((size_t)iss.rdstate()));
    }
    return aNum;
}

int CptOccur(const std::string & aStr,char aC0)
{
   int aCptOccur = 0;
   for (const auto & aC :  aStr)
      aCptOccur +=  (aC==aC0);

   return aCptOccur;
}

int CptSameOccur(const std::string & aStr,const std::string & aStr0)
{
    const char * aC0 = aStr0.c_str();
    MMVII_INTERNAL_ASSERT_tiny(*aC0!=0,"CptSameOccur str empty");

    int aRes = CptOccur(aStr,*(aC0++));
    for (; *aC0; aC0++)
    {
         int aR2 = CptOccur(aStr,*aC0);
	 Fake4ReleaseUseIt(aR2);
	 MMVII_INTERNAL_ASSERT_tiny(aR2==aRes,"Not same counting of " + aStr0);
    }
    return aRes;
}

void  ReadFilesStruct 
      (
	    const std::string &                     aNameFile,
            const std::string &                     aFormat,
            int                                     aL0,
            int                                     aLastL,
            int                                     aComment,
            std::vector<std::vector<std::string>> & aVNames,
            std::vector<cPt3dr>                   & aVXYZ,
            std::vector<cPt3dr>                   & aVWKP,
            std::vector<std::vector<double>>      & aVNums,
	    bool                                    CheckFormat
      )
{
    CurFile = aNameFile;
    if (CheckFormat)
    {
       CptSameOccur(aFormat,"NXYZ");
    }


    if (aLastL<=0) 
       aLastL = 100000000;

    aVNames.clear();
    aVXYZ.clear();
    aVWKP.clear();
    aVNums.clear();

    if (! ExistFile(aNameFile))
    {
       MMVII_UsersErrror(eTyUEr::eOpenFile,std::string("For file ") + aNameFile);
    }
    std::ifstream infile(aNameFile);

    std::string line;
    int aNumL = 0;
    while (std::getline(infile, line))
    {
	    // JOE
MMVII_DEV_WARNING("Dont understand why must add \" \" at end of line ReadFilesStruct");
line += " ";
        CurLine = aNumL+1;  // editor begin at line 1, non 0
        if ((aNumL>=aL0) && (aNumL<aLastL))
	{
            std::istringstream iss(line);
	    int aC0 = iss.get();
            if (aC0 != aComment)
	    {
                iss.unget();
                std::vector<double> aLNum;
                std::vector<std::string> aLNames;
	        cPt3dr aXYZ = cPt3dr::Dummy();
	        cPt3dr aWKP = cPt3dr::Dummy();

                for (const auto & aCar : aFormat)
                {
                    switch (aCar) 
                    {
                         case 'F' : aLNum.push_back(GetV<tREAL8>(iss)); break;
                         case 'X' : aXYZ.x() = GetV<tREAL8>(iss); break;
                         case 'Y' : aXYZ.y() = GetV<tREAL8>(iss); break;
                         case 'Z' : aXYZ.z() = GetV<tREAL8>(iss); break;
                         case 'W' : aWKP.x() = GetV<tREAL8>(iss); break;
                         case 'P' : aWKP.y() = GetV<tREAL8>(iss); break;
                         case 'K' : aWKP.z() = GetV<tREAL8>(iss); break;

			 case 'N' : aLNames.push_back(GetV<std::string>(iss)); break;
			 case 'I' : aLNames.push_back(GetV<std::string>(iss)); break;
			 case 'S' : GetV<std::string>(iss); break;

		         default :
		         break;
                    }
	        }
		aVXYZ.push_back(aXYZ);
		aVWKP.push_back(aWKP);
                aVNames.push_back(aLNames);
                aVNums.push_back(aLNum);
            }
	}
	aNumL++;
    }
}

void  ReadFilesNum (const std::string & aNameFile,const std::string & aFormat,std::vector<std::vector<double>> & aVRes,int aComment)
{
    std::vector<cPt3dr> aVXYZ;
    std::vector<std::vector<std::string>> aVNames;
    ReadFilesStruct 
    (
        aNameFile,aFormat,
        0,1000000,aComment,
        aVNames,
        aVXYZ,aVXYZ,
        aVRes
    );
}

};
