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

/*
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
*/

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


/* *************************************************** */
/*                                                     */
/*                 cReadFilesStruct                    */
/*                                                     */
/* *************************************************** */

cReadFilesStruct::cReadFilesStruct
( 
      const std::string &  aNameFile,
      const std::string & aFormat,
      int aL0,
      int aLastL, 
      int  aComment
)  :
      mNameFile     (aNameFile),
      mFormat       (aFormat),
      mL0           (aL0),
      mLastL        ((aLastL<0) ? 1e9 : aLastL),
      mComment      (aComment),
      mMemoLinesInt (false)
{
}



// const std::vector<std::vector<std::string>>& cReadFilesStruct::VNames () const { return GetVect(mVNames); }
const std::vector<cPt3dr>& cReadFilesStruct::VXYZ () const { return GetVect(mVXYZ); }
const std::vector<cPt3dr>& cReadFilesStruct::VWPK () const { return GetVect(mVWPK); }
const std::vector<cPt2dr>& cReadFilesStruct::Vij  () const { return GetVect(mVij);  }
const std::vector<std::vector<double>> & cReadFilesStruct::VNums () const {return  GetVect(mVNums);}
const std::vector<std::vector<int>> & cReadFilesStruct::VInts () const {return  GetVect(mVInts);}
const std::vector<std::string>& cReadFilesStruct::VNameIm () const { return GetVect(mVNameIm); }
const std::vector<std::string>& cReadFilesStruct::VNamePt () const { return GetVect(mVNamePt); }

const std::vector<std::string>& cReadFilesStruct::VLinesInit () const { return GetVect(mVLinesInit); }

const std::vector<std::vector<std::string>>  & cReadFilesStruct::VStrings () const {return mVStrings;}


int cReadFilesStruct::NbRead() const { return mNbLineRead; }

void cReadFilesStruct::SetMemoLinesInit() 
{
    mMemoLinesInt = true; 
}


void cReadFilesStruct::Read()
{
    CurFile = mNameFile;


    /*
    char  aLut[256];
    for (int aK=0 ; aK<256 ; aK++)
    {
        aLut[aK] = aK;
    }
    cCarLookUpTable aLUT;
    */

    mVNameIm.clear();
    mVNamePt.clear();
    mVXYZ.clear();
    mVij.clear();
    mVWPK.clear();
    mVNums.clear();
    mVLinesInit.clear();

    if (! ExistFile(mNameFile))
    {
       MMVII_UsersErrror(eTyUEr::eOpenFile,std::string("For file ") + mNameFile);
    }
    std::ifstream infile(mNameFile);

    std::string line;
    mNbLineRead = 0;
    int aNumL = 0;
    while (std::getline(infile, line))
    {
// StdOut() << "LllllInnneee=" << aNumL << "\n";
	// JOE
        MMVII_DEV_WARNING("Dont understand why must add \" \" at end of line ReadFilesStruct");
        line += " ";
        CurLine = aNumL+1;  // editor begin at line 1, non 0
	/*
        for (size_t aK=0 ; aK<line.size() ; aK++)
            line[aK] = aLut[line[aK]];
	    */

        if ((aNumL>=mL0) && (aNumL<mLastL))
	{
            if (mMemoLinesInt)
	    {
	        mVLinesInit.push_back(line);
            }
            std::istringstream iss(line);
	    int aC0 = iss.get();
            if (aC0 != mComment)
	    {
	        mNbLineRead++;
                iss.unget();  // as C0 is not  a comment it will have to be parsed (!!=> Ok because there is only one)
			   
                std::vector<double> aLNum;
                std::vector<int>            aLInt;
                std::vector<std::string>    aLString;
	        cPt3dr aXYZ = cPt3dr::Dummy();
	        cPt3dr aWPK = cPt3dr::Dummy();
	        cPt2dr aij =  cPt2dr::Dummy();
                std::string aNameIm;
                std::string aNamePt;
                // std::string aNamePt;

		int  initXYZ=0;
		int  initWPK=0;
		int  initF=0;
		int  initij=0;
		int  initIm=0;
		int  initPt=0;
		int  initI=0;
		int  initString=0;


                for (const auto & aCar : mFormat)
                {
                    switch (aCar) 
                    {
                         case 'F' : aLNum.push_back(GetV<tREAL8>(iss));   initF++; break;
                         case 'E' : aLInt.push_back(GetV<int>(iss));      initI++; break;
                         case 'X' : aXYZ.x() = GetV<tREAL8>(iss);         initXYZ++; break;
                         case 'Y' : aXYZ.y() = GetV<tREAL8>(iss);         initXYZ++;  break;
                         case 'Z' : aXYZ.z() = GetV<tREAL8>(iss);         initXYZ++; break;

                         case 'W' : aWPK.x() = GetV<tREAL8>(iss);         initWPK++; break;
                         case 'P' : aWPK.y() = GetV<tREAL8>(iss);         initWPK++; break;
                         case 'K' : aWPK.z() = GetV<tREAL8>(iss);         initWPK++; break;

			 case 'i' : aij.x() = GetV<tREAL8>(iss);          initij++;  break;
                         case 'j' : aij.y() = GetV<tREAL8>(iss);          initij++;  break;

			 case 'N' : aNamePt = GetV<std::string>(iss);     initPt++; break;
			 case 'I' : aNameIm = GetV<std::string>(iss);     initIm++; break;
			 case 'S' : aLString.push_back(GetV<std::string>(iss)); initString++; break;
			 case '#' : GetV<std::string>(iss); break;

		         default :
                              MMVII_INTERNAL_ERROR(std::string(("Unhandled car in cReadFilesStruct::Read=") + aCar)+"]");
		         break;
                    }
	        }
		if (initXYZ) mVXYZ.push_back(aXYZ);
		if (initWPK) mVWPK.push_back(aWPK);
		if (initij) mVij.push_back(aij);
		if (initF) mVNums.push_back(aLNum);
		if (initI) mVInts.push_back(aLInt);
		if (initIm) mVNameIm.push_back(aNameIm);
		if (initPt) mVNamePt.push_back(aNamePt);
		if (initString) mVStrings.push_back(aLString);
            }
	}
	aNumL++;
    }
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
       aLastL = 1e9;

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

                         // case 'i' : aWKP.x() = GetV<tREAL8>(iss); break;
                         // case 'j' : aWKP.y() = GetV<tREAL8>(iss); break;

			 case 'N' : aLNames.push_back(GetV<std::string>(iss)); break;
			 case 'I' : aLNames.push_back(GetV<std::string>(iss)); break;
                        case  'A' : aLNames.push_back(GetV<std::string>(iss)); break;
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
