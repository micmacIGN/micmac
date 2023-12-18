#include "cMMVII_Appli.h"
#include "MMVII_Sys.h"
#include "MMVII_DeclareCste.h"
#include "MMVII_2Include_Serial_Tpl.h"

namespace MMVII
{

cAppliBenchAnswer cMMVII_Appli::BenchAnswer() const
{
   return cAppliBenchAnswer(false,0.0);
}


int  cMMVII_Appli::ExecuteBench(cParamExeBench &)
{
    MMVII_INTERNAL_ERROR("No Bench for"+ mSpecs.Name());
    return EXIT_FAILURE;
}



/*  ============================================== */
/*                                                 */
/*                cColStrAObl                      */
/*                                                 */
/*  ============================================== */


const  cColStrAObl::tCont & cColStrAObl::V() const {return mV;}
cColStrAObl &  cColStrAObl::operator << (const std::string & aVal) {mV.push_back(aVal); return *this;}
void cColStrAObl::clear() {mV.clear();}
cColStrAObl::cColStrAObl() {}


/*  ============================================== */
/*                                                 */
/*                cColStrAOpt                      */
/*                                                 */
/*  ============================================== */


const  cColStrAOpt::tCont & cColStrAOpt::V() const {return mV;}
cColStrAOpt &  cColStrAOpt::operator << (const t2S & aVal) {mV.push_back(aVal); return *this;}
void cColStrAOpt::clear() {mV.clear();}
cColStrAOpt::cColStrAOpt() {}

const cColStrAOpt cColStrAOpt::Empty;

cColStrAOpt::cColStrAOpt(cExplicitCopy,const cColStrAOpt& aCSAO)  :
  mV (aCSAO.mV)
{
}

/*  ============================================== */
/*                                                 */
/*                cColStrAOpt                      */
/*                                                 */
/*  ============================================== */

cParamCallSys::cParamCallSys() :
    mSpec(nullptr)
{
}

cParamCallSys::cParamCallSys(const cSpecMMVII_Appli & aSpec) :
   mSpec   (&aSpec)
{
}

int cParamCallSys::Execute(bool forceExternal) const
{
   if (mSpec && !forceExternal)
   {
        int aRes = mSpec->AllocExecuteDestruct(mArgv);
       return aRes;
   }
   else
   {
       int aRes = GlobSysCall(*this,false);
       return aRes;
   }
}

#if (THE_MACRO_MMVII_SYS==MMVII_SYS_L || THE_MACRO_MMVII_SYS==MMVII_SYS_A)
static std::string QuoteCmdLine(const std::string& aParam)
{
    constexpr const char* SingleQuoting="!$`\\";
    constexpr const char* DoubleQuoting=" \t~#*?()[]{}<>;&|\"";

    if (aParam.size() == 0)
        return "\"\"";
    if (aParam.find_first_of(SingleQuoting) != std::string::npos) {
           std::string result="'";
           for (const auto& c: aParam) {
               if (c=='\'')
                   result += "'\\''";
               else
                   result += c;
           }
           return result + "'";
    }
    if (aParam.find_first_of(DoubleQuoting) != std::string::npos) {
           std::string result="\"";
           for (const auto& c: aParam) {
               if (c=='"')
                   result += "\\\"";
               else
                   result += c;
           }
           return result + "\"";
    }
    return aParam;
}

#elif (THE_MACRO_MMVII_SYS==MMVII_SYS_W)
static std::string QuoteCmdLine(const std::string& aParam)
{
    constexpr const char* DoubleQuoting=" \t*?&|()<>^\"";

    if (aParam.size() == 0)
        return "\"\"";
    if (aParam.find_first_of(DoubleQuoting) != std::string::npos) {
           std::string result="\"";
           for (const auto& c: aParam) {
               if (c=='"')
                   result += "\\\"";
               else
                   result += c;
           }
           return result + "\"";
    }
    return aParam;
}
#else
#   error Invalid value for macro THE_MACRO_MMVII_SYS
#endif

void cParamCallSys::AddArgs(const std::string & aNewArg)
{
   if (mArgv.size() != 0)
       mCom += " ";
   mCom += QuoteCmdLine(aNewArg);
   mArgv.push_back(aNewArg);
}

const std::string &  cParamCallSys::Com() const
{
   return mCom;
}


/*  ============================================== */
/*                                                 */
/*                cMMVII_Appli                     */
/*                                                 */
/*  ============================================== */

// May be used again for testing value inside initialization
/*
template <class Type> Type PrintArg(const Type & aVal,const std::string & aName)
{
    Std Out() << " For " << aName << " V=" << aVal << "\n";
    return aVal;
}
*/

// ========================= 3 Main function :
// 
//        cMMVII_Appli::~cMMVII_Appli()
//        cMMVII_Appli::cMMVII_Appli ( int argc, char ** argv, const cSpecMMVII_Appli & aSpec) 
//        void cMMVII_Appli::InitParam() => main initialisation must be done after Cstrctor as call virtual methods


std::set<cObj2DelAtEnd *>       cMMVII_Appli::mVectObj2DelAtEnd;

void cMMVII_Appli::AddObj2DelAtEnd(cObj2DelAtEnd * aPtrO)
{
     mVectObj2DelAtEnd.insert(aPtrO);
}

cMMVII_Appli::~cMMVII_Appli()
{
   DoMergeReport();
   if (mMainAppliInsideP)
   {
        for (auto  aPtrO : mVectObj2DelAtEnd)
            delete aPtrO;
   }

   if (mForExe)
   {
      if (! mModeHelp && !mModeArgsSpec)
      {
         RenameFiles(NameFileLog(false),NameFileLog(true));
         LogCommandOut(NameFileLog(true),false);
      }

      if (mGlobalMainAppli)
      {
         LogCommandOut(mFileLogTop,true);
      }
   }

   msInDstructor = (TheStackAppli.size()<=1);  // avoid problem with StdOut 
   // if (msInDstructor) FreeRandom();   // Free memory only top called do it
   if(mForExe)
   {
      AssertInitParam();
      // ======= delete mSetInit;
      mArgObl.clear();
      mArgFac.clear();
   }


   MMVII_INTERNAL_ASSERT_strong(ExistAppli(),"check in Appli Destructor");
   MMVII_INTERNAL_ASSERT_strong(this==TheStackAppli.back(),"check in Appli Destructor");
   TheStackAppli.pop_back();
   mStdCout.Clear();
   // Verifie que tout ce qui a ete alloue a ete desalloue 
   // cMemManager::CheckRestoration(mMemStateBegin);
   mMemStateBegin.SetCheckAtDestroy();
}

/*
static std::vector<std::string> InitFromArgcArgv(int argc, char ** argv)
{ 
   std::vector<std::string> aRes;
   for (int aK=0 ; aK<argc; aK++)
       aRes.push_back(argv[aK]);
   return aRes;
}
*/

template <class Type> const Type & MessageInCstr(const Type & aVal,const std::string & aMsg,int aLine)
{
    StdOut() << aMsg << " at line " << aLine << std::endl;
    return aVal;
}

cMMVII_Appli::cMMVII_Appli
(
      const std::vector<std::string> & aVArgcv,
      const cSpecMMVII_Appli & aSpec,
      tVSPO                    aVSPO
)  :
   cMMVII_Ap_CPU(),
   mMemStateBegin (cMemManager::CurState()),
   mArgv          (aVArgcv),
   mArgc          (mArgv.size()),
   mSpecs         (aSpec),
   mForExe        (true),
   mDirProject    (DirCur()),
   mFileLogTop    (""),
   mModeHelp      (false),
   mDoGlobHelp    (false),
   mDoInternalHelp(false),
   mModeArgsSpec  (false),
   mShowAll       (false),
   mLevelCall     (0),
   mKthCall       (0),
   mSetInit       (cExtSet<const void *>(eTySC::US)),
   mSetVarsSpecObl (cExtSet<const void *>(eTySC::US)),
   mSetVarsSpecFac (cExtSet<const void *>(eTySC::US)),
   mInitParamDone (false),
   mVMainSets     (NbMaxMainSets,tNameSet(eTySC::NonInit)),
   mResulMultiS   (EXIT_FAILURE),
   mRMSWasUsed    (false),
   mNumOutPut     (0),
   mOutPutV1      (false),
   mOutPutV2      (false),
   mHasInputV1    (false),
   mHasInputV2    (false),
   mStdCout       (std::cout),
   mSeedRand      (msDefSeedRand), // In constructor, don't use virtual, wait ...
   mVSPO          (aVSPO),
   mCarPPrefOut   (MMVII_StdDest),
   mCarPPrefIn    (MMVII_StdDest),
   mTiePPrefOut   (MMVII_StdDest),
   mTiePPrefIn    (MMVII_StdDest),
   mIsInBenchMode (false),
   mReportSubDir   (""),
   mPatternInitGMA (MMVII_NONE)
{
   mNumCallInsideP = TheNbCallInsideP;
   TheNbCallInsideP++;
   
   mMainAppliInsideP = (mNumCallInsideP==0);
   TheStackAppli.push_back(this);
   /// Minimal consistency test for installation, does the MicMac binary exist ?
   MMVII_INTERNAL_ASSERT_always(ExistFile(mFullBin),"Could not find MMVII binary (tried with " +  mFullBin + ")");
}

struct cSpecifProfileUserMMVII
{
     public :
         std::string mNameProfile;
};

void AddData(const cAuxAr2007 & anAux,cSpecifProfileUserMMVII & aSpec)
{
     AddData(cAuxAr2007("NameProfile",anAux),aSpec.mNameProfile);
}

void AddData(const cAuxAr2007 & anAux,cParamProfile & aProfile)
{
     AddData(cAuxAr2007("UserName",anAux),aProfile.mUserName);
     AddData(cAuxAr2007("NbProcMax",anAux),aProfile.mNbProcMax);
     EnumAddData(anAux,aProfile.mTaggedDefSerial,"TaggedSerialMode");
     EnumAddData(anAux,aProfile.mVectDefSerial,"VectSerialMode");
}


void cMMVII_Appli::InitMMVIIDirs(const std::string& aMMVIIDir)
{
    if (aMMVIIDir.length() == 0)
        return;

    mTopDirMMVII       = aMMVIIDir;
    MakeNameDir(mTopDirMMVII);
    mDirBinMMVII       = mTopDirMMVII + "bin" + StringDirSeparator();
    mFullBin           = mDirBinMMVII + MMVIIBin2007;
    mDirMicMacv1       = UpDir(mTopDirMMVII);
    mDirMicMacv2       = mTopDirMMVII;
    mDirTestMMVII      = mDirMicMacv2 + MMVIITestDir;
    mDirRessourcesMMVII      = mDirMicMacv2 + MMVIIRessourcesDir;
    mDirLocalParameters      = mDirMicMacv2 + MMVIILocalParametersDir;
    mTmpDirTestMMVII   = mDirTestMMVII + "Tmp" + StringDirSeparator();
    mInputDirTestMMVII = mDirTestMMVII + "Input" + StringDirSeparator();

#if (THE_MACRO_MMVII_SYS == MMVII_SYS_L)
    mMMV1Bin           = mDirMicMacv1 + "bin/mm3d";
#elif (THE_MACRO_MMVII_SYS == MMVII_SYS_A)
    mMMV1Bin           = mDirMicMacv1 + "bin/mm3d";
#elif (THE_MACRO_MMVII_SYS == MMVII_SYS_W)
    mMMV1Bin           = mDirMicMacv1 + "bin/mm3d.exe";
#endif


}

const std::vector<eSharedPO>    cMMVII_Appli::EmptyVSPO;  ///< Deafaut Vector  shared optional parameter


/// This one is always std:: cout, to be used by StdOut and cMMVII_Appli::StdOut ONLY

cMultipleOfs & StdStdOut()
{
// Dont know why, destruction of static object at end fails on Mac
#if (THE_MACRO_MMVII_SYS == MMVII_SYS_A)
   static cMultipleOfs * aPtrMOfs = new cMultipleOfs(std::cout);
   return *aPtrMOfs;
#else
   static cMultipleOfs aMOfs(std::cout);
   return aMOfs;
#endif
}

cMultipleOfs& StdOut()
{
   if (cMMVII_Appli::ExistAppli())
     return cMMVII_Appli::CurrentAppli().StdOut();
   return StdStdOut();
}
cMultipleOfs& HelpOut() {return StdOut();}
cMultipleOfs& ErrOut()  {return StdOut();}



cMultipleOfs &  cMMVII_Appli::StdOut()
{
   /// Maybe mStdCout not correctly initialized if we are in constructor or in destructor ?
   if ((!cMMVII_Appli::ExistAppli()) || msInDstructor)
      return StdStdOut();
   return mStdCout;
}
cMultipleOfs &  cMMVII_Appli::HelpOut() {return StdOut();}
cMultipleOfs &  cMMVII_Appli::ErrOut() {return StdOut();}


void TestMainSet(const cCollecSpecArg2007 & aVSpec,bool &aMain0,bool & aMain1)
{
    for (int aK=0 ; aK<int(aVSpec.size()) ; aK++)
    {
        std::string aNumPat;
        if (aVSpec[aK]->HasType(eTA2007::MPatFile,&aNumPat))
        {
             int aNum =   cStrIO<int>::FromStr(aNumPat);
             if (aNum==0)  aMain0 = true;
             if (aNum==1)  aMain1 = true;
        }
    }
}

bool   cMMVII_Appli::HasSharedSPO(eSharedPO aV) const
{
   return BoolFind(mVSPO,aV);
}

void cMMVII_Appli::SetNot4Exe()
{
   mForExe = false;
}

void cMMVII_Appli::InitParam(cGenArgsSpecContext *aArgsSpecs)
{	
  mSeedRand = DefSeedRand();
  cCollecSpecArg2007 & anArgObl = ArgObl(mArgObl); // Call virtual method
  cCollecSpecArg2007 & anArgFac = ArgOpt(mArgFac); // Call virtual method


  mInitParamDone = true;
  // MMVII_INTERNAL_ASSERT_always(msTheAppli==0,"cMMVII_Appli only one by process");
  // msTheAppli = this;

  // Check that  cCollecSpecArg2007 were used with the good values
  MMVII_INTERNAL_ASSERT_always((&anArgObl)==&mArgObl,"cMMVII_Appli dont respect cCollecSpecArg2007");
  MMVII_INTERNAL_ASSERT_always((&anArgFac)==&mArgFac,"cMMVII_Appli dont respect cCollecSpecArg2007");

  std::string aDP; // mDirProject is handled specially so dont put mDirProject in AOpt2007
                   // becauser  InitParam, it may change the correct value 

  // Add common optional parameters
  cSpecOneArg2007::tAllSemPL aInternal{eTA2007::Internal,eTA2007::Global}; // just to make shorter lines
  cSpecOneArg2007::tAllSemPL aGlob{eTA2007::Global}; // just to make shorter lines
  cSpecOneArg2007::tAllSemPL aGlobHDV{eTA2007::Global,eTA2007::HDV}; // just to make shorter lines


  /*  Decoding AOpt2007(mIntervFilterMS[0],GOP_Int0,"File Filter Interval, Main Set"  ,{eTA2007::Common,{eTA2007::FFI,"0"}})
        mIntervFilterMS[0]  => string member, will store the value
        GOP_Int0 => const name, Global Optionnal Interval , num 0, declared in MMVII_DeclareCste.h
        {eTA2007::Common,{eTA2007::FFI,"0"}}  attibute, it's common, it's intervall with attribute "0"
  */

  if (HasSharedSPO(eSharedPO::eSPO_CarPO))
  {
     mArgFac << AOpt2007(mCarPPrefOut,"CarPOut","Name for Output caracteristic points",{eTA2007::HDV});
  }
  if (HasSharedSPO(eSharedPO::eSPO_CarPI))
  {
     mArgFac << AOpt2007(mCarPPrefIn,"CarPIn","Name for Input caracteristic points",{eTA2007::HDV});
  }





  // To not put intervals in help/parameters when they are not usefull
  {
      bool HasMain0 = false;
      bool HasMain1 = false;
      TestMainSet(anArgObl,HasMain0,HasMain1);
      TestMainSet(anArgFac,HasMain0,HasMain1);
      if (HasMain0)
        mArgFac <<  AOpt2007(mIntervFilterMS[0],GOP_Int0,"File Filter Interval, Main Set"  ,{eTA2007::Shared,{eTA2007::FFI,"0"}});
      if (HasMain1)
        mArgFac <<  AOpt2007(mIntervFilterMS[1],GOP_Int1,"File Filter Interval, Second Set",{eTA2007::Shared,{eTA2007::FFI,"1"}});
  }
  mArgFac
      // <<  AOpt2007(mIntervFilterMS[0],GOP_Int0,"File Filter Interval, Main Set"  ,{eTA2007::Common,{eTA2007::FFI,"0"}})
      // <<  AOpt2007(mIntervFilterMS[1],GOP_Int1,"File Filter Interval, Second Set",{eTA2007::Common,{eTA2007::FFI,"1"}})
      <<  AOpt2007(mNumOutPut,GOP_NumVO,"Num version for output format (1 or 2)",{eTA2007::Global,{eTA2007::Range,"[1,2]"}})
      <<  AOpt2007(mSeedRand,GOP_SeedRand,"Seed for random,if <=0 init from time",aGlobHDV)
      <<  AOpt2007(msWithWarning,GOP_WW,"Do we print warnings",aGlobHDV)
      <<  AOpt2007(mNbProcAllowed,GOP_NbProc,"Number of process allowed in parallelisation",aGlobHDV)
      <<  AOpt2007(aDP ,GOP_DirProj,"Project Directory",{eTA2007::DirProject,eTA2007::Global})
      <<  AOpt2007(mParamStdOut,GOP_StdOut,"Redirection of Ouput (+File for add,"+ MMVII_NONE + "for no out)",aGlob)
      <<  AOpt2007(mLevelCall,GIP_LevCall," Level Of Call",aInternal)
      <<  AOpt2007(mKthCall,GIP_KthCall," Ordre Of Call when multiple call",aInternal)
      <<  AOpt2007(mPatternInitGMA,GIP_PatternGMA,"Initial pattern of global main appli ",aInternal)

      <<  AOpt2007(mShowAll,GIP_ShowAll,"",aInternal)
      <<  AOpt2007(mPrefixGMA,GIP_PGMA," Prefix Global Main Appli",aInternal)
      <<  AOpt2007(mPrefix_TIM_GMA,GIP_TIM_GMA," Prefix for Time of Global Main Appli",aInternal)
      <<  AOpt2007(mDirProjGMA,GIP_DirProjGMA," Folder Project Global Main Appli",aInternal)
      <<  AOpt2007(mExecFrom,GIP_ExecFrom," Name of the frontend that launched this command",aInternal)
  ;

  // Check that names of optionnal parameters begin with alphabetic caracters
  for (const auto & aSpec : mArgFac.Vec())
  {
      aSpec->ReInit();
      if (!std::isalpha(aSpec->Name()[0]))
      {
         MMVII_INTERNAL_ASSERT_always
         (
             false,
             "Name of optional param must begin with alphabetic => ["+aSpec->Name()+"]"
         );
      }
  }

  // Test if we are in help mode
  for (int aKArg=0 ; aKArg<mArgc ; aKArg++)
  {
      const char * aArgK = mArgv[aKArg].c_str();
      if (UCaseBegin("help",aArgK) || UCaseBegin("-help",aArgK)|| UCaseBegin("--help",aArgK))
      {
         mModeHelp = true;
         while (*aArgK=='-') aArgK++;
         mDoGlobHelp = (*aArgK=='H');
         mDoInternalHelp = CaseSBegin("HE",aArgK);

         std::string aName; 
         SplitStringArround(aName,mPatHelp,aArgK,'=',true,false);
      }
  }
  if (mModeHelp)
  {
      GenerateHelp();
      return;
  }

  if (aArgsSpecs) {
      mModeArgsSpec = true;
      GenerateArgsSpec(aArgsSpecs);
      return;
  }

  int aNbObl = mArgObl.size(); //  Number of mandatory argument expected
  int aNbArgGot = 0; // Number of  Arg received untill now
  bool Optional=false; // Are we in the optional phase of argument parsing

  // To be abble to process in  the same loop mandatory and optional
  std::vector<std::string> aVValues;
  tVecArg2007              aVSpec;

  //  Memorize this value was used in spec 
  for (const auto  & aSpec : mArgFac.Vec())
       mSetVarsSpecFac.Add(aSpec->AdrParam()); 
  //  Memorize this value was used in spec 
  for (const auto  & aSpec : mArgObl.Vec())
       mSetVarsSpecObl.Add(aSpec->AdrParam()); 


  for (int aKArg=0 ; aKArg<mArgc ; aKArg++)
  {
      Optional = (aNbArgGot>=aNbObl);
      // If --Name replace by Name, maybe usefull for completion
      if (Optional && (mArgv[aKArg][0]=='-') && (mArgv[aKArg][1]=='-'))
         mArgv[aKArg] = mArgv[aKArg].substr(2);

      const char * aArgK = mArgv[aKArg].c_str();
      if (aKArg<2)
      {
          //  mArgv[0] => MMVII
          //  mArgv[1] => the name of commmand
      }
      else
      {
          if (Optional)
          {
             // while '
             std::string aName,aValue;
             SplitStringArround(aName,aValue,aArgK,'=',true,false);
             int aNbSpecGot=0;
             // Look for spec corresponding to name
             for (const auto  & aSpec : mArgFac.Vec())
             {
                 // if (aSpec->Name() == aName)
                 if (UCaseEqual(aSpec->Name(),aName))
                 {
                    aNbSpecGot++;
                    aVSpec.push_back(aSpec);
                    // Several space have the same name
                    if (aNbSpecGot==2)
                    {
                        MMVII_INTERNAL_ASSERT_always(false,"\""+ aName +"\" is multiple in specification");
                    }
                    // Same name was used several time
                    MMVII_INTERNAL_ASSERT_User
                    (  aSpec->NbMatch()==0  ,eTyUEr::eMulOptParam,"\""+aName +"\" was used multiple time");
                    aSpec->IncrNbMatch();
                 }
             }
             // Name does not correspond to spec
             MMVII_INTERNAL_ASSERT_User
             (  aNbSpecGot!=0  ,eTyUEr::eBadOptParam,"\""+aName +"\" is not a valide optionnal value");

             aVValues.push_back(aValue);
          }
          else
          {
             aVValues.push_back(aArgK);
             aVSpec.push_back(mArgObl[aNbArgGot]);
          }
          aNbArgGot ++;
      }
  }

  size_t aNbArgTot = aVValues.size();

  if (aNbArgGot < aNbObl)
  {
      // Tolerance, like in mmv1, no arg generate helps
      if (aNbArgGot==0)
      {
         mModeHelp = true;  // else Exe() will be executed !!
         GenerateHelp();
         return;
      }
      MMVII_UsersErrror
      (
          eTyUEr::eInsufNbParam,
          "Not enough Arg, expecting " + ToS(aNbObl)  + " , Got only " +  ToS(aNbArgGot)
      );
  }
  MMVII_INTERNAL_ASSERT_always(aNbArgTot==aVSpec.size(),"Interncl check size Value/Spec");


  // First compute the directory of project that may influence all other computation
     // Try with Optional value
  {
     bool HasDirProj=false;
     for (size_t aK=0 ; aK<aNbArgTot; aK++)
     {
        if (aVSpec[aK]->HasType(eTA2007::DirProject))
        {
           MMVII_INTERNAL_ASSERT_always(!HasDirProj,"Multiple dir project");
           HasDirProj = true;
           MakeNameDir(aVValues[aK]);
           mDirProject = aVValues[aK];
        }
     }

  
     {
         bool HasFileDirProj = false;
         for (size_t aK=0 ; aK<aNbArgTot; aK++)
         {
            if (aVSpec[aK]->HasType(eTA2007::FileDirProj))
            {
               MMVII_INTERNAL_ASSERT_always(!HasFileDirProj,"Multiple file dir project");
               if (!HasDirProj)
                  mDirProject = DirOfPath(aVValues[aK],false);
               else
               {
                  // More or less a limit case, dont know really what must be accepted
                  aVValues[aK] = mDirProject + aVValues[aK];
                  mDirProject = DirOfPath(aVValues[aK],false);
               }
               HasFileDirProj = true;
            }
         }
     }
  }
  // Add a "/" at end  if necessary
  MakeNameDir(mDirProject);

  //  Initialize the paramaters
  for (size_t aK=0 ; aK<aNbArgTot; aK++)
  {
       aVSpec[aK]->InitParam(aVValues[aK]);
       mSetInit.Add(aVSpec[aK]->AdrParam()); ///< Memorize this value was initialized
  }
  mNbProcAllowed = std::min(mNbProcAllowed,mNbProcSystem); ///< avoid greedy/gluton user
  mMainProcess   = (mLevelCall==0);
  mGlobalMainAppli = mMainProcess && mMainAppliInsideP;
  // Compute an Id , unique and (more or less ;-) understandable
  // tINT4 aIT0 = round_ni(mT0);
  mPrefixNameAppli =   std::string("MMVII")
                     + std::string("_Tim")  + StrIdTime()
                     + std::string("_Num")  + ToStr(mNumCallInsideP) 
                     + std::string("_Pid")  + ToStr(mPid) 
                     + std::string("_") + mSpecs.Name()
                   ;
   if (mGlobalMainAppli)  // Pour communique aux sous process
   {
       mPrefixGMA  = mPrefixNameAppli;
       mPrefix_TIM_GMA = StrIdTime();
       mDirProjGMA = mDirProject;
   }

  // Manange OutPut redirection
  if (IsInit(&mParamStdOut))
  {
     const char * aPSO = mParamStdOut.c_str();
     bool aModeFileInMore = false;
     bool aModeAppend = true;

     // Do it twice to accept 0+ and +0
     for (int aK=0 ; aK<2 ; aK++)
     {
         //  StdOut=0File.txt => Put in file, erase it before
         if (aPSO[0]=='0')
         {
            aModeAppend = false;
            aPSO++;  
         }
         //  StdOut=+File.txt => redirect output in file and in console
         //  StdOut=0+File.txt => work also
         if (aPSO[0]=='+')
         {
            aModeFileInMore = true;
            aPSO++;  
         }
     }
     // If not on console, supress std:: cout which was in mStdCout
     if (! aModeFileInMore)
     {
         mStdCout.Clear();
     }
     // Keyword NONE means no out at all
     if (MMVII_NONE != aPSO)
     {
         mFileStdOut.reset(new cMMVII_Ofs(aPSO,aModeAppend ? eFileModeOut::AppendText : eFileModeOut::CreateText));
         // separator between each process , to refine ... (date ? Id ?)
         mFileStdOut->Ofs() << "=============================================" << std::endl;
         mStdCout.Add(mFileStdOut->Ofs());
     }
  }

  // If mNumOutPut was set, fix the output version
  if (IsInit(&mNumOutPut))
  {
     if (mNumOutPut==1)
        mOutPutV1 = true;
     else if (mNumOutPut==2)
        mOutPutV2 = true;
     else
     {
         MMVII_INTERNAL_ASSERT_always(false,"Output version must be in {1,2}, got: "+ToStr(mNumOutPut));
     }
  }


  // Test the size of vectors vs possible specifications in
  for (size_t aK=0 ; aK<aNbArgTot; aK++)
  {
      std::string aSpecSize;
      // If the arg contains the semantic
      if (aVSpec[aK]->HasType(eTA2007::ISizeV,&aSpecSize))
      {
         aVSpec[aK]->CheckSize(aSpecSize);  // Then test it
      }


      std::string aNameTag;
      if (aVSpec[aK]->HasType(eTA2007::XmlOfTopTag,&aNameTag))
      {
         if (!IsFileGivenTag(true,aVValues[aK],aNameTag))
	 {
	       MMVII_UsersErrror(eTyUEr::eBadXmlTopTag,"[" + aVValues[aK] + "] is not an existing xml file of main tag <" + aNameTag + ">");
			      // IntervalOk=" + anArg + " Got=" + ToStr(int(aVal.size())));
	 }
	 //        MMVII_UsersErrror(eTyUEr::eBadSize4Vect,"IntervalOk=" + anArg + " Got=" + ToStr(int(aVal.size())));
         // aVSpec[aK]->CheckSize(aSpecSize);  // Then test it
      }
      // XmlOfTag,
  }

  // Analyse the possible main patterns
  for (size_t aK=0 ; aK<aNbArgTot; aK++)
  {
      std::string aNumPat;
      // Test the semantic
      if (aVSpec[aK]->HasType(eTA2007::MPatFile,&aNumPat))
      {
         int aNum =   cStrIO<int>::FromStr(aNumPat);
         // Check range
         CheckRangeMainSet(aNum);

         // don't accept multiple initialisation
         if (!mVMainSets.at(aNum).IsInit())
         {
            // mVMainSets.at(aNum)= SetNameFromString(mDirProject+aVValues[aK],true);
            mVMainSets.at(aNum)= SetNameFromString(mDirProject+FileOfPath(aVValues[aK],false),true);

            //  Filter with interval
            {
               std::string & aNameInterval = mIntervFilterMS[aNum];
               if (IsInit(&aNameInterval))
               {
                   mVMainSets.at(aNum).Filter(Str2Interv<std::string>(aNameInterval));
               }
            }
            // Test non empty
            if (! AcceptEmptySet(aNum) && (mVMainSets.at(aNum).size()==0))
            {
                // if we are in a recall mode, posibly pattern comes file xml, and file insid can be un-existent
                if (mLevelCall>0)
		{
                   mVMainSets.at(aNum).Add(aVValues[aK]);
		}
		else
                {
                   MMVII_UsersErrror(eTyUEr::eEmptyPattern,"Specified set of files was empty");
                }
            }
         }
         else
         {
            MMVII_INTERNAL_ASSERT_always(false,"Multiple main set im for num:"+ToStr(aNum));
         }
/*
         std::string & aNameInterval = mIntervFilterMS[aNum];
         if (IsInit(&aNameInterval))
         {
             mVMainSets.at(aNum).Filter(Str2Interv<std::string>(aNameInterval));
         }
*/
      }
  }
  // Check validity of main set initialization
  for (int aNum=0 ; aNum<NbMaxMainSets ; aNum++)
  {
      // Why should user init interval if there no set ?
      if (IsInit(&mIntervFilterMS[aNum]) && (!  mVMainSets.at(aNum).IsInit()))
      {
         MMVII_UsersErrror(eTyUEr::eIntervWithoutSet,"Interval without filter for num:"+ToStr(aNum));
      }
      if (aNum>0)
      {
         // would be strange to have Mainset2 without MainSet1; probably if this occurs
         // the fault would be from programer's side (not sure)
         if ((! mVMainSets.at(aNum-1).IsInit() ) && ( mVMainSets.at(aNum).IsInit()))
         {
            MMVII_INTERNAL_ASSERT_always(false,"Main set, init for :"+ToStr(aNum) + " and non init for " + ToStr(aNum-1));
         }
      }
  }


  // MakeNameDir(mDirProject);
  
  // Print the info, debugging
  if (mShowAll)
  {
     // Print the value of all parameter
     for (size_t aK=0 ; aK<aNbArgTot; aK++)
     {
         HelpOut() << aVSpec[aK]->Name()  << " => [" << aVValues[aK] << "]" << std::endl;
     }
     HelpOut() << "---------------------------------------" << std::endl;
     HelpOut() << "IS INIT  DP: " << IsInit(&aDP) << std::endl;
     HelpOut() << "DIRPROJ=[" << mDirProject << "]" << std::endl;
  }

    // By default, if calls is done at top level, assure that everything is init
  if (mSeedRand<=0)
  {
     mSeedRand =  std::chrono::system_clock::to_time_t(mT0);
  }

  // Don't fully initialize project if this appli is a special MMVII management applu
  const auto aFeatures = mSpecs.Features();
  if (std::find(aFeatures.cbegin(), aFeatures.cend(), eApF::ManMMVII) != aFeatures.cend()) {
     mForExe = false;   // Don't do special cleaning in cMMVII_Appli destructor
     return;
  }

  mFileLogTop = mDirProject + MMVII_LogFile;

  if (mGlobalMainAppli)
  {
     InitProject();
  }
  if (!mModeHelp)
     LogCommandIn(NameFileLog(false),false);

  if (mMainAppliInsideP) 
     InitProfile();
}

void cMMVII_Appli::InitProfile()
{
  
  // ========================================================================
  // ========================  HANDLING PROFILE USER ETC ... ================
  // ========================================================================


  //  part of code that was used to initialize "at hand", soon will be obsolete...
  if (0)
  {
      StdOut() << "NO USEERRRRRRRRRRRRRR " << std::endl; getchar();

      mParamProfile.mUserName = "Unkown";
      mParamProfile.mNbProcMax = 1000;
      mParamProfile.mVectDefSerial = eTypeSerial::ecsv;
      mParamProfile.mTaggedDefSerial = eTypeSerial::ejson;
      mVectNameDefSerial = E2Str(mParamProfile.mVectDefSerial);
      mTaggedNameDefSerial = E2Str(mParamProfile.mTaggedDefSerial);
      return;
  } 

  /*  Compute the name of file containing the profile of user;  this profile is 
   *
   *     - "MMVII-CurentPofile.xml" if this file exists, to allow tuning by user
   *     - "Default-MMVII-CurentPofile.xml" if it does not exist, this is the file shared on github
   */
  std::string NameFileCurentProfile =  "MMVII-CurentPofile.xml";
  std::string DefaultNameFileCurentProfile =  "Default-" + NameFileCurentProfile;
  std::string NameFileUseOfProfile =  "MMVII-UserOfProfile.xml";

  // if the default file  does not exist, we are probably the first time, or in reinit step because
  // directory has been purged, we create a file containing  the default profile
  if (! ExistFile(mDirLocalParameters+DefaultNameFileCurentProfile))
  {
        cSpecifProfileUserMMVII  aSpec;
        aSpec.mNameProfile = "Default";
        SaveInFile(aSpec,mDirLocalParameters+DefaultNameFileCurentProfile);
  }

  // we set NameFileCurentProfile  to its default or not,
  if (! ExistFile(mDirLocalParameters+NameFileCurentProfile))
  {
      NameFileCurentProfile = DefaultNameFileCurentProfile;
  }

  /**  Compute the "usage" store in the profile, 
   *   init the variable  "mProfileUsage"  and  "mDirProfileUsage"
   */
  {
      cSpecifProfileUserMMVII aSpec;

      ReadFromFile(aSpec,mDirLocalParameters+NameFileCurentProfile);
      mProfileUsage = aSpec.mNameProfile;
      mDirProfileUsage =  mDirLocalParameters + mProfileUsage + StringDirSeparator();
  }

  /**  if file containing users profile does not exist, we create some default one */
  if (! ExistFile(mDirProfileUsage+NameFileUseOfProfile))
  {
      CreateDirectories(mDirProfileUsage,false);

      mParamProfile.mUserName = "Unkown";
      mParamProfile.mNbProcMax = 1000;
      mParamProfile.mVectDefSerial = eTypeSerial::ecsv;
      mParamProfile.mTaggedDefSerial = eTypeSerial::exml;
      SaveInFile(mParamProfile,mDirProfileUsage+NameFileUseOfProfile);
  }
  ReadFromFile(mParamProfile,mDirProfileUsage+NameFileUseOfProfile);
  mVectNameDefSerial = E2Str(mParamProfile.mVectDefSerial);
  mTaggedNameDefSerial = E2Str(mParamProfile.mTaggedDefSerial);

}

const  std::string & cMMVII_Appli::UserName() {return mParamProfile.mUserName;}
const  std::string & cMMVII_Appli::DirProfileUsage() {return mDirProfileUsage;}

eTypeSerial cMMVII_Appli::VectDefSerial() const 
{
    CurrentAppli(); // as member is static assure init was done
    return mParamProfile.mVectDefSerial;
}
eTypeSerial cMMVII_Appli::TaggedDefSerial() const 
{
    CurrentAppli(); // as member is static assure init was done
    return mParamProfile.mTaggedDefSerial;
}

const std::string & cMMVII_Appli::VectNameDefSerial   () const 
{
    CurrentAppli(); // as member is static assure init was done
    return mVectNameDefSerial;
}
const std::string & cMMVII_Appli::TaggedNameDefSerial   () const 
{
    CurrentAppli(); // as member is static assure init was done
    return mTaggedNameDefSerial;
}

const std::string & GlobVectNameDefSerial() {return cMMVII_Appli::CurrentAppli().VectNameDefSerial();}
const std::string & GlobTaggedNameDefSerial() {return cMMVII_Appli::CurrentAppli().TaggedNameDefSerial();}



// const  std::string & cMMVII_Appli::UserName() {return mParamProfile.mUserName;}


tPtrArg2007 cMMVII_Appli::AOptBench()
{
     return   AOpt2007(mIsInBenchMode,GIP_BenchMode,"Is the command executed in bench mode",{eTA2007::Internal,eTA2007::HDV});
}



std::string cMMVII_Appli::NameFileLog(bool Finished) const
{
   return   
              //mDirProject
               mDirProjGMA
             + TmpMMVIIDirGlob
             + TmpMMVIIProcSubDir 
             + mPrefixGMA + StringDirSeparator()
             + mPrefixNameAppli 
             + std::string(Finished ? "_Ok" : "_InProcess")
             + std::string(".txt")
          ;
}

std::string cMMVII_Appli::PrefixPCar(const std::string & aNameIm,const std::string & aPref) const
{
   return mDirProject +TmpMMVIIDirPCar + aNameIm + "/" + aPref;
}
std::string cMMVII_Appli::PrefixPCarOut(const std::string & aNameIm) const
{
   return PrefixPCar(aNameIm,mTiePPrefOut);
}
std::string cMMVII_Appli::PrefixPCarIn(const std::string & aNameIm) const
{
   return PrefixPCar(aNameIm,mTiePPrefIn);
}

std::string cMMVII_Appli::NamePCarImage(const std::string & aNameIm,eTyPyrTieP aType,const std::string & aSpecific,const cPt2di & aTile) const
{
   return NamePCarGen(aNameIm,eModeOutPCar::eMOPC_Image,aType,false,aSpecific,aTile);
}

std::string  cMMVII_Appli::NamePCar
             (const std::string & aNameIm,eModeOutPCar aMode,eTyPyrTieP aType,bool Input,bool IsMax,const cPt2di & aTile) const
{
   return NamePCarGen(aNameIm,aMode,aType,Input,(IsMax ? "Max" : "Min"),aTile);
}

std::string  cMMVII_Appli::StdNamePCarIn(const std::string & aNameIm,eTyPyrTieP aType,bool IsMax) const
{
    return NamePCar(aNameIm,eModeOutPCar::eMNO_BinPCarV2,aType,true,IsMax,cPt2di(-1,-1));
}



std::string  cMMVII_Appli::NamePCarGen
             (
                  const std::string & aNameIm,
                  eModeOutPCar        aMode,
                  eTyPyrTieP          aType,
                  bool InPut,
                  const std::string & aSpecific,
                  const cPt2di & aTile
             ) const
{
    std::string aPref = (InPut ? PrefixPCarIn(aNameIm) : PrefixPCarOut(aNameIm));
    
    std::string  aStrMode;
    std::string  aPost;

    if (aMode==eModeOutPCar::eMOPC_Image)
    {
        aStrMode = "Ima";
        aPost    = "tif";
    }
    else if (aMode==eModeOutPCar::eMNO_PCarV1)
    {
        aStrMode = "V1AimePCar";
        aPost    = "dmp";
    }
    else if (aMode==eModeOutPCar::eMNO_BinPCarV2)
    {
        aStrMode = "V2AimePCar";
        aPost    = "dmp";
    }
    else if (aMode==eModeOutPCar::eMNO_XmlPCarV2)
    {
        aStrMode = "V2AimePCar";
        aPost    = "xml";
    }
    
    std::string aMinus("-");
    
    std::string  aStrTile;
    if (aTile.x()>=0)
    {
         aStrTile =  "-Tile" + ToStr(aTile.x())+ "_"+ ToStr(aTile.y());
    }

    std::string aRes =    aPref 
                        + aMinus + aStrMode 
                        + aMinus + E2Str(aType)
                        + aMinus + aSpecific  
                        + aStrTile
                        + "." +  aPost;

    return aRes;
}


// Is called only when global main applu
void cMMVII_Appli::InitProject()
{
   // Create Dir for tmp file, process etc ...
   std::string aDir = mDirProject+TmpMMVIIDirGlob;
   CreateDirectories(aDir,true);

   // Create Dir for Caracteristic point, if Appli output them
   if (HasSharedSPO(eSharedPO::eSPO_CarPO))
   {
      CreateDirectories(mDirProject +TmpMMVIIDirPCar,true);
   }

   aDir += TmpMMVIIProcSubDir;
   CreateDirectories(aDir,true);

   if (! mModeHelp)
   {
      aDir += mPrefixNameAppli;
      CreateDirectories(aDir,true);
      LogCommandIn(mFileLogTop,true);
   }
}

void cMMVII_Appli::LogCommandIn(const std::string & aName,bool MainLogFile)
{
    if (aName == "")
      return;
   cMMVII_Ofs  aOfs(aName,eFileModeOut::AppendText);
   aOfs.Ofs() << "========================================================================\n";
   aOfs.Ofs() << "  Id : " <<  mPrefixNameAppli << "\n";
   aOfs.Ofs() << "  begining at : " <<  StrDateCur() << "\n\n";
   aOfs.Ofs() << "  " << CommandOfMain().Com() << "\n\n";
   aOfs.Ofs().close();
}

void cMMVII_Appli::LogCommandOut(const std::string & aName,bool MainLogFile)
{
   if (aName == "")
      return;
   cMMVII_Ofs  aOfs(aName,eFileModeOut::AppendText);
   // Add id, if several process were throw in // there is a mix and we no longer know which was closed
   aOfs.Ofs() << "  ending correctly at : " <<  StrDateCur()  << " (Id=" << mPrefixNameAppli << ")\n\n";
   aOfs.Ofs().close();
}


void cMMVII_Appli::LogCommandAbortOnError(std::string& aMessage)
{
   if (mFileLogTop == "")
      return;
   for (const auto& aLogName : {mFileLogTop, NameFileLog(false)})
   {
      cMMVII_Ofs  aOfs(aLogName,eFileModeOut::AppendText);
      aOfs.Ofs() << "  ABORT on error at : " <<  StrDateCur()  << " (Id=" << mPrefixNameAppli << ")\n";
      bool nl = true;
      for (const auto& c: aMessage)
      {
         if (nl)
            aOfs.Ofs() << "  > ";
         aOfs.Ofs() << c;
         nl = c == '\n';
      }
      aOfs.Ofs() << std::endl;
      aOfs.Ofs().close();
   }
}



static std::string JsonEscaped(const std::string& s)
{
    std::string res;

    for (const auto& c : s) {
        switch (c) {
        case '\b' : res += "\\b"; break;
        case '\f' : res += "\\f"; break;
        case '\n' : res += "\\n"; break;
        case '\r' : res += "\\r"; break;
        case '\t' : res += "\\t"; break;
        case '"'  : res += "\\\""; break;
        case '\\' : res += "\\\\"; break;
        default   :
            if (c < ' ') {
                std::stringstream ss;
                ss << std::hex << std::setw(4) << std::setfill('0') << static_cast<unsigned>(c);
                res += "\\u" + ss.str();
            } else {
                res += c;
            }
        }
    }
    return res;
}

void cMMVII_Appli::GenerateOneArgSpec(cCollecSpecArg2007& aSpecArgs, const std::string& aSpecName, bool aOptional, cGenArgsSpecContext *aArgsSpec)
{
    if (aOptional)
        aArgsSpec->jsonSpec += "      \"optional\": [";
    else
        aArgsSpec->jsonSpec += "      \"mandatory\": [";

    int num = 1;
    for (const auto & Arg : aSpecArgs.Vec())
    {
        if (Arg->HasType(eTA2007::Internal))
            continue;

        if (num != 1)
            aArgsSpec->jsonSpec +=  ",";

        // semantic checks
        std::string argName;
        if (aOptional)
            argName = Arg->Name();
        else
            argName = "obl #" + std::to_string(num);
        num++;

        std::string fileType,dirType;
        bool hasMPF = false;
        for (const auto& a : Arg->SemPL()) {
            if (aArgsSpec->fileTypes.find(a.Type()) != aArgsSpec->fileTypes.end()) {
                if (fileType.length() != 0)
                   aArgsSpec->errors += "WARNING: " + aSpecName + ": " + argName + ": has " + fileType + " and " + E2Str(a.Type()) + " file semantic.\n";
                fileType = E2Str(a.Type());
            }
            if (std::find(aArgsSpec->prjSubDirList.begin(), aArgsSpec->prjSubDirList.end(),a.Type()) != aArgsSpec->prjSubDirList.end()) {
                if (dirType.length() != 0)
                    aArgsSpec->errors += "WARNING: " + aSpecName + ": " + argName + ": has " + dirType + " and " + E2Str(a.Type()) + " dir semantic.\n";
                dirType = E2Str(a.Type());
            }
            if (a.Type() == eTA2007::MPatFile)
            hasMPF = true;
        }
        if (fileType.length() != 0 && dirType.length() != 0)
            aArgsSpec->errors += "WARNING: " + aSpecName + ": " + argName + ": has " + dirType + " and " + fileType + " semantics.\n";

        if (Arg->IsVector()  && !Arg->HasType(eTA2007::ISizeV))
            aArgsSpec->errors += "WARNING: " + aSpecName + ": " + argName + ": is a vector with no ISizeV semantic.\n";
        if (Arg->HasType(eTA2007::FileDirProj) && fileType.length() == 0 && !hasMPF)
            aArgsSpec->errors += "WARNING: " + aSpecName + ": " + argName + ": has FileDirProj semantic with no File type semantic.\n";
#if 0
// if no Input/Output/OptionalExist, assume Input
        bool hasFileInOut = Arg->HasType(eTA2007::Input) || Arg->HasType(eTA2007::Output) || Arg->HasType(eTA2007::OptionalExist);
        if ( ! hasFileInOut && fileType.length() != 0)
            aArgsSpec->errors += "WARNING: " + aSpecName + ": " + argName + ": type " + fileType + ": Missing [Input|Output|OptionalExist] semantic.\n";
        if ( ! hasFileInOut && dirType.length() != 0)
            aArgsSpec->errors += "WARNING: " + aSpecName + ": " + argName + ": type " + dirType + ": Missing [Input|Output|OptionalExist] semantic.\n";
#endif
        // end of checks

        std::vector<std::string> semantic;
        std::vector<std::string> allowed;
        std::string range;
        std::string vectorSize;
        for (const auto& a : Arg->SemPL()) {

            if (a.Type() < eTA2007::AddCom) {
                semantic.push_back(E2Str(a.Type()));
            }
            if (a.Type() == eTA2007::AllowedValues) {
                allowed = SplitString(a.Aux(),",");
            }
            if (a.Type() == eTA2007::Range) {
                range = a.Aux();
            }
            if (a.Type() == eTA2007::ISizeV) {
                vectorSize = a.Aux();
            }
        }

        aArgsSpec->jsonSpec +=  "\n        {\n";
        if (aOptional) {
            std::string level = Arg->HasType(eTA2007::Global) ? "global" : Arg->HasType(eTA2007::Tuning) ? "tuning" : "normal";
            aArgsSpec->jsonSpec +=  "            \"name\": \"" + JsonEscaped(Arg->Name()) + "\",\n";
            aArgsSpec->jsonSpec +=  "            \"level\": \"" + level + "\",\n";
        }
        aArgsSpec->jsonSpec +=  "            \"type\": \"" + JsonEscaped(Arg->NameType()) + "\"";
        if (semantic.size()) {
            aArgsSpec->jsonSpec +=  ",\n            \"semantic\": [";
            for (unsigned i=0; i<semantic.size(); i++) {
                if (i > 0)
                    aArgsSpec->jsonSpec +=  ",";
                aArgsSpec->jsonSpec +=  "\"" + semantic[i] + "\"";
            }
            aArgsSpec->jsonSpec +=  "]";
        }
        if (allowed.size()) {
            aArgsSpec->jsonSpec +=  ",\n            \"allowed\" : [";
            for (unsigned i=0; i<allowed.size(); i++) {
                if (i > 0)
                    aArgsSpec->jsonSpec +=  ",";
                aArgsSpec->jsonSpec +=  "\"" + allowed[i] + "\"";
            }
            aArgsSpec->jsonSpec +=  "]";
        }
        if (range.size())
            aArgsSpec->jsonSpec +=  ",\n            \"range\" : \"" + range + "\"";
        if (vectorSize.size())
            aArgsSpec->jsonSpec +=  ",\n            \"vsize\" : \"" + vectorSize + "\"";
        if (Arg->HasType((eTA2007::HDV)))
            aArgsSpec->jsonSpec +=  ",\n            \"default\": \"" + JsonEscaped(Arg->NameValue()) + "\"";
        if (Arg->Com().size())
            aArgsSpec->jsonSpec +=  ",\n            \"comment\": \"" + JsonEscaped(Arg->Com()) + "\"";
        aArgsSpec->jsonSpec +=  "\n        }";
    }
    aArgsSpec->jsonSpec +=  "\n      ]";
}

template <typename VE>
static std::string enumsVectorToStr(const VE& aVe)
{
    std::string s="";

    for (const auto& e : aVe) {
        if (s != "") s += ",";
        s += "\"" + E2Str(e) + "\"";
    }
    return s;
}

void cMMVII_Appli::GenerateArgsSpec(cGenArgsSpecContext *aArgsSpec)
{
   aArgsSpec->jsonSpec += "    {\n";
   aArgsSpec->jsonSpec += "      \"name\": \"" + JsonEscaped(mSpecs.Name()) + "\",\n";

   aArgsSpec->jsonSpec += "      \"comment\": \"" + JsonEscaped(mSpecs.Comment()) + "\",\n";
   aArgsSpec->jsonSpec += "      \"source\": \"" + JsonEscaped(mSpecs.NameFile()) + "\",\n";
   aArgsSpec->jsonSpec += "      \"features\": [" + enumsVectorToStr(mSpecs.Features()) + "],\n";
   aArgsSpec->jsonSpec += "      \"inputs\": [" + enumsVectorToStr(mSpecs.VInputs()) + "],\n";
   aArgsSpec->jsonSpec += "      \"outputs\": [" + enumsVectorToStr(mSpecs.VOutputs()) + "],\n";

   std::string aErr;
   GenerateOneArgSpec(mArgObl,mSpecs.Name(),false, aArgsSpec);
   aArgsSpec->jsonSpec += ",\n";
   GenerateOneArgSpec(mArgFac,mSpecs.Name(),true, aArgsSpec);
   aArgsSpec->jsonSpec += "\n    }";
   if (aErr.size())
       aArgsSpec->errors += aErr + "\n";
}

    // ========== Help ============

void cMMVII_Appli::PrintAdditionnalComments(tPtrArg2007 anArg)
{
   if (mDoGlobHelp)
   {
      for (const auto  & aCom : anArg->AddComs())
      {
          HelpOut() << "    - " << aCom << "\n";
      }
   }
}

void cMMVII_Appli::GenerateHelp()
{
   HelpOut() << "\n";

   HelpOut() << "**********************************\n";
   HelpOut() << "*   Help project 2007/MMVII      *\n";
   HelpOut() << "**********************************\n";

   HelpOut() << "\n";
   HelpOut() << "  For command : " << mSpecs.Name() << " \n";
   HelpOut() << "   => " << mSpecs.Comment() << "\n";
   HelpOut() << "   => Srce code entry in :" << mSpecs.NameFile() << "\n";
   HelpOut() << "\n";

   HelpOut() << " == Mandatory unnamed args : ==\n";

   for (const auto & Arg : mArgObl.Vec())
   {
       HelpOut() << "  * " << Arg->NameType() << Arg->Name4Help() << " :: " << Arg->Com()  << "\n";
       PrintAdditionnalComments(Arg);
   }

   tNameSelector  aSelName =  AllocRegex(mPatHelp);

   HelpOut() << "\n";
   HelpOut() << " == Optional named args : ==\n";
   //  Help to write only once the #### XXXX ###
   bool InternalMet = false;
   bool GlobalMet   = false;
   bool TuningMet   = false;
   for (int aKTime=0 ; aKTime<2; aKTime++)
   {
      for (const auto & Arg : mArgFac.Vec())
      {
          const std::string & aNameA = Arg->Name();
          if (aSelName.Match(aNameA))
          {
             bool IsIinternal = Arg->HasType(eTA2007::Internal);
             bool IsTuning = Arg->HasType(eTA2007::Tuning);
             if ((! (IsIinternal || IsTuning)) || mDoInternalHelp)
             {
                bool IsGlobHelp = Arg->HasType(eTA2007::Global);
                // First time do std args, second time to others (tune,glob,inter ...)
                bool DoIt = (aKTime==0) ^  (IsIinternal || IsTuning || IsGlobHelp);
                if (DoIt && ((!IsGlobHelp) || mDoGlobHelp))
                {
                   if (IsTuning && (!TuningMet)) 
                   {
                      HelpOut() << "       ####### TUNING #######\n" ; 
                      TuningMet = true;
                   }
                   else if (IsIinternal && (!InternalMet)) 
                   {
                      HelpOut() << "       ####### INTERNAL #######\n" ; 
                      InternalMet = true;
                   }
                   else if (IsGlobHelp && (!GlobalMet)) 
                   {
                      HelpOut() << "       ####### GLOBAL   #######\n" ; 
                      GlobalMet = true;
                   }

                   HelpOut()  << "  * [Name=" <<  Arg->Name()   << "] " << Arg->NameType() << Arg->Name4Help() << " :: " ;
                   if (IsIinternal)
                       HelpOut()  << "(!!== INTERNAL DONT USE DIRECTLY ==!!)";
                   HelpOut()  << Arg->Com() ;
                   bool HasDefVal = Arg->HasType(eTA2007::HDV);
                   if (HasDefVal)
                   {
                      HelpOut() << " ,[Default="  << Arg->NameValue() << "]"; 
                   }

                   HelpOut()  << "\n";

                   // Check tuning comes at end =  when tuning is reached, we have non standard param
/*
                if (TuningMet)
                {
                   MMVII_INTERNAL_ASSERT_always
                   (
                       (IsIinternal||IsGlobHelp||IsTuning),
                       "Tuning parameter must comes at end"
                   );
                }
*/
                }
	        if (DoIt)
                    PrintAdditionnalComments(Arg);
             }
          }
      }
   }
   HelpOut() << "\n";

   // Eventually, print samples of "good" uses , only with Help
   if (mDoGlobHelp)
   {
       std::vector<std::string> aVS = Samples ();
       if (! aVS.empty())
       {
          HelpOut() << " ############## ----  EXAMPLES --------- ##########\n" ; 
          for (const auto & aStr : aVS)
          {
              HelpOut() << " - " <<  aStr  << "\n";
          }
       }
       HelpOut() << "\n";
   }
}

void cMMVII_Appli::ShowAllParams() 
{
    StdOut()  << "=================== PARAM AFTER FULL INIT ============== " << std::endl;
    for (auto & Arg : mArgFac.Vec())
    {
        if (( IsInit(Arg->AdrParam()) ||  Arg->HasType(eTA2007::HDV)) &&  (!Arg->HasType(eTA2007::Global)) )
        {
            StdOut() << " * " <<  Arg->Name() << "=" <<  Arg->NameValue() << std::endl;
        }
    }
}


bool cMMVII_Appli::ModeHelp() const
{
   return mModeHelp;
}

bool cMMVII_Appli::ModeArgsSpec() const
{
   return mModeArgsSpec;
}

    // ========== Handling of Mains Sets

const tNameSet &  cMMVII_Appli::MainSet0() const { return MainSet(0); }
const tNameSet &  cMMVII_Appli::MainSet1() const { return MainSet(1); }
const tNameSet &  cMMVII_Appli::MainSet(int aK) const 
{
   CheckRangeMainSet(aK);
   if (! mVMainSets.at(aK).IsInit())
   {
      MMVII_INTERNAL_ASSERT_always(false,"No mMainSet created for K="+ ToStr(aK));
   }
   return  mVMainSets.at(aK);
}
bool   cMMVII_Appli::AcceptEmptySet(int) const {return false;}

std::vector<std::string> cMMVII_Appli::VectMainSet(int aK) const
{
   return ToVect(MainSet(aK));
}

std::string  cMMVII_Appli::UniqueStr(int aK) const
{
    auto aV = VectMainSet(aK);
    MMVII_INTERNAL_ASSERT_always(aV.size()==1,"cMMVII_Appli::UniqueStr");

    return aV[0];
}



void cMMVII_Appli::CheckRangeMainSet(int aK) const
{
   if ((aK<0) || (aK>=NbMaxMainSets))
   {
      MMVII_INTERNAL_ASSERT_always(false,"CheckRangeMainSet, out for :" + ToStr(aK));
   }
}

    // ========== Handling of V1/V2 format for output =================

void cMMVII_Appli::SignalInputFormat(int aNumV)
{
   cMMVII_Appli & TheAp = CurrentAppli();
   if (aNumV==0)
   {
   }
   else if (aNumV==1)
   {
      TheAp.mHasInputV1 = true;
   }
   else if (aNumV==2)
   {
      TheAp.mHasInputV2 = true;
   }
   else 
   {
      MMVII_INTERNAL_ASSERT_always(false,"Input version must be in {0,1,2}, got: "+ToStr(aNumV));
   }
}

// Necessary for forward use of cMMVII_Appli::OutV2Forma
bool GlobOutV2Format() { return cMMVII_Appli::OutV2Format(); }
bool   cMMVII_Appli::OutV2Format() 
{
   const cMMVII_Appli & TheAp = CurrentAppli();
   // Priority to specified output if exist
   if (TheAp.mOutPutV2) return true;
   if (TheAp.mOutPutV1) return false;
   //  In input, set it, priority to V2
   if (TheAp.mHasInputV2) return true;
   if (TheAp.mHasInputV1) return false;
   // by default V2
   return true;
}

    // ========== Handling of global Appli =================

std::vector<cMMVII_Appli *>  cMMVII_Appli::TheStackAppli ;
int  cMMVII_Appli::TheNbCallInsideP=0;
bool  cMMVII_Appli::msInDstructor = false;
cMMVII_Appli & cMMVII_Appli::CurrentAppli()
{
  MMVII_INTERNAL_ASSERT_strong(ExistAppli(),"cMMVII_Appli not created");
  return *(TheStackAppli.back());
}
bool cMMVII_Appli::ExistAppli()
{
  return !TheStackAppli.empty();
}

bool UserIsMPD()
{
     return cMMVII_Appli::CurrentAppli().UserName() == "MPD";
}
 
    // ========== Random seed  =================

const int cMMVII_Appli::msDefSeedRand = 42;
int  cMMVII_Appli::SeedRandom()
{
    return ExistAppli() ?  CurrentAppli().mSeedRand  : msDefSeedRand;
}
int  cMMVII_Appli::DefSeedRand()
{
   return msDefSeedRand;
}

const cCollecSpecArg2007 &   cMMVII_Appli::ArgObl() const { return mArgObl; }

bool cMMVII_Appli::msWithWarning =  true;
bool cMMVII_Appli::WithWarnings() {return  msWithWarning;}

    // ========== Miscelaneous functions =================

void cMMVII_Appli::AssertInitParam() const
{
  MMVII_INTERNAL_ASSERT_always(mInitParamDone,"Init Param was forgotten");
}
bool  cMMVII_Appli::IsInit(const void * aPtr) const
{
    return  mSetInit.In(aPtr);
}
void cMMVII_Appli::SetVarInit(void * aPtr)
{
    mSetInit.Add(aPtr); 
}

bool  cMMVII_Appli::IsInSpecObl(const void * aPtr)
{
    return  mSetVarsSpecObl.In(aPtr);
}
bool  cMMVII_Appli::IsInSpecFac(const void * aPtr)
{
    return  mSetVarsSpecFac.In(aPtr);
}

bool  cMMVII_Appli::IsInSpec(const void * aPtr)
{
    return IsInSpecObl(aPtr) || IsInSpecFac(aPtr);
}


void cMMVII_Appli::MMVII_WARNING(const std::string & aMes)
{
   StdOut() << "===================================================================" << std::endl;
   StdOut() <<  aMes << std::endl;
   StdOut() << "===================================================================" << std::endl;
}

std::string cMMVII_Appli::mDirBinMMVII;
std::string cMMVII_Appli::mTmpDirTestMMVII;
std::string cMMVII_Appli::mInputDirTestMMVII;
std::string cMMVII_Appli::mTopDirMMVII;
std::string cMMVII_Appli::mFullBin;
std::string cMMVII_Appli::mDirTestMMVII;
std::string cMMVII_Appli::mDirRessourcesMMVII;
std::string cMMVII_Appli::mDirLocalParameters;
std::string cMMVII_Appli::mProfileUsage;
std::string cMMVII_Appli::mDirProfileUsage;
cParamProfile cMMVII_Appli::mParamProfile;
std::string cMMVII_Appli::mDirMicMacv1;
std::string cMMVII_Appli::mDirMicMacv2;
std::string cMMVII_Appli::mVectNameDefSerial;
std::string cMMVII_Appli::mTaggedNameDefSerial;
std::string cMMVII_Appli::mMMV1Bin;

              // static Accessors
const std::string & cMMVII_Appli::TmpDirTestMMVII()   {return mTmpDirTestMMVII;}
const std::string & cMMVII_Appli::InputDirTestMMVII() {return mInputDirTestMMVII;}
const std::string & cMMVII_Appli::TopDirMMVII()       {return mTopDirMMVII;}
const std::string & cMMVII_Appli::DirBinMMVII()       {return mDirBinMMVII; }
const std::string & cMMVII_Appli::FullBin()           {return mFullBin;}
const std::string & cMMVII_Appli::DirTestMMVII()      {return mDirTestMMVII;}
const std::string & cMMVII_Appli::DirMicMacv1()       {return mDirMicMacv1;}
const std::string & cMMVII_Appli::MMV1Bin()           {return mMMV1Bin;}


const std::string & cMMVII_Appli::DirRessourcesMMVII()      {return mDirRessourcesMMVII;}
              // Accessors
const std::string & cMMVII_Appli::DirProject() const  {return mDirProject;}
int cMMVII_Appli::NbProcAllowed () const {return mNbProcAllowed;}
const std::string & cMMVII_Appli::PrefixGMA () const {return mPrefixGMA;}
const std::string & cMMVII_Appli::Prefix_TIM_GMA () const {return mPrefix_TIM_GMA;}

std::string  cMMVII_Appli::DirTmpOfCmd(eModeCreateDir aMode) const
{
   std::string aRes = DirProject() + TmpMMVIIDirPrefix + mSpecs.Name() + StringDirSeparator();
   ActionDir(aRes,aMode);
   return aRes;
}
//  std::string  cMMVII_Appli::DirTmpOfProcess(eModeCreateDir) const;


void cMMVII_Appli::InitOutFromIn(std::string &aFileOut,const std::string& aFileIn)
{
   if (! IsInit(&aFileOut))
   {
      aFileOut = mDirProject + FileOfPath(aFileIn,false);
   }
   else
   {
      aFileOut = mDirProject + aFileOut;
   } 
}

    // ==========  MMVII  Call MMVII =================

cColStrAObl& cMMVII_Appli::StrObl() {return mColStrAObl;}
cColStrAOpt& cMMVII_Appli::StrOpt() {return mColStrAOpt;}



cParamCallSys  cMMVII_Appli::StrCallMMVII
               (
		  int   aKthCall,
                  const cSpecMMVII_Appli & aCom2007,
                  const cColStrAObl& anAObl,
                  const cColStrAOpt& anAOpt,
                  const cColStrAOpt&  aSubst,
                  const std::string & aPatInit
               )
{
  cParamCallSys aRes(aCom2007);
  MMVII_INTERNAL_ASSERT_always(&anAObl==&mColStrAObl,"StrCallMMVII use StrObl() !!");
  MMVII_INTERNAL_ASSERT_always(&anAOpt==&mColStrAOpt,"StrCallMMVII use StrOpt() !!");

   // std::string aComGlob = mFullBin + " ";
   aRes.AddArgs(mFullBin);
   int aNbSubst=0;
   std::vector<bool>  aVUsedSubst(aSubst.V().size(),false);
/*
   cSpecMMVII_Appli*  aSpec = cSpecMMVII_Appli::SpecOfName(aCom2007,false); // false => dont accept no match
   if (! aSpec)  // Will see if we can di better, however SpecOfNam has generated error
      return "";
*/

   // Theoretically we can create the command  (dealing with unik msTheAppli before !) and check
   // the parameters, but it will be done in the call so maybe it's not worth the price ?
  
   // aComGlob += aCom2007.Name() + " ";
   aRes.AddArgs(aCom2007.Name());

   
   // Add mandatory args
   int aK=0;
   for (const auto & aStr : anAObl.V())
   {
       std::string aStrK = ToStr(aK);
       std::string aVal = aStr;
       // See if there is a subst for arg K
       int aKSubst=0;
       for (const auto & aPSubst :  aSubst.V())
       {
           if (aPSubst.first==aStrK)
           {
              MMVII_INTERNAL_ASSERT_always(aVal==aStr,"Multiple KSubst in StrCallMMVII ");
              aVal = aPSubst.second;
              aNbSubst++; 
              aVUsedSubst[aKSubst] = true;
           }
           aKSubst++;
       }
       aRes.AddArgs(aVal);
       aK++;
   }

   // Add optionnal args
   for (const auto & aPOpt : anAOpt.V())
   {
       // Special case, it may have be add by the auto recal process , but it will be handled separately
       if (
	          (aPOpt.first != GIP_LevCall) 
               && (aPOpt.first != GIP_PGMA) 
	       && (aPOpt.first!=  GIP_TIM_GMA)
	       && (aPOpt.first != GIP_DirProjGMA)
	       && (aPOpt.first!=  GOP_WW)
	       && (aPOpt.first!=  GIP_KthCall)
	       && (aPOpt.first!=  GIP_PatternGMA)
          )
       {
          std::string aVal = aPOpt.second;
          int aKSubst=0;
          for (const auto & aPSubst :  aSubst.V())
          {
              if (aPSubst.first==aPOpt.first)
              {
                 MMVII_INTERNAL_ASSERT_always(aVal==aPOpt.second,"Multiple Opt-Subst in StrCallMMVII ");
                 aVal = aPSubst.second;
                 aNbSubst++; 
                 aVUsedSubst[aKSubst] = true;
              }
              aKSubst++;
          }
          aRes.AddArgs(aPOpt.first + "=" + aVal);
       }
   }
   // MMVII_INTERNAL_ASSERT_always(aNbSubst==(int)aSubst.V().size(),"Impossible Subst in StrCallMMVII ");

   // Take into account the call level which must increase
   // aComGlob += GIP_LevCall + "=" + ToStr(mLevelCall+1);
   aRes.AddArgs(GIP_LevCall + "=" + ToStr(mLevelCall+1));
   aRes.AddArgs(GIP_PGMA + "=" + mPrefixGMA);
   aRes.AddArgs(GIP_TIM_GMA + "=" + mPrefix_TIM_GMA);
   aRes.AddArgs(GIP_DirProjGMA + "=" + mDirProjGMA);
   aRes.AddArgs(GOP_WW + "=" +  ToStr(WithWarnings()));
   aRes.AddArgs(GIP_KthCall + "=" +  ToStr(aKthCall));

   if (aPatInit !="")
   {
        aRes.AddArgs(GIP_PatternGMA + "=" +  Quote(aPatInit));
   }

   // If no substitution, it means it was to be added simply
   int aKSubst=0;
   for (const auto & aPSubst :  aSubst.V())
   {
      if (!aVUsedSubst[aKSubst])
      {
         aRes.AddArgs(aPSubst.first + "=" + aPSubst.second);
      }
      aKSubst++;
   }

   mColStrAObl.clear();
   mColStrAOpt.clear();
   return aRes;
}

std::list<cParamCallSys>  cMMVII_Appli::ListStrCallMMVII
                        ( 
                              const cSpecMMVII_Appli & aCom2007,const cColStrAObl& anAObl,const cColStrAOpt& anAOpt,
                              const std::string & aNameOpt  , const std::vector<std::string> &  aLVals
                        )
{
    std::list<cParamCallSys> aRes;
     
    int aKthCall=0;
    for (const auto & aVal : aLVals)
    {
       cColStrAOpt  aNewSubst; 
       aNewSubst << t2S(aNameOpt,aVal);
       aRes.push_back(StrCallMMVII(aKthCall,aCom2007,anAObl,anAOpt,aNewSubst));
       aKthCall++;
    }

    return aRes;
}

int cMMVII_Appli::ExtSysCall(const cParamCallSys & aCom, bool SVP)
{
   std::string aName  = NameFileLog(false);
   {
      cMMVII_Ofs  aOfs(aName, eFileModeOut::AppendText);
      aOfs.Ofs() << "  ---   begining at : " <<  StrDateCur() << "\n";
      aOfs.Ofs() << "        ExtCom : [" <<  aCom.Com() << "]\n";
      aOfs.Ofs().close();
   }
   int aResult = GlobSysCall(aCom,SVP);
   {
      cMMVII_Ofs  aOfs(aName, eFileModeOut::AppendText);
      aOfs.Ofs() << "  ---   ending at : " <<  StrDateCur() << "\n\n";
      aOfs.Ofs().close();
   }
   return aResult;
}



int  cMMVII_Appli::ExeCallMMVII
     (
         const cSpecMMVII_Appli&  aCom2007,
         const cColStrAObl& anAObl,
         const cColStrAOpt& anAOpt
      )
{
    cParamCallSys aComGlob = StrCallMMVII(0,aCom2007,anAObl,anAOpt);
    return  GlobSysCall(aComGlob,false);
}

int cMMVII_Appli::ExeComSerial(const std::list<cParamCallSys> & aL, bool forceExternal)
{
    for (const auto & aPCS : aL)
    {
        int aRes = aPCS.Execute(forceExternal);
        if (aRes != EXIT_SUCCESS)
        {
            MMVII_INTERNAL_ASSERT_always(false,"Error in serial com");
            return aRes;
        }
    }
    return EXIT_SUCCESS;
}

int  cMMVII_Appli::ExeOnePackComParal(const std::list<cParamCallSys> & aLCom,bool Silence)
{
   if (aLCom.empty()) return EXIT_SUCCESS;
   std::string aNameMk = "MkFile_" + mPrefixNameAppli ;
 
   std::string aName  = NameFileLog(false);
   {
      cMMVII_Ofs  aOfs(aName, eFileModeOut::AppendText);
      aOfs.Ofs() <<  "<<=============== Execute " << aLCom.size() << " in paral" << "\n";
      for (const auto & aCom : aLCom)
          aOfs.Ofs() <<  "   Com=" << aCom.Com() << "\n";
      aOfs.Ofs().close();
   }

   int aResult =  GlobParalSysCallByMkF(aNameMk,aLCom,mNbProcAllowed,false,Silence);
   {
      cMMVII_Ofs  aOfs(aName, eFileModeOut::AppendText);
      if (aResult == EXIT_SUCCESS)
         aOfs.Ofs() <<  ">>======== Done correctly paral in ===== \n";
      else
         aOfs.Ofs() <<  "!!!!!!!!!!!!  Failure in one of the commands !!!!!!!! \n";
      aOfs.Ofs().close();
   }
   return aResult;
}

int  cMMVII_Appli::ExeComParal(const std::list<cParamCallSys> & aGlobLCom,bool Silence)
{
    unsigned aNbMaxInFile =  round_up(mNbProcSystem * mMulNbInMk); // Size of allow task

    std::list<cParamCallSys> aSubList; // List that will contain a limited size of task

    for (const auto & aCom : aGlobLCom)
    {
        aSubList.push_back(aCom);
        if (aSubList.size() == aNbMaxInFile) // if we have reached the limit exec an clear
        {
            int aResult = ExeOnePackComParal(aSubList,Silence); 
            if (aResult != EXIT_SUCCESS)
               return aResult;
            aSubList.clear();
        }
    }
    // It may remain some command
    return ExeOnePackComParal(aSubList,Silence);

    // return aResult;
}


void cMMVII_Appli::InitColFromVInit()
{
   mColStrAObl.clear();
   mColStrAOpt.clear();
   for (int aK=0; aK< (int)mArgObl.size() ; aK++)
   {
       mColStrAObl << mArgObl[aK]->Value();
   }

   for (int aK=0; aK< (int)mArgFac.size() ; aK++)
   {
      if ( mArgFac[aK]->NbMatch())
      {
          mColStrAOpt << t2S(mArgFac[aK]->Name(),mArgFac[aK]->Value());
      }
   }
}

std::list<cParamCallSys>  cMMVII_Appli::ListStrAutoRecallMMVII
                          ( 
                                const std::string & aNameOpt  , 
                                const std::vector<std::string> &  aLVals,
                                const cColStrAOpt &  aLSubstInit
                          )
{
    std::list<cParamCallSys> aRes;
    std::string aPatInit="";
    //  if mandatory arg we try to recover initial value of expanded pattern
    if( std::isdigit(aNameOpt[0]))
    {
        aPatInit =  mArgObl[cStrIO<int>::FromStr(aNameOpt)]->Value();
    }

    int aKthVal = 0;
    for (const auto & aVal : aLVals) // For each value to substitute/add
    {
         InitColFromVInit(); // mColStrAObl and mColStrAOpt contains copy  command line

         cColStrAOpt  aNewSubst(cExplicitCopy(),aLSubstInit);  // make copy of aLSubstInit as it is const
         aNewSubst << t2S(aNameOpt,aVal); // subsitute/add  aVal with "named" arg aVal
         aRes.push_back(StrCallMMVII(aKthVal,mSpecs,mColStrAObl,mColStrAOpt,aNewSubst,aPatInit));
	 aKthVal++;
    }
    return aRes;
}

bool cMMVII_Appli::RunMultiSet(int aKParam,int aKSet,bool MkFSilence)
{
    std::vector<std::string> aVSetPluDir;
    {
       const std::vector<std::string> &  aVSetIm = VectMainSet(aKSet);
       for (const auto & aName : aVSetIm)
          aVSetPluDir.push_back(mDirProject + aName);
    }


    if (aVSetPluDir.size() != 1)  // Multiple image, run in parall 
    {
         eTyModeRecall aMode = MkFSilence ?  eTyModeRecall::eTMR_ParallSilence  : eTyModeRecall::eTMR_Parall  ;
         ExeMultiAutoRecallMMVII(ToStr(aKParam),aVSetPluDir,cColStrAOpt::Empty, aMode); // Recall with substitute recall itself
         mResulMultiS = (aVSetPluDir.empty()) ? EXIT_FAILURE : EXIT_SUCCESS;
         mRMSWasUsed = true;
         return true;
    }

    // so that the pattern is defined with coherent value even if run with a single file
    if (mLevelCall==0)
       mPatternInitGMA = mArgObl[aKParam]->Value();

    mRMSWasUsed = false;
    return false;
}

int cMMVII_Appli::ResultMultiSet() const
{
   MMVII_INTERNAL_ASSERT_strong(mRMSWasUsed,"MultiSet not executed, ResultMultiSet required");

   return mResulMultiS;
}


void   cMMVII_Appli::ExeMultiAutoRecallMMVII
       ( 
           const std::string & aNameOpt  , 
           const std::vector<std::string> &  aLVals,
           const cColStrAOpt &  aLSubstInit,
           eTyModeRecall  aMode
       )
{
    std::list<cParamCallSys>  aLPCS = ListStrAutoRecallMMVII(aNameOpt,aLVals,aLSubstInit);
    switch (aMode) {
    case eTyModeRecall::eTMR_Parall:
         ExeComParal(aLPCS,false);
         break;
    case eTyModeRecall::eTMR_ParallSilence:
         ExeComParal(aLPCS,true);
         break;
    case eTyModeRecall::eTMR_Serial:
         ExeComSerial(aLPCS, true);
         break;
    case eTyModeRecall::eTMR_Inside:
         ExeComSerial(aLPCS, false);
         break;
    case eTyModeRecall::eNbVals:
         break;
    }
}

int   cMMVII_Appli::LevelCall() const { return mLevelCall; }
int   cMMVII_Appli::KthCall() const { return mKthCall; }

cParamCallSys cMMVII_Appli::CommandOfMain() const
{
    cParamCallSys aRes;
    for (const auto& aK: mArgv)
         aRes.AddArgs(aK);
    return aRes;
}

std::vector<std::string>  cMMVII_Appli::Samples() const
{
   return std::vector<std::string>();
}

bool IsInit(const void * anAdr)
{
    return cMMVII_Appli::CurrentAppli().IsInit(anAdr);
}

int  cMMVII_Appli::ExeOnParsedBox()
{
    MMVII_INTERNAL_ERROR("Call to undefined method ExeOnParsedBox()");
    return EXIT_FAILURE;
}

};

