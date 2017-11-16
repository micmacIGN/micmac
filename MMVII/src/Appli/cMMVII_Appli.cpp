#include "include/MMVII_all.h"


namespace MMVII
{

/*  ============================================== */
/*                                                 */
/*                cMMVII_Appli                     */
/*                                                 */
/*  ============================================== */

cMMVII_Appli *  cMMVII_Appli::msTheAppli = 0;


cMMVII_Appli & cMMVII_Appli::TheAppli()
{
  MMVII_INTERNAL_ASSERT_medium(msTheAppli!=0,"cMMVII_Appli not created");

  return *msTheAppli;
}

bool cMMVII_Appli::ExistAppli()
{
  return msTheAppli != 0;
}

void cMMVII_Appli::AssertInitParam()
{
  MMVII_INTERNAL_ASSERT_always(mInitParamDone,"Init Param was forgotten");
}


cMMVII_Appli::~cMMVII_Appli()
{
   AssertInitParam();
   delete mSetInit;
   mArgObl.clear();
   mArgFac.clear();
   // Verifie que tout ce qui a ete alloue a ete desalloue 
   cMemManager::CheckRestoration(mMemStateBegin);
}

bool  cMMVII_Appli::IsInit(void * aPtr)
{
    return  mSetInit->In(aPtr);
}

template <class Type> Type PrintArg(const Type & aVal,const std::string & aName)
{
    std::cout << " For " << aName << " V=" << aVal << "\n";
    return aVal;
}

cMMVII_Appli::cMMVII_Appli
(
      int argc,
      char ** argv
)  :
   mMemStateBegin (cMemManager::CurState()),
   mArgc          (argc),
   mArgv          (argv),
   mFullBin       (mArgv[0]),
   // mFullBin       (AbsoluteName(mArgv[0])),
   // mFullBin       (PrintArg(AbsoluteName(mArgv[0]),"ABS")),
   mDirMMVII      (DirOfPath(mFullBin)),
   mBinMMVII      (FileOfPath(mFullBin)),
   mDirMicMacv1   (UpDir(mDirMMVII,2)),
   mDirMicMacv2   (UpDir(mDirMMVII,1)),
   mDirProject    (DirCur()),
   mDirTestMMVII  (mDirMicMacv2 + MMVIITestDir),
   mModeHelp      (false),
   mDoGlobHelp    (false),
   mDoInternalHelp(false),
   mShowAll       (false),
   mLevelCall     (0),
   mDoInitProj    (false),
   mSetInit       (AllocUS<void *> ()),
   mInitParamDone (false)
{
}

void cMMVII_Appli::InitParam(cCollecArg2007 & anArgObl, cCollecArg2007 & anArgFac)
{
  mInitParamDone = true;
  // Check that  cCollecArg2007 were used with the good values
  MMVII_INTERNAL_ASSERT_always((&anArgObl)==&mArgObl,"cMMVII_Appli dont respect cCollecArg2007");
  MMVII_INTERNAL_ASSERT_always((&anArgFac)==&mArgFac,"cMMVII_Appli dont respect cCollecArg2007");

  std::string aDP; // mDirProject is handled specially so dont put mDirProject in AOpt2007
                   // becauser  InitParam, it may change the correct value 

  // Add common optional parameters
  mArgFac
      <<  AOpt2007(aDP ,NameDirProj,"Project Directory",{eTA2007::DirProject,eTA2007::Common})
      <<  AOpt2007(mLevelCall,"LevCall","Internal : Don't Use !!",{eTA2007::Internal,eTA2007::Common})
      <<  AOpt2007(mShowAll,"ShowAll","Internal : Don't Use !!",{eTA2007::Internal,eTA2007::Common})
  ;

  // Check that name optionnal parameters begin with alphabetic caracters
  for (auto aSpec : mArgFac.Vec())
  {
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
      char * aArgK = mArgv[aKArg];
      if (UCaseBegin("help",aArgK) || UCaseBegin("-help",aArgK)|| UCaseBegin("--help",aArgK))
      {
         mModeHelp = true;
         while (*aArgK=='-') aArgK++;
         mDoGlobHelp = (*aArgK=='H');
         mDoInternalHelp = CaseSBegin("HELP",aArgK);
      }
  }
  if (mModeHelp)
  {
      GenerateHelp();
      return;
  }

  // std::cout  <<  "SIZE ARGS " <<  mArgFac.size() << " " << mArgObl.size() << "\n";


  MMVII_INTERNAL_ASSERT_always(msTheAppli==0,"cMMVII_Appli only one by process");
  msTheAppli = this;

  // std::cout << "MMV1 "  << mDirMicMacv1  << "\n";

  int aNbObl = mArgObl.size(); //  Number of mandatory argument expected
  int aNbArgGot = 0; // Number of  Arg received untill now
  bool Optional=false; // Are we in the optional phase of argument parsing

  // To be abble to process in  the same loop mandatory and optional
  std::vector<std::string> aVValues;
  tVecArg2007              aVSpec;

  for (int aKArg=0 ; aKArg<mArgc ; aKArg++)
  {
      Optional = (aNbArgGot>=aNbObl);
      // If --Name replace by Name, maybe usefull for completion
      if (Optional && (mArgv[aKArg][0]=='-') && (mArgv[aKArg][1]=='-'))
         mArgv[aKArg] += 2;
      char * aArgK = mArgv[aKArg];
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
             for (auto aSpec : mArgFac.Vec())
             {
                 if (aSpec->Name() == aName)
                 {
                    aNbSpecGot++;
                    aVSpec.push_back(aSpec);
                    // Should be
                    if (aNbSpecGot==2)
                    {
                        MMVII_INTERNAL_ASSERT_always(false,"\""+ aName +"\" is multiple in specification");
                    }
                    if (aSpec->NbMatch() !=0)
                    {
                        MMVII_INTERNAL_ASSERT_user(false,"\""+aName +"\" was used multiple time");
                    }
                    aSpec->IncrNbMatch();
                 }
             }
             if (aNbSpecGot==0)
             {
                MMVII_INTERNAL_ASSERT_user(false,"\""+aName +"\" is not a valide optionnal value");
             }
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
      MMVII_INTERNAL_ASSERT_user
      (
          false,
          "Not enough Arg, expecting " + ToS(aNbObl)  + " , Got only " +  ToS(aNbArgGot)
      );
  }
  MMVII_INTERNAL_ASSERT_always(aNbArgTot==aVSpec.size(),"Interncl check size Value/Spec");


  // First compute the directory of project that may influence all other computation
  for (size_t aK=0 ; aK<aNbArgTot; aK++)
  {
     if (aVSpec[aK]->HasType(eTA2007::DirProject))
        mDirProject = aVValues[aK];
     else if (aVSpec[aK]->HasType(eTA2007::FileDirProj))
        mDirProject = DirOfPath(aVValues[aK],false);
  }
  MakeNameDir(mDirProject);



  //  Initialize the paramaters
  for (size_t aK=0 ; aK<aNbArgTot; aK++)
  {
       aVSpec[aK]->InitParam(aVValues[aK]);
       mSetInit->Add(aVSpec[aK]->AdrParam()); ///< Memorize this value was initialized
  }
  // MakeNameDir(mDirProject);
  
  // Print the info, debugging
  if (mShowAll)
  {
     // Print the value of all parameter
     for (size_t aK=0 ; aK<aNbArgTot; aK++)
     {
         std::cout << aVSpec[aK]->Name()  << " => [" << aVValues[aK] << "]" << std::endl;
     }
     std::cout << "---------------------------------------" << std::endl;
     std::cout << "IS INIT  DP: " << IsInit(&aDP) << std::endl;

     std::cout << "DIRPROJ=[" << mDirProject << "]" << std::endl;
  }

  // By default, if calls is done at top level, assure that everything is init
  if (!IsInit(&mDoInitProj))
     mDoInitProj = (mLevelCall==0);

  if (mDoInitProj)
  {
     InitProject();
  }

  mLevelCall++; // So that is incremented if a new call is made

for (int aK=0 ; aK<100 ; aK++)
{
    std::cout << "Lettre SFPT a diffuser \n";
}

}

void cMMVII_Appli::InitProject()
{
   CreateDirectories(mDirProject+TmpMMVIIDir,true);
}

void cMMVII_Appli::GenerateHelp()
{
   std::cout << "== Mandatory unnamed args : ==\n";

   for (auto Arg : mArgObl.Vec())
   {
       std::cout << " * " << Arg->NameType() << " :: " << Arg->Com() << "\n";
   }

   std::cout << "== Optional named args : ==\n";
   for (auto Arg : mArgFac.Vec())
   {
       bool IsIinternal = Arg->HasType(eTA2007::Internal);
       if ((! IsIinternal) || mDoInternalHelp)
       {
          bool isGlobHelp = Arg->HasType(eTA2007::Common);
          if ((!isGlobHelp) || mDoGlobHelp)
          {
             if (IsIinternal) 
                std::cout << " #III " ; 
             else if (isGlobHelp) 
                std::cout << " #COM " ; 
             else
                std::cout << " * " ; 
             std::cout << "[Name=" <<  Arg->Name()   << "] " << Arg->NameType() << " :: " << Arg->Com() << "\n";
          }
       }
   }
}

bool cMMVII_Appli::ModeHelp() const
{
   return mModeHelp;
}



};

