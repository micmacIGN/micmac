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

cMMVII_Appli::~cMMVII_Appli()
{
   mArgObl.clear();
   mArgFac.clear();
   // Verifie que tout ce qui a ete alloue a ete desalloue 
   cMemManager::CheckRestoration(mMemStateBegin);
}


cMMVII_Appli::cMMVII_Appli
(
      int argc,
      char ** argv
)  :
   mArgc          (argc),
   mArgv          (argv),
   mFullBin       (mArgv[0]),
   mDirMMVII      (DirOfPath(mFullBin)),
   mBinMMVII      (FileOfPath(mFullBin)),
   mDirMicMacv1   (UpDir(mDirMMVII,2)),
   mDirProject    (DirCur()),
   mModeHelp      (false),
   mLevelCall     (0),
   mMemStateBegin (cMemManager::CurState())
{
}

void cMMVII_Appli::InitParam(cCollecArg2007 & anArgObl, cCollecArg2007 & anArgFac)
{
  // Check that  cCollecArg2007 were used with the good values
  MMVII_INTERNAL_ASSERT_always((&anArgObl)==&mArgObl,"cMMVII_Appli dont respect cCollecArg2007");
  MMVII_INTERNAL_ASSERT_always((&anArgFac)==&mArgFac,"cMMVII_Appli dont respect cCollecArg2007");

  // Add common optional parameters
  mArgFac
      <<  AOpt2007(mDirProject ,"DProj","Project Directory",{eTA2007::ProjectDir,eTA2007::Common})
      <<  AOpt2007(mLevelCall,"LevCall","Internal : Don't Use !!",{eTA2007::Internal,eTA2007::Common})
  ;

  // Check that optionnal parameters begin with alphabetic caracters
  for (auto aSpec : mArgFac.Vec())
  {
      if (!std::isalpha(aSpec->Name()[0]))
      {
          MMVII_INTERNAL_ASSERT_always(false,"Name of optional param must begin with alphabetic => ["+aSpec->Name()+"]");
      }
  }


  // Test if we are in help mode
  for (int aKArg=0 ; aKArg<mArgc ; aKArg++)
  {
      char * aArgK = mArgv[aKArg];
      if (UCaseBegin("help",aArgK) || UCaseBegin("-help",aArgK)|| UCaseBegin("--help",aArgK))
      {
         mModeHelp = true;
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


  mLevelCall++; // So that is incremented if a new call is made

  if (aNbArgGot < aNbObl)
  {
      MMVII_INTERNAL_ASSERT_user
      (
          false,
          "Not enough Arg, expecting " + ToS(aNbObl)  + " , Got only " +  ToS(aNbArgGot)
      );
  }
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
       if (! IsIinternal)
       {
          bool GlobHelp = Arg->HasType(eTA2007::Common);
          if (GlobHelp) 
             std::cout << " #COM " ; 
          else
             std::cout << " * " ; 
          std::cout << "[Name=" <<  Arg->Name()   << "] " << Arg->NameType() << " :: " << Arg->Com() << "\n";
       }
   }
}

bool cMMVII_Appli::ModeHelp() const
{
   return mModeHelp;
}



};

