#include "include/MMVII_all.h"

namespace MMVII
{

template <class Type> class cOneArg
{
    public :
        cOneArg(Type & aVal,const std::string & aName,const std::string & aCom);
    private :
         Type * mVal;
};



cMMVII_Appli * cMMVII_Appli::msTheAppli = 0;


cMMVII_Appli & cMMVII_Appli::TheAppli()
{
  MMVII_INTERNAL_ASSERT_medium(msTheAppli!=0,"cMMVII_Appli not created");

  return *msTheAppli;
}

cMMVII_Appli::~cMMVII_Appli()
{
   // Verifie que tout ce qui a ete alloue a ete desalloue 
   cMemManager::CheckRestoration(mMemStateBegin);
   
   // Par curiosite
   // std::cout << "Nb obj created : " << cMemManager::CurState().NbObjCreated() << "\n";
}


cMMVII_Appli::cMMVII_Appli(int argc,char ** argv,const std::string & aDirChantier,cArgMMVII_Appli)  :
   mArgc          (argc),
   mArgv          (argv),
   mFullBin       (mArgv[0]),
   mDirMMVII      (DirOfPath(mFullBin)),
   mBinMMVII      (FileOfPath(mFullBin)),
   mDirMicMacv1   (UpDir(mDirMMVII,2)),
   mDirChantier   (aDirChantier),
   mMemStateBegin (cMemManager::CurState())
{
  MMVII_INTERNAL_ASSERT_strong(msTheAppli==0,"cMMVII_Appli only one by process");
  msTheAppli = this;

  std::cout << "MMV1 "  << mDirMicMacv1  << "\n";
}



};

