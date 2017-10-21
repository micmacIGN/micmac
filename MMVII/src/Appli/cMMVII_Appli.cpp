#include "include/MMVII_all.h"

namespace MMVII
{

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
   std::cout << "Nb obj created : " << cMemManager::CurState().NbObjCreated() << "\n";
}

cMMVII_Appli::cMMVII_Appli(int,char **,cArgMMVII_Appli)  :
   mMemStateBegin (cMemManager::CurState())
{
  MMVII_INTERNAL_ASSERT_strong(msTheAppli==0,"cMMVII_Appli only one by process");
  msTheAppli = this;
}



};

