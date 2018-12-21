#include "../include/MMVII_all.h"

#include "ResultInstall/ResultInstall.cpp"
using namespace MMVII;


int main(int argc, char ** argv)
{
   std::setlocale(LC_ALL, "C");
   // std::setlocale(LC_ALL, "en_US.UTF-8");

   if (argc>1)
   {
      std::string aNameCom = argv[1];

      // Recherche la specif correspondant au nom de commande
      cSpecMMVII_Appli*  aSpec = cSpecMMVII_Appli::SpecOfName(aNameCom,true);

      // Execute si match
      if (aSpec)
      {
         // Check if a  command respects specif of documentation
         aSpec->Check();
         // Add this one to check  destruction with unique_ptr
         const cMemState  aMemoState= cMemManager::CurState() ;
         int aRes=-1;
         {
            // Use allocator
            tMMVII_UnikPApli anAppli = aSpec->Alloc()(argc,argv,*aSpec);
            // Execute
            anAppli->InitParam();
// std::cout << "IIInnparammm " <<   anAppli->StrOpt().V().size() << " " << anAppli->StrObl().V().size() << "\n"; getchar();
            if (anAppli->ModeHelp())
               aRes = EXIT_SUCCESS;
            else
               aRes = anAppli->Exe();
         }
         cMemManager::CheckRestoration(aMemoState);
         return aRes;
      }
   }

   // Affiche toutes les commandes
   for (const auto & aSpec : cSpecMMVII_Appli::VecAll())
   {
       StdOut()  << aSpec->Name() << " => " << aSpec->Comment() << "\n";
   }
   return EXIT_SUCCESS;
}



