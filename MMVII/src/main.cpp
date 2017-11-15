#include "../include/MMVII_all.h"

using namespace MMVII;

int main(int argc, char ** argv)
{
   std::setlocale(LC_ALL, "C");
   // std::cout << 3.15 << "\n";
   // std::setlocale(LC_ALL, "en_US.UTF-8");
   std::vector<cSpecMMVII_Appli*> &  aVSpecAll = cSpecMMVII_Appli::VecAll();

   std::string aNameCom ;
   if (argc>1)
   {
        aNameCom = argv[1];
        // Recherche la specif correspondant au nom de commande
        for (auto itS=aVSpecAll.begin() ; itS!=aVSpecAll.end() ; itS++)
        {
            // Execute si match
            if ((*itS)->Name()==aNameCom)
            {
                // Ajoute celui la pour teste la destruction avec unique_ptr
                const cMemState  aMemoState= cMemManager::CurState() ;
                int aRes=-1;
                {

                    tMMVII_UnikPApli anAppli = (*itS)->Alloc()(argc,argv);
                    // Verifie si une commande respecte les consignes de documentation
                    (*itS)->Check();
                    // Execute
                    if (anAppli->ModeHelp())
                       aRes = EXIT_SUCCESS;
                    else
                       aRes = anAppli->Exe();
                // delete anAppli;
                }
                cMemManager::CheckRestoration(aMemoState);
                return aRes;
            }
        }
   }

   // Affiche toutes les commandes
   for (auto itS=aVSpecAll.begin() ; itS!=aVSpecAll.end() ; itS++)
   {
       std::cout << (*itS)->Name() << " => " << (*itS)->Comment() << "\n";
   }
   


   return EXIT_SUCCESS;
}



