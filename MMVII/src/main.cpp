#include "../include/MMVII_all.h"


int main(int argc, char ** argv)
{
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



