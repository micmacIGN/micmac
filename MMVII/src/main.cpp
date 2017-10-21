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

                cMMVII_Appli * anAppli = (*itS)->Alloc()(argc,argv);
                // Verifie si une commande respecte les consignes de documentation
                (*itS)->Check();
                // Execute
                int aRes = anAppli->Exe();
                delete anAppli;
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



