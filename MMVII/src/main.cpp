#include "cMMVII_Appli.h"
#include <clocale>
#include "MMVII_Sys.h"

using namespace MMVII;


#ifdef MMVII_KEEP_MMV1_IMAGE
static constexpr const char* ENV_MMVII_USE_MMV1_IMAGE = "MMVII_USE_MMV1_IMAGE";
extern bool mmvii_use_mmv1_image;
#endif



int main(int argc, char ** argv)
{
#ifdef MMVII_KEEP_MMV1_IMAGE
    char *env_mmv1_image = getenv(ENV_MMVII_USE_MMV1_IMAGE);
    mmvii_use_mmv1_image = false;
    if (env_mmv1_image != nullptr) {
        if (UCaseEqual(env_mmv1_image,"on") || UCaseEqual(env_mmv1_image,"true") || UCaseEqual(env_mmv1_image,"1"))
            mmvii_use_mmv1_image = true;
    }
#endif

   std::setlocale(LC_ALL, "C");
   // std::setlocale(LC_ALL, "en_US.UTF-8");

   cMMVII_Appli::InitMMVIIDirs(MMVII_CanonicalRootDirFromExec());
   // Debug, print command
#if 0
   {
       StdOut() << "==========COMM=====   " << std::endl;
       for (int aK=0 ; aK<argc ; aK++)
       {
            if (aK) StdOut() << " ";
            StdOut() << argv[aK];
       }
       StdOut() << std::endl;
   }
#endif
    
   if (argc>1)
   {
      std::string aNameCom = argv[1];

      // Recherche la specif correspondant au nom de commande
      cSpecMMVII_Appli*  aSpec = cSpecMMVII_Appli::SpecOfName(aNameCom,true);

      // Execute si match
      if (aSpec)
      {
         std::vector<std::string> aVArgs;
         for (int aK=0 ; aK<argc; aK++)
             aVArgs.push_back(argv[aK]);
         int aRes =  aSpec->AllocExecuteDestruct(aVArgs);
         return aRes;
      }
   }

   // Affiche toutes les commandes
   for (const auto & aSpec : cSpecMMVII_Appli::VecAll())
   {
      StdOut()  << aSpec->Name() << " => " << aSpec->Comment() << std::endl;
   }

#ifdef MMVII_KEEP_MMV1_IMAGE
   StdOut()
       << "\n"
       << " >>> MMVII is using " << (mmvii_use_mmv1_image ? "<< MicMac v1 >>" : "<< GDal >>") << " to read/write image file.\n"
       << " >>> (Env var '" << ENV_MMVII_USE_MMV1_IMAGE << "' is " << (mmvii_use_mmv1_image ? "" : "NOT ") << "set to 'on', 'true' or '1')" << std::endl;
#endif

   return EXIT_SUCCESS;
}



