#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#elif __APPLE__
#else
#define ELISE_CAR_DIR  '/' 
#endif



void FataleError(const char * aMes)
{
    printf("Error: %s\n",aMes);
    printf("Fatal error : exit\n");
    exit(-1);
}
    

int  BinaireUnique
      (
           const char * aCom,
           int argc,
           char ** argv
      )
{
    printf("Commande =%s\n",aCom);
    char aPath[1024];

     
#ifdef _WIN32
          FataleError("No mm3d path in Windows version");
#elif __APPLE__
          FataleError("No mm3d path in Max version");
#else
  {
      ssize_t len;
      if ( ( len= readlink( "/proc/self/exe", aPath, sizeof(aPath)-1 ) ) != -1 )
      {
          aPath[len] = '\0'; 
      }
      else
      {
          FataleError("Cannot compute mm3d's path");
      }
  }
  int aL = strlen(aPath) -1;
  while (aL && (aPath[aL] != ELISE_CAR_DIR)) aL--;
  if (aL)
      aPath[aL+1] = 0;
  else
  {
      aPath[0] = 0;
      strcat(aPath,"./");
  }   
#endif

  strcat(aPath,"mm3d ");
  strcat(aPath,aCom);

  int aK;
  for (aK=1 ; aK<argc ; aK++)
  {
      strcat(aPath," \"");
      strcat(aPath,argv[aK]);
      strcat(aPath,"\"");
      // AddArg(aPath,argv[aK],aK!=0);
  }

  // printf("Com =%s\n",aPath);
  return system(aPath);
}

