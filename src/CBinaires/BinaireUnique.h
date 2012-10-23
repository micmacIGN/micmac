#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef WIN32
#include <windows.h>
#elif __APPLE__
#include <mach-o/dyld.h>
#else
#define ELISE_CAR_DIR  '/' 
#endif

#define PATH_BUFFER_SIZE 1024


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
    char aPath[PATH_BUFFER_SIZE] = { 0 };
	int aK;

    printf("Commande = %s\n",aCom);

#ifdef WIN32
	{
		// get full path of current executable
		DWORD pathSize = GetModuleFileName(NULL,aPath, PATH_BUFFER_SIZE );
		char *itChar = aPath+(pathSize-1);
	
		// look for the last backslash
		while ( *itChar!='\\' ) itChar--;
		// path ends after the last backslash
		itChar[1] = '\0';
	}
#elif __APPLE__
    {
		char *itChar;
		uint32_t size = PATH_BUFFER_SIZE;
		_NSGetExecutablePath(aPath, &size);
				
		// look for the last slash
		size = strlen( aPath );
		itChar = aPath+(size-1);
		while ( *itChar!='/' ) itChar--;
		// path ends after the last backslash
		itChar[1] = '\0';		
	}
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

	  int aL = strlen(aPath) -1;
	  while (aL && (aPath[aL] != ELISE_CAR_DIR)) aL--;
	  if (aL)
		  aPath[aL+1] = 0;
	  else
	  {
		  aPath[0] = 0;
		  strcat(aPath,"./");
	  }   
  }
#endif

  strcat(aPath,"mm3d ");
  strcat(aPath,aCom);

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

