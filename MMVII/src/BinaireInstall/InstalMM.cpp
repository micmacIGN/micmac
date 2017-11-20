#include <stdio.h>  
#include <unistd.h>
#include <cstdlib>
#include <assert.h>     
#include <iostream>

#include <boost/filesystem.hpp>



FILE * FopenNN(const std::string & aName,const char * aMode)
{
   FILE * aFp = fopen(aName.c_str(),aMode);
   assert(aFp!=0);
   return aFp;
}

using namespace boost::filesystem;


int main(int,char **)
{
   char aSep = path::preferred_separator;
   char Buf[1024];
   getcwd(Buf,1000);
   FILE * aFp =FopenNN("../src/ResultInstall/ResultInstall.cpp","w");
   fprintf(aFp,"#include <string> \n ");
   fprintf(aFp,"namespace MMVII {\n ");
   fprintf(aFp,"  const std::string DirBin2007=\"%s%c\";\n",Buf,aSep);
   fprintf(aFp,"};\n");
   fclose(aFp);
   
   return EXIT_SUCCESS;
}



/*
#include "memory.h"
#include <memory>
#include <fstream>
#include <string>
#include <typeinfo>
#include <vector>
#include <list>
*/


/*
void StrSystem(const std::string & aCom)
{
   int Ok = system(aCom.c_str());
   assert(Ok==EXIT_SUCCESS);
}


std::string GetLineFromFile(const std::string & aName)
{
   std::string aRes;
   FILE * aFp = FopenNN(aName); 
   bool cont=true;
   while(cont)
   {
       int aC = fgetc(aFp);
       if(isspace(aC) || isblank(aC) || aC==EOF)
         cont = false;
       else
          aRes += aC;
   }
   fclose(aFp);
   return aRes;
}

std::string GetPWD()
{
   std::string PDWFile="PWD.txt";
   StrSystem("pwd >"+PDWFile);
   std::string PWDLine = GetLineFromFile(PDWFile);
   return PWDLine;
}
*/
