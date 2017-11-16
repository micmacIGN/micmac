#include "include/MMVII_all.h"

namespace MMVII
{

void MMVVI_Error(const std::string & aType,const std::string &  aMes,const char * aFile,int aLine)
{
    std::cout << "\n\n ######################################""\n\n";
    std::cout << "Level=[" << aType << "]\n";
    std::cout << "Mes=[" << aMes << "]\n";
    std::cout << "at line  " << aLine << " of file " << aFile  << "\n";

    if (!cMMVII_Appli::ExistAppli())
    {
       getchar();
       exit(-1);
    }
}



};

