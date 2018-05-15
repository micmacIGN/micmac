#include "include/MMVII_all.h"

namespace MMVII
{

void MMVVI_Error(const std::string & aType,const std::string &  aMes,const char * aFile,int aLine)
{
    ErrOut() << "\n\n ######################################""\n\n";
    ErrOut() << "Level=[" << aType << "]\n";
    ErrOut() << "Mes=[" << aMes << "]\n";
    if (aFile)
       ErrOut() << "at line  " << aLine << " of file " << aFile  << "\n";

    if (!cMMVII_Appli::ExistAppli())
    {
       getchar();
       exit(-1);
    }
    getchar();
    exit(-1);
}

void MMVII_UsersErrror(const eTyUEr & aRef,const std::string & aMes)
{
    MMVVI_Error
    (
        "UserEr:" +  E2Str(aRef),
        aMes,
        nullptr,
        -1
    );
}


/// Warning : temporary version
/** Will evolve significativelly as MMVII grows
*/
void cMMVII_Appli::Warning(const std::string & aMes,eTyW,int line,const std::string & File)
{
    ErrOut() << "WARNING : " << aMes << "\n";
}



};

