#include "cMMVII_Appli.h"


namespace MMVII
{

/**
     Default Error Function
*/
void Default_MMVII_Error(const std::string & aType,const std::string &  aMes,const char * aFile,int aLine)
{
    std::cout.flush();
    std::cerr.flush();
    StdOut().flush();
    ErrOut() << "\n\n ######################################""\n\n";
    ErrOut() << "Level=[" << aType << "]\n";
    ErrOut() << "Mes=[" << aMes << "]\n";
    if (aFile)
       ErrOut() << "at line  " << aLine << " of file " << aFile  << "\n";
    ErrOut().flush();

    cSpecMMVII_Appli::ShowCmdArgs();        // Writes to std::cout ..
    std::cout.flush();

    if (!cMMVII_Appli::ExistAppli())
    {
        abort();
    }
    abort();
}


/// Initialize Error Handler
PtrMMVII_Error_Handler MMVVI_Error = Default_MMVII_Error;


/// Change Error Handler
void MMVII_SetErrorHandler(PtrMMVII_Error_Handler aHandler)
{
    MMVVI_Error = aHandler;
}

/// Restore to default error handler
void MMVII_RestoreDefaultHandle()
{
    MMVVI_Error = Default_MMVII_Error;
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

void MMVII_UnclasseUsEr(const std::string & aMes)
{
    MMVII_UsersErrror(eTyUEr::eUnClassedError,aMes);
}

/// Warning : temporary version
/** Will evolve significativelly as MMVII grows
*/
void cMMVII_Appli::Warning(const std::string & aMes,eTyW,int line,const std::string & File)
{
    ErrOut() << "WARNING : " << aMes << "\n";
}



};

