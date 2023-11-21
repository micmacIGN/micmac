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
    ErrOut() << "\n\n ######################################""\n" << std::endl;
    std::string errorMsg = "Level=[" + aType + "]\n";
    errorMsg +=  "Mes=[" + aMes + "]\n";
    if (aFile)
        errorMsg += "at line  " + std::to_string(aLine) + " of file "  + aFile  + "\n";
    ErrOut() << errorMsg;
    ErrOut().flush();

    cSpecMMVII_Appli::ShowCmdArgs();        // Writes to std::cout ..
    std::cout.flush();
    std::fflush(nullptr);                   // flush all open files

    if (cMMVII_Appli::ExistAppli())
    {
        cMMVII_Appli::CurrentAppli().LogCommandAbortOnError(errorMsg);
    }

StdOut() << "GETCHARRR " << std::endl; getchar();
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

void  ErrorWarnNone(eLevelCheck aLevel,const std::string & aMes)
{
      switch(aLevel)
      {
              case eLevelCheck::NoCheck : break;

              case eLevelCheck::Warning :
                   MMVII_DEV_WARNING(aMes);
              break;

              case eLevelCheck::Error :
                   MMVII_INTERNAL_ERROR(aMes);
              break;
      }
}

};

