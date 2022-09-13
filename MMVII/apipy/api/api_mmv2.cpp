#include "api_mmv2.h"
#include <iostream>
#include <sstream>
#include <cstdlib>
#include "MMVII_all.h"
#include <stdexcept>

const std::string MMVII::DirBin2007 = std::string(getenv("HOME"))+"/.local/mmv2/MMVII/bin/";
extern bool TheExitOnBrkp;

static void ErrHanlderPy(const std::string & aType,const std::string &  aMes,const char * aFile,int aLine)
{
    MMVII::ErrOut() << "\n\n ######################################""\n\n";
    MMVII::ErrOut() << "\n\n ###### Python API error handler ######""\n\n";
    MMVII::ErrOut() << "Level=[" << aType << "]\n";
    MMVII::ErrOut() << "Mes=[" << aMes << "]\n";
    if (aFile)
       MMVII::ErrOut() << "at line  " << aLine << " of file " << aFile  << "\n";
    throw std::runtime_error(aType + " " + aMes);
}


void mmv2_init()
{
	TheExitOnBrkp =true;
	MMVII::MMVII_SetErrorHandler(ErrHanlderPy);
	MMVII::InitStandAloneAppli("apipy");

	std::cout<<"mmv2 initialized."<<std::endl;
	//delete appli; //the fake apply must exist during module usage
}

//shadow classic ElExit to use throw(runtime_error) instead of exit(code)
void ElExit(int aLine,const char * aFile,int aCode,const std::string & aMessage)
{
    //cFileDebug does not exist in pymm3d for now
   /*cFileDebug::TheOne.Close(aCode);
   if (aCode==0)
      StdEXIT(0);
    std::string aFileName = ( isUsingSeparateDirectories() ?
                              MMTemporaryDirectory() + "MM-Error-"+ GetUnikId() + ".txt" :
                              Dir2Write() + "MM-Error-"+ GetUnikId() + ".txt" );
   FILE * aFP = fopen(aFileName.c_str(),"a+");
   if (aFP)
   {
      fprintf(aFP,"Exit with code %d \n",aCode);
      fprintf(aFP,"Generated from line %d  of file %s \n",aLine,aFile);
      fprintf(aFP,"PID=%d\n",mm_getpid());
      if (aMessage!="")
         fprintf(aFP,"Context=[%s]\n",aMessage.c_str());
      for (int aK=0 ; aK<(int)GlobMessErrContext.size() ; aK++)
         fprintf(aFP,"GMEC:%s\n",GlobMessErrContext[aK].c_str()),
      fprintf(aFP,"MM3D-Command=[%s]\n",GlobArcArgv.c_str());
   }*/

// std::cout << "ELExit " << __FILE__ << __LINE__ << " " << aCode << " " << GlobArcArgv << "\n";
   //StdEXIT(aCode);
   std::ostringstream oss;
   oss<<"MicMac exit with code "<<aCode;
   throw(std::runtime_error(oss.str()));
}
