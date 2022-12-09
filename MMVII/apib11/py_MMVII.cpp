#include "py_MMVII.h"

#include "cMMVII_Appli.h"


namespace MMVII {
    void OpenRandom();
    void CloseRandom();
}

const char* pybindMMVIIDir = PYBIND_MMVII_DIR;

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

//shadows classic ElExit to use throw(runtime_error) instead of exit(code)
void ElExit(int aLine,const char * aFile,int aCode,const std::string & aMessage)
{
   std::ostringstream oss;
   oss<<"MicMac exit with code "<<aCode;
   throw(std::runtime_error(oss.str()));
}


class TheModule
{
    public:
    TheModule()
    {
        MMVII::cMMVII_Appli::InitMMVIIDirs(pybindMMVIIDir);
        MMVII::MMVII_SetErrorHandler(ErrHanlderPy);
        MMVII::InitStandAloneAppli("apipy");
        MMVII::OpenRandom();
        std::cout<<"MMVII initialized."<<std::endl;
    }
    ~TheModule()
    {
        MMVII::CloseRandom();
        std::cout<<"MMVII exited."<<std::endl;
    }
};



PYBIND11_MODULE(MMVII, m) {
    static TheModule theModule;     // Force initialisation after all MMVII initialisation

    m.doc() = "pybind11 MMVII plugin"; // optional module docstring

    pyb_init_Geom(m);
    pyb_init_PerspCamIntrCalib(m);
    pyb_init_DenseMatrix(m);
}

