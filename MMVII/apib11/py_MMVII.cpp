#include "py_MMVII.h"

#include "cMMVII_Appli.h"
#include "MMVII_enums.h"


namespace MMVII {
    void OpenRandom();
    void CloseRandom();
}

static void ErrHanlderPy(const std::string & aType,const std::string &  aMes,const char * aFile,int aLine)
{
    MMVII::ErrOut() << "\n\n######################################";
    MMVII::ErrOut() << "###### Python API error handler ######\n\n";
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


class MM_Module
{
    public:
    MM_Module(const std::string &pybindMMVIIDir)
    {
        if (init) //TODO: improve
        {
            ErrHanlderPy("python","MM_Module already initialized!",__FILE__,__LINE__);
        }
        MMVII::cMMVII_Appli::InitMMVIIDirs(pybindMMVIIDir);
        MMVII::MMVII_SetErrorHandler(ErrHanlderPy);
        MMVII::InitStandAloneAppli("apipy");
        MMVII::OpenRandom();
        std::cout<<"MMVII initialized."<<std::endl;
        init = true;
    }
    ~MM_Module()
    {
        MMVII::CloseRandom();
        std::cout<<"MMVII exited."<<std::endl;
    }
    static bool init;
};

bool MM_Module::init = false;

PYBIND11_MODULE(_MMVII, m) {
    using namespace MMVII;

    m.doc() = "pybind11 MMVII plugin"; // optional module docstring

    py::enum_<eModeInitImage>(m,"ModeInitImage")
            .value("eMIA_Rand", eModeInitImage::eMIA_Rand)
            .value("eMIA_RandCenter", eModeInitImage::eMIA_RandCenter)
            .value("eMIA_Null", eModeInitImage::eMIA_Null)
            .value("eMIA_V1", eModeInitImage::eMIA_V1)
            .value("eMIA_MatrixId", eModeInitImage::eMIA_MatrixId)
            .value("eMIA_NoInit", eModeInitImage::eMIA_NoInit)
            ;

    py::enum_<eTyNums>(m,"TyNums")
            .value("TN_INT1",eTyNums::eTN_INT1)
            .value("TN_U_INT1",eTyNums::eTN_U_INT1)
            .value("TN_INT2",eTyNums::eTN_INT2)
            .value("TN_U_INT2",eTyNums::eTN_U_INT2)
            .value("TN_INT4",eTyNums::eTN_INT4)
            .value("TN_U_INT4",eTyNums::eTN_U_INT4)
            .value("TN_INT8",eTyNums::eTN_INT8)
            .value("TN_REAL4",eTyNums::eTN_REAL4)
            .value("TN_REAL8",eTyNums::eTN_REAL8)
            .value("TN_REAL16",eTyNums::eTN_REAL16)
            ;


    py::class_<MM_Module>(m, "MM_Module")
            .def(py::init<const std::string &>())
            ;

    pyb_init_Ptxd(m);
    pyb_init_DataMappings(m);
    pyb_init_Geom3D(m);
    pyb_init_Images(m);
    pyb_init_Image2D(m);
    pyb_init_DenseMatrix(m);
    pyb_init_PCSens(m);
    pyb_init_Memory(m);
    pyb_init_Aime(m);
}

