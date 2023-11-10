#include "py_MMVII.h"

#include "MMVII_Sensor.h"


using namespace MMVII;

static void pyb_init_cDirsPhProj(py::module_ &m)
{
    py::class_<cDirsPhProj>(m, "DirsPhProj", DOC(MMVII_cDirsPhProj))
        .def("dirIn",&cDirsPhProj::DirIn,DOC(MMVII_cDirsPhProj,DirIn))
        .def("dirOut",&cDirsPhProj::DirOut,DOC(MMVII_cDirsPhProj,DirOut))
        .def("fullDirIn",&cDirsPhProj::FullDirIn,DOC(MMVII_cDirsPhProj,FullDirIn))
        .def("fullDirOut",&cDirsPhProj::FullDirOut,DOC(MMVII_cDirsPhProj,FullDirOut))
        .def("dirLocOfMode",&cDirsPhProj::DirLocOfMode,DOC(MMVII_cDirsPhProj,DirLocOfMode))
        ;
}

static void pyb_init_cPhotogrammetricProject(py::module_ &m)
{
    py::class_<cPhotogrammetricProject>(m, "PhotogrammetricProject", DOC(MMVII_cPhotogrammetricProject))
        .def("dPOrient",&cPhotogrammetricProject::DPOrient,DOC(MMVII_cPhotogrammetricProject,DPOrient))
        ;

}

void pyb_init_Sensor(py::module_ &m)
{

    pyb_init_cDirsPhProj(m);
}
