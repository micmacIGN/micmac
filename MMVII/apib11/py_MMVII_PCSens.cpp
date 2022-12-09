#include "py_MMVII.h"

#include "MMVII_PCSens.h"

PYBIND11_MAKE_OPAQUE(std::vector<double>);

using namespace MMVII;


void pyb_init_PerspCamIntrCalib(py::module_ &m) {
    py::bind_vector<std::vector<double>>(m, "VectorDouble");

    py::class_<cPerspCamIntrCalib>(m, "PerspCamIntrCalib", DOC(MMVII_cPerspCamIntrCalib))
            .def("toFile", &cPerspCamIntrCalib::ToFile)
            .def_static("fromFile", &cPerspCamIntrCalib::FromFile)

            .def("degDir", &cPerspCamIntrCalib::DegDir)
            .def("f", &cPerspCamIntrCalib::F, DOC(MMVII_cPerspCamIntrCalib, F))
            .def("pp", &cPerspCamIntrCalib::PP, DOC(MMVII_cPerspCamIntrCalib, PP))
            .def("name", &cPerspCamIntrCalib::Name,DOC(MMVII_cPerspCamIntrCalib, Name))

            .def("initRandom", &cPerspCamIntrCalib::InitRandom,DOC(MMVII_cPerspCamIntrCalib, InitRandom))

            .def("vParamDist", [](cPerspCamIntrCalib& c){ return &c.VParamDist();}, py::return_value_policy::reference_internal)
            ;

}

