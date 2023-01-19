#include "py_MMVII.h"

#include "MMVII_AimeTieP.h"

using namespace MMVII;


void pyb_init_Aime(py::module_ &m) {
    py::class_<cAimeDescriptor>(m, "AimeDescriptor", DOC(MMVII_cAimeDescriptor, cAimeDescriptor))
            .def(py::init<>())
            .def("ILP", &cAimeDescriptor::ILP, DOC(MMVII_cAimeDescriptor, ILP))
            ;

    py::class_<cAimePCar>(m, "AimePCAR")
            .def(py::init<>())
            .def("Desc", py::overload_cast<>(&cAimePCar::Desc, py::const_), DOC(MMVII_cAimePCar, Desc))
            .def("Pt", py::overload_cast<>(&cAimePCar::Pt, py::const_), DOC(MMVII_cAimePCar, Pt))
            .def("L1Dist", &cAimePCar::L1Dist, DOC(MMVII_cAimePCar, L1Dist))
            .def("__repr__",
                [](const cAimePCar &a) {
                    return "<MMVII.AimePCAR>";
                }
            )
            ;

    py::class_<cSetAimePCAR>(m, "SetAimePCAR", DOC(MMVII_cSetAimePCAR, cSetAimePCAR))
            .def(py::init<>())
            .def("SaveInFile", &cSetAimePCAR::SaveInFile, DOC(MMVII_cSetAimePCAR, SaveInFile))
            .def("InitFromFile", &cSetAimePCAR::InitFromFile, DOC(MMVII_cSetAimePCAR, InitFromFile))

            .def("IsMax", &cSetAimePCAR::IsMax, py::return_value_policy::copy, DOC(MMVII_cSetAimePCAR, IsMax))
            .def("Census", py::overload_cast<>(&cSetAimePCAR::Census, py::const_), DOC(MMVII_cSetAimePCAR, Census))
            
            .def("VPC", &cSetAimePCAR::VPC, DOC(MMVII_cSetAimePCAR, VPC))
            ;

}

