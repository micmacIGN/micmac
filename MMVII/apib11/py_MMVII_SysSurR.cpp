#include "py_MMVII.h"
#include "MMVII_SysSurR.h"


using namespace MMVII;

void pyb_init_SysSurR(py::module_ &m)
{
    py::class_<cLinearOverCstrSys<tREAL8>>(m, "LinearOverCstrSys", DOC(MMVII_cLinearOverCstrSys))
        .def_property_readonly("nbVar",&cLinearOverCstrSys<tREAL8>::NbVar,DOC(MMVII_cLinearOverCstrSys,NbVar))
        ;

    py::class_<cLeasSqtAA<tREAL8>,cLinearOverCstrSys<tREAL8>>(m, "LeasSqtAA", DOC(MMVII_cLeasSqtAA))
        .def(py::init<int>(),py::arg("nbVar"),DOC(MMVII_cLeasSqtAA,cLeasSqtAA))
        ;

    m.def("allocL1_Barrodale",&AllocL1_Barrodale<tREAL8>,py::arg("nbVar"));

}
