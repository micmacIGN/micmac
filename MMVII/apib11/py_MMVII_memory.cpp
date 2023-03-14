#include "py_MMVII.h"

#include "MMVII_memory.h"


using namespace MMVII;

void pyb_init_Memory(py::module_ &m) {
    using namespace pybind11::literals;

    py::class_<cMemState>(m, "MemState", DOC(MMVII_cMemState))
            .def("nbObjCreated", &cMemState::NbObjCreated)
            .def("setCheckAtDestroy",&cMemState::SetCheckAtDestroy)
            .def(py::self == py::self)
            ;

    py::class_<cMemManager>(m, "MemManager", DOC(MMVII_cMemManager))
            .def_static("curState", &cMemManager::CurState)
            .def_static("isOkCheckRestoration", &cMemManager::IsOkCheckRestoration,"state"_a)
            .def_static("checkRestoration", &cMemManager::CheckRestoration,"state"_a)
            .def_static("setActiveMemoryCount", &cMemManager::SetActiveMemoryCount,"on"_a)
            .def_static("isActiveMemoryCount", &cMemManager::IsActiveMemoryCount)
            ;

    py::class_<cMemCheck>(m, "MemCheck", DOC(MMVII_cMemCheck))
            .def_static("nbObjLive", &cMemCheck::NbObjLive)
            ;

}
