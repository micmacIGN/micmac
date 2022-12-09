#ifndef PY_MMVII_H
#define PY_MMVII_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "docstrings.h"

namespace py = pybind11;

void pyb_init_PerspCamIntrCalib(py::module_ &);
void pyb_init_Geom(py::module_ &m);
void pyb_init_DenseMatrix(py::module_ &m);


#endif // PY_MMVII_H
