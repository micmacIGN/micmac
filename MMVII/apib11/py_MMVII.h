#ifndef PY_MMVII_H
#define PY_MMVII_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "docstrings.h"

namespace py = pybind11;

void pyb_init_PCSens(py::module_ &);
void pyb_init_Ptxd(py::module_ &m);
void pyb_init_DataMappings(py::module_ &m);
void pyb_init_Geom3D(py::module_ &m);
void pyb_init_Images(py::module_ &m);
void pyb_init_Image2D(py::module_ &m);
void pyb_init_DenseMatrix(py::module_ &m);
void pyb_init_Memory(py::module_ &m);
void pyb_init_Aime(py::module_ &m);
#endif // PY_MMVII_H
