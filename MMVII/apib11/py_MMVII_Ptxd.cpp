#include "py_MMVII.h"
#include "pybind11/numpy.h"

#include "MMVII_Ptxd.h"


using namespace MMVII;


template<typename T, int Dim>
void pyb_init_cTplBox_tpl(py::module_ &m, const std::string& name) {
    using namespace std::literals;
    using namespace pybind11::literals;

    typedef cTplBox<T,Dim> tBox;
    typedef cPtxd<T,Dim> tPt;

    auto tb = py::class_<tBox>(m, name.c_str())
            .def(py::init<const tPt&, const tPt&,bool>(),"p0"_a,"p1"_a,"allowEmpty"_a = false)
            .def(py::init<const tPt&, bool>(),"size"_a,"allowEmpty"_a = false)
            
            .def_static("empty",&tBox::Empty)
            .def_static("CenteredBoxCste",&tBox::CenteredBoxCste,"val"_a)
            .def_static("bigBox",&tBox::BigBox)
            
            .def("toR",&tBox::ToR)
            .def("toI",&tBox::ToI)
            
            .def("p0",&tBox::P0)
            .def("p1",&tBox::P1)
            .def("sz",&tBox::Sz)

            .def("nbElem",&tBox::NbElem)

            .def("sup",&tBox::Sup)
            .def("inter",&tBox::Inter,"box"_a)
            .def("dilate",py::overload_cast<const tPt&>(&tBox::Dilate,py::const_),"pt"_a)
            .def("dilate",py::overload_cast<const T&>(&tBox::Dilate,py::const_),"val"_a)
            .def("inside",py::overload_cast<const tPt&>(&tBox::Inside,py::const_),"pt"_a)
            .def("inside",py::overload_cast<const T&>(&tBox::Inside,py::const_),"val"_a)

            .def("__repr__",[name](const tBox& tb) {
                 std::ostringstream ss;
                 ss.precision(15);
                 ss << name << "((";
                 for (int i=0; i<Dim; i++) {
                     if (i > 0)
                         ss << ",";
                     ss << tb.P0()[i];
                 }
                 ss << "),(";
                 for (int i=0; i<Dim; i++) {
                     if (i > 0)
                         ss << ",";
                     ss << tb.P1()[i];
                 }
                 ss << "))" ;
                 return ss.str();
             })
            ;
}       

void pyb_init_Ptxd(py::module_ &m)
{
    pyb_init_cTplBox_tpl<int,1>(m,"Box1di");
    pyb_init_cTplBox_tpl<int,2>(m,"Box2di");
    pyb_init_cTplBox_tpl<int,3>(m,"Box3di");
    pyb_init_cTplBox_tpl<double,1>(m,"Box1dr");
    pyb_init_cTplBox_tpl<double,2>(m,"Box2dr");
    pyb_init_cTplBox_tpl<double,3>(m,"Box3dr");
}
