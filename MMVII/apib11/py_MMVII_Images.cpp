#include "py_MMVII.h"

#include "MMVII_Images.h"


using namespace MMVII;


template<int Dim>
void pyb_init_cPixBox_tpl(py::module_ &m, const std::string& name) {
    using namespace std::literals;
    using namespace pybind11::literals;

    typedef cPixBox<Dim> tPixBox;
    // typedef typename tPixBox::tScalPt tScalPt;   // not used for now ...
    typedef typename tPixBox::tPt tPt;
    typedef typename tPixBox::tBox tBox;

    auto pBox = py::class_<tPixBox,tBox>(m, name.c_str())
            .def(py::init<const tPt&, const tPt&,bool>(),"p0"_a,"p1"_a,"allowEmpty"_a = false)
            .def(py::init<const tBox&>(),"box"_a)
            
            .def_static("boxWindow",py::overload_cast<int>(&tPixBox::BoxWindow),"size"_a)
            .def_static("boxWindow",py::overload_cast<const tPt&, int>(&tPixBox::BoxWindow),"center"_a,"size"_a)
            .def_static("boxWindow",py::overload_cast<const tPt&, const tPt&>(&tPixBox::BoxWindow),"center"_a,"size"_a)

            .def("insideBL",&tPixBox::InsideBL,"pt"_a)

            .def("__iter__",
                      [](const tPixBox& pBox) { return py::make_iterator(pBox.begin(),pBox.end()); },
                      py::keep_alive<0, 1>() /* Essential: keep object alive while iterator exists */)

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

void pyb_init_Images(py::module_ &m)
{
    pyb_init_cPixBox_tpl<1>(m,"Rect1");
    pyb_init_cPixBox_tpl<2>(m,"Rect2");
    pyb_init_cPixBox_tpl<3>(m,"Rect3");
}
