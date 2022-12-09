#include "py_MMVII.h"

#include "MMVII_Geom2D.h"


using namespace MMVII;


template<typename T, int Dim>
void pyb_init_Geom_tpl(py::module_ &m, const std::string& name) {
    using namespace std::literals;
    using namespace pybind11::literals;

    typedef cPtxd<T,Dim> cPt;

    auto cd = py::class_<cPt>(m, name.c_str())
            .def_static("pCste",&cPt::PCste)
            .def_static("pRand",&cPt::PRand)
            .def_static("pRandC",&cPt::PRandC)
            .def_static("pRandUnit",&cPt::PRandUnit)
            .def_static("pRandInSphere",&cPt::PRandInSphere)
            .def_static("fromPtInt",&cPt::FromPtInt,"pt")
            .def_static("fromPtR",&cPt::FromPtR,"pt")

//            .def("toStdVector",&cPt::ToStdVector) // toStdVector undefined symbol au link ...

            .def("__repr__",[name](const cPt& p) {
                   std::ostringstream ss;

                   ss.precision(17);
                   ss << name << "(";
                   if constexpr (Dim >= 1)
                           ss << p.x();
                   if constexpr (Dim >= 2)
                           ss << "," << p.y();
                   if constexpr (Dim >= 3)
                           ss << "," << p.z();
                   if constexpr (Dim >= 4)
                           ss << "," << p.t();
                   ss << ')';
                   return ss.str();
             })

            .def(py::self + py::self)
            .def(py::self - py::self)
            .def(- py::self)
            .def(py::self += py::self)
            .def(py::self == py::self)
            .def(py::self != py::self)

            .def(py::self * T())
            .def(T() * py::self)
            .def(py::self / T())
    ;
    if constexpr (Dim == 1)
            cd.def(py::init<T>(),"x"_a);
    if constexpr (Dim == 2)
            cd.def(py::init<T , T>(),"x"_a,"y"_a);
    if constexpr (Dim == 3)
            cd.def(py::init<T, T, T>(),"x"_a,"y"_a,"z"_a);
    if constexpr (Dim == 4)
            cd.def(py::init<T, T, T, T>(),"x"_a,"y"_a,"z"_a,"t"_a);

    if constexpr (Dim >= 1)
            cd.def_property("x",[](const cPt& p){return p.x();},[](cPt& p, T x){ p.x() = x;});
    if constexpr (Dim >= 2)
            cd.def_property("y",[](const cPt& p){return p.y();},[](cPt& p, T y){ p.y() = y;});
    if constexpr (Dim >= 3)
            cd.def_property("z",[](const cPt& p){return p.z();},[](cPt& p, T z){ p.z() = z;});
    if constexpr (Dim >= 4)
            cd.def_property("t",[](const cPt& p){return p.t();},[](cPt& p, T t){ p.t() = t;});

    m.def("supEq",static_cast<bool (*)(const cPt&, const cPt&)>(&SupEq));
    m.def("infStr",static_cast<bool (*)(const cPt&, const cPt&)>(&InfStr));
    m.def("infEq",static_cast<bool (*)(const cPt&, const cPt&)>(&InfEq));

    m.def("ptSupEq",static_cast<cPt (*)(const cPt&, const cPt&)>(&PtSupEq));
    m.def("ptInfEq",static_cast<cPt (*)(const cPt&, const cPt&)>(&PtInfEq));
    m.def("ptInfStr",static_cast<cPt (*)(const cPt&, const cPt&)>(&PtInfStr));


    m.def("normK",&NormK<T,Dim>);
    m.def("norm1",&Norm1<T,Dim>);
    m.def("normInf",&NormInf<T,Dim>);
    m.def("norm2",&Norm2<T,Dim>);
}

void pyb_init_Geom(py::module_ &m)
{
    pyb_init_Geom_tpl<double,1>(m,"Pt1dr");
    pyb_init_Geom_tpl<double,2>(m,"Pt2dr");
    pyb_init_Geom_tpl<double,3>(m,"Pt3dr");
    pyb_init_Geom_tpl<int,1>(m,"Pt1di");
    pyb_init_Geom_tpl<int,2>(m,"Pt2di");
    pyb_init_Geom_tpl<int,3>(m,"Pt3di");
}
