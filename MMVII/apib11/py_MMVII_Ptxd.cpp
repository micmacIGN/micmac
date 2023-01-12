#include "py_MMVII.h"
#include "pybind11/numpy.h"

#include "MMVII_Ptxd.h"


using namespace MMVII;


template<typename T, int Dim>
void pyb_init_cPtxd_tpl(py::module_ &m, const std::string& name) {
    using namespace std::literals;
    using namespace pybind11::literals;

    typedef cPtxd<T,Dim> tPt;

    auto cd = py::class_<tPt>(m, name.c_str());

    if constexpr (Dim == 1)
            cd.def(py::init<T>(),"x"_a);
    if constexpr (Dim == 2)
            cd.def(py::init<T , T>(),"x"_a,"y"_a);
    if constexpr (Dim == 3)
            cd.def(py::init<T, T, T>(),"x"_a,"y"_a,"z"_a);
    if constexpr (Dim == 4)
            cd.def(py::init<T, T, T, T>(),"x"_a,"y"_a,"z"_a,"t"_a);

    cd.def(py::init([](py::tuple t) {
            if (t.size() != Dim)
               throw py::index_error();
            auto pt = new tPt;
            for (int i=0; i<Dim; i++)
               (*pt)[i] = t[i].cast<T>();
            return pt;
    }),"tuple"_a);
    cd.def(py::init([](py::array_t<T, py::array::c_style | py::array::forcecast> array)
         {
             if (array.ndim() != 1 || array.shape(0) != Dim)
                 throw std::runtime_error("array has bad shape");
             tPt *p = new tPt;
             for (int i=0; i<Dim; i++)
               (*p)[i] = array.at(i);
             return p;
         })
         ,"array"_a);


    cd.def_static("pCste",&tPt::PCste)
            .def_static("pRand",&tPt::PRand)
            .def_static("pRandC",&tPt::PRandC)
            .def_static("pRandUnit",&tPt::PRandUnit)
            .def_static("pRandInSphere",&tPt::PRandInSphere)
            .def_static("fromPtInt",&tPt::FromPtInt,"pt")
            .def_static("fromPtR",&tPt::FromPtR,"pt")

            .def("__repr__",[name](const tPt& p) {
                  std::ostringstream ss;
    
                  ss.precision(15);
                  ss << name << "(";
                  for (int i=0; i<Dim; i++) {
                      if (i > 0)
                          ss << ",";
                      ss << p[i];
                  }
                  ss << ')';
                  return ss.str();
             })

            .def("__getitem__",
                       [](const tPt &pt, size_t i) {
                           if (i >= Dim)
                               throw py::index_error();
                           return pt[i];
                       })
            .def("__setitem__",
                       [](tPt& pt, size_t i, T v) {
                           if (i >=Dim)
                               throw py::index_error();
                           pt[i] = v;
                       })
            .def("__len__", [](const tPt& pt){ return Dim;})
            .def("__iter__",
                      [](const tPt& pt) { return py::make_iterator(pt.PtRawData(),pt.PtRawData()+Dim); },
                      py::keep_alive<0, 1>() /* Essential: keep object alive while iterator exists */)
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

    if constexpr (Dim >= 1)
            cd.def_property("x",[](const tPt& p){return p.x();},[](tPt& p, T x){ p.x() = x;});
    if constexpr (Dim >= 2)
            cd.def_property("y",[](const tPt& p){return p.y();},[](tPt& p, T y){ p.y() = y;});
    if constexpr (Dim >= 3)
            cd.def_property("z",[](const tPt& p){return p.z();},[](tPt& p, T z){ p.z() = z;});
    if constexpr (Dim >= 4)
            cd.def_property("t",[](const tPt& p){return p.t();},[](tPt& p, T t){ p.t() = t;});

    m.def("supEq",static_cast<bool (*)(const tPt&, const tPt&)>(&SupEq),"pt1"_a,"pt2"_a);
    m.def("infStr",static_cast<bool (*)(const tPt&, const tPt&)>(&InfStr),"pt1"_a,"pt2"_a);
    m.def("infEq",static_cast<bool (*)(const tPt&, const tPt&)>(&InfEq),"pt1"_a,"pt2"_a);

    m.def("ptSupEq",static_cast<tPt (*)(const tPt&, const tPt&)>(&PtSupEq),"pt1"_a,"pt2"_a);
    m.def("ptInfEq",static_cast<tPt (*)(const tPt&, const tPt&)>(&PtInfEq),"pt1"_a,"pt2"_a);
    m.def("ptInfStr",static_cast<tPt (*)(const tPt&, const tPt&)>(&PtInfStr),"pt1"_a,"pt2"_a);


    m.def("normK",&NormK<T,Dim>,"pt"_a,"exp"_a);
    m.def("norm1",&Norm1<T,Dim>,"pt"_a);
    m.def("normInf",&NormInf<T,Dim>,"pt"_a);
    m.def("norm2",&Norm2<T,Dim>,"pt"_a);

    py::implicitly_convertible<py::tuple, tPt>();
    py::implicitly_convertible<py::array, tPt>();
}

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
            .def_static("boxCste",&tBox::BoxCste,"val"_a)
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
    pyb_init_cPtxd_tpl<int,1>(m,"Pt1di");
    pyb_init_cPtxd_tpl<int,2>(m,"Pt2di");
    pyb_init_cPtxd_tpl<int,3>(m,"Pt3di");
    pyb_init_cPtxd_tpl<float,1>(m,"Pt1df");
    pyb_init_cPtxd_tpl<float,2>(m,"Pt2df");
    pyb_init_cPtxd_tpl<float,3>(m,"Pt3df");
    pyb_init_cPtxd_tpl<double,1>(m,"Pt1dr");
    pyb_init_cPtxd_tpl<double,2>(m,"Pt2dr");
    pyb_init_cPtxd_tpl<double,3>(m,"Pt3dr");

    pyb_init_cTplBox_tpl<int,1>(m,"Box1di");
    pyb_init_cTplBox_tpl<int,2>(m,"Box2di");
    pyb_init_cTplBox_tpl<int,3>(m,"Box3di");
    pyb_init_cTplBox_tpl<double,1>(m,"Box1dr");
    pyb_init_cTplBox_tpl<double,2>(m,"Box2dr");
    pyb_init_cTplBox_tpl<double,3>(m,"Box3dr");

}
