#include "py_MMVII.h"
#include <pybind11/numpy.h>

#include "MMVII_Geom3D.h"

using namespace std::literals;
using namespace pybind11::literals;

using namespace MMVII;


namespace MMVII {


template<typename T>
bool operator==(const cRotation3D<T>& r1, const cRotation3D<T>& r2)
{
    const auto& m1 = r1.Mat();
    const auto& m2 = r2.Mat();
    if (m1.Sz() != m2.Sz())
        return false;
    for (int y=0; y<m1.Sz().y(); y++)
        for (int x=0; x<m1.Sz().x(); x++)
            if (m1.GetElem(x,y) != m2.GetElem(x,y))
                return false;
    return true;
}

template<typename T>
bool operator==(const cIsometry3D<T>& i1, const cIsometry3D<T>& i2)
{
    return i1.Tr() == i2.Tr() && i1.Rot() == i2.Rot();
}


} // namespace MMVII

template<typename T>
void pyb_init_cRotation3D_tpl(py::module_ &m, const std::string& name)
{
    typedef cRotation3D<T> tR3D;
    typedef typename tR3D::tPt tPt;

    py::class_<tR3D>(m, name.c_str() , DOC(MMVII_cRotation3D))
            .def(py::init<const cDenseMatrix<T> &,bool>(),"matrix"_a,"refineIt"_a=false,DOC(MMVII_cRotation3D,cRotation3D))
            .def(py::init([](py::array_t<T, py::array::c_style | py::array::forcecast> array, bool refineIt)
                 {
                     if (array.ndim() != 2 || array.shape(0) != 3 || array.shape(1) != 3)
                         throw std::runtime_error("Matrix must be 3x3");
                     cDenseMatrix<T> m(3,3);
                     for (int y=0; y<3; y++)
                         for (int x=0; x<3; x++)
                             m.SetElem(x,y,array.at(y,x));
                     return new tR3D(m,refineIt);
                 })
                 ,"array"_a,"refineIt"_a = false)

            .def_property_readonly("mat",&tR3D::Mat,DOC(MMVII_cRotation3D,Mat))
            .def("array",[](const tR3D& r) {
                 py::array_t<T>  a(std::vector<ptrdiff_t>{r.Mat().Sz().y(), r.Mat().Sz().x()});
                 auto m = a.template mutable_unchecked<2>();

                 for (int y=0; y<r.Mat().Sz().y(); y++)
                     for (int x=0; x<r.Mat().Sz().x(); x++)
                         m(y,x) = r.Mat().GetElem(x,y);
                 return a;
             })

            .def("value",&tR3D::Value,"pt3d"_a,DOC(MMVII_cRotation3D,Value))
            .def("inverse",&tR3D::Inverse,"pt3d"_a,DOC(MMVII_cRotation3D,Inverse))

            .def("mapInverse",&tR3D::MapInverse,DOC(MMVII_cRotation3D,MapInverse))
            .def(py::self * py::self)
            .def_static("identity",&tR3D::Identity,DOC(MMVII_cRotation3D,Identity))

            .def_property_readonly("axeI",&tR3D::AxeI,DOC(MMVII_cRotation3D,AxeI))
            .def_property_readonly("axeJ",&tR3D::AxeJ,DOC(MMVII_cRotation3D,AxeJ))
            .def_property_readonly("axeK",&tR3D::AxeK,DOC(MMVII_cRotation3D,AxeK))

            .def_static("completeRON",py::overload_cast<const tPt&>(&tR3D::CompleteRON),"pt"_a,DOC(MMVII_cRotation3D,CompleteRON))
            .def_static("completeRON",py::overload_cast<const tPt&, const tPt&>(&tR3D::CompleteRON),"p0"_a,"p1"_a,DOC(MMVII_cRotation3D,CompleteRON))

            .def_static("rotFromAxe",&tR3D::RotFromAxe,"axe"_a,"theta"_a,DOC(MMVII_cRotation3D,RotFromAxe))
            .def_static("rotFromAxiator",&tR3D::RotFromAxiator,"axe"_a,DOC(MMVII_cRotation3D,RotFromAxiator))
            .def_static("RandomRot",py::overload_cast<>(&tR3D::RandomRot),DOC(MMVII_cRotation3D,RandomRot))
            .def_static("RandomRot",py::overload_cast<const T&>(&tR3D::RandomRot),"ampl"_a,DOC(MMVII_cRotation3D,RandomRot_2))

            .def("extractAxe",
                 [](tR3D& r) {
                   tPt axe; T theta; r.ExtractAxe(axe,theta); return std::make_pair(axe,theta);
                 },
                 DOC(MMVII_cRotation3D,ExtractAxe))

            .def_static("rotFromWPK",&tR3D::RotFromWPK,"wpk"_a,DOC(MMVII_cRotation3D,RotFromWPK))
            .def("toWPK",&tR3D::ToWPK,DOC(MMVII_cRotation3D,ToWPK))
            .def_static("totFromYPR",&tR3D::RotFromYPR,"wpk"_a,DOC(MMVII_cRotation3D,RotFromYPR))
            .def("toYPR",&tR3D::ToYPR,DOC(MMVII_cRotation3D,ToYPR))

            .def(py::self == py::self)

            .def("__repr__",
                 [name](const tR3D &r) {
                   std::ostringstream ss;
                   ss.precision(15);
                   ss << name << "([";
                   for (int y=0; y<r.Mat().Sz().y(); y++) {
                       if (y > 0)
                           ss << ",";
                       ss << "[";
                       for (int x=0; x<r.Mat().Sz().x(); x++) {
                           if (x > 0)
                               ss << ",";
                           ss << r.Mat().GetElem(x,y);
                       }
                       ss << "]";
                   }
                   ss << "])";
                   return ss.str();
                 })
            ;
    py::implicitly_convertible<py::array, tR3D>();
    py::implicitly_convertible<py::tuple, tR3D>();

}


template<typename T>
void pyb_init_cIsometry3D_tpl(py::module_ &m, const std::string& name)
{

    typedef cIsometry3D<T> tI3D;
    typedef cRotation3D<T> tR3D;
    typedef typename tI3D::tPt tPt;


    py::class_<tI3D>(m, name.c_str(),DOC(MMVII_cIsometry3D))
            .def(py::init<const tPt&, const cRotation3D<T> &>(),"tr"_a,"rot"_a,DOC(MMVII_cIsometry3D,cIsometry3D))

            .def("mapInverse",&tI3D::MapInverse,DOC(MMVII_cIsometry3D,MapInverse))
            .def(py::self * py::self)
            .def_static("identity",&tI3D::Identity,DOC(MMVII_cIsometry3D,Identity))

            .def_property("rot",&tI3D::Rot,[](tI3D& i, const tR3D& r){ i.SetRotation(r);})
            .def_property("tr",py::overload_cast<>(&tI3D::Tr, py::const_),[](tI3D& i, const tPt& p){ i.Tr() = p;})

            .def("value",&tI3D::Value,"pt3d"_a,DOC(MMVII_cIsometry3D,Value))
            .def("inverse",&tI3D::Inverse,"pt3d"_a,DOC(MMVII_cIsometry3D,Inverse))

            .def_static("fromRotAndInOut",&tI3D::FromRotAndInOut,"rot"_a,"ptIn"_a,"ptOut"_a,DOC(MMVII_cIsometry3D,FromRotAndInOut))
            .def_static("fromTriInAndOut",&tI3D::FromTriInAndOut,"kIn"_a,"triIn"_a,"kOut"_a,"triOut"_a,DOC(MMVII_cIsometry3D,FromTriInAndOut))
            .def_static("fromTriOut",&tI3D::FromTriOut,"kOut"_a,"triOut"_a,"direct"_a,DOC(MMVII_cIsometry3D,FromTriOut))

            .def_static("toPlaneZ0",&tI3D::ToPlaneZ0,"kOut"_a,"triOut"_a,"direct"_a,DOC(MMVII_cIsometry3D,ToPlaneZ0))

//            .def("toSimil",&tI3D::ToSimil,DOC(MMVII_cIsometry3D,ToSimil))

            .def(py::self == py::self)

            .def("__repr__",
                 [name](const tI3D &i) {
                   std::ostringstream ss;
                   ss.precision(15);
                   ss << name << "((";
                   auto p=i.Tr();
                   for (int i=0; i<p.TheDim; i++) {
                       if (i > 0)
                           ss << ",";
                       ss << p[i];
                   }
                   ss << "),(";
                   auto m=i.Rot().Mat();
                   for (int y=0; y<m.Sz().y(); y++) {
                       if (y > 0)
                           ss << ",";
                       ss << "(";
                       for (int x=0; x<m.Sz().x(); x++) {
                           if (x > 0)
                               ss << ",";
                           ss << m.GetElem(x,y);
                       }
                       ss << ")";
                   }
                   ss << "))";
                   return ss.str();
                 })
            ;

}

void pyb_init_Geom3D(py::module_ &m)
{
    pyb_init_cRotation3D_tpl<double>(m,"Rotation3D");
    pyb_init_cIsometry3D_tpl<double>(m,"Isometry3D");
}
