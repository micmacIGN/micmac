#include "py_MMVII.h"

#include "MMVII_Geom2D.h"


using namespace MMVII;


template<typename T>
void pyb_init_Matrix_tpl(py::module_ &m, const std::string& name) {
    using namespace std::literals;
    using namespace pybind11::literals;

    typedef cDenseMatrix<T> cDM;

    auto dm=py::class_<cDM>(m, name.c_str(),py::buffer_protocol());
            dm.def(py::init<int,int,eModeInitImage>(),"x"_a,"y"_a,"modeInitImage"_a = eModeInitImage::eMIA_NoInit);
            dm.def(py::init<int,eModeInitImage>(),"x"_a,"modeInitImage"_a = eModeInitImage::eMIA_NoInit);
            dm.def(py::init<typename cDM::tIm>(),"im2d"_a);

            dm.def("dup",&cDM::Dup);
            dm.def_static("identity",&cDM::Identity,"size"_a);
//            .def_static("Diag",&cDM::Diag)
            dm.def("closestOrthog",&cDM::ClosestOrthog);

            
            dm.def("sz",&cDM::Sz);
            dm.def("show",&cDM::Show);
            
            dm.def("setElem",&cDM::SetElem,"x"_a,"y"_a,"val"_a);
            dm.def("addElem",&cDM::AddElem,"x"_a,"y"_a,"val"_a);

            dm.def("getElem",py::overload_cast<int,int>(&cDM::GetElem, py::const_),"x"_a,"y"_a);
            dm.def("getElem",py::overload_cast<const cPt2di &>(&cDM::GetElem, py::const_),"pt2di"_a);

            // TODO : a verifier le stride/offset !
            dm.def_buffer([](cDM &m) -> py::buffer_info {
                    return py::buffer_info(
                        *m.DIm().ExtractRawData2D(),                               /* Pointer to buffer */
                        sizeof(T),                          /* Size of one scalar */
                        py::format_descriptor<T>::format(), /* Python struct-style format descriptor */
                        2,                                      /* Number of dimensions */
                        { m.Sz().x(), m.Sz().y() },                 /* Buffer dimensions */
                        { sizeof(T) * m.Sz().x(),             /* Strides (in bytes) for each index */
                          sizeof(T) }
                    );})
            ;
            
}

void pyb_init_DenseMatrix(py::module_ &m)
{
    py::enum_<eModeInitImage>(m,"ModeInitImage")
            .value("eMIA_Rand", eModeInitImage::eMIA_Rand)
            .value("eMIA_RandCenter", eModeInitImage::eMIA_RandCenter)
            .value("eMIA_Null", eModeInitImage::eMIA_Null)
            .value("eMIA_V1", eModeInitImage::eMIA_V1)
            .value("eMIA_MatrixId", eModeInitImage::eMIA_MatrixId)
            .value("eMIA_NoInit", eModeInitImage::eMIA_NoInit)
            ;
    
    pyb_init_Matrix_tpl<double>(m,"Matrixd");
    pyb_init_Matrix_tpl<float>(m,"Matrixf");
}
