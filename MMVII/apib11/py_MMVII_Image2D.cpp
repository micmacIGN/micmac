#include "py_MMVII.h"

#include "MMVII_Image2D.h"


using namespace MMVII;


template<typename T>
void pyb_init_cIm2D_tpl(py::module_ &m, const std::string& name) {
    using namespace std::literals;
    using namespace pybind11::literals;

    typedef cIm2D<T> tIm;
    typedef cDataIm2D<T> tDIm;

    auto imd = py::class_<tDIm>(m, ("Data" + name).c_str(),py::buffer_protocol());
    imd.def("sz",&tDIm::Sz);
    imd.def("p0",&tDIm::P0);
    imd.def("p1",&tDIm::P1);
    imd.def("nbPix",&tDIm::NbPix);
    imd.def("toFile",py::overload_cast<const std::string&>(&tDIm::ToFile,py::const_),"fileName"_a);
    imd.def("toFile",py::overload_cast<const std::string&,eTyNums>(&tDIm::ToFile,py::const_),"fileName"_a,"type"_a);
    imd.def("toFile",py::overload_cast<const std::string&,const tDIm&, const tDIm&>(&tDIm::ToFile,py::const_),"fileName"_a,"imG"_a,"imB"_a);
    imd.def("clipToFile",py::overload_cast<const std::string&,const cRect2&>(&tDIm::ClipToFile,py::const_),"fileName"_a,"rect"_a);
    // TODO : a verifier le stride/offset !
    imd.def_buffer([](tDIm &i) -> py::buffer_info {
            return py::buffer_info(
                *i.ExtractRawData2D(),                               /* Pointer to buffer */
                sizeof(T),                          /* Size of one scalar */
                py::format_descriptor<T>::format(), /* Python struct-style format descriptor */
                2,                                      /* Number of dimensions */
                { i.Sz().y(), i.Sz().x() },                 /* Buffer dimensions */
                { sizeof(T) * i.Sz().x(),             /* Strides (in bytes) for each index */
                  sizeof(T) }
            );})
    ;


    auto im = py::class_<tIm>(m, name.c_str(),py::buffer_protocol());
    im.def(py::init(
               [](const cPt2di& p0, const cPt2di& p1, eModeInitImage initImage)
               {
                   return new tIm(p0,p1,nullptr,initImage);
               }
       ),"p0"_a,"p1"_a,"init"_a = eModeInitImage::eMIA_NoInit);
    im.def(py::init(
               [](const cPt2di& size, eModeInitImage initImage=eModeInitImage::eMIA_NoInit)
               {
                   return new tIm(size,nullptr,initImage);
               }
       ),"size"_a,"init"_a  = eModeInitImage::eMIA_NoInit);
     im.def(py::init<const cPt2di&, const cDataFileIm2D>(),"size"_a,"dataFileIm2d"_a);

     // Accept a numpy array. Don't know how to avoid a copy: should increment reference on array.data. How ?
     im.def(py::init([](py::buffer b) {
            py::buffer_info info = b.request();
            if (info.format != py::format_descriptor<T>::format())
                throw std::runtime_error("Incompatible format");
            if (info.ndim != 2)
                throw std::runtime_error("Incompatible dimension");
            if (info.strides[1] != (long)sizeof (T) || info.strides[0] != (long)sizeof(T) * info.shape[1])
                throw std::runtime_error("Incompatible memory layout");
            auto im = new tIm(cPt2di(info.shape[1],info.shape[0]));
            memcpy(*im->DIm().ExtractRawData2D(),info.ptr, sizeof(T) * info.shape[1] * info.shape[0]);
            return im;
// Following works until numpy array is deleted ...
//            return new tIm(cPt2di(info.shape[1],info.shape[0]), static_cast<T *>(info.ptr),eModeInitImage::eMIA_NoInit);
        }));

     im.def("dup",&tIm::Dup);
     im.def("dImCopy",py::overload_cast<>(&tIm::DIm,py::const_));
     im.def("dIm",py::overload_cast<>(&tIm::DIm),py::return_value_policy::reference_internal);
     im.def("dup",&tIm::Dup);
     im.def("transpose",&tIm::Transpose);

     im.def("sz",[](const tIm &im) {return im.DIm().Sz();});
     im.def("nbPix",[](const tIm &im) {return im.DIm().NbPix();});
     im.def("toFile",[](const tIm &im, const std::string& fileName) {im.DIm().ToFile(fileName);},"fileName"_a);
     im.def("toFile",[](const tIm &im, const std::string& fileName, eTyNums tyNum) {im.DIm().ToFile(fileName,tyNum);},"fileName"_a,"type"_a);
     im.def("clipToFile",[](const tIm &im, const std::string& fileName,const cRect2& r) {im.DIm().ClipToFile(fileName,r);},"fileName"_a,"rect"_a);

     im.def_static("fromFile",py::overload_cast<const std::string&>(&tIm::FromFile),"fileName"_a);
     im.def_static("fromFile",py::overload_cast<const std::string&, const cBox2di&>(&tIm::FromFile),"fileName"_a,"box"_a);

     im.def("__getitem__",
           [](const tIm &im, std::pair<py::ssize_t, py::ssize_t> i) {
               if (i.first >= im.DIm().SzY() || i.second >= im.DIm().SzX()) {
                   throw py::index_error();
               }
               return im.DIm().GetV(cPt2di(i.second, i.first));
           });
      im.def("__setitem__",
           [](tIm& im, std::pair<py::ssize_t, py::ssize_t> i, T v) {
               if (i.first >= im.DIm().SzY() || i.second >= im.DIm().SzX()) {
                   throw py::index_error();
               }
               im.DIm().SetV(cPt2di(i.second, i.first), v);
           });

     im.def_buffer([](tIm &i) -> py::buffer_info {
             return py::buffer_info(
                 *i.DIm().ExtractRawData2D(),                               /* Pointer to buffer */
                 sizeof(T),                          /* Size of one scalar */
                 py::format_descriptor<T>::format(), /* Python struct-style format descriptor */
                 2,                                      /* Number of dimensions */
                 { i.DIm().Sz().y(), i.DIm().Sz().x() },                 /* Buffer dimensions */
                 { sizeof(T) * i.DIm().Sz().x(),             /* Strides (in bytes) for each index */
                   sizeof(T) }
             );});


}       

void pyb_init_Image2D(py::module_ &m)
{
    pyb_init_cIm2D_tpl<float>(m,"Im2Df");
    pyb_init_cIm2D_tpl<double>(m,"Im2Dr");
    pyb_init_cIm2D_tpl<int>(m,"Im2Di");
    pyb_init_cIm2D_tpl<unsigned char>(m,"Im2Duc");
}
