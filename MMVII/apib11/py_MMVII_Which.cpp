#include "py_MMVII.h"
#include "pybind11/numpy.h"

#include "MMVII_nums.h"
#include "MMVII_Geom3D.h"

using namespace MMVII;


template<typename T, typename Dim>
void pyb_init_cWhichMin_tpl(py::module_ &m, const std::string& name) {

    using namespace std::literals;
    using namespace pybind11::literals;

    typedef cWhichMin<T,Dim> tWhichMin;
    //typedef  cWhichExtrem<T,Dim,true> tExtrem;
    typedef cWhichExtrem<T,Dim,true> tWhichExtrem;


    auto tb = py::class_<tWhichMin,tWhichExtrem>(m, name.c_str())
            .def(py::init<const T&, const Dim&>(),"Index"_a,"Val"_a)
	    ;

}

template<typename T, typename Dim,const bool IsMin>
void pyb_init_cWhichExtrem_tpl(py::module_ &m, const std::string& name) {

    using namespace std::literals;
    using namespace pybind11::literals;

    typedef cWhichExtrem<T,Dim,IsMin> tWhichExtrem;

    auto tb = py::class_<tWhichExtrem>(m, name.c_str())
            .def(py::init<const T&, const Dim&>(),"Index"_a,"Val"_a)

            .def("indexExtre", &tWhichExtrem::IndexExtre, DOC(MMVII_cWhichExtrem,IndexExtre))
            ;

}



void pyb_init_cWhich(py::module_ &m)
{
    pyb_init_cWhichExtrem_tpl<cIsometry3D<tREAL8>,tREAL8,true>(m,"WhichExtrem_tPose_tREAL8_true");
    pyb_init_cWhichExtrem_tpl<cIsometry3D<tREAL8>,tREAL8,false>(m,"WhichExtrem_tPose_tREAL8_false");
    pyb_init_cWhichMin_tpl<cIsometry3D<tREAL8>,tREAL8>(m,"WhichMin_tPose_tREAL8");
}

