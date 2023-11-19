#include "py_MMVII.h"

#include "MMVII_Mappings.h"
#include "MMVII_Geom2D.h"

using namespace pybind11::literals;

using namespace MMVII;

template<typename T, int DimIn, int DimOut>
void pyb_init_cDataMapping_tpl(py::module_ &m,
                               const std::string& name,
                               const std::string& invertibleName = "",
                               const std::string& invertName = ""
                               )
{
    typedef cDataMapping<T,DimIn,DimOut> tDM;
    typedef typename tDM::tVecIn tVecIn;

     py::class_<tDM>(m, name.c_str() , DOC(MMVII_cDataMapping))
            .def("value",&tDM::Value,"pt"_a, DOC(MMVII_cDataMapping,Value))
            .def("values",py::overload_cast<const tVecIn &>(&tDM::Values,py::const_),"pt_list"_a, DOC(MMVII_cDataMapping,Values))
            ;

    if (DimIn != DimOut)
        return;

    // Can't make tDIM inherits from ancestor tDM: should bind intermediate class
    typedef cDataInvertibleMapping<T,DimIn> tDIM;
    py::class_<tDIM>(m, invertibleName.c_str() , DOC(MMVII_cDataInvertibleMapping))
            .def("value",&tDIM::Value,"pt"_a, DOC(MMVII_cDataMapping,Value))
            .def("values",py::overload_cast<const tVecIn &>(&tDIM::Values,py::const_),"pt_list"_a, DOC(MMVII_cDataMapping,Values))
            .def("inverse",&tDIM::Inverse,"pt"_a, DOC(MMVII_cDataInvertibleMapping,Inverse))
            .def("inverses",py::overload_cast<const tVecIn &>(&tDIM::Inverses,py::const_),"pt_list"_a, DOC(MMVII_cDataInvertibleMapping,Inverses))
            ;

    typedef cDataInvertOfMapping<T,DimIn> tDIOM;
    py::class_<tDIOM,tDIM>(m, invertName.c_str() , DOC(MMVII_cDataInvertOfMapping))
            .def(py::init([](const tDIM * dim) { return new tDIOM(dim,false);}))
            ;
}


template<class cMapElem>
void pyb_init_cInvertMappingFromElem_tpl(py::module_ &m,
                               const std::string& name
                               )
{
    static constexpr int     Dim=cMapElem::TheDim;
    typedef cInvertMappingFromElem<cMapElem> tIDMFE;
    //typedef typename tIDMFE::tVecIn tVecIn;
    typedef typename  cMapElem::tTypeElem  tTypeElem;
    //typedef cMapElem                       tMap;
    //typedef typename cMapElem::tTypeMapInv tMapInv;
    typedef cDataInvertibleMapping<tTypeElem,Dim>  tDataIMap;
    //typedef typename  tDataIMap::tPt             tPt;
    typedef typename  tDataIMap::tVecPt          tVecPt;

     py::class_<tIDMFE>(m, name.c_str() , DOC(MMVII_cInvertMappingFromElem))
            .def("value",&tIDMFE::Value,"pt"_a, DOC(MMVII_cInvertMappingFromElem,Value))
            .def("values", [](const tIDMFE &c, const tVecPt &vi) {tVecPt vo; return c.Values(vo, vi);},"pt_list"_a, DOC(MMVII_cInvertMappingFromElem, Values))
            .def("inverse",&tIDMFE::Inverse,"pt"_a, DOC(MMVII_cInvertMappingFromElem,Inverse))
            .def("inverses",[](const tIDMFE &c, const tVecPt &vi) {tVecPt vo; return c.Inverses(vo, vi);},"pt_list"_a, DOC(MMVII_cInvertMappingFromElem, Inverses))
            ;
}

void pyb_init_DataMappings(py::module_ &m)
{
    pyb_init_cDataMapping_tpl<tREAL8,2,2>(m,"DataMapping2D","DataInvertibleMapping2D","DataInvertOfMapping2D");
    pyb_init_cDataMapping_tpl<tREAL8,3,2>(m,"DataMapping3Dto2D");
    pyb_init_cDataMapping_tpl<tREAL8,2,3>(m,"DataMapping2Dto3D");
    pyb_init_cInvertMappingFromElem_tpl<cHomot2D<tREAL8> >(m,"InvertMappingFromElemHomol");
}
