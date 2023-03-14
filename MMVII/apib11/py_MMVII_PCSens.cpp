#include "py_MMVII.h"

#include "MMVII_PCSens.h"

using namespace pybind11::literals;

using namespace MMVII;

static void pyb_init_CalibStenPerfect(py::module_ &m)
{
    typedef cCalibStenPerfect tCSP;
    typedef tCSP::tScal tScal;
    typedef tCSP::tPt tPt;
    typedef tCSP::tVecPt tVecPt;

    py::class_<cCalibStenPerfect>(m, "CalibStenPerfect", DOC(MMVII_cCalibStenPerfect))
            .def(py::init([](){return new tCSP(1,tPt(0,0));}))
            .def(py::init<tScal,const cPt2dr&>())
            .def("__copy__",  [](const tCSP &self) {
                   return tCSP(self);
               })
            .def("mapInverse", &tCSP::MapInverse)


            .def("value", &tCSP::Value,"pt2dr"_a)
            .def("inverse", &tCSP::Inverse,"pt2dr"_a)
            .def("values", [](const tCSP &c, const tVecPt &vi) {tVecPt vo; return c.Values(vo, vi);},"pt2dr_list"_a)
            .def("inverses", [](const tCSP &c, const tVecPt &vi) {tVecPt vo; return c.Inverses(vo, vi);},"pt2dr_list"_a)

            .def_property("f",[](const tCSP& c){return c.F();},[](cCalibStenPerfect& c, tScal f){ c.F() = f;})
            .def_property("pp",[](const tCSP& c){return c.PP();},[](cCalibStenPerfect& c, const tPt& pp){ c.PP() = pp;})

            .def("__repr__",
                 [](const tCSP &c) {
                   std::ostringstream ss;
                   ss.precision(15);
                   ss << "CalibStenPerfect(" << c.F() << ",(" << c.PP().x() << "," << c.PP().y() << "))";
                   return ss.str();
             })
            ;
}


static void pyb_init_PerspCamIntrCalib(py::module_ &m)
{
    typedef cPerspCamIntrCalib tPCIC;
//    typedef tPCIC::tScal tScal;
//    typedef tPCIC::tPtOut tPtOut;
//    typedef tPCIC::tPtIn tPtIn;
    typedef tPCIC::tVecIn tVecIn;
    typedef tPCIC::tVecOut tVecOut;

    py::class_<tPCIC>(m, "PerspCamIntrCalib", DOC(MMVII_cPerspCamIntrCalib))
            .def("toFile", &tPCIC::ToFile,"filename"_a,DOC(MMVII_cPerspCamIntrCalib, ToFile))
            .def_static("fromFile", &tPCIC::FromFile,"filename"_a,py::return_value_policy::reference, DOC(MMVII_cPerspCamIntrCalib, FromFile))
            .def_static("prefixName", &tPCIC::PrefixName, DOC(MMVII_cPerspCamIntrCalib, PrefixName))

            .def_property_readonly("degDir", &tPCIC::DegDir, DOC(MMVII_cPerspCamIntrCalib, DegDir))
            .def_property_readonly("f", &tPCIC::F, DOC(MMVII_cPerspCamIntrCalib, F))
            .def_property_readonly("pp", &tPCIC::PP, DOC(MMVII_cPerspCamIntrCalib, PP))
            .def_property_readonly("name", &tPCIC::Name,DOC(MMVII_cPerspCamIntrCalib, Name))

            .def("values", [](const tPCIC &c, const tVecIn &vi) {tVecOut vo; return c.Values(vo, vi);},"pt3dr_list"_a, DOC(MMVII_cPerspCamIntrCalib, Values))
            .def("inverses", [](const tPCIC &c, const tVecOut &vo) {tVecIn vi; return c.Inverses(vi, vo);},"pt2dr_list"_a, DOC(MMVII_cPerspCamIntrCalib, Inverses))
            .def("inverse", &tPCIC::Inverse,"ptr2dr"_a, DOC(MMVII_cPerspCamIntrCalib, Inverse))
            .def("value", &tPCIC::Value,"pt3dr"_a)
            .def("invProjIsDef", &tPCIC::InvProjIsDef,"pt2dr"_a, DOC(MMVII_cPerspCamIntrCalib, InvProjIsDef))

            .def("vParamDist", py::overload_cast<>(&tPCIC::VParamDist, py::const_), DOC(MMVII_cPerspCamIntrCalib, VParamDist))
            .def("setThresholdPhgrAccInv", &tPCIC::SetThresholdPhgrAccInv,"thr"_a, DOC(MMVII_cPerspCamIntrCalib, SetThresholdPhgrAccInv))
            .def("setThresholdPixAccInv", &tPCIC::SetThresholdPixAccInv,"thr"_a, DOC(MMVII_cPerspCamIntrCalib, SetThresholdPixAccInv))

            .def("initRandom", &tPCIC::InitRandom,DOC(MMVII_cPerspCamIntrCalib, InitRandom), DOC(MMVII_cPerspCamIntrCalib, InitRandom))

            .def_property_readonly("szPix", &tPCIC::SzPix,DOC(MMVII_cPerspCamIntrCalib, SzPix))

            .def("visibility", &tPCIC::Visibility,"pt3dr"_a, DOC(MMVII_cPerspCamIntrCalib, Visibility))
            .def("visibilityOnImFrame", &tPCIC::VisibilityOnImFrame,"pt2dr"_a, DOC(MMVII_cPerspCamIntrCalib, VisibilityOnImFrame))

            .def("vecInfo", &tPCIC::VecInfo, DOC(MMVII_cDataPerspCamIntrCalib, VecInfo))

            .def("calibStenPerfect", &tPCIC::CalibStenPerfect, py::return_value_policy::reference_internal, DOC(MMVII_cDataPerspCamIntrCalib, CalibStenPerfect))
            .def("dir_Proj",&tPCIC::Dir_Proj,  py::return_value_policy::reference_internal, DOC(MMVII_cPerspCamIntrCalib,Dir_Proj) )
            .def("dir_Dist",&tPCIC::Dir_Dist,  py::return_value_policy::reference_internal, DOC(MMVII_cPerspCamIntrCalib,Dir_Dist) )
            .def("inv_Proj",&tPCIC::Inv_Proj,  py::return_value_policy::reference_internal, DOC(MMVII_cPerspCamIntrCalib,Inv_Proj) )
            .def("dir_DistInvertible",&tPCIC::Dir_DistInvertible,  py::return_value_policy::reference_internal, DOC(MMVII_cPerspCamIntrCalib,Dir_DistInvertible) )


            .def("__repr__",[](const tPCIC&){return "MMVII.PerspCamIntrCalib";})

            ;
}

static void pyb_init_SensorCamPC(py::module_ &m)
{
    typedef cSensorCamPC tSCPC;
    typedef tSCPC::tPose tPose;

    py::class_<tSCPC>(m, "SensorCamPC", DOC(MMVII_cSensorCamPC))
            .def(py::init<const std::string&, const tPose&, cPerspCamIntrCalib*>())

            .def("ground2Image",&tSCPC::Ground2Image,"pt3dr", DOC(MMVII_cSensorCamPC,Ground2Image) )

            .def("visibility",&tSCPC::Visibility,"pt3dr", DOC(MMVII_cSensorCamPC,Visibility) )
            .def("visibilityOnImFrame",&tSCPC::VisibilityOnImFrame,"pt2dr", DOC(MMVII_cSensorCamPC,VisibilityOnImFrame) )

            .def("ground2ImageAndDepth",&tSCPC::Ground2ImageAndDepth,"pt3dr", DOC(MMVII_cSensorCamPC,Ground2ImageAndDepth) )
            .def("imageAndDepth2Ground",&tSCPC::ImageAndDepth2Ground,"pt3dr", DOC(MMVII_cSensorCamPC,ImageAndDepth2Ground) )

            .def_property_readonly("pose",&tSCPC::Pose,DOC(MMVII_cSensorCamPC,Pose) )
            .def_property_readonly("center",&tSCPC::Center,DOC(MMVII_cSensorCamPC,Center) )
            .def_property_readonly("axeI",&tSCPC::AxeI,DOC(MMVII_cSensorCamPC,AxeI) )
            .def_property_readonly("axeJ",&tSCPC::AxeI,DOC(MMVII_cSensorCamPC,AxeI) )
            .def_property_readonly("axeK",&tSCPC::AxeI,DOC(MMVII_cSensorCamPC,AxeI) )

            .def_property_readonly("internalCalib",&tSCPC::InternalCalib,py::return_value_policy::reference_internal ,DOC(MMVII_cSensorCamPC,InternalCalib) )

            .def("omega",&tSCPC::Omega,DOC(MMVII_cSensorCamPC,Omega) )

            .def("toFile",&tSCPC::ToFile,"filename"_a,DOC(MMVII_cSensorCamPC,ToFile) )
            .def_static("fromFile",&tSCPC::FromFile,"filename"_a,DOC(MMVII_cSensorCamPC,FromFile) )
            .def_static("nameOri_From_Image",&tSCPC::NameOri_From_Image,"imagename"_a,DOC(MMVII_cSensorCamPC,NameOri_From_Image) )

            .def_property_readonly("szPix",&tSCPC::SzPix,DOC(MMVII_cSensorCamPC,SzPix) )

            .def_static("prefixName",&tSCPC::PrefixName,DOC(MMVII_cSensorCamPC,PrefixName) )

            .def("__repr__",[](const tSCPC&){return "MMVII.SensorCamPC";})

            ;
}



void pyb_init_PCSens(py::module_ &m)
{
    pyb_init_CalibStenPerfect(m);
    pyb_init_PerspCamIntrCalib(m);
    pyb_init_SensorCamPC(m);
}
