#include "py_MMVII.h"

#include "MMVII_PCSens.h"

using namespace pybind11::literals;

using namespace MMVII;

static void pyb_init_MapPProj2Im(py::module_ &m)
{
    typedef cMapPProj2Im tMPP2I;
    typedef tMPP2I::tPt tPt;

    py::class_<cMapPProj2Im, cInvertMappingFromElem<cHomot2D<tREAL8> > >(m, "MapPProj2Im", DOC(MMVII_cMapPProj2Im))
            .def(py::init([](){return new tMPP2I(1,tPt(0,0));}))
            .def(py::init<tREAL8,const cPt2dr&>())
            .def("__copy__",  [](const tMPP2I &self) {
                   return tMPP2I(self);
               })
            .def("mapInverse",[](const tMPP2I& c){return std::unique_ptr<cMapIm2PProj>(new cMapIm2PProj(c.Map().MapInverse()));}) //lambda necessary, cMapIm2PProj is neither movable nor copyable

            .def_property("f",[](const tMPP2I& c){return c.F();},[](tMPP2I& m, tREAL8 f){ m.F() = f;})
            .def_property("pp",[](const tMPP2I& c){return c.PP();},[](tMPP2I& m, const tPt& pp){ m.PP() = pp;})

            .def("__repr__",
                 [](const tMPP2I &m) {
                   std::ostringstream ss;
                   ss.precision(15);
                   ss << "MapPProj2Im(" << m.F() << ",(" << m.PP().x() << "," << m.PP().y() << "))";
                   return ss.str();
             })
            ;
}

static void pyb_init_MapIm2PProj(py::module_ &m)
{
    typedef cMapIm2PProj tMI2PP;

    py::class_<cMapIm2PProj, cInvertMappingFromElem<cHomot2D<tREAL8> > >(m, "MapIm2PProj", DOC(MMVII_cMapIm2PProj))
            .def(py::init<const cHomot2D<tREAL8>&>(), "aH"_a)

            .def("__repr__",
                 [](const tMI2PP &m) {
                   std::ostringstream ss;
                   ss.precision(15);
                   ss << "MapIm2PProj()";
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
            .def_static("fromFile", &tPCIC::FromFile,"filename"_a,"remanent"_a = true,py::return_value_policy::reference, DOC(MMVII_cPerspCamIntrCalib, FromFile))
            .def_static("prefixName", &tPCIC::PrefixName, DOC(MMVII_cPerspCamIntrCalib, PrefixName))

            .def_property_readonly("degDir", &tPCIC::DegDir, DOC(MMVII_cPerspCamIntrCalib, DegDir))
            .def_property_readonly("f", &tPCIC::F, DOC(MMVII_cPerspCamIntrCalib, F))
            .def_property_readonly("pp", &tPCIC::PP, DOC(MMVII_cPerspCamIntrCalib, PP))
            .def_property_readonly("name", &tPCIC::Name,DOC(MMVII_cPerspCamIntrCalib, Name))

            .def("values", [](const tPCIC &c, const tVecIn &vi) {tVecOut vo; return c.Values(vo, vi);},"pt3dr_list"_a, DOC(MMVII_cPerspCamIntrCalib, Values))
            .def("dirBundles", [](const tPCIC &c, const tVecOut &vo) {tVecIn vi; return c.DirBundles(vi, vo);},"pt2dr_list"_a, DOC(MMVII_cPerspCamIntrCalib, DirBundles))
            .def("dirBundle", &tPCIC::DirBundle,"ptr2dr"_a, DOC(MMVII_cPerspCamIntrCalib, DirBundle))
            .def("value", &tPCIC::Value,"pt3dr"_a)
            .def("invProjIsDef", &tPCIC::InvProjIsDef,"pt2dr"_a, DOC(MMVII_cPerspCamIntrCalib, InvProjIsDef))

            .def("vParamDist", py::overload_cast<>(&tPCIC::VParamDist, py::const_), DOC(MMVII_cPerspCamIntrCalib, VParamDist))
            .def("setThresholdPhgrAccInv", &tPCIC::SetThresholdPhgrAccInv,"thr"_a, DOC(MMVII_cPerspCamIntrCalib, SetThresholdPhgrAccInv))
            .def("setThresholdPixAccInv", &tPCIC::SetThresholdPixAccInv,"thr"_a, DOC(MMVII_cPerspCamIntrCalib, SetThresholdPixAccInv))

            .def("initRandom", &tPCIC::InitRandom,DOC(MMVII_cPerspCamIntrCalib, InitRandom), DOC(MMVII_cPerspCamIntrCalib, InitRandom))

            .def_property_readonly("szPix", &tPCIC::SzPix,DOC(MMVII_cPerspCamIntrCalib, SzPix))

            .def("degreevisibility", &tPCIC::DegreeVisibility,"pt3dr"_a, DOC(MMVII_cPerspCamIntrCalib, DegreeVisibility))
            .def("degreeVisibilityOnImFrame", &tPCIC::DegreeVisibilityOnImFrame,"pt2dr"_a, DOC(MMVII_cPerspCamIntrCalib, DegreeVisibilityOnImFrame))

            .def("mapPProj2Im", &tPCIC::MapPProj2Im, DOC(MMVII_cDataPerspCamIntrCalib, MapPProj2Im))

            .def("dir_Proj",&tPCIC::Dir_Proj,  py::return_value_policy::reference_internal, DOC(MMVII_cPerspCamIntrCalib,Dir_Proj) )
            .def("dir_Dist",&tPCIC::Dir_Dist,  py::return_value_policy::reference_internal, DOC(MMVII_cPerspCamIntrCalib,Dir_Dist) )
            .def("inv_Proj",&tPCIC::Inv_Proj,  py::return_value_policy::reference_internal, DOC(MMVII_cPerspCamIntrCalib,Inv_Proj) )
            .def("dir_DistInvertible",&tPCIC::Dir_DistInvertible,  py::return_value_policy::reference_internal, DOC(MMVII_cPerspCamIntrCalib,Dir_DistInvertible) )

            .def("infoParam",[](tPCIC &c) {
                cGetAdrInfoParam<tREAL8> aGAIP(".*",c);
                c.GetAdrInfoParam(aGAIP);
                auto names  = aGAIP.VNames();
                auto valptr = aGAIP.VAdrs();
                py::dict d;
                for (unsigned i=0; i<names.size(); i++)
                    d[py::cast(names[i])] = *valptr[i];
                return d;
                },  "Return parameter names and values as dict"
            )

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

            .def("degreeVisibility",&tSCPC::DegreeVisibility,"pt3dr", DOC(MMVII_cSensorCamPC,DegreeVisibility) )
            .def("degreeVisibilityOnImFrame",&tSCPC::DegreeVisibilityOnImFrame,"pt2dr", DOC(MMVII_cSensorCamPC,DegreeVisibilityOnImFrame) )

            .def("ground2ImageAndDepth",&tSCPC::Ground2ImageAndDepth,"pt3dr", DOC(MMVII_cSensorCamPC,Ground2ImageAndDepth) )
            .def("imageAndDepth2Ground",&tSCPC::ImageAndDepth2Ground,"pt3dr", DOC(MMVII_cSensorCamPC,ImageAndDepth2Ground) )

            .def_property_readonly("pose",&tSCPC::Pose,DOC(MMVII_cSensorCamPC,Pose) )
            .def_property_readonly("center",&tSCPC::Center,DOC(MMVII_cSensorCamPC,Center) )
            .def_property_readonly("axeI",&tSCPC::AxeI,DOC(MMVII_cSensorCamPC,AxeI) )
            .def_property_readonly("axeJ",&tSCPC::AxeJ,DOC(MMVII_cSensorCamPC,AxeI) )
            .def_property_readonly("axeK",&tSCPC::AxeK,DOC(MMVII_cSensorCamPC,AxeI) )

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
    pyb_init_MapPProj2Im(m);
    pyb_init_MapIm2PProj(m);
    pyb_init_PerspCamIntrCalib(m);
    pyb_init_SensorCamPC(m);
}

