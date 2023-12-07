#include "py_MMVII.h"

#include "MMVII_PoseRel.h"
#include "MMVII_PCSens.h"


using namespace MMVII;


void pyb_init_MatEssential(py::module_ &m) {

    using namespace std::literals;
    using namespace pybind11::literals;

	using namespace std::literals;
        using namespace pybind11::literals;

	py::class_<cHomogCpleIm>(m, "HomogCpleIm", DOC(MMVII_cHomogCpleIm))
                .def(py::init<>(),DOC(MMVII_cHomogCpleIm,cHomogCpleIm))
		.def(py::init<const cPt2dr &, const cPt2dr & >(),DOC(MMVII_cHomogCpleIm,cHomogCpleIm))
		.def_readwrite("p1", &cHomogCpleIm::mP1,DOC(MMVII_cHomogCpleIm,mP1))
		.def_readwrite("p2", &cHomogCpleIm::mP2,DOC(MMVII_cHomogCpleIm,mP2))
		.def("Pt", &cHomogCpleIm::Pt,DOC(MMVII_cHomogCpleIm,Pt))

		.def("__repr__",
                 [](const cHomogCpleIm &m) {
                   std::ostringstream ss;
                   ss.precision(8);
                   ss << "P1=[" << m.mP1.x() << " " << m.mP1.y() << "], P2=[" << m.mP2.x() << " " << m.mP2.y() << "]\n";
                       return ss.str();
                 });

	py::class_<cSetHomogCpleIm>(m, "SetHomogCpleIm", DOC(MMVII_cSetHomogCpleIm))
		.def(py::init<>(),DOC(MMVII_cSetHomogCpleIm,cSetHomogCpleIm))
		.def(py::init<py::ssize_t >(),DOC(MMVII_cSetHomogCpleIm,cSetHomogCpleIm))
		.def("Add", &cSetHomogCpleIm::Add,DOC(MMVII_cSetHomogCpleIm,Add))
		.def_static("fromFile", &cSetHomogCpleIm::FromFile,DOC(MMVII_cSetHomogCpleIm,FromFile)) 
		.def("setH", py::overload_cast<>(&cSetHomogCpleIm::SetH), DOC(MMVII_cSetHomogCpleIm,SetH))
		
		.def("__repr__",
                 [](const cSetHomogCpleIm &m) {
                   std::ostringstream ss;
                   ss.precision(8);
                   ss << "SetHomogCpleIm "  << "\n" ;
                   for (auto aPt : m.SetH())
                       ss << "P1=[" << aPt.mP1.x() << " " << aPt.mP1.y() << "], P2=[ " << aPt.mP2.x() << " " << aPt.mP2.y() << "]\n";
                           return ss.str();
             });


	py::class_<cSetHomogCpleDir>(m, "SetHomogCpleDir", DOC(MMVII_cSetHomogCpleDir))
        .def(py::init<const cSetHomogCpleIm &,const cPerspCamIntrCalib &,const cPerspCamIntrCalib &>(),DOC(MMVII_cSetHomogCpleDir,cSetHomogCpleDir))
            .def("VDir1", &cSetHomogCpleDir::VDir1,DOC(MMVII_cSetHomogCpleDir,VDir1))
            .def("VDir2", &cSetHomogCpleDir::VDir2,DOC(MMVII_cSetHomogCpleDir,VDir2))

	    .def("__repr__",
                 [](const cSetHomogCpleDir &m) {
                   std::ostringstream ss;
                   ss.precision(8);

		   const std::vector<cPt3dr> vecPt1 = m.VDir1();
		   const std::vector<cPt3dr> vecPt2 = m.VDir2();
		   int nbPts = vecPt1.size();

                   ss << "SetHomogCpleDir " << "\n" ;
		   for (int aP=0; aP<nbPts; aP++)
                       ss << "P1=[" << vecPt1[aP].x() << " " << vecPt1[aP].y() << "], P2=[" << vecPt2[aP].x() << " " << vecPt2[aP].y()  << "]\n";
                           return ss.str();
	           
                 });


    py::class_<cMatEssential>(m, "MatEssential", DOC(MMVII_cMatEssential))
        .def(py::init<const cSetHomogCpleDir &,cLinearOverCstrSys<tREAL8> &,int>(),"homs"_a,"sys"_a,"kFix"_a,DOC(MMVII_cMatEssential,cMatEssential))
        .def("cost",&cMatEssential::Cost,"pt3dr1"_a,"pt3dr2"_a,"sigma"_a,DOC(MMVII_cMatEssential,Cost))
        .def("avgCost",&cMatEssential::AvgCost,"homs"_a,"sigma"_a,DOC(MMVII_cMatEssential,AvgCost))
        .def("kthCost",&cMatEssential::KthCost,"homs"_a,"prop"_a,DOC(MMVII_cMatEssential,KthCost))
        .def("mat",&cMatEssential::Mat,DOC(MMVII_cMatEssential,Mat))
        .def("show",&cMatEssential::Show,"homs"_a,DOC(MMVII_cMatEssential,Show))
        .def("computePose",[](const cMatEssential& aMat, const cSetHomogCpleDir & aHoms){return aMat.ComputePose(aHoms);},DOC(MMVII_cMatEssential,ComputePose))     // remove (defaulted) 2nd args of ComputePos
        ;

    m.def("matEss_GetKMax",&MatEss_GetKMax,py::arg("setD"),py::arg("weightStab"),py::arg("show")=false,DOC(MMVII_MatEss_GetKMax));
}
