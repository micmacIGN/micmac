#include "py_MMVII.h"

#include "MMVII_MeasuresIm.h"

using namespace MMVII;


void pyb_init_MeasuresIm(py::module_ &m) {
    py::class_<cMesIm1Pt>(m, "MesIm1Pt", DOC(MMVII_cMesIm1Pt))
            .def(py::init<>(),DOC(MMVII_cMesIm1Pt,cMesIm1Pt))
            .def(py::init<const cPt2dr &,const std::string &,tREAL4>(),DOC(MMVII_cMesIm1Pt,cMesIm1Pt))
            .def_readwrite("pt", &cMesIm1Pt::mPt,DOC(MMVII_cMesIm1Pt,mPt))
            .def_readwrite("namePt", &cMesIm1Pt::mNamePt,DOC(MMVII_cMesIm1Pt,mNamePt))
            .def_property("sXX",[](const cMesIm1Pt& m){return m.mSigma2[0];},[](cMesIm1Pt& m, tREAL8 sXX){ m.mSigma2[0] = sXX;},"Sigma2 of x coordinate")
            .def_property("sYY",[](const cMesIm1Pt& m){return m.mSigma2[2];},[](cMesIm1Pt& m, tREAL8 sYY){ m.mSigma2[2] = sYY;},"Sigma2 of y coordinate")
            .def("__repr__",
                 [](const cMesIm1Pt &m) {
                   std::ostringstream ss;
                   ss.precision(8);
                   ss << "MesIm1Pt " << m.mNamePt << " " << m.mPt << ", sigma2 (xx,xy,yy): "
                      << m.mSigma2[0] << ", " << m.mSigma2[1] << ", " << m.mSigma2[2] << ")";
                   return ss.str();
             })
            ;

    py::class_<cSetMesPtOf1Im>(m, "SetMesPtOf1Im", DOC(MMVII_cSetMesPtOf1Im))
            .def(py::init<>(),DOC(MMVII_cSetMesPtOf1Im,cSetMesPtOf1Im))
            .def(py::init<const std::string &>(),DOC(MMVII_cSetMesPtOf1Im,cSetMesPtOf1Im))
            .def_static("fromFile", &cSetMesPtOf1Im::FromFile,DOC(MMVII_cSetMesPtOf1Im,FromFile))
            .def("AddMeasure", &cSetMesPtOf1Im::AddMeasure,DOC(MMVII_cSetMesPtOf1Im,AddMeasure))
            .def("toFile", py::overload_cast<const std::string &>(&cSetMesPtOf1Im::ToFile, py::const_),DOC(MMVII_cSetMesPtOf1Im,ToFile))
            .def("nameIm", &cSetMesPtOf1Im::NameIm,DOC(MMVII_cSetMesPtOf1Im,NameIm))
            .def("stdNameFile", &cSetMesPtOf1Im::StdNameFile,DOC(MMVII_cSetMesPtOf1Im,StdNameFile))
            .def("measures", py::overload_cast<>(&cSetMesPtOf1Im::Measures), py::return_value_policy::reference_internal, DOC(MMVII_cSetMesPtOf1Im,Measures))

            .def("measuresOfName", py::overload_cast<const std::string & >(&cSetMesPtOf1Im::MeasuresOfName), py::return_value_policy::reference_internal, DOC(MMVII_cSetMesPtOf1Im,MeasuresOfName))
            .def("nameHasMeasure", &cSetMesPtOf1Im::NameHasMeasure, DOC(MMVII_cSetMesPtOf1Im,NameHasMeasure))
            .def("nearestMeasure", &cSetMesPtOf1Im::NearestMeasure, py::return_value_policy::reference_internal, DOC(MMVII_cSetMesPtOf1Im,NearestMeasure))


            .def("__repr__",
                 [](const cSetMesPtOf1Im &m) {
                   std::ostringstream ss;
                   ss.precision(8);
                   ss << "SetMesPtOf1Im " << m.StdNameFile() << "\n" ;
                   for (auto aPt : m.Measures())
                       ss << aPt.mNamePt << " " << aPt.mPt.x() << " " << aPt.mPt.y() << "\n";
                           return ss.str();
             })
            ;

    py::class_<cMes1GCP>(m, "Mes1GCP", DOC(MMVII_cMes1GCP))
            .def(py::init<>(),DOC(MMVII_cMes1GCP,cMes1GCP))
            .def(py::init<const cPt3dr &,const std::string &,tREAL4>(),DOC(MMVII_cMes1GCP,cMes1GCP))
            .def_readwrite("pt", &cMes1GCP::mPt,DOC(MMVII_cMes1GCP,mPt))
            .def_readwrite("namePt", &cMes1GCP::mNamePt,DOC(MMVII_cMes1GCP,mNamePt))
            .def_property("sXX",[](const cMes1GCP& m){return m.mSigma2[m.IndXX];},[](cMes1GCP& m, tREAL8 sXX){ m.mSigma2[m.IndXX] = sXX;},"Sigma2 of x coordinate")
            .def_property("sYY",[](const cMes1GCP& m){return m.mSigma2[m.IndYY];},[](cMes1GCP& m, tREAL8 sYY){ m.mSigma2[m.IndYY] = sYY;},"Sigma2 of y coordinate")
            .def_property("sZZ",[](const cMes1GCP& m){return m.mSigma2[m.IndZZ];},[](cMes1GCP& m, tREAL8 sZZ){ m.mSigma2[m.IndZZ] = sZZ;},"Sigma2 of z coordinate")
            .def("__repr__",
                 [](const cMes1GCP &m) {
                   std::ostringstream ss;
                   ss.precision(8);
                   ss << "Mes1GCP " << m.mNamePt << " " << m.mPt << ", sigma2 (XX,YY,ZZ): ("
                      << m.mSigma2[m.IndXX] << ", " << m.mSigma2[m.IndYY] << ", " << m.mSigma2[m.IndZZ] << ")";
                   return ss.str();
             })
            ;

    py::class_<cSetMesGCP>(m, "SetMesGCP", DOC(MMVII_cSetMesGCP))
            .def(py::init<>(),DOC(MMVII_cSetMesGCP,cSetMesGCP))
            .def(py::init<const std::string &>(),DOC(MMVII_cSetMesGCP,cSetMesGCP))
            .def_static("fromFile", &cSetMesGCP::FromFile,DOC(MMVII_cSetMesGCP,FromFile))
            .def("toFile", py::overload_cast<const std::string &>(&cSetMesGCP::ToFile, py::const_),DOC(MMVII_cSetMesGCP,ToFile))
            .def("stdNameFile", &cSetMesGCP::StdNameFile,DOC(MMVII_cSetMesGCP,StdNameFile))
            .def("measures", &cSetMesGCP::Measures, py::return_value_policy::reference_internal, DOC(MMVII_cSetMesGCP,Measures))
            .def("__repr__",
                 [](const cSetMesGCP &m) {
                   std::ostringstream ss;
                   ss.precision(8);
                   ss << "SetMesGCP " << m.StdNameFile() << "\n" ;
                   for (auto aPt : m.Measures())
                       ss << aPt.mPt.x() << " " << aPt.mPt.y() << " " << aPt.mPt.z() << "\n";
                           return ss.str();
             })
            ;
}