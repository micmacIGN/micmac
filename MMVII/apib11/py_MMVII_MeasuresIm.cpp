#include "py_MMVII.h"

#include "MMVII_MeasuresIm.h"
#include "src/BundleAdjustment/BundleAdjustment.h"

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

    py::class_<cMes1Gnd3D>(m, "Mes1GCP", DOC(MMVII_cMes1Gnd3D))
            .def(py::init<>(),DOC(MMVII_cMes1Gnd3D,cMes1Gnd3D))
            .def(py::init<const cPt3dr &,const std::string &,tREAL4>(),DOC(MMVII_cMes1Gnd3D,cMes1Gnd3D))
            .def_readwrite("pt", &cMes1Gnd3D::mPt,DOC(MMVII_cMes1Gnd3D,mPt))
            .def_readwrite("namePt", &cMes1Gnd3D::mNamePt,DOC(MMVII_cMes1Gnd3D,mNamePt))
            .def_property_readonly("sXX",[](const cMes1Gnd3D& m){return m.Sigma2IsInit()?py::object(py::float_(m.Sigma2().at(m.IndXX))):py::object(pybind11::none());},"Sigma2 of x coordinate")
            .def_property_readonly("sYY",[](const cMes1Gnd3D& m){return m.Sigma2IsInit()?py::object(py::float_(m.Sigma2().at(m.IndYY))):py::object(pybind11::none());},"Sigma2 of y coordinate")
            .def_property_readonly("sZZ",[](const cMes1Gnd3D& m){return m.Sigma2IsInit()?py::object(py::float_(m.Sigma2().at(m.IndZZ))):py::object(pybind11::none());},"Sigma2 of z coordinate")
            .def("SetSigma2",py::overload_cast<const cPt3dr&>(&cMes1Gnd3D::SetSigma2),DOC(MMVII_cMes1Gnd3D,SetSigma2))
            .def("__repr__",
                 [](const cMes1Gnd3D &m) {
                   std::ostringstream ss;
                   ss.precision(8);
                   ss << "Mes1GCP " << m.mNamePt << " " << m.mPt;
                   if (m.Sigma2IsInit())
                      ss << ", sigma2 (XX,YY,ZZ): ("<< m.Sigma2().at(m.IndXX) << ", " << m.Sigma2().at(m.IndYY) << ", " << m.Sigma2().at(m.IndZZ) << ")";
                   return ss.str();
             })
            ;

    py::class_<cSetMesGnd3D>(m, "SetMesGCP", DOC(MMVII_cSetMesGnd3D))
            .def(py::init<>(),DOC(MMVII_cSetMesGnd3D,cSetMesGnd3D))
            .def(py::init<const std::string &>(),DOC(MMVII_cSetMesGnd3D,cSetMesGnd3D))
            .def_static("fromFile", &cSetMesGnd3D::FromFile,DOC(MMVII_cSetMesGnd3D,FromFile))
            .def("toFile", py::overload_cast<const std::string &>(&cSetMesGnd3D::ToFile, py::const_),DOC(MMVII_cSetMesGnd3D,ToFile))
            .def("stdNameFile", &cSetMesGnd3D::StdNameFile,DOC(MMVII_cSetMesGnd3D,StdNameFile))
            .def("measures", &cSetMesGnd3D::Measures, py::return_value_policy::reference_internal, DOC(MMVII_cSetMesGnd3D,Measures))
            .def("__repr__",
                 [](const cSetMesGnd3D &m) {
                   std::ostringstream ss;
                   ss.precision(8);
                   ss << "SetMesGCP " << m.StdNameFile() << "\n" ;
                   for (auto aPt : m.Measures())
                       ss << aPt.mPt.x() << " " << aPt.mPt.y() << " " << aPt.mPt.z() << "\n";
                           return ss.str();
             })
            ;


    py::class_<cPair2D3D>(m, "Pair2D3D", DOC(MMVII_cPair2D3D))
            .def(py::init<const cPt2dr &,const cPt3dr &>(),DOC(MMVII_cPair2D3D,cPair2D3D))
            .def_readwrite("p2", &cPair2D3D::mP2,DOC(MMVII_cPair2D3D,mP2))
            .def_readwrite("p3", &cPair2D3D::mP3,DOC(MMVII_cPair2D3D,mP3))
 
            .def("__repr__",
                 [](const cPair2D3D &m) {
                   std::ostringstream ss;
                   ss.precision(8);
                   ss << "cPair2D3D " << "\n" ;
                   ss << m.mP2.x() << " " << m.mP2.y() << " \n" << m.mP3.x() << " " << m.mP3.y() << " " << m.mP3.z() << "\n";
                           return ss.str();
             })
            ;

    py::class_<cWeightedPair2D3D>(m, "WeightedPair2D3D", DOC(MMVII_cWeightedPair2D3D))
            .def(py::init<const cPt2dr &,const cPt3dr &>(),DOC(MMVII_cWeightedPair2D3D,cWeightedPair2D3D))

            .def_readwrite("weight", &cPair2D3D::mP3,DOC(MMVII_cPair2D3D,mP3))
 
            .def("__repr__",
                 [](const cPair2D3D &m) {
                   std::ostringstream ss;
                   ss.precision(8);
                   ss << "cPair2D3D " << "\n" ;
                   ss << m.mP2.x() << " " << m.mP2.y() << " \n" << m.mP3.x() << " " << m.mP3.y() << " " << m.mP3.z() << "\n";
                           return ss.str();
             })
            ;



    py::class_<cSet2D3D>(m, "Set2D3D", DOC(MMVII_cSet2D3D))
	    .def(py::init<>(),DOC(MMVII_cSet2D3D)) 
            .def("nbPair", &cSet2D3D::NbPair,DOC(MMVII_cSet2D3D,NbPair))
            .def("addPair", py::overload_cast<const cPt2dr&,const cPt3dr&,double>(&cSet2D3D::AddPair), DOC(MMVII_cSet2D3D,AddPair))
            ;

	    

    py::class_<cSetMesGndPt>(m, "SetMesImGCP", DOC(MMVII_cSetMesGndPt))
            .def(py::init<>(),DOC(MMVII_cSetMesGndPt,cSetMesGndPt))
            .def("addMes3D", &cSetMesGndPt::AddMes3D, py::return_value_policy::reference_internal, DOC(MMVII_cSetMesGndPt,AddMes3D))
            .def("add1GCP", &cSetMesGndPt::Add1GCP, py::return_value_policy::reference_internal, DOC(MMVII_cSetMesGndPt,Add1GCP))
           // .def("addMes2D", &cSetMesGndPt::AddMes2D, py::return_value_policy::reference_internal, DOC(MMVII_cSetMesGndPt,AddMes2D))
            .def("extractMes1Im", &cSetMesGndPt::ExtractMes1Im, py::return_value_policy::reference_internal, DOC(MMVII_cSetMesGndPt,ExtractMes1Im))
//          .def("mesGCP", &cSetMesGndPt::MesGCP, py::return_value_policy::reference_internal, DOC(MMVII_cSetMesGndPt,MesGCP))
            .def("mesImOfPt", &cSetMesGndPt::MesImOfPt, py::return_value_policy::reference_internal, DOC(MMVII_cSetMesGndPt,MesImOfPt))
            .def("mesImInitOfName", &cSetMesGndPt::MesImInitOfName, py::return_value_policy::reference_internal, DOC(MMVII_cSetMesGndPt,MesImInitOfName))
	    .def("mesGCPOfName", py::overload_cast<const std::string&>(&cSetMesGndPt::MesGCPOfName, py::const_),DOC(MMVII_cSetMesGndPt,MesGCPOfName))

  /*          //.def("vSens", &cSetMesGndPt::VSens, py::return_value_policy::reference_internal, DOC(MMVII_cSetMesGndPt,VSens)) //cSensorImage not defined in api
*/
            .def("__repr__",
                 [](const cSetMesGndPt &m) {
                   std::ostringstream ss;
                   ss.precision(8);
                   ss << "SetMesImGCP " << "\n" ;
                   for (auto aPt : m.MesGCP())
                       ss << aPt.mPt.x() << " " << aPt.mPt.y() << " " << aPt.mPt.z() << "\n";
                           return ss.str();
             })
            ;


}
