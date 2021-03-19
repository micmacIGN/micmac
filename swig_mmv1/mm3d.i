%module mm3d
%{
//----------------------------------------------------------------------
//includes to be able to compile
#define SWIG_FILE_WITH_INIT
#include "StdAfx.h"
#include "api/api_mm3d.h"
#include "../src/uti_phgrm/NewOri/NewOri.h"
#include "api/NewO_PyWrapper.h"
#include "general/photogram.h"
#include "general/util.h"
#include "general/bitm.h"
#include "general/ptxd.h"
#include "general/geom_vecteur.h"
#include "private/files.h"
#include "XML_GEN/ParamChantierPhotogram.h"
typedef ElAffin2D tOrIntIma ; //mandatory because only declared in cBasicGeomCap3D in general/photogram.h?
#include "../src/TpMMPD/TpPPMD.h"
#include "../src/uti_phgrm/TiepTri/MultTieP.h"
//#include "XML_GEN/xml_gen2_mmByp.h"
#include <sstream>
%}

//----------------------------------------------------------------------
#define ElSTDNS  std::
#define ElTmplSpecNull template <>
%include <std_string.i>
%include <std_vector.i>
%include <std_map.i>
%include <std_list.i>
%include <cpointer.i>
%include stl.i

//def REAL etc to be able to use them in python
%include "general/CMake_defines.h"
%include "general/sys_dep.h"

//----------------------------------------------------------------------
//rename overloaded methods to avoid shadowing
%rename(getCoeffX) cElComposHomographie::CoeffX();
%rename(getCoeffY) cElComposHomographie::CoeffY();
%rename(getCoeff1) cElComposHomographie::Coeff1();
%rename(getCoeff) ElDistRadiale_PolynImpair::Coeff(int);

//True is a reserved name in python3
%rename(_True) FBool::True;
%rename(_MayBe) FBool::MayBe;
%rename(_False) FBool::False;

//----------------------------------------------------------------------
//be able to use python exceptions
%include exception.i       
%exception {
  try {
    $action
  } catch(runtime_error & e) {
    SWIG_exception(SWIG_RuntimeError, e.what());
  } catch(...) {
    SWIG_exception(SWIG_RuntimeError, "Unknown exception");
  }
}

//----------------------------------------------------------------------
//things to ignore in next includes to be able to compile

//need several default constructors
%ignore Im2DGen::neigh_test_and_set;
%ignore GenIm::load_file;
%ignore GenIm::box;
%ignore to_flux;
%ignore Liste_Pts_Gen::all_pts;
%ignore CalcPtsInteret::GetOnePtsInteret;
%ignore cElemMepRelCoplan::Plan;

//implemented in a cpp file not used when compiling mm3d_wrap.cxx
%ignore cTabulKernelInterpol::AdrDisc2Real;
%ignore cTabulKernelInterpol::DerAdrDisc2Real;
%ignore RansacMatriceEssentielle;

//complex template: to fix
%ignore jacobi_diag;

//unimplemented
%ignore tCho2double;
%ignore cComposElMap2D::NewFrom0;
%ignore cComposElMap2D::NewFrom1;
%ignore cComposElMap2D::NewFrom2;
%ignore cComposElMap2D::NewFrom3;
%ignore Monome2dReal::Ampl;
%ignore ChangementSysC;
%ignore cCs2Cs::Delete;
%ignore ElPhotogram::bench_photogram_0;
%ignore cMirePolygonEtal;
%ignore cProjCple;
%ignore cCpleEpip;
%ignore ElPhotogram;
%ignore EcartTotalProjection;
%ignore cProjListHom;
%ignore cDbleGrid::Name;
%ignore TestMEPCoCentrik;
%ignore TestInterPolyCercle;
%ignore cElTriangleComp::Test;
%ignore cMailageSphere::DirMoyH;
%ignore cGridNuageP3D;
%ignore cElNuageLaser::Debug;
%ignore cElNuageLaser::SauvCur;
//misc
%ignore Test_DBL;


//----------------------------------------------------------------------
//templates (has to be before %include "api/api_mm3d.h")
//used to make them usable as python lists
//first, expose templates
%include "general/ptxd.h"
%include "general/bitm.h"
%include "api/files_extract.h"
//then, name templates implementations
%template(Pt2di) Pt2d<INT>;
%template(Pt2dr) Pt2d<REAL>;
%template(Pt3dr) Pt3d<REAL>;
%template(ElRotation3D) TplElRotation3D<REAL>;
%template(ElMatrixr) ElMatrix<REAL>;
%template(MakeFileXML_cSauvegardeNamedRel) MakeFileXML<cSauvegardeNamedRel>;

namespace std {
    %template(IntVector)    vector<int>;
    %template(DoubleVector) vector<double>;
    %template(FloatVector)  vector<float>;
    %template(StringVector) vector<string>;
    %template(HomolList)    list<cNupletPtsHomologues>;
    %template(CpleStringVector) vector<cCpleString>;
    %template(cXml_OneTripletList) list< cXml_OneTriplet >;
    %template(ElSeg3DVector)  vector<ElSeg3D>;
    %template(RelMVector) vector<RelMotion>;
    %template(PtVector) vector<Pt2dr>;
    %template(Pt3drVector) vector<Pt3dr>;
    %template(FeatureMap) map<int,vector<Pt2dr > >;
}
 

//----------------------------------------------------------------------
//classes to export
%include "api/api_mm3d.h"
%include "api/SuperposImage_extract.h"
%include "api/ParamChantierPhotogram_extract.h"
%include "../src/uti_phgrm/NewOri/NewOri.h"
%include "api/NewO_PyWrapper.h"
%include "../src/TpMMPD/TpPPMD.h"
%include "general/util.h"
%include "general/geom_vecteur.h"
%include "general/photogram.h" //45s
%include "../src/uti_phgrm/TiepTri/MultTieP.h"

//%include "XML_GEN/xml_gen2_mmByp.h"

//----------------------------------------------------------------------

//check python version
%pythoncode %{
print("MicMac Python3 API")
mm3d_init();
%}

//----------------------------------------------------------------------
/*%extend cPackNupletsHom {
  std::list<cNupletPtsHomologues> &getList() {
      return $self->mCont;
  }
};*/

//ajout print PtNdx
%extend Pt2d<INT> {
  char *__repr__() {
    static char tmp[1024];
    sprintf(tmp, "[%d, %d]", $self->x, $self->y);
    return tmp;
  }
};
%extend Pt2d<REAL> {
  char *__repr__() {
    static char tmp[1024];
    sprintf(tmp, "[%g, %g]", $self->x, $self->y);
    return tmp;
  }
};
%extend Pt3d<REAL> {
  char *__repr__() {
    static char tmp[1024];
    sprintf(tmp, "[%g, %g, %g]", $self->x, $self->y, $self->z);
    return tmp;
  }
};

/*
%extend std::vector<double> {
  char *__repr__() {
    static char tmp[1024];
    std::ostringstream oss;
    oss<<"[";
    for (unsigned int i=0;i<$self->size();i++)
    {
      oss<<$self->at(i);
      if (i<$self->size()-1)
        oss<<", ";
    }
    oss<<"]";
    strncpy(tmp,oss.str().c_str(),1023);
    return tmp;
  }
};
*/

