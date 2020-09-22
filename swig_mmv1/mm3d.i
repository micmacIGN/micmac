%module mm3d
%{
//----------------------------------------------------------------------
//includes to be able to compile
#define SWIG_FILE_WITH_INIT
#include "StdAfx.h"
#include "api/api_mm3d.h"
#include "general/photogram.h"
#include "general/util.h"
#include "general/bitm.h"
#include "general/ptxd.h"
#include "private/files.h"
#include "XML_GEN/ParamChantierPhotogram.h"
typedef ElAffin2D tOrIntIma ; //mandatory because only declared in cBasicGeomCap3D in general/photogram.h?
#include "api/TpPPMD.h"
#include <sstream>
%}

//----------------------------------------------------------------------
#define ElSTDNS  std::
#define ElTmplSpecNull template <>
%include <std_string.i>
%include <std_vector.i>
%include <std_list.i>
%include <cpointer.i>
%include stl.i

//----------------------------------------------------------------------
//templates (has to be before %include "api/api_mm3d.h")
//used to make them usable as python lists
namespace std {
    %template(IntVector)    vector<int>;
    %template(DoubleVector) vector<double>;
    %template(StringVector) vector<string>;
    %template(HomolList)    list<cNupletPtsHomologues>;
    %template(CpleStringVector) vector<cCpleString>;
    //%template(MakeFileXML_cSauvegardeNamedRel) MakeFileXML<cSauvegardeNamedRel>; not working?
}
 
//def REAL etc to be able to use them in python
%include "general/CMake_defines.h"
%include "general/sys_dep.h"

//----------------------------------------------------------------------
//rename overloaded methods to avoid shadowing
%rename(getCoeffX) cElComposHomographie::CoeffX();
%rename(getCoeffY) cElComposHomographie::CoeffY();
%rename(getCoeff1) cElComposHomographie::Coeff1();
%rename(getCoeff) ElDistRadiale_PolynImpair::Coeff(int);

%rename(read_Pt3dr) ELISE_fp::read(Pt3dr *);
%rename(read_Pt3df) ELISE_fp::read(Pt3df *);
%rename(read_ElMatrix) ELISE_fp::read(ElMatrix<REAL> *);
%rename(read_ElRotation3D) ELISE_fp::read(ElRotation3D *);
%rename(read_Polynome2dReal) ELISE_fp::read(Polynome2dReal *);
%rename(read_REAL8) ELISE_fp::read(REAL8 *);
%rename(read_INT4) ELISE_fp::read(INT4 *);
%rename(read_Pt2dr) ELISE_fp::read(Pt2dr *);
%rename(read_Pt2df) ELISE_fp::read(Pt2df *);
%rename(read_Pt2di) ELISE_fp::read(Pt2di *);
%rename(read_Seg2d) ELISE_fp::read(Seg2d *);
%rename(read_string) ELISE_fp::read(std::string *);
%rename(read_bool) ELISE_fp::read(bool *);
%rename(read_PolynomialEpipolaireCoordinate) ELISE_fp::read(PolynomialEpipolaireCoordinate *);
%rename(read_CpleEpipolaireCoord) ELISE_fp::read(CpleEpipolaireCoord *);
%rename(read_Seg2d_list) ELISE_fp::read(std::list<Seg2d> *);
%rename(read_REAL8_vect) ELISE_fp::read(std::vector<REAL8> *);
%rename(read_int_vect) ELISE_fp::read(std::vector<int> *);
%rename(read_Pt2di_vect) ELISE_fp::read(std::vector<Pt2di> *);
%rename(read_ElCplePtsHomologues) ELISE_fp::read(ElCplePtsHomologues *);
%rename(read_cNupletPtsHomologues) ELISE_fp::read(cNupletPtsHomologues *);

%rename(ToXMLTree_Pt3dr) ToXMLTree(const Pt3dr &      anObj);
%rename(ToXMLTree_bool) ToXMLTree(const std::string & aNameTag,const bool   &      anObj);
%rename(ToXMLTree_double) ToXMLTree(const std::string & aNameTag,const double &      anObj);
%rename(ToXMLTree_int) ToXMLTree(const std::string & aNameTag,const int    &      anObj);
%rename(ToXMLTree_Box2dr) ToXMLTree(const std::string & aNameTag,const Box2dr &      anObj);
%rename(ToXMLTree_Box2di) ToXMLTree(const std::string & aNameTag,const Box2di &      anObj);
%rename(ToXMLTree_Pt2di) ToXMLTree(const std::string & aNameTag,const Pt2di &      anObj);
%rename(ToXMLTree_Pt2dr) ToXMLTree(const std::string & aNameTag,const Pt2dr &      anObj);
%rename(ToXMLTree_string) ToXMLTree(const std::string & aNameTag,const std::string & anObj);
%rename(ToXMLTree_double_vect) ToXMLTree(const std::string & aNameTag,const std::vector<double> & anObj);
%rename(ToXMLTree_int_vect) ToXMLTree(const std::string & aNameTag,const std::vector<int> & anObj);
%rename(ToXMLTree_string_vect) ToXMLTree(const std::string & aNameTag,const std::vector<std::string> & anObj);
%rename(ToXMLTree_Pt3dr) ToXMLTree(const std::string & aNameTag,const Pt3dr &      anObj);
%rename(ToXMLTree_Pt3di) ToXMLTree(const std::string & aNameTag,const Pt3di &      anObj);
%rename(ToXMLTree_cElRegex_Ptr) ToXMLTree(const std::string & aNameTag,const cElRegex_Ptr &      anObj);
%rename(ToXMLTree_XmlXml) ToXMLTree(const std::string & aNameTag,const XmlXml &      anObj);
%rename(ToXMLTree_cCpleString) ToXMLTree(const std::string & aNameTag,const cCpleString   &      anObj);
%rename(ToXMLTree_cMonomXY) ToXMLTree(const std::string & aNameTag,const cMonomXY   &      anObj);
%rename(ToXMLTree_IntSubst) ToXMLTree(const std::string & aNameTag,const IntSubst   &      anObj);
%rename(ToXMLTree_BoolSubst) ToXMLTree(const std::string & aNameTag,const BoolSubst   &      anObj);
%rename(ToXMLTree_DoubleSubst) ToXMLTree(const std::string & aNameTag,const DoubleSubst   &      anObj);
%rename(ToXMLTree_Pt2diSubst) ToXMLTree(const std::string & aNameTag,const Pt2diSubst   &      anObj);
%rename(ToXMLTree_Pt2drSubst) ToXMLTree(const std::string & aNameTag,const Pt2drSubst   &      anObj);

%rename(BinaryDumpInFile_bool) BinaryDumpInFile(ELISE_fp &,const bool &);
%rename(BinaryDumpInFile_double) BinaryDumpInFile(ELISE_fp &,const double &);
%rename(BinaryDumpInFile_int) BinaryDumpInFile(ELISE_fp &,const int &);
%rename(BinaryDumpInFile_Box2dr) BinaryDumpInFile(ELISE_fp &,const Box2dr &);
%rename(BinaryDumpInFile_Box2di) BinaryDumpInFile(ELISE_fp &,const Box2di &);
%rename(BinaryDumpInFile_Pt2dr) BinaryDumpInFile(ELISE_fp &,const Pt2dr &);
%rename(BinaryDumpInFile_Pt2di) BinaryDumpInFile(ELISE_fp &,const Pt2di &);
%rename(BinaryDumpInFile_string) BinaryDumpInFile(ELISE_fp &,const std::string &);
%rename(BinaryDumpInFile_double_vect) BinaryDumpInFile(ELISE_fp &,const std::vector<double> &);
%rename(BinaryDumpInFile_int_vect) BinaryDumpInFile(ELISE_fp &,const std::vector<int> &);
%rename(BinaryDumpInFile_string_vect) BinaryDumpInFile(ELISE_fp &,const std::vector<std::string> &);
%rename(BinaryDumpInFile_Pt3dr) BinaryDumpInFile(ELISE_fp &,const Pt3dr &);
%rename(BinaryDumpInFile_Pt3di) BinaryDumpInFile(ELISE_fp &,const Pt3di &);
%rename(BinaryDumpInFile_cElRegex_Ptr) BinaryDumpInFile(ELISE_fp &,const cElRegex_Ptr &);
%rename(BinaryDumpInFile_cCpleString) BinaryDumpInFile(ELISE_fp &,const cCpleString &);
%rename(BinaryDumpInFile_cMonomXY) BinaryDumpInFile(ELISE_fp &,const cMonomXY &);
%rename(BinaryDumpInFile_IntSubst) BinaryDumpInFile(ELISE_fp &,const IntSubst &);
%rename(BinaryDumpInFile_BoolSubst) BinaryDumpInFile(ELISE_fp &,const BoolSubst &);
%rename(BinaryDumpInFile_DoubleSubst) BinaryDumpInFile(ELISE_fp &,const DoubleSubst &);
%rename(BinaryDumpInFile_Pt2diSubst) BinaryDumpInFile(ELISE_fp &,const Pt2diSubst &);
%rename(BinaryDumpInFile_Pt2drSubst) BinaryDumpInFile(ELISE_fp &,const Pt2drSubst &);
%rename(BinaryDumpInFile_XmlXml) BinaryDumpInFile(ELISE_fp &,const XmlXml &);

%rename(BinaryUnDumpInFile_bool) BinaryUnDumpInFile(bool &,ELISE_fp &);
%rename(BinaryUnDumpInFile_double) BinaryUnDumpInFile(double &,ELISE_fp &);
%rename(BinaryUnDumpInFile_int) BinaryUnDumpInFile(int &,ELISE_fp &);
%rename(BinaryUnDumpInFile_Box2dr) BinaryUnDumpInFile(Box2dr &,ELISE_fp &);
%rename(BinaryUnDumpInFile_Box2di) BinaryUnDumpInFile(Box2di &,ELISE_fp &);
%rename(BinaryUnDumpInFile_Pt2dr) BinaryUnDumpInFile(Pt2dr &,ELISE_fp &);
%rename(BinaryUnDumpInFile_Pt2di) BinaryUnDumpInFile(Pt2di &,ELISE_fp &);
%rename(BinaryUnDumpInFile_string) BinaryUnDumpInFile(std::string &,ELISE_fp &);
%rename(BinaryUnDumpInFile_double_vect) BinaryUnDumpInFile(std::vector<double> &,ELISE_fp &);
%rename(BinaryUnDumpInFile_int_vect) BinaryUnDumpInFile(std::vector<int> &,ELISE_fp &);
%rename(BinaryUnDumpInFile_string_vect) BinaryUnDumpInFile(std::vector<std::string> &,ELISE_fp &);
%rename(BinaryUnDumpInFile_Pt3dr) BinaryUnDumpInFile(Pt3dr &,ELISE_fp &);
%rename(BinaryUnDumpInFile_Pt3di) BinaryUnDumpInFile(Pt3di &,ELISE_fp &);
%rename(BinaryUnDumpInFile_cElRegex_Ptr) BinaryUnDumpInFile(cElRegex_Ptr &,ELISE_fp &);
%rename(BinaryUnDumpInFile_cCpleString) BinaryUnDumpInFile(cCpleString &,ELISE_fp &);
%rename(BinaryUnDumpInFile_cMonomXY) BinaryUnDumpInFile(cMonomXY &,ELISE_fp &);
%rename(BinaryUnDumpInFile_IntSubst) BinaryUnDumpInFile(IntSubst &,ELISE_fp &);
%rename(BinaryUnDumpInFile_BoolSubst) BinaryUnDumpInFile(BoolSubst &,ELISE_fp &);
%rename(BinaryUnDumpInFile_DoubleSubst) BinaryUnDumpInFile(DoubleSubst &,ELISE_fp &);
%rename(BinaryUnDumpInFile_Pt2diSubst) BinaryUnDumpInFile(Pt2diSubst &,ELISE_fp &);
%rename(BinaryUnDumpInFile_Pt2drSubst) BinaryUnDumpInFile(Pt2drSubst &,ELISE_fp &);
%rename(BinaryUnDumpInFile_XmlXml) BinaryUnDumpInFile(XmlXml &,ELISE_fp &);

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
%ignore ELISE_fp::CKK_write_FileOffset8;
%ignore ELISE_fp::read_INT8;
%ignore cElXMLToken::Show;
%ignore cElXMLTree::Attrs;
%ignore ELISE_fp::write;
%ignore Mangling;

//misc
%ignore Test_DBL;
%ignore ELISE_fp::if_not_exist_create_0;

//----------------------------------------------------------------------
//classes to export
%include "api/api_mm3d.h"
%include "api/TpPPMD.h"
%include "general/util.h"
%include "general/bitm.h"
%include "general/ptxd.h"
%include "general/photogram.h"
%include "private/files.h"
%include "XML_GEN/ParamChantierPhotogram.h"

//----------------------------------------------------------------------
%template(Pt2di) Pt2d<INT>;
%template(Pt2dr) Pt2d<REAL>;
%template(Pt3dr) Pt3d<REAL>;
%template(ElRotation3D) TplElRotation3D<REAL>;
%template(ElMatrixr) ElMatrix<REAL>;


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

