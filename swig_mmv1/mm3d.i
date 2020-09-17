%module mm3d
%{
//includes to be able to compile
#define SWIG_FILE_WITH_INIT
#include "StdAfx.h"
#include "api/api_mm3d.h"
#include "general/photogram.h"
#include "general/util.h"
#include "private/files.h"
#include "XML_GEN/ParamChantierPhotogram.h"
typedef ElAffin2D tOrIntIma ;
#include "api/TpPPMD.h"
#include <sstream>
%}

#define ElSTDNS  std::
#define ElTmplSpecNull template <>
typedef ElAffin2D tOrIntIma ;
%include <std_string.i>
%include <std_vector.i>
%include <std_list.i>
%include <cpointer.i>
%include stl.i
//templates (has to be before %include "api/api_mm3d.h")
namespace std {
    %template(IntVector)    vector<int>;
    %template(DoubleVector) vector<double>;
    %template(StringVector) vector<string>;
    %template(HomolList)    list<cNupletPtsHomologues>;
    %template(CpleStringVector) vector<cCpleString>;
}
 
//def de REAL etc pour pouvoir les utiliser en python
%include "general/CMake_defines.h"
%include "general/sys_dep.h"

//renommage de methodes surchargees
//%rename(getCoeffX) cElComposHomographie::CoeffX();
//%rename(getCoeffY) cElComposHomographie::CoeffY();
//%rename(getCoeff1) cElComposHomographie::Coeff1();
//%rename(getCoeff) ElDistRadiale_PolynImpair::Coeff(int);

%rename(_True) FBool::True;
%rename(_False) FBool::False;


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

//classes to export
%include "api/api_mm3d.h"
%include "api/TpPPMD.h"
%include "general/util.h"
//%include "private/files.h" //not working for now
void MakeFileXML(const cSauvegardeNamedRel & anObj,const std::string & aName,const std::string & aTagEnglob=""); //just one version of MakeFileXML for now
%include "XML_GEN/ParamChantierPhotogram.h"

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

