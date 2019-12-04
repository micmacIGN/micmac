%module mm3d
%{
#include "StdAfx.h"
#include "api/api_mm3d.h"
#include "general/photogram.h"
typedef ElAffin2D tOrIntIma ;
%}

#define ElSTDNS  std::
#define ElTmplSpecNull template <>
typedef ElAffin2D tOrIntIma ;
%include <std_string.i>
%include <std_vector.i>
%include <cpointer.i>
 
//def de REAL etc pour pouvoir les utiliser en python
%include "general/CMake_defines.h"
%include "general/sys_dep.h"

//avec les classes a exporter
%include "api/api_mm3d.h"
%template(Pt2di) Pt2d<INT>;
%template(Pt2dr) Pt2d<REAL>;
%template(Pt3dr) Pt3d<REAL>;


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
