/* -*- C -*- */
%module mmv2
%{
  //includes to be able to compile
  #define SWIG_FILE_WITH_INIT
  #include <stdexcept>
  using std::runtime_error;
  #include "MMVII_all.h"
  #include "api/api_mmv2.h"
  #include <sstream>
  using namespace MMVII;
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



//----------------------------------------------------------------------
//rename overloaded methods to avoid shadowing

//Reserved names in python3
%rename(_True) FBool::True;
%rename(_MayBe) FBool::MayBe;
%rename(_False) FBool::False;
%rename(_None) MMVII::eApDT::None;

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
//must ignore forbidden cPtxd methods according to dim
%ignore MMVII::cPtxd<double,2>::cPtxd(const double & x);
%ignore MMVII::cPtxd<double,2>::cPtxd(const double & x,const double &y,const double &z);
%ignore MMVII::cPtxd<double,2>::cPtxd(const double & x,const double &y,const double &z,const double &t);
%ignore MMVII::cPtxd<double,2>::z;
%ignore MMVII::cPtxd<double,2>::t;
%ignore MMVII::cPtxd<int,2>::cPtxd(const int & x);
%ignore MMVII::cPtxd<int,2>::cPtxd(const int & x,const int &y,const int &z);
%ignore MMVII::cPtxd<int,2>::cPtxd(const int & x,const int &y,const int &z,const int &t);
%ignore MMVII::cPtxd<int,2>::z;
%ignore MMVII::cPtxd<int,2>::t;
%ignore MMVII::cPtxd<double,3>::cPtxd(const double & x);
%ignore MMVII::cPtxd<double,3>::cPtxd(const double & x,const double &y);
%ignore MMVII::cPtxd<double,3>::cPtxd(const double & x,const double &y,const double &z,const double &t);
%ignore MMVII::cPtxd<double,3>::t;
//not implemented
%ignore MMVII::cPtxd::Col;
%ignore MMVII::cPtxd::Line;
%ignore MMVII::cPtxd::ToVect;
%ignore MMVII::cPtxd<int,2>::ToStdVector;
%ignore MMVII::cPtxd::FromVect;
%ignore MMVII::cPtxd<int,2>::FromStdVector;

//----------------------------------------------------------------------
//classes to export
%nodefaultctor;
%include "api/api_mmv2.h"
%include "MMVII_enums.h"
%include "MMVII_Ptxd.h"
%include "MMVII_Images.h"
%include "MMVII_memory.h"

//templates have to be named to be exported
namespace MMVII {
  %template(cIm2Du1) cIm2D<tU_INT1>;
  %template(cIm2Dr4) cIm2D<tREAL4>;
  %template(cDataIm2Du1) cDataIm2D<tU_INT1>;
  %template(cDataIm2Dr4) cDataIm2D<tREAL4>;
  //%template(cBox2di) cTplBox<int,2>;

  //%template(Pt1dr) cPtxd<double,1>   ;
  //%template(Pt1di) cPtxd<int,1>      ;
  //%template(Pt1df) cPtxd<float,1>    ;
  %template(Pt2dr) cPtxd<double,2>   ;
  %template(Pt2di) cPtxd<int,2>      ;
  //%template(Pt2df) cPtxd<float,2>    ;
  %template(Pt3dr) cPtxd<double,3>   ;
  //%template(Pt3di) cPtxd<int,3>      ;
  //%template(Pt3df) cPtxd<float,3>    ;
}

//used to make them usable as python lists
namespace std {
    %template(IntVector)    vector<int>;
    %template(DoubleVector) vector<double>;
    %template(FloatVector)  vector<float>;
    %template(StringVector) vector<string>;
    
}


//----------------------------------------------------------------------
//run on import
%pythoncode %{
print("MicMacV2 Python3 API")
mmv2_init();
%}


//----------------------------------------------------------------------
//add functions to classes
%extend MMVII::cPtxd<int,2> {
  char *__repr__() {
    static char tmp[1024];
    sprintf(tmp, "[%d, %d]", $self->x(), $self->y());
    return tmp;
  }
};
%extend MMVII::cPtxd<double,2> {
  char *__repr__() {
    static char tmp[1024];
    sprintf(tmp, "[%f, %f]", $self->x(), $self->y());
    return tmp;
  }
};
%extend MMVII::cPtxd<double,3> {
  char *__repr__() {
    static char tmp[1024];
    sprintf(tmp, "[%f, %f, %f]", $self->x(), $self->y(), $self->z());
    return tmp;
  }
};



