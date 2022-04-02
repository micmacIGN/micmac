%define DOCSTRING
"The `mmv2` module gives access to many MicMac v2 classes,
to read MicMac files and use its functions"
%enddef

%module (docstring=DOCSTRING) mmv2
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
%include "numpy.i"
%init %{
import_array();
%}

#define ElSTDNS  std::
#define ElTmplSpecNull template <>
%include <std_string.i>
%include <std_vector.i>
%include <std_map.i>
%include <std_list.i>
%include <cpointer.i>
%include <stl.i>


//----------------------------------------------------------------------


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
//add typemaps
%include tmp/typemaps.i
%include tmp/rename_nonref.i

//----------------------------------------------------------------------
//add .value(), new_... etc. to manipulate pointers
%pointer_class(unsigned char, ucharp);
%pointer_class(char, charp);
%pointer_class(unsigned short, ushortp);
%pointer_class(short, shortp);
%pointer_class(unsigned int, uintp);
%pointer_class(int, intp);
%pointer_class(unsigned long, ulongp);
%pointer_class(long, longp);
%pointer_class(bool, boolp);
%pointer_class(float, floatp);
%pointer_class(double, doublep);

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
%ignore MMVII::cPtxd<int,3>::cPtxd(const int & x);
%ignore MMVII::cPtxd<int,3>::cPtxd(const int & x,const int &y);
%ignore MMVII::cPtxd<int,3>::cPtxd(const int & x,const int &y,const int &z,const int &t);
%ignore MMVII::cPtxd<int,3>::t;
//not implemented
%ignore MMVII::cPtxd::Col;
%ignore MMVII::cPtxd::Line;
%ignore MMVII::cPtxd::ToVect;
%ignore MMVII::cPtxd<int,2>::ToStdVector;
%ignore MMVII::cPtxd<int,3>::ToStdVector;
%ignore MMVII::cPtxd<double,2>::ToStdVector;
%ignore MMVII::cPtxd<double,3>::ToStdVector;
%ignore MMVII::cPtxd::FromVect;
%ignore MMVII::cPtxd<int,2>::FromStdVector;
%ignore MMVII::cPtxd<int,3>::FromStdVector;
%ignore MMVII::cPtxd<double,2>::FromStdVector;
%ignore MMVII::cPtxd<double,3>::FromStdVector;
//ignore non-const overloading to get direct access to simple types
//(the functions will be read-only, we add setter methods in extend part)
%include tmp/ignore_nonconst_overloading.i
%ignore MMVII::cPtxd<double,2>::x() ;
%ignore MMVII::cPtxd<double,2>::y() ;
%ignore MMVII::cPtxd<double,2>::PtRawData() ;
%ignore MMVII::cPtxd<int,2>::x() ;
%ignore MMVII::cPtxd<int,2>::y() ;
%ignore MMVII::cPtxd<int,2>::PtRawData();
%ignore MMVII::cPtxd<double,3>::x();
%ignore MMVII::cPtxd<double,3>::y();
%ignore MMVII::cPtxd<double,3>::z();
%ignore MMVII::cPtxd<double,3>::PtRawData();
%ignore MMVII::cPtxd<int,3>::x();
%ignore MMVII::cPtxd<int,3>::y();
%ignore MMVII::cPtxd<int,3>::z();
%ignore MMVII::cPtxd<int,3>::PtRawData();

//ignore const overloading
/*%ignore MMVII::cIm2D<tU_INT1>::DIm() const;
%ignore MMVII::cIm2D<tREAL4>::DIm() const;
%ignore MMVII::cDataIm2D<tU_INT1>::ExtractRawData2D() const;
%ignore MMVII::cDataIm2D<tU_INT1>::GetLine(int) const;
%ignore MMVII::cDataIm2D<tREAL4>::ExtractRawData2D() const;
%ignore MMVII::cDataIm2D<tREAL4>::GetLine(int) const;*/
//remove warnings
%ignore MMVII::cPtxd::operator[];
%ignore MMVII::cPtxd::operator[] const;

//rename overloaded methods to avoid shadowing
//here params must appread exactly as in source (no MMVII::...)
%rename(ToI2) ToI(const cPt2dr &);
%rename(ToI3) ToI(const cPt3dr &);
%rename(ToR2) ToR(const cPt2di &);
%rename(ToR3) ToR(const cPt3di &);
%rename(E2Str_TySC)          MMVII::E2Str(const eTySC &);
%rename(E2Str_OpAff)         MMVII::E2Str(const eOpAff &);
%rename(E2Str_TA2007)        MMVII::E2Str(const eTA2007 &);
%rename(E2Str_TyUEr)         MMVII::E2Str(const eTyUEr &);
%rename(E2Str_TyNums)        MMVII::E2Str(const eTyNums &);
%rename(E2Str_TyInvRad)      MMVII::E2Str(const eTyInvRad &);
%rename(E2Str_TyPyrTieP)     MMVII::E2Str(const eTyPyrTieP &);
%rename(E2Str_ModeEpipMatch) MMVII::E2Str(const eModeEpipMatch &);

//----------------------------------------------------------------------
//classes to export
%nodefaultctor;
%include "api/api_mmv2.h"
%include tmp/h_to_include.i

//HERE typedefs are mandatory. include MMVII_nums.h is not working...
typedef float       tREAL4;
typedef double      tREAL8;
typedef long double tREAL16;
typedef signed char  tINT1;
typedef signed short tINT2;
typedef signed int   tINT4;
typedef long int     tINT8;
typedef unsigned char  tU_INT1;
typedef unsigned short tU_INT2;
typedef unsigned int   tU_INT4;
typedef int    tStdInt;  ///< "natural" int
typedef double tStdDouble;  ///< "natural" int

//templates have to be named to be exported
%template(cIm2Du1) MMVII::cIm2D<tU_INT1>;
%template(cIm2Dr4) MMVII::cIm2D<tREAL4>;
%template(cDataIm2Du1) MMVII::cDataIm2D<tU_INT1>;
%template(cDataIm2Dr4) MMVII::cDataIm2D<tREAL4>;
//%template(cBox2di) MMVII::cTplBox<int,2>;

//%template(Pt1dr) MMVII::cPtxd<double,1>   ;
//%template(Pt1di) MMVII::cPtxd<int,1>      ;
//%template(Pt1df) MMVII::cPtxd<float,1>    ;
%template(Pt2dr) MMVII::cPtxd<double,2>   ;

%template(Pt2di) MMVII::cPtxd<int,2>      ;
//%template(Pt2df) MMVII::cPtxd<float,2>    ;
%template(Pt3dr) MMVII::cPtxd<double,3>   ;
%template(Pt3di) MMVII::cPtxd<int,3>      ;
//%template(Pt3df) MMVII::cPtxd<float,3>    ;

%template(tU_INT1Vector) std::vector<tU_INT1>;
%template(IntVector)     std::vector<int>;
%template(DoubleVector)  std::vector<double>;
%template(FloatVector)   std::vector<float>;
%template(StringVector)  std::vector<std::string>;

%template(cWhitchMinIntDouble) MMVII::cWhitchMin<int, double>;
%template(cAimePCarVector)     std::vector<MMVII::cAimePCar>;

//----------------------------------------------------------------------
//run on import
%pythoncode %{
import copy
print("MicMacV2 Python3 API")
mmv2_init();
%}


//----------------------------------------------------------------------
//add functions to classes
%extend MMVII::cPtxd<int,2> {
  char *__repr__() {
    static char tmp[1024];
    sprintf(tmp, "Pt [%d, %d]", $self->x(), $self->y());
    return tmp;
  }
  void setX(int x) { $self->x()=x; }
  void setY(int y) { $self->y()=y; }
}
%extend MMVII::cPtxd<double,2> {
  /*char *__repr__() {
    static char tmp[1024];
    sprintf(tmp, "Pt [%f, %f]", $self->x(), $self->y());
    return tmp;
  }*/
  void setX(double x) { $self->x()=x; }
  void setY(double y) { $self->y()=y; }
}
%extend MMVII::cPtxd<double,3> {
  char *__repr__() {
    static char tmp[1024];
    sprintf(tmp, "Pt [%f, %f, %f]", $self->x(), $self->y(), $self->z());
    return tmp;
  }
  void setX(double x) { $self->x()=x; }
  void setY(double y) { $self->y()=y; }
  void setZ(double z) { $self->z()=z; }
}
%extend MMVII::cPtxd<int,3> {
  char *__repr__() {
    static char tmp[1024];
    sprintf(tmp, "Pt [%d, %d, %d]", $self->x(), $self->y(), $self->z());
    return tmp;
  }
  void setX(int x) { $self->x()=x; }
  void setY(int y) { $self->y()=y; }
  void setZ(int z) { $self->z()=z; }
}

%extend MMVII::cIm2D<tU_INT1> {
  char *__repr__() {
    static char tmp[1024];
    sprintf(tmp, "Im2D [%d, %d]", $self->DIm().SzX(), $self->DIm().SzY());
    return tmp;
  }
}

%extend MMVII::cDataIm2D<tU_INT1> {
  char *__repr__() {
    static char tmp[1024];
    sprintf(tmp, "DataIm2D [%d, %d]", $self->SzX(), $self->SzY());
    return tmp;
  }
}


//must redefine virtual functions to access base class methods,
//when base class is abstract and template?
%extend MMVII::cDataIm2D<tU_INT1> {
  %feature("autodoc", "Returns a copy of image data") getRawData;
  std::vector<tU_INT1> getRawData() {
    int size = $self->SzX()*$self->SzY();
    tU_INT1* data = $self->RawDataLin();
    std::vector<tU_INT1> out(data ,data + size);
    return out;
  }
  //here tU_INT1* IN_ARRAY2 is not recognized
  %feature("autodoc", "Set of image data from uin8 array") setRawData;
  void setRawData(unsigned char* IN_ARRAY2, int DIM1, int DIM2) {
    $self->Resize( MMVII::cPtxd<int,2>(0,0), MMVII::cPtxd<int,2>(DIM2,DIM1) );
    for (long i=0;i<DIM1*DIM2;i++)
        $self->RawDataLin()[i] = IN_ARRAY2[i];
  }
}
%extend MMVII::cDataIm2D<tREAL4> {
  %feature("autodoc", "Returns a copy of image data") getRawData;
  std::vector<tREAL4> getRawData() {
    int size = $self->SzX()*$self->SzY();
    tREAL4* data = $self->RawDataLin();
    std::vector<tREAL4> out(data ,data + size);
    return out;
  }
  //here tREAL4* IN_ARRAY2 is not recognized
  %feature("autodoc", "Set of image data from float32 array") setRawData;
  void setRawData(float* IN_ARRAY2, int DIM1, int DIM2) {
    $self->Resize( MMVII::cPtxd<int,2>(0,0), MMVII::cPtxd<int,2>(DIM2,DIM1) );
    for (long i=0;i<DIM1*DIM2;i++)
        $self->RawDataLin()[i] = IN_ARRAY2[i];
  }
}

%include tmp/return_nonref.i

//add toArray methods to images to take care of std::vector to np.array
%pythoncode %{
import numpy as np
def toArray_uint8(self):
    """Convert to numpy uint8 array"""
    return np.array(self.getRawData(), dtype=np.uint8).reshape(self.SzY(), self.SzX())
def toArray_float32(self):
    """Convert to numpy float32 array"""
    return np.array(self.getRawData(), dtype=np.float32).reshape(self.SzY(), self.SzX())
cDataIm2Du1.toArray = toArray_uint8
cDataIm2Dr4.toArray = toArray_float32
%}

