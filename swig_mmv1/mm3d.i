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
#include "general/geom_vecteur.h"
#include "private/files.h"
#include "XML_GEN/SuperposImage.h"
#include "XML_GEN/ParamChantierPhotogram.h"
typedef ElAffin2D tOrIntIma ; //mandatory because only declared in cBasicGeomCap3D in general/photogram.h?
#include "api/TpPPMD.h"
#include "../src/uti_phgrm/NewOri/NewOri.h"
//#include "XML_GEN/xml_gen2_mmByp.h"
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
    %template(cXml_OneTripletList) list< cXml_OneTriplet >;
    %template(ElSeg3DVector)  vector<ElSeg3D>;
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
//classes to export
%include "api/api_mm3d.h"
%include "api/TpPPMD.h"
%include "general/util.h"
%include "general/bitm.h"
%include "general/ptxd.h"
%include "general/geom_vecteur.h"
%include "general/photogram.h"
%include "../src/uti_phgrm/NewOri/NewOri.h"
//%include "XML_GEN/xml_gen2_mmByp.h"
//%include "private/files.h" //not working for now
void MakeFileXML(const cSauvegardeNamedRel & anObj,const std::string & aName,const std::string & aTagEnglob=""); //just one version of MakeFileXML for now

//%include "XML_GEN/SuperposImage.h"
class cXml_TopoTriplet
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cXml_TopoTriplet & anObj,cElXMLTree * aTree);


        std::list< cXml_OneTriplet > & Triplets();
        const std::list< cXml_OneTriplet > & Triplets()const ;
    private:
        std::list< cXml_OneTriplet > mTriplets;
};
class cXml_OneTriplet
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cXml_OneTriplet & anObj,cElXMLTree * aTree);


        std::string  Name1();
        //const std::string & Name1()const ;

        std::string Name2();
        //const std::string & Name2()const ;

        std::string Name3();
        //const std::string & Name3()const ;
    private:
        std::string mName1;
        std::string mName2;
        std::string mName3;
};

class cXml_Ori3ImInit
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cXml_Ori3ImInit & anObj,cElXMLTree * aTree);


        cXml_Rotation & Ori2On1();
        const cXml_Rotation & Ori2On1()const ;

        cXml_Rotation & Ori3On1();
        const cXml_Rotation & Ori3On1()const ;

        int  NbTriplet();
        //const int & NbTriplet()const ;

        double & ResiduTriplet();
        const double & ResiduTriplet()const ;

        double  BSurH();
        //const double & BSurH()const ;

        Pt3dr & PMed();
        const Pt3dr & PMed()const ;

        cXml_Elips3D & Elips();
        const cXml_Elips3D & Elips()const ;
    private:
        cXml_Rotation mOri2On1;
        cXml_Rotation mOri3On1;
        int mNbTriplet;
        double mResiduTriplet;
        double mBSurH;
        Pt3dr mPMed;
        cXml_Elips3D mElips;
};
class cXml_Rotation
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cXml_Rotation & anObj,cElXMLTree * aTree);


        cTypeCodageMatr  Ori();
        //const cTypeCodageMatr & Ori()const ;

        Pt3dr  Centre();
        //const Pt3dr & Centre()const ;
    private:
        cTypeCodageMatr mOri;
        Pt3dr mCentre;
};
class cTypeCodageMatr
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cTypeCodageMatr & anObj,cElXMLTree * aTree);


        Pt3dr  L1();
        //const Pt3dr & L1()const ;

        Pt3dr  L2();
        //const Pt3dr & L2()const ;

        Pt3dr  L3();
        //const Pt3dr & L3()const ;

        cTplValGesInit< bool > & TrueRot();
        const cTplValGesInit< bool > & TrueRot()const ;
    private:
        Pt3dr mL1;
        Pt3dr mL2;
        Pt3dr mL3;
        cTplValGesInit< bool > mTrueRot;
};




//%include "XML_GEN/ParamChantierPhotogram.h"

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

