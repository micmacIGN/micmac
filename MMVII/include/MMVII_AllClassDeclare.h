#ifndef  _MMVII_AllClassDeclare_H_
#define  _MMVII_AllClassDeclare_H_

#undef _OPENMP

// Header standard c++
#include "memory.h"
#include <memory>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <typeinfo>
#include <vector>
#include <list>
#include <map>
#include <ctime>
#include <chrono>
#include <optional>
#include <cmath>
#include <cstdint>

//========== LIB EXTEN==============


//===========================================

namespace MMVII
{
extern bool MMVII_IN_DEBUG;



/** \file MMVII_AllClassDeclare.h
    \brief Contains declaration  of all class

   As sooner or later many class require a forrward declaration,
 I think it's more readable to systematically declare everything
 here.

*/

enum class eTyUEr;
enum class eTyUnitAngle;

typedef float       tREAL4;
typedef double      tREAL8;
typedef long double tREAL16;

typedef int8_t  tINT1;
typedef int16_t tINT2;
typedef int32_t tINT4;
typedef int64_t tINT8;



typedef uint8_t  tU_INT1;
typedef uint16_t tU_INT2;
typedef uint32_t tU_INT4;
typedef uint64_t tU_INT8;


typedef int    tStdInt;  ///< "natural" int
typedef unsigned int    tStdUInt;  ///< "natural" int
typedef double tStdDouble;  ///< "natural" int



// MMVII_memory.h :  Memory

class  cMemState; // Memory state
class  cMemManager; // Allocator/desallocator tracking memory state
class  cMemCheck;   // Class calling cMemManager for allocation
template<class Type> class cGestObjetEmpruntable;


// MMVII_util.h :  Util
class cCarLookUpTable;
class cMMVII_Ofs ;
class cMMVII_Ifs ;
class cMultipleOfs ;
class cMMVII_Duration;
class cSetIntDyn;
class cSetIExtension;
class cParamRansac;
class cMMVII_Ifs;
class cMMVII_Ofs;
// cMultipleOfs& StdOut(); /// Call the ostream of cMMVII_Appli if exist (else std::cout)
// cMultipleOfs& HelpOut();
// cMultipleOfs& ErrOut();

// MMVII_util_tpl.h

template <class Type> class cExtSet ;
template <class Type> class cSelector ;
template <class Type> class cDataSelector ;
template <class Type> class cOrderedPair ;

typedef cSelector<std::string>      tNameSelector;
typedef cExtSet<std::string>        tNameSet;
typedef cOrderedPair<std::string>   tNamePair; ///< Order does not matter
typedef std::pair<std::string,std::string>  tNameOCple;  ///< Order matters
typedef cExtSet<tNamePair>          tNameRel;



//===================== MMVII_Ptxd.h ===========================
template <class Type,const int Dim> class cPtxd;
template <class Type,const int Dim> class cTplBox;

    ///  1 dimension specialization
typedef cPtxd<double,1>  cPt1dr ;
typedef cPtxd<int,1>     cPt1di ;
typedef cPtxd<float,1>   cPt1df ;

    ///  2 dimension specialization
typedef cPtxd<tREAL16,2> cPt2dLR ;
typedef cPtxd<double,2>  cPt2dr ;
typedef cPtxd<int,2>     cPt2di ;
typedef cPtxd<float,2>   cPt2df ;
    ///  3 dimension specialization
typedef cPtxd<tREAL16,3> cPt3dLR ;
typedef cPtxd<double,3>  cPt3dr ;
typedef cPtxd<int,3>     cPt3di ;
typedef cPtxd<float,3>   cPt3df ;

typedef cPtxd<double,4>  cPt4dr ;
typedef cPtxd<int,4>  cPt4di ;

typedef cTplBox<int,2>  cBox2di;
typedef cTplBox<double,2>  cBox2dr;
typedef cTplBox<int,3>  cBox3di;
typedef cTplBox<double,3>  cBox3dr;

// Later replace cPt3dr  ...  by tPt3dr , more coherent with other notation ...

typedef cPtxd<int,2>     tPt2di ;
typedef cPtxd<int,3>     tPt3di ;
typedef cPtxd<double,2>  tPt2dr ;
typedef cPtxd<double,3>  tPt3dr ;

// MMVII_Bench.h

// cMMVII_Appli.h
// class cSetName;
class cArgMMVII_Appli;
class cSpecMMVII_Appli;
class cMMVII_Ap_NameManip;
class cMMVII_Ap_CPU;
class cMMVII_Appli ;
class cExplicitCopy; ///<  Fake class use for add X(const X&) explicit with  X(cExplicitCopy,const X&)
class cParamExeBench;


// MMVII_Stringifier.h

class  cSpecOneArgCL2007 ;
class cCollecSpecArg2007;

class cAuxAr2007;
class cAr2007;
class cRawData4Serial;

// MMVII_Images.h
template <const int Dim>  class cPixBoxIterator;
template <const int Dim>  class cPixBox;
template <const int Dim>  class cBorderPixBox ;
template <const int Dim>  class cBorderPixBoxIterator ;


template <const int Dim> class cDataGenUnTypedIm ;
template <class Type,const int Dim> class cDataTypedIm ;
class cDataFileIm2D ;
template <class Type>  class cDataIm2D  ;
template <class Type>  class cIm2D  ;
template <class Type>  class cDataIm3D  ;
template <class Type>  class cIm3D  ;
template <class Type>  class cDataIm1D  ;
template <class Type>  class cIm1D  ;

template <class TypeObj,class TypeLayer>  class cLayerData3D ;
template <class TypeObj,class TypeLayer>  class cLayer3D ;

class cInterpolator1D ;
class cDiffInterpolator1D ;



// MMVII_Matrix.h
template <class Type> class  cDenseVect;
class cStdStatRes;


//  MMVII_Triangles.h
template <class Type,const int Dim> class cTriangle ;
typedef   cTriangle<tREAL8,2>  tTri2dr;
typedef   cTriangle<tREAL8,3>  tTri3dr;

template <class Type,const int Dim> class cTriangulation ;
typedef cTriangulation<tREAL8,2>  tTriangul2dr;
typedef cTriangulation<tREAL8,3>  tTriangul3dr;






// MMVII_Mappings.h
template <class Type,const int Dim> class cDataBoundedSet ;
template <class Type,const int DimIn,const int DimOut> class cMapping;
template <class Type,const int DimIn,const int DimOut> class cDataMapping;
template <class Type,const int Dim> class cDataInvertibleMapping ;// :  public cDataMapping<Type,Dim,Dim>
template <class Type,const int Dim> class cDataIterInvertMapping ;// :  public cDataInvertibleMapping<Type,Dim>
template <class Type,const int Dim> class cDataIIMFromMap ; // : public cDataIterInvertMapping<Type,Dim>

template <class Type,const int Dim> class cMappingIdentity ; // :  public cDataMapping<Type,Dim,Dim>
template <class Type,const int DimIn,const int DimOut> class cDataMapCalcSymbDer ;// : public cDataMapping<Type,DimIn,DimOut>
template <class cMapElem> class cInvertMappingFromElem ;
    // :  public cDataInvertibleMapping<typename cMapElem::TheType,cMapElem::TheDim>
template <class Type,const int  DimIn,const int DimOut> class cLeastSqComputeMaps;
template <class Type,const int DimIn,const int DimOut> class cLeastSqCompMapCalcSymb;

template <class Type,const int Dim> class cBijAffMapElem;

class cSysCoordV2;
class cChangSysCoordV2 ;

// MMVII_Geom3D.h
template <class Type> class cRotation3D;
template <class Type> class cIsometry3D;
template <class Type> class cSimilitud3D;
template <class Type> class cTriangulation3D;

// MMVII_ZBuffer.h
class cCountTri3DIterator ;
class cCountTri3DIterator ;
class cMeshTri3DIterator;
enum class eZBufRes;
enum class eZBufModeIter;
struct cResModeSurfD;
class  cZBuffer;


// MMVII_Sensor.h

struct cPair2D3D;
struct cWeightedPair2D3D;
struct cSet2D3D;
class  cSensorImage;
class  cDataPixelDomain ;
class  cPixelDomain;
class  cSensorCamPC;
class  cPhotogrammetricProject;
class  cSIMap_Ground2ImageAndProf ;
class  cPerspCamIntrCalib;
class  cMetaDataImage;
class  cElemCamDataBase;
class  cCamDataBase;


// MMVII_MeasuresIm.h

struct cPair2D3D;
struct cSet2D3D;
class cMesIm1Pt;
class cSetMesPtOf1Im;

class cMesIm1Pt;
class cSetMesPtOf1Im;
class cMes1Gnd3D;
class cSetMesGnd3D;

class cMultipleImPt;
class cSetMesGndPt;

class cVecTiePMul;
class cTiePMul;


// MMVII_Radiom.h
class cImageRadiomData;  ///< store data used for radiometric equalisation
class cFusionIRDSEt;     ///< store fusion

      // radiometric sensor calibration
class cCalibRadiomSensor ;    // base class for representing a calib radiom of a camera
class cRadialCRS ;            // class for "standard" model : radial function

      // radiometric  image calibration
class cCalibRadiomIma ;     //  base class  for representing a calib
class cCalRadIm_Pol ;       // class for standar model :   Cste + Sensor

     // ================  Ground & Image measures of points ========
    
      // without reference to name  of point or image
struct  cPair2D3D;   //  correspondance between a 2D point and a 3D Poitnt
struct  cWeightedPair2D3D ;  // cPair2D3D + a Weight
struct cSet2D3D ;            // Set of cWeightedPair2D3D

class cHomogCpleIm;
class cSetHomogCpleIm;

class cSetMesGndPt;
class cMultipleImPt;
class cSetMesGnd3D ;
class cSetMesPtOf1Im ;
class cMesIm1Pt;

      // Line detection
class cOneLineAntiParal;
class cLinesAntiParal1Im;

// Tiling index
template <class Type>  class  cTiling ;


//   MMVII_BlocRig.h
    //  RIGIDBLOC  
class cSetSensSameId;     
class cBlocMatrixSensor;  
class cDataBlocCam;      
class cBlocOfCamera;

//  Clino
class cOneCalibRelClino;  
class cOneCalibClino;    
class cCalibSetClino;   
class cBA_Clino;
class cOneMesureClino;
class cSetMeasureClino;


//   sys co

class cSysCo;
class cChangeSysCo;
typedef std::shared_ptr<cSysCo>       tPtrSysCo;

class cTabulateGrad;

};

#endif  //  _MMVII_AllClassDeclare_H_
