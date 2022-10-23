#ifndef  _MMVII_AllClassDeclare_H_
#define  _MMVII_AllClassDeclare_H_

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

//========== LIB EXTEN==============


//===========================================

namespace MMVII
{


/** \file MMVII_AllClassDeclare.h
    \brief Contains declaration  of all class

   As sooner or later many class require a forrward declaration,
 I think it's more readable to systematically declare everything
 here.

*/

enum class eTyUEr;

// MMVII_memory.h :  Memory

class  cMemState; // Memory state
class  cMemManager; // Allocator/desallocator tracking memory state
class  cMemCheck;   // Class calling cMemManager for allocation
template<class Type> class cGestObjetEmpruntable;


// MMVII_util.h :  Util
class cCarLookUpTable;
class cMMVII_Ofs ;
class cMMVII_Ifs ;

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



// MMVII_Ptxd.h
template <class Type,const int Dim> class cPtxd;

// MMVII_Bench.h

// cMMVII_Appli.h
// class cSetName;
class cArgMMVII_Appli;
class cSpecMMVII_Appli;
class cMMVII_Ap_NameManip;
class cMMVII_Ap_CPU;
class cMMVII_Appli ;
class cExplicitCopy; ///<  Fake class use for add X(const X&) explicit with  X(cExplicitCopy,const X&)


// MMVII_Stringifier.h

class  cSpecOneArgCL2007 ;
class cCollecSpecArg2007;

class cAuxAr2007;
class cAr2007;

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



// MMVII_Matrix.h
template <class Type> class  cDenseVect;

template <class Type,const int Dim> class cTriangle ;

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

// MMVII_Geom3D.h
template <class Type> class cRotation3D;
template <class Type> class cIsometrie3D;
template <class Type> class cSimilitud3D;
template <class Type> class cTriangulation3D;



};

#endif  //  _MMVII_AllClassDeclare_H_
