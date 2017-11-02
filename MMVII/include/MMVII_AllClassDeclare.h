#ifndef  _MMVII_AllClassDeclare_H_
#define  _MMVII_AllClassDeclare_H_

// Sys
enum class eSYS;

//Error

// MMVII_memory.h :  Memory

class  cMemState; // Memory state
class  cMemManager; // Allocator/desallocator tracking memory state
class  cMemCheck;   // Class calling cMemManager for allocation
template<class Type> class cGestObjetEmpruntable;


// MMVII_util.h :  Util
class cCarLookUpTable;
class cMMVII_Ofs ;
class cMMVII_Ifs ;


// MMVII_Ptxd.h
template <class Type,const int Dim> class cPtxd;
template <class Type> class cPt1d ;
template <class Type> class cPt2d ;

// MMVII_Bench.h

// cMMVII_Appli.h
class cArgMMVII_Appli;
class cSpecMMVII_Appli;
class cMMVII_Ap_NameManip;
class cMMVII_Ap_CPU;
class cMMVII_Appli ;

// MMVII_Stringifier.h

class  cOneArgCL2007 ;
class cCollecArg2007;

class cAuxAr2007;
class cAr2007;



#endif  //  _MMVII_AllClassDeclare_H_
