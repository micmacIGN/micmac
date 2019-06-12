#ifndef  _MMVII_Enums_H_
#define  _MMVII_Enums_H_

namespace MMVII
{

/** \file MMVII_enums.h
    \brief Contains (almost) all enums

    This file contains definition of all global enums,
    I put all in a single file :
    (1) I think it is convenient for searching 
    (2) More important, this file may be generated automatically in case I 
    decide to use code genreration for the enum2string functionnality
*/


/// Defined in MMVII_sys.h
enum class eSYS;


/// Type for Semantic of Arg 2007
enum class eTA2007
           {
            // ---------- Printed --------------
                DirProject,    ///< Exact Dir of Proj
                FileDirProj,   ///< File of Dir Proj
                MPatIm,        ///< Major PaternIm => "" or "0" in sem for set1, "1" or other for set2
                FFI,           ///< File Filter Interval
            // ---------- Not Printed -----------
            // !!!!! Common must be first UNPRINTED  !!!
                Common,        ///< Parameter  Common to all commands
                Internal,      ///< Reserved to internall use by MMVII
                HDV,           ///< Has Default Value, will be printed on help
                eNbVals        ///< Tag for number of value
           };

/// Appli Features
enum class eApF
           {
               Project, ///< Project Managenent
               Test,    ///< Test
               Ori,     ///< Orientation
               Match,   ///< Dense Matching
               TieP,    ///< Tie-Point processing
               TiePLearn,    ///< Tie-Point processing  - Learning step
               Perso,   ///< Personnal
               eNbVals  ///< Tag for number of value
           };

/// Appli Data Type
enum class eApDT
           {
              Ori,    ///< Orientation
              TieP,   ///< Tie Points
              Ply,    ///< Ply file
              None,     ///< Nothing 
              Console,  ///< Console 
              Xml,      ///< Xml-files
              FileSys,      ///< Input is the file system (list of file)
              eNbVals       ///< Tag for number of value
           };


/// Type of set creation
enum class eTySC    
           {
              NonInit,   ///< With Ptr Null
              US,        ///< With unordered set
              eNbVals    ///< Tag for number of value
           };

/// Type of operator

enum class eOpAff
           {
               ePlusEq,   ///< +=
               eMulEq,    ///< *=
               eMinusEq,  ///< *=
               eEq,       ///< =
               eReset,    ///< =0
               eNbVals    ///< Tag for number of value
           };

/// Type of Warning
enum class eTyW
           {
               eWLineAndCart  ///< In EditRel, Circ in mod Cart
           };


/// Type of User's Error
enum class eTyUEr
           {
              eCreateDir,
              eRemoveFile,
              eEmptyPattern,
              eBadFileSetName,
              eBadFileRelName,
              eOpenFile,
              eWriteFile,
              eReadFile,
              eBadBool,
              eBadEnum,
              eMulOptParam,
              eBadOptParam,
              eInsufNbParam,
              eIntervWithoutSet,
              eTooBig4NbDigit,
              eNoModeInEditRel,
              eMultiModeInEditRel,
              e2PatInModeLineEditRel,
              eParseError,
              eUnClassedError,
              eNbVals
           };

/// 
enum class eTyUnitTime
           {
              eUT_Sec,
              eUT_Min,
              eUT_Hour,
              eUT_Day,
              eNbVals
           };

enum class eTyNums
           {
              eTN_INT1,
              eTN_U_INT1,
              eTN_INT2,
              eTN_U_INT2,
              eTN_INT4,
              eTN_U_INT4,
              eTN_INT8,
              eTN_REAL4,
              eTN_REAL8,
              eTN_REAL16,
              eNbVals
           };

enum class eModeInitImage
           {
               eMIA_Rand,        ///< Rand  in [0..1]
               eMIA_RandCenter,        ///< Rand  in [-1 1]
               eMIA_Null,
               eMIA_MatrixId,    ///<  Only for square  Matrix  : Identite, 
               eMIA_NoInit
           };

enum class eTyInvRad
           {
               eTVIR_ACGR,
               eTVIR_ACGT,
               eTVIR_ACR0,
               eTVIR_Curve,
               eNbVals
           };

/**  Type of eigen decomposition in matrix algebra */
enum class eTyEigenDec
           {
               eTED_PHQR, // Pivot Householder QR
               eTED_LLDT, // Cholesky with robust pivoting
               eNbVals
           };


const std::string & E2Str(const eTySC &);         
const std::string & E2Str(const eOpAff &);         
const std::string & E2Str(const eTA2007 &);         
const std::string & E2Str(const eTyUEr &);         
const std::string & E2Str(const eTyNums &);         
const std::string & E2Str(const eTyInvRad &);         

template <class Type> const Type & Str2E(const std::string &); 
template <class Type> std::string   StrAllVall();
template <class Type> std::vector<Type> SubOfPat(const std::string & aPat,bool AcceptEmpty=false);



};

#endif  //  _MMVII_Enums_H_
