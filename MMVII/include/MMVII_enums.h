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
            // !!!!! Shared must be first UNPRINTED  !!! because of test in Name4Help()
                Shared,        ///< Parameter  Shared by many (several) command
                Global,        ///< Parameter  Common to all commands
                Internal,      ///< Reserved to internall use by MMVII
                HDV,           ///< Has Default Value, will be printed on help
                ISizeV,        ///< Interval size vect, print on help
                eNbVals        ///< Tag for number of value
           };

/// Appli Features
enum class eApF
           {
               Project, ///< Project Managenent
               Test,    ///< Test
               ImProc,  ///< Image processing
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
              Image,   ///< Tie Points
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
              eBadDimForPt,
              eBadSize4Vect,
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

/**  Mode of recall sub processes  */
enum class eTyModeRecall
           {
               eTMR_Inside, ///< Recall in the same process
               eTMR_Serial, ///< Recall by sub-process in serial
               eTMR_Parall, ///< Recall by sub-process in parallel
               eNbVals      ///< Tag End of Vals
           };

/**  Type of Tie P, independently of the sens (Min/Max)   */
enum class eTyPyrTieP
{
     eTPTP_Init,     ///< Original Pyramid
     eTPTP_LaplG,    ///< Laplacien of Gaussian
     eTPTP_Corner,   ///< Corner 
     eTPTP_OriNorm,  ///< Original normalized
     eNbVals         ///< Tag End of Vals
};

/**  Value of optional parameter, that are shared, and must be activated */
enum class eSharedPO
{
     eSPO_CarPO,     ///< Caracterestic Points Out
     eSPO_CarPI,     ///< Caracterestic Points Input
     eSPO_TiePO,     ///< Caracterestic Points Out
     eSPO_TiePI,     ///< Caracterestic Points Input
     eNbVals         ///< Tag End of Vals
};

/** Mode of normalization for orientation */
enum class eModeNormOr
{
     eMNO_MaxGray,     ///< Max of gray value
     eMNO_MaxGradT,    ///< Tangential gradient
     eMNO_MaxGradR,    ///< Radial gradient
     eNbVals         ///< Tag End of Vals
};

/** Mode of output file for Pts Carac */
enum class eModeOutPCar
{
     eMOPC_Image,     ///< Mode to save images
     eMNO_PCarV1,     ///< Mode to save PCar to be seen in MMV1
     eMNO_BinPCarV2,  ///< Mode to save PCar to be seen for MMVII, bin mod
     eMNO_XmlPCarV2,  ///< Mode to save PCar to be seen for MMVII, xml mode (not sure needed ?)
     eNbVals          ///< Tag End of Vals
};


const std::string & E2Str(const eTySC &);         
const std::string & E2Str(const eOpAff &);         
const std::string & E2Str(const eTA2007 &);         
const std::string & E2Str(const eTyUEr &);         
const std::string & E2Str(const eTyNums &);         
const std::string & E2Str(const eTyInvRad &);         
const std::string & E2Str(const eTyPyrTieP &);         

template <class Type> const Type & Str2E(const std::string &); 
template <class Type> std::string   StrAllVall();
template <class Type> std::vector<Type> SubOfPat(const std::string & aPat,bool AcceptEmpty=false);



};

#endif  //  _MMVII_Enums_H_
