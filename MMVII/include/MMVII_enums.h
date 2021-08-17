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
                FileImage,     ///< File containing an image
                MPatFile,      ///< Major PaternIm => "" or "0" in sem for set1, "1" or other for set2
                FFI,           ///< File Filter Interval
            // !!!!! AddCom must be last UNPRINTED  !!! because of test in Name4Help()
                AddCom,        ///< Not an attribute, used to embed additionnal comment in Help mode
            // ---------- Not Printed -----------
            // !!!!! Shared must be first UNPRINTED  !!! because of test in Name4Help()
                Shared,        ///< Parameter  Shared by many (several) command
                Global,        ///< Parameter  Common to all commands
                Internal,      ///< Reserved to internall use by MMVII
                Tuning,        ///< Used for testing/tuning command but not targeted for user
                HDV,           ///< Has Default Value, will be printed on help
                ISizeV,        ///< Interval size vect, print on help
                eNbVals        ///< Tag for number of value
           };

/// Appli Features
enum class eApF
           {
               ManMMVII,   ///< Managenent of MMVII
               Project,    ///< Project Managenent (user's)
               Test,       ///< Test
               ImProc,     ///< Image processing
               Ori,        ///< Orientation
               Match,      ///< Dense Matching
               TieP,       ///< Tie-Point processing
               TiePLearn,    ///< Tie-Point processing  - Learning step
               Perso,      ///< Personnal
               eNbVals     ///< Tag for number of value
           };

/// Appli Data Type
enum class eApDT
           {
              Ori,    ///< Orientation
              PCar,   ///< Tie Points
              TieP,   ///< Tie Points
              Image,   ///< Image
              Ply,    ///< Ply file
              None,     ///< Nothing 
              ToDef,     ///< still unclassed
              Console,  ///< Console , (i.e printed message have values)
              Xml,      ///< Xml-files
              FileSys,      ///< Input is the file system (list of file)
              Media,      ///< Input is the file system (list of file)
              eNbVals       ///< Tag for number of value
           };

/// Mode of creation of folder
enum class eModeCreateDir    
           {
              DoNoting,      ///< do nothing 
              CreateIfNew,   ///< create the folder and
              CreatePurge,   ///< create the folder and purge it
              ErrorIfExist,  ///< create the folder if new, error else
              eNbVals        ///< Tag for number of value
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
              eBadInt,
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
              eMultiplePostifx,
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
              eTN_UnKnown,  // Used because for MMV1 Bits Type, we need to handle file, but dont have Bits Images
              eNbVals
           };

enum class eModeInitImage
           {
               eMIA_Rand,        ///< Rand  in [0..1]
               eMIA_RandCenter,        ///< Rand  in [-1 1]
               eMIA_Null,  ///<  0 everywere
               eMIA_V1,   ///<   1 everywhere
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


/** Mode "Matcher" callable in DenseMatchEpipGen */

enum class eModeEpipMatch
{
   eMEM_MMV1,  // Mode MicMac V1
   eMEM_PSMNet,// Mode PSMNet
   eMEM_NoMatch,  // Do no match, used for debuging
   eNbVals
};


/** Mode "Padding" callable in DenseMatchEpipGen */

enum class eModePaddingEpip
{
   eMPE_NoPad,  // No padding, natural size MicMac V1
   eMPE_PxPos,  // Padding force positive paralax
   eMPE_PxNeg,  // Padding force negative paralax
   eMPE_SzEq,  //  Centerd padding, size equal
   eNbVals
};

/** Mode  caracteristic for matching */

enum class eModeCaracMatch
{
     // ----------  Census Quantitatif ------------
   eMCM_CQ1,    ///< Census Quantitatif 1 pixel (~ 3x3)
   eMCM_CQ2,    ///< Census Quantitatif 2 pixel (~ 5x5)
   eMCM_CQ3,    ///< Census Quantitatif 3 pixel (~ 7x7)
   eMCM_CQ4,    ///< Census Quantitatif 4 pixel (~ 9x9)
   eMCM_CQA,    ///< Census Quantitatif all window
   eMCM_CQW,    ///< Census Quantitatif all window, weihted
     // ----------  "Basic" Census, not conived it is usefull, but it's popular ------------
   eMCM_Cen1,    ///< Census  1 pixel (~ 3x3)
   eMCM_Cen2,    ///< Census  2 pixel (~ 5x5)
   eMCM_Cen3,    ///< Census  3 pixel (~ 7x7)
   eMCM_Cen4,    ///< Census  4 pixel (~ 9x9)
   eMCM_CenA,    ///< Census  All
   eMCM_CenW,    ///< Census  weighted
     // ----------  Normalized Cross Correlation ------------
   eMCM_Cor1,    ///< Correl 1 pixel (~ 3x3)
   eMCM_Cor2,    ///< Correl 2 pixel (~ 5x5)
   eMCM_Cor3,    ///< Correl 3 pixel (~ 7x7)
   eMCM_Cor4,    ///< Correl 4 pixel (~ 9x9)
   eMCM_CorA,    ///< Correl  all window
   eMCM_CorW,    ///< Correl  all window, weihted
     // ---------- "Corner", select only best average, correl probably   ------------
   eMCM_CornW180,   ///< Corner, 180 of average
   eMCM_CornW90,    ///< Corner, 90  of average
     // ----------  Best/Worst score on a given neighboor  ------------
     // For best we use correl as, being quite strict, if it is good, then is has signification even with optimi best
     // For worst we use CQ as being "laxist" (good with any homogeneous), if has signifcation even with pessimistic worst
   eMCM_WorstCorrel2,   ///< Worst weighted correl on all window <= 2 (5x5) 
   eMCM_WorstCorrel3,   ///< Worst weighted correl on all window <= 3 (7x7)
   eMCM_BestCorrel2,   ///< Best weighted correl on all window <= 2 (5x5) 
   eMCM_BestCorrel3,   ///< Best weighted correl on all window <= 3 (7x7)
   eMCM_BestCorrel4,   ///< Best weighted correl on all window <= 4 (9x9)
   eMCM_BestCorrel5,   ///< Best weighted correl on all window <= 5 (11x11)
          // ----
   eMCM_BestCQ2,    ///< Best Census quantitative on all windows <= 2 (5x5)
   eMCM_BestCQ3,    ///< Best Census quantitative on all windows <= 3 (7x7)
   eMCM_WorstCQ2,    ///< Worst Census quantitative on all windows <= 2 (5x5)
   eMCM_WorstCQ3,    ///< Worst Census quantitative on all windows <= 3 (7x7)
   eMCM_WorstCQ4,    ///< Worst Census quantitative on all windows <= 4 (9x9)
   eMCM_WorstCQ5,    ///< Worst Census quantitative on all windows <= 5 (11x11)

     // ----------  Using gray level  ------------
   eMCM_DifGray,    ///< Different of gray Gray Level ,
   eMCM_MinGray,    ///<  Min of 2 Gray Level , may indicate fiability
   eMCM_MinStdDev1,  ///< Min a both std dev
   eMCM_MinStdDev2,  ///< Min a both std dev
   eMCM_MinStdDev3,  ///< Min a both std dev
   eMCM_MinStdDev4,  ///< Min a both std dev
   eMCM_MinStdDevW,  ///< Min a both std dev


   eNbVals
};


const std::string & E2Str(const eTySC &);         
const std::string & E2Str(const eOpAff &);         
const std::string & E2Str(const eTA2007 &);         
const std::string & E2Str(const eTyUEr &);         
const std::string & E2Str(const eTyNums &);         
const std::string & E2Str(const eTyInvRad &);         
const std::string & E2Str(const eTyPyrTieP &);         
const std::string & E2Str(const eModeEpipMatch &);         
const std::string & E2Str(const eModePaddingEpip &);         
const std::string & E2Str(const eModeCaracMatch &);         

template <class Type> const Type & Str2E(const std::string &); 
template <class Type> std::string   StrAllVall();
template <class Type> std::vector<Type> SubOfPat(const std::string & aPat,bool AcceptEmpty=false);

template <class TypeEnum> class cEnumAttr;
typedef cEnumAttr<eTA2007> tSemA2007;
template <class Type> tSemA2007  AC_ListVal();  ///< Additional comm giving list of possible values, tagged with eTA2007::AddCom as they are printed only with Help mode

/* To use fully automatic in specif, need to add :
Serial/cReadOneArgCL.cpp:MACRO_INSTANTIATE_ARG2007 =>  For Additional commentary
Serial/cStrIO.cpp:MACRO_INSTANTITATE_STRIO_ENUM => For creating ToStr/FromStr like other type

Serial/uti_e2string.cpp: ..::tMapE2Str cE2Str<eModeEpipMatch>::mE2S
Serial/uti_e2string.cpp:TPL_ENUM_2_STRING
=>  for creating the 2 dictionnaries enum <=> strings
*/


};

#endif  //  _MMVII_Enums_H_
