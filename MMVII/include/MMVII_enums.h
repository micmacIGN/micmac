#ifndef  _MMVII_Enums_H_
#define  _MMVII_Enums_H_

#include "MMVII_AllClassDeclare.h"

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
                FileDirProj,   ///< File that define the  Dir Proj
                FileImage,     ///< File containing an image
                FileCloud,     ///< File containing a cloud file (ply ?)
                File3DRegion,  ///< File containing a 3D region
                MPatFile,      ///< Major PaternIm => "" or "0" in sem for set1, "1" or other for set2
                FFI,           ///< File Filter Interval
                Orient,        ///< Orientation
                Radiom,        ///< Radiometry
                MeshDev,       ///< Mesh Devlopment
                Mask,          ///< Mask of image
                MetaData,      ///< Meta data images
                PointsMeasure, ///< Measure of point , 2D or 3D
                Input,         ///< Is this parameter used as input/read
                Output,        ///< Is this parameter used as output/write
                OptionalExist, ///< if given, the file (image or other) can be unexisting (interface mut allow seizing "at hand")
		PatParamCalib, ///< It's a pattern for parameter of calibration
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
		XmlOfTopTag,   ///< Parameter must be a XML-file containing certain tag
                eNbVals        ///< Tag for number of value
           };

/// Appli Features
enum class eApF
           {
               ManMMVII,   ///< Managenent of MMVII
               Project,    ///< Project Managenent (user's)
               Test,       ///< Test
               ImProc,     ///< Image processing
               Radiometry, ///< Radiometric modelization
               Ori,        ///< Orientation
               Match,      ///< Dense Matching
               TieP,       ///< Tie-Point processing
               TiePLearn,    ///< Tie-Point processing  - Learning step
               Cloud,       ///< Cloud processing
               CodedTarget,  ///< Coded target (generate, match )
               Topo,        ///< Topometry
               Perso,      ///< Personnal
               eNbVals     ///< Tag for number of value
           };

/// Appli Data Type
enum class eApDT
           {
              Ori,    ///< Orientation
              PCar,   ///< Tie Points
              TieP,   ///< Tie Points
              GCP,   ///< Tie Points
              Image,   ///< Image
              Orient,   ///< Orientations files
              Radiom,   ///< Orientations files
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

/// Type of Error
enum class eLevelCheck    
           {
              NoCheck,
              Warning,
              Error
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
              eBadXmlTopTag,
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
              eBadDimForBox,
              eBadSize4Vect,
              eMultiplePostifx,
              eBadPostfix,
              eNoAperture,
              eNoFocale,
              eNoFocaleEqui35,
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
               eTMR_ParallSilence, ///< Recall by sub-process in parallel
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

/** Mode test prop cov */
enum class eModeTestPropCov
{
   eMTPC_MatCovRFix,  // Mode directly add the covariance matrix, rotation fix
   //   eMTPC_MatCovRUk,  // Mode directly add the covariance matrix, rotation unknown (to implemant ?)
   //   eMTPC_SomL2RFix,  // Mode sum of square linear form, Rotation fix (to implement ?)
   eMTPC_SomL2RUk,  // Mode sum of square linear form, Rotation uknonwn
   eMTPC_PtsRFix,  // Mode direct distance to points, rotation fix
   eMTPC_PtsRUk,  // Mode direct distance to points, rotation unknown
   eNbVals
};

/** Mode "Matcher" callable in DenseMatchEpipGen */
enum class eModeEpipMatch
{
   eMEM_MMV1,  // Mode MicMac V1
   eMEM_PSMNet,// Mode PSMNet
   eMEM_NoMatch,  // Do no match, used for debuging
   eNbVals
};

/**  Mode os system "sur resolus"  */
enum class eModeSSR
{
      eSSR_LsqDense,        ///< Least square, normal equation, with dense implementation
      eSSR_LsqNormSparse,   ///< Least square, normal equation, with sparse implementation
      eSSR_LsqSparseGC      ///< Least square, NO normal equation (Conjugate Gradient) , with sparse implementation
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
   //#################  MULTI SCALE CRITERIA ##################

     // ----------  Census Quantitatif ------------
   eMS_CQ1,    ///< Census Quantitatif 1 pixel (~ 3x3)
   eMS_CQ2,    ///< Census Quantitatif 2 pixel (~ 5x5)
   eMS_CQ3,    ///< Census Quantitatif 3 pixel (~ 7x7)
   eMS_CQ4,    ///< Census Quantitatif 4 pixel (~ 9x9)
   eMS_CQA,    ///< Census Quantitatif all window
   eMS_CQW,    ///< Census Quantitatif all window, weihted
     // ----------  "Basic" Census, not conived it is usefull, but it's popular ------------
   eMS_Cen1,    ///< Census  1 pixel (~ 3x3)
   eMS_Cen2,    ///< Census  2 pixel (~ 5x5)
   eMS_Cen3,    ///< Census  3 pixel (~ 7x7)
   eMS_Cen4,    ///< Census  4 pixel (~ 9x9)
   eMS_CenA,    ///< Census  All
   eMS_CenW,    ///< Census  weighted
     // ----------  Normalized Cross Correlation ------------
   eMS_Cor1,    ///< Correl 1 pixel (~ 3x3)
   eMS_Cor2,    ///< Correl 2 pixel (~ 5x5)
   eMS_Cor3,    ///< Correl 3 pixel (~ 7x7)
   eMS_Cor4,    ///< Correl 4 pixel (~ 9x9)
   eMS_CorA,    ///< Correl  all window
   eMS_CorW,    ///< Correl  all window, weihted
     // ---------- "Corner", select only best average, correl probably   ------------
   eMS_CornW180,   ///< Corner, 180 of average
   eMS_CornW90,    ///< Corner, 90  of average
     // ----------  Best/Worst score on a given neighboor  ------------
     // For best we use correl as, being quite strict, if it is good, then is has signification even with optimi best
     // For worst we use CQ as being "laxist" (good with any homogeneous), if has signifcation even with pessimistic worst
   eMS_WorstCorrel2,   ///< Worst weighted correl on all window <= 2 (5x5) 
   eMS_WorstCorrel3,   ///< Worst weighted correl on all window <= 3 (7x7)
   eMS_BestCorrel2,   ///< Best weighted correl on all window <= 2 (5x5) 
   eMS_BestCorrel3,   ///< Best weighted correl on all window <= 3 (7x7)
   eMS_BestCorrel4,   ///< Best weighted correl on all window <= 4 (9x9)
   eMS_BestCorrel5,   ///< Best weighted correl on all window <= 5 (11x11)
          // ----
   eMS_BestCQ2,    ///< Best Census quantitative on all windows <= 2 (5x5)
   eMS_BestCQ3,    ///< Best Census quantitative on all windows <= 3 (7x7)
   eMS_WorstCQ2,    ///< Worst Census quantitative on all windows <= 2 (5x5)
   eMS_WorstCQ3,    ///< Worst Census quantitative on all windows <= 3 (7x7)
   eMS_WorstCQ4,    ///< Worst Census quantitative on all windows <= 4 (9x9)
   eMS_WorstCQ5,    ///< Worst Census quantitative on all windows <= 5 (11x11)

     // ----------  Using gray level  ------------
   eMS_MinStdDev1,  ///< Min a both std dev
   eMS_MinStdDev2,  ///< Min a both std dev
   eMS_MinStdDev3,  ///< Min a both std dev
   eMS_MinStdDev4,  ///< Min a both std dev
   eMS_MinStdDevW,  ///< Min a both std dev

   //#################  CRITERIA ON NORMALIZED IMAGES ##################

   eNI_DifGray,    ///< Different of gray Gray Level ,
   eNI_MinGray,    ///<  Min of 2 Gray Level , may indicate fiability (low contrast=> means pb ?)

   eNI_Diff1,    ///< Som Diff 1 pixel (~ 3x3)
   eNI_Diff2,    ///< Som Diff 2 pixel (~ 3x3)
   eNI_Diff3,    ///< Som Diff 3 pixel (~ 3x3)
   eNI_Diff5,    ///< Som Diff 4 pixel (~ 3x3)
   eNI_Diff7,    ///< Som Diff 4 pixel (~ 3x3)

   //#################  CRITERIA ON STD IMAGES ##################

     // ----------  Normalized Cross Correlation ------------
   eSTD_Cor1,    ///< Correl 1 pixel (~ 3x3)
   eSTD_Cor2,    ///< Correl 2 pixel (~ 5x5)
   eSTD_Cor3,    ///< Correl 3 pixel (~ 7x7)
   eSTD_Cor4,    ///< Correl 4 pixel (~ 9x9)

     // ---------- Non centered  Normalized Cross Correlation ------------
   eSTD_NCCor1,    ///< Correl 1 pixel (~ 3x3)
   eSTD_NCCor2,    ///< Correl 2 pixel (~ 5x5)
   eSTD_NCCor3,    ///< Correl 3 pixel (~ 7x7)
   eSTD_NCCor4,    ///< Correl 4 pixel (~ 9x9)

     // ----------  Normalized Cross Correlation ------------
   eSTD_Diff1,    ///< Correl 1 pixel (~ 3x3)
   eSTD_Diff2,    ///< Correl 1 pixel (~ 3x3)
   eSTD_Diff3,    ///< Correl 1 pixel (~ 3x3)
   eSTD_Diff5,    ///< Correl 1 pixel (~ 3x3)
   eSTD_Diff7,    ///< Correl 1 pixel (~ 3x3)

     // ----------  Census quant ------------
   eSTD_CQ2,    ///< Census Quantitatif 2 pixel (~ 3x3)
   eSTD_CQ4,    ///< Census Quantitatif 4 pixel (~ 3x3)
   eSTD_CQ6,    ///< Census Quantitatif 6 pixel (~ 3x3)
   eSTD_CQ8,    ///< Census Quantitatif 8 pixel (~ 3x3)
     // ----------  Census "normal" ------------
   eSTD_Cen2,    ///< Census 2 pixel (~ 3x3)
   eSTD_Cen4,    ///< Census 4 pixel (~ 3x3)
   eSTD_Cen6,    ///< Census 6 pixel (~ 3x3)
   eSTD_Cen8,    ///< Census 8 pixel (~ 3x3)

   eNbVals
};

/** Filters for detecteting coded target */

enum class eDCTFilters
{
   eSym,       // symetry filters arround pts
   eBin,       // binarity of the  histogramm
   eRad,       // radiality of the distribution
   eGrad,      // average  gradient
   eNbVals
};

enum class eProjPC
{
     eStenope,
     eFE_EquiDist,
     eFE_EquiSolid,
     eStereroGraphik,
     eOrthoGraphik,
     eEquiRect,
     eNbVals
};

enum class eTyCodeTarget
{
    eIGNIndoor,     ///<  checkboard , 
    eIGNDroneSym,    ///<  checkboard , code separate Top/Down
    eIGNDroneTop,   ///<  checkboard Top , code bottom,
    eCERN,          ///<  central circle, coding invariant (AICON, METASHAPE ...)
    eNbVals
};

enum class eMTDIm
           {
              eFocalmm,
              eAperture,
              eModeleCam,
              eNbVals
           };


const std::string & E2Str(const eMTDIm &);
const std::string & E2Str(const eProjPC &);         
const std::string & E2Str(const eDCTFilters &);         
const std::string & E2Str(const eTyCodeTarget &);         
const std::string & E2Str(const eTySC &);         
const std::string & E2Str(const eOpAff &);         
const std::string & E2Str(const eTA2007 &);         
const std::string & E2Str(const eTyUEr &);         
const std::string & E2Str(const eTyNums &);         
const std::string & E2Str(const eTyInvRad &);         
const std::string & E2Str(const eTyPyrTieP &);         
const std::string & E2Str(const eModeEpipMatch &);         
const std::string & E2Str(const eModeTestPropCov &);         
const std::string & E2Str(const eModePaddingEpip &);         
const std::string & E2Str(const eModeCaracMatch &);         

template <class Type> const Type & Str2E(const std::string &); 
template <class Type> std::string   StrAllVall();
/// return a vector with list all label corresponding to aPat
template <class Type> std::vector<Type> SubOfPat(const std::string & aPat,bool AcceptEmpty=false);
/// logically ~ SubOfPat, but returned as a vec of bool, indexable by (int)Label for direct access
template <class Type> std::vector<bool> VBoolOfPat(const std::string & aPat,bool AcceptEmpty=false);

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
