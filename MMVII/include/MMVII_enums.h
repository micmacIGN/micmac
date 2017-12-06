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
                eNbVals
           };

/// Appli Features
enum class eApF
           {
               Project, ///< Project Managenent
               Test,    ///< Test
               Ori,     ///< Orientation
               Match,   ///< Dense Matching
               TieP,    ///< Tie-Point processing
               eNbVals
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
              eNbVals
           };


/// Type of set creation
enum class eTySC    
           {
              NonInit,   ///< With Ptr Null
              US,        ///< With unordered set
              eNbVals
           };

/// Type of operator

enum class eOpAff
           {
               ePlusEq,   ///< +=
               eMulEq,    ///< *=
               eMinusEq,  ///< *=
               eEq,       ///< =
               eReset,    ///< =0
               eNbVals
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
              eIntervWithoutSet
           };


const std::string & E2Str(const eTySC &);         
const std::string & E2Str(const eOpAff &);         
const std::string & E2Str(const eTA2007 &);         
const std::string & E2Str(const eTyUEr &);         

template <class Type> const Type & Str2E(const std::string &); 
template <class Type> std::string   StrAllVall();


};

#endif  //  _MMVII_Enums_H_
