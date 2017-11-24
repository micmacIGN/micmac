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
                PatFile,       ///< Pattern File
                // PatOrXmlFile,    ///< Pattern File
                DirProject,    ///< Exact Dir of Proj
                FileDirProj,   ///< File of Dir Proj
                MPatIm,        ///< Major PaternIm => "" or "0" in sem for set1, "1" or other for set2
                MDirOri,       ///< Major DirOri
                Internal,      ///< Reserved to internall use by MMVII
                Common         ///< Parameter  Common to all commands
           };


/// Appli Features
enum class eApF
           {
               Project, ///< Project Managenent
               Test,    ///< Test
               Ori,     ///< Orientation
               Match,   ///< Dense Matching
               TieP     ///< Tie-Point processing
           };

/// Appli Data Type
enum class eApDT
           {
              Ori,    ///< Orientation
              TieP,   ///< Tie Points
              Ply,    ///< Ply file
              None,     ///< Nothing 
              Console,  ///< Console 
              Xml       ///< Xml-files
           };


/// Type of set creation
enum class eTySC    
           {
              NonInit,  ///< With Ptr Null
              US        ///< With unordered set
           };


/// Type of operator

enum class eOperator
           {
               ePlusEq,   /// +=
               eMulEq,    /// *=
               eMinusEq,  /// *=
               eEq,       /// =
               eReset        /// =
           };

std::string E2Str(const eOperator &);
eOperator   Str2E(const std::string &);


};

#endif  //  _MMVII_Enums_H_
