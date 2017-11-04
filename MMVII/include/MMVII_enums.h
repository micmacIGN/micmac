#ifndef  _MMVII_Enums_H_
#define  _MMVII_Enums_H_

/** \file MMVII_enums.h
    \brief Contains (almost) all enums

    This file contains definition of all global enums,
    I put all in a single file :
    (1) I think it is convenient for searching 
    (2) More important, this file may be generated automatically in case I 
    decide to use code genreration for the enum2string functionnality
*/


enum class eSYS;


/// Type for Semantic of Arg 2007
enum class eTA2007
           {
                PatFile,  ///< Pattern File
                MPatIm,   ///< Major PaternIm
                MDirOri   ///< Major DirOri
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


#endif  //  _MMVII_Enums_H_
