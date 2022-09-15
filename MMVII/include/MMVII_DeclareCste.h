#ifndef  _MMVII_DeclareCste_H_
#define  _MMVII_DeclareCste_H_

namespace MMVII
{

/** \file MMVII_DeclareCste.h
    \brief Contains declaration of all constant string

    The reason for puting all in the same file is that in MMV1,
    I lost a lot of time looking bag for the declartion located
    in many file.
    The definition is also located in a single cpp.
*/



// Xml tags
extern const std::string TagMMVIISerial;    ///< Xml top tag of all generated file
extern const std::string TagSetOfName;      ///< Xml (sub) top tag for  Set of name
extern const std::string TagSetOfCpleName;  ///< Xml (sub) top tag for  Set of Cple of Name (relation)
// ------
extern const std::string TheMMVII_SysName ; ///< Name of Operating system

// Name of current optionnal parameters, to facilitate sharings ...
extern const std::string CurOP_Out;  ///< Many command have an Output file 
extern const std::string CurOP_OutBin;  ///< Generate out in binary/txt mode 

// Name of global optionnal parameters
extern const std::string GOP_DirProj;  ///< Directory of Proj
extern const std::string GOP_NumVO;    ///< NumVOut 
extern const std::string GOP_Int0;     ///< FFI0 => File Filter Interval
extern const std::string GOP_Int1;     ///< FFI1
extern const std::string GOP_StdOut;   ///< StdOut, Output redirection
extern const std::string GOP_SeedRand; ///< If an explicit seed generationyy
extern const std::string GOP_NbProc;   ///< Number of Process in paral
extern const std::string GOP_WW;       ///< With Warning
//  Name of Global INTERNAL optional parameter
extern const std::string GIP_LevCall;     ///< Level of MMVII call
extern const std::string GIP_ShowAll;     ///< Show a lot of intermediary steps
extern const std::string GIP_PGMA;     ///< Prefix Global Main Appli
extern const std::string GIP_DirProjGMA;     ///< Dir Proj of Global Main Application


// Folders
extern const std::string TmpMMVIIDirPrefix;
extern const std::string TmpMMVIIDirGlob;
extern const std::string TmpMMVIIDirPCar;
extern const std::string MMVIITestDir;
extern const std::string TmpMMVIIProcSubDir;

//  String 
extern const std::string BLANK;  // just std::string(" ") to avoid char * + char *
extern const std::vector<std::string> EMPTY_VSTR;  // just std::string(" ") to avoid char * + char *


//  Files
extern const std::string MMVII_LogFile;
extern const char CharProctected;  // => '\' on Gnu/Linux, will see on others

// MicMac-v1  compatiblity 
extern const std::string  MMv1XmlTag_SetName;
extern const std::string  MMv1XmlTag_RelName;

// MicMac Install
extern const   std::string DirBin2007;  ///< computed by MM Instal
extern const   std::string Bin2007; ///< MMVII 4 now

// PostFix 4 files
extern const   std::string PostF_XmlFiles; ///< xml now
extern const   std::string PostF_DumpFiles; ///< dmp now
const   std::string & StdPostF_ArMMVII(bool xml); ///< one of 2 above

// Users Value

extern const   std::string MMVII_NONE;  ///< For command, each time a "no value" can be used
extern const   std::string MMVII_StdDest;  ///< For destination parameter (TieP ...) , def value

 /*=====================================================*/
 /*                                                     */
 /*    Numerical value                                  */
 /*                                                     */
 /*=====================================================*/

#define  DefStdDevImWellSample   0.6266 ///< sqrt(pi/8) see doc GaussPyram \label{GP:SIGMA0}

};

#endif  //  _MMVII_DeclareCste_H_
