#ifndef  _MMVII_DeclareCste_H_
#define  _MMVII_DeclareCste_H_

/** \file MMVII_DeclareCste.h
    \brief Contains declaration of all constant string

    The reason for puting all in the same file is that in MMV1,
    I lost a lot of time looking bag for the declartion located
    in many file.
    The definition is also located in a single cpp.
*/



// Xml tags
extern const std::string TagMMVIISerial;    ///< Xml top tag of all generated file
extern const std::string TagSetOfName;      ///< Xml (sub) top tag for 
// ------
extern const std::string TheMMVII_SysName ; ///< Name of Operating system

extern const std::string NameDirProj;

// Folders
extern const std::string TmpMMVIIDir;
extern const std::string MMVIITestDir;


extern const char CharProctected;  // => '\' on Gnu/Linux, will see on others



#endif  //  _MMVII_DeclareCste_H_
