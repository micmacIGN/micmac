#include "include/MMVII_all.h"

namespace MMVII
{

/* =========================    STRING CONSTANTE  ========================= */

// XML TAG
const std::string TagMMVIISerial = "MMVII_Serialization";
const std::string TagSetOfName = "SetOfName";
// const std::string TheMMVII_SysName => voir Utils/uti_sysdep.cpp
 
// Name of standard directories
const std::string TmpMMVIIDir    = "Tmp-2007-Dir/";
const std::string MMVIITestDir    = "MMVII-TestDir/";

// Name of common parameters
const std::string NameDirProj = "DirProj";

#if (THE_MACRO_MMVII_SYS == MMVII_SYS_L)
const char CharProctected = '\\';
#endif

// MicMac-v1  compatiblity 
const std::string  MMv1XmlTag_SetName = "ListOfName";
const std::string  Bin2007 = "MMVII";



};

