#include "include/MMVII_all.h"

namespace MMVII
{

template <> const std::string  & XMLTagSet<std::string> () {return TagSetOfName;}
template <> const std::string  & XMLTagSet<tNamePair>   () {return TagSetOfCpleName;}


template <> const std::string  & MMv1_XMLTagSet<std::string> () {return MMv1XmlTag_SetName;}
template <> const std::string  & MMv1_XMLTagSet<tNamePair> ()   {return MMv1XmlTag_RelName;}

/* =========================    STRING CONSTANTE  ========================= */

// XML TAG
    // V1 -  MicMac-v1  compatiblity 
const std::string  MMv1XmlTag_SetName = "ListOfName";
const std::string  MMv1XmlTag_RelName = "SauvegardeNamedRel";
    // V2
const std::string TagMMVIISerial = "MMVII_Serialization";
const std::string TagSetOfName = "SetOfName";
const std::string TagSetOfCpleName = "SetCpleOfName";

// const std::string TheMMVII_SysName => voir Utils/uti_sysdep.cpp
 
// Name of standard directories
const std::string MMVII_ComonPrefix      =  "MMVII";
const std::string TmpMMVIIDirPrefix      =  MMVII_ComonPrefix +"-Tmp-Dir";
const std::string TmpMMVIIDirGlob        =  TmpMMVIIDirPrefix + "-Glob" + StringDirSeparator();
const std::string TmpMMVIIDirPCar        =  TmpMMVIIDirPrefix + "-PCar" + StringDirSeparator();

const std::string MMVIIDirOrient      =  MMVII_ComonPrefix +"-Orient" + StringDirSeparator();
const std::string MMVIIDirHomol       =  MMVII_ComonPrefix +"-Homol" + StringDirSeparator();

const std::string MMVIITestDir       = "MMVII-TestDir" +StringDirSeparator();
const std::string TmpMMVIIProcSubDir = "Process" + StringDirSeparator();

const std::string BLANK = " ";
const std::vector<std::string>  EMPTY_VSTR;
// Files
const std::string MMVII_LogFile = "MMVII-LogFile.txt";

// Name of common parameters
      // -- Current
const std::string CurOP_Out = "Out";  ///< Many command have an Output file 
const std::string CurOP_OutBin = "Bin";  ///< Bin format
      // -- External
const std::string GOP_WW       = "WW";
const std::string GOP_DirProj  = "DirProj";
const std::string GOP_NumVO    = "NumVOut";
const std::string GOP_Int0     = "FFI0";
const std::string GOP_Int1     = "FFI1";
const std::string GOP_StdOut   = "StdOut";
const std::string GOP_SeedRand = "SeedRand";
const std::string GOP_NbProc   = "NbProc";
      // -- Internal
const std::string GIP_LevCall = "LevCall";
const std::string GIP_ShowAll = "ShowAll";
const std::string GIP_PGMA = "PrefixGMA";
const std::string GIP_DirProjGMA = "DirGMA";


#if (THE_MACRO_MMVII_SYS == MMVII_SYS_L)
const char CharProctected = '\\';
#endif

const std::string  Bin2007 = "MMVII";

// User/Command
const   std::string MMVII_NONE = "NONE";
const   std::string MMVII_StdDest = "STD";

// PostFix 4 files
const   std::string PostF_XmlFiles  = "xml";
const   std::string PostF_DumpFiles = "dmp";
const   std::string & StdPostF_ArMMVII(bool isXml)
{
    return isXml ? PostF_XmlFiles  : PostF_DumpFiles;
}



};

