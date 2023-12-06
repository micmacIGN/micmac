#include "MMVII_util.h"
#include "MMVII_Sys.h"
#include "MMVII_Stringifier.h"
#include "MMVII_DeclareCste.h"

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
const std::string TagMMVIIData = "Data";
const std::string TagMMVIIRoot = "Root";
const std::string TagMMVIIType = "Type";
const std::string TagMMVIIVersion = "Version";

const std::string TagSetOfName = "SetOfName";
const std::string TagSetOfCpleName = "SetCpleOfName";

// const std::string TheMMVII_SysName => voir Utils/uti_sysdep.cpp
 
// Name of standard directories
const std::string MMVII_ComonPrefix      =  "MMVII";

const std::string MMVII_DirPhp = MMVII_ComonPrefix + "-PhgrProj" + StringDirSeparator();


const std::string TmpMMVIIDirPrefix      =  MMVII_ComonPrefix +"-Tmp-Dir";
const std::string TmpMMVIIDirGlob        =  TmpMMVIIDirPrefix + "-Glob" + StringDirSeparator();
const std::string TmpMMVIIDirPCar        =  TmpMMVIIDirPrefix + "-PCar" + StringDirSeparator();


const std::string MMVIITestDir       = "MMVII-TestDir" +StringDirSeparator();
const std::string TmpMMVIIProcSubDir = "Process" + StringDirSeparator();
const std::string MMVIIRessourcesDir       = "MMVII-RessourceDir" +StringDirSeparator();
const std::string MMVIILocalParametersDir       = "MMVII-LocalParameters" +StringDirSeparator();


const std::string BLANK = " ";
const std::vector<std::string>  EMPTY_VSTR;
// Files
const std::string MMVII_LogFile = "MMVII-LogFile.txt";

// Name of common parameters
      // -- Current
const std::string CurOP_Out = "Out";  ///< Many command have an Output file 
const std::string CurOP_OutBin = "Bin";  ///< Bin format
const std::string CurOP_SkipWhenExist = "SkWE";
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
const std::string GIP_KthCall = "KthCall";
const std::string GIP_LevCall = "LevCall";
const std::string GIP_ShowAll = "ShowAll";
const std::string GIP_PGMA = "PrefixGMA";
const std::string GIP_TIM_GMA = "Prefix_TIM_GMA";
const std::string GIP_DirProjGMA = "DirGMA";
const std::string GIP_BenchMode = "BenchMode";
const std::string GIP_ExecFrom = "ExecFrom";
const std::string GIP_PatternGMA = "PatternInit_GMA";


#if (THE_MACRO_MMVII_SYS == MMVII_SYS_L)
const char CharProctected = '\\';
#endif

// Name of executable
#if (THE_MACRO_MMVII_SYS == MMVII_SYS_L)
const std::string MMVIIBin2007 = "MMVII";
#elif (THE_MACRO_MMVII_SYS == MMVII_SYS_A)
const std::string MMVIIBin2007 = "MMVII";
#elif (THE_MACRO_MMVII_SYS == MMVII_SYS_W)
const std::string MMVIIBin2007 = "MMVII.exe";
#endif

// User/Command
const   std::string MMVII_LocalSys = "Local";
const   std::string MMVII_NONE = "NONE";
const   std::string MMVII_StdDest = "STD";
const   std::string MMVII_PrefRefBench = "RefBench-";

// PostFix 4 files
const   std::string PostF_XmlFiles  = "xml";
const   std::string PostF_DumpFiles = "dmp";
const   std::string PostF_CsvFiles = "csv";
const   std::string & StdPostF_ArMMVII(bool isXml)
{
    return isXml ? PostF_XmlFiles  : PostF_DumpFiles;
}

// PreFix 4 files
const std::string  PrefixCalRadRad = "CalibRadiom-Radial-";
const std::string  PrefixSpecifXML = "MMVII_Specif_";



};

