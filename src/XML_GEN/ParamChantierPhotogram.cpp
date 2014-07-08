#include "StdAfx.h"
//#include "general/all.h"
//#include "private/all.h"
// #include "XML_GEN/ParamChantierPhotogram.h"
// NO MORE
eNewTypeMalt  Str2eNewTypeMalt(const std::string & aName)
{
   if (aName=="eTMalt_Ortho")
      return eTMalt_Ortho;
   else if (aName=="eTMalt_UrbanMNE")
      return eTMalt_UrbanMNE;
   else if (aName=="eTMalt_GeomImage")
      return eTMalt_GeomImage;
   else if (aName=="eTMalt_NbVals")
      return eTMalt_NbVals;
  else
  {
      cout << aName << " is not a correct value for enum eNewTypeMalt\n" ;
      ELISE_ASSERT(false,"XML enum value error");
  }
  return (eNewTypeMalt) 0;
}
void xml_init(eNewTypeMalt & aVal,cElXMLTree * aTree)
{
   aVal= Str2eNewTypeMalt(aTree->Contenu());
}
std::string  eToString(const eNewTypeMalt & anObj)
{
   if (anObj==eTMalt_Ortho)
      return  "eTMalt_Ortho";
   if (anObj==eTMalt_UrbanMNE)
      return  "eTMalt_UrbanMNE";
   if (anObj==eTMalt_GeomImage)
      return  "eTMalt_GeomImage";
   if (anObj==eTMalt_NbVals)
      return  "eTMalt_NbVals";
 std::cout << "Enum = eNewTypeMalt\n";
   ELISE_ASSERT(false,"Bad Value in eToString for enum value ");
   return "";
}

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eNewTypeMalt & anObj)
{
      return  cElXMLTree::ValueNode(aNameTag,eToString(anObj));
}

void  BinaryDumpInFile(ELISE_fp & aFp,const eNewTypeMalt & anObj)
{
   BinaryDumpInFile(aFp,int(anObj));
}

void  BinaryUnDumpFromFile(eNewTypeMalt & anObj,ELISE_fp & aFp)
{
   int aIVal;
   BinaryUnDumpFromFile(aIVal,aFp);
   anObj=(eNewTypeMalt) aIVal;
}

std::string  Mangling( eNewTypeMalt *) {return "640FC5583697CDA3FE3F";};

eTypeTapas  Str2eTypeTapas(const std::string & aName)
{
   if (aName=="eTT_RadialBasic")
      return eTT_RadialBasic;
   else if (aName=="eTT_RadialExtended")
      return eTT_RadialExtended;
   else if (aName=="eTT_Fraser")
      return eTT_Fraser;
   else if (aName=="eTT_FishEyeEqui")
      return eTT_FishEyeEqui;
   else if (aName=="eTT_AutoCal")
      return eTT_AutoCal;
   else if (aName=="eTT_Figee")
      return eTT_Figee;
   else if (aName=="eTT_HemiEqui")
      return eTT_HemiEqui;
   else if (aName=="eTT_RadialStd")
      return eTT_RadialStd;
   else if (aName=="eTT_FraserBasic")
      return eTT_FraserBasic;
   else if (aName=="eTT_FishEyeBasic")
      return eTT_FishEyeBasic;
   else if (aName=="eTT_FE_EquiSolBasic")
      return eTT_FE_EquiSolBasic;
   else if (aName=="eTT_RadGen7x2")
      return eTT_RadGen7x2;
   else if (aName=="eTT_RadGen11x2")
      return eTT_RadGen11x2;
   else if (aName=="eTT_RadGen15x2")
      return eTT_RadGen15x2;
   else if (aName=="eTT_RadGen19x2")
      return eTT_RadGen19x2;
   else if (aName=="eTT_NbVals")
      return eTT_NbVals;
  else
  {
      cout << aName << " is not a correct value for enum eTypeTapas\n" ;
      ELISE_ASSERT(false,"XML enum value error");
  }
  return (eTypeTapas) 0;
}
void xml_init(eTypeTapas & aVal,cElXMLTree * aTree)
{
   aVal= Str2eTypeTapas(aTree->Contenu());
}
std::string  eToString(const eTypeTapas & anObj)
{
   if (anObj==eTT_RadialBasic)
      return  "eTT_RadialBasic";
   if (anObj==eTT_RadialExtended)
      return  "eTT_RadialExtended";
   if (anObj==eTT_Fraser)
      return  "eTT_Fraser";
   if (anObj==eTT_FishEyeEqui)
      return  "eTT_FishEyeEqui";
   if (anObj==eTT_AutoCal)
      return  "eTT_AutoCal";
   if (anObj==eTT_Figee)
      return  "eTT_Figee";
   if (anObj==eTT_HemiEqui)
      return  "eTT_HemiEqui";
   if (anObj==eTT_RadialStd)
      return  "eTT_RadialStd";
   if (anObj==eTT_FraserBasic)
      return  "eTT_FraserBasic";
   if (anObj==eTT_FishEyeBasic)
      return  "eTT_FishEyeBasic";
   if (anObj==eTT_FE_EquiSolBasic)
      return  "eTT_FE_EquiSolBasic";
   if (anObj==eTT_RadGen7x2)
      return  "eTT_RadGen7x2";
   if (anObj==eTT_RadGen11x2)
      return  "eTT_RadGen11x2";
   if (anObj==eTT_RadGen15x2)
      return  "eTT_RadGen15x2";
   if (anObj==eTT_RadGen19x2)
      return  "eTT_RadGen19x2";
   if (anObj==eTT_NbVals)
      return  "eTT_NbVals";
 std::cout << "Enum = eTypeTapas\n";
   ELISE_ASSERT(false,"Bad Value in eToString for enum value ");
   return "";
}

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eTypeTapas & anObj)
{
      return  cElXMLTree::ValueNode(aNameTag,eToString(anObj));
}

void  BinaryDumpInFile(ELISE_fp & aFp,const eTypeTapas & anObj)
{
   BinaryDumpInFile(aFp,int(anObj));
}

void  BinaryUnDumpFromFile(eTypeTapas & anObj,ELISE_fp & aFp)
{
   int aIVal;
   BinaryUnDumpFromFile(aIVal,aFp);
   anObj=(eTypeTapas) aIVal;
}

std::string  Mangling( eTypeTapas *) {return "58CA7259C38FBDFBFDBF";};

eTypeMMByP  Str2eTypeMMByP(const std::string & aName)
{
   if (aName=="eGround")
      return eGround;
   else if (aName=="eStatue")
      return eStatue;
   else if (aName=="eTestIGN")
      return eTestIGN;
   else if (aName=="eNbTypeMMByP")
      return eNbTypeMMByP;
  else
  {
      cout << aName << " is not a correct value for enum eTypeMMByP\n" ;
      ELISE_ASSERT(false,"XML enum value error");
  }
  return (eTypeMMByP) 0;
}
void xml_init(eTypeMMByP & aVal,cElXMLTree * aTree)
{
   aVal= Str2eTypeMMByP(aTree->Contenu());
}
std::string  eToString(const eTypeMMByP & anObj)
{
   if (anObj==eGround)
      return  "eGround";
   if (anObj==eStatue)
      return  "eStatue";
   if (anObj==eTestIGN)
      return  "eTestIGN";
   if (anObj==eNbTypeMMByP)
      return  "eNbTypeMMByP";
 std::cout << "Enum = eTypeMMByP\n";
   ELISE_ASSERT(false,"Bad Value in eToString for enum value ");
   return "";
}

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eTypeMMByP & anObj)
{
      return  cElXMLTree::ValueNode(aNameTag,eToString(anObj));
}

void  BinaryDumpInFile(ELISE_fp & aFp,const eTypeMMByP & anObj)
{
   BinaryDumpInFile(aFp,int(anObj));
}

void  BinaryUnDumpFromFile(eTypeMMByP & anObj,ELISE_fp & aFp)
{
   int aIVal;
   BinaryUnDumpFromFile(aIVal,aFp);
   anObj=(eTypeMMByP) aIVal;
}

std::string  Mangling( eTypeMMByP *) {return "34A853915BAC95FDFDBF";};

eTypeQuality  Str2eTypeQuality(const std::string & aName)
{
   if (aName=="eQual_High")
      return eQual_High;
   else if (aName=="eQual_Average")
      return eQual_Average;
   else if (aName=="eQual_Low")
      return eQual_Low;
   else if (aName=="eNbTypeQual")
      return eNbTypeQual;
  else
  {
      cout << aName << " is not a correct value for enum eTypeQuality\n" ;
      ELISE_ASSERT(false,"XML enum value error");
  }
  return (eTypeQuality) 0;
}
void xml_init(eTypeQuality & aVal,cElXMLTree * aTree)
{
   aVal= Str2eTypeQuality(aTree->Contenu());
}
std::string  eToString(const eTypeQuality & anObj)
{
   if (anObj==eQual_High)
      return  "eQual_High";
   if (anObj==eQual_Average)
      return  "eQual_Average";
   if (anObj==eQual_Low)
      return  "eQual_Low";
   if (anObj==eNbTypeQual)
      return  "eNbTypeQual";
 std::cout << "Enum = eTypeQuality\n";
   ELISE_ASSERT(false,"Bad Value in eToString for enum value ");
   return "";
}

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eTypeQuality & anObj)
{
      return  cElXMLTree::ValueNode(aNameTag,eToString(anObj));
}

void  BinaryDumpInFile(ELISE_fp & aFp,const eTypeQuality & anObj)
{
   BinaryDumpInFile(aFp,int(anObj));
}

void  BinaryUnDumpFromFile(eTypeQuality & anObj,ELISE_fp & aFp)
{
   int aIVal;
   BinaryUnDumpFromFile(aIVal,aFp);
   anObj=(eTypeQuality) aIVal;
}

std::string  Mangling( eTypeQuality *) {return "7412760D4FF1D79CFD3F";};

eTypeMalt  Str2eTypeMalt(const std::string & aName)
{
   if (aName=="eOrtho")
      return eOrtho;
   else if (aName=="eUrbanMNE")
      return eUrbanMNE;
   else if (aName=="eGeomImage")
      return eGeomImage;
   else if (aName=="eNbTypesMNE")
      return eNbTypesMNE;
  else
  {
      cout << aName << " is not a correct value for enum eTypeMalt\n" ;
      ELISE_ASSERT(false,"XML enum value error");
  }
  return (eTypeMalt) 0;
}
void xml_init(eTypeMalt & aVal,cElXMLTree * aTree)
{
   aVal= Str2eTypeMalt(aTree->Contenu());
}
std::string  eToString(const eTypeMalt & anObj)
{
   if (anObj==eOrtho)
      return  "eOrtho";
   if (anObj==eUrbanMNE)
      return  "eUrbanMNE";
   if (anObj==eGeomImage)
      return  "eGeomImage";
   if (anObj==eNbTypesMNE)
      return  "eNbTypesMNE";
 std::cout << "Enum = eTypeMalt\n";
   ELISE_ASSERT(false,"Bad Value in eToString for enum value ");
   return "";
}

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eTypeMalt & anObj)
{
      return  cElXMLTree::ValueNode(aNameTag,eToString(anObj));
}

void  BinaryDumpInFile(ELISE_fp & aFp,const eTypeMalt & anObj)
{
   BinaryDumpInFile(aFp,int(anObj));
}

void  BinaryUnDumpFromFile(eTypeMalt & anObj,ELISE_fp & aFp)
{
   int aIVal;
   BinaryUnDumpFromFile(aIVal,aFp);
   anObj=(eTypeMalt) aIVal;
}

std::string  Mangling( eTypeMalt *) {return "64137C5D34A6E6F5FD3F";};

eTypeFichierApp  Str2eTypeFichierApp(const std::string & aName)
{
   if (aName=="eAppEgels")
      return eAppEgels;
   else if (aName=="eAppGeoCub")
      return eAppGeoCub;
   else if (aName=="eAppInFile")
      return eAppInFile;
   else if (aName=="eAppXML")
      return eAppXML;
   else if (aName=="eNbTypeApp")
      return eNbTypeApp;
  else
  {
      cout << aName << " is not a correct value for enum eTypeFichierApp\n" ;
      ELISE_ASSERT(false,"XML enum value error");
  }
  return (eTypeFichierApp) 0;
}
void xml_init(eTypeFichierApp & aVal,cElXMLTree * aTree)
{
   aVal= Str2eTypeFichierApp(aTree->Contenu());
}
std::string  eToString(const eTypeFichierApp & anObj)
{
   if (anObj==eAppEgels)
      return  "eAppEgels";
   if (anObj==eAppGeoCub)
      return  "eAppGeoCub";
   if (anObj==eAppInFile)
      return  "eAppInFile";
   if (anObj==eAppXML)
      return  "eAppXML";
   if (anObj==eNbTypeApp)
      return  "eNbTypeApp";
 std::cout << "Enum = eTypeFichierApp\n";
   ELISE_ASSERT(false,"Bad Value in eToString for enum value ");
   return "";
}

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eTypeFichierApp & anObj)
{
      return  cElXMLTree::ValueNode(aNameTag,eToString(anObj));
}

void  BinaryDumpInFile(ELISE_fp & aFp,const eTypeFichierApp & anObj)
{
   BinaryDumpInFile(aFp,int(anObj));
}

void  BinaryUnDumpFromFile(eTypeFichierApp & anObj,ELISE_fp & aFp)
{
   int aIVal;
   BinaryUnDumpFromFile(aIVal,aFp);
   anObj=(eTypeFichierApp) aIVal;
}

std::string  Mangling( eTypeFichierApp *) {return "8C91EA7BEC3CC981FD3F";};

eTypeFichierOriTxt  Str2eTypeFichierOriTxt(const std::string & aName)
{
   if (aName=="eOriTxtAgiSoft")
      return eOriTxtAgiSoft;
   else if (aName=="eOriBluh")
      return eOriBluh;
   else if (aName=="eOriTxtInFile")
      return eOriTxtInFile;
   else if (aName=="eNbTypeOriTxt")
      return eNbTypeOriTxt;
  else
  {
      cout << aName << " is not a correct value for enum eTypeFichierOriTxt\n" ;
      ELISE_ASSERT(false,"XML enum value error");
  }
  return (eTypeFichierOriTxt) 0;
}
void xml_init(eTypeFichierOriTxt & aVal,cElXMLTree * aTree)
{
   aVal= Str2eTypeFichierOriTxt(aTree->Contenu());
}
std::string  eToString(const eTypeFichierOriTxt & anObj)
{
   if (anObj==eOriTxtAgiSoft)
      return  "eOriTxtAgiSoft";
   if (anObj==eOriBluh)
      return  "eOriBluh";
   if (anObj==eOriTxtInFile)
      return  "eOriTxtInFile";
   if (anObj==eNbTypeOriTxt)
      return  "eNbTypeOriTxt";
 std::cout << "Enum = eTypeFichierOriTxt\n";
   ELISE_ASSERT(false,"Bad Value in eToString for enum value ");
   return "";
}

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eTypeFichierOriTxt & anObj)
{
      return  cElXMLTree::ValueNode(aNameTag,eToString(anObj));
}

void  BinaryDumpInFile(ELISE_fp & aFp,const eTypeFichierOriTxt & anObj)
{
   BinaryDumpInFile(aFp,int(anObj));
}

void  BinaryUnDumpFromFile(eTypeFichierOriTxt & anObj,ELISE_fp & aFp)
{
   int aIVal;
   BinaryUnDumpFromFile(aIVal,aFp);
   anObj=(eTypeFichierOriTxt) aIVal;
}

std::string  Mangling( eTypeFichierOriTxt *) {return "BEDC75ABDE7DB1BCFF3F";};

eImpaintMethod  Str2eImpaintMethod(const std::string & aName)
{
   if (aName=="eImpaintL2")
      return eImpaintL2;
   else if (aName=="eImpaintMNT")
      return eImpaintMNT;
  else
  {
      cout << aName << " is not a correct value for enum eImpaintMethod\n" ;
      ELISE_ASSERT(false,"XML enum value error");
  }
  return (eImpaintMethod) 0;
}
void xml_init(eImpaintMethod & aVal,cElXMLTree * aTree)
{
   aVal= Str2eImpaintMethod(aTree->Contenu());
}
std::string  eToString(const eImpaintMethod & anObj)
{
   if (anObj==eImpaintL2)
      return  "eImpaintL2";
   if (anObj==eImpaintMNT)
      return  "eImpaintMNT";
 std::cout << "Enum = eImpaintMethod\n";
   ELISE_ASSERT(false,"Bad Value in eToString for enum value ");
   return "";
}

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eImpaintMethod & anObj)
{
      return  cElXMLTree::ValueNode(aNameTag,eToString(anObj));
}

void  BinaryDumpInFile(ELISE_fp & aFp,const eImpaintMethod & anObj)
{
   BinaryDumpInFile(aFp,int(anObj));
}

void  BinaryUnDumpFromFile(eImpaintMethod & anObj,ELISE_fp & aFp)
{
   int aIVal;
   BinaryUnDumpFromFile(aIVal,aFp);
   anObj=(eImpaintMethod) aIVal;
}

std::string  Mangling( eImpaintMethod *) {return "79AA429BB659CB8CFF3F";};

eTypeNumerique  Str2eTypeNumerique(const std::string & aName)
{
   if (aName=="eTN_u_int1")
      return eTN_u_int1;
   else if (aName=="eTN_int1")
      return eTN_int1;
   else if (aName=="eTN_u_int2")
      return eTN_u_int2;
   else if (aName=="eTN_int2")
      return eTN_int2;
   else if (aName=="eTN_int4")
      return eTN_int4;
   else if (aName=="eTN_float")
      return eTN_float;
   else if (aName=="eTN_double")
      return eTN_double;
   else if (aName=="eTN_Bits1MSBF")
      return eTN_Bits1MSBF;
  else
  {
      cout << aName << " is not a correct value for enum eTypeNumerique\n" ;
      ELISE_ASSERT(false,"XML enum value error");
  }
  return (eTypeNumerique) 0;
}
void xml_init(eTypeNumerique & aVal,cElXMLTree * aTree)
{
   aVal= Str2eTypeNumerique(aTree->Contenu());
}
std::string  eToString(const eTypeNumerique & anObj)
{
   if (anObj==eTN_u_int1)
      return  "eTN_u_int1";
   if (anObj==eTN_int1)
      return  "eTN_int1";
   if (anObj==eTN_u_int2)
      return  "eTN_u_int2";
   if (anObj==eTN_int2)
      return  "eTN_int2";
   if (anObj==eTN_int4)
      return  "eTN_int4";
   if (anObj==eTN_float)
      return  "eTN_float";
   if (anObj==eTN_double)
      return  "eTN_double";
   if (anObj==eTN_Bits1MSBF)
      return  "eTN_Bits1MSBF";
 std::cout << "Enum = eTypeNumerique\n";
   ELISE_ASSERT(false,"Bad Value in eToString for enum value ");
   return "";
}

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eTypeNumerique & anObj)
{
      return  cElXMLTree::ValueNode(aNameTag,eToString(anObj));
}

void  BinaryDumpInFile(ELISE_fp & aFp,const eTypeNumerique & anObj)
{
   BinaryDumpInFile(aFp,int(anObj));
}

void  BinaryUnDumpFromFile(eTypeNumerique & anObj,ELISE_fp & aFp)
{
   int aIVal;
   BinaryUnDumpFromFile(aIVal,aFp);
   anObj=(eTypeNumerique) aIVal;
}

std::string  Mangling( eTypeNumerique *) {return "CE79743DF94C5FA5FE3F";};

eComprTiff  Str2eComprTiff(const std::string & aName)
{
   if (aName=="eComprTiff_None")
      return eComprTiff_None;
   else if (aName=="eComprTiff_LZW")
      return eComprTiff_LZW;
   else if (aName=="eComprTiff_FAX4")
      return eComprTiff_FAX4;
   else if (aName=="eComprTiff_PackBits")
      return eComprTiff_PackBits;
  else
  {
      cout << aName << " is not a correct value for enum eComprTiff\n" ;
      ELISE_ASSERT(false,"XML enum value error");
  }
  return (eComprTiff) 0;
}
void xml_init(eComprTiff & aVal,cElXMLTree * aTree)
{
   aVal= Str2eComprTiff(aTree->Contenu());
}
std::string  eToString(const eComprTiff & anObj)
{
   if (anObj==eComprTiff_None)
      return  "eComprTiff_None";
   if (anObj==eComprTiff_LZW)
      return  "eComprTiff_LZW";
   if (anObj==eComprTiff_FAX4)
      return  "eComprTiff_FAX4";
   if (anObj==eComprTiff_PackBits)
      return  "eComprTiff_PackBits";
 std::cout << "Enum = eComprTiff\n";
   ELISE_ASSERT(false,"Bad Value in eToString for enum value ");
   return "";
}

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eComprTiff & anObj)
{
      return  cElXMLTree::ValueNode(aNameTag,eToString(anObj));
}

void  BinaryDumpInFile(ELISE_fp & aFp,const eComprTiff & anObj)
{
   BinaryDumpInFile(aFp,int(anObj));
}

void  BinaryUnDumpFromFile(eComprTiff & anObj,ELISE_fp & aFp)
{
   int aIVal;
   BinaryUnDumpFromFile(aIVal,aFp);
   anObj=(eComprTiff) aIVal;
}

std::string  Mangling( eComprTiff *) {return "06AAF027614A29F7FE3F";};

eTypePreCondRad  Str2eTypePreCondRad(const std::string & aName)
{
   if (aName=="ePCR_Atgt")
      return ePCR_Atgt;
   else if (aName=="ePCR_2SinAtgtS2")
      return ePCR_2SinAtgtS2;
  else
  {
      cout << aName << " is not a correct value for enum eTypePreCondRad\n" ;
      ELISE_ASSERT(false,"XML enum value error");
  }
  return (eTypePreCondRad) 0;
}
void xml_init(eTypePreCondRad & aVal,cElXMLTree * aTree)
{
   aVal= Str2eTypePreCondRad(aTree->Contenu());
}
std::string  eToString(const eTypePreCondRad & anObj)
{
   if (anObj==ePCR_Atgt)
      return  "ePCR_Atgt";
   if (anObj==ePCR_2SinAtgtS2)
      return  "ePCR_2SinAtgtS2";
 std::cout << "Enum = eTypePreCondRad\n";
   ELISE_ASSERT(false,"Bad Value in eToString for enum value ");
   return "";
}

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eTypePreCondRad & anObj)
{
      return  cElXMLTree::ValueNode(aNameTag,eToString(anObj));
}

void  BinaryDumpInFile(ELISE_fp & aFp,const eTypePreCondRad & anObj)
{
   BinaryDumpInFile(aFp,int(anObj));
}

void  BinaryUnDumpFromFile(eTypePreCondRad & anObj,ELISE_fp & aFp)
{
   int aIVal;
   BinaryUnDumpFromFile(aIVal,aFp);
   anObj=(eTypePreCondRad) aIVal;
}

std::string  Mangling( eTypePreCondRad *) {return "F35D0134295568AAFE3F";};

eModeGeomMNT  Str2eModeGeomMNT(const std::string & aName)
{
   if (aName=="eGeomMNTCarto")
      return eGeomMNTCarto;
   else if (aName=="eGeomMNTEuclid")
      return eGeomMNTEuclid;
   else if (aName=="eGeomMNTFaisceauIm1PrCh_Px1D")
      return eGeomMNTFaisceauIm1PrCh_Px1D;
   else if (aName=="eGeomMNTFaisceauIm1PrCh_Px2D")
      return eGeomMNTFaisceauIm1PrCh_Px2D;
   else if (aName=="eGeomMNTFaisceauIm1ZTerrain_Px1D")
      return eGeomMNTFaisceauIm1ZTerrain_Px1D;
   else if (aName=="eGeomMNTFaisceauIm1ZTerrain_Px2D")
      return eGeomMNTFaisceauIm1ZTerrain_Px2D;
   else if (aName=="eGeomPxBiDim")
      return eGeomPxBiDim;
   else if (aName=="eNoGeomMNT")
      return eNoGeomMNT;
   else if (aName=="eGeomMNTFaisceauPrChSpherik")
      return eGeomMNTFaisceauPrChSpherik;
  else
  {
      cout << aName << " is not a correct value for enum eModeGeomMNT\n" ;
      ELISE_ASSERT(false,"XML enum value error");
  }
  return (eModeGeomMNT) 0;
}
void xml_init(eModeGeomMNT & aVal,cElXMLTree * aTree)
{
   aVal= Str2eModeGeomMNT(aTree->Contenu());
}
std::string  eToString(const eModeGeomMNT & anObj)
{
   if (anObj==eGeomMNTCarto)
      return  "eGeomMNTCarto";
   if (anObj==eGeomMNTEuclid)
      return  "eGeomMNTEuclid";
   if (anObj==eGeomMNTFaisceauIm1PrCh_Px1D)
      return  "eGeomMNTFaisceauIm1PrCh_Px1D";
   if (anObj==eGeomMNTFaisceauIm1PrCh_Px2D)
      return  "eGeomMNTFaisceauIm1PrCh_Px2D";
   if (anObj==eGeomMNTFaisceauIm1ZTerrain_Px1D)
      return  "eGeomMNTFaisceauIm1ZTerrain_Px1D";
   if (anObj==eGeomMNTFaisceauIm1ZTerrain_Px2D)
      return  "eGeomMNTFaisceauIm1ZTerrain_Px2D";
   if (anObj==eGeomPxBiDim)
      return  "eGeomPxBiDim";
   if (anObj==eNoGeomMNT)
      return  "eNoGeomMNT";
   if (anObj==eGeomMNTFaisceauPrChSpherik)
      return  "eGeomMNTFaisceauPrChSpherik";
 std::cout << "Enum = eModeGeomMNT\n";
   ELISE_ASSERT(false,"Bad Value in eToString for enum value ");
   return "";
}

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eModeGeomMNT & anObj)
{
      return  cElXMLTree::ValueNode(aNameTag,eToString(anObj));
}

void  BinaryDumpInFile(ELISE_fp & aFp,const eModeGeomMNT & anObj)
{
   BinaryDumpInFile(aFp,int(anObj));
}

void  BinaryUnDumpFromFile(eModeGeomMNT & anObj,ELISE_fp & aFp)
{
   int aIVal;
   BinaryUnDumpFromFile(aIVal,aFp);
   anObj=(eModeGeomMNT) aIVal;
}

std::string  Mangling( eModeGeomMNT *) {return "7F86042F81218AFFFD3F";};

eModeBinSift  Str2eModeBinSift(const std::string & aName)
{
   if (aName=="eModeLeBrisPP")
      return eModeLeBrisPP;
   else if (aName=="eModeAutopano")
      return eModeAutopano;
  else
  {
      cout << aName << " is not a correct value for enum eModeBinSift\n" ;
      ELISE_ASSERT(false,"XML enum value error");
  }
  return (eModeBinSift) 0;
}
void xml_init(eModeBinSift & aVal,cElXMLTree * aTree)
{
   aVal= Str2eModeBinSift(aTree->Contenu());
}
std::string  eToString(const eModeBinSift & anObj)
{
   if (anObj==eModeLeBrisPP)
      return  "eModeLeBrisPP";
   if (anObj==eModeAutopano)
      return  "eModeAutopano";
 std::cout << "Enum = eModeBinSift\n";
   ELISE_ASSERT(false,"Bad Value in eToString for enum value ");
   return "";
}

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eModeBinSift & anObj)
{
      return  cElXMLTree::ValueNode(aNameTag,eToString(anObj));
}

void  BinaryDumpInFile(ELISE_fp & aFp,const eModeBinSift & anObj)
{
   BinaryDumpInFile(aFp,int(anObj));
}

void  BinaryUnDumpFromFile(eModeBinSift & anObj,ELISE_fp & aFp)
{
   int aIVal;
   BinaryUnDumpFromFile(aIVal,aFp);
   anObj=(eModeBinSift) aIVal;
}

std::string  Mangling( eModeBinSift *) {return "F4DC105633F35CDEFE3F";};

eModeSolveurEq  Str2eModeSolveurEq(const std::string & aName)
{
   if (aName=="eSysPlein")
      return eSysPlein;
   else if (aName=="eSysCreuxMap")
      return eSysCreuxMap;
   else if (aName=="eSysCreuxFixe")
      return eSysCreuxFixe;
   else if (aName=="eSysL1Barrodale")
      return eSysL1Barrodale;
   else if (aName=="eSysL2BlocSym")
      return eSysL2BlocSym;
  else
  {
      cout << aName << " is not a correct value for enum eModeSolveurEq\n" ;
      ELISE_ASSERT(false,"XML enum value error");
  }
  return (eModeSolveurEq) 0;
}
void xml_init(eModeSolveurEq & aVal,cElXMLTree * aTree)
{
   aVal= Str2eModeSolveurEq(aTree->Contenu());
}
std::string  eToString(const eModeSolveurEq & anObj)
{
   if (anObj==eSysPlein)
      return  "eSysPlein";
   if (anObj==eSysCreuxMap)
      return  "eSysCreuxMap";
   if (anObj==eSysCreuxFixe)
      return  "eSysCreuxFixe";
   if (anObj==eSysL1Barrodale)
      return  "eSysL1Barrodale";
   if (anObj==eSysL2BlocSym)
      return  "eSysL2BlocSym";
 std::cout << "Enum = eModeSolveurEq\n";
   ELISE_ASSERT(false,"Bad Value in eToString for enum value ");
   return "";
}

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eModeSolveurEq & anObj)
{
      return  cElXMLTree::ValueNode(aNameTag,eToString(anObj));
}

void  BinaryDumpInFile(ELISE_fp & aFp,const eModeSolveurEq & anObj)
{
   BinaryDumpInFile(aFp,int(anObj));
}

void  BinaryUnDumpFromFile(eModeSolveurEq & anObj,ELISE_fp & aFp)
{
   int aIVal;
   BinaryUnDumpFromFile(aIVal,aFp);
   anObj=(eModeSolveurEq) aIVal;
}

std::string  Mangling( eModeSolveurEq *) {return "AEFB457AE0ADEFC1FE3F";};

eUniteAngulaire  Str2eUniteAngulaire(const std::string & aName)
{
   if (aName=="eUniteAngleDegre")
      return eUniteAngleDegre;
   else if (aName=="eUniteAngleGrade")
      return eUniteAngleGrade;
   else if (aName=="eUniteAngleRadian")
      return eUniteAngleRadian;
   else if (aName=="eUniteAngleUnknown")
      return eUniteAngleUnknown;
  else
  {
      cout << aName << " is not a correct value for enum eUniteAngulaire\n" ;
      ELISE_ASSERT(false,"XML enum value error");
  }
  return (eUniteAngulaire) 0;
}
void xml_init(eUniteAngulaire & aVal,cElXMLTree * aTree)
{
   aVal= Str2eUniteAngulaire(aTree->Contenu());
}
std::string  eToString(const eUniteAngulaire & anObj)
{
   if (anObj==eUniteAngleDegre)
      return  "eUniteAngleDegre";
   if (anObj==eUniteAngleGrade)
      return  "eUniteAngleGrade";
   if (anObj==eUniteAngleRadian)
      return  "eUniteAngleRadian";
   if (anObj==eUniteAngleUnknown)
      return  "eUniteAngleUnknown";
 std::cout << "Enum = eUniteAngulaire\n";
   ELISE_ASSERT(false,"Bad Value in eToString for enum value ");
   return "";
}

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eUniteAngulaire & anObj)
{
      return  cElXMLTree::ValueNode(aNameTag,eToString(anObj));
}

void  BinaryDumpInFile(ELISE_fp & aFp,const eUniteAngulaire & anObj)
{
   BinaryDumpInFile(aFp,int(anObj));
}

void  BinaryUnDumpFromFile(eUniteAngulaire & anObj,ELISE_fp & aFp)
{
   int aIVal;
   BinaryUnDumpFromFile(aIVal,aFp);
   anObj=(eUniteAngulaire) aIVal;
}

std::string  Mangling( eUniteAngulaire *) {return "25364EBD43C8E5C9FE3F";};

eDegreLiberteCPP  Str2eDegreLiberteCPP(const std::string & aName)
{
   if (aName=="eCPPFiges")
      return eCPPFiges;
   else if (aName=="eCPPLies")
      return eCPPLies;
   else if (aName=="eCPPLibres")
      return eCPPLibres;
  else
  {
      cout << aName << " is not a correct value for enum eDegreLiberteCPP\n" ;
      ELISE_ASSERT(false,"XML enum value error");
  }
  return (eDegreLiberteCPP) 0;
}
void xml_init(eDegreLiberteCPP & aVal,cElXMLTree * aTree)
{
   aVal= Str2eDegreLiberteCPP(aTree->Contenu());
}
std::string  eToString(const eDegreLiberteCPP & anObj)
{
   if (anObj==eCPPFiges)
      return  "eCPPFiges";
   if (anObj==eCPPLies)
      return  "eCPPLies";
   if (anObj==eCPPLibres)
      return  "eCPPLibres";
 std::cout << "Enum = eDegreLiberteCPP\n";
   ELISE_ASSERT(false,"Bad Value in eToString for enum value ");
   return "";
}

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eDegreLiberteCPP & anObj)
{
      return  cElXMLTree::ValueNode(aNameTag,eToString(anObj));
}

void  BinaryDumpInFile(ELISE_fp & aFp,const eDegreLiberteCPP & anObj)
{
   BinaryDumpInFile(aFp,int(anObj));
}

void  BinaryUnDumpFromFile(eDegreLiberteCPP & anObj,ELISE_fp & aFp)
{
   int aIVal;
   BinaryUnDumpFromFile(aIVal,aFp);
   anObj=(eDegreLiberteCPP) aIVal;
}

std::string  Mangling( eDegreLiberteCPP *) {return "48C3B3CCC34EB0F5FB3F";};

eModelesCalibUnif  Str2eModelesCalibUnif(const std::string & aName)
{
   if (aName=="eModeleEbner")
      return eModeleEbner;
   else if (aName=="eModeleDCBrown")
      return eModeleDCBrown;
   else if (aName=="eModelePolyDeg2")
      return eModelePolyDeg2;
   else if (aName=="eModelePolyDeg3")
      return eModelePolyDeg3;
   else if (aName=="eModelePolyDeg4")
      return eModelePolyDeg4;
   else if (aName=="eModelePolyDeg5")
      return eModelePolyDeg5;
   else if (aName=="eModelePolyDeg6")
      return eModelePolyDeg6;
   else if (aName=="eModelePolyDeg7")
      return eModelePolyDeg7;
   else if (aName=="eModele_FishEye_10_5_5")
      return eModele_FishEye_10_5_5;
   else if (aName=="eModele_EquiSolid_FishEye_10_5_5")
      return eModele_EquiSolid_FishEye_10_5_5;
   else if (aName=="eModele_DRad_PPaEqPPs")
      return eModele_DRad_PPaEqPPs;
   else if (aName=="eModele_Fraser_PPaEqPPs")
      return eModele_Fraser_PPaEqPPs;
   else if (aName=="eModeleRadFour7x2")
      return eModeleRadFour7x2;
   else if (aName=="eModeleRadFour11x2")
      return eModeleRadFour11x2;
   else if (aName=="eModeleRadFour15x2")
      return eModeleRadFour15x2;
   else if (aName=="eModeleRadFour19x2")
      return eModeleRadFour19x2;
  else
  {
      cout << aName << " is not a correct value for enum eModelesCalibUnif\n" ;
      ELISE_ASSERT(false,"XML enum value error");
  }
  return (eModelesCalibUnif) 0;
}
void xml_init(eModelesCalibUnif & aVal,cElXMLTree * aTree)
{
   aVal= Str2eModelesCalibUnif(aTree->Contenu());
}
std::string  eToString(const eModelesCalibUnif & anObj)
{
   if (anObj==eModeleEbner)
      return  "eModeleEbner";
   if (anObj==eModeleDCBrown)
      return  "eModeleDCBrown";
   if (anObj==eModelePolyDeg2)
      return  "eModelePolyDeg2";
   if (anObj==eModelePolyDeg3)
      return  "eModelePolyDeg3";
   if (anObj==eModelePolyDeg4)
      return  "eModelePolyDeg4";
   if (anObj==eModelePolyDeg5)
      return  "eModelePolyDeg5";
   if (anObj==eModelePolyDeg6)
      return  "eModelePolyDeg6";
   if (anObj==eModelePolyDeg7)
      return  "eModelePolyDeg7";
   if (anObj==eModele_FishEye_10_5_5)
      return  "eModele_FishEye_10_5_5";
   if (anObj==eModele_EquiSolid_FishEye_10_5_5)
      return  "eModele_EquiSolid_FishEye_10_5_5";
   if (anObj==eModele_DRad_PPaEqPPs)
      return  "eModele_DRad_PPaEqPPs";
   if (anObj==eModele_Fraser_PPaEqPPs)
      return  "eModele_Fraser_PPaEqPPs";
   if (anObj==eModeleRadFour7x2)
      return  "eModeleRadFour7x2";
   if (anObj==eModeleRadFour11x2)
      return  "eModeleRadFour11x2";
   if (anObj==eModeleRadFour15x2)
      return  "eModeleRadFour15x2";
   if (anObj==eModeleRadFour19x2)
      return  "eModeleRadFour19x2";
 std::cout << "Enum = eModelesCalibUnif\n";
   ELISE_ASSERT(false,"Bad Value in eToString for enum value ");
   return "";
}

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eModelesCalibUnif & anObj)
{
      return  cElXMLTree::ValueNode(aNameTag,eToString(anObj));
}

void  BinaryDumpInFile(ELISE_fp & aFp,const eModelesCalibUnif & anObj)
{
   BinaryDumpInFile(aFp,int(anObj));
}

void  BinaryUnDumpFromFile(eModelesCalibUnif & anObj,ELISE_fp & aFp)
{
   int aIVal;
   BinaryUnDumpFromFile(aIVal,aFp);
   anObj=(eModelesCalibUnif) aIVal;
}

std::string  Mangling( eModelesCalibUnif *) {return "C42CEA817E8FC5ACFCBF";};

eTypeProjectionCam  Str2eTypeProjectionCam(const std::string & aName)
{
   if (aName=="eProjStenope")
      return eProjStenope;
   else if (aName=="eProjOrthographique")
      return eProjOrthographique;
   else if (aName=="eProjGrid")
      return eProjGrid;
  else
  {
      cout << aName << " is not a correct value for enum eTypeProjectionCam\n" ;
      ELISE_ASSERT(false,"XML enum value error");
  }
  return (eTypeProjectionCam) 0;
}
void xml_init(eTypeProjectionCam & aVal,cElXMLTree * aTree)
{
   aVal= Str2eTypeProjectionCam(aTree->Contenu());
}
std::string  eToString(const eTypeProjectionCam & anObj)
{
   if (anObj==eProjStenope)
      return  "eProjStenope";
   if (anObj==eProjOrthographique)
      return  "eProjOrthographique";
   if (anObj==eProjGrid)
      return  "eProjGrid";
 std::cout << "Enum = eTypeProjectionCam\n";
   ELISE_ASSERT(false,"Bad Value in eToString for enum value ");
   return "";
}

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eTypeProjectionCam & anObj)
{
      return  cElXMLTree::ValueNode(aNameTag,eToString(anObj));
}

void  BinaryDumpInFile(ELISE_fp & aFp,const eTypeProjectionCam & anObj)
{
   BinaryDumpInFile(aFp,int(anObj));
}

void  BinaryUnDumpFromFile(eTypeProjectionCam & anObj,ELISE_fp & aFp)
{
   int aIVal;
   BinaryUnDumpFromFile(aIVal,aFp);
   anObj=(eTypeProjectionCam) aIVal;
}

std::string  Mangling( eTypeProjectionCam *) {return "D0797F2BB67678ADFD3F";};

eTypeCoord  Str2eTypeCoord(const std::string & aName)
{
   if (aName=="eTC_WGS84")
      return eTC_WGS84;
   else if (aName=="eTC_GeoCentr")
      return eTC_GeoCentr;
   else if (aName=="eTC_RTL")
      return eTC_RTL;
   else if (aName=="eTC_Polyn")
      return eTC_Polyn;
   else if (aName=="eTC_Unknown")
      return eTC_Unknown;
   else if (aName=="eTC_Lambert93")
      return eTC_Lambert93;
   else if (aName=="eTC_LambertCC")
      return eTC_LambertCC;
   else if (aName=="eTC_Proj4")
      return eTC_Proj4;
  else
  {
      cout << aName << " is not a correct value for enum eTypeCoord\n" ;
      ELISE_ASSERT(false,"XML enum value error");
  }
  return (eTypeCoord) 0;
}
void xml_init(eTypeCoord & aVal,cElXMLTree * aTree)
{
   aVal= Str2eTypeCoord(aTree->Contenu());
}
std::string  eToString(const eTypeCoord & anObj)
{
   if (anObj==eTC_WGS84)
      return  "eTC_WGS84";
   if (anObj==eTC_GeoCentr)
      return  "eTC_GeoCentr";
   if (anObj==eTC_RTL)
      return  "eTC_RTL";
   if (anObj==eTC_Polyn)
      return  "eTC_Polyn";
   if (anObj==eTC_Unknown)
      return  "eTC_Unknown";
   if (anObj==eTC_Lambert93)
      return  "eTC_Lambert93";
   if (anObj==eTC_LambertCC)
      return  "eTC_LambertCC";
   if (anObj==eTC_Proj4)
      return  "eTC_Proj4";
 std::cout << "Enum = eTypeCoord\n";
   ELISE_ASSERT(false,"Bad Value in eToString for enum value ");
   return "";
}

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eTypeCoord & anObj)
{
      return  cElXMLTree::ValueNode(aNameTag,eToString(anObj));
}

void  BinaryDumpInFile(ELISE_fp & aFp,const eTypeCoord & anObj)
{
   BinaryDumpInFile(aFp,int(anObj));
}

void  BinaryUnDumpFromFile(eTypeCoord & anObj,ELISE_fp & aFp)
{
   int aIVal;
   BinaryUnDumpFromFile(aIVal,aFp);
   anObj=(eTypeCoord) aIVal;
}

std::string  Mangling( eTypeCoord *) {return "56B245E315A254CCFE3F";};


std::string & cMicMacConfiguration::DirInstall()
{
   return mDirInstall;
}

const std::string & cMicMacConfiguration::DirInstall()const 
{
   return mDirInstall;
}


int & cMicMacConfiguration::NbProcess()
{
   return mNbProcess;
}

const int & cMicMacConfiguration::NbProcess()const 
{
   return mNbProcess;
}

void  BinaryUnDumpFromFile(cMicMacConfiguration & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.DirInstall(),aFp);
    BinaryUnDumpFromFile(anObj.NbProcess(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cMicMacConfiguration & anObj)
{
    BinaryDumpInFile(aFp,anObj.DirInstall());
    BinaryDumpInFile(aFp,anObj.NbProcess());
}

cElXMLTree * ToXMLTree(const cMicMacConfiguration & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"MicMacConfiguration",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("DirInstall"),anObj.DirInstall())->ReTagThis("DirInstall"));
   aRes->AddFils(::ToXMLTree(std::string("NbProcess"),anObj.NbProcess())->ReTagThis("NbProcess"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cMicMacConfiguration & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.DirInstall(),aTree->Get("DirInstall",1)); //tototo 

   xml_init(anObj.NbProcess(),aTree->Get("NbProcess",1)); //tototo 
}

std::string  Mangling( cMicMacConfiguration *) {return "5B71172C2157C797FE3F";};


eTypeCoord & cBasicSystemeCoord::TypeCoord()
{
   return mTypeCoord;
}

const eTypeCoord & cBasicSystemeCoord::TypeCoord()const 
{
   return mTypeCoord;
}


std::vector< double > & cBasicSystemeCoord::AuxR()
{
   return mAuxR;
}

const std::vector< double > & cBasicSystemeCoord::AuxR()const 
{
   return mAuxR;
}


std::vector< int > & cBasicSystemeCoord::AuxI()
{
   return mAuxI;
}

const std::vector< int > & cBasicSystemeCoord::AuxI()const 
{
   return mAuxI;
}


std::vector< std::string > & cBasicSystemeCoord::AuxStr()
{
   return mAuxStr;
}

const std::vector< std::string > & cBasicSystemeCoord::AuxStr()const 
{
   return mAuxStr;
}


cTplValGesInit< bool > & cBasicSystemeCoord::ByFile()
{
   return mByFile;
}

const cTplValGesInit< bool > & cBasicSystemeCoord::ByFile()const 
{
   return mByFile;
}


std::vector< eUniteAngulaire > & cBasicSystemeCoord::AuxRUnite()
{
   return mAuxRUnite;
}

const std::vector< eUniteAngulaire > & cBasicSystemeCoord::AuxRUnite()const 
{
   return mAuxRUnite;
}

void  BinaryUnDumpFromFile(cBasicSystemeCoord & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.TypeCoord(),aFp);
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             double aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.AuxR().push_back(aVal);
        }
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             int aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.AuxI().push_back(aVal);
        }
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             std::string aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.AuxStr().push_back(aVal);
        }
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ByFile().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ByFile().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ByFile().SetNoInit();
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             eUniteAngulaire aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.AuxRUnite().push_back(aVal);
        }
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cBasicSystemeCoord & anObj)
{
    BinaryDumpInFile(aFp,anObj.TypeCoord());
    BinaryDumpInFile(aFp,(int)anObj.AuxR().size());
    for(  std::vector< double >::const_iterator iT=anObj.AuxR().begin();
         iT!=anObj.AuxR().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,(int)anObj.AuxI().size());
    for(  std::vector< int >::const_iterator iT=anObj.AuxI().begin();
         iT!=anObj.AuxI().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,(int)anObj.AuxStr().size());
    for(  std::vector< std::string >::const_iterator iT=anObj.AuxStr().begin();
         iT!=anObj.AuxStr().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.ByFile().IsInit());
    if (anObj.ByFile().IsInit()) BinaryDumpInFile(aFp,anObj.ByFile().Val());
    BinaryDumpInFile(aFp,(int)anObj.AuxRUnite().size());
    for(  std::vector< eUniteAngulaire >::const_iterator iT=anObj.AuxRUnite().begin();
         iT!=anObj.AuxRUnite().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
}

cElXMLTree * ToXMLTree(const cBasicSystemeCoord & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"BasicSystemeCoord",eXMLBranche);
   aRes->AddFils(ToXMLTree(std::string("TypeCoord"),anObj.TypeCoord())->ReTagThis("TypeCoord"));
  for
  (       std::vector< double >::const_iterator it=anObj.AuxR().begin();
      it !=anObj.AuxR().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("AuxR"),(*it))->ReTagThis("AuxR"));
  for
  (       std::vector< int >::const_iterator it=anObj.AuxI().begin();
      it !=anObj.AuxI().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("AuxI"),(*it))->ReTagThis("AuxI"));
  for
  (       std::vector< std::string >::const_iterator it=anObj.AuxStr().begin();
      it !=anObj.AuxStr().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("AuxStr"),(*it))->ReTagThis("AuxStr"));
   if (anObj.ByFile().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ByFile"),anObj.ByFile().Val())->ReTagThis("ByFile"));
  for
  (       std::vector< eUniteAngulaire >::const_iterator it=anObj.AuxRUnite().begin();
      it !=anObj.AuxRUnite().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree(std::string("AuxRUnite"),(*it))->ReTagThis("AuxRUnite"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cBasicSystemeCoord & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.TypeCoord(),aTree->Get("TypeCoord",1)); //tototo 

   xml_init(anObj.AuxR(),aTree->GetAll("AuxR",false,1));

   xml_init(anObj.AuxI(),aTree->GetAll("AuxI",false,1));

   xml_init(anObj.AuxStr(),aTree->GetAll("AuxStr",false,1));

   xml_init(anObj.ByFile(),aTree->Get("ByFile",1),bool(false)); //tototo 

   xml_init(anObj.AuxRUnite(),aTree->GetAll("AuxRUnite",false,1));
}

std::string  Mangling( cBasicSystemeCoord *) {return "F9740B28E8AC37A8FE3F";};


cTplValGesInit< std::string > & cSystemeCoord::Comment()
{
   return mComment;
}

const cTplValGesInit< std::string > & cSystemeCoord::Comment()const 
{
   return mComment;
}


std::vector< cBasicSystemeCoord > & cSystemeCoord::BSC()
{
   return mBSC;
}

const std::vector< cBasicSystemeCoord > & cSystemeCoord::BSC()const 
{
   return mBSC;
}

void  BinaryUnDumpFromFile(cSystemeCoord & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Comment().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Comment().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Comment().SetNoInit();
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cBasicSystemeCoord aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.BSC().push_back(aVal);
        }
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cSystemeCoord & anObj)
{
    BinaryDumpInFile(aFp,anObj.Comment().IsInit());
    if (anObj.Comment().IsInit()) BinaryDumpInFile(aFp,anObj.Comment().Val());
    BinaryDumpInFile(aFp,(int)anObj.BSC().size());
    for(  std::vector< cBasicSystemeCoord >::const_iterator iT=anObj.BSC().begin();
         iT!=anObj.BSC().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
}

cElXMLTree * ToXMLTree(const cSystemeCoord & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"SystemeCoord",eXMLBranche);
   if (anObj.Comment().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Comment"),anObj.Comment().Val())->ReTagThis("Comment"));
  for
  (       std::vector< cBasicSystemeCoord >::const_iterator it=anObj.BSC().begin();
      it !=anObj.BSC().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("BSC"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cSystemeCoord & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Comment(),aTree->Get("Comment",1)); //tototo 

   xml_init(anObj.BSC(),aTree->GetAll("BSC",false,1));
}

std::string  Mangling( cSystemeCoord *) {return "3203392FE4A68BA9FF3F";};


cSystemeCoord & cChangementCoordonnees::SystemeSource()
{
   return mSystemeSource;
}

const cSystemeCoord & cChangementCoordonnees::SystemeSource()const 
{
   return mSystemeSource;
}


cSystemeCoord & cChangementCoordonnees::SystemeCible()
{
   return mSystemeCible;
}

const cSystemeCoord & cChangementCoordonnees::SystemeCible()const 
{
   return mSystemeCible;
}

void  BinaryUnDumpFromFile(cChangementCoordonnees & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.SystemeSource(),aFp);
    BinaryUnDumpFromFile(anObj.SystemeCible(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cChangementCoordonnees & anObj)
{
    BinaryDumpInFile(aFp,anObj.SystemeSource());
    BinaryDumpInFile(aFp,anObj.SystemeCible());
}

cElXMLTree * ToXMLTree(const cChangementCoordonnees & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ChangementCoordonnees",eXMLBranche);
   aRes->AddFils(ToXMLTree(anObj.SystemeSource())->ReTagThis("SystemeSource"));
   aRes->AddFils(ToXMLTree(anObj.SystemeCible())->ReTagThis("SystemeCible"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cChangementCoordonnees & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.SystemeSource(),aTree->Get("SystemeSource",1)); //tototo 

   xml_init(anObj.SystemeCible(),aTree->Get("SystemeCible",1)); //tototo 
}

std::string  Mangling( cChangementCoordonnees *) {return "9FCAC63671DEEE87FDBF";};


std::string & cFileOriMnt::NameFileMnt()
{
   return mNameFileMnt;
}

const std::string & cFileOriMnt::NameFileMnt()const 
{
   return mNameFileMnt;
}


cTplValGesInit< std::string > & cFileOriMnt::NameFileMasque()
{
   return mNameFileMasque;
}

const cTplValGesInit< std::string > & cFileOriMnt::NameFileMasque()const 
{
   return mNameFileMasque;
}


Pt2di & cFileOriMnt::NombrePixels()
{
   return mNombrePixels;
}

const Pt2di & cFileOriMnt::NombrePixels()const 
{
   return mNombrePixels;
}


Pt2dr & cFileOriMnt::OriginePlani()
{
   return mOriginePlani;
}

const Pt2dr & cFileOriMnt::OriginePlani()const 
{
   return mOriginePlani;
}


Pt2dr & cFileOriMnt::ResolutionPlani()
{
   return mResolutionPlani;
}

const Pt2dr & cFileOriMnt::ResolutionPlani()const 
{
   return mResolutionPlani;
}


double & cFileOriMnt::OrigineAlti()
{
   return mOrigineAlti;
}

const double & cFileOriMnt::OrigineAlti()const 
{
   return mOrigineAlti;
}


double & cFileOriMnt::ResolutionAlti()
{
   return mResolutionAlti;
}

const double & cFileOriMnt::ResolutionAlti()const 
{
   return mResolutionAlti;
}


cTplValGesInit< int > & cFileOriMnt::NumZoneLambert()
{
   return mNumZoneLambert;
}

const cTplValGesInit< int > & cFileOriMnt::NumZoneLambert()const 
{
   return mNumZoneLambert;
}


eModeGeomMNT & cFileOriMnt::Geometrie()
{
   return mGeometrie;
}

const eModeGeomMNT & cFileOriMnt::Geometrie()const 
{
   return mGeometrie;
}


cTplValGesInit< Pt2dr > & cFileOriMnt::OrigineTgtLoc()
{
   return mOrigineTgtLoc;
}

const cTplValGesInit< Pt2dr > & cFileOriMnt::OrigineTgtLoc()const 
{
   return mOrigineTgtLoc;
}


cTplValGesInit< int > & cFileOriMnt::Rounding()
{
   return mRounding;
}

const cTplValGesInit< int > & cFileOriMnt::Rounding()const 
{
   return mRounding;
}

void  BinaryUnDumpFromFile(cFileOriMnt & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.NameFileMnt(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NameFileMasque().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NameFileMasque().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NameFileMasque().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.NombrePixels(),aFp);
    BinaryUnDumpFromFile(anObj.OriginePlani(),aFp);
    BinaryUnDumpFromFile(anObj.ResolutionPlani(),aFp);
    BinaryUnDumpFromFile(anObj.OrigineAlti(),aFp);
    BinaryUnDumpFromFile(anObj.ResolutionAlti(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NumZoneLambert().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NumZoneLambert().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NumZoneLambert().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.Geometrie(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.OrigineTgtLoc().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.OrigineTgtLoc().ValForcedForUnUmp(),aFp);
        }
        else  anObj.OrigineTgtLoc().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Rounding().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Rounding().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Rounding().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cFileOriMnt & anObj)
{
    BinaryDumpInFile(aFp,anObj.NameFileMnt());
    BinaryDumpInFile(aFp,anObj.NameFileMasque().IsInit());
    if (anObj.NameFileMasque().IsInit()) BinaryDumpInFile(aFp,anObj.NameFileMasque().Val());
    BinaryDumpInFile(aFp,anObj.NombrePixels());
    BinaryDumpInFile(aFp,anObj.OriginePlani());
    BinaryDumpInFile(aFp,anObj.ResolutionPlani());
    BinaryDumpInFile(aFp,anObj.OrigineAlti());
    BinaryDumpInFile(aFp,anObj.ResolutionAlti());
    BinaryDumpInFile(aFp,anObj.NumZoneLambert().IsInit());
    if (anObj.NumZoneLambert().IsInit()) BinaryDumpInFile(aFp,anObj.NumZoneLambert().Val());
    BinaryDumpInFile(aFp,anObj.Geometrie());
    BinaryDumpInFile(aFp,anObj.OrigineTgtLoc().IsInit());
    if (anObj.OrigineTgtLoc().IsInit()) BinaryDumpInFile(aFp,anObj.OrigineTgtLoc().Val());
    BinaryDumpInFile(aFp,anObj.Rounding().IsInit());
    if (anObj.Rounding().IsInit()) BinaryDumpInFile(aFp,anObj.Rounding().Val());
}

cElXMLTree * ToXMLTree(const cFileOriMnt & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"FileOriMnt",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("NameFileMnt"),anObj.NameFileMnt())->ReTagThis("NameFileMnt"));
   if (anObj.NameFileMasque().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NameFileMasque"),anObj.NameFileMasque().Val())->ReTagThis("NameFileMasque"));
   aRes->AddFils(::ToXMLTree(std::string("NombrePixels"),anObj.NombrePixels())->ReTagThis("NombrePixels"));
   aRes->AddFils(::ToXMLTree(std::string("OriginePlani"),anObj.OriginePlani())->ReTagThis("OriginePlani"));
   aRes->AddFils(::ToXMLTree(std::string("ResolutionPlani"),anObj.ResolutionPlani())->ReTagThis("ResolutionPlani"));
   aRes->AddFils(::ToXMLTree(std::string("OrigineAlti"),anObj.OrigineAlti())->ReTagThis("OrigineAlti"));
   aRes->AddFils(::ToXMLTree(std::string("ResolutionAlti"),anObj.ResolutionAlti())->ReTagThis("ResolutionAlti"));
   if (anObj.NumZoneLambert().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NumZoneLambert"),anObj.NumZoneLambert().Val())->ReTagThis("NumZoneLambert"));
   aRes->AddFils(ToXMLTree(std::string("Geometrie"),anObj.Geometrie())->ReTagThis("Geometrie"));
   if (anObj.OrigineTgtLoc().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("OrigineTgtLoc"),anObj.OrigineTgtLoc().Val())->ReTagThis("OrigineTgtLoc"));
   if (anObj.Rounding().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Rounding"),anObj.Rounding().Val())->ReTagThis("Rounding"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cFileOriMnt & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.NameFileMnt(),aTree->Get("NameFileMnt",1)); //tototo 

   xml_init(anObj.NameFileMasque(),aTree->Get("NameFileMasque",1)); //tototo 

   xml_init(anObj.NombrePixels(),aTree->Get("NombrePixels",1)); //tototo 

   xml_init(anObj.OriginePlani(),aTree->Get("OriginePlani",1)); //tototo 

   xml_init(anObj.ResolutionPlani(),aTree->Get("ResolutionPlani",1)); //tototo 

   xml_init(anObj.OrigineAlti(),aTree->Get("OrigineAlti",1)); //tototo 

   xml_init(anObj.ResolutionAlti(),aTree->Get("ResolutionAlti",1)); //tototo 

   xml_init(anObj.NumZoneLambert(),aTree->Get("NumZoneLambert",1)); //tototo 

   xml_init(anObj.Geometrie(),aTree->Get("Geometrie",1)); //tototo 

   xml_init(anObj.OrigineTgtLoc(),aTree->Get("OrigineTgtLoc",1)); //tototo 

   xml_init(anObj.Rounding(),aTree->Get("Rounding",1)); //tototo 
}

std::string  Mangling( cFileOriMnt *) {return "2E0F4D388417F3DEFD3F";};


Pt2dr & cRefPlani::Origine()
{
   return mOrigine;
}

const Pt2dr & cRefPlani::Origine()const 
{
   return mOrigine;
}


Pt2dr & cRefPlani::Resolution()
{
   return mResolution;
}

const Pt2dr & cRefPlani::Resolution()const 
{
   return mResolution;
}

void  BinaryUnDumpFromFile(cRefPlani & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Origine(),aFp);
    BinaryUnDumpFromFile(anObj.Resolution(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cRefPlani & anObj)
{
    BinaryDumpInFile(aFp,anObj.Origine());
    BinaryDumpInFile(aFp,anObj.Resolution());
}

cElXMLTree * ToXMLTree(const cRefPlani & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"RefPlani",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Origine"),anObj.Origine())->ReTagThis("Origine"));
   aRes->AddFils(::ToXMLTree(std::string("Resolution"),anObj.Resolution())->ReTagThis("Resolution"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cRefPlani & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Origine(),aTree->Get("Origine",1)); //tototo 

   xml_init(anObj.Resolution(),aTree->Get("Resolution",1)); //tototo 
}

std::string  Mangling( cRefPlani *) {return "735262B2342A5982FDBF";};


double & cRefAlti::Origine()
{
   return mOrigine;
}

const double & cRefAlti::Origine()const 
{
   return mOrigine;
}


double & cRefAlti::Resolution()
{
   return mResolution;
}

const double & cRefAlti::Resolution()const 
{
   return mResolution;
}

void  BinaryUnDumpFromFile(cRefAlti & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Origine(),aFp);
    BinaryUnDumpFromFile(anObj.Resolution(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cRefAlti & anObj)
{
    BinaryDumpInFile(aFp,anObj.Origine());
    BinaryDumpInFile(aFp,anObj.Resolution());
}

cElXMLTree * ToXMLTree(const cRefAlti & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"RefAlti",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Origine"),anObj.Origine())->ReTagThis("Origine"));
   aRes->AddFils(::ToXMLTree(std::string("Resolution"),anObj.Resolution())->ReTagThis("Resolution"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cRefAlti & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Origine(),aTree->Get("Origine",1)); //tototo 

   xml_init(anObj.Resolution(),aTree->Get("Resolution",1)); //tototo 
}

std::string  Mangling( cRefAlti *) {return "F4EB7A39B1B87F8DFC3F";};


cTplValGesInit< cRefAlti > & cGestionAltimetrie::RefAlti()
{
   return mRefAlti;
}

const cTplValGesInit< cRefAlti > & cGestionAltimetrie::RefAlti()const 
{
   return mRefAlti;
}


cTplValGesInit< double > & cGestionAltimetrie::ZMoyen()
{
   return mZMoyen;
}

const cTplValGesInit< double > & cGestionAltimetrie::ZMoyen()const 
{
   return mZMoyen;
}

void  BinaryUnDumpFromFile(cGestionAltimetrie & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.RefAlti().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.RefAlti().ValForcedForUnUmp(),aFp);
        }
        else  anObj.RefAlti().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ZMoyen().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ZMoyen().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ZMoyen().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cGestionAltimetrie & anObj)
{
    BinaryDumpInFile(aFp,anObj.RefAlti().IsInit());
    if (anObj.RefAlti().IsInit()) BinaryDumpInFile(aFp,anObj.RefAlti().Val());
    BinaryDumpInFile(aFp,anObj.ZMoyen().IsInit());
    if (anObj.ZMoyen().IsInit()) BinaryDumpInFile(aFp,anObj.ZMoyen().Val());
}

cElXMLTree * ToXMLTree(const cGestionAltimetrie & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"GestionAltimetrie",eXMLBranche);
   if (anObj.RefAlti().IsInit())
      aRes->AddFils(ToXMLTree(anObj.RefAlti().Val())->ReTagThis("RefAlti"));
   if (anObj.ZMoyen().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ZMoyen"),anObj.ZMoyen().Val())->ReTagThis("ZMoyen"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cGestionAltimetrie & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.RefAlti(),aTree->Get("RefAlti",1)); //tototo 

   xml_init(anObj.ZMoyen(),aTree->Get("ZMoyen",1)); //tototo 
}

std::string  Mangling( cGestionAltimetrie *) {return "6A99A4140E2242D1FE3F";};


cTplValGesInit< cSystemeCoord > & cXmlGeoRefFile::SysCo()
{
   return mSysCo;
}

const cTplValGesInit< cSystemeCoord > & cXmlGeoRefFile::SysCo()const 
{
   return mSysCo;
}


cRefPlani & cXmlGeoRefFile::RefPlani()
{
   return mRefPlani;
}

const cRefPlani & cXmlGeoRefFile::RefPlani()const 
{
   return mRefPlani;
}


cTplValGesInit< cRefAlti > & cXmlGeoRefFile::RefAlti()
{
   return GestionAltimetrie().RefAlti();
}

const cTplValGesInit< cRefAlti > & cXmlGeoRefFile::RefAlti()const 
{
   return GestionAltimetrie().RefAlti();
}


cTplValGesInit< double > & cXmlGeoRefFile::ZMoyen()
{
   return GestionAltimetrie().ZMoyen();
}

const cTplValGesInit< double > & cXmlGeoRefFile::ZMoyen()const 
{
   return GestionAltimetrie().ZMoyen();
}


cGestionAltimetrie & cXmlGeoRefFile::GestionAltimetrie()
{
   return mGestionAltimetrie;
}

const cGestionAltimetrie & cXmlGeoRefFile::GestionAltimetrie()const 
{
   return mGestionAltimetrie;
}

void  BinaryUnDumpFromFile(cXmlGeoRefFile & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SysCo().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SysCo().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SysCo().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.RefPlani(),aFp);
    BinaryUnDumpFromFile(anObj.GestionAltimetrie(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cXmlGeoRefFile & anObj)
{
    BinaryDumpInFile(aFp,anObj.SysCo().IsInit());
    if (anObj.SysCo().IsInit()) BinaryDumpInFile(aFp,anObj.SysCo().Val());
    BinaryDumpInFile(aFp,anObj.RefPlani());
    BinaryDumpInFile(aFp,anObj.GestionAltimetrie());
}

cElXMLTree * ToXMLTree(const cXmlGeoRefFile & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"XmlGeoRefFile",eXMLBranche);
   if (anObj.SysCo().IsInit())
      aRes->AddFils(ToXMLTree(anObj.SysCo().Val())->ReTagThis("SysCo"));
   aRes->AddFils(ToXMLTree(anObj.RefPlani())->ReTagThis("RefPlani"));
   aRes->AddFils(ToXMLTree(anObj.GestionAltimetrie())->ReTagThis("GestionAltimetrie"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cXmlGeoRefFile & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.SysCo(),aTree->Get("SysCo",1)); //tototo 

   xml_init(anObj.RefPlani(),aTree->Get("RefPlani",1)); //tototo 

   xml_init(anObj.GestionAltimetrie(),aTree->Get("GestionAltimetrie",1)); //tototo 
}

std::string  Mangling( cXmlGeoRefFile *) {return "0965D6D84049F1F8FE3F";};


std::string & cSpecExtractFromFile::NameFile()
{
   return mNameFile;
}

const std::string & cSpecExtractFromFile::NameFile()const 
{
   return mNameFile;
}


std::string & cSpecExtractFromFile::NameTag()
{
   return mNameTag;
}

const std::string & cSpecExtractFromFile::NameTag()const 
{
   return mNameTag;
}


cTplValGesInit< bool > & cSpecExtractFromFile::AutorizeNonExisting()
{
   return mAutorizeNonExisting;
}

const cTplValGesInit< bool > & cSpecExtractFromFile::AutorizeNonExisting()const 
{
   return mAutorizeNonExisting;
}

void  BinaryUnDumpFromFile(cSpecExtractFromFile & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.NameFile(),aFp);
    BinaryUnDumpFromFile(anObj.NameTag(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.AutorizeNonExisting().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.AutorizeNonExisting().ValForcedForUnUmp(),aFp);
        }
        else  anObj.AutorizeNonExisting().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cSpecExtractFromFile & anObj)
{
    BinaryDumpInFile(aFp,anObj.NameFile());
    BinaryDumpInFile(aFp,anObj.NameTag());
    BinaryDumpInFile(aFp,anObj.AutorizeNonExisting().IsInit());
    if (anObj.AutorizeNonExisting().IsInit()) BinaryDumpInFile(aFp,anObj.AutorizeNonExisting().Val());
}

cElXMLTree * ToXMLTree(const cSpecExtractFromFile & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"SpecExtractFromFile",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("NameFile"),anObj.NameFile())->ReTagThis("NameFile"));
   aRes->AddFils(::ToXMLTree(std::string("NameTag"),anObj.NameTag())->ReTagThis("NameTag"));
   if (anObj.AutorizeNonExisting().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("AutorizeNonExisting"),anObj.AutorizeNonExisting().Val())->ReTagThis("AutorizeNonExisting"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cSpecExtractFromFile & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.NameFile(),aTree->Get("NameFile",1)); //tototo 

   xml_init(anObj.NameTag(),aTree->Get("NameTag",1)); //tototo 

   xml_init(anObj.AutorizeNonExisting(),aTree->Get("AutorizeNonExisting",1),bool(false)); //tototo 
}

std::string  Mangling( cSpecExtractFromFile *) {return "EFA2B8327413BCA2FE3F";};


cTplValGesInit< std::string > & cSpecifFormatRaw::NameFile()
{
   return mNameFile;
}

const cTplValGesInit< std::string > & cSpecifFormatRaw::NameFile()const 
{
   return mNameFile;
}


Pt2di & cSpecifFormatRaw::Sz()
{
   return mSz;
}

const Pt2di & cSpecifFormatRaw::Sz()const 
{
   return mSz;
}


bool & cSpecifFormatRaw::MSBF()
{
   return mMSBF;
}

const bool & cSpecifFormatRaw::MSBF()const 
{
   return mMSBF;
}


int & cSpecifFormatRaw::NbBitsParPixel()
{
   return mNbBitsParPixel;
}

const int & cSpecifFormatRaw::NbBitsParPixel()const 
{
   return mNbBitsParPixel;
}


bool & cSpecifFormatRaw::IntegerType()
{
   return mIntegerType;
}

const bool & cSpecifFormatRaw::IntegerType()const 
{
   return mIntegerType;
}


bool & cSpecifFormatRaw::SignedType()
{
   return mSignedType;
}

const bool & cSpecifFormatRaw::SignedType()const 
{
   return mSignedType;
}


cTplValGesInit< std::string > & cSpecifFormatRaw::Camera()
{
   return mCamera;
}

const cTplValGesInit< std::string > & cSpecifFormatRaw::Camera()const 
{
   return mCamera;
}


cTplValGesInit< std::string > & cSpecifFormatRaw::BayPat()
{
   return mBayPat;
}

const cTplValGesInit< std::string > & cSpecifFormatRaw::BayPat()const 
{
   return mBayPat;
}


cTplValGesInit< double > & cSpecifFormatRaw::Focalmm()
{
   return mFocalmm;
}

const cTplValGesInit< double > & cSpecifFormatRaw::Focalmm()const 
{
   return mFocalmm;
}


cTplValGesInit< double > & cSpecifFormatRaw::FocalEqui35()
{
   return mFocalEqui35;
}

const cTplValGesInit< double > & cSpecifFormatRaw::FocalEqui35()const 
{
   return mFocalEqui35;
}

void  BinaryUnDumpFromFile(cSpecifFormatRaw & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NameFile().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NameFile().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NameFile().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.Sz(),aFp);
    BinaryUnDumpFromFile(anObj.MSBF(),aFp);
    BinaryUnDumpFromFile(anObj.NbBitsParPixel(),aFp);
    BinaryUnDumpFromFile(anObj.IntegerType(),aFp);
    BinaryUnDumpFromFile(anObj.SignedType(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Camera().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Camera().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Camera().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.BayPat().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.BayPat().ValForcedForUnUmp(),aFp);
        }
        else  anObj.BayPat().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Focalmm().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Focalmm().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Focalmm().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.FocalEqui35().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.FocalEqui35().ValForcedForUnUmp(),aFp);
        }
        else  anObj.FocalEqui35().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cSpecifFormatRaw & anObj)
{
    BinaryDumpInFile(aFp,anObj.NameFile().IsInit());
    if (anObj.NameFile().IsInit()) BinaryDumpInFile(aFp,anObj.NameFile().Val());
    BinaryDumpInFile(aFp,anObj.Sz());
    BinaryDumpInFile(aFp,anObj.MSBF());
    BinaryDumpInFile(aFp,anObj.NbBitsParPixel());
    BinaryDumpInFile(aFp,anObj.IntegerType());
    BinaryDumpInFile(aFp,anObj.SignedType());
    BinaryDumpInFile(aFp,anObj.Camera().IsInit());
    if (anObj.Camera().IsInit()) BinaryDumpInFile(aFp,anObj.Camera().Val());
    BinaryDumpInFile(aFp,anObj.BayPat().IsInit());
    if (anObj.BayPat().IsInit()) BinaryDumpInFile(aFp,anObj.BayPat().Val());
    BinaryDumpInFile(aFp,anObj.Focalmm().IsInit());
    if (anObj.Focalmm().IsInit()) BinaryDumpInFile(aFp,anObj.Focalmm().Val());
    BinaryDumpInFile(aFp,anObj.FocalEqui35().IsInit());
    if (anObj.FocalEqui35().IsInit()) BinaryDumpInFile(aFp,anObj.FocalEqui35().Val());
}

cElXMLTree * ToXMLTree(const cSpecifFormatRaw & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"SpecifFormatRaw",eXMLBranche);
   if (anObj.NameFile().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NameFile"),anObj.NameFile().Val())->ReTagThis("NameFile"));
   aRes->AddFils(::ToXMLTree(std::string("Sz"),anObj.Sz())->ReTagThis("Sz"));
   aRes->AddFils(::ToXMLTree(std::string("MSBF"),anObj.MSBF())->ReTagThis("MSBF"));
   aRes->AddFils(::ToXMLTree(std::string("NbBitsParPixel"),anObj.NbBitsParPixel())->ReTagThis("NbBitsParPixel"));
   aRes->AddFils(::ToXMLTree(std::string("IntegerType"),anObj.IntegerType())->ReTagThis("IntegerType"));
   aRes->AddFils(::ToXMLTree(std::string("SignedType"),anObj.SignedType())->ReTagThis("SignedType"));
   if (anObj.Camera().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Camera"),anObj.Camera().Val())->ReTagThis("Camera"));
   if (anObj.BayPat().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("BayPat"),anObj.BayPat().Val())->ReTagThis("BayPat"));
   if (anObj.Focalmm().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Focalmm"),anObj.Focalmm().Val())->ReTagThis("Focalmm"));
   if (anObj.FocalEqui35().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("FocalEqui35"),anObj.FocalEqui35().Val())->ReTagThis("FocalEqui35"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cSpecifFormatRaw & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.NameFile(),aTree->Get("NameFile",1)); //tototo 

   xml_init(anObj.Sz(),aTree->Get("Sz",1)); //tototo 

   xml_init(anObj.MSBF(),aTree->Get("MSBF",1)); //tototo 

   xml_init(anObj.NbBitsParPixel(),aTree->Get("NbBitsParPixel",1)); //tototo 

   xml_init(anObj.IntegerType(),aTree->Get("IntegerType",1)); //tototo 

   xml_init(anObj.SignedType(),aTree->Get("SignedType",1)); //tototo 

   xml_init(anObj.Camera(),aTree->Get("Camera",1)); //tototo 

   xml_init(anObj.BayPat(),aTree->Get("BayPat",1)); //tototo 

   xml_init(anObj.Focalmm(),aTree->Get("Focalmm",1)); //tototo 

   xml_init(anObj.FocalEqui35(),aTree->Get("FocalEqui35",1)); //tototo 
}

std::string  Mangling( cSpecifFormatRaw *) {return "1631189C8D02CBA8FB3F";};

eTotoModeGeomMEC  Str2eTotoModeGeomMEC(const std::string & aName)
{
   if (aName=="eTotoGeomMECIm1")
      return eTotoGeomMECIm1;
  else
  {
      cout << aName << " is not a correct value for enum eTotoModeGeomMEC\n" ;
      ELISE_ASSERT(false,"XML enum value error");
  }
  return (eTotoModeGeomMEC) 0;
}
void xml_init(eTotoModeGeomMEC & aVal,cElXMLTree * aTree)
{
   aVal= Str2eTotoModeGeomMEC(aTree->Contenu());
}
std::string  eToString(const eTotoModeGeomMEC & anObj)
{
   if (anObj==eTotoGeomMECIm1)
      return  "eTotoGeomMECIm1";
 std::cout << "Enum = eTotoModeGeomMEC\n";
   ELISE_ASSERT(false,"Bad Value in eToString for enum value ");
   return "";
}

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eTotoModeGeomMEC & anObj)
{
      return  cElXMLTree::ValueNode(aNameTag,eToString(anObj));
}

void  BinaryDumpInFile(ELISE_fp & aFp,const eTotoModeGeomMEC & anObj)
{
   BinaryDumpInFile(aFp,int(anObj));
}

void  BinaryUnDumpFromFile(eTotoModeGeomMEC & anObj,ELISE_fp & aFp)
{
   int aIVal;
   BinaryUnDumpFromFile(aIVal,aFp);
   anObj=(eTotoModeGeomMEC) aIVal;
}

std::string  Mangling( eTotoModeGeomMEC *) {return "76D08D033F65F3A7FE3F";};


std::string & cCM_Set::KeySet()
{
   return mKeySet;
}

const std::string & cCM_Set::KeySet()const 
{
   return mKeySet;
}


cTplValGesInit< std::string > & cCM_Set::KeyAssoc()
{
   return mKeyAssoc;
}

const cTplValGesInit< std::string > & cCM_Set::KeyAssoc()const 
{
   return mKeyAssoc;
}


std::string & cCM_Set::NameVarMap()
{
   return mNameVarMap;
}

const std::string & cCM_Set::NameVarMap()const 
{
   return mNameVarMap;
}

void  BinaryUnDumpFromFile(cCM_Set & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.KeySet(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.KeyAssoc().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.KeyAssoc().ValForcedForUnUmp(),aFp);
        }
        else  anObj.KeyAssoc().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.NameVarMap(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cCM_Set & anObj)
{
    BinaryDumpInFile(aFp,anObj.KeySet());
    BinaryDumpInFile(aFp,anObj.KeyAssoc().IsInit());
    if (anObj.KeyAssoc().IsInit()) BinaryDumpInFile(aFp,anObj.KeyAssoc().Val());
    BinaryDumpInFile(aFp,anObj.NameVarMap());
}

cElXMLTree * ToXMLTree(const cCM_Set & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"CM_Set",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("KeySet"),anObj.KeySet())->ReTagThis("KeySet"));
   if (anObj.KeyAssoc().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("KeyAssoc"),anObj.KeyAssoc().Val())->ReTagThis("KeyAssoc"));
   aRes->AddFils(::ToXMLTree(std::string("NameVarMap"),anObj.NameVarMap())->ReTagThis("NameVarMap"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cCM_Set & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.KeySet(),aTree->Get("KeySet",1)); //tototo 

   xml_init(anObj.KeyAssoc(),aTree->Get("KeyAssoc",1)); //tototo 

   xml_init(anObj.NameVarMap(),aTree->Get("NameVarMap",1)); //tototo 
}

std::string  Mangling( cCM_Set *) {return "A034136F47F70091FCBF";};


cTplValGesInit< std::string > & cModeCmdMapeur::CM_One()
{
   return mCM_One;
}

const cTplValGesInit< std::string > & cModeCmdMapeur::CM_One()const 
{
   return mCM_One;
}


std::string & cModeCmdMapeur::KeySet()
{
   return CM_Set().Val().KeySet();
}

const std::string & cModeCmdMapeur::KeySet()const 
{
   return CM_Set().Val().KeySet();
}


cTplValGesInit< std::string > & cModeCmdMapeur::KeyAssoc()
{
   return CM_Set().Val().KeyAssoc();
}

const cTplValGesInit< std::string > & cModeCmdMapeur::KeyAssoc()const 
{
   return CM_Set().Val().KeyAssoc();
}


std::string & cModeCmdMapeur::NameVarMap()
{
   return CM_Set().Val().NameVarMap();
}

const std::string & cModeCmdMapeur::NameVarMap()const 
{
   return CM_Set().Val().NameVarMap();
}


cTplValGesInit< cCM_Set > & cModeCmdMapeur::CM_Set()
{
   return mCM_Set;
}

const cTplValGesInit< cCM_Set > & cModeCmdMapeur::CM_Set()const 
{
   return mCM_Set;
}

void  BinaryUnDumpFromFile(cModeCmdMapeur & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.CM_One().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.CM_One().ValForcedForUnUmp(),aFp);
        }
        else  anObj.CM_One().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.CM_Set().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.CM_Set().ValForcedForUnUmp(),aFp);
        }
        else  anObj.CM_Set().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cModeCmdMapeur & anObj)
{
    BinaryDumpInFile(aFp,anObj.CM_One().IsInit());
    if (anObj.CM_One().IsInit()) BinaryDumpInFile(aFp,anObj.CM_One().Val());
    BinaryDumpInFile(aFp,anObj.CM_Set().IsInit());
    if (anObj.CM_Set().IsInit()) BinaryDumpInFile(aFp,anObj.CM_Set().Val());
}

cElXMLTree * ToXMLTree(const cModeCmdMapeur & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ModeCmdMapeur",eXMLBranche);
   if (anObj.CM_One().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("CM_One"),anObj.CM_One().Val())->ReTagThis("CM_One"));
   if (anObj.CM_Set().IsInit())
      aRes->AddFils(ToXMLTree(anObj.CM_Set().Val())->ReTagThis("CM_Set"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cModeCmdMapeur & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.CM_One(),aTree->Get("CM_One",1)); //tototo 

   xml_init(anObj.CM_Set(),aTree->Get("CM_Set",1)); //tototo 
}

std::string  Mangling( cModeCmdMapeur *) {return "08D14E7A84D6DF95FF3F";};


std::string & cCmdMapRel::KeyRel()
{
   return mKeyRel;
}

const std::string & cCmdMapRel::KeyRel()const 
{
   return mKeyRel;
}


std::string & cCmdMapRel::NameArc()
{
   return mNameArc;
}

const std::string & cCmdMapRel::NameArc()const 
{
   return mNameArc;
}

void  BinaryUnDumpFromFile(cCmdMapRel & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.KeyRel(),aFp);
    BinaryUnDumpFromFile(anObj.NameArc(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cCmdMapRel & anObj)
{
    BinaryDumpInFile(aFp,anObj.KeyRel());
    BinaryDumpInFile(aFp,anObj.NameArc());
}

cElXMLTree * ToXMLTree(const cCmdMapRel & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"CmdMapRel",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("KeyRel"),anObj.KeyRel())->ReTagThis("KeyRel"));
   aRes->AddFils(::ToXMLTree(std::string("NameArc"),anObj.NameArc())->ReTagThis("NameArc"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cCmdMapRel & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.KeyRel(),aTree->Get("KeyRel",1)); //tototo 

   xml_init(anObj.NameArc(),aTree->Get("NameArc",1)); //tototo 
}

std::string  Mangling( cCmdMapRel *) {return "C290D528177F4287FF3F";};


std::list< cCpleString > & cCMVA::NV()
{
   return mNV;
}

const std::list< cCpleString > & cCMVA::NV()const 
{
   return mNV;
}

void  BinaryUnDumpFromFile(cCMVA & anObj,ELISE_fp & aFp)
{
   { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cCpleString aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.NV().push_back(aVal);
        }
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cCMVA & anObj)
{
    BinaryDumpInFile(aFp,(int)anObj.NV().size());
    for(  std::list< cCpleString >::const_iterator iT=anObj.NV().begin();
         iT!=anObj.NV().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
}

cElXMLTree * ToXMLTree(const cCMVA & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"CMVA",eXMLBranche);
  for
  (       std::list< cCpleString >::const_iterator it=anObj.NV().begin();
      it !=anObj.NV().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("NV"),(*it))->ReTagThis("NV"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cCMVA & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.NV(),aTree->GetAll("NV",false,1));
}

std::string  Mangling( cCMVA *) {return "4A667F354B2CD882FD3F";};


bool & cCmdMappeur::ActivateCmdMap()
{
   return mActivateCmdMap;
}

const bool & cCmdMappeur::ActivateCmdMap()const 
{
   return mActivateCmdMap;
}


cTplValGesInit< std::string > & cCmdMappeur::CM_One()
{
   return ModeCmdMapeur().CM_One();
}

const cTplValGesInit< std::string > & cCmdMappeur::CM_One()const 
{
   return ModeCmdMapeur().CM_One();
}


std::string & cCmdMappeur::KeySet()
{
   return ModeCmdMapeur().CM_Set().Val().KeySet();
}

const std::string & cCmdMappeur::KeySet()const 
{
   return ModeCmdMapeur().CM_Set().Val().KeySet();
}


cTplValGesInit< std::string > & cCmdMappeur::KeyAssoc()
{
   return ModeCmdMapeur().CM_Set().Val().KeyAssoc();
}

const cTplValGesInit< std::string > & cCmdMappeur::KeyAssoc()const 
{
   return ModeCmdMapeur().CM_Set().Val().KeyAssoc();
}


std::string & cCmdMappeur::NameVarMap()
{
   return ModeCmdMapeur().CM_Set().Val().NameVarMap();
}

const std::string & cCmdMappeur::NameVarMap()const 
{
   return ModeCmdMapeur().CM_Set().Val().NameVarMap();
}


cTplValGesInit< cCM_Set > & cCmdMappeur::CM_Set()
{
   return ModeCmdMapeur().CM_Set();
}

const cTplValGesInit< cCM_Set > & cCmdMappeur::CM_Set()const 
{
   return ModeCmdMapeur().CM_Set();
}


cModeCmdMapeur & cCmdMappeur::ModeCmdMapeur()
{
   return mModeCmdMapeur;
}

const cModeCmdMapeur & cCmdMappeur::ModeCmdMapeur()const 
{
   return mModeCmdMapeur;
}


std::string & cCmdMappeur::KeyRel()
{
   return CmdMapRel().Val().KeyRel();
}

const std::string & cCmdMappeur::KeyRel()const 
{
   return CmdMapRel().Val().KeyRel();
}


std::string & cCmdMappeur::NameArc()
{
   return CmdMapRel().Val().NameArc();
}

const std::string & cCmdMappeur::NameArc()const 
{
   return CmdMapRel().Val().NameArc();
}


cTplValGesInit< cCmdMapRel > & cCmdMappeur::CmdMapRel()
{
   return mCmdMapRel;
}

const cTplValGesInit< cCmdMapRel > & cCmdMappeur::CmdMapRel()const 
{
   return mCmdMapRel;
}


std::list< cCMVA > & cCmdMappeur::CMVA()
{
   return mCMVA;
}

const std::list< cCMVA > & cCmdMappeur::CMVA()const 
{
   return mCMVA;
}


cTplValGesInit< std::string > & cCmdMappeur::ByMkF()
{
   return mByMkF;
}

const cTplValGesInit< std::string > & cCmdMappeur::ByMkF()const 
{
   return mByMkF;
}


cTplValGesInit< std::string > & cCmdMappeur::KeyTargetMkF()
{
   return mKeyTargetMkF;
}

const cTplValGesInit< std::string > & cCmdMappeur::KeyTargetMkF()const 
{
   return mKeyTargetMkF;
}

void  BinaryUnDumpFromFile(cCmdMappeur & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.ActivateCmdMap(),aFp);
    BinaryUnDumpFromFile(anObj.ModeCmdMapeur(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.CmdMapRel().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.CmdMapRel().ValForcedForUnUmp(),aFp);
        }
        else  anObj.CmdMapRel().SetNoInit();
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cCMVA aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.CMVA().push_back(aVal);
        }
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ByMkF().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ByMkF().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ByMkF().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.KeyTargetMkF().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.KeyTargetMkF().ValForcedForUnUmp(),aFp);
        }
        else  anObj.KeyTargetMkF().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cCmdMappeur & anObj)
{
    BinaryDumpInFile(aFp,anObj.ActivateCmdMap());
    BinaryDumpInFile(aFp,anObj.ModeCmdMapeur());
    BinaryDumpInFile(aFp,anObj.CmdMapRel().IsInit());
    if (anObj.CmdMapRel().IsInit()) BinaryDumpInFile(aFp,anObj.CmdMapRel().Val());
    BinaryDumpInFile(aFp,(int)anObj.CMVA().size());
    for(  std::list< cCMVA >::const_iterator iT=anObj.CMVA().begin();
         iT!=anObj.CMVA().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.ByMkF().IsInit());
    if (anObj.ByMkF().IsInit()) BinaryDumpInFile(aFp,anObj.ByMkF().Val());
    BinaryDumpInFile(aFp,anObj.KeyTargetMkF().IsInit());
    if (anObj.KeyTargetMkF().IsInit()) BinaryDumpInFile(aFp,anObj.KeyTargetMkF().Val());
}

cElXMLTree * ToXMLTree(const cCmdMappeur & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"CmdMappeur",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("ActivateCmdMap"),anObj.ActivateCmdMap())->ReTagThis("ActivateCmdMap"));
   aRes->AddFils(ToXMLTree(anObj.ModeCmdMapeur())->ReTagThis("ModeCmdMapeur"));
   if (anObj.CmdMapRel().IsInit())
      aRes->AddFils(ToXMLTree(anObj.CmdMapRel().Val())->ReTagThis("CmdMapRel"));
  for
  (       std::list< cCMVA >::const_iterator it=anObj.CMVA().begin();
      it !=anObj.CMVA().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("CMVA"));
   if (anObj.ByMkF().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ByMkF"),anObj.ByMkF().Val())->ReTagThis("ByMkF"));
   if (anObj.KeyTargetMkF().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("KeyTargetMkF"),anObj.KeyTargetMkF().Val())->ReTagThis("KeyTargetMkF"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cCmdMappeur & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.ActivateCmdMap(),aTree->Get("ActivateCmdMap",1)); //tototo 

   xml_init(anObj.ModeCmdMapeur(),aTree->Get("ModeCmdMapeur",1)); //tototo 

   xml_init(anObj.CmdMapRel(),aTree->Get("CmdMapRel",1)); //tototo 

   xml_init(anObj.CMVA(),aTree->GetAll("CMVA",false,1));

   xml_init(anObj.ByMkF(),aTree->Get("ByMkF",1)); //tototo 

   xml_init(anObj.KeyTargetMkF(),aTree->Get("KeyTargetMkF",1)); //tototo 
}

std::string  Mangling( cCmdMappeur *) {return "19F049934C430E87FEBF";};


std::list< std::string > & cOneCmdPar::OneCmdSer()
{
   return mOneCmdSer;
}

const std::list< std::string > & cOneCmdPar::OneCmdSer()const 
{
   return mOneCmdSer;
}

void  BinaryUnDumpFromFile(cOneCmdPar & anObj,ELISE_fp & aFp)
{
   { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             std::string aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.OneCmdSer().push_back(aVal);
        }
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cOneCmdPar & anObj)
{
    BinaryDumpInFile(aFp,(int)anObj.OneCmdSer().size());
    for(  std::list< std::string >::const_iterator iT=anObj.OneCmdSer().begin();
         iT!=anObj.OneCmdSer().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
}

cElXMLTree * ToXMLTree(const cOneCmdPar & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"OneCmdPar",eXMLBranche);
  for
  (       std::list< std::string >::const_iterator it=anObj.OneCmdSer().begin();
      it !=anObj.OneCmdSer().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("OneCmdSer"),(*it))->ReTagThis("OneCmdSer"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cOneCmdPar & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.OneCmdSer(),aTree->GetAll("OneCmdSer",false,1));
}

std::string  Mangling( cOneCmdPar *) {return "0AE4B59974B020B3FDBF";};


cTplValGesInit< std::string > & cCmdExePar::NameMkF()
{
   return mNameMkF;
}

const cTplValGesInit< std::string > & cCmdExePar::NameMkF()const 
{
   return mNameMkF;
}


std::list< cOneCmdPar > & cCmdExePar::OneCmdPar()
{
   return mOneCmdPar;
}

const std::list< cOneCmdPar > & cCmdExePar::OneCmdPar()const 
{
   return mOneCmdPar;
}

void  BinaryUnDumpFromFile(cCmdExePar & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NameMkF().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NameMkF().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NameMkF().SetNoInit();
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cOneCmdPar aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.OneCmdPar().push_back(aVal);
        }
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cCmdExePar & anObj)
{
    BinaryDumpInFile(aFp,anObj.NameMkF().IsInit());
    if (anObj.NameMkF().IsInit()) BinaryDumpInFile(aFp,anObj.NameMkF().Val());
    BinaryDumpInFile(aFp,(int)anObj.OneCmdPar().size());
    for(  std::list< cOneCmdPar >::const_iterator iT=anObj.OneCmdPar().begin();
         iT!=anObj.OneCmdPar().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
}

cElXMLTree * ToXMLTree(const cCmdExePar & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"CmdExePar",eXMLBranche);
   if (anObj.NameMkF().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NameMkF"),anObj.NameMkF().Val())->ReTagThis("NameMkF"));
  for
  (       std::list< cOneCmdPar >::const_iterator it=anObj.OneCmdPar().begin();
      it !=anObj.OneCmdPar().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("OneCmdPar"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cCmdExePar & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.NameMkF(),aTree->Get("NameMkF",1),std::string("Make_CmdExePar")); //tototo 

   xml_init(anObj.OneCmdPar(),aTree->GetAll("OneCmdPar",false,1));
}

std::string  Mangling( cCmdExePar *) {return "B1179A681150E1DCFE3F";};


std::string & cPt3drEntries::Key()
{
   return mKey;
}

const std::string & cPt3drEntries::Key()const 
{
   return mKey;
}


Pt3dr & cPt3drEntries::Val()
{
   return mVal;
}

const Pt3dr & cPt3drEntries::Val()const 
{
   return mVal;
}

void  BinaryUnDumpFromFile(cPt3drEntries & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Key(),aFp);
    BinaryUnDumpFromFile(anObj.Val(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cPt3drEntries & anObj)
{
    BinaryDumpInFile(aFp,anObj.Key());
    BinaryDumpInFile(aFp,anObj.Val());
}

cElXMLTree * ToXMLTree(const cPt3drEntries & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"Pt3drEntries",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Key"),anObj.Key())->ReTagThis("Key"));
   aRes->AddFils(ToXMLTree(std::string("Val"),anObj.Val())->ReTagThis("Val"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cPt3drEntries & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Key(),aTree->Get("Key",1)); //tototo 

   xml_init(anObj.Val(),aTree->Get("Val",1)); //tototo 
}

std::string  Mangling( cPt3drEntries *) {return "70589D0F9ECDC1AEFABF";};


std::string & cBasesPt3dr::NameBase()
{
   return mNameBase;
}

const std::string & cBasesPt3dr::NameBase()const 
{
   return mNameBase;
}


std::list< cPt3drEntries > & cBasesPt3dr::Pt3drEntries()
{
   return mPt3drEntries;
}

const std::list< cPt3drEntries > & cBasesPt3dr::Pt3drEntries()const 
{
   return mPt3drEntries;
}

void  BinaryUnDumpFromFile(cBasesPt3dr & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.NameBase(),aFp);
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cPt3drEntries aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.Pt3drEntries().push_back(aVal);
        }
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cBasesPt3dr & anObj)
{
    BinaryDumpInFile(aFp,anObj.NameBase());
    BinaryDumpInFile(aFp,(int)anObj.Pt3drEntries().size());
    for(  std::list< cPt3drEntries >::const_iterator iT=anObj.Pt3drEntries().begin();
         iT!=anObj.Pt3drEntries().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
}

cElXMLTree * ToXMLTree(const cBasesPt3dr & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"BasesPt3dr",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("NameBase"),anObj.NameBase())->ReTagThis("NameBase"));
  for
  (       std::list< cPt3drEntries >::const_iterator it=anObj.Pt3drEntries().begin();
      it !=anObj.Pt3drEntries().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("Pt3drEntries"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cBasesPt3dr & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.NameBase(),aTree->Get("NameBase",1)); //tototo 

   xml_init(anObj.Pt3drEntries(),aTree->GetAll("Pt3drEntries",false,1));
}

std::string  Mangling( cBasesPt3dr *) {return "21A14CC5147F6F94FF3F";};


std::string & cScalEntries::Key()
{
   return mKey;
}

const std::string & cScalEntries::Key()const 
{
   return mKey;
}


double & cScalEntries::Val()
{
   return mVal;
}

const double & cScalEntries::Val()const 
{
   return mVal;
}

void  BinaryUnDumpFromFile(cScalEntries & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Key(),aFp);
    BinaryUnDumpFromFile(anObj.Val(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cScalEntries & anObj)
{
    BinaryDumpInFile(aFp,anObj.Key());
    BinaryDumpInFile(aFp,anObj.Val());
}

cElXMLTree * ToXMLTree(const cScalEntries & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ScalEntries",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Key"),anObj.Key())->ReTagThis("Key"));
   aRes->AddFils(::ToXMLTree(std::string("Val"),anObj.Val())->ReTagThis("Val"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cScalEntries & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Key(),aTree->Get("Key",1)); //tototo 

   xml_init(anObj.Val(),aTree->Get("Val",1)); //tototo 
}

std::string  Mangling( cScalEntries *) {return "80DCA8833CF83F96FBBF";};


std::string & cBasesScal::NameBase()
{
   return mNameBase;
}

const std::string & cBasesScal::NameBase()const 
{
   return mNameBase;
}


std::list< cScalEntries > & cBasesScal::ScalEntries()
{
   return mScalEntries;
}

const std::list< cScalEntries > & cBasesScal::ScalEntries()const 
{
   return mScalEntries;
}

void  BinaryUnDumpFromFile(cBasesScal & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.NameBase(),aFp);
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cScalEntries aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.ScalEntries().push_back(aVal);
        }
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cBasesScal & anObj)
{
    BinaryDumpInFile(aFp,anObj.NameBase());
    BinaryDumpInFile(aFp,(int)anObj.ScalEntries().size());
    for(  std::list< cScalEntries >::const_iterator iT=anObj.ScalEntries().begin();
         iT!=anObj.ScalEntries().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
}

cElXMLTree * ToXMLTree(const cBasesScal & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"BasesScal",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("NameBase"),anObj.NameBase())->ReTagThis("NameBase"));
  for
  (       std::list< cScalEntries >::const_iterator it=anObj.ScalEntries().begin();
      it !=anObj.ScalEntries().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("ScalEntries"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cBasesScal & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.NameBase(),aTree->Get("NameBase",1)); //tototo 

   xml_init(anObj.ScalEntries(),aTree->GetAll("ScalEntries",false,1));
}

std::string  Mangling( cBasesScal *) {return "20877570057E0CADFD3F";};


std::list< cBasesPt3dr > & cBaseDataCD::BasesPt3dr()
{
   return mBasesPt3dr;
}

const std::list< cBasesPt3dr > & cBaseDataCD::BasesPt3dr()const 
{
   return mBasesPt3dr;
}


std::list< cBasesScal > & cBaseDataCD::BasesScal()
{
   return mBasesScal;
}

const std::list< cBasesScal > & cBaseDataCD::BasesScal()const 
{
   return mBasesScal;
}

void  BinaryUnDumpFromFile(cBaseDataCD & anObj,ELISE_fp & aFp)
{
   { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cBasesPt3dr aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.BasesPt3dr().push_back(aVal);
        }
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cBasesScal aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.BasesScal().push_back(aVal);
        }
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cBaseDataCD & anObj)
{
    BinaryDumpInFile(aFp,(int)anObj.BasesPt3dr().size());
    for(  std::list< cBasesPt3dr >::const_iterator iT=anObj.BasesPt3dr().begin();
         iT!=anObj.BasesPt3dr().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,(int)anObj.BasesScal().size());
    for(  std::list< cBasesScal >::const_iterator iT=anObj.BasesScal().begin();
         iT!=anObj.BasesScal().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
}

cElXMLTree * ToXMLTree(const cBaseDataCD & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"BaseDataCD",eXMLBranche);
  for
  (       std::list< cBasesPt3dr >::const_iterator it=anObj.BasesPt3dr().begin();
      it !=anObj.BasesPt3dr().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("BasesPt3dr"));
  for
  (       std::list< cBasesScal >::const_iterator it=anObj.BasesScal().begin();
      it !=anObj.BasesScal().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("BasesScal"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cBaseDataCD & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.BasesPt3dr(),aTree->GetAll("BasesPt3dr",false,1));

   xml_init(anObj.BasesScal(),aTree->GetAll("BasesScal",false,1));
}

std::string  Mangling( cBaseDataCD *) {return "A702120845BA28F0FE3F";};


std::string & cParamVolChantierPhotogram::Directory()
{
   return mDirectory;
}

const std::string & cParamVolChantierPhotogram::Directory()const 
{
   return mDirectory;
}


cTplValGesInit< std::string > & cParamVolChantierPhotogram::DirOrientations()
{
   return mDirOrientations;
}

const cTplValGesInit< std::string > & cParamVolChantierPhotogram::DirOrientations()const 
{
   return mDirOrientations;
}


cElRegex_Ptr & cParamVolChantierPhotogram::NameSelector()
{
   return mNameSelector;
}

const cElRegex_Ptr & cParamVolChantierPhotogram::NameSelector()const 
{
   return mNameSelector;
}


cElRegex_Ptr & cParamVolChantierPhotogram::BandeIdSelector()
{
   return mBandeIdSelector;
}

const cElRegex_Ptr & cParamVolChantierPhotogram::BandeIdSelector()const 
{
   return mBandeIdSelector;
}


std::string & cParamVolChantierPhotogram::NomBandeId()
{
   return mNomBandeId;
}

const std::string & cParamVolChantierPhotogram::NomBandeId()const 
{
   return mNomBandeId;
}


std::string & cParamVolChantierPhotogram::NomIdInBande()
{
   return mNomIdInBande;
}

const std::string & cParamVolChantierPhotogram::NomIdInBande()const 
{
   return mNomIdInBande;
}


std::string & cParamVolChantierPhotogram::NomImage()
{
   return mNomImage;
}

const std::string & cParamVolChantierPhotogram::NomImage()const 
{
   return mNomImage;
}


cTplValGesInit< std::string > & cParamVolChantierPhotogram::DirImages()
{
   return mDirImages;
}

const cTplValGesInit< std::string > & cParamVolChantierPhotogram::DirImages()const 
{
   return mDirImages;
}

void  BinaryUnDumpFromFile(cParamVolChantierPhotogram & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Directory(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.DirOrientations().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.DirOrientations().ValForcedForUnUmp(),aFp);
        }
        else  anObj.DirOrientations().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.NameSelector(),aFp);
    BinaryUnDumpFromFile(anObj.BandeIdSelector(),aFp);
    BinaryUnDumpFromFile(anObj.NomBandeId(),aFp);
    BinaryUnDumpFromFile(anObj.NomIdInBande(),aFp);
    BinaryUnDumpFromFile(anObj.NomImage(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.DirImages().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.DirImages().ValForcedForUnUmp(),aFp);
        }
        else  anObj.DirImages().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cParamVolChantierPhotogram & anObj)
{
    BinaryDumpInFile(aFp,anObj.Directory());
    BinaryDumpInFile(aFp,anObj.DirOrientations().IsInit());
    if (anObj.DirOrientations().IsInit()) BinaryDumpInFile(aFp,anObj.DirOrientations().Val());
    BinaryDumpInFile(aFp,anObj.NameSelector());
    BinaryDumpInFile(aFp,anObj.BandeIdSelector());
    BinaryDumpInFile(aFp,anObj.NomBandeId());
    BinaryDumpInFile(aFp,anObj.NomIdInBande());
    BinaryDumpInFile(aFp,anObj.NomImage());
    BinaryDumpInFile(aFp,anObj.DirImages().IsInit());
    if (anObj.DirImages().IsInit()) BinaryDumpInFile(aFp,anObj.DirImages().Val());
}

cElXMLTree * ToXMLTree(const cParamVolChantierPhotogram & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ParamVolChantierPhotogram",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Directory"),anObj.Directory())->ReTagThis("Directory"));
   if (anObj.DirOrientations().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DirOrientations"),anObj.DirOrientations().Val())->ReTagThis("DirOrientations"));
   aRes->AddFils(::ToXMLTree(std::string("NameSelector"),anObj.NameSelector())->ReTagThis("NameSelector"));
   aRes->AddFils(::ToXMLTree(std::string("BandeIdSelector"),anObj.BandeIdSelector())->ReTagThis("BandeIdSelector"));
   aRes->AddFils(::ToXMLTree(std::string("NomBandeId"),anObj.NomBandeId())->ReTagThis("NomBandeId"));
   aRes->AddFils(::ToXMLTree(std::string("NomIdInBande"),anObj.NomIdInBande())->ReTagThis("NomIdInBande"));
   aRes->AddFils(::ToXMLTree(std::string("NomImage"),anObj.NomImage())->ReTagThis("NomImage"));
   if (anObj.DirImages().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DirImages"),anObj.DirImages().Val())->ReTagThis("DirImages"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cParamVolChantierPhotogram & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Directory(),aTree->Get("Directory",1)); //tototo 

   xml_init(anObj.DirOrientations(),aTree->Get("DirOrientations",1),std::string("")); //tototo 

   xml_init(anObj.NameSelector(),aTree->Get("NameSelector",1)); //tototo 

   xml_init(anObj.BandeIdSelector(),aTree->Get("BandeIdSelector",1)); //tototo 

   xml_init(anObj.NomBandeId(),aTree->Get("NomBandeId",1)); //tototo 

   xml_init(anObj.NomIdInBande(),aTree->Get("NomIdInBande",1)); //tototo 

   xml_init(anObj.NomImage(),aTree->Get("NomImage",1)); //tototo 

   xml_init(anObj.DirImages(),aTree->Get("DirImages",1),std::string("")); //tototo 
}

std::string  Mangling( cParamVolChantierPhotogram *) {return "2BBFCDD9DDF060E9FE3F";};


std::list< cParamVolChantierPhotogram > & cParamChantierPhotogram::ParamVolChantierPhotogram()
{
   return mParamVolChantierPhotogram;
}

const std::list< cParamVolChantierPhotogram > & cParamChantierPhotogram::ParamVolChantierPhotogram()const 
{
   return mParamVolChantierPhotogram;
}

void  BinaryUnDumpFromFile(cParamChantierPhotogram & anObj,ELISE_fp & aFp)
{
   { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cParamVolChantierPhotogram aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.ParamVolChantierPhotogram().push_back(aVal);
        }
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cParamChantierPhotogram & anObj)
{
    BinaryDumpInFile(aFp,(int)anObj.ParamVolChantierPhotogram().size());
    for(  std::list< cParamVolChantierPhotogram >::const_iterator iT=anObj.ParamVolChantierPhotogram().begin();
         iT!=anObj.ParamVolChantierPhotogram().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
}

cElXMLTree * ToXMLTree(const cParamChantierPhotogram & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ParamChantierPhotogram",eXMLBranche);
  for
  (       std::list< cParamVolChantierPhotogram >::const_iterator it=anObj.ParamVolChantierPhotogram().begin();
      it !=anObj.ParamVolChantierPhotogram().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("ParamVolChantierPhotogram"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cParamChantierPhotogram & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.ParamVolChantierPhotogram(),aTree->GetAll("ParamVolChantierPhotogram",false,1));
}

std::string  Mangling( cParamChantierPhotogram *) {return "E207D588A3D944A2FF3F";};


std::string & cPDV::Im()
{
   return mIm;
}

const std::string & cPDV::Im()const 
{
   return mIm;
}


std::string & cPDV::Orient()
{
   return mOrient;
}

const std::string & cPDV::Orient()const 
{
   return mOrient;
}


std::string & cPDV::IdInBande()
{
   return mIdInBande;
}

const std::string & cPDV::IdInBande()const 
{
   return mIdInBande;
}


std::string & cPDV::Bande()
{
   return mBande;
}

const std::string & cPDV::Bande()const 
{
   return mBande;
}

void  BinaryUnDumpFromFile(cPDV & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Im(),aFp);
    BinaryUnDumpFromFile(anObj.Orient(),aFp);
    BinaryUnDumpFromFile(anObj.IdInBande(),aFp);
    BinaryUnDumpFromFile(anObj.Bande(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cPDV & anObj)
{
    BinaryDumpInFile(aFp,anObj.Im());
    BinaryDumpInFile(aFp,anObj.Orient());
    BinaryDumpInFile(aFp,anObj.IdInBande());
    BinaryDumpInFile(aFp,anObj.Bande());
}

cElXMLTree * ToXMLTree(const cPDV & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"PDV",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Im"),anObj.Im())->ReTagThis("Im"));
   aRes->AddFils(::ToXMLTree(std::string("Orient"),anObj.Orient())->ReTagThis("Orient"));
   aRes->AddFils(::ToXMLTree(std::string("IdInBande"),anObj.IdInBande())->ReTagThis("IdInBande"));
   aRes->AddFils(::ToXMLTree(std::string("Bande"),anObj.Bande())->ReTagThis("Bande"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cPDV & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Im(),aTree->Get("Im",1)); //tototo 

   xml_init(anObj.Orient(),aTree->Get("Orient",1)); //tototo 

   xml_init(anObj.IdInBande(),aTree->Get("IdInBande",1)); //tototo 

   xml_init(anObj.Bande(),aTree->Get("Bande",1)); //tototo 
}

std::string  Mangling( cPDV *) {return "B49AD249889AB7DCFD3F";};


std::string & cBandesChantierPhotogram::IdBande()
{
   return mIdBande;
}

const std::string & cBandesChantierPhotogram::IdBande()const 
{
   return mIdBande;
}


std::list< cPDV > & cBandesChantierPhotogram::PDVs()
{
   return mPDVs;
}

const std::list< cPDV > & cBandesChantierPhotogram::PDVs()const 
{
   return mPDVs;
}

void  BinaryUnDumpFromFile(cBandesChantierPhotogram & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.IdBande(),aFp);
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cPDV aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.PDVs().push_back(aVal);
        }
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cBandesChantierPhotogram & anObj)
{
    BinaryDumpInFile(aFp,anObj.IdBande());
    BinaryDumpInFile(aFp,(int)anObj.PDVs().size());
    for(  std::list< cPDV >::const_iterator iT=anObj.PDVs().begin();
         iT!=anObj.PDVs().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
}

cElXMLTree * ToXMLTree(const cBandesChantierPhotogram & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"BandesChantierPhotogram",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("IdBande"),anObj.IdBande())->ReTagThis("IdBande"));
  for
  (       std::list< cPDV >::const_iterator it=anObj.PDVs().begin();
      it !=anObj.PDVs().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("PDVs"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cBandesChantierPhotogram & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.IdBande(),aTree->Get("IdBande",1)); //tototo 

   xml_init(anObj.PDVs(),aTree->GetAll("PDVs",false,1));
}

std::string  Mangling( cBandesChantierPhotogram *) {return "209A6C9EE647C191FABF";};


std::list< cBandesChantierPhotogram > & cVolChantierPhotogram::BandesChantierPhotogram()
{
   return mBandesChantierPhotogram;
}

const std::list< cBandesChantierPhotogram > & cVolChantierPhotogram::BandesChantierPhotogram()const 
{
   return mBandesChantierPhotogram;
}

void  BinaryUnDumpFromFile(cVolChantierPhotogram & anObj,ELISE_fp & aFp)
{
   { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cBandesChantierPhotogram aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.BandesChantierPhotogram().push_back(aVal);
        }
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cVolChantierPhotogram & anObj)
{
    BinaryDumpInFile(aFp,(int)anObj.BandesChantierPhotogram().size());
    for(  std::list< cBandesChantierPhotogram >::const_iterator iT=anObj.BandesChantierPhotogram().begin();
         iT!=anObj.BandesChantierPhotogram().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
}

cElXMLTree * ToXMLTree(const cVolChantierPhotogram & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"VolChantierPhotogram",eXMLBranche);
  for
  (       std::list< cBandesChantierPhotogram >::const_iterator it=anObj.BandesChantierPhotogram().begin();
      it !=anObj.BandesChantierPhotogram().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("BandesChantierPhotogram"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cVolChantierPhotogram & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.BandesChantierPhotogram(),aTree->GetAll("BandesChantierPhotogram",false,1));
}

std::string  Mangling( cVolChantierPhotogram *) {return "18690A7D391663B1FF3F";};


std::list< cVolChantierPhotogram > & cChantierPhotogram::VolChantierPhotogram()
{
   return mVolChantierPhotogram;
}

const std::list< cVolChantierPhotogram > & cChantierPhotogram::VolChantierPhotogram()const 
{
   return mVolChantierPhotogram;
}

void  BinaryUnDumpFromFile(cChantierPhotogram & anObj,ELISE_fp & aFp)
{
   { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cVolChantierPhotogram aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.VolChantierPhotogram().push_back(aVal);
        }
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cChantierPhotogram & anObj)
{
    BinaryDumpInFile(aFp,(int)anObj.VolChantierPhotogram().size());
    for(  std::list< cVolChantierPhotogram >::const_iterator iT=anObj.VolChantierPhotogram().begin();
         iT!=anObj.VolChantierPhotogram().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
}

cElXMLTree * ToXMLTree(const cChantierPhotogram & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ChantierPhotogram",eXMLBranche);
  for
  (       std::list< cVolChantierPhotogram >::const_iterator it=anObj.VolChantierPhotogram().begin();
      it !=anObj.VolChantierPhotogram().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("VolChantierPhotogram"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cChantierPhotogram & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.VolChantierPhotogram(),aTree->GetAll("VolChantierPhotogram",false,1));
}

std::string  Mangling( cChantierPhotogram *) {return "0A0BF8772D8C999FFDBF";};


int & cCplePDV::Id1()
{
   return mId1;
}

const int & cCplePDV::Id1()const 
{
   return mId1;
}


int & cCplePDV::Id2()
{
   return mId2;
}

const int & cCplePDV::Id2()const 
{
   return mId2;
}

void  BinaryUnDumpFromFile(cCplePDV & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Id1(),aFp);
    BinaryUnDumpFromFile(anObj.Id2(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cCplePDV & anObj)
{
    BinaryDumpInFile(aFp,anObj.Id1());
    BinaryDumpInFile(aFp,anObj.Id2());
}

cElXMLTree * ToXMLTree(const cCplePDV & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"CplePDV",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Id1"),anObj.Id1())->ReTagThis("Id1"));
   aRes->AddFils(::ToXMLTree(std::string("Id2"),anObj.Id2())->ReTagThis("Id2"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cCplePDV & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Id1(),aTree->Get("Id1",1)); //tototo 

   xml_init(anObj.Id2(),aTree->Get("Id2",1)); //tototo 
}

std::string  Mangling( cCplePDV *) {return "1006E33477C681E7F9BF";};


Box2dr & cGraphePdv::BoxCh()
{
   return mBoxCh;
}

const Box2dr & cGraphePdv::BoxCh()const 
{
   return mBoxCh;
}


std::vector< cPDV > & cGraphePdv::PDVs()
{
   return mPDVs;
}

const std::vector< cPDV > & cGraphePdv::PDVs()const 
{
   return mPDVs;
}


std::list< cCplePDV > & cGraphePdv::CplePDV()
{
   return mCplePDV;
}

const std::list< cCplePDV > & cGraphePdv::CplePDV()const 
{
   return mCplePDV;
}

void  BinaryUnDumpFromFile(cGraphePdv & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.BoxCh(),aFp);
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cPDV aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.PDVs().push_back(aVal);
        }
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cCplePDV aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.CplePDV().push_back(aVal);
        }
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cGraphePdv & anObj)
{
    BinaryDumpInFile(aFp,anObj.BoxCh());
    BinaryDumpInFile(aFp,(int)anObj.PDVs().size());
    for(  std::vector< cPDV >::const_iterator iT=anObj.PDVs().begin();
         iT!=anObj.PDVs().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,(int)anObj.CplePDV().size());
    for(  std::list< cCplePDV >::const_iterator iT=anObj.CplePDV().begin();
         iT!=anObj.CplePDV().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
}

cElXMLTree * ToXMLTree(const cGraphePdv & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"GraphePdv",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("BoxCh"),anObj.BoxCh())->ReTagThis("BoxCh"));
  for
  (       std::vector< cPDV >::const_iterator it=anObj.PDVs().begin();
      it !=anObj.PDVs().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("PDVs"));
  for
  (       std::list< cCplePDV >::const_iterator it=anObj.CplePDV().begin();
      it !=anObj.CplePDV().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("CplePDV"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cGraphePdv & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.BoxCh(),aTree->Get("BoxCh",1)); //tototo 

   xml_init(anObj.PDVs(),aTree->GetAll("PDVs",false,1));

   xml_init(anObj.CplePDV(),aTree->GetAll("CplePDV",false,1));
}

std::string  Mangling( cGraphePdv *) {return "CC046832CCAD54B6FE3F";};


double & cCercleRelief::Rayon()
{
   return mRayon;
}

const double & cCercleRelief::Rayon()const 
{
   return mRayon;
}


double & cCercleRelief::Profondeur()
{
   return mProfondeur;
}

const double & cCercleRelief::Profondeur()const 
{
   return mProfondeur;
}

void  BinaryUnDumpFromFile(cCercleRelief & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Rayon(),aFp);
    BinaryUnDumpFromFile(anObj.Profondeur(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cCercleRelief & anObj)
{
    BinaryDumpInFile(aFp,anObj.Rayon());
    BinaryDumpInFile(aFp,anObj.Profondeur());
}

cElXMLTree * ToXMLTree(const cCercleRelief & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"CercleRelief",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Rayon"),anObj.Rayon())->ReTagThis("Rayon"));
   aRes->AddFils(::ToXMLTree(std::string("Profondeur"),anObj.Profondeur())->ReTagThis("Profondeur"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cCercleRelief & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Rayon(),aTree->Get("Rayon",1)); //tototo 

   xml_init(anObj.Profondeur(),aTree->Get("Profondeur",1)); //tototo 
}

std::string  Mangling( cCercleRelief *) {return "D083307149A56D90FA3F";};


int & cCibleCalib::Id()
{
   return mId;
}

const int & cCibleCalib::Id()const 
{
   return mId;
}


cTplValGesInit< bool > & cCibleCalib::Negatif()
{
   return mNegatif;
}

const cTplValGesInit< bool > & cCibleCalib::Negatif()const 
{
   return mNegatif;
}


Pt3dr & cCibleCalib::Position()
{
   return mPosition;
}

const Pt3dr & cCibleCalib::Position()const 
{
   return mPosition;
}


Pt3dr & cCibleCalib::Normale()
{
   return mNormale;
}

const Pt3dr & cCibleCalib::Normale()const 
{
   return mNormale;
}


std::vector< double > & cCibleCalib::Rayons()
{
   return mRayons;
}

const std::vector< double > & cCibleCalib::Rayons()const 
{
   return mRayons;
}


bool & cCibleCalib::Ponctuel()
{
   return mPonctuel;
}

const bool & cCibleCalib::Ponctuel()const 
{
   return mPonctuel;
}


bool & cCibleCalib::ReliefIsSortant()
{
   return mReliefIsSortant;
}

const bool & cCibleCalib::ReliefIsSortant()const 
{
   return mReliefIsSortant;
}


std::vector< cCercleRelief > & cCibleCalib::CercleRelief()
{
   return mCercleRelief;
}

const std::vector< cCercleRelief > & cCibleCalib::CercleRelief()const 
{
   return mCercleRelief;
}


std::string & cCibleCalib::NomType()
{
   return mNomType;
}

const std::string & cCibleCalib::NomType()const 
{
   return mNomType;
}


int & cCibleCalib::Qualite()
{
   return mQualite;
}

const int & cCibleCalib::Qualite()const 
{
   return mQualite;
}


cTplValGesInit< double > & cCibleCalib::FacteurElargRechCorrel()
{
   return mFacteurElargRechCorrel;
}

const cTplValGesInit< double > & cCibleCalib::FacteurElargRechCorrel()const 
{
   return mFacteurElargRechCorrel;
}


cTplValGesInit< double > & cCibleCalib::FacteurElargRechRaffine()
{
   return mFacteurElargRechRaffine;
}

const cTplValGesInit< double > & cCibleCalib::FacteurElargRechRaffine()const 
{
   return mFacteurElargRechRaffine;
}

void  BinaryUnDumpFromFile(cCibleCalib & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Id(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Negatif().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Negatif().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Negatif().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.Position(),aFp);
    BinaryUnDumpFromFile(anObj.Normale(),aFp);
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             double aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.Rayons().push_back(aVal);
        }
  } ;
    BinaryUnDumpFromFile(anObj.Ponctuel(),aFp);
    BinaryUnDumpFromFile(anObj.ReliefIsSortant(),aFp);
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cCercleRelief aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.CercleRelief().push_back(aVal);
        }
  } ;
    BinaryUnDumpFromFile(anObj.NomType(),aFp);
    BinaryUnDumpFromFile(anObj.Qualite(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.FacteurElargRechCorrel().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.FacteurElargRechCorrel().ValForcedForUnUmp(),aFp);
        }
        else  anObj.FacteurElargRechCorrel().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.FacteurElargRechRaffine().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.FacteurElargRechRaffine().ValForcedForUnUmp(),aFp);
        }
        else  anObj.FacteurElargRechRaffine().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cCibleCalib & anObj)
{
    BinaryDumpInFile(aFp,anObj.Id());
    BinaryDumpInFile(aFp,anObj.Negatif().IsInit());
    if (anObj.Negatif().IsInit()) BinaryDumpInFile(aFp,anObj.Negatif().Val());
    BinaryDumpInFile(aFp,anObj.Position());
    BinaryDumpInFile(aFp,anObj.Normale());
    BinaryDumpInFile(aFp,(int)anObj.Rayons().size());
    for(  std::vector< double >::const_iterator iT=anObj.Rayons().begin();
         iT!=anObj.Rayons().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.Ponctuel());
    BinaryDumpInFile(aFp,anObj.ReliefIsSortant());
    BinaryDumpInFile(aFp,(int)anObj.CercleRelief().size());
    for(  std::vector< cCercleRelief >::const_iterator iT=anObj.CercleRelief().begin();
         iT!=anObj.CercleRelief().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.NomType());
    BinaryDumpInFile(aFp,anObj.Qualite());
    BinaryDumpInFile(aFp,anObj.FacteurElargRechCorrel().IsInit());
    if (anObj.FacteurElargRechCorrel().IsInit()) BinaryDumpInFile(aFp,anObj.FacteurElargRechCorrel().Val());
    BinaryDumpInFile(aFp,anObj.FacteurElargRechRaffine().IsInit());
    if (anObj.FacteurElargRechRaffine().IsInit()) BinaryDumpInFile(aFp,anObj.FacteurElargRechRaffine().Val());
}

cElXMLTree * ToXMLTree(const cCibleCalib & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"CibleCalib",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Id"),anObj.Id())->ReTagThis("Id"));
   if (anObj.Negatif().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Negatif"),anObj.Negatif().Val())->ReTagThis("Negatif"));
   aRes->AddFils(ToXMLTree(std::string("Position"),anObj.Position())->ReTagThis("Position"));
   aRes->AddFils(ToXMLTree(std::string("Normale"),anObj.Normale())->ReTagThis("Normale"));
  for
  (       std::vector< double >::const_iterator it=anObj.Rayons().begin();
      it !=anObj.Rayons().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("Rayons"),(*it))->ReTagThis("Rayons"));
   aRes->AddFils(::ToXMLTree(std::string("Ponctuel"),anObj.Ponctuel())->ReTagThis("Ponctuel"));
   aRes->AddFils(::ToXMLTree(std::string("ReliefIsSortant"),anObj.ReliefIsSortant())->ReTagThis("ReliefIsSortant"));
  for
  (       std::vector< cCercleRelief >::const_iterator it=anObj.CercleRelief().begin();
      it !=anObj.CercleRelief().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("CercleRelief"));
   aRes->AddFils(::ToXMLTree(std::string("NomType"),anObj.NomType())->ReTagThis("NomType"));
   aRes->AddFils(::ToXMLTree(std::string("Qualite"),anObj.Qualite())->ReTagThis("Qualite"));
   if (anObj.FacteurElargRechCorrel().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("FacteurElargRechCorrel"),anObj.FacteurElargRechCorrel().Val())->ReTagThis("FacteurElargRechCorrel"));
   if (anObj.FacteurElargRechRaffine().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("FacteurElargRechRaffine"),anObj.FacteurElargRechRaffine().Val())->ReTagThis("FacteurElargRechRaffine"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cCibleCalib & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Id(),aTree->Get("Id",1)); //tototo 

   xml_init(anObj.Negatif(),aTree->Get("Negatif",1),bool(false)); //tototo 

   xml_init(anObj.Position(),aTree->Get("Position",1)); //tototo 

   xml_init(anObj.Normale(),aTree->Get("Normale",1)); //tototo 

   xml_init(anObj.Rayons(),aTree->GetAll("Rayons",false,1));

   xml_init(anObj.Ponctuel(),aTree->Get("Ponctuel",1)); //tototo 

   xml_init(anObj.ReliefIsSortant(),aTree->Get("ReliefIsSortant",1)); //tototo 

   xml_init(anObj.CercleRelief(),aTree->GetAll("CercleRelief",false,1));

   xml_init(anObj.NomType(),aTree->Get("NomType",1)); //tototo 

   xml_init(anObj.Qualite(),aTree->Get("Qualite",1)); //tototo 

   xml_init(anObj.FacteurElargRechCorrel(),aTree->Get("FacteurElargRechCorrel",1)); //tototo 

   xml_init(anObj.FacteurElargRechRaffine(),aTree->Get("FacteurElargRechRaffine",1)); //tototo 
}

std::string  Mangling( cCibleCalib *) {return "9F9FCA1365A369B4FD3F";};


std::string & cPolygoneCalib::Name()
{
   return mName;
}

const std::string & cPolygoneCalib::Name()const 
{
   return mName;
}


std::vector< cCibleCalib > & cPolygoneCalib::Cibles()
{
   return mCibles;
}

const std::vector< cCibleCalib > & cPolygoneCalib::Cibles()const 
{
   return mCibles;
}

void  BinaryUnDumpFromFile(cPolygoneCalib & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Name(),aFp);
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cCibleCalib aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.Cibles().push_back(aVal);
        }
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cPolygoneCalib & anObj)
{
    BinaryDumpInFile(aFp,anObj.Name());
    BinaryDumpInFile(aFp,(int)anObj.Cibles().size());
    for(  std::vector< cCibleCalib >::const_iterator iT=anObj.Cibles().begin();
         iT!=anObj.Cibles().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
}

cElXMLTree * ToXMLTree(const cPolygoneCalib & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"PolygoneCalib",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Name"),anObj.Name())->ReTagThis("Name"));
  for
  (       std::vector< cCibleCalib >::const_iterator it=anObj.Cibles().begin();
      it !=anObj.Cibles().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("Cibles"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cPolygoneCalib & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Name(),aTree->Get("Name",1)); //tototo 

   xml_init(anObj.Cibles(),aTree->GetAll("Cibles",false,1));
}

std::string  Mangling( cPolygoneCalib *) {return "986C9723E42ABCBFFCBF";};


std::string & cPointesCibleAC::NameIm()
{
   return mNameIm;
}

const std::string & cPointesCibleAC::NameIm()const 
{
   return mNameIm;
}


Pt2dr & cPointesCibleAC::PtIm()
{
   return mPtIm;
}

const Pt2dr & cPointesCibleAC::PtIm()const 
{
   return mPtIm;
}

void  BinaryUnDumpFromFile(cPointesCibleAC & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.NameIm(),aFp);
    BinaryUnDumpFromFile(anObj.PtIm(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cPointesCibleAC & anObj)
{
    BinaryDumpInFile(aFp,anObj.NameIm());
    BinaryDumpInFile(aFp,anObj.PtIm());
}

cElXMLTree * ToXMLTree(const cPointesCibleAC & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"PointesCibleAC",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("NameIm"),anObj.NameIm())->ReTagThis("NameIm"));
   aRes->AddFils(::ToXMLTree(std::string("PtIm"),anObj.PtIm())->ReTagThis("PtIm"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cPointesCibleAC & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.NameIm(),aTree->Get("NameIm",1)); //tototo 

   xml_init(anObj.PtIm(),aTree->Get("PtIm",1)); //tototo 
}

std::string  Mangling( cPointesCibleAC *) {return "4EE9EC9C8875DEF8FDBF";};


std::string & cCibleACalcByLiaisons::Name()
{
   return mName;
}

const std::string & cCibleACalcByLiaisons::Name()const 
{
   return mName;
}


std::list< cPointesCibleAC > & cCibleACalcByLiaisons::PointesCibleAC()
{
   return mPointesCibleAC;
}

const std::list< cPointesCibleAC > & cCibleACalcByLiaisons::PointesCibleAC()const 
{
   return mPointesCibleAC;
}

void  BinaryUnDumpFromFile(cCibleACalcByLiaisons & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Name(),aFp);
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cPointesCibleAC aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.PointesCibleAC().push_back(aVal);
        }
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cCibleACalcByLiaisons & anObj)
{
    BinaryDumpInFile(aFp,anObj.Name());
    BinaryDumpInFile(aFp,(int)anObj.PointesCibleAC().size());
    for(  std::list< cPointesCibleAC >::const_iterator iT=anObj.PointesCibleAC().begin();
         iT!=anObj.PointesCibleAC().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
}

cElXMLTree * ToXMLTree(const cCibleACalcByLiaisons & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"CibleACalcByLiaisons",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Name"),anObj.Name())->ReTagThis("Name"));
  for
  (       std::list< cPointesCibleAC >::const_iterator it=anObj.PointesCibleAC().begin();
      it !=anObj.PointesCibleAC().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("PointesCibleAC"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cCibleACalcByLiaisons & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Name(),aTree->Get("Name",1)); //tototo 

   xml_init(anObj.PointesCibleAC(),aTree->GetAll("PointesCibleAC",false,1));
}

std::string  Mangling( cCibleACalcByLiaisons *) {return "D004243AC6B16DA4FF3F";};


cTplValGesInit< bool > & cCible2Rech::UseIt()
{
   return mUseIt;
}

const cTplValGesInit< bool > & cCible2Rech::UseIt()const 
{
   return mUseIt;
}


std::vector< int > & cCible2Rech::Id()
{
   return mId;
}

const std::vector< int > & cCible2Rech::Id()const 
{
   return mId;
}

void  BinaryUnDumpFromFile(cCible2Rech & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.UseIt().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.UseIt().ValForcedForUnUmp(),aFp);
        }
        else  anObj.UseIt().SetNoInit();
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             int aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.Id().push_back(aVal);
        }
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cCible2Rech & anObj)
{
    BinaryDumpInFile(aFp,anObj.UseIt().IsInit());
    if (anObj.UseIt().IsInit()) BinaryDumpInFile(aFp,anObj.UseIt().Val());
    BinaryDumpInFile(aFp,(int)anObj.Id().size());
    for(  std::vector< int >::const_iterator iT=anObj.Id().begin();
         iT!=anObj.Id().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
}

cElXMLTree * ToXMLTree(const cCible2Rech & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"Cible2Rech",eXMLBranche);
   if (anObj.UseIt().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("UseIt"),anObj.UseIt().Val())->ReTagThis("UseIt"));
  for
  (       std::vector< int >::const_iterator it=anObj.Id().begin();
      it !=anObj.Id().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("Id"),(*it))->ReTagThis("Id"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cCible2Rech & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.UseIt(),aTree->Get("UseIt",1),bool(false)); //tototo 

   xml_init(anObj.Id(),aTree->GetAll("Id",false,1));
}

std::string  Mangling( cCible2Rech *) {return "C3018B02476EB0E0FC3F";};


cTplValGesInit< bool > & cIm2Select::UseIt()
{
   return mUseIt;
}

const cTplValGesInit< bool > & cIm2Select::UseIt()const 
{
   return mUseIt;
}


std::vector< std::string > & cIm2Select::Id()
{
   return mId;
}

const std::vector< std::string > & cIm2Select::Id()const 
{
   return mId;
}

void  BinaryUnDumpFromFile(cIm2Select & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.UseIt().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.UseIt().ValForcedForUnUmp(),aFp);
        }
        else  anObj.UseIt().SetNoInit();
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             std::string aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.Id().push_back(aVal);
        }
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cIm2Select & anObj)
{
    BinaryDumpInFile(aFp,anObj.UseIt().IsInit());
    if (anObj.UseIt().IsInit()) BinaryDumpInFile(aFp,anObj.UseIt().Val());
    BinaryDumpInFile(aFp,(int)anObj.Id().size());
    for(  std::vector< std::string >::const_iterator iT=anObj.Id().begin();
         iT!=anObj.Id().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
}

cElXMLTree * ToXMLTree(const cIm2Select & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"Im2Select",eXMLBranche);
   if (anObj.UseIt().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("UseIt"),anObj.UseIt().Val())->ReTagThis("UseIt"));
  for
  (       std::vector< std::string >::const_iterator it=anObj.Id().begin();
      it !=anObj.Id().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("Id"),(*it))->ReTagThis("Id"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cIm2Select & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.UseIt(),aTree->Get("UseIt",1),bool(false)); //tototo 

   xml_init(anObj.Id(),aTree->GetAll("Id",false,1));
}

std::string  Mangling( cIm2Select *) {return "AC64F3C184E67595FC3F";};


std::list< cElRegex_Ptr > & cImageUseDirectPointeManuel::Id()
{
   return mId;
}

const std::list< cElRegex_Ptr > & cImageUseDirectPointeManuel::Id()const 
{
   return mId;
}

void  BinaryUnDumpFromFile(cImageUseDirectPointeManuel & anObj,ELISE_fp & aFp)
{
   { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cElRegex_Ptr aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.Id().push_back(aVal);
        }
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cImageUseDirectPointeManuel & anObj)
{
    BinaryDumpInFile(aFp,(int)anObj.Id().size());
    for(  std::list< cElRegex_Ptr >::const_iterator iT=anObj.Id().begin();
         iT!=anObj.Id().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
}

cElXMLTree * ToXMLTree(const cImageUseDirectPointeManuel & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ImageUseDirectPointeManuel",eXMLBranche);
  for
  (       std::list< cElRegex_Ptr >::const_iterator it=anObj.Id().begin();
      it !=anObj.Id().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("Id"),(*it))->ReTagThis("Id"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cImageUseDirectPointeManuel & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Id(),aTree->GetAll("Id",false,1));
}

std::string  Mangling( cImageUseDirectPointeManuel *) {return "84C7928A70C380A0FF3F";};


std::string & cExportAppuisAsDico::NameDico()
{
   return mNameDico;
}

const std::string & cExportAppuisAsDico::NameDico()const 
{
   return mNameDico;
}


Pt3dr & cExportAppuisAsDico::Incertitude()
{
   return mIncertitude;
}

const Pt3dr & cExportAppuisAsDico::Incertitude()const 
{
   return mIncertitude;
}

void  BinaryUnDumpFromFile(cExportAppuisAsDico & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.NameDico(),aFp);
    BinaryUnDumpFromFile(anObj.Incertitude(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cExportAppuisAsDico & anObj)
{
    BinaryDumpInFile(aFp,anObj.NameDico());
    BinaryDumpInFile(aFp,anObj.Incertitude());
}

cElXMLTree * ToXMLTree(const cExportAppuisAsDico & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ExportAppuisAsDico",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("NameDico"),anObj.NameDico())->ReTagThis("NameDico"));
   aRes->AddFils(ToXMLTree(std::string("Incertitude"),anObj.Incertitude())->ReTagThis("Incertitude"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cExportAppuisAsDico & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.NameDico(),aTree->Get("NameDico",1)); //tototo 

   xml_init(anObj.Incertitude(),aTree->Get("Incertitude",1)); //tototo 
}

std::string  Mangling( cExportAppuisAsDico *) {return "F1BD9EAF3944BCEAFE3F";};


std::list< cCibleACalcByLiaisons > & cComplParamEtalPoly::CibleACalcByLiaisons()
{
   return mCibleACalcByLiaisons;
}

const std::list< cCibleACalcByLiaisons > & cComplParamEtalPoly::CibleACalcByLiaisons()const 
{
   return mCibleACalcByLiaisons;
}


cTplValGesInit< cCible2Rech > & cComplParamEtalPoly::Cible2Rech()
{
   return mCible2Rech;
}

const cTplValGesInit< cCible2Rech > & cComplParamEtalPoly::Cible2Rech()const 
{
   return mCible2Rech;
}


cTplValGesInit< cIm2Select > & cComplParamEtalPoly::Im2Select()
{
   return mIm2Select;
}

const cTplValGesInit< cIm2Select > & cComplParamEtalPoly::Im2Select()const 
{
   return mIm2Select;
}


cTplValGesInit< cImageUseDirectPointeManuel > & cComplParamEtalPoly::ImageUseDirectPointeManuel()
{
   return mImageUseDirectPointeManuel;
}

const cTplValGesInit< cImageUseDirectPointeManuel > & cComplParamEtalPoly::ImageUseDirectPointeManuel()const 
{
   return mImageUseDirectPointeManuel;
}


std::string & cComplParamEtalPoly::NameDico()
{
   return ExportAppuisAsDico().Val().NameDico();
}

const std::string & cComplParamEtalPoly::NameDico()const 
{
   return ExportAppuisAsDico().Val().NameDico();
}


Pt3dr & cComplParamEtalPoly::Incertitude()
{
   return ExportAppuisAsDico().Val().Incertitude();
}

const Pt3dr & cComplParamEtalPoly::Incertitude()const 
{
   return ExportAppuisAsDico().Val().Incertitude();
}


cTplValGesInit< cExportAppuisAsDico > & cComplParamEtalPoly::ExportAppuisAsDico()
{
   return mExportAppuisAsDico;
}

const cTplValGesInit< cExportAppuisAsDico > & cComplParamEtalPoly::ExportAppuisAsDico()const 
{
   return mExportAppuisAsDico;
}

void  BinaryUnDumpFromFile(cComplParamEtalPoly & anObj,ELISE_fp & aFp)
{
   { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cCibleACalcByLiaisons aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.CibleACalcByLiaisons().push_back(aVal);
        }
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Cible2Rech().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Cible2Rech().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Cible2Rech().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Im2Select().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Im2Select().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Im2Select().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ImageUseDirectPointeManuel().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ImageUseDirectPointeManuel().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ImageUseDirectPointeManuel().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ExportAppuisAsDico().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ExportAppuisAsDico().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ExportAppuisAsDico().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cComplParamEtalPoly & anObj)
{
    BinaryDumpInFile(aFp,(int)anObj.CibleACalcByLiaisons().size());
    for(  std::list< cCibleACalcByLiaisons >::const_iterator iT=anObj.CibleACalcByLiaisons().begin();
         iT!=anObj.CibleACalcByLiaisons().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.Cible2Rech().IsInit());
    if (anObj.Cible2Rech().IsInit()) BinaryDumpInFile(aFp,anObj.Cible2Rech().Val());
    BinaryDumpInFile(aFp,anObj.Im2Select().IsInit());
    if (anObj.Im2Select().IsInit()) BinaryDumpInFile(aFp,anObj.Im2Select().Val());
    BinaryDumpInFile(aFp,anObj.ImageUseDirectPointeManuel().IsInit());
    if (anObj.ImageUseDirectPointeManuel().IsInit()) BinaryDumpInFile(aFp,anObj.ImageUseDirectPointeManuel().Val());
    BinaryDumpInFile(aFp,anObj.ExportAppuisAsDico().IsInit());
    if (anObj.ExportAppuisAsDico().IsInit()) BinaryDumpInFile(aFp,anObj.ExportAppuisAsDico().Val());
}

cElXMLTree * ToXMLTree(const cComplParamEtalPoly & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ComplParamEtalPoly",eXMLBranche);
  for
  (       std::list< cCibleACalcByLiaisons >::const_iterator it=anObj.CibleACalcByLiaisons().begin();
      it !=anObj.CibleACalcByLiaisons().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("CibleACalcByLiaisons"));
   if (anObj.Cible2Rech().IsInit())
      aRes->AddFils(ToXMLTree(anObj.Cible2Rech().Val())->ReTagThis("Cible2Rech"));
   if (anObj.Im2Select().IsInit())
      aRes->AddFils(ToXMLTree(anObj.Im2Select().Val())->ReTagThis("Im2Select"));
   if (anObj.ImageUseDirectPointeManuel().IsInit())
      aRes->AddFils(ToXMLTree(anObj.ImageUseDirectPointeManuel().Val())->ReTagThis("ImageUseDirectPointeManuel"));
   if (anObj.ExportAppuisAsDico().IsInit())
      aRes->AddFils(ToXMLTree(anObj.ExportAppuisAsDico().Val())->ReTagThis("ExportAppuisAsDico"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cComplParamEtalPoly & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.CibleACalcByLiaisons(),aTree->GetAll("CibleACalcByLiaisons",false,1));

   xml_init(anObj.Cible2Rech(),aTree->Get("Cible2Rech",1)); //tototo 

   xml_init(anObj.Im2Select(),aTree->Get("Im2Select",1)); //tototo 

   xml_init(anObj.ImageUseDirectPointeManuel(),aTree->Get("ImageUseDirectPointeManuel",1)); //tototo 

   xml_init(anObj.ExportAppuisAsDico(),aTree->Get("ExportAppuisAsDico",1)); //tototo 
}

std::string  Mangling( cComplParamEtalPoly *) {return "615CC01B8368D7DEFD3F";};


Pt3dr & cOneAppuisDAF::Pt()
{
   return mPt;
}

const Pt3dr & cOneAppuisDAF::Pt()const 
{
   return mPt;
}


std::string & cOneAppuisDAF::NamePt()
{
   return mNamePt;
}

const std::string & cOneAppuisDAF::NamePt()const 
{
   return mNamePt;
}


Pt3dr & cOneAppuisDAF::Incertitude()
{
   return mIncertitude;
}

const Pt3dr & cOneAppuisDAF::Incertitude()const 
{
   return mIncertitude;
}

void  BinaryUnDumpFromFile(cOneAppuisDAF & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Pt(),aFp);
    BinaryUnDumpFromFile(anObj.NamePt(),aFp);
    BinaryUnDumpFromFile(anObj.Incertitude(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cOneAppuisDAF & anObj)
{
    BinaryDumpInFile(aFp,anObj.Pt());
    BinaryDumpInFile(aFp,anObj.NamePt());
    BinaryDumpInFile(aFp,anObj.Incertitude());
}

cElXMLTree * ToXMLTree(const cOneAppuisDAF & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"OneAppuisDAF",eXMLBranche);
   aRes->AddFils(ToXMLTree(std::string("Pt"),anObj.Pt())->ReTagThis("Pt"));
   aRes->AddFils(::ToXMLTree(std::string("NamePt"),anObj.NamePt())->ReTagThis("NamePt"));
   aRes->AddFils(ToXMLTree(std::string("Incertitude"),anObj.Incertitude())->ReTagThis("Incertitude"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cOneAppuisDAF & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Pt(),aTree->Get("Pt",1)); //tototo 

   xml_init(anObj.NamePt(),aTree->Get("NamePt",1)); //tototo 

   xml_init(anObj.Incertitude(),aTree->Get("Incertitude",1)); //tototo 
}

std::string  Mangling( cOneAppuisDAF *) {return "06493DEE7D35A8B9FF3F";};


std::list< cOneAppuisDAF > & cDicoAppuisFlottant::OneAppuisDAF()
{
   return mOneAppuisDAF;
}

const std::list< cOneAppuisDAF > & cDicoAppuisFlottant::OneAppuisDAF()const 
{
   return mOneAppuisDAF;
}

void  BinaryUnDumpFromFile(cDicoAppuisFlottant & anObj,ELISE_fp & aFp)
{
   { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cOneAppuisDAF aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.OneAppuisDAF().push_back(aVal);
        }
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cDicoAppuisFlottant & anObj)
{
    BinaryDumpInFile(aFp,(int)anObj.OneAppuisDAF().size());
    for(  std::list< cOneAppuisDAF >::const_iterator iT=anObj.OneAppuisDAF().begin();
         iT!=anObj.OneAppuisDAF().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
}

cElXMLTree * ToXMLTree(const cDicoAppuisFlottant & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"DicoAppuisFlottant",eXMLBranche);
  for
  (       std::list< cOneAppuisDAF >::const_iterator it=anObj.OneAppuisDAF().begin();
      it !=anObj.OneAppuisDAF().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("OneAppuisDAF"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cDicoAppuisFlottant & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.OneAppuisDAF(),aTree->GetAll("OneAppuisDAF",false,1));
}

std::string  Mangling( cDicoAppuisFlottant *) {return "6C6FB1C43C4A1DABFC3F";};


std::string & cOneModifIPF::KeyName()
{
   return mKeyName;
}

const std::string & cOneModifIPF::KeyName()const 
{
   return mKeyName;
}


Pt3dr & cOneModifIPF::Incertitude()
{
   return mIncertitude;
}

const Pt3dr & cOneModifIPF::Incertitude()const 
{
   return mIncertitude;
}


cTplValGesInit< bool > & cOneModifIPF::IsMult()
{
   return mIsMult;
}

const cTplValGesInit< bool > & cOneModifIPF::IsMult()const 
{
   return mIsMult;
}

void  BinaryUnDumpFromFile(cOneModifIPF & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.KeyName(),aFp);
    BinaryUnDumpFromFile(anObj.Incertitude(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.IsMult().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.IsMult().ValForcedForUnUmp(),aFp);
        }
        else  anObj.IsMult().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cOneModifIPF & anObj)
{
    BinaryDumpInFile(aFp,anObj.KeyName());
    BinaryDumpInFile(aFp,anObj.Incertitude());
    BinaryDumpInFile(aFp,anObj.IsMult().IsInit());
    if (anObj.IsMult().IsInit()) BinaryDumpInFile(aFp,anObj.IsMult().Val());
}

cElXMLTree * ToXMLTree(const cOneModifIPF & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"OneModifIPF",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("KeyName"),anObj.KeyName())->ReTagThis("KeyName"));
   aRes->AddFils(ToXMLTree(std::string("Incertitude"),anObj.Incertitude())->ReTagThis("Incertitude"));
   if (anObj.IsMult().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("IsMult"),anObj.IsMult().Val())->ReTagThis("IsMult"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cOneModifIPF & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.KeyName(),aTree->Get("KeyName",1)); //tototo 

   xml_init(anObj.Incertitude(),aTree->Get("Incertitude",1)); //tototo 

   xml_init(anObj.IsMult(),aTree->Get("IsMult",1),bool(false)); //tototo 
}

std::string  Mangling( cOneModifIPF *) {return "99C991645C264193FF3F";};


std::list< cOneModifIPF > & cModifIncPtsFlottant::OneModifIPF()
{
   return mOneModifIPF;
}

const std::list< cOneModifIPF > & cModifIncPtsFlottant::OneModifIPF()const 
{
   return mOneModifIPF;
}

void  BinaryUnDumpFromFile(cModifIncPtsFlottant & anObj,ELISE_fp & aFp)
{
   { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cOneModifIPF aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.OneModifIPF().push_back(aVal);
        }
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cModifIncPtsFlottant & anObj)
{
    BinaryDumpInFile(aFp,(int)anObj.OneModifIPF().size());
    for(  std::list< cOneModifIPF >::const_iterator iT=anObj.OneModifIPF().begin();
         iT!=anObj.OneModifIPF().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
}

cElXMLTree * ToXMLTree(const cModifIncPtsFlottant & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ModifIncPtsFlottant",eXMLBranche);
  for
  (       std::list< cOneModifIPF >::const_iterator it=anObj.OneModifIPF().begin();
      it !=anObj.OneModifIPF().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("OneModifIPF"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cModifIncPtsFlottant & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.OneModifIPF(),aTree->GetAll("OneModifIPF",false,1));
}

std::string  Mangling( cModifIncPtsFlottant *) {return "2690A7453F107480FD3F";};


std::string & cOneMesureAF1I::NamePt()
{
   return mNamePt;
}

const std::string & cOneMesureAF1I::NamePt()const 
{
   return mNamePt;
}


Pt2dr & cOneMesureAF1I::PtIm()
{
   return mPtIm;
}

const Pt2dr & cOneMesureAF1I::PtIm()const 
{
   return mPtIm;
}

void  BinaryUnDumpFromFile(cOneMesureAF1I & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.NamePt(),aFp);
    BinaryUnDumpFromFile(anObj.PtIm(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cOneMesureAF1I & anObj)
{
    BinaryDumpInFile(aFp,anObj.NamePt());
    BinaryDumpInFile(aFp,anObj.PtIm());
}

cElXMLTree * ToXMLTree(const cOneMesureAF1I & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"OneMesureAF1I",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("NamePt"),anObj.NamePt())->ReTagThis("NamePt"));
   aRes->AddFils(::ToXMLTree(std::string("PtIm"),anObj.PtIm())->ReTagThis("PtIm"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cOneMesureAF1I & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.NamePt(),aTree->Get("NamePt",1)); //tototo 

   xml_init(anObj.PtIm(),aTree->Get("PtIm",1)); //tototo 
}

std::string  Mangling( cOneMesureAF1I *) {return "2D0F23327C6B4F9AFE3F";};


std::string & cMesureAppuiFlottant1Im::NameIm()
{
   return mNameIm;
}

const std::string & cMesureAppuiFlottant1Im::NameIm()const 
{
   return mNameIm;
}


std::list< cOneMesureAF1I > & cMesureAppuiFlottant1Im::OneMesureAF1I()
{
   return mOneMesureAF1I;
}

const std::list< cOneMesureAF1I > & cMesureAppuiFlottant1Im::OneMesureAF1I()const 
{
   return mOneMesureAF1I;
}

void  BinaryUnDumpFromFile(cMesureAppuiFlottant1Im & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.NameIm(),aFp);
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cOneMesureAF1I aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.OneMesureAF1I().push_back(aVal);
        }
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cMesureAppuiFlottant1Im & anObj)
{
    BinaryDumpInFile(aFp,anObj.NameIm());
    BinaryDumpInFile(aFp,(int)anObj.OneMesureAF1I().size());
    for(  std::list< cOneMesureAF1I >::const_iterator iT=anObj.OneMesureAF1I().begin();
         iT!=anObj.OneMesureAF1I().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
}

cElXMLTree * ToXMLTree(const cMesureAppuiFlottant1Im & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"MesureAppuiFlottant1Im",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("NameIm"),anObj.NameIm())->ReTagThis("NameIm"));
  for
  (       std::list< cOneMesureAF1I >::const_iterator it=anObj.OneMesureAF1I().begin();
      it !=anObj.OneMesureAF1I().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("OneMesureAF1I"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cMesureAppuiFlottant1Im & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.NameIm(),aTree->Get("NameIm",1)); //tototo 

   xml_init(anObj.OneMesureAF1I(),aTree->GetAll("OneMesureAF1I",false,1));
}

std::string  Mangling( cMesureAppuiFlottant1Im *) {return "16E7BC31BE572E8CFF3F";};


std::list< cMesureAppuiFlottant1Im > & cSetOfMesureAppuisFlottants::MesureAppuiFlottant1Im()
{
   return mMesureAppuiFlottant1Im;
}

const std::list< cMesureAppuiFlottant1Im > & cSetOfMesureAppuisFlottants::MesureAppuiFlottant1Im()const 
{
   return mMesureAppuiFlottant1Im;
}

void  BinaryUnDumpFromFile(cSetOfMesureAppuisFlottants & anObj,ELISE_fp & aFp)
{
   { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cMesureAppuiFlottant1Im aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.MesureAppuiFlottant1Im().push_back(aVal);
        }
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cSetOfMesureAppuisFlottants & anObj)
{
    BinaryDumpInFile(aFp,(int)anObj.MesureAppuiFlottant1Im().size());
    for(  std::list< cMesureAppuiFlottant1Im >::const_iterator iT=anObj.MesureAppuiFlottant1Im().begin();
         iT!=anObj.MesureAppuiFlottant1Im().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
}

cElXMLTree * ToXMLTree(const cSetOfMesureAppuisFlottants & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"SetOfMesureAppuisFlottants",eXMLBranche);
  for
  (       std::list< cMesureAppuiFlottant1Im >::const_iterator it=anObj.MesureAppuiFlottant1Im().begin();
      it !=anObj.MesureAppuiFlottant1Im().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("MesureAppuiFlottant1Im"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cSetOfMesureAppuisFlottants & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.MesureAppuiFlottant1Im(),aTree->GetAll("MesureAppuiFlottant1Im",false,1));
}

std::string  Mangling( cSetOfMesureAppuisFlottants *) {return "96CF92ABE8A55E89FF3F";};


std::list< std::string > & cOneMesureSegDr::NamePt()
{
   return mNamePt;
}

const std::list< std::string > & cOneMesureSegDr::NamePt()const 
{
   return mNamePt;
}


Pt2dr & cOneMesureSegDr::Pt1Im()
{
   return mPt1Im;
}

const Pt2dr & cOneMesureSegDr::Pt1Im()const 
{
   return mPt1Im;
}


Pt2dr & cOneMesureSegDr::Pt2Im()
{
   return mPt2Im;
}

const Pt2dr & cOneMesureSegDr::Pt2Im()const 
{
   return mPt2Im;
}

void  BinaryUnDumpFromFile(cOneMesureSegDr & anObj,ELISE_fp & aFp)
{
   { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             std::string aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.NamePt().push_back(aVal);
        }
  } ;
    BinaryUnDumpFromFile(anObj.Pt1Im(),aFp);
    BinaryUnDumpFromFile(anObj.Pt2Im(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cOneMesureSegDr & anObj)
{
    BinaryDumpInFile(aFp,(int)anObj.NamePt().size());
    for(  std::list< std::string >::const_iterator iT=anObj.NamePt().begin();
         iT!=anObj.NamePt().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.Pt1Im());
    BinaryDumpInFile(aFp,anObj.Pt2Im());
}

cElXMLTree * ToXMLTree(const cOneMesureSegDr & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"OneMesureSegDr",eXMLBranche);
  for
  (       std::list< std::string >::const_iterator it=anObj.NamePt().begin();
      it !=anObj.NamePt().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("NamePt"),(*it))->ReTagThis("NamePt"));
   aRes->AddFils(::ToXMLTree(std::string("Pt1Im"),anObj.Pt1Im())->ReTagThis("Pt1Im"));
   aRes->AddFils(::ToXMLTree(std::string("Pt2Im"),anObj.Pt2Im())->ReTagThis("Pt2Im"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cOneMesureSegDr & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.NamePt(),aTree->GetAll("NamePt",false,1));

   xml_init(anObj.Pt1Im(),aTree->Get("Pt1Im",1)); //tototo 

   xml_init(anObj.Pt2Im(),aTree->Get("Pt2Im",1)); //tototo 
}

std::string  Mangling( cOneMesureSegDr *) {return "0264FFB51727F087FF3F";};


std::string & cMesureAppuiSegDr1Im::NameIm()
{
   return mNameIm;
}

const std::string & cMesureAppuiSegDr1Im::NameIm()const 
{
   return mNameIm;
}


std::list< cOneMesureSegDr > & cMesureAppuiSegDr1Im::OneMesureSegDr()
{
   return mOneMesureSegDr;
}

const std::list< cOneMesureSegDr > & cMesureAppuiSegDr1Im::OneMesureSegDr()const 
{
   return mOneMesureSegDr;
}

void  BinaryUnDumpFromFile(cMesureAppuiSegDr1Im & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.NameIm(),aFp);
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cOneMesureSegDr aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.OneMesureSegDr().push_back(aVal);
        }
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cMesureAppuiSegDr1Im & anObj)
{
    BinaryDumpInFile(aFp,anObj.NameIm());
    BinaryDumpInFile(aFp,(int)anObj.OneMesureSegDr().size());
    for(  std::list< cOneMesureSegDr >::const_iterator iT=anObj.OneMesureSegDr().begin();
         iT!=anObj.OneMesureSegDr().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
}

cElXMLTree * ToXMLTree(const cMesureAppuiSegDr1Im & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"MesureAppuiSegDr1Im",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("NameIm"),anObj.NameIm())->ReTagThis("NameIm"));
  for
  (       std::list< cOneMesureSegDr >::const_iterator it=anObj.OneMesureSegDr().begin();
      it !=anObj.OneMesureSegDr().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("OneMesureSegDr"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cMesureAppuiSegDr1Im & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.NameIm(),aTree->Get("NameIm",1)); //tototo 

   xml_init(anObj.OneMesureSegDr(),aTree->GetAll("OneMesureSegDr",false,1));
}

std::string  Mangling( cMesureAppuiSegDr1Im *) {return "68E529CDB22B3FA3FF3F";};


std::list< cMesureAppuiSegDr1Im > & cSetOfMesureSegDr::MesureAppuiSegDr1Im()
{
   return mMesureAppuiSegDr1Im;
}

const std::list< cMesureAppuiSegDr1Im > & cSetOfMesureSegDr::MesureAppuiSegDr1Im()const 
{
   return mMesureAppuiSegDr1Im;
}

void  BinaryUnDumpFromFile(cSetOfMesureSegDr & anObj,ELISE_fp & aFp)
{
   { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cMesureAppuiSegDr1Im aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.MesureAppuiSegDr1Im().push_back(aVal);
        }
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cSetOfMesureSegDr & anObj)
{
    BinaryDumpInFile(aFp,(int)anObj.MesureAppuiSegDr1Im().size());
    for(  std::list< cMesureAppuiSegDr1Im >::const_iterator iT=anObj.MesureAppuiSegDr1Im().begin();
         iT!=anObj.MesureAppuiSegDr1Im().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
}

cElXMLTree * ToXMLTree(const cSetOfMesureSegDr & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"SetOfMesureSegDr",eXMLBranche);
  for
  (       std::list< cMesureAppuiSegDr1Im >::const_iterator it=anObj.MesureAppuiSegDr1Im().begin();
      it !=anObj.MesureAppuiSegDr1Im().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("MesureAppuiSegDr1Im"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cSetOfMesureSegDr & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.MesureAppuiSegDr1Im(),aTree->GetAll("MesureAppuiSegDr1Im",false,1));
}

std::string  Mangling( cSetOfMesureSegDr *) {return "C88F586AD97841E5FBBF";};


cTplValGesInit< int > & cMesureAppuis::Num()
{
   return mNum;
}

const cTplValGesInit< int > & cMesureAppuis::Num()const 
{
   return mNum;
}


Pt2dr & cMesureAppuis::Im()
{
   return mIm;
}

const Pt2dr & cMesureAppuis::Im()const 
{
   return mIm;
}


Pt3dr & cMesureAppuis::Ter()
{
   return mTer;
}

const Pt3dr & cMesureAppuis::Ter()const 
{
   return mTer;
}

void  BinaryUnDumpFromFile(cMesureAppuis & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Num().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Num().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Num().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.Im(),aFp);
    BinaryUnDumpFromFile(anObj.Ter(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cMesureAppuis & anObj)
{
    BinaryDumpInFile(aFp,anObj.Num().IsInit());
    if (anObj.Num().IsInit()) BinaryDumpInFile(aFp,anObj.Num().Val());
    BinaryDumpInFile(aFp,anObj.Im());
    BinaryDumpInFile(aFp,anObj.Ter());
}

cElXMLTree * ToXMLTree(const cMesureAppuis & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"MesureAppuis",eXMLBranche);
   if (anObj.Num().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Num"),anObj.Num().Val())->ReTagThis("Num"));
   aRes->AddFils(::ToXMLTree(std::string("Im"),anObj.Im())->ReTagThis("Im"));
   aRes->AddFils(ToXMLTree(std::string("Ter"),anObj.Ter())->ReTagThis("Ter"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cMesureAppuis & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Num(),aTree->Get("Num",1)); //tototo 

   xml_init(anObj.Im(),aTree->Get("Im",1)); //tototo 

   xml_init(anObj.Ter(),aTree->Get("Ter",1)); //tototo 
}

std::string  Mangling( cMesureAppuis *) {return "CF40CF2C303271D2FF3F";};


cTplValGesInit< std::string > & cListeAppuis1Im::NameImage()
{
   return mNameImage;
}

const cTplValGesInit< std::string > & cListeAppuis1Im::NameImage()const 
{
   return mNameImage;
}


std::list< cMesureAppuis > & cListeAppuis1Im::Mesures()
{
   return mMesures;
}

const std::list< cMesureAppuis > & cListeAppuis1Im::Mesures()const 
{
   return mMesures;
}

void  BinaryUnDumpFromFile(cListeAppuis1Im & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NameImage().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NameImage().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NameImage().SetNoInit();
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cMesureAppuis aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.Mesures().push_back(aVal);
        }
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cListeAppuis1Im & anObj)
{
    BinaryDumpInFile(aFp,anObj.NameImage().IsInit());
    if (anObj.NameImage().IsInit()) BinaryDumpInFile(aFp,anObj.NameImage().Val());
    BinaryDumpInFile(aFp,(int)anObj.Mesures().size());
    for(  std::list< cMesureAppuis >::const_iterator iT=anObj.Mesures().begin();
         iT!=anObj.Mesures().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
}

cElXMLTree * ToXMLTree(const cListeAppuis1Im & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ListeAppuis1Im",eXMLBranche);
   if (anObj.NameImage().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NameImage"),anObj.NameImage().Val())->ReTagThis("NameImage"));
  for
  (       std::list< cMesureAppuis >::const_iterator it=anObj.Mesures().begin();
      it !=anObj.Mesures().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("Mesures"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cListeAppuis1Im & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.NameImage(),aTree->Get("NameImage",1),std::string("NoName")); //tototo 

   xml_init(anObj.Mesures(),aTree->GetAll("Mesures",false,1));
}

std::string  Mangling( cListeAppuis1Im *) {return "4BE7FA0B90B52DA9FD3F";};


double & cVerifOrient::Tol()
{
   return mTol;
}

const double & cVerifOrient::Tol()const 
{
   return mTol;
}


cTplValGesInit< bool > & cVerifOrient::ShowMes()
{
   return mShowMes;
}

const cTplValGesInit< bool > & cVerifOrient::ShowMes()const 
{
   return mShowMes;
}


std::list< cMesureAppuis > & cVerifOrient::Appuis()
{
   return mAppuis;
}

const std::list< cMesureAppuis > & cVerifOrient::Appuis()const 
{
   return mAppuis;
}


cTplValGesInit< bool > & cVerifOrient::IsTest()
{
   return mIsTest;
}

const cTplValGesInit< bool > & cVerifOrient::IsTest()const 
{
   return mIsTest;
}


cTplValGesInit< cListeAppuis1Im > & cVerifOrient::AppuisConv()
{
   return mAppuisConv;
}

const cTplValGesInit< cListeAppuis1Im > & cVerifOrient::AppuisConv()const 
{
   return mAppuisConv;
}

void  BinaryUnDumpFromFile(cVerifOrient & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Tol(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ShowMes().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ShowMes().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ShowMes().SetNoInit();
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cMesureAppuis aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.Appuis().push_back(aVal);
        }
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.IsTest().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.IsTest().ValForcedForUnUmp(),aFp);
        }
        else  anObj.IsTest().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.AppuisConv().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.AppuisConv().ValForcedForUnUmp(),aFp);
        }
        else  anObj.AppuisConv().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cVerifOrient & anObj)
{
    BinaryDumpInFile(aFp,anObj.Tol());
    BinaryDumpInFile(aFp,anObj.ShowMes().IsInit());
    if (anObj.ShowMes().IsInit()) BinaryDumpInFile(aFp,anObj.ShowMes().Val());
    BinaryDumpInFile(aFp,(int)anObj.Appuis().size());
    for(  std::list< cMesureAppuis >::const_iterator iT=anObj.Appuis().begin();
         iT!=anObj.Appuis().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.IsTest().IsInit());
    if (anObj.IsTest().IsInit()) BinaryDumpInFile(aFp,anObj.IsTest().Val());
    BinaryDumpInFile(aFp,anObj.AppuisConv().IsInit());
    if (anObj.AppuisConv().IsInit()) BinaryDumpInFile(aFp,anObj.AppuisConv().Val());
}

cElXMLTree * ToXMLTree(const cVerifOrient & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"VerifOrient",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Tol"),anObj.Tol())->ReTagThis("Tol"));
   if (anObj.ShowMes().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ShowMes"),anObj.ShowMes().Val())->ReTagThis("ShowMes"));
  for
  (       std::list< cMesureAppuis >::const_iterator it=anObj.Appuis().begin();
      it !=anObj.Appuis().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("Appuis"));
   if (anObj.IsTest().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("IsTest"),anObj.IsTest().Val())->ReTagThis("IsTest"));
   if (anObj.AppuisConv().IsInit())
      aRes->AddFils(ToXMLTree(anObj.AppuisConv().Val())->ReTagThis("AppuisConv"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cVerifOrient & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Tol(),aTree->Get("Tol",1)); //tototo 

   xml_init(anObj.ShowMes(),aTree->Get("ShowMes",1),bool(false)); //tototo 

   xml_init(anObj.Appuis(),aTree->GetAll("Appuis",false,1));

   xml_init(anObj.IsTest(),aTree->Get("IsTest",1),bool(false)); //tototo 

   xml_init(anObj.AppuisConv(),aTree->Get("AppuisConv",1)); //tototo 
}

std::string  Mangling( cVerifOrient *) {return "D232494E2900D8FAFE3F";};

eConventionsOrientation  Str2eConventionsOrientation(const std::string & aName)
{
   if (aName=="eConvInconnue")
      return eConvInconnue;
   else if (aName=="eConvApero_DistC2M")
      return eConvApero_DistC2M;
   else if (aName=="eConvApero_DistM2C")
      return eConvApero_DistM2C;
   else if (aName=="eConvOriLib")
      return eConvOriLib;
   else if (aName=="eConvMatrPoivillier_E")
      return eConvMatrPoivillier_E;
   else if (aName=="eConvAngErdas")
      return eConvAngErdas;
   else if (aName=="eConvAngErdas_Grade")
      return eConvAngErdas_Grade;
   else if (aName=="eConvAngAvionJaune")
      return eConvAngAvionJaune;
   else if (aName=="eConvAngSurvey")
      return eConvAngSurvey;
   else if (aName=="eConvAngPhotoMDegre")
      return eConvAngPhotoMDegre;
   else if (aName=="eConvAngPhotoMGrade")
      return eConvAngPhotoMGrade;
   else if (aName=="eConvAngLPSDegre")
      return eConvAngLPSDegre;
   else if (aName=="eConvMatrixInpho")
      return eConvMatrixInpho;
  else
  {
      cout << aName << " is not a correct value for enum eConventionsOrientation\n" ;
      ELISE_ASSERT(false,"XML enum value error");
  }
  return (eConventionsOrientation) 0;
}
void xml_init(eConventionsOrientation & aVal,cElXMLTree * aTree)
{
   aVal= Str2eConventionsOrientation(aTree->Contenu());
}
std::string  eToString(const eConventionsOrientation & anObj)
{
   if (anObj==eConvInconnue)
      return  "eConvInconnue";
   if (anObj==eConvApero_DistC2M)
      return  "eConvApero_DistC2M";
   if (anObj==eConvApero_DistM2C)
      return  "eConvApero_DistM2C";
   if (anObj==eConvOriLib)
      return  "eConvOriLib";
   if (anObj==eConvMatrPoivillier_E)
      return  "eConvMatrPoivillier_E";
   if (anObj==eConvAngErdas)
      return  "eConvAngErdas";
   if (anObj==eConvAngErdas_Grade)
      return  "eConvAngErdas_Grade";
   if (anObj==eConvAngAvionJaune)
      return  "eConvAngAvionJaune";
   if (anObj==eConvAngSurvey)
      return  "eConvAngSurvey";
   if (anObj==eConvAngPhotoMDegre)
      return  "eConvAngPhotoMDegre";
   if (anObj==eConvAngPhotoMGrade)
      return  "eConvAngPhotoMGrade";
   if (anObj==eConvAngLPSDegre)
      return  "eConvAngLPSDegre";
   if (anObj==eConvMatrixInpho)
      return  "eConvMatrixInpho";
 std::cout << "Enum = eConventionsOrientation\n";
   ELISE_ASSERT(false,"Bad Value in eToString for enum value ");
   return "";
}

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eConventionsOrientation & anObj)
{
      return  cElXMLTree::ValueNode(aNameTag,eToString(anObj));
}

void  BinaryDumpInFile(ELISE_fp & aFp,const eConventionsOrientation & anObj)
{
   BinaryDumpInFile(aFp,int(anObj));
}

void  BinaryUnDumpFromFile(eConventionsOrientation & anObj,ELISE_fp & aFp)
{
   int aIVal;
   BinaryUnDumpFromFile(aIVal,aFp);
   anObj=(eConventionsOrientation) aIVal;
}

std::string  Mangling( eConventionsOrientation *) {return "D45887C8F44ACDAFFC3F";};


Pt2dr & cCalibrationInterneGridDef::P0()
{
   return mP0;
}

const Pt2dr & cCalibrationInterneGridDef::P0()const 
{
   return mP0;
}


Pt2dr & cCalibrationInterneGridDef::Sz()
{
   return mSz;
}

const Pt2dr & cCalibrationInterneGridDef::Sz()const 
{
   return mSz;
}


Pt2di & cCalibrationInterneGridDef::Nb()
{
   return mNb;
}

const Pt2di & cCalibrationInterneGridDef::Nb()const 
{
   return mNb;
}


std::vector< Pt2dr > & cCalibrationInterneGridDef::PGr()
{
   return mPGr;
}

const std::vector< Pt2dr > & cCalibrationInterneGridDef::PGr()const 
{
   return mPGr;
}

void  BinaryUnDumpFromFile(cCalibrationInterneGridDef & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.P0(),aFp);
    BinaryUnDumpFromFile(anObj.Sz(),aFp);
    BinaryUnDumpFromFile(anObj.Nb(),aFp);
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             Pt2dr aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.PGr().push_back(aVal);
        }
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cCalibrationInterneGridDef & anObj)
{
    BinaryDumpInFile(aFp,anObj.P0());
    BinaryDumpInFile(aFp,anObj.Sz());
    BinaryDumpInFile(aFp,anObj.Nb());
    BinaryDumpInFile(aFp,(int)anObj.PGr().size());
    for(  std::vector< Pt2dr >::const_iterator iT=anObj.PGr().begin();
         iT!=anObj.PGr().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
}

cElXMLTree * ToXMLTree(const cCalibrationInterneGridDef & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"CalibrationInterneGridDef",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("P0"),anObj.P0())->ReTagThis("P0"));
   aRes->AddFils(::ToXMLTree(std::string("Sz"),anObj.Sz())->ReTagThis("Sz"));
   aRes->AddFils(::ToXMLTree(std::string("Nb"),anObj.Nb())->ReTagThis("Nb"));
  for
  (       std::vector< Pt2dr >::const_iterator it=anObj.PGr().begin();
      it !=anObj.PGr().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("PGr"),(*it))->ReTagThis("PGr"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cCalibrationInterneGridDef & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.P0(),aTree->Get("P0",1)); //tototo 

   xml_init(anObj.Sz(),aTree->Get("Sz",1)); //tototo 

   xml_init(anObj.Nb(),aTree->Get("Nb",1)); //tototo 

   xml_init(anObj.PGr(),aTree->GetAll("PGr",false,1));
}

std::string  Mangling( cCalibrationInterneGridDef *) {return "A4F484050A7D1BC4FD3F";};


Pt2dr & cCalibrationInterneRadiale::CDist()
{
   return mCDist;
}

const Pt2dr & cCalibrationInterneRadiale::CDist()const 
{
   return mCDist;
}


std::vector< double > & cCalibrationInterneRadiale::CoeffDist()
{
   return mCoeffDist;
}

const std::vector< double > & cCalibrationInterneRadiale::CoeffDist()const 
{
   return mCoeffDist;
}


cTplValGesInit< double > & cCalibrationInterneRadiale::RatioDistInv()
{
   return mRatioDistInv;
}

const cTplValGesInit< double > & cCalibrationInterneRadiale::RatioDistInv()const 
{
   return mRatioDistInv;
}


cTplValGesInit< bool > & cCalibrationInterneRadiale::PPaEqPPs()
{
   return mPPaEqPPs;
}

const cTplValGesInit< bool > & cCalibrationInterneRadiale::PPaEqPPs()const 
{
   return mPPaEqPPs;
}

void  BinaryUnDumpFromFile(cCalibrationInterneRadiale & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.CDist(),aFp);
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             double aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.CoeffDist().push_back(aVal);
        }
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.RatioDistInv().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.RatioDistInv().ValForcedForUnUmp(),aFp);
        }
        else  anObj.RatioDistInv().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.PPaEqPPs().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.PPaEqPPs().ValForcedForUnUmp(),aFp);
        }
        else  anObj.PPaEqPPs().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cCalibrationInterneRadiale & anObj)
{
    BinaryDumpInFile(aFp,anObj.CDist());
    BinaryDumpInFile(aFp,(int)anObj.CoeffDist().size());
    for(  std::vector< double >::const_iterator iT=anObj.CoeffDist().begin();
         iT!=anObj.CoeffDist().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.RatioDistInv().IsInit());
    if (anObj.RatioDistInv().IsInit()) BinaryDumpInFile(aFp,anObj.RatioDistInv().Val());
    BinaryDumpInFile(aFp,anObj.PPaEqPPs().IsInit());
    if (anObj.PPaEqPPs().IsInit()) BinaryDumpInFile(aFp,anObj.PPaEqPPs().Val());
}

cElXMLTree * ToXMLTree(const cCalibrationInterneRadiale & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"CalibrationInterneRadiale",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("CDist"),anObj.CDist())->ReTagThis("CDist"));
  for
  (       std::vector< double >::const_iterator it=anObj.CoeffDist().begin();
      it !=anObj.CoeffDist().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("CoeffDist"),(*it))->ReTagThis("CoeffDist"));
   if (anObj.RatioDistInv().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("RatioDistInv"),anObj.RatioDistInv().Val())->ReTagThis("RatioDistInv"));
   if (anObj.PPaEqPPs().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PPaEqPPs"),anObj.PPaEqPPs().Val())->ReTagThis("PPaEqPPs"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cCalibrationInterneRadiale & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.CDist(),aTree->Get("CDist",1)); //tototo 

   xml_init(anObj.CoeffDist(),aTree->GetAll("CoeffDist",false,1));

   xml_init(anObj.RatioDistInv(),aTree->Get("RatioDistInv",1),double(1.3)); //tototo 

   xml_init(anObj.PPaEqPPs(),aTree->Get("PPaEqPPs",1),bool(false)); //tototo 
}

std::string  Mangling( cCalibrationInterneRadiale *) {return "160C948B5429E6E7FE3F";};


cCalibrationInterneRadiale & cCalibrationInternePghrStd::RadialePart()
{
   return mRadialePart;
}

const cCalibrationInterneRadiale & cCalibrationInternePghrStd::RadialePart()const 
{
   return mRadialePart;
}


cTplValGesInit< double > & cCalibrationInternePghrStd::P1()
{
   return mP1;
}

const cTplValGesInit< double > & cCalibrationInternePghrStd::P1()const 
{
   return mP1;
}


cTplValGesInit< double > & cCalibrationInternePghrStd::P2()
{
   return mP2;
}

const cTplValGesInit< double > & cCalibrationInternePghrStd::P2()const 
{
   return mP2;
}


cTplValGesInit< double > & cCalibrationInternePghrStd::b1()
{
   return mb1;
}

const cTplValGesInit< double > & cCalibrationInternePghrStd::b1()const 
{
   return mb1;
}


cTplValGesInit< double > & cCalibrationInternePghrStd::b2()
{
   return mb2;
}

const cTplValGesInit< double > & cCalibrationInternePghrStd::b2()const 
{
   return mb2;
}

void  BinaryUnDumpFromFile(cCalibrationInternePghrStd & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.RadialePart(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.P1().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.P1().ValForcedForUnUmp(),aFp);
        }
        else  anObj.P1().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.P2().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.P2().ValForcedForUnUmp(),aFp);
        }
        else  anObj.P2().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.b1().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.b1().ValForcedForUnUmp(),aFp);
        }
        else  anObj.b1().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.b2().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.b2().ValForcedForUnUmp(),aFp);
        }
        else  anObj.b2().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cCalibrationInternePghrStd & anObj)
{
    BinaryDumpInFile(aFp,anObj.RadialePart());
    BinaryDumpInFile(aFp,anObj.P1().IsInit());
    if (anObj.P1().IsInit()) BinaryDumpInFile(aFp,anObj.P1().Val());
    BinaryDumpInFile(aFp,anObj.P2().IsInit());
    if (anObj.P2().IsInit()) BinaryDumpInFile(aFp,anObj.P2().Val());
    BinaryDumpInFile(aFp,anObj.b1().IsInit());
    if (anObj.b1().IsInit()) BinaryDumpInFile(aFp,anObj.b1().Val());
    BinaryDumpInFile(aFp,anObj.b2().IsInit());
    if (anObj.b2().IsInit()) BinaryDumpInFile(aFp,anObj.b2().Val());
}

cElXMLTree * ToXMLTree(const cCalibrationInternePghrStd & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"CalibrationInternePghrStd",eXMLBranche);
   aRes->AddFils(ToXMLTree(anObj.RadialePart())->ReTagThis("RadialePart"));
   if (anObj.P1().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("P1"),anObj.P1().Val())->ReTagThis("P1"));
   if (anObj.P2().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("P2"),anObj.P2().Val())->ReTagThis("P2"));
   if (anObj.b1().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("b1"),anObj.b1().Val())->ReTagThis("b1"));
   if (anObj.b2().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("b2"),anObj.b2().Val())->ReTagThis("b2"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cCalibrationInternePghrStd & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.RadialePart(),aTree->Get("RadialePart",1)); //tototo 

   xml_init(anObj.P1(),aTree->Get("P1",1),double(0.0)); //tototo 

   xml_init(anObj.P2(),aTree->Get("P2",1),double(0.0)); //tototo 

   xml_init(anObj.b1(),aTree->Get("b1",1),double(0.0)); //tototo 

   xml_init(anObj.b2(),aTree->Get("b2",1),double(0.0)); //tototo 
}

std::string  Mangling( cCalibrationInternePghrStd *) {return "46BB04EBB3BADC9EFF3F";};


eModelesCalibUnif & cCalibrationInterneUnif::TypeModele()
{
   return mTypeModele;
}

const eModelesCalibUnif & cCalibrationInterneUnif::TypeModele()const 
{
   return mTypeModele;
}


std::vector< double > & cCalibrationInterneUnif::Params()
{
   return mParams;
}

const std::vector< double > & cCalibrationInterneUnif::Params()const 
{
   return mParams;
}


std::vector< double > & cCalibrationInterneUnif::Etats()
{
   return mEtats;
}

const std::vector< double > & cCalibrationInterneUnif::Etats()const 
{
   return mEtats;
}

void  BinaryUnDumpFromFile(cCalibrationInterneUnif & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.TypeModele(),aFp);
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             double aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.Params().push_back(aVal);
        }
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             double aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.Etats().push_back(aVal);
        }
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cCalibrationInterneUnif & anObj)
{
    BinaryDumpInFile(aFp,anObj.TypeModele());
    BinaryDumpInFile(aFp,(int)anObj.Params().size());
    for(  std::vector< double >::const_iterator iT=anObj.Params().begin();
         iT!=anObj.Params().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,(int)anObj.Etats().size());
    for(  std::vector< double >::const_iterator iT=anObj.Etats().begin();
         iT!=anObj.Etats().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
}

cElXMLTree * ToXMLTree(const cCalibrationInterneUnif & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"CalibrationInterneUnif",eXMLBranche);
   aRes->AddFils(ToXMLTree(std::string("TypeModele"),anObj.TypeModele())->ReTagThis("TypeModele"));
  for
  (       std::vector< double >::const_iterator it=anObj.Params().begin();
      it !=anObj.Params().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("Params"),(*it))->ReTagThis("Params"));
  for
  (       std::vector< double >::const_iterator it=anObj.Etats().begin();
      it !=anObj.Etats().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("Etats"),(*it))->ReTagThis("Etats"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cCalibrationInterneUnif & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.TypeModele(),aTree->Get("TypeModele",1)); //tototo 

   xml_init(anObj.Params(),aTree->GetAll("Params",false,1));

   xml_init(anObj.Etats(),aTree->GetAll("Etats",false,1));
}

std::string  Mangling( cCalibrationInterneUnif *) {return "A8632B50E42D909BFDBF";};


std::string & cTestNewGrid::A()
{
   return mA;
}

const std::string & cTestNewGrid::A()const 
{
   return mA;
}


Im2D_INT1 & cTestNewGrid::Im()
{
   return mIm;
}

const Im2D_INT1 & cTestNewGrid::Im()const 
{
   return mIm;
}


std::string & cTestNewGrid::Z()
{
   return mZ;
}

const std::string & cTestNewGrid::Z()const 
{
   return mZ;
}

void  BinaryUnDumpFromFile(cTestNewGrid & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.A(),aFp);
    BinaryUnDumpFromFile(anObj.Im(),aFp);
    BinaryUnDumpFromFile(anObj.Z(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cTestNewGrid & anObj)
{
    BinaryDumpInFile(aFp,anObj.A());
    BinaryDumpInFile(aFp,anObj.Im());
    BinaryDumpInFile(aFp,anObj.Z());
}

cElXMLTree * ToXMLTree(const cTestNewGrid & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"TestNewGrid",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("A"),anObj.A())->ReTagThis("A"));
   aRes->AddFils(::ToXMLTree(std::string("Im"),anObj.Im())->ReTagThis("Im"));
   aRes->AddFils(::ToXMLTree(std::string("Z"),anObj.Z())->ReTagThis("Z"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cTestNewGrid & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.A(),aTree->Get("A",1)); //tototo 

   xml_init(anObj.Im(),aTree->Get("Im",1)); //tototo 

   xml_init(anObj.Z(),aTree->Get("Z",1)); //tototo 
}

std::string  Mangling( cTestNewGrid *) {return "C6C1287A857D21C7FD3F";};


Pt2dr & cGridDeform2D::Origine()
{
   return mOrigine;
}

const Pt2dr & cGridDeform2D::Origine()const 
{
   return mOrigine;
}


Pt2dr & cGridDeform2D::Step()
{
   return mStep;
}

const Pt2dr & cGridDeform2D::Step()const 
{
   return mStep;
}


Im2D_REAL8 & cGridDeform2D::ImX()
{
   return mImX;
}

const Im2D_REAL8 & cGridDeform2D::ImX()const 
{
   return mImX;
}


Im2D_REAL8 & cGridDeform2D::ImY()
{
   return mImY;
}

const Im2D_REAL8 & cGridDeform2D::ImY()const 
{
   return mImY;
}

void  BinaryUnDumpFromFile(cGridDeform2D & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Origine(),aFp);
    BinaryUnDumpFromFile(anObj.Step(),aFp);
    BinaryUnDumpFromFile(anObj.ImX(),aFp);
    BinaryUnDumpFromFile(anObj.ImY(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cGridDeform2D & anObj)
{
    BinaryDumpInFile(aFp,anObj.Origine());
    BinaryDumpInFile(aFp,anObj.Step());
    BinaryDumpInFile(aFp,anObj.ImX());
    BinaryDumpInFile(aFp,anObj.ImY());
}

cElXMLTree * ToXMLTree(const cGridDeform2D & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"GridDeform2D",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Origine"),anObj.Origine())->ReTagThis("Origine"));
   aRes->AddFils(::ToXMLTree(std::string("Step"),anObj.Step())->ReTagThis("Step"));
   aRes->AddFils(::ToXMLTree(std::string("ImX"),anObj.ImX())->ReTagThis("ImX"));
   aRes->AddFils(::ToXMLTree(std::string("ImY"),anObj.ImY())->ReTagThis("ImY"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cGridDeform2D & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Origine(),aTree->Get("Origine",1)); //tototo 

   xml_init(anObj.Step(),aTree->Get("Step",1)); //tototo 

   xml_init(anObj.ImX(),aTree->Get("ImX",1)); //tototo 

   xml_init(anObj.ImY(),aTree->Get("ImY",1)); //tototo 
}

std::string  Mangling( cGridDeform2D *) {return "4CABA2FBB21E4D8CFE3F";};


cGridDeform2D & cGridDirecteEtInverse::Directe()
{
   return mDirecte;
}

const cGridDeform2D & cGridDirecteEtInverse::Directe()const 
{
   return mDirecte;
}


cGridDeform2D & cGridDirecteEtInverse::Inverse()
{
   return mInverse;
}

const cGridDeform2D & cGridDirecteEtInverse::Inverse()const 
{
   return mInverse;
}


bool & cGridDirecteEtInverse::AdaptStep()
{
   return mAdaptStep;
}

const bool & cGridDirecteEtInverse::AdaptStep()const 
{
   return mAdaptStep;
}

void  BinaryUnDumpFromFile(cGridDirecteEtInverse & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Directe(),aFp);
    BinaryUnDumpFromFile(anObj.Inverse(),aFp);
    BinaryUnDumpFromFile(anObj.AdaptStep(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cGridDirecteEtInverse & anObj)
{
    BinaryDumpInFile(aFp,anObj.Directe());
    BinaryDumpInFile(aFp,anObj.Inverse());
    BinaryDumpInFile(aFp,anObj.AdaptStep());
}

cElXMLTree * ToXMLTree(const cGridDirecteEtInverse & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"GridDirecteEtInverse",eXMLBranche);
   aRes->AddFils(ToXMLTree(anObj.Directe())->ReTagThis("Directe"));
   aRes->AddFils(ToXMLTree(anObj.Inverse())->ReTagThis("Inverse"));
   aRes->AddFils(::ToXMLTree(std::string("AdaptStep"),anObj.AdaptStep())->ReTagThis("AdaptStep"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cGridDirecteEtInverse & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Directe(),aTree->Get("Directe",1)); //tototo 

   xml_init(anObj.Inverse(),aTree->Get("Inverse",1)); //tototo 

   xml_init(anObj.AdaptStep(),aTree->Get("AdaptStep",1)); //tototo 
}

std::string  Mangling( cGridDirecteEtInverse *) {return "4F2BE2AC5A55D8DEFE3F";};


Pt2dr & cPreCondRadial::C()
{
   return mC;
}

const Pt2dr & cPreCondRadial::C()const 
{
   return mC;
}


double & cPreCondRadial::F()
{
   return mF;
}

const double & cPreCondRadial::F()const 
{
   return mF;
}


eTypePreCondRad & cPreCondRadial::Mode()
{
   return mMode;
}

const eTypePreCondRad & cPreCondRadial::Mode()const 
{
   return mMode;
}

void  BinaryUnDumpFromFile(cPreCondRadial & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.C(),aFp);
    BinaryUnDumpFromFile(anObj.F(),aFp);
    BinaryUnDumpFromFile(anObj.Mode(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cPreCondRadial & anObj)
{
    BinaryDumpInFile(aFp,anObj.C());
    BinaryDumpInFile(aFp,anObj.F());
    BinaryDumpInFile(aFp,anObj.Mode());
}

cElXMLTree * ToXMLTree(const cPreCondRadial & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"PreCondRadial",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("C"),anObj.C())->ReTagThis("C"));
   aRes->AddFils(::ToXMLTree(std::string("F"),anObj.F())->ReTagThis("F"));
   aRes->AddFils(ToXMLTree(std::string("Mode"),anObj.Mode())->ReTagThis("Mode"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cPreCondRadial & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.C(),aTree->Get("C",1)); //tototo 

   xml_init(anObj.F(),aTree->Get("F",1)); //tototo 

   xml_init(anObj.Mode(),aTree->Get("Mode",1)); //tototo 
}

std::string  Mangling( cPreCondRadial *) {return "EB69821946F891BCFE3F";};


Pt2dr & cPreCondGrid::C()
{
   return PreCondRadial().Val().C();
}

const Pt2dr & cPreCondGrid::C()const 
{
   return PreCondRadial().Val().C();
}


double & cPreCondGrid::F()
{
   return PreCondRadial().Val().F();
}

const double & cPreCondGrid::F()const 
{
   return PreCondRadial().Val().F();
}


eTypePreCondRad & cPreCondGrid::Mode()
{
   return PreCondRadial().Val().Mode();
}

const eTypePreCondRad & cPreCondGrid::Mode()const 
{
   return PreCondRadial().Val().Mode();
}


cTplValGesInit< cPreCondRadial > & cPreCondGrid::PreCondRadial()
{
   return mPreCondRadial;
}

const cTplValGesInit< cPreCondRadial > & cPreCondGrid::PreCondRadial()const 
{
   return mPreCondRadial;
}

void  BinaryUnDumpFromFile(cPreCondGrid & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.PreCondRadial().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.PreCondRadial().ValForcedForUnUmp(),aFp);
        }
        else  anObj.PreCondRadial().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cPreCondGrid & anObj)
{
    BinaryDumpInFile(aFp,anObj.PreCondRadial().IsInit());
    if (anObj.PreCondRadial().IsInit()) BinaryDumpInFile(aFp,anObj.PreCondRadial().Val());
}

cElXMLTree * ToXMLTree(const cPreCondGrid & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"PreCondGrid",eXMLBranche);
   if (anObj.PreCondRadial().IsInit())
      aRes->AddFils(ToXMLTree(anObj.PreCondRadial().Val())->ReTagThis("PreCondRadial"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cPreCondGrid & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.PreCondRadial(),aTree->Get("PreCondRadial",1)); //tototo 
}

std::string  Mangling( cPreCondGrid *) {return "60C2E923469BB1A5FE3F";};


Pt2dr & cCalibrationInterneGrid::C()
{
   return PreCondGrid().Val().PreCondRadial().Val().C();
}

const Pt2dr & cCalibrationInterneGrid::C()const 
{
   return PreCondGrid().Val().PreCondRadial().Val().C();
}


double & cCalibrationInterneGrid::F()
{
   return PreCondGrid().Val().PreCondRadial().Val().F();
}

const double & cCalibrationInterneGrid::F()const 
{
   return PreCondGrid().Val().PreCondRadial().Val().F();
}


eTypePreCondRad & cCalibrationInterneGrid::Mode()
{
   return PreCondGrid().Val().PreCondRadial().Val().Mode();
}

const eTypePreCondRad & cCalibrationInterneGrid::Mode()const 
{
   return PreCondGrid().Val().PreCondRadial().Val().Mode();
}


cTplValGesInit< cPreCondRadial > & cCalibrationInterneGrid::PreCondRadial()
{
   return PreCondGrid().Val().PreCondRadial();
}

const cTplValGesInit< cPreCondRadial > & cCalibrationInterneGrid::PreCondRadial()const 
{
   return PreCondGrid().Val().PreCondRadial();
}


cTplValGesInit< cPreCondGrid > & cCalibrationInterneGrid::PreCondGrid()
{
   return mPreCondGrid;
}

const cTplValGesInit< cPreCondGrid > & cCalibrationInterneGrid::PreCondGrid()const 
{
   return mPreCondGrid;
}


cGridDirecteEtInverse & cCalibrationInterneGrid::Grid()
{
   return mGrid;
}

const cGridDirecteEtInverse & cCalibrationInterneGrid::Grid()const 
{
   return mGrid;
}

void  BinaryUnDumpFromFile(cCalibrationInterneGrid & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.PreCondGrid().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.PreCondGrid().ValForcedForUnUmp(),aFp);
        }
        else  anObj.PreCondGrid().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.Grid(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cCalibrationInterneGrid & anObj)
{
    BinaryDumpInFile(aFp,anObj.PreCondGrid().IsInit());
    if (anObj.PreCondGrid().IsInit()) BinaryDumpInFile(aFp,anObj.PreCondGrid().Val());
    BinaryDumpInFile(aFp,anObj.Grid());
}

cElXMLTree * ToXMLTree(const cCalibrationInterneGrid & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"CalibrationInterneGrid",eXMLBranche);
   if (anObj.PreCondGrid().IsInit())
      aRes->AddFils(ToXMLTree(anObj.PreCondGrid().Val())->ReTagThis("PreCondGrid"));
   aRes->AddFils(ToXMLTree(anObj.Grid())->ReTagThis("Grid"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cCalibrationInterneGrid & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.PreCondGrid(),aTree->Get("PreCondGrid",1)); //tototo 

   xml_init(anObj.Grid(),aTree->Get("Grid",1)); //tototo 
}

std::string  Mangling( cCalibrationInterneGrid *) {return "B38A18591DC29CCAFE3F";};


Pt2dr & cSimilitudePlane::Scale()
{
   return mScale;
}

const Pt2dr & cSimilitudePlane::Scale()const 
{
   return mScale;
}


Pt2dr & cSimilitudePlane::Trans()
{
   return mTrans;
}

const Pt2dr & cSimilitudePlane::Trans()const 
{
   return mTrans;
}

void  BinaryUnDumpFromFile(cSimilitudePlane & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Scale(),aFp);
    BinaryUnDumpFromFile(anObj.Trans(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cSimilitudePlane & anObj)
{
    BinaryDumpInFile(aFp,anObj.Scale());
    BinaryDumpInFile(aFp,anObj.Trans());
}

cElXMLTree * ToXMLTree(const cSimilitudePlane & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"SimilitudePlane",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Scale"),anObj.Scale())->ReTagThis("Scale"));
   aRes->AddFils(::ToXMLTree(std::string("Trans"),anObj.Trans())->ReTagThis("Trans"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cSimilitudePlane & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Scale(),aTree->Get("Scale",1)); //tototo 

   xml_init(anObj.Trans(),aTree->Get("Trans",1)); //tototo 
}

std::string  Mangling( cSimilitudePlane *) {return "EE18FE2A05BE6887FD3F";};


Pt2dr & cAffinitePlane::I00()
{
   return mI00;
}

const Pt2dr & cAffinitePlane::I00()const 
{
   return mI00;
}


Pt2dr & cAffinitePlane::V10()
{
   return mV10;
}

const Pt2dr & cAffinitePlane::V10()const 
{
   return mV10;
}


Pt2dr & cAffinitePlane::V01()
{
   return mV01;
}

const Pt2dr & cAffinitePlane::V01()const 
{
   return mV01;
}

void  BinaryUnDumpFromFile(cAffinitePlane & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.I00(),aFp);
    BinaryUnDumpFromFile(anObj.V10(),aFp);
    BinaryUnDumpFromFile(anObj.V01(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cAffinitePlane & anObj)
{
    BinaryDumpInFile(aFp,anObj.I00());
    BinaryDumpInFile(aFp,anObj.V10());
    BinaryDumpInFile(aFp,anObj.V01());
}

cElXMLTree * ToXMLTree(const cAffinitePlane & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"AffinitePlane",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("I00"),anObj.I00())->ReTagThis("I00"));
   aRes->AddFils(::ToXMLTree(std::string("V10"),anObj.V10())->ReTagThis("V10"));
   aRes->AddFils(::ToXMLTree(std::string("V01"),anObj.V01())->ReTagThis("V01"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cAffinitePlane & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.I00(),aTree->Get("I00",1)); //tototo 

   xml_init(anObj.V10(),aTree->Get("V10",1)); //tototo 

   xml_init(anObj.V01(),aTree->Get("V01",1)); //tototo 
}

std::string  Mangling( cAffinitePlane *) {return "3ED1AA4EF9FA47EBFE3F";};


cAffinitePlane & cOrIntGlob::Affinite()
{
   return mAffinite;
}

const cAffinitePlane & cOrIntGlob::Affinite()const 
{
   return mAffinite;
}


bool & cOrIntGlob::C2M()
{
   return mC2M;
}

const bool & cOrIntGlob::C2M()const 
{
   return mC2M;
}

void  BinaryUnDumpFromFile(cOrIntGlob & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Affinite(),aFp);
    BinaryUnDumpFromFile(anObj.C2M(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cOrIntGlob & anObj)
{
    BinaryDumpInFile(aFp,anObj.Affinite());
    BinaryDumpInFile(aFp,anObj.C2M());
}

cElXMLTree * ToXMLTree(const cOrIntGlob & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"OrIntGlob",eXMLBranche);
   aRes->AddFils(ToXMLTree(anObj.Affinite())->ReTagThis("Affinite"));
   aRes->AddFils(::ToXMLTree(std::string("C2M"),anObj.C2M())->ReTagThis("C2M"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cOrIntGlob & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Affinite(),aTree->Get("Affinite",1)); //tototo 

   xml_init(anObj.C2M(),aTree->Get("C2M",1)); //tototo 
}

std::string  Mangling( cOrIntGlob *) {return "B7EA9CEC3C25A9F2FD3F";};


Pt2dr & cParamForGrid::StepGrid()
{
   return mStepGrid;
}

const Pt2dr & cParamForGrid::StepGrid()const 
{
   return mStepGrid;
}


double & cParamForGrid::RayonInv()
{
   return mRayonInv;
}

const double & cParamForGrid::RayonInv()const 
{
   return mRayonInv;
}

void  BinaryUnDumpFromFile(cParamForGrid & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.StepGrid(),aFp);
    BinaryUnDumpFromFile(anObj.RayonInv(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cParamForGrid & anObj)
{
    BinaryDumpInFile(aFp,anObj.StepGrid());
    BinaryDumpInFile(aFp,anObj.RayonInv());
}

cElXMLTree * ToXMLTree(const cParamForGrid & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ParamForGrid",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("StepGrid"),anObj.StepGrid())->ReTagThis("StepGrid"));
   aRes->AddFils(::ToXMLTree(std::string("RayonInv"),anObj.RayonInv())->ReTagThis("RayonInv"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cParamForGrid & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.StepGrid(),aTree->Get("StepGrid",1)); //tototo 

   xml_init(anObj.RayonInv(),aTree->Get("RayonInv",1)); //tototo 
}

std::string  Mangling( cParamForGrid *) {return "90532E6FAE4C1092FABF";};


cTplValGesInit< std::string > & cModNoDist::Inutile()
{
   return mInutile;
}

const cTplValGesInit< std::string > & cModNoDist::Inutile()const 
{
   return mInutile;
}

void  BinaryUnDumpFromFile(cModNoDist & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Inutile().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Inutile().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Inutile().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cModNoDist & anObj)
{
    BinaryDumpInFile(aFp,anObj.Inutile().IsInit());
    if (anObj.Inutile().IsInit()) BinaryDumpInFile(aFp,anObj.Inutile().Val());
}

cElXMLTree * ToXMLTree(const cModNoDist & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ModNoDist",eXMLBranche);
   if (anObj.Inutile().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Inutile"),anObj.Inutile().Val())->ReTagThis("Inutile"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cModNoDist & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Inutile(),aTree->Get("Inutile",1)); //tototo 
}

std::string  Mangling( cModNoDist *) {return "10B0B37A531A6994FBBF";};


cTplValGesInit< std::string > & cCalibDistortion::Inutile()
{
   return ModNoDist().Val().Inutile();
}

const cTplValGesInit< std::string > & cCalibDistortion::Inutile()const 
{
   return ModNoDist().Val().Inutile();
}


cTplValGesInit< cModNoDist > & cCalibDistortion::ModNoDist()
{
   return mModNoDist;
}

const cTplValGesInit< cModNoDist > & cCalibDistortion::ModNoDist()const 
{
   return mModNoDist;
}


cTplValGesInit< cCalibrationInterneRadiale > & cCalibDistortion::ModRad()
{
   return mModRad;
}

const cTplValGesInit< cCalibrationInterneRadiale > & cCalibDistortion::ModRad()const 
{
   return mModRad;
}


cTplValGesInit< cCalibrationInternePghrStd > & cCalibDistortion::ModPhgrStd()
{
   return mModPhgrStd;
}

const cTplValGesInit< cCalibrationInternePghrStd > & cCalibDistortion::ModPhgrStd()const 
{
   return mModPhgrStd;
}


cTplValGesInit< cCalibrationInterneUnif > & cCalibDistortion::ModUnif()
{
   return mModUnif;
}

const cTplValGesInit< cCalibrationInterneUnif > & cCalibDistortion::ModUnif()const 
{
   return mModUnif;
}


cTplValGesInit< cCalibrationInterneGrid > & cCalibDistortion::ModGrid()
{
   return mModGrid;
}

const cTplValGesInit< cCalibrationInterneGrid > & cCalibDistortion::ModGrid()const 
{
   return mModGrid;
}


cTplValGesInit< cCalibrationInterneGridDef > & cCalibDistortion::ModGridDef()
{
   return mModGridDef;
}

const cTplValGesInit< cCalibrationInterneGridDef > & cCalibDistortion::ModGridDef()const 
{
   return mModGridDef;
}

void  BinaryUnDumpFromFile(cCalibDistortion & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ModNoDist().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ModNoDist().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ModNoDist().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ModRad().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ModRad().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ModRad().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ModPhgrStd().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ModPhgrStd().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ModPhgrStd().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ModUnif().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ModUnif().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ModUnif().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ModGrid().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ModGrid().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ModGrid().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ModGridDef().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ModGridDef().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ModGridDef().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cCalibDistortion & anObj)
{
    BinaryDumpInFile(aFp,anObj.ModNoDist().IsInit());
    if (anObj.ModNoDist().IsInit()) BinaryDumpInFile(aFp,anObj.ModNoDist().Val());
    BinaryDumpInFile(aFp,anObj.ModRad().IsInit());
    if (anObj.ModRad().IsInit()) BinaryDumpInFile(aFp,anObj.ModRad().Val());
    BinaryDumpInFile(aFp,anObj.ModPhgrStd().IsInit());
    if (anObj.ModPhgrStd().IsInit()) BinaryDumpInFile(aFp,anObj.ModPhgrStd().Val());
    BinaryDumpInFile(aFp,anObj.ModUnif().IsInit());
    if (anObj.ModUnif().IsInit()) BinaryDumpInFile(aFp,anObj.ModUnif().Val());
    BinaryDumpInFile(aFp,anObj.ModGrid().IsInit());
    if (anObj.ModGrid().IsInit()) BinaryDumpInFile(aFp,anObj.ModGrid().Val());
    BinaryDumpInFile(aFp,anObj.ModGridDef().IsInit());
    if (anObj.ModGridDef().IsInit()) BinaryDumpInFile(aFp,anObj.ModGridDef().Val());
}

cElXMLTree * ToXMLTree(const cCalibDistortion & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"CalibDistortion",eXMLBranche);
   if (anObj.ModNoDist().IsInit())
      aRes->AddFils(ToXMLTree(anObj.ModNoDist().Val())->ReTagThis("ModNoDist"));
   if (anObj.ModRad().IsInit())
      aRes->AddFils(ToXMLTree(anObj.ModRad().Val())->ReTagThis("ModRad"));
   if (anObj.ModPhgrStd().IsInit())
      aRes->AddFils(ToXMLTree(anObj.ModPhgrStd().Val())->ReTagThis("ModPhgrStd"));
   if (anObj.ModUnif().IsInit())
      aRes->AddFils(ToXMLTree(anObj.ModUnif().Val())->ReTagThis("ModUnif"));
   if (anObj.ModGrid().IsInit())
      aRes->AddFils(ToXMLTree(anObj.ModGrid().Val())->ReTagThis("ModGrid"));
   if (anObj.ModGridDef().IsInit())
      aRes->AddFils(ToXMLTree(anObj.ModGridDef().Val())->ReTagThis("ModGridDef"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cCalibDistortion & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.ModNoDist(),aTree->Get("ModNoDist",1)); //tototo 

   xml_init(anObj.ModRad(),aTree->Get("ModRad",1)); //tototo 

   xml_init(anObj.ModPhgrStd(),aTree->Get("ModPhgrStd",1)); //tototo 

   xml_init(anObj.ModUnif(),aTree->Get("ModUnif",1)); //tototo 

   xml_init(anObj.ModGrid(),aTree->Get("ModGrid",1)); //tototo 

   xml_init(anObj.ModGridDef(),aTree->Get("ModGridDef",1)); //tototo 
}

std::string  Mangling( cCalibDistortion *) {return "A820DEEF72768781FF3F";};


std::string & cCorrectionRefractionAPosteriori::FileEstimCam()
{
   return mFileEstimCam;
}

const std::string & cCorrectionRefractionAPosteriori::FileEstimCam()const 
{
   return mFileEstimCam;
}


cTplValGesInit< std::string > & cCorrectionRefractionAPosteriori::NameTag()
{
   return mNameTag;
}

const cTplValGesInit< std::string > & cCorrectionRefractionAPosteriori::NameTag()const 
{
   return mNameTag;
}


double & cCorrectionRefractionAPosteriori::CoeffRefrac()
{
   return mCoeffRefrac;
}

const double & cCorrectionRefractionAPosteriori::CoeffRefrac()const 
{
   return mCoeffRefrac;
}


cTplValGesInit< bool > & cCorrectionRefractionAPosteriori::IntegreDist()
{
   return mIntegreDist;
}

const cTplValGesInit< bool > & cCorrectionRefractionAPosteriori::IntegreDist()const 
{
   return mIntegreDist;
}

void  BinaryUnDumpFromFile(cCorrectionRefractionAPosteriori & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.FileEstimCam(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NameTag().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NameTag().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NameTag().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.CoeffRefrac(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.IntegreDist().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.IntegreDist().ValForcedForUnUmp(),aFp);
        }
        else  anObj.IntegreDist().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cCorrectionRefractionAPosteriori & anObj)
{
    BinaryDumpInFile(aFp,anObj.FileEstimCam());
    BinaryDumpInFile(aFp,anObj.NameTag().IsInit());
    if (anObj.NameTag().IsInit()) BinaryDumpInFile(aFp,anObj.NameTag().Val());
    BinaryDumpInFile(aFp,anObj.CoeffRefrac());
    BinaryDumpInFile(aFp,anObj.IntegreDist().IsInit());
    if (anObj.IntegreDist().IsInit()) BinaryDumpInFile(aFp,anObj.IntegreDist().Val());
}

cElXMLTree * ToXMLTree(const cCorrectionRefractionAPosteriori & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"CorrectionRefractionAPosteriori",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("FileEstimCam"),anObj.FileEstimCam())->ReTagThis("FileEstimCam"));
   if (anObj.NameTag().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NameTag"),anObj.NameTag().Val())->ReTagThis("NameTag"));
   aRes->AddFils(::ToXMLTree(std::string("CoeffRefrac"),anObj.CoeffRefrac())->ReTagThis("CoeffRefrac"));
   if (anObj.IntegreDist().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("IntegreDist"),anObj.IntegreDist().Val())->ReTagThis("IntegreDist"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cCorrectionRefractionAPosteriori & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.FileEstimCam(),aTree->Get("FileEstimCam",1)); //tototo 

   xml_init(anObj.NameTag(),aTree->Get("NameTag",1),std::string("CalibrationInternConique")); //tototo 

   xml_init(anObj.CoeffRefrac(),aTree->Get("CoeffRefrac",1)); //tototo 

   xml_init(anObj.IntegreDist(),aTree->Get("IntegreDist",1),bool(false)); //tototo 
}

std::string  Mangling( cCorrectionRefractionAPosteriori *) {return "400C45B31B4A1CD1FC3F";};


cTplValGesInit< eConventionsOrientation > & cCalibrationInternConique::KnownConv()
{
   return mKnownConv;
}

const cTplValGesInit< eConventionsOrientation > & cCalibrationInternConique::KnownConv()const 
{
   return mKnownConv;
}


std::vector< double > & cCalibrationInternConique::ParamAF()
{
   return mParamAF;
}

const std::vector< double > & cCalibrationInternConique::ParamAF()const 
{
   return mParamAF;
}


Pt2dr & cCalibrationInternConique::PP()
{
   return mPP;
}

const Pt2dr & cCalibrationInternConique::PP()const 
{
   return mPP;
}


double & cCalibrationInternConique::F()
{
   return mF;
}

const double & cCalibrationInternConique::F()const 
{
   return mF;
}


Pt2di & cCalibrationInternConique::SzIm()
{
   return mSzIm;
}

const Pt2di & cCalibrationInternConique::SzIm()const 
{
   return mSzIm;
}


cTplValGesInit< Pt2dr > & cCalibrationInternConique::PixelSzIm()
{
   return mPixelSzIm;
}

const cTplValGesInit< Pt2dr > & cCalibrationInternConique::PixelSzIm()const 
{
   return mPixelSzIm;
}


cTplValGesInit< double > & cCalibrationInternConique::RayonUtile()
{
   return mRayonUtile;
}

const cTplValGesInit< double > & cCalibrationInternConique::RayonUtile()const 
{
   return mRayonUtile;
}


std::vector< bool > & cCalibrationInternConique::ComplIsC2M()
{
   return mComplIsC2M;
}

const std::vector< bool > & cCalibrationInternConique::ComplIsC2M()const 
{
   return mComplIsC2M;
}


cTplValGesInit< bool > & cCalibrationInternConique::ScannedAnalogik()
{
   return mScannedAnalogik;
}

const cTplValGesInit< bool > & cCalibrationInternConique::ScannedAnalogik()const 
{
   return mScannedAnalogik;
}


cAffinitePlane & cCalibrationInternConique::Affinite()
{
   return OrIntGlob().Val().Affinite();
}

const cAffinitePlane & cCalibrationInternConique::Affinite()const 
{
   return OrIntGlob().Val().Affinite();
}


bool & cCalibrationInternConique::C2M()
{
   return OrIntGlob().Val().C2M();
}

const bool & cCalibrationInternConique::C2M()const 
{
   return OrIntGlob().Val().C2M();
}


cTplValGesInit< cOrIntGlob > & cCalibrationInternConique::OrIntGlob()
{
   return mOrIntGlob;
}

const cTplValGesInit< cOrIntGlob > & cCalibrationInternConique::OrIntGlob()const 
{
   return mOrIntGlob;
}


Pt2dr & cCalibrationInternConique::StepGrid()
{
   return ParamForGrid().Val().StepGrid();
}

const Pt2dr & cCalibrationInternConique::StepGrid()const 
{
   return ParamForGrid().Val().StepGrid();
}


double & cCalibrationInternConique::RayonInv()
{
   return ParamForGrid().Val().RayonInv();
}

const double & cCalibrationInternConique::RayonInv()const 
{
   return ParamForGrid().Val().RayonInv();
}


cTplValGesInit< cParamForGrid > & cCalibrationInternConique::ParamForGrid()
{
   return mParamForGrid;
}

const cTplValGesInit< cParamForGrid > & cCalibrationInternConique::ParamForGrid()const 
{
   return mParamForGrid;
}


std::vector< cCalibDistortion > & cCalibrationInternConique::CalibDistortion()
{
   return mCalibDistortion;
}

const std::vector< cCalibDistortion > & cCalibrationInternConique::CalibDistortion()const 
{
   return mCalibDistortion;
}


std::string & cCalibrationInternConique::FileEstimCam()
{
   return CorrectionRefractionAPosteriori().Val().FileEstimCam();
}

const std::string & cCalibrationInternConique::FileEstimCam()const 
{
   return CorrectionRefractionAPosteriori().Val().FileEstimCam();
}


cTplValGesInit< std::string > & cCalibrationInternConique::NameTag()
{
   return CorrectionRefractionAPosteriori().Val().NameTag();
}

const cTplValGesInit< std::string > & cCalibrationInternConique::NameTag()const 
{
   return CorrectionRefractionAPosteriori().Val().NameTag();
}


double & cCalibrationInternConique::CoeffRefrac()
{
   return CorrectionRefractionAPosteriori().Val().CoeffRefrac();
}

const double & cCalibrationInternConique::CoeffRefrac()const 
{
   return CorrectionRefractionAPosteriori().Val().CoeffRefrac();
}


cTplValGesInit< bool > & cCalibrationInternConique::IntegreDist()
{
   return CorrectionRefractionAPosteriori().Val().IntegreDist();
}

const cTplValGesInit< bool > & cCalibrationInternConique::IntegreDist()const 
{
   return CorrectionRefractionAPosteriori().Val().IntegreDist();
}


cTplValGesInit< cCorrectionRefractionAPosteriori > & cCalibrationInternConique::CorrectionRefractionAPosteriori()
{
   return mCorrectionRefractionAPosteriori;
}

const cTplValGesInit< cCorrectionRefractionAPosteriori > & cCalibrationInternConique::CorrectionRefractionAPosteriori()const 
{
   return mCorrectionRefractionAPosteriori;
}

void  BinaryUnDumpFromFile(cCalibrationInternConique & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.KnownConv().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.KnownConv().ValForcedForUnUmp(),aFp);
        }
        else  anObj.KnownConv().SetNoInit();
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             double aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.ParamAF().push_back(aVal);
        }
  } ;
    BinaryUnDumpFromFile(anObj.PP(),aFp);
    BinaryUnDumpFromFile(anObj.F(),aFp);
    BinaryUnDumpFromFile(anObj.SzIm(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.PixelSzIm().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.PixelSzIm().ValForcedForUnUmp(),aFp);
        }
        else  anObj.PixelSzIm().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.RayonUtile().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.RayonUtile().ValForcedForUnUmp(),aFp);
        }
        else  anObj.RayonUtile().SetNoInit();
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             bool aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.ComplIsC2M().push_back(aVal);
        }
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ScannedAnalogik().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ScannedAnalogik().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ScannedAnalogik().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.OrIntGlob().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.OrIntGlob().ValForcedForUnUmp(),aFp);
        }
        else  anObj.OrIntGlob().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ParamForGrid().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ParamForGrid().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ParamForGrid().SetNoInit();
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cCalibDistortion aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.CalibDistortion().push_back(aVal);
        }
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.CorrectionRefractionAPosteriori().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.CorrectionRefractionAPosteriori().ValForcedForUnUmp(),aFp);
        }
        else  anObj.CorrectionRefractionAPosteriori().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cCalibrationInternConique & anObj)
{
    BinaryDumpInFile(aFp,anObj.KnownConv().IsInit());
    if (anObj.KnownConv().IsInit()) BinaryDumpInFile(aFp,anObj.KnownConv().Val());
    BinaryDumpInFile(aFp,(int)anObj.ParamAF().size());
    for(  std::vector< double >::const_iterator iT=anObj.ParamAF().begin();
         iT!=anObj.ParamAF().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.PP());
    BinaryDumpInFile(aFp,anObj.F());
    BinaryDumpInFile(aFp,anObj.SzIm());
    BinaryDumpInFile(aFp,anObj.PixelSzIm().IsInit());
    if (anObj.PixelSzIm().IsInit()) BinaryDumpInFile(aFp,anObj.PixelSzIm().Val());
    BinaryDumpInFile(aFp,anObj.RayonUtile().IsInit());
    if (anObj.RayonUtile().IsInit()) BinaryDumpInFile(aFp,anObj.RayonUtile().Val());
    BinaryDumpInFile(aFp,(int)anObj.ComplIsC2M().size());
    for(  std::vector< bool >::const_iterator iT=anObj.ComplIsC2M().begin();
         iT!=anObj.ComplIsC2M().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.ScannedAnalogik().IsInit());
    if (anObj.ScannedAnalogik().IsInit()) BinaryDumpInFile(aFp,anObj.ScannedAnalogik().Val());
    BinaryDumpInFile(aFp,anObj.OrIntGlob().IsInit());
    if (anObj.OrIntGlob().IsInit()) BinaryDumpInFile(aFp,anObj.OrIntGlob().Val());
    BinaryDumpInFile(aFp,anObj.ParamForGrid().IsInit());
    if (anObj.ParamForGrid().IsInit()) BinaryDumpInFile(aFp,anObj.ParamForGrid().Val());
    BinaryDumpInFile(aFp,(int)anObj.CalibDistortion().size());
    for(  std::vector< cCalibDistortion >::const_iterator iT=anObj.CalibDistortion().begin();
         iT!=anObj.CalibDistortion().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.CorrectionRefractionAPosteriori().IsInit());
    if (anObj.CorrectionRefractionAPosteriori().IsInit()) BinaryDumpInFile(aFp,anObj.CorrectionRefractionAPosteriori().Val());
}

cElXMLTree * ToXMLTree(const cCalibrationInternConique & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"CalibrationInternConique",eXMLBranche);
   if (anObj.KnownConv().IsInit())
      aRes->AddFils(ToXMLTree(std::string("KnownConv"),anObj.KnownConv().Val())->ReTagThis("KnownConv"));
  for
  (       std::vector< double >::const_iterator it=anObj.ParamAF().begin();
      it !=anObj.ParamAF().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("ParamAF"),(*it))->ReTagThis("ParamAF"));
   aRes->AddFils(::ToXMLTree(std::string("PP"),anObj.PP())->ReTagThis("PP"));
   aRes->AddFils(::ToXMLTree(std::string("F"),anObj.F())->ReTagThis("F"));
   aRes->AddFils(::ToXMLTree(std::string("SzIm"),anObj.SzIm())->ReTagThis("SzIm"));
   if (anObj.PixelSzIm().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PixelSzIm"),anObj.PixelSzIm().Val())->ReTagThis("PixelSzIm"));
   if (anObj.RayonUtile().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("RayonUtile"),anObj.RayonUtile().Val())->ReTagThis("RayonUtile"));
  for
  (       std::vector< bool >::const_iterator it=anObj.ComplIsC2M().begin();
      it !=anObj.ComplIsC2M().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("ComplIsC2M"),(*it))->ReTagThis("ComplIsC2M"));
   if (anObj.ScannedAnalogik().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ScannedAnalogik"),anObj.ScannedAnalogik().Val())->ReTagThis("ScannedAnalogik"));
   if (anObj.OrIntGlob().IsInit())
      aRes->AddFils(ToXMLTree(anObj.OrIntGlob().Val())->ReTagThis("OrIntGlob"));
   if (anObj.ParamForGrid().IsInit())
      aRes->AddFils(ToXMLTree(anObj.ParamForGrid().Val())->ReTagThis("ParamForGrid"));
  for
  (       std::vector< cCalibDistortion >::const_iterator it=anObj.CalibDistortion().begin();
      it !=anObj.CalibDistortion().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("CalibDistortion"));
   if (anObj.CorrectionRefractionAPosteriori().IsInit())
      aRes->AddFils(ToXMLTree(anObj.CorrectionRefractionAPosteriori().Val())->ReTagThis("CorrectionRefractionAPosteriori"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cCalibrationInternConique & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.KnownConv(),aTree->Get("KnownConv",1)); //tototo 

   xml_init(anObj.ParamAF(),aTree->GetAll("ParamAF",false,1));

   xml_init(anObj.PP(),aTree->Get("PP",1)); //tototo 

   xml_init(anObj.F(),aTree->Get("F",1)); //tototo 

   xml_init(anObj.SzIm(),aTree->Get("SzIm",1)); //tototo 

   xml_init(anObj.PixelSzIm(),aTree->Get("PixelSzIm",1)); //tototo 

   xml_init(anObj.RayonUtile(),aTree->Get("RayonUtile",1)); //tototo 

   xml_init(anObj.ComplIsC2M(),aTree->GetAll("ComplIsC2M",false,1));

   xml_init(anObj.ScannedAnalogik(),aTree->Get("ScannedAnalogik",1),bool(false)); //tototo 

   xml_init(anObj.OrIntGlob(),aTree->Get("OrIntGlob",1)); //tototo 

   xml_init(anObj.ParamForGrid(),aTree->Get("ParamForGrid",1)); //tototo 

   xml_init(anObj.CalibDistortion(),aTree->GetAll("CalibDistortion",false,1));

   xml_init(anObj.CorrectionRefractionAPosteriori(),aTree->Get("CorrectionRefractionAPosteriori",1)); //tototo 
}

std::string  Mangling( cCalibrationInternConique *) {return "D48EFFE141B11DD4FE3F";};


Pt3dr & cRepereCartesien::Ori()
{
   return mOri;
}

const Pt3dr & cRepereCartesien::Ori()const 
{
   return mOri;
}


Pt3dr & cRepereCartesien::Ox()
{
   return mOx;
}

const Pt3dr & cRepereCartesien::Ox()const 
{
   return mOx;
}


Pt3dr & cRepereCartesien::Oy()
{
   return mOy;
}

const Pt3dr & cRepereCartesien::Oy()const 
{
   return mOy;
}


Pt3dr & cRepereCartesien::Oz()
{
   return mOz;
}

const Pt3dr & cRepereCartesien::Oz()const 
{
   return mOz;
}

void  BinaryUnDumpFromFile(cRepereCartesien & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Ori(),aFp);
    BinaryUnDumpFromFile(anObj.Ox(),aFp);
    BinaryUnDumpFromFile(anObj.Oy(),aFp);
    BinaryUnDumpFromFile(anObj.Oz(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cRepereCartesien & anObj)
{
    BinaryDumpInFile(aFp,anObj.Ori());
    BinaryDumpInFile(aFp,anObj.Ox());
    BinaryDumpInFile(aFp,anObj.Oy());
    BinaryDumpInFile(aFp,anObj.Oz());
}

cElXMLTree * ToXMLTree(const cRepereCartesien & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"RepereCartesien",eXMLBranche);
   aRes->AddFils(ToXMLTree(std::string("Ori"),anObj.Ori())->ReTagThis("Ori"));
   aRes->AddFils(ToXMLTree(std::string("Ox"),anObj.Ox())->ReTagThis("Ox"));
   aRes->AddFils(ToXMLTree(std::string("Oy"),anObj.Oy())->ReTagThis("Oy"));
   aRes->AddFils(ToXMLTree(std::string("Oz"),anObj.Oz())->ReTagThis("Oz"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cRepereCartesien & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Ori(),aTree->Get("Ori",1)); //tototo 

   xml_init(anObj.Ox(),aTree->Get("Ox",1)); //tototo 

   xml_init(anObj.Oy(),aTree->Get("Oy",1)); //tototo 

   xml_init(anObj.Oz(),aTree->Get("Oz",1)); //tototo 
}

std::string  Mangling( cRepereCartesien *) {return "66FDF259A34C1688FD3F";};


Pt3dr & cTypeCodageMatr::L1()
{
   return mL1;
}

const Pt3dr & cTypeCodageMatr::L1()const 
{
   return mL1;
}


Pt3dr & cTypeCodageMatr::L2()
{
   return mL2;
}

const Pt3dr & cTypeCodageMatr::L2()const 
{
   return mL2;
}


Pt3dr & cTypeCodageMatr::L3()
{
   return mL3;
}

const Pt3dr & cTypeCodageMatr::L3()const 
{
   return mL3;
}


cTplValGesInit< bool > & cTypeCodageMatr::TrueRot()
{
   return mTrueRot;
}

const cTplValGesInit< bool > & cTypeCodageMatr::TrueRot()const 
{
   return mTrueRot;
}

void  BinaryUnDumpFromFile(cTypeCodageMatr & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.L1(),aFp);
    BinaryUnDumpFromFile(anObj.L2(),aFp);
    BinaryUnDumpFromFile(anObj.L3(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.TrueRot().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.TrueRot().ValForcedForUnUmp(),aFp);
        }
        else  anObj.TrueRot().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cTypeCodageMatr & anObj)
{
    BinaryDumpInFile(aFp,anObj.L1());
    BinaryDumpInFile(aFp,anObj.L2());
    BinaryDumpInFile(aFp,anObj.L3());
    BinaryDumpInFile(aFp,anObj.TrueRot().IsInit());
    if (anObj.TrueRot().IsInit()) BinaryDumpInFile(aFp,anObj.TrueRot().Val());
}

cElXMLTree * ToXMLTree(const cTypeCodageMatr & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"TypeCodageMatr",eXMLBranche);
   aRes->AddFils(ToXMLTree(std::string("L1"),anObj.L1())->ReTagThis("L1"));
   aRes->AddFils(ToXMLTree(std::string("L2"),anObj.L2())->ReTagThis("L2"));
   aRes->AddFils(ToXMLTree(std::string("L3"),anObj.L3())->ReTagThis("L3"));
   if (anObj.TrueRot().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("TrueRot"),anObj.TrueRot().Val())->ReTagThis("TrueRot"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cTypeCodageMatr & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.L1(),aTree->Get("L1",1)); //tototo 

   xml_init(anObj.L2(),aTree->Get("L2",1)); //tototo 

   xml_init(anObj.L3(),aTree->Get("L3",1)); //tototo 

   xml_init(anObj.TrueRot(),aTree->Get("TrueRot",1),bool(true)); //tototo 
}

std::string  Mangling( cTypeCodageMatr *) {return "80D9B4E09D4A9A92FF3F";};


cTplValGesInit< cTypeCodageMatr > & cRotationVect::CodageMatr()
{
   return mCodageMatr;
}

const cTplValGesInit< cTypeCodageMatr > & cRotationVect::CodageMatr()const 
{
   return mCodageMatr;
}


cTplValGesInit< Pt3dr > & cRotationVect::CodageAngulaire()
{
   return mCodageAngulaire;
}

const cTplValGesInit< Pt3dr > & cRotationVect::CodageAngulaire()const 
{
   return mCodageAngulaire;
}


cTplValGesInit< std::string > & cRotationVect::CodageSymbolique()
{
   return mCodageSymbolique;
}

const cTplValGesInit< std::string > & cRotationVect::CodageSymbolique()const 
{
   return mCodageSymbolique;
}

void  BinaryUnDumpFromFile(cRotationVect & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.CodageMatr().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.CodageMatr().ValForcedForUnUmp(),aFp);
        }
        else  anObj.CodageMatr().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.CodageAngulaire().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.CodageAngulaire().ValForcedForUnUmp(),aFp);
        }
        else  anObj.CodageAngulaire().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.CodageSymbolique().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.CodageSymbolique().ValForcedForUnUmp(),aFp);
        }
        else  anObj.CodageSymbolique().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cRotationVect & anObj)
{
    BinaryDumpInFile(aFp,anObj.CodageMatr().IsInit());
    if (anObj.CodageMatr().IsInit()) BinaryDumpInFile(aFp,anObj.CodageMatr().Val());
    BinaryDumpInFile(aFp,anObj.CodageAngulaire().IsInit());
    if (anObj.CodageAngulaire().IsInit()) BinaryDumpInFile(aFp,anObj.CodageAngulaire().Val());
    BinaryDumpInFile(aFp,anObj.CodageSymbolique().IsInit());
    if (anObj.CodageSymbolique().IsInit()) BinaryDumpInFile(aFp,anObj.CodageSymbolique().Val());
}

cElXMLTree * ToXMLTree(const cRotationVect & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"RotationVect",eXMLBranche);
   if (anObj.CodageMatr().IsInit())
      aRes->AddFils(ToXMLTree(anObj.CodageMatr().Val())->ReTagThis("CodageMatr"));
   if (anObj.CodageAngulaire().IsInit())
      aRes->AddFils(ToXMLTree(std::string("CodageAngulaire"),anObj.CodageAngulaire().Val())->ReTagThis("CodageAngulaire"));
   if (anObj.CodageSymbolique().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("CodageSymbolique"),anObj.CodageSymbolique().Val())->ReTagThis("CodageSymbolique"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cRotationVect & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.CodageMatr(),aTree->Get("CodageMatr",1)); //tototo 

   xml_init(anObj.CodageAngulaire(),aTree->Get("CodageAngulaire",1)); //tototo 

   xml_init(anObj.CodageSymbolique(),aTree->Get("CodageSymbolique",1)); //tototo 
}

std::string  Mangling( cRotationVect *) {return "610DC83CDDDCB987FF3F";};


cTplValGesInit< double > & cOrientationExterneRigide::AltiSol()
{
   return mAltiSol;
}

const cTplValGesInit< double > & cOrientationExterneRigide::AltiSol()const 
{
   return mAltiSol;
}


cTplValGesInit< double > & cOrientationExterneRigide::Profondeur()
{
   return mProfondeur;
}

const cTplValGesInit< double > & cOrientationExterneRigide::Profondeur()const 
{
   return mProfondeur;
}


cTplValGesInit< double > & cOrientationExterneRigide::Time()
{
   return mTime;
}

const cTplValGesInit< double > & cOrientationExterneRigide::Time()const 
{
   return mTime;
}


cTplValGesInit< eConventionsOrientation > & cOrientationExterneRigide::KnownConv()
{
   return mKnownConv;
}

const cTplValGesInit< eConventionsOrientation > & cOrientationExterneRigide::KnownConv()const 
{
   return mKnownConv;
}


Pt3dr & cOrientationExterneRigide::Centre()
{
   return mCentre;
}

const Pt3dr & cOrientationExterneRigide::Centre()const 
{
   return mCentre;
}


cTplValGesInit< Pt3dr > & cOrientationExterneRigide::Vitesse()
{
   return mVitesse;
}

const cTplValGesInit< Pt3dr > & cOrientationExterneRigide::Vitesse()const 
{
   return mVitesse;
}


cTplValGesInit< bool > & cOrientationExterneRigide::VitesseFiable()
{
   return mVitesseFiable;
}

const cTplValGesInit< bool > & cOrientationExterneRigide::VitesseFiable()const 
{
   return mVitesseFiable;
}


cTplValGesInit< Pt3dr > & cOrientationExterneRigide::IncCentre()
{
   return mIncCentre;
}

const cTplValGesInit< Pt3dr > & cOrientationExterneRigide::IncCentre()const 
{
   return mIncCentre;
}


cRotationVect & cOrientationExterneRigide::ParamRotation()
{
   return mParamRotation;
}

const cRotationVect & cOrientationExterneRigide::ParamRotation()const 
{
   return mParamRotation;
}

void  BinaryUnDumpFromFile(cOrientationExterneRigide & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.AltiSol().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.AltiSol().ValForcedForUnUmp(),aFp);
        }
        else  anObj.AltiSol().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Profondeur().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Profondeur().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Profondeur().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Time().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Time().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Time().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.KnownConv().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.KnownConv().ValForcedForUnUmp(),aFp);
        }
        else  anObj.KnownConv().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.Centre(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Vitesse().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Vitesse().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Vitesse().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.VitesseFiable().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.VitesseFiable().ValForcedForUnUmp(),aFp);
        }
        else  anObj.VitesseFiable().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.IncCentre().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.IncCentre().ValForcedForUnUmp(),aFp);
        }
        else  anObj.IncCentre().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.ParamRotation(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cOrientationExterneRigide & anObj)
{
    BinaryDumpInFile(aFp,anObj.AltiSol().IsInit());
    if (anObj.AltiSol().IsInit()) BinaryDumpInFile(aFp,anObj.AltiSol().Val());
    BinaryDumpInFile(aFp,anObj.Profondeur().IsInit());
    if (anObj.Profondeur().IsInit()) BinaryDumpInFile(aFp,anObj.Profondeur().Val());
    BinaryDumpInFile(aFp,anObj.Time().IsInit());
    if (anObj.Time().IsInit()) BinaryDumpInFile(aFp,anObj.Time().Val());
    BinaryDumpInFile(aFp,anObj.KnownConv().IsInit());
    if (anObj.KnownConv().IsInit()) BinaryDumpInFile(aFp,anObj.KnownConv().Val());
    BinaryDumpInFile(aFp,anObj.Centre());
    BinaryDumpInFile(aFp,anObj.Vitesse().IsInit());
    if (anObj.Vitesse().IsInit()) BinaryDumpInFile(aFp,anObj.Vitesse().Val());
    BinaryDumpInFile(aFp,anObj.VitesseFiable().IsInit());
    if (anObj.VitesseFiable().IsInit()) BinaryDumpInFile(aFp,anObj.VitesseFiable().Val());
    BinaryDumpInFile(aFp,anObj.IncCentre().IsInit());
    if (anObj.IncCentre().IsInit()) BinaryDumpInFile(aFp,anObj.IncCentre().Val());
    BinaryDumpInFile(aFp,anObj.ParamRotation());
}

cElXMLTree * ToXMLTree(const cOrientationExterneRigide & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"OrientationExterneRigide",eXMLBranche);
   if (anObj.AltiSol().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("AltiSol"),anObj.AltiSol().Val())->ReTagThis("AltiSol"));
   if (anObj.Profondeur().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Profondeur"),anObj.Profondeur().Val())->ReTagThis("Profondeur"));
   if (anObj.Time().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Time"),anObj.Time().Val())->ReTagThis("Time"));
   if (anObj.KnownConv().IsInit())
      aRes->AddFils(ToXMLTree(std::string("KnownConv"),anObj.KnownConv().Val())->ReTagThis("KnownConv"));
   aRes->AddFils(ToXMLTree(std::string("Centre"),anObj.Centre())->ReTagThis("Centre"));
   if (anObj.Vitesse().IsInit())
      aRes->AddFils(ToXMLTree(std::string("Vitesse"),anObj.Vitesse().Val())->ReTagThis("Vitesse"));
   if (anObj.VitesseFiable().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("VitesseFiable"),anObj.VitesseFiable().Val())->ReTagThis("VitesseFiable"));
   if (anObj.IncCentre().IsInit())
      aRes->AddFils(ToXMLTree(std::string("IncCentre"),anObj.IncCentre().Val())->ReTagThis("IncCentre"));
   aRes->AddFils(ToXMLTree(anObj.ParamRotation())->ReTagThis("ParamRotation"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cOrientationExterneRigide & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.AltiSol(),aTree->Get("AltiSol",1)); //tototo 

   xml_init(anObj.Profondeur(),aTree->Get("Profondeur",1)); //tototo 

   xml_init(anObj.Time(),aTree->Get("Time",1),double(0.0)); //tototo 

   xml_init(anObj.KnownConv(),aTree->Get("KnownConv",1)); //tototo 

   xml_init(anObj.Centre(),aTree->Get("Centre",1)); //tototo 

   xml_init(anObj.Vitesse(),aTree->Get("Vitesse",1)); //tototo 

   xml_init(anObj.VitesseFiable(),aTree->Get("VitesseFiable",1),bool(true)); //tototo 

   xml_init(anObj.IncCentre(),aTree->Get("IncCentre",1)); //tototo 

   xml_init(anObj.ParamRotation(),aTree->Get("ParamRotation",1)); //tototo 
}

std::string  Mangling( cOrientationExterneRigide *) {return "72095403899D1BE4FE3F";};


std::string & cModuleOrientationFile::NameFileOri()
{
   return mNameFileOri;
}

const std::string & cModuleOrientationFile::NameFileOri()const 
{
   return mNameFileOri;
}

void  BinaryUnDumpFromFile(cModuleOrientationFile & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.NameFileOri(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cModuleOrientationFile & anObj)
{
    BinaryDumpInFile(aFp,anObj.NameFileOri());
}

cElXMLTree * ToXMLTree(const cModuleOrientationFile & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ModuleOrientationFile",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("NameFileOri"),anObj.NameFileOri())->ReTagThis("NameFileOri"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cModuleOrientationFile & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.NameFileOri(),aTree->Get("NameFileOri",1)); //tototo 
}

std::string  Mangling( cModuleOrientationFile *) {return "C0AD0881D8B0D8A7FE3F";};


cTplValGesInit< bool > & cConvExplicite::SensYVideo()
{
   return mSensYVideo;
}

const cTplValGesInit< bool > & cConvExplicite::SensYVideo()const 
{
   return mSensYVideo;
}


cTplValGesInit< bool > & cConvExplicite::DistSenC2M()
{
   return mDistSenC2M;
}

const cTplValGesInit< bool > & cConvExplicite::DistSenC2M()const 
{
   return mDistSenC2M;
}


cTplValGesInit< bool > & cConvExplicite::MatrSenC2M()
{
   return mMatrSenC2M;
}

const cTplValGesInit< bool > & cConvExplicite::MatrSenC2M()const 
{
   return mMatrSenC2M;
}


cTplValGesInit< Pt3dr > & cConvExplicite::ColMul()
{
   return mColMul;
}

const cTplValGesInit< Pt3dr > & cConvExplicite::ColMul()const 
{
   return mColMul;
}


cTplValGesInit< Pt3dr > & cConvExplicite::LigMul()
{
   return mLigMul;
}

const cTplValGesInit< Pt3dr > & cConvExplicite::LigMul()const 
{
   return mLigMul;
}


cTplValGesInit< eUniteAngulaire > & cConvExplicite::UniteAngles()
{
   return mUniteAngles;
}

const cTplValGesInit< eUniteAngulaire > & cConvExplicite::UniteAngles()const 
{
   return mUniteAngles;
}


cTplValGesInit< Pt3di > & cConvExplicite::NumAxe()
{
   return mNumAxe;
}

const cTplValGesInit< Pt3di > & cConvExplicite::NumAxe()const 
{
   return mNumAxe;
}


cTplValGesInit< bool > & cConvExplicite::SensCardan()
{
   return mSensCardan;
}

const cTplValGesInit< bool > & cConvExplicite::SensCardan()const 
{
   return mSensCardan;
}


cTplValGesInit< eConventionsOrientation > & cConvExplicite::Convention()
{
   return mConvention;
}

const cTplValGesInit< eConventionsOrientation > & cConvExplicite::Convention()const 
{
   return mConvention;
}

void  BinaryUnDumpFromFile(cConvExplicite & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SensYVideo().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SensYVideo().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SensYVideo().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.DistSenC2M().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.DistSenC2M().ValForcedForUnUmp(),aFp);
        }
        else  anObj.DistSenC2M().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.MatrSenC2M().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.MatrSenC2M().ValForcedForUnUmp(),aFp);
        }
        else  anObj.MatrSenC2M().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ColMul().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ColMul().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ColMul().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.LigMul().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.LigMul().ValForcedForUnUmp(),aFp);
        }
        else  anObj.LigMul().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.UniteAngles().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.UniteAngles().ValForcedForUnUmp(),aFp);
        }
        else  anObj.UniteAngles().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NumAxe().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NumAxe().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NumAxe().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SensCardan().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SensCardan().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SensCardan().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Convention().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Convention().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Convention().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cConvExplicite & anObj)
{
    BinaryDumpInFile(aFp,anObj.SensYVideo().IsInit());
    if (anObj.SensYVideo().IsInit()) BinaryDumpInFile(aFp,anObj.SensYVideo().Val());
    BinaryDumpInFile(aFp,anObj.DistSenC2M().IsInit());
    if (anObj.DistSenC2M().IsInit()) BinaryDumpInFile(aFp,anObj.DistSenC2M().Val());
    BinaryDumpInFile(aFp,anObj.MatrSenC2M().IsInit());
    if (anObj.MatrSenC2M().IsInit()) BinaryDumpInFile(aFp,anObj.MatrSenC2M().Val());
    BinaryDumpInFile(aFp,anObj.ColMul().IsInit());
    if (anObj.ColMul().IsInit()) BinaryDumpInFile(aFp,anObj.ColMul().Val());
    BinaryDumpInFile(aFp,anObj.LigMul().IsInit());
    if (anObj.LigMul().IsInit()) BinaryDumpInFile(aFp,anObj.LigMul().Val());
    BinaryDumpInFile(aFp,anObj.UniteAngles().IsInit());
    if (anObj.UniteAngles().IsInit()) BinaryDumpInFile(aFp,anObj.UniteAngles().Val());
    BinaryDumpInFile(aFp,anObj.NumAxe().IsInit());
    if (anObj.NumAxe().IsInit()) BinaryDumpInFile(aFp,anObj.NumAxe().Val());
    BinaryDumpInFile(aFp,anObj.SensCardan().IsInit());
    if (anObj.SensCardan().IsInit()) BinaryDumpInFile(aFp,anObj.SensCardan().Val());
    BinaryDumpInFile(aFp,anObj.Convention().IsInit());
    if (anObj.Convention().IsInit()) BinaryDumpInFile(aFp,anObj.Convention().Val());
}

cElXMLTree * ToXMLTree(const cConvExplicite & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ConvExplicite",eXMLBranche);
   if (anObj.SensYVideo().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SensYVideo"),anObj.SensYVideo().Val())->ReTagThis("SensYVideo"));
   if (anObj.DistSenC2M().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DistSenC2M"),anObj.DistSenC2M().Val())->ReTagThis("DistSenC2M"));
   if (anObj.MatrSenC2M().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("MatrSenC2M"),anObj.MatrSenC2M().Val())->ReTagThis("MatrSenC2M"));
   if (anObj.ColMul().IsInit())
      aRes->AddFils(ToXMLTree(std::string("ColMul"),anObj.ColMul().Val())->ReTagThis("ColMul"));
   if (anObj.LigMul().IsInit())
      aRes->AddFils(ToXMLTree(std::string("LigMul"),anObj.LigMul().Val())->ReTagThis("LigMul"));
   if (anObj.UniteAngles().IsInit())
      aRes->AddFils(ToXMLTree(std::string("UniteAngles"),anObj.UniteAngles().Val())->ReTagThis("UniteAngles"));
   if (anObj.NumAxe().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NumAxe"),anObj.NumAxe().Val())->ReTagThis("NumAxe"));
   if (anObj.SensCardan().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SensCardan"),anObj.SensCardan().Val())->ReTagThis("SensCardan"));
   if (anObj.Convention().IsInit())
      aRes->AddFils(ToXMLTree(std::string("Convention"),anObj.Convention().Val())->ReTagThis("Convention"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cConvExplicite & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.SensYVideo(),aTree->Get("SensYVideo",1),bool(true)); //tototo 

   xml_init(anObj.DistSenC2M(),aTree->Get("DistSenC2M",1),bool(false)); //tototo 

   xml_init(anObj.MatrSenC2M(),aTree->Get("MatrSenC2M",1),bool(true)); //tototo 

   xml_init(anObj.ColMul(),aTree->Get("ColMul",1),Pt3dr(Pt3dr(1,1,1))); //tototo 

   xml_init(anObj.LigMul(),aTree->Get("LigMul",1),Pt3dr(Pt3dr(1,1,1))); //tototo 

   xml_init(anObj.UniteAngles(),aTree->Get("UniteAngles",1),eUniteAngulaire(eUniteAngleDegre)); //tototo 

   xml_init(anObj.NumAxe(),aTree->Get("NumAxe",1),Pt3di(Pt3di(2,1,0))); //tototo 

   xml_init(anObj.SensCardan(),aTree->Get("SensCardan",1),bool(true)); //tototo 

   xml_init(anObj.Convention(),aTree->Get("Convention",1),eConventionsOrientation(eConvInconnue)); //tototo 
}

std::string  Mangling( cConvExplicite *) {return "E5B77365316BA88BFE3F";};


cTplValGesInit< eConventionsOrientation > & cConvOri::KnownConv()
{
   return mKnownConv;
}

const cTplValGesInit< eConventionsOrientation > & cConvOri::KnownConv()const 
{
   return mKnownConv;
}


cTplValGesInit< bool > & cConvOri::SensYVideo()
{
   return ConvExplicite().Val().SensYVideo();
}

const cTplValGesInit< bool > & cConvOri::SensYVideo()const 
{
   return ConvExplicite().Val().SensYVideo();
}


cTplValGesInit< bool > & cConvOri::DistSenC2M()
{
   return ConvExplicite().Val().DistSenC2M();
}

const cTplValGesInit< bool > & cConvOri::DistSenC2M()const 
{
   return ConvExplicite().Val().DistSenC2M();
}


cTplValGesInit< bool > & cConvOri::MatrSenC2M()
{
   return ConvExplicite().Val().MatrSenC2M();
}

const cTplValGesInit< bool > & cConvOri::MatrSenC2M()const 
{
   return ConvExplicite().Val().MatrSenC2M();
}


cTplValGesInit< Pt3dr > & cConvOri::ColMul()
{
   return ConvExplicite().Val().ColMul();
}

const cTplValGesInit< Pt3dr > & cConvOri::ColMul()const 
{
   return ConvExplicite().Val().ColMul();
}


cTplValGesInit< Pt3dr > & cConvOri::LigMul()
{
   return ConvExplicite().Val().LigMul();
}

const cTplValGesInit< Pt3dr > & cConvOri::LigMul()const 
{
   return ConvExplicite().Val().LigMul();
}


cTplValGesInit< eUniteAngulaire > & cConvOri::UniteAngles()
{
   return ConvExplicite().Val().UniteAngles();
}

const cTplValGesInit< eUniteAngulaire > & cConvOri::UniteAngles()const 
{
   return ConvExplicite().Val().UniteAngles();
}


cTplValGesInit< Pt3di > & cConvOri::NumAxe()
{
   return ConvExplicite().Val().NumAxe();
}

const cTplValGesInit< Pt3di > & cConvOri::NumAxe()const 
{
   return ConvExplicite().Val().NumAxe();
}


cTplValGesInit< bool > & cConvOri::SensCardan()
{
   return ConvExplicite().Val().SensCardan();
}

const cTplValGesInit< bool > & cConvOri::SensCardan()const 
{
   return ConvExplicite().Val().SensCardan();
}


cTplValGesInit< eConventionsOrientation > & cConvOri::Convention()
{
   return ConvExplicite().Val().Convention();
}

const cTplValGesInit< eConventionsOrientation > & cConvOri::Convention()const 
{
   return ConvExplicite().Val().Convention();
}


cTplValGesInit< cConvExplicite > & cConvOri::ConvExplicite()
{
   return mConvExplicite;
}

const cTplValGesInit< cConvExplicite > & cConvOri::ConvExplicite()const 
{
   return mConvExplicite;
}

void  BinaryUnDumpFromFile(cConvOri & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.KnownConv().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.KnownConv().ValForcedForUnUmp(),aFp);
        }
        else  anObj.KnownConv().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ConvExplicite().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ConvExplicite().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ConvExplicite().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cConvOri & anObj)
{
    BinaryDumpInFile(aFp,anObj.KnownConv().IsInit());
    if (anObj.KnownConv().IsInit()) BinaryDumpInFile(aFp,anObj.KnownConv().Val());
    BinaryDumpInFile(aFp,anObj.ConvExplicite().IsInit());
    if (anObj.ConvExplicite().IsInit()) BinaryDumpInFile(aFp,anObj.ConvExplicite().Val());
}

cElXMLTree * ToXMLTree(const cConvOri & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ConvOri",eXMLBranche);
   if (anObj.KnownConv().IsInit())
      aRes->AddFils(ToXMLTree(std::string("KnownConv"),anObj.KnownConv().Val())->ReTagThis("KnownConv"));
   if (anObj.ConvExplicite().IsInit())
      aRes->AddFils(ToXMLTree(anObj.ConvExplicite().Val())->ReTagThis("ConvExplicite"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cConvOri & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.KnownConv(),aTree->Get("KnownConv",1)); //tototo 

   xml_init(anObj.ConvExplicite(),aTree->Get("ConvExplicite",1)); //tototo 
}

std::string  Mangling( cConvOri *) {return "79FF9A6E30E7969EFD3F";};


std::string & cOrientationConique::NameFileOri()
{
   return ModuleOrientationFile().Val().NameFileOri();
}

const std::string & cOrientationConique::NameFileOri()const 
{
   return ModuleOrientationFile().Val().NameFileOri();
}


cTplValGesInit< cModuleOrientationFile > & cOrientationConique::ModuleOrientationFile()
{
   return mModuleOrientationFile;
}

const cTplValGesInit< cModuleOrientationFile > & cOrientationConique::ModuleOrientationFile()const 
{
   return mModuleOrientationFile;
}


cTplValGesInit< cAffinitePlane > & cOrientationConique::OrIntImaM2C()
{
   return mOrIntImaM2C;
}

const cTplValGesInit< cAffinitePlane > & cOrientationConique::OrIntImaM2C()const 
{
   return mOrIntImaM2C;
}


cTplValGesInit< eTypeProjectionCam > & cOrientationConique::TypeProj()
{
   return mTypeProj;
}

const cTplValGesInit< eTypeProjectionCam > & cOrientationConique::TypeProj()const 
{
   return mTypeProj;
}


cTplValGesInit< bool > & cOrientationConique::ZoneUtileInPixel()
{
   return mZoneUtileInPixel;
}

const cTplValGesInit< bool > & cOrientationConique::ZoneUtileInPixel()const 
{
   return mZoneUtileInPixel;
}


cTplValGesInit< cCalibrationInternConique > & cOrientationConique::Interne()
{
   return mInterne;
}

const cTplValGesInit< cCalibrationInternConique > & cOrientationConique::Interne()const 
{
   return mInterne;
}


cTplValGesInit< std::string > & cOrientationConique::FileInterne()
{
   return mFileInterne;
}

const cTplValGesInit< std::string > & cOrientationConique::FileInterne()const 
{
   return mFileInterne;
}


cTplValGesInit< bool > & cOrientationConique::RelativeNameFI()
{
   return mRelativeNameFI;
}

const cTplValGesInit< bool > & cOrientationConique::RelativeNameFI()const 
{
   return mRelativeNameFI;
}


cOrientationExterneRigide & cOrientationConique::Externe()
{
   return mExterne;
}

const cOrientationExterneRigide & cOrientationConique::Externe()const 
{
   return mExterne;
}


cTplValGesInit< cVerifOrient > & cOrientationConique::Verif()
{
   return mVerif;
}

const cTplValGesInit< cVerifOrient > & cOrientationConique::Verif()const 
{
   return mVerif;
}


cTplValGesInit< eConventionsOrientation > & cOrientationConique::KnownConv()
{
   return ConvOri().KnownConv();
}

const cTplValGesInit< eConventionsOrientation > & cOrientationConique::KnownConv()const 
{
   return ConvOri().KnownConv();
}


cTplValGesInit< bool > & cOrientationConique::SensYVideo()
{
   return ConvOri().ConvExplicite().Val().SensYVideo();
}

const cTplValGesInit< bool > & cOrientationConique::SensYVideo()const 
{
   return ConvOri().ConvExplicite().Val().SensYVideo();
}


cTplValGesInit< bool > & cOrientationConique::DistSenC2M()
{
   return ConvOri().ConvExplicite().Val().DistSenC2M();
}

const cTplValGesInit< bool > & cOrientationConique::DistSenC2M()const 
{
   return ConvOri().ConvExplicite().Val().DistSenC2M();
}


cTplValGesInit< bool > & cOrientationConique::MatrSenC2M()
{
   return ConvOri().ConvExplicite().Val().MatrSenC2M();
}

const cTplValGesInit< bool > & cOrientationConique::MatrSenC2M()const 
{
   return ConvOri().ConvExplicite().Val().MatrSenC2M();
}


cTplValGesInit< Pt3dr > & cOrientationConique::ColMul()
{
   return ConvOri().ConvExplicite().Val().ColMul();
}

const cTplValGesInit< Pt3dr > & cOrientationConique::ColMul()const 
{
   return ConvOri().ConvExplicite().Val().ColMul();
}


cTplValGesInit< Pt3dr > & cOrientationConique::LigMul()
{
   return ConvOri().ConvExplicite().Val().LigMul();
}

const cTplValGesInit< Pt3dr > & cOrientationConique::LigMul()const 
{
   return ConvOri().ConvExplicite().Val().LigMul();
}


cTplValGesInit< eUniteAngulaire > & cOrientationConique::UniteAngles()
{
   return ConvOri().ConvExplicite().Val().UniteAngles();
}

const cTplValGesInit< eUniteAngulaire > & cOrientationConique::UniteAngles()const 
{
   return ConvOri().ConvExplicite().Val().UniteAngles();
}


cTplValGesInit< Pt3di > & cOrientationConique::NumAxe()
{
   return ConvOri().ConvExplicite().Val().NumAxe();
}

const cTplValGesInit< Pt3di > & cOrientationConique::NumAxe()const 
{
   return ConvOri().ConvExplicite().Val().NumAxe();
}


cTplValGesInit< bool > & cOrientationConique::SensCardan()
{
   return ConvOri().ConvExplicite().Val().SensCardan();
}

const cTplValGesInit< bool > & cOrientationConique::SensCardan()const 
{
   return ConvOri().ConvExplicite().Val().SensCardan();
}


cTplValGesInit< eConventionsOrientation > & cOrientationConique::Convention()
{
   return ConvOri().ConvExplicite().Val().Convention();
}

const cTplValGesInit< eConventionsOrientation > & cOrientationConique::Convention()const 
{
   return ConvOri().ConvExplicite().Val().Convention();
}


cTplValGesInit< cConvExplicite > & cOrientationConique::ConvExplicite()
{
   return ConvOri().ConvExplicite();
}

const cTplValGesInit< cConvExplicite > & cOrientationConique::ConvExplicite()const 
{
   return ConvOri().ConvExplicite();
}


cConvOri & cOrientationConique::ConvOri()
{
   return mConvOri;
}

const cConvOri & cOrientationConique::ConvOri()const 
{
   return mConvOri;
}

void  BinaryUnDumpFromFile(cOrientationConique & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ModuleOrientationFile().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ModuleOrientationFile().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ModuleOrientationFile().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.OrIntImaM2C().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.OrIntImaM2C().ValForcedForUnUmp(),aFp);
        }
        else  anObj.OrIntImaM2C().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.TypeProj().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.TypeProj().ValForcedForUnUmp(),aFp);
        }
        else  anObj.TypeProj().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ZoneUtileInPixel().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ZoneUtileInPixel().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ZoneUtileInPixel().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Interne().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Interne().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Interne().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.FileInterne().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.FileInterne().ValForcedForUnUmp(),aFp);
        }
        else  anObj.FileInterne().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.RelativeNameFI().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.RelativeNameFI().ValForcedForUnUmp(),aFp);
        }
        else  anObj.RelativeNameFI().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.Externe(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Verif().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Verif().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Verif().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.ConvOri(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cOrientationConique & anObj)
{
    BinaryDumpInFile(aFp,anObj.ModuleOrientationFile().IsInit());
    if (anObj.ModuleOrientationFile().IsInit()) BinaryDumpInFile(aFp,anObj.ModuleOrientationFile().Val());
    BinaryDumpInFile(aFp,anObj.OrIntImaM2C().IsInit());
    if (anObj.OrIntImaM2C().IsInit()) BinaryDumpInFile(aFp,anObj.OrIntImaM2C().Val());
    BinaryDumpInFile(aFp,anObj.TypeProj().IsInit());
    if (anObj.TypeProj().IsInit()) BinaryDumpInFile(aFp,anObj.TypeProj().Val());
    BinaryDumpInFile(aFp,anObj.ZoneUtileInPixel().IsInit());
    if (anObj.ZoneUtileInPixel().IsInit()) BinaryDumpInFile(aFp,anObj.ZoneUtileInPixel().Val());
    BinaryDumpInFile(aFp,anObj.Interne().IsInit());
    if (anObj.Interne().IsInit()) BinaryDumpInFile(aFp,anObj.Interne().Val());
    BinaryDumpInFile(aFp,anObj.FileInterne().IsInit());
    if (anObj.FileInterne().IsInit()) BinaryDumpInFile(aFp,anObj.FileInterne().Val());
    BinaryDumpInFile(aFp,anObj.RelativeNameFI().IsInit());
    if (anObj.RelativeNameFI().IsInit()) BinaryDumpInFile(aFp,anObj.RelativeNameFI().Val());
    BinaryDumpInFile(aFp,anObj.Externe());
    BinaryDumpInFile(aFp,anObj.Verif().IsInit());
    if (anObj.Verif().IsInit()) BinaryDumpInFile(aFp,anObj.Verif().Val());
    BinaryDumpInFile(aFp,anObj.ConvOri());
}

cElXMLTree * ToXMLTree(const cOrientationConique & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"OrientationConique",eXMLBranche);
   if (anObj.ModuleOrientationFile().IsInit())
      aRes->AddFils(ToXMLTree(anObj.ModuleOrientationFile().Val())->ReTagThis("ModuleOrientationFile"));
   if (anObj.OrIntImaM2C().IsInit())
      aRes->AddFils(ToXMLTree(anObj.OrIntImaM2C().Val())->ReTagThis("OrIntImaM2C"));
   if (anObj.TypeProj().IsInit())
      aRes->AddFils(ToXMLTree(std::string("TypeProj"),anObj.TypeProj().Val())->ReTagThis("TypeProj"));
   if (anObj.ZoneUtileInPixel().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ZoneUtileInPixel"),anObj.ZoneUtileInPixel().Val())->ReTagThis("ZoneUtileInPixel"));
   if (anObj.Interne().IsInit())
      aRes->AddFils(ToXMLTree(anObj.Interne().Val())->ReTagThis("Interne"));
   if (anObj.FileInterne().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("FileInterne"),anObj.FileInterne().Val())->ReTagThis("FileInterne"));
   if (anObj.RelativeNameFI().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("RelativeNameFI"),anObj.RelativeNameFI().Val())->ReTagThis("RelativeNameFI"));
   aRes->AddFils(ToXMLTree(anObj.Externe())->ReTagThis("Externe"));
   if (anObj.Verif().IsInit())
      aRes->AddFils(ToXMLTree(anObj.Verif().Val())->ReTagThis("Verif"));
   aRes->AddFils(ToXMLTree(anObj.ConvOri())->ReTagThis("ConvOri"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cOrientationConique & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.ModuleOrientationFile(),aTree->Get("ModuleOrientationFile",1)); //tototo 

   xml_init(anObj.OrIntImaM2C(),aTree->Get("OrIntImaM2C",1)); //tototo 

   xml_init(anObj.TypeProj(),aTree->Get("TypeProj",1),eTypeProjectionCam(eProjStenope)); //tototo 

   xml_init(anObj.ZoneUtileInPixel(),aTree->Get("ZoneUtileInPixel",1),bool(false)); //tototo 

   xml_init(anObj.Interne(),aTree->Get("Interne",1)); //tototo 

   xml_init(anObj.FileInterne(),aTree->Get("FileInterne",1)); //tototo 

   xml_init(anObj.RelativeNameFI(),aTree->Get("RelativeNameFI",1),bool(false)); //tototo 

   xml_init(anObj.Externe(),aTree->Get("Externe",1)); //tototo 

   xml_init(anObj.Verif(),aTree->Get("Verif",1)); //tototo 

   xml_init(anObj.ConvOri(),aTree->Get("ConvOri",1)); //tototo 
}

std::string  Mangling( cOrientationConique *) {return "B0582EB7CCB2DC82FE3F";};


std::string & cMNT2Cmp::NameIm()
{
   return mNameIm;
}

const std::string & cMNT2Cmp::NameIm()const 
{
   return mNameIm;
}


cTplValGesInit< std::string > & cMNT2Cmp::NameXml()
{
   return mNameXml;
}

const cTplValGesInit< std::string > & cMNT2Cmp::NameXml()const 
{
   return mNameXml;
}


cTplValGesInit< int > & cMNT2Cmp::IdIsRef()
{
   return mIdIsRef;
}

const cTplValGesInit< int > & cMNT2Cmp::IdIsRef()const 
{
   return mIdIsRef;
}


cTplValGesInit< std::string > & cMNT2Cmp::ShorName()
{
   return mShorName;
}

const cTplValGesInit< std::string > & cMNT2Cmp::ShorName()const 
{
   return mShorName;
}

void  BinaryUnDumpFromFile(cMNT2Cmp & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.NameIm(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NameXml().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NameXml().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NameXml().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.IdIsRef().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.IdIsRef().ValForcedForUnUmp(),aFp);
        }
        else  anObj.IdIsRef().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ShorName().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ShorName().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ShorName().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cMNT2Cmp & anObj)
{
    BinaryDumpInFile(aFp,anObj.NameIm());
    BinaryDumpInFile(aFp,anObj.NameXml().IsInit());
    if (anObj.NameXml().IsInit()) BinaryDumpInFile(aFp,anObj.NameXml().Val());
    BinaryDumpInFile(aFp,anObj.IdIsRef().IsInit());
    if (anObj.IdIsRef().IsInit()) BinaryDumpInFile(aFp,anObj.IdIsRef().Val());
    BinaryDumpInFile(aFp,anObj.ShorName().IsInit());
    if (anObj.ShorName().IsInit()) BinaryDumpInFile(aFp,anObj.ShorName().Val());
}

cElXMLTree * ToXMLTree(const cMNT2Cmp & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"MNT2Cmp",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("NameIm"),anObj.NameIm())->ReTagThis("NameIm"));
   if (anObj.NameXml().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NameXml"),anObj.NameXml().Val())->ReTagThis("NameXml"));
   if (anObj.IdIsRef().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("IdIsRef"),anObj.IdIsRef().Val())->ReTagThis("IdIsRef"));
   if (anObj.ShorName().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ShorName"),anObj.ShorName().Val())->ReTagThis("ShorName"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cMNT2Cmp & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.NameIm(),aTree->Get("NameIm",1)); //tototo 

   xml_init(anObj.NameXml(),aTree->Get("NameXml",1)); //tototo 

   xml_init(anObj.IdIsRef(),aTree->Get("IdIsRef",1),int(0)); //tototo 

   xml_init(anObj.ShorName(),aTree->Get("ShorName",1)); //tototo 
}

std::string  Mangling( cMNT2Cmp *) {return "081952F23935DDD2FB3F";};


std::list< Pt2di > & cContourPolyCM::Pts()
{
   return mPts;
}

const std::list< Pt2di > & cContourPolyCM::Pts()const 
{
   return mPts;
}

void  BinaryUnDumpFromFile(cContourPolyCM & anObj,ELISE_fp & aFp)
{
   { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             Pt2di aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.Pts().push_back(aVal);
        }
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cContourPolyCM & anObj)
{
    BinaryDumpInFile(aFp,(int)anObj.Pts().size());
    for(  std::list< Pt2di >::const_iterator iT=anObj.Pts().begin();
         iT!=anObj.Pts().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
}

cElXMLTree * ToXMLTree(const cContourPolyCM & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ContourPolyCM",eXMLBranche);
  for
  (       std::list< Pt2di >::const_iterator it=anObj.Pts().begin();
      it !=anObj.Pts().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("Pts"),(*it))->ReTagThis("Pts"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cContourPolyCM & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Pts(),aTree->GetAll("Pts",false,1));
}

std::string  Mangling( cContourPolyCM *) {return "EAE04C28384AB1B9FE3F";};


std::list< Pt2di > & cEnvellopeZoneCM::Pts()
{
   return ContourPolyCM().Val().Pts();
}

const std::list< Pt2di > & cEnvellopeZoneCM::Pts()const 
{
   return ContourPolyCM().Val().Pts();
}


cTplValGesInit< cContourPolyCM > & cEnvellopeZoneCM::ContourPolyCM()
{
   return mContourPolyCM;
}

const cTplValGesInit< cContourPolyCM > & cEnvellopeZoneCM::ContourPolyCM()const 
{
   return mContourPolyCM;
}


cTplValGesInit< Box2dr > & cEnvellopeZoneCM::BoxContourCM()
{
   return mBoxContourCM;
}

const cTplValGesInit< Box2dr > & cEnvellopeZoneCM::BoxContourCM()const 
{
   return mBoxContourCM;
}

void  BinaryUnDumpFromFile(cEnvellopeZoneCM & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ContourPolyCM().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ContourPolyCM().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ContourPolyCM().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.BoxContourCM().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.BoxContourCM().ValForcedForUnUmp(),aFp);
        }
        else  anObj.BoxContourCM().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cEnvellopeZoneCM & anObj)
{
    BinaryDumpInFile(aFp,anObj.ContourPolyCM().IsInit());
    if (anObj.ContourPolyCM().IsInit()) BinaryDumpInFile(aFp,anObj.ContourPolyCM().Val());
    BinaryDumpInFile(aFp,anObj.BoxContourCM().IsInit());
    if (anObj.BoxContourCM().IsInit()) BinaryDumpInFile(aFp,anObj.BoxContourCM().Val());
}

cElXMLTree * ToXMLTree(const cEnvellopeZoneCM & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"EnvellopeZoneCM",eXMLBranche);
   if (anObj.ContourPolyCM().IsInit())
      aRes->AddFils(ToXMLTree(anObj.ContourPolyCM().Val())->ReTagThis("ContourPolyCM"));
   if (anObj.BoxContourCM().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("BoxContourCM"),anObj.BoxContourCM().Val())->ReTagThis("BoxContourCM"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cEnvellopeZoneCM & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.ContourPolyCM(),aTree->Get("ContourPolyCM",1)); //tototo 

   xml_init(anObj.BoxContourCM(),aTree->Get("BoxContourCM",1)); //tototo 
}

std::string  Mangling( cEnvellopeZoneCM *) {return "06A4689B4E500CB6FF3F";};


cTplValGesInit< std::string > & cZoneCmpMnt::NomZone()
{
   return mNomZone;
}

const cTplValGesInit< std::string > & cZoneCmpMnt::NomZone()const 
{
   return mNomZone;
}


std::list< Pt2di > & cZoneCmpMnt::Pts()
{
   return EnvellopeZoneCM().ContourPolyCM().Val().Pts();
}

const std::list< Pt2di > & cZoneCmpMnt::Pts()const 
{
   return EnvellopeZoneCM().ContourPolyCM().Val().Pts();
}


cTplValGesInit< cContourPolyCM > & cZoneCmpMnt::ContourPolyCM()
{
   return EnvellopeZoneCM().ContourPolyCM();
}

const cTplValGesInit< cContourPolyCM > & cZoneCmpMnt::ContourPolyCM()const 
{
   return EnvellopeZoneCM().ContourPolyCM();
}


cTplValGesInit< Box2dr > & cZoneCmpMnt::BoxContourCM()
{
   return EnvellopeZoneCM().BoxContourCM();
}

const cTplValGesInit< Box2dr > & cZoneCmpMnt::BoxContourCM()const 
{
   return EnvellopeZoneCM().BoxContourCM();
}


cEnvellopeZoneCM & cZoneCmpMnt::EnvellopeZoneCM()
{
   return mEnvellopeZoneCM;
}

const cEnvellopeZoneCM & cZoneCmpMnt::EnvellopeZoneCM()const 
{
   return mEnvellopeZoneCM;
}

void  BinaryUnDumpFromFile(cZoneCmpMnt & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NomZone().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NomZone().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NomZone().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.EnvellopeZoneCM(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cZoneCmpMnt & anObj)
{
    BinaryDumpInFile(aFp,anObj.NomZone().IsInit());
    if (anObj.NomZone().IsInit()) BinaryDumpInFile(aFp,anObj.NomZone().Val());
    BinaryDumpInFile(aFp,anObj.EnvellopeZoneCM());
}

cElXMLTree * ToXMLTree(const cZoneCmpMnt & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ZoneCmpMnt",eXMLBranche);
   if (anObj.NomZone().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NomZone"),anObj.NomZone().Val())->ReTagThis("NomZone"));
   aRes->AddFils(ToXMLTree(anObj.EnvellopeZoneCM())->ReTagThis("EnvellopeZoneCM"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cZoneCmpMnt & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.NomZone(),aTree->Get("NomZone",1)); //tototo 

   xml_init(anObj.EnvellopeZoneCM(),aTree->Get("EnvellopeZoneCM",1)); //tototo 
}

std::string  Mangling( cZoneCmpMnt *) {return "E4316350E67AF8D3FE3F";};


double & cEcartZ::DynVisu()
{
   return mDynVisu;
}

const double & cEcartZ::DynVisu()const 
{
   return mDynVisu;
}

void  BinaryUnDumpFromFile(cEcartZ & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.DynVisu(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cEcartZ & anObj)
{
    BinaryDumpInFile(aFp,anObj.DynVisu());
}

cElXMLTree * ToXMLTree(const cEcartZ & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"EcartZ",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("DynVisu"),anObj.DynVisu())->ReTagThis("DynVisu"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cEcartZ & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.DynVisu(),aTree->Get("DynVisu",1)); //tototo 
}

std::string  Mangling( cEcartZ *) {return "D725C7C3FCC91E96FE3F";};


double & cCorrelPente::SzWCP()
{
   return mSzWCP;
}

const double & cCorrelPente::SzWCP()const 
{
   return mSzWCP;
}


double & cCorrelPente::GrMinCP()
{
   return mGrMinCP;
}

const double & cCorrelPente::GrMinCP()const 
{
   return mGrMinCP;
}

void  BinaryUnDumpFromFile(cCorrelPente & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.SzWCP(),aFp);
    BinaryUnDumpFromFile(anObj.GrMinCP(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cCorrelPente & anObj)
{
    BinaryDumpInFile(aFp,anObj.SzWCP());
    BinaryDumpInFile(aFp,anObj.GrMinCP());
}

cElXMLTree * ToXMLTree(const cCorrelPente & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"CorrelPente",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("SzWCP"),anObj.SzWCP())->ReTagThis("SzWCP"));
   aRes->AddFils(::ToXMLTree(std::string("GrMinCP"),anObj.GrMinCP())->ReTagThis("GrMinCP"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cCorrelPente & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.SzWCP(),aTree->Get("SzWCP",1)); //tototo 

   xml_init(anObj.GrMinCP(),aTree->Get("GrMinCP",1)); //tototo 
}

std::string  Mangling( cCorrelPente *) {return "CEACBAD71FE07AECFD3F";};


double & cMesureCmptMnt::DynVisu()
{
   return EcartZ().Val().DynVisu();
}

const double & cMesureCmptMnt::DynVisu()const 
{
   return EcartZ().Val().DynVisu();
}


cTplValGesInit< cEcartZ > & cMesureCmptMnt::EcartZ()
{
   return mEcartZ;
}

const cTplValGesInit< cEcartZ > & cMesureCmptMnt::EcartZ()const 
{
   return mEcartZ;
}


double & cMesureCmptMnt::SzWCP()
{
   return CorrelPente().Val().SzWCP();
}

const double & cMesureCmptMnt::SzWCP()const 
{
   return CorrelPente().Val().SzWCP();
}


double & cMesureCmptMnt::GrMinCP()
{
   return CorrelPente().Val().GrMinCP();
}

const double & cMesureCmptMnt::GrMinCP()const 
{
   return CorrelPente().Val().GrMinCP();
}


cTplValGesInit< cCorrelPente > & cMesureCmptMnt::CorrelPente()
{
   return mCorrelPente;
}

const cTplValGesInit< cCorrelPente > & cMesureCmptMnt::CorrelPente()const 
{
   return mCorrelPente;
}


cTplValGesInit< bool > & cMesureCmptMnt::EcartPente()
{
   return mEcartPente;
}

const cTplValGesInit< bool > & cMesureCmptMnt::EcartPente()const 
{
   return mEcartPente;
}

void  BinaryUnDumpFromFile(cMesureCmptMnt & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.EcartZ().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.EcartZ().ValForcedForUnUmp(),aFp);
        }
        else  anObj.EcartZ().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.CorrelPente().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.CorrelPente().ValForcedForUnUmp(),aFp);
        }
        else  anObj.CorrelPente().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.EcartPente().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.EcartPente().ValForcedForUnUmp(),aFp);
        }
        else  anObj.EcartPente().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cMesureCmptMnt & anObj)
{
    BinaryDumpInFile(aFp,anObj.EcartZ().IsInit());
    if (anObj.EcartZ().IsInit()) BinaryDumpInFile(aFp,anObj.EcartZ().Val());
    BinaryDumpInFile(aFp,anObj.CorrelPente().IsInit());
    if (anObj.CorrelPente().IsInit()) BinaryDumpInFile(aFp,anObj.CorrelPente().Val());
    BinaryDumpInFile(aFp,anObj.EcartPente().IsInit());
    if (anObj.EcartPente().IsInit()) BinaryDumpInFile(aFp,anObj.EcartPente().Val());
}

cElXMLTree * ToXMLTree(const cMesureCmptMnt & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"MesureCmptMnt",eXMLBranche);
   if (anObj.EcartZ().IsInit())
      aRes->AddFils(ToXMLTree(anObj.EcartZ().Val())->ReTagThis("EcartZ"));
   if (anObj.CorrelPente().IsInit())
      aRes->AddFils(ToXMLTree(anObj.CorrelPente().Val())->ReTagThis("CorrelPente"));
   if (anObj.EcartPente().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("EcartPente"),anObj.EcartPente().Val())->ReTagThis("EcartPente"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cMesureCmptMnt & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.EcartZ(),aTree->Get("EcartZ",1)); //tototo 

   xml_init(anObj.CorrelPente(),aTree->Get("CorrelPente",1)); //tototo 

   xml_init(anObj.EcartPente(),aTree->Get("EcartPente",1),bool(false)); //tototo 
}

std::string  Mangling( cMesureCmptMnt *) {return "3AFD4955150F15C7FE3F";};


Pt2dr & cCompareMNT::ResolutionPlaniTerrain()
{
   return mResolutionPlaniTerrain;
}

const Pt2dr & cCompareMNT::ResolutionPlaniTerrain()const 
{
   return mResolutionPlaniTerrain;
}


cTplValGesInit< int > & cCompareMNT::RabLoad()
{
   return mRabLoad;
}

const cTplValGesInit< int > & cCompareMNT::RabLoad()const 
{
   return mRabLoad;
}


std::string & cCompareMNT::NameFileRes()
{
   return mNameFileRes;
}

const std::string & cCompareMNT::NameFileRes()const 
{
   return mNameFileRes;
}


cTplValGesInit< bool > & cCompareMNT::VisuInter()
{
   return mVisuInter;
}

const cTplValGesInit< bool > & cCompareMNT::VisuInter()const 
{
   return mVisuInter;
}


std::list< cMNT2Cmp > & cCompareMNT::MNT2Cmp()
{
   return mMNT2Cmp;
}

const std::list< cMNT2Cmp > & cCompareMNT::MNT2Cmp()const 
{
   return mMNT2Cmp;
}


cTplValGesInit< std::string > & cCompareMNT::MasqGlobalCM()
{
   return mMasqGlobalCM;
}

const cTplValGesInit< std::string > & cCompareMNT::MasqGlobalCM()const 
{
   return mMasqGlobalCM;
}


std::list< cZoneCmpMnt > & cCompareMNT::ZoneCmpMnt()
{
   return mZoneCmpMnt;
}

const std::list< cZoneCmpMnt > & cCompareMNT::ZoneCmpMnt()const 
{
   return mZoneCmpMnt;
}


double & cCompareMNT::DynVisu()
{
   return MesureCmptMnt().EcartZ().Val().DynVisu();
}

const double & cCompareMNT::DynVisu()const 
{
   return MesureCmptMnt().EcartZ().Val().DynVisu();
}


cTplValGesInit< cEcartZ > & cCompareMNT::EcartZ()
{
   return MesureCmptMnt().EcartZ();
}

const cTplValGesInit< cEcartZ > & cCompareMNT::EcartZ()const 
{
   return MesureCmptMnt().EcartZ();
}


double & cCompareMNT::SzWCP()
{
   return MesureCmptMnt().CorrelPente().Val().SzWCP();
}

const double & cCompareMNT::SzWCP()const 
{
   return MesureCmptMnt().CorrelPente().Val().SzWCP();
}


double & cCompareMNT::GrMinCP()
{
   return MesureCmptMnt().CorrelPente().Val().GrMinCP();
}

const double & cCompareMNT::GrMinCP()const 
{
   return MesureCmptMnt().CorrelPente().Val().GrMinCP();
}


cTplValGesInit< cCorrelPente > & cCompareMNT::CorrelPente()
{
   return MesureCmptMnt().CorrelPente();
}

const cTplValGesInit< cCorrelPente > & cCompareMNT::CorrelPente()const 
{
   return MesureCmptMnt().CorrelPente();
}


cTplValGesInit< bool > & cCompareMNT::EcartPente()
{
   return MesureCmptMnt().EcartPente();
}

const cTplValGesInit< bool > & cCompareMNT::EcartPente()const 
{
   return MesureCmptMnt().EcartPente();
}


cMesureCmptMnt & cCompareMNT::MesureCmptMnt()
{
   return mMesureCmptMnt;
}

const cMesureCmptMnt & cCompareMNT::MesureCmptMnt()const 
{
   return mMesureCmptMnt;
}

void  BinaryUnDumpFromFile(cCompareMNT & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.ResolutionPlaniTerrain(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.RabLoad().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.RabLoad().ValForcedForUnUmp(),aFp);
        }
        else  anObj.RabLoad().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.NameFileRes(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.VisuInter().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.VisuInter().ValForcedForUnUmp(),aFp);
        }
        else  anObj.VisuInter().SetNoInit();
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cMNT2Cmp aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.MNT2Cmp().push_back(aVal);
        }
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.MasqGlobalCM().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.MasqGlobalCM().ValForcedForUnUmp(),aFp);
        }
        else  anObj.MasqGlobalCM().SetNoInit();
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cZoneCmpMnt aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.ZoneCmpMnt().push_back(aVal);
        }
  } ;
    BinaryUnDumpFromFile(anObj.MesureCmptMnt(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cCompareMNT & anObj)
{
    BinaryDumpInFile(aFp,anObj.ResolutionPlaniTerrain());
    BinaryDumpInFile(aFp,anObj.RabLoad().IsInit());
    if (anObj.RabLoad().IsInit()) BinaryDumpInFile(aFp,anObj.RabLoad().Val());
    BinaryDumpInFile(aFp,anObj.NameFileRes());
    BinaryDumpInFile(aFp,anObj.VisuInter().IsInit());
    if (anObj.VisuInter().IsInit()) BinaryDumpInFile(aFp,anObj.VisuInter().Val());
    BinaryDumpInFile(aFp,(int)anObj.MNT2Cmp().size());
    for(  std::list< cMNT2Cmp >::const_iterator iT=anObj.MNT2Cmp().begin();
         iT!=anObj.MNT2Cmp().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.MasqGlobalCM().IsInit());
    if (anObj.MasqGlobalCM().IsInit()) BinaryDumpInFile(aFp,anObj.MasqGlobalCM().Val());
    BinaryDumpInFile(aFp,(int)anObj.ZoneCmpMnt().size());
    for(  std::list< cZoneCmpMnt >::const_iterator iT=anObj.ZoneCmpMnt().begin();
         iT!=anObj.ZoneCmpMnt().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.MesureCmptMnt());
}

cElXMLTree * ToXMLTree(const cCompareMNT & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"CompareMNT",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("ResolutionPlaniTerrain"),anObj.ResolutionPlaniTerrain())->ReTagThis("ResolutionPlaniTerrain"));
   if (anObj.RabLoad().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("RabLoad"),anObj.RabLoad().Val())->ReTagThis("RabLoad"));
   aRes->AddFils(::ToXMLTree(std::string("NameFileRes"),anObj.NameFileRes())->ReTagThis("NameFileRes"));
   if (anObj.VisuInter().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("VisuInter"),anObj.VisuInter().Val())->ReTagThis("VisuInter"));
  for
  (       std::list< cMNT2Cmp >::const_iterator it=anObj.MNT2Cmp().begin();
      it !=anObj.MNT2Cmp().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("MNT2Cmp"));
   if (anObj.MasqGlobalCM().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("MasqGlobalCM"),anObj.MasqGlobalCM().Val())->ReTagThis("MasqGlobalCM"));
  for
  (       std::list< cZoneCmpMnt >::const_iterator it=anObj.ZoneCmpMnt().begin();
      it !=anObj.ZoneCmpMnt().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("ZoneCmpMnt"));
   aRes->AddFils(ToXMLTree(anObj.MesureCmptMnt())->ReTagThis("MesureCmptMnt"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cCompareMNT & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.ResolutionPlaniTerrain(),aTree->Get("ResolutionPlaniTerrain",1)); //tototo 

   xml_init(anObj.RabLoad(),aTree->Get("RabLoad",1),int(20)); //tototo 

   xml_init(anObj.NameFileRes(),aTree->Get("NameFileRes",1)); //tototo 

   xml_init(anObj.VisuInter(),aTree->Get("VisuInter",1),bool(false)); //tototo 

   xml_init(anObj.MNT2Cmp(),aTree->GetAll("MNT2Cmp",false,1));

   xml_init(anObj.MasqGlobalCM(),aTree->Get("MasqGlobalCM",1)); //tototo 

   xml_init(anObj.ZoneCmpMnt(),aTree->GetAll("ZoneCmpMnt",false,1));

   xml_init(anObj.MesureCmptMnt(),aTree->Get("MesureCmptMnt",1)); //tototo 
}

std::string  Mangling( cCompareMNT *) {return "3D2B62D216188299FD3F";};


cTplValGesInit< double > & cDataBaseNameTransfo::AddFocMul()
{
   return mAddFocMul;
}

const cTplValGesInit< double > & cDataBaseNameTransfo::AddFocMul()const 
{
   return mAddFocMul;
}


cTplValGesInit< std::string > & cDataBaseNameTransfo::Separateur()
{
   return mSeparateur;
}

const cTplValGesInit< std::string > & cDataBaseNameTransfo::Separateur()const 
{
   return mSeparateur;
}

void  BinaryUnDumpFromFile(cDataBaseNameTransfo & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.AddFocMul().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.AddFocMul().ValForcedForUnUmp(),aFp);
        }
        else  anObj.AddFocMul().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Separateur().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Separateur().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Separateur().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cDataBaseNameTransfo & anObj)
{
    BinaryDumpInFile(aFp,anObj.AddFocMul().IsInit());
    if (anObj.AddFocMul().IsInit()) BinaryDumpInFile(aFp,anObj.AddFocMul().Val());
    BinaryDumpInFile(aFp,anObj.Separateur().IsInit());
    if (anObj.Separateur().IsInit()) BinaryDumpInFile(aFp,anObj.Separateur().Val());
}

cElXMLTree * ToXMLTree(const cDataBaseNameTransfo & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"DataBaseNameTransfo",eXMLBranche);
   if (anObj.AddFocMul().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("AddFocMul"),anObj.AddFocMul().Val())->ReTagThis("AddFocMul"));
   if (anObj.Separateur().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Separateur"),anObj.Separateur().Val())->ReTagThis("Separateur"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cDataBaseNameTransfo & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.AddFocMul(),aTree->Get("AddFocMul",1)); //tototo 

   xml_init(anObj.Separateur(),aTree->Get("Separateur",1),std::string("%")); //tototo 
}

std::string  Mangling( cDataBaseNameTransfo *) {return "5C97038E3E214B9EFD3F";};


cTplValGesInit< std::string > & cInterpoleGrille::Directory()
{
   return mDirectory;
}

const cTplValGesInit< std::string > & cInterpoleGrille::Directory()const 
{
   return mDirectory;
}


std::string & cInterpoleGrille::Grille1()
{
   return mGrille1;
}

const std::string & cInterpoleGrille::Grille1()const 
{
   return mGrille1;
}


std::string & cInterpoleGrille::Grille2()
{
   return mGrille2;
}

const std::string & cInterpoleGrille::Grille2()const 
{
   return mGrille2;
}


std::string & cInterpoleGrille::Grille0()
{
   return mGrille0;
}

const std::string & cInterpoleGrille::Grille0()const 
{
   return mGrille0;
}


cTplValGesInit< Pt2dr > & cInterpoleGrille::StepGrid()
{
   return mStepGrid;
}

const cTplValGesInit< Pt2dr > & cInterpoleGrille::StepGrid()const 
{
   return mStepGrid;
}


double & cInterpoleGrille::Focale1()
{
   return mFocale1;
}

const double & cInterpoleGrille::Focale1()const 
{
   return mFocale1;
}


double & cInterpoleGrille::Focale2()
{
   return mFocale2;
}

const double & cInterpoleGrille::Focale2()const 
{
   return mFocale2;
}


double & cInterpoleGrille::Focale0()
{
   return mFocale0;
}

const double & cInterpoleGrille::Focale0()const 
{
   return mFocale0;
}


cTplValGesInit< int > & cInterpoleGrille::NbPtsByIter()
{
   return mNbPtsByIter;
}

const cTplValGesInit< int > & cInterpoleGrille::NbPtsByIter()const 
{
   return mNbPtsByIter;
}


cTplValGesInit< int > & cInterpoleGrille::DegPoly()
{
   return mDegPoly;
}

const cTplValGesInit< int > & cInterpoleGrille::DegPoly()const 
{
   return mDegPoly;
}


cTplValGesInit< eDegreLiberteCPP > & cInterpoleGrille::LiberteCPP()
{
   return mLiberteCPP;
}

const cTplValGesInit< eDegreLiberteCPP > & cInterpoleGrille::LiberteCPP()const 
{
   return mLiberteCPP;
}

void  BinaryUnDumpFromFile(cInterpoleGrille & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Directory().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Directory().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Directory().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.Grille1(),aFp);
    BinaryUnDumpFromFile(anObj.Grille2(),aFp);
    BinaryUnDumpFromFile(anObj.Grille0(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.StepGrid().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.StepGrid().ValForcedForUnUmp(),aFp);
        }
        else  anObj.StepGrid().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.Focale1(),aFp);
    BinaryUnDumpFromFile(anObj.Focale2(),aFp);
    BinaryUnDumpFromFile(anObj.Focale0(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NbPtsByIter().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NbPtsByIter().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NbPtsByIter().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.DegPoly().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.DegPoly().ValForcedForUnUmp(),aFp);
        }
        else  anObj.DegPoly().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.LiberteCPP().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.LiberteCPP().ValForcedForUnUmp(),aFp);
        }
        else  anObj.LiberteCPP().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cInterpoleGrille & anObj)
{
    BinaryDumpInFile(aFp,anObj.Directory().IsInit());
    if (anObj.Directory().IsInit()) BinaryDumpInFile(aFp,anObj.Directory().Val());
    BinaryDumpInFile(aFp,anObj.Grille1());
    BinaryDumpInFile(aFp,anObj.Grille2());
    BinaryDumpInFile(aFp,anObj.Grille0());
    BinaryDumpInFile(aFp,anObj.StepGrid().IsInit());
    if (anObj.StepGrid().IsInit()) BinaryDumpInFile(aFp,anObj.StepGrid().Val());
    BinaryDumpInFile(aFp,anObj.Focale1());
    BinaryDumpInFile(aFp,anObj.Focale2());
    BinaryDumpInFile(aFp,anObj.Focale0());
    BinaryDumpInFile(aFp,anObj.NbPtsByIter().IsInit());
    if (anObj.NbPtsByIter().IsInit()) BinaryDumpInFile(aFp,anObj.NbPtsByIter().Val());
    BinaryDumpInFile(aFp,anObj.DegPoly().IsInit());
    if (anObj.DegPoly().IsInit()) BinaryDumpInFile(aFp,anObj.DegPoly().Val());
    BinaryDumpInFile(aFp,anObj.LiberteCPP().IsInit());
    if (anObj.LiberteCPP().IsInit()) BinaryDumpInFile(aFp,anObj.LiberteCPP().Val());
}

cElXMLTree * ToXMLTree(const cInterpoleGrille & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"InterpoleGrille",eXMLBranche);
   if (anObj.Directory().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Directory"),anObj.Directory().Val())->ReTagThis("Directory"));
   aRes->AddFils(::ToXMLTree(std::string("Grille1"),anObj.Grille1())->ReTagThis("Grille1"));
   aRes->AddFils(::ToXMLTree(std::string("Grille2"),anObj.Grille2())->ReTagThis("Grille2"));
   aRes->AddFils(::ToXMLTree(std::string("Grille0"),anObj.Grille0())->ReTagThis("Grille0"));
   if (anObj.StepGrid().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("StepGrid"),anObj.StepGrid().Val())->ReTagThis("StepGrid"));
   aRes->AddFils(::ToXMLTree(std::string("Focale1"),anObj.Focale1())->ReTagThis("Focale1"));
   aRes->AddFils(::ToXMLTree(std::string("Focale2"),anObj.Focale2())->ReTagThis("Focale2"));
   aRes->AddFils(::ToXMLTree(std::string("Focale0"),anObj.Focale0())->ReTagThis("Focale0"));
   if (anObj.NbPtsByIter().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NbPtsByIter"),anObj.NbPtsByIter().Val())->ReTagThis("NbPtsByIter"));
   if (anObj.DegPoly().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DegPoly"),anObj.DegPoly().Val())->ReTagThis("DegPoly"));
   if (anObj.LiberteCPP().IsInit())
      aRes->AddFils(ToXMLTree(std::string("LiberteCPP"),anObj.LiberteCPP().Val())->ReTagThis("LiberteCPP"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cInterpoleGrille & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Directory(),aTree->Get("Directory",1),std::string("")); //tototo 

   xml_init(anObj.Grille1(),aTree->Get("Grille1",1)); //tototo 

   xml_init(anObj.Grille2(),aTree->Get("Grille2",1)); //tototo 

   xml_init(anObj.Grille0(),aTree->Get("Grille0",1)); //tototo 

   xml_init(anObj.StepGrid(),aTree->Get("StepGrid",1)); //tototo 

   xml_init(anObj.Focale1(),aTree->Get("Focale1",1)); //tototo 

   xml_init(anObj.Focale2(),aTree->Get("Focale2",1)); //tototo 

   xml_init(anObj.Focale0(),aTree->Get("Focale0",1)); //tototo 

   xml_init(anObj.NbPtsByIter(),aTree->Get("NbPtsByIter",1),int(30)); //tototo 

   xml_init(anObj.DegPoly(),aTree->Get("DegPoly",1),int(3)); //tototo 

   xml_init(anObj.LiberteCPP(),aTree->Get("LiberteCPP",1),eDegreLiberteCPP(eCPPLibres)); //tototo 
}

std::string  Mangling( cInterpoleGrille *) {return "C336A5D1DD1EBEF4FD3F";};


std::string & cOneCalib2Visu::Name()
{
   return mName;
}

const std::string & cOneCalib2Visu::Name()const 
{
   return mName;
}


Pt3dr & cOneCalib2Visu::Coul()
{
   return mCoul;
}

const Pt3dr & cOneCalib2Visu::Coul()const 
{
   return mCoul;
}

void  BinaryUnDumpFromFile(cOneCalib2Visu & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Name(),aFp);
    BinaryUnDumpFromFile(anObj.Coul(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cOneCalib2Visu & anObj)
{
    BinaryDumpInFile(aFp,anObj.Name());
    BinaryDumpInFile(aFp,anObj.Coul());
}

cElXMLTree * ToXMLTree(const cOneCalib2Visu & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"OneCalib2Visu",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Name"),anObj.Name())->ReTagThis("Name"));
   aRes->AddFils(ToXMLTree(std::string("Coul"),anObj.Coul())->ReTagThis("Coul"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cOneCalib2Visu & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Name(),aTree->Get("Name",1)); //tototo 

   xml_init(anObj.Coul(),aTree->Get("Coul",1)); //tototo 
}

std::string  Mangling( cOneCalib2Visu *) {return "F6F0CC04071BBF86FD3F";};


cTplValGesInit< std::string > & cVisuCalibZoom::Directory()
{
   return mDirectory;
}

const cTplValGesInit< std::string > & cVisuCalibZoom::Directory()const 
{
   return mDirectory;
}


Pt2dr & cVisuCalibZoom::SzIm()
{
   return mSzIm;
}

const Pt2dr & cVisuCalibZoom::SzIm()const 
{
   return mSzIm;
}


std::list< cOneCalib2Visu > & cVisuCalibZoom::OneCalib2Visu()
{
   return mOneCalib2Visu;
}

const std::list< cOneCalib2Visu > & cVisuCalibZoom::OneCalib2Visu()const 
{
   return mOneCalib2Visu;
}

void  BinaryUnDumpFromFile(cVisuCalibZoom & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Directory().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Directory().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Directory().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.SzIm(),aFp);
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cOneCalib2Visu aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.OneCalib2Visu().push_back(aVal);
        }
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cVisuCalibZoom & anObj)
{
    BinaryDumpInFile(aFp,anObj.Directory().IsInit());
    if (anObj.Directory().IsInit()) BinaryDumpInFile(aFp,anObj.Directory().Val());
    BinaryDumpInFile(aFp,anObj.SzIm());
    BinaryDumpInFile(aFp,(int)anObj.OneCalib2Visu().size());
    for(  std::list< cOneCalib2Visu >::const_iterator iT=anObj.OneCalib2Visu().begin();
         iT!=anObj.OneCalib2Visu().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
}

cElXMLTree * ToXMLTree(const cVisuCalibZoom & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"VisuCalibZoom",eXMLBranche);
   if (anObj.Directory().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Directory"),anObj.Directory().Val())->ReTagThis("Directory"));
   aRes->AddFils(::ToXMLTree(std::string("SzIm"),anObj.SzIm())->ReTagThis("SzIm"));
  for
  (       std::list< cOneCalib2Visu >::const_iterator it=anObj.OneCalib2Visu().begin();
      it !=anObj.OneCalib2Visu().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("OneCalib2Visu"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cVisuCalibZoom & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Directory(),aTree->Get("Directory",1),std::string("")); //tototo 

   xml_init(anObj.SzIm(),aTree->Get("SzIm",1)); //tototo 

   xml_init(anObj.OneCalib2Visu(),aTree->GetAll("OneCalib2Visu",false,1));
}

std::string  Mangling( cVisuCalibZoom *) {return "2E22ACAF33397CC1FE3F";};


std::string & cFilterLocalisation::KeyAssocOrient()
{
   return mKeyAssocOrient;
}

const std::string & cFilterLocalisation::KeyAssocOrient()const 
{
   return mKeyAssocOrient;
}


std::string & cFilterLocalisation::NameMasq()
{
   return mNameMasq;
}

const std::string & cFilterLocalisation::NameMasq()const 
{
   return mNameMasq;
}


std::string & cFilterLocalisation::NameMTDMasq()
{
   return mNameMTDMasq;
}

const std::string & cFilterLocalisation::NameMTDMasq()const 
{
   return mNameMTDMasq;
}

void  BinaryUnDumpFromFile(cFilterLocalisation & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.KeyAssocOrient(),aFp);
    BinaryUnDumpFromFile(anObj.NameMasq(),aFp);
    BinaryUnDumpFromFile(anObj.NameMTDMasq(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cFilterLocalisation & anObj)
{
    BinaryDumpInFile(aFp,anObj.KeyAssocOrient());
    BinaryDumpInFile(aFp,anObj.NameMasq());
    BinaryDumpInFile(aFp,anObj.NameMTDMasq());
}

cElXMLTree * ToXMLTree(const cFilterLocalisation & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"FilterLocalisation",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("KeyAssocOrient"),anObj.KeyAssocOrient())->ReTagThis("KeyAssocOrient"));
   aRes->AddFils(::ToXMLTree(std::string("NameMasq"),anObj.NameMasq())->ReTagThis("NameMasq"));
   aRes->AddFils(::ToXMLTree(std::string("NameMTDMasq"),anObj.NameMTDMasq())->ReTagThis("NameMTDMasq"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cFilterLocalisation & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.KeyAssocOrient(),aTree->Get("KeyAssocOrient",1)); //tototo 

   xml_init(anObj.NameMasq(),aTree->Get("NameMasq",1)); //tototo 

   xml_init(anObj.NameMTDMasq(),aTree->Get("NameMTDMasq",1)); //tototo 
}

std::string  Mangling( cFilterLocalisation *) {return "C08109AC85155FBCFA3F";};


std::list< std::string > & cKeyExistingFile::KeyAssoc()
{
   return mKeyAssoc;
}

const std::list< std::string > & cKeyExistingFile::KeyAssoc()const 
{
   return mKeyAssoc;
}


bool & cKeyExistingFile::RequireExist()
{
   return mRequireExist;
}

const bool & cKeyExistingFile::RequireExist()const 
{
   return mRequireExist;
}


bool & cKeyExistingFile::RequireForAll()
{
   return mRequireForAll;
}

const bool & cKeyExistingFile::RequireForAll()const 
{
   return mRequireForAll;
}

void  BinaryUnDumpFromFile(cKeyExistingFile & anObj,ELISE_fp & aFp)
{
   { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             std::string aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.KeyAssoc().push_back(aVal);
        }
  } ;
    BinaryUnDumpFromFile(anObj.RequireExist(),aFp);
    BinaryUnDumpFromFile(anObj.RequireForAll(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cKeyExistingFile & anObj)
{
    BinaryDumpInFile(aFp,(int)anObj.KeyAssoc().size());
    for(  std::list< std::string >::const_iterator iT=anObj.KeyAssoc().begin();
         iT!=anObj.KeyAssoc().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.RequireExist());
    BinaryDumpInFile(aFp,anObj.RequireForAll());
}

cElXMLTree * ToXMLTree(const cKeyExistingFile & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"KeyExistingFile",eXMLBranche);
  for
  (       std::list< std::string >::const_iterator it=anObj.KeyAssoc().begin();
      it !=anObj.KeyAssoc().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("KeyAssoc"),(*it))->ReTagThis("KeyAssoc"));
   aRes->AddFils(::ToXMLTree(std::string("RequireExist"),anObj.RequireExist())->ReTagThis("RequireExist"));
   aRes->AddFils(::ToXMLTree(std::string("RequireForAll"),anObj.RequireForAll())->ReTagThis("RequireForAll"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cKeyExistingFile & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.KeyAssoc(),aTree->GetAll("KeyAssoc",false,1));

   xml_init(anObj.RequireExist(),aTree->Get("RequireExist",1)); //tototo 

   xml_init(anObj.RequireForAll(),aTree->Get("RequireForAll",1)); //tototo 
}

std::string  Mangling( cKeyExistingFile *) {return "B6F74E0FB084CF9EFE3F";};


std::list< Pt2drSubst > & cNameFilter::FocMm()
{
   return mFocMm;
}

const std::list< Pt2drSubst > & cNameFilter::FocMm()const 
{
   return mFocMm;
}


cTplValGesInit< std::string > & cNameFilter::Min()
{
   return mMin;
}

const cTplValGesInit< std::string > & cNameFilter::Min()const 
{
   return mMin;
}


cTplValGesInit< std::string > & cNameFilter::Max()
{
   return mMax;
}

const cTplValGesInit< std::string > & cNameFilter::Max()const 
{
   return mMax;
}


cTplValGesInit< int > & cNameFilter::SizeMinFile()
{
   return mSizeMinFile;
}

const cTplValGesInit< int > & cNameFilter::SizeMinFile()const 
{
   return mSizeMinFile;
}


std::list< cKeyExistingFile > & cNameFilter::KeyExistingFile()
{
   return mKeyExistingFile;
}

const std::list< cKeyExistingFile > & cNameFilter::KeyExistingFile()const 
{
   return mKeyExistingFile;
}


cTplValGesInit< cFilterLocalisation > & cNameFilter::KeyLocalisation()
{
   return mKeyLocalisation;
}

const cTplValGesInit< cFilterLocalisation > & cNameFilter::KeyLocalisation()const 
{
   return mKeyLocalisation;
}

void  BinaryUnDumpFromFile(cNameFilter & anObj,ELISE_fp & aFp)
{
   { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             Pt2drSubst aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.FocMm().push_back(aVal);
        }
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Min().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Min().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Min().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Max().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Max().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Max().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SizeMinFile().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SizeMinFile().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SizeMinFile().SetNoInit();
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cKeyExistingFile aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.KeyExistingFile().push_back(aVal);
        }
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.KeyLocalisation().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.KeyLocalisation().ValForcedForUnUmp(),aFp);
        }
        else  anObj.KeyLocalisation().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cNameFilter & anObj)
{
    BinaryDumpInFile(aFp,(int)anObj.FocMm().size());
    for(  std::list< Pt2drSubst >::const_iterator iT=anObj.FocMm().begin();
         iT!=anObj.FocMm().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.Min().IsInit());
    if (anObj.Min().IsInit()) BinaryDumpInFile(aFp,anObj.Min().Val());
    BinaryDumpInFile(aFp,anObj.Max().IsInit());
    if (anObj.Max().IsInit()) BinaryDumpInFile(aFp,anObj.Max().Val());
    BinaryDumpInFile(aFp,anObj.SizeMinFile().IsInit());
    if (anObj.SizeMinFile().IsInit()) BinaryDumpInFile(aFp,anObj.SizeMinFile().Val());
    BinaryDumpInFile(aFp,(int)anObj.KeyExistingFile().size());
    for(  std::list< cKeyExistingFile >::const_iterator iT=anObj.KeyExistingFile().begin();
         iT!=anObj.KeyExistingFile().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.KeyLocalisation().IsInit());
    if (anObj.KeyLocalisation().IsInit()) BinaryDumpInFile(aFp,anObj.KeyLocalisation().Val());
}

cElXMLTree * ToXMLTree(const cNameFilter & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"NameFilter",eXMLBranche);
  for
  (       std::list< Pt2drSubst >::const_iterator it=anObj.FocMm().begin();
      it !=anObj.FocMm().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("FocMm"),(*it))->ReTagThis("FocMm"));
   if (anObj.Min().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Min"),anObj.Min().Val())->ReTagThis("Min"));
   if (anObj.Max().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Max"),anObj.Max().Val())->ReTagThis("Max"));
   if (anObj.SizeMinFile().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SizeMinFile"),anObj.SizeMinFile().Val())->ReTagThis("SizeMinFile"));
  for
  (       std::list< cKeyExistingFile >::const_iterator it=anObj.KeyExistingFile().begin();
      it !=anObj.KeyExistingFile().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("KeyExistingFile"));
   if (anObj.KeyLocalisation().IsInit())
      aRes->AddFils(ToXMLTree(anObj.KeyLocalisation().Val())->ReTagThis("KeyLocalisation"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cNameFilter & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.FocMm(),aTree->GetAll("FocMm",false,1));

   xml_init(anObj.Min(),aTree->Get("Min",1)); //tototo 

   xml_init(anObj.Max(),aTree->Get("Max",1)); //tototo 

   xml_init(anObj.SizeMinFile(),aTree->Get("SizeMinFile",1),int(-1)); //tototo 

   xml_init(anObj.KeyExistingFile(),aTree->GetAll("KeyExistingFile",false,1));

   xml_init(anObj.KeyLocalisation(),aTree->Get("KeyLocalisation",1)); //tototo 
}

std::string  Mangling( cNameFilter *) {return "C0CEB1E81145A1E6FE3F";};


std::string & cBasicAssocNameToName::PatternTransform()
{
   return mPatternTransform;
}

const std::string & cBasicAssocNameToName::PatternTransform()const 
{
   return mPatternTransform;
}


cTplValGesInit< cDataBaseNameTransfo > & cBasicAssocNameToName::NameTransfo()
{
   return mNameTransfo;
}

const cTplValGesInit< cDataBaseNameTransfo > & cBasicAssocNameToName::NameTransfo()const 
{
   return mNameTransfo;
}


cTplValGesInit< std::string > & cBasicAssocNameToName::PatternSelector()
{
   return mPatternSelector;
}

const cTplValGesInit< std::string > & cBasicAssocNameToName::PatternSelector()const 
{
   return mPatternSelector;
}


std::vector< std::string > & cBasicAssocNameToName::CalcName()
{
   return mCalcName;
}

const std::vector< std::string > & cBasicAssocNameToName::CalcName()const 
{
   return mCalcName;
}


cTplValGesInit< std::string > & cBasicAssocNameToName::Separateur()
{
   return mSeparateur;
}

const cTplValGesInit< std::string > & cBasicAssocNameToName::Separateur()const 
{
   return mSeparateur;
}


cTplValGesInit< cNameFilter > & cBasicAssocNameToName::Filter()
{
   return mFilter;
}

const cTplValGesInit< cNameFilter > & cBasicAssocNameToName::Filter()const 
{
   return mFilter;
}

void  BinaryUnDumpFromFile(cBasicAssocNameToName & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.PatternTransform(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NameTransfo().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NameTransfo().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NameTransfo().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.PatternSelector().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.PatternSelector().ValForcedForUnUmp(),aFp);
        }
        else  anObj.PatternSelector().SetNoInit();
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             std::string aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.CalcName().push_back(aVal);
        }
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Separateur().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Separateur().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Separateur().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Filter().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Filter().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Filter().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cBasicAssocNameToName & anObj)
{
    BinaryDumpInFile(aFp,anObj.PatternTransform());
    BinaryDumpInFile(aFp,anObj.NameTransfo().IsInit());
    if (anObj.NameTransfo().IsInit()) BinaryDumpInFile(aFp,anObj.NameTransfo().Val());
    BinaryDumpInFile(aFp,anObj.PatternSelector().IsInit());
    if (anObj.PatternSelector().IsInit()) BinaryDumpInFile(aFp,anObj.PatternSelector().Val());
    BinaryDumpInFile(aFp,(int)anObj.CalcName().size());
    for(  std::vector< std::string >::const_iterator iT=anObj.CalcName().begin();
         iT!=anObj.CalcName().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.Separateur().IsInit());
    if (anObj.Separateur().IsInit()) BinaryDumpInFile(aFp,anObj.Separateur().Val());
    BinaryDumpInFile(aFp,anObj.Filter().IsInit());
    if (anObj.Filter().IsInit()) BinaryDumpInFile(aFp,anObj.Filter().Val());
}

cElXMLTree * ToXMLTree(const cBasicAssocNameToName & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"BasicAssocNameToName",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("PatternTransform"),anObj.PatternTransform())->ReTagThis("PatternTransform"));
   if (anObj.NameTransfo().IsInit())
      aRes->AddFils(ToXMLTree(anObj.NameTransfo().Val())->ReTagThis("NameTransfo"));
   if (anObj.PatternSelector().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PatternSelector"),anObj.PatternSelector().Val())->ReTagThis("PatternSelector"));
  for
  (       std::vector< std::string >::const_iterator it=anObj.CalcName().begin();
      it !=anObj.CalcName().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("CalcName"),(*it))->ReTagThis("CalcName"));
   if (anObj.Separateur().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Separateur"),anObj.Separateur().Val())->ReTagThis("Separateur"));
   if (anObj.Filter().IsInit())
      aRes->AddFils(ToXMLTree(anObj.Filter().Val())->ReTagThis("Filter"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cBasicAssocNameToName & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.PatternTransform(),aTree->Get("PatternTransform",1)); //tototo 

   xml_init(anObj.NameTransfo(),aTree->Get("NameTransfo",1)); //tototo 

   xml_init(anObj.PatternSelector(),aTree->Get("PatternSelector",1)); //tototo 

   xml_init(anObj.CalcName(),aTree->GetAll("CalcName",false,1));

   xml_init(anObj.Separateur(),aTree->Get("Separateur",1),std::string("@")); //tototo 

   xml_init(anObj.Filter(),aTree->Get("Filter",1)); //tototo 
}

std::string  Mangling( cBasicAssocNameToName *) {return "1C1B10FF76A392A3FF3F";};


cTplValGesInit< Pt2di > & cAssocNameToName::Arrite()
{
   return mArrite;
}

const cTplValGesInit< Pt2di > & cAssocNameToName::Arrite()const 
{
   return mArrite;
}


cBasicAssocNameToName & cAssocNameToName::Direct()
{
   return mDirect;
}

const cBasicAssocNameToName & cAssocNameToName::Direct()const 
{
   return mDirect;
}


cTplValGesInit< cBasicAssocNameToName > & cAssocNameToName::Inverse()
{
   return mInverse;
}

const cTplValGesInit< cBasicAssocNameToName > & cAssocNameToName::Inverse()const 
{
   return mInverse;
}


cTplValGesInit< bool > & cAssocNameToName::AutoInverseBySym()
{
   return mAutoInverseBySym;
}

const cTplValGesInit< bool > & cAssocNameToName::AutoInverseBySym()const 
{
   return mAutoInverseBySym;
}

void  BinaryUnDumpFromFile(cAssocNameToName & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Arrite().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Arrite().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Arrite().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.Direct(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Inverse().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Inverse().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Inverse().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.AutoInverseBySym().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.AutoInverseBySym().ValForcedForUnUmp(),aFp);
        }
        else  anObj.AutoInverseBySym().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cAssocNameToName & anObj)
{
    BinaryDumpInFile(aFp,anObj.Arrite().IsInit());
    if (anObj.Arrite().IsInit()) BinaryDumpInFile(aFp,anObj.Arrite().Val());
    BinaryDumpInFile(aFp,anObj.Direct());
    BinaryDumpInFile(aFp,anObj.Inverse().IsInit());
    if (anObj.Inverse().IsInit()) BinaryDumpInFile(aFp,anObj.Inverse().Val());
    BinaryDumpInFile(aFp,anObj.AutoInverseBySym().IsInit());
    if (anObj.AutoInverseBySym().IsInit()) BinaryDumpInFile(aFp,anObj.AutoInverseBySym().Val());
}

cElXMLTree * ToXMLTree(const cAssocNameToName & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"AssocNameToName",eXMLBranche);
   if (anObj.Arrite().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Arrite"),anObj.Arrite().Val())->ReTagThis("Arrite"));
   aRes->AddFils(ToXMLTree(anObj.Direct())->ReTagThis("Direct"));
   if (anObj.Inverse().IsInit())
      aRes->AddFils(ToXMLTree(anObj.Inverse().Val())->ReTagThis("Inverse"));
   if (anObj.AutoInverseBySym().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("AutoInverseBySym"),anObj.AutoInverseBySym().Val())->ReTagThis("AutoInverseBySym"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cAssocNameToName & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Arrite(),aTree->Get("Arrite",1)); //tototo 

   xml_init(anObj.Direct(),aTree->Get("Direct",1)); //tototo 

   xml_init(anObj.Inverse(),aTree->Get("Inverse",1)); //tototo 

   xml_init(anObj.AutoInverseBySym(),aTree->Get("AutoInverseBySym",1),bool(false)); //tototo 
}

std::string  Mangling( cAssocNameToName *) {return "8E3A6A61A8BD2A83FF3F";};


cTplValGesInit< bool > & cSetNameDescriptor::AddDirCur()
{
   return mAddDirCur;
}

const cTplValGesInit< bool > & cSetNameDescriptor::AddDirCur()const 
{
   return mAddDirCur;
}


std::list< std::string > & cSetNameDescriptor::PatternAccepteur()
{
   return mPatternAccepteur;
}

const std::list< std::string > & cSetNameDescriptor::PatternAccepteur()const 
{
   return mPatternAccepteur;
}


std::list< std::string > & cSetNameDescriptor::PatternRefuteur()
{
   return mPatternRefuteur;
}

const std::list< std::string > & cSetNameDescriptor::PatternRefuteur()const 
{
   return mPatternRefuteur;
}


cTplValGesInit< int > & cSetNameDescriptor::NivSubDir()
{
   return mNivSubDir;
}

const cTplValGesInit< int > & cSetNameDescriptor::NivSubDir()const 
{
   return mNivSubDir;
}


cTplValGesInit< bool > & cSetNameDescriptor::NameCompl()
{
   return mNameCompl;
}

const cTplValGesInit< bool > & cSetNameDescriptor::NameCompl()const 
{
   return mNameCompl;
}


cTplValGesInit< std::string > & cSetNameDescriptor::SubDir()
{
   return mSubDir;
}

const cTplValGesInit< std::string > & cSetNameDescriptor::SubDir()const 
{
   return mSubDir;
}


std::list< std::string > & cSetNameDescriptor::Name()
{
   return mName;
}

const std::list< std::string > & cSetNameDescriptor::Name()const 
{
   return mName;
}


cTplValGesInit< std::string > & cSetNameDescriptor::Min()
{
   return mMin;
}

const cTplValGesInit< std::string > & cSetNameDescriptor::Min()const 
{
   return mMin;
}


cTplValGesInit< std::string > & cSetNameDescriptor::Max()
{
   return mMax;
}

const cTplValGesInit< std::string > & cSetNameDescriptor::Max()const 
{
   return mMax;
}


cTplValGesInit< cNameFilter > & cSetNameDescriptor::Filter()
{
   return mFilter;
}

const cTplValGesInit< cNameFilter > & cSetNameDescriptor::Filter()const 
{
   return mFilter;
}

void  BinaryUnDumpFromFile(cSetNameDescriptor & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.AddDirCur().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.AddDirCur().ValForcedForUnUmp(),aFp);
        }
        else  anObj.AddDirCur().SetNoInit();
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             std::string aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.PatternAccepteur().push_back(aVal);
        }
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             std::string aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.PatternRefuteur().push_back(aVal);
        }
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NivSubDir().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NivSubDir().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NivSubDir().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NameCompl().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NameCompl().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NameCompl().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SubDir().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SubDir().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SubDir().SetNoInit();
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             std::string aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.Name().push_back(aVal);
        }
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Min().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Min().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Min().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Max().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Max().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Max().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Filter().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Filter().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Filter().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cSetNameDescriptor & anObj)
{
    BinaryDumpInFile(aFp,anObj.AddDirCur().IsInit());
    if (anObj.AddDirCur().IsInit()) BinaryDumpInFile(aFp,anObj.AddDirCur().Val());
    BinaryDumpInFile(aFp,(int)anObj.PatternAccepteur().size());
    for(  std::list< std::string >::const_iterator iT=anObj.PatternAccepteur().begin();
         iT!=anObj.PatternAccepteur().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,(int)anObj.PatternRefuteur().size());
    for(  std::list< std::string >::const_iterator iT=anObj.PatternRefuteur().begin();
         iT!=anObj.PatternRefuteur().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.NivSubDir().IsInit());
    if (anObj.NivSubDir().IsInit()) BinaryDumpInFile(aFp,anObj.NivSubDir().Val());
    BinaryDumpInFile(aFp,anObj.NameCompl().IsInit());
    if (anObj.NameCompl().IsInit()) BinaryDumpInFile(aFp,anObj.NameCompl().Val());
    BinaryDumpInFile(aFp,anObj.SubDir().IsInit());
    if (anObj.SubDir().IsInit()) BinaryDumpInFile(aFp,anObj.SubDir().Val());
    BinaryDumpInFile(aFp,(int)anObj.Name().size());
    for(  std::list< std::string >::const_iterator iT=anObj.Name().begin();
         iT!=anObj.Name().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.Min().IsInit());
    if (anObj.Min().IsInit()) BinaryDumpInFile(aFp,anObj.Min().Val());
    BinaryDumpInFile(aFp,anObj.Max().IsInit());
    if (anObj.Max().IsInit()) BinaryDumpInFile(aFp,anObj.Max().Val());
    BinaryDumpInFile(aFp,anObj.Filter().IsInit());
    if (anObj.Filter().IsInit()) BinaryDumpInFile(aFp,anObj.Filter().Val());
}

cElXMLTree * ToXMLTree(const cSetNameDescriptor & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"SetNameDescriptor",eXMLBranche);
   if (anObj.AddDirCur().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("AddDirCur"),anObj.AddDirCur().Val())->ReTagThis("AddDirCur"));
  for
  (       std::list< std::string >::const_iterator it=anObj.PatternAccepteur().begin();
      it !=anObj.PatternAccepteur().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("PatternAccepteur"),(*it))->ReTagThis("PatternAccepteur"));
  for
  (       std::list< std::string >::const_iterator it=anObj.PatternRefuteur().begin();
      it !=anObj.PatternRefuteur().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("PatternRefuteur"),(*it))->ReTagThis("PatternRefuteur"));
   if (anObj.NivSubDir().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NivSubDir"),anObj.NivSubDir().Val())->ReTagThis("NivSubDir"));
   if (anObj.NameCompl().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NameCompl"),anObj.NameCompl().Val())->ReTagThis("NameCompl"));
   if (anObj.SubDir().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SubDir"),anObj.SubDir().Val())->ReTagThis("SubDir"));
  for
  (       std::list< std::string >::const_iterator it=anObj.Name().begin();
      it !=anObj.Name().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("Name"),(*it))->ReTagThis("Name"));
   if (anObj.Min().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Min"),anObj.Min().Val())->ReTagThis("Min"));
   if (anObj.Max().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Max"),anObj.Max().Val())->ReTagThis("Max"));
   if (anObj.Filter().IsInit())
      aRes->AddFils(ToXMLTree(anObj.Filter().Val())->ReTagThis("Filter"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cSetNameDescriptor & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.AddDirCur(),aTree->Get("AddDirCur",1),bool(true)); //tototo 

   xml_init(anObj.PatternAccepteur(),aTree->GetAll("PatternAccepteur",false,1));

   xml_init(anObj.PatternRefuteur(),aTree->GetAll("PatternRefuteur",false,1));

   xml_init(anObj.NivSubDir(),aTree->Get("NivSubDir",1),int(1)); //tototo 

   xml_init(anObj.NameCompl(),aTree->Get("NameCompl",1),bool(false)); //tototo 

   xml_init(anObj.SubDir(),aTree->Get("SubDir",1),std::string("")); //tototo 

   xml_init(anObj.Name(),aTree->GetAll("Name",false,1));

   xml_init(anObj.Min(),aTree->Get("Min",1)); //tototo 

   xml_init(anObj.Max(),aTree->Get("Max",1)); //tototo 

   xml_init(anObj.Filter(),aTree->Get("Filter",1)); //tototo 
}

std::string  Mangling( cSetNameDescriptor *) {return "D4AB82B0F59195ABFF3F";};


std::string & cImMatrixStructuration::KeySet()
{
   return mKeySet;
}

const std::string & cImMatrixStructuration::KeySet()const 
{
   return mKeySet;
}


Pt2di & cImMatrixStructuration::Period()
{
   return mPeriod;
}

const Pt2di & cImMatrixStructuration::Period()const 
{
   return mPeriod;
}


bool & cImMatrixStructuration::XCroissants()
{
   return mXCroissants;
}

const bool & cImMatrixStructuration::XCroissants()const 
{
   return mXCroissants;
}


bool & cImMatrixStructuration::YCroissants()
{
   return mYCroissants;
}

const bool & cImMatrixStructuration::YCroissants()const 
{
   return mYCroissants;
}


bool & cImMatrixStructuration::XVarieFirst()
{
   return mXVarieFirst;
}

const bool & cImMatrixStructuration::XVarieFirst()const 
{
   return mXVarieFirst;
}

void  BinaryUnDumpFromFile(cImMatrixStructuration & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.KeySet(),aFp);
    BinaryUnDumpFromFile(anObj.Period(),aFp);
    BinaryUnDumpFromFile(anObj.XCroissants(),aFp);
    BinaryUnDumpFromFile(anObj.YCroissants(),aFp);
    BinaryUnDumpFromFile(anObj.XVarieFirst(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cImMatrixStructuration & anObj)
{
    BinaryDumpInFile(aFp,anObj.KeySet());
    BinaryDumpInFile(aFp,anObj.Period());
    BinaryDumpInFile(aFp,anObj.XCroissants());
    BinaryDumpInFile(aFp,anObj.YCroissants());
    BinaryDumpInFile(aFp,anObj.XVarieFirst());
}

cElXMLTree * ToXMLTree(const cImMatrixStructuration & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ImMatrixStructuration",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("KeySet"),anObj.KeySet())->ReTagThis("KeySet"));
   aRes->AddFils(::ToXMLTree(std::string("Period"),anObj.Period())->ReTagThis("Period"));
   aRes->AddFils(::ToXMLTree(std::string("XCroissants"),anObj.XCroissants())->ReTagThis("XCroissants"));
   aRes->AddFils(::ToXMLTree(std::string("YCroissants"),anObj.YCroissants())->ReTagThis("YCroissants"));
   aRes->AddFils(::ToXMLTree(std::string("XVarieFirst"),anObj.XVarieFirst())->ReTagThis("XVarieFirst"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cImMatrixStructuration & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.KeySet(),aTree->Get("KeySet",1)); //tototo 

   xml_init(anObj.Period(),aTree->Get("Period",1)); //tototo 

   xml_init(anObj.XCroissants(),aTree->Get("XCroissants",1)); //tototo 

   xml_init(anObj.YCroissants(),aTree->Get("YCroissants",1)); //tototo 

   xml_init(anObj.XVarieFirst(),aTree->Get("XVarieFirst",1)); //tototo 
}

std::string  Mangling( cImMatrixStructuration *) {return "70A3ECEE0DE6B78CFD3F";};


std::string & cFiltreEmprise::KeyOri()
{
   return mKeyOri;
}

const std::string & cFiltreEmprise::KeyOri()const 
{
   return mKeyOri;
}


double & cFiltreEmprise::RatioMin()
{
   return mRatioMin;
}

const double & cFiltreEmprise::RatioMin()const 
{
   return mRatioMin;
}


cTplValGesInit< bool > & cFiltreEmprise::MemoFile()
{
   return mMemoFile;
}

const cTplValGesInit< bool > & cFiltreEmprise::MemoFile()const 
{
   return mMemoFile;
}


cTplValGesInit< std::string > & cFiltreEmprise::Tag()
{
   return mTag;
}

const cTplValGesInit< std::string > & cFiltreEmprise::Tag()const 
{
   return mTag;
}

void  BinaryUnDumpFromFile(cFiltreEmprise & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.KeyOri(),aFp);
    BinaryUnDumpFromFile(anObj.RatioMin(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.MemoFile().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.MemoFile().ValForcedForUnUmp(),aFp);
        }
        else  anObj.MemoFile().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Tag().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Tag().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Tag().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cFiltreEmprise & anObj)
{
    BinaryDumpInFile(aFp,anObj.KeyOri());
    BinaryDumpInFile(aFp,anObj.RatioMin());
    BinaryDumpInFile(aFp,anObj.MemoFile().IsInit());
    if (anObj.MemoFile().IsInit()) BinaryDumpInFile(aFp,anObj.MemoFile().Val());
    BinaryDumpInFile(aFp,anObj.Tag().IsInit());
    if (anObj.Tag().IsInit()) BinaryDumpInFile(aFp,anObj.Tag().Val());
}

cElXMLTree * ToXMLTree(const cFiltreEmprise & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"FiltreEmprise",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("KeyOri"),anObj.KeyOri())->ReTagThis("KeyOri"));
   aRes->AddFils(::ToXMLTree(std::string("RatioMin"),anObj.RatioMin())->ReTagThis("RatioMin"));
   if (anObj.MemoFile().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("MemoFile"),anObj.MemoFile().Val())->ReTagThis("MemoFile"));
   if (anObj.Tag().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Tag"),anObj.Tag().Val())->ReTagThis("Tag"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cFiltreEmprise & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.KeyOri(),aTree->Get("KeyOri",1)); //tototo 

   xml_init(anObj.RatioMin(),aTree->Get("RatioMin",1)); //tototo 

   xml_init(anObj.MemoFile(),aTree->Get("MemoFile",1),bool(true)); //tototo 

   xml_init(anObj.Tag(),aTree->Get("Tag",1),std::string("OrientationConique")); //tototo 
}

std::string  Mangling( cFiltreEmprise *) {return "192FD52BB5C10085FF3F";};


std::string & cFiltreByRelSsEch::KeySet()
{
   return mKeySet;
}

const std::string & cFiltreByRelSsEch::KeySet()const 
{
   return mKeySet;
}


std::string & cFiltreByRelSsEch::KeyAssocCple()
{
   return mKeyAssocCple;
}

const std::string & cFiltreByRelSsEch::KeyAssocCple()const 
{
   return mKeyAssocCple;
}


IntSubst & cFiltreByRelSsEch::SeuilBasNbPts()
{
   return mSeuilBasNbPts;
}

const IntSubst & cFiltreByRelSsEch::SeuilBasNbPts()const 
{
   return mSeuilBasNbPts;
}


IntSubst & cFiltreByRelSsEch::SeuilHautNbPts()
{
   return mSeuilHautNbPts;
}

const IntSubst & cFiltreByRelSsEch::SeuilHautNbPts()const 
{
   return mSeuilHautNbPts;
}


IntSubst & cFiltreByRelSsEch::NbMinCple()
{
   return mNbMinCple;
}

const IntSubst & cFiltreByRelSsEch::NbMinCple()const 
{
   return mNbMinCple;
}

void  BinaryUnDumpFromFile(cFiltreByRelSsEch & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.KeySet(),aFp);
    BinaryUnDumpFromFile(anObj.KeyAssocCple(),aFp);
    BinaryUnDumpFromFile(anObj.SeuilBasNbPts(),aFp);
    BinaryUnDumpFromFile(anObj.SeuilHautNbPts(),aFp);
    BinaryUnDumpFromFile(anObj.NbMinCple(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cFiltreByRelSsEch & anObj)
{
    BinaryDumpInFile(aFp,anObj.KeySet());
    BinaryDumpInFile(aFp,anObj.KeyAssocCple());
    BinaryDumpInFile(aFp,anObj.SeuilBasNbPts());
    BinaryDumpInFile(aFp,anObj.SeuilHautNbPts());
    BinaryDumpInFile(aFp,anObj.NbMinCple());
}

cElXMLTree * ToXMLTree(const cFiltreByRelSsEch & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"FiltreByRelSsEch",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("KeySet"),anObj.KeySet())->ReTagThis("KeySet"));
   aRes->AddFils(::ToXMLTree(std::string("KeyAssocCple"),anObj.KeyAssocCple())->ReTagThis("KeyAssocCple"));
   aRes->AddFils(::ToXMLTree(std::string("SeuilBasNbPts"),anObj.SeuilBasNbPts())->ReTagThis("SeuilBasNbPts"));
   aRes->AddFils(::ToXMLTree(std::string("SeuilHautNbPts"),anObj.SeuilHautNbPts())->ReTagThis("SeuilHautNbPts"));
   aRes->AddFils(::ToXMLTree(std::string("NbMinCple"),anObj.NbMinCple())->ReTagThis("NbMinCple"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cFiltreByRelSsEch & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.KeySet(),aTree->Get("KeySet",1)); //tototo 

   xml_init(anObj.KeyAssocCple(),aTree->Get("KeyAssocCple",1)); //tototo 

   xml_init(anObj.SeuilBasNbPts(),aTree->Get("SeuilBasNbPts",1)); //tototo 

   xml_init(anObj.SeuilHautNbPts(),aTree->Get("SeuilHautNbPts",1)); //tototo 

   xml_init(anObj.NbMinCple(),aTree->Get("NbMinCple",1)); //tototo 
}

std::string  Mangling( cFiltreByRelSsEch *) {return "14C6E1727AFFEFA1FE3F";};


cTplValGesInit< std::string > & cFiltreDeRelationOrient::KeyEquiv()
{
   return mKeyEquiv;
}

const cTplValGesInit< std::string > & cFiltreDeRelationOrient::KeyEquiv()const 
{
   return mKeyEquiv;
}


std::string & cFiltreDeRelationOrient::KeyOri()
{
   return FiltreEmprise().Val().KeyOri();
}

const std::string & cFiltreDeRelationOrient::KeyOri()const 
{
   return FiltreEmprise().Val().KeyOri();
}


double & cFiltreDeRelationOrient::RatioMin()
{
   return FiltreEmprise().Val().RatioMin();
}

const double & cFiltreDeRelationOrient::RatioMin()const 
{
   return FiltreEmprise().Val().RatioMin();
}


cTplValGesInit< bool > & cFiltreDeRelationOrient::MemoFile()
{
   return FiltreEmprise().Val().MemoFile();
}

const cTplValGesInit< bool > & cFiltreDeRelationOrient::MemoFile()const 
{
   return FiltreEmprise().Val().MemoFile();
}


cTplValGesInit< std::string > & cFiltreDeRelationOrient::Tag()
{
   return FiltreEmprise().Val().Tag();
}

const cTplValGesInit< std::string > & cFiltreDeRelationOrient::Tag()const 
{
   return FiltreEmprise().Val().Tag();
}


cTplValGesInit< cFiltreEmprise > & cFiltreDeRelationOrient::FiltreEmprise()
{
   return mFiltreEmprise;
}

const cTplValGesInit< cFiltreEmprise > & cFiltreDeRelationOrient::FiltreEmprise()const 
{
   return mFiltreEmprise;
}


cTplValGesInit< std::string > & cFiltreDeRelationOrient::FiltreAdjMatrix()
{
   return mFiltreAdjMatrix;
}

const cTplValGesInit< std::string > & cFiltreDeRelationOrient::FiltreAdjMatrix()const 
{
   return mFiltreAdjMatrix;
}


cTplValGesInit< Pt2di > & cFiltreDeRelationOrient::EcartFiltreMatr()
{
   return mEcartFiltreMatr;
}

const cTplValGesInit< Pt2di > & cFiltreDeRelationOrient::EcartFiltreMatr()const 
{
   return mEcartFiltreMatr;
}


std::string & cFiltreDeRelationOrient::KeySet()
{
   return FiltreByRelSsEch().Val().KeySet();
}

const std::string & cFiltreDeRelationOrient::KeySet()const 
{
   return FiltreByRelSsEch().Val().KeySet();
}


std::string & cFiltreDeRelationOrient::KeyAssocCple()
{
   return FiltreByRelSsEch().Val().KeyAssocCple();
}

const std::string & cFiltreDeRelationOrient::KeyAssocCple()const 
{
   return FiltreByRelSsEch().Val().KeyAssocCple();
}


IntSubst & cFiltreDeRelationOrient::SeuilBasNbPts()
{
   return FiltreByRelSsEch().Val().SeuilBasNbPts();
}

const IntSubst & cFiltreDeRelationOrient::SeuilBasNbPts()const 
{
   return FiltreByRelSsEch().Val().SeuilBasNbPts();
}


IntSubst & cFiltreDeRelationOrient::SeuilHautNbPts()
{
   return FiltreByRelSsEch().Val().SeuilHautNbPts();
}

const IntSubst & cFiltreDeRelationOrient::SeuilHautNbPts()const 
{
   return FiltreByRelSsEch().Val().SeuilHautNbPts();
}


IntSubst & cFiltreDeRelationOrient::NbMinCple()
{
   return FiltreByRelSsEch().Val().NbMinCple();
}

const IntSubst & cFiltreDeRelationOrient::NbMinCple()const 
{
   return FiltreByRelSsEch().Val().NbMinCple();
}


cTplValGesInit< cFiltreByRelSsEch > & cFiltreDeRelationOrient::FiltreByRelSsEch()
{
   return mFiltreByRelSsEch;
}

const cTplValGesInit< cFiltreByRelSsEch > & cFiltreDeRelationOrient::FiltreByRelSsEch()const 
{
   return mFiltreByRelSsEch;
}

void  BinaryUnDumpFromFile(cFiltreDeRelationOrient & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.KeyEquiv().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.KeyEquiv().ValForcedForUnUmp(),aFp);
        }
        else  anObj.KeyEquiv().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.FiltreEmprise().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.FiltreEmprise().ValForcedForUnUmp(),aFp);
        }
        else  anObj.FiltreEmprise().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.FiltreAdjMatrix().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.FiltreAdjMatrix().ValForcedForUnUmp(),aFp);
        }
        else  anObj.FiltreAdjMatrix().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.EcartFiltreMatr().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.EcartFiltreMatr().ValForcedForUnUmp(),aFp);
        }
        else  anObj.EcartFiltreMatr().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.FiltreByRelSsEch().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.FiltreByRelSsEch().ValForcedForUnUmp(),aFp);
        }
        else  anObj.FiltreByRelSsEch().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cFiltreDeRelationOrient & anObj)
{
    BinaryDumpInFile(aFp,anObj.KeyEquiv().IsInit());
    if (anObj.KeyEquiv().IsInit()) BinaryDumpInFile(aFp,anObj.KeyEquiv().Val());
    BinaryDumpInFile(aFp,anObj.FiltreEmprise().IsInit());
    if (anObj.FiltreEmprise().IsInit()) BinaryDumpInFile(aFp,anObj.FiltreEmprise().Val());
    BinaryDumpInFile(aFp,anObj.FiltreAdjMatrix().IsInit());
    if (anObj.FiltreAdjMatrix().IsInit()) BinaryDumpInFile(aFp,anObj.FiltreAdjMatrix().Val());
    BinaryDumpInFile(aFp,anObj.EcartFiltreMatr().IsInit());
    if (anObj.EcartFiltreMatr().IsInit()) BinaryDumpInFile(aFp,anObj.EcartFiltreMatr().Val());
    BinaryDumpInFile(aFp,anObj.FiltreByRelSsEch().IsInit());
    if (anObj.FiltreByRelSsEch().IsInit()) BinaryDumpInFile(aFp,anObj.FiltreByRelSsEch().Val());
}

cElXMLTree * ToXMLTree(const cFiltreDeRelationOrient & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"FiltreDeRelationOrient",eXMLBranche);
   if (anObj.KeyEquiv().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("KeyEquiv"),anObj.KeyEquiv().Val())->ReTagThis("KeyEquiv"));
   if (anObj.FiltreEmprise().IsInit())
      aRes->AddFils(ToXMLTree(anObj.FiltreEmprise().Val())->ReTagThis("FiltreEmprise"));
   if (anObj.FiltreAdjMatrix().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("FiltreAdjMatrix"),anObj.FiltreAdjMatrix().Val())->ReTagThis("FiltreAdjMatrix"));
   if (anObj.EcartFiltreMatr().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("EcartFiltreMatr"),anObj.EcartFiltreMatr().Val())->ReTagThis("EcartFiltreMatr"));
   if (anObj.FiltreByRelSsEch().IsInit())
      aRes->AddFils(ToXMLTree(anObj.FiltreByRelSsEch().Val())->ReTagThis("FiltreByRelSsEch"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cFiltreDeRelationOrient & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.KeyEquiv(),aTree->Get("KeyEquiv",1)); //tototo 

   xml_init(anObj.FiltreEmprise(),aTree->Get("FiltreEmprise",1)); //tototo 

   xml_init(anObj.FiltreAdjMatrix(),aTree->Get("FiltreAdjMatrix",1)); //tototo 

   xml_init(anObj.EcartFiltreMatr(),aTree->Get("EcartFiltreMatr",1),Pt2di(1,1)); //tototo 

   xml_init(anObj.FiltreByRelSsEch(),aTree->Get("FiltreByRelSsEch",1)); //tototo 
}

std::string  Mangling( cFiltreDeRelationOrient *) {return "9A2EDB33175707DAFDBF";};


std::vector< cCpleString > & cSauvegardeNamedRel::Cple()
{
   return mCple;
}

const std::vector< cCpleString > & cSauvegardeNamedRel::Cple()const 
{
   return mCple;
}

void  BinaryUnDumpFromFile(cSauvegardeNamedRel & anObj,ELISE_fp & aFp)
{
   { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cCpleString aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.Cple().push_back(aVal);
        }
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cSauvegardeNamedRel & anObj)
{
    BinaryDumpInFile(aFp,(int)anObj.Cple().size());
    for(  std::vector< cCpleString >::const_iterator iT=anObj.Cple().begin();
         iT!=anObj.Cple().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
}

cElXMLTree * ToXMLTree(const cSauvegardeNamedRel & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"SauvegardeNamedRel",eXMLBranche);
  for
  (       std::vector< cCpleString >::const_iterator it=anObj.Cple().begin();
      it !=anObj.Cple().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("Cple"),(*it))->ReTagThis("Cple"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cSauvegardeNamedRel & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Cple(),aTree->GetAll("Cple",false,1));
}

std::string  Mangling( cSauvegardeNamedRel *) {return "987C7A7998C3B99FFB3F";};


std::list< std::string > & cSauvegardeSetString::Name()
{
   return mName;
}

const std::list< std::string > & cSauvegardeSetString::Name()const 
{
   return mName;
}

void  BinaryUnDumpFromFile(cSauvegardeSetString & anObj,ELISE_fp & aFp)
{
   { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             std::string aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.Name().push_back(aVal);
        }
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cSauvegardeSetString & anObj)
{
    BinaryDumpInFile(aFp,(int)anObj.Name().size());
    for(  std::list< std::string >::const_iterator iT=anObj.Name().begin();
         iT!=anObj.Name().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
}

cElXMLTree * ToXMLTree(const cSauvegardeSetString & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"SauvegardeSetString",eXMLBranche);
  for
  (       std::list< std::string >::const_iterator it=anObj.Name().begin();
      it !=anObj.Name().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("Name"),(*it))->ReTagThis("Name"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cSauvegardeSetString & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Name(),aTree->GetAll("Name",false,1));
}

std::string  Mangling( cSauvegardeSetString *) {return "901BB088A58C2CCEFABF";};


std::string & cClassEquivDescripteur::KeySet()
{
   return mKeySet;
}

const std::string & cClassEquivDescripteur::KeySet()const 
{
   return mKeySet;
}


std::string & cClassEquivDescripteur::KeyAssocRep()
{
   return mKeyAssocRep;
}

const std::string & cClassEquivDescripteur::KeyAssocRep()const 
{
   return mKeyAssocRep;
}


std::string & cClassEquivDescripteur::KeyClass()
{
   return mKeyClass;
}

const std::string & cClassEquivDescripteur::KeyClass()const 
{
   return mKeyClass;
}

void  BinaryUnDumpFromFile(cClassEquivDescripteur & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.KeySet(),aFp);
    BinaryUnDumpFromFile(anObj.KeyAssocRep(),aFp);
    BinaryUnDumpFromFile(anObj.KeyClass(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cClassEquivDescripteur & anObj)
{
    BinaryDumpInFile(aFp,anObj.KeySet());
    BinaryDumpInFile(aFp,anObj.KeyAssocRep());
    BinaryDumpInFile(aFp,anObj.KeyClass());
}

cElXMLTree * ToXMLTree(const cClassEquivDescripteur & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ClassEquivDescripteur",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("KeySet"),anObj.KeySet())->ReTagThis("KeySet"));
   aRes->AddFils(::ToXMLTree(std::string("KeyAssocRep"),anObj.KeyAssocRep())->ReTagThis("KeyAssocRep"));
   aRes->AddFils(::ToXMLTree(std::string("KeyClass"),anObj.KeyClass())->ReTagThis("KeyClass"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cClassEquivDescripteur & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.KeySet(),aTree->Get("KeySet",1)); //tototo 

   xml_init(anObj.KeyAssocRep(),aTree->Get("KeyAssocRep",1)); //tototo 

   xml_init(anObj.KeyClass(),aTree->Get("KeyClass",1)); //tototo 
}

std::string  Mangling( cClassEquivDescripteur *) {return "1F11AF7EF1FDBAE1FE3F";};


std::vector<std::string> & cOneSpecDelta::Soms()
{
   return mSoms;
}

const std::vector<std::string> & cOneSpecDelta::Soms()const 
{
   return mSoms;
}


std::vector<int>  & cOneSpecDelta::Delta()
{
   return mDelta;
}

const std::vector<int>  & cOneSpecDelta::Delta()const 
{
   return mDelta;
}

void  BinaryUnDumpFromFile(cOneSpecDelta & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Soms(),aFp);
    BinaryUnDumpFromFile(anObj.Delta(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cOneSpecDelta & anObj)
{
    BinaryDumpInFile(aFp,anObj.Soms());
    BinaryDumpInFile(aFp,anObj.Delta());
}

cElXMLTree * ToXMLTree(const cOneSpecDelta & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"OneSpecDelta",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Soms"),anObj.Soms())->ReTagThis("Soms"));
   aRes->AddFils(::ToXMLTree(std::string("Delta"),anObj.Delta())->ReTagThis("Delta"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cOneSpecDelta & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Soms(),aTree->Get("Soms",1)); //tototo 

   xml_init(anObj.Delta(),aTree->Get("Delta",1)); //tototo 
}

std::string  Mangling( cOneSpecDelta *) {return "C00EE64508F9FF88F8BF";};


std::string & cGrByDelta::KeySet()
{
   return mKeySet;
}

const std::string & cGrByDelta::KeySet()const 
{
   return mKeySet;
}


std::list< cOneSpecDelta > & cGrByDelta::OneSpecDelta()
{
   return mOneSpecDelta;
}

const std::list< cOneSpecDelta > & cGrByDelta::OneSpecDelta()const 
{
   return mOneSpecDelta;
}

void  BinaryUnDumpFromFile(cGrByDelta & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.KeySet(),aFp);
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cOneSpecDelta aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.OneSpecDelta().push_back(aVal);
        }
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cGrByDelta & anObj)
{
    BinaryDumpInFile(aFp,anObj.KeySet());
    BinaryDumpInFile(aFp,(int)anObj.OneSpecDelta().size());
    for(  std::list< cOneSpecDelta >::const_iterator iT=anObj.OneSpecDelta().begin();
         iT!=anObj.OneSpecDelta().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
}

cElXMLTree * ToXMLTree(const cGrByDelta & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"GrByDelta",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("KeySet"),anObj.KeySet())->ReTagThis("KeySet"));
  for
  (       std::list< cOneSpecDelta >::const_iterator it=anObj.OneSpecDelta().begin();
      it !=anObj.OneSpecDelta().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("OneSpecDelta"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cGrByDelta & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.KeySet(),aTree->Get("KeySet",1)); //tototo 

   xml_init(anObj.OneSpecDelta(),aTree->GetAll("OneSpecDelta",false,1));
}

std::string  Mangling( cGrByDelta *) {return "8041796FAC7C68DCF93F";};


std::list< cCpleString > & cRelByGrapheExpl::Cples()
{
   return mCples;
}

const std::list< cCpleString > & cRelByGrapheExpl::Cples()const 
{
   return mCples;
}


std::list< std::vector<std::string> > & cRelByGrapheExpl::CpleSymWithFirt()
{
   return mCpleSymWithFirt;
}

const std::list< std::vector<std::string> > & cRelByGrapheExpl::CpleSymWithFirt()const 
{
   return mCpleSymWithFirt;
}


std::list< std::vector<std::string> > & cRelByGrapheExpl::ProdCartesien()
{
   return mProdCartesien;
}

const std::list< std::vector<std::string> > & cRelByGrapheExpl::ProdCartesien()const 
{
   return mProdCartesien;
}


cTplValGesInit< std::string > & cRelByGrapheExpl::Prefix2Name()
{
   return mPrefix2Name;
}

const cTplValGesInit< std::string > & cRelByGrapheExpl::Prefix2Name()const 
{
   return mPrefix2Name;
}


cTplValGesInit< std::string > & cRelByGrapheExpl::Postfix2Name()
{
   return mPostfix2Name;
}

const cTplValGesInit< std::string > & cRelByGrapheExpl::Postfix2Name()const 
{
   return mPostfix2Name;
}


std::list< cGrByDelta > & cRelByGrapheExpl::GrByDelta()
{
   return mGrByDelta;
}

const std::list< cGrByDelta > & cRelByGrapheExpl::GrByDelta()const 
{
   return mGrByDelta;
}

void  BinaryUnDumpFromFile(cRelByGrapheExpl & anObj,ELISE_fp & aFp)
{
   { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cCpleString aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.Cples().push_back(aVal);
        }
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             std::vector<std::string> aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.CpleSymWithFirt().push_back(aVal);
        }
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             std::vector<std::string> aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.ProdCartesien().push_back(aVal);
        }
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Prefix2Name().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Prefix2Name().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Prefix2Name().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Postfix2Name().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Postfix2Name().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Postfix2Name().SetNoInit();
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cGrByDelta aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.GrByDelta().push_back(aVal);
        }
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cRelByGrapheExpl & anObj)
{
    BinaryDumpInFile(aFp,(int)anObj.Cples().size());
    for(  std::list< cCpleString >::const_iterator iT=anObj.Cples().begin();
         iT!=anObj.Cples().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,(int)anObj.CpleSymWithFirt().size());
    for(  std::list< std::vector<std::string> >::const_iterator iT=anObj.CpleSymWithFirt().begin();
         iT!=anObj.CpleSymWithFirt().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,(int)anObj.ProdCartesien().size());
    for(  std::list< std::vector<std::string> >::const_iterator iT=anObj.ProdCartesien().begin();
         iT!=anObj.ProdCartesien().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.Prefix2Name().IsInit());
    if (anObj.Prefix2Name().IsInit()) BinaryDumpInFile(aFp,anObj.Prefix2Name().Val());
    BinaryDumpInFile(aFp,anObj.Postfix2Name().IsInit());
    if (anObj.Postfix2Name().IsInit()) BinaryDumpInFile(aFp,anObj.Postfix2Name().Val());
    BinaryDumpInFile(aFp,(int)anObj.GrByDelta().size());
    for(  std::list< cGrByDelta >::const_iterator iT=anObj.GrByDelta().begin();
         iT!=anObj.GrByDelta().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
}

cElXMLTree * ToXMLTree(const cRelByGrapheExpl & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"RelByGrapheExpl",eXMLBranche);
  for
  (       std::list< cCpleString >::const_iterator it=anObj.Cples().begin();
      it !=anObj.Cples().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("Cples"),(*it))->ReTagThis("Cples"));
  for
  (       std::list< std::vector<std::string> >::const_iterator it=anObj.CpleSymWithFirt().begin();
      it !=anObj.CpleSymWithFirt().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("CpleSymWithFirt"),(*it))->ReTagThis("CpleSymWithFirt"));
  for
  (       std::list< std::vector<std::string> >::const_iterator it=anObj.ProdCartesien().begin();
      it !=anObj.ProdCartesien().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("ProdCartesien"),(*it))->ReTagThis("ProdCartesien"));
   if (anObj.Prefix2Name().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Prefix2Name"),anObj.Prefix2Name().Val())->ReTagThis("Prefix2Name"));
   if (anObj.Postfix2Name().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Postfix2Name"),anObj.Postfix2Name().Val())->ReTagThis("Postfix2Name"));
  for
  (       std::list< cGrByDelta >::const_iterator it=anObj.GrByDelta().begin();
      it !=anObj.GrByDelta().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("GrByDelta"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cRelByGrapheExpl & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Cples(),aTree->GetAll("Cples",false,1));

   xml_init(anObj.CpleSymWithFirt(),aTree->GetAll("CpleSymWithFirt",false,1));

   xml_init(anObj.ProdCartesien(),aTree->GetAll("ProdCartesien",false,1));

   xml_init(anObj.Prefix2Name(),aTree->Get("Prefix2Name",1)); //tototo 

   xml_init(anObj.Postfix2Name(),aTree->Get("Postfix2Name",1)); //tototo 

   xml_init(anObj.GrByDelta(),aTree->GetAll("GrByDelta",false,1));
}

std::string  Mangling( cRelByGrapheExpl *) {return "56DF2D953E59E2C5FE3F";};


std::vector< std::string > & cByAdjDeGroupes::KeySets()
{
   return mKeySets;
}

const std::vector< std::string > & cByAdjDeGroupes::KeySets()const 
{
   return mKeySets;
}


int & cByAdjDeGroupes::DeltaMin()
{
   return mDeltaMin;
}

const int & cByAdjDeGroupes::DeltaMin()const 
{
   return mDeltaMin;
}


int & cByAdjDeGroupes::DeltaMax()
{
   return mDeltaMax;
}

const int & cByAdjDeGroupes::DeltaMax()const 
{
   return mDeltaMax;
}

void  BinaryUnDumpFromFile(cByAdjDeGroupes & anObj,ELISE_fp & aFp)
{
   { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             std::string aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.KeySets().push_back(aVal);
        }
  } ;
    BinaryUnDumpFromFile(anObj.DeltaMin(),aFp);
    BinaryUnDumpFromFile(anObj.DeltaMax(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cByAdjDeGroupes & anObj)
{
    BinaryDumpInFile(aFp,(int)anObj.KeySets().size());
    for(  std::vector< std::string >::const_iterator iT=anObj.KeySets().begin();
         iT!=anObj.KeySets().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.DeltaMin());
    BinaryDumpInFile(aFp,anObj.DeltaMax());
}

cElXMLTree * ToXMLTree(const cByAdjDeGroupes & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ByAdjDeGroupes",eXMLBranche);
  for
  (       std::vector< std::string >::const_iterator it=anObj.KeySets().begin();
      it !=anObj.KeySets().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("KeySets"),(*it))->ReTagThis("KeySets"));
   aRes->AddFils(::ToXMLTree(std::string("DeltaMin"),anObj.DeltaMin())->ReTagThis("DeltaMin"));
   aRes->AddFils(::ToXMLTree(std::string("DeltaMax"),anObj.DeltaMax())->ReTagThis("DeltaMax"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cByAdjDeGroupes & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.KeySets(),aTree->GetAll("KeySets",false,1));

   xml_init(anObj.DeltaMin(),aTree->Get("DeltaMin",1)); //tototo 

   xml_init(anObj.DeltaMax(),aTree->Get("DeltaMax",1)); //tototo 
}

std::string  Mangling( cByAdjDeGroupes *) {return "274C13F0435BCDBEFF3F";};


std::list< cCpleString > & cByGroupesDImages::CplesKey()
{
   return mCplesKey;
}

const std::list< cCpleString > & cByGroupesDImages::CplesKey()const 
{
   return mCplesKey;
}


std::list< cByAdjDeGroupes > & cByGroupesDImages::ByAdjDeGroupes()
{
   return mByAdjDeGroupes;
}

const std::list< cByAdjDeGroupes > & cByGroupesDImages::ByAdjDeGroupes()const 
{
   return mByAdjDeGroupes;
}


cTplValGesInit< cFiltreDeRelationOrient > & cByGroupesDImages::Filtre()
{
   return mFiltre;
}

const cTplValGesInit< cFiltreDeRelationOrient > & cByGroupesDImages::Filtre()const 
{
   return mFiltre;
}


cTplValGesInit< bool > & cByGroupesDImages::Sym()
{
   return mSym;
}

const cTplValGesInit< bool > & cByGroupesDImages::Sym()const 
{
   return mSym;
}


cTplValGesInit< bool > & cByGroupesDImages::Reflexif()
{
   return mReflexif;
}

const cTplValGesInit< bool > & cByGroupesDImages::Reflexif()const 
{
   return mReflexif;
}

void  BinaryUnDumpFromFile(cByGroupesDImages & anObj,ELISE_fp & aFp)
{
   { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cCpleString aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.CplesKey().push_back(aVal);
        }
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cByAdjDeGroupes aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.ByAdjDeGroupes().push_back(aVal);
        }
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Filtre().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Filtre().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Filtre().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Sym().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Sym().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Sym().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Reflexif().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Reflexif().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Reflexif().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cByGroupesDImages & anObj)
{
    BinaryDumpInFile(aFp,(int)anObj.CplesKey().size());
    for(  std::list< cCpleString >::const_iterator iT=anObj.CplesKey().begin();
         iT!=anObj.CplesKey().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,(int)anObj.ByAdjDeGroupes().size());
    for(  std::list< cByAdjDeGroupes >::const_iterator iT=anObj.ByAdjDeGroupes().begin();
         iT!=anObj.ByAdjDeGroupes().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.Filtre().IsInit());
    if (anObj.Filtre().IsInit()) BinaryDumpInFile(aFp,anObj.Filtre().Val());
    BinaryDumpInFile(aFp,anObj.Sym().IsInit());
    if (anObj.Sym().IsInit()) BinaryDumpInFile(aFp,anObj.Sym().Val());
    BinaryDumpInFile(aFp,anObj.Reflexif().IsInit());
    if (anObj.Reflexif().IsInit()) BinaryDumpInFile(aFp,anObj.Reflexif().Val());
}

cElXMLTree * ToXMLTree(const cByGroupesDImages & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ByGroupesDImages",eXMLBranche);
  for
  (       std::list< cCpleString >::const_iterator it=anObj.CplesKey().begin();
      it !=anObj.CplesKey().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("CplesKey"),(*it))->ReTagThis("CplesKey"));
  for
  (       std::list< cByAdjDeGroupes >::const_iterator it=anObj.ByAdjDeGroupes().begin();
      it !=anObj.ByAdjDeGroupes().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("ByAdjDeGroupes"));
   if (anObj.Filtre().IsInit())
      aRes->AddFils(ToXMLTree(anObj.Filtre().Val())->ReTagThis("Filtre"));
   if (anObj.Sym().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Sym"),anObj.Sym().Val())->ReTagThis("Sym"));
   if (anObj.Reflexif().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Reflexif"),anObj.Reflexif().Val())->ReTagThis("Reflexif"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cByGroupesDImages & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.CplesKey(),aTree->GetAll("CplesKey",false,1));

   xml_init(anObj.ByAdjDeGroupes(),aTree->GetAll("ByAdjDeGroupes",false,1));

   xml_init(anObj.Filtre(),aTree->Get("Filtre",1)); //tototo 

   xml_init(anObj.Sym(),aTree->Get("Sym",1),bool(true)); //tototo 

   xml_init(anObj.Reflexif(),aTree->Get("Reflexif",1),bool(false)); //tototo 
}

std::string  Mangling( cByGroupesDImages *) {return "608932C7B599E392FF3F";};


cTplValGesInit< double > & cFiltreDelaunay::DMaxDelaunay()
{
   return mDMaxDelaunay;
}

const cTplValGesInit< double > & cFiltreDelaunay::DMaxDelaunay()const 
{
   return mDMaxDelaunay;
}

void  BinaryUnDumpFromFile(cFiltreDelaunay & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.DMaxDelaunay().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.DMaxDelaunay().ValForcedForUnUmp(),aFp);
        }
        else  anObj.DMaxDelaunay().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cFiltreDelaunay & anObj)
{
    BinaryDumpInFile(aFp,anObj.DMaxDelaunay().IsInit());
    if (anObj.DMaxDelaunay().IsInit()) BinaryDumpInFile(aFp,anObj.DMaxDelaunay().Val());
}

cElXMLTree * ToXMLTree(const cFiltreDelaunay & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"FiltreDelaunay",eXMLBranche);
   if (anObj.DMaxDelaunay().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DMaxDelaunay"),anObj.DMaxDelaunay().Val())->ReTagThis("DMaxDelaunay"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cFiltreDelaunay & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.DMaxDelaunay(),aTree->Get("DMaxDelaunay",1),double(1e9)); //tototo 
}

std::string  Mangling( cFiltreDelaunay *) {return "929750EB790DE990FDBF";};


double & cFiltreDist::DistMax()
{
   return mDistMax;
}

const double & cFiltreDist::DistMax()const 
{
   return mDistMax;
}

void  BinaryUnDumpFromFile(cFiltreDist & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.DistMax(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cFiltreDist & anObj)
{
    BinaryDumpInFile(aFp,anObj.DistMax());
}

cElXMLTree * ToXMLTree(const cFiltreDist & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"FiltreDist",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("DistMax"),anObj.DistMax())->ReTagThis("DistMax"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cFiltreDist & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.DistMax(),aTree->Get("DistMax",1)); //tototo 
}

std::string  Mangling( cFiltreDist *) {return "6084B177BBF432A1FF3F";};


cTplValGesInit< double > & cModeFiltreSpatial::DMaxDelaunay()
{
   return FiltreDelaunay().Val().DMaxDelaunay();
}

const cTplValGesInit< double > & cModeFiltreSpatial::DMaxDelaunay()const 
{
   return FiltreDelaunay().Val().DMaxDelaunay();
}


cTplValGesInit< cFiltreDelaunay > & cModeFiltreSpatial::FiltreDelaunay()
{
   return mFiltreDelaunay;
}

const cTplValGesInit< cFiltreDelaunay > & cModeFiltreSpatial::FiltreDelaunay()const 
{
   return mFiltreDelaunay;
}


double & cModeFiltreSpatial::DistMax()
{
   return FiltreDist().Val().DistMax();
}

const double & cModeFiltreSpatial::DistMax()const 
{
   return FiltreDist().Val().DistMax();
}


cTplValGesInit< cFiltreDist > & cModeFiltreSpatial::FiltreDist()
{
   return mFiltreDist;
}

const cTplValGesInit< cFiltreDist > & cModeFiltreSpatial::FiltreDist()const 
{
   return mFiltreDist;
}

void  BinaryUnDumpFromFile(cModeFiltreSpatial & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.FiltreDelaunay().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.FiltreDelaunay().ValForcedForUnUmp(),aFp);
        }
        else  anObj.FiltreDelaunay().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.FiltreDist().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.FiltreDist().ValForcedForUnUmp(),aFp);
        }
        else  anObj.FiltreDist().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cModeFiltreSpatial & anObj)
{
    BinaryDumpInFile(aFp,anObj.FiltreDelaunay().IsInit());
    if (anObj.FiltreDelaunay().IsInit()) BinaryDumpInFile(aFp,anObj.FiltreDelaunay().Val());
    BinaryDumpInFile(aFp,anObj.FiltreDist().IsInit());
    if (anObj.FiltreDist().IsInit()) BinaryDumpInFile(aFp,anObj.FiltreDist().Val());
}

cElXMLTree * ToXMLTree(const cModeFiltreSpatial & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ModeFiltreSpatial",eXMLBranche);
   if (anObj.FiltreDelaunay().IsInit())
      aRes->AddFils(ToXMLTree(anObj.FiltreDelaunay().Val())->ReTagThis("FiltreDelaunay"));
   if (anObj.FiltreDist().IsInit())
      aRes->AddFils(ToXMLTree(anObj.FiltreDist().Val())->ReTagThis("FiltreDist"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cModeFiltreSpatial & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.FiltreDelaunay(),aTree->Get("FiltreDelaunay",1)); //tototo 

   xml_init(anObj.FiltreDist(),aTree->Get("FiltreDist",1)); //tototo 
}

std::string  Mangling( cModeFiltreSpatial *) {return "47C740F30CAB3FDAFE3F";};


cTplValGesInit< std::string > & cByFiltreSpatial::ByFileTrajecto()
{
   return mByFileTrajecto;
}

const cTplValGesInit< std::string > & cByFiltreSpatial::ByFileTrajecto()const 
{
   return mByFileTrajecto;
}


std::string & cByFiltreSpatial::KeyOri()
{
   return mKeyOri;
}

const std::string & cByFiltreSpatial::KeyOri()const 
{
   return mKeyOri;
}


std::string & cByFiltreSpatial::KeySet()
{
   return mKeySet;
}

const std::string & cByFiltreSpatial::KeySet()const 
{
   return mKeySet;
}


cTplValGesInit< std::string > & cByFiltreSpatial::TagCentre()
{
   return mTagCentre;
}

const cTplValGesInit< std::string > & cByFiltreSpatial::TagCentre()const 
{
   return mTagCentre;
}


cTplValGesInit< bool > & cByFiltreSpatial::Sym()
{
   return mSym;
}

const cTplValGesInit< bool > & cByFiltreSpatial::Sym()const 
{
   return mSym;
}


cTplValGesInit< cFiltreDeRelationOrient > & cByFiltreSpatial::FiltreSup()
{
   return mFiltreSup;
}

const cTplValGesInit< cFiltreDeRelationOrient > & cByFiltreSpatial::FiltreSup()const 
{
   return mFiltreSup;
}


cTplValGesInit< double > & cByFiltreSpatial::DMaxDelaunay()
{
   return ModeFiltreSpatial().FiltreDelaunay().Val().DMaxDelaunay();
}

const cTplValGesInit< double > & cByFiltreSpatial::DMaxDelaunay()const 
{
   return ModeFiltreSpatial().FiltreDelaunay().Val().DMaxDelaunay();
}


cTplValGesInit< cFiltreDelaunay > & cByFiltreSpatial::FiltreDelaunay()
{
   return ModeFiltreSpatial().FiltreDelaunay();
}

const cTplValGesInit< cFiltreDelaunay > & cByFiltreSpatial::FiltreDelaunay()const 
{
   return ModeFiltreSpatial().FiltreDelaunay();
}


double & cByFiltreSpatial::DistMax()
{
   return ModeFiltreSpatial().FiltreDist().Val().DistMax();
}

const double & cByFiltreSpatial::DistMax()const 
{
   return ModeFiltreSpatial().FiltreDist().Val().DistMax();
}


cTplValGesInit< cFiltreDist > & cByFiltreSpatial::FiltreDist()
{
   return ModeFiltreSpatial().FiltreDist();
}

const cTplValGesInit< cFiltreDist > & cByFiltreSpatial::FiltreDist()const 
{
   return ModeFiltreSpatial().FiltreDist();
}


cModeFiltreSpatial & cByFiltreSpatial::ModeFiltreSpatial()
{
   return mModeFiltreSpatial;
}

const cModeFiltreSpatial & cByFiltreSpatial::ModeFiltreSpatial()const 
{
   return mModeFiltreSpatial;
}

void  BinaryUnDumpFromFile(cByFiltreSpatial & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ByFileTrajecto().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ByFileTrajecto().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ByFileTrajecto().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.KeyOri(),aFp);
    BinaryUnDumpFromFile(anObj.KeySet(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.TagCentre().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.TagCentre().ValForcedForUnUmp(),aFp);
        }
        else  anObj.TagCentre().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Sym().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Sym().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Sym().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.FiltreSup().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.FiltreSup().ValForcedForUnUmp(),aFp);
        }
        else  anObj.FiltreSup().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.ModeFiltreSpatial(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cByFiltreSpatial & anObj)
{
    BinaryDumpInFile(aFp,anObj.ByFileTrajecto().IsInit());
    if (anObj.ByFileTrajecto().IsInit()) BinaryDumpInFile(aFp,anObj.ByFileTrajecto().Val());
    BinaryDumpInFile(aFp,anObj.KeyOri());
    BinaryDumpInFile(aFp,anObj.KeySet());
    BinaryDumpInFile(aFp,anObj.TagCentre().IsInit());
    if (anObj.TagCentre().IsInit()) BinaryDumpInFile(aFp,anObj.TagCentre().Val());
    BinaryDumpInFile(aFp,anObj.Sym().IsInit());
    if (anObj.Sym().IsInit()) BinaryDumpInFile(aFp,anObj.Sym().Val());
    BinaryDumpInFile(aFp,anObj.FiltreSup().IsInit());
    if (anObj.FiltreSup().IsInit()) BinaryDumpInFile(aFp,anObj.FiltreSup().Val());
    BinaryDumpInFile(aFp,anObj.ModeFiltreSpatial());
}

cElXMLTree * ToXMLTree(const cByFiltreSpatial & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ByFiltreSpatial",eXMLBranche);
   if (anObj.ByFileTrajecto().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ByFileTrajecto"),anObj.ByFileTrajecto().Val())->ReTagThis("ByFileTrajecto"));
   aRes->AddFils(::ToXMLTree(std::string("KeyOri"),anObj.KeyOri())->ReTagThis("KeyOri"));
   aRes->AddFils(::ToXMLTree(std::string("KeySet"),anObj.KeySet())->ReTagThis("KeySet"));
   if (anObj.TagCentre().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("TagCentre"),anObj.TagCentre().Val())->ReTagThis("TagCentre"));
   if (anObj.Sym().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Sym"),anObj.Sym().Val())->ReTagThis("Sym"));
   if (anObj.FiltreSup().IsInit())
      aRes->AddFils(ToXMLTree(anObj.FiltreSup().Val())->ReTagThis("FiltreSup"));
   aRes->AddFils(ToXMLTree(anObj.ModeFiltreSpatial())->ReTagThis("ModeFiltreSpatial"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cByFiltreSpatial & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.ByFileTrajecto(),aTree->Get("ByFileTrajecto",1)); //tototo 

   xml_init(anObj.KeyOri(),aTree->Get("KeyOri",1)); //tototo 

   xml_init(anObj.KeySet(),aTree->Get("KeySet",1)); //tototo 

   xml_init(anObj.TagCentre(),aTree->Get("TagCentre",1),std::string("Centre")); //tototo 

   xml_init(anObj.Sym(),aTree->Get("Sym",1),bool(true)); //tototo 

   xml_init(anObj.FiltreSup(),aTree->Get("FiltreSup",1)); //tototo 

   xml_init(anObj.ModeFiltreSpatial(),aTree->Get("ModeFiltreSpatial",1)); //tototo 
}

std::string  Mangling( cByFiltreSpatial *) {return "06BF958EE3AB278EFE3F";};


std::vector< std::string > & cByAdjacence::KeySets()
{
   return mKeySets;
}

const std::vector< std::string > & cByAdjacence::KeySets()const 
{
   return mKeySets;
}


cTplValGesInit< IntSubst > & cByAdjacence::DeltaMax()
{
   return mDeltaMax;
}

const cTplValGesInit< IntSubst > & cByAdjacence::DeltaMax()const 
{
   return mDeltaMax;
}


cTplValGesInit< IntSubst > & cByAdjacence::DeltaMin()
{
   return mDeltaMin;
}

const cTplValGesInit< IntSubst > & cByAdjacence::DeltaMin()const 
{
   return mDeltaMin;
}


cTplValGesInit< cFiltreDeRelationOrient > & cByAdjacence::Filtre()
{
   return mFiltre;
}

const cTplValGesInit< cFiltreDeRelationOrient > & cByAdjacence::Filtre()const 
{
   return mFiltre;
}


cTplValGesInit< bool > & cByAdjacence::Sym()
{
   return mSym;
}

const cTplValGesInit< bool > & cByAdjacence::Sym()const 
{
   return mSym;
}


cTplValGesInit< bool > & cByAdjacence::Circ()
{
   return mCirc;
}

const cTplValGesInit< bool > & cByAdjacence::Circ()const 
{
   return mCirc;
}

void  BinaryUnDumpFromFile(cByAdjacence & anObj,ELISE_fp & aFp)
{
   { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             std::string aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.KeySets().push_back(aVal);
        }
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.DeltaMax().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.DeltaMax().ValForcedForUnUmp(),aFp);
        }
        else  anObj.DeltaMax().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.DeltaMin().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.DeltaMin().ValForcedForUnUmp(),aFp);
        }
        else  anObj.DeltaMin().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Filtre().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Filtre().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Filtre().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Sym().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Sym().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Sym().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Circ().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Circ().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Circ().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cByAdjacence & anObj)
{
    BinaryDumpInFile(aFp,(int)anObj.KeySets().size());
    for(  std::vector< std::string >::const_iterator iT=anObj.KeySets().begin();
         iT!=anObj.KeySets().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.DeltaMax().IsInit());
    if (anObj.DeltaMax().IsInit()) BinaryDumpInFile(aFp,anObj.DeltaMax().Val());
    BinaryDumpInFile(aFp,anObj.DeltaMin().IsInit());
    if (anObj.DeltaMin().IsInit()) BinaryDumpInFile(aFp,anObj.DeltaMin().Val());
    BinaryDumpInFile(aFp,anObj.Filtre().IsInit());
    if (anObj.Filtre().IsInit()) BinaryDumpInFile(aFp,anObj.Filtre().Val());
    BinaryDumpInFile(aFp,anObj.Sym().IsInit());
    if (anObj.Sym().IsInit()) BinaryDumpInFile(aFp,anObj.Sym().Val());
    BinaryDumpInFile(aFp,anObj.Circ().IsInit());
    if (anObj.Circ().IsInit()) BinaryDumpInFile(aFp,anObj.Circ().Val());
}

cElXMLTree * ToXMLTree(const cByAdjacence & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ByAdjacence",eXMLBranche);
  for
  (       std::vector< std::string >::const_iterator it=anObj.KeySets().begin();
      it !=anObj.KeySets().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("KeySets"),(*it))->ReTagThis("KeySets"));
   if (anObj.DeltaMax().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DeltaMax"),anObj.DeltaMax().Val())->ReTagThis("DeltaMax"));
   if (anObj.DeltaMin().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DeltaMin"),anObj.DeltaMin().Val())->ReTagThis("DeltaMin"));
   if (anObj.Filtre().IsInit())
      aRes->AddFils(ToXMLTree(anObj.Filtre().Val())->ReTagThis("Filtre"));
   if (anObj.Sym().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Sym"),anObj.Sym().Val())->ReTagThis("Sym"));
   if (anObj.Circ().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Circ"),anObj.Circ().Val())->ReTagThis("Circ"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cByAdjacence & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.KeySets(),aTree->GetAll("KeySets",false,1));

   xml_init(anObj.DeltaMax(),aTree->Get("DeltaMax",1),IntSubst(10000000)); //tototo 

   xml_init(anObj.DeltaMin(),aTree->Get("DeltaMin",1)); //tototo 

   xml_init(anObj.Filtre(),aTree->Get("Filtre",1)); //tototo 

   xml_init(anObj.Sym(),aTree->Get("Sym",1),bool(true)); //tototo 

   xml_init(anObj.Circ(),aTree->Get("Circ",1),bool(false)); //tototo 
}

std::string  Mangling( cByAdjacence *) {return "5BC4D9CBED77D184FF3F";};


std::list< std::string > & cNameRelDescriptor::NameFileIn()
{
   return mNameFileIn;
}

const std::list< std::string > & cNameRelDescriptor::NameFileIn()const 
{
   return mNameFileIn;
}


cTplValGesInit< bool > & cNameRelDescriptor::Reflexif()
{
   return mReflexif;
}

const cTplValGesInit< bool > & cNameRelDescriptor::Reflexif()const 
{
   return mReflexif;
}


cTplValGesInit< std::string > & cNameRelDescriptor::NameFileSauvegarde()
{
   return mNameFileSauvegarde;
}

const cTplValGesInit< std::string > & cNameRelDescriptor::NameFileSauvegarde()const 
{
   return mNameFileSauvegarde;
}


std::list< cRelByGrapheExpl > & cNameRelDescriptor::RelByGrapheExpl()
{
   return mRelByGrapheExpl;
}

const std::list< cRelByGrapheExpl > & cNameRelDescriptor::RelByGrapheExpl()const 
{
   return mRelByGrapheExpl;
}


std::list< cByGroupesDImages > & cNameRelDescriptor::ByGroupesDImages()
{
   return mByGroupesDImages;
}

const std::list< cByGroupesDImages > & cNameRelDescriptor::ByGroupesDImages()const 
{
   return mByGroupesDImages;
}


std::list< cByFiltreSpatial > & cNameRelDescriptor::ByFiltreSpatial()
{
   return mByFiltreSpatial;
}

const std::list< cByFiltreSpatial > & cNameRelDescriptor::ByFiltreSpatial()const 
{
   return mByFiltreSpatial;
}


std::list< cByAdjacence > & cNameRelDescriptor::ByAdjacence()
{
   return mByAdjacence;
}

const std::list< cByAdjacence > & cNameRelDescriptor::ByAdjacence()const 
{
   return mByAdjacence;
}


std::list< cCpleString > & cNameRelDescriptor::CplesExcl()
{
   return mCplesExcl;
}

const std::list< cCpleString > & cNameRelDescriptor::CplesExcl()const 
{
   return mCplesExcl;
}

void  BinaryUnDumpFromFile(cNameRelDescriptor & anObj,ELISE_fp & aFp)
{
   { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             std::string aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.NameFileIn().push_back(aVal);
        }
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Reflexif().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Reflexif().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Reflexif().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NameFileSauvegarde().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NameFileSauvegarde().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NameFileSauvegarde().SetNoInit();
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cRelByGrapheExpl aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.RelByGrapheExpl().push_back(aVal);
        }
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cByGroupesDImages aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.ByGroupesDImages().push_back(aVal);
        }
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cByFiltreSpatial aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.ByFiltreSpatial().push_back(aVal);
        }
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cByAdjacence aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.ByAdjacence().push_back(aVal);
        }
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cCpleString aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.CplesExcl().push_back(aVal);
        }
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cNameRelDescriptor & anObj)
{
    BinaryDumpInFile(aFp,(int)anObj.NameFileIn().size());
    for(  std::list< std::string >::const_iterator iT=anObj.NameFileIn().begin();
         iT!=anObj.NameFileIn().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.Reflexif().IsInit());
    if (anObj.Reflexif().IsInit()) BinaryDumpInFile(aFp,anObj.Reflexif().Val());
    BinaryDumpInFile(aFp,anObj.NameFileSauvegarde().IsInit());
    if (anObj.NameFileSauvegarde().IsInit()) BinaryDumpInFile(aFp,anObj.NameFileSauvegarde().Val());
    BinaryDumpInFile(aFp,(int)anObj.RelByGrapheExpl().size());
    for(  std::list< cRelByGrapheExpl >::const_iterator iT=anObj.RelByGrapheExpl().begin();
         iT!=anObj.RelByGrapheExpl().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,(int)anObj.ByGroupesDImages().size());
    for(  std::list< cByGroupesDImages >::const_iterator iT=anObj.ByGroupesDImages().begin();
         iT!=anObj.ByGroupesDImages().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,(int)anObj.ByFiltreSpatial().size());
    for(  std::list< cByFiltreSpatial >::const_iterator iT=anObj.ByFiltreSpatial().begin();
         iT!=anObj.ByFiltreSpatial().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,(int)anObj.ByAdjacence().size());
    for(  std::list< cByAdjacence >::const_iterator iT=anObj.ByAdjacence().begin();
         iT!=anObj.ByAdjacence().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,(int)anObj.CplesExcl().size());
    for(  std::list< cCpleString >::const_iterator iT=anObj.CplesExcl().begin();
         iT!=anObj.CplesExcl().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
}

cElXMLTree * ToXMLTree(const cNameRelDescriptor & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"NameRelDescriptor",eXMLBranche);
  for
  (       std::list< std::string >::const_iterator it=anObj.NameFileIn().begin();
      it !=anObj.NameFileIn().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("NameFileIn"),(*it))->ReTagThis("NameFileIn"));
   if (anObj.Reflexif().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Reflexif"),anObj.Reflexif().Val())->ReTagThis("Reflexif"));
   if (anObj.NameFileSauvegarde().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NameFileSauvegarde"),anObj.NameFileSauvegarde().Val())->ReTagThis("NameFileSauvegarde"));
  for
  (       std::list< cRelByGrapheExpl >::const_iterator it=anObj.RelByGrapheExpl().begin();
      it !=anObj.RelByGrapheExpl().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("RelByGrapheExpl"));
  for
  (       std::list< cByGroupesDImages >::const_iterator it=anObj.ByGroupesDImages().begin();
      it !=anObj.ByGroupesDImages().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("ByGroupesDImages"));
  for
  (       std::list< cByFiltreSpatial >::const_iterator it=anObj.ByFiltreSpatial().begin();
      it !=anObj.ByFiltreSpatial().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("ByFiltreSpatial"));
  for
  (       std::list< cByAdjacence >::const_iterator it=anObj.ByAdjacence().begin();
      it !=anObj.ByAdjacence().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("ByAdjacence"));
  for
  (       std::list< cCpleString >::const_iterator it=anObj.CplesExcl().begin();
      it !=anObj.CplesExcl().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("CplesExcl"),(*it))->ReTagThis("CplesExcl"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cNameRelDescriptor & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.NameFileIn(),aTree->GetAll("NameFileIn",false,1));

   xml_init(anObj.Reflexif(),aTree->Get("Reflexif",1),bool(false)); //tototo 

   xml_init(anObj.NameFileSauvegarde(),aTree->Get("NameFileSauvegarde",1)); //tototo 

   xml_init(anObj.RelByGrapheExpl(),aTree->GetAll("RelByGrapheExpl",false,1));

   xml_init(anObj.ByGroupesDImages(),aTree->GetAll("ByGroupesDImages",false,1));

   xml_init(anObj.ByFiltreSpatial(),aTree->GetAll("ByFiltreSpatial",false,1));

   xml_init(anObj.ByAdjacence(),aTree->GetAll("ByAdjacence",false,1));

   xml_init(anObj.CplesExcl(),aTree->GetAll("CplesExcl",false,1));
}

std::string  Mangling( cNameRelDescriptor *) {return "EC0BE9AA1F55F380FF3F";};


std::string & cExeRequired::Exe()
{
   return mExe;
}

const std::string & cExeRequired::Exe()const 
{
   return mExe;
}


std::string & cExeRequired::Make()
{
   return mMake;
}

const std::string & cExeRequired::Make()const 
{
   return mMake;
}

void  BinaryUnDumpFromFile(cExeRequired & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Exe(),aFp);
    BinaryUnDumpFromFile(anObj.Make(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cExeRequired & anObj)
{
    BinaryDumpInFile(aFp,anObj.Exe());
    BinaryDumpInFile(aFp,anObj.Make());
}

cElXMLTree * ToXMLTree(const cExeRequired & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ExeRequired",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Exe"),anObj.Exe())->ReTagThis("Exe"));
   aRes->AddFils(::ToXMLTree(std::string("Make"),anObj.Make())->ReTagThis("Make"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cExeRequired & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Exe(),aTree->Get("Exe",1)); //tototo 

   xml_init(anObj.Make(),aTree->Get("Make",1)); //tototo 
}

std::string  Mangling( cExeRequired *) {return "3C1E4B5E8B6442AEFBBF";};


std::list< std::string > & cFileRequired::Pattern()
{
   return mPattern;
}

const std::list< std::string > & cFileRequired::Pattern()const 
{
   return mPattern;
}


cTplValGesInit< int > & cFileRequired::NbMin()
{
   return mNbMin;
}

const cTplValGesInit< int > & cFileRequired::NbMin()const 
{
   return mNbMin;
}


cTplValGesInit< int > & cFileRequired::NbMax()
{
   return mNbMax;
}

const cTplValGesInit< int > & cFileRequired::NbMax()const 
{
   return mNbMax;
}

void  BinaryUnDumpFromFile(cFileRequired & anObj,ELISE_fp & aFp)
{
   { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             std::string aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.Pattern().push_back(aVal);
        }
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NbMin().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NbMin().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NbMin().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NbMax().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NbMax().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NbMax().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cFileRequired & anObj)
{
    BinaryDumpInFile(aFp,(int)anObj.Pattern().size());
    for(  std::list< std::string >::const_iterator iT=anObj.Pattern().begin();
         iT!=anObj.Pattern().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.NbMin().IsInit());
    if (anObj.NbMin().IsInit()) BinaryDumpInFile(aFp,anObj.NbMin().Val());
    BinaryDumpInFile(aFp,anObj.NbMax().IsInit());
    if (anObj.NbMax().IsInit()) BinaryDumpInFile(aFp,anObj.NbMax().Val());
}

cElXMLTree * ToXMLTree(const cFileRequired & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"FileRequired",eXMLBranche);
  for
  (       std::list< std::string >::const_iterator it=anObj.Pattern().begin();
      it !=anObj.Pattern().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("Pattern"),(*it))->ReTagThis("Pattern"));
   if (anObj.NbMin().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NbMin"),anObj.NbMin().Val())->ReTagThis("NbMin"));
   if (anObj.NbMax().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NbMax"),anObj.NbMax().Val())->ReTagThis("NbMax"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cFileRequired & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Pattern(),aTree->GetAll("Pattern",false,1));

   xml_init(anObj.NbMin(),aTree->Get("NbMin",1),int(1)); //tototo 

   xml_init(anObj.NbMax(),aTree->Get("NbMax",1)); //tototo 
}

std::string  Mangling( cFileRequired *) {return "38329C2372040FE1FBBF";};


std::list< cExeRequired > & cBatchRequirement::ExeRequired()
{
   return mExeRequired;
}

const std::list< cExeRequired > & cBatchRequirement::ExeRequired()const 
{
   return mExeRequired;
}


std::list< cFileRequired > & cBatchRequirement::FileRequired()
{
   return mFileRequired;
}

const std::list< cFileRequired > & cBatchRequirement::FileRequired()const 
{
   return mFileRequired;
}

void  BinaryUnDumpFromFile(cBatchRequirement & anObj,ELISE_fp & aFp)
{
   { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cExeRequired aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.ExeRequired().push_back(aVal);
        }
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cFileRequired aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.FileRequired().push_back(aVal);
        }
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cBatchRequirement & anObj)
{
    BinaryDumpInFile(aFp,(int)anObj.ExeRequired().size());
    for(  std::list< cExeRequired >::const_iterator iT=anObj.ExeRequired().begin();
         iT!=anObj.ExeRequired().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,(int)anObj.FileRequired().size());
    for(  std::list< cFileRequired >::const_iterator iT=anObj.FileRequired().begin();
         iT!=anObj.FileRequired().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
}

cElXMLTree * ToXMLTree(const cBatchRequirement & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"BatchRequirement",eXMLBranche);
  for
  (       std::list< cExeRequired >::const_iterator it=anObj.ExeRequired().begin();
      it !=anObj.ExeRequired().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("ExeRequired"));
  for
  (       std::list< cFileRequired >::const_iterator it=anObj.FileRequired().begin();
      it !=anObj.FileRequired().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("FileRequired"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cBatchRequirement & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.ExeRequired(),aTree->GetAll("ExeRequired",false,1));

   xml_init(anObj.FileRequired(),aTree->GetAll("FileRequired",false,1));
}

std::string  Mangling( cBatchRequirement *) {return "30C4FCA41488DE8FFCBF";};


cTplValGesInit< Pt3dr > & cExportApero2MM::DirVertLoc()
{
   return mDirVertLoc;
}

const cTplValGesInit< Pt3dr > & cExportApero2MM::DirVertLoc()const 
{
   return mDirVertLoc;
}


cTplValGesInit< double > & cExportApero2MM::ProfInVertLoc()
{
   return mProfInVertLoc;
}

const cTplValGesInit< double > & cExportApero2MM::ProfInVertLoc()const 
{
   return mProfInVertLoc;
}

void  BinaryUnDumpFromFile(cExportApero2MM & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.DirVertLoc().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.DirVertLoc().ValForcedForUnUmp(),aFp);
        }
        else  anObj.DirVertLoc().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ProfInVertLoc().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ProfInVertLoc().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ProfInVertLoc().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cExportApero2MM & anObj)
{
    BinaryDumpInFile(aFp,anObj.DirVertLoc().IsInit());
    if (anObj.DirVertLoc().IsInit()) BinaryDumpInFile(aFp,anObj.DirVertLoc().Val());
    BinaryDumpInFile(aFp,anObj.ProfInVertLoc().IsInit());
    if (anObj.ProfInVertLoc().IsInit()) BinaryDumpInFile(aFp,anObj.ProfInVertLoc().Val());
}

cElXMLTree * ToXMLTree(const cExportApero2MM & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ExportApero2MM",eXMLBranche);
   if (anObj.DirVertLoc().IsInit())
      aRes->AddFils(ToXMLTree(std::string("DirVertLoc"),anObj.DirVertLoc().Val())->ReTagThis("DirVertLoc"));
   if (anObj.ProfInVertLoc().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ProfInVertLoc"),anObj.ProfInVertLoc().Val())->ReTagThis("ProfInVertLoc"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cExportApero2MM & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.DirVertLoc(),aTree->Get("DirVertLoc",1)); //tototo 

   xml_init(anObj.ProfInVertLoc(),aTree->Get("ProfInVertLoc",1)); //tototo 
}

std::string  Mangling( cExportApero2MM *) {return "AF8050D32387CCA1FE3F";};


int & cXmlHour::H()
{
   return mH;
}

const int & cXmlHour::H()const 
{
   return mH;
}


int & cXmlHour::M()
{
   return mM;
}

const int & cXmlHour::M()const 
{
   return mM;
}


double & cXmlHour::S()
{
   return mS;
}

const double & cXmlHour::S()const 
{
   return mS;
}

void  BinaryUnDumpFromFile(cXmlHour & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.H(),aFp);
    BinaryUnDumpFromFile(anObj.M(),aFp);
    BinaryUnDumpFromFile(anObj.S(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cXmlHour & anObj)
{
    BinaryDumpInFile(aFp,anObj.H());
    BinaryDumpInFile(aFp,anObj.M());
    BinaryDumpInFile(aFp,anObj.S());
}

cElXMLTree * ToXMLTree(const cXmlHour & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"XmlHour",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("H"),anObj.H())->ReTagThis("H"));
   aRes->AddFils(::ToXMLTree(std::string("M"),anObj.M())->ReTagThis("M"));
   aRes->AddFils(::ToXMLTree(std::string("S"),anObj.S())->ReTagThis("S"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cXmlHour & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.H(),aTree->Get("H",1)); //tototo 

   xml_init(anObj.M(),aTree->Get("M",1)); //tototo 

   xml_init(anObj.S(),aTree->Get("S",1)); //tototo 
}

std::string  Mangling( cXmlHour *) {return "7DE0121A11584CEAFD3F";};


int & cXmlDate::Y()
{
   return mY;
}

const int & cXmlDate::Y()const 
{
   return mY;
}


int & cXmlDate::M()
{
   return mM;
}

const int & cXmlDate::M()const 
{
   return mM;
}


int & cXmlDate::D()
{
   return mD;
}

const int & cXmlDate::D()const 
{
   return mD;
}


cXmlHour & cXmlDate::Hour()
{
   return mHour;
}

const cXmlHour & cXmlDate::Hour()const 
{
   return mHour;
}

void  BinaryUnDumpFromFile(cXmlDate & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Y(),aFp);
    BinaryUnDumpFromFile(anObj.M(),aFp);
    BinaryUnDumpFromFile(anObj.D(),aFp);
    BinaryUnDumpFromFile(anObj.Hour(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cXmlDate & anObj)
{
    BinaryDumpInFile(aFp,anObj.Y());
    BinaryDumpInFile(aFp,anObj.M());
    BinaryDumpInFile(aFp,anObj.D());
    BinaryDumpInFile(aFp,anObj.Hour());
}

cElXMLTree * ToXMLTree(const cXmlDate & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"XmlDate",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Y"),anObj.Y())->ReTagThis("Y"));
   aRes->AddFils(::ToXMLTree(std::string("M"),anObj.M())->ReTagThis("M"));
   aRes->AddFils(::ToXMLTree(std::string("D"),anObj.D())->ReTagThis("D"));
   aRes->AddFils(ToXMLTree(anObj.Hour())->ReTagThis("Hour"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cXmlDate & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Y(),aTree->Get("Y",1)); //tototo 

   xml_init(anObj.M(),aTree->Get("M",1)); //tototo 

   xml_init(anObj.D(),aTree->Get("D",1)); //tototo 

   xml_init(anObj.Hour(),aTree->Get("Hour",1)); //tototo 
}

std::string  Mangling( cXmlDate *) {return "C7D8067DE3BE828CFF3F";};


int & cXmlXifInfo::HGRev()
{
   return mHGRev;
}

const int & cXmlXifInfo::HGRev()const 
{
   return mHGRev;
}


cTplValGesInit< double > & cXmlXifInfo::FocMM()
{
   return mFocMM;
}

const cTplValGesInit< double > & cXmlXifInfo::FocMM()const 
{
   return mFocMM;
}


cTplValGesInit< double > & cXmlXifInfo::Foc35()
{
   return mFoc35;
}

const cTplValGesInit< double > & cXmlXifInfo::Foc35()const 
{
   return mFoc35;
}


cTplValGesInit< double > & cXmlXifInfo::ExpTime()
{
   return mExpTime;
}

const cTplValGesInit< double > & cXmlXifInfo::ExpTime()const 
{
   return mExpTime;
}


cTplValGesInit< double > & cXmlXifInfo::Diaph()
{
   return mDiaph;
}

const cTplValGesInit< double > & cXmlXifInfo::Diaph()const 
{
   return mDiaph;
}


cTplValGesInit< double > & cXmlXifInfo::IsoSpeed()
{
   return mIsoSpeed;
}

const cTplValGesInit< double > & cXmlXifInfo::IsoSpeed()const 
{
   return mIsoSpeed;
}


cTplValGesInit< Pt2di > & cXmlXifInfo::Sz()
{
   return mSz;
}

const cTplValGesInit< Pt2di > & cXmlXifInfo::Sz()const 
{
   return mSz;
}


cTplValGesInit< double > & cXmlXifInfo::GPSLat()
{
   return mGPSLat;
}

const cTplValGesInit< double > & cXmlXifInfo::GPSLat()const 
{
   return mGPSLat;
}


cTplValGesInit< double > & cXmlXifInfo::GPSLon()
{
   return mGPSLon;
}

const cTplValGesInit< double > & cXmlXifInfo::GPSLon()const 
{
   return mGPSLon;
}


cTplValGesInit< double > & cXmlXifInfo::GPSAlt()
{
   return mGPSAlt;
}

const cTplValGesInit< double > & cXmlXifInfo::GPSAlt()const 
{
   return mGPSAlt;
}


cTplValGesInit< std::string > & cXmlXifInfo::Cam()
{
   return mCam;
}

const cTplValGesInit< std::string > & cXmlXifInfo::Cam()const 
{
   return mCam;
}


cTplValGesInit< std::string > & cXmlXifInfo::BayPat()
{
   return mBayPat;
}

const cTplValGesInit< std::string > & cXmlXifInfo::BayPat()const 
{
   return mBayPat;
}


cTplValGesInit< cXmlDate > & cXmlXifInfo::Date()
{
   return mDate;
}

const cTplValGesInit< cXmlDate > & cXmlXifInfo::Date()const 
{
   return mDate;
}

void  BinaryUnDumpFromFile(cXmlXifInfo & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.HGRev(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.FocMM().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.FocMM().ValForcedForUnUmp(),aFp);
        }
        else  anObj.FocMM().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Foc35().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Foc35().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Foc35().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ExpTime().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ExpTime().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ExpTime().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Diaph().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Diaph().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Diaph().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.IsoSpeed().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.IsoSpeed().ValForcedForUnUmp(),aFp);
        }
        else  anObj.IsoSpeed().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Sz().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Sz().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Sz().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.GPSLat().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.GPSLat().ValForcedForUnUmp(),aFp);
        }
        else  anObj.GPSLat().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.GPSLon().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.GPSLon().ValForcedForUnUmp(),aFp);
        }
        else  anObj.GPSLon().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.GPSAlt().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.GPSAlt().ValForcedForUnUmp(),aFp);
        }
        else  anObj.GPSAlt().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Cam().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Cam().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Cam().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.BayPat().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.BayPat().ValForcedForUnUmp(),aFp);
        }
        else  anObj.BayPat().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Date().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Date().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Date().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cXmlXifInfo & anObj)
{
    BinaryDumpInFile(aFp,anObj.HGRev());
    BinaryDumpInFile(aFp,anObj.FocMM().IsInit());
    if (anObj.FocMM().IsInit()) BinaryDumpInFile(aFp,anObj.FocMM().Val());
    BinaryDumpInFile(aFp,anObj.Foc35().IsInit());
    if (anObj.Foc35().IsInit()) BinaryDumpInFile(aFp,anObj.Foc35().Val());
    BinaryDumpInFile(aFp,anObj.ExpTime().IsInit());
    if (anObj.ExpTime().IsInit()) BinaryDumpInFile(aFp,anObj.ExpTime().Val());
    BinaryDumpInFile(aFp,anObj.Diaph().IsInit());
    if (anObj.Diaph().IsInit()) BinaryDumpInFile(aFp,anObj.Diaph().Val());
    BinaryDumpInFile(aFp,anObj.IsoSpeed().IsInit());
    if (anObj.IsoSpeed().IsInit()) BinaryDumpInFile(aFp,anObj.IsoSpeed().Val());
    BinaryDumpInFile(aFp,anObj.Sz().IsInit());
    if (anObj.Sz().IsInit()) BinaryDumpInFile(aFp,anObj.Sz().Val());
    BinaryDumpInFile(aFp,anObj.GPSLat().IsInit());
    if (anObj.GPSLat().IsInit()) BinaryDumpInFile(aFp,anObj.GPSLat().Val());
    BinaryDumpInFile(aFp,anObj.GPSLon().IsInit());
    if (anObj.GPSLon().IsInit()) BinaryDumpInFile(aFp,anObj.GPSLon().Val());
    BinaryDumpInFile(aFp,anObj.GPSAlt().IsInit());
    if (anObj.GPSAlt().IsInit()) BinaryDumpInFile(aFp,anObj.GPSAlt().Val());
    BinaryDumpInFile(aFp,anObj.Cam().IsInit());
    if (anObj.Cam().IsInit()) BinaryDumpInFile(aFp,anObj.Cam().Val());
    BinaryDumpInFile(aFp,anObj.BayPat().IsInit());
    if (anObj.BayPat().IsInit()) BinaryDumpInFile(aFp,anObj.BayPat().Val());
    BinaryDumpInFile(aFp,anObj.Date().IsInit());
    if (anObj.Date().IsInit()) BinaryDumpInFile(aFp,anObj.Date().Val());
}

cElXMLTree * ToXMLTree(const cXmlXifInfo & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"XmlXifInfo",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("HGRev"),anObj.HGRev())->ReTagThis("HGRev"));
   if (anObj.FocMM().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("FocMM"),anObj.FocMM().Val())->ReTagThis("FocMM"));
   if (anObj.Foc35().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Foc35"),anObj.Foc35().Val())->ReTagThis("Foc35"));
   if (anObj.ExpTime().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ExpTime"),anObj.ExpTime().Val())->ReTagThis("ExpTime"));
   if (anObj.Diaph().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Diaph"),anObj.Diaph().Val())->ReTagThis("Diaph"));
   if (anObj.IsoSpeed().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("IsoSpeed"),anObj.IsoSpeed().Val())->ReTagThis("IsoSpeed"));
   if (anObj.Sz().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Sz"),anObj.Sz().Val())->ReTagThis("Sz"));
   if (anObj.GPSLat().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("GPSLat"),anObj.GPSLat().Val())->ReTagThis("GPSLat"));
   if (anObj.GPSLon().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("GPSLon"),anObj.GPSLon().Val())->ReTagThis("GPSLon"));
   if (anObj.GPSAlt().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("GPSAlt"),anObj.GPSAlt().Val())->ReTagThis("GPSAlt"));
   if (anObj.Cam().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Cam"),anObj.Cam().Val())->ReTagThis("Cam"));
   if (anObj.BayPat().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("BayPat"),anObj.BayPat().Val())->ReTagThis("BayPat"));
   if (anObj.Date().IsInit())
      aRes->AddFils(ToXMLTree(anObj.Date().Val())->ReTagThis("Date"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cXmlXifInfo & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.HGRev(),aTree->Get("HGRev",1)); //tototo 

   xml_init(anObj.FocMM(),aTree->Get("FocMM",1)); //tototo 

   xml_init(anObj.Foc35(),aTree->Get("Foc35",1)); //tototo 

   xml_init(anObj.ExpTime(),aTree->Get("ExpTime",1)); //tototo 

   xml_init(anObj.Diaph(),aTree->Get("Diaph",1)); //tototo 

   xml_init(anObj.IsoSpeed(),aTree->Get("IsoSpeed",1)); //tototo 

   xml_init(anObj.Sz(),aTree->Get("Sz",1)); //tototo 

   xml_init(anObj.GPSLat(),aTree->Get("GPSLat",1)); //tototo 

   xml_init(anObj.GPSLon(),aTree->Get("GPSLon",1)); //tototo 

   xml_init(anObj.GPSAlt(),aTree->Get("GPSAlt",1)); //tototo 

   xml_init(anObj.Cam(),aTree->Get("Cam",1)); //tototo 

   xml_init(anObj.BayPat(),aTree->Get("BayPat",1)); //tototo 

   xml_init(anObj.Date(),aTree->Get("Date",1)); //tototo 
}

std::string  Mangling( cXmlXifInfo *) {return "BA321A9AE6FC6B9CFE3F";};


std::string & cCameraEntry::Name()
{
   return mName;
}

const std::string & cCameraEntry::Name()const 
{
   return mName;
}


Pt2dr & cCameraEntry::SzCaptMm()
{
   return mSzCaptMm;
}

const Pt2dr & cCameraEntry::SzCaptMm()const 
{
   return mSzCaptMm;
}


std::string & cCameraEntry::ShortName()
{
   return mShortName;
}

const std::string & cCameraEntry::ShortName()const 
{
   return mShortName;
}


cTplValGesInit< bool > & cCameraEntry::BayerSwapRB()
{
   return mBayerSwapRB;
}

const cTplValGesInit< bool > & cCameraEntry::BayerSwapRB()const 
{
   return mBayerSwapRB;
}


cTplValGesInit< bool > & cCameraEntry::DevRawBasic()
{
   return mDevRawBasic;
}

const cTplValGesInit< bool > & cCameraEntry::DevRawBasic()const 
{
   return mDevRawBasic;
}

void  BinaryUnDumpFromFile(cCameraEntry & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Name(),aFp);
    BinaryUnDumpFromFile(anObj.SzCaptMm(),aFp);
    BinaryUnDumpFromFile(anObj.ShortName(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.BayerSwapRB().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.BayerSwapRB().ValForcedForUnUmp(),aFp);
        }
        else  anObj.BayerSwapRB().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.DevRawBasic().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.DevRawBasic().ValForcedForUnUmp(),aFp);
        }
        else  anObj.DevRawBasic().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cCameraEntry & anObj)
{
    BinaryDumpInFile(aFp,anObj.Name());
    BinaryDumpInFile(aFp,anObj.SzCaptMm());
    BinaryDumpInFile(aFp,anObj.ShortName());
    BinaryDumpInFile(aFp,anObj.BayerSwapRB().IsInit());
    if (anObj.BayerSwapRB().IsInit()) BinaryDumpInFile(aFp,anObj.BayerSwapRB().Val());
    BinaryDumpInFile(aFp,anObj.DevRawBasic().IsInit());
    if (anObj.DevRawBasic().IsInit()) BinaryDumpInFile(aFp,anObj.DevRawBasic().Val());
}

cElXMLTree * ToXMLTree(const cCameraEntry & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"CameraEntry",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Name"),anObj.Name())->ReTagThis("Name"));
   aRes->AddFils(::ToXMLTree(std::string("SzCaptMm"),anObj.SzCaptMm())->ReTagThis("SzCaptMm"));
   aRes->AddFils(::ToXMLTree(std::string("ShortName"),anObj.ShortName())->ReTagThis("ShortName"));
   if (anObj.BayerSwapRB().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("BayerSwapRB"),anObj.BayerSwapRB().Val())->ReTagThis("BayerSwapRB"));
   if (anObj.DevRawBasic().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DevRawBasic"),anObj.DevRawBasic().Val())->ReTagThis("DevRawBasic"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cCameraEntry & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Name(),aTree->Get("Name",1)); //tototo 

   xml_init(anObj.SzCaptMm(),aTree->Get("SzCaptMm",1)); //tototo 

   xml_init(anObj.ShortName(),aTree->Get("ShortName",1)); //tototo 

   xml_init(anObj.BayerSwapRB(),aTree->Get("BayerSwapRB",1),bool(false)); //tototo 

   xml_init(anObj.DevRawBasic(),aTree->Get("DevRawBasic",1),bool(false)); //tototo 
}

std::string  Mangling( cCameraEntry *) {return "A0CB2D914BCE10A2F9BF";};


std::list< cCameraEntry > & cMMCameraDataBase::CameraEntry()
{
   return mCameraEntry;
}

const std::list< cCameraEntry > & cMMCameraDataBase::CameraEntry()const 
{
   return mCameraEntry;
}

void  BinaryUnDumpFromFile(cMMCameraDataBase & anObj,ELISE_fp & aFp)
{
   { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cCameraEntry aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.CameraEntry().push_back(aVal);
        }
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cMMCameraDataBase & anObj)
{
    BinaryDumpInFile(aFp,(int)anObj.CameraEntry().size());
    for(  std::list< cCameraEntry >::const_iterator iT=anObj.CameraEntry().begin();
         iT!=anObj.CameraEntry().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
}

cElXMLTree * ToXMLTree(const cMMCameraDataBase & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"MMCameraDataBase",eXMLBranche);
  for
  (       std::list< cCameraEntry >::const_iterator it=anObj.CameraEntry().begin();
      it !=anObj.CameraEntry().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("CameraEntry"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cMMCameraDataBase & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.CameraEntry(),aTree->GetAll("CameraEntry",false,1));
}

std::string  Mangling( cMMCameraDataBase *) {return "E294F46DDAFEB481FE3F";};


std::string & cMakeDataBase::KeySetCollectXif()
{
   return mKeySetCollectXif;
}

const std::string & cMakeDataBase::KeySetCollectXif()const 
{
   return mKeySetCollectXif;
}


std::list< std::string > & cMakeDataBase::KeyAssocNameSup()
{
   return mKeyAssocNameSup;
}

const std::list< std::string > & cMakeDataBase::KeyAssocNameSup()const 
{
   return mKeyAssocNameSup;
}


cTplValGesInit< std::string > & cMakeDataBase::NameFile()
{
   return mNameFile;
}

const cTplValGesInit< std::string > & cMakeDataBase::NameFile()const 
{
   return mNameFile;
}

void  BinaryUnDumpFromFile(cMakeDataBase & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.KeySetCollectXif(),aFp);
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             std::string aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.KeyAssocNameSup().push_back(aVal);
        }
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NameFile().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NameFile().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NameFile().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cMakeDataBase & anObj)
{
    BinaryDumpInFile(aFp,anObj.KeySetCollectXif());
    BinaryDumpInFile(aFp,(int)anObj.KeyAssocNameSup().size());
    for(  std::list< std::string >::const_iterator iT=anObj.KeyAssocNameSup().begin();
         iT!=anObj.KeyAssocNameSup().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.NameFile().IsInit());
    if (anObj.NameFile().IsInit()) BinaryDumpInFile(aFp,anObj.NameFile().Val());
}

cElXMLTree * ToXMLTree(const cMakeDataBase & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"MakeDataBase",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("KeySetCollectXif"),anObj.KeySetCollectXif())->ReTagThis("KeySetCollectXif"));
  for
  (       std::list< std::string >::const_iterator it=anObj.KeyAssocNameSup().begin();
      it !=anObj.KeyAssocNameSup().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("KeyAssocNameSup"),(*it))->ReTagThis("KeyAssocNameSup"));
   if (anObj.NameFile().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NameFile"),anObj.NameFile().Val())->ReTagThis("NameFile"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cMakeDataBase & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.KeySetCollectXif(),aTree->Get("KeySetCollectXif",1)); //tototo 

   xml_init(anObj.KeyAssocNameSup(),aTree->GetAll("KeyAssocNameSup",false,1));

   xml_init(anObj.NameFile(),aTree->Get("NameFile",1),std::string("MicMacDataBase")); //tototo 
}

std::string  Mangling( cMakeDataBase *) {return "E7760627E5B4DAB9FE3F";};


cTplValGesInit< cBatchRequirement > & cBatchChantDesc::Requirement()
{
   return mRequirement;
}

const cTplValGesInit< cBatchRequirement > & cBatchChantDesc::Requirement()const 
{
   return mRequirement;
}


std::string & cBatchChantDesc::Key()
{
   return mKey;
}

const std::string & cBatchChantDesc::Key()const 
{
   return mKey;
}


std::list< std::string > & cBatchChantDesc::Line()
{
   return mLine;
}

const std::list< std::string > & cBatchChantDesc::Line()const 
{
   return mLine;
}

void  BinaryUnDumpFromFile(cBatchChantDesc & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Requirement().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Requirement().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Requirement().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.Key(),aFp);
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             std::string aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.Line().push_back(aVal);
        }
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cBatchChantDesc & anObj)
{
    BinaryDumpInFile(aFp,anObj.Requirement().IsInit());
    if (anObj.Requirement().IsInit()) BinaryDumpInFile(aFp,anObj.Requirement().Val());
    BinaryDumpInFile(aFp,anObj.Key());
    BinaryDumpInFile(aFp,(int)anObj.Line().size());
    for(  std::list< std::string >::const_iterator iT=anObj.Line().begin();
         iT!=anObj.Line().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
}

cElXMLTree * ToXMLTree(const cBatchChantDesc & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"BatchChantDesc",eXMLBranche);
   if (anObj.Requirement().IsInit())
      aRes->AddFils(ToXMLTree(anObj.Requirement().Val())->ReTagThis("Requirement"));
   aRes->AddFils(::ToXMLTree(std::string("Key"),anObj.Key())->ReTagThis("Key"));
  for
  (       std::list< std::string >::const_iterator it=anObj.Line().begin();
      it !=anObj.Line().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("Line"),(*it))->ReTagThis("Line"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cBatchChantDesc & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Requirement(),aTree->Get("Requirement",1)); //tototo 

   xml_init(anObj.Key(),aTree->Get("Key",1)); //tototo 

   xml_init(anObj.Line(),aTree->GetAll("Line",false,1));
}

std::string  Mangling( cBatchChantDesc *) {return "F1C482936B80138CFE3F";};


std::list< std::string > & cOneShowChantDesc::LineBefore()
{
   return mLineBefore;
}

const std::list< std::string > & cOneShowChantDesc::LineBefore()const 
{
   return mLineBefore;
}


cTplValGesInit< bool > & cOneShowChantDesc::ShowKeys()
{
   return mShowKeys;
}

const cTplValGesInit< bool > & cOneShowChantDesc::ShowKeys()const 
{
   return mShowKeys;
}


std::list< std::string > & cOneShowChantDesc::KeyRels()
{
   return mKeyRels;
}

const std::list< std::string > & cOneShowChantDesc::KeyRels()const 
{
   return mKeyRels;
}


std::list< std::string > & cOneShowChantDesc::KeySets()
{
   return mKeySets;
}

const std::list< std::string > & cOneShowChantDesc::KeySets()const 
{
   return mKeySets;
}


std::list< std::string > & cOneShowChantDesc::LineAfter()
{
   return mLineAfter;
}

const std::list< std::string > & cOneShowChantDesc::LineAfter()const 
{
   return mLineAfter;
}

void  BinaryUnDumpFromFile(cOneShowChantDesc & anObj,ELISE_fp & aFp)
{
   { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             std::string aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.LineBefore().push_back(aVal);
        }
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ShowKeys().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ShowKeys().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ShowKeys().SetNoInit();
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             std::string aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.KeyRels().push_back(aVal);
        }
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             std::string aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.KeySets().push_back(aVal);
        }
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             std::string aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.LineAfter().push_back(aVal);
        }
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cOneShowChantDesc & anObj)
{
    BinaryDumpInFile(aFp,(int)anObj.LineBefore().size());
    for(  std::list< std::string >::const_iterator iT=anObj.LineBefore().begin();
         iT!=anObj.LineBefore().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.ShowKeys().IsInit());
    if (anObj.ShowKeys().IsInit()) BinaryDumpInFile(aFp,anObj.ShowKeys().Val());
    BinaryDumpInFile(aFp,(int)anObj.KeyRels().size());
    for(  std::list< std::string >::const_iterator iT=anObj.KeyRels().begin();
         iT!=anObj.KeyRels().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,(int)anObj.KeySets().size());
    for(  std::list< std::string >::const_iterator iT=anObj.KeySets().begin();
         iT!=anObj.KeySets().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,(int)anObj.LineAfter().size());
    for(  std::list< std::string >::const_iterator iT=anObj.LineAfter().begin();
         iT!=anObj.LineAfter().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
}

cElXMLTree * ToXMLTree(const cOneShowChantDesc & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"OneShowChantDesc",eXMLBranche);
  for
  (       std::list< std::string >::const_iterator it=anObj.LineBefore().begin();
      it !=anObj.LineBefore().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("LineBefore"),(*it))->ReTagThis("LineBefore"));
   if (anObj.ShowKeys().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ShowKeys"),anObj.ShowKeys().Val())->ReTagThis("ShowKeys"));
  for
  (       std::list< std::string >::const_iterator it=anObj.KeyRels().begin();
      it !=anObj.KeyRels().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("KeyRels"),(*it))->ReTagThis("KeyRels"));
  for
  (       std::list< std::string >::const_iterator it=anObj.KeySets().begin();
      it !=anObj.KeySets().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("KeySets"),(*it))->ReTagThis("KeySets"));
  for
  (       std::list< std::string >::const_iterator it=anObj.LineAfter().begin();
      it !=anObj.LineAfter().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("LineAfter"),(*it))->ReTagThis("LineAfter"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cOneShowChantDesc & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.LineBefore(),aTree->GetAll("LineBefore",false,1));

   xml_init(anObj.ShowKeys(),aTree->Get("ShowKeys",1),bool(true)); //tototo 

   xml_init(anObj.KeyRels(),aTree->GetAll("KeyRels",false,1));

   xml_init(anObj.KeySets(),aTree->GetAll("KeySets",false,1));

   xml_init(anObj.LineAfter(),aTree->GetAll("LineAfter",false,1));
}

std::string  Mangling( cOneShowChantDesc *) {return "A018A96866DDE481FD3F";};


std::list< cOneShowChantDesc > & cShowChantDesc::OneShowChantDesc()
{
   return mOneShowChantDesc;
}

const std::list< cOneShowChantDesc > & cShowChantDesc::OneShowChantDesc()const 
{
   return mOneShowChantDesc;
}


std::string & cShowChantDesc::File()
{
   return mFile;
}

const std::string & cShowChantDesc::File()const 
{
   return mFile;
}

void  BinaryUnDumpFromFile(cShowChantDesc & anObj,ELISE_fp & aFp)
{
   { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cOneShowChantDesc aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.OneShowChantDesc().push_back(aVal);
        }
  } ;
    BinaryUnDumpFromFile(anObj.File(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cShowChantDesc & anObj)
{
    BinaryDumpInFile(aFp,(int)anObj.OneShowChantDesc().size());
    for(  std::list< cOneShowChantDesc >::const_iterator iT=anObj.OneShowChantDesc().begin();
         iT!=anObj.OneShowChantDesc().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.File());
}

cElXMLTree * ToXMLTree(const cShowChantDesc & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ShowChantDesc",eXMLBranche);
  for
  (       std::list< cOneShowChantDesc >::const_iterator it=anObj.OneShowChantDesc().begin();
      it !=anObj.OneShowChantDesc().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("OneShowChantDesc"));
   aRes->AddFils(::ToXMLTree(std::string("File"),anObj.File())->ReTagThis("File"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cShowChantDesc & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.OneShowChantDesc(),aTree->GetAll("OneShowChantDesc",false,1));

   xml_init(anObj.File(),aTree->Get("File",1)); //tototo 
}

std::string  Mangling( cShowChantDesc *) {return "DC4D73F483B6C28BFD3F";};


std::string & cMatrixSplitBox::KeyMatr()
{
   return mKeyMatr;
}

const std::string & cMatrixSplitBox::KeyMatr()const 
{
   return mKeyMatr;
}


cTplValGesInit< double > & cMatrixSplitBox::Rab()
{
   return mRab;
}

const cTplValGesInit< double > & cMatrixSplitBox::Rab()const 
{
   return mRab;
}

void  BinaryUnDumpFromFile(cMatrixSplitBox & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.KeyMatr(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Rab().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Rab().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Rab().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cMatrixSplitBox & anObj)
{
    BinaryDumpInFile(aFp,anObj.KeyMatr());
    BinaryDumpInFile(aFp,anObj.Rab().IsInit());
    if (anObj.Rab().IsInit()) BinaryDumpInFile(aFp,anObj.Rab().Val());
}

cElXMLTree * ToXMLTree(const cMatrixSplitBox & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"MatrixSplitBox",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("KeyMatr"),anObj.KeyMatr())->ReTagThis("KeyMatr"));
   if (anObj.Rab().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Rab"),anObj.Rab().Val())->ReTagThis("Rab"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cMatrixSplitBox & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.KeyMatr(),aTree->Get("KeyMatr",1)); //tototo 

   xml_init(anObj.Rab(),aTree->Get("Rab",1),double(0.0)); //tototo 
}

std::string  Mangling( cMatrixSplitBox *) {return "222360D231B842D0FE3F";};


cTplValGesInit< std::string > & cContenuAPrioriImage::KeyAutoAdaptScale()
{
   return mKeyAutoAdaptScale;
}

const cTplValGesInit< std::string > & cContenuAPrioriImage::KeyAutoAdaptScale()const 
{
   return mKeyAutoAdaptScale;
}


cTplValGesInit< double > & cContenuAPrioriImage::PdsMaxAdaptScale()
{
   return mPdsMaxAdaptScale;
}

const cTplValGesInit< double > & cContenuAPrioriImage::PdsMaxAdaptScale()const 
{
   return mPdsMaxAdaptScale;
}


cTplValGesInit< double > & cContenuAPrioriImage::Scale()
{
   return mScale;
}

const cTplValGesInit< double > & cContenuAPrioriImage::Scale()const 
{
   return mScale;
}


cTplValGesInit< double > & cContenuAPrioriImage::Teta()
{
   return mTeta;
}

const cTplValGesInit< double > & cContenuAPrioriImage::Teta()const 
{
   return mTeta;
}


cTplValGesInit< Box2di > & cContenuAPrioriImage::BoiteEnglob()
{
   return mBoiteEnglob;
}

const cTplValGesInit< Box2di > & cContenuAPrioriImage::BoiteEnglob()const 
{
   return mBoiteEnglob;
}


cTplValGesInit< std::string > & cContenuAPrioriImage::ElInt_CaPImAddedSet()
{
   return mElInt_CaPImAddedSet;
}

const cTplValGesInit< std::string > & cContenuAPrioriImage::ElInt_CaPImAddedSet()const 
{
   return mElInt_CaPImAddedSet;
}


cTplValGesInit< std::string > & cContenuAPrioriImage::ElInt_CaPImMyKey()
{
   return mElInt_CaPImMyKey;
}

const cTplValGesInit< std::string > & cContenuAPrioriImage::ElInt_CaPImMyKey()const 
{
   return mElInt_CaPImMyKey;
}


std::string & cContenuAPrioriImage::KeyMatr()
{
   return MatrixSplitBox().Val().KeyMatr();
}

const std::string & cContenuAPrioriImage::KeyMatr()const 
{
   return MatrixSplitBox().Val().KeyMatr();
}


cTplValGesInit< double > & cContenuAPrioriImage::Rab()
{
   return MatrixSplitBox().Val().Rab();
}

const cTplValGesInit< double > & cContenuAPrioriImage::Rab()const 
{
   return MatrixSplitBox().Val().Rab();
}


cTplValGesInit< cMatrixSplitBox > & cContenuAPrioriImage::MatrixSplitBox()
{
   return mMatrixSplitBox;
}

const cTplValGesInit< cMatrixSplitBox > & cContenuAPrioriImage::MatrixSplitBox()const 
{
   return mMatrixSplitBox;
}

void  BinaryUnDumpFromFile(cContenuAPrioriImage & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.KeyAutoAdaptScale().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.KeyAutoAdaptScale().ValForcedForUnUmp(),aFp);
        }
        else  anObj.KeyAutoAdaptScale().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.PdsMaxAdaptScale().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.PdsMaxAdaptScale().ValForcedForUnUmp(),aFp);
        }
        else  anObj.PdsMaxAdaptScale().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Scale().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Scale().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Scale().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Teta().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Teta().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Teta().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.BoiteEnglob().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.BoiteEnglob().ValForcedForUnUmp(),aFp);
        }
        else  anObj.BoiteEnglob().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ElInt_CaPImAddedSet().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ElInt_CaPImAddedSet().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ElInt_CaPImAddedSet().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ElInt_CaPImMyKey().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ElInt_CaPImMyKey().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ElInt_CaPImMyKey().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.MatrixSplitBox().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.MatrixSplitBox().ValForcedForUnUmp(),aFp);
        }
        else  anObj.MatrixSplitBox().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cContenuAPrioriImage & anObj)
{
    BinaryDumpInFile(aFp,anObj.KeyAutoAdaptScale().IsInit());
    if (anObj.KeyAutoAdaptScale().IsInit()) BinaryDumpInFile(aFp,anObj.KeyAutoAdaptScale().Val());
    BinaryDumpInFile(aFp,anObj.PdsMaxAdaptScale().IsInit());
    if (anObj.PdsMaxAdaptScale().IsInit()) BinaryDumpInFile(aFp,anObj.PdsMaxAdaptScale().Val());
    BinaryDumpInFile(aFp,anObj.Scale().IsInit());
    if (anObj.Scale().IsInit()) BinaryDumpInFile(aFp,anObj.Scale().Val());
    BinaryDumpInFile(aFp,anObj.Teta().IsInit());
    if (anObj.Teta().IsInit()) BinaryDumpInFile(aFp,anObj.Teta().Val());
    BinaryDumpInFile(aFp,anObj.BoiteEnglob().IsInit());
    if (anObj.BoiteEnglob().IsInit()) BinaryDumpInFile(aFp,anObj.BoiteEnglob().Val());
    BinaryDumpInFile(aFp,anObj.ElInt_CaPImAddedSet().IsInit());
    if (anObj.ElInt_CaPImAddedSet().IsInit()) BinaryDumpInFile(aFp,anObj.ElInt_CaPImAddedSet().Val());
    BinaryDumpInFile(aFp,anObj.ElInt_CaPImMyKey().IsInit());
    if (anObj.ElInt_CaPImMyKey().IsInit()) BinaryDumpInFile(aFp,anObj.ElInt_CaPImMyKey().Val());
    BinaryDumpInFile(aFp,anObj.MatrixSplitBox().IsInit());
    if (anObj.MatrixSplitBox().IsInit()) BinaryDumpInFile(aFp,anObj.MatrixSplitBox().Val());
}

cElXMLTree * ToXMLTree(const cContenuAPrioriImage & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ContenuAPrioriImage",eXMLBranche);
   if (anObj.KeyAutoAdaptScale().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("KeyAutoAdaptScale"),anObj.KeyAutoAdaptScale().Val())->ReTagThis("KeyAutoAdaptScale"));
   if (anObj.PdsMaxAdaptScale().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PdsMaxAdaptScale"),anObj.PdsMaxAdaptScale().Val())->ReTagThis("PdsMaxAdaptScale"));
   if (anObj.Scale().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Scale"),anObj.Scale().Val())->ReTagThis("Scale"));
   if (anObj.Teta().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Teta"),anObj.Teta().Val())->ReTagThis("Teta"));
   if (anObj.BoiteEnglob().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("BoiteEnglob"),anObj.BoiteEnglob().Val())->ReTagThis("BoiteEnglob"));
   if (anObj.ElInt_CaPImAddedSet().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ElInt_CaPImAddedSet"),anObj.ElInt_CaPImAddedSet().Val())->ReTagThis("ElInt_CaPImAddedSet"));
   if (anObj.ElInt_CaPImMyKey().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ElInt_CaPImMyKey"),anObj.ElInt_CaPImMyKey().Val())->ReTagThis("ElInt_CaPImMyKey"));
   if (anObj.MatrixSplitBox().IsInit())
      aRes->AddFils(ToXMLTree(anObj.MatrixSplitBox().Val())->ReTagThis("MatrixSplitBox"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cContenuAPrioriImage & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.KeyAutoAdaptScale(),aTree->Get("KeyAutoAdaptScale",1)); //tototo 

   xml_init(anObj.PdsMaxAdaptScale(),aTree->Get("PdsMaxAdaptScale",1),double(0.5)); //tototo 

   xml_init(anObj.Scale(),aTree->Get("Scale",1),double(1.0)); //tototo 

   xml_init(anObj.Teta(),aTree->Get("Teta",1),double(0.0)); //tototo 

   xml_init(anObj.BoiteEnglob(),aTree->Get("BoiteEnglob",1)); //tototo 

   xml_init(anObj.ElInt_CaPImAddedSet(),aTree->Get("ElInt_CaPImAddedSet",1)); //tototo 

   xml_init(anObj.ElInt_CaPImMyKey(),aTree->Get("ElInt_CaPImMyKey",1)); //tototo 

   xml_init(anObj.MatrixSplitBox(),aTree->Get("MatrixSplitBox",1)); //tototo 
}

std::string  Mangling( cContenuAPrioriImage *) {return "B6AD30ECC72316E1FB3F";};


std::list< std::string > & cAPrioriImage::Names()
{
   return mNames;
}

const std::list< std::string > & cAPrioriImage::Names()const 
{
   return mNames;
}


cTplValGesInit< std::string > & cAPrioriImage::KeyedAddedSet()
{
   return mKeyedAddedSet;
}

const cTplValGesInit< std::string > & cAPrioriImage::KeyedAddedSet()const 
{
   return mKeyedAddedSet;
}


cTplValGesInit< std::string > & cAPrioriImage::Key()
{
   return mKey;
}

const cTplValGesInit< std::string > & cAPrioriImage::Key()const 
{
   return mKey;
}


cTplValGesInit< std::string > & cAPrioriImage::KeyAutoAdaptScale()
{
   return ContenuAPrioriImage().KeyAutoAdaptScale();
}

const cTplValGesInit< std::string > & cAPrioriImage::KeyAutoAdaptScale()const 
{
   return ContenuAPrioriImage().KeyAutoAdaptScale();
}


cTplValGesInit< double > & cAPrioriImage::PdsMaxAdaptScale()
{
   return ContenuAPrioriImage().PdsMaxAdaptScale();
}

const cTplValGesInit< double > & cAPrioriImage::PdsMaxAdaptScale()const 
{
   return ContenuAPrioriImage().PdsMaxAdaptScale();
}


cTplValGesInit< double > & cAPrioriImage::Scale()
{
   return ContenuAPrioriImage().Scale();
}

const cTplValGesInit< double > & cAPrioriImage::Scale()const 
{
   return ContenuAPrioriImage().Scale();
}


cTplValGesInit< double > & cAPrioriImage::Teta()
{
   return ContenuAPrioriImage().Teta();
}

const cTplValGesInit< double > & cAPrioriImage::Teta()const 
{
   return ContenuAPrioriImage().Teta();
}


cTplValGesInit< Box2di > & cAPrioriImage::BoiteEnglob()
{
   return ContenuAPrioriImage().BoiteEnglob();
}

const cTplValGesInit< Box2di > & cAPrioriImage::BoiteEnglob()const 
{
   return ContenuAPrioriImage().BoiteEnglob();
}


cTplValGesInit< std::string > & cAPrioriImage::ElInt_CaPImAddedSet()
{
   return ContenuAPrioriImage().ElInt_CaPImAddedSet();
}

const cTplValGesInit< std::string > & cAPrioriImage::ElInt_CaPImAddedSet()const 
{
   return ContenuAPrioriImage().ElInt_CaPImAddedSet();
}


cTplValGesInit< std::string > & cAPrioriImage::ElInt_CaPImMyKey()
{
   return ContenuAPrioriImage().ElInt_CaPImMyKey();
}

const cTplValGesInit< std::string > & cAPrioriImage::ElInt_CaPImMyKey()const 
{
   return ContenuAPrioriImage().ElInt_CaPImMyKey();
}


std::string & cAPrioriImage::KeyMatr()
{
   return ContenuAPrioriImage().MatrixSplitBox().Val().KeyMatr();
}

const std::string & cAPrioriImage::KeyMatr()const 
{
   return ContenuAPrioriImage().MatrixSplitBox().Val().KeyMatr();
}


cTplValGesInit< double > & cAPrioriImage::Rab()
{
   return ContenuAPrioriImage().MatrixSplitBox().Val().Rab();
}

const cTplValGesInit< double > & cAPrioriImage::Rab()const 
{
   return ContenuAPrioriImage().MatrixSplitBox().Val().Rab();
}


cTplValGesInit< cMatrixSplitBox > & cAPrioriImage::MatrixSplitBox()
{
   return ContenuAPrioriImage().MatrixSplitBox();
}

const cTplValGesInit< cMatrixSplitBox > & cAPrioriImage::MatrixSplitBox()const 
{
   return ContenuAPrioriImage().MatrixSplitBox();
}


cContenuAPrioriImage & cAPrioriImage::ContenuAPrioriImage()
{
   return mContenuAPrioriImage;
}

const cContenuAPrioriImage & cAPrioriImage::ContenuAPrioriImage()const 
{
   return mContenuAPrioriImage;
}

void  BinaryUnDumpFromFile(cAPrioriImage & anObj,ELISE_fp & aFp)
{
   { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             std::string aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.Names().push_back(aVal);
        }
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.KeyedAddedSet().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.KeyedAddedSet().ValForcedForUnUmp(),aFp);
        }
        else  anObj.KeyedAddedSet().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Key().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Key().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Key().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.ContenuAPrioriImage(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cAPrioriImage & anObj)
{
    BinaryDumpInFile(aFp,(int)anObj.Names().size());
    for(  std::list< std::string >::const_iterator iT=anObj.Names().begin();
         iT!=anObj.Names().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.KeyedAddedSet().IsInit());
    if (anObj.KeyedAddedSet().IsInit()) BinaryDumpInFile(aFp,anObj.KeyedAddedSet().Val());
    BinaryDumpInFile(aFp,anObj.Key().IsInit());
    if (anObj.Key().IsInit()) BinaryDumpInFile(aFp,anObj.Key().Val());
    BinaryDumpInFile(aFp,anObj.ContenuAPrioriImage());
}

cElXMLTree * ToXMLTree(const cAPrioriImage & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"APrioriImage",eXMLBranche);
  for
  (       std::list< std::string >::const_iterator it=anObj.Names().begin();
      it !=anObj.Names().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("Names"),(*it))->ReTagThis("Names"));
   if (anObj.KeyedAddedSet().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("KeyedAddedSet"),anObj.KeyedAddedSet().Val())->ReTagThis("KeyedAddedSet"));
   if (anObj.Key().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Key"),anObj.Key().Val())->ReTagThis("Key"));
   aRes->AddFils(ToXMLTree(anObj.ContenuAPrioriImage())->ReTagThis("ContenuAPrioriImage"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cAPrioriImage & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Names(),aTree->GetAll("Names",false,1));

   xml_init(anObj.KeyedAddedSet(),aTree->Get("KeyedAddedSet",1)); //tototo 

   xml_init(anObj.Key(),aTree->Get("Key",1),std::string("DefKey")); //tototo 

   xml_init(anObj.ContenuAPrioriImage(),aTree->Get("ContenuAPrioriImage",1)); //tototo 
}

std::string  Mangling( cAPrioriImage *) {return "F69DF8BDB60EBBC8FE3F";};


cTplValGesInit< bool > & cKeyedNamesAssociations::IsParametrized()
{
   return mIsParametrized;
}

const cTplValGesInit< bool > & cKeyedNamesAssociations::IsParametrized()const 
{
   return mIsParametrized;
}


std::list< cAssocNameToName > & cKeyedNamesAssociations::Calcs()
{
   return mCalcs;
}

const std::list< cAssocNameToName > & cKeyedNamesAssociations::Calcs()const 
{
   return mCalcs;
}


std::string & cKeyedNamesAssociations::Key()
{
   return mKey;
}

const std::string & cKeyedNamesAssociations::Key()const 
{
   return mKey;
}


cTplValGesInit< std::string > & cKeyedNamesAssociations::SubDirAutoMake()
{
   return mSubDirAutoMake;
}

const cTplValGesInit< std::string > & cKeyedNamesAssociations::SubDirAutoMake()const 
{
   return mSubDirAutoMake;
}


cTplValGesInit< bool > & cKeyedNamesAssociations::SubDirAutoMakeRec()
{
   return mSubDirAutoMakeRec;
}

const cTplValGesInit< bool > & cKeyedNamesAssociations::SubDirAutoMakeRec()const 
{
   return mSubDirAutoMakeRec;
}

void  BinaryUnDumpFromFile(cKeyedNamesAssociations & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.IsParametrized().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.IsParametrized().ValForcedForUnUmp(),aFp);
        }
        else  anObj.IsParametrized().SetNoInit();
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cAssocNameToName aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.Calcs().push_back(aVal);
        }
  } ;
    BinaryUnDumpFromFile(anObj.Key(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SubDirAutoMake().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SubDirAutoMake().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SubDirAutoMake().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SubDirAutoMakeRec().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SubDirAutoMakeRec().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SubDirAutoMakeRec().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cKeyedNamesAssociations & anObj)
{
    BinaryDumpInFile(aFp,anObj.IsParametrized().IsInit());
    if (anObj.IsParametrized().IsInit()) BinaryDumpInFile(aFp,anObj.IsParametrized().Val());
    BinaryDumpInFile(aFp,(int)anObj.Calcs().size());
    for(  std::list< cAssocNameToName >::const_iterator iT=anObj.Calcs().begin();
         iT!=anObj.Calcs().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.Key());
    BinaryDumpInFile(aFp,anObj.SubDirAutoMake().IsInit());
    if (anObj.SubDirAutoMake().IsInit()) BinaryDumpInFile(aFp,anObj.SubDirAutoMake().Val());
    BinaryDumpInFile(aFp,anObj.SubDirAutoMakeRec().IsInit());
    if (anObj.SubDirAutoMakeRec().IsInit()) BinaryDumpInFile(aFp,anObj.SubDirAutoMakeRec().Val());
}

cElXMLTree * ToXMLTree(const cKeyedNamesAssociations & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"KeyedNamesAssociations",eXMLBranche);
   if (anObj.IsParametrized().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("IsParametrized"),anObj.IsParametrized().Val())->ReTagThis("IsParametrized"));
  for
  (       std::list< cAssocNameToName >::const_iterator it=anObj.Calcs().begin();
      it !=anObj.Calcs().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("Calcs"));
   aRes->AddFils(::ToXMLTree(std::string("Key"),anObj.Key())->ReTagThis("Key"));
   if (anObj.SubDirAutoMake().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SubDirAutoMake"),anObj.SubDirAutoMake().Val())->ReTagThis("SubDirAutoMake"));
   if (anObj.SubDirAutoMakeRec().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SubDirAutoMakeRec"),anObj.SubDirAutoMakeRec().Val())->ReTagThis("SubDirAutoMakeRec"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cKeyedNamesAssociations & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.IsParametrized(),aTree->Get("IsParametrized",1),bool(false)); //tototo 

   xml_init(anObj.Calcs(),aTree->GetAll("Calcs",false,1));

   xml_init(anObj.Key(),aTree->Get("Key",1)); //tototo 

   xml_init(anObj.SubDirAutoMake(),aTree->Get("SubDirAutoMake",1),std::string("")); //tototo 

   xml_init(anObj.SubDirAutoMakeRec(),aTree->Get("SubDirAutoMakeRec",1),bool(false)); //tototo 
}

std::string  Mangling( cKeyedNamesAssociations *) {return "323341C3245B13B1FD3F";};


cTplValGesInit< bool > & cKeyedSetsOfNames::IsParametrized()
{
   return mIsParametrized;
}

const cTplValGesInit< bool > & cKeyedSetsOfNames::IsParametrized()const 
{
   return mIsParametrized;
}


cSetNameDescriptor & cKeyedSetsOfNames::Sets()
{
   return mSets;
}

const cSetNameDescriptor & cKeyedSetsOfNames::Sets()const 
{
   return mSets;
}


std::string & cKeyedSetsOfNames::Key()
{
   return mKey;
}

const std::string & cKeyedSetsOfNames::Key()const 
{
   return mKey;
}

void  BinaryUnDumpFromFile(cKeyedSetsOfNames & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.IsParametrized().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.IsParametrized().ValForcedForUnUmp(),aFp);
        }
        else  anObj.IsParametrized().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.Sets(),aFp);
    BinaryUnDumpFromFile(anObj.Key(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cKeyedSetsOfNames & anObj)
{
    BinaryDumpInFile(aFp,anObj.IsParametrized().IsInit());
    if (anObj.IsParametrized().IsInit()) BinaryDumpInFile(aFp,anObj.IsParametrized().Val());
    BinaryDumpInFile(aFp,anObj.Sets());
    BinaryDumpInFile(aFp,anObj.Key());
}

cElXMLTree * ToXMLTree(const cKeyedSetsOfNames & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"KeyedSetsOfNames",eXMLBranche);
   if (anObj.IsParametrized().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("IsParametrized"),anObj.IsParametrized().Val())->ReTagThis("IsParametrized"));
   aRes->AddFils(ToXMLTree(anObj.Sets())->ReTagThis("Sets"));
   aRes->AddFils(::ToXMLTree(std::string("Key"),anObj.Key())->ReTagThis("Key"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cKeyedSetsOfNames & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.IsParametrized(),aTree->Get("IsParametrized",1),bool(false)); //tototo 

   xml_init(anObj.Sets(),aTree->Get("Sets",1)); //tototo 

   xml_init(anObj.Key(),aTree->Get("Key",1)); //tototo 
}

std::string  Mangling( cKeyedSetsOfNames *) {return "A823CC96191CA8D9FC3F";};


cTplValGesInit< bool > & cKeyedSetsORels::IsParametrized()
{
   return mIsParametrized;
}

const cTplValGesInit< bool > & cKeyedSetsORels::IsParametrized()const 
{
   return mIsParametrized;
}


cNameRelDescriptor & cKeyedSetsORels::Sets()
{
   return mSets;
}

const cNameRelDescriptor & cKeyedSetsORels::Sets()const 
{
   return mSets;
}


std::string & cKeyedSetsORels::Key()
{
   return mKey;
}

const std::string & cKeyedSetsORels::Key()const 
{
   return mKey;
}

void  BinaryUnDumpFromFile(cKeyedSetsORels & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.IsParametrized().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.IsParametrized().ValForcedForUnUmp(),aFp);
        }
        else  anObj.IsParametrized().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.Sets(),aFp);
    BinaryUnDumpFromFile(anObj.Key(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cKeyedSetsORels & anObj)
{
    BinaryDumpInFile(aFp,anObj.IsParametrized().IsInit());
    if (anObj.IsParametrized().IsInit()) BinaryDumpInFile(aFp,anObj.IsParametrized().Val());
    BinaryDumpInFile(aFp,anObj.Sets());
    BinaryDumpInFile(aFp,anObj.Key());
}

cElXMLTree * ToXMLTree(const cKeyedSetsORels & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"KeyedSetsORels",eXMLBranche);
   if (anObj.IsParametrized().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("IsParametrized"),anObj.IsParametrized().Val())->ReTagThis("IsParametrized"));
   aRes->AddFils(ToXMLTree(anObj.Sets())->ReTagThis("Sets"));
   aRes->AddFils(::ToXMLTree(std::string("Key"),anObj.Key())->ReTagThis("Key"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cKeyedSetsORels & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.IsParametrized(),aTree->Get("IsParametrized",1),bool(false)); //tototo 

   xml_init(anObj.Sets(),aTree->Get("Sets",1)); //tototo 

   xml_init(anObj.Key(),aTree->Get("Key",1)); //tototo 
}

std::string  Mangling( cKeyedSetsORels *) {return "BA288C9CA5111BEAFD3F";};


cImMatrixStructuration & cKeyedMatrixStruct::Matrix()
{
   return mMatrix;
}

const cImMatrixStructuration & cKeyedMatrixStruct::Matrix()const 
{
   return mMatrix;
}


std::string & cKeyedMatrixStruct::Key()
{
   return mKey;
}

const std::string & cKeyedMatrixStruct::Key()const 
{
   return mKey;
}

void  BinaryUnDumpFromFile(cKeyedMatrixStruct & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Matrix(),aFp);
    BinaryUnDumpFromFile(anObj.Key(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cKeyedMatrixStruct & anObj)
{
    BinaryDumpInFile(aFp,anObj.Matrix());
    BinaryDumpInFile(aFp,anObj.Key());
}

cElXMLTree * ToXMLTree(const cKeyedMatrixStruct & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"KeyedMatrixStruct",eXMLBranche);
   aRes->AddFils(ToXMLTree(anObj.Matrix())->ReTagThis("Matrix"));
   aRes->AddFils(::ToXMLTree(std::string("Key"),anObj.Key())->ReTagThis("Key"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cKeyedMatrixStruct & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Matrix(),aTree->Get("Matrix",1)); //tototo 

   xml_init(anObj.Key(),aTree->Get("Key",1)); //tototo 
}

std::string  Mangling( cKeyedMatrixStruct *) {return "46D3A9761F048AE9FC3F";};


cTplValGesInit< bool > & cChantierDescripteur::ExitOnBrkp()
{
   return mExitOnBrkp;
}

const cTplValGesInit< bool > & cChantierDescripteur::ExitOnBrkp()const 
{
   return mExitOnBrkp;
}


std::list< std::string > & cChantierDescripteur::Symb()
{
   return mSymb;
}

const std::list< std::string > & cChantierDescripteur::Symb()const 
{
   return mSymb;
}


std::list< std::string > & cChantierDescripteur::eSymb()
{
   return meSymb;
}

const std::list< std::string > & cChantierDescripteur::eSymb()const 
{
   return meSymb;
}


cTplValGesInit< cMMCameraDataBase > & cChantierDescripteur::LocCamDataBase()
{
   return mLocCamDataBase;
}

const cTplValGesInit< cMMCameraDataBase > & cChantierDescripteur::LocCamDataBase()const 
{
   return mLocCamDataBase;
}


std::string & cChantierDescripteur::KeySetCollectXif()
{
   return MakeDataBase().Val().KeySetCollectXif();
}

const std::string & cChantierDescripteur::KeySetCollectXif()const 
{
   return MakeDataBase().Val().KeySetCollectXif();
}


std::list< std::string > & cChantierDescripteur::KeyAssocNameSup()
{
   return MakeDataBase().Val().KeyAssocNameSup();
}

const std::list< std::string > & cChantierDescripteur::KeyAssocNameSup()const 
{
   return MakeDataBase().Val().KeyAssocNameSup();
}


cTplValGesInit< std::string > & cChantierDescripteur::NameFile()
{
   return MakeDataBase().Val().NameFile();
}

const cTplValGesInit< std::string > & cChantierDescripteur::NameFile()const 
{
   return MakeDataBase().Val().NameFile();
}


cTplValGesInit< cMakeDataBase > & cChantierDescripteur::MakeDataBase()
{
   return mMakeDataBase;
}

const cTplValGesInit< cMakeDataBase > & cChantierDescripteur::MakeDataBase()const 
{
   return mMakeDataBase;
}


cTplValGesInit< std::string > & cChantierDescripteur::KeySuprAbs2Rel()
{
   return mKeySuprAbs2Rel;
}

const cTplValGesInit< std::string > & cChantierDescripteur::KeySuprAbs2Rel()const 
{
   return mKeySuprAbs2Rel;
}


std::list< cBatchChantDesc > & cChantierDescripteur::BatchChantDesc()
{
   return mBatchChantDesc;
}

const std::list< cBatchChantDesc > & cChantierDescripteur::BatchChantDesc()const 
{
   return mBatchChantDesc;
}


std::list< cShowChantDesc > & cChantierDescripteur::ShowChantDesc()
{
   return mShowChantDesc;
}

const std::list< cShowChantDesc > & cChantierDescripteur::ShowChantDesc()const 
{
   return mShowChantDesc;
}


std::list< cAPrioriImage > & cChantierDescripteur::APrioriImage()
{
   return mAPrioriImage;
}

const std::list< cAPrioriImage > & cChantierDescripteur::APrioriImage()const 
{
   return mAPrioriImage;
}


std::list< cKeyedNamesAssociations > & cChantierDescripteur::KeyedNamesAssociations()
{
   return mKeyedNamesAssociations;
}

const std::list< cKeyedNamesAssociations > & cChantierDescripteur::KeyedNamesAssociations()const 
{
   return mKeyedNamesAssociations;
}


std::list< cKeyedSetsOfNames > & cChantierDescripteur::KeyedSetsOfNames()
{
   return mKeyedSetsOfNames;
}

const std::list< cKeyedSetsOfNames > & cChantierDescripteur::KeyedSetsOfNames()const 
{
   return mKeyedSetsOfNames;
}


std::list< cKeyedSetsORels > & cChantierDescripteur::KeyedSetsORels()
{
   return mKeyedSetsORels;
}

const std::list< cKeyedSetsORels > & cChantierDescripteur::KeyedSetsORels()const 
{
   return mKeyedSetsORels;
}


std::list< cKeyedMatrixStruct > & cChantierDescripteur::KeyedMatrixStruct()
{
   return mKeyedMatrixStruct;
}

const std::list< cKeyedMatrixStruct > & cChantierDescripteur::KeyedMatrixStruct()const 
{
   return mKeyedMatrixStruct;
}


std::list< cClassEquivDescripteur > & cChantierDescripteur::KeyedClassEquiv()
{
   return mKeyedClassEquiv;
}

const std::list< cClassEquivDescripteur > & cChantierDescripteur::KeyedClassEquiv()const 
{
   return mKeyedClassEquiv;
}


cTplValGesInit< cBaseDataCD > & cChantierDescripteur::BaseDatas()
{
   return mBaseDatas;
}

const cTplValGesInit< cBaseDataCD > & cChantierDescripteur::BaseDatas()const 
{
   return mBaseDatas;
}


std::list< std::string > & cChantierDescripteur::FilesDatas()
{
   return mFilesDatas;
}

const std::list< std::string > & cChantierDescripteur::FilesDatas()const 
{
   return mFilesDatas;
}

void  BinaryUnDumpFromFile(cChantierDescripteur & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ExitOnBrkp().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ExitOnBrkp().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ExitOnBrkp().SetNoInit();
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             std::string aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.Symb().push_back(aVal);
        }
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             std::string aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.eSymb().push_back(aVal);
        }
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.LocCamDataBase().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.LocCamDataBase().ValForcedForUnUmp(),aFp);
        }
        else  anObj.LocCamDataBase().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.MakeDataBase().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.MakeDataBase().ValForcedForUnUmp(),aFp);
        }
        else  anObj.MakeDataBase().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.KeySuprAbs2Rel().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.KeySuprAbs2Rel().ValForcedForUnUmp(),aFp);
        }
        else  anObj.KeySuprAbs2Rel().SetNoInit();
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cBatchChantDesc aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.BatchChantDesc().push_back(aVal);
        }
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cShowChantDesc aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.ShowChantDesc().push_back(aVal);
        }
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cAPrioriImage aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.APrioriImage().push_back(aVal);
        }
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cKeyedNamesAssociations aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.KeyedNamesAssociations().push_back(aVal);
        }
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cKeyedSetsOfNames aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.KeyedSetsOfNames().push_back(aVal);
        }
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cKeyedSetsORels aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.KeyedSetsORels().push_back(aVal);
        }
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cKeyedMatrixStruct aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.KeyedMatrixStruct().push_back(aVal);
        }
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cClassEquivDescripteur aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.KeyedClassEquiv().push_back(aVal);
        }
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.BaseDatas().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.BaseDatas().ValForcedForUnUmp(),aFp);
        }
        else  anObj.BaseDatas().SetNoInit();
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             std::string aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.FilesDatas().push_back(aVal);
        }
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cChantierDescripteur & anObj)
{
    BinaryDumpInFile(aFp,anObj.ExitOnBrkp().IsInit());
    if (anObj.ExitOnBrkp().IsInit()) BinaryDumpInFile(aFp,anObj.ExitOnBrkp().Val());
    BinaryDumpInFile(aFp,(int)anObj.Symb().size());
    for(  std::list< std::string >::const_iterator iT=anObj.Symb().begin();
         iT!=anObj.Symb().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,(int)anObj.eSymb().size());
    for(  std::list< std::string >::const_iterator iT=anObj.eSymb().begin();
         iT!=anObj.eSymb().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.LocCamDataBase().IsInit());
    if (anObj.LocCamDataBase().IsInit()) BinaryDumpInFile(aFp,anObj.LocCamDataBase().Val());
    BinaryDumpInFile(aFp,anObj.MakeDataBase().IsInit());
    if (anObj.MakeDataBase().IsInit()) BinaryDumpInFile(aFp,anObj.MakeDataBase().Val());
    BinaryDumpInFile(aFp,anObj.KeySuprAbs2Rel().IsInit());
    if (anObj.KeySuprAbs2Rel().IsInit()) BinaryDumpInFile(aFp,anObj.KeySuprAbs2Rel().Val());
    BinaryDumpInFile(aFp,(int)anObj.BatchChantDesc().size());
    for(  std::list< cBatchChantDesc >::const_iterator iT=anObj.BatchChantDesc().begin();
         iT!=anObj.BatchChantDesc().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,(int)anObj.ShowChantDesc().size());
    for(  std::list< cShowChantDesc >::const_iterator iT=anObj.ShowChantDesc().begin();
         iT!=anObj.ShowChantDesc().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,(int)anObj.APrioriImage().size());
    for(  std::list< cAPrioriImage >::const_iterator iT=anObj.APrioriImage().begin();
         iT!=anObj.APrioriImage().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,(int)anObj.KeyedNamesAssociations().size());
    for(  std::list< cKeyedNamesAssociations >::const_iterator iT=anObj.KeyedNamesAssociations().begin();
         iT!=anObj.KeyedNamesAssociations().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,(int)anObj.KeyedSetsOfNames().size());
    for(  std::list< cKeyedSetsOfNames >::const_iterator iT=anObj.KeyedSetsOfNames().begin();
         iT!=anObj.KeyedSetsOfNames().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,(int)anObj.KeyedSetsORels().size());
    for(  std::list< cKeyedSetsORels >::const_iterator iT=anObj.KeyedSetsORels().begin();
         iT!=anObj.KeyedSetsORels().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,(int)anObj.KeyedMatrixStruct().size());
    for(  std::list< cKeyedMatrixStruct >::const_iterator iT=anObj.KeyedMatrixStruct().begin();
         iT!=anObj.KeyedMatrixStruct().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,(int)anObj.KeyedClassEquiv().size());
    for(  std::list< cClassEquivDescripteur >::const_iterator iT=anObj.KeyedClassEquiv().begin();
         iT!=anObj.KeyedClassEquiv().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.BaseDatas().IsInit());
    if (anObj.BaseDatas().IsInit()) BinaryDumpInFile(aFp,anObj.BaseDatas().Val());
    BinaryDumpInFile(aFp,(int)anObj.FilesDatas().size());
    for(  std::list< std::string >::const_iterator iT=anObj.FilesDatas().begin();
         iT!=anObj.FilesDatas().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
}

cElXMLTree * ToXMLTree(const cChantierDescripteur & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ChantierDescripteur",eXMLBranche);
   if (anObj.ExitOnBrkp().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ExitOnBrkp"),anObj.ExitOnBrkp().Val())->ReTagThis("ExitOnBrkp"));
  for
  (       std::list< std::string >::const_iterator it=anObj.Symb().begin();
      it !=anObj.Symb().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("Symb"),(*it))->ReTagThis("Symb"));
  for
  (       std::list< std::string >::const_iterator it=anObj.eSymb().begin();
      it !=anObj.eSymb().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("eSymb"),(*it))->ReTagThis("eSymb"));
   if (anObj.LocCamDataBase().IsInit())
      aRes->AddFils(ToXMLTree(anObj.LocCamDataBase().Val())->ReTagThis("LocCamDataBase"));
   if (anObj.MakeDataBase().IsInit())
      aRes->AddFils(ToXMLTree(anObj.MakeDataBase().Val())->ReTagThis("MakeDataBase"));
   if (anObj.KeySuprAbs2Rel().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("KeySuprAbs2Rel"),anObj.KeySuprAbs2Rel().Val())->ReTagThis("KeySuprAbs2Rel"));
  for
  (       std::list< cBatchChantDesc >::const_iterator it=anObj.BatchChantDesc().begin();
      it !=anObj.BatchChantDesc().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("BatchChantDesc"));
  for
  (       std::list< cShowChantDesc >::const_iterator it=anObj.ShowChantDesc().begin();
      it !=anObj.ShowChantDesc().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("ShowChantDesc"));
  for
  (       std::list< cAPrioriImage >::const_iterator it=anObj.APrioriImage().begin();
      it !=anObj.APrioriImage().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("APrioriImage"));
  for
  (       std::list< cKeyedNamesAssociations >::const_iterator it=anObj.KeyedNamesAssociations().begin();
      it !=anObj.KeyedNamesAssociations().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("KeyedNamesAssociations"));
  for
  (       std::list< cKeyedSetsOfNames >::const_iterator it=anObj.KeyedSetsOfNames().begin();
      it !=anObj.KeyedSetsOfNames().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("KeyedSetsOfNames"));
  for
  (       std::list< cKeyedSetsORels >::const_iterator it=anObj.KeyedSetsORels().begin();
      it !=anObj.KeyedSetsORels().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("KeyedSetsORels"));
  for
  (       std::list< cKeyedMatrixStruct >::const_iterator it=anObj.KeyedMatrixStruct().begin();
      it !=anObj.KeyedMatrixStruct().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("KeyedMatrixStruct"));
  for
  (       std::list< cClassEquivDescripteur >::const_iterator it=anObj.KeyedClassEquiv().begin();
      it !=anObj.KeyedClassEquiv().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("KeyedClassEquiv"));
   if (anObj.BaseDatas().IsInit())
      aRes->AddFils(ToXMLTree(anObj.BaseDatas().Val())->ReTagThis("BaseDatas"));
  for
  (       std::list< std::string >::const_iterator it=anObj.FilesDatas().begin();
      it !=anObj.FilesDatas().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("FilesDatas"),(*it))->ReTagThis("FilesDatas"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cChantierDescripteur & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.ExitOnBrkp(),aTree->Get("ExitOnBrkp",1)); //tototo 

   xml_init(anObj.Symb(),aTree->GetAll("Symb",false,1));

   xml_init(anObj.eSymb(),aTree->GetAll("eSymb",false,1));

   xml_init(anObj.LocCamDataBase(),aTree->Get("LocCamDataBase",1)); //tototo 

   xml_init(anObj.MakeDataBase(),aTree->Get("MakeDataBase",1)); //tototo 

   xml_init(anObj.KeySuprAbs2Rel(),aTree->Get("KeySuprAbs2Rel",1)); //tototo 

   xml_init(anObj.BatchChantDesc(),aTree->GetAll("BatchChantDesc",false,1));

   xml_init(anObj.ShowChantDesc(),aTree->GetAll("ShowChantDesc",false,1));

   xml_init(anObj.APrioriImage(),aTree->GetAll("APrioriImage",false,1));

   xml_init(anObj.KeyedNamesAssociations(),aTree->GetAll("KeyedNamesAssociations",false,1));

   xml_init(anObj.KeyedSetsOfNames(),aTree->GetAll("KeyedSetsOfNames",false,1));

   xml_init(anObj.KeyedSetsORels(),aTree->GetAll("KeyedSetsORels",false,1));

   xml_init(anObj.KeyedMatrixStruct(),aTree->GetAll("KeyedMatrixStruct",false,1));

   xml_init(anObj.KeyedClassEquiv(),aTree->GetAll("KeyedClassEquiv",false,1));

   xml_init(anObj.BaseDatas(),aTree->Get("BaseDatas",1)); //tototo 

   xml_init(anObj.FilesDatas(),aTree->GetAll("FilesDatas",false,1));
}

std::string  Mangling( cChantierDescripteur *) {return "609F05915A5FD8EFFA3F";};


int & cXML_Date::year()
{
   return myear;
}

const int & cXML_Date::year()const 
{
   return myear;
}


int & cXML_Date::month()
{
   return mmonth;
}

const int & cXML_Date::month()const 
{
   return mmonth;
}


int & cXML_Date::day()
{
   return mday;
}

const int & cXML_Date::day()const 
{
   return mday;
}


int & cXML_Date::hour()
{
   return mhour;
}

const int & cXML_Date::hour()const 
{
   return mhour;
}


int & cXML_Date::minute()
{
   return mminute;
}

const int & cXML_Date::minute()const 
{
   return mminute;
}


int & cXML_Date::second()
{
   return msecond;
}

const int & cXML_Date::second()const 
{
   return msecond;
}


std::string & cXML_Date::time_system()
{
   return mtime_system;
}

const std::string & cXML_Date::time_system()const 
{
   return mtime_system;
}

void  BinaryUnDumpFromFile(cXML_Date & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.year(),aFp);
    BinaryUnDumpFromFile(anObj.month(),aFp);
    BinaryUnDumpFromFile(anObj.day(),aFp);
    BinaryUnDumpFromFile(anObj.hour(),aFp);
    BinaryUnDumpFromFile(anObj.minute(),aFp);
    BinaryUnDumpFromFile(anObj.second(),aFp);
    BinaryUnDumpFromFile(anObj.time_system(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cXML_Date & anObj)
{
    BinaryDumpInFile(aFp,anObj.year());
    BinaryDumpInFile(aFp,anObj.month());
    BinaryDumpInFile(aFp,anObj.day());
    BinaryDumpInFile(aFp,anObj.hour());
    BinaryDumpInFile(aFp,anObj.minute());
    BinaryDumpInFile(aFp,anObj.second());
    BinaryDumpInFile(aFp,anObj.time_system());
}

cElXMLTree * ToXMLTree(const cXML_Date & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"XML_Date",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("year"),anObj.year())->ReTagThis("year"));
   aRes->AddFils(::ToXMLTree(std::string("month"),anObj.month())->ReTagThis("month"));
   aRes->AddFils(::ToXMLTree(std::string("day"),anObj.day())->ReTagThis("day"));
   aRes->AddFils(::ToXMLTree(std::string("hour"),anObj.hour())->ReTagThis("hour"));
   aRes->AddFils(::ToXMLTree(std::string("minute"),anObj.minute())->ReTagThis("minute"));
   aRes->AddFils(::ToXMLTree(std::string("second"),anObj.second())->ReTagThis("second"));
   aRes->AddFils(::ToXMLTree(std::string("time_system"),anObj.time_system())->ReTagThis("time_system"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cXML_Date & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.year(),aTree->Get("year",1)); //tototo 

   xml_init(anObj.month(),aTree->Get("month",1)); //tototo 

   xml_init(anObj.day(),aTree->Get("day",1)); //tototo 

   xml_init(anObj.hour(),aTree->Get("hour",1)); //tototo 

   xml_init(anObj.minute(),aTree->Get("minute",1)); //tototo 

   xml_init(anObj.second(),aTree->Get("second",1)); //tototo 

   xml_init(anObj.time_system(),aTree->Get("time_system",1)); //tototo 
}

std::string  Mangling( cXML_Date *) {return "6E9DE981E9779EBFFD3F";};


double & cpt3d::x()
{
   return mx;
}

const double & cpt3d::x()const 
{
   return mx;
}


double & cpt3d::y()
{
   return my;
}

const double & cpt3d::y()const 
{
   return my;
}


double & cpt3d::z()
{
   return mz;
}

const double & cpt3d::z()const 
{
   return mz;
}

void  BinaryUnDumpFromFile(cpt3d & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.x(),aFp);
    BinaryUnDumpFromFile(anObj.y(),aFp);
    BinaryUnDumpFromFile(anObj.z(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cpt3d & anObj)
{
    BinaryDumpInFile(aFp,anObj.x());
    BinaryDumpInFile(aFp,anObj.y());
    BinaryDumpInFile(aFp,anObj.z());
}

cElXMLTree * ToXMLTree(const cpt3d & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"pt3d",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("x"),anObj.x())->ReTagThis("x"));
   aRes->AddFils(::ToXMLTree(std::string("y"),anObj.y())->ReTagThis("y"));
   aRes->AddFils(::ToXMLTree(std::string("z"),anObj.z())->ReTagThis("z"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cpt3d & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.x(),aTree->Get("x",1)); //tototo 

   xml_init(anObj.y(),aTree->Get("y",1)); //tototo 

   xml_init(anObj.z(),aTree->Get("z",1)); //tototo 
}

std::string  Mangling( cpt3d *) {return "80553F1BDE3FE09CF73F";};


double & cXML_LinePt3d::x()
{
   return pt3d().x();
}

const double & cXML_LinePt3d::x()const 
{
   return pt3d().x();
}


double & cXML_LinePt3d::y()
{
   return pt3d().y();
}

const double & cXML_LinePt3d::y()const 
{
   return pt3d().y();
}


double & cXML_LinePt3d::z()
{
   return pt3d().z();
}

const double & cXML_LinePt3d::z()const 
{
   return pt3d().z();
}


cpt3d & cXML_LinePt3d::pt3d()
{
   return mpt3d;
}

const cpt3d & cXML_LinePt3d::pt3d()const 
{
   return mpt3d;
}

void  BinaryUnDumpFromFile(cXML_LinePt3d & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.pt3d(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cXML_LinePt3d & anObj)
{
    BinaryDumpInFile(aFp,anObj.pt3d());
}

cElXMLTree * ToXMLTree(const cXML_LinePt3d & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"XML_LinePt3d",eXMLBranche);
   aRes->AddFils(ToXMLTree(anObj.pt3d())->ReTagThis("pt3d"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cXML_LinePt3d & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.pt3d(),aTree->Get("pt3d",1)); //tototo 
}

std::string  Mangling( cXML_LinePt3d *) {return "7CA95D10E84A4AFCFDBF";};


std::string & cauxiliarydata::image_name()
{
   return mimage_name;
}

const std::string & cauxiliarydata::image_name()const 
{
   return mimage_name;
}


std::string & cauxiliarydata::stereopolis()
{
   return mstereopolis;
}

const std::string & cauxiliarydata::stereopolis()const 
{
   return mstereopolis;
}


cXML_Date & cauxiliarydata::image_date()
{
   return mimage_date;
}

const cXML_Date & cauxiliarydata::image_date()const 
{
   return mimage_date;
}


std::list< std::string > & cauxiliarydata::samples()
{
   return msamples;
}

const std::list< std::string > & cauxiliarydata::samples()const 
{
   return msamples;
}

void  BinaryUnDumpFromFile(cauxiliarydata & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.image_name(),aFp);
    BinaryUnDumpFromFile(anObj.stereopolis(),aFp);
    BinaryUnDumpFromFile(anObj.image_date(),aFp);
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             std::string aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.samples().push_back(aVal);
        }
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cauxiliarydata & anObj)
{
    BinaryDumpInFile(aFp,anObj.image_name());
    BinaryDumpInFile(aFp,anObj.stereopolis());
    BinaryDumpInFile(aFp,anObj.image_date());
    BinaryDumpInFile(aFp,(int)anObj.samples().size());
    for(  std::list< std::string >::const_iterator iT=anObj.samples().begin();
         iT!=anObj.samples().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
}

cElXMLTree * ToXMLTree(const cauxiliarydata & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"auxiliarydata",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("image_name"),anObj.image_name())->ReTagThis("image_name"));
   aRes->AddFils(::ToXMLTree(std::string("stereopolis"),anObj.stereopolis())->ReTagThis("stereopolis"));
   aRes->AddFils(ToXMLTree(anObj.image_date())->ReTagThis("image_date"));
  for
  (       std::list< std::string >::const_iterator it=anObj.samples().begin();
      it !=anObj.samples().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("samples"),(*it))->ReTagThis("samples"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cauxiliarydata & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.image_name(),aTree->Get("image_name",1)); //tototo 

   xml_init(anObj.stereopolis(),aTree->Get("stereopolis",1)); //tototo 

   xml_init(anObj.image_date(),aTree->Get("image_date",1)); //tototo 

   xml_init(anObj.samples(),aTree->GetAll("samples",false,1));
}

std::string  Mangling( cauxiliarydata *) {return "38EA4B12BBC0D9BDFD3F";};


double & ceuclidien::x()
{
   return mx;
}

const double & ceuclidien::x()const 
{
   return mx;
}


double & ceuclidien::y()
{
   return my;
}

const double & ceuclidien::y()const 
{
   return my;
}

void  BinaryUnDumpFromFile(ceuclidien & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.x(),aFp);
    BinaryUnDumpFromFile(anObj.y(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const ceuclidien & anObj)
{
    BinaryDumpInFile(aFp,anObj.x());
    BinaryDumpInFile(aFp,anObj.y());
}

cElXMLTree * ToXMLTree(const ceuclidien & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"euclidien",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("x"),anObj.x())->ReTagThis("x"));
   aRes->AddFils(::ToXMLTree(std::string("y"),anObj.y())->ReTagThis("y"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(ceuclidien & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.x(),aTree->Get("x",1)); //tototo 

   xml_init(anObj.y(),aTree->Get("y",1)); //tototo 
}

std::string  Mangling( ceuclidien *) {return "ECA3BF8D654A9EC4FE3F";};


ceuclidien & csysteme::euclidien()
{
   return meuclidien;
}

const ceuclidien & csysteme::euclidien()const 
{
   return meuclidien;
}


std::string & csysteme::geodesique()
{
   return mgeodesique;
}

const std::string & csysteme::geodesique()const 
{
   return mgeodesique;
}

void  BinaryUnDumpFromFile(csysteme & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.euclidien(),aFp);
    BinaryUnDumpFromFile(anObj.geodesique(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const csysteme & anObj)
{
    BinaryDumpInFile(aFp,anObj.euclidien());
    BinaryDumpInFile(aFp,anObj.geodesique());
}

cElXMLTree * ToXMLTree(const csysteme & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"systeme",eXMLBranche);
   aRes->AddFils(ToXMLTree(anObj.euclidien())->ReTagThis("euclidien"));
   aRes->AddFils(::ToXMLTree(std::string("geodesique"),anObj.geodesique())->ReTagThis("geodesique"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(csysteme & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.euclidien(),aTree->Get("euclidien",1)); //tototo 

   xml_init(anObj.geodesique(),aTree->Get("geodesique",1)); //tototo 
}

std::string  Mangling( csysteme *) {return "0E31A7F55973D3F4FD3F";};


double & csommet::easting()
{
   return measting;
}

const double & csommet::easting()const 
{
   return measting;
}


double & csommet::northing()
{
   return mnorthing;
}

const double & csommet::northing()const 
{
   return mnorthing;
}


double & csommet::altitude()
{
   return maltitude;
}

const double & csommet::altitude()const 
{
   return maltitude;
}

void  BinaryUnDumpFromFile(csommet & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.easting(),aFp);
    BinaryUnDumpFromFile(anObj.northing(),aFp);
    BinaryUnDumpFromFile(anObj.altitude(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const csommet & anObj)
{
    BinaryDumpInFile(aFp,anObj.easting());
    BinaryDumpInFile(aFp,anObj.northing());
    BinaryDumpInFile(aFp,anObj.altitude());
}

cElXMLTree * ToXMLTree(const csommet & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"sommet",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("easting"),anObj.easting())->ReTagThis("easting"));
   aRes->AddFils(::ToXMLTree(std::string("northing"),anObj.northing())->ReTagThis("northing"));
   aRes->AddFils(::ToXMLTree(std::string("altitude"),anObj.altitude())->ReTagThis("altitude"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(csommet & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.easting(),aTree->Get("easting",1)); //tototo 

   xml_init(anObj.northing(),aTree->Get("northing",1)); //tototo 

   xml_init(anObj.altitude(),aTree->Get("altitude",1)); //tototo 
}

std::string  Mangling( csommet *) {return "50D12E8422F23FE1FCBF";};


cXML_LinePt3d & cmat3d::l1()
{
   return ml1;
}

const cXML_LinePt3d & cmat3d::l1()const 
{
   return ml1;
}


cXML_LinePt3d & cmat3d::l2()
{
   return ml2;
}

const cXML_LinePt3d & cmat3d::l2()const 
{
   return ml2;
}


cXML_LinePt3d & cmat3d::l3()
{
   return ml3;
}

const cXML_LinePt3d & cmat3d::l3()const 
{
   return ml3;
}

void  BinaryUnDumpFromFile(cmat3d & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.l1(),aFp);
    BinaryUnDumpFromFile(anObj.l2(),aFp);
    BinaryUnDumpFromFile(anObj.l3(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cmat3d & anObj)
{
    BinaryDumpInFile(aFp,anObj.l1());
    BinaryDumpInFile(aFp,anObj.l2());
    BinaryDumpInFile(aFp,anObj.l3());
}

cElXMLTree * ToXMLTree(const cmat3d & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"mat3d",eXMLBranche);
   aRes->AddFils(ToXMLTree(anObj.l1())->ReTagThis("l1"));
   aRes->AddFils(ToXMLTree(anObj.l2())->ReTagThis("l2"));
   aRes->AddFils(ToXMLTree(anObj.l3())->ReTagThis("l3"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cmat3d & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.l1(),aTree->Get("l1",1)); //tototo 

   xml_init(anObj.l2(),aTree->Get("l2",1)); //tototo 

   xml_init(anObj.l3(),aTree->Get("l3",1)); //tototo 
}

std::string  Mangling( cmat3d *) {return "1A9E448D1A310FE4FF3F";};


bool & crotation::Image2Ground()
{
   return mImage2Ground;
}

const bool & crotation::Image2Ground()const 
{
   return mImage2Ground;
}


cXML_LinePt3d & crotation::l1()
{
   return mat3d().l1();
}

const cXML_LinePt3d & crotation::l1()const 
{
   return mat3d().l1();
}


cXML_LinePt3d & crotation::l2()
{
   return mat3d().l2();
}

const cXML_LinePt3d & crotation::l2()const 
{
   return mat3d().l2();
}


cXML_LinePt3d & crotation::l3()
{
   return mat3d().l3();
}

const cXML_LinePt3d & crotation::l3()const 
{
   return mat3d().l3();
}


cmat3d & crotation::mat3d()
{
   return mmat3d;
}

const cmat3d & crotation::mat3d()const 
{
   return mmat3d;
}

void  BinaryUnDumpFromFile(crotation & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Image2Ground(),aFp);
    BinaryUnDumpFromFile(anObj.mat3d(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const crotation & anObj)
{
    BinaryDumpInFile(aFp,anObj.Image2Ground());
    BinaryDumpInFile(aFp,anObj.mat3d());
}

cElXMLTree * ToXMLTree(const crotation & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"rotation",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Image2Ground"),anObj.Image2Ground())->ReTagThis("Image2Ground"));
   aRes->AddFils(ToXMLTree(anObj.mat3d())->ReTagThis("mat3d"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(crotation & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Image2Ground(),aTree->Get("Image2Ground",1)); //tototo 

   xml_init(anObj.mat3d(),aTree->Get("mat3d",1)); //tototo 
}

std::string  Mangling( crotation *) {return "9EE26910406BD3F5FE3F";};


ceuclidien & cextrinseque::euclidien()
{
   return systeme().euclidien();
}

const ceuclidien & cextrinseque::euclidien()const 
{
   return systeme().euclidien();
}


std::string & cextrinseque::geodesique()
{
   return systeme().geodesique();
}

const std::string & cextrinseque::geodesique()const 
{
   return systeme().geodesique();
}


csysteme & cextrinseque::systeme()
{
   return msysteme;
}

const csysteme & cextrinseque::systeme()const 
{
   return msysteme;
}


std::string & cextrinseque::grid_alti()
{
   return mgrid_alti;
}

const std::string & cextrinseque::grid_alti()const 
{
   return mgrid_alti;
}


double & cextrinseque::easting()
{
   return sommet().easting();
}

const double & cextrinseque::easting()const 
{
   return sommet().easting();
}


double & cextrinseque::northing()
{
   return sommet().northing();
}

const double & cextrinseque::northing()const 
{
   return sommet().northing();
}


double & cextrinseque::altitude()
{
   return sommet().altitude();
}

const double & cextrinseque::altitude()const 
{
   return sommet().altitude();
}


csommet & cextrinseque::sommet()
{
   return msommet;
}

const csommet & cextrinseque::sommet()const 
{
   return msommet;
}


bool & cextrinseque::Image2Ground()
{
   return rotation().Image2Ground();
}

const bool & cextrinseque::Image2Ground()const 
{
   return rotation().Image2Ground();
}


cXML_LinePt3d & cextrinseque::l1()
{
   return rotation().mat3d().l1();
}

const cXML_LinePt3d & cextrinseque::l1()const 
{
   return rotation().mat3d().l1();
}


cXML_LinePt3d & cextrinseque::l2()
{
   return rotation().mat3d().l2();
}

const cXML_LinePt3d & cextrinseque::l2()const 
{
   return rotation().mat3d().l2();
}


cXML_LinePt3d & cextrinseque::l3()
{
   return rotation().mat3d().l3();
}

const cXML_LinePt3d & cextrinseque::l3()const 
{
   return rotation().mat3d().l3();
}


cmat3d & cextrinseque::mat3d()
{
   return rotation().mat3d();
}

const cmat3d & cextrinseque::mat3d()const 
{
   return rotation().mat3d();
}


crotation & cextrinseque::rotation()
{
   return mrotation;
}

const crotation & cextrinseque::rotation()const 
{
   return mrotation;
}

void  BinaryUnDumpFromFile(cextrinseque & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.systeme(),aFp);
    BinaryUnDumpFromFile(anObj.grid_alti(),aFp);
    BinaryUnDumpFromFile(anObj.sommet(),aFp);
    BinaryUnDumpFromFile(anObj.rotation(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cextrinseque & anObj)
{
    BinaryDumpInFile(aFp,anObj.systeme());
    BinaryDumpInFile(aFp,anObj.grid_alti());
    BinaryDumpInFile(aFp,anObj.sommet());
    BinaryDumpInFile(aFp,anObj.rotation());
}

cElXMLTree * ToXMLTree(const cextrinseque & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"extrinseque",eXMLBranche);
   aRes->AddFils(ToXMLTree(anObj.systeme())->ReTagThis("systeme"));
   aRes->AddFils(::ToXMLTree(std::string("grid_alti"),anObj.grid_alti())->ReTagThis("grid_alti"));
   aRes->AddFils(ToXMLTree(anObj.sommet())->ReTagThis("sommet"));
   aRes->AddFils(ToXMLTree(anObj.rotation())->ReTagThis("rotation"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cextrinseque & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.systeme(),aTree->Get("systeme",1)); //tototo 

   xml_init(anObj.grid_alti(),aTree->Get("grid_alti",1)); //tototo 

   xml_init(anObj.sommet(),aTree->Get("sommet",1)); //tototo 

   xml_init(anObj.rotation(),aTree->Get("rotation",1)); //tototo 
}

std::string  Mangling( cextrinseque *) {return "295BC46106045ACFFF3F";};


int & cimage_size::width()
{
   return mwidth;
}

const int & cimage_size::width()const 
{
   return mwidth;
}


int & cimage_size::height()
{
   return mheight;
}

const int & cimage_size::height()const 
{
   return mheight;
}

void  BinaryUnDumpFromFile(cimage_size & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.width(),aFp);
    BinaryUnDumpFromFile(anObj.height(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cimage_size & anObj)
{
    BinaryDumpInFile(aFp,anObj.width());
    BinaryDumpInFile(aFp,anObj.height());
}

cElXMLTree * ToXMLTree(const cimage_size & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"image_size",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("width"),anObj.width())->ReTagThis("width"));
   aRes->AddFils(::ToXMLTree(std::string("height"),anObj.height())->ReTagThis("height"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cimage_size & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.width(),aTree->Get("width",1)); //tototo 

   xml_init(anObj.height(),aTree->Get("height",1)); //tototo 
}

std::string  Mangling( cimage_size *) {return "65555394D39584B9FE3F";};


double & cppa::c()
{
   return mc;
}

const double & cppa::c()const 
{
   return mc;
}


double & cppa::l()
{
   return ml;
}

const double & cppa::l()const 
{
   return ml;
}


double & cppa::focale()
{
   return mfocale;
}

const double & cppa::focale()const 
{
   return mfocale;
}

void  BinaryUnDumpFromFile(cppa & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.c(),aFp);
    BinaryUnDumpFromFile(anObj.l(),aFp);
    BinaryUnDumpFromFile(anObj.focale(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cppa & anObj)
{
    BinaryDumpInFile(aFp,anObj.c());
    BinaryDumpInFile(aFp,anObj.l());
    BinaryDumpInFile(aFp,anObj.focale());
}

cElXMLTree * ToXMLTree(const cppa & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ppa",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("c"),anObj.c())->ReTagThis("c"));
   aRes->AddFils(::ToXMLTree(std::string("l"),anObj.l())->ReTagThis("l"));
   aRes->AddFils(::ToXMLTree(std::string("focale"),anObj.focale())->ReTagThis("focale"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cppa & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.c(),aTree->Get("c",1)); //tototo 

   xml_init(anObj.l(),aTree->Get("l",1)); //tototo 

   xml_init(anObj.focale(),aTree->Get("focale",1)); //tototo 
}

std::string  Mangling( cppa *) {return "D62CBDA12F6E5881FDBF";};


double & cpps::c()
{
   return mc;
}

const double & cpps::c()const 
{
   return mc;
}


double & cpps::l()
{
   return ml;
}

const double & cpps::l()const 
{
   return ml;
}

void  BinaryUnDumpFromFile(cpps & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.c(),aFp);
    BinaryUnDumpFromFile(anObj.l(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cpps & anObj)
{
    BinaryDumpInFile(aFp,anObj.c());
    BinaryDumpInFile(aFp,anObj.l());
}

cElXMLTree * ToXMLTree(const cpps & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"pps",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("c"),anObj.c())->ReTagThis("c"));
   aRes->AddFils(::ToXMLTree(std::string("l"),anObj.l())->ReTagThis("l"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cpps & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.c(),aTree->Get("c",1)); //tototo 

   xml_init(anObj.l(),aTree->Get("l",1)); //tototo 
}

std::string  Mangling( cpps *) {return "0A6F5372043BF593FD3F";};


cpps & cdistortion::pps()
{
   return mpps;
}

const cpps & cdistortion::pps()const 
{
   return mpps;
}


double & cdistortion::r1()
{
   return mr1;
}

const double & cdistortion::r1()const 
{
   return mr1;
}


double & cdistortion::r3()
{
   return mr3;
}

const double & cdistortion::r3()const 
{
   return mr3;
}


double & cdistortion::r5()
{
   return mr5;
}

const double & cdistortion::r5()const 
{
   return mr5;
}


double & cdistortion::r7()
{
   return mr7;
}

const double & cdistortion::r7()const 
{
   return mr7;
}

void  BinaryUnDumpFromFile(cdistortion & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.pps(),aFp);
    BinaryUnDumpFromFile(anObj.r1(),aFp);
    BinaryUnDumpFromFile(anObj.r3(),aFp);
    BinaryUnDumpFromFile(anObj.r5(),aFp);
    BinaryUnDumpFromFile(anObj.r7(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cdistortion & anObj)
{
    BinaryDumpInFile(aFp,anObj.pps());
    BinaryDumpInFile(aFp,anObj.r1());
    BinaryDumpInFile(aFp,anObj.r3());
    BinaryDumpInFile(aFp,anObj.r5());
    BinaryDumpInFile(aFp,anObj.r7());
}

cElXMLTree * ToXMLTree(const cdistortion & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"distortion",eXMLBranche);
   aRes->AddFils(ToXMLTree(anObj.pps())->ReTagThis("pps"));
   aRes->AddFils(::ToXMLTree(std::string("r1"),anObj.r1())->ReTagThis("r1"));
   aRes->AddFils(::ToXMLTree(std::string("r3"),anObj.r3())->ReTagThis("r3"));
   aRes->AddFils(::ToXMLTree(std::string("r5"),anObj.r5())->ReTagThis("r5"));
   aRes->AddFils(::ToXMLTree(std::string("r7"),anObj.r7())->ReTagThis("r7"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cdistortion & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.pps(),aTree->Get("pps",1)); //tototo 

   xml_init(anObj.r1(),aTree->Get("r1",1)); //tototo 

   xml_init(anObj.r3(),aTree->Get("r3",1)); //tototo 

   xml_init(anObj.r5(),aTree->Get("r5",1)); //tototo 

   xml_init(anObj.r7(),aTree->Get("r7",1)); //tototo 
}

std::string  Mangling( cdistortion *) {return "80F8E8561153BECFFCBF";};


std::string & csensor::name()
{
   return mname;
}

const std::string & csensor::name()const 
{
   return mname;
}


cXML_Date & csensor::calibration_date()
{
   return mcalibration_date;
}

const cXML_Date & csensor::calibration_date()const 
{
   return mcalibration_date;
}


std::string & csensor::serial_number()
{
   return mserial_number;
}

const std::string & csensor::serial_number()const 
{
   return mserial_number;
}


int & csensor::width()
{
   return image_size().width();
}

const int & csensor::width()const 
{
   return image_size().width();
}


int & csensor::height()
{
   return image_size().height();
}

const int & csensor::height()const 
{
   return image_size().height();
}


cimage_size & csensor::image_size()
{
   return mimage_size;
}

const cimage_size & csensor::image_size()const 
{
   return mimage_size;
}


cppa & csensor::ppa()
{
   return mppa;
}

const cppa & csensor::ppa()const 
{
   return mppa;
}


cpps & csensor::pps()
{
   return distortion().pps();
}

const cpps & csensor::pps()const 
{
   return distortion().pps();
}


double & csensor::r1()
{
   return distortion().r1();
}

const double & csensor::r1()const 
{
   return distortion().r1();
}


double & csensor::r3()
{
   return distortion().r3();
}

const double & csensor::r3()const 
{
   return distortion().r3();
}


double & csensor::r5()
{
   return distortion().r5();
}

const double & csensor::r5()const 
{
   return distortion().r5();
}


double & csensor::r7()
{
   return distortion().r7();
}

const double & csensor::r7()const 
{
   return distortion().r7();
}


cdistortion & csensor::distortion()
{
   return mdistortion;
}

const cdistortion & csensor::distortion()const 
{
   return mdistortion;
}

void  BinaryUnDumpFromFile(csensor & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.name(),aFp);
    BinaryUnDumpFromFile(anObj.calibration_date(),aFp);
    BinaryUnDumpFromFile(anObj.serial_number(),aFp);
    BinaryUnDumpFromFile(anObj.image_size(),aFp);
    BinaryUnDumpFromFile(anObj.ppa(),aFp);
    BinaryUnDumpFromFile(anObj.distortion(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const csensor & anObj)
{
    BinaryDumpInFile(aFp,anObj.name());
    BinaryDumpInFile(aFp,anObj.calibration_date());
    BinaryDumpInFile(aFp,anObj.serial_number());
    BinaryDumpInFile(aFp,anObj.image_size());
    BinaryDumpInFile(aFp,anObj.ppa());
    BinaryDumpInFile(aFp,anObj.distortion());
}

cElXMLTree * ToXMLTree(const csensor & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"sensor",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("name"),anObj.name())->ReTagThis("name"));
   aRes->AddFils(ToXMLTree(anObj.calibration_date())->ReTagThis("calibration_date"));
   aRes->AddFils(::ToXMLTree(std::string("serial_number"),anObj.serial_number())->ReTagThis("serial_number"));
   aRes->AddFils(ToXMLTree(anObj.image_size())->ReTagThis("image_size"));
   aRes->AddFils(ToXMLTree(anObj.ppa())->ReTagThis("ppa"));
   aRes->AddFils(ToXMLTree(anObj.distortion())->ReTagThis("distortion"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(csensor & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.name(),aTree->Get("name",1)); //tototo 

   xml_init(anObj.calibration_date(),aTree->Get("calibration_date",1)); //tototo 

   xml_init(anObj.serial_number(),aTree->Get("serial_number",1)); //tototo 

   xml_init(anObj.image_size(),aTree->Get("image_size",1)); //tototo 

   xml_init(anObj.ppa(),aTree->Get("ppa",1)); //tototo 

   xml_init(anObj.distortion(),aTree->Get("distortion",1)); //tototo 
}

std::string  Mangling( csensor *) {return "85793CA7D29BB2E2FE3F";};


std::string & cintrinseque::name()
{
   return sensor().name();
}

const std::string & cintrinseque::name()const 
{
   return sensor().name();
}


cXML_Date & cintrinseque::calibration_date()
{
   return sensor().calibration_date();
}

const cXML_Date & cintrinseque::calibration_date()const 
{
   return sensor().calibration_date();
}


std::string & cintrinseque::serial_number()
{
   return sensor().serial_number();
}

const std::string & cintrinseque::serial_number()const 
{
   return sensor().serial_number();
}


int & cintrinseque::width()
{
   return sensor().image_size().width();
}

const int & cintrinseque::width()const 
{
   return sensor().image_size().width();
}


int & cintrinseque::height()
{
   return sensor().image_size().height();
}

const int & cintrinseque::height()const 
{
   return sensor().image_size().height();
}


cimage_size & cintrinseque::image_size()
{
   return sensor().image_size();
}

const cimage_size & cintrinseque::image_size()const 
{
   return sensor().image_size();
}


cppa & cintrinseque::ppa()
{
   return sensor().ppa();
}

const cppa & cintrinseque::ppa()const 
{
   return sensor().ppa();
}


cpps & cintrinseque::pps()
{
   return sensor().distortion().pps();
}

const cpps & cintrinseque::pps()const 
{
   return sensor().distortion().pps();
}


double & cintrinseque::r1()
{
   return sensor().distortion().r1();
}

const double & cintrinseque::r1()const 
{
   return sensor().distortion().r1();
}


double & cintrinseque::r3()
{
   return sensor().distortion().r3();
}

const double & cintrinseque::r3()const 
{
   return sensor().distortion().r3();
}


double & cintrinseque::r5()
{
   return sensor().distortion().r5();
}

const double & cintrinseque::r5()const 
{
   return sensor().distortion().r5();
}


double & cintrinseque::r7()
{
   return sensor().distortion().r7();
}

const double & cintrinseque::r7()const 
{
   return sensor().distortion().r7();
}


cdistortion & cintrinseque::distortion()
{
   return sensor().distortion();
}

const cdistortion & cintrinseque::distortion()const 
{
   return sensor().distortion();
}


csensor & cintrinseque::sensor()
{
   return msensor;
}

const csensor & cintrinseque::sensor()const 
{
   return msensor;
}

void  BinaryUnDumpFromFile(cintrinseque & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.sensor(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cintrinseque & anObj)
{
    BinaryDumpInFile(aFp,anObj.sensor());
}

cElXMLTree * ToXMLTree(const cintrinseque & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"intrinseque",eXMLBranche);
   aRes->AddFils(ToXMLTree(anObj.sensor())->ReTagThis("sensor"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cintrinseque & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.sensor(),aTree->Get("sensor",1)); //tototo 
}

std::string  Mangling( cintrinseque *) {return "CEAEB2EDA292AFF9FDBF";};


ceuclidien & cgeometry::euclidien()
{
   return extrinseque().systeme().euclidien();
}

const ceuclidien & cgeometry::euclidien()const 
{
   return extrinseque().systeme().euclidien();
}


std::string & cgeometry::geodesique()
{
   return extrinseque().systeme().geodesique();
}

const std::string & cgeometry::geodesique()const 
{
   return extrinseque().systeme().geodesique();
}


csysteme & cgeometry::systeme()
{
   return extrinseque().systeme();
}

const csysteme & cgeometry::systeme()const 
{
   return extrinseque().systeme();
}


std::string & cgeometry::grid_alti()
{
   return extrinseque().grid_alti();
}

const std::string & cgeometry::grid_alti()const 
{
   return extrinseque().grid_alti();
}


double & cgeometry::easting()
{
   return extrinseque().sommet().easting();
}

const double & cgeometry::easting()const 
{
   return extrinseque().sommet().easting();
}


double & cgeometry::northing()
{
   return extrinseque().sommet().northing();
}

const double & cgeometry::northing()const 
{
   return extrinseque().sommet().northing();
}


double & cgeometry::altitude()
{
   return extrinseque().sommet().altitude();
}

const double & cgeometry::altitude()const 
{
   return extrinseque().sommet().altitude();
}


csommet & cgeometry::sommet()
{
   return extrinseque().sommet();
}

const csommet & cgeometry::sommet()const 
{
   return extrinseque().sommet();
}


bool & cgeometry::Image2Ground()
{
   return extrinseque().rotation().Image2Ground();
}

const bool & cgeometry::Image2Ground()const 
{
   return extrinseque().rotation().Image2Ground();
}


cXML_LinePt3d & cgeometry::l1()
{
   return extrinseque().rotation().mat3d().l1();
}

const cXML_LinePt3d & cgeometry::l1()const 
{
   return extrinseque().rotation().mat3d().l1();
}


cXML_LinePt3d & cgeometry::l2()
{
   return extrinseque().rotation().mat3d().l2();
}

const cXML_LinePt3d & cgeometry::l2()const 
{
   return extrinseque().rotation().mat3d().l2();
}


cXML_LinePt3d & cgeometry::l3()
{
   return extrinseque().rotation().mat3d().l3();
}

const cXML_LinePt3d & cgeometry::l3()const 
{
   return extrinseque().rotation().mat3d().l3();
}


cmat3d & cgeometry::mat3d()
{
   return extrinseque().rotation().mat3d();
}

const cmat3d & cgeometry::mat3d()const 
{
   return extrinseque().rotation().mat3d();
}


crotation & cgeometry::rotation()
{
   return extrinseque().rotation();
}

const crotation & cgeometry::rotation()const 
{
   return extrinseque().rotation();
}


cextrinseque & cgeometry::extrinseque()
{
   return mextrinseque;
}

const cextrinseque & cgeometry::extrinseque()const 
{
   return mextrinseque;
}


std::string & cgeometry::name()
{
   return intrinseque().sensor().name();
}

const std::string & cgeometry::name()const 
{
   return intrinseque().sensor().name();
}


cXML_Date & cgeometry::calibration_date()
{
   return intrinseque().sensor().calibration_date();
}

const cXML_Date & cgeometry::calibration_date()const 
{
   return intrinseque().sensor().calibration_date();
}


std::string & cgeometry::serial_number()
{
   return intrinseque().sensor().serial_number();
}

const std::string & cgeometry::serial_number()const 
{
   return intrinseque().sensor().serial_number();
}


int & cgeometry::width()
{
   return intrinseque().sensor().image_size().width();
}

const int & cgeometry::width()const 
{
   return intrinseque().sensor().image_size().width();
}


int & cgeometry::height()
{
   return intrinseque().sensor().image_size().height();
}

const int & cgeometry::height()const 
{
   return intrinseque().sensor().image_size().height();
}


cimage_size & cgeometry::image_size()
{
   return intrinseque().sensor().image_size();
}

const cimage_size & cgeometry::image_size()const 
{
   return intrinseque().sensor().image_size();
}


cppa & cgeometry::ppa()
{
   return intrinseque().sensor().ppa();
}

const cppa & cgeometry::ppa()const 
{
   return intrinseque().sensor().ppa();
}


cpps & cgeometry::pps()
{
   return intrinseque().sensor().distortion().pps();
}

const cpps & cgeometry::pps()const 
{
   return intrinseque().sensor().distortion().pps();
}


double & cgeometry::r1()
{
   return intrinseque().sensor().distortion().r1();
}

const double & cgeometry::r1()const 
{
   return intrinseque().sensor().distortion().r1();
}


double & cgeometry::r3()
{
   return intrinseque().sensor().distortion().r3();
}

const double & cgeometry::r3()const 
{
   return intrinseque().sensor().distortion().r3();
}


double & cgeometry::r5()
{
   return intrinseque().sensor().distortion().r5();
}

const double & cgeometry::r5()const 
{
   return intrinseque().sensor().distortion().r5();
}


double & cgeometry::r7()
{
   return intrinseque().sensor().distortion().r7();
}

const double & cgeometry::r7()const 
{
   return intrinseque().sensor().distortion().r7();
}


cdistortion & cgeometry::distortion()
{
   return intrinseque().sensor().distortion();
}

const cdistortion & cgeometry::distortion()const 
{
   return intrinseque().sensor().distortion();
}


csensor & cgeometry::sensor()
{
   return intrinseque().sensor();
}

const csensor & cgeometry::sensor()const 
{
   return intrinseque().sensor();
}


cintrinseque & cgeometry::intrinseque()
{
   return mintrinseque;
}

const cintrinseque & cgeometry::intrinseque()const 
{
   return mintrinseque;
}

void  BinaryUnDumpFromFile(cgeometry & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.extrinseque(),aFp);
    BinaryUnDumpFromFile(anObj.intrinseque(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cgeometry & anObj)
{
    BinaryDumpInFile(aFp,anObj.extrinseque());
    BinaryDumpInFile(aFp,anObj.intrinseque());
}

cElXMLTree * ToXMLTree(const cgeometry & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"geometry",eXMLBranche);
   aRes->AddFils(ToXMLTree(anObj.extrinseque())->ReTagThis("extrinseque"));
   aRes->AddFils(ToXMLTree(anObj.intrinseque())->ReTagThis("intrinseque"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cgeometry & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.extrinseque(),aTree->Get("extrinseque",1)); //tototo 

   xml_init(anObj.intrinseque(),aTree->Get("intrinseque",1)); //tototo 
}

std::string  Mangling( cgeometry *) {return "F9AA3EAAE70C2696FF3F";};


double & corientation::version()
{
   return mversion;
}

const double & corientation::version()const 
{
   return mversion;
}


std::string & corientation::image_name()
{
   return auxiliarydata().image_name();
}

const std::string & corientation::image_name()const 
{
   return auxiliarydata().image_name();
}


std::string & corientation::stereopolis()
{
   return auxiliarydata().stereopolis();
}

const std::string & corientation::stereopolis()const 
{
   return auxiliarydata().stereopolis();
}


cXML_Date & corientation::image_date()
{
   return auxiliarydata().image_date();
}

const cXML_Date & corientation::image_date()const 
{
   return auxiliarydata().image_date();
}


std::list< std::string > & corientation::samples()
{
   return auxiliarydata().samples();
}

const std::list< std::string > & corientation::samples()const 
{
   return auxiliarydata().samples();
}


cauxiliarydata & corientation::auxiliarydata()
{
   return mauxiliarydata;
}

const cauxiliarydata & corientation::auxiliarydata()const 
{
   return mauxiliarydata;
}


ceuclidien & corientation::euclidien()
{
   return geometry().extrinseque().systeme().euclidien();
}

const ceuclidien & corientation::euclidien()const 
{
   return geometry().extrinseque().systeme().euclidien();
}


std::string & corientation::geodesique()
{
   return geometry().extrinseque().systeme().geodesique();
}

const std::string & corientation::geodesique()const 
{
   return geometry().extrinseque().systeme().geodesique();
}


csysteme & corientation::systeme()
{
   return geometry().extrinseque().systeme();
}

const csysteme & corientation::systeme()const 
{
   return geometry().extrinseque().systeme();
}


std::string & corientation::grid_alti()
{
   return geometry().extrinseque().grid_alti();
}

const std::string & corientation::grid_alti()const 
{
   return geometry().extrinseque().grid_alti();
}


double & corientation::easting()
{
   return geometry().extrinseque().sommet().easting();
}

const double & corientation::easting()const 
{
   return geometry().extrinseque().sommet().easting();
}


double & corientation::northing()
{
   return geometry().extrinseque().sommet().northing();
}

const double & corientation::northing()const 
{
   return geometry().extrinseque().sommet().northing();
}


double & corientation::altitude()
{
   return geometry().extrinseque().sommet().altitude();
}

const double & corientation::altitude()const 
{
   return geometry().extrinseque().sommet().altitude();
}


csommet & corientation::sommet()
{
   return geometry().extrinseque().sommet();
}

const csommet & corientation::sommet()const 
{
   return geometry().extrinseque().sommet();
}


bool & corientation::Image2Ground()
{
   return geometry().extrinseque().rotation().Image2Ground();
}

const bool & corientation::Image2Ground()const 
{
   return geometry().extrinseque().rotation().Image2Ground();
}


cXML_LinePt3d & corientation::l1()
{
   return geometry().extrinseque().rotation().mat3d().l1();
}

const cXML_LinePt3d & corientation::l1()const 
{
   return geometry().extrinseque().rotation().mat3d().l1();
}


cXML_LinePt3d & corientation::l2()
{
   return geometry().extrinseque().rotation().mat3d().l2();
}

const cXML_LinePt3d & corientation::l2()const 
{
   return geometry().extrinseque().rotation().mat3d().l2();
}


cXML_LinePt3d & corientation::l3()
{
   return geometry().extrinseque().rotation().mat3d().l3();
}

const cXML_LinePt3d & corientation::l3()const 
{
   return geometry().extrinseque().rotation().mat3d().l3();
}


cmat3d & corientation::mat3d()
{
   return geometry().extrinseque().rotation().mat3d();
}

const cmat3d & corientation::mat3d()const 
{
   return geometry().extrinseque().rotation().mat3d();
}


crotation & corientation::rotation()
{
   return geometry().extrinseque().rotation();
}

const crotation & corientation::rotation()const 
{
   return geometry().extrinseque().rotation();
}


cextrinseque & corientation::extrinseque()
{
   return geometry().extrinseque();
}

const cextrinseque & corientation::extrinseque()const 
{
   return geometry().extrinseque();
}


std::string & corientation::name()
{
   return geometry().intrinseque().sensor().name();
}

const std::string & corientation::name()const 
{
   return geometry().intrinseque().sensor().name();
}


cXML_Date & corientation::calibration_date()
{
   return geometry().intrinseque().sensor().calibration_date();
}

const cXML_Date & corientation::calibration_date()const 
{
   return geometry().intrinseque().sensor().calibration_date();
}


std::string & corientation::serial_number()
{
   return geometry().intrinseque().sensor().serial_number();
}

const std::string & corientation::serial_number()const 
{
   return geometry().intrinseque().sensor().serial_number();
}


int & corientation::width()
{
   return geometry().intrinseque().sensor().image_size().width();
}

const int & corientation::width()const 
{
   return geometry().intrinseque().sensor().image_size().width();
}


int & corientation::height()
{
   return geometry().intrinseque().sensor().image_size().height();
}

const int & corientation::height()const 
{
   return geometry().intrinseque().sensor().image_size().height();
}


cimage_size & corientation::image_size()
{
   return geometry().intrinseque().sensor().image_size();
}

const cimage_size & corientation::image_size()const 
{
   return geometry().intrinseque().sensor().image_size();
}


cppa & corientation::ppa()
{
   return geometry().intrinseque().sensor().ppa();
}

const cppa & corientation::ppa()const 
{
   return geometry().intrinseque().sensor().ppa();
}


cpps & corientation::pps()
{
   return geometry().intrinseque().sensor().distortion().pps();
}

const cpps & corientation::pps()const 
{
   return geometry().intrinseque().sensor().distortion().pps();
}


double & corientation::r1()
{
   return geometry().intrinseque().sensor().distortion().r1();
}

const double & corientation::r1()const 
{
   return geometry().intrinseque().sensor().distortion().r1();
}


double & corientation::r3()
{
   return geometry().intrinseque().sensor().distortion().r3();
}

const double & corientation::r3()const 
{
   return geometry().intrinseque().sensor().distortion().r3();
}


double & corientation::r5()
{
   return geometry().intrinseque().sensor().distortion().r5();
}

const double & corientation::r5()const 
{
   return geometry().intrinseque().sensor().distortion().r5();
}


double & corientation::r7()
{
   return geometry().intrinseque().sensor().distortion().r7();
}

const double & corientation::r7()const 
{
   return geometry().intrinseque().sensor().distortion().r7();
}


cdistortion & corientation::distortion()
{
   return geometry().intrinseque().sensor().distortion();
}

const cdistortion & corientation::distortion()const 
{
   return geometry().intrinseque().sensor().distortion();
}


csensor & corientation::sensor()
{
   return geometry().intrinseque().sensor();
}

const csensor & corientation::sensor()const 
{
   return geometry().intrinseque().sensor();
}


cintrinseque & corientation::intrinseque()
{
   return geometry().intrinseque();
}

const cintrinseque & corientation::intrinseque()const 
{
   return geometry().intrinseque();
}


cgeometry & corientation::geometry()
{
   return mgeometry;
}

const cgeometry & corientation::geometry()const 
{
   return mgeometry;
}

void  BinaryUnDumpFromFile(corientation & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.version(),aFp);
    BinaryUnDumpFromFile(anObj.auxiliarydata(),aFp);
    BinaryUnDumpFromFile(anObj.geometry(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const corientation & anObj)
{
    BinaryDumpInFile(aFp,anObj.version());
    BinaryDumpInFile(aFp,anObj.auxiliarydata());
    BinaryDumpInFile(aFp,anObj.geometry());
}

cElXMLTree * ToXMLTree(const corientation & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"orientation",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("version"),anObj.version())->ReTagThis("version"));
   aRes->AddFils(ToXMLTree(anObj.auxiliarydata())->ReTagThis("auxiliarydata"));
   aRes->AddFils(ToXMLTree(anObj.geometry())->ReTagThis("geometry"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(corientation & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.version(),aTree->Get("version",1)); //tototo 

   xml_init(anObj.auxiliarydata(),aTree->Get("auxiliarydata",1)); //tototo 

   xml_init(anObj.geometry(),aTree->Get("geometry",1)); //tototo 
}

std::string  Mangling( corientation *) {return "4E8A9F1A8D0867B0FE3F";};


std::list< std::string > & cOneSolImageSec::Images()
{
   return mImages;
}

const std::list< std::string > & cOneSolImageSec::Images()const 
{
   return mImages;
}


double & cOneSolImageSec::Coverage()
{
   return mCoverage;
}

const double & cOneSolImageSec::Coverage()const 
{
   return mCoverage;
}


double & cOneSolImageSec::Score()
{
   return mScore;
}

const double & cOneSolImageSec::Score()const 
{
   return mScore;
}

void  BinaryUnDumpFromFile(cOneSolImageSec & anObj,ELISE_fp & aFp)
{
   { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             std::string aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.Images().push_back(aVal);
        }
  } ;
    BinaryUnDumpFromFile(anObj.Coverage(),aFp);
    BinaryUnDumpFromFile(anObj.Score(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cOneSolImageSec & anObj)
{
    BinaryDumpInFile(aFp,(int)anObj.Images().size());
    for(  std::list< std::string >::const_iterator iT=anObj.Images().begin();
         iT!=anObj.Images().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.Coverage());
    BinaryDumpInFile(aFp,anObj.Score());
}

cElXMLTree * ToXMLTree(const cOneSolImageSec & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"OneSolImageSec",eXMLBranche);
  for
  (       std::list< std::string >::const_iterator it=anObj.Images().begin();
      it !=anObj.Images().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("Images"),(*it))->ReTagThis("Images"));
   aRes->AddFils(::ToXMLTree(std::string("Coverage"),anObj.Coverage())->ReTagThis("Coverage"));
   aRes->AddFils(::ToXMLTree(std::string("Score"),anObj.Score())->ReTagThis("Score"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cOneSolImageSec & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Images(),aTree->GetAll("Images",false,1));

   xml_init(anObj.Coverage(),aTree->Get("Coverage",1)); //tototo 

   xml_init(anObj.Score(),aTree->Get("Score",1)); //tototo 
}

std::string  Mangling( cOneSolImageSec *) {return "EAF6D9FE51CF3A93FC3F";};


std::string & cISOM_Vois::Name()
{
   return mName;
}

const std::string & cISOM_Vois::Name()const 
{
   return mName;
}


double & cISOM_Vois::Angle()
{
   return mAngle;
}

const double & cISOM_Vois::Angle()const 
{
   return mAngle;
}


double & cISOM_Vois::Nb()
{
   return mNb;
}

const double & cISOM_Vois::Nb()const 
{
   return mNb;
}

void  BinaryUnDumpFromFile(cISOM_Vois & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Name(),aFp);
    BinaryUnDumpFromFile(anObj.Angle(),aFp);
    BinaryUnDumpFromFile(anObj.Nb(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cISOM_Vois & anObj)
{
    BinaryDumpInFile(aFp,anObj.Name());
    BinaryDumpInFile(aFp,anObj.Angle());
    BinaryDumpInFile(aFp,anObj.Nb());
}

cElXMLTree * ToXMLTree(const cISOM_Vois & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ISOM_Vois",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Name"),anObj.Name())->ReTagThis("Name"));
   aRes->AddFils(::ToXMLTree(std::string("Angle"),anObj.Angle())->ReTagThis("Angle"));
   aRes->AddFils(::ToXMLTree(std::string("Nb"),anObj.Nb())->ReTagThis("Nb"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cISOM_Vois & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Name(),aTree->Get("Name",1)); //tototo 

   xml_init(anObj.Angle(),aTree->Get("Angle",1)); //tototo 

   xml_init(anObj.Nb(),aTree->Get("Nb",1)); //tototo 
}

std::string  Mangling( cISOM_Vois *) {return "F1447E0A24AD088AFF3F";};


std::list< cISOM_Vois > & cISOM_AllVois::ISOM_Vois()
{
   return mISOM_Vois;
}

const std::list< cISOM_Vois > & cISOM_AllVois::ISOM_Vois()const 
{
   return mISOM_Vois;
}

void  BinaryUnDumpFromFile(cISOM_AllVois & anObj,ELISE_fp & aFp)
{
   { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cISOM_Vois aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.ISOM_Vois().push_back(aVal);
        }
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cISOM_AllVois & anObj)
{
    BinaryDumpInFile(aFp,(int)anObj.ISOM_Vois().size());
    for(  std::list< cISOM_Vois >::const_iterator iT=anObj.ISOM_Vois().begin();
         iT!=anObj.ISOM_Vois().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
}

cElXMLTree * ToXMLTree(const cISOM_AllVois & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ISOM_AllVois",eXMLBranche);
  for
  (       std::list< cISOM_Vois >::const_iterator it=anObj.ISOM_Vois().begin();
      it !=anObj.ISOM_Vois().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("ISOM_Vois"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cISOM_AllVois & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.ISOM_Vois(),aTree->GetAll("ISOM_Vois",false,1));
}

std::string  Mangling( cISOM_AllVois *) {return "23C5E225E88EDFD3FE3F";};


std::string & cImSecOfMaster::Master()
{
   return mMaster;
}

const std::string & cImSecOfMaster::Master()const 
{
   return mMaster;
}


std::list< cOneSolImageSec > & cImSecOfMaster::Sols()
{
   return mSols;
}

const std::list< cOneSolImageSec > & cImSecOfMaster::Sols()const 
{
   return mSols;
}


cTplValGesInit< cISOM_AllVois > & cImSecOfMaster::ISOM_AllVois()
{
   return mISOM_AllVois;
}

const cTplValGesInit< cISOM_AllVois > & cImSecOfMaster::ISOM_AllVois()const 
{
   return mISOM_AllVois;
}

void  BinaryUnDumpFromFile(cImSecOfMaster & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Master(),aFp);
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cOneSolImageSec aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.Sols().push_back(aVal);
        }
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ISOM_AllVois().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ISOM_AllVois().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ISOM_AllVois().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cImSecOfMaster & anObj)
{
    BinaryDumpInFile(aFp,anObj.Master());
    BinaryDumpInFile(aFp,(int)anObj.Sols().size());
    for(  std::list< cOneSolImageSec >::const_iterator iT=anObj.Sols().begin();
         iT!=anObj.Sols().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.ISOM_AllVois().IsInit());
    if (anObj.ISOM_AllVois().IsInit()) BinaryDumpInFile(aFp,anObj.ISOM_AllVois().Val());
}

cElXMLTree * ToXMLTree(const cImSecOfMaster & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ImSecOfMaster",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Master"),anObj.Master())->ReTagThis("Master"));
  for
  (       std::list< cOneSolImageSec >::const_iterator it=anObj.Sols().begin();
      it !=anObj.Sols().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("Sols"));
   if (anObj.ISOM_AllVois().IsInit())
      aRes->AddFils(ToXMLTree(anObj.ISOM_AllVois().Val())->ReTagThis("ISOM_AllVois"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cImSecOfMaster & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Master(),aTree->Get("Master",1)); //tototo 

   xml_init(anObj.Sols(),aTree->GetAll("Sols",false,1));

   xml_init(anObj.ISOM_AllVois(),aTree->Get("ISOM_AllVois",1)); //tototo 
}

std::string  Mangling( cImSecOfMaster *) {return "E0CF21D70AAA4AE0FE3F";};


std::string & cParamOrientSHC::IdGrp()
{
   return mIdGrp;
}

const std::string & cParamOrientSHC::IdGrp()const 
{
   return mIdGrp;
}


Pt3dr & cParamOrientSHC::Vecteur()
{
   return mVecteur;
}

const Pt3dr & cParamOrientSHC::Vecteur()const 
{
   return mVecteur;
}


cTypeCodageMatr & cParamOrientSHC::Rot()
{
   return mRot;
}

const cTypeCodageMatr & cParamOrientSHC::Rot()const 
{
   return mRot;
}

void  BinaryUnDumpFromFile(cParamOrientSHC & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.IdGrp(),aFp);
    BinaryUnDumpFromFile(anObj.Vecteur(),aFp);
    BinaryUnDumpFromFile(anObj.Rot(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cParamOrientSHC & anObj)
{
    BinaryDumpInFile(aFp,anObj.IdGrp());
    BinaryDumpInFile(aFp,anObj.Vecteur());
    BinaryDumpInFile(aFp,anObj.Rot());
}

cElXMLTree * ToXMLTree(const cParamOrientSHC & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ParamOrientSHC",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("IdGrp"),anObj.IdGrp())->ReTagThis("IdGrp"));
   aRes->AddFils(ToXMLTree(std::string("Vecteur"),anObj.Vecteur())->ReTagThis("Vecteur"));
   aRes->AddFils(ToXMLTree(anObj.Rot())->ReTagThis("Rot"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cParamOrientSHC & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.IdGrp(),aTree->Get("IdGrp",1)); //tototo 

   xml_init(anObj.Vecteur(),aTree->Get("Vecteur",1)); //tototo 

   xml_init(anObj.Rot(),aTree->Get("Rot",1)); //tototo 
}

std::string  Mangling( cParamOrientSHC *) {return "1839B2585465798AFBBF";};


std::list< cParamOrientSHC > & cLiaisonsSHC::ParamOrientSHC()
{
   return mParamOrientSHC;
}

const std::list< cParamOrientSHC > & cLiaisonsSHC::ParamOrientSHC()const 
{
   return mParamOrientSHC;
}

void  BinaryUnDumpFromFile(cLiaisonsSHC & anObj,ELISE_fp & aFp)
{
   { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cParamOrientSHC aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.ParamOrientSHC().push_back(aVal);
        }
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cLiaisonsSHC & anObj)
{
    BinaryDumpInFile(aFp,(int)anObj.ParamOrientSHC().size());
    for(  std::list< cParamOrientSHC >::const_iterator iT=anObj.ParamOrientSHC().begin();
         iT!=anObj.ParamOrientSHC().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
}

cElXMLTree * ToXMLTree(const cLiaisonsSHC & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"LiaisonsSHC",eXMLBranche);
  for
  (       std::list< cParamOrientSHC >::const_iterator it=anObj.ParamOrientSHC().begin();
      it !=anObj.ParamOrientSHC().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("ParamOrientSHC"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cLiaisonsSHC & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.ParamOrientSHC(),aTree->GetAll("ParamOrientSHC",false,1));
}

std::string  Mangling( cLiaisonsSHC *) {return "8614379849CDAEAFFE3F";};


std::string & cStructBlockCam::KeyIm2TimeCam()
{
   return mKeyIm2TimeCam;
}

const std::string & cStructBlockCam::KeyIm2TimeCam()const 
{
   return mKeyIm2TimeCam;
}


std::list< cParamOrientSHC > & cStructBlockCam::ParamOrientSHC()
{
   return LiaisonsSHC().Val().ParamOrientSHC();
}

const std::list< cParamOrientSHC > & cStructBlockCam::ParamOrientSHC()const 
{
   return LiaisonsSHC().Val().ParamOrientSHC();
}


cTplValGesInit< cLiaisonsSHC > & cStructBlockCam::LiaisonsSHC()
{
   return mLiaisonsSHC;
}

const cTplValGesInit< cLiaisonsSHC > & cStructBlockCam::LiaisonsSHC()const 
{
   return mLiaisonsSHC;
}

void  BinaryUnDumpFromFile(cStructBlockCam & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.KeyIm2TimeCam(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.LiaisonsSHC().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.LiaisonsSHC().ValForcedForUnUmp(),aFp);
        }
        else  anObj.LiaisonsSHC().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cStructBlockCam & anObj)
{
    BinaryDumpInFile(aFp,anObj.KeyIm2TimeCam());
    BinaryDumpInFile(aFp,anObj.LiaisonsSHC().IsInit());
    if (anObj.LiaisonsSHC().IsInit()) BinaryDumpInFile(aFp,anObj.LiaisonsSHC().Val());
}

cElXMLTree * ToXMLTree(const cStructBlockCam & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"StructBlockCam",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("KeyIm2TimeCam"),anObj.KeyIm2TimeCam())->ReTagThis("KeyIm2TimeCam"));
   if (anObj.LiaisonsSHC().IsInit())
      aRes->AddFils(ToXMLTree(anObj.LiaisonsSHC().Val())->ReTagThis("LiaisonsSHC"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cStructBlockCam & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.KeyIm2TimeCam(),aTree->Get("KeyIm2TimeCam",1)); //tototo 

   xml_init(anObj.LiaisonsSHC(),aTree->Get("LiaisonsSHC",1)); //tototo 
}

std::string  Mangling( cStructBlockCam *) {return "B06598583EB111DCFE3F";};


std::list< std::string > & cXmlExivEntry::Names()
{
   return mNames;
}

const std::list< std::string > & cXmlExivEntry::Names()const 
{
   return mNames;
}


double & cXmlExivEntry::Focale()
{
   return mFocale;
}

const double & cXmlExivEntry::Focale()const 
{
   return mFocale;
}

void  BinaryUnDumpFromFile(cXmlExivEntry & anObj,ELISE_fp & aFp)
{
   { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             std::string aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.Names().push_back(aVal);
        }
  } ;
    BinaryUnDumpFromFile(anObj.Focale(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cXmlExivEntry & anObj)
{
    BinaryDumpInFile(aFp,(int)anObj.Names().size());
    for(  std::list< std::string >::const_iterator iT=anObj.Names().begin();
         iT!=anObj.Names().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.Focale());
}

cElXMLTree * ToXMLTree(const cXmlExivEntry & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"XmlExivEntry",eXMLBranche);
  for
  (       std::list< std::string >::const_iterator it=anObj.Names().begin();
      it !=anObj.Names().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("Names"),(*it))->ReTagThis("Names"));
   aRes->AddFils(::ToXMLTree(std::string("Focale"),anObj.Focale())->ReTagThis("Focale"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cXmlExivEntry & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Names(),aTree->GetAll("Names",false,1));

   xml_init(anObj.Focale(),aTree->Get("Focale",1)); //tototo 
}

std::string  Mangling( cXmlExivEntry *) {return "0C4C0FEBB27288FFFE3F";};


int & cXmlDataBase::MajNumVers()
{
   return mMajNumVers;
}

const int & cXmlDataBase::MajNumVers()const 
{
   return mMajNumVers;
}


int & cXmlDataBase::MinNumVers()
{
   return mMinNumVers;
}

const int & cXmlDataBase::MinNumVers()const 
{
   return mMinNumVers;
}


std::list< cXmlExivEntry > & cXmlDataBase::Exiv()
{
   return mExiv;
}

const std::list< cXmlExivEntry > & cXmlDataBase::Exiv()const 
{
   return mExiv;
}

void  BinaryUnDumpFromFile(cXmlDataBase & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.MajNumVers(),aFp);
    BinaryUnDumpFromFile(anObj.MinNumVers(),aFp);
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cXmlExivEntry aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.Exiv().push_back(aVal);
        }
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cXmlDataBase & anObj)
{
    BinaryDumpInFile(aFp,anObj.MajNumVers());
    BinaryDumpInFile(aFp,anObj.MinNumVers());
    BinaryDumpInFile(aFp,(int)anObj.Exiv().size());
    for(  std::list< cXmlExivEntry >::const_iterator iT=anObj.Exiv().begin();
         iT!=anObj.Exiv().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
}

cElXMLTree * ToXMLTree(const cXmlDataBase & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"XmlDataBase",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("MajNumVers"),anObj.MajNumVers())->ReTagThis("MajNumVers"));
   aRes->AddFils(::ToXMLTree(std::string("MinNumVers"),anObj.MinNumVers())->ReTagThis("MinNumVers"));
  for
  (       std::list< cXmlExivEntry >::const_iterator it=anObj.Exiv().begin();
      it !=anObj.Exiv().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("Exiv"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cXmlDataBase & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.MajNumVers(),aTree->Get("MajNumVers",1)); //tototo 

   xml_init(anObj.MinNumVers(),aTree->Get("MinNumVers",1)); //tototo 

   xml_init(anObj.Exiv(),aTree->GetAll("Exiv",false,1));
}

std::string  Mangling( cXmlDataBase *) {return "5D80473B6933F396FE3F";};


std::string & cListImByDelta::KeySplitName()
{
   return mKeySplitName;
}

const std::string & cListImByDelta::KeySplitName()const 
{
   return mKeySplitName;
}


std::list< int > & cListImByDelta::Delta()
{
   return mDelta;
}

const std::list< int > & cListImByDelta::Delta()const 
{
   return mDelta;
}

void  BinaryUnDumpFromFile(cListImByDelta & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.KeySplitName(),aFp);
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             int aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.Delta().push_back(aVal);
        }
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cListImByDelta & anObj)
{
    BinaryDumpInFile(aFp,anObj.KeySplitName());
    BinaryDumpInFile(aFp,(int)anObj.Delta().size());
    for(  std::list< int >::const_iterator iT=anObj.Delta().begin();
         iT!=anObj.Delta().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
}

cElXMLTree * ToXMLTree(const cListImByDelta & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ListImByDelta",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("KeySplitName"),anObj.KeySplitName())->ReTagThis("KeySplitName"));
  for
  (       std::list< int >::const_iterator it=anObj.Delta().begin();
      it !=anObj.Delta().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("Delta"),(*it))->ReTagThis("Delta"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cListImByDelta & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.KeySplitName(),aTree->Get("KeySplitName",1)); //tototo 

   xml_init(anObj.Delta(),aTree->GetAll("Delta",false,1));
}

std::string  Mangling( cListImByDelta *) {return "83CD3120743E4599FF3F";};


cTplValGesInit< std::string > & cMMUserEnvironment::TiePDetect()
{
   return mTiePDetect;
}

const cTplValGesInit< std::string > & cMMUserEnvironment::TiePDetect()const 
{
   return mTiePDetect;
}


cTplValGesInit< std::string > & cMMUserEnvironment::TiePMatch()
{
   return mTiePMatch;
}

const cTplValGesInit< std::string > & cMMUserEnvironment::TiePMatch()const 
{
   return mTiePMatch;
}


cTplValGesInit< std::string > & cMMUserEnvironment::UserName()
{
   return mUserName;
}

const cTplValGesInit< std::string > & cMMUserEnvironment::UserName()const 
{
   return mUserName;
}


cTplValGesInit< int > & cMMUserEnvironment::NbMaxProc()
{
   return mNbMaxProc;
}

const cTplValGesInit< int > & cMMUserEnvironment::NbMaxProc()const 
{
   return mNbMaxProc;
}


cTplValGesInit< bool > & cMMUserEnvironment::UseSeparateDirectories()
{
   return mUseSeparateDirectories;
}

const cTplValGesInit< bool > & cMMUserEnvironment::UseSeparateDirectories()const 
{
   return mUseSeparateDirectories;
}


cTplValGesInit< std::string > & cMMUserEnvironment::OutputDirectory()
{
   return mOutputDirectory;
}

const cTplValGesInit< std::string > & cMMUserEnvironment::OutputDirectory()const 
{
   return mOutputDirectory;
}


cTplValGesInit< std::string > & cMMUserEnvironment::LogDirectory()
{
   return mLogDirectory;
}

const cTplValGesInit< std::string > & cMMUserEnvironment::LogDirectory()const 
{
   return mLogDirectory;
}

void  BinaryUnDumpFromFile(cMMUserEnvironment & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.TiePDetect().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.TiePDetect().ValForcedForUnUmp(),aFp);
        }
        else  anObj.TiePDetect().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.TiePMatch().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.TiePMatch().ValForcedForUnUmp(),aFp);
        }
        else  anObj.TiePMatch().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.UserName().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.UserName().ValForcedForUnUmp(),aFp);
        }
        else  anObj.UserName().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NbMaxProc().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NbMaxProc().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NbMaxProc().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.UseSeparateDirectories().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.UseSeparateDirectories().ValForcedForUnUmp(),aFp);
        }
        else  anObj.UseSeparateDirectories().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.OutputDirectory().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.OutputDirectory().ValForcedForUnUmp(),aFp);
        }
        else  anObj.OutputDirectory().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.LogDirectory().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.LogDirectory().ValForcedForUnUmp(),aFp);
        }
        else  anObj.LogDirectory().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cMMUserEnvironment & anObj)
{
    BinaryDumpInFile(aFp,anObj.TiePDetect().IsInit());
    if (anObj.TiePDetect().IsInit()) BinaryDumpInFile(aFp,anObj.TiePDetect().Val());
    BinaryDumpInFile(aFp,anObj.TiePMatch().IsInit());
    if (anObj.TiePMatch().IsInit()) BinaryDumpInFile(aFp,anObj.TiePMatch().Val());
    BinaryDumpInFile(aFp,anObj.UserName().IsInit());
    if (anObj.UserName().IsInit()) BinaryDumpInFile(aFp,anObj.UserName().Val());
    BinaryDumpInFile(aFp,anObj.NbMaxProc().IsInit());
    if (anObj.NbMaxProc().IsInit()) BinaryDumpInFile(aFp,anObj.NbMaxProc().Val());
    BinaryDumpInFile(aFp,anObj.UseSeparateDirectories().IsInit());
    if (anObj.UseSeparateDirectories().IsInit()) BinaryDumpInFile(aFp,anObj.UseSeparateDirectories().Val());
    BinaryDumpInFile(aFp,anObj.OutputDirectory().IsInit());
    if (anObj.OutputDirectory().IsInit()) BinaryDumpInFile(aFp,anObj.OutputDirectory().Val());
    BinaryDumpInFile(aFp,anObj.LogDirectory().IsInit());
    if (anObj.LogDirectory().IsInit()) BinaryDumpInFile(aFp,anObj.LogDirectory().Val());
}

cElXMLTree * ToXMLTree(const cMMUserEnvironment & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"MMUserEnvironment",eXMLBranche);
   if (anObj.TiePDetect().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("TiePDetect"),anObj.TiePDetect().Val())->ReTagThis("TiePDetect"));
   if (anObj.TiePMatch().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("TiePMatch"),anObj.TiePMatch().Val())->ReTagThis("TiePMatch"));
   if (anObj.UserName().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("UserName"),anObj.UserName().Val())->ReTagThis("UserName"));
   if (anObj.NbMaxProc().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NbMaxProc"),anObj.NbMaxProc().Val())->ReTagThis("NbMaxProc"));
   if (anObj.UseSeparateDirectories().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("UseSeparateDirectories"),anObj.UseSeparateDirectories().Val())->ReTagThis("UseSeparateDirectories"));
   if (anObj.OutputDirectory().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("OutputDirectory"),anObj.OutputDirectory().Val())->ReTagThis("OutputDirectory"));
   if (anObj.LogDirectory().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("LogDirectory"),anObj.LogDirectory().Val())->ReTagThis("LogDirectory"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cMMUserEnvironment & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.TiePDetect(),aTree->Get("TiePDetect",1)); //tototo 

   xml_init(anObj.TiePMatch(),aTree->Get("TiePMatch",1)); //tototo 

   xml_init(anObj.UserName(),aTree->Get("UserName",1),std::string("Anonymous")); //tototo 

   xml_init(anObj.NbMaxProc(),aTree->Get("NbMaxProc",1),int(10000000)); //tototo 

   xml_init(anObj.UseSeparateDirectories(),aTree->Get("UseSeparateDirectories",1),bool(false)); //tototo 

   xml_init(anObj.OutputDirectory(),aTree->Get("OutputDirectory",1)); //tototo 

   xml_init(anObj.LogDirectory(),aTree->Get("LogDirectory",1)); //tototo 
}

std::string  Mangling( cMMUserEnvironment *) {return "121CD2EABC920CBEFDBF";};


double & cItem::Scale()
{
   return mScale;
}

const double & cItem::Scale()const 
{
   return mScale;
}


Pt3dr & cItem::Rotation()
{
   return mRotation;
}

const Pt3dr & cItem::Rotation()const 
{
   return mRotation;
}


Pt3dr & cItem::Translation()
{
   return mTranslation;
}

const Pt3dr & cItem::Translation()const 
{
   return mTranslation;
}


std::list< Pt2dr > & cItem::Pt()
{
   return mPt;
}

const std::list< Pt2dr > & cItem::Pt()const 
{
   return mPt;
}


int & cItem::Mode()
{
   return mMode;
}

const int & cItem::Mode()const 
{
   return mMode;
}

void  BinaryUnDumpFromFile(cItem & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Scale(),aFp);
    BinaryUnDumpFromFile(anObj.Rotation(),aFp);
    BinaryUnDumpFromFile(anObj.Translation(),aFp);
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             Pt2dr aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.Pt().push_back(aVal);
        }
  } ;
    BinaryUnDumpFromFile(anObj.Mode(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cItem & anObj)
{
    BinaryDumpInFile(aFp,anObj.Scale());
    BinaryDumpInFile(aFp,anObj.Rotation());
    BinaryDumpInFile(aFp,anObj.Translation());
    BinaryDumpInFile(aFp,(int)anObj.Pt().size());
    for(  std::list< Pt2dr >::const_iterator iT=anObj.Pt().begin();
         iT!=anObj.Pt().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.Mode());
}

cElXMLTree * ToXMLTree(const cItem & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"Item",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Scale"),anObj.Scale())->ReTagThis("Scale"));
   aRes->AddFils(ToXMLTree(std::string("Rotation"),anObj.Rotation())->ReTagThis("Rotation"));
   aRes->AddFils(ToXMLTree(std::string("Translation"),anObj.Translation())->ReTagThis("Translation"));
  for
  (       std::list< Pt2dr >::const_iterator it=anObj.Pt().begin();
      it !=anObj.Pt().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("Pt"),(*it))->ReTagThis("Pt"));
   aRes->AddFils(::ToXMLTree(std::string("Mode"),anObj.Mode())->ReTagThis("Mode"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cItem & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Scale(),aTree->Get("Scale",1)); //tototo 

   xml_init(anObj.Rotation(),aTree->Get("Rotation",1)); //tototo 

   xml_init(anObj.Translation(),aTree->Get("Translation",1)); //tototo 

   xml_init(anObj.Pt(),aTree->GetAll("Pt",false,1));

   xml_init(anObj.Mode(),aTree->Get("Mode",1)); //tototo 
}

std::string  Mangling( cItem *) {return "20AD72079FB2A1D4FCBF";};


std::list< cItem > & cSelectionInfos::Item()
{
   return mItem;
}

const std::list< cItem > & cSelectionInfos::Item()const 
{
   return mItem;
}

void  BinaryUnDumpFromFile(cSelectionInfos & anObj,ELISE_fp & aFp)
{
   { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cItem aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.Item().push_back(aVal);
        }
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cSelectionInfos & anObj)
{
    BinaryDumpInFile(aFp,(int)anObj.Item().size());
    for(  std::list< cItem >::const_iterator iT=anObj.Item().begin();
         iT!=anObj.Item().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
}

cElXMLTree * ToXMLTree(const cSelectionInfos & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"SelectionInfos",eXMLBranche);
  for
  (       std::list< cItem >::const_iterator it=anObj.Item().begin();
      it !=anObj.Item().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("Item"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cSelectionInfos & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Item(),aTree->GetAll("Item",false,1));
}

std::string  Mangling( cSelectionInfos *) {return "4E437E69780B5C96FE3F";};


Pt2di & cMTDCoher::Dec2()
{
   return mDec2;
}

const Pt2di & cMTDCoher::Dec2()const 
{
   return mDec2;
}

void  BinaryUnDumpFromFile(cMTDCoher & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Dec2(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cMTDCoher & anObj)
{
    BinaryDumpInFile(aFp,anObj.Dec2());
}

cElXMLTree * ToXMLTree(const cMTDCoher & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"MTDCoher",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Dec2"),anObj.Dec2())->ReTagThis("Dec2"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cMTDCoher & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Dec2(),aTree->Get("Dec2",1)); //tototo 
}

std::string  Mangling( cMTDCoher *) {return "FB80FE2B97F96881FE3F";};

// };
