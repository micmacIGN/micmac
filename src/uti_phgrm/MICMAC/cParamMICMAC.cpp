#include "StdAfx.h"
// UNUSED
// Quelque chose
eModeGeomMEC  Str2eModeGeomMEC(const std::string & aName)
{
   if (aName=="eGeomMECIm1")
      return eGeomMECIm1;
   else if (aName=="eGeomMECTerrain")
      return eGeomMECTerrain;
   else if (aName=="eNoGeomMEC")
      return eNoGeomMEC;
  else
  {
      cout << aName << " is not a correct value for enum eModeGeomMEC\n" ;
      ELISE_ASSERT(false,"XML enum value error");
  }
  return (eModeGeomMEC) 0;
}
void xml_init(eModeGeomMEC & aVal,cElXMLTree * aTree)
{
   aVal= Str2eModeGeomMEC(aTree->Contenu());
}
std::string  eToString(const eModeGeomMEC & anObj)
{
   if (anObj==eGeomMECIm1)
      return  "eGeomMECIm1";
   if (anObj==eGeomMECTerrain)
      return  "eGeomMECTerrain";
   if (anObj==eNoGeomMEC)
      return  "eNoGeomMEC";
 std::cout << "Enum = eModeGeomMEC\n";
   ELISE_ASSERT(false,"Bad Value in eToString for enum value ");
   return "";
}

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eModeGeomMEC & anObj)
{
      return  cElXMLTree::ValueNode(aNameTag,eToString(anObj));
}

void  BinaryDumpInFile(ELISE_fp & aFp,const eModeGeomMEC & anObj)
{
   BinaryDumpInFile(aFp,int(anObj));
}

void  BinaryUnDumpFromFile(eModeGeomMEC & anObj,ELISE_fp & aFp)
{
   int aIVal;
   BinaryUnDumpFromFile(aIVal,aFp);
   anObj=(eModeGeomMEC) aIVal;
}

std::string  Mangling( eModeGeomMEC *) {return "851B850F578D3AAAFE3F";};

eModeCensusCost  Str2eModeCensusCost(const std::string & aName)
{
   if (aName=="eMCC_GrCensus")
      return eMCC_GrCensus;
   else if (aName=="eMCC_CensusBasic")
      return eMCC_CensusBasic;
   else if (aName=="eMCC_CensusCorrel")
      return eMCC_CensusCorrel;
   else if (aName=="eMCC_CensusQuantitatif")
      return eMCC_CensusQuantitatif;
   else if (aName=="eMCC_CensusMixCorrelBasic")
      return eMCC_CensusMixCorrelBasic;
  else
  {
      cout << aName << " is not a correct value for enum eModeCensusCost\n" ;
      ELISE_ASSERT(false,"XML enum value error");
  }
  return (eModeCensusCost) 0;
}
void xml_init(eModeCensusCost & aVal,cElXMLTree * aTree)
{
   aVal= Str2eModeCensusCost(aTree->Contenu());
}
std::string  eToString(const eModeCensusCost & anObj)
{
   if (anObj==eMCC_GrCensus)
      return  "eMCC_GrCensus";
   if (anObj==eMCC_CensusBasic)
      return  "eMCC_CensusBasic";
   if (anObj==eMCC_CensusCorrel)
      return  "eMCC_CensusCorrel";
   if (anObj==eMCC_CensusQuantitatif)
      return  "eMCC_CensusQuantitatif";
   if (anObj==eMCC_CensusMixCorrelBasic)
      return  "eMCC_CensusMixCorrelBasic";
 std::cout << "Enum = eModeCensusCost\n";
   ELISE_ASSERT(false,"Bad Value in eToString for enum value ");
   return "";
}

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eModeCensusCost & anObj)
{
      return  cElXMLTree::ValueNode(aNameTag,eToString(anObj));
}

void  BinaryDumpInFile(ELISE_fp & aFp,const eModeCensusCost & anObj)
{
   BinaryDumpInFile(aFp,int(anObj));
}

void  BinaryUnDumpFromFile(eModeCensusCost & anObj,ELISE_fp & aFp)
{
   int aIVal;
   BinaryUnDumpFromFile(aIVal,aFp);
   anObj=(eModeCensusCost) aIVal;
}

std::string  Mangling( eModeCensusCost *) {return "20F055988F7786B1FE3F";};

eTypeModeleAnalytique  Str2eTypeModeleAnalytique(const std::string & aName)
{
   if (aName=="eTMA_Homologues")
      return eTMA_Homologues;
   else if (aName=="eTMA_DHomD")
      return eTMA_DHomD;
   else if (aName=="eTMA_Ori")
      return eTMA_Ori;
   else if (aName=="eTMA_Nuage3D")
      return eTMA_Nuage3D;
  else
  {
      cout << aName << " is not a correct value for enum eTypeModeleAnalytique\n" ;
      ELISE_ASSERT(false,"XML enum value error");
  }
  return (eTypeModeleAnalytique) 0;
}
void xml_init(eTypeModeleAnalytique & aVal,cElXMLTree * aTree)
{
   aVal= Str2eTypeModeleAnalytique(aTree->Contenu());
}
std::string  eToString(const eTypeModeleAnalytique & anObj)
{
   if (anObj==eTMA_Homologues)
      return  "eTMA_Homologues";
   if (anObj==eTMA_DHomD)
      return  "eTMA_DHomD";
   if (anObj==eTMA_Ori)
      return  "eTMA_Ori";
   if (anObj==eTMA_Nuage3D)
      return  "eTMA_Nuage3D";
 std::cout << "Enum = eTypeModeleAnalytique\n";
   ELISE_ASSERT(false,"Bad Value in eToString for enum value ");
   return "";
}

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eTypeModeleAnalytique & anObj)
{
      return  cElXMLTree::ValueNode(aNameTag,eToString(anObj));
}

void  BinaryDumpInFile(ELISE_fp & aFp,const eTypeModeleAnalytique & anObj)
{
   BinaryDumpInFile(aFp,int(anObj));
}

void  BinaryUnDumpFromFile(eTypeModeleAnalytique & anObj,ELISE_fp & aFp)
{
   int aIVal;
   BinaryUnDumpFromFile(aIVal,aFp);
   anObj=(eTypeModeleAnalytique) aIVal;
}

std::string  Mangling( eTypeModeleAnalytique *) {return "BD73BD505216D6B2FE3F";};

eModeGeomImage  Str2eModeGeomImage(const std::string & aName)
{
   if (aName=="eGeomImageOri")
      return eGeomImageOri;
   else if (aName=="eGeomImageModule")
      return eGeomImageModule;
   else if (aName=="eGeomImageGrille")
      return eGeomImageGrille;
   else if (aName=="eGeomImageRTO")
      return eGeomImageRTO;
   else if (aName=="eGeomImageCON")
      return eGeomImageCON;
   else if (aName=="eGeomImageDHD_Px")
      return eGeomImageDHD_Px;
   else if (aName=="eGeomImage_Hom_Px")
      return eGeomImage_Hom_Px;
   else if (aName=="eGeomImageDH_Px_HD")
      return eGeomImageDH_Px_HD;
   else if (aName=="eGeomImage_Epip")
      return eGeomImage_Epip;
   else if (aName=="eGeomImage_EpipolairePure")
      return eGeomImage_EpipolairePure;
   else if (aName=="eGeomGen")
      return eGeomGen;
   else if (aName=="eNoGeomIm")
      return eNoGeomIm;
  else
  {
      cout << aName << " is not a correct value for enum eModeGeomImage\n" ;
      ELISE_ASSERT(false,"XML enum value error");
  }
  return (eModeGeomImage) 0;
}
void xml_init(eModeGeomImage & aVal,cElXMLTree * aTree)
{
   aVal= Str2eModeGeomImage(aTree->Contenu());
}
std::string  eToString(const eModeGeomImage & anObj)
{
   if (anObj==eGeomImageOri)
      return  "eGeomImageOri";
   if (anObj==eGeomImageModule)
      return  "eGeomImageModule";
   if (anObj==eGeomImageGrille)
      return  "eGeomImageGrille";
   if (anObj==eGeomImageRTO)
      return  "eGeomImageRTO";
   if (anObj==eGeomImageCON)
      return  "eGeomImageCON";
   if (anObj==eGeomImageDHD_Px)
      return  "eGeomImageDHD_Px";
   if (anObj==eGeomImage_Hom_Px)
      return  "eGeomImage_Hom_Px";
   if (anObj==eGeomImageDH_Px_HD)
      return  "eGeomImageDH_Px_HD";
   if (anObj==eGeomImage_Epip)
      return  "eGeomImage_Epip";
   if (anObj==eGeomImage_EpipolairePure)
      return  "eGeomImage_EpipolairePure";
   if (anObj==eGeomGen)
      return  "eGeomGen";
   if (anObj==eNoGeomIm)
      return  "eNoGeomIm";
 std::cout << "Enum = eModeGeomImage\n";
   ELISE_ASSERT(false,"Bad Value in eToString for enum value ");
   return "";
}

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eModeGeomImage & anObj)
{
      return  cElXMLTree::ValueNode(aNameTag,eToString(anObj));
}

void  BinaryDumpInFile(ELISE_fp & aFp,const eModeGeomImage & anObj)
{
   BinaryDumpInFile(aFp,int(anObj));
}

void  BinaryUnDumpFromFile(eModeGeomImage & anObj,ELISE_fp & aFp)
{
   int aIVal;
   BinaryUnDumpFromFile(aIVal,aFp);
   anObj=(eModeGeomImage) aIVal;
}

std::string  Mangling( eModeGeomImage *) {return "6E8475AD465FD68FFE3F";};

eOnEmptyImSecApero  Str2eOnEmptyImSecApero(const std::string & aName)
{
   if (aName=="eOEISA_error")
      return eOEISA_error;
   else if (aName=="eOEISA_exit")
      return eOEISA_exit;
   else if (aName=="eOEISA_goon")
      return eOEISA_goon;
  else
  {
      cout << aName << " is not a correct value for enum eOnEmptyImSecApero\n" ;
      ELISE_ASSERT(false,"XML enum value error");
  }
  return (eOnEmptyImSecApero) 0;
}
void xml_init(eOnEmptyImSecApero & aVal,cElXMLTree * aTree)
{
   aVal= Str2eOnEmptyImSecApero(aTree->Contenu());
}
std::string  eToString(const eOnEmptyImSecApero & anObj)
{
   if (anObj==eOEISA_error)
      return  "eOEISA_error";
   if (anObj==eOEISA_exit)
      return  "eOEISA_exit";
   if (anObj==eOEISA_goon)
      return  "eOEISA_goon";
 std::cout << "Enum = eOnEmptyImSecApero\n";
   ELISE_ASSERT(false,"Bad Value in eToString for enum value ");
   return "";
}

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eOnEmptyImSecApero & anObj)
{
      return  cElXMLTree::ValueNode(aNameTag,eToString(anObj));
}

void  BinaryDumpInFile(ELISE_fp & aFp,const eOnEmptyImSecApero & anObj)
{
   BinaryDumpInFile(aFp,int(anObj));
}

void  BinaryUnDumpFromFile(eOnEmptyImSecApero & anObj,ELISE_fp & aFp)
{
   int aIVal;
   BinaryUnDumpFromFile(aIVal,aFp);
   anObj=(eOnEmptyImSecApero) aIVal;
}

std::string  Mangling( eOnEmptyImSecApero *) {return "7058D2410C4E51B4FE3F";};

eModeAggregCorr  Str2eModeAggregCorr(const std::string & aName)
{
   if (aName=="eAggregSymetrique")
      return eAggregSymetrique;
   else if (aName=="eAggregIm1Maitre")
      return eAggregIm1Maitre;
   else if (aName=="eAggregInfoMut")
      return eAggregInfoMut;
   else if (aName=="eAggregMaxIm1Maitre")
      return eAggregMaxIm1Maitre;
   else if (aName=="eAggregMinIm1Maitre")
      return eAggregMinIm1Maitre;
   else if (aName=="eAggregMoyMedIm1Maitre")
      return eAggregMoyMedIm1Maitre;
  else
  {
      cout << aName << " is not a correct value for enum eModeAggregCorr\n" ;
      ELISE_ASSERT(false,"XML enum value error");
  }
  return (eModeAggregCorr) 0;
}
void xml_init(eModeAggregCorr & aVal,cElXMLTree * aTree)
{
   aVal= Str2eModeAggregCorr(aTree->Contenu());
}
std::string  eToString(const eModeAggregCorr & anObj)
{
   if (anObj==eAggregSymetrique)
      return  "eAggregSymetrique";
   if (anObj==eAggregIm1Maitre)
      return  "eAggregIm1Maitre";
   if (anObj==eAggregInfoMut)
      return  "eAggregInfoMut";
   if (anObj==eAggregMaxIm1Maitre)
      return  "eAggregMaxIm1Maitre";
   if (anObj==eAggregMinIm1Maitre)
      return  "eAggregMinIm1Maitre";
   if (anObj==eAggregMoyMedIm1Maitre)
      return  "eAggregMoyMedIm1Maitre";
 std::cout << "Enum = eModeAggregCorr\n";
   ELISE_ASSERT(false,"Bad Value in eToString for enum value ");
   return "";
}

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eModeAggregCorr & anObj)
{
      return  cElXMLTree::ValueNode(aNameTag,eToString(anObj));
}

void  BinaryDumpInFile(ELISE_fp & aFp,const eModeAggregCorr & anObj)
{
   BinaryDumpInFile(aFp,int(anObj));
}

void  BinaryUnDumpFromFile(eModeAggregCorr & anObj,ELISE_fp & aFp)
{
   int aIVal;
   BinaryUnDumpFromFile(aIVal,aFp);
   anObj=(eModeAggregCorr) aIVal;
}

std::string  Mangling( eModeAggregCorr *) {return "DEE9CBF8DE9F33F9FE3F";};

eModeDynamiqueCorrel  Str2eModeDynamiqueCorrel(const std::string & aName)
{
   if (aName=="eCoeffCorrelStd")
      return eCoeffCorrelStd;
   else if (aName=="eCoeffAngle")
      return eCoeffAngle;
   else if (aName=="eCoeffGamma")
      return eCoeffGamma;
  else
  {
      cout << aName << " is not a correct value for enum eModeDynamiqueCorrel\n" ;
      ELISE_ASSERT(false,"XML enum value error");
  }
  return (eModeDynamiqueCorrel) 0;
}
void xml_init(eModeDynamiqueCorrel & aVal,cElXMLTree * aTree)
{
   aVal= Str2eModeDynamiqueCorrel(aTree->Contenu());
}
std::string  eToString(const eModeDynamiqueCorrel & anObj)
{
   if (anObj==eCoeffCorrelStd)
      return  "eCoeffCorrelStd";
   if (anObj==eCoeffAngle)
      return  "eCoeffAngle";
   if (anObj==eCoeffGamma)
      return  "eCoeffGamma";
 std::cout << "Enum = eModeDynamiqueCorrel\n";
   ELISE_ASSERT(false,"Bad Value in eToString for enum value ");
   return "";
}

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eModeDynamiqueCorrel & anObj)
{
      return  cElXMLTree::ValueNode(aNameTag,eToString(anObj));
}

void  BinaryDumpInFile(ELISE_fp & aFp,const eModeDynamiqueCorrel & anObj)
{
   BinaryDumpInFile(aFp,int(anObj));
}

void  BinaryUnDumpFromFile(eModeDynamiqueCorrel & anObj,ELISE_fp & aFp)
{
   int aIVal;
   BinaryUnDumpFromFile(aIVal,aFp);
   anObj=(eModeDynamiqueCorrel) aIVal;
}

std::string  Mangling( eModeDynamiqueCorrel *) {return "F7893E137D608EC9FE3F";};

eTypeImPyram  Str2eTypeImPyram(const std::string & aName)
{
   if (aName=="eUInt8Bits")
      return eUInt8Bits;
   else if (aName=="eUInt16Bits")
      return eUInt16Bits;
   else if (aName=="eFloat32Bits")
      return eFloat32Bits;
  else
  {
      cout << aName << " is not a correct value for enum eTypeImPyram\n" ;
      ELISE_ASSERT(false,"XML enum value error");
  }
  return (eTypeImPyram) 0;
}
void xml_init(eTypeImPyram & aVal,cElXMLTree * aTree)
{
   aVal= Str2eTypeImPyram(aTree->Contenu());
}
std::string  eToString(const eTypeImPyram & anObj)
{
   if (anObj==eUInt8Bits)
      return  "eUInt8Bits";
   if (anObj==eUInt16Bits)
      return  "eUInt16Bits";
   if (anObj==eFloat32Bits)
      return  "eFloat32Bits";
 std::cout << "Enum = eTypeImPyram\n";
   ELISE_ASSERT(false,"Bad Value in eToString for enum value ");
   return "";
}

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eTypeImPyram & anObj)
{
      return  cElXMLTree::ValueNode(aNameTag,eToString(anObj));
}

void  BinaryDumpInFile(ELISE_fp & aFp,const eTypeImPyram & anObj)
{
   BinaryDumpInFile(aFp,int(anObj));
}

void  BinaryUnDumpFromFile(eTypeImPyram & anObj,ELISE_fp & aFp)
{
   int aIVal;
   BinaryUnDumpFromFile(aIVal,aFp);
   anObj=(eTypeImPyram) aIVal;
}

std::string  Mangling( eTypeImPyram *) {return "0005C5948F2EC4A5FD3F";};

eAlgoRegul  Str2eAlgoRegul(const std::string & aName)
{
   if (aName=="eAlgoCoxRoy")
      return eAlgoCoxRoy;
   else if (aName=="eAlgo2PrgDyn")
      return eAlgo2PrgDyn;
   else if (aName=="eAlgoMaxOfScore")
      return eAlgoMaxOfScore;
   else if (aName=="eAlgoCoxRoySiPossible")
      return eAlgoCoxRoySiPossible;
   else if (aName=="eAlgoOptimDifferentielle")
      return eAlgoOptimDifferentielle;
   else if (aName=="eAlgoDequant")
      return eAlgoDequant;
   else if (aName=="eAlgoLeastSQ")
      return eAlgoLeastSQ;
   else if (aName=="eAlgoTestGPU")
      return eAlgoTestGPU;
   else if (aName=="eAlgoIdentite")
      return eAlgoIdentite;
  else
  {
      cout << aName << " is not a correct value for enum eAlgoRegul\n" ;
      ELISE_ASSERT(false,"XML enum value error");
  }
  return (eAlgoRegul) 0;
}
void xml_init(eAlgoRegul & aVal,cElXMLTree * aTree)
{
   aVal= Str2eAlgoRegul(aTree->Contenu());
}
std::string  eToString(const eAlgoRegul & anObj)
{
   if (anObj==eAlgoCoxRoy)
      return  "eAlgoCoxRoy";
   if (anObj==eAlgo2PrgDyn)
      return  "eAlgo2PrgDyn";
   if (anObj==eAlgoMaxOfScore)
      return  "eAlgoMaxOfScore";
   if (anObj==eAlgoCoxRoySiPossible)
      return  "eAlgoCoxRoySiPossible";
   if (anObj==eAlgoOptimDifferentielle)
      return  "eAlgoOptimDifferentielle";
   if (anObj==eAlgoDequant)
      return  "eAlgoDequant";
   if (anObj==eAlgoLeastSQ)
      return  "eAlgoLeastSQ";
   if (anObj==eAlgoTestGPU)
      return  "eAlgoTestGPU";
   if (anObj==eAlgoIdentite)
      return  "eAlgoIdentite";
 std::cout << "Enum = eAlgoRegul\n";
   ELISE_ASSERT(false,"Bad Value in eToString for enum value ");
   return "";
}

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eAlgoRegul & anObj)
{
      return  cElXMLTree::ValueNode(aNameTag,eToString(anObj));
}

void  BinaryDumpInFile(ELISE_fp & aFp,const eAlgoRegul & anObj)
{
   BinaryDumpInFile(aFp,int(anObj));
}

void  BinaryUnDumpFromFile(eAlgoRegul & anObj,ELISE_fp & aFp)
{
   int aIVal;
   BinaryUnDumpFromFile(aIVal,aFp);
   anObj=(eAlgoRegul) aIVal;
}

std::string  Mangling( eAlgoRegul *) {return "4A6093808B3B6EC6FD3F";};

eModeInterpolation  Str2eModeInterpolation(const std::string & aName)
{
   if (aName=="eInterpolPPV")
      return eInterpolPPV;
   else if (aName=="eInterpolBiLin")
      return eInterpolBiLin;
   else if (aName=="eInterpolBiCub")
      return eInterpolBiCub;
   else if (aName=="eInterpolSinCard")
      return eInterpolSinCard;
   else if (aName=="eOldInterpolSinCard")
      return eOldInterpolSinCard;
   else if (aName=="eInterpolMPD")
      return eInterpolMPD;
   else if (aName=="eInterpolBicubOpt")
      return eInterpolBicubOpt;
  else
  {
      cout << aName << " is not a correct value for enum eModeInterpolation\n" ;
      ELISE_ASSERT(false,"XML enum value error");
  }
  return (eModeInterpolation) 0;
}
void xml_init(eModeInterpolation & aVal,cElXMLTree * aTree)
{
   aVal= Str2eModeInterpolation(aTree->Contenu());
}
std::string  eToString(const eModeInterpolation & anObj)
{
   if (anObj==eInterpolPPV)
      return  "eInterpolPPV";
   if (anObj==eInterpolBiLin)
      return  "eInterpolBiLin";
   if (anObj==eInterpolBiCub)
      return  "eInterpolBiCub";
   if (anObj==eInterpolSinCard)
      return  "eInterpolSinCard";
   if (anObj==eOldInterpolSinCard)
      return  "eOldInterpolSinCard";
   if (anObj==eInterpolMPD)
      return  "eInterpolMPD";
   if (anObj==eInterpolBicubOpt)
      return  "eInterpolBicubOpt";
 std::cout << "Enum = eModeInterpolation\n";
   ELISE_ASSERT(false,"Bad Value in eToString for enum value ");
   return "";
}

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eModeInterpolation & anObj)
{
      return  cElXMLTree::ValueNode(aNameTag,eToString(anObj));
}

void  BinaryDumpInFile(ELISE_fp & aFp,const eModeInterpolation & anObj)
{
   BinaryDumpInFile(aFp,int(anObj));
}

void  BinaryUnDumpFromFile(eModeInterpolation & anObj,ELISE_fp & aFp)
{
   int aIVal;
   BinaryUnDumpFromFile(aIVal,aFp);
   anObj=(eModeInterpolation) aIVal;
}

std::string  Mangling( eModeInterpolation *) {return "462974AD24A5DCA0FD3F";};

eTypeFiltrage  Str2eTypeFiltrage(const std::string & aName)
{
   if (aName=="eFiltrageMedian")
      return eFiltrageMedian;
   else if (aName=="eFiltrageMoyenne")
      return eFiltrageMoyenne;
   else if (aName=="eFiltrageDeriche")
      return eFiltrageDeriche;
   else if (aName=="eFiltrageGamma")
      return eFiltrageGamma;
   else if (aName=="eFiltrageEqLoc")
      return eFiltrageEqLoc;
  else
  {
      cout << aName << " is not a correct value for enum eTypeFiltrage\n" ;
      ELISE_ASSERT(false,"XML enum value error");
  }
  return (eTypeFiltrage) 0;
}
void xml_init(eTypeFiltrage & aVal,cElXMLTree * aTree)
{
   aVal= Str2eTypeFiltrage(aTree->Contenu());
}
std::string  eToString(const eTypeFiltrage & anObj)
{
   if (anObj==eFiltrageMedian)
      return  "eFiltrageMedian";
   if (anObj==eFiltrageMoyenne)
      return  "eFiltrageMoyenne";
   if (anObj==eFiltrageDeriche)
      return  "eFiltrageDeriche";
   if (anObj==eFiltrageGamma)
      return  "eFiltrageGamma";
   if (anObj==eFiltrageEqLoc)
      return  "eFiltrageEqLoc";
 std::cout << "Enum = eTypeFiltrage\n";
   ELISE_ASSERT(false,"Bad Value in eToString for enum value ");
   return "";
}

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eTypeFiltrage & anObj)
{
      return  cElXMLTree::ValueNode(aNameTag,eToString(anObj));
}

void  BinaryDumpInFile(ELISE_fp & aFp,const eTypeFiltrage & anObj)
{
   BinaryDumpInFile(aFp,int(anObj));
}

void  BinaryUnDumpFromFile(eTypeFiltrage & anObj,ELISE_fp & aFp)
{
   int aIVal;
   BinaryUnDumpFromFile(aIVal,aFp);
   anObj=(eTypeFiltrage) aIVal;
}

std::string  Mangling( eTypeFiltrage *) {return "DD77A8DC56F2AEC1FE3F";};

ePxApply  Str2ePxApply(const std::string & aName)
{
   if (aName=="eApplyPx1")
      return eApplyPx1;
   else if (aName=="eApplyPx2")
      return eApplyPx2;
   else if (aName=="eApplyPx12")
      return eApplyPx12;
  else
  {
      cout << aName << " is not a correct value for enum ePxApply\n" ;
      ELISE_ASSERT(false,"XML enum value error");
  }
  return (ePxApply) 0;
}
void xml_init(ePxApply & aVal,cElXMLTree * aTree)
{
   aVal= Str2ePxApply(aTree->Contenu());
}
std::string  eToString(const ePxApply & anObj)
{
   if (anObj==eApplyPx1)
      return  "eApplyPx1";
   if (anObj==eApplyPx2)
      return  "eApplyPx2";
   if (anObj==eApplyPx12)
      return  "eApplyPx12";
 std::cout << "Enum = ePxApply\n";
   ELISE_ASSERT(false,"Bad Value in eToString for enum value ");
   return "";
}

cElXMLTree * ToXMLTree(const std::string & aNameTag,const ePxApply & anObj)
{
      return  cElXMLTree::ValueNode(aNameTag,eToString(anObj));
}

void  BinaryDumpInFile(ELISE_fp & aFp,const ePxApply & anObj)
{
   BinaryDumpInFile(aFp,int(anObj));
}

void  BinaryUnDumpFromFile(ePxApply & anObj,ELISE_fp & aFp)
{
   int aIVal;
   BinaryUnDumpFromFile(aIVal,aFp);
   anObj=(ePxApply) aIVal;
}

std::string  Mangling( ePxApply *) {return "0817925BAC5D0ABEFC3F";};

eModeAggregProgDyn  Str2eModeAggregProgDyn(const std::string & aName)
{
   if (aName=="ePrgDAgrSomme")
      return ePrgDAgrSomme;
   else if (aName=="ePrgDAgrMax")
      return ePrgDAgrMax;
   else if (aName=="ePrgDAgrReinject")
      return ePrgDAgrReinject;
   else if (aName=="ePrgDAgrProgressif")
      return ePrgDAgrProgressif;
  else
  {
      cout << aName << " is not a correct value for enum eModeAggregProgDyn\n" ;
      ELISE_ASSERT(false,"XML enum value error");
  }
  return (eModeAggregProgDyn) 0;
}
void xml_init(eModeAggregProgDyn & aVal,cElXMLTree * aTree)
{
   aVal= Str2eModeAggregProgDyn(aTree->Contenu());
}
std::string  eToString(const eModeAggregProgDyn & anObj)
{
   if (anObj==ePrgDAgrSomme)
      return  "ePrgDAgrSomme";
   if (anObj==ePrgDAgrMax)
      return  "ePrgDAgrMax";
   if (anObj==ePrgDAgrReinject)
      return  "ePrgDAgrReinject";
   if (anObj==ePrgDAgrProgressif)
      return  "ePrgDAgrProgressif";
 std::cout << "Enum = eModeAggregProgDyn\n";
   ELISE_ASSERT(false,"Bad Value in eToString for enum value ");
   return "";
}

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eModeAggregProgDyn & anObj)
{
      return  cElXMLTree::ValueNode(aNameTag,eToString(anObj));
}

void  BinaryDumpInFile(ELISE_fp & aFp,const eModeAggregProgDyn & anObj)
{
   BinaryDumpInFile(aFp,int(anObj));
}

void  BinaryUnDumpFromFile(eModeAggregProgDyn & anObj,ELISE_fp & aFp)
{
   int aIVal;
   BinaryUnDumpFromFile(aIVal,aFp);
   anObj=(eModeAggregProgDyn) aIVal;
}

std::string  Mangling( eModeAggregProgDyn *) {return "E743EC1CAD47A0B3FE3F";};

eMicMacCodeRetourErreur  Str2eMicMacCodeRetourErreur(const std::string & aName)
{
   if (aName=="eErrNbPointInEqOriRel")
      return eErrNbPointInEqOriRel;
   else if (aName=="eErrImageFileEmpty")
      return eErrImageFileEmpty;
   else if (aName=="eErrPtHomHorsImage")
      return eErrPtHomHorsImage;
   else if (aName=="eErrRecouvrInsuffisant")
      return eErrRecouvrInsuffisant;
   else if (aName=="eErrGrilleInverseNonDisponible")
      return eErrGrilleInverseNonDisponible;
  else
  {
      cout << aName << " is not a correct value for enum eMicMacCodeRetourErreur\n" ;
      ELISE_ASSERT(false,"XML enum value error");
  }
  return (eMicMacCodeRetourErreur) 0;
}
void xml_init(eMicMacCodeRetourErreur & aVal,cElXMLTree * aTree)
{
   aVal= Str2eMicMacCodeRetourErreur(aTree->Contenu());
}
std::string  eToString(const eMicMacCodeRetourErreur & anObj)
{
   if (anObj==eErrNbPointInEqOriRel)
      return  "eErrNbPointInEqOriRel";
   if (anObj==eErrImageFileEmpty)
      return  "eErrImageFileEmpty";
   if (anObj==eErrPtHomHorsImage)
      return  "eErrPtHomHorsImage";
   if (anObj==eErrRecouvrInsuffisant)
      return  "eErrRecouvrInsuffisant";
   if (anObj==eErrGrilleInverseNonDisponible)
      return  "eErrGrilleInverseNonDisponible";
 std::cout << "Enum = eMicMacCodeRetourErreur\n";
   ELISE_ASSERT(false,"Bad Value in eToString for enum value ");
   return "";
}

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eMicMacCodeRetourErreur & anObj)
{
      return  cElXMLTree::ValueNode(aNameTag,eToString(anObj));
}

void  BinaryDumpInFile(ELISE_fp & aFp,const eMicMacCodeRetourErreur & anObj)
{
   BinaryDumpInFile(aFp,int(anObj));
}

void  BinaryUnDumpFromFile(eMicMacCodeRetourErreur & anObj,ELISE_fp & aFp)
{
   int aIVal;
   BinaryUnDumpFromFile(aIVal,aFp);
   anObj=(eMicMacCodeRetourErreur) aIVal;
}

std::string  Mangling( eMicMacCodeRetourErreur *) {return "286647A05FCD56B0FDBF";};

eTypeWinCorrel  Str2eTypeWinCorrel(const std::string & aName)
{
   if (aName=="eWInCorrelFixe")
      return eWInCorrelFixe;
   else if (aName=="eWInCorrelExp")
      return eWInCorrelExp;
   else if (aName=="eWInCorrelRectSpec")
      return eWInCorrelRectSpec;
  else
  {
      cout << aName << " is not a correct value for enum eTypeWinCorrel\n" ;
      ELISE_ASSERT(false,"XML enum value error");
  }
  return (eTypeWinCorrel) 0;
}
void xml_init(eTypeWinCorrel & aVal,cElXMLTree * aTree)
{
   aVal= Str2eTypeWinCorrel(aTree->Contenu());
}
std::string  eToString(const eTypeWinCorrel & anObj)
{
   if (anObj==eWInCorrelFixe)
      return  "eWInCorrelFixe";
   if (anObj==eWInCorrelExp)
      return  "eWInCorrelExp";
   if (anObj==eWInCorrelRectSpec)
      return  "eWInCorrelRectSpec";
 std::cout << "Enum = eTypeWinCorrel\n";
   ELISE_ASSERT(false,"Bad Value in eToString for enum value ");
   return "";
}

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eTypeWinCorrel & anObj)
{
      return  cElXMLTree::ValueNode(aNameTag,eToString(anObj));
}

void  BinaryDumpInFile(ELISE_fp & aFp,const eTypeWinCorrel & anObj)
{
   BinaryDumpInFile(aFp,int(anObj));
}

void  BinaryUnDumpFromFile(eTypeWinCorrel & anObj,ELISE_fp & aFp)
{
   int aIVal;
   BinaryUnDumpFromFile(aIVal,aFp);
   anObj=(eTypeWinCorrel) aIVal;
}

std::string  Mangling( eTypeWinCorrel *) {return "5417F09DE39EB8A3FCBF";};

eTypeModeEchantPtsI  Str2eTypeModeEchantPtsI(const std::string & aName)
{
   if (aName=="eModeEchantRegulier")
      return eModeEchantRegulier;
   else if (aName=="eModeEchantNonAutoCor")
      return eModeEchantNonAutoCor;
   else if (aName=="eModeEchantAleatoire")
      return eModeEchantAleatoire;
   else if (aName=="eModeEchantPtsIntByComandeExterne")
      return eModeEchantPtsIntByComandeExterne;
  else
  {
      cout << aName << " is not a correct value for enum eTypeModeEchantPtsI\n" ;
      ELISE_ASSERT(false,"XML enum value error");
  }
  return (eTypeModeEchantPtsI) 0;
}
void xml_init(eTypeModeEchantPtsI & aVal,cElXMLTree * aTree)
{
   aVal= Str2eTypeModeEchantPtsI(aTree->Contenu());
}
std::string  eToString(const eTypeModeEchantPtsI & anObj)
{
   if (anObj==eModeEchantRegulier)
      return  "eModeEchantRegulier";
   if (anObj==eModeEchantNonAutoCor)
      return  "eModeEchantNonAutoCor";
   if (anObj==eModeEchantAleatoire)
      return  "eModeEchantAleatoire";
   if (anObj==eModeEchantPtsIntByComandeExterne)
      return  "eModeEchantPtsIntByComandeExterne";
 std::cout << "Enum = eTypeModeEchantPtsI\n";
   ELISE_ASSERT(false,"Bad Value in eToString for enum value ");
   return "";
}

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eTypeModeEchantPtsI & anObj)
{
      return  cElXMLTree::ValueNode(aNameTag,eToString(anObj));
}

void  BinaryDumpInFile(ELISE_fp & aFp,const eTypeModeEchantPtsI & anObj)
{
   BinaryDumpInFile(aFp,int(anObj));
}

void  BinaryUnDumpFromFile(eTypeModeEchantPtsI & anObj,ELISE_fp & aFp)
{
   int aIVal;
   BinaryUnDumpFromFile(aIVal,aFp);
   anObj=(eTypeModeEchantPtsI) aIVal;
}

std::string  Mangling( eTypeModeEchantPtsI *) {return "E83276A996C383B6FC3F";};

eSemantiqueLL  Str2eSemantiqueLL(const std::string & aName)
{
   if (aName=="eSLL_Geom_X")
      return eSLL_Geom_X;
   else if (aName=="eSLL_Geom_Y")
      return eSLL_Geom_Y;
   else if (aName=="eSLL_Geom_Z")
      return eSLL_Geom_Z;
   else if (aName=="eSLL_Geom_dir_X")
      return eSLL_Geom_dir_X;
   else if (aName=="eSLL_Geom_dir_Y")
      return eSLL_Geom_dir_Y;
   else if (aName=="eSLL_Geom_dir_Z")
      return eSLL_Geom_dir_Z;
   else if (aName=="eSLL_Radiom_R")
      return eSLL_Radiom_R;
   else if (aName=="eSLL_Radiom_G")
      return eSLL_Radiom_G;
   else if (aName=="eSLL_Radiom_B")
      return eSLL_Radiom_B;
   else if (aName=="eSLL_Radiom_Panchro")
      return eSLL_Radiom_Panchro;
   else if (aName=="eSLL_Radiom_Pir")
      return eSLL_Radiom_Pir;
   else if (aName=="eSLL_Radiom_Lidar")
      return eSLL_Radiom_Lidar;
   else if (aName=="eSLL_Radiom_Unknown")
      return eSLL_Radiom_Unknown;
   else if (aName=="eSLL_Unknown")
      return eSLL_Unknown;
  else
  {
      cout << aName << " is not a correct value for enum eSemantiqueLL\n" ;
      ELISE_ASSERT(false,"XML enum value error");
  }
  return (eSemantiqueLL) 0;
}
void xml_init(eSemantiqueLL & aVal,cElXMLTree * aTree)
{
   aVal= Str2eSemantiqueLL(aTree->Contenu());
}
std::string  eToString(const eSemantiqueLL & anObj)
{
   if (anObj==eSLL_Geom_X)
      return  "eSLL_Geom_X";
   if (anObj==eSLL_Geom_Y)
      return  "eSLL_Geom_Y";
   if (anObj==eSLL_Geom_Z)
      return  "eSLL_Geom_Z";
   if (anObj==eSLL_Geom_dir_X)
      return  "eSLL_Geom_dir_X";
   if (anObj==eSLL_Geom_dir_Y)
      return  "eSLL_Geom_dir_Y";
   if (anObj==eSLL_Geom_dir_Z)
      return  "eSLL_Geom_dir_Z";
   if (anObj==eSLL_Radiom_R)
      return  "eSLL_Radiom_R";
   if (anObj==eSLL_Radiom_G)
      return  "eSLL_Radiom_G";
   if (anObj==eSLL_Radiom_B)
      return  "eSLL_Radiom_B";
   if (anObj==eSLL_Radiom_Panchro)
      return  "eSLL_Radiom_Panchro";
   if (anObj==eSLL_Radiom_Pir)
      return  "eSLL_Radiom_Pir";
   if (anObj==eSLL_Radiom_Lidar)
      return  "eSLL_Radiom_Lidar";
   if (anObj==eSLL_Radiom_Unknown)
      return  "eSLL_Radiom_Unknown";
   if (anObj==eSLL_Unknown)
      return  "eSLL_Unknown";
 std::cout << "Enum = eSemantiqueLL\n";
   ELISE_ASSERT(false,"Bad Value in eToString for enum value ");
   return "";
}

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eSemantiqueLL & anObj)
{
      return  cElXMLTree::ValueNode(aNameTag,eToString(anObj));
}

void  BinaryDumpInFile(ELISE_fp & aFp,const eSemantiqueLL & anObj)
{
   BinaryDumpInFile(aFp,int(anObj));
}

void  BinaryUnDumpFromFile(eSemantiqueLL & anObj,ELISE_fp & aFp)
{
   int aIVal;
   BinaryUnDumpFromFile(aIVal,aFp);
   anObj=(eSemantiqueLL) aIVal;
}

std::string  Mangling( eSemantiqueLL *) {return "EA4B02855BB4CBACFF3F";};


eTypeFiltrage & cSpecFitrageImage::TypeFiltrage()
{
   return mTypeFiltrage;
}

const eTypeFiltrage & cSpecFitrageImage::TypeFiltrage()const 
{
   return mTypeFiltrage;
}


double & cSpecFitrageImage::SzFiltrage()
{
   return mSzFiltrage;
}

const double & cSpecFitrageImage::SzFiltrage()const 
{
   return mSzFiltrage;
}


cTplValGesInit< double > & cSpecFitrageImage::SzFiltrNonAd()
{
   return mSzFiltrNonAd;
}

const cTplValGesInit< double > & cSpecFitrageImage::SzFiltrNonAd()const 
{
   return mSzFiltrNonAd;
}


cTplValGesInit< ePxApply > & cSpecFitrageImage::PxApply()
{
   return mPxApply;
}

const cTplValGesInit< ePxApply > & cSpecFitrageImage::PxApply()const 
{
   return mPxApply;
}


cTplValGesInit< cElRegex_Ptr > & cSpecFitrageImage::PatternSelFiltre()
{
   return mPatternSelFiltre;
}

const cTplValGesInit< cElRegex_Ptr > & cSpecFitrageImage::PatternSelFiltre()const 
{
   return mPatternSelFiltre;
}


cTplValGesInit< int > & cSpecFitrageImage::NbIteration()
{
   return mNbIteration;
}

const cTplValGesInit< int > & cSpecFitrageImage::NbIteration()const 
{
   return mNbIteration;
}


cTplValGesInit< int > & cSpecFitrageImage::NbItereIntern()
{
   return mNbItereIntern;
}

const cTplValGesInit< int > & cSpecFitrageImage::NbItereIntern()const 
{
   return mNbItereIntern;
}


cTplValGesInit< double > & cSpecFitrageImage::AmplitudeSignal()
{
   return mAmplitudeSignal;
}

const cTplValGesInit< double > & cSpecFitrageImage::AmplitudeSignal()const 
{
   return mAmplitudeSignal;
}


cTplValGesInit< bool > & cSpecFitrageImage::UseIt()
{
   return mUseIt;
}

const cTplValGesInit< bool > & cSpecFitrageImage::UseIt()const 
{
   return mUseIt;
}

void  BinaryUnDumpFromFile(cSpecFitrageImage & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.TypeFiltrage(),aFp);
    BinaryUnDumpFromFile(anObj.SzFiltrage(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SzFiltrNonAd().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SzFiltrNonAd().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SzFiltrNonAd().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.PxApply().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.PxApply().ValForcedForUnUmp(),aFp);
        }
        else  anObj.PxApply().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.PatternSelFiltre().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.PatternSelFiltre().ValForcedForUnUmp(),aFp);
        }
        else  anObj.PatternSelFiltre().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NbIteration().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NbIteration().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NbIteration().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NbItereIntern().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NbItereIntern().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NbItereIntern().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.AmplitudeSignal().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.AmplitudeSignal().ValForcedForUnUmp(),aFp);
        }
        else  anObj.AmplitudeSignal().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.UseIt().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.UseIt().ValForcedForUnUmp(),aFp);
        }
        else  anObj.UseIt().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cSpecFitrageImage & anObj)
{
    BinaryDumpInFile(aFp,anObj.TypeFiltrage());
    BinaryDumpInFile(aFp,anObj.SzFiltrage());
    BinaryDumpInFile(aFp,anObj.SzFiltrNonAd().IsInit());
    if (anObj.SzFiltrNonAd().IsInit()) BinaryDumpInFile(aFp,anObj.SzFiltrNonAd().Val());
    BinaryDumpInFile(aFp,anObj.PxApply().IsInit());
    if (anObj.PxApply().IsInit()) BinaryDumpInFile(aFp,anObj.PxApply().Val());
    BinaryDumpInFile(aFp,anObj.PatternSelFiltre().IsInit());
    if (anObj.PatternSelFiltre().IsInit()) BinaryDumpInFile(aFp,anObj.PatternSelFiltre().Val());
    BinaryDumpInFile(aFp,anObj.NbIteration().IsInit());
    if (anObj.NbIteration().IsInit()) BinaryDumpInFile(aFp,anObj.NbIteration().Val());
    BinaryDumpInFile(aFp,anObj.NbItereIntern().IsInit());
    if (anObj.NbItereIntern().IsInit()) BinaryDumpInFile(aFp,anObj.NbItereIntern().Val());
    BinaryDumpInFile(aFp,anObj.AmplitudeSignal().IsInit());
    if (anObj.AmplitudeSignal().IsInit()) BinaryDumpInFile(aFp,anObj.AmplitudeSignal().Val());
    BinaryDumpInFile(aFp,anObj.UseIt().IsInit());
    if (anObj.UseIt().IsInit()) BinaryDumpInFile(aFp,anObj.UseIt().Val());
}

cElXMLTree * ToXMLTree(const cSpecFitrageImage & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"SpecFitrageImage",eXMLBranche);
   aRes->AddFils(ToXMLTree(std::string("TypeFiltrage"),anObj.TypeFiltrage())->ReTagThis("TypeFiltrage"));
   aRes->AddFils(::ToXMLTree(std::string("SzFiltrage"),anObj.SzFiltrage())->ReTagThis("SzFiltrage"));
   if (anObj.SzFiltrNonAd().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SzFiltrNonAd"),anObj.SzFiltrNonAd().Val())->ReTagThis("SzFiltrNonAd"));
   if (anObj.PxApply().IsInit())
      aRes->AddFils(ToXMLTree(std::string("PxApply"),anObj.PxApply().Val())->ReTagThis("PxApply"));
   if (anObj.PatternSelFiltre().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PatternSelFiltre"),anObj.PatternSelFiltre().Val())->ReTagThis("PatternSelFiltre"));
   if (anObj.NbIteration().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NbIteration"),anObj.NbIteration().Val())->ReTagThis("NbIteration"));
   if (anObj.NbItereIntern().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NbItereIntern"),anObj.NbItereIntern().Val())->ReTagThis("NbItereIntern"));
   if (anObj.AmplitudeSignal().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("AmplitudeSignal"),anObj.AmplitudeSignal().Val())->ReTagThis("AmplitudeSignal"));
   if (anObj.UseIt().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("UseIt"),anObj.UseIt().Val())->ReTagThis("UseIt"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cSpecFitrageImage & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.TypeFiltrage(),aTree->Get("TypeFiltrage",1)); //tototo 

   xml_init(anObj.SzFiltrage(),aTree->Get("SzFiltrage",1)); //tototo 

   xml_init(anObj.SzFiltrNonAd(),aTree->Get("SzFiltrNonAd",1),double(0.0)); //tototo 

   xml_init(anObj.PxApply(),aTree->Get("PxApply",1),ePxApply(eApplyPx12)); //tototo 

   xml_init(anObj.PatternSelFiltre(),aTree->Get("PatternSelFiltre",1)); //tototo 

   xml_init(anObj.NbIteration(),aTree->Get("NbIteration",1),int(1)); //tototo 

   xml_init(anObj.NbItereIntern(),aTree->Get("NbItereIntern",1),int(1)); //tototo 

   xml_init(anObj.AmplitudeSignal(),aTree->Get("AmplitudeSignal",1),double(255)); //tototo 

   xml_init(anObj.UseIt(),aTree->Get("UseIt",1),bool(true)); //tototo 
}

std::string  Mangling( cSpecFitrageImage *) {return "305A77304BD912DDFE3F";};


double & cXML_RatioCorrImage::Ratio()
{
   return mRatio;
}

const double & cXML_RatioCorrImage::Ratio()const 
{
   return mRatio;
}


cTplValGesInit< int > & cXML_RatioCorrImage::NbPt()
{
   return mNbPt;
}

const cTplValGesInit< int > & cXML_RatioCorrImage::NbPt()const 
{
   return mNbPt;
}

void  BinaryUnDumpFromFile(cXML_RatioCorrImage & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Ratio(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NbPt().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NbPt().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NbPt().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cXML_RatioCorrImage & anObj)
{
    BinaryDumpInFile(aFp,anObj.Ratio());
    BinaryDumpInFile(aFp,anObj.NbPt().IsInit());
    if (anObj.NbPt().IsInit()) BinaryDumpInFile(aFp,anObj.NbPt().Val());
}

cElXMLTree * ToXMLTree(const cXML_RatioCorrImage & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"XML_RatioCorrImage",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Ratio"),anObj.Ratio())->ReTagThis("Ratio"));
   if (anObj.NbPt().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NbPt"),anObj.NbPt().Val())->ReTagThis("NbPt"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cXML_RatioCorrImage & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.Ratio(),aTree->Get("Ratio",1)); //tototo 

   xml_init(anObj.NbPt(),aTree->Get("NbPt",1)); //tototo 
}

std::string  Mangling( cXML_RatioCorrImage *) {return "6754566BFE6187BFFF3F";};


Pt2dr & cCorrectionPxTransverse::DirPx()
{
   return mDirPx;
}

const Pt2dr & cCorrectionPxTransverse::DirPx()const 
{
   return mDirPx;
}


Im2D_REAL4 & cCorrectionPxTransverse::ValeurPx()
{
   return mValeurPx;
}

const Im2D_REAL4 & cCorrectionPxTransverse::ValeurPx()const 
{
   return mValeurPx;
}


double & cCorrectionPxTransverse::SsResol()
{
   return mSsResol;
}

const double & cCorrectionPxTransverse::SsResol()const 
{
   return mSsResol;
}

void  BinaryUnDumpFromFile(cCorrectionPxTransverse & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.DirPx(),aFp);
    BinaryUnDumpFromFile(anObj.ValeurPx(),aFp);
    BinaryUnDumpFromFile(anObj.SsResol(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cCorrectionPxTransverse & anObj)
{
    BinaryDumpInFile(aFp,anObj.DirPx());
    BinaryDumpInFile(aFp,anObj.ValeurPx());
    BinaryDumpInFile(aFp,anObj.SsResol());
}

cElXMLTree * ToXMLTree(const cCorrectionPxTransverse & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"CorrectionPxTransverse",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("DirPx"),anObj.DirPx())->ReTagThis("DirPx"));
   aRes->AddFils(::ToXMLTree(std::string("ValeurPx"),anObj.ValeurPx())->ReTagThis("ValeurPx"));
   aRes->AddFils(::ToXMLTree(std::string("SsResol"),anObj.SsResol())->ReTagThis("SsResol"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cCorrectionPxTransverse & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.DirPx(),aTree->Get("DirPx",1)); //tototo 

   xml_init(anObj.ValeurPx(),aTree->Get("ValeurPx",1)); //tototo 

   xml_init(anObj.SsResol(),aTree->Get("SsResol",1)); //tototo 
}

std::string  Mangling( cCorrectionPxTransverse *) {return "F4FEF1B89BF629B6FB3F";};


std::string & cLidarLayer::NameFile()
{
   return mNameFile;
}

const std::string & cLidarLayer::NameFile()const 
{
   return mNameFile;
}


eSemantiqueLL & cLidarLayer::Semantic()
{
   return mSemantic;
}

const eSemantiqueLL & cLidarLayer::Semantic()const 
{
   return mSemantic;
}


cTplValGesInit< double > & cLidarLayer::LongueurDOnde()
{
   return mLongueurDOnde;
}

const cTplValGesInit< double > & cLidarLayer::LongueurDOnde()const 
{
   return mLongueurDOnde;
}


cTplValGesInit< double > & cLidarLayer::OffsetValues()
{
   return mOffsetValues;
}

const cTplValGesInit< double > & cLidarLayer::OffsetValues()const 
{
   return mOffsetValues;
}


cTplValGesInit< double > & cLidarLayer::StepValues()
{
   return mStepValues;
}

const cTplValGesInit< double > & cLidarLayer::StepValues()const 
{
   return mStepValues;
}


bool & cLidarLayer::IntegerValues()
{
   return mIntegerValues;
}

const bool & cLidarLayer::IntegerValues()const 
{
   return mIntegerValues;
}


bool & cLidarLayer::SignedValues()
{
   return mSignedValues;
}

const bool & cLidarLayer::SignedValues()const 
{
   return mSignedValues;
}


int & cLidarLayer::BytePerValues()
{
   return mBytePerValues;
}

const int & cLidarLayer::BytePerValues()const 
{
   return mBytePerValues;
}


int & cLidarLayer::OffsetDataInFile()
{
   return mOffsetDataInFile;
}

const int & cLidarLayer::OffsetDataInFile()const 
{
   return mOffsetDataInFile;
}

void  BinaryUnDumpFromFile(cLidarLayer & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.NameFile(),aFp);
    BinaryUnDumpFromFile(anObj.Semantic(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.LongueurDOnde().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.LongueurDOnde().ValForcedForUnUmp(),aFp);
        }
        else  anObj.LongueurDOnde().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.OffsetValues().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.OffsetValues().ValForcedForUnUmp(),aFp);
        }
        else  anObj.OffsetValues().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.StepValues().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.StepValues().ValForcedForUnUmp(),aFp);
        }
        else  anObj.StepValues().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.IntegerValues(),aFp);
    BinaryUnDumpFromFile(anObj.SignedValues(),aFp);
    BinaryUnDumpFromFile(anObj.BytePerValues(),aFp);
    BinaryUnDumpFromFile(anObj.OffsetDataInFile(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cLidarLayer & anObj)
{
    BinaryDumpInFile(aFp,anObj.NameFile());
    BinaryDumpInFile(aFp,anObj.Semantic());
    BinaryDumpInFile(aFp,anObj.LongueurDOnde().IsInit());
    if (anObj.LongueurDOnde().IsInit()) BinaryDumpInFile(aFp,anObj.LongueurDOnde().Val());
    BinaryDumpInFile(aFp,anObj.OffsetValues().IsInit());
    if (anObj.OffsetValues().IsInit()) BinaryDumpInFile(aFp,anObj.OffsetValues().Val());
    BinaryDumpInFile(aFp,anObj.StepValues().IsInit());
    if (anObj.StepValues().IsInit()) BinaryDumpInFile(aFp,anObj.StepValues().Val());
    BinaryDumpInFile(aFp,anObj.IntegerValues());
    BinaryDumpInFile(aFp,anObj.SignedValues());
    BinaryDumpInFile(aFp,anObj.BytePerValues());
    BinaryDumpInFile(aFp,anObj.OffsetDataInFile());
}

cElXMLTree * ToXMLTree(const cLidarLayer & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"LidarLayer",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("NameFile"),anObj.NameFile())->ReTagThis("NameFile"));
   aRes->AddFils(ToXMLTree(std::string("Semantic"),anObj.Semantic())->ReTagThis("Semantic"));
   if (anObj.LongueurDOnde().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("LongueurDOnde"),anObj.LongueurDOnde().Val())->ReTagThis("LongueurDOnde"));
   if (anObj.OffsetValues().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("OffsetValues"),anObj.OffsetValues().Val())->ReTagThis("OffsetValues"));
   if (anObj.StepValues().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("StepValues"),anObj.StepValues().Val())->ReTagThis("StepValues"));
   aRes->AddFils(::ToXMLTree(std::string("IntegerValues"),anObj.IntegerValues())->ReTagThis("IntegerValues"));
   aRes->AddFils(::ToXMLTree(std::string("SignedValues"),anObj.SignedValues())->ReTagThis("SignedValues"));
   aRes->AddFils(::ToXMLTree(std::string("BytePerValues"),anObj.BytePerValues())->ReTagThis("BytePerValues"));
   aRes->AddFils(::ToXMLTree(std::string("OffsetDataInFile"),anObj.OffsetDataInFile())->ReTagThis("OffsetDataInFile"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cLidarLayer & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.NameFile(),aTree->Get("NameFile",1)); //tototo 

   xml_init(anObj.Semantic(),aTree->Get("Semantic",1)); //tototo 

   xml_init(anObj.LongueurDOnde(),aTree->Get("LongueurDOnde",1)); //tototo 

   xml_init(anObj.OffsetValues(),aTree->Get("OffsetValues",1),double(0.0)); //tototo 

   xml_init(anObj.StepValues(),aTree->Get("StepValues",1),double(0.0)); //tototo 

   xml_init(anObj.IntegerValues(),aTree->Get("IntegerValues",1)); //tototo 

   xml_init(anObj.SignedValues(),aTree->Get("SignedValues",1)); //tototo 

   xml_init(anObj.BytePerValues(),aTree->Get("BytePerValues",1)); //tototo 

   xml_init(anObj.OffsetDataInFile(),aTree->Get("OffsetDataInFile",1)); //tototo 
}

std::string  Mangling( cLidarLayer *) {return "1661D233E7D33BB9FDBF";};


Pt2dr & cGeometrieAffineApprochee::ImTerain_P00()
{
   return mImTerain_P00;
}

const Pt2dr & cGeometrieAffineApprochee::ImTerain_P00()const 
{
   return mImTerain_P00;
}


Pt2dr & cGeometrieAffineApprochee::DerImTerain_Di()
{
   return mDerImTerain_Di;
}

const Pt2dr & cGeometrieAffineApprochee::DerImTerain_Di()const 
{
   return mDerImTerain_Di;
}


Pt2dr & cGeometrieAffineApprochee::DerImTerain_Dj()
{
   return mDerImTerain_Dj;
}

const Pt2dr & cGeometrieAffineApprochee::DerImTerain_Dj()const 
{
   return mDerImTerain_Dj;
}

void  BinaryUnDumpFromFile(cGeometrieAffineApprochee & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.ImTerain_P00(),aFp);
    BinaryUnDumpFromFile(anObj.DerImTerain_Di(),aFp);
    BinaryUnDumpFromFile(anObj.DerImTerain_Dj(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cGeometrieAffineApprochee & anObj)
{
    BinaryDumpInFile(aFp,anObj.ImTerain_P00());
    BinaryDumpInFile(aFp,anObj.DerImTerain_Di());
    BinaryDumpInFile(aFp,anObj.DerImTerain_Dj());
}

cElXMLTree * ToXMLTree(const cGeometrieAffineApprochee & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"GeometrieAffineApprochee",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("ImTerain_P00"),anObj.ImTerain_P00())->ReTagThis("ImTerain_P00"));
   aRes->AddFils(::ToXMLTree(std::string("DerImTerain_Di"),anObj.DerImTerain_Di())->ReTagThis("DerImTerain_Di"));
   aRes->AddFils(::ToXMLTree(std::string("DerImTerain_Dj"),anObj.DerImTerain_Dj())->ReTagThis("DerImTerain_Dj"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cGeometrieAffineApprochee & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.ImTerain_P00(),aTree->Get("ImTerain_P00",1)); //tototo 

   xml_init(anObj.DerImTerain_Di(),aTree->Get("DerImTerain_Di",1)); //tototo 

   xml_init(anObj.DerImTerain_Dj(),aTree->Get("DerImTerain_Dj",1)); //tototo 
}

std::string  Mangling( cGeometrieAffineApprochee *) {return "4035A997FAA5C4D1FCBF";};


std::list< cLidarLayer > & cLidarStrip::LidarLayer()
{
   return mLidarLayer;
}

const std::list< cLidarLayer > & cLidarStrip::LidarLayer()const 
{
   return mLidarLayer;
}


bool & cLidarStrip::FileIs2DStructured()
{
   return mFileIs2DStructured;
}

const bool & cLidarStrip::FileIs2DStructured()const 
{
   return mFileIs2DStructured;
}


Pt2dr & cLidarStrip::ImTerain_P00()
{
   return GeometrieAffineApprochee().Val().ImTerain_P00();
}

const Pt2dr & cLidarStrip::ImTerain_P00()const 
{
   return GeometrieAffineApprochee().Val().ImTerain_P00();
}


Pt2dr & cLidarStrip::DerImTerain_Di()
{
   return GeometrieAffineApprochee().Val().DerImTerain_Di();
}

const Pt2dr & cLidarStrip::DerImTerain_Di()const 
{
   return GeometrieAffineApprochee().Val().DerImTerain_Di();
}


Pt2dr & cLidarStrip::DerImTerain_Dj()
{
   return GeometrieAffineApprochee().Val().DerImTerain_Dj();
}

const Pt2dr & cLidarStrip::DerImTerain_Dj()const 
{
   return GeometrieAffineApprochee().Val().DerImTerain_Dj();
}


cTplValGesInit< cGeometrieAffineApprochee > & cLidarStrip::GeometrieAffineApprochee()
{
   return mGeometrieAffineApprochee;
}

const cTplValGesInit< cGeometrieAffineApprochee > & cLidarStrip::GeometrieAffineApprochee()const 
{
   return mGeometrieAffineApprochee;
}


Box2dr & cLidarStrip::BoiteEnglob()
{
   return mBoiteEnglob;
}

const Box2dr & cLidarStrip::BoiteEnglob()const 
{
   return mBoiteEnglob;
}

void  BinaryUnDumpFromFile(cLidarStrip & anObj,ELISE_fp & aFp)
{
   { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cLidarLayer aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.LidarLayer().push_back(aVal);
        }
  } ;
    BinaryUnDumpFromFile(anObj.FileIs2DStructured(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.GeometrieAffineApprochee().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.GeometrieAffineApprochee().ValForcedForUnUmp(),aFp);
        }
        else  anObj.GeometrieAffineApprochee().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.BoiteEnglob(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cLidarStrip & anObj)
{
    BinaryDumpInFile(aFp,(int)anObj.LidarLayer().size());
    for(  std::list< cLidarLayer >::const_iterator iT=anObj.LidarLayer().begin();
         iT!=anObj.LidarLayer().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.FileIs2DStructured());
    BinaryDumpInFile(aFp,anObj.GeometrieAffineApprochee().IsInit());
    if (anObj.GeometrieAffineApprochee().IsInit()) BinaryDumpInFile(aFp,anObj.GeometrieAffineApprochee().Val());
    BinaryDumpInFile(aFp,anObj.BoiteEnglob());
}

cElXMLTree * ToXMLTree(const cLidarStrip & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"LidarStrip",eXMLBranche);
  for
  (       std::list< cLidarLayer >::const_iterator it=anObj.LidarLayer().begin();
      it !=anObj.LidarLayer().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("LidarLayer"));
   aRes->AddFils(::ToXMLTree(std::string("FileIs2DStructured"),anObj.FileIs2DStructured())->ReTagThis("FileIs2DStructured"));
   if (anObj.GeometrieAffineApprochee().IsInit())
      aRes->AddFils(ToXMLTree(anObj.GeometrieAffineApprochee().Val())->ReTagThis("GeometrieAffineApprochee"));
   aRes->AddFils(::ToXMLTree(std::string("BoiteEnglob"),anObj.BoiteEnglob())->ReTagThis("BoiteEnglob"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cLidarStrip & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.LidarLayer(),aTree->GetAll("LidarLayer",false,1));

   xml_init(anObj.FileIs2DStructured(),aTree->Get("FileIs2DStructured",1)); //tototo 

   xml_init(anObj.GeometrieAffineApprochee(),aTree->Get("GeometrieAffineApprochee",1)); //tototo 

   xml_init(anObj.BoiteEnglob(),aTree->Get("BoiteEnglob",1)); //tototo 
}

std::string  Mangling( cLidarStrip *) {return "586A5FB13B4DB1F2FD3F";};


std::string & cLidarFlight::SystemeCoordonnees()
{
   return mSystemeCoordonnees;
}

const std::string & cLidarFlight::SystemeCoordonnees()const 
{
   return mSystemeCoordonnees;
}


std::list< cLidarStrip > & cLidarFlight::LidarStrip()
{
   return mLidarStrip;
}

const std::list< cLidarStrip > & cLidarFlight::LidarStrip()const 
{
   return mLidarStrip;
}


Box2dr & cLidarFlight::BoiteEnglob()
{
   return mBoiteEnglob;
}

const Box2dr & cLidarFlight::BoiteEnglob()const 
{
   return mBoiteEnglob;
}

void  BinaryUnDumpFromFile(cLidarFlight & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.SystemeCoordonnees(),aFp);
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cLidarStrip aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.LidarStrip().push_back(aVal);
        }
  } ;
    BinaryUnDumpFromFile(anObj.BoiteEnglob(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cLidarFlight & anObj)
{
    BinaryDumpInFile(aFp,anObj.SystemeCoordonnees());
    BinaryDumpInFile(aFp,(int)anObj.LidarStrip().size());
    for(  std::list< cLidarStrip >::const_iterator iT=anObj.LidarStrip().begin();
         iT!=anObj.LidarStrip().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.BoiteEnglob());
}

cElXMLTree * ToXMLTree(const cLidarFlight & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"LidarFlight",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("SystemeCoordonnees"),anObj.SystemeCoordonnees())->ReTagThis("SystemeCoordonnees"));
  for
  (       std::list< cLidarStrip >::const_iterator it=anObj.LidarStrip().begin();
      it !=anObj.LidarStrip().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("LidarStrip"));
   aRes->AddFils(::ToXMLTree(std::string("BoiteEnglob"),anObj.BoiteEnglob())->ReTagThis("BoiteEnglob"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cLidarFlight & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.SystemeCoordonnees(),aTree->Get("SystemeCoordonnees",1)); //tototo 

   xml_init(anObj.LidarStrip(),aTree->GetAll("LidarStrip",false,1));

   xml_init(anObj.BoiteEnglob(),aTree->Get("BoiteEnglob",1)); //tototo 
}

std::string  Mangling( cLidarFlight *) {return "F0DE3F62C443B38DFF3F";};


cTplValGesInit< int > & cMemPartMICMAC::NbMaxImageOn1Point()
{
   return mNbMaxImageOn1Point;
}

const cTplValGesInit< int > & cMemPartMICMAC::NbMaxImageOn1Point()const 
{
   return mNbMaxImageOn1Point;
}


cTplValGesInit< double > & cMemPartMICMAC::BSurHGlob()
{
   return mBSurHGlob;
}

const cTplValGesInit< double > & cMemPartMICMAC::BSurHGlob()const 
{
   return mBSurHGlob;
}


cTplValGesInit< int > & cMemPartMICMAC::DeZoomLast()
{
   return mDeZoomLast;
}

const cTplValGesInit< int > & cMemPartMICMAC::DeZoomLast()const 
{
   return mDeZoomLast;
}


cTplValGesInit< int > & cMemPartMICMAC::NumLastEtape()
{
   return mNumLastEtape;
}

const cTplValGesInit< int > & cMemPartMICMAC::NumLastEtape()const 
{
   return mNumLastEtape;
}

void  BinaryUnDumpFromFile(cMemPartMICMAC & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NbMaxImageOn1Point().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NbMaxImageOn1Point().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NbMaxImageOn1Point().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.BSurHGlob().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.BSurHGlob().ValForcedForUnUmp(),aFp);
        }
        else  anObj.BSurHGlob().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.DeZoomLast().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.DeZoomLast().ValForcedForUnUmp(),aFp);
        }
        else  anObj.DeZoomLast().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NumLastEtape().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NumLastEtape().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NumLastEtape().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cMemPartMICMAC & anObj)
{
    BinaryDumpInFile(aFp,anObj.NbMaxImageOn1Point().IsInit());
    if (anObj.NbMaxImageOn1Point().IsInit()) BinaryDumpInFile(aFp,anObj.NbMaxImageOn1Point().Val());
    BinaryDumpInFile(aFp,anObj.BSurHGlob().IsInit());
    if (anObj.BSurHGlob().IsInit()) BinaryDumpInFile(aFp,anObj.BSurHGlob().Val());
    BinaryDumpInFile(aFp,anObj.DeZoomLast().IsInit());
    if (anObj.DeZoomLast().IsInit()) BinaryDumpInFile(aFp,anObj.DeZoomLast().Val());
    BinaryDumpInFile(aFp,anObj.NumLastEtape().IsInit());
    if (anObj.NumLastEtape().IsInit()) BinaryDumpInFile(aFp,anObj.NumLastEtape().Val());
}

cElXMLTree * ToXMLTree(const cMemPartMICMAC & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"MemPartMICMAC",eXMLBranche);
   if (anObj.NbMaxImageOn1Point().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NbMaxImageOn1Point"),anObj.NbMaxImageOn1Point().Val())->ReTagThis("NbMaxImageOn1Point"));
   if (anObj.BSurHGlob().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("BSurHGlob"),anObj.BSurHGlob().Val())->ReTagThis("BSurHGlob"));
   if (anObj.DeZoomLast().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DeZoomLast"),anObj.DeZoomLast().Val())->ReTagThis("DeZoomLast"));
   if (anObj.NumLastEtape().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NumLastEtape"),anObj.NumLastEtape().Val())->ReTagThis("NumLastEtape"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cMemPartMICMAC & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.NbMaxImageOn1Point(),aTree->Get("NbMaxImageOn1Point",1)); //tototo 

   xml_init(anObj.BSurHGlob(),aTree->Get("BSurHGlob",1)); //tototo 

   xml_init(anObj.DeZoomLast(),aTree->Get("DeZoomLast",1)); //tototo 

   xml_init(anObj.NumLastEtape(),aTree->Get("NumLastEtape",1)); //tototo 
}

std::string  Mangling( cMemPartMICMAC *) {return "01995C9700E5D1BAFE3F";};


Box2dr & cParamMasqAnam::BoxTer()
{
   return mBoxTer;
}

const Box2dr & cParamMasqAnam::BoxTer()const 
{
   return mBoxTer;
}


double & cParamMasqAnam::Resol()
{
   return mResol;
}

const double & cParamMasqAnam::Resol()const 
{
   return mResol;
}

void  BinaryUnDumpFromFile(cParamMasqAnam & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.BoxTer(),aFp);
    BinaryUnDumpFromFile(anObj.Resol(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cParamMasqAnam & anObj)
{
    BinaryDumpInFile(aFp,anObj.BoxTer());
    BinaryDumpInFile(aFp,anObj.Resol());
}

cElXMLTree * ToXMLTree(const cParamMasqAnam & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ParamMasqAnam",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("BoxTer"),anObj.BoxTer())->ReTagThis("BoxTer"));
   aRes->AddFils(::ToXMLTree(std::string("Resol"),anObj.Resol())->ReTagThis("Resol"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cParamMasqAnam & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.BoxTer(),aTree->Get("BoxTer",1)); //tototo 

   xml_init(anObj.Resol(),aTree->Get("Resol",1)); //tototo 
}

std::string  Mangling( cParamMasqAnam *) {return "10537F9863ABAEF1FE3F";};


bool & cMM_EtatAvancement::AllDone()
{
   return mAllDone;
}

const bool & cMM_EtatAvancement::AllDone()const 
{
   return mAllDone;
}

void  BinaryUnDumpFromFile(cMM_EtatAvancement & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.AllDone(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cMM_EtatAvancement & anObj)
{
    BinaryDumpInFile(aFp,anObj.AllDone());
}

cElXMLTree * ToXMLTree(const cMM_EtatAvancement & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"MM_EtatAvancement",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("AllDone"),anObj.AllDone())->ReTagThis("AllDone"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cMM_EtatAvancement & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.AllDone(),aTree->Get("AllDone",1)); //tototo 
}

std::string  Mangling( cMM_EtatAvancement *) {return "795E74F5599B19B5FF3F";};


std::string & cImageFDC::FDCIm()
{
   return mFDCIm;
}

const std::string & cImageFDC::FDCIm()const 
{
   return mFDCIm;
}


cTplValGesInit< Pt2dr > & cImageFDC::DirEpipTransv()
{
   return mDirEpipTransv;
}

const cTplValGesInit< Pt2dr > & cImageFDC::DirEpipTransv()const 
{
   return mDirEpipTransv;
}

void  BinaryUnDumpFromFile(cImageFDC & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.FDCIm(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.DirEpipTransv().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.DirEpipTransv().ValForcedForUnUmp(),aFp);
        }
        else  anObj.DirEpipTransv().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cImageFDC & anObj)
{
    BinaryDumpInFile(aFp,anObj.FDCIm());
    BinaryDumpInFile(aFp,anObj.DirEpipTransv().IsInit());
    if (anObj.DirEpipTransv().IsInit()) BinaryDumpInFile(aFp,anObj.DirEpipTransv().Val());
}

cElXMLTree * ToXMLTree(const cImageFDC & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ImageFDC",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("FDCIm"),anObj.FDCIm())->ReTagThis("FDCIm"));
   if (anObj.DirEpipTransv().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DirEpipTransv"),anObj.DirEpipTransv().Val())->ReTagThis("DirEpipTransv"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cImageFDC & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.FDCIm(),aTree->Get("FDCIm",1)); //tototo 

   xml_init(anObj.DirEpipTransv(),aTree->Get("DirEpipTransv",1)); //tototo 
}

std::string  Mangling( cImageFDC *) {return "CD145414AB80EFB3FF3F";};


std::string & cCouplesFDC::FDCIm1()
{
   return mFDCIm1;
}

const std::string & cCouplesFDC::FDCIm1()const 
{
   return mFDCIm1;
}


std::string & cCouplesFDC::FDCIm2()
{
   return mFDCIm2;
}

const std::string & cCouplesFDC::FDCIm2()const 
{
   return mFDCIm2;
}


cTplValGesInit< double > & cCouplesFDC::BSurH()
{
   return mBSurH;
}

const cTplValGesInit< double > & cCouplesFDC::BSurH()const 
{
   return mBSurH;
}

void  BinaryUnDumpFromFile(cCouplesFDC & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.FDCIm1(),aFp);
    BinaryUnDumpFromFile(anObj.FDCIm2(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.BSurH().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.BSurH().ValForcedForUnUmp(),aFp);
        }
        else  anObj.BSurH().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cCouplesFDC & anObj)
{
    BinaryDumpInFile(aFp,anObj.FDCIm1());
    BinaryDumpInFile(aFp,anObj.FDCIm2());
    BinaryDumpInFile(aFp,anObj.BSurH().IsInit());
    if (anObj.BSurH().IsInit()) BinaryDumpInFile(aFp,anObj.BSurH().Val());
}

cElXMLTree * ToXMLTree(const cCouplesFDC & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"CouplesFDC",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("FDCIm1"),anObj.FDCIm1())->ReTagThis("FDCIm1"));
   aRes->AddFils(::ToXMLTree(std::string("FDCIm2"),anObj.FDCIm2())->ReTagThis("FDCIm2"));
   if (anObj.BSurH().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("BSurH"),anObj.BSurH().Val())->ReTagThis("BSurH"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cCouplesFDC & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.FDCIm1(),aTree->Get("FDCIm1",1)); //tototo 

   xml_init(anObj.FDCIm2(),aTree->Get("FDCIm2",1)); //tototo 

   xml_init(anObj.BSurH(),aTree->Get("BSurH",1)); //tototo 
}

std::string  Mangling( cCouplesFDC *) {return "B886D988076CB893FB3F";};


std::list< cImageFDC > & cFileDescriptionChantier::ImageFDC()
{
   return mImageFDC;
}

const std::list< cImageFDC > & cFileDescriptionChantier::ImageFDC()const 
{
   return mImageFDC;
}


std::list< cCouplesFDC > & cFileDescriptionChantier::CouplesFDC()
{
   return mCouplesFDC;
}

const std::list< cCouplesFDC > & cFileDescriptionChantier::CouplesFDC()const 
{
   return mCouplesFDC;
}

void  BinaryUnDumpFromFile(cFileDescriptionChantier & anObj,ELISE_fp & aFp)
{
   { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cImageFDC aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.ImageFDC().push_back(aVal);
        }
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cCouplesFDC aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.CouplesFDC().push_back(aVal);
        }
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cFileDescriptionChantier & anObj)
{
    BinaryDumpInFile(aFp,(int)anObj.ImageFDC().size());
    for(  std::list< cImageFDC >::const_iterator iT=anObj.ImageFDC().begin();
         iT!=anObj.ImageFDC().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,(int)anObj.CouplesFDC().size());
    for(  std::list< cCouplesFDC >::const_iterator iT=anObj.CouplesFDC().begin();
         iT!=anObj.CouplesFDC().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
}

cElXMLTree * ToXMLTree(const cFileDescriptionChantier & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"FileDescriptionChantier",eXMLBranche);
  for
  (       std::list< cImageFDC >::const_iterator it=anObj.ImageFDC().begin();
      it !=anObj.ImageFDC().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("ImageFDC"));
  for
  (       std::list< cCouplesFDC >::const_iterator it=anObj.CouplesFDC().begin();
      it !=anObj.CouplesFDC().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("CouplesFDC"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cFileDescriptionChantier & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.ImageFDC(),aTree->GetAll("ImageFDC",false,1));

   xml_init(anObj.CouplesFDC(),aTree->GetAll("CouplesFDC",false,1));
}

std::string  Mangling( cFileDescriptionChantier *) {return "36644FE46324E4D8FD3F";};


Box2dr & cBoxMasqIsBoxTer::Box()
{
   return mBox;
}

const Box2dr & cBoxMasqIsBoxTer::Box()const 
{
   return mBox;
}

void  BinaryUnDumpFromFile(cBoxMasqIsBoxTer & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Box(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cBoxMasqIsBoxTer & anObj)
{
    BinaryDumpInFile(aFp,anObj.Box());
}

cElXMLTree * ToXMLTree(const cBoxMasqIsBoxTer & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"BoxMasqIsBoxTer",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Box"),anObj.Box())->ReTagThis("Box"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cBoxMasqIsBoxTer & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.Box(),aTree->Get("Box",1)); //tototo 
}

std::string  Mangling( cBoxMasqIsBoxTer *) {return "50081D7388B748CEF9BF";};


std::string & cMNT_Init::MNT_Init_Image()
{
   return mMNT_Init_Image;
}

const std::string & cMNT_Init::MNT_Init_Image()const 
{
   return mMNT_Init_Image;
}


std::string & cMNT_Init::MNT_Init_Xml()
{
   return mMNT_Init_Xml;
}

const std::string & cMNT_Init::MNT_Init_Xml()const 
{
   return mMNT_Init_Xml;
}


cTplValGesInit< double > & cMNT_Init::MNT_Offset()
{
   return mMNT_Offset;
}

const cTplValGesInit< double > & cMNT_Init::MNT_Offset()const 
{
   return mMNT_Offset;
}

void  BinaryUnDumpFromFile(cMNT_Init & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.MNT_Init_Image(),aFp);
    BinaryUnDumpFromFile(anObj.MNT_Init_Xml(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.MNT_Offset().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.MNT_Offset().ValForcedForUnUmp(),aFp);
        }
        else  anObj.MNT_Offset().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cMNT_Init & anObj)
{
    BinaryDumpInFile(aFp,anObj.MNT_Init_Image());
    BinaryDumpInFile(aFp,anObj.MNT_Init_Xml());
    BinaryDumpInFile(aFp,anObj.MNT_Offset().IsInit());
    if (anObj.MNT_Offset().IsInit()) BinaryDumpInFile(aFp,anObj.MNT_Offset().Val());
}

cElXMLTree * ToXMLTree(const cMNT_Init & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"MNT_Init",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("MNT_Init_Image"),anObj.MNT_Init_Image())->ReTagThis("MNT_Init_Image"));
   aRes->AddFils(::ToXMLTree(std::string("MNT_Init_Xml"),anObj.MNT_Init_Xml())->ReTagThis("MNT_Init_Xml"));
   if (anObj.MNT_Offset().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("MNT_Offset"),anObj.MNT_Offset().Val())->ReTagThis("MNT_Offset"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cMNT_Init & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.MNT_Init_Image(),aTree->Get("MNT_Init_Image",1)); //tototo 

   xml_init(anObj.MNT_Init_Xml(),aTree->Get("MNT_Init_Xml",1)); //tototo 

   xml_init(anObj.MNT_Offset(),aTree->Get("MNT_Offset",1),double(0.0)); //tototo 
}

std::string  Mangling( cMNT_Init *) {return "2A8151F5F46A2E98FD3F";};


std::string & cEnveloppeMNT_INIT::ZInf()
{
   return mZInf;
}

const std::string & cEnveloppeMNT_INIT::ZInf()const 
{
   return mZInf;
}


std::string & cEnveloppeMNT_INIT::ZSup()
{
   return mZSup;
}

const std::string & cEnveloppeMNT_INIT::ZSup()const 
{
   return mZSup;
}

void  BinaryUnDumpFromFile(cEnveloppeMNT_INIT & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.ZInf(),aFp);
    BinaryUnDumpFromFile(anObj.ZSup(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cEnveloppeMNT_INIT & anObj)
{
    BinaryDumpInFile(aFp,anObj.ZInf());
    BinaryDumpInFile(aFp,anObj.ZSup());
}

cElXMLTree * ToXMLTree(const cEnveloppeMNT_INIT & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"EnveloppeMNT_INIT",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("ZInf"),anObj.ZInf())->ReTagThis("ZInf"));
   aRes->AddFils(::ToXMLTree(std::string("ZSup"),anObj.ZSup())->ReTagThis("ZSup"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cEnveloppeMNT_INIT & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.ZInf(),aTree->Get("ZInf",1)); //tototo 

   xml_init(anObj.ZSup(),aTree->Get("ZSup",1)); //tototo 
}

std::string  Mangling( cEnveloppeMNT_INIT *) {return "EACA33104168F789FF3F";};


cTplValGesInit< double > & cIntervAltimetrie::ZMoyen()
{
   return mZMoyen;
}

const cTplValGesInit< double > & cIntervAltimetrie::ZMoyen()const 
{
   return mZMoyen;
}


double & cIntervAltimetrie::ZIncCalc()
{
   return mZIncCalc;
}

const double & cIntervAltimetrie::ZIncCalc()const 
{
   return mZIncCalc;
}


cTplValGesInit< bool > & cIntervAltimetrie::ZIncIsProp()
{
   return mZIncIsProp;
}

const cTplValGesInit< bool > & cIntervAltimetrie::ZIncIsProp()const 
{
   return mZIncIsProp;
}


cTplValGesInit< double > & cIntervAltimetrie::ZIncZonage()
{
   return mZIncZonage;
}

const cTplValGesInit< double > & cIntervAltimetrie::ZIncZonage()const 
{
   return mZIncZonage;
}


std::string & cIntervAltimetrie::MNT_Init_Image()
{
   return MNT_Init().Val().MNT_Init_Image();
}

const std::string & cIntervAltimetrie::MNT_Init_Image()const 
{
   return MNT_Init().Val().MNT_Init_Image();
}


std::string & cIntervAltimetrie::MNT_Init_Xml()
{
   return MNT_Init().Val().MNT_Init_Xml();
}

const std::string & cIntervAltimetrie::MNT_Init_Xml()const 
{
   return MNT_Init().Val().MNT_Init_Xml();
}


cTplValGesInit< double > & cIntervAltimetrie::MNT_Offset()
{
   return MNT_Init().Val().MNT_Offset();
}

const cTplValGesInit< double > & cIntervAltimetrie::MNT_Offset()const 
{
   return MNT_Init().Val().MNT_Offset();
}


cTplValGesInit< cMNT_Init > & cIntervAltimetrie::MNT_Init()
{
   return mMNT_Init;
}

const cTplValGesInit< cMNT_Init > & cIntervAltimetrie::MNT_Init()const 
{
   return mMNT_Init;
}


std::string & cIntervAltimetrie::ZInf()
{
   return EnveloppeMNT_INIT().Val().ZInf();
}

const std::string & cIntervAltimetrie::ZInf()const 
{
   return EnveloppeMNT_INIT().Val().ZInf();
}


std::string & cIntervAltimetrie::ZSup()
{
   return EnveloppeMNT_INIT().Val().ZSup();
}

const std::string & cIntervAltimetrie::ZSup()const 
{
   return EnveloppeMNT_INIT().Val().ZSup();
}


cTplValGesInit< cEnveloppeMNT_INIT > & cIntervAltimetrie::EnveloppeMNT_INIT()
{
   return mEnveloppeMNT_INIT;
}

const cTplValGesInit< cEnveloppeMNT_INIT > & cIntervAltimetrie::EnveloppeMNT_INIT()const 
{
   return mEnveloppeMNT_INIT;
}

void  BinaryUnDumpFromFile(cIntervAltimetrie & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ZMoyen().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ZMoyen().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ZMoyen().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.ZIncCalc(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ZIncIsProp().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ZIncIsProp().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ZIncIsProp().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ZIncZonage().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ZIncZonage().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ZIncZonage().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.MNT_Init().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.MNT_Init().ValForcedForUnUmp(),aFp);
        }
        else  anObj.MNT_Init().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.EnveloppeMNT_INIT().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.EnveloppeMNT_INIT().ValForcedForUnUmp(),aFp);
        }
        else  anObj.EnveloppeMNT_INIT().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cIntervAltimetrie & anObj)
{
    BinaryDumpInFile(aFp,anObj.ZMoyen().IsInit());
    if (anObj.ZMoyen().IsInit()) BinaryDumpInFile(aFp,anObj.ZMoyen().Val());
    BinaryDumpInFile(aFp,anObj.ZIncCalc());
    BinaryDumpInFile(aFp,anObj.ZIncIsProp().IsInit());
    if (anObj.ZIncIsProp().IsInit()) BinaryDumpInFile(aFp,anObj.ZIncIsProp().Val());
    BinaryDumpInFile(aFp,anObj.ZIncZonage().IsInit());
    if (anObj.ZIncZonage().IsInit()) BinaryDumpInFile(aFp,anObj.ZIncZonage().Val());
    BinaryDumpInFile(aFp,anObj.MNT_Init().IsInit());
    if (anObj.MNT_Init().IsInit()) BinaryDumpInFile(aFp,anObj.MNT_Init().Val());
    BinaryDumpInFile(aFp,anObj.EnveloppeMNT_INIT().IsInit());
    if (anObj.EnveloppeMNT_INIT().IsInit()) BinaryDumpInFile(aFp,anObj.EnveloppeMNT_INIT().Val());
}

cElXMLTree * ToXMLTree(const cIntervAltimetrie & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"IntervAltimetrie",eXMLBranche);
   if (anObj.ZMoyen().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ZMoyen"),anObj.ZMoyen().Val())->ReTagThis("ZMoyen"));
   aRes->AddFils(::ToXMLTree(std::string("ZIncCalc"),anObj.ZIncCalc())->ReTagThis("ZIncCalc"));
   if (anObj.ZIncIsProp().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ZIncIsProp"),anObj.ZIncIsProp().Val())->ReTagThis("ZIncIsProp"));
   if (anObj.ZIncZonage().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ZIncZonage"),anObj.ZIncZonage().Val())->ReTagThis("ZIncZonage"));
   if (anObj.MNT_Init().IsInit())
      aRes->AddFils(ToXMLTree(anObj.MNT_Init().Val())->ReTagThis("MNT_Init"));
   if (anObj.EnveloppeMNT_INIT().IsInit())
      aRes->AddFils(ToXMLTree(anObj.EnveloppeMNT_INIT().Val())->ReTagThis("EnveloppeMNT_INIT"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cIntervAltimetrie & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.ZMoyen(),aTree->Get("ZMoyen",1)); //tototo 

   xml_init(anObj.ZIncCalc(),aTree->Get("ZIncCalc",1)); //tototo 

   xml_init(anObj.ZIncIsProp(),aTree->Get("ZIncIsProp",1)); //tototo 

   xml_init(anObj.ZIncZonage(),aTree->Get("ZIncZonage",1)); //tototo 

   xml_init(anObj.MNT_Init(),aTree->Get("MNT_Init",1)); //tototo 

   xml_init(anObj.EnveloppeMNT_INIT(),aTree->Get("EnveloppeMNT_INIT",1)); //tototo 
}

std::string  Mangling( cIntervAltimetrie *) {return "3E0CF10B6C4ECFE3FC3F";};


cTplValGesInit< double > & cIntervParalaxe::Px1Moy()
{
   return mPx1Moy;
}

const cTplValGesInit< double > & cIntervParalaxe::Px1Moy()const 
{
   return mPx1Moy;
}


cTplValGesInit< double > & cIntervParalaxe::Px2Moy()
{
   return mPx2Moy;
}

const cTplValGesInit< double > & cIntervParalaxe::Px2Moy()const 
{
   return mPx2Moy;
}


double & cIntervParalaxe::Px1IncCalc()
{
   return mPx1IncCalc;
}

const double & cIntervParalaxe::Px1IncCalc()const 
{
   return mPx1IncCalc;
}


cTplValGesInit< double > & cIntervParalaxe::Px1PropProf()
{
   return mPx1PropProf;
}

const cTplValGesInit< double > & cIntervParalaxe::Px1PropProf()const 
{
   return mPx1PropProf;
}


cTplValGesInit< double > & cIntervParalaxe::Px2IncCalc()
{
   return mPx2IncCalc;
}

const cTplValGesInit< double > & cIntervParalaxe::Px2IncCalc()const 
{
   return mPx2IncCalc;
}


cTplValGesInit< double > & cIntervParalaxe::Px1IncZonage()
{
   return mPx1IncZonage;
}

const cTplValGesInit< double > & cIntervParalaxe::Px1IncZonage()const 
{
   return mPx1IncZonage;
}


cTplValGesInit< double > & cIntervParalaxe::Px2IncZonage()
{
   return mPx2IncZonage;
}

const cTplValGesInit< double > & cIntervParalaxe::Px2IncZonage()const 
{
   return mPx2IncZonage;
}

void  BinaryUnDumpFromFile(cIntervParalaxe & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Px1Moy().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Px1Moy().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Px1Moy().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Px2Moy().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Px2Moy().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Px2Moy().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.Px1IncCalc(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Px1PropProf().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Px1PropProf().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Px1PropProf().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Px2IncCalc().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Px2IncCalc().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Px2IncCalc().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Px1IncZonage().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Px1IncZonage().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Px1IncZonage().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Px2IncZonage().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Px2IncZonage().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Px2IncZonage().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cIntervParalaxe & anObj)
{
    BinaryDumpInFile(aFp,anObj.Px1Moy().IsInit());
    if (anObj.Px1Moy().IsInit()) BinaryDumpInFile(aFp,anObj.Px1Moy().Val());
    BinaryDumpInFile(aFp,anObj.Px2Moy().IsInit());
    if (anObj.Px2Moy().IsInit()) BinaryDumpInFile(aFp,anObj.Px2Moy().Val());
    BinaryDumpInFile(aFp,anObj.Px1IncCalc());
    BinaryDumpInFile(aFp,anObj.Px1PropProf().IsInit());
    if (anObj.Px1PropProf().IsInit()) BinaryDumpInFile(aFp,anObj.Px1PropProf().Val());
    BinaryDumpInFile(aFp,anObj.Px2IncCalc().IsInit());
    if (anObj.Px2IncCalc().IsInit()) BinaryDumpInFile(aFp,anObj.Px2IncCalc().Val());
    BinaryDumpInFile(aFp,anObj.Px1IncZonage().IsInit());
    if (anObj.Px1IncZonage().IsInit()) BinaryDumpInFile(aFp,anObj.Px1IncZonage().Val());
    BinaryDumpInFile(aFp,anObj.Px2IncZonage().IsInit());
    if (anObj.Px2IncZonage().IsInit()) BinaryDumpInFile(aFp,anObj.Px2IncZonage().Val());
}

cElXMLTree * ToXMLTree(const cIntervParalaxe & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"IntervParalaxe",eXMLBranche);
   if (anObj.Px1Moy().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Px1Moy"),anObj.Px1Moy().Val())->ReTagThis("Px1Moy"));
   if (anObj.Px2Moy().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Px2Moy"),anObj.Px2Moy().Val())->ReTagThis("Px2Moy"));
   aRes->AddFils(::ToXMLTree(std::string("Px1IncCalc"),anObj.Px1IncCalc())->ReTagThis("Px1IncCalc"));
   if (anObj.Px1PropProf().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Px1PropProf"),anObj.Px1PropProf().Val())->ReTagThis("Px1PropProf"));
   if (anObj.Px2IncCalc().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Px2IncCalc"),anObj.Px2IncCalc().Val())->ReTagThis("Px2IncCalc"));
   if (anObj.Px1IncZonage().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Px1IncZonage"),anObj.Px1IncZonage().Val())->ReTagThis("Px1IncZonage"));
   if (anObj.Px2IncZonage().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Px2IncZonage"),anObj.Px2IncZonage().Val())->ReTagThis("Px2IncZonage"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cIntervParalaxe & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.Px1Moy(),aTree->Get("Px1Moy",1)); //tototo 

   xml_init(anObj.Px2Moy(),aTree->Get("Px2Moy",1)); //tototo 

   xml_init(anObj.Px1IncCalc(),aTree->Get("Px1IncCalc",1)); //tototo 

   xml_init(anObj.Px1PropProf(),aTree->Get("Px1PropProf",1),double(0.0)); //tototo 

   xml_init(anObj.Px2IncCalc(),aTree->Get("Px2IncCalc",1)); //tototo 

   xml_init(anObj.Px1IncZonage(),aTree->Get("Px1IncZonage",1)); //tototo 

   xml_init(anObj.Px2IncZonage(),aTree->Get("Px2IncZonage",1)); //tototo 
}

std::string  Mangling( cIntervParalaxe *) {return "8408ACE1BEEA2BA9FD3F";};


std::string & cNuageXMLInit::NameNuageXML()
{
   return mNameNuageXML;
}

const std::string & cNuageXMLInit::NameNuageXML()const 
{
   return mNameNuageXML;
}


cTplValGesInit< bool > & cNuageXMLInit::CanAdaptGeom()
{
   return mCanAdaptGeom;
}

const cTplValGesInit< bool > & cNuageXMLInit::CanAdaptGeom()const 
{
   return mCanAdaptGeom;
}

void  BinaryUnDumpFromFile(cNuageXMLInit & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.NameNuageXML(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.CanAdaptGeom().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.CanAdaptGeom().ValForcedForUnUmp(),aFp);
        }
        else  anObj.CanAdaptGeom().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cNuageXMLInit & anObj)
{
    BinaryDumpInFile(aFp,anObj.NameNuageXML());
    BinaryDumpInFile(aFp,anObj.CanAdaptGeom().IsInit());
    if (anObj.CanAdaptGeom().IsInit()) BinaryDumpInFile(aFp,anObj.CanAdaptGeom().Val());
}

cElXMLTree * ToXMLTree(const cNuageXMLInit & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"NuageXMLInit",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("NameNuageXML"),anObj.NameNuageXML())->ReTagThis("NameNuageXML"));
   if (anObj.CanAdaptGeom().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("CanAdaptGeom"),anObj.CanAdaptGeom().Val())->ReTagThis("CanAdaptGeom"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cNuageXMLInit & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.NameNuageXML(),aTree->Get("NameNuageXML",1)); //tototo 

   xml_init(anObj.CanAdaptGeom(),aTree->Get("CanAdaptGeom",1),bool(false)); //tototo 
}

std::string  Mangling( cNuageXMLInit *) {return "F26291BCAB165EDEFE3F";};


double & cIntervSpecialZInv::MulZMin()
{
   return mMulZMin;
}

const double & cIntervSpecialZInv::MulZMin()const 
{
   return mMulZMin;
}


double & cIntervSpecialZInv::MulZMax()
{
   return mMulZMax;
}

const double & cIntervSpecialZInv::MulZMax()const 
{
   return mMulZMax;
}

void  BinaryUnDumpFromFile(cIntervSpecialZInv & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.MulZMin(),aFp);
    BinaryUnDumpFromFile(anObj.MulZMax(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cIntervSpecialZInv & anObj)
{
    BinaryDumpInFile(aFp,anObj.MulZMin());
    BinaryDumpInFile(aFp,anObj.MulZMax());
}

cElXMLTree * ToXMLTree(const cIntervSpecialZInv & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"IntervSpecialZInv",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("MulZMin"),anObj.MulZMin())->ReTagThis("MulZMin"));
   aRes->AddFils(::ToXMLTree(std::string("MulZMax"),anObj.MulZMax())->ReTagThis("MulZMax"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cIntervSpecialZInv & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.MulZMin(),aTree->Get("MulZMin",1)); //tototo 

   xml_init(anObj.MulZMax(),aTree->Get("MulZMax",1)); //tototo 
}

std::string  Mangling( cIntervSpecialZInv *) {return "F267DE435F0FB3F6FD3F";};


std::list< Pt2dr > & cListePointsInclus::Pt()
{
   return mPt;
}

const std::list< Pt2dr > & cListePointsInclus::Pt()const 
{
   return mPt;
}


std::string & cListePointsInclus::Im()
{
   return mIm;
}

const std::string & cListePointsInclus::Im()const 
{
   return mIm;
}

void  BinaryUnDumpFromFile(cListePointsInclus & anObj,ELISE_fp & aFp)
{
   { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             Pt2dr aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.Pt().push_back(aVal);
        }
  } ;
    BinaryUnDumpFromFile(anObj.Im(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cListePointsInclus & anObj)
{
    BinaryDumpInFile(aFp,(int)anObj.Pt().size());
    for(  std::list< Pt2dr >::const_iterator iT=anObj.Pt().begin();
         iT!=anObj.Pt().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.Im());
}

cElXMLTree * ToXMLTree(const cListePointsInclus & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ListePointsInclus",eXMLBranche);
  for
  (       std::list< Pt2dr >::const_iterator it=anObj.Pt().begin();
      it !=anObj.Pt().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("Pt"),(*it))->ReTagThis("Pt"));
   aRes->AddFils(::ToXMLTree(std::string("Im"),anObj.Im())->ReTagThis("Im"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cListePointsInclus & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.Pt(),aTree->GetAll("Pt",false,1));

   xml_init(anObj.Im(),aTree->Get("Im",1)); //tototo 
}

std::string  Mangling( cListePointsInclus *) {return "28BF4F8423DC5CF9FE3F";};


cTplValGesInit< std::string > & cMasqueTerrain::FileBoxMasqIsBoxTer()
{
   return mFileBoxMasqIsBoxTer;
}

const cTplValGesInit< std::string > & cMasqueTerrain::FileBoxMasqIsBoxTer()const 
{
   return mFileBoxMasqIsBoxTer;
}


std::string & cMasqueTerrain::MT_Image()
{
   return mMT_Image;
}

const std::string & cMasqueTerrain::MT_Image()const 
{
   return mMT_Image;
}


std::string & cMasqueTerrain::MT_Xml()
{
   return mMT_Xml;
}

const std::string & cMasqueTerrain::MT_Xml()const 
{
   return mMT_Xml;
}

void  BinaryUnDumpFromFile(cMasqueTerrain & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.FileBoxMasqIsBoxTer().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.FileBoxMasqIsBoxTer().ValForcedForUnUmp(),aFp);
        }
        else  anObj.FileBoxMasqIsBoxTer().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.MT_Image(),aFp);
    BinaryUnDumpFromFile(anObj.MT_Xml(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cMasqueTerrain & anObj)
{
    BinaryDumpInFile(aFp,anObj.FileBoxMasqIsBoxTer().IsInit());
    if (anObj.FileBoxMasqIsBoxTer().IsInit()) BinaryDumpInFile(aFp,anObj.FileBoxMasqIsBoxTer().Val());
    BinaryDumpInFile(aFp,anObj.MT_Image());
    BinaryDumpInFile(aFp,anObj.MT_Xml());
}

cElXMLTree * ToXMLTree(const cMasqueTerrain & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"MasqueTerrain",eXMLBranche);
   if (anObj.FileBoxMasqIsBoxTer().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("FileBoxMasqIsBoxTer"),anObj.FileBoxMasqIsBoxTer().Val())->ReTagThis("FileBoxMasqIsBoxTer"));
   aRes->AddFils(::ToXMLTree(std::string("MT_Image"),anObj.MT_Image())->ReTagThis("MT_Image"));
   aRes->AddFils(::ToXMLTree(std::string("MT_Xml"),anObj.MT_Xml())->ReTagThis("MT_Xml"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cMasqueTerrain & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.FileBoxMasqIsBoxTer(),aTree->Get("FileBoxMasqIsBoxTer",1)); //tototo 

   xml_init(anObj.MT_Image(),aTree->Get("MT_Image",1)); //tototo 

   xml_init(anObj.MT_Xml(),aTree->Get("MT_Xml",1)); //tototo 
}

std::string  Mangling( cMasqueTerrain *) {return "82DB8F0FA4C339E2FE3F";};


cTplValGesInit< Box2dr > & cPlanimetrie::BoxTerrain()
{
   return mBoxTerrain;
}

const cTplValGesInit< Box2dr > & cPlanimetrie::BoxTerrain()const 
{
   return mBoxTerrain;
}


std::list< cListePointsInclus > & cPlanimetrie::ListePointsInclus()
{
   return mListePointsInclus;
}

const std::list< cListePointsInclus > & cPlanimetrie::ListePointsInclus()const 
{
   return mListePointsInclus;
}


cTplValGesInit< double > & cPlanimetrie::RatioResolImage()
{
   return mRatioResolImage;
}

const cTplValGesInit< double > & cPlanimetrie::RatioResolImage()const 
{
   return mRatioResolImage;
}


cTplValGesInit< double > & cPlanimetrie::ResolutionTerrain()
{
   return mResolutionTerrain;
}

const cTplValGesInit< double > & cPlanimetrie::ResolutionTerrain()const 
{
   return mResolutionTerrain;
}


cTplValGesInit< bool > & cPlanimetrie::RoundSpecifiedRT()
{
   return mRoundSpecifiedRT;
}

const cTplValGesInit< bool > & cPlanimetrie::RoundSpecifiedRT()const 
{
   return mRoundSpecifiedRT;
}


cTplValGesInit< std::string > & cPlanimetrie::FilterEstimTerrain()
{
   return mFilterEstimTerrain;
}

const cTplValGesInit< std::string > & cPlanimetrie::FilterEstimTerrain()const 
{
   return mFilterEstimTerrain;
}


cTplValGesInit< std::string > & cPlanimetrie::FileBoxMasqIsBoxTer()
{
   return MasqueTerrain().Val().FileBoxMasqIsBoxTer();
}

const cTplValGesInit< std::string > & cPlanimetrie::FileBoxMasqIsBoxTer()const 
{
   return MasqueTerrain().Val().FileBoxMasqIsBoxTer();
}


std::string & cPlanimetrie::MT_Image()
{
   return MasqueTerrain().Val().MT_Image();
}

const std::string & cPlanimetrie::MT_Image()const 
{
   return MasqueTerrain().Val().MT_Image();
}


std::string & cPlanimetrie::MT_Xml()
{
   return MasqueTerrain().Val().MT_Xml();
}

const std::string & cPlanimetrie::MT_Xml()const 
{
   return MasqueTerrain().Val().MT_Xml();
}


cTplValGesInit< cMasqueTerrain > & cPlanimetrie::MasqueTerrain()
{
   return mMasqueTerrain;
}

const cTplValGesInit< cMasqueTerrain > & cPlanimetrie::MasqueTerrain()const 
{
   return mMasqueTerrain;
}


cTplValGesInit< double > & cPlanimetrie::RecouvrementMinimal()
{
   return mRecouvrementMinimal;
}

const cTplValGesInit< double > & cPlanimetrie::RecouvrementMinimal()const 
{
   return mRecouvrementMinimal;
}

void  BinaryUnDumpFromFile(cPlanimetrie & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.BoxTerrain().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.BoxTerrain().ValForcedForUnUmp(),aFp);
        }
        else  anObj.BoxTerrain().SetNoInit();
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cListePointsInclus aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.ListePointsInclus().push_back(aVal);
        }
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.RatioResolImage().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.RatioResolImage().ValForcedForUnUmp(),aFp);
        }
        else  anObj.RatioResolImage().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ResolutionTerrain().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ResolutionTerrain().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ResolutionTerrain().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.RoundSpecifiedRT().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.RoundSpecifiedRT().ValForcedForUnUmp(),aFp);
        }
        else  anObj.RoundSpecifiedRT().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.FilterEstimTerrain().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.FilterEstimTerrain().ValForcedForUnUmp(),aFp);
        }
        else  anObj.FilterEstimTerrain().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.MasqueTerrain().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.MasqueTerrain().ValForcedForUnUmp(),aFp);
        }
        else  anObj.MasqueTerrain().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.RecouvrementMinimal().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.RecouvrementMinimal().ValForcedForUnUmp(),aFp);
        }
        else  anObj.RecouvrementMinimal().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cPlanimetrie & anObj)
{
    BinaryDumpInFile(aFp,anObj.BoxTerrain().IsInit());
    if (anObj.BoxTerrain().IsInit()) BinaryDumpInFile(aFp,anObj.BoxTerrain().Val());
    BinaryDumpInFile(aFp,(int)anObj.ListePointsInclus().size());
    for(  std::list< cListePointsInclus >::const_iterator iT=anObj.ListePointsInclus().begin();
         iT!=anObj.ListePointsInclus().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.RatioResolImage().IsInit());
    if (anObj.RatioResolImage().IsInit()) BinaryDumpInFile(aFp,anObj.RatioResolImage().Val());
    BinaryDumpInFile(aFp,anObj.ResolutionTerrain().IsInit());
    if (anObj.ResolutionTerrain().IsInit()) BinaryDumpInFile(aFp,anObj.ResolutionTerrain().Val());
    BinaryDumpInFile(aFp,anObj.RoundSpecifiedRT().IsInit());
    if (anObj.RoundSpecifiedRT().IsInit()) BinaryDumpInFile(aFp,anObj.RoundSpecifiedRT().Val());
    BinaryDumpInFile(aFp,anObj.FilterEstimTerrain().IsInit());
    if (anObj.FilterEstimTerrain().IsInit()) BinaryDumpInFile(aFp,anObj.FilterEstimTerrain().Val());
    BinaryDumpInFile(aFp,anObj.MasqueTerrain().IsInit());
    if (anObj.MasqueTerrain().IsInit()) BinaryDumpInFile(aFp,anObj.MasqueTerrain().Val());
    BinaryDumpInFile(aFp,anObj.RecouvrementMinimal().IsInit());
    if (anObj.RecouvrementMinimal().IsInit()) BinaryDumpInFile(aFp,anObj.RecouvrementMinimal().Val());
}

cElXMLTree * ToXMLTree(const cPlanimetrie & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"Planimetrie",eXMLBranche);
   if (anObj.BoxTerrain().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("BoxTerrain"),anObj.BoxTerrain().Val())->ReTagThis("BoxTerrain"));
  for
  (       std::list< cListePointsInclus >::const_iterator it=anObj.ListePointsInclus().begin();
      it !=anObj.ListePointsInclus().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("ListePointsInclus"));
   if (anObj.RatioResolImage().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("RatioResolImage"),anObj.RatioResolImage().Val())->ReTagThis("RatioResolImage"));
   if (anObj.ResolutionTerrain().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ResolutionTerrain"),anObj.ResolutionTerrain().Val())->ReTagThis("ResolutionTerrain"));
   if (anObj.RoundSpecifiedRT().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("RoundSpecifiedRT"),anObj.RoundSpecifiedRT().Val())->ReTagThis("RoundSpecifiedRT"));
   if (anObj.FilterEstimTerrain().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("FilterEstimTerrain"),anObj.FilterEstimTerrain().Val())->ReTagThis("FilterEstimTerrain"));
   if (anObj.MasqueTerrain().IsInit())
      aRes->AddFils(ToXMLTree(anObj.MasqueTerrain().Val())->ReTagThis("MasqueTerrain"));
   if (anObj.RecouvrementMinimal().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("RecouvrementMinimal"),anObj.RecouvrementMinimal().Val())->ReTagThis("RecouvrementMinimal"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cPlanimetrie & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.BoxTerrain(),aTree->Get("BoxTerrain",1)); //tototo 

   xml_init(anObj.ListePointsInclus(),aTree->GetAll("ListePointsInclus",false,1));

   xml_init(anObj.RatioResolImage(),aTree->Get("RatioResolImage",1)); //tototo 

   xml_init(anObj.ResolutionTerrain(),aTree->Get("ResolutionTerrain",1)); //tototo 

   xml_init(anObj.RoundSpecifiedRT(),aTree->Get("RoundSpecifiedRT",1)); //tototo 

   xml_init(anObj.FilterEstimTerrain(),aTree->Get("FilterEstimTerrain",1),std::string(".*")); //tototo 

   xml_init(anObj.MasqueTerrain(),aTree->Get("MasqueTerrain",1)); //tototo 

   xml_init(anObj.RecouvrementMinimal(),aTree->Get("RecouvrementMinimal",1)); //tototo 
}

std::string  Mangling( cPlanimetrie *) {return "C0F9749E630C0AB9FB3F";};


cTplValGesInit< double > & cRugositeMNT::EnergieExpCorrel()
{
   return mEnergieExpCorrel;
}

const cTplValGesInit< double > & cRugositeMNT::EnergieExpCorrel()const 
{
   return mEnergieExpCorrel;
}


cTplValGesInit< double > & cRugositeMNT::EnergieExpRegulPlani()
{
   return mEnergieExpRegulPlani;
}

const cTplValGesInit< double > & cRugositeMNT::EnergieExpRegulPlani()const 
{
   return mEnergieExpRegulPlani;
}


cTplValGesInit< double > & cRugositeMNT::EnergieExpRegulAlti()
{
   return mEnergieExpRegulAlti;
}

const cTplValGesInit< double > & cRugositeMNT::EnergieExpRegulAlti()const 
{
   return mEnergieExpRegulAlti;
}

void  BinaryUnDumpFromFile(cRugositeMNT & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.EnergieExpCorrel().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.EnergieExpCorrel().ValForcedForUnUmp(),aFp);
        }
        else  anObj.EnergieExpCorrel().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.EnergieExpRegulPlani().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.EnergieExpRegulPlani().ValForcedForUnUmp(),aFp);
        }
        else  anObj.EnergieExpRegulPlani().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.EnergieExpRegulAlti().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.EnergieExpRegulAlti().ValForcedForUnUmp(),aFp);
        }
        else  anObj.EnergieExpRegulAlti().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cRugositeMNT & anObj)
{
    BinaryDumpInFile(aFp,anObj.EnergieExpCorrel().IsInit());
    if (anObj.EnergieExpCorrel().IsInit()) BinaryDumpInFile(aFp,anObj.EnergieExpCorrel().Val());
    BinaryDumpInFile(aFp,anObj.EnergieExpRegulPlani().IsInit());
    if (anObj.EnergieExpRegulPlani().IsInit()) BinaryDumpInFile(aFp,anObj.EnergieExpRegulPlani().Val());
    BinaryDumpInFile(aFp,anObj.EnergieExpRegulAlti().IsInit());
    if (anObj.EnergieExpRegulAlti().IsInit()) BinaryDumpInFile(aFp,anObj.EnergieExpRegulAlti().Val());
}

cElXMLTree * ToXMLTree(const cRugositeMNT & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"RugositeMNT",eXMLBranche);
   if (anObj.EnergieExpCorrel().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("EnergieExpCorrel"),anObj.EnergieExpCorrel().Val())->ReTagThis("EnergieExpCorrel"));
   if (anObj.EnergieExpRegulPlani().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("EnergieExpRegulPlani"),anObj.EnergieExpRegulPlani().Val())->ReTagThis("EnergieExpRegulPlani"));
   if (anObj.EnergieExpRegulAlti().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("EnergieExpRegulAlti"),anObj.EnergieExpRegulAlti().Val())->ReTagThis("EnergieExpRegulAlti"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cRugositeMNT & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.EnergieExpCorrel(),aTree->Get("EnergieExpCorrel",1),double(-2.0)); //tototo 

   xml_init(anObj.EnergieExpRegulPlani(),aTree->Get("EnergieExpRegulPlani",1),double(-1.0)); //tototo 

   xml_init(anObj.EnergieExpRegulAlti(),aTree->Get("EnergieExpRegulAlti",1),double(-1.0)); //tototo 
}

std::string  Mangling( cRugositeMNT *) {return "2AC2ED03679F85F3FE3F";};


cTplValGesInit< bool > & cSection_Terrain::IntervalPaxIsProportion()
{
   return mIntervalPaxIsProportion;
}

const cTplValGesInit< bool > & cSection_Terrain::IntervalPaxIsProportion()const 
{
   return mIntervalPaxIsProportion;
}


cTplValGesInit< double > & cSection_Terrain::RatioAltiPlani()
{
   return mRatioAltiPlani;
}

const cTplValGesInit< double > & cSection_Terrain::RatioAltiPlani()const 
{
   return mRatioAltiPlani;
}


cTplValGesInit< bool > & cSection_Terrain::EstimPxPrefZ2Prof()
{
   return mEstimPxPrefZ2Prof;
}

const cTplValGesInit< bool > & cSection_Terrain::EstimPxPrefZ2Prof()const 
{
   return mEstimPxPrefZ2Prof;
}


cTplValGesInit< double > & cSection_Terrain::ZMoyen()
{
   return IntervAltimetrie().Val().ZMoyen();
}

const cTplValGesInit< double > & cSection_Terrain::ZMoyen()const 
{
   return IntervAltimetrie().Val().ZMoyen();
}


double & cSection_Terrain::ZIncCalc()
{
   return IntervAltimetrie().Val().ZIncCalc();
}

const double & cSection_Terrain::ZIncCalc()const 
{
   return IntervAltimetrie().Val().ZIncCalc();
}


cTplValGesInit< bool > & cSection_Terrain::ZIncIsProp()
{
   return IntervAltimetrie().Val().ZIncIsProp();
}

const cTplValGesInit< bool > & cSection_Terrain::ZIncIsProp()const 
{
   return IntervAltimetrie().Val().ZIncIsProp();
}


cTplValGesInit< double > & cSection_Terrain::ZIncZonage()
{
   return IntervAltimetrie().Val().ZIncZonage();
}

const cTplValGesInit< double > & cSection_Terrain::ZIncZonage()const 
{
   return IntervAltimetrie().Val().ZIncZonage();
}


std::string & cSection_Terrain::MNT_Init_Image()
{
   return IntervAltimetrie().Val().MNT_Init().Val().MNT_Init_Image();
}

const std::string & cSection_Terrain::MNT_Init_Image()const 
{
   return IntervAltimetrie().Val().MNT_Init().Val().MNT_Init_Image();
}


std::string & cSection_Terrain::MNT_Init_Xml()
{
   return IntervAltimetrie().Val().MNT_Init().Val().MNT_Init_Xml();
}

const std::string & cSection_Terrain::MNT_Init_Xml()const 
{
   return IntervAltimetrie().Val().MNT_Init().Val().MNT_Init_Xml();
}


cTplValGesInit< double > & cSection_Terrain::MNT_Offset()
{
   return IntervAltimetrie().Val().MNT_Init().Val().MNT_Offset();
}

const cTplValGesInit< double > & cSection_Terrain::MNT_Offset()const 
{
   return IntervAltimetrie().Val().MNT_Init().Val().MNT_Offset();
}


cTplValGesInit< cMNT_Init > & cSection_Terrain::MNT_Init()
{
   return IntervAltimetrie().Val().MNT_Init();
}

const cTplValGesInit< cMNT_Init > & cSection_Terrain::MNT_Init()const 
{
   return IntervAltimetrie().Val().MNT_Init();
}


std::string & cSection_Terrain::ZInf()
{
   return IntervAltimetrie().Val().EnveloppeMNT_INIT().Val().ZInf();
}

const std::string & cSection_Terrain::ZInf()const 
{
   return IntervAltimetrie().Val().EnveloppeMNT_INIT().Val().ZInf();
}


std::string & cSection_Terrain::ZSup()
{
   return IntervAltimetrie().Val().EnveloppeMNT_INIT().Val().ZSup();
}

const std::string & cSection_Terrain::ZSup()const 
{
   return IntervAltimetrie().Val().EnveloppeMNT_INIT().Val().ZSup();
}


cTplValGesInit< cEnveloppeMNT_INIT > & cSection_Terrain::EnveloppeMNT_INIT()
{
   return IntervAltimetrie().Val().EnveloppeMNT_INIT();
}

const cTplValGesInit< cEnveloppeMNT_INIT > & cSection_Terrain::EnveloppeMNT_INIT()const 
{
   return IntervAltimetrie().Val().EnveloppeMNT_INIT();
}


cTplValGesInit< cIntervAltimetrie > & cSection_Terrain::IntervAltimetrie()
{
   return mIntervAltimetrie;
}

const cTplValGesInit< cIntervAltimetrie > & cSection_Terrain::IntervAltimetrie()const 
{
   return mIntervAltimetrie;
}


cTplValGesInit< double > & cSection_Terrain::Px1Moy()
{
   return IntervParalaxe().Val().Px1Moy();
}

const cTplValGesInit< double > & cSection_Terrain::Px1Moy()const 
{
   return IntervParalaxe().Val().Px1Moy();
}


cTplValGesInit< double > & cSection_Terrain::Px2Moy()
{
   return IntervParalaxe().Val().Px2Moy();
}

const cTplValGesInit< double > & cSection_Terrain::Px2Moy()const 
{
   return IntervParalaxe().Val().Px2Moy();
}


double & cSection_Terrain::Px1IncCalc()
{
   return IntervParalaxe().Val().Px1IncCalc();
}

const double & cSection_Terrain::Px1IncCalc()const 
{
   return IntervParalaxe().Val().Px1IncCalc();
}


cTplValGesInit< double > & cSection_Terrain::Px1PropProf()
{
   return IntervParalaxe().Val().Px1PropProf();
}

const cTplValGesInit< double > & cSection_Terrain::Px1PropProf()const 
{
   return IntervParalaxe().Val().Px1PropProf();
}


cTplValGesInit< double > & cSection_Terrain::Px2IncCalc()
{
   return IntervParalaxe().Val().Px2IncCalc();
}

const cTplValGesInit< double > & cSection_Terrain::Px2IncCalc()const 
{
   return IntervParalaxe().Val().Px2IncCalc();
}


cTplValGesInit< double > & cSection_Terrain::Px1IncZonage()
{
   return IntervParalaxe().Val().Px1IncZonage();
}

const cTplValGesInit< double > & cSection_Terrain::Px1IncZonage()const 
{
   return IntervParalaxe().Val().Px1IncZonage();
}


cTplValGesInit< double > & cSection_Terrain::Px2IncZonage()
{
   return IntervParalaxe().Val().Px2IncZonage();
}

const cTplValGesInit< double > & cSection_Terrain::Px2IncZonage()const 
{
   return IntervParalaxe().Val().Px2IncZonage();
}


cTplValGesInit< cIntervParalaxe > & cSection_Terrain::IntervParalaxe()
{
   return mIntervParalaxe;
}

const cTplValGesInit< cIntervParalaxe > & cSection_Terrain::IntervParalaxe()const 
{
   return mIntervParalaxe;
}


std::string & cSection_Terrain::NameNuageXML()
{
   return NuageXMLInit().Val().NameNuageXML();
}

const std::string & cSection_Terrain::NameNuageXML()const 
{
   return NuageXMLInit().Val().NameNuageXML();
}


cTplValGesInit< bool > & cSection_Terrain::CanAdaptGeom()
{
   return NuageXMLInit().Val().CanAdaptGeom();
}

const cTplValGesInit< bool > & cSection_Terrain::CanAdaptGeom()const 
{
   return NuageXMLInit().Val().CanAdaptGeom();
}


cTplValGesInit< cNuageXMLInit > & cSection_Terrain::NuageXMLInit()
{
   return mNuageXMLInit;
}

const cTplValGesInit< cNuageXMLInit > & cSection_Terrain::NuageXMLInit()const 
{
   return mNuageXMLInit;
}


double & cSection_Terrain::MulZMin()
{
   return IntervSpecialZInv().Val().MulZMin();
}

const double & cSection_Terrain::MulZMin()const 
{
   return IntervSpecialZInv().Val().MulZMin();
}


double & cSection_Terrain::MulZMax()
{
   return IntervSpecialZInv().Val().MulZMax();
}

const double & cSection_Terrain::MulZMax()const 
{
   return IntervSpecialZInv().Val().MulZMax();
}


cTplValGesInit< cIntervSpecialZInv > & cSection_Terrain::IntervSpecialZInv()
{
   return mIntervSpecialZInv;
}

const cTplValGesInit< cIntervSpecialZInv > & cSection_Terrain::IntervSpecialZInv()const 
{
   return mIntervSpecialZInv;
}


cTplValGesInit< bool > & cSection_Terrain::GeoRefAutoRoundResol()
{
   return mGeoRefAutoRoundResol;
}

const cTplValGesInit< bool > & cSection_Terrain::GeoRefAutoRoundResol()const 
{
   return mGeoRefAutoRoundResol;
}


cTplValGesInit< bool > & cSection_Terrain::GeoRefAutoRoundBox()
{
   return mGeoRefAutoRoundBox;
}

const cTplValGesInit< bool > & cSection_Terrain::GeoRefAutoRoundBox()const 
{
   return mGeoRefAutoRoundBox;
}


cTplValGesInit< Box2dr > & cSection_Terrain::BoxTerrain()
{
   return Planimetrie().Val().BoxTerrain();
}

const cTplValGesInit< Box2dr > & cSection_Terrain::BoxTerrain()const 
{
   return Planimetrie().Val().BoxTerrain();
}


std::list< cListePointsInclus > & cSection_Terrain::ListePointsInclus()
{
   return Planimetrie().Val().ListePointsInclus();
}

const std::list< cListePointsInclus > & cSection_Terrain::ListePointsInclus()const 
{
   return Planimetrie().Val().ListePointsInclus();
}


cTplValGesInit< double > & cSection_Terrain::RatioResolImage()
{
   return Planimetrie().Val().RatioResolImage();
}

const cTplValGesInit< double > & cSection_Terrain::RatioResolImage()const 
{
   return Planimetrie().Val().RatioResolImage();
}


cTplValGesInit< double > & cSection_Terrain::ResolutionTerrain()
{
   return Planimetrie().Val().ResolutionTerrain();
}

const cTplValGesInit< double > & cSection_Terrain::ResolutionTerrain()const 
{
   return Planimetrie().Val().ResolutionTerrain();
}


cTplValGesInit< bool > & cSection_Terrain::RoundSpecifiedRT()
{
   return Planimetrie().Val().RoundSpecifiedRT();
}

const cTplValGesInit< bool > & cSection_Terrain::RoundSpecifiedRT()const 
{
   return Planimetrie().Val().RoundSpecifiedRT();
}


cTplValGesInit< std::string > & cSection_Terrain::FilterEstimTerrain()
{
   return Planimetrie().Val().FilterEstimTerrain();
}

const cTplValGesInit< std::string > & cSection_Terrain::FilterEstimTerrain()const 
{
   return Planimetrie().Val().FilterEstimTerrain();
}


cTplValGesInit< std::string > & cSection_Terrain::FileBoxMasqIsBoxTer()
{
   return Planimetrie().Val().MasqueTerrain().Val().FileBoxMasqIsBoxTer();
}

const cTplValGesInit< std::string > & cSection_Terrain::FileBoxMasqIsBoxTer()const 
{
   return Planimetrie().Val().MasqueTerrain().Val().FileBoxMasqIsBoxTer();
}


std::string & cSection_Terrain::MT_Image()
{
   return Planimetrie().Val().MasqueTerrain().Val().MT_Image();
}

const std::string & cSection_Terrain::MT_Image()const 
{
   return Planimetrie().Val().MasqueTerrain().Val().MT_Image();
}


std::string & cSection_Terrain::MT_Xml()
{
   return Planimetrie().Val().MasqueTerrain().Val().MT_Xml();
}

const std::string & cSection_Terrain::MT_Xml()const 
{
   return Planimetrie().Val().MasqueTerrain().Val().MT_Xml();
}


cTplValGesInit< cMasqueTerrain > & cSection_Terrain::MasqueTerrain()
{
   return Planimetrie().Val().MasqueTerrain();
}

const cTplValGesInit< cMasqueTerrain > & cSection_Terrain::MasqueTerrain()const 
{
   return Planimetrie().Val().MasqueTerrain();
}


cTplValGesInit< double > & cSection_Terrain::RecouvrementMinimal()
{
   return Planimetrie().Val().RecouvrementMinimal();
}

const cTplValGesInit< double > & cSection_Terrain::RecouvrementMinimal()const 
{
   return Planimetrie().Val().RecouvrementMinimal();
}


cTplValGesInit< cPlanimetrie > & cSection_Terrain::Planimetrie()
{
   return mPlanimetrie;
}

const cTplValGesInit< cPlanimetrie > & cSection_Terrain::Planimetrie()const 
{
   return mPlanimetrie;
}


cTplValGesInit< std::string > & cSection_Terrain::FileOriMnt()
{
   return mFileOriMnt;
}

const cTplValGesInit< std::string > & cSection_Terrain::FileOriMnt()const 
{
   return mFileOriMnt;
}


cTplValGesInit< double > & cSection_Terrain::EnergieExpCorrel()
{
   return RugositeMNT().Val().EnergieExpCorrel();
}

const cTplValGesInit< double > & cSection_Terrain::EnergieExpCorrel()const 
{
   return RugositeMNT().Val().EnergieExpCorrel();
}


cTplValGesInit< double > & cSection_Terrain::EnergieExpRegulPlani()
{
   return RugositeMNT().Val().EnergieExpRegulPlani();
}

const cTplValGesInit< double > & cSection_Terrain::EnergieExpRegulPlani()const 
{
   return RugositeMNT().Val().EnergieExpRegulPlani();
}


cTplValGesInit< double > & cSection_Terrain::EnergieExpRegulAlti()
{
   return RugositeMNT().Val().EnergieExpRegulAlti();
}

const cTplValGesInit< double > & cSection_Terrain::EnergieExpRegulAlti()const 
{
   return RugositeMNT().Val().EnergieExpRegulAlti();
}


cTplValGesInit< cRugositeMNT > & cSection_Terrain::RugositeMNT()
{
   return mRugositeMNT;
}

const cTplValGesInit< cRugositeMNT > & cSection_Terrain::RugositeMNT()const 
{
   return mRugositeMNT;
}

void  BinaryUnDumpFromFile(cSection_Terrain & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.IntervalPaxIsProportion().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.IntervalPaxIsProportion().ValForcedForUnUmp(),aFp);
        }
        else  anObj.IntervalPaxIsProportion().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.RatioAltiPlani().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.RatioAltiPlani().ValForcedForUnUmp(),aFp);
        }
        else  anObj.RatioAltiPlani().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.EstimPxPrefZ2Prof().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.EstimPxPrefZ2Prof().ValForcedForUnUmp(),aFp);
        }
        else  anObj.EstimPxPrefZ2Prof().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.IntervAltimetrie().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.IntervAltimetrie().ValForcedForUnUmp(),aFp);
        }
        else  anObj.IntervAltimetrie().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.IntervParalaxe().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.IntervParalaxe().ValForcedForUnUmp(),aFp);
        }
        else  anObj.IntervParalaxe().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NuageXMLInit().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NuageXMLInit().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NuageXMLInit().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.IntervSpecialZInv().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.IntervSpecialZInv().ValForcedForUnUmp(),aFp);
        }
        else  anObj.IntervSpecialZInv().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.GeoRefAutoRoundResol().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.GeoRefAutoRoundResol().ValForcedForUnUmp(),aFp);
        }
        else  anObj.GeoRefAutoRoundResol().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.GeoRefAutoRoundBox().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.GeoRefAutoRoundBox().ValForcedForUnUmp(),aFp);
        }
        else  anObj.GeoRefAutoRoundBox().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Planimetrie().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Planimetrie().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Planimetrie().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.FileOriMnt().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.FileOriMnt().ValForcedForUnUmp(),aFp);
        }
        else  anObj.FileOriMnt().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.RugositeMNT().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.RugositeMNT().ValForcedForUnUmp(),aFp);
        }
        else  anObj.RugositeMNT().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cSection_Terrain & anObj)
{
    BinaryDumpInFile(aFp,anObj.IntervalPaxIsProportion().IsInit());
    if (anObj.IntervalPaxIsProportion().IsInit()) BinaryDumpInFile(aFp,anObj.IntervalPaxIsProportion().Val());
    BinaryDumpInFile(aFp,anObj.RatioAltiPlani().IsInit());
    if (anObj.RatioAltiPlani().IsInit()) BinaryDumpInFile(aFp,anObj.RatioAltiPlani().Val());
    BinaryDumpInFile(aFp,anObj.EstimPxPrefZ2Prof().IsInit());
    if (anObj.EstimPxPrefZ2Prof().IsInit()) BinaryDumpInFile(aFp,anObj.EstimPxPrefZ2Prof().Val());
    BinaryDumpInFile(aFp,anObj.IntervAltimetrie().IsInit());
    if (anObj.IntervAltimetrie().IsInit()) BinaryDumpInFile(aFp,anObj.IntervAltimetrie().Val());
    BinaryDumpInFile(aFp,anObj.IntervParalaxe().IsInit());
    if (anObj.IntervParalaxe().IsInit()) BinaryDumpInFile(aFp,anObj.IntervParalaxe().Val());
    BinaryDumpInFile(aFp,anObj.NuageXMLInit().IsInit());
    if (anObj.NuageXMLInit().IsInit()) BinaryDumpInFile(aFp,anObj.NuageXMLInit().Val());
    BinaryDumpInFile(aFp,anObj.IntervSpecialZInv().IsInit());
    if (anObj.IntervSpecialZInv().IsInit()) BinaryDumpInFile(aFp,anObj.IntervSpecialZInv().Val());
    BinaryDumpInFile(aFp,anObj.GeoRefAutoRoundResol().IsInit());
    if (anObj.GeoRefAutoRoundResol().IsInit()) BinaryDumpInFile(aFp,anObj.GeoRefAutoRoundResol().Val());
    BinaryDumpInFile(aFp,anObj.GeoRefAutoRoundBox().IsInit());
    if (anObj.GeoRefAutoRoundBox().IsInit()) BinaryDumpInFile(aFp,anObj.GeoRefAutoRoundBox().Val());
    BinaryDumpInFile(aFp,anObj.Planimetrie().IsInit());
    if (anObj.Planimetrie().IsInit()) BinaryDumpInFile(aFp,anObj.Planimetrie().Val());
    BinaryDumpInFile(aFp,anObj.FileOriMnt().IsInit());
    if (anObj.FileOriMnt().IsInit()) BinaryDumpInFile(aFp,anObj.FileOriMnt().Val());
    BinaryDumpInFile(aFp,anObj.RugositeMNT().IsInit());
    if (anObj.RugositeMNT().IsInit()) BinaryDumpInFile(aFp,anObj.RugositeMNT().Val());
}

cElXMLTree * ToXMLTree(const cSection_Terrain & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"Section_Terrain",eXMLBranche);
   if (anObj.IntervalPaxIsProportion().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("IntervalPaxIsProportion"),anObj.IntervalPaxIsProportion().Val())->ReTagThis("IntervalPaxIsProportion"));
   if (anObj.RatioAltiPlani().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("RatioAltiPlani"),anObj.RatioAltiPlani().Val())->ReTagThis("RatioAltiPlani"));
   if (anObj.EstimPxPrefZ2Prof().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("EstimPxPrefZ2Prof"),anObj.EstimPxPrefZ2Prof().Val())->ReTagThis("EstimPxPrefZ2Prof"));
   if (anObj.IntervAltimetrie().IsInit())
      aRes->AddFils(ToXMLTree(anObj.IntervAltimetrie().Val())->ReTagThis("IntervAltimetrie"));
   if (anObj.IntervParalaxe().IsInit())
      aRes->AddFils(ToXMLTree(anObj.IntervParalaxe().Val())->ReTagThis("IntervParalaxe"));
   if (anObj.NuageXMLInit().IsInit())
      aRes->AddFils(ToXMLTree(anObj.NuageXMLInit().Val())->ReTagThis("NuageXMLInit"));
   if (anObj.IntervSpecialZInv().IsInit())
      aRes->AddFils(ToXMLTree(anObj.IntervSpecialZInv().Val())->ReTagThis("IntervSpecialZInv"));
   if (anObj.GeoRefAutoRoundResol().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("GeoRefAutoRoundResol"),anObj.GeoRefAutoRoundResol().Val())->ReTagThis("GeoRefAutoRoundResol"));
   if (anObj.GeoRefAutoRoundBox().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("GeoRefAutoRoundBox"),anObj.GeoRefAutoRoundBox().Val())->ReTagThis("GeoRefAutoRoundBox"));
   if (anObj.Planimetrie().IsInit())
      aRes->AddFils(ToXMLTree(anObj.Planimetrie().Val())->ReTagThis("Planimetrie"));
   if (anObj.FileOriMnt().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("FileOriMnt"),anObj.FileOriMnt().Val())->ReTagThis("FileOriMnt"));
   if (anObj.RugositeMNT().IsInit())
      aRes->AddFils(ToXMLTree(anObj.RugositeMNT().Val())->ReTagThis("RugositeMNT"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cSection_Terrain & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.IntervalPaxIsProportion(),aTree->Get("IntervalPaxIsProportion",1),bool(false)); //tototo 

   xml_init(anObj.RatioAltiPlani(),aTree->Get("RatioAltiPlani",1)); //tototo 

   xml_init(anObj.EstimPxPrefZ2Prof(),aTree->Get("EstimPxPrefZ2Prof",1),bool(false)); //tototo 

   xml_init(anObj.IntervAltimetrie(),aTree->Get("IntervAltimetrie",1)); //tototo 

   xml_init(anObj.IntervParalaxe(),aTree->Get("IntervParalaxe",1)); //tototo 

   xml_init(anObj.NuageXMLInit(),aTree->Get("NuageXMLInit",1)); //tototo 

   xml_init(anObj.IntervSpecialZInv(),aTree->Get("IntervSpecialZInv",1)); //tototo 

   xml_init(anObj.GeoRefAutoRoundResol(),aTree->Get("GeoRefAutoRoundResol",1)); //tototo 

   xml_init(anObj.GeoRefAutoRoundBox(),aTree->Get("GeoRefAutoRoundBox",1)); //tototo 

   xml_init(anObj.Planimetrie(),aTree->Get("Planimetrie",1)); //tototo 

   xml_init(anObj.FileOriMnt(),aTree->Get("FileOriMnt",1)); //tototo 

   xml_init(anObj.RugositeMNT(),aTree->Get("RugositeMNT",1)); //tototo 
}

std::string  Mangling( cSection_Terrain *) {return "D2CAA26C77848BF0FD3F";};


cElRegex_Ptr & cOneMasqueImage::PatternSel()
{
   return mPatternSel;
}

const cElRegex_Ptr & cOneMasqueImage::PatternSel()const 
{
   return mPatternSel;
}


std::string & cOneMasqueImage::NomMasq()
{
   return mNomMasq;
}

const std::string & cOneMasqueImage::NomMasq()const 
{
   return mNomMasq;
}

void  BinaryUnDumpFromFile(cOneMasqueImage & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.PatternSel(),aFp);
    BinaryUnDumpFromFile(anObj.NomMasq(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cOneMasqueImage & anObj)
{
    BinaryDumpInFile(aFp,anObj.PatternSel());
    BinaryDumpInFile(aFp,anObj.NomMasq());
}

cElXMLTree * ToXMLTree(const cOneMasqueImage & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"OneMasqueImage",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("PatternSel"),anObj.PatternSel())->ReTagThis("PatternSel"));
   aRes->AddFils(::ToXMLTree(std::string("NomMasq"),anObj.NomMasq())->ReTagThis("NomMasq"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cOneMasqueImage & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.PatternSel(),aTree->Get("PatternSel",1)); //tototo 

   xml_init(anObj.NomMasq(),aTree->Get("NomMasq",1)); //tototo 
}

std::string  Mangling( cOneMasqueImage *) {return "70ECF80571E42B8EFE3F";};


std::list< cOneMasqueImage > & cMasqImageIn::OneMasqueImage()
{
   return mOneMasqueImage;
}

const std::list< cOneMasqueImage > & cMasqImageIn::OneMasqueImage()const 
{
   return mOneMasqueImage;
}


cTplValGesInit< bool > & cMasqImageIn::AcceptNonExistingFile()
{
   return mAcceptNonExistingFile;
}

const cTplValGesInit< bool > & cMasqImageIn::AcceptNonExistingFile()const 
{
   return mAcceptNonExistingFile;
}

void  BinaryUnDumpFromFile(cMasqImageIn & anObj,ELISE_fp & aFp)
{
   { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cOneMasqueImage aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.OneMasqueImage().push_back(aVal);
        }
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.AcceptNonExistingFile().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.AcceptNonExistingFile().ValForcedForUnUmp(),aFp);
        }
        else  anObj.AcceptNonExistingFile().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cMasqImageIn & anObj)
{
    BinaryDumpInFile(aFp,(int)anObj.OneMasqueImage().size());
    for(  std::list< cOneMasqueImage >::const_iterator iT=anObj.OneMasqueImage().begin();
         iT!=anObj.OneMasqueImage().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.AcceptNonExistingFile().IsInit());
    if (anObj.AcceptNonExistingFile().IsInit()) BinaryDumpInFile(aFp,anObj.AcceptNonExistingFile().Val());
}

cElXMLTree * ToXMLTree(const cMasqImageIn & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"MasqImageIn",eXMLBranche);
  for
  (       std::list< cOneMasqueImage >::const_iterator it=anObj.OneMasqueImage().begin();
      it !=anObj.OneMasqueImage().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("OneMasqueImage"));
   if (anObj.AcceptNonExistingFile().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("AcceptNonExistingFile"),anObj.AcceptNonExistingFile().Val())->ReTagThis("AcceptNonExistingFile"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cMasqImageIn & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.OneMasqueImage(),aTree->GetAll("OneMasqueImage",false,1));

   xml_init(anObj.AcceptNonExistingFile(),aTree->Get("AcceptNonExistingFile",1),bool(false)); //tototo 
}

std::string  Mangling( cMasqImageIn *) {return "3B811787D3389995FE3F";};


std::string & cModuleGeomImage::NomModule()
{
   return mNomModule;
}

const std::string & cModuleGeomImage::NomModule()const 
{
   return mNomModule;
}


std::string & cModuleGeomImage::NomGeometrie()
{
   return mNomGeometrie;
}

const std::string & cModuleGeomImage::NomGeometrie()const 
{
   return mNomGeometrie;
}

void  BinaryUnDumpFromFile(cModuleGeomImage & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.NomModule(),aFp);
    BinaryUnDumpFromFile(anObj.NomGeometrie(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cModuleGeomImage & anObj)
{
    BinaryDumpInFile(aFp,anObj.NomModule());
    BinaryDumpInFile(aFp,anObj.NomGeometrie());
}

cElXMLTree * ToXMLTree(const cModuleGeomImage & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ModuleGeomImage",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("NomModule"),anObj.NomModule())->ReTagThis("NomModule"));
   aRes->AddFils(::ToXMLTree(std::string("NomGeometrie"),anObj.NomGeometrie())->ReTagThis("NomGeometrie"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cModuleGeomImage & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.NomModule(),aTree->Get("NomModule",1)); //tototo 

   xml_init(anObj.NomGeometrie(),aTree->Get("NomGeometrie",1)); //tototo 
}

std::string  Mangling( cModuleGeomImage *) {return "AA494CDF8AFCC1DDFE3F";};


std::string & cFCND_CalcIm2fromIm1::I2FromI1Key()
{
   return mI2FromI1Key;
}

const std::string & cFCND_CalcIm2fromIm1::I2FromI1Key()const 
{
   return mI2FromI1Key;
}


bool & cFCND_CalcIm2fromIm1::I2FromI1SensDirect()
{
   return mI2FromI1SensDirect;
}

const bool & cFCND_CalcIm2fromIm1::I2FromI1SensDirect()const 
{
   return mI2FromI1SensDirect;
}

void  BinaryUnDumpFromFile(cFCND_CalcIm2fromIm1 & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.I2FromI1Key(),aFp);
    BinaryUnDumpFromFile(anObj.I2FromI1SensDirect(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cFCND_CalcIm2fromIm1 & anObj)
{
    BinaryDumpInFile(aFp,anObj.I2FromI1Key());
    BinaryDumpInFile(aFp,anObj.I2FromI1SensDirect());
}

cElXMLTree * ToXMLTree(const cFCND_CalcIm2fromIm1 & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"FCND_CalcIm2fromIm1",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("I2FromI1Key"),anObj.I2FromI1Key())->ReTagThis("I2FromI1Key"));
   aRes->AddFils(::ToXMLTree(std::string("I2FromI1SensDirect"),anObj.I2FromI1SensDirect())->ReTagThis("I2FromI1SensDirect"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cFCND_CalcIm2fromIm1 & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.I2FromI1Key(),aTree->Get("I2FromI1Key",1)); //tototo 

   xml_init(anObj.I2FromI1SensDirect(),aTree->Get("I2FromI1SensDirect",1)); //tototo 
}

std::string  Mangling( cFCND_CalcIm2fromIm1 *) {return "8200D94781B450BBFF3F";};


std::string & cImSecCalcApero::Key()
{
   return mKey;
}

const std::string & cImSecCalcApero::Key()const 
{
   return mKey;
}


cTplValGesInit< int > & cImSecCalcApero::Nb()
{
   return mNb;
}

const cTplValGesInit< int > & cImSecCalcApero::Nb()const 
{
   return mNb;
}


cTplValGesInit< int > & cImSecCalcApero::NbMin()
{
   return mNbMin;
}

const cTplValGesInit< int > & cImSecCalcApero::NbMin()const 
{
   return mNbMin;
}


cTplValGesInit< int > & cImSecCalcApero::NbMax()
{
   return mNbMax;
}

const cTplValGesInit< int > & cImSecCalcApero::NbMax()const 
{
   return mNbMax;
}


cTplValGesInit< eOnEmptyImSecApero > & cImSecCalcApero::OnEmpty()
{
   return mOnEmpty;
}

const cTplValGesInit< eOnEmptyImSecApero > & cImSecCalcApero::OnEmpty()const 
{
   return mOnEmpty;
}

void  BinaryUnDumpFromFile(cImSecCalcApero & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Key(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Nb().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Nb().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Nb().SetNoInit();
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
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.OnEmpty().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.OnEmpty().ValForcedForUnUmp(),aFp);
        }
        else  anObj.OnEmpty().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cImSecCalcApero & anObj)
{
    BinaryDumpInFile(aFp,anObj.Key());
    BinaryDumpInFile(aFp,anObj.Nb().IsInit());
    if (anObj.Nb().IsInit()) BinaryDumpInFile(aFp,anObj.Nb().Val());
    BinaryDumpInFile(aFp,anObj.NbMin().IsInit());
    if (anObj.NbMin().IsInit()) BinaryDumpInFile(aFp,anObj.NbMin().Val());
    BinaryDumpInFile(aFp,anObj.NbMax().IsInit());
    if (anObj.NbMax().IsInit()) BinaryDumpInFile(aFp,anObj.NbMax().Val());
    BinaryDumpInFile(aFp,anObj.OnEmpty().IsInit());
    if (anObj.OnEmpty().IsInit()) BinaryDumpInFile(aFp,anObj.OnEmpty().Val());
}

cElXMLTree * ToXMLTree(const cImSecCalcApero & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ImSecCalcApero",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Key"),anObj.Key())->ReTagThis("Key"));
   if (anObj.Nb().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Nb"),anObj.Nb().Val())->ReTagThis("Nb"));
   if (anObj.NbMin().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NbMin"),anObj.NbMin().Val())->ReTagThis("NbMin"));
   if (anObj.NbMax().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NbMax"),anObj.NbMax().Val())->ReTagThis("NbMax"));
   if (anObj.OnEmpty().IsInit())
      aRes->AddFils(ToXMLTree(std::string("OnEmpty"),anObj.OnEmpty().Val())->ReTagThis("OnEmpty"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cImSecCalcApero & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.Key(),aTree->Get("Key",1)); //tototo 

   xml_init(anObj.Nb(),aTree->Get("Nb",1),int(-1)); //tototo 

   xml_init(anObj.NbMin(),aTree->Get("NbMin",1),int(-1)); //tototo 

   xml_init(anObj.NbMax(),aTree->Get("NbMax",1),int(1000)); //tototo 

   xml_init(anObj.OnEmpty(),aTree->Get("OnEmpty",1),eOnEmptyImSecApero(eOEISA_error)); //tototo 
}

std::string  Mangling( cImSecCalcApero *) {return "4A151FE6A20ED7A4FD3F";};


double & cAutoSelectionneImSec::RecouvrMin()
{
   return mRecouvrMin;
}

const double & cAutoSelectionneImSec::RecouvrMin()const 
{
   return mRecouvrMin;
}

void  BinaryUnDumpFromFile(cAutoSelectionneImSec & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.RecouvrMin(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cAutoSelectionneImSec & anObj)
{
    BinaryDumpInFile(aFp,anObj.RecouvrMin());
}

cElXMLTree * ToXMLTree(const cAutoSelectionneImSec & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"AutoSelectionneImSec",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("RecouvrMin"),anObj.RecouvrMin())->ReTagThis("RecouvrMin"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cAutoSelectionneImSec & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.RecouvrMin(),aTree->Get("RecouvrMin",1)); //tototo 
}

std::string  Mangling( cAutoSelectionneImSec *) {return "A8C479B481740280FE3F";};


cTplValGesInit< std::string > & cImages::Im1()
{
   return mIm1;
}

const cTplValGesInit< std::string > & cImages::Im1()const 
{
   return mIm1;
}


cTplValGesInit< std::string > & cImages::Im2()
{
   return mIm2;
}

const cTplValGesInit< std::string > & cImages::Im2()const 
{
   return mIm2;
}


std::string & cImages::I2FromI1Key()
{
   return FCND_CalcIm2fromIm1().Val().I2FromI1Key();
}

const std::string & cImages::I2FromI1Key()const 
{
   return FCND_CalcIm2fromIm1().Val().I2FromI1Key();
}


bool & cImages::I2FromI1SensDirect()
{
   return FCND_CalcIm2fromIm1().Val().I2FromI1SensDirect();
}

const bool & cImages::I2FromI1SensDirect()const 
{
   return FCND_CalcIm2fromIm1().Val().I2FromI1SensDirect();
}


cTplValGesInit< cFCND_CalcIm2fromIm1 > & cImages::FCND_CalcIm2fromIm1()
{
   return mFCND_CalcIm2fromIm1;
}

const cTplValGesInit< cFCND_CalcIm2fromIm1 > & cImages::FCND_CalcIm2fromIm1()const 
{
   return mFCND_CalcIm2fromIm1;
}


std::list< std::string > & cImages::ImPat()
{
   return mImPat;
}

const std::list< std::string > & cImages::ImPat()const 
{
   return mImPat;
}


cTplValGesInit< std::string > & cImages::ImageSecByCAWSI()
{
   return mImageSecByCAWSI;
}

const cTplValGesInit< std::string > & cImages::ImageSecByCAWSI()const 
{
   return mImageSecByCAWSI;
}


std::string & cImages::Key()
{
   return ImSecCalcApero().Val().Key();
}

const std::string & cImages::Key()const 
{
   return ImSecCalcApero().Val().Key();
}


cTplValGesInit< int > & cImages::Nb()
{
   return ImSecCalcApero().Val().Nb();
}

const cTplValGesInit< int > & cImages::Nb()const 
{
   return ImSecCalcApero().Val().Nb();
}


cTplValGesInit< int > & cImages::NbMin()
{
   return ImSecCalcApero().Val().NbMin();
}

const cTplValGesInit< int > & cImages::NbMin()const 
{
   return ImSecCalcApero().Val().NbMin();
}


cTplValGesInit< int > & cImages::NbMax()
{
   return ImSecCalcApero().Val().NbMax();
}

const cTplValGesInit< int > & cImages::NbMax()const 
{
   return ImSecCalcApero().Val().NbMax();
}


cTplValGesInit< eOnEmptyImSecApero > & cImages::OnEmpty()
{
   return ImSecCalcApero().Val().OnEmpty();
}

const cTplValGesInit< eOnEmptyImSecApero > & cImages::OnEmpty()const 
{
   return ImSecCalcApero().Val().OnEmpty();
}


cTplValGesInit< cImSecCalcApero > & cImages::ImSecCalcApero()
{
   return mImSecCalcApero;
}

const cTplValGesInit< cImSecCalcApero > & cImages::ImSecCalcApero()const 
{
   return mImSecCalcApero;
}


cTplValGesInit< cParamGenereStrVois > & cImages::RelGlobSelecteur()
{
   return mRelGlobSelecteur;
}

const cTplValGesInit< cParamGenereStrVois > & cImages::RelGlobSelecteur()const 
{
   return mRelGlobSelecteur;
}


cTplValGesInit< cNameFilter > & cImages::Filter()
{
   return mFilter;
}

const cTplValGesInit< cNameFilter > & cImages::Filter()const 
{
   return mFilter;
}


double & cImages::RecouvrMin()
{
   return AutoSelectionneImSec().Val().RecouvrMin();
}

const double & cImages::RecouvrMin()const 
{
   return AutoSelectionneImSec().Val().RecouvrMin();
}


cTplValGesInit< cAutoSelectionneImSec > & cImages::AutoSelectionneImSec()
{
   return mAutoSelectionneImSec;
}

const cTplValGesInit< cAutoSelectionneImSec > & cImages::AutoSelectionneImSec()const 
{
   return mAutoSelectionneImSec;
}


cTplValGesInit< cListImByDelta > & cImages::ImSecByDelta()
{
   return mImSecByDelta;
}

const cTplValGesInit< cListImByDelta > & cImages::ImSecByDelta()const 
{
   return mImSecByDelta;
}


cTplValGesInit< std::string > & cImages::Im3Superp()
{
   return mIm3Superp;
}

const cTplValGesInit< std::string > & cImages::Im3Superp()const 
{
   return mIm3Superp;
}

void  BinaryUnDumpFromFile(cImages & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Im1().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Im1().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Im1().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Im2().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Im2().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Im2().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.FCND_CalcIm2fromIm1().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.FCND_CalcIm2fromIm1().ValForcedForUnUmp(),aFp);
        }
        else  anObj.FCND_CalcIm2fromIm1().SetNoInit();
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             std::string aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.ImPat().push_back(aVal);
        }
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ImageSecByCAWSI().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ImageSecByCAWSI().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ImageSecByCAWSI().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ImSecCalcApero().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ImSecCalcApero().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ImSecCalcApero().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.RelGlobSelecteur().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.RelGlobSelecteur().ValForcedForUnUmp(),aFp);
        }
        else  anObj.RelGlobSelecteur().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Filter().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Filter().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Filter().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.AutoSelectionneImSec().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.AutoSelectionneImSec().ValForcedForUnUmp(),aFp);
        }
        else  anObj.AutoSelectionneImSec().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ImSecByDelta().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ImSecByDelta().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ImSecByDelta().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Im3Superp().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Im3Superp().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Im3Superp().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cImages & anObj)
{
    BinaryDumpInFile(aFp,anObj.Im1().IsInit());
    if (anObj.Im1().IsInit()) BinaryDumpInFile(aFp,anObj.Im1().Val());
    BinaryDumpInFile(aFp,anObj.Im2().IsInit());
    if (anObj.Im2().IsInit()) BinaryDumpInFile(aFp,anObj.Im2().Val());
    BinaryDumpInFile(aFp,anObj.FCND_CalcIm2fromIm1().IsInit());
    if (anObj.FCND_CalcIm2fromIm1().IsInit()) BinaryDumpInFile(aFp,anObj.FCND_CalcIm2fromIm1().Val());
    BinaryDumpInFile(aFp,(int)anObj.ImPat().size());
    for(  std::list< std::string >::const_iterator iT=anObj.ImPat().begin();
         iT!=anObj.ImPat().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.ImageSecByCAWSI().IsInit());
    if (anObj.ImageSecByCAWSI().IsInit()) BinaryDumpInFile(aFp,anObj.ImageSecByCAWSI().Val());
    BinaryDumpInFile(aFp,anObj.ImSecCalcApero().IsInit());
    if (anObj.ImSecCalcApero().IsInit()) BinaryDumpInFile(aFp,anObj.ImSecCalcApero().Val());
    BinaryDumpInFile(aFp,anObj.RelGlobSelecteur().IsInit());
    if (anObj.RelGlobSelecteur().IsInit()) BinaryDumpInFile(aFp,anObj.RelGlobSelecteur().Val());
    BinaryDumpInFile(aFp,anObj.Filter().IsInit());
    if (anObj.Filter().IsInit()) BinaryDumpInFile(aFp,anObj.Filter().Val());
    BinaryDumpInFile(aFp,anObj.AutoSelectionneImSec().IsInit());
    if (anObj.AutoSelectionneImSec().IsInit()) BinaryDumpInFile(aFp,anObj.AutoSelectionneImSec().Val());
    BinaryDumpInFile(aFp,anObj.ImSecByDelta().IsInit());
    if (anObj.ImSecByDelta().IsInit()) BinaryDumpInFile(aFp,anObj.ImSecByDelta().Val());
    BinaryDumpInFile(aFp,anObj.Im3Superp().IsInit());
    if (anObj.Im3Superp().IsInit()) BinaryDumpInFile(aFp,anObj.Im3Superp().Val());
}

cElXMLTree * ToXMLTree(const cImages & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"Images",eXMLBranche);
   if (anObj.Im1().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Im1"),anObj.Im1().Val())->ReTagThis("Im1"));
   if (anObj.Im2().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Im2"),anObj.Im2().Val())->ReTagThis("Im2"));
   if (anObj.FCND_CalcIm2fromIm1().IsInit())
      aRes->AddFils(ToXMLTree(anObj.FCND_CalcIm2fromIm1().Val())->ReTagThis("FCND_CalcIm2fromIm1"));
  for
  (       std::list< std::string >::const_iterator it=anObj.ImPat().begin();
      it !=anObj.ImPat().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("ImPat"),(*it))->ReTagThis("ImPat"));
   if (anObj.ImageSecByCAWSI().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ImageSecByCAWSI"),anObj.ImageSecByCAWSI().Val())->ReTagThis("ImageSecByCAWSI"));
   if (anObj.ImSecCalcApero().IsInit())
      aRes->AddFils(ToXMLTree(anObj.ImSecCalcApero().Val())->ReTagThis("ImSecCalcApero"));
   if (anObj.RelGlobSelecteur().IsInit())
      aRes->AddFils(ToXMLTree(anObj.RelGlobSelecteur().Val())->ReTagThis("RelGlobSelecteur"));
   if (anObj.Filter().IsInit())
      aRes->AddFils(ToXMLTree(anObj.Filter().Val())->ReTagThis("Filter"));
   if (anObj.AutoSelectionneImSec().IsInit())
      aRes->AddFils(ToXMLTree(anObj.AutoSelectionneImSec().Val())->ReTagThis("AutoSelectionneImSec"));
   if (anObj.ImSecByDelta().IsInit())
      aRes->AddFils(ToXMLTree(anObj.ImSecByDelta().Val())->ReTagThis("ImSecByDelta"));
   if (anObj.Im3Superp().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Im3Superp"),anObj.Im3Superp().Val())->ReTagThis("Im3Superp"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cImages & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.Im1(),aTree->Get("Im1",1)); //tototo 

   xml_init(anObj.Im2(),aTree->Get("Im2",1)); //tototo 

   xml_init(anObj.FCND_CalcIm2fromIm1(),aTree->Get("FCND_CalcIm2fromIm1",1)); //tototo 

   xml_init(anObj.ImPat(),aTree->GetAll("ImPat",false,1));

   xml_init(anObj.ImageSecByCAWSI(),aTree->Get("ImageSecByCAWSI",1)); //tototo 

   xml_init(anObj.ImSecCalcApero(),aTree->Get("ImSecCalcApero",1)); //tototo 

   xml_init(anObj.RelGlobSelecteur(),aTree->Get("RelGlobSelecteur",1)); //tototo 

   xml_init(anObj.Filter(),aTree->Get("Filter",1)); //tototo 

   xml_init(anObj.AutoSelectionneImSec(),aTree->Get("AutoSelectionneImSec",1)); //tototo 

   xml_init(anObj.ImSecByDelta(),aTree->Get("ImSecByDelta",1)); //tototo 

   xml_init(anObj.Im3Superp(),aTree->Get("Im3Superp",1)); //tototo 
}

std::string  Mangling( cImages *) {return "F57B9FC8D5C28BD2FE3F";};


std::string & cFCND_Mode_GeomIm::FCND_GeomCalc()
{
   return mFCND_GeomCalc;
}

const std::string & cFCND_Mode_GeomIm::FCND_GeomCalc()const 
{
   return mFCND_GeomCalc;
}


cTplValGesInit< cElRegex_Ptr > & cFCND_Mode_GeomIm::FCND_GeomApply()
{
   return mFCND_GeomApply;
}

const cTplValGesInit< cElRegex_Ptr > & cFCND_Mode_GeomIm::FCND_GeomApply()const 
{
   return mFCND_GeomApply;
}

void  BinaryUnDumpFromFile(cFCND_Mode_GeomIm & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.FCND_GeomCalc(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.FCND_GeomApply().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.FCND_GeomApply().ValForcedForUnUmp(),aFp);
        }
        else  anObj.FCND_GeomApply().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cFCND_Mode_GeomIm & anObj)
{
    BinaryDumpInFile(aFp,anObj.FCND_GeomCalc());
    BinaryDumpInFile(aFp,anObj.FCND_GeomApply().IsInit());
    if (anObj.FCND_GeomApply().IsInit()) BinaryDumpInFile(aFp,anObj.FCND_GeomApply().Val());
}

cElXMLTree * ToXMLTree(const cFCND_Mode_GeomIm & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"FCND_Mode_GeomIm",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("FCND_GeomCalc"),anObj.FCND_GeomCalc())->ReTagThis("FCND_GeomCalc"));
   if (anObj.FCND_GeomApply().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("FCND_GeomApply"),anObj.FCND_GeomApply().Val())->ReTagThis("FCND_GeomApply"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cFCND_Mode_GeomIm & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.FCND_GeomCalc(),aTree->Get("FCND_GeomCalc",1)); //tototo 

   xml_init(anObj.FCND_GeomApply(),aTree->Get("FCND_GeomApply",1)); //tototo 
}

std::string  Mangling( cFCND_Mode_GeomIm *) {return "FC2AE84047D79BC0FD3F";};


std::string & cNGI_StdDir::StdDir()
{
   return mStdDir;
}

const std::string & cNGI_StdDir::StdDir()const 
{
   return mStdDir;
}


cTplValGesInit< cElRegex_Ptr > & cNGI_StdDir::NGI_StdDir_Apply()
{
   return mNGI_StdDir_Apply;
}

const cTplValGesInit< cElRegex_Ptr > & cNGI_StdDir::NGI_StdDir_Apply()const 
{
   return mNGI_StdDir_Apply;
}

void  BinaryUnDumpFromFile(cNGI_StdDir & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.StdDir(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NGI_StdDir_Apply().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NGI_StdDir_Apply().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NGI_StdDir_Apply().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cNGI_StdDir & anObj)
{
    BinaryDumpInFile(aFp,anObj.StdDir());
    BinaryDumpInFile(aFp,anObj.NGI_StdDir_Apply().IsInit());
    if (anObj.NGI_StdDir_Apply().IsInit()) BinaryDumpInFile(aFp,anObj.NGI_StdDir_Apply().Val());
}

cElXMLTree * ToXMLTree(const cNGI_StdDir & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"NGI_StdDir",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("StdDir"),anObj.StdDir())->ReTagThis("StdDir"));
   if (anObj.NGI_StdDir_Apply().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NGI_StdDir_Apply"),anObj.NGI_StdDir_Apply().Val())->ReTagThis("NGI_StdDir_Apply"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cNGI_StdDir & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.StdDir(),aTree->Get("StdDir",1)); //tototo 

   xml_init(anObj.NGI_StdDir_Apply(),aTree->Get("NGI_StdDir_Apply",1)); //tototo 
}

std::string  Mangling( cNGI_StdDir *) {return "DD8BCA004654DDF1FE3F";};


std::string & cModuleImageLoader::NomModule()
{
   return mNomModule;
}

const std::string & cModuleImageLoader::NomModule()const 
{
   return mNomModule;
}


std::string & cModuleImageLoader::NomLoader()
{
   return mNomLoader;
}

const std::string & cModuleImageLoader::NomLoader()const 
{
   return mNomLoader;
}

void  BinaryUnDumpFromFile(cModuleImageLoader & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.NomModule(),aFp);
    BinaryUnDumpFromFile(anObj.NomLoader(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cModuleImageLoader & anObj)
{
    BinaryDumpInFile(aFp,anObj.NomModule());
    BinaryDumpInFile(aFp,anObj.NomLoader());
}

cElXMLTree * ToXMLTree(const cModuleImageLoader & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ModuleImageLoader",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("NomModule"),anObj.NomModule())->ReTagThis("NomModule"));
   aRes->AddFils(::ToXMLTree(std::string("NomLoader"),anObj.NomLoader())->ReTagThis("NomLoader"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cModuleImageLoader & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.NomModule(),aTree->Get("NomModule",1)); //tototo 

   xml_init(anObj.NomLoader(),aTree->Get("NomLoader",1)); //tototo 
}

std::string  Mangling( cModuleImageLoader *) {return "422AF23EBFEDBCBAFE3F";};


cTplValGesInit< double > & cCropAndScale::Scale()
{
   return mScale;
}

const cTplValGesInit< double > & cCropAndScale::Scale()const 
{
   return mScale;
}


cTplValGesInit< Pt2dr > & cCropAndScale::Crop()
{
   return mCrop;
}

const cTplValGesInit< Pt2dr > & cCropAndScale::Crop()const 
{
   return mCrop;
}


cTplValGesInit< double > & cCropAndScale::ScaleY()
{
   return mScaleY;
}

const cTplValGesInit< double > & cCropAndScale::ScaleY()const 
{
   return mScaleY;
}

void  BinaryUnDumpFromFile(cCropAndScale & anObj,ELISE_fp & aFp)
{
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
             anObj.Crop().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Crop().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Crop().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ScaleY().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ScaleY().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ScaleY().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cCropAndScale & anObj)
{
    BinaryDumpInFile(aFp,anObj.Scale().IsInit());
    if (anObj.Scale().IsInit()) BinaryDumpInFile(aFp,anObj.Scale().Val());
    BinaryDumpInFile(aFp,anObj.Crop().IsInit());
    if (anObj.Crop().IsInit()) BinaryDumpInFile(aFp,anObj.Crop().Val());
    BinaryDumpInFile(aFp,anObj.ScaleY().IsInit());
    if (anObj.ScaleY().IsInit()) BinaryDumpInFile(aFp,anObj.ScaleY().Val());
}

cElXMLTree * ToXMLTree(const cCropAndScale & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"CropAndScale",eXMLBranche);
   if (anObj.Scale().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Scale"),anObj.Scale().Val())->ReTagThis("Scale"));
   if (anObj.Crop().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Crop"),anObj.Crop().Val())->ReTagThis("Crop"));
   if (anObj.ScaleY().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ScaleY"),anObj.ScaleY().Val())->ReTagThis("ScaleY"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cCropAndScale & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.Scale(),aTree->Get("Scale",1),double(1.0)); //tototo 

   xml_init(anObj.Crop(),aTree->Get("Crop",1),Pt2dr(0.0 , 0.0)); //tototo 

   xml_init(anObj.ScaleY(),aTree->Get("ScaleY",1)); //tototo 
}

std::string  Mangling( cCropAndScale *) {return "ECB07A64EBAB2CC5FF3F";};


cTplValGesInit< double > & cGeom::Scale()
{
   return CropAndScale().Val().Scale();
}

const cTplValGesInit< double > & cGeom::Scale()const 
{
   return CropAndScale().Val().Scale();
}


cTplValGesInit< Pt2dr > & cGeom::Crop()
{
   return CropAndScale().Val().Crop();
}

const cTplValGesInit< Pt2dr > & cGeom::Crop()const 
{
   return CropAndScale().Val().Crop();
}


cTplValGesInit< double > & cGeom::ScaleY()
{
   return CropAndScale().Val().ScaleY();
}

const cTplValGesInit< double > & cGeom::ScaleY()const 
{
   return CropAndScale().Val().ScaleY();
}


cTplValGesInit< cCropAndScale > & cGeom::CropAndScale()
{
   return mCropAndScale;
}

const cTplValGesInit< cCropAndScale > & cGeom::CropAndScale()const 
{
   return mCropAndScale;
}


cTplValGesInit< std::string > & cGeom::NamePxTr()
{
   return mNamePxTr;
}

const cTplValGesInit< std::string > & cGeom::NamePxTr()const 
{
   return mNamePxTr;
}

void  BinaryUnDumpFromFile(cGeom & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.CropAndScale().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.CropAndScale().ValForcedForUnUmp(),aFp);
        }
        else  anObj.CropAndScale().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NamePxTr().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NamePxTr().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NamePxTr().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cGeom & anObj)
{
    BinaryDumpInFile(aFp,anObj.CropAndScale().IsInit());
    if (anObj.CropAndScale().IsInit()) BinaryDumpInFile(aFp,anObj.CropAndScale().Val());
    BinaryDumpInFile(aFp,anObj.NamePxTr().IsInit());
    if (anObj.NamePxTr().IsInit()) BinaryDumpInFile(aFp,anObj.NamePxTr().Val());
}

cElXMLTree * ToXMLTree(const cGeom & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"Geom",eXMLBranche);
   if (anObj.CropAndScale().IsInit())
      aRes->AddFils(ToXMLTree(anObj.CropAndScale().Val())->ReTagThis("CropAndScale"));
   if (anObj.NamePxTr().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NamePxTr"),anObj.NamePxTr().Val())->ReTagThis("NamePxTr"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cGeom & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.CropAndScale(),aTree->Get("CropAndScale",1)); //tototo 

   xml_init(anObj.NamePxTr(),aTree->Get("NamePxTr",1)); //tototo 
}

std::string  Mangling( cGeom *) {return "8EF44028264CBF82FF3F";};


cTplValGesInit< double > & cModifieurGeometrie::Scale()
{
   return Geom().CropAndScale().Val().Scale();
}

const cTplValGesInit< double > & cModifieurGeometrie::Scale()const 
{
   return Geom().CropAndScale().Val().Scale();
}


cTplValGesInit< Pt2dr > & cModifieurGeometrie::Crop()
{
   return Geom().CropAndScale().Val().Crop();
}

const cTplValGesInit< Pt2dr > & cModifieurGeometrie::Crop()const 
{
   return Geom().CropAndScale().Val().Crop();
}


cTplValGesInit< double > & cModifieurGeometrie::ScaleY()
{
   return Geom().CropAndScale().Val().ScaleY();
}

const cTplValGesInit< double > & cModifieurGeometrie::ScaleY()const 
{
   return Geom().CropAndScale().Val().ScaleY();
}


cTplValGesInit< cCropAndScale > & cModifieurGeometrie::CropAndScale()
{
   return Geom().CropAndScale();
}

const cTplValGesInit< cCropAndScale > & cModifieurGeometrie::CropAndScale()const 
{
   return Geom().CropAndScale();
}


cTplValGesInit< std::string > & cModifieurGeometrie::NamePxTr()
{
   return Geom().NamePxTr();
}

const cTplValGesInit< std::string > & cModifieurGeometrie::NamePxTr()const 
{
   return Geom().NamePxTr();
}


cGeom & cModifieurGeometrie::Geom()
{
   return mGeom;
}

const cGeom & cModifieurGeometrie::Geom()const 
{
   return mGeom;
}


cTplValGesInit< cElRegex_Ptr > & cModifieurGeometrie::Apply()
{
   return mApply;
}

const cTplValGesInit< cElRegex_Ptr > & cModifieurGeometrie::Apply()const 
{
   return mApply;
}

void  BinaryUnDumpFromFile(cModifieurGeometrie & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Geom(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Apply().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Apply().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Apply().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cModifieurGeometrie & anObj)
{
    BinaryDumpInFile(aFp,anObj.Geom());
    BinaryDumpInFile(aFp,anObj.Apply().IsInit());
    if (anObj.Apply().IsInit()) BinaryDumpInFile(aFp,anObj.Apply().Val());
}

cElXMLTree * ToXMLTree(const cModifieurGeometrie & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ModifieurGeometrie",eXMLBranche);
   aRes->AddFils(ToXMLTree(anObj.Geom())->ReTagThis("Geom"));
   if (anObj.Apply().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Apply"),anObj.Apply().Val())->ReTagThis("Apply"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cModifieurGeometrie & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.Geom(),aTree->Get("Geom",1)); //tototo 

   xml_init(anObj.Apply(),aTree->Get("Apply",1)); //tototo 
}

std::string  Mangling( cModifieurGeometrie *) {return "557D2ABE454D8EE8FE3F";};


cTplValGesInit< bool > & cNomsGeometrieImage::UseIt()
{
   return mUseIt;
}

const cTplValGesInit< bool > & cNomsGeometrieImage::UseIt()const 
{
   return mUseIt;
}


cTplValGesInit< std::string > & cNomsGeometrieImage::PatternSel()
{
   return mPatternSel;
}

const cTplValGesInit< std::string > & cNomsGeometrieImage::PatternSel()const 
{
   return mPatternSel;
}


cTplValGesInit< std::string > & cNomsGeometrieImage::PatNameGeom()
{
   return mPatNameGeom;
}

const cTplValGesInit< std::string > & cNomsGeometrieImage::PatNameGeom()const 
{
   return mPatNameGeom;
}


cTplValGesInit< std::string > & cNomsGeometrieImage::PatternNameIm1Im2()
{
   return mPatternNameIm1Im2;
}

const cTplValGesInit< std::string > & cNomsGeometrieImage::PatternNameIm1Im2()const 
{
   return mPatternNameIm1Im2;
}


std::string & cNomsGeometrieImage::FCND_GeomCalc()
{
   return FCND_Mode_GeomIm().Val().FCND_GeomCalc();
}

const std::string & cNomsGeometrieImage::FCND_GeomCalc()const 
{
   return FCND_Mode_GeomIm().Val().FCND_GeomCalc();
}


cTplValGesInit< cElRegex_Ptr > & cNomsGeometrieImage::FCND_GeomApply()
{
   return FCND_Mode_GeomIm().Val().FCND_GeomApply();
}

const cTplValGesInit< cElRegex_Ptr > & cNomsGeometrieImage::FCND_GeomApply()const 
{
   return FCND_Mode_GeomIm().Val().FCND_GeomApply();
}


cTplValGesInit< cFCND_Mode_GeomIm > & cNomsGeometrieImage::FCND_Mode_GeomIm()
{
   return mFCND_Mode_GeomIm;
}

const cTplValGesInit< cFCND_Mode_GeomIm > & cNomsGeometrieImage::FCND_Mode_GeomIm()const 
{
   return mFCND_Mode_GeomIm;
}


std::string & cNomsGeometrieImage::StdDir()
{
   return NGI_StdDir().Val().StdDir();
}

const std::string & cNomsGeometrieImage::StdDir()const 
{
   return NGI_StdDir().Val().StdDir();
}


cTplValGesInit< cElRegex_Ptr > & cNomsGeometrieImage::NGI_StdDir_Apply()
{
   return NGI_StdDir().Val().NGI_StdDir_Apply();
}

const cTplValGesInit< cElRegex_Ptr > & cNomsGeometrieImage::NGI_StdDir_Apply()const 
{
   return NGI_StdDir().Val().NGI_StdDir_Apply();
}


cTplValGesInit< cNGI_StdDir > & cNomsGeometrieImage::NGI_StdDir()
{
   return mNGI_StdDir;
}

const cTplValGesInit< cNGI_StdDir > & cNomsGeometrieImage::NGI_StdDir()const 
{
   return mNGI_StdDir;
}


cTplValGesInit< bool > & cNomsGeometrieImage::AddNumToNameGeom()
{
   return mAddNumToNameGeom;
}

const cTplValGesInit< bool > & cNomsGeometrieImage::AddNumToNameGeom()const 
{
   return mAddNumToNameGeom;
}


std::string & cNomsGeometrieImage::NomModule()
{
   return ModuleImageLoader().Val().NomModule();
}

const std::string & cNomsGeometrieImage::NomModule()const 
{
   return ModuleImageLoader().Val().NomModule();
}


std::string & cNomsGeometrieImage::NomLoader()
{
   return ModuleImageLoader().Val().NomLoader();
}

const std::string & cNomsGeometrieImage::NomLoader()const 
{
   return ModuleImageLoader().Val().NomLoader();
}


cTplValGesInit< cModuleImageLoader > & cNomsGeometrieImage::ModuleImageLoader()
{
   return mModuleImageLoader;
}

const cTplValGesInit< cModuleImageLoader > & cNomsGeometrieImage::ModuleImageLoader()const 
{
   return mModuleImageLoader;
}


std::list< int > & cNomsGeometrieImage::GenereOriDeZoom()
{
   return mGenereOriDeZoom;
}

const std::list< int > & cNomsGeometrieImage::GenereOriDeZoom()const 
{
   return mGenereOriDeZoom;
}


std::list< cModifieurGeometrie > & cNomsGeometrieImage::ModifieurGeometrie()
{
   return mModifieurGeometrie;
}

const std::list< cModifieurGeometrie > & cNomsGeometrieImage::ModifieurGeometrie()const 
{
   return mModifieurGeometrie;
}

void  BinaryUnDumpFromFile(cNomsGeometrieImage & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.UseIt().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.UseIt().ValForcedForUnUmp(),aFp);
        }
        else  anObj.UseIt().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.PatternSel().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.PatternSel().ValForcedForUnUmp(),aFp);
        }
        else  anObj.PatternSel().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.PatNameGeom().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.PatNameGeom().ValForcedForUnUmp(),aFp);
        }
        else  anObj.PatNameGeom().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.PatternNameIm1Im2().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.PatternNameIm1Im2().ValForcedForUnUmp(),aFp);
        }
        else  anObj.PatternNameIm1Im2().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.FCND_Mode_GeomIm().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.FCND_Mode_GeomIm().ValForcedForUnUmp(),aFp);
        }
        else  anObj.FCND_Mode_GeomIm().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NGI_StdDir().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NGI_StdDir().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NGI_StdDir().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.AddNumToNameGeom().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.AddNumToNameGeom().ValForcedForUnUmp(),aFp);
        }
        else  anObj.AddNumToNameGeom().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ModuleImageLoader().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ModuleImageLoader().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ModuleImageLoader().SetNoInit();
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             int aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.GenereOriDeZoom().push_back(aVal);
        }
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cModifieurGeometrie aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.ModifieurGeometrie().push_back(aVal);
        }
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cNomsGeometrieImage & anObj)
{
    BinaryDumpInFile(aFp,anObj.UseIt().IsInit());
    if (anObj.UseIt().IsInit()) BinaryDumpInFile(aFp,anObj.UseIt().Val());
    BinaryDumpInFile(aFp,anObj.PatternSel().IsInit());
    if (anObj.PatternSel().IsInit()) BinaryDumpInFile(aFp,anObj.PatternSel().Val());
    BinaryDumpInFile(aFp,anObj.PatNameGeom().IsInit());
    if (anObj.PatNameGeom().IsInit()) BinaryDumpInFile(aFp,anObj.PatNameGeom().Val());
    BinaryDumpInFile(aFp,anObj.PatternNameIm1Im2().IsInit());
    if (anObj.PatternNameIm1Im2().IsInit()) BinaryDumpInFile(aFp,anObj.PatternNameIm1Im2().Val());
    BinaryDumpInFile(aFp,anObj.FCND_Mode_GeomIm().IsInit());
    if (anObj.FCND_Mode_GeomIm().IsInit()) BinaryDumpInFile(aFp,anObj.FCND_Mode_GeomIm().Val());
    BinaryDumpInFile(aFp,anObj.NGI_StdDir().IsInit());
    if (anObj.NGI_StdDir().IsInit()) BinaryDumpInFile(aFp,anObj.NGI_StdDir().Val());
    BinaryDumpInFile(aFp,anObj.AddNumToNameGeom().IsInit());
    if (anObj.AddNumToNameGeom().IsInit()) BinaryDumpInFile(aFp,anObj.AddNumToNameGeom().Val());
    BinaryDumpInFile(aFp,anObj.ModuleImageLoader().IsInit());
    if (anObj.ModuleImageLoader().IsInit()) BinaryDumpInFile(aFp,anObj.ModuleImageLoader().Val());
    BinaryDumpInFile(aFp,(int)anObj.GenereOriDeZoom().size());
    for(  std::list< int >::const_iterator iT=anObj.GenereOriDeZoom().begin();
         iT!=anObj.GenereOriDeZoom().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,(int)anObj.ModifieurGeometrie().size());
    for(  std::list< cModifieurGeometrie >::const_iterator iT=anObj.ModifieurGeometrie().begin();
         iT!=anObj.ModifieurGeometrie().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
}

cElXMLTree * ToXMLTree(const cNomsGeometrieImage & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"NomsGeometrieImage",eXMLBranche);
   if (anObj.UseIt().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("UseIt"),anObj.UseIt().Val())->ReTagThis("UseIt"));
   if (anObj.PatternSel().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PatternSel"),anObj.PatternSel().Val())->ReTagThis("PatternSel"));
   if (anObj.PatNameGeom().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PatNameGeom"),anObj.PatNameGeom().Val())->ReTagThis("PatNameGeom"));
   if (anObj.PatternNameIm1Im2().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PatternNameIm1Im2"),anObj.PatternNameIm1Im2().Val())->ReTagThis("PatternNameIm1Im2"));
   if (anObj.FCND_Mode_GeomIm().IsInit())
      aRes->AddFils(ToXMLTree(anObj.FCND_Mode_GeomIm().Val())->ReTagThis("FCND_Mode_GeomIm"));
   if (anObj.NGI_StdDir().IsInit())
      aRes->AddFils(ToXMLTree(anObj.NGI_StdDir().Val())->ReTagThis("NGI_StdDir"));
   if (anObj.AddNumToNameGeom().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("AddNumToNameGeom"),anObj.AddNumToNameGeom().Val())->ReTagThis("AddNumToNameGeom"));
   if (anObj.ModuleImageLoader().IsInit())
      aRes->AddFils(ToXMLTree(anObj.ModuleImageLoader().Val())->ReTagThis("ModuleImageLoader"));
  for
  (       std::list< int >::const_iterator it=anObj.GenereOriDeZoom().begin();
      it !=anObj.GenereOriDeZoom().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("GenereOriDeZoom"),(*it))->ReTagThis("GenereOriDeZoom"));
  for
  (       std::list< cModifieurGeometrie >::const_iterator it=anObj.ModifieurGeometrie().begin();
      it !=anObj.ModifieurGeometrie().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("ModifieurGeometrie"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cNomsGeometrieImage & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.UseIt(),aTree->Get("UseIt",1),bool(true)); //tototo 

   xml_init(anObj.PatternSel(),aTree->Get("PatternSel",1)); //tototo 

   xml_init(anObj.PatNameGeom(),aTree->Get("PatNameGeom",1)); //tototo 

   xml_init(anObj.PatternNameIm1Im2(),aTree->Get("PatternNameIm1Im2",1)); //tototo 

   xml_init(anObj.FCND_Mode_GeomIm(),aTree->Get("FCND_Mode_GeomIm",1)); //tototo 

   xml_init(anObj.NGI_StdDir(),aTree->Get("NGI_StdDir",1)); //tototo 

   xml_init(anObj.AddNumToNameGeom(),aTree->Get("AddNumToNameGeom",1),bool(false)); //tototo 

   xml_init(anObj.ModuleImageLoader(),aTree->Get("ModuleImageLoader",1)); //tototo 

   xml_init(anObj.GenereOriDeZoom(),aTree->GetAll("GenereOriDeZoom",false,1));

   xml_init(anObj.ModifieurGeometrie(),aTree->GetAll("ModifieurGeometrie",false,1));
}

std::string  Mangling( cNomsGeometrieImage *) {return "F00AEA2711B5B9B9FF3F";};


std::string & cNomsHomomologues::PatternSel()
{
   return mPatternSel;
}

const std::string & cNomsHomomologues::PatternSel()const 
{
   return mPatternSel;
}


std::string & cNomsHomomologues::PatNameGeom()
{
   return mPatNameGeom;
}

const std::string & cNomsHomomologues::PatNameGeom()const 
{
   return mPatNameGeom;
}


cTplValGesInit< std::string > & cNomsHomomologues::SeparateurHom()
{
   return mSeparateurHom;
}

const cTplValGesInit< std::string > & cNomsHomomologues::SeparateurHom()const 
{
   return mSeparateurHom;
}

void  BinaryUnDumpFromFile(cNomsHomomologues & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.PatternSel(),aFp);
    BinaryUnDumpFromFile(anObj.PatNameGeom(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SeparateurHom().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SeparateurHom().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SeparateurHom().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cNomsHomomologues & anObj)
{
    BinaryDumpInFile(aFp,anObj.PatternSel());
    BinaryDumpInFile(aFp,anObj.PatNameGeom());
    BinaryDumpInFile(aFp,anObj.SeparateurHom().IsInit());
    if (anObj.SeparateurHom().IsInit()) BinaryDumpInFile(aFp,anObj.SeparateurHom().Val());
}

cElXMLTree * ToXMLTree(const cNomsHomomologues & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"NomsHomomologues",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("PatternSel"),anObj.PatternSel())->ReTagThis("PatternSel"));
   aRes->AddFils(::ToXMLTree(std::string("PatNameGeom"),anObj.PatNameGeom())->ReTagThis("PatNameGeom"));
   if (anObj.SeparateurHom().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SeparateurHom"),anObj.SeparateurHom().Val())->ReTagThis("SeparateurHom"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cNomsHomomologues & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.PatternSel(),aTree->Get("PatternSel",1)); //tototo 

   xml_init(anObj.PatNameGeom(),aTree->Get("PatNameGeom",1)); //tototo 

   xml_init(anObj.SeparateurHom(),aTree->Get("SeparateurHom",1),std::string("")); //tototo 
}

std::string  Mangling( cNomsHomomologues *) {return "4E61DDA221FF08F2FE3F";};


cTplValGesInit< int > & cSection_PriseDeVue::BordImage()
{
   return mBordImage;
}

const cTplValGesInit< int > & cSection_PriseDeVue::BordImage()const 
{
   return mBordImage;
}


cTplValGesInit< bool > & cSection_PriseDeVue::ConvertToSameOriPtTgtLoc()
{
   return mConvertToSameOriPtTgtLoc;
}

const cTplValGesInit< bool > & cSection_PriseDeVue::ConvertToSameOriPtTgtLoc()const 
{
   return mConvertToSameOriPtTgtLoc;
}


cTplValGesInit< int > & cSection_PriseDeVue::ValSpecNotImage()
{
   return mValSpecNotImage;
}

const cTplValGesInit< int > & cSection_PriseDeVue::ValSpecNotImage()const 
{
   return mValSpecNotImage;
}


cTplValGesInit< std::string > & cSection_PriseDeVue::PrefixMasqImRes()
{
   return mPrefixMasqImRes;
}

const cTplValGesInit< std::string > & cSection_PriseDeVue::PrefixMasqImRes()const 
{
   return mPrefixMasqImRes;
}


cTplValGesInit< std::string > & cSection_PriseDeVue::DirMasqueImages()
{
   return mDirMasqueImages;
}

const cTplValGesInit< std::string > & cSection_PriseDeVue::DirMasqueImages()const 
{
   return mDirMasqueImages;
}


std::list< cMasqImageIn > & cSection_PriseDeVue::MasqImageIn()
{
   return mMasqImageIn;
}

const std::list< cMasqImageIn > & cSection_PriseDeVue::MasqImageIn()const 
{
   return mMasqImageIn;
}


std::list< cSpecFitrageImage > & cSection_PriseDeVue::FiltreImageIn()
{
   return mFiltreImageIn;
}

const std::list< cSpecFitrageImage > & cSection_PriseDeVue::FiltreImageIn()const 
{
   return mFiltreImageIn;
}


eModeGeomImage & cSection_PriseDeVue::GeomImages()
{
   return mGeomImages;
}

const eModeGeomImage & cSection_PriseDeVue::GeomImages()const 
{
   return mGeomImages;
}


std::string & cSection_PriseDeVue::NomModule()
{
   return ModuleGeomImage().Val().NomModule();
}

const std::string & cSection_PriseDeVue::NomModule()const 
{
   return ModuleGeomImage().Val().NomModule();
}


std::string & cSection_PriseDeVue::NomGeometrie()
{
   return ModuleGeomImage().Val().NomGeometrie();
}

const std::string & cSection_PriseDeVue::NomGeometrie()const 
{
   return ModuleGeomImage().Val().NomGeometrie();
}


cTplValGesInit< cModuleGeomImage > & cSection_PriseDeVue::ModuleGeomImage()
{
   return mModuleGeomImage;
}

const cTplValGesInit< cModuleGeomImage > & cSection_PriseDeVue::ModuleGeomImage()const 
{
   return mModuleGeomImage;
}


cTplValGesInit< std::string > & cSection_PriseDeVue::Im1()
{
   return Images().Im1();
}

const cTplValGesInit< std::string > & cSection_PriseDeVue::Im1()const 
{
   return Images().Im1();
}


cTplValGesInit< std::string > & cSection_PriseDeVue::Im2()
{
   return Images().Im2();
}

const cTplValGesInit< std::string > & cSection_PriseDeVue::Im2()const 
{
   return Images().Im2();
}


std::string & cSection_PriseDeVue::I2FromI1Key()
{
   return Images().FCND_CalcIm2fromIm1().Val().I2FromI1Key();
}

const std::string & cSection_PriseDeVue::I2FromI1Key()const 
{
   return Images().FCND_CalcIm2fromIm1().Val().I2FromI1Key();
}


bool & cSection_PriseDeVue::I2FromI1SensDirect()
{
   return Images().FCND_CalcIm2fromIm1().Val().I2FromI1SensDirect();
}

const bool & cSection_PriseDeVue::I2FromI1SensDirect()const 
{
   return Images().FCND_CalcIm2fromIm1().Val().I2FromI1SensDirect();
}


cTplValGesInit< cFCND_CalcIm2fromIm1 > & cSection_PriseDeVue::FCND_CalcIm2fromIm1()
{
   return Images().FCND_CalcIm2fromIm1();
}

const cTplValGesInit< cFCND_CalcIm2fromIm1 > & cSection_PriseDeVue::FCND_CalcIm2fromIm1()const 
{
   return Images().FCND_CalcIm2fromIm1();
}


std::list< std::string > & cSection_PriseDeVue::ImPat()
{
   return Images().ImPat();
}

const std::list< std::string > & cSection_PriseDeVue::ImPat()const 
{
   return Images().ImPat();
}


cTplValGesInit< std::string > & cSection_PriseDeVue::ImageSecByCAWSI()
{
   return Images().ImageSecByCAWSI();
}

const cTplValGesInit< std::string > & cSection_PriseDeVue::ImageSecByCAWSI()const 
{
   return Images().ImageSecByCAWSI();
}


std::string & cSection_PriseDeVue::Key()
{
   return Images().ImSecCalcApero().Val().Key();
}

const std::string & cSection_PriseDeVue::Key()const 
{
   return Images().ImSecCalcApero().Val().Key();
}


cTplValGesInit< int > & cSection_PriseDeVue::Nb()
{
   return Images().ImSecCalcApero().Val().Nb();
}

const cTplValGesInit< int > & cSection_PriseDeVue::Nb()const 
{
   return Images().ImSecCalcApero().Val().Nb();
}


cTplValGesInit< int > & cSection_PriseDeVue::NbMin()
{
   return Images().ImSecCalcApero().Val().NbMin();
}

const cTplValGesInit< int > & cSection_PriseDeVue::NbMin()const 
{
   return Images().ImSecCalcApero().Val().NbMin();
}


cTplValGesInit< int > & cSection_PriseDeVue::NbMax()
{
   return Images().ImSecCalcApero().Val().NbMax();
}

const cTplValGesInit< int > & cSection_PriseDeVue::NbMax()const 
{
   return Images().ImSecCalcApero().Val().NbMax();
}


cTplValGesInit< eOnEmptyImSecApero > & cSection_PriseDeVue::OnEmpty()
{
   return Images().ImSecCalcApero().Val().OnEmpty();
}

const cTplValGesInit< eOnEmptyImSecApero > & cSection_PriseDeVue::OnEmpty()const 
{
   return Images().ImSecCalcApero().Val().OnEmpty();
}


cTplValGesInit< cImSecCalcApero > & cSection_PriseDeVue::ImSecCalcApero()
{
   return Images().ImSecCalcApero();
}

const cTplValGesInit< cImSecCalcApero > & cSection_PriseDeVue::ImSecCalcApero()const 
{
   return Images().ImSecCalcApero();
}


cTplValGesInit< cParamGenereStrVois > & cSection_PriseDeVue::RelGlobSelecteur()
{
   return Images().RelGlobSelecteur();
}

const cTplValGesInit< cParamGenereStrVois > & cSection_PriseDeVue::RelGlobSelecteur()const 
{
   return Images().RelGlobSelecteur();
}


cTplValGesInit< cNameFilter > & cSection_PriseDeVue::Filter()
{
   return Images().Filter();
}

const cTplValGesInit< cNameFilter > & cSection_PriseDeVue::Filter()const 
{
   return Images().Filter();
}


double & cSection_PriseDeVue::RecouvrMin()
{
   return Images().AutoSelectionneImSec().Val().RecouvrMin();
}

const double & cSection_PriseDeVue::RecouvrMin()const 
{
   return Images().AutoSelectionneImSec().Val().RecouvrMin();
}


cTplValGesInit< cAutoSelectionneImSec > & cSection_PriseDeVue::AutoSelectionneImSec()
{
   return Images().AutoSelectionneImSec();
}

const cTplValGesInit< cAutoSelectionneImSec > & cSection_PriseDeVue::AutoSelectionneImSec()const 
{
   return Images().AutoSelectionneImSec();
}


cTplValGesInit< cListImByDelta > & cSection_PriseDeVue::ImSecByDelta()
{
   return Images().ImSecByDelta();
}

const cTplValGesInit< cListImByDelta > & cSection_PriseDeVue::ImSecByDelta()const 
{
   return Images().ImSecByDelta();
}


cTplValGesInit< std::string > & cSection_PriseDeVue::Im3Superp()
{
   return Images().Im3Superp();
}

const cTplValGesInit< std::string > & cSection_PriseDeVue::Im3Superp()const 
{
   return Images().Im3Superp();
}


cImages & cSection_PriseDeVue::Images()
{
   return mImages;
}

const cImages & cSection_PriseDeVue::Images()const 
{
   return mImages;
}


std::list< cNomsGeometrieImage > & cSection_PriseDeVue::NomsGeometrieImage()
{
   return mNomsGeometrieImage;
}

const std::list< cNomsGeometrieImage > & cSection_PriseDeVue::NomsGeometrieImage()const 
{
   return mNomsGeometrieImage;
}


std::string & cSection_PriseDeVue::PatternSel()
{
   return NomsHomomologues().Val().PatternSel();
}

const std::string & cSection_PriseDeVue::PatternSel()const 
{
   return NomsHomomologues().Val().PatternSel();
}


std::string & cSection_PriseDeVue::PatNameGeom()
{
   return NomsHomomologues().Val().PatNameGeom();
}

const std::string & cSection_PriseDeVue::PatNameGeom()const 
{
   return NomsHomomologues().Val().PatNameGeom();
}


cTplValGesInit< std::string > & cSection_PriseDeVue::SeparateurHom()
{
   return NomsHomomologues().Val().SeparateurHom();
}

const cTplValGesInit< std::string > & cSection_PriseDeVue::SeparateurHom()const 
{
   return NomsHomomologues().Val().SeparateurHom();
}


cTplValGesInit< cNomsHomomologues > & cSection_PriseDeVue::NomsHomomologues()
{
   return mNomsHomomologues;
}

const cTplValGesInit< cNomsHomomologues > & cSection_PriseDeVue::NomsHomomologues()const 
{
   return mNomsHomomologues;
}


cTplValGesInit< std::string > & cSection_PriseDeVue::FCND_CalcHomFromI1I2()
{
   return mFCND_CalcHomFromI1I2;
}

const cTplValGesInit< std::string > & cSection_PriseDeVue::FCND_CalcHomFromI1I2()const 
{
   return mFCND_CalcHomFromI1I2;
}


cTplValGesInit< bool > & cSection_PriseDeVue::SingulariteInCorresp_I1I2()
{
   return mSingulariteInCorresp_I1I2;
}

const cTplValGesInit< bool > & cSection_PriseDeVue::SingulariteInCorresp_I1I2()const 
{
   return mSingulariteInCorresp_I1I2;
}


cTplValGesInit< cMapName2Name > & cSection_PriseDeVue::ClassEquivalenceImage()
{
   return mClassEquivalenceImage;
}

const cTplValGesInit< cMapName2Name > & cSection_PriseDeVue::ClassEquivalenceImage()const 
{
   return mClassEquivalenceImage;
}

void  BinaryUnDumpFromFile(cSection_PriseDeVue & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.BordImage().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.BordImage().ValForcedForUnUmp(),aFp);
        }
        else  anObj.BordImage().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ConvertToSameOriPtTgtLoc().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ConvertToSameOriPtTgtLoc().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ConvertToSameOriPtTgtLoc().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ValSpecNotImage().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ValSpecNotImage().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ValSpecNotImage().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.PrefixMasqImRes().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.PrefixMasqImRes().ValForcedForUnUmp(),aFp);
        }
        else  anObj.PrefixMasqImRes().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.DirMasqueImages().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.DirMasqueImages().ValForcedForUnUmp(),aFp);
        }
        else  anObj.DirMasqueImages().SetNoInit();
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cMasqImageIn aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.MasqImageIn().push_back(aVal);
        }
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cSpecFitrageImage aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.FiltreImageIn().push_back(aVal);
        }
  } ;
    BinaryUnDumpFromFile(anObj.GeomImages(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ModuleGeomImage().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ModuleGeomImage().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ModuleGeomImage().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.Images(),aFp);
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cNomsGeometrieImage aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.NomsGeometrieImage().push_back(aVal);
        }
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NomsHomomologues().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NomsHomomologues().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NomsHomomologues().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.FCND_CalcHomFromI1I2().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.FCND_CalcHomFromI1I2().ValForcedForUnUmp(),aFp);
        }
        else  anObj.FCND_CalcHomFromI1I2().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SingulariteInCorresp_I1I2().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SingulariteInCorresp_I1I2().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SingulariteInCorresp_I1I2().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ClassEquivalenceImage().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ClassEquivalenceImage().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ClassEquivalenceImage().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cSection_PriseDeVue & anObj)
{
    BinaryDumpInFile(aFp,anObj.BordImage().IsInit());
    if (anObj.BordImage().IsInit()) BinaryDumpInFile(aFp,anObj.BordImage().Val());
    BinaryDumpInFile(aFp,anObj.ConvertToSameOriPtTgtLoc().IsInit());
    if (anObj.ConvertToSameOriPtTgtLoc().IsInit()) BinaryDumpInFile(aFp,anObj.ConvertToSameOriPtTgtLoc().Val());
    BinaryDumpInFile(aFp,anObj.ValSpecNotImage().IsInit());
    if (anObj.ValSpecNotImage().IsInit()) BinaryDumpInFile(aFp,anObj.ValSpecNotImage().Val());
    BinaryDumpInFile(aFp,anObj.PrefixMasqImRes().IsInit());
    if (anObj.PrefixMasqImRes().IsInit()) BinaryDumpInFile(aFp,anObj.PrefixMasqImRes().Val());
    BinaryDumpInFile(aFp,anObj.DirMasqueImages().IsInit());
    if (anObj.DirMasqueImages().IsInit()) BinaryDumpInFile(aFp,anObj.DirMasqueImages().Val());
    BinaryDumpInFile(aFp,(int)anObj.MasqImageIn().size());
    for(  std::list< cMasqImageIn >::const_iterator iT=anObj.MasqImageIn().begin();
         iT!=anObj.MasqImageIn().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,(int)anObj.FiltreImageIn().size());
    for(  std::list< cSpecFitrageImage >::const_iterator iT=anObj.FiltreImageIn().begin();
         iT!=anObj.FiltreImageIn().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.GeomImages());
    BinaryDumpInFile(aFp,anObj.ModuleGeomImage().IsInit());
    if (anObj.ModuleGeomImage().IsInit()) BinaryDumpInFile(aFp,anObj.ModuleGeomImage().Val());
    BinaryDumpInFile(aFp,anObj.Images());
    BinaryDumpInFile(aFp,(int)anObj.NomsGeometrieImage().size());
    for(  std::list< cNomsGeometrieImage >::const_iterator iT=anObj.NomsGeometrieImage().begin();
         iT!=anObj.NomsGeometrieImage().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.NomsHomomologues().IsInit());
    if (anObj.NomsHomomologues().IsInit()) BinaryDumpInFile(aFp,anObj.NomsHomomologues().Val());
    BinaryDumpInFile(aFp,anObj.FCND_CalcHomFromI1I2().IsInit());
    if (anObj.FCND_CalcHomFromI1I2().IsInit()) BinaryDumpInFile(aFp,anObj.FCND_CalcHomFromI1I2().Val());
    BinaryDumpInFile(aFp,anObj.SingulariteInCorresp_I1I2().IsInit());
    if (anObj.SingulariteInCorresp_I1I2().IsInit()) BinaryDumpInFile(aFp,anObj.SingulariteInCorresp_I1I2().Val());
    BinaryDumpInFile(aFp,anObj.ClassEquivalenceImage().IsInit());
    if (anObj.ClassEquivalenceImage().IsInit()) BinaryDumpInFile(aFp,anObj.ClassEquivalenceImage().Val());
}

cElXMLTree * ToXMLTree(const cSection_PriseDeVue & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"Section_PriseDeVue",eXMLBranche);
   if (anObj.BordImage().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("BordImage"),anObj.BordImage().Val())->ReTagThis("BordImage"));
   if (anObj.ConvertToSameOriPtTgtLoc().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ConvertToSameOriPtTgtLoc"),anObj.ConvertToSameOriPtTgtLoc().Val())->ReTagThis("ConvertToSameOriPtTgtLoc"));
   if (anObj.ValSpecNotImage().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ValSpecNotImage"),anObj.ValSpecNotImage().Val())->ReTagThis("ValSpecNotImage"));
   if (anObj.PrefixMasqImRes().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PrefixMasqImRes"),anObj.PrefixMasqImRes().Val())->ReTagThis("PrefixMasqImRes"));
   if (anObj.DirMasqueImages().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DirMasqueImages"),anObj.DirMasqueImages().Val())->ReTagThis("DirMasqueImages"));
  for
  (       std::list< cMasqImageIn >::const_iterator it=anObj.MasqImageIn().begin();
      it !=anObj.MasqImageIn().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("MasqImageIn"));
  for
  (       std::list< cSpecFitrageImage >::const_iterator it=anObj.FiltreImageIn().begin();
      it !=anObj.FiltreImageIn().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("FiltreImageIn"));
   aRes->AddFils(ToXMLTree(std::string("GeomImages"),anObj.GeomImages())->ReTagThis("GeomImages"));
   if (anObj.ModuleGeomImage().IsInit())
      aRes->AddFils(ToXMLTree(anObj.ModuleGeomImage().Val())->ReTagThis("ModuleGeomImage"));
   aRes->AddFils(ToXMLTree(anObj.Images())->ReTagThis("Images"));
  for
  (       std::list< cNomsGeometrieImage >::const_iterator it=anObj.NomsGeometrieImage().begin();
      it !=anObj.NomsGeometrieImage().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("NomsGeometrieImage"));
   if (anObj.NomsHomomologues().IsInit())
      aRes->AddFils(ToXMLTree(anObj.NomsHomomologues().Val())->ReTagThis("NomsHomomologues"));
   if (anObj.FCND_CalcHomFromI1I2().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("FCND_CalcHomFromI1I2"),anObj.FCND_CalcHomFromI1I2().Val())->ReTagThis("FCND_CalcHomFromI1I2"));
   if (anObj.SingulariteInCorresp_I1I2().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SingulariteInCorresp_I1I2"),anObj.SingulariteInCorresp_I1I2().Val())->ReTagThis("SingulariteInCorresp_I1I2"));
   if (anObj.ClassEquivalenceImage().IsInit())
      aRes->AddFils(ToXMLTree(anObj.ClassEquivalenceImage().Val())->ReTagThis("ClassEquivalenceImage"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cSection_PriseDeVue & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.BordImage(),aTree->Get("BordImage",1),int(5)); //tototo 

   xml_init(anObj.ConvertToSameOriPtTgtLoc(),aTree->Get("ConvertToSameOriPtTgtLoc",1)); //tototo 

   xml_init(anObj.ValSpecNotImage(),aTree->Get("ValSpecNotImage",1)); //tototo 

   xml_init(anObj.PrefixMasqImRes(),aTree->Get("PrefixMasqImRes",1),std::string("MasqIm")); //tototo 

   xml_init(anObj.DirMasqueImages(),aTree->Get("DirMasqueImages",1),std::string("")); //tototo 

   xml_init(anObj.MasqImageIn(),aTree->GetAll("MasqImageIn",false,1));

   xml_init(anObj.FiltreImageIn(),aTree->GetAll("FiltreImageIn",false,1));

   xml_init(anObj.GeomImages(),aTree->Get("GeomImages",1)); //tototo 

   xml_init(anObj.ModuleGeomImage(),aTree->Get("ModuleGeomImage",1)); //tototo 

   xml_init(anObj.Images(),aTree->Get("Images",1)); //tototo 

   xml_init(anObj.NomsGeometrieImage(),aTree->GetAll("NomsGeometrieImage",false,1));

   xml_init(anObj.NomsHomomologues(),aTree->Get("NomsHomomologues",1)); //tototo 

   xml_init(anObj.FCND_CalcHomFromI1I2(),aTree->Get("FCND_CalcHomFromI1I2",1)); //tototo 

   xml_init(anObj.SingulariteInCorresp_I1I2(),aTree->Get("SingulariteInCorresp_I1I2",1),bool(false)); //tototo 

   xml_init(anObj.ClassEquivalenceImage(),aTree->Get("ClassEquivalenceImage",1)); //tototo 
}

std::string  Mangling( cSection_PriseDeVue *) {return "02C3FBEF3BE2D087FF3F";};


int & cEchantillonagePtsInterets::FreqEchantPtsI()
{
   return mFreqEchantPtsI;
}

const int & cEchantillonagePtsInterets::FreqEchantPtsI()const 
{
   return mFreqEchantPtsI;
}


eTypeModeEchantPtsI & cEchantillonagePtsInterets::ModeEchantPtsI()
{
   return mModeEchantPtsI;
}

const eTypeModeEchantPtsI & cEchantillonagePtsInterets::ModeEchantPtsI()const 
{
   return mModeEchantPtsI;
}


cTplValGesInit< std::string > & cEchantillonagePtsInterets::KeyCommandeExterneInteret()
{
   return mKeyCommandeExterneInteret;
}

const cTplValGesInit< std::string > & cEchantillonagePtsInterets::KeyCommandeExterneInteret()const 
{
   return mKeyCommandeExterneInteret;
}


cTplValGesInit< int > & cEchantillonagePtsInterets::SzVAutoCorrel()
{
   return mSzVAutoCorrel;
}

const cTplValGesInit< int > & cEchantillonagePtsInterets::SzVAutoCorrel()const 
{
   return mSzVAutoCorrel;
}


cTplValGesInit< double > & cEchantillonagePtsInterets::EstmBrAutoCorrel()
{
   return mEstmBrAutoCorrel;
}

const cTplValGesInit< double > & cEchantillonagePtsInterets::EstmBrAutoCorrel()const 
{
   return mEstmBrAutoCorrel;
}


cTplValGesInit< double > & cEchantillonagePtsInterets::SeuilLambdaAutoCorrel()
{
   return mSeuilLambdaAutoCorrel;
}

const cTplValGesInit< double > & cEchantillonagePtsInterets::SeuilLambdaAutoCorrel()const 
{
   return mSeuilLambdaAutoCorrel;
}


cTplValGesInit< double > & cEchantillonagePtsInterets::SeuilEcartTypeAutoCorrel()
{
   return mSeuilEcartTypeAutoCorrel;
}

const cTplValGesInit< double > & cEchantillonagePtsInterets::SeuilEcartTypeAutoCorrel()const 
{
   return mSeuilEcartTypeAutoCorrel;
}


cTplValGesInit< double > & cEchantillonagePtsInterets::RepartExclusion()
{
   return mRepartExclusion;
}

const cTplValGesInit< double > & cEchantillonagePtsInterets::RepartExclusion()const 
{
   return mRepartExclusion;
}


cTplValGesInit< double > & cEchantillonagePtsInterets::RepartEvitement()
{
   return mRepartEvitement;
}

const cTplValGesInit< double > & cEchantillonagePtsInterets::RepartEvitement()const 
{
   return mRepartEvitement;
}

void  BinaryUnDumpFromFile(cEchantillonagePtsInterets & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.FreqEchantPtsI(),aFp);
    BinaryUnDumpFromFile(anObj.ModeEchantPtsI(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.KeyCommandeExterneInteret().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.KeyCommandeExterneInteret().ValForcedForUnUmp(),aFp);
        }
        else  anObj.KeyCommandeExterneInteret().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SzVAutoCorrel().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SzVAutoCorrel().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SzVAutoCorrel().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.EstmBrAutoCorrel().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.EstmBrAutoCorrel().ValForcedForUnUmp(),aFp);
        }
        else  anObj.EstmBrAutoCorrel().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SeuilLambdaAutoCorrel().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SeuilLambdaAutoCorrel().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SeuilLambdaAutoCorrel().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SeuilEcartTypeAutoCorrel().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SeuilEcartTypeAutoCorrel().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SeuilEcartTypeAutoCorrel().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.RepartExclusion().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.RepartExclusion().ValForcedForUnUmp(),aFp);
        }
        else  anObj.RepartExclusion().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.RepartEvitement().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.RepartEvitement().ValForcedForUnUmp(),aFp);
        }
        else  anObj.RepartEvitement().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cEchantillonagePtsInterets & anObj)
{
    BinaryDumpInFile(aFp,anObj.FreqEchantPtsI());
    BinaryDumpInFile(aFp,anObj.ModeEchantPtsI());
    BinaryDumpInFile(aFp,anObj.KeyCommandeExterneInteret().IsInit());
    if (anObj.KeyCommandeExterneInteret().IsInit()) BinaryDumpInFile(aFp,anObj.KeyCommandeExterneInteret().Val());
    BinaryDumpInFile(aFp,anObj.SzVAutoCorrel().IsInit());
    if (anObj.SzVAutoCorrel().IsInit()) BinaryDumpInFile(aFp,anObj.SzVAutoCorrel().Val());
    BinaryDumpInFile(aFp,anObj.EstmBrAutoCorrel().IsInit());
    if (anObj.EstmBrAutoCorrel().IsInit()) BinaryDumpInFile(aFp,anObj.EstmBrAutoCorrel().Val());
    BinaryDumpInFile(aFp,anObj.SeuilLambdaAutoCorrel().IsInit());
    if (anObj.SeuilLambdaAutoCorrel().IsInit()) BinaryDumpInFile(aFp,anObj.SeuilLambdaAutoCorrel().Val());
    BinaryDumpInFile(aFp,anObj.SeuilEcartTypeAutoCorrel().IsInit());
    if (anObj.SeuilEcartTypeAutoCorrel().IsInit()) BinaryDumpInFile(aFp,anObj.SeuilEcartTypeAutoCorrel().Val());
    BinaryDumpInFile(aFp,anObj.RepartExclusion().IsInit());
    if (anObj.RepartExclusion().IsInit()) BinaryDumpInFile(aFp,anObj.RepartExclusion().Val());
    BinaryDumpInFile(aFp,anObj.RepartEvitement().IsInit());
    if (anObj.RepartEvitement().IsInit()) BinaryDumpInFile(aFp,anObj.RepartEvitement().Val());
}

cElXMLTree * ToXMLTree(const cEchantillonagePtsInterets & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"EchantillonagePtsInterets",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("FreqEchantPtsI"),anObj.FreqEchantPtsI())->ReTagThis("FreqEchantPtsI"));
   aRes->AddFils(ToXMLTree(std::string("ModeEchantPtsI"),anObj.ModeEchantPtsI())->ReTagThis("ModeEchantPtsI"));
   if (anObj.KeyCommandeExterneInteret().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("KeyCommandeExterneInteret"),anObj.KeyCommandeExterneInteret().Val())->ReTagThis("KeyCommandeExterneInteret"));
   if (anObj.SzVAutoCorrel().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SzVAutoCorrel"),anObj.SzVAutoCorrel().Val())->ReTagThis("SzVAutoCorrel"));
   if (anObj.EstmBrAutoCorrel().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("EstmBrAutoCorrel"),anObj.EstmBrAutoCorrel().Val())->ReTagThis("EstmBrAutoCorrel"));
   if (anObj.SeuilLambdaAutoCorrel().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SeuilLambdaAutoCorrel"),anObj.SeuilLambdaAutoCorrel().Val())->ReTagThis("SeuilLambdaAutoCorrel"));
   if (anObj.SeuilEcartTypeAutoCorrel().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SeuilEcartTypeAutoCorrel"),anObj.SeuilEcartTypeAutoCorrel().Val())->ReTagThis("SeuilEcartTypeAutoCorrel"));
   if (anObj.RepartExclusion().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("RepartExclusion"),anObj.RepartExclusion().Val())->ReTagThis("RepartExclusion"));
   if (anObj.RepartEvitement().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("RepartEvitement"),anObj.RepartEvitement().Val())->ReTagThis("RepartEvitement"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cEchantillonagePtsInterets & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.FreqEchantPtsI(),aTree->Get("FreqEchantPtsI",1)); //tototo 

   xml_init(anObj.ModeEchantPtsI(),aTree->Get("ModeEchantPtsI",1)); //tototo 

   xml_init(anObj.KeyCommandeExterneInteret(),aTree->Get("KeyCommandeExterneInteret",1)); //tototo 

   xml_init(anObj.SzVAutoCorrel(),aTree->Get("SzVAutoCorrel",1),int(2)); //tototo 

   xml_init(anObj.EstmBrAutoCorrel(),aTree->Get("EstmBrAutoCorrel",1),double(-1.0)); //tototo 

   xml_init(anObj.SeuilLambdaAutoCorrel(),aTree->Get("SeuilLambdaAutoCorrel",1),double(0.0)); //tototo 

   xml_init(anObj.SeuilEcartTypeAutoCorrel(),aTree->Get("SeuilEcartTypeAutoCorrel",1),double(0.0)); //tototo 

   xml_init(anObj.RepartExclusion(),aTree->Get("RepartExclusion",1),double(0.4)); //tototo 

   xml_init(anObj.RepartEvitement(),aTree->Get("RepartEvitement",1),double(1.0)); //tototo 
}

std::string  Mangling( cEchantillonagePtsInterets *) {return "18CC366F62510ABEFE3F";};


cTplValGesInit< double > & cAdapteDynCov::CovLim()
{
   return mCovLim;
}

const cTplValGesInit< double > & cAdapteDynCov::CovLim()const 
{
   return mCovLim;
}


cTplValGesInit< double > & cAdapteDynCov::TermeDecr()
{
   return mTermeDecr;
}

const cTplValGesInit< double > & cAdapteDynCov::TermeDecr()const 
{
   return mTermeDecr;
}


cTplValGesInit< int > & cAdapteDynCov::SzRef()
{
   return mSzRef;
}

const cTplValGesInit< int > & cAdapteDynCov::SzRef()const 
{
   return mSzRef;
}


cTplValGesInit< double > & cAdapteDynCov::ValRef()
{
   return mValRef;
}

const cTplValGesInit< double > & cAdapteDynCov::ValRef()const 
{
   return mValRef;
}

void  BinaryUnDumpFromFile(cAdapteDynCov & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.CovLim().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.CovLim().ValForcedForUnUmp(),aFp);
        }
        else  anObj.CovLim().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.TermeDecr().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.TermeDecr().ValForcedForUnUmp(),aFp);
        }
        else  anObj.TermeDecr().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SzRef().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SzRef().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SzRef().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ValRef().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ValRef().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ValRef().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cAdapteDynCov & anObj)
{
    BinaryDumpInFile(aFp,anObj.CovLim().IsInit());
    if (anObj.CovLim().IsInit()) BinaryDumpInFile(aFp,anObj.CovLim().Val());
    BinaryDumpInFile(aFp,anObj.TermeDecr().IsInit());
    if (anObj.TermeDecr().IsInit()) BinaryDumpInFile(aFp,anObj.TermeDecr().Val());
    BinaryDumpInFile(aFp,anObj.SzRef().IsInit());
    if (anObj.SzRef().IsInit()) BinaryDumpInFile(aFp,anObj.SzRef().Val());
    BinaryDumpInFile(aFp,anObj.ValRef().IsInit());
    if (anObj.ValRef().IsInit()) BinaryDumpInFile(aFp,anObj.ValRef().Val());
}

cElXMLTree * ToXMLTree(const cAdapteDynCov & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"AdapteDynCov",eXMLBranche);
   if (anObj.CovLim().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("CovLim"),anObj.CovLim().Val())->ReTagThis("CovLim"));
   if (anObj.TermeDecr().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("TermeDecr"),anObj.TermeDecr().Val())->ReTagThis("TermeDecr"));
   if (anObj.SzRef().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SzRef"),anObj.SzRef().Val())->ReTagThis("SzRef"));
   if (anObj.ValRef().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ValRef"),anObj.ValRef().Val())->ReTagThis("ValRef"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cAdapteDynCov & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.CovLim(),aTree->Get("CovLim",1),double(0.005)); //tototo 

   xml_init(anObj.TermeDecr(),aTree->Get("TermeDecr",1),double(0.06)); //tototo 

   xml_init(anObj.SzRef(),aTree->Get("SzRef",1),int(3)); //tototo 

   xml_init(anObj.ValRef(),aTree->Get("ValRef",1),double(0.048)); //tototo 
}

std::string  Mangling( cAdapteDynCov *) {return "9550E383FA9718BBFE3F";};


std::string & cMMUseMasq3D::NameMasq()
{
   return mNameMasq;
}

const std::string & cMMUseMasq3D::NameMasq()const 
{
   return mNameMasq;
}


cTplValGesInit< int > & cMMUseMasq3D::ZoomBegin()
{
   return mZoomBegin;
}

const cTplValGesInit< int > & cMMUseMasq3D::ZoomBegin()const 
{
   return mZoomBegin;
}


cTplValGesInit< int > & cMMUseMasq3D::Dilate()
{
   return mDilate;
}

const cTplValGesInit< int > & cMMUseMasq3D::Dilate()const 
{
   return mDilate;
}


cTplValGesInit< std::string > & cMMUseMasq3D::PrefixNuage()
{
   return mPrefixNuage;
}

const cTplValGesInit< std::string > & cMMUseMasq3D::PrefixNuage()const 
{
   return mPrefixNuage;
}

void  BinaryUnDumpFromFile(cMMUseMasq3D & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.NameMasq(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ZoomBegin().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ZoomBegin().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ZoomBegin().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Dilate().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Dilate().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Dilate().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.PrefixNuage().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.PrefixNuage().ValForcedForUnUmp(),aFp);
        }
        else  anObj.PrefixNuage().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cMMUseMasq3D & anObj)
{
    BinaryDumpInFile(aFp,anObj.NameMasq());
    BinaryDumpInFile(aFp,anObj.ZoomBegin().IsInit());
    if (anObj.ZoomBegin().IsInit()) BinaryDumpInFile(aFp,anObj.ZoomBegin().Val());
    BinaryDumpInFile(aFp,anObj.Dilate().IsInit());
    if (anObj.Dilate().IsInit()) BinaryDumpInFile(aFp,anObj.Dilate().Val());
    BinaryDumpInFile(aFp,anObj.PrefixNuage().IsInit());
    if (anObj.PrefixNuage().IsInit()) BinaryDumpInFile(aFp,anObj.PrefixNuage().Val());
}

cElXMLTree * ToXMLTree(const cMMUseMasq3D & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"MMUseMasq3D",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("NameMasq"),anObj.NameMasq())->ReTagThis("NameMasq"));
   if (anObj.ZoomBegin().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ZoomBegin"),anObj.ZoomBegin().Val())->ReTagThis("ZoomBegin"));
   if (anObj.Dilate().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Dilate"),anObj.Dilate().Val())->ReTagThis("Dilate"));
   if (anObj.PrefixNuage().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PrefixNuage"),anObj.PrefixNuage().Val())->ReTagThis("PrefixNuage"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cMMUseMasq3D & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.NameMasq(),aTree->Get("NameMasq",1)); //tototo 

   xml_init(anObj.ZoomBegin(),aTree->Get("ZoomBegin",1),int(16)); //tototo 

   xml_init(anObj.Dilate(),aTree->Get("Dilate",1),int(2)); //tototo 

   xml_init(anObj.PrefixNuage(),aTree->Get("PrefixNuage",1)); //tototo 
}

std::string  Mangling( cMMUseMasq3D *) {return "A68920AD59E3C7CEFDBF";};


Pt2di & cOneParamCMS::SzW()
{
   return mSzW;
}

const Pt2di & cOneParamCMS::SzW()const 
{
   return mSzW;
}


double & cOneParamCMS::Sigma()
{
   return mSigma;
}

const double & cOneParamCMS::Sigma()const 
{
   return mSigma;
}


double & cOneParamCMS::Pds()
{
   return mPds;
}

const double & cOneParamCMS::Pds()const 
{
   return mPds;
}


cTplValGesInit< bool > & cOneParamCMS::SquareW()
{
   return mSquareW;
}

const cTplValGesInit< bool > & cOneParamCMS::SquareW()const 
{
   return mSquareW;
}

void  BinaryUnDumpFromFile(cOneParamCMS & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.SzW(),aFp);
    BinaryUnDumpFromFile(anObj.Sigma(),aFp);
    BinaryUnDumpFromFile(anObj.Pds(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SquareW().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SquareW().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SquareW().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cOneParamCMS & anObj)
{
    BinaryDumpInFile(aFp,anObj.SzW());
    BinaryDumpInFile(aFp,anObj.Sigma());
    BinaryDumpInFile(aFp,anObj.Pds());
    BinaryDumpInFile(aFp,anObj.SquareW().IsInit());
    if (anObj.SquareW().IsInit()) BinaryDumpInFile(aFp,anObj.SquareW().Val());
}

cElXMLTree * ToXMLTree(const cOneParamCMS & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"OneParamCMS",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("SzW"),anObj.SzW())->ReTagThis("SzW"));
   aRes->AddFils(::ToXMLTree(std::string("Sigma"),anObj.Sigma())->ReTagThis("Sigma"));
   aRes->AddFils(::ToXMLTree(std::string("Pds"),anObj.Pds())->ReTagThis("Pds"));
   if (anObj.SquareW().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SquareW"),anObj.SquareW().Val())->ReTagThis("SquareW"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cOneParamCMS & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.SzW(),aTree->Get("SzW",1)); //tototo 

   xml_init(anObj.Sigma(),aTree->Get("Sigma",1)); //tototo 

   xml_init(anObj.Pds(),aTree->Get("Pds",1)); //tototo 

   xml_init(anObj.SquareW(),aTree->Get("SquareW",1),bool(false)); //tototo 
}

std::string  Mangling( cOneParamCMS *) {return "B2CC5E5196B2C2EDFE3F";};


cTplValGesInit< bool > & cCorrelMultiScale::UseGpGpu()
{
   return mUseGpGpu;
}

const cTplValGesInit< bool > & cCorrelMultiScale::UseGpGpu()const 
{
   return mUseGpGpu;
}


cTplValGesInit< bool > & cCorrelMultiScale::ModeDense()
{
   return mModeDense;
}

const cTplValGesInit< bool > & cCorrelMultiScale::ModeDense()const 
{
   return mModeDense;
}


cTplValGesInit< bool > & cCorrelMultiScale::UseWAdapt()
{
   return mUseWAdapt;
}

const cTplValGesInit< bool > & cCorrelMultiScale::UseWAdapt()const 
{
   return mUseWAdapt;
}


cTplValGesInit< bool > & cCorrelMultiScale::ModeMax()
{
   return mModeMax;
}

const cTplValGesInit< bool > & cCorrelMultiScale::ModeMax()const 
{
   return mModeMax;
}


std::vector< cOneParamCMS > & cCorrelMultiScale::OneParamCMS()
{
   return mOneParamCMS;
}

const std::vector< cOneParamCMS > & cCorrelMultiScale::OneParamCMS()const 
{
   return mOneParamCMS;
}

void  BinaryUnDumpFromFile(cCorrelMultiScale & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.UseGpGpu().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.UseGpGpu().ValForcedForUnUmp(),aFp);
        }
        else  anObj.UseGpGpu().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ModeDense().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ModeDense().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ModeDense().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.UseWAdapt().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.UseWAdapt().ValForcedForUnUmp(),aFp);
        }
        else  anObj.UseWAdapt().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ModeMax().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ModeMax().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ModeMax().SetNoInit();
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cOneParamCMS aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.OneParamCMS().push_back(aVal);
        }
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cCorrelMultiScale & anObj)
{
    BinaryDumpInFile(aFp,anObj.UseGpGpu().IsInit());
    if (anObj.UseGpGpu().IsInit()) BinaryDumpInFile(aFp,anObj.UseGpGpu().Val());
    BinaryDumpInFile(aFp,anObj.ModeDense().IsInit());
    if (anObj.ModeDense().IsInit()) BinaryDumpInFile(aFp,anObj.ModeDense().Val());
    BinaryDumpInFile(aFp,anObj.UseWAdapt().IsInit());
    if (anObj.UseWAdapt().IsInit()) BinaryDumpInFile(aFp,anObj.UseWAdapt().Val());
    BinaryDumpInFile(aFp,anObj.ModeMax().IsInit());
    if (anObj.ModeMax().IsInit()) BinaryDumpInFile(aFp,anObj.ModeMax().Val());
    BinaryDumpInFile(aFp,(int)anObj.OneParamCMS().size());
    for(  std::vector< cOneParamCMS >::const_iterator iT=anObj.OneParamCMS().begin();
         iT!=anObj.OneParamCMS().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
}

cElXMLTree * ToXMLTree(const cCorrelMultiScale & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"CorrelMultiScale",eXMLBranche);
   if (anObj.UseGpGpu().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("UseGpGpu"),anObj.UseGpGpu().Val())->ReTagThis("UseGpGpu"));
   if (anObj.ModeDense().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ModeDense"),anObj.ModeDense().Val())->ReTagThis("ModeDense"));
   if (anObj.UseWAdapt().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("UseWAdapt"),anObj.UseWAdapt().Val())->ReTagThis("UseWAdapt"));
   if (anObj.ModeMax().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ModeMax"),anObj.ModeMax().Val())->ReTagThis("ModeMax"));
  for
  (       std::vector< cOneParamCMS >::const_iterator it=anObj.OneParamCMS().begin();
      it !=anObj.OneParamCMS().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("OneParamCMS"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cCorrelMultiScale & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.UseGpGpu(),aTree->Get("UseGpGpu",1),bool(false)); //tototo 

   xml_init(anObj.ModeDense(),aTree->Get("ModeDense",1)); //tototo 

   xml_init(anObj.UseWAdapt(),aTree->Get("UseWAdapt",1),bool(false)); //tototo 

   xml_init(anObj.ModeMax(),aTree->Get("ModeMax",1),bool(false)); //tototo 

   xml_init(anObj.OneParamCMS(),aTree->GetAll("OneParamCMS",false,1));
}

std::string  Mangling( cCorrelMultiScale *) {return "6C14BAD3172CC7F4FD3F";};


cTplValGesInit< double > & cCensusCost::Dyn()
{
   return mDyn;
}

const cTplValGesInit< double > & cCensusCost::Dyn()const 
{
   return mDyn;
}


eModeCensusCost & cCensusCost::TypeCost()
{
   return mTypeCost;
}

const eModeCensusCost & cCensusCost::TypeCost()const 
{
   return mTypeCost;
}


cTplValGesInit< bool > & cCensusCost::Verif()
{
   return mVerif;
}

const cTplValGesInit< bool > & cCensusCost::Verif()const 
{
   return mVerif;
}


cTplValGesInit< double > & cCensusCost::AttenDist()
{
   return mAttenDist;
}

const cTplValGesInit< double > & cCensusCost::AttenDist()const 
{
   return mAttenDist;
}


cTplValGesInit< double > & cCensusCost::SeuilHautCorMixte()
{
   return mSeuilHautCorMixte;
}

const cTplValGesInit< double > & cCensusCost::SeuilHautCorMixte()const 
{
   return mSeuilHautCorMixte;
}


cTplValGesInit< double > & cCensusCost::SeuilBasCorMixte()
{
   return mSeuilBasCorMixte;
}

const cTplValGesInit< double > & cCensusCost::SeuilBasCorMixte()const 
{
   return mSeuilBasCorMixte;
}

void  BinaryUnDumpFromFile(cCensusCost & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Dyn().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Dyn().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Dyn().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.TypeCost(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Verif().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Verif().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Verif().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.AttenDist().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.AttenDist().ValForcedForUnUmp(),aFp);
        }
        else  anObj.AttenDist().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SeuilHautCorMixte().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SeuilHautCorMixte().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SeuilHautCorMixte().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SeuilBasCorMixte().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SeuilBasCorMixte().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SeuilBasCorMixte().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cCensusCost & anObj)
{
    BinaryDumpInFile(aFp,anObj.Dyn().IsInit());
    if (anObj.Dyn().IsInit()) BinaryDumpInFile(aFp,anObj.Dyn().Val());
    BinaryDumpInFile(aFp,anObj.TypeCost());
    BinaryDumpInFile(aFp,anObj.Verif().IsInit());
    if (anObj.Verif().IsInit()) BinaryDumpInFile(aFp,anObj.Verif().Val());
    BinaryDumpInFile(aFp,anObj.AttenDist().IsInit());
    if (anObj.AttenDist().IsInit()) BinaryDumpInFile(aFp,anObj.AttenDist().Val());
    BinaryDumpInFile(aFp,anObj.SeuilHautCorMixte().IsInit());
    if (anObj.SeuilHautCorMixte().IsInit()) BinaryDumpInFile(aFp,anObj.SeuilHautCorMixte().Val());
    BinaryDumpInFile(aFp,anObj.SeuilBasCorMixte().IsInit());
    if (anObj.SeuilBasCorMixte().IsInit()) BinaryDumpInFile(aFp,anObj.SeuilBasCorMixte().Val());
}

cElXMLTree * ToXMLTree(const cCensusCost & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"CensusCost",eXMLBranche);
   if (anObj.Dyn().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Dyn"),anObj.Dyn().Val())->ReTagThis("Dyn"));
   aRes->AddFils(ToXMLTree(std::string("TypeCost"),anObj.TypeCost())->ReTagThis("TypeCost"));
   if (anObj.Verif().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Verif"),anObj.Verif().Val())->ReTagThis("Verif"));
   if (anObj.AttenDist().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("AttenDist"),anObj.AttenDist().Val())->ReTagThis("AttenDist"));
   if (anObj.SeuilHautCorMixte().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SeuilHautCorMixte"),anObj.SeuilHautCorMixte().Val())->ReTagThis("SeuilHautCorMixte"));
   if (anObj.SeuilBasCorMixte().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SeuilBasCorMixte"),anObj.SeuilBasCorMixte().Val())->ReTagThis("SeuilBasCorMixte"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cCensusCost & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.Dyn(),aTree->Get("Dyn",1),double(1.0)); //tototo 

   xml_init(anObj.TypeCost(),aTree->Get("TypeCost",1)); //tototo 

   xml_init(anObj.Verif(),aTree->Get("Verif",1),bool(false)); //tototo 

   xml_init(anObj.AttenDist(),aTree->Get("AttenDist",1),double(0.0)); //tototo 

   xml_init(anObj.SeuilHautCorMixte(),aTree->Get("SeuilHautCorMixte",1),double(0.8)); //tototo 

   xml_init(anObj.SeuilBasCorMixte(),aTree->Get("SeuilBasCorMixte",1),double(0.6)); //tototo 
}

std::string  Mangling( cCensusCost *) {return "4A6290B0D5407AD0FD3F";};


int & cCorrel2DLeastSquare::SzW()
{
   return mSzW;
}

const int & cCorrel2DLeastSquare::SzW()const 
{
   return mSzW;
}


int & cCorrel2DLeastSquare::PeriodEch()
{
   return mPeriodEch;
}

const int & cCorrel2DLeastSquare::PeriodEch()const 
{
   return mPeriodEch;
}


cTplValGesInit< double > & cCorrel2DLeastSquare::Step()
{
   return mStep;
}

const cTplValGesInit< double > & cCorrel2DLeastSquare::Step()const 
{
   return mStep;
}

void  BinaryUnDumpFromFile(cCorrel2DLeastSquare & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.SzW(),aFp);
    BinaryUnDumpFromFile(anObj.PeriodEch(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Step().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Step().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Step().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cCorrel2DLeastSquare & anObj)
{
    BinaryDumpInFile(aFp,anObj.SzW());
    BinaryDumpInFile(aFp,anObj.PeriodEch());
    BinaryDumpInFile(aFp,anObj.Step().IsInit());
    if (anObj.Step().IsInit()) BinaryDumpInFile(aFp,anObj.Step().Val());
}

cElXMLTree * ToXMLTree(const cCorrel2DLeastSquare & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"Correl2DLeastSquare",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("SzW"),anObj.SzW())->ReTagThis("SzW"));
   aRes->AddFils(::ToXMLTree(std::string("PeriodEch"),anObj.PeriodEch())->ReTagThis("PeriodEch"));
   if (anObj.Step().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Step"),anObj.Step().Val())->ReTagThis("Step"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cCorrel2DLeastSquare & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.SzW(),aTree->Get("SzW",1)); //tototo 

   xml_init(anObj.PeriodEch(),aTree->Get("PeriodEch",1)); //tototo 

   xml_init(anObj.Step(),aTree->Get("Step",1),double(1.0)); //tototo 
}

std::string  Mangling( cCorrel2DLeastSquare *) {return "3EF976DA4F9A4184FE3F";};


cTplValGesInit< std::string > & cGPU_Correl::Unused()
{
   return mUnused;
}

const cTplValGesInit< std::string > & cGPU_Correl::Unused()const 
{
   return mUnused;
}

void  BinaryUnDumpFromFile(cGPU_Correl & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Unused().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Unused().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Unused().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cGPU_Correl & anObj)
{
    BinaryDumpInFile(aFp,anObj.Unused().IsInit());
    if (anObj.Unused().IsInit()) BinaryDumpInFile(aFp,anObj.Unused().Val());
}

cElXMLTree * ToXMLTree(const cGPU_Correl & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"GPU_Correl",eXMLBranche);
   if (anObj.Unused().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Unused"),anObj.Unused().Val())->ReTagThis("Unused"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cGPU_Correl & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.Unused(),aTree->Get("Unused",1)); //tototo 
}

std::string  Mangling( cGPU_Correl *) {return "70BE79E33E0436B7FDBF";};


cTplValGesInit< std::string > & cMutiCorrelOrthoExt::Cmd()
{
   return mCmd;
}

const cTplValGesInit< std::string > & cMutiCorrelOrthoExt::Cmd()const 
{
   return mCmd;
}


cTplValGesInit< std::string > & cMutiCorrelOrthoExt::Options()
{
   return mOptions;
}

const cTplValGesInit< std::string > & cMutiCorrelOrthoExt::Options()const 
{
   return mOptions;
}


cTplValGesInit< int > & cMutiCorrelOrthoExt::DeltaZ()
{
   return mDeltaZ;
}

const cTplValGesInit< int > & cMutiCorrelOrthoExt::DeltaZ()const 
{
   return mDeltaZ;
}

void  BinaryUnDumpFromFile(cMutiCorrelOrthoExt & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Cmd().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Cmd().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Cmd().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Options().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Options().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Options().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.DeltaZ().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.DeltaZ().ValForcedForUnUmp(),aFp);
        }
        else  anObj.DeltaZ().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cMutiCorrelOrthoExt & anObj)
{
    BinaryDumpInFile(aFp,anObj.Cmd().IsInit());
    if (anObj.Cmd().IsInit()) BinaryDumpInFile(aFp,anObj.Cmd().Val());
    BinaryDumpInFile(aFp,anObj.Options().IsInit());
    if (anObj.Options().IsInit()) BinaryDumpInFile(aFp,anObj.Options().Val());
    BinaryDumpInFile(aFp,anObj.DeltaZ().IsInit());
    if (anObj.DeltaZ().IsInit()) BinaryDumpInFile(aFp,anObj.DeltaZ().Val());
}

cElXMLTree * ToXMLTree(const cMutiCorrelOrthoExt & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"MutiCorrelOrthoExt",eXMLBranche);
   if (anObj.Cmd().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Cmd"),anObj.Cmd().Val())->ReTagThis("Cmd"));
   if (anObj.Options().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Options"),anObj.Options().Val())->ReTagThis("Options"));
   if (anObj.DeltaZ().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DeltaZ"),anObj.DeltaZ().Val())->ReTagThis("DeltaZ"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cMutiCorrelOrthoExt & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.Cmd(),aTree->Get("Cmd",1),std::string("MMVII  DM4MatchMultipleOrtho ")); //tototo 

   xml_init(anObj.Options(),aTree->Get("Options",1)); //tototo 

   xml_init(anObj.DeltaZ(),aTree->Get("DeltaZ",1),int(50)); //tototo 
}

std::string  Mangling( cMutiCorrelOrthoExt *) {return "006BFD020F1FA285FF3F";};


cTplValGesInit< std::string > & cGPU_CorrelBasik::Unused()
{
   return mUnused;
}

const cTplValGesInit< std::string > & cGPU_CorrelBasik::Unused()const 
{
   return mUnused;
}

void  BinaryUnDumpFromFile(cGPU_CorrelBasik & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Unused().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Unused().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Unused().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cGPU_CorrelBasik & anObj)
{
    BinaryDumpInFile(aFp,anObj.Unused().IsInit());
    if (anObj.Unused().IsInit()) BinaryDumpInFile(aFp,anObj.Unused().Val());
}

cElXMLTree * ToXMLTree(const cGPU_CorrelBasik & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"GPU_CorrelBasik",eXMLBranche);
   if (anObj.Unused().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Unused"),anObj.Unused().Val())->ReTagThis("Unused"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cGPU_CorrelBasik & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.Unused(),aTree->Get("Unused",1)); //tototo 
}

std::string  Mangling( cGPU_CorrelBasik *) {return "BA35F60E2463F2F7FE3F";};


double & cMCP_AttachePixel::Pds()
{
   return mPds;
}

const double & cMCP_AttachePixel::Pds()const 
{
   return mPds;
}


std::string & cMCP_AttachePixel::KeyRatio()
{
   return mKeyRatio;
}

const std::string & cMCP_AttachePixel::KeyRatio()const 
{
   return mKeyRatio;
}

void  BinaryUnDumpFromFile(cMCP_AttachePixel & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Pds(),aFp);
    BinaryUnDumpFromFile(anObj.KeyRatio(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cMCP_AttachePixel & anObj)
{
    BinaryDumpInFile(aFp,anObj.Pds());
    BinaryDumpInFile(aFp,anObj.KeyRatio());
}

cElXMLTree * ToXMLTree(const cMCP_AttachePixel & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"MCP_AttachePixel",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Pds"),anObj.Pds())->ReTagThis("Pds"));
   aRes->AddFils(::ToXMLTree(std::string("KeyRatio"),anObj.KeyRatio())->ReTagThis("KeyRatio"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cMCP_AttachePixel & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.Pds(),aTree->Get("Pds",1)); //tototo 

   xml_init(anObj.KeyRatio(),aTree->Get("KeyRatio",1)); //tototo 
}

std::string  Mangling( cMCP_AttachePixel *) {return "B2E84BE0DB927B84FE3F";};


double & cMultiCorrelPonctuel::PdsCorrelStd()
{
   return mPdsCorrelStd;
}

const double & cMultiCorrelPonctuel::PdsCorrelStd()const 
{
   return mPdsCorrelStd;
}


double & cMultiCorrelPonctuel::PdsCorrelCroise()
{
   return mPdsCorrelCroise;
}

const double & cMultiCorrelPonctuel::PdsCorrelCroise()const 
{
   return mPdsCorrelCroise;
}


cTplValGesInit< double > & cMultiCorrelPonctuel::DynRadCorrelPonct()
{
   return mDynRadCorrelPonct;
}

const cTplValGesInit< double > & cMultiCorrelPonctuel::DynRadCorrelPonct()const 
{
   return mDynRadCorrelPonct;
}


cTplValGesInit< double > & cMultiCorrelPonctuel::DefCost()
{
   return mDefCost;
}

const cTplValGesInit< double > & cMultiCorrelPonctuel::DefCost()const 
{
   return mDefCost;
}


double & cMultiCorrelPonctuel::Pds()
{
   return MCP_AttachePixel().Val().Pds();
}

const double & cMultiCorrelPonctuel::Pds()const 
{
   return MCP_AttachePixel().Val().Pds();
}


std::string & cMultiCorrelPonctuel::KeyRatio()
{
   return MCP_AttachePixel().Val().KeyRatio();
}

const std::string & cMultiCorrelPonctuel::KeyRatio()const 
{
   return MCP_AttachePixel().Val().KeyRatio();
}


cTplValGesInit< cMCP_AttachePixel > & cMultiCorrelPonctuel::MCP_AttachePixel()
{
   return mMCP_AttachePixel;
}

const cTplValGesInit< cMCP_AttachePixel > & cMultiCorrelPonctuel::MCP_AttachePixel()const 
{
   return mMCP_AttachePixel;
}

void  BinaryUnDumpFromFile(cMultiCorrelPonctuel & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.PdsCorrelStd(),aFp);
    BinaryUnDumpFromFile(anObj.PdsCorrelCroise(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.DynRadCorrelPonct().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.DynRadCorrelPonct().ValForcedForUnUmp(),aFp);
        }
        else  anObj.DynRadCorrelPonct().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.DefCost().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.DefCost().ValForcedForUnUmp(),aFp);
        }
        else  anObj.DefCost().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.MCP_AttachePixel().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.MCP_AttachePixel().ValForcedForUnUmp(),aFp);
        }
        else  anObj.MCP_AttachePixel().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cMultiCorrelPonctuel & anObj)
{
    BinaryDumpInFile(aFp,anObj.PdsCorrelStd());
    BinaryDumpInFile(aFp,anObj.PdsCorrelCroise());
    BinaryDumpInFile(aFp,anObj.DynRadCorrelPonct().IsInit());
    if (anObj.DynRadCorrelPonct().IsInit()) BinaryDumpInFile(aFp,anObj.DynRadCorrelPonct().Val());
    BinaryDumpInFile(aFp,anObj.DefCost().IsInit());
    if (anObj.DefCost().IsInit()) BinaryDumpInFile(aFp,anObj.DefCost().Val());
    BinaryDumpInFile(aFp,anObj.MCP_AttachePixel().IsInit());
    if (anObj.MCP_AttachePixel().IsInit()) BinaryDumpInFile(aFp,anObj.MCP_AttachePixel().Val());
}

cElXMLTree * ToXMLTree(const cMultiCorrelPonctuel & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"MultiCorrelPonctuel",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("PdsCorrelStd"),anObj.PdsCorrelStd())->ReTagThis("PdsCorrelStd"));
   aRes->AddFils(::ToXMLTree(std::string("PdsCorrelCroise"),anObj.PdsCorrelCroise())->ReTagThis("PdsCorrelCroise"));
   if (anObj.DynRadCorrelPonct().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DynRadCorrelPonct"),anObj.DynRadCorrelPonct().Val())->ReTagThis("DynRadCorrelPonct"));
   if (anObj.DefCost().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DefCost"),anObj.DefCost().Val())->ReTagThis("DefCost"));
   if (anObj.MCP_AttachePixel().IsInit())
      aRes->AddFils(ToXMLTree(anObj.MCP_AttachePixel().Val())->ReTagThis("MCP_AttachePixel"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cMultiCorrelPonctuel & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.PdsCorrelStd(),aTree->Get("PdsCorrelStd",1)); //tototo 

   xml_init(anObj.PdsCorrelCroise(),aTree->Get("PdsCorrelCroise",1)); //tototo 

   xml_init(anObj.DynRadCorrelPonct(),aTree->Get("DynRadCorrelPonct",1),double(1.0)); //tototo 

   xml_init(anObj.DefCost(),aTree->Get("DefCost",1),double(0.1)); //tototo 

   xml_init(anObj.MCP_AttachePixel(),aTree->Get("MCP_AttachePixel",1)); //tototo 
}

std::string  Mangling( cMultiCorrelPonctuel *) {return "74A694492496F5B7FE3F";};


std::string & cScoreLearnedMMVII::FileModeleCost()
{
   return mFileModeleCost;
}

const std::string & cScoreLearnedMMVII::FileModeleCost()const 
{
   return mFileModeleCost;
}


cTplValGesInit< double > & cScoreLearnedMMVII::CostDyn()
{
   return mCostDyn;
}

const cTplValGesInit< double > & cScoreLearnedMMVII::CostDyn()const 
{
   return mCostDyn;
}


cTplValGesInit< double > & cScoreLearnedMMVII::CostExp()
{
   return mCostExp;
}

const cTplValGesInit< double > & cScoreLearnedMMVII::CostExp()const 
{
   return mCostExp;
}


cTplValGesInit< std::string > & cScoreLearnedMMVII::Cmp_FileMC()
{
   return mCmp_FileMC;
}

const cTplValGesInit< std::string > & cScoreLearnedMMVII::Cmp_FileMC()const 
{
   return mCmp_FileMC;
}


cTplValGesInit< int > & cScoreLearnedMMVII::Cmp_NbDisc()
{
   return mCmp_NbDisc;
}

const cTplValGesInit< int > & cScoreLearnedMMVII::Cmp_NbDisc()const 
{
   return mCmp_NbDisc;
}

void  BinaryUnDumpFromFile(cScoreLearnedMMVII & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.FileModeleCost(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.CostDyn().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.CostDyn().ValForcedForUnUmp(),aFp);
        }
        else  anObj.CostDyn().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.CostExp().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.CostExp().ValForcedForUnUmp(),aFp);
        }
        else  anObj.CostExp().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Cmp_FileMC().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Cmp_FileMC().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Cmp_FileMC().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Cmp_NbDisc().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Cmp_NbDisc().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Cmp_NbDisc().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cScoreLearnedMMVII & anObj)
{
    BinaryDumpInFile(aFp,anObj.FileModeleCost());
    BinaryDumpInFile(aFp,anObj.CostDyn().IsInit());
    if (anObj.CostDyn().IsInit()) BinaryDumpInFile(aFp,anObj.CostDyn().Val());
    BinaryDumpInFile(aFp,anObj.CostExp().IsInit());
    if (anObj.CostExp().IsInit()) BinaryDumpInFile(aFp,anObj.CostExp().Val());
    BinaryDumpInFile(aFp,anObj.Cmp_FileMC().IsInit());
    if (anObj.Cmp_FileMC().IsInit()) BinaryDumpInFile(aFp,anObj.Cmp_FileMC().Val());
    BinaryDumpInFile(aFp,anObj.Cmp_NbDisc().IsInit());
    if (anObj.Cmp_NbDisc().IsInit()) BinaryDumpInFile(aFp,anObj.Cmp_NbDisc().Val());
}

cElXMLTree * ToXMLTree(const cScoreLearnedMMVII & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ScoreLearnedMMVII",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("FileModeleCost"),anObj.FileModeleCost())->ReTagThis("FileModeleCost"));
   if (anObj.CostDyn().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("CostDyn"),anObj.CostDyn().Val())->ReTagThis("CostDyn"));
   if (anObj.CostExp().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("CostExp"),anObj.CostExp().Val())->ReTagThis("CostExp"));
   if (anObj.Cmp_FileMC().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Cmp_FileMC"),anObj.Cmp_FileMC().Val())->ReTagThis("Cmp_FileMC"));
   if (anObj.Cmp_NbDisc().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Cmp_NbDisc"),anObj.Cmp_NbDisc().Val())->ReTagThis("Cmp_NbDisc"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cScoreLearnedMMVII & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.FileModeleCost(),aTree->Get("FileModeleCost",1)); //tototo 

   xml_init(anObj.CostDyn(),aTree->Get("CostDyn",1),double(0.3333)); //tototo 

   xml_init(anObj.CostExp(),aTree->Get("CostExp",1),double(0.5)); //tototo 

   xml_init(anObj.Cmp_FileMC(),aTree->Get("Cmp_FileMC",1)); //tototo 

   xml_init(anObj.Cmp_NbDisc(),aTree->Get("Cmp_NbDisc",1),int(200)); //tototo 
}

std::string  Mangling( cScoreLearnedMMVII *) {return "A35B5BF6225F1883FE3F";};


cTplValGesInit< double > & cCorrel_Ponctuel2ImGeomI::RatioI1I2()
{
   return mRatioI1I2;
}

const cTplValGesInit< double > & cCorrel_Ponctuel2ImGeomI::RatioI1I2()const 
{
   return mRatioI1I2;
}

void  BinaryUnDumpFromFile(cCorrel_Ponctuel2ImGeomI & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.RatioI1I2().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.RatioI1I2().ValForcedForUnUmp(),aFp);
        }
        else  anObj.RatioI1I2().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cCorrel_Ponctuel2ImGeomI & anObj)
{
    BinaryDumpInFile(aFp,anObj.RatioI1I2().IsInit());
    if (anObj.RatioI1I2().IsInit()) BinaryDumpInFile(aFp,anObj.RatioI1I2().Val());
}

cElXMLTree * ToXMLTree(const cCorrel_Ponctuel2ImGeomI & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"Correl_Ponctuel2ImGeomI",eXMLBranche);
   if (anObj.RatioI1I2().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("RatioI1I2"),anObj.RatioI1I2().Val())->ReTagThis("RatioI1I2"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cCorrel_Ponctuel2ImGeomI & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.RatioI1I2(),aTree->Get("RatioI1I2",1),double(1.0)); //tototo 
}

std::string  Mangling( cCorrel_Ponctuel2ImGeomI *) {return "530CA2D89EFF2BACFD3F";};


cTplValGesInit< double > & cCorrel_PonctuelleCroisee::RatioI1I2()
{
   return mRatioI1I2;
}

const cTplValGesInit< double > & cCorrel_PonctuelleCroisee::RatioI1I2()const 
{
   return mRatioI1I2;
}


double & cCorrel_PonctuelleCroisee::PdsPonctuel()
{
   return mPdsPonctuel;
}

const double & cCorrel_PonctuelleCroisee::PdsPonctuel()const 
{
   return mPdsPonctuel;
}


double & cCorrel_PonctuelleCroisee::PdsCroisee()
{
   return mPdsCroisee;
}

const double & cCorrel_PonctuelleCroisee::PdsCroisee()const 
{
   return mPdsCroisee;
}

void  BinaryUnDumpFromFile(cCorrel_PonctuelleCroisee & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.RatioI1I2().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.RatioI1I2().ValForcedForUnUmp(),aFp);
        }
        else  anObj.RatioI1I2().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.PdsPonctuel(),aFp);
    BinaryUnDumpFromFile(anObj.PdsCroisee(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cCorrel_PonctuelleCroisee & anObj)
{
    BinaryDumpInFile(aFp,anObj.RatioI1I2().IsInit());
    if (anObj.RatioI1I2().IsInit()) BinaryDumpInFile(aFp,anObj.RatioI1I2().Val());
    BinaryDumpInFile(aFp,anObj.PdsPonctuel());
    BinaryDumpInFile(aFp,anObj.PdsCroisee());
}

cElXMLTree * ToXMLTree(const cCorrel_PonctuelleCroisee & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"Correl_PonctuelleCroisee",eXMLBranche);
   if (anObj.RatioI1I2().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("RatioI1I2"),anObj.RatioI1I2().Val())->ReTagThis("RatioI1I2"));
   aRes->AddFils(::ToXMLTree(std::string("PdsPonctuel"),anObj.PdsPonctuel())->ReTagThis("PdsPonctuel"));
   aRes->AddFils(::ToXMLTree(std::string("PdsCroisee"),anObj.PdsCroisee())->ReTagThis("PdsCroisee"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cCorrel_PonctuelleCroisee & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.RatioI1I2(),aTree->Get("RatioI1I2",1),double(1.0)); //tototo 

   xml_init(anObj.PdsPonctuel(),aTree->Get("PdsPonctuel",1)); //tototo 

   xml_init(anObj.PdsCroisee(),aTree->Get("PdsCroisee",1)); //tototo 
}

std::string  Mangling( cCorrel_PonctuelleCroisee *) {return "CC50770C3674C099FE3F";};


int & cCorrel_MultiFen::NbFen()
{
   return mNbFen;
}

const int & cCorrel_MultiFen::NbFen()const 
{
   return mNbFen;
}

void  BinaryUnDumpFromFile(cCorrel_MultiFen & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.NbFen(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cCorrel_MultiFen & anObj)
{
    BinaryDumpInFile(aFp,anObj.NbFen());
}

cElXMLTree * ToXMLTree(const cCorrel_MultiFen & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"Correl_MultiFen",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("NbFen"),anObj.NbFen())->ReTagThis("NbFen"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cCorrel_MultiFen & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.NbFen(),aTree->Get("NbFen",1)); //tototo 
}

std::string  Mangling( cCorrel_MultiFen *) {return "60FB1DED467C2F9EFE3F";};


double & cCorrel_Correl_MNE_ZPredic::SeuilDZ()
{
   return mSeuilDZ;
}

const double & cCorrel_Correl_MNE_ZPredic::SeuilDZ()const 
{
   return mSeuilDZ;
}

void  BinaryUnDumpFromFile(cCorrel_Correl_MNE_ZPredic & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.SeuilDZ(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cCorrel_Correl_MNE_ZPredic & anObj)
{
    BinaryDumpInFile(aFp,anObj.SeuilDZ());
}

cElXMLTree * ToXMLTree(const cCorrel_Correl_MNE_ZPredic & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"Correl_Correl_MNE_ZPredic",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("SeuilDZ"),anObj.SeuilDZ())->ReTagThis("SeuilDZ"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cCorrel_Correl_MNE_ZPredic & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.SeuilDZ(),aTree->Get("SeuilDZ",1)); //tototo 
}

std::string  Mangling( cCorrel_Correl_MNE_ZPredic *) {return "C06875CA6A5F9FDCF9BF";};


cTplValGesInit< std::string > & cCorrel_NC_Robuste::Unused()
{
   return mUnused;
}

const cTplValGesInit< std::string > & cCorrel_NC_Robuste::Unused()const 
{
   return mUnused;
}

void  BinaryUnDumpFromFile(cCorrel_NC_Robuste & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Unused().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Unused().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Unused().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cCorrel_NC_Robuste & anObj)
{
    BinaryDumpInFile(aFp,anObj.Unused().IsInit());
    if (anObj.Unused().IsInit()) BinaryDumpInFile(aFp,anObj.Unused().Val());
}

cElXMLTree * ToXMLTree(const cCorrel_NC_Robuste & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"Correl_NC_Robuste",eXMLBranche);
   if (anObj.Unused().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Unused"),anObj.Unused().Val())->ReTagThis("Unused"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cCorrel_NC_Robuste & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.Unused(),aTree->Get("Unused",1)); //tototo 
}

std::string  Mangling( cCorrel_NC_Robuste *) {return "BC9F108668CFDFE6FD3F";};


cTplValGesInit< bool > & cComputeAndExportEnveloppe::EndAfter()
{
   return mEndAfter;
}

const cTplValGesInit< bool > & cComputeAndExportEnveloppe::EndAfter()const 
{
   return mEndAfter;
}


cTplValGesInit< std::string > & cComputeAndExportEnveloppe::NuageExport()
{
   return mNuageExport;
}

const cTplValGesInit< std::string > & cComputeAndExportEnveloppe::NuageExport()const 
{
   return mNuageExport;
}


cTplValGesInit< double > & cComputeAndExportEnveloppe::SsEchFilter()
{
   return mSsEchFilter;
}

const cTplValGesInit< double > & cComputeAndExportEnveloppe::SsEchFilter()const 
{
   return mSsEchFilter;
}


cTplValGesInit< int > & cComputeAndExportEnveloppe::SzFilter()
{
   return mSzFilter;
}

const cTplValGesInit< int > & cComputeAndExportEnveloppe::SzFilter()const 
{
   return mSzFilter;
}


cTplValGesInit< double > & cComputeAndExportEnveloppe::ParamPropFilter()
{
   return mParamPropFilter;
}

const cTplValGesInit< double > & cComputeAndExportEnveloppe::ParamPropFilter()const 
{
   return mParamPropFilter;
}


cTplValGesInit< double > & cComputeAndExportEnveloppe::ProlResolCible()
{
   return mProlResolCible;
}

const cTplValGesInit< double > & cComputeAndExportEnveloppe::ProlResolCible()const 
{
   return mProlResolCible;
}


cTplValGesInit< double > & cComputeAndExportEnveloppe::ProlResolCur()
{
   return mProlResolCur;
}

const cTplValGesInit< double > & cComputeAndExportEnveloppe::ProlResolCur()const 
{
   return mProlResolCur;
}


cTplValGesInit< double > & cComputeAndExportEnveloppe::ProlDistAdd()
{
   return mProlDistAdd;
}

const cTplValGesInit< double > & cComputeAndExportEnveloppe::ProlDistAdd()const 
{
   return mProlDistAdd;
}


cTplValGesInit< double > & cComputeAndExportEnveloppe::ProlDistAddMax()
{
   return mProlDistAddMax;
}

const cTplValGesInit< double > & cComputeAndExportEnveloppe::ProlDistAddMax()const 
{
   return mProlDistAddMax;
}


cTplValGesInit< int > & cComputeAndExportEnveloppe::DilatAltiCible()
{
   return mDilatAltiCible;
}

const cTplValGesInit< int > & cComputeAndExportEnveloppe::DilatAltiCible()const 
{
   return mDilatAltiCible;
}


cTplValGesInit< int > & cComputeAndExportEnveloppe::DilatPlaniCible()
{
   return mDilatPlaniCible;
}

const cTplValGesInit< int > & cComputeAndExportEnveloppe::DilatPlaniCible()const 
{
   return mDilatPlaniCible;
}


cTplValGesInit< int > & cComputeAndExportEnveloppe::DilatPlaniCur()
{
   return mDilatPlaniCur;
}

const cTplValGesInit< int > & cComputeAndExportEnveloppe::DilatPlaniCur()const 
{
   return mDilatPlaniCur;
}


cTplValGesInit< int > & cComputeAndExportEnveloppe::DilatAltiCur()
{
   return mDilatAltiCur;
}

const cTplValGesInit< int > & cComputeAndExportEnveloppe::DilatAltiCur()const 
{
   return mDilatAltiCur;
}

void  BinaryUnDumpFromFile(cComputeAndExportEnveloppe & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.EndAfter().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.EndAfter().ValForcedForUnUmp(),aFp);
        }
        else  anObj.EndAfter().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NuageExport().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NuageExport().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NuageExport().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SsEchFilter().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SsEchFilter().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SsEchFilter().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SzFilter().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SzFilter().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SzFilter().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ParamPropFilter().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ParamPropFilter().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ParamPropFilter().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ProlResolCible().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ProlResolCible().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ProlResolCible().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ProlResolCur().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ProlResolCur().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ProlResolCur().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ProlDistAdd().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ProlDistAdd().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ProlDistAdd().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ProlDistAddMax().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ProlDistAddMax().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ProlDistAddMax().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.DilatAltiCible().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.DilatAltiCible().ValForcedForUnUmp(),aFp);
        }
        else  anObj.DilatAltiCible().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.DilatPlaniCible().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.DilatPlaniCible().ValForcedForUnUmp(),aFp);
        }
        else  anObj.DilatPlaniCible().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.DilatPlaniCur().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.DilatPlaniCur().ValForcedForUnUmp(),aFp);
        }
        else  anObj.DilatPlaniCur().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.DilatAltiCur().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.DilatAltiCur().ValForcedForUnUmp(),aFp);
        }
        else  anObj.DilatAltiCur().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cComputeAndExportEnveloppe & anObj)
{
    BinaryDumpInFile(aFp,anObj.EndAfter().IsInit());
    if (anObj.EndAfter().IsInit()) BinaryDumpInFile(aFp,anObj.EndAfter().Val());
    BinaryDumpInFile(aFp,anObj.NuageExport().IsInit());
    if (anObj.NuageExport().IsInit()) BinaryDumpInFile(aFp,anObj.NuageExport().Val());
    BinaryDumpInFile(aFp,anObj.SsEchFilter().IsInit());
    if (anObj.SsEchFilter().IsInit()) BinaryDumpInFile(aFp,anObj.SsEchFilter().Val());
    BinaryDumpInFile(aFp,anObj.SzFilter().IsInit());
    if (anObj.SzFilter().IsInit()) BinaryDumpInFile(aFp,anObj.SzFilter().Val());
    BinaryDumpInFile(aFp,anObj.ParamPropFilter().IsInit());
    if (anObj.ParamPropFilter().IsInit()) BinaryDumpInFile(aFp,anObj.ParamPropFilter().Val());
    BinaryDumpInFile(aFp,anObj.ProlResolCible().IsInit());
    if (anObj.ProlResolCible().IsInit()) BinaryDumpInFile(aFp,anObj.ProlResolCible().Val());
    BinaryDumpInFile(aFp,anObj.ProlResolCur().IsInit());
    if (anObj.ProlResolCur().IsInit()) BinaryDumpInFile(aFp,anObj.ProlResolCur().Val());
    BinaryDumpInFile(aFp,anObj.ProlDistAdd().IsInit());
    if (anObj.ProlDistAdd().IsInit()) BinaryDumpInFile(aFp,anObj.ProlDistAdd().Val());
    BinaryDumpInFile(aFp,anObj.ProlDistAddMax().IsInit());
    if (anObj.ProlDistAddMax().IsInit()) BinaryDumpInFile(aFp,anObj.ProlDistAddMax().Val());
    BinaryDumpInFile(aFp,anObj.DilatAltiCible().IsInit());
    if (anObj.DilatAltiCible().IsInit()) BinaryDumpInFile(aFp,anObj.DilatAltiCible().Val());
    BinaryDumpInFile(aFp,anObj.DilatPlaniCible().IsInit());
    if (anObj.DilatPlaniCible().IsInit()) BinaryDumpInFile(aFp,anObj.DilatPlaniCible().Val());
    BinaryDumpInFile(aFp,anObj.DilatPlaniCur().IsInit());
    if (anObj.DilatPlaniCur().IsInit()) BinaryDumpInFile(aFp,anObj.DilatPlaniCur().Val());
    BinaryDumpInFile(aFp,anObj.DilatAltiCur().IsInit());
    if (anObj.DilatAltiCur().IsInit()) BinaryDumpInFile(aFp,anObj.DilatAltiCur().Val());
}

cElXMLTree * ToXMLTree(const cComputeAndExportEnveloppe & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ComputeAndExportEnveloppe",eXMLBranche);
   if (anObj.EndAfter().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("EndAfter"),anObj.EndAfter().Val())->ReTagThis("EndAfter"));
   if (anObj.NuageExport().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NuageExport"),anObj.NuageExport().Val())->ReTagThis("NuageExport"));
   if (anObj.SsEchFilter().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SsEchFilter"),anObj.SsEchFilter().Val())->ReTagThis("SsEchFilter"));
   if (anObj.SzFilter().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SzFilter"),anObj.SzFilter().Val())->ReTagThis("SzFilter"));
   if (anObj.ParamPropFilter().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ParamPropFilter"),anObj.ParamPropFilter().Val())->ReTagThis("ParamPropFilter"));
   if (anObj.ProlResolCible().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ProlResolCible"),anObj.ProlResolCible().Val())->ReTagThis("ProlResolCible"));
   if (anObj.ProlResolCur().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ProlResolCur"),anObj.ProlResolCur().Val())->ReTagThis("ProlResolCur"));
   if (anObj.ProlDistAdd().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ProlDistAdd"),anObj.ProlDistAdd().Val())->ReTagThis("ProlDistAdd"));
   if (anObj.ProlDistAddMax().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ProlDistAddMax"),anObj.ProlDistAddMax().Val())->ReTagThis("ProlDistAddMax"));
   if (anObj.DilatAltiCible().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DilatAltiCible"),anObj.DilatAltiCible().Val())->ReTagThis("DilatAltiCible"));
   if (anObj.DilatPlaniCible().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DilatPlaniCible"),anObj.DilatPlaniCible().Val())->ReTagThis("DilatPlaniCible"));
   if (anObj.DilatPlaniCur().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DilatPlaniCur"),anObj.DilatPlaniCur().Val())->ReTagThis("DilatPlaniCur"));
   if (anObj.DilatAltiCur().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DilatAltiCur"),anObj.DilatAltiCur().Val())->ReTagThis("DilatAltiCur"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cComputeAndExportEnveloppe & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.EndAfter(),aTree->Get("EndAfter",1),bool(true)); //tototo 

   xml_init(anObj.NuageExport(),aTree->Get("NuageExport",1)); //tototo 

   xml_init(anObj.SsEchFilter(),aTree->Get("SsEchFilter",1),double(3.0)); //tototo 

   xml_init(anObj.SzFilter(),aTree->Get("SzFilter",1),int(7)); //tototo 

   xml_init(anObj.ParamPropFilter(),aTree->Get("ParamPropFilter",1),double(0.9)); //tototo 

   xml_init(anObj.ProlResolCible(),aTree->Get("ProlResolCible",1),double(25)); //tototo 

   xml_init(anObj.ProlResolCur(),aTree->Get("ProlResolCur",1),double(10)); //tototo 

   xml_init(anObj.ProlDistAdd(),aTree->Get("ProlDistAdd",1),double(0.25)); //tototo 

   xml_init(anObj.ProlDistAddMax(),aTree->Get("ProlDistAddMax",1),double(3.0)); //tototo 

   xml_init(anObj.DilatAltiCible(),aTree->Get("DilatAltiCible",1),int(5)); //tototo 

   xml_init(anObj.DilatPlaniCible(),aTree->Get("DilatPlaniCible",1),int(5)); //tototo 

   xml_init(anObj.DilatPlaniCur(),aTree->Get("DilatPlaniCur",1),int(2)); //tototo 

   xml_init(anObj.DilatAltiCur(),aTree->Get("DilatAltiCur",1),int(2)); //tototo 
}

std::string  Mangling( cComputeAndExportEnveloppe *) {return "1B3C95B3BEBD3385FF3F";};


cTplValGesInit< double > & cmmtpFilterSky::PertPerPix()
{
   return mPertPerPix;
}

const cTplValGesInit< double > & cmmtpFilterSky::PertPerPix()const 
{
   return mPertPerPix;
}


cTplValGesInit< int > & cmmtpFilterSky::SzKernelHom()
{
   return mSzKernelHom;
}

const cTplValGesInit< int > & cmmtpFilterSky::SzKernelHom()const 
{
   return mSzKernelHom;
}


cTplValGesInit< double > & cmmtpFilterSky::PropZonec()
{
   return mPropZonec;
}

const cTplValGesInit< double > & cmmtpFilterSky::PropZonec()const 
{
   return mPropZonec;
}

void  BinaryUnDumpFromFile(cmmtpFilterSky & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.PertPerPix().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.PertPerPix().ValForcedForUnUmp(),aFp);
        }
        else  anObj.PertPerPix().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SzKernelHom().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SzKernelHom().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SzKernelHom().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.PropZonec().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.PropZonec().ValForcedForUnUmp(),aFp);
        }
        else  anObj.PropZonec().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cmmtpFilterSky & anObj)
{
    BinaryDumpInFile(aFp,anObj.PertPerPix().IsInit());
    if (anObj.PertPerPix().IsInit()) BinaryDumpInFile(aFp,anObj.PertPerPix().Val());
    BinaryDumpInFile(aFp,anObj.SzKernelHom().IsInit());
    if (anObj.SzKernelHom().IsInit()) BinaryDumpInFile(aFp,anObj.SzKernelHom().Val());
    BinaryDumpInFile(aFp,anObj.PropZonec().IsInit());
    if (anObj.PropZonec().IsInit()) BinaryDumpInFile(aFp,anObj.PropZonec().Val());
}

cElXMLTree * ToXMLTree(const cmmtpFilterSky & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"mmtpFilterSky",eXMLBranche);
   if (anObj.PertPerPix().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PertPerPix"),anObj.PertPerPix().Val())->ReTagThis("PertPerPix"));
   if (anObj.SzKernelHom().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SzKernelHom"),anObj.SzKernelHom().Val())->ReTagThis("SzKernelHom"));
   if (anObj.PropZonec().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PropZonec"),anObj.PropZonec().Val())->ReTagThis("PropZonec"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cmmtpFilterSky & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.PertPerPix(),aTree->Get("PertPerPix",1),double(0.005)); //tototo 

   xml_init(anObj.SzKernelHom(),aTree->Get("SzKernelHom",1),int(5)); //tototo 

   xml_init(anObj.PropZonec(),aTree->Get("PropZonec",1),double(0.001)); //tototo 
}

std::string  Mangling( cmmtpFilterSky *) {return "1E09691F7669669DFF3F";};


int & cTiePMasqIm::DeZoomRel()
{
   return mDeZoomRel;
}

const int & cTiePMasqIm::DeZoomRel()const 
{
   return mDeZoomRel;
}


int & cTiePMasqIm::Dilate()
{
   return mDilate;
}

const int & cTiePMasqIm::Dilate()const 
{
   return mDilate;
}

void  BinaryUnDumpFromFile(cTiePMasqIm & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.DeZoomRel(),aFp);
    BinaryUnDumpFromFile(anObj.Dilate(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cTiePMasqIm & anObj)
{
    BinaryDumpInFile(aFp,anObj.DeZoomRel());
    BinaryDumpInFile(aFp,anObj.Dilate());
}

cElXMLTree * ToXMLTree(const cTiePMasqIm & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"TiePMasqIm",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("DeZoomRel"),anObj.DeZoomRel())->ReTagThis("DeZoomRel"));
   aRes->AddFils(::ToXMLTree(std::string("Dilate"),anObj.Dilate())->ReTagThis("Dilate"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cTiePMasqIm & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.DeZoomRel(),aTree->Get("DeZoomRel",1)); //tototo 

   xml_init(anObj.Dilate(),aTree->Get("Dilate",1)); //tototo 
}

std::string  Mangling( cTiePMasqIm *) {return "23193741CFD014D6FE3F";};


cTplValGesInit< cParamFiltreDepthByPrgDyn > & cMasqueAutoByTieP::FilterPrgDyn()
{
   return mFilterPrgDyn;
}

const cTplValGesInit< cParamFiltreDepthByPrgDyn > & cMasqueAutoByTieP::FilterPrgDyn()const 
{
   return mFilterPrgDyn;
}


cTplValGesInit< bool > & cMasqueAutoByTieP::EndAfter()
{
   return ComputeAndExportEnveloppe().Val().EndAfter();
}

const cTplValGesInit< bool > & cMasqueAutoByTieP::EndAfter()const 
{
   return ComputeAndExportEnveloppe().Val().EndAfter();
}


cTplValGesInit< std::string > & cMasqueAutoByTieP::NuageExport()
{
   return ComputeAndExportEnveloppe().Val().NuageExport();
}

const cTplValGesInit< std::string > & cMasqueAutoByTieP::NuageExport()const 
{
   return ComputeAndExportEnveloppe().Val().NuageExport();
}


cTplValGesInit< double > & cMasqueAutoByTieP::SsEchFilter()
{
   return ComputeAndExportEnveloppe().Val().SsEchFilter();
}

const cTplValGesInit< double > & cMasqueAutoByTieP::SsEchFilter()const 
{
   return ComputeAndExportEnveloppe().Val().SsEchFilter();
}


cTplValGesInit< int > & cMasqueAutoByTieP::SzFilter()
{
   return ComputeAndExportEnveloppe().Val().SzFilter();
}

const cTplValGesInit< int > & cMasqueAutoByTieP::SzFilter()const 
{
   return ComputeAndExportEnveloppe().Val().SzFilter();
}


cTplValGesInit< double > & cMasqueAutoByTieP::ParamPropFilter()
{
   return ComputeAndExportEnveloppe().Val().ParamPropFilter();
}

const cTplValGesInit< double > & cMasqueAutoByTieP::ParamPropFilter()const 
{
   return ComputeAndExportEnveloppe().Val().ParamPropFilter();
}


cTplValGesInit< double > & cMasqueAutoByTieP::ProlResolCible()
{
   return ComputeAndExportEnveloppe().Val().ProlResolCible();
}

const cTplValGesInit< double > & cMasqueAutoByTieP::ProlResolCible()const 
{
   return ComputeAndExportEnveloppe().Val().ProlResolCible();
}


cTplValGesInit< double > & cMasqueAutoByTieP::ProlResolCur()
{
   return ComputeAndExportEnveloppe().Val().ProlResolCur();
}

const cTplValGesInit< double > & cMasqueAutoByTieP::ProlResolCur()const 
{
   return ComputeAndExportEnveloppe().Val().ProlResolCur();
}


cTplValGesInit< double > & cMasqueAutoByTieP::ProlDistAdd()
{
   return ComputeAndExportEnveloppe().Val().ProlDistAdd();
}

const cTplValGesInit< double > & cMasqueAutoByTieP::ProlDistAdd()const 
{
   return ComputeAndExportEnveloppe().Val().ProlDistAdd();
}


cTplValGesInit< double > & cMasqueAutoByTieP::ProlDistAddMax()
{
   return ComputeAndExportEnveloppe().Val().ProlDistAddMax();
}

const cTplValGesInit< double > & cMasqueAutoByTieP::ProlDistAddMax()const 
{
   return ComputeAndExportEnveloppe().Val().ProlDistAddMax();
}


cTplValGesInit< int > & cMasqueAutoByTieP::DilatAltiCible()
{
   return ComputeAndExportEnveloppe().Val().DilatAltiCible();
}

const cTplValGesInit< int > & cMasqueAutoByTieP::DilatAltiCible()const 
{
   return ComputeAndExportEnveloppe().Val().DilatAltiCible();
}


cTplValGesInit< int > & cMasqueAutoByTieP::DilatPlaniCible()
{
   return ComputeAndExportEnveloppe().Val().DilatPlaniCible();
}

const cTplValGesInit< int > & cMasqueAutoByTieP::DilatPlaniCible()const 
{
   return ComputeAndExportEnveloppe().Val().DilatPlaniCible();
}


cTplValGesInit< int > & cMasqueAutoByTieP::DilatPlaniCur()
{
   return ComputeAndExportEnveloppe().Val().DilatPlaniCur();
}

const cTplValGesInit< int > & cMasqueAutoByTieP::DilatPlaniCur()const 
{
   return ComputeAndExportEnveloppe().Val().DilatPlaniCur();
}


cTplValGesInit< int > & cMasqueAutoByTieP::DilatAltiCur()
{
   return ComputeAndExportEnveloppe().Val().DilatAltiCur();
}

const cTplValGesInit< int > & cMasqueAutoByTieP::DilatAltiCur()const 
{
   return ComputeAndExportEnveloppe().Val().DilatAltiCur();
}


cTplValGesInit< cComputeAndExportEnveloppe > & cMasqueAutoByTieP::ComputeAndExportEnveloppe()
{
   return mComputeAndExportEnveloppe;
}

const cTplValGesInit< cComputeAndExportEnveloppe > & cMasqueAutoByTieP::ComputeAndExportEnveloppe()const 
{
   return mComputeAndExportEnveloppe;
}


cTplValGesInit< double > & cMasqueAutoByTieP::PertPerPix()
{
   return mmtpFilterSky().Val().PertPerPix();
}

const cTplValGesInit< double > & cMasqueAutoByTieP::PertPerPix()const 
{
   return mmtpFilterSky().Val().PertPerPix();
}


cTplValGesInit< int > & cMasqueAutoByTieP::SzKernelHom()
{
   return mmtpFilterSky().Val().SzKernelHom();
}

const cTplValGesInit< int > & cMasqueAutoByTieP::SzKernelHom()const 
{
   return mmtpFilterSky().Val().SzKernelHom();
}


cTplValGesInit< double > & cMasqueAutoByTieP::PropZonec()
{
   return mmtpFilterSky().Val().PropZonec();
}

const cTplValGesInit< double > & cMasqueAutoByTieP::PropZonec()const 
{
   return mmtpFilterSky().Val().PropZonec();
}


cTplValGesInit< cmmtpFilterSky > & cMasqueAutoByTieP::mmtpFilterSky()
{
   return mmmtpFilterSky;
}

const cTplValGesInit< cmmtpFilterSky > & cMasqueAutoByTieP::mmtpFilterSky()const 
{
   return mmmtpFilterSky;
}


cTplValGesInit< bool > & cMasqueAutoByTieP::BasicOneIter()
{
   return mBasicOneIter;
}

const cTplValGesInit< bool > & cMasqueAutoByTieP::BasicOneIter()const 
{
   return mBasicOneIter;
}


cTplValGesInit< std::string > & cMasqueAutoByTieP::Masq3D()
{
   return mMasq3D;
}

const cTplValGesInit< std::string > & cMasqueAutoByTieP::Masq3D()const 
{
   return mMasq3D;
}


cTplValGesInit< cParamFiltreDetecRegulProf > & cMasqueAutoByTieP::ParamFiltreRegProf()
{
   return mParamFiltreRegProf;
}

const cTplValGesInit< cParamFiltreDetecRegulProf > & cMasqueAutoByTieP::ParamFiltreRegProf()const 
{
   return mParamFiltreRegProf;
}


cTplValGesInit< std::string > & cMasqueAutoByTieP::GlobFilePt3D()
{
   return mGlobFilePt3D;
}

const cTplValGesInit< std::string > & cMasqueAutoByTieP::GlobFilePt3D()const 
{
   return mGlobFilePt3D;
}


std::string & cMasqueAutoByTieP::KeyImFilePt3D()
{
   return mKeyImFilePt3D;
}

const std::string & cMasqueAutoByTieP::KeyImFilePt3D()const 
{
   return mKeyImFilePt3D;
}


int & cMasqueAutoByTieP::DeltaZ()
{
   return mDeltaZ;
}

const int & cMasqueAutoByTieP::DeltaZ()const 
{
   return mDeltaZ;
}


double & cMasqueAutoByTieP::SeuilSomCostCorrel()
{
   return mSeuilSomCostCorrel;
}

const double & cMasqueAutoByTieP::SeuilSomCostCorrel()const 
{
   return mSeuilSomCostCorrel;
}


double & cMasqueAutoByTieP::SeuilMaxCostCorrel()
{
   return mSeuilMaxCostCorrel;
}

const double & cMasqueAutoByTieP::SeuilMaxCostCorrel()const 
{
   return mSeuilMaxCostCorrel;
}


double & cMasqueAutoByTieP::SeuilMedCostCorrel()
{
   return mSeuilMedCostCorrel;
}

const double & cMasqueAutoByTieP::SeuilMedCostCorrel()const 
{
   return mSeuilMedCostCorrel;
}


cTplValGesInit< bool > & cMasqueAutoByTieP::Visu()
{
   return mVisu;
}

const cTplValGesInit< bool > & cMasqueAutoByTieP::Visu()const 
{
   return mVisu;
}


cTplValGesInit< eImpaintMethod > & cMasqueAutoByTieP::ImPaintResult()
{
   return mImPaintResult;
}

const cTplValGesInit< eImpaintMethod > & cMasqueAutoByTieP::ImPaintResult()const 
{
   return mImPaintResult;
}


cTplValGesInit< double > & cMasqueAutoByTieP::ParamIPMnt()
{
   return mParamIPMnt;
}

const cTplValGesInit< double > & cMasqueAutoByTieP::ParamIPMnt()const 
{
   return mParamIPMnt;
}


int & cMasqueAutoByTieP::DeZoomRel()
{
   return TiePMasqIm().Val().DeZoomRel();
}

const int & cMasqueAutoByTieP::DeZoomRel()const 
{
   return TiePMasqIm().Val().DeZoomRel();
}


int & cMasqueAutoByTieP::Dilate()
{
   return TiePMasqIm().Val().Dilate();
}

const int & cMasqueAutoByTieP::Dilate()const 
{
   return TiePMasqIm().Val().Dilate();
}


cTplValGesInit< cTiePMasqIm > & cMasqueAutoByTieP::TiePMasqIm()
{
   return mTiePMasqIm;
}

const cTplValGesInit< cTiePMasqIm > & cMasqueAutoByTieP::TiePMasqIm()const 
{
   return mTiePMasqIm;
}


cTplValGesInit< bool > & cMasqueAutoByTieP::DoImageLabel()
{
   return mDoImageLabel;
}

const cTplValGesInit< bool > & cMasqueAutoByTieP::DoImageLabel()const 
{
   return mDoImageLabel;
}

void  BinaryUnDumpFromFile(cMasqueAutoByTieP & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.FilterPrgDyn().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.FilterPrgDyn().ValForcedForUnUmp(),aFp);
        }
        else  anObj.FilterPrgDyn().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ComputeAndExportEnveloppe().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ComputeAndExportEnveloppe().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ComputeAndExportEnveloppe().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.mmtpFilterSky().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.mmtpFilterSky().ValForcedForUnUmp(),aFp);
        }
        else  anObj.mmtpFilterSky().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.BasicOneIter().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.BasicOneIter().ValForcedForUnUmp(),aFp);
        }
        else  anObj.BasicOneIter().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Masq3D().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Masq3D().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Masq3D().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ParamFiltreRegProf().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ParamFiltreRegProf().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ParamFiltreRegProf().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.GlobFilePt3D().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.GlobFilePt3D().ValForcedForUnUmp(),aFp);
        }
        else  anObj.GlobFilePt3D().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.KeyImFilePt3D(),aFp);
    BinaryUnDumpFromFile(anObj.DeltaZ(),aFp);
    BinaryUnDumpFromFile(anObj.SeuilSomCostCorrel(),aFp);
    BinaryUnDumpFromFile(anObj.SeuilMaxCostCorrel(),aFp);
    BinaryUnDumpFromFile(anObj.SeuilMedCostCorrel(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Visu().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Visu().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Visu().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ImPaintResult().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ImPaintResult().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ImPaintResult().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ParamIPMnt().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ParamIPMnt().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ParamIPMnt().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.TiePMasqIm().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.TiePMasqIm().ValForcedForUnUmp(),aFp);
        }
        else  anObj.TiePMasqIm().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.DoImageLabel().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.DoImageLabel().ValForcedForUnUmp(),aFp);
        }
        else  anObj.DoImageLabel().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cMasqueAutoByTieP & anObj)
{
    BinaryDumpInFile(aFp,anObj.FilterPrgDyn().IsInit());
    if (anObj.FilterPrgDyn().IsInit()) BinaryDumpInFile(aFp,anObj.FilterPrgDyn().Val());
    BinaryDumpInFile(aFp,anObj.ComputeAndExportEnveloppe().IsInit());
    if (anObj.ComputeAndExportEnveloppe().IsInit()) BinaryDumpInFile(aFp,anObj.ComputeAndExportEnveloppe().Val());
    BinaryDumpInFile(aFp,anObj.mmtpFilterSky().IsInit());
    if (anObj.mmtpFilterSky().IsInit()) BinaryDumpInFile(aFp,anObj.mmtpFilterSky().Val());
    BinaryDumpInFile(aFp,anObj.BasicOneIter().IsInit());
    if (anObj.BasicOneIter().IsInit()) BinaryDumpInFile(aFp,anObj.BasicOneIter().Val());
    BinaryDumpInFile(aFp,anObj.Masq3D().IsInit());
    if (anObj.Masq3D().IsInit()) BinaryDumpInFile(aFp,anObj.Masq3D().Val());
    BinaryDumpInFile(aFp,anObj.ParamFiltreRegProf().IsInit());
    if (anObj.ParamFiltreRegProf().IsInit()) BinaryDumpInFile(aFp,anObj.ParamFiltreRegProf().Val());
    BinaryDumpInFile(aFp,anObj.GlobFilePt3D().IsInit());
    if (anObj.GlobFilePt3D().IsInit()) BinaryDumpInFile(aFp,anObj.GlobFilePt3D().Val());
    BinaryDumpInFile(aFp,anObj.KeyImFilePt3D());
    BinaryDumpInFile(aFp,anObj.DeltaZ());
    BinaryDumpInFile(aFp,anObj.SeuilSomCostCorrel());
    BinaryDumpInFile(aFp,anObj.SeuilMaxCostCorrel());
    BinaryDumpInFile(aFp,anObj.SeuilMedCostCorrel());
    BinaryDumpInFile(aFp,anObj.Visu().IsInit());
    if (anObj.Visu().IsInit()) BinaryDumpInFile(aFp,anObj.Visu().Val());
    BinaryDumpInFile(aFp,anObj.ImPaintResult().IsInit());
    if (anObj.ImPaintResult().IsInit()) BinaryDumpInFile(aFp,anObj.ImPaintResult().Val());
    BinaryDumpInFile(aFp,anObj.ParamIPMnt().IsInit());
    if (anObj.ParamIPMnt().IsInit()) BinaryDumpInFile(aFp,anObj.ParamIPMnt().Val());
    BinaryDumpInFile(aFp,anObj.TiePMasqIm().IsInit());
    if (anObj.TiePMasqIm().IsInit()) BinaryDumpInFile(aFp,anObj.TiePMasqIm().Val());
    BinaryDumpInFile(aFp,anObj.DoImageLabel().IsInit());
    if (anObj.DoImageLabel().IsInit()) BinaryDumpInFile(aFp,anObj.DoImageLabel().Val());
}

cElXMLTree * ToXMLTree(const cMasqueAutoByTieP & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"MasqueAutoByTieP",eXMLBranche);
   if (anObj.FilterPrgDyn().IsInit())
      aRes->AddFils(ToXMLTree(anObj.FilterPrgDyn().Val())->ReTagThis("FilterPrgDyn"));
   if (anObj.ComputeAndExportEnveloppe().IsInit())
      aRes->AddFils(ToXMLTree(anObj.ComputeAndExportEnveloppe().Val())->ReTagThis("ComputeAndExportEnveloppe"));
   if (anObj.mmtpFilterSky().IsInit())
      aRes->AddFils(ToXMLTree(anObj.mmtpFilterSky().Val())->ReTagThis("mmtpFilterSky"));
   if (anObj.BasicOneIter().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("BasicOneIter"),anObj.BasicOneIter().Val())->ReTagThis("BasicOneIter"));
   if (anObj.Masq3D().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Masq3D"),anObj.Masq3D().Val())->ReTagThis("Masq3D"));
   if (anObj.ParamFiltreRegProf().IsInit())
      aRes->AddFils(ToXMLTree(anObj.ParamFiltreRegProf().Val())->ReTagThis("ParamFiltreRegProf"));
   if (anObj.GlobFilePt3D().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("GlobFilePt3D"),anObj.GlobFilePt3D().Val())->ReTagThis("GlobFilePt3D"));
   aRes->AddFils(::ToXMLTree(std::string("KeyImFilePt3D"),anObj.KeyImFilePt3D())->ReTagThis("KeyImFilePt3D"));
   aRes->AddFils(::ToXMLTree(std::string("DeltaZ"),anObj.DeltaZ())->ReTagThis("DeltaZ"));
   aRes->AddFils(::ToXMLTree(std::string("SeuilSomCostCorrel"),anObj.SeuilSomCostCorrel())->ReTagThis("SeuilSomCostCorrel"));
   aRes->AddFils(::ToXMLTree(std::string("SeuilMaxCostCorrel"),anObj.SeuilMaxCostCorrel())->ReTagThis("SeuilMaxCostCorrel"));
   aRes->AddFils(::ToXMLTree(std::string("SeuilMedCostCorrel"),anObj.SeuilMedCostCorrel())->ReTagThis("SeuilMedCostCorrel"));
   if (anObj.Visu().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Visu"),anObj.Visu().Val())->ReTagThis("Visu"));
   if (anObj.ImPaintResult().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ImPaintResult"),anObj.ImPaintResult().Val())->ReTagThis("ImPaintResult"));
   if (anObj.ParamIPMnt().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ParamIPMnt"),anObj.ParamIPMnt().Val())->ReTagThis("ParamIPMnt"));
   if (anObj.TiePMasqIm().IsInit())
      aRes->AddFils(ToXMLTree(anObj.TiePMasqIm().Val())->ReTagThis("TiePMasqIm"));
   if (anObj.DoImageLabel().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DoImageLabel"),anObj.DoImageLabel().Val())->ReTagThis("DoImageLabel"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cMasqueAutoByTieP & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.FilterPrgDyn(),aTree->Get("FilterPrgDyn",1)); //tototo 

   xml_init(anObj.ComputeAndExportEnveloppe(),aTree->Get("ComputeAndExportEnveloppe",1)); //tototo 

   xml_init(anObj.mmtpFilterSky(),aTree->Get("mmtpFilterSky",1)); //tototo 

   xml_init(anObj.BasicOneIter(),aTree->Get("BasicOneIter",1),bool(true)); //tototo 

   xml_init(anObj.Masq3D(),aTree->Get("Masq3D",1)); //tototo 

   xml_init(anObj.ParamFiltreRegProf(),aTree->Get("ParamFiltreRegProf",1)); //tototo 

   xml_init(anObj.GlobFilePt3D(),aTree->Get("GlobFilePt3D",1)); //tototo 

   xml_init(anObj.KeyImFilePt3D(),aTree->Get("KeyImFilePt3D",1)); //tototo 

   xml_init(anObj.DeltaZ(),aTree->Get("DeltaZ",1)); //tototo 

   xml_init(anObj.SeuilSomCostCorrel(),aTree->Get("SeuilSomCostCorrel",1)); //tototo 

   xml_init(anObj.SeuilMaxCostCorrel(),aTree->Get("SeuilMaxCostCorrel",1)); //tototo 

   xml_init(anObj.SeuilMedCostCorrel(),aTree->Get("SeuilMedCostCorrel",1)); //tototo 

   xml_init(anObj.Visu(),aTree->Get("Visu",1),bool(false)); //tototo 

   xml_init(anObj.ImPaintResult(),aTree->Get("ImPaintResult",1),eImpaintMethod(eImpaintL2)); //tototo 

   xml_init(anObj.ParamIPMnt(),aTree->Get("ParamIPMnt",1),double(1.0)); //tototo 

   xml_init(anObj.TiePMasqIm(),aTree->Get("TiePMasqIm",1)); //tototo 

   xml_init(anObj.DoImageLabel(),aTree->Get("DoImageLabel",1),bool(false)); //tototo 
}

std::string  Mangling( cMasqueAutoByTieP *) {return "BBAB9742295EF9DCFC3F";};


cTplValGesInit< cCensusCost > & cTypeCAH::CensusCost()
{
   return mCensusCost;
}

const cTplValGesInit< cCensusCost > & cTypeCAH::CensusCost()const 
{
   return mCensusCost;
}


cTplValGesInit< cCorrel2DLeastSquare > & cTypeCAH::Correl2DLeastSquare()
{
   return mCorrel2DLeastSquare;
}

const cTplValGesInit< cCorrel2DLeastSquare > & cTypeCAH::Correl2DLeastSquare()const 
{
   return mCorrel2DLeastSquare;
}


cTplValGesInit< cGPU_Correl > & cTypeCAH::GPU_Correl()
{
   return mGPU_Correl;
}

const cTplValGesInit< cGPU_Correl > & cTypeCAH::GPU_Correl()const 
{
   return mGPU_Correl;
}


cTplValGesInit< cMutiCorrelOrthoExt > & cTypeCAH::MutiCorrelOrthoExt()
{
   return mMutiCorrelOrthoExt;
}

const cTplValGesInit< cMutiCorrelOrthoExt > & cTypeCAH::MutiCorrelOrthoExt()const 
{
   return mMutiCorrelOrthoExt;
}


cTplValGesInit< cGPU_CorrelBasik > & cTypeCAH::GPU_CorrelBasik()
{
   return mGPU_CorrelBasik;
}

const cTplValGesInit< cGPU_CorrelBasik > & cTypeCAH::GPU_CorrelBasik()const 
{
   return mGPU_CorrelBasik;
}


cTplValGesInit< cMultiCorrelPonctuel > & cTypeCAH::MultiCorrelPonctuel()
{
   return mMultiCorrelPonctuel;
}

const cTplValGesInit< cMultiCorrelPonctuel > & cTypeCAH::MultiCorrelPonctuel()const 
{
   return mMultiCorrelPonctuel;
}


cTplValGesInit< cScoreLearnedMMVII > & cTypeCAH::ScoreLearnedMMVII()
{
   return mScoreLearnedMMVII;
}

const cTplValGesInit< cScoreLearnedMMVII > & cTypeCAH::ScoreLearnedMMVII()const 
{
   return mScoreLearnedMMVII;
}


cTplValGesInit< cCorrel_Ponctuel2ImGeomI > & cTypeCAH::Correl_Ponctuel2ImGeomI()
{
   return mCorrel_Ponctuel2ImGeomI;
}

const cTplValGesInit< cCorrel_Ponctuel2ImGeomI > & cTypeCAH::Correl_Ponctuel2ImGeomI()const 
{
   return mCorrel_Ponctuel2ImGeomI;
}


cTplValGesInit< cCorrel_PonctuelleCroisee > & cTypeCAH::Correl_PonctuelleCroisee()
{
   return mCorrel_PonctuelleCroisee;
}

const cTplValGesInit< cCorrel_PonctuelleCroisee > & cTypeCAH::Correl_PonctuelleCroisee()const 
{
   return mCorrel_PonctuelleCroisee;
}


cTplValGesInit< cCorrel_MultiFen > & cTypeCAH::Correl_MultiFen()
{
   return mCorrel_MultiFen;
}

const cTplValGesInit< cCorrel_MultiFen > & cTypeCAH::Correl_MultiFen()const 
{
   return mCorrel_MultiFen;
}


cTplValGesInit< cCorrel_Correl_MNE_ZPredic > & cTypeCAH::Correl_Correl_MNE_ZPredic()
{
   return mCorrel_Correl_MNE_ZPredic;
}

const cTplValGesInit< cCorrel_Correl_MNE_ZPredic > & cTypeCAH::Correl_Correl_MNE_ZPredic()const 
{
   return mCorrel_Correl_MNE_ZPredic;
}


cTplValGesInit< cCorrel_NC_Robuste > & cTypeCAH::Correl_NC_Robuste()
{
   return mCorrel_NC_Robuste;
}

const cTplValGesInit< cCorrel_NC_Robuste > & cTypeCAH::Correl_NC_Robuste()const 
{
   return mCorrel_NC_Robuste;
}


cTplValGesInit< cMasqueAutoByTieP > & cTypeCAH::MasqueAutoByTieP()
{
   return mMasqueAutoByTieP;
}

const cTplValGesInit< cMasqueAutoByTieP > & cTypeCAH::MasqueAutoByTieP()const 
{
   return mMasqueAutoByTieP;
}

void  BinaryUnDumpFromFile(cTypeCAH & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.CensusCost().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.CensusCost().ValForcedForUnUmp(),aFp);
        }
        else  anObj.CensusCost().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Correl2DLeastSquare().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Correl2DLeastSquare().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Correl2DLeastSquare().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.GPU_Correl().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.GPU_Correl().ValForcedForUnUmp(),aFp);
        }
        else  anObj.GPU_Correl().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.MutiCorrelOrthoExt().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.MutiCorrelOrthoExt().ValForcedForUnUmp(),aFp);
        }
        else  anObj.MutiCorrelOrthoExt().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.GPU_CorrelBasik().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.GPU_CorrelBasik().ValForcedForUnUmp(),aFp);
        }
        else  anObj.GPU_CorrelBasik().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.MultiCorrelPonctuel().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.MultiCorrelPonctuel().ValForcedForUnUmp(),aFp);
        }
        else  anObj.MultiCorrelPonctuel().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ScoreLearnedMMVII().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ScoreLearnedMMVII().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ScoreLearnedMMVII().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Correl_Ponctuel2ImGeomI().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Correl_Ponctuel2ImGeomI().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Correl_Ponctuel2ImGeomI().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Correl_PonctuelleCroisee().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Correl_PonctuelleCroisee().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Correl_PonctuelleCroisee().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Correl_MultiFen().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Correl_MultiFen().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Correl_MultiFen().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Correl_Correl_MNE_ZPredic().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Correl_Correl_MNE_ZPredic().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Correl_Correl_MNE_ZPredic().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Correl_NC_Robuste().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Correl_NC_Robuste().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Correl_NC_Robuste().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.MasqueAutoByTieP().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.MasqueAutoByTieP().ValForcedForUnUmp(),aFp);
        }
        else  anObj.MasqueAutoByTieP().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cTypeCAH & anObj)
{
    BinaryDumpInFile(aFp,anObj.CensusCost().IsInit());
    if (anObj.CensusCost().IsInit()) BinaryDumpInFile(aFp,anObj.CensusCost().Val());
    BinaryDumpInFile(aFp,anObj.Correl2DLeastSquare().IsInit());
    if (anObj.Correl2DLeastSquare().IsInit()) BinaryDumpInFile(aFp,anObj.Correl2DLeastSquare().Val());
    BinaryDumpInFile(aFp,anObj.GPU_Correl().IsInit());
    if (anObj.GPU_Correl().IsInit()) BinaryDumpInFile(aFp,anObj.GPU_Correl().Val());
    BinaryDumpInFile(aFp,anObj.MutiCorrelOrthoExt().IsInit());
    if (anObj.MutiCorrelOrthoExt().IsInit()) BinaryDumpInFile(aFp,anObj.MutiCorrelOrthoExt().Val());
    BinaryDumpInFile(aFp,anObj.GPU_CorrelBasik().IsInit());
    if (anObj.GPU_CorrelBasik().IsInit()) BinaryDumpInFile(aFp,anObj.GPU_CorrelBasik().Val());
    BinaryDumpInFile(aFp,anObj.MultiCorrelPonctuel().IsInit());
    if (anObj.MultiCorrelPonctuel().IsInit()) BinaryDumpInFile(aFp,anObj.MultiCorrelPonctuel().Val());
    BinaryDumpInFile(aFp,anObj.ScoreLearnedMMVII().IsInit());
    if (anObj.ScoreLearnedMMVII().IsInit()) BinaryDumpInFile(aFp,anObj.ScoreLearnedMMVII().Val());
    BinaryDumpInFile(aFp,anObj.Correl_Ponctuel2ImGeomI().IsInit());
    if (anObj.Correl_Ponctuel2ImGeomI().IsInit()) BinaryDumpInFile(aFp,anObj.Correl_Ponctuel2ImGeomI().Val());
    BinaryDumpInFile(aFp,anObj.Correl_PonctuelleCroisee().IsInit());
    if (anObj.Correl_PonctuelleCroisee().IsInit()) BinaryDumpInFile(aFp,anObj.Correl_PonctuelleCroisee().Val());
    BinaryDumpInFile(aFp,anObj.Correl_MultiFen().IsInit());
    if (anObj.Correl_MultiFen().IsInit()) BinaryDumpInFile(aFp,anObj.Correl_MultiFen().Val());
    BinaryDumpInFile(aFp,anObj.Correl_Correl_MNE_ZPredic().IsInit());
    if (anObj.Correl_Correl_MNE_ZPredic().IsInit()) BinaryDumpInFile(aFp,anObj.Correl_Correl_MNE_ZPredic().Val());
    BinaryDumpInFile(aFp,anObj.Correl_NC_Robuste().IsInit());
    if (anObj.Correl_NC_Robuste().IsInit()) BinaryDumpInFile(aFp,anObj.Correl_NC_Robuste().Val());
    BinaryDumpInFile(aFp,anObj.MasqueAutoByTieP().IsInit());
    if (anObj.MasqueAutoByTieP().IsInit()) BinaryDumpInFile(aFp,anObj.MasqueAutoByTieP().Val());
}

cElXMLTree * ToXMLTree(const cTypeCAH & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"TypeCAH",eXMLBranche);
   if (anObj.CensusCost().IsInit())
      aRes->AddFils(ToXMLTree(anObj.CensusCost().Val())->ReTagThis("CensusCost"));
   if (anObj.Correl2DLeastSquare().IsInit())
      aRes->AddFils(ToXMLTree(anObj.Correl2DLeastSquare().Val())->ReTagThis("Correl2DLeastSquare"));
   if (anObj.GPU_Correl().IsInit())
      aRes->AddFils(ToXMLTree(anObj.GPU_Correl().Val())->ReTagThis("GPU_Correl"));
   if (anObj.MutiCorrelOrthoExt().IsInit())
      aRes->AddFils(ToXMLTree(anObj.MutiCorrelOrthoExt().Val())->ReTagThis("MutiCorrelOrthoExt"));
   if (anObj.GPU_CorrelBasik().IsInit())
      aRes->AddFils(ToXMLTree(anObj.GPU_CorrelBasik().Val())->ReTagThis("GPU_CorrelBasik"));
   if (anObj.MultiCorrelPonctuel().IsInit())
      aRes->AddFils(ToXMLTree(anObj.MultiCorrelPonctuel().Val())->ReTagThis("MultiCorrelPonctuel"));
   if (anObj.ScoreLearnedMMVII().IsInit())
      aRes->AddFils(ToXMLTree(anObj.ScoreLearnedMMVII().Val())->ReTagThis("ScoreLearnedMMVII"));
   if (anObj.Correl_Ponctuel2ImGeomI().IsInit())
      aRes->AddFils(ToXMLTree(anObj.Correl_Ponctuel2ImGeomI().Val())->ReTagThis("Correl_Ponctuel2ImGeomI"));
   if (anObj.Correl_PonctuelleCroisee().IsInit())
      aRes->AddFils(ToXMLTree(anObj.Correl_PonctuelleCroisee().Val())->ReTagThis("Correl_PonctuelleCroisee"));
   if (anObj.Correl_MultiFen().IsInit())
      aRes->AddFils(ToXMLTree(anObj.Correl_MultiFen().Val())->ReTagThis("Correl_MultiFen"));
   if (anObj.Correl_Correl_MNE_ZPredic().IsInit())
      aRes->AddFils(ToXMLTree(anObj.Correl_Correl_MNE_ZPredic().Val())->ReTagThis("Correl_Correl_MNE_ZPredic"));
   if (anObj.Correl_NC_Robuste().IsInit())
      aRes->AddFils(ToXMLTree(anObj.Correl_NC_Robuste().Val())->ReTagThis("Correl_NC_Robuste"));
   if (anObj.MasqueAutoByTieP().IsInit())
      aRes->AddFils(ToXMLTree(anObj.MasqueAutoByTieP().Val())->ReTagThis("MasqueAutoByTieP"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cTypeCAH & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.CensusCost(),aTree->Get("CensusCost",1)); //tototo 

   xml_init(anObj.Correl2DLeastSquare(),aTree->Get("Correl2DLeastSquare",1)); //tototo 

   xml_init(anObj.GPU_Correl(),aTree->Get("GPU_Correl",1)); //tototo 

   xml_init(anObj.MutiCorrelOrthoExt(),aTree->Get("MutiCorrelOrthoExt",1)); //tototo 

   xml_init(anObj.GPU_CorrelBasik(),aTree->Get("GPU_CorrelBasik",1)); //tototo 

   xml_init(anObj.MultiCorrelPonctuel(),aTree->Get("MultiCorrelPonctuel",1)); //tototo 

   xml_init(anObj.ScoreLearnedMMVII(),aTree->Get("ScoreLearnedMMVII",1)); //tototo 

   xml_init(anObj.Correl_Ponctuel2ImGeomI(),aTree->Get("Correl_Ponctuel2ImGeomI",1)); //tototo 

   xml_init(anObj.Correl_PonctuelleCroisee(),aTree->Get("Correl_PonctuelleCroisee",1)); //tototo 

   xml_init(anObj.Correl_MultiFen(),aTree->Get("Correl_MultiFen",1)); //tototo 

   xml_init(anObj.Correl_Correl_MNE_ZPredic(),aTree->Get("Correl_Correl_MNE_ZPredic",1)); //tototo 

   xml_init(anObj.Correl_NC_Robuste(),aTree->Get("Correl_NC_Robuste",1)); //tototo 

   xml_init(anObj.MasqueAutoByTieP(),aTree->Get("MasqueAutoByTieP",1)); //tototo 
}

std::string  Mangling( cTypeCAH *) {return "402BA28C661D14BEFDBF";};


cTplValGesInit< double > & cCorrelAdHoc::EpsilonAddMoyenne()
{
   return mEpsilonAddMoyenne;
}

const cTplValGesInit< double > & cCorrelAdHoc::EpsilonAddMoyenne()const 
{
   return mEpsilonAddMoyenne;
}


cTplValGesInit< double > & cCorrelAdHoc::EpsilonMulMoyenne()
{
   return mEpsilonMulMoyenne;
}

const cTplValGesInit< double > & cCorrelAdHoc::EpsilonMulMoyenne()const 
{
   return mEpsilonMulMoyenne;
}


cTplValGesInit< int > & cCorrelAdHoc::SzBlocAH()
{
   return mSzBlocAH;
}

const cTplValGesInit< int > & cCorrelAdHoc::SzBlocAH()const 
{
   return mSzBlocAH;
}


cTplValGesInit< bool > & cCorrelAdHoc::UseGpGpu()
{
   return CorrelMultiScale().Val().UseGpGpu();
}

const cTplValGesInit< bool > & cCorrelAdHoc::UseGpGpu()const 
{
   return CorrelMultiScale().Val().UseGpGpu();
}


cTplValGesInit< bool > & cCorrelAdHoc::ModeDense()
{
   return CorrelMultiScale().Val().ModeDense();
}

const cTplValGesInit< bool > & cCorrelAdHoc::ModeDense()const 
{
   return CorrelMultiScale().Val().ModeDense();
}


cTplValGesInit< bool > & cCorrelAdHoc::UseWAdapt()
{
   return CorrelMultiScale().Val().UseWAdapt();
}

const cTplValGesInit< bool > & cCorrelAdHoc::UseWAdapt()const 
{
   return CorrelMultiScale().Val().UseWAdapt();
}


cTplValGesInit< bool > & cCorrelAdHoc::ModeMax()
{
   return CorrelMultiScale().Val().ModeMax();
}

const cTplValGesInit< bool > & cCorrelAdHoc::ModeMax()const 
{
   return CorrelMultiScale().Val().ModeMax();
}


std::vector< cOneParamCMS > & cCorrelAdHoc::OneParamCMS()
{
   return CorrelMultiScale().Val().OneParamCMS();
}

const std::vector< cOneParamCMS > & cCorrelAdHoc::OneParamCMS()const 
{
   return CorrelMultiScale().Val().OneParamCMS();
}


cTplValGesInit< cCorrelMultiScale > & cCorrelAdHoc::CorrelMultiScale()
{
   return mCorrelMultiScale;
}

const cTplValGesInit< cCorrelMultiScale > & cCorrelAdHoc::CorrelMultiScale()const 
{
   return mCorrelMultiScale;
}


cTplValGesInit< cCensusCost > & cCorrelAdHoc::CensusCost()
{
   return TypeCAH().CensusCost();
}

const cTplValGesInit< cCensusCost > & cCorrelAdHoc::CensusCost()const 
{
   return TypeCAH().CensusCost();
}


cTplValGesInit< cCorrel2DLeastSquare > & cCorrelAdHoc::Correl2DLeastSquare()
{
   return TypeCAH().Correl2DLeastSquare();
}

const cTplValGesInit< cCorrel2DLeastSquare > & cCorrelAdHoc::Correl2DLeastSquare()const 
{
   return TypeCAH().Correl2DLeastSquare();
}


cTplValGesInit< cGPU_Correl > & cCorrelAdHoc::GPU_Correl()
{
   return TypeCAH().GPU_Correl();
}

const cTplValGesInit< cGPU_Correl > & cCorrelAdHoc::GPU_Correl()const 
{
   return TypeCAH().GPU_Correl();
}


cTplValGesInit< cMutiCorrelOrthoExt > & cCorrelAdHoc::MutiCorrelOrthoExt()
{
   return TypeCAH().MutiCorrelOrthoExt();
}

const cTplValGesInit< cMutiCorrelOrthoExt > & cCorrelAdHoc::MutiCorrelOrthoExt()const 
{
   return TypeCAH().MutiCorrelOrthoExt();
}


cTplValGesInit< cGPU_CorrelBasik > & cCorrelAdHoc::GPU_CorrelBasik()
{
   return TypeCAH().GPU_CorrelBasik();
}

const cTplValGesInit< cGPU_CorrelBasik > & cCorrelAdHoc::GPU_CorrelBasik()const 
{
   return TypeCAH().GPU_CorrelBasik();
}


cTplValGesInit< cMultiCorrelPonctuel > & cCorrelAdHoc::MultiCorrelPonctuel()
{
   return TypeCAH().MultiCorrelPonctuel();
}

const cTplValGesInit< cMultiCorrelPonctuel > & cCorrelAdHoc::MultiCorrelPonctuel()const 
{
   return TypeCAH().MultiCorrelPonctuel();
}


cTplValGesInit< cScoreLearnedMMVII > & cCorrelAdHoc::ScoreLearnedMMVII()
{
   return TypeCAH().ScoreLearnedMMVII();
}

const cTplValGesInit< cScoreLearnedMMVII > & cCorrelAdHoc::ScoreLearnedMMVII()const 
{
   return TypeCAH().ScoreLearnedMMVII();
}


cTplValGesInit< cCorrel_Ponctuel2ImGeomI > & cCorrelAdHoc::Correl_Ponctuel2ImGeomI()
{
   return TypeCAH().Correl_Ponctuel2ImGeomI();
}

const cTplValGesInit< cCorrel_Ponctuel2ImGeomI > & cCorrelAdHoc::Correl_Ponctuel2ImGeomI()const 
{
   return TypeCAH().Correl_Ponctuel2ImGeomI();
}


cTplValGesInit< cCorrel_PonctuelleCroisee > & cCorrelAdHoc::Correl_PonctuelleCroisee()
{
   return TypeCAH().Correl_PonctuelleCroisee();
}

const cTplValGesInit< cCorrel_PonctuelleCroisee > & cCorrelAdHoc::Correl_PonctuelleCroisee()const 
{
   return TypeCAH().Correl_PonctuelleCroisee();
}


cTplValGesInit< cCorrel_MultiFen > & cCorrelAdHoc::Correl_MultiFen()
{
   return TypeCAH().Correl_MultiFen();
}

const cTplValGesInit< cCorrel_MultiFen > & cCorrelAdHoc::Correl_MultiFen()const 
{
   return TypeCAH().Correl_MultiFen();
}


cTplValGesInit< cCorrel_Correl_MNE_ZPredic > & cCorrelAdHoc::Correl_Correl_MNE_ZPredic()
{
   return TypeCAH().Correl_Correl_MNE_ZPredic();
}

const cTplValGesInit< cCorrel_Correl_MNE_ZPredic > & cCorrelAdHoc::Correl_Correl_MNE_ZPredic()const 
{
   return TypeCAH().Correl_Correl_MNE_ZPredic();
}


cTplValGesInit< cCorrel_NC_Robuste > & cCorrelAdHoc::Correl_NC_Robuste()
{
   return TypeCAH().Correl_NC_Robuste();
}

const cTplValGesInit< cCorrel_NC_Robuste > & cCorrelAdHoc::Correl_NC_Robuste()const 
{
   return TypeCAH().Correl_NC_Robuste();
}


cTplValGesInit< cMasqueAutoByTieP > & cCorrelAdHoc::MasqueAutoByTieP()
{
   return TypeCAH().MasqueAutoByTieP();
}

const cTplValGesInit< cMasqueAutoByTieP > & cCorrelAdHoc::MasqueAutoByTieP()const 
{
   return TypeCAH().MasqueAutoByTieP();
}


cTypeCAH & cCorrelAdHoc::TypeCAH()
{
   return mTypeCAH;
}

const cTypeCAH & cCorrelAdHoc::TypeCAH()const 
{
   return mTypeCAH;
}

void  BinaryUnDumpFromFile(cCorrelAdHoc & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.EpsilonAddMoyenne().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.EpsilonAddMoyenne().ValForcedForUnUmp(),aFp);
        }
        else  anObj.EpsilonAddMoyenne().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.EpsilonMulMoyenne().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.EpsilonMulMoyenne().ValForcedForUnUmp(),aFp);
        }
        else  anObj.EpsilonMulMoyenne().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SzBlocAH().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SzBlocAH().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SzBlocAH().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.CorrelMultiScale().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.CorrelMultiScale().ValForcedForUnUmp(),aFp);
        }
        else  anObj.CorrelMultiScale().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.TypeCAH(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cCorrelAdHoc & anObj)
{
    BinaryDumpInFile(aFp,anObj.EpsilonAddMoyenne().IsInit());
    if (anObj.EpsilonAddMoyenne().IsInit()) BinaryDumpInFile(aFp,anObj.EpsilonAddMoyenne().Val());
    BinaryDumpInFile(aFp,anObj.EpsilonMulMoyenne().IsInit());
    if (anObj.EpsilonMulMoyenne().IsInit()) BinaryDumpInFile(aFp,anObj.EpsilonMulMoyenne().Val());
    BinaryDumpInFile(aFp,anObj.SzBlocAH().IsInit());
    if (anObj.SzBlocAH().IsInit()) BinaryDumpInFile(aFp,anObj.SzBlocAH().Val());
    BinaryDumpInFile(aFp,anObj.CorrelMultiScale().IsInit());
    if (anObj.CorrelMultiScale().IsInit()) BinaryDumpInFile(aFp,anObj.CorrelMultiScale().Val());
    BinaryDumpInFile(aFp,anObj.TypeCAH());
}

cElXMLTree * ToXMLTree(const cCorrelAdHoc & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"CorrelAdHoc",eXMLBranche);
   if (anObj.EpsilonAddMoyenne().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("EpsilonAddMoyenne"),anObj.EpsilonAddMoyenne().Val())->ReTagThis("EpsilonAddMoyenne"));
   if (anObj.EpsilonMulMoyenne().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("EpsilonMulMoyenne"),anObj.EpsilonMulMoyenne().Val())->ReTagThis("EpsilonMulMoyenne"));
   if (anObj.SzBlocAH().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SzBlocAH"),anObj.SzBlocAH().Val())->ReTagThis("SzBlocAH"));
   if (anObj.CorrelMultiScale().IsInit())
      aRes->AddFils(ToXMLTree(anObj.CorrelMultiScale().Val())->ReTagThis("CorrelMultiScale"));
   aRes->AddFils(ToXMLTree(anObj.TypeCAH())->ReTagThis("TypeCAH"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cCorrelAdHoc & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.EpsilonAddMoyenne(),aTree->Get("EpsilonAddMoyenne",1),double(0.0)); //tototo 

   xml_init(anObj.EpsilonMulMoyenne(),aTree->Get("EpsilonMulMoyenne",1),double(0.0)); //tototo 

   xml_init(anObj.SzBlocAH(),aTree->Get("SzBlocAH",1),int(40)); //tototo 

   xml_init(anObj.CorrelMultiScale(),aTree->Get("CorrelMultiScale",1)); //tototo 

   xml_init(anObj.TypeCAH(),aTree->Get("TypeCAH",1)); //tototo 
}

std::string  Mangling( cCorrelAdHoc *) {return "90BFEB1511F29DABFE3F";};


cTplValGesInit< double > & cDoImageBSurH::Dyn()
{
   return mDyn;
}

const cTplValGesInit< double > & cDoImageBSurH::Dyn()const 
{
   return mDyn;
}


cTplValGesInit< double > & cDoImageBSurH::Offset()
{
   return mOffset;
}

const cTplValGesInit< double > & cDoImageBSurH::Offset()const 
{
   return mOffset;
}


cTplValGesInit< double > & cDoImageBSurH::SeuilMasqExport()
{
   return mSeuilMasqExport;
}

const cTplValGesInit< double > & cDoImageBSurH::SeuilMasqExport()const 
{
   return mSeuilMasqExport;
}


std::string & cDoImageBSurH::Name()
{
   return mName;
}

const std::string & cDoImageBSurH::Name()const 
{
   return mName;
}


double & cDoImageBSurH::ScaleNuage()
{
   return mScaleNuage;
}

const double & cDoImageBSurH::ScaleNuage()const 
{
   return mScaleNuage;
}


std::string & cDoImageBSurH::NameNuage()
{
   return mNameNuage;
}

const std::string & cDoImageBSurH::NameNuage()const 
{
   return mNameNuage;
}

void  BinaryUnDumpFromFile(cDoImageBSurH & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Dyn().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Dyn().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Dyn().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Offset().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Offset().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Offset().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SeuilMasqExport().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SeuilMasqExport().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SeuilMasqExport().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.Name(),aFp);
    BinaryUnDumpFromFile(anObj.ScaleNuage(),aFp);
    BinaryUnDumpFromFile(anObj.NameNuage(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cDoImageBSurH & anObj)
{
    BinaryDumpInFile(aFp,anObj.Dyn().IsInit());
    if (anObj.Dyn().IsInit()) BinaryDumpInFile(aFp,anObj.Dyn().Val());
    BinaryDumpInFile(aFp,anObj.Offset().IsInit());
    if (anObj.Offset().IsInit()) BinaryDumpInFile(aFp,anObj.Offset().Val());
    BinaryDumpInFile(aFp,anObj.SeuilMasqExport().IsInit());
    if (anObj.SeuilMasqExport().IsInit()) BinaryDumpInFile(aFp,anObj.SeuilMasqExport().Val());
    BinaryDumpInFile(aFp,anObj.Name());
    BinaryDumpInFile(aFp,anObj.ScaleNuage());
    BinaryDumpInFile(aFp,anObj.NameNuage());
}

cElXMLTree * ToXMLTree(const cDoImageBSurH & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"DoImageBSurH",eXMLBranche);
   if (anObj.Dyn().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Dyn"),anObj.Dyn().Val())->ReTagThis("Dyn"));
   if (anObj.Offset().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Offset"),anObj.Offset().Val())->ReTagThis("Offset"));
   if (anObj.SeuilMasqExport().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SeuilMasqExport"),anObj.SeuilMasqExport().Val())->ReTagThis("SeuilMasqExport"));
   aRes->AddFils(::ToXMLTree(std::string("Name"),anObj.Name())->ReTagThis("Name"));
   aRes->AddFils(::ToXMLTree(std::string("ScaleNuage"),anObj.ScaleNuage())->ReTagThis("ScaleNuage"));
   aRes->AddFils(::ToXMLTree(std::string("NameNuage"),anObj.NameNuage())->ReTagThis("NameNuage"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cDoImageBSurH & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.Dyn(),aTree->Get("Dyn",1),double(1e-2)); //tototo 

   xml_init(anObj.Offset(),aTree->Get("Offset",1),double(0)); //tototo 

   xml_init(anObj.SeuilMasqExport(),aTree->Get("SeuilMasqExport",1)); //tototo 

   xml_init(anObj.Name(),aTree->Get("Name",1)); //tototo 

   xml_init(anObj.ScaleNuage(),aTree->Get("ScaleNuage",1)); //tototo 

   xml_init(anObj.NameNuage(),aTree->Get("NameNuage",1)); //tototo 
}

std::string  Mangling( cDoImageBSurH *) {return "4EC741069F468E89FE3F";};


bool & cDoStatResult::DoRatio2Im()
{
   return mDoRatio2Im;
}

const bool & cDoStatResult::DoRatio2Im()const 
{
   return mDoRatio2Im;
}

void  BinaryUnDumpFromFile(cDoStatResult & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.DoRatio2Im(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cDoStatResult & anObj)
{
    BinaryDumpInFile(aFp,anObj.DoRatio2Im());
}

cElXMLTree * ToXMLTree(const cDoStatResult & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"DoStatResult",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("DoRatio2Im"),anObj.DoRatio2Im())->ReTagThis("DoRatio2Im"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cDoStatResult & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.DoRatio2Im(),aTree->Get("DoRatio2Im",1)); //tototo 
}

std::string  Mangling( cDoStatResult *) {return "7990A80AD7C98C9EFE3F";};


cElRegex_Ptr & cMasqOfEtape::PatternApply()
{
   return mPatternApply;
}

const cElRegex_Ptr & cMasqOfEtape::PatternApply()const 
{
   return mPatternApply;
}


cTplValGesInit< Box2dr > & cMasqOfEtape::RectInclus()
{
   return mRectInclus;
}

const cTplValGesInit< Box2dr > & cMasqOfEtape::RectInclus()const 
{
   return mRectInclus;
}

void  BinaryUnDumpFromFile(cMasqOfEtape & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.PatternApply(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.RectInclus().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.RectInclus().ValForcedForUnUmp(),aFp);
        }
        else  anObj.RectInclus().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cMasqOfEtape & anObj)
{
    BinaryDumpInFile(aFp,anObj.PatternApply());
    BinaryDumpInFile(aFp,anObj.RectInclus().IsInit());
    if (anObj.RectInclus().IsInit()) BinaryDumpInFile(aFp,anObj.RectInclus().Val());
}

cElXMLTree * ToXMLTree(const cMasqOfEtape & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"MasqOfEtape",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("PatternApply"),anObj.PatternApply())->ReTagThis("PatternApply"));
   if (anObj.RectInclus().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("RectInclus"),anObj.RectInclus().Val())->ReTagThis("RectInclus"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cMasqOfEtape & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.PatternApply(),aTree->Get("PatternApply",1)); //tototo 

   xml_init(anObj.RectInclus(),aTree->Get("RectInclus",1)); //tototo 
}

std::string  Mangling( cMasqOfEtape *) {return "28265BFCF994E1D5FE3F";};


cTplValGesInit< std::vector<double> > & cEtapeProgDyn::Px1MultRegul()
{
   return mPx1MultRegul;
}

const cTplValGesInit< std::vector<double> > & cEtapeProgDyn::Px1MultRegul()const 
{
   return mPx1MultRegul;
}


cTplValGesInit< std::vector<double> > & cEtapeProgDyn::Px2MultRegul()
{
   return mPx2MultRegul;
}

const cTplValGesInit< std::vector<double> > & cEtapeProgDyn::Px2MultRegul()const 
{
   return mPx2MultRegul;
}


cTplValGesInit< int > & cEtapeProgDyn::NbDir()
{
   return mNbDir;
}

const cTplValGesInit< int > & cEtapeProgDyn::NbDir()const 
{
   return mNbDir;
}


eModeAggregProgDyn & cEtapeProgDyn::ModeAgreg()
{
   return mModeAgreg;
}

const eModeAggregProgDyn & cEtapeProgDyn::ModeAgreg()const 
{
   return mModeAgreg;
}


cTplValGesInit< double > & cEtapeProgDyn::Teta0()
{
   return mTeta0;
}

const cTplValGesInit< double > & cEtapeProgDyn::Teta0()const 
{
   return mTeta0;
}

void  BinaryUnDumpFromFile(cEtapeProgDyn & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Px1MultRegul().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Px1MultRegul().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Px1MultRegul().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Px2MultRegul().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Px2MultRegul().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Px2MultRegul().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NbDir().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NbDir().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NbDir().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.ModeAgreg(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Teta0().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Teta0().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Teta0().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cEtapeProgDyn & anObj)
{
    BinaryDumpInFile(aFp,anObj.Px1MultRegul().IsInit());
    if (anObj.Px1MultRegul().IsInit()) BinaryDumpInFile(aFp,anObj.Px1MultRegul().Val());
    BinaryDumpInFile(aFp,anObj.Px2MultRegul().IsInit());
    if (anObj.Px2MultRegul().IsInit()) BinaryDumpInFile(aFp,anObj.Px2MultRegul().Val());
    BinaryDumpInFile(aFp,anObj.NbDir().IsInit());
    if (anObj.NbDir().IsInit()) BinaryDumpInFile(aFp,anObj.NbDir().Val());
    BinaryDumpInFile(aFp,anObj.ModeAgreg());
    BinaryDumpInFile(aFp,anObj.Teta0().IsInit());
    if (anObj.Teta0().IsInit()) BinaryDumpInFile(aFp,anObj.Teta0().Val());
}

cElXMLTree * ToXMLTree(const cEtapeProgDyn & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"EtapeProgDyn",eXMLBranche);
   if (anObj.Px1MultRegul().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Px1MultRegul"),anObj.Px1MultRegul().Val())->ReTagThis("Px1MultRegul"));
   if (anObj.Px2MultRegul().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Px2MultRegul"),anObj.Px2MultRegul().Val())->ReTagThis("Px2MultRegul"));
   if (anObj.NbDir().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NbDir"),anObj.NbDir().Val())->ReTagThis("NbDir"));
   aRes->AddFils(ToXMLTree(std::string("ModeAgreg"),anObj.ModeAgreg())->ReTagThis("ModeAgreg"));
   if (anObj.Teta0().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Teta0"),anObj.Teta0().Val())->ReTagThis("Teta0"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cEtapeProgDyn & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.Px1MultRegul(),aTree->Get("Px1MultRegul",1)); //tototo 

   xml_init(anObj.Px2MultRegul(),aTree->Get("Px2MultRegul",1)); //tototo 

   xml_init(anObj.NbDir(),aTree->Get("NbDir",1),int(2)); //tototo 

   xml_init(anObj.ModeAgreg(),aTree->Get("ModeAgreg",1)); //tototo 

   xml_init(anObj.Teta0(),aTree->Get("Teta0",1),double(0.0)); //tototo 
}

std::string  Mangling( cEtapeProgDyn *) {return "D45D75EFA13318C5FDBF";};


double & cEtiqBestImage::CostChangeEtiq()
{
   return mCostChangeEtiq;
}

const double & cEtiqBestImage::CostChangeEtiq()const 
{
   return mCostChangeEtiq;
}


cTplValGesInit< bool > & cEtiqBestImage::SauvEtiq()
{
   return mSauvEtiq;
}

const cTplValGesInit< bool > & cEtiqBestImage::SauvEtiq()const 
{
   return mSauvEtiq;
}

void  BinaryUnDumpFromFile(cEtiqBestImage & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.CostChangeEtiq(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SauvEtiq().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SauvEtiq().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SauvEtiq().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cEtiqBestImage & anObj)
{
    BinaryDumpInFile(aFp,anObj.CostChangeEtiq());
    BinaryDumpInFile(aFp,anObj.SauvEtiq().IsInit());
    if (anObj.SauvEtiq().IsInit()) BinaryDumpInFile(aFp,anObj.SauvEtiq().Val());
}

cElXMLTree * ToXMLTree(const cEtiqBestImage & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"EtiqBestImage",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("CostChangeEtiq"),anObj.CostChangeEtiq())->ReTagThis("CostChangeEtiq"));
   if (anObj.SauvEtiq().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SauvEtiq"),anObj.SauvEtiq().Val())->ReTagThis("SauvEtiq"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cEtiqBestImage & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.CostChangeEtiq(),aTree->Get("CostChangeEtiq",1)); //tototo 

   xml_init(anObj.SauvEtiq(),aTree->Get("SauvEtiq",1),bool(false)); //tototo 
}

std::string  Mangling( cEtiqBestImage *) {return "8CF7F4A86723FBA4FF3F";};


double & cArgMaskAuto::ValDefCorrel()
{
   return mValDefCorrel;
}

const double & cArgMaskAuto::ValDefCorrel()const 
{
   return mValDefCorrel;
}


double & cArgMaskAuto::CostTrans()
{
   return mCostTrans;
}

const double & cArgMaskAuto::CostTrans()const 
{
   return mCostTrans;
}


cTplValGesInit< bool > & cArgMaskAuto::ReInjectMask()
{
   return mReInjectMask;
}

const cTplValGesInit< bool > & cArgMaskAuto::ReInjectMask()const 
{
   return mReInjectMask;
}


cTplValGesInit< double > & cArgMaskAuto::AmplKLPostTr()
{
   return mAmplKLPostTr;
}

const cTplValGesInit< double > & cArgMaskAuto::AmplKLPostTr()const 
{
   return mAmplKLPostTr;
}


cTplValGesInit< int > & cArgMaskAuto::Erod32Mask()
{
   return mErod32Mask;
}

const cTplValGesInit< int > & cArgMaskAuto::Erod32Mask()const 
{
   return mErod32Mask;
}


cTplValGesInit< int > & cArgMaskAuto::SzOpen32()
{
   return mSzOpen32;
}

const cTplValGesInit< int > & cArgMaskAuto::SzOpen32()const 
{
   return mSzOpen32;
}


cTplValGesInit< int > & cArgMaskAuto::SeuilZC()
{
   return mSeuilZC;
}

const cTplValGesInit< int > & cArgMaskAuto::SeuilZC()const 
{
   return mSeuilZC;
}


double & cArgMaskAuto::CostChangeEtiq()
{
   return EtiqBestImage().Val().CostChangeEtiq();
}

const double & cArgMaskAuto::CostChangeEtiq()const 
{
   return EtiqBestImage().Val().CostChangeEtiq();
}


cTplValGesInit< bool > & cArgMaskAuto::SauvEtiq()
{
   return EtiqBestImage().Val().SauvEtiq();
}

const cTplValGesInit< bool > & cArgMaskAuto::SauvEtiq()const 
{
   return EtiqBestImage().Val().SauvEtiq();
}


cTplValGesInit< cEtiqBestImage > & cArgMaskAuto::EtiqBestImage()
{
   return mEtiqBestImage;
}

const cTplValGesInit< cEtiqBestImage > & cArgMaskAuto::EtiqBestImage()const 
{
   return mEtiqBestImage;
}

void  BinaryUnDumpFromFile(cArgMaskAuto & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.ValDefCorrel(),aFp);
    BinaryUnDumpFromFile(anObj.CostTrans(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ReInjectMask().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ReInjectMask().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ReInjectMask().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.AmplKLPostTr().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.AmplKLPostTr().ValForcedForUnUmp(),aFp);
        }
        else  anObj.AmplKLPostTr().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Erod32Mask().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Erod32Mask().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Erod32Mask().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SzOpen32().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SzOpen32().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SzOpen32().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SeuilZC().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SeuilZC().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SeuilZC().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.EtiqBestImage().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.EtiqBestImage().ValForcedForUnUmp(),aFp);
        }
        else  anObj.EtiqBestImage().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cArgMaskAuto & anObj)
{
    BinaryDumpInFile(aFp,anObj.ValDefCorrel());
    BinaryDumpInFile(aFp,anObj.CostTrans());
    BinaryDumpInFile(aFp,anObj.ReInjectMask().IsInit());
    if (anObj.ReInjectMask().IsInit()) BinaryDumpInFile(aFp,anObj.ReInjectMask().Val());
    BinaryDumpInFile(aFp,anObj.AmplKLPostTr().IsInit());
    if (anObj.AmplKLPostTr().IsInit()) BinaryDumpInFile(aFp,anObj.AmplKLPostTr().Val());
    BinaryDumpInFile(aFp,anObj.Erod32Mask().IsInit());
    if (anObj.Erod32Mask().IsInit()) BinaryDumpInFile(aFp,anObj.Erod32Mask().Val());
    BinaryDumpInFile(aFp,anObj.SzOpen32().IsInit());
    if (anObj.SzOpen32().IsInit()) BinaryDumpInFile(aFp,anObj.SzOpen32().Val());
    BinaryDumpInFile(aFp,anObj.SeuilZC().IsInit());
    if (anObj.SeuilZC().IsInit()) BinaryDumpInFile(aFp,anObj.SeuilZC().Val());
    BinaryDumpInFile(aFp,anObj.EtiqBestImage().IsInit());
    if (anObj.EtiqBestImage().IsInit()) BinaryDumpInFile(aFp,anObj.EtiqBestImage().Val());
}

cElXMLTree * ToXMLTree(const cArgMaskAuto & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ArgMaskAuto",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("ValDefCorrel"),anObj.ValDefCorrel())->ReTagThis("ValDefCorrel"));
   aRes->AddFils(::ToXMLTree(std::string("CostTrans"),anObj.CostTrans())->ReTagThis("CostTrans"));
   if (anObj.ReInjectMask().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ReInjectMask"),anObj.ReInjectMask().Val())->ReTagThis("ReInjectMask"));
   if (anObj.AmplKLPostTr().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("AmplKLPostTr"),anObj.AmplKLPostTr().Val())->ReTagThis("AmplKLPostTr"));
   if (anObj.Erod32Mask().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Erod32Mask"),anObj.Erod32Mask().Val())->ReTagThis("Erod32Mask"));
   if (anObj.SzOpen32().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SzOpen32"),anObj.SzOpen32().Val())->ReTagThis("SzOpen32"));
   if (anObj.SeuilZC().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SeuilZC"),anObj.SeuilZC().Val())->ReTagThis("SeuilZC"));
   if (anObj.EtiqBestImage().IsInit())
      aRes->AddFils(ToXMLTree(anObj.EtiqBestImage().Val())->ReTagThis("EtiqBestImage"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cArgMaskAuto & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.ValDefCorrel(),aTree->Get("ValDefCorrel",1)); //tototo 

   xml_init(anObj.CostTrans(),aTree->Get("CostTrans",1)); //tototo 

   xml_init(anObj.ReInjectMask(),aTree->Get("ReInjectMask",1),bool(true)); //tototo 

   xml_init(anObj.AmplKLPostTr(),aTree->Get("AmplKLPostTr",1),double(100.0)); //tototo 

   xml_init(anObj.Erod32Mask(),aTree->Get("Erod32Mask",1)); //tototo 

   xml_init(anObj.SzOpen32(),aTree->Get("SzOpen32",1),int(9)); //tototo 

   xml_init(anObj.SeuilZC(),aTree->Get("SeuilZC",1),int(200)); //tototo 

   xml_init(anObj.EtiqBestImage(),aTree->Get("EtiqBestImage",1)); //tototo 
}

std::string  Mangling( cArgMaskAuto *) {return "5EFCBB6806FDC698FE3F";};


std::list< cEtapeProgDyn > & cModulationProgDyn::EtapeProgDyn()
{
   return mEtapeProgDyn;
}

const std::list< cEtapeProgDyn > & cModulationProgDyn::EtapeProgDyn()const 
{
   return mEtapeProgDyn;
}


cTplValGesInit< double > & cModulationProgDyn::Px1PenteMax()
{
   return mPx1PenteMax;
}

const cTplValGesInit< double > & cModulationProgDyn::Px1PenteMax()const 
{
   return mPx1PenteMax;
}


cTplValGesInit< double > & cModulationProgDyn::Px2PenteMax()
{
   return mPx2PenteMax;
}

const cTplValGesInit< double > & cModulationProgDyn::Px2PenteMax()const 
{
   return mPx2PenteMax;
}


cTplValGesInit< bool > & cModulationProgDyn::ChoixNewProg()
{
   return mChoixNewProg;
}

const cTplValGesInit< bool > & cModulationProgDyn::ChoixNewProg()const 
{
   return mChoixNewProg;
}


double & cModulationProgDyn::ValDefCorrel()
{
   return ArgMaskAuto().Val().ValDefCorrel();
}

const double & cModulationProgDyn::ValDefCorrel()const 
{
   return ArgMaskAuto().Val().ValDefCorrel();
}


double & cModulationProgDyn::CostTrans()
{
   return ArgMaskAuto().Val().CostTrans();
}

const double & cModulationProgDyn::CostTrans()const 
{
   return ArgMaskAuto().Val().CostTrans();
}


cTplValGesInit< bool > & cModulationProgDyn::ReInjectMask()
{
   return ArgMaskAuto().Val().ReInjectMask();
}

const cTplValGesInit< bool > & cModulationProgDyn::ReInjectMask()const 
{
   return ArgMaskAuto().Val().ReInjectMask();
}


cTplValGesInit< double > & cModulationProgDyn::AmplKLPostTr()
{
   return ArgMaskAuto().Val().AmplKLPostTr();
}

const cTplValGesInit< double > & cModulationProgDyn::AmplKLPostTr()const 
{
   return ArgMaskAuto().Val().AmplKLPostTr();
}


cTplValGesInit< int > & cModulationProgDyn::Erod32Mask()
{
   return ArgMaskAuto().Val().Erod32Mask();
}

const cTplValGesInit< int > & cModulationProgDyn::Erod32Mask()const 
{
   return ArgMaskAuto().Val().Erod32Mask();
}


cTplValGesInit< int > & cModulationProgDyn::SzOpen32()
{
   return ArgMaskAuto().Val().SzOpen32();
}

const cTplValGesInit< int > & cModulationProgDyn::SzOpen32()const 
{
   return ArgMaskAuto().Val().SzOpen32();
}


cTplValGesInit< int > & cModulationProgDyn::SeuilZC()
{
   return ArgMaskAuto().Val().SeuilZC();
}

const cTplValGesInit< int > & cModulationProgDyn::SeuilZC()const 
{
   return ArgMaskAuto().Val().SeuilZC();
}


double & cModulationProgDyn::CostChangeEtiq()
{
   return ArgMaskAuto().Val().EtiqBestImage().Val().CostChangeEtiq();
}

const double & cModulationProgDyn::CostChangeEtiq()const 
{
   return ArgMaskAuto().Val().EtiqBestImage().Val().CostChangeEtiq();
}


cTplValGesInit< bool > & cModulationProgDyn::SauvEtiq()
{
   return ArgMaskAuto().Val().EtiqBestImage().Val().SauvEtiq();
}

const cTplValGesInit< bool > & cModulationProgDyn::SauvEtiq()const 
{
   return ArgMaskAuto().Val().EtiqBestImage().Val().SauvEtiq();
}


cTplValGesInit< cEtiqBestImage > & cModulationProgDyn::EtiqBestImage()
{
   return ArgMaskAuto().Val().EtiqBestImage();
}

const cTplValGesInit< cEtiqBestImage > & cModulationProgDyn::EtiqBestImage()const 
{
   return ArgMaskAuto().Val().EtiqBestImage();
}


cTplValGesInit< cArgMaskAuto > & cModulationProgDyn::ArgMaskAuto()
{
   return mArgMaskAuto;
}

const cTplValGesInit< cArgMaskAuto > & cModulationProgDyn::ArgMaskAuto()const 
{
   return mArgMaskAuto;
}

void  BinaryUnDumpFromFile(cModulationProgDyn & anObj,ELISE_fp & aFp)
{
   { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cEtapeProgDyn aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.EtapeProgDyn().push_back(aVal);
        }
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Px1PenteMax().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Px1PenteMax().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Px1PenteMax().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Px2PenteMax().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Px2PenteMax().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Px2PenteMax().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ChoixNewProg().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ChoixNewProg().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ChoixNewProg().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ArgMaskAuto().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ArgMaskAuto().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ArgMaskAuto().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cModulationProgDyn & anObj)
{
    BinaryDumpInFile(aFp,(int)anObj.EtapeProgDyn().size());
    for(  std::list< cEtapeProgDyn >::const_iterator iT=anObj.EtapeProgDyn().begin();
         iT!=anObj.EtapeProgDyn().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.Px1PenteMax().IsInit());
    if (anObj.Px1PenteMax().IsInit()) BinaryDumpInFile(aFp,anObj.Px1PenteMax().Val());
    BinaryDumpInFile(aFp,anObj.Px2PenteMax().IsInit());
    if (anObj.Px2PenteMax().IsInit()) BinaryDumpInFile(aFp,anObj.Px2PenteMax().Val());
    BinaryDumpInFile(aFp,anObj.ChoixNewProg().IsInit());
    if (anObj.ChoixNewProg().IsInit()) BinaryDumpInFile(aFp,anObj.ChoixNewProg().Val());
    BinaryDumpInFile(aFp,anObj.ArgMaskAuto().IsInit());
    if (anObj.ArgMaskAuto().IsInit()) BinaryDumpInFile(aFp,anObj.ArgMaskAuto().Val());
}

cElXMLTree * ToXMLTree(const cModulationProgDyn & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ModulationProgDyn",eXMLBranche);
  for
  (       std::list< cEtapeProgDyn >::const_iterator it=anObj.EtapeProgDyn().begin();
      it !=anObj.EtapeProgDyn().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("EtapeProgDyn"));
   if (anObj.Px1PenteMax().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Px1PenteMax"),anObj.Px1PenteMax().Val())->ReTagThis("Px1PenteMax"));
   if (anObj.Px2PenteMax().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Px2PenteMax"),anObj.Px2PenteMax().Val())->ReTagThis("Px2PenteMax"));
   if (anObj.ChoixNewProg().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ChoixNewProg"),anObj.ChoixNewProg().Val())->ReTagThis("ChoixNewProg"));
   if (anObj.ArgMaskAuto().IsInit())
      aRes->AddFils(ToXMLTree(anObj.ArgMaskAuto().Val())->ReTagThis("ArgMaskAuto"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cModulationProgDyn & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.EtapeProgDyn(),aTree->GetAll("EtapeProgDyn",false,1));

   xml_init(anObj.Px1PenteMax(),aTree->Get("Px1PenteMax",1),double(10.0)); //tototo 

   xml_init(anObj.Px2PenteMax(),aTree->Get("Px2PenteMax",1),double(10.0)); //tototo 

   xml_init(anObj.ChoixNewProg(),aTree->Get("ChoixNewProg",1)); //tototo 

   xml_init(anObj.ArgMaskAuto(),aTree->Get("ArgMaskAuto",1)); //tototo 
}

std::string  Mangling( cModulationProgDyn *) {return "0BEDDDA944673CA6FE3F";};


std::list< cSpecFitrageImage > & cPostFiltragePx::OneFitragePx()
{
   return mOneFitragePx;
}

const std::list< cSpecFitrageImage > & cPostFiltragePx::OneFitragePx()const 
{
   return mOneFitragePx;
}

void  BinaryUnDumpFromFile(cPostFiltragePx & anObj,ELISE_fp & aFp)
{
   { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cSpecFitrageImage aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.OneFitragePx().push_back(aVal);
        }
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cPostFiltragePx & anObj)
{
    BinaryDumpInFile(aFp,(int)anObj.OneFitragePx().size());
    for(  std::list< cSpecFitrageImage >::const_iterator iT=anObj.OneFitragePx().begin();
         iT!=anObj.OneFitragePx().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
}

cElXMLTree * ToXMLTree(const cPostFiltragePx & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"PostFiltragePx",eXMLBranche);
  for
  (       std::list< cSpecFitrageImage >::const_iterator it=anObj.OneFitragePx().begin();
      it !=anObj.OneFitragePx().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("OneFitragePx"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cPostFiltragePx & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.OneFitragePx(),aTree->GetAll("OneFitragePx",false,1));
}

std::string  Mangling( cPostFiltragePx *) {return "7206265CB6FE31F7FD3F";};


double & cPostFiltrageDiscont::SzFiltre()
{
   return mSzFiltre;
}

const double & cPostFiltrageDiscont::SzFiltre()const 
{
   return mSzFiltre;
}


cTplValGesInit< int > & cPostFiltrageDiscont::NbIter()
{
   return mNbIter;
}

const cTplValGesInit< int > & cPostFiltrageDiscont::NbIter()const 
{
   return mNbIter;
}


cTplValGesInit< double > & cPostFiltrageDiscont::ExposPonderGrad()
{
   return mExposPonderGrad;
}

const cTplValGesInit< double > & cPostFiltrageDiscont::ExposPonderGrad()const 
{
   return mExposPonderGrad;
}


cTplValGesInit< double > & cPostFiltrageDiscont::DericheFactEPC()
{
   return mDericheFactEPC;
}

const cTplValGesInit< double > & cPostFiltrageDiscont::DericheFactEPC()const 
{
   return mDericheFactEPC;
}


cTplValGesInit< double > & cPostFiltrageDiscont::ValGradAtten()
{
   return mValGradAtten;
}

const cTplValGesInit< double > & cPostFiltrageDiscont::ValGradAtten()const 
{
   return mValGradAtten;
}


cTplValGesInit< double > & cPostFiltrageDiscont::ExposPonderCorr()
{
   return mExposPonderCorr;
}

const cTplValGesInit< double > & cPostFiltrageDiscont::ExposPonderCorr()const 
{
   return mExposPonderCorr;
}

void  BinaryUnDumpFromFile(cPostFiltrageDiscont & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.SzFiltre(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NbIter().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NbIter().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NbIter().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ExposPonderGrad().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ExposPonderGrad().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ExposPonderGrad().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.DericheFactEPC().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.DericheFactEPC().ValForcedForUnUmp(),aFp);
        }
        else  anObj.DericheFactEPC().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ValGradAtten().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ValGradAtten().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ValGradAtten().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ExposPonderCorr().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ExposPonderCorr().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ExposPonderCorr().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cPostFiltrageDiscont & anObj)
{
    BinaryDumpInFile(aFp,anObj.SzFiltre());
    BinaryDumpInFile(aFp,anObj.NbIter().IsInit());
    if (anObj.NbIter().IsInit()) BinaryDumpInFile(aFp,anObj.NbIter().Val());
    BinaryDumpInFile(aFp,anObj.ExposPonderGrad().IsInit());
    if (anObj.ExposPonderGrad().IsInit()) BinaryDumpInFile(aFp,anObj.ExposPonderGrad().Val());
    BinaryDumpInFile(aFp,anObj.DericheFactEPC().IsInit());
    if (anObj.DericheFactEPC().IsInit()) BinaryDumpInFile(aFp,anObj.DericheFactEPC().Val());
    BinaryDumpInFile(aFp,anObj.ValGradAtten().IsInit());
    if (anObj.ValGradAtten().IsInit()) BinaryDumpInFile(aFp,anObj.ValGradAtten().Val());
    BinaryDumpInFile(aFp,anObj.ExposPonderCorr().IsInit());
    if (anObj.ExposPonderCorr().IsInit()) BinaryDumpInFile(aFp,anObj.ExposPonderCorr().Val());
}

cElXMLTree * ToXMLTree(const cPostFiltrageDiscont & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"PostFiltrageDiscont",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("SzFiltre"),anObj.SzFiltre())->ReTagThis("SzFiltre"));
   if (anObj.NbIter().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NbIter"),anObj.NbIter().Val())->ReTagThis("NbIter"));
   if (anObj.ExposPonderGrad().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ExposPonderGrad"),anObj.ExposPonderGrad().Val())->ReTagThis("ExposPonderGrad"));
   if (anObj.DericheFactEPC().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DericheFactEPC"),anObj.DericheFactEPC().Val())->ReTagThis("DericheFactEPC"));
   if (anObj.ValGradAtten().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ValGradAtten"),anObj.ValGradAtten().Val())->ReTagThis("ValGradAtten"));
   if (anObj.ExposPonderCorr().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ExposPonderCorr"),anObj.ExposPonderCorr().Val())->ReTagThis("ExposPonderCorr"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cPostFiltrageDiscont & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.SzFiltre(),aTree->Get("SzFiltre",1)); //tototo 

   xml_init(anObj.NbIter(),aTree->Get("NbIter",1),int(2)); //tototo 

   xml_init(anObj.ExposPonderGrad(),aTree->Get("ExposPonderGrad",1),double(2.0)); //tototo 

   xml_init(anObj.DericheFactEPC(),aTree->Get("DericheFactEPC",1),double(1.0)); //tototo 

   xml_init(anObj.ValGradAtten(),aTree->Get("ValGradAtten",1)); //tototo 

   xml_init(anObj.ExposPonderCorr(),aTree->Get("ExposPonderCorr",1)); //tototo 
}

std::string  Mangling( cPostFiltrageDiscont *) {return "9A9B4D0EE5BB9883FF3F";};


bool & cImageSelecteur::ModeExclusion()
{
   return mModeExclusion;
}

const bool & cImageSelecteur::ModeExclusion()const 
{
   return mModeExclusion;
}


std::list< std::string > & cImageSelecteur::PatternSel()
{
   return mPatternSel;
}

const std::list< std::string > & cImageSelecteur::PatternSel()const 
{
   return mPatternSel;
}

void  BinaryUnDumpFromFile(cImageSelecteur & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.ModeExclusion(),aFp);
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             std::string aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.PatternSel().push_back(aVal);
        }
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cImageSelecteur & anObj)
{
    BinaryDumpInFile(aFp,anObj.ModeExclusion());
    BinaryDumpInFile(aFp,(int)anObj.PatternSel().size());
    for(  std::list< std::string >::const_iterator iT=anObj.PatternSel().begin();
         iT!=anObj.PatternSel().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
}

cElXMLTree * ToXMLTree(const cImageSelecteur & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ImageSelecteur",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("ModeExclusion"),anObj.ModeExclusion())->ReTagThis("ModeExclusion"));
  for
  (       std::list< std::string >::const_iterator it=anObj.PatternSel().begin();
      it !=anObj.PatternSel().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("PatternSel"),(*it))->ReTagThis("PatternSel"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cImageSelecteur & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.ModeExclusion(),aTree->Get("ModeExclusion",1)); //tototo 

   xml_init(anObj.PatternSel(),aTree->GetAll("PatternSel",false,1));
}

std::string  Mangling( cImageSelecteur *) {return "9E79EC740088489FFCBF";};


std::string & cGenerateImageRedr::FCND_CalcRedr()
{
   return mFCND_CalcRedr;
}

const std::string & cGenerateImageRedr::FCND_CalcRedr()const 
{
   return mFCND_CalcRedr;
}


cTplValGesInit< eTypeNumerique > & cGenerateImageRedr::Type()
{
   return mType;
}

const cTplValGesInit< eTypeNumerique > & cGenerateImageRedr::Type()const 
{
   return mType;
}

void  BinaryUnDumpFromFile(cGenerateImageRedr & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.FCND_CalcRedr(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Type().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Type().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Type().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cGenerateImageRedr & anObj)
{
    BinaryDumpInFile(aFp,anObj.FCND_CalcRedr());
    BinaryDumpInFile(aFp,anObj.Type().IsInit());
    if (anObj.Type().IsInit()) BinaryDumpInFile(aFp,anObj.Type().Val());
}

cElXMLTree * ToXMLTree(const cGenerateImageRedr & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"GenerateImageRedr",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("FCND_CalcRedr"),anObj.FCND_CalcRedr())->ReTagThis("FCND_CalcRedr"));
   if (anObj.Type().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Type"),anObj.Type().Val())->ReTagThis("Type"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cGenerateImageRedr & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.FCND_CalcRedr(),aTree->Get("FCND_CalcRedr",1)); //tototo 

   xml_init(anObj.Type(),aTree->Get("Type",1)); //tototo 
}

std::string  Mangling( cGenerateImageRedr *) {return "60C25B2D94CABED4F93F";};


std::list< int > & cGenerateProjectionInImages::NumsImageDontApply()
{
   return mNumsImageDontApply;
}

const std::list< int > & cGenerateProjectionInImages::NumsImageDontApply()const 
{
   return mNumsImageDontApply;
}


std::string & cGenerateProjectionInImages::FCND_CalcProj()
{
   return mFCND_CalcProj;
}

const std::string & cGenerateProjectionInImages::FCND_CalcProj()const 
{
   return mFCND_CalcProj;
}


cTplValGesInit< bool > & cGenerateProjectionInImages::SubsXY()
{
   return mSubsXY;
}

const cTplValGesInit< bool > & cGenerateProjectionInImages::SubsXY()const 
{
   return mSubsXY;
}


cTplValGesInit< bool > & cGenerateProjectionInImages::Polar()
{
   return mPolar;
}

const cTplValGesInit< bool > & cGenerateProjectionInImages::Polar()const 
{
   return mPolar;
}


std::string & cGenerateProjectionInImages::FCND_CalcRedr()
{
   return GenerateImageRedr().Val().FCND_CalcRedr();
}

const std::string & cGenerateProjectionInImages::FCND_CalcRedr()const 
{
   return GenerateImageRedr().Val().FCND_CalcRedr();
}


cTplValGesInit< eTypeNumerique > & cGenerateProjectionInImages::Type()
{
   return GenerateImageRedr().Val().Type();
}

const cTplValGesInit< eTypeNumerique > & cGenerateProjectionInImages::Type()const 
{
   return GenerateImageRedr().Val().Type();
}


cTplValGesInit< cGenerateImageRedr > & cGenerateProjectionInImages::GenerateImageRedr()
{
   return mGenerateImageRedr;
}

const cTplValGesInit< cGenerateImageRedr > & cGenerateProjectionInImages::GenerateImageRedr()const 
{
   return mGenerateImageRedr;
}

void  BinaryUnDumpFromFile(cGenerateProjectionInImages & anObj,ELISE_fp & aFp)
{
   { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             int aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.NumsImageDontApply().push_back(aVal);
        }
  } ;
    BinaryUnDumpFromFile(anObj.FCND_CalcProj(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SubsXY().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SubsXY().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SubsXY().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Polar().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Polar().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Polar().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.GenerateImageRedr().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.GenerateImageRedr().ValForcedForUnUmp(),aFp);
        }
        else  anObj.GenerateImageRedr().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cGenerateProjectionInImages & anObj)
{
    BinaryDumpInFile(aFp,(int)anObj.NumsImageDontApply().size());
    for(  std::list< int >::const_iterator iT=anObj.NumsImageDontApply().begin();
         iT!=anObj.NumsImageDontApply().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.FCND_CalcProj());
    BinaryDumpInFile(aFp,anObj.SubsXY().IsInit());
    if (anObj.SubsXY().IsInit()) BinaryDumpInFile(aFp,anObj.SubsXY().Val());
    BinaryDumpInFile(aFp,anObj.Polar().IsInit());
    if (anObj.Polar().IsInit()) BinaryDumpInFile(aFp,anObj.Polar().Val());
    BinaryDumpInFile(aFp,anObj.GenerateImageRedr().IsInit());
    if (anObj.GenerateImageRedr().IsInit()) BinaryDumpInFile(aFp,anObj.GenerateImageRedr().Val());
}

cElXMLTree * ToXMLTree(const cGenerateProjectionInImages & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"GenerateProjectionInImages",eXMLBranche);
  for
  (       std::list< int >::const_iterator it=anObj.NumsImageDontApply().begin();
      it !=anObj.NumsImageDontApply().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("NumsImageDontApply"),(*it))->ReTagThis("NumsImageDontApply"));
   aRes->AddFils(::ToXMLTree(std::string("FCND_CalcProj"),anObj.FCND_CalcProj())->ReTagThis("FCND_CalcProj"));
   if (anObj.SubsXY().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SubsXY"),anObj.SubsXY().Val())->ReTagThis("SubsXY"));
   if (anObj.Polar().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Polar"),anObj.Polar().Val())->ReTagThis("Polar"));
   if (anObj.GenerateImageRedr().IsInit())
      aRes->AddFils(ToXMLTree(anObj.GenerateImageRedr().Val())->ReTagThis("GenerateImageRedr"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cGenerateProjectionInImages & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.NumsImageDontApply(),aTree->GetAll("NumsImageDontApply",false,1));

   xml_init(anObj.FCND_CalcProj(),aTree->Get("FCND_CalcProj",1)); //tototo 

   xml_init(anObj.SubsXY(),aTree->Get("SubsXY",1),bool(false)); //tototo 

   xml_init(anObj.Polar(),aTree->Get("Polar",1),bool(false)); //tototo 

   xml_init(anObj.GenerateImageRedr(),aTree->Get("GenerateImageRedr",1)); //tototo 
}

std::string  Mangling( cGenerateProjectionInImages *) {return "9010CCC95B5C80A6FF3F";};


double & cGenCorPxTransv::SsResolPx()
{
   return mSsResolPx;
}

const double & cGenCorPxTransv::SsResolPx()const 
{
   return mSsResolPx;
}


std::string & cGenCorPxTransv::NameXMLFile()
{
   return mNameXMLFile;
}

const std::string & cGenCorPxTransv::NameXMLFile()const 
{
   return mNameXMLFile;
}

void  BinaryUnDumpFromFile(cGenCorPxTransv & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.SsResolPx(),aFp);
    BinaryUnDumpFromFile(anObj.NameXMLFile(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cGenCorPxTransv & anObj)
{
    BinaryDumpInFile(aFp,anObj.SsResolPx());
    BinaryDumpInFile(aFp,anObj.NameXMLFile());
}

cElXMLTree * ToXMLTree(const cGenCorPxTransv & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"GenCorPxTransv",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("SsResolPx"),anObj.SsResolPx())->ReTagThis("SsResolPx"));
   aRes->AddFils(::ToXMLTree(std::string("NameXMLFile"),anObj.NameXMLFile())->ReTagThis("NameXMLFile"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cGenCorPxTransv & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.SsResolPx(),aTree->Get("SsResolPx",1)); //tototo 

   xml_init(anObj.NameXMLFile(),aTree->Get("NameXMLFile",1)); //tototo 
}

std::string  Mangling( cGenCorPxTransv *) {return "B0060CDC4FA05893FD3F";};


double & cSimulFrac::CoutFrac()
{
   return mCoutFrac;
}

const double & cSimulFrac::CoutFrac()const 
{
   return mCoutFrac;
}

void  BinaryUnDumpFromFile(cSimulFrac & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.CoutFrac(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cSimulFrac & anObj)
{
    BinaryDumpInFile(aFp,anObj.CoutFrac());
}

cElXMLTree * ToXMLTree(const cSimulFrac & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"SimulFrac",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("CoutFrac"),anObj.CoutFrac())->ReTagThis("CoutFrac"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cSimulFrac & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.CoutFrac(),aTree->Get("CoutFrac",1)); //tototo 
}

std::string  Mangling( cSimulFrac *) {return "B59B6D020DD43192FF3F";};


cTplValGesInit< bool > & cInterfaceVisualisation::VisuTerrainIm()
{
   return mVisuTerrainIm;
}

const cTplValGesInit< bool > & cInterfaceVisualisation::VisuTerrainIm()const 
{
   return mVisuTerrainIm;
}


cTplValGesInit< int > & cInterfaceVisualisation::SzWTerr()
{
   return mSzWTerr;
}

const cTplValGesInit< int > & cInterfaceVisualisation::SzWTerr()const 
{
   return mSzWTerr;
}


std::list< std::string > & cInterfaceVisualisation::UnSelectedImage()
{
   return mUnSelectedImage;
}

const std::list< std::string > & cInterfaceVisualisation::UnSelectedImage()const 
{
   return mUnSelectedImage;
}


Pt2di & cInterfaceVisualisation::CentreVisuTerrain()
{
   return mCentreVisuTerrain;
}

const Pt2di & cInterfaceVisualisation::CentreVisuTerrain()const 
{
   return mCentreVisuTerrain;
}


cTplValGesInit< int > & cInterfaceVisualisation::ZoomTerr()
{
   return mZoomTerr;
}

const cTplValGesInit< int > & cInterfaceVisualisation::ZoomTerr()const 
{
   return mZoomTerr;
}


cTplValGesInit< int > & cInterfaceVisualisation::NbDiscHistoPartieFrac()
{
   return mNbDiscHistoPartieFrac;
}

const cTplValGesInit< int > & cInterfaceVisualisation::NbDiscHistoPartieFrac()const 
{
   return mNbDiscHistoPartieFrac;
}


double & cInterfaceVisualisation::CoutFrac()
{
   return SimulFrac().Val().CoutFrac();
}

const double & cInterfaceVisualisation::CoutFrac()const 
{
   return SimulFrac().Val().CoutFrac();
}


cTplValGesInit< cSimulFrac > & cInterfaceVisualisation::SimulFrac()
{
   return mSimulFrac;
}

const cTplValGesInit< cSimulFrac > & cInterfaceVisualisation::SimulFrac()const 
{
   return mSimulFrac;
}

void  BinaryUnDumpFromFile(cInterfaceVisualisation & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.VisuTerrainIm().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.VisuTerrainIm().ValForcedForUnUmp(),aFp);
        }
        else  anObj.VisuTerrainIm().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SzWTerr().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SzWTerr().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SzWTerr().SetNoInit();
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             std::string aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.UnSelectedImage().push_back(aVal);
        }
  } ;
    BinaryUnDumpFromFile(anObj.CentreVisuTerrain(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ZoomTerr().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ZoomTerr().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ZoomTerr().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NbDiscHistoPartieFrac().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NbDiscHistoPartieFrac().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NbDiscHistoPartieFrac().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SimulFrac().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SimulFrac().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SimulFrac().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cInterfaceVisualisation & anObj)
{
    BinaryDumpInFile(aFp,anObj.VisuTerrainIm().IsInit());
    if (anObj.VisuTerrainIm().IsInit()) BinaryDumpInFile(aFp,anObj.VisuTerrainIm().Val());
    BinaryDumpInFile(aFp,anObj.SzWTerr().IsInit());
    if (anObj.SzWTerr().IsInit()) BinaryDumpInFile(aFp,anObj.SzWTerr().Val());
    BinaryDumpInFile(aFp,(int)anObj.UnSelectedImage().size());
    for(  std::list< std::string >::const_iterator iT=anObj.UnSelectedImage().begin();
         iT!=anObj.UnSelectedImage().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.CentreVisuTerrain());
    BinaryDumpInFile(aFp,anObj.ZoomTerr().IsInit());
    if (anObj.ZoomTerr().IsInit()) BinaryDumpInFile(aFp,anObj.ZoomTerr().Val());
    BinaryDumpInFile(aFp,anObj.NbDiscHistoPartieFrac().IsInit());
    if (anObj.NbDiscHistoPartieFrac().IsInit()) BinaryDumpInFile(aFp,anObj.NbDiscHistoPartieFrac().Val());
    BinaryDumpInFile(aFp,anObj.SimulFrac().IsInit());
    if (anObj.SimulFrac().IsInit()) BinaryDumpInFile(aFp,anObj.SimulFrac().Val());
}

cElXMLTree * ToXMLTree(const cInterfaceVisualisation & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"InterfaceVisualisation",eXMLBranche);
   if (anObj.VisuTerrainIm().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("VisuTerrainIm"),anObj.VisuTerrainIm().Val())->ReTagThis("VisuTerrainIm"));
   if (anObj.SzWTerr().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SzWTerr"),anObj.SzWTerr().Val())->ReTagThis("SzWTerr"));
  for
  (       std::list< std::string >::const_iterator it=anObj.UnSelectedImage().begin();
      it !=anObj.UnSelectedImage().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("UnSelectedImage"),(*it))->ReTagThis("UnSelectedImage"));
   aRes->AddFils(::ToXMLTree(std::string("CentreVisuTerrain"),anObj.CentreVisuTerrain())->ReTagThis("CentreVisuTerrain"));
   if (anObj.ZoomTerr().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ZoomTerr"),anObj.ZoomTerr().Val())->ReTagThis("ZoomTerr"));
   if (anObj.NbDiscHistoPartieFrac().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NbDiscHistoPartieFrac"),anObj.NbDiscHistoPartieFrac().Val())->ReTagThis("NbDiscHistoPartieFrac"));
   if (anObj.SimulFrac().IsInit())
      aRes->AddFils(ToXMLTree(anObj.SimulFrac().Val())->ReTagThis("SimulFrac"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cInterfaceVisualisation & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.VisuTerrainIm(),aTree->Get("VisuTerrainIm",1),bool(true)); //tototo 

   xml_init(anObj.SzWTerr(),aTree->Get("SzWTerr",1),int(100)); //tototo 

   xml_init(anObj.UnSelectedImage(),aTree->GetAll("UnSelectedImage",false,1));

   xml_init(anObj.CentreVisuTerrain(),aTree->Get("CentreVisuTerrain",1)); //tototo 

   xml_init(anObj.ZoomTerr(),aTree->Get("ZoomTerr",1),int(1)); //tototo 

   xml_init(anObj.NbDiscHistoPartieFrac(),aTree->Get("NbDiscHistoPartieFrac",1),int(-1)); //tototo 

   xml_init(anObj.SimulFrac(),aTree->Get("SimulFrac",1)); //tototo 
}

std::string  Mangling( cInterfaceVisualisation *) {return "0603D8357DFB7D82FF3F";};


cTplValGesInit< bool > & cMTD_Nuage_Maille::DataInside()
{
   return mDataInside;
}

const cTplValGesInit< bool > & cMTD_Nuage_Maille::DataInside()const 
{
   return mDataInside;
}


std::string & cMTD_Nuage_Maille::KeyNameMTD()
{
   return mKeyNameMTD;
}

const std::string & cMTD_Nuage_Maille::KeyNameMTD()const 
{
   return mKeyNameMTD;
}


cTplValGesInit< double > & cMTD_Nuage_Maille::RatioPseudoConik()
{
   return mRatioPseudoConik;
}

const cTplValGesInit< double > & cMTD_Nuage_Maille::RatioPseudoConik()const 
{
   return mRatioPseudoConik;
}

void  BinaryUnDumpFromFile(cMTD_Nuage_Maille & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.DataInside().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.DataInside().ValForcedForUnUmp(),aFp);
        }
        else  anObj.DataInside().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.KeyNameMTD(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.RatioPseudoConik().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.RatioPseudoConik().ValForcedForUnUmp(),aFp);
        }
        else  anObj.RatioPseudoConik().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cMTD_Nuage_Maille & anObj)
{
    BinaryDumpInFile(aFp,anObj.DataInside().IsInit());
    if (anObj.DataInside().IsInit()) BinaryDumpInFile(aFp,anObj.DataInside().Val());
    BinaryDumpInFile(aFp,anObj.KeyNameMTD());
    BinaryDumpInFile(aFp,anObj.RatioPseudoConik().IsInit());
    if (anObj.RatioPseudoConik().IsInit()) BinaryDumpInFile(aFp,anObj.RatioPseudoConik().Val());
}

cElXMLTree * ToXMLTree(const cMTD_Nuage_Maille & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"MTD_Nuage_Maille",eXMLBranche);
   if (anObj.DataInside().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DataInside"),anObj.DataInside().Val())->ReTagThis("DataInside"));
   aRes->AddFils(::ToXMLTree(std::string("KeyNameMTD"),anObj.KeyNameMTD())->ReTagThis("KeyNameMTD"));
   if (anObj.RatioPseudoConik().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("RatioPseudoConik"),anObj.RatioPseudoConik().Val())->ReTagThis("RatioPseudoConik"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cMTD_Nuage_Maille & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.DataInside(),aTree->Get("DataInside",1),bool(false)); //tototo 

   xml_init(anObj.KeyNameMTD(),aTree->Get("KeyNameMTD",1)); //tototo 

   xml_init(anObj.RatioPseudoConik(),aTree->Get("RatioPseudoConik",1),double(1000)); //tototo 
}

std::string  Mangling( cMTD_Nuage_Maille *) {return "DAAE55A48478DBE9FE3F";};


std::string & cCannauxExportPly::NameIm()
{
   return mNameIm;
}

const std::string & cCannauxExportPly::NameIm()const 
{
   return mNameIm;
}


std::vector< std::string > & cCannauxExportPly::NamesProperty()
{
   return mNamesProperty;
}

const std::vector< std::string > & cCannauxExportPly::NamesProperty()const 
{
   return mNamesProperty;
}


cTplValGesInit< int > & cCannauxExportPly::FlagUse()
{
   return mFlagUse;
}

const cTplValGesInit< int > & cCannauxExportPly::FlagUse()const 
{
   return mFlagUse;
}

void  BinaryUnDumpFromFile(cCannauxExportPly & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.NameIm(),aFp);
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             std::string aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.NamesProperty().push_back(aVal);
        }
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.FlagUse().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.FlagUse().ValForcedForUnUmp(),aFp);
        }
        else  anObj.FlagUse().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cCannauxExportPly & anObj)
{
    BinaryDumpInFile(aFp,anObj.NameIm());
    BinaryDumpInFile(aFp,(int)anObj.NamesProperty().size());
    for(  std::vector< std::string >::const_iterator iT=anObj.NamesProperty().begin();
         iT!=anObj.NamesProperty().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.FlagUse().IsInit());
    if (anObj.FlagUse().IsInit()) BinaryDumpInFile(aFp,anObj.FlagUse().Val());
}

cElXMLTree * ToXMLTree(const cCannauxExportPly & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"CannauxExportPly",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("NameIm"),anObj.NameIm())->ReTagThis("NameIm"));
  for
  (       std::vector< std::string >::const_iterator it=anObj.NamesProperty().begin();
      it !=anObj.NamesProperty().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("NamesProperty"),(*it))->ReTagThis("NamesProperty"));
   if (anObj.FlagUse().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("FlagUse"),anObj.FlagUse().Val())->ReTagThis("FlagUse"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cCannauxExportPly & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.NameIm(),aTree->Get("NameIm",1)); //tototo 

   xml_init(anObj.NamesProperty(),aTree->GetAll("NamesProperty",false,1));

   xml_init(anObj.FlagUse(),aTree->Get("FlagUse",1)); //tototo 
}

std::string  Mangling( cCannauxExportPly *) {return "B499F30749FB3BD1FE3F";};


cTplValGesInit< std::string > & cPlyFile::KeyNamePly()
{
   return mKeyNamePly;
}

const cTplValGesInit< std::string > & cPlyFile::KeyNamePly()const 
{
   return mKeyNamePly;
}


bool & cPlyFile::Binary()
{
   return mBinary;
}

const bool & cPlyFile::Binary()const 
{
   return mBinary;
}


double & cPlyFile::Resolution()
{
   return mResolution;
}

const double & cPlyFile::Resolution()const 
{
   return mResolution;
}


std::list< std::string > & cPlyFile::PlyCommentAdd()
{
   return mPlyCommentAdd;
}

const std::list< std::string > & cPlyFile::PlyCommentAdd()const 
{
   return mPlyCommentAdd;
}


std::list< cCannauxExportPly > & cPlyFile::CannauxExportPly()
{
   return mCannauxExportPly;
}

const std::list< cCannauxExportPly > & cPlyFile::CannauxExportPly()const 
{
   return mCannauxExportPly;
}

void  BinaryUnDumpFromFile(cPlyFile & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.KeyNamePly().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.KeyNamePly().ValForcedForUnUmp(),aFp);
        }
        else  anObj.KeyNamePly().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.Binary(),aFp);
    BinaryUnDumpFromFile(anObj.Resolution(),aFp);
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             std::string aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.PlyCommentAdd().push_back(aVal);
        }
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cCannauxExportPly aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.CannauxExportPly().push_back(aVal);
        }
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cPlyFile & anObj)
{
    BinaryDumpInFile(aFp,anObj.KeyNamePly().IsInit());
    if (anObj.KeyNamePly().IsInit()) BinaryDumpInFile(aFp,anObj.KeyNamePly().Val());
    BinaryDumpInFile(aFp,anObj.Binary());
    BinaryDumpInFile(aFp,anObj.Resolution());
    BinaryDumpInFile(aFp,(int)anObj.PlyCommentAdd().size());
    for(  std::list< std::string >::const_iterator iT=anObj.PlyCommentAdd().begin();
         iT!=anObj.PlyCommentAdd().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,(int)anObj.CannauxExportPly().size());
    for(  std::list< cCannauxExportPly >::const_iterator iT=anObj.CannauxExportPly().begin();
         iT!=anObj.CannauxExportPly().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
}

cElXMLTree * ToXMLTree(const cPlyFile & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"PlyFile",eXMLBranche);
   if (anObj.KeyNamePly().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("KeyNamePly"),anObj.KeyNamePly().Val())->ReTagThis("KeyNamePly"));
   aRes->AddFils(::ToXMLTree(std::string("Binary"),anObj.Binary())->ReTagThis("Binary"));
   aRes->AddFils(::ToXMLTree(std::string("Resolution"),anObj.Resolution())->ReTagThis("Resolution"));
  for
  (       std::list< std::string >::const_iterator it=anObj.PlyCommentAdd().begin();
      it !=anObj.PlyCommentAdd().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("PlyCommentAdd"),(*it))->ReTagThis("PlyCommentAdd"));
  for
  (       std::list< cCannauxExportPly >::const_iterator it=anObj.CannauxExportPly().begin();
      it !=anObj.CannauxExportPly().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("CannauxExportPly"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cPlyFile & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.KeyNamePly(),aTree->Get("KeyNamePly",1),std::string("Key-Assoc-Nuage-Ply")); //tototo 

   xml_init(anObj.Binary(),aTree->Get("Binary",1)); //tototo 

   xml_init(anObj.Resolution(),aTree->Get("Resolution",1)); //tototo 

   xml_init(anObj.PlyCommentAdd(),aTree->GetAll("PlyCommentAdd",false,1));

   xml_init(anObj.CannauxExportPly(),aTree->GetAll("CannauxExportPly",false,1));
}

std::string  Mangling( cPlyFile *) {return "90D9B0824CDD69A3FDBF";};


cTplValGesInit< bool > & cMMExportNuage::DataInside()
{
   return MTD_Nuage_Maille().Val().DataInside();
}

const cTplValGesInit< bool > & cMMExportNuage::DataInside()const 
{
   return MTD_Nuage_Maille().Val().DataInside();
}


std::string & cMMExportNuage::KeyNameMTD()
{
   return MTD_Nuage_Maille().Val().KeyNameMTD();
}

const std::string & cMMExportNuage::KeyNameMTD()const 
{
   return MTD_Nuage_Maille().Val().KeyNameMTD();
}


cTplValGesInit< double > & cMMExportNuage::RatioPseudoConik()
{
   return MTD_Nuage_Maille().Val().RatioPseudoConik();
}

const cTplValGesInit< double > & cMMExportNuage::RatioPseudoConik()const 
{
   return MTD_Nuage_Maille().Val().RatioPseudoConik();
}


cTplValGesInit< cMTD_Nuage_Maille > & cMMExportNuage::MTD_Nuage_Maille()
{
   return mMTD_Nuage_Maille;
}

const cTplValGesInit< cMTD_Nuage_Maille > & cMMExportNuage::MTD_Nuage_Maille()const 
{
   return mMTD_Nuage_Maille;
}


cTplValGesInit< std::string > & cMMExportNuage::KeyNamePly()
{
   return PlyFile().Val().KeyNamePly();
}

const cTplValGesInit< std::string > & cMMExportNuage::KeyNamePly()const 
{
   return PlyFile().Val().KeyNamePly();
}


bool & cMMExportNuage::Binary()
{
   return PlyFile().Val().Binary();
}

const bool & cMMExportNuage::Binary()const 
{
   return PlyFile().Val().Binary();
}


double & cMMExportNuage::Resolution()
{
   return PlyFile().Val().Resolution();
}

const double & cMMExportNuage::Resolution()const 
{
   return PlyFile().Val().Resolution();
}


std::list< std::string > & cMMExportNuage::PlyCommentAdd()
{
   return PlyFile().Val().PlyCommentAdd();
}

const std::list< std::string > & cMMExportNuage::PlyCommentAdd()const 
{
   return PlyFile().Val().PlyCommentAdd();
}


std::list< cCannauxExportPly > & cMMExportNuage::CannauxExportPly()
{
   return PlyFile().Val().CannauxExportPly();
}

const std::list< cCannauxExportPly > & cMMExportNuage::CannauxExportPly()const 
{
   return PlyFile().Val().CannauxExportPly();
}


cTplValGesInit< cPlyFile > & cMMExportNuage::PlyFile()
{
   return mPlyFile;
}

const cTplValGesInit< cPlyFile > & cMMExportNuage::PlyFile()const 
{
   return mPlyFile;
}

void  BinaryUnDumpFromFile(cMMExportNuage & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.MTD_Nuage_Maille().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.MTD_Nuage_Maille().ValForcedForUnUmp(),aFp);
        }
        else  anObj.MTD_Nuage_Maille().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.PlyFile().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.PlyFile().ValForcedForUnUmp(),aFp);
        }
        else  anObj.PlyFile().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cMMExportNuage & anObj)
{
    BinaryDumpInFile(aFp,anObj.MTD_Nuage_Maille().IsInit());
    if (anObj.MTD_Nuage_Maille().IsInit()) BinaryDumpInFile(aFp,anObj.MTD_Nuage_Maille().Val());
    BinaryDumpInFile(aFp,anObj.PlyFile().IsInit());
    if (anObj.PlyFile().IsInit()) BinaryDumpInFile(aFp,anObj.PlyFile().Val());
}

cElXMLTree * ToXMLTree(const cMMExportNuage & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"MMExportNuage",eXMLBranche);
   if (anObj.MTD_Nuage_Maille().IsInit())
      aRes->AddFils(ToXMLTree(anObj.MTD_Nuage_Maille().Val())->ReTagThis("MTD_Nuage_Maille"));
   if (anObj.PlyFile().IsInit())
      aRes->AddFils(ToXMLTree(anObj.PlyFile().Val())->ReTagThis("PlyFile"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cMMExportNuage & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.MTD_Nuage_Maille(),aTree->Get("MTD_Nuage_Maille",1)); //tototo 

   xml_init(anObj.PlyFile(),aTree->Get("PlyFile",1)); //tototo 
}

std::string  Mangling( cMMExportNuage *) {return "3FB3F025F0F2A7D3FE3F";};


bool & cReCalclCorrelMultiEchelle::UseIt()
{
   return mUseIt;
}

const bool & cReCalclCorrelMultiEchelle::UseIt()const 
{
   return mUseIt;
}


std::list< Pt2di > & cReCalclCorrelMultiEchelle::ScaleSzW()
{
   return mScaleSzW;
}

const std::list< Pt2di > & cReCalclCorrelMultiEchelle::ScaleSzW()const 
{
   return mScaleSzW;
}


cTplValGesInit< bool > & cReCalclCorrelMultiEchelle::AgregMin()
{
   return mAgregMin;
}

const cTplValGesInit< bool > & cReCalclCorrelMultiEchelle::AgregMin()const 
{
   return mAgregMin;
}


cTplValGesInit< bool > & cReCalclCorrelMultiEchelle::DoImg()
{
   return mDoImg;
}

const cTplValGesInit< bool > & cReCalclCorrelMultiEchelle::DoImg()const 
{
   return mDoImg;
}


double & cReCalclCorrelMultiEchelle::Seuil()
{
   return mSeuil;
}

const double & cReCalclCorrelMultiEchelle::Seuil()const 
{
   return mSeuil;
}

void  BinaryUnDumpFromFile(cReCalclCorrelMultiEchelle & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.UseIt(),aFp);
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             Pt2di aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.ScaleSzW().push_back(aVal);
        }
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.AgregMin().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.AgregMin().ValForcedForUnUmp(),aFp);
        }
        else  anObj.AgregMin().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.DoImg().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.DoImg().ValForcedForUnUmp(),aFp);
        }
        else  anObj.DoImg().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.Seuil(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cReCalclCorrelMultiEchelle & anObj)
{
    BinaryDumpInFile(aFp,anObj.UseIt());
    BinaryDumpInFile(aFp,(int)anObj.ScaleSzW().size());
    for(  std::list< Pt2di >::const_iterator iT=anObj.ScaleSzW().begin();
         iT!=anObj.ScaleSzW().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.AgregMin().IsInit());
    if (anObj.AgregMin().IsInit()) BinaryDumpInFile(aFp,anObj.AgregMin().Val());
    BinaryDumpInFile(aFp,anObj.DoImg().IsInit());
    if (anObj.DoImg().IsInit()) BinaryDumpInFile(aFp,anObj.DoImg().Val());
    BinaryDumpInFile(aFp,anObj.Seuil());
}

cElXMLTree * ToXMLTree(const cReCalclCorrelMultiEchelle & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ReCalclCorrelMultiEchelle",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("UseIt"),anObj.UseIt())->ReTagThis("UseIt"));
  for
  (       std::list< Pt2di >::const_iterator it=anObj.ScaleSzW().begin();
      it !=anObj.ScaleSzW().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("ScaleSzW"),(*it))->ReTagThis("ScaleSzW"));
   if (anObj.AgregMin().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("AgregMin"),anObj.AgregMin().Val())->ReTagThis("AgregMin"));
   if (anObj.DoImg().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DoImg"),anObj.DoImg().Val())->ReTagThis("DoImg"));
   aRes->AddFils(::ToXMLTree(std::string("Seuil"),anObj.Seuil())->ReTagThis("Seuil"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cReCalclCorrelMultiEchelle & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.UseIt(),aTree->Get("UseIt",1)); //tototo 

   xml_init(anObj.ScaleSzW(),aTree->GetAll("ScaleSzW",false,1));

   xml_init(anObj.AgregMin(),aTree->Get("AgregMin",1),bool(true)); //tototo 

   xml_init(anObj.DoImg(),aTree->Get("DoImg",1),bool(true)); //tototo 

   xml_init(anObj.Seuil(),aTree->Get("Seuil",1)); //tototo 
}

std::string  Mangling( cReCalclCorrelMultiEchelle *) {return "9575C0BC15458B9EFE3F";};


cTplValGesInit< bool > & cOneModeleAnalytique::UseIt()
{
   return mUseIt;
}

const cTplValGesInit< bool > & cOneModeleAnalytique::UseIt()const 
{
   return mUseIt;
}


cTplValGesInit< std::string > & cOneModeleAnalytique::KeyNuage3D()
{
   return mKeyNuage3D;
}

const cTplValGesInit< std::string > & cOneModeleAnalytique::KeyNuage3D()const 
{
   return mKeyNuage3D;
}


eTypeModeleAnalytique & cOneModeleAnalytique::TypeModele()
{
   return mTypeModele;
}

const eTypeModeleAnalytique & cOneModeleAnalytique::TypeModele()const 
{
   return mTypeModele;
}


cTplValGesInit< bool > & cOneModeleAnalytique::HomographieL2()
{
   return mHomographieL2;
}

const cTplValGesInit< bool > & cOneModeleAnalytique::HomographieL2()const 
{
   return mHomographieL2;
}


cTplValGesInit< bool > & cOneModeleAnalytique::PolynomeL2()
{
   return mPolynomeL2;
}

const cTplValGesInit< bool > & cOneModeleAnalytique::PolynomeL2()const 
{
   return mPolynomeL2;
}


cTplValGesInit< int > & cOneModeleAnalytique::DegrePol()
{
   return mDegrePol;
}

const cTplValGesInit< int > & cOneModeleAnalytique::DegrePol()const 
{
   return mDegrePol;
}


std::list< int > & cOneModeleAnalytique::NumsAngleFiges()
{
   return mNumsAngleFiges;
}

const std::list< int > & cOneModeleAnalytique::NumsAngleFiges()const 
{
   return mNumsAngleFiges;
}


cTplValGesInit< bool > & cOneModeleAnalytique::L1CalcOri()
{
   return mL1CalcOri;
}

const cTplValGesInit< bool > & cOneModeleAnalytique::L1CalcOri()const 
{
   return mL1CalcOri;
}


cTplValGesInit< std::string > & cOneModeleAnalytique::AutomSelExportOri()
{
   return mAutomSelExportOri;
}

const cTplValGesInit< std::string > & cOneModeleAnalytique::AutomSelExportOri()const 
{
   return mAutomSelExportOri;
}


cTplValGesInit< std::string > & cOneModeleAnalytique::AutomNamesExportOri1()
{
   return mAutomNamesExportOri1;
}

const cTplValGesInit< std::string > & cOneModeleAnalytique::AutomNamesExportOri1()const 
{
   return mAutomNamesExportOri1;
}


cTplValGesInit< std::string > & cOneModeleAnalytique::AutomNamesExportOri2()
{
   return mAutomNamesExportOri2;
}

const cTplValGesInit< std::string > & cOneModeleAnalytique::AutomNamesExportOri2()const 
{
   return mAutomNamesExportOri2;
}


cTplValGesInit< std::string > & cOneModeleAnalytique::AutomNamesExportHomXml()
{
   return mAutomNamesExportHomXml;
}

const cTplValGesInit< std::string > & cOneModeleAnalytique::AutomNamesExportHomXml()const 
{
   return mAutomNamesExportHomXml;
}


cTplValGesInit< std::string > & cOneModeleAnalytique::AutomNamesExportHomTif()
{
   return mAutomNamesExportHomTif;
}

const cTplValGesInit< std::string > & cOneModeleAnalytique::AutomNamesExportHomTif()const 
{
   return mAutomNamesExportHomTif;
}


cTplValGesInit< std::string > & cOneModeleAnalytique::AutomNamesExportHomBin()
{
   return mAutomNamesExportHomBin;
}

const cTplValGesInit< std::string > & cOneModeleAnalytique::AutomNamesExportHomBin()const 
{
   return mAutomNamesExportHomBin;
}


cTplValGesInit< bool > & cOneModeleAnalytique::AffineOrient()
{
   return mAffineOrient;
}

const cTplValGesInit< bool > & cOneModeleAnalytique::AffineOrient()const 
{
   return mAffineOrient;
}


cTplValGesInit< std::string > & cOneModeleAnalytique::KeyNamesExportHomXml()
{
   return mKeyNamesExportHomXml;
}

const cTplValGesInit< std::string > & cOneModeleAnalytique::KeyNamesExportHomXml()const 
{
   return mKeyNamesExportHomXml;
}


cTplValGesInit< double > & cOneModeleAnalytique::SigmaPixPdsExport()
{
   return mSigmaPixPdsExport;
}

const cTplValGesInit< double > & cOneModeleAnalytique::SigmaPixPdsExport()const 
{
   return mSigmaPixPdsExport;
}


cTplValGesInit< bool > & cOneModeleAnalytique::FiltreByCorrel()
{
   return mFiltreByCorrel;
}

const cTplValGesInit< bool > & cOneModeleAnalytique::FiltreByCorrel()const 
{
   return mFiltreByCorrel;
}


cTplValGesInit< double > & cOneModeleAnalytique::SeuilFiltreCorrel()
{
   return mSeuilFiltreCorrel;
}

const cTplValGesInit< double > & cOneModeleAnalytique::SeuilFiltreCorrel()const 
{
   return mSeuilFiltreCorrel;
}


cTplValGesInit< bool > & cOneModeleAnalytique::UseFCBySeuil()
{
   return mUseFCBySeuil;
}

const cTplValGesInit< bool > & cOneModeleAnalytique::UseFCBySeuil()const 
{
   return mUseFCBySeuil;
}


cTplValGesInit< double > & cOneModeleAnalytique::ExposantPondereCorrel()
{
   return mExposantPondereCorrel;
}

const cTplValGesInit< double > & cOneModeleAnalytique::ExposantPondereCorrel()const 
{
   return mExposantPondereCorrel;
}


std::list< cReCalclCorrelMultiEchelle > & cOneModeleAnalytique::ReCalclCorrelMultiEchelle()
{
   return mReCalclCorrelMultiEchelle;
}

const std::list< cReCalclCorrelMultiEchelle > & cOneModeleAnalytique::ReCalclCorrelMultiEchelle()const 
{
   return mReCalclCorrelMultiEchelle;
}


int & cOneModeleAnalytique::PasCalcul()
{
   return mPasCalcul;
}

const int & cOneModeleAnalytique::PasCalcul()const 
{
   return mPasCalcul;
}


cTplValGesInit< bool > & cOneModeleAnalytique::PointUnique()
{
   return mPointUnique;
}

const cTplValGesInit< bool > & cOneModeleAnalytique::PointUnique()const 
{
   return mPointUnique;
}


cTplValGesInit< bool > & cOneModeleAnalytique::ReuseModele()
{
   return mReuseModele;
}

const cTplValGesInit< bool > & cOneModeleAnalytique::ReuseModele()const 
{
   return mReuseModele;
}


cTplValGesInit< bool > & cOneModeleAnalytique::MakeExport()
{
   return mMakeExport;
}

const cTplValGesInit< bool > & cOneModeleAnalytique::MakeExport()const 
{
   return mMakeExport;
}


cTplValGesInit< std::string > & cOneModeleAnalytique::NameExport()
{
   return mNameExport;
}

const cTplValGesInit< std::string > & cOneModeleAnalytique::NameExport()const 
{
   return mNameExport;
}


cTplValGesInit< bool > & cOneModeleAnalytique::ExportImage()
{
   return mExportImage;
}

const cTplValGesInit< bool > & cOneModeleAnalytique::ExportImage()const 
{
   return mExportImage;
}


cTplValGesInit< bool > & cOneModeleAnalytique::ReuseResiduelle()
{
   return mReuseResiduelle;
}

const cTplValGesInit< bool > & cOneModeleAnalytique::ReuseResiduelle()const 
{
   return mReuseResiduelle;
}


cTplValGesInit< std::string > & cOneModeleAnalytique::FCND_ExportModeleGlobal()
{
   return mFCND_ExportModeleGlobal;
}

const cTplValGesInit< std::string > & cOneModeleAnalytique::FCND_ExportModeleGlobal()const 
{
   return mFCND_ExportModeleGlobal;
}


cTplValGesInit< double > & cOneModeleAnalytique::MailleExport()
{
   return mMailleExport;
}

const cTplValGesInit< double > & cOneModeleAnalytique::MailleExport()const 
{
   return mMailleExport;
}


cTplValGesInit< bool > & cOneModeleAnalytique::UseHomologueReference()
{
   return mUseHomologueReference;
}

const cTplValGesInit< bool > & cOneModeleAnalytique::UseHomologueReference()const 
{
   return mUseHomologueReference;
}


cTplValGesInit< bool > & cOneModeleAnalytique::MakeImagePxRef()
{
   return mMakeImagePxRef;
}

const cTplValGesInit< bool > & cOneModeleAnalytique::MakeImagePxRef()const 
{
   return mMakeImagePxRef;
}


cTplValGesInit< int > & cOneModeleAnalytique::NbPtMinValideEqOriRel()
{
   return mNbPtMinValideEqOriRel;
}

const cTplValGesInit< int > & cOneModeleAnalytique::NbPtMinValideEqOriRel()const 
{
   return mNbPtMinValideEqOriRel;
}

void  BinaryUnDumpFromFile(cOneModeleAnalytique & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.UseIt().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.UseIt().ValForcedForUnUmp(),aFp);
        }
        else  anObj.UseIt().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.KeyNuage3D().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.KeyNuage3D().ValForcedForUnUmp(),aFp);
        }
        else  anObj.KeyNuage3D().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.TypeModele(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.HomographieL2().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.HomographieL2().ValForcedForUnUmp(),aFp);
        }
        else  anObj.HomographieL2().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.PolynomeL2().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.PolynomeL2().ValForcedForUnUmp(),aFp);
        }
        else  anObj.PolynomeL2().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.DegrePol().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.DegrePol().ValForcedForUnUmp(),aFp);
        }
        else  anObj.DegrePol().SetNoInit();
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             int aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.NumsAngleFiges().push_back(aVal);
        }
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.L1CalcOri().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.L1CalcOri().ValForcedForUnUmp(),aFp);
        }
        else  anObj.L1CalcOri().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.AutomSelExportOri().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.AutomSelExportOri().ValForcedForUnUmp(),aFp);
        }
        else  anObj.AutomSelExportOri().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.AutomNamesExportOri1().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.AutomNamesExportOri1().ValForcedForUnUmp(),aFp);
        }
        else  anObj.AutomNamesExportOri1().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.AutomNamesExportOri2().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.AutomNamesExportOri2().ValForcedForUnUmp(),aFp);
        }
        else  anObj.AutomNamesExportOri2().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.AutomNamesExportHomXml().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.AutomNamesExportHomXml().ValForcedForUnUmp(),aFp);
        }
        else  anObj.AutomNamesExportHomXml().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.AutomNamesExportHomTif().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.AutomNamesExportHomTif().ValForcedForUnUmp(),aFp);
        }
        else  anObj.AutomNamesExportHomTif().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.AutomNamesExportHomBin().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.AutomNamesExportHomBin().ValForcedForUnUmp(),aFp);
        }
        else  anObj.AutomNamesExportHomBin().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.AffineOrient().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.AffineOrient().ValForcedForUnUmp(),aFp);
        }
        else  anObj.AffineOrient().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.KeyNamesExportHomXml().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.KeyNamesExportHomXml().ValForcedForUnUmp(),aFp);
        }
        else  anObj.KeyNamesExportHomXml().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SigmaPixPdsExport().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SigmaPixPdsExport().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SigmaPixPdsExport().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.FiltreByCorrel().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.FiltreByCorrel().ValForcedForUnUmp(),aFp);
        }
        else  anObj.FiltreByCorrel().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SeuilFiltreCorrel().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SeuilFiltreCorrel().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SeuilFiltreCorrel().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.UseFCBySeuil().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.UseFCBySeuil().ValForcedForUnUmp(),aFp);
        }
        else  anObj.UseFCBySeuil().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ExposantPondereCorrel().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ExposantPondereCorrel().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ExposantPondereCorrel().SetNoInit();
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cReCalclCorrelMultiEchelle aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.ReCalclCorrelMultiEchelle().push_back(aVal);
        }
  } ;
    BinaryUnDumpFromFile(anObj.PasCalcul(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.PointUnique().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.PointUnique().ValForcedForUnUmp(),aFp);
        }
        else  anObj.PointUnique().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ReuseModele().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ReuseModele().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ReuseModele().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.MakeExport().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.MakeExport().ValForcedForUnUmp(),aFp);
        }
        else  anObj.MakeExport().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NameExport().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NameExport().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NameExport().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ExportImage().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ExportImage().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ExportImage().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ReuseResiduelle().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ReuseResiduelle().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ReuseResiduelle().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.FCND_ExportModeleGlobal().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.FCND_ExportModeleGlobal().ValForcedForUnUmp(),aFp);
        }
        else  anObj.FCND_ExportModeleGlobal().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.MailleExport().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.MailleExport().ValForcedForUnUmp(),aFp);
        }
        else  anObj.MailleExport().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.UseHomologueReference().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.UseHomologueReference().ValForcedForUnUmp(),aFp);
        }
        else  anObj.UseHomologueReference().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.MakeImagePxRef().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.MakeImagePxRef().ValForcedForUnUmp(),aFp);
        }
        else  anObj.MakeImagePxRef().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NbPtMinValideEqOriRel().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NbPtMinValideEqOriRel().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NbPtMinValideEqOriRel().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cOneModeleAnalytique & anObj)
{
    BinaryDumpInFile(aFp,anObj.UseIt().IsInit());
    if (anObj.UseIt().IsInit()) BinaryDumpInFile(aFp,anObj.UseIt().Val());
    BinaryDumpInFile(aFp,anObj.KeyNuage3D().IsInit());
    if (anObj.KeyNuage3D().IsInit()) BinaryDumpInFile(aFp,anObj.KeyNuage3D().Val());
    BinaryDumpInFile(aFp,anObj.TypeModele());
    BinaryDumpInFile(aFp,anObj.HomographieL2().IsInit());
    if (anObj.HomographieL2().IsInit()) BinaryDumpInFile(aFp,anObj.HomographieL2().Val());
    BinaryDumpInFile(aFp,anObj.PolynomeL2().IsInit());
    if (anObj.PolynomeL2().IsInit()) BinaryDumpInFile(aFp,anObj.PolynomeL2().Val());
    BinaryDumpInFile(aFp,anObj.DegrePol().IsInit());
    if (anObj.DegrePol().IsInit()) BinaryDumpInFile(aFp,anObj.DegrePol().Val());
    BinaryDumpInFile(aFp,(int)anObj.NumsAngleFiges().size());
    for(  std::list< int >::const_iterator iT=anObj.NumsAngleFiges().begin();
         iT!=anObj.NumsAngleFiges().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.L1CalcOri().IsInit());
    if (anObj.L1CalcOri().IsInit()) BinaryDumpInFile(aFp,anObj.L1CalcOri().Val());
    BinaryDumpInFile(aFp,anObj.AutomSelExportOri().IsInit());
    if (anObj.AutomSelExportOri().IsInit()) BinaryDumpInFile(aFp,anObj.AutomSelExportOri().Val());
    BinaryDumpInFile(aFp,anObj.AutomNamesExportOri1().IsInit());
    if (anObj.AutomNamesExportOri1().IsInit()) BinaryDumpInFile(aFp,anObj.AutomNamesExportOri1().Val());
    BinaryDumpInFile(aFp,anObj.AutomNamesExportOri2().IsInit());
    if (anObj.AutomNamesExportOri2().IsInit()) BinaryDumpInFile(aFp,anObj.AutomNamesExportOri2().Val());
    BinaryDumpInFile(aFp,anObj.AutomNamesExportHomXml().IsInit());
    if (anObj.AutomNamesExportHomXml().IsInit()) BinaryDumpInFile(aFp,anObj.AutomNamesExportHomXml().Val());
    BinaryDumpInFile(aFp,anObj.AutomNamesExportHomTif().IsInit());
    if (anObj.AutomNamesExportHomTif().IsInit()) BinaryDumpInFile(aFp,anObj.AutomNamesExportHomTif().Val());
    BinaryDumpInFile(aFp,anObj.AutomNamesExportHomBin().IsInit());
    if (anObj.AutomNamesExportHomBin().IsInit()) BinaryDumpInFile(aFp,anObj.AutomNamesExportHomBin().Val());
    BinaryDumpInFile(aFp,anObj.AffineOrient().IsInit());
    if (anObj.AffineOrient().IsInit()) BinaryDumpInFile(aFp,anObj.AffineOrient().Val());
    BinaryDumpInFile(aFp,anObj.KeyNamesExportHomXml().IsInit());
    if (anObj.KeyNamesExportHomXml().IsInit()) BinaryDumpInFile(aFp,anObj.KeyNamesExportHomXml().Val());
    BinaryDumpInFile(aFp,anObj.SigmaPixPdsExport().IsInit());
    if (anObj.SigmaPixPdsExport().IsInit()) BinaryDumpInFile(aFp,anObj.SigmaPixPdsExport().Val());
    BinaryDumpInFile(aFp,anObj.FiltreByCorrel().IsInit());
    if (anObj.FiltreByCorrel().IsInit()) BinaryDumpInFile(aFp,anObj.FiltreByCorrel().Val());
    BinaryDumpInFile(aFp,anObj.SeuilFiltreCorrel().IsInit());
    if (anObj.SeuilFiltreCorrel().IsInit()) BinaryDumpInFile(aFp,anObj.SeuilFiltreCorrel().Val());
    BinaryDumpInFile(aFp,anObj.UseFCBySeuil().IsInit());
    if (anObj.UseFCBySeuil().IsInit()) BinaryDumpInFile(aFp,anObj.UseFCBySeuil().Val());
    BinaryDumpInFile(aFp,anObj.ExposantPondereCorrel().IsInit());
    if (anObj.ExposantPondereCorrel().IsInit()) BinaryDumpInFile(aFp,anObj.ExposantPondereCorrel().Val());
    BinaryDumpInFile(aFp,(int)anObj.ReCalclCorrelMultiEchelle().size());
    for(  std::list< cReCalclCorrelMultiEchelle >::const_iterator iT=anObj.ReCalclCorrelMultiEchelle().begin();
         iT!=anObj.ReCalclCorrelMultiEchelle().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.PasCalcul());
    BinaryDumpInFile(aFp,anObj.PointUnique().IsInit());
    if (anObj.PointUnique().IsInit()) BinaryDumpInFile(aFp,anObj.PointUnique().Val());
    BinaryDumpInFile(aFp,anObj.ReuseModele().IsInit());
    if (anObj.ReuseModele().IsInit()) BinaryDumpInFile(aFp,anObj.ReuseModele().Val());
    BinaryDumpInFile(aFp,anObj.MakeExport().IsInit());
    if (anObj.MakeExport().IsInit()) BinaryDumpInFile(aFp,anObj.MakeExport().Val());
    BinaryDumpInFile(aFp,anObj.NameExport().IsInit());
    if (anObj.NameExport().IsInit()) BinaryDumpInFile(aFp,anObj.NameExport().Val());
    BinaryDumpInFile(aFp,anObj.ExportImage().IsInit());
    if (anObj.ExportImage().IsInit()) BinaryDumpInFile(aFp,anObj.ExportImage().Val());
    BinaryDumpInFile(aFp,anObj.ReuseResiduelle().IsInit());
    if (anObj.ReuseResiduelle().IsInit()) BinaryDumpInFile(aFp,anObj.ReuseResiduelle().Val());
    BinaryDumpInFile(aFp,anObj.FCND_ExportModeleGlobal().IsInit());
    if (anObj.FCND_ExportModeleGlobal().IsInit()) BinaryDumpInFile(aFp,anObj.FCND_ExportModeleGlobal().Val());
    BinaryDumpInFile(aFp,anObj.MailleExport().IsInit());
    if (anObj.MailleExport().IsInit()) BinaryDumpInFile(aFp,anObj.MailleExport().Val());
    BinaryDumpInFile(aFp,anObj.UseHomologueReference().IsInit());
    if (anObj.UseHomologueReference().IsInit()) BinaryDumpInFile(aFp,anObj.UseHomologueReference().Val());
    BinaryDumpInFile(aFp,anObj.MakeImagePxRef().IsInit());
    if (anObj.MakeImagePxRef().IsInit()) BinaryDumpInFile(aFp,anObj.MakeImagePxRef().Val());
    BinaryDumpInFile(aFp,anObj.NbPtMinValideEqOriRel().IsInit());
    if (anObj.NbPtMinValideEqOriRel().IsInit()) BinaryDumpInFile(aFp,anObj.NbPtMinValideEqOriRel().Val());
}

cElXMLTree * ToXMLTree(const cOneModeleAnalytique & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"OneModeleAnalytique",eXMLBranche);
   if (anObj.UseIt().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("UseIt"),anObj.UseIt().Val())->ReTagThis("UseIt"));
   if (anObj.KeyNuage3D().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("KeyNuage3D"),anObj.KeyNuage3D().Val())->ReTagThis("KeyNuage3D"));
   aRes->AddFils(ToXMLTree(std::string("TypeModele"),anObj.TypeModele())->ReTagThis("TypeModele"));
   if (anObj.HomographieL2().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("HomographieL2"),anObj.HomographieL2().Val())->ReTagThis("HomographieL2"));
   if (anObj.PolynomeL2().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PolynomeL2"),anObj.PolynomeL2().Val())->ReTagThis("PolynomeL2"));
   if (anObj.DegrePol().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DegrePol"),anObj.DegrePol().Val())->ReTagThis("DegrePol"));
  for
  (       std::list< int >::const_iterator it=anObj.NumsAngleFiges().begin();
      it !=anObj.NumsAngleFiges().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("NumsAngleFiges"),(*it))->ReTagThis("NumsAngleFiges"));
   if (anObj.L1CalcOri().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("L1CalcOri"),anObj.L1CalcOri().Val())->ReTagThis("L1CalcOri"));
   if (anObj.AutomSelExportOri().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("AutomSelExportOri"),anObj.AutomSelExportOri().Val())->ReTagThis("AutomSelExportOri"));
   if (anObj.AutomNamesExportOri1().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("AutomNamesExportOri1"),anObj.AutomNamesExportOri1().Val())->ReTagThis("AutomNamesExportOri1"));
   if (anObj.AutomNamesExportOri2().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("AutomNamesExportOri2"),anObj.AutomNamesExportOri2().Val())->ReTagThis("AutomNamesExportOri2"));
   if (anObj.AutomNamesExportHomXml().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("AutomNamesExportHomXml"),anObj.AutomNamesExportHomXml().Val())->ReTagThis("AutomNamesExportHomXml"));
   if (anObj.AutomNamesExportHomTif().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("AutomNamesExportHomTif"),anObj.AutomNamesExportHomTif().Val())->ReTagThis("AutomNamesExportHomTif"));
   if (anObj.AutomNamesExportHomBin().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("AutomNamesExportHomBin"),anObj.AutomNamesExportHomBin().Val())->ReTagThis("AutomNamesExportHomBin"));
   if (anObj.AffineOrient().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("AffineOrient"),anObj.AffineOrient().Val())->ReTagThis("AffineOrient"));
   if (anObj.KeyNamesExportHomXml().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("KeyNamesExportHomXml"),anObj.KeyNamesExportHomXml().Val())->ReTagThis("KeyNamesExportHomXml"));
   if (anObj.SigmaPixPdsExport().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SigmaPixPdsExport"),anObj.SigmaPixPdsExport().Val())->ReTagThis("SigmaPixPdsExport"));
   if (anObj.FiltreByCorrel().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("FiltreByCorrel"),anObj.FiltreByCorrel().Val())->ReTagThis("FiltreByCorrel"));
   if (anObj.SeuilFiltreCorrel().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SeuilFiltreCorrel"),anObj.SeuilFiltreCorrel().Val())->ReTagThis("SeuilFiltreCorrel"));
   if (anObj.UseFCBySeuil().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("UseFCBySeuil"),anObj.UseFCBySeuil().Val())->ReTagThis("UseFCBySeuil"));
   if (anObj.ExposantPondereCorrel().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ExposantPondereCorrel"),anObj.ExposantPondereCorrel().Val())->ReTagThis("ExposantPondereCorrel"));
  for
  (       std::list< cReCalclCorrelMultiEchelle >::const_iterator it=anObj.ReCalclCorrelMultiEchelle().begin();
      it !=anObj.ReCalclCorrelMultiEchelle().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("ReCalclCorrelMultiEchelle"));
   aRes->AddFils(::ToXMLTree(std::string("PasCalcul"),anObj.PasCalcul())->ReTagThis("PasCalcul"));
   if (anObj.PointUnique().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PointUnique"),anObj.PointUnique().Val())->ReTagThis("PointUnique"));
   if (anObj.ReuseModele().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ReuseModele"),anObj.ReuseModele().Val())->ReTagThis("ReuseModele"));
   if (anObj.MakeExport().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("MakeExport"),anObj.MakeExport().Val())->ReTagThis("MakeExport"));
   if (anObj.NameExport().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NameExport"),anObj.NameExport().Val())->ReTagThis("NameExport"));
   if (anObj.ExportImage().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ExportImage"),anObj.ExportImage().Val())->ReTagThis("ExportImage"));
   if (anObj.ReuseResiduelle().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ReuseResiduelle"),anObj.ReuseResiduelle().Val())->ReTagThis("ReuseResiduelle"));
   if (anObj.FCND_ExportModeleGlobal().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("FCND_ExportModeleGlobal"),anObj.FCND_ExportModeleGlobal().Val())->ReTagThis("FCND_ExportModeleGlobal"));
   if (anObj.MailleExport().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("MailleExport"),anObj.MailleExport().Val())->ReTagThis("MailleExport"));
   if (anObj.UseHomologueReference().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("UseHomologueReference"),anObj.UseHomologueReference().Val())->ReTagThis("UseHomologueReference"));
   if (anObj.MakeImagePxRef().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("MakeImagePxRef"),anObj.MakeImagePxRef().Val())->ReTagThis("MakeImagePxRef"));
   if (anObj.NbPtMinValideEqOriRel().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NbPtMinValideEqOriRel"),anObj.NbPtMinValideEqOriRel().Val())->ReTagThis("NbPtMinValideEqOriRel"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cOneModeleAnalytique & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.UseIt(),aTree->Get("UseIt",1),bool(true)); //tototo 

   xml_init(anObj.KeyNuage3D(),aTree->Get("KeyNuage3D",1)); //tototo 

   xml_init(anObj.TypeModele(),aTree->Get("TypeModele",1)); //tototo 

   xml_init(anObj.HomographieL2(),aTree->Get("HomographieL2",1),bool(true)); //tototo 

   xml_init(anObj.PolynomeL2(),aTree->Get("PolynomeL2",1),bool(true)); //tototo 

   xml_init(anObj.DegrePol(),aTree->Get("DegrePol",1)); //tototo 

   xml_init(anObj.NumsAngleFiges(),aTree->GetAll("NumsAngleFiges",false,1));

   xml_init(anObj.L1CalcOri(),aTree->Get("L1CalcOri",1),bool(false)); //tototo 

   xml_init(anObj.AutomSelExportOri(),aTree->Get("AutomSelExportOri",1)); //tototo 

   xml_init(anObj.AutomNamesExportOri1(),aTree->Get("AutomNamesExportOri1",1)); //tototo 

   xml_init(anObj.AutomNamesExportOri2(),aTree->Get("AutomNamesExportOri2",1)); //tototo 

   xml_init(anObj.AutomNamesExportHomXml(),aTree->Get("AutomNamesExportHomXml",1)); //tototo 

   xml_init(anObj.AutomNamesExportHomTif(),aTree->Get("AutomNamesExportHomTif",1)); //tototo 

   xml_init(anObj.AutomNamesExportHomBin(),aTree->Get("AutomNamesExportHomBin",1)); //tototo 

   xml_init(anObj.AffineOrient(),aTree->Get("AffineOrient",1),bool(true)); //tototo 

   xml_init(anObj.KeyNamesExportHomXml(),aTree->Get("KeyNamesExportHomXml",1)); //tototo 

   xml_init(anObj.SigmaPixPdsExport(),aTree->Get("SigmaPixPdsExport",1)); //tototo 

   xml_init(anObj.FiltreByCorrel(),aTree->Get("FiltreByCorrel",1),bool(false)); //tototo 

   xml_init(anObj.SeuilFiltreCorrel(),aTree->Get("SeuilFiltreCorrel",1),double(0.2)); //tototo 

   xml_init(anObj.UseFCBySeuil(),aTree->Get("UseFCBySeuil",1),bool(true)); //tototo 

   xml_init(anObj.ExposantPondereCorrel(),aTree->Get("ExposantPondereCorrel",1),double(1.0)); //tototo 

   xml_init(anObj.ReCalclCorrelMultiEchelle(),aTree->GetAll("ReCalclCorrelMultiEchelle",false,1));

   xml_init(anObj.PasCalcul(),aTree->Get("PasCalcul",1)); //tototo 

   xml_init(anObj.PointUnique(),aTree->Get("PointUnique",1),bool(false)); //tototo 

   xml_init(anObj.ReuseModele(),aTree->Get("ReuseModele",1)); //tototo 

   xml_init(anObj.MakeExport(),aTree->Get("MakeExport",1),bool(true)); //tototo 

   xml_init(anObj.NameExport(),aTree->Get("NameExport",1),std::string("ModeleAnalytique")); //tototo 

   xml_init(anObj.ExportImage(),aTree->Get("ExportImage",1),bool(false)); //tototo 

   xml_init(anObj.ReuseResiduelle(),aTree->Get("ReuseResiduelle",1),bool(false)); //tototo 

   xml_init(anObj.FCND_ExportModeleGlobal(),aTree->Get("FCND_ExportModeleGlobal",1)); //tototo 

   xml_init(anObj.MailleExport(),aTree->Get("MailleExport",1),double(10.0)); //tototo 

   xml_init(anObj.UseHomologueReference(),aTree->Get("UseHomologueReference",1),bool(false)); //tototo 

   xml_init(anObj.MakeImagePxRef(),aTree->Get("MakeImagePxRef",1),bool(false)); //tototo 

   xml_init(anObj.NbPtMinValideEqOriRel(),aTree->Get("NbPtMinValideEqOriRel",1),int(6)); //tototo 
}

std::string  Mangling( cOneModeleAnalytique *) {return "986025220B75DCE4FC3F";};


std::list< cOneModeleAnalytique > & cModelesAnalytiques::OneModeleAnalytique()
{
   return mOneModeleAnalytique;
}

const std::list< cOneModeleAnalytique > & cModelesAnalytiques::OneModeleAnalytique()const 
{
   return mOneModeleAnalytique;
}

void  BinaryUnDumpFromFile(cModelesAnalytiques & anObj,ELISE_fp & aFp)
{
   { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cOneModeleAnalytique aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.OneModeleAnalytique().push_back(aVal);
        }
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cModelesAnalytiques & anObj)
{
    BinaryDumpInFile(aFp,(int)anObj.OneModeleAnalytique().size());
    for(  std::list< cOneModeleAnalytique >::const_iterator iT=anObj.OneModeleAnalytique().begin();
         iT!=anObj.OneModeleAnalytique().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
}

cElXMLTree * ToXMLTree(const cModelesAnalytiques & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ModelesAnalytiques",eXMLBranche);
  for
  (       std::list< cOneModeleAnalytique >::const_iterator it=anObj.OneModeleAnalytique().begin();
      it !=anObj.OneModeleAnalytique().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("OneModeleAnalytique"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cModelesAnalytiques & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.OneModeleAnalytique(),aTree->GetAll("OneModeleAnalytique",false,1));
}

std::string  Mangling( cModelesAnalytiques *) {return "A0AE4B6547356C9EF9BF";};


std::string & cByFileNomChantier::Prefixe()
{
   return mPrefixe;
}

const std::string & cByFileNomChantier::Prefixe()const 
{
   return mPrefixe;
}


cTplValGesInit< bool > & cByFileNomChantier::NomChantier()
{
   return mNomChantier;
}

const cTplValGesInit< bool > & cByFileNomChantier::NomChantier()const 
{
   return mNomChantier;
}


std::string & cByFileNomChantier::Postfixe()
{
   return mPostfixe;
}

const std::string & cByFileNomChantier::Postfixe()const 
{
   return mPostfixe;
}


cTplValGesInit< std::string > & cByFileNomChantier::NameTag()
{
   return mNameTag;
}

const cTplValGesInit< std::string > & cByFileNomChantier::NameTag()const 
{
   return mNameTag;
}

void  BinaryUnDumpFromFile(cByFileNomChantier & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Prefixe(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NomChantier().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NomChantier().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NomChantier().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.Postfixe(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NameTag().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NameTag().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NameTag().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cByFileNomChantier & anObj)
{
    BinaryDumpInFile(aFp,anObj.Prefixe());
    BinaryDumpInFile(aFp,anObj.NomChantier().IsInit());
    if (anObj.NomChantier().IsInit()) BinaryDumpInFile(aFp,anObj.NomChantier().Val());
    BinaryDumpInFile(aFp,anObj.Postfixe());
    BinaryDumpInFile(aFp,anObj.NameTag().IsInit());
    if (anObj.NameTag().IsInit()) BinaryDumpInFile(aFp,anObj.NameTag().Val());
}

cElXMLTree * ToXMLTree(const cByFileNomChantier & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ByFileNomChantier",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Prefixe"),anObj.Prefixe())->ReTagThis("Prefixe"));
   if (anObj.NomChantier().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NomChantier"),anObj.NomChantier().Val())->ReTagThis("NomChantier"));
   aRes->AddFils(::ToXMLTree(std::string("Postfixe"),anObj.Postfixe())->ReTagThis("Postfixe"));
   if (anObj.NameTag().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NameTag"),anObj.NameTag().Val())->ReTagThis("NameTag"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cByFileNomChantier & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.Prefixe(),aTree->Get("Prefixe",1)); //tototo 

   xml_init(anObj.NomChantier(),aTree->Get("NomChantier",1),bool(true)); //tototo 

   xml_init(anObj.Postfixe(),aTree->Get("Postfixe",1)); //tototo 

   xml_init(anObj.NameTag(),aTree->Get("NameTag",1),std::string("FileOriMnt")); //tototo 
}

std::string  Mangling( cByFileNomChantier *) {return "4CD06A7BC518E6AFFDBF";};


cTplValGesInit< cFileOriMnt > & cOri::Explicite()
{
   return mExplicite;
}

const cTplValGesInit< cFileOriMnt > & cOri::Explicite()const 
{
   return mExplicite;
}


std::string & cOri::Prefixe()
{
   return ByFileNomChantier().Val().Prefixe();
}

const std::string & cOri::Prefixe()const 
{
   return ByFileNomChantier().Val().Prefixe();
}


cTplValGesInit< bool > & cOri::NomChantier()
{
   return ByFileNomChantier().Val().NomChantier();
}

const cTplValGesInit< bool > & cOri::NomChantier()const 
{
   return ByFileNomChantier().Val().NomChantier();
}


std::string & cOri::Postfixe()
{
   return ByFileNomChantier().Val().Postfixe();
}

const std::string & cOri::Postfixe()const 
{
   return ByFileNomChantier().Val().Postfixe();
}


cTplValGesInit< std::string > & cOri::NameTag()
{
   return ByFileNomChantier().Val().NameTag();
}

const cTplValGesInit< std::string > & cOri::NameTag()const 
{
   return ByFileNomChantier().Val().NameTag();
}


cTplValGesInit< cByFileNomChantier > & cOri::ByFileNomChantier()
{
   return mByFileNomChantier;
}

const cTplValGesInit< cByFileNomChantier > & cOri::ByFileNomChantier()const 
{
   return mByFileNomChantier;
}

void  BinaryUnDumpFromFile(cOri & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Explicite().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Explicite().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Explicite().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ByFileNomChantier().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ByFileNomChantier().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ByFileNomChantier().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cOri & anObj)
{
    BinaryDumpInFile(aFp,anObj.Explicite().IsInit());
    if (anObj.Explicite().IsInit()) BinaryDumpInFile(aFp,anObj.Explicite().Val());
    BinaryDumpInFile(aFp,anObj.ByFileNomChantier().IsInit());
    if (anObj.ByFileNomChantier().IsInit()) BinaryDumpInFile(aFp,anObj.ByFileNomChantier().Val());
}

cElXMLTree * ToXMLTree(const cOri & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"Ori",eXMLBranche);
   if (anObj.Explicite().IsInit())
      aRes->AddFils(ToXMLTree(anObj.Explicite().Val())->ReTagThis("Explicite"));
   if (anObj.ByFileNomChantier().IsInit())
      aRes->AddFils(ToXMLTree(anObj.ByFileNomChantier().Val())->ReTagThis("ByFileNomChantier"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cOri & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.Explicite(),aTree->Get("Explicite",1)); //tototo 

   xml_init(anObj.ByFileNomChantier(),aTree->Get("ByFileNomChantier",1)); //tototo 
}

std::string  Mangling( cOri *) {return "DCA5C5CFFE6801ACFE3F";};


cTplValGesInit< cFileOriMnt > & cBasculeRes::Explicite()
{
   return Ori().Explicite();
}

const cTplValGesInit< cFileOriMnt > & cBasculeRes::Explicite()const 
{
   return Ori().Explicite();
}


std::string & cBasculeRes::Prefixe()
{
   return Ori().ByFileNomChantier().Val().Prefixe();
}

const std::string & cBasculeRes::Prefixe()const 
{
   return Ori().ByFileNomChantier().Val().Prefixe();
}


cTplValGesInit< bool > & cBasculeRes::NomChantier()
{
   return Ori().ByFileNomChantier().Val().NomChantier();
}

const cTplValGesInit< bool > & cBasculeRes::NomChantier()const 
{
   return Ori().ByFileNomChantier().Val().NomChantier();
}


std::string & cBasculeRes::Postfixe()
{
   return Ori().ByFileNomChantier().Val().Postfixe();
}

const std::string & cBasculeRes::Postfixe()const 
{
   return Ori().ByFileNomChantier().Val().Postfixe();
}


cTplValGesInit< std::string > & cBasculeRes::NameTag()
{
   return Ori().ByFileNomChantier().Val().NameTag();
}

const cTplValGesInit< std::string > & cBasculeRes::NameTag()const 
{
   return Ori().ByFileNomChantier().Val().NameTag();
}


cTplValGesInit< cByFileNomChantier > & cBasculeRes::ByFileNomChantier()
{
   return Ori().ByFileNomChantier();
}

const cTplValGesInit< cByFileNomChantier > & cBasculeRes::ByFileNomChantier()const 
{
   return Ori().ByFileNomChantier();
}


cOri & cBasculeRes::Ori()
{
   return mOri;
}

const cOri & cBasculeRes::Ori()const 
{
   return mOri;
}


cTplValGesInit< double > & cBasculeRes::OutValue()
{
   return mOutValue;
}

const cTplValGesInit< double > & cBasculeRes::OutValue()const 
{
   return mOutValue;
}

void  BinaryUnDumpFromFile(cBasculeRes & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Ori(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.OutValue().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.OutValue().ValForcedForUnUmp(),aFp);
        }
        else  anObj.OutValue().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cBasculeRes & anObj)
{
    BinaryDumpInFile(aFp,anObj.Ori());
    BinaryDumpInFile(aFp,anObj.OutValue().IsInit());
    if (anObj.OutValue().IsInit()) BinaryDumpInFile(aFp,anObj.OutValue().Val());
}

cElXMLTree * ToXMLTree(const cBasculeRes & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"BasculeRes",eXMLBranche);
   aRes->AddFils(ToXMLTree(anObj.Ori())->ReTagThis("Ori"));
   if (anObj.OutValue().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("OutValue"),anObj.OutValue().Val())->ReTagThis("OutValue"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cBasculeRes & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.Ori(),aTree->Get("Ori",1)); //tototo 

   xml_init(anObj.OutValue(),aTree->Get("OutValue",1),double(0.0)); //tototo 
}

std::string  Mangling( cBasculeRes *) {return "669F4215701328DEFE3F";};


std::string & cVisuSuperposMNT::NameFile()
{
   return mNameFile;
}

const std::string & cVisuSuperposMNT::NameFile()const 
{
   return mNameFile;
}


double & cVisuSuperposMNT::Seuil()
{
   return mSeuil;
}

const double & cVisuSuperposMNT::Seuil()const 
{
   return mSeuil;
}

void  BinaryUnDumpFromFile(cVisuSuperposMNT & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.NameFile(),aFp);
    BinaryUnDumpFromFile(anObj.Seuil(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cVisuSuperposMNT & anObj)
{
    BinaryDumpInFile(aFp,anObj.NameFile());
    BinaryDumpInFile(aFp,anObj.Seuil());
}

cElXMLTree * ToXMLTree(const cVisuSuperposMNT & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"VisuSuperposMNT",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("NameFile"),anObj.NameFile())->ReTagThis("NameFile"));
   aRes->AddFils(::ToXMLTree(std::string("Seuil"),anObj.Seuil())->ReTagThis("Seuil"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cVisuSuperposMNT & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.NameFile(),aTree->Get("NameFile",1)); //tototo 

   xml_init(anObj.Seuil(),aTree->Get("Seuil",1)); //tototo 
}

std::string  Mangling( cVisuSuperposMNT *) {return "AF0D394D57D447B4FE3F";};


cTplValGesInit< std::string > & cMakeMTDMaskOrtho::NameFileSauv()
{
   return mNameFileSauv;
}

const cTplValGesInit< std::string > & cMakeMTDMaskOrtho::NameFileSauv()const 
{
   return mNameFileSauv;
}


cMasqMesures & cMakeMTDMaskOrtho::Mesures()
{
   return mMesures;
}

const cMasqMesures & cMakeMTDMaskOrtho::Mesures()const 
{
   return mMesures;
}

void  BinaryUnDumpFromFile(cMakeMTDMaskOrtho & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NameFileSauv().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NameFileSauv().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NameFileSauv().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.Mesures(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cMakeMTDMaskOrtho & anObj)
{
    BinaryDumpInFile(aFp,anObj.NameFileSauv().IsInit());
    if (anObj.NameFileSauv().IsInit()) BinaryDumpInFile(aFp,anObj.NameFileSauv().Val());
    BinaryDumpInFile(aFp,anObj.Mesures());
}

cElXMLTree * ToXMLTree(const cMakeMTDMaskOrtho & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"MakeMTDMaskOrtho",eXMLBranche);
   if (anObj.NameFileSauv().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NameFileSauv"),anObj.NameFileSauv().Val())->ReTagThis("NameFileSauv"));
   aRes->AddFils(ToXMLTree(anObj.Mesures())->ReTagThis("Mesures"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cMakeMTDMaskOrtho & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.NameFileSauv(),aTree->Get("NameFileSauv",1),std::string("MTDMaskOrtho.xml")); //tototo 

   xml_init(anObj.Mesures(),aTree->Get("Mesures",1)); //tototo 
}

std::string  Mangling( cMakeMTDMaskOrtho *) {return "8C2EA811010194BCFE3F";};


double & cOrthoSinusCard::SzKernel()
{
   return mSzKernel;
}

const double & cOrthoSinusCard::SzKernel()const 
{
   return mSzKernel;
}


double & cOrthoSinusCard::SzApod()
{
   return mSzApod;
}

const double & cOrthoSinusCard::SzApod()const 
{
   return mSzApod;
}

void  BinaryUnDumpFromFile(cOrthoSinusCard & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.SzKernel(),aFp);
    BinaryUnDumpFromFile(anObj.SzApod(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cOrthoSinusCard & anObj)
{
    BinaryDumpInFile(aFp,anObj.SzKernel());
    BinaryDumpInFile(aFp,anObj.SzApod());
}

cElXMLTree * ToXMLTree(const cOrthoSinusCard & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"OrthoSinusCard",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("SzKernel"),anObj.SzKernel())->ReTagThis("SzKernel"));
   aRes->AddFils(::ToXMLTree(std::string("SzApod"),anObj.SzApod())->ReTagThis("SzApod"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cOrthoSinusCard & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.SzKernel(),aTree->Get("SzKernel",1)); //tototo 

   xml_init(anObj.SzApod(),aTree->Get("SzApod",1)); //tototo 
}

std::string  Mangling( cOrthoSinusCard *) {return "7146C17CCFD04C84FEBF";};


cTplValGesInit< std::string > & cMakeOrthoParImage::DirOrtho()
{
   return mDirOrtho;
}

const cTplValGesInit< std::string > & cMakeOrthoParImage::DirOrtho()const 
{
   return mDirOrtho;
}


cTplValGesInit< std::string > & cMakeOrthoParImage::FileMTD()
{
   return mFileMTD;
}

const cTplValGesInit< std::string > & cMakeOrthoParImage::FileMTD()const 
{
   return mFileMTD;
}


cTplValGesInit< std::string > & cMakeOrthoParImage::NameFileSauv()
{
   return MakeMTDMaskOrtho().Val().NameFileSauv();
}

const cTplValGesInit< std::string > & cMakeOrthoParImage::NameFileSauv()const 
{
   return MakeMTDMaskOrtho().Val().NameFileSauv();
}


cMasqMesures & cMakeOrthoParImage::Mesures()
{
   return MakeMTDMaskOrtho().Val().Mesures();
}

const cMasqMesures & cMakeOrthoParImage::Mesures()const 
{
   return MakeMTDMaskOrtho().Val().Mesures();
}


cTplValGesInit< cMakeMTDMaskOrtho > & cMakeOrthoParImage::MakeMTDMaskOrtho()
{
   return mMakeMTDMaskOrtho;
}

const cTplValGesInit< cMakeMTDMaskOrtho > & cMakeOrthoParImage::MakeMTDMaskOrtho()const 
{
   return mMakeMTDMaskOrtho;
}


cTplValGesInit< double > & cMakeOrthoParImage::OrthoBiCub()
{
   return mOrthoBiCub;
}

const cTplValGesInit< double > & cMakeOrthoParImage::OrthoBiCub()const 
{
   return mOrthoBiCub;
}


cTplValGesInit< double > & cMakeOrthoParImage::ScaleBiCub()
{
   return mScaleBiCub;
}

const cTplValGesInit< double > & cMakeOrthoParImage::ScaleBiCub()const 
{
   return mScaleBiCub;
}


cTplValGesInit< cOrthoSinusCard > & cMakeOrthoParImage::OrthoSinusCard()
{
   return mOrthoSinusCard;
}

const cTplValGesInit< cOrthoSinusCard > & cMakeOrthoParImage::OrthoSinusCard()const 
{
   return mOrthoSinusCard;
}


cTplValGesInit< double > & cMakeOrthoParImage::ResolRelOrhto()
{
   return mResolRelOrhto;
}

const cTplValGesInit< double > & cMakeOrthoParImage::ResolRelOrhto()const 
{
   return mResolRelOrhto;
}


cTplValGesInit< double > & cMakeOrthoParImage::ResolAbsOrtho()
{
   return mResolAbsOrtho;
}

const cTplValGesInit< double > & cMakeOrthoParImage::ResolAbsOrtho()const 
{
   return mResolAbsOrtho;
}


cTplValGesInit< Pt2dr > & cMakeOrthoParImage::PixelTerrainPhase()
{
   return mPixelTerrainPhase;
}

const cTplValGesInit< Pt2dr > & cMakeOrthoParImage::PixelTerrainPhase()const 
{
   return mPixelTerrainPhase;
}


std::string & cMakeOrthoParImage::KeyCalcInput()
{
   return mKeyCalcInput;
}

const std::string & cMakeOrthoParImage::KeyCalcInput()const 
{
   return mKeyCalcInput;
}


std::string & cMakeOrthoParImage::KeyCalcOutput()
{
   return mKeyCalcOutput;
}

const std::string & cMakeOrthoParImage::KeyCalcOutput()const 
{
   return mKeyCalcOutput;
}


cTplValGesInit< int > & cMakeOrthoParImage::NbChan()
{
   return mNbChan;
}

const cTplValGesInit< int > & cMakeOrthoParImage::NbChan()const 
{
   return mNbChan;
}


cTplValGesInit< std::string > & cMakeOrthoParImage::KeyCalcIncidHor()
{
   return mKeyCalcIncidHor;
}

const cTplValGesInit< std::string > & cMakeOrthoParImage::KeyCalcIncidHor()const 
{
   return mKeyCalcIncidHor;
}


cTplValGesInit< double > & cMakeOrthoParImage::SsResolIncH()
{
   return mSsResolIncH;
}

const cTplValGesInit< double > & cMakeOrthoParImage::SsResolIncH()const 
{
   return mSsResolIncH;
}


cTplValGesInit< bool > & cMakeOrthoParImage::CalcIncAZMoy()
{
   return mCalcIncAZMoy;
}

const cTplValGesInit< bool > & cMakeOrthoParImage::CalcIncAZMoy()const 
{
   return mCalcIncAZMoy;
}


cTplValGesInit< bool > & cMakeOrthoParImage::ImageIncIsDistFront()
{
   return mImageIncIsDistFront;
}

const cTplValGesInit< bool > & cMakeOrthoParImage::ImageIncIsDistFront()const 
{
   return mImageIncIsDistFront;
}


cTplValGesInit< int > & cMakeOrthoParImage::RepulsFront()
{
   return mRepulsFront;
}

const cTplValGesInit< int > & cMakeOrthoParImage::RepulsFront()const 
{
   return mRepulsFront;
}


cTplValGesInit< double > & cMakeOrthoParImage::ResolIm()
{
   return mResolIm;
}

const cTplValGesInit< double > & cMakeOrthoParImage::ResolIm()const 
{
   return mResolIm;
}


cTplValGesInit< Pt2di > & cMakeOrthoParImage::TranslateIm()
{
   return mTranslateIm;
}

const cTplValGesInit< Pt2di > & cMakeOrthoParImage::TranslateIm()const 
{
   return mTranslateIm;
}

void  BinaryUnDumpFromFile(cMakeOrthoParImage & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.DirOrtho().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.DirOrtho().ValForcedForUnUmp(),aFp);
        }
        else  anObj.DirOrtho().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.FileMTD().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.FileMTD().ValForcedForUnUmp(),aFp);
        }
        else  anObj.FileMTD().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.MakeMTDMaskOrtho().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.MakeMTDMaskOrtho().ValForcedForUnUmp(),aFp);
        }
        else  anObj.MakeMTDMaskOrtho().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.OrthoBiCub().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.OrthoBiCub().ValForcedForUnUmp(),aFp);
        }
        else  anObj.OrthoBiCub().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ScaleBiCub().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ScaleBiCub().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ScaleBiCub().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.OrthoSinusCard().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.OrthoSinusCard().ValForcedForUnUmp(),aFp);
        }
        else  anObj.OrthoSinusCard().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ResolRelOrhto().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ResolRelOrhto().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ResolRelOrhto().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ResolAbsOrtho().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ResolAbsOrtho().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ResolAbsOrtho().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.PixelTerrainPhase().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.PixelTerrainPhase().ValForcedForUnUmp(),aFp);
        }
        else  anObj.PixelTerrainPhase().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.KeyCalcInput(),aFp);
    BinaryUnDumpFromFile(anObj.KeyCalcOutput(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NbChan().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NbChan().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NbChan().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.KeyCalcIncidHor().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.KeyCalcIncidHor().ValForcedForUnUmp(),aFp);
        }
        else  anObj.KeyCalcIncidHor().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SsResolIncH().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SsResolIncH().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SsResolIncH().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.CalcIncAZMoy().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.CalcIncAZMoy().ValForcedForUnUmp(),aFp);
        }
        else  anObj.CalcIncAZMoy().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ImageIncIsDistFront().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ImageIncIsDistFront().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ImageIncIsDistFront().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.RepulsFront().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.RepulsFront().ValForcedForUnUmp(),aFp);
        }
        else  anObj.RepulsFront().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ResolIm().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ResolIm().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ResolIm().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.TranslateIm().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.TranslateIm().ValForcedForUnUmp(),aFp);
        }
        else  anObj.TranslateIm().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cMakeOrthoParImage & anObj)
{
    BinaryDumpInFile(aFp,anObj.DirOrtho().IsInit());
    if (anObj.DirOrtho().IsInit()) BinaryDumpInFile(aFp,anObj.DirOrtho().Val());
    BinaryDumpInFile(aFp,anObj.FileMTD().IsInit());
    if (anObj.FileMTD().IsInit()) BinaryDumpInFile(aFp,anObj.FileMTD().Val());
    BinaryDumpInFile(aFp,anObj.MakeMTDMaskOrtho().IsInit());
    if (anObj.MakeMTDMaskOrtho().IsInit()) BinaryDumpInFile(aFp,anObj.MakeMTDMaskOrtho().Val());
    BinaryDumpInFile(aFp,anObj.OrthoBiCub().IsInit());
    if (anObj.OrthoBiCub().IsInit()) BinaryDumpInFile(aFp,anObj.OrthoBiCub().Val());
    BinaryDumpInFile(aFp,anObj.ScaleBiCub().IsInit());
    if (anObj.ScaleBiCub().IsInit()) BinaryDumpInFile(aFp,anObj.ScaleBiCub().Val());
    BinaryDumpInFile(aFp,anObj.OrthoSinusCard().IsInit());
    if (anObj.OrthoSinusCard().IsInit()) BinaryDumpInFile(aFp,anObj.OrthoSinusCard().Val());
    BinaryDumpInFile(aFp,anObj.ResolRelOrhto().IsInit());
    if (anObj.ResolRelOrhto().IsInit()) BinaryDumpInFile(aFp,anObj.ResolRelOrhto().Val());
    BinaryDumpInFile(aFp,anObj.ResolAbsOrtho().IsInit());
    if (anObj.ResolAbsOrtho().IsInit()) BinaryDumpInFile(aFp,anObj.ResolAbsOrtho().Val());
    BinaryDumpInFile(aFp,anObj.PixelTerrainPhase().IsInit());
    if (anObj.PixelTerrainPhase().IsInit()) BinaryDumpInFile(aFp,anObj.PixelTerrainPhase().Val());
    BinaryDumpInFile(aFp,anObj.KeyCalcInput());
    BinaryDumpInFile(aFp,anObj.KeyCalcOutput());
    BinaryDumpInFile(aFp,anObj.NbChan().IsInit());
    if (anObj.NbChan().IsInit()) BinaryDumpInFile(aFp,anObj.NbChan().Val());
    BinaryDumpInFile(aFp,anObj.KeyCalcIncidHor().IsInit());
    if (anObj.KeyCalcIncidHor().IsInit()) BinaryDumpInFile(aFp,anObj.KeyCalcIncidHor().Val());
    BinaryDumpInFile(aFp,anObj.SsResolIncH().IsInit());
    if (anObj.SsResolIncH().IsInit()) BinaryDumpInFile(aFp,anObj.SsResolIncH().Val());
    BinaryDumpInFile(aFp,anObj.CalcIncAZMoy().IsInit());
    if (anObj.CalcIncAZMoy().IsInit()) BinaryDumpInFile(aFp,anObj.CalcIncAZMoy().Val());
    BinaryDumpInFile(aFp,anObj.ImageIncIsDistFront().IsInit());
    if (anObj.ImageIncIsDistFront().IsInit()) BinaryDumpInFile(aFp,anObj.ImageIncIsDistFront().Val());
    BinaryDumpInFile(aFp,anObj.RepulsFront().IsInit());
    if (anObj.RepulsFront().IsInit()) BinaryDumpInFile(aFp,anObj.RepulsFront().Val());
    BinaryDumpInFile(aFp,anObj.ResolIm().IsInit());
    if (anObj.ResolIm().IsInit()) BinaryDumpInFile(aFp,anObj.ResolIm().Val());
    BinaryDumpInFile(aFp,anObj.TranslateIm().IsInit());
    if (anObj.TranslateIm().IsInit()) BinaryDumpInFile(aFp,anObj.TranslateIm().Val());
}

cElXMLTree * ToXMLTree(const cMakeOrthoParImage & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"MakeOrthoParImage",eXMLBranche);
   if (anObj.DirOrtho().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DirOrtho"),anObj.DirOrtho().Val())->ReTagThis("DirOrtho"));
   if (anObj.FileMTD().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("FileMTD"),anObj.FileMTD().Val())->ReTagThis("FileMTD"));
   if (anObj.MakeMTDMaskOrtho().IsInit())
      aRes->AddFils(ToXMLTree(anObj.MakeMTDMaskOrtho().Val())->ReTagThis("MakeMTDMaskOrtho"));
   if (anObj.OrthoBiCub().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("OrthoBiCub"),anObj.OrthoBiCub().Val())->ReTagThis("OrthoBiCub"));
   if (anObj.ScaleBiCub().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ScaleBiCub"),anObj.ScaleBiCub().Val())->ReTagThis("ScaleBiCub"));
   if (anObj.OrthoSinusCard().IsInit())
      aRes->AddFils(ToXMLTree(anObj.OrthoSinusCard().Val())->ReTagThis("OrthoSinusCard"));
   if (anObj.ResolRelOrhto().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ResolRelOrhto"),anObj.ResolRelOrhto().Val())->ReTagThis("ResolRelOrhto"));
   if (anObj.ResolAbsOrtho().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ResolAbsOrtho"),anObj.ResolAbsOrtho().Val())->ReTagThis("ResolAbsOrtho"));
   if (anObj.PixelTerrainPhase().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PixelTerrainPhase"),anObj.PixelTerrainPhase().Val())->ReTagThis("PixelTerrainPhase"));
   aRes->AddFils(::ToXMLTree(std::string("KeyCalcInput"),anObj.KeyCalcInput())->ReTagThis("KeyCalcInput"));
   aRes->AddFils(::ToXMLTree(std::string("KeyCalcOutput"),anObj.KeyCalcOutput())->ReTagThis("KeyCalcOutput"));
   if (anObj.NbChan().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NbChan"),anObj.NbChan().Val())->ReTagThis("NbChan"));
   if (anObj.KeyCalcIncidHor().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("KeyCalcIncidHor"),anObj.KeyCalcIncidHor().Val())->ReTagThis("KeyCalcIncidHor"));
   if (anObj.SsResolIncH().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SsResolIncH"),anObj.SsResolIncH().Val())->ReTagThis("SsResolIncH"));
   if (anObj.CalcIncAZMoy().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("CalcIncAZMoy"),anObj.CalcIncAZMoy().Val())->ReTagThis("CalcIncAZMoy"));
   if (anObj.ImageIncIsDistFront().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ImageIncIsDistFront"),anObj.ImageIncIsDistFront().Val())->ReTagThis("ImageIncIsDistFront"));
   if (anObj.RepulsFront().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("RepulsFront"),anObj.RepulsFront().Val())->ReTagThis("RepulsFront"));
   if (anObj.ResolIm().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ResolIm"),anObj.ResolIm().Val())->ReTagThis("ResolIm"));
   if (anObj.TranslateIm().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("TranslateIm"),anObj.TranslateIm().Val())->ReTagThis("TranslateIm"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cMakeOrthoParImage & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.DirOrtho(),aTree->Get("DirOrtho",1),std::string("ORTHO/")); //tototo 

   xml_init(anObj.FileMTD(),aTree->Get("FileMTD",1),std::string("MTDOrtho.xml")); //tototo 

   xml_init(anObj.MakeMTDMaskOrtho(),aTree->Get("MakeMTDMaskOrtho",1)); //tototo 

   xml_init(anObj.OrthoBiCub(),aTree->Get("OrthoBiCub",1),double(-0.5)); //tototo 

   xml_init(anObj.ScaleBiCub(),aTree->Get("ScaleBiCub",1),double(1)); //tototo 

   xml_init(anObj.OrthoSinusCard(),aTree->Get("OrthoSinusCard",1)); //tototo 

   xml_init(anObj.ResolRelOrhto(),aTree->Get("ResolRelOrhto",1)); //tototo 

   xml_init(anObj.ResolAbsOrtho(),aTree->Get("ResolAbsOrtho",1)); //tototo 

   xml_init(anObj.PixelTerrainPhase(),aTree->Get("PixelTerrainPhase",1)); //tototo 

   xml_init(anObj.KeyCalcInput(),aTree->Get("KeyCalcInput",1)); //tototo 

   xml_init(anObj.KeyCalcOutput(),aTree->Get("KeyCalcOutput",1)); //tototo 

   xml_init(anObj.NbChan(),aTree->Get("NbChan",1),int(-1)); //tototo 

   xml_init(anObj.KeyCalcIncidHor(),aTree->Get("KeyCalcIncidHor",1)); //tototo 

   xml_init(anObj.SsResolIncH(),aTree->Get("SsResolIncH",1),double(10)); //tototo 

   xml_init(anObj.CalcIncAZMoy(),aTree->Get("CalcIncAZMoy",1)); //tototo 

   xml_init(anObj.ImageIncIsDistFront(),aTree->Get("ImageIncIsDistFront",1),bool(false)); //tototo 

   xml_init(anObj.RepulsFront(),aTree->Get("RepulsFront",1),int(20)); //tototo 

   xml_init(anObj.ResolIm(),aTree->Get("ResolIm",1),double(1.0)); //tototo 

   xml_init(anObj.TranslateIm(),aTree->Get("TranslateIm",1),Pt2di(Pt2di(0,0))); //tototo 
}

std::string  Mangling( cMakeOrthoParImage *) {return "6F95AC68DE45CEA6FE3F";};


cTplValGesInit< bool > & cGenerePartiesCachees::UseIt()
{
   return mUseIt;
}

const cTplValGesInit< bool > & cGenerePartiesCachees::UseIt()const 
{
   return mUseIt;
}


cTplValGesInit< double > & cGenerePartiesCachees::PasDisc()
{
   return mPasDisc;
}

const cTplValGesInit< double > & cGenerePartiesCachees::PasDisc()const 
{
   return mPasDisc;
}


double & cGenerePartiesCachees::SeuilUsePC()
{
   return mSeuilUsePC;
}

const double & cGenerePartiesCachees::SeuilUsePC()const 
{
   return mSeuilUsePC;
}


cTplValGesInit< std::string > & cGenerePartiesCachees::KeyCalcPC()
{
   return mKeyCalcPC;
}

const cTplValGesInit< std::string > & cGenerePartiesCachees::KeyCalcPC()const 
{
   return mKeyCalcPC;
}


cTplValGesInit< bool > & cGenerePartiesCachees::AddChantierKPC()
{
   return mAddChantierKPC;
}

const cTplValGesInit< bool > & cGenerePartiesCachees::AddChantierKPC()const 
{
   return mAddChantierKPC;
}


cTplValGesInit< bool > & cGenerePartiesCachees::SupresExtChantierKPC()
{
   return mSupresExtChantierKPC;
}

const cTplValGesInit< bool > & cGenerePartiesCachees::SupresExtChantierKPC()const 
{
   return mSupresExtChantierKPC;
}


cTplValGesInit< bool > & cGenerePartiesCachees::Dequant()
{
   return mDequant;
}

const cTplValGesInit< bool > & cGenerePartiesCachees::Dequant()const 
{
   return mDequant;
}


cTplValGesInit< bool > & cGenerePartiesCachees::ByMkF()
{
   return mByMkF;
}

const cTplValGesInit< bool > & cGenerePartiesCachees::ByMkF()const 
{
   return mByMkF;
}


cTplValGesInit< std::string > & cGenerePartiesCachees::PatternApply()
{
   return mPatternApply;
}

const cTplValGesInit< std::string > & cGenerePartiesCachees::PatternApply()const 
{
   return mPatternApply;
}


std::string & cGenerePartiesCachees::NameFile()
{
   return VisuSuperposMNT().Val().NameFile();
}

const std::string & cGenerePartiesCachees::NameFile()const 
{
   return VisuSuperposMNT().Val().NameFile();
}


double & cGenerePartiesCachees::Seuil()
{
   return VisuSuperposMNT().Val().Seuil();
}

const double & cGenerePartiesCachees::Seuil()const 
{
   return VisuSuperposMNT().Val().Seuil();
}


cTplValGesInit< cVisuSuperposMNT > & cGenerePartiesCachees::VisuSuperposMNT()
{
   return mVisuSuperposMNT;
}

const cTplValGesInit< cVisuSuperposMNT > & cGenerePartiesCachees::VisuSuperposMNT()const 
{
   return mVisuSuperposMNT;
}


cTplValGesInit< bool > & cGenerePartiesCachees::BufXYZ()
{
   return mBufXYZ;
}

const cTplValGesInit< bool > & cGenerePartiesCachees::BufXYZ()const 
{
   return mBufXYZ;
}


cTplValGesInit< bool > & cGenerePartiesCachees::DoOnlyWhenNew()
{
   return mDoOnlyWhenNew;
}

const cTplValGesInit< bool > & cGenerePartiesCachees::DoOnlyWhenNew()const 
{
   return mDoOnlyWhenNew;
}


cTplValGesInit< int > & cGenerePartiesCachees::SzBloc()
{
   return mSzBloc;
}

const cTplValGesInit< int > & cGenerePartiesCachees::SzBloc()const 
{
   return mSzBloc;
}


cTplValGesInit< int > & cGenerePartiesCachees::SzBord()
{
   return mSzBord;
}

const cTplValGesInit< int > & cGenerePartiesCachees::SzBord()const 
{
   return mSzBord;
}


cTplValGesInit< bool > & cGenerePartiesCachees::ImSuperpMNT()
{
   return mImSuperpMNT;
}

const cTplValGesInit< bool > & cGenerePartiesCachees::ImSuperpMNT()const 
{
   return mImSuperpMNT;
}


cTplValGesInit< double > & cGenerePartiesCachees::ZMoy()
{
   return mZMoy;
}

const cTplValGesInit< double > & cGenerePartiesCachees::ZMoy()const 
{
   return mZMoy;
}


cTplValGesInit< cElRegex_Ptr > & cGenerePartiesCachees::FiltreName()
{
   return mFiltreName;
}

const cTplValGesInit< cElRegex_Ptr > & cGenerePartiesCachees::FiltreName()const 
{
   return mFiltreName;
}


cTplValGesInit< std::string > & cGenerePartiesCachees::DirOrtho()
{
   return MakeOrthoParImage().Val().DirOrtho();
}

const cTplValGesInit< std::string > & cGenerePartiesCachees::DirOrtho()const 
{
   return MakeOrthoParImage().Val().DirOrtho();
}


cTplValGesInit< std::string > & cGenerePartiesCachees::FileMTD()
{
   return MakeOrthoParImage().Val().FileMTD();
}

const cTplValGesInit< std::string > & cGenerePartiesCachees::FileMTD()const 
{
   return MakeOrthoParImage().Val().FileMTD();
}


cTplValGesInit< std::string > & cGenerePartiesCachees::NameFileSauv()
{
   return MakeOrthoParImage().Val().MakeMTDMaskOrtho().Val().NameFileSauv();
}

const cTplValGesInit< std::string > & cGenerePartiesCachees::NameFileSauv()const 
{
   return MakeOrthoParImage().Val().MakeMTDMaskOrtho().Val().NameFileSauv();
}


cMasqMesures & cGenerePartiesCachees::Mesures()
{
   return MakeOrthoParImage().Val().MakeMTDMaskOrtho().Val().Mesures();
}

const cMasqMesures & cGenerePartiesCachees::Mesures()const 
{
   return MakeOrthoParImage().Val().MakeMTDMaskOrtho().Val().Mesures();
}


cTplValGesInit< cMakeMTDMaskOrtho > & cGenerePartiesCachees::MakeMTDMaskOrtho()
{
   return MakeOrthoParImage().Val().MakeMTDMaskOrtho();
}

const cTplValGesInit< cMakeMTDMaskOrtho > & cGenerePartiesCachees::MakeMTDMaskOrtho()const 
{
   return MakeOrthoParImage().Val().MakeMTDMaskOrtho();
}


cTplValGesInit< double > & cGenerePartiesCachees::OrthoBiCub()
{
   return MakeOrthoParImage().Val().OrthoBiCub();
}

const cTplValGesInit< double > & cGenerePartiesCachees::OrthoBiCub()const 
{
   return MakeOrthoParImage().Val().OrthoBiCub();
}


cTplValGesInit< double > & cGenerePartiesCachees::ScaleBiCub()
{
   return MakeOrthoParImage().Val().ScaleBiCub();
}

const cTplValGesInit< double > & cGenerePartiesCachees::ScaleBiCub()const 
{
   return MakeOrthoParImage().Val().ScaleBiCub();
}


cTplValGesInit< cOrthoSinusCard > & cGenerePartiesCachees::OrthoSinusCard()
{
   return MakeOrthoParImage().Val().OrthoSinusCard();
}

const cTplValGesInit< cOrthoSinusCard > & cGenerePartiesCachees::OrthoSinusCard()const 
{
   return MakeOrthoParImage().Val().OrthoSinusCard();
}


cTplValGesInit< double > & cGenerePartiesCachees::ResolRelOrhto()
{
   return MakeOrthoParImage().Val().ResolRelOrhto();
}

const cTplValGesInit< double > & cGenerePartiesCachees::ResolRelOrhto()const 
{
   return MakeOrthoParImage().Val().ResolRelOrhto();
}


cTplValGesInit< double > & cGenerePartiesCachees::ResolAbsOrtho()
{
   return MakeOrthoParImage().Val().ResolAbsOrtho();
}

const cTplValGesInit< double > & cGenerePartiesCachees::ResolAbsOrtho()const 
{
   return MakeOrthoParImage().Val().ResolAbsOrtho();
}


cTplValGesInit< Pt2dr > & cGenerePartiesCachees::PixelTerrainPhase()
{
   return MakeOrthoParImage().Val().PixelTerrainPhase();
}

const cTplValGesInit< Pt2dr > & cGenerePartiesCachees::PixelTerrainPhase()const 
{
   return MakeOrthoParImage().Val().PixelTerrainPhase();
}


std::string & cGenerePartiesCachees::KeyCalcInput()
{
   return MakeOrthoParImage().Val().KeyCalcInput();
}

const std::string & cGenerePartiesCachees::KeyCalcInput()const 
{
   return MakeOrthoParImage().Val().KeyCalcInput();
}


std::string & cGenerePartiesCachees::KeyCalcOutput()
{
   return MakeOrthoParImage().Val().KeyCalcOutput();
}

const std::string & cGenerePartiesCachees::KeyCalcOutput()const 
{
   return MakeOrthoParImage().Val().KeyCalcOutput();
}


cTplValGesInit< int > & cGenerePartiesCachees::NbChan()
{
   return MakeOrthoParImage().Val().NbChan();
}

const cTplValGesInit< int > & cGenerePartiesCachees::NbChan()const 
{
   return MakeOrthoParImage().Val().NbChan();
}


cTplValGesInit< std::string > & cGenerePartiesCachees::KeyCalcIncidHor()
{
   return MakeOrthoParImage().Val().KeyCalcIncidHor();
}

const cTplValGesInit< std::string > & cGenerePartiesCachees::KeyCalcIncidHor()const 
{
   return MakeOrthoParImage().Val().KeyCalcIncidHor();
}


cTplValGesInit< double > & cGenerePartiesCachees::SsResolIncH()
{
   return MakeOrthoParImage().Val().SsResolIncH();
}

const cTplValGesInit< double > & cGenerePartiesCachees::SsResolIncH()const 
{
   return MakeOrthoParImage().Val().SsResolIncH();
}


cTplValGesInit< bool > & cGenerePartiesCachees::CalcIncAZMoy()
{
   return MakeOrthoParImage().Val().CalcIncAZMoy();
}

const cTplValGesInit< bool > & cGenerePartiesCachees::CalcIncAZMoy()const 
{
   return MakeOrthoParImage().Val().CalcIncAZMoy();
}


cTplValGesInit< bool > & cGenerePartiesCachees::ImageIncIsDistFront()
{
   return MakeOrthoParImage().Val().ImageIncIsDistFront();
}

const cTplValGesInit< bool > & cGenerePartiesCachees::ImageIncIsDistFront()const 
{
   return MakeOrthoParImage().Val().ImageIncIsDistFront();
}


cTplValGesInit< int > & cGenerePartiesCachees::RepulsFront()
{
   return MakeOrthoParImage().Val().RepulsFront();
}

const cTplValGesInit< int > & cGenerePartiesCachees::RepulsFront()const 
{
   return MakeOrthoParImage().Val().RepulsFront();
}


cTplValGesInit< double > & cGenerePartiesCachees::ResolIm()
{
   return MakeOrthoParImage().Val().ResolIm();
}

const cTplValGesInit< double > & cGenerePartiesCachees::ResolIm()const 
{
   return MakeOrthoParImage().Val().ResolIm();
}


cTplValGesInit< Pt2di > & cGenerePartiesCachees::TranslateIm()
{
   return MakeOrthoParImage().Val().TranslateIm();
}

const cTplValGesInit< Pt2di > & cGenerePartiesCachees::TranslateIm()const 
{
   return MakeOrthoParImage().Val().TranslateIm();
}


cTplValGesInit< cMakeOrthoParImage > & cGenerePartiesCachees::MakeOrthoParImage()
{
   return mMakeOrthoParImage;
}

const cTplValGesInit< cMakeOrthoParImage > & cGenerePartiesCachees::MakeOrthoParImage()const 
{
   return mMakeOrthoParImage;
}

void  BinaryUnDumpFromFile(cGenerePartiesCachees & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.UseIt().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.UseIt().ValForcedForUnUmp(),aFp);
        }
        else  anObj.UseIt().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.PasDisc().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.PasDisc().ValForcedForUnUmp(),aFp);
        }
        else  anObj.PasDisc().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.SeuilUsePC(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.KeyCalcPC().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.KeyCalcPC().ValForcedForUnUmp(),aFp);
        }
        else  anObj.KeyCalcPC().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.AddChantierKPC().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.AddChantierKPC().ValForcedForUnUmp(),aFp);
        }
        else  anObj.AddChantierKPC().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SupresExtChantierKPC().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SupresExtChantierKPC().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SupresExtChantierKPC().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Dequant().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Dequant().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Dequant().SetNoInit();
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
             anObj.PatternApply().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.PatternApply().ValForcedForUnUmp(),aFp);
        }
        else  anObj.PatternApply().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.VisuSuperposMNT().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.VisuSuperposMNT().ValForcedForUnUmp(),aFp);
        }
        else  anObj.VisuSuperposMNT().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.BufXYZ().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.BufXYZ().ValForcedForUnUmp(),aFp);
        }
        else  anObj.BufXYZ().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.DoOnlyWhenNew().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.DoOnlyWhenNew().ValForcedForUnUmp(),aFp);
        }
        else  anObj.DoOnlyWhenNew().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SzBloc().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SzBloc().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SzBloc().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SzBord().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SzBord().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SzBord().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ImSuperpMNT().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ImSuperpMNT().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ImSuperpMNT().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ZMoy().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ZMoy().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ZMoy().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.FiltreName().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.FiltreName().ValForcedForUnUmp(),aFp);
        }
        else  anObj.FiltreName().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.MakeOrthoParImage().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.MakeOrthoParImage().ValForcedForUnUmp(),aFp);
        }
        else  anObj.MakeOrthoParImage().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cGenerePartiesCachees & anObj)
{
    BinaryDumpInFile(aFp,anObj.UseIt().IsInit());
    if (anObj.UseIt().IsInit()) BinaryDumpInFile(aFp,anObj.UseIt().Val());
    BinaryDumpInFile(aFp,anObj.PasDisc().IsInit());
    if (anObj.PasDisc().IsInit()) BinaryDumpInFile(aFp,anObj.PasDisc().Val());
    BinaryDumpInFile(aFp,anObj.SeuilUsePC());
    BinaryDumpInFile(aFp,anObj.KeyCalcPC().IsInit());
    if (anObj.KeyCalcPC().IsInit()) BinaryDumpInFile(aFp,anObj.KeyCalcPC().Val());
    BinaryDumpInFile(aFp,anObj.AddChantierKPC().IsInit());
    if (anObj.AddChantierKPC().IsInit()) BinaryDumpInFile(aFp,anObj.AddChantierKPC().Val());
    BinaryDumpInFile(aFp,anObj.SupresExtChantierKPC().IsInit());
    if (anObj.SupresExtChantierKPC().IsInit()) BinaryDumpInFile(aFp,anObj.SupresExtChantierKPC().Val());
    BinaryDumpInFile(aFp,anObj.Dequant().IsInit());
    if (anObj.Dequant().IsInit()) BinaryDumpInFile(aFp,anObj.Dequant().Val());
    BinaryDumpInFile(aFp,anObj.ByMkF().IsInit());
    if (anObj.ByMkF().IsInit()) BinaryDumpInFile(aFp,anObj.ByMkF().Val());
    BinaryDumpInFile(aFp,anObj.PatternApply().IsInit());
    if (anObj.PatternApply().IsInit()) BinaryDumpInFile(aFp,anObj.PatternApply().Val());
    BinaryDumpInFile(aFp,anObj.VisuSuperposMNT().IsInit());
    if (anObj.VisuSuperposMNT().IsInit()) BinaryDumpInFile(aFp,anObj.VisuSuperposMNT().Val());
    BinaryDumpInFile(aFp,anObj.BufXYZ().IsInit());
    if (anObj.BufXYZ().IsInit()) BinaryDumpInFile(aFp,anObj.BufXYZ().Val());
    BinaryDumpInFile(aFp,anObj.DoOnlyWhenNew().IsInit());
    if (anObj.DoOnlyWhenNew().IsInit()) BinaryDumpInFile(aFp,anObj.DoOnlyWhenNew().Val());
    BinaryDumpInFile(aFp,anObj.SzBloc().IsInit());
    if (anObj.SzBloc().IsInit()) BinaryDumpInFile(aFp,anObj.SzBloc().Val());
    BinaryDumpInFile(aFp,anObj.SzBord().IsInit());
    if (anObj.SzBord().IsInit()) BinaryDumpInFile(aFp,anObj.SzBord().Val());
    BinaryDumpInFile(aFp,anObj.ImSuperpMNT().IsInit());
    if (anObj.ImSuperpMNT().IsInit()) BinaryDumpInFile(aFp,anObj.ImSuperpMNT().Val());
    BinaryDumpInFile(aFp,anObj.ZMoy().IsInit());
    if (anObj.ZMoy().IsInit()) BinaryDumpInFile(aFp,anObj.ZMoy().Val());
    BinaryDumpInFile(aFp,anObj.FiltreName().IsInit());
    if (anObj.FiltreName().IsInit()) BinaryDumpInFile(aFp,anObj.FiltreName().Val());
    BinaryDumpInFile(aFp,anObj.MakeOrthoParImage().IsInit());
    if (anObj.MakeOrthoParImage().IsInit()) BinaryDumpInFile(aFp,anObj.MakeOrthoParImage().Val());
}

cElXMLTree * ToXMLTree(const cGenerePartiesCachees & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"GenerePartiesCachees",eXMLBranche);
   if (anObj.UseIt().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("UseIt"),anObj.UseIt().Val())->ReTagThis("UseIt"));
   if (anObj.PasDisc().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PasDisc"),anObj.PasDisc().Val())->ReTagThis("PasDisc"));
   aRes->AddFils(::ToXMLTree(std::string("SeuilUsePC"),anObj.SeuilUsePC())->ReTagThis("SeuilUsePC"));
   if (anObj.KeyCalcPC().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("KeyCalcPC"),anObj.KeyCalcPC().Val())->ReTagThis("KeyCalcPC"));
   if (anObj.AddChantierKPC().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("AddChantierKPC"),anObj.AddChantierKPC().Val())->ReTagThis("AddChantierKPC"));
   if (anObj.SupresExtChantierKPC().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SupresExtChantierKPC"),anObj.SupresExtChantierKPC().Val())->ReTagThis("SupresExtChantierKPC"));
   if (anObj.Dequant().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Dequant"),anObj.Dequant().Val())->ReTagThis("Dequant"));
   if (anObj.ByMkF().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ByMkF"),anObj.ByMkF().Val())->ReTagThis("ByMkF"));
   if (anObj.PatternApply().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PatternApply"),anObj.PatternApply().Val())->ReTagThis("PatternApply"));
   if (anObj.VisuSuperposMNT().IsInit())
      aRes->AddFils(ToXMLTree(anObj.VisuSuperposMNT().Val())->ReTagThis("VisuSuperposMNT"));
   if (anObj.BufXYZ().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("BufXYZ"),anObj.BufXYZ().Val())->ReTagThis("BufXYZ"));
   if (anObj.DoOnlyWhenNew().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DoOnlyWhenNew"),anObj.DoOnlyWhenNew().Val())->ReTagThis("DoOnlyWhenNew"));
   if (anObj.SzBloc().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SzBloc"),anObj.SzBloc().Val())->ReTagThis("SzBloc"));
   if (anObj.SzBord().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SzBord"),anObj.SzBord().Val())->ReTagThis("SzBord"));
   if (anObj.ImSuperpMNT().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ImSuperpMNT"),anObj.ImSuperpMNT().Val())->ReTagThis("ImSuperpMNT"));
   if (anObj.ZMoy().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ZMoy"),anObj.ZMoy().Val())->ReTagThis("ZMoy"));
   if (anObj.FiltreName().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("FiltreName"),anObj.FiltreName().Val())->ReTagThis("FiltreName"));
   if (anObj.MakeOrthoParImage().IsInit())
      aRes->AddFils(ToXMLTree(anObj.MakeOrthoParImage().Val())->ReTagThis("MakeOrthoParImage"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cGenerePartiesCachees & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.UseIt(),aTree->Get("UseIt",1),bool(true)); //tototo 

   xml_init(anObj.PasDisc(),aTree->Get("PasDisc",1),double(1.0)); //tototo 

   xml_init(anObj.SeuilUsePC(),aTree->Get("SeuilUsePC",1)); //tototo 

   xml_init(anObj.KeyCalcPC(),aTree->Get("KeyCalcPC",1)); //tototo 

   xml_init(anObj.AddChantierKPC(),aTree->Get("AddChantierKPC",1),bool(false)); //tototo 

   xml_init(anObj.SupresExtChantierKPC(),aTree->Get("SupresExtChantierKPC",1),bool(true)); //tototo 

   xml_init(anObj.Dequant(),aTree->Get("Dequant",1)); //tototo 

   xml_init(anObj.ByMkF(),aTree->Get("ByMkF",1),bool(true)); //tototo 

   xml_init(anObj.PatternApply(),aTree->Get("PatternApply",1),std::string(".*")); //tototo 

   xml_init(anObj.VisuSuperposMNT(),aTree->Get("VisuSuperposMNT",1)); //tototo 

   xml_init(anObj.BufXYZ(),aTree->Get("BufXYZ",1),bool(true)); //tototo 

   xml_init(anObj.DoOnlyWhenNew(),aTree->Get("DoOnlyWhenNew",1),bool(false)); //tototo 

   xml_init(anObj.SzBloc(),aTree->Get("SzBloc",1),int(4000)); //tototo 

   xml_init(anObj.SzBord(),aTree->Get("SzBord",1),int(300)); //tototo 

   xml_init(anObj.ImSuperpMNT(),aTree->Get("ImSuperpMNT",1),bool(false)); //tototo 

   xml_init(anObj.ZMoy(),aTree->Get("ZMoy",1)); //tototo 

   xml_init(anObj.FiltreName(),aTree->Get("FiltreName",1)); //tototo 

   xml_init(anObj.MakeOrthoParImage(),aTree->Get("MakeOrthoParImage",1)); //tototo 
}

std::string  Mangling( cGenerePartiesCachees *) {return "228295E9B58F678EFD3F";};


std::string & cRedrLocAnam::NameOut()
{
   return mNameOut;
}

const std::string & cRedrLocAnam::NameOut()const 
{
   return mNameOut;
}


std::string & cRedrLocAnam::NameMasq()
{
   return mNameMasq;
}

const std::string & cRedrLocAnam::NameMasq()const 
{
   return mNameMasq;
}


std::string & cRedrLocAnam::NameOriGlob()
{
   return mNameOriGlob;
}

const std::string & cRedrLocAnam::NameOriGlob()const 
{
   return mNameOriGlob;
}


cTplValGesInit< std::string > & cRedrLocAnam::NameNuage()
{
   return mNameNuage;
}

const cTplValGesInit< std::string > & cRedrLocAnam::NameNuage()const 
{
   return mNameNuage;
}


cTplValGesInit< int > & cRedrLocAnam::XRecouvrt()
{
   return mXRecouvrt;
}

const cTplValGesInit< int > & cRedrLocAnam::XRecouvrt()const 
{
   return mXRecouvrt;
}


cTplValGesInit< double > & cRedrLocAnam::MemAvalaible()
{
   return mMemAvalaible;
}

const cTplValGesInit< double > & cRedrLocAnam::MemAvalaible()const 
{
   return mMemAvalaible;
}


cTplValGesInit< double > & cRedrLocAnam::FilterMulLargY()
{
   return mFilterMulLargY;
}

const cTplValGesInit< double > & cRedrLocAnam::FilterMulLargY()const 
{
   return mFilterMulLargY;
}


cTplValGesInit< double > & cRedrLocAnam::NbIterFilterY()
{
   return mNbIterFilterY;
}

const cTplValGesInit< double > & cRedrLocAnam::NbIterFilterY()const 
{
   return mNbIterFilterY;
}


cTplValGesInit< int > & cRedrLocAnam::FilterXY()
{
   return mFilterXY;
}

const cTplValGesInit< int > & cRedrLocAnam::FilterXY()const 
{
   return mFilterXY;
}


cTplValGesInit< int > & cRedrLocAnam::NbIterXY()
{
   return mNbIterXY;
}

const cTplValGesInit< int > & cRedrLocAnam::NbIterXY()const 
{
   return mNbIterXY;
}


cTplValGesInit< double > & cRedrLocAnam::DensityHighThresh()
{
   return mDensityHighThresh;
}

const cTplValGesInit< double > & cRedrLocAnam::DensityHighThresh()const 
{
   return mDensityHighThresh;
}


cTplValGesInit< double > & cRedrLocAnam::DensityLowThresh()
{
   return mDensityLowThresh;
}

const cTplValGesInit< double > & cRedrLocAnam::DensityLowThresh()const 
{
   return mDensityLowThresh;
}


cTplValGesInit< bool > & cRedrLocAnam::UseAutoMask()
{
   return mUseAutoMask;
}

const cTplValGesInit< bool > & cRedrLocAnam::UseAutoMask()const 
{
   return mUseAutoMask;
}

void  BinaryUnDumpFromFile(cRedrLocAnam & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.NameOut(),aFp);
    BinaryUnDumpFromFile(anObj.NameMasq(),aFp);
    BinaryUnDumpFromFile(anObj.NameOriGlob(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NameNuage().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NameNuage().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NameNuage().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.XRecouvrt().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.XRecouvrt().ValForcedForUnUmp(),aFp);
        }
        else  anObj.XRecouvrt().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.MemAvalaible().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.MemAvalaible().ValForcedForUnUmp(),aFp);
        }
        else  anObj.MemAvalaible().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.FilterMulLargY().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.FilterMulLargY().ValForcedForUnUmp(),aFp);
        }
        else  anObj.FilterMulLargY().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NbIterFilterY().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NbIterFilterY().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NbIterFilterY().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.FilterXY().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.FilterXY().ValForcedForUnUmp(),aFp);
        }
        else  anObj.FilterXY().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NbIterXY().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NbIterXY().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NbIterXY().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.DensityHighThresh().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.DensityHighThresh().ValForcedForUnUmp(),aFp);
        }
        else  anObj.DensityHighThresh().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.DensityLowThresh().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.DensityLowThresh().ValForcedForUnUmp(),aFp);
        }
        else  anObj.DensityLowThresh().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.UseAutoMask().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.UseAutoMask().ValForcedForUnUmp(),aFp);
        }
        else  anObj.UseAutoMask().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cRedrLocAnam & anObj)
{
    BinaryDumpInFile(aFp,anObj.NameOut());
    BinaryDumpInFile(aFp,anObj.NameMasq());
    BinaryDumpInFile(aFp,anObj.NameOriGlob());
    BinaryDumpInFile(aFp,anObj.NameNuage().IsInit());
    if (anObj.NameNuage().IsInit()) BinaryDumpInFile(aFp,anObj.NameNuage().Val());
    BinaryDumpInFile(aFp,anObj.XRecouvrt().IsInit());
    if (anObj.XRecouvrt().IsInit()) BinaryDumpInFile(aFp,anObj.XRecouvrt().Val());
    BinaryDumpInFile(aFp,anObj.MemAvalaible().IsInit());
    if (anObj.MemAvalaible().IsInit()) BinaryDumpInFile(aFp,anObj.MemAvalaible().Val());
    BinaryDumpInFile(aFp,anObj.FilterMulLargY().IsInit());
    if (anObj.FilterMulLargY().IsInit()) BinaryDumpInFile(aFp,anObj.FilterMulLargY().Val());
    BinaryDumpInFile(aFp,anObj.NbIterFilterY().IsInit());
    if (anObj.NbIterFilterY().IsInit()) BinaryDumpInFile(aFp,anObj.NbIterFilterY().Val());
    BinaryDumpInFile(aFp,anObj.FilterXY().IsInit());
    if (anObj.FilterXY().IsInit()) BinaryDumpInFile(aFp,anObj.FilterXY().Val());
    BinaryDumpInFile(aFp,anObj.NbIterXY().IsInit());
    if (anObj.NbIterXY().IsInit()) BinaryDumpInFile(aFp,anObj.NbIterXY().Val());
    BinaryDumpInFile(aFp,anObj.DensityHighThresh().IsInit());
    if (anObj.DensityHighThresh().IsInit()) BinaryDumpInFile(aFp,anObj.DensityHighThresh().Val());
    BinaryDumpInFile(aFp,anObj.DensityLowThresh().IsInit());
    if (anObj.DensityLowThresh().IsInit()) BinaryDumpInFile(aFp,anObj.DensityLowThresh().Val());
    BinaryDumpInFile(aFp,anObj.UseAutoMask().IsInit());
    if (anObj.UseAutoMask().IsInit()) BinaryDumpInFile(aFp,anObj.UseAutoMask().Val());
}

cElXMLTree * ToXMLTree(const cRedrLocAnam & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"RedrLocAnam",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("NameOut"),anObj.NameOut())->ReTagThis("NameOut"));
   aRes->AddFils(::ToXMLTree(std::string("NameMasq"),anObj.NameMasq())->ReTagThis("NameMasq"));
   aRes->AddFils(::ToXMLTree(std::string("NameOriGlob"),anObj.NameOriGlob())->ReTagThis("NameOriGlob"));
   if (anObj.NameNuage().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NameNuage"),anObj.NameNuage().Val())->ReTagThis("NameNuage"));
   if (anObj.XRecouvrt().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("XRecouvrt"),anObj.XRecouvrt().Val())->ReTagThis("XRecouvrt"));
   if (anObj.MemAvalaible().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("MemAvalaible"),anObj.MemAvalaible().Val())->ReTagThis("MemAvalaible"));
   if (anObj.FilterMulLargY().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("FilterMulLargY"),anObj.FilterMulLargY().Val())->ReTagThis("FilterMulLargY"));
   if (anObj.NbIterFilterY().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NbIterFilterY"),anObj.NbIterFilterY().Val())->ReTagThis("NbIterFilterY"));
   if (anObj.FilterXY().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("FilterXY"),anObj.FilterXY().Val())->ReTagThis("FilterXY"));
   if (anObj.NbIterXY().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NbIterXY"),anObj.NbIterXY().Val())->ReTagThis("NbIterXY"));
   if (anObj.DensityHighThresh().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DensityHighThresh"),anObj.DensityHighThresh().Val())->ReTagThis("DensityHighThresh"));
   if (anObj.DensityLowThresh().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DensityLowThresh"),anObj.DensityLowThresh().Val())->ReTagThis("DensityLowThresh"));
   if (anObj.UseAutoMask().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("UseAutoMask"),anObj.UseAutoMask().Val())->ReTagThis("UseAutoMask"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cRedrLocAnam & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.NameOut(),aTree->Get("NameOut",1)); //tototo 

   xml_init(anObj.NameMasq(),aTree->Get("NameMasq",1)); //tototo 

   xml_init(anObj.NameOriGlob(),aTree->Get("NameOriGlob",1)); //tototo 

   xml_init(anObj.NameNuage(),aTree->Get("NameNuage",1)); //tototo 

   xml_init(anObj.XRecouvrt(),aTree->Get("XRecouvrt",1),int(300)); //tototo 

   xml_init(anObj.MemAvalaible(),aTree->Get("MemAvalaible",1),double(3e7)); //tototo 

   xml_init(anObj.FilterMulLargY(),aTree->Get("FilterMulLargY",1),double(3.0)); //tototo 

   xml_init(anObj.NbIterFilterY(),aTree->Get("NbIterFilterY",1),double(4)); //tototo 

   xml_init(anObj.FilterXY(),aTree->Get("FilterXY",1),int(2)); //tototo 

   xml_init(anObj.NbIterXY(),aTree->Get("NbIterXY",1),int(3)); //tototo 

   xml_init(anObj.DensityHighThresh(),aTree->Get("DensityHighThresh",1),double(0.5)); //tototo 

   xml_init(anObj.DensityLowThresh(),aTree->Get("DensityLowThresh",1),double(0.3)); //tototo 

   xml_init(anObj.UseAutoMask(),aTree->Get("UseAutoMask",1),bool(true)); //tototo 
}

std::string  Mangling( cRedrLocAnam *) {return "3E27EF94880BAA8FFF3F";};


std::string & cNuagePredicteur::KeyAssocIm2Nuage()
{
   return mKeyAssocIm2Nuage;
}

const std::string & cNuagePredicteur::KeyAssocIm2Nuage()const 
{
   return mKeyAssocIm2Nuage;
}


cTplValGesInit< std::string > & cNuagePredicteur::Selector()
{
   return mSelector;
}

const cTplValGesInit< std::string > & cNuagePredicteur::Selector()const 
{
   return mSelector;
}


double & cNuagePredicteur::ScaleNuage()
{
   return mScaleNuage;
}

const double & cNuagePredicteur::ScaleNuage()const 
{
   return mScaleNuage;
}

void  BinaryUnDumpFromFile(cNuagePredicteur & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.KeyAssocIm2Nuage(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Selector().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Selector().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Selector().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.ScaleNuage(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cNuagePredicteur & anObj)
{
    BinaryDumpInFile(aFp,anObj.KeyAssocIm2Nuage());
    BinaryDumpInFile(aFp,anObj.Selector().IsInit());
    if (anObj.Selector().IsInit()) BinaryDumpInFile(aFp,anObj.Selector().Val());
    BinaryDumpInFile(aFp,anObj.ScaleNuage());
}

cElXMLTree * ToXMLTree(const cNuagePredicteur & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"NuagePredicteur",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("KeyAssocIm2Nuage"),anObj.KeyAssocIm2Nuage())->ReTagThis("KeyAssocIm2Nuage"));
   if (anObj.Selector().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Selector"),anObj.Selector().Val())->ReTagThis("Selector"));
   aRes->AddFils(::ToXMLTree(std::string("ScaleNuage"),anObj.ScaleNuage())->ReTagThis("ScaleNuage"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cNuagePredicteur & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.KeyAssocIm2Nuage(),aTree->Get("KeyAssocIm2Nuage",1)); //tototo 

   xml_init(anObj.Selector(),aTree->Get("Selector",1),std::string(".*")); //tototo 

   xml_init(anObj.ScaleNuage(),aTree->Get("ScaleNuage",1)); //tototo 
}

std::string  Mangling( cNuagePredicteur *) {return "AC350D073557C4DDFD3F";};


int & cEtapeMEC::DeZoom()
{
   return mDeZoom;
}

const int & cEtapeMEC::DeZoom()const 
{
   return mDeZoom;
}


cTplValGesInit< double > & cEtapeMEC::EpsilonAddMoyenne()
{
   return CorrelAdHoc().Val().EpsilonAddMoyenne();
}

const cTplValGesInit< double > & cEtapeMEC::EpsilonAddMoyenne()const 
{
   return CorrelAdHoc().Val().EpsilonAddMoyenne();
}


cTplValGesInit< double > & cEtapeMEC::EpsilonMulMoyenne()
{
   return CorrelAdHoc().Val().EpsilonMulMoyenne();
}

const cTplValGesInit< double > & cEtapeMEC::EpsilonMulMoyenne()const 
{
   return CorrelAdHoc().Val().EpsilonMulMoyenne();
}


cTplValGesInit< int > & cEtapeMEC::SzBlocAH()
{
   return CorrelAdHoc().Val().SzBlocAH();
}

const cTplValGesInit< int > & cEtapeMEC::SzBlocAH()const 
{
   return CorrelAdHoc().Val().SzBlocAH();
}


cTplValGesInit< bool > & cEtapeMEC::UseGpGpu()
{
   return CorrelAdHoc().Val().CorrelMultiScale().Val().UseGpGpu();
}

const cTplValGesInit< bool > & cEtapeMEC::UseGpGpu()const 
{
   return CorrelAdHoc().Val().CorrelMultiScale().Val().UseGpGpu();
}


cTplValGesInit< bool > & cEtapeMEC::ModeDense()
{
   return CorrelAdHoc().Val().CorrelMultiScale().Val().ModeDense();
}

const cTplValGesInit< bool > & cEtapeMEC::ModeDense()const 
{
   return CorrelAdHoc().Val().CorrelMultiScale().Val().ModeDense();
}


cTplValGesInit< bool > & cEtapeMEC::UseWAdapt()
{
   return CorrelAdHoc().Val().CorrelMultiScale().Val().UseWAdapt();
}

const cTplValGesInit< bool > & cEtapeMEC::UseWAdapt()const 
{
   return CorrelAdHoc().Val().CorrelMultiScale().Val().UseWAdapt();
}


cTplValGesInit< bool > & cEtapeMEC::ModeMax()
{
   return CorrelAdHoc().Val().CorrelMultiScale().Val().ModeMax();
}

const cTplValGesInit< bool > & cEtapeMEC::ModeMax()const 
{
   return CorrelAdHoc().Val().CorrelMultiScale().Val().ModeMax();
}


std::vector< cOneParamCMS > & cEtapeMEC::OneParamCMS()
{
   return CorrelAdHoc().Val().CorrelMultiScale().Val().OneParamCMS();
}

const std::vector< cOneParamCMS > & cEtapeMEC::OneParamCMS()const 
{
   return CorrelAdHoc().Val().CorrelMultiScale().Val().OneParamCMS();
}


cTplValGesInit< cCorrelMultiScale > & cEtapeMEC::CorrelMultiScale()
{
   return CorrelAdHoc().Val().CorrelMultiScale();
}

const cTplValGesInit< cCorrelMultiScale > & cEtapeMEC::CorrelMultiScale()const 
{
   return CorrelAdHoc().Val().CorrelMultiScale();
}


cTplValGesInit< cCensusCost > & cEtapeMEC::CensusCost()
{
   return CorrelAdHoc().Val().TypeCAH().CensusCost();
}

const cTplValGesInit< cCensusCost > & cEtapeMEC::CensusCost()const 
{
   return CorrelAdHoc().Val().TypeCAH().CensusCost();
}


cTplValGesInit< cCorrel2DLeastSquare > & cEtapeMEC::Correl2DLeastSquare()
{
   return CorrelAdHoc().Val().TypeCAH().Correl2DLeastSquare();
}

const cTplValGesInit< cCorrel2DLeastSquare > & cEtapeMEC::Correl2DLeastSquare()const 
{
   return CorrelAdHoc().Val().TypeCAH().Correl2DLeastSquare();
}


cTplValGesInit< cGPU_Correl > & cEtapeMEC::GPU_Correl()
{
   return CorrelAdHoc().Val().TypeCAH().GPU_Correl();
}

const cTplValGesInit< cGPU_Correl > & cEtapeMEC::GPU_Correl()const 
{
   return CorrelAdHoc().Val().TypeCAH().GPU_Correl();
}


cTplValGesInit< cMutiCorrelOrthoExt > & cEtapeMEC::MutiCorrelOrthoExt()
{
   return CorrelAdHoc().Val().TypeCAH().MutiCorrelOrthoExt();
}

const cTplValGesInit< cMutiCorrelOrthoExt > & cEtapeMEC::MutiCorrelOrthoExt()const 
{
   return CorrelAdHoc().Val().TypeCAH().MutiCorrelOrthoExt();
}


cTplValGesInit< cGPU_CorrelBasik > & cEtapeMEC::GPU_CorrelBasik()
{
   return CorrelAdHoc().Val().TypeCAH().GPU_CorrelBasik();
}

const cTplValGesInit< cGPU_CorrelBasik > & cEtapeMEC::GPU_CorrelBasik()const 
{
   return CorrelAdHoc().Val().TypeCAH().GPU_CorrelBasik();
}


cTplValGesInit< cMultiCorrelPonctuel > & cEtapeMEC::MultiCorrelPonctuel()
{
   return CorrelAdHoc().Val().TypeCAH().MultiCorrelPonctuel();
}

const cTplValGesInit< cMultiCorrelPonctuel > & cEtapeMEC::MultiCorrelPonctuel()const 
{
   return CorrelAdHoc().Val().TypeCAH().MultiCorrelPonctuel();
}


cTplValGesInit< cScoreLearnedMMVII > & cEtapeMEC::ScoreLearnedMMVII()
{
   return CorrelAdHoc().Val().TypeCAH().ScoreLearnedMMVII();
}

const cTplValGesInit< cScoreLearnedMMVII > & cEtapeMEC::ScoreLearnedMMVII()const 
{
   return CorrelAdHoc().Val().TypeCAH().ScoreLearnedMMVII();
}


cTplValGesInit< cCorrel_Ponctuel2ImGeomI > & cEtapeMEC::Correl_Ponctuel2ImGeomI()
{
   return CorrelAdHoc().Val().TypeCAH().Correl_Ponctuel2ImGeomI();
}

const cTplValGesInit< cCorrel_Ponctuel2ImGeomI > & cEtapeMEC::Correl_Ponctuel2ImGeomI()const 
{
   return CorrelAdHoc().Val().TypeCAH().Correl_Ponctuel2ImGeomI();
}


cTplValGesInit< cCorrel_PonctuelleCroisee > & cEtapeMEC::Correl_PonctuelleCroisee()
{
   return CorrelAdHoc().Val().TypeCAH().Correl_PonctuelleCroisee();
}

const cTplValGesInit< cCorrel_PonctuelleCroisee > & cEtapeMEC::Correl_PonctuelleCroisee()const 
{
   return CorrelAdHoc().Val().TypeCAH().Correl_PonctuelleCroisee();
}


cTplValGesInit< cCorrel_MultiFen > & cEtapeMEC::Correl_MultiFen()
{
   return CorrelAdHoc().Val().TypeCAH().Correl_MultiFen();
}

const cTplValGesInit< cCorrel_MultiFen > & cEtapeMEC::Correl_MultiFen()const 
{
   return CorrelAdHoc().Val().TypeCAH().Correl_MultiFen();
}


cTplValGesInit< cCorrel_Correl_MNE_ZPredic > & cEtapeMEC::Correl_Correl_MNE_ZPredic()
{
   return CorrelAdHoc().Val().TypeCAH().Correl_Correl_MNE_ZPredic();
}

const cTplValGesInit< cCorrel_Correl_MNE_ZPredic > & cEtapeMEC::Correl_Correl_MNE_ZPredic()const 
{
   return CorrelAdHoc().Val().TypeCAH().Correl_Correl_MNE_ZPredic();
}


cTplValGesInit< cCorrel_NC_Robuste > & cEtapeMEC::Correl_NC_Robuste()
{
   return CorrelAdHoc().Val().TypeCAH().Correl_NC_Robuste();
}

const cTplValGesInit< cCorrel_NC_Robuste > & cEtapeMEC::Correl_NC_Robuste()const 
{
   return CorrelAdHoc().Val().TypeCAH().Correl_NC_Robuste();
}


cTplValGesInit< cMasqueAutoByTieP > & cEtapeMEC::MasqueAutoByTieP()
{
   return CorrelAdHoc().Val().TypeCAH().MasqueAutoByTieP();
}

const cTplValGesInit< cMasqueAutoByTieP > & cEtapeMEC::MasqueAutoByTieP()const 
{
   return CorrelAdHoc().Val().TypeCAH().MasqueAutoByTieP();
}


cTypeCAH & cEtapeMEC::TypeCAH()
{
   return CorrelAdHoc().Val().TypeCAH();
}

const cTypeCAH & cEtapeMEC::TypeCAH()const 
{
   return CorrelAdHoc().Val().TypeCAH();
}


cTplValGesInit< cCorrelAdHoc > & cEtapeMEC::CorrelAdHoc()
{
   return mCorrelAdHoc;
}

const cTplValGesInit< cCorrelAdHoc > & cEtapeMEC::CorrelAdHoc()const 
{
   return mCorrelAdHoc;
}


cTplValGesInit< cDoImageBSurH > & cEtapeMEC::DoImageBSurH()
{
   return mDoImageBSurH;
}

const cTplValGesInit< cDoImageBSurH > & cEtapeMEC::DoImageBSurH()const 
{
   return mDoImageBSurH;
}


bool & cEtapeMEC::DoRatio2Im()
{
   return DoStatResult().Val().DoRatio2Im();
}

const bool & cEtapeMEC::DoRatio2Im()const 
{
   return DoStatResult().Val().DoRatio2Im();
}


cTplValGesInit< cDoStatResult > & cEtapeMEC::DoStatResult()
{
   return mDoStatResult;
}

const cTplValGesInit< cDoStatResult > & cEtapeMEC::DoStatResult()const 
{
   return mDoStatResult;
}


std::list< cMasqOfEtape > & cEtapeMEC::MasqOfEtape()
{
   return mMasqOfEtape;
}

const std::list< cMasqOfEtape > & cEtapeMEC::MasqOfEtape()const 
{
   return mMasqOfEtape;
}


cTplValGesInit< int > & cEtapeMEC::SzRecouvrtDalles()
{
   return mSzRecouvrtDalles;
}

const cTplValGesInit< int > & cEtapeMEC::SzRecouvrtDalles()const 
{
   return mSzRecouvrtDalles;
}


cTplValGesInit< int > & cEtapeMEC::SzDalleMin()
{
   return mSzDalleMin;
}

const cTplValGesInit< int > & cEtapeMEC::SzDalleMin()const 
{
   return mSzDalleMin;
}


cTplValGesInit< int > & cEtapeMEC::SzDalleMax()
{
   return mSzDalleMax;
}

const cTplValGesInit< int > & cEtapeMEC::SzDalleMax()const 
{
   return mSzDalleMax;
}


cTplValGesInit< eModeDynamiqueCorrel > & cEtapeMEC::DynamiqueCorrel()
{
   return mDynamiqueCorrel;
}

const cTplValGesInit< eModeDynamiqueCorrel > & cEtapeMEC::DynamiqueCorrel()const 
{
   return mDynamiqueCorrel;
}


cTplValGesInit< double > & cEtapeMEC::CorrelMin()
{
   return mCorrelMin;
}

const cTplValGesInit< double > & cEtapeMEC::CorrelMin()const 
{
   return mCorrelMin;
}


cTplValGesInit< double > & cEtapeMEC::GammaCorrel()
{
   return mGammaCorrel;
}

const cTplValGesInit< double > & cEtapeMEC::GammaCorrel()const 
{
   return mGammaCorrel;
}


cTplValGesInit< eModeAggregCorr > & cEtapeMEC::AggregCorr()
{
   return mAggregCorr;
}

const cTplValGesInit< eModeAggregCorr > & cEtapeMEC::AggregCorr()const 
{
   return mAggregCorr;
}


cTplValGesInit< double > & cEtapeMEC::SzW()
{
   return mSzW;
}

const cTplValGesInit< double > & cEtapeMEC::SzW()const 
{
   return mSzW;
}


cTplValGesInit< bool > & cEtapeMEC::WSpecUseMasqGlob()
{
   return mWSpecUseMasqGlob;
}

const cTplValGesInit< bool > & cEtapeMEC::WSpecUseMasqGlob()const 
{
   return mWSpecUseMasqGlob;
}


cTplValGesInit< eTypeWinCorrel > & cEtapeMEC::TypeWCorr()
{
   return mTypeWCorr;
}

const cTplValGesInit< eTypeWinCorrel > & cEtapeMEC::TypeWCorr()const 
{
   return mTypeWCorr;
}


cTplValGesInit< double > & cEtapeMEC::SzWy()
{
   return mSzWy;
}

const cTplValGesInit< double > & cEtapeMEC::SzWy()const 
{
   return mSzWy;
}


cTplValGesInit< int > & cEtapeMEC::NbIterFenSpec()
{
   return mNbIterFenSpec;
}

const cTplValGesInit< int > & cEtapeMEC::NbIterFenSpec()const 
{
   return mNbIterFenSpec;
}


std::list< cSpecFitrageImage > & cEtapeMEC::FiltreImageLoc()
{
   return mFiltreImageLoc;
}

const std::list< cSpecFitrageImage > & cEtapeMEC::FiltreImageLoc()const 
{
   return mFiltreImageLoc;
}


cTplValGesInit< int > & cEtapeMEC::SzWInt()
{
   return mSzWInt;
}

const cTplValGesInit< int > & cEtapeMEC::SzWInt()const 
{
   return mSzWInt;
}


cTplValGesInit< int > & cEtapeMEC::SurEchWCor()
{
   return mSurEchWCor;
}

const cTplValGesInit< int > & cEtapeMEC::SurEchWCor()const 
{
   return mSurEchWCor;
}


cTplValGesInit< eAlgoRegul > & cEtapeMEC::AlgoRegul()
{
   return mAlgoRegul;
}

const cTplValGesInit< eAlgoRegul > & cEtapeMEC::AlgoRegul()const 
{
   return mAlgoRegul;
}


cTplValGesInit< bool > & cEtapeMEC::ExportZAbs()
{
   return mExportZAbs;
}

const cTplValGesInit< bool > & cEtapeMEC::ExportZAbs()const 
{
   return mExportZAbs;
}


cTplValGesInit< eAlgoRegul > & cEtapeMEC::AlgoWenCxRImpossible()
{
   return mAlgoWenCxRImpossible;
}

const cTplValGesInit< eAlgoRegul > & cEtapeMEC::AlgoWenCxRImpossible()const 
{
   return mAlgoWenCxRImpossible;
}


cTplValGesInit< bool > & cEtapeMEC::CoxRoy8Cnx()
{
   return mCoxRoy8Cnx;
}

const cTplValGesInit< bool > & cEtapeMEC::CoxRoy8Cnx()const 
{
   return mCoxRoy8Cnx;
}


cTplValGesInit< bool > & cEtapeMEC::CoxRoyUChar()
{
   return mCoxRoyUChar;
}

const cTplValGesInit< bool > & cEtapeMEC::CoxRoyUChar()const 
{
   return mCoxRoyUChar;
}


std::list< cEtapeProgDyn > & cEtapeMEC::EtapeProgDyn()
{
   return ModulationProgDyn().Val().EtapeProgDyn();
}

const std::list< cEtapeProgDyn > & cEtapeMEC::EtapeProgDyn()const 
{
   return ModulationProgDyn().Val().EtapeProgDyn();
}


cTplValGesInit< double > & cEtapeMEC::Px1PenteMax()
{
   return ModulationProgDyn().Val().Px1PenteMax();
}

const cTplValGesInit< double > & cEtapeMEC::Px1PenteMax()const 
{
   return ModulationProgDyn().Val().Px1PenteMax();
}


cTplValGesInit< double > & cEtapeMEC::Px2PenteMax()
{
   return ModulationProgDyn().Val().Px2PenteMax();
}

const cTplValGesInit< double > & cEtapeMEC::Px2PenteMax()const 
{
   return ModulationProgDyn().Val().Px2PenteMax();
}


cTplValGesInit< bool > & cEtapeMEC::ChoixNewProg()
{
   return ModulationProgDyn().Val().ChoixNewProg();
}

const cTplValGesInit< bool > & cEtapeMEC::ChoixNewProg()const 
{
   return ModulationProgDyn().Val().ChoixNewProg();
}


double & cEtapeMEC::ValDefCorrel()
{
   return ModulationProgDyn().Val().ArgMaskAuto().Val().ValDefCorrel();
}

const double & cEtapeMEC::ValDefCorrel()const 
{
   return ModulationProgDyn().Val().ArgMaskAuto().Val().ValDefCorrel();
}


double & cEtapeMEC::CostTrans()
{
   return ModulationProgDyn().Val().ArgMaskAuto().Val().CostTrans();
}

const double & cEtapeMEC::CostTrans()const 
{
   return ModulationProgDyn().Val().ArgMaskAuto().Val().CostTrans();
}


cTplValGesInit< bool > & cEtapeMEC::ReInjectMask()
{
   return ModulationProgDyn().Val().ArgMaskAuto().Val().ReInjectMask();
}

const cTplValGesInit< bool > & cEtapeMEC::ReInjectMask()const 
{
   return ModulationProgDyn().Val().ArgMaskAuto().Val().ReInjectMask();
}


cTplValGesInit< double > & cEtapeMEC::AmplKLPostTr()
{
   return ModulationProgDyn().Val().ArgMaskAuto().Val().AmplKLPostTr();
}

const cTplValGesInit< double > & cEtapeMEC::AmplKLPostTr()const 
{
   return ModulationProgDyn().Val().ArgMaskAuto().Val().AmplKLPostTr();
}


cTplValGesInit< int > & cEtapeMEC::Erod32Mask()
{
   return ModulationProgDyn().Val().ArgMaskAuto().Val().Erod32Mask();
}

const cTplValGesInit< int > & cEtapeMEC::Erod32Mask()const 
{
   return ModulationProgDyn().Val().ArgMaskAuto().Val().Erod32Mask();
}


cTplValGesInit< int > & cEtapeMEC::SzOpen32()
{
   return ModulationProgDyn().Val().ArgMaskAuto().Val().SzOpen32();
}

const cTplValGesInit< int > & cEtapeMEC::SzOpen32()const 
{
   return ModulationProgDyn().Val().ArgMaskAuto().Val().SzOpen32();
}


cTplValGesInit< int > & cEtapeMEC::SeuilZC()
{
   return ModulationProgDyn().Val().ArgMaskAuto().Val().SeuilZC();
}

const cTplValGesInit< int > & cEtapeMEC::SeuilZC()const 
{
   return ModulationProgDyn().Val().ArgMaskAuto().Val().SeuilZC();
}


double & cEtapeMEC::CostChangeEtiq()
{
   return ModulationProgDyn().Val().ArgMaskAuto().Val().EtiqBestImage().Val().CostChangeEtiq();
}

const double & cEtapeMEC::CostChangeEtiq()const 
{
   return ModulationProgDyn().Val().ArgMaskAuto().Val().EtiqBestImage().Val().CostChangeEtiq();
}


cTplValGesInit< bool > & cEtapeMEC::SauvEtiq()
{
   return ModulationProgDyn().Val().ArgMaskAuto().Val().EtiqBestImage().Val().SauvEtiq();
}

const cTplValGesInit< bool > & cEtapeMEC::SauvEtiq()const 
{
   return ModulationProgDyn().Val().ArgMaskAuto().Val().EtiqBestImage().Val().SauvEtiq();
}


cTplValGesInit< cEtiqBestImage > & cEtapeMEC::EtiqBestImage()
{
   return ModulationProgDyn().Val().ArgMaskAuto().Val().EtiqBestImage();
}

const cTplValGesInit< cEtiqBestImage > & cEtapeMEC::EtiqBestImage()const 
{
   return ModulationProgDyn().Val().ArgMaskAuto().Val().EtiqBestImage();
}


cTplValGesInit< cArgMaskAuto > & cEtapeMEC::ArgMaskAuto()
{
   return ModulationProgDyn().Val().ArgMaskAuto();
}

const cTplValGesInit< cArgMaskAuto > & cEtapeMEC::ArgMaskAuto()const 
{
   return ModulationProgDyn().Val().ArgMaskAuto();
}


cTplValGesInit< cModulationProgDyn > & cEtapeMEC::ModulationProgDyn()
{
   return mModulationProgDyn;
}

const cTplValGesInit< cModulationProgDyn > & cEtapeMEC::ModulationProgDyn()const 
{
   return mModulationProgDyn;
}


cTplValGesInit< int > & cEtapeMEC::SsResolOptim()
{
   return mSsResolOptim;
}

const cTplValGesInit< int > & cEtapeMEC::SsResolOptim()const 
{
   return mSsResolOptim;
}


cTplValGesInit< double > & cEtapeMEC::RatioDeZoomImage()
{
   return mRatioDeZoomImage;
}

const cTplValGesInit< double > & cEtapeMEC::RatioDeZoomImage()const 
{
   return mRatioDeZoomImage;
}


cTplValGesInit< int > & cEtapeMEC::NdDiscKerInterp()
{
   return mNdDiscKerInterp;
}

const cTplValGesInit< int > & cEtapeMEC::NdDiscKerInterp()const 
{
   return mNdDiscKerInterp;
}


cTplValGesInit< eModeInterpolation > & cEtapeMEC::ModeInterpolation()
{
   return mModeInterpolation;
}

const cTplValGesInit< eModeInterpolation > & cEtapeMEC::ModeInterpolation()const 
{
   return mModeInterpolation;
}


cTplValGesInit< double > & cEtapeMEC::CoefInterpolationBicubique()
{
   return mCoefInterpolationBicubique;
}

const cTplValGesInit< double > & cEtapeMEC::CoefInterpolationBicubique()const 
{
   return mCoefInterpolationBicubique;
}


cTplValGesInit< double > & cEtapeMEC::SzSinCard()
{
   return mSzSinCard;
}

const cTplValGesInit< double > & cEtapeMEC::SzSinCard()const 
{
   return mSzSinCard;
}


cTplValGesInit< double > & cEtapeMEC::SzAppodSinCard()
{
   return mSzAppodSinCard;
}

const cTplValGesInit< double > & cEtapeMEC::SzAppodSinCard()const 
{
   return mSzAppodSinCard;
}


cTplValGesInit< int > & cEtapeMEC::TailleFenetreSinusCardinal()
{
   return mTailleFenetreSinusCardinal;
}

const cTplValGesInit< int > & cEtapeMEC::TailleFenetreSinusCardinal()const 
{
   return mTailleFenetreSinusCardinal;
}


cTplValGesInit< bool > & cEtapeMEC::ApodisationSinusCardinal()
{
   return mApodisationSinusCardinal;
}

const cTplValGesInit< bool > & cEtapeMEC::ApodisationSinusCardinal()const 
{
   return mApodisationSinusCardinal;
}


cTplValGesInit< int > & cEtapeMEC::SzGeomDerivable()
{
   return mSzGeomDerivable;
}

const cTplValGesInit< int > & cEtapeMEC::SzGeomDerivable()const 
{
   return mSzGeomDerivable;
}


cTplValGesInit< double > & cEtapeMEC::SeuilAttenZRegul()
{
   return mSeuilAttenZRegul;
}

const cTplValGesInit< double > & cEtapeMEC::SeuilAttenZRegul()const 
{
   return mSeuilAttenZRegul;
}


cTplValGesInit< double > & cEtapeMEC::AttenRelatifSeuilZ()
{
   return mAttenRelatifSeuilZ;
}

const cTplValGesInit< double > & cEtapeMEC::AttenRelatifSeuilZ()const 
{
   return mAttenRelatifSeuilZ;
}


cTplValGesInit< double > & cEtapeMEC::ZRegul_Quad()
{
   return mZRegul_Quad;
}

const cTplValGesInit< double > & cEtapeMEC::ZRegul_Quad()const 
{
   return mZRegul_Quad;
}


cTplValGesInit< double > & cEtapeMEC::ZRegul()
{
   return mZRegul;
}

const cTplValGesInit< double > & cEtapeMEC::ZRegul()const 
{
   return mZRegul;
}


cTplValGesInit< double > & cEtapeMEC::ZPas()
{
   return mZPas;
}

const cTplValGesInit< double > & cEtapeMEC::ZPas()const 
{
   return mZPas;
}


cTplValGesInit< int > & cEtapeMEC::RabZDilatAltiMoins()
{
   return mRabZDilatAltiMoins;
}

const cTplValGesInit< int > & cEtapeMEC::RabZDilatAltiMoins()const 
{
   return mRabZDilatAltiMoins;
}


cTplValGesInit< int > & cEtapeMEC::RabZDilatPlaniMoins()
{
   return mRabZDilatPlaniMoins;
}

const cTplValGesInit< int > & cEtapeMEC::RabZDilatPlaniMoins()const 
{
   return mRabZDilatPlaniMoins;
}


cTplValGesInit< int > & cEtapeMEC::ZDilatAlti()
{
   return mZDilatAlti;
}

const cTplValGesInit< int > & cEtapeMEC::ZDilatAlti()const 
{
   return mZDilatAlti;
}


cTplValGesInit< int > & cEtapeMEC::ZDilatPlani()
{
   return mZDilatPlani;
}

const cTplValGesInit< int > & cEtapeMEC::ZDilatPlani()const 
{
   return mZDilatPlani;
}


cTplValGesInit< double > & cEtapeMEC::ZDilatPlaniPropPtsInt()
{
   return mZDilatPlaniPropPtsInt;
}

const cTplValGesInit< double > & cEtapeMEC::ZDilatPlaniPropPtsInt()const 
{
   return mZDilatPlaniPropPtsInt;
}


cTplValGesInit< bool > & cEtapeMEC::ZRedrPx()
{
   return mZRedrPx;
}

const cTplValGesInit< bool > & cEtapeMEC::ZRedrPx()const 
{
   return mZRedrPx;
}


cTplValGesInit< bool > & cEtapeMEC::ZDeqRedr()
{
   return mZDeqRedr;
}

const cTplValGesInit< bool > & cEtapeMEC::ZDeqRedr()const 
{
   return mZDeqRedr;
}


cTplValGesInit< int > & cEtapeMEC::RedrNbIterMed()
{
   return mRedrNbIterMed;
}

const cTplValGesInit< int > & cEtapeMEC::RedrNbIterMed()const 
{
   return mRedrNbIterMed;
}


cTplValGesInit< int > & cEtapeMEC::RedrSzMed()
{
   return mRedrSzMed;
}

const cTplValGesInit< int > & cEtapeMEC::RedrSzMed()const 
{
   return mRedrSzMed;
}


cTplValGesInit< bool > & cEtapeMEC::RedrSauvBrut()
{
   return mRedrSauvBrut;
}

const cTplValGesInit< bool > & cEtapeMEC::RedrSauvBrut()const 
{
   return mRedrSauvBrut;
}


cTplValGesInit< int > & cEtapeMEC::RedrNbIterMoy()
{
   return mRedrNbIterMoy;
}

const cTplValGesInit< int > & cEtapeMEC::RedrNbIterMoy()const 
{
   return mRedrNbIterMoy;
}


cTplValGesInit< int > & cEtapeMEC::RedrSzMoy()
{
   return mRedrSzMoy;
}

const cTplValGesInit< int > & cEtapeMEC::RedrSzMoy()const 
{
   return mRedrSzMoy;
}


cTplValGesInit< double > & cEtapeMEC::Px1Regul_Quad()
{
   return mPx1Regul_Quad;
}

const cTplValGesInit< double > & cEtapeMEC::Px1Regul_Quad()const 
{
   return mPx1Regul_Quad;
}


cTplValGesInit< double > & cEtapeMEC::Px1Regul()
{
   return mPx1Regul;
}

const cTplValGesInit< double > & cEtapeMEC::Px1Regul()const 
{
   return mPx1Regul;
}


cTplValGesInit< double > & cEtapeMEC::Px1Pas()
{
   return mPx1Pas;
}

const cTplValGesInit< double > & cEtapeMEC::Px1Pas()const 
{
   return mPx1Pas;
}


cTplValGesInit< int > & cEtapeMEC::Px1DilatAlti()
{
   return mPx1DilatAlti;
}

const cTplValGesInit< int > & cEtapeMEC::Px1DilatAlti()const 
{
   return mPx1DilatAlti;
}


cTplValGesInit< int > & cEtapeMEC::Px1DilatPlani()
{
   return mPx1DilatPlani;
}

const cTplValGesInit< int > & cEtapeMEC::Px1DilatPlani()const 
{
   return mPx1DilatPlani;
}


cTplValGesInit< double > & cEtapeMEC::Px1DilatPlaniPropPtsInt()
{
   return mPx1DilatPlaniPropPtsInt;
}

const cTplValGesInit< double > & cEtapeMEC::Px1DilatPlaniPropPtsInt()const 
{
   return mPx1DilatPlaniPropPtsInt;
}


cTplValGesInit< bool > & cEtapeMEC::Px1RedrPx()
{
   return mPx1RedrPx;
}

const cTplValGesInit< bool > & cEtapeMEC::Px1RedrPx()const 
{
   return mPx1RedrPx;
}


cTplValGesInit< bool > & cEtapeMEC::Px1DeqRedr()
{
   return mPx1DeqRedr;
}

const cTplValGesInit< bool > & cEtapeMEC::Px1DeqRedr()const 
{
   return mPx1DeqRedr;
}


cTplValGesInit< double > & cEtapeMEC::Px2Regul_Quad()
{
   return mPx2Regul_Quad;
}

const cTplValGesInit< double > & cEtapeMEC::Px2Regul_Quad()const 
{
   return mPx2Regul_Quad;
}


cTplValGesInit< double > & cEtapeMEC::Px2Regul()
{
   return mPx2Regul;
}

const cTplValGesInit< double > & cEtapeMEC::Px2Regul()const 
{
   return mPx2Regul;
}


cTplValGesInit< double > & cEtapeMEC::Px2Pas()
{
   return mPx2Pas;
}

const cTplValGesInit< double > & cEtapeMEC::Px2Pas()const 
{
   return mPx2Pas;
}


cTplValGesInit< int > & cEtapeMEC::Px2DilatAlti()
{
   return mPx2DilatAlti;
}

const cTplValGesInit< int > & cEtapeMEC::Px2DilatAlti()const 
{
   return mPx2DilatAlti;
}


cTplValGesInit< int > & cEtapeMEC::Px2DilatPlani()
{
   return mPx2DilatPlani;
}

const cTplValGesInit< int > & cEtapeMEC::Px2DilatPlani()const 
{
   return mPx2DilatPlani;
}


cTplValGesInit< double > & cEtapeMEC::Px2DilatPlaniPropPtsInt()
{
   return mPx2DilatPlaniPropPtsInt;
}

const cTplValGesInit< double > & cEtapeMEC::Px2DilatPlaniPropPtsInt()const 
{
   return mPx2DilatPlaniPropPtsInt;
}


cTplValGesInit< bool > & cEtapeMEC::Px2RedrPx()
{
   return mPx2RedrPx;
}

const cTplValGesInit< bool > & cEtapeMEC::Px2RedrPx()const 
{
   return mPx2RedrPx;
}


cTplValGesInit< bool > & cEtapeMEC::Px2DeqRedr()
{
   return mPx2DeqRedr;
}

const cTplValGesInit< bool > & cEtapeMEC::Px2DeqRedr()const 
{
   return mPx2DeqRedr;
}


std::list< cSpecFitrageImage > & cEtapeMEC::OneFitragePx()
{
   return PostFiltragePx().Val().OneFitragePx();
}

const std::list< cSpecFitrageImage > & cEtapeMEC::OneFitragePx()const 
{
   return PostFiltragePx().Val().OneFitragePx();
}


cTplValGesInit< cPostFiltragePx > & cEtapeMEC::PostFiltragePx()
{
   return mPostFiltragePx;
}

const cTplValGesInit< cPostFiltragePx > & cEtapeMEC::PostFiltragePx()const 
{
   return mPostFiltragePx;
}


double & cEtapeMEC::SzFiltre()
{
   return PostFiltrageDiscont().Val().SzFiltre();
}

const double & cEtapeMEC::SzFiltre()const 
{
   return PostFiltrageDiscont().Val().SzFiltre();
}


cTplValGesInit< int > & cEtapeMEC::NbIter()
{
   return PostFiltrageDiscont().Val().NbIter();
}

const cTplValGesInit< int > & cEtapeMEC::NbIter()const 
{
   return PostFiltrageDiscont().Val().NbIter();
}


cTplValGesInit< double > & cEtapeMEC::ExposPonderGrad()
{
   return PostFiltrageDiscont().Val().ExposPonderGrad();
}

const cTplValGesInit< double > & cEtapeMEC::ExposPonderGrad()const 
{
   return PostFiltrageDiscont().Val().ExposPonderGrad();
}


cTplValGesInit< double > & cEtapeMEC::DericheFactEPC()
{
   return PostFiltrageDiscont().Val().DericheFactEPC();
}

const cTplValGesInit< double > & cEtapeMEC::DericheFactEPC()const 
{
   return PostFiltrageDiscont().Val().DericheFactEPC();
}


cTplValGesInit< double > & cEtapeMEC::ValGradAtten()
{
   return PostFiltrageDiscont().Val().ValGradAtten();
}

const cTplValGesInit< double > & cEtapeMEC::ValGradAtten()const 
{
   return PostFiltrageDiscont().Val().ValGradAtten();
}


cTplValGesInit< double > & cEtapeMEC::ExposPonderCorr()
{
   return PostFiltrageDiscont().Val().ExposPonderCorr();
}

const cTplValGesInit< double > & cEtapeMEC::ExposPonderCorr()const 
{
   return PostFiltrageDiscont().Val().ExposPonderCorr();
}


cTplValGesInit< cPostFiltrageDiscont > & cEtapeMEC::PostFiltrageDiscont()
{
   return mPostFiltrageDiscont;
}

const cTplValGesInit< cPostFiltrageDiscont > & cEtapeMEC::PostFiltrageDiscont()const 
{
   return mPostFiltrageDiscont;
}


bool & cEtapeMEC::ModeExclusion()
{
   return ImageSelecteur().Val().ModeExclusion();
}

const bool & cEtapeMEC::ModeExclusion()const 
{
   return ImageSelecteur().Val().ModeExclusion();
}


std::list< std::string > & cEtapeMEC::PatternSel()
{
   return ImageSelecteur().Val().PatternSel();
}

const std::list< std::string > & cEtapeMEC::PatternSel()const 
{
   return ImageSelecteur().Val().PatternSel();
}


cTplValGesInit< cImageSelecteur > & cEtapeMEC::ImageSelecteur()
{
   return mImageSelecteur;
}

const cTplValGesInit< cImageSelecteur > & cEtapeMEC::ImageSelecteur()const 
{
   return mImageSelecteur;
}


cTplValGesInit< cParamGenereStrVois > & cEtapeMEC::RelSelecteur()
{
   return mRelSelecteur;
}

const cTplValGesInit< cParamGenereStrVois > & cEtapeMEC::RelSelecteur()const 
{
   return mRelSelecteur;
}


cTplValGesInit< bool > & cEtapeMEC::Gen8Bits_Px1()
{
   return mGen8Bits_Px1;
}

const cTplValGesInit< bool > & cEtapeMEC::Gen8Bits_Px1()const 
{
   return mGen8Bits_Px1;
}


cTplValGesInit< int > & cEtapeMEC::Offset8Bits_Px1()
{
   return mOffset8Bits_Px1;
}

const cTplValGesInit< int > & cEtapeMEC::Offset8Bits_Px1()const 
{
   return mOffset8Bits_Px1;
}


cTplValGesInit< double > & cEtapeMEC::Dyn8Bits_Px1()
{
   return mDyn8Bits_Px1;
}

const cTplValGesInit< double > & cEtapeMEC::Dyn8Bits_Px1()const 
{
   return mDyn8Bits_Px1;
}


cTplValGesInit< bool > & cEtapeMEC::Gen8Bits_Px2()
{
   return mGen8Bits_Px2;
}

const cTplValGesInit< bool > & cEtapeMEC::Gen8Bits_Px2()const 
{
   return mGen8Bits_Px2;
}


cTplValGesInit< int > & cEtapeMEC::Offset8Bits_Px2()
{
   return mOffset8Bits_Px2;
}

const cTplValGesInit< int > & cEtapeMEC::Offset8Bits_Px2()const 
{
   return mOffset8Bits_Px2;
}


cTplValGesInit< double > & cEtapeMEC::Dyn8Bits_Px2()
{
   return mDyn8Bits_Px2;
}

const cTplValGesInit< double > & cEtapeMEC::Dyn8Bits_Px2()const 
{
   return mDyn8Bits_Px2;
}


std::list< std::string > & cEtapeMEC::ArgGen8Bits()
{
   return mArgGen8Bits;
}

const std::list< std::string > & cEtapeMEC::ArgGen8Bits()const 
{
   return mArgGen8Bits;
}


cTplValGesInit< bool > & cEtapeMEC::GenFilePxRel()
{
   return mGenFilePxRel;
}

const cTplValGesInit< bool > & cEtapeMEC::GenFilePxRel()const 
{
   return mGenFilePxRel;
}


cTplValGesInit< bool > & cEtapeMEC::GenImagesCorrel()
{
   return mGenImagesCorrel;
}

const cTplValGesInit< bool > & cEtapeMEC::GenImagesCorrel()const 
{
   return mGenImagesCorrel;
}


cTplValGesInit< bool > & cEtapeMEC::GenCubeCorrel()
{
   return mGenCubeCorrel;
}

const cTplValGesInit< bool > & cEtapeMEC::GenCubeCorrel()const 
{
   return mGenCubeCorrel;
}


std::list< cGenerateProjectionInImages > & cEtapeMEC::GenerateProjectionInImages()
{
   return mGenerateProjectionInImages;
}

const std::list< cGenerateProjectionInImages > & cEtapeMEC::GenerateProjectionInImages()const 
{
   return mGenerateProjectionInImages;
}


double & cEtapeMEC::SsResolPx()
{
   return GenCorPxTransv().Val().SsResolPx();
}

const double & cEtapeMEC::SsResolPx()const 
{
   return GenCorPxTransv().Val().SsResolPx();
}


std::string & cEtapeMEC::NameXMLFile()
{
   return GenCorPxTransv().Val().NameXMLFile();
}

const std::string & cEtapeMEC::NameXMLFile()const 
{
   return GenCorPxTransv().Val().NameXMLFile();
}


cTplValGesInit< cGenCorPxTransv > & cEtapeMEC::GenCorPxTransv()
{
   return mGenCorPxTransv;
}

const cTplValGesInit< cGenCorPxTransv > & cEtapeMEC::GenCorPxTransv()const 
{
   return mGenCorPxTransv;
}


std::list< cGenereModeleRaster2Analytique > & cEtapeMEC::ExportAsModeleDist()
{
   return mExportAsModeleDist;
}

const std::list< cGenereModeleRaster2Analytique > & cEtapeMEC::ExportAsModeleDist()const 
{
   return mExportAsModeleDist;
}


cTplValGesInit< ePxApply > & cEtapeMEC::OptDif_PxApply()
{
   return mOptDif_PxApply;
}

const cTplValGesInit< ePxApply > & cEtapeMEC::OptDif_PxApply()const 
{
   return mOptDif_PxApply;
}


cTplValGesInit< bool > & cEtapeMEC::VisuTerrainIm()
{
   return InterfaceVisualisation().Val().VisuTerrainIm();
}

const cTplValGesInit< bool > & cEtapeMEC::VisuTerrainIm()const 
{
   return InterfaceVisualisation().Val().VisuTerrainIm();
}


cTplValGesInit< int > & cEtapeMEC::SzWTerr()
{
   return InterfaceVisualisation().Val().SzWTerr();
}

const cTplValGesInit< int > & cEtapeMEC::SzWTerr()const 
{
   return InterfaceVisualisation().Val().SzWTerr();
}


std::list< std::string > & cEtapeMEC::UnSelectedImage()
{
   return InterfaceVisualisation().Val().UnSelectedImage();
}

const std::list< std::string > & cEtapeMEC::UnSelectedImage()const 
{
   return InterfaceVisualisation().Val().UnSelectedImage();
}


Pt2di & cEtapeMEC::CentreVisuTerrain()
{
   return InterfaceVisualisation().Val().CentreVisuTerrain();
}

const Pt2di & cEtapeMEC::CentreVisuTerrain()const 
{
   return InterfaceVisualisation().Val().CentreVisuTerrain();
}


cTplValGesInit< int > & cEtapeMEC::ZoomTerr()
{
   return InterfaceVisualisation().Val().ZoomTerr();
}

const cTplValGesInit< int > & cEtapeMEC::ZoomTerr()const 
{
   return InterfaceVisualisation().Val().ZoomTerr();
}


cTplValGesInit< int > & cEtapeMEC::NbDiscHistoPartieFrac()
{
   return InterfaceVisualisation().Val().NbDiscHistoPartieFrac();
}

const cTplValGesInit< int > & cEtapeMEC::NbDiscHistoPartieFrac()const 
{
   return InterfaceVisualisation().Val().NbDiscHistoPartieFrac();
}


double & cEtapeMEC::CoutFrac()
{
   return InterfaceVisualisation().Val().SimulFrac().Val().CoutFrac();
}

const double & cEtapeMEC::CoutFrac()const 
{
   return InterfaceVisualisation().Val().SimulFrac().Val().CoutFrac();
}


cTplValGesInit< cSimulFrac > & cEtapeMEC::SimulFrac()
{
   return InterfaceVisualisation().Val().SimulFrac();
}

const cTplValGesInit< cSimulFrac > & cEtapeMEC::SimulFrac()const 
{
   return InterfaceVisualisation().Val().SimulFrac();
}


cTplValGesInit< cInterfaceVisualisation > & cEtapeMEC::InterfaceVisualisation()
{
   return mInterfaceVisualisation;
}

const cTplValGesInit< cInterfaceVisualisation > & cEtapeMEC::InterfaceVisualisation()const 
{
   return mInterfaceVisualisation;
}


std::list< cMMExportNuage > & cEtapeMEC::MMExportNuage()
{
   return mMMExportNuage;
}

const std::list< cMMExportNuage > & cEtapeMEC::MMExportNuage()const 
{
   return mMMExportNuage;
}


std::list< cOneModeleAnalytique > & cEtapeMEC::OneModeleAnalytique()
{
   return ModelesAnalytiques().Val().OneModeleAnalytique();
}

const std::list< cOneModeleAnalytique > & cEtapeMEC::OneModeleAnalytique()const 
{
   return ModelesAnalytiques().Val().OneModeleAnalytique();
}


cTplValGesInit< cModelesAnalytiques > & cEtapeMEC::ModelesAnalytiques()
{
   return mModelesAnalytiques;
}

const cTplValGesInit< cModelesAnalytiques > & cEtapeMEC::ModelesAnalytiques()const 
{
   return mModelesAnalytiques;
}


std::list< cBasculeRes > & cEtapeMEC::BasculeRes()
{
   return mBasculeRes;
}

const std::list< cBasculeRes > & cEtapeMEC::BasculeRes()const 
{
   return mBasculeRes;
}


cTplValGesInit< bool > & cEtapeMEC::UseIt()
{
   return GenerePartiesCachees().Val().UseIt();
}

const cTplValGesInit< bool > & cEtapeMEC::UseIt()const 
{
   return GenerePartiesCachees().Val().UseIt();
}


cTplValGesInit< double > & cEtapeMEC::PasDisc()
{
   return GenerePartiesCachees().Val().PasDisc();
}

const cTplValGesInit< double > & cEtapeMEC::PasDisc()const 
{
   return GenerePartiesCachees().Val().PasDisc();
}


double & cEtapeMEC::SeuilUsePC()
{
   return GenerePartiesCachees().Val().SeuilUsePC();
}

const double & cEtapeMEC::SeuilUsePC()const 
{
   return GenerePartiesCachees().Val().SeuilUsePC();
}


cTplValGesInit< std::string > & cEtapeMEC::KeyCalcPC()
{
   return GenerePartiesCachees().Val().KeyCalcPC();
}

const cTplValGesInit< std::string > & cEtapeMEC::KeyCalcPC()const 
{
   return GenerePartiesCachees().Val().KeyCalcPC();
}


cTplValGesInit< bool > & cEtapeMEC::AddChantierKPC()
{
   return GenerePartiesCachees().Val().AddChantierKPC();
}

const cTplValGesInit< bool > & cEtapeMEC::AddChantierKPC()const 
{
   return GenerePartiesCachees().Val().AddChantierKPC();
}


cTplValGesInit< bool > & cEtapeMEC::SupresExtChantierKPC()
{
   return GenerePartiesCachees().Val().SupresExtChantierKPC();
}

const cTplValGesInit< bool > & cEtapeMEC::SupresExtChantierKPC()const 
{
   return GenerePartiesCachees().Val().SupresExtChantierKPC();
}


cTplValGesInit< bool > & cEtapeMEC::Dequant()
{
   return GenerePartiesCachees().Val().Dequant();
}

const cTplValGesInit< bool > & cEtapeMEC::Dequant()const 
{
   return GenerePartiesCachees().Val().Dequant();
}


cTplValGesInit< bool > & cEtapeMEC::ByMkF()
{
   return GenerePartiesCachees().Val().ByMkF();
}

const cTplValGesInit< bool > & cEtapeMEC::ByMkF()const 
{
   return GenerePartiesCachees().Val().ByMkF();
}


cTplValGesInit< std::string > & cEtapeMEC::PatternApply()
{
   return GenerePartiesCachees().Val().PatternApply();
}

const cTplValGesInit< std::string > & cEtapeMEC::PatternApply()const 
{
   return GenerePartiesCachees().Val().PatternApply();
}


std::string & cEtapeMEC::NameFile()
{
   return GenerePartiesCachees().Val().VisuSuperposMNT().Val().NameFile();
}

const std::string & cEtapeMEC::NameFile()const 
{
   return GenerePartiesCachees().Val().VisuSuperposMNT().Val().NameFile();
}


double & cEtapeMEC::Seuil()
{
   return GenerePartiesCachees().Val().VisuSuperposMNT().Val().Seuil();
}

const double & cEtapeMEC::Seuil()const 
{
   return GenerePartiesCachees().Val().VisuSuperposMNT().Val().Seuil();
}


cTplValGesInit< cVisuSuperposMNT > & cEtapeMEC::VisuSuperposMNT()
{
   return GenerePartiesCachees().Val().VisuSuperposMNT();
}

const cTplValGesInit< cVisuSuperposMNT > & cEtapeMEC::VisuSuperposMNT()const 
{
   return GenerePartiesCachees().Val().VisuSuperposMNT();
}


cTplValGesInit< bool > & cEtapeMEC::BufXYZ()
{
   return GenerePartiesCachees().Val().BufXYZ();
}

const cTplValGesInit< bool > & cEtapeMEC::BufXYZ()const 
{
   return GenerePartiesCachees().Val().BufXYZ();
}


cTplValGesInit< bool > & cEtapeMEC::DoOnlyWhenNew()
{
   return GenerePartiesCachees().Val().DoOnlyWhenNew();
}

const cTplValGesInit< bool > & cEtapeMEC::DoOnlyWhenNew()const 
{
   return GenerePartiesCachees().Val().DoOnlyWhenNew();
}


cTplValGesInit< int > & cEtapeMEC::SzBloc()
{
   return GenerePartiesCachees().Val().SzBloc();
}

const cTplValGesInit< int > & cEtapeMEC::SzBloc()const 
{
   return GenerePartiesCachees().Val().SzBloc();
}


cTplValGesInit< int > & cEtapeMEC::SzBord()
{
   return GenerePartiesCachees().Val().SzBord();
}

const cTplValGesInit< int > & cEtapeMEC::SzBord()const 
{
   return GenerePartiesCachees().Val().SzBord();
}


cTplValGesInit< bool > & cEtapeMEC::ImSuperpMNT()
{
   return GenerePartiesCachees().Val().ImSuperpMNT();
}

const cTplValGesInit< bool > & cEtapeMEC::ImSuperpMNT()const 
{
   return GenerePartiesCachees().Val().ImSuperpMNT();
}


cTplValGesInit< double > & cEtapeMEC::ZMoy()
{
   return GenerePartiesCachees().Val().ZMoy();
}

const cTplValGesInit< double > & cEtapeMEC::ZMoy()const 
{
   return GenerePartiesCachees().Val().ZMoy();
}


cTplValGesInit< cElRegex_Ptr > & cEtapeMEC::FiltreName()
{
   return GenerePartiesCachees().Val().FiltreName();
}

const cTplValGesInit< cElRegex_Ptr > & cEtapeMEC::FiltreName()const 
{
   return GenerePartiesCachees().Val().FiltreName();
}


cTplValGesInit< std::string > & cEtapeMEC::DirOrtho()
{
   return GenerePartiesCachees().Val().MakeOrthoParImage().Val().DirOrtho();
}

const cTplValGesInit< std::string > & cEtapeMEC::DirOrtho()const 
{
   return GenerePartiesCachees().Val().MakeOrthoParImage().Val().DirOrtho();
}


cTplValGesInit< std::string > & cEtapeMEC::FileMTD()
{
   return GenerePartiesCachees().Val().MakeOrthoParImage().Val().FileMTD();
}

const cTplValGesInit< std::string > & cEtapeMEC::FileMTD()const 
{
   return GenerePartiesCachees().Val().MakeOrthoParImage().Val().FileMTD();
}


cTplValGesInit< std::string > & cEtapeMEC::NameFileSauv()
{
   return GenerePartiesCachees().Val().MakeOrthoParImage().Val().MakeMTDMaskOrtho().Val().NameFileSauv();
}

const cTplValGesInit< std::string > & cEtapeMEC::NameFileSauv()const 
{
   return GenerePartiesCachees().Val().MakeOrthoParImage().Val().MakeMTDMaskOrtho().Val().NameFileSauv();
}


cMasqMesures & cEtapeMEC::Mesures()
{
   return GenerePartiesCachees().Val().MakeOrthoParImage().Val().MakeMTDMaskOrtho().Val().Mesures();
}

const cMasqMesures & cEtapeMEC::Mesures()const 
{
   return GenerePartiesCachees().Val().MakeOrthoParImage().Val().MakeMTDMaskOrtho().Val().Mesures();
}


cTplValGesInit< cMakeMTDMaskOrtho > & cEtapeMEC::MakeMTDMaskOrtho()
{
   return GenerePartiesCachees().Val().MakeOrthoParImage().Val().MakeMTDMaskOrtho();
}

const cTplValGesInit< cMakeMTDMaskOrtho > & cEtapeMEC::MakeMTDMaskOrtho()const 
{
   return GenerePartiesCachees().Val().MakeOrthoParImage().Val().MakeMTDMaskOrtho();
}


cTplValGesInit< double > & cEtapeMEC::OrthoBiCub()
{
   return GenerePartiesCachees().Val().MakeOrthoParImage().Val().OrthoBiCub();
}

const cTplValGesInit< double > & cEtapeMEC::OrthoBiCub()const 
{
   return GenerePartiesCachees().Val().MakeOrthoParImage().Val().OrthoBiCub();
}


cTplValGesInit< double > & cEtapeMEC::ScaleBiCub()
{
   return GenerePartiesCachees().Val().MakeOrthoParImage().Val().ScaleBiCub();
}

const cTplValGesInit< double > & cEtapeMEC::ScaleBiCub()const 
{
   return GenerePartiesCachees().Val().MakeOrthoParImage().Val().ScaleBiCub();
}


cTplValGesInit< cOrthoSinusCard > & cEtapeMEC::OrthoSinusCard()
{
   return GenerePartiesCachees().Val().MakeOrthoParImage().Val().OrthoSinusCard();
}

const cTplValGesInit< cOrthoSinusCard > & cEtapeMEC::OrthoSinusCard()const 
{
   return GenerePartiesCachees().Val().MakeOrthoParImage().Val().OrthoSinusCard();
}


cTplValGesInit< double > & cEtapeMEC::ResolRelOrhto()
{
   return GenerePartiesCachees().Val().MakeOrthoParImage().Val().ResolRelOrhto();
}

const cTplValGesInit< double > & cEtapeMEC::ResolRelOrhto()const 
{
   return GenerePartiesCachees().Val().MakeOrthoParImage().Val().ResolRelOrhto();
}


cTplValGesInit< double > & cEtapeMEC::ResolAbsOrtho()
{
   return GenerePartiesCachees().Val().MakeOrthoParImage().Val().ResolAbsOrtho();
}

const cTplValGesInit< double > & cEtapeMEC::ResolAbsOrtho()const 
{
   return GenerePartiesCachees().Val().MakeOrthoParImage().Val().ResolAbsOrtho();
}


cTplValGesInit< Pt2dr > & cEtapeMEC::PixelTerrainPhase()
{
   return GenerePartiesCachees().Val().MakeOrthoParImage().Val().PixelTerrainPhase();
}

const cTplValGesInit< Pt2dr > & cEtapeMEC::PixelTerrainPhase()const 
{
   return GenerePartiesCachees().Val().MakeOrthoParImage().Val().PixelTerrainPhase();
}


std::string & cEtapeMEC::KeyCalcInput()
{
   return GenerePartiesCachees().Val().MakeOrthoParImage().Val().KeyCalcInput();
}

const std::string & cEtapeMEC::KeyCalcInput()const 
{
   return GenerePartiesCachees().Val().MakeOrthoParImage().Val().KeyCalcInput();
}


std::string & cEtapeMEC::KeyCalcOutput()
{
   return GenerePartiesCachees().Val().MakeOrthoParImage().Val().KeyCalcOutput();
}

const std::string & cEtapeMEC::KeyCalcOutput()const 
{
   return GenerePartiesCachees().Val().MakeOrthoParImage().Val().KeyCalcOutput();
}


cTplValGesInit< int > & cEtapeMEC::NbChan()
{
   return GenerePartiesCachees().Val().MakeOrthoParImage().Val().NbChan();
}

const cTplValGesInit< int > & cEtapeMEC::NbChan()const 
{
   return GenerePartiesCachees().Val().MakeOrthoParImage().Val().NbChan();
}


cTplValGesInit< std::string > & cEtapeMEC::KeyCalcIncidHor()
{
   return GenerePartiesCachees().Val().MakeOrthoParImage().Val().KeyCalcIncidHor();
}

const cTplValGesInit< std::string > & cEtapeMEC::KeyCalcIncidHor()const 
{
   return GenerePartiesCachees().Val().MakeOrthoParImage().Val().KeyCalcIncidHor();
}


cTplValGesInit< double > & cEtapeMEC::SsResolIncH()
{
   return GenerePartiesCachees().Val().MakeOrthoParImage().Val().SsResolIncH();
}

const cTplValGesInit< double > & cEtapeMEC::SsResolIncH()const 
{
   return GenerePartiesCachees().Val().MakeOrthoParImage().Val().SsResolIncH();
}


cTplValGesInit< bool > & cEtapeMEC::CalcIncAZMoy()
{
   return GenerePartiesCachees().Val().MakeOrthoParImage().Val().CalcIncAZMoy();
}

const cTplValGesInit< bool > & cEtapeMEC::CalcIncAZMoy()const 
{
   return GenerePartiesCachees().Val().MakeOrthoParImage().Val().CalcIncAZMoy();
}


cTplValGesInit< bool > & cEtapeMEC::ImageIncIsDistFront()
{
   return GenerePartiesCachees().Val().MakeOrthoParImage().Val().ImageIncIsDistFront();
}

const cTplValGesInit< bool > & cEtapeMEC::ImageIncIsDistFront()const 
{
   return GenerePartiesCachees().Val().MakeOrthoParImage().Val().ImageIncIsDistFront();
}


cTplValGesInit< int > & cEtapeMEC::RepulsFront()
{
   return GenerePartiesCachees().Val().MakeOrthoParImage().Val().RepulsFront();
}

const cTplValGesInit< int > & cEtapeMEC::RepulsFront()const 
{
   return GenerePartiesCachees().Val().MakeOrthoParImage().Val().RepulsFront();
}


cTplValGesInit< double > & cEtapeMEC::ResolIm()
{
   return GenerePartiesCachees().Val().MakeOrthoParImage().Val().ResolIm();
}

const cTplValGesInit< double > & cEtapeMEC::ResolIm()const 
{
   return GenerePartiesCachees().Val().MakeOrthoParImage().Val().ResolIm();
}


cTplValGesInit< Pt2di > & cEtapeMEC::TranslateIm()
{
   return GenerePartiesCachees().Val().MakeOrthoParImage().Val().TranslateIm();
}

const cTplValGesInit< Pt2di > & cEtapeMEC::TranslateIm()const 
{
   return GenerePartiesCachees().Val().MakeOrthoParImage().Val().TranslateIm();
}


cTplValGesInit< cMakeOrthoParImage > & cEtapeMEC::MakeOrthoParImage()
{
   return GenerePartiesCachees().Val().MakeOrthoParImage();
}

const cTplValGesInit< cMakeOrthoParImage > & cEtapeMEC::MakeOrthoParImage()const 
{
   return GenerePartiesCachees().Val().MakeOrthoParImage();
}


cTplValGesInit< cGenerePartiesCachees > & cEtapeMEC::GenerePartiesCachees()
{
   return mGenerePartiesCachees;
}

const cTplValGesInit< cGenerePartiesCachees > & cEtapeMEC::GenerePartiesCachees()const 
{
   return mGenerePartiesCachees;
}


std::string & cEtapeMEC::NameOut()
{
   return RedrLocAnam().Val().NameOut();
}

const std::string & cEtapeMEC::NameOut()const 
{
   return RedrLocAnam().Val().NameOut();
}


std::string & cEtapeMEC::NameMasq()
{
   return RedrLocAnam().Val().NameMasq();
}

const std::string & cEtapeMEC::NameMasq()const 
{
   return RedrLocAnam().Val().NameMasq();
}


std::string & cEtapeMEC::NameOriGlob()
{
   return RedrLocAnam().Val().NameOriGlob();
}

const std::string & cEtapeMEC::NameOriGlob()const 
{
   return RedrLocAnam().Val().NameOriGlob();
}


cTplValGesInit< std::string > & cEtapeMEC::NameNuage()
{
   return RedrLocAnam().Val().NameNuage();
}

const cTplValGesInit< std::string > & cEtapeMEC::NameNuage()const 
{
   return RedrLocAnam().Val().NameNuage();
}


cTplValGesInit< int > & cEtapeMEC::XRecouvrt()
{
   return RedrLocAnam().Val().XRecouvrt();
}

const cTplValGesInit< int > & cEtapeMEC::XRecouvrt()const 
{
   return RedrLocAnam().Val().XRecouvrt();
}


cTplValGesInit< double > & cEtapeMEC::MemAvalaible()
{
   return RedrLocAnam().Val().MemAvalaible();
}

const cTplValGesInit< double > & cEtapeMEC::MemAvalaible()const 
{
   return RedrLocAnam().Val().MemAvalaible();
}


cTplValGesInit< double > & cEtapeMEC::FilterMulLargY()
{
   return RedrLocAnam().Val().FilterMulLargY();
}

const cTplValGesInit< double > & cEtapeMEC::FilterMulLargY()const 
{
   return RedrLocAnam().Val().FilterMulLargY();
}


cTplValGesInit< double > & cEtapeMEC::NbIterFilterY()
{
   return RedrLocAnam().Val().NbIterFilterY();
}

const cTplValGesInit< double > & cEtapeMEC::NbIterFilterY()const 
{
   return RedrLocAnam().Val().NbIterFilterY();
}


cTplValGesInit< int > & cEtapeMEC::FilterXY()
{
   return RedrLocAnam().Val().FilterXY();
}

const cTplValGesInit< int > & cEtapeMEC::FilterXY()const 
{
   return RedrLocAnam().Val().FilterXY();
}


cTplValGesInit< int > & cEtapeMEC::NbIterXY()
{
   return RedrLocAnam().Val().NbIterXY();
}

const cTplValGesInit< int > & cEtapeMEC::NbIterXY()const 
{
   return RedrLocAnam().Val().NbIterXY();
}


cTplValGesInit< double > & cEtapeMEC::DensityHighThresh()
{
   return RedrLocAnam().Val().DensityHighThresh();
}

const cTplValGesInit< double > & cEtapeMEC::DensityHighThresh()const 
{
   return RedrLocAnam().Val().DensityHighThresh();
}


cTplValGesInit< double > & cEtapeMEC::DensityLowThresh()
{
   return RedrLocAnam().Val().DensityLowThresh();
}

const cTplValGesInit< double > & cEtapeMEC::DensityLowThresh()const 
{
   return RedrLocAnam().Val().DensityLowThresh();
}


cTplValGesInit< bool > & cEtapeMEC::UseAutoMask()
{
   return RedrLocAnam().Val().UseAutoMask();
}

const cTplValGesInit< bool > & cEtapeMEC::UseAutoMask()const 
{
   return RedrLocAnam().Val().UseAutoMask();
}


cTplValGesInit< cRedrLocAnam > & cEtapeMEC::RedrLocAnam()
{
   return mRedrLocAnam;
}

const cTplValGesInit< cRedrLocAnam > & cEtapeMEC::RedrLocAnam()const 
{
   return mRedrLocAnam;
}


cTplValGesInit< bool > & cEtapeMEC::UsePartiesCachee()
{
   return mUsePartiesCachee;
}

const cTplValGesInit< bool > & cEtapeMEC::UsePartiesCachee()const 
{
   return mUsePartiesCachee;
}


cTplValGesInit< std::string > & cEtapeMEC::NameVisuTestPC()
{
   return mNameVisuTestPC;
}

const cTplValGesInit< std::string > & cEtapeMEC::NameVisuTestPC()const 
{
   return mNameVisuTestPC;
}


std::string & cEtapeMEC::KeyAssocIm2Nuage()
{
   return NuagePredicteur().Val().KeyAssocIm2Nuage();
}

const std::string & cEtapeMEC::KeyAssocIm2Nuage()const 
{
   return NuagePredicteur().Val().KeyAssocIm2Nuage();
}


cTplValGesInit< std::string > & cEtapeMEC::Selector()
{
   return NuagePredicteur().Val().Selector();
}

const cTplValGesInit< std::string > & cEtapeMEC::Selector()const 
{
   return NuagePredicteur().Val().Selector();
}


double & cEtapeMEC::ScaleNuage()
{
   return NuagePredicteur().Val().ScaleNuage();
}

const double & cEtapeMEC::ScaleNuage()const 
{
   return NuagePredicteur().Val().ScaleNuage();
}


cTplValGesInit< cNuagePredicteur > & cEtapeMEC::NuagePredicteur()
{
   return mNuagePredicteur;
}

const cTplValGesInit< cNuagePredicteur > & cEtapeMEC::NuagePredicteur()const 
{
   return mNuagePredicteur;
}

void  BinaryUnDumpFromFile(cEtapeMEC & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.DeZoom(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.CorrelAdHoc().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.CorrelAdHoc().ValForcedForUnUmp(),aFp);
        }
        else  anObj.CorrelAdHoc().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.DoImageBSurH().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.DoImageBSurH().ValForcedForUnUmp(),aFp);
        }
        else  anObj.DoImageBSurH().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.DoStatResult().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.DoStatResult().ValForcedForUnUmp(),aFp);
        }
        else  anObj.DoStatResult().SetNoInit();
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cMasqOfEtape aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.MasqOfEtape().push_back(aVal);
        }
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SzRecouvrtDalles().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SzRecouvrtDalles().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SzRecouvrtDalles().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SzDalleMin().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SzDalleMin().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SzDalleMin().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SzDalleMax().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SzDalleMax().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SzDalleMax().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.DynamiqueCorrel().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.DynamiqueCorrel().ValForcedForUnUmp(),aFp);
        }
        else  anObj.DynamiqueCorrel().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.CorrelMin().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.CorrelMin().ValForcedForUnUmp(),aFp);
        }
        else  anObj.CorrelMin().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.GammaCorrel().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.GammaCorrel().ValForcedForUnUmp(),aFp);
        }
        else  anObj.GammaCorrel().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.AggregCorr().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.AggregCorr().ValForcedForUnUmp(),aFp);
        }
        else  anObj.AggregCorr().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SzW().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SzW().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SzW().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.WSpecUseMasqGlob().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.WSpecUseMasqGlob().ValForcedForUnUmp(),aFp);
        }
        else  anObj.WSpecUseMasqGlob().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.TypeWCorr().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.TypeWCorr().ValForcedForUnUmp(),aFp);
        }
        else  anObj.TypeWCorr().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SzWy().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SzWy().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SzWy().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NbIterFenSpec().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NbIterFenSpec().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NbIterFenSpec().SetNoInit();
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cSpecFitrageImage aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.FiltreImageLoc().push_back(aVal);
        }
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SzWInt().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SzWInt().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SzWInt().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SurEchWCor().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SurEchWCor().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SurEchWCor().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.AlgoRegul().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.AlgoRegul().ValForcedForUnUmp(),aFp);
        }
        else  anObj.AlgoRegul().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ExportZAbs().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ExportZAbs().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ExportZAbs().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.AlgoWenCxRImpossible().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.AlgoWenCxRImpossible().ValForcedForUnUmp(),aFp);
        }
        else  anObj.AlgoWenCxRImpossible().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.CoxRoy8Cnx().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.CoxRoy8Cnx().ValForcedForUnUmp(),aFp);
        }
        else  anObj.CoxRoy8Cnx().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.CoxRoyUChar().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.CoxRoyUChar().ValForcedForUnUmp(),aFp);
        }
        else  anObj.CoxRoyUChar().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ModulationProgDyn().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ModulationProgDyn().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ModulationProgDyn().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SsResolOptim().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SsResolOptim().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SsResolOptim().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.RatioDeZoomImage().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.RatioDeZoomImage().ValForcedForUnUmp(),aFp);
        }
        else  anObj.RatioDeZoomImage().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NdDiscKerInterp().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NdDiscKerInterp().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NdDiscKerInterp().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ModeInterpolation().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ModeInterpolation().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ModeInterpolation().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.CoefInterpolationBicubique().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.CoefInterpolationBicubique().ValForcedForUnUmp(),aFp);
        }
        else  anObj.CoefInterpolationBicubique().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SzSinCard().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SzSinCard().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SzSinCard().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SzAppodSinCard().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SzAppodSinCard().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SzAppodSinCard().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.TailleFenetreSinusCardinal().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.TailleFenetreSinusCardinal().ValForcedForUnUmp(),aFp);
        }
        else  anObj.TailleFenetreSinusCardinal().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ApodisationSinusCardinal().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ApodisationSinusCardinal().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ApodisationSinusCardinal().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SzGeomDerivable().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SzGeomDerivable().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SzGeomDerivable().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SeuilAttenZRegul().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SeuilAttenZRegul().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SeuilAttenZRegul().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.AttenRelatifSeuilZ().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.AttenRelatifSeuilZ().ValForcedForUnUmp(),aFp);
        }
        else  anObj.AttenRelatifSeuilZ().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ZRegul_Quad().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ZRegul_Quad().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ZRegul_Quad().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ZRegul().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ZRegul().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ZRegul().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ZPas().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ZPas().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ZPas().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.RabZDilatAltiMoins().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.RabZDilatAltiMoins().ValForcedForUnUmp(),aFp);
        }
        else  anObj.RabZDilatAltiMoins().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.RabZDilatPlaniMoins().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.RabZDilatPlaniMoins().ValForcedForUnUmp(),aFp);
        }
        else  anObj.RabZDilatPlaniMoins().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ZDilatAlti().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ZDilatAlti().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ZDilatAlti().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ZDilatPlani().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ZDilatPlani().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ZDilatPlani().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ZDilatPlaniPropPtsInt().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ZDilatPlaniPropPtsInt().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ZDilatPlaniPropPtsInt().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ZRedrPx().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ZRedrPx().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ZRedrPx().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ZDeqRedr().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ZDeqRedr().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ZDeqRedr().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.RedrNbIterMed().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.RedrNbIterMed().ValForcedForUnUmp(),aFp);
        }
        else  anObj.RedrNbIterMed().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.RedrSzMed().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.RedrSzMed().ValForcedForUnUmp(),aFp);
        }
        else  anObj.RedrSzMed().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.RedrSauvBrut().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.RedrSauvBrut().ValForcedForUnUmp(),aFp);
        }
        else  anObj.RedrSauvBrut().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.RedrNbIterMoy().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.RedrNbIterMoy().ValForcedForUnUmp(),aFp);
        }
        else  anObj.RedrNbIterMoy().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.RedrSzMoy().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.RedrSzMoy().ValForcedForUnUmp(),aFp);
        }
        else  anObj.RedrSzMoy().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Px1Regul_Quad().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Px1Regul_Quad().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Px1Regul_Quad().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Px1Regul().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Px1Regul().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Px1Regul().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Px1Pas().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Px1Pas().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Px1Pas().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Px1DilatAlti().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Px1DilatAlti().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Px1DilatAlti().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Px1DilatPlani().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Px1DilatPlani().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Px1DilatPlani().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Px1DilatPlaniPropPtsInt().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Px1DilatPlaniPropPtsInt().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Px1DilatPlaniPropPtsInt().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Px1RedrPx().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Px1RedrPx().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Px1RedrPx().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Px1DeqRedr().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Px1DeqRedr().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Px1DeqRedr().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Px2Regul_Quad().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Px2Regul_Quad().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Px2Regul_Quad().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Px2Regul().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Px2Regul().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Px2Regul().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Px2Pas().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Px2Pas().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Px2Pas().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Px2DilatAlti().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Px2DilatAlti().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Px2DilatAlti().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Px2DilatPlani().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Px2DilatPlani().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Px2DilatPlani().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Px2DilatPlaniPropPtsInt().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Px2DilatPlaniPropPtsInt().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Px2DilatPlaniPropPtsInt().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Px2RedrPx().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Px2RedrPx().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Px2RedrPx().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Px2DeqRedr().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Px2DeqRedr().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Px2DeqRedr().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.PostFiltragePx().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.PostFiltragePx().ValForcedForUnUmp(),aFp);
        }
        else  anObj.PostFiltragePx().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.PostFiltrageDiscont().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.PostFiltrageDiscont().ValForcedForUnUmp(),aFp);
        }
        else  anObj.PostFiltrageDiscont().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ImageSelecteur().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ImageSelecteur().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ImageSelecteur().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.RelSelecteur().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.RelSelecteur().ValForcedForUnUmp(),aFp);
        }
        else  anObj.RelSelecteur().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Gen8Bits_Px1().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Gen8Bits_Px1().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Gen8Bits_Px1().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Offset8Bits_Px1().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Offset8Bits_Px1().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Offset8Bits_Px1().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Dyn8Bits_Px1().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Dyn8Bits_Px1().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Dyn8Bits_Px1().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Gen8Bits_Px2().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Gen8Bits_Px2().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Gen8Bits_Px2().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Offset8Bits_Px2().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Offset8Bits_Px2().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Offset8Bits_Px2().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Dyn8Bits_Px2().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Dyn8Bits_Px2().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Dyn8Bits_Px2().SetNoInit();
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             std::string aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.ArgGen8Bits().push_back(aVal);
        }
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.GenFilePxRel().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.GenFilePxRel().ValForcedForUnUmp(),aFp);
        }
        else  anObj.GenFilePxRel().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.GenImagesCorrel().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.GenImagesCorrel().ValForcedForUnUmp(),aFp);
        }
        else  anObj.GenImagesCorrel().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.GenCubeCorrel().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.GenCubeCorrel().ValForcedForUnUmp(),aFp);
        }
        else  anObj.GenCubeCorrel().SetNoInit();
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cGenerateProjectionInImages aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.GenerateProjectionInImages().push_back(aVal);
        }
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.GenCorPxTransv().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.GenCorPxTransv().ValForcedForUnUmp(),aFp);
        }
        else  anObj.GenCorPxTransv().SetNoInit();
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cGenereModeleRaster2Analytique aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.ExportAsModeleDist().push_back(aVal);
        }
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.OptDif_PxApply().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.OptDif_PxApply().ValForcedForUnUmp(),aFp);
        }
        else  anObj.OptDif_PxApply().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.InterfaceVisualisation().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.InterfaceVisualisation().ValForcedForUnUmp(),aFp);
        }
        else  anObj.InterfaceVisualisation().SetNoInit();
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cMMExportNuage aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.MMExportNuage().push_back(aVal);
        }
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ModelesAnalytiques().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ModelesAnalytiques().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ModelesAnalytiques().SetNoInit();
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cBasculeRes aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.BasculeRes().push_back(aVal);
        }
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.GenerePartiesCachees().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.GenerePartiesCachees().ValForcedForUnUmp(),aFp);
        }
        else  anObj.GenerePartiesCachees().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.RedrLocAnam().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.RedrLocAnam().ValForcedForUnUmp(),aFp);
        }
        else  anObj.RedrLocAnam().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.UsePartiesCachee().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.UsePartiesCachee().ValForcedForUnUmp(),aFp);
        }
        else  anObj.UsePartiesCachee().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NameVisuTestPC().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NameVisuTestPC().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NameVisuTestPC().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NuagePredicteur().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NuagePredicteur().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NuagePredicteur().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cEtapeMEC & anObj)
{
    BinaryDumpInFile(aFp,anObj.DeZoom());
    BinaryDumpInFile(aFp,anObj.CorrelAdHoc().IsInit());
    if (anObj.CorrelAdHoc().IsInit()) BinaryDumpInFile(aFp,anObj.CorrelAdHoc().Val());
    BinaryDumpInFile(aFp,anObj.DoImageBSurH().IsInit());
    if (anObj.DoImageBSurH().IsInit()) BinaryDumpInFile(aFp,anObj.DoImageBSurH().Val());
    BinaryDumpInFile(aFp,anObj.DoStatResult().IsInit());
    if (anObj.DoStatResult().IsInit()) BinaryDumpInFile(aFp,anObj.DoStatResult().Val());
    BinaryDumpInFile(aFp,(int)anObj.MasqOfEtape().size());
    for(  std::list< cMasqOfEtape >::const_iterator iT=anObj.MasqOfEtape().begin();
         iT!=anObj.MasqOfEtape().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.SzRecouvrtDalles().IsInit());
    if (anObj.SzRecouvrtDalles().IsInit()) BinaryDumpInFile(aFp,anObj.SzRecouvrtDalles().Val());
    BinaryDumpInFile(aFp,anObj.SzDalleMin().IsInit());
    if (anObj.SzDalleMin().IsInit()) BinaryDumpInFile(aFp,anObj.SzDalleMin().Val());
    BinaryDumpInFile(aFp,anObj.SzDalleMax().IsInit());
    if (anObj.SzDalleMax().IsInit()) BinaryDumpInFile(aFp,anObj.SzDalleMax().Val());
    BinaryDumpInFile(aFp,anObj.DynamiqueCorrel().IsInit());
    if (anObj.DynamiqueCorrel().IsInit()) BinaryDumpInFile(aFp,anObj.DynamiqueCorrel().Val());
    BinaryDumpInFile(aFp,anObj.CorrelMin().IsInit());
    if (anObj.CorrelMin().IsInit()) BinaryDumpInFile(aFp,anObj.CorrelMin().Val());
    BinaryDumpInFile(aFp,anObj.GammaCorrel().IsInit());
    if (anObj.GammaCorrel().IsInit()) BinaryDumpInFile(aFp,anObj.GammaCorrel().Val());
    BinaryDumpInFile(aFp,anObj.AggregCorr().IsInit());
    if (anObj.AggregCorr().IsInit()) BinaryDumpInFile(aFp,anObj.AggregCorr().Val());
    BinaryDumpInFile(aFp,anObj.SzW().IsInit());
    if (anObj.SzW().IsInit()) BinaryDumpInFile(aFp,anObj.SzW().Val());
    BinaryDumpInFile(aFp,anObj.WSpecUseMasqGlob().IsInit());
    if (anObj.WSpecUseMasqGlob().IsInit()) BinaryDumpInFile(aFp,anObj.WSpecUseMasqGlob().Val());
    BinaryDumpInFile(aFp,anObj.TypeWCorr().IsInit());
    if (anObj.TypeWCorr().IsInit()) BinaryDumpInFile(aFp,anObj.TypeWCorr().Val());
    BinaryDumpInFile(aFp,anObj.SzWy().IsInit());
    if (anObj.SzWy().IsInit()) BinaryDumpInFile(aFp,anObj.SzWy().Val());
    BinaryDumpInFile(aFp,anObj.NbIterFenSpec().IsInit());
    if (anObj.NbIterFenSpec().IsInit()) BinaryDumpInFile(aFp,anObj.NbIterFenSpec().Val());
    BinaryDumpInFile(aFp,(int)anObj.FiltreImageLoc().size());
    for(  std::list< cSpecFitrageImage >::const_iterator iT=anObj.FiltreImageLoc().begin();
         iT!=anObj.FiltreImageLoc().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.SzWInt().IsInit());
    if (anObj.SzWInt().IsInit()) BinaryDumpInFile(aFp,anObj.SzWInt().Val());
    BinaryDumpInFile(aFp,anObj.SurEchWCor().IsInit());
    if (anObj.SurEchWCor().IsInit()) BinaryDumpInFile(aFp,anObj.SurEchWCor().Val());
    BinaryDumpInFile(aFp,anObj.AlgoRegul().IsInit());
    if (anObj.AlgoRegul().IsInit()) BinaryDumpInFile(aFp,anObj.AlgoRegul().Val());
    BinaryDumpInFile(aFp,anObj.ExportZAbs().IsInit());
    if (anObj.ExportZAbs().IsInit()) BinaryDumpInFile(aFp,anObj.ExportZAbs().Val());
    BinaryDumpInFile(aFp,anObj.AlgoWenCxRImpossible().IsInit());
    if (anObj.AlgoWenCxRImpossible().IsInit()) BinaryDumpInFile(aFp,anObj.AlgoWenCxRImpossible().Val());
    BinaryDumpInFile(aFp,anObj.CoxRoy8Cnx().IsInit());
    if (anObj.CoxRoy8Cnx().IsInit()) BinaryDumpInFile(aFp,anObj.CoxRoy8Cnx().Val());
    BinaryDumpInFile(aFp,anObj.CoxRoyUChar().IsInit());
    if (anObj.CoxRoyUChar().IsInit()) BinaryDumpInFile(aFp,anObj.CoxRoyUChar().Val());
    BinaryDumpInFile(aFp,anObj.ModulationProgDyn().IsInit());
    if (anObj.ModulationProgDyn().IsInit()) BinaryDumpInFile(aFp,anObj.ModulationProgDyn().Val());
    BinaryDumpInFile(aFp,anObj.SsResolOptim().IsInit());
    if (anObj.SsResolOptim().IsInit()) BinaryDumpInFile(aFp,anObj.SsResolOptim().Val());
    BinaryDumpInFile(aFp,anObj.RatioDeZoomImage().IsInit());
    if (anObj.RatioDeZoomImage().IsInit()) BinaryDumpInFile(aFp,anObj.RatioDeZoomImage().Val());
    BinaryDumpInFile(aFp,anObj.NdDiscKerInterp().IsInit());
    if (anObj.NdDiscKerInterp().IsInit()) BinaryDumpInFile(aFp,anObj.NdDiscKerInterp().Val());
    BinaryDumpInFile(aFp,anObj.ModeInterpolation().IsInit());
    if (anObj.ModeInterpolation().IsInit()) BinaryDumpInFile(aFp,anObj.ModeInterpolation().Val());
    BinaryDumpInFile(aFp,anObj.CoefInterpolationBicubique().IsInit());
    if (anObj.CoefInterpolationBicubique().IsInit()) BinaryDumpInFile(aFp,anObj.CoefInterpolationBicubique().Val());
    BinaryDumpInFile(aFp,anObj.SzSinCard().IsInit());
    if (anObj.SzSinCard().IsInit()) BinaryDumpInFile(aFp,anObj.SzSinCard().Val());
    BinaryDumpInFile(aFp,anObj.SzAppodSinCard().IsInit());
    if (anObj.SzAppodSinCard().IsInit()) BinaryDumpInFile(aFp,anObj.SzAppodSinCard().Val());
    BinaryDumpInFile(aFp,anObj.TailleFenetreSinusCardinal().IsInit());
    if (anObj.TailleFenetreSinusCardinal().IsInit()) BinaryDumpInFile(aFp,anObj.TailleFenetreSinusCardinal().Val());
    BinaryDumpInFile(aFp,anObj.ApodisationSinusCardinal().IsInit());
    if (anObj.ApodisationSinusCardinal().IsInit()) BinaryDumpInFile(aFp,anObj.ApodisationSinusCardinal().Val());
    BinaryDumpInFile(aFp,anObj.SzGeomDerivable().IsInit());
    if (anObj.SzGeomDerivable().IsInit()) BinaryDumpInFile(aFp,anObj.SzGeomDerivable().Val());
    BinaryDumpInFile(aFp,anObj.SeuilAttenZRegul().IsInit());
    if (anObj.SeuilAttenZRegul().IsInit()) BinaryDumpInFile(aFp,anObj.SeuilAttenZRegul().Val());
    BinaryDumpInFile(aFp,anObj.AttenRelatifSeuilZ().IsInit());
    if (anObj.AttenRelatifSeuilZ().IsInit()) BinaryDumpInFile(aFp,anObj.AttenRelatifSeuilZ().Val());
    BinaryDumpInFile(aFp,anObj.ZRegul_Quad().IsInit());
    if (anObj.ZRegul_Quad().IsInit()) BinaryDumpInFile(aFp,anObj.ZRegul_Quad().Val());
    BinaryDumpInFile(aFp,anObj.ZRegul().IsInit());
    if (anObj.ZRegul().IsInit()) BinaryDumpInFile(aFp,anObj.ZRegul().Val());
    BinaryDumpInFile(aFp,anObj.ZPas().IsInit());
    if (anObj.ZPas().IsInit()) BinaryDumpInFile(aFp,anObj.ZPas().Val());
    BinaryDumpInFile(aFp,anObj.RabZDilatAltiMoins().IsInit());
    if (anObj.RabZDilatAltiMoins().IsInit()) BinaryDumpInFile(aFp,anObj.RabZDilatAltiMoins().Val());
    BinaryDumpInFile(aFp,anObj.RabZDilatPlaniMoins().IsInit());
    if (anObj.RabZDilatPlaniMoins().IsInit()) BinaryDumpInFile(aFp,anObj.RabZDilatPlaniMoins().Val());
    BinaryDumpInFile(aFp,anObj.ZDilatAlti().IsInit());
    if (anObj.ZDilatAlti().IsInit()) BinaryDumpInFile(aFp,anObj.ZDilatAlti().Val());
    BinaryDumpInFile(aFp,anObj.ZDilatPlani().IsInit());
    if (anObj.ZDilatPlani().IsInit()) BinaryDumpInFile(aFp,anObj.ZDilatPlani().Val());
    BinaryDumpInFile(aFp,anObj.ZDilatPlaniPropPtsInt().IsInit());
    if (anObj.ZDilatPlaniPropPtsInt().IsInit()) BinaryDumpInFile(aFp,anObj.ZDilatPlaniPropPtsInt().Val());
    BinaryDumpInFile(aFp,anObj.ZRedrPx().IsInit());
    if (anObj.ZRedrPx().IsInit()) BinaryDumpInFile(aFp,anObj.ZRedrPx().Val());
    BinaryDumpInFile(aFp,anObj.ZDeqRedr().IsInit());
    if (anObj.ZDeqRedr().IsInit()) BinaryDumpInFile(aFp,anObj.ZDeqRedr().Val());
    BinaryDumpInFile(aFp,anObj.RedrNbIterMed().IsInit());
    if (anObj.RedrNbIterMed().IsInit()) BinaryDumpInFile(aFp,anObj.RedrNbIterMed().Val());
    BinaryDumpInFile(aFp,anObj.RedrSzMed().IsInit());
    if (anObj.RedrSzMed().IsInit()) BinaryDumpInFile(aFp,anObj.RedrSzMed().Val());
    BinaryDumpInFile(aFp,anObj.RedrSauvBrut().IsInit());
    if (anObj.RedrSauvBrut().IsInit()) BinaryDumpInFile(aFp,anObj.RedrSauvBrut().Val());
    BinaryDumpInFile(aFp,anObj.RedrNbIterMoy().IsInit());
    if (anObj.RedrNbIterMoy().IsInit()) BinaryDumpInFile(aFp,anObj.RedrNbIterMoy().Val());
    BinaryDumpInFile(aFp,anObj.RedrSzMoy().IsInit());
    if (anObj.RedrSzMoy().IsInit()) BinaryDumpInFile(aFp,anObj.RedrSzMoy().Val());
    BinaryDumpInFile(aFp,anObj.Px1Regul_Quad().IsInit());
    if (anObj.Px1Regul_Quad().IsInit()) BinaryDumpInFile(aFp,anObj.Px1Regul_Quad().Val());
    BinaryDumpInFile(aFp,anObj.Px1Regul().IsInit());
    if (anObj.Px1Regul().IsInit()) BinaryDumpInFile(aFp,anObj.Px1Regul().Val());
    BinaryDumpInFile(aFp,anObj.Px1Pas().IsInit());
    if (anObj.Px1Pas().IsInit()) BinaryDumpInFile(aFp,anObj.Px1Pas().Val());
    BinaryDumpInFile(aFp,anObj.Px1DilatAlti().IsInit());
    if (anObj.Px1DilatAlti().IsInit()) BinaryDumpInFile(aFp,anObj.Px1DilatAlti().Val());
    BinaryDumpInFile(aFp,anObj.Px1DilatPlani().IsInit());
    if (anObj.Px1DilatPlani().IsInit()) BinaryDumpInFile(aFp,anObj.Px1DilatPlani().Val());
    BinaryDumpInFile(aFp,anObj.Px1DilatPlaniPropPtsInt().IsInit());
    if (anObj.Px1DilatPlaniPropPtsInt().IsInit()) BinaryDumpInFile(aFp,anObj.Px1DilatPlaniPropPtsInt().Val());
    BinaryDumpInFile(aFp,anObj.Px1RedrPx().IsInit());
    if (anObj.Px1RedrPx().IsInit()) BinaryDumpInFile(aFp,anObj.Px1RedrPx().Val());
    BinaryDumpInFile(aFp,anObj.Px1DeqRedr().IsInit());
    if (anObj.Px1DeqRedr().IsInit()) BinaryDumpInFile(aFp,anObj.Px1DeqRedr().Val());
    BinaryDumpInFile(aFp,anObj.Px2Regul_Quad().IsInit());
    if (anObj.Px2Regul_Quad().IsInit()) BinaryDumpInFile(aFp,anObj.Px2Regul_Quad().Val());
    BinaryDumpInFile(aFp,anObj.Px2Regul().IsInit());
    if (anObj.Px2Regul().IsInit()) BinaryDumpInFile(aFp,anObj.Px2Regul().Val());
    BinaryDumpInFile(aFp,anObj.Px2Pas().IsInit());
    if (anObj.Px2Pas().IsInit()) BinaryDumpInFile(aFp,anObj.Px2Pas().Val());
    BinaryDumpInFile(aFp,anObj.Px2DilatAlti().IsInit());
    if (anObj.Px2DilatAlti().IsInit()) BinaryDumpInFile(aFp,anObj.Px2DilatAlti().Val());
    BinaryDumpInFile(aFp,anObj.Px2DilatPlani().IsInit());
    if (anObj.Px2DilatPlani().IsInit()) BinaryDumpInFile(aFp,anObj.Px2DilatPlani().Val());
    BinaryDumpInFile(aFp,anObj.Px2DilatPlaniPropPtsInt().IsInit());
    if (anObj.Px2DilatPlaniPropPtsInt().IsInit()) BinaryDumpInFile(aFp,anObj.Px2DilatPlaniPropPtsInt().Val());
    BinaryDumpInFile(aFp,anObj.Px2RedrPx().IsInit());
    if (anObj.Px2RedrPx().IsInit()) BinaryDumpInFile(aFp,anObj.Px2RedrPx().Val());
    BinaryDumpInFile(aFp,anObj.Px2DeqRedr().IsInit());
    if (anObj.Px2DeqRedr().IsInit()) BinaryDumpInFile(aFp,anObj.Px2DeqRedr().Val());
    BinaryDumpInFile(aFp,anObj.PostFiltragePx().IsInit());
    if (anObj.PostFiltragePx().IsInit()) BinaryDumpInFile(aFp,anObj.PostFiltragePx().Val());
    BinaryDumpInFile(aFp,anObj.PostFiltrageDiscont().IsInit());
    if (anObj.PostFiltrageDiscont().IsInit()) BinaryDumpInFile(aFp,anObj.PostFiltrageDiscont().Val());
    BinaryDumpInFile(aFp,anObj.ImageSelecteur().IsInit());
    if (anObj.ImageSelecteur().IsInit()) BinaryDumpInFile(aFp,anObj.ImageSelecteur().Val());
    BinaryDumpInFile(aFp,anObj.RelSelecteur().IsInit());
    if (anObj.RelSelecteur().IsInit()) BinaryDumpInFile(aFp,anObj.RelSelecteur().Val());
    BinaryDumpInFile(aFp,anObj.Gen8Bits_Px1().IsInit());
    if (anObj.Gen8Bits_Px1().IsInit()) BinaryDumpInFile(aFp,anObj.Gen8Bits_Px1().Val());
    BinaryDumpInFile(aFp,anObj.Offset8Bits_Px1().IsInit());
    if (anObj.Offset8Bits_Px1().IsInit()) BinaryDumpInFile(aFp,anObj.Offset8Bits_Px1().Val());
    BinaryDumpInFile(aFp,anObj.Dyn8Bits_Px1().IsInit());
    if (anObj.Dyn8Bits_Px1().IsInit()) BinaryDumpInFile(aFp,anObj.Dyn8Bits_Px1().Val());
    BinaryDumpInFile(aFp,anObj.Gen8Bits_Px2().IsInit());
    if (anObj.Gen8Bits_Px2().IsInit()) BinaryDumpInFile(aFp,anObj.Gen8Bits_Px2().Val());
    BinaryDumpInFile(aFp,anObj.Offset8Bits_Px2().IsInit());
    if (anObj.Offset8Bits_Px2().IsInit()) BinaryDumpInFile(aFp,anObj.Offset8Bits_Px2().Val());
    BinaryDumpInFile(aFp,anObj.Dyn8Bits_Px2().IsInit());
    if (anObj.Dyn8Bits_Px2().IsInit()) BinaryDumpInFile(aFp,anObj.Dyn8Bits_Px2().Val());
    BinaryDumpInFile(aFp,(int)anObj.ArgGen8Bits().size());
    for(  std::list< std::string >::const_iterator iT=anObj.ArgGen8Bits().begin();
         iT!=anObj.ArgGen8Bits().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.GenFilePxRel().IsInit());
    if (anObj.GenFilePxRel().IsInit()) BinaryDumpInFile(aFp,anObj.GenFilePxRel().Val());
    BinaryDumpInFile(aFp,anObj.GenImagesCorrel().IsInit());
    if (anObj.GenImagesCorrel().IsInit()) BinaryDumpInFile(aFp,anObj.GenImagesCorrel().Val());
    BinaryDumpInFile(aFp,anObj.GenCubeCorrel().IsInit());
    if (anObj.GenCubeCorrel().IsInit()) BinaryDumpInFile(aFp,anObj.GenCubeCorrel().Val());
    BinaryDumpInFile(aFp,(int)anObj.GenerateProjectionInImages().size());
    for(  std::list< cGenerateProjectionInImages >::const_iterator iT=anObj.GenerateProjectionInImages().begin();
         iT!=anObj.GenerateProjectionInImages().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.GenCorPxTransv().IsInit());
    if (anObj.GenCorPxTransv().IsInit()) BinaryDumpInFile(aFp,anObj.GenCorPxTransv().Val());
    BinaryDumpInFile(aFp,(int)anObj.ExportAsModeleDist().size());
    for(  std::list< cGenereModeleRaster2Analytique >::const_iterator iT=anObj.ExportAsModeleDist().begin();
         iT!=anObj.ExportAsModeleDist().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.OptDif_PxApply().IsInit());
    if (anObj.OptDif_PxApply().IsInit()) BinaryDumpInFile(aFp,anObj.OptDif_PxApply().Val());
    BinaryDumpInFile(aFp,anObj.InterfaceVisualisation().IsInit());
    if (anObj.InterfaceVisualisation().IsInit()) BinaryDumpInFile(aFp,anObj.InterfaceVisualisation().Val());
    BinaryDumpInFile(aFp,(int)anObj.MMExportNuage().size());
    for(  std::list< cMMExportNuage >::const_iterator iT=anObj.MMExportNuage().begin();
         iT!=anObj.MMExportNuage().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.ModelesAnalytiques().IsInit());
    if (anObj.ModelesAnalytiques().IsInit()) BinaryDumpInFile(aFp,anObj.ModelesAnalytiques().Val());
    BinaryDumpInFile(aFp,(int)anObj.BasculeRes().size());
    for(  std::list< cBasculeRes >::const_iterator iT=anObj.BasculeRes().begin();
         iT!=anObj.BasculeRes().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.GenerePartiesCachees().IsInit());
    if (anObj.GenerePartiesCachees().IsInit()) BinaryDumpInFile(aFp,anObj.GenerePartiesCachees().Val());
    BinaryDumpInFile(aFp,anObj.RedrLocAnam().IsInit());
    if (anObj.RedrLocAnam().IsInit()) BinaryDumpInFile(aFp,anObj.RedrLocAnam().Val());
    BinaryDumpInFile(aFp,anObj.UsePartiesCachee().IsInit());
    if (anObj.UsePartiesCachee().IsInit()) BinaryDumpInFile(aFp,anObj.UsePartiesCachee().Val());
    BinaryDumpInFile(aFp,anObj.NameVisuTestPC().IsInit());
    if (anObj.NameVisuTestPC().IsInit()) BinaryDumpInFile(aFp,anObj.NameVisuTestPC().Val());
    BinaryDumpInFile(aFp,anObj.NuagePredicteur().IsInit());
    if (anObj.NuagePredicteur().IsInit()) BinaryDumpInFile(aFp,anObj.NuagePredicteur().Val());
}

cElXMLTree * ToXMLTree(const cEtapeMEC & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"EtapeMEC",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("DeZoom"),anObj.DeZoom())->ReTagThis("DeZoom"));
   if (anObj.CorrelAdHoc().IsInit())
      aRes->AddFils(ToXMLTree(anObj.CorrelAdHoc().Val())->ReTagThis("CorrelAdHoc"));
   if (anObj.DoImageBSurH().IsInit())
      aRes->AddFils(ToXMLTree(anObj.DoImageBSurH().Val())->ReTagThis("DoImageBSurH"));
   if (anObj.DoStatResult().IsInit())
      aRes->AddFils(ToXMLTree(anObj.DoStatResult().Val())->ReTagThis("DoStatResult"));
  for
  (       std::list< cMasqOfEtape >::const_iterator it=anObj.MasqOfEtape().begin();
      it !=anObj.MasqOfEtape().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("MasqOfEtape"));
   if (anObj.SzRecouvrtDalles().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SzRecouvrtDalles"),anObj.SzRecouvrtDalles().Val())->ReTagThis("SzRecouvrtDalles"));
   if (anObj.SzDalleMin().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SzDalleMin"),anObj.SzDalleMin().Val())->ReTagThis("SzDalleMin"));
   if (anObj.SzDalleMax().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SzDalleMax"),anObj.SzDalleMax().Val())->ReTagThis("SzDalleMax"));
   if (anObj.DynamiqueCorrel().IsInit())
      aRes->AddFils(ToXMLTree(std::string("DynamiqueCorrel"),anObj.DynamiqueCorrel().Val())->ReTagThis("DynamiqueCorrel"));
   if (anObj.CorrelMin().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("CorrelMin"),anObj.CorrelMin().Val())->ReTagThis("CorrelMin"));
   if (anObj.GammaCorrel().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("GammaCorrel"),anObj.GammaCorrel().Val())->ReTagThis("GammaCorrel"));
   if (anObj.AggregCorr().IsInit())
      aRes->AddFils(ToXMLTree(std::string("AggregCorr"),anObj.AggregCorr().Val())->ReTagThis("AggregCorr"));
   if (anObj.SzW().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SzW"),anObj.SzW().Val())->ReTagThis("SzW"));
   if (anObj.WSpecUseMasqGlob().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("WSpecUseMasqGlob"),anObj.WSpecUseMasqGlob().Val())->ReTagThis("WSpecUseMasqGlob"));
   if (anObj.TypeWCorr().IsInit())
      aRes->AddFils(ToXMLTree(std::string("TypeWCorr"),anObj.TypeWCorr().Val())->ReTagThis("TypeWCorr"));
   if (anObj.SzWy().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SzWy"),anObj.SzWy().Val())->ReTagThis("SzWy"));
   if (anObj.NbIterFenSpec().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NbIterFenSpec"),anObj.NbIterFenSpec().Val())->ReTagThis("NbIterFenSpec"));
  for
  (       std::list< cSpecFitrageImage >::const_iterator it=anObj.FiltreImageLoc().begin();
      it !=anObj.FiltreImageLoc().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("FiltreImageLoc"));
   if (anObj.SzWInt().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SzWInt"),anObj.SzWInt().Val())->ReTagThis("SzWInt"));
   if (anObj.SurEchWCor().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SurEchWCor"),anObj.SurEchWCor().Val())->ReTagThis("SurEchWCor"));
   if (anObj.AlgoRegul().IsInit())
      aRes->AddFils(ToXMLTree(std::string("AlgoRegul"),anObj.AlgoRegul().Val())->ReTagThis("AlgoRegul"));
   if (anObj.ExportZAbs().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ExportZAbs"),anObj.ExportZAbs().Val())->ReTagThis("ExportZAbs"));
   if (anObj.AlgoWenCxRImpossible().IsInit())
      aRes->AddFils(ToXMLTree(std::string("AlgoWenCxRImpossible"),anObj.AlgoWenCxRImpossible().Val())->ReTagThis("AlgoWenCxRImpossible"));
   if (anObj.CoxRoy8Cnx().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("CoxRoy8Cnx"),anObj.CoxRoy8Cnx().Val())->ReTagThis("CoxRoy8Cnx"));
   if (anObj.CoxRoyUChar().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("CoxRoyUChar"),anObj.CoxRoyUChar().Val())->ReTagThis("CoxRoyUChar"));
   if (anObj.ModulationProgDyn().IsInit())
      aRes->AddFils(ToXMLTree(anObj.ModulationProgDyn().Val())->ReTagThis("ModulationProgDyn"));
   if (anObj.SsResolOptim().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SsResolOptim"),anObj.SsResolOptim().Val())->ReTagThis("SsResolOptim"));
   if (anObj.RatioDeZoomImage().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("RatioDeZoomImage"),anObj.RatioDeZoomImage().Val())->ReTagThis("RatioDeZoomImage"));
   if (anObj.NdDiscKerInterp().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NdDiscKerInterp"),anObj.NdDiscKerInterp().Val())->ReTagThis("NdDiscKerInterp"));
   if (anObj.ModeInterpolation().IsInit())
      aRes->AddFils(ToXMLTree(std::string("ModeInterpolation"),anObj.ModeInterpolation().Val())->ReTagThis("ModeInterpolation"));
   if (anObj.CoefInterpolationBicubique().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("CoefInterpolationBicubique"),anObj.CoefInterpolationBicubique().Val())->ReTagThis("CoefInterpolationBicubique"));
   if (anObj.SzSinCard().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SzSinCard"),anObj.SzSinCard().Val())->ReTagThis("SzSinCard"));
   if (anObj.SzAppodSinCard().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SzAppodSinCard"),anObj.SzAppodSinCard().Val())->ReTagThis("SzAppodSinCard"));
   if (anObj.TailleFenetreSinusCardinal().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("TailleFenetreSinusCardinal"),anObj.TailleFenetreSinusCardinal().Val())->ReTagThis("TailleFenetreSinusCardinal"));
   if (anObj.ApodisationSinusCardinal().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ApodisationSinusCardinal"),anObj.ApodisationSinusCardinal().Val())->ReTagThis("ApodisationSinusCardinal"));
   if (anObj.SzGeomDerivable().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SzGeomDerivable"),anObj.SzGeomDerivable().Val())->ReTagThis("SzGeomDerivable"));
   if (anObj.SeuilAttenZRegul().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SeuilAttenZRegul"),anObj.SeuilAttenZRegul().Val())->ReTagThis("SeuilAttenZRegul"));
   if (anObj.AttenRelatifSeuilZ().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("AttenRelatifSeuilZ"),anObj.AttenRelatifSeuilZ().Val())->ReTagThis("AttenRelatifSeuilZ"));
   if (anObj.ZRegul_Quad().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ZRegul_Quad"),anObj.ZRegul_Quad().Val())->ReTagThis("ZRegul_Quad"));
   if (anObj.ZRegul().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ZRegul"),anObj.ZRegul().Val())->ReTagThis("ZRegul"));
   if (anObj.ZPas().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ZPas"),anObj.ZPas().Val())->ReTagThis("ZPas"));
   if (anObj.RabZDilatAltiMoins().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("RabZDilatAltiMoins"),anObj.RabZDilatAltiMoins().Val())->ReTagThis("RabZDilatAltiMoins"));
   if (anObj.RabZDilatPlaniMoins().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("RabZDilatPlaniMoins"),anObj.RabZDilatPlaniMoins().Val())->ReTagThis("RabZDilatPlaniMoins"));
   if (anObj.ZDilatAlti().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ZDilatAlti"),anObj.ZDilatAlti().Val())->ReTagThis("ZDilatAlti"));
   if (anObj.ZDilatPlani().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ZDilatPlani"),anObj.ZDilatPlani().Val())->ReTagThis("ZDilatPlani"));
   if (anObj.ZDilatPlaniPropPtsInt().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ZDilatPlaniPropPtsInt"),anObj.ZDilatPlaniPropPtsInt().Val())->ReTagThis("ZDilatPlaniPropPtsInt"));
   if (anObj.ZRedrPx().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ZRedrPx"),anObj.ZRedrPx().Val())->ReTagThis("ZRedrPx"));
   if (anObj.ZDeqRedr().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ZDeqRedr"),anObj.ZDeqRedr().Val())->ReTagThis("ZDeqRedr"));
   if (anObj.RedrNbIterMed().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("RedrNbIterMed"),anObj.RedrNbIterMed().Val())->ReTagThis("RedrNbIterMed"));
   if (anObj.RedrSzMed().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("RedrSzMed"),anObj.RedrSzMed().Val())->ReTagThis("RedrSzMed"));
   if (anObj.RedrSauvBrut().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("RedrSauvBrut"),anObj.RedrSauvBrut().Val())->ReTagThis("RedrSauvBrut"));
   if (anObj.RedrNbIterMoy().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("RedrNbIterMoy"),anObj.RedrNbIterMoy().Val())->ReTagThis("RedrNbIterMoy"));
   if (anObj.RedrSzMoy().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("RedrSzMoy"),anObj.RedrSzMoy().Val())->ReTagThis("RedrSzMoy"));
   if (anObj.Px1Regul_Quad().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Px1Regul_Quad"),anObj.Px1Regul_Quad().Val())->ReTagThis("Px1Regul_Quad"));
   if (anObj.Px1Regul().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Px1Regul"),anObj.Px1Regul().Val())->ReTagThis("Px1Regul"));
   if (anObj.Px1Pas().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Px1Pas"),anObj.Px1Pas().Val())->ReTagThis("Px1Pas"));
   if (anObj.Px1DilatAlti().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Px1DilatAlti"),anObj.Px1DilatAlti().Val())->ReTagThis("Px1DilatAlti"));
   if (anObj.Px1DilatPlani().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Px1DilatPlani"),anObj.Px1DilatPlani().Val())->ReTagThis("Px1DilatPlani"));
   if (anObj.Px1DilatPlaniPropPtsInt().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Px1DilatPlaniPropPtsInt"),anObj.Px1DilatPlaniPropPtsInt().Val())->ReTagThis("Px1DilatPlaniPropPtsInt"));
   if (anObj.Px1RedrPx().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Px1RedrPx"),anObj.Px1RedrPx().Val())->ReTagThis("Px1RedrPx"));
   if (anObj.Px1DeqRedr().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Px1DeqRedr"),anObj.Px1DeqRedr().Val())->ReTagThis("Px1DeqRedr"));
   if (anObj.Px2Regul_Quad().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Px2Regul_Quad"),anObj.Px2Regul_Quad().Val())->ReTagThis("Px2Regul_Quad"));
   if (anObj.Px2Regul().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Px2Regul"),anObj.Px2Regul().Val())->ReTagThis("Px2Regul"));
   if (anObj.Px2Pas().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Px2Pas"),anObj.Px2Pas().Val())->ReTagThis("Px2Pas"));
   if (anObj.Px2DilatAlti().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Px2DilatAlti"),anObj.Px2DilatAlti().Val())->ReTagThis("Px2DilatAlti"));
   if (anObj.Px2DilatPlani().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Px2DilatPlani"),anObj.Px2DilatPlani().Val())->ReTagThis("Px2DilatPlani"));
   if (anObj.Px2DilatPlaniPropPtsInt().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Px2DilatPlaniPropPtsInt"),anObj.Px2DilatPlaniPropPtsInt().Val())->ReTagThis("Px2DilatPlaniPropPtsInt"));
   if (anObj.Px2RedrPx().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Px2RedrPx"),anObj.Px2RedrPx().Val())->ReTagThis("Px2RedrPx"));
   if (anObj.Px2DeqRedr().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Px2DeqRedr"),anObj.Px2DeqRedr().Val())->ReTagThis("Px2DeqRedr"));
   if (anObj.PostFiltragePx().IsInit())
      aRes->AddFils(ToXMLTree(anObj.PostFiltragePx().Val())->ReTagThis("PostFiltragePx"));
   if (anObj.PostFiltrageDiscont().IsInit())
      aRes->AddFils(ToXMLTree(anObj.PostFiltrageDiscont().Val())->ReTagThis("PostFiltrageDiscont"));
   if (anObj.ImageSelecteur().IsInit())
      aRes->AddFils(ToXMLTree(anObj.ImageSelecteur().Val())->ReTagThis("ImageSelecteur"));
   if (anObj.RelSelecteur().IsInit())
      aRes->AddFils(ToXMLTree(anObj.RelSelecteur().Val())->ReTagThis("RelSelecteur"));
   if (anObj.Gen8Bits_Px1().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Gen8Bits_Px1"),anObj.Gen8Bits_Px1().Val())->ReTagThis("Gen8Bits_Px1"));
   if (anObj.Offset8Bits_Px1().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Offset8Bits_Px1"),anObj.Offset8Bits_Px1().Val())->ReTagThis("Offset8Bits_Px1"));
   if (anObj.Dyn8Bits_Px1().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Dyn8Bits_Px1"),anObj.Dyn8Bits_Px1().Val())->ReTagThis("Dyn8Bits_Px1"));
   if (anObj.Gen8Bits_Px2().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Gen8Bits_Px2"),anObj.Gen8Bits_Px2().Val())->ReTagThis("Gen8Bits_Px2"));
   if (anObj.Offset8Bits_Px2().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Offset8Bits_Px2"),anObj.Offset8Bits_Px2().Val())->ReTagThis("Offset8Bits_Px2"));
   if (anObj.Dyn8Bits_Px2().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Dyn8Bits_Px2"),anObj.Dyn8Bits_Px2().Val())->ReTagThis("Dyn8Bits_Px2"));
  for
  (       std::list< std::string >::const_iterator it=anObj.ArgGen8Bits().begin();
      it !=anObj.ArgGen8Bits().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("ArgGen8Bits"),(*it))->ReTagThis("ArgGen8Bits"));
   if (anObj.GenFilePxRel().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("GenFilePxRel"),anObj.GenFilePxRel().Val())->ReTagThis("GenFilePxRel"));
   if (anObj.GenImagesCorrel().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("GenImagesCorrel"),anObj.GenImagesCorrel().Val())->ReTagThis("GenImagesCorrel"));
   if (anObj.GenCubeCorrel().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("GenCubeCorrel"),anObj.GenCubeCorrel().Val())->ReTagThis("GenCubeCorrel"));
  for
  (       std::list< cGenerateProjectionInImages >::const_iterator it=anObj.GenerateProjectionInImages().begin();
      it !=anObj.GenerateProjectionInImages().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("GenerateProjectionInImages"));
   if (anObj.GenCorPxTransv().IsInit())
      aRes->AddFils(ToXMLTree(anObj.GenCorPxTransv().Val())->ReTagThis("GenCorPxTransv"));
  for
  (       std::list< cGenereModeleRaster2Analytique >::const_iterator it=anObj.ExportAsModeleDist().begin();
      it !=anObj.ExportAsModeleDist().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("ExportAsModeleDist"));
   if (anObj.OptDif_PxApply().IsInit())
      aRes->AddFils(ToXMLTree(std::string("OptDif_PxApply"),anObj.OptDif_PxApply().Val())->ReTagThis("OptDif_PxApply"));
   if (anObj.InterfaceVisualisation().IsInit())
      aRes->AddFils(ToXMLTree(anObj.InterfaceVisualisation().Val())->ReTagThis("InterfaceVisualisation"));
  for
  (       std::list< cMMExportNuage >::const_iterator it=anObj.MMExportNuage().begin();
      it !=anObj.MMExportNuage().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("MMExportNuage"));
   if (anObj.ModelesAnalytiques().IsInit())
      aRes->AddFils(ToXMLTree(anObj.ModelesAnalytiques().Val())->ReTagThis("ModelesAnalytiques"));
  for
  (       std::list< cBasculeRes >::const_iterator it=anObj.BasculeRes().begin();
      it !=anObj.BasculeRes().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("BasculeRes"));
   if (anObj.GenerePartiesCachees().IsInit())
      aRes->AddFils(ToXMLTree(anObj.GenerePartiesCachees().Val())->ReTagThis("GenerePartiesCachees"));
   if (anObj.RedrLocAnam().IsInit())
      aRes->AddFils(ToXMLTree(anObj.RedrLocAnam().Val())->ReTagThis("RedrLocAnam"));
   if (anObj.UsePartiesCachee().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("UsePartiesCachee"),anObj.UsePartiesCachee().Val())->ReTagThis("UsePartiesCachee"));
   if (anObj.NameVisuTestPC().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NameVisuTestPC"),anObj.NameVisuTestPC().Val())->ReTagThis("NameVisuTestPC"));
   if (anObj.NuagePredicteur().IsInit())
      aRes->AddFils(ToXMLTree(anObj.NuagePredicteur().Val())->ReTagThis("NuagePredicteur"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cEtapeMEC & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.DeZoom(),aTree->Get("DeZoom",1)); //tototo 

   xml_init(anObj.CorrelAdHoc(),aTree->Get("CorrelAdHoc",1)); //tototo 

   xml_init(anObj.DoImageBSurH(),aTree->Get("DoImageBSurH",1)); //tototo 

   xml_init(anObj.DoStatResult(),aTree->Get("DoStatResult",1)); //tototo 

   xml_init(anObj.MasqOfEtape(),aTree->GetAll("MasqOfEtape",false,1));

   xml_init(anObj.SzRecouvrtDalles(),aTree->Get("SzRecouvrtDalles",1)); //tototo 

   xml_init(anObj.SzDalleMin(),aTree->Get("SzDalleMin",1)); //tototo 

   xml_init(anObj.SzDalleMax(),aTree->Get("SzDalleMax",1)); //tototo 

   xml_init(anObj.DynamiqueCorrel(),aTree->Get("DynamiqueCorrel",1)); //tototo 

   xml_init(anObj.CorrelMin(),aTree->Get("CorrelMin",1)); //tototo 

   xml_init(anObj.GammaCorrel(),aTree->Get("GammaCorrel",1)); //tototo 

   xml_init(anObj.AggregCorr(),aTree->Get("AggregCorr",1)); //tototo 

   xml_init(anObj.SzW(),aTree->Get("SzW",1)); //tototo 

   xml_init(anObj.WSpecUseMasqGlob(),aTree->Get("WSpecUseMasqGlob",1)); //tototo 

   xml_init(anObj.TypeWCorr(),aTree->Get("TypeWCorr",1)); //tototo 

   xml_init(anObj.SzWy(),aTree->Get("SzWy",1)); //tototo 

   xml_init(anObj.NbIterFenSpec(),aTree->Get("NbIterFenSpec",1)); //tototo 

   xml_init(anObj.FiltreImageLoc(),aTree->GetAll("FiltreImageLoc",false,1));

   xml_init(anObj.SzWInt(),aTree->Get("SzWInt",1)); //tototo 

   xml_init(anObj.SurEchWCor(),aTree->Get("SurEchWCor",1)); //tototo 

   xml_init(anObj.AlgoRegul(),aTree->Get("AlgoRegul",1)); //tototo 

   xml_init(anObj.ExportZAbs(),aTree->Get("ExportZAbs",1),bool(false)); //tototo 

   xml_init(anObj.AlgoWenCxRImpossible(),aTree->Get("AlgoWenCxRImpossible",1)); //tototo 

   xml_init(anObj.CoxRoy8Cnx(),aTree->Get("CoxRoy8Cnx",1)); //tototo 

   xml_init(anObj.CoxRoyUChar(),aTree->Get("CoxRoyUChar",1)); //tototo 

   xml_init(anObj.ModulationProgDyn(),aTree->Get("ModulationProgDyn",1)); //tototo 

   xml_init(anObj.SsResolOptim(),aTree->Get("SsResolOptim",1)); //tototo 

   xml_init(anObj.RatioDeZoomImage(),aTree->Get("RatioDeZoomImage",1)); //tototo 

   xml_init(anObj.NdDiscKerInterp(),aTree->Get("NdDiscKerInterp",1)); //tototo 

   xml_init(anObj.ModeInterpolation(),aTree->Get("ModeInterpolation",1)); //tototo 

   xml_init(anObj.CoefInterpolationBicubique(),aTree->Get("CoefInterpolationBicubique",1)); //tototo 

   xml_init(anObj.SzSinCard(),aTree->Get("SzSinCard",1)); //tototo 

   xml_init(anObj.SzAppodSinCard(),aTree->Get("SzAppodSinCard",1)); //tototo 

   xml_init(anObj.TailleFenetreSinusCardinal(),aTree->Get("TailleFenetreSinusCardinal",1),int(3)); //tototo 

   xml_init(anObj.ApodisationSinusCardinal(),aTree->Get("ApodisationSinusCardinal",1),bool(false)); //tototo 

   xml_init(anObj.SzGeomDerivable(),aTree->Get("SzGeomDerivable",1)); //tototo 

   xml_init(anObj.SeuilAttenZRegul(),aTree->Get("SeuilAttenZRegul",1)); //tototo 

   xml_init(anObj.AttenRelatifSeuilZ(),aTree->Get("AttenRelatifSeuilZ",1)); //tototo 

   xml_init(anObj.ZRegul_Quad(),aTree->Get("ZRegul_Quad",1)); //tototo 

   xml_init(anObj.ZRegul(),aTree->Get("ZRegul",1)); //tototo 

   xml_init(anObj.ZPas(),aTree->Get("ZPas",1)); //tototo 

   xml_init(anObj.RabZDilatAltiMoins(),aTree->Get("RabZDilatAltiMoins",1)); //tototo 

   xml_init(anObj.RabZDilatPlaniMoins(),aTree->Get("RabZDilatPlaniMoins",1)); //tototo 

   xml_init(anObj.ZDilatAlti(),aTree->Get("ZDilatAlti",1)); //tototo 

   xml_init(anObj.ZDilatPlani(),aTree->Get("ZDilatPlani",1)); //tototo 

   xml_init(anObj.ZDilatPlaniPropPtsInt(),aTree->Get("ZDilatPlaniPropPtsInt",1)); //tototo 

   xml_init(anObj.ZRedrPx(),aTree->Get("ZRedrPx",1)); //tototo 

   xml_init(anObj.ZDeqRedr(),aTree->Get("ZDeqRedr",1)); //tototo 

   xml_init(anObj.RedrNbIterMed(),aTree->Get("RedrNbIterMed",1)); //tototo 

   xml_init(anObj.RedrSzMed(),aTree->Get("RedrSzMed",1)); //tototo 

   xml_init(anObj.RedrSauvBrut(),aTree->Get("RedrSauvBrut",1)); //tototo 

   xml_init(anObj.RedrNbIterMoy(),aTree->Get("RedrNbIterMoy",1)); //tototo 

   xml_init(anObj.RedrSzMoy(),aTree->Get("RedrSzMoy",1)); //tototo 

   xml_init(anObj.Px1Regul_Quad(),aTree->Get("Px1Regul_Quad",1)); //tototo 

   xml_init(anObj.Px1Regul(),aTree->Get("Px1Regul",1)); //tototo 

   xml_init(anObj.Px1Pas(),aTree->Get("Px1Pas",1)); //tototo 

   xml_init(anObj.Px1DilatAlti(),aTree->Get("Px1DilatAlti",1)); //tototo 

   xml_init(anObj.Px1DilatPlani(),aTree->Get("Px1DilatPlani",1)); //tototo 

   xml_init(anObj.Px1DilatPlaniPropPtsInt(),aTree->Get("Px1DilatPlaniPropPtsInt",1)); //tototo 

   xml_init(anObj.Px1RedrPx(),aTree->Get("Px1RedrPx",1)); //tototo 

   xml_init(anObj.Px1DeqRedr(),aTree->Get("Px1DeqRedr",1)); //tototo 

   xml_init(anObj.Px2Regul_Quad(),aTree->Get("Px2Regul_Quad",1)); //tototo 

   xml_init(anObj.Px2Regul(),aTree->Get("Px2Regul",1)); //tototo 

   xml_init(anObj.Px2Pas(),aTree->Get("Px2Pas",1)); //tototo 

   xml_init(anObj.Px2DilatAlti(),aTree->Get("Px2DilatAlti",1)); //tototo 

   xml_init(anObj.Px2DilatPlani(),aTree->Get("Px2DilatPlani",1)); //tototo 

   xml_init(anObj.Px2DilatPlaniPropPtsInt(),aTree->Get("Px2DilatPlaniPropPtsInt",1)); //tototo 

   xml_init(anObj.Px2RedrPx(),aTree->Get("Px2RedrPx",1)); //tototo 

   xml_init(anObj.Px2DeqRedr(),aTree->Get("Px2DeqRedr",1)); //tototo 

   xml_init(anObj.PostFiltragePx(),aTree->Get("PostFiltragePx",1)); //tototo 

   xml_init(anObj.PostFiltrageDiscont(),aTree->Get("PostFiltrageDiscont",1)); //tototo 

   xml_init(anObj.ImageSelecteur(),aTree->Get("ImageSelecteur",1)); //tototo 

   xml_init(anObj.RelSelecteur(),aTree->Get("RelSelecteur",1)); //tototo 

   xml_init(anObj.Gen8Bits_Px1(),aTree->Get("Gen8Bits_Px1",1)); //tototo 

   xml_init(anObj.Offset8Bits_Px1(),aTree->Get("Offset8Bits_Px1",1)); //tototo 

   xml_init(anObj.Dyn8Bits_Px1(),aTree->Get("Dyn8Bits_Px1",1)); //tototo 

   xml_init(anObj.Gen8Bits_Px2(),aTree->Get("Gen8Bits_Px2",1)); //tototo 

   xml_init(anObj.Offset8Bits_Px2(),aTree->Get("Offset8Bits_Px2",1)); //tototo 

   xml_init(anObj.Dyn8Bits_Px2(),aTree->Get("Dyn8Bits_Px2",1)); //tototo 

   xml_init(anObj.ArgGen8Bits(),aTree->GetAll("ArgGen8Bits",false,1));

   xml_init(anObj.GenFilePxRel(),aTree->Get("GenFilePxRel",1)); //tototo 

   xml_init(anObj.GenImagesCorrel(),aTree->Get("GenImagesCorrel",1)); //tototo 

   xml_init(anObj.GenCubeCorrel(),aTree->Get("GenCubeCorrel",1)); //tototo 

   xml_init(anObj.GenerateProjectionInImages(),aTree->GetAll("GenerateProjectionInImages",false,1));

   xml_init(anObj.GenCorPxTransv(),aTree->Get("GenCorPxTransv",1)); //tototo 

   xml_init(anObj.ExportAsModeleDist(),aTree->GetAll("ExportAsModeleDist",false,1));

   xml_init(anObj.OptDif_PxApply(),aTree->Get("OptDif_PxApply",1)); //tototo 

   xml_init(anObj.InterfaceVisualisation(),aTree->Get("InterfaceVisualisation",1)); //tototo 

   xml_init(anObj.MMExportNuage(),aTree->GetAll("MMExportNuage",false,1));

   xml_init(anObj.ModelesAnalytiques(),aTree->Get("ModelesAnalytiques",1)); //tototo 

   xml_init(anObj.BasculeRes(),aTree->GetAll("BasculeRes",false,1));

   xml_init(anObj.GenerePartiesCachees(),aTree->Get("GenerePartiesCachees",1)); //tototo 

   xml_init(anObj.RedrLocAnam(),aTree->Get("RedrLocAnam",1)); //tototo 

   xml_init(anObj.UsePartiesCachee(),aTree->Get("UsePartiesCachee",1)); //tototo 

   xml_init(anObj.NameVisuTestPC(),aTree->Get("NameVisuTestPC",1)); //tototo 

   xml_init(anObj.NuagePredicteur(),aTree->Get("NuagePredicteur",1)); //tototo 
}

std::string  Mangling( cEtapeMEC *) {return "CA2807F9C86831B4FE3F";};


int & cTypePyramImage::Resol()
{
   return mResol;
}

const int & cTypePyramImage::Resol()const 
{
   return mResol;
}


cTplValGesInit< int > & cTypePyramImage::DivIm()
{
   return mDivIm;
}

const cTplValGesInit< int > & cTypePyramImage::DivIm()const 
{
   return mDivIm;
}


eTypeImPyram & cTypePyramImage::TypeEl()
{
   return mTypeEl;
}

const eTypeImPyram & cTypePyramImage::TypeEl()const 
{
   return mTypeEl;
}

void  BinaryUnDumpFromFile(cTypePyramImage & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Resol(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.DivIm().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.DivIm().ValForcedForUnUmp(),aFp);
        }
        else  anObj.DivIm().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.TypeEl(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cTypePyramImage & anObj)
{
    BinaryDumpInFile(aFp,anObj.Resol());
    BinaryDumpInFile(aFp,anObj.DivIm().IsInit());
    if (anObj.DivIm().IsInit()) BinaryDumpInFile(aFp,anObj.DivIm().Val());
    BinaryDumpInFile(aFp,anObj.TypeEl());
}

cElXMLTree * ToXMLTree(const cTypePyramImage & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"TypePyramImage",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Resol"),anObj.Resol())->ReTagThis("Resol"));
   if (anObj.DivIm().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DivIm"),anObj.DivIm().Val())->ReTagThis("DivIm"));
   aRes->AddFils(ToXMLTree(std::string("TypeEl"),anObj.TypeEl())->ReTagThis("TypeEl"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cTypePyramImage & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.Resol(),aTree->Get("Resol",1)); //tototo 

   xml_init(anObj.DivIm(),aTree->Get("DivIm",1),int(16)); //tototo 

   xml_init(anObj.TypeEl(),aTree->Get("TypeEl",1)); //tototo 
}

std::string  Mangling( cTypePyramImage *) {return "8E3BA7DF9AA809A2FE3F";};


cTplValGesInit< double > & cSection_MEC::ExtensionIntervZ()
{
   return mExtensionIntervZ;
}

const cTplValGesInit< double > & cSection_MEC::ExtensionIntervZ()const 
{
   return mExtensionIntervZ;
}


cTplValGesInit< bool > & cSection_MEC::PasIsInPixel()
{
   return mPasIsInPixel;
}

const cTplValGesInit< bool > & cSection_MEC::PasIsInPixel()const 
{
   return mPasIsInPixel;
}


cTplValGesInit< Box2dr > & cSection_MEC::ProportionClipMEC()
{
   return mProportionClipMEC;
}

const cTplValGesInit< Box2dr > & cSection_MEC::ProportionClipMEC()const 
{
   return mProportionClipMEC;
}


cTplValGesInit< bool > & cSection_MEC::ClipMecIsProp()
{
   return mClipMecIsProp;
}

const cTplValGesInit< bool > & cSection_MEC::ClipMecIsProp()const 
{
   return mClipMecIsProp;
}


cTplValGesInit< double > & cSection_MEC::ZoomClipMEC()
{
   return mZoomClipMEC;
}

const cTplValGesInit< double > & cSection_MEC::ZoomClipMEC()const 
{
   return mZoomClipMEC;
}


cTplValGesInit< int > & cSection_MEC::NbMinImagesVisibles()
{
   return mNbMinImagesVisibles;
}

const cTplValGesInit< int > & cSection_MEC::NbMinImagesVisibles()const 
{
   return mNbMinImagesVisibles;
}


cTplValGesInit< bool > & cSection_MEC::OneDefCorAllPxDefCor()
{
   return mOneDefCorAllPxDefCor;
}

const cTplValGesInit< bool > & cSection_MEC::OneDefCorAllPxDefCor()const 
{
   return mOneDefCorAllPxDefCor;
}


cTplValGesInit< int > & cSection_MEC::ZoomBeginODC_APDC()
{
   return mZoomBeginODC_APDC;
}

const cTplValGesInit< int > & cSection_MEC::ZoomBeginODC_APDC()const 
{
   return mZoomBeginODC_APDC;
}


cTplValGesInit< double > & cSection_MEC::DefCorrelation()
{
   return mDefCorrelation;
}

const cTplValGesInit< double > & cSection_MEC::DefCorrelation()const 
{
   return mDefCorrelation;
}


cTplValGesInit< bool > & cSection_MEC::ReprojPixelNoVal()
{
   return mReprojPixelNoVal;
}

const cTplValGesInit< bool > & cSection_MEC::ReprojPixelNoVal()const 
{
   return mReprojPixelNoVal;
}


cTplValGesInit< double > & cSection_MEC::EpsilonCorrelation()
{
   return mEpsilonCorrelation;
}

const cTplValGesInit< double > & cSection_MEC::EpsilonCorrelation()const 
{
   return mEpsilonCorrelation;
}


int & cSection_MEC::FreqEchantPtsI()
{
   return EchantillonagePtsInterets().Val().FreqEchantPtsI();
}

const int & cSection_MEC::FreqEchantPtsI()const 
{
   return EchantillonagePtsInterets().Val().FreqEchantPtsI();
}


eTypeModeEchantPtsI & cSection_MEC::ModeEchantPtsI()
{
   return EchantillonagePtsInterets().Val().ModeEchantPtsI();
}

const eTypeModeEchantPtsI & cSection_MEC::ModeEchantPtsI()const 
{
   return EchantillonagePtsInterets().Val().ModeEchantPtsI();
}


cTplValGesInit< std::string > & cSection_MEC::KeyCommandeExterneInteret()
{
   return EchantillonagePtsInterets().Val().KeyCommandeExterneInteret();
}

const cTplValGesInit< std::string > & cSection_MEC::KeyCommandeExterneInteret()const 
{
   return EchantillonagePtsInterets().Val().KeyCommandeExterneInteret();
}


cTplValGesInit< int > & cSection_MEC::SzVAutoCorrel()
{
   return EchantillonagePtsInterets().Val().SzVAutoCorrel();
}

const cTplValGesInit< int > & cSection_MEC::SzVAutoCorrel()const 
{
   return EchantillonagePtsInterets().Val().SzVAutoCorrel();
}


cTplValGesInit< double > & cSection_MEC::EstmBrAutoCorrel()
{
   return EchantillonagePtsInterets().Val().EstmBrAutoCorrel();
}

const cTplValGesInit< double > & cSection_MEC::EstmBrAutoCorrel()const 
{
   return EchantillonagePtsInterets().Val().EstmBrAutoCorrel();
}


cTplValGesInit< double > & cSection_MEC::SeuilLambdaAutoCorrel()
{
   return EchantillonagePtsInterets().Val().SeuilLambdaAutoCorrel();
}

const cTplValGesInit< double > & cSection_MEC::SeuilLambdaAutoCorrel()const 
{
   return EchantillonagePtsInterets().Val().SeuilLambdaAutoCorrel();
}


cTplValGesInit< double > & cSection_MEC::SeuilEcartTypeAutoCorrel()
{
   return EchantillonagePtsInterets().Val().SeuilEcartTypeAutoCorrel();
}

const cTplValGesInit< double > & cSection_MEC::SeuilEcartTypeAutoCorrel()const 
{
   return EchantillonagePtsInterets().Val().SeuilEcartTypeAutoCorrel();
}


cTplValGesInit< double > & cSection_MEC::RepartExclusion()
{
   return EchantillonagePtsInterets().Val().RepartExclusion();
}

const cTplValGesInit< double > & cSection_MEC::RepartExclusion()const 
{
   return EchantillonagePtsInterets().Val().RepartExclusion();
}


cTplValGesInit< double > & cSection_MEC::RepartEvitement()
{
   return EchantillonagePtsInterets().Val().RepartEvitement();
}

const cTplValGesInit< double > & cSection_MEC::RepartEvitement()const 
{
   return EchantillonagePtsInterets().Val().RepartEvitement();
}


cTplValGesInit< cEchantillonagePtsInterets > & cSection_MEC::EchantillonagePtsInterets()
{
   return mEchantillonagePtsInterets;
}

const cTplValGesInit< cEchantillonagePtsInterets > & cSection_MEC::EchantillonagePtsInterets()const 
{
   return mEchantillonagePtsInterets;
}


cTplValGesInit< bool > & cSection_MEC::ChantierFullImage1()
{
   return mChantierFullImage1;
}

const cTplValGesInit< bool > & cSection_MEC::ChantierFullImage1()const 
{
   return mChantierFullImage1;
}


cTplValGesInit< bool > & cSection_MEC::ChantierFullMaskImage1()
{
   return mChantierFullMaskImage1;
}

const cTplValGesInit< bool > & cSection_MEC::ChantierFullMaskImage1()const 
{
   return mChantierFullMaskImage1;
}


cTplValGesInit< bool > & cSection_MEC::ExportForMultiplePointsHomologues()
{
   return mExportForMultiplePointsHomologues;
}

const cTplValGesInit< bool > & cSection_MEC::ExportForMultiplePointsHomologues()const 
{
   return mExportForMultiplePointsHomologues;
}


cTplValGesInit< double > & cSection_MEC::CovLim()
{
   return AdapteDynCov().Val().CovLim();
}

const cTplValGesInit< double > & cSection_MEC::CovLim()const 
{
   return AdapteDynCov().Val().CovLim();
}


cTplValGesInit< double > & cSection_MEC::TermeDecr()
{
   return AdapteDynCov().Val().TermeDecr();
}

const cTplValGesInit< double > & cSection_MEC::TermeDecr()const 
{
   return AdapteDynCov().Val().TermeDecr();
}


cTplValGesInit< int > & cSection_MEC::SzRef()
{
   return AdapteDynCov().Val().SzRef();
}

const cTplValGesInit< int > & cSection_MEC::SzRef()const 
{
   return AdapteDynCov().Val().SzRef();
}


cTplValGesInit< double > & cSection_MEC::ValRef()
{
   return AdapteDynCov().Val().ValRef();
}

const cTplValGesInit< double > & cSection_MEC::ValRef()const 
{
   return AdapteDynCov().Val().ValRef();
}


cTplValGesInit< cAdapteDynCov > & cSection_MEC::AdapteDynCov()
{
   return mAdapteDynCov;
}

const cTplValGesInit< cAdapteDynCov > & cSection_MEC::AdapteDynCov()const 
{
   return mAdapteDynCov;
}


cTplValGesInit< cMMUseMasq3D > & cSection_MEC::MMUseMasq3D()
{
   return mMMUseMasq3D;
}

const cTplValGesInit< cMMUseMasq3D > & cSection_MEC::MMUseMasq3D()const 
{
   return mMMUseMasq3D;
}


std::list< cEtapeMEC > & cSection_MEC::EtapeMEC()
{
   return mEtapeMEC;
}

const std::list< cEtapeMEC > & cSection_MEC::EtapeMEC()const 
{
   return mEtapeMEC;
}


std::list< cTypePyramImage > & cSection_MEC::TypePyramImage()
{
   return mTypePyramImage;
}

const std::list< cTypePyramImage > & cSection_MEC::TypePyramImage()const 
{
   return mTypePyramImage;
}


cTplValGesInit< bool > & cSection_MEC::HighPrecPyrIm()
{
   return mHighPrecPyrIm;
}

const cTplValGesInit< bool > & cSection_MEC::HighPrecPyrIm()const 
{
   return mHighPrecPyrIm;
}


cTplValGesInit< bool > & cSection_MEC::Correl16Bits()
{
   return mCorrel16Bits;
}

const cTplValGesInit< bool > & cSection_MEC::Correl16Bits()const 
{
   return mCorrel16Bits;
}

void  BinaryUnDumpFromFile(cSection_MEC & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ExtensionIntervZ().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ExtensionIntervZ().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ExtensionIntervZ().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.PasIsInPixel().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.PasIsInPixel().ValForcedForUnUmp(),aFp);
        }
        else  anObj.PasIsInPixel().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ProportionClipMEC().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ProportionClipMEC().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ProportionClipMEC().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ClipMecIsProp().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ClipMecIsProp().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ClipMecIsProp().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ZoomClipMEC().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ZoomClipMEC().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ZoomClipMEC().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NbMinImagesVisibles().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NbMinImagesVisibles().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NbMinImagesVisibles().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.OneDefCorAllPxDefCor().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.OneDefCorAllPxDefCor().ValForcedForUnUmp(),aFp);
        }
        else  anObj.OneDefCorAllPxDefCor().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ZoomBeginODC_APDC().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ZoomBeginODC_APDC().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ZoomBeginODC_APDC().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.DefCorrelation().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.DefCorrelation().ValForcedForUnUmp(),aFp);
        }
        else  anObj.DefCorrelation().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ReprojPixelNoVal().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ReprojPixelNoVal().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ReprojPixelNoVal().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.EpsilonCorrelation().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.EpsilonCorrelation().ValForcedForUnUmp(),aFp);
        }
        else  anObj.EpsilonCorrelation().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.EchantillonagePtsInterets().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.EchantillonagePtsInterets().ValForcedForUnUmp(),aFp);
        }
        else  anObj.EchantillonagePtsInterets().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ChantierFullImage1().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ChantierFullImage1().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ChantierFullImage1().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ChantierFullMaskImage1().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ChantierFullMaskImage1().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ChantierFullMaskImage1().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ExportForMultiplePointsHomologues().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ExportForMultiplePointsHomologues().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ExportForMultiplePointsHomologues().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.AdapteDynCov().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.AdapteDynCov().ValForcedForUnUmp(),aFp);
        }
        else  anObj.AdapteDynCov().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.MMUseMasq3D().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.MMUseMasq3D().ValForcedForUnUmp(),aFp);
        }
        else  anObj.MMUseMasq3D().SetNoInit();
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cEtapeMEC aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.EtapeMEC().push_back(aVal);
        }
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cTypePyramImage aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.TypePyramImage().push_back(aVal);
        }
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.HighPrecPyrIm().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.HighPrecPyrIm().ValForcedForUnUmp(),aFp);
        }
        else  anObj.HighPrecPyrIm().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Correl16Bits().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Correl16Bits().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Correl16Bits().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cSection_MEC & anObj)
{
    BinaryDumpInFile(aFp,anObj.ExtensionIntervZ().IsInit());
    if (anObj.ExtensionIntervZ().IsInit()) BinaryDumpInFile(aFp,anObj.ExtensionIntervZ().Val());
    BinaryDumpInFile(aFp,anObj.PasIsInPixel().IsInit());
    if (anObj.PasIsInPixel().IsInit()) BinaryDumpInFile(aFp,anObj.PasIsInPixel().Val());
    BinaryDumpInFile(aFp,anObj.ProportionClipMEC().IsInit());
    if (anObj.ProportionClipMEC().IsInit()) BinaryDumpInFile(aFp,anObj.ProportionClipMEC().Val());
    BinaryDumpInFile(aFp,anObj.ClipMecIsProp().IsInit());
    if (anObj.ClipMecIsProp().IsInit()) BinaryDumpInFile(aFp,anObj.ClipMecIsProp().Val());
    BinaryDumpInFile(aFp,anObj.ZoomClipMEC().IsInit());
    if (anObj.ZoomClipMEC().IsInit()) BinaryDumpInFile(aFp,anObj.ZoomClipMEC().Val());
    BinaryDumpInFile(aFp,anObj.NbMinImagesVisibles().IsInit());
    if (anObj.NbMinImagesVisibles().IsInit()) BinaryDumpInFile(aFp,anObj.NbMinImagesVisibles().Val());
    BinaryDumpInFile(aFp,anObj.OneDefCorAllPxDefCor().IsInit());
    if (anObj.OneDefCorAllPxDefCor().IsInit()) BinaryDumpInFile(aFp,anObj.OneDefCorAllPxDefCor().Val());
    BinaryDumpInFile(aFp,anObj.ZoomBeginODC_APDC().IsInit());
    if (anObj.ZoomBeginODC_APDC().IsInit()) BinaryDumpInFile(aFp,anObj.ZoomBeginODC_APDC().Val());
    BinaryDumpInFile(aFp,anObj.DefCorrelation().IsInit());
    if (anObj.DefCorrelation().IsInit()) BinaryDumpInFile(aFp,anObj.DefCorrelation().Val());
    BinaryDumpInFile(aFp,anObj.ReprojPixelNoVal().IsInit());
    if (anObj.ReprojPixelNoVal().IsInit()) BinaryDumpInFile(aFp,anObj.ReprojPixelNoVal().Val());
    BinaryDumpInFile(aFp,anObj.EpsilonCorrelation().IsInit());
    if (anObj.EpsilonCorrelation().IsInit()) BinaryDumpInFile(aFp,anObj.EpsilonCorrelation().Val());
    BinaryDumpInFile(aFp,anObj.EchantillonagePtsInterets().IsInit());
    if (anObj.EchantillonagePtsInterets().IsInit()) BinaryDumpInFile(aFp,anObj.EchantillonagePtsInterets().Val());
    BinaryDumpInFile(aFp,anObj.ChantierFullImage1().IsInit());
    if (anObj.ChantierFullImage1().IsInit()) BinaryDumpInFile(aFp,anObj.ChantierFullImage1().Val());
    BinaryDumpInFile(aFp,anObj.ChantierFullMaskImage1().IsInit());
    if (anObj.ChantierFullMaskImage1().IsInit()) BinaryDumpInFile(aFp,anObj.ChantierFullMaskImage1().Val());
    BinaryDumpInFile(aFp,anObj.ExportForMultiplePointsHomologues().IsInit());
    if (anObj.ExportForMultiplePointsHomologues().IsInit()) BinaryDumpInFile(aFp,anObj.ExportForMultiplePointsHomologues().Val());
    BinaryDumpInFile(aFp,anObj.AdapteDynCov().IsInit());
    if (anObj.AdapteDynCov().IsInit()) BinaryDumpInFile(aFp,anObj.AdapteDynCov().Val());
    BinaryDumpInFile(aFp,anObj.MMUseMasq3D().IsInit());
    if (anObj.MMUseMasq3D().IsInit()) BinaryDumpInFile(aFp,anObj.MMUseMasq3D().Val());
    BinaryDumpInFile(aFp,(int)anObj.EtapeMEC().size());
    for(  std::list< cEtapeMEC >::const_iterator iT=anObj.EtapeMEC().begin();
         iT!=anObj.EtapeMEC().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,(int)anObj.TypePyramImage().size());
    for(  std::list< cTypePyramImage >::const_iterator iT=anObj.TypePyramImage().begin();
         iT!=anObj.TypePyramImage().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.HighPrecPyrIm().IsInit());
    if (anObj.HighPrecPyrIm().IsInit()) BinaryDumpInFile(aFp,anObj.HighPrecPyrIm().Val());
    BinaryDumpInFile(aFp,anObj.Correl16Bits().IsInit());
    if (anObj.Correl16Bits().IsInit()) BinaryDumpInFile(aFp,anObj.Correl16Bits().Val());
}

cElXMLTree * ToXMLTree(const cSection_MEC & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"Section_MEC",eXMLBranche);
   if (anObj.ExtensionIntervZ().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ExtensionIntervZ"),anObj.ExtensionIntervZ().Val())->ReTagThis("ExtensionIntervZ"));
   if (anObj.PasIsInPixel().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PasIsInPixel"),anObj.PasIsInPixel().Val())->ReTagThis("PasIsInPixel"));
   if (anObj.ProportionClipMEC().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ProportionClipMEC"),anObj.ProportionClipMEC().Val())->ReTagThis("ProportionClipMEC"));
   if (anObj.ClipMecIsProp().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ClipMecIsProp"),anObj.ClipMecIsProp().Val())->ReTagThis("ClipMecIsProp"));
   if (anObj.ZoomClipMEC().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ZoomClipMEC"),anObj.ZoomClipMEC().Val())->ReTagThis("ZoomClipMEC"));
   if (anObj.NbMinImagesVisibles().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NbMinImagesVisibles"),anObj.NbMinImagesVisibles().Val())->ReTagThis("NbMinImagesVisibles"));
   if (anObj.OneDefCorAllPxDefCor().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("OneDefCorAllPxDefCor"),anObj.OneDefCorAllPxDefCor().Val())->ReTagThis("OneDefCorAllPxDefCor"));
   if (anObj.ZoomBeginODC_APDC().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ZoomBeginODC_APDC"),anObj.ZoomBeginODC_APDC().Val())->ReTagThis("ZoomBeginODC_APDC"));
   if (anObj.DefCorrelation().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DefCorrelation"),anObj.DefCorrelation().Val())->ReTagThis("DefCorrelation"));
   if (anObj.ReprojPixelNoVal().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ReprojPixelNoVal"),anObj.ReprojPixelNoVal().Val())->ReTagThis("ReprojPixelNoVal"));
   if (anObj.EpsilonCorrelation().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("EpsilonCorrelation"),anObj.EpsilonCorrelation().Val())->ReTagThis("EpsilonCorrelation"));
   if (anObj.EchantillonagePtsInterets().IsInit())
      aRes->AddFils(ToXMLTree(anObj.EchantillonagePtsInterets().Val())->ReTagThis("EchantillonagePtsInterets"));
   if (anObj.ChantierFullImage1().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ChantierFullImage1"),anObj.ChantierFullImage1().Val())->ReTagThis("ChantierFullImage1"));
   if (anObj.ChantierFullMaskImage1().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ChantierFullMaskImage1"),anObj.ChantierFullMaskImage1().Val())->ReTagThis("ChantierFullMaskImage1"));
   if (anObj.ExportForMultiplePointsHomologues().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ExportForMultiplePointsHomologues"),anObj.ExportForMultiplePointsHomologues().Val())->ReTagThis("ExportForMultiplePointsHomologues"));
   if (anObj.AdapteDynCov().IsInit())
      aRes->AddFils(ToXMLTree(anObj.AdapteDynCov().Val())->ReTagThis("AdapteDynCov"));
   if (anObj.MMUseMasq3D().IsInit())
      aRes->AddFils(ToXMLTree(anObj.MMUseMasq3D().Val())->ReTagThis("MMUseMasq3D"));
  for
  (       std::list< cEtapeMEC >::const_iterator it=anObj.EtapeMEC().begin();
      it !=anObj.EtapeMEC().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("EtapeMEC"));
  for
  (       std::list< cTypePyramImage >::const_iterator it=anObj.TypePyramImage().begin();
      it !=anObj.TypePyramImage().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("TypePyramImage"));
   if (anObj.HighPrecPyrIm().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("HighPrecPyrIm"),anObj.HighPrecPyrIm().Val())->ReTagThis("HighPrecPyrIm"));
   if (anObj.Correl16Bits().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Correl16Bits"),anObj.Correl16Bits().Val())->ReTagThis("Correl16Bits"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cSection_MEC & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.ExtensionIntervZ(),aTree->Get("ExtensionIntervZ",1)); //tototo 

   xml_init(anObj.PasIsInPixel(),aTree->Get("PasIsInPixel",1),bool(false)); //tototo 

   xml_init(anObj.ProportionClipMEC(),aTree->Get("ProportionClipMEC",1)); //tototo 

   xml_init(anObj.ClipMecIsProp(),aTree->Get("ClipMecIsProp",1),bool(true)); //tototo 

   xml_init(anObj.ZoomClipMEC(),aTree->Get("ZoomClipMEC",1),double(1.0)); //tototo 

   xml_init(anObj.NbMinImagesVisibles(),aTree->Get("NbMinImagesVisibles",1),int(2)); //tototo 

   xml_init(anObj.OneDefCorAllPxDefCor(),aTree->Get("OneDefCorAllPxDefCor",1),bool(false)); //tototo 

   xml_init(anObj.ZoomBeginODC_APDC(),aTree->Get("ZoomBeginODC_APDC",1),int(4)); //tototo 

   xml_init(anObj.DefCorrelation(),aTree->Get("DefCorrelation",1),double(-0.01234)); //tototo 

   xml_init(anObj.ReprojPixelNoVal(),aTree->Get("ReprojPixelNoVal",1),bool(true)); //tototo 

   xml_init(anObj.EpsilonCorrelation(),aTree->Get("EpsilonCorrelation",1),double(1e-5)); //tototo 

   xml_init(anObj.EchantillonagePtsInterets(),aTree->Get("EchantillonagePtsInterets",1)); //tototo 

   xml_init(anObj.ChantierFullImage1(),aTree->Get("ChantierFullImage1",1),bool(false)); //tototo 

   xml_init(anObj.ChantierFullMaskImage1(),aTree->Get("ChantierFullMaskImage1",1),bool(true)); //tototo 

   xml_init(anObj.ExportForMultiplePointsHomologues(),aTree->Get("ExportForMultiplePointsHomologues",1),bool(false)); //tototo 

   xml_init(anObj.AdapteDynCov(),aTree->Get("AdapteDynCov",1)); //tototo 

   xml_init(anObj.MMUseMasq3D(),aTree->Get("MMUseMasq3D",1)); //tototo 
 
  //  CAS SPECIAL Delta Prec
  {
     std::list<cElXMLTree *> aLTr = aTree->GetAll("EtapeMEC");
     std::list<cElXMLTree *>::iterator itLTr = aLTr.begin();
     xml_init(anObj.mGlobEtapeMEC,*itLTr);
     // itLTr++;
     while (itLTr!=aLTr.end())
     {
        cEtapeMEC aVal= anObj.mGlobEtapeMEC;

        xml_init(aVal.DeZoom(),(*itLTr)->Get("DeZoom",1)); //tototo 
        if ((*itLTr)->HasFilsPorteeGlob("DeZoom"))
          anObj.mGlobEtapeMEC.DeZoom() = aVal.DeZoom();

        xml_init(aVal.CorrelAdHoc(),(*itLTr)->Get("CorrelAdHoc",1)); //tototo 
        if ((*itLTr)->HasFilsPorteeGlob("CorrelAdHoc"))
          anObj.mGlobEtapeMEC.CorrelAdHoc() = aVal.CorrelAdHoc();

        xml_init(aVal.DoImageBSurH(),(*itLTr)->Get("DoImageBSurH",1)); //tototo 
        if ((*itLTr)->HasFilsPorteeGlob("DoImageBSurH"))
          anObj.mGlobEtapeMEC.DoImageBSurH() = aVal.DoImageBSurH();

        xml_init(aVal.DoStatResult(),(*itLTr)->Get("DoStatResult",1)); //tototo 
        if ((*itLTr)->HasFilsPorteeGlob("DoStatResult"))
          anObj.mGlobEtapeMEC.DoStatResult() = aVal.DoStatResult();

        xml_init(aVal.MasqOfEtape(),(*itLTr)->GetAll("MasqOfEtape",false,1));
        if ((*itLTr)->HasFilsPorteeGlob("MasqOfEtape"))
          anObj.mGlobEtapeMEC.MasqOfEtape() = aVal.MasqOfEtape();

        xml_init(aVal.SzRecouvrtDalles(),(*itLTr)->Get("SzRecouvrtDalles",1)); //tototo 
        if ((*itLTr)->HasFilsPorteeGlob("SzRecouvrtDalles"))
          anObj.mGlobEtapeMEC.SzRecouvrtDalles() = aVal.SzRecouvrtDalles();

        xml_init(aVal.SzDalleMin(),(*itLTr)->Get("SzDalleMin",1)); //tototo 
        if ((*itLTr)->HasFilsPorteeGlob("SzDalleMin"))
          anObj.mGlobEtapeMEC.SzDalleMin() = aVal.SzDalleMin();

        xml_init(aVal.SzDalleMax(),(*itLTr)->Get("SzDalleMax",1)); //tototo 
        if ((*itLTr)->HasFilsPorteeGlob("SzDalleMax"))
          anObj.mGlobEtapeMEC.SzDalleMax() = aVal.SzDalleMax();

        xml_init(aVal.DynamiqueCorrel(),(*itLTr)->Get("DynamiqueCorrel",1)); //tototo 
        if ((*itLTr)->HasFilsPorteeGlob("DynamiqueCorrel"))
          anObj.mGlobEtapeMEC.DynamiqueCorrel() = aVal.DynamiqueCorrel();

        xml_init(aVal.CorrelMin(),(*itLTr)->Get("CorrelMin",1)); //tototo 
        if ((*itLTr)->HasFilsPorteeGlob("CorrelMin"))
          anObj.mGlobEtapeMEC.CorrelMin() = aVal.CorrelMin();

        xml_init(aVal.GammaCorrel(),(*itLTr)->Get("GammaCorrel",1)); //tototo 
        if ((*itLTr)->HasFilsPorteeGlob("GammaCorrel"))
          anObj.mGlobEtapeMEC.GammaCorrel() = aVal.GammaCorrel();

        xml_init(aVal.AggregCorr(),(*itLTr)->Get("AggregCorr",1)); //tototo 
        if ((*itLTr)->HasFilsPorteeGlob("AggregCorr"))
          anObj.mGlobEtapeMEC.AggregCorr() = aVal.AggregCorr();

        xml_init(aVal.SzW(),(*itLTr)->Get("SzW",1)); //tototo 
        if ((*itLTr)->HasFilsPorteeGlob("SzW"))
          anObj.mGlobEtapeMEC.SzW() = aVal.SzW();

        xml_init(aVal.WSpecUseMasqGlob(),(*itLTr)->Get("WSpecUseMasqGlob",1)); //tototo 
        if ((*itLTr)->HasFilsPorteeGlob("WSpecUseMasqGlob"))
          anObj.mGlobEtapeMEC.WSpecUseMasqGlob() = aVal.WSpecUseMasqGlob();

        xml_init(aVal.TypeWCorr(),(*itLTr)->Get("TypeWCorr",1)); //tototo 
        if ((*itLTr)->HasFilsPorteeGlob("TypeWCorr"))
          anObj.mGlobEtapeMEC.TypeWCorr() = aVal.TypeWCorr();

        xml_init(aVal.SzWy(),(*itLTr)->Get("SzWy",1)); //tototo 
        if ((*itLTr)->HasFilsPorteeGlob("SzWy"))
          anObj.mGlobEtapeMEC.SzWy() = aVal.SzWy();

        xml_init(aVal.NbIterFenSpec(),(*itLTr)->Get("NbIterFenSpec",1)); //tototo 
        if ((*itLTr)->HasFilsPorteeGlob("NbIterFenSpec"))
          anObj.mGlobEtapeMEC.NbIterFenSpec() = aVal.NbIterFenSpec();

        xml_init(aVal.FiltreImageLoc(),(*itLTr)->GetAll("FiltreImageLoc",false,1));
        if ((*itLTr)->HasFilsPorteeGlob("FiltreImageLoc"))
          anObj.mGlobEtapeMEC.FiltreImageLoc() = aVal.FiltreImageLoc();

        xml_init(aVal.SzWInt(),(*itLTr)->Get("SzWInt",1)); //tototo 
        if ((*itLTr)->HasFilsPorteeGlob("SzWInt"))
          anObj.mGlobEtapeMEC.SzWInt() = aVal.SzWInt();

        xml_init(aVal.SurEchWCor(),(*itLTr)->Get("SurEchWCor",1)); //tototo 
        if ((*itLTr)->HasFilsPorteeGlob("SurEchWCor"))
          anObj.mGlobEtapeMEC.SurEchWCor() = aVal.SurEchWCor();

        xml_init(aVal.AlgoRegul(),(*itLTr)->Get("AlgoRegul",1)); //tototo 
        if ((*itLTr)->HasFilsPorteeGlob("AlgoRegul"))
          anObj.mGlobEtapeMEC.AlgoRegul() = aVal.AlgoRegul();

        xml_init(aVal.ExportZAbs(),(*itLTr)->Get("ExportZAbs",1),bool(false)); //tototo 
        if ((*itLTr)->HasFilsPorteeGlob("ExportZAbs"))
          anObj.mGlobEtapeMEC.ExportZAbs() = aVal.ExportZAbs();

        xml_init(aVal.AlgoWenCxRImpossible(),(*itLTr)->Get("AlgoWenCxRImpossible",1)); //tototo 
        if ((*itLTr)->HasFilsPorteeGlob("AlgoWenCxRImpossible"))
          anObj.mGlobEtapeMEC.AlgoWenCxRImpossible() = aVal.AlgoWenCxRImpossible();

        xml_init(aVal.CoxRoy8Cnx(),(*itLTr)->Get("CoxRoy8Cnx",1)); //tototo 
        if ((*itLTr)->HasFilsPorteeGlob("CoxRoy8Cnx"))
          anObj.mGlobEtapeMEC.CoxRoy8Cnx() = aVal.CoxRoy8Cnx();

        xml_init(aVal.CoxRoyUChar(),(*itLTr)->Get("CoxRoyUChar",1)); //tototo 
        if ((*itLTr)->HasFilsPorteeGlob("CoxRoyUChar"))
          anObj.mGlobEtapeMEC.CoxRoyUChar() = aVal.CoxRoyUChar();

        xml_init(aVal.ModulationProgDyn(),(*itLTr)->Get("ModulationProgDyn",1)); //tototo 
        if ((*itLTr)->HasFilsPorteeGlob("ModulationProgDyn"))
          anObj.mGlobEtapeMEC.ModulationProgDyn() = aVal.ModulationProgDyn();

        xml_init(aVal.SsResolOptim(),(*itLTr)->Get("SsResolOptim",1)); //tototo 
        if ((*itLTr)->HasFilsPorteeGlob("SsResolOptim"))
          anObj.mGlobEtapeMEC.SsResolOptim() = aVal.SsResolOptim();

        xml_init(aVal.RatioDeZoomImage(),(*itLTr)->Get("RatioDeZoomImage",1)); //tototo 
        if ((*itLTr)->HasFilsPorteeGlob("RatioDeZoomImage"))
          anObj.mGlobEtapeMEC.RatioDeZoomImage() = aVal.RatioDeZoomImage();

        xml_init(aVal.NdDiscKerInterp(),(*itLTr)->Get("NdDiscKerInterp",1)); //tototo 
        if ((*itLTr)->HasFilsPorteeGlob("NdDiscKerInterp"))
          anObj.mGlobEtapeMEC.NdDiscKerInterp() = aVal.NdDiscKerInterp();

        xml_init(aVal.ModeInterpolation(),(*itLTr)->Get("ModeInterpolation",1)); //tototo 
        if ((*itLTr)->HasFilsPorteeGlob("ModeInterpolation"))
          anObj.mGlobEtapeMEC.ModeInterpolation() = aVal.ModeInterpolation();

        xml_init(aVal.CoefInterpolationBicubique(),(*itLTr)->Get("CoefInterpolationBicubique",1)); //tototo 
        if ((*itLTr)->HasFilsPorteeGlob("CoefInterpolationBicubique"))
          anObj.mGlobEtapeMEC.CoefInterpolationBicubique() = aVal.CoefInterpolationBicubique();

        xml_init(aVal.SzSinCard(),(*itLTr)->Get("SzSinCard",1)); //tototo 
        if ((*itLTr)->HasFilsPorteeGlob("SzSinCard"))
          anObj.mGlobEtapeMEC.SzSinCard() = aVal.SzSinCard();

        xml_init(aVal.SzAppodSinCard(),(*itLTr)->Get("SzAppodSinCard",1)); //tototo 
        if ((*itLTr)->HasFilsPorteeGlob("SzAppodSinCard"))
          anObj.mGlobEtapeMEC.SzAppodSinCard() = aVal.SzAppodSinCard();

        xml_init(aVal.TailleFenetreSinusCardinal(),(*itLTr)->Get("TailleFenetreSinusCardinal",1),int(3)); //tototo 
        if ((*itLTr)->HasFilsPorteeGlob("TailleFenetreSinusCardinal"))
          anObj.mGlobEtapeMEC.TailleFenetreSinusCardinal() = aVal.TailleFenetreSinusCardinal();

        xml_init(aVal.ApodisationSinusCardinal(),(*itLTr)->Get("ApodisationSinusCardinal",1),bool(false)); //tototo 
        if ((*itLTr)->HasFilsPorteeGlob("ApodisationSinusCardinal"))
          anObj.mGlobEtapeMEC.ApodisationSinusCardinal() = aVal.ApodisationSinusCardinal();

        xml_init(aVal.SzGeomDerivable(),(*itLTr)->Get("SzGeomDerivable",1)); //tototo 
        if ((*itLTr)->HasFilsPorteeGlob("SzGeomDerivable"))
          anObj.mGlobEtapeMEC.SzGeomDerivable() = aVal.SzGeomDerivable();

        xml_init(aVal.SeuilAttenZRegul(),(*itLTr)->Get("SeuilAttenZRegul",1)); //tototo 
        if ((*itLTr)->HasFilsPorteeGlob("SeuilAttenZRegul"))
          anObj.mGlobEtapeMEC.SeuilAttenZRegul() = aVal.SeuilAttenZRegul();

        xml_init(aVal.AttenRelatifSeuilZ(),(*itLTr)->Get("AttenRelatifSeuilZ",1)); //tototo 
        if ((*itLTr)->HasFilsPorteeGlob("AttenRelatifSeuilZ"))
          anObj.mGlobEtapeMEC.AttenRelatifSeuilZ() = aVal.AttenRelatifSeuilZ();

        xml_init(aVal.ZRegul_Quad(),(*itLTr)->Get("ZRegul_Quad",1)); //tototo 
        if ((*itLTr)->HasFilsPorteeGlob("ZRegul_Quad"))
          anObj.mGlobEtapeMEC.ZRegul_Quad() = aVal.ZRegul_Quad();

        xml_init(aVal.ZRegul(),(*itLTr)->Get("ZRegul",1)); //tototo 
        if ((*itLTr)->HasFilsPorteeGlob("ZRegul"))
          anObj.mGlobEtapeMEC.ZRegul() = aVal.ZRegul();

        xml_init(aVal.ZPas(),(*itLTr)->Get("ZPas",1)); //tototo 
        if ((*itLTr)->HasFilsPorteeGlob("ZPas"))
          anObj.mGlobEtapeMEC.ZPas() = aVal.ZPas();

        xml_init(aVal.RabZDilatAltiMoins(),(*itLTr)->Get("RabZDilatAltiMoins",1)); //tototo 
        if ((*itLTr)->HasFilsPorteeGlob("RabZDilatAltiMoins"))
          anObj.mGlobEtapeMEC.RabZDilatAltiMoins() = aVal.RabZDilatAltiMoins();

        xml_init(aVal.RabZDilatPlaniMoins(),(*itLTr)->Get("RabZDilatPlaniMoins",1)); //tototo 
        if ((*itLTr)->HasFilsPorteeGlob("RabZDilatPlaniMoins"))
          anObj.mGlobEtapeMEC.RabZDilatPlaniMoins() = aVal.RabZDilatPlaniMoins();

        xml_init(aVal.ZDilatAlti(),(*itLTr)->Get("ZDilatAlti",1)); //tototo 
        if ((*itLTr)->HasFilsPorteeGlob("ZDilatAlti"))
          anObj.mGlobEtapeMEC.ZDilatAlti() = aVal.ZDilatAlti();

        xml_init(aVal.ZDilatPlani(),(*itLTr)->Get("ZDilatPlani",1)); //tototo 
        if ((*itLTr)->HasFilsPorteeGlob("ZDilatPlani"))
          anObj.mGlobEtapeMEC.ZDilatPlani() = aVal.ZDilatPlani();

        xml_init(aVal.ZDilatPlaniPropPtsInt(),(*itLTr)->Get("ZDilatPlaniPropPtsInt",1)); //tototo 
        if ((*itLTr)->HasFilsPorteeGlob("ZDilatPlaniPropPtsInt"))
          anObj.mGlobEtapeMEC.ZDilatPlaniPropPtsInt() = aVal.ZDilatPlaniPropPtsInt();

        xml_init(aVal.ZRedrPx(),(*itLTr)->Get("ZRedrPx",1)); //tototo 
        if ((*itLTr)->HasFilsPorteeGlob("ZRedrPx"))
          anObj.mGlobEtapeMEC.ZRedrPx() = aVal.ZRedrPx();

        xml_init(aVal.ZDeqRedr(),(*itLTr)->Get("ZDeqRedr",1)); //tototo 
        if ((*itLTr)->HasFilsPorteeGlob("ZDeqRedr"))
          anObj.mGlobEtapeMEC.ZDeqRedr() = aVal.ZDeqRedr();

        xml_init(aVal.RedrNbIterMed(),(*itLTr)->Get("RedrNbIterMed",1)); //tototo 
        if ((*itLTr)->HasFilsPorteeGlob("RedrNbIterMed"))
          anObj.mGlobEtapeMEC.RedrNbIterMed() = aVal.RedrNbIterMed();

        xml_init(aVal.RedrSzMed(),(*itLTr)->Get("RedrSzMed",1)); //tototo 
        if ((*itLTr)->HasFilsPorteeGlob("RedrSzMed"))
          anObj.mGlobEtapeMEC.RedrSzMed() = aVal.RedrSzMed();

        xml_init(aVal.RedrSauvBrut(),(*itLTr)->Get("RedrSauvBrut",1)); //tototo 
        if ((*itLTr)->HasFilsPorteeGlob("RedrSauvBrut"))
          anObj.mGlobEtapeMEC.RedrSauvBrut() = aVal.RedrSauvBrut();

        xml_init(aVal.RedrNbIterMoy(),(*itLTr)->Get("RedrNbIterMoy",1)); //tototo 
        if ((*itLTr)->HasFilsPorteeGlob("RedrNbIterMoy"))
          anObj.mGlobEtapeMEC.RedrNbIterMoy() = aVal.RedrNbIterMoy();

        xml_init(aVal.RedrSzMoy(),(*itLTr)->Get("RedrSzMoy",1)); //tototo 
        if ((*itLTr)->HasFilsPorteeGlob("RedrSzMoy"))
          anObj.mGlobEtapeMEC.RedrSzMoy() = aVal.RedrSzMoy();

        xml_init(aVal.Px1Regul_Quad(),(*itLTr)->Get("Px1Regul_Quad",1)); //tototo 
        if ((*itLTr)->HasFilsPorteeGlob("Px1Regul_Quad"))
          anObj.mGlobEtapeMEC.Px1Regul_Quad() = aVal.Px1Regul_Quad();

        xml_init(aVal.Px1Regul(),(*itLTr)->Get("Px1Regul",1)); //tototo 
        if ((*itLTr)->HasFilsPorteeGlob("Px1Regul"))
          anObj.mGlobEtapeMEC.Px1Regul() = aVal.Px1Regul();

        xml_init(aVal.Px1Pas(),(*itLTr)->Get("Px1Pas",1)); //tototo 
        if ((*itLTr)->HasFilsPorteeGlob("Px1Pas"))
          anObj.mGlobEtapeMEC.Px1Pas() = aVal.Px1Pas();

        xml_init(aVal.Px1DilatAlti(),(*itLTr)->Get("Px1DilatAlti",1)); //tototo 
        if ((*itLTr)->HasFilsPorteeGlob("Px1DilatAlti"))
          anObj.mGlobEtapeMEC.Px1DilatAlti() = aVal.Px1DilatAlti();

        xml_init(aVal.Px1DilatPlani(),(*itLTr)->Get("Px1DilatPlani",1)); //tototo 
        if ((*itLTr)->HasFilsPorteeGlob("Px1DilatPlani"))
          anObj.mGlobEtapeMEC.Px1DilatPlani() = aVal.Px1DilatPlani();

        xml_init(aVal.Px1DilatPlaniPropPtsInt(),(*itLTr)->Get("Px1DilatPlaniPropPtsInt",1)); //tototo 
        if ((*itLTr)->HasFilsPorteeGlob("Px1DilatPlaniPropPtsInt"))
          anObj.mGlobEtapeMEC.Px1DilatPlaniPropPtsInt() = aVal.Px1DilatPlaniPropPtsInt();

        xml_init(aVal.Px1RedrPx(),(*itLTr)->Get("Px1RedrPx",1)); //tototo 
        if ((*itLTr)->HasFilsPorteeGlob("Px1RedrPx"))
          anObj.mGlobEtapeMEC.Px1RedrPx() = aVal.Px1RedrPx();

        xml_init(aVal.Px1DeqRedr(),(*itLTr)->Get("Px1DeqRedr",1)); //tototo 
        if ((*itLTr)->HasFilsPorteeGlob("Px1DeqRedr"))
          anObj.mGlobEtapeMEC.Px1DeqRedr() = aVal.Px1DeqRedr();

        xml_init(aVal.Px2Regul_Quad(),(*itLTr)->Get("Px2Regul_Quad",1)); //tototo 
        if ((*itLTr)->HasFilsPorteeGlob("Px2Regul_Quad"))
          anObj.mGlobEtapeMEC.Px2Regul_Quad() = aVal.Px2Regul_Quad();

        xml_init(aVal.Px2Regul(),(*itLTr)->Get("Px2Regul",1)); //tototo 
        if ((*itLTr)->HasFilsPorteeGlob("Px2Regul"))
          anObj.mGlobEtapeMEC.Px2Regul() = aVal.Px2Regul();

        xml_init(aVal.Px2Pas(),(*itLTr)->Get("Px2Pas",1)); //tototo 
        if ((*itLTr)->HasFilsPorteeGlob("Px2Pas"))
          anObj.mGlobEtapeMEC.Px2Pas() = aVal.Px2Pas();

        xml_init(aVal.Px2DilatAlti(),(*itLTr)->Get("Px2DilatAlti",1)); //tototo 
        if ((*itLTr)->HasFilsPorteeGlob("Px2DilatAlti"))
          anObj.mGlobEtapeMEC.Px2DilatAlti() = aVal.Px2DilatAlti();

        xml_init(aVal.Px2DilatPlani(),(*itLTr)->Get("Px2DilatPlani",1)); //tototo 
        if ((*itLTr)->HasFilsPorteeGlob("Px2DilatPlani"))
          anObj.mGlobEtapeMEC.Px2DilatPlani() = aVal.Px2DilatPlani();

        xml_init(aVal.Px2DilatPlaniPropPtsInt(),(*itLTr)->Get("Px2DilatPlaniPropPtsInt",1)); //tototo 
        if ((*itLTr)->HasFilsPorteeGlob("Px2DilatPlaniPropPtsInt"))
          anObj.mGlobEtapeMEC.Px2DilatPlaniPropPtsInt() = aVal.Px2DilatPlaniPropPtsInt();

        xml_init(aVal.Px2RedrPx(),(*itLTr)->Get("Px2RedrPx",1)); //tototo 
        if ((*itLTr)->HasFilsPorteeGlob("Px2RedrPx"))
          anObj.mGlobEtapeMEC.Px2RedrPx() = aVal.Px2RedrPx();

        xml_init(aVal.Px2DeqRedr(),(*itLTr)->Get("Px2DeqRedr",1)); //tototo 
        if ((*itLTr)->HasFilsPorteeGlob("Px2DeqRedr"))
          anObj.mGlobEtapeMEC.Px2DeqRedr() = aVal.Px2DeqRedr();

        xml_init(aVal.PostFiltragePx(),(*itLTr)->Get("PostFiltragePx",1)); //tototo 
        if ((*itLTr)->HasFilsPorteeGlob("PostFiltragePx"))
          anObj.mGlobEtapeMEC.PostFiltragePx() = aVal.PostFiltragePx();

        xml_init(aVal.PostFiltrageDiscont(),(*itLTr)->Get("PostFiltrageDiscont",1)); //tototo 
        if ((*itLTr)->HasFilsPorteeGlob("PostFiltrageDiscont"))
          anObj.mGlobEtapeMEC.PostFiltrageDiscont() = aVal.PostFiltrageDiscont();

        xml_init(aVal.ImageSelecteur(),(*itLTr)->Get("ImageSelecteur",1)); //tototo 
        if ((*itLTr)->HasFilsPorteeGlob("ImageSelecteur"))
          anObj.mGlobEtapeMEC.ImageSelecteur() = aVal.ImageSelecteur();

        xml_init(aVal.RelSelecteur(),(*itLTr)->Get("RelSelecteur",1)); //tototo 
        if ((*itLTr)->HasFilsPorteeGlob("RelSelecteur"))
          anObj.mGlobEtapeMEC.RelSelecteur() = aVal.RelSelecteur();

        xml_init(aVal.Gen8Bits_Px1(),(*itLTr)->Get("Gen8Bits_Px1",1)); //tototo 
        if ((*itLTr)->HasFilsPorteeGlob("Gen8Bits_Px1"))
          anObj.mGlobEtapeMEC.Gen8Bits_Px1() = aVal.Gen8Bits_Px1();

        xml_init(aVal.Offset8Bits_Px1(),(*itLTr)->Get("Offset8Bits_Px1",1)); //tototo 
        if ((*itLTr)->HasFilsPorteeGlob("Offset8Bits_Px1"))
          anObj.mGlobEtapeMEC.Offset8Bits_Px1() = aVal.Offset8Bits_Px1();

        xml_init(aVal.Dyn8Bits_Px1(),(*itLTr)->Get("Dyn8Bits_Px1",1)); //tototo 
        if ((*itLTr)->HasFilsPorteeGlob("Dyn8Bits_Px1"))
          anObj.mGlobEtapeMEC.Dyn8Bits_Px1() = aVal.Dyn8Bits_Px1();

        xml_init(aVal.Gen8Bits_Px2(),(*itLTr)->Get("Gen8Bits_Px2",1)); //tototo 
        if ((*itLTr)->HasFilsPorteeGlob("Gen8Bits_Px2"))
          anObj.mGlobEtapeMEC.Gen8Bits_Px2() = aVal.Gen8Bits_Px2();

        xml_init(aVal.Offset8Bits_Px2(),(*itLTr)->Get("Offset8Bits_Px2",1)); //tototo 
        if ((*itLTr)->HasFilsPorteeGlob("Offset8Bits_Px2"))
          anObj.mGlobEtapeMEC.Offset8Bits_Px2() = aVal.Offset8Bits_Px2();

        xml_init(aVal.Dyn8Bits_Px2(),(*itLTr)->Get("Dyn8Bits_Px2",1)); //tototo 
        if ((*itLTr)->HasFilsPorteeGlob("Dyn8Bits_Px2"))
          anObj.mGlobEtapeMEC.Dyn8Bits_Px2() = aVal.Dyn8Bits_Px2();

        xml_init(aVal.ArgGen8Bits(),(*itLTr)->GetAll("ArgGen8Bits",false,1));
        if ((*itLTr)->HasFilsPorteeGlob("ArgGen8Bits"))
          anObj.mGlobEtapeMEC.ArgGen8Bits() = aVal.ArgGen8Bits();

        xml_init(aVal.GenFilePxRel(),(*itLTr)->Get("GenFilePxRel",1)); //tototo 
        if ((*itLTr)->HasFilsPorteeGlob("GenFilePxRel"))
          anObj.mGlobEtapeMEC.GenFilePxRel() = aVal.GenFilePxRel();

        xml_init(aVal.GenImagesCorrel(),(*itLTr)->Get("GenImagesCorrel",1)); //tototo 
        if ((*itLTr)->HasFilsPorteeGlob("GenImagesCorrel"))
          anObj.mGlobEtapeMEC.GenImagesCorrel() = aVal.GenImagesCorrel();

        xml_init(aVal.GenCubeCorrel(),(*itLTr)->Get("GenCubeCorrel",1)); //tototo 
        if ((*itLTr)->HasFilsPorteeGlob("GenCubeCorrel"))
          anObj.mGlobEtapeMEC.GenCubeCorrel() = aVal.GenCubeCorrel();

        xml_init(aVal.GenerateProjectionInImages(),(*itLTr)->GetAll("GenerateProjectionInImages",false,1));
        if ((*itLTr)->HasFilsPorteeGlob("GenerateProjectionInImages"))
          anObj.mGlobEtapeMEC.GenerateProjectionInImages() = aVal.GenerateProjectionInImages();

        xml_init(aVal.GenCorPxTransv(),(*itLTr)->Get("GenCorPxTransv",1)); //tototo 
        if ((*itLTr)->HasFilsPorteeGlob("GenCorPxTransv"))
          anObj.mGlobEtapeMEC.GenCorPxTransv() = aVal.GenCorPxTransv();

        xml_init(aVal.ExportAsModeleDist(),(*itLTr)->GetAll("ExportAsModeleDist",false,1));
        if ((*itLTr)->HasFilsPorteeGlob("ExportAsModeleDist"))
          anObj.mGlobEtapeMEC.ExportAsModeleDist() = aVal.ExportAsModeleDist();

        xml_init(aVal.OptDif_PxApply(),(*itLTr)->Get("OptDif_PxApply",1)); //tototo 
        if ((*itLTr)->HasFilsPorteeGlob("OptDif_PxApply"))
          anObj.mGlobEtapeMEC.OptDif_PxApply() = aVal.OptDif_PxApply();

        xml_init(aVal.InterfaceVisualisation(),(*itLTr)->Get("InterfaceVisualisation",1)); //tototo 
        if ((*itLTr)->HasFilsPorteeGlob("InterfaceVisualisation"))
          anObj.mGlobEtapeMEC.InterfaceVisualisation() = aVal.InterfaceVisualisation();

        xml_init(aVal.MMExportNuage(),(*itLTr)->GetAll("MMExportNuage",false,1));
        if ((*itLTr)->HasFilsPorteeGlob("MMExportNuage"))
          anObj.mGlobEtapeMEC.MMExportNuage() = aVal.MMExportNuage();

        xml_init(aVal.ModelesAnalytiques(),(*itLTr)->Get("ModelesAnalytiques",1)); //tototo 
        if ((*itLTr)->HasFilsPorteeGlob("ModelesAnalytiques"))
          anObj.mGlobEtapeMEC.ModelesAnalytiques() = aVal.ModelesAnalytiques();

        xml_init(aVal.BasculeRes(),(*itLTr)->GetAll("BasculeRes",false,1));
        if ((*itLTr)->HasFilsPorteeGlob("BasculeRes"))
          anObj.mGlobEtapeMEC.BasculeRes() = aVal.BasculeRes();

        xml_init(aVal.GenerePartiesCachees(),(*itLTr)->Get("GenerePartiesCachees",1)); //tototo 
        if ((*itLTr)->HasFilsPorteeGlob("GenerePartiesCachees"))
          anObj.mGlobEtapeMEC.GenerePartiesCachees() = aVal.GenerePartiesCachees();

        xml_init(aVal.RedrLocAnam(),(*itLTr)->Get("RedrLocAnam",1)); //tototo 
        if ((*itLTr)->HasFilsPorteeGlob("RedrLocAnam"))
          anObj.mGlobEtapeMEC.RedrLocAnam() = aVal.RedrLocAnam();

        xml_init(aVal.UsePartiesCachee(),(*itLTr)->Get("UsePartiesCachee",1)); //tototo 
        if ((*itLTr)->HasFilsPorteeGlob("UsePartiesCachee"))
          anObj.mGlobEtapeMEC.UsePartiesCachee() = aVal.UsePartiesCachee();

        xml_init(aVal.NameVisuTestPC(),(*itLTr)->Get("NameVisuTestPC",1)); //tototo 
        if ((*itLTr)->HasFilsPorteeGlob("NameVisuTestPC"))
          anObj.mGlobEtapeMEC.NameVisuTestPC() = aVal.NameVisuTestPC();

        xml_init(aVal.NuagePredicteur(),(*itLTr)->Get("NuagePredicteur",1)); //tototo 
        if ((*itLTr)->HasFilsPorteeGlob("NuagePredicteur"))
          anObj.mGlobEtapeMEC.NuagePredicteur() = aVal.NuagePredicteur();

        anObj.mEtapeMEC.push_back(aVal);
        itLTr++;
     }
  }

   xml_init(anObj.TypePyramImage(),aTree->GetAll("TypePyramImage",false,1));

   xml_init(anObj.HighPrecPyrIm(),aTree->Get("HighPrecPyrIm",1),bool(true)); //tototo 

   xml_init(anObj.Correl16Bits(),aTree->Get("Correl16Bits",1)); //tototo 
}

std::string  Mangling( cSection_MEC *) {return "2C67CF6DB6CB9184FCBF";};


cTplValGesInit< bool > & cDoNothingBut::ButDoPyram()
{
   return mButDoPyram;
}

const cTplValGesInit< bool > & cDoNothingBut::ButDoPyram()const 
{
   return mButDoPyram;
}


cTplValGesInit< bool > & cDoNothingBut::ButDoMasqIm()
{
   return mButDoMasqIm;
}

const cTplValGesInit< bool > & cDoNothingBut::ButDoMasqIm()const 
{
   return mButDoMasqIm;
}


cTplValGesInit< bool > & cDoNothingBut::ButDoMemPart()
{
   return mButDoMemPart;
}

const cTplValGesInit< bool > & cDoNothingBut::ButDoMemPart()const 
{
   return mButDoMemPart;
}


cTplValGesInit< bool > & cDoNothingBut::ButDoTA()
{
   return mButDoTA;
}

const cTplValGesInit< bool > & cDoNothingBut::ButDoTA()const 
{
   return mButDoTA;
}


cTplValGesInit< bool > & cDoNothingBut::ButDoMasqueChantier()
{
   return mButDoMasqueChantier;
}

const cTplValGesInit< bool > & cDoNothingBut::ButDoMasqueChantier()const 
{
   return mButDoMasqueChantier;
}


cTplValGesInit< bool > & cDoNothingBut::ButDoOriMNT()
{
   return mButDoOriMNT;
}

const cTplValGesInit< bool > & cDoNothingBut::ButDoOriMNT()const 
{
   return mButDoOriMNT;
}


cTplValGesInit< bool > & cDoNothingBut::ButDoMTDNuage()
{
   return mButDoMTDNuage;
}

const cTplValGesInit< bool > & cDoNothingBut::ButDoMTDNuage()const 
{
   return mButDoMTDNuage;
}


cTplValGesInit< bool > & cDoNothingBut::ButDoFDC()
{
   return mButDoFDC;
}

const cTplValGesInit< bool > & cDoNothingBut::ButDoFDC()const 
{
   return mButDoFDC;
}


cTplValGesInit< bool > & cDoNothingBut::ButDoExtendParam()
{
   return mButDoExtendParam;
}

const cTplValGesInit< bool > & cDoNothingBut::ButDoExtendParam()const 
{
   return mButDoExtendParam;
}


cTplValGesInit< bool > & cDoNothingBut::ButDoGenCorPxTransv()
{
   return mButDoGenCorPxTransv;
}

const cTplValGesInit< bool > & cDoNothingBut::ButDoGenCorPxTransv()const 
{
   return mButDoGenCorPxTransv;
}


cTplValGesInit< bool > & cDoNothingBut::ButDoPartiesCachees()
{
   return mButDoPartiesCachees;
}

const cTplValGesInit< bool > & cDoNothingBut::ButDoPartiesCachees()const 
{
   return mButDoPartiesCachees;
}


cTplValGesInit< bool > & cDoNothingBut::ButDoOrtho()
{
   return mButDoOrtho;
}

const cTplValGesInit< bool > & cDoNothingBut::ButDoOrtho()const 
{
   return mButDoOrtho;
}


cTplValGesInit< bool > & cDoNothingBut::ButDoSimul()
{
   return mButDoSimul;
}

const cTplValGesInit< bool > & cDoNothingBut::ButDoSimul()const 
{
   return mButDoSimul;
}


cTplValGesInit< bool > & cDoNothingBut::ButDoRedrLocAnam()
{
   return mButDoRedrLocAnam;
}

const cTplValGesInit< bool > & cDoNothingBut::ButDoRedrLocAnam()const 
{
   return mButDoRedrLocAnam;
}

void  BinaryUnDumpFromFile(cDoNothingBut & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ButDoPyram().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ButDoPyram().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ButDoPyram().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ButDoMasqIm().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ButDoMasqIm().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ButDoMasqIm().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ButDoMemPart().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ButDoMemPart().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ButDoMemPart().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ButDoTA().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ButDoTA().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ButDoTA().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ButDoMasqueChantier().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ButDoMasqueChantier().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ButDoMasqueChantier().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ButDoOriMNT().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ButDoOriMNT().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ButDoOriMNT().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ButDoMTDNuage().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ButDoMTDNuage().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ButDoMTDNuage().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ButDoFDC().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ButDoFDC().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ButDoFDC().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ButDoExtendParam().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ButDoExtendParam().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ButDoExtendParam().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ButDoGenCorPxTransv().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ButDoGenCorPxTransv().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ButDoGenCorPxTransv().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ButDoPartiesCachees().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ButDoPartiesCachees().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ButDoPartiesCachees().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ButDoOrtho().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ButDoOrtho().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ButDoOrtho().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ButDoSimul().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ButDoSimul().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ButDoSimul().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ButDoRedrLocAnam().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ButDoRedrLocAnam().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ButDoRedrLocAnam().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cDoNothingBut & anObj)
{
    BinaryDumpInFile(aFp,anObj.ButDoPyram().IsInit());
    if (anObj.ButDoPyram().IsInit()) BinaryDumpInFile(aFp,anObj.ButDoPyram().Val());
    BinaryDumpInFile(aFp,anObj.ButDoMasqIm().IsInit());
    if (anObj.ButDoMasqIm().IsInit()) BinaryDumpInFile(aFp,anObj.ButDoMasqIm().Val());
    BinaryDumpInFile(aFp,anObj.ButDoMemPart().IsInit());
    if (anObj.ButDoMemPart().IsInit()) BinaryDumpInFile(aFp,anObj.ButDoMemPart().Val());
    BinaryDumpInFile(aFp,anObj.ButDoTA().IsInit());
    if (anObj.ButDoTA().IsInit()) BinaryDumpInFile(aFp,anObj.ButDoTA().Val());
    BinaryDumpInFile(aFp,anObj.ButDoMasqueChantier().IsInit());
    if (anObj.ButDoMasqueChantier().IsInit()) BinaryDumpInFile(aFp,anObj.ButDoMasqueChantier().Val());
    BinaryDumpInFile(aFp,anObj.ButDoOriMNT().IsInit());
    if (anObj.ButDoOriMNT().IsInit()) BinaryDumpInFile(aFp,anObj.ButDoOriMNT().Val());
    BinaryDumpInFile(aFp,anObj.ButDoMTDNuage().IsInit());
    if (anObj.ButDoMTDNuage().IsInit()) BinaryDumpInFile(aFp,anObj.ButDoMTDNuage().Val());
    BinaryDumpInFile(aFp,anObj.ButDoFDC().IsInit());
    if (anObj.ButDoFDC().IsInit()) BinaryDumpInFile(aFp,anObj.ButDoFDC().Val());
    BinaryDumpInFile(aFp,anObj.ButDoExtendParam().IsInit());
    if (anObj.ButDoExtendParam().IsInit()) BinaryDumpInFile(aFp,anObj.ButDoExtendParam().Val());
    BinaryDumpInFile(aFp,anObj.ButDoGenCorPxTransv().IsInit());
    if (anObj.ButDoGenCorPxTransv().IsInit()) BinaryDumpInFile(aFp,anObj.ButDoGenCorPxTransv().Val());
    BinaryDumpInFile(aFp,anObj.ButDoPartiesCachees().IsInit());
    if (anObj.ButDoPartiesCachees().IsInit()) BinaryDumpInFile(aFp,anObj.ButDoPartiesCachees().Val());
    BinaryDumpInFile(aFp,anObj.ButDoOrtho().IsInit());
    if (anObj.ButDoOrtho().IsInit()) BinaryDumpInFile(aFp,anObj.ButDoOrtho().Val());
    BinaryDumpInFile(aFp,anObj.ButDoSimul().IsInit());
    if (anObj.ButDoSimul().IsInit()) BinaryDumpInFile(aFp,anObj.ButDoSimul().Val());
    BinaryDumpInFile(aFp,anObj.ButDoRedrLocAnam().IsInit());
    if (anObj.ButDoRedrLocAnam().IsInit()) BinaryDumpInFile(aFp,anObj.ButDoRedrLocAnam().Val());
}

cElXMLTree * ToXMLTree(const cDoNothingBut & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"DoNothingBut",eXMLBranche);
   if (anObj.ButDoPyram().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ButDoPyram"),anObj.ButDoPyram().Val())->ReTagThis("ButDoPyram"));
   if (anObj.ButDoMasqIm().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ButDoMasqIm"),anObj.ButDoMasqIm().Val())->ReTagThis("ButDoMasqIm"));
   if (anObj.ButDoMemPart().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ButDoMemPart"),anObj.ButDoMemPart().Val())->ReTagThis("ButDoMemPart"));
   if (anObj.ButDoTA().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ButDoTA"),anObj.ButDoTA().Val())->ReTagThis("ButDoTA"));
   if (anObj.ButDoMasqueChantier().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ButDoMasqueChantier"),anObj.ButDoMasqueChantier().Val())->ReTagThis("ButDoMasqueChantier"));
   if (anObj.ButDoOriMNT().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ButDoOriMNT"),anObj.ButDoOriMNT().Val())->ReTagThis("ButDoOriMNT"));
   if (anObj.ButDoMTDNuage().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ButDoMTDNuage"),anObj.ButDoMTDNuage().Val())->ReTagThis("ButDoMTDNuage"));
   if (anObj.ButDoFDC().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ButDoFDC"),anObj.ButDoFDC().Val())->ReTagThis("ButDoFDC"));
   if (anObj.ButDoExtendParam().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ButDoExtendParam"),anObj.ButDoExtendParam().Val())->ReTagThis("ButDoExtendParam"));
   if (anObj.ButDoGenCorPxTransv().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ButDoGenCorPxTransv"),anObj.ButDoGenCorPxTransv().Val())->ReTagThis("ButDoGenCorPxTransv"));
   if (anObj.ButDoPartiesCachees().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ButDoPartiesCachees"),anObj.ButDoPartiesCachees().Val())->ReTagThis("ButDoPartiesCachees"));
   if (anObj.ButDoOrtho().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ButDoOrtho"),anObj.ButDoOrtho().Val())->ReTagThis("ButDoOrtho"));
   if (anObj.ButDoSimul().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ButDoSimul"),anObj.ButDoSimul().Val())->ReTagThis("ButDoSimul"));
   if (anObj.ButDoRedrLocAnam().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ButDoRedrLocAnam"),anObj.ButDoRedrLocAnam().Val())->ReTagThis("ButDoRedrLocAnam"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cDoNothingBut & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.ButDoPyram(),aTree->Get("ButDoPyram",1),bool(false)); //tototo 

   xml_init(anObj.ButDoMasqIm(),aTree->Get("ButDoMasqIm",1),bool(false)); //tototo 

   xml_init(anObj.ButDoMemPart(),aTree->Get("ButDoMemPart",1),bool(false)); //tototo 

   xml_init(anObj.ButDoTA(),aTree->Get("ButDoTA",1),bool(false)); //tototo 

   xml_init(anObj.ButDoMasqueChantier(),aTree->Get("ButDoMasqueChantier",1),bool(false)); //tototo 

   xml_init(anObj.ButDoOriMNT(),aTree->Get("ButDoOriMNT",1),bool(false)); //tototo 

   xml_init(anObj.ButDoMTDNuage(),aTree->Get("ButDoMTDNuage",1),bool(false)); //tototo 

   xml_init(anObj.ButDoFDC(),aTree->Get("ButDoFDC",1),bool(false)); //tototo 

   xml_init(anObj.ButDoExtendParam(),aTree->Get("ButDoExtendParam",1),bool(false)); //tototo 

   xml_init(anObj.ButDoGenCorPxTransv(),aTree->Get("ButDoGenCorPxTransv",1),bool(false)); //tototo 

   xml_init(anObj.ButDoPartiesCachees(),aTree->Get("ButDoPartiesCachees",1),bool(false)); //tototo 

   xml_init(anObj.ButDoOrtho(),aTree->Get("ButDoOrtho",1),bool(false)); //tototo 

   xml_init(anObj.ButDoSimul(),aTree->Get("ButDoSimul",1),bool(false)); //tototo 

   xml_init(anObj.ButDoRedrLocAnam(),aTree->Get("ButDoRedrLocAnam",1),bool(false)); //tototo 
}

std::string  Mangling( cDoNothingBut *) {return "1E28063E33C0A894FD3F";};


Pt2dr & cFoncPer::Per()
{
   return mPer;
}

const Pt2dr & cFoncPer::Per()const 
{
   return mPer;
}


double & cFoncPer::Ampl()
{
   return mAmpl;
}

const double & cFoncPer::Ampl()const 
{
   return mAmpl;
}


cTplValGesInit< bool > & cFoncPer::AmplIsDer()
{
   return mAmplIsDer;
}

const cTplValGesInit< bool > & cFoncPer::AmplIsDer()const 
{
   return mAmplIsDer;
}

void  BinaryUnDumpFromFile(cFoncPer & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Per(),aFp);
    BinaryUnDumpFromFile(anObj.Ampl(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.AmplIsDer().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.AmplIsDer().ValForcedForUnUmp(),aFp);
        }
        else  anObj.AmplIsDer().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cFoncPer & anObj)
{
    BinaryDumpInFile(aFp,anObj.Per());
    BinaryDumpInFile(aFp,anObj.Ampl());
    BinaryDumpInFile(aFp,anObj.AmplIsDer().IsInit());
    if (anObj.AmplIsDer().IsInit()) BinaryDumpInFile(aFp,anObj.AmplIsDer().Val());
}

cElXMLTree * ToXMLTree(const cFoncPer & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"FoncPer",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Per"),anObj.Per())->ReTagThis("Per"));
   aRes->AddFils(::ToXMLTree(std::string("Ampl"),anObj.Ampl())->ReTagThis("Ampl"));
   if (anObj.AmplIsDer().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("AmplIsDer"),anObj.AmplIsDer().Val())->ReTagThis("AmplIsDer"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cFoncPer & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.Per(),aTree->Get("Per",1)); //tototo 

   xml_init(anObj.Ampl(),aTree->Get("Ampl",1)); //tototo 

   xml_init(anObj.AmplIsDer(),aTree->Get("AmplIsDer",1),bool(true)); //tototo 
}

std::string  Mangling( cFoncPer *) {return "D94DBB939256DCD8FD3F";};


cTplValGesInit< Pt2dr > & cMNTPart::PenteGlob()
{
   return mPenteGlob;
}

const cTplValGesInit< Pt2dr > & cMNTPart::PenteGlob()const 
{
   return mPenteGlob;
}


std::list< cFoncPer > & cMNTPart::FoncPer()
{
   return mFoncPer;
}

const std::list< cFoncPer > & cMNTPart::FoncPer()const 
{
   return mFoncPer;
}

void  BinaryUnDumpFromFile(cMNTPart & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.PenteGlob().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.PenteGlob().ValForcedForUnUmp(),aFp);
        }
        else  anObj.PenteGlob().SetNoInit();
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cFoncPer aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.FoncPer().push_back(aVal);
        }
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cMNTPart & anObj)
{
    BinaryDumpInFile(aFp,anObj.PenteGlob().IsInit());
    if (anObj.PenteGlob().IsInit()) BinaryDumpInFile(aFp,anObj.PenteGlob().Val());
    BinaryDumpInFile(aFp,(int)anObj.FoncPer().size());
    for(  std::list< cFoncPer >::const_iterator iT=anObj.FoncPer().begin();
         iT!=anObj.FoncPer().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
}

cElXMLTree * ToXMLTree(const cMNTPart & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"MNTPart",eXMLBranche);
   if (anObj.PenteGlob().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PenteGlob"),anObj.PenteGlob().Val())->ReTagThis("PenteGlob"));
  for
  (       std::list< cFoncPer >::const_iterator it=anObj.FoncPer().begin();
      it !=anObj.FoncPer().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("FoncPer"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cMNTPart & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.PenteGlob(),aTree->Get("PenteGlob",1),Pt2dr(Pt2dr(0,0))); //tototo 

   xml_init(anObj.FoncPer(),aTree->GetAll("FoncPer",false,1));
}

std::string  Mangling( cMNTPart *) {return "709CCCA9E206D280FA3F";};


int & cSimulBarres::Nb()
{
   return mNb;
}

const int & cSimulBarres::Nb()const 
{
   return mNb;
}


cTplValGesInit< double > & cSimulBarres::PowDistLongueur()
{
   return mPowDistLongueur;
}

const cTplValGesInit< double > & cSimulBarres::PowDistLongueur()const 
{
   return mPowDistLongueur;
}


Pt2dr & cSimulBarres::IntervLongeur()
{
   return mIntervLongeur;
}

const Pt2dr & cSimulBarres::IntervLongeur()const 
{
   return mIntervLongeur;
}


Pt2dr & cSimulBarres::IntervLargeur()
{
   return mIntervLargeur;
}

const Pt2dr & cSimulBarres::IntervLargeur()const 
{
   return mIntervLargeur;
}


Pt2dr & cSimulBarres::IntervPentes()
{
   return mIntervPentes;
}

const Pt2dr & cSimulBarres::IntervPentes()const 
{
   return mIntervPentes;
}


Pt2dr & cSimulBarres::IntervHauteur()
{
   return mIntervHauteur;
}

const Pt2dr & cSimulBarres::IntervHauteur()const 
{
   return mIntervHauteur;
}


cTplValGesInit< double > & cSimulBarres::ProbSortant()
{
   return mProbSortant;
}

const cTplValGesInit< double > & cSimulBarres::ProbSortant()const 
{
   return mProbSortant;
}

void  BinaryUnDumpFromFile(cSimulBarres & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Nb(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.PowDistLongueur().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.PowDistLongueur().ValForcedForUnUmp(),aFp);
        }
        else  anObj.PowDistLongueur().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.IntervLongeur(),aFp);
    BinaryUnDumpFromFile(anObj.IntervLargeur(),aFp);
    BinaryUnDumpFromFile(anObj.IntervPentes(),aFp);
    BinaryUnDumpFromFile(anObj.IntervHauteur(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ProbSortant().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ProbSortant().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ProbSortant().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cSimulBarres & anObj)
{
    BinaryDumpInFile(aFp,anObj.Nb());
    BinaryDumpInFile(aFp,anObj.PowDistLongueur().IsInit());
    if (anObj.PowDistLongueur().IsInit()) BinaryDumpInFile(aFp,anObj.PowDistLongueur().Val());
    BinaryDumpInFile(aFp,anObj.IntervLongeur());
    BinaryDumpInFile(aFp,anObj.IntervLargeur());
    BinaryDumpInFile(aFp,anObj.IntervPentes());
    BinaryDumpInFile(aFp,anObj.IntervHauteur());
    BinaryDumpInFile(aFp,anObj.ProbSortant().IsInit());
    if (anObj.ProbSortant().IsInit()) BinaryDumpInFile(aFp,anObj.ProbSortant().Val());
}

cElXMLTree * ToXMLTree(const cSimulBarres & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"SimulBarres",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Nb"),anObj.Nb())->ReTagThis("Nb"));
   if (anObj.PowDistLongueur().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PowDistLongueur"),anObj.PowDistLongueur().Val())->ReTagThis("PowDistLongueur"));
   aRes->AddFils(::ToXMLTree(std::string("IntervLongeur"),anObj.IntervLongeur())->ReTagThis("IntervLongeur"));
   aRes->AddFils(::ToXMLTree(std::string("IntervLargeur"),anObj.IntervLargeur())->ReTagThis("IntervLargeur"));
   aRes->AddFils(::ToXMLTree(std::string("IntervPentes"),anObj.IntervPentes())->ReTagThis("IntervPentes"));
   aRes->AddFils(::ToXMLTree(std::string("IntervHauteur"),anObj.IntervHauteur())->ReTagThis("IntervHauteur"));
   if (anObj.ProbSortant().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ProbSortant"),anObj.ProbSortant().Val())->ReTagThis("ProbSortant"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cSimulBarres & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.Nb(),aTree->Get("Nb",1)); //tototo 

   xml_init(anObj.PowDistLongueur(),aTree->Get("PowDistLongueur",1),double(2.0)); //tototo 

   xml_init(anObj.IntervLongeur(),aTree->Get("IntervLongeur",1)); //tototo 

   xml_init(anObj.IntervLargeur(),aTree->Get("IntervLargeur",1)); //tototo 

   xml_init(anObj.IntervPentes(),aTree->Get("IntervPentes",1)); //tototo 

   xml_init(anObj.IntervHauteur(),aTree->Get("IntervHauteur",1)); //tototo 

   xml_init(anObj.ProbSortant(),aTree->Get("ProbSortant",1),double(0.5)); //tototo 
}

std::string  Mangling( cSimulBarres *) {return "348DCBA8013DF7A5FE3F";};


std::list< cSimulBarres > & cMNEPart::SimulBarres()
{
   return mSimulBarres;
}

const std::list< cSimulBarres > & cMNEPart::SimulBarres()const 
{
   return mSimulBarres;
}

void  BinaryUnDumpFromFile(cMNEPart & anObj,ELISE_fp & aFp)
{
   { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cSimulBarres aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.SimulBarres().push_back(aVal);
        }
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cMNEPart & anObj)
{
    BinaryDumpInFile(aFp,(int)anObj.SimulBarres().size());
    for(  std::list< cSimulBarres >::const_iterator iT=anObj.SimulBarres().begin();
         iT!=anObj.SimulBarres().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
}

cElXMLTree * ToXMLTree(const cMNEPart & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"MNEPart",eXMLBranche);
  for
  (       std::list< cSimulBarres >::const_iterator it=anObj.SimulBarres().begin();
      it !=anObj.SimulBarres().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("SimulBarres"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cMNEPart & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.SimulBarres(),aTree->GetAll("SimulBarres",false,1));
}

std::string  Mangling( cMNEPart *) {return "04C681DDA7E443BCFF3F";};


cTplValGesInit< bool > & cSimulRelief::DoItR()
{
   return mDoItR;
}

const cTplValGesInit< bool > & cSimulRelief::DoItR()const 
{
   return mDoItR;
}


cTplValGesInit< Pt2dr > & cSimulRelief::PenteGlob()
{
   return MNTPart().PenteGlob();
}

const cTplValGesInit< Pt2dr > & cSimulRelief::PenteGlob()const 
{
   return MNTPart().PenteGlob();
}


std::list< cFoncPer > & cSimulRelief::FoncPer()
{
   return MNTPart().FoncPer();
}

const std::list< cFoncPer > & cSimulRelief::FoncPer()const 
{
   return MNTPart().FoncPer();
}


cMNTPart & cSimulRelief::MNTPart()
{
   return mMNTPart;
}

const cMNTPart & cSimulRelief::MNTPart()const 
{
   return mMNTPart;
}


std::list< cSimulBarres > & cSimulRelief::SimulBarres()
{
   return MNEPart().SimulBarres();
}

const std::list< cSimulBarres > & cSimulRelief::SimulBarres()const 
{
   return MNEPart().SimulBarres();
}


cMNEPart & cSimulRelief::MNEPart()
{
   return mMNEPart;
}

const cMNEPart & cSimulRelief::MNEPart()const 
{
   return mMNEPart;
}

void  BinaryUnDumpFromFile(cSimulRelief & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.DoItR().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.DoItR().ValForcedForUnUmp(),aFp);
        }
        else  anObj.DoItR().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.MNTPart(),aFp);
    BinaryUnDumpFromFile(anObj.MNEPart(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cSimulRelief & anObj)
{
    BinaryDumpInFile(aFp,anObj.DoItR().IsInit());
    if (anObj.DoItR().IsInit()) BinaryDumpInFile(aFp,anObj.DoItR().Val());
    BinaryDumpInFile(aFp,anObj.MNTPart());
    BinaryDumpInFile(aFp,anObj.MNEPart());
}

cElXMLTree * ToXMLTree(const cSimulRelief & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"SimulRelief",eXMLBranche);
   if (anObj.DoItR().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DoItR"),anObj.DoItR().Val())->ReTagThis("DoItR"));
   aRes->AddFils(ToXMLTree(anObj.MNTPart())->ReTagThis("MNTPart"));
   aRes->AddFils(ToXMLTree(anObj.MNEPart())->ReTagThis("MNEPart"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cSimulRelief & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.DoItR(),aTree->Get("DoItR",1),bool(false)); //tototo 

   xml_init(anObj.MNTPart(),aTree->Get("MNTPart",1)); //tototo 

   xml_init(anObj.MNEPart(),aTree->Get("MNEPart",1)); //tototo 
}

std::string  Mangling( cSimulRelief *) {return "4EDC074E673D3797FE3F";};


std::string & cTexturePart::Texton()
{
   return mTexton;
}

const std::string & cTexturePart::Texton()const 
{
   return mTexton;
}


std::string & cTexturePart::ImRes()
{
   return mImRes;
}

const std::string & cTexturePart::ImRes()const 
{
   return mImRes;
}

void  BinaryUnDumpFromFile(cTexturePart & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Texton(),aFp);
    BinaryUnDumpFromFile(anObj.ImRes(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cTexturePart & anObj)
{
    BinaryDumpInFile(aFp,anObj.Texton());
    BinaryDumpInFile(aFp,anObj.ImRes());
}

cElXMLTree * ToXMLTree(const cTexturePart & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"TexturePart",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Texton"),anObj.Texton())->ReTagThis("Texton"));
   aRes->AddFils(::ToXMLTree(std::string("ImRes"),anObj.ImRes())->ReTagThis("ImRes"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cTexturePart & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.Texton(),aTree->Get("Texton",1)); //tototo 

   xml_init(anObj.ImRes(),aTree->Get("ImRes",1)); //tototo 
}

std::string  Mangling( cTexturePart *) {return "974152881D7F84BEFE3F";};


cElRegex_Ptr & cProjImPart::PatternSel()
{
   return mPatternSel;
}

const cElRegex_Ptr & cProjImPart::PatternSel()const 
{
   return mPatternSel;
}


cTplValGesInit< int > & cProjImPart::SzBloc()
{
   return mSzBloc;
}

const cTplValGesInit< int > & cProjImPart::SzBloc()const 
{
   return mSzBloc;
}


cTplValGesInit< int > & cProjImPart::SzBrd()
{
   return mSzBrd;
}

const cTplValGesInit< int > & cProjImPart::SzBrd()const 
{
   return mSzBrd;
}


cTplValGesInit< double > & cProjImPart::RatioSurResol()
{
   return mRatioSurResol;
}

const cTplValGesInit< double > & cProjImPart::RatioSurResol()const 
{
   return mRatioSurResol;
}


cTplValGesInit< std::string > & cProjImPart::KeyProjMNT()
{
   return mKeyProjMNT;
}

const cTplValGesInit< std::string > & cProjImPart::KeyProjMNT()const 
{
   return mKeyProjMNT;
}


cTplValGesInit< std::string > & cProjImPart::KeyIm()
{
   return mKeyIm;
}

const cTplValGesInit< std::string > & cProjImPart::KeyIm()const 
{
   return mKeyIm;
}


cTplValGesInit< double > & cProjImPart::BicubParam()
{
   return mBicubParam;
}

const cTplValGesInit< double > & cProjImPart::BicubParam()const 
{
   return mBicubParam;
}


cTplValGesInit< bool > & cProjImPart::ReprojInverse()
{
   return mReprojInverse;
}

const cTplValGesInit< bool > & cProjImPart::ReprojInverse()const 
{
   return mReprojInverse;
}


cTplValGesInit< double > & cProjImPart::SzFTM()
{
   return mSzFTM;
}

const cTplValGesInit< double > & cProjImPart::SzFTM()const 
{
   return mSzFTM;
}


cTplValGesInit< double > & cProjImPart::Bruit()
{
   return mBruit;
}

const cTplValGesInit< double > & cProjImPart::Bruit()const 
{
   return mBruit;
}

void  BinaryUnDumpFromFile(cProjImPart & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.PatternSel(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SzBloc().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SzBloc().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SzBloc().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SzBrd().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SzBrd().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SzBrd().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.RatioSurResol().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.RatioSurResol().ValForcedForUnUmp(),aFp);
        }
        else  anObj.RatioSurResol().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.KeyProjMNT().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.KeyProjMNT().ValForcedForUnUmp(),aFp);
        }
        else  anObj.KeyProjMNT().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.KeyIm().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.KeyIm().ValForcedForUnUmp(),aFp);
        }
        else  anObj.KeyIm().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.BicubParam().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.BicubParam().ValForcedForUnUmp(),aFp);
        }
        else  anObj.BicubParam().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ReprojInverse().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ReprojInverse().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ReprojInverse().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SzFTM().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SzFTM().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SzFTM().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Bruit().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Bruit().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Bruit().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cProjImPart & anObj)
{
    BinaryDumpInFile(aFp,anObj.PatternSel());
    BinaryDumpInFile(aFp,anObj.SzBloc().IsInit());
    if (anObj.SzBloc().IsInit()) BinaryDumpInFile(aFp,anObj.SzBloc().Val());
    BinaryDumpInFile(aFp,anObj.SzBrd().IsInit());
    if (anObj.SzBrd().IsInit()) BinaryDumpInFile(aFp,anObj.SzBrd().Val());
    BinaryDumpInFile(aFp,anObj.RatioSurResol().IsInit());
    if (anObj.RatioSurResol().IsInit()) BinaryDumpInFile(aFp,anObj.RatioSurResol().Val());
    BinaryDumpInFile(aFp,anObj.KeyProjMNT().IsInit());
    if (anObj.KeyProjMNT().IsInit()) BinaryDumpInFile(aFp,anObj.KeyProjMNT().Val());
    BinaryDumpInFile(aFp,anObj.KeyIm().IsInit());
    if (anObj.KeyIm().IsInit()) BinaryDumpInFile(aFp,anObj.KeyIm().Val());
    BinaryDumpInFile(aFp,anObj.BicubParam().IsInit());
    if (anObj.BicubParam().IsInit()) BinaryDumpInFile(aFp,anObj.BicubParam().Val());
    BinaryDumpInFile(aFp,anObj.ReprojInverse().IsInit());
    if (anObj.ReprojInverse().IsInit()) BinaryDumpInFile(aFp,anObj.ReprojInverse().Val());
    BinaryDumpInFile(aFp,anObj.SzFTM().IsInit());
    if (anObj.SzFTM().IsInit()) BinaryDumpInFile(aFp,anObj.SzFTM().Val());
    BinaryDumpInFile(aFp,anObj.Bruit().IsInit());
    if (anObj.Bruit().IsInit()) BinaryDumpInFile(aFp,anObj.Bruit().Val());
}

cElXMLTree * ToXMLTree(const cProjImPart & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ProjImPart",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("PatternSel"),anObj.PatternSel())->ReTagThis("PatternSel"));
   if (anObj.SzBloc().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SzBloc"),anObj.SzBloc().Val())->ReTagThis("SzBloc"));
   if (anObj.SzBrd().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SzBrd"),anObj.SzBrd().Val())->ReTagThis("SzBrd"));
   if (anObj.RatioSurResol().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("RatioSurResol"),anObj.RatioSurResol().Val())->ReTagThis("RatioSurResol"));
   if (anObj.KeyProjMNT().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("KeyProjMNT"),anObj.KeyProjMNT().Val())->ReTagThis("KeyProjMNT"));
   if (anObj.KeyIm().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("KeyIm"),anObj.KeyIm().Val())->ReTagThis("KeyIm"));
   if (anObj.BicubParam().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("BicubParam"),anObj.BicubParam().Val())->ReTagThis("BicubParam"));
   if (anObj.ReprojInverse().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ReprojInverse"),anObj.ReprojInverse().Val())->ReTagThis("ReprojInverse"));
   if (anObj.SzFTM().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SzFTM"),anObj.SzFTM().Val())->ReTagThis("SzFTM"));
   if (anObj.Bruit().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Bruit"),anObj.Bruit().Val())->ReTagThis("Bruit"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cProjImPart & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.PatternSel(),aTree->Get("PatternSel",1)); //tototo 

   xml_init(anObj.SzBloc(),aTree->Get("SzBloc",1),int(100000)); //tototo 

   xml_init(anObj.SzBrd(),aTree->Get("SzBrd",1),int(20)); //tototo 

   xml_init(anObj.RatioSurResol(),aTree->Get("RatioSurResol",1)); //tototo 

   xml_init(anObj.KeyProjMNT(),aTree->Get("KeyProjMNT",1)); //tototo 

   xml_init(anObj.KeyIm(),aTree->Get("KeyIm",1)); //tototo 

   xml_init(anObj.BicubParam(),aTree->Get("BicubParam",1),double(-0.5)); //tototo 

   xml_init(anObj.ReprojInverse(),aTree->Get("ReprojInverse",1),bool(false)); //tototo 

   xml_init(anObj.SzFTM(),aTree->Get("SzFTM",1),double(1.0)); //tototo 

   xml_init(anObj.Bruit(),aTree->Get("Bruit",1)); //tototo 
}

std::string  Mangling( cProjImPart *) {return "D79AD7807ABF20CEFE3F";};


cTplValGesInit< bool > & cSectionSimulation::DoItR()
{
   return SimulRelief().DoItR();
}

const cTplValGesInit< bool > & cSectionSimulation::DoItR()const 
{
   return SimulRelief().DoItR();
}


cTplValGesInit< Pt2dr > & cSectionSimulation::PenteGlob()
{
   return SimulRelief().MNTPart().PenteGlob();
}

const cTplValGesInit< Pt2dr > & cSectionSimulation::PenteGlob()const 
{
   return SimulRelief().MNTPart().PenteGlob();
}


std::list< cFoncPer > & cSectionSimulation::FoncPer()
{
   return SimulRelief().MNTPart().FoncPer();
}

const std::list< cFoncPer > & cSectionSimulation::FoncPer()const 
{
   return SimulRelief().MNTPart().FoncPer();
}


cMNTPart & cSectionSimulation::MNTPart()
{
   return SimulRelief().MNTPart();
}

const cMNTPart & cSectionSimulation::MNTPart()const 
{
   return SimulRelief().MNTPart();
}


std::list< cSimulBarres > & cSectionSimulation::SimulBarres()
{
   return SimulRelief().MNEPart().SimulBarres();
}

const std::list< cSimulBarres > & cSectionSimulation::SimulBarres()const 
{
   return SimulRelief().MNEPart().SimulBarres();
}


cMNEPart & cSectionSimulation::MNEPart()
{
   return SimulRelief().MNEPart();
}

const cMNEPart & cSectionSimulation::MNEPart()const 
{
   return SimulRelief().MNEPart();
}


cSimulRelief & cSectionSimulation::SimulRelief()
{
   return mSimulRelief;
}

const cSimulRelief & cSectionSimulation::SimulRelief()const 
{
   return mSimulRelief;
}


std::string & cSectionSimulation::Texton()
{
   return TexturePart().Texton();
}

const std::string & cSectionSimulation::Texton()const 
{
   return TexturePart().Texton();
}


std::string & cSectionSimulation::ImRes()
{
   return TexturePart().ImRes();
}

const std::string & cSectionSimulation::ImRes()const 
{
   return TexturePart().ImRes();
}


cTexturePart & cSectionSimulation::TexturePart()
{
   return mTexturePart;
}

const cTexturePart & cSectionSimulation::TexturePart()const 
{
   return mTexturePart;
}


cElRegex_Ptr & cSectionSimulation::PatternSel()
{
   return ProjImPart().PatternSel();
}

const cElRegex_Ptr & cSectionSimulation::PatternSel()const 
{
   return ProjImPart().PatternSel();
}


cTplValGesInit< int > & cSectionSimulation::SzBloc()
{
   return ProjImPart().SzBloc();
}

const cTplValGesInit< int > & cSectionSimulation::SzBloc()const 
{
   return ProjImPart().SzBloc();
}


cTplValGesInit< int > & cSectionSimulation::SzBrd()
{
   return ProjImPart().SzBrd();
}

const cTplValGesInit< int > & cSectionSimulation::SzBrd()const 
{
   return ProjImPart().SzBrd();
}


cTplValGesInit< double > & cSectionSimulation::RatioSurResol()
{
   return ProjImPart().RatioSurResol();
}

const cTplValGesInit< double > & cSectionSimulation::RatioSurResol()const 
{
   return ProjImPart().RatioSurResol();
}


cTplValGesInit< std::string > & cSectionSimulation::KeyProjMNT()
{
   return ProjImPart().KeyProjMNT();
}

const cTplValGesInit< std::string > & cSectionSimulation::KeyProjMNT()const 
{
   return ProjImPart().KeyProjMNT();
}


cTplValGesInit< std::string > & cSectionSimulation::KeyIm()
{
   return ProjImPart().KeyIm();
}

const cTplValGesInit< std::string > & cSectionSimulation::KeyIm()const 
{
   return ProjImPart().KeyIm();
}


cTplValGesInit< double > & cSectionSimulation::BicubParam()
{
   return ProjImPart().BicubParam();
}

const cTplValGesInit< double > & cSectionSimulation::BicubParam()const 
{
   return ProjImPart().BicubParam();
}


cTplValGesInit< bool > & cSectionSimulation::ReprojInverse()
{
   return ProjImPart().ReprojInverse();
}

const cTplValGesInit< bool > & cSectionSimulation::ReprojInverse()const 
{
   return ProjImPart().ReprojInverse();
}


cTplValGesInit< double > & cSectionSimulation::SzFTM()
{
   return ProjImPart().SzFTM();
}

const cTplValGesInit< double > & cSectionSimulation::SzFTM()const 
{
   return ProjImPart().SzFTM();
}


cTplValGesInit< double > & cSectionSimulation::Bruit()
{
   return ProjImPart().Bruit();
}

const cTplValGesInit< double > & cSectionSimulation::Bruit()const 
{
   return ProjImPart().Bruit();
}


cProjImPart & cSectionSimulation::ProjImPart()
{
   return mProjImPart;
}

const cProjImPart & cSectionSimulation::ProjImPart()const 
{
   return mProjImPart;
}

void  BinaryUnDumpFromFile(cSectionSimulation & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.SimulRelief(),aFp);
    BinaryUnDumpFromFile(anObj.TexturePart(),aFp);
    BinaryUnDumpFromFile(anObj.ProjImPart(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cSectionSimulation & anObj)
{
    BinaryDumpInFile(aFp,anObj.SimulRelief());
    BinaryDumpInFile(aFp,anObj.TexturePart());
    BinaryDumpInFile(aFp,anObj.ProjImPart());
}

cElXMLTree * ToXMLTree(const cSectionSimulation & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"SectionSimulation",eXMLBranche);
   aRes->AddFils(ToXMLTree(anObj.SimulRelief())->ReTagThis("SimulRelief"));
   aRes->AddFils(ToXMLTree(anObj.TexturePart())->ReTagThis("TexturePart"));
   aRes->AddFils(ToXMLTree(anObj.ProjImPart())->ReTagThis("ProjImPart"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cSectionSimulation & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.SimulRelief(),aTree->Get("SimulRelief",1)); //tototo 

   xml_init(anObj.TexturePart(),aTree->Get("TexturePart",1)); //tototo 

   xml_init(anObj.ProjImPart(),aTree->Get("ProjImPart",1)); //tototo 
}

std::string  Mangling( cSectionSimulation *) {return "56421F152403C7F7FE3F";};


std::string & cAnamSurfaceAnalytique::NameFile()
{
   return mNameFile;
}

const std::string & cAnamSurfaceAnalytique::NameFile()const 
{
   return mNameFile;
}


std::string & cAnamSurfaceAnalytique::Id()
{
   return mId;
}

const std::string & cAnamSurfaceAnalytique::Id()const 
{
   return mId;
}

void  BinaryUnDumpFromFile(cAnamSurfaceAnalytique & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.NameFile(),aFp);
    BinaryUnDumpFromFile(anObj.Id(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cAnamSurfaceAnalytique & anObj)
{
    BinaryDumpInFile(aFp,anObj.NameFile());
    BinaryDumpInFile(aFp,anObj.Id());
}

cElXMLTree * ToXMLTree(const cAnamSurfaceAnalytique & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"AnamSurfaceAnalytique",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("NameFile"),anObj.NameFile())->ReTagThis("NameFile"));
   aRes->AddFils(::ToXMLTree(std::string("Id"),anObj.Id())->ReTagThis("Id"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cAnamSurfaceAnalytique & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.NameFile(),aTree->Get("NameFile",1)); //tototo 

   xml_init(anObj.Id(),aTree->Get("Id",1)); //tototo 
}

std::string  Mangling( cAnamSurfaceAnalytique *) {return "A649061F52682FC6FE3F";};


cTplValGesInit< double > & cMakeMaskImNadir::DynIncid()
{
   return mDynIncid;
}

const cTplValGesInit< double > & cMakeMaskImNadir::DynIncid()const 
{
   return mDynIncid;
}


cTplValGesInit< bool > & cMakeMaskImNadir::MakeAlsoMaskTerrain()
{
   return mMakeAlsoMaskTerrain;
}

const cTplValGesInit< bool > & cMakeMaskImNadir::MakeAlsoMaskTerrain()const 
{
   return mMakeAlsoMaskTerrain;
}


int & cMakeMaskImNadir::KBest()
{
   return mKBest;
}

const int & cMakeMaskImNadir::KBest()const 
{
   return mKBest;
}


cTplValGesInit< double > & cMakeMaskImNadir::IncertAngle()
{
   return mIncertAngle;
}

const cTplValGesInit< double > & cMakeMaskImNadir::IncertAngle()const 
{
   return mIncertAngle;
}


cTplValGesInit< int > & cMakeMaskImNadir::Dilat32()
{
   return mDilat32;
}

const cTplValGesInit< int > & cMakeMaskImNadir::Dilat32()const 
{
   return mDilat32;
}


cTplValGesInit< int > & cMakeMaskImNadir::Erod32()
{
   return mErod32;
}

const cTplValGesInit< int > & cMakeMaskImNadir::Erod32()const 
{
   return mErod32;
}

void  BinaryUnDumpFromFile(cMakeMaskImNadir & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.DynIncid().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.DynIncid().ValForcedForUnUmp(),aFp);
        }
        else  anObj.DynIncid().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.MakeAlsoMaskTerrain().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.MakeAlsoMaskTerrain().ValForcedForUnUmp(),aFp);
        }
        else  anObj.MakeAlsoMaskTerrain().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.KBest(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.IncertAngle().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.IncertAngle().ValForcedForUnUmp(),aFp);
        }
        else  anObj.IncertAngle().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Dilat32().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Dilat32().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Dilat32().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Erod32().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Erod32().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Erod32().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cMakeMaskImNadir & anObj)
{
    BinaryDumpInFile(aFp,anObj.DynIncid().IsInit());
    if (anObj.DynIncid().IsInit()) BinaryDumpInFile(aFp,anObj.DynIncid().Val());
    BinaryDumpInFile(aFp,anObj.MakeAlsoMaskTerrain().IsInit());
    if (anObj.MakeAlsoMaskTerrain().IsInit()) BinaryDumpInFile(aFp,anObj.MakeAlsoMaskTerrain().Val());
    BinaryDumpInFile(aFp,anObj.KBest());
    BinaryDumpInFile(aFp,anObj.IncertAngle().IsInit());
    if (anObj.IncertAngle().IsInit()) BinaryDumpInFile(aFp,anObj.IncertAngle().Val());
    BinaryDumpInFile(aFp,anObj.Dilat32().IsInit());
    if (anObj.Dilat32().IsInit()) BinaryDumpInFile(aFp,anObj.Dilat32().Val());
    BinaryDumpInFile(aFp,anObj.Erod32().IsInit());
    if (anObj.Erod32().IsInit()) BinaryDumpInFile(aFp,anObj.Erod32().Val());
}

cElXMLTree * ToXMLTree(const cMakeMaskImNadir & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"MakeMaskImNadir",eXMLBranche);
   if (anObj.DynIncid().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DynIncid"),anObj.DynIncid().Val())->ReTagThis("DynIncid"));
   if (anObj.MakeAlsoMaskTerrain().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("MakeAlsoMaskTerrain"),anObj.MakeAlsoMaskTerrain().Val())->ReTagThis("MakeAlsoMaskTerrain"));
   aRes->AddFils(::ToXMLTree(std::string("KBest"),anObj.KBest())->ReTagThis("KBest"));
   if (anObj.IncertAngle().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("IncertAngle"),anObj.IncertAngle().Val())->ReTagThis("IncertAngle"));
   if (anObj.Dilat32().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Dilat32"),anObj.Dilat32().Val())->ReTagThis("Dilat32"));
   if (anObj.Erod32().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Erod32"),anObj.Erod32().Val())->ReTagThis("Erod32"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cMakeMaskImNadir & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.DynIncid(),aTree->Get("DynIncid",1),double(1e4)); //tototo 

   xml_init(anObj.MakeAlsoMaskTerrain(),aTree->Get("MakeAlsoMaskTerrain",1),bool(false)); //tototo 

   xml_init(anObj.KBest(),aTree->Get("KBest",1)); //tototo 

   xml_init(anObj.IncertAngle(),aTree->Get("IncertAngle",1),double(0.1)); //tototo 

   xml_init(anObj.Dilat32(),aTree->Get("Dilat32",1),int(6)); //tototo 

   xml_init(anObj.Erod32(),aTree->Get("Erod32",1),int(3)); //tototo 
}

std::string  Mangling( cMakeMaskImNadir *) {return "0025D4BBC4A14F83FCBF";};


cTplValGesInit< bool > & cAnamorphoseGeometrieMNT::UnUseAnamXCste()
{
   return mUnUseAnamXCste;
}

const cTplValGesInit< bool > & cAnamorphoseGeometrieMNT::UnUseAnamXCste()const 
{
   return mUnUseAnamXCste;
}


std::string & cAnamorphoseGeometrieMNT::NameFile()
{
   return AnamSurfaceAnalytique().Val().NameFile();
}

const std::string & cAnamorphoseGeometrieMNT::NameFile()const 
{
   return AnamSurfaceAnalytique().Val().NameFile();
}


std::string & cAnamorphoseGeometrieMNT::Id()
{
   return AnamSurfaceAnalytique().Val().Id();
}

const std::string & cAnamorphoseGeometrieMNT::Id()const 
{
   return AnamSurfaceAnalytique().Val().Id();
}


cTplValGesInit< cAnamSurfaceAnalytique > & cAnamorphoseGeometrieMNT::AnamSurfaceAnalytique()
{
   return mAnamSurfaceAnalytique;
}

const cTplValGesInit< cAnamSurfaceAnalytique > & cAnamorphoseGeometrieMNT::AnamSurfaceAnalytique()const 
{
   return mAnamSurfaceAnalytique;
}


cTplValGesInit< int > & cAnamorphoseGeometrieMNT::AnamDeZoomMasq()
{
   return mAnamDeZoomMasq;
}

const cTplValGesInit< int > & cAnamorphoseGeometrieMNT::AnamDeZoomMasq()const 
{
   return mAnamDeZoomMasq;
}


cTplValGesInit< double > & cAnamorphoseGeometrieMNT::AnamLimAngleVisib()
{
   return mAnamLimAngleVisib;
}

const cTplValGesInit< double > & cAnamorphoseGeometrieMNT::AnamLimAngleVisib()const 
{
   return mAnamLimAngleVisib;
}


cTplValGesInit< double > & cAnamorphoseGeometrieMNT::DynIncid()
{
   return MakeMaskImNadir().Val().DynIncid();
}

const cTplValGesInit< double > & cAnamorphoseGeometrieMNT::DynIncid()const 
{
   return MakeMaskImNadir().Val().DynIncid();
}


cTplValGesInit< bool > & cAnamorphoseGeometrieMNT::MakeAlsoMaskTerrain()
{
   return MakeMaskImNadir().Val().MakeAlsoMaskTerrain();
}

const cTplValGesInit< bool > & cAnamorphoseGeometrieMNT::MakeAlsoMaskTerrain()const 
{
   return MakeMaskImNadir().Val().MakeAlsoMaskTerrain();
}


int & cAnamorphoseGeometrieMNT::KBest()
{
   return MakeMaskImNadir().Val().KBest();
}

const int & cAnamorphoseGeometrieMNT::KBest()const 
{
   return MakeMaskImNadir().Val().KBest();
}


cTplValGesInit< double > & cAnamorphoseGeometrieMNT::IncertAngle()
{
   return MakeMaskImNadir().Val().IncertAngle();
}

const cTplValGesInit< double > & cAnamorphoseGeometrieMNT::IncertAngle()const 
{
   return MakeMaskImNadir().Val().IncertAngle();
}


cTplValGesInit< int > & cAnamorphoseGeometrieMNT::Dilat32()
{
   return MakeMaskImNadir().Val().Dilat32();
}

const cTplValGesInit< int > & cAnamorphoseGeometrieMNT::Dilat32()const 
{
   return MakeMaskImNadir().Val().Dilat32();
}


cTplValGesInit< int > & cAnamorphoseGeometrieMNT::Erod32()
{
   return MakeMaskImNadir().Val().Erod32();
}

const cTplValGesInit< int > & cAnamorphoseGeometrieMNT::Erod32()const 
{
   return MakeMaskImNadir().Val().Erod32();
}


cTplValGesInit< cMakeMaskImNadir > & cAnamorphoseGeometrieMNT::MakeMaskImNadir()
{
   return mMakeMaskImNadir;
}

const cTplValGesInit< cMakeMaskImNadir > & cAnamorphoseGeometrieMNT::MakeMaskImNadir()const 
{
   return mMakeMaskImNadir;
}

void  BinaryUnDumpFromFile(cAnamorphoseGeometrieMNT & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.UnUseAnamXCste().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.UnUseAnamXCste().ValForcedForUnUmp(),aFp);
        }
        else  anObj.UnUseAnamXCste().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.AnamSurfaceAnalytique().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.AnamSurfaceAnalytique().ValForcedForUnUmp(),aFp);
        }
        else  anObj.AnamSurfaceAnalytique().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.AnamDeZoomMasq().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.AnamDeZoomMasq().ValForcedForUnUmp(),aFp);
        }
        else  anObj.AnamDeZoomMasq().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.AnamLimAngleVisib().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.AnamLimAngleVisib().ValForcedForUnUmp(),aFp);
        }
        else  anObj.AnamLimAngleVisib().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.MakeMaskImNadir().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.MakeMaskImNadir().ValForcedForUnUmp(),aFp);
        }
        else  anObj.MakeMaskImNadir().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cAnamorphoseGeometrieMNT & anObj)
{
    BinaryDumpInFile(aFp,anObj.UnUseAnamXCste().IsInit());
    if (anObj.UnUseAnamXCste().IsInit()) BinaryDumpInFile(aFp,anObj.UnUseAnamXCste().Val());
    BinaryDumpInFile(aFp,anObj.AnamSurfaceAnalytique().IsInit());
    if (anObj.AnamSurfaceAnalytique().IsInit()) BinaryDumpInFile(aFp,anObj.AnamSurfaceAnalytique().Val());
    BinaryDumpInFile(aFp,anObj.AnamDeZoomMasq().IsInit());
    if (anObj.AnamDeZoomMasq().IsInit()) BinaryDumpInFile(aFp,anObj.AnamDeZoomMasq().Val());
    BinaryDumpInFile(aFp,anObj.AnamLimAngleVisib().IsInit());
    if (anObj.AnamLimAngleVisib().IsInit()) BinaryDumpInFile(aFp,anObj.AnamLimAngleVisib().Val());
    BinaryDumpInFile(aFp,anObj.MakeMaskImNadir().IsInit());
    if (anObj.MakeMaskImNadir().IsInit()) BinaryDumpInFile(aFp,anObj.MakeMaskImNadir().Val());
}

cElXMLTree * ToXMLTree(const cAnamorphoseGeometrieMNT & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"AnamorphoseGeometrieMNT",eXMLBranche);
   if (anObj.UnUseAnamXCste().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("UnUseAnamXCste"),anObj.UnUseAnamXCste().Val())->ReTagThis("UnUseAnamXCste"));
   if (anObj.AnamSurfaceAnalytique().IsInit())
      aRes->AddFils(ToXMLTree(anObj.AnamSurfaceAnalytique().Val())->ReTagThis("AnamSurfaceAnalytique"));
   if (anObj.AnamDeZoomMasq().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("AnamDeZoomMasq"),anObj.AnamDeZoomMasq().Val())->ReTagThis("AnamDeZoomMasq"));
   if (anObj.AnamLimAngleVisib().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("AnamLimAngleVisib"),anObj.AnamLimAngleVisib().Val())->ReTagThis("AnamLimAngleVisib"));
   if (anObj.MakeMaskImNadir().IsInit())
      aRes->AddFils(ToXMLTree(anObj.MakeMaskImNadir().Val())->ReTagThis("MakeMaskImNadir"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cAnamorphoseGeometrieMNT & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.UnUseAnamXCste(),aTree->Get("UnUseAnamXCste",1),bool(false)); //tototo 

   xml_init(anObj.AnamSurfaceAnalytique(),aTree->Get("AnamSurfaceAnalytique",1)); //tototo 

   xml_init(anObj.AnamDeZoomMasq(),aTree->Get("AnamDeZoomMasq",1),int(16)); //tototo 

   xml_init(anObj.AnamLimAngleVisib(),aTree->Get("AnamLimAngleVisib",1),double(1.05)); //tototo 

   xml_init(anObj.MakeMaskImNadir(),aTree->Get("MakeMaskImNadir",1)); //tototo 
}

std::string  Mangling( cAnamorphoseGeometrieMNT *) {return "683ECB3AC5AC49F2FBBF";};


cElRegex_Ptr & cColorimetriesCanaux::CanalSelector()
{
   return mCanalSelector;
}

const cElRegex_Ptr & cColorimetriesCanaux::CanalSelector()const 
{
   return mCanalSelector;
}


cTplValGesInit< double > & cColorimetriesCanaux::ValBlanc()
{
   return mValBlanc;
}

const cTplValGesInit< double > & cColorimetriesCanaux::ValBlanc()const 
{
   return mValBlanc;
}


cTplValGesInit< double > & cColorimetriesCanaux::ValNoir()
{
   return mValNoir;
}

const cTplValGesInit< double > & cColorimetriesCanaux::ValNoir()const 
{
   return mValNoir;
}

void  BinaryUnDumpFromFile(cColorimetriesCanaux & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.CanalSelector(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ValBlanc().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ValBlanc().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ValBlanc().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ValNoir().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ValNoir().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ValNoir().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cColorimetriesCanaux & anObj)
{
    BinaryDumpInFile(aFp,anObj.CanalSelector());
    BinaryDumpInFile(aFp,anObj.ValBlanc().IsInit());
    if (anObj.ValBlanc().IsInit()) BinaryDumpInFile(aFp,anObj.ValBlanc().Val());
    BinaryDumpInFile(aFp,anObj.ValNoir().IsInit());
    if (anObj.ValNoir().IsInit()) BinaryDumpInFile(aFp,anObj.ValNoir().Val());
}

cElXMLTree * ToXMLTree(const cColorimetriesCanaux & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ColorimetriesCanaux",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("CanalSelector"),anObj.CanalSelector())->ReTagThis("CanalSelector"));
   if (anObj.ValBlanc().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ValBlanc"),anObj.ValBlanc().Val())->ReTagThis("ValBlanc"));
   if (anObj.ValNoir().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ValNoir"),anObj.ValNoir().Val())->ReTagThis("ValNoir"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cColorimetriesCanaux & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.CanalSelector(),aTree->Get("CanalSelector",1)); //tototo 

   xml_init(anObj.ValBlanc(),aTree->Get("ValBlanc",1)); //tototo 

   xml_init(anObj.ValNoir(),aTree->Get("ValNoir",1),double(0.0)); //tototo 
}

std::string  Mangling( cColorimetriesCanaux *) {return "10E32DEAA5D24EBDFE3F";};


Pt3di & cSuperpositionImages::OrdreChannels()
{
   return mOrdreChannels;
}

const Pt3di & cSuperpositionImages::OrdreChannels()const 
{
   return mOrdreChannels;
}


cTplValGesInit< Pt2di > & cSuperpositionImages::PtBalanceBlancs()
{
   return mPtBalanceBlancs;
}

const cTplValGesInit< Pt2di > & cSuperpositionImages::PtBalanceBlancs()const 
{
   return mPtBalanceBlancs;
}


cTplValGesInit< Pt2di > & cSuperpositionImages::P0Sup()
{
   return mP0Sup;
}

const cTplValGesInit< Pt2di > & cSuperpositionImages::P0Sup()const 
{
   return mP0Sup;
}


cTplValGesInit< Pt2di > & cSuperpositionImages::SzSup()
{
   return mSzSup;
}

const cTplValGesInit< Pt2di > & cSuperpositionImages::SzSup()const 
{
   return mSzSup;
}


cElRegex_Ptr & cSuperpositionImages::PatternSelGrid()
{
   return mPatternSelGrid;
}

const cElRegex_Ptr & cSuperpositionImages::PatternSelGrid()const 
{
   return mPatternSelGrid;
}


std::string & cSuperpositionImages::PatternNameGrid()
{
   return mPatternNameGrid;
}

const std::string & cSuperpositionImages::PatternNameGrid()const 
{
   return mPatternNameGrid;
}


std::list< cColorimetriesCanaux > & cSuperpositionImages::ColorimetriesCanaux()
{
   return mColorimetriesCanaux;
}

const std::list< cColorimetriesCanaux > & cSuperpositionImages::ColorimetriesCanaux()const 
{
   return mColorimetriesCanaux;
}


cTplValGesInit< double > & cSuperpositionImages::GammaCorrection()
{
   return mGammaCorrection;
}

const cTplValGesInit< double > & cSuperpositionImages::GammaCorrection()const 
{
   return mGammaCorrection;
}


cTplValGesInit< double > & cSuperpositionImages::MultiplicateurBlanc()
{
   return mMultiplicateurBlanc;
}

const cTplValGesInit< double > & cSuperpositionImages::MultiplicateurBlanc()const 
{
   return mMultiplicateurBlanc;
}


cTplValGesInit< bool > & cSuperpositionImages::GenFileImages()
{
   return mGenFileImages;
}

const cTplValGesInit< bool > & cSuperpositionImages::GenFileImages()const 
{
   return mGenFileImages;
}

void  BinaryUnDumpFromFile(cSuperpositionImages & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.OrdreChannels(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.PtBalanceBlancs().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.PtBalanceBlancs().ValForcedForUnUmp(),aFp);
        }
        else  anObj.PtBalanceBlancs().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.P0Sup().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.P0Sup().ValForcedForUnUmp(),aFp);
        }
        else  anObj.P0Sup().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SzSup().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SzSup().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SzSup().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.PatternSelGrid(),aFp);
    BinaryUnDumpFromFile(anObj.PatternNameGrid(),aFp);
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cColorimetriesCanaux aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.ColorimetriesCanaux().push_back(aVal);
        }
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.GammaCorrection().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.GammaCorrection().ValForcedForUnUmp(),aFp);
        }
        else  anObj.GammaCorrection().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.MultiplicateurBlanc().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.MultiplicateurBlanc().ValForcedForUnUmp(),aFp);
        }
        else  anObj.MultiplicateurBlanc().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.GenFileImages().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.GenFileImages().ValForcedForUnUmp(),aFp);
        }
        else  anObj.GenFileImages().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cSuperpositionImages & anObj)
{
    BinaryDumpInFile(aFp,anObj.OrdreChannels());
    BinaryDumpInFile(aFp,anObj.PtBalanceBlancs().IsInit());
    if (anObj.PtBalanceBlancs().IsInit()) BinaryDumpInFile(aFp,anObj.PtBalanceBlancs().Val());
    BinaryDumpInFile(aFp,anObj.P0Sup().IsInit());
    if (anObj.P0Sup().IsInit()) BinaryDumpInFile(aFp,anObj.P0Sup().Val());
    BinaryDumpInFile(aFp,anObj.SzSup().IsInit());
    if (anObj.SzSup().IsInit()) BinaryDumpInFile(aFp,anObj.SzSup().Val());
    BinaryDumpInFile(aFp,anObj.PatternSelGrid());
    BinaryDumpInFile(aFp,anObj.PatternNameGrid());
    BinaryDumpInFile(aFp,(int)anObj.ColorimetriesCanaux().size());
    for(  std::list< cColorimetriesCanaux >::const_iterator iT=anObj.ColorimetriesCanaux().begin();
         iT!=anObj.ColorimetriesCanaux().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.GammaCorrection().IsInit());
    if (anObj.GammaCorrection().IsInit()) BinaryDumpInFile(aFp,anObj.GammaCorrection().Val());
    BinaryDumpInFile(aFp,anObj.MultiplicateurBlanc().IsInit());
    if (anObj.MultiplicateurBlanc().IsInit()) BinaryDumpInFile(aFp,anObj.MultiplicateurBlanc().Val());
    BinaryDumpInFile(aFp,anObj.GenFileImages().IsInit());
    if (anObj.GenFileImages().IsInit()) BinaryDumpInFile(aFp,anObj.GenFileImages().Val());
}

cElXMLTree * ToXMLTree(const cSuperpositionImages & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"SuperpositionImages",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("OrdreChannels"),anObj.OrdreChannels())->ReTagThis("OrdreChannels"));
   if (anObj.PtBalanceBlancs().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PtBalanceBlancs"),anObj.PtBalanceBlancs().Val())->ReTagThis("PtBalanceBlancs"));
   if (anObj.P0Sup().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("P0Sup"),anObj.P0Sup().Val())->ReTagThis("P0Sup"));
   if (anObj.SzSup().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SzSup"),anObj.SzSup().Val())->ReTagThis("SzSup"));
   aRes->AddFils(::ToXMLTree(std::string("PatternSelGrid"),anObj.PatternSelGrid())->ReTagThis("PatternSelGrid"));
   aRes->AddFils(::ToXMLTree(std::string("PatternNameGrid"),anObj.PatternNameGrid())->ReTagThis("PatternNameGrid"));
  for
  (       std::list< cColorimetriesCanaux >::const_iterator it=anObj.ColorimetriesCanaux().begin();
      it !=anObj.ColorimetriesCanaux().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("ColorimetriesCanaux"));
   if (anObj.GammaCorrection().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("GammaCorrection"),anObj.GammaCorrection().Val())->ReTagThis("GammaCorrection"));
   if (anObj.MultiplicateurBlanc().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("MultiplicateurBlanc"),anObj.MultiplicateurBlanc().Val())->ReTagThis("MultiplicateurBlanc"));
   if (anObj.GenFileImages().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("GenFileImages"),anObj.GenFileImages().Val())->ReTagThis("GenFileImages"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cSuperpositionImages & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.OrdreChannels(),aTree->Get("OrdreChannels",1)); //tototo 

   xml_init(anObj.PtBalanceBlancs(),aTree->Get("PtBalanceBlancs",1)); //tototo 

   xml_init(anObj.P0Sup(),aTree->Get("P0Sup",1)); //tototo 

   xml_init(anObj.SzSup(),aTree->Get("SzSup",1)); //tototo 

   xml_init(anObj.PatternSelGrid(),aTree->Get("PatternSelGrid",1)); //tototo 

   xml_init(anObj.PatternNameGrid(),aTree->Get("PatternNameGrid",1)); //tototo 

   xml_init(anObj.ColorimetriesCanaux(),aTree->GetAll("ColorimetriesCanaux",false,1));

   xml_init(anObj.GammaCorrection(),aTree->Get("GammaCorrection",1),double(1.0)); //tototo 

   xml_init(anObj.MultiplicateurBlanc(),aTree->Get("MultiplicateurBlanc",1),double(1.0)); //tototo 

   xml_init(anObj.GenFileImages(),aTree->Get("GenFileImages",1),bool(false)); //tototo 
}

std::string  Mangling( cSuperpositionImages *) {return "F3E9DC5911090DAAFE3F";};


cTplValGesInit< bool > & cSection_Results::Use_MM_EtatAvancement()
{
   return mUse_MM_EtatAvancement;
}

const cTplValGesInit< bool > & cSection_Results::Use_MM_EtatAvancement()const 
{
   return mUse_MM_EtatAvancement;
}


cTplValGesInit< bool > & cSection_Results::ButDoPyram()
{
   return DoNothingBut().Val().ButDoPyram();
}

const cTplValGesInit< bool > & cSection_Results::ButDoPyram()const 
{
   return DoNothingBut().Val().ButDoPyram();
}


cTplValGesInit< bool > & cSection_Results::ButDoMasqIm()
{
   return DoNothingBut().Val().ButDoMasqIm();
}

const cTplValGesInit< bool > & cSection_Results::ButDoMasqIm()const 
{
   return DoNothingBut().Val().ButDoMasqIm();
}


cTplValGesInit< bool > & cSection_Results::ButDoMemPart()
{
   return DoNothingBut().Val().ButDoMemPart();
}

const cTplValGesInit< bool > & cSection_Results::ButDoMemPart()const 
{
   return DoNothingBut().Val().ButDoMemPart();
}


cTplValGesInit< bool > & cSection_Results::ButDoTA()
{
   return DoNothingBut().Val().ButDoTA();
}

const cTplValGesInit< bool > & cSection_Results::ButDoTA()const 
{
   return DoNothingBut().Val().ButDoTA();
}


cTplValGesInit< bool > & cSection_Results::ButDoMasqueChantier()
{
   return DoNothingBut().Val().ButDoMasqueChantier();
}

const cTplValGesInit< bool > & cSection_Results::ButDoMasqueChantier()const 
{
   return DoNothingBut().Val().ButDoMasqueChantier();
}


cTplValGesInit< bool > & cSection_Results::ButDoOriMNT()
{
   return DoNothingBut().Val().ButDoOriMNT();
}

const cTplValGesInit< bool > & cSection_Results::ButDoOriMNT()const 
{
   return DoNothingBut().Val().ButDoOriMNT();
}


cTplValGesInit< bool > & cSection_Results::ButDoMTDNuage()
{
   return DoNothingBut().Val().ButDoMTDNuage();
}

const cTplValGesInit< bool > & cSection_Results::ButDoMTDNuage()const 
{
   return DoNothingBut().Val().ButDoMTDNuage();
}


cTplValGesInit< bool > & cSection_Results::ButDoFDC()
{
   return DoNothingBut().Val().ButDoFDC();
}

const cTplValGesInit< bool > & cSection_Results::ButDoFDC()const 
{
   return DoNothingBut().Val().ButDoFDC();
}


cTplValGesInit< bool > & cSection_Results::ButDoExtendParam()
{
   return DoNothingBut().Val().ButDoExtendParam();
}

const cTplValGesInit< bool > & cSection_Results::ButDoExtendParam()const 
{
   return DoNothingBut().Val().ButDoExtendParam();
}


cTplValGesInit< bool > & cSection_Results::ButDoGenCorPxTransv()
{
   return DoNothingBut().Val().ButDoGenCorPxTransv();
}

const cTplValGesInit< bool > & cSection_Results::ButDoGenCorPxTransv()const 
{
   return DoNothingBut().Val().ButDoGenCorPxTransv();
}


cTplValGesInit< bool > & cSection_Results::ButDoPartiesCachees()
{
   return DoNothingBut().Val().ButDoPartiesCachees();
}

const cTplValGesInit< bool > & cSection_Results::ButDoPartiesCachees()const 
{
   return DoNothingBut().Val().ButDoPartiesCachees();
}


cTplValGesInit< bool > & cSection_Results::ButDoOrtho()
{
   return DoNothingBut().Val().ButDoOrtho();
}

const cTplValGesInit< bool > & cSection_Results::ButDoOrtho()const 
{
   return DoNothingBut().Val().ButDoOrtho();
}


cTplValGesInit< bool > & cSection_Results::ButDoSimul()
{
   return DoNothingBut().Val().ButDoSimul();
}

const cTplValGesInit< bool > & cSection_Results::ButDoSimul()const 
{
   return DoNothingBut().Val().ButDoSimul();
}


cTplValGesInit< bool > & cSection_Results::ButDoRedrLocAnam()
{
   return DoNothingBut().Val().ButDoRedrLocAnam();
}

const cTplValGesInit< bool > & cSection_Results::ButDoRedrLocAnam()const 
{
   return DoNothingBut().Val().ButDoRedrLocAnam();
}


cTplValGesInit< cDoNothingBut > & cSection_Results::DoNothingBut()
{
   return mDoNothingBut;
}

const cTplValGesInit< cDoNothingBut > & cSection_Results::DoNothingBut()const 
{
   return mDoNothingBut;
}


cTplValGesInit< int > & cSection_Results::Paral_Pc_IdProcess()
{
   return mParal_Pc_IdProcess;
}

const cTplValGesInit< int > & cSection_Results::Paral_Pc_IdProcess()const 
{
   return mParal_Pc_IdProcess;
}


cTplValGesInit< int > & cSection_Results::Paral_Pc_NbProcess()
{
   return mParal_Pc_NbProcess;
}

const cTplValGesInit< int > & cSection_Results::Paral_Pc_NbProcess()const 
{
   return mParal_Pc_NbProcess;
}


cTplValGesInit< double > & cSection_Results::X_DirPlanInterFaisceau()
{
   return mX_DirPlanInterFaisceau;
}

const cTplValGesInit< double > & cSection_Results::X_DirPlanInterFaisceau()const 
{
   return mX_DirPlanInterFaisceau;
}


cTplValGesInit< double > & cSection_Results::Y_DirPlanInterFaisceau()
{
   return mY_DirPlanInterFaisceau;
}

const cTplValGesInit< double > & cSection_Results::Y_DirPlanInterFaisceau()const 
{
   return mY_DirPlanInterFaisceau;
}


cTplValGesInit< double > & cSection_Results::Z_DirPlanInterFaisceau()
{
   return mZ_DirPlanInterFaisceau;
}

const cTplValGesInit< double > & cSection_Results::Z_DirPlanInterFaisceau()const 
{
   return mZ_DirPlanInterFaisceau;
}


eModeGeomMNT & cSection_Results::GeomMNT()
{
   return mGeomMNT;
}

const eModeGeomMNT & cSection_Results::GeomMNT()const 
{
   return mGeomMNT;
}


cTplValGesInit< cSectionSimulation > & cSection_Results::SectionSimulation()
{
   return mSectionSimulation;
}

const cTplValGesInit< cSectionSimulation > & cSection_Results::SectionSimulation()const 
{
   return mSectionSimulation;
}


cTplValGesInit< bool > & cSection_Results::Prio2OwnAltisolForEmprise()
{
   return mPrio2OwnAltisolForEmprise;
}

const cTplValGesInit< bool > & cSection_Results::Prio2OwnAltisolForEmprise()const 
{
   return mPrio2OwnAltisolForEmprise;
}


cTplValGesInit< bool > & cSection_Results::UnUseAnamXCste()
{
   return AnamorphoseGeometrieMNT().Val().UnUseAnamXCste();
}

const cTplValGesInit< bool > & cSection_Results::UnUseAnamXCste()const 
{
   return AnamorphoseGeometrieMNT().Val().UnUseAnamXCste();
}


std::string & cSection_Results::NameFile()
{
   return AnamorphoseGeometrieMNT().Val().AnamSurfaceAnalytique().Val().NameFile();
}

const std::string & cSection_Results::NameFile()const 
{
   return AnamorphoseGeometrieMNT().Val().AnamSurfaceAnalytique().Val().NameFile();
}


std::string & cSection_Results::Id()
{
   return AnamorphoseGeometrieMNT().Val().AnamSurfaceAnalytique().Val().Id();
}

const std::string & cSection_Results::Id()const 
{
   return AnamorphoseGeometrieMNT().Val().AnamSurfaceAnalytique().Val().Id();
}


cTplValGesInit< cAnamSurfaceAnalytique > & cSection_Results::AnamSurfaceAnalytique()
{
   return AnamorphoseGeometrieMNT().Val().AnamSurfaceAnalytique();
}

const cTplValGesInit< cAnamSurfaceAnalytique > & cSection_Results::AnamSurfaceAnalytique()const 
{
   return AnamorphoseGeometrieMNT().Val().AnamSurfaceAnalytique();
}


cTplValGesInit< int > & cSection_Results::AnamDeZoomMasq()
{
   return AnamorphoseGeometrieMNT().Val().AnamDeZoomMasq();
}

const cTplValGesInit< int > & cSection_Results::AnamDeZoomMasq()const 
{
   return AnamorphoseGeometrieMNT().Val().AnamDeZoomMasq();
}


cTplValGesInit< double > & cSection_Results::AnamLimAngleVisib()
{
   return AnamorphoseGeometrieMNT().Val().AnamLimAngleVisib();
}

const cTplValGesInit< double > & cSection_Results::AnamLimAngleVisib()const 
{
   return AnamorphoseGeometrieMNT().Val().AnamLimAngleVisib();
}


cTplValGesInit< double > & cSection_Results::DynIncid()
{
   return AnamorphoseGeometrieMNT().Val().MakeMaskImNadir().Val().DynIncid();
}

const cTplValGesInit< double > & cSection_Results::DynIncid()const 
{
   return AnamorphoseGeometrieMNT().Val().MakeMaskImNadir().Val().DynIncid();
}


cTplValGesInit< bool > & cSection_Results::MakeAlsoMaskTerrain()
{
   return AnamorphoseGeometrieMNT().Val().MakeMaskImNadir().Val().MakeAlsoMaskTerrain();
}

const cTplValGesInit< bool > & cSection_Results::MakeAlsoMaskTerrain()const 
{
   return AnamorphoseGeometrieMNT().Val().MakeMaskImNadir().Val().MakeAlsoMaskTerrain();
}


int & cSection_Results::KBest()
{
   return AnamorphoseGeometrieMNT().Val().MakeMaskImNadir().Val().KBest();
}

const int & cSection_Results::KBest()const 
{
   return AnamorphoseGeometrieMNT().Val().MakeMaskImNadir().Val().KBest();
}


cTplValGesInit< double > & cSection_Results::IncertAngle()
{
   return AnamorphoseGeometrieMNT().Val().MakeMaskImNadir().Val().IncertAngle();
}

const cTplValGesInit< double > & cSection_Results::IncertAngle()const 
{
   return AnamorphoseGeometrieMNT().Val().MakeMaskImNadir().Val().IncertAngle();
}


cTplValGesInit< int > & cSection_Results::Dilat32()
{
   return AnamorphoseGeometrieMNT().Val().MakeMaskImNadir().Val().Dilat32();
}

const cTplValGesInit< int > & cSection_Results::Dilat32()const 
{
   return AnamorphoseGeometrieMNT().Val().MakeMaskImNadir().Val().Dilat32();
}


cTplValGesInit< int > & cSection_Results::Erod32()
{
   return AnamorphoseGeometrieMNT().Val().MakeMaskImNadir().Val().Erod32();
}

const cTplValGesInit< int > & cSection_Results::Erod32()const 
{
   return AnamorphoseGeometrieMNT().Val().MakeMaskImNadir().Val().Erod32();
}


cTplValGesInit< cMakeMaskImNadir > & cSection_Results::MakeMaskImNadir()
{
   return AnamorphoseGeometrieMNT().Val().MakeMaskImNadir();
}

const cTplValGesInit< cMakeMaskImNadir > & cSection_Results::MakeMaskImNadir()const 
{
   return AnamorphoseGeometrieMNT().Val().MakeMaskImNadir();
}


cTplValGesInit< cAnamorphoseGeometrieMNT > & cSection_Results::AnamorphoseGeometrieMNT()
{
   return mAnamorphoseGeometrieMNT;
}

const cTplValGesInit< cAnamorphoseGeometrieMNT > & cSection_Results::AnamorphoseGeometrieMNT()const 
{
   return mAnamorphoseGeometrieMNT;
}


cTplValGesInit< std::string > & cSection_Results::RepereCorrel()
{
   return mRepereCorrel;
}

const cTplValGesInit< std::string > & cSection_Results::RepereCorrel()const 
{
   return mRepereCorrel;
}


cTplValGesInit< std::string > & cSection_Results::TagRepereCorrel()
{
   return mTagRepereCorrel;
}

const cTplValGesInit< std::string > & cSection_Results::TagRepereCorrel()const 
{
   return mTagRepereCorrel;
}


cTplValGesInit< bool > & cSection_Results::DoMEC()
{
   return mDoMEC;
}

const cTplValGesInit< bool > & cSection_Results::DoMEC()const 
{
   return mDoMEC;
}


cTplValGesInit< std::string > & cSection_Results::NonExistingFileDoMEC()
{
   return mNonExistingFileDoMEC;
}

const cTplValGesInit< std::string > & cSection_Results::NonExistingFileDoMEC()const 
{
   return mNonExistingFileDoMEC;
}


cTplValGesInit< bool > & cSection_Results::DoFDC()
{
   return mDoFDC;
}

const cTplValGesInit< bool > & cSection_Results::DoFDC()const 
{
   return mDoFDC;
}


cTplValGesInit< bool > & cSection_Results::GenereXMLComp()
{
   return mGenereXMLComp;
}

const cTplValGesInit< bool > & cSection_Results::GenereXMLComp()const 
{
   return mGenereXMLComp;
}


cTplValGesInit< int > & cSection_Results::TAUseMasqNadirKBest()
{
   return mTAUseMasqNadirKBest;
}

const cTplValGesInit< int > & cSection_Results::TAUseMasqNadirKBest()const 
{
   return mTAUseMasqNadirKBest;
}


cTplValGesInit< int > & cSection_Results::ZoomMakeTA()
{
   return mZoomMakeTA;
}

const cTplValGesInit< int > & cSection_Results::ZoomMakeTA()const 
{
   return mZoomMakeTA;
}


cTplValGesInit< double > & cSection_Results::SaturationTA()
{
   return mSaturationTA;
}

const cTplValGesInit< double > & cSection_Results::SaturationTA()const 
{
   return mSaturationTA;
}


cTplValGesInit< bool > & cSection_Results::OrthoTA()
{
   return mOrthoTA;
}

const cTplValGesInit< bool > & cSection_Results::OrthoTA()const 
{
   return mOrthoTA;
}


cTplValGesInit< int > & cSection_Results::ZoomMakeMasq()
{
   return mZoomMakeMasq;
}

const cTplValGesInit< int > & cSection_Results::ZoomMakeMasq()const 
{
   return mZoomMakeMasq;
}


cTplValGesInit< bool > & cSection_Results::LazyZoomMaskTerrain()
{
   return mLazyZoomMaskTerrain;
}

const cTplValGesInit< bool > & cSection_Results::LazyZoomMaskTerrain()const 
{
   return mLazyZoomMaskTerrain;
}


cTplValGesInit< bool > & cSection_Results::MakeImCptTA()
{
   return mMakeImCptTA;
}

const cTplValGesInit< bool > & cSection_Results::MakeImCptTA()const 
{
   return mMakeImCptTA;
}


cTplValGesInit< std::string > & cSection_Results::FilterTA()
{
   return mFilterTA;
}

const cTplValGesInit< std::string > & cSection_Results::FilterTA()const 
{
   return mFilterTA;
}


cTplValGesInit< double > & cSection_Results::GammaVisu()
{
   return mGammaVisu;
}

const cTplValGesInit< double > & cSection_Results::GammaVisu()const 
{
   return mGammaVisu;
}


cTplValGesInit< int > & cSection_Results::ZoomVisuLiaison()
{
   return mZoomVisuLiaison;
}

const cTplValGesInit< int > & cSection_Results::ZoomVisuLiaison()const 
{
   return mZoomVisuLiaison;
}


cTplValGesInit< double > & cSection_Results::TolerancePointHomInImage()
{
   return mTolerancePointHomInImage;
}

const cTplValGesInit< double > & cSection_Results::TolerancePointHomInImage()const 
{
   return mTolerancePointHomInImage;
}


cTplValGesInit< double > & cSection_Results::FiltragePointHomInImage()
{
   return mFiltragePointHomInImage;
}

const cTplValGesInit< double > & cSection_Results::FiltragePointHomInImage()const 
{
   return mFiltragePointHomInImage;
}


cTplValGesInit< int > & cSection_Results::BaseCodeRetourMicmacErreur()
{
   return mBaseCodeRetourMicmacErreur;
}

const cTplValGesInit< int > & cSection_Results::BaseCodeRetourMicmacErreur()const 
{
   return mBaseCodeRetourMicmacErreur;
}


Pt3di & cSection_Results::OrdreChannels()
{
   return SuperpositionImages().Val().OrdreChannels();
}

const Pt3di & cSection_Results::OrdreChannels()const 
{
   return SuperpositionImages().Val().OrdreChannels();
}


cTplValGesInit< Pt2di > & cSection_Results::PtBalanceBlancs()
{
   return SuperpositionImages().Val().PtBalanceBlancs();
}

const cTplValGesInit< Pt2di > & cSection_Results::PtBalanceBlancs()const 
{
   return SuperpositionImages().Val().PtBalanceBlancs();
}


cTplValGesInit< Pt2di > & cSection_Results::P0Sup()
{
   return SuperpositionImages().Val().P0Sup();
}

const cTplValGesInit< Pt2di > & cSection_Results::P0Sup()const 
{
   return SuperpositionImages().Val().P0Sup();
}


cTplValGesInit< Pt2di > & cSection_Results::SzSup()
{
   return SuperpositionImages().Val().SzSup();
}

const cTplValGesInit< Pt2di > & cSection_Results::SzSup()const 
{
   return SuperpositionImages().Val().SzSup();
}


cElRegex_Ptr & cSection_Results::PatternSelGrid()
{
   return SuperpositionImages().Val().PatternSelGrid();
}

const cElRegex_Ptr & cSection_Results::PatternSelGrid()const 
{
   return SuperpositionImages().Val().PatternSelGrid();
}


std::string & cSection_Results::PatternNameGrid()
{
   return SuperpositionImages().Val().PatternNameGrid();
}

const std::string & cSection_Results::PatternNameGrid()const 
{
   return SuperpositionImages().Val().PatternNameGrid();
}


std::list< cColorimetriesCanaux > & cSection_Results::ColorimetriesCanaux()
{
   return SuperpositionImages().Val().ColorimetriesCanaux();
}

const std::list< cColorimetriesCanaux > & cSection_Results::ColorimetriesCanaux()const 
{
   return SuperpositionImages().Val().ColorimetriesCanaux();
}


cTplValGesInit< double > & cSection_Results::GammaCorrection()
{
   return SuperpositionImages().Val().GammaCorrection();
}

const cTplValGesInit< double > & cSection_Results::GammaCorrection()const 
{
   return SuperpositionImages().Val().GammaCorrection();
}


cTplValGesInit< double > & cSection_Results::MultiplicateurBlanc()
{
   return SuperpositionImages().Val().MultiplicateurBlanc();
}

const cTplValGesInit< double > & cSection_Results::MultiplicateurBlanc()const 
{
   return SuperpositionImages().Val().MultiplicateurBlanc();
}


cTplValGesInit< bool > & cSection_Results::GenFileImages()
{
   return SuperpositionImages().Val().GenFileImages();
}

const cTplValGesInit< bool > & cSection_Results::GenFileImages()const 
{
   return SuperpositionImages().Val().GenFileImages();
}


cTplValGesInit< cSuperpositionImages > & cSection_Results::SuperpositionImages()
{
   return mSuperpositionImages;
}

const cTplValGesInit< cSuperpositionImages > & cSection_Results::SuperpositionImages()const 
{
   return mSuperpositionImages;
}

void  BinaryUnDumpFromFile(cSection_Results & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Use_MM_EtatAvancement().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Use_MM_EtatAvancement().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Use_MM_EtatAvancement().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.DoNothingBut().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.DoNothingBut().ValForcedForUnUmp(),aFp);
        }
        else  anObj.DoNothingBut().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Paral_Pc_IdProcess().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Paral_Pc_IdProcess().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Paral_Pc_IdProcess().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Paral_Pc_NbProcess().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Paral_Pc_NbProcess().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Paral_Pc_NbProcess().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.X_DirPlanInterFaisceau().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.X_DirPlanInterFaisceau().ValForcedForUnUmp(),aFp);
        }
        else  anObj.X_DirPlanInterFaisceau().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Y_DirPlanInterFaisceau().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Y_DirPlanInterFaisceau().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Y_DirPlanInterFaisceau().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Z_DirPlanInterFaisceau().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Z_DirPlanInterFaisceau().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Z_DirPlanInterFaisceau().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.GeomMNT(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SectionSimulation().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SectionSimulation().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SectionSimulation().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Prio2OwnAltisolForEmprise().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Prio2OwnAltisolForEmprise().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Prio2OwnAltisolForEmprise().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.AnamorphoseGeometrieMNT().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.AnamorphoseGeometrieMNT().ValForcedForUnUmp(),aFp);
        }
        else  anObj.AnamorphoseGeometrieMNT().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.RepereCorrel().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.RepereCorrel().ValForcedForUnUmp(),aFp);
        }
        else  anObj.RepereCorrel().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.TagRepereCorrel().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.TagRepereCorrel().ValForcedForUnUmp(),aFp);
        }
        else  anObj.TagRepereCorrel().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.DoMEC().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.DoMEC().ValForcedForUnUmp(),aFp);
        }
        else  anObj.DoMEC().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NonExistingFileDoMEC().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NonExistingFileDoMEC().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NonExistingFileDoMEC().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.DoFDC().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.DoFDC().ValForcedForUnUmp(),aFp);
        }
        else  anObj.DoFDC().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.GenereXMLComp().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.GenereXMLComp().ValForcedForUnUmp(),aFp);
        }
        else  anObj.GenereXMLComp().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.TAUseMasqNadirKBest().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.TAUseMasqNadirKBest().ValForcedForUnUmp(),aFp);
        }
        else  anObj.TAUseMasqNadirKBest().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ZoomMakeTA().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ZoomMakeTA().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ZoomMakeTA().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SaturationTA().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SaturationTA().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SaturationTA().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.OrthoTA().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.OrthoTA().ValForcedForUnUmp(),aFp);
        }
        else  anObj.OrthoTA().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ZoomMakeMasq().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ZoomMakeMasq().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ZoomMakeMasq().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.LazyZoomMaskTerrain().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.LazyZoomMaskTerrain().ValForcedForUnUmp(),aFp);
        }
        else  anObj.LazyZoomMaskTerrain().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.MakeImCptTA().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.MakeImCptTA().ValForcedForUnUmp(),aFp);
        }
        else  anObj.MakeImCptTA().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.FilterTA().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.FilterTA().ValForcedForUnUmp(),aFp);
        }
        else  anObj.FilterTA().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.GammaVisu().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.GammaVisu().ValForcedForUnUmp(),aFp);
        }
        else  anObj.GammaVisu().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ZoomVisuLiaison().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ZoomVisuLiaison().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ZoomVisuLiaison().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.TolerancePointHomInImage().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.TolerancePointHomInImage().ValForcedForUnUmp(),aFp);
        }
        else  anObj.TolerancePointHomInImage().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.FiltragePointHomInImage().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.FiltragePointHomInImage().ValForcedForUnUmp(),aFp);
        }
        else  anObj.FiltragePointHomInImage().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.BaseCodeRetourMicmacErreur().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.BaseCodeRetourMicmacErreur().ValForcedForUnUmp(),aFp);
        }
        else  anObj.BaseCodeRetourMicmacErreur().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SuperpositionImages().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SuperpositionImages().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SuperpositionImages().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cSection_Results & anObj)
{
    BinaryDumpInFile(aFp,anObj.Use_MM_EtatAvancement().IsInit());
    if (anObj.Use_MM_EtatAvancement().IsInit()) BinaryDumpInFile(aFp,anObj.Use_MM_EtatAvancement().Val());
    BinaryDumpInFile(aFp,anObj.DoNothingBut().IsInit());
    if (anObj.DoNothingBut().IsInit()) BinaryDumpInFile(aFp,anObj.DoNothingBut().Val());
    BinaryDumpInFile(aFp,anObj.Paral_Pc_IdProcess().IsInit());
    if (anObj.Paral_Pc_IdProcess().IsInit()) BinaryDumpInFile(aFp,anObj.Paral_Pc_IdProcess().Val());
    BinaryDumpInFile(aFp,anObj.Paral_Pc_NbProcess().IsInit());
    if (anObj.Paral_Pc_NbProcess().IsInit()) BinaryDumpInFile(aFp,anObj.Paral_Pc_NbProcess().Val());
    BinaryDumpInFile(aFp,anObj.X_DirPlanInterFaisceau().IsInit());
    if (anObj.X_DirPlanInterFaisceau().IsInit()) BinaryDumpInFile(aFp,anObj.X_DirPlanInterFaisceau().Val());
    BinaryDumpInFile(aFp,anObj.Y_DirPlanInterFaisceau().IsInit());
    if (anObj.Y_DirPlanInterFaisceau().IsInit()) BinaryDumpInFile(aFp,anObj.Y_DirPlanInterFaisceau().Val());
    BinaryDumpInFile(aFp,anObj.Z_DirPlanInterFaisceau().IsInit());
    if (anObj.Z_DirPlanInterFaisceau().IsInit()) BinaryDumpInFile(aFp,anObj.Z_DirPlanInterFaisceau().Val());
    BinaryDumpInFile(aFp,anObj.GeomMNT());
    BinaryDumpInFile(aFp,anObj.SectionSimulation().IsInit());
    if (anObj.SectionSimulation().IsInit()) BinaryDumpInFile(aFp,anObj.SectionSimulation().Val());
    BinaryDumpInFile(aFp,anObj.Prio2OwnAltisolForEmprise().IsInit());
    if (anObj.Prio2OwnAltisolForEmprise().IsInit()) BinaryDumpInFile(aFp,anObj.Prio2OwnAltisolForEmprise().Val());
    BinaryDumpInFile(aFp,anObj.AnamorphoseGeometrieMNT().IsInit());
    if (anObj.AnamorphoseGeometrieMNT().IsInit()) BinaryDumpInFile(aFp,anObj.AnamorphoseGeometrieMNT().Val());
    BinaryDumpInFile(aFp,anObj.RepereCorrel().IsInit());
    if (anObj.RepereCorrel().IsInit()) BinaryDumpInFile(aFp,anObj.RepereCorrel().Val());
    BinaryDumpInFile(aFp,anObj.TagRepereCorrel().IsInit());
    if (anObj.TagRepereCorrel().IsInit()) BinaryDumpInFile(aFp,anObj.TagRepereCorrel().Val());
    BinaryDumpInFile(aFp,anObj.DoMEC().IsInit());
    if (anObj.DoMEC().IsInit()) BinaryDumpInFile(aFp,anObj.DoMEC().Val());
    BinaryDumpInFile(aFp,anObj.NonExistingFileDoMEC().IsInit());
    if (anObj.NonExistingFileDoMEC().IsInit()) BinaryDumpInFile(aFp,anObj.NonExistingFileDoMEC().Val());
    BinaryDumpInFile(aFp,anObj.DoFDC().IsInit());
    if (anObj.DoFDC().IsInit()) BinaryDumpInFile(aFp,anObj.DoFDC().Val());
    BinaryDumpInFile(aFp,anObj.GenereXMLComp().IsInit());
    if (anObj.GenereXMLComp().IsInit()) BinaryDumpInFile(aFp,anObj.GenereXMLComp().Val());
    BinaryDumpInFile(aFp,anObj.TAUseMasqNadirKBest().IsInit());
    if (anObj.TAUseMasqNadirKBest().IsInit()) BinaryDumpInFile(aFp,anObj.TAUseMasqNadirKBest().Val());
    BinaryDumpInFile(aFp,anObj.ZoomMakeTA().IsInit());
    if (anObj.ZoomMakeTA().IsInit()) BinaryDumpInFile(aFp,anObj.ZoomMakeTA().Val());
    BinaryDumpInFile(aFp,anObj.SaturationTA().IsInit());
    if (anObj.SaturationTA().IsInit()) BinaryDumpInFile(aFp,anObj.SaturationTA().Val());
    BinaryDumpInFile(aFp,anObj.OrthoTA().IsInit());
    if (anObj.OrthoTA().IsInit()) BinaryDumpInFile(aFp,anObj.OrthoTA().Val());
    BinaryDumpInFile(aFp,anObj.ZoomMakeMasq().IsInit());
    if (anObj.ZoomMakeMasq().IsInit()) BinaryDumpInFile(aFp,anObj.ZoomMakeMasq().Val());
    BinaryDumpInFile(aFp,anObj.LazyZoomMaskTerrain().IsInit());
    if (anObj.LazyZoomMaskTerrain().IsInit()) BinaryDumpInFile(aFp,anObj.LazyZoomMaskTerrain().Val());
    BinaryDumpInFile(aFp,anObj.MakeImCptTA().IsInit());
    if (anObj.MakeImCptTA().IsInit()) BinaryDumpInFile(aFp,anObj.MakeImCptTA().Val());
    BinaryDumpInFile(aFp,anObj.FilterTA().IsInit());
    if (anObj.FilterTA().IsInit()) BinaryDumpInFile(aFp,anObj.FilterTA().Val());
    BinaryDumpInFile(aFp,anObj.GammaVisu().IsInit());
    if (anObj.GammaVisu().IsInit()) BinaryDumpInFile(aFp,anObj.GammaVisu().Val());
    BinaryDumpInFile(aFp,anObj.ZoomVisuLiaison().IsInit());
    if (anObj.ZoomVisuLiaison().IsInit()) BinaryDumpInFile(aFp,anObj.ZoomVisuLiaison().Val());
    BinaryDumpInFile(aFp,anObj.TolerancePointHomInImage().IsInit());
    if (anObj.TolerancePointHomInImage().IsInit()) BinaryDumpInFile(aFp,anObj.TolerancePointHomInImage().Val());
    BinaryDumpInFile(aFp,anObj.FiltragePointHomInImage().IsInit());
    if (anObj.FiltragePointHomInImage().IsInit()) BinaryDumpInFile(aFp,anObj.FiltragePointHomInImage().Val());
    BinaryDumpInFile(aFp,anObj.BaseCodeRetourMicmacErreur().IsInit());
    if (anObj.BaseCodeRetourMicmacErreur().IsInit()) BinaryDumpInFile(aFp,anObj.BaseCodeRetourMicmacErreur().Val());
    BinaryDumpInFile(aFp,anObj.SuperpositionImages().IsInit());
    if (anObj.SuperpositionImages().IsInit()) BinaryDumpInFile(aFp,anObj.SuperpositionImages().Val());
}

cElXMLTree * ToXMLTree(const cSection_Results & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"Section_Results",eXMLBranche);
   if (anObj.Use_MM_EtatAvancement().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Use_MM_EtatAvancement"),anObj.Use_MM_EtatAvancement().Val())->ReTagThis("Use_MM_EtatAvancement"));
   if (anObj.DoNothingBut().IsInit())
      aRes->AddFils(ToXMLTree(anObj.DoNothingBut().Val())->ReTagThis("DoNothingBut"));
   if (anObj.Paral_Pc_IdProcess().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Paral_Pc_IdProcess"),anObj.Paral_Pc_IdProcess().Val())->ReTagThis("Paral_Pc_IdProcess"));
   if (anObj.Paral_Pc_NbProcess().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Paral_Pc_NbProcess"),anObj.Paral_Pc_NbProcess().Val())->ReTagThis("Paral_Pc_NbProcess"));
   if (anObj.X_DirPlanInterFaisceau().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("X_DirPlanInterFaisceau"),anObj.X_DirPlanInterFaisceau().Val())->ReTagThis("X_DirPlanInterFaisceau"));
   if (anObj.Y_DirPlanInterFaisceau().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Y_DirPlanInterFaisceau"),anObj.Y_DirPlanInterFaisceau().Val())->ReTagThis("Y_DirPlanInterFaisceau"));
   if (anObj.Z_DirPlanInterFaisceau().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Z_DirPlanInterFaisceau"),anObj.Z_DirPlanInterFaisceau().Val())->ReTagThis("Z_DirPlanInterFaisceau"));
   aRes->AddFils(::ToXMLTree(std::string("GeomMNT"),anObj.GeomMNT())->ReTagThis("GeomMNT"));
   if (anObj.SectionSimulation().IsInit())
      aRes->AddFils(ToXMLTree(anObj.SectionSimulation().Val())->ReTagThis("SectionSimulation"));
   if (anObj.Prio2OwnAltisolForEmprise().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Prio2OwnAltisolForEmprise"),anObj.Prio2OwnAltisolForEmprise().Val())->ReTagThis("Prio2OwnAltisolForEmprise"));
   if (anObj.AnamorphoseGeometrieMNT().IsInit())
      aRes->AddFils(ToXMLTree(anObj.AnamorphoseGeometrieMNT().Val())->ReTagThis("AnamorphoseGeometrieMNT"));
   if (anObj.RepereCorrel().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("RepereCorrel"),anObj.RepereCorrel().Val())->ReTagThis("RepereCorrel"));
   if (anObj.TagRepereCorrel().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("TagRepereCorrel"),anObj.TagRepereCorrel().Val())->ReTagThis("TagRepereCorrel"));
   if (anObj.DoMEC().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DoMEC"),anObj.DoMEC().Val())->ReTagThis("DoMEC"));
   if (anObj.NonExistingFileDoMEC().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NonExistingFileDoMEC"),anObj.NonExistingFileDoMEC().Val())->ReTagThis("NonExistingFileDoMEC"));
   if (anObj.DoFDC().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DoFDC"),anObj.DoFDC().Val())->ReTagThis("DoFDC"));
   if (anObj.GenereXMLComp().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("GenereXMLComp"),anObj.GenereXMLComp().Val())->ReTagThis("GenereXMLComp"));
   if (anObj.TAUseMasqNadirKBest().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("TAUseMasqNadirKBest"),anObj.TAUseMasqNadirKBest().Val())->ReTagThis("TAUseMasqNadirKBest"));
   if (anObj.ZoomMakeTA().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ZoomMakeTA"),anObj.ZoomMakeTA().Val())->ReTagThis("ZoomMakeTA"));
   if (anObj.SaturationTA().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SaturationTA"),anObj.SaturationTA().Val())->ReTagThis("SaturationTA"));
   if (anObj.OrthoTA().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("OrthoTA"),anObj.OrthoTA().Val())->ReTagThis("OrthoTA"));
   if (anObj.ZoomMakeMasq().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ZoomMakeMasq"),anObj.ZoomMakeMasq().Val())->ReTagThis("ZoomMakeMasq"));
   if (anObj.LazyZoomMaskTerrain().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("LazyZoomMaskTerrain"),anObj.LazyZoomMaskTerrain().Val())->ReTagThis("LazyZoomMaskTerrain"));
   if (anObj.MakeImCptTA().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("MakeImCptTA"),anObj.MakeImCptTA().Val())->ReTagThis("MakeImCptTA"));
   if (anObj.FilterTA().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("FilterTA"),anObj.FilterTA().Val())->ReTagThis("FilterTA"));
   if (anObj.GammaVisu().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("GammaVisu"),anObj.GammaVisu().Val())->ReTagThis("GammaVisu"));
   if (anObj.ZoomVisuLiaison().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ZoomVisuLiaison"),anObj.ZoomVisuLiaison().Val())->ReTagThis("ZoomVisuLiaison"));
   if (anObj.TolerancePointHomInImage().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("TolerancePointHomInImage"),anObj.TolerancePointHomInImage().Val())->ReTagThis("TolerancePointHomInImage"));
   if (anObj.FiltragePointHomInImage().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("FiltragePointHomInImage"),anObj.FiltragePointHomInImage().Val())->ReTagThis("FiltragePointHomInImage"));
   if (anObj.BaseCodeRetourMicmacErreur().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("BaseCodeRetourMicmacErreur"),anObj.BaseCodeRetourMicmacErreur().Val())->ReTagThis("BaseCodeRetourMicmacErreur"));
   if (anObj.SuperpositionImages().IsInit())
      aRes->AddFils(ToXMLTree(anObj.SuperpositionImages().Val())->ReTagThis("SuperpositionImages"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cSection_Results & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.Use_MM_EtatAvancement(),aTree->Get("Use_MM_EtatAvancement",1),bool(false)); //tototo 

   xml_init(anObj.DoNothingBut(),aTree->Get("DoNothingBut",1)); //tototo 

   xml_init(anObj.Paral_Pc_IdProcess(),aTree->Get("Paral_Pc_IdProcess",1)); //tototo 

   xml_init(anObj.Paral_Pc_NbProcess(),aTree->Get("Paral_Pc_NbProcess",1)); //tototo 

   xml_init(anObj.X_DirPlanInterFaisceau(),aTree->Get("X_DirPlanInterFaisceau",1),double(0)); //tototo 

   xml_init(anObj.Y_DirPlanInterFaisceau(),aTree->Get("Y_DirPlanInterFaisceau",1),double(0)); //tototo 

   xml_init(anObj.Z_DirPlanInterFaisceau(),aTree->Get("Z_DirPlanInterFaisceau",1),double(0)); //tototo 

   xml_init(anObj.GeomMNT(),aTree->Get("GeomMNT",1)); //tototo 

   xml_init(anObj.SectionSimulation(),aTree->Get("SectionSimulation",1)); //tototo 

   xml_init(anObj.Prio2OwnAltisolForEmprise(),aTree->Get("Prio2OwnAltisolForEmprise",1),bool(false)); //tototo 

   xml_init(anObj.AnamorphoseGeometrieMNT(),aTree->Get("AnamorphoseGeometrieMNT",1)); //tototo 

   xml_init(anObj.RepereCorrel(),aTree->Get("RepereCorrel",1)); //tototo 

   xml_init(anObj.TagRepereCorrel(),aTree->Get("TagRepereCorrel",1),std::string("RepereCartesien")); //tototo 

   xml_init(anObj.DoMEC(),aTree->Get("DoMEC",1),bool(true)); //tototo 

   xml_init(anObj.NonExistingFileDoMEC(),aTree->Get("NonExistingFileDoMEC",1)); //tototo 

   xml_init(anObj.DoFDC(),aTree->Get("DoFDC",1),bool(false)); //tototo 

   xml_init(anObj.GenereXMLComp(),aTree->Get("GenereXMLComp",1),bool(true)); //tototo 

   xml_init(anObj.TAUseMasqNadirKBest(),aTree->Get("TAUseMasqNadirKBest",1)); //tototo 

   xml_init(anObj.ZoomMakeTA(),aTree->Get("ZoomMakeTA",1)); //tototo 

   xml_init(anObj.SaturationTA(),aTree->Get("SaturationTA",1),double(50.0)); //tototo 

   xml_init(anObj.OrthoTA(),aTree->Get("OrthoTA",1),bool(false)); //tototo 

   xml_init(anObj.ZoomMakeMasq(),aTree->Get("ZoomMakeMasq",1)); //tototo 

   xml_init(anObj.LazyZoomMaskTerrain(),aTree->Get("LazyZoomMaskTerrain",1),bool(false)); //tototo 

   xml_init(anObj.MakeImCptTA(),aTree->Get("MakeImCptTA",1),bool(false)); //tototo 

   xml_init(anObj.FilterTA(),aTree->Get("FilterTA",1)); //tototo 

   xml_init(anObj.GammaVisu(),aTree->Get("GammaVisu",1),double(1.0)); //tototo 

   xml_init(anObj.ZoomVisuLiaison(),aTree->Get("ZoomVisuLiaison",1),int(-1)); //tototo 

   xml_init(anObj.TolerancePointHomInImage(),aTree->Get("TolerancePointHomInImage",1),double(0.0)); //tototo 

   xml_init(anObj.FiltragePointHomInImage(),aTree->Get("FiltragePointHomInImage",1),double(0.0)); //tototo 

   xml_init(anObj.BaseCodeRetourMicmacErreur(),aTree->Get("BaseCodeRetourMicmacErreur",1),int(100)); //tototo 

   xml_init(anObj.SuperpositionImages(),aTree->Get("SuperpositionImages",1)); //tototo 
}

std::string  Mangling( cSection_Results *) {return "86EE4C352287FAF3FD3F";};


std::string & cCalcNomChantier::PatternSelChantier()
{
   return mPatternSelChantier;
}

const std::string & cCalcNomChantier::PatternSelChantier()const 
{
   return mPatternSelChantier;
}


std::string & cCalcNomChantier::PatNameChantier()
{
   return mPatNameChantier;
}

const std::string & cCalcNomChantier::PatNameChantier()const 
{
   return mPatNameChantier;
}


cTplValGesInit< std::string > & cCalcNomChantier::SeparateurChantier()
{
   return mSeparateurChantier;
}

const cTplValGesInit< std::string > & cCalcNomChantier::SeparateurChantier()const 
{
   return mSeparateurChantier;
}

void  BinaryUnDumpFromFile(cCalcNomChantier & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.PatternSelChantier(),aFp);
    BinaryUnDumpFromFile(anObj.PatNameChantier(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SeparateurChantier().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SeparateurChantier().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SeparateurChantier().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cCalcNomChantier & anObj)
{
    BinaryDumpInFile(aFp,anObj.PatternSelChantier());
    BinaryDumpInFile(aFp,anObj.PatNameChantier());
    BinaryDumpInFile(aFp,anObj.SeparateurChantier().IsInit());
    if (anObj.SeparateurChantier().IsInit()) BinaryDumpInFile(aFp,anObj.SeparateurChantier().Val());
}

cElXMLTree * ToXMLTree(const cCalcNomChantier & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"CalcNomChantier",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("PatternSelChantier"),anObj.PatternSelChantier())->ReTagThis("PatternSelChantier"));
   aRes->AddFils(::ToXMLTree(std::string("PatNameChantier"),anObj.PatNameChantier())->ReTagThis("PatNameChantier"));
   if (anObj.SeparateurChantier().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SeparateurChantier"),anObj.SeparateurChantier().Val())->ReTagThis("SeparateurChantier"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cCalcNomChantier & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.PatternSelChantier(),aTree->Get("PatternSelChantier",1)); //tototo 

   xml_init(anObj.PatNameChantier(),aTree->Get("PatNameChantier",1)); //tototo 

   xml_init(anObj.SeparateurChantier(),aTree->Get("SeparateurChantier",1),std::string("")); //tototo 
}

std::string  Mangling( cCalcNomChantier *) {return "0061C18B9E16DDE0FA3F";};


std::string & cPurgeFiles::PatternSelPurge()
{
   return mPatternSelPurge;
}

const std::string & cPurgeFiles::PatternSelPurge()const 
{
   return mPatternSelPurge;
}


bool & cPurgeFiles::PurgeToSupress()
{
   return mPurgeToSupress;
}

const bool & cPurgeFiles::PurgeToSupress()const 
{
   return mPurgeToSupress;
}

void  BinaryUnDumpFromFile(cPurgeFiles & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.PatternSelPurge(),aFp);
    BinaryUnDumpFromFile(anObj.PurgeToSupress(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cPurgeFiles & anObj)
{
    BinaryDumpInFile(aFp,anObj.PatternSelPurge());
    BinaryDumpInFile(aFp,anObj.PurgeToSupress());
}

cElXMLTree * ToXMLTree(const cPurgeFiles & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"PurgeFiles",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("PatternSelPurge"),anObj.PatternSelPurge())->ReTagThis("PatternSelPurge"));
   aRes->AddFils(::ToXMLTree(std::string("PurgeToSupress"),anObj.PurgeToSupress())->ReTagThis("PurgeToSupress"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cPurgeFiles & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.PatternSelPurge(),aTree->Get("PatternSelPurge",1)); //tototo 

   xml_init(anObj.PurgeToSupress(),aTree->Get("PurgeToSupress",1)); //tototo 
}

std::string  Mangling( cPurgeFiles *) {return "042C166DA63CEA9AFE3F";};


cTplValGesInit< std::string > & cSection_WorkSpace::FileExportApero2MM()
{
   return mFileExportApero2MM;
}

const cTplValGesInit< std::string > & cSection_WorkSpace::FileExportApero2MM()const 
{
   return mFileExportApero2MM;
}


cTplValGesInit< bool > & cSection_WorkSpace::UseProfInVertLoc()
{
   return mUseProfInVertLoc;
}

const cTplValGesInit< bool > & cSection_WorkSpace::UseProfInVertLoc()const 
{
   return mUseProfInVertLoc;
}


cTplValGesInit< std::string > & cSection_WorkSpace::NameFileParamMICMAC()
{
   return mNameFileParamMICMAC;
}

const cTplValGesInit< std::string > & cSection_WorkSpace::NameFileParamMICMAC()const 
{
   return mNameFileParamMICMAC;
}


std::string & cSection_WorkSpace::WorkDir()
{
   return mWorkDir;
}

const std::string & cSection_WorkSpace::WorkDir()const 
{
   return mWorkDir;
}


cTplValGesInit< std::string > & cSection_WorkSpace::DirImagesOri()
{
   return mDirImagesOri;
}

const cTplValGesInit< std::string > & cSection_WorkSpace::DirImagesOri()const 
{
   return mDirImagesOri;
}


std::string & cSection_WorkSpace::TmpMEC()
{
   return mTmpMEC;
}

const std::string & cSection_WorkSpace::TmpMEC()const 
{
   return mTmpMEC;
}


cTplValGesInit< std::string > & cSection_WorkSpace::TmpPyr()
{
   return mTmpPyr;
}

const cTplValGesInit< std::string > & cSection_WorkSpace::TmpPyr()const 
{
   return mTmpPyr;
}


cTplValGesInit< std::string > & cSection_WorkSpace::TmpGeom()
{
   return mTmpGeom;
}

const cTplValGesInit< std::string > & cSection_WorkSpace::TmpGeom()const 
{
   return mTmpGeom;
}


cTplValGesInit< std::string > & cSection_WorkSpace::TmpResult()
{
   return mTmpResult;
}

const cTplValGesInit< std::string > & cSection_WorkSpace::TmpResult()const 
{
   return mTmpResult;
}


cTplValGesInit< bool > & cSection_WorkSpace::CalledByProcess()
{
   return mCalledByProcess;
}

const cTplValGesInit< bool > & cSection_WorkSpace::CalledByProcess()const 
{
   return mCalledByProcess;
}


cTplValGesInit< int > & cSection_WorkSpace::IdMasterProcess()
{
   return mIdMasterProcess;
}

const cTplValGesInit< int > & cSection_WorkSpace::IdMasterProcess()const 
{
   return mIdMasterProcess;
}


cTplValGesInit< bool > & cSection_WorkSpace::CreateGrayFileAtBegin()
{
   return mCreateGrayFileAtBegin;
}

const cTplValGesInit< bool > & cSection_WorkSpace::CreateGrayFileAtBegin()const 
{
   return mCreateGrayFileAtBegin;
}


cTplValGesInit< bool > & cSection_WorkSpace::Visu()
{
   return mVisu;
}

const cTplValGesInit< bool > & cSection_WorkSpace::Visu()const 
{
   return mVisu;
}


cTplValGesInit< int > & cSection_WorkSpace::ByProcess()
{
   return mByProcess;
}

const cTplValGesInit< int > & cSection_WorkSpace::ByProcess()const 
{
   return mByProcess;
}


cTplValGesInit< bool > & cSection_WorkSpace::StopOnEchecFils()
{
   return mStopOnEchecFils;
}

const cTplValGesInit< bool > & cSection_WorkSpace::StopOnEchecFils()const 
{
   return mStopOnEchecFils;
}


cTplValGesInit< int > & cSection_WorkSpace::AvalaibleMemory()
{
   return mAvalaibleMemory;
}

const cTplValGesInit< int > & cSection_WorkSpace::AvalaibleMemory()const 
{
   return mAvalaibleMemory;
}


cTplValGesInit< int > & cSection_WorkSpace::SzRecouvrtDalles()
{
   return mSzRecouvrtDalles;
}

const cTplValGesInit< int > & cSection_WorkSpace::SzRecouvrtDalles()const 
{
   return mSzRecouvrtDalles;
}


cTplValGesInit< int > & cSection_WorkSpace::SzDalleMin()
{
   return mSzDalleMin;
}

const cTplValGesInit< int > & cSection_WorkSpace::SzDalleMin()const 
{
   return mSzDalleMin;
}


cTplValGesInit< int > & cSection_WorkSpace::SzDalleMax()
{
   return mSzDalleMax;
}

const cTplValGesInit< int > & cSection_WorkSpace::SzDalleMax()const 
{
   return mSzDalleMax;
}


cTplValGesInit< double > & cSection_WorkSpace::NbCelluleMax()
{
   return mNbCelluleMax;
}

const cTplValGesInit< double > & cSection_WorkSpace::NbCelluleMax()const 
{
   return mNbCelluleMax;
}


cTplValGesInit< int > & cSection_WorkSpace::SzMinDecomposCalc()
{
   return mSzMinDecomposCalc;
}

const cTplValGesInit< int > & cSection_WorkSpace::SzMinDecomposCalc()const 
{
   return mSzMinDecomposCalc;
}


cTplValGesInit< bool > & cSection_WorkSpace::AutorizeSplitRec()
{
   return mAutorizeSplitRec;
}

const cTplValGesInit< bool > & cSection_WorkSpace::AutorizeSplitRec()const 
{
   return mAutorizeSplitRec;
}


cTplValGesInit< int > & cSection_WorkSpace::DefTileFile()
{
   return mDefTileFile;
}

const cTplValGesInit< int > & cSection_WorkSpace::DefTileFile()const 
{
   return mDefTileFile;
}


cTplValGesInit< double > & cSection_WorkSpace::NbPixDefFilesAux()
{
   return mNbPixDefFilesAux;
}

const cTplValGesInit< double > & cSection_WorkSpace::NbPixDefFilesAux()const 
{
   return mNbPixDefFilesAux;
}


cTplValGesInit< int > & cSection_WorkSpace::DeZoomDefMinFileAux()
{
   return mDeZoomDefMinFileAux;
}

const cTplValGesInit< int > & cSection_WorkSpace::DeZoomDefMinFileAux()const 
{
   return mDeZoomDefMinFileAux;
}


cTplValGesInit< int > & cSection_WorkSpace::FirstEtapeMEC()
{
   return mFirstEtapeMEC;
}

const cTplValGesInit< int > & cSection_WorkSpace::FirstEtapeMEC()const 
{
   return mFirstEtapeMEC;
}


cTplValGesInit< int > & cSection_WorkSpace::LastEtapeMEC()
{
   return mLastEtapeMEC;
}

const cTplValGesInit< int > & cSection_WorkSpace::LastEtapeMEC()const 
{
   return mLastEtapeMEC;
}


cTplValGesInit< int > & cSection_WorkSpace::FirstBoiteMEC()
{
   return mFirstBoiteMEC;
}

const cTplValGesInit< int > & cSection_WorkSpace::FirstBoiteMEC()const 
{
   return mFirstBoiteMEC;
}


cTplValGesInit< int > & cSection_WorkSpace::NbBoitesMEC()
{
   return mNbBoitesMEC;
}

const cTplValGesInit< int > & cSection_WorkSpace::NbBoitesMEC()const 
{
   return mNbBoitesMEC;
}


cTplValGesInit< std::string > & cSection_WorkSpace::NomChantier()
{
   return mNomChantier;
}

const cTplValGesInit< std::string > & cSection_WorkSpace::NomChantier()const 
{
   return mNomChantier;
}


std::string & cSection_WorkSpace::PatternSelChantier()
{
   return CalcNomChantier().Val().PatternSelChantier();
}

const std::string & cSection_WorkSpace::PatternSelChantier()const 
{
   return CalcNomChantier().Val().PatternSelChantier();
}


std::string & cSection_WorkSpace::PatNameChantier()
{
   return CalcNomChantier().Val().PatNameChantier();
}

const std::string & cSection_WorkSpace::PatNameChantier()const 
{
   return CalcNomChantier().Val().PatNameChantier();
}


cTplValGesInit< std::string > & cSection_WorkSpace::SeparateurChantier()
{
   return CalcNomChantier().Val().SeparateurChantier();
}

const cTplValGesInit< std::string > & cSection_WorkSpace::SeparateurChantier()const 
{
   return CalcNomChantier().Val().SeparateurChantier();
}


cTplValGesInit< cCalcNomChantier > & cSection_WorkSpace::CalcNomChantier()
{
   return mCalcNomChantier;
}

const cTplValGesInit< cCalcNomChantier > & cSection_WorkSpace::CalcNomChantier()const 
{
   return mCalcNomChantier;
}


cTplValGesInit< std::string > & cSection_WorkSpace::PatternSelPyr()
{
   return mPatternSelPyr;
}

const cTplValGesInit< std::string > & cSection_WorkSpace::PatternSelPyr()const 
{
   return mPatternSelPyr;
}


cTplValGesInit< std::string > & cSection_WorkSpace::PatternNomPyr()
{
   return mPatternNomPyr;
}

const cTplValGesInit< std::string > & cSection_WorkSpace::PatternNomPyr()const 
{
   return mPatternNomPyr;
}


cTplValGesInit< std::string > & cSection_WorkSpace::SeparateurPyr()
{
   return mSeparateurPyr;
}

const cTplValGesInit< std::string > & cSection_WorkSpace::SeparateurPyr()const 
{
   return mSeparateurPyr;
}


cTplValGesInit< std::string > & cSection_WorkSpace::KeyCalNamePyr()
{
   return mKeyCalNamePyr;
}

const cTplValGesInit< std::string > & cSection_WorkSpace::KeyCalNamePyr()const 
{
   return mKeyCalNamePyr;
}


cTplValGesInit< bool > & cSection_WorkSpace::ActivePurge()
{
   return mActivePurge;
}

const cTplValGesInit< bool > & cSection_WorkSpace::ActivePurge()const 
{
   return mActivePurge;
}


std::list< cPurgeFiles > & cSection_WorkSpace::PurgeFiles()
{
   return mPurgeFiles;
}

const std::list< cPurgeFiles > & cSection_WorkSpace::PurgeFiles()const 
{
   return mPurgeFiles;
}


cTplValGesInit< bool > & cSection_WorkSpace::PurgeMECResultBefore()
{
   return mPurgeMECResultBefore;
}

const cTplValGesInit< bool > & cSection_WorkSpace::PurgeMECResultBefore()const 
{
   return mPurgeMECResultBefore;
}


cTplValGesInit< std::string > & cSection_WorkSpace::PreservedFile()
{
   return mPreservedFile;
}

const cTplValGesInit< std::string > & cSection_WorkSpace::PreservedFile()const 
{
   return mPreservedFile;
}


cTplValGesInit< bool > & cSection_WorkSpace::UseChantierNameDescripteur()
{
   return mUseChantierNameDescripteur;
}

const cTplValGesInit< bool > & cSection_WorkSpace::UseChantierNameDescripteur()const 
{
   return mUseChantierNameDescripteur;
}


cTplValGesInit< string > & cSection_WorkSpace::FileChantierNameDescripteur()
{
   return mFileChantierNameDescripteur;
}

const cTplValGesInit< string > & cSection_WorkSpace::FileChantierNameDescripteur()const 
{
   return mFileChantierNameDescripteur;
}


cTplValGesInit< cCmdMappeur > & cSection_WorkSpace::MapMicMac()
{
   return mMapMicMac;
}

const cTplValGesInit< cCmdMappeur > & cSection_WorkSpace::MapMicMac()const 
{
   return mMapMicMac;
}


cTplValGesInit< cCmdExePar > & cSection_WorkSpace::PostProcess()
{
   return mPostProcess;
}

const cTplValGesInit< cCmdExePar > & cSection_WorkSpace::PostProcess()const 
{
   return mPostProcess;
}


cTplValGesInit< eComprTiff > & cSection_WorkSpace::ComprMasque()
{
   return mComprMasque;
}

const cTplValGesInit< eComprTiff > & cSection_WorkSpace::ComprMasque()const 
{
   return mComprMasque;
}


cTplValGesInit< eTypeNumerique > & cSection_WorkSpace::TypeMasque()
{
   return mTypeMasque;
}

const cTplValGesInit< eTypeNumerique > & cSection_WorkSpace::TypeMasque()const 
{
   return mTypeMasque;
}

void  BinaryUnDumpFromFile(cSection_WorkSpace & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.FileExportApero2MM().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.FileExportApero2MM().ValForcedForUnUmp(),aFp);
        }
        else  anObj.FileExportApero2MM().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.UseProfInVertLoc().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.UseProfInVertLoc().ValForcedForUnUmp(),aFp);
        }
        else  anObj.UseProfInVertLoc().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NameFileParamMICMAC().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NameFileParamMICMAC().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NameFileParamMICMAC().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.WorkDir(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.DirImagesOri().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.DirImagesOri().ValForcedForUnUmp(),aFp);
        }
        else  anObj.DirImagesOri().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.TmpMEC(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.TmpPyr().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.TmpPyr().ValForcedForUnUmp(),aFp);
        }
        else  anObj.TmpPyr().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.TmpGeom().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.TmpGeom().ValForcedForUnUmp(),aFp);
        }
        else  anObj.TmpGeom().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.TmpResult().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.TmpResult().ValForcedForUnUmp(),aFp);
        }
        else  anObj.TmpResult().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.CalledByProcess().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.CalledByProcess().ValForcedForUnUmp(),aFp);
        }
        else  anObj.CalledByProcess().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.IdMasterProcess().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.IdMasterProcess().ValForcedForUnUmp(),aFp);
        }
        else  anObj.IdMasterProcess().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.CreateGrayFileAtBegin().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.CreateGrayFileAtBegin().ValForcedForUnUmp(),aFp);
        }
        else  anObj.CreateGrayFileAtBegin().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Visu().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Visu().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Visu().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ByProcess().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ByProcess().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ByProcess().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.StopOnEchecFils().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.StopOnEchecFils().ValForcedForUnUmp(),aFp);
        }
        else  anObj.StopOnEchecFils().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.AvalaibleMemory().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.AvalaibleMemory().ValForcedForUnUmp(),aFp);
        }
        else  anObj.AvalaibleMemory().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SzRecouvrtDalles().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SzRecouvrtDalles().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SzRecouvrtDalles().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SzDalleMin().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SzDalleMin().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SzDalleMin().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SzDalleMax().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SzDalleMax().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SzDalleMax().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NbCelluleMax().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NbCelluleMax().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NbCelluleMax().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SzMinDecomposCalc().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SzMinDecomposCalc().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SzMinDecomposCalc().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.AutorizeSplitRec().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.AutorizeSplitRec().ValForcedForUnUmp(),aFp);
        }
        else  anObj.AutorizeSplitRec().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.DefTileFile().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.DefTileFile().ValForcedForUnUmp(),aFp);
        }
        else  anObj.DefTileFile().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NbPixDefFilesAux().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NbPixDefFilesAux().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NbPixDefFilesAux().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.DeZoomDefMinFileAux().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.DeZoomDefMinFileAux().ValForcedForUnUmp(),aFp);
        }
        else  anObj.DeZoomDefMinFileAux().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.FirstEtapeMEC().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.FirstEtapeMEC().ValForcedForUnUmp(),aFp);
        }
        else  anObj.FirstEtapeMEC().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.LastEtapeMEC().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.LastEtapeMEC().ValForcedForUnUmp(),aFp);
        }
        else  anObj.LastEtapeMEC().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.FirstBoiteMEC().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.FirstBoiteMEC().ValForcedForUnUmp(),aFp);
        }
        else  anObj.FirstBoiteMEC().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NbBoitesMEC().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NbBoitesMEC().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NbBoitesMEC().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NomChantier().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NomChantier().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NomChantier().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.CalcNomChantier().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.CalcNomChantier().ValForcedForUnUmp(),aFp);
        }
        else  anObj.CalcNomChantier().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.PatternSelPyr().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.PatternSelPyr().ValForcedForUnUmp(),aFp);
        }
        else  anObj.PatternSelPyr().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.PatternNomPyr().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.PatternNomPyr().ValForcedForUnUmp(),aFp);
        }
        else  anObj.PatternNomPyr().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SeparateurPyr().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SeparateurPyr().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SeparateurPyr().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.KeyCalNamePyr().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.KeyCalNamePyr().ValForcedForUnUmp(),aFp);
        }
        else  anObj.KeyCalNamePyr().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ActivePurge().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ActivePurge().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ActivePurge().SetNoInit();
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cPurgeFiles aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.PurgeFiles().push_back(aVal);
        }
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.PurgeMECResultBefore().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.PurgeMECResultBefore().ValForcedForUnUmp(),aFp);
        }
        else  anObj.PurgeMECResultBefore().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.PreservedFile().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.PreservedFile().ValForcedForUnUmp(),aFp);
        }
        else  anObj.PreservedFile().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.UseChantierNameDescripteur().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.UseChantierNameDescripteur().ValForcedForUnUmp(),aFp);
        }
        else  anObj.UseChantierNameDescripteur().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.FileChantierNameDescripteur().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.FileChantierNameDescripteur().ValForcedForUnUmp(),aFp);
        }
        else  anObj.FileChantierNameDescripteur().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.MapMicMac().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.MapMicMac().ValForcedForUnUmp(),aFp);
        }
        else  anObj.MapMicMac().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.PostProcess().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.PostProcess().ValForcedForUnUmp(),aFp);
        }
        else  anObj.PostProcess().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ComprMasque().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ComprMasque().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ComprMasque().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.TypeMasque().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.TypeMasque().ValForcedForUnUmp(),aFp);
        }
        else  anObj.TypeMasque().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cSection_WorkSpace & anObj)
{
    BinaryDumpInFile(aFp,anObj.FileExportApero2MM().IsInit());
    if (anObj.FileExportApero2MM().IsInit()) BinaryDumpInFile(aFp,anObj.FileExportApero2MM().Val());
    BinaryDumpInFile(aFp,anObj.UseProfInVertLoc().IsInit());
    if (anObj.UseProfInVertLoc().IsInit()) BinaryDumpInFile(aFp,anObj.UseProfInVertLoc().Val());
    BinaryDumpInFile(aFp,anObj.NameFileParamMICMAC().IsInit());
    if (anObj.NameFileParamMICMAC().IsInit()) BinaryDumpInFile(aFp,anObj.NameFileParamMICMAC().Val());
    BinaryDumpInFile(aFp,anObj.WorkDir());
    BinaryDumpInFile(aFp,anObj.DirImagesOri().IsInit());
    if (anObj.DirImagesOri().IsInit()) BinaryDumpInFile(aFp,anObj.DirImagesOri().Val());
    BinaryDumpInFile(aFp,anObj.TmpMEC());
    BinaryDumpInFile(aFp,anObj.TmpPyr().IsInit());
    if (anObj.TmpPyr().IsInit()) BinaryDumpInFile(aFp,anObj.TmpPyr().Val());
    BinaryDumpInFile(aFp,anObj.TmpGeom().IsInit());
    if (anObj.TmpGeom().IsInit()) BinaryDumpInFile(aFp,anObj.TmpGeom().Val());
    BinaryDumpInFile(aFp,anObj.TmpResult().IsInit());
    if (anObj.TmpResult().IsInit()) BinaryDumpInFile(aFp,anObj.TmpResult().Val());
    BinaryDumpInFile(aFp,anObj.CalledByProcess().IsInit());
    if (anObj.CalledByProcess().IsInit()) BinaryDumpInFile(aFp,anObj.CalledByProcess().Val());
    BinaryDumpInFile(aFp,anObj.IdMasterProcess().IsInit());
    if (anObj.IdMasterProcess().IsInit()) BinaryDumpInFile(aFp,anObj.IdMasterProcess().Val());
    BinaryDumpInFile(aFp,anObj.CreateGrayFileAtBegin().IsInit());
    if (anObj.CreateGrayFileAtBegin().IsInit()) BinaryDumpInFile(aFp,anObj.CreateGrayFileAtBegin().Val());
    BinaryDumpInFile(aFp,anObj.Visu().IsInit());
    if (anObj.Visu().IsInit()) BinaryDumpInFile(aFp,anObj.Visu().Val());
    BinaryDumpInFile(aFp,anObj.ByProcess().IsInit());
    if (anObj.ByProcess().IsInit()) BinaryDumpInFile(aFp,anObj.ByProcess().Val());
    BinaryDumpInFile(aFp,anObj.StopOnEchecFils().IsInit());
    if (anObj.StopOnEchecFils().IsInit()) BinaryDumpInFile(aFp,anObj.StopOnEchecFils().Val());
    BinaryDumpInFile(aFp,anObj.AvalaibleMemory().IsInit());
    if (anObj.AvalaibleMemory().IsInit()) BinaryDumpInFile(aFp,anObj.AvalaibleMemory().Val());
    BinaryDumpInFile(aFp,anObj.SzRecouvrtDalles().IsInit());
    if (anObj.SzRecouvrtDalles().IsInit()) BinaryDumpInFile(aFp,anObj.SzRecouvrtDalles().Val());
    BinaryDumpInFile(aFp,anObj.SzDalleMin().IsInit());
    if (anObj.SzDalleMin().IsInit()) BinaryDumpInFile(aFp,anObj.SzDalleMin().Val());
    BinaryDumpInFile(aFp,anObj.SzDalleMax().IsInit());
    if (anObj.SzDalleMax().IsInit()) BinaryDumpInFile(aFp,anObj.SzDalleMax().Val());
    BinaryDumpInFile(aFp,anObj.NbCelluleMax().IsInit());
    if (anObj.NbCelluleMax().IsInit()) BinaryDumpInFile(aFp,anObj.NbCelluleMax().Val());
    BinaryDumpInFile(aFp,anObj.SzMinDecomposCalc().IsInit());
    if (anObj.SzMinDecomposCalc().IsInit()) BinaryDumpInFile(aFp,anObj.SzMinDecomposCalc().Val());
    BinaryDumpInFile(aFp,anObj.AutorizeSplitRec().IsInit());
    if (anObj.AutorizeSplitRec().IsInit()) BinaryDumpInFile(aFp,anObj.AutorizeSplitRec().Val());
    BinaryDumpInFile(aFp,anObj.DefTileFile().IsInit());
    if (anObj.DefTileFile().IsInit()) BinaryDumpInFile(aFp,anObj.DefTileFile().Val());
    BinaryDumpInFile(aFp,anObj.NbPixDefFilesAux().IsInit());
    if (anObj.NbPixDefFilesAux().IsInit()) BinaryDumpInFile(aFp,anObj.NbPixDefFilesAux().Val());
    BinaryDumpInFile(aFp,anObj.DeZoomDefMinFileAux().IsInit());
    if (anObj.DeZoomDefMinFileAux().IsInit()) BinaryDumpInFile(aFp,anObj.DeZoomDefMinFileAux().Val());
    BinaryDumpInFile(aFp,anObj.FirstEtapeMEC().IsInit());
    if (anObj.FirstEtapeMEC().IsInit()) BinaryDumpInFile(aFp,anObj.FirstEtapeMEC().Val());
    BinaryDumpInFile(aFp,anObj.LastEtapeMEC().IsInit());
    if (anObj.LastEtapeMEC().IsInit()) BinaryDumpInFile(aFp,anObj.LastEtapeMEC().Val());
    BinaryDumpInFile(aFp,anObj.FirstBoiteMEC().IsInit());
    if (anObj.FirstBoiteMEC().IsInit()) BinaryDumpInFile(aFp,anObj.FirstBoiteMEC().Val());
    BinaryDumpInFile(aFp,anObj.NbBoitesMEC().IsInit());
    if (anObj.NbBoitesMEC().IsInit()) BinaryDumpInFile(aFp,anObj.NbBoitesMEC().Val());
    BinaryDumpInFile(aFp,anObj.NomChantier().IsInit());
    if (anObj.NomChantier().IsInit()) BinaryDumpInFile(aFp,anObj.NomChantier().Val());
    BinaryDumpInFile(aFp,anObj.CalcNomChantier().IsInit());
    if (anObj.CalcNomChantier().IsInit()) BinaryDumpInFile(aFp,anObj.CalcNomChantier().Val());
    BinaryDumpInFile(aFp,anObj.PatternSelPyr().IsInit());
    if (anObj.PatternSelPyr().IsInit()) BinaryDumpInFile(aFp,anObj.PatternSelPyr().Val());
    BinaryDumpInFile(aFp,anObj.PatternNomPyr().IsInit());
    if (anObj.PatternNomPyr().IsInit()) BinaryDumpInFile(aFp,anObj.PatternNomPyr().Val());
    BinaryDumpInFile(aFp,anObj.SeparateurPyr().IsInit());
    if (anObj.SeparateurPyr().IsInit()) BinaryDumpInFile(aFp,anObj.SeparateurPyr().Val());
    BinaryDumpInFile(aFp,anObj.KeyCalNamePyr().IsInit());
    if (anObj.KeyCalNamePyr().IsInit()) BinaryDumpInFile(aFp,anObj.KeyCalNamePyr().Val());
    BinaryDumpInFile(aFp,anObj.ActivePurge().IsInit());
    if (anObj.ActivePurge().IsInit()) BinaryDumpInFile(aFp,anObj.ActivePurge().Val());
    BinaryDumpInFile(aFp,(int)anObj.PurgeFiles().size());
    for(  std::list< cPurgeFiles >::const_iterator iT=anObj.PurgeFiles().begin();
         iT!=anObj.PurgeFiles().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.PurgeMECResultBefore().IsInit());
    if (anObj.PurgeMECResultBefore().IsInit()) BinaryDumpInFile(aFp,anObj.PurgeMECResultBefore().Val());
    BinaryDumpInFile(aFp,anObj.PreservedFile().IsInit());
    if (anObj.PreservedFile().IsInit()) BinaryDumpInFile(aFp,anObj.PreservedFile().Val());
    BinaryDumpInFile(aFp,anObj.UseChantierNameDescripteur().IsInit());
    if (anObj.UseChantierNameDescripteur().IsInit()) BinaryDumpInFile(aFp,anObj.UseChantierNameDescripteur().Val());
    BinaryDumpInFile(aFp,anObj.FileChantierNameDescripteur().IsInit());
    if (anObj.FileChantierNameDescripteur().IsInit()) BinaryDumpInFile(aFp,anObj.FileChantierNameDescripteur().Val());
    BinaryDumpInFile(aFp,anObj.MapMicMac().IsInit());
    if (anObj.MapMicMac().IsInit()) BinaryDumpInFile(aFp,anObj.MapMicMac().Val());
    BinaryDumpInFile(aFp,anObj.PostProcess().IsInit());
    if (anObj.PostProcess().IsInit()) BinaryDumpInFile(aFp,anObj.PostProcess().Val());
    BinaryDumpInFile(aFp,anObj.ComprMasque().IsInit());
    if (anObj.ComprMasque().IsInit()) BinaryDumpInFile(aFp,anObj.ComprMasque().Val());
    BinaryDumpInFile(aFp,anObj.TypeMasque().IsInit());
    if (anObj.TypeMasque().IsInit()) BinaryDumpInFile(aFp,anObj.TypeMasque().Val());
}

cElXMLTree * ToXMLTree(const cSection_WorkSpace & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"Section_WorkSpace",eXMLBranche);
   if (anObj.FileExportApero2MM().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("FileExportApero2MM"),anObj.FileExportApero2MM().Val())->ReTagThis("FileExportApero2MM"));
   if (anObj.UseProfInVertLoc().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("UseProfInVertLoc"),anObj.UseProfInVertLoc().Val())->ReTagThis("UseProfInVertLoc"));
   if (anObj.NameFileParamMICMAC().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NameFileParamMICMAC"),anObj.NameFileParamMICMAC().Val())->ReTagThis("NameFileParamMICMAC"));
   aRes->AddFils(::ToXMLTree(std::string("WorkDir"),anObj.WorkDir())->ReTagThis("WorkDir"));
   if (anObj.DirImagesOri().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DirImagesOri"),anObj.DirImagesOri().Val())->ReTagThis("DirImagesOri"));
   aRes->AddFils(::ToXMLTree(std::string("TmpMEC"),anObj.TmpMEC())->ReTagThis("TmpMEC"));
   if (anObj.TmpPyr().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("TmpPyr"),anObj.TmpPyr().Val())->ReTagThis("TmpPyr"));
   if (anObj.TmpGeom().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("TmpGeom"),anObj.TmpGeom().Val())->ReTagThis("TmpGeom"));
   if (anObj.TmpResult().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("TmpResult"),anObj.TmpResult().Val())->ReTagThis("TmpResult"));
   if (anObj.CalledByProcess().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("CalledByProcess"),anObj.CalledByProcess().Val())->ReTagThis("CalledByProcess"));
   if (anObj.IdMasterProcess().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("IdMasterProcess"),anObj.IdMasterProcess().Val())->ReTagThis("IdMasterProcess"));
   if (anObj.CreateGrayFileAtBegin().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("CreateGrayFileAtBegin"),anObj.CreateGrayFileAtBegin().Val())->ReTagThis("CreateGrayFileAtBegin"));
   if (anObj.Visu().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Visu"),anObj.Visu().Val())->ReTagThis("Visu"));
   if (anObj.ByProcess().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ByProcess"),anObj.ByProcess().Val())->ReTagThis("ByProcess"));
   if (anObj.StopOnEchecFils().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("StopOnEchecFils"),anObj.StopOnEchecFils().Val())->ReTagThis("StopOnEchecFils"));
   if (anObj.AvalaibleMemory().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("AvalaibleMemory"),anObj.AvalaibleMemory().Val())->ReTagThis("AvalaibleMemory"));
   if (anObj.SzRecouvrtDalles().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SzRecouvrtDalles"),anObj.SzRecouvrtDalles().Val())->ReTagThis("SzRecouvrtDalles"));
   if (anObj.SzDalleMin().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SzDalleMin"),anObj.SzDalleMin().Val())->ReTagThis("SzDalleMin"));
   if (anObj.SzDalleMax().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SzDalleMax"),anObj.SzDalleMax().Val())->ReTagThis("SzDalleMax"));
   if (anObj.NbCelluleMax().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NbCelluleMax"),anObj.NbCelluleMax().Val())->ReTagThis("NbCelluleMax"));
   if (anObj.SzMinDecomposCalc().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SzMinDecomposCalc"),anObj.SzMinDecomposCalc().Val())->ReTagThis("SzMinDecomposCalc"));
   if (anObj.AutorizeSplitRec().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("AutorizeSplitRec"),anObj.AutorizeSplitRec().Val())->ReTagThis("AutorizeSplitRec"));
   if (anObj.DefTileFile().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DefTileFile"),anObj.DefTileFile().Val())->ReTagThis("DefTileFile"));
   if (anObj.NbPixDefFilesAux().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NbPixDefFilesAux"),anObj.NbPixDefFilesAux().Val())->ReTagThis("NbPixDefFilesAux"));
   if (anObj.DeZoomDefMinFileAux().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DeZoomDefMinFileAux"),anObj.DeZoomDefMinFileAux().Val())->ReTagThis("DeZoomDefMinFileAux"));
   if (anObj.FirstEtapeMEC().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("FirstEtapeMEC"),anObj.FirstEtapeMEC().Val())->ReTagThis("FirstEtapeMEC"));
   if (anObj.LastEtapeMEC().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("LastEtapeMEC"),anObj.LastEtapeMEC().Val())->ReTagThis("LastEtapeMEC"));
   if (anObj.FirstBoiteMEC().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("FirstBoiteMEC"),anObj.FirstBoiteMEC().Val())->ReTagThis("FirstBoiteMEC"));
   if (anObj.NbBoitesMEC().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NbBoitesMEC"),anObj.NbBoitesMEC().Val())->ReTagThis("NbBoitesMEC"));
   if (anObj.NomChantier().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NomChantier"),anObj.NomChantier().Val())->ReTagThis("NomChantier"));
   if (anObj.CalcNomChantier().IsInit())
      aRes->AddFils(ToXMLTree(anObj.CalcNomChantier().Val())->ReTagThis("CalcNomChantier"));
   if (anObj.PatternSelPyr().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PatternSelPyr"),anObj.PatternSelPyr().Val())->ReTagThis("PatternSelPyr"));
   if (anObj.PatternNomPyr().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PatternNomPyr"),anObj.PatternNomPyr().Val())->ReTagThis("PatternNomPyr"));
   if (anObj.SeparateurPyr().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SeparateurPyr"),anObj.SeparateurPyr().Val())->ReTagThis("SeparateurPyr"));
   if (anObj.KeyCalNamePyr().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("KeyCalNamePyr"),anObj.KeyCalNamePyr().Val())->ReTagThis("KeyCalNamePyr"));
   if (anObj.ActivePurge().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ActivePurge"),anObj.ActivePurge().Val())->ReTagThis("ActivePurge"));
  for
  (       std::list< cPurgeFiles >::const_iterator it=anObj.PurgeFiles().begin();
      it !=anObj.PurgeFiles().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("PurgeFiles"));
   if (anObj.PurgeMECResultBefore().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PurgeMECResultBefore"),anObj.PurgeMECResultBefore().Val())->ReTagThis("PurgeMECResultBefore"));
   if (anObj.PreservedFile().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PreservedFile"),anObj.PreservedFile().Val())->ReTagThis("PreservedFile"));
   if (anObj.UseChantierNameDescripteur().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("UseChantierNameDescripteur"),anObj.UseChantierNameDescripteur().Val())->ReTagThis("UseChantierNameDescripteur"));
   if (anObj.FileChantierNameDescripteur().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("FileChantierNameDescripteur"),anObj.FileChantierNameDescripteur().Val())->ReTagThis("FileChantierNameDescripteur"));
   if (anObj.MapMicMac().IsInit())
      aRes->AddFils(ToXMLTree(anObj.MapMicMac().Val())->ReTagThis("MapMicMac"));
   if (anObj.PostProcess().IsInit())
      aRes->AddFils(ToXMLTree(anObj.PostProcess().Val())->ReTagThis("PostProcess"));
   if (anObj.ComprMasque().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ComprMasque"),anObj.ComprMasque().Val())->ReTagThis("ComprMasque"));
   if (anObj.TypeMasque().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("TypeMasque"),anObj.TypeMasque().Val())->ReTagThis("TypeMasque"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cSection_WorkSpace & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.FileExportApero2MM(),aTree->Get("FileExportApero2MM",1)); //tototo 

   xml_init(anObj.UseProfInVertLoc(),aTree->Get("UseProfInVertLoc",1),bool(true)); //tototo 

   xml_init(anObj.NameFileParamMICMAC(),aTree->Get("NameFileParamMICMAC",1)); //tototo 

   xml_init(anObj.WorkDir(),aTree->Get("WorkDir",1)); //tototo 

   xml_init(anObj.DirImagesOri(),aTree->Get("DirImagesOri",1)); //tototo 

   xml_init(anObj.TmpMEC(),aTree->Get("TmpMEC",1)); //tototo 

   xml_init(anObj.TmpPyr(),aTree->Get("TmpPyr",1)); //tototo 

   xml_init(anObj.TmpGeom(),aTree->Get("TmpGeom",1),std::string("")); //tototo 

   xml_init(anObj.TmpResult(),aTree->Get("TmpResult",1),std::string("Result/")); //tototo 

   xml_init(anObj.CalledByProcess(),aTree->Get("CalledByProcess",1),bool(false)); //tototo 

   xml_init(anObj.IdMasterProcess(),aTree->Get("IdMasterProcess",1),int(-1)); //tototo 

   xml_init(anObj.CreateGrayFileAtBegin(),aTree->Get("CreateGrayFileAtBegin",1),bool(false)); //tototo 

   xml_init(anObj.Visu(),aTree->Get("Visu",1),bool(false)); //tototo 

   xml_init(anObj.ByProcess(),aTree->Get("ByProcess",1),int(0)); //tototo 

   xml_init(anObj.StopOnEchecFils(),aTree->Get("StopOnEchecFils",1),bool(true)); //tototo 

   xml_init(anObj.AvalaibleMemory(),aTree->Get("AvalaibleMemory",1),int(128)); //tototo 

   xml_init(anObj.SzRecouvrtDalles(),aTree->Get("SzRecouvrtDalles",1),int(50)); //tototo 

   xml_init(anObj.SzDalleMin(),aTree->Get("SzDalleMin",1),int(400)); //tototo 

   xml_init(anObj.SzDalleMax(),aTree->Get("SzDalleMax",1),int(800)); //tototo 

   xml_init(anObj.NbCelluleMax(),aTree->Get("NbCelluleMax",1),double(2e7)); //tototo 

   xml_init(anObj.SzMinDecomposCalc(),aTree->Get("SzMinDecomposCalc",1),int(10)); //tototo 

   xml_init(anObj.AutorizeSplitRec(),aTree->Get("AutorizeSplitRec",1)); //tototo 

   xml_init(anObj.DefTileFile(),aTree->Get("DefTileFile",1),int(10000)); //tototo 

   xml_init(anObj.NbPixDefFilesAux(),aTree->Get("NbPixDefFilesAux",1),double(3.0e7)); //tototo 

   xml_init(anObj.DeZoomDefMinFileAux(),aTree->Get("DeZoomDefMinFileAux",1),int(4)); //tototo 

   xml_init(anObj.FirstEtapeMEC(),aTree->Get("FirstEtapeMEC",1),int(0)); //tototo 

   xml_init(anObj.LastEtapeMEC(),aTree->Get("LastEtapeMEC",1),int(10000)); //tototo 

   xml_init(anObj.FirstBoiteMEC(),aTree->Get("FirstBoiteMEC",1),int(0)); //tototo 

   xml_init(anObj.NbBoitesMEC(),aTree->Get("NbBoitesMEC",1),int(100000000)); //tototo 

   xml_init(anObj.NomChantier(),aTree->Get("NomChantier",1),std::string("LeChantier")); //tototo 

   xml_init(anObj.CalcNomChantier(),aTree->Get("CalcNomChantier",1)); //tototo 

   xml_init(anObj.PatternSelPyr(),aTree->Get("PatternSelPyr",1),std::string("(.*)@(.*)")); //tototo 

   xml_init(anObj.PatternNomPyr(),aTree->Get("PatternNomPyr",1),std::string("$1DeZoom$2.tif")); //tototo 

   xml_init(anObj.SeparateurPyr(),aTree->Get("SeparateurPyr",1),std::string("@")); //tototo 

   xml_init(anObj.KeyCalNamePyr(),aTree->Get("KeyCalNamePyr",1),std::string("Key-Assoc-Pyram-MM")); //tototo 

   xml_init(anObj.ActivePurge(),aTree->Get("ActivePurge",1),bool(false)); //tototo 

   xml_init(anObj.PurgeFiles(),aTree->GetAll("PurgeFiles",false,1));

   xml_init(anObj.PurgeMECResultBefore(),aTree->Get("PurgeMECResultBefore",1),bool(false)); //tototo 

   xml_init(anObj.PreservedFile(),aTree->Get("PreservedFile",1)); //tototo 

   xml_init(anObj.UseChantierNameDescripteur(),aTree->Get("UseChantierNameDescripteur",1),bool(false)); //tototo 

   xml_init(anObj.FileChantierNameDescripteur(),aTree->Get("FileChantierNameDescripteur",1)); //tototo 

   xml_init(anObj.MapMicMac(),aTree->Get("MapMicMac",1)); //tototo 

   xml_init(anObj.PostProcess(),aTree->Get("PostProcess",1)); //tototo 

   xml_init(anObj.ComprMasque(),aTree->Get("ComprMasque",1),eComprTiff(eComprTiff_FAX4)); //tototo 

   xml_init(anObj.TypeMasque(),aTree->Get("TypeMasque",1),eTypeNumerique(eTN_Bits1MSBF)); //tototo 
}

std::string  Mangling( cSection_WorkSpace *) {return "66898FB8C27CB1D0FE3F";};


std::string & cOneBatch::PatternSelImBatch()
{
   return mPatternSelImBatch;
}

const std::string & cOneBatch::PatternSelImBatch()const 
{
   return mPatternSelImBatch;
}


std::list< std::string > & cOneBatch::PatternCommandeBatch()
{
   return mPatternCommandeBatch;
}

const std::list< std::string > & cOneBatch::PatternCommandeBatch()const 
{
   return mPatternCommandeBatch;
}

void  BinaryUnDumpFromFile(cOneBatch & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.PatternSelImBatch(),aFp);
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             std::string aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.PatternCommandeBatch().push_back(aVal);
        }
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cOneBatch & anObj)
{
    BinaryDumpInFile(aFp,anObj.PatternSelImBatch());
    BinaryDumpInFile(aFp,(int)anObj.PatternCommandeBatch().size());
    for(  std::list< std::string >::const_iterator iT=anObj.PatternCommandeBatch().begin();
         iT!=anObj.PatternCommandeBatch().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
}

cElXMLTree * ToXMLTree(const cOneBatch & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"OneBatch",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("PatternSelImBatch"),anObj.PatternSelImBatch())->ReTagThis("PatternSelImBatch"));
  for
  (       std::list< std::string >::const_iterator it=anObj.PatternCommandeBatch().begin();
      it !=anObj.PatternCommandeBatch().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("PatternCommandeBatch"),(*it))->ReTagThis("PatternCommandeBatch"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cOneBatch & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.PatternSelImBatch(),aTree->Get("PatternSelImBatch",1)); //tototo 

   xml_init(anObj.PatternCommandeBatch(),aTree->GetAll("PatternCommandeBatch",false,1));
}

std::string  Mangling( cOneBatch *) {return "818118AEF22B27A2FE3F";};


cTplValGesInit< bool > & cSectionBatch::ExeBatch()
{
   return mExeBatch;
}

const cTplValGesInit< bool > & cSectionBatch::ExeBatch()const 
{
   return mExeBatch;
}


std::list< cOneBatch > & cSectionBatch::OneBatch()
{
   return mOneBatch;
}

const std::list< cOneBatch > & cSectionBatch::OneBatch()const 
{
   return mOneBatch;
}


std::list< std::string > & cSectionBatch::NextMicMacFile2Exec()
{
   return mNextMicMacFile2Exec;
}

const std::list< std::string > & cSectionBatch::NextMicMacFile2Exec()const 
{
   return mNextMicMacFile2Exec;
}

void  BinaryUnDumpFromFile(cSectionBatch & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ExeBatch().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ExeBatch().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ExeBatch().SetNoInit();
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cOneBatch aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.OneBatch().push_back(aVal);
        }
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             std::string aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.NextMicMacFile2Exec().push_back(aVal);
        }
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cSectionBatch & anObj)
{
    BinaryDumpInFile(aFp,anObj.ExeBatch().IsInit());
    if (anObj.ExeBatch().IsInit()) BinaryDumpInFile(aFp,anObj.ExeBatch().Val());
    BinaryDumpInFile(aFp,(int)anObj.OneBatch().size());
    for(  std::list< cOneBatch >::const_iterator iT=anObj.OneBatch().begin();
         iT!=anObj.OneBatch().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,(int)anObj.NextMicMacFile2Exec().size());
    for(  std::list< std::string >::const_iterator iT=anObj.NextMicMacFile2Exec().begin();
         iT!=anObj.NextMicMacFile2Exec().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
}

cElXMLTree * ToXMLTree(const cSectionBatch & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"SectionBatch",eXMLBranche);
   if (anObj.ExeBatch().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ExeBatch"),anObj.ExeBatch().Val())->ReTagThis("ExeBatch"));
  for
  (       std::list< cOneBatch >::const_iterator it=anObj.OneBatch().begin();
      it !=anObj.OneBatch().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("OneBatch"));
  for
  (       std::list< std::string >::const_iterator it=anObj.NextMicMacFile2Exec().begin();
      it !=anObj.NextMicMacFile2Exec().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("NextMicMacFile2Exec"),(*it))->ReTagThis("NextMicMacFile2Exec"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cSectionBatch & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.ExeBatch(),aTree->Get("ExeBatch",1),bool(true)); //tototo 

   xml_init(anObj.OneBatch(),aTree->GetAll("OneBatch",false,1));

   xml_init(anObj.NextMicMacFile2Exec(),aTree->GetAll("NextMicMacFile2Exec",false,1));
}

std::string  Mangling( cSectionBatch *) {return "8347090E638BD29CFE3F";};


Pt2dr & cListTestCpleHomol::PtIm1()
{
   return mPtIm1;
}

const Pt2dr & cListTestCpleHomol::PtIm1()const 
{
   return mPtIm1;
}


Pt2dr & cListTestCpleHomol::PtIm2()
{
   return mPtIm2;
}

const Pt2dr & cListTestCpleHomol::PtIm2()const 
{
   return mPtIm2;
}

void  BinaryUnDumpFromFile(cListTestCpleHomol & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.PtIm1(),aFp);
    BinaryUnDumpFromFile(anObj.PtIm2(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cListTestCpleHomol & anObj)
{
    BinaryDumpInFile(aFp,anObj.PtIm1());
    BinaryDumpInFile(aFp,anObj.PtIm2());
}

cElXMLTree * ToXMLTree(const cListTestCpleHomol & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ListTestCpleHomol",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("PtIm1"),anObj.PtIm1())->ReTagThis("PtIm1"));
   aRes->AddFils(::ToXMLTree(std::string("PtIm2"),anObj.PtIm2())->ReTagThis("PtIm2"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cListTestCpleHomol & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.PtIm1(),aTree->Get("PtIm1",1)); //tototo 

   xml_init(anObj.PtIm2(),aTree->Get("PtIm2",1)); //tototo 
}

std::string  Mangling( cListTestCpleHomol *) {return "027E45BB83D50DD1FD3F";};


Pt2di & cDebugEscalier::P1()
{
   return mP1;
}

const Pt2di & cDebugEscalier::P1()const 
{
   return mP1;
}


Pt2di & cDebugEscalier::P2()
{
   return mP2;
}

const Pt2di & cDebugEscalier::P2()const 
{
   return mP2;
}


cTplValGesInit< bool > & cDebugEscalier::ShowDerivZ()
{
   return mShowDerivZ;
}

const cTplValGesInit< bool > & cDebugEscalier::ShowDerivZ()const 
{
   return mShowDerivZ;
}

void  BinaryUnDumpFromFile(cDebugEscalier & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.P1(),aFp);
    BinaryUnDumpFromFile(anObj.P2(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ShowDerivZ().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ShowDerivZ().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ShowDerivZ().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cDebugEscalier & anObj)
{
    BinaryDumpInFile(aFp,anObj.P1());
    BinaryDumpInFile(aFp,anObj.P2());
    BinaryDumpInFile(aFp,anObj.ShowDerivZ().IsInit());
    if (anObj.ShowDerivZ().IsInit()) BinaryDumpInFile(aFp,anObj.ShowDerivZ().Val());
}

cElXMLTree * ToXMLTree(const cDebugEscalier & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"DebugEscalier",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("P1"),anObj.P1())->ReTagThis("P1"));
   aRes->AddFils(::ToXMLTree(std::string("P2"),anObj.P2())->ReTagThis("P2"));
   if (anObj.ShowDerivZ().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ShowDerivZ"),anObj.ShowDerivZ().Val())->ReTagThis("ShowDerivZ"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cDebugEscalier & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.P1(),aTree->Get("P1",1)); //tototo 

   xml_init(anObj.P2(),aTree->Get("P2",1)); //tototo 

   xml_init(anObj.ShowDerivZ(),aTree->Get("ShowDerivZ",1),bool(false)); //tototo 
}

std::string  Mangling( cDebugEscalier *) {return "C4B2F9CF8AD914FFFD3F";};


Pt2di & cSectionDebug::P1()
{
   return DebugEscalier().Val().P1();
}

const Pt2di & cSectionDebug::P1()const 
{
   return DebugEscalier().Val().P1();
}


Pt2di & cSectionDebug::P2()
{
   return DebugEscalier().Val().P2();
}

const Pt2di & cSectionDebug::P2()const 
{
   return DebugEscalier().Val().P2();
}


cTplValGesInit< bool > & cSectionDebug::ShowDerivZ()
{
   return DebugEscalier().Val().ShowDerivZ();
}

const cTplValGesInit< bool > & cSectionDebug::ShowDerivZ()const 
{
   return DebugEscalier().Val().ShowDerivZ();
}


cTplValGesInit< cDebugEscalier > & cSectionDebug::DebugEscalier()
{
   return mDebugEscalier;
}

const cTplValGesInit< cDebugEscalier > & cSectionDebug::DebugEscalier()const 
{
   return mDebugEscalier;
}

void  BinaryUnDumpFromFile(cSectionDebug & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.DebugEscalier().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.DebugEscalier().ValForcedForUnUmp(),aFp);
        }
        else  anObj.DebugEscalier().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cSectionDebug & anObj)
{
    BinaryDumpInFile(aFp,anObj.DebugEscalier().IsInit());
    if (anObj.DebugEscalier().IsInit()) BinaryDumpInFile(aFp,anObj.DebugEscalier().Val());
}

cElXMLTree * ToXMLTree(const cSectionDebug & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"SectionDebug",eXMLBranche);
   if (anObj.DebugEscalier().IsInit())
      aRes->AddFils(ToXMLTree(anObj.DebugEscalier().Val())->ReTagThis("DebugEscalier"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cSectionDebug & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.DebugEscalier(),aTree->Get("DebugEscalier",1)); //tototo 
}

std::string  Mangling( cSectionDebug *) {return "E800380D4A9561EAFD3F";};


cTplValGesInit< bool > & cSection_Vrac::DebugMM()
{
   return mDebugMM;
}

const cTplValGesInit< bool > & cSection_Vrac::DebugMM()const 
{
   return mDebugMM;
}


cTplValGesInit< int > & cSection_Vrac::SL_XSzW()
{
   return mSL_XSzW;
}

const cTplValGesInit< int > & cSection_Vrac::SL_XSzW()const 
{
   return mSL_XSzW;
}


cTplValGesInit< int > & cSection_Vrac::SL_YSzW()
{
   return mSL_YSzW;
}

const cTplValGesInit< int > & cSection_Vrac::SL_YSzW()const 
{
   return mSL_YSzW;
}


cTplValGesInit< bool > & cSection_Vrac::SL_Epip()
{
   return mSL_Epip;
}

const cTplValGesInit< bool > & cSection_Vrac::SL_Epip()const 
{
   return mSL_Epip;
}


cTplValGesInit< int > & cSection_Vrac::SL_YDecEpip()
{
   return mSL_YDecEpip;
}

const cTplValGesInit< int > & cSection_Vrac::SL_YDecEpip()const 
{
   return mSL_YDecEpip;
}


cTplValGesInit< std::string > & cSection_Vrac::SL_PackHom0()
{
   return mSL_PackHom0;
}

const cTplValGesInit< std::string > & cSection_Vrac::SL_PackHom0()const 
{
   return mSL_PackHom0;
}


cTplValGesInit< bool > & cSection_Vrac::SL_RedrOnCur()
{
   return mSL_RedrOnCur;
}

const cTplValGesInit< bool > & cSection_Vrac::SL_RedrOnCur()const 
{
   return mSL_RedrOnCur;
}


cTplValGesInit< bool > & cSection_Vrac::SL_NewRedrCur()
{
   return mSL_NewRedrCur;
}

const cTplValGesInit< bool > & cSection_Vrac::SL_NewRedrCur()const 
{
   return mSL_NewRedrCur;
}


cTplValGesInit< bool > & cSection_Vrac::SL_L2Estim()
{
   return mSL_L2Estim;
}

const cTplValGesInit< bool > & cSection_Vrac::SL_L2Estim()const 
{
   return mSL_L2Estim;
}


cTplValGesInit< std::vector<std::string> > & cSection_Vrac::SL_FILTER()
{
   return mSL_FILTER;
}

const cTplValGesInit< std::vector<std::string> > & cSection_Vrac::SL_FILTER()const 
{
   return mSL_FILTER;
}


cTplValGesInit< bool > & cSection_Vrac::SL_TJS_FILTER()
{
   return mSL_TJS_FILTER;
}

const cTplValGesInit< bool > & cSection_Vrac::SL_TJS_FILTER()const 
{
   return mSL_TJS_FILTER;
}


cTplValGesInit< double > & cSection_Vrac::SL_Step_Grid()
{
   return mSL_Step_Grid;
}

const cTplValGesInit< double > & cSection_Vrac::SL_Step_Grid()const 
{
   return mSL_Step_Grid;
}


cTplValGesInit< std::string > & cSection_Vrac::SL_Name_Grid_Exp()
{
   return mSL_Name_Grid_Exp;
}

const cTplValGesInit< std::string > & cSection_Vrac::SL_Name_Grid_Exp()const 
{
   return mSL_Name_Grid_Exp;
}


cTplValGesInit< double > & cSection_Vrac::VSG_DynImRed()
{
   return mVSG_DynImRed;
}

const cTplValGesInit< double > & cSection_Vrac::VSG_DynImRed()const 
{
   return mVSG_DynImRed;
}


cTplValGesInit< int > & cSection_Vrac::VSG_DeZoomContr()
{
   return mVSG_DeZoomContr;
}

const cTplValGesInit< int > & cSection_Vrac::VSG_DeZoomContr()const 
{
   return mVSG_DeZoomContr;
}


cTplValGesInit< Pt2di > & cSection_Vrac::PtDebug()
{
   return mPtDebug;
}

const cTplValGesInit< Pt2di > & cSection_Vrac::PtDebug()const 
{
   return mPtDebug;
}


cTplValGesInit< bool > & cSection_Vrac::DumpNappesEnglob()
{
   return mDumpNappesEnglob;
}

const cTplValGesInit< bool > & cSection_Vrac::DumpNappesEnglob()const 
{
   return mDumpNappesEnglob;
}


cTplValGesInit< bool > & cSection_Vrac::InterditAccelerationCorrSpec()
{
   return mInterditAccelerationCorrSpec;
}

const cTplValGesInit< bool > & cSection_Vrac::InterditAccelerationCorrSpec()const 
{
   return mInterditAccelerationCorrSpec;
}


cTplValGesInit< bool > & cSection_Vrac::InterditCorrelRapide()
{
   return mInterditCorrelRapide;
}

const cTplValGesInit< bool > & cSection_Vrac::InterditCorrelRapide()const 
{
   return mInterditCorrelRapide;
}


cTplValGesInit< bool > & cSection_Vrac::ForceCorrelationByRect()
{
   return mForceCorrelationByRect;
}

const cTplValGesInit< bool > & cSection_Vrac::ForceCorrelationByRect()const 
{
   return mForceCorrelationByRect;
}


std::list< cListTestCpleHomol > & cSection_Vrac::ListTestCpleHomol()
{
   return mListTestCpleHomol;
}

const std::list< cListTestCpleHomol > & cSection_Vrac::ListTestCpleHomol()const 
{
   return mListTestCpleHomol;
}


std::list< Pt3dr > & cSection_Vrac::ListeTestPointsTerrain()
{
   return mListeTestPointsTerrain;
}

const std::list< Pt3dr > & cSection_Vrac::ListeTestPointsTerrain()const 
{
   return mListeTestPointsTerrain;
}


cTplValGesInit< bool > & cSection_Vrac::WithMessage()
{
   return mWithMessage;
}

const cTplValGesInit< bool > & cSection_Vrac::WithMessage()const 
{
   return mWithMessage;
}


cTplValGesInit< bool > & cSection_Vrac::ShowLoadedImage()
{
   return mShowLoadedImage;
}

const cTplValGesInit< bool > & cSection_Vrac::ShowLoadedImage()const 
{
   return mShowLoadedImage;
}


Pt2di & cSection_Vrac::P1()
{
   return SectionDebug().Val().DebugEscalier().Val().P1();
}

const Pt2di & cSection_Vrac::P1()const 
{
   return SectionDebug().Val().DebugEscalier().Val().P1();
}


Pt2di & cSection_Vrac::P2()
{
   return SectionDebug().Val().DebugEscalier().Val().P2();
}

const Pt2di & cSection_Vrac::P2()const 
{
   return SectionDebug().Val().DebugEscalier().Val().P2();
}


cTplValGesInit< bool > & cSection_Vrac::ShowDerivZ()
{
   return SectionDebug().Val().DebugEscalier().Val().ShowDerivZ();
}

const cTplValGesInit< bool > & cSection_Vrac::ShowDerivZ()const 
{
   return SectionDebug().Val().DebugEscalier().Val().ShowDerivZ();
}


cTplValGesInit< cDebugEscalier > & cSection_Vrac::DebugEscalier()
{
   return SectionDebug().Val().DebugEscalier();
}

const cTplValGesInit< cDebugEscalier > & cSection_Vrac::DebugEscalier()const 
{
   return SectionDebug().Val().DebugEscalier();
}


cTplValGesInit< cSectionDebug > & cSection_Vrac::SectionDebug()
{
   return mSectionDebug;
}

const cTplValGesInit< cSectionDebug > & cSection_Vrac::SectionDebug()const 
{
   return mSectionDebug;
}

void  BinaryUnDumpFromFile(cSection_Vrac & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.DebugMM().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.DebugMM().ValForcedForUnUmp(),aFp);
        }
        else  anObj.DebugMM().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SL_XSzW().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SL_XSzW().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SL_XSzW().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SL_YSzW().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SL_YSzW().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SL_YSzW().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SL_Epip().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SL_Epip().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SL_Epip().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SL_YDecEpip().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SL_YDecEpip().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SL_YDecEpip().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SL_PackHom0().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SL_PackHom0().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SL_PackHom0().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SL_RedrOnCur().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SL_RedrOnCur().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SL_RedrOnCur().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SL_NewRedrCur().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SL_NewRedrCur().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SL_NewRedrCur().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SL_L2Estim().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SL_L2Estim().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SL_L2Estim().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SL_FILTER().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SL_FILTER().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SL_FILTER().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SL_TJS_FILTER().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SL_TJS_FILTER().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SL_TJS_FILTER().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SL_Step_Grid().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SL_Step_Grid().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SL_Step_Grid().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SL_Name_Grid_Exp().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SL_Name_Grid_Exp().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SL_Name_Grid_Exp().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.VSG_DynImRed().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.VSG_DynImRed().ValForcedForUnUmp(),aFp);
        }
        else  anObj.VSG_DynImRed().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.VSG_DeZoomContr().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.VSG_DeZoomContr().ValForcedForUnUmp(),aFp);
        }
        else  anObj.VSG_DeZoomContr().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.PtDebug().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.PtDebug().ValForcedForUnUmp(),aFp);
        }
        else  anObj.PtDebug().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.DumpNappesEnglob().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.DumpNappesEnglob().ValForcedForUnUmp(),aFp);
        }
        else  anObj.DumpNappesEnglob().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.InterditAccelerationCorrSpec().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.InterditAccelerationCorrSpec().ValForcedForUnUmp(),aFp);
        }
        else  anObj.InterditAccelerationCorrSpec().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.InterditCorrelRapide().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.InterditCorrelRapide().ValForcedForUnUmp(),aFp);
        }
        else  anObj.InterditCorrelRapide().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ForceCorrelationByRect().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ForceCorrelationByRect().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ForceCorrelationByRect().SetNoInit();
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cListTestCpleHomol aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.ListTestCpleHomol().push_back(aVal);
        }
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             Pt3dr aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.ListeTestPointsTerrain().push_back(aVal);
        }
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.WithMessage().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.WithMessage().ValForcedForUnUmp(),aFp);
        }
        else  anObj.WithMessage().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ShowLoadedImage().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ShowLoadedImage().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ShowLoadedImage().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SectionDebug().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SectionDebug().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SectionDebug().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cSection_Vrac & anObj)
{
    BinaryDumpInFile(aFp,anObj.DebugMM().IsInit());
    if (anObj.DebugMM().IsInit()) BinaryDumpInFile(aFp,anObj.DebugMM().Val());
    BinaryDumpInFile(aFp,anObj.SL_XSzW().IsInit());
    if (anObj.SL_XSzW().IsInit()) BinaryDumpInFile(aFp,anObj.SL_XSzW().Val());
    BinaryDumpInFile(aFp,anObj.SL_YSzW().IsInit());
    if (anObj.SL_YSzW().IsInit()) BinaryDumpInFile(aFp,anObj.SL_YSzW().Val());
    BinaryDumpInFile(aFp,anObj.SL_Epip().IsInit());
    if (anObj.SL_Epip().IsInit()) BinaryDumpInFile(aFp,anObj.SL_Epip().Val());
    BinaryDumpInFile(aFp,anObj.SL_YDecEpip().IsInit());
    if (anObj.SL_YDecEpip().IsInit()) BinaryDumpInFile(aFp,anObj.SL_YDecEpip().Val());
    BinaryDumpInFile(aFp,anObj.SL_PackHom0().IsInit());
    if (anObj.SL_PackHom0().IsInit()) BinaryDumpInFile(aFp,anObj.SL_PackHom0().Val());
    BinaryDumpInFile(aFp,anObj.SL_RedrOnCur().IsInit());
    if (anObj.SL_RedrOnCur().IsInit()) BinaryDumpInFile(aFp,anObj.SL_RedrOnCur().Val());
    BinaryDumpInFile(aFp,anObj.SL_NewRedrCur().IsInit());
    if (anObj.SL_NewRedrCur().IsInit()) BinaryDumpInFile(aFp,anObj.SL_NewRedrCur().Val());
    BinaryDumpInFile(aFp,anObj.SL_L2Estim().IsInit());
    if (anObj.SL_L2Estim().IsInit()) BinaryDumpInFile(aFp,anObj.SL_L2Estim().Val());
    BinaryDumpInFile(aFp,anObj.SL_FILTER().IsInit());
    if (anObj.SL_FILTER().IsInit()) BinaryDumpInFile(aFp,anObj.SL_FILTER().Val());
    BinaryDumpInFile(aFp,anObj.SL_TJS_FILTER().IsInit());
    if (anObj.SL_TJS_FILTER().IsInit()) BinaryDumpInFile(aFp,anObj.SL_TJS_FILTER().Val());
    BinaryDumpInFile(aFp,anObj.SL_Step_Grid().IsInit());
    if (anObj.SL_Step_Grid().IsInit()) BinaryDumpInFile(aFp,anObj.SL_Step_Grid().Val());
    BinaryDumpInFile(aFp,anObj.SL_Name_Grid_Exp().IsInit());
    if (anObj.SL_Name_Grid_Exp().IsInit()) BinaryDumpInFile(aFp,anObj.SL_Name_Grid_Exp().Val());
    BinaryDumpInFile(aFp,anObj.VSG_DynImRed().IsInit());
    if (anObj.VSG_DynImRed().IsInit()) BinaryDumpInFile(aFp,anObj.VSG_DynImRed().Val());
    BinaryDumpInFile(aFp,anObj.VSG_DeZoomContr().IsInit());
    if (anObj.VSG_DeZoomContr().IsInit()) BinaryDumpInFile(aFp,anObj.VSG_DeZoomContr().Val());
    BinaryDumpInFile(aFp,anObj.PtDebug().IsInit());
    if (anObj.PtDebug().IsInit()) BinaryDumpInFile(aFp,anObj.PtDebug().Val());
    BinaryDumpInFile(aFp,anObj.DumpNappesEnglob().IsInit());
    if (anObj.DumpNappesEnglob().IsInit()) BinaryDumpInFile(aFp,anObj.DumpNappesEnglob().Val());
    BinaryDumpInFile(aFp,anObj.InterditAccelerationCorrSpec().IsInit());
    if (anObj.InterditAccelerationCorrSpec().IsInit()) BinaryDumpInFile(aFp,anObj.InterditAccelerationCorrSpec().Val());
    BinaryDumpInFile(aFp,anObj.InterditCorrelRapide().IsInit());
    if (anObj.InterditCorrelRapide().IsInit()) BinaryDumpInFile(aFp,anObj.InterditCorrelRapide().Val());
    BinaryDumpInFile(aFp,anObj.ForceCorrelationByRect().IsInit());
    if (anObj.ForceCorrelationByRect().IsInit()) BinaryDumpInFile(aFp,anObj.ForceCorrelationByRect().Val());
    BinaryDumpInFile(aFp,(int)anObj.ListTestCpleHomol().size());
    for(  std::list< cListTestCpleHomol >::const_iterator iT=anObj.ListTestCpleHomol().begin();
         iT!=anObj.ListTestCpleHomol().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,(int)anObj.ListeTestPointsTerrain().size());
    for(  std::list< Pt3dr >::const_iterator iT=anObj.ListeTestPointsTerrain().begin();
         iT!=anObj.ListeTestPointsTerrain().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.WithMessage().IsInit());
    if (anObj.WithMessage().IsInit()) BinaryDumpInFile(aFp,anObj.WithMessage().Val());
    BinaryDumpInFile(aFp,anObj.ShowLoadedImage().IsInit());
    if (anObj.ShowLoadedImage().IsInit()) BinaryDumpInFile(aFp,anObj.ShowLoadedImage().Val());
    BinaryDumpInFile(aFp,anObj.SectionDebug().IsInit());
    if (anObj.SectionDebug().IsInit()) BinaryDumpInFile(aFp,anObj.SectionDebug().Val());
}

cElXMLTree * ToXMLTree(const cSection_Vrac & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"Section_Vrac",eXMLBranche);
   if (anObj.DebugMM().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DebugMM"),anObj.DebugMM().Val())->ReTagThis("DebugMM"));
   if (anObj.SL_XSzW().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SL_XSzW"),anObj.SL_XSzW().Val())->ReTagThis("SL_XSzW"));
   if (anObj.SL_YSzW().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SL_YSzW"),anObj.SL_YSzW().Val())->ReTagThis("SL_YSzW"));
   if (anObj.SL_Epip().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SL_Epip"),anObj.SL_Epip().Val())->ReTagThis("SL_Epip"));
   if (anObj.SL_YDecEpip().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SL_YDecEpip"),anObj.SL_YDecEpip().Val())->ReTagThis("SL_YDecEpip"));
   if (anObj.SL_PackHom0().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SL_PackHom0"),anObj.SL_PackHom0().Val())->ReTagThis("SL_PackHom0"));
   if (anObj.SL_RedrOnCur().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SL_RedrOnCur"),anObj.SL_RedrOnCur().Val())->ReTagThis("SL_RedrOnCur"));
   if (anObj.SL_NewRedrCur().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SL_NewRedrCur"),anObj.SL_NewRedrCur().Val())->ReTagThis("SL_NewRedrCur"));
   if (anObj.SL_L2Estim().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SL_L2Estim"),anObj.SL_L2Estim().Val())->ReTagThis("SL_L2Estim"));
   if (anObj.SL_FILTER().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SL_FILTER"),anObj.SL_FILTER().Val())->ReTagThis("SL_FILTER"));
   if (anObj.SL_TJS_FILTER().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SL_TJS_FILTER"),anObj.SL_TJS_FILTER().Val())->ReTagThis("SL_TJS_FILTER"));
   if (anObj.SL_Step_Grid().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SL_Step_Grid"),anObj.SL_Step_Grid().Val())->ReTagThis("SL_Step_Grid"));
   if (anObj.SL_Name_Grid_Exp().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SL_Name_Grid_Exp"),anObj.SL_Name_Grid_Exp().Val())->ReTagThis("SL_Name_Grid_Exp"));
   if (anObj.VSG_DynImRed().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("VSG_DynImRed"),anObj.VSG_DynImRed().Val())->ReTagThis("VSG_DynImRed"));
   if (anObj.VSG_DeZoomContr().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("VSG_DeZoomContr"),anObj.VSG_DeZoomContr().Val())->ReTagThis("VSG_DeZoomContr"));
   if (anObj.PtDebug().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PtDebug"),anObj.PtDebug().Val())->ReTagThis("PtDebug"));
   if (anObj.DumpNappesEnglob().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DumpNappesEnglob"),anObj.DumpNappesEnglob().Val())->ReTagThis("DumpNappesEnglob"));
   if (anObj.InterditAccelerationCorrSpec().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("InterditAccelerationCorrSpec"),anObj.InterditAccelerationCorrSpec().Val())->ReTagThis("InterditAccelerationCorrSpec"));
   if (anObj.InterditCorrelRapide().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("InterditCorrelRapide"),anObj.InterditCorrelRapide().Val())->ReTagThis("InterditCorrelRapide"));
   if (anObj.ForceCorrelationByRect().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ForceCorrelationByRect"),anObj.ForceCorrelationByRect().Val())->ReTagThis("ForceCorrelationByRect"));
  for
  (       std::list< cListTestCpleHomol >::const_iterator it=anObj.ListTestCpleHomol().begin();
      it !=anObj.ListTestCpleHomol().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("ListTestCpleHomol"));
  for
  (       std::list< Pt3dr >::const_iterator it=anObj.ListeTestPointsTerrain().begin();
      it !=anObj.ListeTestPointsTerrain().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("ListeTestPointsTerrain"),(*it))->ReTagThis("ListeTestPointsTerrain"));
   if (anObj.WithMessage().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("WithMessage"),anObj.WithMessage().Val())->ReTagThis("WithMessage"));
   if (anObj.ShowLoadedImage().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ShowLoadedImage"),anObj.ShowLoadedImage().Val())->ReTagThis("ShowLoadedImage"));
   if (anObj.SectionDebug().IsInit())
      aRes->AddFils(ToXMLTree(anObj.SectionDebug().Val())->ReTagThis("SectionDebug"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cSection_Vrac & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.DebugMM(),aTree->Get("DebugMM",1),bool(false)); //tototo 

   xml_init(anObj.SL_XSzW(),aTree->Get("SL_XSzW",1),int(1000)); //tototo 

   xml_init(anObj.SL_YSzW(),aTree->Get("SL_YSzW",1),int(900)); //tototo 

   xml_init(anObj.SL_Epip(),aTree->Get("SL_Epip",1),bool(false)); //tototo 

   xml_init(anObj.SL_YDecEpip(),aTree->Get("SL_YDecEpip",1),int(0)); //tototo 

   xml_init(anObj.SL_PackHom0(),aTree->Get("SL_PackHom0",1),std::string("")); //tototo 

   xml_init(anObj.SL_RedrOnCur(),aTree->Get("SL_RedrOnCur",1),bool(false)); //tototo 

   xml_init(anObj.SL_NewRedrCur(),aTree->Get("SL_NewRedrCur",1),bool(false)); //tototo 

   xml_init(anObj.SL_L2Estim(),aTree->Get("SL_L2Estim",1),bool(true)); //tototo 

   xml_init(anObj.SL_FILTER(),aTree->Get("SL_FILTER",1)); //tototo 

   xml_init(anObj.SL_TJS_FILTER(),aTree->Get("SL_TJS_FILTER",1),bool(false)); //tototo 

   xml_init(anObj.SL_Step_Grid(),aTree->Get("SL_Step_Grid",1),double(10.0)); //tototo 

   xml_init(anObj.SL_Name_Grid_Exp(),aTree->Get("SL_Name_Grid_Exp",1),std::string("GridMap_%I_To_%J")); //tototo 

   xml_init(anObj.VSG_DynImRed(),aTree->Get("VSG_DynImRed",1),double(5.0)); //tototo 

   xml_init(anObj.VSG_DeZoomContr(),aTree->Get("VSG_DeZoomContr",1),int(16)); //tototo 

   xml_init(anObj.PtDebug(),aTree->Get("PtDebug",1)); //tototo 

   xml_init(anObj.DumpNappesEnglob(),aTree->Get("DumpNappesEnglob",1),bool(false)); //tototo 

   xml_init(anObj.InterditAccelerationCorrSpec(),aTree->Get("InterditAccelerationCorrSpec",1),bool(false)); //tototo 

   xml_init(anObj.InterditCorrelRapide(),aTree->Get("InterditCorrelRapide",1),bool(false)); //tototo 

   xml_init(anObj.ForceCorrelationByRect(),aTree->Get("ForceCorrelationByRect",1),bool(false)); //tototo 

   xml_init(anObj.ListTestCpleHomol(),aTree->GetAll("ListTestCpleHomol",false,1));

   xml_init(anObj.ListeTestPointsTerrain(),aTree->GetAll("ListeTestPointsTerrain",false,1));

   xml_init(anObj.WithMessage(),aTree->Get("WithMessage",1),bool(false)); //tototo 

   xml_init(anObj.ShowLoadedImage(),aTree->Get("ShowLoadedImage",1),bool(false)); //tototo 

   xml_init(anObj.SectionDebug(),aTree->Get("SectionDebug",1)); //tototo 
}

std::string  Mangling( cSection_Vrac *) {return "B02026EFB14373EBFA3F";};


cTplValGesInit< cChantierDescripteur > & cParamMICMAC::DicoLoc()
{
   return mDicoLoc;
}

const cTplValGesInit< cChantierDescripteur > & cParamMICMAC::DicoLoc()const 
{
   return mDicoLoc;
}


cTplValGesInit< bool > & cParamMICMAC::IntervalPaxIsProportion()
{
   return Section_Terrain().IntervalPaxIsProportion();
}

const cTplValGesInit< bool > & cParamMICMAC::IntervalPaxIsProportion()const 
{
   return Section_Terrain().IntervalPaxIsProportion();
}


cTplValGesInit< double > & cParamMICMAC::RatioAltiPlani()
{
   return Section_Terrain().RatioAltiPlani();
}

const cTplValGesInit< double > & cParamMICMAC::RatioAltiPlani()const 
{
   return Section_Terrain().RatioAltiPlani();
}


cTplValGesInit< bool > & cParamMICMAC::EstimPxPrefZ2Prof()
{
   return Section_Terrain().EstimPxPrefZ2Prof();
}

const cTplValGesInit< bool > & cParamMICMAC::EstimPxPrefZ2Prof()const 
{
   return Section_Terrain().EstimPxPrefZ2Prof();
}


cTplValGesInit< double > & cParamMICMAC::ZMoyen()
{
   return Section_Terrain().IntervAltimetrie().Val().ZMoyen();
}

const cTplValGesInit< double > & cParamMICMAC::ZMoyen()const 
{
   return Section_Terrain().IntervAltimetrie().Val().ZMoyen();
}


double & cParamMICMAC::ZIncCalc()
{
   return Section_Terrain().IntervAltimetrie().Val().ZIncCalc();
}

const double & cParamMICMAC::ZIncCalc()const 
{
   return Section_Terrain().IntervAltimetrie().Val().ZIncCalc();
}


cTplValGesInit< bool > & cParamMICMAC::ZIncIsProp()
{
   return Section_Terrain().IntervAltimetrie().Val().ZIncIsProp();
}

const cTplValGesInit< bool > & cParamMICMAC::ZIncIsProp()const 
{
   return Section_Terrain().IntervAltimetrie().Val().ZIncIsProp();
}


cTplValGesInit< double > & cParamMICMAC::ZIncZonage()
{
   return Section_Terrain().IntervAltimetrie().Val().ZIncZonage();
}

const cTplValGesInit< double > & cParamMICMAC::ZIncZonage()const 
{
   return Section_Terrain().IntervAltimetrie().Val().ZIncZonage();
}


std::string & cParamMICMAC::MNT_Init_Image()
{
   return Section_Terrain().IntervAltimetrie().Val().MNT_Init().Val().MNT_Init_Image();
}

const std::string & cParamMICMAC::MNT_Init_Image()const 
{
   return Section_Terrain().IntervAltimetrie().Val().MNT_Init().Val().MNT_Init_Image();
}


std::string & cParamMICMAC::MNT_Init_Xml()
{
   return Section_Terrain().IntervAltimetrie().Val().MNT_Init().Val().MNT_Init_Xml();
}

const std::string & cParamMICMAC::MNT_Init_Xml()const 
{
   return Section_Terrain().IntervAltimetrie().Val().MNT_Init().Val().MNT_Init_Xml();
}


cTplValGesInit< double > & cParamMICMAC::MNT_Offset()
{
   return Section_Terrain().IntervAltimetrie().Val().MNT_Init().Val().MNT_Offset();
}

const cTplValGesInit< double > & cParamMICMAC::MNT_Offset()const 
{
   return Section_Terrain().IntervAltimetrie().Val().MNT_Init().Val().MNT_Offset();
}


cTplValGesInit< cMNT_Init > & cParamMICMAC::MNT_Init()
{
   return Section_Terrain().IntervAltimetrie().Val().MNT_Init();
}

const cTplValGesInit< cMNT_Init > & cParamMICMAC::MNT_Init()const 
{
   return Section_Terrain().IntervAltimetrie().Val().MNT_Init();
}


std::string & cParamMICMAC::ZInf()
{
   return Section_Terrain().IntervAltimetrie().Val().EnveloppeMNT_INIT().Val().ZInf();
}

const std::string & cParamMICMAC::ZInf()const 
{
   return Section_Terrain().IntervAltimetrie().Val().EnveloppeMNT_INIT().Val().ZInf();
}


std::string & cParamMICMAC::ZSup()
{
   return Section_Terrain().IntervAltimetrie().Val().EnveloppeMNT_INIT().Val().ZSup();
}

const std::string & cParamMICMAC::ZSup()const 
{
   return Section_Terrain().IntervAltimetrie().Val().EnveloppeMNT_INIT().Val().ZSup();
}


cTplValGesInit< cEnveloppeMNT_INIT > & cParamMICMAC::EnveloppeMNT_INIT()
{
   return Section_Terrain().IntervAltimetrie().Val().EnveloppeMNT_INIT();
}

const cTplValGesInit< cEnveloppeMNT_INIT > & cParamMICMAC::EnveloppeMNT_INIT()const 
{
   return Section_Terrain().IntervAltimetrie().Val().EnveloppeMNT_INIT();
}


cTplValGesInit< cIntervAltimetrie > & cParamMICMAC::IntervAltimetrie()
{
   return Section_Terrain().IntervAltimetrie();
}

const cTplValGesInit< cIntervAltimetrie > & cParamMICMAC::IntervAltimetrie()const 
{
   return Section_Terrain().IntervAltimetrie();
}


cTplValGesInit< double > & cParamMICMAC::Px1Moy()
{
   return Section_Terrain().IntervParalaxe().Val().Px1Moy();
}

const cTplValGesInit< double > & cParamMICMAC::Px1Moy()const 
{
   return Section_Terrain().IntervParalaxe().Val().Px1Moy();
}


cTplValGesInit< double > & cParamMICMAC::Px2Moy()
{
   return Section_Terrain().IntervParalaxe().Val().Px2Moy();
}

const cTplValGesInit< double > & cParamMICMAC::Px2Moy()const 
{
   return Section_Terrain().IntervParalaxe().Val().Px2Moy();
}


double & cParamMICMAC::Px1IncCalc()
{
   return Section_Terrain().IntervParalaxe().Val().Px1IncCalc();
}

const double & cParamMICMAC::Px1IncCalc()const 
{
   return Section_Terrain().IntervParalaxe().Val().Px1IncCalc();
}


cTplValGesInit< double > & cParamMICMAC::Px1PropProf()
{
   return Section_Terrain().IntervParalaxe().Val().Px1PropProf();
}

const cTplValGesInit< double > & cParamMICMAC::Px1PropProf()const 
{
   return Section_Terrain().IntervParalaxe().Val().Px1PropProf();
}


cTplValGesInit< double > & cParamMICMAC::Px2IncCalc()
{
   return Section_Terrain().IntervParalaxe().Val().Px2IncCalc();
}

const cTplValGesInit< double > & cParamMICMAC::Px2IncCalc()const 
{
   return Section_Terrain().IntervParalaxe().Val().Px2IncCalc();
}


cTplValGesInit< double > & cParamMICMAC::Px1IncZonage()
{
   return Section_Terrain().IntervParalaxe().Val().Px1IncZonage();
}

const cTplValGesInit< double > & cParamMICMAC::Px1IncZonage()const 
{
   return Section_Terrain().IntervParalaxe().Val().Px1IncZonage();
}


cTplValGesInit< double > & cParamMICMAC::Px2IncZonage()
{
   return Section_Terrain().IntervParalaxe().Val().Px2IncZonage();
}

const cTplValGesInit< double > & cParamMICMAC::Px2IncZonage()const 
{
   return Section_Terrain().IntervParalaxe().Val().Px2IncZonage();
}


cTplValGesInit< cIntervParalaxe > & cParamMICMAC::IntervParalaxe()
{
   return Section_Terrain().IntervParalaxe();
}

const cTplValGesInit< cIntervParalaxe > & cParamMICMAC::IntervParalaxe()const 
{
   return Section_Terrain().IntervParalaxe();
}


std::string & cParamMICMAC::NameNuageXML()
{
   return Section_Terrain().NuageXMLInit().Val().NameNuageXML();
}

const std::string & cParamMICMAC::NameNuageXML()const 
{
   return Section_Terrain().NuageXMLInit().Val().NameNuageXML();
}


cTplValGesInit< bool > & cParamMICMAC::CanAdaptGeom()
{
   return Section_Terrain().NuageXMLInit().Val().CanAdaptGeom();
}

const cTplValGesInit< bool > & cParamMICMAC::CanAdaptGeom()const 
{
   return Section_Terrain().NuageXMLInit().Val().CanAdaptGeom();
}


cTplValGesInit< cNuageXMLInit > & cParamMICMAC::NuageXMLInit()
{
   return Section_Terrain().NuageXMLInit();
}

const cTplValGesInit< cNuageXMLInit > & cParamMICMAC::NuageXMLInit()const 
{
   return Section_Terrain().NuageXMLInit();
}


double & cParamMICMAC::MulZMin()
{
   return Section_Terrain().IntervSpecialZInv().Val().MulZMin();
}

const double & cParamMICMAC::MulZMin()const 
{
   return Section_Terrain().IntervSpecialZInv().Val().MulZMin();
}


double & cParamMICMAC::MulZMax()
{
   return Section_Terrain().IntervSpecialZInv().Val().MulZMax();
}

const double & cParamMICMAC::MulZMax()const 
{
   return Section_Terrain().IntervSpecialZInv().Val().MulZMax();
}


cTplValGesInit< cIntervSpecialZInv > & cParamMICMAC::IntervSpecialZInv()
{
   return Section_Terrain().IntervSpecialZInv();
}

const cTplValGesInit< cIntervSpecialZInv > & cParamMICMAC::IntervSpecialZInv()const 
{
   return Section_Terrain().IntervSpecialZInv();
}


cTplValGesInit< bool > & cParamMICMAC::GeoRefAutoRoundResol()
{
   return Section_Terrain().GeoRefAutoRoundResol();
}

const cTplValGesInit< bool > & cParamMICMAC::GeoRefAutoRoundResol()const 
{
   return Section_Terrain().GeoRefAutoRoundResol();
}


cTplValGesInit< bool > & cParamMICMAC::GeoRefAutoRoundBox()
{
   return Section_Terrain().GeoRefAutoRoundBox();
}

const cTplValGesInit< bool > & cParamMICMAC::GeoRefAutoRoundBox()const 
{
   return Section_Terrain().GeoRefAutoRoundBox();
}


cTplValGesInit< Box2dr > & cParamMICMAC::BoxTerrain()
{
   return Section_Terrain().Planimetrie().Val().BoxTerrain();
}

const cTplValGesInit< Box2dr > & cParamMICMAC::BoxTerrain()const 
{
   return Section_Terrain().Planimetrie().Val().BoxTerrain();
}


std::list< cListePointsInclus > & cParamMICMAC::ListePointsInclus()
{
   return Section_Terrain().Planimetrie().Val().ListePointsInclus();
}

const std::list< cListePointsInclus > & cParamMICMAC::ListePointsInclus()const 
{
   return Section_Terrain().Planimetrie().Val().ListePointsInclus();
}


cTplValGesInit< double > & cParamMICMAC::RatioResolImage()
{
   return Section_Terrain().Planimetrie().Val().RatioResolImage();
}

const cTplValGesInit< double > & cParamMICMAC::RatioResolImage()const 
{
   return Section_Terrain().Planimetrie().Val().RatioResolImage();
}


cTplValGesInit< double > & cParamMICMAC::ResolutionTerrain()
{
   return Section_Terrain().Planimetrie().Val().ResolutionTerrain();
}

const cTplValGesInit< double > & cParamMICMAC::ResolutionTerrain()const 
{
   return Section_Terrain().Planimetrie().Val().ResolutionTerrain();
}


cTplValGesInit< bool > & cParamMICMAC::RoundSpecifiedRT()
{
   return Section_Terrain().Planimetrie().Val().RoundSpecifiedRT();
}

const cTplValGesInit< bool > & cParamMICMAC::RoundSpecifiedRT()const 
{
   return Section_Terrain().Planimetrie().Val().RoundSpecifiedRT();
}


cTplValGesInit< std::string > & cParamMICMAC::FilterEstimTerrain()
{
   return Section_Terrain().Planimetrie().Val().FilterEstimTerrain();
}

const cTplValGesInit< std::string > & cParamMICMAC::FilterEstimTerrain()const 
{
   return Section_Terrain().Planimetrie().Val().FilterEstimTerrain();
}


cTplValGesInit< std::string > & cParamMICMAC::FileBoxMasqIsBoxTer()
{
   return Section_Terrain().Planimetrie().Val().MasqueTerrain().Val().FileBoxMasqIsBoxTer();
}

const cTplValGesInit< std::string > & cParamMICMAC::FileBoxMasqIsBoxTer()const 
{
   return Section_Terrain().Planimetrie().Val().MasqueTerrain().Val().FileBoxMasqIsBoxTer();
}


std::string & cParamMICMAC::MT_Image()
{
   return Section_Terrain().Planimetrie().Val().MasqueTerrain().Val().MT_Image();
}

const std::string & cParamMICMAC::MT_Image()const 
{
   return Section_Terrain().Planimetrie().Val().MasqueTerrain().Val().MT_Image();
}


std::string & cParamMICMAC::MT_Xml()
{
   return Section_Terrain().Planimetrie().Val().MasqueTerrain().Val().MT_Xml();
}

const std::string & cParamMICMAC::MT_Xml()const 
{
   return Section_Terrain().Planimetrie().Val().MasqueTerrain().Val().MT_Xml();
}


cTplValGesInit< cMasqueTerrain > & cParamMICMAC::MasqueTerrain()
{
   return Section_Terrain().Planimetrie().Val().MasqueTerrain();
}

const cTplValGesInit< cMasqueTerrain > & cParamMICMAC::MasqueTerrain()const 
{
   return Section_Terrain().Planimetrie().Val().MasqueTerrain();
}


cTplValGesInit< double > & cParamMICMAC::RecouvrementMinimal()
{
   return Section_Terrain().Planimetrie().Val().RecouvrementMinimal();
}

const cTplValGesInit< double > & cParamMICMAC::RecouvrementMinimal()const 
{
   return Section_Terrain().Planimetrie().Val().RecouvrementMinimal();
}


cTplValGesInit< cPlanimetrie > & cParamMICMAC::Planimetrie()
{
   return Section_Terrain().Planimetrie();
}

const cTplValGesInit< cPlanimetrie > & cParamMICMAC::Planimetrie()const 
{
   return Section_Terrain().Planimetrie();
}


cTplValGesInit< std::string > & cParamMICMAC::FileOriMnt()
{
   return Section_Terrain().FileOriMnt();
}

const cTplValGesInit< std::string > & cParamMICMAC::FileOriMnt()const 
{
   return Section_Terrain().FileOriMnt();
}


cTplValGesInit< double > & cParamMICMAC::EnergieExpCorrel()
{
   return Section_Terrain().RugositeMNT().Val().EnergieExpCorrel();
}

const cTplValGesInit< double > & cParamMICMAC::EnergieExpCorrel()const 
{
   return Section_Terrain().RugositeMNT().Val().EnergieExpCorrel();
}


cTplValGesInit< double > & cParamMICMAC::EnergieExpRegulPlani()
{
   return Section_Terrain().RugositeMNT().Val().EnergieExpRegulPlani();
}

const cTplValGesInit< double > & cParamMICMAC::EnergieExpRegulPlani()const 
{
   return Section_Terrain().RugositeMNT().Val().EnergieExpRegulPlani();
}


cTplValGesInit< double > & cParamMICMAC::EnergieExpRegulAlti()
{
   return Section_Terrain().RugositeMNT().Val().EnergieExpRegulAlti();
}

const cTplValGesInit< double > & cParamMICMAC::EnergieExpRegulAlti()const 
{
   return Section_Terrain().RugositeMNT().Val().EnergieExpRegulAlti();
}


cTplValGesInit< cRugositeMNT > & cParamMICMAC::RugositeMNT()
{
   return Section_Terrain().RugositeMNT();
}

const cTplValGesInit< cRugositeMNT > & cParamMICMAC::RugositeMNT()const 
{
   return Section_Terrain().RugositeMNT();
}


cSection_Terrain & cParamMICMAC::Section_Terrain()
{
   return mSection_Terrain;
}

const cSection_Terrain & cParamMICMAC::Section_Terrain()const 
{
   return mSection_Terrain;
}


cTplValGesInit< int > & cParamMICMAC::BordImage()
{
   return Section_PriseDeVue().BordImage();
}

const cTplValGesInit< int > & cParamMICMAC::BordImage()const 
{
   return Section_PriseDeVue().BordImage();
}


cTplValGesInit< bool > & cParamMICMAC::ConvertToSameOriPtTgtLoc()
{
   return Section_PriseDeVue().ConvertToSameOriPtTgtLoc();
}

const cTplValGesInit< bool > & cParamMICMAC::ConvertToSameOriPtTgtLoc()const 
{
   return Section_PriseDeVue().ConvertToSameOriPtTgtLoc();
}


cTplValGesInit< int > & cParamMICMAC::ValSpecNotImage()
{
   return Section_PriseDeVue().ValSpecNotImage();
}

const cTplValGesInit< int > & cParamMICMAC::ValSpecNotImage()const 
{
   return Section_PriseDeVue().ValSpecNotImage();
}


cTplValGesInit< std::string > & cParamMICMAC::PrefixMasqImRes()
{
   return Section_PriseDeVue().PrefixMasqImRes();
}

const cTplValGesInit< std::string > & cParamMICMAC::PrefixMasqImRes()const 
{
   return Section_PriseDeVue().PrefixMasqImRes();
}


cTplValGesInit< std::string > & cParamMICMAC::DirMasqueImages()
{
   return Section_PriseDeVue().DirMasqueImages();
}

const cTplValGesInit< std::string > & cParamMICMAC::DirMasqueImages()const 
{
   return Section_PriseDeVue().DirMasqueImages();
}


std::list< cMasqImageIn > & cParamMICMAC::MasqImageIn()
{
   return Section_PriseDeVue().MasqImageIn();
}

const std::list< cMasqImageIn > & cParamMICMAC::MasqImageIn()const 
{
   return Section_PriseDeVue().MasqImageIn();
}


std::list< cSpecFitrageImage > & cParamMICMAC::FiltreImageIn()
{
   return Section_PriseDeVue().FiltreImageIn();
}

const std::list< cSpecFitrageImage > & cParamMICMAC::FiltreImageIn()const 
{
   return Section_PriseDeVue().FiltreImageIn();
}


eModeGeomImage & cParamMICMAC::GeomImages()
{
   return Section_PriseDeVue().GeomImages();
}

const eModeGeomImage & cParamMICMAC::GeomImages()const 
{
   return Section_PriseDeVue().GeomImages();
}


std::string & cParamMICMAC::NomModule()
{
   return Section_PriseDeVue().ModuleGeomImage().Val().NomModule();
}

const std::string & cParamMICMAC::NomModule()const 
{
   return Section_PriseDeVue().ModuleGeomImage().Val().NomModule();
}


std::string & cParamMICMAC::NomGeometrie()
{
   return Section_PriseDeVue().ModuleGeomImage().Val().NomGeometrie();
}

const std::string & cParamMICMAC::NomGeometrie()const 
{
   return Section_PriseDeVue().ModuleGeomImage().Val().NomGeometrie();
}


cTplValGesInit< cModuleGeomImage > & cParamMICMAC::ModuleGeomImage()
{
   return Section_PriseDeVue().ModuleGeomImage();
}

const cTplValGesInit< cModuleGeomImage > & cParamMICMAC::ModuleGeomImage()const 
{
   return Section_PriseDeVue().ModuleGeomImage();
}


cTplValGesInit< std::string > & cParamMICMAC::Im1()
{
   return Section_PriseDeVue().Images().Im1();
}

const cTplValGesInit< std::string > & cParamMICMAC::Im1()const 
{
   return Section_PriseDeVue().Images().Im1();
}


cTplValGesInit< std::string > & cParamMICMAC::Im2()
{
   return Section_PriseDeVue().Images().Im2();
}

const cTplValGesInit< std::string > & cParamMICMAC::Im2()const 
{
   return Section_PriseDeVue().Images().Im2();
}


std::string & cParamMICMAC::I2FromI1Key()
{
   return Section_PriseDeVue().Images().FCND_CalcIm2fromIm1().Val().I2FromI1Key();
}

const std::string & cParamMICMAC::I2FromI1Key()const 
{
   return Section_PriseDeVue().Images().FCND_CalcIm2fromIm1().Val().I2FromI1Key();
}


bool & cParamMICMAC::I2FromI1SensDirect()
{
   return Section_PriseDeVue().Images().FCND_CalcIm2fromIm1().Val().I2FromI1SensDirect();
}

const bool & cParamMICMAC::I2FromI1SensDirect()const 
{
   return Section_PriseDeVue().Images().FCND_CalcIm2fromIm1().Val().I2FromI1SensDirect();
}


cTplValGesInit< cFCND_CalcIm2fromIm1 > & cParamMICMAC::FCND_CalcIm2fromIm1()
{
   return Section_PriseDeVue().Images().FCND_CalcIm2fromIm1();
}

const cTplValGesInit< cFCND_CalcIm2fromIm1 > & cParamMICMAC::FCND_CalcIm2fromIm1()const 
{
   return Section_PriseDeVue().Images().FCND_CalcIm2fromIm1();
}


std::list< std::string > & cParamMICMAC::ImPat()
{
   return Section_PriseDeVue().Images().ImPat();
}

const std::list< std::string > & cParamMICMAC::ImPat()const 
{
   return Section_PriseDeVue().Images().ImPat();
}


cTplValGesInit< std::string > & cParamMICMAC::ImageSecByCAWSI()
{
   return Section_PriseDeVue().Images().ImageSecByCAWSI();
}

const cTplValGesInit< std::string > & cParamMICMAC::ImageSecByCAWSI()const 
{
   return Section_PriseDeVue().Images().ImageSecByCAWSI();
}


std::string & cParamMICMAC::Key()
{
   return Section_PriseDeVue().Images().ImSecCalcApero().Val().Key();
}

const std::string & cParamMICMAC::Key()const 
{
   return Section_PriseDeVue().Images().ImSecCalcApero().Val().Key();
}


cTplValGesInit< int > & cParamMICMAC::Nb()
{
   return Section_PriseDeVue().Images().ImSecCalcApero().Val().Nb();
}

const cTplValGesInit< int > & cParamMICMAC::Nb()const 
{
   return Section_PriseDeVue().Images().ImSecCalcApero().Val().Nb();
}


cTplValGesInit< int > & cParamMICMAC::NbMin()
{
   return Section_PriseDeVue().Images().ImSecCalcApero().Val().NbMin();
}

const cTplValGesInit< int > & cParamMICMAC::NbMin()const 
{
   return Section_PriseDeVue().Images().ImSecCalcApero().Val().NbMin();
}


cTplValGesInit< int > & cParamMICMAC::NbMax()
{
   return Section_PriseDeVue().Images().ImSecCalcApero().Val().NbMax();
}

const cTplValGesInit< int > & cParamMICMAC::NbMax()const 
{
   return Section_PriseDeVue().Images().ImSecCalcApero().Val().NbMax();
}


cTplValGesInit< eOnEmptyImSecApero > & cParamMICMAC::OnEmpty()
{
   return Section_PriseDeVue().Images().ImSecCalcApero().Val().OnEmpty();
}

const cTplValGesInit< eOnEmptyImSecApero > & cParamMICMAC::OnEmpty()const 
{
   return Section_PriseDeVue().Images().ImSecCalcApero().Val().OnEmpty();
}


cTplValGesInit< cImSecCalcApero > & cParamMICMAC::ImSecCalcApero()
{
   return Section_PriseDeVue().Images().ImSecCalcApero();
}

const cTplValGesInit< cImSecCalcApero > & cParamMICMAC::ImSecCalcApero()const 
{
   return Section_PriseDeVue().Images().ImSecCalcApero();
}


cTplValGesInit< cParamGenereStrVois > & cParamMICMAC::RelGlobSelecteur()
{
   return Section_PriseDeVue().Images().RelGlobSelecteur();
}

const cTplValGesInit< cParamGenereStrVois > & cParamMICMAC::RelGlobSelecteur()const 
{
   return Section_PriseDeVue().Images().RelGlobSelecteur();
}


cTplValGesInit< cNameFilter > & cParamMICMAC::Filter()
{
   return Section_PriseDeVue().Images().Filter();
}

const cTplValGesInit< cNameFilter > & cParamMICMAC::Filter()const 
{
   return Section_PriseDeVue().Images().Filter();
}


double & cParamMICMAC::RecouvrMin()
{
   return Section_PriseDeVue().Images().AutoSelectionneImSec().Val().RecouvrMin();
}

const double & cParamMICMAC::RecouvrMin()const 
{
   return Section_PriseDeVue().Images().AutoSelectionneImSec().Val().RecouvrMin();
}


cTplValGesInit< cAutoSelectionneImSec > & cParamMICMAC::AutoSelectionneImSec()
{
   return Section_PriseDeVue().Images().AutoSelectionneImSec();
}

const cTplValGesInit< cAutoSelectionneImSec > & cParamMICMAC::AutoSelectionneImSec()const 
{
   return Section_PriseDeVue().Images().AutoSelectionneImSec();
}


cTplValGesInit< cListImByDelta > & cParamMICMAC::ImSecByDelta()
{
   return Section_PriseDeVue().Images().ImSecByDelta();
}

const cTplValGesInit< cListImByDelta > & cParamMICMAC::ImSecByDelta()const 
{
   return Section_PriseDeVue().Images().ImSecByDelta();
}


cTplValGesInit< std::string > & cParamMICMAC::Im3Superp()
{
   return Section_PriseDeVue().Images().Im3Superp();
}

const cTplValGesInit< std::string > & cParamMICMAC::Im3Superp()const 
{
   return Section_PriseDeVue().Images().Im3Superp();
}


cImages & cParamMICMAC::Images()
{
   return Section_PriseDeVue().Images();
}

const cImages & cParamMICMAC::Images()const 
{
   return Section_PriseDeVue().Images();
}


std::list< cNomsGeometrieImage > & cParamMICMAC::NomsGeometrieImage()
{
   return Section_PriseDeVue().NomsGeometrieImage();
}

const std::list< cNomsGeometrieImage > & cParamMICMAC::NomsGeometrieImage()const 
{
   return Section_PriseDeVue().NomsGeometrieImage();
}


std::string & cParamMICMAC::PatternSel()
{
   return Section_PriseDeVue().NomsHomomologues().Val().PatternSel();
}

const std::string & cParamMICMAC::PatternSel()const 
{
   return Section_PriseDeVue().NomsHomomologues().Val().PatternSel();
}


std::string & cParamMICMAC::PatNameGeom()
{
   return Section_PriseDeVue().NomsHomomologues().Val().PatNameGeom();
}

const std::string & cParamMICMAC::PatNameGeom()const 
{
   return Section_PriseDeVue().NomsHomomologues().Val().PatNameGeom();
}


cTplValGesInit< std::string > & cParamMICMAC::SeparateurHom()
{
   return Section_PriseDeVue().NomsHomomologues().Val().SeparateurHom();
}

const cTplValGesInit< std::string > & cParamMICMAC::SeparateurHom()const 
{
   return Section_PriseDeVue().NomsHomomologues().Val().SeparateurHom();
}


cTplValGesInit< cNomsHomomologues > & cParamMICMAC::NomsHomomologues()
{
   return Section_PriseDeVue().NomsHomomologues();
}

const cTplValGesInit< cNomsHomomologues > & cParamMICMAC::NomsHomomologues()const 
{
   return Section_PriseDeVue().NomsHomomologues();
}


cTplValGesInit< std::string > & cParamMICMAC::FCND_CalcHomFromI1I2()
{
   return Section_PriseDeVue().FCND_CalcHomFromI1I2();
}

const cTplValGesInit< std::string > & cParamMICMAC::FCND_CalcHomFromI1I2()const 
{
   return Section_PriseDeVue().FCND_CalcHomFromI1I2();
}


cTplValGesInit< bool > & cParamMICMAC::SingulariteInCorresp_I1I2()
{
   return Section_PriseDeVue().SingulariteInCorresp_I1I2();
}

const cTplValGesInit< bool > & cParamMICMAC::SingulariteInCorresp_I1I2()const 
{
   return Section_PriseDeVue().SingulariteInCorresp_I1I2();
}


cTplValGesInit< cMapName2Name > & cParamMICMAC::ClassEquivalenceImage()
{
   return Section_PriseDeVue().ClassEquivalenceImage();
}

const cTplValGesInit< cMapName2Name > & cParamMICMAC::ClassEquivalenceImage()const 
{
   return Section_PriseDeVue().ClassEquivalenceImage();
}


cSection_PriseDeVue & cParamMICMAC::Section_PriseDeVue()
{
   return mSection_PriseDeVue;
}

const cSection_PriseDeVue & cParamMICMAC::Section_PriseDeVue()const 
{
   return mSection_PriseDeVue;
}


cTplValGesInit< double > & cParamMICMAC::ExtensionIntervZ()
{
   return Section_MEC().ExtensionIntervZ();
}

const cTplValGesInit< double > & cParamMICMAC::ExtensionIntervZ()const 
{
   return Section_MEC().ExtensionIntervZ();
}


cTplValGesInit< bool > & cParamMICMAC::PasIsInPixel()
{
   return Section_MEC().PasIsInPixel();
}

const cTplValGesInit< bool > & cParamMICMAC::PasIsInPixel()const 
{
   return Section_MEC().PasIsInPixel();
}


cTplValGesInit< Box2dr > & cParamMICMAC::ProportionClipMEC()
{
   return Section_MEC().ProportionClipMEC();
}

const cTplValGesInit< Box2dr > & cParamMICMAC::ProportionClipMEC()const 
{
   return Section_MEC().ProportionClipMEC();
}


cTplValGesInit< bool > & cParamMICMAC::ClipMecIsProp()
{
   return Section_MEC().ClipMecIsProp();
}

const cTplValGesInit< bool > & cParamMICMAC::ClipMecIsProp()const 
{
   return Section_MEC().ClipMecIsProp();
}


cTplValGesInit< double > & cParamMICMAC::ZoomClipMEC()
{
   return Section_MEC().ZoomClipMEC();
}

const cTplValGesInit< double > & cParamMICMAC::ZoomClipMEC()const 
{
   return Section_MEC().ZoomClipMEC();
}


cTplValGesInit< int > & cParamMICMAC::NbMinImagesVisibles()
{
   return Section_MEC().NbMinImagesVisibles();
}

const cTplValGesInit< int > & cParamMICMAC::NbMinImagesVisibles()const 
{
   return Section_MEC().NbMinImagesVisibles();
}


cTplValGesInit< bool > & cParamMICMAC::OneDefCorAllPxDefCor()
{
   return Section_MEC().OneDefCorAllPxDefCor();
}

const cTplValGesInit< bool > & cParamMICMAC::OneDefCorAllPxDefCor()const 
{
   return Section_MEC().OneDefCorAllPxDefCor();
}


cTplValGesInit< int > & cParamMICMAC::ZoomBeginODC_APDC()
{
   return Section_MEC().ZoomBeginODC_APDC();
}

const cTplValGesInit< int > & cParamMICMAC::ZoomBeginODC_APDC()const 
{
   return Section_MEC().ZoomBeginODC_APDC();
}


cTplValGesInit< double > & cParamMICMAC::DefCorrelation()
{
   return Section_MEC().DefCorrelation();
}

const cTplValGesInit< double > & cParamMICMAC::DefCorrelation()const 
{
   return Section_MEC().DefCorrelation();
}


cTplValGesInit< bool > & cParamMICMAC::ReprojPixelNoVal()
{
   return Section_MEC().ReprojPixelNoVal();
}

const cTplValGesInit< bool > & cParamMICMAC::ReprojPixelNoVal()const 
{
   return Section_MEC().ReprojPixelNoVal();
}


cTplValGesInit< double > & cParamMICMAC::EpsilonCorrelation()
{
   return Section_MEC().EpsilonCorrelation();
}

const cTplValGesInit< double > & cParamMICMAC::EpsilonCorrelation()const 
{
   return Section_MEC().EpsilonCorrelation();
}


int & cParamMICMAC::FreqEchantPtsI()
{
   return Section_MEC().EchantillonagePtsInterets().Val().FreqEchantPtsI();
}

const int & cParamMICMAC::FreqEchantPtsI()const 
{
   return Section_MEC().EchantillonagePtsInterets().Val().FreqEchantPtsI();
}


eTypeModeEchantPtsI & cParamMICMAC::ModeEchantPtsI()
{
   return Section_MEC().EchantillonagePtsInterets().Val().ModeEchantPtsI();
}

const eTypeModeEchantPtsI & cParamMICMAC::ModeEchantPtsI()const 
{
   return Section_MEC().EchantillonagePtsInterets().Val().ModeEchantPtsI();
}


cTplValGesInit< std::string > & cParamMICMAC::KeyCommandeExterneInteret()
{
   return Section_MEC().EchantillonagePtsInterets().Val().KeyCommandeExterneInteret();
}

const cTplValGesInit< std::string > & cParamMICMAC::KeyCommandeExterneInteret()const 
{
   return Section_MEC().EchantillonagePtsInterets().Val().KeyCommandeExterneInteret();
}


cTplValGesInit< int > & cParamMICMAC::SzVAutoCorrel()
{
   return Section_MEC().EchantillonagePtsInterets().Val().SzVAutoCorrel();
}

const cTplValGesInit< int > & cParamMICMAC::SzVAutoCorrel()const 
{
   return Section_MEC().EchantillonagePtsInterets().Val().SzVAutoCorrel();
}


cTplValGesInit< double > & cParamMICMAC::EstmBrAutoCorrel()
{
   return Section_MEC().EchantillonagePtsInterets().Val().EstmBrAutoCorrel();
}

const cTplValGesInit< double > & cParamMICMAC::EstmBrAutoCorrel()const 
{
   return Section_MEC().EchantillonagePtsInterets().Val().EstmBrAutoCorrel();
}


cTplValGesInit< double > & cParamMICMAC::SeuilLambdaAutoCorrel()
{
   return Section_MEC().EchantillonagePtsInterets().Val().SeuilLambdaAutoCorrel();
}

const cTplValGesInit< double > & cParamMICMAC::SeuilLambdaAutoCorrel()const 
{
   return Section_MEC().EchantillonagePtsInterets().Val().SeuilLambdaAutoCorrel();
}


cTplValGesInit< double > & cParamMICMAC::SeuilEcartTypeAutoCorrel()
{
   return Section_MEC().EchantillonagePtsInterets().Val().SeuilEcartTypeAutoCorrel();
}

const cTplValGesInit< double > & cParamMICMAC::SeuilEcartTypeAutoCorrel()const 
{
   return Section_MEC().EchantillonagePtsInterets().Val().SeuilEcartTypeAutoCorrel();
}


cTplValGesInit< double > & cParamMICMAC::RepartExclusion()
{
   return Section_MEC().EchantillonagePtsInterets().Val().RepartExclusion();
}

const cTplValGesInit< double > & cParamMICMAC::RepartExclusion()const 
{
   return Section_MEC().EchantillonagePtsInterets().Val().RepartExclusion();
}


cTplValGesInit< double > & cParamMICMAC::RepartEvitement()
{
   return Section_MEC().EchantillonagePtsInterets().Val().RepartEvitement();
}

const cTplValGesInit< double > & cParamMICMAC::RepartEvitement()const 
{
   return Section_MEC().EchantillonagePtsInterets().Val().RepartEvitement();
}


cTplValGesInit< cEchantillonagePtsInterets > & cParamMICMAC::EchantillonagePtsInterets()
{
   return Section_MEC().EchantillonagePtsInterets();
}

const cTplValGesInit< cEchantillonagePtsInterets > & cParamMICMAC::EchantillonagePtsInterets()const 
{
   return Section_MEC().EchantillonagePtsInterets();
}


cTplValGesInit< bool > & cParamMICMAC::ChantierFullImage1()
{
   return Section_MEC().ChantierFullImage1();
}

const cTplValGesInit< bool > & cParamMICMAC::ChantierFullImage1()const 
{
   return Section_MEC().ChantierFullImage1();
}


cTplValGesInit< bool > & cParamMICMAC::ChantierFullMaskImage1()
{
   return Section_MEC().ChantierFullMaskImage1();
}

const cTplValGesInit< bool > & cParamMICMAC::ChantierFullMaskImage1()const 
{
   return Section_MEC().ChantierFullMaskImage1();
}


cTplValGesInit< bool > & cParamMICMAC::ExportForMultiplePointsHomologues()
{
   return Section_MEC().ExportForMultiplePointsHomologues();
}

const cTplValGesInit< bool > & cParamMICMAC::ExportForMultiplePointsHomologues()const 
{
   return Section_MEC().ExportForMultiplePointsHomologues();
}


cTplValGesInit< double > & cParamMICMAC::CovLim()
{
   return Section_MEC().AdapteDynCov().Val().CovLim();
}

const cTplValGesInit< double > & cParamMICMAC::CovLim()const 
{
   return Section_MEC().AdapteDynCov().Val().CovLim();
}


cTplValGesInit< double > & cParamMICMAC::TermeDecr()
{
   return Section_MEC().AdapteDynCov().Val().TermeDecr();
}

const cTplValGesInit< double > & cParamMICMAC::TermeDecr()const 
{
   return Section_MEC().AdapteDynCov().Val().TermeDecr();
}


cTplValGesInit< int > & cParamMICMAC::SzRef()
{
   return Section_MEC().AdapteDynCov().Val().SzRef();
}

const cTplValGesInit< int > & cParamMICMAC::SzRef()const 
{
   return Section_MEC().AdapteDynCov().Val().SzRef();
}


cTplValGesInit< double > & cParamMICMAC::ValRef()
{
   return Section_MEC().AdapteDynCov().Val().ValRef();
}

const cTplValGesInit< double > & cParamMICMAC::ValRef()const 
{
   return Section_MEC().AdapteDynCov().Val().ValRef();
}


cTplValGesInit< cAdapteDynCov > & cParamMICMAC::AdapteDynCov()
{
   return Section_MEC().AdapteDynCov();
}

const cTplValGesInit< cAdapteDynCov > & cParamMICMAC::AdapteDynCov()const 
{
   return Section_MEC().AdapteDynCov();
}


cTplValGesInit< cMMUseMasq3D > & cParamMICMAC::MMUseMasq3D()
{
   return Section_MEC().MMUseMasq3D();
}

const cTplValGesInit< cMMUseMasq3D > & cParamMICMAC::MMUseMasq3D()const 
{
   return Section_MEC().MMUseMasq3D();
}


std::list< cEtapeMEC > & cParamMICMAC::EtapeMEC()
{
   return Section_MEC().EtapeMEC();
}

const std::list< cEtapeMEC > & cParamMICMAC::EtapeMEC()const 
{
   return Section_MEC().EtapeMEC();
}


std::list< cTypePyramImage > & cParamMICMAC::TypePyramImage()
{
   return Section_MEC().TypePyramImage();
}

const std::list< cTypePyramImage > & cParamMICMAC::TypePyramImage()const 
{
   return Section_MEC().TypePyramImage();
}


cTplValGesInit< bool > & cParamMICMAC::HighPrecPyrIm()
{
   return Section_MEC().HighPrecPyrIm();
}

const cTplValGesInit< bool > & cParamMICMAC::HighPrecPyrIm()const 
{
   return Section_MEC().HighPrecPyrIm();
}


cTplValGesInit< bool > & cParamMICMAC::Correl16Bits()
{
   return Section_MEC().Correl16Bits();
}

const cTplValGesInit< bool > & cParamMICMAC::Correl16Bits()const 
{
   return Section_MEC().Correl16Bits();
}


cSection_MEC & cParamMICMAC::Section_MEC()
{
   return mSection_MEC;
}

const cSection_MEC & cParamMICMAC::Section_MEC()const 
{
   return mSection_MEC;
}


cTplValGesInit< bool > & cParamMICMAC::Use_MM_EtatAvancement()
{
   return Section_Results().Use_MM_EtatAvancement();
}

const cTplValGesInit< bool > & cParamMICMAC::Use_MM_EtatAvancement()const 
{
   return Section_Results().Use_MM_EtatAvancement();
}


cTplValGesInit< bool > & cParamMICMAC::ButDoPyram()
{
   return Section_Results().DoNothingBut().Val().ButDoPyram();
}

const cTplValGesInit< bool > & cParamMICMAC::ButDoPyram()const 
{
   return Section_Results().DoNothingBut().Val().ButDoPyram();
}


cTplValGesInit< bool > & cParamMICMAC::ButDoMasqIm()
{
   return Section_Results().DoNothingBut().Val().ButDoMasqIm();
}

const cTplValGesInit< bool > & cParamMICMAC::ButDoMasqIm()const 
{
   return Section_Results().DoNothingBut().Val().ButDoMasqIm();
}


cTplValGesInit< bool > & cParamMICMAC::ButDoMemPart()
{
   return Section_Results().DoNothingBut().Val().ButDoMemPart();
}

const cTplValGesInit< bool > & cParamMICMAC::ButDoMemPart()const 
{
   return Section_Results().DoNothingBut().Val().ButDoMemPart();
}


cTplValGesInit< bool > & cParamMICMAC::ButDoTA()
{
   return Section_Results().DoNothingBut().Val().ButDoTA();
}

const cTplValGesInit< bool > & cParamMICMAC::ButDoTA()const 
{
   return Section_Results().DoNothingBut().Val().ButDoTA();
}


cTplValGesInit< bool > & cParamMICMAC::ButDoMasqueChantier()
{
   return Section_Results().DoNothingBut().Val().ButDoMasqueChantier();
}

const cTplValGesInit< bool > & cParamMICMAC::ButDoMasqueChantier()const 
{
   return Section_Results().DoNothingBut().Val().ButDoMasqueChantier();
}


cTplValGesInit< bool > & cParamMICMAC::ButDoOriMNT()
{
   return Section_Results().DoNothingBut().Val().ButDoOriMNT();
}

const cTplValGesInit< bool > & cParamMICMAC::ButDoOriMNT()const 
{
   return Section_Results().DoNothingBut().Val().ButDoOriMNT();
}


cTplValGesInit< bool > & cParamMICMAC::ButDoMTDNuage()
{
   return Section_Results().DoNothingBut().Val().ButDoMTDNuage();
}

const cTplValGesInit< bool > & cParamMICMAC::ButDoMTDNuage()const 
{
   return Section_Results().DoNothingBut().Val().ButDoMTDNuage();
}


cTplValGesInit< bool > & cParamMICMAC::ButDoFDC()
{
   return Section_Results().DoNothingBut().Val().ButDoFDC();
}

const cTplValGesInit< bool > & cParamMICMAC::ButDoFDC()const 
{
   return Section_Results().DoNothingBut().Val().ButDoFDC();
}


cTplValGesInit< bool > & cParamMICMAC::ButDoExtendParam()
{
   return Section_Results().DoNothingBut().Val().ButDoExtendParam();
}

const cTplValGesInit< bool > & cParamMICMAC::ButDoExtendParam()const 
{
   return Section_Results().DoNothingBut().Val().ButDoExtendParam();
}


cTplValGesInit< bool > & cParamMICMAC::ButDoGenCorPxTransv()
{
   return Section_Results().DoNothingBut().Val().ButDoGenCorPxTransv();
}

const cTplValGesInit< bool > & cParamMICMAC::ButDoGenCorPxTransv()const 
{
   return Section_Results().DoNothingBut().Val().ButDoGenCorPxTransv();
}


cTplValGesInit< bool > & cParamMICMAC::ButDoPartiesCachees()
{
   return Section_Results().DoNothingBut().Val().ButDoPartiesCachees();
}

const cTplValGesInit< bool > & cParamMICMAC::ButDoPartiesCachees()const 
{
   return Section_Results().DoNothingBut().Val().ButDoPartiesCachees();
}


cTplValGesInit< bool > & cParamMICMAC::ButDoOrtho()
{
   return Section_Results().DoNothingBut().Val().ButDoOrtho();
}

const cTplValGesInit< bool > & cParamMICMAC::ButDoOrtho()const 
{
   return Section_Results().DoNothingBut().Val().ButDoOrtho();
}


cTplValGesInit< bool > & cParamMICMAC::ButDoSimul()
{
   return Section_Results().DoNothingBut().Val().ButDoSimul();
}

const cTplValGesInit< bool > & cParamMICMAC::ButDoSimul()const 
{
   return Section_Results().DoNothingBut().Val().ButDoSimul();
}


cTplValGesInit< bool > & cParamMICMAC::ButDoRedrLocAnam()
{
   return Section_Results().DoNothingBut().Val().ButDoRedrLocAnam();
}

const cTplValGesInit< bool > & cParamMICMAC::ButDoRedrLocAnam()const 
{
   return Section_Results().DoNothingBut().Val().ButDoRedrLocAnam();
}


cTplValGesInit< cDoNothingBut > & cParamMICMAC::DoNothingBut()
{
   return Section_Results().DoNothingBut();
}

const cTplValGesInit< cDoNothingBut > & cParamMICMAC::DoNothingBut()const 
{
   return Section_Results().DoNothingBut();
}


cTplValGesInit< int > & cParamMICMAC::Paral_Pc_IdProcess()
{
   return Section_Results().Paral_Pc_IdProcess();
}

const cTplValGesInit< int > & cParamMICMAC::Paral_Pc_IdProcess()const 
{
   return Section_Results().Paral_Pc_IdProcess();
}


cTplValGesInit< int > & cParamMICMAC::Paral_Pc_NbProcess()
{
   return Section_Results().Paral_Pc_NbProcess();
}

const cTplValGesInit< int > & cParamMICMAC::Paral_Pc_NbProcess()const 
{
   return Section_Results().Paral_Pc_NbProcess();
}


cTplValGesInit< double > & cParamMICMAC::X_DirPlanInterFaisceau()
{
   return Section_Results().X_DirPlanInterFaisceau();
}

const cTplValGesInit< double > & cParamMICMAC::X_DirPlanInterFaisceau()const 
{
   return Section_Results().X_DirPlanInterFaisceau();
}


cTplValGesInit< double > & cParamMICMAC::Y_DirPlanInterFaisceau()
{
   return Section_Results().Y_DirPlanInterFaisceau();
}

const cTplValGesInit< double > & cParamMICMAC::Y_DirPlanInterFaisceau()const 
{
   return Section_Results().Y_DirPlanInterFaisceau();
}


cTplValGesInit< double > & cParamMICMAC::Z_DirPlanInterFaisceau()
{
   return Section_Results().Z_DirPlanInterFaisceau();
}

const cTplValGesInit< double > & cParamMICMAC::Z_DirPlanInterFaisceau()const 
{
   return Section_Results().Z_DirPlanInterFaisceau();
}


eModeGeomMNT & cParamMICMAC::GeomMNT()
{
   return Section_Results().GeomMNT();
}

const eModeGeomMNT & cParamMICMAC::GeomMNT()const 
{
   return Section_Results().GeomMNT();
}


cTplValGesInit< cSectionSimulation > & cParamMICMAC::SectionSimulation()
{
   return Section_Results().SectionSimulation();
}

const cTplValGesInit< cSectionSimulation > & cParamMICMAC::SectionSimulation()const 
{
   return Section_Results().SectionSimulation();
}


cTplValGesInit< bool > & cParamMICMAC::Prio2OwnAltisolForEmprise()
{
   return Section_Results().Prio2OwnAltisolForEmprise();
}

const cTplValGesInit< bool > & cParamMICMAC::Prio2OwnAltisolForEmprise()const 
{
   return Section_Results().Prio2OwnAltisolForEmprise();
}


cTplValGesInit< bool > & cParamMICMAC::UnUseAnamXCste()
{
   return Section_Results().AnamorphoseGeometrieMNT().Val().UnUseAnamXCste();
}

const cTplValGesInit< bool > & cParamMICMAC::UnUseAnamXCste()const 
{
   return Section_Results().AnamorphoseGeometrieMNT().Val().UnUseAnamXCste();
}


std::string & cParamMICMAC::NameFile()
{
   return Section_Results().AnamorphoseGeometrieMNT().Val().AnamSurfaceAnalytique().Val().NameFile();
}

const std::string & cParamMICMAC::NameFile()const 
{
   return Section_Results().AnamorphoseGeometrieMNT().Val().AnamSurfaceAnalytique().Val().NameFile();
}


std::string & cParamMICMAC::Id()
{
   return Section_Results().AnamorphoseGeometrieMNT().Val().AnamSurfaceAnalytique().Val().Id();
}

const std::string & cParamMICMAC::Id()const 
{
   return Section_Results().AnamorphoseGeometrieMNT().Val().AnamSurfaceAnalytique().Val().Id();
}


cTplValGesInit< cAnamSurfaceAnalytique > & cParamMICMAC::AnamSurfaceAnalytique()
{
   return Section_Results().AnamorphoseGeometrieMNT().Val().AnamSurfaceAnalytique();
}

const cTplValGesInit< cAnamSurfaceAnalytique > & cParamMICMAC::AnamSurfaceAnalytique()const 
{
   return Section_Results().AnamorphoseGeometrieMNT().Val().AnamSurfaceAnalytique();
}


cTplValGesInit< int > & cParamMICMAC::AnamDeZoomMasq()
{
   return Section_Results().AnamorphoseGeometrieMNT().Val().AnamDeZoomMasq();
}

const cTplValGesInit< int > & cParamMICMAC::AnamDeZoomMasq()const 
{
   return Section_Results().AnamorphoseGeometrieMNT().Val().AnamDeZoomMasq();
}


cTplValGesInit< double > & cParamMICMAC::AnamLimAngleVisib()
{
   return Section_Results().AnamorphoseGeometrieMNT().Val().AnamLimAngleVisib();
}

const cTplValGesInit< double > & cParamMICMAC::AnamLimAngleVisib()const 
{
   return Section_Results().AnamorphoseGeometrieMNT().Val().AnamLimAngleVisib();
}


cTplValGesInit< double > & cParamMICMAC::DynIncid()
{
   return Section_Results().AnamorphoseGeometrieMNT().Val().MakeMaskImNadir().Val().DynIncid();
}

const cTplValGesInit< double > & cParamMICMAC::DynIncid()const 
{
   return Section_Results().AnamorphoseGeometrieMNT().Val().MakeMaskImNadir().Val().DynIncid();
}


cTplValGesInit< bool > & cParamMICMAC::MakeAlsoMaskTerrain()
{
   return Section_Results().AnamorphoseGeometrieMNT().Val().MakeMaskImNadir().Val().MakeAlsoMaskTerrain();
}

const cTplValGesInit< bool > & cParamMICMAC::MakeAlsoMaskTerrain()const 
{
   return Section_Results().AnamorphoseGeometrieMNT().Val().MakeMaskImNadir().Val().MakeAlsoMaskTerrain();
}


int & cParamMICMAC::KBest()
{
   return Section_Results().AnamorphoseGeometrieMNT().Val().MakeMaskImNadir().Val().KBest();
}

const int & cParamMICMAC::KBest()const 
{
   return Section_Results().AnamorphoseGeometrieMNT().Val().MakeMaskImNadir().Val().KBest();
}


cTplValGesInit< double > & cParamMICMAC::IncertAngle()
{
   return Section_Results().AnamorphoseGeometrieMNT().Val().MakeMaskImNadir().Val().IncertAngle();
}

const cTplValGesInit< double > & cParamMICMAC::IncertAngle()const 
{
   return Section_Results().AnamorphoseGeometrieMNT().Val().MakeMaskImNadir().Val().IncertAngle();
}


cTplValGesInit< int > & cParamMICMAC::Dilat32()
{
   return Section_Results().AnamorphoseGeometrieMNT().Val().MakeMaskImNadir().Val().Dilat32();
}

const cTplValGesInit< int > & cParamMICMAC::Dilat32()const 
{
   return Section_Results().AnamorphoseGeometrieMNT().Val().MakeMaskImNadir().Val().Dilat32();
}


cTplValGesInit< int > & cParamMICMAC::Erod32()
{
   return Section_Results().AnamorphoseGeometrieMNT().Val().MakeMaskImNadir().Val().Erod32();
}

const cTplValGesInit< int > & cParamMICMAC::Erod32()const 
{
   return Section_Results().AnamorphoseGeometrieMNT().Val().MakeMaskImNadir().Val().Erod32();
}


cTplValGesInit< cMakeMaskImNadir > & cParamMICMAC::MakeMaskImNadir()
{
   return Section_Results().AnamorphoseGeometrieMNT().Val().MakeMaskImNadir();
}

const cTplValGesInit< cMakeMaskImNadir > & cParamMICMAC::MakeMaskImNadir()const 
{
   return Section_Results().AnamorphoseGeometrieMNT().Val().MakeMaskImNadir();
}


cTplValGesInit< cAnamorphoseGeometrieMNT > & cParamMICMAC::AnamorphoseGeometrieMNT()
{
   return Section_Results().AnamorphoseGeometrieMNT();
}

const cTplValGesInit< cAnamorphoseGeometrieMNT > & cParamMICMAC::AnamorphoseGeometrieMNT()const 
{
   return Section_Results().AnamorphoseGeometrieMNT();
}


cTplValGesInit< std::string > & cParamMICMAC::RepereCorrel()
{
   return Section_Results().RepereCorrel();
}

const cTplValGesInit< std::string > & cParamMICMAC::RepereCorrel()const 
{
   return Section_Results().RepereCorrel();
}


cTplValGesInit< std::string > & cParamMICMAC::TagRepereCorrel()
{
   return Section_Results().TagRepereCorrel();
}

const cTplValGesInit< std::string > & cParamMICMAC::TagRepereCorrel()const 
{
   return Section_Results().TagRepereCorrel();
}


cTplValGesInit< bool > & cParamMICMAC::DoMEC()
{
   return Section_Results().DoMEC();
}

const cTplValGesInit< bool > & cParamMICMAC::DoMEC()const 
{
   return Section_Results().DoMEC();
}


cTplValGesInit< std::string > & cParamMICMAC::NonExistingFileDoMEC()
{
   return Section_Results().NonExistingFileDoMEC();
}

const cTplValGesInit< std::string > & cParamMICMAC::NonExistingFileDoMEC()const 
{
   return Section_Results().NonExistingFileDoMEC();
}


cTplValGesInit< bool > & cParamMICMAC::DoFDC()
{
   return Section_Results().DoFDC();
}

const cTplValGesInit< bool > & cParamMICMAC::DoFDC()const 
{
   return Section_Results().DoFDC();
}


cTplValGesInit< bool > & cParamMICMAC::GenereXMLComp()
{
   return Section_Results().GenereXMLComp();
}

const cTplValGesInit< bool > & cParamMICMAC::GenereXMLComp()const 
{
   return Section_Results().GenereXMLComp();
}


cTplValGesInit< int > & cParamMICMAC::TAUseMasqNadirKBest()
{
   return Section_Results().TAUseMasqNadirKBest();
}

const cTplValGesInit< int > & cParamMICMAC::TAUseMasqNadirKBest()const 
{
   return Section_Results().TAUseMasqNadirKBest();
}


cTplValGesInit< int > & cParamMICMAC::ZoomMakeTA()
{
   return Section_Results().ZoomMakeTA();
}

const cTplValGesInit< int > & cParamMICMAC::ZoomMakeTA()const 
{
   return Section_Results().ZoomMakeTA();
}


cTplValGesInit< double > & cParamMICMAC::SaturationTA()
{
   return Section_Results().SaturationTA();
}

const cTplValGesInit< double > & cParamMICMAC::SaturationTA()const 
{
   return Section_Results().SaturationTA();
}


cTplValGesInit< bool > & cParamMICMAC::OrthoTA()
{
   return Section_Results().OrthoTA();
}

const cTplValGesInit< bool > & cParamMICMAC::OrthoTA()const 
{
   return Section_Results().OrthoTA();
}


cTplValGesInit< int > & cParamMICMAC::ZoomMakeMasq()
{
   return Section_Results().ZoomMakeMasq();
}

const cTplValGesInit< int > & cParamMICMAC::ZoomMakeMasq()const 
{
   return Section_Results().ZoomMakeMasq();
}


cTplValGesInit< bool > & cParamMICMAC::LazyZoomMaskTerrain()
{
   return Section_Results().LazyZoomMaskTerrain();
}

const cTplValGesInit< bool > & cParamMICMAC::LazyZoomMaskTerrain()const 
{
   return Section_Results().LazyZoomMaskTerrain();
}


cTplValGesInit< bool > & cParamMICMAC::MakeImCptTA()
{
   return Section_Results().MakeImCptTA();
}

const cTplValGesInit< bool > & cParamMICMAC::MakeImCptTA()const 
{
   return Section_Results().MakeImCptTA();
}


cTplValGesInit< std::string > & cParamMICMAC::FilterTA()
{
   return Section_Results().FilterTA();
}

const cTplValGesInit< std::string > & cParamMICMAC::FilterTA()const 
{
   return Section_Results().FilterTA();
}


cTplValGesInit< double > & cParamMICMAC::GammaVisu()
{
   return Section_Results().GammaVisu();
}

const cTplValGesInit< double > & cParamMICMAC::GammaVisu()const 
{
   return Section_Results().GammaVisu();
}


cTplValGesInit< int > & cParamMICMAC::ZoomVisuLiaison()
{
   return Section_Results().ZoomVisuLiaison();
}

const cTplValGesInit< int > & cParamMICMAC::ZoomVisuLiaison()const 
{
   return Section_Results().ZoomVisuLiaison();
}


cTplValGesInit< double > & cParamMICMAC::TolerancePointHomInImage()
{
   return Section_Results().TolerancePointHomInImage();
}

const cTplValGesInit< double > & cParamMICMAC::TolerancePointHomInImage()const 
{
   return Section_Results().TolerancePointHomInImage();
}


cTplValGesInit< double > & cParamMICMAC::FiltragePointHomInImage()
{
   return Section_Results().FiltragePointHomInImage();
}

const cTplValGesInit< double > & cParamMICMAC::FiltragePointHomInImage()const 
{
   return Section_Results().FiltragePointHomInImage();
}


cTplValGesInit< int > & cParamMICMAC::BaseCodeRetourMicmacErreur()
{
   return Section_Results().BaseCodeRetourMicmacErreur();
}

const cTplValGesInit< int > & cParamMICMAC::BaseCodeRetourMicmacErreur()const 
{
   return Section_Results().BaseCodeRetourMicmacErreur();
}


Pt3di & cParamMICMAC::OrdreChannels()
{
   return Section_Results().SuperpositionImages().Val().OrdreChannels();
}

const Pt3di & cParamMICMAC::OrdreChannels()const 
{
   return Section_Results().SuperpositionImages().Val().OrdreChannels();
}


cTplValGesInit< Pt2di > & cParamMICMAC::PtBalanceBlancs()
{
   return Section_Results().SuperpositionImages().Val().PtBalanceBlancs();
}

const cTplValGesInit< Pt2di > & cParamMICMAC::PtBalanceBlancs()const 
{
   return Section_Results().SuperpositionImages().Val().PtBalanceBlancs();
}


cTplValGesInit< Pt2di > & cParamMICMAC::P0Sup()
{
   return Section_Results().SuperpositionImages().Val().P0Sup();
}

const cTplValGesInit< Pt2di > & cParamMICMAC::P0Sup()const 
{
   return Section_Results().SuperpositionImages().Val().P0Sup();
}


cTplValGesInit< Pt2di > & cParamMICMAC::SzSup()
{
   return Section_Results().SuperpositionImages().Val().SzSup();
}

const cTplValGesInit< Pt2di > & cParamMICMAC::SzSup()const 
{
   return Section_Results().SuperpositionImages().Val().SzSup();
}


cElRegex_Ptr & cParamMICMAC::PatternSelGrid()
{
   return Section_Results().SuperpositionImages().Val().PatternSelGrid();
}

const cElRegex_Ptr & cParamMICMAC::PatternSelGrid()const 
{
   return Section_Results().SuperpositionImages().Val().PatternSelGrid();
}


std::string & cParamMICMAC::PatternNameGrid()
{
   return Section_Results().SuperpositionImages().Val().PatternNameGrid();
}

const std::string & cParamMICMAC::PatternNameGrid()const 
{
   return Section_Results().SuperpositionImages().Val().PatternNameGrid();
}


std::list< cColorimetriesCanaux > & cParamMICMAC::ColorimetriesCanaux()
{
   return Section_Results().SuperpositionImages().Val().ColorimetriesCanaux();
}

const std::list< cColorimetriesCanaux > & cParamMICMAC::ColorimetriesCanaux()const 
{
   return Section_Results().SuperpositionImages().Val().ColorimetriesCanaux();
}


cTplValGesInit< double > & cParamMICMAC::GammaCorrection()
{
   return Section_Results().SuperpositionImages().Val().GammaCorrection();
}

const cTplValGesInit< double > & cParamMICMAC::GammaCorrection()const 
{
   return Section_Results().SuperpositionImages().Val().GammaCorrection();
}


cTplValGesInit< double > & cParamMICMAC::MultiplicateurBlanc()
{
   return Section_Results().SuperpositionImages().Val().MultiplicateurBlanc();
}

const cTplValGesInit< double > & cParamMICMAC::MultiplicateurBlanc()const 
{
   return Section_Results().SuperpositionImages().Val().MultiplicateurBlanc();
}


cTplValGesInit< bool > & cParamMICMAC::GenFileImages()
{
   return Section_Results().SuperpositionImages().Val().GenFileImages();
}

const cTplValGesInit< bool > & cParamMICMAC::GenFileImages()const 
{
   return Section_Results().SuperpositionImages().Val().GenFileImages();
}


cTplValGesInit< cSuperpositionImages > & cParamMICMAC::SuperpositionImages()
{
   return Section_Results().SuperpositionImages();
}

const cTplValGesInit< cSuperpositionImages > & cParamMICMAC::SuperpositionImages()const 
{
   return Section_Results().SuperpositionImages();
}


cSection_Results & cParamMICMAC::Section_Results()
{
   return mSection_Results;
}

const cSection_Results & cParamMICMAC::Section_Results()const 
{
   return mSection_Results;
}


cTplValGesInit< std::string > & cParamMICMAC::FileExportApero2MM()
{
   return Section_WorkSpace().FileExportApero2MM();
}

const cTplValGesInit< std::string > & cParamMICMAC::FileExportApero2MM()const 
{
   return Section_WorkSpace().FileExportApero2MM();
}


cTplValGesInit< bool > & cParamMICMAC::UseProfInVertLoc()
{
   return Section_WorkSpace().UseProfInVertLoc();
}

const cTplValGesInit< bool > & cParamMICMAC::UseProfInVertLoc()const 
{
   return Section_WorkSpace().UseProfInVertLoc();
}


cTplValGesInit< std::string > & cParamMICMAC::NameFileParamMICMAC()
{
   return Section_WorkSpace().NameFileParamMICMAC();
}

const cTplValGesInit< std::string > & cParamMICMAC::NameFileParamMICMAC()const 
{
   return Section_WorkSpace().NameFileParamMICMAC();
}


std::string & cParamMICMAC::WorkDir()
{
   return Section_WorkSpace().WorkDir();
}

const std::string & cParamMICMAC::WorkDir()const 
{
   return Section_WorkSpace().WorkDir();
}


cTplValGesInit< std::string > & cParamMICMAC::DirImagesOri()
{
   return Section_WorkSpace().DirImagesOri();
}

const cTplValGesInit< std::string > & cParamMICMAC::DirImagesOri()const 
{
   return Section_WorkSpace().DirImagesOri();
}


std::string & cParamMICMAC::TmpMEC()
{
   return Section_WorkSpace().TmpMEC();
}

const std::string & cParamMICMAC::TmpMEC()const 
{
   return Section_WorkSpace().TmpMEC();
}


cTplValGesInit< std::string > & cParamMICMAC::TmpPyr()
{
   return Section_WorkSpace().TmpPyr();
}

const cTplValGesInit< std::string > & cParamMICMAC::TmpPyr()const 
{
   return Section_WorkSpace().TmpPyr();
}


cTplValGesInit< std::string > & cParamMICMAC::TmpGeom()
{
   return Section_WorkSpace().TmpGeom();
}

const cTplValGesInit< std::string > & cParamMICMAC::TmpGeom()const 
{
   return Section_WorkSpace().TmpGeom();
}


cTplValGesInit< std::string > & cParamMICMAC::TmpResult()
{
   return Section_WorkSpace().TmpResult();
}

const cTplValGesInit< std::string > & cParamMICMAC::TmpResult()const 
{
   return Section_WorkSpace().TmpResult();
}


cTplValGesInit< bool > & cParamMICMAC::CalledByProcess()
{
   return Section_WorkSpace().CalledByProcess();
}

const cTplValGesInit< bool > & cParamMICMAC::CalledByProcess()const 
{
   return Section_WorkSpace().CalledByProcess();
}


cTplValGesInit< int > & cParamMICMAC::IdMasterProcess()
{
   return Section_WorkSpace().IdMasterProcess();
}

const cTplValGesInit< int > & cParamMICMAC::IdMasterProcess()const 
{
   return Section_WorkSpace().IdMasterProcess();
}


cTplValGesInit< bool > & cParamMICMAC::CreateGrayFileAtBegin()
{
   return Section_WorkSpace().CreateGrayFileAtBegin();
}

const cTplValGesInit< bool > & cParamMICMAC::CreateGrayFileAtBegin()const 
{
   return Section_WorkSpace().CreateGrayFileAtBegin();
}


cTplValGesInit< bool > & cParamMICMAC::Visu()
{
   return Section_WorkSpace().Visu();
}

const cTplValGesInit< bool > & cParamMICMAC::Visu()const 
{
   return Section_WorkSpace().Visu();
}


cTplValGesInit< int > & cParamMICMAC::ByProcess()
{
   return Section_WorkSpace().ByProcess();
}

const cTplValGesInit< int > & cParamMICMAC::ByProcess()const 
{
   return Section_WorkSpace().ByProcess();
}


cTplValGesInit< bool > & cParamMICMAC::StopOnEchecFils()
{
   return Section_WorkSpace().StopOnEchecFils();
}

const cTplValGesInit< bool > & cParamMICMAC::StopOnEchecFils()const 
{
   return Section_WorkSpace().StopOnEchecFils();
}


cTplValGesInit< int > & cParamMICMAC::AvalaibleMemory()
{
   return Section_WorkSpace().AvalaibleMemory();
}

const cTplValGesInit< int > & cParamMICMAC::AvalaibleMemory()const 
{
   return Section_WorkSpace().AvalaibleMemory();
}


cTplValGesInit< int > & cParamMICMAC::SzRecouvrtDalles()
{
   return Section_WorkSpace().SzRecouvrtDalles();
}

const cTplValGesInit< int > & cParamMICMAC::SzRecouvrtDalles()const 
{
   return Section_WorkSpace().SzRecouvrtDalles();
}


cTplValGesInit< int > & cParamMICMAC::SzDalleMin()
{
   return Section_WorkSpace().SzDalleMin();
}

const cTplValGesInit< int > & cParamMICMAC::SzDalleMin()const 
{
   return Section_WorkSpace().SzDalleMin();
}


cTplValGesInit< int > & cParamMICMAC::SzDalleMax()
{
   return Section_WorkSpace().SzDalleMax();
}

const cTplValGesInit< int > & cParamMICMAC::SzDalleMax()const 
{
   return Section_WorkSpace().SzDalleMax();
}


cTplValGesInit< double > & cParamMICMAC::NbCelluleMax()
{
   return Section_WorkSpace().NbCelluleMax();
}

const cTplValGesInit< double > & cParamMICMAC::NbCelluleMax()const 
{
   return Section_WorkSpace().NbCelluleMax();
}


cTplValGesInit< int > & cParamMICMAC::SzMinDecomposCalc()
{
   return Section_WorkSpace().SzMinDecomposCalc();
}

const cTplValGesInit< int > & cParamMICMAC::SzMinDecomposCalc()const 
{
   return Section_WorkSpace().SzMinDecomposCalc();
}


cTplValGesInit< bool > & cParamMICMAC::AutorizeSplitRec()
{
   return Section_WorkSpace().AutorizeSplitRec();
}

const cTplValGesInit< bool > & cParamMICMAC::AutorizeSplitRec()const 
{
   return Section_WorkSpace().AutorizeSplitRec();
}


cTplValGesInit< int > & cParamMICMAC::DefTileFile()
{
   return Section_WorkSpace().DefTileFile();
}

const cTplValGesInit< int > & cParamMICMAC::DefTileFile()const 
{
   return Section_WorkSpace().DefTileFile();
}


cTplValGesInit< double > & cParamMICMAC::NbPixDefFilesAux()
{
   return Section_WorkSpace().NbPixDefFilesAux();
}

const cTplValGesInit< double > & cParamMICMAC::NbPixDefFilesAux()const 
{
   return Section_WorkSpace().NbPixDefFilesAux();
}


cTplValGesInit< int > & cParamMICMAC::DeZoomDefMinFileAux()
{
   return Section_WorkSpace().DeZoomDefMinFileAux();
}

const cTplValGesInit< int > & cParamMICMAC::DeZoomDefMinFileAux()const 
{
   return Section_WorkSpace().DeZoomDefMinFileAux();
}


cTplValGesInit< int > & cParamMICMAC::FirstEtapeMEC()
{
   return Section_WorkSpace().FirstEtapeMEC();
}

const cTplValGesInit< int > & cParamMICMAC::FirstEtapeMEC()const 
{
   return Section_WorkSpace().FirstEtapeMEC();
}


cTplValGesInit< int > & cParamMICMAC::LastEtapeMEC()
{
   return Section_WorkSpace().LastEtapeMEC();
}

const cTplValGesInit< int > & cParamMICMAC::LastEtapeMEC()const 
{
   return Section_WorkSpace().LastEtapeMEC();
}


cTplValGesInit< int > & cParamMICMAC::FirstBoiteMEC()
{
   return Section_WorkSpace().FirstBoiteMEC();
}

const cTplValGesInit< int > & cParamMICMAC::FirstBoiteMEC()const 
{
   return Section_WorkSpace().FirstBoiteMEC();
}


cTplValGesInit< int > & cParamMICMAC::NbBoitesMEC()
{
   return Section_WorkSpace().NbBoitesMEC();
}

const cTplValGesInit< int > & cParamMICMAC::NbBoitesMEC()const 
{
   return Section_WorkSpace().NbBoitesMEC();
}


cTplValGesInit< std::string > & cParamMICMAC::NomChantier()
{
   return Section_WorkSpace().NomChantier();
}

const cTplValGesInit< std::string > & cParamMICMAC::NomChantier()const 
{
   return Section_WorkSpace().NomChantier();
}


std::string & cParamMICMAC::PatternSelChantier()
{
   return Section_WorkSpace().CalcNomChantier().Val().PatternSelChantier();
}

const std::string & cParamMICMAC::PatternSelChantier()const 
{
   return Section_WorkSpace().CalcNomChantier().Val().PatternSelChantier();
}


std::string & cParamMICMAC::PatNameChantier()
{
   return Section_WorkSpace().CalcNomChantier().Val().PatNameChantier();
}

const std::string & cParamMICMAC::PatNameChantier()const 
{
   return Section_WorkSpace().CalcNomChantier().Val().PatNameChantier();
}


cTplValGesInit< std::string > & cParamMICMAC::SeparateurChantier()
{
   return Section_WorkSpace().CalcNomChantier().Val().SeparateurChantier();
}

const cTplValGesInit< std::string > & cParamMICMAC::SeparateurChantier()const 
{
   return Section_WorkSpace().CalcNomChantier().Val().SeparateurChantier();
}


cTplValGesInit< cCalcNomChantier > & cParamMICMAC::CalcNomChantier()
{
   return Section_WorkSpace().CalcNomChantier();
}

const cTplValGesInit< cCalcNomChantier > & cParamMICMAC::CalcNomChantier()const 
{
   return Section_WorkSpace().CalcNomChantier();
}


cTplValGesInit< std::string > & cParamMICMAC::PatternSelPyr()
{
   return Section_WorkSpace().PatternSelPyr();
}

const cTplValGesInit< std::string > & cParamMICMAC::PatternSelPyr()const 
{
   return Section_WorkSpace().PatternSelPyr();
}


cTplValGesInit< std::string > & cParamMICMAC::PatternNomPyr()
{
   return Section_WorkSpace().PatternNomPyr();
}

const cTplValGesInit< std::string > & cParamMICMAC::PatternNomPyr()const 
{
   return Section_WorkSpace().PatternNomPyr();
}


cTplValGesInit< std::string > & cParamMICMAC::SeparateurPyr()
{
   return Section_WorkSpace().SeparateurPyr();
}

const cTplValGesInit< std::string > & cParamMICMAC::SeparateurPyr()const 
{
   return Section_WorkSpace().SeparateurPyr();
}


cTplValGesInit< std::string > & cParamMICMAC::KeyCalNamePyr()
{
   return Section_WorkSpace().KeyCalNamePyr();
}

const cTplValGesInit< std::string > & cParamMICMAC::KeyCalNamePyr()const 
{
   return Section_WorkSpace().KeyCalNamePyr();
}


cTplValGesInit< bool > & cParamMICMAC::ActivePurge()
{
   return Section_WorkSpace().ActivePurge();
}

const cTplValGesInit< bool > & cParamMICMAC::ActivePurge()const 
{
   return Section_WorkSpace().ActivePurge();
}


std::list< cPurgeFiles > & cParamMICMAC::PurgeFiles()
{
   return Section_WorkSpace().PurgeFiles();
}

const std::list< cPurgeFiles > & cParamMICMAC::PurgeFiles()const 
{
   return Section_WorkSpace().PurgeFiles();
}


cTplValGesInit< bool > & cParamMICMAC::PurgeMECResultBefore()
{
   return Section_WorkSpace().PurgeMECResultBefore();
}

const cTplValGesInit< bool > & cParamMICMAC::PurgeMECResultBefore()const 
{
   return Section_WorkSpace().PurgeMECResultBefore();
}


cTplValGesInit< std::string > & cParamMICMAC::PreservedFile()
{
   return Section_WorkSpace().PreservedFile();
}

const cTplValGesInit< std::string > & cParamMICMAC::PreservedFile()const 
{
   return Section_WorkSpace().PreservedFile();
}


cTplValGesInit< bool > & cParamMICMAC::UseChantierNameDescripteur()
{
   return Section_WorkSpace().UseChantierNameDescripteur();
}

const cTplValGesInit< bool > & cParamMICMAC::UseChantierNameDescripteur()const 
{
   return Section_WorkSpace().UseChantierNameDescripteur();
}


cTplValGesInit< string > & cParamMICMAC::FileChantierNameDescripteur()
{
   return Section_WorkSpace().FileChantierNameDescripteur();
}

const cTplValGesInit< string > & cParamMICMAC::FileChantierNameDescripteur()const 
{
   return Section_WorkSpace().FileChantierNameDescripteur();
}


cTplValGesInit< cCmdMappeur > & cParamMICMAC::MapMicMac()
{
   return Section_WorkSpace().MapMicMac();
}

const cTplValGesInit< cCmdMappeur > & cParamMICMAC::MapMicMac()const 
{
   return Section_WorkSpace().MapMicMac();
}


cTplValGesInit< cCmdExePar > & cParamMICMAC::PostProcess()
{
   return Section_WorkSpace().PostProcess();
}

const cTplValGesInit< cCmdExePar > & cParamMICMAC::PostProcess()const 
{
   return Section_WorkSpace().PostProcess();
}


cTplValGesInit< eComprTiff > & cParamMICMAC::ComprMasque()
{
   return Section_WorkSpace().ComprMasque();
}

const cTplValGesInit< eComprTiff > & cParamMICMAC::ComprMasque()const 
{
   return Section_WorkSpace().ComprMasque();
}


cTplValGesInit< eTypeNumerique > & cParamMICMAC::TypeMasque()
{
   return Section_WorkSpace().TypeMasque();
}

const cTplValGesInit< eTypeNumerique > & cParamMICMAC::TypeMasque()const 
{
   return Section_WorkSpace().TypeMasque();
}


cSection_WorkSpace & cParamMICMAC::Section_WorkSpace()
{
   return mSection_WorkSpace;
}

const cSection_WorkSpace & cParamMICMAC::Section_WorkSpace()const 
{
   return mSection_WorkSpace;
}


cTplValGesInit< bool > & cParamMICMAC::ExeBatch()
{
   return SectionBatch().Val().ExeBatch();
}

const cTplValGesInit< bool > & cParamMICMAC::ExeBatch()const 
{
   return SectionBatch().Val().ExeBatch();
}


std::list< cOneBatch > & cParamMICMAC::OneBatch()
{
   return SectionBatch().Val().OneBatch();
}

const std::list< cOneBatch > & cParamMICMAC::OneBatch()const 
{
   return SectionBatch().Val().OneBatch();
}


std::list< std::string > & cParamMICMAC::NextMicMacFile2Exec()
{
   return SectionBatch().Val().NextMicMacFile2Exec();
}

const std::list< std::string > & cParamMICMAC::NextMicMacFile2Exec()const 
{
   return SectionBatch().Val().NextMicMacFile2Exec();
}


cTplValGesInit< cSectionBatch > & cParamMICMAC::SectionBatch()
{
   return mSectionBatch;
}

const cTplValGesInit< cSectionBatch > & cParamMICMAC::SectionBatch()const 
{
   return mSectionBatch;
}


cTplValGesInit< bool > & cParamMICMAC::DebugMM()
{
   return Section_Vrac().DebugMM();
}

const cTplValGesInit< bool > & cParamMICMAC::DebugMM()const 
{
   return Section_Vrac().DebugMM();
}


cTplValGesInit< int > & cParamMICMAC::SL_XSzW()
{
   return Section_Vrac().SL_XSzW();
}

const cTplValGesInit< int > & cParamMICMAC::SL_XSzW()const 
{
   return Section_Vrac().SL_XSzW();
}


cTplValGesInit< int > & cParamMICMAC::SL_YSzW()
{
   return Section_Vrac().SL_YSzW();
}

const cTplValGesInit< int > & cParamMICMAC::SL_YSzW()const 
{
   return Section_Vrac().SL_YSzW();
}


cTplValGesInit< bool > & cParamMICMAC::SL_Epip()
{
   return Section_Vrac().SL_Epip();
}

const cTplValGesInit< bool > & cParamMICMAC::SL_Epip()const 
{
   return Section_Vrac().SL_Epip();
}


cTplValGesInit< int > & cParamMICMAC::SL_YDecEpip()
{
   return Section_Vrac().SL_YDecEpip();
}

const cTplValGesInit< int > & cParamMICMAC::SL_YDecEpip()const 
{
   return Section_Vrac().SL_YDecEpip();
}


cTplValGesInit< std::string > & cParamMICMAC::SL_PackHom0()
{
   return Section_Vrac().SL_PackHom0();
}

const cTplValGesInit< std::string > & cParamMICMAC::SL_PackHom0()const 
{
   return Section_Vrac().SL_PackHom0();
}


cTplValGesInit< bool > & cParamMICMAC::SL_RedrOnCur()
{
   return Section_Vrac().SL_RedrOnCur();
}

const cTplValGesInit< bool > & cParamMICMAC::SL_RedrOnCur()const 
{
   return Section_Vrac().SL_RedrOnCur();
}


cTplValGesInit< bool > & cParamMICMAC::SL_NewRedrCur()
{
   return Section_Vrac().SL_NewRedrCur();
}

const cTplValGesInit< bool > & cParamMICMAC::SL_NewRedrCur()const 
{
   return Section_Vrac().SL_NewRedrCur();
}


cTplValGesInit< bool > & cParamMICMAC::SL_L2Estim()
{
   return Section_Vrac().SL_L2Estim();
}

const cTplValGesInit< bool > & cParamMICMAC::SL_L2Estim()const 
{
   return Section_Vrac().SL_L2Estim();
}


cTplValGesInit< std::vector<std::string> > & cParamMICMAC::SL_FILTER()
{
   return Section_Vrac().SL_FILTER();
}

const cTplValGesInit< std::vector<std::string> > & cParamMICMAC::SL_FILTER()const 
{
   return Section_Vrac().SL_FILTER();
}


cTplValGesInit< bool > & cParamMICMAC::SL_TJS_FILTER()
{
   return Section_Vrac().SL_TJS_FILTER();
}

const cTplValGesInit< bool > & cParamMICMAC::SL_TJS_FILTER()const 
{
   return Section_Vrac().SL_TJS_FILTER();
}


cTplValGesInit< double > & cParamMICMAC::SL_Step_Grid()
{
   return Section_Vrac().SL_Step_Grid();
}

const cTplValGesInit< double > & cParamMICMAC::SL_Step_Grid()const 
{
   return Section_Vrac().SL_Step_Grid();
}


cTplValGesInit< std::string > & cParamMICMAC::SL_Name_Grid_Exp()
{
   return Section_Vrac().SL_Name_Grid_Exp();
}

const cTplValGesInit< std::string > & cParamMICMAC::SL_Name_Grid_Exp()const 
{
   return Section_Vrac().SL_Name_Grid_Exp();
}


cTplValGesInit< double > & cParamMICMAC::VSG_DynImRed()
{
   return Section_Vrac().VSG_DynImRed();
}

const cTplValGesInit< double > & cParamMICMAC::VSG_DynImRed()const 
{
   return Section_Vrac().VSG_DynImRed();
}


cTplValGesInit< int > & cParamMICMAC::VSG_DeZoomContr()
{
   return Section_Vrac().VSG_DeZoomContr();
}

const cTplValGesInit< int > & cParamMICMAC::VSG_DeZoomContr()const 
{
   return Section_Vrac().VSG_DeZoomContr();
}


cTplValGesInit< Pt2di > & cParamMICMAC::PtDebug()
{
   return Section_Vrac().PtDebug();
}

const cTplValGesInit< Pt2di > & cParamMICMAC::PtDebug()const 
{
   return Section_Vrac().PtDebug();
}


cTplValGesInit< bool > & cParamMICMAC::DumpNappesEnglob()
{
   return Section_Vrac().DumpNappesEnglob();
}

const cTplValGesInit< bool > & cParamMICMAC::DumpNappesEnglob()const 
{
   return Section_Vrac().DumpNappesEnglob();
}


cTplValGesInit< bool > & cParamMICMAC::InterditAccelerationCorrSpec()
{
   return Section_Vrac().InterditAccelerationCorrSpec();
}

const cTplValGesInit< bool > & cParamMICMAC::InterditAccelerationCorrSpec()const 
{
   return Section_Vrac().InterditAccelerationCorrSpec();
}


cTplValGesInit< bool > & cParamMICMAC::InterditCorrelRapide()
{
   return Section_Vrac().InterditCorrelRapide();
}

const cTplValGesInit< bool > & cParamMICMAC::InterditCorrelRapide()const 
{
   return Section_Vrac().InterditCorrelRapide();
}


cTplValGesInit< bool > & cParamMICMAC::ForceCorrelationByRect()
{
   return Section_Vrac().ForceCorrelationByRect();
}

const cTplValGesInit< bool > & cParamMICMAC::ForceCorrelationByRect()const 
{
   return Section_Vrac().ForceCorrelationByRect();
}


std::list< cListTestCpleHomol > & cParamMICMAC::ListTestCpleHomol()
{
   return Section_Vrac().ListTestCpleHomol();
}

const std::list< cListTestCpleHomol > & cParamMICMAC::ListTestCpleHomol()const 
{
   return Section_Vrac().ListTestCpleHomol();
}


std::list< Pt3dr > & cParamMICMAC::ListeTestPointsTerrain()
{
   return Section_Vrac().ListeTestPointsTerrain();
}

const std::list< Pt3dr > & cParamMICMAC::ListeTestPointsTerrain()const 
{
   return Section_Vrac().ListeTestPointsTerrain();
}


cTplValGesInit< bool > & cParamMICMAC::WithMessage()
{
   return Section_Vrac().WithMessage();
}

const cTplValGesInit< bool > & cParamMICMAC::WithMessage()const 
{
   return Section_Vrac().WithMessage();
}


cTplValGesInit< bool > & cParamMICMAC::ShowLoadedImage()
{
   return Section_Vrac().ShowLoadedImage();
}

const cTplValGesInit< bool > & cParamMICMAC::ShowLoadedImage()const 
{
   return Section_Vrac().ShowLoadedImage();
}


Pt2di & cParamMICMAC::P1()
{
   return Section_Vrac().SectionDebug().Val().DebugEscalier().Val().P1();
}

const Pt2di & cParamMICMAC::P1()const 
{
   return Section_Vrac().SectionDebug().Val().DebugEscalier().Val().P1();
}


Pt2di & cParamMICMAC::P2()
{
   return Section_Vrac().SectionDebug().Val().DebugEscalier().Val().P2();
}

const Pt2di & cParamMICMAC::P2()const 
{
   return Section_Vrac().SectionDebug().Val().DebugEscalier().Val().P2();
}


cTplValGesInit< bool > & cParamMICMAC::ShowDerivZ()
{
   return Section_Vrac().SectionDebug().Val().DebugEscalier().Val().ShowDerivZ();
}

const cTplValGesInit< bool > & cParamMICMAC::ShowDerivZ()const 
{
   return Section_Vrac().SectionDebug().Val().DebugEscalier().Val().ShowDerivZ();
}


cTplValGesInit< cDebugEscalier > & cParamMICMAC::DebugEscalier()
{
   return Section_Vrac().SectionDebug().Val().DebugEscalier();
}

const cTplValGesInit< cDebugEscalier > & cParamMICMAC::DebugEscalier()const 
{
   return Section_Vrac().SectionDebug().Val().DebugEscalier();
}


cTplValGesInit< cSectionDebug > & cParamMICMAC::SectionDebug()
{
   return Section_Vrac().SectionDebug();
}

const cTplValGesInit< cSectionDebug > & cParamMICMAC::SectionDebug()const 
{
   return Section_Vrac().SectionDebug();
}


cSection_Vrac & cParamMICMAC::Section_Vrac()
{
   return mSection_Vrac;
}

const cSection_Vrac & cParamMICMAC::Section_Vrac()const 
{
   return mSection_Vrac;
}

void  BinaryUnDumpFromFile(cParamMICMAC & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.DicoLoc().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.DicoLoc().ValForcedForUnUmp(),aFp);
        }
        else  anObj.DicoLoc().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.Section_Terrain(),aFp);
    BinaryUnDumpFromFile(anObj.Section_PriseDeVue(),aFp);
    BinaryUnDumpFromFile(anObj.Section_MEC(),aFp);
    BinaryUnDumpFromFile(anObj.Section_Results(),aFp);
    BinaryUnDumpFromFile(anObj.Section_WorkSpace(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SectionBatch().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SectionBatch().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SectionBatch().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.Section_Vrac(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cParamMICMAC & anObj)
{
    BinaryDumpInFile(aFp,anObj.DicoLoc().IsInit());
    if (anObj.DicoLoc().IsInit()) BinaryDumpInFile(aFp,anObj.DicoLoc().Val());
    BinaryDumpInFile(aFp,anObj.Section_Terrain());
    BinaryDumpInFile(aFp,anObj.Section_PriseDeVue());
    BinaryDumpInFile(aFp,anObj.Section_MEC());
    BinaryDumpInFile(aFp,anObj.Section_Results());
    BinaryDumpInFile(aFp,anObj.Section_WorkSpace());
    BinaryDumpInFile(aFp,anObj.SectionBatch().IsInit());
    if (anObj.SectionBatch().IsInit()) BinaryDumpInFile(aFp,anObj.SectionBatch().Val());
    BinaryDumpInFile(aFp,anObj.Section_Vrac());
}

cElXMLTree * ToXMLTree(const cParamMICMAC & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ParamMICMAC",eXMLBranche);
   if (anObj.DicoLoc().IsInit())
      aRes->AddFils(ToXMLTree(anObj.DicoLoc().Val())->ReTagThis("DicoLoc"));
   aRes->AddFils(ToXMLTree(anObj.Section_Terrain())->ReTagThis("Section_Terrain"));
   aRes->AddFils(ToXMLTree(anObj.Section_PriseDeVue())->ReTagThis("Section_PriseDeVue"));
   aRes->AddFils(ToXMLTree(anObj.Section_MEC())->ReTagThis("Section_MEC"));
   aRes->AddFils(ToXMLTree(anObj.Section_Results())->ReTagThis("Section_Results"));
   aRes->AddFils(ToXMLTree(anObj.Section_WorkSpace())->ReTagThis("Section_WorkSpace"));
   if (anObj.SectionBatch().IsInit())
      aRes->AddFils(ToXMLTree(anObj.SectionBatch().Val())->ReTagThis("SectionBatch"));
   aRes->AddFils(ToXMLTree(anObj.Section_Vrac())->ReTagThis("Section_Vrac"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cParamMICMAC & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.DicoLoc(),aTree->Get("DicoLoc",1)); //tototo 

   xml_init(anObj.Section_Terrain(),aTree->Get("Section_Terrain",1)); //tototo 

   xml_init(anObj.Section_PriseDeVue(),aTree->Get("Section_PriseDeVue",1)); //tototo 

   xml_init(anObj.Section_MEC(),aTree->Get("Section_MEC",1)); //tototo 

   xml_init(anObj.Section_Results(),aTree->Get("Section_Results",1)); //tototo 

   xml_init(anObj.Section_WorkSpace(),aTree->Get("Section_WorkSpace",1)); //tototo 

   xml_init(anObj.SectionBatch(),aTree->Get("SectionBatch",1)); //tototo 

   xml_init(anObj.Section_Vrac(),aTree->Get("Section_Vrac",1)); //tototo 
}

std::string  Mangling( cParamMICMAC *) {return "90B2E932151BA0A1FE3F";};

// Quelque chose
