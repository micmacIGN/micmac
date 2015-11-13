#include "general/all.h"
#include "private/all.h"
#include "cParamApero.h"
namespace NS_ParamApero{
eTypeContrainteCalibCamera  Str2eTypeContrainteCalibCamera(const std::string & aName)
{
   if (aName=="eAllParamLibres")
      return eAllParamLibres;
   else if (aName=="eAllParamFiges")
      return eAllParamFiges;
   else if (aName=="eLiberteParamDeg_0")
      return eLiberteParamDeg_0;
   else if (aName=="eLiberteParamDeg_1")
      return eLiberteParamDeg_1;
   else if (aName=="eLiberteParamDeg_2")
      return eLiberteParamDeg_2;
   else if (aName=="eLiberteParamDeg_3")
      return eLiberteParamDeg_3;
   else if (aName=="eLiberteParamDeg_4")
      return eLiberteParamDeg_4;
   else if (aName=="eLiberteParamDeg_5")
      return eLiberteParamDeg_5;
   else if (aName=="eLiberteParamDeg_6")
      return eLiberteParamDeg_6;
   else if (aName=="eLiberteParamDeg_7")
      return eLiberteParamDeg_7;
   else if (aName=="eLiberteParamDeg_2_NoAff")
      return eLiberteParamDeg_2_NoAff;
   else if (aName=="eLiberteParamDeg_3_NoAff")
      return eLiberteParamDeg_3_NoAff;
   else if (aName=="eLiberteParamDeg_4_NoAff")
      return eLiberteParamDeg_4_NoAff;
   else if (aName=="eLiberteParamDeg_5_NoAff")
      return eLiberteParamDeg_5_NoAff;
   else if (aName=="eLiberteFocale_0")
      return eLiberteFocale_0;
   else if (aName=="eLiberteFocale_1")
      return eLiberteFocale_1;
   else if (aName=="eLib_PP_CD_00")
      return eLib_PP_CD_00;
   else if (aName=="eLib_PP_CD_10")
      return eLib_PP_CD_10;
   else if (aName=="eLib_PP_CD_01")
      return eLib_PP_CD_01;
   else if (aName=="eLib_PP_CD_11")
      return eLib_PP_CD_11;
   else if (aName=="eLib_PP_CD_Lies")
      return eLib_PP_CD_Lies;
   else if (aName=="eLiberte_DR0")
      return eLiberte_DR0;
   else if (aName=="eLiberte_DR1")
      return eLiberte_DR1;
   else if (aName=="eLiberte_DR2")
      return eLiberte_DR2;
   else if (aName=="eLiberte_DR3")
      return eLiberte_DR3;
   else if (aName=="eLiberte_DR4")
      return eLiberte_DR4;
   else if (aName=="eLiberte_DR5")
      return eLiberte_DR5;
   else if (aName=="eLiberte_DR6")
      return eLiberte_DR6;
   else if (aName=="eLiberte_DR7")
      return eLiberte_DR7;
   else if (aName=="eLiberte_DR8")
      return eLiberte_DR8;
   else if (aName=="eLiberte_DR9")
      return eLiberte_DR9;
   else if (aName=="eLiberte_DR10")
      return eLiberte_DR10;
   else if (aName=="eLiberte_Dec0")
      return eLiberte_Dec0;
   else if (aName=="eLiberte_Dec1")
      return eLiberte_Dec1;
   else if (aName=="eLiberte_Dec2")
      return eLiberte_Dec2;
   else if (aName=="eLiberte_Dec3")
      return eLiberte_Dec3;
   else if (aName=="eLiberte_Dec4")
      return eLiberte_Dec4;
   else if (aName=="eLiberte_Dec5")
      return eLiberte_Dec5;
   else if (aName=="eLiberte_Phgr_Std_Aff")
      return eLiberte_Phgr_Std_Aff;
   else if (aName=="eLiberte_Phgr_Std_Dec")
      return eLiberte_Phgr_Std_Dec;
   else if (aName=="eFige_Phgr_Std_Aff")
      return eFige_Phgr_Std_Aff;
   else if (aName=="eFige_Phgr_Std_Dec")
      return eFige_Phgr_Std_Dec;
   else if (aName=="eLiberte_AFocal0")
      return eLiberte_AFocal0;
   else if (aName=="eLiberte_AFocal1")
      return eLiberte_AFocal1;
   else if (aName=="eFige_AFocal0")
      return eFige_AFocal0;
   else if (aName=="eFige_AFocal1")
      return eFige_AFocal1;
  else
  {
      cout << aName << " is not a correct value for enum eTypeContrainteCalibCamera\n" ;
      ELISE_ASSERT(false,"XML enum value error");
  }
  return (eTypeContrainteCalibCamera) 0;
}
void xml_init(eTypeContrainteCalibCamera & aVal,cElXMLTree * aTree)
{
   aVal= Str2eTypeContrainteCalibCamera(aTree->Contenu());
}
std::string  eToString(const eTypeContrainteCalibCamera & anObj)
{
   if (anObj==eAllParamLibres)
      return  "eAllParamLibres";
   if (anObj==eAllParamFiges)
      return  "eAllParamFiges";
   if (anObj==eLiberteParamDeg_0)
      return  "eLiberteParamDeg_0";
   if (anObj==eLiberteParamDeg_1)
      return  "eLiberteParamDeg_1";
   if (anObj==eLiberteParamDeg_2)
      return  "eLiberteParamDeg_2";
   if (anObj==eLiberteParamDeg_3)
      return  "eLiberteParamDeg_3";
   if (anObj==eLiberteParamDeg_4)
      return  "eLiberteParamDeg_4";
   if (anObj==eLiberteParamDeg_5)
      return  "eLiberteParamDeg_5";
   if (anObj==eLiberteParamDeg_6)
      return  "eLiberteParamDeg_6";
   if (anObj==eLiberteParamDeg_7)
      return  "eLiberteParamDeg_7";
   if (anObj==eLiberteParamDeg_2_NoAff)
      return  "eLiberteParamDeg_2_NoAff";
   if (anObj==eLiberteParamDeg_3_NoAff)
      return  "eLiberteParamDeg_3_NoAff";
   if (anObj==eLiberteParamDeg_4_NoAff)
      return  "eLiberteParamDeg_4_NoAff";
   if (anObj==eLiberteParamDeg_5_NoAff)
      return  "eLiberteParamDeg_5_NoAff";
   if (anObj==eLiberteFocale_0)
      return  "eLiberteFocale_0";
   if (anObj==eLiberteFocale_1)
      return  "eLiberteFocale_1";
   if (anObj==eLib_PP_CD_00)
      return  "eLib_PP_CD_00";
   if (anObj==eLib_PP_CD_10)
      return  "eLib_PP_CD_10";
   if (anObj==eLib_PP_CD_01)
      return  "eLib_PP_CD_01";
   if (anObj==eLib_PP_CD_11)
      return  "eLib_PP_CD_11";
   if (anObj==eLib_PP_CD_Lies)
      return  "eLib_PP_CD_Lies";
   if (anObj==eLiberte_DR0)
      return  "eLiberte_DR0";
   if (anObj==eLiberte_DR1)
      return  "eLiberte_DR1";
   if (anObj==eLiberte_DR2)
      return  "eLiberte_DR2";
   if (anObj==eLiberte_DR3)
      return  "eLiberte_DR3";
   if (anObj==eLiberte_DR4)
      return  "eLiberte_DR4";
   if (anObj==eLiberte_DR5)
      return  "eLiberte_DR5";
   if (anObj==eLiberte_DR6)
      return  "eLiberte_DR6";
   if (anObj==eLiberte_DR7)
      return  "eLiberte_DR7";
   if (anObj==eLiberte_DR8)
      return  "eLiberte_DR8";
   if (anObj==eLiberte_DR9)
      return  "eLiberte_DR9";
   if (anObj==eLiberte_DR10)
      return  "eLiberte_DR10";
   if (anObj==eLiberte_Dec0)
      return  "eLiberte_Dec0";
   if (anObj==eLiberte_Dec1)
      return  "eLiberte_Dec1";
   if (anObj==eLiberte_Dec2)
      return  "eLiberte_Dec2";
   if (anObj==eLiberte_Dec3)
      return  "eLiberte_Dec3";
   if (anObj==eLiberte_Dec4)
      return  "eLiberte_Dec4";
   if (anObj==eLiberte_Dec5)
      return  "eLiberte_Dec5";
   if (anObj==eLiberte_Phgr_Std_Aff)
      return  "eLiberte_Phgr_Std_Aff";
   if (anObj==eLiberte_Phgr_Std_Dec)
      return  "eLiberte_Phgr_Std_Dec";
   if (anObj==eFige_Phgr_Std_Aff)
      return  "eFige_Phgr_Std_Aff";
   if (anObj==eFige_Phgr_Std_Dec)
      return  "eFige_Phgr_Std_Dec";
   if (anObj==eLiberte_AFocal0)
      return  "eLiberte_AFocal0";
   if (anObj==eLiberte_AFocal1)
      return  "eLiberte_AFocal1";
   if (anObj==eFige_AFocal0)
      return  "eFige_AFocal0";
   if (anObj==eFige_AFocal1)
      return  "eFige_AFocal1";
 std::cout << "Enum = eTypeContrainteCalibCamera\n";
   ELISE_ASSERT(false,"Bad Value in eToString for enum value ");
   return "";
}

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eTypeContrainteCalibCamera & anObj)
{
      return  cElXMLTree::ValueNode(aNameTag,eToString(anObj));
}

eTypeCalibAutom  Str2eTypeCalibAutom(const std::string & aName)
{
   if (aName=="eCalibAutomRadial")
      return eCalibAutomRadial;
   else if (aName=="eCalibAutomPhgrStd")
      return eCalibAutomPhgrStd;
   else if (aName=="eCalibAutomFishEyeLineaire")
      return eCalibAutomFishEyeLineaire;
   else if (aName=="eCalibAutomFishEyeEquiSolid")
      return eCalibAutomFishEyeEquiSolid;
   else if (aName=="eCalibAutomRadialBasic")
      return eCalibAutomRadialBasic;
   else if (aName=="eCalibAutomPhgrStdBasic")
      return eCalibAutomPhgrStdBasic;
   else if (aName=="eCalibAutomNone")
      return eCalibAutomNone;
  else
  {
      cout << aName << " is not a correct value for enum eTypeCalibAutom\n" ;
      ELISE_ASSERT(false,"XML enum value error");
  }
  return (eTypeCalibAutom) 0;
}
void xml_init(eTypeCalibAutom & aVal,cElXMLTree * aTree)
{
   aVal= Str2eTypeCalibAutom(aTree->Contenu());
}
std::string  eToString(const eTypeCalibAutom & anObj)
{
   if (anObj==eCalibAutomRadial)
      return  "eCalibAutomRadial";
   if (anObj==eCalibAutomPhgrStd)
      return  "eCalibAutomPhgrStd";
   if (anObj==eCalibAutomFishEyeLineaire)
      return  "eCalibAutomFishEyeLineaire";
   if (anObj==eCalibAutomFishEyeEquiSolid)
      return  "eCalibAutomFishEyeEquiSolid";
   if (anObj==eCalibAutomRadialBasic)
      return  "eCalibAutomRadialBasic";
   if (anObj==eCalibAutomPhgrStdBasic)
      return  "eCalibAutomPhgrStdBasic";
   if (anObj==eCalibAutomNone)
      return  "eCalibAutomNone";
 std::cout << "Enum = eTypeCalibAutom\n";
   ELISE_ASSERT(false,"Bad Value in eToString for enum value ");
   return "";
}

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eTypeCalibAutom & anObj)
{
      return  cElXMLTree::ValueNode(aNameTag,eToString(anObj));
}

eTypeContraintePoseCamera  Str2eTypeContraintePoseCamera(const std::string & aName)
{
   if (aName=="ePoseLibre")
      return ePoseLibre;
   else if (aName=="ePoseFigee")
      return ePoseFigee;
   else if (aName=="ePoseBaseNormee")
      return ePoseBaseNormee;
   else if (aName=="ePoseVraieBaseNormee")
      return ePoseVraieBaseNormee;
   else if (aName=="eCentreFige")
      return eCentreFige;
  else
  {
      cout << aName << " is not a correct value for enum eTypeContraintePoseCamera\n" ;
      ELISE_ASSERT(false,"XML enum value error");
  }
  return (eTypeContraintePoseCamera) 0;
}
void xml_init(eTypeContraintePoseCamera & aVal,cElXMLTree * aTree)
{
   aVal= Str2eTypeContraintePoseCamera(aTree->Contenu());
}
std::string  eToString(const eTypeContraintePoseCamera & anObj)
{
   if (anObj==ePoseLibre)
      return  "ePoseLibre";
   if (anObj==ePoseFigee)
      return  "ePoseFigee";
   if (anObj==ePoseBaseNormee)
      return  "ePoseBaseNormee";
   if (anObj==ePoseVraieBaseNormee)
      return  "ePoseVraieBaseNormee";
   if (anObj==eCentreFige)
      return  "eCentreFige";
 std::cout << "Enum = eTypeContraintePoseCamera\n";
   ELISE_ASSERT(false,"Bad Value in eToString for enum value ");
   return "";
}

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eTypeContraintePoseCamera & anObj)
{
      return  cElXMLTree::ValueNode(aNameTag,eToString(anObj));
}

eTypeVerif  Str2eTypeVerif(const std::string & aName)
{
   if (aName=="eVerifDZ")
      return eVerifDZ;
   else if (aName=="eVerifResPerIm")
      return eVerifResPerIm;
  else
  {
      cout << aName << " is not a correct value for enum eTypeVerif\n" ;
      ELISE_ASSERT(false,"XML enum value error");
  }
  return (eTypeVerif) 0;
}
void xml_init(eTypeVerif & aVal,cElXMLTree * aTree)
{
   aVal= Str2eTypeVerif(aTree->Contenu());
}
std::string  eToString(const eTypeVerif & anObj)
{
   if (anObj==eVerifDZ)
      return  "eVerifDZ";
   if (anObj==eVerifResPerIm)
      return  "eVerifResPerIm";
 std::cout << "Enum = eTypeVerif\n";
   ELISE_ASSERT(false,"Bad Value in eToString for enum value ");
   return "";
}

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eTypeVerif & anObj)
{
      return  cElXMLTree::ValueNode(aNameTag,eToString(anObj));
}

eTypePondMST_MEP  Str2eTypePondMST_MEP(const std::string & aName)
{
   if (aName=="eMST_PondCard")
      return eMST_PondCard;
  else
  {
      cout << aName << " is not a correct value for enum eTypePondMST_MEP\n" ;
      ELISE_ASSERT(false,"XML enum value error");
  }
  return (eTypePondMST_MEP) 0;
}
void xml_init(eTypePondMST_MEP & aVal,cElXMLTree * aTree)
{
   aVal= Str2eTypePondMST_MEP(aTree->Contenu());
}
std::string  eToString(const eTypePondMST_MEP & anObj)
{
   if (anObj==eMST_PondCard)
      return  "eMST_PondCard";
 std::cout << "Enum = eTypePondMST_MEP\n";
   ELISE_ASSERT(false,"Bad Value in eToString for enum value ");
   return "";
}

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eTypePondMST_MEP & anObj)
{
      return  cElXMLTree::ValueNode(aNameTag,eToString(anObj));
}

eControleDescDic  Str2eControleDescDic(const std::string & aName)
{
   if (aName=="eCDD_Jamais")
      return eCDD_Jamais;
   else if (aName=="eCDD_OnRemontee")
      return eCDD_OnRemontee;
   else if (aName=="eCDD_Toujours")
      return eCDD_Toujours;
  else
  {
      cout << aName << " is not a correct value for enum eControleDescDic\n" ;
      ELISE_ASSERT(false,"XML enum value error");
  }
  return (eControleDescDic) 0;
}
void xml_init(eControleDescDic & aVal,cElXMLTree * aTree)
{
   aVal= Str2eControleDescDic(aTree->Contenu());
}
std::string  eToString(const eControleDescDic & anObj)
{
   if (anObj==eCDD_Jamais)
      return  "eCDD_Jamais";
   if (anObj==eCDD_OnRemontee)
      return  "eCDD_OnRemontee";
   if (anObj==eCDD_Toujours)
      return  "eCDD_Toujours";
 std::cout << "Enum = eControleDescDic\n";
   ELISE_ASSERT(false,"Bad Value in eToString for enum value ");
   return "";
}

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eControleDescDic & anObj)
{
      return  cElXMLTree::ValueNode(aNameTag,eToString(anObj));
}

eModePonderationRobuste  Str2eModePonderationRobuste(const std::string & aName)
{
   if (aName=="ePondL2")
      return ePondL2;
   else if (aName=="ePondL1")
      return ePondL1;
   else if (aName=="ePondLK")
      return ePondLK;
   else if (aName=="ePondGauss")
      return ePondGauss;
   else if (aName=="eL1Secured")
      return eL1Secured;
  else
  {
      cout << aName << " is not a correct value for enum eModePonderationRobuste\n" ;
      ELISE_ASSERT(false,"XML enum value error");
  }
  return (eModePonderationRobuste) 0;
}
void xml_init(eModePonderationRobuste & aVal,cElXMLTree * aTree)
{
   aVal= Str2eModePonderationRobuste(aTree->Contenu());
}
std::string  eToString(const eModePonderationRobuste & anObj)
{
   if (anObj==ePondL2)
      return  "ePondL2";
   if (anObj==ePondL1)
      return  "ePondL1";
   if (anObj==ePondLK)
      return  "ePondLK";
   if (anObj==ePondGauss)
      return  "ePondGauss";
   if (anObj==eL1Secured)
      return  "eL1Secured";
 std::cout << "Enum = eModePonderationRobuste\n";
   ELISE_ASSERT(false,"Bad Value in eToString for enum value ");
   return "";
}

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eModePonderationRobuste & anObj)
{
      return  cElXMLTree::ValueNode(aNameTag,eToString(anObj));
}

eUniteMesureErreur  Str2eUniteMesureErreur(const std::string & aName)
{
   if (aName=="eUME_Radian")
      return eUME_Radian;
   else if (aName=="eUME_Image")
      return eUME_Image;
   else if (aName=="eUME_Terrain")
      return eUME_Terrain;
   else if (aName=="eUME_Naturel")
      return eUME_Naturel;
  else
  {
      cout << aName << " is not a correct value for enum eUniteMesureErreur\n" ;
      ELISE_ASSERT(false,"XML enum value error");
  }
  return (eUniteMesureErreur) 0;
}
void xml_init(eUniteMesureErreur & aVal,cElXMLTree * aTree)
{
   aVal= Str2eUniteMesureErreur(aTree->Contenu());
}
std::string  eToString(const eUniteMesureErreur & anObj)
{
   if (anObj==eUME_Radian)
      return  "eUME_Radian";
   if (anObj==eUME_Image)
      return  "eUME_Image";
   if (anObj==eUME_Terrain)
      return  "eUME_Terrain";
   if (anObj==eUME_Naturel)
      return  "eUME_Naturel";
 std::cout << "Enum = eUniteMesureErreur\n";
   ELISE_ASSERT(false,"Bad Value in eToString for enum value ");
   return "";
}

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eUniteMesureErreur & anObj)
{
      return  cElXMLTree::ValueNode(aNameTag,eToString(anObj));
}

eNiveauShowMessage  Str2eNiveauShowMessage(const std::string & aName)
{
   if (aName=="eNSM_None")
      return eNSM_None;
   else if (aName=="eNSM_Iter")
      return eNSM_Iter;
   else if (aName=="eNSM_Paquet")
      return eNSM_Paquet;
   else if (aName=="eNSM_Percentile")
      return eNSM_Percentile;
   else if (aName=="eNSM_CpleIm")
      return eNSM_CpleIm;
   else if (aName=="eNSM_Indiv")
      return eNSM_Indiv;
  else
  {
      cout << aName << " is not a correct value for enum eNiveauShowMessage\n" ;
      ELISE_ASSERT(false,"XML enum value error");
  }
  return (eNiveauShowMessage) 0;
}
void xml_init(eNiveauShowMessage & aVal,cElXMLTree * aTree)
{
   aVal= Str2eNiveauShowMessage(aTree->Contenu());
}
std::string  eToString(const eNiveauShowMessage & anObj)
{
   if (anObj==eNSM_None)
      return  "eNSM_None";
   if (anObj==eNSM_Iter)
      return  "eNSM_Iter";
   if (anObj==eNSM_Paquet)
      return  "eNSM_Paquet";
   if (anObj==eNSM_Percentile)
      return  "eNSM_Percentile";
   if (anObj==eNSM_CpleIm)
      return  "eNSM_CpleIm";
   if (anObj==eNSM_Indiv)
      return  "eNSM_Indiv";
 std::cout << "Enum = eNiveauShowMessage\n";
   ELISE_ASSERT(false,"Bad Value in eToString for enum value ");
   return "";
}

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eNiveauShowMessage & anObj)
{
      return  cElXMLTree::ValueNode(aNameTag,eToString(anObj));
}

eModePointLiaison  Str2eModePointLiaison(const std::string & aName)
{
   if (aName=="eMPL_DbleCoplanIm")
      return eMPL_DbleCoplanIm;
   else if (aName=="eMPL_PtTerrainInc")
      return eMPL_PtTerrainInc;
  else
  {
      cout << aName << " is not a correct value for enum eModePointLiaison\n" ;
      ELISE_ASSERT(false,"XML enum value error");
  }
  return (eModePointLiaison) 0;
}
void xml_init(eModePointLiaison & aVal,cElXMLTree * aTree)
{
   aVal= Str2eModePointLiaison(aTree->Contenu());
}
std::string  eToString(const eModePointLiaison & anObj)
{
   if (anObj==eMPL_DbleCoplanIm)
      return  "eMPL_DbleCoplanIm";
   if (anObj==eMPL_PtTerrainInc)
      return  "eMPL_PtTerrainInc";
 std::cout << "Enum = eModePointLiaison\n";
   ELISE_ASSERT(false,"Bad Value in eToString for enum value ");
   return "";
}

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eModePointLiaison & anObj)
{
      return  cElXMLTree::ValueNode(aNameTag,eToString(anObj));
}


std::string & cPowPointLiaisons::Id()
{
   return mId;
}

const std::string & cPowPointLiaisons::Id()const 
{
   return mId;
}


int & cPowPointLiaisons::NbTot()
{
   return mNbTot;
}

const int & cPowPointLiaisons::NbTot()const 
{
   return mNbTot;
}


cTplValGesInit< double > & cPowPointLiaisons::Pds()
{
   return mPds;
}

const cTplValGesInit< double > & cPowPointLiaisons::Pds()const 
{
   return mPds;
}

cElXMLTree * ToXMLTree(const cPowPointLiaisons & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"PowPointLiaisons",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Id"),anObj.Id())->ReTagThis("Id"));
   aRes->AddFils(::ToXMLTree(std::string("NbTot"),anObj.NbTot())->ReTagThis("NbTot"));
   if (anObj.Pds().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Pds"),anObj.Pds().Val())->ReTagThis("Pds"));
  return aRes;
}

void xml_init(cPowPointLiaisons & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.Id(),aTree->Get("Id",1)); //tototo 

   xml_init(anObj.NbTot(),aTree->Get("NbTot",1)); //tototo 

   xml_init(anObj.Pds(),aTree->Get("Pds",1),double(1.0)); //tototo 
}


std::list< cPowPointLiaisons > & cOptimizationPowel::PowPointLiaisons()
{
   return mPowPointLiaisons;
}

const std::list< cPowPointLiaisons > & cOptimizationPowel::PowPointLiaisons()const 
{
   return mPowPointLiaisons;
}

cElXMLTree * ToXMLTree(const cOptimizationPowel & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"OptimizationPowel",eXMLBranche);
  for
  (       std::list< cPowPointLiaisons >::const_iterator it=anObj.PowPointLiaisons().begin();
      it !=anObj.PowPointLiaisons().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("PowPointLiaisons"));
  return aRes;
}

void xml_init(cOptimizationPowel & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.PowPointLiaisons(),aTree->GetAll("PowPointLiaisons",false,1));
}


cTplValGesInit< int > & cShowPbLiaison::NbMinPtsMul()
{
   return mNbMinPtsMul;
}

const cTplValGesInit< int > & cShowPbLiaison::NbMinPtsMul()const 
{
   return mNbMinPtsMul;
}


cTplValGesInit< bool > & cShowPbLiaison::Actif()
{
   return mActif;
}

const cTplValGesInit< bool > & cShowPbLiaison::Actif()const 
{
   return mActif;
}


cTplValGesInit< bool > & cShowPbLiaison::GetCharOnPb()
{
   return mGetCharOnPb;
}

const cTplValGesInit< bool > & cShowPbLiaison::GetCharOnPb()const 
{
   return mGetCharOnPb;
}

cElXMLTree * ToXMLTree(const cShowPbLiaison & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ShowPbLiaison",eXMLBranche);
   if (anObj.NbMinPtsMul().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NbMinPtsMul"),anObj.NbMinPtsMul().Val())->ReTagThis("NbMinPtsMul"));
   if (anObj.Actif().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Actif"),anObj.Actif().Val())->ReTagThis("Actif"));
   if (anObj.GetCharOnPb().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("GetCharOnPb"),anObj.GetCharOnPb().Val())->ReTagThis("GetCharOnPb"));
  return aRes;
}

void xml_init(cShowPbLiaison & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.NbMinPtsMul(),aTree->Get("NbMinPtsMul",1)); //tototo 

   xml_init(anObj.Actif(),aTree->Get("Actif",1),bool(true)); //tototo 

   xml_init(anObj.GetCharOnPb(),aTree->Get("GetCharOnPb",1),bool(true)); //tototo 
}


double & cPonderationPackMesure::EcartMesureIndiv()
{
   return mEcartMesureIndiv;
}

const double & cPonderationPackMesure::EcartMesureIndiv()const 
{
   return mEcartMesureIndiv;
}


cTplValGesInit< bool > & cPonderationPackMesure::Add2Compens()
{
   return mAdd2Compens;
}

const cTplValGesInit< bool > & cPonderationPackMesure::Add2Compens()const 
{
   return mAdd2Compens;
}


cTplValGesInit< eModePonderationRobuste > & cPonderationPackMesure::ModePonderation()
{
   return mModePonderation;
}

const cTplValGesInit< eModePonderationRobuste > & cPonderationPackMesure::ModePonderation()const 
{
   return mModePonderation;
}


cTplValGesInit< double > & cPonderationPackMesure::EcartMax()
{
   return mEcartMax;
}

const cTplValGesInit< double > & cPonderationPackMesure::EcartMax()const 
{
   return mEcartMax;
}


cTplValGesInit< double > & cPonderationPackMesure::ExposantLK()
{
   return mExposantLK;
}

const cTplValGesInit< double > & cPonderationPackMesure::ExposantLK()const 
{
   return mExposantLK;
}


cTplValGesInit< double > & cPonderationPackMesure::SigmaPond()
{
   return mSigmaPond;
}

const cTplValGesInit< double > & cPonderationPackMesure::SigmaPond()const 
{
   return mSigmaPond;
}


cTplValGesInit< double > & cPonderationPackMesure::NbMax()
{
   return mNbMax;
}

const cTplValGesInit< double > & cPonderationPackMesure::NbMax()const 
{
   return mNbMax;
}


cTplValGesInit< eNiveauShowMessage > & cPonderationPackMesure::Show()
{
   return mShow;
}

const cTplValGesInit< eNiveauShowMessage > & cPonderationPackMesure::Show()const 
{
   return mShow;
}


cTplValGesInit< bool > & cPonderationPackMesure::GetChar()
{
   return mGetChar;
}

const cTplValGesInit< bool > & cPonderationPackMesure::GetChar()const 
{
   return mGetChar;
}


cTplValGesInit< int > & cPonderationPackMesure::NbMinMultShowIndiv()
{
   return mNbMinMultShowIndiv;
}

const cTplValGesInit< int > & cPonderationPackMesure::NbMinMultShowIndiv()const 
{
   return mNbMinMultShowIndiv;
}


cTplValGesInit< std::vector<double> > & cPonderationPackMesure::ShowPercentile()
{
   return mShowPercentile;
}

const cTplValGesInit< std::vector<double> > & cPonderationPackMesure::ShowPercentile()const 
{
   return mShowPercentile;
}


cTplValGesInit< double > & cPonderationPackMesure::ExposantPoidsMult()
{
   return mExposantPoidsMult;
}

const cTplValGesInit< double > & cPonderationPackMesure::ExposantPoidsMult()const 
{
   return mExposantPoidsMult;
}


cTplValGesInit< std::string > & cPonderationPackMesure::IdFilter3D()
{
   return mIdFilter3D;
}

const cTplValGesInit< std::string > & cPonderationPackMesure::IdFilter3D()const 
{
   return mIdFilter3D;
}

cElXMLTree * ToXMLTree(const cPonderationPackMesure & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"PonderationPackMesure",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("EcartMesureIndiv"),anObj.EcartMesureIndiv())->ReTagThis("EcartMesureIndiv"));
   if (anObj.Add2Compens().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Add2Compens"),anObj.Add2Compens().Val())->ReTagThis("Add2Compens"));
   if (anObj.ModePonderation().IsInit())
      aRes->AddFils(ToXMLTree(std::string("ModePonderation"),anObj.ModePonderation().Val())->ReTagThis("ModePonderation"));
   if (anObj.EcartMax().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("EcartMax"),anObj.EcartMax().Val())->ReTagThis("EcartMax"));
   if (anObj.ExposantLK().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ExposantLK"),anObj.ExposantLK().Val())->ReTagThis("ExposantLK"));
   if (anObj.SigmaPond().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SigmaPond"),anObj.SigmaPond().Val())->ReTagThis("SigmaPond"));
   if (anObj.NbMax().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NbMax"),anObj.NbMax().Val())->ReTagThis("NbMax"));
   if (anObj.Show().IsInit())
      aRes->AddFils(ToXMLTree(std::string("Show"),anObj.Show().Val())->ReTagThis("Show"));
   if (anObj.GetChar().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("GetChar"),anObj.GetChar().Val())->ReTagThis("GetChar"));
   if (anObj.NbMinMultShowIndiv().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NbMinMultShowIndiv"),anObj.NbMinMultShowIndiv().Val())->ReTagThis("NbMinMultShowIndiv"));
   if (anObj.ShowPercentile().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ShowPercentile"),anObj.ShowPercentile().Val())->ReTagThis("ShowPercentile"));
   if (anObj.ExposantPoidsMult().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ExposantPoidsMult"),anObj.ExposantPoidsMult().Val())->ReTagThis("ExposantPoidsMult"));
   if (anObj.IdFilter3D().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("IdFilter3D"),anObj.IdFilter3D().Val())->ReTagThis("IdFilter3D"));
  return aRes;
}

void xml_init(cPonderationPackMesure & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.EcartMesureIndiv(),aTree->Get("EcartMesureIndiv",1)); //tototo 

   xml_init(anObj.Add2Compens(),aTree->Get("Add2Compens",1),bool(true)); //tototo 

   xml_init(anObj.ModePonderation(),aTree->Get("ModePonderation",1),eModePonderationRobuste(ePondL2)); //tototo 

   xml_init(anObj.EcartMax(),aTree->Get("EcartMax",1),double(1e20)); //tototo 

   xml_init(anObj.ExposantLK(),aTree->Get("ExposantLK",1),double(1.2)); //tototo 

   xml_init(anObj.SigmaPond(),aTree->Get("SigmaPond",1),double(1e20)); //tototo 

   xml_init(anObj.NbMax(),aTree->Get("NbMax",1),double(1e20)); //tototo 

   xml_init(anObj.Show(),aTree->Get("Show",1),eNiveauShowMessage(eNSM_None)); //tototo 

   xml_init(anObj.GetChar(),aTree->Get("GetChar",1),bool(false)); //tototo 

   xml_init(anObj.NbMinMultShowIndiv(),aTree->Get("NbMinMultShowIndiv",1),int(2)); //tototo 

   xml_init(anObj.ShowPercentile(),aTree->Get("ShowPercentile",1)); //tototo 

   xml_init(anObj.ExposantPoidsMult(),aTree->Get("ExposantPoidsMult",1),double(1)); //tototo 

   xml_init(anObj.IdFilter3D(),aTree->Get("IdFilter3D",1)); //tototo 
}


cTplValGesInit< std::string > & cParamEstimPlan::AttrSup()
{
   return mAttrSup;
}

const cTplValGesInit< std::string > & cParamEstimPlan::AttrSup()const 
{
   return mAttrSup;
}


cTplValGesInit< std::string > & cParamEstimPlan::KeyCalculMasq()
{
   return mKeyCalculMasq;
}

const cTplValGesInit< std::string > & cParamEstimPlan::KeyCalculMasq()const 
{
   return mKeyCalculMasq;
}


std::string & cParamEstimPlan::IdBdl()
{
   return mIdBdl;
}

const std::string & cParamEstimPlan::IdBdl()const 
{
   return mIdBdl;
}


cPonderationPackMesure & cParamEstimPlan::Pond()
{
   return mPond;
}

const cPonderationPackMesure & cParamEstimPlan::Pond()const 
{
   return mPond;
}


cTplValGesInit< double > & cParamEstimPlan::LimBSurH()
{
   return mLimBSurH;
}

const cTplValGesInit< double > & cParamEstimPlan::LimBSurH()const 
{
   return mLimBSurH;
}

cElXMLTree * ToXMLTree(const cParamEstimPlan & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ParamEstimPlan",eXMLBranche);
   if (anObj.AttrSup().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("AttrSup"),anObj.AttrSup().Val())->ReTagThis("AttrSup"));
   if (anObj.KeyCalculMasq().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("KeyCalculMasq"),anObj.KeyCalculMasq().Val())->ReTagThis("KeyCalculMasq"));
   aRes->AddFils(::ToXMLTree(std::string("IdBdl"),anObj.IdBdl())->ReTagThis("IdBdl"));
   aRes->AddFils(ToXMLTree(anObj.Pond())->ReTagThis("Pond"));
   if (anObj.LimBSurH().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("LimBSurH"),anObj.LimBSurH().Val())->ReTagThis("LimBSurH"));
  return aRes;
}

void xml_init(cParamEstimPlan & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.AttrSup(),aTree->Get("AttrSup",1)); //tototo 

   xml_init(anObj.KeyCalculMasq(),aTree->Get("KeyCalculMasq",1)); //tototo 

   xml_init(anObj.IdBdl(),aTree->Get("IdBdl",1)); //tototo 

   xml_init(anObj.Pond(),aTree->Get("Pond",1)); //tototo 

   xml_init(anObj.LimBSurH(),aTree->Get("LimBSurH",1),double(1e-2)); //tototo 
}


Pt2dr & cAperoPointeStereo::P1()
{
   return mP1;
}

const Pt2dr & cAperoPointeStereo::P1()const 
{
   return mP1;
}


std::string & cAperoPointeStereo::Im1()
{
   return mIm1;
}

const std::string & cAperoPointeStereo::Im1()const 
{
   return mIm1;
}


Pt2dr & cAperoPointeStereo::P2()
{
   return mP2;
}

const Pt2dr & cAperoPointeStereo::P2()const 
{
   return mP2;
}


std::string & cAperoPointeStereo::Im2()
{
   return mIm2;
}

const std::string & cAperoPointeStereo::Im2()const 
{
   return mIm2;
}

cElXMLTree * ToXMLTree(const cAperoPointeStereo & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"AperoPointeStereo",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("P1"),anObj.P1())->ReTagThis("P1"));
   aRes->AddFils(::ToXMLTree(std::string("Im1"),anObj.Im1())->ReTagThis("Im1"));
   aRes->AddFils(::ToXMLTree(std::string("P2"),anObj.P2())->ReTagThis("P2"));
   aRes->AddFils(::ToXMLTree(std::string("Im2"),anObj.Im2())->ReTagThis("Im2"));
  return aRes;
}

void xml_init(cAperoPointeStereo & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.P1(),aTree->Get("P1",1)); //tototo 

   xml_init(anObj.Im1(),aTree->Get("Im1",1)); //tototo 

   xml_init(anObj.P2(),aTree->Get("P2",1)); //tototo 

   xml_init(anObj.Im2(),aTree->Get("Im2",1)); //tototo 
}


Pt2dr & cAperoPointeMono::Pt()
{
   return mPt;
}

const Pt2dr & cAperoPointeMono::Pt()const 
{
   return mPt;
}


std::string & cAperoPointeMono::Im()
{
   return mIm;
}

const std::string & cAperoPointeMono::Im()const 
{
   return mIm;
}

cElXMLTree * ToXMLTree(const cAperoPointeMono & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"AperoPointeMono",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Pt"),anObj.Pt())->ReTagThis("Pt"));
   aRes->AddFils(::ToXMLTree(std::string("Im"),anObj.Im())->ReTagThis("Im"));
  return aRes;
}

void xml_init(cAperoPointeMono & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.Pt(),aTree->Get("Pt",1)); //tototo 

   xml_init(anObj.Im(),aTree->Get("Im",1)); //tototo 
}


std::string & cApero2PointeFromFile::File()
{
   return mFile;
}

const std::string & cApero2PointeFromFile::File()const 
{
   return mFile;
}


std::string & cApero2PointeFromFile::NameP1()
{
   return mNameP1;
}

const std::string & cApero2PointeFromFile::NameP1()const 
{
   return mNameP1;
}


std::string & cApero2PointeFromFile::NameP2()
{
   return mNameP2;
}

const std::string & cApero2PointeFromFile::NameP2()const 
{
   return mNameP2;
}

cElXMLTree * ToXMLTree(const cApero2PointeFromFile & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"Apero2PointeFromFile",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("File"),anObj.File())->ReTagThis("File"));
   aRes->AddFils(::ToXMLTree(std::string("NameP1"),anObj.NameP1())->ReTagThis("NameP1"));
   aRes->AddFils(::ToXMLTree(std::string("NameP2"),anObj.NameP2())->ReTagThis("NameP2"));
  return aRes;
}

void xml_init(cApero2PointeFromFile & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.File(),aTree->Get("File",1)); //tototo 

   xml_init(anObj.NameP1(),aTree->Get("NameP1",1)); //tototo 

   xml_init(anObj.NameP2(),aTree->Get("NameP2",1)); //tototo 
}


cElRegex_Ptr & cParamForceRappel::PatternNameApply()
{
   return mPatternNameApply;
}

const cElRegex_Ptr & cParamForceRappel::PatternNameApply()const 
{
   return mPatternNameApply;
}


std::vector< double > & cParamForceRappel::Incertitude()
{
   return mIncertitude;
}

const std::vector< double > & cParamForceRappel::Incertitude()const 
{
   return mIncertitude;
}


cTplValGesInit< bool > & cParamForceRappel::OnCur()
{
   return mOnCur;
}

const cTplValGesInit< bool > & cParamForceRappel::OnCur()const 
{
   return mOnCur;
}

cElXMLTree * ToXMLTree(const cParamForceRappel & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ParamForceRappel",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("PatternNameApply"),anObj.PatternNameApply())->ReTagThis("PatternNameApply"));
  for
  (       std::vector< double >::const_iterator it=anObj.Incertitude().begin();
      it !=anObj.Incertitude().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("Incertitude"),(*it))->ReTagThis("Incertitude"));
   if (anObj.OnCur().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("OnCur"),anObj.OnCur().Val())->ReTagThis("OnCur"));
  return aRes;
}

void xml_init(cParamForceRappel & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.PatternNameApply(),aTree->Get("PatternNameApply",1)); //tototo 

   xml_init(anObj.Incertitude(),aTree->GetAll("Incertitude",false,1));

   xml_init(anObj.OnCur(),aTree->Get("OnCur",1)); //tototo 
}


cParamForceRappel & cRappelOnAngles::ParamF()
{
   return mParamF;
}

const cParamForceRappel & cRappelOnAngles::ParamF()const 
{
   return mParamF;
}


std::vector< int > & cRappelOnAngles::TetaApply()
{
   return mTetaApply;
}

const std::vector< int > & cRappelOnAngles::TetaApply()const 
{
   return mTetaApply;
}

cElXMLTree * ToXMLTree(const cRappelOnAngles & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"RappelOnAngles",eXMLBranche);
   aRes->AddFils(ToXMLTree(anObj.ParamF())->ReTagThis("ParamF"));
  for
  (       std::vector< int >::const_iterator it=anObj.TetaApply().begin();
      it !=anObj.TetaApply().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("TetaApply"),(*it))->ReTagThis("TetaApply"));
  return aRes;
}

void xml_init(cRappelOnAngles & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.ParamF(),aTree->Get("ParamF",1)); //tototo 

   xml_init(anObj.TetaApply(),aTree->GetAll("TetaApply",false,1));
}


cParamForceRappel & cRappelOnCentres::ParamF()
{
   return mParamF;
}

const cParamForceRappel & cRappelOnCentres::ParamF()const 
{
   return mParamF;
}


cTplValGesInit< bool > & cRappelOnCentres::OnlyWhenNoCentreInit()
{
   return mOnlyWhenNoCentreInit;
}

const cTplValGesInit< bool > & cRappelOnCentres::OnlyWhenNoCentreInit()const 
{
   return mOnlyWhenNoCentreInit;
}

cElXMLTree * ToXMLTree(const cRappelOnCentres & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"RappelOnCentres",eXMLBranche);
   aRes->AddFils(ToXMLTree(anObj.ParamF())->ReTagThis("ParamF"));
   if (anObj.OnlyWhenNoCentreInit().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("OnlyWhenNoCentreInit"),anObj.OnlyWhenNoCentreInit().Val())->ReTagThis("OnlyWhenNoCentreInit"));
  return aRes;
}

void xml_init(cRappelOnCentres & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.ParamF(),aTree->Get("ParamF",1)); //tototo 

   xml_init(anObj.OnlyWhenNoCentreInit(),aTree->Get("OnlyWhenNoCentreInit",1),bool(true)); //tototo 
}


std::list< cRappelOnAngles > & cSectionLevenbergMarkard::RappelOnAngles()
{
   return mRappelOnAngles;
}

const std::list< cRappelOnAngles > & cSectionLevenbergMarkard::RappelOnAngles()const 
{
   return mRappelOnAngles;
}


std::list< cRappelOnCentres > & cSectionLevenbergMarkard::RappelOnCentres()
{
   return mRappelOnCentres;
}

const std::list< cRappelOnCentres > & cSectionLevenbergMarkard::RappelOnCentres()const 
{
   return mRappelOnCentres;
}

cElXMLTree * ToXMLTree(const cSectionLevenbergMarkard & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"SectionLevenbergMarkard",eXMLBranche);
  for
  (       std::list< cRappelOnAngles >::const_iterator it=anObj.RappelOnAngles().begin();
      it !=anObj.RappelOnAngles().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("RappelOnAngles"));
  for
  (       std::list< cRappelOnCentres >::const_iterator it=anObj.RappelOnCentres().begin();
      it !=anObj.RappelOnCentres().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("RappelOnCentres"));
  return aRes;
}

void xml_init(cSectionLevenbergMarkard & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.RappelOnAngles(),aTree->GetAll("RappelOnAngles",false,1));

   xml_init(anObj.RappelOnCentres(),aTree->GetAll("RappelOnCentres",false,1));
}


std::string & cSetOrientationInterne::KeyFile()
{
   return mKeyFile;
}

const std::string & cSetOrientationInterne::KeyFile()const 
{
   return mKeyFile;
}


cTplValGesInit< std::string > & cSetOrientationInterne::PatternSel()
{
   return mPatternSel;
}

const cTplValGesInit< std::string > & cSetOrientationInterne::PatternSel()const 
{
   return mPatternSel;
}


cTplValGesInit< std::string > & cSetOrientationInterne::Tag()
{
   return mTag;
}

const cTplValGesInit< std::string > & cSetOrientationInterne::Tag()const 
{
   return mTag;
}


bool & cSetOrientationInterne::AddToCur()
{
   return mAddToCur;
}

const bool & cSetOrientationInterne::AddToCur()const 
{
   return mAddToCur;
}


bool & cSetOrientationInterne::M2C()
{
   return mM2C;
}

const bool & cSetOrientationInterne::M2C()const 
{
   return mM2C;
}

cElXMLTree * ToXMLTree(const cSetOrientationInterne & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"SetOrientationInterne",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("KeyFile"),anObj.KeyFile())->ReTagThis("KeyFile"));
   if (anObj.PatternSel().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PatternSel"),anObj.PatternSel().Val())->ReTagThis("PatternSel"));
   if (anObj.Tag().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Tag"),anObj.Tag().Val())->ReTagThis("Tag"));
   aRes->AddFils(::ToXMLTree(std::string("AddToCur"),anObj.AddToCur())->ReTagThis("AddToCur"));
   aRes->AddFils(::ToXMLTree(std::string("M2C"),anObj.M2C())->ReTagThis("M2C"));
  return aRes;
}

void xml_init(cSetOrientationInterne & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.KeyFile(),aTree->Get("KeyFile",1)); //tototo 

   xml_init(anObj.PatternSel(),aTree->Get("PatternSel",1),std::string(".*")); //tototo 

   xml_init(anObj.Tag(),aTree->Get("Tag",1),std::string("AffinitePlane")); //tototo 

   xml_init(anObj.AddToCur(),aTree->Get("AddToCur",1)); //tototo 

   xml_init(anObj.M2C(),aTree->Get("M2C",1)); //tototo 
}


Pt2dr & cExportAsNewGrid::Step()
{
   return mStep;
}

const Pt2dr & cExportAsNewGrid::Step()const 
{
   return mStep;
}


cTplValGesInit< double > & cExportAsNewGrid::RayonInv()
{
   return mRayonInv;
}

const cTplValGesInit< double > & cExportAsNewGrid::RayonInv()const 
{
   return mRayonInv;
}


cTplValGesInit< double > & cExportAsNewGrid::RayonInvRelFE()
{
   return mRayonInvRelFE;
}

const cTplValGesInit< double > & cExportAsNewGrid::RayonInvRelFE()const 
{
   return mRayonInvRelFE;
}

cElXMLTree * ToXMLTree(const cExportAsNewGrid & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ExportAsNewGrid",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Step"),anObj.Step())->ReTagThis("Step"));
   if (anObj.RayonInv().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("RayonInv"),anObj.RayonInv().Val())->ReTagThis("RayonInv"));
   if (anObj.RayonInvRelFE().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("RayonInvRelFE"),anObj.RayonInvRelFE().Val())->ReTagThis("RayonInvRelFE"));
  return aRes;
}

void xml_init(cExportAsNewGrid & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.Step(),aTree->Get("Step",1)); //tototo 

   xml_init(anObj.RayonInv(),aTree->Get("RayonInv",1),double(-1)); //tototo 

   xml_init(anObj.RayonInvRelFE(),aTree->Get("RayonInvRelFE",1),double(-1)); //tototo 
}


cTplValGesInit< bool > & cShowSection::ShowMes()
{
   return mShowMes;
}

const cTplValGesInit< bool > & cShowSection::ShowMes()const 
{
   return mShowMes;
}


cTplValGesInit< std::string > & cShowSection::LogFile()
{
   return mLogFile;
}

const cTplValGesInit< std::string > & cShowSection::LogFile()const 
{
   return mLogFile;
}

cElXMLTree * ToXMLTree(const cShowSection & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ShowSection",eXMLBranche);
   if (anObj.ShowMes().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ShowMes"),anObj.ShowMes().Val())->ReTagThis("ShowMes"));
   if (anObj.LogFile().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("LogFile"),anObj.LogFile().Val())->ReTagThis("LogFile"));
  return aRes;
}

void xml_init(cShowSection & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.ShowMes(),aTree->Get("ShowMes",1),bool(true)); //tototo 

   xml_init(anObj.LogFile(),aTree->Get("LogFile",1)); //tototo 
}


Pt2dr & cSzImForInvY::SzIm1()
{
   return mSzIm1;
}

const Pt2dr & cSzImForInvY::SzIm1()const 
{
   return mSzIm1;
}


Pt2dr & cSzImForInvY::SzIm2()
{
   return mSzIm2;
}

const Pt2dr & cSzImForInvY::SzIm2()const 
{
   return mSzIm2;
}

cElXMLTree * ToXMLTree(const cSzImForInvY & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"SzImForInvY",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("SzIm1"),anObj.SzIm1())->ReTagThis("SzIm1"));
   aRes->AddFils(::ToXMLTree(std::string("SzIm2"),anObj.SzIm2())->ReTagThis("SzIm2"));
  return aRes;
}

void xml_init(cSzImForInvY & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.SzIm1(),aTree->Get("SzIm1",1)); //tototo 

   xml_init(anObj.SzIm2(),aTree->Get("SzIm2",1)); //tototo 
}


std::string & cSplitLayer::IdLayer()
{
   return mIdLayer;
}

const std::string & cSplitLayer::IdLayer()const 
{
   return mIdLayer;
}


std::string & cSplitLayer::KeyCalHomSplit()
{
   return mKeyCalHomSplit;
}

const std::string & cSplitLayer::KeyCalHomSplit()const 
{
   return mKeyCalHomSplit;
}

cElXMLTree * ToXMLTree(const cSplitLayer & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"SplitLayer",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("IdLayer"),anObj.IdLayer())->ReTagThis("IdLayer"));
   aRes->AddFils(::ToXMLTree(std::string("KeyCalHomSplit"),anObj.KeyCalHomSplit())->ReTagThis("KeyCalHomSplit"));
  return aRes;
}

void xml_init(cSplitLayer & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.IdLayer(),aTree->Get("IdLayer",1)); //tototo 

   xml_init(anObj.KeyCalHomSplit(),aTree->Get("KeyCalHomSplit",1)); //tototo 
}


cTplValGesInit< int > & cBDD_PtsLiaisons::TestForMatin()
{
   return mTestForMatin;
}

const cTplValGesInit< int > & cBDD_PtsLiaisons::TestForMatin()const 
{
   return mTestForMatin;
}


cTplValGesInit< bool > & cBDD_PtsLiaisons::UseAsPtMultiple()
{
   return mUseAsPtMultiple;
}

const cTplValGesInit< bool > & cBDD_PtsLiaisons::UseAsPtMultiple()const 
{
   return mUseAsPtMultiple;
}


std::string & cBDD_PtsLiaisons::Id()
{
   return mId;
}

const std::string & cBDD_PtsLiaisons::Id()const 
{
   return mId;
}


cTplValGesInit< std::string > & cBDD_PtsLiaisons::IdFilterSameGrp()
{
   return mIdFilterSameGrp;
}

const cTplValGesInit< std::string > & cBDD_PtsLiaisons::IdFilterSameGrp()const 
{
   return mIdFilterSameGrp;
}


cTplValGesInit< bool > & cBDD_PtsLiaisons::AutoSuprReflexif()
{
   return mAutoSuprReflexif;
}

const cTplValGesInit< bool > & cBDD_PtsLiaisons::AutoSuprReflexif()const 
{
   return mAutoSuprReflexif;
}


std::vector< std::string > & cBDD_PtsLiaisons::KeySet()
{
   return mKeySet;
}

const std::vector< std::string > & cBDD_PtsLiaisons::KeySet()const 
{
   return mKeySet;
}


std::vector< std::string > & cBDD_PtsLiaisons::KeyAssoc()
{
   return mKeyAssoc;
}

const std::vector< std::string > & cBDD_PtsLiaisons::KeyAssoc()const 
{
   return mKeyAssoc;
}


std::list< std::string > & cBDD_PtsLiaisons::XMLKeySetOrPat()
{
   return mXMLKeySetOrPat;
}

const std::list< std::string > & cBDD_PtsLiaisons::XMLKeySetOrPat()const 
{
   return mXMLKeySetOrPat;
}


Pt2dr & cBDD_PtsLiaisons::SzIm1()
{
   return SzImForInvY().Val().SzIm1();
}

const Pt2dr & cBDD_PtsLiaisons::SzIm1()const 
{
   return SzImForInvY().Val().SzIm1();
}


Pt2dr & cBDD_PtsLiaisons::SzIm2()
{
   return SzImForInvY().Val().SzIm2();
}

const Pt2dr & cBDD_PtsLiaisons::SzIm2()const 
{
   return SzImForInvY().Val().SzIm2();
}


cTplValGesInit< cSzImForInvY > & cBDD_PtsLiaisons::SzImForInvY()
{
   return mSzImForInvY;
}

const cTplValGesInit< cSzImForInvY > & cBDD_PtsLiaisons::SzImForInvY()const 
{
   return mSzImForInvY;
}


std::string & cBDD_PtsLiaisons::IdLayer()
{
   return SplitLayer().Val().IdLayer();
}

const std::string & cBDD_PtsLiaisons::IdLayer()const 
{
   return SplitLayer().Val().IdLayer();
}


std::string & cBDD_PtsLiaisons::KeyCalHomSplit()
{
   return SplitLayer().Val().KeyCalHomSplit();
}

const std::string & cBDD_PtsLiaisons::KeyCalHomSplit()const 
{
   return SplitLayer().Val().KeyCalHomSplit();
}


cTplValGesInit< cSplitLayer > & cBDD_PtsLiaisons::SplitLayer()
{
   return mSplitLayer;
}

const cTplValGesInit< cSplitLayer > & cBDD_PtsLiaisons::SplitLayer()const 
{
   return mSplitLayer;
}

cElXMLTree * ToXMLTree(const cBDD_PtsLiaisons & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"BDD_PtsLiaisons",eXMLBranche);
   if (anObj.TestForMatin().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("TestForMatin"),anObj.TestForMatin().Val())->ReTagThis("TestForMatin"));
   if (anObj.UseAsPtMultiple().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("UseAsPtMultiple"),anObj.UseAsPtMultiple().Val())->ReTagThis("UseAsPtMultiple"));
   aRes->AddFils(::ToXMLTree(std::string("Id"),anObj.Id())->ReTagThis("Id"));
   if (anObj.IdFilterSameGrp().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("IdFilterSameGrp"),anObj.IdFilterSameGrp().Val())->ReTagThis("IdFilterSameGrp"));
   if (anObj.AutoSuprReflexif().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("AutoSuprReflexif"),anObj.AutoSuprReflexif().Val())->ReTagThis("AutoSuprReflexif"));
  for
  (       std::vector< std::string >::const_iterator it=anObj.KeySet().begin();
      it !=anObj.KeySet().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("KeySet"),(*it))->ReTagThis("KeySet"));
  for
  (       std::vector< std::string >::const_iterator it=anObj.KeyAssoc().begin();
      it !=anObj.KeyAssoc().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("KeyAssoc"),(*it))->ReTagThis("KeyAssoc"));
  for
  (       std::list< std::string >::const_iterator it=anObj.XMLKeySetOrPat().begin();
      it !=anObj.XMLKeySetOrPat().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("XMLKeySetOrPat"),(*it))->ReTagThis("XMLKeySetOrPat"));
   if (anObj.SzImForInvY().IsInit())
      aRes->AddFils(ToXMLTree(anObj.SzImForInvY().Val())->ReTagThis("SzImForInvY"));
   if (anObj.SplitLayer().IsInit())
      aRes->AddFils(ToXMLTree(anObj.SplitLayer().Val())->ReTagThis("SplitLayer"));
  return aRes;
}

void xml_init(cBDD_PtsLiaisons & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.TestForMatin(),aTree->Get("TestForMatin",1)); //tototo 

   xml_init(anObj.UseAsPtMultiple(),aTree->Get("UseAsPtMultiple",1)); //tototo 

   xml_init(anObj.Id(),aTree->Get("Id",1)); //tototo 

   xml_init(anObj.IdFilterSameGrp(),aTree->Get("IdFilterSameGrp",1)); //tototo 

   xml_init(anObj.AutoSuprReflexif(),aTree->Get("AutoSuprReflexif",1),bool(true)); //tototo 

   xml_init(anObj.KeySet(),aTree->GetAll("KeySet",false,1));

   xml_init(anObj.KeyAssoc(),aTree->GetAll("KeyAssoc",false,1));

   xml_init(anObj.XMLKeySetOrPat(),aTree->GetAll("XMLKeySetOrPat",false,1));

   xml_init(anObj.SzImForInvY(),aTree->Get("SzImForInvY",1)); //tototo 

   xml_init(anObj.SplitLayer(),aTree->Get("SplitLayer",1)); //tototo 
}


double & cBddApp_AutoNum::DistFusion()
{
   return mDistFusion;
}

const double & cBddApp_AutoNum::DistFusion()const 
{
   return mDistFusion;
}


double & cBddApp_AutoNum::DistAmbiguite()
{
   return mDistAmbiguite;
}

const double & cBddApp_AutoNum::DistAmbiguite()const 
{
   return mDistAmbiguite;
}

cElXMLTree * ToXMLTree(const cBddApp_AutoNum & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"BddApp_AutoNum",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("DistFusion"),anObj.DistFusion())->ReTagThis("DistFusion"));
   aRes->AddFils(::ToXMLTree(std::string("DistAmbiguite"),anObj.DistAmbiguite())->ReTagThis("DistAmbiguite"));
  return aRes;
}

void xml_init(cBddApp_AutoNum & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.DistFusion(),aTree->Get("DistFusion",1)); //tototo 

   xml_init(anObj.DistAmbiguite(),aTree->Get("DistAmbiguite",1)); //tototo 
}


std::string & cBDD_PtsAppuis::Id()
{
   return mId;
}

const std::string & cBDD_PtsAppuis::Id()const 
{
   return mId;
}


std::string & cBDD_PtsAppuis::KeySet()
{
   return mKeySet;
}

const std::string & cBDD_PtsAppuis::KeySet()const 
{
   return mKeySet;
}


std::string & cBDD_PtsAppuis::KeyAssoc()
{
   return mKeyAssoc;
}

const std::string & cBDD_PtsAppuis::KeyAssoc()const 
{
   return mKeyAssoc;
}


cTplValGesInit< Pt2dr > & cBDD_PtsAppuis::SzImForInvY()
{
   return mSzImForInvY;
}

const cTplValGesInit< Pt2dr > & cBDD_PtsAppuis::SzImForInvY()const 
{
   return mSzImForInvY;
}


cTplValGesInit< bool > & cBDD_PtsAppuis::InvXY()
{
   return mInvXY;
}

const cTplValGesInit< bool > & cBDD_PtsAppuis::InvXY()const 
{
   return mInvXY;
}


cTplValGesInit< Pt3dr > & cBDD_PtsAppuis::ToSubstract()
{
   return mToSubstract;
}

const cTplValGesInit< Pt3dr > & cBDD_PtsAppuis::ToSubstract()const 
{
   return mToSubstract;
}


cTplValGesInit< std::string > & cBDD_PtsAppuis::TagExtract()
{
   return mTagExtract;
}

const cTplValGesInit< std::string > & cBDD_PtsAppuis::TagExtract()const 
{
   return mTagExtract;
}


double & cBDD_PtsAppuis::DistFusion()
{
   return BddApp_AutoNum().Val().DistFusion();
}

const double & cBDD_PtsAppuis::DistFusion()const 
{
   return BddApp_AutoNum().Val().DistFusion();
}


double & cBDD_PtsAppuis::DistAmbiguite()
{
   return BddApp_AutoNum().Val().DistAmbiguite();
}

const double & cBDD_PtsAppuis::DistAmbiguite()const 
{
   return BddApp_AutoNum().Val().DistAmbiguite();
}


cTplValGesInit< cBddApp_AutoNum > & cBDD_PtsAppuis::BddApp_AutoNum()
{
   return mBddApp_AutoNum;
}

const cTplValGesInit< cBddApp_AutoNum > & cBDD_PtsAppuis::BddApp_AutoNum()const 
{
   return mBddApp_AutoNum;
}

cElXMLTree * ToXMLTree(const cBDD_PtsAppuis & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"BDD_PtsAppuis",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Id"),anObj.Id())->ReTagThis("Id"));
   aRes->AddFils(::ToXMLTree(std::string("KeySet"),anObj.KeySet())->ReTagThis("KeySet"));
   aRes->AddFils(::ToXMLTree(std::string("KeyAssoc"),anObj.KeyAssoc())->ReTagThis("KeyAssoc"));
   if (anObj.SzImForInvY().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SzImForInvY"),anObj.SzImForInvY().Val())->ReTagThis("SzImForInvY"));
   if (anObj.InvXY().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("InvXY"),anObj.InvXY().Val())->ReTagThis("InvXY"));
   if (anObj.ToSubstract().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ToSubstract"),anObj.ToSubstract().Val())->ReTagThis("ToSubstract"));
   if (anObj.TagExtract().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("TagExtract"),anObj.TagExtract().Val())->ReTagThis("TagExtract"));
   if (anObj.BddApp_AutoNum().IsInit())
      aRes->AddFils(ToXMLTree(anObj.BddApp_AutoNum().Val())->ReTagThis("BddApp_AutoNum"));
  return aRes;
}

void xml_init(cBDD_PtsAppuis & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.Id(),aTree->Get("Id",1)); //tototo 

   xml_init(anObj.KeySet(),aTree->Get("KeySet",1)); //tototo 

   xml_init(anObj.KeyAssoc(),aTree->Get("KeyAssoc",1)); //tototo 

   xml_init(anObj.SzImForInvY(),aTree->Get("SzImForInvY",1)); //tototo 

   xml_init(anObj.InvXY(),aTree->Get("InvXY",1),bool(false)); //tototo 

   xml_init(anObj.ToSubstract(),aTree->Get("ToSubstract",1)); //tototo 

   xml_init(anObj.TagExtract(),aTree->Get("TagExtract",1),std::string("ListeAppuis1Im")); //tototo 

   xml_init(anObj.BddApp_AutoNum(),aTree->Get("BddApp_AutoNum",1)); //tototo 
}


cTplValGesInit< Pt2dr > & cBDD_ObsAppuisFlottant::OffsetIm()
{
   return mOffsetIm;
}

const cTplValGesInit< Pt2dr > & cBDD_ObsAppuisFlottant::OffsetIm()const 
{
   return mOffsetIm;
}


std::string & cBDD_ObsAppuisFlottant::Id()
{
   return mId;
}

const std::string & cBDD_ObsAppuisFlottant::Id()const 
{
   return mId;
}


std::string & cBDD_ObsAppuisFlottant::KeySetOrPat()
{
   return mKeySetOrPat;
}

const std::string & cBDD_ObsAppuisFlottant::KeySetOrPat()const 
{
   return mKeySetOrPat;
}


cTplValGesInit< std::string > & cBDD_ObsAppuisFlottant::NameAppuiSelector()
{
   return mNameAppuiSelector;
}

const cTplValGesInit< std::string > & cBDD_ObsAppuisFlottant::NameAppuiSelector()const 
{
   return mNameAppuiSelector;
}


cTplValGesInit< bool > & cBDD_ObsAppuisFlottant::AcceptNoGround()
{
   return mAcceptNoGround;
}

const cTplValGesInit< bool > & cBDD_ObsAppuisFlottant::AcceptNoGround()const 
{
   return mAcceptNoGround;
}

cElXMLTree * ToXMLTree(const cBDD_ObsAppuisFlottant & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"BDD_ObsAppuisFlottant",eXMLBranche);
   if (anObj.OffsetIm().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("OffsetIm"),anObj.OffsetIm().Val())->ReTagThis("OffsetIm"));
   aRes->AddFils(::ToXMLTree(std::string("Id"),anObj.Id())->ReTagThis("Id"));
   aRes->AddFils(::ToXMLTree(std::string("KeySetOrPat"),anObj.KeySetOrPat())->ReTagThis("KeySetOrPat"));
   if (anObj.NameAppuiSelector().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NameAppuiSelector"),anObj.NameAppuiSelector().Val())->ReTagThis("NameAppuiSelector"));
   if (anObj.AcceptNoGround().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("AcceptNoGround"),anObj.AcceptNoGround().Val())->ReTagThis("AcceptNoGround"));
  return aRes;
}

void xml_init(cBDD_ObsAppuisFlottant & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.OffsetIm(),aTree->Get("OffsetIm",1),Pt2dr(Pt2dr(0,0))); //tototo 

   xml_init(anObj.Id(),aTree->Get("Id",1)); //tototo 

   xml_init(anObj.KeySetOrPat(),aTree->Get("KeySetOrPat",1)); //tototo 

   xml_init(anObj.NameAppuiSelector(),aTree->Get("NameAppuiSelector",1)); //tototo 

   xml_init(anObj.AcceptNoGround(),aTree->Get("AcceptNoGround",1),bool(false)); //tototo 
}


std::string & cBDD_Orient::Id()
{
   return mId;
}

const std::string & cBDD_Orient::Id()const 
{
   return mId;
}


std::string & cBDD_Orient::KeySet()
{
   return mKeySet;
}

const std::string & cBDD_Orient::KeySet()const 
{
   return mKeySet;
}


std::string & cBDD_Orient::KeyAssoc()
{
   return mKeyAssoc;
}

const std::string & cBDD_Orient::KeyAssoc()const 
{
   return mKeyAssoc;
}


cTplValGesInit< eConventionsOrientation > & cBDD_Orient::ConvOr()
{
   return mConvOr;
}

const cTplValGesInit< eConventionsOrientation > & cBDD_Orient::ConvOr()const 
{
   return mConvOr;
}

cElXMLTree * ToXMLTree(const cBDD_Orient & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"BDD_Orient",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Id"),anObj.Id())->ReTagThis("Id"));
   aRes->AddFils(::ToXMLTree(std::string("KeySet"),anObj.KeySet())->ReTagThis("KeySet"));
   aRes->AddFils(::ToXMLTree(std::string("KeyAssoc"),anObj.KeyAssoc())->ReTagThis("KeyAssoc"));
   if (anObj.ConvOr().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ConvOr"),anObj.ConvOr().Val())->ReTagThis("ConvOr"));
  return aRes;
}

void xml_init(cBDD_Orient & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.Id(),aTree->Get("Id",1)); //tototo 

   xml_init(anObj.KeySet(),aTree->Get("KeySet",1)); //tototo 

   xml_init(anObj.KeyAssoc(),aTree->Get("KeyAssoc",1)); //tototo 

   xml_init(anObj.ConvOr(),aTree->Get("ConvOr",1)); //tototo 
}


std::string & cCalcOffsetCentre::IdBase()
{
   return mIdBase;
}

const std::string & cCalcOffsetCentre::IdBase()const 
{
   return mIdBase;
}


std::string & cCalcOffsetCentre::KeyCalcBande()
{
   return mKeyCalcBande;
}

const std::string & cCalcOffsetCentre::KeyCalcBande()const 
{
   return mKeyCalcBande;
}


cTplValGesInit< bool > & cCalcOffsetCentre::OffsetUnknown()
{
   return mOffsetUnknown;
}

const cTplValGesInit< bool > & cCalcOffsetCentre::OffsetUnknown()const 
{
   return mOffsetUnknown;
}

cElXMLTree * ToXMLTree(const cCalcOffsetCentre & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"CalcOffsetCentre",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("IdBase"),anObj.IdBase())->ReTagThis("IdBase"));
   aRes->AddFils(::ToXMLTree(std::string("KeyCalcBande"),anObj.KeyCalcBande())->ReTagThis("KeyCalcBande"));
   if (anObj.OffsetUnknown().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("OffsetUnknown"),anObj.OffsetUnknown().Val())->ReTagThis("OffsetUnknown"));
  return aRes;
}

void xml_init(cCalcOffsetCentre & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.IdBase(),aTree->Get("IdBase",1)); //tototo 

   xml_init(anObj.KeyCalcBande(),aTree->Get("KeyCalcBande",1)); //tototo 

   xml_init(anObj.OffsetUnknown(),aTree->Get("OffsetUnknown",1),bool(false)); //tototo 
}


std::string & cBDD_Centre::Id()
{
   return mId;
}

const std::string & cBDD_Centre::Id()const 
{
   return mId;
}


std::string & cBDD_Centre::KeySet()
{
   return mKeySet;
}

const std::string & cBDD_Centre::KeySet()const 
{
   return mKeySet;
}


std::string & cBDD_Centre::KeyAssoc()
{
   return mKeyAssoc;
}

const std::string & cBDD_Centre::KeyAssoc()const 
{
   return mKeyAssoc;
}


cTplValGesInit< std::string > & cBDD_Centre::Tag()
{
   return mTag;
}

const cTplValGesInit< std::string > & cBDD_Centre::Tag()const 
{
   return mTag;
}


cTplValGesInit< std::string > & cBDD_Centre::ByFileTrajecto()
{
   return mByFileTrajecto;
}

const cTplValGesInit< std::string > & cBDD_Centre::ByFileTrajecto()const 
{
   return mByFileTrajecto;
}


cTplValGesInit< std::string > & cBDD_Centre::PatternFileTrajecto()
{
   return mPatternFileTrajecto;
}

const cTplValGesInit< std::string > & cBDD_Centre::PatternFileTrajecto()const 
{
   return mPatternFileTrajecto;
}


cTplValGesInit< std::string > & cBDD_Centre::PatternRefutFileTrajecto()
{
   return mPatternRefutFileTrajecto;
}

const cTplValGesInit< std::string > & cBDD_Centre::PatternRefutFileTrajecto()const 
{
   return mPatternRefutFileTrajecto;
}


std::string & cBDD_Centre::IdBase()
{
   return CalcOffsetCentre().Val().IdBase();
}

const std::string & cBDD_Centre::IdBase()const 
{
   return CalcOffsetCentre().Val().IdBase();
}


std::string & cBDD_Centre::KeyCalcBande()
{
   return CalcOffsetCentre().Val().KeyCalcBande();
}

const std::string & cBDD_Centre::KeyCalcBande()const 
{
   return CalcOffsetCentre().Val().KeyCalcBande();
}


cTplValGesInit< bool > & cBDD_Centre::OffsetUnknown()
{
   return CalcOffsetCentre().Val().OffsetUnknown();
}

const cTplValGesInit< bool > & cBDD_Centre::OffsetUnknown()const 
{
   return CalcOffsetCentre().Val().OffsetUnknown();
}


cTplValGesInit< cCalcOffsetCentre > & cBDD_Centre::CalcOffsetCentre()
{
   return mCalcOffsetCentre;
}

const cTplValGesInit< cCalcOffsetCentre > & cBDD_Centre::CalcOffsetCentre()const 
{
   return mCalcOffsetCentre;
}

cElXMLTree * ToXMLTree(const cBDD_Centre & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"BDD_Centre",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Id"),anObj.Id())->ReTagThis("Id"));
   aRes->AddFils(::ToXMLTree(std::string("KeySet"),anObj.KeySet())->ReTagThis("KeySet"));
   aRes->AddFils(::ToXMLTree(std::string("KeyAssoc"),anObj.KeyAssoc())->ReTagThis("KeyAssoc"));
   if (anObj.Tag().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Tag"),anObj.Tag().Val())->ReTagThis("Tag"));
   if (anObj.ByFileTrajecto().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ByFileTrajecto"),anObj.ByFileTrajecto().Val())->ReTagThis("ByFileTrajecto"));
   if (anObj.PatternFileTrajecto().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PatternFileTrajecto"),anObj.PatternFileTrajecto().Val())->ReTagThis("PatternFileTrajecto"));
   if (anObj.PatternRefutFileTrajecto().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PatternRefutFileTrajecto"),anObj.PatternRefutFileTrajecto().Val())->ReTagThis("PatternRefutFileTrajecto"));
   if (anObj.CalcOffsetCentre().IsInit())
      aRes->AddFils(ToXMLTree(anObj.CalcOffsetCentre().Val())->ReTagThis("CalcOffsetCentre"));
  return aRes;
}

void xml_init(cBDD_Centre & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.Id(),aTree->Get("Id",1)); //tototo 

   xml_init(anObj.KeySet(),aTree->Get("KeySet",1)); //tototo 

   xml_init(anObj.KeyAssoc(),aTree->Get("KeyAssoc",1)); //tototo 

   xml_init(anObj.Tag(),aTree->Get("Tag",1),std::string("Centre")); //tototo 

   xml_init(anObj.ByFileTrajecto(),aTree->Get("ByFileTrajecto",1)); //tototo 

   xml_init(anObj.PatternFileTrajecto(),aTree->Get("PatternFileTrajecto",1)); //tototo 

   xml_init(anObj.PatternRefutFileTrajecto(),aTree->Get("PatternRefutFileTrajecto",1)); //tototo 

   xml_init(anObj.CalcOffsetCentre(),aTree->Get("CalcOffsetCentre",1)); //tototo 
}


std::string & cFilterProj3D::Id()
{
   return mId;
}

const std::string & cFilterProj3D::Id()const 
{
   return mId;
}


std::string & cFilterProj3D::PatternSel()
{
   return mPatternSel;
}

const std::string & cFilterProj3D::PatternSel()const 
{
   return mPatternSel;
}


std::string & cFilterProj3D::AttrSup()
{
   return mAttrSup;
}

const std::string & cFilterProj3D::AttrSup()const 
{
   return mAttrSup;
}


std::string & cFilterProj3D::KeyCalculMasq()
{
   return mKeyCalculMasq;
}

const std::string & cFilterProj3D::KeyCalculMasq()const 
{
   return mKeyCalculMasq;
}

cElXMLTree * ToXMLTree(const cFilterProj3D & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"FilterProj3D",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Id"),anObj.Id())->ReTagThis("Id"));
   aRes->AddFils(::ToXMLTree(std::string("PatternSel"),anObj.PatternSel())->ReTagThis("PatternSel"));
   aRes->AddFils(::ToXMLTree(std::string("AttrSup"),anObj.AttrSup())->ReTagThis("AttrSup"));
   aRes->AddFils(::ToXMLTree(std::string("KeyCalculMasq"),anObj.KeyCalculMasq())->ReTagThis("KeyCalculMasq"));
  return aRes;
}

void xml_init(cFilterProj3D & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.Id(),aTree->Get("Id",1)); //tototo 

   xml_init(anObj.PatternSel(),aTree->Get("PatternSel",1)); //tototo 

   xml_init(anObj.AttrSup(),aTree->Get("AttrSup",1)); //tototo 

   xml_init(anObj.KeyCalculMasq(),aTree->Get("KeyCalculMasq",1)); //tototo 
}


cTplValGesInit< std::string > & cLayerTerrain::KeyAssocGeoref()
{
   return mKeyAssocGeoref;
}

const cTplValGesInit< std::string > & cLayerTerrain::KeyAssocGeoref()const 
{
   return mKeyAssocGeoref;
}


std::string & cLayerTerrain::KeyAssocOrImage()
{
   return mKeyAssocOrImage;
}

const std::string & cLayerTerrain::KeyAssocOrImage()const 
{
   return mKeyAssocOrImage;
}


std::string & cLayerTerrain::SysCoIm()
{
   return mSysCoIm;
}

const std::string & cLayerTerrain::SysCoIm()const 
{
   return mSysCoIm;
}


cTplValGesInit< std::string > & cLayerTerrain::TagOri()
{
   return mTagOri;
}

const cTplValGesInit< std::string > & cLayerTerrain::TagOri()const 
{
   return mTagOri;
}


cTplValGesInit< double > & cLayerTerrain::ZMoyen()
{
   return mZMoyen;
}

const cTplValGesInit< double > & cLayerTerrain::ZMoyen()const 
{
   return mZMoyen;
}

cElXMLTree * ToXMLTree(const cLayerTerrain & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"LayerTerrain",eXMLBranche);
   if (anObj.KeyAssocGeoref().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("KeyAssocGeoref"),anObj.KeyAssocGeoref().Val())->ReTagThis("KeyAssocGeoref"));
   aRes->AddFils(::ToXMLTree(std::string("KeyAssocOrImage"),anObj.KeyAssocOrImage())->ReTagThis("KeyAssocOrImage"));
   aRes->AddFils(::ToXMLTree(std::string("SysCoIm"),anObj.SysCoIm())->ReTagThis("SysCoIm"));
   if (anObj.TagOri().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("TagOri"),anObj.TagOri().Val())->ReTagThis("TagOri"));
   if (anObj.ZMoyen().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ZMoyen"),anObj.ZMoyen().Val())->ReTagThis("ZMoyen"));
  return aRes;
}

void xml_init(cLayerTerrain & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.KeyAssocGeoref(),aTree->Get("KeyAssocGeoref",1),std::string("KeyStd-Assoc-ChangExt@xml")); //tototo 

   xml_init(anObj.KeyAssocOrImage(),aTree->Get("KeyAssocOrImage",1)); //tototo 

   xml_init(anObj.SysCoIm(),aTree->Get("SysCoIm",1)); //tototo 

   xml_init(anObj.TagOri(),aTree->Get("TagOri",1),std::string("OrientationConique")); //tototo 

   xml_init(anObj.ZMoyen(),aTree->Get("ZMoyen",1)); //tototo 
}


std::string & cLayerImageToPose::Id()
{
   return mId;
}

const std::string & cLayerImageToPose::Id()const 
{
   return mId;
}


std::string & cLayerImageToPose::KeyCalculImage()
{
   return mKeyCalculImage;
}

const std::string & cLayerImageToPose::KeyCalculImage()const 
{
   return mKeyCalculImage;
}


int & cLayerImageToPose::FactRed()
{
   return mFactRed;
}

const int & cLayerImageToPose::FactRed()const 
{
   return mFactRed;
}


cTplValGesInit< std::string > & cLayerImageToPose::KeyNameRed()
{
   return mKeyNameRed;
}

const cTplValGesInit< std::string > & cLayerImageToPose::KeyNameRed()const 
{
   return mKeyNameRed;
}


cTplValGesInit< int > & cLayerImageToPose::FactCoherence()
{
   return mFactCoherence;
}

const cTplValGesInit< int > & cLayerImageToPose::FactCoherence()const 
{
   return mFactCoherence;
}


std::vector< int > & cLayerImageToPose::EtiqPrio()
{
   return mEtiqPrio;
}

const std::vector< int > & cLayerImageToPose::EtiqPrio()const 
{
   return mEtiqPrio;
}


cTplValGesInit< std::string > & cLayerImageToPose::KeyAssocGeoref()
{
   return LayerTerrain().Val().KeyAssocGeoref();
}

const cTplValGesInit< std::string > & cLayerImageToPose::KeyAssocGeoref()const 
{
   return LayerTerrain().Val().KeyAssocGeoref();
}


std::string & cLayerImageToPose::KeyAssocOrImage()
{
   return LayerTerrain().Val().KeyAssocOrImage();
}

const std::string & cLayerImageToPose::KeyAssocOrImage()const 
{
   return LayerTerrain().Val().KeyAssocOrImage();
}


std::string & cLayerImageToPose::SysCoIm()
{
   return LayerTerrain().Val().SysCoIm();
}

const std::string & cLayerImageToPose::SysCoIm()const 
{
   return LayerTerrain().Val().SysCoIm();
}


cTplValGesInit< std::string > & cLayerImageToPose::TagOri()
{
   return LayerTerrain().Val().TagOri();
}

const cTplValGesInit< std::string > & cLayerImageToPose::TagOri()const 
{
   return LayerTerrain().Val().TagOri();
}


cTplValGesInit< double > & cLayerImageToPose::ZMoyen()
{
   return LayerTerrain().Val().ZMoyen();
}

const cTplValGesInit< double > & cLayerImageToPose::ZMoyen()const 
{
   return LayerTerrain().Val().ZMoyen();
}


cTplValGesInit< cLayerTerrain > & cLayerImageToPose::LayerTerrain()
{
   return mLayerTerrain;
}

const cTplValGesInit< cLayerTerrain > & cLayerImageToPose::LayerTerrain()const 
{
   return mLayerTerrain;
}

cElXMLTree * ToXMLTree(const cLayerImageToPose & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"LayerImageToPose",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Id"),anObj.Id())->ReTagThis("Id"));
   aRes->AddFils(::ToXMLTree(std::string("KeyCalculImage"),anObj.KeyCalculImage())->ReTagThis("KeyCalculImage"));
   aRes->AddFils(::ToXMLTree(std::string("FactRed"),anObj.FactRed())->ReTagThis("FactRed"));
   if (anObj.KeyNameRed().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("KeyNameRed"),anObj.KeyNameRed().Val())->ReTagThis("KeyNameRed"));
   if (anObj.FactCoherence().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("FactCoherence"),anObj.FactCoherence().Val())->ReTagThis("FactCoherence"));
  for
  (       std::vector< int >::const_iterator it=anObj.EtiqPrio().begin();
      it !=anObj.EtiqPrio().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("EtiqPrio"),(*it))->ReTagThis("EtiqPrio"));
   if (anObj.LayerTerrain().IsInit())
      aRes->AddFils(ToXMLTree(anObj.LayerTerrain().Val())->ReTagThis("LayerTerrain"));
  return aRes;
}

void xml_init(cLayerImageToPose & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.Id(),aTree->Get("Id",1)); //tototo 

   xml_init(anObj.KeyCalculImage(),aTree->Get("KeyCalculImage",1)); //tototo 

   xml_init(anObj.FactRed(),aTree->Get("FactRed",1)); //tototo 

   xml_init(anObj.KeyNameRed(),aTree->Get("KeyNameRed",1),std::string("KeyStd-Assoc-AddPref@Layer-Reduc-")); //tototo 

   xml_init(anObj.FactCoherence(),aTree->Get("FactCoherence",1),int(-1)); //tototo 

   xml_init(anObj.EtiqPrio(),aTree->GetAll("EtiqPrio",false,1));

   xml_init(anObj.LayerTerrain(),aTree->Get("LayerTerrain",1)); //tototo 
}


std::list< cBDD_PtsLiaisons > & cSectionBDD_Observation::BDD_PtsLiaisons()
{
   return mBDD_PtsLiaisons;
}

const std::list< cBDD_PtsLiaisons > & cSectionBDD_Observation::BDD_PtsLiaisons()const 
{
   return mBDD_PtsLiaisons;
}


std::list< cBDD_PtsAppuis > & cSectionBDD_Observation::BDD_PtsAppuis()
{
   return mBDD_PtsAppuis;
}

const std::list< cBDD_PtsAppuis > & cSectionBDD_Observation::BDD_PtsAppuis()const 
{
   return mBDD_PtsAppuis;
}


std::list< cBDD_ObsAppuisFlottant > & cSectionBDD_Observation::BDD_ObsAppuisFlottant()
{
   return mBDD_ObsAppuisFlottant;
}

const std::list< cBDD_ObsAppuisFlottant > & cSectionBDD_Observation::BDD_ObsAppuisFlottant()const 
{
   return mBDD_ObsAppuisFlottant;
}


std::list< cBDD_Orient > & cSectionBDD_Observation::BDD_Orient()
{
   return mBDD_Orient;
}

const std::list< cBDD_Orient > & cSectionBDD_Observation::BDD_Orient()const 
{
   return mBDD_Orient;
}


std::list< cBDD_Centre > & cSectionBDD_Observation::BDD_Centre()
{
   return mBDD_Centre;
}

const std::list< cBDD_Centre > & cSectionBDD_Observation::BDD_Centre()const 
{
   return mBDD_Centre;
}


std::list< cFilterProj3D > & cSectionBDD_Observation::FilterProj3D()
{
   return mFilterProj3D;
}

const std::list< cFilterProj3D > & cSectionBDD_Observation::FilterProj3D()const 
{
   return mFilterProj3D;
}


std::list< cLayerImageToPose > & cSectionBDD_Observation::LayerImageToPose()
{
   return mLayerImageToPose;
}

const std::list< cLayerImageToPose > & cSectionBDD_Observation::LayerImageToPose()const 
{
   return mLayerImageToPose;
}


cTplValGesInit< double > & cSectionBDD_Observation::LimInfBSurHPMoy()
{
   return mLimInfBSurHPMoy;
}

const cTplValGesInit< double > & cSectionBDD_Observation::LimInfBSurHPMoy()const 
{
   return mLimInfBSurHPMoy;
}


cTplValGesInit< double > & cSectionBDD_Observation::LimSupBSurHPMoy()
{
   return mLimSupBSurHPMoy;
}

const cTplValGesInit< double > & cSectionBDD_Observation::LimSupBSurHPMoy()const 
{
   return mLimSupBSurHPMoy;
}

cElXMLTree * ToXMLTree(const cSectionBDD_Observation & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"SectionBDD_Observation",eXMLBranche);
  for
  (       std::list< cBDD_PtsLiaisons >::const_iterator it=anObj.BDD_PtsLiaisons().begin();
      it !=anObj.BDD_PtsLiaisons().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("BDD_PtsLiaisons"));
  for
  (       std::list< cBDD_PtsAppuis >::const_iterator it=anObj.BDD_PtsAppuis().begin();
      it !=anObj.BDD_PtsAppuis().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("BDD_PtsAppuis"));
  for
  (       std::list< cBDD_ObsAppuisFlottant >::const_iterator it=anObj.BDD_ObsAppuisFlottant().begin();
      it !=anObj.BDD_ObsAppuisFlottant().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("BDD_ObsAppuisFlottant"));
  for
  (       std::list< cBDD_Orient >::const_iterator it=anObj.BDD_Orient().begin();
      it !=anObj.BDD_Orient().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("BDD_Orient"));
  for
  (       std::list< cBDD_Centre >::const_iterator it=anObj.BDD_Centre().begin();
      it !=anObj.BDD_Centre().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("BDD_Centre"));
  for
  (       std::list< cFilterProj3D >::const_iterator it=anObj.FilterProj3D().begin();
      it !=anObj.FilterProj3D().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("FilterProj3D"));
  for
  (       std::list< cLayerImageToPose >::const_iterator it=anObj.LayerImageToPose().begin();
      it !=anObj.LayerImageToPose().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("LayerImageToPose"));
   if (anObj.LimInfBSurHPMoy().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("LimInfBSurHPMoy"),anObj.LimInfBSurHPMoy().Val())->ReTagThis("LimInfBSurHPMoy"));
   if (anObj.LimSupBSurHPMoy().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("LimSupBSurHPMoy"),anObj.LimSupBSurHPMoy().Val())->ReTagThis("LimSupBSurHPMoy"));
  return aRes;
}

void xml_init(cSectionBDD_Observation & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.BDD_PtsLiaisons(),aTree->GetAll("BDD_PtsLiaisons",false,1));

   xml_init(anObj.BDD_PtsAppuis(),aTree->GetAll("BDD_PtsAppuis",false,1));

   xml_init(anObj.BDD_ObsAppuisFlottant(),aTree->GetAll("BDD_ObsAppuisFlottant",false,1));

   xml_init(anObj.BDD_Orient(),aTree->GetAll("BDD_Orient",false,1));

   xml_init(anObj.BDD_Centre(),aTree->GetAll("BDD_Centre",false,1));

   xml_init(anObj.FilterProj3D(),aTree->GetAll("FilterProj3D",false,1));

   xml_init(anObj.LayerImageToPose(),aTree->GetAll("LayerImageToPose",false,1));

   xml_init(anObj.LimInfBSurHPMoy(),aTree->Get("LimInfBSurHPMoy",1),double(1e-2)); //tototo 

   xml_init(anObj.LimSupBSurHPMoy(),aTree->Get("LimSupBSurHPMoy",1),double(2e-1)); //tototo 
}


eTypeCalibAutom & cCalibAutomNoDist::TypeDist()
{
   return mTypeDist;
}

const eTypeCalibAutom & cCalibAutomNoDist::TypeDist()const 
{
   return mTypeDist;
}


cTplValGesInit< std::string > & cCalibAutomNoDist::NameIm()
{
   return mNameIm;
}

const cTplValGesInit< std::string > & cCalibAutomNoDist::NameIm()const 
{
   return mNameIm;
}


cTplValGesInit< std::string > & cCalibAutomNoDist::KeyFileSauv()
{
   return mKeyFileSauv;
}

const cTplValGesInit< std::string > & cCalibAutomNoDist::KeyFileSauv()const 
{
   return mKeyFileSauv;
}


cTplValGesInit< Pt2dr > & cCalibAutomNoDist::PositionRelPP()
{
   return mPositionRelPP;
}

const cTplValGesInit< Pt2dr > & cCalibAutomNoDist::PositionRelPP()const 
{
   return mPositionRelPP;
}

cElXMLTree * ToXMLTree(const cCalibAutomNoDist & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"CalibAutomNoDist",eXMLBranche);
   aRes->AddFils(ToXMLTree(std::string("TypeDist"),anObj.TypeDist())->ReTagThis("TypeDist"));
   if (anObj.NameIm().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NameIm"),anObj.NameIm().Val())->ReTagThis("NameIm"));
   if (anObj.KeyFileSauv().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("KeyFileSauv"),anObj.KeyFileSauv().Val())->ReTagThis("KeyFileSauv"));
   if (anObj.PositionRelPP().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PositionRelPP"),anObj.PositionRelPP().Val())->ReTagThis("PositionRelPP"));
  return aRes;
}

void xml_init(cCalibAutomNoDist & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.TypeDist(),aTree->Get("TypeDist",1)); //tototo 

   xml_init(anObj.NameIm(),aTree->Get("NameIm",1)); //tototo 

   xml_init(anObj.KeyFileSauv(),aTree->Get("KeyFileSauv",1)); //tototo 

   xml_init(anObj.PositionRelPP(),aTree->Get("PositionRelPP",1),Pt2dr(Pt2dr(0.5,0.5))); //tototo 
}


cTplValGesInit< cCalibrationInternConique > & cCalValueInit::CalFromValues()
{
   return mCalFromValues;
}

const cTplValGesInit< cCalibrationInternConique > & cCalValueInit::CalFromValues()const 
{
   return mCalFromValues;
}


cTplValGesInit< cSpecExtractFromFile > & cCalValueInit::CalFromFileExtern()
{
   return mCalFromFileExtern;
}

const cTplValGesInit< cSpecExtractFromFile > & cCalValueInit::CalFromFileExtern()const 
{
   return mCalFromFileExtern;
}


cTplValGesInit< bool > & cCalValueInit::CalibFromMmBD()
{
   return mCalibFromMmBD;
}

const cTplValGesInit< bool > & cCalValueInit::CalibFromMmBD()const 
{
   return mCalibFromMmBD;
}


eTypeCalibAutom & cCalValueInit::TypeDist()
{
   return CalibAutomNoDist().Val().TypeDist();
}

const eTypeCalibAutom & cCalValueInit::TypeDist()const 
{
   return CalibAutomNoDist().Val().TypeDist();
}


cTplValGesInit< std::string > & cCalValueInit::NameIm()
{
   return CalibAutomNoDist().Val().NameIm();
}

const cTplValGesInit< std::string > & cCalValueInit::NameIm()const 
{
   return CalibAutomNoDist().Val().NameIm();
}


cTplValGesInit< std::string > & cCalValueInit::KeyFileSauv()
{
   return CalibAutomNoDist().Val().KeyFileSauv();
}

const cTplValGesInit< std::string > & cCalValueInit::KeyFileSauv()const 
{
   return CalibAutomNoDist().Val().KeyFileSauv();
}


cTplValGesInit< Pt2dr > & cCalValueInit::PositionRelPP()
{
   return CalibAutomNoDist().Val().PositionRelPP();
}

const cTplValGesInit< Pt2dr > & cCalValueInit::PositionRelPP()const 
{
   return CalibAutomNoDist().Val().PositionRelPP();
}


cTplValGesInit< cCalibAutomNoDist > & cCalValueInit::CalibAutomNoDist()
{
   return mCalibAutomNoDist;
}

const cTplValGesInit< cCalibAutomNoDist > & cCalValueInit::CalibAutomNoDist()const 
{
   return mCalibAutomNoDist;
}

cElXMLTree * ToXMLTree(const cCalValueInit & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"CalValueInit",eXMLBranche);
   if (anObj.CalFromValues().IsInit())
      aRes->AddFils(ToXMLTree(anObj.CalFromValues().Val())->ReTagThis("CalFromValues"));
   if (anObj.CalFromFileExtern().IsInit())
      aRes->AddFils(ToXMLTree(anObj.CalFromFileExtern().Val())->ReTagThis("CalFromFileExtern"));
   if (anObj.CalibFromMmBD().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("CalibFromMmBD"),anObj.CalibFromMmBD().Val())->ReTagThis("CalibFromMmBD"));
   if (anObj.CalibAutomNoDist().IsInit())
      aRes->AddFils(ToXMLTree(anObj.CalibAutomNoDist().Val())->ReTagThis("CalibAutomNoDist"));
  return aRes;
}

void xml_init(cCalValueInit & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.CalFromValues(),aTree->Get("CalFromValues",1)); //tototo 

   xml_init(anObj.CalFromFileExtern(),aTree->Get("CalFromFileExtern",1)); //tototo 

   xml_init(anObj.CalibFromMmBD(),aTree->Get("CalibFromMmBD",1),bool(true)); //tototo 

   xml_init(anObj.CalibAutomNoDist(),aTree->Get("CalibAutomNoDist",1)); //tototo 
}


std::vector< double > & cAddParamAFocal::Coeffs()
{
   return mCoeffs;
}

const std::vector< double > & cAddParamAFocal::Coeffs()const 
{
   return mCoeffs;
}

cElXMLTree * ToXMLTree(const cAddParamAFocal & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"AddParamAFocal",eXMLBranche);
  for
  (       std::vector< double >::const_iterator it=anObj.Coeffs().begin();
      it !=anObj.Coeffs().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("Coeffs"),(*it))->ReTagThis("Coeffs"));
  return aRes;
}

void xml_init(cAddParamAFocal & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.Coeffs(),aTree->GetAll("Coeffs",false,1));
}


std::string & cCalibPerPose::KeyPose2Cal()
{
   return mKeyPose2Cal;
}

const std::string & cCalibPerPose::KeyPose2Cal()const 
{
   return mKeyPose2Cal;
}


cTplValGesInit< std::string > & cCalibPerPose::KeyInitFromPose()
{
   return mKeyInitFromPose;
}

const cTplValGesInit< std::string > & cCalibPerPose::KeyInitFromPose()const 
{
   return mKeyInitFromPose;
}

cElXMLTree * ToXMLTree(const cCalibPerPose & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"CalibPerPose",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("KeyPose2Cal"),anObj.KeyPose2Cal())->ReTagThis("KeyPose2Cal"));
   if (anObj.KeyInitFromPose().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("KeyInitFromPose"),anObj.KeyInitFromPose().Val())->ReTagThis("KeyInitFromPose"));
  return aRes;
}

void xml_init(cCalibPerPose & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.KeyPose2Cal(),aTree->Get("KeyPose2Cal",1)); //tototo 

   xml_init(anObj.KeyInitFromPose(),aTree->Get("KeyInitFromPose",1)); //tototo 
}


std::string & cCalibrationCameraInc::Name()
{
   return mName;
}

const std::string & cCalibrationCameraInc::Name()const 
{
   return mName;
}


cTplValGesInit< eConventionsOrientation > & cCalibrationCameraInc::ConvCal()
{
   return mConvCal;
}

const cTplValGesInit< eConventionsOrientation > & cCalibrationCameraInc::ConvCal()const 
{
   return mConvCal;
}


cTplValGesInit< std::string > & cCalibrationCameraInc::Directory()
{
   return mDirectory;
}

const cTplValGesInit< std::string > & cCalibrationCameraInc::Directory()const 
{
   return mDirectory;
}


cTplValGesInit< cCalibrationInternConique > & cCalibrationCameraInc::CalFromValues()
{
   return CalValueInit().CalFromValues();
}

const cTplValGesInit< cCalibrationInternConique > & cCalibrationCameraInc::CalFromValues()const 
{
   return CalValueInit().CalFromValues();
}


cTplValGesInit< cSpecExtractFromFile > & cCalibrationCameraInc::CalFromFileExtern()
{
   return CalValueInit().CalFromFileExtern();
}

const cTplValGesInit< cSpecExtractFromFile > & cCalibrationCameraInc::CalFromFileExtern()const 
{
   return CalValueInit().CalFromFileExtern();
}


cTplValGesInit< bool > & cCalibrationCameraInc::CalibFromMmBD()
{
   return CalValueInit().CalibFromMmBD();
}

const cTplValGesInit< bool > & cCalibrationCameraInc::CalibFromMmBD()const 
{
   return CalValueInit().CalibFromMmBD();
}


eTypeCalibAutom & cCalibrationCameraInc::TypeDist()
{
   return CalValueInit().CalibAutomNoDist().Val().TypeDist();
}

const eTypeCalibAutom & cCalibrationCameraInc::TypeDist()const 
{
   return CalValueInit().CalibAutomNoDist().Val().TypeDist();
}


cTplValGesInit< std::string > & cCalibrationCameraInc::NameIm()
{
   return CalValueInit().CalibAutomNoDist().Val().NameIm();
}

const cTplValGesInit< std::string > & cCalibrationCameraInc::NameIm()const 
{
   return CalValueInit().CalibAutomNoDist().Val().NameIm();
}


cTplValGesInit< std::string > & cCalibrationCameraInc::KeyFileSauv()
{
   return CalValueInit().CalibAutomNoDist().Val().KeyFileSauv();
}

const cTplValGesInit< std::string > & cCalibrationCameraInc::KeyFileSauv()const 
{
   return CalValueInit().CalibAutomNoDist().Val().KeyFileSauv();
}


cTplValGesInit< Pt2dr > & cCalibrationCameraInc::PositionRelPP()
{
   return CalValueInit().CalibAutomNoDist().Val().PositionRelPP();
}

const cTplValGesInit< Pt2dr > & cCalibrationCameraInc::PositionRelPP()const 
{
   return CalValueInit().CalibAutomNoDist().Val().PositionRelPP();
}


cTplValGesInit< cCalibAutomNoDist > & cCalibrationCameraInc::CalibAutomNoDist()
{
   return CalValueInit().CalibAutomNoDist();
}

const cTplValGesInit< cCalibAutomNoDist > & cCalibrationCameraInc::CalibAutomNoDist()const 
{
   return CalValueInit().CalibAutomNoDist();
}


cCalValueInit & cCalibrationCameraInc::CalValueInit()
{
   return mCalValueInit;
}

const cCalValueInit & cCalibrationCameraInc::CalValueInit()const 
{
   return mCalValueInit;
}


cTplValGesInit< cCalibDistortion > & cCalibrationCameraInc::DistortionAddInc()
{
   return mDistortionAddInc;
}

const cTplValGesInit< cCalibDistortion > & cCalibrationCameraInc::DistortionAddInc()const 
{
   return mDistortionAddInc;
}


std::vector< double > & cCalibrationCameraInc::Coeffs()
{
   return AddParamAFocal().Val().Coeffs();
}

const std::vector< double > & cCalibrationCameraInc::Coeffs()const 
{
   return AddParamAFocal().Val().Coeffs();
}


cTplValGesInit< cAddParamAFocal > & cCalibrationCameraInc::AddParamAFocal()
{
   return mAddParamAFocal;
}

const cTplValGesInit< cAddParamAFocal > & cCalibrationCameraInc::AddParamAFocal()const 
{
   return mAddParamAFocal;
}


cTplValGesInit< double > & cCalibrationCameraInc::RayMaxUtile()
{
   return mRayMaxUtile;
}

const cTplValGesInit< double > & cCalibrationCameraInc::RayMaxUtile()const 
{
   return mRayMaxUtile;
}


cTplValGesInit< bool > & cCalibrationCameraInc::RayIsRelatifDiag()
{
   return mRayIsRelatifDiag;
}

const cTplValGesInit< bool > & cCalibrationCameraInc::RayIsRelatifDiag()const 
{
   return mRayIsRelatifDiag;
}


cTplValGesInit< bool > & cCalibrationCameraInc::RayApplyOnlyFE()
{
   return mRayApplyOnlyFE;
}

const cTplValGesInit< bool > & cCalibrationCameraInc::RayApplyOnlyFE()const 
{
   return mRayApplyOnlyFE;
}


cTplValGesInit< double > & cCalibrationCameraInc::PropDiagUtile()
{
   return mPropDiagUtile;
}

const cTplValGesInit< double > & cCalibrationCameraInc::PropDiagUtile()const 
{
   return mPropDiagUtile;
}


std::string & cCalibrationCameraInc::KeyPose2Cal()
{
   return CalibPerPose().Val().KeyPose2Cal();
}

const std::string & cCalibrationCameraInc::KeyPose2Cal()const 
{
   return CalibPerPose().Val().KeyPose2Cal();
}


cTplValGesInit< std::string > & cCalibrationCameraInc::KeyInitFromPose()
{
   return CalibPerPose().Val().KeyInitFromPose();
}

const cTplValGesInit< std::string > & cCalibrationCameraInc::KeyInitFromPose()const 
{
   return CalibPerPose().Val().KeyInitFromPose();
}


cTplValGesInit< cCalibPerPose > & cCalibrationCameraInc::CalibPerPose()
{
   return mCalibPerPose;
}

const cTplValGesInit< cCalibPerPose > & cCalibrationCameraInc::CalibPerPose()const 
{
   return mCalibPerPose;
}

cElXMLTree * ToXMLTree(const cCalibrationCameraInc & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"CalibrationCameraInc",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Name"),anObj.Name())->ReTagThis("Name"));
   if (anObj.ConvCal().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ConvCal"),anObj.ConvCal().Val())->ReTagThis("ConvCal"));
   if (anObj.Directory().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Directory"),anObj.Directory().Val())->ReTagThis("Directory"));
   aRes->AddFils(ToXMLTree(anObj.CalValueInit())->ReTagThis("CalValueInit"));
   if (anObj.DistortionAddInc().IsInit())
      aRes->AddFils(ToXMLTree(anObj.DistortionAddInc().Val())->ReTagThis("DistortionAddInc"));
   if (anObj.AddParamAFocal().IsInit())
      aRes->AddFils(ToXMLTree(anObj.AddParamAFocal().Val())->ReTagThis("AddParamAFocal"));
   if (anObj.RayMaxUtile().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("RayMaxUtile"),anObj.RayMaxUtile().Val())->ReTagThis("RayMaxUtile"));
   if (anObj.RayIsRelatifDiag().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("RayIsRelatifDiag"),anObj.RayIsRelatifDiag().Val())->ReTagThis("RayIsRelatifDiag"));
   if (anObj.RayApplyOnlyFE().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("RayApplyOnlyFE"),anObj.RayApplyOnlyFE().Val())->ReTagThis("RayApplyOnlyFE"));
   if (anObj.PropDiagUtile().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PropDiagUtile"),anObj.PropDiagUtile().Val())->ReTagThis("PropDiagUtile"));
   if (anObj.CalibPerPose().IsInit())
      aRes->AddFils(ToXMLTree(anObj.CalibPerPose().Val())->ReTagThis("CalibPerPose"));
  return aRes;
}

void xml_init(cCalibrationCameraInc & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.Name(),aTree->Get("Name",1)); //tototo 

   xml_init(anObj.ConvCal(),aTree->Get("ConvCal",1),eConventionsOrientation(eConvApero_DistM2C)); //tototo 

   xml_init(anObj.Directory(),aTree->Get("Directory",1),std::string("")); //tototo 

   xml_init(anObj.CalValueInit(),aTree->Get("CalValueInit",1)); //tototo 

   xml_init(anObj.DistortionAddInc(),aTree->Get("DistortionAddInc",1)); //tototo 

   xml_init(anObj.AddParamAFocal(),aTree->Get("AddParamAFocal",1)); //tototo 

   xml_init(anObj.RayMaxUtile(),aTree->Get("RayMaxUtile",1),double(1e20)); //tototo 

   xml_init(anObj.RayIsRelatifDiag(),aTree->Get("RayIsRelatifDiag",1),bool(false)); //tototo 

   xml_init(anObj.RayApplyOnlyFE(),aTree->Get("RayApplyOnlyFE",1),bool(false)); //tototo 

   xml_init(anObj.PropDiagUtile(),aTree->Get("PropDiagUtile",1),double(1.0)); //tototo 

   xml_init(anObj.CalibPerPose(),aTree->Get("CalibPerPose",1)); //tototo 
}


cTplValGesInit< bool > & cMEP_SPEC_MST::Show()
{
   return mShow;
}

const cTplValGesInit< bool > & cMEP_SPEC_MST::Show()const 
{
   return mShow;
}


cTplValGesInit< int > & cMEP_SPEC_MST::MinNbPtsInit()
{
   return mMinNbPtsInit;
}

const cTplValGesInit< int > & cMEP_SPEC_MST::MinNbPtsInit()const 
{
   return mMinNbPtsInit;
}


cTplValGesInit< double > & cMEP_SPEC_MST::ExpDist()
{
   return mExpDist;
}

const cTplValGesInit< double > & cMEP_SPEC_MST::ExpDist()const 
{
   return mExpDist;
}


cTplValGesInit< double > & cMEP_SPEC_MST::ExpNb()
{
   return mExpNb;
}

const cTplValGesInit< double > & cMEP_SPEC_MST::ExpNb()const 
{
   return mExpNb;
}


cTplValGesInit< bool > & cMEP_SPEC_MST::MontageOnInit()
{
   return mMontageOnInit;
}

const cTplValGesInit< bool > & cMEP_SPEC_MST::MontageOnInit()const 
{
   return mMontageOnInit;
}


cTplValGesInit< int > & cMEP_SPEC_MST::NbInitMinBeforeUnconnect()
{
   return mNbInitMinBeforeUnconnect;
}

const cTplValGesInit< int > & cMEP_SPEC_MST::NbInitMinBeforeUnconnect()const 
{
   return mNbInitMinBeforeUnconnect;
}

cElXMLTree * ToXMLTree(const cMEP_SPEC_MST & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"MEP_SPEC_MST",eXMLBranche);
   if (anObj.Show().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Show"),anObj.Show().Val())->ReTagThis("Show"));
   if (anObj.MinNbPtsInit().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("MinNbPtsInit"),anObj.MinNbPtsInit().Val())->ReTagThis("MinNbPtsInit"));
   if (anObj.ExpDist().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ExpDist"),anObj.ExpDist().Val())->ReTagThis("ExpDist"));
   if (anObj.ExpNb().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ExpNb"),anObj.ExpNb().Val())->ReTagThis("ExpNb"));
   if (anObj.MontageOnInit().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("MontageOnInit"),anObj.MontageOnInit().Val())->ReTagThis("MontageOnInit"));
   if (anObj.NbInitMinBeforeUnconnect().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NbInitMinBeforeUnconnect"),anObj.NbInitMinBeforeUnconnect().Val())->ReTagThis("NbInitMinBeforeUnconnect"));
  return aRes;
}

void xml_init(cMEP_SPEC_MST & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.Show(),aTree->Get("Show",1),bool(false)); //tototo 

   xml_init(anObj.MinNbPtsInit(),aTree->Get("MinNbPtsInit",1),int(8)); //tototo 

   xml_init(anObj.ExpDist(),aTree->Get("ExpDist",1),double(2.0)); //tototo 

   xml_init(anObj.ExpNb(),aTree->Get("ExpNb",1),double(1.0)); //tototo 

   xml_init(anObj.MontageOnInit(),aTree->Get("MontageOnInit",1),bool(false)); //tototo 

   xml_init(anObj.NbInitMinBeforeUnconnect(),aTree->Get("NbInitMinBeforeUnconnect",1),int(10000000)); //tototo 
}


eTypeContraintePoseCamera & cApplyOAI::Cstr()
{
   return mCstr;
}

const eTypeContraintePoseCamera & cApplyOAI::Cstr()const 
{
   return mCstr;
}


std::string & cApplyOAI::PatternApply()
{
   return mPatternApply;
}

const std::string & cApplyOAI::PatternApply()const 
{
   return mPatternApply;
}

cElXMLTree * ToXMLTree(const cApplyOAI & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ApplyOAI",eXMLBranche);
   aRes->AddFils(ToXMLTree(std::string("Cstr"),anObj.Cstr())->ReTagThis("Cstr"));
   aRes->AddFils(::ToXMLTree(std::string("PatternApply"),anObj.PatternApply())->ReTagThis("PatternApply"));
  return aRes;
}

void xml_init(cApplyOAI & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.Cstr(),aTree->Get("Cstr",1)); //tototo 

   xml_init(anObj.PatternApply(),aTree->Get("PatternApply",1)); //tototo 
}


cOptimizationPowel & cOptimizeAfterInit::ParamOptim()
{
   return mParamOptim;
}

const cOptimizationPowel & cOptimizeAfterInit::ParamOptim()const 
{
   return mParamOptim;
}


std::list< cApplyOAI > & cOptimizeAfterInit::ApplyOAI()
{
   return mApplyOAI;
}

const std::list< cApplyOAI > & cOptimizeAfterInit::ApplyOAI()const 
{
   return mApplyOAI;
}

cElXMLTree * ToXMLTree(const cOptimizeAfterInit & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"OptimizeAfterInit",eXMLBranche);
   aRes->AddFils(ToXMLTree(anObj.ParamOptim())->ReTagThis("ParamOptim"));
  for
  (       std::list< cApplyOAI >::const_iterator it=anObj.ApplyOAI().begin();
      it !=anObj.ApplyOAI().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("ApplyOAI"));
  return aRes;
}

void xml_init(cOptimizeAfterInit & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.ParamOptim(),aTree->Get("ParamOptim",1)); //tototo 

   xml_init(anObj.ApplyOAI(),aTree->GetAll("ApplyOAI",false,1));
}


std::string & cPosFromBDAppuis::Id()
{
   return mId;
}

const std::string & cPosFromBDAppuis::Id()const 
{
   return mId;
}


int & cPosFromBDAppuis::NbTestRansac()
{
   return mNbTestRansac;
}

const int & cPosFromBDAppuis::NbTestRansac()const 
{
   return mNbTestRansac;
}


cTplValGesInit< Pt3dr > & cPosFromBDAppuis::DirApprox()
{
   return mDirApprox;
}

const cTplValGesInit< Pt3dr > & cPosFromBDAppuis::DirApprox()const 
{
   return mDirApprox;
}

cElXMLTree * ToXMLTree(const cPosFromBDAppuis & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"PosFromBDAppuis",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Id"),anObj.Id())->ReTagThis("Id"));
   aRes->AddFils(::ToXMLTree(std::string("NbTestRansac"),anObj.NbTestRansac())->ReTagThis("NbTestRansac"));
   if (anObj.DirApprox().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DirApprox"),anObj.DirApprox().Val())->ReTagThis("DirApprox"));
  return aRes;
}

void xml_init(cPosFromBDAppuis & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.Id(),aTree->Get("Id",1)); //tototo 

   xml_init(anObj.NbTestRansac(),aTree->Get("NbTestRansac",1)); //tototo 

   xml_init(anObj.DirApprox(),aTree->Get("DirApprox",1)); //tototo 
}


cTplValGesInit< std::string > & cLiaisonsInit::OnZonePlane()
{
   return mOnZonePlane;
}

const cTplValGesInit< std::string > & cLiaisonsInit::OnZonePlane()const 
{
   return mOnZonePlane;
}


cTplValGesInit< bool > & cLiaisonsInit::TestSolPlane()
{
   return mTestSolPlane;
}

const cTplValGesInit< bool > & cLiaisonsInit::TestSolPlane()const 
{
   return mTestSolPlane;
}


cTplValGesInit< int > & cLiaisonsInit::NbRansacSolAppui()
{
   return mNbRansacSolAppui;
}

const cTplValGesInit< int > & cLiaisonsInit::NbRansacSolAppui()const 
{
   return mNbRansacSolAppui;
}


cTplValGesInit< bool > & cLiaisonsInit::InitOrientPure()
{
   return mInitOrientPure;
}

const cTplValGesInit< bool > & cLiaisonsInit::InitOrientPure()const 
{
   return mInitOrientPure;
}


cTplValGesInit< int > & cLiaisonsInit::NbPtsRansacOrPure()
{
   return mNbPtsRansacOrPure;
}

const cTplValGesInit< int > & cLiaisonsInit::NbPtsRansacOrPure()const 
{
   return mNbPtsRansacOrPure;
}


cTplValGesInit< int > & cLiaisonsInit::NbTestRansacOrPure()
{
   return mNbTestRansacOrPure;
}

const cTplValGesInit< int > & cLiaisonsInit::NbTestRansacOrPure()const 
{
   return mNbTestRansacOrPure;
}


cTplValGesInit< int > & cLiaisonsInit::NbMinPtsRanAp()
{
   return mNbMinPtsRanAp;
}

const cTplValGesInit< int > & cLiaisonsInit::NbMinPtsRanAp()const 
{
   return mNbMinPtsRanAp;
}


cTplValGesInit< int > & cLiaisonsInit::NbMaxPtsRanAp()
{
   return mNbMaxPtsRanAp;
}

const cTplValGesInit< int > & cLiaisonsInit::NbMaxPtsRanAp()const 
{
   return mNbMaxPtsRanAp;
}


cTplValGesInit< double > & cLiaisonsInit::PropMinPtsMult()
{
   return mPropMinPtsMult;
}

const cTplValGesInit< double > & cLiaisonsInit::PropMinPtsMult()const 
{
   return mPropMinPtsMult;
}


std::string & cLiaisonsInit::NameCam()
{
   return mNameCam;
}

const std::string & cLiaisonsInit::NameCam()const 
{
   return mNameCam;
}


cTplValGesInit< bool > & cLiaisonsInit::NameCamIsKeyCalc()
{
   return mNameCamIsKeyCalc;
}

const cTplValGesInit< bool > & cLiaisonsInit::NameCamIsKeyCalc()const 
{
   return mNameCamIsKeyCalc;
}


cTplValGesInit< bool > & cLiaisonsInit::KeyCalcIsIDir()
{
   return mKeyCalcIsIDir;
}

const cTplValGesInit< bool > & cLiaisonsInit::KeyCalcIsIDir()const 
{
   return mKeyCalcIsIDir;
}


std::string & cLiaisonsInit::IdBD()
{
   return mIdBD;
}

const std::string & cLiaisonsInit::IdBD()const 
{
   return mIdBD;
}


cTplValGesInit< double > & cLiaisonsInit::ProfSceneCouple()
{
   return mProfSceneCouple;
}

const cTplValGesInit< double > & cLiaisonsInit::ProfSceneCouple()const 
{
   return mProfSceneCouple;
}


cTplValGesInit< bool > & cLiaisonsInit::L2EstimPlan()
{
   return mL2EstimPlan;
}

const cTplValGesInit< bool > & cLiaisonsInit::L2EstimPlan()const 
{
   return mL2EstimPlan;
}


cTplValGesInit< double > & cLiaisonsInit::LongueurBase()
{
   return mLongueurBase;
}

const cTplValGesInit< double > & cLiaisonsInit::LongueurBase()const 
{
   return mLongueurBase;
}

cElXMLTree * ToXMLTree(const cLiaisonsInit & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"LiaisonsInit",eXMLBranche);
   if (anObj.OnZonePlane().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("OnZonePlane"),anObj.OnZonePlane().Val())->ReTagThis("OnZonePlane"));
   if (anObj.TestSolPlane().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("TestSolPlane"),anObj.TestSolPlane().Val())->ReTagThis("TestSolPlane"));
   if (anObj.NbRansacSolAppui().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NbRansacSolAppui"),anObj.NbRansacSolAppui().Val())->ReTagThis("NbRansacSolAppui"));
   if (anObj.InitOrientPure().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("InitOrientPure"),anObj.InitOrientPure().Val())->ReTagThis("InitOrientPure"));
   if (anObj.NbPtsRansacOrPure().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NbPtsRansacOrPure"),anObj.NbPtsRansacOrPure().Val())->ReTagThis("NbPtsRansacOrPure"));
   if (anObj.NbTestRansacOrPure().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NbTestRansacOrPure"),anObj.NbTestRansacOrPure().Val())->ReTagThis("NbTestRansacOrPure"));
   if (anObj.NbMinPtsRanAp().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NbMinPtsRanAp"),anObj.NbMinPtsRanAp().Val())->ReTagThis("NbMinPtsRanAp"));
   if (anObj.NbMaxPtsRanAp().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NbMaxPtsRanAp"),anObj.NbMaxPtsRanAp().Val())->ReTagThis("NbMaxPtsRanAp"));
   if (anObj.PropMinPtsMult().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PropMinPtsMult"),anObj.PropMinPtsMult().Val())->ReTagThis("PropMinPtsMult"));
   aRes->AddFils(::ToXMLTree(std::string("NameCam"),anObj.NameCam())->ReTagThis("NameCam"));
   if (anObj.NameCamIsKeyCalc().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NameCamIsKeyCalc"),anObj.NameCamIsKeyCalc().Val())->ReTagThis("NameCamIsKeyCalc"));
   if (anObj.KeyCalcIsIDir().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("KeyCalcIsIDir"),anObj.KeyCalcIsIDir().Val())->ReTagThis("KeyCalcIsIDir"));
   aRes->AddFils(::ToXMLTree(std::string("IdBD"),anObj.IdBD())->ReTagThis("IdBD"));
   if (anObj.ProfSceneCouple().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ProfSceneCouple"),anObj.ProfSceneCouple().Val())->ReTagThis("ProfSceneCouple"));
   if (anObj.L2EstimPlan().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("L2EstimPlan"),anObj.L2EstimPlan().Val())->ReTagThis("L2EstimPlan"));
   if (anObj.LongueurBase().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("LongueurBase"),anObj.LongueurBase().Val())->ReTagThis("LongueurBase"));
  return aRes;
}

void xml_init(cLiaisonsInit & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.OnZonePlane(),aTree->Get("OnZonePlane",1)); //tototo 

   xml_init(anObj.TestSolPlane(),aTree->Get("TestSolPlane",1),bool(true)); //tototo 

   xml_init(anObj.NbRansacSolAppui(),aTree->Get("NbRansacSolAppui",1),int(200)); //tototo 

   xml_init(anObj.InitOrientPure(),aTree->Get("InitOrientPure",1),bool(false)); //tototo 

   xml_init(anObj.NbPtsRansacOrPure(),aTree->Get("NbPtsRansacOrPure",1),int(200)); //tototo 

   xml_init(anObj.NbTestRansacOrPure(),aTree->Get("NbTestRansacOrPure",1),int(500)); //tototo 

   xml_init(anObj.NbMinPtsRanAp(),aTree->Get("NbMinPtsRanAp",1),int(0)); //tototo 

   xml_init(anObj.NbMaxPtsRanAp(),aTree->Get("NbMaxPtsRanAp",1),int(500)); //tototo 

   xml_init(anObj.PropMinPtsMult(),aTree->Get("PropMinPtsMult",1),double(0.5)); //tototo 

   xml_init(anObj.NameCam(),aTree->Get("NameCam",1)); //tototo 

   xml_init(anObj.NameCamIsKeyCalc(),aTree->Get("NameCamIsKeyCalc",1),bool(false)); //tototo 

   xml_init(anObj.KeyCalcIsIDir(),aTree->Get("KeyCalcIsIDir",1),bool(true)); //tototo 

   xml_init(anObj.IdBD(),aTree->Get("IdBD",1)); //tototo 

   xml_init(anObj.ProfSceneCouple(),aTree->Get("ProfSceneCouple",1)); //tototo 

   xml_init(anObj.L2EstimPlan(),aTree->Get("L2EstimPlan",1),bool(true)); //tototo 

   xml_init(anObj.LongueurBase(),aTree->Get("LongueurBase",1)); //tototo 
}


std::vector< cLiaisonsInit > & cPoseFromLiaisons::LiaisonsInit()
{
   return mLiaisonsInit;
}

const std::vector< cLiaisonsInit > & cPoseFromLiaisons::LiaisonsInit()const 
{
   return mLiaisonsInit;
}

cElXMLTree * ToXMLTree(const cPoseFromLiaisons & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"PoseFromLiaisons",eXMLBranche);
  for
  (       std::vector< cLiaisonsInit >::const_iterator it=anObj.LiaisonsInit().begin();
      it !=anObj.LiaisonsInit().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("LiaisonsInit"));
  return aRes;
}

void xml_init(cPoseFromLiaisons & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.LiaisonsInit(),aTree->GetAll("LiaisonsInit",false,1));
}


cMesureAppuis & cMesurePIFRP::Ap1()
{
   return mAp1;
}

const cMesureAppuis & cMesurePIFRP::Ap1()const 
{
   return mAp1;
}


cMesureAppuis & cMesurePIFRP::Ap2()
{
   return mAp2;
}

const cMesureAppuis & cMesurePIFRP::Ap2()const 
{
   return mAp2;
}


cMesureAppuis & cMesurePIFRP::Ap3()
{
   return mAp3;
}

const cMesureAppuis & cMesurePIFRP::Ap3()const 
{
   return mAp3;
}

cElXMLTree * ToXMLTree(const cMesurePIFRP & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"MesurePIFRP",eXMLBranche);
   aRes->AddFils(ToXMLTree(anObj.Ap1())->ReTagThis("Ap1"));
   aRes->AddFils(ToXMLTree(anObj.Ap2())->ReTagThis("Ap2"));
   aRes->AddFils(ToXMLTree(anObj.Ap3())->ReTagThis("Ap3"));
  return aRes;
}

void xml_init(cMesurePIFRP & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.Ap1(),aTree->Get("Ap1",1)); //tototo 

   xml_init(anObj.Ap2(),aTree->Get("Ap2",1)); //tototo 

   xml_init(anObj.Ap3(),aTree->Get("Ap3",1)); //tototo 
}


cMesureAppuis & cInitPIFRP::Ap1()
{
   return MesurePIFRP().Val().Ap1();
}

const cMesureAppuis & cInitPIFRP::Ap1()const 
{
   return MesurePIFRP().Val().Ap1();
}


cMesureAppuis & cInitPIFRP::Ap2()
{
   return MesurePIFRP().Val().Ap2();
}

const cMesureAppuis & cInitPIFRP::Ap2()const 
{
   return MesurePIFRP().Val().Ap2();
}


cMesureAppuis & cInitPIFRP::Ap3()
{
   return MesurePIFRP().Val().Ap3();
}

const cMesureAppuis & cInitPIFRP::Ap3()const 
{
   return MesurePIFRP().Val().Ap3();
}


cTplValGesInit< cMesurePIFRP > & cInitPIFRP::MesurePIFRP()
{
   return mMesurePIFRP;
}

const cTplValGesInit< cMesurePIFRP > & cInitPIFRP::MesurePIFRP()const 
{
   return mMesurePIFRP;
}


cTplValGesInit< Pt3dr > & cInitPIFRP::DirPlan()
{
   return mDirPlan;
}

const cTplValGesInit< Pt3dr > & cInitPIFRP::DirPlan()const 
{
   return mDirPlan;
}

cElXMLTree * ToXMLTree(const cInitPIFRP & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"InitPIFRP",eXMLBranche);
   if (anObj.MesurePIFRP().IsInit())
      aRes->AddFils(ToXMLTree(anObj.MesurePIFRP().Val())->ReTagThis("MesurePIFRP"));
   if (anObj.DirPlan().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DirPlan"),anObj.DirPlan().Val())->ReTagThis("DirPlan"));
  return aRes;
}

void xml_init(cInitPIFRP & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.MesurePIFRP(),aTree->Get("MesurePIFRP",1)); //tototo 

   xml_init(anObj.DirPlan(),aTree->Get("DirPlan",1)); //tototo 
}


std::string & cPoseInitFromReperePlan::OnZonePlane()
{
   return mOnZonePlane;
}

const std::string & cPoseInitFromReperePlan::OnZonePlane()const 
{
   return mOnZonePlane;
}


cTplValGesInit< bool > & cPoseInitFromReperePlan::L2EstimPlan()
{
   return mL2EstimPlan;
}

const cTplValGesInit< bool > & cPoseInitFromReperePlan::L2EstimPlan()const 
{
   return mL2EstimPlan;
}


std::string & cPoseInitFromReperePlan::IdBD()
{
   return mIdBD;
}

const std::string & cPoseInitFromReperePlan::IdBD()const 
{
   return mIdBD;
}


std::string & cPoseInitFromReperePlan::NameCam()
{
   return mNameCam;
}

const std::string & cPoseInitFromReperePlan::NameCam()const 
{
   return mNameCam;
}


cTplValGesInit< double > & cPoseInitFromReperePlan::DEuclidPlan()
{
   return mDEuclidPlan;
}

const cTplValGesInit< double > & cPoseInitFromReperePlan::DEuclidPlan()const 
{
   return mDEuclidPlan;
}


cMesureAppuis & cPoseInitFromReperePlan::Ap1()
{
   return InitPIFRP().MesurePIFRP().Val().Ap1();
}

const cMesureAppuis & cPoseInitFromReperePlan::Ap1()const 
{
   return InitPIFRP().MesurePIFRP().Val().Ap1();
}


cMesureAppuis & cPoseInitFromReperePlan::Ap2()
{
   return InitPIFRP().MesurePIFRP().Val().Ap2();
}

const cMesureAppuis & cPoseInitFromReperePlan::Ap2()const 
{
   return InitPIFRP().MesurePIFRP().Val().Ap2();
}


cMesureAppuis & cPoseInitFromReperePlan::Ap3()
{
   return InitPIFRP().MesurePIFRP().Val().Ap3();
}

const cMesureAppuis & cPoseInitFromReperePlan::Ap3()const 
{
   return InitPIFRP().MesurePIFRP().Val().Ap3();
}


cTplValGesInit< cMesurePIFRP > & cPoseInitFromReperePlan::MesurePIFRP()
{
   return InitPIFRP().MesurePIFRP();
}

const cTplValGesInit< cMesurePIFRP > & cPoseInitFromReperePlan::MesurePIFRP()const 
{
   return InitPIFRP().MesurePIFRP();
}


cTplValGesInit< Pt3dr > & cPoseInitFromReperePlan::DirPlan()
{
   return InitPIFRP().DirPlan();
}

const cTplValGesInit< Pt3dr > & cPoseInitFromReperePlan::DirPlan()const 
{
   return InitPIFRP().DirPlan();
}


cInitPIFRP & cPoseInitFromReperePlan::InitPIFRP()
{
   return mInitPIFRP;
}

const cInitPIFRP & cPoseInitFromReperePlan::InitPIFRP()const 
{
   return mInitPIFRP;
}

cElXMLTree * ToXMLTree(const cPoseInitFromReperePlan & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"PoseInitFromReperePlan",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("OnZonePlane"),anObj.OnZonePlane())->ReTagThis("OnZonePlane"));
   if (anObj.L2EstimPlan().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("L2EstimPlan"),anObj.L2EstimPlan().Val())->ReTagThis("L2EstimPlan"));
   aRes->AddFils(::ToXMLTree(std::string("IdBD"),anObj.IdBD())->ReTagThis("IdBD"));
   aRes->AddFils(::ToXMLTree(std::string("NameCam"),anObj.NameCam())->ReTagThis("NameCam"));
   if (anObj.DEuclidPlan().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DEuclidPlan"),anObj.DEuclidPlan().Val())->ReTagThis("DEuclidPlan"));
   aRes->AddFils(ToXMLTree(anObj.InitPIFRP())->ReTagThis("InitPIFRP"));
  return aRes;
}

void xml_init(cPoseInitFromReperePlan & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.OnZonePlane(),aTree->Get("OnZonePlane",1)); //tototo 

   xml_init(anObj.L2EstimPlan(),aTree->Get("L2EstimPlan",1),bool(true)); //tototo 

   xml_init(anObj.IdBD(),aTree->Get("IdBD",1)); //tototo 

   xml_init(anObj.NameCam(),aTree->Get("NameCam",1)); //tototo 

   xml_init(anObj.DEuclidPlan(),aTree->Get("DEuclidPlan",1)); //tototo 

   xml_init(anObj.InitPIFRP(),aTree->Get("InitPIFRP",1)); //tototo 
}


cTplValGesInit< std::string > & cPosValueInit::PosId()
{
   return mPosId;
}

const cTplValGesInit< std::string > & cPosValueInit::PosId()const 
{
   return mPosId;
}


cTplValGesInit< std::string > & cPosValueInit::PosFromBDOrient()
{
   return mPosFromBDOrient;
}

const cTplValGesInit< std::string > & cPosValueInit::PosFromBDOrient()const 
{
   return mPosFromBDOrient;
}


std::string & cPosValueInit::Id()
{
   return PosFromBDAppuis().Val().Id();
}

const std::string & cPosValueInit::Id()const 
{
   return PosFromBDAppuis().Val().Id();
}


int & cPosValueInit::NbTestRansac()
{
   return PosFromBDAppuis().Val().NbTestRansac();
}

const int & cPosValueInit::NbTestRansac()const 
{
   return PosFromBDAppuis().Val().NbTestRansac();
}


cTplValGesInit< Pt3dr > & cPosValueInit::DirApprox()
{
   return PosFromBDAppuis().Val().DirApprox();
}

const cTplValGesInit< Pt3dr > & cPosValueInit::DirApprox()const 
{
   return PosFromBDAppuis().Val().DirApprox();
}


cTplValGesInit< cPosFromBDAppuis > & cPosValueInit::PosFromBDAppuis()
{
   return mPosFromBDAppuis;
}

const cTplValGesInit< cPosFromBDAppuis > & cPosValueInit::PosFromBDAppuis()const 
{
   return mPosFromBDAppuis;
}


std::vector< cLiaisonsInit > & cPosValueInit::LiaisonsInit()
{
   return PoseFromLiaisons().Val().LiaisonsInit();
}

const std::vector< cLiaisonsInit > & cPosValueInit::LiaisonsInit()const 
{
   return PoseFromLiaisons().Val().LiaisonsInit();
}


cTplValGesInit< cPoseFromLiaisons > & cPosValueInit::PoseFromLiaisons()
{
   return mPoseFromLiaisons;
}

const cTplValGesInit< cPoseFromLiaisons > & cPosValueInit::PoseFromLiaisons()const 
{
   return mPoseFromLiaisons;
}


std::string & cPosValueInit::OnZonePlane()
{
   return PoseInitFromReperePlan().Val().OnZonePlane();
}

const std::string & cPosValueInit::OnZonePlane()const 
{
   return PoseInitFromReperePlan().Val().OnZonePlane();
}


cTplValGesInit< bool > & cPosValueInit::L2EstimPlan()
{
   return PoseInitFromReperePlan().Val().L2EstimPlan();
}

const cTplValGesInit< bool > & cPosValueInit::L2EstimPlan()const 
{
   return PoseInitFromReperePlan().Val().L2EstimPlan();
}


std::string & cPosValueInit::IdBD()
{
   return PoseInitFromReperePlan().Val().IdBD();
}

const std::string & cPosValueInit::IdBD()const 
{
   return PoseInitFromReperePlan().Val().IdBD();
}


std::string & cPosValueInit::NameCam()
{
   return PoseInitFromReperePlan().Val().NameCam();
}

const std::string & cPosValueInit::NameCam()const 
{
   return PoseInitFromReperePlan().Val().NameCam();
}


cTplValGesInit< double > & cPosValueInit::DEuclidPlan()
{
   return PoseInitFromReperePlan().Val().DEuclidPlan();
}

const cTplValGesInit< double > & cPosValueInit::DEuclidPlan()const 
{
   return PoseInitFromReperePlan().Val().DEuclidPlan();
}


cMesureAppuis & cPosValueInit::Ap1()
{
   return PoseInitFromReperePlan().Val().InitPIFRP().MesurePIFRP().Val().Ap1();
}

const cMesureAppuis & cPosValueInit::Ap1()const 
{
   return PoseInitFromReperePlan().Val().InitPIFRP().MesurePIFRP().Val().Ap1();
}


cMesureAppuis & cPosValueInit::Ap2()
{
   return PoseInitFromReperePlan().Val().InitPIFRP().MesurePIFRP().Val().Ap2();
}

const cMesureAppuis & cPosValueInit::Ap2()const 
{
   return PoseInitFromReperePlan().Val().InitPIFRP().MesurePIFRP().Val().Ap2();
}


cMesureAppuis & cPosValueInit::Ap3()
{
   return PoseInitFromReperePlan().Val().InitPIFRP().MesurePIFRP().Val().Ap3();
}

const cMesureAppuis & cPosValueInit::Ap3()const 
{
   return PoseInitFromReperePlan().Val().InitPIFRP().MesurePIFRP().Val().Ap3();
}


cTplValGesInit< cMesurePIFRP > & cPosValueInit::MesurePIFRP()
{
   return PoseInitFromReperePlan().Val().InitPIFRP().MesurePIFRP();
}

const cTplValGesInit< cMesurePIFRP > & cPosValueInit::MesurePIFRP()const 
{
   return PoseInitFromReperePlan().Val().InitPIFRP().MesurePIFRP();
}


cTplValGesInit< Pt3dr > & cPosValueInit::DirPlan()
{
   return PoseInitFromReperePlan().Val().InitPIFRP().DirPlan();
}

const cTplValGesInit< Pt3dr > & cPosValueInit::DirPlan()const 
{
   return PoseInitFromReperePlan().Val().InitPIFRP().DirPlan();
}


cInitPIFRP & cPosValueInit::InitPIFRP()
{
   return PoseInitFromReperePlan().Val().InitPIFRP();
}

const cInitPIFRP & cPosValueInit::InitPIFRP()const 
{
   return PoseInitFromReperePlan().Val().InitPIFRP();
}


cTplValGesInit< cPoseInitFromReperePlan > & cPosValueInit::PoseInitFromReperePlan()
{
   return mPoseInitFromReperePlan;
}

const cTplValGesInit< cPoseInitFromReperePlan > & cPosValueInit::PoseInitFromReperePlan()const 
{
   return mPoseInitFromReperePlan;
}

cElXMLTree * ToXMLTree(const cPosValueInit & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"PosValueInit",eXMLBranche);
   if (anObj.PosId().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PosId"),anObj.PosId().Val())->ReTagThis("PosId"));
   if (anObj.PosFromBDOrient().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PosFromBDOrient"),anObj.PosFromBDOrient().Val())->ReTagThis("PosFromBDOrient"));
   if (anObj.PosFromBDAppuis().IsInit())
      aRes->AddFils(ToXMLTree(anObj.PosFromBDAppuis().Val())->ReTagThis("PosFromBDAppuis"));
   if (anObj.PoseFromLiaisons().IsInit())
      aRes->AddFils(ToXMLTree(anObj.PoseFromLiaisons().Val())->ReTagThis("PoseFromLiaisons"));
   if (anObj.PoseInitFromReperePlan().IsInit())
      aRes->AddFils(ToXMLTree(anObj.PoseInitFromReperePlan().Val())->ReTagThis("PoseInitFromReperePlan"));
  return aRes;
}

void xml_init(cPosValueInit & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.PosId(),aTree->Get("PosId",1)); //tototo 

   xml_init(anObj.PosFromBDOrient(),aTree->Get("PosFromBDOrient",1)); //tototo 

   xml_init(anObj.PosFromBDAppuis(),aTree->Get("PosFromBDAppuis",1)); //tototo 

   xml_init(anObj.PoseFromLiaisons(),aTree->Get("PoseFromLiaisons",1)); //tototo 

   xml_init(anObj.PoseInitFromReperePlan(),aTree->Get("PoseInitFromReperePlan",1)); //tototo 
}


cTplValGesInit< cSetOrientationInterne > & cPoseCameraInc::OrInterne()
{
   return mOrInterne;
}

const cTplValGesInit< cSetOrientationInterne > & cPoseCameraInc::OrInterne()const 
{
   return mOrInterne;
}


cTplValGesInit< std::string > & cPoseCameraInc::IdBDCentre()
{
   return mIdBDCentre;
}

const cTplValGesInit< std::string > & cPoseCameraInc::IdBDCentre()const 
{
   return mIdBDCentre;
}


cTplValGesInit< bool > & cPoseCameraInc::InitNow()
{
   return mInitNow;
}

const cTplValGesInit< bool > & cPoseCameraInc::InitNow()const 
{
   return mInitNow;
}


cTplValGesInit< double > & cPoseCameraInc::ProfSceneImage()
{
   return mProfSceneImage;
}

const cTplValGesInit< double > & cPoseCameraInc::ProfSceneImage()const 
{
   return mProfSceneImage;
}


cTplValGesInit< std::string > & cPoseCameraInc::Directory()
{
   return mDirectory;
}

const cTplValGesInit< std::string > & cPoseCameraInc::Directory()const 
{
   return mDirectory;
}


std::list< std::string > & cPoseCameraInc::PatternName()
{
   return mPatternName;
}

const std::list< std::string > & cPoseCameraInc::PatternName()const 
{
   return mPatternName;
}


cTplValGesInit< std::string > & cPoseCameraInc::AutomGetImC()
{
   return mAutomGetImC;
}

const cTplValGesInit< std::string > & cPoseCameraInc::AutomGetImC()const 
{
   return mAutomGetImC;
}


cTplValGesInit< std::string > & cPoseCameraInc::TestFewerTiePoints()
{
   return mTestFewerTiePoints;
}

const cTplValGesInit< std::string > & cPoseCameraInc::TestFewerTiePoints()const 
{
   return mTestFewerTiePoints;
}


cTplValGesInit< cNameFilter > & cPoseCameraInc::Filter()
{
   return mFilter;
}

const cTplValGesInit< cNameFilter > & cPoseCameraInc::Filter()const 
{
   return mFilter;
}


cTplValGesInit< cElRegex_Ptr > & cPoseCameraInc::PatternRefuteur()
{
   return mPatternRefuteur;
}

const cTplValGesInit< cElRegex_Ptr > & cPoseCameraInc::PatternRefuteur()const 
{
   return mPatternRefuteur;
}


cTplValGesInit< bool > & cPoseCameraInc::AutoRefutDupl()
{
   return mAutoRefutDupl;
}

const cTplValGesInit< bool > & cPoseCameraInc::AutoRefutDupl()const 
{
   return mAutoRefutDupl;
}


cTplValGesInit< std::string > & cPoseCameraInc::KeyTranscriptionName()
{
   return mKeyTranscriptionName;
}

const cTplValGesInit< std::string > & cPoseCameraInc::KeyTranscriptionName()const 
{
   return mKeyTranscriptionName;
}


cTplValGesInit< std::string > & cPoseCameraInc::AddAllNameConnectedBy()
{
   return mAddAllNameConnectedBy;
}

const cTplValGesInit< std::string > & cPoseCameraInc::AddAllNameConnectedBy()const 
{
   return mAddAllNameConnectedBy;
}


cTplValGesInit< std::string > & cPoseCameraInc::FilterConnecBy()
{
   return mFilterConnecBy;
}

const cTplValGesInit< std::string > & cPoseCameraInc::FilterConnecBy()const 
{
   return mFilterConnecBy;
}


cTplValGesInit< bool > & cPoseCameraInc::Show()
{
   return MEP_SPEC_MST().Val().Show();
}

const cTplValGesInit< bool > & cPoseCameraInc::Show()const 
{
   return MEP_SPEC_MST().Val().Show();
}


cTplValGesInit< int > & cPoseCameraInc::MinNbPtsInit()
{
   return MEP_SPEC_MST().Val().MinNbPtsInit();
}

const cTplValGesInit< int > & cPoseCameraInc::MinNbPtsInit()const 
{
   return MEP_SPEC_MST().Val().MinNbPtsInit();
}


cTplValGesInit< double > & cPoseCameraInc::ExpDist()
{
   return MEP_SPEC_MST().Val().ExpDist();
}

const cTplValGesInit< double > & cPoseCameraInc::ExpDist()const 
{
   return MEP_SPEC_MST().Val().ExpDist();
}


cTplValGesInit< double > & cPoseCameraInc::ExpNb()
{
   return MEP_SPEC_MST().Val().ExpNb();
}

const cTplValGesInit< double > & cPoseCameraInc::ExpNb()const 
{
   return MEP_SPEC_MST().Val().ExpNb();
}


cTplValGesInit< bool > & cPoseCameraInc::MontageOnInit()
{
   return MEP_SPEC_MST().Val().MontageOnInit();
}

const cTplValGesInit< bool > & cPoseCameraInc::MontageOnInit()const 
{
   return MEP_SPEC_MST().Val().MontageOnInit();
}


cTplValGesInit< int > & cPoseCameraInc::NbInitMinBeforeUnconnect()
{
   return MEP_SPEC_MST().Val().NbInitMinBeforeUnconnect();
}

const cTplValGesInit< int > & cPoseCameraInc::NbInitMinBeforeUnconnect()const 
{
   return MEP_SPEC_MST().Val().NbInitMinBeforeUnconnect();
}


cTplValGesInit< cMEP_SPEC_MST > & cPoseCameraInc::MEP_SPEC_MST()
{
   return mMEP_SPEC_MST;
}

const cTplValGesInit< cMEP_SPEC_MST > & cPoseCameraInc::MEP_SPEC_MST()const 
{
   return mMEP_SPEC_MST;
}


cOptimizationPowel & cPoseCameraInc::ParamOptim()
{
   return OptimizeAfterInit().Val().ParamOptim();
}

const cOptimizationPowel & cPoseCameraInc::ParamOptim()const 
{
   return OptimizeAfterInit().Val().ParamOptim();
}


std::list< cApplyOAI > & cPoseCameraInc::ApplyOAI()
{
   return OptimizeAfterInit().Val().ApplyOAI();
}

const std::list< cApplyOAI > & cPoseCameraInc::ApplyOAI()const 
{
   return OptimizeAfterInit().Val().ApplyOAI();
}


cTplValGesInit< cOptimizeAfterInit > & cPoseCameraInc::OptimizeAfterInit()
{
   return mOptimizeAfterInit;
}

const cTplValGesInit< cOptimizeAfterInit > & cPoseCameraInc::OptimizeAfterInit()const 
{
   return mOptimizeAfterInit;
}


cTplValGesInit< bool > & cPoseCameraInc::ReverseOrderName()
{
   return mReverseOrderName;
}

const cTplValGesInit< bool > & cPoseCameraInc::ReverseOrderName()const 
{
   return mReverseOrderName;
}


std::string & cPoseCameraInc::CalcNameCalib()
{
   return mCalcNameCalib;
}

const std::string & cPoseCameraInc::CalcNameCalib()const 
{
   return mCalcNameCalib;
}


cTplValGesInit< std::string > & cPoseCameraInc::PosesDeRattachement()
{
   return mPosesDeRattachement;
}

const cTplValGesInit< std::string > & cPoseCameraInc::PosesDeRattachement()const 
{
   return mPosesDeRattachement;
}


cTplValGesInit< bool > & cPoseCameraInc::NoErroOnRat()
{
   return mNoErroOnRat;
}

const cTplValGesInit< bool > & cPoseCameraInc::NoErroOnRat()const 
{
   return mNoErroOnRat;
}


cTplValGesInit< bool > & cPoseCameraInc::ByPattern()
{
   return mByPattern;
}

const cTplValGesInit< bool > & cPoseCameraInc::ByPattern()const 
{
   return mByPattern;
}


cTplValGesInit< std::string > & cPoseCameraInc::KeyFilterExistingFile()
{
   return mKeyFilterExistingFile;
}

const cTplValGesInit< std::string > & cPoseCameraInc::KeyFilterExistingFile()const 
{
   return mKeyFilterExistingFile;
}


cTplValGesInit< bool > & cPoseCameraInc::ByKey()
{
   return mByKey;
}

const cTplValGesInit< bool > & cPoseCameraInc::ByKey()const 
{
   return mByKey;
}


cTplValGesInit< bool > & cPoseCameraInc::ByFile()
{
   return mByFile;
}

const cTplValGesInit< bool > & cPoseCameraInc::ByFile()const 
{
   return mByFile;
}


cTplValGesInit< std::string > & cPoseCameraInc::PosId()
{
   return PosValueInit().PosId();
}

const cTplValGesInit< std::string > & cPoseCameraInc::PosId()const 
{
   return PosValueInit().PosId();
}


cTplValGesInit< std::string > & cPoseCameraInc::PosFromBDOrient()
{
   return PosValueInit().PosFromBDOrient();
}

const cTplValGesInit< std::string > & cPoseCameraInc::PosFromBDOrient()const 
{
   return PosValueInit().PosFromBDOrient();
}


std::string & cPoseCameraInc::Id()
{
   return PosValueInit().PosFromBDAppuis().Val().Id();
}

const std::string & cPoseCameraInc::Id()const 
{
   return PosValueInit().PosFromBDAppuis().Val().Id();
}


int & cPoseCameraInc::NbTestRansac()
{
   return PosValueInit().PosFromBDAppuis().Val().NbTestRansac();
}

const int & cPoseCameraInc::NbTestRansac()const 
{
   return PosValueInit().PosFromBDAppuis().Val().NbTestRansac();
}


cTplValGesInit< Pt3dr > & cPoseCameraInc::DirApprox()
{
   return PosValueInit().PosFromBDAppuis().Val().DirApprox();
}

const cTplValGesInit< Pt3dr > & cPoseCameraInc::DirApprox()const 
{
   return PosValueInit().PosFromBDAppuis().Val().DirApprox();
}


cTplValGesInit< cPosFromBDAppuis > & cPoseCameraInc::PosFromBDAppuis()
{
   return PosValueInit().PosFromBDAppuis();
}

const cTplValGesInit< cPosFromBDAppuis > & cPoseCameraInc::PosFromBDAppuis()const 
{
   return PosValueInit().PosFromBDAppuis();
}


std::vector< cLiaisonsInit > & cPoseCameraInc::LiaisonsInit()
{
   return PosValueInit().PoseFromLiaisons().Val().LiaisonsInit();
}

const std::vector< cLiaisonsInit > & cPoseCameraInc::LiaisonsInit()const 
{
   return PosValueInit().PoseFromLiaisons().Val().LiaisonsInit();
}


cTplValGesInit< cPoseFromLiaisons > & cPoseCameraInc::PoseFromLiaisons()
{
   return PosValueInit().PoseFromLiaisons();
}

const cTplValGesInit< cPoseFromLiaisons > & cPoseCameraInc::PoseFromLiaisons()const 
{
   return PosValueInit().PoseFromLiaisons();
}


std::string & cPoseCameraInc::OnZonePlane()
{
   return PosValueInit().PoseInitFromReperePlan().Val().OnZonePlane();
}

const std::string & cPoseCameraInc::OnZonePlane()const 
{
   return PosValueInit().PoseInitFromReperePlan().Val().OnZonePlane();
}


cTplValGesInit< bool > & cPoseCameraInc::L2EstimPlan()
{
   return PosValueInit().PoseInitFromReperePlan().Val().L2EstimPlan();
}

const cTplValGesInit< bool > & cPoseCameraInc::L2EstimPlan()const 
{
   return PosValueInit().PoseInitFromReperePlan().Val().L2EstimPlan();
}


std::string & cPoseCameraInc::IdBD()
{
   return PosValueInit().PoseInitFromReperePlan().Val().IdBD();
}

const std::string & cPoseCameraInc::IdBD()const 
{
   return PosValueInit().PoseInitFromReperePlan().Val().IdBD();
}


std::string & cPoseCameraInc::NameCam()
{
   return PosValueInit().PoseInitFromReperePlan().Val().NameCam();
}

const std::string & cPoseCameraInc::NameCam()const 
{
   return PosValueInit().PoseInitFromReperePlan().Val().NameCam();
}


cTplValGesInit< double > & cPoseCameraInc::DEuclidPlan()
{
   return PosValueInit().PoseInitFromReperePlan().Val().DEuclidPlan();
}

const cTplValGesInit< double > & cPoseCameraInc::DEuclidPlan()const 
{
   return PosValueInit().PoseInitFromReperePlan().Val().DEuclidPlan();
}


cMesureAppuis & cPoseCameraInc::Ap1()
{
   return PosValueInit().PoseInitFromReperePlan().Val().InitPIFRP().MesurePIFRP().Val().Ap1();
}

const cMesureAppuis & cPoseCameraInc::Ap1()const 
{
   return PosValueInit().PoseInitFromReperePlan().Val().InitPIFRP().MesurePIFRP().Val().Ap1();
}


cMesureAppuis & cPoseCameraInc::Ap2()
{
   return PosValueInit().PoseInitFromReperePlan().Val().InitPIFRP().MesurePIFRP().Val().Ap2();
}

const cMesureAppuis & cPoseCameraInc::Ap2()const 
{
   return PosValueInit().PoseInitFromReperePlan().Val().InitPIFRP().MesurePIFRP().Val().Ap2();
}


cMesureAppuis & cPoseCameraInc::Ap3()
{
   return PosValueInit().PoseInitFromReperePlan().Val().InitPIFRP().MesurePIFRP().Val().Ap3();
}

const cMesureAppuis & cPoseCameraInc::Ap3()const 
{
   return PosValueInit().PoseInitFromReperePlan().Val().InitPIFRP().MesurePIFRP().Val().Ap3();
}


cTplValGesInit< cMesurePIFRP > & cPoseCameraInc::MesurePIFRP()
{
   return PosValueInit().PoseInitFromReperePlan().Val().InitPIFRP().MesurePIFRP();
}

const cTplValGesInit< cMesurePIFRP > & cPoseCameraInc::MesurePIFRP()const 
{
   return PosValueInit().PoseInitFromReperePlan().Val().InitPIFRP().MesurePIFRP();
}


cTplValGesInit< Pt3dr > & cPoseCameraInc::DirPlan()
{
   return PosValueInit().PoseInitFromReperePlan().Val().InitPIFRP().DirPlan();
}

const cTplValGesInit< Pt3dr > & cPoseCameraInc::DirPlan()const 
{
   return PosValueInit().PoseInitFromReperePlan().Val().InitPIFRP().DirPlan();
}


cInitPIFRP & cPoseCameraInc::InitPIFRP()
{
   return PosValueInit().PoseInitFromReperePlan().Val().InitPIFRP();
}

const cInitPIFRP & cPoseCameraInc::InitPIFRP()const 
{
   return PosValueInit().PoseInitFromReperePlan().Val().InitPIFRP();
}


cTplValGesInit< cPoseInitFromReperePlan > & cPoseCameraInc::PoseInitFromReperePlan()
{
   return PosValueInit().PoseInitFromReperePlan();
}

const cTplValGesInit< cPoseInitFromReperePlan > & cPoseCameraInc::PoseInitFromReperePlan()const 
{
   return PosValueInit().PoseInitFromReperePlan();
}


cPosValueInit & cPoseCameraInc::PosValueInit()
{
   return mPosValueInit;
}

const cPosValueInit & cPoseCameraInc::PosValueInit()const 
{
   return mPosValueInit;
}

cElXMLTree * ToXMLTree(const cPoseCameraInc & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"PoseCameraInc",eXMLBranche);
   if (anObj.OrInterne().IsInit())
      aRes->AddFils(ToXMLTree(anObj.OrInterne().Val())->ReTagThis("OrInterne"));
   if (anObj.IdBDCentre().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("IdBDCentre"),anObj.IdBDCentre().Val())->ReTagThis("IdBDCentre"));
   if (anObj.InitNow().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("InitNow"),anObj.InitNow().Val())->ReTagThis("InitNow"));
   if (anObj.ProfSceneImage().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ProfSceneImage"),anObj.ProfSceneImage().Val())->ReTagThis("ProfSceneImage"));
   if (anObj.Directory().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Directory"),anObj.Directory().Val())->ReTagThis("Directory"));
  for
  (       std::list< std::string >::const_iterator it=anObj.PatternName().begin();
      it !=anObj.PatternName().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("PatternName"),(*it))->ReTagThis("PatternName"));
   if (anObj.AutomGetImC().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("AutomGetImC"),anObj.AutomGetImC().Val())->ReTagThis("AutomGetImC"));
   if (anObj.TestFewerTiePoints().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("TestFewerTiePoints"),anObj.TestFewerTiePoints().Val())->ReTagThis("TestFewerTiePoints"));
   if (anObj.Filter().IsInit())
      aRes->AddFils(ToXMLTree(anObj.Filter().Val())->ReTagThis("Filter"));
   if (anObj.PatternRefuteur().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PatternRefuteur"),anObj.PatternRefuteur().Val())->ReTagThis("PatternRefuteur"));
   if (anObj.AutoRefutDupl().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("AutoRefutDupl"),anObj.AutoRefutDupl().Val())->ReTagThis("AutoRefutDupl"));
   if (anObj.KeyTranscriptionName().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("KeyTranscriptionName"),anObj.KeyTranscriptionName().Val())->ReTagThis("KeyTranscriptionName"));
   if (anObj.AddAllNameConnectedBy().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("AddAllNameConnectedBy"),anObj.AddAllNameConnectedBy().Val())->ReTagThis("AddAllNameConnectedBy"));
   if (anObj.FilterConnecBy().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("FilterConnecBy"),anObj.FilterConnecBy().Val())->ReTagThis("FilterConnecBy"));
   if (anObj.MEP_SPEC_MST().IsInit())
      aRes->AddFils(ToXMLTree(anObj.MEP_SPEC_MST().Val())->ReTagThis("MEP_SPEC_MST"));
   if (anObj.OptimizeAfterInit().IsInit())
      aRes->AddFils(ToXMLTree(anObj.OptimizeAfterInit().Val())->ReTagThis("OptimizeAfterInit"));
   if (anObj.ReverseOrderName().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ReverseOrderName"),anObj.ReverseOrderName().Val())->ReTagThis("ReverseOrderName"));
   aRes->AddFils(::ToXMLTree(std::string("CalcNameCalib"),anObj.CalcNameCalib())->ReTagThis("CalcNameCalib"));
   if (anObj.PosesDeRattachement().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PosesDeRattachement"),anObj.PosesDeRattachement().Val())->ReTagThis("PosesDeRattachement"));
   if (anObj.NoErroOnRat().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NoErroOnRat"),anObj.NoErroOnRat().Val())->ReTagThis("NoErroOnRat"));
   if (anObj.ByPattern().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ByPattern"),anObj.ByPattern().Val())->ReTagThis("ByPattern"));
   if (anObj.KeyFilterExistingFile().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("KeyFilterExistingFile"),anObj.KeyFilterExistingFile().Val())->ReTagThis("KeyFilterExistingFile"));
   if (anObj.ByKey().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ByKey"),anObj.ByKey().Val())->ReTagThis("ByKey"));
   if (anObj.ByFile().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ByFile"),anObj.ByFile().Val())->ReTagThis("ByFile"));
   aRes->AddFils(ToXMLTree(anObj.PosValueInit())->ReTagThis("PosValueInit"));
  return aRes;
}

void xml_init(cPoseCameraInc & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.OrInterne(),aTree->Get("OrInterne",1)); //tototo 

   xml_init(anObj.IdBDCentre(),aTree->Get("IdBDCentre",1)); //tototo 

   xml_init(anObj.InitNow(),aTree->Get("InitNow",1),bool(true)); //tototo 

   xml_init(anObj.ProfSceneImage(),aTree->Get("ProfSceneImage",1)); //tototo 

   xml_init(anObj.Directory(),aTree->Get("Directory",1),std::string("")); //tototo 

   xml_init(anObj.PatternName(),aTree->GetAll("PatternName",false,1));

   xml_init(anObj.AutomGetImC(),aTree->Get("AutomGetImC",1)); //tototo 

   xml_init(anObj.TestFewerTiePoints(),aTree->Get("TestFewerTiePoints",1)); //tototo 

   xml_init(anObj.Filter(),aTree->Get("Filter",1)); //tototo 

   xml_init(anObj.PatternRefuteur(),aTree->Get("PatternRefuteur",1)); //tototo 

   xml_init(anObj.AutoRefutDupl(),aTree->Get("AutoRefutDupl",1),bool(true)); //tototo 

   xml_init(anObj.KeyTranscriptionName(),aTree->Get("KeyTranscriptionName",1)); //tototo 

   xml_init(anObj.AddAllNameConnectedBy(),aTree->Get("AddAllNameConnectedBy",1)); //tototo 

   xml_init(anObj.FilterConnecBy(),aTree->Get("FilterConnecBy",1)); //tototo 

   xml_init(anObj.MEP_SPEC_MST(),aTree->Get("MEP_SPEC_MST",1)); //tototo 

   xml_init(anObj.OptimizeAfterInit(),aTree->Get("OptimizeAfterInit",1)); //tototo 

   xml_init(anObj.ReverseOrderName(),aTree->Get("ReverseOrderName",1),bool(false)); //tototo 

   xml_init(anObj.CalcNameCalib(),aTree->Get("CalcNameCalib",1)); //tototo 

   xml_init(anObj.PosesDeRattachement(),aTree->Get("PosesDeRattachement",1)); //tototo 

   xml_init(anObj.NoErroOnRat(),aTree->Get("NoErroOnRat",1),bool(true)); //tototo 

   xml_init(anObj.ByPattern(),aTree->Get("ByPattern",1),bool(true)); //tototo 

   xml_init(anObj.KeyFilterExistingFile(),aTree->Get("KeyFilterExistingFile",1)); //tototo 

   xml_init(anObj.ByKey(),aTree->Get("ByKey",1),bool(false)); //tototo 

   xml_init(anObj.ByFile(),aTree->Get("ByFile",1),bool(false)); //tototo 

   xml_init(anObj.PosValueInit(),aTree->Get("PosValueInit",1)); //tototo 
}


std::string & cGroupeDePose::KeyPose2Grp()
{
   return mKeyPose2Grp;
}

const std::string & cGroupeDePose::KeyPose2Grp()const 
{
   return mKeyPose2Grp;
}


std::string & cGroupeDePose::Id()
{
   return mId;
}

const std::string & cGroupeDePose::Id()const 
{
   return mId;
}


cTplValGesInit< bool > & cGroupeDePose::ShowCreate()
{
   return mShowCreate;
}

const cTplValGesInit< bool > & cGroupeDePose::ShowCreate()const 
{
   return mShowCreate;
}

cElXMLTree * ToXMLTree(const cGroupeDePose & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"GroupeDePose",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("KeyPose2Grp"),anObj.KeyPose2Grp())->ReTagThis("KeyPose2Grp"));
   aRes->AddFils(::ToXMLTree(std::string("Id"),anObj.Id())->ReTagThis("Id"));
   if (anObj.ShowCreate().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ShowCreate"),anObj.ShowCreate().Val())->ReTagThis("ShowCreate"));
  return aRes;
}

void xml_init(cGroupeDePose & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.KeyPose2Grp(),aTree->Get("KeyPose2Grp",1)); //tototo 

   xml_init(anObj.Id(),aTree->Get("Id",1)); //tototo 

   xml_init(anObj.ShowCreate(),aTree->Get("ShowCreate",1),bool(false)); //tototo 
}


std::string & cLiaisonsApplyContrainte::NameRef()
{
   return mNameRef;
}

const std::string & cLiaisonsApplyContrainte::NameRef()const 
{
   return mNameRef;
}


std::string & cLiaisonsApplyContrainte::PatternI1()
{
   return mPatternI1;
}

const std::string & cLiaisonsApplyContrainte::PatternI1()const 
{
   return mPatternI1;
}


cTplValGesInit< std::string > & cLiaisonsApplyContrainte::PatternI2()
{
   return mPatternI2;
}

const cTplValGesInit< std::string > & cLiaisonsApplyContrainte::PatternI2()const 
{
   return mPatternI2;
}

cElXMLTree * ToXMLTree(const cLiaisonsApplyContrainte & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"LiaisonsApplyContrainte",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("NameRef"),anObj.NameRef())->ReTagThis("NameRef"));
   aRes->AddFils(::ToXMLTree(std::string("PatternI1"),anObj.PatternI1())->ReTagThis("PatternI1"));
   if (anObj.PatternI2().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PatternI2"),anObj.PatternI2().Val())->ReTagThis("PatternI2"));
  return aRes;
}

void xml_init(cLiaisonsApplyContrainte & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.NameRef(),aTree->Get("NameRef",1)); //tototo 

   xml_init(anObj.PatternI1(),aTree->Get("PatternI1",1)); //tototo 

   xml_init(anObj.PatternI2(),aTree->Get("PatternI2",1),std::string(".*")); //tototo 
}


cTplValGesInit< std::string > & cInitSurf::ZonePlane()
{
   return mZonePlane;
}

const cTplValGesInit< std::string > & cInitSurf::ZonePlane()const 
{
   return mZonePlane;
}

cElXMLTree * ToXMLTree(const cInitSurf & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"InitSurf",eXMLBranche);
   if (anObj.ZonePlane().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ZonePlane"),anObj.ZonePlane().Val())->ReTagThis("ZonePlane"));
  return aRes;
}

void xml_init(cInitSurf & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.ZonePlane(),aTree->Get("ZonePlane",1)); //tototo 
}


std::list< cLiaisonsApplyContrainte > & cSurfParamInc::LiaisonsApplyContrainte()
{
   return mLiaisonsApplyContrainte;
}

const std::list< cLiaisonsApplyContrainte > & cSurfParamInc::LiaisonsApplyContrainte()const 
{
   return mLiaisonsApplyContrainte;
}


cTplValGesInit< std::string > & cSurfParamInc::ZonePlane()
{
   return InitSurf().ZonePlane();
}

const cTplValGesInit< std::string > & cSurfParamInc::ZonePlane()const 
{
   return InitSurf().ZonePlane();
}


cInitSurf & cSurfParamInc::InitSurf()
{
   return mInitSurf;
}

const cInitSurf & cSurfParamInc::InitSurf()const 
{
   return mInitSurf;
}

cElXMLTree * ToXMLTree(const cSurfParamInc & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"SurfParamInc",eXMLBranche);
  for
  (       std::list< cLiaisonsApplyContrainte >::const_iterator it=anObj.LiaisonsApplyContrainte().begin();
      it !=anObj.LiaisonsApplyContrainte().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("LiaisonsApplyContrainte"));
   aRes->AddFils(ToXMLTree(anObj.InitSurf())->ReTagThis("InitSurf"));
  return aRes;
}

void xml_init(cSurfParamInc & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.LiaisonsApplyContrainte(),aTree->GetAll("LiaisonsApplyContrainte",false,1));

   xml_init(anObj.InitSurf(),aTree->Get("InitSurf",1)); //tototo 
}


std::string & cPointFlottantInc::Id()
{
   return mId;
}

const std::string & cPointFlottantInc::Id()const 
{
   return mId;
}


std::string & cPointFlottantInc::KeySetOrPat()
{
   return mKeySetOrPat;
}

const std::string & cPointFlottantInc::KeySetOrPat()const 
{
   return mKeySetOrPat;
}


cTplValGesInit< cModifIncPtsFlottant > & cPointFlottantInc::ModifInc()
{
   return mModifInc;
}

const cTplValGesInit< cModifIncPtsFlottant > & cPointFlottantInc::ModifInc()const 
{
   return mModifInc;
}

cElXMLTree * ToXMLTree(const cPointFlottantInc & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"PointFlottantInc",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Id"),anObj.Id())->ReTagThis("Id"));
   aRes->AddFils(::ToXMLTree(std::string("KeySetOrPat"),anObj.KeySetOrPat())->ReTagThis("KeySetOrPat"));
   if (anObj.ModifInc().IsInit())
      aRes->AddFils(ToXMLTree(anObj.ModifInc().Val())->ReTagThis("ModifInc"));
  return aRes;
}

void xml_init(cPointFlottantInc & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.Id(),aTree->Get("Id",1)); //tototo 

   xml_init(anObj.KeySetOrPat(),aTree->Get("KeySetOrPat",1)); //tototo 

   xml_init(anObj.ModifInc(),aTree->Get("ModifInc",1)); //tototo 
}


cTplValGesInit< double > & cSectionInconnues::SeuilAutomFE()
{
   return mSeuilAutomFE;
}

const cTplValGesInit< double > & cSectionInconnues::SeuilAutomFE()const 
{
   return mSeuilAutomFE;
}


cTplValGesInit< bool > & cSectionInconnues::AutoriseToujoursUneSeuleLiaison()
{
   return mAutoriseToujoursUneSeuleLiaison;
}

const cTplValGesInit< bool > & cSectionInconnues::AutoriseToujoursUneSeuleLiaison()const 
{
   return mAutoriseToujoursUneSeuleLiaison;
}


cTplValGesInit< cMapName2Name > & cSectionInconnues::MapMaskHom()
{
   return mMapMaskHom;
}

const cTplValGesInit< cMapName2Name > & cSectionInconnues::MapMaskHom()const 
{
   return mMapMaskHom;
}


cTplValGesInit< bool > & cSectionInconnues::SauvePMoyenOnlyWithMasq()
{
   return mSauvePMoyenOnlyWithMasq;
}

const cTplValGesInit< bool > & cSectionInconnues::SauvePMoyenOnlyWithMasq()const 
{
   return mSauvePMoyenOnlyWithMasq;
}


std::list< cCalibrationCameraInc > & cSectionInconnues::CalibrationCameraInc()
{
   return mCalibrationCameraInc;
}

const std::list< cCalibrationCameraInc > & cSectionInconnues::CalibrationCameraInc()const 
{
   return mCalibrationCameraInc;
}


cTplValGesInit< int > & cSectionInconnues::SeuilL1EstimMatrEss()
{
   return mSeuilL1EstimMatrEss;
}

const cTplValGesInit< int > & cSectionInconnues::SeuilL1EstimMatrEss()const 
{
   return mSeuilL1EstimMatrEss;
}


cTplValGesInit< cSetOrientationInterne > & cSectionInconnues::GlobOrInterne()
{
   return mGlobOrInterne;
}

const cTplValGesInit< cSetOrientationInterne > & cSectionInconnues::GlobOrInterne()const 
{
   return mGlobOrInterne;
}


std::list< cPoseCameraInc > & cSectionInconnues::PoseCameraInc()
{
   return mPoseCameraInc;
}

const std::list< cPoseCameraInc > & cSectionInconnues::PoseCameraInc()const 
{
   return mPoseCameraInc;
}


std::list< cGroupeDePose > & cSectionInconnues::GroupeDePose()
{
   return mGroupeDePose;
}

const std::list< cGroupeDePose > & cSectionInconnues::GroupeDePose()const 
{
   return mGroupeDePose;
}


std::list< cSurfParamInc > & cSectionInconnues::SurfParamInc()
{
   return mSurfParamInc;
}

const std::list< cSurfParamInc > & cSectionInconnues::SurfParamInc()const 
{
   return mSurfParamInc;
}


std::list< cPointFlottantInc > & cSectionInconnues::PointFlottantInc()
{
   return mPointFlottantInc;
}

const std::list< cPointFlottantInc > & cSectionInconnues::PointFlottantInc()const 
{
   return mPointFlottantInc;
}

cElXMLTree * ToXMLTree(const cSectionInconnues & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"SectionInconnues",eXMLBranche);
   if (anObj.SeuilAutomFE().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SeuilAutomFE"),anObj.SeuilAutomFE().Val())->ReTagThis("SeuilAutomFE"));
   if (anObj.AutoriseToujoursUneSeuleLiaison().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("AutoriseToujoursUneSeuleLiaison"),anObj.AutoriseToujoursUneSeuleLiaison().Val())->ReTagThis("AutoriseToujoursUneSeuleLiaison"));
   if (anObj.MapMaskHom().IsInit())
      aRes->AddFils(ToXMLTree(anObj.MapMaskHom().Val())->ReTagThis("MapMaskHom"));
   if (anObj.SauvePMoyenOnlyWithMasq().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SauvePMoyenOnlyWithMasq"),anObj.SauvePMoyenOnlyWithMasq().Val())->ReTagThis("SauvePMoyenOnlyWithMasq"));
  for
  (       std::list< cCalibrationCameraInc >::const_iterator it=anObj.CalibrationCameraInc().begin();
      it !=anObj.CalibrationCameraInc().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("CalibrationCameraInc"));
   if (anObj.SeuilL1EstimMatrEss().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SeuilL1EstimMatrEss"),anObj.SeuilL1EstimMatrEss().Val())->ReTagThis("SeuilL1EstimMatrEss"));
   if (anObj.GlobOrInterne().IsInit())
      aRes->AddFils(ToXMLTree(anObj.GlobOrInterne().Val())->ReTagThis("GlobOrInterne"));
  for
  (       std::list< cPoseCameraInc >::const_iterator it=anObj.PoseCameraInc().begin();
      it !=anObj.PoseCameraInc().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("PoseCameraInc"));
  for
  (       std::list< cGroupeDePose >::const_iterator it=anObj.GroupeDePose().begin();
      it !=anObj.GroupeDePose().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("GroupeDePose"));
  for
  (       std::list< cSurfParamInc >::const_iterator it=anObj.SurfParamInc().begin();
      it !=anObj.SurfParamInc().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("SurfParamInc"));
  for
  (       std::list< cPointFlottantInc >::const_iterator it=anObj.PointFlottantInc().begin();
      it !=anObj.PointFlottantInc().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("PointFlottantInc"));
  return aRes;
}

void xml_init(cSectionInconnues & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.SeuilAutomFE(),aTree->Get("SeuilAutomFE",1),double(-1)); //tototo 

   xml_init(anObj.AutoriseToujoursUneSeuleLiaison(),aTree->Get("AutoriseToujoursUneSeuleLiaison",1),bool(false)); //tototo 

   xml_init(anObj.MapMaskHom(),aTree->Get("MapMaskHom",1)); //tototo 

   xml_init(anObj.SauvePMoyenOnlyWithMasq(),aTree->Get("SauvePMoyenOnlyWithMasq",1),bool(true)); //tototo 

   xml_init(anObj.CalibrationCameraInc(),aTree->GetAll("CalibrationCameraInc",false,1));

   xml_init(anObj.SeuilL1EstimMatrEss(),aTree->Get("SeuilL1EstimMatrEss",1),int(150)); //tototo 

   xml_init(anObj.GlobOrInterne(),aTree->Get("GlobOrInterne",1)); //tototo 

   xml_init(anObj.PoseCameraInc(),aTree->GetAll("PoseCameraInc",false,1));

   xml_init(anObj.GroupeDePose(),aTree->GetAll("GroupeDePose",false,1));

   xml_init(anObj.SurfParamInc(),aTree->GetAll("SurfParamInc",false,1));

   xml_init(anObj.PointFlottantInc(),aTree->GetAll("PointFlottantInc",false,1));
}


double & cTimeLinkage::DeltaMax()
{
   return mDeltaMax;
}

const double & cTimeLinkage::DeltaMax()const 
{
   return mDeltaMax;
}

cElXMLTree * ToXMLTree(const cTimeLinkage & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"TimeLinkage",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("DeltaMax"),anObj.DeltaMax())->ReTagThis("DeltaMax"));
  return aRes;
}

void xml_init(cTimeLinkage & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.DeltaMax(),aTree->Get("DeltaMax",1)); //tototo 
}


cTplValGesInit< std::string > & cSectionChantier::FileSauvParam()
{
   return mFileSauvParam;
}

const cTplValGesInit< std::string > & cSectionChantier::FileSauvParam()const 
{
   return mFileSauvParam;
}


cTplValGesInit< bool > & cSectionChantier::GenereErreurOnContraineCam()
{
   return mGenereErreurOnContraineCam;
}

const cTplValGesInit< bool > & cSectionChantier::GenereErreurOnContraineCam()const 
{
   return mGenereErreurOnContraineCam;
}


cTplValGesInit< double > & cSectionChantier::ProfSceneChantier()
{
   return mProfSceneChantier;
}

const cTplValGesInit< double > & cSectionChantier::ProfSceneChantier()const 
{
   return mProfSceneChantier;
}


cTplValGesInit< std::string > & cSectionChantier::DirectoryChantier()
{
   return mDirectoryChantier;
}

const cTplValGesInit< std::string > & cSectionChantier::DirectoryChantier()const 
{
   return mDirectoryChantier;
}


cTplValGesInit< string > & cSectionChantier::FileChantierNameDescripteur()
{
   return mFileChantierNameDescripteur;
}

const cTplValGesInit< string > & cSectionChantier::FileChantierNameDescripteur()const 
{
   return mFileChantierNameDescripteur;
}


cTplValGesInit< std::string > & cSectionChantier::NameParamEtal()
{
   return mNameParamEtal;
}

const cTplValGesInit< std::string > & cSectionChantier::NameParamEtal()const 
{
   return mNameParamEtal;
}


cTplValGesInit< std::string > & cSectionChantier::PatternTracePose()
{
   return mPatternTracePose;
}

const cTplValGesInit< std::string > & cSectionChantier::PatternTracePose()const 
{
   return mPatternTracePose;
}


cTplValGesInit< bool > & cSectionChantier::TraceGimbalLock()
{
   return mTraceGimbalLock;
}

const cTplValGesInit< bool > & cSectionChantier::TraceGimbalLock()const 
{
   return mTraceGimbalLock;
}


cTplValGesInit< double > & cSectionChantier::MaxDistErrorPtsTerr()
{
   return mMaxDistErrorPtsTerr;
}

const cTplValGesInit< double > & cSectionChantier::MaxDistErrorPtsTerr()const 
{
   return mMaxDistErrorPtsTerr;
}


cTplValGesInit< double > & cSectionChantier::MaxDistWarnPtsTerr()
{
   return mMaxDistWarnPtsTerr;
}

const cTplValGesInit< double > & cSectionChantier::MaxDistWarnPtsTerr()const 
{
   return mMaxDistWarnPtsTerr;
}


cTplValGesInit< cShowPbLiaison > & cSectionChantier::DefPbLiaison()
{
   return mDefPbLiaison;
}

const cTplValGesInit< cShowPbLiaison > & cSectionChantier::DefPbLiaison()const 
{
   return mDefPbLiaison;
}


cTplValGesInit< bool > & cSectionChantier::DoCompensation()
{
   return mDoCompensation;
}

const cTplValGesInit< bool > & cSectionChantier::DoCompensation()const 
{
   return mDoCompensation;
}


double & cSectionChantier::DeltaMax()
{
   return TimeLinkage().Val().DeltaMax();
}

const double & cSectionChantier::DeltaMax()const 
{
   return TimeLinkage().Val().DeltaMax();
}


cTplValGesInit< cTimeLinkage > & cSectionChantier::TimeLinkage()
{
   return mTimeLinkage;
}

const cTplValGesInit< cTimeLinkage > & cSectionChantier::TimeLinkage()const 
{
   return mTimeLinkage;
}


cTplValGesInit< bool > & cSectionChantier::DebugPbCondFaisceau()
{
   return mDebugPbCondFaisceau;
}

const cTplValGesInit< bool > & cSectionChantier::DebugPbCondFaisceau()const 
{
   return mDebugPbCondFaisceau;
}


cTplValGesInit< std::string > & cSectionChantier::SauvAutom()
{
   return mSauvAutom;
}

const cTplValGesInit< std::string > & cSectionChantier::SauvAutom()const 
{
   return mSauvAutom;
}


cTplValGesInit< bool > & cSectionChantier::SauvAutomBasic()
{
   return mSauvAutomBasic;
}

const cTplValGesInit< bool > & cSectionChantier::SauvAutomBasic()const 
{
   return mSauvAutomBasic;
}


cTplValGesInit< double > & cSectionChantier::ThresholdWarnPointsBehind()
{
   return mThresholdWarnPointsBehind;
}

const cTplValGesInit< double > & cSectionChantier::ThresholdWarnPointsBehind()const 
{
   return mThresholdWarnPointsBehind;
}

cElXMLTree * ToXMLTree(const cSectionChantier & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"SectionChantier",eXMLBranche);
   if (anObj.FileSauvParam().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("FileSauvParam"),anObj.FileSauvParam().Val())->ReTagThis("FileSauvParam"));
   if (anObj.GenereErreurOnContraineCam().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("GenereErreurOnContraineCam"),anObj.GenereErreurOnContraineCam().Val())->ReTagThis("GenereErreurOnContraineCam"));
   if (anObj.ProfSceneChantier().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ProfSceneChantier"),anObj.ProfSceneChantier().Val())->ReTagThis("ProfSceneChantier"));
   if (anObj.DirectoryChantier().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DirectoryChantier"),anObj.DirectoryChantier().Val())->ReTagThis("DirectoryChantier"));
   if (anObj.FileChantierNameDescripteur().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("FileChantierNameDescripteur"),anObj.FileChantierNameDescripteur().Val())->ReTagThis("FileChantierNameDescripteur"));
   if (anObj.NameParamEtal().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NameParamEtal"),anObj.NameParamEtal().Val())->ReTagThis("NameParamEtal"));
   if (anObj.PatternTracePose().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PatternTracePose"),anObj.PatternTracePose().Val())->ReTagThis("PatternTracePose"));
   if (anObj.TraceGimbalLock().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("TraceGimbalLock"),anObj.TraceGimbalLock().Val())->ReTagThis("TraceGimbalLock"));
   if (anObj.MaxDistErrorPtsTerr().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("MaxDistErrorPtsTerr"),anObj.MaxDistErrorPtsTerr().Val())->ReTagThis("MaxDistErrorPtsTerr"));
   if (anObj.MaxDistWarnPtsTerr().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("MaxDistWarnPtsTerr"),anObj.MaxDistWarnPtsTerr().Val())->ReTagThis("MaxDistWarnPtsTerr"));
   if (anObj.DefPbLiaison().IsInit())
      aRes->AddFils(ToXMLTree(anObj.DefPbLiaison().Val())->ReTagThis("DefPbLiaison"));
   if (anObj.DoCompensation().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DoCompensation"),anObj.DoCompensation().Val())->ReTagThis("DoCompensation"));
   if (anObj.TimeLinkage().IsInit())
      aRes->AddFils(ToXMLTree(anObj.TimeLinkage().Val())->ReTagThis("TimeLinkage"));
   if (anObj.DebugPbCondFaisceau().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DebugPbCondFaisceau"),anObj.DebugPbCondFaisceau().Val())->ReTagThis("DebugPbCondFaisceau"));
   if (anObj.SauvAutom().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SauvAutom"),anObj.SauvAutom().Val())->ReTagThis("SauvAutom"));
   if (anObj.SauvAutomBasic().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SauvAutomBasic"),anObj.SauvAutomBasic().Val())->ReTagThis("SauvAutomBasic"));
   if (anObj.ThresholdWarnPointsBehind().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ThresholdWarnPointsBehind"),anObj.ThresholdWarnPointsBehind().Val())->ReTagThis("ThresholdWarnPointsBehind"));
  return aRes;
}

void xml_init(cSectionChantier & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.FileSauvParam(),aTree->Get("FileSauvParam",1)); //tototo 

   xml_init(anObj.GenereErreurOnContraineCam(),aTree->Get("GenereErreurOnContraineCam",1),bool(true)); //tototo 

   xml_init(anObj.ProfSceneChantier(),aTree->Get("ProfSceneChantier",1),double(10.0)); //tototo 

   xml_init(anObj.DirectoryChantier(),aTree->Get("DirectoryChantier",1),std::string("")); //tototo 

   xml_init(anObj.FileChantierNameDescripteur(),aTree->Get("FileChantierNameDescripteur",1)); //tototo 

   xml_init(anObj.NameParamEtal(),aTree->Get("NameParamEtal",1)); //tototo 

   xml_init(anObj.PatternTracePose(),aTree->Get("PatternTracePose",1)); //tototo 

   xml_init(anObj.TraceGimbalLock(),aTree->Get("TraceGimbalLock",1),bool(true)); //tototo 

   xml_init(anObj.MaxDistErrorPtsTerr(),aTree->Get("MaxDistErrorPtsTerr",1),double(1e50)); //tototo 

   xml_init(anObj.MaxDistWarnPtsTerr(),aTree->Get("MaxDistWarnPtsTerr",1),double(1e30)); //tototo 

   xml_init(anObj.DefPbLiaison(),aTree->Get("DefPbLiaison",1)); //tototo 

   xml_init(anObj.DoCompensation(),aTree->Get("DoCompensation",1),bool(true)); //tototo 

   xml_init(anObj.TimeLinkage(),aTree->Get("TimeLinkage",1)); //tototo 

   xml_init(anObj.DebugPbCondFaisceau(),aTree->Get("DebugPbCondFaisceau",1),bool(false)); //tototo 

   xml_init(anObj.SauvAutom(),aTree->Get("SauvAutom",1)); //tototo 

   xml_init(anObj.SauvAutomBasic(),aTree->Get("SauvAutomBasic",1),bool(false)); //tototo 

   xml_init(anObj.ThresholdWarnPointsBehind(),aTree->Get("ThresholdWarnPointsBehind",1),double(0.01)); //tototo 
}


cTplValGesInit< bool > & cSectionSolveur::AllMatSym()
{
   return mAllMatSym;
}

const cTplValGesInit< bool > & cSectionSolveur::AllMatSym()const 
{
   return mAllMatSym;
}


eModeSolveurEq & cSectionSolveur::ModeResolution()
{
   return mModeResolution;
}

const eModeSolveurEq & cSectionSolveur::ModeResolution()const 
{
   return mModeResolution;
}


cTplValGesInit< eControleDescDic > & cSectionSolveur::ModeControleDescDic()
{
   return mModeControleDescDic;
}

const cTplValGesInit< eControleDescDic > & cSectionSolveur::ModeControleDescDic()const 
{
   return mModeControleDescDic;
}


cTplValGesInit< int > & cSectionSolveur::SeuilBas_CDD()
{
   return mSeuilBas_CDD;
}

const cTplValGesInit< int > & cSectionSolveur::SeuilBas_CDD()const 
{
   return mSeuilBas_CDD;
}


cTplValGesInit< int > & cSectionSolveur::SeuilHaut_CDD()
{
   return mSeuilHaut_CDD;
}

const cTplValGesInit< int > & cSectionSolveur::SeuilHaut_CDD()const 
{
   return mSeuilHaut_CDD;
}


cTplValGesInit< bool > & cSectionSolveur::InhibeAMD()
{
   return mInhibeAMD;
}

const cTplValGesInit< bool > & cSectionSolveur::InhibeAMD()const 
{
   return mInhibeAMD;
}


cTplValGesInit< bool > & cSectionSolveur::AMDSpecInterne()
{
   return mAMDSpecInterne;
}

const cTplValGesInit< bool > & cSectionSolveur::AMDSpecInterne()const 
{
   return mAMDSpecInterne;
}


cTplValGesInit< bool > & cSectionSolveur::ShowCholesky()
{
   return mShowCholesky;
}

const cTplValGesInit< bool > & cSectionSolveur::ShowCholesky()const 
{
   return mShowCholesky;
}


cTplValGesInit< bool > & cSectionSolveur::TestPermutVar()
{
   return mTestPermutVar;
}

const cTplValGesInit< bool > & cSectionSolveur::TestPermutVar()const 
{
   return mTestPermutVar;
}


cTplValGesInit< bool > & cSectionSolveur::ShowPermutVar()
{
   return mShowPermutVar;
}

const cTplValGesInit< bool > & cSectionSolveur::ShowPermutVar()const 
{
   return mShowPermutVar;
}


cTplValGesInit< bool > & cSectionSolveur::PermutIndex()
{
   return mPermutIndex;
}

const cTplValGesInit< bool > & cSectionSolveur::PermutIndex()const 
{
   return mPermutIndex;
}


cTplValGesInit< bool > & cSectionSolveur::NormaliseEqSc()
{
   return mNormaliseEqSc;
}

const cTplValGesInit< bool > & cSectionSolveur::NormaliseEqSc()const 
{
   return mNormaliseEqSc;
}


cTplValGesInit< bool > & cSectionSolveur::NormaliseEqTr()
{
   return mNormaliseEqTr;
}

const cTplValGesInit< bool > & cSectionSolveur::NormaliseEqTr()const 
{
   return mNormaliseEqTr;
}


cTplValGesInit< double > & cSectionSolveur::LimBsHProj()
{
   return mLimBsHProj;
}

const cTplValGesInit< double > & cSectionSolveur::LimBsHProj()const 
{
   return mLimBsHProj;
}


cTplValGesInit< double > & cSectionSolveur::LimBsHRefut()
{
   return mLimBsHRefut;
}

const cTplValGesInit< double > & cSectionSolveur::LimBsHRefut()const 
{
   return mLimBsHRefut;
}


cTplValGesInit< double > & cSectionSolveur::LimModeGL()
{
   return mLimModeGL;
}

const cTplValGesInit< double > & cSectionSolveur::LimModeGL()const 
{
   return mLimModeGL;
}


cTplValGesInit< bool > & cSectionSolveur::GridOptimKnownDist()
{
   return mGridOptimKnownDist;
}

const cTplValGesInit< bool > & cSectionSolveur::GridOptimKnownDist()const 
{
   return mGridOptimKnownDist;
}


cTplValGesInit< cSectionLevenbergMarkard > & cSectionSolveur::SLMGlob()
{
   return mSLMGlob;
}

const cTplValGesInit< cSectionLevenbergMarkard > & cSectionSolveur::SLMGlob()const 
{
   return mSLMGlob;
}


cTplValGesInit< double > & cSectionSolveur::MultSLMGlob()
{
   return mMultSLMGlob;
}

const cTplValGesInit< double > & cSectionSolveur::MultSLMGlob()const 
{
   return mMultSLMGlob;
}


cTplValGesInit< cElRegex_Ptr > & cSectionSolveur::Im2Aff()
{
   return mIm2Aff;
}

const cTplValGesInit< cElRegex_Ptr > & cSectionSolveur::Im2Aff()const 
{
   return mIm2Aff;
}

cElXMLTree * ToXMLTree(const cSectionSolveur & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"SectionSolveur",eXMLBranche);
   if (anObj.AllMatSym().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("AllMatSym"),anObj.AllMatSym().Val())->ReTagThis("AllMatSym"));
   aRes->AddFils(::ToXMLTree(std::string("ModeResolution"),anObj.ModeResolution())->ReTagThis("ModeResolution"));
   if (anObj.ModeControleDescDic().IsInit())
      aRes->AddFils(ToXMLTree(std::string("ModeControleDescDic"),anObj.ModeControleDescDic().Val())->ReTagThis("ModeControleDescDic"));
   if (anObj.SeuilBas_CDD().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SeuilBas_CDD"),anObj.SeuilBas_CDD().Val())->ReTagThis("SeuilBas_CDD"));
   if (anObj.SeuilHaut_CDD().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SeuilHaut_CDD"),anObj.SeuilHaut_CDD().Val())->ReTagThis("SeuilHaut_CDD"));
   if (anObj.InhibeAMD().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("InhibeAMD"),anObj.InhibeAMD().Val())->ReTagThis("InhibeAMD"));
   if (anObj.AMDSpecInterne().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("AMDSpecInterne"),anObj.AMDSpecInterne().Val())->ReTagThis("AMDSpecInterne"));
   if (anObj.ShowCholesky().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ShowCholesky"),anObj.ShowCholesky().Val())->ReTagThis("ShowCholesky"));
   if (anObj.TestPermutVar().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("TestPermutVar"),anObj.TestPermutVar().Val())->ReTagThis("TestPermutVar"));
   if (anObj.ShowPermutVar().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ShowPermutVar"),anObj.ShowPermutVar().Val())->ReTagThis("ShowPermutVar"));
   if (anObj.PermutIndex().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PermutIndex"),anObj.PermutIndex().Val())->ReTagThis("PermutIndex"));
   if (anObj.NormaliseEqSc().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NormaliseEqSc"),anObj.NormaliseEqSc().Val())->ReTagThis("NormaliseEqSc"));
   if (anObj.NormaliseEqTr().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NormaliseEqTr"),anObj.NormaliseEqTr().Val())->ReTagThis("NormaliseEqTr"));
   if (anObj.LimBsHProj().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("LimBsHProj"),anObj.LimBsHProj().Val())->ReTagThis("LimBsHProj"));
   if (anObj.LimBsHRefut().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("LimBsHRefut"),anObj.LimBsHRefut().Val())->ReTagThis("LimBsHRefut"));
   if (anObj.LimModeGL().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("LimModeGL"),anObj.LimModeGL().Val())->ReTagThis("LimModeGL"));
   if (anObj.GridOptimKnownDist().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("GridOptimKnownDist"),anObj.GridOptimKnownDist().Val())->ReTagThis("GridOptimKnownDist"));
   if (anObj.SLMGlob().IsInit())
      aRes->AddFils(ToXMLTree(anObj.SLMGlob().Val())->ReTagThis("SLMGlob"));
   if (anObj.MultSLMGlob().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("MultSLMGlob"),anObj.MultSLMGlob().Val())->ReTagThis("MultSLMGlob"));
   if (anObj.Im2Aff().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Im2Aff"),anObj.Im2Aff().Val())->ReTagThis("Im2Aff"));
  return aRes;
}

void xml_init(cSectionSolveur & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.AllMatSym(),aTree->Get("AllMatSym",1),bool(true)); //tototo 

   xml_init(anObj.ModeResolution(),aTree->Get("ModeResolution",1)); //tototo 

   xml_init(anObj.ModeControleDescDic(),aTree->Get("ModeControleDescDic",1),eControleDescDic(eCDD_Jamais)); //tototo 

   xml_init(anObj.SeuilBas_CDD(),aTree->Get("SeuilBas_CDD",1),int(4)); //tototo 

   xml_init(anObj.SeuilHaut_CDD(),aTree->Get("SeuilHaut_CDD",1),int(10)); //tototo 

   xml_init(anObj.InhibeAMD(),aTree->Get("InhibeAMD",1),bool(false)); //tototo 

   xml_init(anObj.AMDSpecInterne(),aTree->Get("AMDSpecInterne",1),bool(false)); //tototo 

   xml_init(anObj.ShowCholesky(),aTree->Get("ShowCholesky",1),bool(false)); //tototo 

   xml_init(anObj.TestPermutVar(),aTree->Get("TestPermutVar",1),bool(false)); //tototo 

   xml_init(anObj.ShowPermutVar(),aTree->Get("ShowPermutVar",1),bool(false)); //tototo 

   xml_init(anObj.PermutIndex(),aTree->Get("PermutIndex",1),bool(true)); //tototo 

   xml_init(anObj.NormaliseEqSc(),aTree->Get("NormaliseEqSc",1),bool(true)); //tototo 

   xml_init(anObj.NormaliseEqTr(),aTree->Get("NormaliseEqTr",1),bool(true)); //tototo 

   xml_init(anObj.LimBsHProj(),aTree->Get("LimBsHProj",1),double(0.1)); //tototo 

   xml_init(anObj.LimBsHRefut(),aTree->Get("LimBsHRefut",1),double(1e-6)); //tototo 

   xml_init(anObj.LimModeGL(),aTree->Get("LimModeGL",1),double(0.3)); //tototo 

   xml_init(anObj.GridOptimKnownDist(),aTree->Get("GridOptimKnownDist",1),bool(false)); //tototo 

   xml_init(anObj.SLMGlob(),aTree->Get("SLMGlob",1)); //tototo 

   xml_init(anObj.MultSLMGlob(),aTree->Get("MultSLMGlob",1)); //tototo 

   xml_init(anObj.Im2Aff(),aTree->Get("Im2Aff",1)); //tototo 
}


std::vector<int> & cPose2Init::ProfMin()
{
   return mProfMin;
}

const std::vector<int> & cPose2Init::ProfMin()const 
{
   return mProfMin;
}


cTplValGesInit< bool > & cPose2Init::Show()
{
   return mShow;
}

const cTplValGesInit< bool > & cPose2Init::Show()const 
{
   return mShow;
}


cTplValGesInit< int > & cPose2Init::StepComplemAuto()
{
   return mStepComplemAuto;
}

const cTplValGesInit< int > & cPose2Init::StepComplemAuto()const 
{
   return mStepComplemAuto;
}

cElXMLTree * ToXMLTree(const cPose2Init & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"Pose2Init",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("ProfMin"),anObj.ProfMin())->ReTagThis("ProfMin"));
   if (anObj.Show().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Show"),anObj.Show().Val())->ReTagThis("Show"));
   if (anObj.StepComplemAuto().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("StepComplemAuto"),anObj.StepComplemAuto().Val())->ReTagThis("StepComplemAuto"));
  return aRes;
}

void xml_init(cPose2Init & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.ProfMin(),aTree->Get("ProfMin",1)); //tototo 

   xml_init(anObj.Show(),aTree->Get("Show",1),bool(false)); //tototo 

   xml_init(anObj.StepComplemAuto(),aTree->Get("StepComplemAuto",1),int(0)); //tototo 
}


std::string & cSetRayMaxUtileCalib::Name()
{
   return mName;
}

const std::string & cSetRayMaxUtileCalib::Name()const 
{
   return mName;
}


double & cSetRayMaxUtileCalib::Ray()
{
   return mRay;
}

const double & cSetRayMaxUtileCalib::Ray()const 
{
   return mRay;
}


cTplValGesInit< bool > & cSetRayMaxUtileCalib::IsRelatifDiag()
{
   return mIsRelatifDiag;
}

const cTplValGesInit< bool > & cSetRayMaxUtileCalib::IsRelatifDiag()const 
{
   return mIsRelatifDiag;
}


cTplValGesInit< bool > & cSetRayMaxUtileCalib::ApplyOnlyFE()
{
   return mApplyOnlyFE;
}

const cTplValGesInit< bool > & cSetRayMaxUtileCalib::ApplyOnlyFE()const 
{
   return mApplyOnlyFE;
}

cElXMLTree * ToXMLTree(const cSetRayMaxUtileCalib & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"SetRayMaxUtileCalib",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Name"),anObj.Name())->ReTagThis("Name"));
   aRes->AddFils(::ToXMLTree(std::string("Ray"),anObj.Ray())->ReTagThis("Ray"));
   if (anObj.IsRelatifDiag().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("IsRelatifDiag"),anObj.IsRelatifDiag().Val())->ReTagThis("IsRelatifDiag"));
   if (anObj.ApplyOnlyFE().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ApplyOnlyFE"),anObj.ApplyOnlyFE().Val())->ReTagThis("ApplyOnlyFE"));
  return aRes;
}

void xml_init(cSetRayMaxUtileCalib & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.Name(),aTree->Get("Name",1)); //tototo 

   xml_init(anObj.Ray(),aTree->Get("Ray",1)); //tototo 

   xml_init(anObj.IsRelatifDiag(),aTree->Get("IsRelatifDiag",1),bool(false)); //tototo 

   xml_init(anObj.ApplyOnlyFE(),aTree->Get("ApplyOnlyFE",1),bool(false)); //tototo 
}


cTplValGesInit< std::string > & cBascOnCentre::PoseCentrale()
{
   return mPoseCentrale;
}

const cTplValGesInit< std::string > & cBascOnCentre::PoseCentrale()const 
{
   return mPoseCentrale;
}

cElXMLTree * ToXMLTree(const cBascOnCentre & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"BascOnCentre",eXMLBranche);
   if (anObj.PoseCentrale().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PoseCentrale"),anObj.PoseCentrale().Val())->ReTagThis("PoseCentrale"));
  return aRes;
}

void xml_init(cBascOnCentre & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.PoseCentrale(),aTree->Get("PoseCentrale",1)); //tototo 
}


std::string & cBascOnAppuis::NameRef()
{
   return mNameRef;
}

const std::string & cBascOnAppuis::NameRef()const 
{
   return mNameRef;
}

cElXMLTree * ToXMLTree(const cBascOnAppuis & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"BascOnAppuis",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("NameRef"),anObj.NameRef())->ReTagThis("NameRef"));
  return aRes;
}

void xml_init(cBascOnAppuis & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.NameRef(),aTree->Get("NameRef",1)); //tototo 
}


cTplValGesInit< std::string > & cBasculeOnPoints::PoseCentrale()
{
   return BascOnCentre().Val().PoseCentrale();
}

const cTplValGesInit< std::string > & cBasculeOnPoints::PoseCentrale()const 
{
   return BascOnCentre().Val().PoseCentrale();
}


cTplValGesInit< cBascOnCentre > & cBasculeOnPoints::BascOnCentre()
{
   return mBascOnCentre;
}

const cTplValGesInit< cBascOnCentre > & cBasculeOnPoints::BascOnCentre()const 
{
   return mBascOnCentre;
}


std::string & cBasculeOnPoints::NameRef()
{
   return BascOnAppuis().Val().NameRef();
}

const std::string & cBasculeOnPoints::NameRef()const 
{
   return BascOnAppuis().Val().NameRef();
}


cTplValGesInit< cBascOnAppuis > & cBasculeOnPoints::BascOnAppuis()
{
   return mBascOnAppuis;
}

const cTplValGesInit< cBascOnAppuis > & cBasculeOnPoints::BascOnAppuis()const 
{
   return mBascOnAppuis;
}


cTplValGesInit< bool > & cBasculeOnPoints::ModeL2()
{
   return mModeL2;
}

const cTplValGesInit< bool > & cBasculeOnPoints::ModeL2()const 
{
   return mModeL2;
}

cElXMLTree * ToXMLTree(const cBasculeOnPoints & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"BasculeOnPoints",eXMLBranche);
   if (anObj.BascOnCentre().IsInit())
      aRes->AddFils(ToXMLTree(anObj.BascOnCentre().Val())->ReTagThis("BascOnCentre"));
   if (anObj.BascOnAppuis().IsInit())
      aRes->AddFils(ToXMLTree(anObj.BascOnAppuis().Val())->ReTagThis("BascOnAppuis"));
   if (anObj.ModeL2().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ModeL2"),anObj.ModeL2().Val())->ReTagThis("ModeL2"));
  return aRes;
}

void xml_init(cBasculeOnPoints & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.BascOnCentre(),aTree->Get("BascOnCentre",1)); //tototo 

   xml_init(anObj.BascOnAppuis(),aTree->Get("BascOnAppuis",1)); //tototo 

   xml_init(anObj.ModeL2(),aTree->Get("ModeL2",1),bool(true)); //tototo 
}


cTplValGesInit< double > & cOrientInPlane::DistFixEch()
{
   return mDistFixEch;
}

const cTplValGesInit< double > & cOrientInPlane::DistFixEch()const 
{
   return mDistFixEch;
}


std::string & cOrientInPlane::FileMesures()
{
   return mFileMesures;
}

const std::string & cOrientInPlane::FileMesures()const 
{
   return mFileMesures;
}


cTplValGesInit< std::string > & cOrientInPlane::AlignOn()
{
   return mAlignOn;
}

const cTplValGesInit< std::string > & cOrientInPlane::AlignOn()const 
{
   return mAlignOn;
}

cElXMLTree * ToXMLTree(const cOrientInPlane & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"OrientInPlane",eXMLBranche);
   if (anObj.DistFixEch().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DistFixEch"),anObj.DistFixEch().Val())->ReTagThis("DistFixEch"));
   aRes->AddFils(::ToXMLTree(std::string("FileMesures"),anObj.FileMesures())->ReTagThis("FileMesures"));
   if (anObj.AlignOn().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("AlignOn"),anObj.AlignOn().Val())->ReTagThis("AlignOn"));
  return aRes;
}

void xml_init(cOrientInPlane & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.DistFixEch(),aTree->Get("DistFixEch",1)); //tototo 

   xml_init(anObj.FileMesures(),aTree->Get("FileMesures",1)); //tototo 

   xml_init(anObj.AlignOn(),aTree->Get("AlignOn",1),std::string("ki")); //tototo 
}


cParamEstimPlan & cBasculeLiaisonOnPlan::EstimPl()
{
   return mEstimPl;
}

const cParamEstimPlan & cBasculeLiaisonOnPlan::EstimPl()const 
{
   return mEstimPl;
}


cTplValGesInit< double > & cBasculeLiaisonOnPlan::DistFixEch()
{
   return OrientInPlane().Val().DistFixEch();
}

const cTplValGesInit< double > & cBasculeLiaisonOnPlan::DistFixEch()const 
{
   return OrientInPlane().Val().DistFixEch();
}


std::string & cBasculeLiaisonOnPlan::FileMesures()
{
   return OrientInPlane().Val().FileMesures();
}

const std::string & cBasculeLiaisonOnPlan::FileMesures()const 
{
   return OrientInPlane().Val().FileMesures();
}


cTplValGesInit< std::string > & cBasculeLiaisonOnPlan::AlignOn()
{
   return OrientInPlane().Val().AlignOn();
}

const cTplValGesInit< std::string > & cBasculeLiaisonOnPlan::AlignOn()const 
{
   return OrientInPlane().Val().AlignOn();
}


cTplValGesInit< cOrientInPlane > & cBasculeLiaisonOnPlan::OrientInPlane()
{
   return mOrientInPlane;
}

const cTplValGesInit< cOrientInPlane > & cBasculeLiaisonOnPlan::OrientInPlane()const 
{
   return mOrientInPlane;
}

cElXMLTree * ToXMLTree(const cBasculeLiaisonOnPlan & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"BasculeLiaisonOnPlan",eXMLBranche);
   aRes->AddFils(ToXMLTree(anObj.EstimPl())->ReTagThis("EstimPl"));
   if (anObj.OrientInPlane().IsInit())
      aRes->AddFils(ToXMLTree(anObj.OrientInPlane().Val())->ReTagThis("OrientInPlane"));
  return aRes;
}

void xml_init(cBasculeLiaisonOnPlan & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.EstimPl(),aTree->Get("EstimPl",1)); //tototo 

   xml_init(anObj.OrientInPlane(),aTree->Get("OrientInPlane",1)); //tototo 
}


cTplValGesInit< std::string > & cModeBascule::PoseCentrale()
{
   return BasculeOnPoints().Val().BascOnCentre().Val().PoseCentrale();
}

const cTplValGesInit< std::string > & cModeBascule::PoseCentrale()const 
{
   return BasculeOnPoints().Val().BascOnCentre().Val().PoseCentrale();
}


cTplValGesInit< cBascOnCentre > & cModeBascule::BascOnCentre()
{
   return BasculeOnPoints().Val().BascOnCentre();
}

const cTplValGesInit< cBascOnCentre > & cModeBascule::BascOnCentre()const 
{
   return BasculeOnPoints().Val().BascOnCentre();
}


std::string & cModeBascule::NameRef()
{
   return BasculeOnPoints().Val().BascOnAppuis().Val().NameRef();
}

const std::string & cModeBascule::NameRef()const 
{
   return BasculeOnPoints().Val().BascOnAppuis().Val().NameRef();
}


cTplValGesInit< cBascOnAppuis > & cModeBascule::BascOnAppuis()
{
   return BasculeOnPoints().Val().BascOnAppuis();
}

const cTplValGesInit< cBascOnAppuis > & cModeBascule::BascOnAppuis()const 
{
   return BasculeOnPoints().Val().BascOnAppuis();
}


cTplValGesInit< bool > & cModeBascule::ModeL2()
{
   return BasculeOnPoints().Val().ModeL2();
}

const cTplValGesInit< bool > & cModeBascule::ModeL2()const 
{
   return BasculeOnPoints().Val().ModeL2();
}


cTplValGesInit< cBasculeOnPoints > & cModeBascule::BasculeOnPoints()
{
   return mBasculeOnPoints;
}

const cTplValGesInit< cBasculeOnPoints > & cModeBascule::BasculeOnPoints()const 
{
   return mBasculeOnPoints;
}


cParamEstimPlan & cModeBascule::EstimPl()
{
   return BasculeLiaisonOnPlan().Val().EstimPl();
}

const cParamEstimPlan & cModeBascule::EstimPl()const 
{
   return BasculeLiaisonOnPlan().Val().EstimPl();
}


cTplValGesInit< double > & cModeBascule::DistFixEch()
{
   return BasculeLiaisonOnPlan().Val().OrientInPlane().Val().DistFixEch();
}

const cTplValGesInit< double > & cModeBascule::DistFixEch()const 
{
   return BasculeLiaisonOnPlan().Val().OrientInPlane().Val().DistFixEch();
}


std::string & cModeBascule::FileMesures()
{
   return BasculeLiaisonOnPlan().Val().OrientInPlane().Val().FileMesures();
}

const std::string & cModeBascule::FileMesures()const 
{
   return BasculeLiaisonOnPlan().Val().OrientInPlane().Val().FileMesures();
}


cTplValGesInit< std::string > & cModeBascule::AlignOn()
{
   return BasculeLiaisonOnPlan().Val().OrientInPlane().Val().AlignOn();
}

const cTplValGesInit< std::string > & cModeBascule::AlignOn()const 
{
   return BasculeLiaisonOnPlan().Val().OrientInPlane().Val().AlignOn();
}


cTplValGesInit< cOrientInPlane > & cModeBascule::OrientInPlane()
{
   return BasculeLiaisonOnPlan().Val().OrientInPlane();
}

const cTplValGesInit< cOrientInPlane > & cModeBascule::OrientInPlane()const 
{
   return BasculeLiaisonOnPlan().Val().OrientInPlane();
}


cTplValGesInit< cBasculeLiaisonOnPlan > & cModeBascule::BasculeLiaisonOnPlan()
{
   return mBasculeLiaisonOnPlan;
}

const cTplValGesInit< cBasculeLiaisonOnPlan > & cModeBascule::BasculeLiaisonOnPlan()const 
{
   return mBasculeLiaisonOnPlan;
}

cElXMLTree * ToXMLTree(const cModeBascule & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ModeBascule",eXMLBranche);
   if (anObj.BasculeOnPoints().IsInit())
      aRes->AddFils(ToXMLTree(anObj.BasculeOnPoints().Val())->ReTagThis("BasculeOnPoints"));
   if (anObj.BasculeLiaisonOnPlan().IsInit())
      aRes->AddFils(ToXMLTree(anObj.BasculeLiaisonOnPlan().Val())->ReTagThis("BasculeLiaisonOnPlan"));
  return aRes;
}

void xml_init(cModeBascule & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.BasculeOnPoints(),aTree->Get("BasculeOnPoints",1)); //tototo 

   xml_init(anObj.BasculeLiaisonOnPlan(),aTree->Get("BasculeLiaisonOnPlan",1)); //tototo 
}


cTplValGesInit< bool > & cBasculeOrientation::AfterCompens()
{
   return mAfterCompens;
}

const cTplValGesInit< bool > & cBasculeOrientation::AfterCompens()const 
{
   return mAfterCompens;
}


cTplValGesInit< std::string > & cBasculeOrientation::PatternNameApply()
{
   return mPatternNameApply;
}

const cTplValGesInit< std::string > & cBasculeOrientation::PatternNameApply()const 
{
   return mPatternNameApply;
}


cTplValGesInit< std::string > & cBasculeOrientation::PatternNameEstim()
{
   return mPatternNameEstim;
}

const cTplValGesInit< std::string > & cBasculeOrientation::PatternNameEstim()const 
{
   return mPatternNameEstim;
}


cTplValGesInit< std::string > & cBasculeOrientation::PoseCentrale()
{
   return ModeBascule().BasculeOnPoints().Val().BascOnCentre().Val().PoseCentrale();
}

const cTplValGesInit< std::string > & cBasculeOrientation::PoseCentrale()const 
{
   return ModeBascule().BasculeOnPoints().Val().BascOnCentre().Val().PoseCentrale();
}


cTplValGesInit< cBascOnCentre > & cBasculeOrientation::BascOnCentre()
{
   return ModeBascule().BasculeOnPoints().Val().BascOnCentre();
}

const cTplValGesInit< cBascOnCentre > & cBasculeOrientation::BascOnCentre()const 
{
   return ModeBascule().BasculeOnPoints().Val().BascOnCentre();
}


std::string & cBasculeOrientation::NameRef()
{
   return ModeBascule().BasculeOnPoints().Val().BascOnAppuis().Val().NameRef();
}

const std::string & cBasculeOrientation::NameRef()const 
{
   return ModeBascule().BasculeOnPoints().Val().BascOnAppuis().Val().NameRef();
}


cTplValGesInit< cBascOnAppuis > & cBasculeOrientation::BascOnAppuis()
{
   return ModeBascule().BasculeOnPoints().Val().BascOnAppuis();
}

const cTplValGesInit< cBascOnAppuis > & cBasculeOrientation::BascOnAppuis()const 
{
   return ModeBascule().BasculeOnPoints().Val().BascOnAppuis();
}


cTplValGesInit< bool > & cBasculeOrientation::ModeL2()
{
   return ModeBascule().BasculeOnPoints().Val().ModeL2();
}

const cTplValGesInit< bool > & cBasculeOrientation::ModeL2()const 
{
   return ModeBascule().BasculeOnPoints().Val().ModeL2();
}


cTplValGesInit< cBasculeOnPoints > & cBasculeOrientation::BasculeOnPoints()
{
   return ModeBascule().BasculeOnPoints();
}

const cTplValGesInit< cBasculeOnPoints > & cBasculeOrientation::BasculeOnPoints()const 
{
   return ModeBascule().BasculeOnPoints();
}


cParamEstimPlan & cBasculeOrientation::EstimPl()
{
   return ModeBascule().BasculeLiaisonOnPlan().Val().EstimPl();
}

const cParamEstimPlan & cBasculeOrientation::EstimPl()const 
{
   return ModeBascule().BasculeLiaisonOnPlan().Val().EstimPl();
}


cTplValGesInit< double > & cBasculeOrientation::DistFixEch()
{
   return ModeBascule().BasculeLiaisonOnPlan().Val().OrientInPlane().Val().DistFixEch();
}

const cTplValGesInit< double > & cBasculeOrientation::DistFixEch()const 
{
   return ModeBascule().BasculeLiaisonOnPlan().Val().OrientInPlane().Val().DistFixEch();
}


std::string & cBasculeOrientation::FileMesures()
{
   return ModeBascule().BasculeLiaisonOnPlan().Val().OrientInPlane().Val().FileMesures();
}

const std::string & cBasculeOrientation::FileMesures()const 
{
   return ModeBascule().BasculeLiaisonOnPlan().Val().OrientInPlane().Val().FileMesures();
}


cTplValGesInit< std::string > & cBasculeOrientation::AlignOn()
{
   return ModeBascule().BasculeLiaisonOnPlan().Val().OrientInPlane().Val().AlignOn();
}

const cTplValGesInit< std::string > & cBasculeOrientation::AlignOn()const 
{
   return ModeBascule().BasculeLiaisonOnPlan().Val().OrientInPlane().Val().AlignOn();
}


cTplValGesInit< cOrientInPlane > & cBasculeOrientation::OrientInPlane()
{
   return ModeBascule().BasculeLiaisonOnPlan().Val().OrientInPlane();
}

const cTplValGesInit< cOrientInPlane > & cBasculeOrientation::OrientInPlane()const 
{
   return ModeBascule().BasculeLiaisonOnPlan().Val().OrientInPlane();
}


cTplValGesInit< cBasculeLiaisonOnPlan > & cBasculeOrientation::BasculeLiaisonOnPlan()
{
   return ModeBascule().BasculeLiaisonOnPlan();
}

const cTplValGesInit< cBasculeLiaisonOnPlan > & cBasculeOrientation::BasculeLiaisonOnPlan()const 
{
   return ModeBascule().BasculeLiaisonOnPlan();
}


cModeBascule & cBasculeOrientation::ModeBascule()
{
   return mModeBascule;
}

const cModeBascule & cBasculeOrientation::ModeBascule()const 
{
   return mModeBascule;
}

cElXMLTree * ToXMLTree(const cBasculeOrientation & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"BasculeOrientation",eXMLBranche);
   if (anObj.AfterCompens().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("AfterCompens"),anObj.AfterCompens().Val())->ReTagThis("AfterCompens"));
   if (anObj.PatternNameApply().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PatternNameApply"),anObj.PatternNameApply().Val())->ReTagThis("PatternNameApply"));
   if (anObj.PatternNameEstim().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PatternNameEstim"),anObj.PatternNameEstim().Val())->ReTagThis("PatternNameEstim"));
   aRes->AddFils(ToXMLTree(anObj.ModeBascule())->ReTagThis("ModeBascule"));
  return aRes;
}

void xml_init(cBasculeOrientation & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.AfterCompens(),aTree->Get("AfterCompens",1),bool(true)); //tototo 

   xml_init(anObj.PatternNameApply(),aTree->Get("PatternNameApply",1),std::string(".*")); //tototo 

   xml_init(anObj.PatternNameEstim(),aTree->Get("PatternNameEstim",1),std::string(".*")); //tototo 

   xml_init(anObj.ModeBascule(),aTree->Get("ModeBascule",1)); //tototo 
}


std::vector< cAperoPointeStereo > & cStereoFE::HomFE()
{
   return mHomFE;
}

const std::vector< cAperoPointeStereo > & cStereoFE::HomFE()const 
{
   return mHomFE;
}

cElXMLTree * ToXMLTree(const cStereoFE & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"StereoFE",eXMLBranche);
  for
  (       std::vector< cAperoPointeStereo >::const_iterator it=anObj.HomFE().begin();
      it !=anObj.HomFE().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("HomFE"));
  return aRes;
}

void xml_init(cStereoFE & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.HomFE(),aTree->GetAll("HomFE",false,1));
}


std::vector< cAperoPointeStereo > & cModeFE::HomFE()
{
   return StereoFE().Val().HomFE();
}

const std::vector< cAperoPointeStereo > & cModeFE::HomFE()const 
{
   return StereoFE().Val().HomFE();
}


cTplValGesInit< cStereoFE > & cModeFE::StereoFE()
{
   return mStereoFE;
}

const cTplValGesInit< cStereoFE > & cModeFE::StereoFE()const 
{
   return mStereoFE;
}


cTplValGesInit< cApero2PointeFromFile > & cModeFE::FEFromFile()
{
   return mFEFromFile;
}

const cTplValGesInit< cApero2PointeFromFile > & cModeFE::FEFromFile()const 
{
   return mFEFromFile;
}

cElXMLTree * ToXMLTree(const cModeFE & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ModeFE",eXMLBranche);
   if (anObj.StereoFE().IsInit())
      aRes->AddFils(ToXMLTree(anObj.StereoFE().Val())->ReTagThis("StereoFE"));
   if (anObj.FEFromFile().IsInit())
      aRes->AddFils(ToXMLTree(anObj.FEFromFile().Val())->ReTagThis("FEFromFile"));
  return aRes;
}

void xml_init(cModeFE & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.StereoFE(),aTree->Get("StereoFE",1)); //tototo 

   xml_init(anObj.FEFromFile(),aTree->Get("FEFromFile",1)); //tototo 
}


std::vector< cAperoPointeStereo > & cFixeEchelle::HomFE()
{
   return ModeFE().StereoFE().Val().HomFE();
}

const std::vector< cAperoPointeStereo > & cFixeEchelle::HomFE()const 
{
   return ModeFE().StereoFE().Val().HomFE();
}


cTplValGesInit< cStereoFE > & cFixeEchelle::StereoFE()
{
   return ModeFE().StereoFE();
}

const cTplValGesInit< cStereoFE > & cFixeEchelle::StereoFE()const 
{
   return ModeFE().StereoFE();
}


cTplValGesInit< cApero2PointeFromFile > & cFixeEchelle::FEFromFile()
{
   return ModeFE().FEFromFile();
}

const cTplValGesInit< cApero2PointeFromFile > & cFixeEchelle::FEFromFile()const 
{
   return ModeFE().FEFromFile();
}


cModeFE & cFixeEchelle::ModeFE()
{
   return mModeFE;
}

const cModeFE & cFixeEchelle::ModeFE()const 
{
   return mModeFE;
}


double & cFixeEchelle::DistVraie()
{
   return mDistVraie;
}

const double & cFixeEchelle::DistVraie()const 
{
   return mDistVraie;
}

cElXMLTree * ToXMLTree(const cFixeEchelle & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"FixeEchelle",eXMLBranche);
   aRes->AddFils(ToXMLTree(anObj.ModeFE())->ReTagThis("ModeFE"));
   aRes->AddFils(::ToXMLTree(std::string("DistVraie"),anObj.DistVraie())->ReTagThis("DistVraie"));
  return aRes;
}

void xml_init(cFixeEchelle & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.ModeFE(),aTree->Get("ModeFE",1)); //tototo 

   xml_init(anObj.DistVraie(),aTree->Get("DistVraie",1)); //tototo 
}


std::vector< cAperoPointeMono > & cHorFOP::VecFOH()
{
   return mVecFOH;
}

const std::vector< cAperoPointeMono > & cHorFOP::VecFOH()const 
{
   return mVecFOH;
}


cTplValGesInit< double > & cHorFOP::Z()
{
   return mZ;
}

const cTplValGesInit< double > & cHorFOP::Z()const 
{
   return mZ;
}

cElXMLTree * ToXMLTree(const cHorFOP & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"HorFOP",eXMLBranche);
  for
  (       std::vector< cAperoPointeMono >::const_iterator it=anObj.VecFOH().begin();
      it !=anObj.VecFOH().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("VecFOH"));
   if (anObj.Z().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Z"),anObj.Z().Val())->ReTagThis("Z"));
  return aRes;
}

void xml_init(cHorFOP & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.VecFOH(),aTree->GetAll("VecFOH",false,1));

   xml_init(anObj.Z(),aTree->Get("Z",1),double(0)); //tototo 
}


std::vector< cAperoPointeMono > & cModeFOP::VecFOH()
{
   return HorFOP().Val().VecFOH();
}

const std::vector< cAperoPointeMono > & cModeFOP::VecFOH()const 
{
   return HorFOP().Val().VecFOH();
}


cTplValGesInit< double > & cModeFOP::Z()
{
   return HorFOP().Val().Z();
}

const cTplValGesInit< double > & cModeFOP::Z()const 
{
   return HorFOP().Val().Z();
}


cTplValGesInit< cHorFOP > & cModeFOP::HorFOP()
{
   return mHorFOP;
}

const cTplValGesInit< cHorFOP > & cModeFOP::HorFOP()const 
{
   return mHorFOP;
}


cTplValGesInit< cApero2PointeFromFile > & cModeFOP::HorFromFile()
{
   return mHorFromFile;
}

const cTplValGesInit< cApero2PointeFromFile > & cModeFOP::HorFromFile()const 
{
   return mHorFromFile;
}

cElXMLTree * ToXMLTree(const cModeFOP & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ModeFOP",eXMLBranche);
   if (anObj.HorFOP().IsInit())
      aRes->AddFils(ToXMLTree(anObj.HorFOP().Val())->ReTagThis("HorFOP"));
   if (anObj.HorFromFile().IsInit())
      aRes->AddFils(ToXMLTree(anObj.HorFromFile().Val())->ReTagThis("HorFromFile"));
  return aRes;
}

void xml_init(cModeFOP & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.HorFOP(),aTree->Get("HorFOP",1)); //tototo 

   xml_init(anObj.HorFromFile(),aTree->Get("HorFromFile",1)); //tototo 
}


std::vector< cAperoPointeMono > & cFixeOrientPlane::VecFOH()
{
   return ModeFOP().HorFOP().Val().VecFOH();
}

const std::vector< cAperoPointeMono > & cFixeOrientPlane::VecFOH()const 
{
   return ModeFOP().HorFOP().Val().VecFOH();
}


cTplValGesInit< double > & cFixeOrientPlane::Z()
{
   return ModeFOP().HorFOP().Val().Z();
}

const cTplValGesInit< double > & cFixeOrientPlane::Z()const 
{
   return ModeFOP().HorFOP().Val().Z();
}


cTplValGesInit< cHorFOP > & cFixeOrientPlane::HorFOP()
{
   return ModeFOP().HorFOP();
}

const cTplValGesInit< cHorFOP > & cFixeOrientPlane::HorFOP()const 
{
   return ModeFOP().HorFOP();
}


cTplValGesInit< cApero2PointeFromFile > & cFixeOrientPlane::HorFromFile()
{
   return ModeFOP().HorFromFile();
}

const cTplValGesInit< cApero2PointeFromFile > & cFixeOrientPlane::HorFromFile()const 
{
   return ModeFOP().HorFromFile();
}


cModeFOP & cFixeOrientPlane::ModeFOP()
{
   return mModeFOP;
}

const cModeFOP & cFixeOrientPlane::ModeFOP()const 
{
   return mModeFOP;
}


Pt2dr & cFixeOrientPlane::Vecteur()
{
   return mVecteur;
}

const Pt2dr & cFixeOrientPlane::Vecteur()const 
{
   return mVecteur;
}

cElXMLTree * ToXMLTree(const cFixeOrientPlane & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"FixeOrientPlane",eXMLBranche);
   aRes->AddFils(ToXMLTree(anObj.ModeFOP())->ReTagThis("ModeFOP"));
   aRes->AddFils(::ToXMLTree(std::string("Vecteur"),anObj.Vecteur())->ReTagThis("Vecteur"));
  return aRes;
}

void xml_init(cFixeOrientPlane & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.ModeFOP(),aTree->Get("ModeFOP",1)); //tototo 

   xml_init(anObj.Vecteur(),aTree->Get("Vecteur",1)); //tototo 
}


std::string & cBlocBascule::Pattern1()
{
   return mPattern1;
}

const std::string & cBlocBascule::Pattern1()const 
{
   return mPattern1;
}


std::string & cBlocBascule::Pattern2()
{
   return mPattern2;
}

const std::string & cBlocBascule::Pattern2()const 
{
   return mPattern2;
}


std::string & cBlocBascule::IdBdl()
{
   return mIdBdl;
}

const std::string & cBlocBascule::IdBdl()const 
{
   return mIdBdl;
}

cElXMLTree * ToXMLTree(const cBlocBascule & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"BlocBascule",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Pattern1"),anObj.Pattern1())->ReTagThis("Pattern1"));
   aRes->AddFils(::ToXMLTree(std::string("Pattern2"),anObj.Pattern2())->ReTagThis("Pattern2"));
   aRes->AddFils(::ToXMLTree(std::string("IdBdl"),anObj.IdBdl())->ReTagThis("IdBdl"));
  return aRes;
}

void xml_init(cBlocBascule & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.Pattern1(),aTree->Get("Pattern1",1)); //tototo 

   xml_init(anObj.Pattern2(),aTree->Get("Pattern2",1)); //tototo 

   xml_init(anObj.IdBdl(),aTree->Get("IdBdl",1)); //tototo 
}


int & cMesureErreurTournante::Periode()
{
   return mPeriode;
}

const int & cMesureErreurTournante::Periode()const 
{
   return mPeriode;
}


cTplValGesInit< int > & cMesureErreurTournante::NbTest()
{
   return mNbTest;
}

const cTplValGesInit< int > & cMesureErreurTournante::NbTest()const 
{
   return mNbTest;
}


cTplValGesInit< int > & cMesureErreurTournante::NbIter()
{
   return mNbIter;
}

const cTplValGesInit< int > & cMesureErreurTournante::NbIter()const 
{
   return mNbIter;
}


cTplValGesInit< bool > & cMesureErreurTournante::ApplyAppuis()
{
   return mApplyAppuis;
}

const cTplValGesInit< bool > & cMesureErreurTournante::ApplyAppuis()const 
{
   return mApplyAppuis;
}


cTplValGesInit< bool > & cMesureErreurTournante::ApplyLiaisons()
{
   return mApplyLiaisons;
}

const cTplValGesInit< bool > & cMesureErreurTournante::ApplyLiaisons()const 
{
   return mApplyLiaisons;
}

cElXMLTree * ToXMLTree(const cMesureErreurTournante & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"MesureErreurTournante",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Periode"),anObj.Periode())->ReTagThis("Periode"));
   if (anObj.NbTest().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NbTest"),anObj.NbTest().Val())->ReTagThis("NbTest"));
   if (anObj.NbIter().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NbIter"),anObj.NbIter().Val())->ReTagThis("NbIter"));
   if (anObj.ApplyAppuis().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ApplyAppuis"),anObj.ApplyAppuis().Val())->ReTagThis("ApplyAppuis"));
   if (anObj.ApplyLiaisons().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ApplyLiaisons"),anObj.ApplyLiaisons().Val())->ReTagThis("ApplyLiaisons"));
  return aRes;
}

void xml_init(cMesureErreurTournante & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.Periode(),aTree->Get("Periode",1)); //tototo 

   xml_init(anObj.NbTest(),aTree->Get("NbTest",1)); //tototo 

   xml_init(anObj.NbIter(),aTree->Get("NbIter",1),int(4)); //tototo 

   xml_init(anObj.ApplyAppuis(),aTree->Get("ApplyAppuis",1),bool(true)); //tototo 

   xml_init(anObj.ApplyLiaisons(),aTree->Get("ApplyLiaisons",1),bool(false)); //tototo 
}


cTplValGesInit< double > & cContraintesCamerasInc::TolContrainte()
{
   return mTolContrainte;
}

const cTplValGesInit< double > & cContraintesCamerasInc::TolContrainte()const 
{
   return mTolContrainte;
}


cTplValGesInit< std::string > & cContraintesCamerasInc::PatternNameApply()
{
   return mPatternNameApply;
}

const cTplValGesInit< std::string > & cContraintesCamerasInc::PatternNameApply()const 
{
   return mPatternNameApply;
}


std::list< eTypeContrainteCalibCamera > & cContraintesCamerasInc::Val()
{
   return mVal;
}

const std::list< eTypeContrainteCalibCamera > & cContraintesCamerasInc::Val()const 
{
   return mVal;
}


cTplValGesInit< cElRegex_Ptr > & cContraintesCamerasInc::PatternRefuteur()
{
   return mPatternRefuteur;
}

const cTplValGesInit< cElRegex_Ptr > & cContraintesCamerasInc::PatternRefuteur()const 
{
   return mPatternRefuteur;
}

cElXMLTree * ToXMLTree(const cContraintesCamerasInc & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ContraintesCamerasInc",eXMLBranche);
   if (anObj.TolContrainte().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("TolContrainte"),anObj.TolContrainte().Val())->ReTagThis("TolContrainte"));
   if (anObj.PatternNameApply().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PatternNameApply"),anObj.PatternNameApply().Val())->ReTagThis("PatternNameApply"));
  for
  (       std::list< eTypeContrainteCalibCamera >::const_iterator it=anObj.Val().begin();
      it !=anObj.Val().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree(std::string("Val"),(*it))->ReTagThis("Val"));
   if (anObj.PatternRefuteur().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PatternRefuteur"),anObj.PatternRefuteur().Val())->ReTagThis("PatternRefuteur"));
  return aRes;
}

void xml_init(cContraintesCamerasInc & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.TolContrainte(),aTree->Get("TolContrainte",1),double(-1)); //tototo 

   xml_init(anObj.PatternNameApply(),aTree->Get("PatternNameApply",1),std::string(".*")); //tototo 

   xml_init(anObj.Val(),aTree->GetAll("Val",false,1));

   xml_init(anObj.PatternRefuteur(),aTree->Get("PatternRefuteur",1)); //tototo 
}


cTplValGesInit< bool > & cContraintesPoses::ByPattern()
{
   return mByPattern;
}

const cTplValGesInit< bool > & cContraintesPoses::ByPattern()const 
{
   return mByPattern;
}


cTplValGesInit< std::string > & cContraintesPoses::PatternRefuteur()
{
   return mPatternRefuteur;
}

const cTplValGesInit< std::string > & cContraintesPoses::PatternRefuteur()const 
{
   return mPatternRefuteur;
}


cTplValGesInit< double > & cContraintesPoses::TolAng()
{
   return mTolAng;
}

const cTplValGesInit< double > & cContraintesPoses::TolAng()const 
{
   return mTolAng;
}


cTplValGesInit< double > & cContraintesPoses::TolCoord()
{
   return mTolCoord;
}

const cTplValGesInit< double > & cContraintesPoses::TolCoord()const 
{
   return mTolCoord;
}


std::string & cContraintesPoses::NamePose()
{
   return mNamePose;
}

const std::string & cContraintesPoses::NamePose()const 
{
   return mNamePose;
}


eTypeContraintePoseCamera & cContraintesPoses::Val()
{
   return mVal;
}

const eTypeContraintePoseCamera & cContraintesPoses::Val()const 
{
   return mVal;
}


cTplValGesInit< std::string > & cContraintesPoses::PoseRattachement()
{
   return mPoseRattachement;
}

const cTplValGesInit< std::string > & cContraintesPoses::PoseRattachement()const 
{
   return mPoseRattachement;
}

cElXMLTree * ToXMLTree(const cContraintesPoses & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ContraintesPoses",eXMLBranche);
   if (anObj.ByPattern().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ByPattern"),anObj.ByPattern().Val())->ReTagThis("ByPattern"));
   if (anObj.PatternRefuteur().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PatternRefuteur"),anObj.PatternRefuteur().Val())->ReTagThis("PatternRefuteur"));
   if (anObj.TolAng().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("TolAng"),anObj.TolAng().Val())->ReTagThis("TolAng"));
   if (anObj.TolCoord().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("TolCoord"),anObj.TolCoord().Val())->ReTagThis("TolCoord"));
   aRes->AddFils(::ToXMLTree(std::string("NamePose"),anObj.NamePose())->ReTagThis("NamePose"));
   aRes->AddFils(ToXMLTree(std::string("Val"),anObj.Val())->ReTagThis("Val"));
   if (anObj.PoseRattachement().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PoseRattachement"),anObj.PoseRattachement().Val())->ReTagThis("PoseRattachement"));
  return aRes;
}

void xml_init(cContraintesPoses & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.ByPattern(),aTree->Get("ByPattern",1),bool(false)); //tototo 

   xml_init(anObj.PatternRefuteur(),aTree->Get("PatternRefuteur",1)); //tototo 

   xml_init(anObj.TolAng(),aTree->Get("TolAng",1),double(-1)); //tototo 

   xml_init(anObj.TolCoord(),aTree->Get("TolCoord",1),double(-1)); //tototo 

   xml_init(anObj.NamePose(),aTree->Get("NamePose",1)); //tototo 

   xml_init(anObj.Val(),aTree->Get("Val",1)); //tototo 

   xml_init(anObj.PoseRattachement(),aTree->Get("PoseRattachement",1)); //tototo 
}


std::list< cContraintesCamerasInc > & cSectionContraintes::ContraintesCamerasInc()
{
   return mContraintesCamerasInc;
}

const std::list< cContraintesCamerasInc > & cSectionContraintes::ContraintesCamerasInc()const 
{
   return mContraintesCamerasInc;
}


std::list< cContraintesPoses > & cSectionContraintes::ContraintesPoses()
{
   return mContraintesPoses;
}

const std::list< cContraintesPoses > & cSectionContraintes::ContraintesPoses()const 
{
   return mContraintesPoses;
}

cElXMLTree * ToXMLTree(const cSectionContraintes & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"SectionContraintes",eXMLBranche);
  for
  (       std::list< cContraintesCamerasInc >::const_iterator it=anObj.ContraintesCamerasInc().begin();
      it !=anObj.ContraintesCamerasInc().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("ContraintesCamerasInc"));
  for
  (       std::list< cContraintesPoses >::const_iterator it=anObj.ContraintesPoses().begin();
      it !=anObj.ContraintesPoses().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("ContraintesPoses"));
  return aRes;
}

void xml_init(cSectionContraintes & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.ContraintesCamerasInc(),aTree->GetAll("ContraintesCamerasInc",false,1));

   xml_init(anObj.ContraintesPoses(),aTree->GetAll("ContraintesPoses",false,1));
}


std::string & cVisuPtsMult::Cam1()
{
   return mCam1;
}

const std::string & cVisuPtsMult::Cam1()const 
{
   return mCam1;
}


std::string & cVisuPtsMult::Id()
{
   return mId;
}

const std::string & cVisuPtsMult::Id()const 
{
   return mId;
}


cTplValGesInit< int > & cVisuPtsMult::SzWPrinc()
{
   return mSzWPrinc;
}

const cTplValGesInit< int > & cVisuPtsMult::SzWPrinc()const 
{
   return mSzWPrinc;
}


cTplValGesInit< int > & cVisuPtsMult::SzWAux()
{
   return mSzWAux;
}

const cTplValGesInit< int > & cVisuPtsMult::SzWAux()const 
{
   return mSzWAux;
}


cTplValGesInit< int > & cVisuPtsMult::ZoomWAux()
{
   return mZoomWAux;
}

const cTplValGesInit< int > & cVisuPtsMult::ZoomWAux()const 
{
   return mZoomWAux;
}


Pt2di & cVisuPtsMult::NbWAux()
{
   return mNbWAux;
}

const Pt2di & cVisuPtsMult::NbWAux()const 
{
   return mNbWAux;
}


bool & cVisuPtsMult::AuxEnDessous()
{
   return mAuxEnDessous;
}

const bool & cVisuPtsMult::AuxEnDessous()const 
{
   return mAuxEnDessous;
}


cTplValGesInit< double > & cVisuPtsMult::MaxDistReproj()
{
   return mMaxDistReproj;
}

const cTplValGesInit< double > & cVisuPtsMult::MaxDistReproj()const 
{
   return mMaxDistReproj;
}


cTplValGesInit< double > & cVisuPtsMult::MaxDistSift()
{
   return mMaxDistSift;
}

const cTplValGesInit< double > & cVisuPtsMult::MaxDistSift()const 
{
   return mMaxDistSift;
}


cTplValGesInit< double > & cVisuPtsMult::MaxDistProjCorr()
{
   return mMaxDistProjCorr;
}

const cTplValGesInit< double > & cVisuPtsMult::MaxDistProjCorr()const 
{
   return mMaxDistProjCorr;
}


cTplValGesInit< double > & cVisuPtsMult::SeuilCorrel()
{
   return mSeuilCorrel;
}

const cTplValGesInit< double > & cVisuPtsMult::SeuilCorrel()const 
{
   return mSeuilCorrel;
}

cElXMLTree * ToXMLTree(const cVisuPtsMult & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"VisuPtsMult",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Cam1"),anObj.Cam1())->ReTagThis("Cam1"));
   aRes->AddFils(::ToXMLTree(std::string("Id"),anObj.Id())->ReTagThis("Id"));
   if (anObj.SzWPrinc().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SzWPrinc"),anObj.SzWPrinc().Val())->ReTagThis("SzWPrinc"));
   if (anObj.SzWAux().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SzWAux"),anObj.SzWAux().Val())->ReTagThis("SzWAux"));
   if (anObj.ZoomWAux().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ZoomWAux"),anObj.ZoomWAux().Val())->ReTagThis("ZoomWAux"));
   aRes->AddFils(::ToXMLTree(std::string("NbWAux"),anObj.NbWAux())->ReTagThis("NbWAux"));
   aRes->AddFils(::ToXMLTree(std::string("AuxEnDessous"),anObj.AuxEnDessous())->ReTagThis("AuxEnDessous"));
   if (anObj.MaxDistReproj().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("MaxDistReproj"),anObj.MaxDistReproj().Val())->ReTagThis("MaxDistReproj"));
   if (anObj.MaxDistSift().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("MaxDistSift"),anObj.MaxDistSift().Val())->ReTagThis("MaxDistSift"));
   if (anObj.MaxDistProjCorr().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("MaxDistProjCorr"),anObj.MaxDistProjCorr().Val())->ReTagThis("MaxDistProjCorr"));
   if (anObj.SeuilCorrel().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SeuilCorrel"),anObj.SeuilCorrel().Val())->ReTagThis("SeuilCorrel"));
  return aRes;
}

void xml_init(cVisuPtsMult & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.Cam1(),aTree->Get("Cam1",1)); //tototo 

   xml_init(anObj.Id(),aTree->Get("Id",1)); //tototo 

   xml_init(anObj.SzWPrinc(),aTree->Get("SzWPrinc",1),int(500)); //tototo 

   xml_init(anObj.SzWAux(),aTree->Get("SzWAux",1),int(100)); //tototo 

   xml_init(anObj.ZoomWAux(),aTree->Get("ZoomWAux",1),int(5)); //tototo 

   xml_init(anObj.NbWAux(),aTree->Get("NbWAux",1)); //tototo 

   xml_init(anObj.AuxEnDessous(),aTree->Get("AuxEnDessous",1)); //tototo 

   xml_init(anObj.MaxDistReproj(),aTree->Get("MaxDistReproj",1),double(1.5)); //tototo 

   xml_init(anObj.MaxDistSift(),aTree->Get("MaxDistSift",1),double(1.0)); //tototo 

   xml_init(anObj.MaxDistProjCorr(),aTree->Get("MaxDistProjCorr",1),double(3.0)); //tototo 

   xml_init(anObj.SeuilCorrel(),aTree->Get("SeuilCorrel",1),double(0.85)); //tototo 
}


std::string & cVerifAero::PatternApply()
{
   return mPatternApply;
}

const std::string & cVerifAero::PatternApply()const 
{
   return mPatternApply;
}


std::string & cVerifAero::IdBdLiaison()
{
   return mIdBdLiaison;
}

const std::string & cVerifAero::IdBdLiaison()const 
{
   return mIdBdLiaison;
}


cPonderationPackMesure & cVerifAero::Pond()
{
   return mPond;
}

const cPonderationPackMesure & cVerifAero::Pond()const 
{
   return mPond;
}


std::string & cVerifAero::Prefixe()
{
   return mPrefixe;
}

const std::string & cVerifAero::Prefixe()const 
{
   return mPrefixe;
}


eTypeVerif & cVerifAero::TypeVerif()
{
   return mTypeVerif;
}

const eTypeVerif & cVerifAero::TypeVerif()const 
{
   return mTypeVerif;
}


double & cVerifAero::SeuilTxt()
{
   return mSeuilTxt;
}

const double & cVerifAero::SeuilTxt()const 
{
   return mSeuilTxt;
}


double & cVerifAero::Resol()
{
   return mResol;
}

const double & cVerifAero::Resol()const 
{
   return mResol;
}


double & cVerifAero::PasR()
{
   return mPasR;
}

const double & cVerifAero::PasR()const 
{
   return mPasR;
}


double & cVerifAero::PasB()
{
   return mPasB;
}

const double & cVerifAero::PasB()const 
{
   return mPasB;
}

cElXMLTree * ToXMLTree(const cVerifAero & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"VerifAero",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("PatternApply"),anObj.PatternApply())->ReTagThis("PatternApply"));
   aRes->AddFils(::ToXMLTree(std::string("IdBdLiaison"),anObj.IdBdLiaison())->ReTagThis("IdBdLiaison"));
   aRes->AddFils(ToXMLTree(anObj.Pond())->ReTagThis("Pond"));
   aRes->AddFils(::ToXMLTree(std::string("Prefixe"),anObj.Prefixe())->ReTagThis("Prefixe"));
   aRes->AddFils(ToXMLTree(std::string("TypeVerif"),anObj.TypeVerif())->ReTagThis("TypeVerif"));
   aRes->AddFils(::ToXMLTree(std::string("SeuilTxt"),anObj.SeuilTxt())->ReTagThis("SeuilTxt"));
   aRes->AddFils(::ToXMLTree(std::string("Resol"),anObj.Resol())->ReTagThis("Resol"));
   aRes->AddFils(::ToXMLTree(std::string("PasR"),anObj.PasR())->ReTagThis("PasR"));
   aRes->AddFils(::ToXMLTree(std::string("PasB"),anObj.PasB())->ReTagThis("PasB"));
  return aRes;
}

void xml_init(cVerifAero & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.PatternApply(),aTree->Get("PatternApply",1)); //tototo 

   xml_init(anObj.IdBdLiaison(),aTree->Get("IdBdLiaison",1)); //tototo 

   xml_init(anObj.Pond(),aTree->Get("Pond",1)); //tototo 

   xml_init(anObj.Prefixe(),aTree->Get("Prefixe",1)); //tototo 

   xml_init(anObj.TypeVerif(),aTree->Get("TypeVerif",1)); //tototo 

   xml_init(anObj.SeuilTxt(),aTree->Get("SeuilTxt",1)); //tototo 

   xml_init(anObj.Resol(),aTree->Get("Resol",1)); //tototo 

   xml_init(anObj.PasR(),aTree->Get("PasR",1)); //tototo 

   xml_init(anObj.PasB(),aTree->Get("PasB",1)); //tototo 
}


Pt3dr & cGPtsTer_By_ImProf::Origine()
{
   return mOrigine;
}

const Pt3dr & cGPtsTer_By_ImProf::Origine()const 
{
   return mOrigine;
}


Pt3dr & cGPtsTer_By_ImProf::Step()
{
   return mStep;
}

const Pt3dr & cGPtsTer_By_ImProf::Step()const 
{
   return mStep;
}


int & cGPtsTer_By_ImProf::NbPts()
{
   return mNbPts;
}

const int & cGPtsTer_By_ImProf::NbPts()const 
{
   return mNbPts;
}


bool & cGPtsTer_By_ImProf::OnGrid()
{
   return mOnGrid;
}

const bool & cGPtsTer_By_ImProf::OnGrid()const 
{
   return mOnGrid;
}


std::string & cGPtsTer_By_ImProf::File()
{
   return mFile;
}

const std::string & cGPtsTer_By_ImProf::File()const 
{
   return mFile;
}


cTplValGesInit< double > & cGPtsTer_By_ImProf::RandomizeInGrid()
{
   return mRandomizeInGrid;
}

const cTplValGesInit< double > & cGPtsTer_By_ImProf::RandomizeInGrid()const 
{
   return mRandomizeInGrid;
}


cTplValGesInit< std::string > & cGPtsTer_By_ImProf::ImMaitresse()
{
   return mImMaitresse;
}

const cTplValGesInit< std::string > & cGPtsTer_By_ImProf::ImMaitresse()const 
{
   return mImMaitresse;
}


cTplValGesInit< bool > & cGPtsTer_By_ImProf::DTMIsZ()
{
   return mDTMIsZ;
}

const cTplValGesInit< bool > & cGPtsTer_By_ImProf::DTMIsZ()const 
{
   return mDTMIsZ;
}

cElXMLTree * ToXMLTree(const cGPtsTer_By_ImProf & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"GPtsTer_By_ImProf",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Origine"),anObj.Origine())->ReTagThis("Origine"));
   aRes->AddFils(::ToXMLTree(std::string("Step"),anObj.Step())->ReTagThis("Step"));
   aRes->AddFils(::ToXMLTree(std::string("NbPts"),anObj.NbPts())->ReTagThis("NbPts"));
   aRes->AddFils(::ToXMLTree(std::string("OnGrid"),anObj.OnGrid())->ReTagThis("OnGrid"));
   aRes->AddFils(::ToXMLTree(std::string("File"),anObj.File())->ReTagThis("File"));
   if (anObj.RandomizeInGrid().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("RandomizeInGrid"),anObj.RandomizeInGrid().Val())->ReTagThis("RandomizeInGrid"));
   if (anObj.ImMaitresse().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ImMaitresse"),anObj.ImMaitresse().Val())->ReTagThis("ImMaitresse"));
   if (anObj.DTMIsZ().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DTMIsZ"),anObj.DTMIsZ().Val())->ReTagThis("DTMIsZ"));
  return aRes;
}

void xml_init(cGPtsTer_By_ImProf & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.Origine(),aTree->Get("Origine",1)); //tototo 

   xml_init(anObj.Step(),aTree->Get("Step",1)); //tototo 

   xml_init(anObj.NbPts(),aTree->Get("NbPts",1)); //tototo 

   xml_init(anObj.OnGrid(),aTree->Get("OnGrid",1)); //tototo 

   xml_init(anObj.File(),aTree->Get("File",1)); //tototo 

   xml_init(anObj.RandomizeInGrid(),aTree->Get("RandomizeInGrid",1),double(0.0)); //tototo 

   xml_init(anObj.ImMaitresse(),aTree->Get("ImMaitresse",1)); //tototo 

   xml_init(anObj.DTMIsZ(),aTree->Get("DTMIsZ",1),bool(true)); //tototo 
}


cTplValGesInit< std::string > & cGeneratePointsTerrains::GPtsTer_By_File()
{
   return mGPtsTer_By_File;
}

const cTplValGesInit< std::string > & cGeneratePointsTerrains::GPtsTer_By_File()const 
{
   return mGPtsTer_By_File;
}


Pt3dr & cGeneratePointsTerrains::Origine()
{
   return GPtsTer_By_ImProf().Val().Origine();
}

const Pt3dr & cGeneratePointsTerrains::Origine()const 
{
   return GPtsTer_By_ImProf().Val().Origine();
}


Pt3dr & cGeneratePointsTerrains::Step()
{
   return GPtsTer_By_ImProf().Val().Step();
}

const Pt3dr & cGeneratePointsTerrains::Step()const 
{
   return GPtsTer_By_ImProf().Val().Step();
}


int & cGeneratePointsTerrains::NbPts()
{
   return GPtsTer_By_ImProf().Val().NbPts();
}

const int & cGeneratePointsTerrains::NbPts()const 
{
   return GPtsTer_By_ImProf().Val().NbPts();
}


bool & cGeneratePointsTerrains::OnGrid()
{
   return GPtsTer_By_ImProf().Val().OnGrid();
}

const bool & cGeneratePointsTerrains::OnGrid()const 
{
   return GPtsTer_By_ImProf().Val().OnGrid();
}


std::string & cGeneratePointsTerrains::File()
{
   return GPtsTer_By_ImProf().Val().File();
}

const std::string & cGeneratePointsTerrains::File()const 
{
   return GPtsTer_By_ImProf().Val().File();
}


cTplValGesInit< double > & cGeneratePointsTerrains::RandomizeInGrid()
{
   return GPtsTer_By_ImProf().Val().RandomizeInGrid();
}

const cTplValGesInit< double > & cGeneratePointsTerrains::RandomizeInGrid()const 
{
   return GPtsTer_By_ImProf().Val().RandomizeInGrid();
}


cTplValGesInit< std::string > & cGeneratePointsTerrains::ImMaitresse()
{
   return GPtsTer_By_ImProf().Val().ImMaitresse();
}

const cTplValGesInit< std::string > & cGeneratePointsTerrains::ImMaitresse()const 
{
   return GPtsTer_By_ImProf().Val().ImMaitresse();
}


cTplValGesInit< bool > & cGeneratePointsTerrains::DTMIsZ()
{
   return GPtsTer_By_ImProf().Val().DTMIsZ();
}

const cTplValGesInit< bool > & cGeneratePointsTerrains::DTMIsZ()const 
{
   return GPtsTer_By_ImProf().Val().DTMIsZ();
}


cTplValGesInit< cGPtsTer_By_ImProf > & cGeneratePointsTerrains::GPtsTer_By_ImProf()
{
   return mGPtsTer_By_ImProf;
}

const cTplValGesInit< cGPtsTer_By_ImProf > & cGeneratePointsTerrains::GPtsTer_By_ImProf()const 
{
   return mGPtsTer_By_ImProf;
}

cElXMLTree * ToXMLTree(const cGeneratePointsTerrains & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"GeneratePointsTerrains",eXMLBranche);
   if (anObj.GPtsTer_By_File().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("GPtsTer_By_File"),anObj.GPtsTer_By_File().Val())->ReTagThis("GPtsTer_By_File"));
   if (anObj.GPtsTer_By_ImProf().IsInit())
      aRes->AddFils(ToXMLTree(anObj.GPtsTer_By_ImProf().Val())->ReTagThis("GPtsTer_By_ImProf"));
  return aRes;
}

void xml_init(cGeneratePointsTerrains & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.GPtsTer_By_File(),aTree->Get("GPtsTer_By_File",1)); //tototo 

   xml_init(anObj.GPtsTer_By_ImProf(),aTree->Get("GPtsTer_By_ImProf",1)); //tototo 
}


std::string & cGenerateLiaisons::KeyAssoc()
{
   return mKeyAssoc;
}

const std::string & cGenerateLiaisons::KeyAssoc()const 
{
   return mKeyAssoc;
}


cTplValGesInit< std::string > & cGenerateLiaisons::FilterIm1()
{
   return mFilterIm1;
}

const cTplValGesInit< std::string > & cGenerateLiaisons::FilterIm1()const 
{
   return mFilterIm1;
}


cTplValGesInit< std::string > & cGenerateLiaisons::FilterIm2()
{
   return mFilterIm2;
}

const cTplValGesInit< std::string > & cGenerateLiaisons::FilterIm2()const 
{
   return mFilterIm2;
}


double & cGenerateLiaisons::BruitIm1()
{
   return mBruitIm1;
}

const double & cGenerateLiaisons::BruitIm1()const 
{
   return mBruitIm1;
}


double & cGenerateLiaisons::BruitIm2()
{
   return mBruitIm2;
}

const double & cGenerateLiaisons::BruitIm2()const 
{
   return mBruitIm2;
}

cElXMLTree * ToXMLTree(const cGenerateLiaisons & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"GenerateLiaisons",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("KeyAssoc"),anObj.KeyAssoc())->ReTagThis("KeyAssoc"));
   if (anObj.FilterIm1().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("FilterIm1"),anObj.FilterIm1().Val())->ReTagThis("FilterIm1"));
   if (anObj.FilterIm2().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("FilterIm2"),anObj.FilterIm2().Val())->ReTagThis("FilterIm2"));
   aRes->AddFils(::ToXMLTree(std::string("BruitIm1"),anObj.BruitIm1())->ReTagThis("BruitIm1"));
   aRes->AddFils(::ToXMLTree(std::string("BruitIm2"),anObj.BruitIm2())->ReTagThis("BruitIm2"));
  return aRes;
}

void xml_init(cGenerateLiaisons & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.KeyAssoc(),aTree->Get("KeyAssoc",1)); //tototo 

   xml_init(anObj.FilterIm1(),aTree->Get("FilterIm1",1),std::string(".*")); //tototo 

   xml_init(anObj.FilterIm2(),aTree->Get("FilterIm2",1),std::string(".*")); //tototo 

   xml_init(anObj.BruitIm1(),aTree->Get("BruitIm1",1)); //tototo 

   xml_init(anObj.BruitIm2(),aTree->Get("BruitIm2",1)); //tototo 
}


cTplValGesInit< std::string > & cExportSimulation::GPtsTer_By_File()
{
   return GeneratePointsTerrains().GPtsTer_By_File();
}

const cTplValGesInit< std::string > & cExportSimulation::GPtsTer_By_File()const 
{
   return GeneratePointsTerrains().GPtsTer_By_File();
}


Pt3dr & cExportSimulation::Origine()
{
   return GeneratePointsTerrains().GPtsTer_By_ImProf().Val().Origine();
}

const Pt3dr & cExportSimulation::Origine()const 
{
   return GeneratePointsTerrains().GPtsTer_By_ImProf().Val().Origine();
}


Pt3dr & cExportSimulation::Step()
{
   return GeneratePointsTerrains().GPtsTer_By_ImProf().Val().Step();
}

const Pt3dr & cExportSimulation::Step()const 
{
   return GeneratePointsTerrains().GPtsTer_By_ImProf().Val().Step();
}


int & cExportSimulation::NbPts()
{
   return GeneratePointsTerrains().GPtsTer_By_ImProf().Val().NbPts();
}

const int & cExportSimulation::NbPts()const 
{
   return GeneratePointsTerrains().GPtsTer_By_ImProf().Val().NbPts();
}


bool & cExportSimulation::OnGrid()
{
   return GeneratePointsTerrains().GPtsTer_By_ImProf().Val().OnGrid();
}

const bool & cExportSimulation::OnGrid()const 
{
   return GeneratePointsTerrains().GPtsTer_By_ImProf().Val().OnGrid();
}


std::string & cExportSimulation::File()
{
   return GeneratePointsTerrains().GPtsTer_By_ImProf().Val().File();
}

const std::string & cExportSimulation::File()const 
{
   return GeneratePointsTerrains().GPtsTer_By_ImProf().Val().File();
}


cTplValGesInit< double > & cExportSimulation::RandomizeInGrid()
{
   return GeneratePointsTerrains().GPtsTer_By_ImProf().Val().RandomizeInGrid();
}

const cTplValGesInit< double > & cExportSimulation::RandomizeInGrid()const 
{
   return GeneratePointsTerrains().GPtsTer_By_ImProf().Val().RandomizeInGrid();
}


cTplValGesInit< std::string > & cExportSimulation::ImMaitresse()
{
   return GeneratePointsTerrains().GPtsTer_By_ImProf().Val().ImMaitresse();
}

const cTplValGesInit< std::string > & cExportSimulation::ImMaitresse()const 
{
   return GeneratePointsTerrains().GPtsTer_By_ImProf().Val().ImMaitresse();
}


cTplValGesInit< bool > & cExportSimulation::DTMIsZ()
{
   return GeneratePointsTerrains().GPtsTer_By_ImProf().Val().DTMIsZ();
}

const cTplValGesInit< bool > & cExportSimulation::DTMIsZ()const 
{
   return GeneratePointsTerrains().GPtsTer_By_ImProf().Val().DTMIsZ();
}


cTplValGesInit< cGPtsTer_By_ImProf > & cExportSimulation::GPtsTer_By_ImProf()
{
   return GeneratePointsTerrains().GPtsTer_By_ImProf();
}

const cTplValGesInit< cGPtsTer_By_ImProf > & cExportSimulation::GPtsTer_By_ImProf()const 
{
   return GeneratePointsTerrains().GPtsTer_By_ImProf();
}


cGeneratePointsTerrains & cExportSimulation::GeneratePointsTerrains()
{
   return mGeneratePointsTerrains;
}

const cGeneratePointsTerrains & cExportSimulation::GeneratePointsTerrains()const 
{
   return mGeneratePointsTerrains;
}


std::list< cGenerateLiaisons > & cExportSimulation::GenerateLiaisons()
{
   return mGenerateLiaisons;
}

const std::list< cGenerateLiaisons > & cExportSimulation::GenerateLiaisons()const 
{
   return mGenerateLiaisons;
}

cElXMLTree * ToXMLTree(const cExportSimulation & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ExportSimulation",eXMLBranche);
   aRes->AddFils(ToXMLTree(anObj.GeneratePointsTerrains())->ReTagThis("GeneratePointsTerrains"));
  for
  (       std::list< cGenerateLiaisons >::const_iterator it=anObj.GenerateLiaisons().begin();
      it !=anObj.GenerateLiaisons().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("GenerateLiaisons"));
  return aRes;
}

void xml_init(cExportSimulation & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.GeneratePointsTerrains(),aTree->Get("GeneratePointsTerrains",1)); //tototo 

   xml_init(anObj.GenerateLiaisons(),aTree->GetAll("GenerateLiaisons",false,1));
}


cTplValGesInit< bool > & cTestInteractif::AvantCompens()
{
   return mAvantCompens;
}

const cTplValGesInit< bool > & cTestInteractif::AvantCompens()const 
{
   return mAvantCompens;
}


cTplValGesInit< bool > & cTestInteractif::ApresCompens()
{
   return mApresCompens;
}

const cTplValGesInit< bool > & cTestInteractif::ApresCompens()const 
{
   return mApresCompens;
}


cTplValGesInit< bool > & cTestInteractif::TestF2C2()
{
   return mTestF2C2;
}

const cTplValGesInit< bool > & cTestInteractif::TestF2C2()const 
{
   return mTestF2C2;
}


cTplValGesInit< bool > & cTestInteractif::SetStepByStep()
{
   return mSetStepByStep;
}

const cTplValGesInit< bool > & cTestInteractif::SetStepByStep()const 
{
   return mSetStepByStep;
}

cElXMLTree * ToXMLTree(const cTestInteractif & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"TestInteractif",eXMLBranche);
   if (anObj.AvantCompens().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("AvantCompens"),anObj.AvantCompens().Val())->ReTagThis("AvantCompens"));
   if (anObj.ApresCompens().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ApresCompens"),anObj.ApresCompens().Val())->ReTagThis("ApresCompens"));
   if (anObj.TestF2C2().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("TestF2C2"),anObj.TestF2C2().Val())->ReTagThis("TestF2C2"));
   if (anObj.SetStepByStep().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SetStepByStep"),anObj.SetStepByStep().Val())->ReTagThis("SetStepByStep"));
  return aRes;
}

void xml_init(cTestInteractif & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.AvantCompens(),aTree->Get("AvantCompens",1),bool(false)); //tototo 

   xml_init(anObj.ApresCompens(),aTree->Get("ApresCompens",1),bool(false)); //tototo 

   xml_init(anObj.TestF2C2(),aTree->Get("TestF2C2",1),bool(false)); //tototo 

   xml_init(anObj.SetStepByStep(),aTree->Get("SetStepByStep",1),bool(false)); //tototo 
}


cTplValGesInit< cSectionLevenbergMarkard > & cIterationsCompensation::SLMIter()
{
   return mSLMIter;
}

const cTplValGesInit< cSectionLevenbergMarkard > & cIterationsCompensation::SLMIter()const 
{
   return mSLMIter;
}


cTplValGesInit< cSectionLevenbergMarkard > & cIterationsCompensation::SLMEtape()
{
   return mSLMEtape;
}

const cTplValGesInit< cSectionLevenbergMarkard > & cIterationsCompensation::SLMEtape()const 
{
   return mSLMEtape;
}


cTplValGesInit< cSectionLevenbergMarkard > & cIterationsCompensation::SLMGlob()
{
   return mSLMGlob;
}

const cTplValGesInit< cSectionLevenbergMarkard > & cIterationsCompensation::SLMGlob()const 
{
   return mSLMGlob;
}


cTplValGesInit< double > & cIterationsCompensation::MultSLMIter()
{
   return mMultSLMIter;
}

const cTplValGesInit< double > & cIterationsCompensation::MultSLMIter()const 
{
   return mMultSLMIter;
}


cTplValGesInit< double > & cIterationsCompensation::MultSLMEtape()
{
   return mMultSLMEtape;
}

const cTplValGesInit< double > & cIterationsCompensation::MultSLMEtape()const 
{
   return mMultSLMEtape;
}


cTplValGesInit< double > & cIterationsCompensation::MultSLMGlob()
{
   return mMultSLMGlob;
}

const cTplValGesInit< double > & cIterationsCompensation::MultSLMGlob()const 
{
   return mMultSLMGlob;
}


std::vector<int> & cIterationsCompensation::ProfMin()
{
   return Pose2Init().Val().ProfMin();
}

const std::vector<int> & cIterationsCompensation::ProfMin()const 
{
   return Pose2Init().Val().ProfMin();
}


cTplValGesInit< bool > & cIterationsCompensation::Show()
{
   return Pose2Init().Val().Show();
}

const cTplValGesInit< bool > & cIterationsCompensation::Show()const 
{
   return Pose2Init().Val().Show();
}


cTplValGesInit< int > & cIterationsCompensation::StepComplemAuto()
{
   return Pose2Init().Val().StepComplemAuto();
}

const cTplValGesInit< int > & cIterationsCompensation::StepComplemAuto()const 
{
   return Pose2Init().Val().StepComplemAuto();
}


cTplValGesInit< cPose2Init > & cIterationsCompensation::Pose2Init()
{
   return mPose2Init;
}

const cTplValGesInit< cPose2Init > & cIterationsCompensation::Pose2Init()const 
{
   return mPose2Init;
}


std::list< cSetRayMaxUtileCalib > & cIterationsCompensation::SetRayMaxUtileCalib()
{
   return mSetRayMaxUtileCalib;
}

const std::list< cSetRayMaxUtileCalib > & cIterationsCompensation::SetRayMaxUtileCalib()const 
{
   return mSetRayMaxUtileCalib;
}


cTplValGesInit< bool > & cIterationsCompensation::AfterCompens()
{
   return BasculeOrientation().Val().AfterCompens();
}

const cTplValGesInit< bool > & cIterationsCompensation::AfterCompens()const 
{
   return BasculeOrientation().Val().AfterCompens();
}


cTplValGesInit< std::string > & cIterationsCompensation::PatternNameApply()
{
   return BasculeOrientation().Val().PatternNameApply();
}

const cTplValGesInit< std::string > & cIterationsCompensation::PatternNameApply()const 
{
   return BasculeOrientation().Val().PatternNameApply();
}


cTplValGesInit< std::string > & cIterationsCompensation::PatternNameEstim()
{
   return BasculeOrientation().Val().PatternNameEstim();
}

const cTplValGesInit< std::string > & cIterationsCompensation::PatternNameEstim()const 
{
   return BasculeOrientation().Val().PatternNameEstim();
}


cTplValGesInit< std::string > & cIterationsCompensation::PoseCentrale()
{
   return BasculeOrientation().Val().ModeBascule().BasculeOnPoints().Val().BascOnCentre().Val().PoseCentrale();
}

const cTplValGesInit< std::string > & cIterationsCompensation::PoseCentrale()const 
{
   return BasculeOrientation().Val().ModeBascule().BasculeOnPoints().Val().BascOnCentre().Val().PoseCentrale();
}


cTplValGesInit< cBascOnCentre > & cIterationsCompensation::BascOnCentre()
{
   return BasculeOrientation().Val().ModeBascule().BasculeOnPoints().Val().BascOnCentre();
}

const cTplValGesInit< cBascOnCentre > & cIterationsCompensation::BascOnCentre()const 
{
   return BasculeOrientation().Val().ModeBascule().BasculeOnPoints().Val().BascOnCentre();
}


std::string & cIterationsCompensation::NameRef()
{
   return BasculeOrientation().Val().ModeBascule().BasculeOnPoints().Val().BascOnAppuis().Val().NameRef();
}

const std::string & cIterationsCompensation::NameRef()const 
{
   return BasculeOrientation().Val().ModeBascule().BasculeOnPoints().Val().BascOnAppuis().Val().NameRef();
}


cTplValGesInit< cBascOnAppuis > & cIterationsCompensation::BascOnAppuis()
{
   return BasculeOrientation().Val().ModeBascule().BasculeOnPoints().Val().BascOnAppuis();
}

const cTplValGesInit< cBascOnAppuis > & cIterationsCompensation::BascOnAppuis()const 
{
   return BasculeOrientation().Val().ModeBascule().BasculeOnPoints().Val().BascOnAppuis();
}


cTplValGesInit< bool > & cIterationsCompensation::ModeL2()
{
   return BasculeOrientation().Val().ModeBascule().BasculeOnPoints().Val().ModeL2();
}

const cTplValGesInit< bool > & cIterationsCompensation::ModeL2()const 
{
   return BasculeOrientation().Val().ModeBascule().BasculeOnPoints().Val().ModeL2();
}


cTplValGesInit< cBasculeOnPoints > & cIterationsCompensation::BasculeOnPoints()
{
   return BasculeOrientation().Val().ModeBascule().BasculeOnPoints();
}

const cTplValGesInit< cBasculeOnPoints > & cIterationsCompensation::BasculeOnPoints()const 
{
   return BasculeOrientation().Val().ModeBascule().BasculeOnPoints();
}


cParamEstimPlan & cIterationsCompensation::EstimPl()
{
   return BasculeOrientation().Val().ModeBascule().BasculeLiaisonOnPlan().Val().EstimPl();
}

const cParamEstimPlan & cIterationsCompensation::EstimPl()const 
{
   return BasculeOrientation().Val().ModeBascule().BasculeLiaisonOnPlan().Val().EstimPl();
}


cTplValGesInit< double > & cIterationsCompensation::DistFixEch()
{
   return BasculeOrientation().Val().ModeBascule().BasculeLiaisonOnPlan().Val().OrientInPlane().Val().DistFixEch();
}

const cTplValGesInit< double > & cIterationsCompensation::DistFixEch()const 
{
   return BasculeOrientation().Val().ModeBascule().BasculeLiaisonOnPlan().Val().OrientInPlane().Val().DistFixEch();
}


std::string & cIterationsCompensation::FileMesures()
{
   return BasculeOrientation().Val().ModeBascule().BasculeLiaisonOnPlan().Val().OrientInPlane().Val().FileMesures();
}

const std::string & cIterationsCompensation::FileMesures()const 
{
   return BasculeOrientation().Val().ModeBascule().BasculeLiaisonOnPlan().Val().OrientInPlane().Val().FileMesures();
}


cTplValGesInit< std::string > & cIterationsCompensation::AlignOn()
{
   return BasculeOrientation().Val().ModeBascule().BasculeLiaisonOnPlan().Val().OrientInPlane().Val().AlignOn();
}

const cTplValGesInit< std::string > & cIterationsCompensation::AlignOn()const 
{
   return BasculeOrientation().Val().ModeBascule().BasculeLiaisonOnPlan().Val().OrientInPlane().Val().AlignOn();
}


cTplValGesInit< cOrientInPlane > & cIterationsCompensation::OrientInPlane()
{
   return BasculeOrientation().Val().ModeBascule().BasculeLiaisonOnPlan().Val().OrientInPlane();
}

const cTplValGesInit< cOrientInPlane > & cIterationsCompensation::OrientInPlane()const 
{
   return BasculeOrientation().Val().ModeBascule().BasculeLiaisonOnPlan().Val().OrientInPlane();
}


cTplValGesInit< cBasculeLiaisonOnPlan > & cIterationsCompensation::BasculeLiaisonOnPlan()
{
   return BasculeOrientation().Val().ModeBascule().BasculeLiaisonOnPlan();
}

const cTplValGesInit< cBasculeLiaisonOnPlan > & cIterationsCompensation::BasculeLiaisonOnPlan()const 
{
   return BasculeOrientation().Val().ModeBascule().BasculeLiaisonOnPlan();
}


cModeBascule & cIterationsCompensation::ModeBascule()
{
   return BasculeOrientation().Val().ModeBascule();
}

const cModeBascule & cIterationsCompensation::ModeBascule()const 
{
   return BasculeOrientation().Val().ModeBascule();
}


cTplValGesInit< cBasculeOrientation > & cIterationsCompensation::BasculeOrientation()
{
   return mBasculeOrientation;
}

const cTplValGesInit< cBasculeOrientation > & cIterationsCompensation::BasculeOrientation()const 
{
   return mBasculeOrientation;
}


std::vector< cAperoPointeStereo > & cIterationsCompensation::HomFE()
{
   return FixeEchelle().Val().ModeFE().StereoFE().Val().HomFE();
}

const std::vector< cAperoPointeStereo > & cIterationsCompensation::HomFE()const 
{
   return FixeEchelle().Val().ModeFE().StereoFE().Val().HomFE();
}


cTplValGesInit< cStereoFE > & cIterationsCompensation::StereoFE()
{
   return FixeEchelle().Val().ModeFE().StereoFE();
}

const cTplValGesInit< cStereoFE > & cIterationsCompensation::StereoFE()const 
{
   return FixeEchelle().Val().ModeFE().StereoFE();
}


cTplValGesInit< cApero2PointeFromFile > & cIterationsCompensation::FEFromFile()
{
   return FixeEchelle().Val().ModeFE().FEFromFile();
}

const cTplValGesInit< cApero2PointeFromFile > & cIterationsCompensation::FEFromFile()const 
{
   return FixeEchelle().Val().ModeFE().FEFromFile();
}


cModeFE & cIterationsCompensation::ModeFE()
{
   return FixeEchelle().Val().ModeFE();
}

const cModeFE & cIterationsCompensation::ModeFE()const 
{
   return FixeEchelle().Val().ModeFE();
}


double & cIterationsCompensation::DistVraie()
{
   return FixeEchelle().Val().DistVraie();
}

const double & cIterationsCompensation::DistVraie()const 
{
   return FixeEchelle().Val().DistVraie();
}


cTplValGesInit< cFixeEchelle > & cIterationsCompensation::FixeEchelle()
{
   return mFixeEchelle;
}

const cTplValGesInit< cFixeEchelle > & cIterationsCompensation::FixeEchelle()const 
{
   return mFixeEchelle;
}


std::vector< cAperoPointeMono > & cIterationsCompensation::VecFOH()
{
   return FixeOrientPlane().Val().ModeFOP().HorFOP().Val().VecFOH();
}

const std::vector< cAperoPointeMono > & cIterationsCompensation::VecFOH()const 
{
   return FixeOrientPlane().Val().ModeFOP().HorFOP().Val().VecFOH();
}


cTplValGesInit< double > & cIterationsCompensation::Z()
{
   return FixeOrientPlane().Val().ModeFOP().HorFOP().Val().Z();
}

const cTplValGesInit< double > & cIterationsCompensation::Z()const 
{
   return FixeOrientPlane().Val().ModeFOP().HorFOP().Val().Z();
}


cTplValGesInit< cHorFOP > & cIterationsCompensation::HorFOP()
{
   return FixeOrientPlane().Val().ModeFOP().HorFOP();
}

const cTplValGesInit< cHorFOP > & cIterationsCompensation::HorFOP()const 
{
   return FixeOrientPlane().Val().ModeFOP().HorFOP();
}


cTplValGesInit< cApero2PointeFromFile > & cIterationsCompensation::HorFromFile()
{
   return FixeOrientPlane().Val().ModeFOP().HorFromFile();
}

const cTplValGesInit< cApero2PointeFromFile > & cIterationsCompensation::HorFromFile()const 
{
   return FixeOrientPlane().Val().ModeFOP().HorFromFile();
}


cModeFOP & cIterationsCompensation::ModeFOP()
{
   return FixeOrientPlane().Val().ModeFOP();
}

const cModeFOP & cIterationsCompensation::ModeFOP()const 
{
   return FixeOrientPlane().Val().ModeFOP();
}


Pt2dr & cIterationsCompensation::Vecteur()
{
   return FixeOrientPlane().Val().Vecteur();
}

const Pt2dr & cIterationsCompensation::Vecteur()const 
{
   return FixeOrientPlane().Val().Vecteur();
}


cTplValGesInit< cFixeOrientPlane > & cIterationsCompensation::FixeOrientPlane()
{
   return mFixeOrientPlane;
}

const cTplValGesInit< cFixeOrientPlane > & cIterationsCompensation::FixeOrientPlane()const 
{
   return mFixeOrientPlane;
}


cTplValGesInit< std::string > & cIterationsCompensation::BasicOrPl()
{
   return mBasicOrPl;
}

const cTplValGesInit< std::string > & cIterationsCompensation::BasicOrPl()const 
{
   return mBasicOrPl;
}


std::string & cIterationsCompensation::Pattern1()
{
   return BlocBascule().Val().Pattern1();
}

const std::string & cIterationsCompensation::Pattern1()const 
{
   return BlocBascule().Val().Pattern1();
}


std::string & cIterationsCompensation::Pattern2()
{
   return BlocBascule().Val().Pattern2();
}

const std::string & cIterationsCompensation::Pattern2()const 
{
   return BlocBascule().Val().Pattern2();
}


std::string & cIterationsCompensation::IdBdl()
{
   return BlocBascule().Val().IdBdl();
}

const std::string & cIterationsCompensation::IdBdl()const 
{
   return BlocBascule().Val().IdBdl();
}


cTplValGesInit< cBlocBascule > & cIterationsCompensation::BlocBascule()
{
   return mBlocBascule;
}

const cTplValGesInit< cBlocBascule > & cIterationsCompensation::BlocBascule()const 
{
   return mBlocBascule;
}


int & cIterationsCompensation::Periode()
{
   return MesureErreurTournante().Val().Periode();
}

const int & cIterationsCompensation::Periode()const 
{
   return MesureErreurTournante().Val().Periode();
}


cTplValGesInit< int > & cIterationsCompensation::NbTest()
{
   return MesureErreurTournante().Val().NbTest();
}

const cTplValGesInit< int > & cIterationsCompensation::NbTest()const 
{
   return MesureErreurTournante().Val().NbTest();
}


cTplValGesInit< int > & cIterationsCompensation::NbIter()
{
   return MesureErreurTournante().Val().NbIter();
}

const cTplValGesInit< int > & cIterationsCompensation::NbIter()const 
{
   return MesureErreurTournante().Val().NbIter();
}


cTplValGesInit< bool > & cIterationsCompensation::ApplyAppuis()
{
   return MesureErreurTournante().Val().ApplyAppuis();
}

const cTplValGesInit< bool > & cIterationsCompensation::ApplyAppuis()const 
{
   return MesureErreurTournante().Val().ApplyAppuis();
}


cTplValGesInit< bool > & cIterationsCompensation::ApplyLiaisons()
{
   return MesureErreurTournante().Val().ApplyLiaisons();
}

const cTplValGesInit< bool > & cIterationsCompensation::ApplyLiaisons()const 
{
   return MesureErreurTournante().Val().ApplyLiaisons();
}


cTplValGesInit< cMesureErreurTournante > & cIterationsCompensation::MesureErreurTournante()
{
   return mMesureErreurTournante;
}

const cTplValGesInit< cMesureErreurTournante > & cIterationsCompensation::MesureErreurTournante()const 
{
   return mMesureErreurTournante;
}


std::list< cContraintesCamerasInc > & cIterationsCompensation::ContraintesCamerasInc()
{
   return SectionContraintes().Val().ContraintesCamerasInc();
}

const std::list< cContraintesCamerasInc > & cIterationsCompensation::ContraintesCamerasInc()const 
{
   return SectionContraintes().Val().ContraintesCamerasInc();
}


std::list< cContraintesPoses > & cIterationsCompensation::ContraintesPoses()
{
   return SectionContraintes().Val().ContraintesPoses();
}

const std::list< cContraintesPoses > & cIterationsCompensation::ContraintesPoses()const 
{
   return SectionContraintes().Val().ContraintesPoses();
}


cTplValGesInit< cSectionContraintes > & cIterationsCompensation::SectionContraintes()
{
   return mSectionContraintes;
}

const cTplValGesInit< cSectionContraintes > & cIterationsCompensation::SectionContraintes()const 
{
   return mSectionContraintes;
}


std::list< std::string > & cIterationsCompensation::Messages()
{
   return mMessages;
}

const std::list< std::string > & cIterationsCompensation::Messages()const 
{
   return mMessages;
}


std::list< cVisuPtsMult > & cIterationsCompensation::VisuPtsMult()
{
   return mVisuPtsMult;
}

const std::list< cVisuPtsMult > & cIterationsCompensation::VisuPtsMult()const 
{
   return mVisuPtsMult;
}


std::list< cVerifAero > & cIterationsCompensation::VerifAero()
{
   return mVerifAero;
}

const std::list< cVerifAero > & cIterationsCompensation::VerifAero()const 
{
   return mVerifAero;
}


std::list< cExportSimulation > & cIterationsCompensation::ExportSimulation()
{
   return mExportSimulation;
}

const std::list< cExportSimulation > & cIterationsCompensation::ExportSimulation()const 
{
   return mExportSimulation;
}


cTplValGesInit< bool > & cIterationsCompensation::AvantCompens()
{
   return TestInteractif().Val().AvantCompens();
}

const cTplValGesInit< bool > & cIterationsCompensation::AvantCompens()const 
{
   return TestInteractif().Val().AvantCompens();
}


cTplValGesInit< bool > & cIterationsCompensation::ApresCompens()
{
   return TestInteractif().Val().ApresCompens();
}

const cTplValGesInit< bool > & cIterationsCompensation::ApresCompens()const 
{
   return TestInteractif().Val().ApresCompens();
}


cTplValGesInit< bool > & cIterationsCompensation::TestF2C2()
{
   return TestInteractif().Val().TestF2C2();
}

const cTplValGesInit< bool > & cIterationsCompensation::TestF2C2()const 
{
   return TestInteractif().Val().TestF2C2();
}


cTplValGesInit< bool > & cIterationsCompensation::SetStepByStep()
{
   return TestInteractif().Val().SetStepByStep();
}

const cTplValGesInit< bool > & cIterationsCompensation::SetStepByStep()const 
{
   return TestInteractif().Val().SetStepByStep();
}


cTplValGesInit< cTestInteractif > & cIterationsCompensation::TestInteractif()
{
   return mTestInteractif;
}

const cTplValGesInit< cTestInteractif > & cIterationsCompensation::TestInteractif()const 
{
   return mTestInteractif;
}

cElXMLTree * ToXMLTree(const cIterationsCompensation & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"IterationsCompensation",eXMLBranche);
   if (anObj.SLMIter().IsInit())
      aRes->AddFils(ToXMLTree(anObj.SLMIter().Val())->ReTagThis("SLMIter"));
   if (anObj.SLMEtape().IsInit())
      aRes->AddFils(ToXMLTree(anObj.SLMEtape().Val())->ReTagThis("SLMEtape"));
   if (anObj.SLMGlob().IsInit())
      aRes->AddFils(ToXMLTree(anObj.SLMGlob().Val())->ReTagThis("SLMGlob"));
   if (anObj.MultSLMIter().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("MultSLMIter"),anObj.MultSLMIter().Val())->ReTagThis("MultSLMIter"));
   if (anObj.MultSLMEtape().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("MultSLMEtape"),anObj.MultSLMEtape().Val())->ReTagThis("MultSLMEtape"));
   if (anObj.MultSLMGlob().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("MultSLMGlob"),anObj.MultSLMGlob().Val())->ReTagThis("MultSLMGlob"));
   if (anObj.Pose2Init().IsInit())
      aRes->AddFils(ToXMLTree(anObj.Pose2Init().Val())->ReTagThis("Pose2Init"));
  for
  (       std::list< cSetRayMaxUtileCalib >::const_iterator it=anObj.SetRayMaxUtileCalib().begin();
      it !=anObj.SetRayMaxUtileCalib().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("SetRayMaxUtileCalib"));
   if (anObj.BasculeOrientation().IsInit())
      aRes->AddFils(ToXMLTree(anObj.BasculeOrientation().Val())->ReTagThis("BasculeOrientation"));
   if (anObj.FixeEchelle().IsInit())
      aRes->AddFils(ToXMLTree(anObj.FixeEchelle().Val())->ReTagThis("FixeEchelle"));
   if (anObj.FixeOrientPlane().IsInit())
      aRes->AddFils(ToXMLTree(anObj.FixeOrientPlane().Val())->ReTagThis("FixeOrientPlane"));
   if (anObj.BasicOrPl().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("BasicOrPl"),anObj.BasicOrPl().Val())->ReTagThis("BasicOrPl"));
   if (anObj.BlocBascule().IsInit())
      aRes->AddFils(ToXMLTree(anObj.BlocBascule().Val())->ReTagThis("BlocBascule"));
   if (anObj.MesureErreurTournante().IsInit())
      aRes->AddFils(ToXMLTree(anObj.MesureErreurTournante().Val())->ReTagThis("MesureErreurTournante"));
   if (anObj.SectionContraintes().IsInit())
      aRes->AddFils(ToXMLTree(anObj.SectionContraintes().Val())->ReTagThis("SectionContraintes"));
  for
  (       std::list< std::string >::const_iterator it=anObj.Messages().begin();
      it !=anObj.Messages().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("Messages"),(*it))->ReTagThis("Messages"));
  for
  (       std::list< cVisuPtsMult >::const_iterator it=anObj.VisuPtsMult().begin();
      it !=anObj.VisuPtsMult().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("VisuPtsMult"));
  for
  (       std::list< cVerifAero >::const_iterator it=anObj.VerifAero().begin();
      it !=anObj.VerifAero().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("VerifAero"));
  for
  (       std::list< cExportSimulation >::const_iterator it=anObj.ExportSimulation().begin();
      it !=anObj.ExportSimulation().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("ExportSimulation"));
   if (anObj.TestInteractif().IsInit())
      aRes->AddFils(ToXMLTree(anObj.TestInteractif().Val())->ReTagThis("TestInteractif"));
  return aRes;
}

void xml_init(cIterationsCompensation & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.SLMIter(),aTree->Get("SLMIter",1)); //tototo 

   xml_init(anObj.SLMEtape(),aTree->Get("SLMEtape",1)); //tototo 

   xml_init(anObj.SLMGlob(),aTree->Get("SLMGlob",1)); //tototo 

   xml_init(anObj.MultSLMIter(),aTree->Get("MultSLMIter",1)); //tototo 

   xml_init(anObj.MultSLMEtape(),aTree->Get("MultSLMEtape",1)); //tototo 

   xml_init(anObj.MultSLMGlob(),aTree->Get("MultSLMGlob",1)); //tototo 

   xml_init(anObj.Pose2Init(),aTree->Get("Pose2Init",1)); //tototo 

   xml_init(anObj.SetRayMaxUtileCalib(),aTree->GetAll("SetRayMaxUtileCalib",false,1));

   xml_init(anObj.BasculeOrientation(),aTree->Get("BasculeOrientation",1)); //tototo 

   xml_init(anObj.FixeEchelle(),aTree->Get("FixeEchelle",1)); //tototo 

   xml_init(anObj.FixeOrientPlane(),aTree->Get("FixeOrientPlane",1)); //tototo 

   xml_init(anObj.BasicOrPl(),aTree->Get("BasicOrPl",1)); //tototo 

   xml_init(anObj.BlocBascule(),aTree->Get("BlocBascule",1)); //tototo 

   xml_init(anObj.MesureErreurTournante(),aTree->Get("MesureErreurTournante",1)); //tototo 

   xml_init(anObj.SectionContraintes(),aTree->Get("SectionContraintes",1)); //tototo 

   xml_init(anObj.Messages(),aTree->GetAll("Messages",false,1));

   xml_init(anObj.VisuPtsMult(),aTree->GetAll("VisuPtsMult",false,1));

   xml_init(anObj.VerifAero(),aTree->GetAll("VerifAero",false,1));

   xml_init(anObj.ExportSimulation(),aTree->GetAll("ExportSimulation",false,1));

   xml_init(anObj.TestInteractif(),aTree->Get("TestInteractif",1)); //tototo 
}


std::string & cTraceCpleHom::Id()
{
   return mId;
}

const std::string & cTraceCpleHom::Id()const 
{
   return mId;
}

cElXMLTree * ToXMLTree(const cTraceCpleHom & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"TraceCpleHom",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Id"),anObj.Id())->ReTagThis("Id"));
  return aRes;
}

void xml_init(cTraceCpleHom & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.Id(),aTree->Get("Id",1)); //tototo 
}


std::string & cTraceCpleCam::Cam1()
{
   return mCam1;
}

const std::string & cTraceCpleCam::Cam1()const 
{
   return mCam1;
}


std::string & cTraceCpleCam::Cam2()
{
   return mCam2;
}

const std::string & cTraceCpleCam::Cam2()const 
{
   return mCam2;
}


std::list< cTraceCpleHom > & cTraceCpleCam::TraceCpleHom()
{
   return mTraceCpleHom;
}

const std::list< cTraceCpleHom > & cTraceCpleCam::TraceCpleHom()const 
{
   return mTraceCpleHom;
}

cElXMLTree * ToXMLTree(const cTraceCpleCam & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"TraceCpleCam",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Cam1"),anObj.Cam1())->ReTagThis("Cam1"));
   aRes->AddFils(::ToXMLTree(std::string("Cam2"),anObj.Cam2())->ReTagThis("Cam2"));
  for
  (       std::list< cTraceCpleHom >::const_iterator it=anObj.TraceCpleHom().begin();
      it !=anObj.TraceCpleHom().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("TraceCpleHom"));
  return aRes;
}

void xml_init(cTraceCpleCam & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.Cam1(),aTree->Get("Cam1",1)); //tototo 

   xml_init(anObj.Cam2(),aTree->Get("Cam2",1)); //tototo 

   xml_init(anObj.TraceCpleHom(),aTree->GetAll("TraceCpleHom",false,1));
}


std::list< cTraceCpleCam > & cSectionTracage::TraceCpleCam()
{
   return mTraceCpleCam;
}

const std::list< cTraceCpleCam > & cSectionTracage::TraceCpleCam()const 
{
   return mTraceCpleCam;
}


cTplValGesInit< bool > & cSectionTracage::GetChar()
{
   return mGetChar;
}

const cTplValGesInit< bool > & cSectionTracage::GetChar()const 
{
   return mGetChar;
}

cElXMLTree * ToXMLTree(const cSectionTracage & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"SectionTracage",eXMLBranche);
  for
  (       std::list< cTraceCpleCam >::const_iterator it=anObj.TraceCpleCam().begin();
      it !=anObj.TraceCpleCam().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("TraceCpleCam"));
   if (anObj.GetChar().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("GetChar"),anObj.GetChar().Val())->ReTagThis("GetChar"));
  return aRes;
}

void xml_init(cSectionTracage & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.TraceCpleCam(),aTree->GetAll("TraceCpleCam",false,1));

   xml_init(anObj.GetChar(),aTree->Get("GetChar",1),bool(true)); //tototo 
}


std::string & cROA_FichierImg::Name()
{
   return mName;
}

const std::string & cROA_FichierImg::Name()const 
{
   return mName;
}


double & cROA_FichierImg::Sz()
{
   return mSz;
}

const double & cROA_FichierImg::Sz()const 
{
   return mSz;
}


double & cROA_FichierImg::Exag()
{
   return mExag;
}

const double & cROA_FichierImg::Exag()const 
{
   return mExag;
}


cTplValGesInit< bool > & cROA_FichierImg::VisuVideo()
{
   return mVisuVideo;
}

const cTplValGesInit< bool > & cROA_FichierImg::VisuVideo()const 
{
   return mVisuVideo;
}

cElXMLTree * ToXMLTree(const cROA_FichierImg & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ROA_FichierImg",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Name"),anObj.Name())->ReTagThis("Name"));
   aRes->AddFils(::ToXMLTree(std::string("Sz"),anObj.Sz())->ReTagThis("Sz"));
   aRes->AddFils(::ToXMLTree(std::string("Exag"),anObj.Exag())->ReTagThis("Exag"));
   if (anObj.VisuVideo().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("VisuVideo"),anObj.VisuVideo().Val())->ReTagThis("VisuVideo"));
  return aRes;
}

void xml_init(cROA_FichierImg & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.Name(),aTree->Get("Name",1)); //tototo 

   xml_init(anObj.Sz(),aTree->Get("Sz",1)); //tototo 

   xml_init(anObj.Exag(),aTree->Get("Exag",1)); //tototo 

   xml_init(anObj.VisuVideo(),aTree->Get("VisuVideo",1),bool(false)); //tototo 
}


cTplValGesInit< bool > & cRapportObsAppui::OnlyLastIter()
{
   return mOnlyLastIter;
}

const cTplValGesInit< bool > & cRapportObsAppui::OnlyLastIter()const 
{
   return mOnlyLastIter;
}


std::string & cRapportObsAppui::FichierTxt()
{
   return mFichierTxt;
}

const std::string & cRapportObsAppui::FichierTxt()const 
{
   return mFichierTxt;
}


cTplValGesInit< bool > & cRapportObsAppui::ColPerPose()
{
   return mColPerPose;
}

const cTplValGesInit< bool > & cRapportObsAppui::ColPerPose()const 
{
   return mColPerPose;
}


cTplValGesInit< double > & cRapportObsAppui::SeuilColOut()
{
   return mSeuilColOut;
}

const cTplValGesInit< double > & cRapportObsAppui::SeuilColOut()const 
{
   return mSeuilColOut;
}


std::string & cRapportObsAppui::Name()
{
   return ROA_FichierImg().Val().Name();
}

const std::string & cRapportObsAppui::Name()const 
{
   return ROA_FichierImg().Val().Name();
}


double & cRapportObsAppui::Sz()
{
   return ROA_FichierImg().Val().Sz();
}

const double & cRapportObsAppui::Sz()const 
{
   return ROA_FichierImg().Val().Sz();
}


double & cRapportObsAppui::Exag()
{
   return ROA_FichierImg().Val().Exag();
}

const double & cRapportObsAppui::Exag()const 
{
   return ROA_FichierImg().Val().Exag();
}


cTplValGesInit< bool > & cRapportObsAppui::VisuVideo()
{
   return ROA_FichierImg().Val().VisuVideo();
}

const cTplValGesInit< bool > & cRapportObsAppui::VisuVideo()const 
{
   return ROA_FichierImg().Val().VisuVideo();
}


cTplValGesInit< cROA_FichierImg > & cRapportObsAppui::ROA_FichierImg()
{
   return mROA_FichierImg;
}

const cTplValGesInit< cROA_FichierImg > & cRapportObsAppui::ROA_FichierImg()const 
{
   return mROA_FichierImg;
}

cElXMLTree * ToXMLTree(const cRapportObsAppui & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"RapportObsAppui",eXMLBranche);
   if (anObj.OnlyLastIter().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("OnlyLastIter"),anObj.OnlyLastIter().Val())->ReTagThis("OnlyLastIter"));
   aRes->AddFils(::ToXMLTree(std::string("FichierTxt"),anObj.FichierTxt())->ReTagThis("FichierTxt"));
   if (anObj.ColPerPose().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ColPerPose"),anObj.ColPerPose().Val())->ReTagThis("ColPerPose"));
   if (anObj.SeuilColOut().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SeuilColOut"),anObj.SeuilColOut().Val())->ReTagThis("SeuilColOut"));
   if (anObj.ROA_FichierImg().IsInit())
      aRes->AddFils(ToXMLTree(anObj.ROA_FichierImg().Val())->ReTagThis("ROA_FichierImg"));
  return aRes;
}

void xml_init(cRapportObsAppui & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.OnlyLastIter(),aTree->Get("OnlyLastIter",1),bool(true)); //tototo 

   xml_init(anObj.FichierTxt(),aTree->Get("FichierTxt",1)); //tototo 

   xml_init(anObj.ColPerPose(),aTree->Get("ColPerPose",1),bool(false)); //tototo 

   xml_init(anObj.SeuilColOut(),aTree->Get("SeuilColOut",1),double(1e3)); //tototo 

   xml_init(anObj.ROA_FichierImg(),aTree->Get("ROA_FichierImg",1)); //tototo 
}


std::string & cObsAppuis::NameRef()
{
   return mNameRef;
}

const std::string & cObsAppuis::NameRef()const 
{
   return mNameRef;
}


cPonderationPackMesure & cObsAppuis::Pond()
{
   return mPond;
}

const cPonderationPackMesure & cObsAppuis::Pond()const 
{
   return mPond;
}


cTplValGesInit< bool > & cObsAppuis::OnlyLastIter()
{
   return RapportObsAppui().Val().OnlyLastIter();
}

const cTplValGesInit< bool > & cObsAppuis::OnlyLastIter()const 
{
   return RapportObsAppui().Val().OnlyLastIter();
}


std::string & cObsAppuis::FichierTxt()
{
   return RapportObsAppui().Val().FichierTxt();
}

const std::string & cObsAppuis::FichierTxt()const 
{
   return RapportObsAppui().Val().FichierTxt();
}


cTplValGesInit< bool > & cObsAppuis::ColPerPose()
{
   return RapportObsAppui().Val().ColPerPose();
}

const cTplValGesInit< bool > & cObsAppuis::ColPerPose()const 
{
   return RapportObsAppui().Val().ColPerPose();
}


cTplValGesInit< double > & cObsAppuis::SeuilColOut()
{
   return RapportObsAppui().Val().SeuilColOut();
}

const cTplValGesInit< double > & cObsAppuis::SeuilColOut()const 
{
   return RapportObsAppui().Val().SeuilColOut();
}


std::string & cObsAppuis::Name()
{
   return RapportObsAppui().Val().ROA_FichierImg().Val().Name();
}

const std::string & cObsAppuis::Name()const 
{
   return RapportObsAppui().Val().ROA_FichierImg().Val().Name();
}


double & cObsAppuis::Sz()
{
   return RapportObsAppui().Val().ROA_FichierImg().Val().Sz();
}

const double & cObsAppuis::Sz()const 
{
   return RapportObsAppui().Val().ROA_FichierImg().Val().Sz();
}


double & cObsAppuis::Exag()
{
   return RapportObsAppui().Val().ROA_FichierImg().Val().Exag();
}

const double & cObsAppuis::Exag()const 
{
   return RapportObsAppui().Val().ROA_FichierImg().Val().Exag();
}


cTplValGesInit< bool > & cObsAppuis::VisuVideo()
{
   return RapportObsAppui().Val().ROA_FichierImg().Val().VisuVideo();
}

const cTplValGesInit< bool > & cObsAppuis::VisuVideo()const 
{
   return RapportObsAppui().Val().ROA_FichierImg().Val().VisuVideo();
}


cTplValGesInit< cROA_FichierImg > & cObsAppuis::ROA_FichierImg()
{
   return RapportObsAppui().Val().ROA_FichierImg();
}

const cTplValGesInit< cROA_FichierImg > & cObsAppuis::ROA_FichierImg()const 
{
   return RapportObsAppui().Val().ROA_FichierImg();
}


cTplValGesInit< cRapportObsAppui > & cObsAppuis::RapportObsAppui()
{
   return mRapportObsAppui;
}

const cTplValGesInit< cRapportObsAppui > & cObsAppuis::RapportObsAppui()const 
{
   return mRapportObsAppui;
}

cElXMLTree * ToXMLTree(const cObsAppuis & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ObsAppuis",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("NameRef"),anObj.NameRef())->ReTagThis("NameRef"));
   aRes->AddFils(ToXMLTree(anObj.Pond())->ReTagThis("Pond"));
   if (anObj.RapportObsAppui().IsInit())
      aRes->AddFils(ToXMLTree(anObj.RapportObsAppui().Val())->ReTagThis("RapportObsAppui"));
  return aRes;
}

void xml_init(cObsAppuis & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.NameRef(),aTree->Get("NameRef",1)); //tototo 

   xml_init(anObj.Pond(),aTree->Get("Pond",1)); //tototo 

   xml_init(anObj.RapportObsAppui(),aTree->Get("RapportObsAppui",1)); //tototo 
}


std::string & cObsAppuisFlottant::NameRef()
{
   return mNameRef;
}

const std::string & cObsAppuisFlottant::NameRef()const 
{
   return mNameRef;
}


cPonderationPackMesure & cObsAppuisFlottant::PondIm()
{
   return mPondIm;
}

const cPonderationPackMesure & cObsAppuisFlottant::PondIm()const 
{
   return mPondIm;
}


std::list< cElRegex_Ptr > & cObsAppuisFlottant::PtsShowDet()
{
   return mPtsShowDet;
}

const std::list< cElRegex_Ptr > & cObsAppuisFlottant::PtsShowDet()const 
{
   return mPtsShowDet;
}


cTplValGesInit< bool > & cObsAppuisFlottant::DetShow3D()
{
   return mDetShow3D;
}

const cTplValGesInit< bool > & cObsAppuisFlottant::DetShow3D()const 
{
   return mDetShow3D;
}


cTplValGesInit< double > & cObsAppuisFlottant::NivAlerteDetail()
{
   return mNivAlerteDetail;
}

const cTplValGesInit< double > & cObsAppuisFlottant::NivAlerteDetail()const 
{
   return mNivAlerteDetail;
}


cTplValGesInit< bool > & cObsAppuisFlottant::ShowMax()
{
   return mShowMax;
}

const cTplValGesInit< bool > & cObsAppuisFlottant::ShowMax()const 
{
   return mShowMax;
}


cTplValGesInit< bool > & cObsAppuisFlottant::ShowSom()
{
   return mShowSom;
}

const cTplValGesInit< bool > & cObsAppuisFlottant::ShowSom()const 
{
   return mShowSom;
}

cElXMLTree * ToXMLTree(const cObsAppuisFlottant & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ObsAppuisFlottant",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("NameRef"),anObj.NameRef())->ReTagThis("NameRef"));
   aRes->AddFils(ToXMLTree(anObj.PondIm())->ReTagThis("PondIm"));
  for
  (       std::list< cElRegex_Ptr >::const_iterator it=anObj.PtsShowDet().begin();
      it !=anObj.PtsShowDet().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("PtsShowDet"),(*it))->ReTagThis("PtsShowDet"));
   if (anObj.DetShow3D().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DetShow3D"),anObj.DetShow3D().Val())->ReTagThis("DetShow3D"));
   if (anObj.NivAlerteDetail().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NivAlerteDetail"),anObj.NivAlerteDetail().Val())->ReTagThis("NivAlerteDetail"));
   if (anObj.ShowMax().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ShowMax"),anObj.ShowMax().Val())->ReTagThis("ShowMax"));
   if (anObj.ShowSom().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ShowSom"),anObj.ShowSom().Val())->ReTagThis("ShowSom"));
  return aRes;
}

void xml_init(cObsAppuisFlottant & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.NameRef(),aTree->Get("NameRef",1)); //tototo 

   xml_init(anObj.PondIm(),aTree->Get("PondIm",1)); //tototo 

   xml_init(anObj.PtsShowDet(),aTree->GetAll("PtsShowDet",false,1));

   xml_init(anObj.DetShow3D(),aTree->Get("DetShow3D",1),bool(false)); //tototo 

   xml_init(anObj.NivAlerteDetail(),aTree->Get("NivAlerteDetail",1),double(1e9)); //tototo 

   xml_init(anObj.ShowMax(),aTree->Get("ShowMax",1),bool(false)); //tototo 

   xml_init(anObj.ShowSom(),aTree->Get("ShowSom",1),bool(false)); //tototo 
}


double & cRappelOnZ::Z()
{
   return mZ;
}

const double & cRappelOnZ::Z()const 
{
   return mZ;
}


double & cRappelOnZ::IncC()
{
   return mIncC;
}

const double & cRappelOnZ::IncC()const 
{
   return mIncC;
}


cTplValGesInit< double > & cRappelOnZ::IncE()
{
   return mIncE;
}

const cTplValGesInit< double > & cRappelOnZ::IncE()const 
{
   return mIncE;
}


cTplValGesInit< double > & cRappelOnZ::SeuilR()
{
   return mSeuilR;
}

const cTplValGesInit< double > & cRappelOnZ::SeuilR()const 
{
   return mSeuilR;
}


cTplValGesInit< std::string > & cRappelOnZ::LayerMasq()
{
   return mLayerMasq;
}

const cTplValGesInit< std::string > & cRappelOnZ::LayerMasq()const 
{
   return mLayerMasq;
}

cElXMLTree * ToXMLTree(const cRappelOnZ & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"RappelOnZ",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Z"),anObj.Z())->ReTagThis("Z"));
   aRes->AddFils(::ToXMLTree(std::string("IncC"),anObj.IncC())->ReTagThis("IncC"));
   if (anObj.IncE().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("IncE"),anObj.IncE().Val())->ReTagThis("IncE"));
   if (anObj.SeuilR().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SeuilR"),anObj.SeuilR().Val())->ReTagThis("SeuilR"));
   if (anObj.LayerMasq().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("LayerMasq"),anObj.LayerMasq().Val())->ReTagThis("LayerMasq"));
  return aRes;
}

void xml_init(cRappelOnZ & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.Z(),aTree->Get("Z",1)); //tototo 

   xml_init(anObj.IncC(),aTree->Get("IncC",1)); //tototo 

   xml_init(anObj.IncE(),aTree->Get("IncE",1)); //tototo 

   xml_init(anObj.SeuilR(),aTree->Get("SeuilR",1)); //tototo 

   xml_init(anObj.LayerMasq(),aTree->Get("LayerMasq",1)); //tototo 
}


std::string & cObsLiaisons::NameRef()
{
   return mNameRef;
}

const std::string & cObsLiaisons::NameRef()const 
{
   return mNameRef;
}


cPonderationPackMesure & cObsLiaisons::Pond()
{
   return mPond;
}

const cPonderationPackMesure & cObsLiaisons::Pond()const 
{
   return mPond;
}


cTplValGesInit< cPonderationPackMesure > & cObsLiaisons::PondSurf()
{
   return mPondSurf;
}

const cTplValGesInit< cPonderationPackMesure > & cObsLiaisons::PondSurf()const 
{
   return mPondSurf;
}


double & cObsLiaisons::Z()
{
   return RappelOnZ().Val().Z();
}

const double & cObsLiaisons::Z()const 
{
   return RappelOnZ().Val().Z();
}


double & cObsLiaisons::IncC()
{
   return RappelOnZ().Val().IncC();
}

const double & cObsLiaisons::IncC()const 
{
   return RappelOnZ().Val().IncC();
}


cTplValGesInit< double > & cObsLiaisons::IncE()
{
   return RappelOnZ().Val().IncE();
}

const cTplValGesInit< double > & cObsLiaisons::IncE()const 
{
   return RappelOnZ().Val().IncE();
}


cTplValGesInit< double > & cObsLiaisons::SeuilR()
{
   return RappelOnZ().Val().SeuilR();
}

const cTplValGesInit< double > & cObsLiaisons::SeuilR()const 
{
   return RappelOnZ().Val().SeuilR();
}


cTplValGesInit< std::string > & cObsLiaisons::LayerMasq()
{
   return RappelOnZ().Val().LayerMasq();
}

const cTplValGesInit< std::string > & cObsLiaisons::LayerMasq()const 
{
   return RappelOnZ().Val().LayerMasq();
}


cTplValGesInit< cRappelOnZ > & cObsLiaisons::RappelOnZ()
{
   return mRappelOnZ;
}

const cTplValGesInit< cRappelOnZ > & cObsLiaisons::RappelOnZ()const 
{
   return mRappelOnZ;
}

cElXMLTree * ToXMLTree(const cObsLiaisons & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ObsLiaisons",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("NameRef"),anObj.NameRef())->ReTagThis("NameRef"));
   aRes->AddFils(ToXMLTree(anObj.Pond())->ReTagThis("Pond"));
   if (anObj.PondSurf().IsInit())
      aRes->AddFils(ToXMLTree(anObj.PondSurf().Val())->ReTagThis("PondSurf"));
   if (anObj.RappelOnZ().IsInit())
      aRes->AddFils(ToXMLTree(anObj.RappelOnZ().Val())->ReTagThis("RappelOnZ"));
  return aRes;
}

void xml_init(cObsLiaisons & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.NameRef(),aTree->Get("NameRef",1)); //tototo 

   xml_init(anObj.Pond(),aTree->Get("Pond",1)); //tototo 

   xml_init(anObj.PondSurf(),aTree->Get("PondSurf",1)); //tototo 

   xml_init(anObj.RappelOnZ(),aTree->Get("RappelOnZ",1)); //tototo 
}


cTplValGesInit< cElRegex_Ptr > & cObsCentrePDV::PatternApply()
{
   return mPatternApply;
}

const cTplValGesInit< cElRegex_Ptr > & cObsCentrePDV::PatternApply()const 
{
   return mPatternApply;
}


cPonderationPackMesure & cObsCentrePDV::Pond()
{
   return mPond;
}

const cPonderationPackMesure & cObsCentrePDV::Pond()const 
{
   return mPond;
}


cTplValGesInit< cPonderationPackMesure > & cObsCentrePDV::PondAlti()
{
   return mPondAlti;
}

const cTplValGesInit< cPonderationPackMesure > & cObsCentrePDV::PondAlti()const 
{
   return mPondAlti;
}


cTplValGesInit< bool > & cObsCentrePDV::ShowTestVitesse()
{
   return mShowTestVitesse;
}

const cTplValGesInit< bool > & cObsCentrePDV::ShowTestVitesse()const 
{
   return mShowTestVitesse;
}

cElXMLTree * ToXMLTree(const cObsCentrePDV & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ObsCentrePDV",eXMLBranche);
   if (anObj.PatternApply().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PatternApply"),anObj.PatternApply().Val())->ReTagThis("PatternApply"));
   aRes->AddFils(ToXMLTree(anObj.Pond())->ReTagThis("Pond"));
   if (anObj.PondAlti().IsInit())
      aRes->AddFils(ToXMLTree(anObj.PondAlti().Val())->ReTagThis("PondAlti"));
   if (anObj.ShowTestVitesse().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ShowTestVitesse"),anObj.ShowTestVitesse().Val())->ReTagThis("ShowTestVitesse"));
  return aRes;
}

void xml_init(cObsCentrePDV & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.PatternApply(),aTree->Get("PatternApply",1)); //tototo 

   xml_init(anObj.Pond(),aTree->Get("Pond",1)); //tototo 

   xml_init(anObj.PondAlti(),aTree->Get("PondAlti",1)); //tototo 

   xml_init(anObj.ShowTestVitesse(),aTree->Get("ShowTestVitesse",1),bool(false)); //tototo 
}


Pt3dr & cORGI_CentreCommun::Incertitude()
{
   return mIncertitude;
}

const Pt3dr & cORGI_CentreCommun::Incertitude()const 
{
   return mIncertitude;
}

cElXMLTree * ToXMLTree(const cORGI_CentreCommun & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ORGI_CentreCommun",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Incertitude"),anObj.Incertitude())->ReTagThis("Incertitude"));
  return aRes;
}

void xml_init(cORGI_CentreCommun & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.Incertitude(),aTree->Get("Incertitude",1)); //tototo 
}


Pt3dr & cORGI_TetaCommun::Incertitude()
{
   return mIncertitude;
}

const Pt3dr & cORGI_TetaCommun::Incertitude()const 
{
   return mIncertitude;
}

cElXMLTree * ToXMLTree(const cORGI_TetaCommun & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ORGI_TetaCommun",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Incertitude"),anObj.Incertitude())->ReTagThis("Incertitude"));
  return aRes;
}

void xml_init(cORGI_TetaCommun & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.Incertitude(),aTree->Get("Incertitude",1)); //tototo 
}


std::string & cObsRigidGrpImage::RefGrp()
{
   return mRefGrp;
}

const std::string & cObsRigidGrpImage::RefGrp()const 
{
   return mRefGrp;
}


cTplValGesInit< cORGI_CentreCommun > & cObsRigidGrpImage::ORGI_CentreCommun()
{
   return mORGI_CentreCommun;
}

const cTplValGesInit< cORGI_CentreCommun > & cObsRigidGrpImage::ORGI_CentreCommun()const 
{
   return mORGI_CentreCommun;
}


cTplValGesInit< cORGI_TetaCommun > & cObsRigidGrpImage::ORGI_TetaCommun()
{
   return mORGI_TetaCommun;
}

const cTplValGesInit< cORGI_TetaCommun > & cObsRigidGrpImage::ORGI_TetaCommun()const 
{
   return mORGI_TetaCommun;
}

cElXMLTree * ToXMLTree(const cObsRigidGrpImage & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ObsRigidGrpImage",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("RefGrp"),anObj.RefGrp())->ReTagThis("RefGrp"));
   if (anObj.ORGI_CentreCommun().IsInit())
      aRes->AddFils(ToXMLTree(anObj.ORGI_CentreCommun().Val())->ReTagThis("ORGI_CentreCommun"));
   if (anObj.ORGI_TetaCommun().IsInit())
      aRes->AddFils(ToXMLTree(anObj.ORGI_TetaCommun().Val())->ReTagThis("ORGI_TetaCommun"));
  return aRes;
}

void xml_init(cObsRigidGrpImage & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.RefGrp(),aTree->Get("RefGrp",1)); //tototo 

   xml_init(anObj.ORGI_CentreCommun(),aTree->Get("ORGI_CentreCommun",1)); //tototo 

   xml_init(anObj.ORGI_TetaCommun(),aTree->Get("ORGI_TetaCommun",1)); //tototo 
}


std::string & cTxtRapDetaille::NameFile()
{
   return mNameFile;
}

const std::string & cTxtRapDetaille::NameFile()const 
{
   return mNameFile;
}

cElXMLTree * ToXMLTree(const cTxtRapDetaille & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"TxtRapDetaille",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("NameFile"),anObj.NameFile())->ReTagThis("NameFile"));
  return aRes;
}

void xml_init(cTxtRapDetaille & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.NameFile(),aTree->Get("NameFile",1)); //tototo 
}


std::list< cObsAppuis > & cSectionObservations::ObsAppuis()
{
   return mObsAppuis;
}

const std::list< cObsAppuis > & cSectionObservations::ObsAppuis()const 
{
   return mObsAppuis;
}


std::list< cObsAppuisFlottant > & cSectionObservations::ObsAppuisFlottant()
{
   return mObsAppuisFlottant;
}

const std::list< cObsAppuisFlottant > & cSectionObservations::ObsAppuisFlottant()const 
{
   return mObsAppuisFlottant;
}


std::list< cObsLiaisons > & cSectionObservations::ObsLiaisons()
{
   return mObsLiaisons;
}

const std::list< cObsLiaisons > & cSectionObservations::ObsLiaisons()const 
{
   return mObsLiaisons;
}


std::list< cObsCentrePDV > & cSectionObservations::ObsCentrePDV()
{
   return mObsCentrePDV;
}

const std::list< cObsCentrePDV > & cSectionObservations::ObsCentrePDV()const 
{
   return mObsCentrePDV;
}


std::list< cObsRigidGrpImage > & cSectionObservations::ObsRigidGrpImage()
{
   return mObsRigidGrpImage;
}

const std::list< cObsRigidGrpImage > & cSectionObservations::ObsRigidGrpImage()const 
{
   return mObsRigidGrpImage;
}


std::string & cSectionObservations::NameFile()
{
   return TxtRapDetaille().Val().NameFile();
}

const std::string & cSectionObservations::NameFile()const 
{
   return TxtRapDetaille().Val().NameFile();
}


cTplValGesInit< cTxtRapDetaille > & cSectionObservations::TxtRapDetaille()
{
   return mTxtRapDetaille;
}

const cTplValGesInit< cTxtRapDetaille > & cSectionObservations::TxtRapDetaille()const 
{
   return mTxtRapDetaille;
}

cElXMLTree * ToXMLTree(const cSectionObservations & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"SectionObservations",eXMLBranche);
  for
  (       std::list< cObsAppuis >::const_iterator it=anObj.ObsAppuis().begin();
      it !=anObj.ObsAppuis().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("ObsAppuis"));
  for
  (       std::list< cObsAppuisFlottant >::const_iterator it=anObj.ObsAppuisFlottant().begin();
      it !=anObj.ObsAppuisFlottant().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("ObsAppuisFlottant"));
  for
  (       std::list< cObsLiaisons >::const_iterator it=anObj.ObsLiaisons().begin();
      it !=anObj.ObsLiaisons().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("ObsLiaisons"));
  for
  (       std::list< cObsCentrePDV >::const_iterator it=anObj.ObsCentrePDV().begin();
      it !=anObj.ObsCentrePDV().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("ObsCentrePDV"));
  for
  (       std::list< cObsRigidGrpImage >::const_iterator it=anObj.ObsRigidGrpImage().begin();
      it !=anObj.ObsRigidGrpImage().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("ObsRigidGrpImage"));
   if (anObj.TxtRapDetaille().IsInit())
      aRes->AddFils(ToXMLTree(anObj.TxtRapDetaille().Val())->ReTagThis("TxtRapDetaille"));
  return aRes;
}

void xml_init(cSectionObservations & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.ObsAppuis(),aTree->GetAll("ObsAppuis",false,1));

   xml_init(anObj.ObsAppuisFlottant(),aTree->GetAll("ObsAppuisFlottant",false,1));

   xml_init(anObj.ObsLiaisons(),aTree->GetAll("ObsLiaisons",false,1));

   xml_init(anObj.ObsCentrePDV(),aTree->GetAll("ObsCentrePDV",false,1));

   xml_init(anObj.ObsRigidGrpImage(),aTree->GetAll("ObsRigidGrpImage",false,1));

   xml_init(anObj.TxtRapDetaille(),aTree->Get("TxtRapDetaille",1)); //tototo 
}


cTplValGesInit< bool > & cExportAsGrid::DoExport()
{
   return mDoExport;
}

const cTplValGesInit< bool > & cExportAsGrid::DoExport()const 
{
   return mDoExport;
}


std::string & cExportAsGrid::Name()
{
   return mName;
}

const std::string & cExportAsGrid::Name()const 
{
   return mName;
}


cTplValGesInit< std::string > & cExportAsGrid::XML_Supl()
{
   return mXML_Supl;
}

const cTplValGesInit< std::string > & cExportAsGrid::XML_Supl()const 
{
   return mXML_Supl;
}


cTplValGesInit< bool > & cExportAsGrid::XML_Autonome()
{
   return mXML_Autonome;
}

const cTplValGesInit< bool > & cExportAsGrid::XML_Autonome()const 
{
   return mXML_Autonome;
}


cTplValGesInit< Pt2dr > & cExportAsGrid::RabPt()
{
   return mRabPt;
}

const cTplValGesInit< Pt2dr > & cExportAsGrid::RabPt()const 
{
   return mRabPt;
}


cTplValGesInit< Pt2dr > & cExportAsGrid::Step()
{
   return mStep;
}

const cTplValGesInit< Pt2dr > & cExportAsGrid::Step()const 
{
   return mStep;
}

cElXMLTree * ToXMLTree(const cExportAsGrid & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ExportAsGrid",eXMLBranche);
   if (anObj.DoExport().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DoExport"),anObj.DoExport().Val())->ReTagThis("DoExport"));
   aRes->AddFils(::ToXMLTree(std::string("Name"),anObj.Name())->ReTagThis("Name"));
   if (anObj.XML_Supl().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("XML_Supl"),anObj.XML_Supl().Val())->ReTagThis("XML_Supl"));
   if (anObj.XML_Autonome().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("XML_Autonome"),anObj.XML_Autonome().Val())->ReTagThis("XML_Autonome"));
   if (anObj.RabPt().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("RabPt"),anObj.RabPt().Val())->ReTagThis("RabPt"));
   if (anObj.Step().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Step"),anObj.Step().Val())->ReTagThis("Step"));
  return aRes;
}

void xml_init(cExportAsGrid & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.DoExport(),aTree->Get("DoExport",1),bool(true)); //tototo 

   xml_init(anObj.Name(),aTree->Get("Name",1)); //tototo 

   xml_init(anObj.XML_Supl(),aTree->Get("XML_Supl",1)); //tototo 

   xml_init(anObj.XML_Autonome(),aTree->Get("XML_Autonome",1),bool(false)); //tototo 

   xml_init(anObj.RabPt(),aTree->Get("RabPt",1),Pt2dr(Pt2dr(200,200))); //tototo 

   xml_init(anObj.Step(),aTree->Get("Step",1),Pt2dr(Pt2dr(20,20))); //tototo 
}


cTplValGesInit< std::string > & cExportCalib::PatternSel()
{
   return mPatternSel;
}

const cTplValGesInit< std::string > & cExportCalib::PatternSel()const 
{
   return mPatternSel;
}


std::string & cExportCalib::KeyAssoc()
{
   return mKeyAssoc;
}

const std::string & cExportCalib::KeyAssoc()const 
{
   return mKeyAssoc;
}


cTplValGesInit< std::string > & cExportCalib::Prefix()
{
   return mPrefix;
}

const cTplValGesInit< std::string > & cExportCalib::Prefix()const 
{
   return mPrefix;
}


cTplValGesInit< std::string > & cExportCalib::Postfix()
{
   return mPostfix;
}

const cTplValGesInit< std::string > & cExportCalib::Postfix()const 
{
   return mPostfix;
}


cTplValGesInit< bool > & cExportCalib::KeyIsName()
{
   return mKeyIsName;
}

const cTplValGesInit< bool > & cExportCalib::KeyIsName()const 
{
   return mKeyIsName;
}


cTplValGesInit< bool > & cExportCalib::DoExport()
{
   return ExportAsGrid().Val().DoExport();
}

const cTplValGesInit< bool > & cExportCalib::DoExport()const 
{
   return ExportAsGrid().Val().DoExport();
}


std::string & cExportCalib::Name()
{
   return ExportAsGrid().Val().Name();
}

const std::string & cExportCalib::Name()const 
{
   return ExportAsGrid().Val().Name();
}


cTplValGesInit< std::string > & cExportCalib::XML_Supl()
{
   return ExportAsGrid().Val().XML_Supl();
}

const cTplValGesInit< std::string > & cExportCalib::XML_Supl()const 
{
   return ExportAsGrid().Val().XML_Supl();
}


cTplValGesInit< bool > & cExportCalib::XML_Autonome()
{
   return ExportAsGrid().Val().XML_Autonome();
}

const cTplValGesInit< bool > & cExportCalib::XML_Autonome()const 
{
   return ExportAsGrid().Val().XML_Autonome();
}


cTplValGesInit< Pt2dr > & cExportCalib::RabPt()
{
   return ExportAsGrid().Val().RabPt();
}

const cTplValGesInit< Pt2dr > & cExportCalib::RabPt()const 
{
   return ExportAsGrid().Val().RabPt();
}


cTplValGesInit< Pt2dr > & cExportCalib::Step()
{
   return ExportAsGrid().Val().Step();
}

const cTplValGesInit< Pt2dr > & cExportCalib::Step()const 
{
   return ExportAsGrid().Val().Step();
}


cTplValGesInit< cExportAsGrid > & cExportCalib::ExportAsGrid()
{
   return mExportAsGrid;
}

const cTplValGesInit< cExportAsGrid > & cExportCalib::ExportAsGrid()const 
{
   return mExportAsGrid;
}


cTplValGesInit< cExportAsNewGrid > & cExportCalib::ExportAsNewGrid()
{
   return mExportAsNewGrid;
}

const cTplValGesInit< cExportAsNewGrid > & cExportCalib::ExportAsNewGrid()const 
{
   return mExportAsNewGrid;
}

cElXMLTree * ToXMLTree(const cExportCalib & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ExportCalib",eXMLBranche);
   if (anObj.PatternSel().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PatternSel"),anObj.PatternSel().Val())->ReTagThis("PatternSel"));
   aRes->AddFils(::ToXMLTree(std::string("KeyAssoc"),anObj.KeyAssoc())->ReTagThis("KeyAssoc"));
   if (anObj.Prefix().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Prefix"),anObj.Prefix().Val())->ReTagThis("Prefix"));
   if (anObj.Postfix().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Postfix"),anObj.Postfix().Val())->ReTagThis("Postfix"));
   if (anObj.KeyIsName().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("KeyIsName"),anObj.KeyIsName().Val())->ReTagThis("KeyIsName"));
   if (anObj.ExportAsGrid().IsInit())
      aRes->AddFils(ToXMLTree(anObj.ExportAsGrid().Val())->ReTagThis("ExportAsGrid"));
   if (anObj.ExportAsNewGrid().IsInit())
      aRes->AddFils(ToXMLTree(anObj.ExportAsNewGrid().Val())->ReTagThis("ExportAsNewGrid"));
  return aRes;
}

void xml_init(cExportCalib & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.PatternSel(),aTree->Get("PatternSel",1),std::string(".*")); //tototo 

   xml_init(anObj.KeyAssoc(),aTree->Get("KeyAssoc",1)); //tototo 

   xml_init(anObj.Prefix(),aTree->Get("Prefix",1),std::string("")); //tototo 

   xml_init(anObj.Postfix(),aTree->Get("Postfix",1),std::string("")); //tototo 

   xml_init(anObj.KeyIsName(),aTree->Get("KeyIsName",1),bool(false)); //tototo 

   xml_init(anObj.ExportAsGrid(),aTree->Get("ExportAsGrid",1)); //tototo 

   xml_init(anObj.ExportAsNewGrid(),aTree->Get("ExportAsNewGrid",1)); //tototo 
}


cTplValGesInit< cChangementCoordonnees > & cExportPose::ChC()
{
   return mChC;
}

const cTplValGesInit< cChangementCoordonnees > & cExportPose::ChC()const 
{
   return mChC;
}


std::string & cExportPose::KeyAssoc()
{
   return mKeyAssoc;
}

const std::string & cExportPose::KeyAssoc()const 
{
   return mKeyAssoc;
}


cTplValGesInit< bool > & cExportPose::AddCalib()
{
   return mAddCalib;
}

const cTplValGesInit< bool > & cExportPose::AddCalib()const 
{
   return mAddCalib;
}


cTplValGesInit< cExportAsNewGrid > & cExportPose::ExportAsNewGrid()
{
   return mExportAsNewGrid;
}

const cTplValGesInit< cExportAsNewGrid > & cExportPose::ExportAsNewGrid()const 
{
   return mExportAsNewGrid;
}


cTplValGesInit< std::string > & cExportPose::FileExtern()
{
   return mFileExtern;
}

const cTplValGesInit< std::string > & cExportPose::FileExtern()const 
{
   return mFileExtern;
}


cTplValGesInit< bool > & cExportPose::FileExternIsKey()
{
   return mFileExternIsKey;
}

const cTplValGesInit< bool > & cExportPose::FileExternIsKey()const 
{
   return mFileExternIsKey;
}


cTplValGesInit< bool > & cExportPose::CalcKeyFromCalib()
{
   return mCalcKeyFromCalib;
}

const cTplValGesInit< bool > & cExportPose::CalcKeyFromCalib()const 
{
   return mCalcKeyFromCalib;
}


cTplValGesInit< bool > & cExportPose::RelativeNameFE()
{
   return mRelativeNameFE;
}

const cTplValGesInit< bool > & cExportPose::RelativeNameFE()const 
{
   return mRelativeNameFE;
}


cTplValGesInit< bool > & cExportPose::ModeAngulaire()
{
   return mModeAngulaire;
}

const cTplValGesInit< bool > & cExportPose::ModeAngulaire()const 
{
   return mModeAngulaire;
}


cTplValGesInit< std::string > & cExportPose::PatternSel()
{
   return mPatternSel;
}

const cTplValGesInit< std::string > & cExportPose::PatternSel()const 
{
   return mPatternSel;
}


cTplValGesInit< int > & cExportPose::NbVerif()
{
   return mNbVerif;
}

const cTplValGesInit< int > & cExportPose::NbVerif()const 
{
   return mNbVerif;
}


cTplValGesInit< Pt3di > & cExportPose::VerifDeterm()
{
   return mVerifDeterm;
}

const cTplValGesInit< Pt3di > & cExportPose::VerifDeterm()const 
{
   return mVerifDeterm;
}


cTplValGesInit< bool > & cExportPose::ShowWhenVerif()
{
   return mShowWhenVerif;
}

const cTplValGesInit< bool > & cExportPose::ShowWhenVerif()const 
{
   return mShowWhenVerif;
}


cTplValGesInit< double > & cExportPose::TolWhenVerif()
{
   return mTolWhenVerif;
}

const cTplValGesInit< double > & cExportPose::TolWhenVerif()const 
{
   return mTolWhenVerif;
}

cElXMLTree * ToXMLTree(const cExportPose & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ExportPose",eXMLBranche);
   if (anObj.ChC().IsInit())
      aRes->AddFils(ToXMLTree(anObj.ChC().Val())->ReTagThis("ChC"));
   aRes->AddFils(::ToXMLTree(std::string("KeyAssoc"),anObj.KeyAssoc())->ReTagThis("KeyAssoc"));
   if (anObj.AddCalib().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("AddCalib"),anObj.AddCalib().Val())->ReTagThis("AddCalib"));
   if (anObj.ExportAsNewGrid().IsInit())
      aRes->AddFils(ToXMLTree(anObj.ExportAsNewGrid().Val())->ReTagThis("ExportAsNewGrid"));
   if (anObj.FileExtern().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("FileExtern"),anObj.FileExtern().Val())->ReTagThis("FileExtern"));
   if (anObj.FileExternIsKey().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("FileExternIsKey"),anObj.FileExternIsKey().Val())->ReTagThis("FileExternIsKey"));
   if (anObj.CalcKeyFromCalib().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("CalcKeyFromCalib"),anObj.CalcKeyFromCalib().Val())->ReTagThis("CalcKeyFromCalib"));
   if (anObj.RelativeNameFE().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("RelativeNameFE"),anObj.RelativeNameFE().Val())->ReTagThis("RelativeNameFE"));
   if (anObj.ModeAngulaire().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ModeAngulaire"),anObj.ModeAngulaire().Val())->ReTagThis("ModeAngulaire"));
   if (anObj.PatternSel().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PatternSel"),anObj.PatternSel().Val())->ReTagThis("PatternSel"));
   if (anObj.NbVerif().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NbVerif"),anObj.NbVerif().Val())->ReTagThis("NbVerif"));
   if (anObj.VerifDeterm().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("VerifDeterm"),anObj.VerifDeterm().Val())->ReTagThis("VerifDeterm"));
   if (anObj.ShowWhenVerif().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ShowWhenVerif"),anObj.ShowWhenVerif().Val())->ReTagThis("ShowWhenVerif"));
   if (anObj.TolWhenVerif().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("TolWhenVerif"),anObj.TolWhenVerif().Val())->ReTagThis("TolWhenVerif"));
  return aRes;
}

void xml_init(cExportPose & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.ChC(),aTree->Get("ChC",1)); //tototo 

   xml_init(anObj.KeyAssoc(),aTree->Get("KeyAssoc",1)); //tototo 

   xml_init(anObj.AddCalib(),aTree->Get("AddCalib",1),bool(true)); //tototo 

   xml_init(anObj.ExportAsNewGrid(),aTree->Get("ExportAsNewGrid",1)); //tototo 

   xml_init(anObj.FileExtern(),aTree->Get("FileExtern",1)); //tototo 

   xml_init(anObj.FileExternIsKey(),aTree->Get("FileExternIsKey",1),bool(false)); //tototo 

   xml_init(anObj.CalcKeyFromCalib(),aTree->Get("CalcKeyFromCalib",1),bool(false)); //tototo 

   xml_init(anObj.RelativeNameFE(),aTree->Get("RelativeNameFE",1),bool(true)); //tototo 

   xml_init(anObj.ModeAngulaire(),aTree->Get("ModeAngulaire",1),bool(false)); //tototo 

   xml_init(anObj.PatternSel(),aTree->Get("PatternSel",1),std::string(".*")); //tototo 

   xml_init(anObj.NbVerif(),aTree->Get("NbVerif",1),int(0)); //tototo 

   xml_init(anObj.VerifDeterm(),aTree->Get("VerifDeterm",1)); //tototo 

   xml_init(anObj.ShowWhenVerif(),aTree->Get("ShowWhenVerif",1),bool(true)); //tototo 

   xml_init(anObj.TolWhenVerif(),aTree->Get("TolWhenVerif",1),double(1e-3)); //tototo 
}


std::string & cExportAttrPose::KeyAssoc()
{
   return mKeyAssoc;
}

const std::string & cExportAttrPose::KeyAssoc()const 
{
   return mKeyAssoc;
}


cTplValGesInit< std::string > & cExportAttrPose::AttrSup()
{
   return mAttrSup;
}

const cTplValGesInit< std::string > & cExportAttrPose::AttrSup()const 
{
   return mAttrSup;
}


std::string & cExportAttrPose::PatternApply()
{
   return mPatternApply;
}

const std::string & cExportAttrPose::PatternApply()const 
{
   return mPatternApply;
}


cTplValGesInit< cParamEstimPlan > & cExportAttrPose::ExportDirVerticaleLocale()
{
   return mExportDirVerticaleLocale;
}

const cTplValGesInit< cParamEstimPlan > & cExportAttrPose::ExportDirVerticaleLocale()const 
{
   return mExportDirVerticaleLocale;
}

cElXMLTree * ToXMLTree(const cExportAttrPose & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ExportAttrPose",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("KeyAssoc"),anObj.KeyAssoc())->ReTagThis("KeyAssoc"));
   if (anObj.AttrSup().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("AttrSup"),anObj.AttrSup().Val())->ReTagThis("AttrSup"));
   aRes->AddFils(::ToXMLTree(std::string("PatternApply"),anObj.PatternApply())->ReTagThis("PatternApply"));
   if (anObj.ExportDirVerticaleLocale().IsInit())
      aRes->AddFils(ToXMLTree(anObj.ExportDirVerticaleLocale().Val())->ReTagThis("ExportDirVerticaleLocale"));
  return aRes;
}

void xml_init(cExportAttrPose & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.KeyAssoc(),aTree->Get("KeyAssoc",1)); //tototo 

   xml_init(anObj.AttrSup(),aTree->Get("AttrSup",1)); //tototo 

   xml_init(anObj.PatternApply(),aTree->Get("PatternApply",1)); //tototo 

   xml_init(anObj.ExportDirVerticaleLocale(),aTree->Get("ExportDirVerticaleLocale",1)); //tototo 
}


cTplValGesInit< bool > & cExportOrthoCyl::UseIt()
{
   return mUseIt;
}

const cTplValGesInit< bool > & cExportOrthoCyl::UseIt()const 
{
   return mUseIt;
}


cTplValGesInit< std::string > & cExportOrthoCyl::PatternEstimAxe()
{
   return mPatternEstimAxe;
}

const cTplValGesInit< std::string > & cExportOrthoCyl::PatternEstimAxe()const 
{
   return mPatternEstimAxe;
}


bool & cExportOrthoCyl::AngulCorr()
{
   return mAngulCorr;
}

const bool & cExportOrthoCyl::AngulCorr()const 
{
   return mAngulCorr;
}


cTplValGesInit< bool > & cExportOrthoCyl::L2EstimAxe()
{
   return mL2EstimAxe;
}

const cTplValGesInit< bool > & cExportOrthoCyl::L2EstimAxe()const 
{
   return mL2EstimAxe;
}

cElXMLTree * ToXMLTree(const cExportOrthoCyl & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ExportOrthoCyl",eXMLBranche);
   if (anObj.UseIt().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("UseIt"),anObj.UseIt().Val())->ReTagThis("UseIt"));
   if (anObj.PatternEstimAxe().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PatternEstimAxe"),anObj.PatternEstimAxe().Val())->ReTagThis("PatternEstimAxe"));
   aRes->AddFils(::ToXMLTree(std::string("AngulCorr"),anObj.AngulCorr())->ReTagThis("AngulCorr"));
   if (anObj.L2EstimAxe().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("L2EstimAxe"),anObj.L2EstimAxe().Val())->ReTagThis("L2EstimAxe"));
  return aRes;
}

void xml_init(cExportOrthoCyl & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.UseIt(),aTree->Get("UseIt",1),bool(true)); //tototo 

   xml_init(anObj.PatternEstimAxe(),aTree->Get("PatternEstimAxe",1)); //tototo 

   xml_init(anObj.AngulCorr(),aTree->Get("AngulCorr",1)); //tototo 

   xml_init(anObj.L2EstimAxe(),aTree->Get("L2EstimAxe",1),bool(true)); //tototo 
}


std::string & cExportRepereLoc::NameRepere()
{
   return mNameRepere;
}

const std::string & cExportRepereLoc::NameRepere()const 
{
   return mNameRepere;
}


std::string & cExportRepereLoc::PatternEstimPl()
{
   return mPatternEstimPl;
}

const std::string & cExportRepereLoc::PatternEstimPl()const 
{
   return mPatternEstimPl;
}


cParamEstimPlan & cExportRepereLoc::EstimPlanHor()
{
   return mEstimPlanHor;
}

const cParamEstimPlan & cExportRepereLoc::EstimPlanHor()const 
{
   return mEstimPlanHor;
}


cTplValGesInit< std::string > & cExportRepereLoc::ImP1P2()
{
   return mImP1P2;
}

const cTplValGesInit< std::string > & cExportRepereLoc::ImP1P2()const 
{
   return mImP1P2;
}


Pt2dr & cExportRepereLoc::P1()
{
   return mP1;
}

const Pt2dr & cExportRepereLoc::P1()const 
{
   return mP1;
}


Pt2dr & cExportRepereLoc::P2()
{
   return mP2;
}

const Pt2dr & cExportRepereLoc::P2()const 
{
   return mP2;
}


cTplValGesInit< Pt2dr > & cExportRepereLoc::AxeDef()
{
   return mAxeDef;
}

const cTplValGesInit< Pt2dr > & cExportRepereLoc::AxeDef()const 
{
   return mAxeDef;
}


cTplValGesInit< Pt2dr > & cExportRepereLoc::Origine()
{
   return mOrigine;
}

const cTplValGesInit< Pt2dr > & cExportRepereLoc::Origine()const 
{
   return mOrigine;
}


cTplValGesInit< std::string > & cExportRepereLoc::NameImOri()
{
   return mNameImOri;
}

const cTplValGesInit< std::string > & cExportRepereLoc::NameImOri()const 
{
   return mNameImOri;
}


cTplValGesInit< bool > & cExportRepereLoc::UseIt()
{
   return ExportOrthoCyl().Val().UseIt();
}

const cTplValGesInit< bool > & cExportRepereLoc::UseIt()const 
{
   return ExportOrthoCyl().Val().UseIt();
}


cTplValGesInit< std::string > & cExportRepereLoc::PatternEstimAxe()
{
   return ExportOrthoCyl().Val().PatternEstimAxe();
}

const cTplValGesInit< std::string > & cExportRepereLoc::PatternEstimAxe()const 
{
   return ExportOrthoCyl().Val().PatternEstimAxe();
}


bool & cExportRepereLoc::AngulCorr()
{
   return ExportOrthoCyl().Val().AngulCorr();
}

const bool & cExportRepereLoc::AngulCorr()const 
{
   return ExportOrthoCyl().Val().AngulCorr();
}


cTplValGesInit< bool > & cExportRepereLoc::L2EstimAxe()
{
   return ExportOrthoCyl().Val().L2EstimAxe();
}

const cTplValGesInit< bool > & cExportRepereLoc::L2EstimAxe()const 
{
   return ExportOrthoCyl().Val().L2EstimAxe();
}


cTplValGesInit< cExportOrthoCyl > & cExportRepereLoc::ExportOrthoCyl()
{
   return mExportOrthoCyl;
}

const cTplValGesInit< cExportOrthoCyl > & cExportRepereLoc::ExportOrthoCyl()const 
{
   return mExportOrthoCyl;
}

cElXMLTree * ToXMLTree(const cExportRepereLoc & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ExportRepereLoc",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("NameRepere"),anObj.NameRepere())->ReTagThis("NameRepere"));
   aRes->AddFils(::ToXMLTree(std::string("PatternEstimPl"),anObj.PatternEstimPl())->ReTagThis("PatternEstimPl"));
   aRes->AddFils(ToXMLTree(anObj.EstimPlanHor())->ReTagThis("EstimPlanHor"));
   if (anObj.ImP1P2().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ImP1P2"),anObj.ImP1P2().Val())->ReTagThis("ImP1P2"));
   aRes->AddFils(::ToXMLTree(std::string("P1"),anObj.P1())->ReTagThis("P1"));
   aRes->AddFils(::ToXMLTree(std::string("P2"),anObj.P2())->ReTagThis("P2"));
   if (anObj.AxeDef().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("AxeDef"),anObj.AxeDef().Val())->ReTagThis("AxeDef"));
   if (anObj.Origine().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Origine"),anObj.Origine().Val())->ReTagThis("Origine"));
   if (anObj.NameImOri().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NameImOri"),anObj.NameImOri().Val())->ReTagThis("NameImOri"));
   if (anObj.ExportOrthoCyl().IsInit())
      aRes->AddFils(ToXMLTree(anObj.ExportOrthoCyl().Val())->ReTagThis("ExportOrthoCyl"));
  return aRes;
}

void xml_init(cExportRepereLoc & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.NameRepere(),aTree->Get("NameRepere",1)); //tototo 

   xml_init(anObj.PatternEstimPl(),aTree->Get("PatternEstimPl",1)); //tototo 

   xml_init(anObj.EstimPlanHor(),aTree->Get("EstimPlanHor",1)); //tototo 

   xml_init(anObj.ImP1P2(),aTree->Get("ImP1P2",1)); //tototo 

   xml_init(anObj.P1(),aTree->Get("P1",1)); //tototo 

   xml_init(anObj.P2(),aTree->Get("P2",1)); //tototo 

   xml_init(anObj.AxeDef(),aTree->Get("AxeDef",1),Pt2dr(Pt2dr(1,0))); //tototo 

   xml_init(anObj.Origine(),aTree->Get("Origine",1)); //tototo 

   xml_init(anObj.NameImOri(),aTree->Get("NameImOri",1)); //tototo 

   xml_init(anObj.ExportOrthoCyl(),aTree->Get("ExportOrthoCyl",1)); //tototo 
}


std::list< std::string > & cCartes2Export::Im1()
{
   return mIm1;
}

const std::list< std::string > & cCartes2Export::Im1()const 
{
   return mIm1;
}


std::string & cCartes2Export::Nuage()
{
   return mNuage;
}

const std::string & cCartes2Export::Nuage()const 
{
   return mNuage;
}


std::list< std::string > & cCartes2Export::ImN()
{
   return mImN;
}

const std::list< std::string > & cCartes2Export::ImN()const 
{
   return mImN;
}


cTplValGesInit< std::string > & cCartes2Export::FilterIm2()
{
   return mFilterIm2;
}

const cTplValGesInit< std::string > & cCartes2Export::FilterIm2()const 
{
   return mFilterIm2;
}

cElXMLTree * ToXMLTree(const cCartes2Export & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"Cartes2Export",eXMLBranche);
  for
  (       std::list< std::string >::const_iterator it=anObj.Im1().begin();
      it !=anObj.Im1().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("Im1"),(*it))->ReTagThis("Im1"));
   aRes->AddFils(::ToXMLTree(std::string("Nuage"),anObj.Nuage())->ReTagThis("Nuage"));
  for
  (       std::list< std::string >::const_iterator it=anObj.ImN().begin();
      it !=anObj.ImN().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("ImN"),(*it))->ReTagThis("ImN"));
   if (anObj.FilterIm2().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("FilterIm2"),anObj.FilterIm2().Val())->ReTagThis("FilterIm2"));
  return aRes;
}

void xml_init(cCartes2Export & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.Im1(),aTree->GetAll("Im1",false,1));

   xml_init(anObj.Nuage(),aTree->Get("Nuage",1)); //tototo 

   xml_init(anObj.ImN(),aTree->GetAll("ImN",false,1));

   xml_init(anObj.FilterIm2(),aTree->Get("FilterIm2",1),std::string(".*")); //tototo 
}


std::list< cCartes2Export > & cExportMesuresFromCarteProf::Cartes2Export()
{
   return mCartes2Export;
}

const std::list< cCartes2Export > & cExportMesuresFromCarteProf::Cartes2Export()const 
{
   return mCartes2Export;
}


std::string & cExportMesuresFromCarteProf::IdBdLiaisonIn()
{
   return mIdBdLiaisonIn;
}

const std::string & cExportMesuresFromCarteProf::IdBdLiaisonIn()const 
{
   return mIdBdLiaisonIn;
}


cTplValGesInit< std::string > & cExportMesuresFromCarteProf::KeyAssocLiaisons12()
{
   return mKeyAssocLiaisons12;
}

const cTplValGesInit< std::string > & cExportMesuresFromCarteProf::KeyAssocLiaisons12()const 
{
   return mKeyAssocLiaisons12;
}


cTplValGesInit< std::string > & cExportMesuresFromCarteProf::KeyAssocLiaisons21()
{
   return mKeyAssocLiaisons21;
}

const cTplValGesInit< std::string > & cExportMesuresFromCarteProf::KeyAssocLiaisons21()const 
{
   return mKeyAssocLiaisons21;
}


cTplValGesInit< std::string > & cExportMesuresFromCarteProf::KeyAssocAppuis()
{
   return mKeyAssocAppuis;
}

const cTplValGesInit< std::string > & cExportMesuresFromCarteProf::KeyAssocAppuis()const 
{
   return mKeyAssocAppuis;
}


cTplValGesInit< bool > & cExportMesuresFromCarteProf::AppuisModeAdd()
{
   return mAppuisModeAdd;
}

const cTplValGesInit< bool > & cExportMesuresFromCarteProf::AppuisModeAdd()const 
{
   return mAppuisModeAdd;
}


cTplValGesInit< bool > & cExportMesuresFromCarteProf::LiaisonModeAdd()
{
   return mLiaisonModeAdd;
}

const cTplValGesInit< bool > & cExportMesuresFromCarteProf::LiaisonModeAdd()const 
{
   return mLiaisonModeAdd;
}

cElXMLTree * ToXMLTree(const cExportMesuresFromCarteProf & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ExportMesuresFromCarteProf",eXMLBranche);
  for
  (       std::list< cCartes2Export >::const_iterator it=anObj.Cartes2Export().begin();
      it !=anObj.Cartes2Export().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("Cartes2Export"));
   aRes->AddFils(::ToXMLTree(std::string("IdBdLiaisonIn"),anObj.IdBdLiaisonIn())->ReTagThis("IdBdLiaisonIn"));
   if (anObj.KeyAssocLiaisons12().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("KeyAssocLiaisons12"),anObj.KeyAssocLiaisons12().Val())->ReTagThis("KeyAssocLiaisons12"));
   if (anObj.KeyAssocLiaisons21().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("KeyAssocLiaisons21"),anObj.KeyAssocLiaisons21().Val())->ReTagThis("KeyAssocLiaisons21"));
   if (anObj.KeyAssocAppuis().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("KeyAssocAppuis"),anObj.KeyAssocAppuis().Val())->ReTagThis("KeyAssocAppuis"));
   if (anObj.AppuisModeAdd().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("AppuisModeAdd"),anObj.AppuisModeAdd().Val())->ReTagThis("AppuisModeAdd"));
   if (anObj.LiaisonModeAdd().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("LiaisonModeAdd"),anObj.LiaisonModeAdd().Val())->ReTagThis("LiaisonModeAdd"));
  return aRes;
}

void xml_init(cExportMesuresFromCarteProf & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.Cartes2Export(),aTree->GetAll("Cartes2Export",false,1));

   xml_init(anObj.IdBdLiaisonIn(),aTree->Get("IdBdLiaisonIn",1)); //tototo 

   xml_init(anObj.KeyAssocLiaisons12(),aTree->Get("KeyAssocLiaisons12",1)); //tototo 

   xml_init(anObj.KeyAssocLiaisons21(),aTree->Get("KeyAssocLiaisons21",1)); //tototo 

   xml_init(anObj.KeyAssocAppuis(),aTree->Get("KeyAssocAppuis",1)); //tototo 

   xml_init(anObj.AppuisModeAdd(),aTree->Get("AppuisModeAdd",1),bool(true)); //tototo 

   xml_init(anObj.LiaisonModeAdd(),aTree->Get("LiaisonModeAdd",1),bool(false)); //tototo 
}


std::list< std::string > & cExportVisuConfigGrpPose::PatternSel()
{
   return mPatternSel;
}

const std::list< std::string > & cExportVisuConfigGrpPose::PatternSel()const 
{
   return mPatternSel;
}


std::string & cExportVisuConfigGrpPose::NameFile()
{
   return mNameFile;
}

const std::string & cExportVisuConfigGrpPose::NameFile()const 
{
   return mNameFile;
}

cElXMLTree * ToXMLTree(const cExportVisuConfigGrpPose & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ExportVisuConfigGrpPose",eXMLBranche);
  for
  (       std::list< std::string >::const_iterator it=anObj.PatternSel().begin();
      it !=anObj.PatternSel().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("PatternSel"),(*it))->ReTagThis("PatternSel"));
   aRes->AddFils(::ToXMLTree(std::string("NameFile"),anObj.NameFile())->ReTagThis("NameFile"));
  return aRes;
}

void xml_init(cExportVisuConfigGrpPose & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.PatternSel(),aTree->GetAll("PatternSel",false,1));

   xml_init(anObj.NameFile(),aTree->Get("NameFile",1)); //tototo 
}


cTplValGesInit< std::string > & cExportPtsFlottant::PatternSel()
{
   return mPatternSel;
}

const cTplValGesInit< std::string > & cExportPtsFlottant::PatternSel()const 
{
   return mPatternSel;
}


cTplValGesInit< std::string > & cExportPtsFlottant::NameFileXml()
{
   return mNameFileXml;
}

const cTplValGesInit< std::string > & cExportPtsFlottant::NameFileXml()const 
{
   return mNameFileXml;
}


cTplValGesInit< std::string > & cExportPtsFlottant::NameFileTxt()
{
   return mNameFileTxt;
}

const cTplValGesInit< std::string > & cExportPtsFlottant::NameFileTxt()const 
{
   return mNameFileTxt;
}


cTplValGesInit< std::string > & cExportPtsFlottant::TextComplTxt()
{
   return mTextComplTxt;
}

const cTplValGesInit< std::string > & cExportPtsFlottant::TextComplTxt()const 
{
   return mTextComplTxt;
}

cElXMLTree * ToXMLTree(const cExportPtsFlottant & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ExportPtsFlottant",eXMLBranche);
   if (anObj.PatternSel().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PatternSel"),anObj.PatternSel().Val())->ReTagThis("PatternSel"));
   if (anObj.NameFileXml().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NameFileXml"),anObj.NameFileXml().Val())->ReTagThis("NameFileXml"));
   if (anObj.NameFileTxt().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NameFileTxt"),anObj.NameFileTxt().Val())->ReTagThis("NameFileTxt"));
   if (anObj.TextComplTxt().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("TextComplTxt"),anObj.TextComplTxt().Val())->ReTagThis("TextComplTxt"));
  return aRes;
}

void xml_init(cExportPtsFlottant & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.PatternSel(),aTree->Get("PatternSel",1),std::string(".*")); //tototo 

   xml_init(anObj.NameFileXml(),aTree->Get("NameFileXml",1)); //tototo 

   xml_init(anObj.NameFileTxt(),aTree->Get("NameFileTxt",1)); //tototo 

   xml_init(anObj.TextComplTxt(),aTree->Get("TextComplTxt",1)); //tototo 
}


std::string & cResidusIndiv::Pattern()
{
   return mPattern;
}

const std::string & cResidusIndiv::Pattern()const 
{
   return mPattern;
}


std::string & cResidusIndiv::Name()
{
   return mName;
}

const std::string & cResidusIndiv::Name()const 
{
   return mName;
}

cElXMLTree * ToXMLTree(const cResidusIndiv & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ResidusIndiv",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Pattern"),anObj.Pattern())->ReTagThis("Pattern"));
   aRes->AddFils(::ToXMLTree(std::string("Name"),anObj.Name())->ReTagThis("Name"));
  return aRes;
}

void xml_init(cResidusIndiv & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.Pattern(),aTree->Get("Pattern",1)); //tototo 

   xml_init(anObj.Name(),aTree->Get("Name",1)); //tototo 
}


cTplValGesInit< bool > & cExportImResiduLiaison::Signed()
{
   return mSigned;
}

const cTplValGesInit< bool > & cExportImResiduLiaison::Signed()const 
{
   return mSigned;
}


std::string & cExportImResiduLiaison::PatternGlobCalIm()
{
   return mPatternGlobCalIm;
}

const std::string & cExportImResiduLiaison::PatternGlobCalIm()const 
{
   return mPatternGlobCalIm;
}


std::string & cExportImResiduLiaison::NameGlobCalIm()
{
   return mNameGlobCalIm;
}

const std::string & cExportImResiduLiaison::NameGlobCalIm()const 
{
   return mNameGlobCalIm;
}


double & cExportImResiduLiaison::ScaleIm()
{
   return mScaleIm;
}

const double & cExportImResiduLiaison::ScaleIm()const 
{
   return mScaleIm;
}


double & cExportImResiduLiaison::DynIm()
{
   return mDynIm;
}

const double & cExportImResiduLiaison::DynIm()const 
{
   return mDynIm;
}


std::string & cExportImResiduLiaison::Pattern()
{
   return ResidusIndiv().Val().Pattern();
}

const std::string & cExportImResiduLiaison::Pattern()const 
{
   return ResidusIndiv().Val().Pattern();
}


std::string & cExportImResiduLiaison::Name()
{
   return ResidusIndiv().Val().Name();
}

const std::string & cExportImResiduLiaison::Name()const 
{
   return ResidusIndiv().Val().Name();
}


cTplValGesInit< cResidusIndiv > & cExportImResiduLiaison::ResidusIndiv()
{
   return mResidusIndiv;
}

const cTplValGesInit< cResidusIndiv > & cExportImResiduLiaison::ResidusIndiv()const 
{
   return mResidusIndiv;
}

cElXMLTree * ToXMLTree(const cExportImResiduLiaison & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ExportImResiduLiaison",eXMLBranche);
   if (anObj.Signed().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Signed"),anObj.Signed().Val())->ReTagThis("Signed"));
   aRes->AddFils(::ToXMLTree(std::string("PatternGlobCalIm"),anObj.PatternGlobCalIm())->ReTagThis("PatternGlobCalIm"));
   aRes->AddFils(::ToXMLTree(std::string("NameGlobCalIm"),anObj.NameGlobCalIm())->ReTagThis("NameGlobCalIm"));
   aRes->AddFils(::ToXMLTree(std::string("ScaleIm"),anObj.ScaleIm())->ReTagThis("ScaleIm"));
   aRes->AddFils(::ToXMLTree(std::string("DynIm"),anObj.DynIm())->ReTagThis("DynIm"));
   if (anObj.ResidusIndiv().IsInit())
      aRes->AddFils(ToXMLTree(anObj.ResidusIndiv().Val())->ReTagThis("ResidusIndiv"));
  return aRes;
}

void xml_init(cExportImResiduLiaison & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.Signed(),aTree->Get("Signed",1),bool(true)); //tototo 

   xml_init(anObj.PatternGlobCalIm(),aTree->Get("PatternGlobCalIm",1)); //tototo 

   xml_init(anObj.NameGlobCalIm(),aTree->Get("NameGlobCalIm",1)); //tototo 

   xml_init(anObj.ScaleIm(),aTree->Get("ScaleIm",1)); //tototo 

   xml_init(anObj.DynIm(),aTree->Get("DynIm",1)); //tototo 

   xml_init(anObj.ResidusIndiv(),aTree->Get("ResidusIndiv",1)); //tototo 
}


cTplValGesInit< double > & cExportRedressement::Dyn()
{
   return mDyn;
}

const cTplValGesInit< double > & cExportRedressement::Dyn()const 
{
   return mDyn;
}


cTplValGesInit< double > & cExportRedressement::Gamma()
{
   return mGamma;
}

const cTplValGesInit< double > & cExportRedressement::Gamma()const 
{
   return mGamma;
}


cTplValGesInit< eTypeNumerique > & cExportRedressement::TypeNum()
{
   return mTypeNum;
}

const cTplValGesInit< eTypeNumerique > & cExportRedressement::TypeNum()const 
{
   return mTypeNum;
}


cTplValGesInit< double > & cExportRedressement::Offset()
{
   return mOffset;
}

const cTplValGesInit< double > & cExportRedressement::Offset()const 
{
   return mOffset;
}


cTplValGesInit< cElRegex_Ptr > & cExportRedressement::PatternSel()
{
   return mPatternSel;
}

const cTplValGesInit< cElRegex_Ptr > & cExportRedressement::PatternSel()const 
{
   return mPatternSel;
}


cTplValGesInit< std::string > & cExportRedressement::KeyAssocIn()
{
   return mKeyAssocIn;
}

const cTplValGesInit< std::string > & cExportRedressement::KeyAssocIn()const 
{
   return mKeyAssocIn;
}


cTplValGesInit< Pt2dr > & cExportRedressement::OffsetIm()
{
   return mOffsetIm;
}

const cTplValGesInit< Pt2dr > & cExportRedressement::OffsetIm()const 
{
   return mOffsetIm;
}


cTplValGesInit< double > & cExportRedressement::ScaleIm()
{
   return mScaleIm;
}

const cTplValGesInit< double > & cExportRedressement::ScaleIm()const 
{
   return mScaleIm;
}


std::string & cExportRedressement::KeyAssocOut()
{
   return mKeyAssocOut;
}

const std::string & cExportRedressement::KeyAssocOut()const 
{
   return mKeyAssocOut;
}


cTplValGesInit< double > & cExportRedressement::ZSol()
{
   return mZSol;
}

const cTplValGesInit< double > & cExportRedressement::ZSol()const 
{
   return mZSol;
}


double & cExportRedressement::Resol()
{
   return mResol;
}

const double & cExportRedressement::Resol()const 
{
   return mResol;
}


bool & cExportRedressement::ResolIsRel()
{
   return mResolIsRel;
}

const bool & cExportRedressement::ResolIsRel()const 
{
   return mResolIsRel;
}


cTplValGesInit< bool > & cExportRedressement::DoTFW()
{
   return mDoTFW;
}

const cTplValGesInit< bool > & cExportRedressement::DoTFW()const 
{
   return mDoTFW;
}


double & cExportRedressement::TetaLimite()
{
   return mTetaLimite;
}

const double & cExportRedressement::TetaLimite()const 
{
   return mTetaLimite;
}


cTplValGesInit< Pt3dr > & cExportRedressement::DirTetaLim()
{
   return mDirTetaLim;
}

const cTplValGesInit< Pt3dr > & cExportRedressement::DirTetaLim()const 
{
   return mDirTetaLim;
}


cTplValGesInit< bool > & cExportRedressement::DoOnlyIfNew()
{
   return mDoOnlyIfNew;
}

const cTplValGesInit< bool > & cExportRedressement::DoOnlyIfNew()const 
{
   return mDoOnlyIfNew;
}

cElXMLTree * ToXMLTree(const cExportRedressement & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ExportRedressement",eXMLBranche);
   if (anObj.Dyn().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Dyn"),anObj.Dyn().Val())->ReTagThis("Dyn"));
   if (anObj.Gamma().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Gamma"),anObj.Gamma().Val())->ReTagThis("Gamma"));
   if (anObj.TypeNum().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("TypeNum"),anObj.TypeNum().Val())->ReTagThis("TypeNum"));
   if (anObj.Offset().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Offset"),anObj.Offset().Val())->ReTagThis("Offset"));
   if (anObj.PatternSel().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PatternSel"),anObj.PatternSel().Val())->ReTagThis("PatternSel"));
   if (anObj.KeyAssocIn().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("KeyAssocIn"),anObj.KeyAssocIn().Val())->ReTagThis("KeyAssocIn"));
   if (anObj.OffsetIm().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("OffsetIm"),anObj.OffsetIm().Val())->ReTagThis("OffsetIm"));
   if (anObj.ScaleIm().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ScaleIm"),anObj.ScaleIm().Val())->ReTagThis("ScaleIm"));
   aRes->AddFils(::ToXMLTree(std::string("KeyAssocOut"),anObj.KeyAssocOut())->ReTagThis("KeyAssocOut"));
   if (anObj.ZSol().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ZSol"),anObj.ZSol().Val())->ReTagThis("ZSol"));
   aRes->AddFils(::ToXMLTree(std::string("Resol"),anObj.Resol())->ReTagThis("Resol"));
   aRes->AddFils(::ToXMLTree(std::string("ResolIsRel"),anObj.ResolIsRel())->ReTagThis("ResolIsRel"));
   if (anObj.DoTFW().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DoTFW"),anObj.DoTFW().Val())->ReTagThis("DoTFW"));
   aRes->AddFils(::ToXMLTree(std::string("TetaLimite"),anObj.TetaLimite())->ReTagThis("TetaLimite"));
   if (anObj.DirTetaLim().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DirTetaLim"),anObj.DirTetaLim().Val())->ReTagThis("DirTetaLim"));
   if (anObj.DoOnlyIfNew().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DoOnlyIfNew"),anObj.DoOnlyIfNew().Val())->ReTagThis("DoOnlyIfNew"));
  return aRes;
}

void xml_init(cExportRedressement & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.Dyn(),aTree->Get("Dyn",1),double(1.0)); //tototo 

   xml_init(anObj.Gamma(),aTree->Get("Gamma",1),double(1.0)); //tototo 

   xml_init(anObj.TypeNum(),aTree->Get("TypeNum",1)); //tototo 

   xml_init(anObj.Offset(),aTree->Get("Offset",1),double(0.0)); //tototo 

   xml_init(anObj.PatternSel(),aTree->Get("PatternSel",1)); //tototo 

   xml_init(anObj.KeyAssocIn(),aTree->Get("KeyAssocIn",1)); //tototo 

   xml_init(anObj.OffsetIm(),aTree->Get("OffsetIm",1),Pt2dr(Pt2dr(0,0))); //tototo 

   xml_init(anObj.ScaleIm(),aTree->Get("ScaleIm",1),double(1.0)); //tototo 

   xml_init(anObj.KeyAssocOut(),aTree->Get("KeyAssocOut",1)); //tototo 

   xml_init(anObj.ZSol(),aTree->Get("ZSol",1)); //tototo 

   xml_init(anObj.Resol(),aTree->Get("Resol",1)); //tototo 

   xml_init(anObj.ResolIsRel(),aTree->Get("ResolIsRel",1)); //tototo 

   xml_init(anObj.DoTFW(),aTree->Get("DoTFW",1),bool(false)); //tototo 

   xml_init(anObj.TetaLimite(),aTree->Get("TetaLimite",1)); //tototo 

   xml_init(anObj.DirTetaLim(),aTree->Get("DirTetaLim",1),Pt3dr(Pt3dr(0,0,-1))); //tototo 

   xml_init(anObj.DoOnlyIfNew(),aTree->Get("DoOnlyIfNew",1),bool(true)); //tototo 
}


Pt3di & cNuagePutCam::ColCadre()
{
   return mColCadre;
}

const Pt3di & cNuagePutCam::ColCadre()const 
{
   return mColCadre;
}


cTplValGesInit< Pt3di > & cNuagePutCam::ColRay()
{
   return mColRay;
}

const cTplValGesInit< Pt3di > & cNuagePutCam::ColRay()const 
{
   return mColRay;
}


double & cNuagePutCam::Long()
{
   return mLong;
}

const double & cNuagePutCam::Long()const 
{
   return mLong;
}


double & cNuagePutCam::StepSeg()
{
   return mStepSeg;
}

const double & cNuagePutCam::StepSeg()const 
{
   return mStepSeg;
}


cTplValGesInit< double > & cNuagePutCam::StepImage()
{
   return mStepImage;
}

const cTplValGesInit< double > & cNuagePutCam::StepImage()const 
{
   return mStepImage;
}

cElXMLTree * ToXMLTree(const cNuagePutCam & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"NuagePutCam",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("ColCadre"),anObj.ColCadre())->ReTagThis("ColCadre"));
   if (anObj.ColRay().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ColRay"),anObj.ColRay().Val())->ReTagThis("ColRay"));
   aRes->AddFils(::ToXMLTree(std::string("Long"),anObj.Long())->ReTagThis("Long"));
   aRes->AddFils(::ToXMLTree(std::string("StepSeg"),anObj.StepSeg())->ReTagThis("StepSeg"));
   if (anObj.StepImage().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("StepImage"),anObj.StepImage().Val())->ReTagThis("StepImage"));
  return aRes;
}

void xml_init(cNuagePutCam & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.ColCadre(),aTree->Get("ColCadre",1)); //tototo 

   xml_init(anObj.ColRay(),aTree->Get("ColRay",1)); //tototo 

   xml_init(anObj.Long(),aTree->Get("Long",1)); //tototo 

   xml_init(anObj.StepSeg(),aTree->Get("StepSeg",1)); //tototo 

   xml_init(anObj.StepImage(),aTree->Get("StepImage",1),double(-1.0)); //tototo 
}


std::string & cExportNuage::NameOut()
{
   return mNameOut;
}

const std::string & cExportNuage::NameOut()const 
{
   return mNameOut;
}


cTplValGesInit< bool > & cExportNuage::PlyModeBin()
{
   return mPlyModeBin;
}

const cTplValGesInit< bool > & cExportNuage::PlyModeBin()const 
{
   return mPlyModeBin;
}


std::list< std::string > & cExportNuage::NameRefLiaison()
{
   return mNameRefLiaison;
}

const std::list< std::string > & cExportNuage::NameRefLiaison()const 
{
   return mNameRefLiaison;
}


cTplValGesInit< std::string > & cExportNuage::PatternSel()
{
   return mPatternSel;
}

const cTplValGesInit< std::string > & cExportNuage::PatternSel()const 
{
   return mPatternSel;
}


cPonderationPackMesure & cExportNuage::Pond()
{
   return mPond;
}

const cPonderationPackMesure & cExportNuage::Pond()const 
{
   return mPond;
}


std::string & cExportNuage::KeyFileColImage()
{
   return mKeyFileColImage;
}

const std::string & cExportNuage::KeyFileColImage()const 
{
   return mKeyFileColImage;
}


cTplValGesInit< int > & cExportNuage::NbChan()
{
   return mNbChan;
}

const cTplValGesInit< int > & cExportNuage::NbChan()const 
{
   return mNbChan;
}


cTplValGesInit< Pt3dr > & cExportNuage::DirCol()
{
   return mDirCol;
}

const cTplValGesInit< Pt3dr > & cExportNuage::DirCol()const 
{
   return mDirCol;
}


cTplValGesInit< double > & cExportNuage::PerCol()
{
   return mPerCol;
}

const cTplValGesInit< double > & cExportNuage::PerCol()const 
{
   return mPerCol;
}


cTplValGesInit< double > & cExportNuage::LimBSurH()
{
   return mLimBSurH;
}

const cTplValGesInit< double > & cExportNuage::LimBSurH()const 
{
   return mLimBSurH;
}


cTplValGesInit< std::string > & cExportNuage::ImExpoRef()
{
   return mImExpoRef;
}

const cTplValGesInit< std::string > & cExportNuage::ImExpoRef()const 
{
   return mImExpoRef;
}


Pt3di & cExportNuage::ColCadre()
{
   return NuagePutCam().Val().ColCadre();
}

const Pt3di & cExportNuage::ColCadre()const 
{
   return NuagePutCam().Val().ColCadre();
}


cTplValGesInit< Pt3di > & cExportNuage::ColRay()
{
   return NuagePutCam().Val().ColRay();
}

const cTplValGesInit< Pt3di > & cExportNuage::ColRay()const 
{
   return NuagePutCam().Val().ColRay();
}


double & cExportNuage::Long()
{
   return NuagePutCam().Val().Long();
}

const double & cExportNuage::Long()const 
{
   return NuagePutCam().Val().Long();
}


double & cExportNuage::StepSeg()
{
   return NuagePutCam().Val().StepSeg();
}

const double & cExportNuage::StepSeg()const 
{
   return NuagePutCam().Val().StepSeg();
}


cTplValGesInit< double > & cExportNuage::StepImage()
{
   return NuagePutCam().Val().StepImage();
}

const cTplValGesInit< double > & cExportNuage::StepImage()const 
{
   return NuagePutCam().Val().StepImage();
}


cTplValGesInit< cNuagePutCam > & cExportNuage::NuagePutCam()
{
   return mNuagePutCam;
}

const cTplValGesInit< cNuagePutCam > & cExportNuage::NuagePutCam()const 
{
   return mNuagePutCam;
}

cElXMLTree * ToXMLTree(const cExportNuage & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ExportNuage",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("NameOut"),anObj.NameOut())->ReTagThis("NameOut"));
   if (anObj.PlyModeBin().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PlyModeBin"),anObj.PlyModeBin().Val())->ReTagThis("PlyModeBin"));
  for
  (       std::list< std::string >::const_iterator it=anObj.NameRefLiaison().begin();
      it !=anObj.NameRefLiaison().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("NameRefLiaison"),(*it))->ReTagThis("NameRefLiaison"));
   if (anObj.PatternSel().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PatternSel"),anObj.PatternSel().Val())->ReTagThis("PatternSel"));
   aRes->AddFils(ToXMLTree(anObj.Pond())->ReTagThis("Pond"));
   aRes->AddFils(::ToXMLTree(std::string("KeyFileColImage"),anObj.KeyFileColImage())->ReTagThis("KeyFileColImage"));
   if (anObj.NbChan().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NbChan"),anObj.NbChan().Val())->ReTagThis("NbChan"));
   if (anObj.DirCol().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DirCol"),anObj.DirCol().Val())->ReTagThis("DirCol"));
   if (anObj.PerCol().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PerCol"),anObj.PerCol().Val())->ReTagThis("PerCol"));
   if (anObj.LimBSurH().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("LimBSurH"),anObj.LimBSurH().Val())->ReTagThis("LimBSurH"));
   if (anObj.ImExpoRef().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ImExpoRef"),anObj.ImExpoRef().Val())->ReTagThis("ImExpoRef"));
   if (anObj.NuagePutCam().IsInit())
      aRes->AddFils(ToXMLTree(anObj.NuagePutCam().Val())->ReTagThis("NuagePutCam"));
  return aRes;
}

void xml_init(cExportNuage & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.NameOut(),aTree->Get("NameOut",1)); //tototo 

   xml_init(anObj.PlyModeBin(),aTree->Get("PlyModeBin",1),bool(true)); //tototo 

   xml_init(anObj.NameRefLiaison(),aTree->GetAll("NameRefLiaison",false,1));

   xml_init(anObj.PatternSel(),aTree->Get("PatternSel",1)); //tototo 

   xml_init(anObj.Pond(),aTree->Get("Pond",1)); //tototo 

   xml_init(anObj.KeyFileColImage(),aTree->Get("KeyFileColImage",1)); //tototo 

   xml_init(anObj.NbChan(),aTree->Get("NbChan",1),int(-1)); //tototo 

   xml_init(anObj.DirCol(),aTree->Get("DirCol",1),Pt3dr(Pt3dr(0,0,1))); //tototo 

   xml_init(anObj.PerCol(),aTree->Get("PerCol",1),double(2.0)); //tototo 

   xml_init(anObj.LimBSurH(),aTree->Get("LimBSurH",1),double(1e-2)); //tototo 

   xml_init(anObj.ImExpoRef(),aTree->Get("ImExpoRef",1)); //tototo 

   xml_init(anObj.NuagePutCam(),aTree->Get("NuagePutCam",1)); //tototo 
}


cTplValGesInit< std::string > & cChoixImSec::PatternSel()
{
   return mPatternSel;
}

const cTplValGesInit< std::string > & cChoixImSec::PatternSel()const 
{
   return mPatternSel;
}


int & cChoixImSec::NbMin()
{
   return mNbMin;
}

const int & cChoixImSec::NbMin()const 
{
   return mNbMin;
}


std::string & cChoixImSec::IdBdl()
{
   return mIdBdl;
}

const std::string & cChoixImSec::IdBdl()const 
{
   return mIdBdl;
}


cTplValGesInit< int > & cChoixImSec::NbMaxPresel()
{
   return mNbMaxPresel;
}

const cTplValGesInit< int > & cChoixImSec::NbMaxPresel()const 
{
   return mNbMaxPresel;
}


cTplValGesInit< int > & cChoixImSec::NbMinPtsHom()
{
   return mNbMinPtsHom;
}

const cTplValGesInit< int > & cChoixImSec::NbMinPtsHom()const 
{
   return mNbMinPtsHom;
}


cTplValGesInit< double > & cChoixImSec::TetaMaxPreSel()
{
   return mTetaMaxPreSel;
}

const cTplValGesInit< double > & cChoixImSec::TetaMaxPreSel()const 
{
   return mTetaMaxPreSel;
}


cTplValGesInit< int > & cChoixImSec::NbMinPresel()
{
   return mNbMinPresel;
}

const cTplValGesInit< int > & cChoixImSec::NbMinPresel()const 
{
   return mNbMinPresel;
}


cTplValGesInit< double > & cChoixImSec::TetaOpt()
{
   return mTetaOpt;
}

const cTplValGesInit< double > & cChoixImSec::TetaOpt()const 
{
   return mTetaOpt;
}

cElXMLTree * ToXMLTree(const cChoixImSec & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ChoixImSec",eXMLBranche);
   if (anObj.PatternSel().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PatternSel"),anObj.PatternSel().Val())->ReTagThis("PatternSel"));
   aRes->AddFils(::ToXMLTree(std::string("NbMin"),anObj.NbMin())->ReTagThis("NbMin"));
   aRes->AddFils(::ToXMLTree(std::string("IdBdl"),anObj.IdBdl())->ReTagThis("IdBdl"));
   if (anObj.NbMaxPresel().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NbMaxPresel"),anObj.NbMaxPresel().Val())->ReTagThis("NbMaxPresel"));
   if (anObj.NbMinPtsHom().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NbMinPtsHom"),anObj.NbMinPtsHom().Val())->ReTagThis("NbMinPtsHom"));
   if (anObj.TetaMaxPreSel().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("TetaMaxPreSel"),anObj.TetaMaxPreSel().Val())->ReTagThis("TetaMaxPreSel"));
   if (anObj.NbMinPresel().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NbMinPresel"),anObj.NbMinPresel().Val())->ReTagThis("NbMinPresel"));
   if (anObj.TetaOpt().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("TetaOpt"),anObj.TetaOpt().Val())->ReTagThis("TetaOpt"));
  return aRes;
}

void xml_init(cChoixImSec & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.PatternSel(),aTree->Get("PatternSel",1),std::string(".*")); //tototo 

   xml_init(anObj.NbMin(),aTree->Get("NbMin",1)); //tototo 

   xml_init(anObj.IdBdl(),aTree->Get("IdBdl",1)); //tototo 

   xml_init(anObj.NbMaxPresel(),aTree->Get("NbMaxPresel",1),int(12)); //tototo 

   xml_init(anObj.NbMinPtsHom(),aTree->Get("NbMinPtsHom",1),int(5)); //tototo 

   xml_init(anObj.TetaMaxPreSel(),aTree->Get("TetaMaxPreSel",1),double(0.30)); //tototo 

   xml_init(anObj.NbMinPresel(),aTree->Get("NbMinPresel",1),int(6)); //tototo 

   xml_init(anObj.TetaOpt(),aTree->Get("TetaOpt",1),double(0.15)); //tototo 
}


cTplValGesInit< std::string > & cChoixImMM::PatternSel()
{
   return ChoixImSec().PatternSel();
}

const cTplValGesInit< std::string > & cChoixImMM::PatternSel()const 
{
   return ChoixImSec().PatternSel();
}


int & cChoixImMM::NbMin()
{
   return ChoixImSec().NbMin();
}

const int & cChoixImMM::NbMin()const 
{
   return ChoixImSec().NbMin();
}


std::string & cChoixImMM::IdBdl()
{
   return ChoixImSec().IdBdl();
}

const std::string & cChoixImMM::IdBdl()const 
{
   return ChoixImSec().IdBdl();
}


cTplValGesInit< int > & cChoixImMM::NbMaxPresel()
{
   return ChoixImSec().NbMaxPresel();
}

const cTplValGesInit< int > & cChoixImMM::NbMaxPresel()const 
{
   return ChoixImSec().NbMaxPresel();
}


cTplValGesInit< int > & cChoixImMM::NbMinPtsHom()
{
   return ChoixImSec().NbMinPtsHom();
}

const cTplValGesInit< int > & cChoixImMM::NbMinPtsHom()const 
{
   return ChoixImSec().NbMinPtsHom();
}


cTplValGesInit< double > & cChoixImMM::TetaMaxPreSel()
{
   return ChoixImSec().TetaMaxPreSel();
}

const cTplValGesInit< double > & cChoixImMM::TetaMaxPreSel()const 
{
   return ChoixImSec().TetaMaxPreSel();
}


cTplValGesInit< int > & cChoixImMM::NbMinPresel()
{
   return ChoixImSec().NbMinPresel();
}

const cTplValGesInit< int > & cChoixImMM::NbMinPresel()const 
{
   return ChoixImSec().NbMinPresel();
}


cTplValGesInit< double > & cChoixImMM::TetaOpt()
{
   return ChoixImSec().TetaOpt();
}

const cTplValGesInit< double > & cChoixImMM::TetaOpt()const 
{
   return ChoixImSec().TetaOpt();
}


cChoixImSec & cChoixImMM::ChoixImSec()
{
   return mChoixImSec;
}

const cChoixImSec & cChoixImMM::ChoixImSec()const 
{
   return mChoixImSec;
}

cElXMLTree * ToXMLTree(const cChoixImMM & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ChoixImMM",eXMLBranche);
   aRes->AddFils(ToXMLTree(anObj.ChoixImSec())->ReTagThis("ChoixImSec"));
  return aRes;
}

void xml_init(cChoixImMM & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.ChoixImSec(),aTree->Get("ChoixImSec",1)); //tototo 
}


std::list< cExportCalib > & cSectionExport::ExportCalib()
{
   return mExportCalib;
}

const std::list< cExportCalib > & cSectionExport::ExportCalib()const 
{
   return mExportCalib;
}


std::list< cExportPose > & cSectionExport::ExportPose()
{
   return mExportPose;
}

const std::list< cExportPose > & cSectionExport::ExportPose()const 
{
   return mExportPose;
}


std::list< cExportAttrPose > & cSectionExport::ExportAttrPose()
{
   return mExportAttrPose;
}

const std::list< cExportAttrPose > & cSectionExport::ExportAttrPose()const 
{
   return mExportAttrPose;
}


std::list< cExportRepereLoc > & cSectionExport::ExportRepereLoc()
{
   return mExportRepereLoc;
}

const std::list< cExportRepereLoc > & cSectionExport::ExportRepereLoc()const 
{
   return mExportRepereLoc;
}


std::list< cExportMesuresFromCarteProf > & cSectionExport::ExportMesuresFromCarteProf()
{
   return mExportMesuresFromCarteProf;
}

const std::list< cExportMesuresFromCarteProf > & cSectionExport::ExportMesuresFromCarteProf()const 
{
   return mExportMesuresFromCarteProf;
}


std::list< cExportVisuConfigGrpPose > & cSectionExport::ExportVisuConfigGrpPose()
{
   return mExportVisuConfigGrpPose;
}

const std::list< cExportVisuConfigGrpPose > & cSectionExport::ExportVisuConfigGrpPose()const 
{
   return mExportVisuConfigGrpPose;
}


cTplValGesInit< cExportPtsFlottant > & cSectionExport::ExportPtsFlottant()
{
   return mExportPtsFlottant;
}

const cTplValGesInit< cExportPtsFlottant > & cSectionExport::ExportPtsFlottant()const 
{
   return mExportPtsFlottant;
}


std::list< cExportImResiduLiaison > & cSectionExport::ExportImResiduLiaison()
{
   return mExportImResiduLiaison;
}

const std::list< cExportImResiduLiaison > & cSectionExport::ExportImResiduLiaison()const 
{
   return mExportImResiduLiaison;
}


std::list< cExportRedressement > & cSectionExport::ExportRedressement()
{
   return mExportRedressement;
}

const std::list< cExportRedressement > & cSectionExport::ExportRedressement()const 
{
   return mExportRedressement;
}


std::list< cExportNuage > & cSectionExport::ExportNuage()
{
   return mExportNuage;
}

const std::list< cExportNuage > & cSectionExport::ExportNuage()const 
{
   return mExportNuage;
}


cTplValGesInit< std::string > & cSectionExport::PatternSel()
{
   return ChoixImMM().Val().ChoixImSec().PatternSel();
}

const cTplValGesInit< std::string > & cSectionExport::PatternSel()const 
{
   return ChoixImMM().Val().ChoixImSec().PatternSel();
}


int & cSectionExport::NbMin()
{
   return ChoixImMM().Val().ChoixImSec().NbMin();
}

const int & cSectionExport::NbMin()const 
{
   return ChoixImMM().Val().ChoixImSec().NbMin();
}


std::string & cSectionExport::IdBdl()
{
   return ChoixImMM().Val().ChoixImSec().IdBdl();
}

const std::string & cSectionExport::IdBdl()const 
{
   return ChoixImMM().Val().ChoixImSec().IdBdl();
}


cTplValGesInit< int > & cSectionExport::NbMaxPresel()
{
   return ChoixImMM().Val().ChoixImSec().NbMaxPresel();
}

const cTplValGesInit< int > & cSectionExport::NbMaxPresel()const 
{
   return ChoixImMM().Val().ChoixImSec().NbMaxPresel();
}


cTplValGesInit< int > & cSectionExport::NbMinPtsHom()
{
   return ChoixImMM().Val().ChoixImSec().NbMinPtsHom();
}

const cTplValGesInit< int > & cSectionExport::NbMinPtsHom()const 
{
   return ChoixImMM().Val().ChoixImSec().NbMinPtsHom();
}


cTplValGesInit< double > & cSectionExport::TetaMaxPreSel()
{
   return ChoixImMM().Val().ChoixImSec().TetaMaxPreSel();
}

const cTplValGesInit< double > & cSectionExport::TetaMaxPreSel()const 
{
   return ChoixImMM().Val().ChoixImSec().TetaMaxPreSel();
}


cTplValGesInit< int > & cSectionExport::NbMinPresel()
{
   return ChoixImMM().Val().ChoixImSec().NbMinPresel();
}

const cTplValGesInit< int > & cSectionExport::NbMinPresel()const 
{
   return ChoixImMM().Val().ChoixImSec().NbMinPresel();
}


cTplValGesInit< double > & cSectionExport::TetaOpt()
{
   return ChoixImMM().Val().ChoixImSec().TetaOpt();
}

const cTplValGesInit< double > & cSectionExport::TetaOpt()const 
{
   return ChoixImMM().Val().ChoixImSec().TetaOpt();
}


cChoixImSec & cSectionExport::ChoixImSec()
{
   return ChoixImMM().Val().ChoixImSec();
}

const cChoixImSec & cSectionExport::ChoixImSec()const 
{
   return ChoixImMM().Val().ChoixImSec();
}


cTplValGesInit< cChoixImMM > & cSectionExport::ChoixImMM()
{
   return mChoixImMM;
}

const cTplValGesInit< cChoixImMM > & cSectionExport::ChoixImMM()const 
{
   return mChoixImMM;
}

cElXMLTree * ToXMLTree(const cSectionExport & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"SectionExport",eXMLBranche);
  for
  (       std::list< cExportCalib >::const_iterator it=anObj.ExportCalib().begin();
      it !=anObj.ExportCalib().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("ExportCalib"));
  for
  (       std::list< cExportPose >::const_iterator it=anObj.ExportPose().begin();
      it !=anObj.ExportPose().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("ExportPose"));
  for
  (       std::list< cExportAttrPose >::const_iterator it=anObj.ExportAttrPose().begin();
      it !=anObj.ExportAttrPose().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("ExportAttrPose"));
  for
  (       std::list< cExportRepereLoc >::const_iterator it=anObj.ExportRepereLoc().begin();
      it !=anObj.ExportRepereLoc().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("ExportRepereLoc"));
  for
  (       std::list< cExportMesuresFromCarteProf >::const_iterator it=anObj.ExportMesuresFromCarteProf().begin();
      it !=anObj.ExportMesuresFromCarteProf().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("ExportMesuresFromCarteProf"));
  for
  (       std::list< cExportVisuConfigGrpPose >::const_iterator it=anObj.ExportVisuConfigGrpPose().begin();
      it !=anObj.ExportVisuConfigGrpPose().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("ExportVisuConfigGrpPose"));
   if (anObj.ExportPtsFlottant().IsInit())
      aRes->AddFils(ToXMLTree(anObj.ExportPtsFlottant().Val())->ReTagThis("ExportPtsFlottant"));
  for
  (       std::list< cExportImResiduLiaison >::const_iterator it=anObj.ExportImResiduLiaison().begin();
      it !=anObj.ExportImResiduLiaison().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("ExportImResiduLiaison"));
  for
  (       std::list< cExportRedressement >::const_iterator it=anObj.ExportRedressement().begin();
      it !=anObj.ExportRedressement().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("ExportRedressement"));
  for
  (       std::list< cExportNuage >::const_iterator it=anObj.ExportNuage().begin();
      it !=anObj.ExportNuage().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("ExportNuage"));
   if (anObj.ChoixImMM().IsInit())
      aRes->AddFils(ToXMLTree(anObj.ChoixImMM().Val())->ReTagThis("ChoixImMM"));
  return aRes;
}

void xml_init(cSectionExport & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.ExportCalib(),aTree->GetAll("ExportCalib",false,1));

   xml_init(anObj.ExportPose(),aTree->GetAll("ExportPose",false,1));

   xml_init(anObj.ExportAttrPose(),aTree->GetAll("ExportAttrPose",false,1));

   xml_init(anObj.ExportRepereLoc(),aTree->GetAll("ExportRepereLoc",false,1));

   xml_init(anObj.ExportMesuresFromCarteProf(),aTree->GetAll("ExportMesuresFromCarteProf",false,1));

   xml_init(anObj.ExportVisuConfigGrpPose(),aTree->GetAll("ExportVisuConfigGrpPose",false,1));

   xml_init(anObj.ExportPtsFlottant(),aTree->Get("ExportPtsFlottant",1)); //tototo 

   xml_init(anObj.ExportImResiduLiaison(),aTree->GetAll("ExportImResiduLiaison",false,1));

   xml_init(anObj.ExportRedressement(),aTree->GetAll("ExportRedressement",false,1));

   xml_init(anObj.ExportNuage(),aTree->GetAll("ExportNuage",false,1));

   xml_init(anObj.ChoixImMM(),aTree->Get("ChoixImMM",1)); //tototo 
}


std::vector< cIterationsCompensation > & cEtapeCompensation::IterationsCompensation()
{
   return mIterationsCompensation;
}

const std::vector< cIterationsCompensation > & cEtapeCompensation::IterationsCompensation()const 
{
   return mIterationsCompensation;
}


std::list< cTraceCpleCam > & cEtapeCompensation::TraceCpleCam()
{
   return SectionTracage().Val().TraceCpleCam();
}

const std::list< cTraceCpleCam > & cEtapeCompensation::TraceCpleCam()const 
{
   return SectionTracage().Val().TraceCpleCam();
}


cTplValGesInit< bool > & cEtapeCompensation::GetChar()
{
   return SectionTracage().Val().GetChar();
}

const cTplValGesInit< bool > & cEtapeCompensation::GetChar()const 
{
   return SectionTracage().Val().GetChar();
}


cTplValGesInit< cSectionTracage > & cEtapeCompensation::SectionTracage()
{
   return mSectionTracage;
}

const cTplValGesInit< cSectionTracage > & cEtapeCompensation::SectionTracage()const 
{
   return mSectionTracage;
}


cTplValGesInit< cSectionLevenbergMarkard > & cEtapeCompensation::SLMEtape()
{
   return mSLMEtape;
}

const cTplValGesInit< cSectionLevenbergMarkard > & cEtapeCompensation::SLMEtape()const 
{
   return mSLMEtape;
}


cTplValGesInit< cSectionLevenbergMarkard > & cEtapeCompensation::SLMGlob()
{
   return mSLMGlob;
}

const cTplValGesInit< cSectionLevenbergMarkard > & cEtapeCompensation::SLMGlob()const 
{
   return mSLMGlob;
}


cTplValGesInit< double > & cEtapeCompensation::MultSLMEtape()
{
   return mMultSLMEtape;
}

const cTplValGesInit< double > & cEtapeCompensation::MultSLMEtape()const 
{
   return mMultSLMEtape;
}


cTplValGesInit< double > & cEtapeCompensation::MultSLMGlob()
{
   return mMultSLMGlob;
}

const cTplValGesInit< double > & cEtapeCompensation::MultSLMGlob()const 
{
   return mMultSLMGlob;
}


std::list< cObsAppuis > & cEtapeCompensation::ObsAppuis()
{
   return SectionObservations().ObsAppuis();
}

const std::list< cObsAppuis > & cEtapeCompensation::ObsAppuis()const 
{
   return SectionObservations().ObsAppuis();
}


std::list< cObsAppuisFlottant > & cEtapeCompensation::ObsAppuisFlottant()
{
   return SectionObservations().ObsAppuisFlottant();
}

const std::list< cObsAppuisFlottant > & cEtapeCompensation::ObsAppuisFlottant()const 
{
   return SectionObservations().ObsAppuisFlottant();
}


std::list< cObsLiaisons > & cEtapeCompensation::ObsLiaisons()
{
   return SectionObservations().ObsLiaisons();
}

const std::list< cObsLiaisons > & cEtapeCompensation::ObsLiaisons()const 
{
   return SectionObservations().ObsLiaisons();
}


std::list< cObsCentrePDV > & cEtapeCompensation::ObsCentrePDV()
{
   return SectionObservations().ObsCentrePDV();
}

const std::list< cObsCentrePDV > & cEtapeCompensation::ObsCentrePDV()const 
{
   return SectionObservations().ObsCentrePDV();
}


std::list< cObsRigidGrpImage > & cEtapeCompensation::ObsRigidGrpImage()
{
   return SectionObservations().ObsRigidGrpImage();
}

const std::list< cObsRigidGrpImage > & cEtapeCompensation::ObsRigidGrpImage()const 
{
   return SectionObservations().ObsRigidGrpImage();
}


std::string & cEtapeCompensation::NameFile()
{
   return SectionObservations().TxtRapDetaille().Val().NameFile();
}

const std::string & cEtapeCompensation::NameFile()const 
{
   return SectionObservations().TxtRapDetaille().Val().NameFile();
}


cTplValGesInit< cTxtRapDetaille > & cEtapeCompensation::TxtRapDetaille()
{
   return SectionObservations().TxtRapDetaille();
}

const cTplValGesInit< cTxtRapDetaille > & cEtapeCompensation::TxtRapDetaille()const 
{
   return SectionObservations().TxtRapDetaille();
}


cSectionObservations & cEtapeCompensation::SectionObservations()
{
   return mSectionObservations;
}

const cSectionObservations & cEtapeCompensation::SectionObservations()const 
{
   return mSectionObservations;
}


std::list< cExportCalib > & cEtapeCompensation::ExportCalib()
{
   return SectionExport().Val().ExportCalib();
}

const std::list< cExportCalib > & cEtapeCompensation::ExportCalib()const 
{
   return SectionExport().Val().ExportCalib();
}


std::list< cExportPose > & cEtapeCompensation::ExportPose()
{
   return SectionExport().Val().ExportPose();
}

const std::list< cExportPose > & cEtapeCompensation::ExportPose()const 
{
   return SectionExport().Val().ExportPose();
}


std::list< cExportAttrPose > & cEtapeCompensation::ExportAttrPose()
{
   return SectionExport().Val().ExportAttrPose();
}

const std::list< cExportAttrPose > & cEtapeCompensation::ExportAttrPose()const 
{
   return SectionExport().Val().ExportAttrPose();
}


std::list< cExportRepereLoc > & cEtapeCompensation::ExportRepereLoc()
{
   return SectionExport().Val().ExportRepereLoc();
}

const std::list< cExportRepereLoc > & cEtapeCompensation::ExportRepereLoc()const 
{
   return SectionExport().Val().ExportRepereLoc();
}


std::list< cExportMesuresFromCarteProf > & cEtapeCompensation::ExportMesuresFromCarteProf()
{
   return SectionExport().Val().ExportMesuresFromCarteProf();
}

const std::list< cExportMesuresFromCarteProf > & cEtapeCompensation::ExportMesuresFromCarteProf()const 
{
   return SectionExport().Val().ExportMesuresFromCarteProf();
}


std::list< cExportVisuConfigGrpPose > & cEtapeCompensation::ExportVisuConfigGrpPose()
{
   return SectionExport().Val().ExportVisuConfigGrpPose();
}

const std::list< cExportVisuConfigGrpPose > & cEtapeCompensation::ExportVisuConfigGrpPose()const 
{
   return SectionExport().Val().ExportVisuConfigGrpPose();
}


cTplValGesInit< cExportPtsFlottant > & cEtapeCompensation::ExportPtsFlottant()
{
   return SectionExport().Val().ExportPtsFlottant();
}

const cTplValGesInit< cExportPtsFlottant > & cEtapeCompensation::ExportPtsFlottant()const 
{
   return SectionExport().Val().ExportPtsFlottant();
}


std::list< cExportImResiduLiaison > & cEtapeCompensation::ExportImResiduLiaison()
{
   return SectionExport().Val().ExportImResiduLiaison();
}

const std::list< cExportImResiduLiaison > & cEtapeCompensation::ExportImResiduLiaison()const 
{
   return SectionExport().Val().ExportImResiduLiaison();
}


std::list< cExportRedressement > & cEtapeCompensation::ExportRedressement()
{
   return SectionExport().Val().ExportRedressement();
}

const std::list< cExportRedressement > & cEtapeCompensation::ExportRedressement()const 
{
   return SectionExport().Val().ExportRedressement();
}


std::list< cExportNuage > & cEtapeCompensation::ExportNuage()
{
   return SectionExport().Val().ExportNuage();
}

const std::list< cExportNuage > & cEtapeCompensation::ExportNuage()const 
{
   return SectionExport().Val().ExportNuage();
}


cTplValGesInit< std::string > & cEtapeCompensation::PatternSel()
{
   return SectionExport().Val().ChoixImMM().Val().ChoixImSec().PatternSel();
}

const cTplValGesInit< std::string > & cEtapeCompensation::PatternSel()const 
{
   return SectionExport().Val().ChoixImMM().Val().ChoixImSec().PatternSel();
}


int & cEtapeCompensation::NbMin()
{
   return SectionExport().Val().ChoixImMM().Val().ChoixImSec().NbMin();
}

const int & cEtapeCompensation::NbMin()const 
{
   return SectionExport().Val().ChoixImMM().Val().ChoixImSec().NbMin();
}


std::string & cEtapeCompensation::IdBdl()
{
   return SectionExport().Val().ChoixImMM().Val().ChoixImSec().IdBdl();
}

const std::string & cEtapeCompensation::IdBdl()const 
{
   return SectionExport().Val().ChoixImMM().Val().ChoixImSec().IdBdl();
}


cTplValGesInit< int > & cEtapeCompensation::NbMaxPresel()
{
   return SectionExport().Val().ChoixImMM().Val().ChoixImSec().NbMaxPresel();
}

const cTplValGesInit< int > & cEtapeCompensation::NbMaxPresel()const 
{
   return SectionExport().Val().ChoixImMM().Val().ChoixImSec().NbMaxPresel();
}


cTplValGesInit< int > & cEtapeCompensation::NbMinPtsHom()
{
   return SectionExport().Val().ChoixImMM().Val().ChoixImSec().NbMinPtsHom();
}

const cTplValGesInit< int > & cEtapeCompensation::NbMinPtsHom()const 
{
   return SectionExport().Val().ChoixImMM().Val().ChoixImSec().NbMinPtsHom();
}


cTplValGesInit< double > & cEtapeCompensation::TetaMaxPreSel()
{
   return SectionExport().Val().ChoixImMM().Val().ChoixImSec().TetaMaxPreSel();
}

const cTplValGesInit< double > & cEtapeCompensation::TetaMaxPreSel()const 
{
   return SectionExport().Val().ChoixImMM().Val().ChoixImSec().TetaMaxPreSel();
}


cTplValGesInit< int > & cEtapeCompensation::NbMinPresel()
{
   return SectionExport().Val().ChoixImMM().Val().ChoixImSec().NbMinPresel();
}

const cTplValGesInit< int > & cEtapeCompensation::NbMinPresel()const 
{
   return SectionExport().Val().ChoixImMM().Val().ChoixImSec().NbMinPresel();
}


cTplValGesInit< double > & cEtapeCompensation::TetaOpt()
{
   return SectionExport().Val().ChoixImMM().Val().ChoixImSec().TetaOpt();
}

const cTplValGesInit< double > & cEtapeCompensation::TetaOpt()const 
{
   return SectionExport().Val().ChoixImMM().Val().ChoixImSec().TetaOpt();
}


cChoixImSec & cEtapeCompensation::ChoixImSec()
{
   return SectionExport().Val().ChoixImMM().Val().ChoixImSec();
}

const cChoixImSec & cEtapeCompensation::ChoixImSec()const 
{
   return SectionExport().Val().ChoixImMM().Val().ChoixImSec();
}


cTplValGesInit< cChoixImMM > & cEtapeCompensation::ChoixImMM()
{
   return SectionExport().Val().ChoixImMM();
}

const cTplValGesInit< cChoixImMM > & cEtapeCompensation::ChoixImMM()const 
{
   return SectionExport().Val().ChoixImMM();
}


cTplValGesInit< cSectionExport > & cEtapeCompensation::SectionExport()
{
   return mSectionExport;
}

const cTplValGesInit< cSectionExport > & cEtapeCompensation::SectionExport()const 
{
   return mSectionExport;
}

cElXMLTree * ToXMLTree(const cEtapeCompensation & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"EtapeCompensation",eXMLBranche);
  for
  (       std::vector< cIterationsCompensation >::const_iterator it=anObj.IterationsCompensation().begin();
      it !=anObj.IterationsCompensation().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("IterationsCompensation"));
   if (anObj.SectionTracage().IsInit())
      aRes->AddFils(ToXMLTree(anObj.SectionTracage().Val())->ReTagThis("SectionTracage"));
   if (anObj.SLMEtape().IsInit())
      aRes->AddFils(ToXMLTree(anObj.SLMEtape().Val())->ReTagThis("SLMEtape"));
   if (anObj.SLMGlob().IsInit())
      aRes->AddFils(ToXMLTree(anObj.SLMGlob().Val())->ReTagThis("SLMGlob"));
   if (anObj.MultSLMEtape().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("MultSLMEtape"),anObj.MultSLMEtape().Val())->ReTagThis("MultSLMEtape"));
   if (anObj.MultSLMGlob().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("MultSLMGlob"),anObj.MultSLMGlob().Val())->ReTagThis("MultSLMGlob"));
   aRes->AddFils(ToXMLTree(anObj.SectionObservations())->ReTagThis("SectionObservations"));
   if (anObj.SectionExport().IsInit())
      aRes->AddFils(ToXMLTree(anObj.SectionExport().Val())->ReTagThis("SectionExport"));
  return aRes;
}

void xml_init(cEtapeCompensation & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.IterationsCompensation(),aTree->GetAll("IterationsCompensation",false,1));

   xml_init(anObj.SectionTracage(),aTree->Get("SectionTracage",1)); //tototo 

   xml_init(anObj.SLMEtape(),aTree->Get("SLMEtape",1)); //tototo 

   xml_init(anObj.SLMGlob(),aTree->Get("SLMGlob",1)); //tototo 

   xml_init(anObj.MultSLMEtape(),aTree->Get("MultSLMEtape",1)); //tototo 

   xml_init(anObj.MultSLMGlob(),aTree->Get("MultSLMGlob",1)); //tototo 

   xml_init(anObj.SectionObservations(),aTree->Get("SectionObservations",1)); //tototo 

   xml_init(anObj.SectionExport(),aTree->Get("SectionExport",1)); //tototo 
}


std::list< cEtapeCompensation > & cSectionCompensation::EtapeCompensation()
{
   return mEtapeCompensation;
}

const std::list< cEtapeCompensation > & cSectionCompensation::EtapeCompensation()const 
{
   return mEtapeCompensation;
}

cElXMLTree * ToXMLTree(const cSectionCompensation & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"SectionCompensation",eXMLBranche);
  for
  (       std::list< cEtapeCompensation >::const_iterator it=anObj.EtapeCompensation().begin();
      it !=anObj.EtapeCompensation().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("EtapeCompensation"));
  return aRes;
}

void xml_init(cSectionCompensation & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.EtapeCompensation(),aTree->GetAll("EtapeCompensation",false,1));
}


cTplValGesInit< cChantierDescripteur > & cParamApero::DicoLoc()
{
   return mDicoLoc;
}

const cTplValGesInit< cChantierDescripteur > & cParamApero::DicoLoc()const 
{
   return mDicoLoc;
}


cTplValGesInit< bool > & cParamApero::ShowMes()
{
   return ShowSection().Val().ShowMes();
}

const cTplValGesInit< bool > & cParamApero::ShowMes()const 
{
   return ShowSection().Val().ShowMes();
}


cTplValGesInit< std::string > & cParamApero::LogFile()
{
   return ShowSection().Val().LogFile();
}

const cTplValGesInit< std::string > & cParamApero::LogFile()const 
{
   return ShowSection().Val().LogFile();
}


cTplValGesInit< cShowSection > & cParamApero::ShowSection()
{
   return mShowSection;
}

const cTplValGesInit< cShowSection > & cParamApero::ShowSection()const 
{
   return mShowSection;
}


cTplValGesInit< bool > & cParamApero::CalledByItself()
{
   return mCalledByItself;
}

const cTplValGesInit< bool > & cParamApero::CalledByItself()const 
{
   return mCalledByItself;
}


cTplValGesInit< cCmdMappeur > & cParamApero::SectionMapApero()
{
   return mSectionMapApero;
}

const cTplValGesInit< cCmdMappeur > & cParamApero::SectionMapApero()const 
{
   return mSectionMapApero;
}


std::list< cBDD_PtsLiaisons > & cParamApero::BDD_PtsLiaisons()
{
   return SectionBDD_Observation().BDD_PtsLiaisons();
}

const std::list< cBDD_PtsLiaisons > & cParamApero::BDD_PtsLiaisons()const 
{
   return SectionBDD_Observation().BDD_PtsLiaisons();
}


std::list< cBDD_PtsAppuis > & cParamApero::BDD_PtsAppuis()
{
   return SectionBDD_Observation().BDD_PtsAppuis();
}

const std::list< cBDD_PtsAppuis > & cParamApero::BDD_PtsAppuis()const 
{
   return SectionBDD_Observation().BDD_PtsAppuis();
}


std::list< cBDD_ObsAppuisFlottant > & cParamApero::BDD_ObsAppuisFlottant()
{
   return SectionBDD_Observation().BDD_ObsAppuisFlottant();
}

const std::list< cBDD_ObsAppuisFlottant > & cParamApero::BDD_ObsAppuisFlottant()const 
{
   return SectionBDD_Observation().BDD_ObsAppuisFlottant();
}


std::list< cBDD_Orient > & cParamApero::BDD_Orient()
{
   return SectionBDD_Observation().BDD_Orient();
}

const std::list< cBDD_Orient > & cParamApero::BDD_Orient()const 
{
   return SectionBDD_Observation().BDD_Orient();
}


std::list< cBDD_Centre > & cParamApero::BDD_Centre()
{
   return SectionBDD_Observation().BDD_Centre();
}

const std::list< cBDD_Centre > & cParamApero::BDD_Centre()const 
{
   return SectionBDD_Observation().BDD_Centre();
}


std::list< cFilterProj3D > & cParamApero::FilterProj3D()
{
   return SectionBDD_Observation().FilterProj3D();
}

const std::list< cFilterProj3D > & cParamApero::FilterProj3D()const 
{
   return SectionBDD_Observation().FilterProj3D();
}


std::list< cLayerImageToPose > & cParamApero::LayerImageToPose()
{
   return SectionBDD_Observation().LayerImageToPose();
}

const std::list< cLayerImageToPose > & cParamApero::LayerImageToPose()const 
{
   return SectionBDD_Observation().LayerImageToPose();
}


cTplValGesInit< double > & cParamApero::LimInfBSurHPMoy()
{
   return SectionBDD_Observation().LimInfBSurHPMoy();
}

const cTplValGesInit< double > & cParamApero::LimInfBSurHPMoy()const 
{
   return SectionBDD_Observation().LimInfBSurHPMoy();
}


cTplValGesInit< double > & cParamApero::LimSupBSurHPMoy()
{
   return SectionBDD_Observation().LimSupBSurHPMoy();
}

const cTplValGesInit< double > & cParamApero::LimSupBSurHPMoy()const 
{
   return SectionBDD_Observation().LimSupBSurHPMoy();
}


cSectionBDD_Observation & cParamApero::SectionBDD_Observation()
{
   return mSectionBDD_Observation;
}

const cSectionBDD_Observation & cParamApero::SectionBDD_Observation()const 
{
   return mSectionBDD_Observation;
}


cTplValGesInit< double > & cParamApero::SeuilAutomFE()
{
   return SectionInconnues().SeuilAutomFE();
}

const cTplValGesInit< double > & cParamApero::SeuilAutomFE()const 
{
   return SectionInconnues().SeuilAutomFE();
}


cTplValGesInit< bool > & cParamApero::AutoriseToujoursUneSeuleLiaison()
{
   return SectionInconnues().AutoriseToujoursUneSeuleLiaison();
}

const cTplValGesInit< bool > & cParamApero::AutoriseToujoursUneSeuleLiaison()const 
{
   return SectionInconnues().AutoriseToujoursUneSeuleLiaison();
}


cTplValGesInit< cMapName2Name > & cParamApero::MapMaskHom()
{
   return SectionInconnues().MapMaskHom();
}

const cTplValGesInit< cMapName2Name > & cParamApero::MapMaskHom()const 
{
   return SectionInconnues().MapMaskHom();
}


cTplValGesInit< bool > & cParamApero::SauvePMoyenOnlyWithMasq()
{
   return SectionInconnues().SauvePMoyenOnlyWithMasq();
}

const cTplValGesInit< bool > & cParamApero::SauvePMoyenOnlyWithMasq()const 
{
   return SectionInconnues().SauvePMoyenOnlyWithMasq();
}


std::list< cCalibrationCameraInc > & cParamApero::CalibrationCameraInc()
{
   return SectionInconnues().CalibrationCameraInc();
}

const std::list< cCalibrationCameraInc > & cParamApero::CalibrationCameraInc()const 
{
   return SectionInconnues().CalibrationCameraInc();
}


cTplValGesInit< int > & cParamApero::SeuilL1EstimMatrEss()
{
   return SectionInconnues().SeuilL1EstimMatrEss();
}

const cTplValGesInit< int > & cParamApero::SeuilL1EstimMatrEss()const 
{
   return SectionInconnues().SeuilL1EstimMatrEss();
}


cTplValGesInit< cSetOrientationInterne > & cParamApero::GlobOrInterne()
{
   return SectionInconnues().GlobOrInterne();
}

const cTplValGesInit< cSetOrientationInterne > & cParamApero::GlobOrInterne()const 
{
   return SectionInconnues().GlobOrInterne();
}


std::list< cPoseCameraInc > & cParamApero::PoseCameraInc()
{
   return SectionInconnues().PoseCameraInc();
}

const std::list< cPoseCameraInc > & cParamApero::PoseCameraInc()const 
{
   return SectionInconnues().PoseCameraInc();
}


std::list< cGroupeDePose > & cParamApero::GroupeDePose()
{
   return SectionInconnues().GroupeDePose();
}

const std::list< cGroupeDePose > & cParamApero::GroupeDePose()const 
{
   return SectionInconnues().GroupeDePose();
}


std::list< cSurfParamInc > & cParamApero::SurfParamInc()
{
   return SectionInconnues().SurfParamInc();
}

const std::list< cSurfParamInc > & cParamApero::SurfParamInc()const 
{
   return SectionInconnues().SurfParamInc();
}


std::list< cPointFlottantInc > & cParamApero::PointFlottantInc()
{
   return SectionInconnues().PointFlottantInc();
}

const std::list< cPointFlottantInc > & cParamApero::PointFlottantInc()const 
{
   return SectionInconnues().PointFlottantInc();
}


cSectionInconnues & cParamApero::SectionInconnues()
{
   return mSectionInconnues;
}

const cSectionInconnues & cParamApero::SectionInconnues()const 
{
   return mSectionInconnues;
}


cTplValGesInit< std::string > & cParamApero::FileSauvParam()
{
   return SectionChantier().FileSauvParam();
}

const cTplValGesInit< std::string > & cParamApero::FileSauvParam()const 
{
   return SectionChantier().FileSauvParam();
}


cTplValGesInit< bool > & cParamApero::GenereErreurOnContraineCam()
{
   return SectionChantier().GenereErreurOnContraineCam();
}

const cTplValGesInit< bool > & cParamApero::GenereErreurOnContraineCam()const 
{
   return SectionChantier().GenereErreurOnContraineCam();
}


cTplValGesInit< double > & cParamApero::ProfSceneChantier()
{
   return SectionChantier().ProfSceneChantier();
}

const cTplValGesInit< double > & cParamApero::ProfSceneChantier()const 
{
   return SectionChantier().ProfSceneChantier();
}


cTplValGesInit< std::string > & cParamApero::DirectoryChantier()
{
   return SectionChantier().DirectoryChantier();
}

const cTplValGesInit< std::string > & cParamApero::DirectoryChantier()const 
{
   return SectionChantier().DirectoryChantier();
}


cTplValGesInit< string > & cParamApero::FileChantierNameDescripteur()
{
   return SectionChantier().FileChantierNameDescripteur();
}

const cTplValGesInit< string > & cParamApero::FileChantierNameDescripteur()const 
{
   return SectionChantier().FileChantierNameDescripteur();
}


cTplValGesInit< std::string > & cParamApero::NameParamEtal()
{
   return SectionChantier().NameParamEtal();
}

const cTplValGesInit< std::string > & cParamApero::NameParamEtal()const 
{
   return SectionChantier().NameParamEtal();
}


cTplValGesInit< std::string > & cParamApero::PatternTracePose()
{
   return SectionChantier().PatternTracePose();
}

const cTplValGesInit< std::string > & cParamApero::PatternTracePose()const 
{
   return SectionChantier().PatternTracePose();
}


cTplValGesInit< bool > & cParamApero::TraceGimbalLock()
{
   return SectionChantier().TraceGimbalLock();
}

const cTplValGesInit< bool > & cParamApero::TraceGimbalLock()const 
{
   return SectionChantier().TraceGimbalLock();
}


cTplValGesInit< double > & cParamApero::MaxDistErrorPtsTerr()
{
   return SectionChantier().MaxDistErrorPtsTerr();
}

const cTplValGesInit< double > & cParamApero::MaxDistErrorPtsTerr()const 
{
   return SectionChantier().MaxDistErrorPtsTerr();
}


cTplValGesInit< double > & cParamApero::MaxDistWarnPtsTerr()
{
   return SectionChantier().MaxDistWarnPtsTerr();
}

const cTplValGesInit< double > & cParamApero::MaxDistWarnPtsTerr()const 
{
   return SectionChantier().MaxDistWarnPtsTerr();
}


cTplValGesInit< cShowPbLiaison > & cParamApero::DefPbLiaison()
{
   return SectionChantier().DefPbLiaison();
}

const cTplValGesInit< cShowPbLiaison > & cParamApero::DefPbLiaison()const 
{
   return SectionChantier().DefPbLiaison();
}


cTplValGesInit< bool > & cParamApero::DoCompensation()
{
   return SectionChantier().DoCompensation();
}

const cTplValGesInit< bool > & cParamApero::DoCompensation()const 
{
   return SectionChantier().DoCompensation();
}


double & cParamApero::DeltaMax()
{
   return SectionChantier().TimeLinkage().Val().DeltaMax();
}

const double & cParamApero::DeltaMax()const 
{
   return SectionChantier().TimeLinkage().Val().DeltaMax();
}


cTplValGesInit< cTimeLinkage > & cParamApero::TimeLinkage()
{
   return SectionChantier().TimeLinkage();
}

const cTplValGesInit< cTimeLinkage > & cParamApero::TimeLinkage()const 
{
   return SectionChantier().TimeLinkage();
}


cTplValGesInit< bool > & cParamApero::DebugPbCondFaisceau()
{
   return SectionChantier().DebugPbCondFaisceau();
}

const cTplValGesInit< bool > & cParamApero::DebugPbCondFaisceau()const 
{
   return SectionChantier().DebugPbCondFaisceau();
}


cTplValGesInit< std::string > & cParamApero::SauvAutom()
{
   return SectionChantier().SauvAutom();
}

const cTplValGesInit< std::string > & cParamApero::SauvAutom()const 
{
   return SectionChantier().SauvAutom();
}


cTplValGesInit< bool > & cParamApero::SauvAutomBasic()
{
   return SectionChantier().SauvAutomBasic();
}

const cTplValGesInit< bool > & cParamApero::SauvAutomBasic()const 
{
   return SectionChantier().SauvAutomBasic();
}


cTplValGesInit< double > & cParamApero::ThresholdWarnPointsBehind()
{
   return SectionChantier().ThresholdWarnPointsBehind();
}

const cTplValGesInit< double > & cParamApero::ThresholdWarnPointsBehind()const 
{
   return SectionChantier().ThresholdWarnPointsBehind();
}


cSectionChantier & cParamApero::SectionChantier()
{
   return mSectionChantier;
}

const cSectionChantier & cParamApero::SectionChantier()const 
{
   return mSectionChantier;
}


cTplValGesInit< bool > & cParamApero::AllMatSym()
{
   return SectionSolveur().AllMatSym();
}

const cTplValGesInit< bool > & cParamApero::AllMatSym()const 
{
   return SectionSolveur().AllMatSym();
}


eModeSolveurEq & cParamApero::ModeResolution()
{
   return SectionSolveur().ModeResolution();
}

const eModeSolveurEq & cParamApero::ModeResolution()const 
{
   return SectionSolveur().ModeResolution();
}


cTplValGesInit< eControleDescDic > & cParamApero::ModeControleDescDic()
{
   return SectionSolveur().ModeControleDescDic();
}

const cTplValGesInit< eControleDescDic > & cParamApero::ModeControleDescDic()const 
{
   return SectionSolveur().ModeControleDescDic();
}


cTplValGesInit< int > & cParamApero::SeuilBas_CDD()
{
   return SectionSolveur().SeuilBas_CDD();
}

const cTplValGesInit< int > & cParamApero::SeuilBas_CDD()const 
{
   return SectionSolveur().SeuilBas_CDD();
}


cTplValGesInit< int > & cParamApero::SeuilHaut_CDD()
{
   return SectionSolveur().SeuilHaut_CDD();
}

const cTplValGesInit< int > & cParamApero::SeuilHaut_CDD()const 
{
   return SectionSolveur().SeuilHaut_CDD();
}


cTplValGesInit< bool > & cParamApero::InhibeAMD()
{
   return SectionSolveur().InhibeAMD();
}

const cTplValGesInit< bool > & cParamApero::InhibeAMD()const 
{
   return SectionSolveur().InhibeAMD();
}


cTplValGesInit< bool > & cParamApero::AMDSpecInterne()
{
   return SectionSolveur().AMDSpecInterne();
}

const cTplValGesInit< bool > & cParamApero::AMDSpecInterne()const 
{
   return SectionSolveur().AMDSpecInterne();
}


cTplValGesInit< bool > & cParamApero::ShowCholesky()
{
   return SectionSolveur().ShowCholesky();
}

const cTplValGesInit< bool > & cParamApero::ShowCholesky()const 
{
   return SectionSolveur().ShowCholesky();
}


cTplValGesInit< bool > & cParamApero::TestPermutVar()
{
   return SectionSolveur().TestPermutVar();
}

const cTplValGesInit< bool > & cParamApero::TestPermutVar()const 
{
   return SectionSolveur().TestPermutVar();
}


cTplValGesInit< bool > & cParamApero::ShowPermutVar()
{
   return SectionSolveur().ShowPermutVar();
}

const cTplValGesInit< bool > & cParamApero::ShowPermutVar()const 
{
   return SectionSolveur().ShowPermutVar();
}


cTplValGesInit< bool > & cParamApero::PermutIndex()
{
   return SectionSolveur().PermutIndex();
}

const cTplValGesInit< bool > & cParamApero::PermutIndex()const 
{
   return SectionSolveur().PermutIndex();
}


cTplValGesInit< bool > & cParamApero::NormaliseEqSc()
{
   return SectionSolveur().NormaliseEqSc();
}

const cTplValGesInit< bool > & cParamApero::NormaliseEqSc()const 
{
   return SectionSolveur().NormaliseEqSc();
}


cTplValGesInit< bool > & cParamApero::NormaliseEqTr()
{
   return SectionSolveur().NormaliseEqTr();
}

const cTplValGesInit< bool > & cParamApero::NormaliseEqTr()const 
{
   return SectionSolveur().NormaliseEqTr();
}


cTplValGesInit< double > & cParamApero::LimBsHProj()
{
   return SectionSolveur().LimBsHProj();
}

const cTplValGesInit< double > & cParamApero::LimBsHProj()const 
{
   return SectionSolveur().LimBsHProj();
}


cTplValGesInit< double > & cParamApero::LimBsHRefut()
{
   return SectionSolveur().LimBsHRefut();
}

const cTplValGesInit< double > & cParamApero::LimBsHRefut()const 
{
   return SectionSolveur().LimBsHRefut();
}


cTplValGesInit< double > & cParamApero::LimModeGL()
{
   return SectionSolveur().LimModeGL();
}

const cTplValGesInit< double > & cParamApero::LimModeGL()const 
{
   return SectionSolveur().LimModeGL();
}


cTplValGesInit< bool > & cParamApero::GridOptimKnownDist()
{
   return SectionSolveur().GridOptimKnownDist();
}

const cTplValGesInit< bool > & cParamApero::GridOptimKnownDist()const 
{
   return SectionSolveur().GridOptimKnownDist();
}


cTplValGesInit< cSectionLevenbergMarkard > & cParamApero::SLMGlob()
{
   return SectionSolveur().SLMGlob();
}

const cTplValGesInit< cSectionLevenbergMarkard > & cParamApero::SLMGlob()const 
{
   return SectionSolveur().SLMGlob();
}


cTplValGesInit< double > & cParamApero::MultSLMGlob()
{
   return SectionSolveur().MultSLMGlob();
}

const cTplValGesInit< double > & cParamApero::MultSLMGlob()const 
{
   return SectionSolveur().MultSLMGlob();
}


cTplValGesInit< cElRegex_Ptr > & cParamApero::Im2Aff()
{
   return SectionSolveur().Im2Aff();
}

const cTplValGesInit< cElRegex_Ptr > & cParamApero::Im2Aff()const 
{
   return SectionSolveur().Im2Aff();
}


cSectionSolveur & cParamApero::SectionSolveur()
{
   return mSectionSolveur;
}

const cSectionSolveur & cParamApero::SectionSolveur()const 
{
   return mSectionSolveur;
}


std::list< cEtapeCompensation > & cParamApero::EtapeCompensation()
{
   return SectionCompensation().EtapeCompensation();
}

const std::list< cEtapeCompensation > & cParamApero::EtapeCompensation()const 
{
   return SectionCompensation().EtapeCompensation();
}


cSectionCompensation & cParamApero::SectionCompensation()
{
   return mSectionCompensation;
}

const cSectionCompensation & cParamApero::SectionCompensation()const 
{
   return mSectionCompensation;
}

cElXMLTree * ToXMLTree(const cParamApero & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ParamApero",eXMLBranche);
   if (anObj.DicoLoc().IsInit())
      aRes->AddFils(ToXMLTree(anObj.DicoLoc().Val())->ReTagThis("DicoLoc"));
   if (anObj.ShowSection().IsInit())
      aRes->AddFils(ToXMLTree(anObj.ShowSection().Val())->ReTagThis("ShowSection"));
   if (anObj.CalledByItself().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("CalledByItself"),anObj.CalledByItself().Val())->ReTagThis("CalledByItself"));
   if (anObj.SectionMapApero().IsInit())
      aRes->AddFils(ToXMLTree(anObj.SectionMapApero().Val())->ReTagThis("SectionMapApero"));
   aRes->AddFils(ToXMLTree(anObj.SectionBDD_Observation())->ReTagThis("SectionBDD_Observation"));
   aRes->AddFils(ToXMLTree(anObj.SectionInconnues())->ReTagThis("SectionInconnues"));
   aRes->AddFils(ToXMLTree(anObj.SectionChantier())->ReTagThis("SectionChantier"));
   aRes->AddFils(ToXMLTree(anObj.SectionSolveur())->ReTagThis("SectionSolveur"));
   aRes->AddFils(ToXMLTree(anObj.SectionCompensation())->ReTagThis("SectionCompensation"));
  return aRes;
}

void xml_init(cParamApero & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.DicoLoc(),aTree->Get("DicoLoc",1)); //tototo 

   xml_init(anObj.ShowSection(),aTree->Get("ShowSection",1)); //tototo 

   xml_init(anObj.CalledByItself(),aTree->Get("CalledByItself",1),bool(false)); //tototo 

   xml_init(anObj.SectionMapApero(),aTree->Get("SectionMapApero",1)); //tototo 

   xml_init(anObj.SectionBDD_Observation(),aTree->Get("SectionBDD_Observation",1)); //tototo 

   xml_init(anObj.SectionInconnues(),aTree->Get("SectionInconnues",1)); //tototo 

   xml_init(anObj.SectionChantier(),aTree->Get("SectionChantier",1)); //tototo 

   xml_init(anObj.SectionSolveur(),aTree->Get("SectionSolveur",1)); //tototo 

   xml_init(anObj.SectionCompensation(),aTree->Get("SectionCompensation",1)); //tototo 
}

};
