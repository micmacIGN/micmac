#include "StdAfx.h"
// UNUSED #include "cParamApero.h"
// NO MORE ...
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

void  BinaryDumpInFile(ELISE_fp & aFp,const eTypeContrainteCalibCamera & anObj)
{
   BinaryDumpInFile(aFp,int(anObj));
}

void  BinaryUnDumpFromFile(eTypeContrainteCalibCamera & anObj,ELISE_fp & aFp)
{
   int aIVal;
   BinaryUnDumpFromFile(aIVal,aFp);
   anObj=(eTypeContrainteCalibCamera) aIVal;
}

std::string  Mangling( eTypeContrainteCalibCamera *) {return "2A9FBC8D7962D48EFD3F";};

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
   else if (aName=="eCalibAutomFour7x2")
      return eCalibAutomFour7x2;
   else if (aName=="eCalibAutomFour11x2")
      return eCalibAutomFour11x2;
   else if (aName=="eCalibAutomFour15x2")
      return eCalibAutomFour15x2;
   else if (aName=="eCalibAutomFour19x2")
      return eCalibAutomFour19x2;
   else if (aName=="eCalibAutomEbner")
      return eCalibAutomEbner;
   else if (aName=="eCalibAutomBrown")
      return eCalibAutomBrown;
   else if (aName=="eCalibAutomFishEyeStereographique")
      return eCalibAutomFishEyeStereographique;
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
   if (anObj==eCalibAutomFour7x2)
      return  "eCalibAutomFour7x2";
   if (anObj==eCalibAutomFour11x2)
      return  "eCalibAutomFour11x2";
   if (anObj==eCalibAutomFour15x2)
      return  "eCalibAutomFour15x2";
   if (anObj==eCalibAutomFour19x2)
      return  "eCalibAutomFour19x2";
   if (anObj==eCalibAutomEbner)
      return  "eCalibAutomEbner";
   if (anObj==eCalibAutomBrown)
      return  "eCalibAutomBrown";
   if (anObj==eCalibAutomFishEyeStereographique)
      return  "eCalibAutomFishEyeStereographique";
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

void  BinaryDumpInFile(ELISE_fp & aFp,const eTypeCalibAutom & anObj)
{
   BinaryDumpInFile(aFp,int(anObj));
}

void  BinaryUnDumpFromFile(eTypeCalibAutom & anObj,ELISE_fp & aFp)
{
   int aIVal;
   BinaryUnDumpFromFile(aIVal,aFp);
   anObj=(eTypeCalibAutom) aIVal;
}

std::string  Mangling( eTypeCalibAutom *) {return "93BCB8F66AB20F99FE3F";};

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
   else if (aName=="eAnglesFiges")
      return eAnglesFiges;
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
   if (anObj==eAnglesFiges)
      return  "eAnglesFiges";
 std::cout << "Enum = eTypeContraintePoseCamera\n";
   ELISE_ASSERT(false,"Bad Value in eToString for enum value ");
   return "";
}

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eTypeContraintePoseCamera & anObj)
{
      return  cElXMLTree::ValueNode(aNameTag,eToString(anObj));
}

void  BinaryDumpInFile(ELISE_fp & aFp,const eTypeContraintePoseCamera & anObj)
{
   BinaryDumpInFile(aFp,int(anObj));
}

void  BinaryUnDumpFromFile(eTypeContraintePoseCamera & anObj,ELISE_fp & aFp)
{
   int aIVal;
   BinaryUnDumpFromFile(aIVal,aFp);
   anObj=(eTypeContraintePoseCamera) aIVal;
}

std::string  Mangling( eTypeContraintePoseCamera *) {return "56693E40073CC98CFC3F";};

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

void  BinaryDumpInFile(ELISE_fp & aFp,const eTypeVerif & anObj)
{
   BinaryDumpInFile(aFp,int(anObj));
}

void  BinaryUnDumpFromFile(eTypeVerif & anObj,ELISE_fp & aFp)
{
   int aIVal;
   BinaryUnDumpFromFile(aIVal,aFp);
   anObj=(eTypeVerif) aIVal;
}

std::string  Mangling( eTypeVerif *) {return "A4761FEAD4F4298EFF3F";};

eTypeResulPtsBundle  Str2eTypeResulPtsBundle(const std::string & aName)
{
   if (aName=="eTRPB_Ok")
      return eTRPB_Ok;
   else if (aName=="eTRPB_InsufPoseInit")
      return eTRPB_InsufPoseInit;
   else if (aName=="eTRPB_PdsResNull")
      return eTRPB_PdsResNull;
   else if (aName=="eTRPB_NotInMasq3D")
      return eTRPB_NotInMasq3D;
   else if (aName=="eTRPB_BSurH")
      return eTRPB_BSurH;
   else if (aName=="eTRPB_Behind")
      return eTRPB_Behind;
   else if (aName=="eTRPB_VisibIm")
      return eTRPB_VisibIm;
   else if (aName=="eTRPB_OutIm")
      return eTRPB_OutIm;
   else if (aName=="eTRPB_PbInterBundle")
      return eTRPB_PbInterBundle;
   else if (aName=="eTRPB_RatioDistP2Cam")
      return eTRPB_RatioDistP2Cam;
   else if (aName=="eTRPB_Unknown")
      return eTRPB_Unknown;
   else if (aName=="eTRPB_NbVals")
      return eTRPB_NbVals;
  else
  {
      cout << aName << " is not a correct value for enum eTypeResulPtsBundle\n" ;
      ELISE_ASSERT(false,"XML enum value error");
  }
  return (eTypeResulPtsBundle) 0;
}
void xml_init(eTypeResulPtsBundle & aVal,cElXMLTree * aTree)
{
   aVal= Str2eTypeResulPtsBundle(aTree->Contenu());
}
std::string  eToString(const eTypeResulPtsBundle & anObj)
{
   if (anObj==eTRPB_Ok)
      return  "eTRPB_Ok";
   if (anObj==eTRPB_InsufPoseInit)
      return  "eTRPB_InsufPoseInit";
   if (anObj==eTRPB_PdsResNull)
      return  "eTRPB_PdsResNull";
   if (anObj==eTRPB_NotInMasq3D)
      return  "eTRPB_NotInMasq3D";
   if (anObj==eTRPB_BSurH)
      return  "eTRPB_BSurH";
   if (anObj==eTRPB_Behind)
      return  "eTRPB_Behind";
   if (anObj==eTRPB_VisibIm)
      return  "eTRPB_VisibIm";
   if (anObj==eTRPB_OutIm)
      return  "eTRPB_OutIm";
   if (anObj==eTRPB_PbInterBundle)
      return  "eTRPB_PbInterBundle";
   if (anObj==eTRPB_RatioDistP2Cam)
      return  "eTRPB_RatioDistP2Cam";
   if (anObj==eTRPB_Unknown)
      return  "eTRPB_Unknown";
   if (anObj==eTRPB_NbVals)
      return  "eTRPB_NbVals";
 std::cout << "Enum = eTypeResulPtsBundle\n";
   ELISE_ASSERT(false,"Bad Value in eToString for enum value ");
   return "";
}

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eTypeResulPtsBundle & anObj)
{
      return  cElXMLTree::ValueNode(aNameTag,eToString(anObj));
}

void  BinaryDumpInFile(ELISE_fp & aFp,const eTypeResulPtsBundle & anObj)
{
   BinaryDumpInFile(aFp,int(anObj));
}

void  BinaryUnDumpFromFile(eTypeResulPtsBundle & anObj,ELISE_fp & aFp)
{
   int aIVal;
   BinaryUnDumpFromFile(aIVal,aFp);
   anObj=(eTypeResulPtsBundle) aIVal;
}

std::string  Mangling( eTypeResulPtsBundle *) {return "054E6ACDF2582EDCFD3F";};

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

void  BinaryDumpInFile(ELISE_fp & aFp,const eTypePondMST_MEP & anObj)
{
   BinaryDumpInFile(aFp,int(anObj));
}

void  BinaryUnDumpFromFile(eTypePondMST_MEP & anObj,ELISE_fp & aFp)
{
   int aIVal;
   BinaryUnDumpFromFile(aIVal,aFp);
   anObj=(eTypePondMST_MEP) aIVal;
}

std::string  Mangling( eTypePondMST_MEP *) {return "44B19BE5E78640BCFE3F";};

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

void  BinaryDumpInFile(ELISE_fp & aFp,const eControleDescDic & anObj)
{
   BinaryDumpInFile(aFp,int(anObj));
}

void  BinaryUnDumpFromFile(eControleDescDic & anObj,ELISE_fp & aFp)
{
   int aIVal;
   BinaryUnDumpFromFile(aIVal,aFp);
   anObj=(eControleDescDic) aIVal;
}

std::string  Mangling( eControleDescDic *) {return "A59768EBFC71FA8FFEBF";};

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

void  BinaryDumpInFile(ELISE_fp & aFp,const eModePonderationRobuste & anObj)
{
   BinaryDumpInFile(aFp,int(anObj));
}

void  BinaryUnDumpFromFile(eModePonderationRobuste & anObj,ELISE_fp & aFp)
{
   int aIVal;
   BinaryUnDumpFromFile(aIVal,aFp);
   anObj=(eModePonderationRobuste) aIVal;
}

std::string  Mangling( eModePonderationRobuste *) {return "74627F43374791B4FE3F";};

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

void  BinaryDumpInFile(ELISE_fp & aFp,const eUniteMesureErreur & anObj)
{
   BinaryDumpInFile(aFp,int(anObj));
}

void  BinaryUnDumpFromFile(eUniteMesureErreur & anObj,ELISE_fp & aFp)
{
   int aIVal;
   BinaryUnDumpFromFile(aIVal,aFp);
   anObj=(eUniteMesureErreur) aIVal;
}

std::string  Mangling( eUniteMesureErreur *) {return "EDCF38BD57896797FE3F";};

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

void  BinaryDumpInFile(ELISE_fp & aFp,const eNiveauShowMessage & anObj)
{
   BinaryDumpInFile(aFp,int(anObj));
}

void  BinaryUnDumpFromFile(eNiveauShowMessage & anObj,ELISE_fp & aFp)
{
   int aIVal;
   BinaryUnDumpFromFile(aIVal,aFp);
   anObj=(eNiveauShowMessage) aIVal;
}

std::string  Mangling( eNiveauShowMessage *) {return "FE0F4716AD679488FF3F";};

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

void  BinaryDumpInFile(ELISE_fp & aFp,const eModePointLiaison & anObj)
{
   BinaryDumpInFile(aFp,int(anObj));
}

void  BinaryUnDumpFromFile(eModePointLiaison & anObj,ELISE_fp & aFp)
{
   int aIVal;
   BinaryUnDumpFromFile(aIVal,aFp);
   anObj=(eModePointLiaison) aIVal;
}

std::string  Mangling( eModePointLiaison *) {return "407480E1CA11AA8DF83F";};


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

void  BinaryUnDumpFromFile(cPowPointLiaisons & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Id(),aFp);
    BinaryUnDumpFromFile(anObj.NbTot(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Pds().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Pds().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Pds().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cPowPointLiaisons & anObj)
{
    BinaryDumpInFile(aFp,anObj.Id());
    BinaryDumpInFile(aFp,anObj.NbTot());
    BinaryDumpInFile(aFp,anObj.Pds().IsInit());
    if (anObj.Pds().IsInit()) BinaryDumpInFile(aFp,anObj.Pds().Val());
}

cElXMLTree * ToXMLTree(const cPowPointLiaisons & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"PowPointLiaisons",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Id"),anObj.Id())->ReTagThis("Id"));
   aRes->AddFils(::ToXMLTree(std::string("NbTot"),anObj.NbTot())->ReTagThis("NbTot"));
   if (anObj.Pds().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Pds"),anObj.Pds().Val())->ReTagThis("Pds"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cPowPointLiaisons & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.Id(),aTree->Get("Id",1)); //tototo 

   xml_init(anObj.NbTot(),aTree->Get("NbTot",1)); //tototo 

   xml_init(anObj.Pds(),aTree->Get("Pds",1),double(1.0)); //tototo 
}

std::string  Mangling( cPowPointLiaisons *) {return "C74CCAB555A1A096FF3F";};


std::list< cPowPointLiaisons > & cOptimizationPowel::PowPointLiaisons()
{
   return mPowPointLiaisons;
}

const std::list< cPowPointLiaisons > & cOptimizationPowel::PowPointLiaisons()const 
{
   return mPowPointLiaisons;
}

void  BinaryUnDumpFromFile(cOptimizationPowel & anObj,ELISE_fp & aFp)
{
   { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cPowPointLiaisons aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.PowPointLiaisons().push_back(aVal);
        }
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cOptimizationPowel & anObj)
{
    BinaryDumpInFile(aFp,(int)anObj.PowPointLiaisons().size());
    for(  std::list< cPowPointLiaisons >::const_iterator iT=anObj.PowPointLiaisons().begin();
         iT!=anObj.PowPointLiaisons().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
}

cElXMLTree * ToXMLTree(const cOptimizationPowel & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"OptimizationPowel",eXMLBranche);
  for
  (       std::list< cPowPointLiaisons >::const_iterator it=anObj.PowPointLiaisons().begin();
      it !=anObj.PowPointLiaisons().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("PowPointLiaisons"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cOptimizationPowel & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.PowPointLiaisons(),aTree->GetAll("PowPointLiaisons",false,1));
}

std::string  Mangling( cOptimizationPowel *) {return "58F9D6C6ECA230A0FB3F";};


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

void  BinaryUnDumpFromFile(cShowPbLiaison & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NbMinPtsMul().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NbMinPtsMul().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NbMinPtsMul().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Actif().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Actif().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Actif().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.GetCharOnPb().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.GetCharOnPb().ValForcedForUnUmp(),aFp);
        }
        else  anObj.GetCharOnPb().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cShowPbLiaison & anObj)
{
    BinaryDumpInFile(aFp,anObj.NbMinPtsMul().IsInit());
    if (anObj.NbMinPtsMul().IsInit()) BinaryDumpInFile(aFp,anObj.NbMinPtsMul().Val());
    BinaryDumpInFile(aFp,anObj.Actif().IsInit());
    if (anObj.Actif().IsInit()) BinaryDumpInFile(aFp,anObj.Actif().Val());
    BinaryDumpInFile(aFp,anObj.GetCharOnPb().IsInit());
    if (anObj.GetCharOnPb().IsInit()) BinaryDumpInFile(aFp,anObj.GetCharOnPb().Val());
}

cElXMLTree * ToXMLTree(const cShowPbLiaison & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ShowPbLiaison",eXMLBranche);
   if (anObj.NbMinPtsMul().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NbMinPtsMul"),anObj.NbMinPtsMul().Val())->ReTagThis("NbMinPtsMul"));
   if (anObj.Actif().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Actif"),anObj.Actif().Val())->ReTagThis("Actif"));
   if (anObj.GetCharOnPb().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("GetCharOnPb"),anObj.GetCharOnPb().Val())->ReTagThis("GetCharOnPb"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cShowPbLiaison & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.NbMinPtsMul(),aTree->Get("NbMinPtsMul",1)); //tototo 

   xml_init(anObj.Actif(),aTree->Get("Actif",1),bool(true)); //tototo 

   xml_init(anObj.GetCharOnPb(),aTree->Get("GetCharOnPb",1),bool(true)); //tototo 
}

std::string  Mangling( cShowPbLiaison *) {return "0C785C0A5325B6AAFD3F";};


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

void  BinaryUnDumpFromFile(cPonderationPackMesure & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.EcartMesureIndiv(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Add2Compens().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Add2Compens().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Add2Compens().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ModePonderation().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ModePonderation().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ModePonderation().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.EcartMax().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.EcartMax().ValForcedForUnUmp(),aFp);
        }
        else  anObj.EcartMax().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ExposantLK().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ExposantLK().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ExposantLK().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SigmaPond().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SigmaPond().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SigmaPond().SetNoInit();
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
             anObj.Show().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Show().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Show().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.GetChar().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.GetChar().ValForcedForUnUmp(),aFp);
        }
        else  anObj.GetChar().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NbMinMultShowIndiv().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NbMinMultShowIndiv().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NbMinMultShowIndiv().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ShowPercentile().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ShowPercentile().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ShowPercentile().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ExposantPoidsMult().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ExposantPoidsMult().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ExposantPoidsMult().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.IdFilter3D().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.IdFilter3D().ValForcedForUnUmp(),aFp);
        }
        else  anObj.IdFilter3D().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cPonderationPackMesure & anObj)
{
    BinaryDumpInFile(aFp,anObj.EcartMesureIndiv());
    BinaryDumpInFile(aFp,anObj.Add2Compens().IsInit());
    if (anObj.Add2Compens().IsInit()) BinaryDumpInFile(aFp,anObj.Add2Compens().Val());
    BinaryDumpInFile(aFp,anObj.ModePonderation().IsInit());
    if (anObj.ModePonderation().IsInit()) BinaryDumpInFile(aFp,anObj.ModePonderation().Val());
    BinaryDumpInFile(aFp,anObj.EcartMax().IsInit());
    if (anObj.EcartMax().IsInit()) BinaryDumpInFile(aFp,anObj.EcartMax().Val());
    BinaryDumpInFile(aFp,anObj.ExposantLK().IsInit());
    if (anObj.ExposantLK().IsInit()) BinaryDumpInFile(aFp,anObj.ExposantLK().Val());
    BinaryDumpInFile(aFp,anObj.SigmaPond().IsInit());
    if (anObj.SigmaPond().IsInit()) BinaryDumpInFile(aFp,anObj.SigmaPond().Val());
    BinaryDumpInFile(aFp,anObj.NbMax().IsInit());
    if (anObj.NbMax().IsInit()) BinaryDumpInFile(aFp,anObj.NbMax().Val());
    BinaryDumpInFile(aFp,anObj.Show().IsInit());
    if (anObj.Show().IsInit()) BinaryDumpInFile(aFp,anObj.Show().Val());
    BinaryDumpInFile(aFp,anObj.GetChar().IsInit());
    if (anObj.GetChar().IsInit()) BinaryDumpInFile(aFp,anObj.GetChar().Val());
    BinaryDumpInFile(aFp,anObj.NbMinMultShowIndiv().IsInit());
    if (anObj.NbMinMultShowIndiv().IsInit()) BinaryDumpInFile(aFp,anObj.NbMinMultShowIndiv().Val());
    BinaryDumpInFile(aFp,anObj.ShowPercentile().IsInit());
    if (anObj.ShowPercentile().IsInit()) BinaryDumpInFile(aFp,anObj.ShowPercentile().Val());
    BinaryDumpInFile(aFp,anObj.ExposantPoidsMult().IsInit());
    if (anObj.ExposantPoidsMult().IsInit()) BinaryDumpInFile(aFp,anObj.ExposantPoidsMult().Val());
    BinaryDumpInFile(aFp,anObj.IdFilter3D().IsInit());
    if (anObj.IdFilter3D().IsInit()) BinaryDumpInFile(aFp,anObj.IdFilter3D().Val());
}

cElXMLTree * ToXMLTree(const cPonderationPackMesure & anObj)
{
  XMLPushContext(anObj.mGXml);
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
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cPonderationPackMesure & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

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

std::string  Mangling( cPonderationPackMesure *) {return "50E3CEDF6B8DAD9AFF3F";};


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


cTplValGesInit< bool > & cParamEstimPlan::AcceptDefPlanIfNoPoint()
{
   return mAcceptDefPlanIfNoPoint;
}

const cTplValGesInit< bool > & cParamEstimPlan::AcceptDefPlanIfNoPoint()const 
{
   return mAcceptDefPlanIfNoPoint;
}

void  BinaryUnDumpFromFile(cParamEstimPlan & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.AttrSup().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.AttrSup().ValForcedForUnUmp(),aFp);
        }
        else  anObj.AttrSup().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.KeyCalculMasq().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.KeyCalculMasq().ValForcedForUnUmp(),aFp);
        }
        else  anObj.KeyCalculMasq().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.IdBdl(),aFp);
    BinaryUnDumpFromFile(anObj.Pond(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.LimBSurH().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.LimBSurH().ValForcedForUnUmp(),aFp);
        }
        else  anObj.LimBSurH().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.AcceptDefPlanIfNoPoint().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.AcceptDefPlanIfNoPoint().ValForcedForUnUmp(),aFp);
        }
        else  anObj.AcceptDefPlanIfNoPoint().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cParamEstimPlan & anObj)
{
    BinaryDumpInFile(aFp,anObj.AttrSup().IsInit());
    if (anObj.AttrSup().IsInit()) BinaryDumpInFile(aFp,anObj.AttrSup().Val());
    BinaryDumpInFile(aFp,anObj.KeyCalculMasq().IsInit());
    if (anObj.KeyCalculMasq().IsInit()) BinaryDumpInFile(aFp,anObj.KeyCalculMasq().Val());
    BinaryDumpInFile(aFp,anObj.IdBdl());
    BinaryDumpInFile(aFp,anObj.Pond());
    BinaryDumpInFile(aFp,anObj.LimBSurH().IsInit());
    if (anObj.LimBSurH().IsInit()) BinaryDumpInFile(aFp,anObj.LimBSurH().Val());
    BinaryDumpInFile(aFp,anObj.AcceptDefPlanIfNoPoint().IsInit());
    if (anObj.AcceptDefPlanIfNoPoint().IsInit()) BinaryDumpInFile(aFp,anObj.AcceptDefPlanIfNoPoint().Val());
}

cElXMLTree * ToXMLTree(const cParamEstimPlan & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ParamEstimPlan",eXMLBranche);
   if (anObj.AttrSup().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("AttrSup"),anObj.AttrSup().Val())->ReTagThis("AttrSup"));
   if (anObj.KeyCalculMasq().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("KeyCalculMasq"),anObj.KeyCalculMasq().Val())->ReTagThis("KeyCalculMasq"));
   aRes->AddFils(::ToXMLTree(std::string("IdBdl"),anObj.IdBdl())->ReTagThis("IdBdl"));
   aRes->AddFils(ToXMLTree(anObj.Pond())->ReTagThis("Pond"));
   if (anObj.LimBSurH().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("LimBSurH"),anObj.LimBSurH().Val())->ReTagThis("LimBSurH"));
   if (anObj.AcceptDefPlanIfNoPoint().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("AcceptDefPlanIfNoPoint"),anObj.AcceptDefPlanIfNoPoint().Val())->ReTagThis("AcceptDefPlanIfNoPoint"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cParamEstimPlan & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.AttrSup(),aTree->Get("AttrSup",1)); //tototo 

   xml_init(anObj.KeyCalculMasq(),aTree->Get("KeyCalculMasq",1)); //tototo 

   xml_init(anObj.IdBdl(),aTree->Get("IdBdl",1)); //tototo 

   xml_init(anObj.Pond(),aTree->Get("Pond",1)); //tototo 

   xml_init(anObj.LimBSurH(),aTree->Get("LimBSurH",1),double(1e-2)); //tototo 

   xml_init(anObj.AcceptDefPlanIfNoPoint(),aTree->Get("AcceptDefPlanIfNoPoint",1),bool(false)); //tototo 
}

std::string  Mangling( cParamEstimPlan *) {return "A0BF187C3947119CFB3F";};


double & cRigidBlockWeighting::PondOnTr()
{
   return mPondOnTr;
}

const double & cRigidBlockWeighting::PondOnTr()const 
{
   return mPondOnTr;
}


double & cRigidBlockWeighting::PondOnRot()
{
   return mPondOnRot;
}

const double & cRigidBlockWeighting::PondOnRot()const 
{
   return mPondOnRot;
}


cTplValGesInit< double > & cRigidBlockWeighting::PondOnTrFinal()
{
   return mPondOnTrFinal;
}

const cTplValGesInit< double > & cRigidBlockWeighting::PondOnTrFinal()const 
{
   return mPondOnTrFinal;
}


cTplValGesInit< double > & cRigidBlockWeighting::PondOnRotFinal()
{
   return mPondOnRotFinal;
}

const cTplValGesInit< double > & cRigidBlockWeighting::PondOnRotFinal()const 
{
   return mPondOnRotFinal;
}

void  BinaryUnDumpFromFile(cRigidBlockWeighting & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.PondOnTr(),aFp);
    BinaryUnDumpFromFile(anObj.PondOnRot(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.PondOnTrFinal().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.PondOnTrFinal().ValForcedForUnUmp(),aFp);
        }
        else  anObj.PondOnTrFinal().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.PondOnRotFinal().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.PondOnRotFinal().ValForcedForUnUmp(),aFp);
        }
        else  anObj.PondOnRotFinal().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cRigidBlockWeighting & anObj)
{
    BinaryDumpInFile(aFp,anObj.PondOnTr());
    BinaryDumpInFile(aFp,anObj.PondOnRot());
    BinaryDumpInFile(aFp,anObj.PondOnTrFinal().IsInit());
    if (anObj.PondOnTrFinal().IsInit()) BinaryDumpInFile(aFp,anObj.PondOnTrFinal().Val());
    BinaryDumpInFile(aFp,anObj.PondOnRotFinal().IsInit());
    if (anObj.PondOnRotFinal().IsInit()) BinaryDumpInFile(aFp,anObj.PondOnRotFinal().Val());
}

cElXMLTree * ToXMLTree(const cRigidBlockWeighting & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"RigidBlockWeighting",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("PondOnTr"),anObj.PondOnTr())->ReTagThis("PondOnTr"));
   aRes->AddFils(::ToXMLTree(std::string("PondOnRot"),anObj.PondOnRot())->ReTagThis("PondOnRot"));
   if (anObj.PondOnTrFinal().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PondOnTrFinal"),anObj.PondOnTrFinal().Val())->ReTagThis("PondOnTrFinal"));
   if (anObj.PondOnRotFinal().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PondOnRotFinal"),anObj.PondOnRotFinal().Val())->ReTagThis("PondOnRotFinal"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cRigidBlockWeighting & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.PondOnTr(),aTree->Get("PondOnTr",1)); //tototo 

   xml_init(anObj.PondOnRot(),aTree->Get("PondOnRot",1)); //tototo 

   xml_init(anObj.PondOnTrFinal(),aTree->Get("PondOnTrFinal",1)); //tototo 

   xml_init(anObj.PondOnRotFinal(),aTree->Get("PondOnRotFinal",1)); //tototo 
}

std::string  Mangling( cRigidBlockWeighting *) {return "8C60C020C7E3E3E5FD3F";};


double & cGpsRelativeWeighting::SigmaPerSec()
{
   return mSigmaPerSec;
}

const double & cGpsRelativeWeighting::SigmaPerSec()const 
{
   return mSigmaPerSec;
}


double & cGpsRelativeWeighting::SigmaMin()
{
   return mSigmaMin;
}

const double & cGpsRelativeWeighting::SigmaMin()const 
{
   return mSigmaMin;
}


cTplValGesInit< double > & cGpsRelativeWeighting::MaxResidu()
{
   return mMaxResidu;
}

const cTplValGesInit< double > & cGpsRelativeWeighting::MaxResidu()const 
{
   return mMaxResidu;
}

void  BinaryUnDumpFromFile(cGpsRelativeWeighting & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.SigmaPerSec(),aFp);
    BinaryUnDumpFromFile(anObj.SigmaMin(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.MaxResidu().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.MaxResidu().ValForcedForUnUmp(),aFp);
        }
        else  anObj.MaxResidu().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cGpsRelativeWeighting & anObj)
{
    BinaryDumpInFile(aFp,anObj.SigmaPerSec());
    BinaryDumpInFile(aFp,anObj.SigmaMin());
    BinaryDumpInFile(aFp,anObj.MaxResidu().IsInit());
    if (anObj.MaxResidu().IsInit()) BinaryDumpInFile(aFp,anObj.MaxResidu().Val());
}

cElXMLTree * ToXMLTree(const cGpsRelativeWeighting & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"GpsRelativeWeighting",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("SigmaPerSec"),anObj.SigmaPerSec())->ReTagThis("SigmaPerSec"));
   aRes->AddFils(::ToXMLTree(std::string("SigmaMin"),anObj.SigmaMin())->ReTagThis("SigmaMin"));
   if (anObj.MaxResidu().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("MaxResidu"),anObj.MaxResidu().Val())->ReTagThis("MaxResidu"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cGpsRelativeWeighting & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.SigmaPerSec(),aTree->Get("SigmaPerSec",1)); //tototo 

   xml_init(anObj.SigmaMin(),aTree->Get("SigmaMin",1)); //tototo 

   xml_init(anObj.MaxResidu(),aTree->Get("MaxResidu",1)); //tototo 
}

std::string  Mangling( cGpsRelativeWeighting *) {return "26A929420A8405AAFF3F";};


double & cXml_OneObsPlane::Sigma()
{
   return mSigma;
}

const double & cXml_OneObsPlane::Sigma()const 
{
   return mSigma;
}


double & cXml_OneObsPlane::Cste()
{
   return mCste;
}

const double & cXml_OneObsPlane::Cste()const 
{
   return mCste;
}


Pt3dr & cXml_OneObsPlane::Vect()
{
   return mVect;
}

const Pt3dr & cXml_OneObsPlane::Vect()const 
{
   return mVect;
}

void  BinaryUnDumpFromFile(cXml_OneObsPlane & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Sigma(),aFp);
    BinaryUnDumpFromFile(anObj.Cste(),aFp);
    BinaryUnDumpFromFile(anObj.Vect(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cXml_OneObsPlane & anObj)
{
    BinaryDumpInFile(aFp,anObj.Sigma());
    BinaryDumpInFile(aFp,anObj.Cste());
    BinaryDumpInFile(aFp,anObj.Vect());
}

cElXMLTree * ToXMLTree(const cXml_OneObsPlane & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"Xml_OneObsPlane",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Sigma"),anObj.Sigma())->ReTagThis("Sigma"));
   aRes->AddFils(::ToXMLTree(std::string("Cste"),anObj.Cste())->ReTagThis("Cste"));
   aRes->AddFils(::ToXMLTree(std::string("Vect"),anObj.Vect())->ReTagThis("Vect"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cXml_OneObsPlane & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.Sigma(),aTree->Get("Sigma",1)); //tototo 

   xml_init(anObj.Cste(),aTree->Get("Cste",1)); //tototo 

   xml_init(anObj.Vect(),aTree->Get("Vect",1)); //tototo 
}

std::string  Mangling( cXml_OneObsPlane *) {return "ACFEC39587E769E0FDBF";};


std::string & cXml_ObsPlaneOnPose::NameIm()
{
   return mNameIm;
}

const std::string & cXml_ObsPlaneOnPose::NameIm()const 
{
   return mNameIm;
}


std::list< cXml_OneObsPlane > & cXml_ObsPlaneOnPose::Obs1Plane()
{
   return mObs1Plane;
}

const std::list< cXml_OneObsPlane > & cXml_ObsPlaneOnPose::Obs1Plane()const 
{
   return mObs1Plane;
}

void  BinaryUnDumpFromFile(cXml_ObsPlaneOnPose & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.NameIm(),aFp);
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cXml_OneObsPlane aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.Obs1Plane().push_back(aVal);
        }
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cXml_ObsPlaneOnPose & anObj)
{
    BinaryDumpInFile(aFp,anObj.NameIm());
    BinaryDumpInFile(aFp,(int)anObj.Obs1Plane().size());
    for(  std::list< cXml_OneObsPlane >::const_iterator iT=anObj.Obs1Plane().begin();
         iT!=anObj.Obs1Plane().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
}

cElXMLTree * ToXMLTree(const cXml_ObsPlaneOnPose & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"Xml_ObsPlaneOnPose",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("NameIm"),anObj.NameIm())->ReTagThis("NameIm"));
  for
  (       std::list< cXml_OneObsPlane >::const_iterator it=anObj.Obs1Plane().begin();
      it !=anObj.Obs1Plane().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("Obs1Plane"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cXml_ObsPlaneOnPose & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.NameIm(),aTree->Get("NameIm",1)); //tototo 

   xml_init(anObj.Obs1Plane(),aTree->GetAll("Obs1Plane",false,1));
}

std::string  Mangling( cXml_ObsPlaneOnPose *) {return "1CA76FB81F4F68A2FF3F";};


std::map< std::string,cXml_ObsPlaneOnPose > & cXml_FileObsPlane::Obs1Im()
{
   return mObs1Im;
}

const std::map< std::string,cXml_ObsPlaneOnPose > & cXml_FileObsPlane::Obs1Im()const 
{
   return mObs1Im;
}

void  BinaryUnDumpFromFile(cXml_FileObsPlane & anObj,ELISE_fp & aFp)
{
     ELISE_ASSERT(false,"No Support for this conainer in bin dump");
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cXml_FileObsPlane & anObj)
{
    ELISE_ASSERT(false,"No Support for this conainer in bin dump");
}

cElXMLTree * ToXMLTree(const cXml_FileObsPlane & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"Xml_FileObsPlane",eXMLBranche);
  for
  (       std::map< std::string,cXml_ObsPlaneOnPose >::const_iterator it=anObj.Obs1Im().begin();
      it !=anObj.Obs1Im().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it).second)->ReTagThis("Obs1Im"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cXml_FileObsPlane & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.Obs1Im(),aTree->GetAll("Obs1Im",false,1),"NameIm");
}

std::string  Mangling( cXml_FileObsPlane *) {return "E6B5A3918636E7CCFD3F";};


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

void  BinaryUnDumpFromFile(cAperoPointeStereo & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.P1(),aFp);
    BinaryUnDumpFromFile(anObj.Im1(),aFp);
    BinaryUnDumpFromFile(anObj.P2(),aFp);
    BinaryUnDumpFromFile(anObj.Im2(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cAperoPointeStereo & anObj)
{
    BinaryDumpInFile(aFp,anObj.P1());
    BinaryDumpInFile(aFp,anObj.Im1());
    BinaryDumpInFile(aFp,anObj.P2());
    BinaryDumpInFile(aFp,anObj.Im2());
}

cElXMLTree * ToXMLTree(const cAperoPointeStereo & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"AperoPointeStereo",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("P1"),anObj.P1())->ReTagThis("P1"));
   aRes->AddFils(::ToXMLTree(std::string("Im1"),anObj.Im1())->ReTagThis("Im1"));
   aRes->AddFils(::ToXMLTree(std::string("P2"),anObj.P2())->ReTagThis("P2"));
   aRes->AddFils(::ToXMLTree(std::string("Im2"),anObj.Im2())->ReTagThis("Im2"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cAperoPointeStereo & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.P1(),aTree->Get("P1",1)); //tototo 

   xml_init(anObj.Im1(),aTree->Get("Im1",1)); //tototo 

   xml_init(anObj.P2(),aTree->Get("P2",1)); //tototo 

   xml_init(anObj.Im2(),aTree->Get("Im2",1)); //tototo 
}

std::string  Mangling( cAperoPointeStereo *) {return "C0CC0592E993FBBCFD3F";};


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

void  BinaryUnDumpFromFile(cAperoPointeMono & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Pt(),aFp);
    BinaryUnDumpFromFile(anObj.Im(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cAperoPointeMono & anObj)
{
    BinaryDumpInFile(aFp,anObj.Pt());
    BinaryDumpInFile(aFp,anObj.Im());
}

cElXMLTree * ToXMLTree(const cAperoPointeMono & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"AperoPointeMono",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Pt"),anObj.Pt())->ReTagThis("Pt"));
   aRes->AddFils(::ToXMLTree(std::string("Im"),anObj.Im())->ReTagThis("Im"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cAperoPointeMono & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.Pt(),aTree->Get("Pt",1)); //tototo 

   xml_init(anObj.Im(),aTree->Get("Im",1)); //tototo 
}

std::string  Mangling( cAperoPointeMono *) {return "272F4F6B4C864AD1FE3F";};


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

void  BinaryUnDumpFromFile(cApero2PointeFromFile & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.File(),aFp);
    BinaryUnDumpFromFile(anObj.NameP1(),aFp);
    BinaryUnDumpFromFile(anObj.NameP2(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cApero2PointeFromFile & anObj)
{
    BinaryDumpInFile(aFp,anObj.File());
    BinaryDumpInFile(aFp,anObj.NameP1());
    BinaryDumpInFile(aFp,anObj.NameP2());
}

cElXMLTree * ToXMLTree(const cApero2PointeFromFile & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"Apero2PointeFromFile",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("File"),anObj.File())->ReTagThis("File"));
   aRes->AddFils(::ToXMLTree(std::string("NameP1"),anObj.NameP1())->ReTagThis("NameP1"));
   aRes->AddFils(::ToXMLTree(std::string("NameP2"),anObj.NameP2())->ReTagThis("NameP2"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cApero2PointeFromFile & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.File(),aTree->Get("File",1)); //tototo 

   xml_init(anObj.NameP1(),aTree->Get("NameP1",1)); //tototo 

   xml_init(anObj.NameP2(),aTree->Get("NameP2",1)); //tototo 
}

std::string  Mangling( cApero2PointeFromFile *) {return "30B4EDE1A63D09AEFB3F";};


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

void  BinaryUnDumpFromFile(cParamForceRappel & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.PatternNameApply(),aFp);
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             double aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.Incertitude().push_back(aVal);
        }
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.OnCur().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.OnCur().ValForcedForUnUmp(),aFp);
        }
        else  anObj.OnCur().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cParamForceRappel & anObj)
{
    BinaryDumpInFile(aFp,anObj.PatternNameApply());
    BinaryDumpInFile(aFp,(int)anObj.Incertitude().size());
    for(  std::vector< double >::const_iterator iT=anObj.Incertitude().begin();
         iT!=anObj.Incertitude().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.OnCur().IsInit());
    if (anObj.OnCur().IsInit()) BinaryDumpInFile(aFp,anObj.OnCur().Val());
}

cElXMLTree * ToXMLTree(const cParamForceRappel & anObj)
{
  XMLPushContext(anObj.mGXml);
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
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cParamForceRappel & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.PatternNameApply(),aTree->Get("PatternNameApply",1)); //tototo 

   xml_init(anObj.Incertitude(),aTree->GetAll("Incertitude",false,1));

   xml_init(anObj.OnCur(),aTree->Get("OnCur",1)); //tototo 
}

std::string  Mangling( cParamForceRappel *) {return "DB8C6BF151807696FF3F";};


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

void  BinaryUnDumpFromFile(cRappelOnAngles & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.ParamF(),aFp);
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             int aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.TetaApply().push_back(aVal);
        }
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cRappelOnAngles & anObj)
{
    BinaryDumpInFile(aFp,anObj.ParamF());
    BinaryDumpInFile(aFp,(int)anObj.TetaApply().size());
    for(  std::vector< int >::const_iterator iT=anObj.TetaApply().begin();
         iT!=anObj.TetaApply().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
}

cElXMLTree * ToXMLTree(const cRappelOnAngles & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"RappelOnAngles",eXMLBranche);
   aRes->AddFils(ToXMLTree(anObj.ParamF())->ReTagThis("ParamF"));
  for
  (       std::vector< int >::const_iterator it=anObj.TetaApply().begin();
      it !=anObj.TetaApply().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("TetaApply"),(*it))->ReTagThis("TetaApply"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cRappelOnAngles & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.ParamF(),aTree->Get("ParamF",1)); //tototo 

   xml_init(anObj.TetaApply(),aTree->GetAll("TetaApply",false,1));
}

std::string  Mangling( cRappelOnAngles *) {return "5EDA9B3BB66E1AB5FE3F";};


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

void  BinaryUnDumpFromFile(cRappelOnCentres & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.ParamF(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.OnlyWhenNoCentreInit().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.OnlyWhenNoCentreInit().ValForcedForUnUmp(),aFp);
        }
        else  anObj.OnlyWhenNoCentreInit().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cRappelOnCentres & anObj)
{
    BinaryDumpInFile(aFp,anObj.ParamF());
    BinaryDumpInFile(aFp,anObj.OnlyWhenNoCentreInit().IsInit());
    if (anObj.OnlyWhenNoCentreInit().IsInit()) BinaryDumpInFile(aFp,anObj.OnlyWhenNoCentreInit().Val());
}

cElXMLTree * ToXMLTree(const cRappelOnCentres & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"RappelOnCentres",eXMLBranche);
   aRes->AddFils(ToXMLTree(anObj.ParamF())->ReTagThis("ParamF"));
   if (anObj.OnlyWhenNoCentreInit().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("OnlyWhenNoCentreInit"),anObj.OnlyWhenNoCentreInit().Val())->ReTagThis("OnlyWhenNoCentreInit"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cRappelOnCentres & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.ParamF(),aTree->Get("ParamF",1)); //tototo 

   xml_init(anObj.OnlyWhenNoCentreInit(),aTree->Get("OnlyWhenNoCentreInit",1),bool(true)); //tototo 
}

std::string  Mangling( cRappelOnCentres *) {return "A2CA6920A1657380FE3F";};


cParamForceRappel & cRappelOnIntrinseque::ParamF()
{
   return mParamF;
}

const cParamForceRappel & cRappelOnIntrinseque::ParamF()const 
{
   return mParamF;
}

void  BinaryUnDumpFromFile(cRappelOnIntrinseque & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.ParamF(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cRappelOnIntrinseque & anObj)
{
    BinaryDumpInFile(aFp,anObj.ParamF());
}

cElXMLTree * ToXMLTree(const cRappelOnIntrinseque & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"RappelOnIntrinseque",eXMLBranche);
   aRes->AddFils(ToXMLTree(anObj.ParamF())->ReTagThis("ParamF"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cRappelOnIntrinseque & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.ParamF(),aTree->Get("ParamF",1)); //tototo 
}

std::string  Mangling( cRappelOnIntrinseque *) {return "754E5035B214CC95FE3F";};


double & cXmlSLM_RappelOnPt::CondMax()
{
   return mCondMax;
}

const double & cXmlSLM_RappelOnPt::CondMax()const 
{
   return mCondMax;
}

void  BinaryUnDumpFromFile(cXmlSLM_RappelOnPt & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.CondMax(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cXmlSLM_RappelOnPt & anObj)
{
    BinaryDumpInFile(aFp,anObj.CondMax());
}

cElXMLTree * ToXMLTree(const cXmlSLM_RappelOnPt & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"XmlSLM_RappelOnPt",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("CondMax"),anObj.CondMax())->ReTagThis("CondMax"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cXmlSLM_RappelOnPt & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.CondMax(),aTree->Get("CondMax",1)); //tototo 
}

std::string  Mangling( cXmlSLM_RappelOnPt *) {return "0C90024C4E4431F8FB3F";};


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


std::list< cRappelOnIntrinseque > & cSectionLevenbergMarkard::RappelOnIntrinseque()
{
   return mRappelOnIntrinseque;
}

const std::list< cRappelOnIntrinseque > & cSectionLevenbergMarkard::RappelOnIntrinseque()const 
{
   return mRappelOnIntrinseque;
}


double & cSectionLevenbergMarkard::CondMax()
{
   return XmlSLM_RappelOnPt().Val().CondMax();
}

const double & cSectionLevenbergMarkard::CondMax()const 
{
   return XmlSLM_RappelOnPt().Val().CondMax();
}


cTplValGesInit< cXmlSLM_RappelOnPt > & cSectionLevenbergMarkard::XmlSLM_RappelOnPt()
{
   return mXmlSLM_RappelOnPt;
}

const cTplValGesInit< cXmlSLM_RappelOnPt > & cSectionLevenbergMarkard::XmlSLM_RappelOnPt()const 
{
   return mXmlSLM_RappelOnPt;
}

void  BinaryUnDumpFromFile(cSectionLevenbergMarkard & anObj,ELISE_fp & aFp)
{
   { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cRappelOnAngles aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.RappelOnAngles().push_back(aVal);
        }
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cRappelOnCentres aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.RappelOnCentres().push_back(aVal);
        }
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cRappelOnIntrinseque aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.RappelOnIntrinseque().push_back(aVal);
        }
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.XmlSLM_RappelOnPt().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.XmlSLM_RappelOnPt().ValForcedForUnUmp(),aFp);
        }
        else  anObj.XmlSLM_RappelOnPt().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cSectionLevenbergMarkard & anObj)
{
    BinaryDumpInFile(aFp,(int)anObj.RappelOnAngles().size());
    for(  std::list< cRappelOnAngles >::const_iterator iT=anObj.RappelOnAngles().begin();
         iT!=anObj.RappelOnAngles().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,(int)anObj.RappelOnCentres().size());
    for(  std::list< cRappelOnCentres >::const_iterator iT=anObj.RappelOnCentres().begin();
         iT!=anObj.RappelOnCentres().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,(int)anObj.RappelOnIntrinseque().size());
    for(  std::list< cRappelOnIntrinseque >::const_iterator iT=anObj.RappelOnIntrinseque().begin();
         iT!=anObj.RappelOnIntrinseque().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.XmlSLM_RappelOnPt().IsInit());
    if (anObj.XmlSLM_RappelOnPt().IsInit()) BinaryDumpInFile(aFp,anObj.XmlSLM_RappelOnPt().Val());
}

cElXMLTree * ToXMLTree(const cSectionLevenbergMarkard & anObj)
{
  XMLPushContext(anObj.mGXml);
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
  for
  (       std::list< cRappelOnIntrinseque >::const_iterator it=anObj.RappelOnIntrinseque().begin();
      it !=anObj.RappelOnIntrinseque().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("RappelOnIntrinseque"));
   if (anObj.XmlSLM_RappelOnPt().IsInit())
      aRes->AddFils(ToXMLTree(anObj.XmlSLM_RappelOnPt().Val())->ReTagThis("XmlSLM_RappelOnPt"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cSectionLevenbergMarkard & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.RappelOnAngles(),aTree->GetAll("RappelOnAngles",false,1));

   xml_init(anObj.RappelOnCentres(),aTree->GetAll("RappelOnCentres",false,1));

   xml_init(anObj.RappelOnIntrinseque(),aTree->GetAll("RappelOnIntrinseque",false,1));

   xml_init(anObj.XmlSLM_RappelOnPt(),aTree->Get("XmlSLM_RappelOnPt",1)); //tototo 
}

std::string  Mangling( cSectionLevenbergMarkard *) {return "48BA5FEF1A4B43A3FE3F";};


double & cXml_SigmaRot::Ang()
{
   return mAng;
}

const double & cXml_SigmaRot::Ang()const 
{
   return mAng;
}


double & cXml_SigmaRot::Center()
{
   return mCenter;
}

const double & cXml_SigmaRot::Center()const 
{
   return mCenter;
}

void  BinaryUnDumpFromFile(cXml_SigmaRot & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Ang(),aFp);
    BinaryUnDumpFromFile(anObj.Center(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cXml_SigmaRot & anObj)
{
    BinaryDumpInFile(aFp,anObj.Ang());
    BinaryDumpInFile(aFp,anObj.Center());
}

cElXMLTree * ToXMLTree(const cXml_SigmaRot & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"Xml_SigmaRot",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Ang"),anObj.Ang())->ReTagThis("Ang"));
   aRes->AddFils(::ToXMLTree(std::string("Center"),anObj.Center())->ReTagThis("Center"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cXml_SigmaRot & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.Ang(),aTree->Get("Ang",1)); //tototo 

   xml_init(anObj.Center(),aTree->Get("Center",1)); //tototo 
}

std::string  Mangling( cXml_SigmaRot *) {return "11F0B3FB5B516CA0FE3F";};


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

void  BinaryUnDumpFromFile(cSetOrientationInterne & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.KeyFile(),aFp);
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
             anObj.Tag().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Tag().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Tag().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.AddToCur(),aFp);
    BinaryUnDumpFromFile(anObj.M2C(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cSetOrientationInterne & anObj)
{
    BinaryDumpInFile(aFp,anObj.KeyFile());
    BinaryDumpInFile(aFp,anObj.PatternSel().IsInit());
    if (anObj.PatternSel().IsInit()) BinaryDumpInFile(aFp,anObj.PatternSel().Val());
    BinaryDumpInFile(aFp,anObj.Tag().IsInit());
    if (anObj.Tag().IsInit()) BinaryDumpInFile(aFp,anObj.Tag().Val());
    BinaryDumpInFile(aFp,anObj.AddToCur());
    BinaryDumpInFile(aFp,anObj.M2C());
}

cElXMLTree * ToXMLTree(const cSetOrientationInterne & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"SetOrientationInterne",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("KeyFile"),anObj.KeyFile())->ReTagThis("KeyFile"));
   if (anObj.PatternSel().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PatternSel"),anObj.PatternSel().Val())->ReTagThis("PatternSel"));
   if (anObj.Tag().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Tag"),anObj.Tag().Val())->ReTagThis("Tag"));
   aRes->AddFils(::ToXMLTree(std::string("AddToCur"),anObj.AddToCur())->ReTagThis("AddToCur"));
   aRes->AddFils(::ToXMLTree(std::string("M2C"),anObj.M2C())->ReTagThis("M2C"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cSetOrientationInterne & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.KeyFile(),aTree->Get("KeyFile",1)); //tototo 

   xml_init(anObj.PatternSel(),aTree->Get("PatternSel",1),std::string(".*")); //tototo 

   xml_init(anObj.Tag(),aTree->Get("Tag",1),std::string("AffinitePlane")); //tototo 

   xml_init(anObj.AddToCur(),aTree->Get("AddToCur",1)); //tototo 

   xml_init(anObj.M2C(),aTree->Get("M2C",1)); //tototo 
}

std::string  Mangling( cSetOrientationInterne *) {return "E47E304009D485E3FD3F";};


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

void  BinaryUnDumpFromFile(cExportAsNewGrid & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Step(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.RayonInv().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.RayonInv().ValForcedForUnUmp(),aFp);
        }
        else  anObj.RayonInv().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.RayonInvRelFE().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.RayonInvRelFE().ValForcedForUnUmp(),aFp);
        }
        else  anObj.RayonInvRelFE().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cExportAsNewGrid & anObj)
{
    BinaryDumpInFile(aFp,anObj.Step());
    BinaryDumpInFile(aFp,anObj.RayonInv().IsInit());
    if (anObj.RayonInv().IsInit()) BinaryDumpInFile(aFp,anObj.RayonInv().Val());
    BinaryDumpInFile(aFp,anObj.RayonInvRelFE().IsInit());
    if (anObj.RayonInvRelFE().IsInit()) BinaryDumpInFile(aFp,anObj.RayonInvRelFE().Val());
}

cElXMLTree * ToXMLTree(const cExportAsNewGrid & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ExportAsNewGrid",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Step"),anObj.Step())->ReTagThis("Step"));
   if (anObj.RayonInv().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("RayonInv"),anObj.RayonInv().Val())->ReTagThis("RayonInv"));
   if (anObj.RayonInvRelFE().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("RayonInvRelFE"),anObj.RayonInvRelFE().Val())->ReTagThis("RayonInvRelFE"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cExportAsNewGrid & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.Step(),aTree->Get("Step",1)); //tototo 

   xml_init(anObj.RayonInv(),aTree->Get("RayonInv",1),double(-1)); //tototo 

   xml_init(anObj.RayonInvRelFE(),aTree->Get("RayonInvRelFE",1),double(-1)); //tototo 
}

std::string  Mangling( cExportAsNewGrid *) {return "E6D3B778CA74C3EEFD3F";};


double & cXmlPondRegDist::Pds0()
{
   return mPds0;
}

const double & cXmlPondRegDist::Pds0()const 
{
   return mPds0;
}


double & cXmlPondRegDist::Pds1()
{
   return mPds1;
}

const double & cXmlPondRegDist::Pds1()const 
{
   return mPds1;
}


double & cXmlPondRegDist::Pds2()
{
   return mPds2;
}

const double & cXmlPondRegDist::Pds2()const 
{
   return mPds2;
}


double & cXmlPondRegDist::NbCase()
{
   return mNbCase;
}

const double & cXmlPondRegDist::NbCase()const 
{
   return mNbCase;
}


double & cXmlPondRegDist::SeuilNbPtsByCase()
{
   return mSeuilNbPtsByCase;
}

const double & cXmlPondRegDist::SeuilNbPtsByCase()const 
{
   return mSeuilNbPtsByCase;
}

void  BinaryUnDumpFromFile(cXmlPondRegDist & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Pds0(),aFp);
    BinaryUnDumpFromFile(anObj.Pds1(),aFp);
    BinaryUnDumpFromFile(anObj.Pds2(),aFp);
    BinaryUnDumpFromFile(anObj.NbCase(),aFp);
    BinaryUnDumpFromFile(anObj.SeuilNbPtsByCase(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cXmlPondRegDist & anObj)
{
    BinaryDumpInFile(aFp,anObj.Pds0());
    BinaryDumpInFile(aFp,anObj.Pds1());
    BinaryDumpInFile(aFp,anObj.Pds2());
    BinaryDumpInFile(aFp,anObj.NbCase());
    BinaryDumpInFile(aFp,anObj.SeuilNbPtsByCase());
}

cElXMLTree * ToXMLTree(const cXmlPondRegDist & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"XmlPondRegDist",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Pds0"),anObj.Pds0())->ReTagThis("Pds0"));
   aRes->AddFils(::ToXMLTree(std::string("Pds1"),anObj.Pds1())->ReTagThis("Pds1"));
   aRes->AddFils(::ToXMLTree(std::string("Pds2"),anObj.Pds2())->ReTagThis("Pds2"));
   aRes->AddFils(::ToXMLTree(std::string("NbCase"),anObj.NbCase())->ReTagThis("NbCase"));
   aRes->AddFils(::ToXMLTree(std::string("SeuilNbPtsByCase"),anObj.SeuilNbPtsByCase())->ReTagThis("SeuilNbPtsByCase"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cXmlPondRegDist & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.Pds0(),aTree->Get("Pds0",1)); //tototo 

   xml_init(anObj.Pds1(),aTree->Get("Pds1",1)); //tototo 

   xml_init(anObj.Pds2(),aTree->Get("Pds2",1)); //tototo 

   xml_init(anObj.NbCase(),aTree->Get("NbCase",1)); //tototo 

   xml_init(anObj.SeuilNbPtsByCase(),aTree->Get("SeuilNbPtsByCase",1)); //tototo 
}

std::string  Mangling( cXmlPondRegDist *) {return "5D81B7440ADB498DFD3F";};


std::string & cXml_EstimateOrientationInitBlockCamera::Id()
{
   return mId;
}

const std::string & cXml_EstimateOrientationInitBlockCamera::Id()const 
{
   return mId;
}


cTplValGesInit< bool > & cXml_EstimateOrientationInitBlockCamera::Show()
{
   return mShow;
}

const cTplValGesInit< bool > & cXml_EstimateOrientationInitBlockCamera::Show()const 
{
   return mShow;
}

void  BinaryUnDumpFromFile(cXml_EstimateOrientationInitBlockCamera & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Id(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Show().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Show().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Show().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cXml_EstimateOrientationInitBlockCamera & anObj)
{
    BinaryDumpInFile(aFp,anObj.Id());
    BinaryDumpInFile(aFp,anObj.Show().IsInit());
    if (anObj.Show().IsInit()) BinaryDumpInFile(aFp,anObj.Show().Val());
}

cElXMLTree * ToXMLTree(const cXml_EstimateOrientationInitBlockCamera & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"Xml_EstimateOrientationInitBlockCamera",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Id"),anObj.Id())->ReTagThis("Id"));
   if (anObj.Show().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Show"),anObj.Show().Val())->ReTagThis("Show"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cXml_EstimateOrientationInitBlockCamera & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.Id(),aTree->Get("Id",1)); //tototo 

   xml_init(anObj.Show(),aTree->Get("Show",1),bool(false)); //tototo 
}

std::string  Mangling( cXml_EstimateOrientationInitBlockCamera *) {return "45400A05984F87F5FD3F";};


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

void  BinaryUnDumpFromFile(cShowSection & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ShowMes().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ShowMes().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ShowMes().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.LogFile().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.LogFile().ValForcedForUnUmp(),aFp);
        }
        else  anObj.LogFile().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cShowSection & anObj)
{
    BinaryDumpInFile(aFp,anObj.ShowMes().IsInit());
    if (anObj.ShowMes().IsInit()) BinaryDumpInFile(aFp,anObj.ShowMes().Val());
    BinaryDumpInFile(aFp,anObj.LogFile().IsInit());
    if (anObj.LogFile().IsInit()) BinaryDumpInFile(aFp,anObj.LogFile().Val());
}

cElXMLTree * ToXMLTree(const cShowSection & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ShowSection",eXMLBranche);
   if (anObj.ShowMes().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ShowMes"),anObj.ShowMes().Val())->ReTagThis("ShowMes"));
   if (anObj.LogFile().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("LogFile"),anObj.LogFile().Val())->ReTagThis("LogFile"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cShowSection & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.ShowMes(),aTree->Get("ShowMes",1),bool(true)); //tototo 

   xml_init(anObj.LogFile(),aTree->Get("LogFile",1)); //tototo 
}

std::string  Mangling( cShowSection *) {return "ACC9A64E908EBEE4FBBF";};


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

void  BinaryUnDumpFromFile(cSzImForInvY & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.SzIm1(),aFp);
    BinaryUnDumpFromFile(anObj.SzIm2(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cSzImForInvY & anObj)
{
    BinaryDumpInFile(aFp,anObj.SzIm1());
    BinaryDumpInFile(aFp,anObj.SzIm2());
}

cElXMLTree * ToXMLTree(const cSzImForInvY & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"SzImForInvY",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("SzIm1"),anObj.SzIm1())->ReTagThis("SzIm1"));
   aRes->AddFils(::ToXMLTree(std::string("SzIm2"),anObj.SzIm2())->ReTagThis("SzIm2"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cSzImForInvY & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.SzIm1(),aTree->Get("SzIm1",1)); //tototo 

   xml_init(anObj.SzIm2(),aTree->Get("SzIm2",1)); //tototo 
}

std::string  Mangling( cSzImForInvY *) {return "9CA460A13A093CCBFC3F";};


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

void  BinaryUnDumpFromFile(cSplitLayer & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.IdLayer(),aFp);
    BinaryUnDumpFromFile(anObj.KeyCalHomSplit(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cSplitLayer & anObj)
{
    BinaryDumpInFile(aFp,anObj.IdLayer());
    BinaryDumpInFile(aFp,anObj.KeyCalHomSplit());
}

cElXMLTree * ToXMLTree(const cSplitLayer & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"SplitLayer",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("IdLayer"),anObj.IdLayer())->ReTagThis("IdLayer"));
   aRes->AddFils(::ToXMLTree(std::string("KeyCalHomSplit"),anObj.KeyCalHomSplit())->ReTagThis("KeyCalHomSplit"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cSplitLayer & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.IdLayer(),aTree->Get("IdLayer",1)); //tototo 

   xml_init(anObj.KeyCalHomSplit(),aTree->Get("KeyCalHomSplit",1)); //tototo 
}

std::string  Mangling( cSplitLayer *) {return "F6A6448905DCBF8BFF3F";};


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

void  BinaryUnDumpFromFile(cBDD_PtsLiaisons & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.TestForMatin().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.TestForMatin().ValForcedForUnUmp(),aFp);
        }
        else  anObj.TestForMatin().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.UseAsPtMultiple().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.UseAsPtMultiple().ValForcedForUnUmp(),aFp);
        }
        else  anObj.UseAsPtMultiple().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.Id(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.IdFilterSameGrp().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.IdFilterSameGrp().ValForcedForUnUmp(),aFp);
        }
        else  anObj.IdFilterSameGrp().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.AutoSuprReflexif().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.AutoSuprReflexif().ValForcedForUnUmp(),aFp);
        }
        else  anObj.AutoSuprReflexif().SetNoInit();
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             std::string aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.KeySet().push_back(aVal);
        }
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             std::string aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.KeyAssoc().push_back(aVal);
        }
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             std::string aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.XMLKeySetOrPat().push_back(aVal);
        }
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SzImForInvY().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SzImForInvY().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SzImForInvY().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SplitLayer().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SplitLayer().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SplitLayer().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cBDD_PtsLiaisons & anObj)
{
    BinaryDumpInFile(aFp,anObj.TestForMatin().IsInit());
    if (anObj.TestForMatin().IsInit()) BinaryDumpInFile(aFp,anObj.TestForMatin().Val());
    BinaryDumpInFile(aFp,anObj.UseAsPtMultiple().IsInit());
    if (anObj.UseAsPtMultiple().IsInit()) BinaryDumpInFile(aFp,anObj.UseAsPtMultiple().Val());
    BinaryDumpInFile(aFp,anObj.Id());
    BinaryDumpInFile(aFp,anObj.IdFilterSameGrp().IsInit());
    if (anObj.IdFilterSameGrp().IsInit()) BinaryDumpInFile(aFp,anObj.IdFilterSameGrp().Val());
    BinaryDumpInFile(aFp,anObj.AutoSuprReflexif().IsInit());
    if (anObj.AutoSuprReflexif().IsInit()) BinaryDumpInFile(aFp,anObj.AutoSuprReflexif().Val());
    BinaryDumpInFile(aFp,(int)anObj.KeySet().size());
    for(  std::vector< std::string >::const_iterator iT=anObj.KeySet().begin();
         iT!=anObj.KeySet().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,(int)anObj.KeyAssoc().size());
    for(  std::vector< std::string >::const_iterator iT=anObj.KeyAssoc().begin();
         iT!=anObj.KeyAssoc().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,(int)anObj.XMLKeySetOrPat().size());
    for(  std::list< std::string >::const_iterator iT=anObj.XMLKeySetOrPat().begin();
         iT!=anObj.XMLKeySetOrPat().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.SzImForInvY().IsInit());
    if (anObj.SzImForInvY().IsInit()) BinaryDumpInFile(aFp,anObj.SzImForInvY().Val());
    BinaryDumpInFile(aFp,anObj.SplitLayer().IsInit());
    if (anObj.SplitLayer().IsInit()) BinaryDumpInFile(aFp,anObj.SplitLayer().Val());
}

cElXMLTree * ToXMLTree(const cBDD_PtsLiaisons & anObj)
{
  XMLPushContext(anObj.mGXml);
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
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cBDD_PtsLiaisons & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

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

std::string  Mangling( cBDD_PtsLiaisons *) {return "1AA694C1489847B3FCBF";};


std::string & cBDD_NewPtMul::Id()
{
   return mId;
}

const std::string & cBDD_NewPtMul::Id()const 
{
   return mId;
}


std::string & cBDD_NewPtMul::SH()
{
   return mSH;
}

const std::string & cBDD_NewPtMul::SH()const 
{
   return mSH;
}


bool & cBDD_NewPtMul::BinaryMode()
{
   return mBinaryMode;
}

const bool & cBDD_NewPtMul::BinaryMode()const 
{
   return mBinaryMode;
}


bool & cBDD_NewPtMul::SupressStdHom()
{
   return mSupressStdHom;
}

const bool & cBDD_NewPtMul::SupressStdHom()const 
{
   return mSupressStdHom;
}

void  BinaryUnDumpFromFile(cBDD_NewPtMul & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Id(),aFp);
    BinaryUnDumpFromFile(anObj.SH(),aFp);
    BinaryUnDumpFromFile(anObj.BinaryMode(),aFp);
    BinaryUnDumpFromFile(anObj.SupressStdHom(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cBDD_NewPtMul & anObj)
{
    BinaryDumpInFile(aFp,anObj.Id());
    BinaryDumpInFile(aFp,anObj.SH());
    BinaryDumpInFile(aFp,anObj.BinaryMode());
    BinaryDumpInFile(aFp,anObj.SupressStdHom());
}

cElXMLTree * ToXMLTree(const cBDD_NewPtMul & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"BDD_NewPtMul",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Id"),anObj.Id())->ReTagThis("Id"));
   aRes->AddFils(::ToXMLTree(std::string("SH"),anObj.SH())->ReTagThis("SH"));
   aRes->AddFils(::ToXMLTree(std::string("BinaryMode"),anObj.BinaryMode())->ReTagThis("BinaryMode"));
   aRes->AddFils(::ToXMLTree(std::string("SupressStdHom"),anObj.SupressStdHom())->ReTagThis("SupressStdHom"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cBDD_NewPtMul & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.Id(),aTree->Get("Id",1)); //tototo 

   xml_init(anObj.SH(),aTree->Get("SH",1)); //tototo 

   xml_init(anObj.BinaryMode(),aTree->Get("BinaryMode",1)); //tototo 

   xml_init(anObj.SupressStdHom(),aTree->Get("SupressStdHom",1)); //tototo 
}

std::string  Mangling( cBDD_NewPtMul *) {return "CF4E72270E625780FE3F";};


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

void  BinaryUnDumpFromFile(cBddApp_AutoNum & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.DistFusion(),aFp);
    BinaryUnDumpFromFile(anObj.DistAmbiguite(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cBddApp_AutoNum & anObj)
{
    BinaryDumpInFile(aFp,anObj.DistFusion());
    BinaryDumpInFile(aFp,anObj.DistAmbiguite());
}

cElXMLTree * ToXMLTree(const cBddApp_AutoNum & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"BddApp_AutoNum",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("DistFusion"),anObj.DistFusion())->ReTagThis("DistFusion"));
   aRes->AddFils(::ToXMLTree(std::string("DistAmbiguite"),anObj.DistAmbiguite())->ReTagThis("DistAmbiguite"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cBddApp_AutoNum & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.DistFusion(),aTree->Get("DistFusion",1)); //tototo 

   xml_init(anObj.DistAmbiguite(),aTree->Get("DistAmbiguite",1)); //tototo 
}

std::string  Mangling( cBddApp_AutoNum *) {return "B4C64B6142BDB6B0FC3F";};


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

void  BinaryUnDumpFromFile(cBDD_PtsAppuis & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Id(),aFp);
    BinaryUnDumpFromFile(anObj.KeySet(),aFp);
    BinaryUnDumpFromFile(anObj.KeyAssoc(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SzImForInvY().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SzImForInvY().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SzImForInvY().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.InvXY().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.InvXY().ValForcedForUnUmp(),aFp);
        }
        else  anObj.InvXY().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ToSubstract().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ToSubstract().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ToSubstract().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.TagExtract().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.TagExtract().ValForcedForUnUmp(),aFp);
        }
        else  anObj.TagExtract().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.BddApp_AutoNum().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.BddApp_AutoNum().ValForcedForUnUmp(),aFp);
        }
        else  anObj.BddApp_AutoNum().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cBDD_PtsAppuis & anObj)
{
    BinaryDumpInFile(aFp,anObj.Id());
    BinaryDumpInFile(aFp,anObj.KeySet());
    BinaryDumpInFile(aFp,anObj.KeyAssoc());
    BinaryDumpInFile(aFp,anObj.SzImForInvY().IsInit());
    if (anObj.SzImForInvY().IsInit()) BinaryDumpInFile(aFp,anObj.SzImForInvY().Val());
    BinaryDumpInFile(aFp,anObj.InvXY().IsInit());
    if (anObj.InvXY().IsInit()) BinaryDumpInFile(aFp,anObj.InvXY().Val());
    BinaryDumpInFile(aFp,anObj.ToSubstract().IsInit());
    if (anObj.ToSubstract().IsInit()) BinaryDumpInFile(aFp,anObj.ToSubstract().Val());
    BinaryDumpInFile(aFp,anObj.TagExtract().IsInit());
    if (anObj.TagExtract().IsInit()) BinaryDumpInFile(aFp,anObj.TagExtract().Val());
    BinaryDumpInFile(aFp,anObj.BddApp_AutoNum().IsInit());
    if (anObj.BddApp_AutoNum().IsInit()) BinaryDumpInFile(aFp,anObj.BddApp_AutoNum().Val());
}

cElXMLTree * ToXMLTree(const cBDD_PtsAppuis & anObj)
{
  XMLPushContext(anObj.mGXml);
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
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cBDD_PtsAppuis & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.Id(),aTree->Get("Id",1)); //tototo 

   xml_init(anObj.KeySet(),aTree->Get("KeySet",1)); //tototo 

   xml_init(anObj.KeyAssoc(),aTree->Get("KeyAssoc",1)); //tototo 

   xml_init(anObj.SzImForInvY(),aTree->Get("SzImForInvY",1)); //tototo 

   xml_init(anObj.InvXY(),aTree->Get("InvXY",1),bool(false)); //tototo 

   xml_init(anObj.ToSubstract(),aTree->Get("ToSubstract",1)); //tototo 

   xml_init(anObj.TagExtract(),aTree->Get("TagExtract",1),std::string("ListeAppuis1Im")); //tototo 

   xml_init(anObj.BddApp_AutoNum(),aTree->Get("BddApp_AutoNum",1)); //tototo 
}

std::string  Mangling( cBDD_PtsAppuis *) {return "E862FEDE920F378EFBBF";};


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


cTplValGesInit< std::string > & cBDD_ObsAppuisFlottant::KeySetOrPat()
{
   return mKeySetOrPat;
}

const cTplValGesInit< std::string > & cBDD_ObsAppuisFlottant::KeySetOrPat()const 
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


cTplValGesInit< std::string > & cBDD_ObsAppuisFlottant::KeySetSegDroite()
{
   return mKeySetSegDroite;
}

const cTplValGesInit< std::string > & cBDD_ObsAppuisFlottant::KeySetSegDroite()const 
{
   return mKeySetSegDroite;
}

void  BinaryUnDumpFromFile(cBDD_ObsAppuisFlottant & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.OffsetIm().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.OffsetIm().ValForcedForUnUmp(),aFp);
        }
        else  anObj.OffsetIm().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.Id(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.KeySetOrPat().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.KeySetOrPat().ValForcedForUnUmp(),aFp);
        }
        else  anObj.KeySetOrPat().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NameAppuiSelector().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NameAppuiSelector().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NameAppuiSelector().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.AcceptNoGround().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.AcceptNoGround().ValForcedForUnUmp(),aFp);
        }
        else  anObj.AcceptNoGround().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.KeySetSegDroite().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.KeySetSegDroite().ValForcedForUnUmp(),aFp);
        }
        else  anObj.KeySetSegDroite().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cBDD_ObsAppuisFlottant & anObj)
{
    BinaryDumpInFile(aFp,anObj.OffsetIm().IsInit());
    if (anObj.OffsetIm().IsInit()) BinaryDumpInFile(aFp,anObj.OffsetIm().Val());
    BinaryDumpInFile(aFp,anObj.Id());
    BinaryDumpInFile(aFp,anObj.KeySetOrPat().IsInit());
    if (anObj.KeySetOrPat().IsInit()) BinaryDumpInFile(aFp,anObj.KeySetOrPat().Val());
    BinaryDumpInFile(aFp,anObj.NameAppuiSelector().IsInit());
    if (anObj.NameAppuiSelector().IsInit()) BinaryDumpInFile(aFp,anObj.NameAppuiSelector().Val());
    BinaryDumpInFile(aFp,anObj.AcceptNoGround().IsInit());
    if (anObj.AcceptNoGround().IsInit()) BinaryDumpInFile(aFp,anObj.AcceptNoGround().Val());
    BinaryDumpInFile(aFp,anObj.KeySetSegDroite().IsInit());
    if (anObj.KeySetSegDroite().IsInit()) BinaryDumpInFile(aFp,anObj.KeySetSegDroite().Val());
}

cElXMLTree * ToXMLTree(const cBDD_ObsAppuisFlottant & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"BDD_ObsAppuisFlottant",eXMLBranche);
   if (anObj.OffsetIm().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("OffsetIm"),anObj.OffsetIm().Val())->ReTagThis("OffsetIm"));
   aRes->AddFils(::ToXMLTree(std::string("Id"),anObj.Id())->ReTagThis("Id"));
   if (anObj.KeySetOrPat().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("KeySetOrPat"),anObj.KeySetOrPat().Val())->ReTagThis("KeySetOrPat"));
   if (anObj.NameAppuiSelector().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NameAppuiSelector"),anObj.NameAppuiSelector().Val())->ReTagThis("NameAppuiSelector"));
   if (anObj.AcceptNoGround().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("AcceptNoGround"),anObj.AcceptNoGround().Val())->ReTagThis("AcceptNoGround"));
   if (anObj.KeySetSegDroite().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("KeySetSegDroite"),anObj.KeySetSegDroite().Val())->ReTagThis("KeySetSegDroite"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cBDD_ObsAppuisFlottant & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.OffsetIm(),aTree->Get("OffsetIm",1),Pt2dr(Pt2dr(0,0))); //tototo 

   xml_init(anObj.Id(),aTree->Get("Id",1)); //tototo 

   xml_init(anObj.KeySetOrPat(),aTree->Get("KeySetOrPat",1)); //tototo 

   xml_init(anObj.NameAppuiSelector(),aTree->Get("NameAppuiSelector",1)); //tototo 

   xml_init(anObj.AcceptNoGround(),aTree->Get("AcceptNoGround",1),bool(true)); //tototo 

   xml_init(anObj.KeySetSegDroite(),aTree->Get("KeySetSegDroite",1)); //tototo 
}

std::string  Mangling( cBDD_ObsAppuisFlottant *) {return "3F15C340C116E4C6FE3F";};


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

void  BinaryUnDumpFromFile(cBDD_Orient & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Id(),aFp);
    BinaryUnDumpFromFile(anObj.KeySet(),aFp);
    BinaryUnDumpFromFile(anObj.KeyAssoc(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ConvOr().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ConvOr().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ConvOr().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cBDD_Orient & anObj)
{
    BinaryDumpInFile(aFp,anObj.Id());
    BinaryDumpInFile(aFp,anObj.KeySet());
    BinaryDumpInFile(aFp,anObj.KeyAssoc());
    BinaryDumpInFile(aFp,anObj.ConvOr().IsInit());
    if (anObj.ConvOr().IsInit()) BinaryDumpInFile(aFp,anObj.ConvOr().Val());
}

cElXMLTree * ToXMLTree(const cBDD_Orient & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"BDD_Orient",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Id"),anObj.Id())->ReTagThis("Id"));
   aRes->AddFils(::ToXMLTree(std::string("KeySet"),anObj.KeySet())->ReTagThis("KeySet"));
   aRes->AddFils(::ToXMLTree(std::string("KeyAssoc"),anObj.KeyAssoc())->ReTagThis("KeyAssoc"));
   if (anObj.ConvOr().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ConvOr"),anObj.ConvOr().Val())->ReTagThis("ConvOr"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cBDD_Orient & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.Id(),aTree->Get("Id",1)); //tototo 

   xml_init(anObj.KeySet(),aTree->Get("KeySet",1)); //tototo 

   xml_init(anObj.KeyAssoc(),aTree->Get("KeyAssoc",1)); //tototo 

   xml_init(anObj.ConvOr(),aTree->Get("ConvOr",1)); //tototo 
}

std::string  Mangling( cBDD_Orient *) {return "97DAF21E9BF6C0CFFE3F";};


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

void  BinaryUnDumpFromFile(cCalcOffsetCentre & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.IdBase(),aFp);
    BinaryUnDumpFromFile(anObj.KeyCalcBande(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.OffsetUnknown().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.OffsetUnknown().ValForcedForUnUmp(),aFp);
        }
        else  anObj.OffsetUnknown().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cCalcOffsetCentre & anObj)
{
    BinaryDumpInFile(aFp,anObj.IdBase());
    BinaryDumpInFile(aFp,anObj.KeyCalcBande());
    BinaryDumpInFile(aFp,anObj.OffsetUnknown().IsInit());
    if (anObj.OffsetUnknown().IsInit()) BinaryDumpInFile(aFp,anObj.OffsetUnknown().Val());
}

cElXMLTree * ToXMLTree(const cCalcOffsetCentre & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"CalcOffsetCentre",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("IdBase"),anObj.IdBase())->ReTagThis("IdBase"));
   aRes->AddFils(::ToXMLTree(std::string("KeyCalcBande"),anObj.KeyCalcBande())->ReTagThis("KeyCalcBande"));
   if (anObj.OffsetUnknown().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("OffsetUnknown"),anObj.OffsetUnknown().Val())->ReTagThis("OffsetUnknown"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cCalcOffsetCentre & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.IdBase(),aTree->Get("IdBase",1)); //tototo 

   xml_init(anObj.KeyCalcBande(),aTree->Get("KeyCalcBande",1)); //tototo 

   xml_init(anObj.OffsetUnknown(),aTree->Get("OffsetUnknown",1),bool(false)); //tototo 
}

std::string  Mangling( cCalcOffsetCentre *) {return "2A0748632BBC3FF5FD3F";};


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

void  BinaryUnDumpFromFile(cBDD_Centre & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Id(),aFp);
    BinaryUnDumpFromFile(anObj.KeySet(),aFp);
    BinaryUnDumpFromFile(anObj.KeyAssoc(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Tag().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Tag().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Tag().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ByFileTrajecto().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ByFileTrajecto().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ByFileTrajecto().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.PatternFileTrajecto().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.PatternFileTrajecto().ValForcedForUnUmp(),aFp);
        }
        else  anObj.PatternFileTrajecto().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.PatternRefutFileTrajecto().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.PatternRefutFileTrajecto().ValForcedForUnUmp(),aFp);
        }
        else  anObj.PatternRefutFileTrajecto().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.CalcOffsetCentre().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.CalcOffsetCentre().ValForcedForUnUmp(),aFp);
        }
        else  anObj.CalcOffsetCentre().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cBDD_Centre & anObj)
{
    BinaryDumpInFile(aFp,anObj.Id());
    BinaryDumpInFile(aFp,anObj.KeySet());
    BinaryDumpInFile(aFp,anObj.KeyAssoc());
    BinaryDumpInFile(aFp,anObj.Tag().IsInit());
    if (anObj.Tag().IsInit()) BinaryDumpInFile(aFp,anObj.Tag().Val());
    BinaryDumpInFile(aFp,anObj.ByFileTrajecto().IsInit());
    if (anObj.ByFileTrajecto().IsInit()) BinaryDumpInFile(aFp,anObj.ByFileTrajecto().Val());
    BinaryDumpInFile(aFp,anObj.PatternFileTrajecto().IsInit());
    if (anObj.PatternFileTrajecto().IsInit()) BinaryDumpInFile(aFp,anObj.PatternFileTrajecto().Val());
    BinaryDumpInFile(aFp,anObj.PatternRefutFileTrajecto().IsInit());
    if (anObj.PatternRefutFileTrajecto().IsInit()) BinaryDumpInFile(aFp,anObj.PatternRefutFileTrajecto().Val());
    BinaryDumpInFile(aFp,anObj.CalcOffsetCentre().IsInit());
    if (anObj.CalcOffsetCentre().IsInit()) BinaryDumpInFile(aFp,anObj.CalcOffsetCentre().Val());
}

cElXMLTree * ToXMLTree(const cBDD_Centre & anObj)
{
  XMLPushContext(anObj.mGXml);
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
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cBDD_Centre & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.Id(),aTree->Get("Id",1)); //tototo 

   xml_init(anObj.KeySet(),aTree->Get("KeySet",1)); //tototo 

   xml_init(anObj.KeyAssoc(),aTree->Get("KeyAssoc",1)); //tototo 

   xml_init(anObj.Tag(),aTree->Get("Tag",1),std::string("Centre")); //tototo 

   xml_init(anObj.ByFileTrajecto(),aTree->Get("ByFileTrajecto",1)); //tototo 

   xml_init(anObj.PatternFileTrajecto(),aTree->Get("PatternFileTrajecto",1)); //tototo 

   xml_init(anObj.PatternRefutFileTrajecto(),aTree->Get("PatternRefutFileTrajecto",1)); //tototo 

   xml_init(anObj.CalcOffsetCentre(),aTree->Get("CalcOffsetCentre",1)); //tototo 
}

std::string  Mangling( cBDD_Centre *) {return "A07839AA37B0DE8DFB3F";};


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

void  BinaryUnDumpFromFile(cFilterProj3D & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Id(),aFp);
    BinaryUnDumpFromFile(anObj.PatternSel(),aFp);
    BinaryUnDumpFromFile(anObj.AttrSup(),aFp);
    BinaryUnDumpFromFile(anObj.KeyCalculMasq(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cFilterProj3D & anObj)
{
    BinaryDumpInFile(aFp,anObj.Id());
    BinaryDumpInFile(aFp,anObj.PatternSel());
    BinaryDumpInFile(aFp,anObj.AttrSup());
    BinaryDumpInFile(aFp,anObj.KeyCalculMasq());
}

cElXMLTree * ToXMLTree(const cFilterProj3D & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"FilterProj3D",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Id"),anObj.Id())->ReTagThis("Id"));
   aRes->AddFils(::ToXMLTree(std::string("PatternSel"),anObj.PatternSel())->ReTagThis("PatternSel"));
   aRes->AddFils(::ToXMLTree(std::string("AttrSup"),anObj.AttrSup())->ReTagThis("AttrSup"));
   aRes->AddFils(::ToXMLTree(std::string("KeyCalculMasq"),anObj.KeyCalculMasq())->ReTagThis("KeyCalculMasq"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cFilterProj3D & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.Id(),aTree->Get("Id",1)); //tototo 

   xml_init(anObj.PatternSel(),aTree->Get("PatternSel",1)); //tototo 

   xml_init(anObj.AttrSup(),aTree->Get("AttrSup",1)); //tototo 

   xml_init(anObj.KeyCalculMasq(),aTree->Get("KeyCalculMasq",1)); //tototo 
}

std::string  Mangling( cFilterProj3D *) {return "F021A3EAE351FEE2FD3F";};


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

void  BinaryUnDumpFromFile(cLayerTerrain & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.KeyAssocGeoref().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.KeyAssocGeoref().ValForcedForUnUmp(),aFp);
        }
        else  anObj.KeyAssocGeoref().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.KeyAssocOrImage(),aFp);
    BinaryUnDumpFromFile(anObj.SysCoIm(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.TagOri().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.TagOri().ValForcedForUnUmp(),aFp);
        }
        else  anObj.TagOri().SetNoInit();
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

void  BinaryDumpInFile(ELISE_fp & aFp,const cLayerTerrain & anObj)
{
    BinaryDumpInFile(aFp,anObj.KeyAssocGeoref().IsInit());
    if (anObj.KeyAssocGeoref().IsInit()) BinaryDumpInFile(aFp,anObj.KeyAssocGeoref().Val());
    BinaryDumpInFile(aFp,anObj.KeyAssocOrImage());
    BinaryDumpInFile(aFp,anObj.SysCoIm());
    BinaryDumpInFile(aFp,anObj.TagOri().IsInit());
    if (anObj.TagOri().IsInit()) BinaryDumpInFile(aFp,anObj.TagOri().Val());
    BinaryDumpInFile(aFp,anObj.ZMoyen().IsInit());
    if (anObj.ZMoyen().IsInit()) BinaryDumpInFile(aFp,anObj.ZMoyen().Val());
}

cElXMLTree * ToXMLTree(const cLayerTerrain & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"LayerTerrain",eXMLBranche);
   if (anObj.KeyAssocGeoref().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("KeyAssocGeoref"),anObj.KeyAssocGeoref().Val())->ReTagThis("KeyAssocGeoref"));
   aRes->AddFils(::ToXMLTree(std::string("KeyAssocOrImage"),anObj.KeyAssocOrImage())->ReTagThis("KeyAssocOrImage"));
   aRes->AddFils(::ToXMLTree(std::string("SysCoIm"),anObj.SysCoIm())->ReTagThis("SysCoIm"));
   if (anObj.TagOri().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("TagOri"),anObj.TagOri().Val())->ReTagThis("TagOri"));
   if (anObj.ZMoyen().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ZMoyen"),anObj.ZMoyen().Val())->ReTagThis("ZMoyen"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cLayerTerrain & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.KeyAssocGeoref(),aTree->Get("KeyAssocGeoref",1),std::string("KeyStd-Assoc-ChangExt@xml")); //tototo 

   xml_init(anObj.KeyAssocOrImage(),aTree->Get("KeyAssocOrImage",1)); //tototo 

   xml_init(anObj.SysCoIm(),aTree->Get("SysCoIm",1)); //tototo 

   xml_init(anObj.TagOri(),aTree->Get("TagOri",1),std::string("OrientationConique")); //tototo 

   xml_init(anObj.ZMoyen(),aTree->Get("ZMoyen",1)); //tototo 
}

std::string  Mangling( cLayerTerrain *) {return "905547036DBE29D5FE3F";};


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

void  BinaryUnDumpFromFile(cLayerImageToPose & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Id(),aFp);
    BinaryUnDumpFromFile(anObj.KeyCalculImage(),aFp);
    BinaryUnDumpFromFile(anObj.FactRed(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.KeyNameRed().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.KeyNameRed().ValForcedForUnUmp(),aFp);
        }
        else  anObj.KeyNameRed().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.FactCoherence().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.FactCoherence().ValForcedForUnUmp(),aFp);
        }
        else  anObj.FactCoherence().SetNoInit();
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             int aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.EtiqPrio().push_back(aVal);
        }
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.LayerTerrain().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.LayerTerrain().ValForcedForUnUmp(),aFp);
        }
        else  anObj.LayerTerrain().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cLayerImageToPose & anObj)
{
    BinaryDumpInFile(aFp,anObj.Id());
    BinaryDumpInFile(aFp,anObj.KeyCalculImage());
    BinaryDumpInFile(aFp,anObj.FactRed());
    BinaryDumpInFile(aFp,anObj.KeyNameRed().IsInit());
    if (anObj.KeyNameRed().IsInit()) BinaryDumpInFile(aFp,anObj.KeyNameRed().Val());
    BinaryDumpInFile(aFp,anObj.FactCoherence().IsInit());
    if (anObj.FactCoherence().IsInit()) BinaryDumpInFile(aFp,anObj.FactCoherence().Val());
    BinaryDumpInFile(aFp,(int)anObj.EtiqPrio().size());
    for(  std::vector< int >::const_iterator iT=anObj.EtiqPrio().begin();
         iT!=anObj.EtiqPrio().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.LayerTerrain().IsInit());
    if (anObj.LayerTerrain().IsInit()) BinaryDumpInFile(aFp,anObj.LayerTerrain().Val());
}

cElXMLTree * ToXMLTree(const cLayerImageToPose & anObj)
{
  XMLPushContext(anObj.mGXml);
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
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cLayerImageToPose & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.Id(),aTree->Get("Id",1)); //tototo 

   xml_init(anObj.KeyCalculImage(),aTree->Get("KeyCalculImage",1)); //tototo 

   xml_init(anObj.FactRed(),aTree->Get("FactRed",1)); //tototo 

   xml_init(anObj.KeyNameRed(),aTree->Get("KeyNameRed",1),std::string("KeyStd-Assoc-AddPref@Layer-Reduc-")); //tototo 

   xml_init(anObj.FactCoherence(),aTree->Get("FactCoherence",1),int(-1)); //tototo 

   xml_init(anObj.EtiqPrio(),aTree->GetAll("EtiqPrio",false,1));

   xml_init(anObj.LayerTerrain(),aTree->Get("LayerTerrain",1)); //tototo 
}

std::string  Mangling( cLayerImageToPose *) {return "C8573647DED717BFFB3F";};


std::string & cDeclareObsRelGPS::PatternSel()
{
   return mPatternSel;
}

const std::string & cDeclareObsRelGPS::PatternSel()const 
{
   return mPatternSel;
}


std::string & cDeclareObsRelGPS::Id()
{
   return mId;
}

const std::string & cDeclareObsRelGPS::Id()const 
{
   return mId;
}

void  BinaryUnDumpFromFile(cDeclareObsRelGPS & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.PatternSel(),aFp);
    BinaryUnDumpFromFile(anObj.Id(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cDeclareObsRelGPS & anObj)
{
    BinaryDumpInFile(aFp,anObj.PatternSel());
    BinaryDumpInFile(aFp,anObj.Id());
}

cElXMLTree * ToXMLTree(const cDeclareObsRelGPS & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"DeclareObsRelGPS",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("PatternSel"),anObj.PatternSel())->ReTagThis("PatternSel"));
   aRes->AddFils(::ToXMLTree(std::string("Id"),anObj.Id())->ReTagThis("Id"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cDeclareObsRelGPS & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.PatternSel(),aTree->Get("PatternSel",1)); //tototo 

   xml_init(anObj.Id(),aTree->Get("Id",1)); //tototo 
}

std::string  Mangling( cDeclareObsRelGPS *) {return "BACAA294FA32E5D4FD3F";};


std::string & cDeclareObsCalConseq::PatternSel()
{
   return mPatternSel;
}

const std::string & cDeclareObsCalConseq::PatternSel()const 
{
   return mPatternSel;
}


std::string & cDeclareObsCalConseq::Key()
{
   return mKey;
}

const std::string & cDeclareObsCalConseq::Key()const 
{
   return mKey;
}


cTplValGesInit< std::string > & cDeclareObsCalConseq::KeyJump()
{
   return mKeyJump;
}

const cTplValGesInit< std::string > & cDeclareObsCalConseq::KeyJump()const 
{
   return mKeyJump;
}


bool & cDeclareObsCalConseq::AddFreeRot()
{
   return mAddFreeRot;
}

const bool & cDeclareObsCalConseq::AddFreeRot()const 
{
   return mAddFreeRot;
}

void  BinaryUnDumpFromFile(cDeclareObsCalConseq & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.PatternSel(),aFp);
    BinaryUnDumpFromFile(anObj.Key(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.KeyJump().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.KeyJump().ValForcedForUnUmp(),aFp);
        }
        else  anObj.KeyJump().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.AddFreeRot(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cDeclareObsCalConseq & anObj)
{
    BinaryDumpInFile(aFp,anObj.PatternSel());
    BinaryDumpInFile(aFp,anObj.Key());
    BinaryDumpInFile(aFp,anObj.KeyJump().IsInit());
    if (anObj.KeyJump().IsInit()) BinaryDumpInFile(aFp,anObj.KeyJump().Val());
    BinaryDumpInFile(aFp,anObj.AddFreeRot());
}

cElXMLTree * ToXMLTree(const cDeclareObsCalConseq & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"DeclareObsCalConseq",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("PatternSel"),anObj.PatternSel())->ReTagThis("PatternSel"));
   aRes->AddFils(::ToXMLTree(std::string("Key"),anObj.Key())->ReTagThis("Key"));
   if (anObj.KeyJump().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("KeyJump"),anObj.KeyJump().Val())->ReTagThis("KeyJump"));
   aRes->AddFils(::ToXMLTree(std::string("AddFreeRot"),anObj.AddFreeRot())->ReTagThis("AddFreeRot"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cDeclareObsCalConseq & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.PatternSel(),aTree->Get("PatternSel",1)); //tototo 

   xml_init(anObj.Key(),aTree->Get("Key",1)); //tototo 

   xml_init(anObj.KeyJump(),aTree->Get("KeyJump",1)); //tototo 

   xml_init(anObj.AddFreeRot(),aTree->Get("AddFreeRot",1)); //tototo 
}

std::string  Mangling( cDeclareObsCalConseq *) {return "02437223BDBFB5C4FE3F";};


std::list< cBDD_PtsLiaisons > & cSectionBDD_Observation::BDD_PtsLiaisons()
{
   return mBDD_PtsLiaisons;
}

const std::list< cBDD_PtsLiaisons > & cSectionBDD_Observation::BDD_PtsLiaisons()const 
{
   return mBDD_PtsLiaisons;
}


std::list< cBDD_NewPtMul > & cSectionBDD_Observation::BDD_NewPtMul()
{
   return mBDD_NewPtMul;
}

const std::list< cBDD_NewPtMul > & cSectionBDD_Observation::BDD_NewPtMul()const 
{
   return mBDD_NewPtMul;
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


std::list< cDeclareObsRelGPS > & cSectionBDD_Observation::DeclareObsRelGPS()
{
   return mDeclareObsRelGPS;
}

const std::list< cDeclareObsRelGPS > & cSectionBDD_Observation::DeclareObsRelGPS()const 
{
   return mDeclareObsRelGPS;
}


std::string & cSectionBDD_Observation::PatternSel()
{
   return DeclareObsCalConseq().Val().PatternSel();
}

const std::string & cSectionBDD_Observation::PatternSel()const 
{
   return DeclareObsCalConseq().Val().PatternSel();
}


std::string & cSectionBDD_Observation::Key()
{
   return DeclareObsCalConseq().Val().Key();
}

const std::string & cSectionBDD_Observation::Key()const 
{
   return DeclareObsCalConseq().Val().Key();
}


cTplValGesInit< std::string > & cSectionBDD_Observation::KeyJump()
{
   return DeclareObsCalConseq().Val().KeyJump();
}

const cTplValGesInit< std::string > & cSectionBDD_Observation::KeyJump()const 
{
   return DeclareObsCalConseq().Val().KeyJump();
}


bool & cSectionBDD_Observation::AddFreeRot()
{
   return DeclareObsCalConseq().Val().AddFreeRot();
}

const bool & cSectionBDD_Observation::AddFreeRot()const 
{
   return DeclareObsCalConseq().Val().AddFreeRot();
}


cTplValGesInit< cDeclareObsCalConseq > & cSectionBDD_Observation::DeclareObsCalConseq()
{
   return mDeclareObsCalConseq;
}

const cTplValGesInit< cDeclareObsCalConseq > & cSectionBDD_Observation::DeclareObsCalConseq()const 
{
   return mDeclareObsCalConseq;
}

void  BinaryUnDumpFromFile(cSectionBDD_Observation & anObj,ELISE_fp & aFp)
{
   { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cBDD_PtsLiaisons aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.BDD_PtsLiaisons().push_back(aVal);
        }
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cBDD_NewPtMul aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.BDD_NewPtMul().push_back(aVal);
        }
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cBDD_PtsAppuis aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.BDD_PtsAppuis().push_back(aVal);
        }
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cBDD_ObsAppuisFlottant aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.BDD_ObsAppuisFlottant().push_back(aVal);
        }
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cBDD_Orient aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.BDD_Orient().push_back(aVal);
        }
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cBDD_Centre aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.BDD_Centre().push_back(aVal);
        }
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cFilterProj3D aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.FilterProj3D().push_back(aVal);
        }
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cLayerImageToPose aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.LayerImageToPose().push_back(aVal);
        }
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.LimInfBSurHPMoy().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.LimInfBSurHPMoy().ValForcedForUnUmp(),aFp);
        }
        else  anObj.LimInfBSurHPMoy().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.LimSupBSurHPMoy().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.LimSupBSurHPMoy().ValForcedForUnUmp(),aFp);
        }
        else  anObj.LimSupBSurHPMoy().SetNoInit();
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cDeclareObsRelGPS aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.DeclareObsRelGPS().push_back(aVal);
        }
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.DeclareObsCalConseq().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.DeclareObsCalConseq().ValForcedForUnUmp(),aFp);
        }
        else  anObj.DeclareObsCalConseq().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cSectionBDD_Observation & anObj)
{
    BinaryDumpInFile(aFp,(int)anObj.BDD_PtsLiaisons().size());
    for(  std::list< cBDD_PtsLiaisons >::const_iterator iT=anObj.BDD_PtsLiaisons().begin();
         iT!=anObj.BDD_PtsLiaisons().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,(int)anObj.BDD_NewPtMul().size());
    for(  std::list< cBDD_NewPtMul >::const_iterator iT=anObj.BDD_NewPtMul().begin();
         iT!=anObj.BDD_NewPtMul().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,(int)anObj.BDD_PtsAppuis().size());
    for(  std::list< cBDD_PtsAppuis >::const_iterator iT=anObj.BDD_PtsAppuis().begin();
         iT!=anObj.BDD_PtsAppuis().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,(int)anObj.BDD_ObsAppuisFlottant().size());
    for(  std::list< cBDD_ObsAppuisFlottant >::const_iterator iT=anObj.BDD_ObsAppuisFlottant().begin();
         iT!=anObj.BDD_ObsAppuisFlottant().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,(int)anObj.BDD_Orient().size());
    for(  std::list< cBDD_Orient >::const_iterator iT=anObj.BDD_Orient().begin();
         iT!=anObj.BDD_Orient().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,(int)anObj.BDD_Centre().size());
    for(  std::list< cBDD_Centre >::const_iterator iT=anObj.BDD_Centre().begin();
         iT!=anObj.BDD_Centre().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,(int)anObj.FilterProj3D().size());
    for(  std::list< cFilterProj3D >::const_iterator iT=anObj.FilterProj3D().begin();
         iT!=anObj.FilterProj3D().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,(int)anObj.LayerImageToPose().size());
    for(  std::list< cLayerImageToPose >::const_iterator iT=anObj.LayerImageToPose().begin();
         iT!=anObj.LayerImageToPose().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.LimInfBSurHPMoy().IsInit());
    if (anObj.LimInfBSurHPMoy().IsInit()) BinaryDumpInFile(aFp,anObj.LimInfBSurHPMoy().Val());
    BinaryDumpInFile(aFp,anObj.LimSupBSurHPMoy().IsInit());
    if (anObj.LimSupBSurHPMoy().IsInit()) BinaryDumpInFile(aFp,anObj.LimSupBSurHPMoy().Val());
    BinaryDumpInFile(aFp,(int)anObj.DeclareObsRelGPS().size());
    for(  std::list< cDeclareObsRelGPS >::const_iterator iT=anObj.DeclareObsRelGPS().begin();
         iT!=anObj.DeclareObsRelGPS().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.DeclareObsCalConseq().IsInit());
    if (anObj.DeclareObsCalConseq().IsInit()) BinaryDumpInFile(aFp,anObj.DeclareObsCalConseq().Val());
}

cElXMLTree * ToXMLTree(const cSectionBDD_Observation & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"SectionBDD_Observation",eXMLBranche);
  for
  (       std::list< cBDD_PtsLiaisons >::const_iterator it=anObj.BDD_PtsLiaisons().begin();
      it !=anObj.BDD_PtsLiaisons().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("BDD_PtsLiaisons"));
  for
  (       std::list< cBDD_NewPtMul >::const_iterator it=anObj.BDD_NewPtMul().begin();
      it !=anObj.BDD_NewPtMul().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("BDD_NewPtMul"));
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
  for
  (       std::list< cDeclareObsRelGPS >::const_iterator it=anObj.DeclareObsRelGPS().begin();
      it !=anObj.DeclareObsRelGPS().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("DeclareObsRelGPS"));
   if (anObj.DeclareObsCalConseq().IsInit())
      aRes->AddFils(ToXMLTree(anObj.DeclareObsCalConseq().Val())->ReTagThis("DeclareObsCalConseq"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cSectionBDD_Observation & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.BDD_PtsLiaisons(),aTree->GetAll("BDD_PtsLiaisons",false,1));

   xml_init(anObj.BDD_NewPtMul(),aTree->GetAll("BDD_NewPtMul",false,1));

   xml_init(anObj.BDD_PtsAppuis(),aTree->GetAll("BDD_PtsAppuis",false,1));

   xml_init(anObj.BDD_ObsAppuisFlottant(),aTree->GetAll("BDD_ObsAppuisFlottant",false,1));

   xml_init(anObj.BDD_Orient(),aTree->GetAll("BDD_Orient",false,1));

   xml_init(anObj.BDD_Centre(),aTree->GetAll("BDD_Centre",false,1));

   xml_init(anObj.FilterProj3D(),aTree->GetAll("FilterProj3D",false,1));

   xml_init(anObj.LayerImageToPose(),aTree->GetAll("LayerImageToPose",false,1));

   xml_init(anObj.LimInfBSurHPMoy(),aTree->Get("LimInfBSurHPMoy",1),double(1e-2)); //tototo 

   xml_init(anObj.LimSupBSurHPMoy(),aTree->Get("LimSupBSurHPMoy",1),double(2e-1)); //tototo 

   xml_init(anObj.DeclareObsRelGPS(),aTree->GetAll("DeclareObsRelGPS",false,1));

   xml_init(anObj.DeclareObsCalConseq(),aTree->Get("DeclareObsCalConseq",1)); //tototo 
}

std::string  Mangling( cSectionBDD_Observation *) {return "0268A1DB0723538BFF3F";};


Pt3dr & cGpsOffset::ValInit()
{
   return mValInit;
}

const Pt3dr & cGpsOffset::ValInit()const 
{
   return mValInit;
}


std::string & cGpsOffset::Id()
{
   return mId;
}

const std::string & cGpsOffset::Id()const 
{
   return mId;
}


cTplValGesInit< Pt3dr > & cGpsOffset::Inc()
{
   return mInc;
}

const cTplValGesInit< Pt3dr > & cGpsOffset::Inc()const 
{
   return mInc;
}

void  BinaryUnDumpFromFile(cGpsOffset & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.ValInit(),aFp);
    BinaryUnDumpFromFile(anObj.Id(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Inc().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Inc().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Inc().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cGpsOffset & anObj)
{
    BinaryDumpInFile(aFp,anObj.ValInit());
    BinaryDumpInFile(aFp,anObj.Id());
    BinaryDumpInFile(aFp,anObj.Inc().IsInit());
    if (anObj.Inc().IsInit()) BinaryDumpInFile(aFp,anObj.Inc().Val());
}

cElXMLTree * ToXMLTree(const cGpsOffset & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"GpsOffset",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("ValInit"),anObj.ValInit())->ReTagThis("ValInit"));
   aRes->AddFils(::ToXMLTree(std::string("Id"),anObj.Id())->ReTagThis("Id"));
   if (anObj.Inc().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Inc"),anObj.Inc().Val())->ReTagThis("Inc"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cGpsOffset & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.ValInit(),aTree->Get("ValInit",1)); //tototo 

   xml_init(anObj.Id(),aTree->Get("Id",1)); //tototo 

   xml_init(anObj.Inc(),aTree->Get("Inc",1)); //tototo 
}

std::string  Mangling( cGpsOffset *) {return "B1A66427BA26C3B0FF3F";};


std::string & cDataObsPlane::Id()
{
   return mId;
}

const std::string & cDataObsPlane::Id()const 
{
   return mId;
}


std::string & cDataObsPlane::NameFile()
{
   return mNameFile;
}

const std::string & cDataObsPlane::NameFile()const 
{
   return mNameFile;
}


cTplValGesInit< double > & cDataObsPlane::Weight()
{
   return mWeight;
}

const cTplValGesInit< double > & cDataObsPlane::Weight()const 
{
   return mWeight;
}


cXml_FileObsPlane & cDataObsPlane::Data()
{
   return mData;
}

const cXml_FileObsPlane & cDataObsPlane::Data()const 
{
   return mData;
}

void  BinaryUnDumpFromFile(cDataObsPlane & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Id(),aFp);
    BinaryUnDumpFromFile(anObj.NameFile(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Weight().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Weight().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Weight().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.Data(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cDataObsPlane & anObj)
{
    BinaryDumpInFile(aFp,anObj.Id());
    BinaryDumpInFile(aFp,anObj.NameFile());
    BinaryDumpInFile(aFp,anObj.Weight().IsInit());
    if (anObj.Weight().IsInit()) BinaryDumpInFile(aFp,anObj.Weight().Val());
    BinaryDumpInFile(aFp,anObj.Data());
}

cElXMLTree * ToXMLTree(const cDataObsPlane & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"DataObsPlane",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Id"),anObj.Id())->ReTagThis("Id"));
   aRes->AddFils(::ToXMLTree(std::string("NameFile"),anObj.NameFile())->ReTagThis("NameFile"));
   if (anObj.Weight().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Weight"),anObj.Weight().Val())->ReTagThis("Weight"));
   aRes->AddFils(ToXMLTree(anObj.Data())->ReTagThis("Data"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cDataObsPlane & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.Id(),aTree->Get("Id",1)); //tototo 

   xml_init(anObj.NameFile(),aTree->Get("NameFile",1)); //tototo 

   xml_init(anObj.Weight(),aTree->Get("Weight",1),double(1.0)); //tototo 

   xml_init(anObj.Data(),aTree->Get("Data",1)); //tototo 
}

std::string  Mangling( cDataObsPlane *) {return "1DF7CA55C725788BFD3F";};


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

void  BinaryUnDumpFromFile(cCalibAutomNoDist & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.TypeDist(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NameIm().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NameIm().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NameIm().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.KeyFileSauv().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.KeyFileSauv().ValForcedForUnUmp(),aFp);
        }
        else  anObj.KeyFileSauv().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.PositionRelPP().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.PositionRelPP().ValForcedForUnUmp(),aFp);
        }
        else  anObj.PositionRelPP().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cCalibAutomNoDist & anObj)
{
    BinaryDumpInFile(aFp,anObj.TypeDist());
    BinaryDumpInFile(aFp,anObj.NameIm().IsInit());
    if (anObj.NameIm().IsInit()) BinaryDumpInFile(aFp,anObj.NameIm().Val());
    BinaryDumpInFile(aFp,anObj.KeyFileSauv().IsInit());
    if (anObj.KeyFileSauv().IsInit()) BinaryDumpInFile(aFp,anObj.KeyFileSauv().Val());
    BinaryDumpInFile(aFp,anObj.PositionRelPP().IsInit());
    if (anObj.PositionRelPP().IsInit()) BinaryDumpInFile(aFp,anObj.PositionRelPP().Val());
}

cElXMLTree * ToXMLTree(const cCalibAutomNoDist & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"CalibAutomNoDist",eXMLBranche);
   aRes->AddFils(ToXMLTree(std::string("TypeDist"),anObj.TypeDist())->ReTagThis("TypeDist"));
   if (anObj.NameIm().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NameIm"),anObj.NameIm().Val())->ReTagThis("NameIm"));
   if (anObj.KeyFileSauv().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("KeyFileSauv"),anObj.KeyFileSauv().Val())->ReTagThis("KeyFileSauv"));
   if (anObj.PositionRelPP().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PositionRelPP"),anObj.PositionRelPP().Val())->ReTagThis("PositionRelPP"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cCalibAutomNoDist & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.TypeDist(),aTree->Get("TypeDist",1)); //tototo 

   xml_init(anObj.NameIm(),aTree->Get("NameIm",1)); //tototo 

   xml_init(anObj.KeyFileSauv(),aTree->Get("KeyFileSauv",1)); //tototo 

   xml_init(anObj.PositionRelPP(),aTree->Get("PositionRelPP",1),Pt2dr(Pt2dr(0.5,0.5))); //tototo 
}

std::string  Mangling( cCalibAutomNoDist *) {return "78EC5D33965178BEFE3F";};


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

void  BinaryUnDumpFromFile(cCalValueInit & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.CalFromValues().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.CalFromValues().ValForcedForUnUmp(),aFp);
        }
        else  anObj.CalFromValues().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.CalFromFileExtern().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.CalFromFileExtern().ValForcedForUnUmp(),aFp);
        }
        else  anObj.CalFromFileExtern().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.CalibFromMmBD().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.CalibFromMmBD().ValForcedForUnUmp(),aFp);
        }
        else  anObj.CalibFromMmBD().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.CalibAutomNoDist().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.CalibAutomNoDist().ValForcedForUnUmp(),aFp);
        }
        else  anObj.CalibAutomNoDist().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cCalValueInit & anObj)
{
    BinaryDumpInFile(aFp,anObj.CalFromValues().IsInit());
    if (anObj.CalFromValues().IsInit()) BinaryDumpInFile(aFp,anObj.CalFromValues().Val());
    BinaryDumpInFile(aFp,anObj.CalFromFileExtern().IsInit());
    if (anObj.CalFromFileExtern().IsInit()) BinaryDumpInFile(aFp,anObj.CalFromFileExtern().Val());
    BinaryDumpInFile(aFp,anObj.CalibFromMmBD().IsInit());
    if (anObj.CalibFromMmBD().IsInit()) BinaryDumpInFile(aFp,anObj.CalibFromMmBD().Val());
    BinaryDumpInFile(aFp,anObj.CalibAutomNoDist().IsInit());
    if (anObj.CalibAutomNoDist().IsInit()) BinaryDumpInFile(aFp,anObj.CalibAutomNoDist().Val());
}

cElXMLTree * ToXMLTree(const cCalValueInit & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"CalValueInit",eXMLBranche);
   if (anObj.CalFromValues().IsInit())
      aRes->AddFils(ToXMLTree(anObj.CalFromValues().Val())->ReTagThis("CalFromValues"));
   if (anObj.CalFromFileExtern().IsInit())
      aRes->AddFils(ToXMLTree(anObj.CalFromFileExtern().Val())->ReTagThis("CalFromFileExtern"));
   if (anObj.CalibFromMmBD().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("CalibFromMmBD"),anObj.CalibFromMmBD().Val())->ReTagThis("CalibFromMmBD"));
   if (anObj.CalibAutomNoDist().IsInit())
      aRes->AddFils(ToXMLTree(anObj.CalibAutomNoDist().Val())->ReTagThis("CalibAutomNoDist"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cCalValueInit & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.CalFromValues(),aTree->Get("CalFromValues",1)); //tototo 

   xml_init(anObj.CalFromFileExtern(),aTree->Get("CalFromFileExtern",1)); //tototo 

   xml_init(anObj.CalibFromMmBD(),aTree->Get("CalibFromMmBD",1),bool(true)); //tototo 

   xml_init(anObj.CalibAutomNoDist(),aTree->Get("CalibAutomNoDist",1)); //tototo 
}

std::string  Mangling( cCalValueInit *) {return "6130291441ACEDC3FD3F";};


std::vector< double > & cAddParamAFocal::Coeffs()
{
   return mCoeffs;
}

const std::vector< double > & cAddParamAFocal::Coeffs()const 
{
   return mCoeffs;
}

void  BinaryUnDumpFromFile(cAddParamAFocal & anObj,ELISE_fp & aFp)
{
   { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             double aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.Coeffs().push_back(aVal);
        }
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cAddParamAFocal & anObj)
{
    BinaryDumpInFile(aFp,(int)anObj.Coeffs().size());
    for(  std::vector< double >::const_iterator iT=anObj.Coeffs().begin();
         iT!=anObj.Coeffs().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
}

cElXMLTree * ToXMLTree(const cAddParamAFocal & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"AddParamAFocal",eXMLBranche);
  for
  (       std::vector< double >::const_iterator it=anObj.Coeffs().begin();
      it !=anObj.Coeffs().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("Coeffs"),(*it))->ReTagThis("Coeffs"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cAddParamAFocal & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.Coeffs(),aTree->GetAll("Coeffs",false,1));
}

std::string  Mangling( cAddParamAFocal *) {return "B909ECE6CA728892FDBF";};


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

void  BinaryUnDumpFromFile(cCalibPerPose & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.KeyPose2Cal(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.KeyInitFromPose().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.KeyInitFromPose().ValForcedForUnUmp(),aFp);
        }
        else  anObj.KeyInitFromPose().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cCalibPerPose & anObj)
{
    BinaryDumpInFile(aFp,anObj.KeyPose2Cal());
    BinaryDumpInFile(aFp,anObj.KeyInitFromPose().IsInit());
    if (anObj.KeyInitFromPose().IsInit()) BinaryDumpInFile(aFp,anObj.KeyInitFromPose().Val());
}

cElXMLTree * ToXMLTree(const cCalibPerPose & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"CalibPerPose",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("KeyPose2Cal"),anObj.KeyPose2Cal())->ReTagThis("KeyPose2Cal"));
   if (anObj.KeyInitFromPose().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("KeyInitFromPose"),anObj.KeyInitFromPose().Val())->ReTagThis("KeyInitFromPose"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cCalibPerPose & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.KeyPose2Cal(),aTree->Get("KeyPose2Cal",1)); //tototo 

   xml_init(anObj.KeyInitFromPose(),aTree->Get("KeyInitFromPose",1)); //tototo 
}

std::string  Mangling( cCalibPerPose *) {return "90EF96A9C2F03D9FF83F";};


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


cTplValGesInit< bool > & cCalibrationCameraInc::AddDirCur()
{
   return mAddDirCur;
}

const cTplValGesInit< bool > & cCalibrationCameraInc::AddDirCur()const 
{
   return mAddDirCur;
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

void  BinaryUnDumpFromFile(cCalibrationCameraInc & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Name(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ConvCal().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ConvCal().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ConvCal().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Directory().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Directory().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Directory().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.AddDirCur().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.AddDirCur().ValForcedForUnUmp(),aFp);
        }
        else  anObj.AddDirCur().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.CalValueInit(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.DistortionAddInc().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.DistortionAddInc().ValForcedForUnUmp(),aFp);
        }
        else  anObj.DistortionAddInc().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.AddParamAFocal().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.AddParamAFocal().ValForcedForUnUmp(),aFp);
        }
        else  anObj.AddParamAFocal().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.RayMaxUtile().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.RayMaxUtile().ValForcedForUnUmp(),aFp);
        }
        else  anObj.RayMaxUtile().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.RayIsRelatifDiag().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.RayIsRelatifDiag().ValForcedForUnUmp(),aFp);
        }
        else  anObj.RayIsRelatifDiag().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.RayApplyOnlyFE().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.RayApplyOnlyFE().ValForcedForUnUmp(),aFp);
        }
        else  anObj.RayApplyOnlyFE().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.PropDiagUtile().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.PropDiagUtile().ValForcedForUnUmp(),aFp);
        }
        else  anObj.PropDiagUtile().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.CalibPerPose().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.CalibPerPose().ValForcedForUnUmp(),aFp);
        }
        else  anObj.CalibPerPose().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cCalibrationCameraInc & anObj)
{
    BinaryDumpInFile(aFp,anObj.Name());
    BinaryDumpInFile(aFp,anObj.ConvCal().IsInit());
    if (anObj.ConvCal().IsInit()) BinaryDumpInFile(aFp,anObj.ConvCal().Val());
    BinaryDumpInFile(aFp,anObj.Directory().IsInit());
    if (anObj.Directory().IsInit()) BinaryDumpInFile(aFp,anObj.Directory().Val());
    BinaryDumpInFile(aFp,anObj.AddDirCur().IsInit());
    if (anObj.AddDirCur().IsInit()) BinaryDumpInFile(aFp,anObj.AddDirCur().Val());
    BinaryDumpInFile(aFp,anObj.CalValueInit());
    BinaryDumpInFile(aFp,anObj.DistortionAddInc().IsInit());
    if (anObj.DistortionAddInc().IsInit()) BinaryDumpInFile(aFp,anObj.DistortionAddInc().Val());
    BinaryDumpInFile(aFp,anObj.AddParamAFocal().IsInit());
    if (anObj.AddParamAFocal().IsInit()) BinaryDumpInFile(aFp,anObj.AddParamAFocal().Val());
    BinaryDumpInFile(aFp,anObj.RayMaxUtile().IsInit());
    if (anObj.RayMaxUtile().IsInit()) BinaryDumpInFile(aFp,anObj.RayMaxUtile().Val());
    BinaryDumpInFile(aFp,anObj.RayIsRelatifDiag().IsInit());
    if (anObj.RayIsRelatifDiag().IsInit()) BinaryDumpInFile(aFp,anObj.RayIsRelatifDiag().Val());
    BinaryDumpInFile(aFp,anObj.RayApplyOnlyFE().IsInit());
    if (anObj.RayApplyOnlyFE().IsInit()) BinaryDumpInFile(aFp,anObj.RayApplyOnlyFE().Val());
    BinaryDumpInFile(aFp,anObj.PropDiagUtile().IsInit());
    if (anObj.PropDiagUtile().IsInit()) BinaryDumpInFile(aFp,anObj.PropDiagUtile().Val());
    BinaryDumpInFile(aFp,anObj.CalibPerPose().IsInit());
    if (anObj.CalibPerPose().IsInit()) BinaryDumpInFile(aFp,anObj.CalibPerPose().Val());
}

cElXMLTree * ToXMLTree(const cCalibrationCameraInc & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"CalibrationCameraInc",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Name"),anObj.Name())->ReTagThis("Name"));
   if (anObj.ConvCal().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ConvCal"),anObj.ConvCal().Val())->ReTagThis("ConvCal"));
   if (anObj.Directory().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Directory"),anObj.Directory().Val())->ReTagThis("Directory"));
   if (anObj.AddDirCur().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("AddDirCur"),anObj.AddDirCur().Val())->ReTagThis("AddDirCur"));
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
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cCalibrationCameraInc & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.Name(),aTree->Get("Name",1)); //tototo 

   xml_init(anObj.ConvCal(),aTree->Get("ConvCal",1),eConventionsOrientation(eConvApero_DistM2C)); //tototo 

   xml_init(anObj.Directory(),aTree->Get("Directory",1),std::string("")); //tototo 

   xml_init(anObj.AddDirCur(),aTree->Get("AddDirCur",1),bool(true)); //tototo 

   xml_init(anObj.CalValueInit(),aTree->Get("CalValueInit",1)); //tototo 

   xml_init(anObj.DistortionAddInc(),aTree->Get("DistortionAddInc",1)); //tototo 

   xml_init(anObj.AddParamAFocal(),aTree->Get("AddParamAFocal",1)); //tototo 

   xml_init(anObj.RayMaxUtile(),aTree->Get("RayMaxUtile",1),double(1e20)); //tototo 

   xml_init(anObj.RayIsRelatifDiag(),aTree->Get("RayIsRelatifDiag",1),bool(false)); //tototo 

   xml_init(anObj.RayApplyOnlyFE(),aTree->Get("RayApplyOnlyFE",1),bool(false)); //tototo 

   xml_init(anObj.PropDiagUtile(),aTree->Get("PropDiagUtile",1),double(1.0)); //tototo 

   xml_init(anObj.CalibPerPose(),aTree->Get("CalibPerPose",1)); //tototo 
}

std::string  Mangling( cCalibrationCameraInc *) {return "E2A53DD235D4AE8BFF3F";};


cTplValGesInit< cXml_SigmaRot > & cBlockGlobalBundle::SigmaV0()
{
   return mSigmaV0;
}

const cTplValGesInit< cXml_SigmaRot > & cBlockGlobalBundle::SigmaV0()const 
{
   return mSigmaV0;
}


cTplValGesInit< bool > & cBlockGlobalBundle::V0Stricte()
{
   return mV0Stricte;
}

const cTplValGesInit< bool > & cBlockGlobalBundle::V0Stricte()const 
{
   return mV0Stricte;
}


cTplValGesInit< double > & cBlockGlobalBundle::SigmaSimDist()
{
   return mSigmaSimDist;
}

const cTplValGesInit< double > & cBlockGlobalBundle::SigmaSimDist()const 
{
   return mSigmaSimDist;
}

void  BinaryUnDumpFromFile(cBlockGlobalBundle & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SigmaV0().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SigmaV0().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SigmaV0().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.V0Stricte().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.V0Stricte().ValForcedForUnUmp(),aFp);
        }
        else  anObj.V0Stricte().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SigmaSimDist().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SigmaSimDist().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SigmaSimDist().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cBlockGlobalBundle & anObj)
{
    BinaryDumpInFile(aFp,anObj.SigmaV0().IsInit());
    if (anObj.SigmaV0().IsInit()) BinaryDumpInFile(aFp,anObj.SigmaV0().Val());
    BinaryDumpInFile(aFp,anObj.V0Stricte().IsInit());
    if (anObj.V0Stricte().IsInit()) BinaryDumpInFile(aFp,anObj.V0Stricte().Val());
    BinaryDumpInFile(aFp,anObj.SigmaSimDist().IsInit());
    if (anObj.SigmaSimDist().IsInit()) BinaryDumpInFile(aFp,anObj.SigmaSimDist().Val());
}

cElXMLTree * ToXMLTree(const cBlockGlobalBundle & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"BlockGlobalBundle",eXMLBranche);
   if (anObj.SigmaV0().IsInit())
      aRes->AddFils(ToXMLTree(anObj.SigmaV0().Val())->ReTagThis("SigmaV0"));
   if (anObj.V0Stricte().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("V0Stricte"),anObj.V0Stricte().Val())->ReTagThis("V0Stricte"));
   if (anObj.SigmaSimDist().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SigmaSimDist"),anObj.SigmaSimDist().Val())->ReTagThis("SigmaSimDist"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cBlockGlobalBundle & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.SigmaV0(),aTree->Get("SigmaV0",1)); //tototo 

   xml_init(anObj.V0Stricte(),aTree->Get("V0Stricte",1)); //tototo 

   xml_init(anObj.SigmaSimDist(),aTree->Get("SigmaSimDist",1)); //tototo 
}

std::string  Mangling( cBlockGlobalBundle *) {return "3095591CE188A1F8FBBF";};


cTplValGesInit< cXml_SigmaRot > & cUseForBundle::SigmaV0()
{
   return BlockGlobalBundle().Val().SigmaV0();
}

const cTplValGesInit< cXml_SigmaRot > & cUseForBundle::SigmaV0()const 
{
   return BlockGlobalBundle().Val().SigmaV0();
}


cTplValGesInit< bool > & cUseForBundle::V0Stricte()
{
   return BlockGlobalBundle().Val().V0Stricte();
}

const cTplValGesInit< bool > & cUseForBundle::V0Stricte()const 
{
   return BlockGlobalBundle().Val().V0Stricte();
}


cTplValGesInit< double > & cUseForBundle::SigmaSimDist()
{
   return BlockGlobalBundle().Val().SigmaSimDist();
}

const cTplValGesInit< double > & cUseForBundle::SigmaSimDist()const 
{
   return BlockGlobalBundle().Val().SigmaSimDist();
}


cTplValGesInit< cBlockGlobalBundle > & cUseForBundle::BlockGlobalBundle()
{
   return mBlockGlobalBundle;
}

const cTplValGesInit< cBlockGlobalBundle > & cUseForBundle::BlockGlobalBundle()const 
{
   return mBlockGlobalBundle;
}


bool & cUseForBundle::RelTimeBundle()
{
   return mRelTimeBundle;
}

const bool & cUseForBundle::RelTimeBundle()const 
{
   return mRelTimeBundle;
}


cTplValGesInit< bool > & cUseForBundle::RelDistTimeBundle()
{
   return mRelDistTimeBundle;
}

const cTplValGesInit< bool > & cUseForBundle::RelDistTimeBundle()const 
{
   return mRelDistTimeBundle;
}


cTplValGesInit< bool > & cUseForBundle::GlobDistTimeBundle()
{
   return mGlobDistTimeBundle;
}

const cTplValGesInit< bool > & cUseForBundle::GlobDistTimeBundle()const 
{
   return mGlobDistTimeBundle;
}

void  BinaryUnDumpFromFile(cUseForBundle & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.BlockGlobalBundle().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.BlockGlobalBundle().ValForcedForUnUmp(),aFp);
        }
        else  anObj.BlockGlobalBundle().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.RelTimeBundle(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.RelDistTimeBundle().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.RelDistTimeBundle().ValForcedForUnUmp(),aFp);
        }
        else  anObj.RelDistTimeBundle().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.GlobDistTimeBundle().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.GlobDistTimeBundle().ValForcedForUnUmp(),aFp);
        }
        else  anObj.GlobDistTimeBundle().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cUseForBundle & anObj)
{
    BinaryDumpInFile(aFp,anObj.BlockGlobalBundle().IsInit());
    if (anObj.BlockGlobalBundle().IsInit()) BinaryDumpInFile(aFp,anObj.BlockGlobalBundle().Val());
    BinaryDumpInFile(aFp,anObj.RelTimeBundle());
    BinaryDumpInFile(aFp,anObj.RelDistTimeBundle().IsInit());
    if (anObj.RelDistTimeBundle().IsInit()) BinaryDumpInFile(aFp,anObj.RelDistTimeBundle().Val());
    BinaryDumpInFile(aFp,anObj.GlobDistTimeBundle().IsInit());
    if (anObj.GlobDistTimeBundle().IsInit()) BinaryDumpInFile(aFp,anObj.GlobDistTimeBundle().Val());
}

cElXMLTree * ToXMLTree(const cUseForBundle & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"UseForBundle",eXMLBranche);
   if (anObj.BlockGlobalBundle().IsInit())
      aRes->AddFils(ToXMLTree(anObj.BlockGlobalBundle().Val())->ReTagThis("BlockGlobalBundle"));
   aRes->AddFils(::ToXMLTree(std::string("RelTimeBundle"),anObj.RelTimeBundle())->ReTagThis("RelTimeBundle"));
   if (anObj.RelDistTimeBundle().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("RelDistTimeBundle"),anObj.RelDistTimeBundle().Val())->ReTagThis("RelDistTimeBundle"));
   if (anObj.GlobDistTimeBundle().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("GlobDistTimeBundle"),anObj.GlobDistTimeBundle().Val())->ReTagThis("GlobDistTimeBundle"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cUseForBundle & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.BlockGlobalBundle(),aTree->Get("BlockGlobalBundle",1)); //tototo 

   xml_init(anObj.RelTimeBundle(),aTree->Get("RelTimeBundle",1)); //tototo 

   xml_init(anObj.RelDistTimeBundle(),aTree->Get("RelDistTimeBundle",1),bool(false)); //tototo 

   xml_init(anObj.GlobDistTimeBundle(),aTree->Get("GlobDistTimeBundle",1),bool(false)); //tototo 
}

std::string  Mangling( cUseForBundle *) {return "CA50D8AF8E0D6D87FF3F";};


std::string & cBlockCamera::NameFile()
{
   return mNameFile;
}

const std::string & cBlockCamera::NameFile()const 
{
   return mNameFile;
}


cTplValGesInit< std::string > & cBlockCamera::Id()
{
   return mId;
}

const cTplValGesInit< std::string > & cBlockCamera::Id()const 
{
   return mId;
}


cTplValGesInit< cXml_SigmaRot > & cBlockCamera::SigmaV0()
{
   return UseForBundle().Val().BlockGlobalBundle().Val().SigmaV0();
}

const cTplValGesInit< cXml_SigmaRot > & cBlockCamera::SigmaV0()const 
{
   return UseForBundle().Val().BlockGlobalBundle().Val().SigmaV0();
}


cTplValGesInit< bool > & cBlockCamera::V0Stricte()
{
   return UseForBundle().Val().BlockGlobalBundle().Val().V0Stricte();
}

const cTplValGesInit< bool > & cBlockCamera::V0Stricte()const 
{
   return UseForBundle().Val().BlockGlobalBundle().Val().V0Stricte();
}


cTplValGesInit< double > & cBlockCamera::SigmaSimDist()
{
   return UseForBundle().Val().BlockGlobalBundle().Val().SigmaSimDist();
}

const cTplValGesInit< double > & cBlockCamera::SigmaSimDist()const 
{
   return UseForBundle().Val().BlockGlobalBundle().Val().SigmaSimDist();
}


cTplValGesInit< cBlockGlobalBundle > & cBlockCamera::BlockGlobalBundle()
{
   return UseForBundle().Val().BlockGlobalBundle();
}

const cTplValGesInit< cBlockGlobalBundle > & cBlockCamera::BlockGlobalBundle()const 
{
   return UseForBundle().Val().BlockGlobalBundle();
}


bool & cBlockCamera::RelTimeBundle()
{
   return UseForBundle().Val().RelTimeBundle();
}

const bool & cBlockCamera::RelTimeBundle()const 
{
   return UseForBundle().Val().RelTimeBundle();
}


cTplValGesInit< bool > & cBlockCamera::RelDistTimeBundle()
{
   return UseForBundle().Val().RelDistTimeBundle();
}

const cTplValGesInit< bool > & cBlockCamera::RelDistTimeBundle()const 
{
   return UseForBundle().Val().RelDistTimeBundle();
}


cTplValGesInit< bool > & cBlockCamera::GlobDistTimeBundle()
{
   return UseForBundle().Val().GlobDistTimeBundle();
}

const cTplValGesInit< bool > & cBlockCamera::GlobDistTimeBundle()const 
{
   return UseForBundle().Val().GlobDistTimeBundle();
}


cTplValGesInit< cUseForBundle > & cBlockCamera::UseForBundle()
{
   return mUseForBundle;
}

const cTplValGesInit< cUseForBundle > & cBlockCamera::UseForBundle()const 
{
   return mUseForBundle;
}

void  BinaryUnDumpFromFile(cBlockCamera & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.NameFile(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Id().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Id().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Id().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.UseForBundle().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.UseForBundle().ValForcedForUnUmp(),aFp);
        }
        else  anObj.UseForBundle().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cBlockCamera & anObj)
{
    BinaryDumpInFile(aFp,anObj.NameFile());
    BinaryDumpInFile(aFp,anObj.Id().IsInit());
    if (anObj.Id().IsInit()) BinaryDumpInFile(aFp,anObj.Id().Val());
    BinaryDumpInFile(aFp,anObj.UseForBundle().IsInit());
    if (anObj.UseForBundle().IsInit()) BinaryDumpInFile(aFp,anObj.UseForBundle().Val());
}

cElXMLTree * ToXMLTree(const cBlockCamera & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"BlockCamera",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("NameFile"),anObj.NameFile())->ReTagThis("NameFile"));
   if (anObj.Id().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Id"),anObj.Id().Val())->ReTagThis("Id"));
   if (anObj.UseForBundle().IsInit())
      aRes->AddFils(ToXMLTree(anObj.UseForBundle().Val())->ReTagThis("UseForBundle"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cBlockCamera & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.NameFile(),aTree->Get("NameFile",1)); //tototo 

   xml_init(anObj.Id(),aTree->Get("Id",1)); //tototo 

   xml_init(anObj.UseForBundle(),aTree->Get("UseForBundle",1)); //tototo 
}

std::string  Mangling( cBlockCamera *) {return "47790627742D6E88FE3F";};


cElRegex_Ptr & cCamGenInc::PatterName()
{
   return mPatterName;
}

const cElRegex_Ptr & cCamGenInc::PatterName()const 
{
   return mPatterName;
}


std::string & cCamGenInc::Orient()
{
   return mOrient;
}

const std::string & cCamGenInc::Orient()const 
{
   return mOrient;
}


cTplValGesInit< bool > & cCamGenInc::ErrorWhenEmpytPat()
{
   return mErrorWhenEmpytPat;
}

const cTplValGesInit< bool > & cCamGenInc::ErrorWhenEmpytPat()const 
{
   return mErrorWhenEmpytPat;
}


cTplValGesInit< bool > & cCamGenInc::ErrorWhenNoFileOrient()
{
   return mErrorWhenNoFileOrient;
}

const cTplValGesInit< bool > & cCamGenInc::ErrorWhenNoFileOrient()const 
{
   return mErrorWhenNoFileOrient;
}

void  BinaryUnDumpFromFile(cCamGenInc & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.PatterName(),aFp);
    BinaryUnDumpFromFile(anObj.Orient(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ErrorWhenEmpytPat().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ErrorWhenEmpytPat().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ErrorWhenEmpytPat().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ErrorWhenNoFileOrient().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ErrorWhenNoFileOrient().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ErrorWhenNoFileOrient().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cCamGenInc & anObj)
{
    BinaryDumpInFile(aFp,anObj.PatterName());
    BinaryDumpInFile(aFp,anObj.Orient());
    BinaryDumpInFile(aFp,anObj.ErrorWhenEmpytPat().IsInit());
    if (anObj.ErrorWhenEmpytPat().IsInit()) BinaryDumpInFile(aFp,anObj.ErrorWhenEmpytPat().Val());
    BinaryDumpInFile(aFp,anObj.ErrorWhenNoFileOrient().IsInit());
    if (anObj.ErrorWhenNoFileOrient().IsInit()) BinaryDumpInFile(aFp,anObj.ErrorWhenNoFileOrient().Val());
}

cElXMLTree * ToXMLTree(const cCamGenInc & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"CamGenInc",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("PatterName"),anObj.PatterName())->ReTagThis("PatterName"));
   aRes->AddFils(::ToXMLTree(std::string("Orient"),anObj.Orient())->ReTagThis("Orient"));
   if (anObj.ErrorWhenEmpytPat().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ErrorWhenEmpytPat"),anObj.ErrorWhenEmpytPat().Val())->ReTagThis("ErrorWhenEmpytPat"));
   if (anObj.ErrorWhenNoFileOrient().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ErrorWhenNoFileOrient"),anObj.ErrorWhenNoFileOrient().Val())->ReTagThis("ErrorWhenNoFileOrient"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cCamGenInc & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.PatterName(),aTree->Get("PatterName",1)); //tototo 

   xml_init(anObj.Orient(),aTree->Get("Orient",1)); //tototo 

   xml_init(anObj.ErrorWhenEmpytPat(),aTree->Get("ErrorWhenEmpytPat",1),bool(true)); //tototo 

   xml_init(anObj.ErrorWhenNoFileOrient(),aTree->Get("ErrorWhenNoFileOrient",1),bool(true)); //tototo 
}

std::string  Mangling( cCamGenInc *) {return "8A8C5C1816652EFDFD3F";};


cTplValGesInit< std::string > & cMEP_SPEC_MST::MSTBlockRigid()
{
   return mMSTBlockRigid;
}

const cTplValGesInit< std::string > & cMEP_SPEC_MST::MSTBlockRigid()const 
{
   return mMSTBlockRigid;
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

void  BinaryUnDumpFromFile(cMEP_SPEC_MST & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.MSTBlockRigid().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.MSTBlockRigid().ValForcedForUnUmp(),aFp);
        }
        else  anObj.MSTBlockRigid().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Show().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Show().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Show().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.MinNbPtsInit().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.MinNbPtsInit().ValForcedForUnUmp(),aFp);
        }
        else  anObj.MinNbPtsInit().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ExpDist().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ExpDist().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ExpDist().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ExpNb().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ExpNb().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ExpNb().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.MontageOnInit().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.MontageOnInit().ValForcedForUnUmp(),aFp);
        }
        else  anObj.MontageOnInit().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NbInitMinBeforeUnconnect().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NbInitMinBeforeUnconnect().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NbInitMinBeforeUnconnect().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cMEP_SPEC_MST & anObj)
{
    BinaryDumpInFile(aFp,anObj.MSTBlockRigid().IsInit());
    if (anObj.MSTBlockRigid().IsInit()) BinaryDumpInFile(aFp,anObj.MSTBlockRigid().Val());
    BinaryDumpInFile(aFp,anObj.Show().IsInit());
    if (anObj.Show().IsInit()) BinaryDumpInFile(aFp,anObj.Show().Val());
    BinaryDumpInFile(aFp,anObj.MinNbPtsInit().IsInit());
    if (anObj.MinNbPtsInit().IsInit()) BinaryDumpInFile(aFp,anObj.MinNbPtsInit().Val());
    BinaryDumpInFile(aFp,anObj.ExpDist().IsInit());
    if (anObj.ExpDist().IsInit()) BinaryDumpInFile(aFp,anObj.ExpDist().Val());
    BinaryDumpInFile(aFp,anObj.ExpNb().IsInit());
    if (anObj.ExpNb().IsInit()) BinaryDumpInFile(aFp,anObj.ExpNb().Val());
    BinaryDumpInFile(aFp,anObj.MontageOnInit().IsInit());
    if (anObj.MontageOnInit().IsInit()) BinaryDumpInFile(aFp,anObj.MontageOnInit().Val());
    BinaryDumpInFile(aFp,anObj.NbInitMinBeforeUnconnect().IsInit());
    if (anObj.NbInitMinBeforeUnconnect().IsInit()) BinaryDumpInFile(aFp,anObj.NbInitMinBeforeUnconnect().Val());
}

cElXMLTree * ToXMLTree(const cMEP_SPEC_MST & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"MEP_SPEC_MST",eXMLBranche);
   if (anObj.MSTBlockRigid().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("MSTBlockRigid"),anObj.MSTBlockRigid().Val())->ReTagThis("MSTBlockRigid"));
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
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cMEP_SPEC_MST & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.MSTBlockRigid(),aTree->Get("MSTBlockRigid",1)); //tototo 

   xml_init(anObj.Show(),aTree->Get("Show",1),bool(false)); //tototo 

   xml_init(anObj.MinNbPtsInit(),aTree->Get("MinNbPtsInit",1),int(8)); //tototo 

   xml_init(anObj.ExpDist(),aTree->Get("ExpDist",1),double(2.0)); //tototo 

   xml_init(anObj.ExpNb(),aTree->Get("ExpNb",1),double(1.0)); //tototo 

   xml_init(anObj.MontageOnInit(),aTree->Get("MontageOnInit",1),bool(false)); //tototo 

   xml_init(anObj.NbInitMinBeforeUnconnect(),aTree->Get("NbInitMinBeforeUnconnect",1),int(10000000)); //tototo 
}

std::string  Mangling( cMEP_SPEC_MST *) {return "C22C12BD9BD3CB8CFDBF";};


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

void  BinaryUnDumpFromFile(cApplyOAI & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Cstr(),aFp);
    BinaryUnDumpFromFile(anObj.PatternApply(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cApplyOAI & anObj)
{
    BinaryDumpInFile(aFp,anObj.Cstr());
    BinaryDumpInFile(aFp,anObj.PatternApply());
}

cElXMLTree * ToXMLTree(const cApplyOAI & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ApplyOAI",eXMLBranche);
   aRes->AddFils(ToXMLTree(std::string("Cstr"),anObj.Cstr())->ReTagThis("Cstr"));
   aRes->AddFils(::ToXMLTree(std::string("PatternApply"),anObj.PatternApply())->ReTagThis("PatternApply"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cApplyOAI & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.Cstr(),aTree->Get("Cstr",1)); //tototo 

   xml_init(anObj.PatternApply(),aTree->Get("PatternApply",1)); //tototo 
}

std::string  Mangling( cApplyOAI *) {return "97D73F5D3DFBBD94FE3F";};


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

void  BinaryUnDumpFromFile(cOptimizeAfterInit & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.ParamOptim(),aFp);
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cApplyOAI aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.ApplyOAI().push_back(aVal);
        }
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cOptimizeAfterInit & anObj)
{
    BinaryDumpInFile(aFp,anObj.ParamOptim());
    BinaryDumpInFile(aFp,(int)anObj.ApplyOAI().size());
    for(  std::list< cApplyOAI >::const_iterator iT=anObj.ApplyOAI().begin();
         iT!=anObj.ApplyOAI().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
}

cElXMLTree * ToXMLTree(const cOptimizeAfterInit & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"OptimizeAfterInit",eXMLBranche);
   aRes->AddFils(ToXMLTree(anObj.ParamOptim())->ReTagThis("ParamOptim"));
  for
  (       std::list< cApplyOAI >::const_iterator it=anObj.ApplyOAI().begin();
      it !=anObj.ApplyOAI().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("ApplyOAI"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cOptimizeAfterInit & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.ParamOptim(),aTree->Get("ParamOptim",1)); //tototo 

   xml_init(anObj.ApplyOAI(),aTree->GetAll("ApplyOAI",false,1));
}

std::string  Mangling( cOptimizeAfterInit *) {return "1E84110660378D81FF3F";};


std::string & cCalcNameOnExistingTag::KeyCalcFileOriExt()
{
   return mKeyCalcFileOriExt;
}

const std::string & cCalcNameOnExistingTag::KeyCalcFileOriExt()const 
{
   return mKeyCalcFileOriExt;
}


std::string & cCalcNameOnExistingTag::KeyCalcName()
{
   return mKeyCalcName;
}

const std::string & cCalcNameOnExistingTag::KeyCalcName()const 
{
   return mKeyCalcName;
}


std::string & cCalcNameOnExistingTag::TagExist()
{
   return mTagExist;
}

const std::string & cCalcNameOnExistingTag::TagExist()const 
{
   return mTagExist;
}


std::string & cCalcNameOnExistingTag::TagNotExist()
{
   return mTagNotExist;
}

const std::string & cCalcNameOnExistingTag::TagNotExist()const 
{
   return mTagNotExist;
}


cTplValGesInit< bool > & cCalcNameOnExistingTag::ExigCohTags()
{
   return mExigCohTags;
}

const cTplValGesInit< bool > & cCalcNameOnExistingTag::ExigCohTags()const 
{
   return mExigCohTags;
}

void  BinaryUnDumpFromFile(cCalcNameOnExistingTag & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.KeyCalcFileOriExt(),aFp);
    BinaryUnDumpFromFile(anObj.KeyCalcName(),aFp);
    BinaryUnDumpFromFile(anObj.TagExist(),aFp);
    BinaryUnDumpFromFile(anObj.TagNotExist(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ExigCohTags().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ExigCohTags().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ExigCohTags().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cCalcNameOnExistingTag & anObj)
{
    BinaryDumpInFile(aFp,anObj.KeyCalcFileOriExt());
    BinaryDumpInFile(aFp,anObj.KeyCalcName());
    BinaryDumpInFile(aFp,anObj.TagExist());
    BinaryDumpInFile(aFp,anObj.TagNotExist());
    BinaryDumpInFile(aFp,anObj.ExigCohTags().IsInit());
    if (anObj.ExigCohTags().IsInit()) BinaryDumpInFile(aFp,anObj.ExigCohTags().Val());
}

cElXMLTree * ToXMLTree(const cCalcNameOnExistingTag & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"CalcNameOnExistingTag",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("KeyCalcFileOriExt"),anObj.KeyCalcFileOriExt())->ReTagThis("KeyCalcFileOriExt"));
   aRes->AddFils(::ToXMLTree(std::string("KeyCalcName"),anObj.KeyCalcName())->ReTagThis("KeyCalcName"));
   aRes->AddFils(::ToXMLTree(std::string("TagExist"),anObj.TagExist())->ReTagThis("TagExist"));
   aRes->AddFils(::ToXMLTree(std::string("TagNotExist"),anObj.TagNotExist())->ReTagThis("TagNotExist"));
   if (anObj.ExigCohTags().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ExigCohTags"),anObj.ExigCohTags().Val())->ReTagThis("ExigCohTags"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cCalcNameOnExistingTag & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.KeyCalcFileOriExt(),aTree->Get("KeyCalcFileOriExt",1)); //tototo 

   xml_init(anObj.KeyCalcName(),aTree->Get("KeyCalcName",1)); //tototo 

   xml_init(anObj.TagExist(),aTree->Get("TagExist",1)); //tototo 

   xml_init(anObj.TagNotExist(),aTree->Get("TagNotExist",1)); //tototo 

   xml_init(anObj.ExigCohTags(),aTree->Get("ExigCohTags",1),bool(true)); //tototo 
}

std::string  Mangling( cCalcNameOnExistingTag *) {return "DE7E3593F95CBDA1FF3F";};


std::string & cCalcNameCalibAux::KeyCalcFileOriExt()
{
   return CalcNameOnExistingTag().Val().KeyCalcFileOriExt();
}

const std::string & cCalcNameCalibAux::KeyCalcFileOriExt()const 
{
   return CalcNameOnExistingTag().Val().KeyCalcFileOriExt();
}


std::string & cCalcNameCalibAux::KeyCalcName()
{
   return CalcNameOnExistingTag().Val().KeyCalcName();
}

const std::string & cCalcNameCalibAux::KeyCalcName()const 
{
   return CalcNameOnExistingTag().Val().KeyCalcName();
}


std::string & cCalcNameCalibAux::TagExist()
{
   return CalcNameOnExistingTag().Val().TagExist();
}

const std::string & cCalcNameCalibAux::TagExist()const 
{
   return CalcNameOnExistingTag().Val().TagExist();
}


std::string & cCalcNameCalibAux::TagNotExist()
{
   return CalcNameOnExistingTag().Val().TagNotExist();
}

const std::string & cCalcNameCalibAux::TagNotExist()const 
{
   return CalcNameOnExistingTag().Val().TagNotExist();
}


cTplValGesInit< bool > & cCalcNameCalibAux::ExigCohTags()
{
   return CalcNameOnExistingTag().Val().ExigCohTags();
}

const cTplValGesInit< bool > & cCalcNameCalibAux::ExigCohTags()const 
{
   return CalcNameOnExistingTag().Val().ExigCohTags();
}


cTplValGesInit< cCalcNameOnExistingTag > & cCalcNameCalibAux::CalcNameOnExistingTag()
{
   return mCalcNameOnExistingTag;
}

const cTplValGesInit< cCalcNameOnExistingTag > & cCalcNameCalibAux::CalcNameOnExistingTag()const 
{
   return mCalcNameOnExistingTag;
}


cTplValGesInit< std::string > & cCalcNameCalibAux::KeyCalcNameDef()
{
   return mKeyCalcNameDef;
}

const cTplValGesInit< std::string > & cCalcNameCalibAux::KeyCalcNameDef()const 
{
   return mKeyCalcNameDef;
}

void  BinaryUnDumpFromFile(cCalcNameCalibAux & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.CalcNameOnExistingTag().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.CalcNameOnExistingTag().ValForcedForUnUmp(),aFp);
        }
        else  anObj.CalcNameOnExistingTag().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.KeyCalcNameDef().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.KeyCalcNameDef().ValForcedForUnUmp(),aFp);
        }
        else  anObj.KeyCalcNameDef().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cCalcNameCalibAux & anObj)
{
    BinaryDumpInFile(aFp,anObj.CalcNameOnExistingTag().IsInit());
    if (anObj.CalcNameOnExistingTag().IsInit()) BinaryDumpInFile(aFp,anObj.CalcNameOnExistingTag().Val());
    BinaryDumpInFile(aFp,anObj.KeyCalcNameDef().IsInit());
    if (anObj.KeyCalcNameDef().IsInit()) BinaryDumpInFile(aFp,anObj.KeyCalcNameDef().Val());
}

cElXMLTree * ToXMLTree(const cCalcNameCalibAux & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"CalcNameCalibAux",eXMLBranche);
   if (anObj.CalcNameOnExistingTag().IsInit())
      aRes->AddFils(ToXMLTree(anObj.CalcNameOnExistingTag().Val())->ReTagThis("CalcNameOnExistingTag"));
   if (anObj.KeyCalcNameDef().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("KeyCalcNameDef"),anObj.KeyCalcNameDef().Val())->ReTagThis("KeyCalcNameDef"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cCalcNameCalibAux & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.CalcNameOnExistingTag(),aTree->Get("CalcNameOnExistingTag",1)); //tototo 

   xml_init(anObj.KeyCalcNameDef(),aTree->Get("KeyCalcNameDef",1)); //tototo 
}

std::string  Mangling( cCalcNameCalibAux *) {return "18AEF6FF3EA24095FA3F";};


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

void  BinaryUnDumpFromFile(cPosFromBDAppuis & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Id(),aFp);
    BinaryUnDumpFromFile(anObj.NbTestRansac(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.DirApprox().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.DirApprox().ValForcedForUnUmp(),aFp);
        }
        else  anObj.DirApprox().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cPosFromBDAppuis & anObj)
{
    BinaryDumpInFile(aFp,anObj.Id());
    BinaryDumpInFile(aFp,anObj.NbTestRansac());
    BinaryDumpInFile(aFp,anObj.DirApprox().IsInit());
    if (anObj.DirApprox().IsInit()) BinaryDumpInFile(aFp,anObj.DirApprox().Val());
}

cElXMLTree * ToXMLTree(const cPosFromBDAppuis & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"PosFromBDAppuis",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Id"),anObj.Id())->ReTagThis("Id"));
   aRes->AddFils(::ToXMLTree(std::string("NbTestRansac"),anObj.NbTestRansac())->ReTagThis("NbTestRansac"));
   if (anObj.DirApprox().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DirApprox"),anObj.DirApprox().Val())->ReTagThis("DirApprox"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cPosFromBDAppuis & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.Id(),aTree->Get("Id",1)); //tototo 

   xml_init(anObj.NbTestRansac(),aTree->Get("NbTestRansac",1)); //tototo 

   xml_init(anObj.DirApprox(),aTree->Get("DirApprox",1)); //tototo 
}

std::string  Mangling( cPosFromBDAppuis *) {return "115A0200677AA1B5FE3F";};


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

void  BinaryUnDumpFromFile(cLiaisonsInit & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.OnZonePlane().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.OnZonePlane().ValForcedForUnUmp(),aFp);
        }
        else  anObj.OnZonePlane().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.TestSolPlane().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.TestSolPlane().ValForcedForUnUmp(),aFp);
        }
        else  anObj.TestSolPlane().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NbRansacSolAppui().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NbRansacSolAppui().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NbRansacSolAppui().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.InitOrientPure().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.InitOrientPure().ValForcedForUnUmp(),aFp);
        }
        else  anObj.InitOrientPure().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NbPtsRansacOrPure().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NbPtsRansacOrPure().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NbPtsRansacOrPure().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NbTestRansacOrPure().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NbTestRansacOrPure().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NbTestRansacOrPure().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NbMinPtsRanAp().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NbMinPtsRanAp().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NbMinPtsRanAp().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NbMaxPtsRanAp().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NbMaxPtsRanAp().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NbMaxPtsRanAp().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.PropMinPtsMult().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.PropMinPtsMult().ValForcedForUnUmp(),aFp);
        }
        else  anObj.PropMinPtsMult().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.NameCam(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NameCamIsKeyCalc().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NameCamIsKeyCalc().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NameCamIsKeyCalc().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.KeyCalcIsIDir().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.KeyCalcIsIDir().ValForcedForUnUmp(),aFp);
        }
        else  anObj.KeyCalcIsIDir().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.IdBD(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ProfSceneCouple().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ProfSceneCouple().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ProfSceneCouple().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.L2EstimPlan().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.L2EstimPlan().ValForcedForUnUmp(),aFp);
        }
        else  anObj.L2EstimPlan().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.LongueurBase().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.LongueurBase().ValForcedForUnUmp(),aFp);
        }
        else  anObj.LongueurBase().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cLiaisonsInit & anObj)
{
    BinaryDumpInFile(aFp,anObj.OnZonePlane().IsInit());
    if (anObj.OnZonePlane().IsInit()) BinaryDumpInFile(aFp,anObj.OnZonePlane().Val());
    BinaryDumpInFile(aFp,anObj.TestSolPlane().IsInit());
    if (anObj.TestSolPlane().IsInit()) BinaryDumpInFile(aFp,anObj.TestSolPlane().Val());
    BinaryDumpInFile(aFp,anObj.NbRansacSolAppui().IsInit());
    if (anObj.NbRansacSolAppui().IsInit()) BinaryDumpInFile(aFp,anObj.NbRansacSolAppui().Val());
    BinaryDumpInFile(aFp,anObj.InitOrientPure().IsInit());
    if (anObj.InitOrientPure().IsInit()) BinaryDumpInFile(aFp,anObj.InitOrientPure().Val());
    BinaryDumpInFile(aFp,anObj.NbPtsRansacOrPure().IsInit());
    if (anObj.NbPtsRansacOrPure().IsInit()) BinaryDumpInFile(aFp,anObj.NbPtsRansacOrPure().Val());
    BinaryDumpInFile(aFp,anObj.NbTestRansacOrPure().IsInit());
    if (anObj.NbTestRansacOrPure().IsInit()) BinaryDumpInFile(aFp,anObj.NbTestRansacOrPure().Val());
    BinaryDumpInFile(aFp,anObj.NbMinPtsRanAp().IsInit());
    if (anObj.NbMinPtsRanAp().IsInit()) BinaryDumpInFile(aFp,anObj.NbMinPtsRanAp().Val());
    BinaryDumpInFile(aFp,anObj.NbMaxPtsRanAp().IsInit());
    if (anObj.NbMaxPtsRanAp().IsInit()) BinaryDumpInFile(aFp,anObj.NbMaxPtsRanAp().Val());
    BinaryDumpInFile(aFp,anObj.PropMinPtsMult().IsInit());
    if (anObj.PropMinPtsMult().IsInit()) BinaryDumpInFile(aFp,anObj.PropMinPtsMult().Val());
    BinaryDumpInFile(aFp,anObj.NameCam());
    BinaryDumpInFile(aFp,anObj.NameCamIsKeyCalc().IsInit());
    if (anObj.NameCamIsKeyCalc().IsInit()) BinaryDumpInFile(aFp,anObj.NameCamIsKeyCalc().Val());
    BinaryDumpInFile(aFp,anObj.KeyCalcIsIDir().IsInit());
    if (anObj.KeyCalcIsIDir().IsInit()) BinaryDumpInFile(aFp,anObj.KeyCalcIsIDir().Val());
    BinaryDumpInFile(aFp,anObj.IdBD());
    BinaryDumpInFile(aFp,anObj.ProfSceneCouple().IsInit());
    if (anObj.ProfSceneCouple().IsInit()) BinaryDumpInFile(aFp,anObj.ProfSceneCouple().Val());
    BinaryDumpInFile(aFp,anObj.L2EstimPlan().IsInit());
    if (anObj.L2EstimPlan().IsInit()) BinaryDumpInFile(aFp,anObj.L2EstimPlan().Val());
    BinaryDumpInFile(aFp,anObj.LongueurBase().IsInit());
    if (anObj.LongueurBase().IsInit()) BinaryDumpInFile(aFp,anObj.LongueurBase().Val());
}

cElXMLTree * ToXMLTree(const cLiaisonsInit & anObj)
{
  XMLPushContext(anObj.mGXml);
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
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cLiaisonsInit & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

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

std::string  Mangling( cLiaisonsInit *) {return "10F1A114809128DEFBBF";};


std::vector< cLiaisonsInit > & cPoseFromLiaisons::LiaisonsInit()
{
   return mLiaisonsInit;
}

const std::vector< cLiaisonsInit > & cPoseFromLiaisons::LiaisonsInit()const 
{
   return mLiaisonsInit;
}

void  BinaryUnDumpFromFile(cPoseFromLiaisons & anObj,ELISE_fp & aFp)
{
   { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cLiaisonsInit aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.LiaisonsInit().push_back(aVal);
        }
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cPoseFromLiaisons & anObj)
{
    BinaryDumpInFile(aFp,(int)anObj.LiaisonsInit().size());
    for(  std::vector< cLiaisonsInit >::const_iterator iT=anObj.LiaisonsInit().begin();
         iT!=anObj.LiaisonsInit().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
}

cElXMLTree * ToXMLTree(const cPoseFromLiaisons & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"PoseFromLiaisons",eXMLBranche);
  for
  (       std::vector< cLiaisonsInit >::const_iterator it=anObj.LiaisonsInit().begin();
      it !=anObj.LiaisonsInit().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("LiaisonsInit"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cPoseFromLiaisons & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.LiaisonsInit(),aTree->GetAll("LiaisonsInit",false,1));
}

std::string  Mangling( cPoseFromLiaisons *) {return "AC5BE1501BBBFADAFE3F";};


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

void  BinaryUnDumpFromFile(cMesurePIFRP & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Ap1(),aFp);
    BinaryUnDumpFromFile(anObj.Ap2(),aFp);
    BinaryUnDumpFromFile(anObj.Ap3(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cMesurePIFRP & anObj)
{
    BinaryDumpInFile(aFp,anObj.Ap1());
    BinaryDumpInFile(aFp,anObj.Ap2());
    BinaryDumpInFile(aFp,anObj.Ap3());
}

cElXMLTree * ToXMLTree(const cMesurePIFRP & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"MesurePIFRP",eXMLBranche);
   aRes->AddFils(ToXMLTree(anObj.Ap1())->ReTagThis("Ap1"));
   aRes->AddFils(ToXMLTree(anObj.Ap2())->ReTagThis("Ap2"));
   aRes->AddFils(ToXMLTree(anObj.Ap3())->ReTagThis("Ap3"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cMesurePIFRP & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.Ap1(),aTree->Get("Ap1",1)); //tototo 

   xml_init(anObj.Ap2(),aTree->Get("Ap2",1)); //tototo 

   xml_init(anObj.Ap3(),aTree->Get("Ap3",1)); //tototo 
}

std::string  Mangling( cMesurePIFRP *) {return "26D98BFB6C6CD1ADFE3F";};


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

void  BinaryUnDumpFromFile(cInitPIFRP & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.MesurePIFRP().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.MesurePIFRP().ValForcedForUnUmp(),aFp);
        }
        else  anObj.MesurePIFRP().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.DirPlan().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.DirPlan().ValForcedForUnUmp(),aFp);
        }
        else  anObj.DirPlan().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cInitPIFRP & anObj)
{
    BinaryDumpInFile(aFp,anObj.MesurePIFRP().IsInit());
    if (anObj.MesurePIFRP().IsInit()) BinaryDumpInFile(aFp,anObj.MesurePIFRP().Val());
    BinaryDumpInFile(aFp,anObj.DirPlan().IsInit());
    if (anObj.DirPlan().IsInit()) BinaryDumpInFile(aFp,anObj.DirPlan().Val());
}

cElXMLTree * ToXMLTree(const cInitPIFRP & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"InitPIFRP",eXMLBranche);
   if (anObj.MesurePIFRP().IsInit())
      aRes->AddFils(ToXMLTree(anObj.MesurePIFRP().Val())->ReTagThis("MesurePIFRP"));
   if (anObj.DirPlan().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DirPlan"),anObj.DirPlan().Val())->ReTagThis("DirPlan"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cInitPIFRP & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.MesurePIFRP(),aTree->Get("MesurePIFRP",1)); //tototo 

   xml_init(anObj.DirPlan(),aTree->Get("DirPlan",1)); //tototo 
}

std::string  Mangling( cInitPIFRP *) {return "67DE4CD1511638C4FE3F";};


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

void  BinaryUnDumpFromFile(cPoseInitFromReperePlan & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.OnZonePlane(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.L2EstimPlan().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.L2EstimPlan().ValForcedForUnUmp(),aFp);
        }
        else  anObj.L2EstimPlan().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.IdBD(),aFp);
    BinaryUnDumpFromFile(anObj.NameCam(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.DEuclidPlan().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.DEuclidPlan().ValForcedForUnUmp(),aFp);
        }
        else  anObj.DEuclidPlan().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.InitPIFRP(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cPoseInitFromReperePlan & anObj)
{
    BinaryDumpInFile(aFp,anObj.OnZonePlane());
    BinaryDumpInFile(aFp,anObj.L2EstimPlan().IsInit());
    if (anObj.L2EstimPlan().IsInit()) BinaryDumpInFile(aFp,anObj.L2EstimPlan().Val());
    BinaryDumpInFile(aFp,anObj.IdBD());
    BinaryDumpInFile(aFp,anObj.NameCam());
    BinaryDumpInFile(aFp,anObj.DEuclidPlan().IsInit());
    if (anObj.DEuclidPlan().IsInit()) BinaryDumpInFile(aFp,anObj.DEuclidPlan().Val());
    BinaryDumpInFile(aFp,anObj.InitPIFRP());
}

cElXMLTree * ToXMLTree(const cPoseInitFromReperePlan & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"PoseInitFromReperePlan",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("OnZonePlane"),anObj.OnZonePlane())->ReTagThis("OnZonePlane"));
   if (anObj.L2EstimPlan().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("L2EstimPlan"),anObj.L2EstimPlan().Val())->ReTagThis("L2EstimPlan"));
   aRes->AddFils(::ToXMLTree(std::string("IdBD"),anObj.IdBD())->ReTagThis("IdBD"));
   aRes->AddFils(::ToXMLTree(std::string("NameCam"),anObj.NameCam())->ReTagThis("NameCam"));
   if (anObj.DEuclidPlan().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DEuclidPlan"),anObj.DEuclidPlan().Val())->ReTagThis("DEuclidPlan"));
   aRes->AddFils(ToXMLTree(anObj.InitPIFRP())->ReTagThis("InitPIFRP"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cPoseInitFromReperePlan & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.OnZonePlane(),aTree->Get("OnZonePlane",1)); //tototo 

   xml_init(anObj.L2EstimPlan(),aTree->Get("L2EstimPlan",1),bool(true)); //tototo 

   xml_init(anObj.IdBD(),aTree->Get("IdBD",1)); //tototo 

   xml_init(anObj.NameCam(),aTree->Get("NameCam",1)); //tototo 

   xml_init(anObj.DEuclidPlan(),aTree->Get("DEuclidPlan",1)); //tototo 

   xml_init(anObj.InitPIFRP(),aTree->Get("InitPIFRP",1)); //tototo 
}

std::string  Mangling( cPoseInitFromReperePlan *) {return "2C704EF1CF14BF91FE3F";};


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


cTplValGesInit< std::string > & cPosValueInit::PosFromBlockRigid()
{
   return mPosFromBlockRigid;
}

const cTplValGesInit< std::string > & cPosValueInit::PosFromBlockRigid()const 
{
   return mPosFromBlockRigid;
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

void  BinaryUnDumpFromFile(cPosValueInit & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.PosId().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.PosId().ValForcedForUnUmp(),aFp);
        }
        else  anObj.PosId().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.PosFromBDOrient().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.PosFromBDOrient().ValForcedForUnUmp(),aFp);
        }
        else  anObj.PosFromBDOrient().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.PosFromBlockRigid().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.PosFromBlockRigid().ValForcedForUnUmp(),aFp);
        }
        else  anObj.PosFromBlockRigid().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.PosFromBDAppuis().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.PosFromBDAppuis().ValForcedForUnUmp(),aFp);
        }
        else  anObj.PosFromBDAppuis().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.PoseFromLiaisons().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.PoseFromLiaisons().ValForcedForUnUmp(),aFp);
        }
        else  anObj.PoseFromLiaisons().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.PoseInitFromReperePlan().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.PoseInitFromReperePlan().ValForcedForUnUmp(),aFp);
        }
        else  anObj.PoseInitFromReperePlan().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cPosValueInit & anObj)
{
    BinaryDumpInFile(aFp,anObj.PosId().IsInit());
    if (anObj.PosId().IsInit()) BinaryDumpInFile(aFp,anObj.PosId().Val());
    BinaryDumpInFile(aFp,anObj.PosFromBDOrient().IsInit());
    if (anObj.PosFromBDOrient().IsInit()) BinaryDumpInFile(aFp,anObj.PosFromBDOrient().Val());
    BinaryDumpInFile(aFp,anObj.PosFromBlockRigid().IsInit());
    if (anObj.PosFromBlockRigid().IsInit()) BinaryDumpInFile(aFp,anObj.PosFromBlockRigid().Val());
    BinaryDumpInFile(aFp,anObj.PosFromBDAppuis().IsInit());
    if (anObj.PosFromBDAppuis().IsInit()) BinaryDumpInFile(aFp,anObj.PosFromBDAppuis().Val());
    BinaryDumpInFile(aFp,anObj.PoseFromLiaisons().IsInit());
    if (anObj.PoseFromLiaisons().IsInit()) BinaryDumpInFile(aFp,anObj.PoseFromLiaisons().Val());
    BinaryDumpInFile(aFp,anObj.PoseInitFromReperePlan().IsInit());
    if (anObj.PoseInitFromReperePlan().IsInit()) BinaryDumpInFile(aFp,anObj.PoseInitFromReperePlan().Val());
}

cElXMLTree * ToXMLTree(const cPosValueInit & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"PosValueInit",eXMLBranche);
   if (anObj.PosId().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PosId"),anObj.PosId().Val())->ReTagThis("PosId"));
   if (anObj.PosFromBDOrient().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PosFromBDOrient"),anObj.PosFromBDOrient().Val())->ReTagThis("PosFromBDOrient"));
   if (anObj.PosFromBlockRigid().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PosFromBlockRigid"),anObj.PosFromBlockRigid().Val())->ReTagThis("PosFromBlockRigid"));
   if (anObj.PosFromBDAppuis().IsInit())
      aRes->AddFils(ToXMLTree(anObj.PosFromBDAppuis().Val())->ReTagThis("PosFromBDAppuis"));
   if (anObj.PoseFromLiaisons().IsInit())
      aRes->AddFils(ToXMLTree(anObj.PoseFromLiaisons().Val())->ReTagThis("PoseFromLiaisons"));
   if (anObj.PoseInitFromReperePlan().IsInit())
      aRes->AddFils(ToXMLTree(anObj.PoseInitFromReperePlan().Val())->ReTagThis("PoseInitFromReperePlan"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cPosValueInit & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.PosId(),aTree->Get("PosId",1)); //tototo 

   xml_init(anObj.PosFromBDOrient(),aTree->Get("PosFromBDOrient",1)); //tototo 

   xml_init(anObj.PosFromBlockRigid(),aTree->Get("PosFromBlockRigid",1)); //tototo 

   xml_init(anObj.PosFromBDAppuis(),aTree->Get("PosFromBDAppuis",1)); //tototo 

   xml_init(anObj.PoseFromLiaisons(),aTree->Get("PoseFromLiaisons",1)); //tototo 

   xml_init(anObj.PoseInitFromReperePlan(),aTree->Get("PoseInitFromReperePlan",1)); //tototo 
}

std::string  Mangling( cPosValueInit *) {return "A83A79E7F0B2DBE7FDBF";};


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


cTplValGesInit< std::string > & cPoseCameraInc::IdOffsetGPS()
{
   return mIdOffsetGPS;
}

const cTplValGesInit< std::string > & cPoseCameraInc::IdOffsetGPS()const 
{
   return mIdOffsetGPS;
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


cTplValGesInit< std::string > & cPoseCameraInc::MSTBlockRigid()
{
   return MEP_SPEC_MST().Val().MSTBlockRigid();
}

const cTplValGesInit< std::string > & cPoseCameraInc::MSTBlockRigid()const 
{
   return MEP_SPEC_MST().Val().MSTBlockRigid();
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


cTplValGesInit< std::string > & cPoseCameraInc::CalcNameCalib()
{
   return mCalcNameCalib;
}

const cTplValGesInit< std::string > & cPoseCameraInc::CalcNameCalib()const 
{
   return mCalcNameCalib;
}


std::list< cCalcNameCalibAux > & cPoseCameraInc::CalcNameCalibAux()
{
   return mCalcNameCalibAux;
}

const std::list< cCalcNameCalibAux > & cPoseCameraInc::CalcNameCalibAux()const 
{
   return mCalcNameCalibAux;
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


cTplValGesInit< std::string > & cPoseCameraInc::PosFromBlockRigid()
{
   return PosValueInit().PosFromBlockRigid();
}

const cTplValGesInit< std::string > & cPoseCameraInc::PosFromBlockRigid()const 
{
   return PosValueInit().PosFromBlockRigid();
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

void  BinaryUnDumpFromFile(cPoseCameraInc & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.OrInterne().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.OrInterne().ValForcedForUnUmp(),aFp);
        }
        else  anObj.OrInterne().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.IdBDCentre().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.IdBDCentre().ValForcedForUnUmp(),aFp);
        }
        else  anObj.IdBDCentre().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.IdOffsetGPS().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.IdOffsetGPS().ValForcedForUnUmp(),aFp);
        }
        else  anObj.IdOffsetGPS().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.InitNow().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.InitNow().ValForcedForUnUmp(),aFp);
        }
        else  anObj.InitNow().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ProfSceneImage().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ProfSceneImage().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ProfSceneImage().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Directory().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Directory().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Directory().SetNoInit();
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             std::string aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.PatternName().push_back(aVal);
        }
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.AutomGetImC().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.AutomGetImC().ValForcedForUnUmp(),aFp);
        }
        else  anObj.AutomGetImC().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.TestFewerTiePoints().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.TestFewerTiePoints().ValForcedForUnUmp(),aFp);
        }
        else  anObj.TestFewerTiePoints().SetNoInit();
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
             anObj.PatternRefuteur().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.PatternRefuteur().ValForcedForUnUmp(),aFp);
        }
        else  anObj.PatternRefuteur().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.AutoRefutDupl().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.AutoRefutDupl().ValForcedForUnUmp(),aFp);
        }
        else  anObj.AutoRefutDupl().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.KeyTranscriptionName().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.KeyTranscriptionName().ValForcedForUnUmp(),aFp);
        }
        else  anObj.KeyTranscriptionName().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.AddAllNameConnectedBy().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.AddAllNameConnectedBy().ValForcedForUnUmp(),aFp);
        }
        else  anObj.AddAllNameConnectedBy().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.FilterConnecBy().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.FilterConnecBy().ValForcedForUnUmp(),aFp);
        }
        else  anObj.FilterConnecBy().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.MEP_SPEC_MST().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.MEP_SPEC_MST().ValForcedForUnUmp(),aFp);
        }
        else  anObj.MEP_SPEC_MST().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.OptimizeAfterInit().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.OptimizeAfterInit().ValForcedForUnUmp(),aFp);
        }
        else  anObj.OptimizeAfterInit().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ReverseOrderName().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ReverseOrderName().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ReverseOrderName().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.CalcNameCalib().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.CalcNameCalib().ValForcedForUnUmp(),aFp);
        }
        else  anObj.CalcNameCalib().SetNoInit();
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cCalcNameCalibAux aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.CalcNameCalibAux().push_back(aVal);
        }
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.PosesDeRattachement().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.PosesDeRattachement().ValForcedForUnUmp(),aFp);
        }
        else  anObj.PosesDeRattachement().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NoErroOnRat().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NoErroOnRat().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NoErroOnRat().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ByPattern().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ByPattern().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ByPattern().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.KeyFilterExistingFile().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.KeyFilterExistingFile().ValForcedForUnUmp(),aFp);
        }
        else  anObj.KeyFilterExistingFile().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ByKey().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ByKey().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ByKey().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ByFile().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ByFile().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ByFile().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.PosValueInit(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cPoseCameraInc & anObj)
{
    BinaryDumpInFile(aFp,anObj.OrInterne().IsInit());
    if (anObj.OrInterne().IsInit()) BinaryDumpInFile(aFp,anObj.OrInterne().Val());
    BinaryDumpInFile(aFp,anObj.IdBDCentre().IsInit());
    if (anObj.IdBDCentre().IsInit()) BinaryDumpInFile(aFp,anObj.IdBDCentre().Val());
    BinaryDumpInFile(aFp,anObj.IdOffsetGPS().IsInit());
    if (anObj.IdOffsetGPS().IsInit()) BinaryDumpInFile(aFp,anObj.IdOffsetGPS().Val());
    BinaryDumpInFile(aFp,anObj.InitNow().IsInit());
    if (anObj.InitNow().IsInit()) BinaryDumpInFile(aFp,anObj.InitNow().Val());
    BinaryDumpInFile(aFp,anObj.ProfSceneImage().IsInit());
    if (anObj.ProfSceneImage().IsInit()) BinaryDumpInFile(aFp,anObj.ProfSceneImage().Val());
    BinaryDumpInFile(aFp,anObj.Directory().IsInit());
    if (anObj.Directory().IsInit()) BinaryDumpInFile(aFp,anObj.Directory().Val());
    BinaryDumpInFile(aFp,(int)anObj.PatternName().size());
    for(  std::list< std::string >::const_iterator iT=anObj.PatternName().begin();
         iT!=anObj.PatternName().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.AutomGetImC().IsInit());
    if (anObj.AutomGetImC().IsInit()) BinaryDumpInFile(aFp,anObj.AutomGetImC().Val());
    BinaryDumpInFile(aFp,anObj.TestFewerTiePoints().IsInit());
    if (anObj.TestFewerTiePoints().IsInit()) BinaryDumpInFile(aFp,anObj.TestFewerTiePoints().Val());
    BinaryDumpInFile(aFp,anObj.Filter().IsInit());
    if (anObj.Filter().IsInit()) BinaryDumpInFile(aFp,anObj.Filter().Val());
    BinaryDumpInFile(aFp,anObj.PatternRefuteur().IsInit());
    if (anObj.PatternRefuteur().IsInit()) BinaryDumpInFile(aFp,anObj.PatternRefuteur().Val());
    BinaryDumpInFile(aFp,anObj.AutoRefutDupl().IsInit());
    if (anObj.AutoRefutDupl().IsInit()) BinaryDumpInFile(aFp,anObj.AutoRefutDupl().Val());
    BinaryDumpInFile(aFp,anObj.KeyTranscriptionName().IsInit());
    if (anObj.KeyTranscriptionName().IsInit()) BinaryDumpInFile(aFp,anObj.KeyTranscriptionName().Val());
    BinaryDumpInFile(aFp,anObj.AddAllNameConnectedBy().IsInit());
    if (anObj.AddAllNameConnectedBy().IsInit()) BinaryDumpInFile(aFp,anObj.AddAllNameConnectedBy().Val());
    BinaryDumpInFile(aFp,anObj.FilterConnecBy().IsInit());
    if (anObj.FilterConnecBy().IsInit()) BinaryDumpInFile(aFp,anObj.FilterConnecBy().Val());
    BinaryDumpInFile(aFp,anObj.MEP_SPEC_MST().IsInit());
    if (anObj.MEP_SPEC_MST().IsInit()) BinaryDumpInFile(aFp,anObj.MEP_SPEC_MST().Val());
    BinaryDumpInFile(aFp,anObj.OptimizeAfterInit().IsInit());
    if (anObj.OptimizeAfterInit().IsInit()) BinaryDumpInFile(aFp,anObj.OptimizeAfterInit().Val());
    BinaryDumpInFile(aFp,anObj.ReverseOrderName().IsInit());
    if (anObj.ReverseOrderName().IsInit()) BinaryDumpInFile(aFp,anObj.ReverseOrderName().Val());
    BinaryDumpInFile(aFp,anObj.CalcNameCalib().IsInit());
    if (anObj.CalcNameCalib().IsInit()) BinaryDumpInFile(aFp,anObj.CalcNameCalib().Val());
    BinaryDumpInFile(aFp,(int)anObj.CalcNameCalibAux().size());
    for(  std::list< cCalcNameCalibAux >::const_iterator iT=anObj.CalcNameCalibAux().begin();
         iT!=anObj.CalcNameCalibAux().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.PosesDeRattachement().IsInit());
    if (anObj.PosesDeRattachement().IsInit()) BinaryDumpInFile(aFp,anObj.PosesDeRattachement().Val());
    BinaryDumpInFile(aFp,anObj.NoErroOnRat().IsInit());
    if (anObj.NoErroOnRat().IsInit()) BinaryDumpInFile(aFp,anObj.NoErroOnRat().Val());
    BinaryDumpInFile(aFp,anObj.ByPattern().IsInit());
    if (anObj.ByPattern().IsInit()) BinaryDumpInFile(aFp,anObj.ByPattern().Val());
    BinaryDumpInFile(aFp,anObj.KeyFilterExistingFile().IsInit());
    if (anObj.KeyFilterExistingFile().IsInit()) BinaryDumpInFile(aFp,anObj.KeyFilterExistingFile().Val());
    BinaryDumpInFile(aFp,anObj.ByKey().IsInit());
    if (anObj.ByKey().IsInit()) BinaryDumpInFile(aFp,anObj.ByKey().Val());
    BinaryDumpInFile(aFp,anObj.ByFile().IsInit());
    if (anObj.ByFile().IsInit()) BinaryDumpInFile(aFp,anObj.ByFile().Val());
    BinaryDumpInFile(aFp,anObj.PosValueInit());
}

cElXMLTree * ToXMLTree(const cPoseCameraInc & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"PoseCameraInc",eXMLBranche);
   if (anObj.OrInterne().IsInit())
      aRes->AddFils(ToXMLTree(anObj.OrInterne().Val())->ReTagThis("OrInterne"));
   if (anObj.IdBDCentre().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("IdBDCentre"),anObj.IdBDCentre().Val())->ReTagThis("IdBDCentre"));
   if (anObj.IdOffsetGPS().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("IdOffsetGPS"),anObj.IdOffsetGPS().Val())->ReTagThis("IdOffsetGPS"));
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
   if (anObj.CalcNameCalib().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("CalcNameCalib"),anObj.CalcNameCalib().Val())->ReTagThis("CalcNameCalib"));
  for
  (       std::list< cCalcNameCalibAux >::const_iterator it=anObj.CalcNameCalibAux().begin();
      it !=anObj.CalcNameCalibAux().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("CalcNameCalibAux"));
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
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cPoseCameraInc & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.OrInterne(),aTree->Get("OrInterne",1)); //tototo 

   xml_init(anObj.IdBDCentre(),aTree->Get("IdBDCentre",1)); //tototo 

   xml_init(anObj.IdOffsetGPS(),aTree->Get("IdOffsetGPS",1)); //tototo 

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

   xml_init(anObj.CalcNameCalibAux(),aTree->GetAll("CalcNameCalibAux",false,1));

   xml_init(anObj.PosesDeRattachement(),aTree->Get("PosesDeRattachement",1)); //tototo 

   xml_init(anObj.NoErroOnRat(),aTree->Get("NoErroOnRat",1),bool(true)); //tototo 

   xml_init(anObj.ByPattern(),aTree->Get("ByPattern",1),bool(true)); //tototo 

   xml_init(anObj.KeyFilterExistingFile(),aTree->Get("KeyFilterExistingFile",1)); //tototo 

   xml_init(anObj.ByKey(),aTree->Get("ByKey",1),bool(false)); //tototo 

   xml_init(anObj.ByFile(),aTree->Get("ByFile",1),bool(false)); //tototo 

   xml_init(anObj.PosValueInit(),aTree->Get("PosValueInit",1)); //tototo 
}

std::string  Mangling( cPoseCameraInc *) {return "24D23BFD5A4B8D96FE3F";};


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

void  BinaryUnDumpFromFile(cGroupeDePose & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.KeyPose2Grp(),aFp);
    BinaryUnDumpFromFile(anObj.Id(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ShowCreate().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ShowCreate().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ShowCreate().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cGroupeDePose & anObj)
{
    BinaryDumpInFile(aFp,anObj.KeyPose2Grp());
    BinaryDumpInFile(aFp,anObj.Id());
    BinaryDumpInFile(aFp,anObj.ShowCreate().IsInit());
    if (anObj.ShowCreate().IsInit()) BinaryDumpInFile(aFp,anObj.ShowCreate().Val());
}

cElXMLTree * ToXMLTree(const cGroupeDePose & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"GroupeDePose",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("KeyPose2Grp"),anObj.KeyPose2Grp())->ReTagThis("KeyPose2Grp"));
   aRes->AddFils(::ToXMLTree(std::string("Id"),anObj.Id())->ReTagThis("Id"));
   if (anObj.ShowCreate().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ShowCreate"),anObj.ShowCreate().Val())->ReTagThis("ShowCreate"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cGroupeDePose & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.KeyPose2Grp(),aTree->Get("KeyPose2Grp",1)); //tototo 

   xml_init(anObj.Id(),aTree->Get("Id",1)); //tototo 

   xml_init(anObj.ShowCreate(),aTree->Get("ShowCreate",1),bool(false)); //tototo 
}

std::string  Mangling( cGroupeDePose *) {return "40DCF2BC5AA3F49BFA3F";};


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

void  BinaryUnDumpFromFile(cLiaisonsApplyContrainte & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.NameRef(),aFp);
    BinaryUnDumpFromFile(anObj.PatternI1(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.PatternI2().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.PatternI2().ValForcedForUnUmp(),aFp);
        }
        else  anObj.PatternI2().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cLiaisonsApplyContrainte & anObj)
{
    BinaryDumpInFile(aFp,anObj.NameRef());
    BinaryDumpInFile(aFp,anObj.PatternI1());
    BinaryDumpInFile(aFp,anObj.PatternI2().IsInit());
    if (anObj.PatternI2().IsInit()) BinaryDumpInFile(aFp,anObj.PatternI2().Val());
}

cElXMLTree * ToXMLTree(const cLiaisonsApplyContrainte & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"LiaisonsApplyContrainte",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("NameRef"),anObj.NameRef())->ReTagThis("NameRef"));
   aRes->AddFils(::ToXMLTree(std::string("PatternI1"),anObj.PatternI1())->ReTagThis("PatternI1"));
   if (anObj.PatternI2().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PatternI2"),anObj.PatternI2().Val())->ReTagThis("PatternI2"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cLiaisonsApplyContrainte & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.NameRef(),aTree->Get("NameRef",1)); //tototo 

   xml_init(anObj.PatternI1(),aTree->Get("PatternI1",1)); //tototo 

   xml_init(anObj.PatternI2(),aTree->Get("PatternI2",1),std::string(".*")); //tototo 
}

std::string  Mangling( cLiaisonsApplyContrainte *) {return "D5AD2935B8687FA4FE3F";};


cTplValGesInit< std::string > & cInitSurf::ZonePlane()
{
   return mZonePlane;
}

const cTplValGesInit< std::string > & cInitSurf::ZonePlane()const 
{
   return mZonePlane;
}

void  BinaryUnDumpFromFile(cInitSurf & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ZonePlane().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ZonePlane().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ZonePlane().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cInitSurf & anObj)
{
    BinaryDumpInFile(aFp,anObj.ZonePlane().IsInit());
    if (anObj.ZonePlane().IsInit()) BinaryDumpInFile(aFp,anObj.ZonePlane().Val());
}

cElXMLTree * ToXMLTree(const cInitSurf & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"InitSurf",eXMLBranche);
   if (anObj.ZonePlane().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ZonePlane"),anObj.ZonePlane().Val())->ReTagThis("ZonePlane"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cInitSurf & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.ZonePlane(),aTree->Get("ZonePlane",1)); //tototo 
}

std::string  Mangling( cInitSurf *) {return "D0FC4E8E4F427B88FA3F";};


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

void  BinaryUnDumpFromFile(cSurfParamInc & anObj,ELISE_fp & aFp)
{
   { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cLiaisonsApplyContrainte aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.LiaisonsApplyContrainte().push_back(aVal);
        }
  } ;
    BinaryUnDumpFromFile(anObj.InitSurf(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cSurfParamInc & anObj)
{
    BinaryDumpInFile(aFp,(int)anObj.LiaisonsApplyContrainte().size());
    for(  std::list< cLiaisonsApplyContrainte >::const_iterator iT=anObj.LiaisonsApplyContrainte().begin();
         iT!=anObj.LiaisonsApplyContrainte().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.InitSurf());
}

cElXMLTree * ToXMLTree(const cSurfParamInc & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"SurfParamInc",eXMLBranche);
  for
  (       std::list< cLiaisonsApplyContrainte >::const_iterator it=anObj.LiaisonsApplyContrainte().begin();
      it !=anObj.LiaisonsApplyContrainte().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("LiaisonsApplyContrainte"));
   aRes->AddFils(ToXMLTree(anObj.InitSurf())->ReTagThis("InitSurf"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cSurfParamInc & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.LiaisonsApplyContrainte(),aTree->GetAll("LiaisonsApplyContrainte",false,1));

   xml_init(anObj.InitSurf(),aTree->Get("InitSurf",1)); //tototo 
}

std::string  Mangling( cSurfParamInc *) {return "F0B4A74010040AABFABF";};


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

void  BinaryUnDumpFromFile(cPointFlottantInc & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Id(),aFp);
    BinaryUnDumpFromFile(anObj.KeySetOrPat(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ModifInc().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ModifInc().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ModifInc().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cPointFlottantInc & anObj)
{
    BinaryDumpInFile(aFp,anObj.Id());
    BinaryDumpInFile(aFp,anObj.KeySetOrPat());
    BinaryDumpInFile(aFp,anObj.ModifInc().IsInit());
    if (anObj.ModifInc().IsInit()) BinaryDumpInFile(aFp,anObj.ModifInc().Val());
}

cElXMLTree * ToXMLTree(const cPointFlottantInc & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"PointFlottantInc",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Id"),anObj.Id())->ReTagThis("Id"));
   aRes->AddFils(::ToXMLTree(std::string("KeySetOrPat"),anObj.KeySetOrPat())->ReTagThis("KeySetOrPat"));
   if (anObj.ModifInc().IsInit())
      aRes->AddFils(ToXMLTree(anObj.ModifInc().Val())->ReTagThis("ModifInc"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cPointFlottantInc & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.Id(),aTree->Get("Id",1)); //tototo 

   xml_init(anObj.KeySetOrPat(),aTree->Get("KeySetOrPat",1)); //tototo 

   xml_init(anObj.ModifInc(),aTree->Get("ModifInc",1)); //tototo 
}

std::string  Mangling( cPointFlottantInc *) {return "82777025E4A68DE3FE3F";};


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


std::list< cGpsOffset > & cSectionInconnues::GpsOffset()
{
   return mGpsOffset;
}

const std::list< cGpsOffset > & cSectionInconnues::GpsOffset()const 
{
   return mGpsOffset;
}


std::list< cDataObsPlane > & cSectionInconnues::DataObsPlane()
{
   return mDataObsPlane;
}

const std::list< cDataObsPlane > & cSectionInconnues::DataObsPlane()const 
{
   return mDataObsPlane;
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


std::list< cBlockCamera > & cSectionInconnues::BlockCamera()
{
   return mBlockCamera;
}

const std::list< cBlockCamera > & cSectionInconnues::BlockCamera()const 
{
   return mBlockCamera;
}


cTplValGesInit< cSetOrientationInterne > & cSectionInconnues::GlobOrInterne()
{
   return mGlobOrInterne;
}

const cTplValGesInit< cSetOrientationInterne > & cSectionInconnues::GlobOrInterne()const 
{
   return mGlobOrInterne;
}


std::list< cCamGenInc > & cSectionInconnues::CamGenInc()
{
   return mCamGenInc;
}

const std::list< cCamGenInc > & cSectionInconnues::CamGenInc()const 
{
   return mCamGenInc;
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

void  BinaryUnDumpFromFile(cSectionInconnues & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SeuilAutomFE().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SeuilAutomFE().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SeuilAutomFE().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.AutoriseToujoursUneSeuleLiaison().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.AutoriseToujoursUneSeuleLiaison().ValForcedForUnUmp(),aFp);
        }
        else  anObj.AutoriseToujoursUneSeuleLiaison().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.MapMaskHom().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.MapMaskHom().ValForcedForUnUmp(),aFp);
        }
        else  anObj.MapMaskHom().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SauvePMoyenOnlyWithMasq().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SauvePMoyenOnlyWithMasq().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SauvePMoyenOnlyWithMasq().SetNoInit();
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cGpsOffset aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.GpsOffset().push_back(aVal);
        }
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cDataObsPlane aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.DataObsPlane().push_back(aVal);
        }
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cCalibrationCameraInc aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.CalibrationCameraInc().push_back(aVal);
        }
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SeuilL1EstimMatrEss().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SeuilL1EstimMatrEss().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SeuilL1EstimMatrEss().SetNoInit();
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cBlockCamera aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.BlockCamera().push_back(aVal);
        }
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.GlobOrInterne().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.GlobOrInterne().ValForcedForUnUmp(),aFp);
        }
        else  anObj.GlobOrInterne().SetNoInit();
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cCamGenInc aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.CamGenInc().push_back(aVal);
        }
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cPoseCameraInc aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.PoseCameraInc().push_back(aVal);
        }
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cGroupeDePose aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.GroupeDePose().push_back(aVal);
        }
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cSurfParamInc aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.SurfParamInc().push_back(aVal);
        }
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cPointFlottantInc aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.PointFlottantInc().push_back(aVal);
        }
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cSectionInconnues & anObj)
{
    BinaryDumpInFile(aFp,anObj.SeuilAutomFE().IsInit());
    if (anObj.SeuilAutomFE().IsInit()) BinaryDumpInFile(aFp,anObj.SeuilAutomFE().Val());
    BinaryDumpInFile(aFp,anObj.AutoriseToujoursUneSeuleLiaison().IsInit());
    if (anObj.AutoriseToujoursUneSeuleLiaison().IsInit()) BinaryDumpInFile(aFp,anObj.AutoriseToujoursUneSeuleLiaison().Val());
    BinaryDumpInFile(aFp,anObj.MapMaskHom().IsInit());
    if (anObj.MapMaskHom().IsInit()) BinaryDumpInFile(aFp,anObj.MapMaskHom().Val());
    BinaryDumpInFile(aFp,anObj.SauvePMoyenOnlyWithMasq().IsInit());
    if (anObj.SauvePMoyenOnlyWithMasq().IsInit()) BinaryDumpInFile(aFp,anObj.SauvePMoyenOnlyWithMasq().Val());
    BinaryDumpInFile(aFp,(int)anObj.GpsOffset().size());
    for(  std::list< cGpsOffset >::const_iterator iT=anObj.GpsOffset().begin();
         iT!=anObj.GpsOffset().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,(int)anObj.DataObsPlane().size());
    for(  std::list< cDataObsPlane >::const_iterator iT=anObj.DataObsPlane().begin();
         iT!=anObj.DataObsPlane().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,(int)anObj.CalibrationCameraInc().size());
    for(  std::list< cCalibrationCameraInc >::const_iterator iT=anObj.CalibrationCameraInc().begin();
         iT!=anObj.CalibrationCameraInc().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.SeuilL1EstimMatrEss().IsInit());
    if (anObj.SeuilL1EstimMatrEss().IsInit()) BinaryDumpInFile(aFp,anObj.SeuilL1EstimMatrEss().Val());
    BinaryDumpInFile(aFp,(int)anObj.BlockCamera().size());
    for(  std::list< cBlockCamera >::const_iterator iT=anObj.BlockCamera().begin();
         iT!=anObj.BlockCamera().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.GlobOrInterne().IsInit());
    if (anObj.GlobOrInterne().IsInit()) BinaryDumpInFile(aFp,anObj.GlobOrInterne().Val());
    BinaryDumpInFile(aFp,(int)anObj.CamGenInc().size());
    for(  std::list< cCamGenInc >::const_iterator iT=anObj.CamGenInc().begin();
         iT!=anObj.CamGenInc().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,(int)anObj.PoseCameraInc().size());
    for(  std::list< cPoseCameraInc >::const_iterator iT=anObj.PoseCameraInc().begin();
         iT!=anObj.PoseCameraInc().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,(int)anObj.GroupeDePose().size());
    for(  std::list< cGroupeDePose >::const_iterator iT=anObj.GroupeDePose().begin();
         iT!=anObj.GroupeDePose().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,(int)anObj.SurfParamInc().size());
    for(  std::list< cSurfParamInc >::const_iterator iT=anObj.SurfParamInc().begin();
         iT!=anObj.SurfParamInc().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,(int)anObj.PointFlottantInc().size());
    for(  std::list< cPointFlottantInc >::const_iterator iT=anObj.PointFlottantInc().begin();
         iT!=anObj.PointFlottantInc().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
}

cElXMLTree * ToXMLTree(const cSectionInconnues & anObj)
{
  XMLPushContext(anObj.mGXml);
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
  (       std::list< cGpsOffset >::const_iterator it=anObj.GpsOffset().begin();
      it !=anObj.GpsOffset().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("GpsOffset"));
  for
  (       std::list< cDataObsPlane >::const_iterator it=anObj.DataObsPlane().begin();
      it !=anObj.DataObsPlane().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("DataObsPlane"));
  for
  (       std::list< cCalibrationCameraInc >::const_iterator it=anObj.CalibrationCameraInc().begin();
      it !=anObj.CalibrationCameraInc().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("CalibrationCameraInc"));
   if (anObj.SeuilL1EstimMatrEss().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SeuilL1EstimMatrEss"),anObj.SeuilL1EstimMatrEss().Val())->ReTagThis("SeuilL1EstimMatrEss"));
  for
  (       std::list< cBlockCamera >::const_iterator it=anObj.BlockCamera().begin();
      it !=anObj.BlockCamera().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("BlockCamera"));
   if (anObj.GlobOrInterne().IsInit())
      aRes->AddFils(ToXMLTree(anObj.GlobOrInterne().Val())->ReTagThis("GlobOrInterne"));
  for
  (       std::list< cCamGenInc >::const_iterator it=anObj.CamGenInc().begin();
      it !=anObj.CamGenInc().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("CamGenInc"));
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
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cSectionInconnues & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.SeuilAutomFE(),aTree->Get("SeuilAutomFE",1),double(-1)); //tototo 

   xml_init(anObj.AutoriseToujoursUneSeuleLiaison(),aTree->Get("AutoriseToujoursUneSeuleLiaison",1),bool(false)); //tototo 

   xml_init(anObj.MapMaskHom(),aTree->Get("MapMaskHom",1)); //tototo 

   xml_init(anObj.SauvePMoyenOnlyWithMasq(),aTree->Get("SauvePMoyenOnlyWithMasq",1),bool(true)); //tototo 

   xml_init(anObj.GpsOffset(),aTree->GetAll("GpsOffset",false,1));

   xml_init(anObj.DataObsPlane(),aTree->GetAll("DataObsPlane",false,1));

   xml_init(anObj.CalibrationCameraInc(),aTree->GetAll("CalibrationCameraInc",false,1));

   xml_init(anObj.SeuilL1EstimMatrEss(),aTree->Get("SeuilL1EstimMatrEss",1),int(150)); //tototo 

   xml_init(anObj.BlockCamera(),aTree->GetAll("BlockCamera",false,1));

   xml_init(anObj.GlobOrInterne(),aTree->Get("GlobOrInterne",1)); //tototo 

   xml_init(anObj.CamGenInc(),aTree->GetAll("CamGenInc",false,1));

   xml_init(anObj.PoseCameraInc(),aTree->GetAll("PoseCameraInc",false,1));

   xml_init(anObj.GroupeDePose(),aTree->GetAll("GroupeDePose",false,1));

   xml_init(anObj.SurfParamInc(),aTree->GetAll("SurfParamInc",false,1));

   xml_init(anObj.PointFlottantInc(),aTree->GetAll("PointFlottantInc",false,1));
}

std::string  Mangling( cSectionInconnues *) {return "90B85CB08187B987FF3F";};


std::string & cRappelPose::IdOrient()
{
   return mIdOrient;
}

const std::string & cRappelPose::IdOrient()const 
{
   return mIdOrient;
}


double & cRappelPose::SigmaC()
{
   return mSigmaC;
}

const double & cRappelPose::SigmaC()const 
{
   return mSigmaC;
}


double & cRappelPose::SigmaR()
{
   return mSigmaR;
}

const double & cRappelPose::SigmaR()const 
{
   return mSigmaR;
}


cElRegex_Ptr & cRappelPose::PatternApply()
{
   return mPatternApply;
}

const cElRegex_Ptr & cRappelPose::PatternApply()const 
{
   return mPatternApply;
}

void  BinaryUnDumpFromFile(cRappelPose & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.IdOrient(),aFp);
    BinaryUnDumpFromFile(anObj.SigmaC(),aFp);
    BinaryUnDumpFromFile(anObj.SigmaR(),aFp);
    BinaryUnDumpFromFile(anObj.PatternApply(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cRappelPose & anObj)
{
    BinaryDumpInFile(aFp,anObj.IdOrient());
    BinaryDumpInFile(aFp,anObj.SigmaC());
    BinaryDumpInFile(aFp,anObj.SigmaR());
    BinaryDumpInFile(aFp,anObj.PatternApply());
}

cElXMLTree * ToXMLTree(const cRappelPose & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"RappelPose",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("IdOrient"),anObj.IdOrient())->ReTagThis("IdOrient"));
   aRes->AddFils(::ToXMLTree(std::string("SigmaC"),anObj.SigmaC())->ReTagThis("SigmaC"));
   aRes->AddFils(::ToXMLTree(std::string("SigmaR"),anObj.SigmaR())->ReTagThis("SigmaR"));
   aRes->AddFils(::ToXMLTree(std::string("PatternApply"),anObj.PatternApply())->ReTagThis("PatternApply"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cRappelPose & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.IdOrient(),aTree->Get("IdOrient",1)); //tototo 

   xml_init(anObj.SigmaC(),aTree->Get("SigmaC",1)); //tototo 

   xml_init(anObj.SigmaR(),aTree->Get("SigmaR",1)); //tototo 

   xml_init(anObj.PatternApply(),aTree->Get("PatternApply",1)); //tototo 
}

std::string  Mangling( cRappelPose *) {return "B7342D8BF89C60F2FE3F";};


cTplValGesInit< double > & cUseExportImageResidu::SzByPair()
{
   return mSzByPair;
}

const cTplValGesInit< double > & cUseExportImageResidu::SzByPair()const 
{
   return mSzByPair;
}


cTplValGesInit< double > & cUseExportImageResidu::SzByPose()
{
   return mSzByPose;
}

const cTplValGesInit< double > & cUseExportImageResidu::SzByPose()const 
{
   return mSzByPose;
}


cTplValGesInit< double > & cUseExportImageResidu::SzByCam()
{
   return mSzByCam;
}

const cTplValGesInit< double > & cUseExportImageResidu::SzByCam()const 
{
   return mSzByCam;
}


cTplValGesInit< double > & cUseExportImageResidu::NbMesByCase()
{
   return mNbMesByCase;
}

const cTplValGesInit< double > & cUseExportImageResidu::NbMesByCase()const 
{
   return mNbMesByCase;
}


std::string & cUseExportImageResidu::AeroExport()
{
   return mAeroExport;
}

const std::string & cUseExportImageResidu::AeroExport()const 
{
   return mAeroExport;
}


cTplValGesInit< bool > & cUseExportImageResidu::GeneratePly()
{
   return mGeneratePly;
}

const cTplValGesInit< bool > & cUseExportImageResidu::GeneratePly()const 
{
   return mGeneratePly;
}


cTplValGesInit< int > & cUseExportImageResidu::SzOrtho()
{
   return mSzOrtho;
}

const cTplValGesInit< int > & cUseExportImageResidu::SzOrtho()const 
{
   return mSzOrtho;
}

void  BinaryUnDumpFromFile(cUseExportImageResidu & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SzByPair().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SzByPair().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SzByPair().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SzByPose().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SzByPose().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SzByPose().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SzByCam().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SzByCam().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SzByCam().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NbMesByCase().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NbMesByCase().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NbMesByCase().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.AeroExport(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.GeneratePly().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.GeneratePly().ValForcedForUnUmp(),aFp);
        }
        else  anObj.GeneratePly().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SzOrtho().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SzOrtho().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SzOrtho().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cUseExportImageResidu & anObj)
{
    BinaryDumpInFile(aFp,anObj.SzByPair().IsInit());
    if (anObj.SzByPair().IsInit()) BinaryDumpInFile(aFp,anObj.SzByPair().Val());
    BinaryDumpInFile(aFp,anObj.SzByPose().IsInit());
    if (anObj.SzByPose().IsInit()) BinaryDumpInFile(aFp,anObj.SzByPose().Val());
    BinaryDumpInFile(aFp,anObj.SzByCam().IsInit());
    if (anObj.SzByCam().IsInit()) BinaryDumpInFile(aFp,anObj.SzByCam().Val());
    BinaryDumpInFile(aFp,anObj.NbMesByCase().IsInit());
    if (anObj.NbMesByCase().IsInit()) BinaryDumpInFile(aFp,anObj.NbMesByCase().Val());
    BinaryDumpInFile(aFp,anObj.AeroExport());
    BinaryDumpInFile(aFp,anObj.GeneratePly().IsInit());
    if (anObj.GeneratePly().IsInit()) BinaryDumpInFile(aFp,anObj.GeneratePly().Val());
    BinaryDumpInFile(aFp,anObj.SzOrtho().IsInit());
    if (anObj.SzOrtho().IsInit()) BinaryDumpInFile(aFp,anObj.SzOrtho().Val());
}

cElXMLTree * ToXMLTree(const cUseExportImageResidu & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"UseExportImageResidu",eXMLBranche);
   if (anObj.SzByPair().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SzByPair"),anObj.SzByPair().Val())->ReTagThis("SzByPair"));
   if (anObj.SzByPose().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SzByPose"),anObj.SzByPose().Val())->ReTagThis("SzByPose"));
   if (anObj.SzByCam().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SzByCam"),anObj.SzByCam().Val())->ReTagThis("SzByCam"));
   if (anObj.NbMesByCase().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NbMesByCase"),anObj.NbMesByCase().Val())->ReTagThis("NbMesByCase"));
   aRes->AddFils(::ToXMLTree(std::string("AeroExport"),anObj.AeroExport())->ReTagThis("AeroExport"));
   if (anObj.GeneratePly().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("GeneratePly"),anObj.GeneratePly().Val())->ReTagThis("GeneratePly"));
   if (anObj.SzOrtho().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SzOrtho"),anObj.SzOrtho().Val())->ReTagThis("SzOrtho"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cUseExportImageResidu & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.SzByPair(),aTree->Get("SzByPair",1),double(30)); //tototo 

   xml_init(anObj.SzByPose(),aTree->Get("SzByPose",1),double(50)); //tototo 

   xml_init(anObj.SzByCam(),aTree->Get("SzByCam",1),double(100)); //tototo 

   xml_init(anObj.NbMesByCase(),aTree->Get("NbMesByCase",1),double(10)); //tototo 

   xml_init(anObj.AeroExport(),aTree->Get("AeroExport",1)); //tototo 

   xml_init(anObj.GeneratePly(),aTree->Get("GeneratePly",1),bool(true)); //tototo 

   xml_init(anObj.SzOrtho(),aTree->Get("SzOrtho",1),int(-1)); //tototo 
}

std::string  Mangling( cUseExportImageResidu *) {return "66F3871A49136483FD3F";};


double & cTimeLinkage::DeltaMax()
{
   return mDeltaMax;
}

const double & cTimeLinkage::DeltaMax()const 
{
   return mDeltaMax;
}

void  BinaryUnDumpFromFile(cTimeLinkage & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.DeltaMax(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cTimeLinkage & anObj)
{
    BinaryDumpInFile(aFp,anObj.DeltaMax());
}

cElXMLTree * ToXMLTree(const cTimeLinkage & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"TimeLinkage",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("DeltaMax"),anObj.DeltaMax())->ReTagThis("DeltaMax"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cTimeLinkage & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.DeltaMax(),aTree->Get("DeltaMax",1)); //tototo 
}

std::string  Mangling( cTimeLinkage *) {return "BEB337D7A41F8CD1FD3F";};


std::string & cSectionChantier::IdOrient()
{
   return RappelPose().Val().IdOrient();
}

const std::string & cSectionChantier::IdOrient()const 
{
   return RappelPose().Val().IdOrient();
}


double & cSectionChantier::SigmaC()
{
   return RappelPose().Val().SigmaC();
}

const double & cSectionChantier::SigmaC()const 
{
   return RappelPose().Val().SigmaC();
}


double & cSectionChantier::SigmaR()
{
   return RappelPose().Val().SigmaR();
}

const double & cSectionChantier::SigmaR()const 
{
   return RappelPose().Val().SigmaR();
}


cElRegex_Ptr & cSectionChantier::PatternApply()
{
   return RappelPose().Val().PatternApply();
}

const cElRegex_Ptr & cSectionChantier::PatternApply()const 
{
   return RappelPose().Val().PatternApply();
}


cTplValGesInit< cRappelPose > & cSectionChantier::RappelPose()
{
   return mRappelPose;
}

const cTplValGesInit< cRappelPose > & cSectionChantier::RappelPose()const 
{
   return mRappelPose;
}


cTplValGesInit< int > & cSectionChantier::NumAttrPdsNewF()
{
   return mNumAttrPdsNewF;
}

const cTplValGesInit< int > & cSectionChantier::NumAttrPdsNewF()const 
{
   return mNumAttrPdsNewF;
}


cTplValGesInit< double > & cSectionChantier::RatioMaxDistCS()
{
   return mRatioMaxDistCS;
}

const cTplValGesInit< double > & cSectionChantier::RatioMaxDistCS()const 
{
   return mRatioMaxDistCS;
}


cTplValGesInit< std::string > & cSectionChantier::DebugVecElimTieP()
{
   return mDebugVecElimTieP;
}

const cTplValGesInit< std::string > & cSectionChantier::DebugVecElimTieP()const 
{
   return mDebugVecElimTieP;
}


cTplValGesInit< int > & cSectionChantier::DoStatElimBundle()
{
   return mDoStatElimBundle;
}

const cTplValGesInit< int > & cSectionChantier::DoStatElimBundle()const 
{
   return mDoStatElimBundle;
}


cTplValGesInit< double > & cSectionChantier::SzByPair()
{
   return UseExportImageResidu().Val().SzByPair();
}

const cTplValGesInit< double > & cSectionChantier::SzByPair()const 
{
   return UseExportImageResidu().Val().SzByPair();
}


cTplValGesInit< double > & cSectionChantier::SzByPose()
{
   return UseExportImageResidu().Val().SzByPose();
}

const cTplValGesInit< double > & cSectionChantier::SzByPose()const 
{
   return UseExportImageResidu().Val().SzByPose();
}


cTplValGesInit< double > & cSectionChantier::SzByCam()
{
   return UseExportImageResidu().Val().SzByCam();
}

const cTplValGesInit< double > & cSectionChantier::SzByCam()const 
{
   return UseExportImageResidu().Val().SzByCam();
}


cTplValGesInit< double > & cSectionChantier::NbMesByCase()
{
   return UseExportImageResidu().Val().NbMesByCase();
}

const cTplValGesInit< double > & cSectionChantier::NbMesByCase()const 
{
   return UseExportImageResidu().Val().NbMesByCase();
}


std::string & cSectionChantier::AeroExport()
{
   return UseExportImageResidu().Val().AeroExport();
}

const std::string & cSectionChantier::AeroExport()const 
{
   return UseExportImageResidu().Val().AeroExport();
}


cTplValGesInit< bool > & cSectionChantier::GeneratePly()
{
   return UseExportImageResidu().Val().GeneratePly();
}

const cTplValGesInit< bool > & cSectionChantier::GeneratePly()const 
{
   return UseExportImageResidu().Val().GeneratePly();
}


cTplValGesInit< int > & cSectionChantier::SzOrtho()
{
   return UseExportImageResidu().Val().SzOrtho();
}

const cTplValGesInit< int > & cSectionChantier::SzOrtho()const 
{
   return UseExportImageResidu().Val().SzOrtho();
}


cTplValGesInit< cUseExportImageResidu > & cSectionChantier::UseExportImageResidu()
{
   return mUseExportImageResidu;
}

const cTplValGesInit< cUseExportImageResidu > & cSectionChantier::UseExportImageResidu()const 
{
   return mUseExportImageResidu;
}


cTplValGesInit< bool > & cSectionChantier::UseRegulDist()
{
   return mUseRegulDist;
}

const cTplValGesInit< bool > & cSectionChantier::UseRegulDist()const 
{
   return mUseRegulDist;
}


cTplValGesInit< bool > & cSectionChantier::GBCamSupresStenCam()
{
   return mGBCamSupresStenCam;
}

const cTplValGesInit< bool > & cSectionChantier::GBCamSupresStenCam()const 
{
   return mGBCamSupresStenCam;
}


cTplValGesInit< bool > & cSectionChantier::StenCamSupresGBCam()
{
   return mStenCamSupresGBCam;
}

const cTplValGesInit< bool > & cSectionChantier::StenCamSupresGBCam()const 
{
   return mStenCamSupresGBCam;
}


cTplValGesInit< bool > & cSectionChantier::IsAperiCloud()
{
   return mIsAperiCloud;
}

const cTplValGesInit< bool > & cSectionChantier::IsAperiCloud()const 
{
   return mIsAperiCloud;
}


cTplValGesInit< bool > & cSectionChantier::IsChoixImSec()
{
   return mIsChoixImSec;
}

const cTplValGesInit< bool > & cSectionChantier::IsChoixImSec()const 
{
   return mIsChoixImSec;
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


cTplValGesInit< bool > & cSectionChantier::ExportMatrixMarket()
{
   return mExportMatrixMarket;
}

const cTplValGesInit< bool > & cSectionChantier::ExportMatrixMarket()const 
{
   return mExportMatrixMarket;
}


cTplValGesInit< double > & cSectionChantier::ExtensionIntervZ()
{
   return mExtensionIntervZ;
}

const cTplValGesInit< double > & cSectionChantier::ExtensionIntervZ()const 
{
   return mExtensionIntervZ;
}

void  BinaryUnDumpFromFile(cSectionChantier & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.RappelPose().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.RappelPose().ValForcedForUnUmp(),aFp);
        }
        else  anObj.RappelPose().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NumAttrPdsNewF().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NumAttrPdsNewF().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NumAttrPdsNewF().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.RatioMaxDistCS().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.RatioMaxDistCS().ValForcedForUnUmp(),aFp);
        }
        else  anObj.RatioMaxDistCS().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.DebugVecElimTieP().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.DebugVecElimTieP().ValForcedForUnUmp(),aFp);
        }
        else  anObj.DebugVecElimTieP().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.DoStatElimBundle().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.DoStatElimBundle().ValForcedForUnUmp(),aFp);
        }
        else  anObj.DoStatElimBundle().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.UseExportImageResidu().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.UseExportImageResidu().ValForcedForUnUmp(),aFp);
        }
        else  anObj.UseExportImageResidu().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.UseRegulDist().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.UseRegulDist().ValForcedForUnUmp(),aFp);
        }
        else  anObj.UseRegulDist().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.GBCamSupresStenCam().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.GBCamSupresStenCam().ValForcedForUnUmp(),aFp);
        }
        else  anObj.GBCamSupresStenCam().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.StenCamSupresGBCam().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.StenCamSupresGBCam().ValForcedForUnUmp(),aFp);
        }
        else  anObj.StenCamSupresGBCam().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.IsAperiCloud().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.IsAperiCloud().ValForcedForUnUmp(),aFp);
        }
        else  anObj.IsAperiCloud().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.IsChoixImSec().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.IsChoixImSec().ValForcedForUnUmp(),aFp);
        }
        else  anObj.IsChoixImSec().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.FileSauvParam().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.FileSauvParam().ValForcedForUnUmp(),aFp);
        }
        else  anObj.FileSauvParam().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.GenereErreurOnContraineCam().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.GenereErreurOnContraineCam().ValForcedForUnUmp(),aFp);
        }
        else  anObj.GenereErreurOnContraineCam().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ProfSceneChantier().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ProfSceneChantier().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ProfSceneChantier().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.DirectoryChantier().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.DirectoryChantier().ValForcedForUnUmp(),aFp);
        }
        else  anObj.DirectoryChantier().SetNoInit();
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
             anObj.NameParamEtal().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NameParamEtal().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NameParamEtal().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.PatternTracePose().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.PatternTracePose().ValForcedForUnUmp(),aFp);
        }
        else  anObj.PatternTracePose().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.TraceGimbalLock().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.TraceGimbalLock().ValForcedForUnUmp(),aFp);
        }
        else  anObj.TraceGimbalLock().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.MaxDistErrorPtsTerr().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.MaxDistErrorPtsTerr().ValForcedForUnUmp(),aFp);
        }
        else  anObj.MaxDistErrorPtsTerr().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.MaxDistWarnPtsTerr().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.MaxDistWarnPtsTerr().ValForcedForUnUmp(),aFp);
        }
        else  anObj.MaxDistWarnPtsTerr().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.DefPbLiaison().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.DefPbLiaison().ValForcedForUnUmp(),aFp);
        }
        else  anObj.DefPbLiaison().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.DoCompensation().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.DoCompensation().ValForcedForUnUmp(),aFp);
        }
        else  anObj.DoCompensation().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.TimeLinkage().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.TimeLinkage().ValForcedForUnUmp(),aFp);
        }
        else  anObj.TimeLinkage().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.DebugPbCondFaisceau().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.DebugPbCondFaisceau().ValForcedForUnUmp(),aFp);
        }
        else  anObj.DebugPbCondFaisceau().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SauvAutom().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SauvAutom().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SauvAutom().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SauvAutomBasic().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SauvAutomBasic().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SauvAutomBasic().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ThresholdWarnPointsBehind().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ThresholdWarnPointsBehind().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ThresholdWarnPointsBehind().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ExportMatrixMarket().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ExportMatrixMarket().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ExportMatrixMarket().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ExtensionIntervZ().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ExtensionIntervZ().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ExtensionIntervZ().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cSectionChantier & anObj)
{
    BinaryDumpInFile(aFp,anObj.RappelPose().IsInit());
    if (anObj.RappelPose().IsInit()) BinaryDumpInFile(aFp,anObj.RappelPose().Val());
    BinaryDumpInFile(aFp,anObj.NumAttrPdsNewF().IsInit());
    if (anObj.NumAttrPdsNewF().IsInit()) BinaryDumpInFile(aFp,anObj.NumAttrPdsNewF().Val());
    BinaryDumpInFile(aFp,anObj.RatioMaxDistCS().IsInit());
    if (anObj.RatioMaxDistCS().IsInit()) BinaryDumpInFile(aFp,anObj.RatioMaxDistCS().Val());
    BinaryDumpInFile(aFp,anObj.DebugVecElimTieP().IsInit());
    if (anObj.DebugVecElimTieP().IsInit()) BinaryDumpInFile(aFp,anObj.DebugVecElimTieP().Val());
    BinaryDumpInFile(aFp,anObj.DoStatElimBundle().IsInit());
    if (anObj.DoStatElimBundle().IsInit()) BinaryDumpInFile(aFp,anObj.DoStatElimBundle().Val());
    BinaryDumpInFile(aFp,anObj.UseExportImageResidu().IsInit());
    if (anObj.UseExportImageResidu().IsInit()) BinaryDumpInFile(aFp,anObj.UseExportImageResidu().Val());
    BinaryDumpInFile(aFp,anObj.UseRegulDist().IsInit());
    if (anObj.UseRegulDist().IsInit()) BinaryDumpInFile(aFp,anObj.UseRegulDist().Val());
    BinaryDumpInFile(aFp,anObj.GBCamSupresStenCam().IsInit());
    if (anObj.GBCamSupresStenCam().IsInit()) BinaryDumpInFile(aFp,anObj.GBCamSupresStenCam().Val());
    BinaryDumpInFile(aFp,anObj.StenCamSupresGBCam().IsInit());
    if (anObj.StenCamSupresGBCam().IsInit()) BinaryDumpInFile(aFp,anObj.StenCamSupresGBCam().Val());
    BinaryDumpInFile(aFp,anObj.IsAperiCloud().IsInit());
    if (anObj.IsAperiCloud().IsInit()) BinaryDumpInFile(aFp,anObj.IsAperiCloud().Val());
    BinaryDumpInFile(aFp,anObj.IsChoixImSec().IsInit());
    if (anObj.IsChoixImSec().IsInit()) BinaryDumpInFile(aFp,anObj.IsChoixImSec().Val());
    BinaryDumpInFile(aFp,anObj.FileSauvParam().IsInit());
    if (anObj.FileSauvParam().IsInit()) BinaryDumpInFile(aFp,anObj.FileSauvParam().Val());
    BinaryDumpInFile(aFp,anObj.GenereErreurOnContraineCam().IsInit());
    if (anObj.GenereErreurOnContraineCam().IsInit()) BinaryDumpInFile(aFp,anObj.GenereErreurOnContraineCam().Val());
    BinaryDumpInFile(aFp,anObj.ProfSceneChantier().IsInit());
    if (anObj.ProfSceneChantier().IsInit()) BinaryDumpInFile(aFp,anObj.ProfSceneChantier().Val());
    BinaryDumpInFile(aFp,anObj.DirectoryChantier().IsInit());
    if (anObj.DirectoryChantier().IsInit()) BinaryDumpInFile(aFp,anObj.DirectoryChantier().Val());
    BinaryDumpInFile(aFp,anObj.FileChantierNameDescripteur().IsInit());
    if (anObj.FileChantierNameDescripteur().IsInit()) BinaryDumpInFile(aFp,anObj.FileChantierNameDescripteur().Val());
    BinaryDumpInFile(aFp,anObj.NameParamEtal().IsInit());
    if (anObj.NameParamEtal().IsInit()) BinaryDumpInFile(aFp,anObj.NameParamEtal().Val());
    BinaryDumpInFile(aFp,anObj.PatternTracePose().IsInit());
    if (anObj.PatternTracePose().IsInit()) BinaryDumpInFile(aFp,anObj.PatternTracePose().Val());
    BinaryDumpInFile(aFp,anObj.TraceGimbalLock().IsInit());
    if (anObj.TraceGimbalLock().IsInit()) BinaryDumpInFile(aFp,anObj.TraceGimbalLock().Val());
    BinaryDumpInFile(aFp,anObj.MaxDistErrorPtsTerr().IsInit());
    if (anObj.MaxDistErrorPtsTerr().IsInit()) BinaryDumpInFile(aFp,anObj.MaxDistErrorPtsTerr().Val());
    BinaryDumpInFile(aFp,anObj.MaxDistWarnPtsTerr().IsInit());
    if (anObj.MaxDistWarnPtsTerr().IsInit()) BinaryDumpInFile(aFp,anObj.MaxDistWarnPtsTerr().Val());
    BinaryDumpInFile(aFp,anObj.DefPbLiaison().IsInit());
    if (anObj.DefPbLiaison().IsInit()) BinaryDumpInFile(aFp,anObj.DefPbLiaison().Val());
    BinaryDumpInFile(aFp,anObj.DoCompensation().IsInit());
    if (anObj.DoCompensation().IsInit()) BinaryDumpInFile(aFp,anObj.DoCompensation().Val());
    BinaryDumpInFile(aFp,anObj.TimeLinkage().IsInit());
    if (anObj.TimeLinkage().IsInit()) BinaryDumpInFile(aFp,anObj.TimeLinkage().Val());
    BinaryDumpInFile(aFp,anObj.DebugPbCondFaisceau().IsInit());
    if (anObj.DebugPbCondFaisceau().IsInit()) BinaryDumpInFile(aFp,anObj.DebugPbCondFaisceau().Val());
    BinaryDumpInFile(aFp,anObj.SauvAutom().IsInit());
    if (anObj.SauvAutom().IsInit()) BinaryDumpInFile(aFp,anObj.SauvAutom().Val());
    BinaryDumpInFile(aFp,anObj.SauvAutomBasic().IsInit());
    if (anObj.SauvAutomBasic().IsInit()) BinaryDumpInFile(aFp,anObj.SauvAutomBasic().Val());
    BinaryDumpInFile(aFp,anObj.ThresholdWarnPointsBehind().IsInit());
    if (anObj.ThresholdWarnPointsBehind().IsInit()) BinaryDumpInFile(aFp,anObj.ThresholdWarnPointsBehind().Val());
    BinaryDumpInFile(aFp,anObj.ExportMatrixMarket().IsInit());
    if (anObj.ExportMatrixMarket().IsInit()) BinaryDumpInFile(aFp,anObj.ExportMatrixMarket().Val());
    BinaryDumpInFile(aFp,anObj.ExtensionIntervZ().IsInit());
    if (anObj.ExtensionIntervZ().IsInit()) BinaryDumpInFile(aFp,anObj.ExtensionIntervZ().Val());
}

cElXMLTree * ToXMLTree(const cSectionChantier & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"SectionChantier",eXMLBranche);
   if (anObj.RappelPose().IsInit())
      aRes->AddFils(ToXMLTree(anObj.RappelPose().Val())->ReTagThis("RappelPose"));
   if (anObj.NumAttrPdsNewF().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NumAttrPdsNewF"),anObj.NumAttrPdsNewF().Val())->ReTagThis("NumAttrPdsNewF"));
   if (anObj.RatioMaxDistCS().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("RatioMaxDistCS"),anObj.RatioMaxDistCS().Val())->ReTagThis("RatioMaxDistCS"));
   if (anObj.DebugVecElimTieP().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DebugVecElimTieP"),anObj.DebugVecElimTieP().Val())->ReTagThis("DebugVecElimTieP"));
   if (anObj.DoStatElimBundle().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DoStatElimBundle"),anObj.DoStatElimBundle().Val())->ReTagThis("DoStatElimBundle"));
   if (anObj.UseExportImageResidu().IsInit())
      aRes->AddFils(ToXMLTree(anObj.UseExportImageResidu().Val())->ReTagThis("UseExportImageResidu"));
   if (anObj.UseRegulDist().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("UseRegulDist"),anObj.UseRegulDist().Val())->ReTagThis("UseRegulDist"));
   if (anObj.GBCamSupresStenCam().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("GBCamSupresStenCam"),anObj.GBCamSupresStenCam().Val())->ReTagThis("GBCamSupresStenCam"));
   if (anObj.StenCamSupresGBCam().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("StenCamSupresGBCam"),anObj.StenCamSupresGBCam().Val())->ReTagThis("StenCamSupresGBCam"));
   if (anObj.IsAperiCloud().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("IsAperiCloud"),anObj.IsAperiCloud().Val())->ReTagThis("IsAperiCloud"));
   if (anObj.IsChoixImSec().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("IsChoixImSec"),anObj.IsChoixImSec().Val())->ReTagThis("IsChoixImSec"));
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
   if (anObj.ExportMatrixMarket().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ExportMatrixMarket"),anObj.ExportMatrixMarket().Val())->ReTagThis("ExportMatrixMarket"));
   if (anObj.ExtensionIntervZ().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ExtensionIntervZ"),anObj.ExtensionIntervZ().Val())->ReTagThis("ExtensionIntervZ"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cSectionChantier & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.RappelPose(),aTree->Get("RappelPose",1)); //tototo 

   xml_init(anObj.NumAttrPdsNewF(),aTree->Get("NumAttrPdsNewF",1),int(-1)); //tototo 

   xml_init(anObj.RatioMaxDistCS(),aTree->Get("RatioMaxDistCS",1),double(30.0)); //tototo 

   xml_init(anObj.DebugVecElimTieP(),aTree->Get("DebugVecElimTieP",1)); //tototo 

   xml_init(anObj.DoStatElimBundle(),aTree->Get("DoStatElimBundle",1)); //tototo 

   xml_init(anObj.UseExportImageResidu(),aTree->Get("UseExportImageResidu",1)); //tototo 

   xml_init(anObj.UseRegulDist(),aTree->Get("UseRegulDist",1),bool(false)); //tototo 

   xml_init(anObj.GBCamSupresStenCam(),aTree->Get("GBCamSupresStenCam",1),bool(true)); //tototo 

   xml_init(anObj.StenCamSupresGBCam(),aTree->Get("StenCamSupresGBCam",1),bool(false)); //tototo 

   xml_init(anObj.IsAperiCloud(),aTree->Get("IsAperiCloud",1),bool(false)); //tototo 

   xml_init(anObj.IsChoixImSec(),aTree->Get("IsChoixImSec",1),bool(false)); //tototo 

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

   xml_init(anObj.ExportMatrixMarket(),aTree->Get("ExportMatrixMarket",1),bool(false)); //tototo 

   xml_init(anObj.ExtensionIntervZ(),aTree->Get("ExtensionIntervZ",1)); //tototo 
}

std::string  Mangling( cSectionChantier *) {return "9009C6EDC21F2DBFFE3F";};


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


cTplValGesInit< cXmlPondRegDist > & cSectionSolveur::RegDistGlob()
{
   return mRegDistGlob;
}

const cTplValGesInit< cXmlPondRegDist > & cSectionSolveur::RegDistGlob()const 
{
   return mRegDistGlob;
}

void  BinaryUnDumpFromFile(cSectionSolveur & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.AllMatSym().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.AllMatSym().ValForcedForUnUmp(),aFp);
        }
        else  anObj.AllMatSym().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.ModeResolution(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ModeControleDescDic().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ModeControleDescDic().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ModeControleDescDic().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SeuilBas_CDD().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SeuilBas_CDD().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SeuilBas_CDD().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SeuilHaut_CDD().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SeuilHaut_CDD().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SeuilHaut_CDD().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.InhibeAMD().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.InhibeAMD().ValForcedForUnUmp(),aFp);
        }
        else  anObj.InhibeAMD().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.AMDSpecInterne().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.AMDSpecInterne().ValForcedForUnUmp(),aFp);
        }
        else  anObj.AMDSpecInterne().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ShowCholesky().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ShowCholesky().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ShowCholesky().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.TestPermutVar().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.TestPermutVar().ValForcedForUnUmp(),aFp);
        }
        else  anObj.TestPermutVar().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ShowPermutVar().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ShowPermutVar().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ShowPermutVar().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.PermutIndex().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.PermutIndex().ValForcedForUnUmp(),aFp);
        }
        else  anObj.PermutIndex().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NormaliseEqSc().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NormaliseEqSc().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NormaliseEqSc().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NormaliseEqTr().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NormaliseEqTr().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NormaliseEqTr().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.LimBsHProj().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.LimBsHProj().ValForcedForUnUmp(),aFp);
        }
        else  anObj.LimBsHProj().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.LimBsHRefut().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.LimBsHRefut().ValForcedForUnUmp(),aFp);
        }
        else  anObj.LimBsHRefut().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.LimModeGL().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.LimModeGL().ValForcedForUnUmp(),aFp);
        }
        else  anObj.LimModeGL().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.GridOptimKnownDist().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.GridOptimKnownDist().ValForcedForUnUmp(),aFp);
        }
        else  anObj.GridOptimKnownDist().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SLMGlob().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SLMGlob().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SLMGlob().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.MultSLMGlob().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.MultSLMGlob().ValForcedForUnUmp(),aFp);
        }
        else  anObj.MultSLMGlob().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Im2Aff().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Im2Aff().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Im2Aff().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.RegDistGlob().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.RegDistGlob().ValForcedForUnUmp(),aFp);
        }
        else  anObj.RegDistGlob().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cSectionSolveur & anObj)
{
    BinaryDumpInFile(aFp,anObj.AllMatSym().IsInit());
    if (anObj.AllMatSym().IsInit()) BinaryDumpInFile(aFp,anObj.AllMatSym().Val());
    BinaryDumpInFile(aFp,anObj.ModeResolution());
    BinaryDumpInFile(aFp,anObj.ModeControleDescDic().IsInit());
    if (anObj.ModeControleDescDic().IsInit()) BinaryDumpInFile(aFp,anObj.ModeControleDescDic().Val());
    BinaryDumpInFile(aFp,anObj.SeuilBas_CDD().IsInit());
    if (anObj.SeuilBas_CDD().IsInit()) BinaryDumpInFile(aFp,anObj.SeuilBas_CDD().Val());
    BinaryDumpInFile(aFp,anObj.SeuilHaut_CDD().IsInit());
    if (anObj.SeuilHaut_CDD().IsInit()) BinaryDumpInFile(aFp,anObj.SeuilHaut_CDD().Val());
    BinaryDumpInFile(aFp,anObj.InhibeAMD().IsInit());
    if (anObj.InhibeAMD().IsInit()) BinaryDumpInFile(aFp,anObj.InhibeAMD().Val());
    BinaryDumpInFile(aFp,anObj.AMDSpecInterne().IsInit());
    if (anObj.AMDSpecInterne().IsInit()) BinaryDumpInFile(aFp,anObj.AMDSpecInterne().Val());
    BinaryDumpInFile(aFp,anObj.ShowCholesky().IsInit());
    if (anObj.ShowCholesky().IsInit()) BinaryDumpInFile(aFp,anObj.ShowCholesky().Val());
    BinaryDumpInFile(aFp,anObj.TestPermutVar().IsInit());
    if (anObj.TestPermutVar().IsInit()) BinaryDumpInFile(aFp,anObj.TestPermutVar().Val());
    BinaryDumpInFile(aFp,anObj.ShowPermutVar().IsInit());
    if (anObj.ShowPermutVar().IsInit()) BinaryDumpInFile(aFp,anObj.ShowPermutVar().Val());
    BinaryDumpInFile(aFp,anObj.PermutIndex().IsInit());
    if (anObj.PermutIndex().IsInit()) BinaryDumpInFile(aFp,anObj.PermutIndex().Val());
    BinaryDumpInFile(aFp,anObj.NormaliseEqSc().IsInit());
    if (anObj.NormaliseEqSc().IsInit()) BinaryDumpInFile(aFp,anObj.NormaliseEqSc().Val());
    BinaryDumpInFile(aFp,anObj.NormaliseEqTr().IsInit());
    if (anObj.NormaliseEqTr().IsInit()) BinaryDumpInFile(aFp,anObj.NormaliseEqTr().Val());
    BinaryDumpInFile(aFp,anObj.LimBsHProj().IsInit());
    if (anObj.LimBsHProj().IsInit()) BinaryDumpInFile(aFp,anObj.LimBsHProj().Val());
    BinaryDumpInFile(aFp,anObj.LimBsHRefut().IsInit());
    if (anObj.LimBsHRefut().IsInit()) BinaryDumpInFile(aFp,anObj.LimBsHRefut().Val());
    BinaryDumpInFile(aFp,anObj.LimModeGL().IsInit());
    if (anObj.LimModeGL().IsInit()) BinaryDumpInFile(aFp,anObj.LimModeGL().Val());
    BinaryDumpInFile(aFp,anObj.GridOptimKnownDist().IsInit());
    if (anObj.GridOptimKnownDist().IsInit()) BinaryDumpInFile(aFp,anObj.GridOptimKnownDist().Val());
    BinaryDumpInFile(aFp,anObj.SLMGlob().IsInit());
    if (anObj.SLMGlob().IsInit()) BinaryDumpInFile(aFp,anObj.SLMGlob().Val());
    BinaryDumpInFile(aFp,anObj.MultSLMGlob().IsInit());
    if (anObj.MultSLMGlob().IsInit()) BinaryDumpInFile(aFp,anObj.MultSLMGlob().Val());
    BinaryDumpInFile(aFp,anObj.Im2Aff().IsInit());
    if (anObj.Im2Aff().IsInit()) BinaryDumpInFile(aFp,anObj.Im2Aff().Val());
    BinaryDumpInFile(aFp,anObj.RegDistGlob().IsInit());
    if (anObj.RegDistGlob().IsInit()) BinaryDumpInFile(aFp,anObj.RegDistGlob().Val());
}

cElXMLTree * ToXMLTree(const cSectionSolveur & anObj)
{
  XMLPushContext(anObj.mGXml);
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
   if (anObj.RegDistGlob().IsInit())
      aRes->AddFils(ToXMLTree(anObj.RegDistGlob().Val())->ReTagThis("RegDistGlob"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cSectionSolveur & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

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

   xml_init(anObj.RegDistGlob(),aTree->Get("RegDistGlob",1)); //tototo 
}

std::string  Mangling( cSectionSolveur *) {return "CC1E52859DF6A480FE3F";};


double & cAutoAdaptLVM::Mult()
{
   return mMult;
}

const double & cAutoAdaptLVM::Mult()const 
{
   return mMult;
}


cTplValGesInit< bool > & cAutoAdaptLVM::ModeMin()
{
   return mModeMin;
}

const cTplValGesInit< bool > & cAutoAdaptLVM::ModeMin()const 
{
   return mModeMin;
}

void  BinaryUnDumpFromFile(cAutoAdaptLVM & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Mult(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ModeMin().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ModeMin().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ModeMin().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cAutoAdaptLVM & anObj)
{
    BinaryDumpInFile(aFp,anObj.Mult());
    BinaryDumpInFile(aFp,anObj.ModeMin().IsInit());
    if (anObj.ModeMin().IsInit()) BinaryDumpInFile(aFp,anObj.ModeMin().Val());
}

cElXMLTree * ToXMLTree(const cAutoAdaptLVM & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"AutoAdaptLVM",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Mult"),anObj.Mult())->ReTagThis("Mult"));
   if (anObj.ModeMin().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ModeMin"),anObj.ModeMin().Val())->ReTagThis("ModeMin"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cAutoAdaptLVM & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.Mult(),aTree->Get("Mult",1)); //tototo 

   xml_init(anObj.ModeMin(),aTree->Get("ModeMin",1),bool(true)); //tototo 
}

std::string  Mangling( cAutoAdaptLVM *) {return "B4565ED5A3F1528DFF3F";};


cTplValGesInit< int > & cCtrlTimeCompens::NbMin()
{
   return mNbMin;
}

const cTplValGesInit< int > & cCtrlTimeCompens::NbMin()const 
{
   return mNbMin;
}


int & cCtrlTimeCompens::NbMax()
{
   return mNbMax;
}

const int & cCtrlTimeCompens::NbMax()const 
{
   return mNbMax;
}


double & cCtrlTimeCompens::SeuilEvolMoy()
{
   return mSeuilEvolMoy;
}

const double & cCtrlTimeCompens::SeuilEvolMoy()const 
{
   return mSeuilEvolMoy;
}


cTplValGesInit< double > & cCtrlTimeCompens::SeuilEvolMax()
{
   return mSeuilEvolMax;
}

const cTplValGesInit< double > & cCtrlTimeCompens::SeuilEvolMax()const 
{
   return mSeuilEvolMax;
}


double & cCtrlTimeCompens::Mult()
{
   return AutoAdaptLVM().Val().Mult();
}

const double & cCtrlTimeCompens::Mult()const 
{
   return AutoAdaptLVM().Val().Mult();
}


cTplValGesInit< bool > & cCtrlTimeCompens::ModeMin()
{
   return AutoAdaptLVM().Val().ModeMin();
}

const cTplValGesInit< bool > & cCtrlTimeCompens::ModeMin()const 
{
   return AutoAdaptLVM().Val().ModeMin();
}


cTplValGesInit< cAutoAdaptLVM > & cCtrlTimeCompens::AutoAdaptLVM()
{
   return mAutoAdaptLVM;
}

const cTplValGesInit< cAutoAdaptLVM > & cCtrlTimeCompens::AutoAdaptLVM()const 
{
   return mAutoAdaptLVM;
}

void  BinaryUnDumpFromFile(cCtrlTimeCompens & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NbMin().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NbMin().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NbMin().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.NbMax(),aFp);
    BinaryUnDumpFromFile(anObj.SeuilEvolMoy(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SeuilEvolMax().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SeuilEvolMax().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SeuilEvolMax().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.AutoAdaptLVM().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.AutoAdaptLVM().ValForcedForUnUmp(),aFp);
        }
        else  anObj.AutoAdaptLVM().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cCtrlTimeCompens & anObj)
{
    BinaryDumpInFile(aFp,anObj.NbMin().IsInit());
    if (anObj.NbMin().IsInit()) BinaryDumpInFile(aFp,anObj.NbMin().Val());
    BinaryDumpInFile(aFp,anObj.NbMax());
    BinaryDumpInFile(aFp,anObj.SeuilEvolMoy());
    BinaryDumpInFile(aFp,anObj.SeuilEvolMax().IsInit());
    if (anObj.SeuilEvolMax().IsInit()) BinaryDumpInFile(aFp,anObj.SeuilEvolMax().Val());
    BinaryDumpInFile(aFp,anObj.AutoAdaptLVM().IsInit());
    if (anObj.AutoAdaptLVM().IsInit()) BinaryDumpInFile(aFp,anObj.AutoAdaptLVM().Val());
}

cElXMLTree * ToXMLTree(const cCtrlTimeCompens & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"CtrlTimeCompens",eXMLBranche);
   if (anObj.NbMin().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NbMin"),anObj.NbMin().Val())->ReTagThis("NbMin"));
   aRes->AddFils(::ToXMLTree(std::string("NbMax"),anObj.NbMax())->ReTagThis("NbMax"));
   aRes->AddFils(::ToXMLTree(std::string("SeuilEvolMoy"),anObj.SeuilEvolMoy())->ReTagThis("SeuilEvolMoy"));
   if (anObj.SeuilEvolMax().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SeuilEvolMax"),anObj.SeuilEvolMax().Val())->ReTagThis("SeuilEvolMax"));
   if (anObj.AutoAdaptLVM().IsInit())
      aRes->AddFils(ToXMLTree(anObj.AutoAdaptLVM().Val())->ReTagThis("AutoAdaptLVM"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cCtrlTimeCompens & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.NbMin(),aTree->Get("NbMin",1),int(0)); //tototo 

   xml_init(anObj.NbMax(),aTree->Get("NbMax",1)); //tototo 

   xml_init(anObj.SeuilEvolMoy(),aTree->Get("SeuilEvolMoy",1)); //tototo 

   xml_init(anObj.SeuilEvolMax(),aTree->Get("SeuilEvolMax",1)); //tototo 

   xml_init(anObj.AutoAdaptLVM(),aTree->Get("AutoAdaptLVM",1)); //tototo 
}

std::string  Mangling( cCtrlTimeCompens *) {return "FCB58A56BF9443A5FD3F";};


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

void  BinaryUnDumpFromFile(cPose2Init & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.ProfMin(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Show().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Show().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Show().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.StepComplemAuto().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.StepComplemAuto().ValForcedForUnUmp(),aFp);
        }
        else  anObj.StepComplemAuto().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cPose2Init & anObj)
{
    BinaryDumpInFile(aFp,anObj.ProfMin());
    BinaryDumpInFile(aFp,anObj.Show().IsInit());
    if (anObj.Show().IsInit()) BinaryDumpInFile(aFp,anObj.Show().Val());
    BinaryDumpInFile(aFp,anObj.StepComplemAuto().IsInit());
    if (anObj.StepComplemAuto().IsInit()) BinaryDumpInFile(aFp,anObj.StepComplemAuto().Val());
}

cElXMLTree * ToXMLTree(const cPose2Init & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"Pose2Init",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("ProfMin"),anObj.ProfMin())->ReTagThis("ProfMin"));
   if (anObj.Show().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Show"),anObj.Show().Val())->ReTagThis("Show"));
   if (anObj.StepComplemAuto().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("StepComplemAuto"),anObj.StepComplemAuto().Val())->ReTagThis("StepComplemAuto"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cPose2Init & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.ProfMin(),aTree->Get("ProfMin",1)); //tototo 

   xml_init(anObj.Show(),aTree->Get("Show",1),bool(false)); //tototo 

   xml_init(anObj.StepComplemAuto(),aTree->Get("StepComplemAuto",1),int(0)); //tototo 
}

std::string  Mangling( cPose2Init *) {return "4027240DF6C6979DFABF";};


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

void  BinaryUnDumpFromFile(cSetRayMaxUtileCalib & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Name(),aFp);
    BinaryUnDumpFromFile(anObj.Ray(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.IsRelatifDiag().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.IsRelatifDiag().ValForcedForUnUmp(),aFp);
        }
        else  anObj.IsRelatifDiag().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ApplyOnlyFE().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ApplyOnlyFE().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ApplyOnlyFE().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cSetRayMaxUtileCalib & anObj)
{
    BinaryDumpInFile(aFp,anObj.Name());
    BinaryDumpInFile(aFp,anObj.Ray());
    BinaryDumpInFile(aFp,anObj.IsRelatifDiag().IsInit());
    if (anObj.IsRelatifDiag().IsInit()) BinaryDumpInFile(aFp,anObj.IsRelatifDiag().Val());
    BinaryDumpInFile(aFp,anObj.ApplyOnlyFE().IsInit());
    if (anObj.ApplyOnlyFE().IsInit()) BinaryDumpInFile(aFp,anObj.ApplyOnlyFE().Val());
}

cElXMLTree * ToXMLTree(const cSetRayMaxUtileCalib & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"SetRayMaxUtileCalib",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Name"),anObj.Name())->ReTagThis("Name"));
   aRes->AddFils(::ToXMLTree(std::string("Ray"),anObj.Ray())->ReTagThis("Ray"));
   if (anObj.IsRelatifDiag().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("IsRelatifDiag"),anObj.IsRelatifDiag().Val())->ReTagThis("IsRelatifDiag"));
   if (anObj.ApplyOnlyFE().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ApplyOnlyFE"),anObj.ApplyOnlyFE().Val())->ReTagThis("ApplyOnlyFE"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cSetRayMaxUtileCalib & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.Name(),aTree->Get("Name",1)); //tototo 

   xml_init(anObj.Ray(),aTree->Get("Ray",1)); //tototo 

   xml_init(anObj.IsRelatifDiag(),aTree->Get("IsRelatifDiag",1),bool(false)); //tototo 

   xml_init(anObj.ApplyOnlyFE(),aTree->Get("ApplyOnlyFE",1),bool(false)); //tototo 
}

std::string  Mangling( cSetRayMaxUtileCalib *) {return "1CCA751716D41980FF3F";};


cTplValGesInit< std::string > & cBascOnCentre::PoseCentrale()
{
   return mPoseCentrale;
}

const cTplValGesInit< std::string > & cBascOnCentre::PoseCentrale()const 
{
   return mPoseCentrale;
}


cTplValGesInit< bool > & cBascOnCentre::EstimateSpeed()
{
   return mEstimateSpeed;
}

const cTplValGesInit< bool > & cBascOnCentre::EstimateSpeed()const 
{
   return mEstimateSpeed;
}


cTplValGesInit< double > & cBascOnCentre::ForceVertical()
{
   return mForceVertical;
}

const cTplValGesInit< double > & cBascOnCentre::ForceVertical()const 
{
   return mForceVertical;
}

void  BinaryUnDumpFromFile(cBascOnCentre & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.PoseCentrale().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.PoseCentrale().ValForcedForUnUmp(),aFp);
        }
        else  anObj.PoseCentrale().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.EstimateSpeed().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.EstimateSpeed().ValForcedForUnUmp(),aFp);
        }
        else  anObj.EstimateSpeed().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ForceVertical().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ForceVertical().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ForceVertical().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cBascOnCentre & anObj)
{
    BinaryDumpInFile(aFp,anObj.PoseCentrale().IsInit());
    if (anObj.PoseCentrale().IsInit()) BinaryDumpInFile(aFp,anObj.PoseCentrale().Val());
    BinaryDumpInFile(aFp,anObj.EstimateSpeed().IsInit());
    if (anObj.EstimateSpeed().IsInit()) BinaryDumpInFile(aFp,anObj.EstimateSpeed().Val());
    BinaryDumpInFile(aFp,anObj.ForceVertical().IsInit());
    if (anObj.ForceVertical().IsInit()) BinaryDumpInFile(aFp,anObj.ForceVertical().Val());
}

cElXMLTree * ToXMLTree(const cBascOnCentre & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"BascOnCentre",eXMLBranche);
   if (anObj.PoseCentrale().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PoseCentrale"),anObj.PoseCentrale().Val())->ReTagThis("PoseCentrale"));
   if (anObj.EstimateSpeed().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("EstimateSpeed"),anObj.EstimateSpeed().Val())->ReTagThis("EstimateSpeed"));
   if (anObj.ForceVertical().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ForceVertical"),anObj.ForceVertical().Val())->ReTagThis("ForceVertical"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cBascOnCentre & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.PoseCentrale(),aTree->Get("PoseCentrale",1)); //tototo 

   xml_init(anObj.EstimateSpeed(),aTree->Get("EstimateSpeed",1),bool(false)); //tototo 

   xml_init(anObj.ForceVertical(),aTree->Get("ForceVertical",1)); //tototo 
}

std::string  Mangling( cBascOnCentre *) {return "7B6AAFD286F4B990FD3F";};


std::string & cBascOnAppuis::NameRef()
{
   return mNameRef;
}

const std::string & cBascOnAppuis::NameRef()const 
{
   return mNameRef;
}

void  BinaryUnDumpFromFile(cBascOnAppuis & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.NameRef(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cBascOnAppuis & anObj)
{
    BinaryDumpInFile(aFp,anObj.NameRef());
}

cElXMLTree * ToXMLTree(const cBascOnAppuis & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"BascOnAppuis",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("NameRef"),anObj.NameRef())->ReTagThis("NameRef"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cBascOnAppuis & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.NameRef(),aTree->Get("NameRef",1)); //tototo 
}

std::string  Mangling( cBascOnAppuis *) {return "8F6D4799BEDF989BFE3F";};


cTplValGesInit< int > & cAerialDeformNonLin::FlagX()
{
   return mFlagX;
}

const cTplValGesInit< int > & cAerialDeformNonLin::FlagX()const 
{
   return mFlagX;
}


cTplValGesInit< int > & cAerialDeformNonLin::FlagY()
{
   return mFlagY;
}

const cTplValGesInit< int > & cAerialDeformNonLin::FlagY()const 
{
   return mFlagY;
}


cTplValGesInit< int > & cAerialDeformNonLin::FlagZ()
{
   return mFlagZ;
}

const cTplValGesInit< int > & cAerialDeformNonLin::FlagZ()const 
{
   return mFlagZ;
}


cTplValGesInit< bool > & cAerialDeformNonLin::ForceTrueRot()
{
   return mForceTrueRot;
}

const cTplValGesInit< bool > & cAerialDeformNonLin::ForceTrueRot()const 
{
   return mForceTrueRot;
}


cTplValGesInit< std::string > & cAerialDeformNonLin::PattEstim()
{
   return mPattEstim;
}

const cTplValGesInit< std::string > & cAerialDeformNonLin::PattEstim()const 
{
   return mPattEstim;
}


cTplValGesInit< bool > & cAerialDeformNonLin::Show()
{
   return mShow;
}

const cTplValGesInit< bool > & cAerialDeformNonLin::Show()const 
{
   return mShow;
}

void  BinaryUnDumpFromFile(cAerialDeformNonLin & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.FlagX().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.FlagX().ValForcedForUnUmp(),aFp);
        }
        else  anObj.FlagX().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.FlagY().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.FlagY().ValForcedForUnUmp(),aFp);
        }
        else  anObj.FlagY().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.FlagZ().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.FlagZ().ValForcedForUnUmp(),aFp);
        }
        else  anObj.FlagZ().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ForceTrueRot().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ForceTrueRot().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ForceTrueRot().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.PattEstim().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.PattEstim().ValForcedForUnUmp(),aFp);
        }
        else  anObj.PattEstim().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Show().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Show().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Show().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cAerialDeformNonLin & anObj)
{
    BinaryDumpInFile(aFp,anObj.FlagX().IsInit());
    if (anObj.FlagX().IsInit()) BinaryDumpInFile(aFp,anObj.FlagX().Val());
    BinaryDumpInFile(aFp,anObj.FlagY().IsInit());
    if (anObj.FlagY().IsInit()) BinaryDumpInFile(aFp,anObj.FlagY().Val());
    BinaryDumpInFile(aFp,anObj.FlagZ().IsInit());
    if (anObj.FlagZ().IsInit()) BinaryDumpInFile(aFp,anObj.FlagZ().Val());
    BinaryDumpInFile(aFp,anObj.ForceTrueRot().IsInit());
    if (anObj.ForceTrueRot().IsInit()) BinaryDumpInFile(aFp,anObj.ForceTrueRot().Val());
    BinaryDumpInFile(aFp,anObj.PattEstim().IsInit());
    if (anObj.PattEstim().IsInit()) BinaryDumpInFile(aFp,anObj.PattEstim().Val());
    BinaryDumpInFile(aFp,anObj.Show().IsInit());
    if (anObj.Show().IsInit()) BinaryDumpInFile(aFp,anObj.Show().Val());
}

cElXMLTree * ToXMLTree(const cAerialDeformNonLin & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"AerialDeformNonLin",eXMLBranche);
   if (anObj.FlagX().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("FlagX"),anObj.FlagX().Val())->ReTagThis("FlagX"));
   if (anObj.FlagY().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("FlagY"),anObj.FlagY().Val())->ReTagThis("FlagY"));
   if (anObj.FlagZ().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("FlagZ"),anObj.FlagZ().Val())->ReTagThis("FlagZ"));
   if (anObj.ForceTrueRot().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ForceTrueRot"),anObj.ForceTrueRot().Val())->ReTagThis("ForceTrueRot"));
   if (anObj.PattEstim().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PattEstim"),anObj.PattEstim().Val())->ReTagThis("PattEstim"));
   if (anObj.Show().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Show"),anObj.Show().Val())->ReTagThis("Show"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cAerialDeformNonLin & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.FlagX(),aTree->Get("FlagX",1),int(63)); //tototo 

   xml_init(anObj.FlagY(),aTree->Get("FlagY",1),int(63)); //tototo 

   xml_init(anObj.FlagZ(),aTree->Get("FlagZ",1),int(63)); //tototo 

   xml_init(anObj.ForceTrueRot(),aTree->Get("ForceTrueRot",1),bool(true)); //tototo 

   xml_init(anObj.PattEstim(),aTree->Get("PattEstim",1),std::string(".*")); //tototo 

   xml_init(anObj.Show(),aTree->Get("Show",1),bool(false)); //tototo 
}

std::string  Mangling( cAerialDeformNonLin *) {return "D4BF4C04AB361D9AFDBF";};


cTplValGesInit< std::string > & cBasculeOnPoints::ForceSol()
{
   return mForceSol;
}

const cTplValGesInit< std::string > & cBasculeOnPoints::ForceSol()const 
{
   return mForceSol;
}


cTplValGesInit< std::string > & cBasculeOnPoints::PoseCentrale()
{
   return BascOnCentre().Val().PoseCentrale();
}

const cTplValGesInit< std::string > & cBasculeOnPoints::PoseCentrale()const 
{
   return BascOnCentre().Val().PoseCentrale();
}


cTplValGesInit< bool > & cBasculeOnPoints::EstimateSpeed()
{
   return BascOnCentre().Val().EstimateSpeed();
}

const cTplValGesInit< bool > & cBasculeOnPoints::EstimateSpeed()const 
{
   return BascOnCentre().Val().EstimateSpeed();
}


cTplValGesInit< double > & cBasculeOnPoints::ForceVertical()
{
   return BascOnCentre().Val().ForceVertical();
}

const cTplValGesInit< double > & cBasculeOnPoints::ForceVertical()const 
{
   return BascOnCentre().Val().ForceVertical();
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


cTplValGesInit< cAerialDeformNonLin > & cBasculeOnPoints::AerialDeformNonLin()
{
   return mAerialDeformNonLin;
}

const cTplValGesInit< cAerialDeformNonLin > & cBasculeOnPoints::AerialDeformNonLin()const 
{
   return mAerialDeformNonLin;
}


cTplValGesInit< std::string > & cBasculeOnPoints::NameExport()
{
   return mNameExport;
}

const cTplValGesInit< std::string > & cBasculeOnPoints::NameExport()const 
{
   return mNameExport;
}

void  BinaryUnDumpFromFile(cBasculeOnPoints & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ForceSol().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ForceSol().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ForceSol().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.BascOnCentre().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.BascOnCentre().ValForcedForUnUmp(),aFp);
        }
        else  anObj.BascOnCentre().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.BascOnAppuis().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.BascOnAppuis().ValForcedForUnUmp(),aFp);
        }
        else  anObj.BascOnAppuis().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ModeL2().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ModeL2().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ModeL2().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.AerialDeformNonLin().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.AerialDeformNonLin().ValForcedForUnUmp(),aFp);
        }
        else  anObj.AerialDeformNonLin().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NameExport().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NameExport().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NameExport().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cBasculeOnPoints & anObj)
{
    BinaryDumpInFile(aFp,anObj.ForceSol().IsInit());
    if (anObj.ForceSol().IsInit()) BinaryDumpInFile(aFp,anObj.ForceSol().Val());
    BinaryDumpInFile(aFp,anObj.BascOnCentre().IsInit());
    if (anObj.BascOnCentre().IsInit()) BinaryDumpInFile(aFp,anObj.BascOnCentre().Val());
    BinaryDumpInFile(aFp,anObj.BascOnAppuis().IsInit());
    if (anObj.BascOnAppuis().IsInit()) BinaryDumpInFile(aFp,anObj.BascOnAppuis().Val());
    BinaryDumpInFile(aFp,anObj.ModeL2().IsInit());
    if (anObj.ModeL2().IsInit()) BinaryDumpInFile(aFp,anObj.ModeL2().Val());
    BinaryDumpInFile(aFp,anObj.AerialDeformNonLin().IsInit());
    if (anObj.AerialDeformNonLin().IsInit()) BinaryDumpInFile(aFp,anObj.AerialDeformNonLin().Val());
    BinaryDumpInFile(aFp,anObj.NameExport().IsInit());
    if (anObj.NameExport().IsInit()) BinaryDumpInFile(aFp,anObj.NameExport().Val());
}

cElXMLTree * ToXMLTree(const cBasculeOnPoints & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"BasculeOnPoints",eXMLBranche);
   if (anObj.ForceSol().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ForceSol"),anObj.ForceSol().Val())->ReTagThis("ForceSol"));
   if (anObj.BascOnCentre().IsInit())
      aRes->AddFils(ToXMLTree(anObj.BascOnCentre().Val())->ReTagThis("BascOnCentre"));
   if (anObj.BascOnAppuis().IsInit())
      aRes->AddFils(ToXMLTree(anObj.BascOnAppuis().Val())->ReTagThis("BascOnAppuis"));
   if (anObj.ModeL2().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ModeL2"),anObj.ModeL2().Val())->ReTagThis("ModeL2"));
   if (anObj.AerialDeformNonLin().IsInit())
      aRes->AddFils(ToXMLTree(anObj.AerialDeformNonLin().Val())->ReTagThis("AerialDeformNonLin"));
   if (anObj.NameExport().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NameExport"),anObj.NameExport().Val())->ReTagThis("NameExport"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cBasculeOnPoints & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.ForceSol(),aTree->Get("ForceSol",1)); //tototo 

   xml_init(anObj.BascOnCentre(),aTree->Get("BascOnCentre",1)); //tototo 

   xml_init(anObj.BascOnAppuis(),aTree->Get("BascOnAppuis",1)); //tototo 

   xml_init(anObj.ModeL2(),aTree->Get("ModeL2",1),bool(true)); //tototo 

   xml_init(anObj.AerialDeformNonLin(),aTree->Get("AerialDeformNonLin",1)); //tototo 

   xml_init(anObj.NameExport(),aTree->Get("NameExport",1)); //tototo 
}

std::string  Mangling( cBasculeOnPoints *) {return "BA355CF08CAA7CABFE3F";};


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

void  BinaryUnDumpFromFile(cOrientInPlane & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.DistFixEch().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.DistFixEch().ValForcedForUnUmp(),aFp);
        }
        else  anObj.DistFixEch().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.FileMesures(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.AlignOn().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.AlignOn().ValForcedForUnUmp(),aFp);
        }
        else  anObj.AlignOn().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cOrientInPlane & anObj)
{
    BinaryDumpInFile(aFp,anObj.DistFixEch().IsInit());
    if (anObj.DistFixEch().IsInit()) BinaryDumpInFile(aFp,anObj.DistFixEch().Val());
    BinaryDumpInFile(aFp,anObj.FileMesures());
    BinaryDumpInFile(aFp,anObj.AlignOn().IsInit());
    if (anObj.AlignOn().IsInit()) BinaryDumpInFile(aFp,anObj.AlignOn().Val());
}

cElXMLTree * ToXMLTree(const cOrientInPlane & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"OrientInPlane",eXMLBranche);
   if (anObj.DistFixEch().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DistFixEch"),anObj.DistFixEch().Val())->ReTagThis("DistFixEch"));
   aRes->AddFils(::ToXMLTree(std::string("FileMesures"),anObj.FileMesures())->ReTagThis("FileMesures"));
   if (anObj.AlignOn().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("AlignOn"),anObj.AlignOn().Val())->ReTagThis("AlignOn"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cOrientInPlane & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.DistFixEch(),aTree->Get("DistFixEch",1)); //tototo 

   xml_init(anObj.FileMesures(),aTree->Get("FileMesures",1)); //tototo 

   xml_init(anObj.AlignOn(),aTree->Get("AlignOn",1),std::string("ki")); //tototo 
}

std::string  Mangling( cOrientInPlane *) {return "BD95D46ED388EDA6FE3F";};


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

void  BinaryUnDumpFromFile(cBasculeLiaisonOnPlan & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.EstimPl(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.OrientInPlane().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.OrientInPlane().ValForcedForUnUmp(),aFp);
        }
        else  anObj.OrientInPlane().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cBasculeLiaisonOnPlan & anObj)
{
    BinaryDumpInFile(aFp,anObj.EstimPl());
    BinaryDumpInFile(aFp,anObj.OrientInPlane().IsInit());
    if (anObj.OrientInPlane().IsInit()) BinaryDumpInFile(aFp,anObj.OrientInPlane().Val());
}

cElXMLTree * ToXMLTree(const cBasculeLiaisonOnPlan & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"BasculeLiaisonOnPlan",eXMLBranche);
   aRes->AddFils(ToXMLTree(anObj.EstimPl())->ReTagThis("EstimPl"));
   if (anObj.OrientInPlane().IsInit())
      aRes->AddFils(ToXMLTree(anObj.OrientInPlane().Val())->ReTagThis("OrientInPlane"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cBasculeLiaisonOnPlan & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.EstimPl(),aTree->Get("EstimPl",1)); //tototo 

   xml_init(anObj.OrientInPlane(),aTree->Get("OrientInPlane",1)); //tototo 
}

std::string  Mangling( cBasculeLiaisonOnPlan *) {return "FE56783381A4FFB0FDBF";};


cTplValGesInit< std::string > & cModeBascule::ForceSol()
{
   return BasculeOnPoints().Val().ForceSol();
}

const cTplValGesInit< std::string > & cModeBascule::ForceSol()const 
{
   return BasculeOnPoints().Val().ForceSol();
}


cTplValGesInit< std::string > & cModeBascule::PoseCentrale()
{
   return BasculeOnPoints().Val().BascOnCentre().Val().PoseCentrale();
}

const cTplValGesInit< std::string > & cModeBascule::PoseCentrale()const 
{
   return BasculeOnPoints().Val().BascOnCentre().Val().PoseCentrale();
}


cTplValGesInit< bool > & cModeBascule::EstimateSpeed()
{
   return BasculeOnPoints().Val().BascOnCentre().Val().EstimateSpeed();
}

const cTplValGesInit< bool > & cModeBascule::EstimateSpeed()const 
{
   return BasculeOnPoints().Val().BascOnCentre().Val().EstimateSpeed();
}


cTplValGesInit< double > & cModeBascule::ForceVertical()
{
   return BasculeOnPoints().Val().BascOnCentre().Val().ForceVertical();
}

const cTplValGesInit< double > & cModeBascule::ForceVertical()const 
{
   return BasculeOnPoints().Val().BascOnCentre().Val().ForceVertical();
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


cTplValGesInit< cAerialDeformNonLin > & cModeBascule::AerialDeformNonLin()
{
   return BasculeOnPoints().Val().AerialDeformNonLin();
}

const cTplValGesInit< cAerialDeformNonLin > & cModeBascule::AerialDeformNonLin()const 
{
   return BasculeOnPoints().Val().AerialDeformNonLin();
}


cTplValGesInit< std::string > & cModeBascule::NameExport()
{
   return BasculeOnPoints().Val().NameExport();
}

const cTplValGesInit< std::string > & cModeBascule::NameExport()const 
{
   return BasculeOnPoints().Val().NameExport();
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

void  BinaryUnDumpFromFile(cModeBascule & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.BasculeOnPoints().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.BasculeOnPoints().ValForcedForUnUmp(),aFp);
        }
        else  anObj.BasculeOnPoints().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.BasculeLiaisonOnPlan().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.BasculeLiaisonOnPlan().ValForcedForUnUmp(),aFp);
        }
        else  anObj.BasculeLiaisonOnPlan().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cModeBascule & anObj)
{
    BinaryDumpInFile(aFp,anObj.BasculeOnPoints().IsInit());
    if (anObj.BasculeOnPoints().IsInit()) BinaryDumpInFile(aFp,anObj.BasculeOnPoints().Val());
    BinaryDumpInFile(aFp,anObj.BasculeLiaisonOnPlan().IsInit());
    if (anObj.BasculeLiaisonOnPlan().IsInit()) BinaryDumpInFile(aFp,anObj.BasculeLiaisonOnPlan().Val());
}

cElXMLTree * ToXMLTree(const cModeBascule & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ModeBascule",eXMLBranche);
   if (anObj.BasculeOnPoints().IsInit())
      aRes->AddFils(ToXMLTree(anObj.BasculeOnPoints().Val())->ReTagThis("BasculeOnPoints"));
   if (anObj.BasculeLiaisonOnPlan().IsInit())
      aRes->AddFils(ToXMLTree(anObj.BasculeLiaisonOnPlan().Val())->ReTagThis("BasculeLiaisonOnPlan"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cModeBascule & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.BasculeOnPoints(),aTree->Get("BasculeOnPoints",1)); //tototo 

   xml_init(anObj.BasculeLiaisonOnPlan(),aTree->Get("BasculeLiaisonOnPlan",1)); //tototo 
}

std::string  Mangling( cModeBascule *) {return "9A5B68592A82D3B7FE3F";};


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


cTplValGesInit< std::string > & cBasculeOrientation::FileExportDir()
{
   return mFileExportDir;
}

const cTplValGesInit< std::string > & cBasculeOrientation::FileExportDir()const 
{
   return mFileExportDir;
}


cTplValGesInit< std::string > & cBasculeOrientation::FileExportInv()
{
   return mFileExportInv;
}

const cTplValGesInit< std::string > & cBasculeOrientation::FileExportInv()const 
{
   return mFileExportInv;
}


cTplValGesInit< std::string > & cBasculeOrientation::ForceSol()
{
   return ModeBascule().BasculeOnPoints().Val().ForceSol();
}

const cTplValGesInit< std::string > & cBasculeOrientation::ForceSol()const 
{
   return ModeBascule().BasculeOnPoints().Val().ForceSol();
}


cTplValGesInit< std::string > & cBasculeOrientation::PoseCentrale()
{
   return ModeBascule().BasculeOnPoints().Val().BascOnCentre().Val().PoseCentrale();
}

const cTplValGesInit< std::string > & cBasculeOrientation::PoseCentrale()const 
{
   return ModeBascule().BasculeOnPoints().Val().BascOnCentre().Val().PoseCentrale();
}


cTplValGesInit< bool > & cBasculeOrientation::EstimateSpeed()
{
   return ModeBascule().BasculeOnPoints().Val().BascOnCentre().Val().EstimateSpeed();
}

const cTplValGesInit< bool > & cBasculeOrientation::EstimateSpeed()const 
{
   return ModeBascule().BasculeOnPoints().Val().BascOnCentre().Val().EstimateSpeed();
}


cTplValGesInit< double > & cBasculeOrientation::ForceVertical()
{
   return ModeBascule().BasculeOnPoints().Val().BascOnCentre().Val().ForceVertical();
}

const cTplValGesInit< double > & cBasculeOrientation::ForceVertical()const 
{
   return ModeBascule().BasculeOnPoints().Val().BascOnCentre().Val().ForceVertical();
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


cTplValGesInit< cAerialDeformNonLin > & cBasculeOrientation::AerialDeformNonLin()
{
   return ModeBascule().BasculeOnPoints().Val().AerialDeformNonLin();
}

const cTplValGesInit< cAerialDeformNonLin > & cBasculeOrientation::AerialDeformNonLin()const 
{
   return ModeBascule().BasculeOnPoints().Val().AerialDeformNonLin();
}


cTplValGesInit< std::string > & cBasculeOrientation::NameExport()
{
   return ModeBascule().BasculeOnPoints().Val().NameExport();
}

const cTplValGesInit< std::string > & cBasculeOrientation::NameExport()const 
{
   return ModeBascule().BasculeOnPoints().Val().NameExport();
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

void  BinaryUnDumpFromFile(cBasculeOrientation & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.AfterCompens().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.AfterCompens().ValForcedForUnUmp(),aFp);
        }
        else  anObj.AfterCompens().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.PatternNameApply().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.PatternNameApply().ValForcedForUnUmp(),aFp);
        }
        else  anObj.PatternNameApply().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.PatternNameEstim().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.PatternNameEstim().ValForcedForUnUmp(),aFp);
        }
        else  anObj.PatternNameEstim().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.FileExportDir().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.FileExportDir().ValForcedForUnUmp(),aFp);
        }
        else  anObj.FileExportDir().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.FileExportInv().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.FileExportInv().ValForcedForUnUmp(),aFp);
        }
        else  anObj.FileExportInv().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.ModeBascule(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cBasculeOrientation & anObj)
{
    BinaryDumpInFile(aFp,anObj.AfterCompens().IsInit());
    if (anObj.AfterCompens().IsInit()) BinaryDumpInFile(aFp,anObj.AfterCompens().Val());
    BinaryDumpInFile(aFp,anObj.PatternNameApply().IsInit());
    if (anObj.PatternNameApply().IsInit()) BinaryDumpInFile(aFp,anObj.PatternNameApply().Val());
    BinaryDumpInFile(aFp,anObj.PatternNameEstim().IsInit());
    if (anObj.PatternNameEstim().IsInit()) BinaryDumpInFile(aFp,anObj.PatternNameEstim().Val());
    BinaryDumpInFile(aFp,anObj.FileExportDir().IsInit());
    if (anObj.FileExportDir().IsInit()) BinaryDumpInFile(aFp,anObj.FileExportDir().Val());
    BinaryDumpInFile(aFp,anObj.FileExportInv().IsInit());
    if (anObj.FileExportInv().IsInit()) BinaryDumpInFile(aFp,anObj.FileExportInv().Val());
    BinaryDumpInFile(aFp,anObj.ModeBascule());
}

cElXMLTree * ToXMLTree(const cBasculeOrientation & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"BasculeOrientation",eXMLBranche);
   if (anObj.AfterCompens().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("AfterCompens"),anObj.AfterCompens().Val())->ReTagThis("AfterCompens"));
   if (anObj.PatternNameApply().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PatternNameApply"),anObj.PatternNameApply().Val())->ReTagThis("PatternNameApply"));
   if (anObj.PatternNameEstim().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PatternNameEstim"),anObj.PatternNameEstim().Val())->ReTagThis("PatternNameEstim"));
   if (anObj.FileExportDir().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("FileExportDir"),anObj.FileExportDir().Val())->ReTagThis("FileExportDir"));
   if (anObj.FileExportInv().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("FileExportInv"),anObj.FileExportInv().Val())->ReTagThis("FileExportInv"));
   aRes->AddFils(ToXMLTree(anObj.ModeBascule())->ReTagThis("ModeBascule"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cBasculeOrientation & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.AfterCompens(),aTree->Get("AfterCompens",1),bool(true)); //tototo 

   xml_init(anObj.PatternNameApply(),aTree->Get("PatternNameApply",1),std::string(".*")); //tototo 

   xml_init(anObj.PatternNameEstim(),aTree->Get("PatternNameEstim",1),std::string(".*")); //tototo 

   xml_init(anObj.FileExportDir(),aTree->Get("FileExportDir",1)); //tototo 

   xml_init(anObj.FileExportInv(),aTree->Get("FileExportInv",1)); //tototo 

   xml_init(anObj.ModeBascule(),aTree->Get("ModeBascule",1)); //tototo 
}

std::string  Mangling( cBasculeOrientation *) {return "0405942329320F9CFE3F";};


std::vector< cAperoPointeStereo > & cStereoFE::HomFE()
{
   return mHomFE;
}

const std::vector< cAperoPointeStereo > & cStereoFE::HomFE()const 
{
   return mHomFE;
}

void  BinaryUnDumpFromFile(cStereoFE & anObj,ELISE_fp & aFp)
{
   { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cAperoPointeStereo aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.HomFE().push_back(aVal);
        }
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cStereoFE & anObj)
{
    BinaryDumpInFile(aFp,(int)anObj.HomFE().size());
    for(  std::vector< cAperoPointeStereo >::const_iterator iT=anObj.HomFE().begin();
         iT!=anObj.HomFE().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
}

cElXMLTree * ToXMLTree(const cStereoFE & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"StereoFE",eXMLBranche);
  for
  (       std::vector< cAperoPointeStereo >::const_iterator it=anObj.HomFE().begin();
      it !=anObj.HomFE().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("HomFE"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cStereoFE & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.HomFE(),aTree->GetAll("HomFE",false,1));
}

std::string  Mangling( cStereoFE *) {return "ED1462C47D4A1880FE3F";};


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

void  BinaryUnDumpFromFile(cModeFE & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.StereoFE().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.StereoFE().ValForcedForUnUmp(),aFp);
        }
        else  anObj.StereoFE().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.FEFromFile().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.FEFromFile().ValForcedForUnUmp(),aFp);
        }
        else  anObj.FEFromFile().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cModeFE & anObj)
{
    BinaryDumpInFile(aFp,anObj.StereoFE().IsInit());
    if (anObj.StereoFE().IsInit()) BinaryDumpInFile(aFp,anObj.StereoFE().Val());
    BinaryDumpInFile(aFp,anObj.FEFromFile().IsInit());
    if (anObj.FEFromFile().IsInit()) BinaryDumpInFile(aFp,anObj.FEFromFile().Val());
}

cElXMLTree * ToXMLTree(const cModeFE & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ModeFE",eXMLBranche);
   if (anObj.StereoFE().IsInit())
      aRes->AddFils(ToXMLTree(anObj.StereoFE().Val())->ReTagThis("StereoFE"));
   if (anObj.FEFromFile().IsInit())
      aRes->AddFils(ToXMLTree(anObj.FEFromFile().Val())->ReTagThis("FEFromFile"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cModeFE & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.StereoFE(),aTree->Get("StereoFE",1)); //tototo 

   xml_init(anObj.FEFromFile(),aTree->Get("FEFromFile",1)); //tototo 
}

std::string  Mangling( cModeFE *) {return "7F4BA88F6AF9A59DFF3F";};


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

void  BinaryUnDumpFromFile(cFixeEchelle & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.ModeFE(),aFp);
    BinaryUnDumpFromFile(anObj.DistVraie(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cFixeEchelle & anObj)
{
    BinaryDumpInFile(aFp,anObj.ModeFE());
    BinaryDumpInFile(aFp,anObj.DistVraie());
}

cElXMLTree * ToXMLTree(const cFixeEchelle & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"FixeEchelle",eXMLBranche);
   aRes->AddFils(ToXMLTree(anObj.ModeFE())->ReTagThis("ModeFE"));
   aRes->AddFils(::ToXMLTree(std::string("DistVraie"),anObj.DistVraie())->ReTagThis("DistVraie"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cFixeEchelle & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.ModeFE(),aTree->Get("ModeFE",1)); //tototo 

   xml_init(anObj.DistVraie(),aTree->Get("DistVraie",1)); //tototo 
}

std::string  Mangling( cFixeEchelle *) {return "3A3FC6A63B959ED2FE3F";};


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

void  BinaryUnDumpFromFile(cHorFOP & anObj,ELISE_fp & aFp)
{
   { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cAperoPointeMono aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.VecFOH().push_back(aVal);
        }
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Z().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Z().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Z().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cHorFOP & anObj)
{
    BinaryDumpInFile(aFp,(int)anObj.VecFOH().size());
    for(  std::vector< cAperoPointeMono >::const_iterator iT=anObj.VecFOH().begin();
         iT!=anObj.VecFOH().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.Z().IsInit());
    if (anObj.Z().IsInit()) BinaryDumpInFile(aFp,anObj.Z().Val());
}

cElXMLTree * ToXMLTree(const cHorFOP & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"HorFOP",eXMLBranche);
  for
  (       std::vector< cAperoPointeMono >::const_iterator it=anObj.VecFOH().begin();
      it !=anObj.VecFOH().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("VecFOH"));
   if (anObj.Z().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Z"),anObj.Z().Val())->ReTagThis("Z"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cHorFOP & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.VecFOH(),aTree->GetAll("VecFOH",false,1));

   xml_init(anObj.Z(),aTree->Get("Z",1),double(0)); //tototo 
}

std::string  Mangling( cHorFOP *) {return "4E402B40CD5D1AAFFF3F";};


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

void  BinaryUnDumpFromFile(cModeFOP & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.HorFOP().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.HorFOP().ValForcedForUnUmp(),aFp);
        }
        else  anObj.HorFOP().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.HorFromFile().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.HorFromFile().ValForcedForUnUmp(),aFp);
        }
        else  anObj.HorFromFile().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cModeFOP & anObj)
{
    BinaryDumpInFile(aFp,anObj.HorFOP().IsInit());
    if (anObj.HorFOP().IsInit()) BinaryDumpInFile(aFp,anObj.HorFOP().Val());
    BinaryDumpInFile(aFp,anObj.HorFromFile().IsInit());
    if (anObj.HorFromFile().IsInit()) BinaryDumpInFile(aFp,anObj.HorFromFile().Val());
}

cElXMLTree * ToXMLTree(const cModeFOP & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ModeFOP",eXMLBranche);
   if (anObj.HorFOP().IsInit())
      aRes->AddFils(ToXMLTree(anObj.HorFOP().Val())->ReTagThis("HorFOP"));
   if (anObj.HorFromFile().IsInit())
      aRes->AddFils(ToXMLTree(anObj.HorFromFile().Val())->ReTagThis("HorFromFile"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cModeFOP & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.HorFOP(),aTree->Get("HorFOP",1)); //tototo 

   xml_init(anObj.HorFromFile(),aTree->Get("HorFromFile",1)); //tototo 
}

std::string  Mangling( cModeFOP *) {return "B66BAEFAB1650A8BFEBF";};


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

void  BinaryUnDumpFromFile(cFixeOrientPlane & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.ModeFOP(),aFp);
    BinaryUnDumpFromFile(anObj.Vecteur(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cFixeOrientPlane & anObj)
{
    BinaryDumpInFile(aFp,anObj.ModeFOP());
    BinaryDumpInFile(aFp,anObj.Vecteur());
}

cElXMLTree * ToXMLTree(const cFixeOrientPlane & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"FixeOrientPlane",eXMLBranche);
   aRes->AddFils(ToXMLTree(anObj.ModeFOP())->ReTagThis("ModeFOP"));
   aRes->AddFils(::ToXMLTree(std::string("Vecteur"),anObj.Vecteur())->ReTagThis("Vecteur"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cFixeOrientPlane & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.ModeFOP(),aTree->Get("ModeFOP",1)); //tototo 

   xml_init(anObj.Vecteur(),aTree->Get("Vecteur",1)); //tototo 
}

std::string  Mangling( cFixeOrientPlane *) {return "1A2A4DFF6CAFE1FCFD3F";};


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

void  BinaryUnDumpFromFile(cBlocBascule & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Pattern1(),aFp);
    BinaryUnDumpFromFile(anObj.Pattern2(),aFp);
    BinaryUnDumpFromFile(anObj.IdBdl(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cBlocBascule & anObj)
{
    BinaryDumpInFile(aFp,anObj.Pattern1());
    BinaryDumpInFile(aFp,anObj.Pattern2());
    BinaryDumpInFile(aFp,anObj.IdBdl());
}

cElXMLTree * ToXMLTree(const cBlocBascule & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"BlocBascule",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Pattern1"),anObj.Pattern1())->ReTagThis("Pattern1"));
   aRes->AddFils(::ToXMLTree(std::string("Pattern2"),anObj.Pattern2())->ReTagThis("Pattern2"));
   aRes->AddFils(::ToXMLTree(std::string("IdBdl"),anObj.IdBdl())->ReTagThis("IdBdl"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cBlocBascule & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.Pattern1(),aTree->Get("Pattern1",1)); //tototo 

   xml_init(anObj.Pattern2(),aTree->Get("Pattern2",1)); //tototo 

   xml_init(anObj.IdBdl(),aTree->Get("IdBdl",1)); //tototo 
}

std::string  Mangling( cBlocBascule *) {return "37738DE6BE7DDBC7FE3F";};


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

void  BinaryUnDumpFromFile(cMesureErreurTournante & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Periode(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NbTest().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NbTest().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NbTest().SetNoInit();
  } ;
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
             anObj.ApplyAppuis().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ApplyAppuis().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ApplyAppuis().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ApplyLiaisons().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ApplyLiaisons().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ApplyLiaisons().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cMesureErreurTournante & anObj)
{
    BinaryDumpInFile(aFp,anObj.Periode());
    BinaryDumpInFile(aFp,anObj.NbTest().IsInit());
    if (anObj.NbTest().IsInit()) BinaryDumpInFile(aFp,anObj.NbTest().Val());
    BinaryDumpInFile(aFp,anObj.NbIter().IsInit());
    if (anObj.NbIter().IsInit()) BinaryDumpInFile(aFp,anObj.NbIter().Val());
    BinaryDumpInFile(aFp,anObj.ApplyAppuis().IsInit());
    if (anObj.ApplyAppuis().IsInit()) BinaryDumpInFile(aFp,anObj.ApplyAppuis().Val());
    BinaryDumpInFile(aFp,anObj.ApplyLiaisons().IsInit());
    if (anObj.ApplyLiaisons().IsInit()) BinaryDumpInFile(aFp,anObj.ApplyLiaisons().Val());
}

cElXMLTree * ToXMLTree(const cMesureErreurTournante & anObj)
{
  XMLPushContext(anObj.mGXml);
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
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cMesureErreurTournante & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.Periode(),aTree->Get("Periode",1)); //tototo 

   xml_init(anObj.NbTest(),aTree->Get("NbTest",1)); //tototo 

   xml_init(anObj.NbIter(),aTree->Get("NbIter",1),int(4)); //tototo 

   xml_init(anObj.ApplyAppuis(),aTree->Get("ApplyAppuis",1),bool(true)); //tototo 

   xml_init(anObj.ApplyLiaisons(),aTree->Get("ApplyLiaisons",1),bool(false)); //tototo 
}

std::string  Mangling( cMesureErreurTournante *) {return "CE78A085F854EE9FFD3F";};


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

void  BinaryUnDumpFromFile(cContraintesCamerasInc & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.TolContrainte().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.TolContrainte().ValForcedForUnUmp(),aFp);
        }
        else  anObj.TolContrainte().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.PatternNameApply().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.PatternNameApply().ValForcedForUnUmp(),aFp);
        }
        else  anObj.PatternNameApply().SetNoInit();
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             eTypeContrainteCalibCamera aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.Val().push_back(aVal);
        }
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.PatternRefuteur().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.PatternRefuteur().ValForcedForUnUmp(),aFp);
        }
        else  anObj.PatternRefuteur().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cContraintesCamerasInc & anObj)
{
    BinaryDumpInFile(aFp,anObj.TolContrainte().IsInit());
    if (anObj.TolContrainte().IsInit()) BinaryDumpInFile(aFp,anObj.TolContrainte().Val());
    BinaryDumpInFile(aFp,anObj.PatternNameApply().IsInit());
    if (anObj.PatternNameApply().IsInit()) BinaryDumpInFile(aFp,anObj.PatternNameApply().Val());
    BinaryDumpInFile(aFp,(int)anObj.Val().size());
    for(  std::list< eTypeContrainteCalibCamera >::const_iterator iT=anObj.Val().begin();
         iT!=anObj.Val().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.PatternRefuteur().IsInit());
    if (anObj.PatternRefuteur().IsInit()) BinaryDumpInFile(aFp,anObj.PatternRefuteur().Val());
}

cElXMLTree * ToXMLTree(const cContraintesCamerasInc & anObj)
{
  XMLPushContext(anObj.mGXml);
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
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cContraintesCamerasInc & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.TolContrainte(),aTree->Get("TolContrainte",1),double(-1)); //tototo 

   xml_init(anObj.PatternNameApply(),aTree->Get("PatternNameApply",1),std::string(".*")); //tototo 

   xml_init(anObj.Val(),aTree->GetAll("Val",false,1));

   xml_init(anObj.PatternRefuteur(),aTree->Get("PatternRefuteur",1)); //tototo 
}

std::string  Mangling( cContraintesCamerasInc *) {return "68D6E37FEA65F096FE3F";};


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

void  BinaryUnDumpFromFile(cContraintesPoses & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ByPattern().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ByPattern().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ByPattern().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.PatternRefuteur().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.PatternRefuteur().ValForcedForUnUmp(),aFp);
        }
        else  anObj.PatternRefuteur().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.TolAng().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.TolAng().ValForcedForUnUmp(),aFp);
        }
        else  anObj.TolAng().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.TolCoord().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.TolCoord().ValForcedForUnUmp(),aFp);
        }
        else  anObj.TolCoord().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.NamePose(),aFp);
    BinaryUnDumpFromFile(anObj.Val(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.PoseRattachement().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.PoseRattachement().ValForcedForUnUmp(),aFp);
        }
        else  anObj.PoseRattachement().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cContraintesPoses & anObj)
{
    BinaryDumpInFile(aFp,anObj.ByPattern().IsInit());
    if (anObj.ByPattern().IsInit()) BinaryDumpInFile(aFp,anObj.ByPattern().Val());
    BinaryDumpInFile(aFp,anObj.PatternRefuteur().IsInit());
    if (anObj.PatternRefuteur().IsInit()) BinaryDumpInFile(aFp,anObj.PatternRefuteur().Val());
    BinaryDumpInFile(aFp,anObj.TolAng().IsInit());
    if (anObj.TolAng().IsInit()) BinaryDumpInFile(aFp,anObj.TolAng().Val());
    BinaryDumpInFile(aFp,anObj.TolCoord().IsInit());
    if (anObj.TolCoord().IsInit()) BinaryDumpInFile(aFp,anObj.TolCoord().Val());
    BinaryDumpInFile(aFp,anObj.NamePose());
    BinaryDumpInFile(aFp,anObj.Val());
    BinaryDumpInFile(aFp,anObj.PoseRattachement().IsInit());
    if (anObj.PoseRattachement().IsInit()) BinaryDumpInFile(aFp,anObj.PoseRattachement().Val());
}

cElXMLTree * ToXMLTree(const cContraintesPoses & anObj)
{
  XMLPushContext(anObj.mGXml);
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
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cContraintesPoses & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.ByPattern(),aTree->Get("ByPattern",1),bool(false)); //tototo 

   xml_init(anObj.PatternRefuteur(),aTree->Get("PatternRefuteur",1)); //tototo 

   xml_init(anObj.TolAng(),aTree->Get("TolAng",1),double(-1)); //tototo 

   xml_init(anObj.TolCoord(),aTree->Get("TolCoord",1),double(-1)); //tototo 

   xml_init(anObj.NamePose(),aTree->Get("NamePose",1)); //tototo 

   xml_init(anObj.Val(),aTree->Get("Val",1)); //tototo 

   xml_init(anObj.PoseRattachement(),aTree->Get("PoseRattachement",1)); //tototo 
}

std::string  Mangling( cContraintesPoses *) {return "B6172DE91F7E698BFE3F";};


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

void  BinaryUnDumpFromFile(cSectionContraintes & anObj,ELISE_fp & aFp)
{
   { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cContraintesCamerasInc aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.ContraintesCamerasInc().push_back(aVal);
        }
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cContraintesPoses aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.ContraintesPoses().push_back(aVal);
        }
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cSectionContraintes & anObj)
{
    BinaryDumpInFile(aFp,(int)anObj.ContraintesCamerasInc().size());
    for(  std::list< cContraintesCamerasInc >::const_iterator iT=anObj.ContraintesCamerasInc().begin();
         iT!=anObj.ContraintesCamerasInc().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,(int)anObj.ContraintesPoses().size());
    for(  std::list< cContraintesPoses >::const_iterator iT=anObj.ContraintesPoses().begin();
         iT!=anObj.ContraintesPoses().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
}

cElXMLTree * ToXMLTree(const cSectionContraintes & anObj)
{
  XMLPushContext(anObj.mGXml);
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
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cSectionContraintes & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.ContraintesCamerasInc(),aTree->GetAll("ContraintesCamerasInc",false,1));

   xml_init(anObj.ContraintesPoses(),aTree->GetAll("ContraintesPoses",false,1));
}

std::string  Mangling( cSectionContraintes *) {return "B80C02D83D46D5FEFE3F";};


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

void  BinaryUnDumpFromFile(cVisuPtsMult & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Cam1(),aFp);
    BinaryUnDumpFromFile(anObj.Id(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SzWPrinc().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SzWPrinc().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SzWPrinc().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SzWAux().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SzWAux().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SzWAux().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ZoomWAux().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ZoomWAux().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ZoomWAux().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.NbWAux(),aFp);
    BinaryUnDumpFromFile(anObj.AuxEnDessous(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.MaxDistReproj().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.MaxDistReproj().ValForcedForUnUmp(),aFp);
        }
        else  anObj.MaxDistReproj().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.MaxDistSift().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.MaxDistSift().ValForcedForUnUmp(),aFp);
        }
        else  anObj.MaxDistSift().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.MaxDistProjCorr().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.MaxDistProjCorr().ValForcedForUnUmp(),aFp);
        }
        else  anObj.MaxDistProjCorr().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SeuilCorrel().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SeuilCorrel().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SeuilCorrel().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cVisuPtsMult & anObj)
{
    BinaryDumpInFile(aFp,anObj.Cam1());
    BinaryDumpInFile(aFp,anObj.Id());
    BinaryDumpInFile(aFp,anObj.SzWPrinc().IsInit());
    if (anObj.SzWPrinc().IsInit()) BinaryDumpInFile(aFp,anObj.SzWPrinc().Val());
    BinaryDumpInFile(aFp,anObj.SzWAux().IsInit());
    if (anObj.SzWAux().IsInit()) BinaryDumpInFile(aFp,anObj.SzWAux().Val());
    BinaryDumpInFile(aFp,anObj.ZoomWAux().IsInit());
    if (anObj.ZoomWAux().IsInit()) BinaryDumpInFile(aFp,anObj.ZoomWAux().Val());
    BinaryDumpInFile(aFp,anObj.NbWAux());
    BinaryDumpInFile(aFp,anObj.AuxEnDessous());
    BinaryDumpInFile(aFp,anObj.MaxDistReproj().IsInit());
    if (anObj.MaxDistReproj().IsInit()) BinaryDumpInFile(aFp,anObj.MaxDistReproj().Val());
    BinaryDumpInFile(aFp,anObj.MaxDistSift().IsInit());
    if (anObj.MaxDistSift().IsInit()) BinaryDumpInFile(aFp,anObj.MaxDistSift().Val());
    BinaryDumpInFile(aFp,anObj.MaxDistProjCorr().IsInit());
    if (anObj.MaxDistProjCorr().IsInit()) BinaryDumpInFile(aFp,anObj.MaxDistProjCorr().Val());
    BinaryDumpInFile(aFp,anObj.SeuilCorrel().IsInit());
    if (anObj.SeuilCorrel().IsInit()) BinaryDumpInFile(aFp,anObj.SeuilCorrel().Val());
}

cElXMLTree * ToXMLTree(const cVisuPtsMult & anObj)
{
  XMLPushContext(anObj.mGXml);
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
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cVisuPtsMult & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

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

std::string  Mangling( cVisuPtsMult *) {return "3280A19A840098A0FF3F";};


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

void  BinaryUnDumpFromFile(cVerifAero & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.PatternApply(),aFp);
    BinaryUnDumpFromFile(anObj.IdBdLiaison(),aFp);
    BinaryUnDumpFromFile(anObj.Pond(),aFp);
    BinaryUnDumpFromFile(anObj.Prefixe(),aFp);
    BinaryUnDumpFromFile(anObj.TypeVerif(),aFp);
    BinaryUnDumpFromFile(anObj.SeuilTxt(),aFp);
    BinaryUnDumpFromFile(anObj.Resol(),aFp);
    BinaryUnDumpFromFile(anObj.PasR(),aFp);
    BinaryUnDumpFromFile(anObj.PasB(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cVerifAero & anObj)
{
    BinaryDumpInFile(aFp,anObj.PatternApply());
    BinaryDumpInFile(aFp,anObj.IdBdLiaison());
    BinaryDumpInFile(aFp,anObj.Pond());
    BinaryDumpInFile(aFp,anObj.Prefixe());
    BinaryDumpInFile(aFp,anObj.TypeVerif());
    BinaryDumpInFile(aFp,anObj.SeuilTxt());
    BinaryDumpInFile(aFp,anObj.Resol());
    BinaryDumpInFile(aFp,anObj.PasR());
    BinaryDumpInFile(aFp,anObj.PasB());
}

cElXMLTree * ToXMLTree(const cVerifAero & anObj)
{
  XMLPushContext(anObj.mGXml);
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
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cVerifAero & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

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

std::string  Mangling( cVerifAero *) {return "FCD29A8B984014A6FC3F";};


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

void  BinaryUnDumpFromFile(cGPtsTer_By_ImProf & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Origine(),aFp);
    BinaryUnDumpFromFile(anObj.Step(),aFp);
    BinaryUnDumpFromFile(anObj.NbPts(),aFp);
    BinaryUnDumpFromFile(anObj.OnGrid(),aFp);
    BinaryUnDumpFromFile(anObj.File(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.RandomizeInGrid().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.RandomizeInGrid().ValForcedForUnUmp(),aFp);
        }
        else  anObj.RandomizeInGrid().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ImMaitresse().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ImMaitresse().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ImMaitresse().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.DTMIsZ().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.DTMIsZ().ValForcedForUnUmp(),aFp);
        }
        else  anObj.DTMIsZ().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cGPtsTer_By_ImProf & anObj)
{
    BinaryDumpInFile(aFp,anObj.Origine());
    BinaryDumpInFile(aFp,anObj.Step());
    BinaryDumpInFile(aFp,anObj.NbPts());
    BinaryDumpInFile(aFp,anObj.OnGrid());
    BinaryDumpInFile(aFp,anObj.File());
    BinaryDumpInFile(aFp,anObj.RandomizeInGrid().IsInit());
    if (anObj.RandomizeInGrid().IsInit()) BinaryDumpInFile(aFp,anObj.RandomizeInGrid().Val());
    BinaryDumpInFile(aFp,anObj.ImMaitresse().IsInit());
    if (anObj.ImMaitresse().IsInit()) BinaryDumpInFile(aFp,anObj.ImMaitresse().Val());
    BinaryDumpInFile(aFp,anObj.DTMIsZ().IsInit());
    if (anObj.DTMIsZ().IsInit()) BinaryDumpInFile(aFp,anObj.DTMIsZ().Val());
}

cElXMLTree * ToXMLTree(const cGPtsTer_By_ImProf & anObj)
{
  XMLPushContext(anObj.mGXml);
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
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cGPtsTer_By_ImProf & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.Origine(),aTree->Get("Origine",1)); //tototo 

   xml_init(anObj.Step(),aTree->Get("Step",1)); //tototo 

   xml_init(anObj.NbPts(),aTree->Get("NbPts",1)); //tototo 

   xml_init(anObj.OnGrid(),aTree->Get("OnGrid",1)); //tototo 

   xml_init(anObj.File(),aTree->Get("File",1)); //tototo 

   xml_init(anObj.RandomizeInGrid(),aTree->Get("RandomizeInGrid",1),double(0.0)); //tototo 

   xml_init(anObj.ImMaitresse(),aTree->Get("ImMaitresse",1)); //tototo 

   xml_init(anObj.DTMIsZ(),aTree->Get("DTMIsZ",1),bool(true)); //tototo 
}

std::string  Mangling( cGPtsTer_By_ImProf *) {return "9C2C497A440C8F88FE3F";};


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

void  BinaryUnDumpFromFile(cGeneratePointsTerrains & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.GPtsTer_By_File().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.GPtsTer_By_File().ValForcedForUnUmp(),aFp);
        }
        else  anObj.GPtsTer_By_File().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.GPtsTer_By_ImProf().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.GPtsTer_By_ImProf().ValForcedForUnUmp(),aFp);
        }
        else  anObj.GPtsTer_By_ImProf().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cGeneratePointsTerrains & anObj)
{
    BinaryDumpInFile(aFp,anObj.GPtsTer_By_File().IsInit());
    if (anObj.GPtsTer_By_File().IsInit()) BinaryDumpInFile(aFp,anObj.GPtsTer_By_File().Val());
    BinaryDumpInFile(aFp,anObj.GPtsTer_By_ImProf().IsInit());
    if (anObj.GPtsTer_By_ImProf().IsInit()) BinaryDumpInFile(aFp,anObj.GPtsTer_By_ImProf().Val());
}

cElXMLTree * ToXMLTree(const cGeneratePointsTerrains & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"GeneratePointsTerrains",eXMLBranche);
   if (anObj.GPtsTer_By_File().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("GPtsTer_By_File"),anObj.GPtsTer_By_File().Val())->ReTagThis("GPtsTer_By_File"));
   if (anObj.GPtsTer_By_ImProf().IsInit())
      aRes->AddFils(ToXMLTree(anObj.GPtsTer_By_ImProf().Val())->ReTagThis("GPtsTer_By_ImProf"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cGeneratePointsTerrains & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.GPtsTer_By_File(),aTree->Get("GPtsTer_By_File",1)); //tototo 

   xml_init(anObj.GPtsTer_By_ImProf(),aTree->Get("GPtsTer_By_ImProf",1)); //tototo 
}

std::string  Mangling( cGeneratePointsTerrains *) {return "C04B5099702B03D0F93F";};


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

void  BinaryUnDumpFromFile(cGenerateLiaisons & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.KeyAssoc(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.FilterIm1().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.FilterIm1().ValForcedForUnUmp(),aFp);
        }
        else  anObj.FilterIm1().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.FilterIm2().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.FilterIm2().ValForcedForUnUmp(),aFp);
        }
        else  anObj.FilterIm2().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.BruitIm1(),aFp);
    BinaryUnDumpFromFile(anObj.BruitIm2(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cGenerateLiaisons & anObj)
{
    BinaryDumpInFile(aFp,anObj.KeyAssoc());
    BinaryDumpInFile(aFp,anObj.FilterIm1().IsInit());
    if (anObj.FilterIm1().IsInit()) BinaryDumpInFile(aFp,anObj.FilterIm1().Val());
    BinaryDumpInFile(aFp,anObj.FilterIm2().IsInit());
    if (anObj.FilterIm2().IsInit()) BinaryDumpInFile(aFp,anObj.FilterIm2().Val());
    BinaryDumpInFile(aFp,anObj.BruitIm1());
    BinaryDumpInFile(aFp,anObj.BruitIm2());
}

cElXMLTree * ToXMLTree(const cGenerateLiaisons & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"GenerateLiaisons",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("KeyAssoc"),anObj.KeyAssoc())->ReTagThis("KeyAssoc"));
   if (anObj.FilterIm1().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("FilterIm1"),anObj.FilterIm1().Val())->ReTagThis("FilterIm1"));
   if (anObj.FilterIm2().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("FilterIm2"),anObj.FilterIm2().Val())->ReTagThis("FilterIm2"));
   aRes->AddFils(::ToXMLTree(std::string("BruitIm1"),anObj.BruitIm1())->ReTagThis("BruitIm1"));
   aRes->AddFils(::ToXMLTree(std::string("BruitIm2"),anObj.BruitIm2())->ReTagThis("BruitIm2"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cGenerateLiaisons & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.KeyAssoc(),aTree->Get("KeyAssoc",1)); //tototo 

   xml_init(anObj.FilterIm1(),aTree->Get("FilterIm1",1),std::string(".*")); //tototo 

   xml_init(anObj.FilterIm2(),aTree->Get("FilterIm2",1),std::string(".*")); //tototo 

   xml_init(anObj.BruitIm1(),aTree->Get("BruitIm1",1)); //tototo 

   xml_init(anObj.BruitIm2(),aTree->Get("BruitIm2",1)); //tototo 
}

std::string  Mangling( cGenerateLiaisons *) {return "2E225C912F2D2D95FF3F";};


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

void  BinaryUnDumpFromFile(cExportSimulation & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.GeneratePointsTerrains(),aFp);
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cGenerateLiaisons aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.GenerateLiaisons().push_back(aVal);
        }
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cExportSimulation & anObj)
{
    BinaryDumpInFile(aFp,anObj.GeneratePointsTerrains());
    BinaryDumpInFile(aFp,(int)anObj.GenerateLiaisons().size());
    for(  std::list< cGenerateLiaisons >::const_iterator iT=anObj.GenerateLiaisons().begin();
         iT!=anObj.GenerateLiaisons().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
}

cElXMLTree * ToXMLTree(const cExportSimulation & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ExportSimulation",eXMLBranche);
   aRes->AddFils(ToXMLTree(anObj.GeneratePointsTerrains())->ReTagThis("GeneratePointsTerrains"));
  for
  (       std::list< cGenerateLiaisons >::const_iterator it=anObj.GenerateLiaisons().begin();
      it !=anObj.GenerateLiaisons().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("GenerateLiaisons"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cExportSimulation & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.GeneratePointsTerrains(),aTree->Get("GeneratePointsTerrains",1)); //tototo 

   xml_init(anObj.GenerateLiaisons(),aTree->GetAll("GenerateLiaisons",false,1));
}

std::string  Mangling( cExportSimulation *) {return "AC57783C58ACBBEFFE3F";};


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

void  BinaryUnDumpFromFile(cTestInteractif & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.AvantCompens().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.AvantCompens().ValForcedForUnUmp(),aFp);
        }
        else  anObj.AvantCompens().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ApresCompens().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ApresCompens().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ApresCompens().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.TestF2C2().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.TestF2C2().ValForcedForUnUmp(),aFp);
        }
        else  anObj.TestF2C2().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SetStepByStep().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SetStepByStep().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SetStepByStep().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cTestInteractif & anObj)
{
    BinaryDumpInFile(aFp,anObj.AvantCompens().IsInit());
    if (anObj.AvantCompens().IsInit()) BinaryDumpInFile(aFp,anObj.AvantCompens().Val());
    BinaryDumpInFile(aFp,anObj.ApresCompens().IsInit());
    if (anObj.ApresCompens().IsInit()) BinaryDumpInFile(aFp,anObj.ApresCompens().Val());
    BinaryDumpInFile(aFp,anObj.TestF2C2().IsInit());
    if (anObj.TestF2C2().IsInit()) BinaryDumpInFile(aFp,anObj.TestF2C2().Val());
    BinaryDumpInFile(aFp,anObj.SetStepByStep().IsInit());
    if (anObj.SetStepByStep().IsInit()) BinaryDumpInFile(aFp,anObj.SetStepByStep().Val());
}

cElXMLTree * ToXMLTree(const cTestInteractif & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"TestInteractif",eXMLBranche);
   if (anObj.AvantCompens().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("AvantCompens"),anObj.AvantCompens().Val())->ReTagThis("AvantCompens"));
   if (anObj.ApresCompens().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ApresCompens"),anObj.ApresCompens().Val())->ReTagThis("ApresCompens"));
   if (anObj.TestF2C2().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("TestF2C2"),anObj.TestF2C2().Val())->ReTagThis("TestF2C2"));
   if (anObj.SetStepByStep().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SetStepByStep"),anObj.SetStepByStep().Val())->ReTagThis("SetStepByStep"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cTestInteractif & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.AvantCompens(),aTree->Get("AvantCompens",1),bool(false)); //tototo 

   xml_init(anObj.ApresCompens(),aTree->Get("ApresCompens",1),bool(false)); //tototo 

   xml_init(anObj.TestF2C2(),aTree->Get("TestF2C2",1),bool(false)); //tototo 

   xml_init(anObj.SetStepByStep(),aTree->Get("SetStepByStep",1),bool(false)); //tototo 
}

std::string  Mangling( cTestInteractif *) {return "6CBE59924D172FDAFC3F";};


cTplValGesInit< int > & cIterationsCompensation::NbMin()
{
   return CtrlTimeCompens().Val().NbMin();
}

const cTplValGesInit< int > & cIterationsCompensation::NbMin()const 
{
   return CtrlTimeCompens().Val().NbMin();
}


int & cIterationsCompensation::NbMax()
{
   return CtrlTimeCompens().Val().NbMax();
}

const int & cIterationsCompensation::NbMax()const 
{
   return CtrlTimeCompens().Val().NbMax();
}


double & cIterationsCompensation::SeuilEvolMoy()
{
   return CtrlTimeCompens().Val().SeuilEvolMoy();
}

const double & cIterationsCompensation::SeuilEvolMoy()const 
{
   return CtrlTimeCompens().Val().SeuilEvolMoy();
}


cTplValGesInit< double > & cIterationsCompensation::SeuilEvolMax()
{
   return CtrlTimeCompens().Val().SeuilEvolMax();
}

const cTplValGesInit< double > & cIterationsCompensation::SeuilEvolMax()const 
{
   return CtrlTimeCompens().Val().SeuilEvolMax();
}


double & cIterationsCompensation::Mult()
{
   return CtrlTimeCompens().Val().AutoAdaptLVM().Val().Mult();
}

const double & cIterationsCompensation::Mult()const 
{
   return CtrlTimeCompens().Val().AutoAdaptLVM().Val().Mult();
}


cTplValGesInit< bool > & cIterationsCompensation::ModeMin()
{
   return CtrlTimeCompens().Val().AutoAdaptLVM().Val().ModeMin();
}

const cTplValGesInit< bool > & cIterationsCompensation::ModeMin()const 
{
   return CtrlTimeCompens().Val().AutoAdaptLVM().Val().ModeMin();
}


cTplValGesInit< cAutoAdaptLVM > & cIterationsCompensation::AutoAdaptLVM()
{
   return CtrlTimeCompens().Val().AutoAdaptLVM();
}

const cTplValGesInit< cAutoAdaptLVM > & cIterationsCompensation::AutoAdaptLVM()const 
{
   return CtrlTimeCompens().Val().AutoAdaptLVM();
}


cTplValGesInit< cCtrlTimeCompens > & cIterationsCompensation::CtrlTimeCompens()
{
   return mCtrlTimeCompens;
}

const cTplValGesInit< cCtrlTimeCompens > & cIterationsCompensation::CtrlTimeCompens()const 
{
   return mCtrlTimeCompens;
}


cTplValGesInit< bool > & cIterationsCompensation::DoIt()
{
   return mDoIt;
}

const cTplValGesInit< bool > & cIterationsCompensation::DoIt()const 
{
   return mDoIt;
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


cTplValGesInit< std::string > & cIterationsCompensation::FileExportDir()
{
   return BasculeOrientation().Val().FileExportDir();
}

const cTplValGesInit< std::string > & cIterationsCompensation::FileExportDir()const 
{
   return BasculeOrientation().Val().FileExportDir();
}


cTplValGesInit< std::string > & cIterationsCompensation::FileExportInv()
{
   return BasculeOrientation().Val().FileExportInv();
}

const cTplValGesInit< std::string > & cIterationsCompensation::FileExportInv()const 
{
   return BasculeOrientation().Val().FileExportInv();
}


cTplValGesInit< std::string > & cIterationsCompensation::ForceSol()
{
   return BasculeOrientation().Val().ModeBascule().BasculeOnPoints().Val().ForceSol();
}

const cTplValGesInit< std::string > & cIterationsCompensation::ForceSol()const 
{
   return BasculeOrientation().Val().ModeBascule().BasculeOnPoints().Val().ForceSol();
}


cTplValGesInit< std::string > & cIterationsCompensation::PoseCentrale()
{
   return BasculeOrientation().Val().ModeBascule().BasculeOnPoints().Val().BascOnCentre().Val().PoseCentrale();
}

const cTplValGesInit< std::string > & cIterationsCompensation::PoseCentrale()const 
{
   return BasculeOrientation().Val().ModeBascule().BasculeOnPoints().Val().BascOnCentre().Val().PoseCentrale();
}


cTplValGesInit< bool > & cIterationsCompensation::EstimateSpeed()
{
   return BasculeOrientation().Val().ModeBascule().BasculeOnPoints().Val().BascOnCentre().Val().EstimateSpeed();
}

const cTplValGesInit< bool > & cIterationsCompensation::EstimateSpeed()const 
{
   return BasculeOrientation().Val().ModeBascule().BasculeOnPoints().Val().BascOnCentre().Val().EstimateSpeed();
}


cTplValGesInit< double > & cIterationsCompensation::ForceVertical()
{
   return BasculeOrientation().Val().ModeBascule().BasculeOnPoints().Val().BascOnCentre().Val().ForceVertical();
}

const cTplValGesInit< double > & cIterationsCompensation::ForceVertical()const 
{
   return BasculeOrientation().Val().ModeBascule().BasculeOnPoints().Val().BascOnCentre().Val().ForceVertical();
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


cTplValGesInit< cAerialDeformNonLin > & cIterationsCompensation::AerialDeformNonLin()
{
   return BasculeOrientation().Val().ModeBascule().BasculeOnPoints().Val().AerialDeformNonLin();
}

const cTplValGesInit< cAerialDeformNonLin > & cIterationsCompensation::AerialDeformNonLin()const 
{
   return BasculeOrientation().Val().ModeBascule().BasculeOnPoints().Val().AerialDeformNonLin();
}


cTplValGesInit< std::string > & cIterationsCompensation::NameExport()
{
   return BasculeOrientation().Val().ModeBascule().BasculeOnPoints().Val().NameExport();
}

const cTplValGesInit< std::string > & cIterationsCompensation::NameExport()const 
{
   return BasculeOrientation().Val().ModeBascule().BasculeOnPoints().Val().NameExport();
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


std::list< cXml_EstimateOrientationInitBlockCamera > & cIterationsCompensation::EstimateOrientationInitBlockCamera()
{
   return mEstimateOrientationInitBlockCamera;
}

const std::list< cXml_EstimateOrientationInitBlockCamera > & cIterationsCompensation::EstimateOrientationInitBlockCamera()const 
{
   return mEstimateOrientationInitBlockCamera;
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

void  BinaryUnDumpFromFile(cIterationsCompensation & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.CtrlTimeCompens().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.CtrlTimeCompens().ValForcedForUnUmp(),aFp);
        }
        else  anObj.CtrlTimeCompens().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.DoIt().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.DoIt().ValForcedForUnUmp(),aFp);
        }
        else  anObj.DoIt().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SLMIter().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SLMIter().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SLMIter().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SLMEtape().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SLMEtape().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SLMEtape().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SLMGlob().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SLMGlob().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SLMGlob().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.MultSLMIter().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.MultSLMIter().ValForcedForUnUmp(),aFp);
        }
        else  anObj.MultSLMIter().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.MultSLMEtape().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.MultSLMEtape().ValForcedForUnUmp(),aFp);
        }
        else  anObj.MultSLMEtape().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.MultSLMGlob().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.MultSLMGlob().ValForcedForUnUmp(),aFp);
        }
        else  anObj.MultSLMGlob().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Pose2Init().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Pose2Init().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Pose2Init().SetNoInit();
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cSetRayMaxUtileCalib aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.SetRayMaxUtileCalib().push_back(aVal);
        }
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.BasculeOrientation().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.BasculeOrientation().ValForcedForUnUmp(),aFp);
        }
        else  anObj.BasculeOrientation().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.FixeEchelle().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.FixeEchelle().ValForcedForUnUmp(),aFp);
        }
        else  anObj.FixeEchelle().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.FixeOrientPlane().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.FixeOrientPlane().ValForcedForUnUmp(),aFp);
        }
        else  anObj.FixeOrientPlane().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.BasicOrPl().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.BasicOrPl().ValForcedForUnUmp(),aFp);
        }
        else  anObj.BasicOrPl().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.BlocBascule().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.BlocBascule().ValForcedForUnUmp(),aFp);
        }
        else  anObj.BlocBascule().SetNoInit();
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cXml_EstimateOrientationInitBlockCamera aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.EstimateOrientationInitBlockCamera().push_back(aVal);
        }
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.MesureErreurTournante().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.MesureErreurTournante().ValForcedForUnUmp(),aFp);
        }
        else  anObj.MesureErreurTournante().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SectionContraintes().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SectionContraintes().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SectionContraintes().SetNoInit();
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             std::string aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.Messages().push_back(aVal);
        }
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cVisuPtsMult aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.VisuPtsMult().push_back(aVal);
        }
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cVerifAero aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.VerifAero().push_back(aVal);
        }
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cExportSimulation aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.ExportSimulation().push_back(aVal);
        }
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.TestInteractif().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.TestInteractif().ValForcedForUnUmp(),aFp);
        }
        else  anObj.TestInteractif().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cIterationsCompensation & anObj)
{
    BinaryDumpInFile(aFp,anObj.CtrlTimeCompens().IsInit());
    if (anObj.CtrlTimeCompens().IsInit()) BinaryDumpInFile(aFp,anObj.CtrlTimeCompens().Val());
    BinaryDumpInFile(aFp,anObj.DoIt().IsInit());
    if (anObj.DoIt().IsInit()) BinaryDumpInFile(aFp,anObj.DoIt().Val());
    BinaryDumpInFile(aFp,anObj.SLMIter().IsInit());
    if (anObj.SLMIter().IsInit()) BinaryDumpInFile(aFp,anObj.SLMIter().Val());
    BinaryDumpInFile(aFp,anObj.SLMEtape().IsInit());
    if (anObj.SLMEtape().IsInit()) BinaryDumpInFile(aFp,anObj.SLMEtape().Val());
    BinaryDumpInFile(aFp,anObj.SLMGlob().IsInit());
    if (anObj.SLMGlob().IsInit()) BinaryDumpInFile(aFp,anObj.SLMGlob().Val());
    BinaryDumpInFile(aFp,anObj.MultSLMIter().IsInit());
    if (anObj.MultSLMIter().IsInit()) BinaryDumpInFile(aFp,anObj.MultSLMIter().Val());
    BinaryDumpInFile(aFp,anObj.MultSLMEtape().IsInit());
    if (anObj.MultSLMEtape().IsInit()) BinaryDumpInFile(aFp,anObj.MultSLMEtape().Val());
    BinaryDumpInFile(aFp,anObj.MultSLMGlob().IsInit());
    if (anObj.MultSLMGlob().IsInit()) BinaryDumpInFile(aFp,anObj.MultSLMGlob().Val());
    BinaryDumpInFile(aFp,anObj.Pose2Init().IsInit());
    if (anObj.Pose2Init().IsInit()) BinaryDumpInFile(aFp,anObj.Pose2Init().Val());
    BinaryDumpInFile(aFp,(int)anObj.SetRayMaxUtileCalib().size());
    for(  std::list< cSetRayMaxUtileCalib >::const_iterator iT=anObj.SetRayMaxUtileCalib().begin();
         iT!=anObj.SetRayMaxUtileCalib().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.BasculeOrientation().IsInit());
    if (anObj.BasculeOrientation().IsInit()) BinaryDumpInFile(aFp,anObj.BasculeOrientation().Val());
    BinaryDumpInFile(aFp,anObj.FixeEchelle().IsInit());
    if (anObj.FixeEchelle().IsInit()) BinaryDumpInFile(aFp,anObj.FixeEchelle().Val());
    BinaryDumpInFile(aFp,anObj.FixeOrientPlane().IsInit());
    if (anObj.FixeOrientPlane().IsInit()) BinaryDumpInFile(aFp,anObj.FixeOrientPlane().Val());
    BinaryDumpInFile(aFp,anObj.BasicOrPl().IsInit());
    if (anObj.BasicOrPl().IsInit()) BinaryDumpInFile(aFp,anObj.BasicOrPl().Val());
    BinaryDumpInFile(aFp,anObj.BlocBascule().IsInit());
    if (anObj.BlocBascule().IsInit()) BinaryDumpInFile(aFp,anObj.BlocBascule().Val());
    BinaryDumpInFile(aFp,(int)anObj.EstimateOrientationInitBlockCamera().size());
    for(  std::list< cXml_EstimateOrientationInitBlockCamera >::const_iterator iT=anObj.EstimateOrientationInitBlockCamera().begin();
         iT!=anObj.EstimateOrientationInitBlockCamera().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.MesureErreurTournante().IsInit());
    if (anObj.MesureErreurTournante().IsInit()) BinaryDumpInFile(aFp,anObj.MesureErreurTournante().Val());
    BinaryDumpInFile(aFp,anObj.SectionContraintes().IsInit());
    if (anObj.SectionContraintes().IsInit()) BinaryDumpInFile(aFp,anObj.SectionContraintes().Val());
    BinaryDumpInFile(aFp,(int)anObj.Messages().size());
    for(  std::list< std::string >::const_iterator iT=anObj.Messages().begin();
         iT!=anObj.Messages().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,(int)anObj.VisuPtsMult().size());
    for(  std::list< cVisuPtsMult >::const_iterator iT=anObj.VisuPtsMult().begin();
         iT!=anObj.VisuPtsMult().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,(int)anObj.VerifAero().size());
    for(  std::list< cVerifAero >::const_iterator iT=anObj.VerifAero().begin();
         iT!=anObj.VerifAero().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,(int)anObj.ExportSimulation().size());
    for(  std::list< cExportSimulation >::const_iterator iT=anObj.ExportSimulation().begin();
         iT!=anObj.ExportSimulation().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.TestInteractif().IsInit());
    if (anObj.TestInteractif().IsInit()) BinaryDumpInFile(aFp,anObj.TestInteractif().Val());
}

cElXMLTree * ToXMLTree(const cIterationsCompensation & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"IterationsCompensation",eXMLBranche);
   if (anObj.CtrlTimeCompens().IsInit())
      aRes->AddFils(ToXMLTree(anObj.CtrlTimeCompens().Val())->ReTagThis("CtrlTimeCompens"));
   if (anObj.DoIt().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DoIt"),anObj.DoIt().Val())->ReTagThis("DoIt"));
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
  for
  (       std::list< cXml_EstimateOrientationInitBlockCamera >::const_iterator it=anObj.EstimateOrientationInitBlockCamera().begin();
      it !=anObj.EstimateOrientationInitBlockCamera().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("EstimateOrientationInitBlockCamera"));
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
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cIterationsCompensation & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.CtrlTimeCompens(),aTree->Get("CtrlTimeCompens",1)); //tototo 

   xml_init(anObj.DoIt(),aTree->Get("DoIt",1),bool(true)); //tototo 

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

   xml_init(anObj.EstimateOrientationInitBlockCamera(),aTree->GetAll("EstimateOrientationInitBlockCamera",false,1));

   xml_init(anObj.MesureErreurTournante(),aTree->Get("MesureErreurTournante",1)); //tototo 

   xml_init(anObj.SectionContraintes(),aTree->Get("SectionContraintes",1)); //tototo 

   xml_init(anObj.Messages(),aTree->GetAll("Messages",false,1));

   xml_init(anObj.VisuPtsMult(),aTree->GetAll("VisuPtsMult",false,1));

   xml_init(anObj.VerifAero(),aTree->GetAll("VerifAero",false,1));

   xml_init(anObj.ExportSimulation(),aTree->GetAll("ExportSimulation",false,1));

   xml_init(anObj.TestInteractif(),aTree->Get("TestInteractif",1)); //tototo 
}

std::string  Mangling( cIterationsCompensation *) {return "DE3F0102545222E4FC3F";};


std::string & cTraceCpleHom::Id()
{
   return mId;
}

const std::string & cTraceCpleHom::Id()const 
{
   return mId;
}

void  BinaryUnDumpFromFile(cTraceCpleHom & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Id(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cTraceCpleHom & anObj)
{
    BinaryDumpInFile(aFp,anObj.Id());
}

cElXMLTree * ToXMLTree(const cTraceCpleHom & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"TraceCpleHom",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Id"),anObj.Id())->ReTagThis("Id"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cTraceCpleHom & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.Id(),aTree->Get("Id",1)); //tototo 
}

std::string  Mangling( cTraceCpleHom *) {return "A05C4FAD846536CAFC3F";};


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

void  BinaryUnDumpFromFile(cTraceCpleCam & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Cam1(),aFp);
    BinaryUnDumpFromFile(anObj.Cam2(),aFp);
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cTraceCpleHom aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.TraceCpleHom().push_back(aVal);
        }
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cTraceCpleCam & anObj)
{
    BinaryDumpInFile(aFp,anObj.Cam1());
    BinaryDumpInFile(aFp,anObj.Cam2());
    BinaryDumpInFile(aFp,(int)anObj.TraceCpleHom().size());
    for(  std::list< cTraceCpleHom >::const_iterator iT=anObj.TraceCpleHom().begin();
         iT!=anObj.TraceCpleHom().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
}

cElXMLTree * ToXMLTree(const cTraceCpleCam & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"TraceCpleCam",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Cam1"),anObj.Cam1())->ReTagThis("Cam1"));
   aRes->AddFils(::ToXMLTree(std::string("Cam2"),anObj.Cam2())->ReTagThis("Cam2"));
  for
  (       std::list< cTraceCpleHom >::const_iterator it=anObj.TraceCpleHom().begin();
      it !=anObj.TraceCpleHom().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("TraceCpleHom"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cTraceCpleCam & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.Cam1(),aTree->Get("Cam1",1)); //tototo 

   xml_init(anObj.Cam2(),aTree->Get("Cam2",1)); //tototo 

   xml_init(anObj.TraceCpleHom(),aTree->GetAll("TraceCpleHom",false,1));
}

std::string  Mangling( cTraceCpleCam *) {return "D3ADD5A204DE7BA6FE3F";};


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

void  BinaryUnDumpFromFile(cSectionTracage & anObj,ELISE_fp & aFp)
{
   { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cTraceCpleCam aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.TraceCpleCam().push_back(aVal);
        }
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.GetChar().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.GetChar().ValForcedForUnUmp(),aFp);
        }
        else  anObj.GetChar().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cSectionTracage & anObj)
{
    BinaryDumpInFile(aFp,(int)anObj.TraceCpleCam().size());
    for(  std::list< cTraceCpleCam >::const_iterator iT=anObj.TraceCpleCam().begin();
         iT!=anObj.TraceCpleCam().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.GetChar().IsInit());
    if (anObj.GetChar().IsInit()) BinaryDumpInFile(aFp,anObj.GetChar().Val());
}

cElXMLTree * ToXMLTree(const cSectionTracage & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"SectionTracage",eXMLBranche);
  for
  (       std::list< cTraceCpleCam >::const_iterator it=anObj.TraceCpleCam().begin();
      it !=anObj.TraceCpleCam().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("TraceCpleCam"));
   if (anObj.GetChar().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("GetChar"),anObj.GetChar().Val())->ReTagThis("GetChar"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cSectionTracage & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.TraceCpleCam(),aTree->GetAll("TraceCpleCam",false,1));

   xml_init(anObj.GetChar(),aTree->Get("GetChar",1),bool(true)); //tototo 
}

std::string  Mangling( cSectionTracage *) {return "7D80059324CDBA80FF3F";};


double & cContrCamConseq::SigmaPix()
{
   return mSigmaPix;
}

const double & cContrCamConseq::SigmaPix()const 
{
   return mSigmaPix;
}


int & cContrCamConseq::NbGrid()
{
   return mNbGrid;
}

const int & cContrCamConseq::NbGrid()const 
{
   return mNbGrid;
}

void  BinaryUnDumpFromFile(cContrCamConseq & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.SigmaPix(),aFp);
    BinaryUnDumpFromFile(anObj.NbGrid(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cContrCamConseq & anObj)
{
    BinaryDumpInFile(aFp,anObj.SigmaPix());
    BinaryDumpInFile(aFp,anObj.NbGrid());
}

cElXMLTree * ToXMLTree(const cContrCamConseq & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ContrCamConseq",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("SigmaPix"),anObj.SigmaPix())->ReTagThis("SigmaPix"));
   aRes->AddFils(::ToXMLTree(std::string("NbGrid"),anObj.NbGrid())->ReTagThis("NbGrid"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cContrCamConseq & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.SigmaPix(),aTree->Get("SigmaPix",1)); //tototo 

   xml_init(anObj.NbGrid(),aTree->Get("NbGrid",1)); //tototo 
}

std::string  Mangling( cContrCamConseq *) {return "747338A99DC7F188FC3F";};


cTplValGesInit< std::string > & cContrCamGenInc::PatternApply()
{
   return mPatternApply;
}

const cTplValGesInit< std::string > & cContrCamGenInc::PatternApply()const 
{
   return mPatternApply;
}


cTplValGesInit< double > & cContrCamGenInc::PdsAttachToId()
{
   return mPdsAttachToId;
}

const cTplValGesInit< double > & cContrCamGenInc::PdsAttachToId()const 
{
   return mPdsAttachToId;
}


cTplValGesInit< double > & cContrCamGenInc::PdsAttachToLast()
{
   return mPdsAttachToLast;
}

const cTplValGesInit< double > & cContrCamGenInc::PdsAttachToLast()const 
{
   return mPdsAttachToLast;
}


cTplValGesInit< double > & cContrCamGenInc::PdsAttachRGLob()
{
   return mPdsAttachRGLob;
}

const cTplValGesInit< double > & cContrCamGenInc::PdsAttachRGLob()const 
{
   return mPdsAttachRGLob;
}

void  BinaryUnDumpFromFile(cContrCamGenInc & anObj,ELISE_fp & aFp)
{
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
             anObj.PdsAttachToId().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.PdsAttachToId().ValForcedForUnUmp(),aFp);
        }
        else  anObj.PdsAttachToId().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.PdsAttachToLast().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.PdsAttachToLast().ValForcedForUnUmp(),aFp);
        }
        else  anObj.PdsAttachToLast().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.PdsAttachRGLob().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.PdsAttachRGLob().ValForcedForUnUmp(),aFp);
        }
        else  anObj.PdsAttachRGLob().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cContrCamGenInc & anObj)
{
    BinaryDumpInFile(aFp,anObj.PatternApply().IsInit());
    if (anObj.PatternApply().IsInit()) BinaryDumpInFile(aFp,anObj.PatternApply().Val());
    BinaryDumpInFile(aFp,anObj.PdsAttachToId().IsInit());
    if (anObj.PdsAttachToId().IsInit()) BinaryDumpInFile(aFp,anObj.PdsAttachToId().Val());
    BinaryDumpInFile(aFp,anObj.PdsAttachToLast().IsInit());
    if (anObj.PdsAttachToLast().IsInit()) BinaryDumpInFile(aFp,anObj.PdsAttachToLast().Val());
    BinaryDumpInFile(aFp,anObj.PdsAttachRGLob().IsInit());
    if (anObj.PdsAttachRGLob().IsInit()) BinaryDumpInFile(aFp,anObj.PdsAttachRGLob().Val());
}

cElXMLTree * ToXMLTree(const cContrCamGenInc & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ContrCamGenInc",eXMLBranche);
   if (anObj.PatternApply().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PatternApply"),anObj.PatternApply().Val())->ReTagThis("PatternApply"));
   if (anObj.PdsAttachToId().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PdsAttachToId"),anObj.PdsAttachToId().Val())->ReTagThis("PdsAttachToId"));
   if (anObj.PdsAttachToLast().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PdsAttachToLast"),anObj.PdsAttachToLast().Val())->ReTagThis("PdsAttachToLast"));
   if (anObj.PdsAttachRGLob().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PdsAttachRGLob"),anObj.PdsAttachRGLob().Val())->ReTagThis("PdsAttachRGLob"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cContrCamGenInc & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.PatternApply(),aTree->Get("PatternApply",1)); //tototo 

   xml_init(anObj.PdsAttachToId(),aTree->Get("PdsAttachToId",1)); //tototo 

   xml_init(anObj.PdsAttachToLast(),aTree->Get("PdsAttachToLast",1)); //tototo 

   xml_init(anObj.PdsAttachRGLob(),aTree->Get("PdsAttachRGLob",1)); //tototo 
}

std::string  Mangling( cContrCamGenInc *) {return "A8AE434A1F7E7894FE3F";};


std::string & cObsBlockCamRig::Id()
{
   return mId;
}

const std::string & cObsBlockCamRig::Id()const 
{
   return mId;
}


cTplValGesInit< bool > & cObsBlockCamRig::Show()
{
   return mShow;
}

const cTplValGesInit< bool > & cObsBlockCamRig::Show()const 
{
   return mShow;
}


cTplValGesInit< cRigidBlockWeighting > & cObsBlockCamRig::GlobalPond()
{
   return mGlobalPond;
}

const cTplValGesInit< cRigidBlockWeighting > & cObsBlockCamRig::GlobalPond()const 
{
   return mGlobalPond;
}


cTplValGesInit< cRigidBlockWeighting > & cObsBlockCamRig::RelTimePond()
{
   return mRelTimePond;
}

const cTplValGesInit< cRigidBlockWeighting > & cObsBlockCamRig::RelTimePond()const 
{
   return mRelTimePond;
}


cTplValGesInit< cRigidBlockWeighting > & cObsBlockCamRig::GlobalDistPond()
{
   return mGlobalDistPond;
}

const cTplValGesInit< cRigidBlockWeighting > & cObsBlockCamRig::GlobalDistPond()const 
{
   return mGlobalDistPond;
}


cTplValGesInit< cRigidBlockWeighting > & cObsBlockCamRig::RelTimeDistPond()
{
   return mRelTimeDistPond;
}

const cTplValGesInit< cRigidBlockWeighting > & cObsBlockCamRig::RelTimeDistPond()const 
{
   return mRelTimeDistPond;
}

void  BinaryUnDumpFromFile(cObsBlockCamRig & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Id(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Show().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Show().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Show().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.GlobalPond().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.GlobalPond().ValForcedForUnUmp(),aFp);
        }
        else  anObj.GlobalPond().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.RelTimePond().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.RelTimePond().ValForcedForUnUmp(),aFp);
        }
        else  anObj.RelTimePond().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.GlobalDistPond().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.GlobalDistPond().ValForcedForUnUmp(),aFp);
        }
        else  anObj.GlobalDistPond().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.RelTimeDistPond().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.RelTimeDistPond().ValForcedForUnUmp(),aFp);
        }
        else  anObj.RelTimeDistPond().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cObsBlockCamRig & anObj)
{
    BinaryDumpInFile(aFp,anObj.Id());
    BinaryDumpInFile(aFp,anObj.Show().IsInit());
    if (anObj.Show().IsInit()) BinaryDumpInFile(aFp,anObj.Show().Val());
    BinaryDumpInFile(aFp,anObj.GlobalPond().IsInit());
    if (anObj.GlobalPond().IsInit()) BinaryDumpInFile(aFp,anObj.GlobalPond().Val());
    BinaryDumpInFile(aFp,anObj.RelTimePond().IsInit());
    if (anObj.RelTimePond().IsInit()) BinaryDumpInFile(aFp,anObj.RelTimePond().Val());
    BinaryDumpInFile(aFp,anObj.GlobalDistPond().IsInit());
    if (anObj.GlobalDistPond().IsInit()) BinaryDumpInFile(aFp,anObj.GlobalDistPond().Val());
    BinaryDumpInFile(aFp,anObj.RelTimeDistPond().IsInit());
    if (anObj.RelTimeDistPond().IsInit()) BinaryDumpInFile(aFp,anObj.RelTimeDistPond().Val());
}

cElXMLTree * ToXMLTree(const cObsBlockCamRig & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ObsBlockCamRig",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Id"),anObj.Id())->ReTagThis("Id"));
   if (anObj.Show().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Show"),anObj.Show().Val())->ReTagThis("Show"));
   if (anObj.GlobalPond().IsInit())
      aRes->AddFils(ToXMLTree(anObj.GlobalPond().Val())->ReTagThis("GlobalPond"));
   if (anObj.RelTimePond().IsInit())
      aRes->AddFils(ToXMLTree(anObj.RelTimePond().Val())->ReTagThis("RelTimePond"));
   if (anObj.GlobalDistPond().IsInit())
      aRes->AddFils(ToXMLTree(anObj.GlobalDistPond().Val())->ReTagThis("GlobalDistPond"));
   if (anObj.RelTimeDistPond().IsInit())
      aRes->AddFils(ToXMLTree(anObj.RelTimeDistPond().Val())->ReTagThis("RelTimeDistPond"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cObsBlockCamRig & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.Id(),aTree->Get("Id",1)); //tototo 

   xml_init(anObj.Show(),aTree->Get("Show",1),bool(false)); //tototo 

   xml_init(anObj.GlobalPond(),aTree->Get("GlobalPond",1)); //tototo 

   xml_init(anObj.RelTimePond(),aTree->Get("RelTimePond",1)); //tototo 

   xml_init(anObj.GlobalDistPond(),aTree->Get("GlobalDistPond",1)); //tototo 

   xml_init(anObj.RelTimeDistPond(),aTree->Get("RelTimeDistPond",1)); //tototo 
}

std::string  Mangling( cObsBlockCamRig *) {return "804BA908F6FE7F9CFABF";};


std::string & cObsCenterInPlane::Id()
{
   return mId;
}

const std::string & cObsCenterInPlane::Id()const 
{
   return mId;
}

void  BinaryUnDumpFromFile(cObsCenterInPlane & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Id(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cObsCenterInPlane & anObj)
{
    BinaryDumpInFile(aFp,anObj.Id());
}

cElXMLTree * ToXMLTree(const cObsCenterInPlane & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ObsCenterInPlane",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Id"),anObj.Id())->ReTagThis("Id"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cObsCenterInPlane & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.Id(),aTree->Get("Id",1)); //tototo 
}

std::string  Mangling( cObsCenterInPlane *) {return "0170FFB96818F9A0FDBF";};


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

void  BinaryUnDumpFromFile(cROA_FichierImg & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Name(),aFp);
    BinaryUnDumpFromFile(anObj.Sz(),aFp);
    BinaryUnDumpFromFile(anObj.Exag(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.VisuVideo().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.VisuVideo().ValForcedForUnUmp(),aFp);
        }
        else  anObj.VisuVideo().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cROA_FichierImg & anObj)
{
    BinaryDumpInFile(aFp,anObj.Name());
    BinaryDumpInFile(aFp,anObj.Sz());
    BinaryDumpInFile(aFp,anObj.Exag());
    BinaryDumpInFile(aFp,anObj.VisuVideo().IsInit());
    if (anObj.VisuVideo().IsInit()) BinaryDumpInFile(aFp,anObj.VisuVideo().Val());
}

cElXMLTree * ToXMLTree(const cROA_FichierImg & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ROA_FichierImg",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Name"),anObj.Name())->ReTagThis("Name"));
   aRes->AddFils(::ToXMLTree(std::string("Sz"),anObj.Sz())->ReTagThis("Sz"));
   aRes->AddFils(::ToXMLTree(std::string("Exag"),anObj.Exag())->ReTagThis("Exag"));
   if (anObj.VisuVideo().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("VisuVideo"),anObj.VisuVideo().Val())->ReTagThis("VisuVideo"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cROA_FichierImg & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.Name(),aTree->Get("Name",1)); //tototo 

   xml_init(anObj.Sz(),aTree->Get("Sz",1)); //tototo 

   xml_init(anObj.Exag(),aTree->Get("Exag",1)); //tototo 

   xml_init(anObj.VisuVideo(),aTree->Get("VisuVideo",1),bool(false)); //tototo 
}

std::string  Mangling( cROA_FichierImg *) {return "8F73D2948A57868AFF3F";};


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

void  BinaryUnDumpFromFile(cRapportObsAppui & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.OnlyLastIter().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.OnlyLastIter().ValForcedForUnUmp(),aFp);
        }
        else  anObj.OnlyLastIter().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.FichierTxt(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ColPerPose().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ColPerPose().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ColPerPose().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SeuilColOut().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SeuilColOut().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SeuilColOut().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ROA_FichierImg().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ROA_FichierImg().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ROA_FichierImg().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cRapportObsAppui & anObj)
{
    BinaryDumpInFile(aFp,anObj.OnlyLastIter().IsInit());
    if (anObj.OnlyLastIter().IsInit()) BinaryDumpInFile(aFp,anObj.OnlyLastIter().Val());
    BinaryDumpInFile(aFp,anObj.FichierTxt());
    BinaryDumpInFile(aFp,anObj.ColPerPose().IsInit());
    if (anObj.ColPerPose().IsInit()) BinaryDumpInFile(aFp,anObj.ColPerPose().Val());
    BinaryDumpInFile(aFp,anObj.SeuilColOut().IsInit());
    if (anObj.SeuilColOut().IsInit()) BinaryDumpInFile(aFp,anObj.SeuilColOut().Val());
    BinaryDumpInFile(aFp,anObj.ROA_FichierImg().IsInit());
    if (anObj.ROA_FichierImg().IsInit()) BinaryDumpInFile(aFp,anObj.ROA_FichierImg().Val());
}

cElXMLTree * ToXMLTree(const cRapportObsAppui & anObj)
{
  XMLPushContext(anObj.mGXml);
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
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cRapportObsAppui & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.OnlyLastIter(),aTree->Get("OnlyLastIter",1),bool(true)); //tototo 

   xml_init(anObj.FichierTxt(),aTree->Get("FichierTxt",1)); //tototo 

   xml_init(anObj.ColPerPose(),aTree->Get("ColPerPose",1),bool(false)); //tototo 

   xml_init(anObj.SeuilColOut(),aTree->Get("SeuilColOut",1),double(1e3)); //tototo 

   xml_init(anObj.ROA_FichierImg(),aTree->Get("ROA_FichierImg",1)); //tototo 
}

std::string  Mangling( cRapportObsAppui *) {return "B05E10BB9A7729A6FF3F";};


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

void  BinaryUnDumpFromFile(cObsAppuis & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.NameRef(),aFp);
    BinaryUnDumpFromFile(anObj.Pond(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.RapportObsAppui().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.RapportObsAppui().ValForcedForUnUmp(),aFp);
        }
        else  anObj.RapportObsAppui().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cObsAppuis & anObj)
{
    BinaryDumpInFile(aFp,anObj.NameRef());
    BinaryDumpInFile(aFp,anObj.Pond());
    BinaryDumpInFile(aFp,anObj.RapportObsAppui().IsInit());
    if (anObj.RapportObsAppui().IsInit()) BinaryDumpInFile(aFp,anObj.RapportObsAppui().Val());
}

cElXMLTree * ToXMLTree(const cObsAppuis & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ObsAppuis",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("NameRef"),anObj.NameRef())->ReTagThis("NameRef"));
   aRes->AddFils(ToXMLTree(anObj.Pond())->ReTagThis("Pond"));
   if (anObj.RapportObsAppui().IsInit())
      aRes->AddFils(ToXMLTree(anObj.RapportObsAppui().Val())->ReTagThis("RapportObsAppui"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cObsAppuis & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.NameRef(),aTree->Get("NameRef",1)); //tototo 

   xml_init(anObj.Pond(),aTree->Get("Pond",1)); //tototo 

   xml_init(anObj.RapportObsAppui(),aTree->Get("RapportObsAppui",1)); //tototo 
}

std::string  Mangling( cObsAppuis *) {return "1C65017EDD3FB6B0FF3F";};


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


cTplValGesInit< bool > & cObsAppuisFlottant::ShowUnused()
{
   return mShowUnused;
}

const cTplValGesInit< bool > & cObsAppuisFlottant::ShowUnused()const 
{
   return mShowUnused;
}

void  BinaryUnDumpFromFile(cObsAppuisFlottant & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.NameRef(),aFp);
    BinaryUnDumpFromFile(anObj.PondIm(),aFp);
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cElRegex_Ptr aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.PtsShowDet().push_back(aVal);
        }
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.DetShow3D().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.DetShow3D().ValForcedForUnUmp(),aFp);
        }
        else  anObj.DetShow3D().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NivAlerteDetail().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NivAlerteDetail().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NivAlerteDetail().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ShowMax().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ShowMax().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ShowMax().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ShowSom().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ShowSom().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ShowSom().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ShowUnused().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ShowUnused().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ShowUnused().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cObsAppuisFlottant & anObj)
{
    BinaryDumpInFile(aFp,anObj.NameRef());
    BinaryDumpInFile(aFp,anObj.PondIm());
    BinaryDumpInFile(aFp,(int)anObj.PtsShowDet().size());
    for(  std::list< cElRegex_Ptr >::const_iterator iT=anObj.PtsShowDet().begin();
         iT!=anObj.PtsShowDet().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.DetShow3D().IsInit());
    if (anObj.DetShow3D().IsInit()) BinaryDumpInFile(aFp,anObj.DetShow3D().Val());
    BinaryDumpInFile(aFp,anObj.NivAlerteDetail().IsInit());
    if (anObj.NivAlerteDetail().IsInit()) BinaryDumpInFile(aFp,anObj.NivAlerteDetail().Val());
    BinaryDumpInFile(aFp,anObj.ShowMax().IsInit());
    if (anObj.ShowMax().IsInit()) BinaryDumpInFile(aFp,anObj.ShowMax().Val());
    BinaryDumpInFile(aFp,anObj.ShowSom().IsInit());
    if (anObj.ShowSom().IsInit()) BinaryDumpInFile(aFp,anObj.ShowSom().Val());
    BinaryDumpInFile(aFp,anObj.ShowUnused().IsInit());
    if (anObj.ShowUnused().IsInit()) BinaryDumpInFile(aFp,anObj.ShowUnused().Val());
}

cElXMLTree * ToXMLTree(const cObsAppuisFlottant & anObj)
{
  XMLPushContext(anObj.mGXml);
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
   if (anObj.ShowUnused().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ShowUnused"),anObj.ShowUnused().Val())->ReTagThis("ShowUnused"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cObsAppuisFlottant & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.NameRef(),aTree->Get("NameRef",1)); //tototo 

   xml_init(anObj.PondIm(),aTree->Get("PondIm",1)); //tototo 

   xml_init(anObj.PtsShowDet(),aTree->GetAll("PtsShowDet",false,1));

   xml_init(anObj.DetShow3D(),aTree->Get("DetShow3D",1),bool(false)); //tototo 

   xml_init(anObj.NivAlerteDetail(),aTree->Get("NivAlerteDetail",1),double(1e9)); //tototo 

   xml_init(anObj.ShowMax(),aTree->Get("ShowMax",1),bool(false)); //tototo 

   xml_init(anObj.ShowSom(),aTree->Get("ShowSom",1),bool(false)); //tototo 

   xml_init(anObj.ShowUnused(),aTree->Get("ShowUnused",1),bool(true)); //tototo 
}

std::string  Mangling( cObsAppuisFlottant *) {return "6979847B82116EC1FE3F";};


cTplValGesInit< std::string > & cRappelOnZ::KeyGrpApply()
{
   return mKeyGrpApply;
}

const cTplValGesInit< std::string > & cRappelOnZ::KeyGrpApply()const 
{
   return mKeyGrpApply;
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

void  BinaryUnDumpFromFile(cRappelOnZ & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.KeyGrpApply().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.KeyGrpApply().ValForcedForUnUmp(),aFp);
        }
        else  anObj.KeyGrpApply().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.Z(),aFp);
    BinaryUnDumpFromFile(anObj.IncC(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.IncE().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.IncE().ValForcedForUnUmp(),aFp);
        }
        else  anObj.IncE().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SeuilR().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SeuilR().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SeuilR().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.LayerMasq().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.LayerMasq().ValForcedForUnUmp(),aFp);
        }
        else  anObj.LayerMasq().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cRappelOnZ & anObj)
{
    BinaryDumpInFile(aFp,anObj.KeyGrpApply().IsInit());
    if (anObj.KeyGrpApply().IsInit()) BinaryDumpInFile(aFp,anObj.KeyGrpApply().Val());
    BinaryDumpInFile(aFp,anObj.Z());
    BinaryDumpInFile(aFp,anObj.IncC());
    BinaryDumpInFile(aFp,anObj.IncE().IsInit());
    if (anObj.IncE().IsInit()) BinaryDumpInFile(aFp,anObj.IncE().Val());
    BinaryDumpInFile(aFp,anObj.SeuilR().IsInit());
    if (anObj.SeuilR().IsInit()) BinaryDumpInFile(aFp,anObj.SeuilR().Val());
    BinaryDumpInFile(aFp,anObj.LayerMasq().IsInit());
    if (anObj.LayerMasq().IsInit()) BinaryDumpInFile(aFp,anObj.LayerMasq().Val());
}

cElXMLTree * ToXMLTree(const cRappelOnZ & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"RappelOnZ",eXMLBranche);
   if (anObj.KeyGrpApply().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("KeyGrpApply"),anObj.KeyGrpApply().Val())->ReTagThis("KeyGrpApply"));
   aRes->AddFils(::ToXMLTree(std::string("Z"),anObj.Z())->ReTagThis("Z"));
   aRes->AddFils(::ToXMLTree(std::string("IncC"),anObj.IncC())->ReTagThis("IncC"));
   if (anObj.IncE().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("IncE"),anObj.IncE().Val())->ReTagThis("IncE"));
   if (anObj.SeuilR().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SeuilR"),anObj.SeuilR().Val())->ReTagThis("SeuilR"));
   if (anObj.LayerMasq().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("LayerMasq"),anObj.LayerMasq().Val())->ReTagThis("LayerMasq"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cRappelOnZ & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.KeyGrpApply(),aTree->Get("KeyGrpApply",1)); //tototo 

   xml_init(anObj.Z(),aTree->Get("Z",1)); //tototo 

   xml_init(anObj.IncC(),aTree->Get("IncC",1)); //tototo 

   xml_init(anObj.IncE(),aTree->Get("IncE",1)); //tototo 

   xml_init(anObj.SeuilR(),aTree->Get("SeuilR",1)); //tototo 

   xml_init(anObj.LayerMasq(),aTree->Get("LayerMasq",1)); //tototo 
}

std::string  Mangling( cRappelOnZ *) {return "28FE37DEF854479BFE3F";};


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


cTplValGesInit< std::string > & cObsLiaisons::KeyGrpApply()
{
   return RappelOnZ().Val().KeyGrpApply();
}

const cTplValGesInit< std::string > & cObsLiaisons::KeyGrpApply()const 
{
   return RappelOnZ().Val().KeyGrpApply();
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

void  BinaryUnDumpFromFile(cObsLiaisons & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.NameRef(),aFp);
    BinaryUnDumpFromFile(anObj.Pond(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.PondSurf().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.PondSurf().ValForcedForUnUmp(),aFp);
        }
        else  anObj.PondSurf().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.RappelOnZ().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.RappelOnZ().ValForcedForUnUmp(),aFp);
        }
        else  anObj.RappelOnZ().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cObsLiaisons & anObj)
{
    BinaryDumpInFile(aFp,anObj.NameRef());
    BinaryDumpInFile(aFp,anObj.Pond());
    BinaryDumpInFile(aFp,anObj.PondSurf().IsInit());
    if (anObj.PondSurf().IsInit()) BinaryDumpInFile(aFp,anObj.PondSurf().Val());
    BinaryDumpInFile(aFp,anObj.RappelOnZ().IsInit());
    if (anObj.RappelOnZ().IsInit()) BinaryDumpInFile(aFp,anObj.RappelOnZ().Val());
}

cElXMLTree * ToXMLTree(const cObsLiaisons & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ObsLiaisons",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("NameRef"),anObj.NameRef())->ReTagThis("NameRef"));
   aRes->AddFils(ToXMLTree(anObj.Pond())->ReTagThis("Pond"));
   if (anObj.PondSurf().IsInit())
      aRes->AddFils(ToXMLTree(anObj.PondSurf().Val())->ReTagThis("PondSurf"));
   if (anObj.RappelOnZ().IsInit())
      aRes->AddFils(ToXMLTree(anObj.RappelOnZ().Val())->ReTagThis("RappelOnZ"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cObsLiaisons & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.NameRef(),aTree->Get("NameRef",1)); //tototo 

   xml_init(anObj.Pond(),aTree->Get("Pond",1)); //tototo 

   xml_init(anObj.PondSurf(),aTree->Get("PondSurf",1)); //tototo 

   xml_init(anObj.RappelOnZ(),aTree->Get("RappelOnZ",1)); //tototo 
}

std::string  Mangling( cObsLiaisons *) {return "19784B05C4689683FE3F";};


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

void  BinaryUnDumpFromFile(cObsCentrePDV & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.PatternApply().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.PatternApply().ValForcedForUnUmp(),aFp);
        }
        else  anObj.PatternApply().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.Pond(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.PondAlti().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.PondAlti().ValForcedForUnUmp(),aFp);
        }
        else  anObj.PondAlti().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ShowTestVitesse().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ShowTestVitesse().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ShowTestVitesse().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cObsCentrePDV & anObj)
{
    BinaryDumpInFile(aFp,anObj.PatternApply().IsInit());
    if (anObj.PatternApply().IsInit()) BinaryDumpInFile(aFp,anObj.PatternApply().Val());
    BinaryDumpInFile(aFp,anObj.Pond());
    BinaryDumpInFile(aFp,anObj.PondAlti().IsInit());
    if (anObj.PondAlti().IsInit()) BinaryDumpInFile(aFp,anObj.PondAlti().Val());
    BinaryDumpInFile(aFp,anObj.ShowTestVitesse().IsInit());
    if (anObj.ShowTestVitesse().IsInit()) BinaryDumpInFile(aFp,anObj.ShowTestVitesse().Val());
}

cElXMLTree * ToXMLTree(const cObsCentrePDV & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ObsCentrePDV",eXMLBranche);
   if (anObj.PatternApply().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PatternApply"),anObj.PatternApply().Val())->ReTagThis("PatternApply"));
   aRes->AddFils(ToXMLTree(anObj.Pond())->ReTagThis("Pond"));
   if (anObj.PondAlti().IsInit())
      aRes->AddFils(ToXMLTree(anObj.PondAlti().Val())->ReTagThis("PondAlti"));
   if (anObj.ShowTestVitesse().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ShowTestVitesse"),anObj.ShowTestVitesse().Val())->ReTagThis("ShowTestVitesse"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cObsCentrePDV & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.PatternApply(),aTree->Get("PatternApply",1)); //tototo 

   xml_init(anObj.Pond(),aTree->Get("Pond",1)); //tototo 

   xml_init(anObj.PondAlti(),aTree->Get("PondAlti",1)); //tototo 

   xml_init(anObj.ShowTestVitesse(),aTree->Get("ShowTestVitesse",1),bool(false)); //tototo 
}

std::string  Mangling( cObsCentrePDV *) {return "78E455663D9E6E9FFF3F";};


Pt3dr & cORGI_CentreCommun::Incertitude()
{
   return mIncertitude;
}

const Pt3dr & cORGI_CentreCommun::Incertitude()const 
{
   return mIncertitude;
}

void  BinaryUnDumpFromFile(cORGI_CentreCommun & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Incertitude(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cORGI_CentreCommun & anObj)
{
    BinaryDumpInFile(aFp,anObj.Incertitude());
}

cElXMLTree * ToXMLTree(const cORGI_CentreCommun & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ORGI_CentreCommun",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Incertitude"),anObj.Incertitude())->ReTagThis("Incertitude"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cORGI_CentreCommun & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.Incertitude(),aTree->Get("Incertitude",1)); //tototo 
}

std::string  Mangling( cORGI_CentreCommun *) {return "318646706D3153A2FE3F";};


Pt3dr & cORGI_TetaCommun::Incertitude()
{
   return mIncertitude;
}

const Pt3dr & cORGI_TetaCommun::Incertitude()const 
{
   return mIncertitude;
}

void  BinaryUnDumpFromFile(cORGI_TetaCommun & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Incertitude(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cORGI_TetaCommun & anObj)
{
    BinaryDumpInFile(aFp,anObj.Incertitude());
}

cElXMLTree * ToXMLTree(const cORGI_TetaCommun & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ORGI_TetaCommun",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Incertitude"),anObj.Incertitude())->ReTagThis("Incertitude"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cORGI_TetaCommun & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.Incertitude(),aTree->Get("Incertitude",1)); //tototo 
}

std::string  Mangling( cORGI_TetaCommun *) {return "5F090BBCC7E029AEFE3F";};


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

void  BinaryUnDumpFromFile(cObsRigidGrpImage & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.RefGrp(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ORGI_CentreCommun().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ORGI_CentreCommun().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ORGI_CentreCommun().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ORGI_TetaCommun().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ORGI_TetaCommun().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ORGI_TetaCommun().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cObsRigidGrpImage & anObj)
{
    BinaryDumpInFile(aFp,anObj.RefGrp());
    BinaryDumpInFile(aFp,anObj.ORGI_CentreCommun().IsInit());
    if (anObj.ORGI_CentreCommun().IsInit()) BinaryDumpInFile(aFp,anObj.ORGI_CentreCommun().Val());
    BinaryDumpInFile(aFp,anObj.ORGI_TetaCommun().IsInit());
    if (anObj.ORGI_TetaCommun().IsInit()) BinaryDumpInFile(aFp,anObj.ORGI_TetaCommun().Val());
}

cElXMLTree * ToXMLTree(const cObsRigidGrpImage & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ObsRigidGrpImage",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("RefGrp"),anObj.RefGrp())->ReTagThis("RefGrp"));
   if (anObj.ORGI_CentreCommun().IsInit())
      aRes->AddFils(ToXMLTree(anObj.ORGI_CentreCommun().Val())->ReTagThis("ORGI_CentreCommun"));
   if (anObj.ORGI_TetaCommun().IsInit())
      aRes->AddFils(ToXMLTree(anObj.ORGI_TetaCommun().Val())->ReTagThis("ORGI_TetaCommun"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cObsRigidGrpImage & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.RefGrp(),aTree->Get("RefGrp",1)); //tototo 

   xml_init(anObj.ORGI_CentreCommun(),aTree->Get("ORGI_CentreCommun",1)); //tototo 

   xml_init(anObj.ORGI_TetaCommun(),aTree->Get("ORGI_TetaCommun",1)); //tototo 
}

std::string  Mangling( cObsRigidGrpImage *) {return "AE1EFF5CABB7D6C9FF3F";};


std::string & cTxtRapDetaille::NameFile()
{
   return mNameFile;
}

const std::string & cTxtRapDetaille::NameFile()const 
{
   return mNameFile;
}

void  BinaryUnDumpFromFile(cTxtRapDetaille & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.NameFile(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cTxtRapDetaille & anObj)
{
    BinaryDumpInFile(aFp,anObj.NameFile());
}

cElXMLTree * ToXMLTree(const cTxtRapDetaille & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"TxtRapDetaille",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("NameFile"),anObj.NameFile())->ReTagThis("NameFile"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cTxtRapDetaille & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.NameFile(),aTree->Get("NameFile",1)); //tototo 
}

std::string  Mangling( cTxtRapDetaille *) {return "57C6422C1B465E84FD3F";};


std::string & cObsRelGPS::Id()
{
   return mId;
}

const std::string & cObsRelGPS::Id()const 
{
   return mId;
}


cGpsRelativeWeighting & cObsRelGPS::Pond()
{
   return mPond;
}

const cGpsRelativeWeighting & cObsRelGPS::Pond()const 
{
   return mPond;
}

void  BinaryUnDumpFromFile(cObsRelGPS & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Id(),aFp);
    BinaryUnDumpFromFile(anObj.Pond(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cObsRelGPS & anObj)
{
    BinaryDumpInFile(aFp,anObj.Id());
    BinaryDumpInFile(aFp,anObj.Pond());
}

cElXMLTree * ToXMLTree(const cObsRelGPS & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ObsRelGPS",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Id"),anObj.Id())->ReTagThis("Id"));
   aRes->AddFils(ToXMLTree(anObj.Pond())->ReTagThis("Pond"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cObsRelGPS & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.Id(),aTree->Get("Id",1)); //tototo 

   xml_init(anObj.Pond(),aTree->Get("Pond",1)); //tototo 
}

std::string  Mangling( cObsRelGPS *) {return "900F479610F790BFFC3F";};


double & cSectionObservations::SigmaPix()
{
   return ContrCamConseq().Val().SigmaPix();
}

const double & cSectionObservations::SigmaPix()const 
{
   return ContrCamConseq().Val().SigmaPix();
}


int & cSectionObservations::NbGrid()
{
   return ContrCamConseq().Val().NbGrid();
}

const int & cSectionObservations::NbGrid()const 
{
   return ContrCamConseq().Val().NbGrid();
}


cTplValGesInit< cContrCamConseq > & cSectionObservations::ContrCamConseq()
{
   return mContrCamConseq;
}

const cTplValGesInit< cContrCamConseq > & cSectionObservations::ContrCamConseq()const 
{
   return mContrCamConseq;
}


std::list< cContrCamGenInc > & cSectionObservations::ContrCamGenInc()
{
   return mContrCamGenInc;
}

const std::list< cContrCamGenInc > & cSectionObservations::ContrCamGenInc()const 
{
   return mContrCamGenInc;
}


std::list< cObsBlockCamRig > & cSectionObservations::ObsBlockCamRig()
{
   return mObsBlockCamRig;
}

const std::list< cObsBlockCamRig > & cSectionObservations::ObsBlockCamRig()const 
{
   return mObsBlockCamRig;
}


std::list< cObsCenterInPlane > & cSectionObservations::ObsCenterInPlane()
{
   return mObsCenterInPlane;
}

const std::list< cObsCenterInPlane > & cSectionObservations::ObsCenterInPlane()const 
{
   return mObsCenterInPlane;
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


std::list< cObsRelGPS > & cSectionObservations::ObsRelGPS()
{
   return mObsRelGPS;
}

const std::list< cObsRelGPS > & cSectionObservations::ObsRelGPS()const 
{
   return mObsRelGPS;
}

void  BinaryUnDumpFromFile(cSectionObservations & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ContrCamConseq().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ContrCamConseq().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ContrCamConseq().SetNoInit();
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cContrCamGenInc aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.ContrCamGenInc().push_back(aVal);
        }
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cObsBlockCamRig aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.ObsBlockCamRig().push_back(aVal);
        }
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cObsCenterInPlane aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.ObsCenterInPlane().push_back(aVal);
        }
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cObsAppuis aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.ObsAppuis().push_back(aVal);
        }
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cObsAppuisFlottant aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.ObsAppuisFlottant().push_back(aVal);
        }
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cObsLiaisons aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.ObsLiaisons().push_back(aVal);
        }
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cObsCentrePDV aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.ObsCentrePDV().push_back(aVal);
        }
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cObsRigidGrpImage aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.ObsRigidGrpImage().push_back(aVal);
        }
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.TxtRapDetaille().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.TxtRapDetaille().ValForcedForUnUmp(),aFp);
        }
        else  anObj.TxtRapDetaille().SetNoInit();
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cObsRelGPS aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.ObsRelGPS().push_back(aVal);
        }
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cSectionObservations & anObj)
{
    BinaryDumpInFile(aFp,anObj.ContrCamConseq().IsInit());
    if (anObj.ContrCamConseq().IsInit()) BinaryDumpInFile(aFp,anObj.ContrCamConseq().Val());
    BinaryDumpInFile(aFp,(int)anObj.ContrCamGenInc().size());
    for(  std::list< cContrCamGenInc >::const_iterator iT=anObj.ContrCamGenInc().begin();
         iT!=anObj.ContrCamGenInc().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,(int)anObj.ObsBlockCamRig().size());
    for(  std::list< cObsBlockCamRig >::const_iterator iT=anObj.ObsBlockCamRig().begin();
         iT!=anObj.ObsBlockCamRig().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,(int)anObj.ObsCenterInPlane().size());
    for(  std::list< cObsCenterInPlane >::const_iterator iT=anObj.ObsCenterInPlane().begin();
         iT!=anObj.ObsCenterInPlane().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,(int)anObj.ObsAppuis().size());
    for(  std::list< cObsAppuis >::const_iterator iT=anObj.ObsAppuis().begin();
         iT!=anObj.ObsAppuis().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,(int)anObj.ObsAppuisFlottant().size());
    for(  std::list< cObsAppuisFlottant >::const_iterator iT=anObj.ObsAppuisFlottant().begin();
         iT!=anObj.ObsAppuisFlottant().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,(int)anObj.ObsLiaisons().size());
    for(  std::list< cObsLiaisons >::const_iterator iT=anObj.ObsLiaisons().begin();
         iT!=anObj.ObsLiaisons().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,(int)anObj.ObsCentrePDV().size());
    for(  std::list< cObsCentrePDV >::const_iterator iT=anObj.ObsCentrePDV().begin();
         iT!=anObj.ObsCentrePDV().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,(int)anObj.ObsRigidGrpImage().size());
    for(  std::list< cObsRigidGrpImage >::const_iterator iT=anObj.ObsRigidGrpImage().begin();
         iT!=anObj.ObsRigidGrpImage().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.TxtRapDetaille().IsInit());
    if (anObj.TxtRapDetaille().IsInit()) BinaryDumpInFile(aFp,anObj.TxtRapDetaille().Val());
    BinaryDumpInFile(aFp,(int)anObj.ObsRelGPS().size());
    for(  std::list< cObsRelGPS >::const_iterator iT=anObj.ObsRelGPS().begin();
         iT!=anObj.ObsRelGPS().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
}

cElXMLTree * ToXMLTree(const cSectionObservations & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"SectionObservations",eXMLBranche);
   if (anObj.ContrCamConseq().IsInit())
      aRes->AddFils(ToXMLTree(anObj.ContrCamConseq().Val())->ReTagThis("ContrCamConseq"));
  for
  (       std::list< cContrCamGenInc >::const_iterator it=anObj.ContrCamGenInc().begin();
      it !=anObj.ContrCamGenInc().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("ContrCamGenInc"));
  for
  (       std::list< cObsBlockCamRig >::const_iterator it=anObj.ObsBlockCamRig().begin();
      it !=anObj.ObsBlockCamRig().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("ObsBlockCamRig"));
  for
  (       std::list< cObsCenterInPlane >::const_iterator it=anObj.ObsCenterInPlane().begin();
      it !=anObj.ObsCenterInPlane().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("ObsCenterInPlane"));
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
  for
  (       std::list< cObsRelGPS >::const_iterator it=anObj.ObsRelGPS().begin();
      it !=anObj.ObsRelGPS().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("ObsRelGPS"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cSectionObservations & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.ContrCamConseq(),aTree->Get("ContrCamConseq",1)); //tototo 

   xml_init(anObj.ContrCamGenInc(),aTree->GetAll("ContrCamGenInc",false,1));

   xml_init(anObj.ObsBlockCamRig(),aTree->GetAll("ObsBlockCamRig",false,1));

   xml_init(anObj.ObsCenterInPlane(),aTree->GetAll("ObsCenterInPlane",false,1));

   xml_init(anObj.ObsAppuis(),aTree->GetAll("ObsAppuis",false,1));

   xml_init(anObj.ObsAppuisFlottant(),aTree->GetAll("ObsAppuisFlottant",false,1));

   xml_init(anObj.ObsLiaisons(),aTree->GetAll("ObsLiaisons",false,1));

   xml_init(anObj.ObsCentrePDV(),aTree->GetAll("ObsCentrePDV",false,1));

   xml_init(anObj.ObsRigidGrpImage(),aTree->GetAll("ObsRigidGrpImage",false,1));

   xml_init(anObj.TxtRapDetaille(),aTree->Get("TxtRapDetaille",1)); //tototo 

   xml_init(anObj.ObsRelGPS(),aTree->GetAll("ObsRelGPS",false,1));
}

std::string  Mangling( cSectionObservations *) {return "B511A440F3F96695FF3F";};


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

void  BinaryUnDumpFromFile(cExportAsGrid & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.DoExport().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.DoExport().ValForcedForUnUmp(),aFp);
        }
        else  anObj.DoExport().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.Name(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.XML_Supl().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.XML_Supl().ValForcedForUnUmp(),aFp);
        }
        else  anObj.XML_Supl().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.XML_Autonome().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.XML_Autonome().ValForcedForUnUmp(),aFp);
        }
        else  anObj.XML_Autonome().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.RabPt().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.RabPt().ValForcedForUnUmp(),aFp);
        }
        else  anObj.RabPt().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Step().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Step().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Step().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cExportAsGrid & anObj)
{
    BinaryDumpInFile(aFp,anObj.DoExport().IsInit());
    if (anObj.DoExport().IsInit()) BinaryDumpInFile(aFp,anObj.DoExport().Val());
    BinaryDumpInFile(aFp,anObj.Name());
    BinaryDumpInFile(aFp,anObj.XML_Supl().IsInit());
    if (anObj.XML_Supl().IsInit()) BinaryDumpInFile(aFp,anObj.XML_Supl().Val());
    BinaryDumpInFile(aFp,anObj.XML_Autonome().IsInit());
    if (anObj.XML_Autonome().IsInit()) BinaryDumpInFile(aFp,anObj.XML_Autonome().Val());
    BinaryDumpInFile(aFp,anObj.RabPt().IsInit());
    if (anObj.RabPt().IsInit()) BinaryDumpInFile(aFp,anObj.RabPt().Val());
    BinaryDumpInFile(aFp,anObj.Step().IsInit());
    if (anObj.Step().IsInit()) BinaryDumpInFile(aFp,anObj.Step().Val());
}

cElXMLTree * ToXMLTree(const cExportAsGrid & anObj)
{
  XMLPushContext(anObj.mGXml);
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
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cExportAsGrid & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.DoExport(),aTree->Get("DoExport",1),bool(true)); //tototo 

   xml_init(anObj.Name(),aTree->Get("Name",1)); //tototo 

   xml_init(anObj.XML_Supl(),aTree->Get("XML_Supl",1)); //tototo 

   xml_init(anObj.XML_Autonome(),aTree->Get("XML_Autonome",1),bool(false)); //tototo 

   xml_init(anObj.RabPt(),aTree->Get("RabPt",1),Pt2dr(Pt2dr(200,200))); //tototo 

   xml_init(anObj.Step(),aTree->Get("Step",1),Pt2dr(Pt2dr(20,20))); //tototo 
}

std::string  Mangling( cExportAsGrid *) {return "0876CEE5A09E14AEFE3F";};


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

void  BinaryUnDumpFromFile(cExportCalib & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.PatternSel().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.PatternSel().ValForcedForUnUmp(),aFp);
        }
        else  anObj.PatternSel().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.KeyAssoc(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Prefix().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Prefix().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Prefix().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Postfix().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Postfix().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Postfix().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.KeyIsName().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.KeyIsName().ValForcedForUnUmp(),aFp);
        }
        else  anObj.KeyIsName().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ExportAsGrid().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ExportAsGrid().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ExportAsGrid().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ExportAsNewGrid().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ExportAsNewGrid().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ExportAsNewGrid().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cExportCalib & anObj)
{
    BinaryDumpInFile(aFp,anObj.PatternSel().IsInit());
    if (anObj.PatternSel().IsInit()) BinaryDumpInFile(aFp,anObj.PatternSel().Val());
    BinaryDumpInFile(aFp,anObj.KeyAssoc());
    BinaryDumpInFile(aFp,anObj.Prefix().IsInit());
    if (anObj.Prefix().IsInit()) BinaryDumpInFile(aFp,anObj.Prefix().Val());
    BinaryDumpInFile(aFp,anObj.Postfix().IsInit());
    if (anObj.Postfix().IsInit()) BinaryDumpInFile(aFp,anObj.Postfix().Val());
    BinaryDumpInFile(aFp,anObj.KeyIsName().IsInit());
    if (anObj.KeyIsName().IsInit()) BinaryDumpInFile(aFp,anObj.KeyIsName().Val());
    BinaryDumpInFile(aFp,anObj.ExportAsGrid().IsInit());
    if (anObj.ExportAsGrid().IsInit()) BinaryDumpInFile(aFp,anObj.ExportAsGrid().Val());
    BinaryDumpInFile(aFp,anObj.ExportAsNewGrid().IsInit());
    if (anObj.ExportAsNewGrid().IsInit()) BinaryDumpInFile(aFp,anObj.ExportAsNewGrid().Val());
}

cElXMLTree * ToXMLTree(const cExportCalib & anObj)
{
  XMLPushContext(anObj.mGXml);
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
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cExportCalib & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.PatternSel(),aTree->Get("PatternSel",1),std::string(".*")); //tototo 

   xml_init(anObj.KeyAssoc(),aTree->Get("KeyAssoc",1)); //tototo 

   xml_init(anObj.Prefix(),aTree->Get("Prefix",1),std::string("")); //tototo 

   xml_init(anObj.Postfix(),aTree->Get("Postfix",1),std::string("")); //tototo 

   xml_init(anObj.KeyIsName(),aTree->Get("KeyIsName",1),bool(false)); //tototo 

   xml_init(anObj.ExportAsGrid(),aTree->Get("ExportAsGrid",1)); //tototo 

   xml_init(anObj.ExportAsNewGrid(),aTree->Get("ExportAsNewGrid",1)); //tototo 
}

std::string  Mangling( cExportCalib *) {return "40A45FC0E7AB4AC7FDBF";};


cTplValGesInit< bool > & cForce2ObsOnC::WhenExist()
{
   return mWhenExist;
}

const cTplValGesInit< bool > & cForce2ObsOnC::WhenExist()const 
{
   return mWhenExist;
}

void  BinaryUnDumpFromFile(cForce2ObsOnC & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.WhenExist().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.WhenExist().ValForcedForUnUmp(),aFp);
        }
        else  anObj.WhenExist().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cForce2ObsOnC & anObj)
{
    BinaryDumpInFile(aFp,anObj.WhenExist().IsInit());
    if (anObj.WhenExist().IsInit()) BinaryDumpInFile(aFp,anObj.WhenExist().Val());
}

cElXMLTree * ToXMLTree(const cForce2ObsOnC & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"Force2ObsOnC",eXMLBranche);
   if (anObj.WhenExist().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("WhenExist"),anObj.WhenExist().Val())->ReTagThis("WhenExist"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cForce2ObsOnC & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.WhenExist(),aTree->Get("WhenExist",1),bool(false)); //tototo 
}

std::string  Mangling( cForce2ObsOnC *) {return "85BDB9F433974586FEBF";};


cTplValGesInit< bool > & cExportPose::WhenExist()
{
   return Force2ObsOnC().Val().WhenExist();
}

const cTplValGesInit< bool > & cExportPose::WhenExist()const 
{
   return Force2ObsOnC().Val().WhenExist();
}


cTplValGesInit< cForce2ObsOnC > & cExportPose::Force2ObsOnC()
{
   return mForce2ObsOnC;
}

const cTplValGesInit< cForce2ObsOnC > & cExportPose::Force2ObsOnC()const 
{
   return mForce2ObsOnC;
}


cTplValGesInit< std::string > & cExportPose::ChC()
{
   return mChC;
}

const cTplValGesInit< std::string > & cExportPose::ChC()const 
{
   return mChC;
}


cTplValGesInit< bool > & cExportPose::ChCForceRot()
{
   return mChCForceRot;
}

const cTplValGesInit< bool > & cExportPose::ChCForceRot()const 
{
   return mChCForceRot;
}


std::string & cExportPose::KeyAssoc()
{
   return mKeyAssoc;
}

const std::string & cExportPose::KeyAssoc()const 
{
   return mKeyAssoc;
}


cTplValGesInit< std::string > & cExportPose::StdNameMMDir()
{
   return mStdNameMMDir;
}

const cTplValGesInit< std::string > & cExportPose::StdNameMMDir()const 
{
   return mStdNameMMDir;
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

void  BinaryUnDumpFromFile(cExportPose & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Force2ObsOnC().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Force2ObsOnC().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Force2ObsOnC().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ChC().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ChC().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ChC().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ChCForceRot().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ChCForceRot().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ChCForceRot().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.KeyAssoc(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.StdNameMMDir().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.StdNameMMDir().ValForcedForUnUmp(),aFp);
        }
        else  anObj.StdNameMMDir().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.AddCalib().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.AddCalib().ValForcedForUnUmp(),aFp);
        }
        else  anObj.AddCalib().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ExportAsNewGrid().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ExportAsNewGrid().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ExportAsNewGrid().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.FileExtern().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.FileExtern().ValForcedForUnUmp(),aFp);
        }
        else  anObj.FileExtern().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.FileExternIsKey().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.FileExternIsKey().ValForcedForUnUmp(),aFp);
        }
        else  anObj.FileExternIsKey().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.CalcKeyFromCalib().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.CalcKeyFromCalib().ValForcedForUnUmp(),aFp);
        }
        else  anObj.CalcKeyFromCalib().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.RelativeNameFE().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.RelativeNameFE().ValForcedForUnUmp(),aFp);
        }
        else  anObj.RelativeNameFE().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ModeAngulaire().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ModeAngulaire().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ModeAngulaire().SetNoInit();
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
             anObj.NbVerif().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NbVerif().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NbVerif().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.VerifDeterm().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.VerifDeterm().ValForcedForUnUmp(),aFp);
        }
        else  anObj.VerifDeterm().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ShowWhenVerif().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ShowWhenVerif().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ShowWhenVerif().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.TolWhenVerif().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.TolWhenVerif().ValForcedForUnUmp(),aFp);
        }
        else  anObj.TolWhenVerif().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cExportPose & anObj)
{
    BinaryDumpInFile(aFp,anObj.Force2ObsOnC().IsInit());
    if (anObj.Force2ObsOnC().IsInit()) BinaryDumpInFile(aFp,anObj.Force2ObsOnC().Val());
    BinaryDumpInFile(aFp,anObj.ChC().IsInit());
    if (anObj.ChC().IsInit()) BinaryDumpInFile(aFp,anObj.ChC().Val());
    BinaryDumpInFile(aFp,anObj.ChCForceRot().IsInit());
    if (anObj.ChCForceRot().IsInit()) BinaryDumpInFile(aFp,anObj.ChCForceRot().Val());
    BinaryDumpInFile(aFp,anObj.KeyAssoc());
    BinaryDumpInFile(aFp,anObj.StdNameMMDir().IsInit());
    if (anObj.StdNameMMDir().IsInit()) BinaryDumpInFile(aFp,anObj.StdNameMMDir().Val());
    BinaryDumpInFile(aFp,anObj.AddCalib().IsInit());
    if (anObj.AddCalib().IsInit()) BinaryDumpInFile(aFp,anObj.AddCalib().Val());
    BinaryDumpInFile(aFp,anObj.ExportAsNewGrid().IsInit());
    if (anObj.ExportAsNewGrid().IsInit()) BinaryDumpInFile(aFp,anObj.ExportAsNewGrid().Val());
    BinaryDumpInFile(aFp,anObj.FileExtern().IsInit());
    if (anObj.FileExtern().IsInit()) BinaryDumpInFile(aFp,anObj.FileExtern().Val());
    BinaryDumpInFile(aFp,anObj.FileExternIsKey().IsInit());
    if (anObj.FileExternIsKey().IsInit()) BinaryDumpInFile(aFp,anObj.FileExternIsKey().Val());
    BinaryDumpInFile(aFp,anObj.CalcKeyFromCalib().IsInit());
    if (anObj.CalcKeyFromCalib().IsInit()) BinaryDumpInFile(aFp,anObj.CalcKeyFromCalib().Val());
    BinaryDumpInFile(aFp,anObj.RelativeNameFE().IsInit());
    if (anObj.RelativeNameFE().IsInit()) BinaryDumpInFile(aFp,anObj.RelativeNameFE().Val());
    BinaryDumpInFile(aFp,anObj.ModeAngulaire().IsInit());
    if (anObj.ModeAngulaire().IsInit()) BinaryDumpInFile(aFp,anObj.ModeAngulaire().Val());
    BinaryDumpInFile(aFp,anObj.PatternSel().IsInit());
    if (anObj.PatternSel().IsInit()) BinaryDumpInFile(aFp,anObj.PatternSel().Val());
    BinaryDumpInFile(aFp,anObj.NbVerif().IsInit());
    if (anObj.NbVerif().IsInit()) BinaryDumpInFile(aFp,anObj.NbVerif().Val());
    BinaryDumpInFile(aFp,anObj.VerifDeterm().IsInit());
    if (anObj.VerifDeterm().IsInit()) BinaryDumpInFile(aFp,anObj.VerifDeterm().Val());
    BinaryDumpInFile(aFp,anObj.ShowWhenVerif().IsInit());
    if (anObj.ShowWhenVerif().IsInit()) BinaryDumpInFile(aFp,anObj.ShowWhenVerif().Val());
    BinaryDumpInFile(aFp,anObj.TolWhenVerif().IsInit());
    if (anObj.TolWhenVerif().IsInit()) BinaryDumpInFile(aFp,anObj.TolWhenVerif().Val());
}

cElXMLTree * ToXMLTree(const cExportPose & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ExportPose",eXMLBranche);
   if (anObj.Force2ObsOnC().IsInit())
      aRes->AddFils(ToXMLTree(anObj.Force2ObsOnC().Val())->ReTagThis("Force2ObsOnC"));
   if (anObj.ChC().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ChC"),anObj.ChC().Val())->ReTagThis("ChC"));
   if (anObj.ChCForceRot().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ChCForceRot"),anObj.ChCForceRot().Val())->ReTagThis("ChCForceRot"));
   aRes->AddFils(::ToXMLTree(std::string("KeyAssoc"),anObj.KeyAssoc())->ReTagThis("KeyAssoc"));
   if (anObj.StdNameMMDir().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("StdNameMMDir"),anObj.StdNameMMDir().Val())->ReTagThis("StdNameMMDir"));
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
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cExportPose & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.Force2ObsOnC(),aTree->Get("Force2ObsOnC",1)); //tototo 

   xml_init(anObj.ChC(),aTree->Get("ChC",1)); //tototo 

   xml_init(anObj.ChCForceRot(),aTree->Get("ChCForceRot",1),bool(false)); //tototo 

   xml_init(anObj.KeyAssoc(),aTree->Get("KeyAssoc",1)); //tototo 

   xml_init(anObj.StdNameMMDir(),aTree->Get("StdNameMMDir",1)); //tototo 

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

std::string  Mangling( cExportPose *) {return "EE750CF0C96C05F1FD3F";};


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

void  BinaryUnDumpFromFile(cExportAttrPose & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.KeyAssoc(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.AttrSup().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.AttrSup().ValForcedForUnUmp(),aFp);
        }
        else  anObj.AttrSup().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.PatternApply(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ExportDirVerticaleLocale().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ExportDirVerticaleLocale().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ExportDirVerticaleLocale().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cExportAttrPose & anObj)
{
    BinaryDumpInFile(aFp,anObj.KeyAssoc());
    BinaryDumpInFile(aFp,anObj.AttrSup().IsInit());
    if (anObj.AttrSup().IsInit()) BinaryDumpInFile(aFp,anObj.AttrSup().Val());
    BinaryDumpInFile(aFp,anObj.PatternApply());
    BinaryDumpInFile(aFp,anObj.ExportDirVerticaleLocale().IsInit());
    if (anObj.ExportDirVerticaleLocale().IsInit()) BinaryDumpInFile(aFp,anObj.ExportDirVerticaleLocale().Val());
}

cElXMLTree * ToXMLTree(const cExportAttrPose & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ExportAttrPose",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("KeyAssoc"),anObj.KeyAssoc())->ReTagThis("KeyAssoc"));
   if (anObj.AttrSup().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("AttrSup"),anObj.AttrSup().Val())->ReTagThis("AttrSup"));
   aRes->AddFils(::ToXMLTree(std::string("PatternApply"),anObj.PatternApply())->ReTagThis("PatternApply"));
   if (anObj.ExportDirVerticaleLocale().IsInit())
      aRes->AddFils(ToXMLTree(anObj.ExportDirVerticaleLocale().Val())->ReTagThis("ExportDirVerticaleLocale"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cExportAttrPose & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.KeyAssoc(),aTree->Get("KeyAssoc",1)); //tototo 

   xml_init(anObj.AttrSup(),aTree->Get("AttrSup",1)); //tototo 

   xml_init(anObj.PatternApply(),aTree->Get("PatternApply",1)); //tototo 

   xml_init(anObj.ExportDirVerticaleLocale(),aTree->Get("ExportDirVerticaleLocale",1)); //tototo 
}

std::string  Mangling( cExportAttrPose *) {return "10C43B70FF363AE2FA3F";};


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

void  BinaryUnDumpFromFile(cExportOrthoCyl & anObj,ELISE_fp & aFp)
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
             anObj.PatternEstimAxe().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.PatternEstimAxe().ValForcedForUnUmp(),aFp);
        }
        else  anObj.PatternEstimAxe().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.AngulCorr(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.L2EstimAxe().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.L2EstimAxe().ValForcedForUnUmp(),aFp);
        }
        else  anObj.L2EstimAxe().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cExportOrthoCyl & anObj)
{
    BinaryDumpInFile(aFp,anObj.UseIt().IsInit());
    if (anObj.UseIt().IsInit()) BinaryDumpInFile(aFp,anObj.UseIt().Val());
    BinaryDumpInFile(aFp,anObj.PatternEstimAxe().IsInit());
    if (anObj.PatternEstimAxe().IsInit()) BinaryDumpInFile(aFp,anObj.PatternEstimAxe().Val());
    BinaryDumpInFile(aFp,anObj.AngulCorr());
    BinaryDumpInFile(aFp,anObj.L2EstimAxe().IsInit());
    if (anObj.L2EstimAxe().IsInit()) BinaryDumpInFile(aFp,anObj.L2EstimAxe().Val());
}

cElXMLTree * ToXMLTree(const cExportOrthoCyl & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ExportOrthoCyl",eXMLBranche);
   if (anObj.UseIt().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("UseIt"),anObj.UseIt().Val())->ReTagThis("UseIt"));
   if (anObj.PatternEstimAxe().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PatternEstimAxe"),anObj.PatternEstimAxe().Val())->ReTagThis("PatternEstimAxe"));
   aRes->AddFils(::ToXMLTree(std::string("AngulCorr"),anObj.AngulCorr())->ReTagThis("AngulCorr"));
   if (anObj.L2EstimAxe().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("L2EstimAxe"),anObj.L2EstimAxe().Val())->ReTagThis("L2EstimAxe"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cExportOrthoCyl & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.UseIt(),aTree->Get("UseIt",1),bool(true)); //tototo 

   xml_init(anObj.PatternEstimAxe(),aTree->Get("PatternEstimAxe",1)); //tototo 

   xml_init(anObj.AngulCorr(),aTree->Get("AngulCorr",1)); //tototo 

   xml_init(anObj.L2EstimAxe(),aTree->Get("L2EstimAxe",1),bool(true)); //tototo 
}

std::string  Mangling( cExportOrthoCyl *) {return "66F681BB4D1A6C90FE3F";};


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


cTplValGesInit< bool > & cExportRepereLoc::P1P2Hor()
{
   return mP1P2Hor;
}

const cTplValGesInit< bool > & cExportRepereLoc::P1P2Hor()const 
{
   return mP1P2Hor;
}


cTplValGesInit< bool > & cExportRepereLoc::P1P2HorYVert()
{
   return mP1P2HorYVert;
}

const cTplValGesInit< bool > & cExportRepereLoc::P1P2HorYVert()const 
{
   return mP1P2HorYVert;
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

void  BinaryUnDumpFromFile(cExportRepereLoc & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.NameRepere(),aFp);
    BinaryUnDumpFromFile(anObj.PatternEstimPl(),aFp);
    BinaryUnDumpFromFile(anObj.EstimPlanHor(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ImP1P2().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ImP1P2().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ImP1P2().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.P1(),aFp);
    BinaryUnDumpFromFile(anObj.P2(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.AxeDef().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.AxeDef().ValForcedForUnUmp(),aFp);
        }
        else  anObj.AxeDef().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Origine().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Origine().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Origine().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NameImOri().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NameImOri().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NameImOri().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.P1P2Hor().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.P1P2Hor().ValForcedForUnUmp(),aFp);
        }
        else  anObj.P1P2Hor().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.P1P2HorYVert().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.P1P2HorYVert().ValForcedForUnUmp(),aFp);
        }
        else  anObj.P1P2HorYVert().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ExportOrthoCyl().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ExportOrthoCyl().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ExportOrthoCyl().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cExportRepereLoc & anObj)
{
    BinaryDumpInFile(aFp,anObj.NameRepere());
    BinaryDumpInFile(aFp,anObj.PatternEstimPl());
    BinaryDumpInFile(aFp,anObj.EstimPlanHor());
    BinaryDumpInFile(aFp,anObj.ImP1P2().IsInit());
    if (anObj.ImP1P2().IsInit()) BinaryDumpInFile(aFp,anObj.ImP1P2().Val());
    BinaryDumpInFile(aFp,anObj.P1());
    BinaryDumpInFile(aFp,anObj.P2());
    BinaryDumpInFile(aFp,anObj.AxeDef().IsInit());
    if (anObj.AxeDef().IsInit()) BinaryDumpInFile(aFp,anObj.AxeDef().Val());
    BinaryDumpInFile(aFp,anObj.Origine().IsInit());
    if (anObj.Origine().IsInit()) BinaryDumpInFile(aFp,anObj.Origine().Val());
    BinaryDumpInFile(aFp,anObj.NameImOri().IsInit());
    if (anObj.NameImOri().IsInit()) BinaryDumpInFile(aFp,anObj.NameImOri().Val());
    BinaryDumpInFile(aFp,anObj.P1P2Hor().IsInit());
    if (anObj.P1P2Hor().IsInit()) BinaryDumpInFile(aFp,anObj.P1P2Hor().Val());
    BinaryDumpInFile(aFp,anObj.P1P2HorYVert().IsInit());
    if (anObj.P1P2HorYVert().IsInit()) BinaryDumpInFile(aFp,anObj.P1P2HorYVert().Val());
    BinaryDumpInFile(aFp,anObj.ExportOrthoCyl().IsInit());
    if (anObj.ExportOrthoCyl().IsInit()) BinaryDumpInFile(aFp,anObj.ExportOrthoCyl().Val());
}

cElXMLTree * ToXMLTree(const cExportRepereLoc & anObj)
{
  XMLPushContext(anObj.mGXml);
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
   if (anObj.P1P2Hor().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("P1P2Hor"),anObj.P1P2Hor().Val())->ReTagThis("P1P2Hor"));
   if (anObj.P1P2HorYVert().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("P1P2HorYVert"),anObj.P1P2HorYVert().Val())->ReTagThis("P1P2HorYVert"));
   if (anObj.ExportOrthoCyl().IsInit())
      aRes->AddFils(ToXMLTree(anObj.ExportOrthoCyl().Val())->ReTagThis("ExportOrthoCyl"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cExportRepereLoc & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.NameRepere(),aTree->Get("NameRepere",1)); //tototo 

   xml_init(anObj.PatternEstimPl(),aTree->Get("PatternEstimPl",1)); //tototo 

   xml_init(anObj.EstimPlanHor(),aTree->Get("EstimPlanHor",1)); //tototo 

   xml_init(anObj.ImP1P2(),aTree->Get("ImP1P2",1)); //tototo 

   xml_init(anObj.P1(),aTree->Get("P1",1)); //tototo 

   xml_init(anObj.P2(),aTree->Get("P2",1)); //tototo 

   xml_init(anObj.AxeDef(),aTree->Get("AxeDef",1),Pt2dr(Pt2dr(1,0))); //tototo 

   xml_init(anObj.Origine(),aTree->Get("Origine",1)); //tototo 

   xml_init(anObj.NameImOri(),aTree->Get("NameImOri",1)); //tototo 

   xml_init(anObj.P1P2Hor(),aTree->Get("P1P2Hor",1),bool(false)); //tototo 

   xml_init(anObj.P1P2HorYVert(),aTree->Get("P1P2HorYVert",1),bool(false)); //tototo 

   xml_init(anObj.ExportOrthoCyl(),aTree->Get("ExportOrthoCyl",1)); //tototo 
}

std::string  Mangling( cExportRepereLoc *) {return "146E069118A724D9FB3F";};


std::string & cExportBlockCamera::Id()
{
   return mId;
}

const std::string & cExportBlockCamera::Id()const 
{
   return mId;
}


std::string & cExportBlockCamera::NameFile()
{
   return mNameFile;
}

const std::string & cExportBlockCamera::NameFile()const 
{
   return mNameFile;
}


cTplValGesInit< cXml_EstimateOrientationInitBlockCamera > & cExportBlockCamera::Estimate()
{
   return mEstimate;
}

const cTplValGesInit< cXml_EstimateOrientationInitBlockCamera > & cExportBlockCamera::Estimate()const 
{
   return mEstimate;
}

void  BinaryUnDumpFromFile(cExportBlockCamera & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Id(),aFp);
    BinaryUnDumpFromFile(anObj.NameFile(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Estimate().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Estimate().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Estimate().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cExportBlockCamera & anObj)
{
    BinaryDumpInFile(aFp,anObj.Id());
    BinaryDumpInFile(aFp,anObj.NameFile());
    BinaryDumpInFile(aFp,anObj.Estimate().IsInit());
    if (anObj.Estimate().IsInit()) BinaryDumpInFile(aFp,anObj.Estimate().Val());
}

cElXMLTree * ToXMLTree(const cExportBlockCamera & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ExportBlockCamera",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Id"),anObj.Id())->ReTagThis("Id"));
   aRes->AddFils(::ToXMLTree(std::string("NameFile"),anObj.NameFile())->ReTagThis("NameFile"));
   if (anObj.Estimate().IsInit())
      aRes->AddFils(ToXMLTree(anObj.Estimate().Val())->ReTagThis("Estimate"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cExportBlockCamera & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.Id(),aTree->Get("Id",1)); //tototo 

   xml_init(anObj.NameFile(),aTree->Get("NameFile",1)); //tototo 

   xml_init(anObj.Estimate(),aTree->Get("Estimate",1)); //tototo 
}

std::string  Mangling( cExportBlockCamera *) {return "5427958307805DE5FD3F";};


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

void  BinaryUnDumpFromFile(cCartes2Export & anObj,ELISE_fp & aFp)
{
   { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             std::string aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.Im1().push_back(aVal);
        }
  } ;
    BinaryUnDumpFromFile(anObj.Nuage(),aFp);
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             std::string aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.ImN().push_back(aVal);
        }
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.FilterIm2().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.FilterIm2().ValForcedForUnUmp(),aFp);
        }
        else  anObj.FilterIm2().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cCartes2Export & anObj)
{
    BinaryDumpInFile(aFp,(int)anObj.Im1().size());
    for(  std::list< std::string >::const_iterator iT=anObj.Im1().begin();
         iT!=anObj.Im1().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.Nuage());
    BinaryDumpInFile(aFp,(int)anObj.ImN().size());
    for(  std::list< std::string >::const_iterator iT=anObj.ImN().begin();
         iT!=anObj.ImN().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.FilterIm2().IsInit());
    if (anObj.FilterIm2().IsInit()) BinaryDumpInFile(aFp,anObj.FilterIm2().Val());
}

cElXMLTree * ToXMLTree(const cCartes2Export & anObj)
{
  XMLPushContext(anObj.mGXml);
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
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cCartes2Export & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.Im1(),aTree->GetAll("Im1",false,1));

   xml_init(anObj.Nuage(),aTree->Get("Nuage",1)); //tototo 

   xml_init(anObj.ImN(),aTree->GetAll("ImN",false,1));

   xml_init(anObj.FilterIm2(),aTree->Get("FilterIm2",1),std::string(".*")); //tototo 
}

std::string  Mangling( cCartes2Export *) {return "289C2ABA635076EEFB3F";};


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

void  BinaryUnDumpFromFile(cExportMesuresFromCarteProf & anObj,ELISE_fp & aFp)
{
   { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cCartes2Export aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.Cartes2Export().push_back(aVal);
        }
  } ;
    BinaryUnDumpFromFile(anObj.IdBdLiaisonIn(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.KeyAssocLiaisons12().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.KeyAssocLiaisons12().ValForcedForUnUmp(),aFp);
        }
        else  anObj.KeyAssocLiaisons12().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.KeyAssocLiaisons21().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.KeyAssocLiaisons21().ValForcedForUnUmp(),aFp);
        }
        else  anObj.KeyAssocLiaisons21().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.KeyAssocAppuis().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.KeyAssocAppuis().ValForcedForUnUmp(),aFp);
        }
        else  anObj.KeyAssocAppuis().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.AppuisModeAdd().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.AppuisModeAdd().ValForcedForUnUmp(),aFp);
        }
        else  anObj.AppuisModeAdd().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.LiaisonModeAdd().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.LiaisonModeAdd().ValForcedForUnUmp(),aFp);
        }
        else  anObj.LiaisonModeAdd().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cExportMesuresFromCarteProf & anObj)
{
    BinaryDumpInFile(aFp,(int)anObj.Cartes2Export().size());
    for(  std::list< cCartes2Export >::const_iterator iT=anObj.Cartes2Export().begin();
         iT!=anObj.Cartes2Export().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.IdBdLiaisonIn());
    BinaryDumpInFile(aFp,anObj.KeyAssocLiaisons12().IsInit());
    if (anObj.KeyAssocLiaisons12().IsInit()) BinaryDumpInFile(aFp,anObj.KeyAssocLiaisons12().Val());
    BinaryDumpInFile(aFp,anObj.KeyAssocLiaisons21().IsInit());
    if (anObj.KeyAssocLiaisons21().IsInit()) BinaryDumpInFile(aFp,anObj.KeyAssocLiaisons21().Val());
    BinaryDumpInFile(aFp,anObj.KeyAssocAppuis().IsInit());
    if (anObj.KeyAssocAppuis().IsInit()) BinaryDumpInFile(aFp,anObj.KeyAssocAppuis().Val());
    BinaryDumpInFile(aFp,anObj.AppuisModeAdd().IsInit());
    if (anObj.AppuisModeAdd().IsInit()) BinaryDumpInFile(aFp,anObj.AppuisModeAdd().Val());
    BinaryDumpInFile(aFp,anObj.LiaisonModeAdd().IsInit());
    if (anObj.LiaisonModeAdd().IsInit()) BinaryDumpInFile(aFp,anObj.LiaisonModeAdd().Val());
}

cElXMLTree * ToXMLTree(const cExportMesuresFromCarteProf & anObj)
{
  XMLPushContext(anObj.mGXml);
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
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cExportMesuresFromCarteProf & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.Cartes2Export(),aTree->GetAll("Cartes2Export",false,1));

   xml_init(anObj.IdBdLiaisonIn(),aTree->Get("IdBdLiaisonIn",1)); //tototo 

   xml_init(anObj.KeyAssocLiaisons12(),aTree->Get("KeyAssocLiaisons12",1)); //tototo 

   xml_init(anObj.KeyAssocLiaisons21(),aTree->Get("KeyAssocLiaisons21",1)); //tototo 

   xml_init(anObj.KeyAssocAppuis(),aTree->Get("KeyAssocAppuis",1)); //tototo 

   xml_init(anObj.AppuisModeAdd(),aTree->Get("AppuisModeAdd",1),bool(true)); //tototo 

   xml_init(anObj.LiaisonModeAdd(),aTree->Get("LiaisonModeAdd",1),bool(false)); //tototo 
}

std::string  Mangling( cExportMesuresFromCarteProf *) {return "8676B24FB3DFF582FF3F";};


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

void  BinaryUnDumpFromFile(cExportVisuConfigGrpPose & anObj,ELISE_fp & aFp)
{
   { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             std::string aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.PatternSel().push_back(aVal);
        }
  } ;
    BinaryUnDumpFromFile(anObj.NameFile(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cExportVisuConfigGrpPose & anObj)
{
    BinaryDumpInFile(aFp,(int)anObj.PatternSel().size());
    for(  std::list< std::string >::const_iterator iT=anObj.PatternSel().begin();
         iT!=anObj.PatternSel().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.NameFile());
}

cElXMLTree * ToXMLTree(const cExportVisuConfigGrpPose & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ExportVisuConfigGrpPose",eXMLBranche);
  for
  (       std::list< std::string >::const_iterator it=anObj.PatternSel().begin();
      it !=anObj.PatternSel().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("PatternSel"),(*it))->ReTagThis("PatternSel"));
   aRes->AddFils(::ToXMLTree(std::string("NameFile"),anObj.NameFile())->ReTagThis("NameFile"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cExportVisuConfigGrpPose & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.PatternSel(),aTree->GetAll("PatternSel",false,1));

   xml_init(anObj.NameFile(),aTree->Get("NameFile",1)); //tototo 
}

std::string  Mangling( cExportVisuConfigGrpPose *) {return "9F651CD6DD9DDF82FE3F";};


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


cTplValGesInit< std::string > & cExportPtsFlottant::NameFileJSON()
{
   return mNameFileJSON;
}

const cTplValGesInit< std::string > & cExportPtsFlottant::NameFileJSON()const 
{
   return mNameFileJSON;
}


cTplValGesInit< std::string > & cExportPtsFlottant::TextComplTxt()
{
   return mTextComplTxt;
}

const cTplValGesInit< std::string > & cExportPtsFlottant::TextComplTxt()const 
{
   return mTextComplTxt;
}

void  BinaryUnDumpFromFile(cExportPtsFlottant & anObj,ELISE_fp & aFp)
{
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
             anObj.NameFileXml().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NameFileXml().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NameFileXml().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NameFileTxt().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NameFileTxt().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NameFileTxt().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NameFileJSON().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NameFileJSON().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NameFileJSON().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.TextComplTxt().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.TextComplTxt().ValForcedForUnUmp(),aFp);
        }
        else  anObj.TextComplTxt().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cExportPtsFlottant & anObj)
{
    BinaryDumpInFile(aFp,anObj.PatternSel().IsInit());
    if (anObj.PatternSel().IsInit()) BinaryDumpInFile(aFp,anObj.PatternSel().Val());
    BinaryDumpInFile(aFp,anObj.NameFileXml().IsInit());
    if (anObj.NameFileXml().IsInit()) BinaryDumpInFile(aFp,anObj.NameFileXml().Val());
    BinaryDumpInFile(aFp,anObj.NameFileTxt().IsInit());
    if (anObj.NameFileTxt().IsInit()) BinaryDumpInFile(aFp,anObj.NameFileTxt().Val());
    BinaryDumpInFile(aFp,anObj.NameFileJSON().IsInit());
    if (anObj.NameFileJSON().IsInit()) BinaryDumpInFile(aFp,anObj.NameFileJSON().Val());
    BinaryDumpInFile(aFp,anObj.TextComplTxt().IsInit());
    if (anObj.TextComplTxt().IsInit()) BinaryDumpInFile(aFp,anObj.TextComplTxt().Val());
}

cElXMLTree * ToXMLTree(const cExportPtsFlottant & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ExportPtsFlottant",eXMLBranche);
   if (anObj.PatternSel().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PatternSel"),anObj.PatternSel().Val())->ReTagThis("PatternSel"));
   if (anObj.NameFileXml().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NameFileXml"),anObj.NameFileXml().Val())->ReTagThis("NameFileXml"));
   if (anObj.NameFileTxt().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NameFileTxt"),anObj.NameFileTxt().Val())->ReTagThis("NameFileTxt"));
   if (anObj.NameFileJSON().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NameFileJSON"),anObj.NameFileJSON().Val())->ReTagThis("NameFileJSON"));
   if (anObj.TextComplTxt().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("TextComplTxt"),anObj.TextComplTxt().Val())->ReTagThis("TextComplTxt"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cExportPtsFlottant & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.PatternSel(),aTree->Get("PatternSel",1),std::string(".*")); //tototo 

   xml_init(anObj.NameFileXml(),aTree->Get("NameFileXml",1)); //tototo 

   xml_init(anObj.NameFileTxt(),aTree->Get("NameFileTxt",1)); //tototo 

   xml_init(anObj.NameFileJSON(),aTree->Get("NameFileJSON",1)); //tototo 

   xml_init(anObj.TextComplTxt(),aTree->Get("TextComplTxt",1)); //tototo 
}

std::string  Mangling( cExportPtsFlottant *) {return "70608FBC0A37809FFE3F";};


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

void  BinaryUnDumpFromFile(cResidusIndiv & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Pattern(),aFp);
    BinaryUnDumpFromFile(anObj.Name(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cResidusIndiv & anObj)
{
    BinaryDumpInFile(aFp,anObj.Pattern());
    BinaryDumpInFile(aFp,anObj.Name());
}

cElXMLTree * ToXMLTree(const cResidusIndiv & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ResidusIndiv",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Pattern"),anObj.Pattern())->ReTagThis("Pattern"));
   aRes->AddFils(::ToXMLTree(std::string("Name"),anObj.Name())->ReTagThis("Name"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cResidusIndiv & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.Pattern(),aTree->Get("Pattern",1)); //tototo 

   xml_init(anObj.Name(),aTree->Get("Name",1)); //tototo 
}

std::string  Mangling( cResidusIndiv *) {return "D80FD46784BE21D2FE3F";};


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

void  BinaryUnDumpFromFile(cExportImResiduLiaison & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Signed().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Signed().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Signed().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.PatternGlobCalIm(),aFp);
    BinaryUnDumpFromFile(anObj.NameGlobCalIm(),aFp);
    BinaryUnDumpFromFile(anObj.ScaleIm(),aFp);
    BinaryUnDumpFromFile(anObj.DynIm(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ResidusIndiv().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ResidusIndiv().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ResidusIndiv().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cExportImResiduLiaison & anObj)
{
    BinaryDumpInFile(aFp,anObj.Signed().IsInit());
    if (anObj.Signed().IsInit()) BinaryDumpInFile(aFp,anObj.Signed().Val());
    BinaryDumpInFile(aFp,anObj.PatternGlobCalIm());
    BinaryDumpInFile(aFp,anObj.NameGlobCalIm());
    BinaryDumpInFile(aFp,anObj.ScaleIm());
    BinaryDumpInFile(aFp,anObj.DynIm());
    BinaryDumpInFile(aFp,anObj.ResidusIndiv().IsInit());
    if (anObj.ResidusIndiv().IsInit()) BinaryDumpInFile(aFp,anObj.ResidusIndiv().Val());
}

cElXMLTree * ToXMLTree(const cExportImResiduLiaison & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ExportImResiduLiaison",eXMLBranche);
   if (anObj.Signed().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Signed"),anObj.Signed().Val())->ReTagThis("Signed"));
   aRes->AddFils(::ToXMLTree(std::string("PatternGlobCalIm"),anObj.PatternGlobCalIm())->ReTagThis("PatternGlobCalIm"));
   aRes->AddFils(::ToXMLTree(std::string("NameGlobCalIm"),anObj.NameGlobCalIm())->ReTagThis("NameGlobCalIm"));
   aRes->AddFils(::ToXMLTree(std::string("ScaleIm"),anObj.ScaleIm())->ReTagThis("ScaleIm"));
   aRes->AddFils(::ToXMLTree(std::string("DynIm"),anObj.DynIm())->ReTagThis("DynIm"));
   if (anObj.ResidusIndiv().IsInit())
      aRes->AddFils(ToXMLTree(anObj.ResidusIndiv().Val())->ReTagThis("ResidusIndiv"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cExportImResiduLiaison & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.Signed(),aTree->Get("Signed",1),bool(true)); //tototo 

   xml_init(anObj.PatternGlobCalIm(),aTree->Get("PatternGlobCalIm",1)); //tototo 

   xml_init(anObj.NameGlobCalIm(),aTree->Get("NameGlobCalIm",1)); //tototo 

   xml_init(anObj.ScaleIm(),aTree->Get("ScaleIm",1)); //tototo 

   xml_init(anObj.DynIm(),aTree->Get("DynIm",1)); //tototo 

   xml_init(anObj.ResidusIndiv(),aTree->Get("ResidusIndiv",1)); //tototo 
}

std::string  Mangling( cExportImResiduLiaison *) {return "8839DF5FEEACE5B1FF3F";};


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

void  BinaryUnDumpFromFile(cExportRedressement & anObj,ELISE_fp & aFp)
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
             anObj.Gamma().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Gamma().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Gamma().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.TypeNum().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.TypeNum().ValForcedForUnUmp(),aFp);
        }
        else  anObj.TypeNum().SetNoInit();
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
             anObj.PatternSel().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.PatternSel().ValForcedForUnUmp(),aFp);
        }
        else  anObj.PatternSel().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.KeyAssocIn().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.KeyAssocIn().ValForcedForUnUmp(),aFp);
        }
        else  anObj.KeyAssocIn().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.OffsetIm().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.OffsetIm().ValForcedForUnUmp(),aFp);
        }
        else  anObj.OffsetIm().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ScaleIm().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ScaleIm().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ScaleIm().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.KeyAssocOut(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ZSol().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ZSol().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ZSol().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.Resol(),aFp);
    BinaryUnDumpFromFile(anObj.ResolIsRel(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.DoTFW().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.DoTFW().ValForcedForUnUmp(),aFp);
        }
        else  anObj.DoTFW().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.TetaLimite(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.DirTetaLim().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.DirTetaLim().ValForcedForUnUmp(),aFp);
        }
        else  anObj.DirTetaLim().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.DoOnlyIfNew().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.DoOnlyIfNew().ValForcedForUnUmp(),aFp);
        }
        else  anObj.DoOnlyIfNew().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cExportRedressement & anObj)
{
    BinaryDumpInFile(aFp,anObj.Dyn().IsInit());
    if (anObj.Dyn().IsInit()) BinaryDumpInFile(aFp,anObj.Dyn().Val());
    BinaryDumpInFile(aFp,anObj.Gamma().IsInit());
    if (anObj.Gamma().IsInit()) BinaryDumpInFile(aFp,anObj.Gamma().Val());
    BinaryDumpInFile(aFp,anObj.TypeNum().IsInit());
    if (anObj.TypeNum().IsInit()) BinaryDumpInFile(aFp,anObj.TypeNum().Val());
    BinaryDumpInFile(aFp,anObj.Offset().IsInit());
    if (anObj.Offset().IsInit()) BinaryDumpInFile(aFp,anObj.Offset().Val());
    BinaryDumpInFile(aFp,anObj.PatternSel().IsInit());
    if (anObj.PatternSel().IsInit()) BinaryDumpInFile(aFp,anObj.PatternSel().Val());
    BinaryDumpInFile(aFp,anObj.KeyAssocIn().IsInit());
    if (anObj.KeyAssocIn().IsInit()) BinaryDumpInFile(aFp,anObj.KeyAssocIn().Val());
    BinaryDumpInFile(aFp,anObj.OffsetIm().IsInit());
    if (anObj.OffsetIm().IsInit()) BinaryDumpInFile(aFp,anObj.OffsetIm().Val());
    BinaryDumpInFile(aFp,anObj.ScaleIm().IsInit());
    if (anObj.ScaleIm().IsInit()) BinaryDumpInFile(aFp,anObj.ScaleIm().Val());
    BinaryDumpInFile(aFp,anObj.KeyAssocOut());
    BinaryDumpInFile(aFp,anObj.ZSol().IsInit());
    if (anObj.ZSol().IsInit()) BinaryDumpInFile(aFp,anObj.ZSol().Val());
    BinaryDumpInFile(aFp,anObj.Resol());
    BinaryDumpInFile(aFp,anObj.ResolIsRel());
    BinaryDumpInFile(aFp,anObj.DoTFW().IsInit());
    if (anObj.DoTFW().IsInit()) BinaryDumpInFile(aFp,anObj.DoTFW().Val());
    BinaryDumpInFile(aFp,anObj.TetaLimite());
    BinaryDumpInFile(aFp,anObj.DirTetaLim().IsInit());
    if (anObj.DirTetaLim().IsInit()) BinaryDumpInFile(aFp,anObj.DirTetaLim().Val());
    BinaryDumpInFile(aFp,anObj.DoOnlyIfNew().IsInit());
    if (anObj.DoOnlyIfNew().IsInit()) BinaryDumpInFile(aFp,anObj.DoOnlyIfNew().Val());
}

cElXMLTree * ToXMLTree(const cExportRedressement & anObj)
{
  XMLPushContext(anObj.mGXml);
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
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cExportRedressement & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

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

std::string  Mangling( cExportRedressement *) {return "C15E9CA328BD4E9CFE3F";};


std::string & cExportNuageByImage::KeyCalc()
{
   return mKeyCalc;
}

const std::string & cExportNuageByImage::KeyCalc()const 
{
   return mKeyCalc;
}


cTplValGesInit< bool > & cExportNuageByImage::SymPts()
{
   return mSymPts;
}

const cTplValGesInit< bool > & cExportNuageByImage::SymPts()const 
{
   return mSymPts;
}

void  BinaryUnDumpFromFile(cExportNuageByImage & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.KeyCalc(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SymPts().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SymPts().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SymPts().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cExportNuageByImage & anObj)
{
    BinaryDumpInFile(aFp,anObj.KeyCalc());
    BinaryDumpInFile(aFp,anObj.SymPts().IsInit());
    if (anObj.SymPts().IsInit()) BinaryDumpInFile(aFp,anObj.SymPts().Val());
}

cElXMLTree * ToXMLTree(const cExportNuageByImage & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ExportNuageByImage",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("KeyCalc"),anObj.KeyCalc())->ReTagThis("KeyCalc"));
   if (anObj.SymPts().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SymPts"),anObj.SymPts().Val())->ReTagThis("SymPts"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cExportNuageByImage & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.KeyCalc(),aTree->Get("KeyCalc",1)); //tototo 

   xml_init(anObj.SymPts(),aTree->Get("SymPts",1),bool(true)); //tototo 
}

std::string  Mangling( cExportNuageByImage *) {return "ED1163206714FBDFFE3F";};


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


cTplValGesInit< std::string > & cNuagePutCam::KeyCalName()
{
   return mKeyCalName;
}

const cTplValGesInit< std::string > & cNuagePutCam::KeyCalName()const 
{
   return mKeyCalName;
}


cTplValGesInit< double > & cNuagePutCam::StepImage()
{
   return mStepImage;
}

const cTplValGesInit< double > & cNuagePutCam::StepImage()const 
{
   return mStepImage;
}


cTplValGesInit< std::string > & cNuagePutCam::HomolRay()
{
   return mHomolRay;
}

const cTplValGesInit< std::string > & cNuagePutCam::HomolRay()const 
{
   return mHomolRay;
}


cTplValGesInit< Pt3di > & cNuagePutCam::ColRayHomol()
{
   return mColRayHomol;
}

const cTplValGesInit< Pt3di > & cNuagePutCam::ColRayHomol()const 
{
   return mColRayHomol;
}

void  BinaryUnDumpFromFile(cNuagePutCam & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.ColCadre(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ColRay().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ColRay().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ColRay().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.Long(),aFp);
    BinaryUnDumpFromFile(anObj.StepSeg(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.KeyCalName().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.KeyCalName().ValForcedForUnUmp(),aFp);
        }
        else  anObj.KeyCalName().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.StepImage().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.StepImage().ValForcedForUnUmp(),aFp);
        }
        else  anObj.StepImage().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.HomolRay().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.HomolRay().ValForcedForUnUmp(),aFp);
        }
        else  anObj.HomolRay().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ColRayHomol().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ColRayHomol().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ColRayHomol().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cNuagePutCam & anObj)
{
    BinaryDumpInFile(aFp,anObj.ColCadre());
    BinaryDumpInFile(aFp,anObj.ColRay().IsInit());
    if (anObj.ColRay().IsInit()) BinaryDumpInFile(aFp,anObj.ColRay().Val());
    BinaryDumpInFile(aFp,anObj.Long());
    BinaryDumpInFile(aFp,anObj.StepSeg());
    BinaryDumpInFile(aFp,anObj.KeyCalName().IsInit());
    if (anObj.KeyCalName().IsInit()) BinaryDumpInFile(aFp,anObj.KeyCalName().Val());
    BinaryDumpInFile(aFp,anObj.StepImage().IsInit());
    if (anObj.StepImage().IsInit()) BinaryDumpInFile(aFp,anObj.StepImage().Val());
    BinaryDumpInFile(aFp,anObj.HomolRay().IsInit());
    if (anObj.HomolRay().IsInit()) BinaryDumpInFile(aFp,anObj.HomolRay().Val());
    BinaryDumpInFile(aFp,anObj.ColRayHomol().IsInit());
    if (anObj.ColRayHomol().IsInit()) BinaryDumpInFile(aFp,anObj.ColRayHomol().Val());
}

cElXMLTree * ToXMLTree(const cNuagePutCam & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"NuagePutCam",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("ColCadre"),anObj.ColCadre())->ReTagThis("ColCadre"));
   if (anObj.ColRay().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ColRay"),anObj.ColRay().Val())->ReTagThis("ColRay"));
   aRes->AddFils(::ToXMLTree(std::string("Long"),anObj.Long())->ReTagThis("Long"));
   aRes->AddFils(::ToXMLTree(std::string("StepSeg"),anObj.StepSeg())->ReTagThis("StepSeg"));
   if (anObj.KeyCalName().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("KeyCalName"),anObj.KeyCalName().Val())->ReTagThis("KeyCalName"));
   if (anObj.StepImage().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("StepImage"),anObj.StepImage().Val())->ReTagThis("StepImage"));
   if (anObj.HomolRay().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("HomolRay"),anObj.HomolRay().Val())->ReTagThis("HomolRay"));
   if (anObj.ColRayHomol().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ColRayHomol"),anObj.ColRayHomol().Val())->ReTagThis("ColRayHomol"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cNuagePutCam & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.ColCadre(),aTree->Get("ColCadre",1)); //tototo 

   xml_init(anObj.ColRay(),aTree->Get("ColRay",1)); //tototo 

   xml_init(anObj.Long(),aTree->Get("Long",1)); //tototo 

   xml_init(anObj.StepSeg(),aTree->Get("StepSeg",1)); //tototo 

   xml_init(anObj.KeyCalName(),aTree->Get("KeyCalName",1)); //tototo 

   xml_init(anObj.StepImage(),aTree->Get("StepImage",1),double(-1.0)); //tototo 

   xml_init(anObj.HomolRay(),aTree->Get("HomolRay",1)); //tototo 

   xml_init(anObj.ColRayHomol(),aTree->Get("ColRayHomol",1)); //tototo 
}

std::string  Mangling( cNuagePutCam *) {return "E6197C8F606628A3FCBF";};


std::string & cNuagePutInterPMul::NamePMul()
{
   return mNamePMul;
}

const std::string & cNuagePutInterPMul::NamePMul()const 
{
   return mNamePMul;
}


double & cNuagePutInterPMul::StepDr()
{
   return mStepDr;
}

const double & cNuagePutInterPMul::StepDr()const 
{
   return mStepDr;
}


cTplValGesInit< double > & cNuagePutInterPMul::RabDr()
{
   return mRabDr;
}

const cTplValGesInit< double > & cNuagePutInterPMul::RabDr()const 
{
   return mRabDr;
}


Pt3di & cNuagePutInterPMul::ColRayInter()
{
   return mColRayInter;
}

const Pt3di & cNuagePutInterPMul::ColRayInter()const 
{
   return mColRayInter;
}


cTplValGesInit< double > & cNuagePutInterPMul::Epais()
{
   return mEpais;
}

const cTplValGesInit< double > & cNuagePutInterPMul::Epais()const 
{
   return mEpais;
}

void  BinaryUnDumpFromFile(cNuagePutInterPMul & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.NamePMul(),aFp);
    BinaryUnDumpFromFile(anObj.StepDr(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.RabDr().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.RabDr().ValForcedForUnUmp(),aFp);
        }
        else  anObj.RabDr().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.ColRayInter(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Epais().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Epais().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Epais().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cNuagePutInterPMul & anObj)
{
    BinaryDumpInFile(aFp,anObj.NamePMul());
    BinaryDumpInFile(aFp,anObj.StepDr());
    BinaryDumpInFile(aFp,anObj.RabDr().IsInit());
    if (anObj.RabDr().IsInit()) BinaryDumpInFile(aFp,anObj.RabDr().Val());
    BinaryDumpInFile(aFp,anObj.ColRayInter());
    BinaryDumpInFile(aFp,anObj.Epais().IsInit());
    if (anObj.Epais().IsInit()) BinaryDumpInFile(aFp,anObj.Epais().Val());
}

cElXMLTree * ToXMLTree(const cNuagePutInterPMul & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"NuagePutInterPMul",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("NamePMul"),anObj.NamePMul())->ReTagThis("NamePMul"));
   aRes->AddFils(::ToXMLTree(std::string("StepDr"),anObj.StepDr())->ReTagThis("StepDr"));
   if (anObj.RabDr().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("RabDr"),anObj.RabDr().Val())->ReTagThis("RabDr"));
   aRes->AddFils(::ToXMLTree(std::string("ColRayInter"),anObj.ColRayInter())->ReTagThis("ColRayInter"));
   if (anObj.Epais().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Epais"),anObj.Epais().Val())->ReTagThis("Epais"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cNuagePutInterPMul & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.NamePMul(),aTree->Get("NamePMul",1)); //tototo 

   xml_init(anObj.StepDr(),aTree->Get("StepDr",1)); //tototo 

   xml_init(anObj.RabDr(),aTree->Get("RabDr",1),double(0.0)); //tototo 

   xml_init(anObj.ColRayInter(),aTree->Get("ColRayInter",1)); //tototo 

   xml_init(anObj.Epais(),aTree->Get("Epais",1)); //tototo 
}

std::string  Mangling( cNuagePutInterPMul *) {return "5845359DC19F0CEBFBBF";};


std::string & cNuagePutGCPCtrl::NameGCPIm()
{
   return mNameGCPIm;
}

const std::string & cNuagePutGCPCtrl::NameGCPIm()const 
{
   return mNameGCPIm;
}


std::string & cNuagePutGCPCtrl::NameGCPTerr()
{
   return mNameGCPTerr;
}

const std::string & cNuagePutGCPCtrl::NameGCPTerr()const 
{
   return mNameGCPTerr;
}


double & cNuagePutGCPCtrl::ScaleVec()
{
   return mScaleVec;
}

const double & cNuagePutGCPCtrl::ScaleVec()const 
{
   return mScaleVec;
}

void  BinaryUnDumpFromFile(cNuagePutGCPCtrl & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.NameGCPIm(),aFp);
    BinaryUnDumpFromFile(anObj.NameGCPTerr(),aFp);
    BinaryUnDumpFromFile(anObj.ScaleVec(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cNuagePutGCPCtrl & anObj)
{
    BinaryDumpInFile(aFp,anObj.NameGCPIm());
    BinaryDumpInFile(aFp,anObj.NameGCPTerr());
    BinaryDumpInFile(aFp,anObj.ScaleVec());
}

cElXMLTree * ToXMLTree(const cNuagePutGCPCtrl & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"NuagePutGCPCtrl",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("NameGCPIm"),anObj.NameGCPIm())->ReTagThis("NameGCPIm"));
   aRes->AddFils(::ToXMLTree(std::string("NameGCPTerr"),anObj.NameGCPTerr())->ReTagThis("NameGCPTerr"));
   aRes->AddFils(::ToXMLTree(std::string("ScaleVec"),anObj.ScaleVec())->ReTagThis("ScaleVec"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cNuagePutGCPCtrl & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.NameGCPIm(),aTree->Get("NameGCPIm",1)); //tototo 

   xml_init(anObj.NameGCPTerr(),aTree->Get("NameGCPTerr",1)); //tototo 

   xml_init(anObj.ScaleVec(),aTree->Get("ScaleVec",1)); //tototo 
}

std::string  Mangling( cNuagePutGCPCtrl *) {return "559A3267A8359E9EFF3F";};


std::string & cExportNuage::NameOut()
{
   return mNameOut;
}

const std::string & cExportNuage::NameOut()const 
{
   return mNameOut;
}


std::string & cExportNuage::KeyCalc()
{
   return ExportNuageByImage().Val().KeyCalc();
}

const std::string & cExportNuage::KeyCalc()const 
{
   return ExportNuageByImage().Val().KeyCalc();
}


cTplValGesInit< bool > & cExportNuage::SymPts()
{
   return ExportNuageByImage().Val().SymPts();
}

const cTplValGesInit< bool > & cExportNuage::SymPts()const 
{
   return ExportNuageByImage().Val().SymPts();
}


cTplValGesInit< cExportNuageByImage > & cExportNuage::ExportNuageByImage()
{
   return mExportNuageByImage;
}

const cTplValGesInit< cExportNuageByImage > & cExportNuage::ExportNuageByImage()const 
{
   return mExportNuageByImage;
}


cTplValGesInit< bool > & cExportNuage::PlyModeBin()
{
   return mPlyModeBin;
}

const cTplValGesInit< bool > & cExportNuage::PlyModeBin()const 
{
   return mPlyModeBin;
}


cTplValGesInit< bool > & cExportNuage::SavePtsCol()
{
   return mSavePtsCol;
}

const cTplValGesInit< bool > & cExportNuage::SavePtsCol()const 
{
   return mSavePtsCol;
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


cTplValGesInit< std::string > & cExportNuage::KeyCalName()
{
   return NuagePutCam().Val().KeyCalName();
}

const cTplValGesInit< std::string > & cExportNuage::KeyCalName()const 
{
   return NuagePutCam().Val().KeyCalName();
}


cTplValGesInit< double > & cExportNuage::StepImage()
{
   return NuagePutCam().Val().StepImage();
}

const cTplValGesInit< double > & cExportNuage::StepImage()const 
{
   return NuagePutCam().Val().StepImage();
}


cTplValGesInit< std::string > & cExportNuage::HomolRay()
{
   return NuagePutCam().Val().HomolRay();
}

const cTplValGesInit< std::string > & cExportNuage::HomolRay()const 
{
   return NuagePutCam().Val().HomolRay();
}


cTplValGesInit< Pt3di > & cExportNuage::ColRayHomol()
{
   return NuagePutCam().Val().ColRayHomol();
}

const cTplValGesInit< Pt3di > & cExportNuage::ColRayHomol()const 
{
   return NuagePutCam().Val().ColRayHomol();
}


cTplValGesInit< cNuagePutCam > & cExportNuage::NuagePutCam()
{
   return mNuagePutCam;
}

const cTplValGesInit< cNuagePutCam > & cExportNuage::NuagePutCam()const 
{
   return mNuagePutCam;
}


std::string & cExportNuage::NamePMul()
{
   return NuagePutInterPMul().Val().NamePMul();
}

const std::string & cExportNuage::NamePMul()const 
{
   return NuagePutInterPMul().Val().NamePMul();
}


double & cExportNuage::StepDr()
{
   return NuagePutInterPMul().Val().StepDr();
}

const double & cExportNuage::StepDr()const 
{
   return NuagePutInterPMul().Val().StepDr();
}


cTplValGesInit< double > & cExportNuage::RabDr()
{
   return NuagePutInterPMul().Val().RabDr();
}

const cTplValGesInit< double > & cExportNuage::RabDr()const 
{
   return NuagePutInterPMul().Val().RabDr();
}


Pt3di & cExportNuage::ColRayInter()
{
   return NuagePutInterPMul().Val().ColRayInter();
}

const Pt3di & cExportNuage::ColRayInter()const 
{
   return NuagePutInterPMul().Val().ColRayInter();
}


cTplValGesInit< double > & cExportNuage::Epais()
{
   return NuagePutInterPMul().Val().Epais();
}

const cTplValGesInit< double > & cExportNuage::Epais()const 
{
   return NuagePutInterPMul().Val().Epais();
}


cTplValGesInit< cNuagePutInterPMul > & cExportNuage::NuagePutInterPMul()
{
   return mNuagePutInterPMul;
}

const cTplValGesInit< cNuagePutInterPMul > & cExportNuage::NuagePutInterPMul()const 
{
   return mNuagePutInterPMul;
}


std::string & cExportNuage::NameGCPIm()
{
   return NuagePutGCPCtrl().Val().NameGCPIm();
}

const std::string & cExportNuage::NameGCPIm()const 
{
   return NuagePutGCPCtrl().Val().NameGCPIm();
}


std::string & cExportNuage::NameGCPTerr()
{
   return NuagePutGCPCtrl().Val().NameGCPTerr();
}

const std::string & cExportNuage::NameGCPTerr()const 
{
   return NuagePutGCPCtrl().Val().NameGCPTerr();
}


double & cExportNuage::ScaleVec()
{
   return NuagePutGCPCtrl().Val().ScaleVec();
}

const double & cExportNuage::ScaleVec()const 
{
   return NuagePutGCPCtrl().Val().ScaleVec();
}


cTplValGesInit< cNuagePutGCPCtrl > & cExportNuage::NuagePutGCPCtrl()
{
   return mNuagePutGCPCtrl;
}

const cTplValGesInit< cNuagePutGCPCtrl > & cExportNuage::NuagePutGCPCtrl()const 
{
   return mNuagePutGCPCtrl;
}


cTplValGesInit< int > & cExportNuage::NormByC()
{
   return mNormByC;
}

const cTplValGesInit< int > & cExportNuage::NormByC()const 
{
   return mNormByC;
}

void  BinaryUnDumpFromFile(cExportNuage & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.NameOut(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ExportNuageByImage().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ExportNuageByImage().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ExportNuageByImage().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.PlyModeBin().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.PlyModeBin().ValForcedForUnUmp(),aFp);
        }
        else  anObj.PlyModeBin().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SavePtsCol().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SavePtsCol().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SavePtsCol().SetNoInit();
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             std::string aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.NameRefLiaison().push_back(aVal);
        }
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.PatternSel().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.PatternSel().ValForcedForUnUmp(),aFp);
        }
        else  anObj.PatternSel().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.Pond(),aFp);
    BinaryUnDumpFromFile(anObj.KeyFileColImage(),aFp);
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
             anObj.DirCol().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.DirCol().ValForcedForUnUmp(),aFp);
        }
        else  anObj.DirCol().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.PerCol().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.PerCol().ValForcedForUnUmp(),aFp);
        }
        else  anObj.PerCol().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.LimBSurH().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.LimBSurH().ValForcedForUnUmp(),aFp);
        }
        else  anObj.LimBSurH().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ImExpoRef().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ImExpoRef().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ImExpoRef().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NuagePutCam().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NuagePutCam().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NuagePutCam().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NuagePutInterPMul().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NuagePutInterPMul().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NuagePutInterPMul().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NuagePutGCPCtrl().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NuagePutGCPCtrl().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NuagePutGCPCtrl().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NormByC().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NormByC().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NormByC().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cExportNuage & anObj)
{
    BinaryDumpInFile(aFp,anObj.NameOut());
    BinaryDumpInFile(aFp,anObj.ExportNuageByImage().IsInit());
    if (anObj.ExportNuageByImage().IsInit()) BinaryDumpInFile(aFp,anObj.ExportNuageByImage().Val());
    BinaryDumpInFile(aFp,anObj.PlyModeBin().IsInit());
    if (anObj.PlyModeBin().IsInit()) BinaryDumpInFile(aFp,anObj.PlyModeBin().Val());
    BinaryDumpInFile(aFp,anObj.SavePtsCol().IsInit());
    if (anObj.SavePtsCol().IsInit()) BinaryDumpInFile(aFp,anObj.SavePtsCol().Val());
    BinaryDumpInFile(aFp,(int)anObj.NameRefLiaison().size());
    for(  std::list< std::string >::const_iterator iT=anObj.NameRefLiaison().begin();
         iT!=anObj.NameRefLiaison().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.PatternSel().IsInit());
    if (anObj.PatternSel().IsInit()) BinaryDumpInFile(aFp,anObj.PatternSel().Val());
    BinaryDumpInFile(aFp,anObj.Pond());
    BinaryDumpInFile(aFp,anObj.KeyFileColImage());
    BinaryDumpInFile(aFp,anObj.NbChan().IsInit());
    if (anObj.NbChan().IsInit()) BinaryDumpInFile(aFp,anObj.NbChan().Val());
    BinaryDumpInFile(aFp,anObj.DirCol().IsInit());
    if (anObj.DirCol().IsInit()) BinaryDumpInFile(aFp,anObj.DirCol().Val());
    BinaryDumpInFile(aFp,anObj.PerCol().IsInit());
    if (anObj.PerCol().IsInit()) BinaryDumpInFile(aFp,anObj.PerCol().Val());
    BinaryDumpInFile(aFp,anObj.LimBSurH().IsInit());
    if (anObj.LimBSurH().IsInit()) BinaryDumpInFile(aFp,anObj.LimBSurH().Val());
    BinaryDumpInFile(aFp,anObj.ImExpoRef().IsInit());
    if (anObj.ImExpoRef().IsInit()) BinaryDumpInFile(aFp,anObj.ImExpoRef().Val());
    BinaryDumpInFile(aFp,anObj.NuagePutCam().IsInit());
    if (anObj.NuagePutCam().IsInit()) BinaryDumpInFile(aFp,anObj.NuagePutCam().Val());
    BinaryDumpInFile(aFp,anObj.NuagePutInterPMul().IsInit());
    if (anObj.NuagePutInterPMul().IsInit()) BinaryDumpInFile(aFp,anObj.NuagePutInterPMul().Val());
    BinaryDumpInFile(aFp,anObj.NuagePutGCPCtrl().IsInit());
    if (anObj.NuagePutGCPCtrl().IsInit()) BinaryDumpInFile(aFp,anObj.NuagePutGCPCtrl().Val());
    BinaryDumpInFile(aFp,anObj.NormByC().IsInit());
    if (anObj.NormByC().IsInit()) BinaryDumpInFile(aFp,anObj.NormByC().Val());
}

cElXMLTree * ToXMLTree(const cExportNuage & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ExportNuage",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("NameOut"),anObj.NameOut())->ReTagThis("NameOut"));
   if (anObj.ExportNuageByImage().IsInit())
      aRes->AddFils(ToXMLTree(anObj.ExportNuageByImage().Val())->ReTagThis("ExportNuageByImage"));
   if (anObj.PlyModeBin().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PlyModeBin"),anObj.PlyModeBin().Val())->ReTagThis("PlyModeBin"));
   if (anObj.SavePtsCol().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SavePtsCol"),anObj.SavePtsCol().Val())->ReTagThis("SavePtsCol"));
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
   if (anObj.NuagePutInterPMul().IsInit())
      aRes->AddFils(ToXMLTree(anObj.NuagePutInterPMul().Val())->ReTagThis("NuagePutInterPMul"));
   if (anObj.NuagePutGCPCtrl().IsInit())
      aRes->AddFils(ToXMLTree(anObj.NuagePutGCPCtrl().Val())->ReTagThis("NuagePutGCPCtrl"));
   if (anObj.NormByC().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NormByC"),anObj.NormByC().Val())->ReTagThis("NormByC"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cExportNuage & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.NameOut(),aTree->Get("NameOut",1)); //tototo 

   xml_init(anObj.ExportNuageByImage(),aTree->Get("ExportNuageByImage",1)); //tototo 

   xml_init(anObj.PlyModeBin(),aTree->Get("PlyModeBin",1),bool(true)); //tototo 

   xml_init(anObj.SavePtsCol(),aTree->Get("SavePtsCol",1),bool(true)); //tototo 

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

   xml_init(anObj.NuagePutInterPMul(),aTree->Get("NuagePutInterPMul",1)); //tototo 

   xml_init(anObj.NuagePutGCPCtrl(),aTree->Get("NuagePutGCPCtrl",1)); //tototo 

   xml_init(anObj.NormByC(),aTree->Get("NormByC",1)); //tototo 
}

std::string  Mangling( cExportNuage *) {return "C87222E10740E8C2FD3F";};


cTplValGesInit< std::string > & cChoixImSec::KeyExistingFile()
{
   return mKeyExistingFile;
}

const cTplValGesInit< std::string > & cChoixImSec::KeyExistingFile()const 
{
   return mKeyExistingFile;
}


cTplValGesInit< std::string > & cChoixImSec::FileImSel()
{
   return mFileImSel;
}

const cTplValGesInit< std::string > & cChoixImSec::FileImSel()const 
{
   return mFileImSel;
}


cTplValGesInit< std::string > & cChoixImSec::KeyAssoc()
{
   return mKeyAssoc;
}

const cTplValGesInit< std::string > & cChoixImSec::KeyAssoc()const 
{
   return mKeyAssoc;
}


cTplValGesInit< std::string > & cChoixImSec::PatternSel()
{
   return mPatternSel;
}

const cTplValGesInit< std::string > & cChoixImSec::PatternSel()const 
{
   return mPatternSel;
}


cTplValGesInit< int > & cChoixImSec::CardMaxSub()
{
   return mCardMaxSub;
}

const cTplValGesInit< int > & cChoixImSec::CardMaxSub()const 
{
   return mCardMaxSub;
}


cTplValGesInit< double > & cChoixImSec::PenalNbIm()
{
   return mPenalNbIm;
}

const cTplValGesInit< double > & cChoixImSec::PenalNbIm()const 
{
   return mPenalNbIm;
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


cTplValGesInit< int > & cChoixImSec::NbMinPtsHom()
{
   return mNbMinPtsHom;
}

const cTplValGesInit< int > & cChoixImSec::NbMinPtsHom()const 
{
   return mNbMinPtsHom;
}


cTplValGesInit< double > & cChoixImSec::TetaMinPreSel()
{
   return mTetaMinPreSel;
}

const cTplValGesInit< double > & cChoixImSec::TetaMinPreSel()const 
{
   return mTetaMinPreSel;
}


cTplValGesInit< double > & cChoixImSec::TetaOpt()
{
   return mTetaOpt;
}

const cTplValGesInit< double > & cChoixImSec::TetaOpt()const 
{
   return mTetaOpt;
}


cTplValGesInit< double > & cChoixImSec::TetaMaxPreSel()
{
   return mTetaMaxPreSel;
}

const cTplValGesInit< double > & cChoixImSec::TetaMaxPreSel()const 
{
   return mTetaMaxPreSel;
}


cTplValGesInit< double > & cChoixImSec::RatioDistMin()
{
   return mRatioDistMin;
}

const cTplValGesInit< double > & cChoixImSec::RatioDistMin()const 
{
   return mRatioDistMin;
}


cTplValGesInit< double > & cChoixImSec::RatioStereoVertMax()
{
   return mRatioStereoVertMax;
}

const cTplValGesInit< double > & cChoixImSec::RatioStereoVertMax()const 
{
   return mRatioStereoVertMax;
}


cTplValGesInit< double > & cChoixImSec::Teta2Min()
{
   return mTeta2Min;
}

const cTplValGesInit< double > & cChoixImSec::Teta2Min()const 
{
   return mTeta2Min;
}


cTplValGesInit< double > & cChoixImSec::Teta2Max()
{
   return mTeta2Max;
}

const cTplValGesInit< double > & cChoixImSec::Teta2Max()const 
{
   return mTeta2Max;
}


cTplValGesInit< int > & cChoixImSec::NbMaxPresel()
{
   return mNbMaxPresel;
}

const cTplValGesInit< int > & cChoixImSec::NbMaxPresel()const 
{
   return mNbMaxPresel;
}


cTplValGesInit< int > & cChoixImSec::NbTestPrecis()
{
   return mNbTestPrecis;
}

const cTplValGesInit< int > & cChoixImSec::NbTestPrecis()const 
{
   return mNbTestPrecis;
}


cTplValGesInit< int > & cChoixImSec::NbCellOccAng()
{
   return mNbCellOccAng;
}

const cTplValGesInit< int > & cChoixImSec::NbCellOccAng()const 
{
   return mNbCellOccAng;
}


cTplValGesInit< int > & cChoixImSec::NbCaseIm()
{
   return mNbCaseIm;
}

const cTplValGesInit< int > & cChoixImSec::NbCaseIm()const 
{
   return mNbCaseIm;
}


cTplValGesInit< std::string > & cChoixImSec::Masq3D()
{
   return mMasq3D;
}

const cTplValGesInit< std::string > & cChoixImSec::Masq3D()const 
{
   return mMasq3D;
}

void  BinaryUnDumpFromFile(cChoixImSec & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.KeyExistingFile().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.KeyExistingFile().ValForcedForUnUmp(),aFp);
        }
        else  anObj.KeyExistingFile().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.FileImSel().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.FileImSel().ValForcedForUnUmp(),aFp);
        }
        else  anObj.FileImSel().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.KeyAssoc().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.KeyAssoc().ValForcedForUnUmp(),aFp);
        }
        else  anObj.KeyAssoc().SetNoInit();
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
             anObj.CardMaxSub().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.CardMaxSub().ValForcedForUnUmp(),aFp);
        }
        else  anObj.CardMaxSub().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.PenalNbIm().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.PenalNbIm().ValForcedForUnUmp(),aFp);
        }
        else  anObj.PenalNbIm().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.NbMin(),aFp);
    BinaryUnDumpFromFile(anObj.IdBdl(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NbMinPtsHom().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NbMinPtsHom().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NbMinPtsHom().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.TetaMinPreSel().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.TetaMinPreSel().ValForcedForUnUmp(),aFp);
        }
        else  anObj.TetaMinPreSel().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.TetaOpt().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.TetaOpt().ValForcedForUnUmp(),aFp);
        }
        else  anObj.TetaOpt().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.TetaMaxPreSel().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.TetaMaxPreSel().ValForcedForUnUmp(),aFp);
        }
        else  anObj.TetaMaxPreSel().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.RatioDistMin().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.RatioDistMin().ValForcedForUnUmp(),aFp);
        }
        else  anObj.RatioDistMin().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.RatioStereoVertMax().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.RatioStereoVertMax().ValForcedForUnUmp(),aFp);
        }
        else  anObj.RatioStereoVertMax().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Teta2Min().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Teta2Min().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Teta2Min().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Teta2Max().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Teta2Max().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Teta2Max().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NbMaxPresel().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NbMaxPresel().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NbMaxPresel().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NbTestPrecis().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NbTestPrecis().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NbTestPrecis().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NbCellOccAng().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NbCellOccAng().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NbCellOccAng().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NbCaseIm().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NbCaseIm().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NbCaseIm().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Masq3D().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Masq3D().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Masq3D().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cChoixImSec & anObj)
{
    BinaryDumpInFile(aFp,anObj.KeyExistingFile().IsInit());
    if (anObj.KeyExistingFile().IsInit()) BinaryDumpInFile(aFp,anObj.KeyExistingFile().Val());
    BinaryDumpInFile(aFp,anObj.FileImSel().IsInit());
    if (anObj.FileImSel().IsInit()) BinaryDumpInFile(aFp,anObj.FileImSel().Val());
    BinaryDumpInFile(aFp,anObj.KeyAssoc().IsInit());
    if (anObj.KeyAssoc().IsInit()) BinaryDumpInFile(aFp,anObj.KeyAssoc().Val());
    BinaryDumpInFile(aFp,anObj.PatternSel().IsInit());
    if (anObj.PatternSel().IsInit()) BinaryDumpInFile(aFp,anObj.PatternSel().Val());
    BinaryDumpInFile(aFp,anObj.CardMaxSub().IsInit());
    if (anObj.CardMaxSub().IsInit()) BinaryDumpInFile(aFp,anObj.CardMaxSub().Val());
    BinaryDumpInFile(aFp,anObj.PenalNbIm().IsInit());
    if (anObj.PenalNbIm().IsInit()) BinaryDumpInFile(aFp,anObj.PenalNbIm().Val());
    BinaryDumpInFile(aFp,anObj.NbMin());
    BinaryDumpInFile(aFp,anObj.IdBdl());
    BinaryDumpInFile(aFp,anObj.NbMinPtsHom().IsInit());
    if (anObj.NbMinPtsHom().IsInit()) BinaryDumpInFile(aFp,anObj.NbMinPtsHom().Val());
    BinaryDumpInFile(aFp,anObj.TetaMinPreSel().IsInit());
    if (anObj.TetaMinPreSel().IsInit()) BinaryDumpInFile(aFp,anObj.TetaMinPreSel().Val());
    BinaryDumpInFile(aFp,anObj.TetaOpt().IsInit());
    if (anObj.TetaOpt().IsInit()) BinaryDumpInFile(aFp,anObj.TetaOpt().Val());
    BinaryDumpInFile(aFp,anObj.TetaMaxPreSel().IsInit());
    if (anObj.TetaMaxPreSel().IsInit()) BinaryDumpInFile(aFp,anObj.TetaMaxPreSel().Val());
    BinaryDumpInFile(aFp,anObj.RatioDistMin().IsInit());
    if (anObj.RatioDistMin().IsInit()) BinaryDumpInFile(aFp,anObj.RatioDistMin().Val());
    BinaryDumpInFile(aFp,anObj.RatioStereoVertMax().IsInit());
    if (anObj.RatioStereoVertMax().IsInit()) BinaryDumpInFile(aFp,anObj.RatioStereoVertMax().Val());
    BinaryDumpInFile(aFp,anObj.Teta2Min().IsInit());
    if (anObj.Teta2Min().IsInit()) BinaryDumpInFile(aFp,anObj.Teta2Min().Val());
    BinaryDumpInFile(aFp,anObj.Teta2Max().IsInit());
    if (anObj.Teta2Max().IsInit()) BinaryDumpInFile(aFp,anObj.Teta2Max().Val());
    BinaryDumpInFile(aFp,anObj.NbMaxPresel().IsInit());
    if (anObj.NbMaxPresel().IsInit()) BinaryDumpInFile(aFp,anObj.NbMaxPresel().Val());
    BinaryDumpInFile(aFp,anObj.NbTestPrecis().IsInit());
    if (anObj.NbTestPrecis().IsInit()) BinaryDumpInFile(aFp,anObj.NbTestPrecis().Val());
    BinaryDumpInFile(aFp,anObj.NbCellOccAng().IsInit());
    if (anObj.NbCellOccAng().IsInit()) BinaryDumpInFile(aFp,anObj.NbCellOccAng().Val());
    BinaryDumpInFile(aFp,anObj.NbCaseIm().IsInit());
    if (anObj.NbCaseIm().IsInit()) BinaryDumpInFile(aFp,anObj.NbCaseIm().Val());
    BinaryDumpInFile(aFp,anObj.Masq3D().IsInit());
    if (anObj.Masq3D().IsInit()) BinaryDumpInFile(aFp,anObj.Masq3D().Val());
}

cElXMLTree * ToXMLTree(const cChoixImSec & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ChoixImSec",eXMLBranche);
   if (anObj.KeyExistingFile().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("KeyExistingFile"),anObj.KeyExistingFile().Val())->ReTagThis("KeyExistingFile"));
   if (anObj.FileImSel().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("FileImSel"),anObj.FileImSel().Val())->ReTagThis("FileImSel"));
   if (anObj.KeyAssoc().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("KeyAssoc"),anObj.KeyAssoc().Val())->ReTagThis("KeyAssoc"));
   if (anObj.PatternSel().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PatternSel"),anObj.PatternSel().Val())->ReTagThis("PatternSel"));
   if (anObj.CardMaxSub().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("CardMaxSub"),anObj.CardMaxSub().Val())->ReTagThis("CardMaxSub"));
   if (anObj.PenalNbIm().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PenalNbIm"),anObj.PenalNbIm().Val())->ReTagThis("PenalNbIm"));
   aRes->AddFils(::ToXMLTree(std::string("NbMin"),anObj.NbMin())->ReTagThis("NbMin"));
   aRes->AddFils(::ToXMLTree(std::string("IdBdl"),anObj.IdBdl())->ReTagThis("IdBdl"));
   if (anObj.NbMinPtsHom().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NbMinPtsHom"),anObj.NbMinPtsHom().Val())->ReTagThis("NbMinPtsHom"));
   if (anObj.TetaMinPreSel().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("TetaMinPreSel"),anObj.TetaMinPreSel().Val())->ReTagThis("TetaMinPreSel"));
   if (anObj.TetaOpt().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("TetaOpt"),anObj.TetaOpt().Val())->ReTagThis("TetaOpt"));
   if (anObj.TetaMaxPreSel().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("TetaMaxPreSel"),anObj.TetaMaxPreSel().Val())->ReTagThis("TetaMaxPreSel"));
   if (anObj.RatioDistMin().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("RatioDistMin"),anObj.RatioDistMin().Val())->ReTagThis("RatioDistMin"));
   if (anObj.RatioStereoVertMax().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("RatioStereoVertMax"),anObj.RatioStereoVertMax().Val())->ReTagThis("RatioStereoVertMax"));
   if (anObj.Teta2Min().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Teta2Min"),anObj.Teta2Min().Val())->ReTagThis("Teta2Min"));
   if (anObj.Teta2Max().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Teta2Max"),anObj.Teta2Max().Val())->ReTagThis("Teta2Max"));
   if (anObj.NbMaxPresel().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NbMaxPresel"),anObj.NbMaxPresel().Val())->ReTagThis("NbMaxPresel"));
   if (anObj.NbTestPrecis().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NbTestPrecis"),anObj.NbTestPrecis().Val())->ReTagThis("NbTestPrecis"));
   if (anObj.NbCellOccAng().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NbCellOccAng"),anObj.NbCellOccAng().Val())->ReTagThis("NbCellOccAng"));
   if (anObj.NbCaseIm().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NbCaseIm"),anObj.NbCaseIm().Val())->ReTagThis("NbCaseIm"));
   if (anObj.Masq3D().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Masq3D"),anObj.Masq3D().Val())->ReTagThis("Masq3D"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cChoixImSec & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.KeyExistingFile(),aTree->Get("KeyExistingFile",1)); //tototo 

   xml_init(anObj.FileImSel(),aTree->Get("FileImSel",1)); //tototo 

   xml_init(anObj.KeyAssoc(),aTree->Get("KeyAssoc",1)); //tototo 

   xml_init(anObj.PatternSel(),aTree->Get("PatternSel",1),std::string(".*")); //tototo 

   xml_init(anObj.CardMaxSub(),aTree->Get("CardMaxSub",1),int(6)); //tototo 

   xml_init(anObj.PenalNbIm(),aTree->Get("PenalNbIm",1),double(0.3)); //tototo 

   xml_init(anObj.NbMin(),aTree->Get("NbMin",1)); //tototo 

   xml_init(anObj.IdBdl(),aTree->Get("IdBdl",1)); //tototo 

   xml_init(anObj.NbMinPtsHom(),aTree->Get("NbMinPtsHom",1),int(15)); //tototo 

   xml_init(anObj.TetaMinPreSel(),aTree->Get("TetaMinPreSel",1),double(0.025)); //tototo 

   xml_init(anObj.TetaOpt(),aTree->Get("TetaOpt",1),double(0.25)); //tototo 

   xml_init(anObj.TetaMaxPreSel(),aTree->Get("TetaMaxPreSel",1),double(0.80)); //tototo 

   xml_init(anObj.RatioDistMin(),aTree->Get("RatioDistMin",1),double(0.50)); //tototo 

   xml_init(anObj.RatioStereoVertMax(),aTree->Get("RatioStereoVertMax",1),double(100.0)); //tototo 

   xml_init(anObj.Teta2Min(),aTree->Get("Teta2Min",1),double(0.65)); //tototo 

   xml_init(anObj.Teta2Max(),aTree->Get("Teta2Max",1),double(1.1)); //tototo 

   xml_init(anObj.NbMaxPresel(),aTree->Get("NbMaxPresel",1),int(10)); //tototo 

   xml_init(anObj.NbTestPrecis(),aTree->Get("NbTestPrecis",1),int(10)); //tototo 

   xml_init(anObj.NbCellOccAng(),aTree->Get("NbCellOccAng",1),int(50)); //tototo 

   xml_init(anObj.NbCaseIm(),aTree->Get("NbCaseIm",1),int(10)); //tototo 

   xml_init(anObj.Masq3D(),aTree->Get("Masq3D",1)); //tototo 
}

std::string  Mangling( cChoixImSec *) {return "143A1EAA4ACDC6A3FF3F";};


cTplValGesInit< std::string > & cChoixImMM::KeyExistingFile()
{
   return ChoixImSec().KeyExistingFile();
}

const cTplValGesInit< std::string > & cChoixImMM::KeyExistingFile()const 
{
   return ChoixImSec().KeyExistingFile();
}


cTplValGesInit< std::string > & cChoixImMM::FileImSel()
{
   return ChoixImSec().FileImSel();
}

const cTplValGesInit< std::string > & cChoixImMM::FileImSel()const 
{
   return ChoixImSec().FileImSel();
}


cTplValGesInit< std::string > & cChoixImMM::KeyAssoc()
{
   return ChoixImSec().KeyAssoc();
}

const cTplValGesInit< std::string > & cChoixImMM::KeyAssoc()const 
{
   return ChoixImSec().KeyAssoc();
}


cTplValGesInit< std::string > & cChoixImMM::PatternSel()
{
   return ChoixImSec().PatternSel();
}

const cTplValGesInit< std::string > & cChoixImMM::PatternSel()const 
{
   return ChoixImSec().PatternSel();
}


cTplValGesInit< int > & cChoixImMM::CardMaxSub()
{
   return ChoixImSec().CardMaxSub();
}

const cTplValGesInit< int > & cChoixImMM::CardMaxSub()const 
{
   return ChoixImSec().CardMaxSub();
}


cTplValGesInit< double > & cChoixImMM::PenalNbIm()
{
   return ChoixImSec().PenalNbIm();
}

const cTplValGesInit< double > & cChoixImMM::PenalNbIm()const 
{
   return ChoixImSec().PenalNbIm();
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


cTplValGesInit< int > & cChoixImMM::NbMinPtsHom()
{
   return ChoixImSec().NbMinPtsHom();
}

const cTplValGesInit< int > & cChoixImMM::NbMinPtsHom()const 
{
   return ChoixImSec().NbMinPtsHom();
}


cTplValGesInit< double > & cChoixImMM::TetaMinPreSel()
{
   return ChoixImSec().TetaMinPreSel();
}

const cTplValGesInit< double > & cChoixImMM::TetaMinPreSel()const 
{
   return ChoixImSec().TetaMinPreSel();
}


cTplValGesInit< double > & cChoixImMM::TetaOpt()
{
   return ChoixImSec().TetaOpt();
}

const cTplValGesInit< double > & cChoixImMM::TetaOpt()const 
{
   return ChoixImSec().TetaOpt();
}


cTplValGesInit< double > & cChoixImMM::TetaMaxPreSel()
{
   return ChoixImSec().TetaMaxPreSel();
}

const cTplValGesInit< double > & cChoixImMM::TetaMaxPreSel()const 
{
   return ChoixImSec().TetaMaxPreSel();
}


cTplValGesInit< double > & cChoixImMM::RatioDistMin()
{
   return ChoixImSec().RatioDistMin();
}

const cTplValGesInit< double > & cChoixImMM::RatioDistMin()const 
{
   return ChoixImSec().RatioDistMin();
}


cTplValGesInit< double > & cChoixImMM::RatioStereoVertMax()
{
   return ChoixImSec().RatioStereoVertMax();
}

const cTplValGesInit< double > & cChoixImMM::RatioStereoVertMax()const 
{
   return ChoixImSec().RatioStereoVertMax();
}


cTplValGesInit< double > & cChoixImMM::Teta2Min()
{
   return ChoixImSec().Teta2Min();
}

const cTplValGesInit< double > & cChoixImMM::Teta2Min()const 
{
   return ChoixImSec().Teta2Min();
}


cTplValGesInit< double > & cChoixImMM::Teta2Max()
{
   return ChoixImSec().Teta2Max();
}

const cTplValGesInit< double > & cChoixImMM::Teta2Max()const 
{
   return ChoixImSec().Teta2Max();
}


cTplValGesInit< int > & cChoixImMM::NbMaxPresel()
{
   return ChoixImSec().NbMaxPresel();
}

const cTplValGesInit< int > & cChoixImMM::NbMaxPresel()const 
{
   return ChoixImSec().NbMaxPresel();
}


cTplValGesInit< int > & cChoixImMM::NbTestPrecis()
{
   return ChoixImSec().NbTestPrecis();
}

const cTplValGesInit< int > & cChoixImMM::NbTestPrecis()const 
{
   return ChoixImSec().NbTestPrecis();
}


cTplValGesInit< int > & cChoixImMM::NbCellOccAng()
{
   return ChoixImSec().NbCellOccAng();
}

const cTplValGesInit< int > & cChoixImMM::NbCellOccAng()const 
{
   return ChoixImSec().NbCellOccAng();
}


cTplValGesInit< int > & cChoixImMM::NbCaseIm()
{
   return ChoixImSec().NbCaseIm();
}

const cTplValGesInit< int > & cChoixImMM::NbCaseIm()const 
{
   return ChoixImSec().NbCaseIm();
}


cTplValGesInit< std::string > & cChoixImMM::Masq3D()
{
   return ChoixImSec().Masq3D();
}

const cTplValGesInit< std::string > & cChoixImMM::Masq3D()const 
{
   return ChoixImSec().Masq3D();
}


cChoixImSec & cChoixImMM::ChoixImSec()
{
   return mChoixImSec;
}

const cChoixImSec & cChoixImMM::ChoixImSec()const 
{
   return mChoixImSec;
}

void  BinaryUnDumpFromFile(cChoixImMM & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.ChoixImSec(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cChoixImMM & anObj)
{
    BinaryDumpInFile(aFp,anObj.ChoixImSec());
}

cElXMLTree * ToXMLTree(const cChoixImMM & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ChoixImMM",eXMLBranche);
   aRes->AddFils(ToXMLTree(anObj.ChoixImSec())->ReTagThis("ChoixImSec"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cChoixImMM & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.ChoixImSec(),aTree->Get("ChoixImSec",1)); //tototo 
}

std::string  Mangling( cChoixImMM *) {return "51B8116A46BBCFC9FD3F";};


std::string & cExportSensibParamAero::Dir()
{
   return mDir;
}

const std::string & cExportSensibParamAero::Dir()const 
{
   return mDir;
}

void  BinaryUnDumpFromFile(cExportSensibParamAero & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Dir(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cExportSensibParamAero & anObj)
{
    BinaryDumpInFile(aFp,anObj.Dir());
}

cElXMLTree * ToXMLTree(const cExportSensibParamAero & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ExportSensibParamAero",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Dir"),anObj.Dir())->ReTagThis("Dir"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cExportSensibParamAero & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.Dir(),aTree->Get("Dir",1)); //tototo 
}

std::string  Mangling( cExportSensibParamAero *) {return "0E03D19D3C1397E9FD3F";};


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


std::list< cExportBlockCamera > & cSectionExport::ExportBlockCamera()
{
   return mExportBlockCamera;
}

const std::list< cExportBlockCamera > & cSectionExport::ExportBlockCamera()const 
{
   return mExportBlockCamera;
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


cTplValGesInit< std::string > & cSectionExport::KeyExistingFile()
{
   return ChoixImMM().Val().ChoixImSec().KeyExistingFile();
}

const cTplValGesInit< std::string > & cSectionExport::KeyExistingFile()const 
{
   return ChoixImMM().Val().ChoixImSec().KeyExistingFile();
}


cTplValGesInit< std::string > & cSectionExport::FileImSel()
{
   return ChoixImMM().Val().ChoixImSec().FileImSel();
}

const cTplValGesInit< std::string > & cSectionExport::FileImSel()const 
{
   return ChoixImMM().Val().ChoixImSec().FileImSel();
}


cTplValGesInit< std::string > & cSectionExport::KeyAssoc()
{
   return ChoixImMM().Val().ChoixImSec().KeyAssoc();
}

const cTplValGesInit< std::string > & cSectionExport::KeyAssoc()const 
{
   return ChoixImMM().Val().ChoixImSec().KeyAssoc();
}


cTplValGesInit< std::string > & cSectionExport::PatternSel()
{
   return ChoixImMM().Val().ChoixImSec().PatternSel();
}

const cTplValGesInit< std::string > & cSectionExport::PatternSel()const 
{
   return ChoixImMM().Val().ChoixImSec().PatternSel();
}


cTplValGesInit< int > & cSectionExport::CardMaxSub()
{
   return ChoixImMM().Val().ChoixImSec().CardMaxSub();
}

const cTplValGesInit< int > & cSectionExport::CardMaxSub()const 
{
   return ChoixImMM().Val().ChoixImSec().CardMaxSub();
}


cTplValGesInit< double > & cSectionExport::PenalNbIm()
{
   return ChoixImMM().Val().ChoixImSec().PenalNbIm();
}

const cTplValGesInit< double > & cSectionExport::PenalNbIm()const 
{
   return ChoixImMM().Val().ChoixImSec().PenalNbIm();
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


cTplValGesInit< int > & cSectionExport::NbMinPtsHom()
{
   return ChoixImMM().Val().ChoixImSec().NbMinPtsHom();
}

const cTplValGesInit< int > & cSectionExport::NbMinPtsHom()const 
{
   return ChoixImMM().Val().ChoixImSec().NbMinPtsHom();
}


cTplValGesInit< double > & cSectionExport::TetaMinPreSel()
{
   return ChoixImMM().Val().ChoixImSec().TetaMinPreSel();
}

const cTplValGesInit< double > & cSectionExport::TetaMinPreSel()const 
{
   return ChoixImMM().Val().ChoixImSec().TetaMinPreSel();
}


cTplValGesInit< double > & cSectionExport::TetaOpt()
{
   return ChoixImMM().Val().ChoixImSec().TetaOpt();
}

const cTplValGesInit< double > & cSectionExport::TetaOpt()const 
{
   return ChoixImMM().Val().ChoixImSec().TetaOpt();
}


cTplValGesInit< double > & cSectionExport::TetaMaxPreSel()
{
   return ChoixImMM().Val().ChoixImSec().TetaMaxPreSel();
}

const cTplValGesInit< double > & cSectionExport::TetaMaxPreSel()const 
{
   return ChoixImMM().Val().ChoixImSec().TetaMaxPreSel();
}


cTplValGesInit< double > & cSectionExport::RatioDistMin()
{
   return ChoixImMM().Val().ChoixImSec().RatioDistMin();
}

const cTplValGesInit< double > & cSectionExport::RatioDistMin()const 
{
   return ChoixImMM().Val().ChoixImSec().RatioDistMin();
}


cTplValGesInit< double > & cSectionExport::RatioStereoVertMax()
{
   return ChoixImMM().Val().ChoixImSec().RatioStereoVertMax();
}

const cTplValGesInit< double > & cSectionExport::RatioStereoVertMax()const 
{
   return ChoixImMM().Val().ChoixImSec().RatioStereoVertMax();
}


cTplValGesInit< double > & cSectionExport::Teta2Min()
{
   return ChoixImMM().Val().ChoixImSec().Teta2Min();
}

const cTplValGesInit< double > & cSectionExport::Teta2Min()const 
{
   return ChoixImMM().Val().ChoixImSec().Teta2Min();
}


cTplValGesInit< double > & cSectionExport::Teta2Max()
{
   return ChoixImMM().Val().ChoixImSec().Teta2Max();
}

const cTplValGesInit< double > & cSectionExport::Teta2Max()const 
{
   return ChoixImMM().Val().ChoixImSec().Teta2Max();
}


cTplValGesInit< int > & cSectionExport::NbMaxPresel()
{
   return ChoixImMM().Val().ChoixImSec().NbMaxPresel();
}

const cTplValGesInit< int > & cSectionExport::NbMaxPresel()const 
{
   return ChoixImMM().Val().ChoixImSec().NbMaxPresel();
}


cTplValGesInit< int > & cSectionExport::NbTestPrecis()
{
   return ChoixImMM().Val().ChoixImSec().NbTestPrecis();
}

const cTplValGesInit< int > & cSectionExport::NbTestPrecis()const 
{
   return ChoixImMM().Val().ChoixImSec().NbTestPrecis();
}


cTplValGesInit< int > & cSectionExport::NbCellOccAng()
{
   return ChoixImMM().Val().ChoixImSec().NbCellOccAng();
}

const cTplValGesInit< int > & cSectionExport::NbCellOccAng()const 
{
   return ChoixImMM().Val().ChoixImSec().NbCellOccAng();
}


cTplValGesInit< int > & cSectionExport::NbCaseIm()
{
   return ChoixImMM().Val().ChoixImSec().NbCaseIm();
}

const cTplValGesInit< int > & cSectionExport::NbCaseIm()const 
{
   return ChoixImMM().Val().ChoixImSec().NbCaseIm();
}


cTplValGesInit< std::string > & cSectionExport::Masq3D()
{
   return ChoixImMM().Val().ChoixImSec().Masq3D();
}

const cTplValGesInit< std::string > & cSectionExport::Masq3D()const 
{
   return ChoixImMM().Val().ChoixImSec().Masq3D();
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


cTplValGesInit< std::string > & cSectionExport::ExportResiduXml()
{
   return mExportResiduXml;
}

const cTplValGesInit< std::string > & cSectionExport::ExportResiduXml()const 
{
   return mExportResiduXml;
}


std::string & cSectionExport::Dir()
{
   return ExportSensibParamAero().Val().Dir();
}

const std::string & cSectionExport::Dir()const 
{
   return ExportSensibParamAero().Val().Dir();
}


cTplValGesInit< cExportSensibParamAero > & cSectionExport::ExportSensibParamAero()
{
   return mExportSensibParamAero;
}

const cTplValGesInit< cExportSensibParamAero > & cSectionExport::ExportSensibParamAero()const 
{
   return mExportSensibParamAero;
}

void  BinaryUnDumpFromFile(cSectionExport & anObj,ELISE_fp & aFp)
{
   { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cExportCalib aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.ExportCalib().push_back(aVal);
        }
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cExportPose aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.ExportPose().push_back(aVal);
        }
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cExportAttrPose aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.ExportAttrPose().push_back(aVal);
        }
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cExportRepereLoc aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.ExportRepereLoc().push_back(aVal);
        }
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cExportBlockCamera aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.ExportBlockCamera().push_back(aVal);
        }
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cExportMesuresFromCarteProf aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.ExportMesuresFromCarteProf().push_back(aVal);
        }
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cExportVisuConfigGrpPose aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.ExportVisuConfigGrpPose().push_back(aVal);
        }
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ExportPtsFlottant().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ExportPtsFlottant().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ExportPtsFlottant().SetNoInit();
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cExportImResiduLiaison aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.ExportImResiduLiaison().push_back(aVal);
        }
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cExportRedressement aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.ExportRedressement().push_back(aVal);
        }
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cExportNuage aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.ExportNuage().push_back(aVal);
        }
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ChoixImMM().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ChoixImMM().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ChoixImMM().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ExportResiduXml().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ExportResiduXml().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ExportResiduXml().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ExportSensibParamAero().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ExportSensibParamAero().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ExportSensibParamAero().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cSectionExport & anObj)
{
    BinaryDumpInFile(aFp,(int)anObj.ExportCalib().size());
    for(  std::list< cExportCalib >::const_iterator iT=anObj.ExportCalib().begin();
         iT!=anObj.ExportCalib().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,(int)anObj.ExportPose().size());
    for(  std::list< cExportPose >::const_iterator iT=anObj.ExportPose().begin();
         iT!=anObj.ExportPose().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,(int)anObj.ExportAttrPose().size());
    for(  std::list< cExportAttrPose >::const_iterator iT=anObj.ExportAttrPose().begin();
         iT!=anObj.ExportAttrPose().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,(int)anObj.ExportRepereLoc().size());
    for(  std::list< cExportRepereLoc >::const_iterator iT=anObj.ExportRepereLoc().begin();
         iT!=anObj.ExportRepereLoc().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,(int)anObj.ExportBlockCamera().size());
    for(  std::list< cExportBlockCamera >::const_iterator iT=anObj.ExportBlockCamera().begin();
         iT!=anObj.ExportBlockCamera().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,(int)anObj.ExportMesuresFromCarteProf().size());
    for(  std::list< cExportMesuresFromCarteProf >::const_iterator iT=anObj.ExportMesuresFromCarteProf().begin();
         iT!=anObj.ExportMesuresFromCarteProf().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,(int)anObj.ExportVisuConfigGrpPose().size());
    for(  std::list< cExportVisuConfigGrpPose >::const_iterator iT=anObj.ExportVisuConfigGrpPose().begin();
         iT!=anObj.ExportVisuConfigGrpPose().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.ExportPtsFlottant().IsInit());
    if (anObj.ExportPtsFlottant().IsInit()) BinaryDumpInFile(aFp,anObj.ExportPtsFlottant().Val());
    BinaryDumpInFile(aFp,(int)anObj.ExportImResiduLiaison().size());
    for(  std::list< cExportImResiduLiaison >::const_iterator iT=anObj.ExportImResiduLiaison().begin();
         iT!=anObj.ExportImResiduLiaison().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,(int)anObj.ExportRedressement().size());
    for(  std::list< cExportRedressement >::const_iterator iT=anObj.ExportRedressement().begin();
         iT!=anObj.ExportRedressement().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,(int)anObj.ExportNuage().size());
    for(  std::list< cExportNuage >::const_iterator iT=anObj.ExportNuage().begin();
         iT!=anObj.ExportNuage().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.ChoixImMM().IsInit());
    if (anObj.ChoixImMM().IsInit()) BinaryDumpInFile(aFp,anObj.ChoixImMM().Val());
    BinaryDumpInFile(aFp,anObj.ExportResiduXml().IsInit());
    if (anObj.ExportResiduXml().IsInit()) BinaryDumpInFile(aFp,anObj.ExportResiduXml().Val());
    BinaryDumpInFile(aFp,anObj.ExportSensibParamAero().IsInit());
    if (anObj.ExportSensibParamAero().IsInit()) BinaryDumpInFile(aFp,anObj.ExportSensibParamAero().Val());
}

cElXMLTree * ToXMLTree(const cSectionExport & anObj)
{
  XMLPushContext(anObj.mGXml);
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
  (       std::list< cExportBlockCamera >::const_iterator it=anObj.ExportBlockCamera().begin();
      it !=anObj.ExportBlockCamera().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("ExportBlockCamera"));
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
   if (anObj.ExportResiduXml().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ExportResiduXml"),anObj.ExportResiduXml().Val())->ReTagThis("ExportResiduXml"));
   if (anObj.ExportSensibParamAero().IsInit())
      aRes->AddFils(ToXMLTree(anObj.ExportSensibParamAero().Val())->ReTagThis("ExportSensibParamAero"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cSectionExport & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.ExportCalib(),aTree->GetAll("ExportCalib",false,1));

   xml_init(anObj.ExportPose(),aTree->GetAll("ExportPose",false,1));

   xml_init(anObj.ExportAttrPose(),aTree->GetAll("ExportAttrPose",false,1));

   xml_init(anObj.ExportRepereLoc(),aTree->GetAll("ExportRepereLoc",false,1));

   xml_init(anObj.ExportBlockCamera(),aTree->GetAll("ExportBlockCamera",false,1));

   xml_init(anObj.ExportMesuresFromCarteProf(),aTree->GetAll("ExportMesuresFromCarteProf",false,1));

   xml_init(anObj.ExportVisuConfigGrpPose(),aTree->GetAll("ExportVisuConfigGrpPose",false,1));

   xml_init(anObj.ExportPtsFlottant(),aTree->Get("ExportPtsFlottant",1)); //tototo 

   xml_init(anObj.ExportImResiduLiaison(),aTree->GetAll("ExportImResiduLiaison",false,1));

   xml_init(anObj.ExportRedressement(),aTree->GetAll("ExportRedressement",false,1));

   xml_init(anObj.ExportNuage(),aTree->GetAll("ExportNuage",false,1));

   xml_init(anObj.ChoixImMM(),aTree->Get("ChoixImMM",1)); //tototo 

   xml_init(anObj.ExportResiduXml(),aTree->Get("ExportResiduXml",1)); //tototo 

   xml_init(anObj.ExportSensibParamAero(),aTree->Get("ExportSensibParamAero",1)); //tototo 
}

std::string  Mangling( cSectionExport *) {return "1EA1E65FF8E0C2B8FF3F";};


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


double & cEtapeCompensation::SigmaPix()
{
   return SectionObservations().ContrCamConseq().Val().SigmaPix();
}

const double & cEtapeCompensation::SigmaPix()const 
{
   return SectionObservations().ContrCamConseq().Val().SigmaPix();
}


int & cEtapeCompensation::NbGrid()
{
   return SectionObservations().ContrCamConseq().Val().NbGrid();
}

const int & cEtapeCompensation::NbGrid()const 
{
   return SectionObservations().ContrCamConseq().Val().NbGrid();
}


cTplValGesInit< cContrCamConseq > & cEtapeCompensation::ContrCamConseq()
{
   return SectionObservations().ContrCamConseq();
}

const cTplValGesInit< cContrCamConseq > & cEtapeCompensation::ContrCamConseq()const 
{
   return SectionObservations().ContrCamConseq();
}


std::list< cContrCamGenInc > & cEtapeCompensation::ContrCamGenInc()
{
   return SectionObservations().ContrCamGenInc();
}

const std::list< cContrCamGenInc > & cEtapeCompensation::ContrCamGenInc()const 
{
   return SectionObservations().ContrCamGenInc();
}


std::list< cObsBlockCamRig > & cEtapeCompensation::ObsBlockCamRig()
{
   return SectionObservations().ObsBlockCamRig();
}

const std::list< cObsBlockCamRig > & cEtapeCompensation::ObsBlockCamRig()const 
{
   return SectionObservations().ObsBlockCamRig();
}


std::list< cObsCenterInPlane > & cEtapeCompensation::ObsCenterInPlane()
{
   return SectionObservations().ObsCenterInPlane();
}

const std::list< cObsCenterInPlane > & cEtapeCompensation::ObsCenterInPlane()const 
{
   return SectionObservations().ObsCenterInPlane();
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


std::list< cObsRelGPS > & cEtapeCompensation::ObsRelGPS()
{
   return SectionObservations().ObsRelGPS();
}

const std::list< cObsRelGPS > & cEtapeCompensation::ObsRelGPS()const 
{
   return SectionObservations().ObsRelGPS();
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


std::list< cExportBlockCamera > & cEtapeCompensation::ExportBlockCamera()
{
   return SectionExport().Val().ExportBlockCamera();
}

const std::list< cExportBlockCamera > & cEtapeCompensation::ExportBlockCamera()const 
{
   return SectionExport().Val().ExportBlockCamera();
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


cTplValGesInit< std::string > & cEtapeCompensation::KeyExistingFile()
{
   return SectionExport().Val().ChoixImMM().Val().ChoixImSec().KeyExistingFile();
}

const cTplValGesInit< std::string > & cEtapeCompensation::KeyExistingFile()const 
{
   return SectionExport().Val().ChoixImMM().Val().ChoixImSec().KeyExistingFile();
}


cTplValGesInit< std::string > & cEtapeCompensation::FileImSel()
{
   return SectionExport().Val().ChoixImMM().Val().ChoixImSec().FileImSel();
}

const cTplValGesInit< std::string > & cEtapeCompensation::FileImSel()const 
{
   return SectionExport().Val().ChoixImMM().Val().ChoixImSec().FileImSel();
}


cTplValGesInit< std::string > & cEtapeCompensation::KeyAssoc()
{
   return SectionExport().Val().ChoixImMM().Val().ChoixImSec().KeyAssoc();
}

const cTplValGesInit< std::string > & cEtapeCompensation::KeyAssoc()const 
{
   return SectionExport().Val().ChoixImMM().Val().ChoixImSec().KeyAssoc();
}


cTplValGesInit< std::string > & cEtapeCompensation::PatternSel()
{
   return SectionExport().Val().ChoixImMM().Val().ChoixImSec().PatternSel();
}

const cTplValGesInit< std::string > & cEtapeCompensation::PatternSel()const 
{
   return SectionExport().Val().ChoixImMM().Val().ChoixImSec().PatternSel();
}


cTplValGesInit< int > & cEtapeCompensation::CardMaxSub()
{
   return SectionExport().Val().ChoixImMM().Val().ChoixImSec().CardMaxSub();
}

const cTplValGesInit< int > & cEtapeCompensation::CardMaxSub()const 
{
   return SectionExport().Val().ChoixImMM().Val().ChoixImSec().CardMaxSub();
}


cTplValGesInit< double > & cEtapeCompensation::PenalNbIm()
{
   return SectionExport().Val().ChoixImMM().Val().ChoixImSec().PenalNbIm();
}

const cTplValGesInit< double > & cEtapeCompensation::PenalNbIm()const 
{
   return SectionExport().Val().ChoixImMM().Val().ChoixImSec().PenalNbIm();
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


cTplValGesInit< int > & cEtapeCompensation::NbMinPtsHom()
{
   return SectionExport().Val().ChoixImMM().Val().ChoixImSec().NbMinPtsHom();
}

const cTplValGesInit< int > & cEtapeCompensation::NbMinPtsHom()const 
{
   return SectionExport().Val().ChoixImMM().Val().ChoixImSec().NbMinPtsHom();
}


cTplValGesInit< double > & cEtapeCompensation::TetaMinPreSel()
{
   return SectionExport().Val().ChoixImMM().Val().ChoixImSec().TetaMinPreSel();
}

const cTplValGesInit< double > & cEtapeCompensation::TetaMinPreSel()const 
{
   return SectionExport().Val().ChoixImMM().Val().ChoixImSec().TetaMinPreSel();
}


cTplValGesInit< double > & cEtapeCompensation::TetaOpt()
{
   return SectionExport().Val().ChoixImMM().Val().ChoixImSec().TetaOpt();
}

const cTplValGesInit< double > & cEtapeCompensation::TetaOpt()const 
{
   return SectionExport().Val().ChoixImMM().Val().ChoixImSec().TetaOpt();
}


cTplValGesInit< double > & cEtapeCompensation::TetaMaxPreSel()
{
   return SectionExport().Val().ChoixImMM().Val().ChoixImSec().TetaMaxPreSel();
}

const cTplValGesInit< double > & cEtapeCompensation::TetaMaxPreSel()const 
{
   return SectionExport().Val().ChoixImMM().Val().ChoixImSec().TetaMaxPreSel();
}


cTplValGesInit< double > & cEtapeCompensation::RatioDistMin()
{
   return SectionExport().Val().ChoixImMM().Val().ChoixImSec().RatioDistMin();
}

const cTplValGesInit< double > & cEtapeCompensation::RatioDistMin()const 
{
   return SectionExport().Val().ChoixImMM().Val().ChoixImSec().RatioDistMin();
}


cTplValGesInit< double > & cEtapeCompensation::RatioStereoVertMax()
{
   return SectionExport().Val().ChoixImMM().Val().ChoixImSec().RatioStereoVertMax();
}

const cTplValGesInit< double > & cEtapeCompensation::RatioStereoVertMax()const 
{
   return SectionExport().Val().ChoixImMM().Val().ChoixImSec().RatioStereoVertMax();
}


cTplValGesInit< double > & cEtapeCompensation::Teta2Min()
{
   return SectionExport().Val().ChoixImMM().Val().ChoixImSec().Teta2Min();
}

const cTplValGesInit< double > & cEtapeCompensation::Teta2Min()const 
{
   return SectionExport().Val().ChoixImMM().Val().ChoixImSec().Teta2Min();
}


cTplValGesInit< double > & cEtapeCompensation::Teta2Max()
{
   return SectionExport().Val().ChoixImMM().Val().ChoixImSec().Teta2Max();
}

const cTplValGesInit< double > & cEtapeCompensation::Teta2Max()const 
{
   return SectionExport().Val().ChoixImMM().Val().ChoixImSec().Teta2Max();
}


cTplValGesInit< int > & cEtapeCompensation::NbMaxPresel()
{
   return SectionExport().Val().ChoixImMM().Val().ChoixImSec().NbMaxPresel();
}

const cTplValGesInit< int > & cEtapeCompensation::NbMaxPresel()const 
{
   return SectionExport().Val().ChoixImMM().Val().ChoixImSec().NbMaxPresel();
}


cTplValGesInit< int > & cEtapeCompensation::NbTestPrecis()
{
   return SectionExport().Val().ChoixImMM().Val().ChoixImSec().NbTestPrecis();
}

const cTplValGesInit< int > & cEtapeCompensation::NbTestPrecis()const 
{
   return SectionExport().Val().ChoixImMM().Val().ChoixImSec().NbTestPrecis();
}


cTplValGesInit< int > & cEtapeCompensation::NbCellOccAng()
{
   return SectionExport().Val().ChoixImMM().Val().ChoixImSec().NbCellOccAng();
}

const cTplValGesInit< int > & cEtapeCompensation::NbCellOccAng()const 
{
   return SectionExport().Val().ChoixImMM().Val().ChoixImSec().NbCellOccAng();
}


cTplValGesInit< int > & cEtapeCompensation::NbCaseIm()
{
   return SectionExport().Val().ChoixImMM().Val().ChoixImSec().NbCaseIm();
}

const cTplValGesInit< int > & cEtapeCompensation::NbCaseIm()const 
{
   return SectionExport().Val().ChoixImMM().Val().ChoixImSec().NbCaseIm();
}


cTplValGesInit< std::string > & cEtapeCompensation::Masq3D()
{
   return SectionExport().Val().ChoixImMM().Val().ChoixImSec().Masq3D();
}

const cTplValGesInit< std::string > & cEtapeCompensation::Masq3D()const 
{
   return SectionExport().Val().ChoixImMM().Val().ChoixImSec().Masq3D();
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


cTplValGesInit< std::string > & cEtapeCompensation::ExportResiduXml()
{
   return SectionExport().Val().ExportResiduXml();
}

const cTplValGesInit< std::string > & cEtapeCompensation::ExportResiduXml()const 
{
   return SectionExport().Val().ExportResiduXml();
}


std::string & cEtapeCompensation::Dir()
{
   return SectionExport().Val().ExportSensibParamAero().Val().Dir();
}

const std::string & cEtapeCompensation::Dir()const 
{
   return SectionExport().Val().ExportSensibParamAero().Val().Dir();
}


cTplValGesInit< cExportSensibParamAero > & cEtapeCompensation::ExportSensibParamAero()
{
   return SectionExport().Val().ExportSensibParamAero();
}

const cTplValGesInit< cExportSensibParamAero > & cEtapeCompensation::ExportSensibParamAero()const 
{
   return SectionExport().Val().ExportSensibParamAero();
}


cTplValGesInit< cSectionExport > & cEtapeCompensation::SectionExport()
{
   return mSectionExport;
}

const cTplValGesInit< cSectionExport > & cEtapeCompensation::SectionExport()const 
{
   return mSectionExport;
}

void  BinaryUnDumpFromFile(cEtapeCompensation & anObj,ELISE_fp & aFp)
{
   { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cIterationsCompensation aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.IterationsCompensation().push_back(aVal);
        }
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SectionTracage().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SectionTracage().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SectionTracage().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SLMEtape().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SLMEtape().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SLMEtape().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SLMGlob().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SLMGlob().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SLMGlob().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.MultSLMEtape().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.MultSLMEtape().ValForcedForUnUmp(),aFp);
        }
        else  anObj.MultSLMEtape().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.MultSLMGlob().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.MultSLMGlob().ValForcedForUnUmp(),aFp);
        }
        else  anObj.MultSLMGlob().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.SectionObservations(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SectionExport().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SectionExport().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SectionExport().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cEtapeCompensation & anObj)
{
    BinaryDumpInFile(aFp,(int)anObj.IterationsCompensation().size());
    for(  std::vector< cIterationsCompensation >::const_iterator iT=anObj.IterationsCompensation().begin();
         iT!=anObj.IterationsCompensation().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.SectionTracage().IsInit());
    if (anObj.SectionTracage().IsInit()) BinaryDumpInFile(aFp,anObj.SectionTracage().Val());
    BinaryDumpInFile(aFp,anObj.SLMEtape().IsInit());
    if (anObj.SLMEtape().IsInit()) BinaryDumpInFile(aFp,anObj.SLMEtape().Val());
    BinaryDumpInFile(aFp,anObj.SLMGlob().IsInit());
    if (anObj.SLMGlob().IsInit()) BinaryDumpInFile(aFp,anObj.SLMGlob().Val());
    BinaryDumpInFile(aFp,anObj.MultSLMEtape().IsInit());
    if (anObj.MultSLMEtape().IsInit()) BinaryDumpInFile(aFp,anObj.MultSLMEtape().Val());
    BinaryDumpInFile(aFp,anObj.MultSLMGlob().IsInit());
    if (anObj.MultSLMGlob().IsInit()) BinaryDumpInFile(aFp,anObj.MultSLMGlob().Val());
    BinaryDumpInFile(aFp,anObj.SectionObservations());
    BinaryDumpInFile(aFp,anObj.SectionExport().IsInit());
    if (anObj.SectionExport().IsInit()) BinaryDumpInFile(aFp,anObj.SectionExport().Val());
}

cElXMLTree * ToXMLTree(const cEtapeCompensation & anObj)
{
  XMLPushContext(anObj.mGXml);
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
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cEtapeCompensation & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.IterationsCompensation(),aTree->GetAll("IterationsCompensation",false,1));

   xml_init(anObj.SectionTracage(),aTree->Get("SectionTracage",1)); //tototo 

   xml_init(anObj.SLMEtape(),aTree->Get("SLMEtape",1)); //tototo 

   xml_init(anObj.SLMGlob(),aTree->Get("SLMGlob",1)); //tototo 

   xml_init(anObj.MultSLMEtape(),aTree->Get("MultSLMEtape",1)); //tototo 

   xml_init(anObj.MultSLMGlob(),aTree->Get("MultSLMGlob",1)); //tototo 

   xml_init(anObj.SectionObservations(),aTree->Get("SectionObservations",1)); //tototo 

   xml_init(anObj.SectionExport(),aTree->Get("SectionExport",1)); //tototo 
}

std::string  Mangling( cEtapeCompensation *) {return "9EE37035D1239EB6FE3F";};


std::list< cEtapeCompensation > & cSectionCompensation::EtapeCompensation()
{
   return mEtapeCompensation;
}

const std::list< cEtapeCompensation > & cSectionCompensation::EtapeCompensation()const 
{
   return mEtapeCompensation;
}

void  BinaryUnDumpFromFile(cSectionCompensation & anObj,ELISE_fp & aFp)
{
   { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cEtapeCompensation aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.EtapeCompensation().push_back(aVal);
        }
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cSectionCompensation & anObj)
{
    BinaryDumpInFile(aFp,(int)anObj.EtapeCompensation().size());
    for(  std::list< cEtapeCompensation >::const_iterator iT=anObj.EtapeCompensation().begin();
         iT!=anObj.EtapeCompensation().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
}

cElXMLTree * ToXMLTree(const cSectionCompensation & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"SectionCompensation",eXMLBranche);
  for
  (       std::list< cEtapeCompensation >::const_iterator it=anObj.EtapeCompensation().begin();
      it !=anObj.EtapeCompensation().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("EtapeCompensation"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cSectionCompensation & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.EtapeCompensation(),aTree->GetAll("EtapeCompensation",false,1));
}

std::string  Mangling( cSectionCompensation *) {return "B4F41333195034C3FD3F";};


cTplValGesInit< cChantierDescripteur > & cParamApero::DicoLoc()
{
   return mDicoLoc;
}

const cTplValGesInit< cChantierDescripteur > & cParamApero::DicoLoc()const 
{
   return mDicoLoc;
}


cTplValGesInit< std::string > & cParamApero::FileDebug()
{
   return mFileDebug;
}

const cTplValGesInit< std::string > & cParamApero::FileDebug()const 
{
   return mFileDebug;
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


std::list< cBDD_NewPtMul > & cParamApero::BDD_NewPtMul()
{
   return SectionBDD_Observation().BDD_NewPtMul();
}

const std::list< cBDD_NewPtMul > & cParamApero::BDD_NewPtMul()const 
{
   return SectionBDD_Observation().BDD_NewPtMul();
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


std::list< cDeclareObsRelGPS > & cParamApero::DeclareObsRelGPS()
{
   return SectionBDD_Observation().DeclareObsRelGPS();
}

const std::list< cDeclareObsRelGPS > & cParamApero::DeclareObsRelGPS()const 
{
   return SectionBDD_Observation().DeclareObsRelGPS();
}


std::string & cParamApero::PatternSel()
{
   return SectionBDD_Observation().DeclareObsCalConseq().Val().PatternSel();
}

const std::string & cParamApero::PatternSel()const 
{
   return SectionBDD_Observation().DeclareObsCalConseq().Val().PatternSel();
}


std::string & cParamApero::Key()
{
   return SectionBDD_Observation().DeclareObsCalConseq().Val().Key();
}

const std::string & cParamApero::Key()const 
{
   return SectionBDD_Observation().DeclareObsCalConseq().Val().Key();
}


cTplValGesInit< std::string > & cParamApero::KeyJump()
{
   return SectionBDD_Observation().DeclareObsCalConseq().Val().KeyJump();
}

const cTplValGesInit< std::string > & cParamApero::KeyJump()const 
{
   return SectionBDD_Observation().DeclareObsCalConseq().Val().KeyJump();
}


bool & cParamApero::AddFreeRot()
{
   return SectionBDD_Observation().DeclareObsCalConseq().Val().AddFreeRot();
}

const bool & cParamApero::AddFreeRot()const 
{
   return SectionBDD_Observation().DeclareObsCalConseq().Val().AddFreeRot();
}


cTplValGesInit< cDeclareObsCalConseq > & cParamApero::DeclareObsCalConseq()
{
   return SectionBDD_Observation().DeclareObsCalConseq();
}

const cTplValGesInit< cDeclareObsCalConseq > & cParamApero::DeclareObsCalConseq()const 
{
   return SectionBDD_Observation().DeclareObsCalConseq();
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


std::list< cGpsOffset > & cParamApero::GpsOffset()
{
   return SectionInconnues().GpsOffset();
}

const std::list< cGpsOffset > & cParamApero::GpsOffset()const 
{
   return SectionInconnues().GpsOffset();
}


std::list< cDataObsPlane > & cParamApero::DataObsPlane()
{
   return SectionInconnues().DataObsPlane();
}

const std::list< cDataObsPlane > & cParamApero::DataObsPlane()const 
{
   return SectionInconnues().DataObsPlane();
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


std::list< cBlockCamera > & cParamApero::BlockCamera()
{
   return SectionInconnues().BlockCamera();
}

const std::list< cBlockCamera > & cParamApero::BlockCamera()const 
{
   return SectionInconnues().BlockCamera();
}


cTplValGesInit< cSetOrientationInterne > & cParamApero::GlobOrInterne()
{
   return SectionInconnues().GlobOrInterne();
}

const cTplValGesInit< cSetOrientationInterne > & cParamApero::GlobOrInterne()const 
{
   return SectionInconnues().GlobOrInterne();
}


std::list< cCamGenInc > & cParamApero::CamGenInc()
{
   return SectionInconnues().CamGenInc();
}

const std::list< cCamGenInc > & cParamApero::CamGenInc()const 
{
   return SectionInconnues().CamGenInc();
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


std::string & cParamApero::IdOrient()
{
   return SectionChantier().RappelPose().Val().IdOrient();
}

const std::string & cParamApero::IdOrient()const 
{
   return SectionChantier().RappelPose().Val().IdOrient();
}


double & cParamApero::SigmaC()
{
   return SectionChantier().RappelPose().Val().SigmaC();
}

const double & cParamApero::SigmaC()const 
{
   return SectionChantier().RappelPose().Val().SigmaC();
}


double & cParamApero::SigmaR()
{
   return SectionChantier().RappelPose().Val().SigmaR();
}

const double & cParamApero::SigmaR()const 
{
   return SectionChantier().RappelPose().Val().SigmaR();
}


cElRegex_Ptr & cParamApero::PatternApply()
{
   return SectionChantier().RappelPose().Val().PatternApply();
}

const cElRegex_Ptr & cParamApero::PatternApply()const 
{
   return SectionChantier().RappelPose().Val().PatternApply();
}


cTplValGesInit< cRappelPose > & cParamApero::RappelPose()
{
   return SectionChantier().RappelPose();
}

const cTplValGesInit< cRappelPose > & cParamApero::RappelPose()const 
{
   return SectionChantier().RappelPose();
}


cTplValGesInit< int > & cParamApero::NumAttrPdsNewF()
{
   return SectionChantier().NumAttrPdsNewF();
}

const cTplValGesInit< int > & cParamApero::NumAttrPdsNewF()const 
{
   return SectionChantier().NumAttrPdsNewF();
}


cTplValGesInit< double > & cParamApero::RatioMaxDistCS()
{
   return SectionChantier().RatioMaxDistCS();
}

const cTplValGesInit< double > & cParamApero::RatioMaxDistCS()const 
{
   return SectionChantier().RatioMaxDistCS();
}


cTplValGesInit< std::string > & cParamApero::DebugVecElimTieP()
{
   return SectionChantier().DebugVecElimTieP();
}

const cTplValGesInit< std::string > & cParamApero::DebugVecElimTieP()const 
{
   return SectionChantier().DebugVecElimTieP();
}


cTplValGesInit< int > & cParamApero::DoStatElimBundle()
{
   return SectionChantier().DoStatElimBundle();
}

const cTplValGesInit< int > & cParamApero::DoStatElimBundle()const 
{
   return SectionChantier().DoStatElimBundle();
}


cTplValGesInit< double > & cParamApero::SzByPair()
{
   return SectionChantier().UseExportImageResidu().Val().SzByPair();
}

const cTplValGesInit< double > & cParamApero::SzByPair()const 
{
   return SectionChantier().UseExportImageResidu().Val().SzByPair();
}


cTplValGesInit< double > & cParamApero::SzByPose()
{
   return SectionChantier().UseExportImageResidu().Val().SzByPose();
}

const cTplValGesInit< double > & cParamApero::SzByPose()const 
{
   return SectionChantier().UseExportImageResidu().Val().SzByPose();
}


cTplValGesInit< double > & cParamApero::SzByCam()
{
   return SectionChantier().UseExportImageResidu().Val().SzByCam();
}

const cTplValGesInit< double > & cParamApero::SzByCam()const 
{
   return SectionChantier().UseExportImageResidu().Val().SzByCam();
}


cTplValGesInit< double > & cParamApero::NbMesByCase()
{
   return SectionChantier().UseExportImageResidu().Val().NbMesByCase();
}

const cTplValGesInit< double > & cParamApero::NbMesByCase()const 
{
   return SectionChantier().UseExportImageResidu().Val().NbMesByCase();
}


std::string & cParamApero::AeroExport()
{
   return SectionChantier().UseExportImageResidu().Val().AeroExport();
}

const std::string & cParamApero::AeroExport()const 
{
   return SectionChantier().UseExportImageResidu().Val().AeroExport();
}


cTplValGesInit< bool > & cParamApero::GeneratePly()
{
   return SectionChantier().UseExportImageResidu().Val().GeneratePly();
}

const cTplValGesInit< bool > & cParamApero::GeneratePly()const 
{
   return SectionChantier().UseExportImageResidu().Val().GeneratePly();
}


cTplValGesInit< int > & cParamApero::SzOrtho()
{
   return SectionChantier().UseExportImageResidu().Val().SzOrtho();
}

const cTplValGesInit< int > & cParamApero::SzOrtho()const 
{
   return SectionChantier().UseExportImageResidu().Val().SzOrtho();
}


cTplValGesInit< cUseExportImageResidu > & cParamApero::UseExportImageResidu()
{
   return SectionChantier().UseExportImageResidu();
}

const cTplValGesInit< cUseExportImageResidu > & cParamApero::UseExportImageResidu()const 
{
   return SectionChantier().UseExportImageResidu();
}


cTplValGesInit< bool > & cParamApero::UseRegulDist()
{
   return SectionChantier().UseRegulDist();
}

const cTplValGesInit< bool > & cParamApero::UseRegulDist()const 
{
   return SectionChantier().UseRegulDist();
}


cTplValGesInit< bool > & cParamApero::GBCamSupresStenCam()
{
   return SectionChantier().GBCamSupresStenCam();
}

const cTplValGesInit< bool > & cParamApero::GBCamSupresStenCam()const 
{
   return SectionChantier().GBCamSupresStenCam();
}


cTplValGesInit< bool > & cParamApero::StenCamSupresGBCam()
{
   return SectionChantier().StenCamSupresGBCam();
}

const cTplValGesInit< bool > & cParamApero::StenCamSupresGBCam()const 
{
   return SectionChantier().StenCamSupresGBCam();
}


cTplValGesInit< bool > & cParamApero::IsAperiCloud()
{
   return SectionChantier().IsAperiCloud();
}

const cTplValGesInit< bool > & cParamApero::IsAperiCloud()const 
{
   return SectionChantier().IsAperiCloud();
}


cTplValGesInit< bool > & cParamApero::IsChoixImSec()
{
   return SectionChantier().IsChoixImSec();
}

const cTplValGesInit< bool > & cParamApero::IsChoixImSec()const 
{
   return SectionChantier().IsChoixImSec();
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


cTplValGesInit< bool > & cParamApero::ExportMatrixMarket()
{
   return SectionChantier().ExportMatrixMarket();
}

const cTplValGesInit< bool > & cParamApero::ExportMatrixMarket()const 
{
   return SectionChantier().ExportMatrixMarket();
}


cTplValGesInit< double > & cParamApero::ExtensionIntervZ()
{
   return SectionChantier().ExtensionIntervZ();
}

const cTplValGesInit< double > & cParamApero::ExtensionIntervZ()const 
{
   return SectionChantier().ExtensionIntervZ();
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


cTplValGesInit< cXmlPondRegDist > & cParamApero::RegDistGlob()
{
   return SectionSolveur().RegDistGlob();
}

const cTplValGesInit< cXmlPondRegDist > & cParamApero::RegDistGlob()const 
{
   return SectionSolveur().RegDistGlob();
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

void  BinaryUnDumpFromFile(cParamApero & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.DicoLoc().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.DicoLoc().ValForcedForUnUmp(),aFp);
        }
        else  anObj.DicoLoc().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.FileDebug().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.FileDebug().ValForcedForUnUmp(),aFp);
        }
        else  anObj.FileDebug().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ShowSection().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ShowSection().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ShowSection().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.CalledByItself().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.CalledByItself().ValForcedForUnUmp(),aFp);
        }
        else  anObj.CalledByItself().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SectionMapApero().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SectionMapApero().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SectionMapApero().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.SectionBDD_Observation(),aFp);
    BinaryUnDumpFromFile(anObj.SectionInconnues(),aFp);
    BinaryUnDumpFromFile(anObj.SectionChantier(),aFp);
    BinaryUnDumpFromFile(anObj.SectionSolveur(),aFp);
    BinaryUnDumpFromFile(anObj.SectionCompensation(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cParamApero & anObj)
{
    BinaryDumpInFile(aFp,anObj.DicoLoc().IsInit());
    if (anObj.DicoLoc().IsInit()) BinaryDumpInFile(aFp,anObj.DicoLoc().Val());
    BinaryDumpInFile(aFp,anObj.FileDebug().IsInit());
    if (anObj.FileDebug().IsInit()) BinaryDumpInFile(aFp,anObj.FileDebug().Val());
    BinaryDumpInFile(aFp,anObj.ShowSection().IsInit());
    if (anObj.ShowSection().IsInit()) BinaryDumpInFile(aFp,anObj.ShowSection().Val());
    BinaryDumpInFile(aFp,anObj.CalledByItself().IsInit());
    if (anObj.CalledByItself().IsInit()) BinaryDumpInFile(aFp,anObj.CalledByItself().Val());
    BinaryDumpInFile(aFp,anObj.SectionMapApero().IsInit());
    if (anObj.SectionMapApero().IsInit()) BinaryDumpInFile(aFp,anObj.SectionMapApero().Val());
    BinaryDumpInFile(aFp,anObj.SectionBDD_Observation());
    BinaryDumpInFile(aFp,anObj.SectionInconnues());
    BinaryDumpInFile(aFp,anObj.SectionChantier());
    BinaryDumpInFile(aFp,anObj.SectionSolveur());
    BinaryDumpInFile(aFp,anObj.SectionCompensation());
}

cElXMLTree * ToXMLTree(const cParamApero & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ParamApero",eXMLBranche);
   if (anObj.DicoLoc().IsInit())
      aRes->AddFils(ToXMLTree(anObj.DicoLoc().Val())->ReTagThis("DicoLoc"));
   if (anObj.FileDebug().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("FileDebug"),anObj.FileDebug().Val())->ReTagThis("FileDebug"));
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
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cParamApero & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.DicoLoc(),aTree->Get("DicoLoc",1)); //tototo 

   xml_init(anObj.FileDebug(),aTree->Get("FileDebug",1)); //tototo 

   xml_init(anObj.ShowSection(),aTree->Get("ShowSection",1)); //tototo 

   xml_init(anObj.CalledByItself(),aTree->Get("CalledByItself",1),bool(false)); //tototo 

   xml_init(anObj.SectionMapApero(),aTree->Get("SectionMapApero",1)); //tototo 

   xml_init(anObj.SectionBDD_Observation(),aTree->Get("SectionBDD_Observation",1)); //tototo 

   xml_init(anObj.SectionInconnues(),aTree->Get("SectionInconnues",1)); //tototo 

   xml_init(anObj.SectionChantier(),aTree->Get("SectionChantier",1)); //tototo 

   xml_init(anObj.SectionSolveur(),aTree->Get("SectionSolveur",1)); //tototo 

   xml_init(anObj.SectionCompensation(),aTree->Get("SectionCompensation",1)); //tototo 
}

std::string  Mangling( cParamApero *) {return "7EC55EA857D9E0ADFF3F";};


std::string & cXmlSauvExportAperoOneIm::Name()
{
   return mName;
}

const std::string & cXmlSauvExportAperoOneIm::Name()const 
{
   return mName;
}


double & cXmlSauvExportAperoOneIm::Residual()
{
   return mResidual;
}

const double & cXmlSauvExportAperoOneIm::Residual()const 
{
   return mResidual;
}


double & cXmlSauvExportAperoOneIm::PercOk()
{
   return mPercOk;
}

const double & cXmlSauvExportAperoOneIm::PercOk()const 
{
   return mPercOk;
}


int & cXmlSauvExportAperoOneIm::NbPts()
{
   return mNbPts;
}

const int & cXmlSauvExportAperoOneIm::NbPts()const 
{
   return mNbPts;
}


int & cXmlSauvExportAperoOneIm::NbPtsMul()
{
   return mNbPtsMul;
}

const int & cXmlSauvExportAperoOneIm::NbPtsMul()const 
{
   return mNbPtsMul;
}

void  BinaryUnDumpFromFile(cXmlSauvExportAperoOneIm & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Name(),aFp);
    BinaryUnDumpFromFile(anObj.Residual(),aFp);
    BinaryUnDumpFromFile(anObj.PercOk(),aFp);
    BinaryUnDumpFromFile(anObj.NbPts(),aFp);
    BinaryUnDumpFromFile(anObj.NbPtsMul(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cXmlSauvExportAperoOneIm & anObj)
{
    BinaryDumpInFile(aFp,anObj.Name());
    BinaryDumpInFile(aFp,anObj.Residual());
    BinaryDumpInFile(aFp,anObj.PercOk());
    BinaryDumpInFile(aFp,anObj.NbPts());
    BinaryDumpInFile(aFp,anObj.NbPtsMul());
}

cElXMLTree * ToXMLTree(const cXmlSauvExportAperoOneIm & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"XmlSauvExportAperoOneIm",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Name"),anObj.Name())->ReTagThis("Name"));
   aRes->AddFils(::ToXMLTree(std::string("Residual"),anObj.Residual())->ReTagThis("Residual"));
   aRes->AddFils(::ToXMLTree(std::string("PercOk"),anObj.PercOk())->ReTagThis("PercOk"));
   aRes->AddFils(::ToXMLTree(std::string("NbPts"),anObj.NbPts())->ReTagThis("NbPts"));
   aRes->AddFils(::ToXMLTree(std::string("NbPtsMul"),anObj.NbPtsMul())->ReTagThis("NbPtsMul"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cXmlSauvExportAperoOneIm & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.Name(),aTree->Get("Name",1)); //tototo 

   xml_init(anObj.Residual(),aTree->Get("Residual",1)); //tototo 

   xml_init(anObj.PercOk(),aTree->Get("PercOk",1)); //tototo 

   xml_init(anObj.NbPts(),aTree->Get("NbPts",1)); //tototo 

   xml_init(anObj.NbPtsMul(),aTree->Get("NbPtsMul",1)); //tototo 
}

std::string  Mangling( cXmlSauvExportAperoOneIm *) {return "029F800C8A9FB184FF3F";};


std::string & cXmlSauvExportAperoOneAppuis::Name()
{
   return mName;
}

const std::string & cXmlSauvExportAperoOneAppuis::Name()const 
{
   return mName;
}


cTplValGesInit< Pt3dr > & cXmlSauvExportAperoOneAppuis::EcartFaiscTerrain()
{
   return mEcartFaiscTerrain;
}

const cTplValGesInit< Pt3dr > & cXmlSauvExportAperoOneAppuis::EcartFaiscTerrain()const 
{
   return mEcartFaiscTerrain;
}


cTplValGesInit< double > & cXmlSauvExportAperoOneAppuis::DistFaiscTerrain()
{
   return mDistFaiscTerrain;
}

const cTplValGesInit< double > & cXmlSauvExportAperoOneAppuis::DistFaiscTerrain()const 
{
   return mDistFaiscTerrain;
}


cTplValGesInit< double > & cXmlSauvExportAperoOneAppuis::EcartImMoy()
{
   return mEcartImMoy;
}

const cTplValGesInit< double > & cXmlSauvExportAperoOneAppuis::EcartImMoy()const 
{
   return mEcartImMoy;
}


cTplValGesInit< double > & cXmlSauvExportAperoOneAppuis::EcartImMax()
{
   return mEcartImMax;
}

const cTplValGesInit< double > & cXmlSauvExportAperoOneAppuis::EcartImMax()const 
{
   return mEcartImMax;
}


cTplValGesInit< std::string > & cXmlSauvExportAperoOneAppuis::NameImMax()
{
   return mNameImMax;
}

const cTplValGesInit< std::string > & cXmlSauvExportAperoOneAppuis::NameImMax()const 
{
   return mNameImMax;
}

void  BinaryUnDumpFromFile(cXmlSauvExportAperoOneAppuis & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Name(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.EcartFaiscTerrain().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.EcartFaiscTerrain().ValForcedForUnUmp(),aFp);
        }
        else  anObj.EcartFaiscTerrain().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.DistFaiscTerrain().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.DistFaiscTerrain().ValForcedForUnUmp(),aFp);
        }
        else  anObj.DistFaiscTerrain().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.EcartImMoy().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.EcartImMoy().ValForcedForUnUmp(),aFp);
        }
        else  anObj.EcartImMoy().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.EcartImMax().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.EcartImMax().ValForcedForUnUmp(),aFp);
        }
        else  anObj.EcartImMax().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NameImMax().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NameImMax().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NameImMax().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cXmlSauvExportAperoOneAppuis & anObj)
{
    BinaryDumpInFile(aFp,anObj.Name());
    BinaryDumpInFile(aFp,anObj.EcartFaiscTerrain().IsInit());
    if (anObj.EcartFaiscTerrain().IsInit()) BinaryDumpInFile(aFp,anObj.EcartFaiscTerrain().Val());
    BinaryDumpInFile(aFp,anObj.DistFaiscTerrain().IsInit());
    if (anObj.DistFaiscTerrain().IsInit()) BinaryDumpInFile(aFp,anObj.DistFaiscTerrain().Val());
    BinaryDumpInFile(aFp,anObj.EcartImMoy().IsInit());
    if (anObj.EcartImMoy().IsInit()) BinaryDumpInFile(aFp,anObj.EcartImMoy().Val());
    BinaryDumpInFile(aFp,anObj.EcartImMax().IsInit());
    if (anObj.EcartImMax().IsInit()) BinaryDumpInFile(aFp,anObj.EcartImMax().Val());
    BinaryDumpInFile(aFp,anObj.NameImMax().IsInit());
    if (anObj.NameImMax().IsInit()) BinaryDumpInFile(aFp,anObj.NameImMax().Val());
}

cElXMLTree * ToXMLTree(const cXmlSauvExportAperoOneAppuis & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"XmlSauvExportAperoOneAppuis",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Name"),anObj.Name())->ReTagThis("Name"));
   if (anObj.EcartFaiscTerrain().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("EcartFaiscTerrain"),anObj.EcartFaiscTerrain().Val())->ReTagThis("EcartFaiscTerrain"));
   if (anObj.DistFaiscTerrain().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DistFaiscTerrain"),anObj.DistFaiscTerrain().Val())->ReTagThis("DistFaiscTerrain"));
   if (anObj.EcartImMoy().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("EcartImMoy"),anObj.EcartImMoy().Val())->ReTagThis("EcartImMoy"));
   if (anObj.EcartImMax().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("EcartImMax"),anObj.EcartImMax().Val())->ReTagThis("EcartImMax"));
   if (anObj.NameImMax().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NameImMax"),anObj.NameImMax().Val())->ReTagThis("NameImMax"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cXmlSauvExportAperoOneAppuis & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.Name(),aTree->Get("Name",1)); //tototo 

   xml_init(anObj.EcartFaiscTerrain(),aTree->Get("EcartFaiscTerrain",1)); //tototo 

   xml_init(anObj.DistFaiscTerrain(),aTree->Get("DistFaiscTerrain",1)); //tototo 

   xml_init(anObj.EcartImMoy(),aTree->Get("EcartImMoy",1)); //tototo 

   xml_init(anObj.EcartImMax(),aTree->Get("EcartImMax",1)); //tototo 

   xml_init(anObj.NameImMax(),aTree->Get("NameImMax",1)); //tototo 
}

std::string  Mangling( cXmlSauvExportAperoOneAppuis *) {return "0008E3516AB32481FD3F";};


int & cXmlSauvExportAperoOneMult::Multiplicity()
{
   return mMultiplicity;
}

const int & cXmlSauvExportAperoOneMult::Multiplicity()const 
{
   return mMultiplicity;
}


double & cXmlSauvExportAperoOneMult::Residual()
{
   return mResidual;
}

const double & cXmlSauvExportAperoOneMult::Residual()const 
{
   return mResidual;
}


int & cXmlSauvExportAperoOneMult::NbPts()
{
   return mNbPts;
}

const int & cXmlSauvExportAperoOneMult::NbPts()const 
{
   return mNbPts;
}


double & cXmlSauvExportAperoOneMult::PercOk()
{
   return mPercOk;
}

const double & cXmlSauvExportAperoOneMult::PercOk()const 
{
   return mPercOk;
}

void  BinaryUnDumpFromFile(cXmlSauvExportAperoOneMult & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Multiplicity(),aFp);
    BinaryUnDumpFromFile(anObj.Residual(),aFp);
    BinaryUnDumpFromFile(anObj.NbPts(),aFp);
    BinaryUnDumpFromFile(anObj.PercOk(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cXmlSauvExportAperoOneMult & anObj)
{
    BinaryDumpInFile(aFp,anObj.Multiplicity());
    BinaryDumpInFile(aFp,anObj.Residual());
    BinaryDumpInFile(aFp,anObj.NbPts());
    BinaryDumpInFile(aFp,anObj.PercOk());
}

cElXMLTree * ToXMLTree(const cXmlSauvExportAperoOneMult & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"XmlSauvExportAperoOneMult",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Multiplicity"),anObj.Multiplicity())->ReTagThis("Multiplicity"));
   aRes->AddFils(::ToXMLTree(std::string("Residual"),anObj.Residual())->ReTagThis("Residual"));
   aRes->AddFils(::ToXMLTree(std::string("NbPts"),anObj.NbPts())->ReTagThis("NbPts"));
   aRes->AddFils(::ToXMLTree(std::string("PercOk"),anObj.PercOk())->ReTagThis("PercOk"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cXmlSauvExportAperoOneMult & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.Multiplicity(),aTree->Get("Multiplicity",1)); //tototo 

   xml_init(anObj.Residual(),aTree->Get("Residual",1)); //tototo 

   xml_init(anObj.NbPts(),aTree->Get("NbPts",1)); //tototo 

   xml_init(anObj.PercOk(),aTree->Get("PercOk",1)); //tototo 
}

std::string  Mangling( cXmlSauvExportAperoOneMult *) {return "E4FF6CB5BD1C7FDFFE3F";};


std::list< cXmlSauvExportAperoOneAppuis > & cXmlSauvExportAperoOneIter::OneAppui()
{
   return mOneAppui;
}

const std::list< cXmlSauvExportAperoOneAppuis > & cXmlSauvExportAperoOneIter::OneAppui()const 
{
   return mOneAppui;
}


std::list< cXmlSauvExportAperoOneIm > & cXmlSauvExportAperoOneIter::OneIm()
{
   return mOneIm;
}

const std::list< cXmlSauvExportAperoOneIm > & cXmlSauvExportAperoOneIter::OneIm()const 
{
   return mOneIm;
}


std::list< cXmlSauvExportAperoOneMult > & cXmlSauvExportAperoOneIter::OneMult()
{
   return mOneMult;
}

const std::list< cXmlSauvExportAperoOneMult > & cXmlSauvExportAperoOneIter::OneMult()const 
{
   return mOneMult;
}


double & cXmlSauvExportAperoOneIter::AverageResidual()
{
   return mAverageResidual;
}

const double & cXmlSauvExportAperoOneIter::AverageResidual()const 
{
   return mAverageResidual;
}


int & cXmlSauvExportAperoOneIter::NumIter()
{
   return mNumIter;
}

const int & cXmlSauvExportAperoOneIter::NumIter()const 
{
   return mNumIter;
}


int & cXmlSauvExportAperoOneIter::NumEtape()
{
   return mNumEtape;
}

const int & cXmlSauvExportAperoOneIter::NumEtape()const 
{
   return mNumEtape;
}


cTplValGesInit< double > & cXmlSauvExportAperoOneIter::EvolMax()
{
   return mEvolMax;
}

const cTplValGesInit< double > & cXmlSauvExportAperoOneIter::EvolMax()const 
{
   return mEvolMax;
}


cTplValGesInit< double > & cXmlSauvExportAperoOneIter::EvolMoy()
{
   return mEvolMoy;
}

const cTplValGesInit< double > & cXmlSauvExportAperoOneIter::EvolMoy()const 
{
   return mEvolMoy;
}


cTplValGesInit< std::string > & cXmlSauvExportAperoOneIter::ImWorstRes()
{
   return mImWorstRes;
}

const cTplValGesInit< std::string > & cXmlSauvExportAperoOneIter::ImWorstRes()const 
{
   return mImWorstRes;
}


cTplValGesInit< double > & cXmlSauvExportAperoOneIter::WorstRes()
{
   return mWorstRes;
}

const cTplValGesInit< double > & cXmlSauvExportAperoOneIter::WorstRes()const 
{
   return mWorstRes;
}

void  BinaryUnDumpFromFile(cXmlSauvExportAperoOneIter & anObj,ELISE_fp & aFp)
{
   { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cXmlSauvExportAperoOneAppuis aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.OneAppui().push_back(aVal);
        }
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cXmlSauvExportAperoOneIm aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.OneIm().push_back(aVal);
        }
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cXmlSauvExportAperoOneMult aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.OneMult().push_back(aVal);
        }
  } ;
    BinaryUnDumpFromFile(anObj.AverageResidual(),aFp);
    BinaryUnDumpFromFile(anObj.NumIter(),aFp);
    BinaryUnDumpFromFile(anObj.NumEtape(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.EvolMax().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.EvolMax().ValForcedForUnUmp(),aFp);
        }
        else  anObj.EvolMax().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.EvolMoy().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.EvolMoy().ValForcedForUnUmp(),aFp);
        }
        else  anObj.EvolMoy().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ImWorstRes().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ImWorstRes().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ImWorstRes().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.WorstRes().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.WorstRes().ValForcedForUnUmp(),aFp);
        }
        else  anObj.WorstRes().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cXmlSauvExportAperoOneIter & anObj)
{
    BinaryDumpInFile(aFp,(int)anObj.OneAppui().size());
    for(  std::list< cXmlSauvExportAperoOneAppuis >::const_iterator iT=anObj.OneAppui().begin();
         iT!=anObj.OneAppui().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,(int)anObj.OneIm().size());
    for(  std::list< cXmlSauvExportAperoOneIm >::const_iterator iT=anObj.OneIm().begin();
         iT!=anObj.OneIm().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,(int)anObj.OneMult().size());
    for(  std::list< cXmlSauvExportAperoOneMult >::const_iterator iT=anObj.OneMult().begin();
         iT!=anObj.OneMult().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.AverageResidual());
    BinaryDumpInFile(aFp,anObj.NumIter());
    BinaryDumpInFile(aFp,anObj.NumEtape());
    BinaryDumpInFile(aFp,anObj.EvolMax().IsInit());
    if (anObj.EvolMax().IsInit()) BinaryDumpInFile(aFp,anObj.EvolMax().Val());
    BinaryDumpInFile(aFp,anObj.EvolMoy().IsInit());
    if (anObj.EvolMoy().IsInit()) BinaryDumpInFile(aFp,anObj.EvolMoy().Val());
    BinaryDumpInFile(aFp,anObj.ImWorstRes().IsInit());
    if (anObj.ImWorstRes().IsInit()) BinaryDumpInFile(aFp,anObj.ImWorstRes().Val());
    BinaryDumpInFile(aFp,anObj.WorstRes().IsInit());
    if (anObj.WorstRes().IsInit()) BinaryDumpInFile(aFp,anObj.WorstRes().Val());
}

cElXMLTree * ToXMLTree(const cXmlSauvExportAperoOneIter & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"XmlSauvExportAperoOneIter",eXMLBranche);
  for
  (       std::list< cXmlSauvExportAperoOneAppuis >::const_iterator it=anObj.OneAppui().begin();
      it !=anObj.OneAppui().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("OneAppui"));
  for
  (       std::list< cXmlSauvExportAperoOneIm >::const_iterator it=anObj.OneIm().begin();
      it !=anObj.OneIm().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("OneIm"));
  for
  (       std::list< cXmlSauvExportAperoOneMult >::const_iterator it=anObj.OneMult().begin();
      it !=anObj.OneMult().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("OneMult"));
   aRes->AddFils(::ToXMLTree(std::string("AverageResidual"),anObj.AverageResidual())->ReTagThis("AverageResidual"));
   aRes->AddFils(::ToXMLTree(std::string("NumIter"),anObj.NumIter())->ReTagThis("NumIter"));
   aRes->AddFils(::ToXMLTree(std::string("NumEtape"),anObj.NumEtape())->ReTagThis("NumEtape"));
   if (anObj.EvolMax().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("EvolMax"),anObj.EvolMax().Val())->ReTagThis("EvolMax"));
   if (anObj.EvolMoy().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("EvolMoy"),anObj.EvolMoy().Val())->ReTagThis("EvolMoy"));
   if (anObj.ImWorstRes().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ImWorstRes"),anObj.ImWorstRes().Val())->ReTagThis("ImWorstRes"));
   if (anObj.WorstRes().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("WorstRes"),anObj.WorstRes().Val())->ReTagThis("WorstRes"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cXmlSauvExportAperoOneIter & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.OneAppui(),aTree->GetAll("OneAppui",false,1));

   xml_init(anObj.OneIm(),aTree->GetAll("OneIm",false,1));

   xml_init(anObj.OneMult(),aTree->GetAll("OneMult",false,1));

   xml_init(anObj.AverageResidual(),aTree->Get("AverageResidual",1)); //tototo 

   xml_init(anObj.NumIter(),aTree->Get("NumIter",1)); //tototo 

   xml_init(anObj.NumEtape(),aTree->Get("NumEtape",1)); //tototo 

   xml_init(anObj.EvolMax(),aTree->Get("EvolMax",1)); //tototo 

   xml_init(anObj.EvolMoy(),aTree->Get("EvolMoy",1)); //tototo 

   xml_init(anObj.ImWorstRes(),aTree->Get("ImWorstRes",1)); //tototo 

   xml_init(anObj.WorstRes(),aTree->Get("WorstRes",1)); //tototo 
}

std::string  Mangling( cXmlSauvExportAperoOneIter *) {return "A8514BA80DE22C9EFCBF";};


std::list< cXmlSauvExportAperoOneIter > & cXmlSauvExportAperoGlob::Iters()
{
   return mIters;
}

const std::list< cXmlSauvExportAperoOneIter > & cXmlSauvExportAperoGlob::Iters()const 
{
   return mIters;
}

void  BinaryUnDumpFromFile(cXmlSauvExportAperoGlob & anObj,ELISE_fp & aFp)
{
   { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cXmlSauvExportAperoOneIter aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.Iters().push_back(aVal);
        }
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cXmlSauvExportAperoGlob & anObj)
{
    BinaryDumpInFile(aFp,(int)anObj.Iters().size());
    for(  std::list< cXmlSauvExportAperoOneIter >::const_iterator iT=anObj.Iters().begin();
         iT!=anObj.Iters().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
}

cElXMLTree * ToXMLTree(const cXmlSauvExportAperoGlob & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"XmlSauvExportAperoGlob",eXMLBranche);
  for
  (       std::list< cXmlSauvExportAperoOneIter >::const_iterator it=anObj.Iters().begin();
      it !=anObj.Iters().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("Iters"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cXmlSauvExportAperoGlob & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.Iters(),aTree->GetAll("Iters",false,1));
}

std::string  Mangling( cXmlSauvExportAperoGlob *) {return "08E72191A4427C8BFD3F";};


double & cXmlOneResultRTA::Mult()
{
   return mMult;
}

const double & cXmlOneResultRTA::Mult()const 
{
   return mMult;
}


double & cXmlOneResultRTA::MoyErr()
{
   return mMoyErr;
}

const double & cXmlOneResultRTA::MoyErr()const 
{
   return mMoyErr;
}


std::list< cXmlSauvExportAperoOneAppuis > & cXmlOneResultRTA::OneAppui()
{
   return mOneAppui;
}

const std::list< cXmlSauvExportAperoOneAppuis > & cXmlOneResultRTA::OneAppui()const 
{
   return mOneAppui;
}

void  BinaryUnDumpFromFile(cXmlOneResultRTA & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Mult(),aFp);
    BinaryUnDumpFromFile(anObj.MoyErr(),aFp);
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cXmlSauvExportAperoOneAppuis aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.OneAppui().push_back(aVal);
        }
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cXmlOneResultRTA & anObj)
{
    BinaryDumpInFile(aFp,anObj.Mult());
    BinaryDumpInFile(aFp,anObj.MoyErr());
    BinaryDumpInFile(aFp,(int)anObj.OneAppui().size());
    for(  std::list< cXmlSauvExportAperoOneAppuis >::const_iterator iT=anObj.OneAppui().begin();
         iT!=anObj.OneAppui().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
}

cElXMLTree * ToXMLTree(const cXmlOneResultRTA & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"XmlOneResultRTA",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Mult"),anObj.Mult())->ReTagThis("Mult"));
   aRes->AddFils(::ToXMLTree(std::string("MoyErr"),anObj.MoyErr())->ReTagThis("MoyErr"));
  for
  (       std::list< cXmlSauvExportAperoOneAppuis >::const_iterator it=anObj.OneAppui().begin();
      it !=anObj.OneAppui().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("OneAppui"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cXmlOneResultRTA & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.Mult(),aTree->Get("Mult",1)); //tototo 

   xml_init(anObj.MoyErr(),aTree->Get("MoyErr",1)); //tototo 

   xml_init(anObj.OneAppui(),aTree->GetAll("OneAppui",false,1));
}

std::string  Mangling( cXmlOneResultRTA *) {return "BEB73333D759BCBAFD3F";};


double & cXmlResultRTA::BestMult()
{
   return mBestMult;
}

const double & cXmlResultRTA::BestMult()const 
{
   return mBestMult;
}


double & cXmlResultRTA::BestMoyErr()
{
   return mBestMoyErr;
}

const double & cXmlResultRTA::BestMoyErr()const 
{
   return mBestMoyErr;
}


std::list< cXmlOneResultRTA > & cXmlResultRTA::RTA()
{
   return mRTA;
}

const std::list< cXmlOneResultRTA > & cXmlResultRTA::RTA()const 
{
   return mRTA;
}

void  BinaryUnDumpFromFile(cXmlResultRTA & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.BestMult(),aFp);
    BinaryUnDumpFromFile(anObj.BestMoyErr(),aFp);
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cXmlOneResultRTA aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.RTA().push_back(aVal);
        }
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cXmlResultRTA & anObj)
{
    BinaryDumpInFile(aFp,anObj.BestMult());
    BinaryDumpInFile(aFp,anObj.BestMoyErr());
    BinaryDumpInFile(aFp,(int)anObj.RTA().size());
    for(  std::list< cXmlOneResultRTA >::const_iterator iT=anObj.RTA().begin();
         iT!=anObj.RTA().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
}

cElXMLTree * ToXMLTree(const cXmlResultRTA & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"XmlResultRTA",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("BestMult"),anObj.BestMult())->ReTagThis("BestMult"));
   aRes->AddFils(::ToXMLTree(std::string("BestMoyErr"),anObj.BestMoyErr())->ReTagThis("BestMoyErr"));
  for
  (       std::list< cXmlOneResultRTA >::const_iterator it=anObj.RTA().begin();
      it !=anObj.RTA().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("RTA"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cXmlResultRTA & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.BestMult(),aTree->Get("BestMult",1)); //tototo 

   xml_init(anObj.BestMoyErr(),aTree->Get("BestMoyErr",1)); //tototo 

   xml_init(anObj.RTA(),aTree->GetAll("RTA",false,1));
}

std::string  Mangling( cXmlResultRTA *) {return "BAE07F625BE4E1EBFE3F";};


std::string & cSensibDateOneInc::NameBloc()
{
   return mNameBloc;
}

const std::string & cSensibDateOneInc::NameBloc()const 
{
   return mNameBloc;
}


std::string & cSensibDateOneInc::NameInc()
{
   return mNameInc;
}

const std::string & cSensibDateOneInc::NameInc()const 
{
   return mNameInc;
}


double & cSensibDateOneInc::SensibParamDir()
{
   return mSensibParamDir;
}

const double & cSensibDateOneInc::SensibParamDir()const 
{
   return mSensibParamDir;
}


double & cSensibDateOneInc::SensibParamInv()
{
   return mSensibParamInv;
}

const double & cSensibDateOneInc::SensibParamInv()const 
{
   return mSensibParamInv;
}


double & cSensibDateOneInc::SensibParamVar()
{
   return mSensibParamVar;
}

const double & cSensibDateOneInc::SensibParamVar()const 
{
   return mSensibParamVar;
}

void  BinaryUnDumpFromFile(cSensibDateOneInc & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.NameBloc(),aFp);
    BinaryUnDumpFromFile(anObj.NameInc(),aFp);
    BinaryUnDumpFromFile(anObj.SensibParamDir(),aFp);
    BinaryUnDumpFromFile(anObj.SensibParamInv(),aFp);
    BinaryUnDumpFromFile(anObj.SensibParamVar(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cSensibDateOneInc & anObj)
{
    BinaryDumpInFile(aFp,anObj.NameBloc());
    BinaryDumpInFile(aFp,anObj.NameInc());
    BinaryDumpInFile(aFp,anObj.SensibParamDir());
    BinaryDumpInFile(aFp,anObj.SensibParamInv());
    BinaryDumpInFile(aFp,anObj.SensibParamVar());
}

cElXMLTree * ToXMLTree(const cSensibDateOneInc & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"SensibDateOneInc",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("NameBloc"),anObj.NameBloc())->ReTagThis("NameBloc"));
   aRes->AddFils(::ToXMLTree(std::string("NameInc"),anObj.NameInc())->ReTagThis("NameInc"));
   aRes->AddFils(::ToXMLTree(std::string("SensibParamDir"),anObj.SensibParamDir())->ReTagThis("SensibParamDir"));
   aRes->AddFils(::ToXMLTree(std::string("SensibParamInv"),anObj.SensibParamInv())->ReTagThis("SensibParamInv"));
   aRes->AddFils(::ToXMLTree(std::string("SensibParamVar"),anObj.SensibParamVar())->ReTagThis("SensibParamVar"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cSensibDateOneInc & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.NameBloc(),aTree->Get("NameBloc",1)); //tototo 

   xml_init(anObj.NameInc(),aTree->Get("NameInc",1)); //tototo 

   xml_init(anObj.SensibParamDir(),aTree->Get("SensibParamDir",1)); //tototo 

   xml_init(anObj.SensibParamInv(),aTree->Get("SensibParamInv",1)); //tototo 

   xml_init(anObj.SensibParamVar(),aTree->Get("SensibParamVar",1)); //tototo 
}

std::string  Mangling( cSensibDateOneInc *) {return "3C562B732E21E4A6FE3F";};


std::vector< cSensibDateOneInc > & cXmlNameSensibs::SensibDateOneInc()
{
   return mSensibDateOneInc;
}

const std::vector< cSensibDateOneInc > & cXmlNameSensibs::SensibDateOneInc()const 
{
   return mSensibDateOneInc;
}

void  BinaryUnDumpFromFile(cXmlNameSensibs & anObj,ELISE_fp & aFp)
{
   { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cSensibDateOneInc aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.SensibDateOneInc().push_back(aVal);
        }
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cXmlNameSensibs & anObj)
{
    BinaryDumpInFile(aFp,(int)anObj.SensibDateOneInc().size());
    for(  std::vector< cSensibDateOneInc >::const_iterator iT=anObj.SensibDateOneInc().begin();
         iT!=anObj.SensibDateOneInc().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
}

cElXMLTree * ToXMLTree(const cXmlNameSensibs & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"XmlNameSensibs",eXMLBranche);
  for
  (       std::vector< cSensibDateOneInc >::const_iterator it=anObj.SensibDateOneInc().begin();
      it !=anObj.SensibDateOneInc().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("SensibDateOneInc"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cXmlNameSensibs & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.SensibDateOneInc(),aTree->GetAll("SensibDateOneInc",false,1));
}

std::string  Mangling( cXmlNameSensibs *) {return "0075E0AC8422AEF9F6BF";};


std::vector< Pt2dr > & cXmlOneContourCamera::Pt()
{
   return mPt;
}

const std::vector< Pt2dr > & cXmlOneContourCamera::Pt()const 
{
   return mPt;
}

void  BinaryUnDumpFromFile(cXmlOneContourCamera & anObj,ELISE_fp & aFp)
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
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cXmlOneContourCamera & anObj)
{
    BinaryDumpInFile(aFp,(int)anObj.Pt().size());
    for(  std::vector< Pt2dr >::const_iterator iT=anObj.Pt().begin();
         iT!=anObj.Pt().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
}

cElXMLTree * ToXMLTree(const cXmlOneContourCamera & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"XmlOneContourCamera",eXMLBranche);
  for
  (       std::vector< Pt2dr >::const_iterator it=anObj.Pt().begin();
      it !=anObj.Pt().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("Pt"),(*it))->ReTagThis("Pt"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cXmlOneContourCamera & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.Pt(),aTree->GetAll("Pt",false,1));
}

std::string  Mangling( cXmlOneContourCamera *) {return "AA553F39973E9CC5FD3F";};

// };
