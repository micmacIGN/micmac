#include "StdAfx.h"
#include "cParamNewRechPH.h"
// NOMORE ...
eTypePtRemark  Str2eTypePtRemark(const std::string & aName)
{
   if (aName=="eTPR_LaplMax")
      return eTPR_LaplMax;
   else if (aName=="eTPR_LaplMin")
      return eTPR_LaplMin;
   else if (aName=="eTPR_GrayMax")
      return eTPR_GrayMax;
   else if (aName=="eTPR_GrayMin")
      return eTPR_GrayMin;
   else if (aName=="eTPR_BifurqMax")
      return eTPR_BifurqMax;
   else if (aName=="eTPR_BifurqMin")
      return eTPR_BifurqMin;
   else if (aName=="eTPR_NoLabel")
      return eTPR_NoLabel;
   else if (aName=="eTPR_GraySadl")
      return eTPR_GraySadl;
   else if (aName=="eTPR_BifurqSadl")
      return eTPR_BifurqSadl;
  else
  {
      cout << aName << " is not a correct value for enum eTypePtRemark\n" ;
      ELISE_ASSERT(false,"XML enum value error");
  }
  return (eTypePtRemark) 0;
}
void xml_init(eTypePtRemark & aVal,cElXMLTree * aTree)
{
   aVal= Str2eTypePtRemark(aTree->Contenu());
}
std::string  eToString(const eTypePtRemark & anObj)
{
   if (anObj==eTPR_LaplMax)
      return  "eTPR_LaplMax";
   if (anObj==eTPR_LaplMin)
      return  "eTPR_LaplMin";
   if (anObj==eTPR_GrayMax)
      return  "eTPR_GrayMax";
   if (anObj==eTPR_GrayMin)
      return  "eTPR_GrayMin";
   if (anObj==eTPR_BifurqMax)
      return  "eTPR_BifurqMax";
   if (anObj==eTPR_BifurqMin)
      return  "eTPR_BifurqMin";
   if (anObj==eTPR_NoLabel)
      return  "eTPR_NoLabel";
   if (anObj==eTPR_GraySadl)
      return  "eTPR_GraySadl";
   if (anObj==eTPR_BifurqSadl)
      return  "eTPR_BifurqSadl";
 std::cout << "Enum = eTypePtRemark\n";
   ELISE_ASSERT(false,"Bad Value in eToString for enum value ");
   return "";
}

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eTypePtRemark & anObj)
{
      return  cElXMLTree::ValueNode(aNameTag,eToString(anObj));
}

void  BinaryDumpInFile(ELISE_fp & aFp,const eTypePtRemark & anObj)
{
   BinaryDumpInFile(aFp,int(anObj));
}

void  BinaryUnDumpFromFile(eTypePtRemark & anObj,ELISE_fp & aFp)
{
   int aIVal;
   BinaryUnDumpFromFile(aIVal,aFp);
   anObj=(eTypePtRemark) aIVal;
}

std::string  Mangling( eTypePtRemark *) {return "26AA66D133691995FF3F";};

eTypeVecInvarR  Str2eTypeVecInvarR(const std::string & aName)
{
   if (aName=="eTVIR_Curve")
      return eTVIR_Curve;
   else if (aName=="eTVIR_ACR0")
      return eTVIR_ACR0;
   else if (aName=="eTVIR_ACGT")
      return eTVIR_ACGT;
   else if (aName=="eTVIR_ACGR")
      return eTVIR_ACGR;
   else if (aName=="eTVIR_LogPol")
      return eTVIR_LogPol;
   else if (aName=="eTVIR_NoLabel")
      return eTVIR_NoLabel;
  else
  {
      cout << aName << " is not a correct value for enum eTypeVecInvarR\n" ;
      ELISE_ASSERT(false,"XML enum value error");
  }
  return (eTypeVecInvarR) 0;
}
void xml_init(eTypeVecInvarR & aVal,cElXMLTree * aTree)
{
   aVal= Str2eTypeVecInvarR(aTree->Contenu());
}
std::string  eToString(const eTypeVecInvarR & anObj)
{
   if (anObj==eTVIR_Curve)
      return  "eTVIR_Curve";
   if (anObj==eTVIR_ACR0)
      return  "eTVIR_ACR0";
   if (anObj==eTVIR_ACGT)
      return  "eTVIR_ACGT";
   if (anObj==eTVIR_ACGR)
      return  "eTVIR_ACGR";
   if (anObj==eTVIR_LogPol)
      return  "eTVIR_LogPol";
   if (anObj==eTVIR_NoLabel)
      return  "eTVIR_NoLabel";
 std::cout << "Enum = eTypeVecInvarR\n";
   ELISE_ASSERT(false,"Bad Value in eToString for enum value ");
   return "";
}

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eTypeVecInvarR & anObj)
{
      return  cElXMLTree::ValueNode(aNameTag,eToString(anObj));
}

void  BinaryDumpInFile(ELISE_fp & aFp,const eTypeVecInvarR & anObj)
{
   BinaryDumpInFile(aFp,int(anObj));
}

void  BinaryUnDumpFromFile(eTypeVecInvarR & anObj,ELISE_fp & aFp)
{
   int aIVal;
   BinaryUnDumpFromFile(aIVal,aFp);
   anObj=(eTypeVecInvarR) aIVal;
}

std::string  Mangling( eTypeVecInvarR *) {return "76F9B402C27F2998FF3F";};

eTypeInvRad  Str2eTypeInvRad(const std::string & aName)
{
   if (aName=="eTIR_Radiom")
      return eTIR_Radiom;
   else if (aName=="eTIR_GradRad")
      return eTIR_GradRad;
   else if (aName=="eTIR_GradCroise")
      return eTIR_GradCroise;
   else if (aName=="eTIR_GradTan")
      return eTIR_GradTan;
   else if (aName=="eTIR_GradTanPiS2")
      return eTIR_GradTanPiS2;
   else if (aName=="eTIR_GradTanPi")
      return eTIR_GradTanPi;
   else if (aName=="eTIR_LaplRad")
      return eTIR_LaplRad;
   else if (aName=="eTIR_LaplTan")
      return eTIR_LaplTan;
   else if (aName=="eTIR_LaplCrois")
      return eTIR_LaplCrois;
   else if (aName=="eTIR_DiffOpposePi")
      return eTIR_DiffOpposePi;
   else if (aName=="eTIR_DiffOpposePiS2")
      return eTIR_DiffOpposePiS2;
   else if (aName=="eTIR_Sq_Radiom")
      return eTIR_Sq_Radiom;
   else if (aName=="eTIR_Sq_GradRad")
      return eTIR_Sq_GradRad;
   else if (aName=="eTIR_Sq_GradCroise")
      return eTIR_Sq_GradCroise;
   else if (aName=="eTIR_Sq_GradTan")
      return eTIR_Sq_GradTan;
   else if (aName=="eTIR_Sq_GradTanPiS2")
      return eTIR_Sq_GradTanPiS2;
   else if (aName=="eTIR_Sq_GradTanPi")
      return eTIR_Sq_GradTanPi;
   else if (aName=="eTIR_Sq_LaplRad")
      return eTIR_Sq_LaplRad;
   else if (aName=="eTIR_Sq_LaplTan")
      return eTIR_Sq_LaplTan;
   else if (aName=="eTIR_Sq_LaplCrois")
      return eTIR_Sq_LaplCrois;
   else if (aName=="eTIR_Sq_DiffOpposePi")
      return eTIR_Sq_DiffOpposePi;
   else if (aName=="eTIR_Sq_DiffOpposePiS2")
      return eTIR_Sq_DiffOpposePiS2;
   else if (aName=="eTIR_Cub_Radiom")
      return eTIR_Cub_Radiom;
   else if (aName=="eTIR_Cub_GradRad")
      return eTIR_Cub_GradRad;
   else if (aName=="eTIR_Cub_GradCroise")
      return eTIR_Cub_GradCroise;
   else if (aName=="eTIR_Cub_GradTan")
      return eTIR_Cub_GradTan;
   else if (aName=="eTIR_Cub_GradTanPiS2")
      return eTIR_Cub_GradTanPiS2;
   else if (aName=="eTIR_Cub_GradTanPi")
      return eTIR_Cub_GradTanPi;
   else if (aName=="eTIR_Cub_LaplRad")
      return eTIR_Cub_LaplRad;
   else if (aName=="eTIR_Cub_LaplTan")
      return eTIR_Cub_LaplTan;
   else if (aName=="eTIR_Cub_LaplCrois")
      return eTIR_Cub_LaplCrois;
   else if (aName=="eTIR_Cub_DiffOpposePi")
      return eTIR_Cub_DiffOpposePi;
   else if (aName=="eTIR_Cub_DiffOpposePiS2")
      return eTIR_Cub_DiffOpposePiS2;
   else if (aName=="eTIR_NoLabel")
      return eTIR_NoLabel;
  else
  {
      cout << aName << " is not a correct value for enum eTypeInvRad\n" ;
      ELISE_ASSERT(false,"XML enum value error");
  }
  return (eTypeInvRad) 0;
}
void xml_init(eTypeInvRad & aVal,cElXMLTree * aTree)
{
   aVal= Str2eTypeInvRad(aTree->Contenu());
}
std::string  eToString(const eTypeInvRad & anObj)
{
   if (anObj==eTIR_Radiom)
      return  "eTIR_Radiom";
   if (anObj==eTIR_GradRad)
      return  "eTIR_GradRad";
   if (anObj==eTIR_GradCroise)
      return  "eTIR_GradCroise";
   if (anObj==eTIR_GradTan)
      return  "eTIR_GradTan";
   if (anObj==eTIR_GradTanPiS2)
      return  "eTIR_GradTanPiS2";
   if (anObj==eTIR_GradTanPi)
      return  "eTIR_GradTanPi";
   if (anObj==eTIR_LaplRad)
      return  "eTIR_LaplRad";
   if (anObj==eTIR_LaplTan)
      return  "eTIR_LaplTan";
   if (anObj==eTIR_LaplCrois)
      return  "eTIR_LaplCrois";
   if (anObj==eTIR_DiffOpposePi)
      return  "eTIR_DiffOpposePi";
   if (anObj==eTIR_DiffOpposePiS2)
      return  "eTIR_DiffOpposePiS2";
   if (anObj==eTIR_Sq_Radiom)
      return  "eTIR_Sq_Radiom";
   if (anObj==eTIR_Sq_GradRad)
      return  "eTIR_Sq_GradRad";
   if (anObj==eTIR_Sq_GradCroise)
      return  "eTIR_Sq_GradCroise";
   if (anObj==eTIR_Sq_GradTan)
      return  "eTIR_Sq_GradTan";
   if (anObj==eTIR_Sq_GradTanPiS2)
      return  "eTIR_Sq_GradTanPiS2";
   if (anObj==eTIR_Sq_GradTanPi)
      return  "eTIR_Sq_GradTanPi";
   if (anObj==eTIR_Sq_LaplRad)
      return  "eTIR_Sq_LaplRad";
   if (anObj==eTIR_Sq_LaplTan)
      return  "eTIR_Sq_LaplTan";
   if (anObj==eTIR_Sq_LaplCrois)
      return  "eTIR_Sq_LaplCrois";
   if (anObj==eTIR_Sq_DiffOpposePi)
      return  "eTIR_Sq_DiffOpposePi";
   if (anObj==eTIR_Sq_DiffOpposePiS2)
      return  "eTIR_Sq_DiffOpposePiS2";
   if (anObj==eTIR_Cub_Radiom)
      return  "eTIR_Cub_Radiom";
   if (anObj==eTIR_Cub_GradRad)
      return  "eTIR_Cub_GradRad";
   if (anObj==eTIR_Cub_GradCroise)
      return  "eTIR_Cub_GradCroise";
   if (anObj==eTIR_Cub_GradTan)
      return  "eTIR_Cub_GradTan";
   if (anObj==eTIR_Cub_GradTanPiS2)
      return  "eTIR_Cub_GradTanPiS2";
   if (anObj==eTIR_Cub_GradTanPi)
      return  "eTIR_Cub_GradTanPi";
   if (anObj==eTIR_Cub_LaplRad)
      return  "eTIR_Cub_LaplRad";
   if (anObj==eTIR_Cub_LaplTan)
      return  "eTIR_Cub_LaplTan";
   if (anObj==eTIR_Cub_LaplCrois)
      return  "eTIR_Cub_LaplCrois";
   if (anObj==eTIR_Cub_DiffOpposePi)
      return  "eTIR_Cub_DiffOpposePi";
   if (anObj==eTIR_Cub_DiffOpposePiS2)
      return  "eTIR_Cub_DiffOpposePiS2";
   if (anObj==eTIR_NoLabel)
      return  "eTIR_NoLabel";
 std::cout << "Enum = eTypeInvRad\n";
   ELISE_ASSERT(false,"Bad Value in eToString for enum value ");
   return "";
}

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eTypeInvRad & anObj)
{
      return  cElXMLTree::ValueNode(aNameTag,eToString(anObj));
}

void  BinaryDumpInFile(ELISE_fp & aFp,const eTypeInvRad & anObj)
{
   BinaryDumpInFile(aFp,int(anObj));
}

void  BinaryUnDumpFromFile(eTypeInvRad & anObj,ELISE_fp & aFp)
{
   int aIVal;
   BinaryUnDumpFromFile(aIVal,aFp);
   anObj=(eTypeInvRad) aIVal;
}

std::string  Mangling( eTypeInvRad *) {return "BBD3C3ECAADC55CAFE3F";};


Pt2dr & cPtSc::Pt()
{
   return mPt;
}

const Pt2dr & cPtSc::Pt()const 
{
   return mPt;
}


double & cPtSc::Scale()
{
   return mScale;
}

const double & cPtSc::Scale()const 
{
   return mScale;
}

void  BinaryUnDumpFromFile(cPtSc & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Pt(),aFp);
    BinaryUnDumpFromFile(anObj.Scale(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cPtSc & anObj)
{
    BinaryDumpInFile(aFp,anObj.Pt());
    BinaryDumpInFile(aFp,anObj.Scale());
}

cElXMLTree * ToXMLTree(const cPtSc & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"PtSc",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Pt"),anObj.Pt())->ReTagThis("Pt"));
   aRes->AddFils(::ToXMLTree(std::string("Scale"),anObj.Scale())->ReTagThis("Scale"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cPtSc & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.Pt(),aTree->Get("Pt",1)); //tototo 

   xml_init(anObj.Scale(),aTree->Get("Scale",1)); //tototo 
}

std::string  Mangling( cPtSc *) {return "4BF07568F29A60FDFD3F";};


Im2D_INT2 & cXml_TestDMP::PxMin()
{
   return mPxMin;
}

const Im2D_INT2 & cXml_TestDMP::PxMin()const 
{
   return mPxMin;
}


Im2D_INT2 & cXml_TestDMP::PxMax()
{
   return mPxMax;
}

const Im2D_INT2 & cXml_TestDMP::PxMax()const 
{
   return mPxMax;
}


Im2D_INT4 & cXml_TestDMP::ImCpt()
{
   return mImCpt;
}

const Im2D_INT4 & cXml_TestDMP::ImCpt()const 
{
   return mImCpt;
}


Im2D_U_INT2 & cXml_TestDMP::DataIm()
{
   return mDataIm;
}

const Im2D_U_INT2 & cXml_TestDMP::DataIm()const 
{
   return mDataIm;
}


double & cXml_TestDMP::StepPx()
{
   return mStepPx;
}

const double & cXml_TestDMP::StepPx()const 
{
   return mStepPx;
}


double & cXml_TestDMP::DynPx()
{
   return mDynPx;
}

const double & cXml_TestDMP::DynPx()const 
{
   return mDynPx;
}

void  BinaryUnDumpFromFile(cXml_TestDMP & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.PxMin(),aFp);
    BinaryUnDumpFromFile(anObj.PxMax(),aFp);
    BinaryUnDumpFromFile(anObj.ImCpt(),aFp);
    BinaryUnDumpFromFile(anObj.DataIm(),aFp);
    BinaryUnDumpFromFile(anObj.StepPx(),aFp);
    BinaryUnDumpFromFile(anObj.DynPx(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cXml_TestDMP & anObj)
{
    BinaryDumpInFile(aFp,anObj.PxMin());
    BinaryDumpInFile(aFp,anObj.PxMax());
    BinaryDumpInFile(aFp,anObj.ImCpt());
    BinaryDumpInFile(aFp,anObj.DataIm());
    BinaryDumpInFile(aFp,anObj.StepPx());
    BinaryDumpInFile(aFp,anObj.DynPx());
}

cElXMLTree * ToXMLTree(const cXml_TestDMP & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"Xml_TestDMP",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("PxMin"),anObj.PxMin())->ReTagThis("PxMin"));
   aRes->AddFils(::ToXMLTree(std::string("PxMax"),anObj.PxMax())->ReTagThis("PxMax"));
   aRes->AddFils(::ToXMLTree(std::string("ImCpt"),anObj.ImCpt())->ReTagThis("ImCpt"));
   aRes->AddFils(::ToXMLTree(std::string("DataIm"),anObj.DataIm())->ReTagThis("DataIm"));
   aRes->AddFils(::ToXMLTree(std::string("StepPx"),anObj.StepPx())->ReTagThis("StepPx"));
   aRes->AddFils(::ToXMLTree(std::string("DynPx"),anObj.DynPx())->ReTagThis("DynPx"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cXml_TestDMP & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.PxMin(),aTree->Get("PxMin",1)); //tototo 

   xml_init(anObj.PxMax(),aTree->Get("PxMax",1)); //tototo 

   xml_init(anObj.ImCpt(),aTree->Get("ImCpt",1)); //tototo 

   xml_init(anObj.DataIm(),aTree->Get("DataIm",1)); //tototo 

   xml_init(anObj.StepPx(),aTree->Get("StepPx",1)); //tototo 

   xml_init(anObj.DynPx(),aTree->Get("DynPx",1)); //tototo 
}

std::string  Mangling( cXml_TestDMP *) {return "88E38D9DFC2D7CCCFB3F";};


Im2D_INT1 & cOneInvRad::ImRad()
{
   return mImRad;
}

const Im2D_INT1 & cOneInvRad::ImRad()const 
{
   return mImRad;
}


Im2D_U_INT2 & cOneInvRad::CodeBinaire()
{
   return mCodeBinaire;
}

const Im2D_U_INT2 & cOneInvRad::CodeBinaire()const 
{
   return mCodeBinaire;
}

void  BinaryUnDumpFromFile(cOneInvRad & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.ImRad(),aFp);
    BinaryUnDumpFromFile(anObj.CodeBinaire(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cOneInvRad & anObj)
{
    BinaryDumpInFile(aFp,anObj.ImRad());
    BinaryDumpInFile(aFp,anObj.CodeBinaire());
}

cElXMLTree * ToXMLTree(const cOneInvRad & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"OneInvRad",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("ImRad"),anObj.ImRad())->ReTagThis("ImRad"));
   aRes->AddFils(::ToXMLTree(std::string("CodeBinaire"),anObj.CodeBinaire())->ReTagThis("CodeBinaire"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cOneInvRad & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.ImRad(),aTree->Get("ImRad",1)); //tototo 

   xml_init(anObj.CodeBinaire(),aTree->Get("CodeBinaire",1)); //tototo 
}

std::string  Mangling( cOneInvRad *) {return "36C1A9F58C648BD9FE3F";};


Im2D_INT1 & cProfilRad::ImProfil()
{
   return mImProfil;
}

const Im2D_INT1 & cProfilRad::ImProfil()const 
{
   return mImProfil;
}

void  BinaryUnDumpFromFile(cProfilRad & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.ImProfil(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cProfilRad & anObj)
{
    BinaryDumpInFile(aFp,anObj.ImProfil());
}

cElXMLTree * ToXMLTree(const cProfilRad & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ProfilRad",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("ImProfil"),anObj.ImProfil())->ReTagThis("ImProfil"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cProfilRad & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.ImProfil(),aTree->Get("ImProfil",1)); //tototo 
}

std::string  Mangling( cProfilRad *) {return "5CAC442132843DCCFD3F";};


Im2D_INT1 & cRotInvarAutoCor::IR0()
{
   return mIR0;
}

const Im2D_INT1 & cRotInvarAutoCor::IR0()const 
{
   return mIR0;
}


Im2D_INT1 & cRotInvarAutoCor::IGT()
{
   return mIGT;
}

const Im2D_INT1 & cRotInvarAutoCor::IGT()const 
{
   return mIGT;
}


Im2D_INT1 & cRotInvarAutoCor::IGR()
{
   return mIGR;
}

const Im2D_INT1 & cRotInvarAutoCor::IGR()const 
{
   return mIGR;
}

void  BinaryUnDumpFromFile(cRotInvarAutoCor & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.IR0(),aFp);
    BinaryUnDumpFromFile(anObj.IGT(),aFp);
    BinaryUnDumpFromFile(anObj.IGR(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cRotInvarAutoCor & anObj)
{
    BinaryDumpInFile(aFp,anObj.IR0());
    BinaryDumpInFile(aFp,anObj.IGT());
    BinaryDumpInFile(aFp,anObj.IGR());
}

cElXMLTree * ToXMLTree(const cRotInvarAutoCor & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"RotInvarAutoCor",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("IR0"),anObj.IR0())->ReTagThis("IR0"));
   aRes->AddFils(::ToXMLTree(std::string("IGT"),anObj.IGT())->ReTagThis("IGT"));
   aRes->AddFils(::ToXMLTree(std::string("IGR"),anObj.IGR())->ReTagThis("IGR"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cRotInvarAutoCor & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.IR0(),aTree->Get("IR0",1)); //tototo 

   xml_init(anObj.IGT(),aTree->Get("IGT",1)); //tototo 

   xml_init(anObj.IGR(),aTree->Get("IGR",1)); //tototo 
}

std::string  Mangling( cRotInvarAutoCor *) {return "7CDC9DE2D1D58BC0FE3F";};


eTypePtRemark & cOnePCarac::Kind()
{
   return mKind;
}

const eTypePtRemark & cOnePCarac::Kind()const 
{
   return mKind;
}


Pt2dr & cOnePCarac::Pt()
{
   return mPt;
}

const Pt2dr & cOnePCarac::Pt()const 
{
   return mPt;
}


Pt2dr & cOnePCarac::Pt0()
{
   return mPt0;
}

const Pt2dr & cOnePCarac::Pt0()const 
{
   return mPt0;
}


int & cOnePCarac::NivScale()
{
   return mNivScale;
}

const int & cOnePCarac::NivScale()const 
{
   return mNivScale;
}


double & cOnePCarac::Scale()
{
   return mScale;
}

const double & cOnePCarac::Scale()const 
{
   return mScale;
}


double & cOnePCarac::ScaleStab()
{
   return mScaleStab;
}

const double & cOnePCarac::ScaleStab()const 
{
   return mScaleStab;
}


double & cOnePCarac::ScaleNature()
{
   return mScaleNature;
}

const double & cOnePCarac::ScaleNature()const 
{
   return mScaleNature;
}


Pt2dr & cOnePCarac::DirMS()
{
   return mDirMS;
}

const Pt2dr & cOnePCarac::DirMS()const 
{
   return mDirMS;
}


Pt2dr & cOnePCarac::DirAC()
{
   return mDirAC;
}

const Pt2dr & cOnePCarac::DirAC()const 
{
   return mDirAC;
}


double & cOnePCarac::Contraste()
{
   return mContraste;
}

const double & cOnePCarac::Contraste()const 
{
   return mContraste;
}


double & cOnePCarac::ContrasteRel()
{
   return mContrasteRel;
}

const double & cOnePCarac::ContrasteRel()const 
{
   return mContrasteRel;
}


double & cOnePCarac::AutoCorrel()
{
   return mAutoCorrel;
}

const double & cOnePCarac::AutoCorrel()const 
{
   return mAutoCorrel;
}


bool & cOnePCarac::OK()
{
   return mOK;
}

const bool & cOnePCarac::OK()const 
{
   return mOK;
}


cOneInvRad & cOnePCarac::InvR()
{
   return mInvR;
}

const cOneInvRad & cOnePCarac::InvR()const 
{
   return mInvR;
}


double & cOnePCarac::MoyLP()
{
   return mMoyLP;
}

const double & cOnePCarac::MoyLP()const 
{
   return mMoyLP;
}


Im2D_INT1 & cOnePCarac::ImLogPol()
{
   return mImLogPol;
}

const Im2D_INT1 & cOnePCarac::ImLogPol()const 
{
   return mImLogPol;
}


std::vector<double> & cOnePCarac::VectRho()
{
   return mVectRho;
}

const std::vector<double> & cOnePCarac::VectRho()const 
{
   return mVectRho;
}


cProfilRad & cOnePCarac::ProfR()
{
   return mProfR;
}

const cProfilRad & cOnePCarac::ProfR()const 
{
   return mProfR;
}


cRotInvarAutoCor & cOnePCarac::RIAC()
{
   return mRIAC;
}

const cRotInvarAutoCor & cOnePCarac::RIAC()const 
{
   return mRIAC;
}


int & cOnePCarac::Id()
{
   return mId;
}

const int & cOnePCarac::Id()const 
{
   return mId;
}


int & cOnePCarac::HeapInd()
{
   return mHeapInd;
}

const int & cOnePCarac::HeapInd()const 
{
   return mHeapInd;
}


double & cOnePCarac::Prio()
{
   return mPrio;
}

const double & cOnePCarac::Prio()const 
{
   return mPrio;
}

void  BinaryUnDumpFromFile(cOnePCarac & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Kind(),aFp);
    BinaryUnDumpFromFile(anObj.Pt(),aFp);
    BinaryUnDumpFromFile(anObj.Pt0(),aFp);
    BinaryUnDumpFromFile(anObj.NivScale(),aFp);
    BinaryUnDumpFromFile(anObj.Scale(),aFp);
    BinaryUnDumpFromFile(anObj.ScaleStab(),aFp);
    BinaryUnDumpFromFile(anObj.ScaleNature(),aFp);
    BinaryUnDumpFromFile(anObj.DirMS(),aFp);
    BinaryUnDumpFromFile(anObj.DirAC(),aFp);
    BinaryUnDumpFromFile(anObj.Contraste(),aFp);
    BinaryUnDumpFromFile(anObj.ContrasteRel(),aFp);
    BinaryUnDumpFromFile(anObj.AutoCorrel(),aFp);
    BinaryUnDumpFromFile(anObj.OK(),aFp);
    BinaryUnDumpFromFile(anObj.InvR(),aFp);
    BinaryUnDumpFromFile(anObj.MoyLP(),aFp);
    BinaryUnDumpFromFile(anObj.ImLogPol(),aFp);
    BinaryUnDumpFromFile(anObj.VectRho(),aFp);
    BinaryUnDumpFromFile(anObj.ProfR(),aFp);
    BinaryUnDumpFromFile(anObj.RIAC(),aFp);
    BinaryUnDumpFromFile(anObj.Id(),aFp);
    BinaryUnDumpFromFile(anObj.HeapInd(),aFp);
    BinaryUnDumpFromFile(anObj.Prio(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cOnePCarac & anObj)
{
    BinaryDumpInFile(aFp,anObj.Kind());
    BinaryDumpInFile(aFp,anObj.Pt());
    BinaryDumpInFile(aFp,anObj.Pt0());
    BinaryDumpInFile(aFp,anObj.NivScale());
    BinaryDumpInFile(aFp,anObj.Scale());
    BinaryDumpInFile(aFp,anObj.ScaleStab());
    BinaryDumpInFile(aFp,anObj.ScaleNature());
    BinaryDumpInFile(aFp,anObj.DirMS());
    BinaryDumpInFile(aFp,anObj.DirAC());
    BinaryDumpInFile(aFp,anObj.Contraste());
    BinaryDumpInFile(aFp,anObj.ContrasteRel());
    BinaryDumpInFile(aFp,anObj.AutoCorrel());
    BinaryDumpInFile(aFp,anObj.OK());
    BinaryDumpInFile(aFp,anObj.InvR());
    BinaryDumpInFile(aFp,anObj.MoyLP());
    BinaryDumpInFile(aFp,anObj.ImLogPol());
    BinaryDumpInFile(aFp,anObj.VectRho());
    BinaryDumpInFile(aFp,anObj.ProfR());
    BinaryDumpInFile(aFp,anObj.RIAC());
    BinaryDumpInFile(aFp,anObj.Id());
    BinaryDumpInFile(aFp,anObj.HeapInd());
    BinaryDumpInFile(aFp,anObj.Prio());
}

cElXMLTree * ToXMLTree(const cOnePCarac & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"OnePCarac",eXMLBranche);
   aRes->AddFils(ToXMLTree(std::string("Kind"),anObj.Kind())->ReTagThis("Kind"));
   aRes->AddFils(::ToXMLTree(std::string("Pt"),anObj.Pt())->ReTagThis("Pt"));
   aRes->AddFils(::ToXMLTree(std::string("Pt0"),anObj.Pt0())->ReTagThis("Pt0"));
   aRes->AddFils(::ToXMLTree(std::string("NivScale"),anObj.NivScale())->ReTagThis("NivScale"));
   aRes->AddFils(::ToXMLTree(std::string("Scale"),anObj.Scale())->ReTagThis("Scale"));
   aRes->AddFils(::ToXMLTree(std::string("ScaleStab"),anObj.ScaleStab())->ReTagThis("ScaleStab"));
   aRes->AddFils(::ToXMLTree(std::string("ScaleNature"),anObj.ScaleNature())->ReTagThis("ScaleNature"));
   aRes->AddFils(::ToXMLTree(std::string("DirMS"),anObj.DirMS())->ReTagThis("DirMS"));
   aRes->AddFils(::ToXMLTree(std::string("DirAC"),anObj.DirAC())->ReTagThis("DirAC"));
   aRes->AddFils(::ToXMLTree(std::string("Contraste"),anObj.Contraste())->ReTagThis("Contraste"));
   aRes->AddFils(::ToXMLTree(std::string("ContrasteRel"),anObj.ContrasteRel())->ReTagThis("ContrasteRel"));
   aRes->AddFils(::ToXMLTree(std::string("AutoCorrel"),anObj.AutoCorrel())->ReTagThis("AutoCorrel"));
   aRes->AddFils(::ToXMLTree(std::string("OK"),anObj.OK())->ReTagThis("OK"));
   aRes->AddFils(ToXMLTree(anObj.InvR())->ReTagThis("InvR"));
   aRes->AddFils(::ToXMLTree(std::string("MoyLP"),anObj.MoyLP())->ReTagThis("MoyLP"));
   aRes->AddFils(::ToXMLTree(std::string("ImLogPol"),anObj.ImLogPol())->ReTagThis("ImLogPol"));
   aRes->AddFils(::ToXMLTree(std::string("VectRho"),anObj.VectRho())->ReTagThis("VectRho"));
   aRes->AddFils(ToXMLTree(anObj.ProfR())->ReTagThis("ProfR"));
   aRes->AddFils(ToXMLTree(anObj.RIAC())->ReTagThis("RIAC"));
   aRes->AddFils(::ToXMLTree(std::string("Id"),anObj.Id())->ReTagThis("Id"));
   aRes->AddFils(::ToXMLTree(std::string("HeapInd"),anObj.HeapInd())->ReTagThis("HeapInd"));
   aRes->AddFils(::ToXMLTree(std::string("Prio"),anObj.Prio())->ReTagThis("Prio"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cOnePCarac & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.Kind(),aTree->Get("Kind",1)); //tototo 

   xml_init(anObj.Pt(),aTree->Get("Pt",1)); //tototo 

   xml_init(anObj.Pt0(),aTree->Get("Pt0",1)); //tototo 

   xml_init(anObj.NivScale(),aTree->Get("NivScale",1)); //tototo 

   xml_init(anObj.Scale(),aTree->Get("Scale",1)); //tototo 

   xml_init(anObj.ScaleStab(),aTree->Get("ScaleStab",1)); //tototo 

   xml_init(anObj.ScaleNature(),aTree->Get("ScaleNature",1)); //tototo 

   xml_init(anObj.DirMS(),aTree->Get("DirMS",1)); //tototo 

   xml_init(anObj.DirAC(),aTree->Get("DirAC",1)); //tototo 

   xml_init(anObj.Contraste(),aTree->Get("Contraste",1)); //tototo 

   xml_init(anObj.ContrasteRel(),aTree->Get("ContrasteRel",1)); //tototo 

   xml_init(anObj.AutoCorrel(),aTree->Get("AutoCorrel",1)); //tototo 

   xml_init(anObj.OK(),aTree->Get("OK",1)); //tototo 

   xml_init(anObj.InvR(),aTree->Get("InvR",1)); //tototo 

   xml_init(anObj.MoyLP(),aTree->Get("MoyLP",1)); //tototo 

   xml_init(anObj.ImLogPol(),aTree->Get("ImLogPol",1)); //tototo 

   xml_init(anObj.VectRho(),aTree->Get("VectRho",1)); //tototo 

   xml_init(anObj.ProfR(),aTree->Get("ProfR",1)); //tototo 

   xml_init(anObj.RIAC(),aTree->Get("RIAC",1)); //tototo 

   xml_init(anObj.Id(),aTree->Get("Id",1)); //tototo 

   xml_init(anObj.HeapInd(),aTree->Get("HeapInd",1)); //tototo 

   xml_init(anObj.Prio(),aTree->Get("Prio",1)); //tototo 
}

std::string  Mangling( cOnePCarac *) {return "62E799CDEDB5DBF1FE3F";};


std::vector< cOnePCarac > & cSetPCarac::OnePCarac()
{
   return mOnePCarac;
}

const std::vector< cOnePCarac > & cSetPCarac::OnePCarac()const 
{
   return mOnePCarac;
}

void  BinaryUnDumpFromFile(cSetPCarac & anObj,ELISE_fp & aFp)
{
   { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cOnePCarac aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.OnePCarac().push_back(aVal);
        }
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cSetPCarac & anObj)
{
    BinaryDumpInFile(aFp,(int)anObj.OnePCarac().size());
    for(  std::vector< cOnePCarac >::const_iterator iT=anObj.OnePCarac().begin();
         iT!=anObj.OnePCarac().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
}

cElXMLTree * ToXMLTree(const cSetPCarac & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"SetPCarac",eXMLBranche);
  for
  (       std::vector< cOnePCarac >::const_iterator it=anObj.OnePCarac().begin();
      it !=anObj.OnePCarac().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("OnePCarac"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cSetPCarac & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.OnePCarac(),aTree->GetAll("OnePCarac",false,1));
}

std::string  Mangling( cSetPCarac *) {return "1279FC99F3A2158DFF3F";};


cOnePCarac & cSRPC_Truth::P1()
{
   return mP1;
}

const cOnePCarac & cSRPC_Truth::P1()const 
{
   return mP1;
}


cOnePCarac & cSRPC_Truth::P2()
{
   return mP2;
}

const cOnePCarac & cSRPC_Truth::P2()const 
{
   return mP2;
}

void  BinaryUnDumpFromFile(cSRPC_Truth & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.P1(),aFp);
    BinaryUnDumpFromFile(anObj.P2(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cSRPC_Truth & anObj)
{
    BinaryDumpInFile(aFp,anObj.P1());
    BinaryDumpInFile(aFp,anObj.P2());
}

cElXMLTree * ToXMLTree(const cSRPC_Truth & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"SRPC_Truth",eXMLBranche);
   aRes->AddFils(ToXMLTree(anObj.P1())->ReTagThis("P1"));
   aRes->AddFils(ToXMLTree(anObj.P2())->ReTagThis("P2"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cSRPC_Truth & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.P1(),aTree->Get("P1",1)); //tototo 

   xml_init(anObj.P2(),aTree->Get("P2",1)); //tototo 
}

std::string  Mangling( cSRPC_Truth *) {return "99142551563FB3C8FEBF";};


std::vector< cSRPC_Truth > & cSetRefPCarac::SRPC_Truth()
{
   return mSRPC_Truth;
}

const std::vector< cSRPC_Truth > & cSetRefPCarac::SRPC_Truth()const 
{
   return mSRPC_Truth;
}


std::vector< cOnePCarac > & cSetRefPCarac::SRPC_Rand()
{
   return mSRPC_Rand;
}

const std::vector< cOnePCarac > & cSetRefPCarac::SRPC_Rand()const 
{
   return mSRPC_Rand;
}

void  BinaryUnDumpFromFile(cSetRefPCarac & anObj,ELISE_fp & aFp)
{
   { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cSRPC_Truth aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.SRPC_Truth().push_back(aVal);
        }
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cOnePCarac aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.SRPC_Rand().push_back(aVal);
        }
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cSetRefPCarac & anObj)
{
    BinaryDumpInFile(aFp,(int)anObj.SRPC_Truth().size());
    for(  std::vector< cSRPC_Truth >::const_iterator iT=anObj.SRPC_Truth().begin();
         iT!=anObj.SRPC_Truth().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,(int)anObj.SRPC_Rand().size());
    for(  std::vector< cOnePCarac >::const_iterator iT=anObj.SRPC_Rand().begin();
         iT!=anObj.SRPC_Rand().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
}

cElXMLTree * ToXMLTree(const cSetRefPCarac & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"SetRefPCarac",eXMLBranche);
  for
  (       std::vector< cSRPC_Truth >::const_iterator it=anObj.SRPC_Truth().begin();
      it !=anObj.SRPC_Truth().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("SRPC_Truth"));
  for
  (       std::vector< cOnePCarac >::const_iterator it=anObj.SRPC_Rand().begin();
      it !=anObj.SRPC_Rand().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("SRPC_Rand"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cSetRefPCarac & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.SRPC_Truth(),aTree->GetAll("SRPC_Truth",false,1));

   xml_init(anObj.SRPC_Rand(),aTree->GetAll("SRPC_Rand",false,1));
}

std::string  Mangling( cSetRefPCarac *) {return "E29DDECC67C470BCFE3F";};


std::vector<double> & cCBOneBit::Coeff()
{
   return mCoeff;
}

const std::vector<double> & cCBOneBit::Coeff()const 
{
   return mCoeff;
}


std::vector<int> & cCBOneBit::IndInV()
{
   return mIndInV;
}

const std::vector<int> & cCBOneBit::IndInV()const 
{
   return mIndInV;
}


int & cCBOneBit::IndBit()
{
   return mIndBit;
}

const int & cCBOneBit::IndBit()const 
{
   return mIndBit;
}

void  BinaryUnDumpFromFile(cCBOneBit & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Coeff(),aFp);
    BinaryUnDumpFromFile(anObj.IndInV(),aFp);
    BinaryUnDumpFromFile(anObj.IndBit(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cCBOneBit & anObj)
{
    BinaryDumpInFile(aFp,anObj.Coeff());
    BinaryDumpInFile(aFp,anObj.IndInV());
    BinaryDumpInFile(aFp,anObj.IndBit());
}

cElXMLTree * ToXMLTree(const cCBOneBit & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"CBOneBit",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Coeff"),anObj.Coeff())->ReTagThis("Coeff"));
   aRes->AddFils(::ToXMLTree(std::string("IndInV"),anObj.IndInV())->ReTagThis("IndInV"));
   aRes->AddFils(::ToXMLTree(std::string("IndBit"),anObj.IndBit())->ReTagThis("IndBit"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cCBOneBit & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.Coeff(),aTree->Get("Coeff",1)); //tototo 

   xml_init(anObj.IndInV(),aTree->Get("IndInV",1)); //tototo 

   xml_init(anObj.IndBit(),aTree->Get("IndBit",1)); //tototo 
}

std::string  Mangling( cCBOneBit *) {return "2EB5B20E63A94F8EFE3F";};


int & cCBOneVect::IndVec()
{
   return mIndVec;
}

const int & cCBOneVect::IndVec()const 
{
   return mIndVec;
}


std::vector< cCBOneBit > & cCBOneVect::CBOneBit()
{
   return mCBOneBit;
}

const std::vector< cCBOneBit > & cCBOneVect::CBOneBit()const 
{
   return mCBOneBit;
}

void  BinaryUnDumpFromFile(cCBOneVect & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.IndVec(),aFp);
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cCBOneBit aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.CBOneBit().push_back(aVal);
        }
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cCBOneVect & anObj)
{
    BinaryDumpInFile(aFp,anObj.IndVec());
    BinaryDumpInFile(aFp,(int)anObj.CBOneBit().size());
    for(  std::vector< cCBOneBit >::const_iterator iT=anObj.CBOneBit().begin();
         iT!=anObj.CBOneBit().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
}

cElXMLTree * ToXMLTree(const cCBOneVect & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"CBOneVect",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("IndVec"),anObj.IndVec())->ReTagThis("IndVec"));
  for
  (       std::vector< cCBOneBit >::const_iterator it=anObj.CBOneBit().begin();
      it !=anObj.CBOneBit().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("CBOneBit"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cCBOneVect & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.IndVec(),aTree->Get("IndVec",1)); //tototo 

   xml_init(anObj.CBOneBit(),aTree->GetAll("CBOneBit",false,1));
}

std::string  Mangling( cCBOneVect *) {return "162A2F0FD2E22AFCFD3F";};


std::vector< cCBOneVect > & cFullParamCB::CBOneVect()
{
   return mCBOneVect;
}

const std::vector< cCBOneVect > & cFullParamCB::CBOneVect()const 
{
   return mCBOneVect;
}

void  BinaryUnDumpFromFile(cFullParamCB & anObj,ELISE_fp & aFp)
{
   { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cCBOneVect aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.CBOneVect().push_back(aVal);
        }
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cFullParamCB & anObj)
{
    BinaryDumpInFile(aFp,(int)anObj.CBOneVect().size());
    for(  std::vector< cCBOneVect >::const_iterator iT=anObj.CBOneVect().begin();
         iT!=anObj.CBOneVect().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
}

cElXMLTree * ToXMLTree(const cFullParamCB & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"FullParamCB",eXMLBranche);
  for
  (       std::vector< cCBOneVect >::const_iterator it=anObj.CBOneVect().begin();
      it !=anObj.CBOneVect().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("CBOneVect"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cFullParamCB & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.CBOneVect(),aTree->GetAll("CBOneVect",false,1));
}

std::string  Mangling( cFullParamCB *) {return "C623DB904AAFA1A5FE3F";};


std::vector<double> & cCompCBOneBit::Coeff()
{
   return mCoeff;
}

const std::vector<double> & cCompCBOneBit::Coeff()const 
{
   return mCoeff;
}


std::vector<int> & cCompCBOneBit::IndX()
{
   return mIndX;
}

const std::vector<int> & cCompCBOneBit::IndX()const 
{
   return mIndX;
}


std::vector<int> & cCompCBOneBit::IndY()
{
   return mIndY;
}

const std::vector<int> & cCompCBOneBit::IndY()const 
{
   return mIndY;
}


int & cCompCBOneBit::IndBit()
{
   return mIndBit;
}

const int & cCompCBOneBit::IndBit()const 
{
   return mIndBit;
}

void  BinaryUnDumpFromFile(cCompCBOneBit & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Coeff(),aFp);
    BinaryUnDumpFromFile(anObj.IndX(),aFp);
    BinaryUnDumpFromFile(anObj.IndY(),aFp);
    BinaryUnDumpFromFile(anObj.IndBit(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cCompCBOneBit & anObj)
{
    BinaryDumpInFile(aFp,anObj.Coeff());
    BinaryDumpInFile(aFp,anObj.IndX());
    BinaryDumpInFile(aFp,anObj.IndY());
    BinaryDumpInFile(aFp,anObj.IndBit());
}

cElXMLTree * ToXMLTree(const cCompCBOneBit & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"CompCBOneBit",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Coeff"),anObj.Coeff())->ReTagThis("Coeff"));
   aRes->AddFils(::ToXMLTree(std::string("IndX"),anObj.IndX())->ReTagThis("IndX"));
   aRes->AddFils(::ToXMLTree(std::string("IndY"),anObj.IndY())->ReTagThis("IndY"));
   aRes->AddFils(::ToXMLTree(std::string("IndBit"),anObj.IndBit())->ReTagThis("IndBit"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cCompCBOneBit & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.Coeff(),aTree->Get("Coeff",1)); //tototo 

   xml_init(anObj.IndX(),aTree->Get("IndX",1)); //tototo 

   xml_init(anObj.IndY(),aTree->Get("IndY",1)); //tototo 

   xml_init(anObj.IndBit(),aTree->Get("IndBit",1)); //tototo 
}

std::string  Mangling( cCompCBOneBit *) {return "1A4BB62C212BD492FE3F";};


int & cCompCB::BitThresh()
{
   return mBitThresh;
}

const int & cCompCB::BitThresh()const 
{
   return mBitThresh;
}


std::vector< cCompCBOneBit > & cCompCB::CompCBOneBit()
{
   return mCompCBOneBit;
}

const std::vector< cCompCBOneBit > & cCompCB::CompCBOneBit()const 
{
   return mCompCBOneBit;
}

void  BinaryUnDumpFromFile(cCompCB & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.BitThresh(),aFp);
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cCompCBOneBit aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.CompCBOneBit().push_back(aVal);
        }
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cCompCB & anObj)
{
    BinaryDumpInFile(aFp,anObj.BitThresh());
    BinaryDumpInFile(aFp,(int)anObj.CompCBOneBit().size());
    for(  std::vector< cCompCBOneBit >::const_iterator iT=anObj.CompCBOneBit().begin();
         iT!=anObj.CompCBOneBit().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
}

cElXMLTree * ToXMLTree(const cCompCB & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"CompCB",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("BitThresh"),anObj.BitThresh())->ReTagThis("BitThresh"));
  for
  (       std::vector< cCompCBOneBit >::const_iterator it=anObj.CompCBOneBit().begin();
      it !=anObj.CompCBOneBit().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("CompCBOneBit"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cCompCB & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.BitThresh(),aTree->Get("BitThresh",1)); //tototo 

   xml_init(anObj.CompCBOneBit(),aTree->GetAll("CompCBOneBit",false,1));
}

std::string  Mangling( cCompCB *) {return "62F1F7F6FA6155DCFDBF";};


std::string & cFitsOneBin::PrefName()
{
   return mPrefName;
}

const std::string & cFitsOneBin::PrefName()const 
{
   return mPrefName;
}


cTplValGesInit< std::string > & cFitsOneBin::PostName()
{
   return mPostName;
}

const cTplValGesInit< std::string > & cFitsOneBin::PostName()const 
{
   return mPostName;
}


cTplValGesInit< cCompCB > & cFitsOneBin::CCB()
{
   return mCCB;
}

const cTplValGesInit< cCompCB > & cFitsOneBin::CCB()const 
{
   return mCCB;
}

void  BinaryUnDumpFromFile(cFitsOneBin & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.PrefName(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.PostName().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.PostName().ValForcedForUnUmp(),aFp);
        }
        else  anObj.PostName().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.CCB().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.CCB().ValForcedForUnUmp(),aFp);
        }
        else  anObj.CCB().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cFitsOneBin & anObj)
{
    BinaryDumpInFile(aFp,anObj.PrefName());
    BinaryDumpInFile(aFp,anObj.PostName().IsInit());
    if (anObj.PostName().IsInit()) BinaryDumpInFile(aFp,anObj.PostName().Val());
    BinaryDumpInFile(aFp,anObj.CCB().IsInit());
    if (anObj.CCB().IsInit()) BinaryDumpInFile(aFp,anObj.CCB().Val());
}

cElXMLTree * ToXMLTree(const cFitsOneBin & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"FitsOneBin",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("PrefName"),anObj.PrefName())->ReTagThis("PrefName"));
   if (anObj.PostName().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PostName"),anObj.PostName().Val())->ReTagThis("PostName"));
   if (anObj.CCB().IsInit())
      aRes->AddFils(ToXMLTree(anObj.CCB().Val())->ReTagThis("CCB"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cFitsOneBin & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.PrefName(),aTree->Get("PrefName",1)); //tototo 

   xml_init(anObj.PostName(),aTree->Get("PostName",1),std::string("_Local.xml")); //tototo 

   xml_init(anObj.CCB(),aTree->Get("CCB",1)); //tototo 
}

std::string  Mangling( cFitsOneBin *) {return "8682B7876A8885EBFE3F";};


eTypePtRemark & cFitsOneLabel::KindOf()
{
   return mKindOf;
}

const eTypePtRemark & cFitsOneLabel::KindOf()const 
{
   return mKindOf;
}


cFitsOneBin & cFitsOneLabel::BinIndexed()
{
   return mBinIndexed;
}

const cFitsOneBin & cFitsOneLabel::BinIndexed()const 
{
   return mBinIndexed;
}


cFitsOneBin & cFitsOneLabel::BinDecisionShort()
{
   return mBinDecisionShort;
}

const cFitsOneBin & cFitsOneLabel::BinDecisionShort()const 
{
   return mBinDecisionShort;
}


cFitsOneBin & cFitsOneLabel::BinDecisionLong()
{
   return mBinDecisionLong;
}

const cFitsOneBin & cFitsOneLabel::BinDecisionLong()const 
{
   return mBinDecisionLong;
}

void  BinaryUnDumpFromFile(cFitsOneLabel & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.KindOf(),aFp);
    BinaryUnDumpFromFile(anObj.BinIndexed(),aFp);
    BinaryUnDumpFromFile(anObj.BinDecisionShort(),aFp);
    BinaryUnDumpFromFile(anObj.BinDecisionLong(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cFitsOneLabel & anObj)
{
    BinaryDumpInFile(aFp,anObj.KindOf());
    BinaryDumpInFile(aFp,anObj.BinIndexed());
    BinaryDumpInFile(aFp,anObj.BinDecisionShort());
    BinaryDumpInFile(aFp,anObj.BinDecisionLong());
}

cElXMLTree * ToXMLTree(const cFitsOneLabel & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"FitsOneLabel",eXMLBranche);
   aRes->AddFils(ToXMLTree(std::string("KindOf"),anObj.KindOf())->ReTagThis("KindOf"));
   aRes->AddFils(ToXMLTree(anObj.BinIndexed())->ReTagThis("BinIndexed"));
   aRes->AddFils(ToXMLTree(anObj.BinDecisionShort())->ReTagThis("BinDecisionShort"));
   aRes->AddFils(ToXMLTree(anObj.BinDecisionLong())->ReTagThis("BinDecisionLong"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cFitsOneLabel & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.KindOf(),aTree->Get("KindOf",1)); //tototo 

   xml_init(anObj.BinIndexed(),aTree->Get("BinIndexed",1)); //tototo 

   xml_init(anObj.BinDecisionShort(),aTree->Get("BinDecisionShort",1)); //tototo 

   xml_init(anObj.BinDecisionLong(),aTree->Get("BinDecisionLong",1)); //tototo 
}

std::string  Mangling( cFitsOneLabel *) {return "9B1A97586663DCA2FF3F";};


cTplValGesInit< double > & cSeuilFitsParam::SeuilCorrDR()
{
   return mSeuilCorrDR;
}

const cTplValGesInit< double > & cSeuilFitsParam::SeuilCorrDR()const 
{
   return mSeuilCorrDR;
}


cTplValGesInit< double > & cSeuilFitsParam::SeuilInc()
{
   return mSeuilInc;
}

const cTplValGesInit< double > & cSeuilFitsParam::SeuilInc()const 
{
   return mSeuilInc;
}


cTplValGesInit< double > & cSeuilFitsParam::SeuilCorrLP()
{
   return mSeuilCorrLP;
}

const cTplValGesInit< double > & cSeuilFitsParam::SeuilCorrLP()const 
{
   return mSeuilCorrLP;
}


cTplValGesInit< double > & cSeuilFitsParam::ExposantPdsDistGrad()
{
   return mExposantPdsDistGrad;
}

const cTplValGesInit< double > & cSeuilFitsParam::ExposantPdsDistGrad()const 
{
   return mExposantPdsDistGrad;
}


cTplValGesInit< double > & cSeuilFitsParam::SeuilDistGrad()
{
   return mSeuilDistGrad;
}

const cTplValGesInit< double > & cSeuilFitsParam::SeuilDistGrad()const 
{
   return mSeuilDistGrad;
}


cTplValGesInit< double > & cSeuilFitsParam::SeuilCorrelRatio12()
{
   return mSeuilCorrelRatio12;
}

const cTplValGesInit< double > & cSeuilFitsParam::SeuilCorrelRatio12()const 
{
   return mSeuilCorrelRatio12;
}


cTplValGesInit< double > & cSeuilFitsParam::SeuilGradRatio12()
{
   return mSeuilGradRatio12;
}

const cTplValGesInit< double > & cSeuilFitsParam::SeuilGradRatio12()const 
{
   return mSeuilGradRatio12;
}

void  BinaryUnDumpFromFile(cSeuilFitsParam & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SeuilCorrDR().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SeuilCorrDR().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SeuilCorrDR().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SeuilInc().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SeuilInc().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SeuilInc().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SeuilCorrLP().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SeuilCorrLP().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SeuilCorrLP().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ExposantPdsDistGrad().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ExposantPdsDistGrad().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ExposantPdsDistGrad().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SeuilDistGrad().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SeuilDistGrad().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SeuilDistGrad().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SeuilCorrelRatio12().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SeuilCorrelRatio12().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SeuilCorrelRatio12().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SeuilGradRatio12().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SeuilGradRatio12().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SeuilGradRatio12().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cSeuilFitsParam & anObj)
{
    BinaryDumpInFile(aFp,anObj.SeuilCorrDR().IsInit());
    if (anObj.SeuilCorrDR().IsInit()) BinaryDumpInFile(aFp,anObj.SeuilCorrDR().Val());
    BinaryDumpInFile(aFp,anObj.SeuilInc().IsInit());
    if (anObj.SeuilInc().IsInit()) BinaryDumpInFile(aFp,anObj.SeuilInc().Val());
    BinaryDumpInFile(aFp,anObj.SeuilCorrLP().IsInit());
    if (anObj.SeuilCorrLP().IsInit()) BinaryDumpInFile(aFp,anObj.SeuilCorrLP().Val());
    BinaryDumpInFile(aFp,anObj.ExposantPdsDistGrad().IsInit());
    if (anObj.ExposantPdsDistGrad().IsInit()) BinaryDumpInFile(aFp,anObj.ExposantPdsDistGrad().Val());
    BinaryDumpInFile(aFp,anObj.SeuilDistGrad().IsInit());
    if (anObj.SeuilDistGrad().IsInit()) BinaryDumpInFile(aFp,anObj.SeuilDistGrad().Val());
    BinaryDumpInFile(aFp,anObj.SeuilCorrelRatio12().IsInit());
    if (anObj.SeuilCorrelRatio12().IsInit()) BinaryDumpInFile(aFp,anObj.SeuilCorrelRatio12().Val());
    BinaryDumpInFile(aFp,anObj.SeuilGradRatio12().IsInit());
    if (anObj.SeuilGradRatio12().IsInit()) BinaryDumpInFile(aFp,anObj.SeuilGradRatio12().Val());
}

cElXMLTree * ToXMLTree(const cSeuilFitsParam & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"SeuilFitsParam",eXMLBranche);
   if (anObj.SeuilCorrDR().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SeuilCorrDR"),anObj.SeuilCorrDR().Val())->ReTagThis("SeuilCorrDR"));
   if (anObj.SeuilInc().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SeuilInc"),anObj.SeuilInc().Val())->ReTagThis("SeuilInc"));
   if (anObj.SeuilCorrLP().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SeuilCorrLP"),anObj.SeuilCorrLP().Val())->ReTagThis("SeuilCorrLP"));
   if (anObj.ExposantPdsDistGrad().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ExposantPdsDistGrad"),anObj.ExposantPdsDistGrad().Val())->ReTagThis("ExposantPdsDistGrad"));
   if (anObj.SeuilDistGrad().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SeuilDistGrad"),anObj.SeuilDistGrad().Val())->ReTagThis("SeuilDistGrad"));
   if (anObj.SeuilCorrelRatio12().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SeuilCorrelRatio12"),anObj.SeuilCorrelRatio12().Val())->ReTagThis("SeuilCorrelRatio12"));
   if (anObj.SeuilGradRatio12().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SeuilGradRatio12"),anObj.SeuilGradRatio12().Val())->ReTagThis("SeuilGradRatio12"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cSeuilFitsParam & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.SeuilCorrDR(),aTree->Get("SeuilCorrDR",1),double(0.7)); //tototo 

   xml_init(anObj.SeuilInc(),aTree->Get("SeuilInc",1),double(0.01)); //tototo 

   xml_init(anObj.SeuilCorrLP(),aTree->Get("SeuilCorrLP",1),double(0.93)); //tototo 

   xml_init(anObj.ExposantPdsDistGrad(),aTree->Get("ExposantPdsDistGrad",1),double(0.5)); //tototo 

   xml_init(anObj.SeuilDistGrad(),aTree->Get("SeuilDistGrad",1),double(0.5)); //tototo 

   xml_init(anObj.SeuilCorrelRatio12(),aTree->Get("SeuilCorrelRatio12",1),double(0.6)); //tototo 

   xml_init(anObj.SeuilGradRatio12(),aTree->Get("SeuilGradRatio12",1),double(0.6)); //tototo 
}

std::string  Mangling( cSeuilFitsParam *) {return "18B18CF1B04BFB82FE3F";};


cFitsOneLabel & cFitsParam::DefInit()
{
   return mDefInit;
}

const cFitsOneLabel & cFitsParam::DefInit()const 
{
   return mDefInit;
}


std::list< cFitsOneLabel > & cFitsParam::GenLabs()
{
   return mGenLabs;
}

const std::list< cFitsOneLabel > & cFitsParam::GenLabs()const 
{
   return mGenLabs;
}


cSeuilFitsParam & cFitsParam::SeuilOL()
{
   return mSeuilOL;
}

const cSeuilFitsParam & cFitsParam::SeuilOL()const 
{
   return mSeuilOL;
}


cSeuilFitsParam & cFitsParam::SeuilGen()
{
   return mSeuilGen;
}

const cSeuilFitsParam & cFitsParam::SeuilGen()const 
{
   return mSeuilGen;
}

void  BinaryUnDumpFromFile(cFitsParam & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.DefInit(),aFp);
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cFitsOneLabel aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.GenLabs().push_back(aVal);
        }
  } ;
    BinaryUnDumpFromFile(anObj.SeuilOL(),aFp);
    BinaryUnDumpFromFile(anObj.SeuilGen(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cFitsParam & anObj)
{
    BinaryDumpInFile(aFp,anObj.DefInit());
    BinaryDumpInFile(aFp,(int)anObj.GenLabs().size());
    for(  std::list< cFitsOneLabel >::const_iterator iT=anObj.GenLabs().begin();
         iT!=anObj.GenLabs().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.SeuilOL());
    BinaryDumpInFile(aFp,anObj.SeuilGen());
}

cElXMLTree * ToXMLTree(const cFitsParam & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"FitsParam",eXMLBranche);
   aRes->AddFils(ToXMLTree(anObj.DefInit())->ReTagThis("DefInit"));
  for
  (       std::list< cFitsOneLabel >::const_iterator it=anObj.GenLabs().begin();
      it !=anObj.GenLabs().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("GenLabs"));
   aRes->AddFils(ToXMLTree(anObj.SeuilOL())->ReTagThis("SeuilOL"));
   aRes->AddFils(ToXMLTree(anObj.SeuilGen())->ReTagThis("SeuilGen"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cFitsParam & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.DefInit(),aTree->Get("DefInit",1)); //tototo 

   xml_init(anObj.GenLabs(),aTree->GetAll("GenLabs",false,1));

   xml_init(anObj.SeuilOL(),aTree->Get("SeuilOL",1)); //tototo 

   xml_init(anObj.SeuilGen(),aTree->Get("SeuilGen",1)); //tototo 
}

std::string  Mangling( cFitsParam *) {return "AABB438DC4A03EC3FD3F";};


std::string & cXAPA_OneMatch::Master()
{
   return mMaster;
}

const std::string & cXAPA_OneMatch::Master()const 
{
   return mMaster;
}


std::string & cXAPA_OneMatch::Pattern()
{
   return mPattern;
}

const std::string & cXAPA_OneMatch::Pattern()const 
{
   return mPattern;
}


std::string & cXAPA_OneMatch::PatternRef()
{
   return mPatternRef;
}

const std::string & cXAPA_OneMatch::PatternRef()const 
{
   return mPatternRef;
}

void  BinaryUnDumpFromFile(cXAPA_OneMatch & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Master(),aFp);
    BinaryUnDumpFromFile(anObj.Pattern(),aFp);
    BinaryUnDumpFromFile(anObj.PatternRef(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cXAPA_OneMatch & anObj)
{
    BinaryDumpInFile(aFp,anObj.Master());
    BinaryDumpInFile(aFp,anObj.Pattern());
    BinaryDumpInFile(aFp,anObj.PatternRef());
}

cElXMLTree * ToXMLTree(const cXAPA_OneMatch & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"XAPA_OneMatch",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Master"),anObj.Master())->ReTagThis("Master"));
   aRes->AddFils(::ToXMLTree(std::string("Pattern"),anObj.Pattern())->ReTagThis("Pattern"));
   aRes->AddFils(::ToXMLTree(std::string("PatternRef"),anObj.PatternRef())->ReTagThis("PatternRef"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cXAPA_OneMatch & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.Master(),aTree->Get("Master",1)); //tototo 

   xml_init(anObj.Pattern(),aTree->Get("Pattern",1)); //tototo 

   xml_init(anObj.PatternRef(),aTree->Get("PatternRef",1)); //tototo 
}

std::string  Mangling( cXAPA_OneMatch *) {return "0EC960CE7677B3EFFABF";};


std::string & cXAPA_PtCar::Pattern()
{
   return mPattern;
}

const std::string & cXAPA_PtCar::Pattern()const 
{
   return mPattern;
}

void  BinaryUnDumpFromFile(cXAPA_PtCar & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Pattern(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cXAPA_PtCar & anObj)
{
    BinaryDumpInFile(aFp,anObj.Pattern());
}

cElXMLTree * ToXMLTree(const cXAPA_PtCar & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"XAPA_PtCar",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Pattern"),anObj.Pattern())->ReTagThis("Pattern"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cXAPA_PtCar & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.Pattern(),aTree->Get("Pattern",1)); //tototo 
}

std::string  Mangling( cXAPA_PtCar *) {return "B9085590833286EFFE3F";};


cTplValGesInit< bool > & cXlmAimeOneDir::DoIt()
{
   return mDoIt;
}

const cTplValGesInit< bool > & cXlmAimeOneDir::DoIt()const 
{
   return mDoIt;
}


cTplValGesInit< bool > & cXlmAimeOneDir::DoMatch()
{
   return mDoMatch;
}

const cTplValGesInit< bool > & cXlmAimeOneDir::DoMatch()const 
{
   return mDoMatch;
}


cTplValGesInit< bool > & cXlmAimeOneDir::DoPtCar()
{
   return mDoPtCar;
}

const cTplValGesInit< bool > & cXlmAimeOneDir::DoPtCar()const 
{
   return mDoPtCar;
}


cTplValGesInit< bool > & cXlmAimeOneDir::DoRef()
{
   return mDoRef;
}

const cTplValGesInit< bool > & cXlmAimeOneDir::DoRef()const 
{
   return mDoRef;
}


cTplValGesInit< int > & cXlmAimeOneDir::ZoomF()
{
   return mZoomF;
}

const cTplValGesInit< int > & cXlmAimeOneDir::ZoomF()const 
{
   return mZoomF;
}


cTplValGesInit< int > & cXlmAimeOneDir::NumMatch()
{
   return mNumMatch;
}

const cTplValGesInit< int > & cXlmAimeOneDir::NumMatch()const 
{
   return mNumMatch;
}


std::string & cXlmAimeOneDir::Dir()
{
   return mDir;
}

const std::string & cXlmAimeOneDir::Dir()const 
{
   return mDir;
}


std::string & cXlmAimeOneDir::Ori()
{
   return mOri;
}

const std::string & cXlmAimeOneDir::Ori()const 
{
   return mOri;
}


std::list< cXAPA_OneMatch > & cXlmAimeOneDir::XAPA_OneMatch()
{
   return mXAPA_OneMatch;
}

const std::list< cXAPA_OneMatch > & cXlmAimeOneDir::XAPA_OneMatch()const 
{
   return mXAPA_OneMatch;
}


std::string & cXlmAimeOneDir::Pattern()
{
   return XAPA_PtCar().Pattern();
}

const std::string & cXlmAimeOneDir::Pattern()const 
{
   return XAPA_PtCar().Pattern();
}


cXAPA_PtCar & cXlmAimeOneDir::XAPA_PtCar()
{
   return mXAPA_PtCar;
}

const cXAPA_PtCar & cXlmAimeOneDir::XAPA_PtCar()const 
{
   return mXAPA_PtCar;
}

void  BinaryUnDumpFromFile(cXlmAimeOneDir & anObj,ELISE_fp & aFp)
{
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
             anObj.DoMatch().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.DoMatch().ValForcedForUnUmp(),aFp);
        }
        else  anObj.DoMatch().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.DoPtCar().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.DoPtCar().ValForcedForUnUmp(),aFp);
        }
        else  anObj.DoPtCar().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.DoRef().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.DoRef().ValForcedForUnUmp(),aFp);
        }
        else  anObj.DoRef().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ZoomF().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ZoomF().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ZoomF().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NumMatch().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NumMatch().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NumMatch().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.Dir(),aFp);
    BinaryUnDumpFromFile(anObj.Ori(),aFp);
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cXAPA_OneMatch aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.XAPA_OneMatch().push_back(aVal);
        }
  } ;
    BinaryUnDumpFromFile(anObj.XAPA_PtCar(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cXlmAimeOneDir & anObj)
{
    BinaryDumpInFile(aFp,anObj.DoIt().IsInit());
    if (anObj.DoIt().IsInit()) BinaryDumpInFile(aFp,anObj.DoIt().Val());
    BinaryDumpInFile(aFp,anObj.DoMatch().IsInit());
    if (anObj.DoMatch().IsInit()) BinaryDumpInFile(aFp,anObj.DoMatch().Val());
    BinaryDumpInFile(aFp,anObj.DoPtCar().IsInit());
    if (anObj.DoPtCar().IsInit()) BinaryDumpInFile(aFp,anObj.DoPtCar().Val());
    BinaryDumpInFile(aFp,anObj.DoRef().IsInit());
    if (anObj.DoRef().IsInit()) BinaryDumpInFile(aFp,anObj.DoRef().Val());
    BinaryDumpInFile(aFp,anObj.ZoomF().IsInit());
    if (anObj.ZoomF().IsInit()) BinaryDumpInFile(aFp,anObj.ZoomF().Val());
    BinaryDumpInFile(aFp,anObj.NumMatch().IsInit());
    if (anObj.NumMatch().IsInit()) BinaryDumpInFile(aFp,anObj.NumMatch().Val());
    BinaryDumpInFile(aFp,anObj.Dir());
    BinaryDumpInFile(aFp,anObj.Ori());
    BinaryDumpInFile(aFp,(int)anObj.XAPA_OneMatch().size());
    for(  std::list< cXAPA_OneMatch >::const_iterator iT=anObj.XAPA_OneMatch().begin();
         iT!=anObj.XAPA_OneMatch().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.XAPA_PtCar());
}

cElXMLTree * ToXMLTree(const cXlmAimeOneDir & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"XlmAimeOneDir",eXMLBranche);
   if (anObj.DoIt().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DoIt"),anObj.DoIt().Val())->ReTagThis("DoIt"));
   if (anObj.DoMatch().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DoMatch"),anObj.DoMatch().Val())->ReTagThis("DoMatch"));
   if (anObj.DoPtCar().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DoPtCar"),anObj.DoPtCar().Val())->ReTagThis("DoPtCar"));
   if (anObj.DoRef().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DoRef"),anObj.DoRef().Val())->ReTagThis("DoRef"));
   if (anObj.ZoomF().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ZoomF"),anObj.ZoomF().Val())->ReTagThis("ZoomF"));
   if (anObj.NumMatch().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NumMatch"),anObj.NumMatch().Val())->ReTagThis("NumMatch"));
   aRes->AddFils(::ToXMLTree(std::string("Dir"),anObj.Dir())->ReTagThis("Dir"));
   aRes->AddFils(::ToXMLTree(std::string("Ori"),anObj.Ori())->ReTagThis("Ori"));
  for
  (       std::list< cXAPA_OneMatch >::const_iterator it=anObj.XAPA_OneMatch().begin();
      it !=anObj.XAPA_OneMatch().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("XAPA_OneMatch"));
   aRes->AddFils(ToXMLTree(anObj.XAPA_PtCar())->ReTagThis("XAPA_PtCar"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cXlmAimeOneDir & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.DoIt(),aTree->Get("DoIt",1)); //tototo 

   xml_init(anObj.DoMatch(),aTree->Get("DoMatch",1)); //tototo 

   xml_init(anObj.DoPtCar(),aTree->Get("DoPtCar",1)); //tototo 

   xml_init(anObj.DoRef(),aTree->Get("DoRef",1)); //tototo 

   xml_init(anObj.ZoomF(),aTree->Get("ZoomF",1),int(4)); //tototo 

   xml_init(anObj.NumMatch(),aTree->Get("NumMatch",1)); //tototo 

   xml_init(anObj.Dir(),aTree->Get("Dir",1)); //tototo 

   xml_init(anObj.Ori(),aTree->Get("Ori",1)); //tototo 

   xml_init(anObj.XAPA_OneMatch(),aTree->GetAll("XAPA_OneMatch",false,1));

   xml_init(anObj.XAPA_PtCar(),aTree->Get("XAPA_PtCar",1)); //tototo 
}

std::string  Mangling( cXlmAimeOneDir *) {return "34191F0ED80DC181FF3F";};


double & cXlmAimeOneApprent::PdsW()
{
   return mPdsW;
}

const double & cXlmAimeOneApprent::PdsW()const 
{
   return mPdsW;
}


int & cXlmAimeOneApprent::NbBB()
{
   return mNbBB;
}

const int & cXlmAimeOneApprent::NbBB()const 
{
   return mNbBB;
}


cTplValGesInit< int > & cXlmAimeOneApprent::BitM()
{
   return mBitM;
}

const cTplValGesInit< int > & cXlmAimeOneApprent::BitM()const 
{
   return mBitM;
}

void  BinaryUnDumpFromFile(cXlmAimeOneApprent & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.PdsW(),aFp);
    BinaryUnDumpFromFile(anObj.NbBB(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.BitM().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.BitM().ValForcedForUnUmp(),aFp);
        }
        else  anObj.BitM().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cXlmAimeOneApprent & anObj)
{
    BinaryDumpInFile(aFp,anObj.PdsW());
    BinaryDumpInFile(aFp,anObj.NbBB());
    BinaryDumpInFile(aFp,anObj.BitM().IsInit());
    if (anObj.BitM().IsInit()) BinaryDumpInFile(aFp,anObj.BitM().Val());
}

cElXMLTree * ToXMLTree(const cXlmAimeOneApprent & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"XlmAimeOneApprent",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("PdsW"),anObj.PdsW())->ReTagThis("PdsW"));
   aRes->AddFils(::ToXMLTree(std::string("NbBB"),anObj.NbBB())->ReTagThis("NbBB"));
   if (anObj.BitM().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("BitM"),anObj.BitM().Val())->ReTagThis("BitM"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cXlmAimeOneApprent & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.PdsW(),aTree->Get("PdsW",1)); //tototo 

   xml_init(anObj.NbBB(),aTree->Get("NbBB",1)); //tototo 

   xml_init(anObj.BitM(),aTree->Get("BitM",1)); //tototo 
}

std::string  Mangling( cXlmAimeOneApprent *) {return "FEEBE48B5B238E8CFE3F";};


int & cXlmAimeApprent::NbExEt0()
{
   return mNbExEt0;
}

const int & cXlmAimeApprent::NbExEt0()const 
{
   return mNbExEt0;
}


int & cXlmAimeApprent::NbExEt1()
{
   return mNbExEt1;
}

const int & cXlmAimeApprent::NbExEt1()const 
{
   return mNbExEt1;
}


cTplValGesInit< double > & cXlmAimeApprent::TimeOut()
{
   return mTimeOut;
}

const cTplValGesInit< double > & cXlmAimeApprent::TimeOut()const 
{
   return mTimeOut;
}


std::list< cXlmAimeOneApprent > & cXlmAimeApprent::XlmAimeOneApprent()
{
   return mXlmAimeOneApprent;
}

const std::list< cXlmAimeOneApprent > & cXlmAimeApprent::XlmAimeOneApprent()const 
{
   return mXlmAimeOneApprent;
}

void  BinaryUnDumpFromFile(cXlmAimeApprent & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.NbExEt0(),aFp);
    BinaryUnDumpFromFile(anObj.NbExEt1(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.TimeOut().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.TimeOut().ValForcedForUnUmp(),aFp);
        }
        else  anObj.TimeOut().SetNoInit();
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cXlmAimeOneApprent aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.XlmAimeOneApprent().push_back(aVal);
        }
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cXlmAimeApprent & anObj)
{
    BinaryDumpInFile(aFp,anObj.NbExEt0());
    BinaryDumpInFile(aFp,anObj.NbExEt1());
    BinaryDumpInFile(aFp,anObj.TimeOut().IsInit());
    if (anObj.TimeOut().IsInit()) BinaryDumpInFile(aFp,anObj.TimeOut().Val());
    BinaryDumpInFile(aFp,(int)anObj.XlmAimeOneApprent().size());
    for(  std::list< cXlmAimeOneApprent >::const_iterator iT=anObj.XlmAimeOneApprent().begin();
         iT!=anObj.XlmAimeOneApprent().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
}

cElXMLTree * ToXMLTree(const cXlmAimeApprent & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"XlmAimeApprent",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("NbExEt0"),anObj.NbExEt0())->ReTagThis("NbExEt0"));
   aRes->AddFils(::ToXMLTree(std::string("NbExEt1"),anObj.NbExEt1())->ReTagThis("NbExEt1"));
   if (anObj.TimeOut().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("TimeOut"),anObj.TimeOut().Val())->ReTagThis("TimeOut"));
  for
  (       std::list< cXlmAimeOneApprent >::const_iterator it=anObj.XlmAimeOneApprent().begin();
      it !=anObj.XlmAimeOneApprent().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("XlmAimeOneApprent"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cXlmAimeApprent & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.NbExEt0(),aTree->Get("NbExEt0",1)); //tototo 

   xml_init(anObj.NbExEt1(),aTree->Get("NbExEt1",1)); //tototo 

   xml_init(anObj.TimeOut(),aTree->Get("TimeOut",1),double(300.0)); //tototo 

   xml_init(anObj.XlmAimeOneApprent(),aTree->GetAll("XlmAimeOneApprent",false,1));
}

std::string  Mangling( cXlmAimeApprent *) {return "448F2106F3DE94B2FCBF";};


std::string & cXmlAimeParamApprentissage::AbsDir()
{
   return mAbsDir;
}

const std::string & cXmlAimeParamApprentissage::AbsDir()const 
{
   return mAbsDir;
}


cTplValGesInit< bool > & cXmlAimeParamApprentissage::DefDoIt()
{
   return mDefDoIt;
}

const cTplValGesInit< bool > & cXmlAimeParamApprentissage::DefDoIt()const 
{
   return mDefDoIt;
}


cTplValGesInit< bool > & cXmlAimeParamApprentissage::DefDoMatch()
{
   return mDefDoMatch;
}

const cTplValGesInit< bool > & cXmlAimeParamApprentissage::DefDoMatch()const 
{
   return mDefDoMatch;
}


cTplValGesInit< bool > & cXmlAimeParamApprentissage::DefDoPtCar()
{
   return mDefDoPtCar;
}

const cTplValGesInit< bool > & cXmlAimeParamApprentissage::DefDoPtCar()const 
{
   return mDefDoPtCar;
}


cTplValGesInit< bool > & cXmlAimeParamApprentissage::DefDoRef()
{
   return mDefDoRef;
}

const cTplValGesInit< bool > & cXmlAimeParamApprentissage::DefDoRef()const 
{
   return mDefDoRef;
}


cTplValGesInit< bool > & cXmlAimeParamApprentissage::DefDoApprComb()
{
   return mDefDoApprComb;
}

const cTplValGesInit< bool > & cXmlAimeParamApprentissage::DefDoApprComb()const 
{
   return mDefDoApprComb;
}


cTplValGesInit< bool > & cXmlAimeParamApprentissage::DefDoApprLocal1()
{
   return mDefDoApprLocal1;
}

const cTplValGesInit< bool > & cXmlAimeParamApprentissage::DefDoApprLocal1()const 
{
   return mDefDoApprLocal1;
}


cTplValGesInit< bool > & cXmlAimeParamApprentissage::DefDoApprLocal2()
{
   return mDefDoApprLocal2;
}

const cTplValGesInit< bool > & cXmlAimeParamApprentissage::DefDoApprLocal2()const 
{
   return mDefDoApprLocal2;
}


cTplValGesInit< std::string > & cXmlAimeParamApprentissage::DefParamPtCar()
{
   return mDefParamPtCar;
}

const cTplValGesInit< std::string > & cXmlAimeParamApprentissage::DefParamPtCar()const 
{
   return mDefParamPtCar;
}


std::list< cXlmAimeOneDir > & cXmlAimeParamApprentissage::XlmAimeOneDir()
{
   return mXlmAimeOneDir;
}

const std::list< cXlmAimeOneDir > & cXmlAimeParamApprentissage::XlmAimeOneDir()const 
{
   return mXlmAimeOneDir;
}


int & cXmlAimeParamApprentissage::NbExEt0()
{
   return XlmAimeApprent().NbExEt0();
}

const int & cXmlAimeParamApprentissage::NbExEt0()const 
{
   return XlmAimeApprent().NbExEt0();
}


int & cXmlAimeParamApprentissage::NbExEt1()
{
   return XlmAimeApprent().NbExEt1();
}

const int & cXmlAimeParamApprentissage::NbExEt1()const 
{
   return XlmAimeApprent().NbExEt1();
}


cTplValGesInit< double > & cXmlAimeParamApprentissage::TimeOut()
{
   return XlmAimeApprent().TimeOut();
}

const cTplValGesInit< double > & cXmlAimeParamApprentissage::TimeOut()const 
{
   return XlmAimeApprent().TimeOut();
}


std::list< cXlmAimeOneApprent > & cXmlAimeParamApprentissage::XlmAimeOneApprent()
{
   return XlmAimeApprent().XlmAimeOneApprent();
}

const std::list< cXlmAimeOneApprent > & cXmlAimeParamApprentissage::XlmAimeOneApprent()const 
{
   return XlmAimeApprent().XlmAimeOneApprent();
}


cXlmAimeApprent & cXmlAimeParamApprentissage::XlmAimeApprent()
{
   return mXlmAimeApprent;
}

const cXlmAimeApprent & cXmlAimeParamApprentissage::XlmAimeApprent()const 
{
   return mXlmAimeApprent;
}

void  BinaryUnDumpFromFile(cXmlAimeParamApprentissage & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.AbsDir(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.DefDoIt().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.DefDoIt().ValForcedForUnUmp(),aFp);
        }
        else  anObj.DefDoIt().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.DefDoMatch().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.DefDoMatch().ValForcedForUnUmp(),aFp);
        }
        else  anObj.DefDoMatch().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.DefDoPtCar().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.DefDoPtCar().ValForcedForUnUmp(),aFp);
        }
        else  anObj.DefDoPtCar().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.DefDoRef().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.DefDoRef().ValForcedForUnUmp(),aFp);
        }
        else  anObj.DefDoRef().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.DefDoApprComb().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.DefDoApprComb().ValForcedForUnUmp(),aFp);
        }
        else  anObj.DefDoApprComb().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.DefDoApprLocal1().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.DefDoApprLocal1().ValForcedForUnUmp(),aFp);
        }
        else  anObj.DefDoApprLocal1().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.DefDoApprLocal2().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.DefDoApprLocal2().ValForcedForUnUmp(),aFp);
        }
        else  anObj.DefDoApprLocal2().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.DefParamPtCar().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.DefParamPtCar().ValForcedForUnUmp(),aFp);
        }
        else  anObj.DefParamPtCar().SetNoInit();
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cXlmAimeOneDir aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.XlmAimeOneDir().push_back(aVal);
        }
  } ;
    BinaryUnDumpFromFile(anObj.XlmAimeApprent(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cXmlAimeParamApprentissage & anObj)
{
    BinaryDumpInFile(aFp,anObj.AbsDir());
    BinaryDumpInFile(aFp,anObj.DefDoIt().IsInit());
    if (anObj.DefDoIt().IsInit()) BinaryDumpInFile(aFp,anObj.DefDoIt().Val());
    BinaryDumpInFile(aFp,anObj.DefDoMatch().IsInit());
    if (anObj.DefDoMatch().IsInit()) BinaryDumpInFile(aFp,anObj.DefDoMatch().Val());
    BinaryDumpInFile(aFp,anObj.DefDoPtCar().IsInit());
    if (anObj.DefDoPtCar().IsInit()) BinaryDumpInFile(aFp,anObj.DefDoPtCar().Val());
    BinaryDumpInFile(aFp,anObj.DefDoRef().IsInit());
    if (anObj.DefDoRef().IsInit()) BinaryDumpInFile(aFp,anObj.DefDoRef().Val());
    BinaryDumpInFile(aFp,anObj.DefDoApprComb().IsInit());
    if (anObj.DefDoApprComb().IsInit()) BinaryDumpInFile(aFp,anObj.DefDoApprComb().Val());
    BinaryDumpInFile(aFp,anObj.DefDoApprLocal1().IsInit());
    if (anObj.DefDoApprLocal1().IsInit()) BinaryDumpInFile(aFp,anObj.DefDoApprLocal1().Val());
    BinaryDumpInFile(aFp,anObj.DefDoApprLocal2().IsInit());
    if (anObj.DefDoApprLocal2().IsInit()) BinaryDumpInFile(aFp,anObj.DefDoApprLocal2().Val());
    BinaryDumpInFile(aFp,anObj.DefParamPtCar().IsInit());
    if (anObj.DefParamPtCar().IsInit()) BinaryDumpInFile(aFp,anObj.DefParamPtCar().Val());
    BinaryDumpInFile(aFp,(int)anObj.XlmAimeOneDir().size());
    for(  std::list< cXlmAimeOneDir >::const_iterator iT=anObj.XlmAimeOneDir().begin();
         iT!=anObj.XlmAimeOneDir().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.XlmAimeApprent());
}

cElXMLTree * ToXMLTree(const cXmlAimeParamApprentissage & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"XmlAimeParamApprentissage",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("AbsDir"),anObj.AbsDir())->ReTagThis("AbsDir"));
   if (anObj.DefDoIt().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DefDoIt"),anObj.DefDoIt().Val())->ReTagThis("DefDoIt"));
   if (anObj.DefDoMatch().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DefDoMatch"),anObj.DefDoMatch().Val())->ReTagThis("DefDoMatch"));
   if (anObj.DefDoPtCar().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DefDoPtCar"),anObj.DefDoPtCar().Val())->ReTagThis("DefDoPtCar"));
   if (anObj.DefDoRef().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DefDoRef"),anObj.DefDoRef().Val())->ReTagThis("DefDoRef"));
   if (anObj.DefDoApprComb().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DefDoApprComb"),anObj.DefDoApprComb().Val())->ReTagThis("DefDoApprComb"));
   if (anObj.DefDoApprLocal1().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DefDoApprLocal1"),anObj.DefDoApprLocal1().Val())->ReTagThis("DefDoApprLocal1"));
   if (anObj.DefDoApprLocal2().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DefDoApprLocal2"),anObj.DefDoApprLocal2().Val())->ReTagThis("DefDoApprLocal2"));
   if (anObj.DefParamPtCar().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DefParamPtCar"),anObj.DefParamPtCar().Val())->ReTagThis("DefParamPtCar"));
  for
  (       std::list< cXlmAimeOneDir >::const_iterator it=anObj.XlmAimeOneDir().begin();
      it !=anObj.XlmAimeOneDir().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("XlmAimeOneDir"));
   aRes->AddFils(ToXMLTree(anObj.XlmAimeApprent())->ReTagThis("XlmAimeApprent"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cXmlAimeParamApprentissage & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.AbsDir(),aTree->Get("AbsDir",1)); //tototo 

   xml_init(anObj.DefDoIt(),aTree->Get("DefDoIt",1),bool(true)); //tototo 

   xml_init(anObj.DefDoMatch(),aTree->Get("DefDoMatch",1),bool(true)); //tototo 

   xml_init(anObj.DefDoPtCar(),aTree->Get("DefDoPtCar",1),bool(true)); //tototo 

   xml_init(anObj.DefDoRef(),aTree->Get("DefDoRef",1),bool(true)); //tototo 

   xml_init(anObj.DefDoApprComb(),aTree->Get("DefDoApprComb",1),bool(true)); //tototo 

   xml_init(anObj.DefDoApprLocal1(),aTree->Get("DefDoApprLocal1",1),bool(true)); //tototo 

   xml_init(anObj.DefDoApprLocal2(),aTree->Get("DefDoApprLocal2",1),bool(true)); //tototo 

   xml_init(anObj.DefParamPtCar(),aTree->Get("DefParamPtCar",1),std::string("")); //tototo 

   xml_init(anObj.XlmAimeOneDir(),aTree->GetAll("XlmAimeOneDir",false,1));

   xml_init(anObj.XlmAimeApprent(),aTree->Get("XlmAimeApprent",1)); //tototo 
}

std::string  Mangling( cXmlAimeParamApprentissage *) {return "E9703653668F268FFE3F";};


Pt2dr & cXml2007Pt::PtInit()
{
   return mPtInit;
}

const Pt2dr & cXml2007Pt::PtInit()const 
{
   return mPtInit;
}


Pt2dr & cXml2007Pt::PtAff()
{
   return mPtAff;
}

const Pt2dr & cXml2007Pt::PtAff()const 
{
   return mPtAff;
}


int & cXml2007Pt::Id()
{
   return mId;
}

const int & cXml2007Pt::Id()const 
{
   return mId;
}


int & cXml2007Pt::NumOct()
{
   return mNumOct;
}

const int & cXml2007Pt::NumOct()const 
{
   return mNumOct;
}


int & cXml2007Pt::NumIm()
{
   return mNumIm;
}

const int & cXml2007Pt::NumIm()const 
{
   return mNumIm;
}


double & cXml2007Pt::ScaleInO()
{
   return mScaleInO;
}

const double & cXml2007Pt::ScaleInO()const 
{
   return mScaleInO;
}


double & cXml2007Pt::ScaleAbs()
{
   return mScaleAbs;
}

const double & cXml2007Pt::ScaleAbs()const 
{
   return mScaleAbs;
}


double & cXml2007Pt::Score()
{
   return mScore;
}

const double & cXml2007Pt::Score()const 
{
   return mScore;
}


double & cXml2007Pt::ScoreRel()
{
   return mScoreRel;
}

const double & cXml2007Pt::ScoreRel()const 
{
   return mScoreRel;
}


std::vector<double> & cXml2007Pt::VectRho()
{
   return mVectRho;
}

const std::vector<double> & cXml2007Pt::VectRho()const 
{
   return mVectRho;
}


std::vector<double> & cXml2007Pt::VectDir()
{
   return mVectDir;
}

const std::vector<double> & cXml2007Pt::VectDir()const 
{
   return mVectDir;
}


double & cXml2007Pt::Var()
{
   return mVar;
}

const double & cXml2007Pt::Var()const 
{
   return mVar;
}


double & cXml2007Pt::AutoCor()
{
   return mAutoCor;
}

const double & cXml2007Pt::AutoCor()const 
{
   return mAutoCor;
}


int & cXml2007Pt::NumChAC()
{
   return mNumChAC;
}

const int & cXml2007Pt::NumChAC()const 
{
   return mNumChAC;
}


bool & cXml2007Pt::OKAc()
{
   return mOKAc;
}

const bool & cXml2007Pt::OKAc()const 
{
   return mOKAc;
}


bool & cXml2007Pt::OKLP()
{
   return mOKLP;
}

const bool & cXml2007Pt::OKLP()const 
{
   return mOKLP;
}


bool & cXml2007Pt::SFSelected()
{
   return mSFSelected;
}

const bool & cXml2007Pt::SFSelected()const 
{
   return mSFSelected;
}


bool & cXml2007Pt::Stable()
{
   return mStable;
}

const bool & cXml2007Pt::Stable()const 
{
   return mStable;
}


bool & cXml2007Pt::ChgMaj()
{
   return mChgMaj;
}

const bool & cXml2007Pt::ChgMaj()const 
{
   return mChgMaj;
}


Im2D_U_INT1 & cXml2007Pt::ImLP()
{
   return mImLP;
}

const Im2D_U_INT1 & cXml2007Pt::ImLP()const 
{
   return mImLP;
}

void  BinaryUnDumpFromFile(cXml2007Pt & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.PtInit(),aFp);
    BinaryUnDumpFromFile(anObj.PtAff(),aFp);
    BinaryUnDumpFromFile(anObj.Id(),aFp);
    BinaryUnDumpFromFile(anObj.NumOct(),aFp);
    BinaryUnDumpFromFile(anObj.NumIm(),aFp);
    BinaryUnDumpFromFile(anObj.ScaleInO(),aFp);
    BinaryUnDumpFromFile(anObj.ScaleAbs(),aFp);
    BinaryUnDumpFromFile(anObj.Score(),aFp);
    BinaryUnDumpFromFile(anObj.ScoreRel(),aFp);
    BinaryUnDumpFromFile(anObj.VectRho(),aFp);
    BinaryUnDumpFromFile(anObj.VectDir(),aFp);
    BinaryUnDumpFromFile(anObj.Var(),aFp);
    BinaryUnDumpFromFile(anObj.AutoCor(),aFp);
    BinaryUnDumpFromFile(anObj.NumChAC(),aFp);
    BinaryUnDumpFromFile(anObj.OKAc(),aFp);
    BinaryUnDumpFromFile(anObj.OKLP(),aFp);
    BinaryUnDumpFromFile(anObj.SFSelected(),aFp);
    BinaryUnDumpFromFile(anObj.Stable(),aFp);
    BinaryUnDumpFromFile(anObj.ChgMaj(),aFp);
    BinaryUnDumpFromFile(anObj.ImLP(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cXml2007Pt & anObj)
{
    BinaryDumpInFile(aFp,anObj.PtInit());
    BinaryDumpInFile(aFp,anObj.PtAff());
    BinaryDumpInFile(aFp,anObj.Id());
    BinaryDumpInFile(aFp,anObj.NumOct());
    BinaryDumpInFile(aFp,anObj.NumIm());
    BinaryDumpInFile(aFp,anObj.ScaleInO());
    BinaryDumpInFile(aFp,anObj.ScaleAbs());
    BinaryDumpInFile(aFp,anObj.Score());
    BinaryDumpInFile(aFp,anObj.ScoreRel());
    BinaryDumpInFile(aFp,anObj.VectRho());
    BinaryDumpInFile(aFp,anObj.VectDir());
    BinaryDumpInFile(aFp,anObj.Var());
    BinaryDumpInFile(aFp,anObj.AutoCor());
    BinaryDumpInFile(aFp,anObj.NumChAC());
    BinaryDumpInFile(aFp,anObj.OKAc());
    BinaryDumpInFile(aFp,anObj.OKLP());
    BinaryDumpInFile(aFp,anObj.SFSelected());
    BinaryDumpInFile(aFp,anObj.Stable());
    BinaryDumpInFile(aFp,anObj.ChgMaj());
    BinaryDumpInFile(aFp,anObj.ImLP());
}

cElXMLTree * ToXMLTree(const cXml2007Pt & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"Xml2007Pt",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("PtInit"),anObj.PtInit())->ReTagThis("PtInit"));
   aRes->AddFils(::ToXMLTree(std::string("PtAff"),anObj.PtAff())->ReTagThis("PtAff"));
   aRes->AddFils(::ToXMLTree(std::string("Id"),anObj.Id())->ReTagThis("Id"));
   aRes->AddFils(::ToXMLTree(std::string("NumOct"),anObj.NumOct())->ReTagThis("NumOct"));
   aRes->AddFils(::ToXMLTree(std::string("NumIm"),anObj.NumIm())->ReTagThis("NumIm"));
   aRes->AddFils(::ToXMLTree(std::string("ScaleInO"),anObj.ScaleInO())->ReTagThis("ScaleInO"));
   aRes->AddFils(::ToXMLTree(std::string("ScaleAbs"),anObj.ScaleAbs())->ReTagThis("ScaleAbs"));
   aRes->AddFils(::ToXMLTree(std::string("Score"),anObj.Score())->ReTagThis("Score"));
   aRes->AddFils(::ToXMLTree(std::string("ScoreRel"),anObj.ScoreRel())->ReTagThis("ScoreRel"));
   aRes->AddFils(::ToXMLTree(std::string("VectRho"),anObj.VectRho())->ReTagThis("VectRho"));
   aRes->AddFils(::ToXMLTree(std::string("VectDir"),anObj.VectDir())->ReTagThis("VectDir"));
   aRes->AddFils(::ToXMLTree(std::string("Var"),anObj.Var())->ReTagThis("Var"));
   aRes->AddFils(::ToXMLTree(std::string("AutoCor"),anObj.AutoCor())->ReTagThis("AutoCor"));
   aRes->AddFils(::ToXMLTree(std::string("NumChAC"),anObj.NumChAC())->ReTagThis("NumChAC"));
   aRes->AddFils(::ToXMLTree(std::string("OKAc"),anObj.OKAc())->ReTagThis("OKAc"));
   aRes->AddFils(::ToXMLTree(std::string("OKLP"),anObj.OKLP())->ReTagThis("OKLP"));
   aRes->AddFils(::ToXMLTree(std::string("SFSelected"),anObj.SFSelected())->ReTagThis("SFSelected"));
   aRes->AddFils(::ToXMLTree(std::string("Stable"),anObj.Stable())->ReTagThis("Stable"));
   aRes->AddFils(::ToXMLTree(std::string("ChgMaj"),anObj.ChgMaj())->ReTagThis("ChgMaj"));
   aRes->AddFils(::ToXMLTree(std::string("ImLP"),anObj.ImLP())->ReTagThis("ImLP"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cXml2007Pt & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.PtInit(),aTree->Get("PtInit",1)); //tototo 

   xml_init(anObj.PtAff(),aTree->Get("PtAff",1)); //tototo 

   xml_init(anObj.Id(),aTree->Get("Id",1)); //tototo 

   xml_init(anObj.NumOct(),aTree->Get("NumOct",1)); //tototo 

   xml_init(anObj.NumIm(),aTree->Get("NumIm",1)); //tototo 

   xml_init(anObj.ScaleInO(),aTree->Get("ScaleInO",1)); //tototo 

   xml_init(anObj.ScaleAbs(),aTree->Get("ScaleAbs",1)); //tototo 

   xml_init(anObj.Score(),aTree->Get("Score",1)); //tototo 

   xml_init(anObj.ScoreRel(),aTree->Get("ScoreRel",1)); //tototo 

   xml_init(anObj.VectRho(),aTree->Get("VectRho",1)); //tototo 

   xml_init(anObj.VectDir(),aTree->Get("VectDir",1)); //tototo 

   xml_init(anObj.Var(),aTree->Get("Var",1)); //tototo 

   xml_init(anObj.AutoCor(),aTree->Get("AutoCor",1)); //tototo 

   xml_init(anObj.NumChAC(),aTree->Get("NumChAC",1)); //tototo 

   xml_init(anObj.OKAc(),aTree->Get("OKAc",1)); //tototo 

   xml_init(anObj.OKLP(),aTree->Get("OKLP",1)); //tototo 

   xml_init(anObj.SFSelected(),aTree->Get("SFSelected",1)); //tototo 

   xml_init(anObj.Stable(),aTree->Get("Stable",1)); //tototo 

   xml_init(anObj.ChgMaj(),aTree->Get("ChgMaj",1)); //tototo 

   xml_init(anObj.ImLP(),aTree->Get("ImLP",1)); //tototo 
}

std::string  Mangling( cXml2007Pt *) {return "9AFAFEB31711B7AAFE3F";};


std::vector< cXml2007Pt > & cXml2007SetPtOneType::Pts()
{
   return mPts;
}

const std::vector< cXml2007Pt > & cXml2007SetPtOneType::Pts()const 
{
   return mPts;
}


bool & cXml2007SetPtOneType::IsMax()
{
   return mIsMax;
}

const bool & cXml2007SetPtOneType::IsMax()const 
{
   return mIsMax;
}


int & cXml2007SetPtOneType::TypePt()
{
   return mTypePt;
}

const int & cXml2007SetPtOneType::TypePt()const 
{
   return mTypePt;
}


std::string & cXml2007SetPtOneType::NameTypePt()
{
   return mNameTypePt;
}

const std::string & cXml2007SetPtOneType::NameTypePt()const 
{
   return mNameTypePt;
}

void  BinaryUnDumpFromFile(cXml2007SetPtOneType & anObj,ELISE_fp & aFp)
{
   { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cXml2007Pt aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.Pts().push_back(aVal);
        }
  } ;
    BinaryUnDumpFromFile(anObj.IsMax(),aFp);
    BinaryUnDumpFromFile(anObj.TypePt(),aFp);
    BinaryUnDumpFromFile(anObj.NameTypePt(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cXml2007SetPtOneType & anObj)
{
    BinaryDumpInFile(aFp,(int)anObj.Pts().size());
    for(  std::vector< cXml2007Pt >::const_iterator iT=anObj.Pts().begin();
         iT!=anObj.Pts().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.IsMax());
    BinaryDumpInFile(aFp,anObj.TypePt());
    BinaryDumpInFile(aFp,anObj.NameTypePt());
}

cElXMLTree * ToXMLTree(const cXml2007SetPtOneType & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"Xml2007SetPtOneType",eXMLBranche);
  for
  (       std::vector< cXml2007Pt >::const_iterator it=anObj.Pts().begin();
      it !=anObj.Pts().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("Pts"));
   aRes->AddFils(::ToXMLTree(std::string("IsMax"),anObj.IsMax())->ReTagThis("IsMax"));
   aRes->AddFils(::ToXMLTree(std::string("TypePt"),anObj.TypePt())->ReTagThis("TypePt"));
   aRes->AddFils(::ToXMLTree(std::string("NameTypePt"),anObj.NameTypePt())->ReTagThis("NameTypePt"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cXml2007SetPtOneType & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.Pts(),aTree->GetAll("Pts",false,1));

   xml_init(anObj.IsMax(),aTree->Get("IsMax",1)); //tototo 

   xml_init(anObj.TypePt(),aTree->Get("TypePt",1)); //tototo 

   xml_init(anObj.NameTypePt(),aTree->Get("NameTypePt",1)); //tototo 
}

std::string  Mangling( cXml2007SetPtOneType *) {return "5E47ABA22613D9D2FD3F";};

// };
