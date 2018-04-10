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
   else if (aName=="eTPR_GraySadl")
      return eTPR_GraySadl;
   else if (aName=="eTPR_NoLabel")
      return eTPR_NoLabel;
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
   if (anObj==eTPR_GraySadl)
      return  "eTPR_GraySadl";
   if (anObj==eTPR_NoLabel)
      return  "eTPR_NoLabel";
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

std::string  Mangling( eTypePtRemark *) {return "42EC1EA5DA0B93EFFE3F";};

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
}

std::string  Mangling( cOnePCarac *) {return "96CDA234A73B1FAAFD3F";};


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

std::string  Mangling( cSetPCarac *) {return "AEC8D8B3BEEB0CA4FE3F";};


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

std::string  Mangling( cSRPC_Truth *) {return "57012D64E0B9E8C3FEBF";};


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

std::string  Mangling( cSetRefPCarac *) {return "4C82BC83862739A7FE3F";};


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


cFitsOneBin & cFitsOneLabel::BinDecision()
{
   return mBinDecision;
}

const cFitsOneBin & cFitsOneLabel::BinDecision()const 
{
   return mBinDecision;
}

void  BinaryUnDumpFromFile(cFitsOneLabel & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.KindOf(),aFp);
    BinaryUnDumpFromFile(anObj.BinIndexed(),aFp);
    BinaryUnDumpFromFile(anObj.BinDecision(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cFitsOneLabel & anObj)
{
    BinaryDumpInFile(aFp,anObj.KindOf());
    BinaryDumpInFile(aFp,anObj.BinIndexed());
    BinaryDumpInFile(aFp,anObj.BinDecision());
}

cElXMLTree * ToXMLTree(const cFitsOneLabel & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"FitsOneLabel",eXMLBranche);
   aRes->AddFils(ToXMLTree(std::string("KindOf"),anObj.KindOf())->ReTagThis("KindOf"));
   aRes->AddFils(ToXMLTree(anObj.BinIndexed())->ReTagThis("BinIndexed"));
   aRes->AddFils(ToXMLTree(anObj.BinDecision())->ReTagThis("BinDecision"));
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

   xml_init(anObj.BinDecision(),aTree->Get("BinDecision",1)); //tototo 
}

std::string  Mangling( cFitsOneLabel *) {return "F97812B24DBE65DCFD3F";};


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
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cSeuilFitsParam & anObj)
{
    BinaryDumpInFile(aFp,anObj.SeuilCorrDR().IsInit());
    if (anObj.SeuilCorrDR().IsInit()) BinaryDumpInFile(aFp,anObj.SeuilCorrDR().Val());
    BinaryDumpInFile(aFp,anObj.SeuilInc().IsInit());
    if (anObj.SeuilInc().IsInit()) BinaryDumpInFile(aFp,anObj.SeuilInc().Val());
    BinaryDumpInFile(aFp,anObj.SeuilCorrLP().IsInit());
    if (anObj.SeuilCorrLP().IsInit()) BinaryDumpInFile(aFp,anObj.SeuilCorrLP().Val());
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
}

std::string  Mangling( cSeuilFitsParam *) {return "F4557778B9893481FF3F";};


cFitsOneLabel & cFitsParam::OverLap()
{
   return mOverLap;
}

const cFitsOneLabel & cFitsParam::OverLap()const 
{
   return mOverLap;
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
     BinaryUnDumpFromFile(anObj.OverLap(),aFp);
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
    BinaryDumpInFile(aFp,anObj.OverLap());
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
   aRes->AddFils(ToXMLTree(anObj.OverLap())->ReTagThis("OverLap"));
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

   xml_init(anObj.OverLap(),aTree->Get("OverLap",1)); //tototo 

   xml_init(anObj.GenLabs(),aTree->GetAll("GenLabs",false,1));

   xml_init(anObj.SeuilOL(),aTree->Get("SeuilOL",1)); //tototo 

   xml_init(anObj.SeuilGen(),aTree->Get("SeuilGen",1)); //tototo 
}

std::string  Mangling( cFitsParam *) {return "E8A6A86802AA2EDFFE3F";};

// };
