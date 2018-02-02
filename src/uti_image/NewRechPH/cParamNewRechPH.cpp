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


std::vector<double> & cOnePCarac::CoeffRadiom()
{
   return mCoeffRadiom;
}

const std::vector<double> & cOnePCarac::CoeffRadiom()const 
{
   return mCoeffRadiom;
}


std::vector<double> & cOnePCarac::CoeffRadiom2()
{
   return mCoeffRadiom2;
}

const std::vector<double> & cOnePCarac::CoeffRadiom2()const 
{
   return mCoeffRadiom2;
}


std::vector<double> & cOnePCarac::CoeffGradRadial()
{
   return mCoeffGradRadial;
}

const std::vector<double> & cOnePCarac::CoeffGradRadial()const 
{
   return mCoeffGradRadial;
}


std::vector<double> & cOnePCarac::CoeffGradTangent()
{
   return mCoeffGradTangent;
}

const std::vector<double> & cOnePCarac::CoeffGradTangent()const 
{
   return mCoeffGradTangent;
}


std::vector<double> & cOnePCarac::CoeffGradTangentPiS2()
{
   return mCoeffGradTangentPiS2;
}

const std::vector<double> & cOnePCarac::CoeffGradTangentPiS2()const 
{
   return mCoeffGradTangentPiS2;
}


std::vector<double> & cOnePCarac::CoeffGradTangentPi()
{
   return mCoeffGradTangentPi;
}

const std::vector<double> & cOnePCarac::CoeffGradTangentPi()const 
{
   return mCoeffGradTangentPi;
}


std::vector<double> & cOnePCarac::CoeffGradCroise()
{
   return mCoeffGradCroise;
}

const std::vector<double> & cOnePCarac::CoeffGradCroise()const 
{
   return mCoeffGradCroise;
}


std::vector<double> & cOnePCarac::CoeffGradCroise2()
{
   return mCoeffGradCroise2;
}

const std::vector<double> & cOnePCarac::CoeffGradCroise2()const 
{
   return mCoeffGradCroise2;
}


std::vector<double> & cOnePCarac::CoeffDiffOpposePi()
{
   return mCoeffDiffOpposePi;
}

const std::vector<double> & cOnePCarac::CoeffDiffOpposePi()const 
{
   return mCoeffDiffOpposePi;
}


std::vector<double> & cOnePCarac::CoeffDiffOppose2Pi()
{
   return mCoeffDiffOppose2Pi;
}

const std::vector<double> & cOnePCarac::CoeffDiffOppose2Pi()const 
{
   return mCoeffDiffOppose2Pi;
}


std::vector<double> & cOnePCarac::CoeffDiffOpposePiS2()
{
   return mCoeffDiffOpposePiS2;
}

const std::vector<double> & cOnePCarac::CoeffDiffOpposePiS2()const 
{
   return mCoeffDiffOpposePiS2;
}


std::vector<double> & cOnePCarac::CoeffDiffOppose2PiS2()
{
   return mCoeffDiffOppose2PiS2;
}

const std::vector<double> & cOnePCarac::CoeffDiffOppose2PiS2()const 
{
   return mCoeffDiffOppose2PiS2;
}


int & cOnePCarac::CodeBinaireCompl()
{
   return mCodeBinaireCompl;
}

const int & cOnePCarac::CodeBinaireCompl()const 
{
   return mCodeBinaireCompl;
}


int & cOnePCarac::CodeBinaireIndex()
{
   return mCodeBinaireIndex;
}

const int & cOnePCarac::CodeBinaireIndex()const 
{
   return mCodeBinaireIndex;
}


Im2D_REAL4 & cOnePCarac::ImRad()
{
   return mImRad;
}

const Im2D_REAL4 & cOnePCarac::ImRad()const 
{
   return mImRad;
}


std::vector<double> & cOnePCarac::VectRho()
{
   return mVectRho;
}

const std::vector<double> & cOnePCarac::VectRho()const 
{
   return mVectRho;
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
    BinaryUnDumpFromFile(anObj.CoeffRadiom(),aFp);
    BinaryUnDumpFromFile(anObj.CoeffRadiom2(),aFp);
    BinaryUnDumpFromFile(anObj.CoeffGradRadial(),aFp);
    BinaryUnDumpFromFile(anObj.CoeffGradTangent(),aFp);
    BinaryUnDumpFromFile(anObj.CoeffGradTangentPiS2(),aFp);
    BinaryUnDumpFromFile(anObj.CoeffGradTangentPi(),aFp);
    BinaryUnDumpFromFile(anObj.CoeffGradCroise(),aFp);
    BinaryUnDumpFromFile(anObj.CoeffGradCroise2(),aFp);
    BinaryUnDumpFromFile(anObj.CoeffDiffOpposePi(),aFp);
    BinaryUnDumpFromFile(anObj.CoeffDiffOppose2Pi(),aFp);
    BinaryUnDumpFromFile(anObj.CoeffDiffOpposePiS2(),aFp);
    BinaryUnDumpFromFile(anObj.CoeffDiffOppose2PiS2(),aFp);
    BinaryUnDumpFromFile(anObj.CodeBinaireCompl(),aFp);
    BinaryUnDumpFromFile(anObj.CodeBinaireIndex(),aFp);
    BinaryUnDumpFromFile(anObj.ImRad(),aFp);
    BinaryUnDumpFromFile(anObj.VectRho(),aFp);
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
    BinaryDumpInFile(aFp,anObj.CoeffRadiom());
    BinaryDumpInFile(aFp,anObj.CoeffRadiom2());
    BinaryDumpInFile(aFp,anObj.CoeffGradRadial());
    BinaryDumpInFile(aFp,anObj.CoeffGradTangent());
    BinaryDumpInFile(aFp,anObj.CoeffGradTangentPiS2());
    BinaryDumpInFile(aFp,anObj.CoeffGradTangentPi());
    BinaryDumpInFile(aFp,anObj.CoeffGradCroise());
    BinaryDumpInFile(aFp,anObj.CoeffGradCroise2());
    BinaryDumpInFile(aFp,anObj.CoeffDiffOpposePi());
    BinaryDumpInFile(aFp,anObj.CoeffDiffOppose2Pi());
    BinaryDumpInFile(aFp,anObj.CoeffDiffOpposePiS2());
    BinaryDumpInFile(aFp,anObj.CoeffDiffOppose2PiS2());
    BinaryDumpInFile(aFp,anObj.CodeBinaireCompl());
    BinaryDumpInFile(aFp,anObj.CodeBinaireIndex());
    BinaryDumpInFile(aFp,anObj.ImRad());
    BinaryDumpInFile(aFp,anObj.VectRho());
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
   aRes->AddFils(::ToXMLTree(std::string("CoeffRadiom"),anObj.CoeffRadiom())->ReTagThis("CoeffRadiom"));
   aRes->AddFils(::ToXMLTree(std::string("CoeffRadiom2"),anObj.CoeffRadiom2())->ReTagThis("CoeffRadiom2"));
   aRes->AddFils(::ToXMLTree(std::string("CoeffGradRadial"),anObj.CoeffGradRadial())->ReTagThis("CoeffGradRadial"));
   aRes->AddFils(::ToXMLTree(std::string("CoeffGradTangent"),anObj.CoeffGradTangent())->ReTagThis("CoeffGradTangent"));
   aRes->AddFils(::ToXMLTree(std::string("CoeffGradTangentPiS2"),anObj.CoeffGradTangentPiS2())->ReTagThis("CoeffGradTangentPiS2"));
   aRes->AddFils(::ToXMLTree(std::string("CoeffGradTangentPi"),anObj.CoeffGradTangentPi())->ReTagThis("CoeffGradTangentPi"));
   aRes->AddFils(::ToXMLTree(std::string("CoeffGradCroise"),anObj.CoeffGradCroise())->ReTagThis("CoeffGradCroise"));
   aRes->AddFils(::ToXMLTree(std::string("CoeffGradCroise2"),anObj.CoeffGradCroise2())->ReTagThis("CoeffGradCroise2"));
   aRes->AddFils(::ToXMLTree(std::string("CoeffDiffOpposePi"),anObj.CoeffDiffOpposePi())->ReTagThis("CoeffDiffOpposePi"));
   aRes->AddFils(::ToXMLTree(std::string("CoeffDiffOppose2Pi"),anObj.CoeffDiffOppose2Pi())->ReTagThis("CoeffDiffOppose2Pi"));
   aRes->AddFils(::ToXMLTree(std::string("CoeffDiffOpposePiS2"),anObj.CoeffDiffOpposePiS2())->ReTagThis("CoeffDiffOpposePiS2"));
   aRes->AddFils(::ToXMLTree(std::string("CoeffDiffOppose2PiS2"),anObj.CoeffDiffOppose2PiS2())->ReTagThis("CoeffDiffOppose2PiS2"));
   aRes->AddFils(::ToXMLTree(std::string("CodeBinaireCompl"),anObj.CodeBinaireCompl())->ReTagThis("CodeBinaireCompl"));
   aRes->AddFils(::ToXMLTree(std::string("CodeBinaireIndex"),anObj.CodeBinaireIndex())->ReTagThis("CodeBinaireIndex"));
   aRes->AddFils(::ToXMLTree(std::string("ImRad"),anObj.ImRad())->ReTagThis("ImRad"));
   aRes->AddFils(::ToXMLTree(std::string("VectRho"),anObj.VectRho())->ReTagThis("VectRho"));
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

   xml_init(anObj.CoeffRadiom(),aTree->Get("CoeffRadiom",1)); //tototo 

   xml_init(anObj.CoeffRadiom2(),aTree->Get("CoeffRadiom2",1)); //tototo 

   xml_init(anObj.CoeffGradRadial(),aTree->Get("CoeffGradRadial",1)); //tototo 

   xml_init(anObj.CoeffGradTangent(),aTree->Get("CoeffGradTangent",1)); //tototo 

   xml_init(anObj.CoeffGradTangentPiS2(),aTree->Get("CoeffGradTangentPiS2",1)); //tototo 

   xml_init(anObj.CoeffGradTangentPi(),aTree->Get("CoeffGradTangentPi",1)); //tototo 

   xml_init(anObj.CoeffGradCroise(),aTree->Get("CoeffGradCroise",1)); //tototo 

   xml_init(anObj.CoeffGradCroise2(),aTree->Get("CoeffGradCroise2",1)); //tototo 

   xml_init(anObj.CoeffDiffOpposePi(),aTree->Get("CoeffDiffOpposePi",1)); //tototo 

   xml_init(anObj.CoeffDiffOppose2Pi(),aTree->Get("CoeffDiffOppose2Pi",1)); //tototo 

   xml_init(anObj.CoeffDiffOpposePiS2(),aTree->Get("CoeffDiffOpposePiS2",1)); //tototo 

   xml_init(anObj.CoeffDiffOppose2PiS2(),aTree->Get("CoeffDiffOppose2PiS2",1)); //tototo 

   xml_init(anObj.CodeBinaireCompl(),aTree->Get("CodeBinaireCompl",1)); //tototo 

   xml_init(anObj.CodeBinaireIndex(),aTree->Get("CodeBinaireIndex",1)); //tototo 

   xml_init(anObj.ImRad(),aTree->Get("ImRad",1)); //tototo 

   xml_init(anObj.VectRho(),aTree->Get("VectRho",1)); //tototo 
}

std::string  Mangling( cOnePCarac *) {return "96E70BBB430BEF93FE3F";};


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

std::string  Mangling( cSetPCarac *) {return "09A2F3D4D5CE3CF4FD3F";};


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

// };
