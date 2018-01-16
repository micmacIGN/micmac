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


std::vector<double> & cOnePCarac::CoeffGradRadialF2()
{
   return mCoeffGradRadialF2;
}

const std::vector<double> & cOnePCarac::CoeffGradRadialF2()const 
{
   return mCoeffGradRadialF2;
}


std::vector<double> & cOnePCarac::CoeffGradTangent()
{
   return mCoeffGradTangent;
}

const std::vector<double> & cOnePCarac::CoeffGradTangent()const 
{
   return mCoeffGradTangent;
}


std::vector<double> & cOnePCarac::CoeffGradTangentPiS4()
{
   return mCoeffGradTangentPiS4;
}

const std::vector<double> & cOnePCarac::CoeffGradTangentPiS4()const 
{
   return mCoeffGradTangentPiS4;
}


std::vector<double> & cOnePCarac::CoeffGradTangentPiS2()
{
   return mCoeffGradTangentPiS2;
}

const std::vector<double> & cOnePCarac::CoeffGradTangentPiS2()const 
{
   return mCoeffGradTangentPiS2;
}

void  BinaryUnDumpFromFile(cOnePCarac & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Kind(),aFp);
    BinaryUnDumpFromFile(anObj.Pt(),aFp);
    BinaryUnDumpFromFile(anObj.NivScale(),aFp);
    BinaryUnDumpFromFile(anObj.Scale(),aFp);
    BinaryUnDumpFromFile(anObj.ScaleStab(),aFp);
    BinaryUnDumpFromFile(anObj.DirMS(),aFp);
    BinaryUnDumpFromFile(anObj.DirAC(),aFp);
    BinaryUnDumpFromFile(anObj.Contraste(),aFp);
    BinaryUnDumpFromFile(anObj.ContrasteRel(),aFp);
    BinaryUnDumpFromFile(anObj.AutoCorrel(),aFp);
    BinaryUnDumpFromFile(anObj.OK(),aFp);
    BinaryUnDumpFromFile(anObj.CoeffRadiom(),aFp);
    BinaryUnDumpFromFile(anObj.CoeffRadiom2(),aFp);
    BinaryUnDumpFromFile(anObj.CoeffGradRadial(),aFp);
    BinaryUnDumpFromFile(anObj.CoeffGradRadialF2(),aFp);
    BinaryUnDumpFromFile(anObj.CoeffGradTangent(),aFp);
    BinaryUnDumpFromFile(anObj.CoeffGradTangentPiS4(),aFp);
    BinaryUnDumpFromFile(anObj.CoeffGradTangentPiS2(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cOnePCarac & anObj)
{
    BinaryDumpInFile(aFp,anObj.Kind());
    BinaryDumpInFile(aFp,anObj.Pt());
    BinaryDumpInFile(aFp,anObj.NivScale());
    BinaryDumpInFile(aFp,anObj.Scale());
    BinaryDumpInFile(aFp,anObj.ScaleStab());
    BinaryDumpInFile(aFp,anObj.DirMS());
    BinaryDumpInFile(aFp,anObj.DirAC());
    BinaryDumpInFile(aFp,anObj.Contraste());
    BinaryDumpInFile(aFp,anObj.ContrasteRel());
    BinaryDumpInFile(aFp,anObj.AutoCorrel());
    BinaryDumpInFile(aFp,anObj.OK());
    BinaryDumpInFile(aFp,anObj.CoeffRadiom());
    BinaryDumpInFile(aFp,anObj.CoeffRadiom2());
    BinaryDumpInFile(aFp,anObj.CoeffGradRadial());
    BinaryDumpInFile(aFp,anObj.CoeffGradRadialF2());
    BinaryDumpInFile(aFp,anObj.CoeffGradTangent());
    BinaryDumpInFile(aFp,anObj.CoeffGradTangentPiS4());
    BinaryDumpInFile(aFp,anObj.CoeffGradTangentPiS2());
}

cElXMLTree * ToXMLTree(const cOnePCarac & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"OnePCarac",eXMLBranche);
   aRes->AddFils(ToXMLTree(std::string("Kind"),anObj.Kind())->ReTagThis("Kind"));
   aRes->AddFils(::ToXMLTree(std::string("Pt"),anObj.Pt())->ReTagThis("Pt"));
   aRes->AddFils(::ToXMLTree(std::string("NivScale"),anObj.NivScale())->ReTagThis("NivScale"));
   aRes->AddFils(::ToXMLTree(std::string("Scale"),anObj.Scale())->ReTagThis("Scale"));
   aRes->AddFils(::ToXMLTree(std::string("ScaleStab"),anObj.ScaleStab())->ReTagThis("ScaleStab"));
   aRes->AddFils(::ToXMLTree(std::string("DirMS"),anObj.DirMS())->ReTagThis("DirMS"));
   aRes->AddFils(::ToXMLTree(std::string("DirAC"),anObj.DirAC())->ReTagThis("DirAC"));
   aRes->AddFils(::ToXMLTree(std::string("Contraste"),anObj.Contraste())->ReTagThis("Contraste"));
   aRes->AddFils(::ToXMLTree(std::string("ContrasteRel"),anObj.ContrasteRel())->ReTagThis("ContrasteRel"));
   aRes->AddFils(::ToXMLTree(std::string("AutoCorrel"),anObj.AutoCorrel())->ReTagThis("AutoCorrel"));
   aRes->AddFils(::ToXMLTree(std::string("OK"),anObj.OK())->ReTagThis("OK"));
   aRes->AddFils(::ToXMLTree(std::string("CoeffRadiom"),anObj.CoeffRadiom())->ReTagThis("CoeffRadiom"));
   aRes->AddFils(::ToXMLTree(std::string("CoeffRadiom2"),anObj.CoeffRadiom2())->ReTagThis("CoeffRadiom2"));
   aRes->AddFils(::ToXMLTree(std::string("CoeffGradRadial"),anObj.CoeffGradRadial())->ReTagThis("CoeffGradRadial"));
   aRes->AddFils(::ToXMLTree(std::string("CoeffGradRadialF2"),anObj.CoeffGradRadialF2())->ReTagThis("CoeffGradRadialF2"));
   aRes->AddFils(::ToXMLTree(std::string("CoeffGradTangent"),anObj.CoeffGradTangent())->ReTagThis("CoeffGradTangent"));
   aRes->AddFils(::ToXMLTree(std::string("CoeffGradTangentPiS4"),anObj.CoeffGradTangentPiS4())->ReTagThis("CoeffGradTangentPiS4"));
   aRes->AddFils(::ToXMLTree(std::string("CoeffGradTangentPiS2"),anObj.CoeffGradTangentPiS2())->ReTagThis("CoeffGradTangentPiS2"));
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

   xml_init(anObj.NivScale(),aTree->Get("NivScale",1)); //tototo 

   xml_init(anObj.Scale(),aTree->Get("Scale",1)); //tototo 

   xml_init(anObj.ScaleStab(),aTree->Get("ScaleStab",1)); //tototo 

   xml_init(anObj.DirMS(),aTree->Get("DirMS",1)); //tototo 

   xml_init(anObj.DirAC(),aTree->Get("DirAC",1)); //tototo 

   xml_init(anObj.Contraste(),aTree->Get("Contraste",1)); //tototo 

   xml_init(anObj.ContrasteRel(),aTree->Get("ContrasteRel",1)); //tototo 

   xml_init(anObj.AutoCorrel(),aTree->Get("AutoCorrel",1)); //tototo 

   xml_init(anObj.OK(),aTree->Get("OK",1)); //tototo 

   xml_init(anObj.CoeffRadiom(),aTree->Get("CoeffRadiom",1)); //tototo 

   xml_init(anObj.CoeffRadiom2(),aTree->Get("CoeffRadiom2",1)); //tototo 

   xml_init(anObj.CoeffGradRadial(),aTree->Get("CoeffGradRadial",1)); //tototo 

   xml_init(anObj.CoeffGradRadialF2(),aTree->Get("CoeffGradRadialF2",1)); //tototo 

   xml_init(anObj.CoeffGradTangent(),aTree->Get("CoeffGradTangent",1)); //tototo 

   xml_init(anObj.CoeffGradTangentPiS4(),aTree->Get("CoeffGradTangentPiS4",1)); //tototo 

   xml_init(anObj.CoeffGradTangentPiS2(),aTree->Get("CoeffGradTangentPiS2",1)); //tototo 
}

std::string  Mangling( cOnePCarac *) {return "808DA0E0F0AE06D9F8BF";};


std::list< cOnePCarac > & cSetPCarac::OnePCarac()
{
   return mOnePCarac;
}

const std::list< cOnePCarac > & cSetPCarac::OnePCarac()const 
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
    for(  std::list< cOnePCarac >::const_iterator iT=anObj.OnePCarac().begin();
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
  (       std::list< cOnePCarac >::const_iterator it=anObj.OnePCarac().begin();
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

std::string  Mangling( cSetPCarac *) {return "3239841B4CCCDFA0FE3F";};

// };
