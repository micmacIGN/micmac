#include "StdAfx.h"
namespace NS_SaisiePts{
eTypePts  Str2eTypePts(const std::string & aName)
{
   if (aName=="eNSM_GeoCube")
      return eNSM_GeoCube;
   else if (aName=="eNSM_Plaquette")
      return eNSM_Plaquette;
   else if (aName=="eNSM_Pts")
      return eNSM_Pts;
   else if (aName=="eNSM_MaxLoc")
      return eNSM_MaxLoc;
   else if (aName=="eNSM_MinLoc")
      return eNSM_MinLoc;
   else if (aName=="eNSM_NonValue")
      return eNSM_NonValue;
  else
  {
      cout << aName << " is not a correct value for enum eTypePts\n" ;
      ELISE_ASSERT(false,"XML enum value error");
  }
  return (eTypePts) 0;
}
void xml_init(eTypePts & aVal,cElXMLTree * aTree)
{
   aVal= Str2eTypePts(aTree->Contenu());
}
std::string  eToString(const eTypePts & anObj)
{
   if (anObj==eNSM_GeoCube)
      return  "eNSM_GeoCube";
   if (anObj==eNSM_Plaquette)
      return  "eNSM_Plaquette";
   if (anObj==eNSM_Pts)
      return  "eNSM_Pts";
   if (anObj==eNSM_MaxLoc)
      return  "eNSM_MaxLoc";
   if (anObj==eNSM_MinLoc)
      return  "eNSM_MinLoc";
   if (anObj==eNSM_NonValue)
      return  "eNSM_NonValue";
 std::cout << "Enum = eTypePts\n";
   ELISE_ASSERT(false,"Bad Value in eToString for enum value ");
   return "";
}

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eTypePts & anObj)
{
      return  cElXMLTree::ValueNode(aNameTag,eToString(anObj));
}

eEtatPointeImage  Str2eEtatPointeImage(const std::string & aName)
{
   if (aName=="eEPI_NonSaisi")
      return eEPI_NonSaisi;
   else if (aName=="eEPI_Refute")
      return eEPI_Refute;
   else if (aName=="eEPI_Douteux")
      return eEPI_Douteux;
   else if (aName=="eEPI_Valide")
      return eEPI_Valide;
   else if (aName=="eEPI_NonValue")
      return eEPI_NonValue;
   else if (aName=="eEPI_Disparu")
      return eEPI_Disparu;
  else
  {
      cout << aName << " is not a correct value for enum eEtatPointeImage\n" ;
      ELISE_ASSERT(false,"XML enum value error");
  }
  return (eEtatPointeImage) 0;
}
void xml_init(eEtatPointeImage & aVal,cElXMLTree * aTree)
{
   aVal= Str2eEtatPointeImage(aTree->Contenu());
}
std::string  eToString(const eEtatPointeImage & anObj)
{
   if (anObj==eEPI_NonSaisi)
      return  "eEPI_NonSaisi";
   if (anObj==eEPI_Refute)
      return  "eEPI_Refute";
   if (anObj==eEPI_Douteux)
      return  "eEPI_Douteux";
   if (anObj==eEPI_Valide)
      return  "eEPI_Valide";
   if (anObj==eEPI_NonValue)
      return  "eEPI_NonValue";
   if (anObj==eEPI_Disparu)
      return  "eEPI_Disparu";
 std::cout << "Enum = eEtatPointeImage\n";
   ELISE_ASSERT(false,"Bad Value in eToString for enum value ");
   return "";
}

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eEtatPointeImage & anObj)
{
      return  cElXMLTree::ValueNode(aNameTag,eToString(anObj));
}


cTplValGesInit< std::string > & cContenuPt::None()
{
   return mNone;
}

const cTplValGesInit< std::string > & cContenuPt::None()const 
{
   return mNone;
}

cElXMLTree * ToXMLTree(const cContenuPt & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ContenuPt",eXMLBranche);
   if (anObj.None().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("None"),anObj.None().Val())->ReTagThis("None"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cContenuPt & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.None(),aTree->Get("None",1)); //tototo 
}


eTypePts & cPointGlob::Type()
{
   return mType;
}

const eTypePts & cPointGlob::Type()const 
{
   return mType;
}


std::string & cPointGlob::Name()
{
   return mName;
}

const std::string & cPointGlob::Name()const 
{
   return mName;
}


cTplValGesInit< Pt3dr > & cPointGlob::P3D()
{
   return mP3D;
}

const cTplValGesInit< Pt3dr > & cPointGlob::P3D()const 
{
   return mP3D;
}


cTplValGesInit< bool > & cPointGlob::Mes3DExportable()
{
   return mMes3DExportable;
}

const cTplValGesInit< bool > & cPointGlob::Mes3DExportable()const 
{
   return mMes3DExportable;
}


cTplValGesInit< Pt3dr > & cPointGlob::Incert()
{
   return mIncert;
}

const cTplValGesInit< Pt3dr > & cPointGlob::Incert()const 
{
   return mIncert;
}


cTplValGesInit< double > & cPointGlob::LargeurFlou()
{
   return mLargeurFlou;
}

const cTplValGesInit< double > & cPointGlob::LargeurFlou()const 
{
   return mLargeurFlou;
}


cTplValGesInit< std::string > & cPointGlob::None()
{
   return ContenuPt().Val().None();
}

const cTplValGesInit< std::string > & cPointGlob::None()const 
{
   return ContenuPt().Val().None();
}


cTplValGesInit< cContenuPt > & cPointGlob::ContenuPt()
{
   return mContenuPt;
}

const cTplValGesInit< cContenuPt > & cPointGlob::ContenuPt()const 
{
   return mContenuPt;
}


cTplValGesInit< int > & cPointGlob::NumAuto()
{
   return mNumAuto;
}

const cTplValGesInit< int > & cPointGlob::NumAuto()const 
{
   return mNumAuto;
}


cTplValGesInit< Pt3dr > & cPointGlob::PS1()
{
   return mPS1;
}

const cTplValGesInit< Pt3dr > & cPointGlob::PS1()const 
{
   return mPS1;
}


cTplValGesInit< Pt3dr > & cPointGlob::PS2()
{
   return mPS2;
}

const cTplValGesInit< Pt3dr > & cPointGlob::PS2()const 
{
   return mPS2;
}


cTplValGesInit< double > & cPointGlob::SzRech()
{
   return mSzRech;
}

const cTplValGesInit< double > & cPointGlob::SzRech()const 
{
   return mSzRech;
}


cTplValGesInit< bool > & cPointGlob::Disparu()
{
   return mDisparu;
}

const cTplValGesInit< bool > & cPointGlob::Disparu()const 
{
   return mDisparu;
}


cTplValGesInit< bool > & cPointGlob::FromDico()
{
   return mFromDico;
}

const cTplValGesInit< bool > & cPointGlob::FromDico()const 
{
   return mFromDico;
}

cElXMLTree * ToXMLTree(const cPointGlob & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"PointGlob",eXMLBranche);
   aRes->AddFils(ToXMLTree(std::string("Type"),anObj.Type())->ReTagThis("Type"));
   aRes->AddFils(::ToXMLTree(std::string("Name"),anObj.Name())->ReTagThis("Name"));
   if (anObj.P3D().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("P3D"),anObj.P3D().Val())->ReTagThis("P3D"));
   if (anObj.Mes3DExportable().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Mes3DExportable"),anObj.Mes3DExportable().Val())->ReTagThis("Mes3DExportable"));
   if (anObj.Incert().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Incert"),anObj.Incert().Val())->ReTagThis("Incert"));
   if (anObj.LargeurFlou().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("LargeurFlou"),anObj.LargeurFlou().Val())->ReTagThis("LargeurFlou"));
   if (anObj.ContenuPt().IsInit())
      aRes->AddFils(ToXMLTree(anObj.ContenuPt().Val())->ReTagThis("ContenuPt"));
   if (anObj.NumAuto().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NumAuto"),anObj.NumAuto().Val())->ReTagThis("NumAuto"));
   if (anObj.PS1().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PS1"),anObj.PS1().Val())->ReTagThis("PS1"));
   if (anObj.PS2().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PS2"),anObj.PS2().Val())->ReTagThis("PS2"));
   if (anObj.SzRech().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SzRech"),anObj.SzRech().Val())->ReTagThis("SzRech"));
   if (anObj.Disparu().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Disparu"),anObj.Disparu().Val())->ReTagThis("Disparu"));
   if (anObj.FromDico().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("FromDico"),anObj.FromDico().Val())->ReTagThis("FromDico"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cPointGlob & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Type(),aTree->Get("Type",1)); //tototo 

   xml_init(anObj.Name(),aTree->Get("Name",1)); //tototo 

   xml_init(anObj.P3D(),aTree->Get("P3D",1)); //tototo 

   xml_init(anObj.Mes3DExportable(),aTree->Get("Mes3DExportable",1)); //tototo 

   xml_init(anObj.Incert(),aTree->Get("Incert",1)); //tototo 

   xml_init(anObj.LargeurFlou(),aTree->Get("LargeurFlou",1),double(0.0)); //tototo 

   xml_init(anObj.ContenuPt(),aTree->Get("ContenuPt",1)); //tototo 

   xml_init(anObj.NumAuto(),aTree->Get("NumAuto",1)); //tototo 

   xml_init(anObj.PS1(),aTree->Get("PS1",1)); //tototo 

   xml_init(anObj.PS2(),aTree->Get("PS2",1)); //tototo 

   xml_init(anObj.SzRech(),aTree->Get("SzRech",1)); //tototo 

   xml_init(anObj.Disparu(),aTree->Get("Disparu",1)); //tototo 

   xml_init(anObj.FromDico(),aTree->Get("FromDico",1)); //tototo 
}


std::list< cPointGlob > & cSetPointGlob::PointGlob()
{
   return mPointGlob;
}

const std::list< cPointGlob > & cSetPointGlob::PointGlob()const 
{
   return mPointGlob;
}

cElXMLTree * ToXMLTree(const cSetPointGlob & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"SetPointGlob",eXMLBranche);
  for
  (       std::list< cPointGlob >::const_iterator it=anObj.PointGlob().begin();
      it !=anObj.PointGlob().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("PointGlob"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cSetPointGlob & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.PointGlob(),aTree->GetAll("PointGlob",false,1));
}


eEtatPointeImage & cOneSaisie::Etat()
{
   return mEtat;
}

const eEtatPointeImage & cOneSaisie::Etat()const 
{
   return mEtat;
}


std::string & cOneSaisie::NamePt()
{
   return mNamePt;
}

const std::string & cOneSaisie::NamePt()const 
{
   return mNamePt;
}


Pt2dr & cOneSaisie::PtIm()
{
   return mPtIm;
}

const Pt2dr & cOneSaisie::PtIm()const 
{
   return mPtIm;
}

cElXMLTree * ToXMLTree(const cOneSaisie & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"OneSaisie",eXMLBranche);
   aRes->AddFils(ToXMLTree(std::string("Etat"),anObj.Etat())->ReTagThis("Etat"));
   aRes->AddFils(::ToXMLTree(std::string("NamePt"),anObj.NamePt())->ReTagThis("NamePt"));
   aRes->AddFils(::ToXMLTree(std::string("PtIm"),anObj.PtIm())->ReTagThis("PtIm"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cOneSaisie & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Etat(),aTree->Get("Etat",1)); //tototo 

   xml_init(anObj.NamePt(),aTree->Get("NamePt",1)); //tototo 

   xml_init(anObj.PtIm(),aTree->Get("PtIm",1)); //tototo 
}


std::string & cSaisiePointeIm::NameIm()
{
   return mNameIm;
}

const std::string & cSaisiePointeIm::NameIm()const 
{
   return mNameIm;
}


std::list< cOneSaisie > & cSaisiePointeIm::OneSaisie()
{
   return mOneSaisie;
}

const std::list< cOneSaisie > & cSaisiePointeIm::OneSaisie()const 
{
   return mOneSaisie;
}

cElXMLTree * ToXMLTree(const cSaisiePointeIm & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"SaisiePointeIm",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("NameIm"),anObj.NameIm())->ReTagThis("NameIm"));
  for
  (       std::list< cOneSaisie >::const_iterator it=anObj.OneSaisie().begin();
      it !=anObj.OneSaisie().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("OneSaisie"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cSaisiePointeIm & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.NameIm(),aTree->Get("NameIm",1)); //tototo 

   xml_init(anObj.OneSaisie(),aTree->GetAll("OneSaisie",false,1));
}


std::list< cSaisiePointeIm > & cSetOfSaisiePointeIm::SaisiePointeIm()
{
   return mSaisiePointeIm;
}

const std::list< cSaisiePointeIm > & cSetOfSaisiePointeIm::SaisiePointeIm()const 
{
   return mSaisiePointeIm;
}

cElXMLTree * ToXMLTree(const cSetOfSaisiePointeIm & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"SetOfSaisiePointeIm",eXMLBranche);
  for
  (       std::list< cSaisiePointeIm >::const_iterator it=anObj.SaisiePointeIm().begin();
      it !=anObj.SaisiePointeIm().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("SaisiePointeIm"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cSetOfSaisiePointeIm & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.SaisiePointeIm(),aTree->GetAll("SaisiePointeIm",false,1));
}


cTplValGesInit< Pt2di > & cSectionWindows::SzTotIm()
{
   return mSzTotIm;
}

const cTplValGesInit< Pt2di > & cSectionWindows::SzTotIm()const 
{
   return mSzTotIm;
}


cTplValGesInit< Pt2di > & cSectionWindows::NbFenIm()
{
   return mNbFenIm;
}

const cTplValGesInit< Pt2di > & cSectionWindows::NbFenIm()const 
{
   return mNbFenIm;
}


cTplValGesInit< Pt2di > & cSectionWindows::SzWZ()
{
   return mSzWZ;
}

const cTplValGesInit< Pt2di > & cSectionWindows::SzWZ()const 
{
   return mSzWZ;
}


cTplValGesInit< bool > & cSectionWindows::ShowDet()
{
   return mShowDet;
}

const cTplValGesInit< bool > & cSectionWindows::ShowDet()const 
{
   return mShowDet;
}


cTplValGesInit< bool > & cSectionWindows::RefInvis()
{
   return mRefInvis;
}

const cTplValGesInit< bool > & cSectionWindows::RefInvis()const 
{
   return mRefInvis;
}

cElXMLTree * ToXMLTree(const cSectionWindows & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"SectionWindows",eXMLBranche);
   if (anObj.SzTotIm().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SzTotIm"),anObj.SzTotIm().Val())->ReTagThis("SzTotIm"));
   if (anObj.NbFenIm().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NbFenIm"),anObj.NbFenIm().Val())->ReTagThis("NbFenIm"));
   if (anObj.SzWZ().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SzWZ"),anObj.SzWZ().Val())->ReTagThis("SzWZ"));
   if (anObj.ShowDet().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ShowDet"),anObj.ShowDet().Val())->ReTagThis("ShowDet"));
   if (anObj.RefInvis().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("RefInvis"),anObj.RefInvis().Val())->ReTagThis("RefInvis"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cSectionWindows & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.SzTotIm(),aTree->Get("SzTotIm",1),Pt2di(Pt2di(700,600))); //tototo 

   xml_init(anObj.NbFenIm(),aTree->Get("NbFenIm",1),Pt2di(Pt2di(2,2))); //tototo 

   xml_init(anObj.SzWZ(),aTree->Get("SzWZ",1)); //tototo 

   xml_init(anObj.ShowDet(),aTree->Get("ShowDet",1),bool(false)); //tototo 

   xml_init(anObj.RefInvis(),aTree->Get("RefInvis",1),bool(false)); //tototo 
}


eTypePts & cImportFromDico::TypePt()
{
   return mTypePt;
}

const eTypePts & cImportFromDico::TypePt()const 
{
   return mTypePt;
}


std::string & cImportFromDico::File()
{
   return mFile;
}

const std::string & cImportFromDico::File()const 
{
   return mFile;
}


cTplValGesInit< double > & cImportFromDico::LargeurFlou()
{
   return mLargeurFlou;
}

const cTplValGesInit< double > & cImportFromDico::LargeurFlou()const 
{
   return mLargeurFlou;
}

cElXMLTree * ToXMLTree(const cImportFromDico & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ImportFromDico",eXMLBranche);
   aRes->AddFils(ToXMLTree(std::string("TypePt"),anObj.TypePt())->ReTagThis("TypePt"));
   aRes->AddFils(::ToXMLTree(std::string("File"),anObj.File())->ReTagThis("File"));
   if (anObj.LargeurFlou().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("LargeurFlou"),anObj.LargeurFlou().Val())->ReTagThis("LargeurFlou"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cImportFromDico & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.TypePt(),aTree->Get("TypePt",1)); //tototo 

   xml_init(anObj.File(),aTree->Get("File",1)); //tototo 

   xml_init(anObj.LargeurFlou(),aTree->Get("LargeurFlou",1),double(0.0)); //tototo 
}


cTplValGesInit< std::string > & cSectionInOut::Prefix2Add2IdPt()
{
   return mPrefix2Add2IdPt;
}

const cTplValGesInit< std::string > & cSectionInOut::Prefix2Add2IdPt()const 
{
   return mPrefix2Add2IdPt;
}


std::list< cImportFromDico > & cSectionInOut::ImportFromDico()
{
   return mImportFromDico;
}

const std::list< cImportFromDico > & cSectionInOut::ImportFromDico()const 
{
   return mImportFromDico;
}


cTplValGesInit< bool > & cSectionInOut::FlouGlobEcras()
{
   return mFlouGlobEcras;
}

const cTplValGesInit< bool > & cSectionInOut::FlouGlobEcras()const 
{
   return mFlouGlobEcras;
}


cTplValGesInit< bool > & cSectionInOut::TypeGlobEcras()
{
   return mTypeGlobEcras;
}

const cTplValGesInit< bool > & cSectionInOut::TypeGlobEcras()const 
{
   return mTypeGlobEcras;
}


cTplValGesInit< std::string > & cSectionInOut::NamePointesImage()
{
   return mNamePointesImage;
}

const cTplValGesInit< std::string > & cSectionInOut::NamePointesImage()const 
{
   return mNamePointesImage;
}


cTplValGesInit< std::string > & cSectionInOut::NamePointsGlobal()
{
   return mNamePointsGlobal;
}

const cTplValGesInit< std::string > & cSectionInOut::NamePointsGlobal()const 
{
   return mNamePointsGlobal;
}


cTplValGesInit< std::string > & cSectionInOut::ExportPointeImage()
{
   return mExportPointeImage;
}

const cTplValGesInit< std::string > & cSectionInOut::ExportPointeImage()const 
{
   return mExportPointeImage;
}


std::list< std::string > & cSectionInOut::FixedName()
{
   return mFixedName;
}

const std::list< std::string > & cSectionInOut::FixedName()const 
{
   return mFixedName;
}


cTplValGesInit< std::string > & cSectionInOut::NameAuto()
{
   return mNameAuto;
}

const cTplValGesInit< std::string > & cSectionInOut::NameAuto()const 
{
   return mNameAuto;
}


cTplValGesInit< bool > & cSectionInOut::EnterName()
{
   return mEnterName;
}

const cTplValGesInit< bool > & cSectionInOut::EnterName()const 
{
   return mEnterName;
}

cElXMLTree * ToXMLTree(const cSectionInOut & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"SectionInOut",eXMLBranche);
   if (anObj.Prefix2Add2IdPt().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Prefix2Add2IdPt"),anObj.Prefix2Add2IdPt().Val())->ReTagThis("Prefix2Add2IdPt"));
  for
  (       std::list< cImportFromDico >::const_iterator it=anObj.ImportFromDico().begin();
      it !=anObj.ImportFromDico().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("ImportFromDico"));
   if (anObj.FlouGlobEcras().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("FlouGlobEcras"),anObj.FlouGlobEcras().Val())->ReTagThis("FlouGlobEcras"));
   if (anObj.TypeGlobEcras().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("TypeGlobEcras"),anObj.TypeGlobEcras().Val())->ReTagThis("TypeGlobEcras"));
   if (anObj.NamePointesImage().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NamePointesImage"),anObj.NamePointesImage().Val())->ReTagThis("NamePointesImage"));
   if (anObj.NamePointsGlobal().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NamePointsGlobal"),anObj.NamePointsGlobal().Val())->ReTagThis("NamePointsGlobal"));
   if (anObj.ExportPointeImage().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ExportPointeImage"),anObj.ExportPointeImage().Val())->ReTagThis("ExportPointeImage"));
  for
  (       std::list< std::string >::const_iterator it=anObj.FixedName().begin();
      it !=anObj.FixedName().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("FixedName"),(*it))->ReTagThis("FixedName"));
   if (anObj.NameAuto().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NameAuto"),anObj.NameAuto().Val())->ReTagThis("NameAuto"));
   if (anObj.EnterName().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("EnterName"),anObj.EnterName().Val())->ReTagThis("EnterName"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cSectionInOut & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Prefix2Add2IdPt(),aTree->Get("Prefix2Add2IdPt",1),std::string("")); //tototo 

   xml_init(anObj.ImportFromDico(),aTree->GetAll("ImportFromDico",false,1));

   xml_init(anObj.FlouGlobEcras(),aTree->Get("FlouGlobEcras",1),bool(false)); //tototo 

   xml_init(anObj.TypeGlobEcras(),aTree->Get("TypeGlobEcras",1),bool(false)); //tototo 

   xml_init(anObj.NamePointesImage(),aTree->Get("NamePointesImage",1),std::string("SP_PointesImageIm.xml")); //tototo 

   xml_init(anObj.NamePointsGlobal(),aTree->Get("NamePointsGlobal",1),std::string("SP_PointesGlobal.xml")); //tototo 

   xml_init(anObj.ExportPointeImage(),aTree->Get("ExportPointeImage",1)); //tototo 

   xml_init(anObj.FixedName(),aTree->GetAll("FixedName",false,1));

   xml_init(anObj.NameAuto(),aTree->Get("NameAuto",1),std::string("NONE")); //tototo 

   xml_init(anObj.EnterName(),aTree->Get("EnterName",1),bool(false)); //tototo 
}


std::string & cSectionImages::SetOfImages()
{
   return mSetOfImages;
}

const std::string & cSectionImages::SetOfImages()const 
{
   return mSetOfImages;
}


cTplValGesInit< bool > & cSectionImages::ForceGray()
{
   return mForceGray;
}

const cTplValGesInit< bool > & cSectionImages::ForceGray()const 
{
   return mForceGray;
}


cTplValGesInit< std::string > & cSectionImages::KeyAssocOri()
{
   return mKeyAssocOri;
}

const cTplValGesInit< std::string > & cSectionImages::KeyAssocOri()const 
{
   return mKeyAssocOri;
}

cElXMLTree * ToXMLTree(const cSectionImages & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"SectionImages",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("SetOfImages"),anObj.SetOfImages())->ReTagThis("SetOfImages"));
   if (anObj.ForceGray().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ForceGray"),anObj.ForceGray().Val())->ReTagThis("ForceGray"));
   if (anObj.KeyAssocOri().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("KeyAssocOri"),anObj.KeyAssocOri().Val())->ReTagThis("KeyAssocOri"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cSectionImages & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.SetOfImages(),aTree->Get("SetOfImages",1)); //tototo 

   xml_init(anObj.ForceGray(),aTree->Get("ForceGray",1),bool(false)); //tototo 

   xml_init(anObj.KeyAssocOri(),aTree->Get("KeyAssocOri",1)); //tototo 
}


cTplValGesInit< double > & cProfEstimator::ZMoyen()
{
   return mZMoyen;
}

const cTplValGesInit< double > & cProfEstimator::ZMoyen()const 
{
   return mZMoyen;
}


cTplValGesInit< std::string > & cProfEstimator::ZMoyenInIma()
{
   return mZMoyenInIma;
}

const cTplValGesInit< std::string > & cProfEstimator::ZMoyenInIma()const 
{
   return mZMoyenInIma;
}

cElXMLTree * ToXMLTree(const cProfEstimator & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ProfEstimator",eXMLBranche);
   if (anObj.ZMoyen().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ZMoyen"),anObj.ZMoyen().Val())->ReTagThis("ZMoyen"));
   if (anObj.ZMoyenInIma().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ZMoyenInIma"),anObj.ZMoyenInIma().Val())->ReTagThis("ZMoyenInIma"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cProfEstimator & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.ZMoyen(),aTree->Get("ZMoyen",1)); //tototo 

   xml_init(anObj.ZMoyenInIma(),aTree->Get("ZMoyenInIma",1)); //tototo 
}


cTplValGesInit< double > & cSectionTerrain::IntervPercProf()
{
   return mIntervPercProf;
}

const cTplValGesInit< double > & cSectionTerrain::IntervPercProf()const 
{
   return mIntervPercProf;
}


cTplValGesInit< double > & cSectionTerrain::ZMoyen()
{
   return ProfEstimator().Val().ZMoyen();
}

const cTplValGesInit< double > & cSectionTerrain::ZMoyen()const 
{
   return ProfEstimator().Val().ZMoyen();
}


cTplValGesInit< std::string > & cSectionTerrain::ZMoyenInIma()
{
   return ProfEstimator().Val().ZMoyenInIma();
}

const cTplValGesInit< std::string > & cSectionTerrain::ZMoyenInIma()const 
{
   return ProfEstimator().Val().ZMoyenInIma();
}


cTplValGesInit< cProfEstimator > & cSectionTerrain::ProfEstimator()
{
   return mProfEstimator;
}

const cTplValGesInit< cProfEstimator > & cSectionTerrain::ProfEstimator()const 
{
   return mProfEstimator;
}

cElXMLTree * ToXMLTree(const cSectionTerrain & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"SectionTerrain",eXMLBranche);
   if (anObj.IntervPercProf().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("IntervPercProf"),anObj.IntervPercProf().Val())->ReTagThis("IntervPercProf"));
   if (anObj.ProfEstimator().IsInit())
      aRes->AddFils(ToXMLTree(anObj.ProfEstimator().Val())->ReTagThis("ProfEstimator"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cSectionTerrain & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.IntervPercProf(),aTree->Get("IntervPercProf",1),double(10.0)); //tototo 

   xml_init(anObj.ProfEstimator(),aTree->Get("ProfEstimator",1)); //tototo 
}


cTplValGesInit< cChantierDescripteur > & cParamSaisiePts::DicoLoc()
{
   return mDicoLoc;
}

const cTplValGesInit< cChantierDescripteur > & cParamSaisiePts::DicoLoc()const 
{
   return mDicoLoc;
}


cTplValGesInit< Pt2di > & cParamSaisiePts::SzTotIm()
{
   return SectionWindows().SzTotIm();
}

const cTplValGesInit< Pt2di > & cParamSaisiePts::SzTotIm()const 
{
   return SectionWindows().SzTotIm();
}


cTplValGesInit< Pt2di > & cParamSaisiePts::NbFenIm()
{
   return SectionWindows().NbFenIm();
}

const cTplValGesInit< Pt2di > & cParamSaisiePts::NbFenIm()const 
{
   return SectionWindows().NbFenIm();
}


cTplValGesInit< Pt2di > & cParamSaisiePts::SzWZ()
{
   return SectionWindows().SzWZ();
}

const cTplValGesInit< Pt2di > & cParamSaisiePts::SzWZ()const 
{
   return SectionWindows().SzWZ();
}


cTplValGesInit< bool > & cParamSaisiePts::ShowDet()
{
   return SectionWindows().ShowDet();
}

const cTplValGesInit< bool > & cParamSaisiePts::ShowDet()const 
{
   return SectionWindows().ShowDet();
}


cTplValGesInit< bool > & cParamSaisiePts::RefInvis()
{
   return SectionWindows().RefInvis();
}

const cTplValGesInit< bool > & cParamSaisiePts::RefInvis()const 
{
   return SectionWindows().RefInvis();
}


cSectionWindows & cParamSaisiePts::SectionWindows()
{
   return mSectionWindows;
}

const cSectionWindows & cParamSaisiePts::SectionWindows()const 
{
   return mSectionWindows;
}


cTplValGesInit< std::string > & cParamSaisiePts::Prefix2Add2IdPt()
{
   return SectionInOut().Prefix2Add2IdPt();
}

const cTplValGesInit< std::string > & cParamSaisiePts::Prefix2Add2IdPt()const 
{
   return SectionInOut().Prefix2Add2IdPt();
}


std::list< cImportFromDico > & cParamSaisiePts::ImportFromDico()
{
   return SectionInOut().ImportFromDico();
}

const std::list< cImportFromDico > & cParamSaisiePts::ImportFromDico()const 
{
   return SectionInOut().ImportFromDico();
}


cTplValGesInit< bool > & cParamSaisiePts::FlouGlobEcras()
{
   return SectionInOut().FlouGlobEcras();
}

const cTplValGesInit< bool > & cParamSaisiePts::FlouGlobEcras()const 
{
   return SectionInOut().FlouGlobEcras();
}


cTplValGesInit< bool > & cParamSaisiePts::TypeGlobEcras()
{
   return SectionInOut().TypeGlobEcras();
}

const cTplValGesInit< bool > & cParamSaisiePts::TypeGlobEcras()const 
{
   return SectionInOut().TypeGlobEcras();
}


cTplValGesInit< std::string > & cParamSaisiePts::NamePointesImage()
{
   return SectionInOut().NamePointesImage();
}

const cTplValGesInit< std::string > & cParamSaisiePts::NamePointesImage()const 
{
   return SectionInOut().NamePointesImage();
}


cTplValGesInit< std::string > & cParamSaisiePts::NamePointsGlobal()
{
   return SectionInOut().NamePointsGlobal();
}

const cTplValGesInit< std::string > & cParamSaisiePts::NamePointsGlobal()const 
{
   return SectionInOut().NamePointsGlobal();
}


cTplValGesInit< std::string > & cParamSaisiePts::ExportPointeImage()
{
   return SectionInOut().ExportPointeImage();
}

const cTplValGesInit< std::string > & cParamSaisiePts::ExportPointeImage()const 
{
   return SectionInOut().ExportPointeImage();
}


std::list< std::string > & cParamSaisiePts::FixedName()
{
   return SectionInOut().FixedName();
}

const std::list< std::string > & cParamSaisiePts::FixedName()const 
{
   return SectionInOut().FixedName();
}


cTplValGesInit< std::string > & cParamSaisiePts::NameAuto()
{
   return SectionInOut().NameAuto();
}

const cTplValGesInit< std::string > & cParamSaisiePts::NameAuto()const 
{
   return SectionInOut().NameAuto();
}


cTplValGesInit< bool > & cParamSaisiePts::EnterName()
{
   return SectionInOut().EnterName();
}

const cTplValGesInit< bool > & cParamSaisiePts::EnterName()const 
{
   return SectionInOut().EnterName();
}


cSectionInOut & cParamSaisiePts::SectionInOut()
{
   return mSectionInOut;
}

const cSectionInOut & cParamSaisiePts::SectionInOut()const 
{
   return mSectionInOut;
}


std::string & cParamSaisiePts::SetOfImages()
{
   return SectionImages().SetOfImages();
}

const std::string & cParamSaisiePts::SetOfImages()const 
{
   return SectionImages().SetOfImages();
}


cTplValGesInit< bool > & cParamSaisiePts::ForceGray()
{
   return SectionImages().ForceGray();
}

const cTplValGesInit< bool > & cParamSaisiePts::ForceGray()const 
{
   return SectionImages().ForceGray();
}


cTplValGesInit< std::string > & cParamSaisiePts::KeyAssocOri()
{
   return SectionImages().KeyAssocOri();
}

const cTplValGesInit< std::string > & cParamSaisiePts::KeyAssocOri()const 
{
   return SectionImages().KeyAssocOri();
}


cSectionImages & cParamSaisiePts::SectionImages()
{
   return mSectionImages;
}

const cSectionImages & cParamSaisiePts::SectionImages()const 
{
   return mSectionImages;
}


cTplValGesInit< double > & cParamSaisiePts::IntervPercProf()
{
   return SectionTerrain().IntervPercProf();
}

const cTplValGesInit< double > & cParamSaisiePts::IntervPercProf()const 
{
   return SectionTerrain().IntervPercProf();
}


cTplValGesInit< double > & cParamSaisiePts::ZMoyen()
{
   return SectionTerrain().ProfEstimator().Val().ZMoyen();
}

const cTplValGesInit< double > & cParamSaisiePts::ZMoyen()const 
{
   return SectionTerrain().ProfEstimator().Val().ZMoyen();
}


cTplValGesInit< std::string > & cParamSaisiePts::ZMoyenInIma()
{
   return SectionTerrain().ProfEstimator().Val().ZMoyenInIma();
}

const cTplValGesInit< std::string > & cParamSaisiePts::ZMoyenInIma()const 
{
   return SectionTerrain().ProfEstimator().Val().ZMoyenInIma();
}


cTplValGesInit< cProfEstimator > & cParamSaisiePts::ProfEstimator()
{
   return SectionTerrain().ProfEstimator();
}

const cTplValGesInit< cProfEstimator > & cParamSaisiePts::ProfEstimator()const 
{
   return SectionTerrain().ProfEstimator();
}


cSectionTerrain & cParamSaisiePts::SectionTerrain()
{
   return mSectionTerrain;
}

const cSectionTerrain & cParamSaisiePts::SectionTerrain()const 
{
   return mSectionTerrain;
}


std::string & cParamSaisiePts::DirectoryChantier()
{
   return mDirectoryChantier;
}

const std::string & cParamSaisiePts::DirectoryChantier()const 
{
   return mDirectoryChantier;
}

cElXMLTree * ToXMLTree(const cParamSaisiePts & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ParamSaisiePts",eXMLBranche);
   if (anObj.DicoLoc().IsInit())
      aRes->AddFils(ToXMLTree(anObj.DicoLoc().Val())->ReTagThis("DicoLoc"));
   aRes->AddFils(ToXMLTree(anObj.SectionWindows())->ReTagThis("SectionWindows"));
   aRes->AddFils(ToXMLTree(anObj.SectionInOut())->ReTagThis("SectionInOut"));
   aRes->AddFils(ToXMLTree(anObj.SectionImages())->ReTagThis("SectionImages"));
   aRes->AddFils(ToXMLTree(anObj.SectionTerrain())->ReTagThis("SectionTerrain"));
   aRes->AddFils(::ToXMLTree(std::string("DirectoryChantier"),anObj.DirectoryChantier())->ReTagThis("DirectoryChantier"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cParamSaisiePts & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.DicoLoc(),aTree->Get("DicoLoc",1)); //tototo 

   xml_init(anObj.SectionWindows(),aTree->Get("SectionWindows",1)); //tototo 

   xml_init(anObj.SectionInOut(),aTree->Get("SectionInOut",1)); //tototo 

   xml_init(anObj.SectionImages(),aTree->Get("SectionImages",1)); //tototo 

   xml_init(anObj.SectionTerrain(),aTree->Get("SectionTerrain",1)); //tototo 

   xml_init(anObj.DirectoryChantier(),aTree->Get("DirectoryChantier",1)); //tototo 
}

};
