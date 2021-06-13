#include "StdAfx.h"
//
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

void  BinaryDumpInFile(ELISE_fp & aFp,const eTypePts & anObj)
{
   BinaryDumpInFile(aFp,int(anObj));
}

void  BinaryUnDumpFromFile(eTypePts & anObj,ELISE_fp & aFp)
{
   int aIVal;
   BinaryUnDumpFromFile(aIVal,aFp);
   anObj=(eTypePts) aIVal;
}

std::string  Mangling( eTypePts *) {return "50A137A783D946ABFC3F";};

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
   else if (aName=="eEPI_Highlight")
      return eEPI_Highlight;
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
   if (anObj==eEPI_Highlight)
      return  "eEPI_Highlight";
 std::cout << "Enum = eEtatPointeImage\n";
   ELISE_ASSERT(false,"Bad Value in eToString for enum value ");
   return "";
}

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eEtatPointeImage & anObj)
{
      return  cElXMLTree::ValueNode(aNameTag,eToString(anObj));
}

void  BinaryDumpInFile(ELISE_fp & aFp,const eEtatPointeImage & anObj)
{
   BinaryDumpInFile(aFp,int(anObj));
}

void  BinaryUnDumpFromFile(eEtatPointeImage & anObj,ELISE_fp & aFp)
{
   int aIVal;
   BinaryUnDumpFromFile(aIVal,aFp);
   anObj=(eEtatPointeImage) aIVal;
}

std::string  Mangling( eEtatPointeImage *) {return "D3AE890CE8E04AD0FE3F";};


cTplValGesInit< std::string > & cContenuPt::None()
{
   return mNone;
}

const cTplValGesInit< std::string > & cContenuPt::None()const 
{
   return mNone;
}

void  BinaryUnDumpFromFile(cContenuPt & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.None().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.None().ValForcedForUnUmp(),aFp);
        }
        else  anObj.None().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cContenuPt & anObj)
{
    BinaryDumpInFile(aFp,anObj.None().IsInit());
    if (anObj.None().IsInit()) BinaryDumpInFile(aFp,anObj.None().Val());
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
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.None(),aTree->Get("None",1)); //tototo 
}

std::string  Mangling( cContenuPt *) {return "0B725EF036B1C6BDFE3F";};


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


cTplValGesInit< Pt3dr > & cPointGlob::Pt3DFromDico()
{
   return mPt3DFromDico;
}

const cTplValGesInit< Pt3dr > & cPointGlob::Pt3DFromDico()const 
{
   return mPt3DFromDico;
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


cTplValGesInit< Pt3dr > & cPointGlob::Normale()
{
   return mNormale;
}

const cTplValGesInit< Pt3dr > & cPointGlob::Normale()const 
{
   return mNormale;
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


std::vector< Pt3dr > & cPointGlob::VPS()
{
   return mVPS;
}

const std::vector< Pt3dr > & cPointGlob::VPS()const 
{
   return mVPS;
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

void  BinaryUnDumpFromFile(cPointGlob & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Type(),aFp);
    BinaryUnDumpFromFile(anObj.Name(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.P3D().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.P3D().ValForcedForUnUmp(),aFp);
        }
        else  anObj.P3D().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Pt3DFromDico().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Pt3DFromDico().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Pt3DFromDico().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Mes3DExportable().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Mes3DExportable().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Mes3DExportable().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Incert().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Incert().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Incert().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Normale().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Normale().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Normale().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.LargeurFlou().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.LargeurFlou().ValForcedForUnUmp(),aFp);
        }
        else  anObj.LargeurFlou().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ContenuPt().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ContenuPt().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ContenuPt().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NumAuto().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NumAuto().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NumAuto().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.PS1().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.PS1().ValForcedForUnUmp(),aFp);
        }
        else  anObj.PS1().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.PS2().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.PS2().ValForcedForUnUmp(),aFp);
        }
        else  anObj.PS2().SetNoInit();
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             Pt3dr aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.VPS().push_back(aVal);
        }
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SzRech().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SzRech().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SzRech().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Disparu().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Disparu().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Disparu().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.FromDico().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.FromDico().ValForcedForUnUmp(),aFp);
        }
        else  anObj.FromDico().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cPointGlob & anObj)
{
    BinaryDumpInFile(aFp,anObj.Type());
    BinaryDumpInFile(aFp,anObj.Name());
    BinaryDumpInFile(aFp,anObj.P3D().IsInit());
    if (anObj.P3D().IsInit()) BinaryDumpInFile(aFp,anObj.P3D().Val());
    BinaryDumpInFile(aFp,anObj.Pt3DFromDico().IsInit());
    if (anObj.Pt3DFromDico().IsInit()) BinaryDumpInFile(aFp,anObj.Pt3DFromDico().Val());
    BinaryDumpInFile(aFp,anObj.Mes3DExportable().IsInit());
    if (anObj.Mes3DExportable().IsInit()) BinaryDumpInFile(aFp,anObj.Mes3DExportable().Val());
    BinaryDumpInFile(aFp,anObj.Incert().IsInit());
    if (anObj.Incert().IsInit()) BinaryDumpInFile(aFp,anObj.Incert().Val());
    BinaryDumpInFile(aFp,anObj.Normale().IsInit());
    if (anObj.Normale().IsInit()) BinaryDumpInFile(aFp,anObj.Normale().Val());
    BinaryDumpInFile(aFp,anObj.LargeurFlou().IsInit());
    if (anObj.LargeurFlou().IsInit()) BinaryDumpInFile(aFp,anObj.LargeurFlou().Val());
    BinaryDumpInFile(aFp,anObj.ContenuPt().IsInit());
    if (anObj.ContenuPt().IsInit()) BinaryDumpInFile(aFp,anObj.ContenuPt().Val());
    BinaryDumpInFile(aFp,anObj.NumAuto().IsInit());
    if (anObj.NumAuto().IsInit()) BinaryDumpInFile(aFp,anObj.NumAuto().Val());
    BinaryDumpInFile(aFp,anObj.PS1().IsInit());
    if (anObj.PS1().IsInit()) BinaryDumpInFile(aFp,anObj.PS1().Val());
    BinaryDumpInFile(aFp,anObj.PS2().IsInit());
    if (anObj.PS2().IsInit()) BinaryDumpInFile(aFp,anObj.PS2().Val());
    BinaryDumpInFile(aFp,(int)anObj.VPS().size());
    for(  std::vector< Pt3dr >::const_iterator iT=anObj.VPS().begin();
         iT!=anObj.VPS().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.SzRech().IsInit());
    if (anObj.SzRech().IsInit()) BinaryDumpInFile(aFp,anObj.SzRech().Val());
    BinaryDumpInFile(aFp,anObj.Disparu().IsInit());
    if (anObj.Disparu().IsInit()) BinaryDumpInFile(aFp,anObj.Disparu().Val());
    BinaryDumpInFile(aFp,anObj.FromDico().IsInit());
    if (anObj.FromDico().IsInit()) BinaryDumpInFile(aFp,anObj.FromDico().Val());
}

cElXMLTree * ToXMLTree(const cPointGlob & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"PointGlob",eXMLBranche);
   aRes->AddFils(ToXMLTree(std::string("Type"),anObj.Type())->ReTagThis("Type"));
   aRes->AddFils(::ToXMLTree(std::string("Name"),anObj.Name())->ReTagThis("Name"));
   if (anObj.P3D().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("P3D"),anObj.P3D().Val())->ReTagThis("P3D"));
   if (anObj.Pt3DFromDico().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Pt3DFromDico"),anObj.Pt3DFromDico().Val())->ReTagThis("Pt3DFromDico"));
   if (anObj.Mes3DExportable().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Mes3DExportable"),anObj.Mes3DExportable().Val())->ReTagThis("Mes3DExportable"));
   if (anObj.Incert().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Incert"),anObj.Incert().Val())->ReTagThis("Incert"));
   if (anObj.Normale().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Normale"),anObj.Normale().Val())->ReTagThis("Normale"));
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
  for
  (       std::vector< Pt3dr >::const_iterator it=anObj.VPS().begin();
      it !=anObj.VPS().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("VPS"),(*it))->ReTagThis("VPS"));
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
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.Type(),aTree->Get("Type",1)); //tototo 

   xml_init(anObj.Name(),aTree->Get("Name",1)); //tototo 

   xml_init(anObj.P3D(),aTree->Get("P3D",1)); //tototo 

   xml_init(anObj.Pt3DFromDico(),aTree->Get("Pt3DFromDico",1)); //tototo 

   xml_init(anObj.Mes3DExportable(),aTree->Get("Mes3DExportable",1)); //tototo 

   xml_init(anObj.Incert(),aTree->Get("Incert",1)); //tototo 

   xml_init(anObj.Normale(),aTree->Get("Normale",1)); //tototo 

   xml_init(anObj.LargeurFlou(),aTree->Get("LargeurFlou",1),double(0.0)); //tototo 

   xml_init(anObj.ContenuPt(),aTree->Get("ContenuPt",1)); //tototo 

   xml_init(anObj.NumAuto(),aTree->Get("NumAuto",1)); //tototo 

   xml_init(anObj.PS1(),aTree->Get("PS1",1)); //tototo 

   xml_init(anObj.PS2(),aTree->Get("PS2",1)); //tototo 

   xml_init(anObj.VPS(),aTree->GetAll("VPS",false,1));

   xml_init(anObj.SzRech(),aTree->Get("SzRech",1)); //tototo 

   xml_init(anObj.Disparu(),aTree->Get("Disparu",1)); //tototo 

   xml_init(anObj.FromDico(),aTree->Get("FromDico",1)); //tototo 
}

std::string  Mangling( cPointGlob *) {return "846910AB12572EDDFCBF";};


std::list< cPointGlob > & cSetPointGlob::PointGlob()
{
   return mPointGlob;
}

const std::list< cPointGlob > & cSetPointGlob::PointGlob()const 
{
   return mPointGlob;
}

void  BinaryUnDumpFromFile(cSetPointGlob & anObj,ELISE_fp & aFp)
{
   { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cPointGlob aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.PointGlob().push_back(aVal);
        }
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cSetPointGlob & anObj)
{
    BinaryDumpInFile(aFp,(int)anObj.PointGlob().size());
    for(  std::list< cPointGlob >::const_iterator iT=anObj.PointGlob().begin();
         iT!=anObj.PointGlob().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
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
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.PointGlob(),aTree->GetAll("PointGlob",false,1));
}

std::string  Mangling( cSetPointGlob *) {return "37436826261BF5A8FF3F";};


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

void  BinaryUnDumpFromFile(cOneSaisie & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Etat(),aFp);
    BinaryUnDumpFromFile(anObj.NamePt(),aFp);
    BinaryUnDumpFromFile(anObj.PtIm(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cOneSaisie & anObj)
{
    BinaryDumpInFile(aFp,anObj.Etat());
    BinaryDumpInFile(aFp,anObj.NamePt());
    BinaryDumpInFile(aFp,anObj.PtIm());
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
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.Etat(),aTree->Get("Etat",1)); //tototo 

   xml_init(anObj.NamePt(),aTree->Get("NamePt",1)); //tototo 

   xml_init(anObj.PtIm(),aTree->Get("PtIm",1)); //tototo 
}

std::string  Mangling( cOneSaisie *) {return "CC09723FBCE52CBBFC3F";};


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

void  BinaryUnDumpFromFile(cSaisiePointeIm & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.NameIm(),aFp);
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cOneSaisie aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.OneSaisie().push_back(aVal);
        }
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cSaisiePointeIm & anObj)
{
    BinaryDumpInFile(aFp,anObj.NameIm());
    BinaryDumpInFile(aFp,(int)anObj.OneSaisie().size());
    for(  std::list< cOneSaisie >::const_iterator iT=anObj.OneSaisie().begin();
         iT!=anObj.OneSaisie().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
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
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.NameIm(),aTree->Get("NameIm",1)); //tototo 

   xml_init(anObj.OneSaisie(),aTree->GetAll("OneSaisie",false,1));
}

std::string  Mangling( cSaisiePointeIm *) {return "94DC419067D1B1DCFC3F";};


std::list< cSaisiePointeIm > & cSetOfSaisiePointeIm::SaisiePointeIm()
{
   return mSaisiePointeIm;
}

const std::list< cSaisiePointeIm > & cSetOfSaisiePointeIm::SaisiePointeIm()const 
{
   return mSaisiePointeIm;
}

void  BinaryUnDumpFromFile(cSetOfSaisiePointeIm & anObj,ELISE_fp & aFp)
{
   { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cSaisiePointeIm aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.SaisiePointeIm().push_back(aVal);
        }
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cSetOfSaisiePointeIm & anObj)
{
    BinaryDumpInFile(aFp,(int)anObj.SaisiePointeIm().size());
    for(  std::list< cSaisiePointeIm >::const_iterator iT=anObj.SaisiePointeIm().begin();
         iT!=anObj.SaisiePointeIm().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
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
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.SaisiePointeIm(),aTree->GetAll("SaisiePointeIm",false,1));
}

std::string  Mangling( cSetOfSaisiePointeIm *) {return "BA84B0AA819A94E5FD3F";};


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


cTplValGesInit< bool > & cSectionWindows::UseMinMaxPt()
{
   return mUseMinMaxPt;
}

const cTplValGesInit< bool > & cSectionWindows::UseMinMaxPt()const 
{
   return mUseMinMaxPt;
}

void  BinaryUnDumpFromFile(cSectionWindows & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SzTotIm().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SzTotIm().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SzTotIm().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NbFenIm().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NbFenIm().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NbFenIm().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SzWZ().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SzWZ().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SzWZ().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ShowDet().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ShowDet().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ShowDet().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.RefInvis().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.RefInvis().ValForcedForUnUmp(),aFp);
        }
        else  anObj.RefInvis().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.UseMinMaxPt().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.UseMinMaxPt().ValForcedForUnUmp(),aFp);
        }
        else  anObj.UseMinMaxPt().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cSectionWindows & anObj)
{
    BinaryDumpInFile(aFp,anObj.SzTotIm().IsInit());
    if (anObj.SzTotIm().IsInit()) BinaryDumpInFile(aFp,anObj.SzTotIm().Val());
    BinaryDumpInFile(aFp,anObj.NbFenIm().IsInit());
    if (anObj.NbFenIm().IsInit()) BinaryDumpInFile(aFp,anObj.NbFenIm().Val());
    BinaryDumpInFile(aFp,anObj.SzWZ().IsInit());
    if (anObj.SzWZ().IsInit()) BinaryDumpInFile(aFp,anObj.SzWZ().Val());
    BinaryDumpInFile(aFp,anObj.ShowDet().IsInit());
    if (anObj.ShowDet().IsInit()) BinaryDumpInFile(aFp,anObj.ShowDet().Val());
    BinaryDumpInFile(aFp,anObj.RefInvis().IsInit());
    if (anObj.RefInvis().IsInit()) BinaryDumpInFile(aFp,anObj.RefInvis().Val());
    BinaryDumpInFile(aFp,anObj.UseMinMaxPt().IsInit());
    if (anObj.UseMinMaxPt().IsInit()) BinaryDumpInFile(aFp,anObj.UseMinMaxPt().Val());
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
   if (anObj.UseMinMaxPt().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("UseMinMaxPt"),anObj.UseMinMaxPt().Val())->ReTagThis("UseMinMaxPt"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cSectionWindows & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.SzTotIm(),aTree->Get("SzTotIm",1),Pt2di(Pt2di(700,600))); //tototo 

   xml_init(anObj.NbFenIm(),aTree->Get("NbFenIm",1),Pt2di(Pt2di(2,2))); //tototo 

   xml_init(anObj.SzWZ(),aTree->Get("SzWZ",1)); //tototo 

   xml_init(anObj.ShowDet(),aTree->Get("ShowDet",1),bool(false)); //tototo 

   xml_init(anObj.RefInvis(),aTree->Get("RefInvis",1),bool(false)); //tototo 

   xml_init(anObj.UseMinMaxPt(),aTree->Get("UseMinMaxPt",1),bool(false)); //tototo 
}

std::string  Mangling( cSectionWindows *) {return "E8949F9F81375EF7FDBF";};


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

void  BinaryUnDumpFromFile(cImportFromDico & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.TypePt(),aFp);
    BinaryUnDumpFromFile(anObj.File(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.LargeurFlou().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.LargeurFlou().ValForcedForUnUmp(),aFp);
        }
        else  anObj.LargeurFlou().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cImportFromDico & anObj)
{
    BinaryDumpInFile(aFp,anObj.TypePt());
    BinaryDumpInFile(aFp,anObj.File());
    BinaryDumpInFile(aFp,anObj.LargeurFlou().IsInit());
    if (anObj.LargeurFlou().IsInit()) BinaryDumpInFile(aFp,anObj.LargeurFlou().Val());
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
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.TypePt(),aTree->Get("TypePt",1)); //tototo 

   xml_init(anObj.File(),aTree->Get("File",1)); //tototo 

   xml_init(anObj.LargeurFlou(),aTree->Get("LargeurFlou",1),double(0.0)); //tototo 
}

std::string  Mangling( cImportFromDico *) {return "DCC85356BC2653D0FE3F";};


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


cTplValGesInit< std::string > & cSectionInOut::PatternNameInputsSec()
{
   return mPatternNameInputsSec;
}

const cTplValGesInit< std::string > & cSectionInOut::PatternNameInputsSec()const 
{
   return mPatternNameInputsSec;
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


cTplValGesInit< bool > & cSectionInOut::ForceExport3D()
{
   return mForceExport3D;
}

const cTplValGesInit< bool > & cSectionInOut::ForceExport3D()const 
{
   return mForceExport3D;
}

void  BinaryUnDumpFromFile(cSectionInOut & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Prefix2Add2IdPt().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Prefix2Add2IdPt().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Prefix2Add2IdPt().SetNoInit();
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cImportFromDico aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.ImportFromDico().push_back(aVal);
        }
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.FlouGlobEcras().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.FlouGlobEcras().ValForcedForUnUmp(),aFp);
        }
        else  anObj.FlouGlobEcras().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.TypeGlobEcras().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.TypeGlobEcras().ValForcedForUnUmp(),aFp);
        }
        else  anObj.TypeGlobEcras().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NamePointesImage().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NamePointesImage().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NamePointesImage().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NamePointsGlobal().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NamePointsGlobal().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NamePointsGlobal().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.PatternNameInputsSec().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.PatternNameInputsSec().ValForcedForUnUmp(),aFp);
        }
        else  anObj.PatternNameInputsSec().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ExportPointeImage().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ExportPointeImage().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ExportPointeImage().SetNoInit();
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             std::string aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.FixedName().push_back(aVal);
        }
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NameAuto().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NameAuto().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NameAuto().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.EnterName().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.EnterName().ValForcedForUnUmp(),aFp);
        }
        else  anObj.EnterName().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ForceExport3D().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ForceExport3D().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ForceExport3D().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cSectionInOut & anObj)
{
    BinaryDumpInFile(aFp,anObj.Prefix2Add2IdPt().IsInit());
    if (anObj.Prefix2Add2IdPt().IsInit()) BinaryDumpInFile(aFp,anObj.Prefix2Add2IdPt().Val());
    BinaryDumpInFile(aFp,(int)anObj.ImportFromDico().size());
    for(  std::list< cImportFromDico >::const_iterator iT=anObj.ImportFromDico().begin();
         iT!=anObj.ImportFromDico().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.FlouGlobEcras().IsInit());
    if (anObj.FlouGlobEcras().IsInit()) BinaryDumpInFile(aFp,anObj.FlouGlobEcras().Val());
    BinaryDumpInFile(aFp,anObj.TypeGlobEcras().IsInit());
    if (anObj.TypeGlobEcras().IsInit()) BinaryDumpInFile(aFp,anObj.TypeGlobEcras().Val());
    BinaryDumpInFile(aFp,anObj.NamePointesImage().IsInit());
    if (anObj.NamePointesImage().IsInit()) BinaryDumpInFile(aFp,anObj.NamePointesImage().Val());
    BinaryDumpInFile(aFp,anObj.NamePointsGlobal().IsInit());
    if (anObj.NamePointsGlobal().IsInit()) BinaryDumpInFile(aFp,anObj.NamePointsGlobal().Val());
    BinaryDumpInFile(aFp,anObj.PatternNameInputsSec().IsInit());
    if (anObj.PatternNameInputsSec().IsInit()) BinaryDumpInFile(aFp,anObj.PatternNameInputsSec().Val());
    BinaryDumpInFile(aFp,anObj.ExportPointeImage().IsInit());
    if (anObj.ExportPointeImage().IsInit()) BinaryDumpInFile(aFp,anObj.ExportPointeImage().Val());
    BinaryDumpInFile(aFp,(int)anObj.FixedName().size());
    for(  std::list< std::string >::const_iterator iT=anObj.FixedName().begin();
         iT!=anObj.FixedName().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.NameAuto().IsInit());
    if (anObj.NameAuto().IsInit()) BinaryDumpInFile(aFp,anObj.NameAuto().Val());
    BinaryDumpInFile(aFp,anObj.EnterName().IsInit());
    if (anObj.EnterName().IsInit()) BinaryDumpInFile(aFp,anObj.EnterName().Val());
    BinaryDumpInFile(aFp,anObj.ForceExport3D().IsInit());
    if (anObj.ForceExport3D().IsInit()) BinaryDumpInFile(aFp,anObj.ForceExport3D().Val());
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
   if (anObj.PatternNameInputsSec().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PatternNameInputsSec"),anObj.PatternNameInputsSec().Val())->ReTagThis("PatternNameInputsSec"));
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
   if (anObj.ForceExport3D().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ForceExport3D"),anObj.ForceExport3D().Val())->ReTagThis("ForceExport3D"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cSectionInOut & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.Prefix2Add2IdPt(),aTree->Get("Prefix2Add2IdPt",1),std::string("")); //tototo 

   xml_init(anObj.ImportFromDico(),aTree->GetAll("ImportFromDico",false,1));

   xml_init(anObj.FlouGlobEcras(),aTree->Get("FlouGlobEcras",1),bool(false)); //tototo 

   xml_init(anObj.TypeGlobEcras(),aTree->Get("TypeGlobEcras",1),bool(false)); //tototo 

   xml_init(anObj.NamePointesImage(),aTree->Get("NamePointesImage",1),std::string("SP_PointesImageIm.xml")); //tototo 

   xml_init(anObj.NamePointsGlobal(),aTree->Get("NamePointsGlobal",1),std::string("SP_PointesGlobal.xml")); //tototo 

   xml_init(anObj.PatternNameInputsSec(),aTree->Get("PatternNameInputsSec",1)); //tototo 

   xml_init(anObj.ExportPointeImage(),aTree->Get("ExportPointeImage",1)); //tototo 

   xml_init(anObj.FixedName(),aTree->GetAll("FixedName",false,1));

   xml_init(anObj.NameAuto(),aTree->Get("NameAuto",1),std::string("NONE")); //tototo 

   xml_init(anObj.EnterName(),aTree->Get("EnterName",1),bool(false)); //tototo 

   xml_init(anObj.ForceExport3D(),aTree->Get("ForceExport3D",1),bool(true)); //tototo 
}

std::string  Mangling( cSectionInOut *) {return "79EC78A33724A2A8FE3F";};


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


cTplValGesInit< double > & cSectionImages::Gama()
{
   return mGama;
}

const cTplValGesInit< double > & cSectionImages::Gama()const 
{
   return mGama;
}


cTplValGesInit< std::string > & cSectionImages::KeyAssocOri()
{
   return mKeyAssocOri;
}

const cTplValGesInit< std::string > & cSectionImages::KeyAssocOri()const 
{
   return mKeyAssocOri;
}

void  BinaryUnDumpFromFile(cSectionImages & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.SetOfImages(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ForceGray().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ForceGray().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ForceGray().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Gama().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Gama().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Gama().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.KeyAssocOri().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.KeyAssocOri().ValForcedForUnUmp(),aFp);
        }
        else  anObj.KeyAssocOri().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cSectionImages & anObj)
{
    BinaryDumpInFile(aFp,anObj.SetOfImages());
    BinaryDumpInFile(aFp,anObj.ForceGray().IsInit());
    if (anObj.ForceGray().IsInit()) BinaryDumpInFile(aFp,anObj.ForceGray().Val());
    BinaryDumpInFile(aFp,anObj.Gama().IsInit());
    if (anObj.Gama().IsInit()) BinaryDumpInFile(aFp,anObj.Gama().Val());
    BinaryDumpInFile(aFp,anObj.KeyAssocOri().IsInit());
    if (anObj.KeyAssocOri().IsInit()) BinaryDumpInFile(aFp,anObj.KeyAssocOri().Val());
}

cElXMLTree * ToXMLTree(const cSectionImages & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"SectionImages",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("SetOfImages"),anObj.SetOfImages())->ReTagThis("SetOfImages"));
   if (anObj.ForceGray().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ForceGray"),anObj.ForceGray().Val())->ReTagThis("ForceGray"));
   if (anObj.Gama().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Gama"),anObj.Gama().Val())->ReTagThis("Gama"));
   if (anObj.KeyAssocOri().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("KeyAssocOri"),anObj.KeyAssocOri().Val())->ReTagThis("KeyAssocOri"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cSectionImages & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.SetOfImages(),aTree->Get("SetOfImages",1)); //tototo 

   xml_init(anObj.ForceGray(),aTree->Get("ForceGray",1),bool(false)); //tototo 

   xml_init(anObj.Gama(),aTree->Get("Gama",1),double(1.0)); //tototo 

   xml_init(anObj.KeyAssocOri(),aTree->Get("KeyAssocOri",1)); //tototo 
}

std::string  Mangling( cSectionImages *) {return "0C3D8CEAD0B8F4BDFE3F";};


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

void  BinaryUnDumpFromFile(cProfEstimator & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ZMoyen().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ZMoyen().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ZMoyen().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ZMoyenInIma().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ZMoyenInIma().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ZMoyenInIma().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cProfEstimator & anObj)
{
    BinaryDumpInFile(aFp,anObj.ZMoyen().IsInit());
    if (anObj.ZMoyen().IsInit()) BinaryDumpInFile(aFp,anObj.ZMoyen().Val());
    BinaryDumpInFile(aFp,anObj.ZMoyenInIma().IsInit());
    if (anObj.ZMoyenInIma().IsInit()) BinaryDumpInFile(aFp,anObj.ZMoyenInIma().Val());
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
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.ZMoyen(),aTree->Get("ZMoyen",1)); //tototo 

   xml_init(anObj.ZMoyenInIma(),aTree->Get("ZMoyenInIma",1)); //tototo 
}

std::string  Mangling( cProfEstimator *) {return "C91AEA13F187C790FF3F";};


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


cTplValGesInit< std::string > & cSectionTerrain::Masq3DFilterVis()
{
   return mMasq3DFilterVis;
}

const cTplValGesInit< std::string > & cSectionTerrain::Masq3DFilterVis()const 
{
   return mMasq3DFilterVis;
}


cTplValGesInit< std::string > & cSectionTerrain::PIMsFilterVis()
{
   return mPIMsFilterVis;
}

const cTplValGesInit< std::string > & cSectionTerrain::PIMsFilterVis()const 
{
   return mPIMsFilterVis;
}


cTplValGesInit< double > & cSectionTerrain::DistMaxVisib()
{
   return mDistMaxVisib;
}

const cTplValGesInit< double > & cSectionTerrain::DistMaxVisib()const 
{
   return mDistMaxVisib;
}


cTplValGesInit< cElRegex_Ptr > & cSectionTerrain::PatternNamePtsVisib()
{
   return mPatternNamePtsVisib;
}

const cTplValGesInit< cElRegex_Ptr > & cSectionTerrain::PatternNamePtsVisib()const 
{
   return mPatternNamePtsVisib;
}

void  BinaryUnDumpFromFile(cSectionTerrain & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.IntervPercProf().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.IntervPercProf().ValForcedForUnUmp(),aFp);
        }
        else  anObj.IntervPercProf().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ProfEstimator().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ProfEstimator().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ProfEstimator().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Masq3DFilterVis().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Masq3DFilterVis().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Masq3DFilterVis().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.PIMsFilterVis().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.PIMsFilterVis().ValForcedForUnUmp(),aFp);
        }
        else  anObj.PIMsFilterVis().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.DistMaxVisib().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.DistMaxVisib().ValForcedForUnUmp(),aFp);
        }
        else  anObj.DistMaxVisib().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.PatternNamePtsVisib().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.PatternNamePtsVisib().ValForcedForUnUmp(),aFp);
        }
        else  anObj.PatternNamePtsVisib().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cSectionTerrain & anObj)
{
    BinaryDumpInFile(aFp,anObj.IntervPercProf().IsInit());
    if (anObj.IntervPercProf().IsInit()) BinaryDumpInFile(aFp,anObj.IntervPercProf().Val());
    BinaryDumpInFile(aFp,anObj.ProfEstimator().IsInit());
    if (anObj.ProfEstimator().IsInit()) BinaryDumpInFile(aFp,anObj.ProfEstimator().Val());
    BinaryDumpInFile(aFp,anObj.Masq3DFilterVis().IsInit());
    if (anObj.Masq3DFilterVis().IsInit()) BinaryDumpInFile(aFp,anObj.Masq3DFilterVis().Val());
    BinaryDumpInFile(aFp,anObj.PIMsFilterVis().IsInit());
    if (anObj.PIMsFilterVis().IsInit()) BinaryDumpInFile(aFp,anObj.PIMsFilterVis().Val());
    BinaryDumpInFile(aFp,anObj.DistMaxVisib().IsInit());
    if (anObj.DistMaxVisib().IsInit()) BinaryDumpInFile(aFp,anObj.DistMaxVisib().Val());
    BinaryDumpInFile(aFp,anObj.PatternNamePtsVisib().IsInit());
    if (anObj.PatternNamePtsVisib().IsInit()) BinaryDumpInFile(aFp,anObj.PatternNamePtsVisib().Val());
}

cElXMLTree * ToXMLTree(const cSectionTerrain & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"SectionTerrain",eXMLBranche);
   if (anObj.IntervPercProf().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("IntervPercProf"),anObj.IntervPercProf().Val())->ReTagThis("IntervPercProf"));
   if (anObj.ProfEstimator().IsInit())
      aRes->AddFils(ToXMLTree(anObj.ProfEstimator().Val())->ReTagThis("ProfEstimator"));
   if (anObj.Masq3DFilterVis().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Masq3DFilterVis"),anObj.Masq3DFilterVis().Val())->ReTagThis("Masq3DFilterVis"));
   if (anObj.PIMsFilterVis().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PIMsFilterVis"),anObj.PIMsFilterVis().Val())->ReTagThis("PIMsFilterVis"));
   if (anObj.DistMaxVisib().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DistMaxVisib"),anObj.DistMaxVisib().Val())->ReTagThis("DistMaxVisib"));
   if (anObj.PatternNamePtsVisib().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PatternNamePtsVisib"),anObj.PatternNamePtsVisib().Val())->ReTagThis("PatternNamePtsVisib"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cSectionTerrain & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.IntervPercProf(),aTree->Get("IntervPercProf",1),double(10.0)); //tototo 

   xml_init(anObj.ProfEstimator(),aTree->Get("ProfEstimator",1)); //tototo 

   xml_init(anObj.Masq3DFilterVis(),aTree->Get("Masq3DFilterVis",1)); //tototo 

   xml_init(anObj.PIMsFilterVis(),aTree->Get("PIMsFilterVis",1)); //tototo 

   xml_init(anObj.DistMaxVisib(),aTree->Get("DistMaxVisib",1)); //tototo 

   xml_init(anObj.PatternNamePtsVisib(),aTree->Get("PatternNamePtsVisib",1)); //tototo 
}

std::string  Mangling( cSectionTerrain *) {return "3CBBC271C12BE8E6FC3F";};


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


cTplValGesInit< bool > & cParamSaisiePts::UseMinMaxPt()
{
   return SectionWindows().UseMinMaxPt();
}

const cTplValGesInit< bool > & cParamSaisiePts::UseMinMaxPt()const 
{
   return SectionWindows().UseMinMaxPt();
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


cTplValGesInit< std::string > & cParamSaisiePts::PatternNameInputsSec()
{
   return SectionInOut().PatternNameInputsSec();
}

const cTplValGesInit< std::string > & cParamSaisiePts::PatternNameInputsSec()const 
{
   return SectionInOut().PatternNameInputsSec();
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


cTplValGesInit< bool > & cParamSaisiePts::ForceExport3D()
{
   return SectionInOut().ForceExport3D();
}

const cTplValGesInit< bool > & cParamSaisiePts::ForceExport3D()const 
{
   return SectionInOut().ForceExport3D();
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


cTplValGesInit< double > & cParamSaisiePts::Gama()
{
   return SectionImages().Gama();
}

const cTplValGesInit< double > & cParamSaisiePts::Gama()const 
{
   return SectionImages().Gama();
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


cTplValGesInit< std::string > & cParamSaisiePts::Masq3DFilterVis()
{
   return SectionTerrain().Masq3DFilterVis();
}

const cTplValGesInit< std::string > & cParamSaisiePts::Masq3DFilterVis()const 
{
   return SectionTerrain().Masq3DFilterVis();
}


cTplValGesInit< std::string > & cParamSaisiePts::PIMsFilterVis()
{
   return SectionTerrain().PIMsFilterVis();
}

const cTplValGesInit< std::string > & cParamSaisiePts::PIMsFilterVis()const 
{
   return SectionTerrain().PIMsFilterVis();
}


cTplValGesInit< double > & cParamSaisiePts::DistMaxVisib()
{
   return SectionTerrain().DistMaxVisib();
}

const cTplValGesInit< double > & cParamSaisiePts::DistMaxVisib()const 
{
   return SectionTerrain().DistMaxVisib();
}


cTplValGesInit< cElRegex_Ptr > & cParamSaisiePts::PatternNamePtsVisib()
{
   return SectionTerrain().PatternNamePtsVisib();
}

const cTplValGesInit< cElRegex_Ptr > & cParamSaisiePts::PatternNamePtsVisib()const 
{
   return SectionTerrain().PatternNamePtsVisib();
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

void  BinaryUnDumpFromFile(cParamSaisiePts & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.DicoLoc().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.DicoLoc().ValForcedForUnUmp(),aFp);
        }
        else  anObj.DicoLoc().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.SectionWindows(),aFp);
    BinaryUnDumpFromFile(anObj.SectionInOut(),aFp);
    BinaryUnDumpFromFile(anObj.SectionImages(),aFp);
    BinaryUnDumpFromFile(anObj.SectionTerrain(),aFp);
    BinaryUnDumpFromFile(anObj.DirectoryChantier(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cParamSaisiePts & anObj)
{
    BinaryDumpInFile(aFp,anObj.DicoLoc().IsInit());
    if (anObj.DicoLoc().IsInit()) BinaryDumpInFile(aFp,anObj.DicoLoc().Val());
    BinaryDumpInFile(aFp,anObj.SectionWindows());
    BinaryDumpInFile(aFp,anObj.SectionInOut());
    BinaryDumpInFile(aFp,anObj.SectionImages());
    BinaryDumpInFile(aFp,anObj.SectionTerrain());
    BinaryDumpInFile(aFp,anObj.DirectoryChantier());
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
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.DicoLoc(),aTree->Get("DicoLoc",1)); //tototo 

   xml_init(anObj.SectionWindows(),aTree->Get("SectionWindows",1)); //tototo 

   xml_init(anObj.SectionInOut(),aTree->Get("SectionInOut",1)); //tototo 

   xml_init(anObj.SectionImages(),aTree->Get("SectionImages",1)); //tototo 

   xml_init(anObj.SectionTerrain(),aTree->Get("SectionTerrain",1)); //tototo 

   xml_init(anObj.DirectoryChantier(),aTree->Get("DirectoryChantier",1)); //tototo 
}

std::string  Mangling( cParamSaisiePts *) {return "5DF90EDA3EA187A4FF3F";};

// };
