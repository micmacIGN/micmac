#include "cParamDigeo.h"
//
eTypeTopolPt  Str2eTypeTopolPt(const std::string & aName)
{
   if (aName=="eTtpSommet")
      return eTtpSommet;
   else if (aName=="eTtpCuvette")
      return eTtpCuvette;
   else if (aName=="eTtpCol")
      return eTtpCol;
   else if (aName=="eTtpCorner")
      return eTtpCorner;
   else if (aName=="eSiftMaxDog")
      return eSiftMaxDog;
   else if (aName=="eSiftMinDog")
      return eSiftMinDog;
  else
  {
      cout << aName << " is not a correct value for enum eTypeTopolPt\n" ;
      ELISE_ASSERT(false,"XML enum value error");
  }
  return (eTypeTopolPt) 0;
}
void xml_init(eTypeTopolPt & aVal,cElXMLTree * aTree)
{
   aVal= Str2eTypeTopolPt(aTree->Contenu());
}
std::string  eToString(const eTypeTopolPt & anObj)
{
   if (anObj==eTtpSommet)
      return  "eTtpSommet";
   if (anObj==eTtpCuvette)
      return  "eTtpCuvette";
   if (anObj==eTtpCol)
      return  "eTtpCol";
   if (anObj==eTtpCorner)
      return  "eTtpCorner";
   if (anObj==eSiftMaxDog)
      return  "eSiftMaxDog";
   if (anObj==eSiftMinDog)
      return  "eSiftMinDog";
 std::cout << "Enum = eTypeTopolPt\n";
   ELISE_ASSERT(false,"Bad Value in eToString for enum value ");
   return "";
}

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eTypeTopolPt & anObj)
{
      return  cElXMLTree::ValueNode(aNameTag,eToString(anObj));
}

void  BinaryDumpInFile(ELISE_fp & aFp,const eTypeTopolPt & anObj)
{
   BinaryDumpInFile(aFp,int(anObj));
}

void  BinaryUnDumpFromFile(eTypeTopolPt & anObj,ELISE_fp & aFp)
{
   int aIVal;
   BinaryUnDumpFromFile(aIVal,aFp);
   anObj=(eTypeTopolPt) aIVal;
}

std::string  Mangling( eTypeTopolPt *) {return "CC0F34E54BDAE3A5FE3F";};

eReducDemiImage  Str2eReducDemiImage(const std::string & aName)
{
   if (aName=="eRDI_121")
      return eRDI_121;
   else if (aName=="eRDI_010")
      return eRDI_010;
   else if (aName=="eRDI_11")
      return eRDI_11;
  else
  {
      cout << aName << " is not a correct value for enum eReducDemiImage\n" ;
      ELISE_ASSERT(false,"XML enum value error");
  }
  return (eReducDemiImage) 0;
}
void xml_init(eReducDemiImage & aVal,cElXMLTree * aTree)
{
   aVal= Str2eReducDemiImage(aTree->Contenu());
}
std::string  eToString(const eReducDemiImage & anObj)
{
   if (anObj==eRDI_121)
      return  "eRDI_121";
   if (anObj==eRDI_010)
      return  "eRDI_010";
   if (anObj==eRDI_11)
      return  "eRDI_11";
 std::cout << "Enum = eReducDemiImage\n";
   ELISE_ASSERT(false,"Bad Value in eToString for enum value ");
   return "";
}

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eReducDemiImage & anObj)
{
      return  cElXMLTree::ValueNode(aNameTag,eToString(anObj));
}

void  BinaryDumpInFile(ELISE_fp & aFp,const eReducDemiImage & anObj)
{
   BinaryDumpInFile(aFp,int(anObj));
}

void  BinaryUnDumpFromFile(eReducDemiImage & anObj,ELISE_fp & aFp)
{
   int aIVal;
   BinaryUnDumpFromFile(aIVal,aFp);
   anObj=(eReducDemiImage) aIVal;
}

std::string  Mangling( eReducDemiImage *) {return "AB929125A8D1908BFF3F";};

eTest12345  Str2eTest12345(const std::string & aName)
{
   if (aName=="eTest12345_A")
      return eTest12345_A;
   else if (aName=="eTest12345_B")
      return eTest12345_B;
   else if (aName=="eTest12345_C")
      return eTest12345_C;
  else
  {
      cout << aName << " is not a correct value for enum eTest12345\n" ;
      ELISE_ASSERT(false,"XML enum value error");
  }
  return (eTest12345) 0;
}
void xml_init(eTest12345 & aVal,cElXMLTree * aTree)
{
   aVal= Str2eTest12345(aTree->Contenu());
}
std::string  eToString(const eTest12345 & anObj)
{
   if (anObj==eTest12345_A)
      return  "eTest12345_A";
   if (anObj==eTest12345_B)
      return  "eTest12345_B";
   if (anObj==eTest12345_C)
      return  "eTest12345_C";
 std::cout << "Enum = eTest12345\n";
   ELISE_ASSERT(false,"Bad Value in eToString for enum value ");
   return "";
}

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eTest12345 & anObj)
{
      return  cElXMLTree::ValueNode(aNameTag,eToString(anObj));
}

void  BinaryDumpInFile(ELISE_fp & aFp,const eTest12345 & anObj)
{
   BinaryDumpInFile(aFp,int(anObj));
}

void  BinaryUnDumpFromFile(eTest12345 & anObj,ELISE_fp & aFp)
{
   int aIVal;
   BinaryUnDumpFromFile(aIVal,aFp);
   anObj=(eTest12345) aIVal;
}

std::string  Mangling( eTest12345 *) {return "2CC9DB599F3AC5BEFD3F";};


cTplValGesInit< int > & cParamExtractCaracIm::SzMinImDeZoom()
{
   return mSzMinImDeZoom;
}

const cTplValGesInit< int > & cParamExtractCaracIm::SzMinImDeZoom()const 
{
   return mSzMinImDeZoom;
}

void  BinaryUnDumpFromFile(cParamExtractCaracIm & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SzMinImDeZoom().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SzMinImDeZoom().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SzMinImDeZoom().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cParamExtractCaracIm & anObj)
{
    BinaryDumpInFile(aFp,anObj.SzMinImDeZoom().IsInit());
    if (anObj.SzMinImDeZoom().IsInit()) BinaryDumpInFile(aFp,anObj.SzMinImDeZoom().Val());
}

cElXMLTree * ToXMLTree(const cParamExtractCaracIm & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ParamExtractCaracIm",eXMLBranche);
   if (anObj.SzMinImDeZoom().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SzMinImDeZoom"),anObj.SzMinImDeZoom().Val())->ReTagThis("SzMinImDeZoom"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cParamExtractCaracIm & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.SzMinImDeZoom(),aTree->Get("SzMinImDeZoom",1),int(500)); //tototo 
}

std::string  Mangling( cParamExtractCaracIm *) {return "E003D59EBA6B1B96FA3F";};


cTplValGesInit< int > & cParamVisuCarac::DynGray()
{
   return mDynGray;
}

const cTplValGesInit< int > & cParamVisuCarac::DynGray()const 
{
   return mDynGray;
}


std::string & cParamVisuCarac::Dir()
{
   return mDir;
}

const std::string & cParamVisuCarac::Dir()const 
{
   return mDir;
}


cTplValGesInit< int > & cParamVisuCarac::Zoom()
{
   return mZoom;
}

const cTplValGesInit< int > & cParamVisuCarac::Zoom()const 
{
   return mZoom;
}


double & cParamVisuCarac::Dyn()
{
   return mDyn;
}

const double & cParamVisuCarac::Dyn()const 
{
   return mDyn;
}


cTplValGesInit< std::string > & cParamVisuCarac::Prefix()
{
   return mPrefix;
}

const cTplValGesInit< std::string > & cParamVisuCarac::Prefix()const 
{
   return mPrefix;
}


cTplValGesInit< bool > & cParamVisuCarac::ShowCaracEchec()
{
   return mShowCaracEchec;
}

const cTplValGesInit< bool > & cParamVisuCarac::ShowCaracEchec()const 
{
   return mShowCaracEchec;
}

void  BinaryUnDumpFromFile(cParamVisuCarac & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.DynGray().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.DynGray().ValForcedForUnUmp(),aFp);
        }
        else  anObj.DynGray().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.Dir(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Zoom().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Zoom().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Zoom().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.Dyn(),aFp);
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
             anObj.ShowCaracEchec().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ShowCaracEchec().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ShowCaracEchec().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cParamVisuCarac & anObj)
{
    BinaryDumpInFile(aFp,anObj.DynGray().IsInit());
    if (anObj.DynGray().IsInit()) BinaryDumpInFile(aFp,anObj.DynGray().Val());
    BinaryDumpInFile(aFp,anObj.Dir());
    BinaryDumpInFile(aFp,anObj.Zoom().IsInit());
    if (anObj.Zoom().IsInit()) BinaryDumpInFile(aFp,anObj.Zoom().Val());
    BinaryDumpInFile(aFp,anObj.Dyn());
    BinaryDumpInFile(aFp,anObj.Prefix().IsInit());
    if (anObj.Prefix().IsInit()) BinaryDumpInFile(aFp,anObj.Prefix().Val());
    BinaryDumpInFile(aFp,anObj.ShowCaracEchec().IsInit());
    if (anObj.ShowCaracEchec().IsInit()) BinaryDumpInFile(aFp,anObj.ShowCaracEchec().Val());
}

cElXMLTree * ToXMLTree(const cParamVisuCarac & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ParamVisuCarac",eXMLBranche);
   if (anObj.DynGray().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DynGray"),anObj.DynGray().Val())->ReTagThis("DynGray"));
   aRes->AddFils(::ToXMLTree(std::string("Dir"),anObj.Dir())->ReTagThis("Dir"));
   if (anObj.Zoom().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Zoom"),anObj.Zoom().Val())->ReTagThis("Zoom"));
   aRes->AddFils(::ToXMLTree(std::string("Dyn"),anObj.Dyn())->ReTagThis("Dyn"));
   if (anObj.Prefix().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Prefix"),anObj.Prefix().Val())->ReTagThis("Prefix"));
   if (anObj.ShowCaracEchec().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ShowCaracEchec"),anObj.ShowCaracEchec().Val())->ReTagThis("ShowCaracEchec"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cParamVisuCarac & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.DynGray(),aTree->Get("DynGray",1),int(128)); //tototo 

   xml_init(anObj.Dir(),aTree->Get("Dir",1)); //tototo 

   xml_init(anObj.Zoom(),aTree->Get("Zoom",1),int(1)); //tototo 

   xml_init(anObj.Dyn(),aTree->Get("Dyn",1)); //tototo 

   xml_init(anObj.Prefix(),aTree->Get("Prefix",1),std::string("VisuCarac_")); //tototo 

   xml_init(anObj.ShowCaracEchec(),aTree->Get("ShowCaracEchec",1),bool(false)); //tototo 
}

std::string  Mangling( cParamVisuCarac *) {return "A34364EF4DE6D7A3FF3F";};


cTplValGesInit< std::string > & cPredicteurGeom::Unused()
{
   return mUnused;
}

const cTplValGesInit< std::string > & cPredicteurGeom::Unused()const 
{
   return mUnused;
}

void  BinaryUnDumpFromFile(cPredicteurGeom & anObj,ELISE_fp & aFp)
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

void  BinaryDumpInFile(ELISE_fp & aFp,const cPredicteurGeom & anObj)
{
    BinaryDumpInFile(aFp,anObj.Unused().IsInit());
    if (anObj.Unused().IsInit()) BinaryDumpInFile(aFp,anObj.Unused().Val());
}

cElXMLTree * ToXMLTree(const cPredicteurGeom & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"PredicteurGeom",eXMLBranche);
   if (anObj.Unused().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Unused"),anObj.Unused().Val())->ReTagThis("Unused"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cPredicteurGeom & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Unused(),aTree->Get("Unused",1)); //tototo 
}

std::string  Mangling( cPredicteurGeom *) {return "40A66D3459CD999CF8BF";};


cTplValGesInit< cParamVisuCarac > & cImageDigeo::VisuCarac()
{
   return mVisuCarac;
}

const cTplValGesInit< cParamVisuCarac > & cImageDigeo::VisuCarac()const 
{
   return mVisuCarac;
}


std::string & cImageDigeo::KeyOrPat()
{
   return mKeyOrPat;
}

const std::string & cImageDigeo::KeyOrPat()const 
{
   return mKeyOrPat;
}


cTplValGesInit< std::string > & cImageDigeo::KeyCalcCalib()
{
   return mKeyCalcCalib;
}

const cTplValGesInit< std::string > & cImageDigeo::KeyCalcCalib()const 
{
   return mKeyCalcCalib;
}


cTplValGesInit< Box2di > & cImageDigeo::BoxImR1()
{
   return mBoxImR1;
}

const cTplValGesInit< Box2di > & cImageDigeo::BoxImR1()const 
{
   return mBoxImR1;
}


cTplValGesInit< double > & cImageDigeo::ResolInit()
{
   return mResolInit;
}

const cTplValGesInit< double > & cImageDigeo::ResolInit()const 
{
   return mResolInit;
}


cTplValGesInit< std::string > & cImageDigeo::Unused()
{
   return PredicteurGeom().Val().Unused();
}

const cTplValGesInit< std::string > & cImageDigeo::Unused()const 
{
   return PredicteurGeom().Val().Unused();
}


cTplValGesInit< cPredicteurGeom > & cImageDigeo::PredicteurGeom()
{
   return mPredicteurGeom;
}

const cTplValGesInit< cPredicteurGeom > & cImageDigeo::PredicteurGeom()const 
{
   return mPredicteurGeom;
}


cTplValGesInit< double > & cImageDigeo::NbOctetLimitLoadImageOnce()
{
   return mNbOctetLimitLoadImageOnce;
}

const cTplValGesInit< double > & cImageDigeo::NbOctetLimitLoadImageOnce()const 
{
   return mNbOctetLimitLoadImageOnce;
}

void  BinaryUnDumpFromFile(cImageDigeo & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.VisuCarac().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.VisuCarac().ValForcedForUnUmp(),aFp);
        }
        else  anObj.VisuCarac().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.KeyOrPat(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.KeyCalcCalib().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.KeyCalcCalib().ValForcedForUnUmp(),aFp);
        }
        else  anObj.KeyCalcCalib().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.BoxImR1().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.BoxImR1().ValForcedForUnUmp(),aFp);
        }
        else  anObj.BoxImR1().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ResolInit().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ResolInit().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ResolInit().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.PredicteurGeom().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.PredicteurGeom().ValForcedForUnUmp(),aFp);
        }
        else  anObj.PredicteurGeom().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NbOctetLimitLoadImageOnce().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NbOctetLimitLoadImageOnce().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NbOctetLimitLoadImageOnce().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cImageDigeo & anObj)
{
    BinaryDumpInFile(aFp,anObj.VisuCarac().IsInit());
    if (anObj.VisuCarac().IsInit()) BinaryDumpInFile(aFp,anObj.VisuCarac().Val());
    BinaryDumpInFile(aFp,anObj.KeyOrPat());
    BinaryDumpInFile(aFp,anObj.KeyCalcCalib().IsInit());
    if (anObj.KeyCalcCalib().IsInit()) BinaryDumpInFile(aFp,anObj.KeyCalcCalib().Val());
    BinaryDumpInFile(aFp,anObj.BoxImR1().IsInit());
    if (anObj.BoxImR1().IsInit()) BinaryDumpInFile(aFp,anObj.BoxImR1().Val());
    BinaryDumpInFile(aFp,anObj.ResolInit().IsInit());
    if (anObj.ResolInit().IsInit()) BinaryDumpInFile(aFp,anObj.ResolInit().Val());
    BinaryDumpInFile(aFp,anObj.PredicteurGeom().IsInit());
    if (anObj.PredicteurGeom().IsInit()) BinaryDumpInFile(aFp,anObj.PredicteurGeom().Val());
    BinaryDumpInFile(aFp,anObj.NbOctetLimitLoadImageOnce().IsInit());
    if (anObj.NbOctetLimitLoadImageOnce().IsInit()) BinaryDumpInFile(aFp,anObj.NbOctetLimitLoadImageOnce().Val());
}

cElXMLTree * ToXMLTree(const cImageDigeo & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ImageDigeo",eXMLBranche);
   if (anObj.VisuCarac().IsInit())
      aRes->AddFils(ToXMLTree(anObj.VisuCarac().Val())->ReTagThis("VisuCarac"));
   aRes->AddFils(::ToXMLTree(std::string("KeyOrPat"),anObj.KeyOrPat())->ReTagThis("KeyOrPat"));
   if (anObj.KeyCalcCalib().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("KeyCalcCalib"),anObj.KeyCalcCalib().Val())->ReTagThis("KeyCalcCalib"));
   if (anObj.BoxImR1().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("BoxImR1"),anObj.BoxImR1().Val())->ReTagThis("BoxImR1"));
   if (anObj.ResolInit().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ResolInit"),anObj.ResolInit().Val())->ReTagThis("ResolInit"));
   if (anObj.PredicteurGeom().IsInit())
      aRes->AddFils(ToXMLTree(anObj.PredicteurGeom().Val())->ReTagThis("PredicteurGeom"));
   if (anObj.NbOctetLimitLoadImageOnce().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NbOctetLimitLoadImageOnce"),anObj.NbOctetLimitLoadImageOnce().Val())->ReTagThis("NbOctetLimitLoadImageOnce"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cImageDigeo & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.VisuCarac(),aTree->Get("VisuCarac",1)); //tototo 

   xml_init(anObj.KeyOrPat(),aTree->Get("KeyOrPat",1)); //tototo 

   xml_init(anObj.KeyCalcCalib(),aTree->Get("KeyCalcCalib",1)); //tototo 

   xml_init(anObj.BoxImR1(),aTree->Get("BoxImR1",1)); //tototo 

   xml_init(anObj.ResolInit(),aTree->Get("ResolInit",1),double(1.0)); //tototo 

   xml_init(anObj.PredicteurGeom(),aTree->Get("PredicteurGeom",1)); //tototo 

   xml_init(anObj.NbOctetLimitLoadImageOnce(),aTree->Get("NbOctetLimitLoadImageOnce",1),double(1e8)); //tototo 
}

std::string  Mangling( cImageDigeo *) {return "F01DEAE619E3809FFE3F";};


eTypeNumerique & cTypeNumeriqueOfNiv::Type()
{
   return mType;
}

const eTypeNumerique & cTypeNumeriqueOfNiv::Type()const 
{
   return mType;
}


int & cTypeNumeriqueOfNiv::Niv()
{
   return mNiv;
}

const int & cTypeNumeriqueOfNiv::Niv()const 
{
   return mNiv;
}

void  BinaryUnDumpFromFile(cTypeNumeriqueOfNiv & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Type(),aFp);
    BinaryUnDumpFromFile(anObj.Niv(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cTypeNumeriqueOfNiv & anObj)
{
    BinaryDumpInFile(aFp,anObj.Type());
    BinaryDumpInFile(aFp,anObj.Niv());
}

cElXMLTree * ToXMLTree(const cTypeNumeriqueOfNiv & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"TypeNumeriqueOfNiv",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Type"),anObj.Type())->ReTagThis("Type"));
   aRes->AddFils(::ToXMLTree(std::string("Niv"),anObj.Niv())->ReTagThis("Niv"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cTypeNumeriqueOfNiv & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Type(),aTree->Get("Type",1)); //tototo 

   xml_init(anObj.Niv(),aTree->Get("Niv",1)); //tototo 
}

std::string  Mangling( cTypeNumeriqueOfNiv *) {return "96E7077E62871A8DFD3F";};


cTplValGesInit< int > & cPyramideGaussienne::NbByOctave()
{
   return mNbByOctave;
}

const cTplValGesInit< int > & cPyramideGaussienne::NbByOctave()const 
{
   return mNbByOctave;
}


cTplValGesInit< double > & cPyramideGaussienne::Sigma0()
{
   return mSigma0;
}

const cTplValGesInit< double > & cPyramideGaussienne::Sigma0()const 
{
   return mSigma0;
}


cTplValGesInit< double > & cPyramideGaussienne::SigmaN()
{
   return mSigmaN;
}

const cTplValGesInit< double > & cPyramideGaussienne::SigmaN()const 
{
   return mSigmaN;
}


cTplValGesInit< int > & cPyramideGaussienne::NbInLastOctave()
{
   return mNbInLastOctave;
}

const cTplValGesInit< int > & cPyramideGaussienne::NbInLastOctave()const 
{
   return mNbInLastOctave;
}


cTplValGesInit< int > & cPyramideGaussienne::IndexFreqInFirstOctave()
{
   return mIndexFreqInFirstOctave;
}

const cTplValGesInit< int > & cPyramideGaussienne::IndexFreqInFirstOctave()const 
{
   return mIndexFreqInFirstOctave;
}


int & cPyramideGaussienne::NivOctaveMax()
{
   return mNivOctaveMax;
}

const int & cPyramideGaussienne::NivOctaveMax()const 
{
   return mNivOctaveMax;
}


cTplValGesInit< double > & cPyramideGaussienne::ConvolFirstImage()
{
   return mConvolFirstImage;
}

const cTplValGesInit< double > & cPyramideGaussienne::ConvolFirstImage()const 
{
   return mConvolFirstImage;
}


cTplValGesInit< double > & cPyramideGaussienne::EpsilonGauss()
{
   return mEpsilonGauss;
}

const cTplValGesInit< double > & cPyramideGaussienne::EpsilonGauss()const 
{
   return mEpsilonGauss;
}


cTplValGesInit< int > & cPyramideGaussienne::NbShift()
{
   return mNbShift;
}

const cTplValGesInit< int > & cPyramideGaussienne::NbShift()const 
{
   return mNbShift;
}


cTplValGesInit< int > & cPyramideGaussienne::SurEchIntegralGauss()
{
   return mSurEchIntegralGauss;
}

const cTplValGesInit< int > & cPyramideGaussienne::SurEchIntegralGauss()const 
{
   return mSurEchIntegralGauss;
}


cTplValGesInit< bool > & cPyramideGaussienne::ConvolIncrem()
{
   return mConvolIncrem;
}

const cTplValGesInit< bool > & cPyramideGaussienne::ConvolIncrem()const 
{
   return mConvolIncrem;
}

void  BinaryUnDumpFromFile(cPyramideGaussienne & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NbByOctave().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NbByOctave().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NbByOctave().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Sigma0().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Sigma0().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Sigma0().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SigmaN().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SigmaN().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SigmaN().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NbInLastOctave().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NbInLastOctave().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NbInLastOctave().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.IndexFreqInFirstOctave().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.IndexFreqInFirstOctave().ValForcedForUnUmp(),aFp);
        }
        else  anObj.IndexFreqInFirstOctave().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.NivOctaveMax(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ConvolFirstImage().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ConvolFirstImage().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ConvolFirstImage().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.EpsilonGauss().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.EpsilonGauss().ValForcedForUnUmp(),aFp);
        }
        else  anObj.EpsilonGauss().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NbShift().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NbShift().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NbShift().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SurEchIntegralGauss().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SurEchIntegralGauss().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SurEchIntegralGauss().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ConvolIncrem().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ConvolIncrem().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ConvolIncrem().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cPyramideGaussienne & anObj)
{
    BinaryDumpInFile(aFp,anObj.NbByOctave().IsInit());
    if (anObj.NbByOctave().IsInit()) BinaryDumpInFile(aFp,anObj.NbByOctave().Val());
    BinaryDumpInFile(aFp,anObj.Sigma0().IsInit());
    if (anObj.Sigma0().IsInit()) BinaryDumpInFile(aFp,anObj.Sigma0().Val());
    BinaryDumpInFile(aFp,anObj.SigmaN().IsInit());
    if (anObj.SigmaN().IsInit()) BinaryDumpInFile(aFp,anObj.SigmaN().Val());
    BinaryDumpInFile(aFp,anObj.NbInLastOctave().IsInit());
    if (anObj.NbInLastOctave().IsInit()) BinaryDumpInFile(aFp,anObj.NbInLastOctave().Val());
    BinaryDumpInFile(aFp,anObj.IndexFreqInFirstOctave().IsInit());
    if (anObj.IndexFreqInFirstOctave().IsInit()) BinaryDumpInFile(aFp,anObj.IndexFreqInFirstOctave().Val());
    BinaryDumpInFile(aFp,anObj.NivOctaveMax());
    BinaryDumpInFile(aFp,anObj.ConvolFirstImage().IsInit());
    if (anObj.ConvolFirstImage().IsInit()) BinaryDumpInFile(aFp,anObj.ConvolFirstImage().Val());
    BinaryDumpInFile(aFp,anObj.EpsilonGauss().IsInit());
    if (anObj.EpsilonGauss().IsInit()) BinaryDumpInFile(aFp,anObj.EpsilonGauss().Val());
    BinaryDumpInFile(aFp,anObj.NbShift().IsInit());
    if (anObj.NbShift().IsInit()) BinaryDumpInFile(aFp,anObj.NbShift().Val());
    BinaryDumpInFile(aFp,anObj.SurEchIntegralGauss().IsInit());
    if (anObj.SurEchIntegralGauss().IsInit()) BinaryDumpInFile(aFp,anObj.SurEchIntegralGauss().Val());
    BinaryDumpInFile(aFp,anObj.ConvolIncrem().IsInit());
    if (anObj.ConvolIncrem().IsInit()) BinaryDumpInFile(aFp,anObj.ConvolIncrem().Val());
}

cElXMLTree * ToXMLTree(const cPyramideGaussienne & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"PyramideGaussienne",eXMLBranche);
   if (anObj.NbByOctave().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NbByOctave"),anObj.NbByOctave().Val())->ReTagThis("NbByOctave"));
   if (anObj.Sigma0().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Sigma0"),anObj.Sigma0().Val())->ReTagThis("Sigma0"));
   if (anObj.SigmaN().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SigmaN"),anObj.SigmaN().Val())->ReTagThis("SigmaN"));
   if (anObj.NbInLastOctave().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NbInLastOctave"),anObj.NbInLastOctave().Val())->ReTagThis("NbInLastOctave"));
   if (anObj.IndexFreqInFirstOctave().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("IndexFreqInFirstOctave"),anObj.IndexFreqInFirstOctave().Val())->ReTagThis("IndexFreqInFirstOctave"));
   aRes->AddFils(::ToXMLTree(std::string("NivOctaveMax"),anObj.NivOctaveMax())->ReTagThis("NivOctaveMax"));
   if (anObj.ConvolFirstImage().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ConvolFirstImage"),anObj.ConvolFirstImage().Val())->ReTagThis("ConvolFirstImage"));
   if (anObj.EpsilonGauss().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("EpsilonGauss"),anObj.EpsilonGauss().Val())->ReTagThis("EpsilonGauss"));
   if (anObj.NbShift().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NbShift"),anObj.NbShift().Val())->ReTagThis("NbShift"));
   if (anObj.SurEchIntegralGauss().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SurEchIntegralGauss"),anObj.SurEchIntegralGauss().Val())->ReTagThis("SurEchIntegralGauss"));
   if (anObj.ConvolIncrem().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ConvolIncrem"),anObj.ConvolIncrem().Val())->ReTagThis("ConvolIncrem"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cPyramideGaussienne & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.NbByOctave(),aTree->Get("NbByOctave",1),int(3)); //tototo 

   xml_init(anObj.Sigma0(),aTree->Get("Sigma0",1),double(1.6)); //tototo 

   xml_init(anObj.SigmaN(),aTree->Get("SigmaN",1)); //tototo 

   xml_init(anObj.NbInLastOctave(),aTree->Get("NbInLastOctave",1)); //tototo 

   xml_init(anObj.IndexFreqInFirstOctave(),aTree->Get("IndexFreqInFirstOctave",1),int(0)); //tototo 

   xml_init(anObj.NivOctaveMax(),aTree->Get("NivOctaveMax",1)); //tototo 

   xml_init(anObj.ConvolFirstImage(),aTree->Get("ConvolFirstImage",1),double(-1)); //tototo 

   xml_init(anObj.EpsilonGauss(),aTree->Get("EpsilonGauss",1),double(1e-3)); //tototo 

   xml_init(anObj.NbShift(),aTree->Get("NbShift",1),int(15)); //tototo 

   xml_init(anObj.SurEchIntegralGauss(),aTree->Get("SurEchIntegralGauss",1),int(10)); //tototo 

   xml_init(anObj.ConvolIncrem(),aTree->Get("ConvolIncrem",1),bool(true)); //tototo 
}

std::string  Mangling( cPyramideGaussienne *) {return "22717116DC74AC97FF3F";};


cTplValGesInit< int > & cTypePyramide::NivPyramBasique()
{
   return mNivPyramBasique;
}

const cTplValGesInit< int > & cTypePyramide::NivPyramBasique()const 
{
   return mNivPyramBasique;
}


cTplValGesInit< int > & cTypePyramide::NbByOctave()
{
   return PyramideGaussienne().Val().NbByOctave();
}

const cTplValGesInit< int > & cTypePyramide::NbByOctave()const 
{
   return PyramideGaussienne().Val().NbByOctave();
}


cTplValGesInit< double > & cTypePyramide::Sigma0()
{
   return PyramideGaussienne().Val().Sigma0();
}

const cTplValGesInit< double > & cTypePyramide::Sigma0()const 
{
   return PyramideGaussienne().Val().Sigma0();
}


cTplValGesInit< double > & cTypePyramide::SigmaN()
{
   return PyramideGaussienne().Val().SigmaN();
}

const cTplValGesInit< double > & cTypePyramide::SigmaN()const 
{
   return PyramideGaussienne().Val().SigmaN();
}


cTplValGesInit< int > & cTypePyramide::NbInLastOctave()
{
   return PyramideGaussienne().Val().NbInLastOctave();
}

const cTplValGesInit< int > & cTypePyramide::NbInLastOctave()const 
{
   return PyramideGaussienne().Val().NbInLastOctave();
}


cTplValGesInit< int > & cTypePyramide::IndexFreqInFirstOctave()
{
   return PyramideGaussienne().Val().IndexFreqInFirstOctave();
}

const cTplValGesInit< int > & cTypePyramide::IndexFreqInFirstOctave()const 
{
   return PyramideGaussienne().Val().IndexFreqInFirstOctave();
}


int & cTypePyramide::NivOctaveMax()
{
   return PyramideGaussienne().Val().NivOctaveMax();
}

const int & cTypePyramide::NivOctaveMax()const 
{
   return PyramideGaussienne().Val().NivOctaveMax();
}


cTplValGesInit< double > & cTypePyramide::ConvolFirstImage()
{
   return PyramideGaussienne().Val().ConvolFirstImage();
}

const cTplValGesInit< double > & cTypePyramide::ConvolFirstImage()const 
{
   return PyramideGaussienne().Val().ConvolFirstImage();
}


cTplValGesInit< double > & cTypePyramide::EpsilonGauss()
{
   return PyramideGaussienne().Val().EpsilonGauss();
}

const cTplValGesInit< double > & cTypePyramide::EpsilonGauss()const 
{
   return PyramideGaussienne().Val().EpsilonGauss();
}


cTplValGesInit< int > & cTypePyramide::NbShift()
{
   return PyramideGaussienne().Val().NbShift();
}

const cTplValGesInit< int > & cTypePyramide::NbShift()const 
{
   return PyramideGaussienne().Val().NbShift();
}


cTplValGesInit< int > & cTypePyramide::SurEchIntegralGauss()
{
   return PyramideGaussienne().Val().SurEchIntegralGauss();
}

const cTplValGesInit< int > & cTypePyramide::SurEchIntegralGauss()const 
{
   return PyramideGaussienne().Val().SurEchIntegralGauss();
}


cTplValGesInit< bool > & cTypePyramide::ConvolIncrem()
{
   return PyramideGaussienne().Val().ConvolIncrem();
}

const cTplValGesInit< bool > & cTypePyramide::ConvolIncrem()const 
{
   return PyramideGaussienne().Val().ConvolIncrem();
}


cTplValGesInit< cPyramideGaussienne > & cTypePyramide::PyramideGaussienne()
{
   return mPyramideGaussienne;
}

const cTplValGesInit< cPyramideGaussienne > & cTypePyramide::PyramideGaussienne()const 
{
   return mPyramideGaussienne;
}

void  BinaryUnDumpFromFile(cTypePyramide & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NivPyramBasique().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NivPyramBasique().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NivPyramBasique().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.PyramideGaussienne().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.PyramideGaussienne().ValForcedForUnUmp(),aFp);
        }
        else  anObj.PyramideGaussienne().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cTypePyramide & anObj)
{
    BinaryDumpInFile(aFp,anObj.NivPyramBasique().IsInit());
    if (anObj.NivPyramBasique().IsInit()) BinaryDumpInFile(aFp,anObj.NivPyramBasique().Val());
    BinaryDumpInFile(aFp,anObj.PyramideGaussienne().IsInit());
    if (anObj.PyramideGaussienne().IsInit()) BinaryDumpInFile(aFp,anObj.PyramideGaussienne().Val());
}

cElXMLTree * ToXMLTree(const cTypePyramide & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"TypePyramide",eXMLBranche);
   if (anObj.NivPyramBasique().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NivPyramBasique"),anObj.NivPyramBasique().Val())->ReTagThis("NivPyramBasique"));
   if (anObj.PyramideGaussienne().IsInit())
      aRes->AddFils(ToXMLTree(anObj.PyramideGaussienne().Val())->ReTagThis("PyramideGaussienne"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cTypePyramide & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.NivPyramBasique(),aTree->Get("NivPyramBasique",1)); //tototo 

   xml_init(anObj.PyramideGaussienne(),aTree->Get("PyramideGaussienne",1)); //tototo 
}

std::string  Mangling( cTypePyramide *) {return "81FC747DDDC76FEFFE3F";};


std::list< cTypeNumeriqueOfNiv > & cPyramideImage::TypeNumeriqueOfNiv()
{
   return mTypeNumeriqueOfNiv;
}

const std::list< cTypeNumeriqueOfNiv > & cPyramideImage::TypeNumeriqueOfNiv()const 
{
   return mTypeNumeriqueOfNiv;
}


cTplValGesInit< bool > & cPyramideImage::MaximDyn()
{
   return mMaximDyn;
}

const cTplValGesInit< bool > & cPyramideImage::MaximDyn()const 
{
   return mMaximDyn;
}


cTplValGesInit< double > & cPyramideImage::ValMaxForDyn()
{
   return mValMaxForDyn;
}

const cTplValGesInit< double > & cPyramideImage::ValMaxForDyn()const 
{
   return mValMaxForDyn;
}


cTplValGesInit< eReducDemiImage > & cPyramideImage::ReducDemiImage()
{
   return mReducDemiImage;
}

const cTplValGesInit< eReducDemiImage > & cPyramideImage::ReducDemiImage()const 
{
   return mReducDemiImage;
}


cTplValGesInit< int > & cPyramideImage::NivPyramBasique()
{
   return TypePyramide().NivPyramBasique();
}

const cTplValGesInit< int > & cPyramideImage::NivPyramBasique()const 
{
   return TypePyramide().NivPyramBasique();
}


cTplValGesInit< int > & cPyramideImage::NbByOctave()
{
   return TypePyramide().PyramideGaussienne().Val().NbByOctave();
}

const cTplValGesInit< int > & cPyramideImage::NbByOctave()const 
{
   return TypePyramide().PyramideGaussienne().Val().NbByOctave();
}


cTplValGesInit< double > & cPyramideImage::Sigma0()
{
   return TypePyramide().PyramideGaussienne().Val().Sigma0();
}

const cTplValGesInit< double > & cPyramideImage::Sigma0()const 
{
   return TypePyramide().PyramideGaussienne().Val().Sigma0();
}


cTplValGesInit< double > & cPyramideImage::SigmaN()
{
   return TypePyramide().PyramideGaussienne().Val().SigmaN();
}

const cTplValGesInit< double > & cPyramideImage::SigmaN()const 
{
   return TypePyramide().PyramideGaussienne().Val().SigmaN();
}


cTplValGesInit< int > & cPyramideImage::NbInLastOctave()
{
   return TypePyramide().PyramideGaussienne().Val().NbInLastOctave();
}

const cTplValGesInit< int > & cPyramideImage::NbInLastOctave()const 
{
   return TypePyramide().PyramideGaussienne().Val().NbInLastOctave();
}


cTplValGesInit< int > & cPyramideImage::IndexFreqInFirstOctave()
{
   return TypePyramide().PyramideGaussienne().Val().IndexFreqInFirstOctave();
}

const cTplValGesInit< int > & cPyramideImage::IndexFreqInFirstOctave()const 
{
   return TypePyramide().PyramideGaussienne().Val().IndexFreqInFirstOctave();
}


int & cPyramideImage::NivOctaveMax()
{
   return TypePyramide().PyramideGaussienne().Val().NivOctaveMax();
}

const int & cPyramideImage::NivOctaveMax()const 
{
   return TypePyramide().PyramideGaussienne().Val().NivOctaveMax();
}


cTplValGesInit< double > & cPyramideImage::ConvolFirstImage()
{
   return TypePyramide().PyramideGaussienne().Val().ConvolFirstImage();
}

const cTplValGesInit< double > & cPyramideImage::ConvolFirstImage()const 
{
   return TypePyramide().PyramideGaussienne().Val().ConvolFirstImage();
}


cTplValGesInit< double > & cPyramideImage::EpsilonGauss()
{
   return TypePyramide().PyramideGaussienne().Val().EpsilonGauss();
}

const cTplValGesInit< double > & cPyramideImage::EpsilonGauss()const 
{
   return TypePyramide().PyramideGaussienne().Val().EpsilonGauss();
}


cTplValGesInit< int > & cPyramideImage::NbShift()
{
   return TypePyramide().PyramideGaussienne().Val().NbShift();
}

const cTplValGesInit< int > & cPyramideImage::NbShift()const 
{
   return TypePyramide().PyramideGaussienne().Val().NbShift();
}


cTplValGesInit< int > & cPyramideImage::SurEchIntegralGauss()
{
   return TypePyramide().PyramideGaussienne().Val().SurEchIntegralGauss();
}

const cTplValGesInit< int > & cPyramideImage::SurEchIntegralGauss()const 
{
   return TypePyramide().PyramideGaussienne().Val().SurEchIntegralGauss();
}


cTplValGesInit< bool > & cPyramideImage::ConvolIncrem()
{
   return TypePyramide().PyramideGaussienne().Val().ConvolIncrem();
}

const cTplValGesInit< bool > & cPyramideImage::ConvolIncrem()const 
{
   return TypePyramide().PyramideGaussienne().Val().ConvolIncrem();
}


cTplValGesInit< cPyramideGaussienne > & cPyramideImage::PyramideGaussienne()
{
   return TypePyramide().PyramideGaussienne();
}

const cTplValGesInit< cPyramideGaussienne > & cPyramideImage::PyramideGaussienne()const 
{
   return TypePyramide().PyramideGaussienne();
}


cTypePyramide & cPyramideImage::TypePyramide()
{
   return mTypePyramide;
}

const cTypePyramide & cPyramideImage::TypePyramide()const 
{
   return mTypePyramide;
}

void  BinaryUnDumpFromFile(cPyramideImage & anObj,ELISE_fp & aFp)
{
   { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cTypeNumeriqueOfNiv aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.TypeNumeriqueOfNiv().push_back(aVal);
        }
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.MaximDyn().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.MaximDyn().ValForcedForUnUmp(),aFp);
        }
        else  anObj.MaximDyn().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ValMaxForDyn().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ValMaxForDyn().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ValMaxForDyn().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ReducDemiImage().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ReducDemiImage().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ReducDemiImage().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.TypePyramide(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cPyramideImage & anObj)
{
    BinaryDumpInFile(aFp,(int)anObj.TypeNumeriqueOfNiv().size());
    for(  std::list< cTypeNumeriqueOfNiv >::const_iterator iT=anObj.TypeNumeriqueOfNiv().begin();
         iT!=anObj.TypeNumeriqueOfNiv().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.MaximDyn().IsInit());
    if (anObj.MaximDyn().IsInit()) BinaryDumpInFile(aFp,anObj.MaximDyn().Val());
    BinaryDumpInFile(aFp,anObj.ValMaxForDyn().IsInit());
    if (anObj.ValMaxForDyn().IsInit()) BinaryDumpInFile(aFp,anObj.ValMaxForDyn().Val());
    BinaryDumpInFile(aFp,anObj.ReducDemiImage().IsInit());
    if (anObj.ReducDemiImage().IsInit()) BinaryDumpInFile(aFp,anObj.ReducDemiImage().Val());
    BinaryDumpInFile(aFp,anObj.TypePyramide());
}

cElXMLTree * ToXMLTree(const cPyramideImage & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"PyramideImage",eXMLBranche);
  for
  (       std::list< cTypeNumeriqueOfNiv >::const_iterator it=anObj.TypeNumeriqueOfNiv().begin();
      it !=anObj.TypeNumeriqueOfNiv().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("TypeNumeriqueOfNiv"));
   if (anObj.MaximDyn().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("MaximDyn"),anObj.MaximDyn().Val())->ReTagThis("MaximDyn"));
   if (anObj.ValMaxForDyn().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ValMaxForDyn"),anObj.ValMaxForDyn().Val())->ReTagThis("ValMaxForDyn"));
   if (anObj.ReducDemiImage().IsInit())
      aRes->AddFils(ToXMLTree(std::string("ReducDemiImage"),anObj.ReducDemiImage().Val())->ReTagThis("ReducDemiImage"));
   aRes->AddFils(ToXMLTree(anObj.TypePyramide())->ReTagThis("TypePyramide"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cPyramideImage & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.TypeNumeriqueOfNiv(),aTree->GetAll("TypeNumeriqueOfNiv",false,1));

   xml_init(anObj.MaximDyn(),aTree->Get("MaximDyn",1)); //tototo 

   xml_init(anObj.ValMaxForDyn(),aTree->Get("ValMaxForDyn",1)); //tototo 

   xml_init(anObj.ReducDemiImage(),aTree->Get("ReducDemiImage",1),eReducDemiImage(eRDI_121)); //tototo 

   xml_init(anObj.TypePyramide(),aTree->Get("TypePyramide",1)); //tototo 
}

std::string  Mangling( cPyramideImage *) {return "DA6F2C90A842D88CFDBF";};


std::list< cImageDigeo > & cDigeoSectionImages::ImageDigeo()
{
   return mImageDigeo;
}

const std::list< cImageDigeo > & cDigeoSectionImages::ImageDigeo()const 
{
   return mImageDigeo;
}


std::list< cTypeNumeriqueOfNiv > & cDigeoSectionImages::TypeNumeriqueOfNiv()
{
   return PyramideImage().TypeNumeriqueOfNiv();
}

const std::list< cTypeNumeriqueOfNiv > & cDigeoSectionImages::TypeNumeriqueOfNiv()const 
{
   return PyramideImage().TypeNumeriqueOfNiv();
}


cTplValGesInit< bool > & cDigeoSectionImages::MaximDyn()
{
   return PyramideImage().MaximDyn();
}

const cTplValGesInit< bool > & cDigeoSectionImages::MaximDyn()const 
{
   return PyramideImage().MaximDyn();
}


cTplValGesInit< double > & cDigeoSectionImages::ValMaxForDyn()
{
   return PyramideImage().ValMaxForDyn();
}

const cTplValGesInit< double > & cDigeoSectionImages::ValMaxForDyn()const 
{
   return PyramideImage().ValMaxForDyn();
}


cTplValGesInit< eReducDemiImage > & cDigeoSectionImages::ReducDemiImage()
{
   return PyramideImage().ReducDemiImage();
}

const cTplValGesInit< eReducDemiImage > & cDigeoSectionImages::ReducDemiImage()const 
{
   return PyramideImage().ReducDemiImage();
}


cTplValGesInit< int > & cDigeoSectionImages::NivPyramBasique()
{
   return PyramideImage().TypePyramide().NivPyramBasique();
}

const cTplValGesInit< int > & cDigeoSectionImages::NivPyramBasique()const 
{
   return PyramideImage().TypePyramide().NivPyramBasique();
}


cTplValGesInit< int > & cDigeoSectionImages::NbByOctave()
{
   return PyramideImage().TypePyramide().PyramideGaussienne().Val().NbByOctave();
}

const cTplValGesInit< int > & cDigeoSectionImages::NbByOctave()const 
{
   return PyramideImage().TypePyramide().PyramideGaussienne().Val().NbByOctave();
}


cTplValGesInit< double > & cDigeoSectionImages::Sigma0()
{
   return PyramideImage().TypePyramide().PyramideGaussienne().Val().Sigma0();
}

const cTplValGesInit< double > & cDigeoSectionImages::Sigma0()const 
{
   return PyramideImage().TypePyramide().PyramideGaussienne().Val().Sigma0();
}


cTplValGesInit< double > & cDigeoSectionImages::SigmaN()
{
   return PyramideImage().TypePyramide().PyramideGaussienne().Val().SigmaN();
}

const cTplValGesInit< double > & cDigeoSectionImages::SigmaN()const 
{
   return PyramideImage().TypePyramide().PyramideGaussienne().Val().SigmaN();
}


cTplValGesInit< int > & cDigeoSectionImages::NbInLastOctave()
{
   return PyramideImage().TypePyramide().PyramideGaussienne().Val().NbInLastOctave();
}

const cTplValGesInit< int > & cDigeoSectionImages::NbInLastOctave()const 
{
   return PyramideImage().TypePyramide().PyramideGaussienne().Val().NbInLastOctave();
}


cTplValGesInit< int > & cDigeoSectionImages::IndexFreqInFirstOctave()
{
   return PyramideImage().TypePyramide().PyramideGaussienne().Val().IndexFreqInFirstOctave();
}

const cTplValGesInit< int > & cDigeoSectionImages::IndexFreqInFirstOctave()const 
{
   return PyramideImage().TypePyramide().PyramideGaussienne().Val().IndexFreqInFirstOctave();
}


int & cDigeoSectionImages::NivOctaveMax()
{
   return PyramideImage().TypePyramide().PyramideGaussienne().Val().NivOctaveMax();
}

const int & cDigeoSectionImages::NivOctaveMax()const 
{
   return PyramideImage().TypePyramide().PyramideGaussienne().Val().NivOctaveMax();
}


cTplValGesInit< double > & cDigeoSectionImages::ConvolFirstImage()
{
   return PyramideImage().TypePyramide().PyramideGaussienne().Val().ConvolFirstImage();
}

const cTplValGesInit< double > & cDigeoSectionImages::ConvolFirstImage()const 
{
   return PyramideImage().TypePyramide().PyramideGaussienne().Val().ConvolFirstImage();
}


cTplValGesInit< double > & cDigeoSectionImages::EpsilonGauss()
{
   return PyramideImage().TypePyramide().PyramideGaussienne().Val().EpsilonGauss();
}

const cTplValGesInit< double > & cDigeoSectionImages::EpsilonGauss()const 
{
   return PyramideImage().TypePyramide().PyramideGaussienne().Val().EpsilonGauss();
}


cTplValGesInit< int > & cDigeoSectionImages::NbShift()
{
   return PyramideImage().TypePyramide().PyramideGaussienne().Val().NbShift();
}

const cTplValGesInit< int > & cDigeoSectionImages::NbShift()const 
{
   return PyramideImage().TypePyramide().PyramideGaussienne().Val().NbShift();
}


cTplValGesInit< int > & cDigeoSectionImages::SurEchIntegralGauss()
{
   return PyramideImage().TypePyramide().PyramideGaussienne().Val().SurEchIntegralGauss();
}

const cTplValGesInit< int > & cDigeoSectionImages::SurEchIntegralGauss()const 
{
   return PyramideImage().TypePyramide().PyramideGaussienne().Val().SurEchIntegralGauss();
}


cTplValGesInit< bool > & cDigeoSectionImages::ConvolIncrem()
{
   return PyramideImage().TypePyramide().PyramideGaussienne().Val().ConvolIncrem();
}

const cTplValGesInit< bool > & cDigeoSectionImages::ConvolIncrem()const 
{
   return PyramideImage().TypePyramide().PyramideGaussienne().Val().ConvolIncrem();
}


cTplValGesInit< cPyramideGaussienne > & cDigeoSectionImages::PyramideGaussienne()
{
   return PyramideImage().TypePyramide().PyramideGaussienne();
}

const cTplValGesInit< cPyramideGaussienne > & cDigeoSectionImages::PyramideGaussienne()const 
{
   return PyramideImage().TypePyramide().PyramideGaussienne();
}


cTypePyramide & cDigeoSectionImages::TypePyramide()
{
   return PyramideImage().TypePyramide();
}

const cTypePyramide & cDigeoSectionImages::TypePyramide()const 
{
   return PyramideImage().TypePyramide();
}


cPyramideImage & cDigeoSectionImages::PyramideImage()
{
   return mPyramideImage;
}

const cPyramideImage & cDigeoSectionImages::PyramideImage()const 
{
   return mPyramideImage;
}

void  BinaryUnDumpFromFile(cDigeoSectionImages & anObj,ELISE_fp & aFp)
{
   { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cImageDigeo aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.ImageDigeo().push_back(aVal);
        }
  } ;
    BinaryUnDumpFromFile(anObj.PyramideImage(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cDigeoSectionImages & anObj)
{
    BinaryDumpInFile(aFp,(int)anObj.ImageDigeo().size());
    for(  std::list< cImageDigeo >::const_iterator iT=anObj.ImageDigeo().begin();
         iT!=anObj.ImageDigeo().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.PyramideImage());
}

cElXMLTree * ToXMLTree(const cDigeoSectionImages & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"DigeoSectionImages",eXMLBranche);
  for
  (       std::list< cImageDigeo >::const_iterator it=anObj.ImageDigeo().begin();
      it !=anObj.ImageDigeo().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("ImageDigeo"));
   aRes->AddFils(ToXMLTree(anObj.PyramideImage())->ReTagThis("PyramideImage"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cDigeoSectionImages & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.ImageDigeo(),aTree->GetAll("ImageDigeo",false,1));

   xml_init(anObj.PyramideImage(),aTree->Get("PyramideImage",1)); //tototo 
}

std::string  Mangling( cDigeoSectionImages *) {return "4BF3C3CA3CBF26F8FE3F";};


eTypeTopolPt & cOneCarac::Type()
{
   return mType;
}

const eTypeTopolPt & cOneCarac::Type()const 
{
   return mType;
}

void  BinaryUnDumpFromFile(cOneCarac & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Type(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cOneCarac & anObj)
{
    BinaryDumpInFile(aFp,anObj.Type());
}

cElXMLTree * ToXMLTree(const cOneCarac & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"OneCarac",eXMLBranche);
   aRes->AddFils(ToXMLTree(std::string("Type"),anObj.Type())->ReTagThis("Type"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cOneCarac & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Type(),aTree->Get("Type",1)); //tototo 
}

std::string  Mangling( cOneCarac *) {return "A4C8D304A8CED5B2FF3F";};


std::list< cOneCarac > & cCaracTopo::OneCarac()
{
   return mOneCarac;
}

const std::list< cOneCarac > & cCaracTopo::OneCarac()const 
{
   return mOneCarac;
}

void  BinaryUnDumpFromFile(cCaracTopo & anObj,ELISE_fp & aFp)
{
   { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cOneCarac aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.OneCarac().push_back(aVal);
        }
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cCaracTopo & anObj)
{
    BinaryDumpInFile(aFp,(int)anObj.OneCarac().size());
    for(  std::list< cOneCarac >::const_iterator iT=anObj.OneCarac().begin();
         iT!=anObj.OneCarac().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
}

cElXMLTree * ToXMLTree(const cCaracTopo & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"CaracTopo",eXMLBranche);
  for
  (       std::list< cOneCarac >::const_iterator it=anObj.OneCarac().begin();
      it !=anObj.OneCarac().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("OneCarac"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cCaracTopo & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.OneCarac(),aTree->GetAll("OneCarac",false,1));
}

std::string  Mangling( cCaracTopo *) {return "6AE63E2144CE7F94FF3F";};


cTplValGesInit< bool > & cSiftCarac::DoMax()
{
   return mDoMax;
}

const cTplValGesInit< bool > & cSiftCarac::DoMax()const 
{
   return mDoMax;
}


cTplValGesInit< bool > & cSiftCarac::DoMin()
{
   return mDoMin;
}

const cTplValGesInit< bool > & cSiftCarac::DoMin()const 
{
   return mDoMin;
}


cTplValGesInit< int > & cSiftCarac::NivEstimGradMoy()
{
   return mNivEstimGradMoy;
}

const cTplValGesInit< int > & cSiftCarac::NivEstimGradMoy()const 
{
   return mNivEstimGradMoy;
}


cTplValGesInit< double > & cSiftCarac::RatioAllongMin()
{
   return mRatioAllongMin;
}

const cTplValGesInit< double > & cSiftCarac::RatioAllongMin()const 
{
   return mRatioAllongMin;
}


cTplValGesInit< double > & cSiftCarac::RatioGrad()
{
   return mRatioGrad;
}

const cTplValGesInit< double > & cSiftCarac::RatioGrad()const 
{
   return mRatioGrad;
}

void  BinaryUnDumpFromFile(cSiftCarac & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.DoMax().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.DoMax().ValForcedForUnUmp(),aFp);
        }
        else  anObj.DoMax().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.DoMin().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.DoMin().ValForcedForUnUmp(),aFp);
        }
        else  anObj.DoMin().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NivEstimGradMoy().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NivEstimGradMoy().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NivEstimGradMoy().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.RatioAllongMin().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.RatioAllongMin().ValForcedForUnUmp(),aFp);
        }
        else  anObj.RatioAllongMin().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.RatioGrad().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.RatioGrad().ValForcedForUnUmp(),aFp);
        }
        else  anObj.RatioGrad().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cSiftCarac & anObj)
{
    BinaryDumpInFile(aFp,anObj.DoMax().IsInit());
    if (anObj.DoMax().IsInit()) BinaryDumpInFile(aFp,anObj.DoMax().Val());
    BinaryDumpInFile(aFp,anObj.DoMin().IsInit());
    if (anObj.DoMin().IsInit()) BinaryDumpInFile(aFp,anObj.DoMin().Val());
    BinaryDumpInFile(aFp,anObj.NivEstimGradMoy().IsInit());
    if (anObj.NivEstimGradMoy().IsInit()) BinaryDumpInFile(aFp,anObj.NivEstimGradMoy().Val());
    BinaryDumpInFile(aFp,anObj.RatioAllongMin().IsInit());
    if (anObj.RatioAllongMin().IsInit()) BinaryDumpInFile(aFp,anObj.RatioAllongMin().Val());
    BinaryDumpInFile(aFp,anObj.RatioGrad().IsInit());
    if (anObj.RatioGrad().IsInit()) BinaryDumpInFile(aFp,anObj.RatioGrad().Val());
}

cElXMLTree * ToXMLTree(const cSiftCarac & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"SiftCarac",eXMLBranche);
   if (anObj.DoMax().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DoMax"),anObj.DoMax().Val())->ReTagThis("DoMax"));
   if (anObj.DoMin().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DoMin"),anObj.DoMin().Val())->ReTagThis("DoMin"));
   if (anObj.NivEstimGradMoy().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NivEstimGradMoy"),anObj.NivEstimGradMoy().Val())->ReTagThis("NivEstimGradMoy"));
   if (anObj.RatioAllongMin().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("RatioAllongMin"),anObj.RatioAllongMin().Val())->ReTagThis("RatioAllongMin"));
   if (anObj.RatioGrad().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("RatioGrad"),anObj.RatioGrad().Val())->ReTagThis("RatioGrad"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cSiftCarac & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.DoMax(),aTree->Get("DoMax",1),bool(true)); //tototo 

   xml_init(anObj.DoMin(),aTree->Get("DoMin",1),bool(true)); //tototo 

   xml_init(anObj.NivEstimGradMoy(),aTree->Get("NivEstimGradMoy",1),int(4)); //tototo 

   xml_init(anObj.RatioAllongMin(),aTree->Get("RatioAllongMin",1),double(8.0)); //tototo 

   xml_init(anObj.RatioGrad(),aTree->Get("RatioGrad",1),double(0.01)); //tototo 
}

std::string  Mangling( cSiftCarac *) {return "3EABBA5BE692B5DEFE3F";};


bool & cSectionCaracImages::ComputeCarac()
{
   return mComputeCarac;
}

const bool & cSectionCaracImages::ComputeCarac()const 
{
   return mComputeCarac;
}


std::list< cOneCarac > & cSectionCaracImages::OneCarac()
{
   return CaracTopo().Val().OneCarac();
}

const std::list< cOneCarac > & cSectionCaracImages::OneCarac()const 
{
   return CaracTopo().Val().OneCarac();
}


cTplValGesInit< cCaracTopo > & cSectionCaracImages::CaracTopo()
{
   return mCaracTopo;
}

const cTplValGesInit< cCaracTopo > & cSectionCaracImages::CaracTopo()const 
{
   return mCaracTopo;
}


cTplValGesInit< bool > & cSectionCaracImages::DoMax()
{
   return SiftCarac().Val().DoMax();
}

const cTplValGesInit< bool > & cSectionCaracImages::DoMax()const 
{
   return SiftCarac().Val().DoMax();
}


cTplValGesInit< bool > & cSectionCaracImages::DoMin()
{
   return SiftCarac().Val().DoMin();
}

const cTplValGesInit< bool > & cSectionCaracImages::DoMin()const 
{
   return SiftCarac().Val().DoMin();
}


cTplValGesInit< int > & cSectionCaracImages::NivEstimGradMoy()
{
   return SiftCarac().Val().NivEstimGradMoy();
}

const cTplValGesInit< int > & cSectionCaracImages::NivEstimGradMoy()const 
{
   return SiftCarac().Val().NivEstimGradMoy();
}


cTplValGesInit< double > & cSectionCaracImages::RatioAllongMin()
{
   return SiftCarac().Val().RatioAllongMin();
}

const cTplValGesInit< double > & cSectionCaracImages::RatioAllongMin()const 
{
   return SiftCarac().Val().RatioAllongMin();
}


cTplValGesInit< double > & cSectionCaracImages::RatioGrad()
{
   return SiftCarac().Val().RatioGrad();
}

const cTplValGesInit< double > & cSectionCaracImages::RatioGrad()const 
{
   return SiftCarac().Val().RatioGrad();
}


cTplValGesInit< cSiftCarac > & cSectionCaracImages::SiftCarac()
{
   return mSiftCarac;
}

const cTplValGesInit< cSiftCarac > & cSectionCaracImages::SiftCarac()const 
{
   return mSiftCarac;
}

void  BinaryUnDumpFromFile(cSectionCaracImages & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.ComputeCarac(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.CaracTopo().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.CaracTopo().ValForcedForUnUmp(),aFp);
        }
        else  anObj.CaracTopo().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SiftCarac().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SiftCarac().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SiftCarac().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cSectionCaracImages & anObj)
{
    BinaryDumpInFile(aFp,anObj.ComputeCarac());
    BinaryDumpInFile(aFp,anObj.CaracTopo().IsInit());
    if (anObj.CaracTopo().IsInit()) BinaryDumpInFile(aFp,anObj.CaracTopo().Val());
    BinaryDumpInFile(aFp,anObj.SiftCarac().IsInit());
    if (anObj.SiftCarac().IsInit()) BinaryDumpInFile(aFp,anObj.SiftCarac().Val());
}

cElXMLTree * ToXMLTree(const cSectionCaracImages & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"SectionCaracImages",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("ComputeCarac"),anObj.ComputeCarac())->ReTagThis("ComputeCarac"));
   if (anObj.CaracTopo().IsInit())
      aRes->AddFils(ToXMLTree(anObj.CaracTopo().Val())->ReTagThis("CaracTopo"));
   if (anObj.SiftCarac().IsInit())
      aRes->AddFils(ToXMLTree(anObj.SiftCarac().Val())->ReTagThis("SiftCarac"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cSectionCaracImages & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.ComputeCarac(),aTree->Get("ComputeCarac",1)); //tototo 

   xml_init(anObj.CaracTopo(),aTree->Get("CaracTopo",1)); //tototo 

   xml_init(anObj.SiftCarac(),aTree->Get("SiftCarac",1)); //tototo 
}

std::string  Mangling( cSectionCaracImages *) {return "15E3137B33A178E7FD3F";};


int & cGenereRandomRect::NbRect()
{
   return mNbRect;
}

const int & cGenereRandomRect::NbRect()const 
{
   return mNbRect;
}


int & cGenereRandomRect::SzRect()
{
   return mSzRect;
}

const int & cGenereRandomRect::SzRect()const 
{
   return mSzRect;
}

void  BinaryUnDumpFromFile(cGenereRandomRect & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.NbRect(),aFp);
    BinaryUnDumpFromFile(anObj.SzRect(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cGenereRandomRect & anObj)
{
    BinaryDumpInFile(aFp,anObj.NbRect());
    BinaryDumpInFile(aFp,anObj.SzRect());
}

cElXMLTree * ToXMLTree(const cGenereRandomRect & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"GenereRandomRect",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("NbRect"),anObj.NbRect())->ReTagThis("NbRect"));
   aRes->AddFils(::ToXMLTree(std::string("SzRect"),anObj.SzRect())->ReTagThis("SzRect"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cGenereRandomRect & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.NbRect(),aTree->Get("NbRect",1)); //tototo 

   xml_init(anObj.SzRect(),aTree->Get("SzRect",1)); //tototo 
}

std::string  Mangling( cGenereRandomRect *) {return "B3169794113A5EB7FE3F";};


int & cGenereCarroyage::PerX()
{
   return mPerX;
}

const int & cGenereCarroyage::PerX()const 
{
   return mPerX;
}


int & cGenereCarroyage::PerY()
{
   return mPerY;
}

const int & cGenereCarroyage::PerY()const 
{
   return mPerY;
}

void  BinaryUnDumpFromFile(cGenereCarroyage & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.PerX(),aFp);
    BinaryUnDumpFromFile(anObj.PerY(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cGenereCarroyage & anObj)
{
    BinaryDumpInFile(aFp,anObj.PerX());
    BinaryDumpInFile(aFp,anObj.PerY());
}

cElXMLTree * ToXMLTree(const cGenereCarroyage & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"GenereCarroyage",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("PerX"),anObj.PerX())->ReTagThis("PerX"));
   aRes->AddFils(::ToXMLTree(std::string("PerY"),anObj.PerY())->ReTagThis("PerY"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cGenereCarroyage & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.PerX(),aTree->Get("PerX",1)); //tototo 

   xml_init(anObj.PerY(),aTree->Get("PerY",1)); //tototo 
}

std::string  Mangling( cGenereCarroyage *) {return "5417934D873F73D9FE3F";};


int & cGenereAllRandom::SzFilter()
{
   return mSzFilter;
}

const int & cGenereAllRandom::SzFilter()const 
{
   return mSzFilter;
}

void  BinaryUnDumpFromFile(cGenereAllRandom & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.SzFilter(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cGenereAllRandom & anObj)
{
    BinaryDumpInFile(aFp,anObj.SzFilter());
}

cElXMLTree * ToXMLTree(const cGenereAllRandom & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"GenereAllRandom",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("SzFilter"),anObj.SzFilter())->ReTagThis("SzFilter"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cGenereAllRandom & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.SzFilter(),aTree->Get("SzFilter",1)); //tototo 
}

std::string  Mangling( cGenereAllRandom *) {return "3F464AECA1F873DEFD3F";};


int & cSectionTest::NbRect()
{
   return GenereRandomRect().Val().NbRect();
}

const int & cSectionTest::NbRect()const 
{
   return GenereRandomRect().Val().NbRect();
}


int & cSectionTest::SzRect()
{
   return GenereRandomRect().Val().SzRect();
}

const int & cSectionTest::SzRect()const 
{
   return GenereRandomRect().Val().SzRect();
}


cTplValGesInit< cGenereRandomRect > & cSectionTest::GenereRandomRect()
{
   return mGenereRandomRect;
}

const cTplValGesInit< cGenereRandomRect > & cSectionTest::GenereRandomRect()const 
{
   return mGenereRandomRect;
}


int & cSectionTest::PerX()
{
   return GenereCarroyage().Val().PerX();
}

const int & cSectionTest::PerX()const 
{
   return GenereCarroyage().Val().PerX();
}


int & cSectionTest::PerY()
{
   return GenereCarroyage().Val().PerY();
}

const int & cSectionTest::PerY()const 
{
   return GenereCarroyage().Val().PerY();
}


cTplValGesInit< cGenereCarroyage > & cSectionTest::GenereCarroyage()
{
   return mGenereCarroyage;
}

const cTplValGesInit< cGenereCarroyage > & cSectionTest::GenereCarroyage()const 
{
   return mGenereCarroyage;
}


int & cSectionTest::SzFilter()
{
   return GenereAllRandom().Val().SzFilter();
}

const int & cSectionTest::SzFilter()const 
{
   return GenereAllRandom().Val().SzFilter();
}


cTplValGesInit< cGenereAllRandom > & cSectionTest::GenereAllRandom()
{
   return mGenereAllRandom;
}

const cTplValGesInit< cGenereAllRandom > & cSectionTest::GenereAllRandom()const 
{
   return mGenereAllRandom;
}


cTplValGesInit< bool > & cSectionTest::VerifExtrema()
{
   return mVerifExtrema;
}

const cTplValGesInit< bool > & cSectionTest::VerifExtrema()const 
{
   return mVerifExtrema;
}

void  BinaryUnDumpFromFile(cSectionTest & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.GenereRandomRect().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.GenereRandomRect().ValForcedForUnUmp(),aFp);
        }
        else  anObj.GenereRandomRect().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.GenereCarroyage().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.GenereCarroyage().ValForcedForUnUmp(),aFp);
        }
        else  anObj.GenereCarroyage().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.GenereAllRandom().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.GenereAllRandom().ValForcedForUnUmp(),aFp);
        }
        else  anObj.GenereAllRandom().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.VerifExtrema().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.VerifExtrema().ValForcedForUnUmp(),aFp);
        }
        else  anObj.VerifExtrema().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cSectionTest & anObj)
{
    BinaryDumpInFile(aFp,anObj.GenereRandomRect().IsInit());
    if (anObj.GenereRandomRect().IsInit()) BinaryDumpInFile(aFp,anObj.GenereRandomRect().Val());
    BinaryDumpInFile(aFp,anObj.GenereCarroyage().IsInit());
    if (anObj.GenereCarroyage().IsInit()) BinaryDumpInFile(aFp,anObj.GenereCarroyage().Val());
    BinaryDumpInFile(aFp,anObj.GenereAllRandom().IsInit());
    if (anObj.GenereAllRandom().IsInit()) BinaryDumpInFile(aFp,anObj.GenereAllRandom().Val());
    BinaryDumpInFile(aFp,anObj.VerifExtrema().IsInit());
    if (anObj.VerifExtrema().IsInit()) BinaryDumpInFile(aFp,anObj.VerifExtrema().Val());
}

cElXMLTree * ToXMLTree(const cSectionTest & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"SectionTest",eXMLBranche);
   if (anObj.GenereRandomRect().IsInit())
      aRes->AddFils(ToXMLTree(anObj.GenereRandomRect().Val())->ReTagThis("GenereRandomRect"));
   if (anObj.GenereCarroyage().IsInit())
      aRes->AddFils(ToXMLTree(anObj.GenereCarroyage().Val())->ReTagThis("GenereCarroyage"));
   if (anObj.GenereAllRandom().IsInit())
      aRes->AddFils(ToXMLTree(anObj.GenereAllRandom().Val())->ReTagThis("GenereAllRandom"));
   if (anObj.VerifExtrema().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("VerifExtrema"),anObj.VerifExtrema().Val())->ReTagThis("VerifExtrema"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cSectionTest & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.GenereRandomRect(),aTree->Get("GenereRandomRect",1)); //tototo 

   xml_init(anObj.GenereCarroyage(),aTree->Get("GenereCarroyage",1)); //tototo 

   xml_init(anObj.GenereAllRandom(),aTree->Get("GenereAllRandom",1)); //tototo 

   xml_init(anObj.VerifExtrema(),aTree->Get("VerifExtrema",1),bool(false)); //tototo 
}

std::string  Mangling( cSectionTest *) {return "FCDB69724757B2EDFE3F";};


cTplValGesInit< std::string > & cSauvPyram::Dir()
{
   return mDir;
}

const cTplValGesInit< std::string > & cSauvPyram::Dir()const 
{
   return mDir;
}


cTplValGesInit< bool > & cSauvPyram::Glob()
{
   return mGlob;
}

const cTplValGesInit< bool > & cSauvPyram::Glob()const 
{
   return mGlob;
}


cTplValGesInit< std::string > & cSauvPyram::Key()
{
   return mKey;
}

const cTplValGesInit< std::string > & cSauvPyram::Key()const 
{
   return mKey;
}


cTplValGesInit< int > & cSauvPyram::StripTifFile()
{
   return mStripTifFile;
}

const cTplValGesInit< int > & cSauvPyram::StripTifFile()const 
{
   return mStripTifFile;
}


cTplValGesInit< bool > & cSauvPyram::Force8B()
{
   return mForce8B;
}

const cTplValGesInit< bool > & cSauvPyram::Force8B()const 
{
   return mForce8B;
}


cTplValGesInit< double > & cSauvPyram::Dyn()
{
   return mDyn;
}

const cTplValGesInit< double > & cSauvPyram::Dyn()const 
{
   return mDyn;
}

void  BinaryUnDumpFromFile(cSauvPyram & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Dir().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Dir().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Dir().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Glob().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Glob().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Glob().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Key().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Key().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Key().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.StripTifFile().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.StripTifFile().ValForcedForUnUmp(),aFp);
        }
        else  anObj.StripTifFile().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Force8B().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Force8B().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Force8B().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Dyn().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Dyn().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Dyn().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cSauvPyram & anObj)
{
    BinaryDumpInFile(aFp,anObj.Dir().IsInit());
    if (anObj.Dir().IsInit()) BinaryDumpInFile(aFp,anObj.Dir().Val());
    BinaryDumpInFile(aFp,anObj.Glob().IsInit());
    if (anObj.Glob().IsInit()) BinaryDumpInFile(aFp,anObj.Glob().Val());
    BinaryDumpInFile(aFp,anObj.Key().IsInit());
    if (anObj.Key().IsInit()) BinaryDumpInFile(aFp,anObj.Key().Val());
    BinaryDumpInFile(aFp,anObj.StripTifFile().IsInit());
    if (anObj.StripTifFile().IsInit()) BinaryDumpInFile(aFp,anObj.StripTifFile().Val());
    BinaryDumpInFile(aFp,anObj.Force8B().IsInit());
    if (anObj.Force8B().IsInit()) BinaryDumpInFile(aFp,anObj.Force8B().Val());
    BinaryDumpInFile(aFp,anObj.Dyn().IsInit());
    if (anObj.Dyn().IsInit()) BinaryDumpInFile(aFp,anObj.Dyn().Val());
}

cElXMLTree * ToXMLTree(const cSauvPyram & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"SauvPyram",eXMLBranche);
   if (anObj.Dir().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Dir"),anObj.Dir().Val())->ReTagThis("Dir"));
   if (anObj.Glob().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Glob"),anObj.Glob().Val())->ReTagThis("Glob"));
   if (anObj.Key().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Key"),anObj.Key().Val())->ReTagThis("Key"));
   if (anObj.StripTifFile().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("StripTifFile"),anObj.StripTifFile().Val())->ReTagThis("StripTifFile"));
   if (anObj.Force8B().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Force8B"),anObj.Force8B().Val())->ReTagThis("Force8B"));
   if (anObj.Dyn().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Dyn"),anObj.Dyn().Val())->ReTagThis("Dyn"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cSauvPyram & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Dir(),aTree->Get("Dir",1),std::string("")); //tototo 

   xml_init(anObj.Glob(),aTree->Get("Glob",1),bool(true)); //tototo 

   xml_init(anObj.Key(),aTree->Get("Key",1),std::string("Key-Assoc-Pyram-MM")); //tototo 

   xml_init(anObj.StripTifFile(),aTree->Get("StripTifFile",1),int(100)); //tototo 

   xml_init(anObj.Force8B(),aTree->Get("Force8B",1),bool(false)); //tototo 

   xml_init(anObj.Dyn(),aTree->Get("Dyn",1),double(1.0)); //tototo 
}

std::string  Mangling( cSauvPyram *) {return "8F0D268600300694FEBF";};


int & cDigeoDecoupageCarac::SzDalle()
{
   return mSzDalle;
}

const int & cDigeoDecoupageCarac::SzDalle()const 
{
   return mSzDalle;
}


int & cDigeoDecoupageCarac::Bord()
{
   return mBord;
}

const int & cDigeoDecoupageCarac::Bord()const 
{
   return mBord;
}

void  BinaryUnDumpFromFile(cDigeoDecoupageCarac & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.SzDalle(),aFp);
    BinaryUnDumpFromFile(anObj.Bord(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cDigeoDecoupageCarac & anObj)
{
    BinaryDumpInFile(aFp,anObj.SzDalle());
    BinaryDumpInFile(aFp,anObj.Bord());
}

cElXMLTree * ToXMLTree(const cDigeoDecoupageCarac & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"DigeoDecoupageCarac",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("SzDalle"),anObj.SzDalle())->ReTagThis("SzDalle"));
   aRes->AddFils(::ToXMLTree(std::string("Bord"),anObj.Bord())->ReTagThis("Bord"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cDigeoDecoupageCarac & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.SzDalle(),aTree->Get("SzDalle",1)); //tototo 

   xml_init(anObj.Bord(),aTree->Get("Bord",1)); //tototo 
}

std::string  Mangling( cDigeoDecoupageCarac *) {return "DC9552C198A336A1FE3F";};


int & cModifGCC::NbByOctave()
{
   return mNbByOctave;
}

const int & cModifGCC::NbByOctave()const 
{
   return mNbByOctave;
}


bool & cModifGCC::ConvolIncrem()
{
   return mConvolIncrem;
}

const bool & cModifGCC::ConvolIncrem()const 
{
   return mConvolIncrem;
}


eTypeNumerique & cModifGCC::TypeNum()
{
   return mTypeNum;
}

const eTypeNumerique & cModifGCC::TypeNum()const 
{
   return mTypeNum;
}

void  BinaryUnDumpFromFile(cModifGCC & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.NbByOctave(),aFp);
    BinaryUnDumpFromFile(anObj.ConvolIncrem(),aFp);
    BinaryUnDumpFromFile(anObj.TypeNum(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cModifGCC & anObj)
{
    BinaryDumpInFile(aFp,anObj.NbByOctave());
    BinaryDumpInFile(aFp,anObj.ConvolIncrem());
    BinaryDumpInFile(aFp,anObj.TypeNum());
}

cElXMLTree * ToXMLTree(const cModifGCC & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ModifGCC",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("NbByOctave"),anObj.NbByOctave())->ReTagThis("NbByOctave"));
   aRes->AddFils(::ToXMLTree(std::string("ConvolIncrem"),anObj.ConvolIncrem())->ReTagThis("ConvolIncrem"));
   aRes->AddFils(::ToXMLTree(std::string("TypeNum"),anObj.TypeNum())->ReTagThis("TypeNum"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cModifGCC & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.NbByOctave(),aTree->Get("NbByOctave",1)); //tototo 

   xml_init(anObj.ConvolIncrem(),aTree->Get("ConvolIncrem",1)); //tototo 

   xml_init(anObj.TypeNum(),aTree->Get("TypeNum",1)); //tototo 
}

std::string  Mangling( cModifGCC *) {return "8B572270006D7AF9FE3F";};


cTplValGesInit< std::string > & cGenereCodeConvol::Dir()
{
   return mDir;
}

const cTplValGesInit< std::string > & cGenereCodeConvol::Dir()const 
{
   return mDir;
}


cTplValGesInit< std::string > & cGenereCodeConvol::File()
{
   return mFile;
}

const cTplValGesInit< std::string > & cGenereCodeConvol::File()const 
{
   return mFile;
}


std::vector< cModifGCC > & cGenereCodeConvol::ModifGCC()
{
   return mModifGCC;
}

const std::vector< cModifGCC > & cGenereCodeConvol::ModifGCC()const 
{
   return mModifGCC;
}

void  BinaryUnDumpFromFile(cGenereCodeConvol & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Dir().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Dir().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Dir().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.File().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.File().ValForcedForUnUmp(),aFp);
        }
        else  anObj.File().SetNoInit();
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cModifGCC aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.ModifGCC().push_back(aVal);
        }
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cGenereCodeConvol & anObj)
{
    BinaryDumpInFile(aFp,anObj.Dir().IsInit());
    if (anObj.Dir().IsInit()) BinaryDumpInFile(aFp,anObj.Dir().Val());
    BinaryDumpInFile(aFp,anObj.File().IsInit());
    if (anObj.File().IsInit()) BinaryDumpInFile(aFp,anObj.File().Val());
    BinaryDumpInFile(aFp,(int)anObj.ModifGCC().size());
    for(  std::vector< cModifGCC >::const_iterator iT=anObj.ModifGCC().begin();
         iT!=anObj.ModifGCC().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
}

cElXMLTree * ToXMLTree(const cGenereCodeConvol & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"GenereCodeConvol",eXMLBranche);
   if (anObj.Dir().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Dir"),anObj.Dir().Val())->ReTagThis("Dir"));
   if (anObj.File().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("File"),anObj.File().Val())->ReTagThis("File"));
  for
  (       std::vector< cModifGCC >::const_iterator it=anObj.ModifGCC().begin();
      it !=anObj.ModifGCC().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("ModifGCC"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cGenereCodeConvol & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Dir(),aTree->Get("Dir",1),std::string("src/uti_image/Digeo/")); //tototo 

   xml_init(anObj.File(),aTree->Get("File",1),std::string("GenConvolSpec")); //tototo 

   xml_init(anObj.ModifGCC(),aTree->GetAll("ModifGCC",false,1));
}

std::string  Mangling( cGenereCodeConvol *) {return "66C8EBA65DCE3097FE3F";};


std::string & cFenVisu::Name()
{
   return mName;
}

const std::string & cFenVisu::Name()const 
{
   return mName;
}


Pt2di & cFenVisu::Sz()
{
   return mSz;
}

const Pt2di & cFenVisu::Sz()const 
{
   return mSz;
}

void  BinaryUnDumpFromFile(cFenVisu & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Name(),aFp);
    BinaryUnDumpFromFile(anObj.Sz(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cFenVisu & anObj)
{
    BinaryDumpInFile(aFp,anObj.Name());
    BinaryDumpInFile(aFp,anObj.Sz());
}

cElXMLTree * ToXMLTree(const cFenVisu & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"FenVisu",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Name"),anObj.Name())->ReTagThis("Name"));
   aRes->AddFils(::ToXMLTree(std::string("Sz"),anObj.Sz())->ReTagThis("Sz"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cFenVisu & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Name(),aTree->Get("Name",1)); //tototo 

   xml_init(anObj.Sz(),aTree->Get("Sz",1)); //tototo 
}

std::string  Mangling( cFenVisu *) {return "90FC0FCCD560C396FC3F";};


cTplValGesInit< std::string > & cSectionWorkSpace::DirectoryChantier()
{
   return mDirectoryChantier;
}

const cTplValGesInit< std::string > & cSectionWorkSpace::DirectoryChantier()const 
{
   return mDirectoryChantier;
}


cTplValGesInit< std::string > & cSectionWorkSpace::Dir()
{
   return SauvPyram().Val().Dir();
}

const cTplValGesInit< std::string > & cSectionWorkSpace::Dir()const 
{
   return SauvPyram().Val().Dir();
}


cTplValGesInit< bool > & cSectionWorkSpace::Glob()
{
   return SauvPyram().Val().Glob();
}

const cTplValGesInit< bool > & cSectionWorkSpace::Glob()const 
{
   return SauvPyram().Val().Glob();
}


cTplValGesInit< std::string > & cSectionWorkSpace::Key()
{
   return SauvPyram().Val().Key();
}

const cTplValGesInit< std::string > & cSectionWorkSpace::Key()const 
{
   return SauvPyram().Val().Key();
}


cTplValGesInit< int > & cSectionWorkSpace::StripTifFile()
{
   return SauvPyram().Val().StripTifFile();
}

const cTplValGesInit< int > & cSectionWorkSpace::StripTifFile()const 
{
   return SauvPyram().Val().StripTifFile();
}


cTplValGesInit< bool > & cSectionWorkSpace::Force8B()
{
   return SauvPyram().Val().Force8B();
}

const cTplValGesInit< bool > & cSectionWorkSpace::Force8B()const 
{
   return SauvPyram().Val().Force8B();
}


cTplValGesInit< double > & cSectionWorkSpace::Dyn()
{
   return SauvPyram().Val().Dyn();
}

const cTplValGesInit< double > & cSectionWorkSpace::Dyn()const 
{
   return SauvPyram().Val().Dyn();
}


cTplValGesInit< cSauvPyram > & cSectionWorkSpace::SauvPyram()
{
   return mSauvPyram;
}

const cTplValGesInit< cSauvPyram > & cSectionWorkSpace::SauvPyram()const 
{
   return mSauvPyram;
}


int & cSectionWorkSpace::SzDalle()
{
   return DigeoDecoupageCarac().Val().SzDalle();
}

const int & cSectionWorkSpace::SzDalle()const 
{
   return DigeoDecoupageCarac().Val().SzDalle();
}


int & cSectionWorkSpace::Bord()
{
   return DigeoDecoupageCarac().Val().Bord();
}

const int & cSectionWorkSpace::Bord()const 
{
   return DigeoDecoupageCarac().Val().Bord();
}


cTplValGesInit< cDigeoDecoupageCarac > & cSectionWorkSpace::DigeoDecoupageCarac()
{
   return mDigeoDecoupageCarac;
}

const cTplValGesInit< cDigeoDecoupageCarac > & cSectionWorkSpace::DigeoDecoupageCarac()const 
{
   return mDigeoDecoupageCarac;
}


cTplValGesInit< bool > & cSectionWorkSpace::ExigeCodeCompile()
{
   return mExigeCodeCompile;
}

const cTplValGesInit< bool > & cSectionWorkSpace::ExigeCodeCompile()const 
{
   return mExigeCodeCompile;
}


cTplValGesInit< cGenereCodeConvol > & cSectionWorkSpace::GenereCodeConvol()
{
   return mGenereCodeConvol;
}

const cTplValGesInit< cGenereCodeConvol > & cSectionWorkSpace::GenereCodeConvol()const 
{
   return mGenereCodeConvol;
}


cTplValGesInit< int > & cSectionWorkSpace::ShowTimes()
{
   return mShowTimes;
}

const cTplValGesInit< int > & cSectionWorkSpace::ShowTimes()const 
{
   return mShowTimes;
}


std::list< cFenVisu > & cSectionWorkSpace::FenVisu()
{
   return mFenVisu;
}

const std::list< cFenVisu > & cSectionWorkSpace::FenVisu()const 
{
   return mFenVisu;
}


cTplValGesInit< bool > & cSectionWorkSpace::ShowConvolSpec()
{
   return mShowConvolSpec;
}

const cTplValGesInit< bool > & cSectionWorkSpace::ShowConvolSpec()const 
{
   return mShowConvolSpec;
}


cTplValGesInit< bool > & cSectionWorkSpace::Verbose()
{
   return mVerbose;
}

const cTplValGesInit< bool > & cSectionWorkSpace::Verbose()const 
{
   return mVerbose;
}

void  BinaryUnDumpFromFile(cSectionWorkSpace & anObj,ELISE_fp & aFp)
{
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
             anObj.SauvPyram().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SauvPyram().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SauvPyram().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.DigeoDecoupageCarac().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.DigeoDecoupageCarac().ValForcedForUnUmp(),aFp);
        }
        else  anObj.DigeoDecoupageCarac().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ExigeCodeCompile().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ExigeCodeCompile().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ExigeCodeCompile().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.GenereCodeConvol().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.GenereCodeConvol().ValForcedForUnUmp(),aFp);
        }
        else  anObj.GenereCodeConvol().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ShowTimes().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ShowTimes().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ShowTimes().SetNoInit();
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cFenVisu aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.FenVisu().push_back(aVal);
        }
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ShowConvolSpec().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ShowConvolSpec().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ShowConvolSpec().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Verbose().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Verbose().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Verbose().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cSectionWorkSpace & anObj)
{
    BinaryDumpInFile(aFp,anObj.DirectoryChantier().IsInit());
    if (anObj.DirectoryChantier().IsInit()) BinaryDumpInFile(aFp,anObj.DirectoryChantier().Val());
    BinaryDumpInFile(aFp,anObj.SauvPyram().IsInit());
    if (anObj.SauvPyram().IsInit()) BinaryDumpInFile(aFp,anObj.SauvPyram().Val());
    BinaryDumpInFile(aFp,anObj.DigeoDecoupageCarac().IsInit());
    if (anObj.DigeoDecoupageCarac().IsInit()) BinaryDumpInFile(aFp,anObj.DigeoDecoupageCarac().Val());
    BinaryDumpInFile(aFp,anObj.ExigeCodeCompile().IsInit());
    if (anObj.ExigeCodeCompile().IsInit()) BinaryDumpInFile(aFp,anObj.ExigeCodeCompile().Val());
    BinaryDumpInFile(aFp,anObj.GenereCodeConvol().IsInit());
    if (anObj.GenereCodeConvol().IsInit()) BinaryDumpInFile(aFp,anObj.GenereCodeConvol().Val());
    BinaryDumpInFile(aFp,anObj.ShowTimes().IsInit());
    if (anObj.ShowTimes().IsInit()) BinaryDumpInFile(aFp,anObj.ShowTimes().Val());
    BinaryDumpInFile(aFp,(int)anObj.FenVisu().size());
    for(  std::list< cFenVisu >::const_iterator iT=anObj.FenVisu().begin();
         iT!=anObj.FenVisu().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.ShowConvolSpec().IsInit());
    if (anObj.ShowConvolSpec().IsInit()) BinaryDumpInFile(aFp,anObj.ShowConvolSpec().Val());
    BinaryDumpInFile(aFp,anObj.Verbose().IsInit());
    if (anObj.Verbose().IsInit()) BinaryDumpInFile(aFp,anObj.Verbose().Val());
}

cElXMLTree * ToXMLTree(const cSectionWorkSpace & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"SectionWorkSpace",eXMLBranche);
   if (anObj.DirectoryChantier().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DirectoryChantier"),anObj.DirectoryChantier().Val())->ReTagThis("DirectoryChantier"));
   if (anObj.SauvPyram().IsInit())
      aRes->AddFils(ToXMLTree(anObj.SauvPyram().Val())->ReTagThis("SauvPyram"));
   if (anObj.DigeoDecoupageCarac().IsInit())
      aRes->AddFils(ToXMLTree(anObj.DigeoDecoupageCarac().Val())->ReTagThis("DigeoDecoupageCarac"));
   if (anObj.ExigeCodeCompile().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ExigeCodeCompile"),anObj.ExigeCodeCompile().Val())->ReTagThis("ExigeCodeCompile"));
   if (anObj.GenereCodeConvol().IsInit())
      aRes->AddFils(ToXMLTree(anObj.GenereCodeConvol().Val())->ReTagThis("GenereCodeConvol"));
   if (anObj.ShowTimes().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ShowTimes"),anObj.ShowTimes().Val())->ReTagThis("ShowTimes"));
  for
  (       std::list< cFenVisu >::const_iterator it=anObj.FenVisu().begin();
      it !=anObj.FenVisu().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("FenVisu"));
   if (anObj.ShowConvolSpec().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ShowConvolSpec"),anObj.ShowConvolSpec().Val())->ReTagThis("ShowConvolSpec"));
   if (anObj.Verbose().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Verbose"),anObj.Verbose().Val())->ReTagThis("Verbose"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cSectionWorkSpace & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.DirectoryChantier(),aTree->Get("DirectoryChantier",1),std::string("")); //tototo 

   xml_init(anObj.SauvPyram(),aTree->Get("SauvPyram",1)); //tototo 

   xml_init(anObj.DigeoDecoupageCarac(),aTree->Get("DigeoDecoupageCarac",1)); //tototo 

   xml_init(anObj.ExigeCodeCompile(),aTree->Get("ExigeCodeCompile",1),bool(true)); //tototo 

   xml_init(anObj.GenereCodeConvol(),aTree->Get("GenereCodeConvol",1)); //tototo 

   xml_init(anObj.ShowTimes(),aTree->Get("ShowTimes",1),int(0)); //tototo 

   xml_init(anObj.FenVisu(),aTree->GetAll("FenVisu",false,1));

   xml_init(anObj.ShowConvolSpec(),aTree->Get("ShowConvolSpec",1),bool(false)); //tototo 

   xml_init(anObj.Verbose(),aTree->Get("Verbose",1),bool(false)); //tototo 
}

std::string  Mangling( cSectionWorkSpace *) {return "9E78B14BACC588ADFE3F";};


cTplValGesInit< cChantierDescripteur > & cParamDigeo::DicoLoc()
{
   return mDicoLoc;
}

const cTplValGesInit< cChantierDescripteur > & cParamDigeo::DicoLoc()const 
{
   return mDicoLoc;
}


std::list< cImageDigeo > & cParamDigeo::ImageDigeo()
{
   return DigeoSectionImages().ImageDigeo();
}

const std::list< cImageDigeo > & cParamDigeo::ImageDigeo()const 
{
   return DigeoSectionImages().ImageDigeo();
}


std::list< cTypeNumeriqueOfNiv > & cParamDigeo::TypeNumeriqueOfNiv()
{
   return DigeoSectionImages().PyramideImage().TypeNumeriqueOfNiv();
}

const std::list< cTypeNumeriqueOfNiv > & cParamDigeo::TypeNumeriqueOfNiv()const 
{
   return DigeoSectionImages().PyramideImage().TypeNumeriqueOfNiv();
}


cTplValGesInit< bool > & cParamDigeo::MaximDyn()
{
   return DigeoSectionImages().PyramideImage().MaximDyn();
}

const cTplValGesInit< bool > & cParamDigeo::MaximDyn()const 
{
   return DigeoSectionImages().PyramideImage().MaximDyn();
}


cTplValGesInit< double > & cParamDigeo::ValMaxForDyn()
{
   return DigeoSectionImages().PyramideImage().ValMaxForDyn();
}

const cTplValGesInit< double > & cParamDigeo::ValMaxForDyn()const 
{
   return DigeoSectionImages().PyramideImage().ValMaxForDyn();
}


cTplValGesInit< eReducDemiImage > & cParamDigeo::ReducDemiImage()
{
   return DigeoSectionImages().PyramideImage().ReducDemiImage();
}

const cTplValGesInit< eReducDemiImage > & cParamDigeo::ReducDemiImage()const 
{
   return DigeoSectionImages().PyramideImage().ReducDemiImage();
}


cTplValGesInit< int > & cParamDigeo::NivPyramBasique()
{
   return DigeoSectionImages().PyramideImage().TypePyramide().NivPyramBasique();
}

const cTplValGesInit< int > & cParamDigeo::NivPyramBasique()const 
{
   return DigeoSectionImages().PyramideImage().TypePyramide().NivPyramBasique();
}


cTplValGesInit< int > & cParamDigeo::NbByOctave()
{
   return DigeoSectionImages().PyramideImage().TypePyramide().PyramideGaussienne().Val().NbByOctave();
}

const cTplValGesInit< int > & cParamDigeo::NbByOctave()const 
{
   return DigeoSectionImages().PyramideImage().TypePyramide().PyramideGaussienne().Val().NbByOctave();
}


cTplValGesInit< double > & cParamDigeo::Sigma0()
{
   return DigeoSectionImages().PyramideImage().TypePyramide().PyramideGaussienne().Val().Sigma0();
}

const cTplValGesInit< double > & cParamDigeo::Sigma0()const 
{
   return DigeoSectionImages().PyramideImage().TypePyramide().PyramideGaussienne().Val().Sigma0();
}


cTplValGesInit< double > & cParamDigeo::SigmaN()
{
   return DigeoSectionImages().PyramideImage().TypePyramide().PyramideGaussienne().Val().SigmaN();
}

const cTplValGesInit< double > & cParamDigeo::SigmaN()const 
{
   return DigeoSectionImages().PyramideImage().TypePyramide().PyramideGaussienne().Val().SigmaN();
}


cTplValGesInit< int > & cParamDigeo::NbInLastOctave()
{
   return DigeoSectionImages().PyramideImage().TypePyramide().PyramideGaussienne().Val().NbInLastOctave();
}

const cTplValGesInit< int > & cParamDigeo::NbInLastOctave()const 
{
   return DigeoSectionImages().PyramideImage().TypePyramide().PyramideGaussienne().Val().NbInLastOctave();
}


cTplValGesInit< int > & cParamDigeo::IndexFreqInFirstOctave()
{
   return DigeoSectionImages().PyramideImage().TypePyramide().PyramideGaussienne().Val().IndexFreqInFirstOctave();
}

const cTplValGesInit< int > & cParamDigeo::IndexFreqInFirstOctave()const 
{
   return DigeoSectionImages().PyramideImage().TypePyramide().PyramideGaussienne().Val().IndexFreqInFirstOctave();
}


int & cParamDigeo::NivOctaveMax()
{
   return DigeoSectionImages().PyramideImage().TypePyramide().PyramideGaussienne().Val().NivOctaveMax();
}

const int & cParamDigeo::NivOctaveMax()const 
{
   return DigeoSectionImages().PyramideImage().TypePyramide().PyramideGaussienne().Val().NivOctaveMax();
}


cTplValGesInit< double > & cParamDigeo::ConvolFirstImage()
{
   return DigeoSectionImages().PyramideImage().TypePyramide().PyramideGaussienne().Val().ConvolFirstImage();
}

const cTplValGesInit< double > & cParamDigeo::ConvolFirstImage()const 
{
   return DigeoSectionImages().PyramideImage().TypePyramide().PyramideGaussienne().Val().ConvolFirstImage();
}


cTplValGesInit< double > & cParamDigeo::EpsilonGauss()
{
   return DigeoSectionImages().PyramideImage().TypePyramide().PyramideGaussienne().Val().EpsilonGauss();
}

const cTplValGesInit< double > & cParamDigeo::EpsilonGauss()const 
{
   return DigeoSectionImages().PyramideImage().TypePyramide().PyramideGaussienne().Val().EpsilonGauss();
}


cTplValGesInit< int > & cParamDigeo::NbShift()
{
   return DigeoSectionImages().PyramideImage().TypePyramide().PyramideGaussienne().Val().NbShift();
}

const cTplValGesInit< int > & cParamDigeo::NbShift()const 
{
   return DigeoSectionImages().PyramideImage().TypePyramide().PyramideGaussienne().Val().NbShift();
}


cTplValGesInit< int > & cParamDigeo::SurEchIntegralGauss()
{
   return DigeoSectionImages().PyramideImage().TypePyramide().PyramideGaussienne().Val().SurEchIntegralGauss();
}

const cTplValGesInit< int > & cParamDigeo::SurEchIntegralGauss()const 
{
   return DigeoSectionImages().PyramideImage().TypePyramide().PyramideGaussienne().Val().SurEchIntegralGauss();
}


cTplValGesInit< bool > & cParamDigeo::ConvolIncrem()
{
   return DigeoSectionImages().PyramideImage().TypePyramide().PyramideGaussienne().Val().ConvolIncrem();
}

const cTplValGesInit< bool > & cParamDigeo::ConvolIncrem()const 
{
   return DigeoSectionImages().PyramideImage().TypePyramide().PyramideGaussienne().Val().ConvolIncrem();
}


cTplValGesInit< cPyramideGaussienne > & cParamDigeo::PyramideGaussienne()
{
   return DigeoSectionImages().PyramideImage().TypePyramide().PyramideGaussienne();
}

const cTplValGesInit< cPyramideGaussienne > & cParamDigeo::PyramideGaussienne()const 
{
   return DigeoSectionImages().PyramideImage().TypePyramide().PyramideGaussienne();
}


cTypePyramide & cParamDigeo::TypePyramide()
{
   return DigeoSectionImages().PyramideImage().TypePyramide();
}

const cTypePyramide & cParamDigeo::TypePyramide()const 
{
   return DigeoSectionImages().PyramideImage().TypePyramide();
}


cPyramideImage & cParamDigeo::PyramideImage()
{
   return DigeoSectionImages().PyramideImage();
}

const cPyramideImage & cParamDigeo::PyramideImage()const 
{
   return DigeoSectionImages().PyramideImage();
}


cDigeoSectionImages & cParamDigeo::DigeoSectionImages()
{
   return mDigeoSectionImages;
}

const cDigeoSectionImages & cParamDigeo::DigeoSectionImages()const 
{
   return mDigeoSectionImages;
}


bool & cParamDigeo::ComputeCarac()
{
   return SectionCaracImages().ComputeCarac();
}

const bool & cParamDigeo::ComputeCarac()const 
{
   return SectionCaracImages().ComputeCarac();
}


std::list< cOneCarac > & cParamDigeo::OneCarac()
{
   return SectionCaracImages().CaracTopo().Val().OneCarac();
}

const std::list< cOneCarac > & cParamDigeo::OneCarac()const 
{
   return SectionCaracImages().CaracTopo().Val().OneCarac();
}


cTplValGesInit< cCaracTopo > & cParamDigeo::CaracTopo()
{
   return SectionCaracImages().CaracTopo();
}

const cTplValGesInit< cCaracTopo > & cParamDigeo::CaracTopo()const 
{
   return SectionCaracImages().CaracTopo();
}


cTplValGesInit< bool > & cParamDigeo::DoMax()
{
   return SectionCaracImages().SiftCarac().Val().DoMax();
}

const cTplValGesInit< bool > & cParamDigeo::DoMax()const 
{
   return SectionCaracImages().SiftCarac().Val().DoMax();
}


cTplValGesInit< bool > & cParamDigeo::DoMin()
{
   return SectionCaracImages().SiftCarac().Val().DoMin();
}

const cTplValGesInit< bool > & cParamDigeo::DoMin()const 
{
   return SectionCaracImages().SiftCarac().Val().DoMin();
}


cTplValGesInit< int > & cParamDigeo::NivEstimGradMoy()
{
   return SectionCaracImages().SiftCarac().Val().NivEstimGradMoy();
}

const cTplValGesInit< int > & cParamDigeo::NivEstimGradMoy()const 
{
   return SectionCaracImages().SiftCarac().Val().NivEstimGradMoy();
}


cTplValGesInit< double > & cParamDigeo::RatioAllongMin()
{
   return SectionCaracImages().SiftCarac().Val().RatioAllongMin();
}

const cTplValGesInit< double > & cParamDigeo::RatioAllongMin()const 
{
   return SectionCaracImages().SiftCarac().Val().RatioAllongMin();
}


cTplValGesInit< double > & cParamDigeo::RatioGrad()
{
   return SectionCaracImages().SiftCarac().Val().RatioGrad();
}

const cTplValGesInit< double > & cParamDigeo::RatioGrad()const 
{
   return SectionCaracImages().SiftCarac().Val().RatioGrad();
}


cTplValGesInit< cSiftCarac > & cParamDigeo::SiftCarac()
{
   return SectionCaracImages().SiftCarac();
}

const cTplValGesInit< cSiftCarac > & cParamDigeo::SiftCarac()const 
{
   return SectionCaracImages().SiftCarac();
}


cSectionCaracImages & cParamDigeo::SectionCaracImages()
{
   return mSectionCaracImages;
}

const cSectionCaracImages & cParamDigeo::SectionCaracImages()const 
{
   return mSectionCaracImages;
}


int & cParamDigeo::NbRect()
{
   return SectionTest().Val().GenereRandomRect().Val().NbRect();
}

const int & cParamDigeo::NbRect()const 
{
   return SectionTest().Val().GenereRandomRect().Val().NbRect();
}


int & cParamDigeo::SzRect()
{
   return SectionTest().Val().GenereRandomRect().Val().SzRect();
}

const int & cParamDigeo::SzRect()const 
{
   return SectionTest().Val().GenereRandomRect().Val().SzRect();
}


cTplValGesInit< cGenereRandomRect > & cParamDigeo::GenereRandomRect()
{
   return SectionTest().Val().GenereRandomRect();
}

const cTplValGesInit< cGenereRandomRect > & cParamDigeo::GenereRandomRect()const 
{
   return SectionTest().Val().GenereRandomRect();
}


int & cParamDigeo::PerX()
{
   return SectionTest().Val().GenereCarroyage().Val().PerX();
}

const int & cParamDigeo::PerX()const 
{
   return SectionTest().Val().GenereCarroyage().Val().PerX();
}


int & cParamDigeo::PerY()
{
   return SectionTest().Val().GenereCarroyage().Val().PerY();
}

const int & cParamDigeo::PerY()const 
{
   return SectionTest().Val().GenereCarroyage().Val().PerY();
}


cTplValGesInit< cGenereCarroyage > & cParamDigeo::GenereCarroyage()
{
   return SectionTest().Val().GenereCarroyage();
}

const cTplValGesInit< cGenereCarroyage > & cParamDigeo::GenereCarroyage()const 
{
   return SectionTest().Val().GenereCarroyage();
}


int & cParamDigeo::SzFilter()
{
   return SectionTest().Val().GenereAllRandom().Val().SzFilter();
}

const int & cParamDigeo::SzFilter()const 
{
   return SectionTest().Val().GenereAllRandom().Val().SzFilter();
}


cTplValGesInit< cGenereAllRandom > & cParamDigeo::GenereAllRandom()
{
   return SectionTest().Val().GenereAllRandom();
}

const cTplValGesInit< cGenereAllRandom > & cParamDigeo::GenereAllRandom()const 
{
   return SectionTest().Val().GenereAllRandom();
}


cTplValGesInit< bool > & cParamDigeo::VerifExtrema()
{
   return SectionTest().Val().VerifExtrema();
}

const cTplValGesInit< bool > & cParamDigeo::VerifExtrema()const 
{
   return SectionTest().Val().VerifExtrema();
}


cTplValGesInit< cSectionTest > & cParamDigeo::SectionTest()
{
   return mSectionTest;
}

const cTplValGesInit< cSectionTest > & cParamDigeo::SectionTest()const 
{
   return mSectionTest;
}


cTplValGesInit< std::string > & cParamDigeo::DirectoryChantier()
{
   return SectionWorkSpace().DirectoryChantier();
}

const cTplValGesInit< std::string > & cParamDigeo::DirectoryChantier()const 
{
   return SectionWorkSpace().DirectoryChantier();
}


cTplValGesInit< std::string > & cParamDigeo::Dir()
{
   return SectionWorkSpace().SauvPyram().Val().Dir();
}

const cTplValGesInit< std::string > & cParamDigeo::Dir()const 
{
   return SectionWorkSpace().SauvPyram().Val().Dir();
}


cTplValGesInit< bool > & cParamDigeo::Glob()
{
   return SectionWorkSpace().SauvPyram().Val().Glob();
}

const cTplValGesInit< bool > & cParamDigeo::Glob()const 
{
   return SectionWorkSpace().SauvPyram().Val().Glob();
}


cTplValGesInit< std::string > & cParamDigeo::Key()
{
   return SectionWorkSpace().SauvPyram().Val().Key();
}

const cTplValGesInit< std::string > & cParamDigeo::Key()const 
{
   return SectionWorkSpace().SauvPyram().Val().Key();
}


cTplValGesInit< int > & cParamDigeo::StripTifFile()
{
   return SectionWorkSpace().SauvPyram().Val().StripTifFile();
}

const cTplValGesInit< int > & cParamDigeo::StripTifFile()const 
{
   return SectionWorkSpace().SauvPyram().Val().StripTifFile();
}


cTplValGesInit< bool > & cParamDigeo::Force8B()
{
   return SectionWorkSpace().SauvPyram().Val().Force8B();
}

const cTplValGesInit< bool > & cParamDigeo::Force8B()const 
{
   return SectionWorkSpace().SauvPyram().Val().Force8B();
}


cTplValGesInit< double > & cParamDigeo::Dyn()
{
   return SectionWorkSpace().SauvPyram().Val().Dyn();
}

const cTplValGesInit< double > & cParamDigeo::Dyn()const 
{
   return SectionWorkSpace().SauvPyram().Val().Dyn();
}


cTplValGesInit< cSauvPyram > & cParamDigeo::SauvPyram()
{
   return SectionWorkSpace().SauvPyram();
}

const cTplValGesInit< cSauvPyram > & cParamDigeo::SauvPyram()const 
{
   return SectionWorkSpace().SauvPyram();
}


int & cParamDigeo::SzDalle()
{
   return SectionWorkSpace().DigeoDecoupageCarac().Val().SzDalle();
}

const int & cParamDigeo::SzDalle()const 
{
   return SectionWorkSpace().DigeoDecoupageCarac().Val().SzDalle();
}


int & cParamDigeo::Bord()
{
   return SectionWorkSpace().DigeoDecoupageCarac().Val().Bord();
}

const int & cParamDigeo::Bord()const 
{
   return SectionWorkSpace().DigeoDecoupageCarac().Val().Bord();
}


cTplValGesInit< cDigeoDecoupageCarac > & cParamDigeo::DigeoDecoupageCarac()
{
   return SectionWorkSpace().DigeoDecoupageCarac();
}

const cTplValGesInit< cDigeoDecoupageCarac > & cParamDigeo::DigeoDecoupageCarac()const 
{
   return SectionWorkSpace().DigeoDecoupageCarac();
}


cTplValGesInit< bool > & cParamDigeo::ExigeCodeCompile()
{
   return SectionWorkSpace().ExigeCodeCompile();
}

const cTplValGesInit< bool > & cParamDigeo::ExigeCodeCompile()const 
{
   return SectionWorkSpace().ExigeCodeCompile();
}


cTplValGesInit< cGenereCodeConvol > & cParamDigeo::GenereCodeConvol()
{
   return SectionWorkSpace().GenereCodeConvol();
}

const cTplValGesInit< cGenereCodeConvol > & cParamDigeo::GenereCodeConvol()const 
{
   return SectionWorkSpace().GenereCodeConvol();
}


cTplValGesInit< int > & cParamDigeo::ShowTimes()
{
   return SectionWorkSpace().ShowTimes();
}

const cTplValGesInit< int > & cParamDigeo::ShowTimes()const 
{
   return SectionWorkSpace().ShowTimes();
}


std::list< cFenVisu > & cParamDigeo::FenVisu()
{
   return SectionWorkSpace().FenVisu();
}

const std::list< cFenVisu > & cParamDigeo::FenVisu()const 
{
   return SectionWorkSpace().FenVisu();
}


cTplValGesInit< bool > & cParamDigeo::ShowConvolSpec()
{
   return SectionWorkSpace().ShowConvolSpec();
}

const cTplValGesInit< bool > & cParamDigeo::ShowConvolSpec()const 
{
   return SectionWorkSpace().ShowConvolSpec();
}


cTplValGesInit< bool > & cParamDigeo::Verbose()
{
   return SectionWorkSpace().Verbose();
}

const cTplValGesInit< bool > & cParamDigeo::Verbose()const 
{
   return SectionWorkSpace().Verbose();
}


cSectionWorkSpace & cParamDigeo::SectionWorkSpace()
{
   return mSectionWorkSpace;
}

const cSectionWorkSpace & cParamDigeo::SectionWorkSpace()const 
{
   return mSectionWorkSpace;
}

void  BinaryUnDumpFromFile(cParamDigeo & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.DicoLoc().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.DicoLoc().ValForcedForUnUmp(),aFp);
        }
        else  anObj.DicoLoc().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.DigeoSectionImages(),aFp);
    BinaryUnDumpFromFile(anObj.SectionCaracImages(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SectionTest().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SectionTest().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SectionTest().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.SectionWorkSpace(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cParamDigeo & anObj)
{
    BinaryDumpInFile(aFp,anObj.DicoLoc().IsInit());
    if (anObj.DicoLoc().IsInit()) BinaryDumpInFile(aFp,anObj.DicoLoc().Val());
    BinaryDumpInFile(aFp,anObj.DigeoSectionImages());
    BinaryDumpInFile(aFp,anObj.SectionCaracImages());
    BinaryDumpInFile(aFp,anObj.SectionTest().IsInit());
    if (anObj.SectionTest().IsInit()) BinaryDumpInFile(aFp,anObj.SectionTest().Val());
    BinaryDumpInFile(aFp,anObj.SectionWorkSpace());
}

cElXMLTree * ToXMLTree(const cParamDigeo & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ParamDigeo",eXMLBranche);
   if (anObj.DicoLoc().IsInit())
      aRes->AddFils(ToXMLTree(anObj.DicoLoc().Val())->ReTagThis("DicoLoc"));
   aRes->AddFils(ToXMLTree(anObj.DigeoSectionImages())->ReTagThis("DigeoSectionImages"));
   aRes->AddFils(ToXMLTree(anObj.SectionCaracImages())->ReTagThis("SectionCaracImages"));
   if (anObj.SectionTest().IsInit())
      aRes->AddFils(ToXMLTree(anObj.SectionTest().Val())->ReTagThis("SectionTest"));
   aRes->AddFils(ToXMLTree(anObj.SectionWorkSpace())->ReTagThis("SectionWorkSpace"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cParamDigeo & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.DicoLoc(),aTree->Get("DicoLoc",1)); //tototo 

   xml_init(anObj.DigeoSectionImages(),aTree->Get("DigeoSectionImages",1)); //tototo 

   xml_init(anObj.SectionCaracImages(),aTree->Get("SectionCaracImages",1)); //tototo 

   xml_init(anObj.SectionTest(),aTree->Get("SectionTest",1)); //tototo 

   xml_init(anObj.SectionWorkSpace(),aTree->Get("SectionWorkSpace",1)); //tototo 
}

std::string  Mangling( cParamDigeo *) {return "92C1DCC6B7DD5E87FF3F";};

// };
