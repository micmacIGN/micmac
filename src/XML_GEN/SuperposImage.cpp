// #include "general/all.h"
// #include "private/all.h"
// #include "XML_GEN/SuperposImage.h"
#include "StdAfx.h"
//
eTypeSurfaceAnalytique  Str2eTypeSurfaceAnalytique(const std::string & aName)
{
   if (aName=="eTSA_CylindreRevolution")
      return eTSA_CylindreRevolution;
  else
  {
      cout << aName << " is not a correct value for enum eTypeSurfaceAnalytique\n" ;
      ELISE_ASSERT(false,"XML enum value error");
  }
  return (eTypeSurfaceAnalytique) 0;
}
void xml_init(eTypeSurfaceAnalytique & aVal,cElXMLTree * aTree)
{
   aVal= Str2eTypeSurfaceAnalytique(aTree->Contenu());
}
std::string  eToString(const eTypeSurfaceAnalytique & anObj)
{
   if (anObj==eTSA_CylindreRevolution)
      return  "eTSA_CylindreRevolution";
 std::cout << "Enum = eTypeSurfaceAnalytique\n";
   ELISE_ASSERT(false,"Bad Value in eToString for enum value ");
   return "";
}

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eTypeSurfaceAnalytique & anObj)
{
      return  cElXMLTree::ValueNode(aNameTag,eToString(anObj));
}

eModeBoxFusion  Str2eModeBoxFusion(const std::string & aName)
{
   if (aName=="eMBF_Union")
      return eMBF_Union;
   else if (aName=="eMBF_Inter")
      return eMBF_Inter;
   else if (aName=="eMBF_First")
      return eMBF_First;
  else
  {
      cout << aName << " is not a correct value for enum eModeBoxFusion\n" ;
      ELISE_ASSERT(false,"XML enum value error");
  }
  return (eModeBoxFusion) 0;
}
void xml_init(eModeBoxFusion & aVal,cElXMLTree * aTree)
{
   aVal= Str2eModeBoxFusion(aTree->Contenu());
}
std::string  eToString(const eModeBoxFusion & anObj)
{
   if (anObj==eMBF_Union)
      return  "eMBF_Union";
   if (anObj==eMBF_Inter)
      return  "eMBF_Inter";
   if (anObj==eMBF_First)
      return  "eMBF_First";
 std::cout << "Enum = eModeBoxFusion\n";
   ELISE_ASSERT(false,"Bad Value in eToString for enum value ");
   return "";
}

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eModeBoxFusion & anObj)
{
      return  cElXMLTree::ValueNode(aNameTag,eToString(anObj));
}


int & cIntervLutConvertion::NivIn()
{
   return mNivIn;
}

const int & cIntervLutConvertion::NivIn()const 
{
   return mNivIn;
}


int & cIntervLutConvertion::NivOut()
{
   return mNivOut;
}

const int & cIntervLutConvertion::NivOut()const 
{
   return mNivOut;
}

cElXMLTree * ToXMLTree(const cIntervLutConvertion & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"IntervLutConvertion",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("NivIn"),anObj.NivIn())->ReTagThis("NivIn"));
   aRes->AddFils(::ToXMLTree(std::string("NivOut"),anObj.NivOut())->ReTagThis("NivOut"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cIntervLutConvertion & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.NivIn(),aTree->Get("NivIn",1)); //tototo 

   xml_init(anObj.NivOut(),aTree->Get("NivOut",1)); //tototo 
}


std::vector< cIntervLutConvertion > & cLutConvertion::IntervLutConvertion()
{
   return mIntervLutConvertion;
}

const std::vector< cIntervLutConvertion > & cLutConvertion::IntervLutConvertion()const 
{
   return mIntervLutConvertion;
}

cElXMLTree * ToXMLTree(const cLutConvertion & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"LutConvertion",eXMLBranche);
  for
  (       std::vector< cIntervLutConvertion >::const_iterator it=anObj.IntervLutConvertion().begin();
      it !=anObj.IntervLutConvertion().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("IntervLutConvertion"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cLutConvertion & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.IntervLutConvertion(),aTree->GetAll("IntervLutConvertion",false,1));
}


cTplValGesInit< std::string > & cWindowSelection::AllPts()
{
   return mAllPts;
}

const cTplValGesInit< std::string > & cWindowSelection::AllPts()const 
{
   return mAllPts;
}


cTplValGesInit< std::string > & cWindowSelection::PtsCenter()
{
   return mPtsCenter;
}

const cTplValGesInit< std::string > & cWindowSelection::PtsCenter()const 
{
   return mPtsCenter;
}


cTplValGesInit< double > & cWindowSelection::Percent()
{
   return mPercent;
}

const cTplValGesInit< double > & cWindowSelection::Percent()const 
{
   return mPercent;
}

cElXMLTree * ToXMLTree(const cWindowSelection & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"WindowSelection",eXMLBranche);
   if (anObj.AllPts().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("AllPts"),anObj.AllPts().Val())->ReTagThis("AllPts"));
   if (anObj.PtsCenter().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PtsCenter"),anObj.PtsCenter().Val())->ReTagThis("PtsCenter"));
   if (anObj.Percent().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Percent"),anObj.Percent().Val())->ReTagThis("Percent"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cWindowSelection & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.AllPts(),aTree->Get("AllPts",1)); //tototo 

   xml_init(anObj.PtsCenter(),aTree->Get("PtsCenter",1)); //tototo 

   xml_init(anObj.Percent(),aTree->Get("Percent",1)); //tototo 
}


std::string & cMasqTerrain::Image()
{
   return mImage;
}

const std::string & cMasqTerrain::Image()const 
{
   return mImage;
}


std::string & cMasqTerrain::XML()
{
   return mXML;
}

const std::string & cMasqTerrain::XML()const 
{
   return mXML;
}


cWindowSelection & cMasqTerrain::SelectPts()
{
   return mSelectPts;
}

const cWindowSelection & cMasqTerrain::SelectPts()const 
{
   return mSelectPts;
}

cElXMLTree * ToXMLTree(const cMasqTerrain & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"MasqTerrain",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Image"),anObj.Image())->ReTagThis("Image"));
   aRes->AddFils(::ToXMLTree(std::string("XML"),anObj.XML())->ReTagThis("XML"));
   aRes->AddFils(ToXMLTree(anObj.SelectPts())->ReTagThis("SelectPts"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cMasqTerrain & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Image(),aTree->Get("Image",1)); //tototo 

   xml_init(anObj.XML(),aTree->Get("XML",1)); //tototo 

   xml_init(anObj.SelectPts(),aTree->Get("SelectPts",1)); //tototo 
}


Pt2di & cBoxPixMort::HautG()
{
   return mHautG;
}

const Pt2di & cBoxPixMort::HautG()const 
{
   return mHautG;
}


Pt2di & cBoxPixMort::BasD()
{
   return mBasD;
}

const Pt2di & cBoxPixMort::BasD()const 
{
   return mBasD;
}

cElXMLTree * ToXMLTree(const cBoxPixMort & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"BoxPixMort",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("HautG"),anObj.HautG())->ReTagThis("HautG"));
   aRes->AddFils(::ToXMLTree(std::string("BasD"),anObj.BasD())->ReTagThis("BasD"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cBoxPixMort & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.HautG(),aTree->Get("HautG",1)); //tototo 

   xml_init(anObj.BasD(),aTree->Get("BasD",1)); //tototo 
}


std::string & cFlattField::NameFile()
{
   return mNameFile;
}

const std::string & cFlattField::NameFile()const 
{
   return mNameFile;
}


std::vector< double > & cFlattField::RefValue()
{
   return mRefValue;
}

const std::vector< double > & cFlattField::RefValue()const 
{
   return mRefValue;
}

cElXMLTree * ToXMLTree(const cFlattField & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"FlattField",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("NameFile"),anObj.NameFile())->ReTagThis("NameFile"));
  for
  (       std::vector< double >::const_iterator it=anObj.RefValue().begin();
      it !=anObj.RefValue().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("RefValue"),(*it))->ReTagThis("RefValue"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cFlattField & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.NameFile(),aTree->Get("NameFile",1)); //tototo 

   xml_init(anObj.RefValue(),aTree->GetAll("RefValue",false,1));
}


cTplValGesInit< double > & cChannelCmpCol::Dyn()
{
   return mDyn;
}

const cTplValGesInit< double > & cChannelCmpCol::Dyn()const 
{
   return mDyn;
}


cTplValGesInit< double > & cChannelCmpCol::Offset()
{
   return mOffset;
}

const cTplValGesInit< double > & cChannelCmpCol::Offset()const 
{
   return mOffset;
}


int & cChannelCmpCol::In()
{
   return mIn;
}

const int & cChannelCmpCol::In()const 
{
   return mIn;
}


int & cChannelCmpCol::Out()
{
   return mOut;
}

const int & cChannelCmpCol::Out()const 
{
   return mOut;
}


cTplValGesInit< double > & cChannelCmpCol::Pds()
{
   return mPds;
}

const cTplValGesInit< double > & cChannelCmpCol::Pds()const 
{
   return mPds;
}


cTplValGesInit< double > & cChannelCmpCol::ParamBiCub()
{
   return mParamBiCub;
}

const cTplValGesInit< double > & cChannelCmpCol::ParamBiCub()const 
{
   return mParamBiCub;
}

cElXMLTree * ToXMLTree(const cChannelCmpCol & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ChannelCmpCol",eXMLBranche);
   if (anObj.Dyn().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Dyn"),anObj.Dyn().Val())->ReTagThis("Dyn"));
   if (anObj.Offset().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Offset"),anObj.Offset().Val())->ReTagThis("Offset"));
   aRes->AddFils(::ToXMLTree(std::string("In"),anObj.In())->ReTagThis("In"));
   aRes->AddFils(::ToXMLTree(std::string("Out"),anObj.Out())->ReTagThis("Out"));
   if (anObj.Pds().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Pds"),anObj.Pds().Val())->ReTagThis("Pds"));
   if (anObj.ParamBiCub().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ParamBiCub"),anObj.ParamBiCub().Val())->ReTagThis("ParamBiCub"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cChannelCmpCol & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Dyn(),aTree->Get("Dyn",1),double(1.0)); //tototo 

   xml_init(anObj.Offset(),aTree->Get("Offset",1),double(0.0)); //tototo 

   xml_init(anObj.In(),aTree->Get("In",1)); //tototo 

   xml_init(anObj.Out(),aTree->Get("Out",1)); //tototo 

   xml_init(anObj.Pds(),aTree->Get("Pds",1),double(1.0)); //tototo 

   xml_init(anObj.ParamBiCub(),aTree->Get("ParamBiCub",1)); //tototo 
}


std::string & cImageCmpCol::NameOrKey()
{
   return mNameOrKey;
}

const std::string & cImageCmpCol::NameOrKey()const 
{
   return mNameOrKey;
}


cTplValGesInit< eTypeNumerique > & cImageCmpCol::TypeTmpIn()
{
   return mTypeTmpIn;
}

const cTplValGesInit< eTypeNumerique > & cImageCmpCol::TypeTmpIn()const 
{
   return mTypeTmpIn;
}


cTplValGesInit< std::string > & cImageCmpCol::KeyCalcNameImOfGeom()
{
   return mKeyCalcNameImOfGeom;
}

const cTplValGesInit< std::string > & cImageCmpCol::KeyCalcNameImOfGeom()const 
{
   return mKeyCalcNameImOfGeom;
}


Pt2di & cImageCmpCol::HautG()
{
   return BoxPixMort().Val().HautG();
}

const Pt2di & cImageCmpCol::HautG()const 
{
   return BoxPixMort().Val().HautG();
}


Pt2di & cImageCmpCol::BasD()
{
   return BoxPixMort().Val().BasD();
}

const Pt2di & cImageCmpCol::BasD()const 
{
   return BoxPixMort().Val().BasD();
}


cTplValGesInit< cBoxPixMort > & cImageCmpCol::BoxPixMort()
{
   return mBoxPixMort;
}

const cTplValGesInit< cBoxPixMort > & cImageCmpCol::BoxPixMort()const 
{
   return mBoxPixMort;
}


std::string & cImageCmpCol::NameFile()
{
   return FlattField().Val().NameFile();
}

const std::string & cImageCmpCol::NameFile()const 
{
   return FlattField().Val().NameFile();
}


std::vector< double > & cImageCmpCol::RefValue()
{
   return FlattField().Val().RefValue();
}

const std::vector< double > & cImageCmpCol::RefValue()const 
{
   return FlattField().Val().RefValue();
}


cTplValGesInit< cFlattField > & cImageCmpCol::FlattField()
{
   return mFlattField;
}

const cTplValGesInit< cFlattField > & cImageCmpCol::FlattField()const 
{
   return mFlattField;
}


std::list< cChannelCmpCol > & cImageCmpCol::ChannelCmpCol()
{
   return mChannelCmpCol;
}

const std::list< cChannelCmpCol > & cImageCmpCol::ChannelCmpCol()const 
{
   return mChannelCmpCol;
}


cTplValGesInit< int > & cImageCmpCol::NbFilter()
{
   return mNbFilter;
}

const cTplValGesInit< int > & cImageCmpCol::NbFilter()const 
{
   return mNbFilter;
}


cTplValGesInit< int > & cImageCmpCol::SzFilter()
{
   return mSzFilter;
}

const cTplValGesInit< int > & cImageCmpCol::SzFilter()const 
{
   return mSzFilter;
}

cElXMLTree * ToXMLTree(const cImageCmpCol & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ImageCmpCol",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("NameOrKey"),anObj.NameOrKey())->ReTagThis("NameOrKey"));
   if (anObj.TypeTmpIn().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("TypeTmpIn"),anObj.TypeTmpIn().Val())->ReTagThis("TypeTmpIn"));
   if (anObj.KeyCalcNameImOfGeom().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("KeyCalcNameImOfGeom"),anObj.KeyCalcNameImOfGeom().Val())->ReTagThis("KeyCalcNameImOfGeom"));
   if (anObj.BoxPixMort().IsInit())
      aRes->AddFils(ToXMLTree(anObj.BoxPixMort().Val())->ReTagThis("BoxPixMort"));
   if (anObj.FlattField().IsInit())
      aRes->AddFils(ToXMLTree(anObj.FlattField().Val())->ReTagThis("FlattField"));
  for
  (       std::list< cChannelCmpCol >::const_iterator it=anObj.ChannelCmpCol().begin();
      it !=anObj.ChannelCmpCol().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("ChannelCmpCol"));
   if (anObj.NbFilter().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NbFilter"),anObj.NbFilter().Val())->ReTagThis("NbFilter"));
   if (anObj.SzFilter().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SzFilter"),anObj.SzFilter().Val())->ReTagThis("SzFilter"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cImageCmpCol & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.NameOrKey(),aTree->Get("NameOrKey",1)); //tototo 

   xml_init(anObj.TypeTmpIn(),aTree->Get("TypeTmpIn",1)); //tototo 

   xml_init(anObj.KeyCalcNameImOfGeom(),aTree->Get("KeyCalcNameImOfGeom",1)); //tototo 

   xml_init(anObj.BoxPixMort(),aTree->Get("BoxPixMort",1)); //tototo 

   xml_init(anObj.FlattField(),aTree->Get("FlattField",1)); //tototo 

   xml_init(anObj.ChannelCmpCol(),aTree->GetAll("ChannelCmpCol",false,1));

   xml_init(anObj.NbFilter(),aTree->Get("NbFilter",1),int(0)); //tototo 

   xml_init(anObj.SzFilter(),aTree->Get("SzFilter",1),int(1)); //tototo 
}


std::vector< int > & cShowCalibsRel::Channel()
{
   return mChannel;
}

const std::vector< int > & cShowCalibsRel::Channel()const 
{
   return mChannel;
}


cTplValGesInit< double > & cShowCalibsRel::MaxRatio()
{
   return mMaxRatio;
}

const cTplValGesInit< double > & cShowCalibsRel::MaxRatio()const 
{
   return mMaxRatio;
}

cElXMLTree * ToXMLTree(const cShowCalibsRel & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ShowCalibsRel",eXMLBranche);
  for
  (       std::vector< int >::const_iterator it=anObj.Channel().begin();
      it !=anObj.Channel().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("Channel"),(*it))->ReTagThis("Channel"));
   if (anObj.MaxRatio().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("MaxRatio"),anObj.MaxRatio().Val())->ReTagThis("MaxRatio"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cShowCalibsRel & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Channel(),aTree->GetAll("Channel",false,1));

   xml_init(anObj.MaxRatio(),aTree->Get("MaxRatio",1),double(2.0)); //tototo 
}


cTplValGesInit< int > & cImResultCC_Gray::Channel()
{
   return mChannel;
}

const cTplValGesInit< int > & cImResultCC_Gray::Channel()const 
{
   return mChannel;
}

cElXMLTree * ToXMLTree(const cImResultCC_Gray & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ImResultCC_Gray",eXMLBranche);
   if (anObj.Channel().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Channel"),anObj.Channel().Val())->ReTagThis("Channel"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cImResultCC_Gray & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Channel(),aTree->Get("Channel",1),int(0)); //tototo 
}


cTplValGesInit< Pt3di > & cImResultCC_RVB::Channel()
{
   return mChannel;
}

const cTplValGesInit< Pt3di > & cImResultCC_RVB::Channel()const 
{
   return mChannel;
}

cElXMLTree * ToXMLTree(const cImResultCC_RVB & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ImResultCC_RVB",eXMLBranche);
   if (anObj.Channel().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Channel"),anObj.Channel().Val())->ReTagThis("Channel"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cImResultCC_RVB & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Channel(),aTree->Get("Channel",1),Pt3di(0,1,2)); //tototo 
}


cTplValGesInit< bool > & cImResultCC_Cnes::ModeMedian()
{
   return mModeMedian;
}

const cTplValGesInit< bool > & cImResultCC_Cnes::ModeMedian()const 
{
   return mModeMedian;
}


cTplValGesInit< Pt2di > & cImResultCC_Cnes::SzF()
{
   return mSzF;
}

const cTplValGesInit< Pt2di > & cImResultCC_Cnes::SzF()const 
{
   return mSzF;
}


cTplValGesInit< std::string > & cImResultCC_Cnes::ValueF()
{
   return mValueF;
}

const cTplValGesInit< std::string > & cImResultCC_Cnes::ValueF()const 
{
   return mValueF;
}


cTplValGesInit< int > & cImResultCC_Cnes::ChannelHF()
{
   return mChannelHF;
}

const cTplValGesInit< int > & cImResultCC_Cnes::ChannelHF()const 
{
   return mChannelHF;
}


cTplValGesInit< std::vector<int> > & cImResultCC_Cnes::ChannelBF()
{
   return mChannelBF;
}

const cTplValGesInit< std::vector<int> > & cImResultCC_Cnes::ChannelBF()const 
{
   return mChannelBF;
}


cTplValGesInit< int > & cImResultCC_Cnes::NbIterFCSte()
{
   return mNbIterFCSte;
}

const cTplValGesInit< int > & cImResultCC_Cnes::NbIterFCSte()const 
{
   return mNbIterFCSte;
}


cTplValGesInit< int > & cImResultCC_Cnes::SzIterFCSte()
{
   return mSzIterFCSte;
}

const cTplValGesInit< int > & cImResultCC_Cnes::SzIterFCSte()const 
{
   return mSzIterFCSte;
}

cElXMLTree * ToXMLTree(const cImResultCC_Cnes & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ImResultCC_Cnes",eXMLBranche);
   if (anObj.ModeMedian().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ModeMedian"),anObj.ModeMedian().Val())->ReTagThis("ModeMedian"));
   if (anObj.SzF().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SzF"),anObj.SzF().Val())->ReTagThis("SzF"));
   if (anObj.ValueF().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ValueF"),anObj.ValueF().Val())->ReTagThis("ValueF"));
   if (anObj.ChannelHF().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ChannelHF"),anObj.ChannelHF().Val())->ReTagThis("ChannelHF"));
   if (anObj.ChannelBF().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ChannelBF"),anObj.ChannelBF().Val())->ReTagThis("ChannelBF"));
   if (anObj.NbIterFCSte().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NbIterFCSte"),anObj.NbIterFCSte().Val())->ReTagThis("NbIterFCSte"));
   if (anObj.SzIterFCSte().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SzIterFCSte"),anObj.SzIterFCSte().Val())->ReTagThis("SzIterFCSte"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cImResultCC_Cnes & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.ModeMedian(),aTree->Get("ModeMedian",1),bool(false)); //tototo 

   xml_init(anObj.SzF(),aTree->Get("SzF",1),Pt2di(3,3)); //tototo 

   xml_init(anObj.ValueF(),aTree->Get("ValueF",1),std::string("1 2 1 2 4 2 1 2 1")); //tototo 

   xml_init(anObj.ChannelHF(),aTree->Get("ChannelHF",1),int(3)); //tototo 

   xml_init(anObj.ChannelBF(),aTree->Get("ChannelBF",1)); //tototo 

   xml_init(anObj.NbIterFCSte(),aTree->Get("NbIterFCSte",1),int(1)); //tototo 

   xml_init(anObj.SzIterFCSte(),aTree->Get("SzIterFCSte",1)); //tototo 
}


cTplValGesInit< std::vector<int> > & cImResultCC_PXs::Channel()
{
   return mChannel;
}

const cTplValGesInit< std::vector<int> > & cImResultCC_PXs::Channel()const 
{
   return mChannel;
}


cTplValGesInit< Pt3dr > & cImResultCC_PXs::AxeRGB()
{
   return mAxeRGB;
}

const cTplValGesInit< Pt3dr > & cImResultCC_PXs::AxeRGB()const 
{
   return mAxeRGB;
}


cTplValGesInit< double > & cImResultCC_PXs::Cste()
{
   return mCste;
}

const cTplValGesInit< double > & cImResultCC_PXs::Cste()const 
{
   return mCste;
}


cTplValGesInit< bool > & cImResultCC_PXs::ApprentisageAxeRGB()
{
   return mApprentisageAxeRGB;
}

const cTplValGesInit< bool > & cImResultCC_PXs::ApprentisageAxeRGB()const 
{
   return mApprentisageAxeRGB;
}


std::list< std::string > & cImResultCC_PXs::UnusedAppr()
{
   return mUnusedAppr;
}

const std::list< std::string > & cImResultCC_PXs::UnusedAppr()const 
{
   return mUnusedAppr;
}

cElXMLTree * ToXMLTree(const cImResultCC_PXs & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ImResultCC_PXs",eXMLBranche);
   if (anObj.Channel().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Channel"),anObj.Channel().Val())->ReTagThis("Channel"));
   if (anObj.AxeRGB().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("AxeRGB"),anObj.AxeRGB().Val())->ReTagThis("AxeRGB"));
   if (anObj.Cste().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Cste"),anObj.Cste().Val())->ReTagThis("Cste"));
   if (anObj.ApprentisageAxeRGB().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ApprentisageAxeRGB"),anObj.ApprentisageAxeRGB().Val())->ReTagThis("ApprentisageAxeRGB"));
  for
  (       std::list< std::string >::const_iterator it=anObj.UnusedAppr().begin();
      it !=anObj.UnusedAppr().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("UnusedAppr"),(*it))->ReTagThis("UnusedAppr"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cImResultCC_PXs & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Channel(),aTree->Get("Channel",1)); //tototo 

   xml_init(anObj.AxeRGB(),aTree->Get("AxeRGB",1),Pt3dr(1.0,1.0,1.0)); //tototo 

   xml_init(anObj.Cste(),aTree->Get("Cste",1),double(0)); //tototo 

   xml_init(anObj.ApprentisageAxeRGB(),aTree->Get("ApprentisageAxeRGB",1),bool(false)); //tototo 

   xml_init(anObj.UnusedAppr(),aTree->GetAll("UnusedAppr",false,1));
}


cTplValGesInit< double > & cPondThom::PondExp()
{
   return mPondExp;
}

const cTplValGesInit< double > & cPondThom::PondExp()const 
{
   return mPondExp;
}


cTplValGesInit< int > & cPondThom::PondCste()
{
   return mPondCste;
}

const cTplValGesInit< int > & cPondThom::PondCste()const 
{
   return mPondCste;
}

cElXMLTree * ToXMLTree(const cPondThom & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"PondThom",eXMLBranche);
   if (anObj.PondExp().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PondExp"),anObj.PondExp().Val())->ReTagThis("PondExp"));
   if (anObj.PondCste().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PondCste"),anObj.PondCste().Val())->ReTagThis("PondCste"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cPondThom & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.PondExp(),aTree->Get("PondExp",1)); //tototo 

   xml_init(anObj.PondCste(),aTree->Get("PondCste",1)); //tototo 
}


double & cThomBidouille::VMin()
{
   return mVMin;
}

const double & cThomBidouille::VMin()const 
{
   return mVMin;
}


double & cThomBidouille::PourCent()
{
   return mPourCent;
}

const double & cThomBidouille::PourCent()const 
{
   return mPourCent;
}

cElXMLTree * ToXMLTree(const cThomBidouille & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ThomBidouille",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("VMin"),anObj.VMin())->ReTagThis("VMin"));
   aRes->AddFils(::ToXMLTree(std::string("PourCent"),anObj.PourCent())->ReTagThis("PourCent"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cThomBidouille & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.VMin(),aTree->Get("VMin",1)); //tototo 

   xml_init(anObj.PourCent(),aTree->Get("PourCent",1)); //tototo 
}


double & cMPDBidouille::EcartMin()
{
   return mEcartMin;
}

const double & cMPDBidouille::EcartMin()const 
{
   return mEcartMin;
}

cElXMLTree * ToXMLTree(const cMPDBidouille & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"MPDBidouille",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("EcartMin"),anObj.EcartMin())->ReTagThis("EcartMin"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cMPDBidouille & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.EcartMin(),aTree->Get("EcartMin",1)); //tototo 
}


double & cThomAgreg::VMin()
{
   return ThomBidouille().Val().VMin();
}

const double & cThomAgreg::VMin()const 
{
   return ThomBidouille().Val().VMin();
}


double & cThomAgreg::PourCent()
{
   return ThomBidouille().Val().PourCent();
}

const double & cThomAgreg::PourCent()const 
{
   return ThomBidouille().Val().PourCent();
}


cTplValGesInit< cThomBidouille > & cThomAgreg::ThomBidouille()
{
   return mThomBidouille;
}

const cTplValGesInit< cThomBidouille > & cThomAgreg::ThomBidouille()const 
{
   return mThomBidouille;
}


double & cThomAgreg::EcartMin()
{
   return MPDBidouille().Val().EcartMin();
}

const double & cThomAgreg::EcartMin()const 
{
   return MPDBidouille().Val().EcartMin();
}


cTplValGesInit< cMPDBidouille > & cThomAgreg::MPDBidouille()
{
   return mMPDBidouille;
}

const cTplValGesInit< cMPDBidouille > & cThomAgreg::MPDBidouille()const 
{
   return mMPDBidouille;
}

cElXMLTree * ToXMLTree(const cThomAgreg & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ThomAgreg",eXMLBranche);
   if (anObj.ThomBidouille().IsInit())
      aRes->AddFils(ToXMLTree(anObj.ThomBidouille().Val())->ReTagThis("ThomBidouille"));
   if (anObj.MPDBidouille().IsInit())
      aRes->AddFils(ToXMLTree(anObj.MPDBidouille().Val())->ReTagThis("MPDBidouille"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cThomAgreg & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.ThomBidouille(),aTree->Get("ThomBidouille",1)); //tototo 

   xml_init(anObj.MPDBidouille(),aTree->Get("MPDBidouille",1)); //tototo 
}


cTplValGesInit< double > & cImResultCC_Thom::PondExp()
{
   return PondThom().PondExp();
}

const cTplValGesInit< double > & cImResultCC_Thom::PondExp()const 
{
   return PondThom().PondExp();
}


cTplValGesInit< int > & cImResultCC_Thom::PondCste()
{
   return PondThom().PondCste();
}

const cTplValGesInit< int > & cImResultCC_Thom::PondCste()const 
{
   return PondThom().PondCste();
}


cPondThom & cImResultCC_Thom::PondThom()
{
   return mPondThom;
}

const cPondThom & cImResultCC_Thom::PondThom()const 
{
   return mPondThom;
}


cTplValGesInit< int > & cImResultCC_Thom::NbIterPond()
{
   return mNbIterPond;
}

const cTplValGesInit< int > & cImResultCC_Thom::NbIterPond()const 
{
   return mNbIterPond;
}


cTplValGesInit< bool > & cImResultCC_Thom::SupressCentre()
{
   return mSupressCentre;
}

const cTplValGesInit< bool > & cImResultCC_Thom::SupressCentre()const 
{
   return mSupressCentre;
}


cTplValGesInit< int > & cImResultCC_Thom::ChannelHF()
{
   return mChannelHF;
}

const cTplValGesInit< int > & cImResultCC_Thom::ChannelHF()const 
{
   return mChannelHF;
}


cTplValGesInit< std::vector<int> > & cImResultCC_Thom::ChannelBF()
{
   return mChannelBF;
}

const cTplValGesInit< std::vector<int> > & cImResultCC_Thom::ChannelBF()const 
{
   return mChannelBF;
}


double & cImResultCC_Thom::VMin()
{
   return ThomAgreg().ThomBidouille().Val().VMin();
}

const double & cImResultCC_Thom::VMin()const 
{
   return ThomAgreg().ThomBidouille().Val().VMin();
}


double & cImResultCC_Thom::PourCent()
{
   return ThomAgreg().ThomBidouille().Val().PourCent();
}

const double & cImResultCC_Thom::PourCent()const 
{
   return ThomAgreg().ThomBidouille().Val().PourCent();
}


cTplValGesInit< cThomBidouille > & cImResultCC_Thom::ThomBidouille()
{
   return ThomAgreg().ThomBidouille();
}

const cTplValGesInit< cThomBidouille > & cImResultCC_Thom::ThomBidouille()const 
{
   return ThomAgreg().ThomBidouille();
}


double & cImResultCC_Thom::EcartMin()
{
   return ThomAgreg().MPDBidouille().Val().EcartMin();
}

const double & cImResultCC_Thom::EcartMin()const 
{
   return ThomAgreg().MPDBidouille().Val().EcartMin();
}


cTplValGesInit< cMPDBidouille > & cImResultCC_Thom::MPDBidouille()
{
   return ThomAgreg().MPDBidouille();
}

const cTplValGesInit< cMPDBidouille > & cImResultCC_Thom::MPDBidouille()const 
{
   return ThomAgreg().MPDBidouille();
}


cThomAgreg & cImResultCC_Thom::ThomAgreg()
{
   return mThomAgreg;
}

const cThomAgreg & cImResultCC_Thom::ThomAgreg()const 
{
   return mThomAgreg;
}

cElXMLTree * ToXMLTree(const cImResultCC_Thom & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ImResultCC_Thom",eXMLBranche);
   aRes->AddFils(ToXMLTree(anObj.PondThom())->ReTagThis("PondThom"));
   if (anObj.NbIterPond().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NbIterPond"),anObj.NbIterPond().Val())->ReTagThis("NbIterPond"));
   if (anObj.SupressCentre().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SupressCentre"),anObj.SupressCentre().Val())->ReTagThis("SupressCentre"));
   if (anObj.ChannelHF().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ChannelHF"),anObj.ChannelHF().Val())->ReTagThis("ChannelHF"));
   if (anObj.ChannelBF().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ChannelBF"),anObj.ChannelBF().Val())->ReTagThis("ChannelBF"));
   aRes->AddFils(ToXMLTree(anObj.ThomAgreg())->ReTagThis("ThomAgreg"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cImResultCC_Thom & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.PondThom(),aTree->Get("PondThom",1)); //tototo 

   xml_init(anObj.NbIterPond(),aTree->Get("NbIterPond",1),int(1)); //tototo 

   xml_init(anObj.SupressCentre(),aTree->Get("SupressCentre",1),bool(false)); //tototo 

   xml_init(anObj.ChannelHF(),aTree->Get("ChannelHF",1),int(3)); //tototo 

   xml_init(anObj.ChannelBF(),aTree->Get("ChannelBF",1)); //tototo 

   xml_init(anObj.ThomAgreg(),aTree->Get("ThomAgreg",1)); //tototo 
}


cTplValGesInit< cImResultCC_Gray > & cImResultCC::ImResultCC_Gray()
{
   return mImResultCC_Gray;
}

const cTplValGesInit< cImResultCC_Gray > & cImResultCC::ImResultCC_Gray()const 
{
   return mImResultCC_Gray;
}


cTplValGesInit< cImResultCC_RVB > & cImResultCC::ImResultCC_RVB()
{
   return mImResultCC_RVB;
}

const cTplValGesInit< cImResultCC_RVB > & cImResultCC::ImResultCC_RVB()const 
{
   return mImResultCC_RVB;
}


cTplValGesInit< cImResultCC_Cnes > & cImResultCC::ImResultCC_Cnes()
{
   return mImResultCC_Cnes;
}

const cTplValGesInit< cImResultCC_Cnes > & cImResultCC::ImResultCC_Cnes()const 
{
   return mImResultCC_Cnes;
}


cTplValGesInit< cImResultCC_PXs > & cImResultCC::ImResultCC_PXs()
{
   return mImResultCC_PXs;
}

const cTplValGesInit< cImResultCC_PXs > & cImResultCC::ImResultCC_PXs()const 
{
   return mImResultCC_PXs;
}


cTplValGesInit< cImResultCC_Thom > & cImResultCC::ImResultCC_Thom()
{
   return mImResultCC_Thom;
}

const cTplValGesInit< cImResultCC_Thom > & cImResultCC::ImResultCC_Thom()const 
{
   return mImResultCC_Thom;
}

cElXMLTree * ToXMLTree(const cImResultCC & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ImResultCC",eXMLBranche);
   if (anObj.ImResultCC_Gray().IsInit())
      aRes->AddFils(ToXMLTree(anObj.ImResultCC_Gray().Val())->ReTagThis("ImResultCC_Gray"));
   if (anObj.ImResultCC_RVB().IsInit())
      aRes->AddFils(ToXMLTree(anObj.ImResultCC_RVB().Val())->ReTagThis("ImResultCC_RVB"));
   if (anObj.ImResultCC_Cnes().IsInit())
      aRes->AddFils(ToXMLTree(anObj.ImResultCC_Cnes().Val())->ReTagThis("ImResultCC_Cnes"));
   if (anObj.ImResultCC_PXs().IsInit())
      aRes->AddFils(ToXMLTree(anObj.ImResultCC_PXs().Val())->ReTagThis("ImResultCC_PXs"));
   if (anObj.ImResultCC_Thom().IsInit())
      aRes->AddFils(ToXMLTree(anObj.ImResultCC_Thom().Val())->ReTagThis("ImResultCC_Thom"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cImResultCC & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.ImResultCC_Gray(),aTree->Get("ImResultCC_Gray",1)); //tototo 

   xml_init(anObj.ImResultCC_RVB(),aTree->Get("ImResultCC_RVB",1)); //tototo 

   xml_init(anObj.ImResultCC_Cnes(),aTree->Get("ImResultCC_Cnes",1)); //tototo 

   xml_init(anObj.ImResultCC_PXs(),aTree->Get("ImResultCC_PXs",1)); //tototo 

   xml_init(anObj.ImResultCC_Thom(),aTree->Get("ImResultCC_Thom",1)); //tototo 
}


cTplValGesInit< double > & cResultCompCol::GamaExport()
{
   return mGamaExport;
}

const cTplValGesInit< double > & cResultCompCol::GamaExport()const 
{
   return mGamaExport;
}


cTplValGesInit< double > & cResultCompCol::RefGama()
{
   return mRefGama;
}

const cTplValGesInit< double > & cResultCompCol::RefGama()const 
{
   return mRefGama;
}


cTplValGesInit< cLutConvertion > & cResultCompCol::LutExport()
{
   return mLutExport;
}

const cTplValGesInit< cLutConvertion > & cResultCompCol::LutExport()const 
{
   return mLutExport;
}


std::string & cResultCompCol::KeyName()
{
   return mKeyName;
}

const std::string & cResultCompCol::KeyName()const 
{
   return mKeyName;
}


cTplValGesInit< eTypeNumerique > & cResultCompCol::Type()
{
   return mType;
}

const cTplValGesInit< eTypeNumerique > & cResultCompCol::Type()const 
{
   return mType;
}


cTplValGesInit< cImResultCC_Gray > & cResultCompCol::ImResultCC_Gray()
{
   return ImResultCC().ImResultCC_Gray();
}

const cTplValGesInit< cImResultCC_Gray > & cResultCompCol::ImResultCC_Gray()const 
{
   return ImResultCC().ImResultCC_Gray();
}


cTplValGesInit< cImResultCC_RVB > & cResultCompCol::ImResultCC_RVB()
{
   return ImResultCC().ImResultCC_RVB();
}

const cTplValGesInit< cImResultCC_RVB > & cResultCompCol::ImResultCC_RVB()const 
{
   return ImResultCC().ImResultCC_RVB();
}


cTplValGesInit< cImResultCC_Cnes > & cResultCompCol::ImResultCC_Cnes()
{
   return ImResultCC().ImResultCC_Cnes();
}

const cTplValGesInit< cImResultCC_Cnes > & cResultCompCol::ImResultCC_Cnes()const 
{
   return ImResultCC().ImResultCC_Cnes();
}


cTplValGesInit< cImResultCC_PXs > & cResultCompCol::ImResultCC_PXs()
{
   return ImResultCC().ImResultCC_PXs();
}

const cTplValGesInit< cImResultCC_PXs > & cResultCompCol::ImResultCC_PXs()const 
{
   return ImResultCC().ImResultCC_PXs();
}


cTplValGesInit< cImResultCC_Thom > & cResultCompCol::ImResultCC_Thom()
{
   return ImResultCC().ImResultCC_Thom();
}

const cTplValGesInit< cImResultCC_Thom > & cResultCompCol::ImResultCC_Thom()const 
{
   return ImResultCC().ImResultCC_Thom();
}


cImResultCC & cResultCompCol::ImResultCC()
{
   return mImResultCC;
}

const cImResultCC & cResultCompCol::ImResultCC()const 
{
   return mImResultCC;
}

cElXMLTree * ToXMLTree(const cResultCompCol & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ResultCompCol",eXMLBranche);
   if (anObj.GamaExport().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("GamaExport"),anObj.GamaExport().Val())->ReTagThis("GamaExport"));
   if (anObj.RefGama().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("RefGama"),anObj.RefGama().Val())->ReTagThis("RefGama"));
   if (anObj.LutExport().IsInit())
      aRes->AddFils(ToXMLTree(anObj.LutExport().Val())->ReTagThis("LutExport"));
   aRes->AddFils(::ToXMLTree(std::string("KeyName"),anObj.KeyName())->ReTagThis("KeyName"));
   if (anObj.Type().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Type"),anObj.Type().Val())->ReTagThis("Type"));
   aRes->AddFils(ToXMLTree(anObj.ImResultCC())->ReTagThis("ImResultCC"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cResultCompCol & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.GamaExport(),aTree->Get("GamaExport",1)); //tototo 

   xml_init(anObj.RefGama(),aTree->Get("RefGama",1),double(256.0)); //tototo 

   xml_init(anObj.LutExport(),aTree->Get("LutExport",1)); //tototo 

   xml_init(anObj.KeyName(),aTree->Get("KeyName",1)); //tototo 

   xml_init(anObj.Type(),aTree->Get("Type",1),eTypeNumerique(eTN_u_int1)); //tototo 

   xml_init(anObj.ImResultCC(),aTree->Get("ImResultCC",1)); //tototo 
}


cTplValGesInit< std::string > & cEspaceResultSuperpCol::EnglobImMaitre()
{
   return mEnglobImMaitre;
}

const cTplValGesInit< std::string > & cEspaceResultSuperpCol::EnglobImMaitre()const 
{
   return mEnglobImMaitre;
}


cTplValGesInit< std::string > & cEspaceResultSuperpCol::EnglobAll()
{
   return mEnglobAll;
}

const cTplValGesInit< std::string > & cEspaceResultSuperpCol::EnglobAll()const 
{
   return mEnglobAll;
}


cTplValGesInit< Box2di > & cEspaceResultSuperpCol::EnglobBoxMaitresse()
{
   return mEnglobBoxMaitresse;
}

const cTplValGesInit< Box2di > & cEspaceResultSuperpCol::EnglobBoxMaitresse()const 
{
   return mEnglobBoxMaitresse;
}

cElXMLTree * ToXMLTree(const cEspaceResultSuperpCol & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"EspaceResultSuperpCol",eXMLBranche);
   if (anObj.EnglobImMaitre().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("EnglobImMaitre"),anObj.EnglobImMaitre().Val())->ReTagThis("EnglobImMaitre"));
   if (anObj.EnglobAll().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("EnglobAll"),anObj.EnglobAll().Val())->ReTagThis("EnglobAll"));
   if (anObj.EnglobBoxMaitresse().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("EnglobBoxMaitresse"),anObj.EnglobBoxMaitresse().Val())->ReTagThis("EnglobBoxMaitresse"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cEspaceResultSuperpCol & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.EnglobImMaitre(),aTree->Get("EnglobImMaitre",1)); //tototo 

   xml_init(anObj.EnglobAll(),aTree->Get("EnglobAll",1)); //tototo 

   xml_init(anObj.EnglobBoxMaitresse(),aTree->Get("EnglobBoxMaitresse",1)); //tototo 
}


std::string & cImages2Verif::X()
{
   return mX;
}

const std::string & cImages2Verif::X()const 
{
   return mX;
}


std::string & cImages2Verif::Y()
{
   return mY;
}

const std::string & cImages2Verif::Y()const 
{
   return mY;
}


double & cImages2Verif::ExagXY()
{
   return mExagXY;
}

const double & cImages2Verif::ExagXY()const 
{
   return mExagXY;
}

cElXMLTree * ToXMLTree(const cImages2Verif & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"Images2Verif",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("X"),anObj.X())->ReTagThis("X"));
   aRes->AddFils(::ToXMLTree(std::string("Y"),anObj.Y())->ReTagThis("Y"));
   aRes->AddFils(::ToXMLTree(std::string("ExagXY"),anObj.ExagXY())->ReTagThis("ExagXY"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cImages2Verif & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.X(),aTree->Get("X",1)); //tototo 

   xml_init(anObj.Y(),aTree->Get("Y",1)); //tototo 

   xml_init(anObj.ExagXY(),aTree->Get("ExagXY",1)); //tototo 
}


double & cVisuEcart::SzW()
{
   return mSzW;
}

const double & cVisuEcart::SzW()const 
{
   return mSzW;
}


double & cVisuEcart::Exag()
{
   return mExag;
}

const double & cVisuEcart::Exag()const 
{
   return mExag;
}


cTplValGesInit< std::string > & cVisuEcart::NameFile()
{
   return mNameFile;
}

const cTplValGesInit< std::string > & cVisuEcart::NameFile()const 
{
   return mNameFile;
}


std::string & cVisuEcart::X()
{
   return Images2Verif().Val().X();
}

const std::string & cVisuEcart::X()const 
{
   return Images2Verif().Val().X();
}


std::string & cVisuEcart::Y()
{
   return Images2Verif().Val().Y();
}

const std::string & cVisuEcart::Y()const 
{
   return Images2Verif().Val().Y();
}


double & cVisuEcart::ExagXY()
{
   return Images2Verif().Val().ExagXY();
}

const double & cVisuEcart::ExagXY()const 
{
   return Images2Verif().Val().ExagXY();
}


cTplValGesInit< cImages2Verif > & cVisuEcart::Images2Verif()
{
   return mImages2Verif;
}

const cTplValGesInit< cImages2Verif > & cVisuEcart::Images2Verif()const 
{
   return mImages2Verif;
}

cElXMLTree * ToXMLTree(const cVisuEcart & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"VisuEcart",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("SzW"),anObj.SzW())->ReTagThis("SzW"));
   aRes->AddFils(::ToXMLTree(std::string("Exag"),anObj.Exag())->ReTagThis("Exag"));
   if (anObj.NameFile().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NameFile"),anObj.NameFile().Val())->ReTagThis("NameFile"));
   if (anObj.Images2Verif().IsInit())
      aRes->AddFils(ToXMLTree(anObj.Images2Verif().Val())->ReTagThis("Images2Verif"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cVisuEcart & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.SzW(),aTree->Get("SzW",1)); //tototo 

   xml_init(anObj.Exag(),aTree->Get("Exag",1)); //tototo 

   xml_init(anObj.NameFile(),aTree->Get("NameFile",1)); //tototo 

   xml_init(anObj.Images2Verif(),aTree->Get("Images2Verif",1)); //tototo 
}


std::string & cVerifHoms::NameOrKeyHomologues()
{
   return mNameOrKeyHomologues;
}

const std::string & cVerifHoms::NameOrKeyHomologues()const 
{
   return mNameOrKeyHomologues;
}


double & cVerifHoms::SzW()
{
   return VisuEcart().Val().SzW();
}

const double & cVerifHoms::SzW()const 
{
   return VisuEcart().Val().SzW();
}


double & cVerifHoms::Exag()
{
   return VisuEcart().Val().Exag();
}

const double & cVerifHoms::Exag()const 
{
   return VisuEcart().Val().Exag();
}


cTplValGesInit< std::string > & cVerifHoms::NameFile()
{
   return VisuEcart().Val().NameFile();
}

const cTplValGesInit< std::string > & cVerifHoms::NameFile()const 
{
   return VisuEcart().Val().NameFile();
}


std::string & cVerifHoms::X()
{
   return VisuEcart().Val().Images2Verif().Val().X();
}

const std::string & cVerifHoms::X()const 
{
   return VisuEcart().Val().Images2Verif().Val().X();
}


std::string & cVerifHoms::Y()
{
   return VisuEcart().Val().Images2Verif().Val().Y();
}

const std::string & cVerifHoms::Y()const 
{
   return VisuEcart().Val().Images2Verif().Val().Y();
}


double & cVerifHoms::ExagXY()
{
   return VisuEcart().Val().Images2Verif().Val().ExagXY();
}

const double & cVerifHoms::ExagXY()const 
{
   return VisuEcart().Val().Images2Verif().Val().ExagXY();
}


cTplValGesInit< cImages2Verif > & cVerifHoms::Images2Verif()
{
   return VisuEcart().Val().Images2Verif();
}

const cTplValGesInit< cImages2Verif > & cVerifHoms::Images2Verif()const 
{
   return VisuEcart().Val().Images2Verif();
}


cTplValGesInit< cVisuEcart > & cVerifHoms::VisuEcart()
{
   return mVisuEcart;
}

const cTplValGesInit< cVisuEcart > & cVerifHoms::VisuEcart()const 
{
   return mVisuEcart;
}

cElXMLTree * ToXMLTree(const cVerifHoms & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"VerifHoms",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("NameOrKeyHomologues"),anObj.NameOrKeyHomologues())->ReTagThis("NameOrKeyHomologues"));
   if (anObj.VisuEcart().IsInit())
      aRes->AddFils(ToXMLTree(anObj.VisuEcart().Val())->ReTagThis("VisuEcart"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cVerifHoms & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.NameOrKeyHomologues(),aTree->Get("NameOrKeyHomologues",1)); //tototo 

   xml_init(anObj.VisuEcart(),aTree->Get("VisuEcart",1)); //tototo 
}


cImageCmpCol & cImSec::Im()
{
   return mIm;
}

const cImageCmpCol & cImSec::Im()const 
{
   return mIm;
}


std::string & cImSec::KeyCalcNameCorresp()
{
   return mKeyCalcNameCorresp;
}

const std::string & cImSec::KeyCalcNameCorresp()const 
{
   return mKeyCalcNameCorresp;
}


cTplValGesInit< Pt2dr > & cImSec::OffsetPt()
{
   return mOffsetPt;
}

const cTplValGesInit< Pt2dr > & cImSec::OffsetPt()const 
{
   return mOffsetPt;
}


cTplValGesInit< std::string > & cImSec::DirCalcCorrep()
{
   return mDirCalcCorrep;
}

const cTplValGesInit< std::string > & cImSec::DirCalcCorrep()const 
{
   return mDirCalcCorrep;
}


std::string & cImSec::NameOrKeyHomologues()
{
   return VerifHoms().Val().NameOrKeyHomologues();
}

const std::string & cImSec::NameOrKeyHomologues()const 
{
   return VerifHoms().Val().NameOrKeyHomologues();
}


double & cImSec::SzW()
{
   return VerifHoms().Val().VisuEcart().Val().SzW();
}

const double & cImSec::SzW()const 
{
   return VerifHoms().Val().VisuEcart().Val().SzW();
}


double & cImSec::Exag()
{
   return VerifHoms().Val().VisuEcart().Val().Exag();
}

const double & cImSec::Exag()const 
{
   return VerifHoms().Val().VisuEcart().Val().Exag();
}


cTplValGesInit< std::string > & cImSec::NameFile()
{
   return VerifHoms().Val().VisuEcart().Val().NameFile();
}

const cTplValGesInit< std::string > & cImSec::NameFile()const 
{
   return VerifHoms().Val().VisuEcart().Val().NameFile();
}


std::string & cImSec::X()
{
   return VerifHoms().Val().VisuEcart().Val().Images2Verif().Val().X();
}

const std::string & cImSec::X()const 
{
   return VerifHoms().Val().VisuEcart().Val().Images2Verif().Val().X();
}


std::string & cImSec::Y()
{
   return VerifHoms().Val().VisuEcart().Val().Images2Verif().Val().Y();
}

const std::string & cImSec::Y()const 
{
   return VerifHoms().Val().VisuEcart().Val().Images2Verif().Val().Y();
}


double & cImSec::ExagXY()
{
   return VerifHoms().Val().VisuEcart().Val().Images2Verif().Val().ExagXY();
}

const double & cImSec::ExagXY()const 
{
   return VerifHoms().Val().VisuEcart().Val().Images2Verif().Val().ExagXY();
}


cTplValGesInit< cImages2Verif > & cImSec::Images2Verif()
{
   return VerifHoms().Val().VisuEcart().Val().Images2Verif();
}

const cTplValGesInit< cImages2Verif > & cImSec::Images2Verif()const 
{
   return VerifHoms().Val().VisuEcart().Val().Images2Verif();
}


cTplValGesInit< cVisuEcart > & cImSec::VisuEcart()
{
   return VerifHoms().Val().VisuEcart();
}

const cTplValGesInit< cVisuEcart > & cImSec::VisuEcart()const 
{
   return VerifHoms().Val().VisuEcart();
}


cTplValGesInit< cVerifHoms > & cImSec::VerifHoms()
{
   return mVerifHoms;
}

const cTplValGesInit< cVerifHoms > & cImSec::VerifHoms()const 
{
   return mVerifHoms;
}


cTplValGesInit< int > & cImSec::NbTestRansacEstimH()
{
   return mNbTestRansacEstimH;
}

const cTplValGesInit< int > & cImSec::NbTestRansacEstimH()const 
{
   return mNbTestRansacEstimH;
}


cTplValGesInit< int > & cImSec::NbPtsRansacEstimH()
{
   return mNbPtsRansacEstimH;
}

const cTplValGesInit< int > & cImSec::NbPtsRansacEstimH()const 
{
   return mNbPtsRansacEstimH;
}


cTplValGesInit< bool > & cImSec::L2EstimH()
{
   return mL2EstimH;
}

const cTplValGesInit< bool > & cImSec::L2EstimH()const 
{
   return mL2EstimH;
}


cTplValGesInit< bool > & cImSec::L1EstimH()
{
   return mL1EstimH;
}

const cTplValGesInit< bool > & cImSec::L1EstimH()const 
{
   return mL1EstimH;
}


std::list< Pt2dr > & cImSec::PonderaL2Iter()
{
   return mPonderaL2Iter;
}

const std::list< Pt2dr > & cImSec::PonderaL2Iter()const 
{
   return mPonderaL2Iter;
}

cElXMLTree * ToXMLTree(const cImSec & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ImSec",eXMLBranche);
   aRes->AddFils(ToXMLTree(anObj.Im())->ReTagThis("Im"));
   aRes->AddFils(::ToXMLTree(std::string("KeyCalcNameCorresp"),anObj.KeyCalcNameCorresp())->ReTagThis("KeyCalcNameCorresp"));
   if (anObj.OffsetPt().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("OffsetPt"),anObj.OffsetPt().Val())->ReTagThis("OffsetPt"));
   if (anObj.DirCalcCorrep().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DirCalcCorrep"),anObj.DirCalcCorrep().Val())->ReTagThis("DirCalcCorrep"));
   if (anObj.VerifHoms().IsInit())
      aRes->AddFils(ToXMLTree(anObj.VerifHoms().Val())->ReTagThis("VerifHoms"));
   if (anObj.NbTestRansacEstimH().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NbTestRansacEstimH"),anObj.NbTestRansacEstimH().Val())->ReTagThis("NbTestRansacEstimH"));
   if (anObj.NbPtsRansacEstimH().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NbPtsRansacEstimH"),anObj.NbPtsRansacEstimH().Val())->ReTagThis("NbPtsRansacEstimH"));
   if (anObj.L2EstimH().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("L2EstimH"),anObj.L2EstimH().Val())->ReTagThis("L2EstimH"));
   if (anObj.L1EstimH().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("L1EstimH"),anObj.L1EstimH().Val())->ReTagThis("L1EstimH"));
  for
  (       std::list< Pt2dr >::const_iterator it=anObj.PonderaL2Iter().begin();
      it !=anObj.PonderaL2Iter().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("PonderaL2Iter"),(*it))->ReTagThis("PonderaL2Iter"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cImSec & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Im(),aTree->Get("Im",1)); //tototo 

   xml_init(anObj.KeyCalcNameCorresp(),aTree->Get("KeyCalcNameCorresp",1)); //tototo 

   xml_init(anObj.OffsetPt(),aTree->Get("OffsetPt",1)); //tototo 

   xml_init(anObj.DirCalcCorrep(),aTree->Get("DirCalcCorrep",1),std::string("")); //tototo 

   xml_init(anObj.VerifHoms(),aTree->Get("VerifHoms",1)); //tototo 

   xml_init(anObj.NbTestRansacEstimH(),aTree->Get("NbTestRansacEstimH",1),int(30000)); //tototo 

   xml_init(anObj.NbPtsRansacEstimH(),aTree->Get("NbPtsRansacEstimH",1),int(1000)); //tototo 

   xml_init(anObj.L2EstimH(),aTree->Get("L2EstimH",1),bool(false)); //tototo 

   xml_init(anObj.L1EstimH(),aTree->Get("L1EstimH",1),bool(false)); //tototo 

   xml_init(anObj.PonderaL2Iter(),aTree->GetAll("PonderaL2Iter",false,1));
}


cTplValGesInit< cChantierDescripteur > & cCreateCompColoree::DicoLoc()
{
   return mDicoLoc;
}

const cTplValGesInit< cChantierDescripteur > & cCreateCompColoree::DicoLoc()const 
{
   return mDicoLoc;
}


std::list< cCmdMappeur > & cCreateCompColoree::MapCCC()
{
   return mMapCCC;
}

const std::list< cCmdMappeur > & cCreateCompColoree::MapCCC()const 
{
   return mMapCCC;
}


cTplValGesInit< double > & cCreateCompColoree::ParamBiCub()
{
   return mParamBiCub;
}

const cTplValGesInit< double > & cCreateCompColoree::ParamBiCub()const 
{
   return mParamBiCub;
}


double & cCreateCompColoree::StepGrid()
{
   return mStepGrid;
}

const double & cCreateCompColoree::StepGrid()const 
{
   return mStepGrid;
}


std::string & cCreateCompColoree::WorkDir()
{
   return mWorkDir;
}

const std::string & cCreateCompColoree::WorkDir()const 
{
   return mWorkDir;
}


std::list< cShowCalibsRel > & cCreateCompColoree::ShowCalibsRel()
{
   return mShowCalibsRel;
}

const std::list< cShowCalibsRel > & cCreateCompColoree::ShowCalibsRel()const 
{
   return mShowCalibsRel;
}


std::list< cResultCompCol > & cCreateCompColoree::ResultCompCol()
{
   return mResultCompCol;
}

const std::list< cResultCompCol > & cCreateCompColoree::ResultCompCol()const 
{
   return mResultCompCol;
}


std::string & cCreateCompColoree::KeyCalcNameCalib()
{
   return mKeyCalcNameCalib;
}

const std::string & cCreateCompColoree::KeyCalcNameCalib()const 
{
   return mKeyCalcNameCalib;
}


cTplValGesInit< string > & cCreateCompColoree::FileChantierNameDescripteur()
{
   return mFileChantierNameDescripteur;
}

const cTplValGesInit< string > & cCreateCompColoree::FileChantierNameDescripteur()const 
{
   return mFileChantierNameDescripteur;
}


cImageCmpCol & cCreateCompColoree::ImMaitresse()
{
   return mImMaitresse;
}

const cImageCmpCol & cCreateCompColoree::ImMaitresse()const 
{
   return mImMaitresse;
}


cTplValGesInit< std::string > & cCreateCompColoree::EnglobImMaitre()
{
   return EspaceResultSuperpCol().EnglobImMaitre();
}

const cTplValGesInit< std::string > & cCreateCompColoree::EnglobImMaitre()const 
{
   return EspaceResultSuperpCol().EnglobImMaitre();
}


cTplValGesInit< std::string > & cCreateCompColoree::EnglobAll()
{
   return EspaceResultSuperpCol().EnglobAll();
}

const cTplValGesInit< std::string > & cCreateCompColoree::EnglobAll()const 
{
   return EspaceResultSuperpCol().EnglobAll();
}


cTplValGesInit< Box2di > & cCreateCompColoree::EnglobBoxMaitresse()
{
   return EspaceResultSuperpCol().EnglobBoxMaitresse();
}

const cTplValGesInit< Box2di > & cCreateCompColoree::EnglobBoxMaitresse()const 
{
   return EspaceResultSuperpCol().EnglobBoxMaitresse();
}


cEspaceResultSuperpCol & cCreateCompColoree::EspaceResultSuperpCol()
{
   return mEspaceResultSuperpCol;
}

const cEspaceResultSuperpCol & cCreateCompColoree::EspaceResultSuperpCol()const 
{
   return mEspaceResultSuperpCol;
}


cTplValGesInit< Box2di > & cCreateCompColoree::BoxCalc()
{
   return mBoxCalc;
}

const cTplValGesInit< Box2di > & cCreateCompColoree::BoxCalc()const 
{
   return mBoxCalc;
}


cTplValGesInit< int > & cCreateCompColoree::TailleBloc()
{
   return mTailleBloc;
}

const cTplValGesInit< int > & cCreateCompColoree::TailleBloc()const 
{
   return mTailleBloc;
}


cTplValGesInit< int > & cCreateCompColoree::KBoxParal()
{
   return mKBoxParal;
}

const cTplValGesInit< int > & cCreateCompColoree::KBoxParal()const 
{
   return mKBoxParal;
}


cTplValGesInit< int > & cCreateCompColoree::ByProcess()
{
   return mByProcess;
}

const cTplValGesInit< int > & cCreateCompColoree::ByProcess()const 
{
   return mByProcess;
}


cTplValGesInit< bool > & cCreateCompColoree::CorDist()
{
   return mCorDist;
}

const cTplValGesInit< bool > & cCreateCompColoree::CorDist()const 
{
   return mCorDist;
}


cTplValGesInit< double > & cCreateCompColoree::ScaleFus()
{
   return mScaleFus;
}

const cTplValGesInit< double > & cCreateCompColoree::ScaleFus()const 
{
   return mScaleFus;
}


std::list< cImSec > & cCreateCompColoree::ImSec()
{
   return mImSec;
}

const std::list< cImSec > & cCreateCompColoree::ImSec()const 
{
   return mImSec;
}

cElXMLTree * ToXMLTree(const cCreateCompColoree & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"CreateCompColoree",eXMLBranche);
   if (anObj.DicoLoc().IsInit())
      aRes->AddFils(ToXMLTree(anObj.DicoLoc().Val())->ReTagThis("DicoLoc"));
  for
  (       std::list< cCmdMappeur >::const_iterator it=anObj.MapCCC().begin();
      it !=anObj.MapCCC().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("MapCCC"));
   if (anObj.ParamBiCub().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ParamBiCub"),anObj.ParamBiCub().Val())->ReTagThis("ParamBiCub"));
   aRes->AddFils(::ToXMLTree(std::string("StepGrid"),anObj.StepGrid())->ReTagThis("StepGrid"));
   aRes->AddFils(::ToXMLTree(std::string("WorkDir"),anObj.WorkDir())->ReTagThis("WorkDir"));
  for
  (       std::list< cShowCalibsRel >::const_iterator it=anObj.ShowCalibsRel().begin();
      it !=anObj.ShowCalibsRel().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("ShowCalibsRel"));
  for
  (       std::list< cResultCompCol >::const_iterator it=anObj.ResultCompCol().begin();
      it !=anObj.ResultCompCol().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("ResultCompCol"));
   aRes->AddFils(::ToXMLTree(std::string("KeyCalcNameCalib"),anObj.KeyCalcNameCalib())->ReTagThis("KeyCalcNameCalib"));
   if (anObj.FileChantierNameDescripteur().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("FileChantierNameDescripteur"),anObj.FileChantierNameDescripteur().Val())->ReTagThis("FileChantierNameDescripteur"));
   aRes->AddFils(ToXMLTree(anObj.ImMaitresse())->ReTagThis("ImMaitresse"));
   aRes->AddFils(ToXMLTree(anObj.EspaceResultSuperpCol())->ReTagThis("EspaceResultSuperpCol"));
   if (anObj.BoxCalc().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("BoxCalc"),anObj.BoxCalc().Val())->ReTagThis("BoxCalc"));
   if (anObj.TailleBloc().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("TailleBloc"),anObj.TailleBloc().Val())->ReTagThis("TailleBloc"));
   if (anObj.KBoxParal().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("KBoxParal"),anObj.KBoxParal().Val())->ReTagThis("KBoxParal"));
   if (anObj.ByProcess().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ByProcess"),anObj.ByProcess().Val())->ReTagThis("ByProcess"));
   if (anObj.CorDist().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("CorDist"),anObj.CorDist().Val())->ReTagThis("CorDist"));
   if (anObj.ScaleFus().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ScaleFus"),anObj.ScaleFus().Val())->ReTagThis("ScaleFus"));
  for
  (       std::list< cImSec >::const_iterator it=anObj.ImSec().begin();
      it !=anObj.ImSec().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("ImSec"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cCreateCompColoree & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.DicoLoc(),aTree->Get("DicoLoc",1)); //tototo 

   xml_init(anObj.MapCCC(),aTree->GetAll("MapCCC",false,1));

   xml_init(anObj.ParamBiCub(),aTree->Get("ParamBiCub",1)); //tototo 

   xml_init(anObj.StepGrid(),aTree->Get("StepGrid",1)); //tototo 

   xml_init(anObj.WorkDir(),aTree->Get("WorkDir",1)); //tototo 

   xml_init(anObj.ShowCalibsRel(),aTree->GetAll("ShowCalibsRel",false,1));

   xml_init(anObj.ResultCompCol(),aTree->GetAll("ResultCompCol",false,1));

   xml_init(anObj.KeyCalcNameCalib(),aTree->Get("KeyCalcNameCalib",1)); //tototo 

   xml_init(anObj.FileChantierNameDescripteur(),aTree->Get("FileChantierNameDescripteur",1)); //tototo 

   xml_init(anObj.ImMaitresse(),aTree->Get("ImMaitresse",1)); //tototo 

   xml_init(anObj.EspaceResultSuperpCol(),aTree->Get("EspaceResultSuperpCol",1)); //tototo 

   xml_init(anObj.BoxCalc(),aTree->Get("BoxCalc",1)); //tototo 

   xml_init(anObj.TailleBloc(),aTree->Get("TailleBloc",1)); //tototo 

   xml_init(anObj.KBoxParal(),aTree->Get("KBoxParal",1)); //tototo 

   xml_init(anObj.ByProcess(),aTree->Get("ByProcess",1)); //tototo 

   xml_init(anObj.CorDist(),aTree->Get("CorDist",1),bool(false)); //tototo 

   xml_init(anObj.ScaleFus(),aTree->Get("ScaleFus",1),double(1.0)); //tototo 

   xml_init(anObj.ImSec(),aTree->GetAll("ImSec",false,1));
}


std::string & cSauvegardeMR2A::NameSauvMR2A()
{
   return mNameSauvMR2A;
}

const std::string & cSauvegardeMR2A::NameSauvMR2A()const 
{
   return mNameSauvMR2A;
}


double & cSauvegardeMR2A::StepGridMR2A()
{
   return mStepGridMR2A;
}

const double & cSauvegardeMR2A::StepGridMR2A()const 
{
   return mStepGridMR2A;
}


cTplValGesInit< std::string > & cSauvegardeMR2A::SauvImgMR2A()
{
   return mSauvImgMR2A;
}

const cTplValGesInit< std::string > & cSauvegardeMR2A::SauvImgMR2A()const 
{
   return mSauvImgMR2A;
}

cElXMLTree * ToXMLTree(const cSauvegardeMR2A & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"SauvegardeMR2A",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("NameSauvMR2A"),anObj.NameSauvMR2A())->ReTagThis("NameSauvMR2A"));
   aRes->AddFils(::ToXMLTree(std::string("StepGridMR2A"),anObj.StepGridMR2A())->ReTagThis("StepGridMR2A"));
   if (anObj.SauvImgMR2A().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SauvImgMR2A"),anObj.SauvImgMR2A().Val())->ReTagThis("SauvImgMR2A"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cSauvegardeMR2A & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.NameSauvMR2A(),aTree->Get("NameSauvMR2A",1)); //tototo 

   xml_init(anObj.StepGridMR2A(),aTree->Get("StepGridMR2A",1)); //tototo 

   xml_init(anObj.SauvImgMR2A(),aTree->Get("SauvImgMR2A",1)); //tototo 
}


std::string & cGenereModeleRaster2Analytique::Dir()
{
   return mDir;
}

const std::string & cGenereModeleRaster2Analytique::Dir()const 
{
   return mDir;
}


std::string & cGenereModeleRaster2Analytique::Im1()
{
   return mIm1;
}

const std::string & cGenereModeleRaster2Analytique::Im1()const 
{
   return mIm1;
}


std::string & cGenereModeleRaster2Analytique::Im2()
{
   return mIm2;
}

const std::string & cGenereModeleRaster2Analytique::Im2()const 
{
   return mIm2;
}


double & cGenereModeleRaster2Analytique::SsResol()
{
   return mSsResol;
}

const double & cGenereModeleRaster2Analytique::SsResol()const 
{
   return mSsResol;
}


Pt2dr & cGenereModeleRaster2Analytique::Pas()
{
   return mPas;
}

const Pt2dr & cGenereModeleRaster2Analytique::Pas()const 
{
   return mPas;
}


cTplValGesInit< Pt2dr > & cGenereModeleRaster2Analytique::Tr0()
{
   return mTr0;
}

const cTplValGesInit< Pt2dr > & cGenereModeleRaster2Analytique::Tr0()const 
{
   return mTr0;
}


cTplValGesInit< bool > & cGenereModeleRaster2Analytique::AutoCalcTr0()
{
   return mAutoCalcTr0;
}

const cTplValGesInit< bool > & cGenereModeleRaster2Analytique::AutoCalcTr0()const 
{
   return mAutoCalcTr0;
}


cTplValGesInit< double > & cGenereModeleRaster2Analytique::RoundTr0()
{
   return mRoundTr0;
}

const cTplValGesInit< double > & cGenereModeleRaster2Analytique::RoundTr0()const 
{
   return mRoundTr0;
}


int & cGenereModeleRaster2Analytique::DegPoly()
{
   return mDegPoly;
}

const int & cGenereModeleRaster2Analytique::DegPoly()const 
{
   return mDegPoly;
}


bool & cGenereModeleRaster2Analytique::CLibre()
{
   return mCLibre;
}

const bool & cGenereModeleRaster2Analytique::CLibre()const 
{
   return mCLibre;
}


bool & cGenereModeleRaster2Analytique::Dequant()
{
   return mDequant;
}

const bool & cGenereModeleRaster2Analytique::Dequant()const 
{
   return mDequant;
}


std::string & cGenereModeleRaster2Analytique::NameSauvMR2A()
{
   return SauvegardeMR2A().Val().NameSauvMR2A();
}

const std::string & cGenereModeleRaster2Analytique::NameSauvMR2A()const 
{
   return SauvegardeMR2A().Val().NameSauvMR2A();
}


double & cGenereModeleRaster2Analytique::StepGridMR2A()
{
   return SauvegardeMR2A().Val().StepGridMR2A();
}

const double & cGenereModeleRaster2Analytique::StepGridMR2A()const 
{
   return SauvegardeMR2A().Val().StepGridMR2A();
}


cTplValGesInit< std::string > & cGenereModeleRaster2Analytique::SauvImgMR2A()
{
   return SauvegardeMR2A().Val().SauvImgMR2A();
}

const cTplValGesInit< std::string > & cGenereModeleRaster2Analytique::SauvImgMR2A()const 
{
   return SauvegardeMR2A().Val().SauvImgMR2A();
}


cTplValGesInit< cSauvegardeMR2A > & cGenereModeleRaster2Analytique::SauvegardeMR2A()
{
   return mSauvegardeMR2A;
}

const cTplValGesInit< cSauvegardeMR2A > & cGenereModeleRaster2Analytique::SauvegardeMR2A()const 
{
   return mSauvegardeMR2A;
}

cElXMLTree * ToXMLTree(const cGenereModeleRaster2Analytique & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"GenereModeleRaster2Analytique",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Dir"),anObj.Dir())->ReTagThis("Dir"));
   aRes->AddFils(::ToXMLTree(std::string("Im1"),anObj.Im1())->ReTagThis("Im1"));
   aRes->AddFils(::ToXMLTree(std::string("Im2"),anObj.Im2())->ReTagThis("Im2"));
   aRes->AddFils(::ToXMLTree(std::string("SsResol"),anObj.SsResol())->ReTagThis("SsResol"));
   aRes->AddFils(::ToXMLTree(std::string("Pas"),anObj.Pas())->ReTagThis("Pas"));
   if (anObj.Tr0().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Tr0"),anObj.Tr0().Val())->ReTagThis("Tr0"));
   if (anObj.AutoCalcTr0().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("AutoCalcTr0"),anObj.AutoCalcTr0().Val())->ReTagThis("AutoCalcTr0"));
   if (anObj.RoundTr0().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("RoundTr0"),anObj.RoundTr0().Val())->ReTagThis("RoundTr0"));
   aRes->AddFils(::ToXMLTree(std::string("DegPoly"),anObj.DegPoly())->ReTagThis("DegPoly"));
   aRes->AddFils(::ToXMLTree(std::string("CLibre"),anObj.CLibre())->ReTagThis("CLibre"));
   aRes->AddFils(::ToXMLTree(std::string("Dequant"),anObj.Dequant())->ReTagThis("Dequant"));
   if (anObj.SauvegardeMR2A().IsInit())
      aRes->AddFils(ToXMLTree(anObj.SauvegardeMR2A().Val())->ReTagThis("SauvegardeMR2A"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cGenereModeleRaster2Analytique & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Dir(),aTree->Get("Dir",1)); //tototo 

   xml_init(anObj.Im1(),aTree->Get("Im1",1)); //tototo 

   xml_init(anObj.Im2(),aTree->Get("Im2",1)); //tototo 

   xml_init(anObj.SsResol(),aTree->Get("SsResol",1)); //tototo 

   xml_init(anObj.Pas(),aTree->Get("Pas",1)); //tototo 

   xml_init(anObj.Tr0(),aTree->Get("Tr0",1),Pt2dr(0,0)); //tototo 

   xml_init(anObj.AutoCalcTr0(),aTree->Get("AutoCalcTr0",1),bool(false)); //tototo 

   xml_init(anObj.RoundTr0(),aTree->Get("RoundTr0",1),double(0.5)); //tototo 

   xml_init(anObj.DegPoly(),aTree->Get("DegPoly",1)); //tototo 

   xml_init(anObj.CLibre(),aTree->Get("CLibre",1)); //tototo 

   xml_init(anObj.Dequant(),aTree->Get("Dequant",1)); //tototo 

   xml_init(anObj.SauvegardeMR2A(),aTree->Get("SauvegardeMR2A",1)); //tototo 
}


std::string & cBayerGridDirecteEtInverse::Ch1()
{
   return mCh1;
}

const std::string & cBayerGridDirecteEtInverse::Ch1()const 
{
   return mCh1;
}


std::string & cBayerGridDirecteEtInverse::Ch2()
{
   return mCh2;
}

const std::string & cBayerGridDirecteEtInverse::Ch2()const 
{
   return mCh2;
}


cGridDirecteEtInverse & cBayerGridDirecteEtInverse::Grid()
{
   return mGrid;
}

const cGridDirecteEtInverse & cBayerGridDirecteEtInverse::Grid()const 
{
   return mGrid;
}

cElXMLTree * ToXMLTree(const cBayerGridDirecteEtInverse & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"BayerGridDirecteEtInverse",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Ch1"),anObj.Ch1())->ReTagThis("Ch1"));
   aRes->AddFils(::ToXMLTree(std::string("Ch2"),anObj.Ch2())->ReTagThis("Ch2"));
   aRes->AddFils(ToXMLTree(anObj.Grid())->ReTagThis("Grid"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cBayerGridDirecteEtInverse & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Ch1(),aTree->Get("Ch1",1)); //tototo 

   xml_init(anObj.Ch2(),aTree->Get("Ch2",1)); //tototo 

   xml_init(anObj.Grid(),aTree->Get("Grid",1)); //tototo 
}


std::list< cBayerGridDirecteEtInverse > & cBayerCalibGeom::Grids()
{
   return mGrids;
}

const std::list< cBayerGridDirecteEtInverse > & cBayerCalibGeom::Grids()const 
{
   return mGrids;
}


cTplValGesInit< Pt3dr > & cBayerCalibGeom::WB()
{
   return mWB;
}

const cTplValGesInit< Pt3dr > & cBayerCalibGeom::WB()const 
{
   return mWB;
}


cTplValGesInit< Pt3dr > & cBayerCalibGeom::PG()
{
   return mPG;
}

const cTplValGesInit< Pt3dr > & cBayerCalibGeom::PG()const 
{
   return mPG;
}

cElXMLTree * ToXMLTree(const cBayerCalibGeom & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"BayerCalibGeom",eXMLBranche);
  for
  (       std::list< cBayerGridDirecteEtInverse >::const_iterator it=anObj.Grids().begin();
      it !=anObj.Grids().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("Grids"));
   if (anObj.WB().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("WB"),anObj.WB().Val())->ReTagThis("WB"));
   if (anObj.PG().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PG"),anObj.PG().Val())->ReTagThis("PG"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cBayerCalibGeom & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Grids(),aTree->GetAll("Grids",false,1));

   xml_init(anObj.WB(),aTree->Get("WB",1)); //tototo 

   xml_init(anObj.PG(),aTree->Get("PG",1)); //tototo 
}


int & cSpecifEtalRelOneChan::DegreOwn()
{
   return mDegreOwn;
}

const int & cSpecifEtalRelOneChan::DegreOwn()const 
{
   return mDegreOwn;
}


int & cSpecifEtalRelOneChan::DegreOther()
{
   return mDegreOther;
}

const int & cSpecifEtalRelOneChan::DegreOther()const 
{
   return mDegreOther;
}

cElXMLTree * ToXMLTree(const cSpecifEtalRelOneChan & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"SpecifEtalRelOneChan",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("DegreOwn"),anObj.DegreOwn())->ReTagThis("DegreOwn"));
   aRes->AddFils(::ToXMLTree(std::string("DegreOther"),anObj.DegreOther())->ReTagThis("DegreOther"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cSpecifEtalRelOneChan & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.DegreOwn(),aTree->Get("DegreOwn",1)); //tototo 

   xml_init(anObj.DegreOther(),aTree->Get("DegreOther",1)); //tototo 
}


std::list< cSpecifEtalRelOneChan > & cSpecifEtalRadiom::Channel()
{
   return mChannel;
}

const std::list< cSpecifEtalRelOneChan > & cSpecifEtalRadiom::Channel()const 
{
   return mChannel;
}

cElXMLTree * ToXMLTree(const cSpecifEtalRadiom & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"SpecifEtalRadiom",eXMLBranche);
  for
  (       std::list< cSpecifEtalRelOneChan >::const_iterator it=anObj.Channel().begin();
      it !=anObj.Channel().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("Channel"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cSpecifEtalRadiom & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Channel(),aTree->GetAll("Channel",false,1));
}


std::vector< int > & cPolyNRadiom::Degre()
{
   return mDegre;
}

const std::vector< int > & cPolyNRadiom::Degre()const 
{
   return mDegre;
}


double & cPolyNRadiom::Val()
{
   return mVal;
}

const double & cPolyNRadiom::Val()const 
{
   return mVal;
}

cElXMLTree * ToXMLTree(const cPolyNRadiom & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"PolyNRadiom",eXMLBranche);
  for
  (       std::vector< int >::const_iterator it=anObj.Degre().begin();
      it !=anObj.Degre().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("Degre"),(*it))->ReTagThis("Degre"));
   aRes->AddFils(::ToXMLTree(std::string("Val"),anObj.Val())->ReTagThis("Val"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cPolyNRadiom & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Degre(),aTree->GetAll("Degre",false,1));

   xml_init(anObj.Val(),aTree->Get("Val",1)); //tototo 
}


std::vector< cPolyNRadiom > & cEtalRelOneChan::PolyNRadiom()
{
   return mPolyNRadiom;
}

const std::vector< cPolyNRadiom > & cEtalRelOneChan::PolyNRadiom()const 
{
   return mPolyNRadiom;
}

cElXMLTree * ToXMLTree(const cEtalRelOneChan & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"EtalRelOneChan",eXMLBranche);
  for
  (       std::vector< cPolyNRadiom >::const_iterator it=anObj.PolyNRadiom().begin();
      it !=anObj.PolyNRadiom().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("PolyNRadiom"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cEtalRelOneChan & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.PolyNRadiom(),aTree->GetAll("PolyNRadiom",false,1));
}


std::vector< cEtalRelOneChan > & cColorCalib::CalibChannel()
{
   return mCalibChannel;
}

const std::vector< cEtalRelOneChan > & cColorCalib::CalibChannel()const 
{
   return mCalibChannel;
}

cElXMLTree * ToXMLTree(const cColorCalib & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ColorCalib",eXMLBranche);
  for
  (       std::vector< cEtalRelOneChan >::const_iterator it=anObj.CalibChannel().begin();
      it !=anObj.CalibChannel().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("CalibChannel"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cColorCalib & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.CalibChannel(),aTree->GetAll("CalibChannel",false,1));
}


std::string & cOneGridECG::Name()
{
   return mName;
}

const std::string & cOneGridECG::Name()const 
{
   return mName;
}


bool & cOneGridECG::Direct()
{
   return mDirect;
}

const bool & cOneGridECG::Direct()const 
{
   return mDirect;
}

cElXMLTree * ToXMLTree(const cOneGridECG & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"OneGridECG",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Name"),anObj.Name())->ReTagThis("Name"));
   aRes->AddFils(::ToXMLTree(std::string("Direct"),anObj.Direct())->ReTagThis("Direct"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cOneGridECG & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Name(),aTree->Get("Name",1)); //tototo 

   xml_init(anObj.Direct(),aTree->Get("Direct",1)); //tototo 
}


cTplValGesInit< std::string > & cEvalComposeGrid::Directory()
{
   return mDirectory;
}

const cTplValGesInit< std::string > & cEvalComposeGrid::Directory()const 
{
   return mDirectory;
}


double & cEvalComposeGrid::Dyn()
{
   return mDyn;
}

const double & cEvalComposeGrid::Dyn()const 
{
   return mDyn;
}


double & cEvalComposeGrid::Resol()
{
   return mResol;
}

const double & cEvalComposeGrid::Resol()const 
{
   return mResol;
}


std::list< cOneGridECG > & cEvalComposeGrid::OneGridECG()
{
   return mOneGridECG;
}

const std::list< cOneGridECG > & cEvalComposeGrid::OneGridECG()const 
{
   return mOneGridECG;
}


cTplValGesInit< std::string > & cEvalComposeGrid::NameNorm()
{
   return mNameNorm;
}

const cTplValGesInit< std::string > & cEvalComposeGrid::NameNorm()const 
{
   return mNameNorm;
}

cElXMLTree * ToXMLTree(const cEvalComposeGrid & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"EvalComposeGrid",eXMLBranche);
   if (anObj.Directory().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Directory"),anObj.Directory().Val())->ReTagThis("Directory"));
   aRes->AddFils(::ToXMLTree(std::string("Dyn"),anObj.Dyn())->ReTagThis("Dyn"));
   aRes->AddFils(::ToXMLTree(std::string("Resol"),anObj.Resol())->ReTagThis("Resol"));
  for
  (       std::list< cOneGridECG >::const_iterator it=anObj.OneGridECG().begin();
      it !=anObj.OneGridECG().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("OneGridECG"));
   if (anObj.NameNorm().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NameNorm"),anObj.NameNorm().Val())->ReTagThis("NameNorm"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cEvalComposeGrid & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Directory(),aTree->Get("Directory",1),std::string("")); //tototo 

   xml_init(anObj.Dyn(),aTree->Get("Dyn",1)); //tototo 

   xml_init(anObj.Resol(),aTree->Get("Resol",1)); //tototo 

   xml_init(anObj.OneGridECG(),aTree->GetAll("OneGridECG",false,1));

   xml_init(anObj.NameNorm(),aTree->Get("NameNorm",1)); //tototo 
}


std::string & cCalcNomFromCouple::Pattern2Match()
{
   return mPattern2Match;
}

const std::string & cCalcNomFromCouple::Pattern2Match()const 
{
   return mPattern2Match;
}


cTplValGesInit< std::string > & cCalcNomFromCouple::Separateur()
{
   return mSeparateur;
}

const cTplValGesInit< std::string > & cCalcNomFromCouple::Separateur()const 
{
   return mSeparateur;
}


std::string & cCalcNomFromCouple::NameCalculated()
{
   return mNameCalculated;
}

const std::string & cCalcNomFromCouple::NameCalculated()const 
{
   return mNameCalculated;
}

cElXMLTree * ToXMLTree(const cCalcNomFromCouple & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"CalcNomFromCouple",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Pattern2Match"),anObj.Pattern2Match())->ReTagThis("Pattern2Match"));
   if (anObj.Separateur().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Separateur"),anObj.Separateur().Val())->ReTagThis("Separateur"));
   aRes->AddFils(::ToXMLTree(std::string("NameCalculated"),anObj.NameCalculated())->ReTagThis("NameCalculated"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cCalcNomFromCouple & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Pattern2Match(),aTree->Get("Pattern2Match",1)); //tototo 

   xml_init(anObj.Separateur(),aTree->Get("Separateur",1),std::string("")); //tototo 

   xml_init(anObj.NameCalculated(),aTree->Get("NameCalculated",1)); //tototo 
}


std::string & cCalcNomFromOne::Pattern2Match()
{
   return mPattern2Match;
}

const std::string & cCalcNomFromOne::Pattern2Match()const 
{
   return mPattern2Match;
}


std::string & cCalcNomFromOne::NameCalculated()
{
   return mNameCalculated;
}

const std::string & cCalcNomFromOne::NameCalculated()const 
{
   return mNameCalculated;
}

cElXMLTree * ToXMLTree(const cCalcNomFromOne & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"CalcNomFromOne",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Pattern2Match"),anObj.Pattern2Match())->ReTagThis("Pattern2Match"));
   aRes->AddFils(::ToXMLTree(std::string("NameCalculated"),anObj.NameCalculated())->ReTagThis("NameCalculated"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cCalcNomFromOne & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Pattern2Match(),aTree->Get("Pattern2Match",1)); //tototo 

   xml_init(anObj.NameCalculated(),aTree->Get("NameCalculated",1)); //tototo 
}


std::string & cOneResync::Dir()
{
   return mDir;
}

const std::string & cOneResync::Dir()const 
{
   return mDir;
}


std::string & cOneResync::PatSel()
{
   return mPatSel;
}

const std::string & cOneResync::PatSel()const 
{
   return mPatSel;
}


std::string & cOneResync::PatRename()
{
   return mPatRename;
}

const std::string & cOneResync::PatRename()const 
{
   return mPatRename;
}


std::string & cOneResync::Rename()
{
   return mRename;
}

const std::string & cOneResync::Rename()const 
{
   return mRename;
}

cElXMLTree * ToXMLTree(const cOneResync & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"OneResync",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Dir"),anObj.Dir())->ReTagThis("Dir"));
   aRes->AddFils(::ToXMLTree(std::string("PatSel"),anObj.PatSel())->ReTagThis("PatSel"));
   aRes->AddFils(::ToXMLTree(std::string("PatRename"),anObj.PatRename())->ReTagThis("PatRename"));
   aRes->AddFils(::ToXMLTree(std::string("Rename"),anObj.Rename())->ReTagThis("Rename"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cOneResync & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Dir(),aTree->Get("Dir",1)); //tototo 

   xml_init(anObj.PatSel(),aTree->Get("PatSel",1)); //tototo 

   xml_init(anObj.PatRename(),aTree->Get("PatRename",1)); //tototo 

   xml_init(anObj.Rename(),aTree->Get("Rename",1)); //tototo 
}


std::list< cOneResync > & cReSynchronImage::OneResync()
{
   return mOneResync;
}

const std::list< cOneResync > & cReSynchronImage::OneResync()const 
{
   return mOneResync;
}


double & cReSynchronImage::EcartMin()
{
   return mEcartMin;
}

const double & cReSynchronImage::EcartMin()const 
{
   return mEcartMin;
}


double & cReSynchronImage::EcartMax()
{
   return mEcartMax;
}

const double & cReSynchronImage::EcartMax()const 
{
   return mEcartMax;
}


cTplValGesInit< double > & cReSynchronImage::EcartRechAuto()
{
   return mEcartRechAuto;
}

const cTplValGesInit< double > & cReSynchronImage::EcartRechAuto()const 
{
   return mEcartRechAuto;
}


cTplValGesInit< double > & cReSynchronImage::SigmaRechAuto()
{
   return mSigmaRechAuto;
}

const cTplValGesInit< double > & cReSynchronImage::SigmaRechAuto()const 
{
   return mSigmaRechAuto;
}


cTplValGesInit< double > & cReSynchronImage::EcartCalcMoyRechAuto()
{
   return mEcartCalcMoyRechAuto;
}

const cTplValGesInit< double > & cReSynchronImage::EcartCalcMoyRechAuto()const 
{
   return mEcartCalcMoyRechAuto;
}

cElXMLTree * ToXMLTree(const cReSynchronImage & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ReSynchronImage",eXMLBranche);
  for
  (       std::list< cOneResync >::const_iterator it=anObj.OneResync().begin();
      it !=anObj.OneResync().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("OneResync"));
   aRes->AddFils(::ToXMLTree(std::string("EcartMin"),anObj.EcartMin())->ReTagThis("EcartMin"));
   aRes->AddFils(::ToXMLTree(std::string("EcartMax"),anObj.EcartMax())->ReTagThis("EcartMax"));
   if (anObj.EcartRechAuto().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("EcartRechAuto"),anObj.EcartRechAuto().Val())->ReTagThis("EcartRechAuto"));
   if (anObj.SigmaRechAuto().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SigmaRechAuto"),anObj.SigmaRechAuto().Val())->ReTagThis("SigmaRechAuto"));
   if (anObj.EcartCalcMoyRechAuto().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("EcartCalcMoyRechAuto"),anObj.EcartCalcMoyRechAuto().Val())->ReTagThis("EcartCalcMoyRechAuto"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cReSynchronImage & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.OneResync(),aTree->GetAll("OneResync",false,1));

   xml_init(anObj.EcartMin(),aTree->Get("EcartMin",1)); //tototo 

   xml_init(anObj.EcartMax(),aTree->Get("EcartMax",1)); //tototo 

   xml_init(anObj.EcartRechAuto(),aTree->Get("EcartRechAuto",1),double(4.0)); //tototo 

   xml_init(anObj.SigmaRechAuto(),aTree->Get("SigmaRechAuto",1),double(1.0)); //tototo 

   xml_init(anObj.EcartCalcMoyRechAuto(),aTree->Get("EcartCalcMoyRechAuto",1),double(1.5)); //tototo 
}


Pt3dr & cXmlCylindreRevolution::P0()
{
   return mP0;
}

const Pt3dr & cXmlCylindreRevolution::P0()const 
{
   return mP0;
}


Pt3dr & cXmlCylindreRevolution::P1()
{
   return mP1;
}

const Pt3dr & cXmlCylindreRevolution::P1()const 
{
   return mP1;
}


Pt3dr & cXmlCylindreRevolution::POnCyl()
{
   return mPOnCyl;
}

const Pt3dr & cXmlCylindreRevolution::POnCyl()const 
{
   return mPOnCyl;
}

cElXMLTree * ToXMLTree(const cXmlCylindreRevolution & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"XmlCylindreRevolution",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("P0"),anObj.P0())->ReTagThis("P0"));
   aRes->AddFils(::ToXMLTree(std::string("P1"),anObj.P1())->ReTagThis("P1"));
   aRes->AddFils(::ToXMLTree(std::string("POnCyl"),anObj.POnCyl())->ReTagThis("POnCyl"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cXmlCylindreRevolution & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.P0(),aTree->Get("P0",1)); //tototo 

   xml_init(anObj.P1(),aTree->Get("P1",1)); //tototo 

   xml_init(anObj.POnCyl(),aTree->Get("POnCyl",1)); //tototo 
}


cRepereCartesien & cXmlOrthoCyl::Repere()
{
   return mRepere;
}

const cRepereCartesien & cXmlOrthoCyl::Repere()const 
{
   return mRepere;
}


Pt3dr & cXmlOrthoCyl::P0()
{
   return mP0;
}

const Pt3dr & cXmlOrthoCyl::P0()const 
{
   return mP0;
}


Pt3dr & cXmlOrthoCyl::P1()
{
   return mP1;
}

const Pt3dr & cXmlOrthoCyl::P1()const 
{
   return mP1;
}


bool & cXmlOrthoCyl::AngulCorr()
{
   return mAngulCorr;
}

const bool & cXmlOrthoCyl::AngulCorr()const 
{
   return mAngulCorr;
}

cElXMLTree * ToXMLTree(const cXmlOrthoCyl & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"XmlOrthoCyl",eXMLBranche);
   aRes->AddFils(ToXMLTree(anObj.Repere())->ReTagThis("Repere"));
   aRes->AddFils(::ToXMLTree(std::string("P0"),anObj.P0())->ReTagThis("P0"));
   aRes->AddFils(::ToXMLTree(std::string("P1"),anObj.P1())->ReTagThis("P1"));
   aRes->AddFils(::ToXMLTree(std::string("AngulCorr"),anObj.AngulCorr())->ReTagThis("AngulCorr"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cXmlOrthoCyl & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Repere(),aTree->Get("Repere",1)); //tototo 

   xml_init(anObj.P0(),aTree->Get("P0",1)); //tototo 

   xml_init(anObj.P1(),aTree->Get("P1",1)); //tototo 

   xml_init(anObj.AngulCorr(),aTree->Get("AngulCorr",1)); //tototo 
}


cTplValGesInit< cXmlCylindreRevolution > & cXmlDescriptionAnalytique::Cyl()
{
   return mCyl;
}

const cTplValGesInit< cXmlCylindreRevolution > & cXmlDescriptionAnalytique::Cyl()const 
{
   return mCyl;
}


cTplValGesInit< cXmlOrthoCyl > & cXmlDescriptionAnalytique::OrthoCyl()
{
   return mOrthoCyl;
}

const cTplValGesInit< cXmlOrthoCyl > & cXmlDescriptionAnalytique::OrthoCyl()const 
{
   return mOrthoCyl;
}

cElXMLTree * ToXMLTree(const cXmlDescriptionAnalytique & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"XmlDescriptionAnalytique",eXMLBranche);
   if (anObj.Cyl().IsInit())
      aRes->AddFils(ToXMLTree(anObj.Cyl().Val())->ReTagThis("Cyl"));
   if (anObj.OrthoCyl().IsInit())
      aRes->AddFils(ToXMLTree(anObj.OrthoCyl().Val())->ReTagThis("OrthoCyl"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cXmlDescriptionAnalytique & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Cyl(),aTree->Get("Cyl",1)); //tototo 

   xml_init(anObj.OrthoCyl(),aTree->Get("OrthoCyl",1)); //tototo 
}


cXmlDescriptionAnalytique & cXmlOneSurfaceAnalytique::XmlDescriptionAnalytique()
{
   return mXmlDescriptionAnalytique;
}

const cXmlDescriptionAnalytique & cXmlOneSurfaceAnalytique::XmlDescriptionAnalytique()const 
{
   return mXmlDescriptionAnalytique;
}


std::string & cXmlOneSurfaceAnalytique::Id()
{
   return mId;
}

const std::string & cXmlOneSurfaceAnalytique::Id()const 
{
   return mId;
}


bool & cXmlOneSurfaceAnalytique::VueDeLExterieur()
{
   return mVueDeLExterieur;
}

const bool & cXmlOneSurfaceAnalytique::VueDeLExterieur()const 
{
   return mVueDeLExterieur;
}

cElXMLTree * ToXMLTree(const cXmlOneSurfaceAnalytique & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"XmlOneSurfaceAnalytique",eXMLBranche);
   aRes->AddFils(ToXMLTree(anObj.XmlDescriptionAnalytique())->ReTagThis("XmlDescriptionAnalytique"));
   aRes->AddFils(::ToXMLTree(std::string("Id"),anObj.Id())->ReTagThis("Id"));
   aRes->AddFils(::ToXMLTree(std::string("VueDeLExterieur"),anObj.VueDeLExterieur())->ReTagThis("VueDeLExterieur"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cXmlOneSurfaceAnalytique & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.XmlDescriptionAnalytique(),aTree->Get("XmlDescriptionAnalytique",1)); //tototo 

   xml_init(anObj.Id(),aTree->Get("Id",1)); //tototo 

   xml_init(anObj.VueDeLExterieur(),aTree->Get("VueDeLExterieur",1)); //tototo 
}


std::list< cXmlOneSurfaceAnalytique > & cXmlModeleSurfaceComplexe::XmlOneSurfaceAnalytique()
{
   return mXmlOneSurfaceAnalytique;
}

const std::list< cXmlOneSurfaceAnalytique > & cXmlModeleSurfaceComplexe::XmlOneSurfaceAnalytique()const 
{
   return mXmlOneSurfaceAnalytique;
}

cElXMLTree * ToXMLTree(const cXmlModeleSurfaceComplexe & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"XmlModeleSurfaceComplexe",eXMLBranche);
  for
  (       std::list< cXmlOneSurfaceAnalytique >::const_iterator it=anObj.XmlOneSurfaceAnalytique().begin();
      it !=anObj.XmlOneSurfaceAnalytique().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("XmlOneSurfaceAnalytique"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cXmlModeleSurfaceComplexe & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.XmlOneSurfaceAnalytique(),aTree->GetAll("XmlOneSurfaceAnalytique",false,1));
}


std::string & cMapByKey::Key()
{
   return mKey;
}

const std::string & cMapByKey::Key()const 
{
   return mKey;
}


cTplValGesInit< bool > & cMapByKey::DefIfFileNotExisting()
{
   return mDefIfFileNotExisting;
}

const cTplValGesInit< bool > & cMapByKey::DefIfFileNotExisting()const 
{
   return mDefIfFileNotExisting;
}

cElXMLTree * ToXMLTree(const cMapByKey & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"MapByKey",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Key"),anObj.Key())->ReTagThis("Key"));
   if (anObj.DefIfFileNotExisting().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DefIfFileNotExisting"),anObj.DefIfFileNotExisting().Val())->ReTagThis("DefIfFileNotExisting"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cMapByKey & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Key(),aTree->Get("Key",1)); //tototo 

   xml_init(anObj.DefIfFileNotExisting(),aTree->Get("DefIfFileNotExisting",1),bool(false)); //tototo 
}


cElRegex_Ptr & cOneAutomMapN2N::MatchPattern()
{
   return mMatchPattern;
}

const cElRegex_Ptr & cOneAutomMapN2N::MatchPattern()const 
{
   return mMatchPattern;
}


cTplValGesInit< cElRegex_Ptr > & cOneAutomMapN2N::AutomSel()
{
   return mAutomSel;
}

const cTplValGesInit< cElRegex_Ptr > & cOneAutomMapN2N::AutomSel()const 
{
   return mAutomSel;
}


std::string & cOneAutomMapN2N::Result()
{
   return mResult;
}

const std::string & cOneAutomMapN2N::Result()const 
{
   return mResult;
}

cElXMLTree * ToXMLTree(const cOneAutomMapN2N & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"OneAutomMapN2N",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("MatchPattern"),anObj.MatchPattern())->ReTagThis("MatchPattern"));
   if (anObj.AutomSel().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("AutomSel"),anObj.AutomSel().Val())->ReTagThis("AutomSel"));
   aRes->AddFils(::ToXMLTree(std::string("Result"),anObj.Result())->ReTagThis("Result"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cOneAutomMapN2N & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.MatchPattern(),aTree->Get("MatchPattern",1)); //tototo 

   xml_init(anObj.AutomSel(),aTree->Get("AutomSel",1)); //tototo 

   xml_init(anObj.Result(),aTree->Get("Result",1)); //tototo 
}


std::vector< cOneAutomMapN2N > & cMapN2NByAutom::OneAutomMapN2N()
{
   return mOneAutomMapN2N;
}

const std::vector< cOneAutomMapN2N > & cMapN2NByAutom::OneAutomMapN2N()const 
{
   return mOneAutomMapN2N;
}

cElXMLTree * ToXMLTree(const cMapN2NByAutom & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"MapN2NByAutom",eXMLBranche);
  for
  (       std::vector< cOneAutomMapN2N >::const_iterator it=anObj.OneAutomMapN2N().begin();
      it !=anObj.OneAutomMapN2N().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("OneAutomMapN2N"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cMapN2NByAutom & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.OneAutomMapN2N(),aTree->GetAll("OneAutomMapN2N",false,1));
}


std::string & cMapName2Name::Key()
{
   return MapByKey().Val().Key();
}

const std::string & cMapName2Name::Key()const 
{
   return MapByKey().Val().Key();
}


cTplValGesInit< bool > & cMapName2Name::DefIfFileNotExisting()
{
   return MapByKey().Val().DefIfFileNotExisting();
}

const cTplValGesInit< bool > & cMapName2Name::DefIfFileNotExisting()const 
{
   return MapByKey().Val().DefIfFileNotExisting();
}


cTplValGesInit< cMapByKey > & cMapName2Name::MapByKey()
{
   return mMapByKey;
}

const cTplValGesInit< cMapByKey > & cMapName2Name::MapByKey()const 
{
   return mMapByKey;
}


std::vector< cOneAutomMapN2N > & cMapName2Name::OneAutomMapN2N()
{
   return MapN2NByAutom().Val().OneAutomMapN2N();
}

const std::vector< cOneAutomMapN2N > & cMapName2Name::OneAutomMapN2N()const 
{
   return MapN2NByAutom().Val().OneAutomMapN2N();
}


cTplValGesInit< cMapN2NByAutom > & cMapName2Name::MapN2NByAutom()
{
   return mMapN2NByAutom;
}

const cTplValGesInit< cMapN2NByAutom > & cMapName2Name::MapN2NByAutom()const 
{
   return mMapN2NByAutom;
}

cElXMLTree * ToXMLTree(const cMapName2Name & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"MapName2Name",eXMLBranche);
   if (anObj.MapByKey().IsInit())
      aRes->AddFils(ToXMLTree(anObj.MapByKey().Val())->ReTagThis("MapByKey"));
   if (anObj.MapN2NByAutom().IsInit())
      aRes->AddFils(ToXMLTree(anObj.MapN2NByAutom().Val())->ReTagThis("MapN2NByAutom"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cMapName2Name & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.MapByKey(),aTree->Get("MapByKey",1)); //tototo 

   xml_init(anObj.MapN2NByAutom(),aTree->Get("MapN2NByAutom",1)); //tototo 
}


std::string & cImage_Point3D::Image()
{
   return mImage;
}

const std::string & cImage_Point3D::Image()const 
{
   return mImage;
}


std::string & cImage_Point3D::Masq()
{
   return mMasq;
}

const std::string & cImage_Point3D::Masq()const 
{
   return mMasq;
}

cElXMLTree * ToXMLTree(const cImage_Point3D & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"Image_Point3D",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Image"),anObj.Image())->ReTagThis("Image"));
   aRes->AddFils(::ToXMLTree(std::string("Masq"),anObj.Masq())->ReTagThis("Masq"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cImage_Point3D & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Image(),aTree->Get("Image",1)); //tototo 

   xml_init(anObj.Masq(),aTree->Get("Masq",1)); //tototo 
}


std::string & cImage_Profondeur::Image()
{
   return mImage;
}

const std::string & cImage_Profondeur::Image()const 
{
   return mImage;
}


std::string & cImage_Profondeur::Masq()
{
   return mMasq;
}

const std::string & cImage_Profondeur::Masq()const 
{
   return mMasq;
}


cTplValGesInit< std::string > & cImage_Profondeur::Correl()
{
   return mCorrel;
}

const cTplValGesInit< std::string > & cImage_Profondeur::Correl()const 
{
   return mCorrel;
}


double & cImage_Profondeur::OrigineAlti()
{
   return mOrigineAlti;
}

const double & cImage_Profondeur::OrigineAlti()const 
{
   return mOrigineAlti;
}


double & cImage_Profondeur::ResolutionAlti()
{
   return mResolutionAlti;
}

const double & cImage_Profondeur::ResolutionAlti()const 
{
   return mResolutionAlti;
}


eModeGeomMNT & cImage_Profondeur::GeomRestit()
{
   return mGeomRestit;
}

const eModeGeomMNT & cImage_Profondeur::GeomRestit()const 
{
   return mGeomRestit;
}

cElXMLTree * ToXMLTree(const cImage_Profondeur & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"Image_Profondeur",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Image"),anObj.Image())->ReTagThis("Image"));
   aRes->AddFils(::ToXMLTree(std::string("Masq"),anObj.Masq())->ReTagThis("Masq"));
   if (anObj.Correl().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Correl"),anObj.Correl().Val())->ReTagThis("Correl"));
   aRes->AddFils(::ToXMLTree(std::string("OrigineAlti"),anObj.OrigineAlti())->ReTagThis("OrigineAlti"));
   aRes->AddFils(::ToXMLTree(std::string("ResolutionAlti"),anObj.ResolutionAlti())->ReTagThis("ResolutionAlti"));
   aRes->AddFils(ToXMLTree(std::string("GeomRestit"),anObj.GeomRestit())->ReTagThis("GeomRestit"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cImage_Profondeur & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Image(),aTree->Get("Image",1)); //tototo 

   xml_init(anObj.Masq(),aTree->Get("Masq",1)); //tototo 

   xml_init(anObj.Correl(),aTree->Get("Correl",1)); //tototo 

   xml_init(anObj.OrigineAlti(),aTree->Get("OrigineAlti",1)); //tototo 

   xml_init(anObj.ResolutionAlti(),aTree->Get("ResolutionAlti",1)); //tototo 

   xml_init(anObj.GeomRestit(),aTree->Get("GeomRestit",1)); //tototo 
}


cTplValGesInit< cImage_Point3D > & cPN3M_Nuage::Image_Point3D()
{
   return mImage_Point3D;
}

const cTplValGesInit< cImage_Point3D > & cPN3M_Nuage::Image_Point3D()const 
{
   return mImage_Point3D;
}


cTplValGesInit< cImage_Profondeur > & cPN3M_Nuage::Image_Profondeur()
{
   return mImage_Profondeur;
}

const cTplValGesInit< cImage_Profondeur > & cPN3M_Nuage::Image_Profondeur()const 
{
   return mImage_Profondeur;
}

cElXMLTree * ToXMLTree(const cPN3M_Nuage & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"PN3M_Nuage",eXMLBranche);
   if (anObj.Image_Point3D().IsInit())
      aRes->AddFils(ToXMLTree(anObj.Image_Point3D().Val())->ReTagThis("Image_Point3D"));
   if (anObj.Image_Profondeur().IsInit())
      aRes->AddFils(ToXMLTree(anObj.Image_Profondeur().Val())->ReTagThis("Image_Profondeur"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cPN3M_Nuage & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Image_Point3D(),aTree->Get("Image_Point3D",1)); //tototo 

   xml_init(anObj.Image_Profondeur(),aTree->Get("Image_Profondeur",1)); //tototo 
}


std::string & cAttributsNuage3D::NameFileImage()
{
   return mNameFileImage;
}

const std::string & cAttributsNuage3D::NameFileImage()const 
{
   return mNameFileImage;
}


cTplValGesInit< bool > & cAttributsNuage3D::AddDir2Name()
{
   return mAddDir2Name;
}

const cTplValGesInit< bool > & cAttributsNuage3D::AddDir2Name()const 
{
   return mAddDir2Name;
}


cTplValGesInit< double > & cAttributsNuage3D::Dyn()
{
   return mDyn;
}

const cTplValGesInit< double > & cAttributsNuage3D::Dyn()const 
{
   return mDyn;
}


cTplValGesInit< double > & cAttributsNuage3D::Scale()
{
   return mScale;
}

const cTplValGesInit< double > & cAttributsNuage3D::Scale()const 
{
   return mScale;
}

cElXMLTree * ToXMLTree(const cAttributsNuage3D & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"AttributsNuage3D",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("NameFileImage"),anObj.NameFileImage())->ReTagThis("NameFileImage"));
   if (anObj.AddDir2Name().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("AddDir2Name"),anObj.AddDir2Name().Val())->ReTagThis("AddDir2Name"));
   if (anObj.Dyn().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Dyn"),anObj.Dyn().Val())->ReTagThis("Dyn"));
   if (anObj.Scale().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Scale"),anObj.Scale().Val())->ReTagThis("Scale"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cAttributsNuage3D & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.NameFileImage(),aTree->Get("NameFileImage",1)); //tototo 

   xml_init(anObj.AddDir2Name(),aTree->Get("AddDir2Name",1),bool(true)); //tototo 

   xml_init(anObj.Dyn(),aTree->Get("Dyn",1),double(1.0)); //tototo 

   xml_init(anObj.Scale(),aTree->Get("Scale",1),double(1.0)); //tototo 
}


Pt3dr & cModeFaisceauxImage::DirFaisceaux()
{
   return mDirFaisceaux;
}

const Pt3dr & cModeFaisceauxImage::DirFaisceaux()const 
{
   return mDirFaisceaux;
}


bool & cModeFaisceauxImage::ZIsInverse()
{
   return mZIsInverse;
}

const bool & cModeFaisceauxImage::ZIsInverse()const 
{
   return mZIsInverse;
}


cTplValGesInit< bool > & cModeFaisceauxImage::IsSpherik()
{
   return mIsSpherik;
}

const cTplValGesInit< bool > & cModeFaisceauxImage::IsSpherik()const 
{
   return mIsSpherik;
}


cTplValGesInit< Pt2dr > & cModeFaisceauxImage::DirTrans()
{
   return mDirTrans;
}

const cTplValGesInit< Pt2dr > & cModeFaisceauxImage::DirTrans()const 
{
   return mDirTrans;
}

cElXMLTree * ToXMLTree(const cModeFaisceauxImage & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ModeFaisceauxImage",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("DirFaisceaux"),anObj.DirFaisceaux())->ReTagThis("DirFaisceaux"));
   aRes->AddFils(::ToXMLTree(std::string("ZIsInverse"),anObj.ZIsInverse())->ReTagThis("ZIsInverse"));
   if (anObj.IsSpherik().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("IsSpherik"),anObj.IsSpherik().Val())->ReTagThis("IsSpherik"));
   if (anObj.DirTrans().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DirTrans"),anObj.DirTrans().Val())->ReTagThis("DirTrans"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cModeFaisceauxImage & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.DirFaisceaux(),aTree->Get("DirFaisceaux",1)); //tototo 

   xml_init(anObj.ZIsInverse(),aTree->Get("ZIsInverse",1)); //tototo 

   xml_init(anObj.IsSpherik(),aTree->Get("IsSpherik",1),bool(false)); //tototo 

   xml_init(anObj.DirTrans(),aTree->Get("DirTrans",1)); //tototo 
}


Pt3dr & cPM3D_ParamSpecifs::DirFaisceaux()
{
   return ModeFaisceauxImage().Val().DirFaisceaux();
}

const Pt3dr & cPM3D_ParamSpecifs::DirFaisceaux()const 
{
   return ModeFaisceauxImage().Val().DirFaisceaux();
}


bool & cPM3D_ParamSpecifs::ZIsInverse()
{
   return ModeFaisceauxImage().Val().ZIsInverse();
}

const bool & cPM3D_ParamSpecifs::ZIsInverse()const 
{
   return ModeFaisceauxImage().Val().ZIsInverse();
}


cTplValGesInit< bool > & cPM3D_ParamSpecifs::IsSpherik()
{
   return ModeFaisceauxImage().Val().IsSpherik();
}

const cTplValGesInit< bool > & cPM3D_ParamSpecifs::IsSpherik()const 
{
   return ModeFaisceauxImage().Val().IsSpherik();
}


cTplValGesInit< Pt2dr > & cPM3D_ParamSpecifs::DirTrans()
{
   return ModeFaisceauxImage().Val().DirTrans();
}

const cTplValGesInit< Pt2dr > & cPM3D_ParamSpecifs::DirTrans()const 
{
   return ModeFaisceauxImage().Val().DirTrans();
}


cTplValGesInit< cModeFaisceauxImage > & cPM3D_ParamSpecifs::ModeFaisceauxImage()
{
   return mModeFaisceauxImage;
}

const cTplValGesInit< cModeFaisceauxImage > & cPM3D_ParamSpecifs::ModeFaisceauxImage()const 
{
   return mModeFaisceauxImage;
}


cTplValGesInit< std::string > & cPM3D_ParamSpecifs::NoParamSpecif()
{
   return mNoParamSpecif;
}

const cTplValGesInit< std::string > & cPM3D_ParamSpecifs::NoParamSpecif()const 
{
   return mNoParamSpecif;
}

cElXMLTree * ToXMLTree(const cPM3D_ParamSpecifs & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"PM3D_ParamSpecifs",eXMLBranche);
   if (anObj.ModeFaisceauxImage().IsInit())
      aRes->AddFils(ToXMLTree(anObj.ModeFaisceauxImage().Val())->ReTagThis("ModeFaisceauxImage"));
   if (anObj.NoParamSpecif().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NoParamSpecif"),anObj.NoParamSpecif().Val())->ReTagThis("NoParamSpecif"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cPM3D_ParamSpecifs & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.ModeFaisceauxImage(),aTree->Get("ModeFaisceauxImage",1)); //tototo 

   xml_init(anObj.NoParamSpecif(),aTree->Get("NoParamSpecif",1)); //tototo 
}


Pt2dr & cVerifNuage::IndIm()
{
   return mIndIm;
}

const Pt2dr & cVerifNuage::IndIm()const 
{
   return mIndIm;
}


double & cVerifNuage::Profondeur()
{
   return mProfondeur;
}

const double & cVerifNuage::Profondeur()const 
{
   return mProfondeur;
}


Pt3dr & cVerifNuage::PointEuclid()
{
   return mPointEuclid;
}

const Pt3dr & cVerifNuage::PointEuclid()const 
{
   return mPointEuclid;
}

cElXMLTree * ToXMLTree(const cVerifNuage & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"VerifNuage",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("IndIm"),anObj.IndIm())->ReTagThis("IndIm"));
   aRes->AddFils(::ToXMLTree(std::string("Profondeur"),anObj.Profondeur())->ReTagThis("Profondeur"));
   aRes->AddFils(::ToXMLTree(std::string("PointEuclid"),anObj.PointEuclid())->ReTagThis("PointEuclid"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cVerifNuage & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.IndIm(),aTree->Get("IndIm",1)); //tototo 

   xml_init(anObj.Profondeur(),aTree->Get("Profondeur",1)); //tototo 

   xml_init(anObj.PointEuclid(),aTree->Get("PointEuclid",1)); //tototo 
}


cTplValGesInit< double > & cXML_ParamNuage3DMaille::SsResolRef()
{
   return mSsResolRef;
}

const cTplValGesInit< double > & cXML_ParamNuage3DMaille::SsResolRef()const 
{
   return mSsResolRef;
}


Pt2di & cXML_ParamNuage3DMaille::NbPixel()
{
   return mNbPixel;
}

const Pt2di & cXML_ParamNuage3DMaille::NbPixel()const 
{
   return mNbPixel;
}


cTplValGesInit< cImage_Point3D > & cXML_ParamNuage3DMaille::Image_Point3D()
{
   return PN3M_Nuage().Image_Point3D();
}

const cTplValGesInit< cImage_Point3D > & cXML_ParamNuage3DMaille::Image_Point3D()const 
{
   return PN3M_Nuage().Image_Point3D();
}


cTplValGesInit< cImage_Profondeur > & cXML_ParamNuage3DMaille::Image_Profondeur()
{
   return PN3M_Nuage().Image_Profondeur();
}

const cTplValGesInit< cImage_Profondeur > & cXML_ParamNuage3DMaille::Image_Profondeur()const 
{
   return PN3M_Nuage().Image_Profondeur();
}


cPN3M_Nuage & cXML_ParamNuage3DMaille::PN3M_Nuage()
{
   return mPN3M_Nuage;
}

const cPN3M_Nuage & cXML_ParamNuage3DMaille::PN3M_Nuage()const 
{
   return mPN3M_Nuage;
}


std::list< cAttributsNuage3D > & cXML_ParamNuage3DMaille::AttributsNuage3D()
{
   return mAttributsNuage3D;
}

const std::list< cAttributsNuage3D > & cXML_ParamNuage3DMaille::AttributsNuage3D()const 
{
   return mAttributsNuage3D;
}


cTplValGesInit< cRepereCartesien > & cXML_ParamNuage3DMaille::RepereGlob()
{
   return mRepereGlob;
}

const cTplValGesInit< cRepereCartesien > & cXML_ParamNuage3DMaille::RepereGlob()const 
{
   return mRepereGlob;
}


cTplValGesInit< cXmlOneSurfaceAnalytique > & cXML_ParamNuage3DMaille::Anam()
{
   return mAnam;
}

const cTplValGesInit< cXmlOneSurfaceAnalytique > & cXML_ParamNuage3DMaille::Anam()const 
{
   return mAnam;
}


cOrientationConique & cXML_ParamNuage3DMaille::Orientation()
{
   return mOrientation;
}

const cOrientationConique & cXML_ParamNuage3DMaille::Orientation()const 
{
   return mOrientation;
}


Pt3dr & cXML_ParamNuage3DMaille::DirFaisceaux()
{
   return PM3D_ParamSpecifs().ModeFaisceauxImage().Val().DirFaisceaux();
}

const Pt3dr & cXML_ParamNuage3DMaille::DirFaisceaux()const 
{
   return PM3D_ParamSpecifs().ModeFaisceauxImage().Val().DirFaisceaux();
}


bool & cXML_ParamNuage3DMaille::ZIsInverse()
{
   return PM3D_ParamSpecifs().ModeFaisceauxImage().Val().ZIsInverse();
}

const bool & cXML_ParamNuage3DMaille::ZIsInverse()const 
{
   return PM3D_ParamSpecifs().ModeFaisceauxImage().Val().ZIsInverse();
}


cTplValGesInit< bool > & cXML_ParamNuage3DMaille::IsSpherik()
{
   return PM3D_ParamSpecifs().ModeFaisceauxImage().Val().IsSpherik();
}

const cTplValGesInit< bool > & cXML_ParamNuage3DMaille::IsSpherik()const 
{
   return PM3D_ParamSpecifs().ModeFaisceauxImage().Val().IsSpherik();
}


cTplValGesInit< Pt2dr > & cXML_ParamNuage3DMaille::DirTrans()
{
   return PM3D_ParamSpecifs().ModeFaisceauxImage().Val().DirTrans();
}

const cTplValGesInit< Pt2dr > & cXML_ParamNuage3DMaille::DirTrans()const 
{
   return PM3D_ParamSpecifs().ModeFaisceauxImage().Val().DirTrans();
}


cTplValGesInit< cModeFaisceauxImage > & cXML_ParamNuage3DMaille::ModeFaisceauxImage()
{
   return PM3D_ParamSpecifs().ModeFaisceauxImage();
}

const cTplValGesInit< cModeFaisceauxImage > & cXML_ParamNuage3DMaille::ModeFaisceauxImage()const 
{
   return PM3D_ParamSpecifs().ModeFaisceauxImage();
}


cTplValGesInit< std::string > & cXML_ParamNuage3DMaille::NoParamSpecif()
{
   return PM3D_ParamSpecifs().NoParamSpecif();
}

const cTplValGesInit< std::string > & cXML_ParamNuage3DMaille::NoParamSpecif()const 
{
   return PM3D_ParamSpecifs().NoParamSpecif();
}


cPM3D_ParamSpecifs & cXML_ParamNuage3DMaille::PM3D_ParamSpecifs()
{
   return mPM3D_ParamSpecifs;
}

const cPM3D_ParamSpecifs & cXML_ParamNuage3DMaille::PM3D_ParamSpecifs()const 
{
   return mPM3D_ParamSpecifs;
}


cTplValGesInit< double > & cXML_ParamNuage3DMaille::TolVerifNuage()
{
   return mTolVerifNuage;
}

const cTplValGesInit< double > & cXML_ParamNuage3DMaille::TolVerifNuage()const 
{
   return mTolVerifNuage;
}


std::list< cVerifNuage > & cXML_ParamNuage3DMaille::VerifNuage()
{
   return mVerifNuage;
}

const std::list< cVerifNuage > & cXML_ParamNuage3DMaille::VerifNuage()const 
{
   return mVerifNuage;
}

cElXMLTree * ToXMLTree(const cXML_ParamNuage3DMaille & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"XML_ParamNuage3DMaille",eXMLBranche);
   if (anObj.SsResolRef().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SsResolRef"),anObj.SsResolRef().Val())->ReTagThis("SsResolRef"));
   aRes->AddFils(::ToXMLTree(std::string("NbPixel"),anObj.NbPixel())->ReTagThis("NbPixel"));
   aRes->AddFils(ToXMLTree(anObj.PN3M_Nuage())->ReTagThis("PN3M_Nuage"));
  for
  (       std::list< cAttributsNuage3D >::const_iterator it=anObj.AttributsNuage3D().begin();
      it !=anObj.AttributsNuage3D().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("AttributsNuage3D"));
   if (anObj.RepereGlob().IsInit())
      aRes->AddFils(ToXMLTree(anObj.RepereGlob().Val())->ReTagThis("RepereGlob"));
   if (anObj.Anam().IsInit())
      aRes->AddFils(ToXMLTree(anObj.Anam().Val())->ReTagThis("Anam"));
   aRes->AddFils(ToXMLTree(anObj.Orientation())->ReTagThis("Orientation"));
   aRes->AddFils(ToXMLTree(anObj.PM3D_ParamSpecifs())->ReTagThis("PM3D_ParamSpecifs"));
   if (anObj.TolVerifNuage().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("TolVerifNuage"),anObj.TolVerifNuage().Val())->ReTagThis("TolVerifNuage"));
  for
  (       std::list< cVerifNuage >::const_iterator it=anObj.VerifNuage().begin();
      it !=anObj.VerifNuage().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("VerifNuage"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cXML_ParamNuage3DMaille & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.SsResolRef(),aTree->Get("SsResolRef",1),double(1.0)); //tototo 

   xml_init(anObj.NbPixel(),aTree->Get("NbPixel",1)); //tototo 

   xml_init(anObj.PN3M_Nuage(),aTree->Get("PN3M_Nuage",1)); //tototo 

   xml_init(anObj.AttributsNuage3D(),aTree->GetAll("AttributsNuage3D",false,1));

   xml_init(anObj.RepereGlob(),aTree->Get("RepereGlob",1)); //tototo 

   xml_init(anObj.Anam(),aTree->Get("Anam",1)); //tototo 

   xml_init(anObj.Orientation(),aTree->Get("Orientation",1)); //tototo 

   xml_init(anObj.PM3D_ParamSpecifs(),aTree->Get("PM3D_ParamSpecifs",1)); //tototo 

   xml_init(anObj.TolVerifNuage(),aTree->Get("TolVerifNuage",1),double(1e-3)); //tototo 

   xml_init(anObj.VerifNuage(),aTree->GetAll("VerifNuage",false,1));
}


std::string & cMasqMesures::NameFile()
{
   return mNameFile;
}

const std::string & cMasqMesures::NameFile()const 
{
   return mNameFile;
}


std::string & cMasqMesures::NameMTD()
{
   return mNameMTD;
}

const std::string & cMasqMesures::NameMTD()const 
{
   return mNameMTD;
}

cElXMLTree * ToXMLTree(const cMasqMesures & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"MasqMesures",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("NameFile"),anObj.NameFile())->ReTagThis("NameFile"));
   aRes->AddFils(::ToXMLTree(std::string("NameMTD"),anObj.NameMTD())->ReTagThis("NameMTD"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cMasqMesures & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.NameFile(),aTree->Get("NameFile",1)); //tototo 

   xml_init(anObj.NameMTD(),aTree->Get("NameMTD",1)); //tototo 
}


cTplValGesInit< std::string > & cCielVisible::UnUsed()
{
   return mUnUsed;
}

const cTplValGesInit< std::string > & cCielVisible::UnUsed()const 
{
   return mUnUsed;
}

cElXMLTree * ToXMLTree(const cCielVisible & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"CielVisible",eXMLBranche);
   if (anObj.UnUsed().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("UnUsed"),anObj.UnUsed().Val())->ReTagThis("UnUsed"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cCielVisible & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.UnUsed(),aTree->Get("UnUsed",1)); //tototo 
}


cTplValGesInit< int > & cXML_ParamOmbrageNuage::ScaleMaxPyr()
{
   return mScaleMaxPyr;
}

const cTplValGesInit< int > & cXML_ParamOmbrageNuage::ScaleMaxPyr()const 
{
   return mScaleMaxPyr;
}


cTplValGesInit< double > & cXML_ParamOmbrageNuage::StepScale()
{
   return mStepScale;
}

const cTplValGesInit< double > & cXML_ParamOmbrageNuage::StepScale()const 
{
   return mStepScale;
}


cTplValGesInit< double > & cXML_ParamOmbrageNuage::RatioOct()
{
   return mRatioOct;
}

const cTplValGesInit< double > & cXML_ParamOmbrageNuage::RatioOct()const 
{
   return mRatioOct;
}


cTplValGesInit< std::string > & cXML_ParamOmbrageNuage::UnUsed()
{
   return CielVisible().Val().UnUsed();
}

const cTplValGesInit< std::string > & cXML_ParamOmbrageNuage::UnUsed()const 
{
   return CielVisible().Val().UnUsed();
}


cTplValGesInit< cCielVisible > & cXML_ParamOmbrageNuage::CielVisible()
{
   return mCielVisible;
}

const cTplValGesInit< cCielVisible > & cXML_ParamOmbrageNuage::CielVisible()const 
{
   return mCielVisible;
}

cElXMLTree * ToXMLTree(const cXML_ParamOmbrageNuage & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"XML_ParamOmbrageNuage",eXMLBranche);
   if (anObj.ScaleMaxPyr().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ScaleMaxPyr"),anObj.ScaleMaxPyr().Val())->ReTagThis("ScaleMaxPyr"));
   if (anObj.StepScale().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("StepScale"),anObj.StepScale().Val())->ReTagThis("StepScale"));
   if (anObj.RatioOct().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("RatioOct"),anObj.RatioOct().Val())->ReTagThis("RatioOct"));
   if (anObj.CielVisible().IsInit())
      aRes->AddFils(ToXMLTree(anObj.CielVisible().Val())->ReTagThis("CielVisible"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cXML_ParamOmbrageNuage & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.ScaleMaxPyr(),aTree->Get("ScaleMaxPyr",1),int(128)); //tototo 

   xml_init(anObj.StepScale(),aTree->Get("StepScale",1),double(1.414)); //tototo 

   xml_init(anObj.RatioOct(),aTree->Get("RatioOct",1),double(2.0)); //tototo 

   xml_init(anObj.CielVisible(),aTree->Get("CielVisible",1)); //tototo 
}


double & cFTrajParamInit2Actuelle::Lambda()
{
   return mLambda;
}

const double & cFTrajParamInit2Actuelle::Lambda()const 
{
   return mLambda;
}


cOrientationExterneRigide & cFTrajParamInit2Actuelle::Orient()
{
   return mOrient;
}

const cOrientationExterneRigide & cFTrajParamInit2Actuelle::Orient()const 
{
   return mOrient;
}

cElXMLTree * ToXMLTree(const cFTrajParamInit2Actuelle & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"FTrajParamInit2Actuelle",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Lambda"),anObj.Lambda())->ReTagThis("Lambda"));
   aRes->AddFils(ToXMLTree(anObj.Orient())->ReTagThis("Orient"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cFTrajParamInit2Actuelle & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Lambda(),aTree->Get("Lambda",1)); //tototo 

   xml_init(anObj.Orient(),aTree->Get("Orient",1)); //tototo 
}


Pt3dr & cPtTrajecto::Pt()
{
   return mPt;
}

const Pt3dr & cPtTrajecto::Pt()const 
{
   return mPt;
}


std::string & cPtTrajecto::IdImage()
{
   return mIdImage;
}

const std::string & cPtTrajecto::IdImage()const 
{
   return mIdImage;
}


std::string & cPtTrajecto::IdBande()
{
   return mIdBande;
}

const std::string & cPtTrajecto::IdBande()const 
{
   return mIdBande;
}


double & cPtTrajecto::Time()
{
   return mTime;
}

const double & cPtTrajecto::Time()const 
{
   return mTime;
}

cElXMLTree * ToXMLTree(const cPtTrajecto & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"PtTrajecto",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Pt"),anObj.Pt())->ReTagThis("Pt"));
   aRes->AddFils(::ToXMLTree(std::string("IdImage"),anObj.IdImage())->ReTagThis("IdImage"));
   aRes->AddFils(::ToXMLTree(std::string("IdBande"),anObj.IdBande())->ReTagThis("IdBande"));
   aRes->AddFils(::ToXMLTree(std::string("Time"),anObj.Time())->ReTagThis("Time"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cPtTrajecto & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Pt(),aTree->Get("Pt",1)); //tototo 

   xml_init(anObj.IdImage(),aTree->Get("IdImage",1)); //tototo 

   xml_init(anObj.IdBande(),aTree->Get("IdBande",1)); //tototo 

   xml_init(anObj.Time(),aTree->Get("Time",1)); //tototo 
}


std::string & cFichier_Trajecto::NameInit()
{
   return mNameInit;
}

const std::string & cFichier_Trajecto::NameInit()const 
{
   return mNameInit;
}


double & cFichier_Trajecto::Lambda()
{
   return FTrajParamInit2Actuelle().Lambda();
}

const double & cFichier_Trajecto::Lambda()const 
{
   return FTrajParamInit2Actuelle().Lambda();
}


cOrientationExterneRigide & cFichier_Trajecto::Orient()
{
   return FTrajParamInit2Actuelle().Orient();
}

const cOrientationExterneRigide & cFichier_Trajecto::Orient()const 
{
   return FTrajParamInit2Actuelle().Orient();
}


cFTrajParamInit2Actuelle & cFichier_Trajecto::FTrajParamInit2Actuelle()
{
   return mFTrajParamInit2Actuelle;
}

const cFTrajParamInit2Actuelle & cFichier_Trajecto::FTrajParamInit2Actuelle()const 
{
   return mFTrajParamInit2Actuelle;
}


std::map< std::string,cPtTrajecto > & cFichier_Trajecto::PtTrajecto()
{
   return mPtTrajecto;
}

const std::map< std::string,cPtTrajecto > & cFichier_Trajecto::PtTrajecto()const 
{
   return mPtTrajecto;
}

cElXMLTree * ToXMLTree(const cFichier_Trajecto & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"Fichier_Trajecto",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("NameInit"),anObj.NameInit())->ReTagThis("NameInit"));
   aRes->AddFils(ToXMLTree(anObj.FTrajParamInit2Actuelle())->ReTagThis("FTrajParamInit2Actuelle"));
  for
  (       std::map< std::string,cPtTrajecto >::const_iterator it=anObj.PtTrajecto().begin();
      it !=anObj.PtTrajecto().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it).second)->ReTagThis("PtTrajecto"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cFichier_Trajecto & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.NameInit(),aTree->Get("NameInit",1)); //tototo 

   xml_init(anObj.FTrajParamInit2Actuelle(),aTree->Get("FTrajParamInit2Actuelle",1)); //tototo 

   xml_init(anObj.PtTrajecto(),aTree->GetAll("PtTrajecto",false,1),"IdImage");
}


cTplValGesInit< std::string > & cSectionEntree::FileMNT()
{
   return mFileMNT;
}

const cTplValGesInit< std::string > & cSectionEntree::FileMNT()const 
{
   return mFileMNT;
}


std::string & cSectionEntree::KeySetIm()
{
   return mKeySetIm;
}

const std::string & cSectionEntree::KeySetIm()const 
{
   return mKeySetIm;
}


std::string & cSectionEntree::KeyAssocMetaData()
{
   return mKeyAssocMetaData;
}

const std::string & cSectionEntree::KeyAssocMetaData()const 
{
   return mKeyAssocMetaData;
}


std::string & cSectionEntree::KeyAssocNamePC()
{
   return mKeyAssocNamePC;
}

const std::string & cSectionEntree::KeyAssocNamePC()const 
{
   return mKeyAssocNamePC;
}


std::string & cSectionEntree::KeyAssocNameIncH()
{
   return mKeyAssocNameIncH;
}

const std::string & cSectionEntree::KeyAssocNameIncH()const 
{
   return mKeyAssocNameIncH;
}


cTplValGesInit< std::string > & cSectionEntree::KeyAssocPriorite()
{
   return mKeyAssocPriorite;
}

const cTplValGesInit< std::string > & cSectionEntree::KeyAssocPriorite()const 
{
   return mKeyAssocPriorite;
}


std::list< cMasqMesures > & cSectionEntree::ListMasqMesures()
{
   return mListMasqMesures;
}

const std::list< cMasqMesures > & cSectionEntree::ListMasqMesures()const 
{
   return mListMasqMesures;
}


std::list< std::string > & cSectionEntree::FileExterneMasqMesures()
{
   return mFileExterneMasqMesures;
}

const std::list< std::string > & cSectionEntree::FileExterneMasqMesures()const 
{
   return mFileExterneMasqMesures;
}

cElXMLTree * ToXMLTree(const cSectionEntree & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"SectionEntree",eXMLBranche);
   if (anObj.FileMNT().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("FileMNT"),anObj.FileMNT().Val())->ReTagThis("FileMNT"));
   aRes->AddFils(::ToXMLTree(std::string("KeySetIm"),anObj.KeySetIm())->ReTagThis("KeySetIm"));
   aRes->AddFils(::ToXMLTree(std::string("KeyAssocMetaData"),anObj.KeyAssocMetaData())->ReTagThis("KeyAssocMetaData"));
   aRes->AddFils(::ToXMLTree(std::string("KeyAssocNamePC"),anObj.KeyAssocNamePC())->ReTagThis("KeyAssocNamePC"));
   aRes->AddFils(::ToXMLTree(std::string("KeyAssocNameIncH"),anObj.KeyAssocNameIncH())->ReTagThis("KeyAssocNameIncH"));
   if (anObj.KeyAssocPriorite().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("KeyAssocPriorite"),anObj.KeyAssocPriorite().Val())->ReTagThis("KeyAssocPriorite"));
  for
  (       std::list< cMasqMesures >::const_iterator it=anObj.ListMasqMesures().begin();
      it !=anObj.ListMasqMesures().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("ListMasqMesures"));
  for
  (       std::list< std::string >::const_iterator it=anObj.FileExterneMasqMesures().begin();
      it !=anObj.FileExterneMasqMesures().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("FileExterneMasqMesures"),(*it))->ReTagThis("FileExterneMasqMesures"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cSectionEntree & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.FileMNT(),aTree->Get("FileMNT",1)); //tototo 

   xml_init(anObj.KeySetIm(),aTree->Get("KeySetIm",1)); //tototo 

   xml_init(anObj.KeyAssocMetaData(),aTree->Get("KeyAssocMetaData",1)); //tototo 

   xml_init(anObj.KeyAssocNamePC(),aTree->Get("KeyAssocNamePC",1)); //tototo 

   xml_init(anObj.KeyAssocNameIncH(),aTree->Get("KeyAssocNameIncH",1)); //tototo 

   xml_init(anObj.KeyAssocPriorite(),aTree->Get("KeyAssocPriorite",1),std::string("Key-Priorite-Ortho")); //tototo 

   xml_init(anObj.ListMasqMesures(),aTree->GetAll("ListMasqMesures",false,1));

   xml_init(anObj.FileExterneMasqMesures(),aTree->GetAll("FileExterneMasqMesures",false,1));
}


cTplValGesInit< int > & cBoucheTrou::SeuilVisib()
{
   return mSeuilVisib;
}

const cTplValGesInit< int > & cBoucheTrou::SeuilVisib()const 
{
   return mSeuilVisib;
}


cTplValGesInit< int > & cBoucheTrou::SeuilVisibBT()
{
   return mSeuilVisibBT;
}

const cTplValGesInit< int > & cBoucheTrou::SeuilVisibBT()const 
{
   return mSeuilVisibBT;
}


cTplValGesInit< double > & cBoucheTrou::CoeffPondAngul()
{
   return mCoeffPondAngul;
}

const cTplValGesInit< double > & cBoucheTrou::CoeffPondAngul()const 
{
   return mCoeffPondAngul;
}

cElXMLTree * ToXMLTree(const cBoucheTrou & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"BoucheTrou",eXMLBranche);
   if (anObj.SeuilVisib().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SeuilVisib"),anObj.SeuilVisib().Val())->ReTagThis("SeuilVisib"));
   if (anObj.SeuilVisibBT().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SeuilVisibBT"),anObj.SeuilVisibBT().Val())->ReTagThis("SeuilVisibBT"));
   if (anObj.CoeffPondAngul().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("CoeffPondAngul"),anObj.CoeffPondAngul().Val())->ReTagThis("CoeffPondAngul"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cBoucheTrou & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.SeuilVisib(),aTree->Get("SeuilVisib",1),int(10)); //tototo 

   xml_init(anObj.SeuilVisibBT(),aTree->Get("SeuilVisibBT",1),int(3)); //tototo 

   xml_init(anObj.CoeffPondAngul(),aTree->Get("CoeffPondAngul",1),double(1.57)); //tototo 
}


cTplValGesInit< double > & cSectionFiltrageIn::SaturThreshold()
{
   return mSaturThreshold;
}

const cTplValGesInit< double > & cSectionFiltrageIn::SaturThreshold()const 
{
   return mSaturThreshold;
}


cTplValGesInit< int > & cSectionFiltrageIn::SzDilatPC()
{
   return mSzDilatPC;
}

const cTplValGesInit< int > & cSectionFiltrageIn::SzDilatPC()const 
{
   return mSzDilatPC;
}


cTplValGesInit< int > & cSectionFiltrageIn::SzOuvPC()
{
   return mSzOuvPC;
}

const cTplValGesInit< int > & cSectionFiltrageIn::SzOuvPC()const 
{
   return mSzOuvPC;
}


cTplValGesInit< int > & cSectionFiltrageIn::SeuilVisib()
{
   return BoucheTrou().Val().SeuilVisib();
}

const cTplValGesInit< int > & cSectionFiltrageIn::SeuilVisib()const 
{
   return BoucheTrou().Val().SeuilVisib();
}


cTplValGesInit< int > & cSectionFiltrageIn::SeuilVisibBT()
{
   return BoucheTrou().Val().SeuilVisibBT();
}

const cTplValGesInit< int > & cSectionFiltrageIn::SeuilVisibBT()const 
{
   return BoucheTrou().Val().SeuilVisibBT();
}


cTplValGesInit< double > & cSectionFiltrageIn::CoeffPondAngul()
{
   return BoucheTrou().Val().CoeffPondAngul();
}

const cTplValGesInit< double > & cSectionFiltrageIn::CoeffPondAngul()const 
{
   return BoucheTrou().Val().CoeffPondAngul();
}


cTplValGesInit< cBoucheTrou > & cSectionFiltrageIn::BoucheTrou()
{
   return mBoucheTrou;
}

const cTplValGesInit< cBoucheTrou > & cSectionFiltrageIn::BoucheTrou()const 
{
   return mBoucheTrou;
}

cElXMLTree * ToXMLTree(const cSectionFiltrageIn & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"SectionFiltrageIn",eXMLBranche);
   if (anObj.SaturThreshold().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SaturThreshold"),anObj.SaturThreshold().Val())->ReTagThis("SaturThreshold"));
   if (anObj.SzDilatPC().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SzDilatPC"),anObj.SzDilatPC().Val())->ReTagThis("SzDilatPC"));
   if (anObj.SzOuvPC().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SzOuvPC"),anObj.SzOuvPC().Val())->ReTagThis("SzOuvPC"));
   if (anObj.BoucheTrou().IsInit())
      aRes->AddFils(ToXMLTree(anObj.BoucheTrou().Val())->ReTagThis("BoucheTrou"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cSectionFiltrageIn & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.SaturThreshold(),aTree->Get("SaturThreshold",1)); //tototo 

   xml_init(anObj.SzDilatPC(),aTree->Get("SzDilatPC",1),int(1)); //tototo 

   xml_init(anObj.SzOuvPC(),aTree->Get("SzOuvPC",1),int(2)); //tototo 

   xml_init(anObj.BoucheTrou(),aTree->Get("BoucheTrou",1)); //tototo 
}


cTplValGesInit< bool > & cSectionSorties::TestDiff()
{
   return mTestDiff;
}

const cTplValGesInit< bool > & cSectionSorties::TestDiff()const 
{
   return mTestDiff;
}


std::string & cSectionSorties::NameOrtho()
{
   return mNameOrtho;
}

const std::string & cSectionSorties::NameOrtho()const 
{
   return mNameOrtho;
}


cTplValGesInit< std::string > & cSectionSorties::NameLabels()
{
   return mNameLabels;
}

const cTplValGesInit< std::string > & cSectionSorties::NameLabels()const 
{
   return mNameLabels;
}


cTplValGesInit< Box2di > & cSectionSorties::BoxCalc()
{
   return mBoxCalc;
}

const cTplValGesInit< Box2di > & cSectionSorties::BoxCalc()const 
{
   return mBoxCalc;
}


int & cSectionSorties::SzDalle()
{
   return mSzDalle;
}

const int & cSectionSorties::SzDalle()const 
{
   return mSzDalle;
}


int & cSectionSorties::SzBrd()
{
   return mSzBrd;
}

const int & cSectionSorties::SzBrd()const 
{
   return mSzBrd;
}


cTplValGesInit< int > & cSectionSorties::SzTileResult()
{
   return mSzTileResult;
}

const cTplValGesInit< int > & cSectionSorties::SzTileResult()const 
{
   return mSzTileResult;
}


cTplValGesInit< bool > & cSectionSorties::Show()
{
   return mShow;
}

const cTplValGesInit< bool > & cSectionSorties::Show()const 
{
   return mShow;
}


cTplValGesInit< double > & cSectionSorties::DynGlob()
{
   return mDynGlob;
}

const cTplValGesInit< double > & cSectionSorties::DynGlob()const 
{
   return mDynGlob;
}

cElXMLTree * ToXMLTree(const cSectionSorties & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"SectionSorties",eXMLBranche);
   if (anObj.TestDiff().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("TestDiff"),anObj.TestDiff().Val())->ReTagThis("TestDiff"));
   aRes->AddFils(::ToXMLTree(std::string("NameOrtho"),anObj.NameOrtho())->ReTagThis("NameOrtho"));
   if (anObj.NameLabels().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NameLabels"),anObj.NameLabels().Val())->ReTagThis("NameLabels"));
   if (anObj.BoxCalc().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("BoxCalc"),anObj.BoxCalc().Val())->ReTagThis("BoxCalc"));
   aRes->AddFils(::ToXMLTree(std::string("SzDalle"),anObj.SzDalle())->ReTagThis("SzDalle"));
   aRes->AddFils(::ToXMLTree(std::string("SzBrd"),anObj.SzBrd())->ReTagThis("SzBrd"));
   if (anObj.SzTileResult().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SzTileResult"),anObj.SzTileResult().Val())->ReTagThis("SzTileResult"));
   if (anObj.Show().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Show"),anObj.Show().Val())->ReTagThis("Show"));
   if (anObj.DynGlob().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DynGlob"),anObj.DynGlob().Val())->ReTagThis("DynGlob"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cSectionSorties & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.TestDiff(),aTree->Get("TestDiff",1),bool(false)); //tototo 

   xml_init(anObj.NameOrtho(),aTree->Get("NameOrtho",1)); //tototo 

   xml_init(anObj.NameLabels(),aTree->Get("NameLabels",1)); //tototo 

   xml_init(anObj.BoxCalc(),aTree->Get("BoxCalc",1)); //tototo 

   xml_init(anObj.SzDalle(),aTree->Get("SzDalle",1)); //tototo 

   xml_init(anObj.SzBrd(),aTree->Get("SzBrd",1)); //tototo 

   xml_init(anObj.SzTileResult(),aTree->Get("SzTileResult",1),int(17000)); //tototo 

   xml_init(anObj.Show(),aTree->Get("Show",1),bool(false)); //tototo 

   xml_init(anObj.DynGlob(),aTree->Get("DynGlob",1),double(1.0)); //tototo 
}


double & cNoiseSSI::Ampl()
{
   return mAmpl;
}

const double & cNoiseSSI::Ampl()const 
{
   return mAmpl;
}


bool & cNoiseSSI::Unif()
{
   return mUnif;
}

const bool & cNoiseSSI::Unif()const 
{
   return mUnif;
}


int & cNoiseSSI::Iter()
{
   return mIter;
}

const int & cNoiseSSI::Iter()const 
{
   return mIter;
}


int & cNoiseSSI::Sz()
{
   return mSz;
}

const int & cNoiseSSI::Sz()const 
{
   return mSz;
}

cElXMLTree * ToXMLTree(const cNoiseSSI & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"NoiseSSI",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Ampl"),anObj.Ampl())->ReTagThis("Ampl"));
   aRes->AddFils(::ToXMLTree(std::string("Unif"),anObj.Unif())->ReTagThis("Unif"));
   aRes->AddFils(::ToXMLTree(std::string("Iter"),anObj.Iter())->ReTagThis("Iter"));
   aRes->AddFils(::ToXMLTree(std::string("Sz"),anObj.Sz())->ReTagThis("Sz"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cNoiseSSI & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Ampl(),aTree->Get("Ampl",1)); //tototo 

   xml_init(anObj.Unif(),aTree->Get("Unif",1)); //tototo 

   xml_init(anObj.Iter(),aTree->Get("Iter",1)); //tototo 

   xml_init(anObj.Sz(),aTree->Get("Sz",1)); //tototo 
}


Pt2dr & cSectionSimulImage::Per1()
{
   return mPer1;
}

const Pt2dr & cSectionSimulImage::Per1()const 
{
   return mPer1;
}


cTplValGesInit< Pt2dr > & cSectionSimulImage::Per2()
{
   return mPer2;
}

const cTplValGesInit< Pt2dr > & cSectionSimulImage::Per2()const 
{
   return mPer2;
}


cTplValGesInit< double > & cSectionSimulImage::Ampl()
{
   return mAmpl;
}

const cTplValGesInit< double > & cSectionSimulImage::Ampl()const 
{
   return mAmpl;
}


std::list< cNoiseSSI > & cSectionSimulImage::NoiseSSI()
{
   return mNoiseSSI;
}

const std::list< cNoiseSSI > & cSectionSimulImage::NoiseSSI()const 
{
   return mNoiseSSI;
}

cElXMLTree * ToXMLTree(const cSectionSimulImage & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"SectionSimulImage",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Per1"),anObj.Per1())->ReTagThis("Per1"));
   if (anObj.Per2().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Per2"),anObj.Per2().Val())->ReTagThis("Per2"));
   if (anObj.Ampl().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Ampl"),anObj.Ampl().Val())->ReTagThis("Ampl"));
  for
  (       std::list< cNoiseSSI >::const_iterator it=anObj.NoiseSSI().begin();
      it !=anObj.NoiseSSI().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("NoiseSSI"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cSectionSimulImage & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Per1(),aTree->Get("Per1",1)); //tototo 

   xml_init(anObj.Per2(),aTree->Get("Per2",1)); //tototo 

   xml_init(anObj.Ampl(),aTree->Get("Ampl",1),double(1.0)); //tototo 

   xml_init(anObj.NoiseSSI(),aTree->GetAll("NoiseSSI",false,1));
}


cTplValGesInit< bool > & cGlobRappInit::DoGlob()
{
   return mDoGlob;
}

const cTplValGesInit< bool > & cGlobRappInit::DoGlob()const 
{
   return mDoGlob;
}


std::vector< Pt2di > & cGlobRappInit::Degres()
{
   return mDegres;
}

const std::vector< Pt2di > & cGlobRappInit::Degres()const 
{
   return mDegres;
}


std::vector< Pt2di > & cGlobRappInit::DegresSec()
{
   return mDegresSec;
}

const std::vector< Pt2di > & cGlobRappInit::DegresSec()const 
{
   return mDegresSec;
}


cTplValGesInit< std::string > & cGlobRappInit::PatternApply()
{
   return mPatternApply;
}

const cTplValGesInit< std::string > & cGlobRappInit::PatternApply()const 
{
   return mPatternApply;
}


cTplValGesInit< bool > & cGlobRappInit::RapelOnEgalPhys()
{
   return mRapelOnEgalPhys;
}

const cTplValGesInit< bool > & cGlobRappInit::RapelOnEgalPhys()const 
{
   return mRapelOnEgalPhys;
}

cElXMLTree * ToXMLTree(const cGlobRappInit & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"GlobRappInit",eXMLBranche);
   if (anObj.DoGlob().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DoGlob"),anObj.DoGlob().Val())->ReTagThis("DoGlob"));
  for
  (       std::vector< Pt2di >::const_iterator it=anObj.Degres().begin();
      it !=anObj.Degres().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("Degres"),(*it))->ReTagThis("Degres"));
  for
  (       std::vector< Pt2di >::const_iterator it=anObj.DegresSec().begin();
      it !=anObj.DegresSec().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("DegresSec"),(*it))->ReTagThis("DegresSec"));
   if (anObj.PatternApply().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PatternApply"),anObj.PatternApply().Val())->ReTagThis("PatternApply"));
   if (anObj.RapelOnEgalPhys().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("RapelOnEgalPhys"),anObj.RapelOnEgalPhys().Val())->ReTagThis("RapelOnEgalPhys"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cGlobRappInit & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.DoGlob(),aTree->Get("DoGlob",1),bool(true)); //tototo 

   xml_init(anObj.Degres(),aTree->GetAll("Degres",false,1));

   xml_init(anObj.DegresSec(),aTree->GetAll("DegresSec",false,1));

   xml_init(anObj.PatternApply(),aTree->Get("PatternApply",1),std::string(".*")); //tototo 

   xml_init(anObj.RapelOnEgalPhys(),aTree->Get("RapelOnEgalPhys",1),bool(true)); //tototo 
}


cTplValGesInit< cMasqTerrain > & cSectionEgalisation::MasqApprent()
{
   return mMasqApprent;
}

const cTplValGesInit< cMasqTerrain > & cSectionEgalisation::MasqApprent()const 
{
   return mMasqApprent;
}


cTplValGesInit< int > & cSectionEgalisation::PeriodEchant()
{
   return mPeriodEchant;
}

const cTplValGesInit< int > & cSectionEgalisation::PeriodEchant()const 
{
   return mPeriodEchant;
}


cTplValGesInit< double > & cSectionEgalisation::NbPEqualMoyPerImage()
{
   return mNbPEqualMoyPerImage;
}

const cTplValGesInit< double > & cSectionEgalisation::NbPEqualMoyPerImage()const 
{
   return mNbPEqualMoyPerImage;
}


int & cSectionEgalisation::SzVois()
{
   return mSzVois;
}

const int & cSectionEgalisation::SzVois()const 
{
   return mSzVois;
}


std::string & cSectionEgalisation::NameFileMesures()
{
   return mNameFileMesures;
}

const std::string & cSectionEgalisation::NameFileMesures()const 
{
   return mNameFileMesures;
}


cTplValGesInit< bool > & cSectionEgalisation::UseFileMesure()
{
   return mUseFileMesure;
}

const cTplValGesInit< bool > & cSectionEgalisation::UseFileMesure()const 
{
   return mUseFileMesure;
}


std::vector< Pt2di > & cSectionEgalisation::DegresEgalVois()
{
   return mDegresEgalVois;
}

const std::vector< Pt2di > & cSectionEgalisation::DegresEgalVois()const 
{
   return mDegresEgalVois;
}


std::vector< Pt2di > & cSectionEgalisation::DegresEgalVoisSec()
{
   return mDegresEgalVoisSec;
}

const std::vector< Pt2di > & cSectionEgalisation::DegresEgalVoisSec()const 
{
   return mDegresEgalVoisSec;
}


cTplValGesInit< double > & cSectionEgalisation::PdsRappelInit()
{
   return mPdsRappelInit;
}

const cTplValGesInit< double > & cSectionEgalisation::PdsRappelInit()const 
{
   return mPdsRappelInit;
}


cTplValGesInit< double > & cSectionEgalisation::PdsSingularite()
{
   return mPdsSingularite;
}

const cTplValGesInit< double > & cSectionEgalisation::PdsSingularite()const 
{
   return mPdsSingularite;
}


cTplValGesInit< bool > & cSectionEgalisation::DoGlob()
{
   return GlobRappInit().DoGlob();
}

const cTplValGesInit< bool > & cSectionEgalisation::DoGlob()const 
{
   return GlobRappInit().DoGlob();
}


std::vector< Pt2di > & cSectionEgalisation::Degres()
{
   return GlobRappInit().Degres();
}

const std::vector< Pt2di > & cSectionEgalisation::Degres()const 
{
   return GlobRappInit().Degres();
}


std::vector< Pt2di > & cSectionEgalisation::DegresSec()
{
   return GlobRappInit().DegresSec();
}

const std::vector< Pt2di > & cSectionEgalisation::DegresSec()const 
{
   return GlobRappInit().DegresSec();
}


cTplValGesInit< std::string > & cSectionEgalisation::PatternApply()
{
   return GlobRappInit().PatternApply();
}

const cTplValGesInit< std::string > & cSectionEgalisation::PatternApply()const 
{
   return GlobRappInit().PatternApply();
}


cTplValGesInit< bool > & cSectionEgalisation::RapelOnEgalPhys()
{
   return GlobRappInit().RapelOnEgalPhys();
}

const cTplValGesInit< bool > & cSectionEgalisation::RapelOnEgalPhys()const 
{
   return GlobRappInit().RapelOnEgalPhys();
}


cGlobRappInit & cSectionEgalisation::GlobRappInit()
{
   return mGlobRappInit;
}

const cGlobRappInit & cSectionEgalisation::GlobRappInit()const 
{
   return mGlobRappInit;
}


bool & cSectionEgalisation::EgaliseSomCh()
{
   return mEgaliseSomCh;
}

const bool & cSectionEgalisation::EgaliseSomCh()const 
{
   return mEgaliseSomCh;
}


cTplValGesInit< int > & cSectionEgalisation::SzMaxVois()
{
   return mSzMaxVois;
}

const cTplValGesInit< int > & cSectionEgalisation::SzMaxVois()const 
{
   return mSzMaxVois;
}


cTplValGesInit< bool > & cSectionEgalisation::Use4Vois()
{
   return mUse4Vois;
}

const cTplValGesInit< bool > & cSectionEgalisation::Use4Vois()const 
{
   return mUse4Vois;
}


cTplValGesInit< double > & cSectionEgalisation::CorrelThreshold()
{
   return mCorrelThreshold;
}

const cTplValGesInit< double > & cSectionEgalisation::CorrelThreshold()const 
{
   return mCorrelThreshold;
}


cTplValGesInit< bool > & cSectionEgalisation::AdjL1ByCple()
{
   return mAdjL1ByCple;
}

const cTplValGesInit< bool > & cSectionEgalisation::AdjL1ByCple()const 
{
   return mAdjL1ByCple;
}


cTplValGesInit< double > & cSectionEgalisation::PercCutAdjL1()
{
   return mPercCutAdjL1;
}

const cTplValGesInit< double > & cSectionEgalisation::PercCutAdjL1()const 
{
   return mPercCutAdjL1;
}


cTplValGesInit< double > & cSectionEgalisation::FactMajorByCutGlob()
{
   return mFactMajorByCutGlob;
}

const cTplValGesInit< double > & cSectionEgalisation::FactMajorByCutGlob()const 
{
   return mFactMajorByCutGlob;
}

cElXMLTree * ToXMLTree(const cSectionEgalisation & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"SectionEgalisation",eXMLBranche);
   if (anObj.MasqApprent().IsInit())
      aRes->AddFils(ToXMLTree(anObj.MasqApprent().Val())->ReTagThis("MasqApprent"));
   if (anObj.PeriodEchant().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PeriodEchant"),anObj.PeriodEchant().Val())->ReTagThis("PeriodEchant"));
   if (anObj.NbPEqualMoyPerImage().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NbPEqualMoyPerImage"),anObj.NbPEqualMoyPerImage().Val())->ReTagThis("NbPEqualMoyPerImage"));
   aRes->AddFils(::ToXMLTree(std::string("SzVois"),anObj.SzVois())->ReTagThis("SzVois"));
   aRes->AddFils(::ToXMLTree(std::string("NameFileMesures"),anObj.NameFileMesures())->ReTagThis("NameFileMesures"));
   if (anObj.UseFileMesure().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("UseFileMesure"),anObj.UseFileMesure().Val())->ReTagThis("UseFileMesure"));
  for
  (       std::vector< Pt2di >::const_iterator it=anObj.DegresEgalVois().begin();
      it !=anObj.DegresEgalVois().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("DegresEgalVois"),(*it))->ReTagThis("DegresEgalVois"));
  for
  (       std::vector< Pt2di >::const_iterator it=anObj.DegresEgalVoisSec().begin();
      it !=anObj.DegresEgalVoisSec().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("DegresEgalVoisSec"),(*it))->ReTagThis("DegresEgalVoisSec"));
   if (anObj.PdsRappelInit().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PdsRappelInit"),anObj.PdsRappelInit().Val())->ReTagThis("PdsRappelInit"));
   if (anObj.PdsSingularite().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PdsSingularite"),anObj.PdsSingularite().Val())->ReTagThis("PdsSingularite"));
   aRes->AddFils(ToXMLTree(anObj.GlobRappInit())->ReTagThis("GlobRappInit"));
   aRes->AddFils(::ToXMLTree(std::string("EgaliseSomCh"),anObj.EgaliseSomCh())->ReTagThis("EgaliseSomCh"));
   if (anObj.SzMaxVois().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SzMaxVois"),anObj.SzMaxVois().Val())->ReTagThis("SzMaxVois"));
   if (anObj.Use4Vois().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Use4Vois"),anObj.Use4Vois().Val())->ReTagThis("Use4Vois"));
   if (anObj.CorrelThreshold().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("CorrelThreshold"),anObj.CorrelThreshold().Val())->ReTagThis("CorrelThreshold"));
   if (anObj.AdjL1ByCple().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("AdjL1ByCple"),anObj.AdjL1ByCple().Val())->ReTagThis("AdjL1ByCple"));
   if (anObj.PercCutAdjL1().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PercCutAdjL1"),anObj.PercCutAdjL1().Val())->ReTagThis("PercCutAdjL1"));
   if (anObj.FactMajorByCutGlob().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("FactMajorByCutGlob"),anObj.FactMajorByCutGlob().Val())->ReTagThis("FactMajorByCutGlob"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cSectionEgalisation & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.MasqApprent(),aTree->Get("MasqApprent",1)); //tototo 

   xml_init(anObj.PeriodEchant(),aTree->Get("PeriodEchant",1)); //tototo 

   xml_init(anObj.NbPEqualMoyPerImage(),aTree->Get("NbPEqualMoyPerImage",1),double(1e4)); //tototo 

   xml_init(anObj.SzVois(),aTree->Get("SzVois",1)); //tototo 

   xml_init(anObj.NameFileMesures(),aTree->Get("NameFileMesures",1)); //tototo 

   xml_init(anObj.UseFileMesure(),aTree->Get("UseFileMesure",1),bool(false)); //tototo 

   xml_init(anObj.DegresEgalVois(),aTree->GetAll("DegresEgalVois",false,1));

   xml_init(anObj.DegresEgalVoisSec(),aTree->GetAll("DegresEgalVoisSec",false,1));

   xml_init(anObj.PdsRappelInit(),aTree->Get("PdsRappelInit",1),double(1e-3)); //tototo 

   xml_init(anObj.PdsSingularite(),aTree->Get("PdsSingularite",1),double(1e-6)); //tototo 

   xml_init(anObj.GlobRappInit(),aTree->Get("GlobRappInit",1)); //tototo 

   xml_init(anObj.EgaliseSomCh(),aTree->Get("EgaliseSomCh",1)); //tototo 

   xml_init(anObj.SzMaxVois(),aTree->Get("SzMaxVois",1),int(5)); //tototo 

   xml_init(anObj.Use4Vois(),aTree->Get("Use4Vois",1),bool(true)); //tototo 

   xml_init(anObj.CorrelThreshold(),aTree->Get("CorrelThreshold",1),double(0.7)); //tototo 

   xml_init(anObj.AdjL1ByCple(),aTree->Get("AdjL1ByCple",1),bool(true)); //tototo 

   xml_init(anObj.PercCutAdjL1(),aTree->Get("PercCutAdjL1",1),double(70)); //tototo 

   xml_init(anObj.FactMajorByCutGlob(),aTree->Get("FactMajorByCutGlob",1),double(1.5)); //tototo 
}


cTplValGesInit< cChantierDescripteur > & cCreateOrtho::DicoLoc()
{
   return mDicoLoc;
}

const cTplValGesInit< cChantierDescripteur > & cCreateOrtho::DicoLoc()const 
{
   return mDicoLoc;
}


cTplValGesInit< string > & cCreateOrtho::FileChantierNameDescripteur()
{
   return mFileChantierNameDescripteur;
}

const cTplValGesInit< string > & cCreateOrtho::FileChantierNameDescripteur()const 
{
   return mFileChantierNameDescripteur;
}


std::string & cCreateOrtho::WorkDir()
{
   return mWorkDir;
}

const std::string & cCreateOrtho::WorkDir()const 
{
   return mWorkDir;
}


cTplValGesInit< int > & cCreateOrtho::KBox0()
{
   return mKBox0;
}

const cTplValGesInit< int > & cCreateOrtho::KBox0()const 
{
   return mKBox0;
}


cTplValGesInit< std::string > & cCreateOrtho::FileMNT()
{
   return SectionEntree().FileMNT();
}

const cTplValGesInit< std::string > & cCreateOrtho::FileMNT()const 
{
   return SectionEntree().FileMNT();
}


std::string & cCreateOrtho::KeySetIm()
{
   return SectionEntree().KeySetIm();
}

const std::string & cCreateOrtho::KeySetIm()const 
{
   return SectionEntree().KeySetIm();
}


std::string & cCreateOrtho::KeyAssocMetaData()
{
   return SectionEntree().KeyAssocMetaData();
}

const std::string & cCreateOrtho::KeyAssocMetaData()const 
{
   return SectionEntree().KeyAssocMetaData();
}


std::string & cCreateOrtho::KeyAssocNamePC()
{
   return SectionEntree().KeyAssocNamePC();
}

const std::string & cCreateOrtho::KeyAssocNamePC()const 
{
   return SectionEntree().KeyAssocNamePC();
}


std::string & cCreateOrtho::KeyAssocNameIncH()
{
   return SectionEntree().KeyAssocNameIncH();
}

const std::string & cCreateOrtho::KeyAssocNameIncH()const 
{
   return SectionEntree().KeyAssocNameIncH();
}


cTplValGesInit< std::string > & cCreateOrtho::KeyAssocPriorite()
{
   return SectionEntree().KeyAssocPriorite();
}

const cTplValGesInit< std::string > & cCreateOrtho::KeyAssocPriorite()const 
{
   return SectionEntree().KeyAssocPriorite();
}


std::list< cMasqMesures > & cCreateOrtho::ListMasqMesures()
{
   return SectionEntree().ListMasqMesures();
}

const std::list< cMasqMesures > & cCreateOrtho::ListMasqMesures()const 
{
   return SectionEntree().ListMasqMesures();
}


std::list< std::string > & cCreateOrtho::FileExterneMasqMesures()
{
   return SectionEntree().FileExterneMasqMesures();
}

const std::list< std::string > & cCreateOrtho::FileExterneMasqMesures()const 
{
   return SectionEntree().FileExterneMasqMesures();
}


cSectionEntree & cCreateOrtho::SectionEntree()
{
   return mSectionEntree;
}

const cSectionEntree & cCreateOrtho::SectionEntree()const 
{
   return mSectionEntree;
}


cTplValGesInit< double > & cCreateOrtho::SaturThreshold()
{
   return SectionFiltrageIn().SaturThreshold();
}

const cTplValGesInit< double > & cCreateOrtho::SaturThreshold()const 
{
   return SectionFiltrageIn().SaturThreshold();
}


cTplValGesInit< int > & cCreateOrtho::SzDilatPC()
{
   return SectionFiltrageIn().SzDilatPC();
}

const cTplValGesInit< int > & cCreateOrtho::SzDilatPC()const 
{
   return SectionFiltrageIn().SzDilatPC();
}


cTplValGesInit< int > & cCreateOrtho::SzOuvPC()
{
   return SectionFiltrageIn().SzOuvPC();
}

const cTplValGesInit< int > & cCreateOrtho::SzOuvPC()const 
{
   return SectionFiltrageIn().SzOuvPC();
}


cTplValGesInit< int > & cCreateOrtho::SeuilVisib()
{
   return SectionFiltrageIn().BoucheTrou().Val().SeuilVisib();
}

const cTplValGesInit< int > & cCreateOrtho::SeuilVisib()const 
{
   return SectionFiltrageIn().BoucheTrou().Val().SeuilVisib();
}


cTplValGesInit< int > & cCreateOrtho::SeuilVisibBT()
{
   return SectionFiltrageIn().BoucheTrou().Val().SeuilVisibBT();
}

const cTplValGesInit< int > & cCreateOrtho::SeuilVisibBT()const 
{
   return SectionFiltrageIn().BoucheTrou().Val().SeuilVisibBT();
}


cTplValGesInit< double > & cCreateOrtho::CoeffPondAngul()
{
   return SectionFiltrageIn().BoucheTrou().Val().CoeffPondAngul();
}

const cTplValGesInit< double > & cCreateOrtho::CoeffPondAngul()const 
{
   return SectionFiltrageIn().BoucheTrou().Val().CoeffPondAngul();
}


cTplValGesInit< cBoucheTrou > & cCreateOrtho::BoucheTrou()
{
   return SectionFiltrageIn().BoucheTrou();
}

const cTplValGesInit< cBoucheTrou > & cCreateOrtho::BoucheTrou()const 
{
   return SectionFiltrageIn().BoucheTrou();
}


cSectionFiltrageIn & cCreateOrtho::SectionFiltrageIn()
{
   return mSectionFiltrageIn;
}

const cSectionFiltrageIn & cCreateOrtho::SectionFiltrageIn()const 
{
   return mSectionFiltrageIn;
}


cTplValGesInit< bool > & cCreateOrtho::TestDiff()
{
   return SectionSorties().TestDiff();
}

const cTplValGesInit< bool > & cCreateOrtho::TestDiff()const 
{
   return SectionSorties().TestDiff();
}


std::string & cCreateOrtho::NameOrtho()
{
   return SectionSorties().NameOrtho();
}

const std::string & cCreateOrtho::NameOrtho()const 
{
   return SectionSorties().NameOrtho();
}


cTplValGesInit< std::string > & cCreateOrtho::NameLabels()
{
   return SectionSorties().NameLabels();
}

const cTplValGesInit< std::string > & cCreateOrtho::NameLabels()const 
{
   return SectionSorties().NameLabels();
}


cTplValGesInit< Box2di > & cCreateOrtho::BoxCalc()
{
   return SectionSorties().BoxCalc();
}

const cTplValGesInit< Box2di > & cCreateOrtho::BoxCalc()const 
{
   return SectionSorties().BoxCalc();
}


int & cCreateOrtho::SzDalle()
{
   return SectionSorties().SzDalle();
}

const int & cCreateOrtho::SzDalle()const 
{
   return SectionSorties().SzDalle();
}


int & cCreateOrtho::SzBrd()
{
   return SectionSorties().SzBrd();
}

const int & cCreateOrtho::SzBrd()const 
{
   return SectionSorties().SzBrd();
}


cTplValGesInit< int > & cCreateOrtho::SzTileResult()
{
   return SectionSorties().SzTileResult();
}

const cTplValGesInit< int > & cCreateOrtho::SzTileResult()const 
{
   return SectionSorties().SzTileResult();
}


cTplValGesInit< bool > & cCreateOrtho::Show()
{
   return SectionSorties().Show();
}

const cTplValGesInit< bool > & cCreateOrtho::Show()const 
{
   return SectionSorties().Show();
}


cTplValGesInit< double > & cCreateOrtho::DynGlob()
{
   return SectionSorties().DynGlob();
}

const cTplValGesInit< double > & cCreateOrtho::DynGlob()const 
{
   return SectionSorties().DynGlob();
}


cSectionSorties & cCreateOrtho::SectionSorties()
{
   return mSectionSorties;
}

const cSectionSorties & cCreateOrtho::SectionSorties()const 
{
   return mSectionSorties;
}


Pt2dr & cCreateOrtho::Per1()
{
   return SectionSimulImage().Val().Per1();
}

const Pt2dr & cCreateOrtho::Per1()const 
{
   return SectionSimulImage().Val().Per1();
}


cTplValGesInit< Pt2dr > & cCreateOrtho::Per2()
{
   return SectionSimulImage().Val().Per2();
}

const cTplValGesInit< Pt2dr > & cCreateOrtho::Per2()const 
{
   return SectionSimulImage().Val().Per2();
}


cTplValGesInit< double > & cCreateOrtho::Ampl()
{
   return SectionSimulImage().Val().Ampl();
}

const cTplValGesInit< double > & cCreateOrtho::Ampl()const 
{
   return SectionSimulImage().Val().Ampl();
}


std::list< cNoiseSSI > & cCreateOrtho::NoiseSSI()
{
   return SectionSimulImage().Val().NoiseSSI();
}

const std::list< cNoiseSSI > & cCreateOrtho::NoiseSSI()const 
{
   return SectionSimulImage().Val().NoiseSSI();
}


cTplValGesInit< cSectionSimulImage > & cCreateOrtho::SectionSimulImage()
{
   return mSectionSimulImage;
}

const cTplValGesInit< cSectionSimulImage > & cCreateOrtho::SectionSimulImage()const 
{
   return mSectionSimulImage;
}


cTplValGesInit< cMasqTerrain > & cCreateOrtho::MasqApprent()
{
   return SectionEgalisation().Val().MasqApprent();
}

const cTplValGesInit< cMasqTerrain > & cCreateOrtho::MasqApprent()const 
{
   return SectionEgalisation().Val().MasqApprent();
}


cTplValGesInit< int > & cCreateOrtho::PeriodEchant()
{
   return SectionEgalisation().Val().PeriodEchant();
}

const cTplValGesInit< int > & cCreateOrtho::PeriodEchant()const 
{
   return SectionEgalisation().Val().PeriodEchant();
}


cTplValGesInit< double > & cCreateOrtho::NbPEqualMoyPerImage()
{
   return SectionEgalisation().Val().NbPEqualMoyPerImage();
}

const cTplValGesInit< double > & cCreateOrtho::NbPEqualMoyPerImage()const 
{
   return SectionEgalisation().Val().NbPEqualMoyPerImage();
}


int & cCreateOrtho::SzVois()
{
   return SectionEgalisation().Val().SzVois();
}

const int & cCreateOrtho::SzVois()const 
{
   return SectionEgalisation().Val().SzVois();
}


std::string & cCreateOrtho::NameFileMesures()
{
   return SectionEgalisation().Val().NameFileMesures();
}

const std::string & cCreateOrtho::NameFileMesures()const 
{
   return SectionEgalisation().Val().NameFileMesures();
}


cTplValGesInit< bool > & cCreateOrtho::UseFileMesure()
{
   return SectionEgalisation().Val().UseFileMesure();
}

const cTplValGesInit< bool > & cCreateOrtho::UseFileMesure()const 
{
   return SectionEgalisation().Val().UseFileMesure();
}


std::vector< Pt2di > & cCreateOrtho::DegresEgalVois()
{
   return SectionEgalisation().Val().DegresEgalVois();
}

const std::vector< Pt2di > & cCreateOrtho::DegresEgalVois()const 
{
   return SectionEgalisation().Val().DegresEgalVois();
}


std::vector< Pt2di > & cCreateOrtho::DegresEgalVoisSec()
{
   return SectionEgalisation().Val().DegresEgalVoisSec();
}

const std::vector< Pt2di > & cCreateOrtho::DegresEgalVoisSec()const 
{
   return SectionEgalisation().Val().DegresEgalVoisSec();
}


cTplValGesInit< double > & cCreateOrtho::PdsRappelInit()
{
   return SectionEgalisation().Val().PdsRappelInit();
}

const cTplValGesInit< double > & cCreateOrtho::PdsRappelInit()const 
{
   return SectionEgalisation().Val().PdsRappelInit();
}


cTplValGesInit< double > & cCreateOrtho::PdsSingularite()
{
   return SectionEgalisation().Val().PdsSingularite();
}

const cTplValGesInit< double > & cCreateOrtho::PdsSingularite()const 
{
   return SectionEgalisation().Val().PdsSingularite();
}


cTplValGesInit< bool > & cCreateOrtho::DoGlob()
{
   return SectionEgalisation().Val().GlobRappInit().DoGlob();
}

const cTplValGesInit< bool > & cCreateOrtho::DoGlob()const 
{
   return SectionEgalisation().Val().GlobRappInit().DoGlob();
}


std::vector< Pt2di > & cCreateOrtho::Degres()
{
   return SectionEgalisation().Val().GlobRappInit().Degres();
}

const std::vector< Pt2di > & cCreateOrtho::Degres()const 
{
   return SectionEgalisation().Val().GlobRappInit().Degres();
}


std::vector< Pt2di > & cCreateOrtho::DegresSec()
{
   return SectionEgalisation().Val().GlobRappInit().DegresSec();
}

const std::vector< Pt2di > & cCreateOrtho::DegresSec()const 
{
   return SectionEgalisation().Val().GlobRappInit().DegresSec();
}


cTplValGesInit< std::string > & cCreateOrtho::PatternApply()
{
   return SectionEgalisation().Val().GlobRappInit().PatternApply();
}

const cTplValGesInit< std::string > & cCreateOrtho::PatternApply()const 
{
   return SectionEgalisation().Val().GlobRappInit().PatternApply();
}


cTplValGesInit< bool > & cCreateOrtho::RapelOnEgalPhys()
{
   return SectionEgalisation().Val().GlobRappInit().RapelOnEgalPhys();
}

const cTplValGesInit< bool > & cCreateOrtho::RapelOnEgalPhys()const 
{
   return SectionEgalisation().Val().GlobRappInit().RapelOnEgalPhys();
}


cGlobRappInit & cCreateOrtho::GlobRappInit()
{
   return SectionEgalisation().Val().GlobRappInit();
}

const cGlobRappInit & cCreateOrtho::GlobRappInit()const 
{
   return SectionEgalisation().Val().GlobRappInit();
}


bool & cCreateOrtho::EgaliseSomCh()
{
   return SectionEgalisation().Val().EgaliseSomCh();
}

const bool & cCreateOrtho::EgaliseSomCh()const 
{
   return SectionEgalisation().Val().EgaliseSomCh();
}


cTplValGesInit< int > & cCreateOrtho::SzMaxVois()
{
   return SectionEgalisation().Val().SzMaxVois();
}

const cTplValGesInit< int > & cCreateOrtho::SzMaxVois()const 
{
   return SectionEgalisation().Val().SzMaxVois();
}


cTplValGesInit< bool > & cCreateOrtho::Use4Vois()
{
   return SectionEgalisation().Val().Use4Vois();
}

const cTplValGesInit< bool > & cCreateOrtho::Use4Vois()const 
{
   return SectionEgalisation().Val().Use4Vois();
}


cTplValGesInit< double > & cCreateOrtho::CorrelThreshold()
{
   return SectionEgalisation().Val().CorrelThreshold();
}

const cTplValGesInit< double > & cCreateOrtho::CorrelThreshold()const 
{
   return SectionEgalisation().Val().CorrelThreshold();
}


cTplValGesInit< bool > & cCreateOrtho::AdjL1ByCple()
{
   return SectionEgalisation().Val().AdjL1ByCple();
}

const cTplValGesInit< bool > & cCreateOrtho::AdjL1ByCple()const 
{
   return SectionEgalisation().Val().AdjL1ByCple();
}


cTplValGesInit< double > & cCreateOrtho::PercCutAdjL1()
{
   return SectionEgalisation().Val().PercCutAdjL1();
}

const cTplValGesInit< double > & cCreateOrtho::PercCutAdjL1()const 
{
   return SectionEgalisation().Val().PercCutAdjL1();
}


cTplValGesInit< double > & cCreateOrtho::FactMajorByCutGlob()
{
   return SectionEgalisation().Val().FactMajorByCutGlob();
}

const cTplValGesInit< double > & cCreateOrtho::FactMajorByCutGlob()const 
{
   return SectionEgalisation().Val().FactMajorByCutGlob();
}


cTplValGesInit< cSectionEgalisation > & cCreateOrtho::SectionEgalisation()
{
   return mSectionEgalisation;
}

const cTplValGesInit< cSectionEgalisation > & cCreateOrtho::SectionEgalisation()const 
{
   return mSectionEgalisation;
}

cElXMLTree * ToXMLTree(const cCreateOrtho & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"CreateOrtho",eXMLBranche);
   if (anObj.DicoLoc().IsInit())
      aRes->AddFils(ToXMLTree(anObj.DicoLoc().Val())->ReTagThis("DicoLoc"));
   if (anObj.FileChantierNameDescripteur().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("FileChantierNameDescripteur"),anObj.FileChantierNameDescripteur().Val())->ReTagThis("FileChantierNameDescripteur"));
   aRes->AddFils(::ToXMLTree(std::string("WorkDir"),anObj.WorkDir())->ReTagThis("WorkDir"));
   if (anObj.KBox0().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("KBox0"),anObj.KBox0().Val())->ReTagThis("KBox0"));
   aRes->AddFils(ToXMLTree(anObj.SectionEntree())->ReTagThis("SectionEntree"));
   aRes->AddFils(ToXMLTree(anObj.SectionFiltrageIn())->ReTagThis("SectionFiltrageIn"));
   aRes->AddFils(ToXMLTree(anObj.SectionSorties())->ReTagThis("SectionSorties"));
   if (anObj.SectionSimulImage().IsInit())
      aRes->AddFils(ToXMLTree(anObj.SectionSimulImage().Val())->ReTagThis("SectionSimulImage"));
   if (anObj.SectionEgalisation().IsInit())
      aRes->AddFils(ToXMLTree(anObj.SectionEgalisation().Val())->ReTagThis("SectionEgalisation"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cCreateOrtho & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.DicoLoc(),aTree->Get("DicoLoc",1)); //tototo 

   xml_init(anObj.FileChantierNameDescripteur(),aTree->Get("FileChantierNameDescripteur",1)); //tototo 

   xml_init(anObj.WorkDir(),aTree->Get("WorkDir",1)); //tototo 

   xml_init(anObj.KBox0(),aTree->Get("KBox0",1),int(0)); //tototo 

   xml_init(anObj.SectionEntree(),aTree->Get("SectionEntree",1)); //tototo 

   xml_init(anObj.SectionFiltrageIn(),aTree->Get("SectionFiltrageIn",1)); //tototo 

   xml_init(anObj.SectionSorties(),aTree->Get("SectionSorties",1)); //tototo 

   xml_init(anObj.SectionSimulImage(),aTree->Get("SectionSimulImage",1)); //tototo 

   xml_init(anObj.SectionEgalisation(),aTree->Get("SectionEgalisation",1)); //tototo 
}


bool & cMetaDataPartiesCachees::Done()
{
   return mDone;
}

const bool & cMetaDataPartiesCachees::Done()const 
{
   return mDone;
}


Pt2di & cMetaDataPartiesCachees::Offset()
{
   return mOffset;
}

const Pt2di & cMetaDataPartiesCachees::Offset()const 
{
   return mOffset;
}


Pt2di & cMetaDataPartiesCachees::Sz()
{
   return mSz;
}

const Pt2di & cMetaDataPartiesCachees::Sz()const 
{
   return mSz;
}


double & cMetaDataPartiesCachees::Pas()
{
   return mPas;
}

const double & cMetaDataPartiesCachees::Pas()const 
{
   return mPas;
}


int & cMetaDataPartiesCachees::SeuilUse()
{
   return mSeuilUse;
}

const int & cMetaDataPartiesCachees::SeuilUse()const 
{
   return mSeuilUse;
}


cTplValGesInit< double > & cMetaDataPartiesCachees::SsResolIncH()
{
   return mSsResolIncH;
}

const cTplValGesInit< double > & cMetaDataPartiesCachees::SsResolIncH()const 
{
   return mSsResolIncH;
}

cElXMLTree * ToXMLTree(const cMetaDataPartiesCachees & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"MetaDataPartiesCachees",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Done"),anObj.Done())->ReTagThis("Done"));
   aRes->AddFils(::ToXMLTree(std::string("Offset"),anObj.Offset())->ReTagThis("Offset"));
   aRes->AddFils(::ToXMLTree(std::string("Sz"),anObj.Sz())->ReTagThis("Sz"));
   aRes->AddFils(::ToXMLTree(std::string("Pas"),anObj.Pas())->ReTagThis("Pas"));
   aRes->AddFils(::ToXMLTree(std::string("SeuilUse"),anObj.SeuilUse())->ReTagThis("SeuilUse"));
   if (anObj.SsResolIncH().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SsResolIncH"),anObj.SsResolIncH().Val())->ReTagThis("SsResolIncH"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cMetaDataPartiesCachees & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Done(),aTree->Get("Done",1)); //tototo 

   xml_init(anObj.Offset(),aTree->Get("Offset",1)); //tototo 

   xml_init(anObj.Sz(),aTree->Get("Sz",1)); //tototo 

   xml_init(anObj.Pas(),aTree->Get("Pas",1)); //tototo 

   xml_init(anObj.SeuilUse(),aTree->Get("SeuilUse",1)); //tototo 

   xml_init(anObj.SsResolIncH(),aTree->Get("SsResolIncH",1)); //tototo 
}


cTplValGesInit< Pt3dr > & cPVPN_Orientation::AngleCardan()
{
   return mAngleCardan;
}

const cTplValGesInit< Pt3dr > & cPVPN_Orientation::AngleCardan()const 
{
   return mAngleCardan;
}

cElXMLTree * ToXMLTree(const cPVPN_Orientation & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"PVPN_Orientation",eXMLBranche);
   if (anObj.AngleCardan().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("AngleCardan"),anObj.AngleCardan().Val())->ReTagThis("AngleCardan"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cPVPN_Orientation & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.AngleCardan(),aTree->Get("AngleCardan",1)); //tototo 
}


cPVPN_Orientation & cPVPN_ImFixe::Orient()
{
   return mOrient;
}

const cPVPN_Orientation & cPVPN_ImFixe::Orient()const 
{
   return mOrient;
}


std::string & cPVPN_ImFixe::Name()
{
   return mName;
}

const std::string & cPVPN_ImFixe::Name()const 
{
   return mName;
}

cElXMLTree * ToXMLTree(const cPVPN_ImFixe & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"PVPN_ImFixe",eXMLBranche);
   aRes->AddFils(ToXMLTree(anObj.Orient())->ReTagThis("Orient"));
   aRes->AddFils(::ToXMLTree(std::string("Name"),anObj.Name())->ReTagThis("Name"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cPVPN_ImFixe & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Orient(),aTree->Get("Orient",1)); //tototo 

   xml_init(anObj.Name(),aTree->Get("Name",1)); //tototo 
}


Pt2di & cPVPN_Camera::NbPixel()
{
   return mNbPixel;
}

const Pt2di & cPVPN_Camera::NbPixel()const 
{
   return mNbPixel;
}


double & cPVPN_Camera::AngleDiag()
{
   return mAngleDiag;
}

const double & cPVPN_Camera::AngleDiag()const 
{
   return mAngleDiag;
}

cElXMLTree * ToXMLTree(const cPVPN_Camera & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"PVPN_Camera",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("NbPixel"),anObj.NbPixel())->ReTagThis("NbPixel"));
   aRes->AddFils(::ToXMLTree(std::string("AngleDiag"),anObj.AngleDiag())->ReTagThis("AngleDiag"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cPVPN_Camera & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.NbPixel(),aTree->Get("NbPixel",1)); //tototo 

   xml_init(anObj.AngleDiag(),aTree->Get("AngleDiag",1)); //tototo 
}


cTplValGesInit< Pt3dr > & cPVPN_Fond::FondConstant()
{
   return mFondConstant;
}

const cTplValGesInit< Pt3dr > & cPVPN_Fond::FondConstant()const 
{
   return mFondConstant;
}

cElXMLTree * ToXMLTree(const cPVPN_Fond & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"PVPN_Fond",eXMLBranche);
   if (anObj.FondConstant().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("FondConstant"),anObj.FondConstant().Val())->ReTagThis("FondConstant"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cPVPN_Fond & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.FondConstant(),aTree->Get("FondConstant",1)); //tototo 
}


std::string & cPVPN_Nuages::Name()
{
   return mName;
}

const std::string & cPVPN_Nuages::Name()const 
{
   return mName;
}

cElXMLTree * ToXMLTree(const cPVPN_Nuages & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"PVPN_Nuages",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Name"),anObj.Name())->ReTagThis("Name"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cPVPN_Nuages & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Name(),aTree->Get("Name",1)); //tototo 
}


std::string & cParamVisuProjNuage::WorkDir()
{
   return mWorkDir;
}

const std::string & cParamVisuProjNuage::WorkDir()const 
{
   return mWorkDir;
}


cTplValGesInit< cChantierDescripteur > & cParamVisuProjNuage::DicoLoc()
{
   return mDicoLoc;
}

const cTplValGesInit< cChantierDescripteur > & cParamVisuProjNuage::DicoLoc()const 
{
   return mDicoLoc;
}


cTplValGesInit< string > & cParamVisuProjNuage::FileChantierNameDescripteur()
{
   return mFileChantierNameDescripteur;
}

const cTplValGesInit< string > & cParamVisuProjNuage::FileChantierNameDescripteur()const 
{
   return mFileChantierNameDescripteur;
}


cPVPN_Orientation & cParamVisuProjNuage::Orient()
{
   return PVPN_ImFixe().Val().Orient();
}

const cPVPN_Orientation & cParamVisuProjNuage::Orient()const 
{
   return PVPN_ImFixe().Val().Orient();
}


std::string & cParamVisuProjNuage::Name()
{
   return PVPN_ImFixe().Val().Name();
}

const std::string & cParamVisuProjNuage::Name()const 
{
   return PVPN_ImFixe().Val().Name();
}


cTplValGesInit< cPVPN_ImFixe > & cParamVisuProjNuage::PVPN_ImFixe()
{
   return mPVPN_ImFixe;
}

const cTplValGesInit< cPVPN_ImFixe > & cParamVisuProjNuage::PVPN_ImFixe()const 
{
   return mPVPN_ImFixe;
}


Pt2di & cParamVisuProjNuage::NbPixel()
{
   return PVPN_Camera().NbPixel();
}

const Pt2di & cParamVisuProjNuage::NbPixel()const 
{
   return PVPN_Camera().NbPixel();
}


double & cParamVisuProjNuage::AngleDiag()
{
   return PVPN_Camera().AngleDiag();
}

const double & cParamVisuProjNuage::AngleDiag()const 
{
   return PVPN_Camera().AngleDiag();
}


cPVPN_Camera & cParamVisuProjNuage::PVPN_Camera()
{
   return mPVPN_Camera;
}

const cPVPN_Camera & cParamVisuProjNuage::PVPN_Camera()const 
{
   return mPVPN_Camera;
}


cTplValGesInit< Pt3dr > & cParamVisuProjNuage::FondConstant()
{
   return PVPN_Fond().FondConstant();
}

const cTplValGesInit< Pt3dr > & cParamVisuProjNuage::FondConstant()const 
{
   return PVPN_Fond().FondConstant();
}


cPVPN_Fond & cParamVisuProjNuage::PVPN_Fond()
{
   return mPVPN_Fond;
}

const cPVPN_Fond & cParamVisuProjNuage::PVPN_Fond()const 
{
   return mPVPN_Fond;
}


std::list< cPVPN_Nuages > & cParamVisuProjNuage::PVPN_Nuages()
{
   return mPVPN_Nuages;
}

const std::list< cPVPN_Nuages > & cParamVisuProjNuage::PVPN_Nuages()const 
{
   return mPVPN_Nuages;
}


cTplValGesInit< double > & cParamVisuProjNuage::SousEchQuickN()
{
   return mSousEchQuickN;
}

const cTplValGesInit< double > & cParamVisuProjNuage::SousEchQuickN()const 
{
   return mSousEchQuickN;
}

cElXMLTree * ToXMLTree(const cParamVisuProjNuage & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ParamVisuProjNuage",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("WorkDir"),anObj.WorkDir())->ReTagThis("WorkDir"));
   if (anObj.DicoLoc().IsInit())
      aRes->AddFils(ToXMLTree(anObj.DicoLoc().Val())->ReTagThis("DicoLoc"));
   if (anObj.FileChantierNameDescripteur().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("FileChantierNameDescripteur"),anObj.FileChantierNameDescripteur().Val())->ReTagThis("FileChantierNameDescripteur"));
   if (anObj.PVPN_ImFixe().IsInit())
      aRes->AddFils(ToXMLTree(anObj.PVPN_ImFixe().Val())->ReTagThis("PVPN_ImFixe"));
   aRes->AddFils(ToXMLTree(anObj.PVPN_Camera())->ReTagThis("PVPN_Camera"));
   aRes->AddFils(ToXMLTree(anObj.PVPN_Fond())->ReTagThis("PVPN_Fond"));
  for
  (       std::list< cPVPN_Nuages >::const_iterator it=anObj.PVPN_Nuages().begin();
      it !=anObj.PVPN_Nuages().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("PVPN_Nuages"));
   if (anObj.SousEchQuickN().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SousEchQuickN"),anObj.SousEchQuickN().Val())->ReTagThis("SousEchQuickN"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cParamVisuProjNuage & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.WorkDir(),aTree->Get("WorkDir",1)); //tototo 

   xml_init(anObj.DicoLoc(),aTree->Get("DicoLoc",1)); //tototo 

   xml_init(anObj.FileChantierNameDescripteur(),aTree->Get("FileChantierNameDescripteur",1)); //tototo 

   xml_init(anObj.PVPN_ImFixe(),aTree->Get("PVPN_ImFixe",1)); //tototo 

   xml_init(anObj.PVPN_Camera(),aTree->Get("PVPN_Camera",1)); //tototo 

   xml_init(anObj.PVPN_Fond(),aTree->Get("PVPN_Fond",1)); //tototo 

   xml_init(anObj.PVPN_Nuages(),aTree->GetAll("PVPN_Nuages",false,1));

   xml_init(anObj.SousEchQuickN(),aTree->Get("SousEchQuickN",1),double(10)); //tototo 
}


double & cPoinAvionJaune::x()
{
   return mx;
}

const double & cPoinAvionJaune::x()const 
{
   return mx;
}


double & cPoinAvionJaune::y()
{
   return my;
}

const double & cPoinAvionJaune::y()const 
{
   return my;
}

cElXMLTree * ToXMLTree(const cPoinAvionJaune & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"PoinAvionJaune",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("x"),anObj.x())->ReTagThis("x"));
   aRes->AddFils(::ToXMLTree(std::string("y"),anObj.y())->ReTagThis("y"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cPoinAvionJaune & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.x(),aTree->Get("x",1)); //tototo 

   xml_init(anObj.y(),aTree->Get("y",1)); //tototo 
}


std::string & cValueAvionJaune::unit()
{
   return munit;
}

const std::string & cValueAvionJaune::unit()const 
{
   return munit;
}


cTplValGesInit< std::string > & cValueAvionJaune::source()
{
   return msource;
}

const cTplValGesInit< std::string > & cValueAvionJaune::source()const 
{
   return msource;
}


cTplValGesInit< double > & cValueAvionJaune::biaisCorrige()
{
   return mbiaisCorrige;
}

const cTplValGesInit< double > & cValueAvionJaune::biaisCorrige()const 
{
   return mbiaisCorrige;
}


double & cValueAvionJaune::value()
{
   return mvalue;
}

const double & cValueAvionJaune::value()const 
{
   return mvalue;
}

cElXMLTree * ToXMLTree(const cValueAvionJaune & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ValueAvionJaune",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("unit"),anObj.unit())->ReTagThis("unit"));
   if (anObj.source().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("source"),anObj.source().Val())->ReTagThis("source"));
   if (anObj.biaisCorrige().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("biaisCorrige"),anObj.biaisCorrige().Val())->ReTagThis("biaisCorrige"));
   aRes->AddFils(::ToXMLTree(std::string("value"),anObj.value())->ReTagThis("value"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cValueAvionJaune & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.unit(),aTree->Get("unit",1)); //tototo 

   xml_init(anObj.source(),aTree->Get("source",1)); //tototo 

   xml_init(anObj.biaisCorrige(),aTree->Get("biaisCorrige",1)); //tototo 

   xml_init(anObj.value(),aTree->Get("value",1)); //tototo 
}


std::string & cValueXYAvionJaune::unit()
{
   return munit;
}

const std::string & cValueXYAvionJaune::unit()const 
{
   return munit;
}


cTplValGesInit< std::string > & cValueXYAvionJaune::source()
{
   return msource;
}

const cTplValGesInit< std::string > & cValueXYAvionJaune::source()const 
{
   return msource;
}


cTplValGesInit< double > & cValueXYAvionJaune::biaisCorrige()
{
   return mbiaisCorrige;
}

const cTplValGesInit< double > & cValueXYAvionJaune::biaisCorrige()const 
{
   return mbiaisCorrige;
}


double & cValueXYAvionJaune::xvalue()
{
   return mxvalue;
}

const double & cValueXYAvionJaune::xvalue()const 
{
   return mxvalue;
}


double & cValueXYAvionJaune::yvalue()
{
   return myvalue;
}

const double & cValueXYAvionJaune::yvalue()const 
{
   return myvalue;
}

cElXMLTree * ToXMLTree(const cValueXYAvionJaune & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ValueXYAvionJaune",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("unit"),anObj.unit())->ReTagThis("unit"));
   if (anObj.source().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("source"),anObj.source().Val())->ReTagThis("source"));
   if (anObj.biaisCorrige().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("biaisCorrige"),anObj.biaisCorrige().Val())->ReTagThis("biaisCorrige"));
   aRes->AddFils(::ToXMLTree(std::string("xvalue"),anObj.xvalue())->ReTagThis("xvalue"));
   aRes->AddFils(::ToXMLTree(std::string("yvalue"),anObj.yvalue())->ReTagThis("yvalue"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cValueXYAvionJaune & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.unit(),aTree->Get("unit",1)); //tototo 

   xml_init(anObj.source(),aTree->Get("source",1)); //tototo 

   xml_init(anObj.biaisCorrige(),aTree->Get("biaisCorrige",1)); //tototo 

   xml_init(anObj.xvalue(),aTree->Get("xvalue",1)); //tototo 

   xml_init(anObj.yvalue(),aTree->Get("yvalue",1)); //tototo 
}


std::string & cnavigation::systemeGeodesique()
{
   return msystemeGeodesique;
}

const std::string & cnavigation::systemeGeodesique()const 
{
   return msystemeGeodesique;
}


std::string & cnavigation::projection()
{
   return mprojection;
}

const std::string & cnavigation::projection()const 
{
   return mprojection;
}


cPoinAvionJaune & cnavigation::sommet()
{
   return msommet;
}

const cPoinAvionJaune & cnavigation::sommet()const 
{
   return msommet;
}


cValueAvionJaune & cnavigation::altitude()
{
   return maltitude;
}

const cValueAvionJaune & cnavigation::altitude()const 
{
   return maltitude;
}


cValueAvionJaune & cnavigation::capAvion()
{
   return mcapAvion;
}

const cValueAvionJaune & cnavigation::capAvion()const 
{
   return mcapAvion;
}


cValueAvionJaune & cnavigation::roulisAvion()
{
   return mroulisAvion;
}

const cValueAvionJaune & cnavigation::roulisAvion()const 
{
   return mroulisAvion;
}


cValueAvionJaune & cnavigation::tangageAvion()
{
   return mtangageAvion;
}

const cValueAvionJaune & cnavigation::tangageAvion()const 
{
   return mtangageAvion;
}


cValueAvionJaune & cnavigation::tempsAutopilote()
{
   return mtempsAutopilote;
}

const cValueAvionJaune & cnavigation::tempsAutopilote()const 
{
   return mtempsAutopilote;
}

cElXMLTree * ToXMLTree(const cnavigation & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"navigation",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("systemeGeodesique"),anObj.systemeGeodesique())->ReTagThis("systemeGeodesique"));
   aRes->AddFils(::ToXMLTree(std::string("projection"),anObj.projection())->ReTagThis("projection"));
   aRes->AddFils(ToXMLTree(anObj.sommet())->ReTagThis("sommet"));
   aRes->AddFils(ToXMLTree(anObj.altitude())->ReTagThis("altitude"));
   aRes->AddFils(ToXMLTree(anObj.capAvion())->ReTagThis("capAvion"));
   aRes->AddFils(ToXMLTree(anObj.roulisAvion())->ReTagThis("roulisAvion"));
   aRes->AddFils(ToXMLTree(anObj.tangageAvion())->ReTagThis("tangageAvion"));
   aRes->AddFils(ToXMLTree(anObj.tempsAutopilote())->ReTagThis("tempsAutopilote"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cnavigation & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.systemeGeodesique(),aTree->Get("systemeGeodesique",1)); //tototo 

   xml_init(anObj.projection(),aTree->Get("projection",1)); //tototo 

   xml_init(anObj.sommet(),aTree->Get("sommet",1)); //tototo 

   xml_init(anObj.altitude(),aTree->Get("altitude",1)); //tototo 

   xml_init(anObj.capAvion(),aTree->Get("capAvion",1)); //tototo 

   xml_init(anObj.roulisAvion(),aTree->Get("roulisAvion",1)); //tototo 

   xml_init(anObj.tangageAvion(),aTree->Get("tangageAvion",1)); //tototo 

   xml_init(anObj.tempsAutopilote(),aTree->Get("tempsAutopilote",1)); //tototo 
}


cValueAvionJaune & cimage::focale()
{
   return mfocale;
}

const cValueAvionJaune & cimage::focale()const 
{
   return mfocale;
}


cValueAvionJaune & cimage::ouverture()
{
   return mouverture;
}

const cValueAvionJaune & cimage::ouverture()const 
{
   return mouverture;
}


cValueAvionJaune & cimage::tempsDExposition()
{
   return mtempsDExposition;
}

const cValueAvionJaune & cimage::tempsDExposition()const 
{
   return mtempsDExposition;
}

cElXMLTree * ToXMLTree(const cimage & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"image",eXMLBranche);
   aRes->AddFils(ToXMLTree(anObj.focale())->ReTagThis("focale"));
   aRes->AddFils(ToXMLTree(anObj.ouverture())->ReTagThis("ouverture"));
   aRes->AddFils(ToXMLTree(anObj.tempsDExposition())->ReTagThis("tempsDExposition"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cimage & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.focale(),aTree->Get("focale",1)); //tototo 

   xml_init(anObj.ouverture(),aTree->Get("ouverture",1)); //tototo 

   xml_init(anObj.tempsDExposition(),aTree->Get("tempsDExposition",1)); //tototo 
}


cValueAvionJaune & cgeometrieAPriori::hauteur()
{
   return mhauteur;
}

const cValueAvionJaune & cgeometrieAPriori::hauteur()const 
{
   return mhauteur;
}


cValueXYAvionJaune & cgeometrieAPriori::resolution()
{
   return mresolution;
}

const cValueXYAvionJaune & cgeometrieAPriori::resolution()const 
{
   return mresolution;
}


cValueAvionJaune & cgeometrieAPriori::orientationAPN()
{
   return morientationAPN;
}

const cValueAvionJaune & cgeometrieAPriori::orientationAPN()const 
{
   return morientationAPN;
}


std::vector< cPoinAvionJaune > & cgeometrieAPriori::coin()
{
   return mcoin;
}

const std::vector< cPoinAvionJaune > & cgeometrieAPriori::coin()const 
{
   return mcoin;
}

cElXMLTree * ToXMLTree(const cgeometrieAPriori & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"geometrieAPriori",eXMLBranche);
   aRes->AddFils(ToXMLTree(anObj.hauteur())->ReTagThis("hauteur"));
   aRes->AddFils(ToXMLTree(anObj.resolution())->ReTagThis("resolution"));
   aRes->AddFils(ToXMLTree(anObj.orientationAPN())->ReTagThis("orientationAPN"));
  for
  (       std::vector< cPoinAvionJaune >::const_iterator it=anObj.coin().begin();
      it !=anObj.coin().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("coin"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cgeometrieAPriori & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.hauteur(),aTree->Get("hauteur",1)); //tototo 

   xml_init(anObj.resolution(),aTree->Get("resolution",1)); //tototo 

   xml_init(anObj.orientationAPN(),aTree->Get("orientationAPN",1)); //tototo 

   xml_init(anObj.coin(),aTree->GetAll("coin",false,1));
}


std::string & cAvionJauneDocument::numeroImage()
{
   return mnumeroImage;
}

const std::string & cAvionJauneDocument::numeroImage()const 
{
   return mnumeroImage;
}


std::string & cAvionJauneDocument::systemeGeodesique()
{
   return navigation().systemeGeodesique();
}

const std::string & cAvionJauneDocument::systemeGeodesique()const 
{
   return navigation().systemeGeodesique();
}


std::string & cAvionJauneDocument::projection()
{
   return navigation().projection();
}

const std::string & cAvionJauneDocument::projection()const 
{
   return navigation().projection();
}


cPoinAvionJaune & cAvionJauneDocument::sommet()
{
   return navigation().sommet();
}

const cPoinAvionJaune & cAvionJauneDocument::sommet()const 
{
   return navigation().sommet();
}


cValueAvionJaune & cAvionJauneDocument::altitude()
{
   return navigation().altitude();
}

const cValueAvionJaune & cAvionJauneDocument::altitude()const 
{
   return navigation().altitude();
}


cValueAvionJaune & cAvionJauneDocument::capAvion()
{
   return navigation().capAvion();
}

const cValueAvionJaune & cAvionJauneDocument::capAvion()const 
{
   return navigation().capAvion();
}


cValueAvionJaune & cAvionJauneDocument::roulisAvion()
{
   return navigation().roulisAvion();
}

const cValueAvionJaune & cAvionJauneDocument::roulisAvion()const 
{
   return navigation().roulisAvion();
}


cValueAvionJaune & cAvionJauneDocument::tangageAvion()
{
   return navigation().tangageAvion();
}

const cValueAvionJaune & cAvionJauneDocument::tangageAvion()const 
{
   return navigation().tangageAvion();
}


cValueAvionJaune & cAvionJauneDocument::tempsAutopilote()
{
   return navigation().tempsAutopilote();
}

const cValueAvionJaune & cAvionJauneDocument::tempsAutopilote()const 
{
   return navigation().tempsAutopilote();
}


cnavigation & cAvionJauneDocument::navigation()
{
   return mnavigation;
}

const cnavigation & cAvionJauneDocument::navigation()const 
{
   return mnavigation;
}


cValueAvionJaune & cAvionJauneDocument::focale()
{
   return image().focale();
}

const cValueAvionJaune & cAvionJauneDocument::focale()const 
{
   return image().focale();
}


cValueAvionJaune & cAvionJauneDocument::ouverture()
{
   return image().ouverture();
}

const cValueAvionJaune & cAvionJauneDocument::ouverture()const 
{
   return image().ouverture();
}


cValueAvionJaune & cAvionJauneDocument::tempsDExposition()
{
   return image().tempsDExposition();
}

const cValueAvionJaune & cAvionJauneDocument::tempsDExposition()const 
{
   return image().tempsDExposition();
}


cimage & cAvionJauneDocument::image()
{
   return mimage;
}

const cimage & cAvionJauneDocument::image()const 
{
   return mimage;
}


cValueAvionJaune & cAvionJauneDocument::hauteur()
{
   return geometrieAPriori().hauteur();
}

const cValueAvionJaune & cAvionJauneDocument::hauteur()const 
{
   return geometrieAPriori().hauteur();
}


cValueXYAvionJaune & cAvionJauneDocument::resolution()
{
   return geometrieAPriori().resolution();
}

const cValueXYAvionJaune & cAvionJauneDocument::resolution()const 
{
   return geometrieAPriori().resolution();
}


cValueAvionJaune & cAvionJauneDocument::orientationAPN()
{
   return geometrieAPriori().orientationAPN();
}

const cValueAvionJaune & cAvionJauneDocument::orientationAPN()const 
{
   return geometrieAPriori().orientationAPN();
}


std::vector< cPoinAvionJaune > & cAvionJauneDocument::coin()
{
   return geometrieAPriori().coin();
}

const std::vector< cPoinAvionJaune > & cAvionJauneDocument::coin()const 
{
   return geometrieAPriori().coin();
}


cgeometrieAPriori & cAvionJauneDocument::geometrieAPriori()
{
   return mgeometrieAPriori;
}

const cgeometrieAPriori & cAvionJauneDocument::geometrieAPriori()const 
{
   return mgeometrieAPriori;
}

cElXMLTree * ToXMLTree(const cAvionJauneDocument & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"AvionJauneDocument",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("numeroImage"),anObj.numeroImage())->ReTagThis("numeroImage"));
   aRes->AddFils(ToXMLTree(anObj.navigation())->ReTagThis("navigation"));
   aRes->AddFils(ToXMLTree(anObj.image())->ReTagThis("image"));
   aRes->AddFils(ToXMLTree(anObj.geometrieAPriori())->ReTagThis("geometrieAPriori"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cAvionJauneDocument & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.numeroImage(),aTree->Get("numeroImage",1)); //tototo 

   xml_init(anObj.navigation(),aTree->Get("navigation",1)); //tototo 

   xml_init(anObj.image(),aTree->Get("image",1)); //tototo 

   xml_init(anObj.geometrieAPriori(),aTree->Get("geometrieAPriori",1)); //tototo 
}


cTplValGesInit< bool > & cTrAJ2_GenerateOrient::Teta1FromCap()
{
   return mTeta1FromCap;
}

const cTplValGesInit< bool > & cTrAJ2_GenerateOrient::Teta1FromCap()const 
{
   return mTeta1FromCap;
}


cTplValGesInit< double > & cTrAJ2_GenerateOrient::CorrecDelayGps()
{
   return mCorrecDelayGps;
}

const cTplValGesInit< double > & cTrAJ2_GenerateOrient::CorrecDelayGps()const 
{
   return mCorrecDelayGps;
}


cTplValGesInit< bool > & cTrAJ2_GenerateOrient::ModeMatrix()
{
   return mModeMatrix;
}

const cTplValGesInit< bool > & cTrAJ2_GenerateOrient::ModeMatrix()const 
{
   return mModeMatrix;
}


std::list< std::string > & cTrAJ2_GenerateOrient::KeyName()
{
   return mKeyName;
}

const std::list< std::string > & cTrAJ2_GenerateOrient::KeyName()const 
{
   return mKeyName;
}


cSystemeCoord & cTrAJ2_GenerateOrient::SysCible()
{
   return mSysCible;
}

const cSystemeCoord & cTrAJ2_GenerateOrient::SysCible()const 
{
   return mSysCible;
}


std::string & cTrAJ2_GenerateOrient::NameCalib()
{
   return mNameCalib;
}

const std::string & cTrAJ2_GenerateOrient::NameCalib()const 
{
   return mNameCalib;
}


double & cTrAJ2_GenerateOrient::AltiSol()
{
   return mAltiSol;
}

const double & cTrAJ2_GenerateOrient::AltiSol()const 
{
   return mAltiSol;
}

cElXMLTree * ToXMLTree(const cTrAJ2_GenerateOrient & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"TrAJ2_GenerateOrient",eXMLBranche);
   if (anObj.Teta1FromCap().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Teta1FromCap"),anObj.Teta1FromCap().Val())->ReTagThis("Teta1FromCap"));
   if (anObj.CorrecDelayGps().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("CorrecDelayGps"),anObj.CorrecDelayGps().Val())->ReTagThis("CorrecDelayGps"));
   if (anObj.ModeMatrix().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ModeMatrix"),anObj.ModeMatrix().Val())->ReTagThis("ModeMatrix"));
  for
  (       std::list< std::string >::const_iterator it=anObj.KeyName().begin();
      it !=anObj.KeyName().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("KeyName"),(*it))->ReTagThis("KeyName"));
   aRes->AddFils(ToXMLTree(anObj.SysCible())->ReTagThis("SysCible"));
   aRes->AddFils(::ToXMLTree(std::string("NameCalib"),anObj.NameCalib())->ReTagThis("NameCalib"));
   aRes->AddFils(::ToXMLTree(std::string("AltiSol"),anObj.AltiSol())->ReTagThis("AltiSol"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cTrAJ2_GenerateOrient & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Teta1FromCap(),aTree->Get("Teta1FromCap",1),bool(false)); //tototo 

   xml_init(anObj.CorrecDelayGps(),aTree->Get("CorrecDelayGps",1)); //tototo 

   xml_init(anObj.ModeMatrix(),aTree->Get("ModeMatrix",1),bool(false)); //tototo 

   xml_init(anObj.KeyName(),aTree->GetAll("KeyName",false,1));

   xml_init(anObj.SysCible(),aTree->Get("SysCible",1)); //tototo 

   xml_init(anObj.NameCalib(),aTree->Get("NameCalib",1)); //tototo 

   xml_init(anObj.AltiSol(),aTree->Get("AltiSol",1)); //tototo 
}


double & cTrAJ2_ModeliseVitesse::DeltaTimeMax()
{
   return mDeltaTimeMax;
}

const double & cTrAJ2_ModeliseVitesse::DeltaTimeMax()const 
{
   return mDeltaTimeMax;
}

cElXMLTree * ToXMLTree(const cTrAJ2_ModeliseVitesse & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"TrAJ2_ModeliseVitesse",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("DeltaTimeMax"),anObj.DeltaTimeMax())->ReTagThis("DeltaTimeMax"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cTrAJ2_ModeliseVitesse & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.DeltaTimeMax(),aTree->Get("DeltaTimeMax",1)); //tototo 
}


cTplValGesInit< eConventionsOrientation > & cTrAJ2_SectionImages::ConvOrCam()
{
   return mConvOrCam;
}

const cTplValGesInit< eConventionsOrientation > & cTrAJ2_SectionImages::ConvOrCam()const 
{
   return mConvOrCam;
}


cRotationVect & cTrAJ2_SectionImages::OrientationCamera()
{
   return mOrientationCamera;
}

const cRotationVect & cTrAJ2_SectionImages::OrientationCamera()const 
{
   return mOrientationCamera;
}


std::string & cTrAJ2_SectionImages::KeySetIm()
{
   return mKeySetIm;
}

const std::string & cTrAJ2_SectionImages::KeySetIm()const 
{
   return mKeySetIm;
}


std::string & cTrAJ2_SectionImages::Id()
{
   return mId;
}

const std::string & cTrAJ2_SectionImages::Id()const 
{
   return mId;
}

cElXMLTree * ToXMLTree(const cTrAJ2_SectionImages & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"TrAJ2_SectionImages",eXMLBranche);
   if (anObj.ConvOrCam().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ConvOrCam"),anObj.ConvOrCam().Val())->ReTagThis("ConvOrCam"));
   aRes->AddFils(ToXMLTree(anObj.OrientationCamera())->ReTagThis("OrientationCamera"));
   aRes->AddFils(::ToXMLTree(std::string("KeySetIm"),anObj.KeySetIm())->ReTagThis("KeySetIm"));
   aRes->AddFils(::ToXMLTree(std::string("Id"),anObj.Id())->ReTagThis("Id"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cTrAJ2_SectionImages & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.ConvOrCam(),aTree->Get("ConvOrCam",1),eConventionsOrientation(eConvOriLib)); //tototo 

   xml_init(anObj.OrientationCamera(),aTree->Get("OrientationCamera",1)); //tototo 

   xml_init(anObj.KeySetIm(),aTree->Get("KeySetIm",1)); //tototo 

   xml_init(anObj.Id(),aTree->Get("Id",1)); //tototo 
}


std::string & cGenerateTabExemple::Name()
{
   return mName;
}

const std::string & cGenerateTabExemple::Name()const 
{
   return mName;
}


Pt3di & cGenerateTabExemple::Nb()
{
   return mNb;
}

const Pt3di & cGenerateTabExemple::Nb()const 
{
   return mNb;
}


cTplValGesInit< double > & cGenerateTabExemple::ZMin()
{
   return mZMin;
}

const cTplValGesInit< double > & cGenerateTabExemple::ZMin()const 
{
   return mZMin;
}


cTplValGesInit< double > & cGenerateTabExemple::ZMax()
{
   return mZMax;
}

const cTplValGesInit< double > & cGenerateTabExemple::ZMax()const 
{
   return mZMax;
}


cTplValGesInit< double > & cGenerateTabExemple::DIntervZ()
{
   return mDIntervZ;
}

const cTplValGesInit< double > & cGenerateTabExemple::DIntervZ()const 
{
   return mDIntervZ;
}


cTplValGesInit< bool > & cGenerateTabExemple::RandomXY()
{
   return mRandomXY;
}

const cTplValGesInit< bool > & cGenerateTabExemple::RandomXY()const 
{
   return mRandomXY;
}


cTplValGesInit< bool > & cGenerateTabExemple::RandomZ()
{
   return mRandomZ;
}

const cTplValGesInit< bool > & cGenerateTabExemple::RandomZ()const 
{
   return mRandomZ;
}

cElXMLTree * ToXMLTree(const cGenerateTabExemple & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"GenerateTabExemple",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Name"),anObj.Name())->ReTagThis("Name"));
   aRes->AddFils(::ToXMLTree(std::string("Nb"),anObj.Nb())->ReTagThis("Nb"));
   if (anObj.ZMin().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ZMin"),anObj.ZMin().Val())->ReTagThis("ZMin"));
   if (anObj.ZMax().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ZMax"),anObj.ZMax().Val())->ReTagThis("ZMax"));
   if (anObj.DIntervZ().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DIntervZ"),anObj.DIntervZ().Val())->ReTagThis("DIntervZ"));
   if (anObj.RandomXY().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("RandomXY"),anObj.RandomXY().Val())->ReTagThis("RandomXY"));
   if (anObj.RandomZ().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("RandomZ"),anObj.RandomZ().Val())->ReTagThis("RandomZ"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cGenerateTabExemple & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Name(),aTree->Get("Name",1)); //tototo 

   xml_init(anObj.Nb(),aTree->Get("Nb",1)); //tototo 

   xml_init(anObj.ZMin(),aTree->Get("ZMin",1)); //tototo 

   xml_init(anObj.ZMax(),aTree->Get("ZMax",1)); //tototo 

   xml_init(anObj.DIntervZ(),aTree->Get("DIntervZ",1),double(0.0)); //tototo 

   xml_init(anObj.RandomXY(),aTree->Get("RandomXY",1),bool(true)); //tototo 

   xml_init(anObj.RandomZ(),aTree->Get("RandomZ",1),bool(true)); //tototo 
}


cTplValGesInit< int > & cFullDate::KYear()
{
   return mKYear;
}

const cTplValGesInit< int > & cFullDate::KYear()const 
{
   return mKYear;
}


cTplValGesInit< int > & cFullDate::DefYear()
{
   return mDefYear;
}

const cTplValGesInit< int > & cFullDate::DefYear()const 
{
   return mDefYear;
}


cTplValGesInit< int > & cFullDate::KMonth()
{
   return mKMonth;
}

const cTplValGesInit< int > & cFullDate::KMonth()const 
{
   return mKMonth;
}


cTplValGesInit< int > & cFullDate::DefMonth()
{
   return mDefMonth;
}

const cTplValGesInit< int > & cFullDate::DefMonth()const 
{
   return mDefMonth;
}


cTplValGesInit< int > & cFullDate::KDay()
{
   return mKDay;
}

const cTplValGesInit< int > & cFullDate::KDay()const 
{
   return mKDay;
}


cTplValGesInit< int > & cFullDate::DefDay()
{
   return mDefDay;
}

const cTplValGesInit< int > & cFullDate::DefDay()const 
{
   return mDefDay;
}


int & cFullDate::KHour()
{
   return mKHour;
}

const int & cFullDate::KHour()const 
{
   return mKHour;
}


int & cFullDate::KMin()
{
   return mKMin;
}

const int & cFullDate::KMin()const 
{
   return mKMin;
}


int & cFullDate::KSec()
{
   return mKSec;
}

const int & cFullDate::KSec()const 
{
   return mKSec;
}


cTplValGesInit< double > & cFullDate::DivSec()
{
   return mDivSec;
}

const cTplValGesInit< double > & cFullDate::DivSec()const 
{
   return mDivSec;
}


cTplValGesInit< int > & cFullDate::KMiliSec()
{
   return mKMiliSec;
}

const cTplValGesInit< int > & cFullDate::KMiliSec()const 
{
   return mKMiliSec;
}


cTplValGesInit< double > & cFullDate::DivMiliSec()
{
   return mDivMiliSec;
}

const cTplValGesInit< double > & cFullDate::DivMiliSec()const 
{
   return mDivMiliSec;
}

cElXMLTree * ToXMLTree(const cFullDate & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"FullDate",eXMLBranche);
   if (anObj.KYear().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("KYear"),anObj.KYear().Val())->ReTagThis("KYear"));
   if (anObj.DefYear().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DefYear"),anObj.DefYear().Val())->ReTagThis("DefYear"));
   if (anObj.KMonth().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("KMonth"),anObj.KMonth().Val())->ReTagThis("KMonth"));
   if (anObj.DefMonth().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DefMonth"),anObj.DefMonth().Val())->ReTagThis("DefMonth"));
   if (anObj.KDay().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("KDay"),anObj.KDay().Val())->ReTagThis("KDay"));
   if (anObj.DefDay().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DefDay"),anObj.DefDay().Val())->ReTagThis("DefDay"));
   aRes->AddFils(::ToXMLTree(std::string("KHour"),anObj.KHour())->ReTagThis("KHour"));
   aRes->AddFils(::ToXMLTree(std::string("KMin"),anObj.KMin())->ReTagThis("KMin"));
   aRes->AddFils(::ToXMLTree(std::string("KSec"),anObj.KSec())->ReTagThis("KSec"));
   if (anObj.DivSec().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DivSec"),anObj.DivSec().Val())->ReTagThis("DivSec"));
   if (anObj.KMiliSec().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("KMiliSec"),anObj.KMiliSec().Val())->ReTagThis("KMiliSec"));
   if (anObj.DivMiliSec().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DivMiliSec"),anObj.DivMiliSec().Val())->ReTagThis("DivMiliSec"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cFullDate & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.KYear(),aTree->Get("KYear",1)); //tototo 

   xml_init(anObj.DefYear(),aTree->Get("DefYear",1),int(2011)); //tototo 

   xml_init(anObj.KMonth(),aTree->Get("KMonth",1)); //tototo 

   xml_init(anObj.DefMonth(),aTree->Get("DefMonth",1),int(1)); //tototo 

   xml_init(anObj.KDay(),aTree->Get("KDay",1)); //tototo 

   xml_init(anObj.DefDay(),aTree->Get("DefDay",1),int(1)); //tototo 

   xml_init(anObj.KHour(),aTree->Get("KHour",1)); //tototo 

   xml_init(anObj.KMin(),aTree->Get("KMin",1)); //tototo 

   xml_init(anObj.KSec(),aTree->Get("KSec",1)); //tototo 

   xml_init(anObj.DivSec(),aTree->Get("DivSec",1),double(1.0)); //tototo 

   xml_init(anObj.KMiliSec(),aTree->Get("KMiliSec",1)); //tototo 

   xml_init(anObj.DivMiliSec(),aTree->Get("DivMiliSec",1),double(1.0)); //tototo 
}


cTplValGesInit< std::string > & cSectionTime::NoTime()
{
   return mNoTime;
}

const cTplValGesInit< std::string > & cSectionTime::NoTime()const 
{
   return mNoTime;
}


cTplValGesInit< int > & cSectionTime::KTime()
{
   return mKTime;
}

const cTplValGesInit< int > & cSectionTime::KTime()const 
{
   return mKTime;
}


cTplValGesInit< int > & cSectionTime::KYear()
{
   return FullDate().Val().KYear();
}

const cTplValGesInit< int > & cSectionTime::KYear()const 
{
   return FullDate().Val().KYear();
}


cTplValGesInit< int > & cSectionTime::DefYear()
{
   return FullDate().Val().DefYear();
}

const cTplValGesInit< int > & cSectionTime::DefYear()const 
{
   return FullDate().Val().DefYear();
}


cTplValGesInit< int > & cSectionTime::KMonth()
{
   return FullDate().Val().KMonth();
}

const cTplValGesInit< int > & cSectionTime::KMonth()const 
{
   return FullDate().Val().KMonth();
}


cTplValGesInit< int > & cSectionTime::DefMonth()
{
   return FullDate().Val().DefMonth();
}

const cTplValGesInit< int > & cSectionTime::DefMonth()const 
{
   return FullDate().Val().DefMonth();
}


cTplValGesInit< int > & cSectionTime::KDay()
{
   return FullDate().Val().KDay();
}

const cTplValGesInit< int > & cSectionTime::KDay()const 
{
   return FullDate().Val().KDay();
}


cTplValGesInit< int > & cSectionTime::DefDay()
{
   return FullDate().Val().DefDay();
}

const cTplValGesInit< int > & cSectionTime::DefDay()const 
{
   return FullDate().Val().DefDay();
}


int & cSectionTime::KHour()
{
   return FullDate().Val().KHour();
}

const int & cSectionTime::KHour()const 
{
   return FullDate().Val().KHour();
}


int & cSectionTime::KMin()
{
   return FullDate().Val().KMin();
}

const int & cSectionTime::KMin()const 
{
   return FullDate().Val().KMin();
}


int & cSectionTime::KSec()
{
   return FullDate().Val().KSec();
}

const int & cSectionTime::KSec()const 
{
   return FullDate().Val().KSec();
}


cTplValGesInit< double > & cSectionTime::DivSec()
{
   return FullDate().Val().DivSec();
}

const cTplValGesInit< double > & cSectionTime::DivSec()const 
{
   return FullDate().Val().DivSec();
}


cTplValGesInit< int > & cSectionTime::KMiliSec()
{
   return FullDate().Val().KMiliSec();
}

const cTplValGesInit< int > & cSectionTime::KMiliSec()const 
{
   return FullDate().Val().KMiliSec();
}


cTplValGesInit< double > & cSectionTime::DivMiliSec()
{
   return FullDate().Val().DivMiliSec();
}

const cTplValGesInit< double > & cSectionTime::DivMiliSec()const 
{
   return FullDate().Val().DivMiliSec();
}


cTplValGesInit< cFullDate > & cSectionTime::FullDate()
{
   return mFullDate;
}

const cTplValGesInit< cFullDate > & cSectionTime::FullDate()const 
{
   return mFullDate;
}

cElXMLTree * ToXMLTree(const cSectionTime & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"SectionTime",eXMLBranche);
   if (anObj.NoTime().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NoTime"),anObj.NoTime().Val())->ReTagThis("NoTime"));
   if (anObj.KTime().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("KTime"),anObj.KTime().Val())->ReTagThis("KTime"));
   if (anObj.FullDate().IsInit())
      aRes->AddFils(ToXMLTree(anObj.FullDate().Val())->ReTagThis("FullDate"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cSectionTime & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.NoTime(),aTree->Get("NoTime",1)); //tototo 

   xml_init(anObj.KTime(),aTree->Get("KTime",1)); //tototo 

   xml_init(anObj.FullDate(),aTree->Get("FullDate",1)); //tototo 
}


eUniteAngulaire & cTrajAngles::Unites()
{
   return mUnites;
}

const eUniteAngulaire & cTrajAngles::Unites()const 
{
   return mUnites;
}


eConventionsOrientation & cTrajAngles::ConvOr()
{
   return mConvOr;
}

const eConventionsOrientation & cTrajAngles::ConvOr()const 
{
   return mConvOr;
}


int & cTrajAngles::KTeta1()
{
   return mKTeta1;
}

const int & cTrajAngles::KTeta1()const 
{
   return mKTeta1;
}


int & cTrajAngles::KTeta2()
{
   return mKTeta2;
}

const int & cTrajAngles::KTeta2()const 
{
   return mKTeta2;
}


int & cTrajAngles::KTeta3()
{
   return mKTeta3;
}

const int & cTrajAngles::KTeta3()const 
{
   return mKTeta3;
}


cTplValGesInit< double > & cTrajAngles::OffsetTeta1()
{
   return mOffsetTeta1;
}

const cTplValGesInit< double > & cTrajAngles::OffsetTeta1()const 
{
   return mOffsetTeta1;
}


cTplValGesInit< double > & cTrajAngles::OffsetTeta2()
{
   return mOffsetTeta2;
}

const cTplValGesInit< double > & cTrajAngles::OffsetTeta2()const 
{
   return mOffsetTeta2;
}


cTplValGesInit< double > & cTrajAngles::OffsetTeta3()
{
   return mOffsetTeta3;
}

const cTplValGesInit< double > & cTrajAngles::OffsetTeta3()const 
{
   return mOffsetTeta3;
}


cTplValGesInit< cRotationVect > & cTrajAngles::RefOrTrajI2C()
{
   return mRefOrTrajI2C;
}

const cTplValGesInit< cRotationVect > & cTrajAngles::RefOrTrajI2C()const 
{
   return mRefOrTrajI2C;
}

cElXMLTree * ToXMLTree(const cTrajAngles & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"TrajAngles",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Unites"),anObj.Unites())->ReTagThis("Unites"));
   aRes->AddFils(::ToXMLTree(std::string("ConvOr"),anObj.ConvOr())->ReTagThis("ConvOr"));
   aRes->AddFils(::ToXMLTree(std::string("KTeta1"),anObj.KTeta1())->ReTagThis("KTeta1"));
   aRes->AddFils(::ToXMLTree(std::string("KTeta2"),anObj.KTeta2())->ReTagThis("KTeta2"));
   aRes->AddFils(::ToXMLTree(std::string("KTeta3"),anObj.KTeta3())->ReTagThis("KTeta3"));
   if (anObj.OffsetTeta1().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("OffsetTeta1"),anObj.OffsetTeta1().Val())->ReTagThis("OffsetTeta1"));
   if (anObj.OffsetTeta2().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("OffsetTeta2"),anObj.OffsetTeta2().Val())->ReTagThis("OffsetTeta2"));
   if (anObj.OffsetTeta3().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("OffsetTeta3"),anObj.OffsetTeta3().Val())->ReTagThis("OffsetTeta3"));
   if (anObj.RefOrTrajI2C().IsInit())
      aRes->AddFils(ToXMLTree(anObj.RefOrTrajI2C().Val())->ReTagThis("RefOrTrajI2C"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cTrajAngles & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Unites(),aTree->Get("Unites",1)); //tototo 

   xml_init(anObj.ConvOr(),aTree->Get("ConvOr",1)); //tototo 

   xml_init(anObj.KTeta1(),aTree->Get("KTeta1",1)); //tototo 

   xml_init(anObj.KTeta2(),aTree->Get("KTeta2",1)); //tototo 

   xml_init(anObj.KTeta3(),aTree->Get("KTeta3",1)); //tototo 

   xml_init(anObj.OffsetTeta1(),aTree->Get("OffsetTeta1",1),double(0)); //tototo 

   xml_init(anObj.OffsetTeta2(),aTree->Get("OffsetTeta2",1),double(0)); //tototo 

   xml_init(anObj.OffsetTeta3(),aTree->Get("OffsetTeta3",1),double(0)); //tototo 

   xml_init(anObj.RefOrTrajI2C(),aTree->Get("RefOrTrajI2C",1)); //tototo 
}


int & cGetImInLog::KIm()
{
   return mKIm;
}

const int & cGetImInLog::KIm()const 
{
   return mKIm;
}

cElXMLTree * ToXMLTree(const cGetImInLog & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"GetImInLog",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("KIm"),anObj.KIm())->ReTagThis("KIm"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cGetImInLog & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.KIm(),aTree->Get("KIm",1)); //tototo 
}


std::list< cGenerateTabExemple > & cTrAJ2_SectionLog::GenerateTabExemple()
{
   return mGenerateTabExemple;
}

const std::list< cGenerateTabExemple > & cTrAJ2_SectionLog::GenerateTabExemple()const 
{
   return mGenerateTabExemple;
}


cTplValGesInit< double > & cTrAJ2_SectionLog::TimeMin()
{
   return mTimeMin;
}

const cTplValGesInit< double > & cTrAJ2_SectionLog::TimeMin()const 
{
   return mTimeMin;
}


cTplValGesInit< int > & cTrAJ2_SectionLog::KLogT0()
{
   return mKLogT0;
}

const cTplValGesInit< int > & cTrAJ2_SectionLog::KLogT0()const 
{
   return mKLogT0;
}


std::string & cTrAJ2_SectionLog::File()
{
   return mFile;
}

const std::string & cTrAJ2_SectionLog::File()const 
{
   return mFile;
}


std::string & cTrAJ2_SectionLog::Autom()
{
   return mAutom;
}

const std::string & cTrAJ2_SectionLog::Autom()const 
{
   return mAutom;
}


cSystemeCoord & cTrAJ2_SectionLog::SysCoord()
{
   return mSysCoord;
}

const cSystemeCoord & cTrAJ2_SectionLog::SysCoord()const 
{
   return mSysCoord;
}


std::string & cTrAJ2_SectionLog::Id()
{
   return mId;
}

const std::string & cTrAJ2_SectionLog::Id()const 
{
   return mId;
}


cTplValGesInit< std::string > & cTrAJ2_SectionLog::NoTime()
{
   return SectionTime().NoTime();
}

const cTplValGesInit< std::string > & cTrAJ2_SectionLog::NoTime()const 
{
   return SectionTime().NoTime();
}


cTplValGesInit< int > & cTrAJ2_SectionLog::KTime()
{
   return SectionTime().KTime();
}

const cTplValGesInit< int > & cTrAJ2_SectionLog::KTime()const 
{
   return SectionTime().KTime();
}


cTplValGesInit< int > & cTrAJ2_SectionLog::KYear()
{
   return SectionTime().FullDate().Val().KYear();
}

const cTplValGesInit< int > & cTrAJ2_SectionLog::KYear()const 
{
   return SectionTime().FullDate().Val().KYear();
}


cTplValGesInit< int > & cTrAJ2_SectionLog::DefYear()
{
   return SectionTime().FullDate().Val().DefYear();
}

const cTplValGesInit< int > & cTrAJ2_SectionLog::DefYear()const 
{
   return SectionTime().FullDate().Val().DefYear();
}


cTplValGesInit< int > & cTrAJ2_SectionLog::KMonth()
{
   return SectionTime().FullDate().Val().KMonth();
}

const cTplValGesInit< int > & cTrAJ2_SectionLog::KMonth()const 
{
   return SectionTime().FullDate().Val().KMonth();
}


cTplValGesInit< int > & cTrAJ2_SectionLog::DefMonth()
{
   return SectionTime().FullDate().Val().DefMonth();
}

const cTplValGesInit< int > & cTrAJ2_SectionLog::DefMonth()const 
{
   return SectionTime().FullDate().Val().DefMonth();
}


cTplValGesInit< int > & cTrAJ2_SectionLog::KDay()
{
   return SectionTime().FullDate().Val().KDay();
}

const cTplValGesInit< int > & cTrAJ2_SectionLog::KDay()const 
{
   return SectionTime().FullDate().Val().KDay();
}


cTplValGesInit< int > & cTrAJ2_SectionLog::DefDay()
{
   return SectionTime().FullDate().Val().DefDay();
}

const cTplValGesInit< int > & cTrAJ2_SectionLog::DefDay()const 
{
   return SectionTime().FullDate().Val().DefDay();
}


int & cTrAJ2_SectionLog::KHour()
{
   return SectionTime().FullDate().Val().KHour();
}

const int & cTrAJ2_SectionLog::KHour()const 
{
   return SectionTime().FullDate().Val().KHour();
}


int & cTrAJ2_SectionLog::KMin()
{
   return SectionTime().FullDate().Val().KMin();
}

const int & cTrAJ2_SectionLog::KMin()const 
{
   return SectionTime().FullDate().Val().KMin();
}


int & cTrAJ2_SectionLog::KSec()
{
   return SectionTime().FullDate().Val().KSec();
}

const int & cTrAJ2_SectionLog::KSec()const 
{
   return SectionTime().FullDate().Val().KSec();
}


cTplValGesInit< double > & cTrAJ2_SectionLog::DivSec()
{
   return SectionTime().FullDate().Val().DivSec();
}

const cTplValGesInit< double > & cTrAJ2_SectionLog::DivSec()const 
{
   return SectionTime().FullDate().Val().DivSec();
}


cTplValGesInit< int > & cTrAJ2_SectionLog::KMiliSec()
{
   return SectionTime().FullDate().Val().KMiliSec();
}

const cTplValGesInit< int > & cTrAJ2_SectionLog::KMiliSec()const 
{
   return SectionTime().FullDate().Val().KMiliSec();
}


cTplValGesInit< double > & cTrAJ2_SectionLog::DivMiliSec()
{
   return SectionTime().FullDate().Val().DivMiliSec();
}

const cTplValGesInit< double > & cTrAJ2_SectionLog::DivMiliSec()const 
{
   return SectionTime().FullDate().Val().DivMiliSec();
}


cTplValGesInit< cFullDate > & cTrAJ2_SectionLog::FullDate()
{
   return SectionTime().FullDate();
}

const cTplValGesInit< cFullDate > & cTrAJ2_SectionLog::FullDate()const 
{
   return SectionTime().FullDate();
}


cSectionTime & cTrAJ2_SectionLog::SectionTime()
{
   return mSectionTime;
}

const cSectionTime & cTrAJ2_SectionLog::SectionTime()const 
{
   return mSectionTime;
}


int & cTrAJ2_SectionLog::KCoord1()
{
   return mKCoord1;
}

const int & cTrAJ2_SectionLog::KCoord1()const 
{
   return mKCoord1;
}


cTplValGesInit< double > & cTrAJ2_SectionLog::DivCoord1()
{
   return mDivCoord1;
}

const cTplValGesInit< double > & cTrAJ2_SectionLog::DivCoord1()const 
{
   return mDivCoord1;
}


int & cTrAJ2_SectionLog::KCoord2()
{
   return mKCoord2;
}

const int & cTrAJ2_SectionLog::KCoord2()const 
{
   return mKCoord2;
}


cTplValGesInit< double > & cTrAJ2_SectionLog::DivCoord2()
{
   return mDivCoord2;
}

const cTplValGesInit< double > & cTrAJ2_SectionLog::DivCoord2()const 
{
   return mDivCoord2;
}


int & cTrAJ2_SectionLog::KCoord3()
{
   return mKCoord3;
}

const int & cTrAJ2_SectionLog::KCoord3()const 
{
   return mKCoord3;
}


cTplValGesInit< double > & cTrAJ2_SectionLog::DivCoord3()
{
   return mDivCoord3;
}

const cTplValGesInit< double > & cTrAJ2_SectionLog::DivCoord3()const 
{
   return mDivCoord3;
}


std::vector< eUniteAngulaire > & cTrAJ2_SectionLog::UnitesCoord()
{
   return mUnitesCoord;
}

const std::vector< eUniteAngulaire > & cTrAJ2_SectionLog::UnitesCoord()const 
{
   return mUnitesCoord;
}


eUniteAngulaire & cTrAJ2_SectionLog::Unites()
{
   return TrajAngles().Val().Unites();
}

const eUniteAngulaire & cTrAJ2_SectionLog::Unites()const 
{
   return TrajAngles().Val().Unites();
}


eConventionsOrientation & cTrAJ2_SectionLog::ConvOr()
{
   return TrajAngles().Val().ConvOr();
}

const eConventionsOrientation & cTrAJ2_SectionLog::ConvOr()const 
{
   return TrajAngles().Val().ConvOr();
}


int & cTrAJ2_SectionLog::KTeta1()
{
   return TrajAngles().Val().KTeta1();
}

const int & cTrAJ2_SectionLog::KTeta1()const 
{
   return TrajAngles().Val().KTeta1();
}


int & cTrAJ2_SectionLog::KTeta2()
{
   return TrajAngles().Val().KTeta2();
}

const int & cTrAJ2_SectionLog::KTeta2()const 
{
   return TrajAngles().Val().KTeta2();
}


int & cTrAJ2_SectionLog::KTeta3()
{
   return TrajAngles().Val().KTeta3();
}

const int & cTrAJ2_SectionLog::KTeta3()const 
{
   return TrajAngles().Val().KTeta3();
}


cTplValGesInit< double > & cTrAJ2_SectionLog::OffsetTeta1()
{
   return TrajAngles().Val().OffsetTeta1();
}

const cTplValGesInit< double > & cTrAJ2_SectionLog::OffsetTeta1()const 
{
   return TrajAngles().Val().OffsetTeta1();
}


cTplValGesInit< double > & cTrAJ2_SectionLog::OffsetTeta2()
{
   return TrajAngles().Val().OffsetTeta2();
}

const cTplValGesInit< double > & cTrAJ2_SectionLog::OffsetTeta2()const 
{
   return TrajAngles().Val().OffsetTeta2();
}


cTplValGesInit< double > & cTrAJ2_SectionLog::OffsetTeta3()
{
   return TrajAngles().Val().OffsetTeta3();
}

const cTplValGesInit< double > & cTrAJ2_SectionLog::OffsetTeta3()const 
{
   return TrajAngles().Val().OffsetTeta3();
}


cTplValGesInit< cRotationVect > & cTrAJ2_SectionLog::RefOrTrajI2C()
{
   return TrajAngles().Val().RefOrTrajI2C();
}

const cTplValGesInit< cRotationVect > & cTrAJ2_SectionLog::RefOrTrajI2C()const 
{
   return TrajAngles().Val().RefOrTrajI2C();
}


cTplValGesInit< cTrajAngles > & cTrAJ2_SectionLog::TrajAngles()
{
   return mTrajAngles;
}

const cTplValGesInit< cTrajAngles > & cTrAJ2_SectionLog::TrajAngles()const 
{
   return mTrajAngles;
}


int & cTrAJ2_SectionLog::KIm()
{
   return GetImInLog().Val().KIm();
}

const int & cTrAJ2_SectionLog::KIm()const 
{
   return GetImInLog().Val().KIm();
}


cTplValGesInit< cGetImInLog > & cTrAJ2_SectionLog::GetImInLog()
{
   return mGetImInLog;
}

const cTplValGesInit< cGetImInLog > & cTrAJ2_SectionLog::GetImInLog()const 
{
   return mGetImInLog;
}

cElXMLTree * ToXMLTree(const cTrAJ2_SectionLog & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"TrAJ2_SectionLog",eXMLBranche);
  for
  (       std::list< cGenerateTabExemple >::const_iterator it=anObj.GenerateTabExemple().begin();
      it !=anObj.GenerateTabExemple().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("GenerateTabExemple"));
   if (anObj.TimeMin().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("TimeMin"),anObj.TimeMin().Val())->ReTagThis("TimeMin"));
   if (anObj.KLogT0().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("KLogT0"),anObj.KLogT0().Val())->ReTagThis("KLogT0"));
   aRes->AddFils(::ToXMLTree(std::string("File"),anObj.File())->ReTagThis("File"));
   aRes->AddFils(::ToXMLTree(std::string("Autom"),anObj.Autom())->ReTagThis("Autom"));
   aRes->AddFils(ToXMLTree(anObj.SysCoord())->ReTagThis("SysCoord"));
   aRes->AddFils(::ToXMLTree(std::string("Id"),anObj.Id())->ReTagThis("Id"));
   aRes->AddFils(ToXMLTree(anObj.SectionTime())->ReTagThis("SectionTime"));
   aRes->AddFils(::ToXMLTree(std::string("KCoord1"),anObj.KCoord1())->ReTagThis("KCoord1"));
   if (anObj.DivCoord1().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DivCoord1"),anObj.DivCoord1().Val())->ReTagThis("DivCoord1"));
   aRes->AddFils(::ToXMLTree(std::string("KCoord2"),anObj.KCoord2())->ReTagThis("KCoord2"));
   if (anObj.DivCoord2().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DivCoord2"),anObj.DivCoord2().Val())->ReTagThis("DivCoord2"));
   aRes->AddFils(::ToXMLTree(std::string("KCoord3"),anObj.KCoord3())->ReTagThis("KCoord3"));
   if (anObj.DivCoord3().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DivCoord3"),anObj.DivCoord3().Val())->ReTagThis("DivCoord3"));
  for
  (       std::vector< eUniteAngulaire >::const_iterator it=anObj.UnitesCoord().begin();
      it !=anObj.UnitesCoord().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("UnitesCoord"),(*it))->ReTagThis("UnitesCoord"));
   if (anObj.TrajAngles().IsInit())
      aRes->AddFils(ToXMLTree(anObj.TrajAngles().Val())->ReTagThis("TrajAngles"));
   if (anObj.GetImInLog().IsInit())
      aRes->AddFils(ToXMLTree(anObj.GetImInLog().Val())->ReTagThis("GetImInLog"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cTrAJ2_SectionLog & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.GenerateTabExemple(),aTree->GetAll("GenerateTabExemple",false,1));

   xml_init(anObj.TimeMin(),aTree->Get("TimeMin",1)); //tototo 

   xml_init(anObj.KLogT0(),aTree->Get("KLogT0",1),int(0)); //tototo 

   xml_init(anObj.File(),aTree->Get("File",1)); //tototo 

   xml_init(anObj.Autom(),aTree->Get("Autom",1)); //tototo 

   xml_init(anObj.SysCoord(),aTree->Get("SysCoord",1)); //tototo 

   xml_init(anObj.Id(),aTree->Get("Id",1)); //tototo 

   xml_init(anObj.SectionTime(),aTree->Get("SectionTime",1)); //tototo 

   xml_init(anObj.KCoord1(),aTree->Get("KCoord1",1)); //tototo 

   xml_init(anObj.DivCoord1(),aTree->Get("DivCoord1",1),double(1.0)); //tototo 

   xml_init(anObj.KCoord2(),aTree->Get("KCoord2",1)); //tototo 

   xml_init(anObj.DivCoord2(),aTree->Get("DivCoord2",1),double(1.0)); //tototo 

   xml_init(anObj.KCoord3(),aTree->Get("KCoord3",1)); //tototo 

   xml_init(anObj.DivCoord3(),aTree->Get("DivCoord3",1),double(1.0)); //tototo 

   xml_init(anObj.UnitesCoord(),aTree->GetAll("UnitesCoord",false,1));

   xml_init(anObj.TrajAngles(),aTree->Get("TrajAngles",1)); //tototo 

   xml_init(anObj.GetImInLog(),aTree->Get("GetImInLog",1)); //tototo 
}


std::string & cLearnByExample::Im0()
{
   return mIm0;
}

const std::string & cLearnByExample::Im0()const 
{
   return mIm0;
}


int & cLearnByExample::Log0()
{
   return mLog0;
}

const int & cLearnByExample::Log0()const 
{
   return mLog0;
}


int & cLearnByExample::DeltaMinRech()
{
   return mDeltaMinRech;
}

const int & cLearnByExample::DeltaMinRech()const 
{
   return mDeltaMinRech;
}


int & cLearnByExample::DeltaMaxRech()
{
   return mDeltaMaxRech;
}

const int & cLearnByExample::DeltaMaxRech()const 
{
   return mDeltaMaxRech;
}


cTplValGesInit< bool > & cLearnByExample::Show()
{
   return mShow;
}

const cTplValGesInit< bool > & cLearnByExample::Show()const 
{
   return mShow;
}


cTplValGesInit< bool > & cLearnByExample::ShowPerc()
{
   return mShowPerc;
}

const cTplValGesInit< bool > & cLearnByExample::ShowPerc()const 
{
   return mShowPerc;
}

cElXMLTree * ToXMLTree(const cLearnByExample & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"LearnByExample",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Im0"),anObj.Im0())->ReTagThis("Im0"));
   aRes->AddFils(::ToXMLTree(std::string("Log0"),anObj.Log0())->ReTagThis("Log0"));
   aRes->AddFils(::ToXMLTree(std::string("DeltaMinRech"),anObj.DeltaMinRech())->ReTagThis("DeltaMinRech"));
   aRes->AddFils(::ToXMLTree(std::string("DeltaMaxRech"),anObj.DeltaMaxRech())->ReTagThis("DeltaMaxRech"));
   if (anObj.Show().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Show"),anObj.Show().Val())->ReTagThis("Show"));
   if (anObj.ShowPerc().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ShowPerc"),anObj.ShowPerc().Val())->ReTagThis("ShowPerc"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cLearnByExample & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Im0(),aTree->Get("Im0",1)); //tototo 

   xml_init(anObj.Log0(),aTree->Get("Log0",1)); //tototo 

   xml_init(anObj.DeltaMinRech(),aTree->Get("DeltaMinRech",1)); //tototo 

   xml_init(anObj.DeltaMaxRech(),aTree->Get("DeltaMaxRech",1)); //tototo 

   xml_init(anObj.Show(),aTree->Get("Show",1),bool(false)); //tototo 

   xml_init(anObj.ShowPerc(),aTree->Get("ShowPerc",1),bool(true)); //tototo 
}


cTplValGesInit< double > & cLearnByStatDiff::MaxEcart()
{
   return mMaxEcart;
}

const cTplValGesInit< double > & cLearnByStatDiff::MaxEcart()const 
{
   return mMaxEcart;
}

cElXMLTree * ToXMLTree(const cLearnByStatDiff & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"LearnByStatDiff",eXMLBranche);
   if (anObj.MaxEcart().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("MaxEcart"),anObj.MaxEcart().Val())->ReTagThis("MaxEcart"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cLearnByStatDiff & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.MaxEcart(),aTree->Get("MaxEcart",1),double(0.52)); //tototo 
}


std::string & cLearnOffset::Im0()
{
   return LearnByExample().Val().Im0();
}

const std::string & cLearnOffset::Im0()const 
{
   return LearnByExample().Val().Im0();
}


int & cLearnOffset::Log0()
{
   return LearnByExample().Val().Log0();
}

const int & cLearnOffset::Log0()const 
{
   return LearnByExample().Val().Log0();
}


int & cLearnOffset::DeltaMinRech()
{
   return LearnByExample().Val().DeltaMinRech();
}

const int & cLearnOffset::DeltaMinRech()const 
{
   return LearnByExample().Val().DeltaMinRech();
}


int & cLearnOffset::DeltaMaxRech()
{
   return LearnByExample().Val().DeltaMaxRech();
}

const int & cLearnOffset::DeltaMaxRech()const 
{
   return LearnByExample().Val().DeltaMaxRech();
}


cTplValGesInit< bool > & cLearnOffset::Show()
{
   return LearnByExample().Val().Show();
}

const cTplValGesInit< bool > & cLearnOffset::Show()const 
{
   return LearnByExample().Val().Show();
}


cTplValGesInit< bool > & cLearnOffset::ShowPerc()
{
   return LearnByExample().Val().ShowPerc();
}

const cTplValGesInit< bool > & cLearnOffset::ShowPerc()const 
{
   return LearnByExample().Val().ShowPerc();
}


cTplValGesInit< cLearnByExample > & cLearnOffset::LearnByExample()
{
   return mLearnByExample;
}

const cTplValGesInit< cLearnByExample > & cLearnOffset::LearnByExample()const 
{
   return mLearnByExample;
}


cTplValGesInit< double > & cLearnOffset::MaxEcart()
{
   return LearnByStatDiff().Val().MaxEcart();
}

const cTplValGesInit< double > & cLearnOffset::MaxEcart()const 
{
   return LearnByStatDiff().Val().MaxEcart();
}


cTplValGesInit< cLearnByStatDiff > & cLearnOffset::LearnByStatDiff()
{
   return mLearnByStatDiff;
}

const cTplValGesInit< cLearnByStatDiff > & cLearnOffset::LearnByStatDiff()const 
{
   return mLearnByStatDiff;
}

cElXMLTree * ToXMLTree(const cLearnOffset & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"LearnOffset",eXMLBranche);
   if (anObj.LearnByExample().IsInit())
      aRes->AddFils(ToXMLTree(anObj.LearnByExample().Val())->ReTagThis("LearnByExample"));
   if (anObj.LearnByStatDiff().IsInit())
      aRes->AddFils(ToXMLTree(anObj.LearnByStatDiff().Val())->ReTagThis("LearnByStatDiff"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cLearnOffset & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.LearnByExample(),aTree->Get("LearnByExample",1)); //tototo 

   xml_init(anObj.LearnByStatDiff(),aTree->Get("LearnByStatDiff",1)); //tototo 
}


double & cMatchNearestIm::TolMatch()
{
   return mTolMatch;
}

const double & cMatchNearestIm::TolMatch()const 
{
   return mTolMatch;
}


double & cMatchNearestIm::TolAmbig()
{
   return mTolAmbig;
}

const double & cMatchNearestIm::TolAmbig()const 
{
   return mTolAmbig;
}

cElXMLTree * ToXMLTree(const cMatchNearestIm & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"MatchNearestIm",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("TolMatch"),anObj.TolMatch())->ReTagThis("TolMatch"));
   aRes->AddFils(::ToXMLTree(std::string("TolAmbig"),anObj.TolAmbig())->ReTagThis("TolAmbig"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cMatchNearestIm & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.TolMatch(),aTree->Get("TolMatch",1)); //tototo 

   xml_init(anObj.TolAmbig(),aTree->Get("TolAmbig",1)); //tototo 
}


std::string & cMatchByName::KeyLog2Im()
{
   return mKeyLog2Im;
}

const std::string & cMatchByName::KeyLog2Im()const 
{
   return mKeyLog2Im;
}

cElXMLTree * ToXMLTree(const cMatchByName & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"MatchByName",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("KeyLog2Im"),anObj.KeyLog2Im())->ReTagThis("KeyLog2Im"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cMatchByName & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.KeyLog2Im(),aTree->Get("KeyLog2Im",1)); //tototo 
}


double & cAlgoMatch::TolMatch()
{
   return MatchNearestIm().Val().TolMatch();
}

const double & cAlgoMatch::TolMatch()const 
{
   return MatchNearestIm().Val().TolMatch();
}


double & cAlgoMatch::TolAmbig()
{
   return MatchNearestIm().Val().TolAmbig();
}

const double & cAlgoMatch::TolAmbig()const 
{
   return MatchNearestIm().Val().TolAmbig();
}


cTplValGesInit< cMatchNearestIm > & cAlgoMatch::MatchNearestIm()
{
   return mMatchNearestIm;
}

const cTplValGesInit< cMatchNearestIm > & cAlgoMatch::MatchNearestIm()const 
{
   return mMatchNearestIm;
}


std::string & cAlgoMatch::KeyLog2Im()
{
   return MatchByName().Val().KeyLog2Im();
}

const std::string & cAlgoMatch::KeyLog2Im()const 
{
   return MatchByName().Val().KeyLog2Im();
}


cTplValGesInit< cMatchByName > & cAlgoMatch::MatchByName()
{
   return mMatchByName;
}

const cTplValGesInit< cMatchByName > & cAlgoMatch::MatchByName()const 
{
   return mMatchByName;
}

cElXMLTree * ToXMLTree(const cAlgoMatch & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"AlgoMatch",eXMLBranche);
   if (anObj.MatchNearestIm().IsInit())
      aRes->AddFils(ToXMLTree(anObj.MatchNearestIm().Val())->ReTagThis("MatchNearestIm"));
   if (anObj.MatchByName().IsInit())
      aRes->AddFils(ToXMLTree(anObj.MatchByName().Val())->ReTagThis("MatchByName"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cAlgoMatch & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.MatchNearestIm(),aTree->Get("MatchNearestIm",1)); //tototo 

   xml_init(anObj.MatchByName(),aTree->Get("MatchByName",1)); //tototo 
}


std::string & cTrAJ2_SectionMatch::IdIm()
{
   return mIdIm;
}

const std::string & cTrAJ2_SectionMatch::IdIm()const 
{
   return mIdIm;
}


std::string & cTrAJ2_SectionMatch::IdLog()
{
   return mIdLog;
}

const std::string & cTrAJ2_SectionMatch::IdLog()const 
{
   return mIdLog;
}


std::string & cTrAJ2_SectionMatch::Im0()
{
   return LearnOffset().Val().LearnByExample().Val().Im0();
}

const std::string & cTrAJ2_SectionMatch::Im0()const 
{
   return LearnOffset().Val().LearnByExample().Val().Im0();
}


int & cTrAJ2_SectionMatch::Log0()
{
   return LearnOffset().Val().LearnByExample().Val().Log0();
}

const int & cTrAJ2_SectionMatch::Log0()const 
{
   return LearnOffset().Val().LearnByExample().Val().Log0();
}


int & cTrAJ2_SectionMatch::DeltaMinRech()
{
   return LearnOffset().Val().LearnByExample().Val().DeltaMinRech();
}

const int & cTrAJ2_SectionMatch::DeltaMinRech()const 
{
   return LearnOffset().Val().LearnByExample().Val().DeltaMinRech();
}


int & cTrAJ2_SectionMatch::DeltaMaxRech()
{
   return LearnOffset().Val().LearnByExample().Val().DeltaMaxRech();
}

const int & cTrAJ2_SectionMatch::DeltaMaxRech()const 
{
   return LearnOffset().Val().LearnByExample().Val().DeltaMaxRech();
}


cTplValGesInit< bool > & cTrAJ2_SectionMatch::Show()
{
   return LearnOffset().Val().LearnByExample().Val().Show();
}

const cTplValGesInit< bool > & cTrAJ2_SectionMatch::Show()const 
{
   return LearnOffset().Val().LearnByExample().Val().Show();
}


cTplValGesInit< bool > & cTrAJ2_SectionMatch::ShowPerc()
{
   return LearnOffset().Val().LearnByExample().Val().ShowPerc();
}

const cTplValGesInit< bool > & cTrAJ2_SectionMatch::ShowPerc()const 
{
   return LearnOffset().Val().LearnByExample().Val().ShowPerc();
}


cTplValGesInit< cLearnByExample > & cTrAJ2_SectionMatch::LearnByExample()
{
   return LearnOffset().Val().LearnByExample();
}

const cTplValGesInit< cLearnByExample > & cTrAJ2_SectionMatch::LearnByExample()const 
{
   return LearnOffset().Val().LearnByExample();
}


cTplValGesInit< double > & cTrAJ2_SectionMatch::MaxEcart()
{
   return LearnOffset().Val().LearnByStatDiff().Val().MaxEcart();
}

const cTplValGesInit< double > & cTrAJ2_SectionMatch::MaxEcart()const 
{
   return LearnOffset().Val().LearnByStatDiff().Val().MaxEcart();
}


cTplValGesInit< cLearnByStatDiff > & cTrAJ2_SectionMatch::LearnByStatDiff()
{
   return LearnOffset().Val().LearnByStatDiff();
}

const cTplValGesInit< cLearnByStatDiff > & cTrAJ2_SectionMatch::LearnByStatDiff()const 
{
   return LearnOffset().Val().LearnByStatDiff();
}


cTplValGesInit< cLearnOffset > & cTrAJ2_SectionMatch::LearnOffset()
{
   return mLearnOffset;
}

const cTplValGesInit< cLearnOffset > & cTrAJ2_SectionMatch::LearnOffset()const 
{
   return mLearnOffset;
}


double & cTrAJ2_SectionMatch::TolMatch()
{
   return AlgoMatch().MatchNearestIm().Val().TolMatch();
}

const double & cTrAJ2_SectionMatch::TolMatch()const 
{
   return AlgoMatch().MatchNearestIm().Val().TolMatch();
}


double & cTrAJ2_SectionMatch::TolAmbig()
{
   return AlgoMatch().MatchNearestIm().Val().TolAmbig();
}

const double & cTrAJ2_SectionMatch::TolAmbig()const 
{
   return AlgoMatch().MatchNearestIm().Val().TolAmbig();
}


cTplValGesInit< cMatchNearestIm > & cTrAJ2_SectionMatch::MatchNearestIm()
{
   return AlgoMatch().MatchNearestIm();
}

const cTplValGesInit< cMatchNearestIm > & cTrAJ2_SectionMatch::MatchNearestIm()const 
{
   return AlgoMatch().MatchNearestIm();
}


std::string & cTrAJ2_SectionMatch::KeyLog2Im()
{
   return AlgoMatch().MatchByName().Val().KeyLog2Im();
}

const std::string & cTrAJ2_SectionMatch::KeyLog2Im()const 
{
   return AlgoMatch().MatchByName().Val().KeyLog2Im();
}


cTplValGesInit< cMatchByName > & cTrAJ2_SectionMatch::MatchByName()
{
   return AlgoMatch().MatchByName();
}

const cTplValGesInit< cMatchByName > & cTrAJ2_SectionMatch::MatchByName()const 
{
   return AlgoMatch().MatchByName();
}


cAlgoMatch & cTrAJ2_SectionMatch::AlgoMatch()
{
   return mAlgoMatch;
}

const cAlgoMatch & cTrAJ2_SectionMatch::AlgoMatch()const 
{
   return mAlgoMatch;
}


cTplValGesInit< cTrAJ2_ModeliseVitesse > & cTrAJ2_SectionMatch::ModeliseVitesse()
{
   return mModeliseVitesse;
}

const cTplValGesInit< cTrAJ2_ModeliseVitesse > & cTrAJ2_SectionMatch::ModeliseVitesse()const 
{
   return mModeliseVitesse;
}


cTplValGesInit< cTrAJ2_GenerateOrient > & cTrAJ2_SectionMatch::GenerateOrient()
{
   return mGenerateOrient;
}

const cTplValGesInit< cTrAJ2_GenerateOrient > & cTrAJ2_SectionMatch::GenerateOrient()const 
{
   return mGenerateOrient;
}

cElXMLTree * ToXMLTree(const cTrAJ2_SectionMatch & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"TrAJ2_SectionMatch",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("IdIm"),anObj.IdIm())->ReTagThis("IdIm"));
   aRes->AddFils(::ToXMLTree(std::string("IdLog"),anObj.IdLog())->ReTagThis("IdLog"));
   if (anObj.LearnOffset().IsInit())
      aRes->AddFils(ToXMLTree(anObj.LearnOffset().Val())->ReTagThis("LearnOffset"));
   aRes->AddFils(ToXMLTree(anObj.AlgoMatch())->ReTagThis("AlgoMatch"));
   if (anObj.ModeliseVitesse().IsInit())
      aRes->AddFils(ToXMLTree(anObj.ModeliseVitesse().Val())->ReTagThis("ModeliseVitesse"));
   if (anObj.GenerateOrient().IsInit())
      aRes->AddFils(ToXMLTree(anObj.GenerateOrient().Val())->ReTagThis("GenerateOrient"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cTrAJ2_SectionMatch & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.IdIm(),aTree->Get("IdIm",1)); //tototo 

   xml_init(anObj.IdLog(),aTree->Get("IdLog",1)); //tototo 

   xml_init(anObj.LearnOffset(),aTree->Get("LearnOffset",1)); //tototo 

   xml_init(anObj.AlgoMatch(),aTree->Get("AlgoMatch",1)); //tototo 

   xml_init(anObj.ModeliseVitesse(),aTree->Get("ModeliseVitesse",1)); //tototo 

   xml_init(anObj.GenerateOrient(),aTree->Get("GenerateOrient",1)); //tototo 
}


std::string & cTraJ2_FilesInputi_Appuis::KeySetOrPat()
{
   return mKeySetOrPat;
}

const std::string & cTraJ2_FilesInputi_Appuis::KeySetOrPat()const 
{
   return mKeySetOrPat;
}


cElRegex_Ptr & cTraJ2_FilesInputi_Appuis::Autom()
{
   return mAutom;
}

const cElRegex_Ptr & cTraJ2_FilesInputi_Appuis::Autom()const 
{
   return mAutom;
}


bool & cTraJ2_FilesInputi_Appuis::GetMesTer()
{
   return mGetMesTer;
}

const bool & cTraJ2_FilesInputi_Appuis::GetMesTer()const 
{
   return mGetMesTer;
}


bool & cTraJ2_FilesInputi_Appuis::GetMesIm()
{
   return mGetMesIm;
}

const bool & cTraJ2_FilesInputi_Appuis::GetMesIm()const 
{
   return mGetMesIm;
}


int & cTraJ2_FilesInputi_Appuis::KIdPt()
{
   return mKIdPt;
}

const int & cTraJ2_FilesInputi_Appuis::KIdPt()const 
{
   return mKIdPt;
}

cElXMLTree * ToXMLTree(const cTraJ2_FilesInputi_Appuis & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"TraJ2_FilesInputi_Appuis",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("KeySetOrPat"),anObj.KeySetOrPat())->ReTagThis("KeySetOrPat"));
   aRes->AddFils(::ToXMLTree(std::string("Autom"),anObj.Autom())->ReTagThis("Autom"));
   aRes->AddFils(::ToXMLTree(std::string("GetMesTer"),anObj.GetMesTer())->ReTagThis("GetMesTer"));
   aRes->AddFils(::ToXMLTree(std::string("GetMesIm"),anObj.GetMesIm())->ReTagThis("GetMesIm"));
   aRes->AddFils(::ToXMLTree(std::string("KIdPt"),anObj.KIdPt())->ReTagThis("KIdPt"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cTraJ2_FilesInputi_Appuis & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.KeySetOrPat(),aTree->Get("KeySetOrPat",1)); //tototo 

   xml_init(anObj.Autom(),aTree->Get("Autom",1)); //tototo 

   xml_init(anObj.GetMesTer(),aTree->Get("GetMesTer",1)); //tototo 

   xml_init(anObj.GetMesIm(),aTree->Get("GetMesIm",1)); //tototo 

   xml_init(anObj.KIdPt(),aTree->Get("KIdPt",1)); //tototo 
}


std::string & cTrAJ2_ConvertionAppuis::Id()
{
   return mId;
}

const std::string & cTrAJ2_ConvertionAppuis::Id()const 
{
   return mId;
}


std::list< cTraJ2_FilesInputi_Appuis > & cTrAJ2_ConvertionAppuis::TraJ2_FilesInputi_Appuis()
{
   return mTraJ2_FilesInputi_Appuis;
}

const std::list< cTraJ2_FilesInputi_Appuis > & cTrAJ2_ConvertionAppuis::TraJ2_FilesInputi_Appuis()const 
{
   return mTraJ2_FilesInputi_Appuis;
}


cTplValGesInit< std::string > & cTrAJ2_ConvertionAppuis::OutMesTer()
{
   return mOutMesTer;
}

const cTplValGesInit< std::string > & cTrAJ2_ConvertionAppuis::OutMesTer()const 
{
   return mOutMesTer;
}


cTplValGesInit< std::string > & cTrAJ2_ConvertionAppuis::OutMesIm()
{
   return mOutMesIm;
}

const cTplValGesInit< std::string > & cTrAJ2_ConvertionAppuis::OutMesIm()const 
{
   return mOutMesIm;
}


cTplValGesInit< cElRegex_Ptr > & cTrAJ2_ConvertionAppuis::AutomComment()
{
   return mAutomComment;
}

const cTplValGesInit< cElRegex_Ptr > & cTrAJ2_ConvertionAppuis::AutomComment()const 
{
   return mAutomComment;
}


std::vector< eUniteAngulaire > & cTrAJ2_ConvertionAppuis::UnitesCoord()
{
   return mUnitesCoord;
}

const std::vector< eUniteAngulaire > & cTrAJ2_ConvertionAppuis::UnitesCoord()const 
{
   return mUnitesCoord;
}


cTplValGesInit< int > & cTrAJ2_ConvertionAppuis::KIncertPlani()
{
   return mKIncertPlani;
}

const cTplValGesInit< int > & cTrAJ2_ConvertionAppuis::KIncertPlani()const 
{
   return mKIncertPlani;
}


cTplValGesInit< int > & cTrAJ2_ConvertionAppuis::KIncertAlti()
{
   return mKIncertAlti;
}

const cTplValGesInit< int > & cTrAJ2_ConvertionAppuis::KIncertAlti()const 
{
   return mKIncertAlti;
}


cTplValGesInit< double > & cTrAJ2_ConvertionAppuis::ValIncertPlani()
{
   return mValIncertPlani;
}

const cTplValGesInit< double > & cTrAJ2_ConvertionAppuis::ValIncertPlani()const 
{
   return mValIncertPlani;
}


cTplValGesInit< double > & cTrAJ2_ConvertionAppuis::ValIncertAlti()
{
   return mValIncertAlti;
}

const cTplValGesInit< double > & cTrAJ2_ConvertionAppuis::ValIncertAlti()const 
{
   return mValIncertAlti;
}


int & cTrAJ2_ConvertionAppuis::KxTer()
{
   return mKxTer;
}

const int & cTrAJ2_ConvertionAppuis::KxTer()const 
{
   return mKxTer;
}


int & cTrAJ2_ConvertionAppuis::KyTer()
{
   return mKyTer;
}

const int & cTrAJ2_ConvertionAppuis::KyTer()const 
{
   return mKyTer;
}


int & cTrAJ2_ConvertionAppuis::KzTer()
{
   return mKzTer;
}

const int & cTrAJ2_ConvertionAppuis::KzTer()const 
{
   return mKzTer;
}


int & cTrAJ2_ConvertionAppuis::KIIm()
{
   return mKIIm;
}

const int & cTrAJ2_ConvertionAppuis::KIIm()const 
{
   return mKIIm;
}


int & cTrAJ2_ConvertionAppuis::KJIm()
{
   return mKJIm;
}

const int & cTrAJ2_ConvertionAppuis::KJIm()const 
{
   return mKJIm;
}


int & cTrAJ2_ConvertionAppuis::KIdIm()
{
   return mKIdIm;
}

const int & cTrAJ2_ConvertionAppuis::KIdIm()const 
{
   return mKIdIm;
}


cTplValGesInit< Pt2di > & cTrAJ2_ConvertionAppuis::OffsetIm()
{
   return mOffsetIm;
}

const cTplValGesInit< Pt2di > & cTrAJ2_ConvertionAppuis::OffsetIm()const 
{
   return mOffsetIm;
}


std::string & cTrAJ2_ConvertionAppuis::KeyId2Im()
{
   return mKeyId2Im;
}

const std::string & cTrAJ2_ConvertionAppuis::KeyId2Im()const 
{
   return mKeyId2Im;
}


cSystemeCoord & cTrAJ2_ConvertionAppuis::SystemeIn()
{
   return mSystemeIn;
}

const cSystemeCoord & cTrAJ2_ConvertionAppuis::SystemeIn()const 
{
   return mSystemeIn;
}


cSystemeCoord & cTrAJ2_ConvertionAppuis::SystemeOut()
{
   return mSystemeOut;
}

const cSystemeCoord & cTrAJ2_ConvertionAppuis::SystemeOut()const 
{
   return mSystemeOut;
}

cElXMLTree * ToXMLTree(const cTrAJ2_ConvertionAppuis & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"TrAJ2_ConvertionAppuis",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Id"),anObj.Id())->ReTagThis("Id"));
  for
  (       std::list< cTraJ2_FilesInputi_Appuis >::const_iterator it=anObj.TraJ2_FilesInputi_Appuis().begin();
      it !=anObj.TraJ2_FilesInputi_Appuis().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("TraJ2_FilesInputi_Appuis"));
   if (anObj.OutMesTer().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("OutMesTer"),anObj.OutMesTer().Val())->ReTagThis("OutMesTer"));
   if (anObj.OutMesIm().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("OutMesIm"),anObj.OutMesIm().Val())->ReTagThis("OutMesIm"));
   if (anObj.AutomComment().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("AutomComment"),anObj.AutomComment().Val())->ReTagThis("AutomComment"));
  for
  (       std::vector< eUniteAngulaire >::const_iterator it=anObj.UnitesCoord().begin();
      it !=anObj.UnitesCoord().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("UnitesCoord"),(*it))->ReTagThis("UnitesCoord"));
   if (anObj.KIncertPlani().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("KIncertPlani"),anObj.KIncertPlani().Val())->ReTagThis("KIncertPlani"));
   if (anObj.KIncertAlti().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("KIncertAlti"),anObj.KIncertAlti().Val())->ReTagThis("KIncertAlti"));
   if (anObj.ValIncertPlani().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ValIncertPlani"),anObj.ValIncertPlani().Val())->ReTagThis("ValIncertPlani"));
   if (anObj.ValIncertAlti().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ValIncertAlti"),anObj.ValIncertAlti().Val())->ReTagThis("ValIncertAlti"));
   aRes->AddFils(::ToXMLTree(std::string("KxTer"),anObj.KxTer())->ReTagThis("KxTer"));
   aRes->AddFils(::ToXMLTree(std::string("KyTer"),anObj.KyTer())->ReTagThis("KyTer"));
   aRes->AddFils(::ToXMLTree(std::string("KzTer"),anObj.KzTer())->ReTagThis("KzTer"));
   aRes->AddFils(::ToXMLTree(std::string("KIIm"),anObj.KIIm())->ReTagThis("KIIm"));
   aRes->AddFils(::ToXMLTree(std::string("KJIm"),anObj.KJIm())->ReTagThis("KJIm"));
   aRes->AddFils(::ToXMLTree(std::string("KIdIm"),anObj.KIdIm())->ReTagThis("KIdIm"));
   if (anObj.OffsetIm().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("OffsetIm"),anObj.OffsetIm().Val())->ReTagThis("OffsetIm"));
   aRes->AddFils(::ToXMLTree(std::string("KeyId2Im"),anObj.KeyId2Im())->ReTagThis("KeyId2Im"));
   aRes->AddFils(ToXMLTree(anObj.SystemeIn())->ReTagThis("SystemeIn"));
   aRes->AddFils(ToXMLTree(anObj.SystemeOut())->ReTagThis("SystemeOut"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cTrAJ2_ConvertionAppuis & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Id(),aTree->Get("Id",1)); //tototo 

   xml_init(anObj.TraJ2_FilesInputi_Appuis(),aTree->GetAll("TraJ2_FilesInputi_Appuis",false,1));

   xml_init(anObj.OutMesTer(),aTree->Get("OutMesTer",1)); //tototo 

   xml_init(anObj.OutMesIm(),aTree->Get("OutMesIm",1)); //tototo 

   xml_init(anObj.AutomComment(),aTree->Get("AutomComment",1)); //tototo 

   xml_init(anObj.UnitesCoord(),aTree->GetAll("UnitesCoord",false,1));

   xml_init(anObj.KIncertPlani(),aTree->Get("KIncertPlani",1),int(-1)); //tototo 

   xml_init(anObj.KIncertAlti(),aTree->Get("KIncertAlti",1),int(-1)); //tototo 

   xml_init(anObj.ValIncertPlani(),aTree->Get("ValIncertPlani",1),double(1.0)); //tototo 

   xml_init(anObj.ValIncertAlti(),aTree->Get("ValIncertAlti",1),double(1.0)); //tototo 

   xml_init(anObj.KxTer(),aTree->Get("KxTer",1)); //tototo 

   xml_init(anObj.KyTer(),aTree->Get("KyTer",1)); //tototo 

   xml_init(anObj.KzTer(),aTree->Get("KzTer",1)); //tototo 

   xml_init(anObj.KIIm(),aTree->Get("KIIm",1)); //tototo 

   xml_init(anObj.KJIm(),aTree->Get("KJIm",1)); //tototo 

   xml_init(anObj.KIdIm(),aTree->Get("KIdIm",1)); //tototo 

   xml_init(anObj.OffsetIm(),aTree->Get("OffsetIm",1),Pt2di(Pt2di(0,0))); //tototo 

   xml_init(anObj.KeyId2Im(),aTree->Get("KeyId2Im",1)); //tototo 

   xml_init(anObj.SystemeIn(),aTree->Get("SystemeIn",1)); //tototo 

   xml_init(anObj.SystemeOut(),aTree->Get("SystemeOut",1)); //tototo 
}


std::string & cTrAJ2_ExportProjImage::NameFileOut()
{
   return mNameFileOut;
}

const std::string & cTrAJ2_ExportProjImage::NameFileOut()const 
{
   return mNameFileOut;
}


std::string & cTrAJ2_ExportProjImage::KeySetOrPatIm()
{
   return mKeySetOrPatIm;
}

const std::string & cTrAJ2_ExportProjImage::KeySetOrPatIm()const 
{
   return mKeySetOrPatIm;
}


std::string & cTrAJ2_ExportProjImage::NameAppuis()
{
   return mNameAppuis;
}

const std::string & cTrAJ2_ExportProjImage::NameAppuis()const 
{
   return mNameAppuis;
}


std::string & cTrAJ2_ExportProjImage::KeyAssocIm2Or()
{
   return mKeyAssocIm2Or;
}

const std::string & cTrAJ2_ExportProjImage::KeyAssocIm2Or()const 
{
   return mKeyAssocIm2Or;
}


cTplValGesInit< std::string > & cTrAJ2_ExportProjImage::KeyGenerateTxt()
{
   return mKeyGenerateTxt;
}

const cTplValGesInit< std::string > & cTrAJ2_ExportProjImage::KeyGenerateTxt()const 
{
   return mKeyGenerateTxt;
}

cElXMLTree * ToXMLTree(const cTrAJ2_ExportProjImage & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"TrAJ2_ExportProjImage",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("NameFileOut"),anObj.NameFileOut())->ReTagThis("NameFileOut"));
   aRes->AddFils(::ToXMLTree(std::string("KeySetOrPatIm"),anObj.KeySetOrPatIm())->ReTagThis("KeySetOrPatIm"));
   aRes->AddFils(::ToXMLTree(std::string("NameAppuis"),anObj.NameAppuis())->ReTagThis("NameAppuis"));
   aRes->AddFils(::ToXMLTree(std::string("KeyAssocIm2Or"),anObj.KeyAssocIm2Or())->ReTagThis("KeyAssocIm2Or"));
   if (anObj.KeyGenerateTxt().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("KeyGenerateTxt"),anObj.KeyGenerateTxt().Val())->ReTagThis("KeyGenerateTxt"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cTrAJ2_ExportProjImage & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.NameFileOut(),aTree->Get("NameFileOut",1)); //tototo 

   xml_init(anObj.KeySetOrPatIm(),aTree->Get("KeySetOrPatIm",1)); //tototo 

   xml_init(anObj.NameAppuis(),aTree->Get("NameAppuis",1)); //tototo 

   xml_init(anObj.KeyAssocIm2Or(),aTree->Get("KeyAssocIm2Or",1)); //tototo 

   xml_init(anObj.KeyGenerateTxt(),aTree->Get("KeyGenerateTxt",1)); //tototo 
}


cTplValGesInit< cChantierDescripteur > & cParam_Traj_AJ::DicoLoc()
{
   return mDicoLoc;
}

const cTplValGesInit< cChantierDescripteur > & cParam_Traj_AJ::DicoLoc()const 
{
   return mDicoLoc;
}


std::list< cTrAJ2_SectionImages > & cParam_Traj_AJ::TrAJ2_SectionImages()
{
   return mTrAJ2_SectionImages;
}

const std::list< cTrAJ2_SectionImages > & cParam_Traj_AJ::TrAJ2_SectionImages()const 
{
   return mTrAJ2_SectionImages;
}


cTplValGesInit< cElRegex_Ptr > & cParam_Traj_AJ::TraceImages()
{
   return mTraceImages;
}

const cTplValGesInit< cElRegex_Ptr > & cParam_Traj_AJ::TraceImages()const 
{
   return mTraceImages;
}


cTplValGesInit< cElRegex_Ptr > & cParam_Traj_AJ::TraceLogs()
{
   return mTraceLogs;
}

const cTplValGesInit< cElRegex_Ptr > & cParam_Traj_AJ::TraceLogs()const 
{
   return mTraceLogs;
}


std::list< cTrAJ2_SectionLog > & cParam_Traj_AJ::TrAJ2_SectionLog()
{
   return mTrAJ2_SectionLog;
}

const std::list< cTrAJ2_SectionLog > & cParam_Traj_AJ::TrAJ2_SectionLog()const 
{
   return mTrAJ2_SectionLog;
}


std::list< cTrAJ2_SectionMatch > & cParam_Traj_AJ::TrAJ2_SectionMatch()
{
   return mTrAJ2_SectionMatch;
}

const std::list< cTrAJ2_SectionMatch > & cParam_Traj_AJ::TrAJ2_SectionMatch()const 
{
   return mTrAJ2_SectionMatch;
}


std::list< cTrAJ2_ConvertionAppuis > & cParam_Traj_AJ::TrAJ2_ConvertionAppuis()
{
   return mTrAJ2_ConvertionAppuis;
}

const std::list< cTrAJ2_ConvertionAppuis > & cParam_Traj_AJ::TrAJ2_ConvertionAppuis()const 
{
   return mTrAJ2_ConvertionAppuis;
}


std::list< cTrAJ2_ExportProjImage > & cParam_Traj_AJ::TrAJ2_ExportProjImage()
{
   return mTrAJ2_ExportProjImage;
}

const std::list< cTrAJ2_ExportProjImage > & cParam_Traj_AJ::TrAJ2_ExportProjImage()const 
{
   return mTrAJ2_ExportProjImage;
}

cElXMLTree * ToXMLTree(const cParam_Traj_AJ & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"Param_Traj_AJ",eXMLBranche);
   if (anObj.DicoLoc().IsInit())
      aRes->AddFils(ToXMLTree(anObj.DicoLoc().Val())->ReTagThis("DicoLoc"));
  for
  (       std::list< cTrAJ2_SectionImages >::const_iterator it=anObj.TrAJ2_SectionImages().begin();
      it !=anObj.TrAJ2_SectionImages().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("TrAJ2_SectionImages"));
   if (anObj.TraceImages().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("TraceImages"),anObj.TraceImages().Val())->ReTagThis("TraceImages"));
   if (anObj.TraceLogs().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("TraceLogs"),anObj.TraceLogs().Val())->ReTagThis("TraceLogs"));
  for
  (       std::list< cTrAJ2_SectionLog >::const_iterator it=anObj.TrAJ2_SectionLog().begin();
      it !=anObj.TrAJ2_SectionLog().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("TrAJ2_SectionLog"));
  for
  (       std::list< cTrAJ2_SectionMatch >::const_iterator it=anObj.TrAJ2_SectionMatch().begin();
      it !=anObj.TrAJ2_SectionMatch().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("TrAJ2_SectionMatch"));
  for
  (       std::list< cTrAJ2_ConvertionAppuis >::const_iterator it=anObj.TrAJ2_ConvertionAppuis().begin();
      it !=anObj.TrAJ2_ConvertionAppuis().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("TrAJ2_ConvertionAppuis"));
  for
  (       std::list< cTrAJ2_ExportProjImage >::const_iterator it=anObj.TrAJ2_ExportProjImage().begin();
      it !=anObj.TrAJ2_ExportProjImage().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("TrAJ2_ExportProjImage"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cParam_Traj_AJ & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.DicoLoc(),aTree->Get("DicoLoc",1)); //tototo 

   xml_init(anObj.TrAJ2_SectionImages(),aTree->GetAll("TrAJ2_SectionImages",false,1));

   xml_init(anObj.TraceImages(),aTree->Get("TraceImages",1)); //tototo 

   xml_init(anObj.TraceLogs(),aTree->Get("TraceLogs",1)); //tototo 

   xml_init(anObj.TrAJ2_SectionLog(),aTree->GetAll("TrAJ2_SectionLog",false,1));

   xml_init(anObj.TrAJ2_SectionMatch(),aTree->GetAll("TrAJ2_SectionMatch",false,1));

   xml_init(anObj.TrAJ2_ConvertionAppuis(),aTree->GetAll("TrAJ2_ConvertionAppuis",false,1));

   xml_init(anObj.TrAJ2_ExportProjImage(),aTree->GetAll("TrAJ2_ExportProjImage",false,1));
}


std::list< std::string > & cParamGenereStr::KeySet()
{
   return mKeySet;
}

const std::list< std::string > & cParamGenereStr::KeySet()const 
{
   return mKeySet;
}


std::list< std::string > & cParamGenereStr::KeyString()
{
   return mKeyString;
}

const std::list< std::string > & cParamGenereStr::KeyString()const 
{
   return mKeyString;
}

cElXMLTree * ToXMLTree(const cParamGenereStr & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ParamGenereStr",eXMLBranche);
  for
  (       std::list< std::string >::const_iterator it=anObj.KeySet().begin();
      it !=anObj.KeySet().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("KeySet"),(*it))->ReTagThis("KeySet"));
  for
  (       std::list< std::string >::const_iterator it=anObj.KeyString().begin();
      it !=anObj.KeyString().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("KeyString"),(*it))->ReTagThis("KeyString"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cParamGenereStr & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.KeySet(),aTree->GetAll("KeySet",false,1));

   xml_init(anObj.KeyString(),aTree->GetAll("KeyString",false,1));
}


std::list< std::string > & cParamGenereStrVois::KeyRel()
{
   return mKeyRel;
}

const std::list< std::string > & cParamGenereStrVois::KeyRel()const 
{
   return mKeyRel;
}


std::list< std::string > & cParamGenereStrVois::KeyString()
{
   return mKeyString;
}

const std::list< std::string > & cParamGenereStrVois::KeyString()const 
{
   return mKeyString;
}


std::list< std::string > & cParamGenereStrVois::KeySet()
{
   return mKeySet;
}

const std::list< std::string > & cParamGenereStrVois::KeySet()const 
{
   return mKeySet;
}


cTplValGesInit< bool > & cParamGenereStrVois::UseIt()
{
   return mUseIt;
}

const cTplValGesInit< bool > & cParamGenereStrVois::UseIt()const 
{
   return mUseIt;
}

cElXMLTree * ToXMLTree(const cParamGenereStrVois & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ParamGenereStrVois",eXMLBranche);
  for
  (       std::list< std::string >::const_iterator it=anObj.KeyRel().begin();
      it !=anObj.KeyRel().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("KeyRel"),(*it))->ReTagThis("KeyRel"));
  for
  (       std::list< std::string >::const_iterator it=anObj.KeyString().begin();
      it !=anObj.KeyString().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("KeyString"),(*it))->ReTagThis("KeyString"));
  for
  (       std::list< std::string >::const_iterator it=anObj.KeySet().begin();
      it !=anObj.KeySet().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("KeySet"),(*it))->ReTagThis("KeySet"));
   if (anObj.UseIt().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("UseIt"),anObj.UseIt().Val())->ReTagThis("UseIt"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cParamGenereStrVois & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.KeyRel(),aTree->GetAll("KeyRel",false,1));

   xml_init(anObj.KeyString(),aTree->GetAll("KeyString",false,1));

   xml_init(anObj.KeySet(),aTree->GetAll("KeySet",false,1));

   xml_init(anObj.UseIt(),aTree->Get("UseIt",1),bool(false)); //tototo 
}


std::string & cSectionName::KeyNuage()
{
   return mKeyNuage;
}

const std::string & cSectionName::KeyNuage()const 
{
   return mKeyNuage;
}


std::string & cSectionName::KeyResult()
{
   return mKeyResult;
}

const std::string & cSectionName::KeyResult()const 
{
   return mKeyResult;
}


cTplValGesInit< std::string > & cSectionName::ModeleNuageResult()
{
   return mModeleNuageResult;
}

const cTplValGesInit< std::string > & cSectionName::ModeleNuageResult()const 
{
   return mModeleNuageResult;
}

cElXMLTree * ToXMLTree(const cSectionName & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"SectionName",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("KeyNuage"),anObj.KeyNuage())->ReTagThis("KeyNuage"));
   aRes->AddFils(::ToXMLTree(std::string("KeyResult"),anObj.KeyResult())->ReTagThis("KeyResult"));
   if (anObj.ModeleNuageResult().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ModeleNuageResult"),anObj.ModeleNuageResult().Val())->ReTagThis("ModeleNuageResult"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cSectionName & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.KeyNuage(),aTree->Get("KeyNuage",1)); //tototo 

   xml_init(anObj.KeyResult(),aTree->Get("KeyResult",1)); //tototo 

   xml_init(anObj.ModeleNuageResult(),aTree->Get("ModeleNuageResult",1)); //tototo 
}


cTplValGesInit< double > & cScoreMM1P::PdsAR()
{
   return mPdsAR;
}

const cTplValGesInit< double > & cScoreMM1P::PdsAR()const 
{
   return mPdsAR;
}


cTplValGesInit< double > & cScoreMM1P::PdsDistor()
{
   return mPdsDistor;
}

const cTplValGesInit< double > & cScoreMM1P::PdsDistor()const 
{
   return mPdsDistor;
}


cTplValGesInit< double > & cScoreMM1P::AmplImDistor()
{
   return mAmplImDistor;
}

const cTplValGesInit< double > & cScoreMM1P::AmplImDistor()const 
{
   return mAmplImDistor;
}


cTplValGesInit< double > & cScoreMM1P::SeuilDist()
{
   return mSeuilDist;
}

const cTplValGesInit< double > & cScoreMM1P::SeuilDist()const 
{
   return mSeuilDist;
}


cTplValGesInit< double > & cScoreMM1P::PdsDistBord()
{
   return mPdsDistBord;
}

const cTplValGesInit< double > & cScoreMM1P::PdsDistBord()const 
{
   return mPdsDistBord;
}


cTplValGesInit< double > & cScoreMM1P::SeuilDisBord()
{
   return mSeuilDisBord;
}

const cTplValGesInit< double > & cScoreMM1P::SeuilDisBord()const 
{
   return mSeuilDisBord;
}

cElXMLTree * ToXMLTree(const cScoreMM1P & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ScoreMM1P",eXMLBranche);
   if (anObj.PdsAR().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PdsAR"),anObj.PdsAR().Val())->ReTagThis("PdsAR"));
   if (anObj.PdsDistor().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PdsDistor"),anObj.PdsDistor().Val())->ReTagThis("PdsDistor"));
   if (anObj.AmplImDistor().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("AmplImDistor"),anObj.AmplImDistor().Val())->ReTagThis("AmplImDistor"));
   if (anObj.SeuilDist().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SeuilDist"),anObj.SeuilDist().Val())->ReTagThis("SeuilDist"));
   if (anObj.PdsDistBord().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PdsDistBord"),anObj.PdsDistBord().Val())->ReTagThis("PdsDistBord"));
   if (anObj.SeuilDisBord().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SeuilDisBord"),anObj.SeuilDisBord().Val())->ReTagThis("SeuilDisBord"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cScoreMM1P & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.PdsAR(),aTree->Get("PdsAR",1),double(1.0)); //tototo 

   xml_init(anObj.PdsDistor(),aTree->Get("PdsDistor",1),double(0.5)); //tototo 

   xml_init(anObj.AmplImDistor(),aTree->Get("AmplImDistor",1),double(100.0)); //tototo 

   xml_init(anObj.SeuilDist(),aTree->Get("SeuilDist",1),double(0.5)); //tototo 

   xml_init(anObj.PdsDistBord(),aTree->Get("PdsDistBord",1),double(0.25)); //tototo 

   xml_init(anObj.SeuilDisBord(),aTree->Get("SeuilDisBord",1),double(3.0)); //tototo 
}


cTplValGesInit< double > & cSectionScoreQualite::PdsAR()
{
   return ScoreMM1P().Val().PdsAR();
}

const cTplValGesInit< double > & cSectionScoreQualite::PdsAR()const 
{
   return ScoreMM1P().Val().PdsAR();
}


cTplValGesInit< double > & cSectionScoreQualite::PdsDistor()
{
   return ScoreMM1P().Val().PdsDistor();
}

const cTplValGesInit< double > & cSectionScoreQualite::PdsDistor()const 
{
   return ScoreMM1P().Val().PdsDistor();
}


cTplValGesInit< double > & cSectionScoreQualite::AmplImDistor()
{
   return ScoreMM1P().Val().AmplImDistor();
}

const cTplValGesInit< double > & cSectionScoreQualite::AmplImDistor()const 
{
   return ScoreMM1P().Val().AmplImDistor();
}


cTplValGesInit< double > & cSectionScoreQualite::SeuilDist()
{
   return ScoreMM1P().Val().SeuilDist();
}

const cTplValGesInit< double > & cSectionScoreQualite::SeuilDist()const 
{
   return ScoreMM1P().Val().SeuilDist();
}


cTplValGesInit< double > & cSectionScoreQualite::PdsDistBord()
{
   return ScoreMM1P().Val().PdsDistBord();
}

const cTplValGesInit< double > & cSectionScoreQualite::PdsDistBord()const 
{
   return ScoreMM1P().Val().PdsDistBord();
}


cTplValGesInit< double > & cSectionScoreQualite::SeuilDisBord()
{
   return ScoreMM1P().Val().SeuilDisBord();
}

const cTplValGesInit< double > & cSectionScoreQualite::SeuilDisBord()const 
{
   return ScoreMM1P().Val().SeuilDisBord();
}


cTplValGesInit< cScoreMM1P > & cSectionScoreQualite::ScoreMM1P()
{
   return mScoreMM1P;
}

const cTplValGesInit< cScoreMM1P > & cSectionScoreQualite::ScoreMM1P()const 
{
   return mScoreMM1P;
}

cElXMLTree * ToXMLTree(const cSectionScoreQualite & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"SectionScoreQualite",eXMLBranche);
   if (anObj.ScoreMM1P().IsInit())
      aRes->AddFils(ToXMLTree(anObj.ScoreMM1P().Val())->ReTagThis("ScoreMM1P"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cSectionScoreQualite & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.ScoreMM1P(),aTree->Get("ScoreMM1P",1)); //tototo 
}


cTplValGesInit< double > & cFMNtBySort::PercFusion()
{
   return mPercFusion;
}

const cTplValGesInit< double > & cFMNtBySort::PercFusion()const 
{
   return mPercFusion;
}

cElXMLTree * ToXMLTree(const cFMNtBySort & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"FMNtBySort",eXMLBranche);
   if (anObj.PercFusion().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PercFusion"),anObj.PercFusion().Val())->ReTagThis("PercFusion"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cFMNtBySort & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.PercFusion(),aTree->Get("PercFusion",1),double(50.0)); //tototo 
}


double & cFMNT_GesNoVal::PenteMax()
{
   return mPenteMax;
}

const double & cFMNT_GesNoVal::PenteMax()const 
{
   return mPenteMax;
}


double & cFMNT_GesNoVal::GainNoVal()
{
   return mGainNoVal;
}

const double & cFMNT_GesNoVal::GainNoVal()const 
{
   return mGainNoVal;
}


double & cFMNT_GesNoVal::Trans()
{
   return mTrans;
}

const double & cFMNT_GesNoVal::Trans()const 
{
   return mTrans;
}

cElXMLTree * ToXMLTree(const cFMNT_GesNoVal & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"FMNT_GesNoVal",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("PenteMax"),anObj.PenteMax())->ReTagThis("PenteMax"));
   aRes->AddFils(::ToXMLTree(std::string("GainNoVal"),anObj.GainNoVal())->ReTagThis("GainNoVal"));
   aRes->AddFils(::ToXMLTree(std::string("Trans"),anObj.Trans())->ReTagThis("Trans"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cFMNT_GesNoVal & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.PenteMax(),aTree->Get("PenteMax",1)); //tototo 

   xml_init(anObj.GainNoVal(),aTree->Get("GainNoVal",1)); //tototo 

   xml_init(anObj.Trans(),aTree->Get("Trans",1)); //tototo 
}


double & cFMNT_ProgDyn::Regul()
{
   return mRegul;
}

const double & cFMNT_ProgDyn::Regul()const 
{
   return mRegul;
}


double & cFMNT_ProgDyn::Sigma0()
{
   return mSigma0;
}

const double & cFMNT_ProgDyn::Sigma0()const 
{
   return mSigma0;
}


int & cFMNT_ProgDyn::NbDir()
{
   return mNbDir;
}

const int & cFMNT_ProgDyn::NbDir()const 
{
   return mNbDir;
}


double & cFMNT_ProgDyn::PenteMax()
{
   return FMNT_GesNoVal().Val().PenteMax();
}

const double & cFMNT_ProgDyn::PenteMax()const 
{
   return FMNT_GesNoVal().Val().PenteMax();
}


double & cFMNT_ProgDyn::GainNoVal()
{
   return FMNT_GesNoVal().Val().GainNoVal();
}

const double & cFMNT_ProgDyn::GainNoVal()const 
{
   return FMNT_GesNoVal().Val().GainNoVal();
}


double & cFMNT_ProgDyn::Trans()
{
   return FMNT_GesNoVal().Val().Trans();
}

const double & cFMNT_ProgDyn::Trans()const 
{
   return FMNT_GesNoVal().Val().Trans();
}


cTplValGesInit< cFMNT_GesNoVal > & cFMNT_ProgDyn::FMNT_GesNoVal()
{
   return mFMNT_GesNoVal;
}

const cTplValGesInit< cFMNT_GesNoVal > & cFMNT_ProgDyn::FMNT_GesNoVal()const 
{
   return mFMNT_GesNoVal;
}

cElXMLTree * ToXMLTree(const cFMNT_ProgDyn & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"FMNT_ProgDyn",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Regul"),anObj.Regul())->ReTagThis("Regul"));
   aRes->AddFils(::ToXMLTree(std::string("Sigma0"),anObj.Sigma0())->ReTagThis("Sigma0"));
   aRes->AddFils(::ToXMLTree(std::string("NbDir"),anObj.NbDir())->ReTagThis("NbDir"));
   if (anObj.FMNT_GesNoVal().IsInit())
      aRes->AddFils(ToXMLTree(anObj.FMNT_GesNoVal().Val())->ReTagThis("FMNT_GesNoVal"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cFMNT_ProgDyn & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Regul(),aTree->Get("Regul",1)); //tototo 

   xml_init(anObj.Sigma0(),aTree->Get("Sigma0",1)); //tototo 

   xml_init(anObj.NbDir(),aTree->Get("NbDir",1)); //tototo 

   xml_init(anObj.FMNT_GesNoVal(),aTree->Get("FMNT_GesNoVal",1)); //tototo 
}


double & cFMNtByMaxEvid::SigmaPds()
{
   return mSigmaPds;
}

const double & cFMNtByMaxEvid::SigmaPds()const 
{
   return mSigmaPds;
}


cTplValGesInit< double > & cFMNtByMaxEvid::SigmaZ()
{
   return mSigmaZ;
}

const cTplValGesInit< double > & cFMNtByMaxEvid::SigmaZ()const 
{
   return mSigmaZ;
}


cTplValGesInit< double > & cFMNtByMaxEvid::MaxDif()
{
   return mMaxDif;
}

const cTplValGesInit< double > & cFMNtByMaxEvid::MaxDif()const 
{
   return mMaxDif;
}


cTplValGesInit< bool > & cFMNtByMaxEvid::QuickExp()
{
   return mQuickExp;
}

const cTplValGesInit< bool > & cFMNtByMaxEvid::QuickExp()const 
{
   return mQuickExp;
}


double & cFMNtByMaxEvid::Regul()
{
   return FMNT_ProgDyn().Val().Regul();
}

const double & cFMNtByMaxEvid::Regul()const 
{
   return FMNT_ProgDyn().Val().Regul();
}


double & cFMNtByMaxEvid::Sigma0()
{
   return FMNT_ProgDyn().Val().Sigma0();
}

const double & cFMNtByMaxEvid::Sigma0()const 
{
   return FMNT_ProgDyn().Val().Sigma0();
}


int & cFMNtByMaxEvid::NbDir()
{
   return FMNT_ProgDyn().Val().NbDir();
}

const int & cFMNtByMaxEvid::NbDir()const 
{
   return FMNT_ProgDyn().Val().NbDir();
}


double & cFMNtByMaxEvid::PenteMax()
{
   return FMNT_ProgDyn().Val().FMNT_GesNoVal().Val().PenteMax();
}

const double & cFMNtByMaxEvid::PenteMax()const 
{
   return FMNT_ProgDyn().Val().FMNT_GesNoVal().Val().PenteMax();
}


double & cFMNtByMaxEvid::GainNoVal()
{
   return FMNT_ProgDyn().Val().FMNT_GesNoVal().Val().GainNoVal();
}

const double & cFMNtByMaxEvid::GainNoVal()const 
{
   return FMNT_ProgDyn().Val().FMNT_GesNoVal().Val().GainNoVal();
}


double & cFMNtByMaxEvid::Trans()
{
   return FMNT_ProgDyn().Val().FMNT_GesNoVal().Val().Trans();
}

const double & cFMNtByMaxEvid::Trans()const 
{
   return FMNT_ProgDyn().Val().FMNT_GesNoVal().Val().Trans();
}


cTplValGesInit< cFMNT_GesNoVal > & cFMNtByMaxEvid::FMNT_GesNoVal()
{
   return FMNT_ProgDyn().Val().FMNT_GesNoVal();
}

const cTplValGesInit< cFMNT_GesNoVal > & cFMNtByMaxEvid::FMNT_GesNoVal()const 
{
   return FMNT_ProgDyn().Val().FMNT_GesNoVal();
}


cTplValGesInit< cFMNT_ProgDyn > & cFMNtByMaxEvid::FMNT_ProgDyn()
{
   return mFMNT_ProgDyn;
}

const cTplValGesInit< cFMNT_ProgDyn > & cFMNtByMaxEvid::FMNT_ProgDyn()const 
{
   return mFMNT_ProgDyn;
}

cElXMLTree * ToXMLTree(const cFMNtByMaxEvid & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"FMNtByMaxEvid",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("SigmaPds"),anObj.SigmaPds())->ReTagThis("SigmaPds"));
   if (anObj.SigmaZ().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SigmaZ"),anObj.SigmaZ().Val())->ReTagThis("SigmaZ"));
   if (anObj.MaxDif().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("MaxDif"),anObj.MaxDif().Val())->ReTagThis("MaxDif"));
   if (anObj.QuickExp().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("QuickExp"),anObj.QuickExp().Val())->ReTagThis("QuickExp"));
   if (anObj.FMNT_ProgDyn().IsInit())
      aRes->AddFils(ToXMLTree(anObj.FMNT_ProgDyn().Val())->ReTagThis("FMNT_ProgDyn"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cFMNtByMaxEvid & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.SigmaPds(),aTree->Get("SigmaPds",1)); //tototo 

   xml_init(anObj.SigmaZ(),aTree->Get("SigmaZ",1)); //tototo 

   xml_init(anObj.MaxDif(),aTree->Get("MaxDif",1),double(1e9)); //tototo 

   xml_init(anObj.QuickExp(),aTree->Get("QuickExp",1),bool(false)); //tototo 

   xml_init(anObj.FMNT_ProgDyn(),aTree->Get("FMNT_ProgDyn",1)); //tototo 
}


cTplValGesInit< double > & cSpecAlgoFMNT::PercFusion()
{
   return FMNtBySort().Val().PercFusion();
}

const cTplValGesInit< double > & cSpecAlgoFMNT::PercFusion()const 
{
   return FMNtBySort().Val().PercFusion();
}


cTplValGesInit< cFMNtBySort > & cSpecAlgoFMNT::FMNtBySort()
{
   return mFMNtBySort;
}

const cTplValGesInit< cFMNtBySort > & cSpecAlgoFMNT::FMNtBySort()const 
{
   return mFMNtBySort;
}


double & cSpecAlgoFMNT::SigmaPds()
{
   return FMNtByMaxEvid().Val().SigmaPds();
}

const double & cSpecAlgoFMNT::SigmaPds()const 
{
   return FMNtByMaxEvid().Val().SigmaPds();
}


cTplValGesInit< double > & cSpecAlgoFMNT::SigmaZ()
{
   return FMNtByMaxEvid().Val().SigmaZ();
}

const cTplValGesInit< double > & cSpecAlgoFMNT::SigmaZ()const 
{
   return FMNtByMaxEvid().Val().SigmaZ();
}


cTplValGesInit< double > & cSpecAlgoFMNT::MaxDif()
{
   return FMNtByMaxEvid().Val().MaxDif();
}

const cTplValGesInit< double > & cSpecAlgoFMNT::MaxDif()const 
{
   return FMNtByMaxEvid().Val().MaxDif();
}


cTplValGesInit< bool > & cSpecAlgoFMNT::QuickExp()
{
   return FMNtByMaxEvid().Val().QuickExp();
}

const cTplValGesInit< bool > & cSpecAlgoFMNT::QuickExp()const 
{
   return FMNtByMaxEvid().Val().QuickExp();
}


double & cSpecAlgoFMNT::Regul()
{
   return FMNtByMaxEvid().Val().FMNT_ProgDyn().Val().Regul();
}

const double & cSpecAlgoFMNT::Regul()const 
{
   return FMNtByMaxEvid().Val().FMNT_ProgDyn().Val().Regul();
}


double & cSpecAlgoFMNT::Sigma0()
{
   return FMNtByMaxEvid().Val().FMNT_ProgDyn().Val().Sigma0();
}

const double & cSpecAlgoFMNT::Sigma0()const 
{
   return FMNtByMaxEvid().Val().FMNT_ProgDyn().Val().Sigma0();
}


int & cSpecAlgoFMNT::NbDir()
{
   return FMNtByMaxEvid().Val().FMNT_ProgDyn().Val().NbDir();
}

const int & cSpecAlgoFMNT::NbDir()const 
{
   return FMNtByMaxEvid().Val().FMNT_ProgDyn().Val().NbDir();
}


double & cSpecAlgoFMNT::PenteMax()
{
   return FMNtByMaxEvid().Val().FMNT_ProgDyn().Val().FMNT_GesNoVal().Val().PenteMax();
}

const double & cSpecAlgoFMNT::PenteMax()const 
{
   return FMNtByMaxEvid().Val().FMNT_ProgDyn().Val().FMNT_GesNoVal().Val().PenteMax();
}


double & cSpecAlgoFMNT::GainNoVal()
{
   return FMNtByMaxEvid().Val().FMNT_ProgDyn().Val().FMNT_GesNoVal().Val().GainNoVal();
}

const double & cSpecAlgoFMNT::GainNoVal()const 
{
   return FMNtByMaxEvid().Val().FMNT_ProgDyn().Val().FMNT_GesNoVal().Val().GainNoVal();
}


double & cSpecAlgoFMNT::Trans()
{
   return FMNtByMaxEvid().Val().FMNT_ProgDyn().Val().FMNT_GesNoVal().Val().Trans();
}

const double & cSpecAlgoFMNT::Trans()const 
{
   return FMNtByMaxEvid().Val().FMNT_ProgDyn().Val().FMNT_GesNoVal().Val().Trans();
}


cTplValGesInit< cFMNT_GesNoVal > & cSpecAlgoFMNT::FMNT_GesNoVal()
{
   return FMNtByMaxEvid().Val().FMNT_ProgDyn().Val().FMNT_GesNoVal();
}

const cTplValGesInit< cFMNT_GesNoVal > & cSpecAlgoFMNT::FMNT_GesNoVal()const 
{
   return FMNtByMaxEvid().Val().FMNT_ProgDyn().Val().FMNT_GesNoVal();
}


cTplValGesInit< cFMNT_ProgDyn > & cSpecAlgoFMNT::FMNT_ProgDyn()
{
   return FMNtByMaxEvid().Val().FMNT_ProgDyn();
}

const cTplValGesInit< cFMNT_ProgDyn > & cSpecAlgoFMNT::FMNT_ProgDyn()const 
{
   return FMNtByMaxEvid().Val().FMNT_ProgDyn();
}


cTplValGesInit< cFMNtByMaxEvid > & cSpecAlgoFMNT::FMNtByMaxEvid()
{
   return mFMNtByMaxEvid;
}

const cTplValGesInit< cFMNtByMaxEvid > & cSpecAlgoFMNT::FMNtByMaxEvid()const 
{
   return mFMNtByMaxEvid;
}

cElXMLTree * ToXMLTree(const cSpecAlgoFMNT & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"SpecAlgoFMNT",eXMLBranche);
   if (anObj.FMNtBySort().IsInit())
      aRes->AddFils(ToXMLTree(anObj.FMNtBySort().Val())->ReTagThis("FMNtBySort"));
   if (anObj.FMNtByMaxEvid().IsInit())
      aRes->AddFils(ToXMLTree(anObj.FMNtByMaxEvid().Val())->ReTagThis("FMNtByMaxEvid"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cSpecAlgoFMNT & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.FMNtBySort(),aTree->Get("FMNtBySort",1)); //tototo 

   xml_init(anObj.FMNtByMaxEvid(),aTree->Get("FMNtByMaxEvid",1)); //tototo 
}


double & cParamAlgoFusionMNT::FMNTSeuilCorrel()
{
   return mFMNTSeuilCorrel;
}

const double & cParamAlgoFusionMNT::FMNTSeuilCorrel()const 
{
   return mFMNTSeuilCorrel;
}


double & cParamAlgoFusionMNT::FMNTGammaCorrel()
{
   return mFMNTGammaCorrel;
}

const double & cParamAlgoFusionMNT::FMNTGammaCorrel()const 
{
   return mFMNTGammaCorrel;
}


cTplValGesInit< double > & cParamAlgoFusionMNT::PercFusion()
{
   return SpecAlgoFMNT().FMNtBySort().Val().PercFusion();
}

const cTplValGesInit< double > & cParamAlgoFusionMNT::PercFusion()const 
{
   return SpecAlgoFMNT().FMNtBySort().Val().PercFusion();
}


cTplValGesInit< cFMNtBySort > & cParamAlgoFusionMNT::FMNtBySort()
{
   return SpecAlgoFMNT().FMNtBySort();
}

const cTplValGesInit< cFMNtBySort > & cParamAlgoFusionMNT::FMNtBySort()const 
{
   return SpecAlgoFMNT().FMNtBySort();
}


double & cParamAlgoFusionMNT::SigmaPds()
{
   return SpecAlgoFMNT().FMNtByMaxEvid().Val().SigmaPds();
}

const double & cParamAlgoFusionMNT::SigmaPds()const 
{
   return SpecAlgoFMNT().FMNtByMaxEvid().Val().SigmaPds();
}


cTplValGesInit< double > & cParamAlgoFusionMNT::SigmaZ()
{
   return SpecAlgoFMNT().FMNtByMaxEvid().Val().SigmaZ();
}

const cTplValGesInit< double > & cParamAlgoFusionMNT::SigmaZ()const 
{
   return SpecAlgoFMNT().FMNtByMaxEvid().Val().SigmaZ();
}


cTplValGesInit< double > & cParamAlgoFusionMNT::MaxDif()
{
   return SpecAlgoFMNT().FMNtByMaxEvid().Val().MaxDif();
}

const cTplValGesInit< double > & cParamAlgoFusionMNT::MaxDif()const 
{
   return SpecAlgoFMNT().FMNtByMaxEvid().Val().MaxDif();
}


cTplValGesInit< bool > & cParamAlgoFusionMNT::QuickExp()
{
   return SpecAlgoFMNT().FMNtByMaxEvid().Val().QuickExp();
}

const cTplValGesInit< bool > & cParamAlgoFusionMNT::QuickExp()const 
{
   return SpecAlgoFMNT().FMNtByMaxEvid().Val().QuickExp();
}


double & cParamAlgoFusionMNT::Regul()
{
   return SpecAlgoFMNT().FMNtByMaxEvid().Val().FMNT_ProgDyn().Val().Regul();
}

const double & cParamAlgoFusionMNT::Regul()const 
{
   return SpecAlgoFMNT().FMNtByMaxEvid().Val().FMNT_ProgDyn().Val().Regul();
}


double & cParamAlgoFusionMNT::Sigma0()
{
   return SpecAlgoFMNT().FMNtByMaxEvid().Val().FMNT_ProgDyn().Val().Sigma0();
}

const double & cParamAlgoFusionMNT::Sigma0()const 
{
   return SpecAlgoFMNT().FMNtByMaxEvid().Val().FMNT_ProgDyn().Val().Sigma0();
}


int & cParamAlgoFusionMNT::NbDir()
{
   return SpecAlgoFMNT().FMNtByMaxEvid().Val().FMNT_ProgDyn().Val().NbDir();
}

const int & cParamAlgoFusionMNT::NbDir()const 
{
   return SpecAlgoFMNT().FMNtByMaxEvid().Val().FMNT_ProgDyn().Val().NbDir();
}


double & cParamAlgoFusionMNT::PenteMax()
{
   return SpecAlgoFMNT().FMNtByMaxEvid().Val().FMNT_ProgDyn().Val().FMNT_GesNoVal().Val().PenteMax();
}

const double & cParamAlgoFusionMNT::PenteMax()const 
{
   return SpecAlgoFMNT().FMNtByMaxEvid().Val().FMNT_ProgDyn().Val().FMNT_GesNoVal().Val().PenteMax();
}


double & cParamAlgoFusionMNT::GainNoVal()
{
   return SpecAlgoFMNT().FMNtByMaxEvid().Val().FMNT_ProgDyn().Val().FMNT_GesNoVal().Val().GainNoVal();
}

const double & cParamAlgoFusionMNT::GainNoVal()const 
{
   return SpecAlgoFMNT().FMNtByMaxEvid().Val().FMNT_ProgDyn().Val().FMNT_GesNoVal().Val().GainNoVal();
}


double & cParamAlgoFusionMNT::Trans()
{
   return SpecAlgoFMNT().FMNtByMaxEvid().Val().FMNT_ProgDyn().Val().FMNT_GesNoVal().Val().Trans();
}

const double & cParamAlgoFusionMNT::Trans()const 
{
   return SpecAlgoFMNT().FMNtByMaxEvid().Val().FMNT_ProgDyn().Val().FMNT_GesNoVal().Val().Trans();
}


cTplValGesInit< cFMNT_GesNoVal > & cParamAlgoFusionMNT::FMNT_GesNoVal()
{
   return SpecAlgoFMNT().FMNtByMaxEvid().Val().FMNT_ProgDyn().Val().FMNT_GesNoVal();
}

const cTplValGesInit< cFMNT_GesNoVal > & cParamAlgoFusionMNT::FMNT_GesNoVal()const 
{
   return SpecAlgoFMNT().FMNtByMaxEvid().Val().FMNT_ProgDyn().Val().FMNT_GesNoVal();
}


cTplValGesInit< cFMNT_ProgDyn > & cParamAlgoFusionMNT::FMNT_ProgDyn()
{
   return SpecAlgoFMNT().FMNtByMaxEvid().Val().FMNT_ProgDyn();
}

const cTplValGesInit< cFMNT_ProgDyn > & cParamAlgoFusionMNT::FMNT_ProgDyn()const 
{
   return SpecAlgoFMNT().FMNtByMaxEvid().Val().FMNT_ProgDyn();
}


cTplValGesInit< cFMNtByMaxEvid > & cParamAlgoFusionMNT::FMNtByMaxEvid()
{
   return SpecAlgoFMNT().FMNtByMaxEvid();
}

const cTplValGesInit< cFMNtByMaxEvid > & cParamAlgoFusionMNT::FMNtByMaxEvid()const 
{
   return SpecAlgoFMNT().FMNtByMaxEvid();
}


cSpecAlgoFMNT & cParamAlgoFusionMNT::SpecAlgoFMNT()
{
   return mSpecAlgoFMNT;
}

const cSpecAlgoFMNT & cParamAlgoFusionMNT::SpecAlgoFMNT()const 
{
   return mSpecAlgoFMNT;
}

cElXMLTree * ToXMLTree(const cParamAlgoFusionMNT & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ParamAlgoFusionMNT",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("FMNTSeuilCorrel"),anObj.FMNTSeuilCorrel())->ReTagThis("FMNTSeuilCorrel"));
   aRes->AddFils(::ToXMLTree(std::string("FMNTGammaCorrel"),anObj.FMNTGammaCorrel())->ReTagThis("FMNTGammaCorrel"));
   aRes->AddFils(ToXMLTree(anObj.SpecAlgoFMNT())->ReTagThis("SpecAlgoFMNT"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cParamAlgoFusionMNT & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.FMNTSeuilCorrel(),aTree->Get("FMNTSeuilCorrel",1)); //tototo 

   xml_init(anObj.FMNTGammaCorrel(),aTree->Get("FMNTGammaCorrel",1)); //tototo 

   xml_init(anObj.SpecAlgoFMNT(),aTree->Get("SpecAlgoFMNT",1)); //tototo 
}


cTplValGesInit< int > & cSectionGestionChantier::SzDalles()
{
   return mSzDalles;
}

const cTplValGesInit< int > & cSectionGestionChantier::SzDalles()const 
{
   return mSzDalles;
}


cTplValGesInit< int > & cSectionGestionChantier::RecouvrtDalles()
{
   return mRecouvrtDalles;
}

const cTplValGesInit< int > & cSectionGestionChantier::RecouvrtDalles()const 
{
   return mRecouvrtDalles;
}


cTplValGesInit< std::string > & cSectionGestionChantier::ParalMkF()
{
   return mParalMkF;
}

const cTplValGesInit< std::string > & cSectionGestionChantier::ParalMkF()const 
{
   return mParalMkF;
}


cTplValGesInit< bool > & cSectionGestionChantier::InterneCalledByProcess()
{
   return mInterneCalledByProcess;
}

const cTplValGesInit< bool > & cSectionGestionChantier::InterneCalledByProcess()const 
{
   return mInterneCalledByProcess;
}


cTplValGesInit< std::string > & cSectionGestionChantier::InterneSingleImage()
{
   return mInterneSingleImage;
}

const cTplValGesInit< std::string > & cSectionGestionChantier::InterneSingleImage()const 
{
   return mInterneSingleImage;
}


cTplValGesInit< int > & cSectionGestionChantier::InterneSingleBox()
{
   return mInterneSingleBox;
}

const cTplValGesInit< int > & cSectionGestionChantier::InterneSingleBox()const 
{
   return mInterneSingleBox;
}


cTplValGesInit< std::string > & cSectionGestionChantier::WorkDirPFM()
{
   return mWorkDirPFM;
}

const cTplValGesInit< std::string > & cSectionGestionChantier::WorkDirPFM()const 
{
   return mWorkDirPFM;
}


cTplValGesInit< Box2di > & cSectionGestionChantier::BoxTest()
{
   return mBoxTest;
}

const cTplValGesInit< Box2di > & cSectionGestionChantier::BoxTest()const 
{
   return mBoxTest;
}

cElXMLTree * ToXMLTree(const cSectionGestionChantier & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"SectionGestionChantier",eXMLBranche);
   if (anObj.SzDalles().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SzDalles"),anObj.SzDalles().Val())->ReTagThis("SzDalles"));
   if (anObj.RecouvrtDalles().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("RecouvrtDalles"),anObj.RecouvrtDalles().Val())->ReTagThis("RecouvrtDalles"));
   if (anObj.ParalMkF().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ParalMkF"),anObj.ParalMkF().Val())->ReTagThis("ParalMkF"));
   if (anObj.InterneCalledByProcess().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("InterneCalledByProcess"),anObj.InterneCalledByProcess().Val())->ReTagThis("InterneCalledByProcess"));
   if (anObj.InterneSingleImage().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("InterneSingleImage"),anObj.InterneSingleImage().Val())->ReTagThis("InterneSingleImage"));
   if (anObj.InterneSingleBox().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("InterneSingleBox"),anObj.InterneSingleBox().Val())->ReTagThis("InterneSingleBox"));
   if (anObj.WorkDirPFM().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("WorkDirPFM"),anObj.WorkDirPFM().Val())->ReTagThis("WorkDirPFM"));
   if (anObj.BoxTest().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("BoxTest"),anObj.BoxTest().Val())->ReTagThis("BoxTest"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cSectionGestionChantier & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.SzDalles(),aTree->Get("SzDalles",1),int(2000)); //tototo 

   xml_init(anObj.RecouvrtDalles(),aTree->Get("RecouvrtDalles",1),int(40)); //tototo 

   xml_init(anObj.ParalMkF(),aTree->Get("ParalMkF",1)); //tototo 

   xml_init(anObj.InterneCalledByProcess(),aTree->Get("InterneCalledByProcess",1),bool(false)); //tototo 

   xml_init(anObj.InterneSingleImage(),aTree->Get("InterneSingleImage",1),std::string("")); //tototo 

   xml_init(anObj.InterneSingleBox(),aTree->Get("InterneSingleBox",1),int(-1)); //tototo 

   xml_init(anObj.WorkDirPFM(),aTree->Get("WorkDirPFM",1)); //tototo 

   xml_init(anObj.BoxTest(),aTree->Get("BoxTest",1)); //tototo 
}


cTplValGesInit< cChantierDescripteur > & cParamFusionMNT::DicoLoc()
{
   return mDicoLoc;
}

const cTplValGesInit< cChantierDescripteur > & cParamFusionMNT::DicoLoc()const 
{
   return mDicoLoc;
}


std::string & cParamFusionMNT::KeyNuage()
{
   return SectionName().KeyNuage();
}

const std::string & cParamFusionMNT::KeyNuage()const 
{
   return SectionName().KeyNuage();
}


std::string & cParamFusionMNT::KeyResult()
{
   return SectionName().KeyResult();
}

const std::string & cParamFusionMNT::KeyResult()const 
{
   return SectionName().KeyResult();
}


cTplValGesInit< std::string > & cParamFusionMNT::ModeleNuageResult()
{
   return SectionName().ModeleNuageResult();
}

const cTplValGesInit< std::string > & cParamFusionMNT::ModeleNuageResult()const 
{
   return SectionName().ModeleNuageResult();
}


cSectionName & cParamFusionMNT::SectionName()
{
   return mSectionName;
}

const cSectionName & cParamFusionMNT::SectionName()const 
{
   return mSectionName;
}


cTplValGesInit< double > & cParamFusionMNT::PdsAR()
{
   return SectionScoreQualite().Val().ScoreMM1P().Val().PdsAR();
}

const cTplValGesInit< double > & cParamFusionMNT::PdsAR()const 
{
   return SectionScoreQualite().Val().ScoreMM1P().Val().PdsAR();
}


cTplValGesInit< double > & cParamFusionMNT::PdsDistor()
{
   return SectionScoreQualite().Val().ScoreMM1P().Val().PdsDistor();
}

const cTplValGesInit< double > & cParamFusionMNT::PdsDistor()const 
{
   return SectionScoreQualite().Val().ScoreMM1P().Val().PdsDistor();
}


cTplValGesInit< double > & cParamFusionMNT::AmplImDistor()
{
   return SectionScoreQualite().Val().ScoreMM1P().Val().AmplImDistor();
}

const cTplValGesInit< double > & cParamFusionMNT::AmplImDistor()const 
{
   return SectionScoreQualite().Val().ScoreMM1P().Val().AmplImDistor();
}


cTplValGesInit< double > & cParamFusionMNT::SeuilDist()
{
   return SectionScoreQualite().Val().ScoreMM1P().Val().SeuilDist();
}

const cTplValGesInit< double > & cParamFusionMNT::SeuilDist()const 
{
   return SectionScoreQualite().Val().ScoreMM1P().Val().SeuilDist();
}


cTplValGesInit< double > & cParamFusionMNT::PdsDistBord()
{
   return SectionScoreQualite().Val().ScoreMM1P().Val().PdsDistBord();
}

const cTplValGesInit< double > & cParamFusionMNT::PdsDistBord()const 
{
   return SectionScoreQualite().Val().ScoreMM1P().Val().PdsDistBord();
}


cTplValGesInit< double > & cParamFusionMNT::SeuilDisBord()
{
   return SectionScoreQualite().Val().ScoreMM1P().Val().SeuilDisBord();
}

const cTplValGesInit< double > & cParamFusionMNT::SeuilDisBord()const 
{
   return SectionScoreQualite().Val().ScoreMM1P().Val().SeuilDisBord();
}


cTplValGesInit< cScoreMM1P > & cParamFusionMNT::ScoreMM1P()
{
   return SectionScoreQualite().Val().ScoreMM1P();
}

const cTplValGesInit< cScoreMM1P > & cParamFusionMNT::ScoreMM1P()const 
{
   return SectionScoreQualite().Val().ScoreMM1P();
}


cTplValGesInit< cSectionScoreQualite > & cParamFusionMNT::SectionScoreQualite()
{
   return mSectionScoreQualite;
}

const cTplValGesInit< cSectionScoreQualite > & cParamFusionMNT::SectionScoreQualite()const 
{
   return mSectionScoreQualite;
}


double & cParamFusionMNT::FMNTSeuilCorrel()
{
   return ParamAlgoFusionMNT().FMNTSeuilCorrel();
}

const double & cParamFusionMNT::FMNTSeuilCorrel()const 
{
   return ParamAlgoFusionMNT().FMNTSeuilCorrel();
}


double & cParamFusionMNT::FMNTGammaCorrel()
{
   return ParamAlgoFusionMNT().FMNTGammaCorrel();
}

const double & cParamFusionMNT::FMNTGammaCorrel()const 
{
   return ParamAlgoFusionMNT().FMNTGammaCorrel();
}


cTplValGesInit< double > & cParamFusionMNT::PercFusion()
{
   return ParamAlgoFusionMNT().SpecAlgoFMNT().FMNtBySort().Val().PercFusion();
}

const cTplValGesInit< double > & cParamFusionMNT::PercFusion()const 
{
   return ParamAlgoFusionMNT().SpecAlgoFMNT().FMNtBySort().Val().PercFusion();
}


cTplValGesInit< cFMNtBySort > & cParamFusionMNT::FMNtBySort()
{
   return ParamAlgoFusionMNT().SpecAlgoFMNT().FMNtBySort();
}

const cTplValGesInit< cFMNtBySort > & cParamFusionMNT::FMNtBySort()const 
{
   return ParamAlgoFusionMNT().SpecAlgoFMNT().FMNtBySort();
}


double & cParamFusionMNT::SigmaPds()
{
   return ParamAlgoFusionMNT().SpecAlgoFMNT().FMNtByMaxEvid().Val().SigmaPds();
}

const double & cParamFusionMNT::SigmaPds()const 
{
   return ParamAlgoFusionMNT().SpecAlgoFMNT().FMNtByMaxEvid().Val().SigmaPds();
}


cTplValGesInit< double > & cParamFusionMNT::SigmaZ()
{
   return ParamAlgoFusionMNT().SpecAlgoFMNT().FMNtByMaxEvid().Val().SigmaZ();
}

const cTplValGesInit< double > & cParamFusionMNT::SigmaZ()const 
{
   return ParamAlgoFusionMNT().SpecAlgoFMNT().FMNtByMaxEvid().Val().SigmaZ();
}


cTplValGesInit< double > & cParamFusionMNT::MaxDif()
{
   return ParamAlgoFusionMNT().SpecAlgoFMNT().FMNtByMaxEvid().Val().MaxDif();
}

const cTplValGesInit< double > & cParamFusionMNT::MaxDif()const 
{
   return ParamAlgoFusionMNT().SpecAlgoFMNT().FMNtByMaxEvid().Val().MaxDif();
}


cTplValGesInit< bool > & cParamFusionMNT::QuickExp()
{
   return ParamAlgoFusionMNT().SpecAlgoFMNT().FMNtByMaxEvid().Val().QuickExp();
}

const cTplValGesInit< bool > & cParamFusionMNT::QuickExp()const 
{
   return ParamAlgoFusionMNT().SpecAlgoFMNT().FMNtByMaxEvid().Val().QuickExp();
}


double & cParamFusionMNT::Regul()
{
   return ParamAlgoFusionMNT().SpecAlgoFMNT().FMNtByMaxEvid().Val().FMNT_ProgDyn().Val().Regul();
}

const double & cParamFusionMNT::Regul()const 
{
   return ParamAlgoFusionMNT().SpecAlgoFMNT().FMNtByMaxEvid().Val().FMNT_ProgDyn().Val().Regul();
}


double & cParamFusionMNT::Sigma0()
{
   return ParamAlgoFusionMNT().SpecAlgoFMNT().FMNtByMaxEvid().Val().FMNT_ProgDyn().Val().Sigma0();
}

const double & cParamFusionMNT::Sigma0()const 
{
   return ParamAlgoFusionMNT().SpecAlgoFMNT().FMNtByMaxEvid().Val().FMNT_ProgDyn().Val().Sigma0();
}


int & cParamFusionMNT::NbDir()
{
   return ParamAlgoFusionMNT().SpecAlgoFMNT().FMNtByMaxEvid().Val().FMNT_ProgDyn().Val().NbDir();
}

const int & cParamFusionMNT::NbDir()const 
{
   return ParamAlgoFusionMNT().SpecAlgoFMNT().FMNtByMaxEvid().Val().FMNT_ProgDyn().Val().NbDir();
}


double & cParamFusionMNT::PenteMax()
{
   return ParamAlgoFusionMNT().SpecAlgoFMNT().FMNtByMaxEvid().Val().FMNT_ProgDyn().Val().FMNT_GesNoVal().Val().PenteMax();
}

const double & cParamFusionMNT::PenteMax()const 
{
   return ParamAlgoFusionMNT().SpecAlgoFMNT().FMNtByMaxEvid().Val().FMNT_ProgDyn().Val().FMNT_GesNoVal().Val().PenteMax();
}


double & cParamFusionMNT::GainNoVal()
{
   return ParamAlgoFusionMNT().SpecAlgoFMNT().FMNtByMaxEvid().Val().FMNT_ProgDyn().Val().FMNT_GesNoVal().Val().GainNoVal();
}

const double & cParamFusionMNT::GainNoVal()const 
{
   return ParamAlgoFusionMNT().SpecAlgoFMNT().FMNtByMaxEvid().Val().FMNT_ProgDyn().Val().FMNT_GesNoVal().Val().GainNoVal();
}


double & cParamFusionMNT::Trans()
{
   return ParamAlgoFusionMNT().SpecAlgoFMNT().FMNtByMaxEvid().Val().FMNT_ProgDyn().Val().FMNT_GesNoVal().Val().Trans();
}

const double & cParamFusionMNT::Trans()const 
{
   return ParamAlgoFusionMNT().SpecAlgoFMNT().FMNtByMaxEvid().Val().FMNT_ProgDyn().Val().FMNT_GesNoVal().Val().Trans();
}


cTplValGesInit< cFMNT_GesNoVal > & cParamFusionMNT::FMNT_GesNoVal()
{
   return ParamAlgoFusionMNT().SpecAlgoFMNT().FMNtByMaxEvid().Val().FMNT_ProgDyn().Val().FMNT_GesNoVal();
}

const cTplValGesInit< cFMNT_GesNoVal > & cParamFusionMNT::FMNT_GesNoVal()const 
{
   return ParamAlgoFusionMNT().SpecAlgoFMNT().FMNtByMaxEvid().Val().FMNT_ProgDyn().Val().FMNT_GesNoVal();
}


cTplValGesInit< cFMNT_ProgDyn > & cParamFusionMNT::FMNT_ProgDyn()
{
   return ParamAlgoFusionMNT().SpecAlgoFMNT().FMNtByMaxEvid().Val().FMNT_ProgDyn();
}

const cTplValGesInit< cFMNT_ProgDyn > & cParamFusionMNT::FMNT_ProgDyn()const 
{
   return ParamAlgoFusionMNT().SpecAlgoFMNT().FMNtByMaxEvid().Val().FMNT_ProgDyn();
}


cTplValGesInit< cFMNtByMaxEvid > & cParamFusionMNT::FMNtByMaxEvid()
{
   return ParamAlgoFusionMNT().SpecAlgoFMNT().FMNtByMaxEvid();
}

const cTplValGesInit< cFMNtByMaxEvid > & cParamFusionMNT::FMNtByMaxEvid()const 
{
   return ParamAlgoFusionMNT().SpecAlgoFMNT().FMNtByMaxEvid();
}


cSpecAlgoFMNT & cParamFusionMNT::SpecAlgoFMNT()
{
   return ParamAlgoFusionMNT().SpecAlgoFMNT();
}

const cSpecAlgoFMNT & cParamFusionMNT::SpecAlgoFMNT()const 
{
   return ParamAlgoFusionMNT().SpecAlgoFMNT();
}


cParamAlgoFusionMNT & cParamFusionMNT::ParamAlgoFusionMNT()
{
   return mParamAlgoFusionMNT;
}

const cParamAlgoFusionMNT & cParamFusionMNT::ParamAlgoFusionMNT()const 
{
   return mParamAlgoFusionMNT;
}


cParamGenereStr & cParamFusionMNT::GenereRes()
{
   return mGenereRes;
}

const cParamGenereStr & cParamFusionMNT::GenereRes()const 
{
   return mGenereRes;
}


cParamGenereStrVois & cParamFusionMNT::GenereInput()
{
   return mGenereInput;
}

const cParamGenereStrVois & cParamFusionMNT::GenereInput()const 
{
   return mGenereInput;
}


cTplValGesInit< int > & cParamFusionMNT::SzDalles()
{
   return SectionGestionChantier().SzDalles();
}

const cTplValGesInit< int > & cParamFusionMNT::SzDalles()const 
{
   return SectionGestionChantier().SzDalles();
}


cTplValGesInit< int > & cParamFusionMNT::RecouvrtDalles()
{
   return SectionGestionChantier().RecouvrtDalles();
}

const cTplValGesInit< int > & cParamFusionMNT::RecouvrtDalles()const 
{
   return SectionGestionChantier().RecouvrtDalles();
}


cTplValGesInit< std::string > & cParamFusionMNT::ParalMkF()
{
   return SectionGestionChantier().ParalMkF();
}

const cTplValGesInit< std::string > & cParamFusionMNT::ParalMkF()const 
{
   return SectionGestionChantier().ParalMkF();
}


cTplValGesInit< bool > & cParamFusionMNT::InterneCalledByProcess()
{
   return SectionGestionChantier().InterneCalledByProcess();
}

const cTplValGesInit< bool > & cParamFusionMNT::InterneCalledByProcess()const 
{
   return SectionGestionChantier().InterneCalledByProcess();
}


cTplValGesInit< std::string > & cParamFusionMNT::InterneSingleImage()
{
   return SectionGestionChantier().InterneSingleImage();
}

const cTplValGesInit< std::string > & cParamFusionMNT::InterneSingleImage()const 
{
   return SectionGestionChantier().InterneSingleImage();
}


cTplValGesInit< int > & cParamFusionMNT::InterneSingleBox()
{
   return SectionGestionChantier().InterneSingleBox();
}

const cTplValGesInit< int > & cParamFusionMNT::InterneSingleBox()const 
{
   return SectionGestionChantier().InterneSingleBox();
}


cTplValGesInit< std::string > & cParamFusionMNT::WorkDirPFM()
{
   return SectionGestionChantier().WorkDirPFM();
}

const cTplValGesInit< std::string > & cParamFusionMNT::WorkDirPFM()const 
{
   return SectionGestionChantier().WorkDirPFM();
}


cTplValGesInit< Box2di > & cParamFusionMNT::BoxTest()
{
   return SectionGestionChantier().BoxTest();
}

const cTplValGesInit< Box2di > & cParamFusionMNT::BoxTest()const 
{
   return SectionGestionChantier().BoxTest();
}


cSectionGestionChantier & cParamFusionMNT::SectionGestionChantier()
{
   return mSectionGestionChantier;
}

const cSectionGestionChantier & cParamFusionMNT::SectionGestionChantier()const 
{
   return mSectionGestionChantier;
}

cElXMLTree * ToXMLTree(const cParamFusionMNT & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ParamFusionMNT",eXMLBranche);
   if (anObj.DicoLoc().IsInit())
      aRes->AddFils(ToXMLTree(anObj.DicoLoc().Val())->ReTagThis("DicoLoc"));
   aRes->AddFils(ToXMLTree(anObj.SectionName())->ReTagThis("SectionName"));
   if (anObj.SectionScoreQualite().IsInit())
      aRes->AddFils(ToXMLTree(anObj.SectionScoreQualite().Val())->ReTagThis("SectionScoreQualite"));
   aRes->AddFils(ToXMLTree(anObj.ParamAlgoFusionMNT())->ReTagThis("ParamAlgoFusionMNT"));
   aRes->AddFils(ToXMLTree(anObj.GenereRes())->ReTagThis("GenereRes"));
   aRes->AddFils(ToXMLTree(anObj.GenereInput())->ReTagThis("GenereInput"));
   aRes->AddFils(ToXMLTree(anObj.SectionGestionChantier())->ReTagThis("SectionGestionChantier"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cParamFusionMNT & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.DicoLoc(),aTree->Get("DicoLoc",1)); //tototo 

   xml_init(anObj.SectionName(),aTree->Get("SectionName",1)); //tototo 

   xml_init(anObj.SectionScoreQualite(),aTree->Get("SectionScoreQualite",1)); //tototo 

   xml_init(anObj.ParamAlgoFusionMNT(),aTree->Get("ParamAlgoFusionMNT",1)); //tototo 

   xml_init(anObj.GenereRes(),aTree->Get("GenereRes",1)); //tototo 

   xml_init(anObj.GenereInput(),aTree->Get("GenereInput",1)); //tototo 

   xml_init(anObj.SectionGestionChantier(),aTree->Get("SectionGestionChantier",1)); //tototo 
}

// };
