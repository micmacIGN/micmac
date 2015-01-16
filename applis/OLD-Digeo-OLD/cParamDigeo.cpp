#include "general/all.h"
#include "private/all.h"
#include "cParamDigeo.h"
namespace NS_ParamDigeo{
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
   ELISE_ASSERT(false,"Bad Value in eToString for enum value ");
   return "";
}

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eTypeTopolPt & anObj)
{
      return  cElXMLTree::ValueNode(aNameTag,eToString(anObj));
}

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
   ELISE_ASSERT(false,"Bad Value in eToString for enum value ");
   return "";
}

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eReducDemiImage & anObj)
{
      return  cElXMLTree::ValueNode(aNameTag,eToString(anObj));
}


cTplValGesInit< int > & cParamExtractCaracIm::SzMinImDeZoom()
{
   return mSzMinImDeZoom;
}

const cTplValGesInit< int > & cParamExtractCaracIm::SzMinImDeZoom()const 
{
   return mSzMinImDeZoom;
}

cElXMLTree * ToXMLTree(const cParamExtractCaracIm & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ParamExtractCaracIm",eXMLBranche);
   if (anObj.SzMinImDeZoom().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SzMinImDeZoom"),anObj.SzMinImDeZoom().Val())->ReTagThis("SzMinImDeZoom"));
  return aRes;
}

void xml_init(cParamExtractCaracIm & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.SzMinImDeZoom(),aTree->Get("SzMinImDeZoom",1),int(500)); //tototo 
}


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

cElXMLTree * ToXMLTree(const cParamVisuCarac & anObj)
{
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
  return aRes;
}

void xml_init(cParamVisuCarac & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.DynGray(),aTree->Get("DynGray",1),int(128)); //tototo 

   xml_init(anObj.Dir(),aTree->Get("Dir",1)); //tototo 

   xml_init(anObj.Zoom(),aTree->Get("Zoom",1),int(1)); //tototo 

   xml_init(anObj.Dyn(),aTree->Get("Dyn",1)); //tototo 

   xml_init(anObj.Prefix(),aTree->Get("Prefix",1),std::string("VisuCarac_")); //tototo 

   xml_init(anObj.ShowCaracEchec(),aTree->Get("ShowCaracEchec",1),bool(false)); //tototo 
}


cTplValGesInit< std::string > & cPredicteurGeom::Unused()
{
   return mUnused;
}

const cTplValGesInit< std::string > & cPredicteurGeom::Unused()const 
{
   return mUnused;
}

cElXMLTree * ToXMLTree(const cPredicteurGeom & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"PredicteurGeom",eXMLBranche);
   if (anObj.Unused().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Unused"),anObj.Unused().Val())->ReTagThis("Unused"));
  return aRes;
}

void xml_init(cPredicteurGeom & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.Unused(),aTree->Get("Unused",1)); //tototo 
}


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


cTplValGesInit< Box2di > & cImageDigeo::BoxIm()
{
   return mBoxIm;
}

const cTplValGesInit< Box2di > & cImageDigeo::BoxIm()const 
{
   return mBoxIm;
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

cElXMLTree * ToXMLTree(const cImageDigeo & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ImageDigeo",eXMLBranche);
   if (anObj.VisuCarac().IsInit())
      aRes->AddFils(ToXMLTree(anObj.VisuCarac().Val())->ReTagThis("VisuCarac"));
   aRes->AddFils(::ToXMLTree(std::string("KeyOrPat"),anObj.KeyOrPat())->ReTagThis("KeyOrPat"));
   if (anObj.KeyCalcCalib().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("KeyCalcCalib"),anObj.KeyCalcCalib().Val())->ReTagThis("KeyCalcCalib"));
   if (anObj.BoxIm().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("BoxIm"),anObj.BoxIm().Val())->ReTagThis("BoxIm"));
   if (anObj.PredicteurGeom().IsInit())
      aRes->AddFils(ToXMLTree(anObj.PredicteurGeom().Val())->ReTagThis("PredicteurGeom"));
  return aRes;
}

void xml_init(cImageDigeo & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.VisuCarac(),aTree->Get("VisuCarac",1)); //tototo 

   xml_init(anObj.KeyOrPat(),aTree->Get("KeyOrPat",1)); //tototo 

   xml_init(anObj.KeyCalcCalib(),aTree->Get("KeyCalcCalib",1)); //tototo 

   xml_init(anObj.BoxIm(),aTree->Get("BoxIm",1)); //tototo 

   xml_init(anObj.PredicteurGeom(),aTree->Get("PredicteurGeom",1)); //tototo 
}


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

cElXMLTree * ToXMLTree(const cTypeNumeriqueOfNiv & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"TypeNumeriqueOfNiv",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Type"),anObj.Type())->ReTagThis("Type"));
   aRes->AddFils(::ToXMLTree(std::string("Niv"),anObj.Niv())->ReTagThis("Niv"));
  return aRes;
}

void xml_init(cTypeNumeriqueOfNiv & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.Type(),aTree->Get("Type",1)); //tototo 

   xml_init(anObj.Niv(),aTree->Get("Niv",1)); //tototo 
}


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

cElXMLTree * ToXMLTree(const cPyramideGaussienne & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"PyramideGaussienne",eXMLBranche);
   if (anObj.NbByOctave().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NbByOctave"),anObj.NbByOctave().Val())->ReTagThis("NbByOctave"));
   if (anObj.Sigma0().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Sigma0"),anObj.Sigma0().Val())->ReTagThis("Sigma0"));
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
  return aRes;
}

void xml_init(cPyramideGaussienne & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.NbByOctave(),aTree->Get("NbByOctave",1),int(3)); //tototo 

   xml_init(anObj.Sigma0(),aTree->Get("Sigma0",1),double(1.6)); //tototo 

   xml_init(anObj.NbInLastOctave(),aTree->Get("NbInLastOctave",1)); //tototo 

   xml_init(anObj.IndexFreqInFirstOctave(),aTree->Get("IndexFreqInFirstOctave",1),int(0)); //tototo 

   xml_init(anObj.NivOctaveMax(),aTree->Get("NivOctaveMax",1)); //tototo 

   xml_init(anObj.ConvolFirstImage(),aTree->Get("ConvolFirstImage",1),double(-1)); //tototo 

   xml_init(anObj.EpsilonGauss(),aTree->Get("EpsilonGauss",1),double(1e-3)); //tototo 

   xml_init(anObj.NbShift(),aTree->Get("NbShift",1),int(15)); //tototo 

   xml_init(anObj.SurEchIntegralGauss(),aTree->Get("SurEchIntegralGauss",1),int(10)); //tototo 

   xml_init(anObj.ConvolIncrem(),aTree->Get("ConvolIncrem",1),bool(true)); //tototo 
}


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

cElXMLTree * ToXMLTree(const cTypePyramide & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"TypePyramide",eXMLBranche);
   if (anObj.NivPyramBasique().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NivPyramBasique"),anObj.NivPyramBasique().Val())->ReTagThis("NivPyramBasique"));
   if (anObj.PyramideGaussienne().IsInit())
      aRes->AddFils(ToXMLTree(anObj.PyramideGaussienne().Val())->ReTagThis("PyramideGaussienne"));
  return aRes;
}

void xml_init(cTypePyramide & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.NivPyramBasique(),aTree->Get("NivPyramBasique",1)); //tototo 

   xml_init(anObj.PyramideGaussienne(),aTree->Get("PyramideGaussienne",1)); //tototo 
}


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

cElXMLTree * ToXMLTree(const cPyramideImage & anObj)
{
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
  return aRes;
}

void xml_init(cPyramideImage & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.TypeNumeriqueOfNiv(),aTree->GetAll("TypeNumeriqueOfNiv",false,1));

   xml_init(anObj.MaximDyn(),aTree->Get("MaximDyn",1)); //tototo 

   xml_init(anObj.ValMaxForDyn(),aTree->Get("ValMaxForDyn",1)); //tototo 

   xml_init(anObj.ReducDemiImage(),aTree->Get("ReducDemiImage",1),eReducDemiImage(eRDI_121)); //tototo 

   xml_init(anObj.TypePyramide(),aTree->Get("TypePyramide",1)); //tototo 
}


std::list< cImageDigeo > & cSectionImages::ImageDigeo()
{
   return mImageDigeo;
}

const std::list< cImageDigeo > & cSectionImages::ImageDigeo()const 
{
   return mImageDigeo;
}


std::list< cTypeNumeriqueOfNiv > & cSectionImages::TypeNumeriqueOfNiv()
{
   return PyramideImage().TypeNumeriqueOfNiv();
}

const std::list< cTypeNumeriqueOfNiv > & cSectionImages::TypeNumeriqueOfNiv()const 
{
   return PyramideImage().TypeNumeriqueOfNiv();
}


cTplValGesInit< bool > & cSectionImages::MaximDyn()
{
   return PyramideImage().MaximDyn();
}

const cTplValGesInit< bool > & cSectionImages::MaximDyn()const 
{
   return PyramideImage().MaximDyn();
}


cTplValGesInit< double > & cSectionImages::ValMaxForDyn()
{
   return PyramideImage().ValMaxForDyn();
}

const cTplValGesInit< double > & cSectionImages::ValMaxForDyn()const 
{
   return PyramideImage().ValMaxForDyn();
}


cTplValGesInit< eReducDemiImage > & cSectionImages::ReducDemiImage()
{
   return PyramideImage().ReducDemiImage();
}

const cTplValGesInit< eReducDemiImage > & cSectionImages::ReducDemiImage()const 
{
   return PyramideImage().ReducDemiImage();
}


cTplValGesInit< int > & cSectionImages::NivPyramBasique()
{
   return PyramideImage().TypePyramide().NivPyramBasique();
}

const cTplValGesInit< int > & cSectionImages::NivPyramBasique()const 
{
   return PyramideImage().TypePyramide().NivPyramBasique();
}


cTplValGesInit< int > & cSectionImages::NbByOctave()
{
   return PyramideImage().TypePyramide().PyramideGaussienne().Val().NbByOctave();
}

const cTplValGesInit< int > & cSectionImages::NbByOctave()const 
{
   return PyramideImage().TypePyramide().PyramideGaussienne().Val().NbByOctave();
}


cTplValGesInit< double > & cSectionImages::Sigma0()
{
   return PyramideImage().TypePyramide().PyramideGaussienne().Val().Sigma0();
}

const cTplValGesInit< double > & cSectionImages::Sigma0()const 
{
   return PyramideImage().TypePyramide().PyramideGaussienne().Val().Sigma0();
}


cTplValGesInit< int > & cSectionImages::NbInLastOctave()
{
   return PyramideImage().TypePyramide().PyramideGaussienne().Val().NbInLastOctave();
}

const cTplValGesInit< int > & cSectionImages::NbInLastOctave()const 
{
   return PyramideImage().TypePyramide().PyramideGaussienne().Val().NbInLastOctave();
}


cTplValGesInit< int > & cSectionImages::IndexFreqInFirstOctave()
{
   return PyramideImage().TypePyramide().PyramideGaussienne().Val().IndexFreqInFirstOctave();
}

const cTplValGesInit< int > & cSectionImages::IndexFreqInFirstOctave()const 
{
   return PyramideImage().TypePyramide().PyramideGaussienne().Val().IndexFreqInFirstOctave();
}


int & cSectionImages::NivOctaveMax()
{
   return PyramideImage().TypePyramide().PyramideGaussienne().Val().NivOctaveMax();
}

const int & cSectionImages::NivOctaveMax()const 
{
   return PyramideImage().TypePyramide().PyramideGaussienne().Val().NivOctaveMax();
}


cTplValGesInit< double > & cSectionImages::ConvolFirstImage()
{
   return PyramideImage().TypePyramide().PyramideGaussienne().Val().ConvolFirstImage();
}

const cTplValGesInit< double > & cSectionImages::ConvolFirstImage()const 
{
   return PyramideImage().TypePyramide().PyramideGaussienne().Val().ConvolFirstImage();
}


cTplValGesInit< double > & cSectionImages::EpsilonGauss()
{
   return PyramideImage().TypePyramide().PyramideGaussienne().Val().EpsilonGauss();
}

const cTplValGesInit< double > & cSectionImages::EpsilonGauss()const 
{
   return PyramideImage().TypePyramide().PyramideGaussienne().Val().EpsilonGauss();
}


cTplValGesInit< int > & cSectionImages::NbShift()
{
   return PyramideImage().TypePyramide().PyramideGaussienne().Val().NbShift();
}

const cTplValGesInit< int > & cSectionImages::NbShift()const 
{
   return PyramideImage().TypePyramide().PyramideGaussienne().Val().NbShift();
}


cTplValGesInit< int > & cSectionImages::SurEchIntegralGauss()
{
   return PyramideImage().TypePyramide().PyramideGaussienne().Val().SurEchIntegralGauss();
}

const cTplValGesInit< int > & cSectionImages::SurEchIntegralGauss()const 
{
   return PyramideImage().TypePyramide().PyramideGaussienne().Val().SurEchIntegralGauss();
}


cTplValGesInit< bool > & cSectionImages::ConvolIncrem()
{
   return PyramideImage().TypePyramide().PyramideGaussienne().Val().ConvolIncrem();
}

const cTplValGesInit< bool > & cSectionImages::ConvolIncrem()const 
{
   return PyramideImage().TypePyramide().PyramideGaussienne().Val().ConvolIncrem();
}


cTplValGesInit< cPyramideGaussienne > & cSectionImages::PyramideGaussienne()
{
   return PyramideImage().TypePyramide().PyramideGaussienne();
}

const cTplValGesInit< cPyramideGaussienne > & cSectionImages::PyramideGaussienne()const 
{
   return PyramideImage().TypePyramide().PyramideGaussienne();
}


cTypePyramide & cSectionImages::TypePyramide()
{
   return PyramideImage().TypePyramide();
}

const cTypePyramide & cSectionImages::TypePyramide()const 
{
   return PyramideImage().TypePyramide();
}


cPyramideImage & cSectionImages::PyramideImage()
{
   return mPyramideImage;
}

const cPyramideImage & cSectionImages::PyramideImage()const 
{
   return mPyramideImage;
}

cElXMLTree * ToXMLTree(const cSectionImages & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"SectionImages",eXMLBranche);
  for
  (       std::list< cImageDigeo >::const_iterator it=anObj.ImageDigeo().begin();
      it !=anObj.ImageDigeo().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("ImageDigeo"));
   aRes->AddFils(ToXMLTree(anObj.PyramideImage())->ReTagThis("PyramideImage"));
  return aRes;
}

void xml_init(cSectionImages & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.ImageDigeo(),aTree->GetAll("ImageDigeo",false,1));

   xml_init(anObj.PyramideImage(),aTree->Get("PyramideImage",1)); //tototo 
}


eTypeTopolPt & cOneCarac::Type()
{
   return mType;
}

const eTypeTopolPt & cOneCarac::Type()const 
{
   return mType;
}

cElXMLTree * ToXMLTree(const cOneCarac & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"OneCarac",eXMLBranche);
   aRes->AddFils(ToXMLTree(std::string("Type"),anObj.Type())->ReTagThis("Type"));
  return aRes;
}

void xml_init(cOneCarac & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.Type(),aTree->Get("Type",1)); //tototo 
}


std::list< cOneCarac > & cCaracTopo::OneCarac()
{
   return mOneCarac;
}

const std::list< cOneCarac > & cCaracTopo::OneCarac()const 
{
   return mOneCarac;
}

cElXMLTree * ToXMLTree(const cCaracTopo & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"CaracTopo",eXMLBranche);
  for
  (       std::list< cOneCarac >::const_iterator it=anObj.OneCarac().begin();
      it !=anObj.OneCarac().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("OneCarac"));
  return aRes;
}

void xml_init(cCaracTopo & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.OneCarac(),aTree->GetAll("OneCarac",false,1));
}


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

cElXMLTree * ToXMLTree(const cSiftCarac & anObj)
{
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
  return aRes;
}

void xml_init(cSiftCarac & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.DoMax(),aTree->Get("DoMax",1),bool(true)); //tototo 

   xml_init(anObj.DoMin(),aTree->Get("DoMin",1),bool(true)); //tototo 

   xml_init(anObj.NivEstimGradMoy(),aTree->Get("NivEstimGradMoy",1),int(4)); //tototo 

   xml_init(anObj.RatioAllongMin(),aTree->Get("RatioAllongMin",1),double(8.0)); //tototo 

   xml_init(anObj.RatioGrad(),aTree->Get("RatioGrad",1),double(0.01)); //tototo 
}


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

cElXMLTree * ToXMLTree(const cSectionCaracImages & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"SectionCaracImages",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("ComputeCarac"),anObj.ComputeCarac())->ReTagThis("ComputeCarac"));
   if (anObj.CaracTopo().IsInit())
      aRes->AddFils(ToXMLTree(anObj.CaracTopo().Val())->ReTagThis("CaracTopo"));
   if (anObj.SiftCarac().IsInit())
      aRes->AddFils(ToXMLTree(anObj.SiftCarac().Val())->ReTagThis("SiftCarac"));
  return aRes;
}

void xml_init(cSectionCaracImages & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.ComputeCarac(),aTree->Get("ComputeCarac",1)); //tototo 

   xml_init(anObj.CaracTopo(),aTree->Get("CaracTopo",1)); //tototo 

   xml_init(anObj.SiftCarac(),aTree->Get("SiftCarac",1)); //tototo 
}


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

cElXMLTree * ToXMLTree(const cGenereRandomRect & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"GenereRandomRect",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("NbRect"),anObj.NbRect())->ReTagThis("NbRect"));
   aRes->AddFils(::ToXMLTree(std::string("SzRect"),anObj.SzRect())->ReTagThis("SzRect"));
  return aRes;
}

void xml_init(cGenereRandomRect & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.NbRect(),aTree->Get("NbRect",1)); //tototo 

   xml_init(anObj.SzRect(),aTree->Get("SzRect",1)); //tototo 
}


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

cElXMLTree * ToXMLTree(const cGenereCarroyage & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"GenereCarroyage",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("PerX"),anObj.PerX())->ReTagThis("PerX"));
   aRes->AddFils(::ToXMLTree(std::string("PerY"),anObj.PerY())->ReTagThis("PerY"));
  return aRes;
}

void xml_init(cGenereCarroyage & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.PerX(),aTree->Get("PerX",1)); //tototo 

   xml_init(anObj.PerY(),aTree->Get("PerY",1)); //tototo 
}


int & cGenereAllRandom::SzFilter()
{
   return mSzFilter;
}

const int & cGenereAllRandom::SzFilter()const 
{
   return mSzFilter;
}

cElXMLTree * ToXMLTree(const cGenereAllRandom & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"GenereAllRandom",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("SzFilter"),anObj.SzFilter())->ReTagThis("SzFilter"));
  return aRes;
}

void xml_init(cGenereAllRandom & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.SzFilter(),aTree->Get("SzFilter",1)); //tototo 
}


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

cElXMLTree * ToXMLTree(const cSectionTest & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"SectionTest",eXMLBranche);
   if (anObj.GenereRandomRect().IsInit())
      aRes->AddFils(ToXMLTree(anObj.GenereRandomRect().Val())->ReTagThis("GenereRandomRect"));
   if (anObj.GenereCarroyage().IsInit())
      aRes->AddFils(ToXMLTree(anObj.GenereCarroyage().Val())->ReTagThis("GenereCarroyage"));
   if (anObj.GenereAllRandom().IsInit())
      aRes->AddFils(ToXMLTree(anObj.GenereAllRandom().Val())->ReTagThis("GenereAllRandom"));
   if (anObj.VerifExtrema().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("VerifExtrema"),anObj.VerifExtrema().Val())->ReTagThis("VerifExtrema"));
  return aRes;
}

void xml_init(cSectionTest & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.GenereRandomRect(),aTree->Get("GenereRandomRect",1)); //tototo 

   xml_init(anObj.GenereCarroyage(),aTree->Get("GenereCarroyage",1)); //tototo 

   xml_init(anObj.GenereAllRandom(),aTree->Get("GenereAllRandom",1)); //tototo 

   xml_init(anObj.VerifExtrema(),aTree->Get("VerifExtrema",1),bool(false)); //tototo 
}


cTplValGesInit< std::string > & cSauvPyram::Dir()
{
   return mDir;
}

const cTplValGesInit< std::string > & cSauvPyram::Dir()const 
{
   return mDir;
}


cTplValGesInit< std::string > & cSauvPyram::Key()
{
   return mKey;
}

const cTplValGesInit< std::string > & cSauvPyram::Key()const 
{
   return mKey;
}


cTplValGesInit< bool > & cSauvPyram::CreateFileWhenExist()
{
   return mCreateFileWhenExist;
}

const cTplValGesInit< bool > & cSauvPyram::CreateFileWhenExist()const 
{
   return mCreateFileWhenExist;
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

cElXMLTree * ToXMLTree(const cSauvPyram & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"SauvPyram",eXMLBranche);
   if (anObj.Dir().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Dir"),anObj.Dir().Val())->ReTagThis("Dir"));
   if (anObj.Key().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Key"),anObj.Key().Val())->ReTagThis("Key"));
   if (anObj.CreateFileWhenExist().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("CreateFileWhenExist"),anObj.CreateFileWhenExist().Val())->ReTagThis("CreateFileWhenExist"));
   if (anObj.StripTifFile().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("StripTifFile"),anObj.StripTifFile().Val())->ReTagThis("StripTifFile"));
   if (anObj.Force8B().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Force8B"),anObj.Force8B().Val())->ReTagThis("Force8B"));
   if (anObj.Dyn().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Dyn"),anObj.Dyn().Val())->ReTagThis("Dyn"));
  return aRes;
}

void xml_init(cSauvPyram & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.Dir(),aTree->Get("Dir",1),std::string("")); //tototo 

   xml_init(anObj.Key(),aTree->Get("Key",1),std::string("Key-Assoc-Pyram-MM")); //tototo 

   xml_init(anObj.CreateFileWhenExist(),aTree->Get("CreateFileWhenExist",1),bool(false)); //tototo 

   xml_init(anObj.StripTifFile(),aTree->Get("StripTifFile",1),int(100)); //tototo 

   xml_init(anObj.Force8B(),aTree->Get("Force8B",1),bool(false)); //tototo 

   xml_init(anObj.Dyn(),aTree->Get("Dyn",1),double(1.0)); //tototo 
}


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

cElXMLTree * ToXMLTree(const cDigeoDecoupageCarac & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"DigeoDecoupageCarac",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("SzDalle"),anObj.SzDalle())->ReTagThis("SzDalle"));
   aRes->AddFils(::ToXMLTree(std::string("Bord"),anObj.Bord())->ReTagThis("Bord"));
  return aRes;
}

void xml_init(cDigeoDecoupageCarac & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.SzDalle(),aTree->Get("SzDalle",1)); //tototo 

   xml_init(anObj.Bord(),aTree->Get("Bord",1)); //tototo 
}


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

cElXMLTree * ToXMLTree(const cModifGCC & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ModifGCC",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("NbByOctave"),anObj.NbByOctave())->ReTagThis("NbByOctave"));
   aRes->AddFils(::ToXMLTree(std::string("ConvolIncrem"),anObj.ConvolIncrem())->ReTagThis("ConvolIncrem"));
  return aRes;
}

void xml_init(cModifGCC & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.NbByOctave(),aTree->Get("NbByOctave",1)); //tototo 

   xml_init(anObj.ConvolIncrem(),aTree->Get("ConvolIncrem",1)); //tototo 
}


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

cElXMLTree * ToXMLTree(const cGenereCodeConvol & anObj)
{
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
  return aRes;
}

void xml_init(cGenereCodeConvol & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.Dir(),aTree->Get("Dir",1),std::string("applis/Digeo/")); //tototo 

   xml_init(anObj.File(),aTree->Get("File",1),std::string("GenConvolSpec")); //tototo 

   xml_init(anObj.ModifGCC(),aTree->GetAll("ModifGCC",false,1));
}


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

cElXMLTree * ToXMLTree(const cFenVisu & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"FenVisu",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Name"),anObj.Name())->ReTagThis("Name"));
   aRes->AddFils(::ToXMLTree(std::string("Sz"),anObj.Sz())->ReTagThis("Sz"));
  return aRes;
}

void xml_init(cFenVisu & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.Name(),aTree->Get("Name",1)); //tototo 

   xml_init(anObj.Sz(),aTree->Get("Sz",1)); //tototo 
}


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


cTplValGesInit< std::string > & cSectionWorkSpace::Key()
{
   return SauvPyram().Val().Key();
}

const cTplValGesInit< std::string > & cSectionWorkSpace::Key()const 
{
   return SauvPyram().Val().Key();
}


cTplValGesInit< bool > & cSectionWorkSpace::CreateFileWhenExist()
{
   return SauvPyram().Val().CreateFileWhenExist();
}

const cTplValGesInit< bool > & cSectionWorkSpace::CreateFileWhenExist()const 
{
   return SauvPyram().Val().CreateFileWhenExist();
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


cTplValGesInit< bool > & cSectionWorkSpace::UseConvolSpec()
{
   return mUseConvolSpec;
}

const cTplValGesInit< bool > & cSectionWorkSpace::UseConvolSpec()const 
{
   return mUseConvolSpec;
}


cTplValGesInit< bool > & cSectionWorkSpace::ShowConvolSpec()
{
   return mShowConvolSpec;
}

const cTplValGesInit< bool > & cSectionWorkSpace::ShowConvolSpec()const 
{
   return mShowConvolSpec;
}

cElXMLTree * ToXMLTree(const cSectionWorkSpace & anObj)
{
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
   if (anObj.UseConvolSpec().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("UseConvolSpec"),anObj.UseConvolSpec().Val())->ReTagThis("UseConvolSpec"));
   if (anObj.ShowConvolSpec().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ShowConvolSpec"),anObj.ShowConvolSpec().Val())->ReTagThis("ShowConvolSpec"));
  return aRes;
}

void xml_init(cSectionWorkSpace & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.DirectoryChantier(),aTree->Get("DirectoryChantier",1),std::string("")); //tototo 

   xml_init(anObj.SauvPyram(),aTree->Get("SauvPyram",1)); //tototo 

   xml_init(anObj.DigeoDecoupageCarac(),aTree->Get("DigeoDecoupageCarac",1)); //tototo 

   xml_init(anObj.ExigeCodeCompile(),aTree->Get("ExigeCodeCompile",1),bool(true)); //tototo 

   xml_init(anObj.GenereCodeConvol(),aTree->Get("GenereCodeConvol",1)); //tototo 

   xml_init(anObj.ShowTimes(),aTree->Get("ShowTimes",1),int(0)); //tototo 

   xml_init(anObj.FenVisu(),aTree->GetAll("FenVisu",false,1));

   xml_init(anObj.UseConvolSpec(),aTree->Get("UseConvolSpec",1),bool(true)); //tototo 

   xml_init(anObj.ShowConvolSpec(),aTree->Get("ShowConvolSpec",1),bool(false)); //tototo 
}


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
   return SectionImages().ImageDigeo();
}

const std::list< cImageDigeo > & cParamDigeo::ImageDigeo()const 
{
   return SectionImages().ImageDigeo();
}


std::list< cTypeNumeriqueOfNiv > & cParamDigeo::TypeNumeriqueOfNiv()
{
   return SectionImages().PyramideImage().TypeNumeriqueOfNiv();
}

const std::list< cTypeNumeriqueOfNiv > & cParamDigeo::TypeNumeriqueOfNiv()const 
{
   return SectionImages().PyramideImage().TypeNumeriqueOfNiv();
}


cTplValGesInit< bool > & cParamDigeo::MaximDyn()
{
   return SectionImages().PyramideImage().MaximDyn();
}

const cTplValGesInit< bool > & cParamDigeo::MaximDyn()const 
{
   return SectionImages().PyramideImage().MaximDyn();
}


cTplValGesInit< double > & cParamDigeo::ValMaxForDyn()
{
   return SectionImages().PyramideImage().ValMaxForDyn();
}

const cTplValGesInit< double > & cParamDigeo::ValMaxForDyn()const 
{
   return SectionImages().PyramideImage().ValMaxForDyn();
}


cTplValGesInit< eReducDemiImage > & cParamDigeo::ReducDemiImage()
{
   return SectionImages().PyramideImage().ReducDemiImage();
}

const cTplValGesInit< eReducDemiImage > & cParamDigeo::ReducDemiImage()const 
{
   return SectionImages().PyramideImage().ReducDemiImage();
}


cTplValGesInit< int > & cParamDigeo::NivPyramBasique()
{
   return SectionImages().PyramideImage().TypePyramide().NivPyramBasique();
}

const cTplValGesInit< int > & cParamDigeo::NivPyramBasique()const 
{
   return SectionImages().PyramideImage().TypePyramide().NivPyramBasique();
}


cTplValGesInit< int > & cParamDigeo::NbByOctave()
{
   return SectionImages().PyramideImage().TypePyramide().PyramideGaussienne().Val().NbByOctave();
}

const cTplValGesInit< int > & cParamDigeo::NbByOctave()const 
{
   return SectionImages().PyramideImage().TypePyramide().PyramideGaussienne().Val().NbByOctave();
}


cTplValGesInit< double > & cParamDigeo::Sigma0()
{
   return SectionImages().PyramideImage().TypePyramide().PyramideGaussienne().Val().Sigma0();
}

const cTplValGesInit< double > & cParamDigeo::Sigma0()const 
{
   return SectionImages().PyramideImage().TypePyramide().PyramideGaussienne().Val().Sigma0();
}


cTplValGesInit< int > & cParamDigeo::NbInLastOctave()
{
   return SectionImages().PyramideImage().TypePyramide().PyramideGaussienne().Val().NbInLastOctave();
}

const cTplValGesInit< int > & cParamDigeo::NbInLastOctave()const 
{
   return SectionImages().PyramideImage().TypePyramide().PyramideGaussienne().Val().NbInLastOctave();
}


cTplValGesInit< int > & cParamDigeo::IndexFreqInFirstOctave()
{
   return SectionImages().PyramideImage().TypePyramide().PyramideGaussienne().Val().IndexFreqInFirstOctave();
}

const cTplValGesInit< int > & cParamDigeo::IndexFreqInFirstOctave()const 
{
   return SectionImages().PyramideImage().TypePyramide().PyramideGaussienne().Val().IndexFreqInFirstOctave();
}


int & cParamDigeo::NivOctaveMax()
{
   return SectionImages().PyramideImage().TypePyramide().PyramideGaussienne().Val().NivOctaveMax();
}

const int & cParamDigeo::NivOctaveMax()const 
{
   return SectionImages().PyramideImage().TypePyramide().PyramideGaussienne().Val().NivOctaveMax();
}


cTplValGesInit< double > & cParamDigeo::ConvolFirstImage()
{
   return SectionImages().PyramideImage().TypePyramide().PyramideGaussienne().Val().ConvolFirstImage();
}

const cTplValGesInit< double > & cParamDigeo::ConvolFirstImage()const 
{
   return SectionImages().PyramideImage().TypePyramide().PyramideGaussienne().Val().ConvolFirstImage();
}


cTplValGesInit< double > & cParamDigeo::EpsilonGauss()
{
   return SectionImages().PyramideImage().TypePyramide().PyramideGaussienne().Val().EpsilonGauss();
}

const cTplValGesInit< double > & cParamDigeo::EpsilonGauss()const 
{
   return SectionImages().PyramideImage().TypePyramide().PyramideGaussienne().Val().EpsilonGauss();
}


cTplValGesInit< int > & cParamDigeo::NbShift()
{
   return SectionImages().PyramideImage().TypePyramide().PyramideGaussienne().Val().NbShift();
}

const cTplValGesInit< int > & cParamDigeo::NbShift()const 
{
   return SectionImages().PyramideImage().TypePyramide().PyramideGaussienne().Val().NbShift();
}


cTplValGesInit< int > & cParamDigeo::SurEchIntegralGauss()
{
   return SectionImages().PyramideImage().TypePyramide().PyramideGaussienne().Val().SurEchIntegralGauss();
}

const cTplValGesInit< int > & cParamDigeo::SurEchIntegralGauss()const 
{
   return SectionImages().PyramideImage().TypePyramide().PyramideGaussienne().Val().SurEchIntegralGauss();
}


cTplValGesInit< bool > & cParamDigeo::ConvolIncrem()
{
   return SectionImages().PyramideImage().TypePyramide().PyramideGaussienne().Val().ConvolIncrem();
}

const cTplValGesInit< bool > & cParamDigeo::ConvolIncrem()const 
{
   return SectionImages().PyramideImage().TypePyramide().PyramideGaussienne().Val().ConvolIncrem();
}


cTplValGesInit< cPyramideGaussienne > & cParamDigeo::PyramideGaussienne()
{
   return SectionImages().PyramideImage().TypePyramide().PyramideGaussienne();
}

const cTplValGesInit< cPyramideGaussienne > & cParamDigeo::PyramideGaussienne()const 
{
   return SectionImages().PyramideImage().TypePyramide().PyramideGaussienne();
}


cTypePyramide & cParamDigeo::TypePyramide()
{
   return SectionImages().PyramideImage().TypePyramide();
}

const cTypePyramide & cParamDigeo::TypePyramide()const 
{
   return SectionImages().PyramideImage().TypePyramide();
}


cPyramideImage & cParamDigeo::PyramideImage()
{
   return SectionImages().PyramideImage();
}

const cPyramideImage & cParamDigeo::PyramideImage()const 
{
   return SectionImages().PyramideImage();
}


cSectionImages & cParamDigeo::SectionImages()
{
   return mSectionImages;
}

const cSectionImages & cParamDigeo::SectionImages()const 
{
   return mSectionImages;
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


cTplValGesInit< std::string > & cParamDigeo::Key()
{
   return SectionWorkSpace().SauvPyram().Val().Key();
}

const cTplValGesInit< std::string > & cParamDigeo::Key()const 
{
   return SectionWorkSpace().SauvPyram().Val().Key();
}


cTplValGesInit< bool > & cParamDigeo::CreateFileWhenExist()
{
   return SectionWorkSpace().SauvPyram().Val().CreateFileWhenExist();
}

const cTplValGesInit< bool > & cParamDigeo::CreateFileWhenExist()const 
{
   return SectionWorkSpace().SauvPyram().Val().CreateFileWhenExist();
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


cTplValGesInit< bool > & cParamDigeo::UseConvolSpec()
{
   return SectionWorkSpace().UseConvolSpec();
}

const cTplValGesInit< bool > & cParamDigeo::UseConvolSpec()const 
{
   return SectionWorkSpace().UseConvolSpec();
}


cTplValGesInit< bool > & cParamDigeo::ShowConvolSpec()
{
   return SectionWorkSpace().ShowConvolSpec();
}

const cTplValGesInit< bool > & cParamDigeo::ShowConvolSpec()const 
{
   return SectionWorkSpace().ShowConvolSpec();
}


cSectionWorkSpace & cParamDigeo::SectionWorkSpace()
{
   return mSectionWorkSpace;
}

const cSectionWorkSpace & cParamDigeo::SectionWorkSpace()const 
{
   return mSectionWorkSpace;
}

cElXMLTree * ToXMLTree(const cParamDigeo & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ParamDigeo",eXMLBranche);
   if (anObj.DicoLoc().IsInit())
      aRes->AddFils(ToXMLTree(anObj.DicoLoc().Val())->ReTagThis("DicoLoc"));
   aRes->AddFils(ToXMLTree(anObj.SectionImages())->ReTagThis("SectionImages"));
   aRes->AddFils(ToXMLTree(anObj.SectionCaracImages())->ReTagThis("SectionCaracImages"));
   if (anObj.SectionTest().IsInit())
      aRes->AddFils(ToXMLTree(anObj.SectionTest().Val())->ReTagThis("SectionTest"));
   aRes->AddFils(ToXMLTree(anObj.SectionWorkSpace())->ReTagThis("SectionWorkSpace"));
  return aRes;
}

void xml_init(cParamDigeo & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.DicoLoc(),aTree->Get("DicoLoc",1)); //tototo 

   xml_init(anObj.SectionImages(),aTree->Get("SectionImages",1)); //tototo 

   xml_init(anObj.SectionCaracImages(),aTree->Get("SectionCaracImages",1)); //tototo 

   xml_init(anObj.SectionTest(),aTree->Get("SectionTest",1)); //tototo 

   xml_init(anObj.SectionWorkSpace(),aTree->Get("SectionWorkSpace",1)); //tototo 
}

};
