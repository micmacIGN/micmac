#include "addon_ParamChantierPhotogram.h"
namespace NS_ParamChantierPhotogram{

int & caffichImg::image()
{
   return mimage;
}

const int & caffichImg::image()const 
{
   return mimage;
}


std::string & caffichImg::fichier()
{
   return mfichier;
}

const std::string & caffichImg::fichier()const 
{
   return mfichier;
}

cElXMLTree * ToXMLTree(const caffichImg & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"affichImg",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("image"),anObj.image())->ReTagThis("image"));
   aRes->AddFils(::ToXMLTree(std::string("fichier"),anObj.fichier())->ReTagThis("fichier"));
  return aRes;
}

void xml_init(caffichImg & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.image(),aTree->Get("image",1)); //tototo 

   xml_init(anObj.fichier(),aTree->Get("fichier",1)); //tototo 
}


int & caffichPaire::image1()
{
   return mimage1;
}

const int & caffichPaire::image1()const 
{
   return mimage1;
}


std::string & caffichPaire::fichier1()
{
   return mfichier1;
}

const std::string & caffichPaire::fichier1()const 
{
   return mfichier1;
}


int & caffichPaire::image2()
{
   return mimage2;
}

const int & caffichPaire::image2()const 
{
   return mimage2;
}


std::string & caffichPaire::fichier2()
{
   return mfichier2;
}

const std::string & caffichPaire::fichier2()const 
{
   return mfichier2;
}


int & caffichPaire::liste()
{
   return mliste;
}

const int & caffichPaire::liste()const 
{
   return mliste;
}


cTplValGesInit< bool > & caffichPaire::trait()
{
   return mtrait;
}

const cTplValGesInit< bool > & caffichPaire::trait()const 
{
   return mtrait;
}

cElXMLTree * ToXMLTree(const caffichPaire & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"affichPaire",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("image1"),anObj.image1())->ReTagThis("image1"));
   aRes->AddFils(::ToXMLTree(std::string("fichier1"),anObj.fichier1())->ReTagThis("fichier1"));
   aRes->AddFils(::ToXMLTree(std::string("image2"),anObj.image2())->ReTagThis("image2"));
   aRes->AddFils(::ToXMLTree(std::string("fichier2"),anObj.fichier2())->ReTagThis("fichier2"));
   aRes->AddFils(::ToXMLTree(std::string("liste"),anObj.liste())->ReTagThis("liste"));
   if (anObj.trait().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("trait"),anObj.trait().Val())->ReTagThis("trait"));
  return aRes;
}

void xml_init(caffichPaire & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.image1(),aTree->Get("image1",1)); //tototo 

   xml_init(anObj.fichier1(),aTree->Get("fichier1",1)); //tototo 

   xml_init(anObj.image2(),aTree->Get("image2",1)); //tototo 

   xml_init(anObj.fichier2(),aTree->Get("fichier2",1)); //tototo 

   xml_init(anObj.liste(),aTree->Get("liste",1)); //tototo 

   xml_init(anObj.trait(),aTree->Get("trait",1),bool(true)); //tototo 
}


string & cParamFusionSift::dossier()
{
   return mdossier;
}

const string & cParamFusionSift::dossier()const 
{
   return mdossier;
}


string & cParamFusionSift::dossierImg()
{
   return mdossierImg;
}

const string & cParamFusionSift::dossierImg()const 
{
   return mdossierImg;
}


cTplValGesInit< std::string > & cParamFusionSift::extensionSortie()
{
   return mextensionSortie;
}

const cTplValGesInit< std::string > & cParamFusionSift::extensionSortie()const 
{
   return mextensionSortie;
}


cTplValGesInit< int > & cParamFusionSift::firstfichier()
{
   return mfirstfichier;
}

const cTplValGesInit< int > & cParamFusionSift::firstfichier()const 
{
   return mfirstfichier;
}


cTplValGesInit< int > & cParamFusionSift::lastfichier()
{
   return mlastfichier;
}

const cTplValGesInit< int > & cParamFusionSift::lastfichier()const 
{
   return mlastfichier;
}


cTplValGesInit< int > & cParamFusionSift::SzMin()
{
   return mSzMin;
}

const cTplValGesInit< int > & cParamFusionSift::SzMin()const 
{
   return mSzMin;
}


cTplValGesInit< int > & cParamFusionSift::NbObjMax()
{
   return mNbObjMax;
}

const cTplValGesInit< int > & cParamFusionSift::NbObjMax()const 
{
   return mNbObjMax;
}


cTplValGesInit< Box2dr > & cParamFusionSift::box()
{
   return mbox;
}

const cTplValGesInit< Box2dr > & cParamFusionSift::box()const 
{
   return mbox;
}


cTplValGesInit< REAL > & cParamFusionSift::distIsol()
{
   return mdistIsol;
}

const cTplValGesInit< REAL > & cParamFusionSift::distIsol()const 
{
   return mdistIsol;
}


cTplValGesInit< int > & cParamFusionSift::ptppi()
{
   return mptppi;
}

const cTplValGesInit< int > & cParamFusionSift::ptppi()const 
{
   return mptppi;
}


cTplValGesInit< double > & cParamFusionSift::mindistalign()
{
   return mmindistalign;
}

const cTplValGesInit< double > & cParamFusionSift::mindistalign()const 
{
   return mmindistalign;
}


cTplValGesInit< bool > & cParamFusionSift::filtre1()
{
   return mfiltre1;
}

const cTplValGesInit< bool > & cParamFusionSift::filtre1()const 
{
   return mfiltre1;
}


cTplValGesInit< bool > & cParamFusionSift::filtre2()
{
   return mfiltre2;
}

const cTplValGesInit< bool > & cParamFusionSift::filtre2()const 
{
   return mfiltre2;
}


cTplValGesInit< bool > & cParamFusionSift::filtre3()
{
   return mfiltre3;
}

const cTplValGesInit< bool > & cParamFusionSift::filtre3()const 
{
   return mfiltre3;
}


cTplValGesInit< REAL > & cParamFusionSift::distIsol2()
{
   return mdistIsol2;
}

const cTplValGesInit< REAL > & cParamFusionSift::distIsol2()const 
{
   return mdistIsol2;
}


cTplValGesInit< bool > & cParamFusionSift::rapide()
{
   return mrapide;
}

const cTplValGesInit< bool > & cParamFusionSift::rapide()const 
{
   return mrapide;
}


cTplValGesInit< double > & cParamFusionSift::aDistInitVois()
{
   return maDistInitVois;
}

const cTplValGesInit< double > & cParamFusionSift::aDistInitVois()const 
{
   return maDistInitVois;
}


cTplValGesInit< double > & cParamFusionSift::aFact()
{
   return maFact;
}

const cTplValGesInit< double > & cParamFusionSift::aFact()const 
{
   return maFact;
}


cTplValGesInit< int > & cParamFusionSift::aNbMax()
{
   return maNbMax;
}

const cTplValGesInit< int > & cParamFusionSift::aNbMax()const 
{
   return maNbMax;
}


cTplValGesInit< int > & cParamFusionSift::aNb1()
{
   return maNb1;
}

const cTplValGesInit< int > & cParamFusionSift::aNb1()const 
{
   return maNb1;
}


cTplValGesInit< int > & cParamFusionSift::aNb2()
{
   return maNb2;
}

const cTplValGesInit< int > & cParamFusionSift::aNb2()const 
{
   return maNb2;
}


cTplValGesInit< double > & cParamFusionSift::seuilCoherenceVois()
{
   return mseuilCoherenceVois;
}

const cTplValGesInit< double > & cParamFusionSift::seuilCoherenceVois()const 
{
   return mseuilCoherenceVois;
}


cTplValGesInit< double > & cParamFusionSift::seuilCoherenceCarre()
{
   return mseuilCoherenceCarre;
}

const cTplValGesInit< double > & cParamFusionSift::seuilCoherenceCarre()const 
{
   return mseuilCoherenceCarre;
}


cTplValGesInit< int > & cParamFusionSift::aNb()
{
   return maNb;
}

const cTplValGesInit< int > & cParamFusionSift::aNb()const 
{
   return maNb;
}


cTplValGesInit< int > & cParamFusionSift::nbEssais()
{
   return mnbEssais;
}

const cTplValGesInit< int > & cParamFusionSift::nbEssais()const 
{
   return mnbEssais;
}


std::list< caffichImg > & cParamFusionSift::affichImg()
{
   return maffichImg;
}

const std::list< caffichImg > & cParamFusionSift::affichImg()const 
{
   return maffichImg;
}


std::list< caffichPaire > & cParamFusionSift::affichPaire()
{
   return maffichPaire;
}

const std::list< caffichPaire > & cParamFusionSift::affichPaire()const 
{
   return maffichPaire;
}

cElXMLTree * ToXMLTree(const cParamFusionSift & anObj)
{
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ParamFusionSift",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("dossier"),anObj.dossier())->ReTagThis("dossier"));
   aRes->AddFils(::ToXMLTree(std::string("dossierImg"),anObj.dossierImg())->ReTagThis("dossierImg"));
   if (anObj.extensionSortie().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("extensionSortie"),anObj.extensionSortie().Val())->ReTagThis("extensionSortie"));
   if (anObj.firstfichier().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("firstfichier"),anObj.firstfichier().Val())->ReTagThis("firstfichier"));
   if (anObj.lastfichier().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("lastfichier"),anObj.lastfichier().Val())->ReTagThis("lastfichier"));
   if (anObj.SzMin().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SzMin"),anObj.SzMin().Val())->ReTagThis("SzMin"));
   if (anObj.NbObjMax().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NbObjMax"),anObj.NbObjMax().Val())->ReTagThis("NbObjMax"));
   if (anObj.box().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("box"),anObj.box().Val())->ReTagThis("box"));
   if (anObj.distIsol().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("distIsol"),anObj.distIsol().Val())->ReTagThis("distIsol"));
   if (anObj.ptppi().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("ptppi"),anObj.ptppi().Val())->ReTagThis("ptppi"));
   if (anObj.mindistalign().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("mindistalign"),anObj.mindistalign().Val())->ReTagThis("mindistalign"));
   if (anObj.filtre1().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("filtre1"),anObj.filtre1().Val())->ReTagThis("filtre1"));
   if (anObj.filtre2().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("filtre2"),anObj.filtre2().Val())->ReTagThis("filtre2"));
   if (anObj.filtre3().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("filtre3"),anObj.filtre3().Val())->ReTagThis("filtre3"));
   if (anObj.distIsol2().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("distIsol2"),anObj.distIsol2().Val())->ReTagThis("distIsol2"));
   if (anObj.rapide().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("rapide"),anObj.rapide().Val())->ReTagThis("rapide"));
   if (anObj.aDistInitVois().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("aDistInitVois"),anObj.aDistInitVois().Val())->ReTagThis("aDistInitVois"));
   if (anObj.aFact().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("aFact"),anObj.aFact().Val())->ReTagThis("aFact"));
   if (anObj.aNbMax().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("aNbMax"),anObj.aNbMax().Val())->ReTagThis("aNbMax"));
   if (anObj.aNb1().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("aNb1"),anObj.aNb1().Val())->ReTagThis("aNb1"));
   if (anObj.aNb2().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("aNb2"),anObj.aNb2().Val())->ReTagThis("aNb2"));
   if (anObj.seuilCoherenceVois().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("seuilCoherenceVois"),anObj.seuilCoherenceVois().Val())->ReTagThis("seuilCoherenceVois"));
   if (anObj.seuilCoherenceCarre().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("seuilCoherenceCarre"),anObj.seuilCoherenceCarre().Val())->ReTagThis("seuilCoherenceCarre"));
   if (anObj.aNb().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("aNb"),anObj.aNb().Val())->ReTagThis("aNb"));
   if (anObj.nbEssais().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("nbEssais"),anObj.nbEssais().Val())->ReTagThis("nbEssais"));
  for
  (       std::list< caffichImg >::const_iterator it=anObj.affichImg().begin();
      it !=anObj.affichImg().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("affichImg"));
  for
  (       std::list< caffichPaire >::const_iterator it=anObj.affichPaire().begin();
      it !=anObj.affichPaire().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("affichPaire"));
  return aRes;
}

void xml_init(cParamFusionSift & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;

   xml_init(anObj.dossier(),aTree->Get("dossier",1)); //tototo 

   xml_init(anObj.dossierImg(),aTree->Get("dossierImg",1)); //tototo 

   xml_init(anObj.extensionSortie(),aTree->Get("extensionSortie",1),std::string("filtre")); //tototo 

   xml_init(anObj.firstfichier(),aTree->Get("firstfichier",1),int(1)); //tototo 

   xml_init(anObj.lastfichier(),aTree->Get("lastfichier",1)); //tototo 

   xml_init(anObj.SzMin(),aTree->Get("SzMin",1),int(50)); //tototo 

   xml_init(anObj.NbObjMax(),aTree->Get("NbObjMax",1),int(10)); //tototo 

   xml_init(anObj.box(),aTree->Get("box",1),Box2dr(Pt2dr(0.0,0.0),Pt2dr(10000.0,10000.0))); //tototo 

   xml_init(anObj.distIsol(),aTree->Get("distIsol",1),REAL(300)); //tototo 

   xml_init(anObj.ptppi(),aTree->Get("ptppi",1),int(10)); //tototo 

   xml_init(anObj.mindistalign(),aTree->Get("mindistalign",1),double(4)); //tototo 

   xml_init(anObj.filtre1(),aTree->Get("filtre1",1),bool(false)); //tototo 

   xml_init(anObj.filtre2(),aTree->Get("filtre2",1),bool(true)); //tototo 

   xml_init(anObj.filtre3(),aTree->Get("filtre3",1),bool(false)); //tototo 

   xml_init(anObj.distIsol2(),aTree->Get("distIsol2",1),REAL(1000)); //tototo 

   xml_init(anObj.rapide(),aTree->Get("rapide",1),bool(false)); //tototo 

   xml_init(anObj.aDistInitVois(),aTree->Get("aDistInitVois",1),double(1)); //tototo 

   xml_init(anObj.aFact(),aTree->Get("aFact",1),double(10.0)); //tototo 

   xml_init(anObj.aNbMax(),aTree->Get("aNbMax",1),int(100)); //tototo 

   xml_init(anObj.aNb1(),aTree->Get("aNb1",1),int(10)); //tototo 

   xml_init(anObj.aNb2(),aTree->Get("aNb2",1),int(20)); //tototo 

   xml_init(anObj.seuilCoherenceVois(),aTree->Get("seuilCoherenceVois",1),double(0.9)); //tototo 

   xml_init(anObj.seuilCoherenceCarre(),aTree->Get("seuilCoherenceCarre",1),double(0.5)); //tototo 

   xml_init(anObj.aNb(),aTree->Get("aNb",1),int(50)); //tototo 

   xml_init(anObj.nbEssais(),aTree->Get("nbEssais",1),int(25)); //tototo 

   xml_init(anObj.affichImg(),aTree->GetAll("affichImg",false,1));

   xml_init(anObj.affichPaire(),aTree->GetAll("affichPaire",false,1));
}

};
