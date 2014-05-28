#include "StdAfx.h"
#include "addon_ParamChantierPhotogram.h"

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

void  BinaryUnDumpFromFile(caffichImg & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.image(),aFp);
    BinaryUnDumpFromFile(anObj.fichier(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const caffichImg & anObj)
{
    BinaryDumpInFile(aFp,anObj.image());
    BinaryDumpInFile(aFp,anObj.fichier());
}

cElXMLTree * ToXMLTree(const caffichImg & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"affichImg",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("image"),anObj.image())->ReTagThis("image"));
   aRes->AddFils(::ToXMLTree(std::string("fichier"),anObj.fichier())->ReTagThis("fichier"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(caffichImg & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.image(),aTree->Get("image",1)); //tototo 

   xml_init(anObj.fichier(),aTree->Get("fichier",1)); //tototo 
}

std::string  Mangling( caffichImg *) {return "C1D6F78E951F9A96FE3F";};


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

void  BinaryUnDumpFromFile(caffichPaire & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.image1(),aFp);
    BinaryUnDumpFromFile(anObj.fichier1(),aFp);
    BinaryUnDumpFromFile(anObj.image2(),aFp);
    BinaryUnDumpFromFile(anObj.fichier2(),aFp);
    BinaryUnDumpFromFile(anObj.liste(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.trait().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.trait().ValForcedForUnUmp(),aFp);
        }
        else  anObj.trait().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const caffichPaire & anObj)
{
    BinaryDumpInFile(aFp,anObj.image1());
    BinaryDumpInFile(aFp,anObj.fichier1());
    BinaryDumpInFile(aFp,anObj.image2());
    BinaryDumpInFile(aFp,anObj.fichier2());
    BinaryDumpInFile(aFp,anObj.liste());
    BinaryDumpInFile(aFp,anObj.trait().IsInit());
    if (anObj.trait().IsInit()) BinaryDumpInFile(aFp,anObj.trait().Val());
}

cElXMLTree * ToXMLTree(const caffichPaire & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"affichPaire",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("image1"),anObj.image1())->ReTagThis("image1"));
   aRes->AddFils(::ToXMLTree(std::string("fichier1"),anObj.fichier1())->ReTagThis("fichier1"));
   aRes->AddFils(::ToXMLTree(std::string("image2"),anObj.image2())->ReTagThis("image2"));
   aRes->AddFils(::ToXMLTree(std::string("fichier2"),anObj.fichier2())->ReTagThis("fichier2"));
   aRes->AddFils(::ToXMLTree(std::string("liste"),anObj.liste())->ReTagThis("liste"));
   if (anObj.trait().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("trait"),anObj.trait().Val())->ReTagThis("trait"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(caffichPaire & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.image1(),aTree->Get("image1",1)); //tototo 

   xml_init(anObj.fichier1(),aTree->Get("fichier1",1)); //tototo 

   xml_init(anObj.image2(),aTree->Get("image2",1)); //tototo 

   xml_init(anObj.fichier2(),aTree->Get("fichier2",1)); //tototo 

   xml_init(anObj.liste(),aTree->Get("liste",1)); //tototo 

   xml_init(anObj.trait(),aTree->Get("trait",1),bool(true)); //tototo 
}

std::string  Mangling( caffichPaire *) {return "853FD75B235E8FABFF3F";};


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

void  BinaryUnDumpFromFile(cParamFusionSift & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.dossier(),aFp);
    BinaryUnDumpFromFile(anObj.dossierImg(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.extensionSortie().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.extensionSortie().ValForcedForUnUmp(),aFp);
        }
        else  anObj.extensionSortie().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.firstfichier().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.firstfichier().ValForcedForUnUmp(),aFp);
        }
        else  anObj.firstfichier().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.lastfichier().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.lastfichier().ValForcedForUnUmp(),aFp);
        }
        else  anObj.lastfichier().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SzMin().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SzMin().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SzMin().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NbObjMax().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NbObjMax().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NbObjMax().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.box().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.box().ValForcedForUnUmp(),aFp);
        }
        else  anObj.box().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.distIsol().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.distIsol().ValForcedForUnUmp(),aFp);
        }
        else  anObj.distIsol().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.ptppi().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.ptppi().ValForcedForUnUmp(),aFp);
        }
        else  anObj.ptppi().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.mindistalign().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.mindistalign().ValForcedForUnUmp(),aFp);
        }
        else  anObj.mindistalign().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.filtre1().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.filtre1().ValForcedForUnUmp(),aFp);
        }
        else  anObj.filtre1().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.filtre2().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.filtre2().ValForcedForUnUmp(),aFp);
        }
        else  anObj.filtre2().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.filtre3().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.filtre3().ValForcedForUnUmp(),aFp);
        }
        else  anObj.filtre3().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.distIsol2().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.distIsol2().ValForcedForUnUmp(),aFp);
        }
        else  anObj.distIsol2().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.rapide().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.rapide().ValForcedForUnUmp(),aFp);
        }
        else  anObj.rapide().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.aDistInitVois().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.aDistInitVois().ValForcedForUnUmp(),aFp);
        }
        else  anObj.aDistInitVois().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.aFact().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.aFact().ValForcedForUnUmp(),aFp);
        }
        else  anObj.aFact().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.aNbMax().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.aNbMax().ValForcedForUnUmp(),aFp);
        }
        else  anObj.aNbMax().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.aNb1().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.aNb1().ValForcedForUnUmp(),aFp);
        }
        else  anObj.aNb1().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.aNb2().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.aNb2().ValForcedForUnUmp(),aFp);
        }
        else  anObj.aNb2().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.seuilCoherenceVois().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.seuilCoherenceVois().ValForcedForUnUmp(),aFp);
        }
        else  anObj.seuilCoherenceVois().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.seuilCoherenceCarre().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.seuilCoherenceCarre().ValForcedForUnUmp(),aFp);
        }
        else  anObj.seuilCoherenceCarre().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.aNb().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.aNb().ValForcedForUnUmp(),aFp);
        }
        else  anObj.aNb().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.nbEssais().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.nbEssais().ValForcedForUnUmp(),aFp);
        }
        else  anObj.nbEssais().SetNoInit();
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             caffichImg aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.affichImg().push_back(aVal);
        }
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             caffichPaire aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.affichPaire().push_back(aVal);
        }
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cParamFusionSift & anObj)
{
    BinaryDumpInFile(aFp,anObj.dossier());
    BinaryDumpInFile(aFp,anObj.dossierImg());
    BinaryDumpInFile(aFp,anObj.extensionSortie().IsInit());
    if (anObj.extensionSortie().IsInit()) BinaryDumpInFile(aFp,anObj.extensionSortie().Val());
    BinaryDumpInFile(aFp,anObj.firstfichier().IsInit());
    if (anObj.firstfichier().IsInit()) BinaryDumpInFile(aFp,anObj.firstfichier().Val());
    BinaryDumpInFile(aFp,anObj.lastfichier().IsInit());
    if (anObj.lastfichier().IsInit()) BinaryDumpInFile(aFp,anObj.lastfichier().Val());
    BinaryDumpInFile(aFp,anObj.SzMin().IsInit());
    if (anObj.SzMin().IsInit()) BinaryDumpInFile(aFp,anObj.SzMin().Val());
    BinaryDumpInFile(aFp,anObj.NbObjMax().IsInit());
    if (anObj.NbObjMax().IsInit()) BinaryDumpInFile(aFp,anObj.NbObjMax().Val());
    BinaryDumpInFile(aFp,anObj.box().IsInit());
    if (anObj.box().IsInit()) BinaryDumpInFile(aFp,anObj.box().Val());
    BinaryDumpInFile(aFp,anObj.distIsol().IsInit());
    if (anObj.distIsol().IsInit()) BinaryDumpInFile(aFp,anObj.distIsol().Val());
    BinaryDumpInFile(aFp,anObj.ptppi().IsInit());
    if (anObj.ptppi().IsInit()) BinaryDumpInFile(aFp,anObj.ptppi().Val());
    BinaryDumpInFile(aFp,anObj.mindistalign().IsInit());
    if (anObj.mindistalign().IsInit()) BinaryDumpInFile(aFp,anObj.mindistalign().Val());
    BinaryDumpInFile(aFp,anObj.filtre1().IsInit());
    if (anObj.filtre1().IsInit()) BinaryDumpInFile(aFp,anObj.filtre1().Val());
    BinaryDumpInFile(aFp,anObj.filtre2().IsInit());
    if (anObj.filtre2().IsInit()) BinaryDumpInFile(aFp,anObj.filtre2().Val());
    BinaryDumpInFile(aFp,anObj.filtre3().IsInit());
    if (anObj.filtre3().IsInit()) BinaryDumpInFile(aFp,anObj.filtre3().Val());
    BinaryDumpInFile(aFp,anObj.distIsol2().IsInit());
    if (anObj.distIsol2().IsInit()) BinaryDumpInFile(aFp,anObj.distIsol2().Val());
    BinaryDumpInFile(aFp,anObj.rapide().IsInit());
    if (anObj.rapide().IsInit()) BinaryDumpInFile(aFp,anObj.rapide().Val());
    BinaryDumpInFile(aFp,anObj.aDistInitVois().IsInit());
    if (anObj.aDistInitVois().IsInit()) BinaryDumpInFile(aFp,anObj.aDistInitVois().Val());
    BinaryDumpInFile(aFp,anObj.aFact().IsInit());
    if (anObj.aFact().IsInit()) BinaryDumpInFile(aFp,anObj.aFact().Val());
    BinaryDumpInFile(aFp,anObj.aNbMax().IsInit());
    if (anObj.aNbMax().IsInit()) BinaryDumpInFile(aFp,anObj.aNbMax().Val());
    BinaryDumpInFile(aFp,anObj.aNb1().IsInit());
    if (anObj.aNb1().IsInit()) BinaryDumpInFile(aFp,anObj.aNb1().Val());
    BinaryDumpInFile(aFp,anObj.aNb2().IsInit());
    if (anObj.aNb2().IsInit()) BinaryDumpInFile(aFp,anObj.aNb2().Val());
    BinaryDumpInFile(aFp,anObj.seuilCoherenceVois().IsInit());
    if (anObj.seuilCoherenceVois().IsInit()) BinaryDumpInFile(aFp,anObj.seuilCoherenceVois().Val());
    BinaryDumpInFile(aFp,anObj.seuilCoherenceCarre().IsInit());
    if (anObj.seuilCoherenceCarre().IsInit()) BinaryDumpInFile(aFp,anObj.seuilCoherenceCarre().Val());
    BinaryDumpInFile(aFp,anObj.aNb().IsInit());
    if (anObj.aNb().IsInit()) BinaryDumpInFile(aFp,anObj.aNb().Val());
    BinaryDumpInFile(aFp,anObj.nbEssais().IsInit());
    if (anObj.nbEssais().IsInit()) BinaryDumpInFile(aFp,anObj.nbEssais().Val());
    BinaryDumpInFile(aFp,(int)anObj.affichImg().size());
    for(  std::list< caffichImg >::const_iterator iT=anObj.affichImg().begin();
         iT!=anObj.affichImg().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,(int)anObj.affichPaire().size());
    for(  std::list< caffichPaire >::const_iterator iT=anObj.affichPaire().begin();
         iT!=anObj.affichPaire().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
}

cElXMLTree * ToXMLTree(const cParamFusionSift & anObj)
{
  XMLPushContext(anObj.mGXml);
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
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cParamFusionSift & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
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

std::string  Mangling( cParamFusionSift *) {return "DCD2B8192A29EEE4FC3F";};

