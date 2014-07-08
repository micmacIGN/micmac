#include "StdAfx.h"
#include "cParamCasa.h"
// NOMORE ...

cTplValGesInit< std::string > & cNuageByImage::NameMasqSup()
{
   return mNameMasqSup;
}

const cTplValGesInit< std::string > & cNuageByImage::NameMasqSup()const 
{
   return mNameMasqSup;
}


std::string & cNuageByImage::NameXMLNuage()
{
   return mNameXMLNuage;
}

const std::string & cNuageByImage::NameXMLNuage()const 
{
   return mNameXMLNuage;
}

void  BinaryUnDumpFromFile(cNuageByImage & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NameMasqSup().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NameMasqSup().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NameMasqSup().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.NameXMLNuage(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cNuageByImage & anObj)
{
    BinaryDumpInFile(aFp,anObj.NameMasqSup().IsInit());
    if (anObj.NameMasqSup().IsInit()) BinaryDumpInFile(aFp,anObj.NameMasqSup().Val());
    BinaryDumpInFile(aFp,anObj.NameXMLNuage());
}

cElXMLTree * ToXMLTree(const cNuageByImage & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"NuageByImage",eXMLBranche);
   if (anObj.NameMasqSup().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NameMasqSup"),anObj.NameMasqSup().Val())->ReTagThis("NameMasqSup"));
   aRes->AddFils(::ToXMLTree(std::string("NameXMLNuage"),anObj.NameXMLNuage())->ReTagThis("NameXMLNuage"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cNuageByImage & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.NameMasqSup(),aTree->Get("NameMasqSup",1)); //tototo 

   xml_init(anObj.NameXMLNuage(),aTree->Get("NameXMLNuage",1)); //tototo 
}

std::string  Mangling( cNuageByImage *) {return "DA722F1123E4F1ECFE3F";};


std::list< cNuageByImage > & cSectionLoadNuage::NuageByImage()
{
   return mNuageByImage;
}

const std::list< cNuageByImage > & cSectionLoadNuage::NuageByImage()const 
{
   return mNuageByImage;
}


cTplValGesInit< double > & cSectionLoadNuage::DistSep()
{
   return mDistSep;
}

const cTplValGesInit< double > & cSectionLoadNuage::DistSep()const 
{
   return mDistSep;
}


cTplValGesInit< double > & cSectionLoadNuage::DistZone()
{
   return mDistZone;
}

const cTplValGesInit< double > & cSectionLoadNuage::DistZone()const 
{
   return mDistZone;
}


cTplValGesInit< Pt2di > & cSectionLoadNuage::SzW()
{
   return mSzW;
}

const cTplValGesInit< Pt2di > & cSectionLoadNuage::SzW()const 
{
   return mSzW;
}

void  BinaryUnDumpFromFile(cSectionLoadNuage & anObj,ELISE_fp & aFp)
{
   { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cNuageByImage aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.NuageByImage().push_back(aVal);
        }
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.DistSep().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.DistSep().ValForcedForUnUmp(),aFp);
        }
        else  anObj.DistSep().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.DistZone().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.DistZone().ValForcedForUnUmp(),aFp);
        }
        else  anObj.DistZone().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.SzW().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.SzW().ValForcedForUnUmp(),aFp);
        }
        else  anObj.SzW().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cSectionLoadNuage & anObj)
{
    BinaryDumpInFile(aFp,(int)anObj.NuageByImage().size());
    for(  std::list< cNuageByImage >::const_iterator iT=anObj.NuageByImage().begin();
         iT!=anObj.NuageByImage().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.DistSep().IsInit());
    if (anObj.DistSep().IsInit()) BinaryDumpInFile(aFp,anObj.DistSep().Val());
    BinaryDumpInFile(aFp,anObj.DistZone().IsInit());
    if (anObj.DistZone().IsInit()) BinaryDumpInFile(aFp,anObj.DistZone().Val());
    BinaryDumpInFile(aFp,anObj.SzW().IsInit());
    if (anObj.SzW().IsInit()) BinaryDumpInFile(aFp,anObj.SzW().Val());
}

cElXMLTree * ToXMLTree(const cSectionLoadNuage & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"SectionLoadNuage",eXMLBranche);
  for
  (       std::list< cNuageByImage >::const_iterator it=anObj.NuageByImage().begin();
      it !=anObj.NuageByImage().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("NuageByImage"));
   if (anObj.DistSep().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DistSep"),anObj.DistSep().Val())->ReTagThis("DistSep"));
   if (anObj.DistZone().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("DistZone"),anObj.DistZone().Val())->ReTagThis("DistZone"));
   if (anObj.SzW().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("SzW"),anObj.SzW().Val())->ReTagThis("SzW"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cSectionLoadNuage & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.NuageByImage(),aTree->GetAll("NuageByImage",false,1));

   xml_init(anObj.DistSep(),aTree->Get("DistSep",1),double(50.0)); //tototo 

   xml_init(anObj.DistZone(),aTree->Get("DistZone",1),double(90.0)); //tototo 

   xml_init(anObj.SzW(),aTree->Get("SzW",1)); //tototo 
}

std::string  Mangling( cSectionLoadNuage *) {return "84C8A5EF81DF75E5FD3F";};


eTypeSurfaceAnalytique & cSectionEstimSurf::TypeSurf()
{
   return mTypeSurf;
}

const eTypeSurfaceAnalytique & cSectionEstimSurf::TypeSurf()const 
{
   return mTypeSurf;
}


cTplValGesInit< int > & cSectionEstimSurf::NbRansac()
{
   return mNbRansac;
}

const cTplValGesInit< int > & cSectionEstimSurf::NbRansac()const 
{
   return mNbRansac;
}

void  BinaryUnDumpFromFile(cSectionEstimSurf & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.TypeSurf(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NbRansac().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NbRansac().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NbRansac().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cSectionEstimSurf & anObj)
{
    BinaryDumpInFile(aFp,anObj.TypeSurf());
    BinaryDumpInFile(aFp,anObj.NbRansac().IsInit());
    if (anObj.NbRansac().IsInit()) BinaryDumpInFile(aFp,anObj.NbRansac().Val());
}

cElXMLTree * ToXMLTree(const cSectionEstimSurf & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"SectionEstimSurf",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("TypeSurf"),anObj.TypeSurf())->ReTagThis("TypeSurf"));
   if (anObj.NbRansac().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NbRansac"),anObj.NbRansac().Val())->ReTagThis("NbRansac"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cSectionEstimSurf & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.TypeSurf(),aTree->Get("TypeSurf",1)); //tototo 

   xml_init(anObj.NbRansac(),aTree->Get("NbRansac",1),int(500)); //tototo 
}

std::string  Mangling( cSectionEstimSurf *) {return "14B52C2247706AA0FC3F";};


std::string & cSectionInitModele::Name()
{
   return mName;
}

const std::string & cSectionInitModele::Name()const 
{
   return mName;
}


std::list< cNuageByImage > & cSectionInitModele::NuageByImage()
{
   return SectionLoadNuage().NuageByImage();
}

const std::list< cNuageByImage > & cSectionInitModele::NuageByImage()const 
{
   return SectionLoadNuage().NuageByImage();
}


cTplValGesInit< double > & cSectionInitModele::DistSep()
{
   return SectionLoadNuage().DistSep();
}

const cTplValGesInit< double > & cSectionInitModele::DistSep()const 
{
   return SectionLoadNuage().DistSep();
}


cTplValGesInit< double > & cSectionInitModele::DistZone()
{
   return SectionLoadNuage().DistZone();
}

const cTplValGesInit< double > & cSectionInitModele::DistZone()const 
{
   return SectionLoadNuage().DistZone();
}


cTplValGesInit< Pt2di > & cSectionInitModele::SzW()
{
   return SectionLoadNuage().SzW();
}

const cTplValGesInit< Pt2di > & cSectionInitModele::SzW()const 
{
   return SectionLoadNuage().SzW();
}


cSectionLoadNuage & cSectionInitModele::SectionLoadNuage()
{
   return mSectionLoadNuage;
}

const cSectionLoadNuage & cSectionInitModele::SectionLoadNuage()const 
{
   return mSectionLoadNuage;
}


eTypeSurfaceAnalytique & cSectionInitModele::TypeSurf()
{
   return SectionEstimSurf().TypeSurf();
}

const eTypeSurfaceAnalytique & cSectionInitModele::TypeSurf()const 
{
   return SectionEstimSurf().TypeSurf();
}


cTplValGesInit< int > & cSectionInitModele::NbRansac()
{
   return SectionEstimSurf().NbRansac();
}

const cTplValGesInit< int > & cSectionInitModele::NbRansac()const 
{
   return SectionEstimSurf().NbRansac();
}


cSectionEstimSurf & cSectionInitModele::SectionEstimSurf()
{
   return mSectionEstimSurf;
}

const cSectionEstimSurf & cSectionInitModele::SectionEstimSurf()const 
{
   return mSectionEstimSurf;
}

void  BinaryUnDumpFromFile(cSectionInitModele & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Name(),aFp);
    BinaryUnDumpFromFile(anObj.SectionLoadNuage(),aFp);
    BinaryUnDumpFromFile(anObj.SectionEstimSurf(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cSectionInitModele & anObj)
{
    BinaryDumpInFile(aFp,anObj.Name());
    BinaryDumpInFile(aFp,anObj.SectionLoadNuage());
    BinaryDumpInFile(aFp,anObj.SectionEstimSurf());
}

cElXMLTree * ToXMLTree(const cSectionInitModele & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"SectionInitModele",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Name"),anObj.Name())->ReTagThis("Name"));
   aRes->AddFils(ToXMLTree(anObj.SectionLoadNuage())->ReTagThis("SectionLoadNuage"));
   aRes->AddFils(ToXMLTree(anObj.SectionEstimSurf())->ReTagThis("SectionEstimSurf"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cSectionInitModele & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Name(),aTree->Get("Name",1)); //tototo 

   xml_init(anObj.SectionLoadNuage(),aTree->Get("SectionLoadNuage",1)); //tototo 

   xml_init(anObj.SectionEstimSurf(),aTree->Get("SectionEstimSurf",1)); //tototo 
}

std::string  Mangling( cSectionInitModele *) {return "758C5C01F0BA8AE4FE3F";};


cTplValGesInit< int > & cCasaEtapeCompensation::NbIter()
{
   return mNbIter;
}

const cTplValGesInit< int > & cCasaEtapeCompensation::NbIter()const 
{
   return mNbIter;
}


cTplValGesInit< std::string > & cCasaEtapeCompensation::Export()
{
   return mExport;
}

const cTplValGesInit< std::string > & cCasaEtapeCompensation::Export()const 
{
   return mExport;
}

void  BinaryUnDumpFromFile(cCasaEtapeCompensation & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.NbIter().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.NbIter().ValForcedForUnUmp(),aFp);
        }
        else  anObj.NbIter().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.Export().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.Export().ValForcedForUnUmp(),aFp);
        }
        else  anObj.Export().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cCasaEtapeCompensation & anObj)
{
    BinaryDumpInFile(aFp,anObj.NbIter().IsInit());
    if (anObj.NbIter().IsInit()) BinaryDumpInFile(aFp,anObj.NbIter().Val());
    BinaryDumpInFile(aFp,anObj.Export().IsInit());
    if (anObj.Export().IsInit()) BinaryDumpInFile(aFp,anObj.Export().Val());
}

cElXMLTree * ToXMLTree(const cCasaEtapeCompensation & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"CasaEtapeCompensation",eXMLBranche);
   if (anObj.NbIter().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NbIter"),anObj.NbIter().Val())->ReTagThis("NbIter"));
   if (anObj.Export().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Export"),anObj.Export().Val())->ReTagThis("Export"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cCasaEtapeCompensation & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.NbIter(),aTree->Get("NbIter",1),int(1)); //tototo 

   xml_init(anObj.Export(),aTree->Get("Export",1)); //tototo 
}

std::string  Mangling( cCasaEtapeCompensation *) {return "76350DA56B69528BFD3F";};


std::list< cCasaEtapeCompensation > & cCasaSectionCompensation::CasaEtapeCompensation()
{
   return mCasaEtapeCompensation;
}

const std::list< cCasaEtapeCompensation > & cCasaSectionCompensation::CasaEtapeCompensation()const 
{
   return mCasaEtapeCompensation;
}


cTplValGesInit< double > & cCasaSectionCompensation::PercCoherenceOrientation()
{
   return mPercCoherenceOrientation;
}

const cTplValGesInit< double > & cCasaSectionCompensation::PercCoherenceOrientation()const 
{
   return mPercCoherenceOrientation;
}

void  BinaryUnDumpFromFile(cCasaSectionCompensation & anObj,ELISE_fp & aFp)
{
   { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cCasaEtapeCompensation aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.CasaEtapeCompensation().push_back(aVal);
        }
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.PercCoherenceOrientation().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.PercCoherenceOrientation().ValForcedForUnUmp(),aFp);
        }
        else  anObj.PercCoherenceOrientation().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cCasaSectionCompensation & anObj)
{
    BinaryDumpInFile(aFp,(int)anObj.CasaEtapeCompensation().size());
    for(  std::list< cCasaEtapeCompensation >::const_iterator iT=anObj.CasaEtapeCompensation().begin();
         iT!=anObj.CasaEtapeCompensation().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.PercCoherenceOrientation().IsInit());
    if (anObj.PercCoherenceOrientation().IsInit()) BinaryDumpInFile(aFp,anObj.PercCoherenceOrientation().Val());
}

cElXMLTree * ToXMLTree(const cCasaSectionCompensation & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"CasaSectionCompensation",eXMLBranche);
  for
  (       std::list< cCasaEtapeCompensation >::const_iterator it=anObj.CasaEtapeCompensation().begin();
      it !=anObj.CasaEtapeCompensation().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("CasaEtapeCompensation"));
   if (anObj.PercCoherenceOrientation().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("PercCoherenceOrientation"),anObj.PercCoherenceOrientation().Val())->ReTagThis("PercCoherenceOrientation"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cCasaSectionCompensation & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.CasaEtapeCompensation(),aTree->GetAll("CasaEtapeCompensation",false,1));

   xml_init(anObj.PercCoherenceOrientation(),aTree->Get("PercCoherenceOrientation",1),double(95.0)); //tototo 
}

std::string  Mangling( cCasaSectionCompensation *) {return "84B5641FF908D9A3FC3F";};


cTplValGesInit< cChantierDescripteur > & cParamCasa::DicoLoc()
{
   return mDicoLoc;
}

const cTplValGesInit< cChantierDescripteur > & cParamCasa::DicoLoc()const 
{
   return mDicoLoc;
}


std::list< cSectionInitModele > & cParamCasa::SectionInitModele()
{
   return mSectionInitModele;
}

const std::list< cSectionInitModele > & cParamCasa::SectionInitModele()const 
{
   return mSectionInitModele;
}


std::list< cCasaEtapeCompensation > & cParamCasa::CasaEtapeCompensation()
{
   return CasaSectionCompensation().CasaEtapeCompensation();
}

const std::list< cCasaEtapeCompensation > & cParamCasa::CasaEtapeCompensation()const 
{
   return CasaSectionCompensation().CasaEtapeCompensation();
}


cTplValGesInit< double > & cParamCasa::PercCoherenceOrientation()
{
   return CasaSectionCompensation().PercCoherenceOrientation();
}

const cTplValGesInit< double > & cParamCasa::PercCoherenceOrientation()const 
{
   return CasaSectionCompensation().PercCoherenceOrientation();
}


cCasaSectionCompensation & cParamCasa::CasaSectionCompensation()
{
   return mCasaSectionCompensation;
}

const cCasaSectionCompensation & cParamCasa::CasaSectionCompensation()const 
{
   return mCasaSectionCompensation;
}


std::string & cParamCasa::DirectoryChantier()
{
   return mDirectoryChantier;
}

const std::string & cParamCasa::DirectoryChantier()const 
{
   return mDirectoryChantier;
}

void  BinaryUnDumpFromFile(cParamCasa & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.DicoLoc().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.DicoLoc().ValForcedForUnUmp(),aFp);
        }
        else  anObj.DicoLoc().SetNoInit();
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cSectionInitModele aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.SectionInitModele().push_back(aVal);
        }
  } ;
    BinaryUnDumpFromFile(anObj.CasaSectionCompensation(),aFp);
    BinaryUnDumpFromFile(anObj.DirectoryChantier(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cParamCasa & anObj)
{
    BinaryDumpInFile(aFp,anObj.DicoLoc().IsInit());
    if (anObj.DicoLoc().IsInit()) BinaryDumpInFile(aFp,anObj.DicoLoc().Val());
    BinaryDumpInFile(aFp,(int)anObj.SectionInitModele().size());
    for(  std::list< cSectionInitModele >::const_iterator iT=anObj.SectionInitModele().begin();
         iT!=anObj.SectionInitModele().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.CasaSectionCompensation());
    BinaryDumpInFile(aFp,anObj.DirectoryChantier());
}

cElXMLTree * ToXMLTree(const cParamCasa & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"ParamCasa",eXMLBranche);
   if (anObj.DicoLoc().IsInit())
      aRes->AddFils(ToXMLTree(anObj.DicoLoc().Val())->ReTagThis("DicoLoc"));
  for
  (       std::list< cSectionInitModele >::const_iterator it=anObj.SectionInitModele().begin();
      it !=anObj.SectionInitModele().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("SectionInitModele"));
   aRes->AddFils(ToXMLTree(anObj.CasaSectionCompensation())->ReTagThis("CasaSectionCompensation"));
   aRes->AddFils(::ToXMLTree(std::string("DirectoryChantier"),anObj.DirectoryChantier())->ReTagThis("DirectoryChantier"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cParamCasa & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.DicoLoc(),aTree->Get("DicoLoc",1)); //tototo 

   xml_init(anObj.SectionInitModele(),aTree->GetAll("SectionInitModele",false,1));

   xml_init(anObj.CasaSectionCompensation(),aTree->Get("CasaSectionCompensation",1)); //tototo 

   xml_init(anObj.DirectoryChantier(),aTree->Get("DirectoryChantier",1)); //tototo 
}

std::string  Mangling( cParamCasa *) {return "D7611FDA0F397EFCFDBF";};

// };
