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


double & cSectionLoadNuage::DistSep()
{
   return mDistSep;
}

const double & cSectionLoadNuage::DistSep()const 
{
   return mDistSep;
}


double & cSectionLoadNuage::DistZone()
{
   return mDistZone;
}

const double & cSectionLoadNuage::DistZone()const 
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
    BinaryUnDumpFromFile(anObj.DistSep(),aFp);
    BinaryUnDumpFromFile(anObj.DistZone(),aFp);
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
    BinaryDumpInFile(aFp,anObj.DistSep());
    BinaryDumpInFile(aFp,anObj.DistZone());
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
   aRes->AddFils(::ToXMLTree(std::string("DistSep"),anObj.DistSep())->ReTagThis("DistSep"));
   aRes->AddFils(::ToXMLTree(std::string("DistZone"),anObj.DistZone())->ReTagThis("DistZone"));
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

   xml_init(anObj.DistSep(),aTree->Get("DistSep",1)); //tototo 

   xml_init(anObj.DistZone(),aTree->Get("DistZone",1)); //tototo 

   xml_init(anObj.SzW(),aTree->Get("SzW",1)); //tototo 
}

std::string  Mangling( cSectionLoadNuage *) {return "48F74EBAABE970FAFCBF";};


eTypeSurfaceAnalytique & cSectionEstimSurf::TypeSurf()
{
   return mTypeSurf;
}

const eTypeSurfaceAnalytique & cSectionEstimSurf::TypeSurf()const 
{
   return mTypeSurf;
}


int & cSectionEstimSurf::NbRansac()
{
   return mNbRansac;
}

const int & cSectionEstimSurf::NbRansac()const 
{
   return mNbRansac;
}

void  BinaryUnDumpFromFile(cSectionEstimSurf & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.TypeSurf(),aFp);
    BinaryUnDumpFromFile(anObj.NbRansac(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cSectionEstimSurf & anObj)
{
    BinaryDumpInFile(aFp,anObj.TypeSurf());
    BinaryDumpInFile(aFp,anObj.NbRansac());
}

cElXMLTree * ToXMLTree(const cSectionEstimSurf & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"SectionEstimSurf",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("TypeSurf"),anObj.TypeSurf())->ReTagThis("TypeSurf"));
   aRes->AddFils(::ToXMLTree(std::string("NbRansac"),anObj.NbRansac())->ReTagThis("NbRansac"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cSectionEstimSurf & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.TypeSurf(),aTree->Get("TypeSurf",1)); //tototo 

   xml_init(anObj.NbRansac(),aTree->Get("NbRansac",1)); //tototo 
}

std::string  Mangling( cSectionEstimSurf *) {return "A38EEC2A56EDFF84FF3F";};


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


double & cSectionInitModele::DistSep()
{
   return SectionLoadNuage().DistSep();
}

const double & cSectionInitModele::DistSep()const 
{
   return SectionLoadNuage().DistSep();
}


double & cSectionInitModele::DistZone()
{
   return SectionLoadNuage().DistZone();
}

const double & cSectionInitModele::DistZone()const 
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


int & cSectionInitModele::NbRansac()
{
   return SectionEstimSurf().NbRansac();
}

const int & cSectionInitModele::NbRansac()const 
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

std::string  Mangling( cSectionInitModele *) {return "855AB8BB94354FEBFE3F";};


std::list< double > & cEtapeCompensation::Sigma()
{
   return mSigma;
}

const std::list< double > & cEtapeCompensation::Sigma()const 
{
   return mSigma;
}


cTplValGesInit< int > & cEtapeCompensation::NbIter()
{
   return mNbIter;
}

const cTplValGesInit< int > & cEtapeCompensation::NbIter()const 
{
   return mNbIter;
}


cTplValGesInit< std::string > & cEtapeCompensation::Export()
{
   return mExport;
}

const cTplValGesInit< std::string > & cEtapeCompensation::Export()const 
{
   return mExport;
}

void  BinaryUnDumpFromFile(cEtapeCompensation & anObj,ELISE_fp & aFp)
{
   { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             double aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.Sigma().push_back(aVal);
        }
  } ;
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

void  BinaryDumpInFile(ELISE_fp & aFp,const cEtapeCompensation & anObj)
{
    BinaryDumpInFile(aFp,(int)anObj.Sigma().size());
    for(  std::list< double >::const_iterator iT=anObj.Sigma().begin();
         iT!=anObj.Sigma().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.NbIter().IsInit());
    if (anObj.NbIter().IsInit()) BinaryDumpInFile(aFp,anObj.NbIter().Val());
    BinaryDumpInFile(aFp,anObj.Export().IsInit());
    if (anObj.Export().IsInit()) BinaryDumpInFile(aFp,anObj.Export().Val());
}

cElXMLTree * ToXMLTree(const cEtapeCompensation & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"EtapeCompensation",eXMLBranche);
  for
  (       std::list< double >::const_iterator it=anObj.Sigma().begin();
      it !=anObj.Sigma().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("Sigma"),(*it))->ReTagThis("Sigma"));
   if (anObj.NbIter().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("NbIter"),anObj.NbIter().Val())->ReTagThis("NbIter"));
   if (anObj.Export().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("Export"),anObj.Export().Val())->ReTagThis("Export"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cEtapeCompensation & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.Sigma(),aTree->GetAll("Sigma",false,1));

   xml_init(anObj.NbIter(),aTree->Get("NbIter",1),int(1)); //tototo 

   xml_init(anObj.Export(),aTree->Get("Export",1)); //tototo 
}

std::string  Mangling( cEtapeCompensation *) {return "D217B9629EC82EE5FE3F";};


std::list< cEtapeCompensation > & cSectionCompensation::EtapeCompensation()
{
   return mEtapeCompensation;
}

const std::list< cEtapeCompensation > & cSectionCompensation::EtapeCompensation()const 
{
   return mEtapeCompensation;
}


cTplValGesInit< double > & cSectionCompensation::CoherenceOrientation()
{
   return mCoherenceOrientation;
}

const cTplValGesInit< double > & cSectionCompensation::CoherenceOrientation()const 
{
   return mCoherenceOrientation;
}

void  BinaryUnDumpFromFile(cSectionCompensation & anObj,ELISE_fp & aFp)
{
   { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cEtapeCompensation aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.EtapeCompensation().push_back(aVal);
        }
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.CoherenceOrientation().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.CoherenceOrientation().ValForcedForUnUmp(),aFp);
        }
        else  anObj.CoherenceOrientation().SetNoInit();
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cSectionCompensation & anObj)
{
    BinaryDumpInFile(aFp,(int)anObj.EtapeCompensation().size());
    for(  std::list< cEtapeCompensation >::const_iterator iT=anObj.EtapeCompensation().begin();
         iT!=anObj.EtapeCompensation().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.CoherenceOrientation().IsInit());
    if (anObj.CoherenceOrientation().IsInit()) BinaryDumpInFile(aFp,anObj.CoherenceOrientation().Val());
}

cElXMLTree * ToXMLTree(const cSectionCompensation & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"SectionCompensation",eXMLBranche);
  for
  (       std::list< cEtapeCompensation >::const_iterator it=anObj.EtapeCompensation().begin();
      it !=anObj.EtapeCompensation().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("EtapeCompensation"));
   if (anObj.CoherenceOrientation().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("CoherenceOrientation"),anObj.CoherenceOrientation().Val())->ReTagThis("CoherenceOrientation"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cSectionCompensation & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.EtapeCompensation(),aTree->GetAll("EtapeCompensation",false,1));

   xml_init(anObj.CoherenceOrientation(),aTree->Get("CoherenceOrientation",1),double(95.0)); //tototo 
}

std::string  Mangling( cSectionCompensation *) {return "FACBBF82DF691F95FD3F";};


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


std::list< cEtapeCompensation > & cParamCasa::EtapeCompensation()
{
   return SectionCompensation().EtapeCompensation();
}

const std::list< cEtapeCompensation > & cParamCasa::EtapeCompensation()const 
{
   return SectionCompensation().EtapeCompensation();
}


cTplValGesInit< double > & cParamCasa::CoherenceOrientation()
{
   return SectionCompensation().CoherenceOrientation();
}

const cTplValGesInit< double > & cParamCasa::CoherenceOrientation()const 
{
   return SectionCompensation().CoherenceOrientation();
}


cSectionCompensation & cParamCasa::SectionCompensation()
{
   return mSectionCompensation;
}

const cSectionCompensation & cParamCasa::SectionCompensation()const 
{
   return mSectionCompensation;
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
    BinaryUnDumpFromFile(anObj.SectionCompensation(),aFp);
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
    BinaryDumpInFile(aFp,anObj.SectionCompensation());
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
   aRes->AddFils(ToXMLTree(anObj.SectionCompensation())->ReTagThis("SectionCompensation"));
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

   xml_init(anObj.SectionCompensation(),aTree->Get("SectionCompensation",1)); //tototo 

   xml_init(anObj.DirectoryChantier(),aTree->Get("DirectoryChantier",1)); //tototo 
}

std::string  Mangling( cParamCasa *) {return "B2233C9892DA5DF3FD3F";};

// };
