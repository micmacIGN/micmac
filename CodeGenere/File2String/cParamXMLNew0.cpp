#include "cParamXMLNew0.h"
// NOMORE ...
eTestDump  Str2eTestDump(const std::string & aName)
{
   if (aName=="eTestDump_0")
      return eTestDump_0;
   else if (aName=="eTestDump_1")
      return eTestDump_1;
   else if (aName=="eTestDump_3")
      return eTestDump_3;
  else
  {
      cout << aName << " is not a correct value for enum eTestDump\n" ;
      ELISE_ASSERT(false,"XML enum value error");
  }
  return (eTestDump) 0;
}
void xml_init(eTestDump & aVal,cElXMLTree * aTree)
{
   aVal= Str2eTestDump(aTree->Contenu());
}
std::string  eToString(const eTestDump & anObj)
{
   if (anObj==eTestDump_0)
      return  "eTestDump_0";
   if (anObj==eTestDump_1)
      return  "eTestDump_1";
   if (anObj==eTestDump_3)
      return  "eTestDump_3";
 std::cout << "Enum = eTestDump\n";
   ELISE_ASSERT(false,"Bad Value in eToString for enum value ");
   return "";
}

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eTestDump & anObj)
{
      return  cElXMLTree::ValueNode(aNameTag,eToString(anObj));
}

void  BinaryDumpInFile(ELISE_fp & aFp,const eTestDump & anObj)
{
   BinaryDumpInFile(aFp,int(anObj));
}

void  BinaryUnDumpFromFile(eTestDump & anObj,ELISE_fp & aFp)
{
   int aIVal;
   BinaryUnDumpFromFile(aIVal,aFp);
   anObj=(eTestDump) aIVal;
}

std::string  Mangling( eTestDump *) {return "767F80144228FCC6FD3F";};


std::string & cTD2REF::K()
{
   return mK;
}

const std::string & cTD2REF::K()const 
{
   return mK;
}


std::list< int > & cTD2REF::V()
{
   return mV;
}

const std::list< int > & cTD2REF::V()const 
{
   return mV;
}

void  BinaryUnDumpFromFile(cTD2REF & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.K(),aFp);
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             int aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.V().push_back(aVal);
        }
  } ;
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cTD2REF & anObj)
{
    BinaryDumpInFile(aFp,anObj.K());
    BinaryDumpInFile(aFp,(int)anObj.V().size());
    for(  std::list< int >::const_iterator iT=anObj.V().begin();
         iT!=anObj.V().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
}

cElXMLTree * ToXMLTree(const cTD2REF & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"TD2REF",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("K"),anObj.K())->ReTagThis("K"));
  for
  (       std::list< int >::const_iterator it=anObj.V().begin();
      it !=anObj.V().end();
      it++
  ) 
      aRes->AddFils(::ToXMLTree(std::string("V"),(*it))->ReTagThis("V"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cTD2REF & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.K(),aTree->Get("K",1)); //tototo 

   xml_init(anObj.V(),aTree->GetAll("V",false,1));
}

std::string  Mangling( cTD2REF *) {return "7EFC8AD6E59EB7D4FE3F";};


double & cCompos::A()
{
   return mA;
}

const double & cCompos::A()const 
{
   return mA;
}


Pt2dr & cCompos::B()
{
   return mB;
}

const Pt2dr & cCompos::B()const 
{
   return mB;
}

void  BinaryUnDumpFromFile(cCompos & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.A(),aFp);
    BinaryUnDumpFromFile(anObj.B(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cCompos & anObj)
{
    BinaryDumpInFile(aFp,anObj.A());
    BinaryDumpInFile(aFp,anObj.B());
}

cElXMLTree * ToXMLTree(const cCompos & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"Compos",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("A"),anObj.A())->ReTagThis("A"));
   aRes->AddFils(::ToXMLTree(std::string("B"),anObj.B())->ReTagThis("B"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cCompos & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.A(),aTree->Get("A",1)); //tototo 

   xml_init(anObj.B(),aTree->Get("B",1)); //tototo 
}

std::string  Mangling( cCompos *) {return "D086D69AE4A1D684FF3F";};


cTplValGesInit< int > & cTestDump::I()
{
   return mI;
}

const cTplValGesInit< int > & cTestDump::I()const 
{
   return mI;
}


cTplValGesInit< Pt2dr > & cTestDump::D()
{
   return mD;
}

const cTplValGesInit< Pt2dr > & cTestDump::D()const 
{
   return mD;
}


eTestDump & cTestDump::E()
{
   return mE;
}

const eTestDump & cTestDump::E()const 
{
   return mE;
}


std::list< eTestDump > & cTestDump::V()
{
   return mV;
}

const std::list< eTestDump > & cTestDump::V()const 
{
   return mV;
}


cTD2REF & cTestDump::R1()
{
   return mR1;
}

const cTD2REF & cTestDump::R1()const 
{
   return mR1;
}


cTplValGesInit< cTD2REF > & cTestDump::R2()
{
   return mR2;
}

const cTplValGesInit< cTD2REF > & cTestDump::R2()const 
{
   return mR2;
}


std::list< cTD2REF > & cTestDump::R3()
{
   return mR3;
}

const std::list< cTD2REF > & cTestDump::R3()const 
{
   return mR3;
}


std::vector< cTD2REF > & cTestDump::R4()
{
   return mR4;
}

const std::vector< cTD2REF > & cTestDump::R4()const 
{
   return mR4;
}


double & cTestDump::A()
{
   return Compos().A();
}

const double & cTestDump::A()const 
{
   return Compos().A();
}


Pt2dr & cTestDump::B()
{
   return Compos().B();
}

const Pt2dr & cTestDump::B()const 
{
   return Compos().B();
}


cCompos & cTestDump::Compos()
{
   return mCompos;
}

const cCompos & cTestDump::Compos()const 
{
   return mCompos;
}

void  BinaryUnDumpFromFile(cTestDump & anObj,ELISE_fp & aFp)
{
   { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.I().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.I().ValForcedForUnUmp(),aFp);
        }
        else  anObj.I().SetNoInit();
  } ;
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.D().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.D().ValForcedForUnUmp(),aFp);
        }
        else  anObj.D().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.E(),aFp);
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             eTestDump aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.V().push_back(aVal);
        }
  } ;
    BinaryUnDumpFromFile(anObj.R1(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) {
             anObj.R2().SetInitForUnUmp();
             BinaryUnDumpFromFile(anObj.R2().ValForcedForUnUmp(),aFp);
        }
        else  anObj.R2().SetNoInit();
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cTD2REF aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.R3().push_back(aVal);
        }
  } ;
  { int aNb;
    BinaryUnDumpFromFile(aNb,aFp);
        for(  int aK=0 ; aK<aNb ; aK++)
        {
             cTD2REF aVal;
              BinaryUnDumpFromFile(aVal,aFp);
              anObj.R4().push_back(aVal);
        }
  } ;
    BinaryUnDumpFromFile(anObj.Compos(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cTestDump & anObj)
{
    BinaryDumpInFile(aFp,anObj.I().IsInit());
    if (anObj.I().IsInit()) BinaryDumpInFile(aFp,anObj.I().Val());
    BinaryDumpInFile(aFp,anObj.D().IsInit());
    if (anObj.D().IsInit()) BinaryDumpInFile(aFp,anObj.D().Val());
    BinaryDumpInFile(aFp,anObj.E());
    BinaryDumpInFile(aFp,(int)anObj.V().size());
    for(  std::list< eTestDump >::const_iterator iT=anObj.V().begin();
         iT!=anObj.V().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.R1());
    BinaryDumpInFile(aFp,anObj.R2().IsInit());
    if (anObj.R2().IsInit()) BinaryDumpInFile(aFp,anObj.R2().Val());
    BinaryDumpInFile(aFp,(int)anObj.R3().size());
    for(  std::list< cTD2REF >::const_iterator iT=anObj.R3().begin();
         iT!=anObj.R3().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,(int)anObj.R4().size());
    for(  std::vector< cTD2REF >::const_iterator iT=anObj.R4().begin();
         iT!=anObj.R4().end();
          iT++
    )
        BinaryDumpInFile(aFp,*iT);
    BinaryDumpInFile(aFp,anObj.Compos());
}

cElXMLTree * ToXMLTree(const cTestDump & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"TestDump",eXMLBranche);
   if (anObj.I().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("I"),anObj.I().Val())->ReTagThis("I"));
   if (anObj.D().IsInit())
      aRes->AddFils(::ToXMLTree(std::string("D"),anObj.D().Val())->ReTagThis("D"));
   aRes->AddFils(ToXMLTree(std::string("E"),anObj.E())->ReTagThis("E"));
  for
  (       std::list< eTestDump >::const_iterator it=anObj.V().begin();
      it !=anObj.V().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree(std::string("V"),(*it))->ReTagThis("V"));
   aRes->AddFils(ToXMLTree(anObj.R1())->ReTagThis("R1"));
   if (anObj.R2().IsInit())
      aRes->AddFils(ToXMLTree(anObj.R2().Val())->ReTagThis("R2"));
  for
  (       std::list< cTD2REF >::const_iterator it=anObj.R3().begin();
      it !=anObj.R3().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("R3"));
  for
  (       std::vector< cTD2REF >::const_iterator it=anObj.R4().begin();
      it !=anObj.R4().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it))->ReTagThis("R4"));
   aRes->AddFils(ToXMLTree(anObj.Compos())->ReTagThis("Compos"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cTestDump & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.I(),aTree->Get("I",1),int(128)); //tototo 

   xml_init(anObj.D(),aTree->Get("D",1)); //tototo 

   xml_init(anObj.E(),aTree->Get("E",1)); //tototo 

   xml_init(anObj.V(),aTree->GetAll("V",false,1));

   xml_init(anObj.R1(),aTree->Get("R1",1)); //tototo 

   xml_init(anObj.R2(),aTree->Get("R2",1)); //tototo 

   xml_init(anObj.R3(),aTree->GetAll("R3",false,1));

   xml_init(anObj.R4(),aTree->GetAll("R4",false,1));

   xml_init(anObj.Compos(),aTree->Get("Compos",1)); //tototo 
}

std::string  Mangling( cTestDump *) {return "128BFF1C5EFFC3DFFD3F";};


std::string & cR5::IdImage()
{
   return mIdImage;
}

const std::string & cR5::IdImage()const 
{
   return mIdImage;
}

void  BinaryUnDumpFromFile(cR5 & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.IdImage(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cR5 & anObj)
{
    BinaryDumpInFile(aFp,anObj.IdImage());
}

cElXMLTree * ToXMLTree(const cR5 & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"R5",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("IdImage"),anObj.IdImage())->ReTagThis("IdImage"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cR5 & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.IdImage(),aTree->Get("IdImage",1)); //tototo 
}

std::string  Mangling( cR5 *) {return "F7EBE053FF11F1E2FD3F";};


std::map< std::string,cR5 > & cTestNoDump::R5()
{
   return mR5;
}

const std::map< std::string,cR5 > & cTestNoDump::R5()const 
{
   return mR5;
}


int & cTestNoDump::AA()
{
   return mAA;
}

const int & cTestNoDump::AA()const 
{
   return mAA;
}


std::vector<int> & cTestNoDump::vvAA()
{
   return mvvAA;
}

const std::vector<int> & cTestNoDump::vvAA()const 
{
   return mvvAA;
}

void  BinaryUnDumpFromFile(cTestNoDump & anObj,ELISE_fp & aFp)
{
     ELISE_ASSERT(false,"No Support for this conainer in bin dump");
    BinaryUnDumpFromFile(anObj.AA(),aFp);
    BinaryUnDumpFromFile(anObj.vvAA(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cTestNoDump & anObj)
{
    ELISE_ASSERT(false,"No Support for this conainer in bin dump");
    BinaryDumpInFile(aFp,anObj.AA());
    BinaryDumpInFile(aFp,anObj.vvAA());
}

cElXMLTree * ToXMLTree(const cTestNoDump & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"TestNoDump",eXMLBranche);
  for
  (       std::map< std::string,cR5 >::const_iterator it=anObj.R5().begin();
      it !=anObj.R5().end();
      it++
  ) 
      aRes->AddFils(ToXMLTree((*it).second)->ReTagThis("R5"));
   aRes->AddFils(::ToXMLTree(std::string("AA"),anObj.AA())->ReTagThis("AA"));
   aRes->AddFils(::ToXMLTree(std::string("vvAA"),anObj.vvAA())->ReTagThis("vvAA"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cTestNoDump & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.R5(),aTree->GetAll("R5",false,1),"IdImage");

   xml_init(anObj.AA(),aTree->Get("AA",1)); //tototo 

   xml_init(anObj.vvAA(),aTree->Get("vvAA",1)); //tototo 
}

std::string  Mangling( cTestNoDump *) {return "A6658EA494FF9EE8FE3F";};

// };
