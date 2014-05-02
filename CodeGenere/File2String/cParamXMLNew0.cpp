#include "cParamXMLNew0.h"
// NOMORE ...
eTestDump  Str2eTestDump(const std::string & aName)
{
   if (aName=="eTestDump_0")
      return eTestDump_0;
   else if (aName=="eTestDump_1")
      return eTestDump_1;
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


std::string & cTD2REF::S()
{
   return mS;
}

const std::string & cTD2REF::S()const 
{
   return mS;
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
     BinaryUnDumpFromFile(anObj.S(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cTD2REF & anObj)
{
    BinaryDumpInFile(aFp,anObj.S());
    BinaryDumpInFile(aFp,(int)anObj.V().size());
}

cElXMLTree * ToXMLTree(const cTD2REF & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"TD2REF",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("S"),anObj.S())->ReTagThis("S"));
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
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.S(),aTree->Get("S",1)); //tototo 

   xml_init(anObj.V(),aTree->GetAll("V",false,1));
}


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
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.A(),aTree->Get("A",1)); //tototo 

   xml_init(anObj.B(),aTree->Get("B",1)); //tototo 
}


int & cTestDump::I()
{
   return mI;
}

const int & cTestDump::I()const 
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
     BinaryUnDumpFromFile(anObj.I(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) BinaryDumpInFile(aFp,anObj.D().ValForcedForUnUmp());
        else  anObj.D().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.E(),aFp);
    BinaryUnDumpFromFile(anObj.R1(),aFp);
  { bool IsInit;
       BinaryUnDumpFromFile(IsInit,aFp);
        if (IsInit) BinaryDumpInFile(aFp,anObj.R2().ValForcedForUnUmp());
        else  anObj.R2().SetNoInit();
  } ;
    BinaryUnDumpFromFile(anObj.Compos(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cTestDump & anObj)
{
    BinaryDumpInFile(aFp,anObj.I());
    BinaryDumpInFile(aFp,anObj.D().IsInit());
    if (anObj.D().IsInit()) BinaryDumpInFile(aFp,anObj.D().Val());
    BinaryDumpInFile(aFp,anObj.E());
    BinaryDumpInFile(aFp,(int)anObj.V().size());
    BinaryDumpInFile(aFp,anObj.R1());
    BinaryDumpInFile(aFp,anObj.R2().IsInit());
    if (anObj.R2().IsInit()) BinaryDumpInFile(aFp,anObj.R2().Val());
    BinaryDumpInFile(aFp,anObj.Compos());
}

cElXMLTree * ToXMLTree(const cTestDump & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"TestDump",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("I"),anObj.I())->ReTagThis("I"));
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
   aRes->AddFils(ToXMLTree(anObj.Compos())->ReTagThis("Compos"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cTestDump & anObj,cElXMLTree * aTree)
{
   anObj.mGXml = aTree->mGXml;
   if (aTree==0) return;

   xml_init(anObj.I(),aTree->Get("I",1)); //tototo 

   xml_init(anObj.D(),aTree->Get("D",1)); //tototo 

   xml_init(anObj.E(),aTree->Get("E",1)); //tototo 

   xml_init(anObj.V(),aTree->GetAll("V",false,1));

   xml_init(anObj.R1(),aTree->Get("R1",1)); //tototo 

   xml_init(anObj.R2(),aTree->Get("R2",1)); //tototo 

   xml_init(anObj.Compos(),aTree->Get("Compos",1)); //tototo 
}

// };
