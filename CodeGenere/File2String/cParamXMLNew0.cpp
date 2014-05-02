#include "cParamXMLNew0.h"
// NOMORE ...

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

cElXMLTree * ToXMLTree(const cTestDump & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"TestDump",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("I"),anObj.I())->ReTagThis("I"));
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

   xml_init(anObj.Compos(),aTree->Get("Compos",1)); //tototo 
}

// };
