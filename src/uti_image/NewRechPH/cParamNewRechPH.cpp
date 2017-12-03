#include "StdAfx.h"
#include "cParamNewRechPH.h"
// NOMORE ...

Pt2dr & cPCarac::Pt()
{
   return mPt;
}

const Pt2dr & cPCarac::Pt()const 
{
   return mPt;
}

void  BinaryUnDumpFromFile(cPCarac & anObj,ELISE_fp & aFp)
{
     BinaryUnDumpFromFile(anObj.Pt(),aFp);
}

void  BinaryDumpInFile(ELISE_fp & aFp,const cPCarac & anObj)
{
    BinaryDumpInFile(aFp,anObj.Pt());
}

cElXMLTree * ToXMLTree(const cPCarac & anObj)
{
  XMLPushContext(anObj.mGXml);
  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,"PCarac",eXMLBranche);
   aRes->AddFils(::ToXMLTree(std::string("Pt"),anObj.Pt())->ReTagThis("Pt"));
  aRes->mGXml = anObj.mGXml;
  XMLPopContext(anObj.mGXml);
  return aRes;
}

void xml_init(cPCarac & anObj,cElXMLTree * aTree)
{
   if (aTree==0) return;
   anObj.mGXml = aTree->mGXml;

   xml_init(anObj.Pt(),aTree->Get("Pt",1)); //tototo 
}

std::string  Mangling( cPCarac *) {return "60171CA6CCD317C1F8BF";};

// };
