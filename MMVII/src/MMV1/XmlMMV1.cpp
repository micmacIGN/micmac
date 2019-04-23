#include "include/V1VII.h"


namespace MMVII
{

//=============  tNameRel ====================

tNameRel  MMV1InitRel(const std::string & aName)
{
   tNameRel aRes;
   cSauvegardeNamedRel aSNR = StdGetFromPCP(aName,SauvegardeNamedRel);
   for (const auto & aCpl : aSNR.Cple())
   {
       aRes.Add(tNamePair(aCpl.N1(),aCpl.N2()));
   }

   return aRes;
}

/// Write a rel in MMV1 format
template<> void  MMv1_SaveInFile(const tNameRel & aSet,const std::string & aName)
{
   std::vector<const tNamePair *> aV;
   aSet.PutInVect(aV,true);

   cSauvegardeNamedRel aSNR;
   for (const auto & aPair : aV)
   {
      aSNR.Cple().push_back(cCpleString(aPair->V1(),aPair->V2()));
   }
   MakeFileXML(aSNR,aName);
}

//=============  tNameSet ====================

/// Read MMV1 Set
tNameSet  MMV1InitSet(const std::string & aName)
{
   tNameSet aRes ;
   cListOfName aLON = StdGetFromPCP(aName,ListOfName);
   for (const auto & el : aLON.Name())
       aRes.Add(el);
   return aRes;
}

/// Write a set in MMV1 format
template<> void  MMv1_SaveInFile(const tNameSet & aSet,const std::string & aName)
{
    std::vector<const std::string *> aV;
    aSet.PutInVect(aV,true);

    cListOfName aLON;
    for (const auto & el : aV)
        aLON.Name().push_back(*el);
    MakeFileXML(aLON,aName);
}



};
