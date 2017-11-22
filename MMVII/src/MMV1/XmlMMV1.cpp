#include "StdAfx.h"
#include "include/MMVII_all.h"


namespace MMVII
{

void  MMV1InitSet(std::vector<std::string> & aRes,const std::string & aName)
{
   aRes.clear();
   cListOfName aLON = StdGetFromPCP(aName,ListOfName);
   std::copy
   (
      aLON.Name().begin(),
      aLON.Name().end(),
      std::back_inserter(aRes)
   );
}

template<> void  MMv1_SaveInFile(const cSetName & aVal,const std::string & aName)
{
    cListOfName aLON;
    for (const auto & el : aVal.Cont())
        aLON.Name().push_back(el);
    MakeFileXML(aLON,aName);
}



};
