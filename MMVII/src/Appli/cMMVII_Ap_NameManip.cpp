#include "cMMVII_Appli.h"


namespace MMVII
{

cMMVII_Ap_NameManip::cMMVII_Ap_NameManip() 
{
    cMMVII_Ap_NameManip::InitLUT();
}

cMMVII_Ap_NameManip::~cMMVII_Ap_NameManip()
{
}


const char *  cMMVII_Ap_NameManip::SkipLut(const cCarLookUpTable * aLut,const char * aC,int aVal)
{
    while (*aC && (aLut->Val(*aC)==aVal))
        aC++;
    return aC;
}

static   std::map<std::string,cCarLookUpTable> mMapLut;
void cMMVII_Ap_NameManip::InitLUT()
{
    if (! mMapLut.empty()) return;

    for (const auto & aStr : {" @",".","t","=","*",","})
    {
          mMapLut[aStr].Init(aStr,1);
    }
}

const cCarLookUpTable & Get_CarLookUpTable(const std::string& aName)
{
   MMVII_INTERNAL_ASSERT_medium(MapBoolFind(mMapLut,aName),"Cannot find CarLookUpTable for "+aName);

   return  mMapLut[aName];
}



void cMMVII_Ap_NameManip::SplitString(std::vector<std::string > & aRes,const std::string & aStr,const std::string & aSpace)
{
/*
{
    static std::set<std::string> aGlobSetSp = {" @",".","t","=","*",","};
    static std::set<std::string> aSetSpace;
    if (!MapBoolFind(aSetSpace,aSpace))
    {
       aSetSpace.insert(aSpace);
       for (const auto & aSp : aSetSpace)
           StdOut() << "[" << aSp << "]";
        StdOut() <<  "  LLLL=" << aSpace;
        StdOut() << "\n ###############################################################################\n";
        MMVII_INTERNAL_ASSERT_tiny(MapBoolFind(aGlobSetSp,aSpace),"cMMVII_Ap_NameManip::SplitString aGlobSetSp");
 
    }
}
     GetCurLut();
     mCurLut->Init(aSpace,1);
*/


     const cCarLookUpTable * aCurLut = & Get_CarLookUpTable(aSpace);

     const char * aC = aStr.c_str();
     if (aCurLut->Val(*aC)==1)
        aRes.push_back("");
     while (*aC)
     {
          aC = SkipLut(aCurLut,aC,1);
          const char * aC0 = aC;
          aC = SkipLut(aCurLut,aC,0);
          std::string aStr(aC0,aC);
          aRes.push_back(aStr);
          // aC = SkipLut(aC,1);
     }
     // const char * aDT = aLUT->Table() ;
     // RendreCurLut();
}

std::vector<std::string > cMMVII_Ap_NameManip::SplitString(const std::string & aStr,const std::string & aSpace)
{
   std::vector<std::string > aRes;
   SplitString(aRes,aStr,aSpace);
   return aRes;
}



};

