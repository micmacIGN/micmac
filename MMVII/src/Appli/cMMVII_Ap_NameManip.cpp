#include "cMMVII_Appli.h"


namespace MMVII
{

cMMVII_Ap_NameManip::cMMVII_Ap_NameManip() :
   mCurLut (0)
{
}

cMMVII_Ap_NameManip::~cMMVII_Ap_NameManip()
{
     MMVII_INTERNAL_ASSERT_medium(mCurLut==0,"~cMMVII_Ap_NameManip => mCurLut");
}

void cMMVII_Ap_NameManip::GetCurLut()
{
     MMVII_INTERNAL_ASSERT_medium(mCurLut==0,"cMMVII_Ap_NameManip::GetCurLut");
     mCurLut = mGoClut.EmprunterOne();
}

void cMMVII_Ap_NameManip::RendreCurLut()
{
     MMVII_INTERNAL_ASSERT_medium(mCurLut!=0,"cMMVII_Ap_NameManip::RendreCurLut");
     mCurLut->UnInit();
     mGoClut.RendreOne(mCurLut);
     mCurLut = 0;
}


const char *  cMMVII_Ap_NameManip::SkipLut(const char * aC,int aVal)
{
    while (*aC && (mCurLut->Val(*aC)==aVal))
        aC++;
    return aC;
}


void cMMVII_Ap_NameManip::SplitString(std::vector<std::string > & aRes,const std::string & aStr,const std::string & aSpace)
{
     GetCurLut();
     mCurLut->Init(aSpace,1);

     const char * aC = aStr.c_str();
     if (mCurLut->Val(*aC)==1)
        aRes.push_back("");
     while (*aC)
     {
          aC = SkipLut(aC,1);
          const char * aC0 = aC;
          aC = SkipLut(aC,0);
          std::string aStr(aC0,aC);
          aRes.push_back(aStr);
          // aC = SkipLut(aC,1);
     }
     // const char * aDT = aLUT->Table() ;
     RendreCurLut();
}

std::vector<std::string > cMMVII_Ap_NameManip::SplitString(const std::string & aStr,const std::string & aSpace)
{
   std::vector<std::string > aRes;
   SplitString(aRes,aStr,aSpace);
   return aRes;
}



};

