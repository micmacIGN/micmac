#include "include/MMVII_all.h"
#include <algorithm>


namespace MMVII
{


cSpecMMVII_Appli::cSpecMMVII_Appli
(
    const std::string & aName,
    tMMVII_AppliAllocator anAlloc,
    const std::string & aComment,
    const std::string & aFeatures,
    const std::string & aInputs,
    const std::string & aOutputs

) :
  mName      (aName),
  mAlloc     (anAlloc),
  mComment   (aComment),
  mFeatures  (aFeatures),
  mInputs    (aInputs),
  mOutputs   (aOutputs)
{
}

const std::string &     cSpecMMVII_Appli::Name() const {return mName;}
tMMVII_AppliAllocator  cSpecMMVII_Appli::Alloc() const {return mAlloc;}
const std::string &     cSpecMMVII_Appli::Comment() const {return mComment;}


// Si aMes=="SVP"=> No Error
bool  CheckIntersect
      (
            const std::string & aMes,
            const std::string & aKeyList,
            const std::string & aList,
            const std::string & aSpace
      )
{
    cMMVII_Appli & anAppli = cMMVII_Appli::TheAppli();

    std::vector<std::string>  aVKey  =  anAppli.SplitString(aKeyList,aSpace);
    std::vector<std::string>  aVTest =  anAppli.SplitString(aList,aSpace);

    for (auto itT=aVTest.begin(); itT!=aVTest.end() ; itT++)
    {
         auto itGet = std::find(aVKey.begin(), aVKey.end(),*itT);
         if (itGet== aVKey.end())
         {
             if (aMes=="SVP") return false;
             // std::string aFullMes = "Cannot find " + *itT  + " in context " +aMes;
             MMVII_INTERNAL_ASSERT_always(false,"Cannot find "+*itT+" in context "+aMes);
         }
    }

    return true;
}


void cSpecMMVII_Appli::Check()
{
    CheckIntersect("Feature of " + mName,SPECMMVII_Feature ,mFeatures," ");
    CheckIntersect("Input of "   + mName,SPECMMVII_DateType,mInputs  ," ");
    CheckIntersect("Output of "  + mName,SPECMMVII_DateType,mOutputs ," ");
    //std::vector<std::string>  aVFeature = anAppli.SPECMMVII_Feature(mFeaturesI
}

extern cSpecMMVII_Appli  TheSpecBench;
extern cSpecMMVII_Appli  TheSpecTestCpp11;
extern cSpecMMVII_Appli  TheSpec_TestBoostSerial;
extern cSpecMMVII_Appli  TheSpec_TestSerial;
  
std::vector<cSpecMMVII_Appli *> & cSpecMMVII_Appli::VecAll()
{
   static std::vector<cSpecMMVII_Appli*>  TheRes;
   
   if (TheRes.size() == 0)
   {    
        TheRes.push_back(&TheSpecBench);
        TheRes.push_back(&TheSpecTestCpp11);
        TheRes.push_back(&TheSpec_TestBoostSerial);
        TheRes.push_back(&TheSpec_TestSerial);
   }
   
   return TheRes;
}


};

