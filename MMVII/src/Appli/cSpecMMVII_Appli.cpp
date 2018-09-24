#include "include/MMVII_all.h"
#include <algorithm>


namespace MMVII
{


cSpecMMVII_Appli::cSpecMMVII_Appli
(
    const std::string & aName,
    tMMVII_AppliAllocator anAlloc,
    const std::string & aComment,
    const tVaF     & aFeatures,
    const tVaDT    & aInputs,   
    const tVaDT    & aOutputs  ,
    const std::string & aNameFile

) :
  mName       (aName),
  mAlloc      (anAlloc),
  mComment    (aComment),
  mVFeatures  (aFeatures),
  mVInputs    (aInputs),
  mVOutputs   (aOutputs),
  mNameFile   (aNameFile)
{
}

const std::string &     cSpecMMVII_Appli::Name() const {return mName;}
tMMVII_AppliAllocator  cSpecMMVII_Appli::Alloc() const {return mAlloc;}
const std::string &     cSpecMMVII_Appli::Comment() const {return mComment;}
const std::string &     cSpecMMVII_Appli::NameFile() const {return mNameFile;}


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
    MMVII_INTERNAL_ASSERT_always(!mVFeatures.empty(),"cSpecMMVII_Appli No Features");
    MMVII_INTERNAL_ASSERT_always(!mVInputs.empty(),"cSpecMMVII_Appli No Inputs");
    MMVII_INTERNAL_ASSERT_always(!mVOutputs.empty(),"cSpecMMVII_Appli No Outputs");
}

const std::vector<cSpecMMVII_Appli *> & cSpecMMVII_Appli::VecAll()
{
   static std::vector<cSpecMMVII_Appli*>  TheRes;
   
   if (TheRes.size() == 0)
   {    
        TheRes.push_back(&TheSpecBench);
        TheRes.push_back(&TheSpecTestCpp11);
        TheRes.push_back(&TheSpec_TestBoostSerial);
        TheRes.push_back(&TheSpecMPDTest);
        TheRes.push_back(&TheSpecEditSet);
        TheRes.push_back(&TheSpecEditRel);
        TheRes.push_back(&TheSpecWalkman);
        TheRes.push_back(&TheSpecDaisy);
   }
   
   return TheRes;
}

cSpecMMVII_Appli*  cSpecMMVII_Appli::SpecOfName(const std::string & aNameCom,bool SVP)
{
    
   for (const auto & aSpec : VecAll())
   {
      if (UCaseEqual(aSpec->Name(),aNameCom))
         return aSpec;
   }
   if (! SVP)
   {
      MMVII_INTERNAL_ASSERT_always(false,"Cannot find command of name ["+ aNameCom + "]");
   }

   return 0;
}


};

