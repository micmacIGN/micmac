#include "include/MMVII_all.h"
#include <algorithm>

namespace MMVII
{

/// Initialize memory for random
void OpenRandom();
/// Free memory allocated for random generation, declared here, only main global appli
/// must use it
void CloseRandom();




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

int cSpecMMVII_Appli::AllocExecuteDestruct(const std::vector<std::string> & aVArgs) const
{
   // A conserver, on le mettra dans les sauvegarde
   if (0)
   {
      StdOut() << "===Com==[";
      for (int aK=0; aK<int(aVArgs.size()) ; aK++)
      {
          if (aK!=0) StdOut() << " ";
          StdOut() << aVArgs[aK];
      }
      StdOut() << "]\n";
   }
   static int aCptCallIntern=0;
   aCptCallIntern++;
   int aNbObjLive = cMemCheck::NbObjLive();
   // Add this one to check  destruction with unique_ptr
   const cMemState  aMemoState= cMemManager::CurState() ;
   int aRes=-1;

   /* Note on Random : as random allocation is global, it had some side effect in memory checking
      because its not treated as local scope variable. To ger rid of this annoying problem,
      I force random creation and deletion at global scope */
   {
        // Use allocator
        tMMVII_UnikPApli anAppli = Alloc()(aVArgs,*this);
        //  Force random creation, just after allocation, because it may need 
        if (aCptCallIntern==1)
        {
           OpenRandom();
        }

        // Execute
        anAppli->InitParam();
        if (anAppli->ModeHelp())
           aRes = EXIT_SUCCESS;
        else
           aRes = anAppli->Exe();
        // A top level free random creation, before appli is killed
        if (aCptCallIntern==1)
        {
            CloseRandom();
        }
    }
    cMemManager::CheckRestoration(aMemoState);
    MMVII_INTERNAL_ASSERT_always(cMemCheck::NbObjLive()==aNbObjLive,"Mem check obj not killed");
    aCptCallIntern--;
    // This was the initial test, stricter, maintain it when call by main
    if (aCptCallIntern==0)
    {
         MMVII_INTERNAL_ASSERT_always(cMemCheck::NbObjLive()==0,"Mem check obj not killed");
    }
    return aRes;
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
    cMMVII_Appli & anAppli = cMMVII_Appli::CurrentAppli();

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


std::vector<cSpecMMVII_Appli*> cSpecMMVII_Appli::TheVecAll;

bool CmpCmd(cSpecMMVII_Appli * aCM1,cSpecMMVII_Appli * aCM2)
{
   return aCM1->Name() < aCM2->Name();
}

std::vector<cSpecMMVII_Appli *> & cSpecMMVII_Appli::InternVecAll()
{
   if (TheVecAll.size() == 0)
   {    
        TheVecAll.push_back(&TheSpecBench);
        TheVecAll.push_back(&TheSpecTestCpp11);
        TheVecAll.push_back(&TheSpecMPDTest);
        TheVecAll.push_back(&TheSpecEditSet);
        TheVecAll.push_back(&TheSpecEditRel);
        TheVecAll.push_back(&TheSpecWalkman);
        TheVecAll.push_back(&TheSpecDaisy);
        TheVecAll.push_back(&TheSpecCatVideo);
        TheVecAll.push_back(&TheSpec_TestEigen);
        TheVecAll.push_back(&TheSpec_ComputeParamIndexBinaire);
        TheVecAll.push_back(&TheSpecTestRecall);
        TheVecAll.push_back(&TheSpecScaleImage);
        TheVecAll.push_back(&TheSpecCalcDiscIm);
        TheVecAll.push_back(&TheSpecCalcDescPCar);
        TheVecAll.push_back(&TheSpecMatchTieP);
        TheVecAll.push_back(&TheSpecEpipGenDenseMatch);
        TheVecAll.push_back(&TheSpecGenSymbDer);
        TheVecAll.push_back(&TheSpecKapture);
        TheVecAll.push_back(&TheSpecFormatTDEDM_WT);
        TheVecAll.push_back(&TheSpecFormatTDEDM_MDLB);
        TheVecAll.push_back(&TheSpecExtractLearnVecDM);
        TheVecAll.push_back(&TheSpecCalcHistoCarac);
        TheVecAll.push_back(&TheSpecCalcHistoNDim);
        TheVecAll.push_back(&TheSpecTestHypStep);
        TheVecAll.push_back(&TheSpecFillCubeCost);
        TheVecAll.push_back(&TheSpecDMEvalRef);
        TheVecAll.push_back(&TheSpecGenCodedTarget);

        std::sort(TheVecAll.begin(),TheVecAll.end(),CmpCmd);
   }
   
   return TheVecAll;
}

const std::vector<cSpecMMVII_Appli *> & cSpecMMVII_Appli::VecAll()
{
    return InternVecAll();
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

