#include "MMVII_DeclareAllCmd.h"
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
#if 0
   {
      StdOut() << "===Com==[";
      for (int aK=0; aK<int(aVArgs.size()) ; aK++)
      {
          if (aK!=0) StdOut() << " ";
          StdOut() << aVArgs[aK];
      }
      StdOut() << "]" << std::endl;
   }
#endif
   static int aCptCallIntern=0;
   aCptCallIntern++;
   if (TheCmdArgs.size() == 0)
       TheCmdArgs = aVArgs;

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

        // Execute
        anAppli->InitParam(nullptr);

        //  MPD : put it after InitParam, else seed modif does not work
        //  Force random creation, just after allocation, because it may need 
        if (aCptCallIntern==1)
        {
           OpenRandom();
        }

        if (anAppli->ModeHelp() || anAppli->ModeArgsSpec())
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
    MMVII_INTERNAL_ASSERT_always(cMemCountable::NbObjLive()==aNbObjLive,"Mem check obj not killed");
    aCptCallIntern--;
    // This was the initial test, stricter, maintain it when call by main
    if (aCptCallIntern==0)
    {
         MMVII_INTERNAL_ASSERT_always(cMemCountable::NbObjLive()==0,"Mem check obj not killed");
    }
    return aRes;
}


const std::string &     cSpecMMVII_Appli::Name() const {return mName;}
tMMVII_AppliAllocator  cSpecMMVII_Appli::Alloc() const {return mAlloc;}
const std::string &     cSpecMMVII_Appli::Comment() const {return mComment;}
const std::string &     cSpecMMVII_Appli::NameFile() const {return mNameFile;}
const cSpecMMVII_Appli::tVaF &  cSpecMMVII_Appli::Features() const {return mVFeatures;}
const cSpecMMVII_Appli::tVaDT &  cSpecMMVII_Appli::VInputs() const {return mVInputs;}
const cSpecMMVII_Appli::tVaDT &  cSpecMMVII_Appli::VOutputs() const {return mVOutputs;}


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
        TheVecAll.push_back(&TheSpec_EditCalcMetaDataImage);
        TheVecAll.push_back(&TheSpecWalkman);
        TheVecAll.push_back(&TheSpecDaisy);
        TheVecAll.push_back(&TheSpecCatVideo);
        TheVecAll.push_back(&TheSpecReduceVideo);
        TheVecAll.push_back(&TheSpecTestGraphPart);
        TheVecAll.push_back(&TheSpec_TestEigen);
        TheVecAll.push_back(&TheSpec_ComputeParamIndexBinaire);
        TheVecAll.push_back(&TheSpecTestRecall);
        TheVecAll.push_back(&TheSpecScaleImage);
        TheVecAll.push_back(&TheSpec_StackIm);
        TheVecAll.push_back(&TheSpecCalcDiscIm);
        TheVecAll.push_back(&TheSpecCalcDescPCar);
        TheVecAll.push_back(&TheSpecMatchTieP);
	TheVecAll.push_back(&TheSpec_TiePConv);
	TheVecAll.push_back(&TheSpec_ToTiePMul);
        TheVecAll.push_back(&TheSpecEpipGenDenseMatch);
        TheVecAll.push_back(&TheSpecEpipDenseMatchEval);
        TheVecAll.push_back(&TheSpecGenSymbDer);
        TheVecAll.push_back(&TheSpecKapture);
        TheVecAll.push_back(&TheSpecFormatTDEDM_WT);
        TheVecAll.push_back(&TheSpecFormatTDEDM_MDLB);
        TheVecAll.push_back(&TheSpecExtractLearnVecDM);
        TheVecAll.push_back(&TheSpecCalcHistoCarac);
        TheVecAll.push_back(&TheSpecCalcHistoNDim);
        TheVecAll.push_back(&TheSpecTestHypStep);
        TheVecAll.push_back(&TheSpecFillCubeCost);
        TheVecAll.push_back(&TheSpecMatchMultipleOrtho);
        TheVecAll.push_back(&TheSpecDMEvalRef);
        TheVecAll.push_back(&TheSpecGenCodedTarget);
        TheVecAll.push_back(&TheSpecExtractCircTarget);
        TheVecAll.push_back(&TheSpecExtractCodedTarget);
        TheVecAll.push_back(&TheSpecGenerateEncoding);
        TheVecAll.push_back(&TheSpecSimulCodedTarget);
        TheVecAll.push_back(&TheSpecCompletUncodedTarget);
        TheVecAll.push_back(&TheSpecDensifyRefMatch);
        TheVecAll.push_back(&TheSpecCloudClip);
        TheVecAll.push_back(&TheSpecMeshDev);
        TheVecAll.push_back(&TheSpecGenMeshDev);
        TheVecAll.push_back(&TheSpecTestCovProp);
        TheVecAll.push_back(&TheSpec_OriConvV1V2);
        TheVecAll.push_back(&TheSpec_OriUncalibSpaceResection);
        TheVecAll.push_back(&TheSpec_OriCalibratedSpaceResection);
        TheVecAll.push_back(&TheSpec_OriCheckGCPDist);
        TheVecAll.push_back(&TheSpec_OriBundlAdj);
        TheVecAll.push_back(&TheSpec_OriRel2Im);
        TheVecAll.push_back(&TheSpecMeshCheck);
        TheVecAll.push_back(&TheSpecProMeshImage);
        TheVecAll.push_back(&TheSpecMeshImageDevlp);
        TheVecAll.push_back(&TheSpecRadiom2ImageSameMod);
        TheVecAll.push_back(&TheSpecRadiomCreateModel);

        TheVecAll.push_back(&TheSpecDistCorrectCirgTarget);
        TheVecAll.push_back(&TheSpec_ImportGCP);
        TheVecAll.push_back(&TheSpec_ImportORGI);
        TheVecAll.push_back(&TheSpec_ImportM32);
        //TheVecAll.push_back(&TheSpecTopoComp);
        TheVecAll.push_back(&TheSpecGenArgsSpec);
        TheVecAll.push_back(&TheSpec_ConvertV1V2_GCPIM);
        TheVecAll.push_back(&TheSpec_SpecSerial);
        TheVecAll.push_back(&TheSpec_PoseCmpReport);
        TheVecAll.push_back(&TheSpec_CGPReport);
        TheVecAll.push_back(&TheSpec_TiePReport);
        TheVecAll.push_back(&TheSpec_BlockCamInit);  // RIGIDBLOC    RB_0_0
        TheVecAll.push_back(&TheSpec_ClinoInit);
        TheVecAll.push_back(&TheSpecRename);
        TheVecAll.push_back(&TheSpec_V2ImportCalib);
        TheVecAll.push_back(&TheSpec_ImportOri);
        TheVecAll.push_back(&TheSpecDicoRename);
        TheVecAll.push_back(&TheSpec_SimulDispl);
        TheVecAll.push_back(&TheSpec_CreateRTL);
        TheVecAll.push_back(&TheSpec_ChSysCo);
        TheVecAll.push_back(&TheSpec_CreateCalib);
        TheVecAll.push_back(&TheSpec_RandomGeneratedDelaunay);
        TheVecAll.push_back(&TheSpec_ComputeTriangleDeformation);
        TheVecAll.push_back(&TheSpec_ComputeTriangleDeformationTrRad);
        TheVecAll.push_back(&TheSpec_ComputeTriangleDeformationTranslation);
        TheVecAll.push_back(&TheSpec_ComputeTriangleDeformationRadiometry);
        TheVecAll.push_back(&TheSpec_ComputeTriangleDeformationRad);
        TheVecAll.push_back(&TheSpec_ImportTiePMul);
        TheVecAll.push_back(&TheSpec_ImportMesImGCP);
        TheVecAll.push_back(&TheSpecImportExtSens);
        TheVecAll.push_back(&TheSpecTestSensor);
        TheVecAll.push_back(&TheSpecParametrizeSensor);
        TheVecAll.push_back(&TheSpec_ChSysCoGCP);
        TheVecAll.push_back(&TheSpec_TutoSerial);
        TheVecAll.push_back(&TheSpec_TutoFormalDeriv);

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

   return nullptr;
}

std::vector<std::string> cSpecMMVII_Appli::TheCmdArgs;

void cSpecMMVII_Appli::ShowCmdArgs(void)
{
    if (TheCmdArgs.size() == 0)
        return;
    std::cout << "========= ARGS OF COMMAND ==========\n";
    for (const auto& aArg: TheCmdArgs)
        std::cout << aArg << " ";
    std::cout << "\n";
}

};

