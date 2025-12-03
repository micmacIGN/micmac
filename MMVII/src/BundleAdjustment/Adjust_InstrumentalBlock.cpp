#include "BundleAdjustment.h"
#include "MMVII_InstrumentalBlock.h"
#include "MMVII_util_tpl.h"


namespace MMVII
{

//   Block[  [NameBloc,SigmaPairTr,SigmPairRot]  [?GaugeTr,?GaugeRot] ]


/* ************************************************************************** */
/*                                                                            */
/*                         cBA_BlockInstr                                     */
/*                                                                            */
/* ************************************************************************** */


class cBA_BlockInstr : public cMemCheck
{
   public :
       cBA_BlockInstr
       (
               cMMVII_BundleAdj& ,
               cIrbComp_Block*,
               const std::vector<std::string> & aVParamPair,
               const std::vector<std::string> & aVParamGauje,
               const std::vector<std::string> & aVParamCur
       );
       virtual ~cBA_BlockInstr();

       void OneItere();
       /** Add the gauge constraints  as all pose/rot/tr in the block are defined up to a global pose, we
        *  depending on hard/soft constraint it must be done before mode equation or inside */
       void AddGauge(bool InEq);

       void SaveSigma();
       cIrbCal_Block &  CalBl();          //< Accessor


   private :
       cResolSysNonLinear<tREAL8> & Sys();

       void OneItere_1TS(cIrbComp_TimeS&);

       // For a given pair of Poses, add the constraint of rigidity
       void OneItere_1PairCam(const cIrb_SigmaInstr&,cIrbComp_TimeS&, const tNamePair & aPair);
       void OneIter_Rattach1Cam(const cIrb_Desc1Intsr&);


       cMMVII_BundleAdj&         mBA;             //<  Bundle Adj Struct
       cResolSysNonLinear<tREAL8> *mSys;
       cIrbComp_Block *          mCompbBl;        //< "Computation block", contains curent poses +
       cIrbCal_Block *           mCalBl;          //< "Calibrationblock", contains relative poses +
       cIrbCal_CamSet*           mCalCams;        //< Cal-Block 4 cameras
       cIrbCal_Cam1 &            mMasterCam;      //<  Master cam to fix Gauje

       std::vector<std::string>  mVParams;        //< copy of parameters
       cCalculator<tREAL8> *     mEqRigCam;       //< Calculator for pair of camera
       cCalculator<tREAL8> *     mEqRatRC;        //< calculator for rattachment to calib of rig-cam
       tREAL8                               mMulSigmaTr;     //< Multiplier for sigma-trans
       tREAL8                               mMulSigmaRot;    //< Multiplier for sigma-rot
       tINT4                                mModeSaveSigma;
       tREAL8                               mGaujeTr;
       tREAL8                               mGaujeRot;

       std::map<tNamePair,cIrb_SigmaInstr>  mSigmaPair;      //< Sigma a posteriori for pair of images
       cWeightAv<tREAL8,tREAL8>             mAvgTr;
       cWeightAv<tREAL8,tREAL8>             mAvgRot;
       std::map<std::string,tPoseR>         mPoseInit;       //< Used in case of rattachment
       bool                                 mUseRat2CurrBR;  //< Is there rattachment to current
       tREAL8                               mMulSigTrCurBR;     //<
       tREAL8                               mMulSigRotCurBR;    //<
       int                                  mNbEqPair;

};

cResolSysNonLinear<tREAL8> & cBA_BlockInstr::Sys()
{
    if (mSys==nullptr)
        mSys = mBA.Sys();

    return *mSys;
}


cBA_BlockInstr::cBA_BlockInstr
(
        cMMVII_BundleAdj& aBA,
        cIrbComp_Block * aCompBl,
        const std::vector<std::string> & aVParamsPair,
        const std::vector<std::string> & aVParamGauje,
        const std::vector<std::string> & aVParamCur
) :
    mBA            (aBA),
    mSys           (nullptr),
    mCompbBl       (aCompBl),
    mCalBl         (&mCompbBl->CalBlock()),
    mCalCams       (&mCalBl->SetCams()),
    mMasterCam     (mCalCams->MasterCam()),
    mVParams       (aVParamsPair),
    mEqRigCam      (EqBlocRig(true,1,true)),
    mEqRatRC       (EqBlocRig_RatE(true,1,true)),
    mMulSigmaTr    (cStrIO<double>::FromStr(GetDef(aVParamsPair,1,std::string("1.0")))),
    mMulSigmaRot   (cStrIO<double>::FromStr(GetDef(aVParamsPair,2,std::string("1.0")))),
    mModeSaveSigma (cStrIO<int>::FromStr(GetDef(aVParamsPair,3,std::string("1")))),
    mGaujeTr       (cStrIO<double>::FromStr(GetDef(aVParamGauje,0,std::string("0.0")))),
    mGaujeRot      (cStrIO<double>::FromStr(GetDef(aVParamGauje,1,std::string("0.0")))),
    mUseRat2CurrBR (! aVParamCur.empty())
{


    //  Add amWithCurrll the pose to construct the Time-Stamp structure
    for (auto aPtrCam : mBA.VSCPC())
        mCompbBl->AddImagePose(aPtrCam,true);

    // communicate to the system of equation the unknowns of the bloc
    for (auto & aCalC : mCalCams->VCams() )
    {
        mBA.SetIntervUK().AddOneObj(&aCalC.PoseUKInBlock());
        mPoseInit[aCalC.NameCal()]= aCalC.PoseUKInBlock().Pose();
    }

    if (mUseRat2CurrBR)
    {
        MMVII_INTERNAL_ASSERT_always(aVParamCur.size()==2,"Bad size for Block-Rat to Cur Block Rigid");
        mMulSigTrCurBR = cStrIO<double>::FromStr(aVParamCur.at(0));
        mMulSigRotCurBR = cStrIO<double>::FromStr(aVParamCur.at(1));
    }
}

cBA_BlockInstr::~cBA_BlockInstr()
{
    delete mCompbBl;
}

cIrbCal_Block &  cBA_BlockInstr::CalBl()
{
    return *mCalBl;
}

void cBA_BlockInstr::OneIter_Rattach1Cam(const cIrb_Desc1Intsr& aDesc)
{
    int aKCam = mCalCams->IndexCamFromNameCalib(aDesc.NameInstr());

    if (aKCam<0) return;

    cPoseWithUK &  aPUK =  mCalCams->KthCam(aKCam).PoseUKInBlock();
    tPoseR aP0 = *MapGet(mPoseInit,aDesc.NameInstr());

/*
    StdOut() << "OneIter_Rattach1CamOneIter_Rattach1Cam "
             << aDesc.NameInstr()
             << " Tr=" << aPUK.Pose().Tr() -aP0.Tr()
             << " Rot="  << (aPUK.Pose().Rot()*aP0.Rot().MapInverse()).ToWPK()
             << "\n";
             */

    if (!mUseRat2CurrBR)
       return;

    // weight multiplier to compense number of equation Pair-Times / Cam
    tREAL8 aMulNb = mNbEqPair / tREAL8(mPoseInit.size());
    std::vector<double>  aWeight;
    for(int aK=0 ; aK<3 ; aK++)
       aWeight.push_back(aMulNb/Square(mMulSigTrCurBR  * aDesc.Sigma().SigmaTr()));

    for(int aK=0 ; aK<9 ; aK++)
       aWeight.push_back(aMulNb/Square(mMulSigRotCurBR * aDesc.Sigma().SigmaRot()));

    // [2.1]  the observation/context are  the coeef of rotation-matrix for linearization ;:
    std::vector<double> aVObs;

    aPUK.PushObs(aVObs,false);
    AppendIn(aVObs,aP0.Tr().ToStdVector());
    aP0.Rot().Mat().PushByLine(aVObs);

    std::vector<int>  aVInd;
    aPUK.PushIndexes(aVInd);

    Sys().R_CalcAndAddObs
    (
       mEqRatRC,  // the equation itself
       aVInd,
       aVObs,
       cResidualWeighterExplicit<tREAL8>(false,aWeight)
    );
}


void cBA_BlockInstr::OneItere_1PairCam
     (
        const cIrb_SigmaInstr& aSigma,
        cIrbComp_TimeS& aDataTS,
        const tNamePair & aPair

     )
{

   //  [0] ============== Extract unkonwns (for bloc & poses) ========

   // [0.1]  Extract indexes of camera-calib in bloc
   int aK1 = mCalCams->IndexCamFromNameCalib(aPair.V1());
   int aK2 = mCalCams->IndexCamFromNameCalib(aPair.V2());

   //  possible that all camera do not belong to bloc
   if ( (aK1<0) || (aK2<0) )
       return;

   cPoseWithUK &  aPBl1 =  mCalCams->KthCam(aK1).PoseUKInBlock();
   cPoseWithUK &  aPBl2 =  mCalCams->KthCam(aK2).PoseUKInBlock();

   // [0.2]  extract  camera-poses from time stamp
   cIrbComp_CamSet &  aCamSet = aDataTS.SetCams();
   cSensorCamPC * aCam1 = aCamSet.KthCam(aK1).CamPC();
   cSensorCamPC * aCam2 = aCamSet.KthCam(aK2).CamPC();

   //  possible all images were not taken/usable
   if ((aCam1==nullptr) || (aCam2==nullptr))
      return;


   //  [1] ============== compute the weightings, taking account "a-priori" sigma and multiplier ========

   std::vector<double>  aWeight;
   for(int aK=0 ; aK<3 ; aK++)
      aWeight.push_back(1.0/Square(mMulSigmaTr  * aSigma.SigmaTr()));

   for(int aK=0 ; aK<9 ; aK++)
      aWeight.push_back(1.0/Square(mMulSigmaRot * aSigma.SigmaRot()));


   //  [2] ============== create vectors of "obs" and indexes ========

   // [2.1]  the observation/context are  the coeef of rotation-matrix for linearization ;:
   std::vector<double> aVObs;

   aCam1->Pose_WU().PushObs(aVObs,false); // false because we dont transpose matrix
   aCam2->Pose_WU().PushObs(aVObs,false);
   aPBl1.PushObs(aVObs,false);
   aPBl2.PushObs(aVObs,false);

    // [2.2] Create a vector of indexes of unknowns :  the pose of block and images
    std::vector<int>  aVInd;
    aCam1->PushIndexes(aVInd);
    aCam2->PushIndexes(aVInd);
    aPBl1.PushIndexes(aVInd);
    aPBl2.PushIndexes(aVInd);


    //  [3] =============   now we are ready to add the equation
    Sys().R_CalcAndAddObs
    (
       mEqRigCam,  // the equation itself
       aVInd,
       aVObs,
       cResidualWeighterExplicit<tREAL8>(false,aWeight)
    );

    // [4] compute residual & accumulates
    tREAL8 aSumTr = 0.0;
    tREAL8 aSumRot = 0.0;

    for (size_t aKU=0 ; aKU<12 ;  aKU++)
    {
        ((aKU<3) ? aSumTr : aSumRot) += (Square(mEqRigCam->ValComp(0,aKU)));
    }


    mAvgTr.Add(1.0,aSumTr);
    mAvgRot.Add(1.0,aSumRot);

    mSigmaPair[aPair].AddNewSigma(cIrb_SigmaInstr(1.0,1.0,std::sqrt(aSumTr),std::sqrt(aSumRot)));
    mNbEqPair++;

}


void cBA_BlockInstr::OneItere_1TS(cIrbComp_TimeS& aDataS)
{
    for ( auto & [aPair,aSigma2] : mCalBl->SigmaPair() )
    {
        const cIrb_Desc1Intsr &  aSI1 = mCalBl->DescrIndiv(aPair.V1());
        const cIrb_Desc1Intsr &  aSI2 = mCalBl->DescrIndiv(aPair.V2());
        if ((aSI1.Type()==eTyInstr::eCamera) && (aSI2.Type()==eTyInstr::eCamera))
        {
               OneItere_1PairCam(aSigma2,aDataS,aPair);
        }
        else
            MMVII_INTERNAL_ERROR("Unhandled combination of instrument in  cBA_BlockInstr::OneItere_1TS");
    }


    //for ()
}


void cBA_BlockInstr::OneItere()
{
    mNbEqPair =0;
   mAvgTr.Reset();
   mAvgRot.Reset();

   mSigmaPair.clear();

   // Parse all "time stamp" to add equation
   for ( auto & [aTimeS,aDataTS] : mCompbBl->DataTS())
   {
       OneItere_1TS(aDataTS);
   }

   for ( auto & [aNameCal,aDescr] : mCalBl->DescrIndiv() )
   {
      if (aDescr.Type() == eTyInstr::eCamera)
      {
         OneIter_Rattach1Cam(aDescr);
      }
   }

   AddGauge(true);

   StdOut() << "  * Residual IntrBlocCam/Pair "
            << " Tr=" << std::sqrt(mAvgTr.Average())
            << " Rot=" << std::sqrt(mAvgRot.Average()) << "\n";
}

void cBA_BlockInstr::AddGauge(bool InEq)
{
     cPoseWithUK &  aPBl =  mMasterCam.PoseUKInBlock();
     cPt3dr &  aC = aPBl.GetRefTr();
     cPt3dr &  aW = aPBl.GetRefOmega();

     if (InEq)
     {
          const cIrb_SigmaInstr & aSigma = mCalBl->DescrIndiv(mMasterCam.NameCal()).Sigma();
          if (mGaujeTr>0)
              Sys().AddEqFixCurVar(aPBl,aC,1.0/Square(mGaujeTr*aSigma.SigmaTr()));
          if (mGaujeRot>0)
              Sys().AddEqFixCurVar(aPBl,aW,1.0/Square(mGaujeRot*aSigma.SigmaRot()));
     }
     else
     {
        if (mGaujeTr<=0)
            Sys().SetFrozenVarCurVal(aPBl,aC);
        if (mGaujeRot<=0)
            Sys().SetFrozenVarCurVal(aPBl,aW);
     }
}

void cBA_BlockInstr::SaveSigma()
{
    if (mModeSaveSigma==0)
    {
        // Case nothing to do
    }
    else if (mModeSaveSigma==1)
    {
        // case we save empirical sigma between pairs
        mCalBl->SetSigmaPair(mSigmaPair);
        mCalBl->SetSigmaIndiv(mSigmaPair);
    }
    else
    {
        // To do later, an evaluation based on var/covar
        MMVII_UnclasseUsEr("Unhandled value for SaveSigma ");
    }
}



/* ************************************************************************** */
/*                                                                            */
/*                         cBA_BlockInstr                                     */
/*                                                                            */
/* ************************************************************************** */


void cMMVII_BundleAdj::AddBlockInstr(const std::vector<std::vector<std::string>> & aVVParam)
{
    if (! mPhProj->DPBlockInstr().DirInIsInit())
    {
        MMVII_UnclasseUsEr("Dir for bloc of instrument not init with parameter for BOI/Compensation");
    }

     const std::vector<std::string> & aVParamPairCam = aVVParam.at(0);
     std::string aNameBlock = GetDef(aVParamPairCam,0,std::string(""));
     if (aNameBlock=="")
         aNameBlock = cIrbCal_Block::theDefaultName;

     cIrbComp_Block * aBlock = new cIrbComp_Block(*mPhProj ,aNameBlock);

     std::vector<std::string> aParamCur;
     if (aVVParam.size() >=3)
     {
         aParamCur = aVVParam.at(2);
     }

     mVecBlockInstrAdj.push_back
     (
          new cBA_BlockInstr(*this,aBlock,aVParamPairCam,aVVParam.at(1),aParamCur)
     );

}

void cMMVII_BundleAdj::SetHardGaugeBlockInstr()
{
    for (auto & aBlock : mVecBlockInstrAdj)
        aBlock->AddGauge(false);
}


void cMMVII_BundleAdj::IterOneBlockInstr()
{
    for (auto & aBlock : mVecBlockInstrAdj)
        aBlock->OneItere();
}

void cMMVII_BundleAdj::SaveBlockInstr()
{
    if (! mPhProj->DPBlockInstr().DirOutIsInit())
    {
        MMVII_USER_WARNING("Block of instrument not saved");
        return;
    }

    for (auto & aBlock : mVecBlockInstrAdj)
    {
        aBlock->SaveSigma();
        mPhProj->SaveRigBoI( aBlock->CalBl());
    }
}



void cMMVII_BundleAdj::DeleteBlockInstr()
{
    DeleteAllAndClear(mVecBlockInstrAdj);
}




};

