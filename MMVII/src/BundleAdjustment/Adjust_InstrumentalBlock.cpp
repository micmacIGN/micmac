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

      typedef std::map<std::string,cWeightAv<tREAL8,tREAL8>>  tMapStrAv;
       cBA_BlockInstr
       (
               cMMVII_BundleAdj& ,
               cIrbComp_Block*,
               cIrbComp_Block*,
               const std::vector<std::string> & aVParamPair,
               const std::vector<std::string> & aVParamGauje,
               const std::vector<std::string> & aVParamCur
       );
       virtual ~cBA_BlockInstr();

       void AddClino
            (
                 const std::vector<std::vector<std::string>> & aParamClino
            );



       void OneItere();
       /** Add the gauge constraints  as all pose/rot/tr in the block are defined up to a global pose, we
        *  depending on hard/soft constraint it must be done before mode equation or inside */
       void AddGauge(bool InEq);

       void SaveSigma();
       cIrbCal_Block &  CalBl();          //< Accessor


   private :
       cResolSysNonLinear<tREAL8> & Sys();

       void OneItere_1TS(cIrbComp_TimeS&,tMapStrAv &);

       // For a given pair of Poses, add the constraint of rigidity
       void OneItere_1PairCam(const cIrb_SigmaInstr&,cIrbComp_TimeS&, const tNamePair & aPair);

       // For a given cam, attach calibration of current block to initial value
       void OneIter_Rattach1Cam(const cIrb_Desc1Intsr&);

       // For a time stamp add equations on clinos
       void OneIterClinos(const cIrbComp_TimeS& aDataS,tMapStrAv &);
       void OneIterOneClino(const cIrbComp_TimeS& aDataS,size_t aKClino,tMapStrAv&);
       tREAL8 AddConstrOrthogClino(const std::string&,const std::string&,const cIrb_CstrOrthog &);
       void FrozenClino();

       cMMVII_BundleAdj&         mBA;             //<  Bundle Adj Struct
       cResolSysNonLinear<tREAL8> *mSys;
       cIrbComp_Block *          mCompbBl;        //< "Computation block", contains curent poses +
       cIrbComp_Block *          mCompbBl0;        //< "Computation block", contains curent poses +
       cIrbCal_Block *           mCalBl;          //< "Calibrationblock", contains relative poses +
       cIrbCal_CamSet*           mCalCams;        //< Cal-Block 4 cameras
       cIrbCal_Cam1 *            mCamCalBlock;       //< If single camera for pose-instr
       cIrbCal_ClinoSet *        mCalClino;
       cIrbCal_Block *           mCalBl0;          //< "Calibrationblock", contains relative poses +
       cIrbCal_ClinoSet *        mCalClino0;
       bool                      mWithClino;
       bool                      mVertClinoFree;
       std::vector<bool>         mDegClinFree;
       tREAL8                    mMulSigmaClino;
       tREAL8                    mMulSigmOrthogCl;


       cIrbCal_Cam1 &            mMasterCam;      //<  Master cam to fix Gauje

       std::vector<std::string>  mVParams;        //< copy of parameters
       cCalculator<tREAL8> *     mEqRigCam;       //< Calculator for pair of camera
       cCalculator<tREAL8> *     mEqRatRC;        //< calculator for rattachment to calib of rig-cam
       cCalculator<tREAL8> *     mEqClino;        //< calculator for clino
       cCalculator<tREAL8> *     mEqOrthog;       //< calculator for enforcing orthogonality of clinos
       cP3dNormWithUK            mVertical;       //< store the possibly unknown vertical

       tREAL8                               mMulSigmaTr;     //< Multiplier for sigma-trans
       tREAL8                               mMulSigmaRot;    //< Multiplier for sigma-rot
       tINT4                                mModeSaveSigma;
       tREAL8                               mGaujeTr;   //< If <=0  : "hard" freeze
       tREAL8                               mGaujeRot;  //< If <=0  : "hard" freeze

       std::map<tNamePair,cIrb_SigmaInstr>  mSigmaPair;      //< Sigma a posteriori for pair of images
       cWeightAv<tREAL8,tREAL8>             mAvgTr;
       cWeightAv<tREAL8,tREAL8>             mAvgRot;
       std::map<std::string,tPoseR>         mPoseInit;       //< Used in case of rattachment
       bool                                 mUseRat2CurrBR;  //< Is there rattachment to current
       tREAL8                               mMulSigTrCurBR;     //<
       tREAL8                               mMulSigRotCurBR;    //<
       int                                  mNbEqPair;

};


cBA_BlockInstr::cBA_BlockInstr
(
        cMMVII_BundleAdj& aBA,
        cIrbComp_Block * aCompBl,
        cIrbComp_Block * aCompBl0,
        const std::vector<std::string> & aVParamsPair,
        const std::vector<std::string> & aVParamGauje,
        const std::vector<std::string> & aVParamCur
) :
    mBA            (aBA),
    mSys           (nullptr),
    mCompbBl       (aCompBl),
    mCompbBl0      (aCompBl0),
    mCalBl         (&mCompbBl->CalBlock()),
    mCalCams       (&mCalBl->SetCams()),
    mCamCalBlock   (mCalCams->SingleCamPoseInstr()),
    mCalClino      (&mCalBl->SetClinos()),

    mCalBl0         (&mCompbBl0->CalBlock()),
    mCalClino0      (&mCalBl0->SetClinos()),

    mWithClino     (false),
    mVertClinoFree (false),
    mDegClinFree   (),
    mMulSigmaClino   (1.0),
    mMulSigmOrthogCl (1.0),
    mMasterCam     (mCalCams->MasterCam()),
    mVParams       (aVParamsPair),
    mEqRigCam      (EqBlocRig(true,1,true)),
    mEqRatRC       (EqBlocRig_RatE(true,1,true)),
    mEqClino       (EqBlocRig_Clino(true,1,true)),
    mEqOrthog      (EqBlocRig_Orthog(true,1,true)),
    mVertical      (cPt3dr(0,0,1),"Vertical","Vertical"),
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


    for (auto & aCalC : mCalCams->VCams() )
    {
        // communicate to the system of equation the unknowns of the bloc
        mBA.SetIntervUK().AddOneObj(&aCalC.PoseUKInBlock());
       // memorize initial value for possible forcing to it
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
    delete mCompbBl0;
}

cIrbCal_Block &  cBA_BlockInstr::CalBl()
{
    return *mCalBl;
}
cResolSysNonLinear<tREAL8> & cBA_BlockInstr::Sys()
{
    if (mSys==nullptr)
        mSys = mBA.Sys();

    return *mSys;
}


/**********************************************************/
/*                                                        */
/*                       Clino                            */
/*                                                        */
/**********************************************************/

void cBA_BlockInstr::OneIterOneClino(const cIrbComp_TimeS& aDataS,size_t aKC,tMapStrAv & aMap)
{
    cSensorCamPC *  aCam = aDataS.SetCams().SingleCamPoseInstr() ;

    // not all image have clino measures
    if (aKC>= aDataS.SetClino().NbMeasure())
        return;

    // StdOut() << " OneIterOneClino " << aCam->NameImage() << "\n";

    cIrbCal_Clino1 &         aCalClino  = mCalClino->KthClino(aKC);
    const cIrbComp_Clino1 &  aDataClino = aDataS.SetClino().KthMeasure(aKC);


    std::vector<int>      aVIndexes;
    std::vector<double>   aVecObs;

    aCam->Pose_WU().PushObs(aVecObs,false);
    aCam->PushIndexes(aVIndexes);
    aCalClino.CurPNorm().AddIdexesAndObs(aVIndexes,aVecObs);
    mVertical.AddIdexesAndObs(aVIndexes,aVecObs);
    aVecObs.push_back(aDataClino.Angle());
    aCalClino.PolCorr().PushIndexes(aVIndexes);


    tREAL8 aSigma = mMulSigmaClino * mCalBl->DescrIndiv(aCalClino.Name()).Sigma().SigmaRot();

    Sys().R_CalcAndAddObs
    (
       mEqClino,  // the equation itself
       aVIndexes,
       aVecObs,
       cResidualWeighterExplicit<tREAL8>(true,{aSigma})
    );

    tREAL8 aResidual = mEqClino->ValComp(0,0);

    if (0)
    {
      StdOut() <<  " OneIterOneClino " << aCalClino.Name()
              << " Residual=" << Rad2DMgon(std::abs(mEqClino->ValComp(0,0)))
              << " Residual=" << aResidual

              << "\n";
    }

    if (0)
    {
        tREAL8 aSTeta = std::sin(aDataClino.Angle());
        cPt3dr aVert = mVertical.GetPNorm();
        cPt3dr aDirC = aCalClino.CurPNorm().GetPNorm();
        cPt3dr aDirInM =aCam->Pose().Rot().Value(aDirC);

        tREAL8 aSinTObs = Scal(aVert,aDirInM);

        StdOut()  << " SIiINN " << Rad2DMgon(aSTeta-aSinTObs)
                  << " " <<  Rad2DMgon(aSTeta)
                  << " " <<  Rad2DMgon(aSinTObs)
                  << " V=" << aVert<< "\n";
     }

    aMap[aCalClino.Name()].Add(1.0,std::abs(aResidual));
}

tREAL8 cBA_BlockInstr::AddConstrOrthogClino(const std::string& aName1,const std::string& aName2,const cIrb_CstrOrthog & aCstrOrthog)
{

    cIrbCal_Clino1 &  aCal1  = *mCalClino->ClinoFromName(aName1);
    cIrbCal_Clino1 &  aCal2  = *mCalClino->ClinoFromName(aName2);

    std::vector<int>      aVIndexes;
    std::vector<double>   aVecObs;

    aCal1.CurPNorm().AddIdexesAndObs(aVIndexes,aVecObs);
    aCal2.CurPNorm().AddIdexesAndObs(aVIndexes,aVecObs);

    tREAL8 aSigma = mMulSigmOrthogCl * aCstrOrthog.Sigma();

    Sys().R_CalcAndAddObs
    (
       mEqOrthog,  // the equation itself
       aVIndexes,
       aVecObs,
       cResidualWeighterExplicit<tREAL8>(true,{aSigma})
    );

//  StdOut() << "SSSSSSSSSS=" << mMulSigmOrthogCl << " " << aSigma << "\n";
    return mEqOrthog->ValComp(0,0);

}


void cBA_BlockInstr::OneIterClinos(const cIrbComp_TimeS& aDataS,tMapStrAv & aMap)
{
    if (!mWithClino)
        return;

    for (size_t aKC=0 ; aKC< mCalClino-> NbClino() ; aKC++)
    {
        OneIterOneClino(aDataS,aKC,aMap);
    }
}


void cBA_BlockInstr::AddClino
     (
          const std::vector<std::vector<std::string>> & aParamClino
     )
{
    mWithClino = true;

    MMVII_INTERNAL_ASSERT_tiny(aParamClino.size()>=1,"Size < 1 for param clino");

    //  [NameClino,SigmClino,SigmOrthog?]   [ VertFree?,OkNewTS?]?    [DegFree?]

    {
         std::vector<std::string> aParamSigma = GetDef(aParamClino,0,std::vector<std::string>());
         mMulSigmaClino  =   cStrIO<double>::FromStr(GetDef(aParamSigma,1,std::string("1.0")));
         mMulSigmOrthogCl =  cStrIO<double>::FromStr(GetDef(aParamSigma,2,std::string("1.0")));
    }

    bool OkNewTS = false;
    {
         std::vector<std::string> aParamFreeze = GetDef(aParamClino,1,std::vector<std::string>());
         mVertClinoFree = cStrIO<int>::FromStr(GetDef(aParamFreeze,0,std::string("0")));
         OkNewTS =  cStrIO<int>::FromStr(GetDef(aParamFreeze,1,std::string("0")));
    }

    {
         std::vector<std::string> aParamDegFree = GetDef(aParamClino,2,std::vector<std::string>());
         for (const auto& aStr : aParamDegFree)
             mDegClinFree.push_back(cStrIO<bool>::FromStr(aStr));
    }


    // do we accept in clino measures data that do not correspond to any time stamp

    // Put the values (angles) of clino at each time stamp
    mCompbBl->SetClinoValues(OkNewTS);

    for (size_t aKC=0 ; aKC< mCalClino-> NbClino() ; aKC++)
    {
        cIrbCal_Clino1 & aCalCK = mCalClino->KthClino(aKC);

        mBA.SetIntervUK().AddOneObj(&aCalCK.CurPNorm());
        mBA.SetIntervUK().AddOneObj(&aCalCK.PolCorr());
    }

     mBA.SetIntervUK().AddOneObj(&mVertical);
}

void cBA_BlockInstr::FrozenClino()
{
    if (! mWithClino) return;

    // For now cannot freeze as AddClino was not executed
    //  StdOut() << "***************************************%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%NO FrozenClino \n";
    if (! mVertClinoFree)
        Sys().SetFrozenVarCurVal(mVertical,mVertical.DuDv());

    for (size_t aKC=0 ; aKC< mCalClino-> NbClino() ; aKC++)
    {
        cIrbCal_Clino1 &         aCalClino  = mCalClino->KthClino(aKC);
        cVectorUK &  aPolCol = aCalClino.PolCorr();
        std::vector<tREAL8> & aVCoeff =  aPolCol.Vect();

        for (size_t aD=0 ; aD<aVCoeff.size() ; aD++)
        {
            if (!GetDef(mDegClinFree,aD,false))
               Sys().SetFrozenVarCurVal(aPolCol,aVCoeff.at(aD));
        }
    }
}


/**********************************************************/
/*                                                        */
/*                       Camera                           */
/*                                                        */
/**********************************************************/

void cBA_BlockInstr::OneIter_Rattach1Cam(const cIrb_Desc1Intsr& aDesc)
{
    if (!mUseRat2CurrBR)
       return;

    int aKCam = mCalCams->IndexCamFromNameCalib(aDesc.NameInstr());

    if (aKCam<0) return;

    cPoseWithUK &  aPUK =  mCalCams->KthCam(aKCam).PoseUKInBlock();
    tPoseR aP0 = *MapGet(mPoseInit,aDesc.NameInstr());


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

  //  StdOut() << " aPUK.PushIndexes " << aVInd << "\n";

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
        FrozenClino();
        if (mGaujeTr<=0)
            Sys().SetFrozenVarCurVal(aPBl,aC);
        if (mGaujeRot<=0)
            Sys().SetFrozenVarCurVal(aPBl,aW);
     }
}

/**********************************************************/
/*                                                        */
/*                    Global                              */
/*                                                        */
/**********************************************************/


void cBA_BlockInstr::OneItere_1TS(cIrbComp_TimeS& aDataS,tMapStrAv &aMap)
{

    // Parse pair of images
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

    OneIterClinos(aDataS,aMap);
}


void cBA_BlockInstr::OneItere()
{
   mNbEqPair =0;
   mAvgTr.Reset();
   mAvgRot.Reset();

   mSigmaPair.clear();

   tMapStrAv aMapClino;

   // Parse all "time stamp" to add equation
   for ( auto & [aTimeS,aDataTS] : mCompbBl->DataTS())
   {
       OneItere_1TS(aDataTS,aMapClino);
   }

   // Add Rattachment to initial block
   for ( auto & [aNameCal,aDescr] : mCalBl->DescrIndiv() )
   {
      if (aDescr.Type() == eTyInstr::eCamera)
      {
         OneIter_Rattach1Cam(aDescr);
      }
   }

   // Orthognal constraint
   if (mWithClino)
   {
      int aNbResOC=0;  // Number of constraint on clino
      std::string aMsg = "Clino-Ortho : ";
      for (const auto& [aPair, aCstr] : mCalBl->CstrOrthog())
      {
          const cIrb_Desc1Intsr &  aSI1 = mCalBl->DescrIndiv(aPair.V1());
          const cIrb_Desc1Intsr &  aSI2 = mCalBl->DescrIndiv(aPair.V2());
          if ((aSI1.Type()==eTyInstr::eClino) && (aSI2.Type()==eTyInstr::eClino))
          {
              tREAL8 aRes = AddConstrOrthogClino(aPair.V1(),aPair.V2(),aCstr);
              aNbResOC ++;
              aMsg += "[" +aPair.V1()+ "," +aPair.V2() + ":" + ToStr(Rad2DMgon(aRes)) + "" ;
          }
          else
              MMVII_INTERNAL_ERROR("Unhandled combination of instrument in  Orthog constraint");
      }
      if (aNbResOC)
          StdOut() << "  * " << aMsg << "\n";

      {
         StdOut() << "  * ResClino : " ;
         cWeightAv<tREAL8,tREAL8> aAvgGlob;
         for (auto & [aName,anAvg] : aMapClino)
         {
             tREAL8 aVAv = anAvg.Average();
             aAvgGlob.Add(1.0,aVAv);
              StdOut() << "[" << aName << " : "  << Rad2DMgon(aVAv) << "] " ;
         }
         StdOut() << " ---- Glob=" << Rad2DMgon(aAvgGlob.Average()) ;
      }
      if (mVertClinoFree)
      {
           StdOut() << " DVert=" << Rad2DMgon(Norm2(mVertical.GetPNorm()-cPt3dr(0,0,1))) ;
      }
      StdOut() << "\n";

      {
         StdOut() <<  "  * EvolClino ";
         cWeightAv<tREAL8,tREAL8> aAvgGlob;
         for (size_t aKC=0 ; aKC<mCalClino->NbClino() ; aKC++)
         {
              cIrbCal_Clino1 & aCl  = mCalClino->KthClino(aKC);
              cIrbCal_Clino1 & aCl0 = mCalClino0->KthClino(aKC);
              cPt3dr aDp = aCl.CurPNorm().GetPNorm()-  aCl0.CurPNorm().GetPNorm();
              StdOut() <<  "[" << aCl.Name() << " : " << Rad2DMgon(Norm2(aDp)) << "] ";
              aAvgGlob.Add(1.0,Norm2(aDp));
         }
         StdOut() << " AvgGlob=" << Rad2DMgon(aAvgGlob.Average()) << "\n";
      }
   }
   //const std::map<tNamePair,cIrb_CstrOrthog> &  CstrOrthog() const;



   // Eventually had "soft" gauge
   AddGauge(true);

   StdOut() << "  * Residual IntrBlocCam/Pair "
            << " Tr=" << std::sqrt(mAvgTr.Average())
            << " Rot=" << std::sqrt(mAvgRot.Average()) << "\n";
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

std::string GetNameBlock(const std::vector<std::string> & aVNames)
{
    std::string aNameBlock = GetDef(aVNames,0,std::string(""));
    return (aNameBlock=="") ? cIrbCal_Block::theDefaultName : aNameBlock;
}


void cMMVII_BundleAdj::AddClinoBlokcInstr(const std::vector<std::vector<std::string>> &aVVParam)
{
    StdOut() << " --- BEGIN AddClinoBlokcInstr\n";

    // std::string aNameBlock = GetDef(aVParamPairCam,0,std::string(""));
    std::vector<std::string> aVParam0 = GetDef(aVVParam,0,std::vector<std::string>());
    std::string aNameBlock = GetNameBlock(aVParam0);

    for (const auto & aBlock : mVecBlockInstrAdj)
    {
        if (aBlock->CalBl().NameBloc()== aNameBlock)
        {
            aBlock->AddClino(aVVParam);
            StdOut() << " --- END AddClinoBlokcInstr\n";
            return;
        }
    }

    MMVII_UnclasseUsEr("Could not find bloc in Clino-Adjust");
}


void cMMVII_BundleAdj::AddBlockInstr(const std::vector<std::vector<std::string>> & aVVParam)
{
    if (! mPhProj->DPBlockInstr().DirInIsInit())
    {
        MMVII_UnclasseUsEr("Dir for bloc of instrument not init with parameter for BOI/Compensation");
    }

     const std::vector<std::string> & aVParamPairCam = aVVParam.at(0);
     // std::string aNameBlock = GetDef(aVParamPairCam,0, cIrbCal_Block::theDefaultName);GetNameBlock
     std::string aNameBlock = GetNameBlock(aVParamPairCam);

     /*
     std::string aNameBlock = GetDef(aVParamPairCam,0,std::string(""));
     if (aNameBlock=="")
         aNameBlock = cIrbCal_Block::theDefaultName;
*/
     cIrbComp_Block * aBlock = new cIrbComp_Block(*mPhProj ,aNameBlock);
     cIrbComp_Block * aBlock0 = new cIrbComp_Block(*mPhProj ,aNameBlock);


     std::vector<std::string> aParamCur;
     if (aVVParam.size() >=3)
     {
         aParamCur = aVVParam.at(2);
     }

     mVecBlockInstrAdj.push_back
     (
          new cBA_BlockInstr(*this,aBlock,aBlock0,aVParamPairCam,aVVParam.at(1),aParamCur)
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

