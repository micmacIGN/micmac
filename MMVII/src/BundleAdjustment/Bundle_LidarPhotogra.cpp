#include "BundleAdjustment.h"
#include "MMVII_Interpolators.h"
#include "MMVII_Tpl_Images.h"
#include "MMVII_Geom2D.h"
#include "../ImagesBase/cGdalApi.h"
#include "cOpticalFlow.h"


namespace MMVII
{



cBA_LidarBase::cBA_LidarBase(cPhotogrammetricProject * aPhProj,
                                     cMMVII_BundleAdj& aBA, const std::vector<std::string>& aParam) :
    mPhProj     (aPhProj),
    mBA         (aBA),                                 // memorize the bundel adj class itself (access to optimizer)
    mInterp     (nullptr),                             // interpolator see bellow
    mEq         (nullptr),                             // equation of egalisation Lidar/Phgr
    mWFactor      (NAN),
    mNbUsedPoints (0),
    mNbUsedObs (0)
{

}

cBA_LidarBase::~cBA_LidarBase()
{
    delete mEq;
    delete mInterp;
}


void cBA_LidarBase::init(const std::vector<std::string>& aParam, size_t aWeightParamIndex, size_t aInterpolParamIndex)
{
    mWFactor = (1/Square(cStrIO<double>::FromStr(aParam.at(aWeightParamIndex))));
    //  By default  use tabulation of apodized sinus cardinal
    std::vector<std::string> aParamInt {"Tabul","1000","SinCApod","10","10"};
    // if interpolator is not empty
    if ((aParam.size() >=aInterpolParamIndex+1) && (!aParam.at(aInterpolParamIndex).empty()))
    {
        // if specified, take user's param
        aParamInt = Str2VStr(aParam.at(aInterpolParamIndex));
    }


    // create the interpolator itself
    mInterp  = cDiffInterpolator1D::AllocFromNames(aParamInt);
}


cBA_LidarPhotogra::cBA_LidarPhotogra(cPhotogrammetricProject * aPhProj,
                                     cMMVII_BundleAdj& aBA, const std::vector<std::string>& aParam) :
    cBA_LidarBase(aPhProj, aBA, aParam),
    mModeSim    (Str2E<eImatchCrit>(aParam.at(0))),    // mode of matching
    mVSCams     ({}),
    mPertRad    (false),
    mPreselectPatches  (false),
    mNbPointByPatch (25),
    mBoxSelected (cBox2dr::Empty()),
    mNbScale(1),
    mInitRes (0.2),
    mDensity(5.0),
    mThresholdAcceptCorrel(0.2),
    mDImQualityMap(nullptr),
    mDImQualityMapY(nullptr)
{
    init(aParam, 2, 3);

    // Re allocate mInterp as it does not get initialized using init method
    std::vector<std::string> aParamInt {"Tabul","1000","SinCApod","10","10"};
    if(aParam.size()>=4 and aParam.at(3)!="")
        aParamInt = Str2VStr(aParam.at(3));

    mInterp  = cDiffInterpolator1D::AllocFromNames(aParamInt);

   if (aParam.size() >=5)
   {
       mPertRad = (aParam.at(4) != "");
   }
   if (aParam.size() >=6)
   {
        mNbPointByPatch = cStrIO<size_t>::FromStr(aParam.at(5));
        MMVII_INTERNAL_ASSERT_User((mModeSim!=eImatchCrit::eDifRad) || (mNbPointByPatch==1),
                                  eTyUEr::eUnClassedError,"Only 1 point per patch in "+ToStr(eImatchCrit::eDifRad)+" mode");
   }


   // Additional params
   if (aParam.size() >=7)
   {
       mDensity = cStrIO<size_t>::FromStr(aParam.at(6));
   }

   if (aParam.size() >=8)
   {
       mThresholdAcceptCorrel = cStrIO<size_t>::FromStr(aParam.at(7));
   }

   if (aParam.size()>=9)
   {
       mNbScale = cStrIO<size_t>::FromStr(aParam.at(8));
   }

   if (aParam.size()>=10)
   {
       mInitRes = cStrIO<tREAL8>::FromStr(aParam.at(9));
   }
   if (aParam.size()>=11)
   {
       mPreselectPatches = (aParam.at(10) != "");
   }

   // parse the camera and create images
   for (const auto aPtrCam : aBA.VSCPC())
   {
       // is it a central perspective camera ?
       if (aPtrCam->IsSensorCamPC())
       {
           mVCam.push_back(aPtrCam->GetSensorCamPC());  // yes get it
           mVIms.push_back(ReadIm2DGen(aPtrCam->NameImage()));  // read the image

           if (mNbScale>1)
           {
                // create multi scale cameras for multi scale patch sampling
               auto aCam = mVCam.back();
               mVSCams.push_back(std::vector<cSensorCamPC *>());
               for ( int aNb=1; aNb< mNbScale; aNb++)
                    {
                       tREAL8 aCurrentScale= pow(2,aNb);
                       tREAL8 aCurrentFocal= aCam->InternalCalib()->F()/aCurrentScale;
                       cPerspCamIntrCalib* aCalib = cPerspCamIntrCalib::Alloc
                           (
                               cDataPerspCamIntrCalib
                               (
                                   aCam->InternalCalib()->Name(),
                                   eProjPC::eStenope,
                                   aCam->InternalCalib()->DegDir(),
                                   aCam->InternalCalib()->VParamDist(),
                                   cMapPProj2Im(aCurrentFocal,
                                                aCam->InternalCalib()->PP()/aCurrentScale),
                                   cDataPixelDomain(Pt_round_up(
                                                      DivCByC(ToR(aCam->InternalCalib()->SzPix()),
                                                                cPt2dr(aCurrentScale,aCurrentScale))
                                                                )
                                                    ),
                                   cPt3di(0,0,1),
                                   10
                                   )
                            );
                       cMMVII_Appli::AddObj2DelAtEnd(aCalib);
                       mVSCams.back().push_back(new cSensorCamPC(aCam->NameImage(),
                                                                  aCam->Pose(),
                                                                  aCalib));
                    }
            }
       }

       else
       {
           MMVII_UnclasseUsEr("cBA_LidarPhotogra : sensor is not central perspective");
       }
   }

    // read images before 1st iteration // TODO: read only images that may correspond to scans?

   if (mPertRad)
    {
        for (const auto aPtrCam : aBA.VSCPC())
        {
            auto & aImage = aPtrCam->LoadImage();

            for (auto  aPix : aImage)
            {
                tREAL8 aMul =   (3+ sin(aPix.x()/70.0)) / 4.0;
                aImage.VI_SetV(aPix,aImage.VI_GetV(aPix)*aMul); // keep using ints?
            }
        }
    }
}

void cBA_LidarPhotogra::InitEq(bool aScanPoseUk)
{
    if (mModeSim==eImatchCrit::eDifRad)
        mEq = EqEqLidarImPonct (true,1,aScanPoseUk);
    else if (mModeSim==eImatchCrit::eCensus)
        mEq = EqEqLidarImCensus(true,1, aScanPoseUk);
    else if (mModeSim==eImatchCrit::eCorrel)
        mEq = EqEqLidarImCorrel(true,1, aScanPoseUk);
    else
    {
        MMVII_UnclasseUsEr("Bad enum for cBA_LidarPhotogra");
    }
}

cBA_LidarPhotograTri::cBA_LidarPhotograTri(cPhotogrammetricProject * aPhProj,
                                           cMMVII_BundleAdj& aBA,
                                           const std::vector<std::string>& aParam) :
    cBA_LidarPhotogra(aPhProj, aBA, aParam), mTri(nullptr)
{
    InitEq(false);

    std::string aLidarFileName = aParam.at(1);
    bool availableReaders = UCaseEqual(LastPostfix(aLidarFileName),"ply") ||
                            UCaseEqual(LastPostfix(aLidarFileName),"laz") ;
    MMVII_INTERNAL_ASSERT_User(availableReaders,
                               eTyUEr::eUnClassedError,"Lidar PLY or LAZ file mandatory in triangulation mode, got \"" + aParam.at(1) + "\"");
    mTri = new cTriangulation3D<tREAL4>(aLidarFileName);
    // Activate patch selection by correl
    isForSelection=true;
    // Creation of the patches, to comment ...
    if (mModeSim!=eImatchCrit::eDifRad)
    {
        tREAL8 aDistMoy = std::sqrt(mNbPointByPatch / (mDensity*M_PI));
        tREAL8 aDistReject =  aDistMoy *1.2;

        if (mPreselectPatches)
        {
            mTri->MakePatches(mLPatchesI,aDistMoy,aDistReject,15);
        }
        else
        {
            if (mNbScale == 1)
                mTri->MakePatchesTargetted(mLPatchesI,
                                          aDistMoy,
                                          aDistReject,
                                          15,
                                          mVCam,
                                          mVIms,
                                          0.85);

            if (mNbScale > 1)
                mTri->MakePatchesTargetted(mLPatchesI,
                                          aDistMoy,
                                          aDistReject,
                                          15,
                                          mVCam,
                                          mVIms,
                                          0.85,
                                          mVSCams,
                                          mNbScale-1);
        }

        StdOut()<<"Selected Patches "<<mLPatchesI.size()<<std::endl;


        std::string NamePlyOut="./autocorrel_criterion_smaller_patches.ply";
        mTri->PlyWriteSelected(NamePlyOut,mLPatchesI,false);


        std::list<std::vector<int> >  aSetOfPatches ;
        cTplBoxOfPts<tREAL8,2> aBoxObj;
        for (const auto & aPatchIndex : mLPatchesI)
        {
            std::vector<cPt3dr> aVP;
            for (const auto anInd : aPatchIndex)
                aVP.push_back(ToR(mTri->KthPts(anInd)));

            std::vector<cData1ImLidPhgr> aVDenseData;
            EvalGeomConsistency(aVP,aVDenseData,mInitRes,true,mNbScale-1);
            if (EvalCorrel(aVDenseData)>mThresholdAcceptCorrel)
            {
                aSetOfPatches.push_back(aPatchIndex);

                for (auto & aPt: aVP)
                {
                    aBoxObj.Add(Proj(aPt));
                }
            }
        }
        mLPatchesI = aSetOfPatches;

        //StdOut()<<"Selected points to correl "<<mLPatchesI.size()<<"  "<<aSetOfPatches.size()<<std::endl;

        mBoxSelected=aBoxObj.CurBox();

        NamePlyOut="./autocorrel_criterion_intercorrel_smaller_patches.ply";
        mTri->PlyWriteSelected(NamePlyOut,mLPatchesI,false);

        StdOut() << "Patches: DistReject=" << aDistReject
                 << " NbPts=" << mTri->NbPts()
                 << " NbPatch=" << mLPatchesI.size()
                 << "\n";
    }
    isForSelection=false;
}



cBA_LidarPhotograRaster::cBA_LidarPhotograRaster(cPhotogrammetricProject * aPhProj,
                                                 cMMVII_BundleAdj& aBA,
                                                 const std::vector<std::string>& aParam) :
    cBA_LidarPhotogra(aPhProj, aBA, aParam)
{
    InitEq(true);

    //read scans files from directory corresponding to pattern in aParam.at(1)
    auto aVScanNames = mPhProj->GetStaticLidarNames(aParam.at(1));
    for (const auto & aNameSens : aVScanNames)
    {
        cStaticLidar * aScan = mBA.AddStaticLidar(aNameSens);
        StdOut() << "Add Scan " << aNameSens << "\n";
        mVScans.push_back({aNameSens, aScan, {}});
    }

    // Creation of the patches, choose a neigborhood around patch centers. TODO: adapt to images ground pixels size?
    if (mModeSim==eImatchCrit::eDifRad)
        mNbPointByPatch = 1;
    for (auto & aScanData: mVScans)
    {
        aScanData.mLidarRaster->MakePatches(aScanData.mLPatchesP,aBA.VSCPC(),mNbPointByPatch,5);
        StdOut() << "Nb patches for " << aScanData.mScanName << ": " << aScanData.mLPatchesP.size() << "\n";
    }
}

cBA_LidarPhotogra::~cBA_LidarPhotogra() 
{
    delete mInterp;
    for (auto & aVC : mVSCams)
    {
        DeleteAllAndClear(aVC);
    }

    DeleteAllAndClear(mVIms);
}

cBA_LidarPhotograTri::~cBA_LidarPhotograTri()
{
    if (mTri) delete mTri;
}

cBA_LidarPhotograRaster::~cBA_LidarPhotograRaster()
{
    //if (mLidarData) delete mLidarData; // automatically deleted at the end
}

void cBA_LidarPhotograTri::AddObs()
{
    mLastResidual.Reset();
    mNbUsedPoints = 0;
    mNbUsedObs = 0;
    mCurrentCorrelVal=0.0;
    mAverageDeltaX.Reset();
    mAverageDeltaY.Reset();

    if (mModeSim==eImatchCrit::eDifRad)
    {
        for (size_t aKP=0 ; aKP<mTri->NbPts() ; aKP++)
        {
            Add1Patch(mWFactor,{ToR(mTri->KthPts(aKP))},0);
        }
    }
    else
    {
        if  (0)
        {
            //initialize quality maps
            cPt2di aSz(Pt_round_up(mBoxSelected.Sz()/mInitRes)); // aStep=0.07 ( GSD 7cm)
            mDImQualityMap= new cDataIm2D<tREAL8> (cPt2di(0,0),aSz);
            mDImQualityMap->InitCste(-9999.0);

            mDImQualityMapY= new cDataIm2D<tREAL8> (cPt2di(0,0),aSz);
            mDImQualityMapY->InitCste(-9999.0);
        }


        // MMVII_UnclasseUsEr("Dont handle Census");
        int idd=0;
        int aNbSc= std::max(0,mNbScale-1-mBA.NbMaxIter());
        for (const auto& aPatchIndex : mLPatchesI)
        {
            mPatchId=idd;
            std::vector<cPt3dr> aVP;
            for (const auto anInd : aPatchIndex)
            {
                aVP.push_back(ToR(mTri->KthPts(anInd)));
            }
            // check multiscale correlation -> start from a lower resolution
            Add1PatchMulScale(mWFactor,aVP,aNbSc);
            idd++;
        }

        if(0)
        {
            if ( (mBA.NbMaxIter()==0) || (mBA.checkIfLastIter()) )
            {
                // save global quality map abs( Delta_X)
                std::string aNameQualityMap="./qualityMap_"+ToStr(mBA.NbMaxIter())+"_X"+".tif";
                tREAL8 aTransform[6]={mBoxSelected.P0().x(),0.07,0,mBoxSelected.P1().y(),0,-0.07};

                std::vector<const cDataIm2D<tREAL8>*> aVIms({mDImQualityMap});

                cDataFileIm2D aDF=cDataFileIm2D::Create(aNameQualityMap,
                                                          eTyNums::eTN_REAL8,
                                                          mDImQualityMap->Sz());
                cGdalApi::ReadWrite(cGdalApi::IoMode::Write,
                                    aVIms,
                                    aDF,
                                    cPt2di(0,0),
                                    1.0,
                                    cPixBox<2>(cPt2di(0,0),mDImQualityMap->Sz()),
                                    aTransform);




                // save global quality map abs ( Delta_Y)

                aNameQualityMap="./qualityMap_"+ToStr(mBA.NbMaxIter())+"_Y"+".tif";

                aVIms={mDImQualityMapY};
                aDF=cDataFileIm2D::Create(aNameQualityMap,
                                                          eTyNums::eTN_REAL8,
                                                          mDImQualityMapY->Sz());

                cGdalApi::ReadWrite(cGdalApi::IoMode::Write,
                                    aVIms,
                                    aDF,
                                    cPt2di(0,0),
                                    1.0,
                                    cPixBox<2>(cPt2di(0,0),mDImQualityMapY->Sz()),
                                    aTransform);
            }
        }
        delete mDImQualityMap;
        delete mDImQualityMapY;
    }

    if (mLastResidual.SW() != 0)
    {
        StdOut() << "  * Lid/Phr Residual Rad " << std::sqrt(mLastResidual.Average())
                 << " ("<<mNbUsedObs<<" obs, "<<mNbUsedPoints<<" points)\n";
        StdOut() <<" * Eval Correl between image patches "<<mCurrentCorrelVal/(tREAL8)mLPatchesI.size();
    }
    else
        StdOut() << "  * Lid/Phr: no obs\n";

    if (0)
    {
        if ( (mBA.NbMaxIter()==0) || (mBA.checkIfLastIter()))
        {
            StdOut() <<" ********************   Planar residual X  "<<mAverageDeltaX.Average()<<"\n";
            StdOut() <<" ********************   Planar residual Y  "<<mAverageDeltaY.Average()<<"\n";
        }
    }
}


void cBA_LidarPhotograRaster::AddObs()
{
    mLastResidual.Reset();
    mNbUsedPoints = 0;
    mNbUsedObs = 0;
    if (mModeSim==eImatchCrit::eDifRad)
    {
        for (auto & aScan : mVScans)
            for (const auto& aPatch : aScan.mLPatchesP)
            {
                Add1Patch(mWFactor,
                          {aScan.mLidarRaster->Image2Ground(*aPatch.begin())},
                          aScan.mScanName);
            }
    }
    else
    {
        for (auto & aScan : mVScans)
            for (const auto& aPatch : aScan.mLPatchesP)
            {
                std::vector<cPt3dr> aVP;
                for (const auto aPt : aPatch)
                    aVP.push_back(aScan.mLidarRaster->Image2Ground(aPt));
                Add1Patch(mWFactor,aVP,aScan.mScanName);
            }
    }

    if (mLastResidual.SW() != 0)
        StdOut() << "  * Lid/Phr Residual Rad " << std::sqrt(mLastResidual.Average())
                 << " ("<<mVScans.size()<<" scans, "<<mNbUsedObs<<" obs, "<<mNbUsedPoints<<" points)\n";
    else
        StdOut() << "  * Lid/Phr: no obs\n";
}


void cBA_LidarPhotograTri::SetVUkVObs
    (
     const cPt3dr&           aPGround,
     std::vector<int> *      aVIndUk,
     std::vector<tREAL8> &   aVObs,
     const cData1ImLidPhgr & aData,
     int                     aKPt
    )
{
    cSensorCamPC * aCam = mBA.VSCPC().at(aData.mKIm);  // extract the camera
    cPt3dr aPCam = aCam->Pt_W2L(aPGround);  // coordinate of point in image system
    tProjImAndGrad aPImGr = aCam->InternalCalib()->DiffGround2Im(aPCam); // compute proj & gradient

    // Vector of indexes of unknwons 
    if (aVIndUk)
    {
       aCam->PushIndexes(*aVIndUk);       // add the unknowns [C,R] of the camera
    }


    // vector that will contains values of observation at this step
    aCam->Pose_WU().PushObs(aVObs,true);  // true because we transpose: we use W->C, which is the transposition of IJK : C->W

    aPGround.PushInStdVector(aVObs);   //
    aPCam.PushInStdVector(aVObs);
            
    aPImGr.mGradI.PushInStdVector(aVObs);  // Grad Proj/PCam
    aPImGr.mGradJ.PushInStdVector(aVObs);
            
    auto [aRad0,aGradIm] = aData.mVGr.at(aKPt);  // Radiom & grad
    aVObs.push_back(aRad0);
    aGradIm.PushInStdVector(aVObs);
}


void cBA_LidarPhotograRaster::SetVUkVObs
    (
        const cPt3dr&           aPGround,
        std::vector<int> *      aVIndUk,
        std::vector<tREAL8> &   aVObs,
        const cData1ImLidPhgr & aData,
        int                     aKPt
        )
{
    cStaticLidar * aScan = mBA.MapTSL().at(aData.mScanAName);
    cPt3dr aPScan = aScan->Pt_W2L(aPGround);  // coordinate of point in ground system
    cSensorCamPC * aCam = mBA.VSCPC().at(aData.mKIm);  // extract the camera
    cPt3dr aPCam = aCam->Pt_W2L(aPGround);  // coordinate of point in image system
    tProjImAndGrad aPImGr = aCam->InternalCalib()->DiffGround2Im(aPCam); // compute proj & gradient

    // Vector of indexes of unknwons
    if (aVIndUk)
    {
        aScan->PushIndexes(*aVIndUk);      // add the unknowns [C,R] of the scan
        aCam->PushIndexes(*aVIndUk);       // add the unknowns [C,R] of the camera
    }

    // vector that will contain values of observation at this step
    aScan->Pose_WU().PushObs(aVObs,false); // no transpose for scan
    aCam->Pose_WU().PushObs(aVObs,true);  // true because we transpose: we use W->C, which is the transposition of IJK : C->W

    aPScan.PushInStdVector(aVObs);   //
    aPCam.PushInStdVector(aVObs);

    aPImGr.mGradI.PushInStdVector(aVObs);  // Grad Proj/PCam
    aPImGr.mGradJ.PushInStdVector(aVObs);

    auto [aRad0,aGradIm] = aData.mVGr.at(aKPt);  // Radiom & grad
    aVObs.push_back(aRad0);
    aGradIm.PushInStdVector(aVObs);
}


void cBA_LidarPhotogra::AddPatchDifRad
     (
           tREAL8 aWeight,
           const std::vector<cPt3dr> & aVPatchGr,
           const std::vector<cData1ImLidPhgr> &aVData
     )
{
     // read the solver now, because was not initialized at creation 
     cResolSysNonLinear<tREAL8> *  aSys = mBA.Sys();

     cWeightAv<tREAL8,tREAL8> aWAv;       // compute average of image for radiom unknown
     for (const auto & aData : aVData)
         aWAv.Add(1.0,aData.mVGr.at(0).first);

     cPt3dr    aPGround = aVPatchGr.at(0);
     std::vector<tREAL8> aVTmpAvg{aWAv.Average()};  // vector for initializingz the temporay (here 1 = average)
     cSetIORSNL_SameTmp<tREAL8>  aStrSubst(aVTmpAvg); // structure for handling schurr eliminatio,
     // parse the data of the patch
     for (const auto & aData : aVData)
     {
         std::vector<int>       aVIndUk{-1}; // first one is a temporary (convention < 0)
         std::vector<tREAL8>    aVObs;
         SetVUkVObs (aPGround,&aVIndUk,aVObs,aData,0);
            
         // accumulate the equation involving the radiom
         aSys->R_AddEq2Subst(aStrSubst,mEq,aVIndUk,aVObs,aWeight);
     }
     // do the substitution & add the equation reduced (Schurr complement)
     aSys->R_AddObsWithTmpUK(aStrSubst);
}

void cBA_LidarPhotogra::AddPatchCensus
     (
           tREAL8 aWeight,
           const std::vector<cPt3dr> & aVPatchGr,
           const std::vector<cData1ImLidPhgr> &aVData
     )
{
     // read the solver now, because was not initialized at creation 
     cResolSysNonLinear<tREAL8> *  aSys = mBA.Sys();
     for (size_t aKPt=1; aKPt<aVPatchGr.size() ; aKPt++)
     {
         // -------------- [1] Calculate the average ratio on all images --------------------
         cWeightAv<tREAL8,tREAL8> aAvRatio;  // stuct for averaging ratio
         for (const auto & aData : aVData)
         {
             tREAL8 aV0 = aData.mVGr.at(0).first;            // radiom of central pixel
             tREAL8 aVK = aData.mVGr.at(aKPt).first;         // radiom of neighbour
             aAvRatio.Add(1.0,NormalisedRatioPos(aV0,aVK)) ; // acumulate the ratio
         }
         std::vector<tREAL8> aVTmpAvg({aAvRatio.Average()});  // vector of value of temporary unknowns

         // -------------- [2] Add the observation --------------------
         cSetIORSNL_SameTmp<tREAL8>  aStrSubst(aVTmpAvg);  // structure for schur complement
         for (const auto & aData : aVData) // parse all the images
         {
             std::vector<int>  aVIndUk{-1} ;  // indexe of unknown
             std::vector<tREAL8>  aVObs;      // observation/context

             SetVUkVObs(aVPatchGr.at(0)  ,&aVIndUk,aVObs,aData,0);            // add unkown AND observations
             SetVUkVObs(aVPatchGr.at(aKPt),nullptr ,aVObs,aData,aKPt);        // add ONLY observations
             aSys->R_AddEq2Subst(aStrSubst,mEq,aVIndUk,aVObs,aWeight); // add the equation in Schurr structure
         }
         // add all the equation to the system with Schurr's elimination
         aSys->R_AddObsWithTmpUK(aStrSubst);
     }
}

void cBA_LidarPhotogra::AddPatchCorrel
     (
           tREAL8 aWeight,
           const std::vector<cPt3dr> & aVPatchGr,
           const std::vector<cData1ImLidPhgr> &aVData
     )
{
     // read the solver now, because was not initialized at creation 
     cResolSysNonLinear<tREAL8> *  aSys = mBA.Sys();
     // -------------- [1] Compute the normalized values --------------------
     size_t aNbPt = aVPatchGr.size();
     //  vector that will store the normalized value (Avg=0, Sigma=1)
     cDenseVect<tREAL8>  aVMoy(aNbPt,eModeInitImage::eMIA_Null);

     //  memorize the radiometries of images as vector
     std::vector<cDenseVect<tREAL8>>  aListVRad;
     for (const auto & aData : aVData)
     {
         // change to vecor format
         cDenseVect<tREAL8> aV(aNbPt);
         for (size_t aK=0 ; aK< aNbPt ; aK++)
         {
             aV(aK)  = aData.mVGr.at(aK).first;
         }
         aListVRad.push_back(aV);
         cDenseVect<tREAL8> aV01 = NormalizeMoyVar(aV);  // noramlize value
         aVMoy += aV01;  //  accumulate in a vector
     }

     aVMoy *=  1/ tREAL8(aVData.size()); // make VMoy, average of normalized
     aVMoy =  NormalizeMoyVar(aVMoy);  // re normalized  
       
     // -------------- [2] Intialize the temporary  --------------------

     /*  Say we have N points, M images,  tempory values will be stored "a la queue leu-leu" as :
               R1 .. RN  A0  B0 A1 B1 ... AM BM
             * where Ri are the unknown radiometry of the normalize patch
             * where Aj are the unkonw for tranfering radiom of image j to normalize patch such that

                 Ri =  Aj Imj(pij) + Bj

             Noting pij the projection of Pi in Imj
     */

     std::vector<tREAL8> aVTmp = aVMoy.ToStdVect(); // push first values of normalized patch
     int aK0Im = aVTmp.size();

     // push the initial values of Aj Bj
     for (const auto &  aVRad : aListVRad)
     {
         auto [A,B] =  LstSq_Fit_AxPBEqY(aVRad,aVMoy);  // solve  Ri = Aj Imj + Bj
         aVTmp.push_back(A); // add tmp unknown for Aj
         aVTmp.push_back(B); // add tmp unknown for Bj
     }
     cSetIORSNL_SameTmp<tREAL8>  aStrSubst(aVTmp); // structure for handling schurr eliminatio,

             // three structure for forcing conservation of normalizattion (Avg,Sigma) for VMoy
     std::vector<int> aVIndPt;       // indexe of unkown of norm radiom
     std::vector<tREAL8> aVFixAvg;   // vector for forcing average
     std::vector<tREAL8> aVFixVar;   // vector for forcing std dev

     // -------------- [3] Add the equation  --------------------


     for (int aKPt=0 ; aKPt <  (int) aNbPt ; aKPt++)  // parse all points
     {
         int aIndPt = -(1+aKPt);     // indexe of point are {-1,-2,....}
         aVIndPt.push_back(aIndPt);  // accumulat set of global indexe of unknown patch
         aVFixAvg.push_back(1.0);     //  Sum Rk = 0 => all weight = 1
         //  S(R+dR) ^ 2 =1   ;  S (2 R dR ) = 1 - S(R^2)  ; but S(R^2)=1 by construction ...
         aVFixVar.push_back(2*aVMoy(aKPt));

         for (int aKIm=0 ;  aKIm< (int) aVData.size() ; aKIm++)
         {
             int aIndIm = -(1+aK0Im+2*aKIm);  // compute indexe assumming "a la queue leu-leu"
             std::vector<int>       aVIndUk{aIndPt,aIndIm,aIndIm-1} ;  // indexes of 3 unknown
             std::vector<tREAL8>    aVObs;  // vector of observations 
             SetVUkVObs (aVPatchGr.at(aKPt),&aVIndUk,aVObs,aVData.at(aKIm),aKPt);  // read obs & global Uk
             aSys->R_AddEq2Subst(aStrSubst,mEq,aVIndUk,aVObs,aWeight);  // add equation in tmp struct
         }
     }

     aStrSubst.AddOneLinearObs(aNbPt,aVIndPt,aVFixAvg,0.0);  // force average
     aStrSubst.AddOneLinearObs(aNbPt,aVIndPt,aVFixVar,0.0);  // force standard dev

     aSys->R_AddObsWithTmpUK(aStrSubst);
}

void cBA_LidarPhotogra::EvaluatePlanarDisplacements(std::vector<std::string> & aVecOrthoNames,
                                                    std::vector<tREAL8*> & aVecTransforms,
                                                    bool isStandalone=false)
{
    //std::list<cParamCallSys> aMasqsDefinition;
    int aMin=0;
    int aMax=256;
    for(size_t aKIM=0; aKIM<aVecOrthoNames.size();aKIM++)
    {
        std::string aImName=aVecOrthoNames[aKIM];
        cParamCallSys aCom(
            cMMVII_Appli::MMV1Bin(),
            "MasqMaker",
            aImName,
            ToStr(aMin),
            ToStr(aMax)
            //"@ExitOnBrkp"
            );
        aCom.Execute(true);
        //aMasqsDefinition.push_back(aCom);

        //StdOut()<<aCom.Com()<<std::endl;
    }
    //mBA.getPhProj()->Appli().ExeComSerial(aMasqsDefinition,false);


    if (isStandalone)
    {
        // use code for optical flow calculation

        for(size_t aKMaster=0; aKMaster<aVecOrthoNames.size();aKMaster++)
        {
            for (size_t aKSec=aKMaster+1;aKSec<aVecOrthoNames.size();aKSec++)
            {

                // ortho1 ortho2 masq1
                std::string ortho1 = aVecOrthoNames[aKMaster];
                std::string ortho2 = aVecOrthoNames[aKSec];
                std::string  masq1 = LastPrefix(aVecOrthoNames[aKMaster])+"_Masq.tif";
                cOpticalFlow<tREAL8> anOpt(ortho1,
                                           ortho2,
                                           masq1);
                std::string aNameXY=LastPrefix(aVecOrthoNames[aKMaster])+LastPrefix(aVecOrthoNames[aKSec]);
                anOpt.SolveDispDirect(aNameXY,
                                      aVecTransforms[aKMaster]);
            }
        }
    }
    else
    {

        //std::list<cParamCallSys> aComsDisplacements;
        for(size_t aKMaster=0; aKMaster<aVecOrthoNames.size();aKMaster++)
        {
            for (size_t aKSec=aKMaster+1;aKSec<aVecOrthoNames.size();aKSec++)
            {
                cParamCallSys aCom(
                    cMMVII_Appli::MMV1Bin(),
                    "MM2DPosSism",
                    aVecOrthoNames[aKMaster],
                    aVecOrthoNames[aKSec],
                    "Masq="+LastPrefix(aVecOrthoNames[aKMaster])+"_Masq.tif",
                    "DirMEC=MECDISP_"+LastPrefix(aVecOrthoNames[aKMaster])+"_"+LastPrefix(aVecOrthoNames[aKSec])+"/",
                    "SzW=3"
                    //"@ExitOnBrkp"
                    );
                aCom.Execute(true);
                //aComsDisplacements.push_back(aCom);
            }
        }

        // Execute displacements estimation
        //mBA.getPhProj()->Appli().ExeComSerial(aComsDisplacements,false);
        // read displacement maps and compute average planar displacements over all patches


        tREAL8 aDeltaX=0.0;
        tREAL8 aDeltaY=0.0;

        int aCountX=0;
        int aCountY=0;

        cPt2di aSz=cIm2D<tREAL8>::FromFile(aVecOrthoNames[0]).DIm().Sz();
        cDataIm2D<tREAL8> aMaxDelatX= cDataIm2D<tREAL8>(cPt2di(0,0),aSz);
        cDataIm2D<tREAL8> aMaxDelatY= cDataIm2D<tREAL8>(cPt2di(0,0),aSz);

        aMaxDelatX.InitCste(-9999.0);
        aMaxDelatY.InitCste(-9999.0);
        tREAL8 aVx,aVy;

        for(size_t aKMaster=0; aKMaster<aVecOrthoNames.size();aKMaster++)
        {
            for (size_t aKSec=aKMaster+1;aKSec<aVecOrthoNames.size();aKSec++)
            {
                std::string aPax1="MECDISP_"+LastPrefix(aVecOrthoNames[aKMaster])+"_"+
                                    LastPrefix(aVecOrthoNames[aKSec])+
                                    "/"+"Px1_Num6_DeZoom1_LeChantier.tif";
                std::string aPax2="MECDISP_"+LastPrefix(aVecOrthoNames[aKMaster])+"_"+
                                    LastPrefix(aVecOrthoNames[aKSec])+
                                    "/"+"Px2_Num6_DeZoom1_LeChantier.tif";
                std::string aMasqDefined=LastPrefix(aVecOrthoNames[aKMaster])+"_Masq.tif";

                // read images

                cIm2D<tREAL8> aImPax1=cIm2D<tREAL8>::FromFile(aPax1);
                cDataIm2D<tREAL8> & aDPax1 = aImPax1.DIm();


                cIm2D<tREAL8> aImPax2=cIm2D<tREAL8>::FromFile(aPax2);
                cDataIm2D<tREAL8> & aDPax2 = aImPax2.DIm();

                cIm2D<tU_INT1> aImMasq=cIm2D<tU_INT1>::FromFile(aMasqDefined);
                cDataIm2D<tU_INT1> & aDMasq = aImMasq.DIm();

                cPt2di aPix;
                for (aPix.x()=0; aPix.x()<aDPax1.SzX();aPix.x()++)
                {
                    for (aPix.y()=0; aPix.y()<aDPax1.SzY();aPix.y()++)
                    {
                        // check if masq is defined
                        if (aDMasq.GetV(aPix))
                        {
                            aVx=aDPax1.GetV(aPix);
                            aVy=aDPax2.GetV(aPix);
                            if (aVx!=0.0)
                            {

                                aDeltaX+=abs(aVx);
                                aCountX++;

                            }
                            if (abs(aVx)>aMaxDelatX.GetV(aPix))
                                aMaxDelatX.SetV(aPix,abs(aVx));

                            if (aVy!=0.0)
                            {
                                aDeltaY+=abs(aVy);
                                aCountY++;
                            }
                            if (abs(aVy)>aMaxDelatY.GetV(aPix))
                                aMaxDelatY.SetV(aPix,abs(aVy));
                        }
                    }
                }
            }
        }


        // save patch error map


        cPt2dr anOffsetGlobal((aVecTransforms[0][0]-mBoxSelected.P0().x())/aVecTransforms[0][1],
                                (aVecTransforms[0][3]-mBoxSelected.P1().y())/aVecTransforms[0][5]);


        cPt2di aPix;
        for (aPix.x()=0; aPix.x()<aMaxDelatX.SzX();aPix.x()++)
        {
            for (aPix.y()=0; aPix.y()<aMaxDelatX.SzY();aPix.y()++)
            {
                //StdOut()<<" kkk  "<<mDImQualityMap->Sz()<<"   "<<ToI(anOffsetGlobal)+aPix<<std::endl;
                // fill global map
                mDImQualityMap->SetV(ToI(anOffsetGlobal)+aPix, aMaxDelatX.GetV(aPix));
                mDImQualityMapY->SetV(ToI(anOffsetGlobal)+aPix, aMaxDelatY.GetV(aPix));
            }
        }
        // normalize
        aDeltaX/=(aCountX+1e-8);
        aDeltaY/=(aCountY+1e-8);
        mAverageDeltaX.Add(1.0,aDeltaX);
        mAverageDeltaY.Add(1.0,aDeltaY);
        StdOut()<<" aMean Displacement "<<aDeltaX<<"   "<<aDeltaY<<std::endl;
    }
}


void cBA_LidarPhotogra::EvalGeomConsistency(const std::vector<cPt3dr>& aVPatchGr,
                                            std::vector<cData1ImLidPhgr>& aVData,
                                            tREAL8 aPas,
                                            bool sparse,
                                            int aNbs)
{

    if (! sparse)
    {
        //aPas = aPas * pow (2,aNbs);
        //  Parse all the image, we will select the images where all point of a patch are visible
        std::vector<cPt2dr> aVPatchGr2D;
        std::vector <cPt3di> aFaces;
        cTriangulation3D aTri3D=cTriangulation3D<tREAL8>(aVPatchGr,aFaces);
        cBox2dr aBox=aTri3D.Box2D();

        // Empty grid of points to sample with size bbox/aStep
        cIm2D<tREAL8> aD_Grid=cIm2D<tREAL8>(cPt2di(Pt_round_up(aBox.Sz()/aPas)),
                                        nullptr,
                                        eModeInitImage::eMIA_V1);

        // Empty grid to store radiometry projections
        cIm2D<tREAL8> aD_GridRadiom=cIm2D<tREAL8>(cPt2di(Pt_round_up(aBox.Sz()/aPas)),
                                              nullptr,
                                              eModeInitImage::eMIA_V1);


        cDataIm2D<tREAL8> * aDD_Grid=& (aD_Grid.DIm());

        cDataIm2D<tREAL8> * aDD_GridRadiom=& (aD_GridRadiom.DIm());

        //std::cout<<"ASZ  "<<aD_Grid.Sz()<<std::endl;
        aDD_Grid->InitCste(-9999);
        aDD_GridRadiom->InitCste(0.0);

        // 2D point in local grid coordinate system
        for (const auto & aPt: aVPatchGr)
        {
            cPt2dr aPDr;
            cPt2dr aPt2d;
            aPt2d=Proj(aPt);
            // convention in geographic projected coordinate sytem x left --> right , y down --> up
            aPDr.x()=(aPt2d.x()-aBox.P0().x())/aPas;
            aPDr.y()=(-aPt2d.y()+aBox.P1().y())/aPas;
            aVPatchGr2D.push_back(aPDr);
        }

        cTriangulation2D<tREAL8> aTriangul(aVPatchGr2D);
        //StdOut()<<"make tri"<<std::endl;
        aTriangul.MakeDelaunay();
        //StdOut()<<"make tri done"<<std::endl;

        // Interpolate Lidar Patch intensity
        for (size_t aKTri=0 ; aKTri<aTriangul.NbFace() ; aKTri++)
        {
            cTriangle2DCompiled aTriangle=cTriangle2DCompiled(aTriangul.KthTri(aKTri));
            cPt3dr aPPx;
            // z values of the triangle
            aPPx[0]=aVPatchGr[aTriangul.VFaces()[aKTri].x()].z();
            aPPx[1]=aVPatchGr[aTriangul.VFaces()[aKTri].y()].z();
            aPPx[2]=aVPatchGr[aTriangul.VFaces()[aKTri].z()].z();

            //get all pixeles inside triangle
            static std::vector<cPt2di> aVPixTri;
            aTriangle.PixelsInside(aVPixTri);
            for (const auto & aPix : aVPixTri)
            {
                //std::cout<<"APX  "<<aPix<<std::endl;
                if(aDD_Grid->Inside(aPix))
                {
                    aDD_Grid->VD_SetV(aPix,aTriangle.ValueInterpol(ToR(aPix),aPPx));
                }
            }
        }
        long value_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::time_point_cast<std::chrono::milliseconds>(
                                std::chrono::high_resolution_clock::now()).time_since_epoch()
                            ).count();

        // Reproject image radiometry (ortho) and compute correl in image or ground geometry
        std::vector<std::string> OrthosName;
        std::vector<tREAL8 *> aVecTransforms;
        for (size_t aKIm=0 ; aKIm<mVCam.size() ; aKIm++)
        {
            cSensorCamPC * aCam =  (aNbs==0) ? mVCam[aKIm] :  mVSCams[aKIm][aNbs-1]; // extract cam
            /*cDataIm2D<tU_INT1>*/ auto & aDIm = mVIms[aKIm];

            if (aCam->IsVisible(aVPatchGr.at(0))) // first test : is central point visible
            {
                aDD_GridRadiom->InitCste(0.0);
                cData1ImLidPhgr  aData; // data that will be filled
                aData.mKIm = aKIm;
                cPt2di aPix;
                for (aPix.x()=0;aPix.x()<aDD_Grid->SzX();aPix.x()++)
                {
                    for (aPix.y()=0;aPix.y()<aDD_Grid->SzY();aPix.y()++)
                    {
                        if (aDD_Grid->VD_GetV(aPix)!=-9999)
                        {
                            cPt3dr aP3D;
                            aP3D.x()=aBox.P0().x()+aPix.x()*aPas;
                            aP3D.y()=aBox.P1().y()-aPix.y()*aPas;
                            aP3D.z()=aDD_Grid->VD_GetV(aPix);
                            if (aCam->IsVisible(aP3D))
                            {
                                cPt2dr aPIm = aCam->Ground2Image(aP3D); // extract the image  projection
                                // unscale point
                                aPIm = MulCByC(aPIm,cPt2dr(pow (2,aNbs),pow (2,aNbs)));
                                if (aDIm->InsideInterpolator(*mInterp,aPIm,1.0))  // is it sufficiently inside
                                {
                                    auto aVGr = aDIm->GetValueAndGradInterpol(*mInterp,aPIm); // extract pair Value/Grad of image
                                    aDD_GridRadiom->VD_SetV(aPix,aVGr.first);
                                    aData.mVGr.push_back(aVGr); // push it at end of stack
                                }
                            }
                        }
                    }
                }
                aVData.push_back(aData); // memorize the data for this image
                // save ortho
                // save orthos to check registration accuracy
                if(0)
                {
                    if ( ( (mBA.NbMaxIter()==0) || (mBA.checkIfLastIter()) )
                        && (isForSelection))
                    {
                        std::string aName="gridRadiom"+
                                            std::to_string(value_ms)+
                                            "_"+
                                            ToStr(aKIm)+
                                            "_"+
                                            ToStr(mPatchId)+
                                            ".tif";
                        OrthosName.push_back(aName);
                        cDataFileIm2D aDF=cDataFileIm2D::Create(aName,
                                                                  eTyNums::eTN_REAL8,
                                                                  aDD_Grid->Sz());

                        // add geotiff transform
                        tREAL8 transform[6]={aBox.P0().x(),aPas,0,aBox.P1().y(),0,-aPas};
                        std::vector<const cDataIm2D<tREAL8>*> aVIms({aDD_GridRadiom});

                        cGdalApi::ReadWrite(cGdalApi::IoMode::Write,
                                            aVIms,
                                            aDF,
                                            cPt2di(0,0),
                                            1.0,
                                            cPixBox<2>(cPt2di(0,0),aDD_Grid->Sz()),
                                            transform);
                        aVecTransforms.push_back(transform);
                    }
                }
            }
        }

        // perfom displacement measurment between patches
        if(0)
        {
            if ( ( (mBA.NbMaxIter()==0) || (mBA.checkIfLastIter()) )
                && (!isForSelection) && (OrthosName.size()>1) )
            {
                EvaluatePlanarDisplacements(OrthosName,aVecTransforms,false);
            }
        }
   }

    else
    {
       for (size_t aKIm=0 ; aKIm<mVCam.size() ; aKIm++)
       {
           cSensorCamPC * aCam =  (aNbs==0) ? mVCam[aKIm] :  mVSCams[aKIm][aNbs-1]; // extract cam
           /*cDataIm2D<tU_INT1>*/ auto  & aGenDIm = mVIms[aKIm];
            //auto & aGenDIm = aCam->LoadImage();

           if (aCam->IsVisible(aVPatchGr.at(0))) // first test : is central point visible
           {
               cData1ImLidPhgr  aData; // data that will be filled
               aData.mKIm = aKIm;
               for (size_t aKPt=0 ; aKPt<aVPatchGr.size() ; aKPt++) // parse the points of the patch
               {
                   cPt3dr aPGround = aVPatchGr.at(aKPt);
                   if (aCam->IsVisible(aPGround))  // is the point visible in the camera
                   {
                       cPt2dr aPIm = aCam->Ground2Image(aPGround); // extract the image  projection
                       if (aGenDIm->InsideInterpolator(*mInterp,aPIm,1.0))  // is it sufficiently inside
                       {
                           auto aVGr = aGenDIm->GetValueAndGradInterpol(*mInterp,aPIm); // extract pair Value/Grad of image
                           aData.mVGr.push_back(aVGr); // push it at end of stack
                       }
                   }
               }
               //  Does all the point of the patch were inside the image ?
               if (aData.mVGr.size() == aVPatchGr.size())
               {
                   aVData.push_back(aData); // memorize the data for this image
               }
           }
       }

    }
}


tREAL8 cBA_LidarPhotogra::EvalCorrel(const std::vector<cData1ImLidPhgr> & aVData)
{
    if  ( !(aVData.size()>1) ) // case when only one image is seen by patch
        return -1;
    auto aDataMaster = aVData.at(0);
    size_t aNbPt=aVData.at(0).mVGr.size();
    if (aNbPt < 5)
        return -1.0;
    cDenseVect<tREAL8> aVMaster(aNbPt);
    for (size_t aK=0 ; aK< aNbPt ; aK++)
    {
        aVMaster(aK)  = aDataMaster.mVGr.at(aK).first;
    }
    //aVMaster=NormalizeMoyVar(aVMaster);
    tREAL8 aMeanCorrel=0.0;
    for (size_t aInd=1; aInd<aVData.size();aInd++)
    {
        size_t aNbPtSec=aVData.at(aInd).mVGr.size();
        if (aNbPtSec!=aNbPt)
            continue;
        cDenseVect<tREAL8> aSecVec(aNbPt);
        for (size_t aK=0 ; aK< aNbPt ; aK++)
        {
            aSecVec(aK)  = aVData[aInd].mVGr.at(aK).first;
        }

        //aSecVec=NormalizeMoyVar(aSecVec);

        // compute correl
        cMatIner2Var<tREAL8>  aMat;
        for (size_t aK=0; aK<aNbPt; aK++)
        {
            aMat.Add(aVMaster(aK),aSecVec(aK));
        }
        //StdOut()<<"Current correl "<<aCorrel<<std::endl;
        aMeanCorrel+=aMat.Correl();
        //MMVII_INTERNAL_ASSERT_strong(aCorrel<=1.0 && aCorrel>=-1.0,"Correl not correctly measured !");
    }

    aMeanCorrel /=(aVData.size()-1);
    return aMeanCorrel;
}

void  cBA_LidarPhotogra::Add1Patch(tREAL8 aWeight,const std::vector<cPt3dr> & aVPatchGr, const std::string & aScanName)
{
     std::vector<cData1ImLidPhgr> aVData; // for each image where patch is visible will store the data
     cComputeStdDev<tREAL8>   aStdDev;    // compute the standard deviation of projected radiometry (indicator) 

     //  Parse all the image, we will select the images where all point of a patch are visible
     for (size_t aKIm=0 ; aKIm<mBA.VSCPC().size() ; aKIm++)
     {
          cSensorCamPC * aCam = mBA.VSCPC()[aKIm]; // extract cam
          auto & aGenDIm = aCam->LoadImage();
          if (aCam->IsVisible(aVPatchGr.at(0))) // first test : is central point visible
          {
              cData1ImLidPhgr  aData; // data that will be filled
              aData.mScanAName = aScanName;
              aData.mKIm = aKIm;
              for (size_t aKPt=0 ; aKPt<aVPatchGr.size() ; aKPt++) // parse the points of the patch
              {
                   cPt3dr aPGround = aVPatchGr.at(aKPt);
                   if (aCam->IsVisible(aPGround))  // is the point visible in the camera
                   {
                        cPt2dr aPIm = mBA.VSCPC()[aKIm]->Ground2Image(aPGround); // extract the image  projection
                        if (aGenDIm.InsideInterpolator(*mInterp,aPIm,1.0))  // is it sufficiently inside
                        {
                            auto aVGr = aGenDIm.GetValueAndGradInterpol(*mInterp,aPIm); // extract pair Value/Grad of image
                            aData.mVGr.push_back(aVGr); // push it at end of stack
                        }
                   }
              }
              //  Does all the point of the patch were inside the image ?
              if (aData.mVGr.size() == aVPatchGr.size())
              {
                  aVData.push_back(aData); // memorize the data for this image

                  tREAL8 aValIm = aData.mVGr.at(0).first;   // value of first/central pixel in this image
                  //StdOut()<<aCam->NameImage()<<"    aValIm  "<<aValIm<<"  aRatio "<<aData.mVGr.at(1).first/aValIm<< "\n";
                  // aWAv.Add(1.0,aValIm);     // compute average
                  aStdDev.Add(1.0,aValIm);  // compute std deviation
              }
          }
     }

     // if less than 2 images : nothing valuable to do
     if (aVData.size()<2) return;

//#define NUMPATCHDEBUG 0
#ifdef NUMPATCHDEBUG
     // debug patch
     int aPixSz = 15;
     int aSpaceSz = 1;
     if (mNbUsedPoints==NUMPATCHDEBUG)
     {
         for (const auto & aData : aVData)
         {
            tREAL4 aW = sqrt(aData.mVGr.size());
            cRGBImage  aImDist8b(cPt2di(aW, aW)*(aPixSz+aSpaceSz)+cPt2di(aSpaceSz,aSpaceSz), cRGBImage::Gray128);
            int aI = 0;
            int aJ = 0;
            // make a vect of gray in correct order
            std::vector<tREAL4> aVGrOrdered(aData.mVGr.size());
            for (size_t i=0; i<aData.mVGr.size() ; ++i)
            {
                if (i==0)
                    aVGrOrdered[aData.mVGr.size()/2] = aData.mVGr[0].first; //center
                else if (i<=aData.mVGr.size()/2)
                    aVGrOrdered[i-1] = aData.mVGr[i].first;
                else
                    aVGrOrdered[i] = aData.mVGr[i].first;
            }
            for (const auto & aV : aVGrOrdered)
            {
                aImDist8b.FillRectangle(cPt3di(aV,aV,aV),
                                        cPt2di(aI, aJ)*(aPixSz+aSpaceSz)+cPt2di(aSpaceSz,aSpaceSz),
                                        cPt2di(aI+1, aJ+1)*(aPixSz+aSpaceSz),
                                        cPt3dr(0.,0.,0.));
                //aImDist8b.RawSetPoint(cPt2di(aI, aJ)*(aPixSz+aSpaceSz), aV, aV, aV);
                aI++;
                if (aI==aW)
                {
                    aJ++;
                    aI = 0;
                }
            }
            std::string aPath = mPhProj->DirVisuAppli() + "iter" + ToStr(mBA.NbIter(),1) + "_" + mBA.VSCPC()[aData.mKIm]->NameImage() + "_patch.png";
            aImDist8b.
                (aPath);
         }
     }
#endif

     mNbUsedPoints++;
     mNbUsedObs+=aVData.size();

     // accumlulate for computing average of deviation
     // mLastResidual.Add(1.0,  (aStdDev.StdDev(1e-5) *aVData.size()) / (aVData.size()-1.0));
     // mLastResidual.Add(1.0,  (aStdDev.StdDev(1e-5) ) );
     mLastResidual.Add(aVData.size(),  Square(aStdDev.StdDev(1e-5) ) );

     if (mModeSim==eImatchCrit::eDifRad)
     {
        AddPatchDifRad(aWeight,aVPatchGr,aVData);
     }
     else if (mModeSim==eImatchCrit::eCensus)
     {
        AddPatchCensus(aWeight,aVPatchGr,aVData);
     }
     else if (mModeSim==eImatchCrit::eCorrel)
     {
         std::vector<cData1ImLidPhgr> aVDenseData;
         EvalGeomConsistency(aVPatchGr,aVDenseData,mInitRes, true);
         mCurrentCorrelVal+=EvalCorrel(aVDenseData);
         AddPatchCorrel(aWeight,aVPatchGr,aVData);
     }
}


void cBA_LidarPhotogra::Add1PatchMulScale(tREAL8 aWeight,const std::vector<cPt3dr> & aVPatchGr, int aNbs)
{
    std::vector<cData1ImLidPhgr> aVData; // for each image where patch is visible will store the data
    cComputeStdDev<tREAL8>   aStdDev;    // compute the standard deviation of projected radiometry (indicator)
    //  Parse all the image, we will select the images where all point of a patch are visible
    for (size_t aKIm=0 ; aKIm<mVCam.size() ; aKIm++)
    {
        cSensorCamPC * aCam = (aNbs==0) ? mVCam[aKIm] : mVSCams[aKIm][aNbs-1]; // extract cam
        /*cDataIm2D<tU_INT1>*/ auto & aDIm = mVIms[aKIm]; // extract image
        if (aCam->IsVisible(aVPatchGr.at(0))) // first test : is central point visible
        {
            cData1ImLidPhgr  aData; // data that will be filled
            aData.mKIm = aKIm;
            for (size_t aKPt=0 ; aKPt<aVPatchGr.size() ; aKPt++) // parse the points of the patch
            {
                cPt3dr aPGround = aVPatchGr.at(aKPt);
                if (aCam->IsVisible(aPGround))  // is the point visible in the camera
                {
                    cPt2dr aPIm = MulCByC(mVCam[aKIm]->Ground2Image(aPGround),
                                          cPt2dr(pow(2,aNbs), pow(2,aNbs))); // extract the image  projection

                    if (aDIm->InsideInterpolator(*mInterp,aPIm,1.0))  // is it sufficiently inside
                    {
                        auto aVGr = aDIm->GetValueAndGradInterpol(*mInterp,aPIm); // extract pair Value/Grad of image
                        aData.mVGr.push_back(aVGr); // push it at end of stack
                    }
                }
            }
            //  Does all the point of the patch were inside the image ?
            if (aData.mVGr.size() == aVPatchGr.size())
            {
                aVData.push_back(aData); // memorize the data for this image
                tREAL8 aValIm = aData.mVGr.at(0).first;   // value of first/central pixel in this image
                aStdDev.Add(1.0,aValIm);  // compute std deviation
            }
        }
    }
    // if less than 2 images : nothing valuable to do
    if (aVData.size()<2) return;

    mNbUsedPoints++;
    mNbUsedObs+=aVData.size();


    mLastResidual.Add(aVData.size(),  Square(aStdDev.StdDev(1e-5) ) );

    if (mModeSim==eImatchCrit::eDifRad)
    {
        AddPatchDifRad(aWeight,aVPatchGr,aVData);
    }
    else if (mModeSim==eImatchCrit::eCensus)
    {
        AddPatchCensus(aWeight,aVPatchGr,aVData);
    }
    else if (mModeSim==eImatchCrit::eCorrel)
    {
        std::vector<cData1ImLidPhgr> aVDenseData;
        EvalGeomConsistency(aVPatchGr,aVDenseData,mInitRes,true, aNbs);
        mCurrentCorrelVal+=EvalCorrel(aVDenseData);
        AddPatchCorrel(aWeight,aVPatchGr,aVData);
    }
}



cBA_LidarLidarRaster::cBA_LidarLidarRaster(cPhotogrammetricProject * aPhProj,
                                           cMMVII_BundleAdj& aBA, const std::vector<std::string>& aParam) :
    cBA_LidarBase(aPhProj, aBA, aParam)
{
    mEq = EqEqLidarLidar (true,1);
    std::vector<std::string> aParamBis = aParam;
    // if interpolator is empty, force linear
    if (aParamBis.size() < 5)
        aParamBis.resize(5);
    if (aParamBis.at(2).empty())
    {
        aParamBis[2] = "1."; // default threshold init
    }
    if (aParamBis.at(3).empty())
    {
        aParamBis[3] = "0.1"; // default threshold final
    }
    if (aParamBis.at(4).empty())
    {
        aParamBis[4] = "[Linear]";
    }
    init(aParamBis, 1, 4);

    mThresholdInit = cStrIO<double>::FromStr(aParamBis[2]);
    if (mThresholdInit<0)
        mThresholdInit = INFINITY;
    mThresholdFinal = cStrIO<double>::FromStr(aParamBis[3]);
    if (mThresholdFinal<0)
        mThresholdFinal = INFINITY;

    //read scans files from directory corresponding to pattern in aParam.at(1)
    auto aVScanNames = mPhProj->GetStaticLidarNames(aParam.at(0));
    for (const auto & aNameSens : aVScanNames)
    {
        cStaticLidar * aScan = mBA.AddStaticLidar(aNameSens);
        StdOut() << "Add Scan " << aNameSens << "\n";
        mVScans.push_back({aNameSens, aScan, {}});
    }

    // Creation of the patches, here juste center point
    for (auto & aScanData: mVScans)
    {
        aScanData.mLidarRaster->MakePatches(aScanData.mLPatchesP,aBA.VSCPC(),1,5);
        StdOut() << "Nb patches for " << aScanData.mScanName << ": " << aScanData.mLPatchesP.size() << "\n";

        //for (auto &aTestRasterPoint: {cPt2di(10672,2238), cPt2di(2552,2121) })
        //    StdOut() << "Test " << aScanData.mScanName << " " << aTestRasterPoint << ": "
        //             << aScanData.mLidarRaster->Image2Ground(aTestRasterPoint) <<"\n";
    }
}

void cBA_LidarLidarRaster::UpdateWeightersMap()
{
    tREAL4 aTh = mThresholdFinal + ((mThresholdInit - mThresholdFinal)*(mBA.NbMaxIter()-mBA.Iter()))/mBA.NbMaxIter();
    std::cout << "up weighters, th="<<aTh<<"\n";
    if (aTh>10000)
        aTh = -1;
    for (auto & aScanDataA: mVScans)
    {
        for (auto & aScanDataB: mVScans)
        {
            tREAL8 aSigmaAB = sqrt(aScanDataA.mLidarRaster->Sigma()*aScanDataA.mLidarRaster->Sigma()
                                   +aScanDataB.mLidarRaster->Sigma()*aScanDataB.mLidarRaster->Sigma());
            mWeightersMap[aScanDataA.mScanName+"-"+aScanDataB.mScanName]
                = cStdWeighterResidual(sqrt(mWFactor)*aSigmaAB, aTh / 20., aTh, 1);
        }
    }
}

cBA_LidarLidarRaster::~cBA_LidarLidarRaster()
{
}


#define SCANSCANDEBUG 10

void cBA_LidarLidarRaster::AddObs()
{
    mLastResidual.Reset();
    mNbUsedPoints = 0;
    mNbUsedObs = 0;

    // update the weighters map
    UpdateWeightersMap();

    for (auto & aScan : mVScans)
    {
#ifdef SCANSCANDEBUG
        cIm2D<tREAL4> aResImage(aScan.mLidarRaster->InternalCalib()->SzPix(),0,eModeInitImage::eMIA_Null);
        auto & aResImageData = aResImage.DIm();
        int aPtSize = 1 + aScan.mLidarRaster->InternalCalib()->SzPix().x()/1000;
        if (mBA.Iter()%SCANSCANDEBUG==0)
        {
            for (int y=0; y<aScan.mLidarRaster->InternalCalib()->SzPix().y();++y)
                for (int x=0; x<aScan.mLidarRaster->InternalCalib()->SzPix().x();++x)
                    aResImageData.SetV({x,y}, 999);
        }
#endif

        for (const auto& aPatch : aScan.mLPatchesP)
        {
            //if (*aPatch.begin()==cPt2di(4278, 2245)) //10677, 2481
            //    std::cout<<"!\n";
            [[maybe_unused]] auto aMinRes = Add1Patch(aScan.mLidarRaster->Image2Ground(*aPatch.begin()),
                                                      aScan.mScanName);
#ifdef SCANSCANDEBUG
            if (mBA.Iter()%SCANSCANDEBUG==0)
            {
                auto aC = *aPatch.begin();
                for (int y=aC.y()-aPtSize; y<=aC.y()+aPtSize;++y)
                    for (int x=aC.x()-aPtSize; x<=aC.x()+aPtSize;++x)
                        aResImageData.SetVTruncIfInside({x,y}, aMinRes);
            }
#endif
        }

#ifdef SCANSCANDEBUG
        if (mBA.Iter()%SCANSCANDEBUG==0)
        {
            std::string aPath = mPhProj->DirVisuAppli() + aScan.mScanName + "_iter_" + ToStr(mBA.Iter())+ ".tif";
            aResImageData.ToFile(aPath, {"COMPRESS=DEFLATE"});
        }
#endif

    }
    if (mLastResidual.SW() != 0)
        StdOut() << "  * Lid/Lid Residual dist " << std::sqrt(mLastResidual.Average())
                 << "m ("<<mVScans.size()<<" scans, "<<mNbUsedObs<<" obs, "<<mNbUsedPoints<<" points)\n";
    else
        StdOut() << "  * Lid/Lid: no obs\n";
}


void cBA_LidarLidarRaster::SetVUkVObs
    (const cPt3dr&           aPGround,
     std::vector<int> *      aVIndUk,
     std::vector<tREAL8> &   aVObs,
     const cData1ImLidPhgr & aData,
     int                     aKPt
     )
{
    cStaticLidar * aScanA = mBA.MapTSL().at(aData.mScanAName);
    cPt3dr aPScanA = aScanA->Pt_W2L(aPGround);  // coordinate of point in ground system
    cStaticLidar * aScanB = mBA.MapTSL().at(aData.mScanBName);
    cPt3dr aPScanB0 = aScanB->Pt_W2L(aPGround);  // coordinate of point in image system
    tProjImAndGrad aPImGr = aScanB->InternalCalib()->DiffGround2Im(aPScanB0); // compute proj & gradient

    // Vector of indexes of unknwons
    if (aVIndUk)
    {
        aScanA->PushIndexes(*aVIndUk);      // add the unknowns [C,R] of the scan
        aScanB->PushIndexes(*aVIndUk);       // add the unknowns [C,R] of the camera
    }

    // vector that will contain values of observation at this step
    aScanA->Pose_WU().PushObs(aVObs,false); // no transpose for scan
    aScanB->Pose_WU().PushObs(aVObs,true);  // true because we transpose: we use W->C, which is the transposition of IJK : C->W

    aPScanA.PushInStdVector(aVObs);   //
    aPScanB0.PushInStdVector(aVObs);

    aPImGr.mGradI.PushInStdVector(aVObs);  // Grad Proj/PCam
    aPImGr.mGradJ.PushInStdVector(aVObs);

    auto [aRad0,aGradIm] = aData.mVGr.at(aKPt);  // Radiom & grad
    aVObs.push_back(aRad0);
    aGradIm.PushInStdVector(aVObs);
}

tREAL8 cBA_LidarLidarRaster::Add1Patch(const cPt3dr & aPGround, const std::string & aScanName)
{
    std::vector<cData1ImLidPhgr> aVData; // for each image where patch is visible will store the data
    cWeightAv<tREAL8>   aAvgRes;    // compute average residual
    tREAL8 aMinResidual = INFINITY;

    //  Parse all the scans, we will select the ones where the patch is visible
    for (auto & aScanData: mVScans)
    {
        auto & aWeighter = mWeightersMap.at(aScanName+"-"+aScanData.mScanName);
        if (aScanData.mScanName==aScanName)
            continue; // no obs on the same scan
        cStaticLidar * aScanTo = aScanData.mLidarRaster;
        cDataGenUnTypedIm<2> & aGenDImDist = aScanTo->getRasterDistance();
        if (aScanTo->IsVisible(aPGround)) // first test : is central point visible
        {
            cData1ImLidPhgr  aData; // data that will be filled
            aData.mScanAName = aScanName;
            aData.mScanBName = aScanData.mScanName;
            cPt2dr aPIm = aScanTo->Ground2Image(aPGround); // extract the image  projection
            tREAL8 aDist = Norm2(aPGround-aScanTo->Center());
            if (!aScanTo->IsValidPoint(aPIm))
                continue;
            if (aGenDImDist.InsideInterpolator(*mInterp,aPIm,1.0))  // is it sufficiently inside
            {
                auto aVGr = aGenDImDist.GetValueAndGradInterpol(*mInterp,aPIm); // extract pair Value/Grad of image
                aData.mVGr = {aVGr};
                tREAL8 aValIm = aData.mVGr.at(0).first;   // value of first/central pixel in this image
                tREAL8 aResidual = aValIm-aDist;
                if (fabs(aResidual)<fabs(aMinResidual))
                    aMinResidual = aResidual;
                if (aWeighter.SingleWOfResidual( std::vector<tREAL8>{aResidual})==0.0)
                    continue;
                aAvgRes.Add(1.0,fabs(aResidual));  // compute std deviation
                aVData.push_back(aData); // memorize the data for this image
            }
        }
    }

    // if less than 1 scan to: nothing valuable to do
    if (aVData.size()<1) return 999;
    // accumlulate for computing average of deviation
    // mLastResidual.Add(1.0,  (aStdDev.StdDev(1e-5) *aVData.size()) / (aVData.size()-1.0));
    // mLastResidual.Add(1.0,  (aStdDev.StdDev(1e-5) ) );
    mLastResidual.Add(aVData.size(),  Square(aAvgRes.Average() ) );

    AddPatchDist(aPGround,aVData);

    return aMinResidual;
}


void cBA_LidarLidarRaster::AddPatchDist
    (const cPt3dr & aPGround,
     const std::vector<cData1ImLidPhgr> &aVData
     )
{
    // read the solver now, because was not initialized at creation
    cResolSysNonLinear<tREAL8> *  aSys = mBA.Sys();
    // parse the data of the patch
    for (const auto & aData : aVData)
    {
        std::vector<int>       aVIndUk;
        std::vector<tREAL8>    aVObs;
        SetVUkVObs (aPGround,&aVIndUk,aVObs,aData,0);
        aSys->CalcAndAddObs(mEq,aVIndUk,aVObs,
                            mWeightersMap.at(aData.mScanAName+"-"+aData.mScanBName));
    }
}

};
