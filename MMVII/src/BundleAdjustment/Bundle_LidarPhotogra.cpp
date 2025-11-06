#include "BundleAdjustment.h"
#include "MMVII_Interpolators.h"
#include "MMVII_2Include_Tiling.h"
#include "MMVII_Tpl_Images.h"

namespace MMVII
{


/**  Class for geometrically indexing the lidars (on 2D point) for patches creation , used
   to instantiate cTilingIndex 
*/


cBA_LidarPhotogra::cBA_LidarPhotogra(cPhotogrammetricProject * aPhProj,
                                     cMMVII_BundleAdj& aBA,const std::vector<std::string>& aParam, cBA_LidarPhotogra_Type aType) :
    mPhProj     (aPhProj),
    mBA         (aBA),                                 // memorize the bundel adj class itself (access to optimizer)
    mModeSim    (Str2E<eImatchCrit>(aParam.at(0))),    // mode of matching
    mTri        (nullptr),                             // if triangulation type: Lidar point themself, stored as a triangulation
    mLidarData  (nullptr),                             // if raster type: raster lidar data
    mInterp     (nullptr),                             // interpolator see bellow
    mEqLidPhgr  (nullptr),                             // equation of egalisation Lidar/Phgr
    mPertRad    (false),
    mNbPointByPatch (32),
    mWeight      (1/Square(cStrIO<double>::FromStr(aParam.at(2)))),
    mNbUsedPoints (0),
    mNbUsedObs (0)
{
    if (aType==cBA_LidarPhotogra_Type::Triangulation)
    {
        std::string aLidarFileName = aParam.at(1);
        MMVII_INTERNAL_ASSERT_User(UCaseEqual(LastPostfix(aLidarFileName),"ply"),
                                   eTyUEr::eUnClassedError,"Lidar PLY file mandatory in triangulation mode, got \"" + aParam.at(1) + "\"");
        mTri = new cTriangulation3D<tREAL4>(aLidarFileName);
    } else {
        //read all xml files from directory ?
        mPhProj->DPStaticLidar().SetDirIn(aParam.at(1));
        std::string aPat2Sup =  "Scan-(.*)-(.*)\\." + GlobTaggedNameDefSerial()  ;
        std::string aFullPat2Sup = mPhProj->DPStaticLidar().FullDirIn() + aPat2Sup;
        tNameSet aSet = SetNameFromPat(aFullPat2Sup);
        std::vector<std::string> aVect = ToVect(aSet);
        for (const auto & aNameSens : aVect)
        {
            // TODO: make a vector of lidar data?
            mLidarData = mPhProj->ReadStaticLidar(mPhProj->DPStaticLidar(), aNameSens, true);
        }

        MMVII_INTERNAL_ASSERT_User(mLidarData,
                                   eTyUEr::eUnClassedError,"Error opening static scans " + aFullPat2Sup);
    }

   if (mModeSim==eImatchCrit::eDifRad) 
      mEqLidPhgr = EqEqLidarImPonct (true,1);
   else if (mModeSim==eImatchCrit::eCensus) 
      mEqLidPhgr = EqEqLidarImCensus(true,1);
   else if (mModeSim==eImatchCrit::eCorrel) 
      mEqLidPhgr = EqEqLidarImCorrel(true,1);
   else
   {
      MMVII_UnclasseUsEr("Bad enum for cBA_LidarPhotogra");
   }

   //  By default  use tabulation of apodized sinus cardinal
   std::vector<std::string> aParamInt {"Tabul","1000","SinCApod","10","10"};
   // if interpolator is not empty
   if ((aParam.size() >=4) && (!aParam.at(3).empty()))
   {
      // if specified, take user's param
      aParamInt = Str2VStr(aParam.at(3));
   }
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
   // create the interpolator itself
   mInterp  = cDiffInterpolator1D::AllocFromNames(aParamInt);
// delete mInterp;
// mInterp = cScaledInterpolator::AllocTab(cCubicInterpolator(-0.5),3,1000);

   // parse the camera and create images
   for (const auto aPtrCam : aBA.VSIm())
   {
       // is it a central perspective camera ?
       if (aPtrCam->IsSensorCamPC())
       {
           mVCam.push_back(aPtrCam->GetSensorCamPC());  // yes get it
           mVIms.push_back(cIm2D<tU_INT1>::FromFile(aPtrCam->NameImage()));  // read the image
           if (mPertRad)
           {
               cDataIm2D<tU_INT1> &  aDIm = mVIms.back().DIm();
               for (auto  aPix : aDIm)
               {
                   tREAL8 aMul =   (3+ sin(aPix.x()/70.0)) / 4.0;
                   aDIm.SetV(aPix,aDIm.GetV(aPix)*aMul);
               }
           }
       }
       else
       {
          MMVII_UnclasseUsEr("cBA_LidarPhotogra : sensor is not central perspective");
       }
   }

   // Creation of the patches, to comment ...
   if (mModeSim!=eImatchCrit::eDifRad)
   {
        if (mTri)
        {
           // cBox2dr aBox = BoxOfTri(mTri);
           cBox2dr aBox = mTri->Box2D();
           // estimate the distance for computing patching assuming a uniform  distributio,
           // Pi d^ 2  /NbByP = Surf / NbTot
           tREAL8 aDistMoy = std::sqrt(mNbPointByPatch *aBox.NbElem()/ (mTri->NbPts()*M_PI));
           tREAL8 aDistReject =  aDistMoy *1.5;
            mTri->MakePatches(mLPatchesI,aDistMoy,aDistReject,5);

            StdOut() << "Patches: DistReject=" << aDistReject
                     << " NbPts=" << mTri->NbPts()
                     << " NbPatch=" << mLPatchesI.size()
                     << "\n";
        }
   }
   if (mLidarData)
   {
       // Creation of the patches, choose a neigborhood around patch centers. TODO: adapt to images ground pixels size?
       tREAL4 aGndPixelSize = 0.01; // TODO !! Make it different for each patch center...
       if (mModeSim==eImatchCrit::eDifRad)
           mNbPointByPatch = 1;
       mLidarData->MakePatches(mLPatchesP,aGndPixelSize,mNbPointByPatch,5);
       StdOut() << "Nb patches: " << mLPatchesP.size() << "\n";
   }
}

cBA_LidarPhotogra::~cBA_LidarPhotogra() 
{
    delete mEqLidPhgr;
    delete mInterp;
    //if (mLidarData) delete mLidarData; // automatically deleted at the end
    if (mTri) delete mTri;
}

void cBA_LidarPhotogra::AddObs()
{
    mLastResidual.Reset();
    mNbUsedPoints = 0;
    mNbUsedObs = 0;
    if (mModeSim==eImatchCrit::eDifRad)
    {
        if (mTri)
        {
            for (size_t aKP=0 ; aKP<mTri->NbPts() ; aKP++)
            {
                Add1Patch(mWeight,{ToR(mTri->KthPts(aKP))});
            }
        } else {
            for (const auto& aPatch : mLPatchesP)
            {
                Add1Patch(mWeight,{mLidarData->to3D(aPatch.front())});
            }
        }
    }
    else
    {
        // MMVII_UnclasseUsEr("Dont handle Census");
        if (mTri)
        {
            for (const auto& aPatchIndex : mLPatchesI)
            {
                std::vector<cPt3dr> aVP;
                for (const auto anInd : aPatchIndex)
                    aVP.push_back(ToR(mTri->KthPts(anInd)));
                Add1Patch(mWeight,aVP);
            }
        } else {
            for (const auto& aPatch : mLPatchesP)
            {
                std::vector<cPt3dr> aVP;
                for (const auto aPt : aPatch)
                    aVP.push_back(mLidarData->to3D(aPt));
                Add1Patch(mWeight,aVP);
            }
        }
    }

    if (mLastResidual.SW() != 0)
       StdOut() << "  * Lid/Phr Residual Rad " << std::sqrt(mLastResidual.Average())
                 << " ("<<mNbUsedObs<<" obs, "<<mNbUsedPoints<<" points)\n";
}

void cBA_LidarPhotogra::SetVUkVObs
     (
         const cPt3dr&           aPGround,
         std::vector<int> *      aVIndUk,
         std::vector<tREAL8> &   aVObs,
         const cData1ImLidPhgr & aData,
         int                     aKPt
     )
{
    cSensorCamPC * aCam = mVCam.at(aData.mKIm);  // extract the camera
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
         aSys->R_AddEq2Subst(aStrSubst,mEqLidPhgr,aVIndUk,aVObs,aWeight);
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
             aSys->R_AddEq2Subst(aStrSubst,mEqLidPhgr,aVIndUk,aVObs,aWeight); // add the equation in Schurr structure
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
             aSys->R_AddEq2Subst(aStrSubst,mEqLidPhgr,aVIndUk,aVObs,aWeight);  // add equation in tmp struct
         }
     }

     aStrSubst.AddOneLinearObs(aNbPt,aVIndPt,aVFixAvg,0.0);  // force average
     aStrSubst.AddOneLinearObs(aNbPt,aVIndPt,aVFixVar,0.0);  // force standard dev

     aSys->R_AddObsWithTmpUK(aStrSubst);
}

void  cBA_LidarPhotogra::Add1Patch(tREAL8 aWeight,const std::vector<cPt3dr> & aVPatchGr)
{
     std::vector<cData1ImLidPhgr> aVData; // for each image where patch is visible will store the data
     cComputeStdDev<tREAL8>   aStdDev;    // compute the standard deviation of projected radiometry (indicator) 

     //  Parse all the image, we will select the images where all point of a patch are visible
     for (size_t aKIm=0 ; aKIm<mVCam.size() ; aKIm++)
     {
          cSensorCamPC * aCam = mVCam[aKIm]; // extract cam
          cDataIm2D<tU_INT1> & aDIm = mVIms[aKIm].DIm(); // extract image

          if (aCam->IsVisible(aVPatchGr.at(0))) // first test : is central point visible
          {
              cData1ImLidPhgr  aData; // data that will be filled
              aData.mKIm = aKIm;
              for (size_t aKPt=0 ; aKPt<aVPatchGr.size() ; aKPt++) // parse the points of the patch
              {
                   cPt3dr aPGround = aVPatchGr.at(aKPt);
                   if (aCam->IsVisible(aPGround))  // is the point visible in the camera
                   {
                        cPt2dr aPIm = mVCam[aKIm]->Ground2Image(aPGround); // extract the image  projection
                        if (aDIm.InsideInterpolator(*mInterp,aPIm,1.0))  // is it sufficiently inside
                        {
                            auto aVGr = aDIm.GetValueAndGradInterpol(*mInterp,aPIm); // extract pair Value/Grad of image
                            aData.mVGr.push_back(aVGr); // push it at end of stack
                        }
                   }
              }
              //  Does all the point of the patch were inside the image ?
              if (aData.mVGr.size() == aVPatchGr.size())
              {
                  aVData.push_back(aData); // memorize the data for this image

                  tREAL8 aValIm = aData.mVGr.at(0).first;   // value of first/central pixel in this image
                  // aWAv.Add(1.0,aValIm);     // compute average
                  aStdDev.Add(1.0,aValIm);  // compute std deviation
              }
          }
     }

     // if less than 2 images : nothing valuable to do
     if (aVData.size()<2) return;

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
        AddPatchCorrel(aWeight,aVPatchGr,aVData);
     }
}

};
