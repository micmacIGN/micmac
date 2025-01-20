#include "BundleAdjustment.h"
#include "MMVII_Interpolators.h"
#include "MMVII_2Include_Tiling.h"

namespace MMVII
{


/**  Class for geometrically indexing the lidars (on 2D point) for patches creation */

template <class Type> class cTil2DTri3D
{
    public :
        static constexpr int TheDim = 2;
        typedef cPt2dr             tPrimGeom;
        typedef cTriangulation3D<Type> *  tArgPG;

        tPrimGeom  GetPrimGeom(tArgPG aPtrTri) const {return Proj(ToR(aPtrTri->KthPts(mInd)));}

        cTil2DTri3D(size_t anInd) : mInd(anInd) {}
        size_t  Ind() const {return mInd;}
       

    private :
        size_t  mInd;
};


cBA_LidarPhotogra::cBA_LidarPhotogra(cMMVII_BundleAdj& aBA,const std::vector<std::string>& aParam) :
    mBA         (aBA),                                 // memorize the bundel adj class itself (access to optimizer)
    mNumMode    (cStrIO<int>::FromStr(aParam.at(0))),  // mode of matching (int 4 now) 0 ponct, 1 Census
    mTri        (aParam.at(1)),                        // Lidar point themself, stored as a triangulation
    mInterp     (nullptr),                            // interpolator see bellow
    mEqLidPhgr  ( (mNumMode==0) ? EqEqLidarImPonct(true,1) : EqEqLidarImCensus(true,1))  // equation of egalisation Lidar/Phgr
{
   //  By default  use tabulation of apodized sinus cardinal
   std::vector<std::string> aParamInt {"Tabul","1000","SinCApod","10","10"};
   if (aParam.size() >=3)
   {
      // if specified, take user's param
      aParamInt = Str2VStr(aParam.at(2));
   }
   // create the interpaltor itself
   mInterp  = cDiffInterpolator1D::AllocFromNames(aParamInt);

   // parse the camera and create images
   for (const auto aPtrCam : aBA.VSIm())
   {
       // is it a central perspective camera ?
       if (aPtrCam->IsSensorCamPC())
       {
           mVCam.push_back(aPtrCam->GetSensorCamPC());  // yes get it
           mVIms.push_back(cIm2D<tU_INT1>::FromFile(aPtrCam->NameImage()));  // read the image
       }
       else
       {
          MMVII_UnclasseUsEr("cBA_LidarPhotogra : sensor is not central perspective");
       }
   }

   // Creation of the patches, to comment ...
   if (1)
   {
/*
        cTplBoxOfPts<tREAL8,2> aBoxObj;
        int aNbPtsByPtch = 32;

        for (size_t aKP=0 ; aKP<mTri.NbPts() ; aKP++)
        {
             aBoxObj.Add(ToR(Proj(mTri.KthPts(aKP))));
        }
        cBox2dr aBox = aBoxObj.CurBox();
        // Pi d^ 2  /NbByP = Surf / NbTot
        tREAL8 aDistMoy = std::sqrt(aNbPtsByPtch *aBox.NbElem()/ (mTri.NbPts()*M_PI));
        tREAL8 aDistReject =  aDistMoy *1.5;


        cTiling<cTil2DTri3D<tREAL4> >  aTileAll(aBox,true,mTri.NbPts()/20,&mTri);
        cTiling<cTil2DTri3D<tREAL4> >  aTileSelect(aBox,true,mTri.NbPts()/20,&mTri);

        for (size_t aKP=0 ; aKP<mTri.NbPts() ; aKP++)
        {
             aTileAll.Add(cTil2DTri3D<tREAL4>(aKP));
        }


        int aCpt=0;

        for (size_t aKP=0 ; aKP<mTri.NbPts() ; aKP++)
        {
             cPt2dr aPt  = ToR(Proj(mTri.KthPts(aKP)));
             if (aTileSelect.GetObjAtDist(aPt,aDistReject).empty())
             {
                 aTileSelect.Add(cTil2DTri3D<tREAL4>(aKP));
                 auto aLIptr = aTileAll.GetObjAtDist(aPt,aDistMoy);
                 std::vector<int> aPatch;
                 aPatch.push_back(aKP);
                 for (const auto aPtrI : aLIptr)
                 {
                     if (aPtrI->Ind() !=aKP)
                     {
                        aPatch.push_back(aPtrI->Ind());
                     }
                 }
                 if (aPatch.size() > 5)
                 {
                     aCpt += aPatch.size();
                     mLPatches.push_back(aPatch);
                }
             }
        }


        StdOut() << "Patches: DistReject=" << aDistReject 
                << " NbPts=" << mTri.NbPts() << " => " << aCpt 
                << " NbPatch=" << mLPatches.size() << " NbAvg => " <<  aCpt / double(mLPatches.size())
                << "\n";
*/
   }
}

cBA_LidarPhotogra::~cBA_LidarPhotogra() 
{
    delete mEqLidPhgr;
    delete mInterp;
}

void cBA_LidarPhotogra::AddObs(tREAL8 aW)
{
    mLastResidual.Reset();
    if (mNumMode==0)
    {
       for (size_t aKP=0 ; aKP<mTri.NbPts() ; aKP++)
       {
           Add1Patch(aW,{ToR(mTri.KthPts(aKP))});
       }
    }
    else
    {
        MMVII_UnclasseUsEr("Dont handle Census");
        for (const auto& aPatchIndex : mLPatches)
        {
            std::vector<cPt3dr> aVP;
            for (const auto anInd : aPatchIndex)
                aVP.push_back(ToR(mTri.KthPts(anInd)));
            Add1Patch(aW,aVP);
        }
    }


    if (mLastResidual.SW() != 0)
       StdOut() << "  * Lid/Phr Residual Rad " << mLastResidual.Average() << "\n";
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
}



void  cBA_LidarPhotogra::Add1Patch(tREAL8 aWeight,const std::vector<cPt3dr> & aVPatchGr)
{
     // read the solver now, because was not initialized at creation 
     cResolSysNonLinear<tREAL8> *  aSys = mBA.Sys();

     std::vector<cData1ImLidPhgr> aVData; // for each image where patch is visible will store the data
     cWeightAv<tREAL8,tREAL8> aWAv;       // compute average of image for radiom unknown
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
                  aWAv.Add(1.0,aValIm);     // compute average
                  aStdDev.Add(1.0,aValIm);  // compute std deviation
              }

          }
     }

     // if less than 2 images : nothing valuable to do
     if (aVData.size()<2) return;

     // accumlulate for computing average of deviation
     mLastResidual.Add(1.0,  (aStdDev.StdDev(1e-5) *aVData.size()) / (aVData.size()-1.0));


     if (mNumMode==0)
     {
        cPt3dr    aPGround = aVPatchGr.at(0);
        std::vector<tREAL8> aVTmpAvg{aWAv.Average()};  // vector for initializingz the temporay (here 1 = average)
        cSetIORSNL_SameTmp<tREAL8>  aStrSubst(aVTmpAvg); // structure for handling schurr eliminatio,
        // parse the data of the patch
        for (const auto & aData : aVData)
        {
            cSensorCamPC * aCam = mVCam.at(aData.mKIm);  // extract the camera
            cPt3dr aPCam = aCam->Pt_W2L(aPGround);  // coordinate of point in image system
            tProjImAndGrad aPImGr = aCam->InternalCalib()->DiffGround2Im(aPCam); // compute proj & gradient

            // Vector of indexes of unknwons 
            std::vector<int>  aVIndUk{-1} ;   // first one is a temporary (convention < 0)
            aCam->PushIndexes(aVIndUk);       // add the unknowns [C,R] of the camera


            // vector that will contains values of observation at this step
            std::vector<tREAL8> aVObs;  
            aCam->Pose_WU().PushObs(aVObs,true);  // true because we transpose: we use W->C, which is the transposition of IJK : C->W

            aPGround.PushInStdVector(aVObs);   //
            aPCam.PushInStdVector(aVObs);
            
            aPImGr.mGradI.PushInStdVector(aVObs);  // Grad Proj/PCam
            aPImGr.mGradJ.PushInStdVector(aVObs);
            
            auto [aRad0,aGradIm] = aData.mVGr.at(0);  // Radiom & grad
            aVObs.push_back(aRad0);
            aGradIm.PushInStdVector(aVObs);
            
            // accumulate the equation involving the radiom
            aSys->R_AddEq2Subst(aStrSubst,mEqLidPhgr,aVIndUk,aVObs,aWeight);
        }
        // do the substitution & add the equation reduced (Schurr complement)
        aSys->R_AddObsWithTmpUK(aStrSubst);
     }
     else if (mNumMode==1)
     {
            // to complete ...
     }
}

};
