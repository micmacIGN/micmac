#include "BundleAdjustment.h"
#include "MMVII_Interpolators.h"
#include "MMVII_2Include_Tiling.h"
#include "MMVII_Tpl_Images.h"

namespace MMVII
{


/**  Class for geometrically indexing the lidars (on 2D point) for patches creation , used
   to instantiate cTilingIndex 
*/

template <class Type> class cTil2DTri3D
{
    public :
        static constexpr int TheDim = 2;          // Pre-requite for instantite cTilingIndex
        typedef cPt2dr             tPrimGeom;     // Pre-requite for instantite cTilingIndex
        typedef cTriangulation3D<Type> *  tArgPG; // Pre-requite for instantite cTilingIndex

        /**  Pre-requite for instantite cTilingIndex : indicate how we extract geometric primitive from one object */

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
    mEqLidPhgr  (nullptr), // equation of egalisation Lidar/Phgr
    mPertRad    (false)
{
   if      (mNumMode==0) mEqLidPhgr = EqEqLidarImPonct (true,1);
   else if (mNumMode==1) mEqLidPhgr = EqEqLidarImCensus(true,1);
   else if (mNumMode==2) mEqLidPhgr = EqEqLidarImCorrel(true,1);

   //  By default  use tabulation of apodized sinus cardinal
   std::vector<std::string> aParamInt {"Tabul","1000","SinCApod","10","10"};
   // if interpolator is not empty
   if ((aParam.size() >=3) && (!aParam.at(2).empty()))
   {
      // if specified, take user's param
      aParamInt = Str2VStr(aParam.at(2));
   }
   if (aParam.size() >=4)
   {
       mPertRad = true;
   }
   // create the interpaltor itself
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
   if (mNumMode!=0)
   {
        int aNbPtsByPtch = 32;   // approximative number of point by patch
        
        // create the bounding box of all points
        cTplBoxOfPts<tREAL8,2> aBoxObj;  // Box of object 
        for (size_t aKP=0 ; aKP<mTri.NbPts() ; aKP++)
        {
             // Proj 3d -> 2d   , ToR  float -> real
             aBoxObj.Add(ToR(Proj(mTri.KthPts(aKP))));
        }
        // create the "compiled" box from the dynamix
        cBox2dr aBox = aBoxObj.CurBox();

        // estimate the distance for computing patching assuming a uniform  distributio,
        // Pi d^ 2  /NbByP = Surf / NbTot
        tREAL8 aDistMoy = std::sqrt(aNbPtsByPtch *aBox.NbElem()/ (mTri.NbPts()*M_PI));
        tREAL8 aDistReject =  aDistMoy *1.5;

        // indexation of all points
        cTiling<cTil2DTri3D<tREAL4> >  aTileAll(aBox,true,mTri.NbPts()/20,&mTri);
        for (size_t aKP=0 ; aKP<mTri.NbPts() ; aKP++)
        {
             aTileAll.Add(cTil2DTri3D<tREAL4>(aKP));
        }

        int aCpt=0;
        // indexation of all points selecte as center of patches
        cTiling<cTil2DTri3D<tREAL4> >  aTileSelect(aBox,true,mTri.NbPts()/20,&mTri);
        // parse all points
        for (size_t aKP=0 ; aKP<mTri.NbPts() ; aKP++)
        {
             cPt2dr aPt  = ToR(Proj(mTri.KthPts(aKP)));
             // if the points is not close to an existing center of patch : create a new patch
             if (aTileSelect.GetObjAtDist(aPt,aDistReject).empty())
             {
                //  Add it in the tiling of select 
                 aTileSelect.Add(cTil2DTri3D<tREAL4>(aKP));
                 // extract all the point close enough to the center
                 auto aLIptr = aTileAll.GetObjAtDist(aPt,aDistMoy);
                 std::vector<int> aPatch; // the patch itself = index of points
                 aPatch.push_back(aKP);  // add the center at begining
                 for (const auto aPtrI : aLIptr)
                 {
                     if (aPtrI->Ind() !=aKP) // dont add the center twice
                     {
                        aPatch.push_back(aPtrI->Ind());
                     }
                 }
                 // some requirement on minimal size
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
        // MMVII_UnclasseUsEr("Dont handle Census");
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
            std::vector<int>       aVIndUk{-1}; // first one is a temporary (convention < 0)
            std::vector<tREAL8>    aVObs;
            SetVUkVObs (aPGround,&aVIndUk,aVObs,aData,0);
            
            // accumulate the equation involving the radiom
            aSys->R_AddEq2Subst(aStrSubst,mEqLidPhgr,aVIndUk,aVObs,aWeight);
        }
        // do the substitution & add the equation reduced (Schurr complement)
        aSys->R_AddObsWithTmpUK(aStrSubst);
     }
     else if (mNumMode==1)
     {
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
     else if (mNumMode==2)  // mode correlation
     {
         size_t aNbPt = aVPatchGr.size();
         cDenseVect<tREAL8>  aVMoy(aNbPt,eModeInitImage::eMIA_Null);
	 std::vector<cDenseVect<tREAL8>>  aListVRad;
         for (const auto & aData : aVData)
         {
              cDenseVect<tREAL8> aV(aNbPt);
              for (size_t aK=0 ; aK< aNbPt ; aK++)
              {
                  aV(aK)  = aData.mVGr.at(aK).first;
              }
	      aListVRad.push_back(aV);
              cDenseVect<tREAL8> aV01 = NormalizeMoyVar(aV);
	      aVMoy += aV01;
         }

	 aVMoy *=  1/ tREAL8(aVData.size());
         aVMoy =  NormalizeMoyVar(aVMoy);

         std::vector<tREAL8> aVTmp = aVMoy.ToStdVect();
         size_t aK0Im = aVTmp.size();

         for (const auto &  aVRad : aListVRad)
         {
             auto [A,B] =  LstSq_Fit_AxPBEqY(aVRad,aVMoy);
             aVTmp.push_back(A);
             aVTmp.push_back(B);
         }
         cSetIORSNL_SameTmp<tREAL8>  aStrSubst(aVTmp); // structure for handling schurr eliminatio,
         std::vector<int> aVIndPt;
         std::vector<tREAL8> aVFixAvg;
         std::vector<tREAL8> aVFixVar;

         for (int aKPt=0 ; aKPt <  (int) aNbPt ; aKPt++)
         {
             int aIndPt = -(1+aKPt);
             aVIndPt.push_back(aIndPt);
             aVFixAvg.push_back(1.0);
             //  S(R+dR) ^ 2 =1   ;  S (2 R dR ) = 1 - S(R^2)  ; but S(R^2)=1 by construction ...
             aVFixVar.push_back(2*aVMoy(aKPt));

             for (int aKIm=0 ;  aKIm< (int) aVData.size() ; aKIm++)
             {
                 int aIndIm = -(1+aK0Im+2*aKIm);
                 std::vector<int>       aVIndUk{aIndPt,aIndIm,aIndIm-1} ;
                 std::vector<tREAL8>    aVObs;
                 SetVUkVObs (aVPatchGr.at(aKPt),&aVIndUk,aVObs,aVData.at(aKIm),aKPt);
                 aSys->R_AddEq2Subst(aStrSubst,mEqLidPhgr,aVIndUk,aVObs,aWeight);
             }
         }
         aStrSubst.AddOneLinearObs(aNbPt,aVIndPt,aVFixAvg,0.0);
         aStrSubst.AddOneLinearObs(aNbPt,aVIndPt,aVFixVar,0.0);

         aSys->R_AddObsWithTmpUK(aStrSubst);
     }
}

};
