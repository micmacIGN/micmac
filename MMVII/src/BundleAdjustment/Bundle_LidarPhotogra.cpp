#include "BundleAdjustment.h"
#include "MMVII_Interpolators.h"
#include "MMVII_2Include_Tiling.h"

namespace MMVII
{


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
    mBA         (aBA),
    mNumMode    (cStrIO<int>::FromStr(aParam.at(0))),
    mTri        (aParam.at(1)),
    // mInterp     (new cCubicInterpolator(-0.5)),
    mInterp     (nullptr),
    mEqLidPhgr  ( (mNumMode==0) ? EqEqLidarImPonct(true,1) : EqEqLidarImCensus(true,1))
{
   //cSinCApodInterpolator aSinC(20,20);
   //mInterp  = new cTabulatedDiffInterpolator(aSinC,1000);
   std::vector<std::string> aParamInt {"Tabul","1000","SinCApod","10","10"};
   if (aParam.size() >=3)
   {
      aParamInt = Str2VStr(aParam.at(2));
   }
   mInterp  = cDiffInterpolator1D::AllocFromNames(aParamInt);

   for (const auto aPtrCam : aBA.VSIm())
   {
       if (aPtrCam->IsSensorCamPC())
       {
           mVCam.push_back(aPtrCam->GetSensorCamPC());
           mVIms.push_back(cIm2D<tU_INT1>::FromFile(aPtrCam->NameImage()));
       }
       else
       {
          MMVII_UnclasseUsEr("cBA_LidarPhotogra : sensor is not central perspective");
       }
   }

   if (1)
   {
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
           // cPt3df aPF  = mTri.KthPts(aKP);
           // AddObs(aW,cPt3dr(aPF.x(),aPF.y(),aPF.z()));
           Add1Patch(aW,{ToR(mTri.KthPts(aKP))});
       }
    }
    else
    {
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

   cSensorCamPC * aCam = mVCam.at(aData.mKIm);
   cPt3dr aPCam = aCam->Pt_W2L(aPGround);
   tProjImAndGrad aPImGr = aCam->InternalCalib()->DiffGround2Im(aPCam);

   if (aVIndUk)
   {
       // *aVIndUk = std::vector<int>  {-1} ;   // Index for temporary radiom
       aCam->PushIndexes(*aVIndUk);
   }

   cPoseWithUK& aPUK = aCam->Pose_WU();


   aPUK.PushObs(aVObs,true);  // true this is the W->C, wich the transposition of IJK : C->W
   aPGround.PushInStdVector(aVObs);

   aPCam.PushInStdVector(aVObs);
   aPImGr.mGradI.PushInStdVector(aVObs);
   aPImGr.mGradJ.PushInStdVector(aVObs);

   aVObs.push_back(aData.mVGr.at(aKPt).first);
   aData.mVGr.at(aKPt).second.PushInStdVector(aVObs);
}



void  cBA_LidarPhotogra::Add1Patch(tREAL8 aWeight,const std::vector<cPt3dr> & aVPatchGr)
{
     cResolSysNonLinear<tREAL8> *  aSys = mBA.Sys();
     std::vector<cData1ImLidPhgr> aVData;
     cWeightAv<tREAL8,tREAL8> aWAv;

     cComputeStdDev<tREAL8> aStdDev;

     for (size_t aKIm=0 ; aKIm<mVCam.size() ; aKIm++)
     {
          cSensorCamPC * aCam = mVCam[aKIm];
          cDataIm2D<tU_INT1> & aDIm = mVIms[aKIm].DIm();

          if (aCam->IsVisible(aVPatchGr.at(0)))
          {
              cData1ImLidPhgr  aData;
              aData.mKIm = aKIm;
              for (size_t aKPt=0 ; aKPt<aVPatchGr.size() ; aKPt++)
              {
                   cPt3dr aPGround = aVPatchGr.at(aKPt);
                   if (aCam->IsVisible(aPGround))
                   {
                        cPt2dr aPIm = mVCam[aKIm]->Ground2Image(aPGround);
                        if (aDIm.InsideInterpolator(*mInterp,aPIm,1.0))
                        {
                            auto aVGr = aDIm.GetValueAndGradInterpol(*mInterp,aPIm);
                            aData.mVGr.push_back(aVGr);
                        }
                   }
              }
              if (aData.mVGr.size() == aVPatchGr.size())
              {
                  tREAL8 aValIm = aData.mVGr.at(0).first;
                  aVData.push_back(aData);
                  aWAv.Add(1.0,aValIm);
                  aStdDev.Add(1.0,aValIm);
              }

          }
     }


     if (aVData.size()<2) return;

     mLastResidual.Add(1.0,  (aStdDev.StdDev(1e-5) *aVData.size()) / (aVData.size()-1.0));



     if (mNumMode==0)
     {
        std::vector<tREAL8> aVTmpAvg({aWAv.Average()});
        cSetIORSNL_SameTmp<tREAL8>  aStrSubst(aVTmpAvg);
        for (const auto & aData : aVData)
        {
           std::vector<int>  aVIndUk{-1} ;
           std::vector<tREAL8>  aVObs;
           SetVUkVObs(aVPatchGr.at(0),&aVIndUk,aVObs,aData,0);
           aSys->R_AddEq2Subst(aStrSubst,mEqLidPhgr,aVIndUk,aVObs,aWeight);
        }
        aSys->R_AddObsWithTmpUK(aStrSubst);
    }
    else if (mNumMode==1)
    {
        for (size_t aKPt=1; aKPt<aVPatchGr.size() ; aKPt++)
        {
             cWeightAv<tREAL8,tREAL8> aAvRatio;
             for (const auto & aData : aVData)
             {
                 tREAL8 aV0 = aData.mVGr.at(0).first;
                 tREAL8 aVK = aData.mVGr.at(aKPt).first;
                 aAvRatio.Add(1.0,NormalisedRatioPos(aV0,aVK)) ;
             }
             std::vector<tREAL8> aVTmpAvg({aAvRatio.Average()});
             cSetIORSNL_SameTmp<tREAL8>  aStrSubst(aVTmpAvg);
             for (const auto & aData : aVData)
             {
                std::vector<int>  aVIndUk{-1} ;
                std::vector<tREAL8>  aVObs;

                SetVUkVObs(aVPatchGr.at(0)  ,&aVIndUk,aVObs,aData,0);
                SetVUkVObs(aVPatchGr.at(aKPt),nullptr ,aVObs,aData,aKPt);
                aSys->R_AddEq2Subst(aStrSubst,mEqLidPhgr,aVIndUk,aVObs,aWeight);
            }
        aSys->R_AddObsWithTmpUK(aStrSubst);
        }
    }


}

};
