#include "BundleAdjustment.h"
#include "MMVII_Interpolators.h"

namespace MMVII
{

cBA_LidarPhotogra::cBA_LidarPhotogra(cMMVII_BundleAdj& aBA,const std::vector<std::string>& aParam) :
    mBA         (aBA),
    mTri        (aParam.at(0)),
    mInterp     (new cCubicInterpolator(-0.5)),
    mEqLidPhgr  (EqEqLidarImPonct(true,1))
{
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
}

cBA_LidarPhotogra::~cBA_LidarPhotogra() 
{
    delete mEqLidPhgr;
    delete mInterp;
}


void cBA_LidarPhotogra::AddObs(tREAL8 aW)
{
    for (size_t aKP=0 ; aKP<mTri.NbPts() ; aKP++)
    {
        cPt3df aPF  = mTri.KthPts(aKP);
        AddObs(aW,cPt3dr(aPF.x(),aPF.y(),aPF.z()));
    }
}


void  cBA_LidarPhotogra::AddObs(tREAL8 aWeight,const cPt3dr & aPGround)
{
    cResolSysNonLinear<tREAL8> *  aSys = mBA.Sys();

     class cData1ImLidPhgr
     {
        public :
          size_t mKIm;
          tREAL8 mValIm;
          cPt2dr mGradIm;
     };
     std::vector<cData1ImLidPhgr> aVData;
     cWeightAv<tREAL8,tREAL8> aWAv;

     for (size_t aKIm=0 ; aKIm<mVCam.size() ; aKIm++)
     {
          if (mVCam[aKIm]->IsVisible(aPGround))
          {
              cPt2dr aPIm = mVCam[aKIm]->Ground2Image(aPGround);
              cDataIm2D<tU_INT1> & aDIm = mVIms[aKIm].DIm();
              if (aDIm.InsideInterpolator(*mInterp,aPIm,1.0))
              {
                  auto [aVal,aGrad] = aDIm.GetValueAndGradInterpol(*mInterp,aPIm);
                  cData1ImLidPhgr  aData;

                  aData.mKIm = aKIm;
                  aData.mValIm = aVal;
                  aData.mGradIm = aGrad;
                  aVData.push_back(aData);
                  aWAv.Add(1.0,aVal);
              }
          }
     }

     if (aVData.size()<2) return;

     std::vector<tREAL8> aVTmpAvg({aWAv.Average()});
     cSetIORSNL_SameTmp<tREAL8>  aStrSubst(aVTmpAvg);


     for (const auto & aData : aVData)
     {
        std::vector<int>  aVIndUk {-1} ;   // Index for temporary radiom
        cSensorCamPC * aCam = mVCam.at(aData.mKIm);
        cPt3dr aPCam = aCam->Pt_W2L(aPGround);
        tProjImAndGrad aPImGr = aCam->InternalCalib()->DiffGround2Im(aPCam);

        cPoseWithUK& aPUK = aCam->Pose_WU();
        aPUK.PushIndexes(aVIndUk);
        // void PutUknowsInSetInterval(cSetInterUK_MultipeObj<tREAL8> * aSetInterv) ;

        std::vector<tREAL8>  aVObs;
        aPGround.PushInStdVector(aVObs);
        aPUK.PushObs(aVObs,true);  // true this is the W->C, wich the transposition of IJK : C->W

        aPCam.PushInStdVector(aVObs);
        aPImGr.mGradI.PushInStdVector(aVObs);
        aPImGr.mGradJ.PushInStdVector(aVObs);

        aVObs.push_back(aData.mValIm);
        aData.mGradIm.PushInStdVector(aVObs);

StdOut() << "VUKKKK " << aVIndUk << "\n";
        aSys->R_AddEq2Subst(aStrSubst,mEqLidPhgr,aVIndUk,aVObs,aWeight);
     }

     aSys->R_AddObsWithTmpUK(aStrSubst);
}

};
