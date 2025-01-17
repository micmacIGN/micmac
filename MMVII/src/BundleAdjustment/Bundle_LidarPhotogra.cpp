#include "BundleAdjustment.h"
#include "MMVII_Interpolators.h"

namespace MMVII
{

class cBA_LidarPhotogra
{
    public :

       void AddObs(tREAL8 aW);

    private :

        void AddObs(tREAL8 aW,const cPt3dr & aPt);

        std::vector<cPt3dr>             mPts;
        std::vector<cSensorCamPC *>     mVCam;
        cDiffInterpolator1D *           mInterp;
        std::vector<cIm2D<tU_INT1>>     mImages;
        cCalculator<double>  *          mEqLidPhgr;
        cResolSysNonLinear<tREAL8> *    mSys;
};

class cData1ImLidPhgr
{
   public :
     size_t mKIm;
     tREAL8 mValIm;
     cPt2dr mGradIm;
};


void  cBA_LidarPhotogra::AddObs(tREAL8 aW,const cPt3dr & aPGround)
{
     std::vector<cData1ImLidPhgr> aVData;
     cWeightAv<tREAL8,tREAL8> aWAv;

     for (size_t aKIm=0 ; aKIm<mVCam.size() ; aKIm++)
     {
          if (mVCam[aKIm]->IsVisible(aPGround))
          {
              cPt2dr aPIm = mVCam[aKIm]->Ground2Image(aPGround);
              cDataIm2D<tU_INT1> & aDIm = mImages[aKIm].DIm();
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

     std::vector<int>  aVIndUk {-1} ;   // Index for temporary radiom
     for (const auto & aData : aVData)
     {
        cSensorCamPC * aCam = mVCam.at(aData.mKIm);
        aCam->Pose_WU().PushIndexes(aVIndUk);

        std::vector<tREAL8>  aVObs;
        aPGround.PushInStdVector(aVObs);
/*
   for (auto & anObj : aSens->GetAllUK())
                anObj->PushIndexes(aVIndGlob);
*/

FakeUseIt(aData);

          
     }

}

};
