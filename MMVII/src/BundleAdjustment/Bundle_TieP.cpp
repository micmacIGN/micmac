#include "BundleAdjustment.h"


namespace MMVII
{

    /* ---------------------------------------- */
    /*            Tie Points                    */
    /* ---------------------------------------- */

void cMMVII_BundleAdj::AddMTieP(cComputeMergeMulTieP  * aMTP,const cStdWeighterResidual & aWIm)
{
     mMTP = aMTP;
     mTieP_Weighter = aWIm;
}


void cMMVII_BundleAdj::OneItere_TieP()
{
   if (!mMTP)
      return;

   // update the bundle point by 3D-intersection:
   // To see : maybe don't update each time; probably add some robust option
   mMTP->SetPGround();

   cWeightAv<tREAL8> aWeigthedRes;
   for (const auto & aPair : mMTP->Pts())
   {
       const auto & aConfig  = aPair.first;

       // local vector of sensor & colinearity equation, directly indexale in [0,NbIm]
       std::vector<cSensorImage *> aVS ; 
       std::vector<cCalculator<double> *> aVEqCol ;

       for (size_t aKIm : aConfig)
       {
           aVS.push_back(mVSIm.at(aKIm));
	   aVEqCol.push_back(mVSIm.at(aKIm)->GetEqColinearity());
       }

       const auto & aVals  = aPair.second;
       size_t aNbIm = aConfig.size();
       size_t aNbPts = aVals.mVPGround.size();


       //  parse all the multiple tie points of a given config
       for (size_t aKPts=0; aKPts<aNbPts ; aKPts++)
       {
           const cPt3dr & aPGr = aVals.mVPGround.at(aKPts);
	   cSetIORSNL_SameTmp<tREAL8>  aStrSubst(aPGr.ToStdVector());

	   size_t aNbEqAdded = 0;
           for (size_t aKIm=0 ; aKIm<aNbIm ; aKIm++)
           {
               const cPt2dr & aPIm =  aVals.mVPIm.at(aKPts*aNbIm+aKIm);
	       cSensorImage* aSens = aVS.at(aKIm);

	       cPt2dr aResidual  = aPIm-aSens->Ground2Image(aPGr);
               tREAL8 aWeightImage =  mTieP_Weighter.SingleWOfResidual(aResidual);

	       cCalculator<double> * anEqColin =  aVEqCol.at(aKIm);

               std::vector<double> aVObs = aPIm.ToStdVector();  // put Xim & Yim as observation
               aSens->PushOwnObsColinearity(aVObs);  // add eventual observation of sensor (as rot with central persp)

               std::vector<int> aVIndGlob = {-1,-2,-3};  // index of unknown, begins with temporay
               for (auto & anObj : aSens->GetAllUK())  // now put sensor unknown
                  anObj->PushIndexes(aVIndGlob);

	       if (aWeightImage>0)
	       {
                   aWeigthedRes.Add(aWeightImage,Norm2(aResidual));
                   mSys->R_AddEq2Subst(aStrSubst,anEqColin,aVIndGlob,aVObs,aWeightImage);
		   aNbEqAdded++;
               }
           }

	   // if at least 2 tie-point, we can add equation with schurr-complement
	   if (aNbEqAdded>=2)
              mSys->R_AddObsWithTmpUK(aStrSubst);  // finnaly add obs accummulated
       }

   }
   StdOut() << "Weighted Residual=" << aWeigthedRes.Average() << std::endl;
}


};



