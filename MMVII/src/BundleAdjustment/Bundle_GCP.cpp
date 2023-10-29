#include "BundleAdjustment.h"


namespace MMVII
{

/* -------------------------------------------------------------- */
/*                cMMVII_BundleAdj::GCP                           */
/* -------------------------------------------------------------- */

void cMMVII_BundleAdj::AddGCP(tREAL8 aSigmaGCP,const  cStdWeighterResidual & aWeighter, cSetMesImGCP *  aMesGCP)
{
    mMesGCP = aMesGCP;
    mSigmaGCP = aSigmaGCP;
    mGCPIm_Weighter = aWeighter;

    if (1)
    {
        StdOut()<<  "MESIM=" << mMesGCP->MesImOfPt().size() << " MesGCP=" << mMesGCP->MesGCP().size()  << std::endl;
    }
}

void cMMVII_BundleAdj::InitItereGCP()
{
    if (
            (mMesGCP!=nullptr)   //  if GCP where initialized
         && (mSigmaGCP > 0)  // is GGP are unknown
       )
    {
        for (const auto & aGCP : mMesGCP->MesGCP())
	{
            cPt3dr_UK * aPtrUK = new cPt3dr_UK(aGCP.mPt);
            mGCP_UK.push_back(aPtrUK);
	    mSetIntervUK.AddOneObj(aPtrUK);
	}
    }
}

void cMMVII_BundleAdj::OneItere_OnePackGCP(const cSetMesImGCP * aSet)
{
    if (aSet==nullptr) return;
    //   W>0  obs is an unknown "like others"
    //   W=0 , obs is fix , use schurr subst and fix the variables
    //   W<0 , obs is substitued
     const std::vector<cMes1GCP> &      aVMesGCP = aSet->MesGCP();
     const std::vector<cMultipleImPt> & aVMesIm  = aSet->MesImOfPt() ;
     const std::vector<cSensorImage*> & aVSens   = aSet->VSens() ;

    // StdOut() << "GCP " << aVMesGCP.size() << " " << aVMesIm.size() << " " << aVSens.size() << std::endl;

     size_t aNbGCP = aVMesGCP.size();

     bool  aGcpUk = (mSigmaGCP>0);  // are GCP unknowns
     bool  aGcpFix = (mSigmaGCP==0);  // is GCP just an obervation
     tREAL8 aWeightGround =   aGcpFix ? 1.0 : (1.0/Square(mSigmaGCP)) ;  // standard formula, avoid 1/0

     if (1)
     {
        StdOut() << "  Res;  Gcp0: " << aSet->AvgSqResidual() ;
        if (aGcpUk)
        {
            mNewGCP = *aSet;
	    for (size_t aK=0 ; aK< aNbGCP ; aK++)
                mNewGCP.MesGCP()[aK].mPt = mGCP_UK[aK]->Pt();
            StdOut() << "  GcpNew: " << mNewGCP.AvgSqResidual() ;
        }
        StdOut() << std::endl;
     }

    // MMVII_INTERNAL_ASSERT_tiny(!aGcpUk,"Dont handle GCP UK 4 Now");

     //  Three temporary unknowns for x-y-z of the 3d point
     std::vector<int> aVIndGround = {-1,-2,-3};
     std::vector<int> aVIndFix = (aGcpFix ? aVIndGround : std::vector<int>());

     //  Parse all GCP
     for (size_t aKp=0 ; aKp < aNbGCP ; aKp++)
     {
           const cPt3dr & aPGr = aVMesGCP.at(aKp).mPt;
           cPt3dr_UK * aPtrGcpUk =  aGcpUk ? mGCP_UK[aKp] : nullptr;
	   cSetIORSNL_SameTmp<tREAL8>  aStrSubst(aPGr.ToStdVector(),aVIndFix);

	   const std::vector<cPt2dr> & aVPIm  = aVMesIm.at(aKp).VMeasures();
	   const std::vector<int> &  aVIndIm  = aVMesIm.at(aKp).VImages();

	   // Parse all image having a measure with this GCP
           for (size_t aKIm=0 ; aKIm<aVPIm.size() ; aKIm++)
           {
               int aIndIm = aVIndIm.at(aKIm);
               cSensorImage * aSens = aVSens.at(aIndIm);
               const cPt2dr & aPIm = aVPIm.at(aKIm);

	       // compute indexe of unknown, if GCp are !UK we have fix index for temporary
               std::vector<int> aVIndGlob = aGcpUk ? (std::vector<int>()) : aVIndGround;
               if (aGcpUk)  // if GCP are UK, we have to fill with its index
               {
                    aPtrGcpUk->PushIndexes(aVIndGlob);
               }
	       //  Add index of sensor (Pose+Calib for PC Cam)
               for (auto & anObj : aSens->GetAllUK())
                  anObj->PushIndexes(aVIndGlob);

	       // Do something only if GCP is visible 
               if (aSens->IsVisibleOnImFrame(aPIm) && (aSens->IsVisible(aPGr)))
               {
	             cPt2dr aResidual = aPIm - aSens->Ground2Image(aPGr);
                     tREAL8 aWeightImage =   mGCPIm_Weighter.SingleWOfResidual(aResidual);
	             cCalculator<double> * anEqColin =  mVEqCol.at(aIndIm);
                     // the "obs" are made of 2 point and, possibily, current rotation (for PC cams)
                     std::vector<double> aVObs = aPIm.ToStdVector();
		     aSens->PushOwnObsColinearity(aVObs);

		     if (aGcpUk)  // Case Uknown, we just add the equation
		     {
		        mSys->R_CalcAndAddObs(anEqColin,aVIndGlob,aVObs,aWeightImage);
		     }
		     else  // Case to subst by schur compl,we accumulate in aStrSubst
		     {
                        mSys->R_AddEq2Subst(aStrSubst,anEqColin,aVIndGlob,aVObs,aWeightImage);
		     }
               }
	    }

	    if (! aGcpUk) // case  subst,  we now can make schurr commpl and subst
	    {
                if (! aGcpFix)  // if GCP is not hard fix, we must add obs on ground
		{
                    for (auto & aIndGr : aVIndGround)
                        aStrSubst.AddFixCurVarTmp(aIndGr,aWeightGround);
		}
                mSys->R_AddObsWithTmpUK(aStrSubst);  // finnaly add obs accummulated
	    }
	    else
	    {
                 //  Add observation fixing GCP
                 mR8_Sys->AddEqFixCurVar(*aPtrGcpUk,aPtrGcpUk->Pt(),aWeightGround);
	    }
    }
}


void cMMVII_BundleAdj::OneItere_GCP()
{
     if (mMesGCP)
     {
	OneItere_OnePackGCP(mMesGCP);
     }
}

    /* ---------------------------------------- */
    /*            cPt3dr_UK                     */
    /* ---------------------------------------- */

cPt3dr_UK::cPt3dr_UK(const cPt3dr & aPt) :
    mPt  (aPt)
{
}

void cPt3dr_UK::PutUknowsInSetInterval()
{
    mSetInterv->AddOneInterv(mPt);
}

const cPt3dr & cPt3dr_UK::Pt() const {return mPt;}

cPt3dr_UK::~cPt3dr_UK()
{
        Reset();
}

    /* ---------------------------------------- */
    /*            AddViscosity                  */
    /* ---------------------------------------- */

void cMMVII_BundleAdj::AddPoseViscosity()
{
     //  parse all centra
     for (auto aPcPtr : mVSCPC)
     {
         if (aPcPtr!=nullptr)
         {
            if (mSigmaViscCenter>0)
	    {
               mR8_Sys->AddEqFixCurVar(*aPcPtr,aPcPtr->Center(),Square(1/mSigmaViscCenter));
	    }
            if (mSigmaViscAngles>0)
	    {
               mR8_Sys->AddEqFixCurVar(*aPcPtr,aPcPtr->Omega(),Square(1/mSigmaViscAngles));
	    }
	 }
     }
}

void cMMVII_BundleAdj::SetViscosity(const tREAL8& aViscTr,const tREAL8& aViscAngle)
{
    mSigmaViscCenter = aViscTr;
    mSigmaViscAngles = aViscAngle;
}

    /* ---------------------------------------- */
    /*            AddViscosity                  */
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
	   aVEqCol.push_back(mVEqCol.at(aKIm));
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
   StdOut() << "Weighted Residual=" << aWeigthedRes.Average() << "\n";
}


};



