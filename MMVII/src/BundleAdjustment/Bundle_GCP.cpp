#include "BundleAdjustment.h"


namespace MMVII
{

/* -------------------------------------------------------------- */
/*                cMMVII_BundleAdj::GCP                           */
/* -------------------------------------------------------------- */

void cMMVII_BundleAdj::AddGCP(const  std::vector<double>& aWeightGCP, cSetMesImGCP *  aMesGCP)
{
    mMesGCP = aMesGCP;
    mWeightGCP = aWeightGCP;

    if (1)
    {
        StdOut()<<  "MESIM=" << mMesGCP->MesImOfPt().size() << " MesGCP=" << mMesGCP->MesGCP().size()  << std::endl;
    }
}

void cMMVII_BundleAdj::InitItereGCP()
{
    if (
            (mMesGCP!=nullptr)   //  if GCP where initialized
         && (mWeightGCP[0] > 0)  // is GGP are unknown
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

void cMMVII_BundleAdj::OneItere_OnePackGCP(const cSetMesImGCP * aSet,const std::vector<double> & aVW)
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

     tREAL8 aSigmaGCP = aVW[0];
     bool  aGcpUk = (aSigmaGCP>0);  // are GCP unknowns
     bool  aGcpFix = (aSigmaGCP==0);  // is GCP just an obervation
     tREAL8 aWeightGround =   aGcpFix ? 1.0 : (1.0/Square(aSigmaGCP)) ;  // standard formula, avoid 1/0
     tREAL8 aWeightImage =   (1.0/Square(aVW[1])) ;  // standard formula

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

	    if (! aGcpUk) // case  subst we now can make schurr commpl and subst
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
	OneItere_OnePackGCP(mMesGCP,mWeightGCP);
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



};


