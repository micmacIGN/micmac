#include "BundleAdjustment.h"

#include "MMVII_MeasuresIm.h"

namespace MMVII
{

/* -------------------------------------------------------------- */
/*                cMes2DDirInfo                                     */
/* -------------------------------------------------------------- */

cMes2DDirInfo::cMes2DDirInfo (const std::string &aDirNameIn, const cStdWeighterResidual &aWeighter) :
    mDirNameIn(aDirNameIn), mWeighter(aWeighter)
{

}

cMes2DDirInfo* cMes2DDirInfo::addMes2DDirInfo(cBA_GCP &aBA_GCP, const std::string & aDirNameIn,
                                    const cStdWeighterResidual & aStdWeighterResidual)
{
    aBA_GCP.mAllMes2DDirInfo.push_back(new cMes2DDirInfo(aDirNameIn,aStdWeighterResidual));
    return aBA_GCP.mAllMes2DDirInfo.back();
}


/* -------------------------------------------------------------- */
/*                cMes3DDirInfo                                     */
/* -------------------------------------------------------------- */

cMes3DDirInfo::cMes3DDirInfo (const std::string &aDirNameIn, const std::string &aDirNameOut, tREAL8 aSGlob) :
    mDirNameIn(aDirNameIn), mDirNameOut(aDirNameOut), mSGlob(aSGlob)
{

}

cMes3DDirInfo* cMes3DDirInfo::addMes3DDirInfo(cBA_GCP &aBA_GCP, const std::string & aDirNameIn,
                                        const std::string & aDirNameOut, tREAL8 aSGlob)
{
    aBA_GCP.mAllMes3DDirInfo.push_back(new cMes3DDirInfo(aDirNameIn,aDirNameOut, aSGlob));
    return aBA_GCP.mAllMes3DDirInfo.back();
}

/* -------------------------------------------------------------- */
/*                cBA_GCP                                         */
/* -------------------------------------------------------------- */

cBA_GCP::cBA_GCP()
{
}

cBA_GCP::~cBA_GCP() 
{
    DeleteAllAndClear(mGCP_UK);
    DeleteAllAndClear(mAllMes2DDirInfo);
    DeleteAllAndClear(mAllMes3DDirInfo);
}

void cBA_GCP::AddGCP3D(cMes3DDirInfo * aMesDirInfo, cSetMesGnd3D &aSetMesGnd3D, bool verbose)
{
    mMesGCP.AddMes3D(aSetMesGnd3D, aMesDirInfo);
}


void cBA_GCP::AddMes2D(cSetMesPtOf1Im &aSetMesIm, cMes2DDirInfo *aMesDirInfo, cSensorImage* cSensorImage, eLevelCheck OnNonExistP)
{
    mMesGCP.AddMes2D(aSetMesIm, aMesDirInfo, cSensorImage, OnNonExistP);
}

/* -------------------------------------------------------------- */
/*                cMMVII_BundleAdj::GCP                           */
/* -------------------------------------------------------------- */

void cMMVII_BundleAdj::AddGCP3D(cMes3DDirInfo * aMesDirInfo, cSetMesGnd3D &aSetMesGnd3D, bool verbose)
{
    mGCP.AddGCP3D(aMesDirInfo, aSetMesGnd3D, verbose);

    if (verbose && mVerbose)
    {
        StdOut()<< " MesGCP=" << mGCP.getMesGCP().MesGCP().size()  << std::endl;
    }
}

void cMMVII_BundleAdj::AddGCP2D(cMes2DDirInfo *aMesDirInfo, cSetMesPtOf1Im & aSetMesIm, cSensorImage* aSens, eLevelCheck aOnNonExistGCP, bool verbose)
{
    mGCP.AddMes2D(aSetMesIm, aMesDirInfo, aSens, aOnNonExistGCP);
}


void cMMVII_BundleAdj::InitItereGCP()
{
    for (const auto & aGCP : mGCP.getMesGCP().MesGCP())
    {
        if (aGCP.mMesDirInfo->mSGlob>0)
        {
            cPt3dr_UK * aPtrUK = new cPt3dr_UK(aGCP.mPt,aGCP.mNamePt);
            mGCP.mGCP_UK.push_back(aPtrUK);
            mSetIntervUK.AddOneObj(aPtrUK);
        } else {
            mGCP.mGCP_UK.push_back(nullptr); // to keep as many elements as mMesGCP
        }
    }
}


void cMMVII_BundleAdj::OneItere_GCP()
{
    auto & aSet                           = mGCP.getMesGCP();
    if (!aSet.IsPhaseGCPFinished())
        return;
    cSetMesGndPt&   aNewGCP               = mGCP.mNewGCP;
    std::vector<cPt3dr_UK*> & aGCP_UK     = mGCP.mGCP_UK;

    const std::vector<cMes1Gnd3D> &    aVMesGCP = aSet.MesGCP();
    const std::vector<cMultipleImPt> & aVMesIm  = aSet.MesImOfPt() ;
    const std::vector<cSensorImage*> & aVSens   = aSet.VSens() ;


    // StdOut() << "GCP " << aVMesGCP.size() << " " << aVMesIm.size() << " " << aVSens.size() << std::endl;

    size_t aNbGCP = aVMesGCP.size();


    int aNbGCPVis = 0;
    int aAvgVis = 0;
    int aAvgNonVis = 0;
    if (aNbGCP!=0)
    {
        aNewGCP = aSet; //copy
        for (size_t aK=0 ; aK< aNbGCP ; aK++)
        {
            if (!aGCP_UK[aK]) continue;
            aNewGCP.MesGCP()[aK].mPt = aGCP_UK[aK]->Pt();
        }
        if (mVerbose)
        {
            StdOut() << "  * Gcp0=" << aSet.AvgSqResidual() ;
            StdOut() << " , GcpNew=" << aNewGCP.AvgSqResidual() ; // getchar();
        }
    }


    // MMVII_INTERNAL_ASSERT_tiny(!aGcpUk,"Dont handle GCP UK 4 Now");

    std::vector<int> aVIndGround = {-1,-2,-3};

    //  Parse all GCP
    for (size_t aKp=0 ; aKp < aNbGCP ; aKp++)
    {
        const tREAL8 & aSigmaGCP              = aVMesGCP.at(aKp).mMesDirInfo->mSGlob;
        //   W>0  obs is an unknown "like others"
        //   W=0 , obs is fix , use schurr subst and fix the variables
        //   W<0 , obs is substitued
        bool  aGcpUk = (aSigmaGCP>0);  // are GCP unknowns
        bool  aGcpFix = (aSigmaGCP==0);  // is GCP just an obervation
        //  Three temporary unknowns for x-y-z of the 3d point
        std::vector<int> aVIndFix = (aGcpFix ? aVIndGround : std::vector<int>());

        const cPt3dr & aPGr = aVMesGCP.at(aKp).mPt;
        const cPt3dr & aPtSigmas = aVMesGCP.at(aKp).SigmasXYZ();
        cPt3dr_UK * aPtrGcpUk =  aGcpUk ? aGCP_UK[aKp] : nullptr;
        cSetIORSNL_SameTmp<tREAL8>  aStrSubst(aPGr.ToStdVector(),aVIndFix);

        const std::vector<cPt2dr> & aVPIm  = aVMesIm.at(aKp).VMeasures();
        const std::vector<int> &  aVIndIm  = aVMesIm.at(aKp).VImages();

        int aNbImVis  = 0;
        // Parse all image having a measure with this GCP
        for (size_t aKIm=0 ; aKIm<aVPIm.size() ; aKIm++)
        {
            cStdWeighterResidual& aGCPIm_Weighter = aVMesIm.at(aKp).VMesDirInfo().at(aKIm)->mWeighter;
            int aIndIm = aVIndIm.at(aKIm);
            cSensorImage * aSens = aVSens.at(aIndIm);
            const cPt2dr & aPIm = aVPIm.at(aKIm);
            //StdOut() << "aSensaSensaSens " << aSens->NameImage() << " " << aVIndIm << "\n";

            // compute indexe of unknown, if GCp are !UK we have fix index for temporary
            std::vector<int> aVIndGlob = aGcpUk ? (std::vector<int>()) : aVIndGround;
            if (aGcpUk)  // if GCP are UK, we have to fill with its index
            {
                aPtrGcpUk->PushIndexes(aVIndGlob);
            }
            //  Add index of sensor (Pose+Calib for PC Cam)
            for (auto & anObj : aSens->GetAllUK())
            {
                anObj->PushIndexes(aVIndGlob);
            }

            /*StdOut() << "VISSSS " << aSens->IsVisibleOnImFrame(aPIm)
                << " " << aPGr
                << " "<< aSens->IsVisible(aPGr)
            << "\n";*/

            // Do something only if GCP is visible
            if (aSens->IsVisibleOnImFrame(aPIm) && (aSens->IsVisible(aPGr)))
            {
                aNbImVis++;
                cPt2dr aResidual = aPIm - aSens->Ground2Image(aPGr);
                tREAL8 aWeightImage =   aGCPIm_Weighter.SingleWOfResidual(aResidual);
                cCalculator<double> * anEqColin =  aSens->GetEqColinearity();
                // the "obs" are made of 2 point and, possibily, current rotation (for PC cams)
                std::vector<double> aVObs = aPIm.ToStdVector();

                aSens->PushOwnObsColinearity(aVObs,aPGr);

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
        aAvgVis += aNbImVis;
        aAvgNonVis += aVPIm.size() -aNbImVis;
        aNbGCPVis += (aNbImVis !=0);

        // bool  aGcpUk = (aSigmaGCP>0);  // are GCP unknowns
        // bool  aGcpFix = (aSigmaGCP==0);  // is GCP just an obervation
        if (aVMesGCP.at(aKp).isFree())
            continue;

        cPt3dr aWeightGroundXYZ(1., 1., 1.);
        if (!aGcpFix)
            aWeightGroundXYZ = DivCByC( {1., 1., 1.}, Square(aSigmaGCP)* MulCByC(aPtSigmas,aPtSigmas));

        if (! aGcpUk) // case  subst,  we now can make schurr commpl and subst aSigmaGCP<=0
        {
            if (! aGcpFix)  // if GCP is not hard fix, we must add obs on ground
            {
                for (auto i = 0; i < 3; ++i)
                {
                    aStrSubst.AddFixCurVarTmp(aVIndGround[i],aWeightGroundXYZ[i]);
                }
            }
            mSys->R_AddObsWithTmpUK(aStrSubst);  // finnaly add obs accummulated
        }
        else  // aSigmaGCP >0
        {
            //  Add observation fixing GCP  aPGr
            // mR8_Sys->AddEqFixCurVar(*aPtrGcpUk,aPtrGcpUk->Pt(),aWeightGround);
            // FIX TO GCP INIT NOT TO LAST ESTIMATION
            for (auto i = 0; i < 3; ++i)
            {

                mR8_Sys->AddEqFixNewVal(*aPtrGcpUk,aPtrGcpUk->Pt()[i],aPGr[i],aWeightGroundXYZ[i]);
            }
            //previously: mR8_Sys->AddEqFixNewVal(*aPtrGcpUk,aPtrGcpUk->Pt(),aPGr,1/1000);
        }
    }

    if (mVerbose && (aNbGCP!=0))
    {
        StdOut() << " PropVis1Im=" << aNbGCPVis /double(aNbGCP)
                 << " AvgVis=" << aAvgVis/double(aNbGCP)
                 << " NonVis=" << aAvgNonVis/double(aNbGCP)
                    ;
        StdOut() << std::endl;
    }
}


void cMMVII_BundleAdj::Save_newGCP3D()
{
    mPhProj->SaveGCP3D(mGCP.mNewGCP.ExtractSetGCP("NewGCP"), "", true);
}

    /* ---------------------------------------- */
    /*            cPt3dr_UK                     */
    /* ---------------------------------------- */

template <const int Dim> cPtxdr_UK<Dim>::cPtxdr_UK(const tPt & aPt,const std::string& aName)  :
    mPt    (aPt),
    mName  (aName)
{
}

std::vector<std::string> VNameCoordsPt = {"x","y","z","t"};

template <const int Dim>  void cPtxdr_UK<Dim>::FillGetAdrInfoParam(cGetAdrInfoParam<tREAL8> & aGAIP)
{
    for (int aD=0 ; aD<Dim ; aD++)
    {
        aGAIP.TestParam(this,&mPt[aD],VNameCoordsPt.at(aD));
    }
    aGAIP.SetNameType("GCP");
    aGAIP.SetIdObj(mName);
}

template <const int Dim>  cPtxdr_UK<Dim>::~cPtxdr_UK()
{
        OUK_Reset();
}

template <const int Dim> void cPtxdr_UK<Dim>::PutUknowsInSetInterval()
{
    mSetInterv->AddOneInterv(mPt);
}

template <const int Dim>  const cPtxd<tREAL8,Dim> & cPtxdr_UK<Dim>::Pt() const {return mPt;}
template <const int Dim>  cPtxd<tREAL8,Dim> & cPtxdr_UK<Dim>::Pt() {return mPt;}

template class cPtxdr_UK<2>;
template class cPtxdr_UK<3>;

};



