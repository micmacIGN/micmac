#include "Topo.h"
#include "MMVII_PhgrDist.h"
#include "ctopopoint.h"
#include "ctopoobsset.h"
#include "ctopoobs.h"
#include "../BundleAdjustment/BundleAdjustment.h"
#include "cMMVII_Appli.h"

namespace MMVII
{

void cMMVII_BundleAdj::InitItereTopo()
{
    if (mTopo)
    {
        std::cout<<"cMMVII_BundleAdj::InitItereTopo\n";
        mTopo->FromFile(mVGCP, mPhProj);
        mTopo->AddToSys(mSetIntervUK); //after all is created
    }
}

cBA_Topo::cBA_Topo
(cPhotogrammetricProject *aPhProj, const std::string & aTopoFilePath)  :
    mPhProj  (aPhProj),
    mTopoObsType2equation
    {
        {eTopoObsType::eDist, EqTopoDist(true,1)},
        {eTopoObsType::eHz, EqTopoHz(true,1)},
        {eTopoObsType::eZen, EqTopoZen(true,1)},
        {eTopoObsType::eDX, EqTopoDX(true,1)},
        {eTopoObsType::eDY, EqTopoDY(true,1)},
        {eTopoObsType::eDZ, EqTopoDZ(true,1)},
        //{eTopoObsType::eDist, EqDist3D(true,1)},
        //{eTopoObsType::eSubFrame, EqTopoSubFrame(true,1)},
        //{eTopoObsType::eDistParam, EqDist3DParam(true,1)},
    },
    mInFile(aTopoFilePath), mIsReady(false)
{
    std::string aPost = Postfix(mInFile,'.',true);
    if (UCaseEqual(aPost,"obs"))
    {
        cTopoData aTopoData;
        aTopoData.FromCompFile(mInFile);
        mInFile = mInFile + ".json";
        aTopoData.ToFile(mInFile);
    }
}

cBA_Topo::~cBA_Topo()
{
    //std::cout<<"delete cBA_Topo"<<std::endl;
    clear();
    for (auto& [_, aEq] : mTopoObsType2equation)
                  delete aEq;

}

void cBA_Topo::clear()
{
    mAllPts.clear();
    std::for_each(mAllObsSets.begin(), mAllObsSets.end(), [](auto s){ delete s; });
    mAllObsSets.clear();
    mIsReady = false;
}

void cBA_Topo::makePtsUnknowns(const std::vector<cBA_GCP*> & vGCP, cPhotogrammetricProject *aPhProj)
{
    for (auto & [aName, aPtT] : getAllPts())
    {
        aPtT.findOrMakeUK(vGCP, aPhProj, aPtT.getInitCoord());
    }
}

void cBA_Topo::ToFile(const std::string & aName)
{
    cTopoData aTopoData(this);
    aTopoData.ToFile(aName);
}

void cBA_Topo::FromData(const cTopoData &aTopoData, const std::vector<cBA_GCP *> & vGCP, cPhotogrammetricProject *aPhProj)
{
    for (auto & aPointData: aTopoData.mAllPoints)
    {
        mAllPts[aPointData.mName] = cTopoPoint(
                    aPointData.mName,aPointData.mInitCoord,aPointData.mIsFree,
                    aPointData.mSigmas);
        if (aPointData.mVertDefl.has_value())
            mAllPts[aPointData.mName].setVertDefl(aPointData.mVertDefl.value_or(cPt2dr(0.,0.)));
    }

    makePtsUnknowns(vGCP, aPhProj);

    for (auto & aSetData: aTopoData.mAllObsSets)
    {
        // create set
        switch (aSetData.mType) {
        case eTopoObsSetType::eStation:
            mAllObsSets.push_back(make_TopoObsSet<cTopoObsSetStation>(this));
            break;
        default:
            MMVII_INTERNAL_ASSERT_User(false, eTyUEr::eUnClassedError, "Error: unknown eTopoObsSetType.")
        }
        cTopoObsSet *aSet = mAllObsSets.back();

        // fill obs
        for (auto & aObsData: aSetData.mObs)
        {
            aSet->addObs(aObsData.mType, this, aObsData.mPtsNames, aObsData.mMeasures,
                         {true, aObsData.mSigmas});
        }

        // finish initialization
        switch (aSetData.mType) {
        case eTopoObsSetType::eStation:
        {
            cTopoObsSetStation * st1 = static_cast<cTopoObsSetStation*>(aSet);
            st1->setIsOriented(aSetData.mStationIsOriented.value_or(false));
            st1->setIsVericalized(aSetData.mStationIsVericalized.value_or(false));
            std::string aOriginName;
            MMVII_INTERNAL_ASSERT_User(st1->getAllObs().size()>0, eTyUEr::eUnClassedError, "Error: Obs Set without obs.")
            aOriginName = aSet->getObs(0)->getPointName(0);
            // check that every obs goes from the same point
            for (auto &aObs : st1->getAllObs())
                MMVII_INTERNAL_ASSERT_User(aObs->getPointName(0)==aOriginName, eTyUEr::eUnClassedError, "Error: Obs Set with several origins")
            st1->setOrigin(aOriginName, true); // use 1st from name as station name
            // MMVII_DEV_WARNING("TODO: read if topo station is verticalized")
            break;
        }
        default:
            MMVII_INTERNAL_ASSERT_User(false, eTyUEr::eUnClassedError, "Error: unknown eTopoObsSetType.")
        }

        MMVII_INTERNAL_ASSERT_User(aSet->initialize(), eTyUEr::eUnClassedError,
                                   "Error: Station initialization failed.")
    }
    mIsReady = true;
}

void cBA_Topo::FromFile(const std::vector<cBA_GCP*> & vGCP, cPhotogrammetricProject *aPhProj)
{
    cTopoData aTopoData;
    aTopoData.FromFile(mInFile);
    FromData(aTopoData, vGCP, aPhProj);
}

void cBA_Topo::print()
{
    std::cout<<"Points:\n";
    for (auto& [aName, aPtT] : mAllPts)
        std::cout<<" - "<<aPtT.toString()<<"\n";
    std::cout<<"ObsSets:\n";
    for (auto &obsSet: mAllObsSets)
        std::cout<<" - "<<obsSet->toString()<<"\n";
}

void cBA_Topo::printObs(bool withDetails)
{
    tREAL8 aAvgResNorm = 0.0;
    int nbObs =0;
    for (auto &obsSet: mAllObsSets)
        for (auto & obs: obsSet->getAllObs())
        {
            if (withDetails)
                StdOut() << obs->toString()<< "\n";
            for (unsigned int i=0;i<obs->getMeasures().size();++i)
            {
                ++nbObs;
                aAvgResNorm += fabs(obs->getResiduals()[i]) / obs->getWeights().getSigmas()[i];
            }
        }
    aAvgResNorm /= nbObs;
    StdOut() << "Topo average std residual: " << aAvgResNorm << "\n";
}

void cBA_Topo::AddToSys(cSetInterUK_MultipeObj<tREAL8> & aSet)
{
    MMVII_INTERNAL_ASSERT_strong(mIsReady,"cBA_Topo is not ready");
    for (auto& [aName, aPt]: mAllPts)
        aPt.AddToSys(aSet);
    for (auto& anObsSet: mAllObsSets)
        anObsSet->AddToSys(aSet);
}

bool cBA_Topo::mergeUnknowns(cResolSysNonLinear<tREAL8> &aSys)
{
    bool ok = true;
    for (auto &set: mAllObsSets)
    {
        switch (set->getType()) {
        default:
            break;
        }
    }
    return ok;
}


void cBA_Topo::makeConstraints(cResolSysNonLinear<tREAL8> &aSys)
{
    for (auto& [aName, aPt]: mAllPts)
        aPt.makeConstraints(aSys);
    for (auto &set: mAllObsSets)
        set->makeConstraints(aSys);
}

cCalculator<double>*  cBA_Topo::getEquation(eTopoObsType tot) const {
    auto eq = mTopoObsType2equation.find(tot);
    if (eq != mTopoObsType2equation.end())
        return mTopoObsType2equation.at(tot);
    else
    {
        MMVII_INTERNAL_ERROR("unknown equation for obs type")
        return nullptr;
    }
}

cTopoPoint & cBA_Topo::getPoint(std::string name)
{
    if (mAllPts.count(name)==0)
    {
        MMVII_INTERNAL_ASSERT_User(false, eTyUEr::eUnClassedError, "Error: unknown point "+name)
    }
    return mAllPts.at(name);
}

/**  In a bundle adjusment its current that some variable are "hard" frozen, i.e they are
 *   considered as constant.  We could write specific equation, but generally it's more
 *   economic (from software devlopment) to just indicate this frozen part to the system.
 *
 *   Here the frozen unknown are the poses of the master camera of each bloc because the
 *   calibration is purely relative
 *
 */
void cBA_Topo::SetFrozenAndSharedVars(cResolSysNonLinear<tREAL8> & aSys)
{
     // create unknowns for all stations
     mergeUnknowns(aSys); //
     makeConstraints(aSys);
}

void cBA_Topo::AddTopoEquations(cResolSysNonLinear<tREAL8> & aSys)
{
    mSigma0 = 0.0;
    int aNbObs = 0;
    int aNbUk = 0;
    for (auto &obsSet: mAllObsSets)
        for (size_t i=0;i<obsSet->nbObs();++i)
        {
            cTopoObs* obs = obsSet->getObs(i);
            //std::cout<<"add eq: "<<obs->toString()<<" ";
            auto equation = getEquation(obs->getType());
            aSys.CalcAndAddObs(equation, obs->getIndices(this), obs->getVals(), obs->getWeights());
            for (unsigned int i=0; i<obs->getMeasures().size();++i)
            {
                double residual = equation->ValComp(0,i);
                obs->getResiduals().at(i) = residual;
#ifdef VERBOSE_TOPO
                StdOut() << "  resid: " << residual << " ";
#endif
                mSigma0 += Square(residual/obs->getWeights().getSigmas()[i]);
            }
            aNbObs += obs->getMeasures().size();
#ifdef VERBOSE_TOPO
            StdOut() << "\n";
#endif
        }

    aNbUk = aSys.NbVar() - aSys.GetNbLinearConstraints();
    mSigma0 = sqrt(mSigma0/(aNbObs-aNbUk));
    //StdOut() << "Sigma0 topo: " << mSigma0 << "\n";
}



//-------------------------------------------------------------------

void BenchTopoComp1example(cTopoData aTopoData, tREAL4 targetSigma0)
{
    cSetInterUK_MultipeObj<double> aSetIntervMultObj;
    double aLVM = 0.;

    aTopoData.ToFile(cMMVII_Appli::TmpDirTestMMVII()+"bench-in.json");

    cBA_Topo aTopo(nullptr, cMMVII_Appli::TmpDirTestMMVII()+"bench-in.json");
    aTopo.FromFile( {}, nullptr);


#ifdef VERBOSE_TOPO
    aTopo.print();
#endif
    aTopo.AddToSys(aSetIntervMultObj);
    cDenseVect<double> aVUk = aSetIntervMultObj.GetVUnKnowns();
    cResolSysNonLinear<double>  aSys = cResolSysNonLinear<double>(eModeSSR::eSSR_LsqNormSparse,aVUk);

    for (int iter=0; iter<6; ++iter)
    {
#ifdef VERBOSE_TOPO
        std::cout<<"Iter "<<iter<<std::endl;
#endif
        aTopo.SetFrozenAndSharedVars(aSys);
        aTopo.AddTopoEquations(aSys);
        const auto & aVectSol = aSys.R_SolveUpdateReset(aLVM);
        aSetIntervMultObj.SetVUnKnowns(aVectSol);
#ifdef VERBOSE_TOPO
        std::cout<<"  Sigma0: "<<aTopo.Sigma0()<<std::endl;
        aTopo.print();
#endif
    }
    aTopo.ToFile(cMMVII_Appli::TmpDirTestMMVII()+"bench-out.json");

    MMVII_INTERNAL_ASSERT_bench(std::abs(aTopo.Sigma0()-targetSigma0)<1e-5,"TopoComp sigma0 final");

    aSetIntervMultObj.SIUK_Reset();
}

void BenchTopoComp(cParamExeBench & aParam)
{
    if (! aParam.NewBench("TopoComp")) return;

    BenchTopoComp1example(cTopoData::createEx1(), 0.70711);
    BenchTopoComp1example(cTopoData::createEx3(), 1.41421);

    //std::cout<<"Bench Topo finished."<<std::endl;
    aParam.EndBench();
    return;
}


};

