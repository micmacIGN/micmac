#include "MMVII_Topo.h"
#include "MMVII_PhgrDist.h"
#include "../BundleAdjustment/BundleAdjustment.h"
#include "cMMVII_Appli.h"
#include <algorithm>

namespace MMVII
{

void cMMVII_BundleAdj::InitItereTopo()
{
    if (mTopo)
    {
        mTopo->FromData(mGCP, mPhProj);
        mTopo->AddToSys(mSetIntervUK); //after all is created
    }
}


cBA_Topo::cBA_Topo
(cPhotogrammetricProject *aPhProj)  :
    mPhProj  (aPhProj),
    mTopoObsType2equation
    {
        {eTopoObsType::eDist, EqTopoDist(true,1)},
        {eTopoObsType::eHz,   EqTopoHz(true,1)},
        {eTopoObsType::eZen,  EqTopoZen(true,1)},
        {eTopoObsType::eDX,   EqTopoDX(true,1)},
        {eTopoObsType::eDY,   EqTopoDY(true,1)},
        {eTopoObsType::eDZ,   EqTopoDZ(true,1)},
        {eTopoObsType::eDH,   EqTopoDH(true,1)},
        //{eTopoObsType::eDist, EqDist3D(true,1)},
        //{eTopoObsType::eDistParam, EqDist3DParam(true,1)},
    },
    mIsReady(false),
    mSysCo(nullptr)
{
#ifdef VERBOSE_TOPO
    for (auto& [_, aEq] : mTopoObsType2equation)
        aEq->SetDebugEnabled(true);
#endif

    if (aPhProj)
    {
        for (auto & aInFile: aPhProj->ReadTopoMes())
        {
            std::string aPost = Postfix(aInFile,'.',true);
            if (UCaseEqual(aPost,"obs"))
            {
                mAllTopoDataIn.InsertCompObsFile( aPhProj->DPTopoMes().FullDirIn() + aInFile );
            } else {
                cTopoData aTopoData;
                aTopoData.FromFile( aPhProj->DPTopoMes().FullDirIn() + aInFile );
                mAllTopoDataIn.InsertTopoData(aTopoData);
            }
        }
        mSysCo = mPhProj->CurSysCoGCP();
    } else {
        // no PhProj: this is a bench, topodata will be added later
        mSysCo = cSysCo::MakeSysCo("RTL*45*0*0*+proj=latlong");
    }
}

cBA_Topo::~cBA_Topo()
{
    clear();
    for (auto& [_, aEq] : mTopoObsType2equation)
          {
            (void)_;
            delete aEq;
           }
}

void cBA_Topo::clear()
{
    mAllPts.clear();
    std::for_each(mAllObsSets.begin(), mAllObsSets.end(), [](auto s){ delete s; });
    mAllObsSets.clear();
    mIsReady = false;
}

void cBA_Topo::findPtsUnknowns(const cBA_GCP & aBA_GCP, cPhotogrammetricProject *aPhProj)
{
    for (auto & [aName, aTopoPt] : getAllPts())
    {
        aTopoPt.findUK(aBA_GCP, aPhProj);
    }
}

void cBA_Topo::ToFile(const std::string & aName) const
{
    cTopoData aTopoData(this);
    aTopoData.ToFile(aName);
}


void cBA_Topo::AddPointsFromDataToGCP(cBA_GCP &aBA_GCP)
{
    // fill every ObsSet types
    if (!mAllTopoDataIn.mObsSetSimple.mObs.empty())
    {
        auto aSet = make_TopoObsSet<cTopoObsSetSimple>(this);
        mAllObsSets.push_back(aSet);
        for (auto & aObsData: mAllTopoDataIn.mObsSetSimple.mObs)
        {
            aSet->addObs(aObsData.mType, this, aObsData.mPtsNames, aObsData.mMeasures,
                         {true, aObsData.mSigmas});
        }
    }
    for (auto & aSetData: mAllTopoDataIn.mAllObsSetStations)
    {
        auto aSet = make_TopoObsSet<cTopoObsSetStation>(this);
        mAllObsSets.push_back(aSet);
        aSet->setOriStatus(aSetData.mStationOriStat); //< fill specific to this type of set
        for (auto & aObsData: aSetData.mObs)
        {
            aSet->addObs(aObsData.mType, this, aObsData.mPtsNames, aObsData.mMeasures,
                         {true, aObsData.mSigmas});
        }
    }

    std::set<std::string> aAllPointsNames; //< will create cTopoPoints for all points refered to in observations
    for (const auto & aSet: mAllObsSets)
        for (const auto & aObs: aSet->getAllObs())
            for (const auto & aName: aObs->getPointNames())
                aAllPointsNames.insert(aName);

    for (auto & aPointName: aAllPointsNames)
    {
        mAllPts[aPointName] = cTopoPoint(aPointName);
    }

    // add new points to GCP
    std::set<std::string> aAllPointsNamesNotFound;
    for (auto & aPointName: aAllPointsNames)
    {
        bool found = false;
        for (auto &aMesGCP: aBA_GCP.getMesGCP().MesGCP())
        {
            if (aMesGCP.mNamePt == aPointName)
            {
               found = true;
               break;
            }
        }
        if (!found)
            aAllPointsNamesNotFound.insert(aPointName);
    }

    cMes3DDirInfo * aMes3DDirInfo = nullptr;
    if (!aAllPointsNamesNotFound.empty())
    {
        aMes3DDirInfo = cMes3DDirInfo::addMes3DDirInfo(aBA_GCP, "newTopoIn",
                                          mPhProj?mPhProj->DPTopoMes().DirIn():"newTopoOut",1.0); // aDirNameIn and aSGlob are not used
    }

    for (auto & aPointName: aAllPointsNamesNotFound)
    {
        auto aMes3D = cMes1Gnd3D(cPt3dr::Dummy(), aPointName);
        aMes3D.mMesDirInfo = aMes3DDirInfo;
        aBA_GCP.getMesGCP().Add1GCP( aMes3D ); // points non-init
    }

    mAllTopoDataIn.clear(); // if this function is called again, nothing more to add
}

void cBA_Topo::FromData(const cBA_GCP & aBA_GCP, cPhotogrammetricProject *aPhProj)
{
    findPtsUnknowns(aBA_GCP, aPhProj);

    // initialization
    tryInitAll();

    // check that everything is initialized
    std::string aPtsNamesUninit="";
    for (auto& [aName, aTopoPt] : mAllPts)
    {
        if (!aTopoPt.isInit())
            aPtsNamesUninit += aName + " ";
    }
    MMVII_INTERNAL_ASSERT_User(aPtsNamesUninit.empty(), eTyUEr::eUnClassedError,
                               "Error: Initialization has failed for points: "+aPtsNamesUninit)
    for (auto & aSet: mAllObsSets)
    {
        MMVII_INTERNAL_ASSERT_User(aSet->isInit(), eTyUEr::eUnClassedError,
                                   "Error: Obs Set initialization failed: \""+aSet->getObs(0)->toString()+"\"")
    }
    mIsReady = true;
}


void cBA_Topo::print()
{
    StdOut() << "Points:\n";
    for (auto& [aName, aTopoPt] : mAllPts)
        {
            (void)aName;
            StdOut() << " - "<<aTopoPt.toString()<<"\n";
        }
    StdOut() << "ObsSets:\n";
    for (auto &obsSet: mAllObsSets)
        StdOut() << " - "<<obsSet->toString()<<"\n";
    printObs(false);
}

void cBA_Topo::printObs(bool withDetails)
{
    int nbObs =0;
    for (auto &obsSet: mAllObsSets)
        for (auto & obs: obsSet->getAllObs())
        {
            if (withDetails)
                StdOut() << obs->toString()<< "\n";
            nbObs += obs->getMeasures().size();
        }
    StdOut() << "Topo sigma0: " << mSigma0 << " (" << nbObs <<  " obs)\n";
}

std::vector<cTopoObs*> cBA_Topo::GetObsPoint(std::string aPtName) const
{
    std::vector<cTopoObs*> aVectObs;
    for (auto &obsSet: mAllObsSets)
        for (auto & obs: obsSet->getAllObs())
        {
            auto & aVectObsPtNames = obs->getPointNames();
            if (std::find(aVectObsPtNames.begin(), aVectObsPtNames.end(), aPtName) != aVectObsPtNames.end())
                aVectObs.push_back(obs);
        }
    return aVectObs;
}

void cBA_Topo::AddToSys(cSetInterUK_MultipeObj<tREAL8> & aSetInterUK)
{
    MMVII_INTERNAL_ASSERT_strong(mIsReady,"cBA_Topo is not ready");
    for (auto& anObsSet: mAllObsSets)
        aSetInterUK.AddOneObj(anObsSet);
}

bool cBA_Topo::mergeUnknowns()
{
    bool ok = true;
    for (auto &set: mAllObsSets)
    {
        switch (set->getType()) {
        case eTopoObsSetType::eSimple:
            break;
        case eTopoObsSetType::eStation:
            break;
        case eTopoObsSetType::eNbVals:
            MMVII_INTERNAL_ERROR("unknown obs set type")
        }
    }
    return ok;
}


void cBA_Topo::makeConstraints(cResolSysNonLinear<tREAL8> &aSys)
{
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

const cTopoPoint & cBA_Topo::getPoint(std::string name) const
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
     mergeUnknowns(); //
     makeConstraints(aSys);
}

void cBA_Topo::AddTopoEquations(cResolSysNonLinear<tREAL8> & aSys)
{
    mSigma0 = 0.0; // TODOJM compute it correctly will all obs, gcp constr etc.
    for (auto &obsSet: mAllObsSets)
        for (size_t i=0;i<obsSet->nbObs();++i)
        {
            cTopoObs* obs = obsSet->getObs(i);
            auto equation = getEquation(obs->getType());
            aSys.CalcAndAddObs(equation, obs->getIndices(), obs->getVals(), obs->getWeights());
#ifdef VERBOSE_TOPO
                StdOut() << obs->toString() << "      " ;
#endif
            for (unsigned int i=0; i<obs->getMeasures().size();++i)
            {
                double residual = equation->ValComp(0,i);
                obs->getResiduals().at(i) = residual;
                double residual_norm = residual/obs->getWeights().getSigmas()[i];
#ifdef VERBOSE_TOPO
                StdOut() << "  resid: " << residual_norm << " ";
#endif
                mSigma0 += Square(residual_norm);
            }
#ifdef VERBOSE_TOPO
            StdOut() << "\n";
#endif
        }

    int aNbObs = aSys.GetNbObs();
    int aNbUk = aSys.NbVar() - aSys.GetNbLinearConstraints();
    mSigma0 = sqrt(mSigma0/(aNbObs-aNbUk));
}

bool cBA_Topo::tryInitAll()
{

    for (auto & aSet: mAllObsSets)
        aSet->initialize(); // to get origin point for stations

    // get all stations ordered by origin to optimize research
    mAllStations.clear();

    for (auto & aSet: mAllObsSets)
    {
        if (aSet->getType() ==  eTopoObsSetType::eStation)
        {
            cTopoObsSetStation* set = dynamic_cast<cTopoObsSetStation*>(aSet);
            if (!set)
                MMVII_INTERNAL_ERROR("error set type")
            mAllStations[set->getPtOrigin()].push_back(set);
        }
    }

    // get all obs from simple stations in both directions
    mAllSimpleObs.clear();
    for (auto & aSet: mAllObsSets)
    {
        if (aSet->getType() ==  eTopoObsSetType::eSimple)
        {
            cTopoObsSetSimple* set = dynamic_cast<cTopoObsSetSimple*>(aSet);
            if (!set)
                MMVII_INTERNAL_ERROR("error set type")
            for (auto &aObs:set->getAllObs())
            {
                if (aObs->getPointNames().size()==2)
                {
                    // those 2-point obs are recorded for both points
                    mAllSimpleObs[&getPoint(aObs->getPointName(0))].push_back( aObs );
                    mAllSimpleObs[&getPoint(aObs->getPointName(1))].push_back( aObs );
                }
            }
        }
    }

    int aNbUninit=0;
    for (auto & aSet: mAllObsSets)
        if (!aSet->isInit())
            ++aNbUninit;
    for (auto& [aName, aTopoPt] : mAllPts)
        {
            (void)aName;
        if (!aTopoPt.isInit())
            ++aNbUninit;
        }
    int aPreviousNbUninit = aNbUninit + 1; // kickstart

    while (aPreviousNbUninit>aNbUninit)
    {
#ifdef VERBOSE_TOPO
        StdOut() << "tryInitAll: " << aNbUninit << " to init.\n";
#endif

        for (auto& [aName, aTopoPt] : mAllPts)
            {
                (void)aName;
            if (!aTopoPt.isInit())
                tryInit(aTopoPt, mAllStations, mAllSimpleObs);
        for (auto & aSet: mAllObsSets)
            if (!aSet->isInit())
                aSet->initialize();

        aPreviousNbUninit = aNbUninit;
        aNbUninit = 0;
        for (auto & aSet: mAllObsSets)
            if (!aSet->isInit())
                ++aNbUninit;
        for (auto& [aName, aTopoPt] : mAllPts)
            {
                (void)aName;
            if (!aTopoPt.isInit())
                ++aNbUninit;
            }
              }
      }
    return aNbUninit==0;
}
bool cBA_Topo::tryInit(cTopoPoint & aPtToInit, tStationsMap &stationsMap, tSimpleObsMap &allSimpleObs)
{
    if (aPtToInit.isInit())
        return true;
#ifdef VERBOSE_TOPO
    StdOut() << "tryInit: " << aPtToInit.getName() <<".\n";
#endif
    bool ok =    tryInit3Obs1Station(aPtToInit)
              || tryInitVertStations(aPtToInit)
                 ;
#ifdef VERBOSE_TOPO
    if (ok)
        StdOut() << "init coords: " << *aPtToInit.getPt() <<"\n";
#endif
    return ok;
}

//-------------------------------------------------------------------

void BenchTopoComp1example(const std::pair<cTopoData, cSetMesGnd3D>& aBenchData, tREAL4 targetSigma0)
{
    double aLVM = 0.;
    int aNbIter = 15;

    cMMVII_BundleAdj  aBA(nullptr);
    aBA.setVerbose(false);
    aBA.AddTopo();
    cBA_Topo * aTopo = aBA.getTopo();
    aTopo->mAllTopoDataIn.InsertTopoData(aBenchData.first);

    cSetMesGnd3D aMesGCP3Dtmp = aBenchData.second;
    cMes3DDirInfo * aMes3DDirInfo = cMes3DDirInfo::addMes3DDirInfo(aBA.getGCP(), "in","out",1.0);
    aBA.AddGCP3D(aMes3DDirInfo, aMesGCP3Dtmp, false);
    aTopo->AddPointsFromDataToGCP(aBA.getGCP());
    //here no 2d mes, fake it
    cMes2DDirInfo * aMes2DDirInfo = cMes2DDirInfo::addMes2DDirInfo(aBA.getGCP(), "in",cStdWeighterResidual());
    cSetMesPtOf1Im aSetMesPtOf1Im;
    aBA.AddGCP2D(aMes2DDirInfo, aSetMesPtOf1Im, nullptr, eLevelCheck::NoCheck);

    for (int aKIter=0 ; aKIter<aNbIter ; aKIter++)
    {
        aBA.OneIteration(aLVM);
    }

    aTopo->ToFile(cMMVII_Appli::TmpDirTestMMVII()+"bench-out.json");

    // StdOut() << "TOPOOOERR=" << std::abs(aTopo->Sigma0()-targetSigma0) << "\n";
    MMVII_INTERNAL_ASSERT_bench(std::abs(aTopo->Sigma0()-targetSigma0)<1e-5,"TopoComp sigma0 final");
}

void BenchTopoComp(cParamExeBench & aParam)
{
    if (! aParam.NewBench("TopoComp")) return;

    BenchTopoComp1example(cTopoData::createEx1(), 0.70711);
    BenchTopoComp1example(cTopoData::createEx3(), 1.00918);
    BenchTopoComp1example(cTopoData::createEx4(), 0.);

    //std::cout<<"Bench Topo finished."<<std::endl;
    aParam.EndBench();
    return;
}


};

