#include "ctopocomp.h"
#include "MMVII_PhgrDist.h"
#include <memory>
namespace MMVII
{

cTopoComp::cTopoComp():
    isInit(false),
    mSetIntervMultObj(new cSetInterUK_MultipeObj<double>()), mSys(nullptr),
    mTopoObsType2equation
    {
        {TopoObsType::dist, EqDist3D(true,1)},
        {TopoObsType::subFrame, EqTopoSubFrame(true,1)},
        {TopoObsType::distParam, EqDist3DParam(true,1)},
    }
{
}

cTopoComp::~cTopoComp()
{
    std::for_each(mTopoObsType2equation.begin(), mTopoObsType2equation.end(), [](auto &e){ delete e.second; });
    delete mSys;
    delete mSetIntervMultObj;
    std::for_each(allPts.begin(), allPts.end(), [](auto p){ delete p; });
}

void cTopoComp::initializeLeastSquares()
{
    MMVII_INTERNAL_ASSERT_strong(!isInit, "cTopoComp: multiple initializeLeastSquares")

    for (auto & pt: allPts)
        mSetIntervMultObj->AddOneObj(pt); //before mSys creation
    for (auto &obsSet: allObsSets)
        mSetIntervMultObj->AddOneObj(obsSet.get()); //before mSys creation

    cDenseVect<double> aVUk = mSetIntervMultObj->GetVUnKnowns();
    mSys = new cResolSysNonLinear<double>(eModeSSR::eSSR_LsqNormSparse,aVUk);

    if (verbose)
    {
        StdOut()<<" nb free var: "<<mSys->CountFreeVariables()<<std::endl;
        StdOut()  <<  " init: " <<  mSys->CurGlobSol() << std::endl;
    }

    isInit = true;
}

cCalculator<double>*  cTopoComp::getEquation(TopoObsType tot) const {
    auto eq = mTopoObsType2equation.find(tot);
    if (eq != mTopoObsType2equation.end())
        return mTopoObsType2equation.at(tot);
    else
    {
        MMVII_INTERNAL_ERROR("unknown equation for obs type")
        return nullptr;
    }
}

void cTopoComp::print()
{
    std::cout<<"Points:\n";
    for (auto& pt: allPts)
        std::cout<<" - "<<pt->toString()<<"\n";
    std::cout<<"ObsSets:\n";
    for (auto &obsSet: allObsSets)
        std::cout<<" - "<<obsSet.get()->toString()<<"\n";
}

void cTopoComp::createEx1()
{
    //verbose = true;

    //create fixed points
    allPts.push_back(new cTopoPoint("ptA", cPt3dr(10,10,10), false));
    allPts.push_back(new cTopoPoint("ptB", cPt3dr(20,10,10), false));
    allPts.push_back(new cTopoPoint("ptC", cPt3dr(15,20,10), false));
    auto ptA = allPts[0];
    auto ptB = allPts[1];
    auto ptC = allPts[2];

    //add measured dist to point D
    allObsSets.push_back(make_TopoObsSet<cTopoObsSetSimple>());
    auto obsSet1 = allObsSets[0].get();
    allPts.push_back(new cTopoPoint("ptD", cPt3dr(14,14,14), true));
    auto ptD = allPts[3];
#define WW 0.01
    cTopoObs(obsSet1, TopoObsType::dist, std::vector{ptA, ptD}, {10.0}, {true, {WW}});
    cTopoObs(obsSet1, TopoObsType::dist, std::vector{ptB, ptD}, {10.0}, {true, {WW}});
    cTopoObs(obsSet1, TopoObsType::dist, std::vector{ptC, ptD}, {10.0}, {true, {WW}});
    cTopoObs(obsSet1, TopoObsType::dist, std::vector{ptC, ptD}, {10.1}, {true, {0.1}});

    //add point E to an unknown common dist
    allObsSets.push_back(make_TopoObsSet<cTopoObsSetDistParam>());
    auto obsSet2 = allObsSets[1].get();
    allPts.push_back(new cTopoPoint("ptE", cPt3dr(11,11,11), true));
    auto ptE = allPts[4];
    cTopoObs(obsSet2, TopoObsType::distParam, std::vector{ptE, ptA}, {}, {true, {WW}});
    cTopoObs(obsSet2, TopoObsType::distParam, std::vector{ptE, ptB}, {}, {true, {WW}});
    cTopoObs(obsSet2, TopoObsType::distParam, std::vector{ptE, ptC}, {}, {true, {WW}});
    cTopoObs(obsSet2, TopoObsType::distParam, std::vector{ptE, ptD}, {}, {true, {WW}});

    //add subframe obs
    allObsSets.push_back(make_TopoObsSet<cTopoObsSetSubFrame>());
    auto obsSet3 = allObsSets[2].get();
    cTopoObs(obsSet3, TopoObsType::subFrame, std::vector{ptE, ptA}, {-5., -3.75, -1.4}, {true, {WW,WW,WW}});
    cTopoObs(obsSet3, TopoObsType::subFrame, std::vector{ptE, ptB}, { 5., -3.75, -1.4}, {true, {WW,WW,WW}});
    cTopoObs(obsSet3, TopoObsType::subFrame, std::vector{ptE, ptC}, { 0.,  6.25, -1.4}, {true, {WW,WW,WW}});
    cTopoObs(obsSet3, TopoObsType::subFrame, std::vector{ptE, ptD}, { 0.,  0.,    6.4}, {true, {WW,WW,WW}});
}

double cTopoComp::getSigma0() const
{
    if (verbose) StdOut()<<" res: [";
    double resid2 = 0.0;
    for (auto &obsSet: allObsSets)
        for (size_t i=0;i<obsSet->nbObs();++i)
        {
            cTopoObs* obs = obsSet->getObs(i);
            auto res = obs->getResiduals(this);
            auto sigmas = obs->getWeights().getSigmas();
            for (size_t j=0;j<res.size();++j)
            {
                double obs_res = res[j]/sigmas[j];
                resid2 += obs_res*obs_res;
            }
            if (verbose) for (size_t j=0;j<res.size();++j) StdOut()<<res[j]<<" ";
        }
    if (verbose) StdOut()<<"]" << std::endl;
    return sqrt(resid2/(mSys->GetNbObs()-mSys->CountFreeVariables()));
}

bool cTopoComp::OneIteration()
{
    if (!isInit) initializeLeastSquares();
    if (verbose) StdOut()  << "iter" << std::endl;
    //add points constraints
    for (auto & pt: allPts)
        pt->addConstraints(this);

    //add observations
    for (auto &obsSet: allObsSets)
        for (size_t i=0;i<obsSet->nbObs();++i)
        {
            cTopoObs* obs = obsSet->getObs(i);
            mSys->CalcAndAddObs(getEquation(obs->getType()), obs->getIndices(), obs->getVals(), obs->getWeights());
        }

    //solve
    try
    {
        const auto & aVectSol = mSys->SolveUpdateReset();
        mSetIntervMultObj->SetVUnKnowns(aVectSol); //update params
        if (verbose) StdOut()  <<  " sol: " <<  mSys->CurGlobSol() << std::endl;
    } catch(...) {
        StdOut()  <<  " Error solving system..." << std::endl;
        return false;
    }

    if (verbose)
    {
        StdOut()<<" nb obs: "<<mSys->GetNbObs()<<std::endl;
        double sigma0 = getSigma0();
        StdOut()<<" sigma0: "<<sigma0<<std::endl;
    }
    return true;
}

//-------------------------------------------------------------------


void BenchTopoComp(cParamExeBench & aParam)
{
    if (! aParam.NewBench("TopoComp")) return;

    cTopoComp  aTopoComp;
    aTopoComp.createEx1();
    for (int iter=0; iter<5; ++iter)
    {
        //StdOut()<<"Iter "<<iter<<std::endl;
        //mTopoComp.print();
        if (!aTopoComp.OneIteration()) break;
    }
    auto targetSigma0 = 0.35271;
    MMVII_INTERNAL_ASSERT_bench(std::abs(aTopoComp.getSigma0()-targetSigma0)<1e-5,"TopoComp sigma0 final");

    aParam.EndBench();
    return;
}

//-------------------------------------------------------------------

/*cAppli_TopoComp::cAppli_TopoComp(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli  (aVArgs,aSpec)
{
}

cCollecSpecArg2007 & cAppli_TopoComp::ArgObl(cCollecSpecArg2007 & anArgObl)
{
 return
      anArgObl

   ;
}

cCollecSpecArg2007 & cAppli_TopoComp::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return anArgOpt

   ;
}


int  cAppli_TopoComp::Exe()
{

    StdOut() << "-------------------------------------------------------------------" << std::endl;

    mTopoComp.createEx1();
    for (int iter=0; iter<5; ++iter)
    {
        StdOut()<<"Iter "<<iter<<std::endl;
        //mTopoComp.print();
        if (!mTopoComp.OneIteration()) break;
    }

    StdOut() << "-------------------------------------------------------------------" << std::endl;

   return EXIT_SUCCESS;
}

//-------------------------------------------------------------------

tMMVII_UnikPApli Alloc_TopoComp(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_TopoComp(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpecTopoComp
(
     "TopoComp",
      Alloc_TopoComp,
      "Topometric compensation",
      {eApF::Topo},
      {eApDT::ToDef},
      {eApDT::ToDef},
      __FILE__
);
*/
};
