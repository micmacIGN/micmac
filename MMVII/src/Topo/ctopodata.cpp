#include "ctopodata.h"
#include "MMVII_PhgrDist.h"
#include "ctopopoint.h"
#include "MMVII_2Include_Serial_Tpl.h"
#include <memory>
namespace MMVII
{

cTopoData::cTopoData():
    mSetIntervMultObj(new cSetInterUK_MultipeObj<double>()), mSys(nullptr),
    mTopoObsType2equation
    {
        {eTopoObsType::eDist, EqDist3D(true,1)},
        {eTopoObsType::eSubFrame, EqTopoSubFrame(true,1)},
        {eTopoObsType::eDistParam, EqDist3DParam(true,1)},
    }
{
    createEx1();
    print();
}

cTopoData::~cTopoData()
{
    std::for_each(mTopoObsType2equation.begin(), mTopoObsType2equation.end(), [](auto &e){ delete e.second; });
    delete mSys;
    delete mSetIntervMultObj;
    std::for_each(allPts.begin(), allPts.end(), [](auto p){ delete p; });
}

cTopoData * cTopoData::FromFile(const std::string & aNameFile)
{
    cTopoData * aRes = new cTopoData;
    // ReadFromFile(aRes,aNameFile); // TODO
    return aRes;
}

void cTopoData::AddData(const  cAuxAr2007 & anAuxInit)
{
     cAuxAr2007 anAux("TopoData",anAuxInit);

     MMVII::AddData(cAuxAr2007("AllPts",anAux),allPts);
     MMVII::AddData(cAuxAr2007("AllObsSets",anAux),allObsSets);
}

void AddData(const cAuxAr2007 & anAux, cTopoData & aTopoData)
{
     aTopoData.AddData(anAux);
}

void cTopoData::ToFile(const std::string & aName) const
{
    SaveInFile(*this,aName);
}


cCalculator<double>*  cTopoData::getEquation(eTopoObsType tot) const {
    auto eq = mTopoObsType2equation.find(tot);
    if (eq != mTopoObsType2equation.end())
        return mTopoObsType2equation.at(tot);
    else
    {
        MMVII_INTERNAL_ERROR("unknown equation for obs type")
        return nullptr;
    }
}

void cTopoData::print()
{
    std::cout<<"Points:\n";
    for (auto& pt: allPts)
        std::cout<<" - "<<pt->toString()<<"\n";
    std::cout<<"ObsSets:\n";
    for (auto &obsSet: allObsSets)
        std::cout<<" - "<<obsSet.get()->toString()<<"\n";
}

void cTopoData::createEx1()
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
    cTopoObs(obsSet1, eTopoObsType::eDist, std::vector{ptA, ptD}, {10.0}, {true, {WW}});
    cTopoObs(obsSet1, eTopoObsType::eDist, std::vector{ptB, ptD}, {10.0}, {true, {WW}});
    cTopoObs(obsSet1, eTopoObsType::eDist, std::vector{ptC, ptD}, {10.0}, {true, {WW}});
    cTopoObs(obsSet1, eTopoObsType::eDist, std::vector{ptC, ptD}, {10.1}, {true, {0.1}});
    //add point E to an unknown common dist
    allObsSets.push_back(make_TopoObsSet<cTopoObsSetDistParam>());
    auto obsSet2 = allObsSets[1].get();
    allPts.push_back(new cTopoPoint("ptE", cPt3dr(11,11,11), true));
    auto ptE = allPts[4];
    cTopoObs(obsSet2, eTopoObsType::eDistParam, std::vector{ptE, ptA}, {}, {true, {WW}});
    cTopoObs(obsSet2, eTopoObsType::eDistParam, std::vector{ptE, ptB}, {}, {true, {WW}});
    cTopoObs(obsSet2, eTopoObsType::eDistParam, std::vector{ptE, ptC}, {}, {true, {WW}});
    cTopoObs(obsSet2, eTopoObsType::eDistParam, std::vector{ptE, ptD}, {}, {true, {WW}});

    //add subframe obs
    allObsSets.push_back(make_TopoObsSet<cTopoObsSetSubFrame>());
    auto obsSet3 = allObsSets[2].get();
    cTopoObs(obsSet3, eTopoObsType::eSubFrame, std::vector{ptE, ptA}, {-5., -3.75, -1.4}, {true, {WW,WW,WW}});
    cTopoObs(obsSet3, eTopoObsType::eSubFrame, std::vector{ptE, ptB}, { 5., -3.75, -1.4}, {true, {WW,WW,WW}});
    cTopoObs(obsSet3, eTopoObsType::eSubFrame, std::vector{ptE, ptC}, { 0.,  6.25, -1.4}, {true, {WW,WW,WW}});
    cTopoObs(obsSet3, eTopoObsType::eSubFrame, std::vector{ptE, ptD}, { 0.,  0.,    6.4}, {true, {WW,WW,WW}});
}


//-------------------------------------------------------------------


};
