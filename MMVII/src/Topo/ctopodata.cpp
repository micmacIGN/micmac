#include "ctopodata.h"
#include "MMVII_PhgrDist.h"
#include "ctopopoint.h"
#include "ctopoobsset.h"
#include "ctopoobs.h"
#include "MMVII_2Include_Serial_Tpl.h"
#include "Topo.h"
#include <memory>
#include <fstream>
#include <sstream>

namespace MMVII
{


void cTopoPointData::AddData(const  cAuxAr2007 & anAuxInit)
{
    //std::cout<<"Add data point '"<<toString()<<"'"<<std::endl;
    cAuxAr2007 anAux("TopoPointData",anAuxInit);

    MMVII::AddData(cAuxAr2007("Name",anAux),mName);
    MMVII::AddData(cAuxAr2007("InitCoord",anAux),mInitCoord);
    MMVII::AddData(cAuxAr2007("IsFree",anAux),mIsFree);
    MMVII::AddData(cAuxAr2007("Sigmas",anAux),mSigmas);
    AddOptData(anAux,"VertDefl",mVertDefl);
    AddOptData(anAux,"FinalCoord",mFinalCoord);
}

void AddData(const cAuxAr2007 & anAux, cTopoPointData & aTopoObs)
{
     aTopoObs.AddData(anAux);
}


std::string cTopoPointData::toString()
{
    std::ostringstream oss;
    oss<<"TopoPoint "<<mName<<" "<<mInitCoord;
    if (mFinalCoord.has_value())
        oss<<" => "<<*mFinalCoord;
    return  oss.str();
}


// -------------------------

void cTopoObsData::AddData(const  cAuxAr2007 & anAuxInit)
{
    //std::cout<<"Add data obs '"<<toString()<<"'"<<std::endl;
    cAuxAr2007 anAux("TopoObsData",anAuxInit);

    MMVII::EnumAddData(anAux,mType,"Type");
    MMVII::AddData(cAuxAr2007("Pts",anAux),mPtsNames);
    MMVII::AddData(cAuxAr2007("Measures",anAux),mMeasures);
    MMVII::AddData(cAuxAr2007("Sigmas",anAux),mSigmas);
}

void AddData(const cAuxAr2007 & anAux, cTopoObsData & aTopoObs)
{
     aTopoObs.AddData(anAux);
}


// -------------------------
cTopoData::cTopoData(cBA_Topo* aBA_topo)
{
    for (auto & [aName, aPt] : aBA_topo->getAllPts())
    {
        cTopoPointData aPtData = {aName, aPt.mInitCoord, aPt.mIsFree, aPt.mSigmas,
                                 aPt.mVertDefl, std::nullopt};
        if (aPt.isReady())
            aPtData.mFinalCoord = *aPt.getPt();
        mAllPoints.push_back(aPtData);
    }

    for (auto & aSet : aBA_topo->mAllObsSets)
    {
        cTopoObsSetData aSetData;
        aSetData.mType = aSet->mType;
        for (auto & aObs : aSet->mObs)
        {
            cTopoObsData aObsData = {aObs->mType, aObs->mPtsNames, aObs->mMeasures, aObs->mWeights.getSigmas()};
            aSetData.mObs.push_back(aObsData);
        }
        mAllObsSets.push_back(aSetData);
    }
}

void cTopoData::AddData(const  cAuxAr2007 & anAuxInit)
{
     cAuxAr2007 anAux("TopoData",anAuxInit);

     MMVII::AddData(cAuxAr2007("AllPts",anAux),mAllPoints);
     MMVII::AddData(cAuxAr2007("AllObsSets",anAux),mAllObsSets);
}

void AddData(const cAuxAr2007 & anAux, cTopoData & aTopoData)
{
     aTopoData.AddData(anAux);
}

void cTopoData::ToFile(const std::string & aName) const
{
    SaveInFile(*this,aName);
}

void cTopoData::FromFile(const std::string & aName)
{
    ReadFromFile(*this, aName);
}

bool cTopoData::FromCompFile(const std::string & aName)
{
    std::ifstream infile(aName);
    if (infile.bad())
    {
        StdOut() << "Error: can't open file \""<<aName<<"\""<<std::endl;
        return false;
    }

    std::set<std::string> allPointsNames;

    StdOut() << "Reading file \""<<aName<<"\"..."<<std::endl;
    std::string line;
    int line_num = 0;
    while (std::getline(infile, line))
    {
        ++line_num;
        std::istringstream iss(line);
        int code;
        if (!(iss >> code)) continue; // line ignored
        std::string nameFrom, nameTo;
        double val, sigma;
        if (!(iss >> nameFrom >> nameTo >> val >> sigma))
        {
            StdOut() << "Error reading line " << line_num << ": \""<<aName<<"\"\n";
            continue;
        }
        allPointsNames.insert(nameFrom);
        allPointsNames.insert(nameTo);
        if (!addObs((eCompObsTypes)code, nameFrom, nameTo, val, sigma))
            StdOut() << "Error interpreting line " << line_num << ": \""<<aName<<"\"\n";
    }
    for (const auto & aName : allPointsNames)
    {
        cTopoPointData aPtData = {aName, {0.,0.,0.}, true, {0.,0.,0.}, std::nullopt, std::nullopt};
        mAllPoints.push_back(aPtData);
    }

    long nb_obs = 0;
    for (const auto & set : mAllObsSets)
        for (const auto & obs : set.mObs)
            nb_obs += obs.mMeasures.size();
    StdOut() << "Reading file finished. " << mAllPoints.size() << " points and " << nb_obs << " obs found." << std::endl;

    return true;
}

bool cTopoData::addObs(eCompObsTypes code, const std::string & nameFrom, const std::string & nameTo, double val, double sigma)
{
    cTopoObsSetData * aSetDataStation = nullptr;
    // search for a suitable set
    // TODO: create new set if open circle
    for (auto &aObsSet : mAllObsSets)
    {
        // TODO: must search from end
        if ((!aObsSet.mObs.empty()) && (aObsSet.mObs.at(0).mPtsNames.at(0) == nameFrom))
        {
            aSetDataStation = &aObsSet;
            break;
        }
    }
    if (!aSetDataStation)
    {
        mAllObsSets.push_back( {} );
        aSetDataStation = &mAllObsSets.back();
        aSetDataStation->mType = eTopoObsSetType::eStation;
    }

    cTopoObsData aObsData;

    switch (code) {
    case eCompObsTypes::eCompDist:
        aObsData = {eTopoObsType::eDist, {nameFrom,nameTo}, {val}, {sigma}};
        break;
    default:
        StdOut() << "Error, unknown obs code " << (int)code <<".\n";
        return false;
    }
    aSetDataStation->mObs.push_back(aObsData);
    return true;
}


void cTopoData::print()
{
    //std::cout<<"Points:\n";
    //for (auto& [aName, aPtT] : mAllPts)
    //    std::cout<<" - "<<aPtT.toString()<<"\n";
    //std::cout<<"ObsSets:\n";
    //for (auto &obsSet: mAllObsSets)
    //    std::cout<<" - "<<obsSet.toString()<<"\n";
}

/*
void cTopoData::addTopoPoint(cTopoPoint aPtT)
{
    MMVII_INTERNAL_ASSERT_User(mAllPts.count(aPtT.getName())==0, eTyUEr::eUnClassedError, "Error: TopoPoint " + aPtT.getName() + " already in TopoData.")
    mAllPts[aPtT.getName()] = aPtT;
}
*/

cTopoData cTopoData::createEx1()
{
    //verbose = true;

    //create fixed points
    cTopoPointData ptA = {"ptA",cPt3dr(10,10,10),false};
    cTopoPointData ptB = {"ptB",cPt3dr(20,10,10),false};
    cTopoPointData ptC = {"ptC",cPt3dr(15,20,10),false};
#define WW 0.01
    // unknown point
    cTopoPointData ptD = {"ptD",cPt3dr(14, 14, 14),true};// 14, 14, 14 // 15.000000, 13.744999, 17.80863
    // distances to fixed points
    /*cTopoObsData aObs1 = {eTopoObsType::eDist, {"ptA", "ptD"},  {10.}, {WW}};
    cTopoObsData aObs2 = {eTopoObsType::eDist, {"ptB", "ptD"},  {10.}, {WW}};
    cTopoObsData aObs3 = {eTopoObsType::eDist, {"ptC", "ptD"},  {10.}, {WW}};
    cTopoObsData aObs4 = {eTopoObsType::eDist, {"ptC", "ptD"},  {10+WW}, {WW}};*/
    cTopoObsData aObs1 = {eTopoObsType::eDist, {"ptA", "ptD"},  {10.}, {WW}};
    cTopoObsData aObs2 = {eTopoObsType::eDist, {"ptD", "ptB"},  {10.}, {WW}}; // error when previous LinearConstraint bug
    cTopoObsData aObs3 = {eTopoObsType::eDist, {"ptC", "ptD"},  {10.}, {WW}};
    cTopoObsData aObs4 = {eTopoObsType::eDist, {"ptC", "ptD"},  {10+WW}, {WW}};
    /*cTopoObsData aObs1 = {eTopoObsType::eDist, {"ptD", "ptA"},  {10.}, {WW}};
    cTopoObsData aObs2 = {eTopoObsType::eDist, {"ptD", "ptB"},  {10.}, {WW}};
    cTopoObsData aObs3 = {eTopoObsType::eDist, {"ptD", "ptC"},  {10.}, {WW}};
    cTopoObsData aObs4 = {eTopoObsType::eDist, {"ptD", "ptC"},  {10+WW}, {WW}};*/

    cTopoObsSetData aSet1;
    aSet1.mType = eTopoObsSetType::eStation;
    aSet1.mObs = {aObs1};
    cTopoObsSetData aSet2;
    aSet2.mType = eTopoObsSetType::eStation;
    aSet2.mObs = {aObs2};
    cTopoObsSetData aSet3;
    aSet3.mType = eTopoObsSetType::eStation;
    aSet3.mObs = {aObs3, aObs4};


    cTopoData aTopoData;
    aTopoData.mAllPoints = {ptA, ptB, ptC, ptD};
    aTopoData.mAllObsSets = {aSet1, aSet2, aSet3};
    return aTopoData;
/*
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
    obsSet1->addObs(eTopoObsType::eDist, std::vector{ptA, ptD}, {10.0}, {true, {WW}});
    obsSet1->addObs(eTopoObsType::eDist, std::vector{ptB, ptD}, {10.0}, {true, {WW}});
    obsSet1->addObs(eTopoObsType::eDist, std::vector{ptC, ptD}, {10.0}, {true, {WW}});
    obsSet1->addObs(eTopoObsType::eDist, std::vector{ptC, ptD}, {10.1}, {true, {0.1}});
    //add point E to an unknown common dist
    allObsSets.push_back(make_TopoObsSet<cTopoObsSetDistParam>());
    auto obsSet2 = allObsSets[1].get();
    allPts.push_back(new cTopoPoint("ptE", cPt3dr(11,11,11), true));
    auto ptE = allPts[4];
    obsSet2->addObs(eTopoObsType::eDistParam, std::vector{ptE, ptA}, {}, {true, {WW}});
    obsSet2->addObs(eTopoObsType::eDistParam, std::vector{ptE, ptB}, {}, {true, {WW}});
    obsSet2->addObs(eTopoObsType::eDistParam, std::vector{ptE, ptC}, {}, {true, {WW}});
    obsSet2->addObs(eTopoObsType::eDistParam, std::vector{ptE, ptD}, {}, {true, {WW}});

    //add subframe obs
    allObsSets.push_back(make_TopoObsSet<cTopoObsSetSubFrame>());
    auto obsSet3 = allObsSets[2].get();
    obsSet3->addObs(eTopoObsType::eSubFrame, std::vector{ptE, ptA}, {-5., -3.75, -1.4}, {true, {WW,WW,WW}});
    obsSet3->addObs(eTopoObsType::eSubFrame, std::vector{ptE, ptA}, { 5., -3.75, -1.4}, {true, {WW,WW,WW}});
    obsSet3->addObs(eTopoObsType::eSubFrame, std::vector{ptE, ptA}, { 0.,  6.25, -1.4}, {true, {WW,WW,WW}});
    obsSet3->addObs(eTopoObsType::eSubFrame, std::vector{ptE, ptA}, { 0.,  0.,    6.4}, {true, {WW,WW,WW}});
    */
}

/*
void cTopoData::createEx2()
{
    auto from_name = "DSCF3297_L.jpg";
    auto to_name = "DSCF3298_L.jpg";
    // auto ptFrom = mBA_Topo->getPointWithUK(from_name);
    // auto ptTo = mBA_Topo->getPointWithUK(to_name);

    allObsSets.push_back(make_TopoObsSet<cTopoObsSetSimple>());
    auto obsSet1 = allObsSets[0].get();
    obsSet1->addObs(eTopoObsType::eDist, {from_name, to_name}, {0.3170}, {true, {0.001} });

}*/

cTopoData cTopoData::createEx3()
{
    cTopoPointData aPt1 = {"St1",cPt3dr(100,100,100),false,cPt3dr(0.01,0.01,0.01)};
    cTopoPointData aPt2 = {"Tr1",cPt3dr(105,115,105),true};

    cTopoObsData aObs1 = {eTopoObsType::eHz, {"St1", "Tr1"},  {M_PI/2.}, {0.001}};
    cTopoObsData aObs2 = {eTopoObsType::eZen, {"St1", "Tr1"},  {0.}, {0.001}};
    cTopoObsData aObs3 = {eTopoObsType::eDist, {"St1", "Tr1"},  {10.}, {0.001}};
    cTopoObsData aObs4 = {eTopoObsType::eDist, {"St1", "Tr1"},  {10.002}, {0.001}};

    cTopoObsSetData aSet1;
    aSet1.mType = eTopoObsSetType::eStation;
    aSet1.mObs = {aObs1, aObs2, aObs3, aObs4};

    cTopoObsData aObs5 = {eTopoObsType::eDist, {"Tr1", "St1"},  {10.002}, {0.001}};
    cTopoObsSetData aSet2;
    aSet2.mType = eTopoObsSetType::eStation;
    aSet2.mObs = {aObs5};

    cTopoData aTopoData;
    aTopoData.mAllPoints = {aPt1, aPt2};
    aTopoData.mAllObsSets = {aSet1, aSet2};

    return aTopoData;
}

};
