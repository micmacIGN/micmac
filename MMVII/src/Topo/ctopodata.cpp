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
    AddOptData(anAux,"LastResiduals",mLastResiduals);
}

void AddData(const cAuxAr2007 & anAux, cTopoObsData & aTopoObs)
{
     aTopoObs.AddData(anAux);
}


// -------------------------

void cTopoObsSetData::AddData(const  cAuxAr2007 & anAuxInit)
{
    cAuxAr2007 anAux("TopoObsSetData",anAuxInit);
    //std::cout<<"Add data obs set '"<<toString()<<"'"<<std::endl;
    MMVII::EnumAddData(anAux,mType,"Type");
    MMVII::AddData(cAuxAr2007("AllObs",anAux),mObs);

    AddOptData(anAux,"StationIsVericalized",mStationIsVericalized);
    AddOptData(anAux,"StationIsOriented",mStationIsOriented);
    AddOptData(anAux,"G0",mStationG0);
    //AddOptData(anAux,"StationRot",mRotVert2Instr); // TODO
}


void AddData(const cAuxAr2007 & anAux, cTopoObsSetData &aObsSet)
{
     aObsSet.AddData(anAux);
}


// ------------------------------------

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
        switch (aSet->mType) {
        case eTopoObsSetType::eStation:
        {
            cTopoObsSetStation* set = dynamic_cast<cTopoObsSetStation*>(aSet);
            if (!set)
                MMVII_INTERNAL_ERROR("error set type")
            aSetData.mStationIsOriented = set->isOriented();
            aSetData.mStationIsVericalized = set->isVericalized();
            aSetData.mStationG0 = set->getG0();
            aSetData.mRotVert2Instr = set->getRotVert2Instr();
            break;
        }
        default:
            MMVII_INTERNAL_ERROR("unknown obs set type")
        }

        for (auto & aObs : aSet->mObs)
        {
            cTopoObsData aObsData = {
                aObs->mType, aObs->mPtsNames, aObs->mMeasures,
                aObs->mWeights.getSigmas(), aObs->getResiduals()
            };
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
    case eCompObsTypes::eCompHz:
        aObsData = {eTopoObsType::eHz, {nameFrom,nameTo}, {val}, {sigma}};
        break;
    case eCompObsTypes::eCompZen:
        aObsData = {eTopoObsType::eZen, {nameFrom,nameTo}, {val}, {sigma}};
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
    cTopoObsData aObs1 = {eTopoObsType::eDist, {"ptD", "ptA"},  {10.}, {WW}};
    cTopoObsData aObs2 = {eTopoObsType::eDist, {"ptD", "ptB"},  {10.}, {WW}};
    cTopoObsData aObs3 = {eTopoObsType::eDist, {"ptD", "ptC"},  {10.}, {WW}};
    cTopoObsData aObs4 = {eTopoObsType::eDist, {"ptD", "ptC"},  {10+WW}, {WW}};

    cTopoObsSetData aSet1;
    aSet1.mType = eTopoObsSetType::eStation;
    aSet1.mObs = {aObs1, aObs2, aObs3, aObs4};

    cTopoData aTopoData;
    aTopoData.mAllPoints = {ptA, ptB, ptC, ptD};
    aTopoData.mAllObsSets = {aSet1};
    return aTopoData;
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
    cTopoPointData aPt1 = {"Ori1",cPt3dr(100,110,100),false,cPt3dr(0.01,0.01,0.01)};
    cTopoPointData aPt2 = {"St1",cPt3dr(100,100,100),false,cPt3dr(0.01,0.01,0.01)};
    cTopoPointData aPt3 = {"Tr1",cPt3dr(105,115,105),true}; // 107.072, 107.072, 100

    double g0 = 2.2;
    cTopoObsData aObs1 = {eTopoObsType::eHz, {"St1", "Ori1"},  {0. - g0}, {0.001}};
    cTopoObsData aObs2 = {eTopoObsType::eHz, {"St1", "Tr1"},  {M_PI/4. - g0}, {0.001}};
    cTopoObsData aObs3 = {eTopoObsType::eZen, {"St1", "Tr1"},  {0.}, {0.001}};
    cTopoObsData aObs4 = {eTopoObsType::eDist, {"St1", "Tr1"},  {10.}, {0.001}};
    cTopoObsData aObs5 = {eTopoObsType::eDist, {"St1", "Tr1"},  {10.002}, {0.001}};

    cTopoObsSetData aSet1;
    aSet1.mType = eTopoObsSetType::eStation;
    aSet1.mStationIsVericalized = true;
    aSet1.mStationIsOriented = false;
    aSet1.mObs = {aObs1, aObs2, aObs3, aObs4, aObs5};

    cTopoData aTopoData;
    aTopoData.mAllPoints = {aPt1, aPt2, aPt3};
    aTopoData.mAllObsSets = {aSet1};

    return aTopoData;
}

};
