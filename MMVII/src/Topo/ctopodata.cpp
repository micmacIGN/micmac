#include "ctopodata.h"
#include "MMVII_PhgrDist.h"
#include "ctopopoint.h"
#include "MMVII_2Include_Serial_Tpl.h"
#include "Topo.h"
#include <memory>
#include <fstream>
#include <sstream>

namespace MMVII
{

cTopoData::cTopoData(const std::string & aName, cBA_Topo *aBA_Topo):
    mBA_Topo(aBA_Topo), mUniqueObsSetSimple(nullptr)
   /* mSetIntervMultObj(new cSetInterUK_MultipeObj<double>()), mSys(nullptr),
    mTopoObsType2equation
    {
        {eTopoObsType::eDist, EqDist3D(true,1)},
        {eTopoObsType::eSubFrame, EqTopoSubFrame(true,1)},
        {eTopoObsType::eDistParam, EqDist3DParam(true,1)},
    }*/
{
    // create the unique ObsSetSimple
    allObsSets.push_back(make_TopoObsSet<cTopoObsSetSimple>());
    mUniqueObsSetSimple = allObsSets.at(0).get();

    bool ok = FromCompFile(aName);
    MMVII_INTERNAL_ASSERT_strong(ok, "Topo: error reading file "+aName);
    //createEx2();
    print();
}

cTopoData::~cTopoData()
{
    /*std::for_each(mTopoObsType2equation.begin(), mTopoObsType2equation.end(), [](auto &e){ delete e.second; });
    delete mSys;
    delete mSetIntervMultObj;
    std::for_each(allPts.begin(), allPts.end(), [](auto p){ delete p; });*/
}

void cTopoData::AddData(const  cAuxAr2007 & anAuxInit)
{
     cAuxAr2007 anAux("TopoData",anAuxInit);

     //MMVII::AddData(cAuxAr2007("AllPts",anAux),allPts);
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

//void cTopoData::FromFile(const std::string & aName)
//{
//    ReadFromFile(*this, aName);
//}


bool cTopoData::FromCompFile(const std::string & aName)
{
    std::ifstream infile(aName);
    if (infile.bad())
    {
        StdOut() << "Error: can't open file \""<<aName<<"\""<<std::endl;
        return false;
    }

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
            StdOut() << "Error reading line " << line_num << ": \""<<aName<<"\"";
            continue;
        }
        if (!addObs(code, nameFrom, nameTo, val, sigma))
            StdOut() << "Error interpreting line " << line_num << ": \""<<aName<<"\"";
    }
    long nb_obs = 0;
    for (const auto & set : allObsSets)
        for (const auto & obs : set->getAllObs())
            nb_obs += obs->getVals().size();
    StdOut() << "Reading file finished. " << nb_obs << " obs found." << std::endl;
    return true;
}

bool cTopoData::addObs(int code, const std::string & nameFrom, const std::string & nameTo, double val, double sigma)
{
    // TODO

    //tmp
    switch (code) {
    case 3:
        mUniqueObsSetSimple->addObs(eTopoObsType::eDist, {nameFrom, nameTo}, {val}, {true, {sigma} });
        return true;
    default:
        StdOut() << "Error, unknown obs code " << code <<".\n";
    }
    return false;
}

/*cCalculator<double>*  cTopoData::getEquation(eTopoObsType tot) const {
    auto eq = mTopoObsType2equation.find(tot);
    if (eq != mTopoObsType2equation.end())
        return mTopoObsType2equation.at(tot);
    else
    {
        MMVII_INTERNAL_ERROR("unknown equation for obs type")
        return nullptr;
    }
}*/

void cTopoData::print()
{
    std::cout<<"Points:\n";
    //for (auto& pt: allPts)
    //    std::cout<<" - "<<pt->toString()<<"\n";
    std::cout<<"ObsSets:\n";
    for (auto &obsSet: allObsSets)
        std::cout<<" - "<<obsSet.get()->toString()<<"\n";
}


/*void cTopoData::createEx1()
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
}

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


};
