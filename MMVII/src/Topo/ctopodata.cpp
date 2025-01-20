#include "ctopodata.h"
#include "MMVII_PhgrDist.h"
#include "MMVII_2Include_Serial_Tpl.h"
#include "MMVII_Topo.h"
#include <memory>
#include <fstream>
#include <sstream>

namespace MMVII
{

void cTopoObsData::AddData(const  cAuxAr2007 & anAuxInit)
{
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
    MMVII::EnumAddData(anAux,mType,"Type");
    MMVII::AddData(cAuxAr2007("AllObs",anAux),mObs);

    AddSupData(anAux);
}


void AddData(const cAuxAr2007 & anAux, cTopoObsSetData &aObsSet)
{
     aObsSet.AddData(anAux);
}

cTopoObsSetStationData::cTopoObsSetStationData()
{
    mType = eTopoObsSetType::eSimple;
}

void cTopoObsSetStationData::AddSupData(const  cAuxAr2007 & anAux)
{
    MMVII::EnumAddData(anAux,mStationOriStat,"StationOriStatus");
    AddOptData(anAux,"Out_G0",mStationG0);
}


// ------------------------------------

cTopoData::cTopoData(const cBA_Topo* aBA_topo)
{
    for (auto & aSet : aBA_topo->mAllObsSets)
    {
        std::vector <cTopoObsData> aVectObsData;
        for (auto & aObs : aSet->mObs)
        {
            cTopoObsData aObsData = {
                aObs->mType, aObs->mPtsNames, aObs->mMeasures,
                aObs->mWeights.getSigmas(), aObs->getResiduals()
            };
            aVectObsData.push_back(aObsData);
        }

        switch (aSet->mType) {
        case eTopoObsSetType::eSimple:
        {
            for (auto & aObs : aVectObsData)
                mObsSetSimple.mObs.push_back( aObs );
            break;
        }
        case eTopoObsSetType::eStation:
        {
            cTopoObsSetStation* set = dynamic_cast<cTopoObsSetStation*>(aSet);
            if (!set)
                MMVII_INTERNAL_ERROR("error set type")
            cTopoObsSetStationData aSetStationData;
            aSetStationData.mStationOriStat = set->getOriStatus();
            aSetStationData.mStationG0 = set->getG0();
            aSetStationData.mRotVert2Instr = set->getRotVert2Instr();
            for (auto & aObs : aVectObsData)
                aSetStationData.mObs.push_back( aObs );
            mAllObsSetStations.push_back( aSetStationData );
            break;
        }
        case eTopoObsSetType::eNbVals:
            MMVII_INTERNAL_ERROR("unknown obs set type")
        }
    }
}

void cTopoData::AddData(const  cAuxAr2007 & anAuxInit)
{
     cAuxAr2007 anAux("TopoData",anAuxInit);
     MMVII::AddData(cAuxAr2007("AllObsSetStations",anAux),mAllObsSetStations);
     MMVII::AddData(cAuxAr2007("ObsSetSimple",anAux),mObsSetSimple);

}

void AddData(const cAuxAr2007 & anAux, cTopoData & aTopoData)
{
     aTopoData.AddData(anAux);
}

void cTopoData::ToFile(const std::string & aFileName) const
{
    SaveInFile(*this,aFileName);
}

void cTopoData::FromFile(const std::string & aFileName)
{
    ReadFromFile(*this, aFileName);
}

void cTopoData::InsertTopoData(const cTopoData & aOtherTopoData)
{
    for (auto& aObsSet: aOtherTopoData.mAllObsSetStations)
        mAllObsSetStations.push_back(aObsSet);
    for (auto& aObs: aOtherTopoData.mObsSetSimple.mObs)
        mObsSetSimple.mObs.push_back(aObs);
}


eCompObsType intToCompObsType(int i)
{
    eCompObsType res = static_cast<eCompObsType>(i);

    switch(res) {
    case eCompObsType::eCompError:
    case eCompObsType::eCompDX:
    case eCompObsType::eCompDY:
    case eCompObsType::eCompDZ:
    case eCompObsType::eCompDH:
    case eCompObsType::eCompDist:
    case eCompObsType::eCompHz:
    case eCompObsType::eCompHzOpen:
    case eCompObsType::eCompZen:
        return res;
    }
    return eCompObsType::eCompError;
}

/**
 * @brief cleanCompLine remove heading spaces, and trailing comments adn spaces
 * @param str
 */
void cleanCompLine( std::string& str)
{
    auto pos_start = str.find_first_not_of(" \t\n\r\f\v");
    if (pos_start == std::string::npos)
    {
        str = "";
        //std::cout<<"line cleaned: \""<<str<<"\"\n";
        return;
    }
    auto pos_comment = str.find_first_of("*");
    if (pos_comment == std::string::npos)
        pos_comment = str.size();
    auto pos_end = pos_comment>pos_start?str.find_last_not_of(" \t\n\r\f\v", pos_comment - 1)+1:0;
    //std::cout<<pos_start<<" "<<pos_comment<<" "<<pos_end<<std::endl;
    if (pos_end > pos_start)
        str = str.substr(pos_start, pos_end - pos_start);
    else
        str = "";
    //std::cout<<"line cleaned: \""<<str<<"\"\n";
}

bool cTopoData::InsertCompObsFile(const std::string & aFileName)
{
    int aNbNewObs = 0;
    cTopoObsSetData aCurrentObsSetSimple;
    std::vector<cTopoObsSetStationData> aCurrentVectObsSetStations;
    eTopoStOriStat aCurrStationStatus = eTopoStOriStat::eTopoStOriVert;

    std::ifstream infile(aFileName);
    if (!infile.is_open())
    {
        MMVII_INTERNAL_ERROR("Error: can't open obs file \""+aFileName+"\"")
        return false;
    }

    StdOut() << "Reading obs file \""<<aFileName<<"\"..."<<std::endl;
    std::string line;
    int line_num = 0;
    while (std::getline(infile, line))
    {
        ++line_num;
        cleanCompLine(line);
        if (line.empty())
            continue;

        std::istringstream iss(line);

        eTopoStOriStat aNewStationStatus = Str2E<eTopoStOriStat>(line, true);

        if (aNewStationStatus != eTopoStOriStat::eNbVals)
        {
            addObsSets(aCurrentVectObsSetStations); // if a station status is given, use only new stations
            if (aNewStationStatus != eTopoStOriStat::eTopoStOriContinue)
                aCurrStationStatus = aNewStationStatus;
            continue;
        }

        int code;
        if (!(iss >> code))
        {
            MMVII_INTERNAL_ERROR("Error reading "+aFileName+" at line "+std::to_string(line_num)+": \""+line+"\"")
            continue;
        }

        eCompObsType code_comp  = intToCompObsType(code);
        // Check if the conversion succeeded
        if (code_comp == eCompObsType::eCompError) {
            MMVII_INTERNAL_ERROR("Error reading "+aFileName+" at line "+std::to_string(line_num)+": \""+line+"\"")
            continue;
        }

        std::string nameFrom, nameTo;
        double val, sigma;
        if (!(iss >> nameFrom >> nameTo >> val >> sigma))
        {
            MMVII_INTERNAL_ERROR("Error reading "+aFileName+" at line "+std::to_string(line_num)+": \""+line+"\"")
            continue;
        }
        if (sigma<0)
        {
            StdOut() << "skip: \""+line+"\"  (sigma<0)\n";
            continue;
        }

        switch (code_comp) {
        case eCompObsType::eCompError:
        case eCompObsType::eCompDist:
        case eCompObsType::eCompDX:
        case eCompObsType::eCompDY:
        case eCompObsType::eCompDZ:
        case eCompObsType::eCompDH:
            break;
        case eCompObsType::eCompHzOpen:
        case eCompObsType::eCompHz:
        case eCompObsType::eCompZen:
            // Angles in comp file are in gon. Transform it into rad
            val /= AngleInRad(eTyUnitAngle::eUA_gon);
            sigma /= AngleInRad(eTyUnitAngle::eUA_gon);
            MMVII_INTERNAL_ASSERT_strong(AssertRadAngleInOneRound(val, false),
                                         "Angle out of range for "+aFileName+" at line "+std::to_string(line_num)+": \""+line+"\"")
            break;
        }

        if (!addObs(aCurrentVectObsSetStations, code_comp, nameFrom, nameTo,
                    val, sigma, aCurrStationStatus))
            MMVII_INTERNAL_ERROR("Error interpreting line "+std::to_string(line_num)+": \""+aFileName+"\"")

        ++aNbNewObs;
    }

    addObsSets(aCurrentVectObsSetStations);

    StdOut() << "Reading file finished, added " << aNbNewObs << " obs." << std::endl;
    return true;
}

void cTopoData::clear()
{
    mAllObsSetStations.clear();
    mObsSetSimple.mObs.clear();
}

bool cTopoData::addObs(std::vector<cTopoObsSetStationData> &aCurrentVectObsSetStations, eCompObsType code,
                       const std::string & nameFrom, const std::string & nameTo, double val,
                       double sigma, eTopoStOriStat aStationStatus)
{
    cTopoObsSetStationData * aSetDataStation = nullptr;

    // find set
    switch (code) {
    case eCompObsType::eCompDist:
    case eCompObsType::eCompDH:
        // always add to mObsSetSimple
        break;
    case eCompObsType::eCompHzOpen:
    case eCompObsType::eCompHz:
    case eCompObsType::eCompZen:
    case eCompObsType::eCompDX:
    case eCompObsType::eCompDY:
    case eCompObsType::eCompDZ:
        if (code != eCompObsType::eCompHzOpen) // new set if HzOpen
        {
            // search for a suitable set
            for (auto riter = aCurrentVectObsSetStations.rbegin();
                     riter != aCurrentVectObsSetStations.rend(); ++riter)
            {
                auto &aObsSet = *riter;
                if ((!aObsSet.mObs.empty()) && (aObsSet.mObs.front().mPtsNames.front() == nameFrom))
                {
                    aSetDataStation = &aObsSet;
                    break;
                }
            }
        }
        if (!aSetDataStation)
        {
            aCurrentVectObsSetStations.push_back( {} );
            aSetDataStation = &aCurrentVectObsSetStations.back();
            aSetDataStation->mType = eTopoObsSetType::eStation;
            aSetDataStation->mStationOriStat = aStationStatus;
        }
        break;
    case eCompObsType::eCompError:
        return false;
    }

    cTopoObsData aObsData;
    // create obs
    switch (code) {
    case eCompObsType::eCompDist:
        aObsData = {eTopoObsType::eDist, {nameFrom,nameTo}, {val}, {sigma}};
        break;
    case eCompObsType::eCompHzOpen:
    case eCompObsType::eCompHz:
        aObsData = {eTopoObsType::eHz, {nameFrom,nameTo}, {val}, {sigma}};
        break;
    case eCompObsType::eCompZen:
        aObsData = {eTopoObsType::eZen, {nameFrom,nameTo}, {val}, {sigma}};
        break;
    case eCompObsType::eCompDX:
        aObsData = {eTopoObsType::eDX, {nameFrom,nameTo}, {val}, {sigma}};
        break;
    case eCompObsType::eCompDY:
        aObsData = {eTopoObsType::eDY, {nameFrom,nameTo}, {val}, {sigma}};
        break;
    case eCompObsType::eCompDZ:
        aObsData = {eTopoObsType::eDZ, {nameFrom,nameTo}, {val}, {sigma}};
        break;
    case eCompObsType::eCompDH:
        aObsData = {eTopoObsType::eDH, {nameFrom,nameTo}, {val}, {sigma}};
        break;
    case eCompObsType::eCompError:
        return false;
    }

    // add obs to set
    switch (code) {
    case eCompObsType::eCompDist:
    case eCompObsType::eCompDH:
        mObsSetSimple.mObs.push_back(aObsData);
        break;
    case eCompObsType::eCompHzOpen:
    case eCompObsType::eCompHz:
    case eCompObsType::eCompZen:
    case eCompObsType::eCompDX:
    case eCompObsType::eCompDY:
    case eCompObsType::eCompDZ:
        aSetDataStation->mObs.push_back(aObsData);
        break;
    case eCompObsType::eCompError:
        return false;
    }

    return true;
}

void cTopoData::addObsSets(std::vector<cTopoObsSetStationData> &aCurrentVectObsSets)
{
    mAllObsSetStations.insert(mAllObsSetStations.end(), aCurrentVectObsSets.begin(), aCurrentVectObsSets.end());
    aCurrentVectObsSets.clear();
}

std::pair<cTopoData, cSetMesGnd3D> cTopoData::createEx1()
{
    //verbose = true;

    //create fixed points
    cSetMesGnd3D aSetPts;
    aSetPts.AddMeasure3D( cMes1Gnd3D(cPt3dr(10,10,10), "ptA", 0.001) );
    aSetPts.AddMeasure3D( cMes1Gnd3D(cPt3dr(20,10,10), "ptB", 0.001) );
    aSetPts.AddMeasure3D( cMes1Gnd3D(cPt3dr(15,20,10), "ptC", 0.001) );
    // unknown point
    aSetPts.AddMeasure3D( cMes1Gnd3D(cPt3dr(14,14,14), "ptD") );


#define WW 0.01
    // distances to fixed points
    cTopoObsData aObs1 = {eTopoObsType::eDist, {"ptD", "ptA"},  {10.}, {WW}};
    cTopoObsData aObs2 = {eTopoObsType::eDist, {"ptD", "ptB"},  {10.}, {WW}};
    cTopoObsData aObs3 = {eTopoObsType::eDist, {"ptD", "ptC"},  {10.}, {WW}};
    cTopoObsData aObs4 = {eTopoObsType::eDist, {"ptD", "ptC"},  {10+WW}, {WW}};

    cTopoObsSetData aSet1;
    aSet1.mType = eTopoObsSetType::eSimple;
    aSet1.mObs = {aObs1, aObs2, aObs3, aObs4};

    cTopoData aTopoData;
    aTopoData.mObsSetSimple = aSet1;
    return {aTopoData, aSetPts};
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

std::pair<cTopoData, cSetMesGnd3D>  cTopoData::createEx3()
{
    cSetMesGnd3D aSetPts;
    aSetPts.AddMeasure3D( cMes1Gnd3D(cPt3dr(100,110,100), "Ori1", 0.001) );
    aSetPts.AddMeasure3D( cMes1Gnd3D(cPt3dr(100,100,100), "St1", 0.001) );
    //aSetPts.AddMeasure( cMes1GCP(cPt3dr(105,115,105), "Tr1") ); // init not needed. Final: 107.072, 107.072, 100

    double g0 = 2.2;
    cTopoObsData aObs1 = {eTopoObsType::eHz, {"St1", "Ori1"},  {0. - g0}, {0.0001}};
    cTopoObsData aObs2 = {eTopoObsType::eHz, {"St1", "Tr1"},  {M_PI/4. - g0}, {0.0001}};
    cTopoObsData aObs3 = {eTopoObsType::eZen, {"St1", "Tr1"},  {M_PI/2.}, {0.0001}};

    cTopoObsData aObs4 = {eTopoObsType::eDist, {"St1", "Tr1"},  {10.}, {0.001}};
    cTopoObsData aObs5 = {eTopoObsType::eDist, {"St1", "Tr1"},  {10.002}, {0.001}};
    cTopoObsData aObs6 = {eTopoObsType::eDH, {"St1", "Tr1"},  {0.0002}, {0.0001}};

    cTopoObsSetStationData aSet1;
    aSet1.mType = eTopoObsSetType::eStation;
    aSet1.mStationOriStat = eTopoStOriStat::eTopoStOriVert;
    aSet1.mObs = {aObs1, aObs2, aObs3};

    cTopoObsSetStationData aSet2;
    aSet2.mType = eTopoObsSetType::eSimple;
    aSet2.mObs = {aObs4, aObs5, aObs6};

    cTopoData aTopoData;
    aTopoData.mAllObsSetStations = {aSet1};
    aTopoData.mObsSetSimple = aSet2;

    return {aTopoData, aSetPts};
}


std::pair<cTopoData, cSetMesGnd3D> cTopoData::createEx4()
{
    std::array aVectPoints = {cPt3dr(-10,0,0), cPt3dr(0,-50,0), cPt3dr(20,0,0), cPt3dr(0,50,0)};

    cSetMesGnd3D aSetPts;
    cTopoData aTopoData;

    // create points with an random translation
    cPt3dr aTr = cPt3dr::PRandC()*1000.;
    for (unsigned int i=0;i<aVectPoints.size();++i)
    {
        aSetPts.AddMeasure3D( cMes1Gnd3D(aVectPoints[i]+aTr, std::string("Pt")+(char)('1'+i), 0.001) );
    }
    // do not add station origin, to let automatic initialization set a correct value

    // create x y z obs with a random rotation
    cTopoObsSetStationData aSet1;
    aSet1.mType = eTopoObsSetType::eStation;
    aSet1.mStationOriStat = eTopoStOriStat::eTopoStOriBasc;
    auto aRot = cRotation3D<tREAL8>::RandomRot(M_PI);
    //auto aRot = cRotation3D<tREAL8>::Identity();
#ifdef VERBOSE_TOPO
    StdOut()<<"cTopoData::createEx4() tr: "<<aTr<<"\n";
    StdOut()<<"cTopoData::createEx4() rot:\n";
    StdOut()<<aRot.AxeI()<<"\n";
    StdOut()<<aRot.AxeJ()<<"\n";
    StdOut()<<aRot.AxeK()<<"\n";
#endif
    for (unsigned int i=0;i<aVectPoints.size();++i)
    {
        auto aMesRot = aRot.Value(aVectPoints[i]);
        aSet1.mObs.push_back(
                        {    eTopoObsType::eDX,
                             {"St", std::string("Pt")+(char)('1'+i)},
                             {aMesRot.x()},
                             {0.001}         }     );
        aSet1.mObs.push_back(
                        {    eTopoObsType::eDY,
                             {"St", std::string("Pt")+(char)('1'+i)},
                             {aMesRot.y()},
                             {0.001}         }     );
        aSet1.mObs.push_back(
                        {    eTopoObsType::eDZ,
                             {"St", std::string("Pt")+(char)('1'+i)},
                             {aMesRot.z()},
                             {0.001}         }     );
    }

    aTopoData.mAllObsSetStations = {aSet1};

    return {aTopoData, aSetPts};
}

};
