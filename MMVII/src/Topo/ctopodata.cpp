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

    //AddOptData(anAux,"StationOriStatus",mStationOriStat); // TODOJM
    AddOptData(anAux,"Out_G0",mStationG0);
    // AddOptData(anAux,"Out_RotVert2Instr",mRotVert2Instr); // TODOJM
}


void AddData(const cAuxAr2007 & anAux, cTopoObsSetData &aObsSet)
{
     aObsSet.AddData(anAux);
}


// ------------------------------------

cTopoData::cTopoData(const cBA_Topo* aBA_topo)
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
            aSetData.mStationOriStat = set->getOriStatus();
            aSetData.mStationG0 = set->getG0();
            aSetData.mRotVert2Instr = set->getRotVert2Instr();
            break;
        }
        case eTopoObsSetType::eNbVals:
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
    for (auto& aPtName: aOtherTopoData.mAllPointsNames)
        mAllPointsNames.insert(aPtName);

    for (auto& aObsSet: aOtherTopoData.mAllObsSets)
        mAllObsSets.push_back(aObsSet);

    for (auto& aPoint: aOtherTopoData.mAllPoints)
        mAllPoints.push_back(aPoint);

    // check that points are not duplicated
    std::set<std::string> aTmpNamesSet;
    for (auto& aPoint: aOtherTopoData.mAllPoints)
    {
        auto result = aTmpNamesSet.insert(aPoint.mName);
        MMVII_INTERNAL_ASSERT_User(result.second, eTyUEr::eUnClassedError,
                                   "Error: Point named "+aPoint.mName+
                                   " appears several times in Topo data")
    }
}


eCompObsType intToCompObsType(int i)
{
    eCompObsType res = static_cast<eCompObsType>(i);

    switch(res) {
    case eCompObsType::eCompError:
    case eCompObsType::eCompDX:
    case eCompObsType::eCompDY:
    case eCompObsType::eCompDZ:
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
    std::vector<cTopoObsSetData> aCurrentVectObsSets;
    eTopoStOriStat aCurrStationStatus = eTopoStOriStat::eTopoStOriVert;

    std::ifstream infile(aFileName);
    if (!infile.is_open())
    {
        StdOut() << "Error: can't open obs file \""<<aFileName<<"\""<<std::endl;
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
            addObsSets(aCurrentVectObsSets); // if a station status is given, use only new stations
            aCurrStationStatus = aNewStationStatus;
            continue;
        }

        int code;
        if (!(iss >> code))
        {
            StdOut() << "Error reading "<<aFileName<<" at line " << line_num << ": \""<<line<<"\"\n";
            continue;
        }

        eCompObsType code_comp  = intToCompObsType(code);
        // Check if the conversion succeeded
        if (code_comp == eCompObsType::eCompError) {
            StdOut() << "Error reading "<<aFileName<<" at line " << line_num << ": \""<<line<<"\"\n";
            continue;
        }

        std::string nameFrom, nameTo;
        double val, sigma;
        if (!(iss >> nameFrom >> nameTo >> val >> sigma))
        {
            StdOut() << "Error reading "<<aFileName<<" at line " << line_num << ": \""<<line<<"\"\n";
            continue;
        }
        mAllPointsNames.insert(nameFrom);
        mAllPointsNames.insert(nameTo);

        switch (code_comp) {
        case eCompObsType::eCompError:
        case eCompObsType::eCompDist:
        case eCompObsType::eCompDX:
        case eCompObsType::eCompDY:
        case eCompObsType::eCompDZ:
            break;
        case eCompObsType::eCompHzOpen:
        case eCompObsType::eCompHz:
        case eCompObsType::eCompZen:
            // Angles in comp file are in gon. Transform it into rad
            val /= AngleInRad(eTyUnitAngle::eUA_gon);
            sigma /= AngleInRad(eTyUnitAngle::eUA_gon);
            break;
        }

        if (!addObs(aCurrentVectObsSets, code_comp, nameFrom, nameTo,
                    val, sigma, aCurrStationStatus))
            StdOut() << "Error interpreting line " << line_num << ": \""<<aFileName<<"\"\n";

        ++aNbNewObs;
    }

    addObsSets(aCurrentVectObsSets);

    StdOut() << "Reading file finished, added " << aNbNewObs << " obs." << std::endl;
    return true;
}


bool cTopoData::InsertCompCorFile(const std::string & aFileName)
{
    std::ifstream infile(aFileName);
    if (!infile.is_open())
    {
        StdOut() << "Error: can't open cor file \""<<aFileName<<"\""<<std::endl;
        return false;
    }

    StdOut() << "Reading cor file \""<<aFileName<<"\"..."<<std::endl;
    std::string line;
    int line_num = 0;
    while (std::getline(infile, line))
    {
        ++line_num;
        //std::cout<<"read line "<<line_num<<": \""<<line<<"\""<<std::endl;
        cleanCompLine(line);
        if (line.empty())
            continue;

        std::istringstream iss(line);
        int code;
        if (!(iss >> code))
        {
            StdOut() << "Error reading "<<aFileName<<" at line " << line_num << ": \""<<line<<"\"\n";
            continue;
        }

        eCompCorType code_comp  = static_cast<eCompCorType>(code);
        // Check if the conversion succeeded
        if (static_cast<int>(code_comp) != code) {
            StdOut() << "Error reading "<<aFileName<<" at line " << line_num << ": \""<<line<<"\"\n";
            continue;
        }

        std::string name;
        double x, y, z;
        if (!(iss >> name >> x >> y >> z))
        {
            StdOut() << "Error reading "<<aFileName<<" at line " << line_num << ": \""<<line<<"\"\n";
            continue;
        }
        mAllPointsNames.insert(name);
        mAllPoints.push_back({name, {x, y, z}, code_comp==eCompCorType::eCompFree});
    }


    StdOut() << "Reading file finished. " << mAllPointsNames.size() << " points found." << std::endl;

    return true;
}

bool cTopoData::addObs(std::vector<cTopoObsSetData> &aCurrentVectObsSets, eCompObsType code,
                       const std::string & nameFrom, const std::string & nameTo, double val,
                       double sigma, eTopoStOriStat aStationStatus)
{
    cTopoObsSetData * aSetDataStation = nullptr;
    if (code != eCompObsType::eCompHzOpen) // new set if HzOpen
    {
        // search for a suitable set
        for (auto &aObsSet : aCurrentVectObsSets)
        {
            // TODO: must search from end
            if ((!aObsSet.mObs.empty()) && (aObsSet.mObs.at(0).mPtsNames.at(0) == nameFrom))
            {
                aSetDataStation = &aObsSet;
                break;
            }
        }
    }
    if (!aSetDataStation)
    {
        aCurrentVectObsSets.push_back( {} );
        aSetDataStation = &aCurrentVectObsSets.back();
        aSetDataStation->mType = eTopoObsSetType::eStation;
        aSetDataStation->mStationOriStat = aStationStatus;
    }

    cTopoObsData aObsData;

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
    case eCompObsType::eCompError:
        return false;
    }
    aSetDataStation->mObs.push_back(aObsData);
    return true;
}

void cTopoData::addObsSets(std::vector<cTopoObsSetData> & aCurrentVectObsSets)
{
    mAllObsSets.insert(mAllObsSets.end(), aCurrentVectObsSets.begin(), aCurrentVectObsSets.end());
    aCurrentVectObsSets.clear();
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
    aSet1.mStationOriStat = eTopoStOriStat::eTopoStOriFixed;
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
    cTopoObsData aObs3 = {eTopoObsType::eZen, {"St1", "Tr1"},  {M_PI/2.}, {0.001}};
    cTopoObsData aObs4 = {eTopoObsType::eDist, {"St1", "Tr1"},  {10.}, {0.001}};
    cTopoObsData aObs5 = {eTopoObsType::eDist, {"St1", "Tr1"},  {10.002}, {0.001}};

    cTopoObsSetData aSet1;
    aSet1.mType = eTopoObsSetType::eStation;
    aSet1.mStationOriStat = eTopoStOriStat::eTopoStOriVert;
    aSet1.mObs = {aObs1, aObs2, aObs3, aObs4, aObs5};

    cTopoData aTopoData;
    aTopoData.mAllPoints = {aPt1, aPt2, aPt3};
    aTopoData.mAllObsSets = {aSet1};

    return aTopoData;
}


cTopoData cTopoData::createEx4()
{
    std::array aVectPoints = {cPt3dr(-50,0,0), cPt3dr(0,-50,0), cPt3dr(50,0,0), cPt3dr(0,50,0)};

    cTopoData aTopoData;
    // create points with an random translation
    cPt3dr aTr = cPt3dr::PRandC()*1000.;
    for (unsigned int i=0;i<aVectPoints.size();++i)
    {
        aTopoData.mAllPoints.push_back( { std::string("Pt")+(char)('1'+i),
                                          aVectPoints[i]+aTr,false,cPt3dr(0.001,0.001,0.001)
                                        } );
    }
    aTopoData.mAllPoints.push_back( { "St",{0.,0.,0.},true,{0.,0.,0.} } ); // the station is free

    // create x y z obs with a random rotation
    cTopoObsSetData aSet1;
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

    aTopoData.mAllObsSets = {aSet1};

    return aTopoData;
}

};
