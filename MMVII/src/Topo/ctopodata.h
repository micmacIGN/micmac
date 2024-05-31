#ifndef CTOPODATA_H
#define CTOPODATA_H

#include "MMVII_Geom3D.h"
#include "SymbDer/SymbDer_Common.h"

using namespace NS_SymbolicDerivative;


namespace MMVII
{

class cBA_Topo;

/**
 * @brief The cTopoPointData class represents the serializable data of a cTopoPoint
 */
class cTopoPointData
{
public:
    void AddData(const  cAuxAr2007 & anAuxInit);
    std::string toString();

    std::string mName;
    cPtxd<tREAL8, 3> mInitCoord;
    bool mIsFree;
    cPtxd<tREAL8, 3> mSigmas;
    std::optional<cPtxd<tREAL8, 2> > mVertDefl;
    std::optional<cPtxd<tREAL8, 3> > mFinalCoord; // just for output
};
void AddData(const cAuxAr2007 & anAux, cTopoPointData & aObsSet);

// ----------------------------------------

/**
 * @brief The cTopoObsData class represents the serializable data of a cTopoObs
 */
class cTopoObsData
{
public:
    void AddData(const  cAuxAr2007 & anAuxInit);
    eTopoObsType mType;
    //std::vector<cTopoPoint*> mPts;
    std::vector<std::string> mPtsNames;
    std::vector<tREAL8> mMeasures;
    std::vector<tREAL8> mSigmas;
    std::optional<std::vector<tREAL8>> mLastResiduals; // just for output
};

void AddData(const cAuxAr2007 & anAux, cTopoObsData & aObsSet);

// ----------------------------------------

/**
 * @brief The cTopoObsSetData class represents the serializable data of a cTopoObsSet
 */
class cTopoObsSetData
{
public:
    void AddData(const  cAuxAr2007 & anAuxInit);
    eTopoObsSetType mType;
    std::vector<cTopoObsData> mObs;

    // just for station
    std::optional<eTopoStOriStat> mStationOriStat;
    std::optional<tREAL8> mStationG0; // just output
    std::optional<cRotation3D<tREAL8>> mRotVert2Instr; // just output
};

void AddData(const cAuxAr2007 & anAux, cTopoObsSetData & aObsSet);

// ------------------------------------------------

/**
 * @brief Obs codes for Comp3D file interpretation
 */
enum class eCompObsType
{
    eCompDist=3,
    eCompHz=5,
    eCompHzOpen=7,
    eCompZen=6,
    eCompDX=14,
    eCompDY=15,
    eCompDZ=16,
    //eDistParam=22,

    eCompError,
};

eCompObsType intToCompObsType(int i); //< return eCompError if incorrect

/**
 * @brief Cor codes for Comp3D file interpretation
 */
enum class eCompCorType
{
        eCompFree=0,
        eCompFixed=1,
};

/**
 * @brief The cTopoData class represents topometric data for serialization
 */
class cTopoData
{
public:
    cTopoData() {}
    cTopoData(const cBA_Topo *aBA_topo); //< fill with actual computation data
    void InsertTopoData(const cTopoData & aOtherTopoData);
    void AddData(const  cAuxAr2007 & anAuxInit);
    void ToFile(const std::string & aFileName) const;
    void FromFile(const std::string & aFileName);
    bool InsertCompObsFile(const std::string & aFileName);
    bool InsertCompCorFile(const std::string & aFileName);
    void print();
    static cTopoData createEx1();
    //void createEx2();
    static cTopoData createEx3();
    static cTopoData createEx4();

    static bool addObs(std::vector<cTopoObsSetData> & aCurrentVectObsSets, MMVII::eCompObsType code, const std::string & nameFrom, const std::string & nameTo, double val, double sigma, eTopoStOriStat aStationStatus);
    std::set<std::string> mAllPointsNames; // all points names
    std::vector<cTopoPointData> mAllPoints; //< for points data extracted from .cor files
    std::vector<cTopoObsSetData> mAllObsSets;

protected:
    void addObsSets(std::vector<cTopoObsSetData> & aCurrentVectObsSets);
};


///  Global function with standard interface required for serialization => just call member
void AddData(const cAuxAr2007 & anAux, cTopoData & aTopoData) ;


};
#endif // CTOPODATA_H
