#ifndef CTOPODATA_H
#define CTOPODATA_H

#include "ctopoobsset.h"
//#include "cMMVII_Appli.h"
//#include "SymbDer/SymbDer_Common.h"

using namespace NS_SymbolicDerivative;


namespace MMVII
{


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
    std::optional<bool> mStationIsVericalized;
    std::optional<bool> mStationIsOriented;
    std::optional<tREAL8> mStationG0; // just output
    std::optional<cRotation3D<tREAL8>> mRotVert2Instr; // just output
};

void AddData(const cAuxAr2007 & anAux, cTopoObsSetData & aObsSet);

// ------------------------------------------------

/**
 * @brief Obs codes for comp3d file interpretation
 */
enum class eCompObsTypes
{
        eCompDist=3,
        eCompHz=5,
        eCompZen=6,
        eCompDX=14,
        eCompDY=15,
        eCompDZ=4,
        //eSubFrame=11,
        //eDistParam=22,
};



/**
 * @brief The cTopoData class represents topometric data for serialization
 */
class cTopoData
{
public:
    cTopoData() {}
    cTopoData(cBA_Topo* aBA_topo); //< fill with actual computation data
    void AddData(const  cAuxAr2007 & anAuxInit);
    void ToFile(const std::string & aName) const;
    void FromFile(const std::string & aName);
    bool FromCompFile(const std::string & aName);
    void print();
    static cTopoData createEx1();
    //void createEx2();
    static cTopoData createEx3();

    bool addObs(MMVII::eCompObsTypes code, const std::string & nameFrom, const std::string & nameTo, double val, double sigma);

    std::vector<cTopoPointData> mAllPoints;
    std::vector<cTopoObsSetData> mAllObsSets;
};


///  Global function with standard interface required for serialization => just call member
void AddData(const cAuxAr2007 & anAux, cTopoData & aTopoData) ;


};
#endif // CTOPODATA_H
