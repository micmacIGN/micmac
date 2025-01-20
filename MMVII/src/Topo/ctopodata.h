#ifndef CTOPODATA_H
#define CTOPODATA_H

#include "MMVII_Geom3D.h"
#include "SymbDer/SymbDer_Common.h"

using namespace NS_SymbolicDerivative;


namespace MMVII
{

class cBA_Topo;

/**
 * @brief The cTopoObsData class represents the serializable data of a cTopoObs
 */
class cTopoObsData
{
public:
    void AddData(const  cAuxAr2007 & anAuxInit);
    eTopoObsType mType;
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
    virtual ~cTopoObsSetData() {};
    void AddData(const  cAuxAr2007 & anAuxInit);
    virtual void AddSupData(const  cAuxAr2007 & anAux) {};
    eTopoObsSetType mType = eTopoObsSetType::eSimple;
    std::vector<cTopoObsData> mObs;
};

void AddData(const cAuxAr2007 & anAux, cTopoObsSetData & aObsSet);

class cTopoObsSetStationData: public cTopoObsSetData
{
public:
    cTopoObsSetStationData();
    virtual ~cTopoObsSetStationData() {}
    virtual void AddSupData(const cAuxAr2007 & anAux) override;

    eTopoStOriStat mStationOriStat;
    std::optional<tREAL8> mStationG0; // just output
    std::optional<cRotation3D<tREAL8>> mRotVert2Instr; // just output
};

// ------------------------------------------------

/**
 * @brief Obs codes for Comp3D file interpretation
 */
enum class eCompObsType
{
    eCompDist=3,
    eCompDH=4,
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
 * @brief The cTopoData class represents topo survey data for serialization
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
    void clear();
    static std::pair<cTopoData, cSetMesGnd3D>  createEx1();
    //static std::pair<cTopoData, cSetMesGCP>  createEx2();
    static std::pair<cTopoData, cSetMesGnd3D>  createEx3();
    static std::pair<cTopoData, cSetMesGnd3D>  createEx4();

    bool addObs(std::vector<cTopoObsSetStationData> & aCurrentVectObsSetStations, MMVII::eCompObsType code,
                       const std::string & nameFrom, const std::string & nameTo, double val, double sigma, eTopoStOriStat aStationStatus);

    std::vector<cTopoObsSetStationData> mAllObsSetStations;
    cTopoObsSetData mObsSetSimple;
protected:
    void addObsSets(std::vector<cTopoObsSetStationData> & aCurrentVectObsSets);
};


///  Global function with standard interface required for serialization => just call member
void AddData(const cAuxAr2007 & anAux, cTopoData & aTopoData) ;


};
#endif // CTOPODATA_H
