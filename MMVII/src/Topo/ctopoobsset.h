#ifndef CTOPOOBSSET_H
#define CTOPOOBSSET_H

#include "ctopoobs.h"
#include "ctopodata.h"
#include "MMVII_Geom3D.h"
#include "MMVII_SysSurR.h"
#include "MMVII_enums.h"
#include "MMVII_PCSens.h"

namespace MMVII
{
class cTopoObsSetData;

typedef std::map<const cTopoPoint*, cPt3dr> tPointToVectorMap;

/**
 * @brief The cTopoObsSet class represents a set of observations sharing the same set of parameters.
 */
class cTopoObsSet : public cObjWithUnkowns<tREAL8>, public cMemCheck
{
    friend class cTopoObs;
    friend class cTopoData;
public:
    virtual ~cTopoObsSet();
    void PutUknowsInSetInterval() override ; ///< describes its unknowns
    virtual void OnUpdate() override = 0;    ///< "reaction" after linear update, eventually update inversion
    virtual std::string toString() const;
    eTopoObsSetType getType() const {return mType;}
    std::vector<int> getParamIndices() const;
    size_t nbObs() const {return mObs.size();}
    cTopoObs* getObs(size_t i) {return mObs.at(i);}
    std::vector<cTopoObs*> & getAllObs() {return mObs;}
    bool addObs(eTopoObsType type, cBA_Topo * aBA_Topo, const std::vector<std::string> &pts, const std::vector<tREAL8> & vals,  const cResidualWeighterExplicit<tREAL8> & aWeights);
    virtual void makeConstraints(cResolSysNonLinear<tREAL8> & aSys) = 0; ///< add constraints for current set
    virtual bool initialize() = 0; ///< initialize set parameters, after all obs and points were added
    bool isInit() const {return mInit;}
protected:
    cTopoObsSet(cBA_Topo *aBA_Topo, eTopoObsSetType type);
    cTopoObsSet(cTopoObsSet const&) = delete;
    cTopoObsSet& operator=(cTopoObsSet const&) = delete;
    virtual void createAllowedObsTypes() = 0;
    void create(); ///< will be called automatically by make_TopoObsSet()
    eTopoObsSetType mType;
    std::vector<cTopoObs*> mObs;
    std::vector<tREAL8> mParams; ///< the only copy of the parameters
    std::vector<eTopoObsType> mAllowedObsTypes; ///< to check if new obs are allowed
    cBA_Topo * mBA_Topo;
    bool mInit; ///< is the set initialized
};

/**
 * Have to use make_TopoObsSet() to create cTopoObsSet with initialization
 */
template <class T>
T * make_TopoObsSet(cBA_Topo *aBA_Topo)
{
    auto o = new T(aBA_Topo);
    o->create();
    return o;
}


/**
 * @brief The cTopoObsSetSimple class represents a set of observations without parameters
 */
class cTopoObsSetSimple : public cTopoObsSet
{
    friend cTopoObsSetSimple * make_TopoObsSet<cTopoObsSetSimple>(cBA_Topo *aBA_Topo);
public:
    virtual ~cTopoObsSetSimple() override {}
    void OnUpdate() override;    ///< "reaction" after linear update, eventually update inversion
    virtual std::string toString() const override;
    void makeConstraints(cResolSysNonLinear<tREAL8> &aSys) override;
    virtual bool initialize() override; ///< initialize rotation

protected:
    cTopoObsSetSimple(cBA_Topo *aBA_Topo);
    void createAllowedObsTypes() override;
};


/**
 * @brief The cTopoObsSetStation class represents a set of observation from one station,
 * that has a rotation unknown
 *  mRotVert2Instr unknown is recorded as mParams[0..2]
 */
class cTopoObsSetStation : public cTopoObsSet
{
    typedef cRotation3D<tREAL8> tRot;
    friend cTopoObsSetStation * make_TopoObsSet<cTopoObsSetStation>(cBA_Topo *aBA_Topo);
public:
    virtual ~cTopoObsSetStation() override {}
    void OnUpdate() override;    ///< "reaction" after linear update, eventually update inversion
    virtual std::string toString() const override;
    void makeConstraints(cResolSysNonLinear<tREAL8> &aSys) override;
    virtual bool initialize() override; ///< initialize rotation
    void GetAdrInfoParam(cGetAdrInfoParam<tREAL8> & aGAIP) override;

    void setOrigin(std::string _OriginName);
    void PushRotObs(std::vector<double> & aVObs) const;
    cPt3dr getRotOmega() const { return cPt3dr::FromStdVector(mParams); }
    const cTopoPoint * getPtOrigin() const { return mPtOrigin; }
    tREAL8 getG0() const;
    const tRot & getRotVert2Instr() const { return mRotVert2Instr; }
    tRot getRotSysCo2Instr() const { return mRotVert2Instr * mRotSysCo2Vert; }
    cPt3dr PtSysCo2Vert(const cTopoPoint &aPt) const;
    cPt3dr PtSysCo2Instr(const cTopoPoint & aPt) const;
    cPt3dr PtInstr2SysCo(const cPt3dr &aVect) const;
    eTopoStOriStat getOriStatus() const { return mOriStatus; }
    void setOriStatus(eTopoStOriStat aOriStatus) { mOriStatus = aOriStatus; }
    cPt3dr obs2InstrVector(const std::string & aPtToName) const; //< find several obs to same point and convert it into a vector in instrument frame. Returns dummy if not enough points
    tPointToVectorMap toInstrVectorMap(); //< compute the instrument vectors for each seen points

protected:
    cTopoObsSetStation(cBA_Topo *aBA_Topo);
    //cTopoObsSetStation(cTopoObsSetStation const&) = delete;
    //cTopoObsSetStation& operator=(cTopoObsSetStation const&) = delete;
    void createAllowedObsTypes() override;
    void updateVertMat();
    void resetRotOmega(); //< reset rotation unknowns to 0 (in mParams)
    eTopoStOriStat mOriStatus; //< is bubbled, fixed or 3d rot free
    tRot mRotSysCo2Vert; //< rotation between global SysCo and local vertical frame
    tRot mRotVert2Instr; //< current value for rotation from local vertical frame to instrument frame
    std::string mOriginName;
    const cTopoPoint * mPtOrigin;
};



/**
 * @brief The cTopoObsSetDistParam class represents a set of observation of type distParam,
 * where the distance between two points is the same for several pair of points
 *
 */
/*
class cTopoObsSetDistParam : public cTopoObsSet
{
    friend std::unique_ptr<cTopoObsSet> make_TopoObsSet<cTopoObsSetDistParam>(cBA_Topo *aBA_Topo);
public:
    void OnUpdate() override;    ///< "reaction" after linear update, eventually update inversion
    void makeConstraints(const cResolSysNonLinear<tREAL8> & aSys) override;
protected:
    cTopoObsSetDistParam(cTopoData& aTopoData);
    cTopoObsSetDistParam(cTopoObsSetDistParam const&) = delete;
    cTopoObsSetDistParam& operator=(cTopoObsSetDistParam const&) = delete;
    void createAllowedObsTypes() override;
    void createParams() override;
};*/

};
#endif // CTOPOOBSSET_H
