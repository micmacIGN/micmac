#ifndef CTOPOOBSSET_H
#define CTOPOOBSSET_H

#include "ctopoobs.h"
#include "MMVII_Geom3D.h"
#include "MMVII_SysSurR.h"
#include "MMVII_enums.h"
#include "MMVII_PCSens.h"

namespace MMVII
{

/**
 * @brief The cTopoObsSet class represents a set of observations sharing the same set of parameters.
 */
class cTopoObsSet : public cObjWithUnkowns<tREAL8>, public cMemCheck
{
    friend class cTopoObs;
    friend class cTopoData;
public:
    virtual ~cTopoObsSet();
    virtual void AddToSys(cSetInterUK_MultipeObj<tREAL8> & aSet);
    virtual void PutUknowsInSetInterval() override ; ///< describes its unknowns
    virtual void OnUpdate() override = 0;    ///< "reaction" after linear update, eventually update inversion
    virtual std::string toString() const;
    eTopoObsSetType getType() const {return mType;}
    std::vector<int> getParamIndices() const;
    size_t nbObs() const {return mObs.size();}
    cTopoObs* getObs(size_t i) {return mObs.at(i);}
    std::vector<cTopoObs*> & getAllObs() {return mObs;}
    bool addObs(eTopoObsType type, cBA_Topo * aBA_Topo, const std::vector<std::string> &pts, const std::vector<tREAL8> & vals,  const cResidualWeighterExplicit<tREAL8> & aWeights);
    virtual void makeConstraints(cResolSysNonLinear<tREAL8> & aSys) = 0; // add constraints for current set
    virtual bool initialize() = 0; // initialize set parameters
protected:
    cTopoObsSet(cBA_Topo *aBA_Topo, eTopoObsSetType type);
    cTopoObsSet(cTopoObsSet const&) = delete;
    cTopoObsSet& operator=(cTopoObsSet const&) = delete;
    virtual void createAllowedObsTypes() = 0;
    virtual void createParams() = 0;
    void init(); ///< will be called automatically by make_TopoObsSet()
    eTopoObsSetType mType;
    std::vector<cTopoObs*> mObs;
    //cObjWithUnkowns<tREAL8> * mUK;
    std::vector<tREAL8> mParams; //the only copy of the parameters
    std::vector<eTopoObsType> mAllowedObsTypes;//to check if new obs are allowed
    cBA_Topo * mBA_Topo;
};

/**
 * Have to use make_TopoObsSet() to create cTopoObsSet with initialization
 */
template <class T>
cTopoObsSet * make_TopoObsSet(cBA_Topo *aBA_Topo)
{
    auto o = new T(aBA_Topo);
    o->init();
    return o;
}

/**
 * @brief The cTopoObsSetSimple class represents a set of observation without parameters
 */
class cTopoObsSetStation : public cTopoObsSet
{
    typedef cRotation3D<tREAL8> tRot;
    friend cTopoObsSet * make_TopoObsSet<cTopoObsSetStation>(cBA_Topo *aBA_Topo);
public:
    virtual ~cTopoObsSetStation() override {}
    virtual void PutUknowsInSetInterval() override ; ///< describes its unknowns
    void AddToSys(cSetInterUK_MultipeObj<tREAL8> & aSet) override;
    void OnUpdate() override;    ///< "reaction" after linear update, eventually update inversion
    virtual std::string toString() const override;
    void makeConstraints(cResolSysNonLinear<tREAL8> &aSys) override;
    virtual bool initialize() override; // initialize rotation

    void setOrigin(std::string _OriginName, bool _IsVericalized);
    void PushRotObs(std::vector<double> & aVObs) const;
    cPt3dr_UK & getRotOmega() { return mRotOmega; }
    cTopoPoint * getPtOrigin() const { return mPtOrigin; }
    tREAL8 getG0() const;
    const tRot & getRotVert2Instr() const { return mRotVert2Instr; }
    bool isVericalized(){ return mIsVericalized; }
    void setIsVericalized(bool isVert){ mIsVericalized = isVert; }
    bool isOriented(){ return mIsOriented; }
    void setIsOriented(bool isOri){ mIsOriented = isOri; }
protected:
    cTopoObsSetStation(cBA_Topo *aBA_Topo);
    //cTopoObsSetStation(cTopoObsSetStation const&) = delete;
    //cTopoObsSetStation& operator=(cTopoObsSetStation const&) = delete;
    void createAllowedObsTypes() override;
    void createParams() override;

    bool mIsVericalized; // bubbled (orientation free only around vertical)
    bool mIsOriented;    // rotation around vertical is fixed
    tRot mRotVert2Instr;        //< the station orientation from local vertical frame
    cPt3dr_UK mRotOmega; //< the station orientation unknown
    std::string mOriginName;
    cTopoPoint *mPtOrigin;
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

/**
 * @brief The cTopoObsSubFrame class represents a sub frame, where observations
 */
/*class cTopoObsSetSubFrame : public cTopoObsSet
{
    friend std::unique_ptr<cTopoObsSet> make_TopoObsSet<cTopoObsSetSubFrame>(cBA_Topo *aBA_Topo);
public:
    void OnUpdate() override;    ///< "reaction" after linear update, eventually update inversion
    void makeConstraints(const cResolSysNonLinear<tREAL8> &aSys) override;
    std::vector<tREAL8> getRot() const;
protected:
    cTopoObsSetSubFrame(cTopoData& aTopoData);
    cTopoObsSetSubFrame(cTopoObsSetSubFrame const&) = delete;
    cTopoObsSetSubFrame& operator=(cTopoObsSetSubFrame const&) = delete;
    void createAllowedObsTypes() override;
    void createParams() override;
    cRotation3D<tREAL8> mRot; ///< the roation matrix, its small changes are the params
};
*/
};
#endif // CTOPOOBSSET_H
