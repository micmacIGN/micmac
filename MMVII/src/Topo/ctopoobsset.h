#ifndef CTOPOOBSSET_H
#define CTOPOOBSSET_H

#include "ctopoobs.h"
#include "MMVII_Geom3D.h"
#include "MMVII_SysSurR.h"
#include "MMVII_enums.h"

namespace MMVII
{


/**
 * @brief The cTopoObsSet class represents a set of observations sharing the same set of parameters.
 */
class cTopoObsSet : public cObjWithUnkowns<tREAL8>
{
    friend class cTopoObs;
public:
    virtual ~cTopoObsSet() {}
    void PutUknowsInSetInterval() override ;///< describes its unknowns
    virtual void OnUpdate() override = 0;    ///< "reaction" after linear update, eventually update inversion
    void AddData(const  cAuxAr2007 & anAuxInit);
    std::string toString() const;
    eTopoObsSetType getType() const {return mType;}
    std::vector<int> getParamIndices() const;
    size_t nbObs() const {return mObs.size();}
    cTopoObs* getObs(size_t i) {return mObs.at(i);}
    virtual std::string type2string() const = 0;
protected:
    cTopoObsSet(eTopoObsSetType type);
    cTopoObsSet(cTopoObsSet const&) = delete;
    cTopoObsSet& operator=(cTopoObsSet const&) = delete;
    virtual void createAllowedObsTypes() = 0;
    virtual void createParams() = 0;
    void init(); ///< will be called automatically by make_TopoObsSet()
    bool addObs(cTopoObs *obs);
    eTopoObsSetType mType;
    std::vector<cTopoObs*> mObs;
    std::vector<tREAL8> mParams; //the only copy of the parameters
    std::vector<eTopoObsType> mAllowedObsTypes;//to check if new obs are allowed
};

void AddData(const cAuxAr2007 & anAux, std::unique_ptr<cTopoObsSet> &aObsSet);

/**
 * Have to use make_TopoObsSet() to create cTopoObsSet with initialization
 */
template <class T>
std::unique_ptr<cTopoObsSet> make_TopoObsSet()
{
    auto o = std::unique_ptr<T>(new T());
    o->init();
    return o;
}

/**
 * @brief The cTopoObsSetSimple class represents a set of observation without parameters
 */
class cTopoObsSetSimple : public cTopoObsSet
{
    friend std::unique_ptr<cTopoObsSet> make_TopoObsSet<cTopoObsSetSimple>();
public:
    void OnUpdate() override;    ///< "reaction" after linear update, eventually update inversion
    std::string type2string() const override;
protected:
    cTopoObsSetSimple();
    cTopoObsSetSimple(cTopoObsSetSimple const&) = delete;
    cTopoObsSetSimple& operator=(cTopoObsSetSimple const&) = delete;
    void createAllowedObsTypes() override;
    void createParams() override;
};

/**
 * @brief The cTopoObsSetDistParam class represents a set of observation of type distParam,
 * where the distance between two points is the same for several pair of points
 *
 */
class cTopoObsSetDistParam : public cTopoObsSet
{
    friend std::unique_ptr<cTopoObsSet> make_TopoObsSet<cTopoObsSetDistParam>();
public:
    void OnUpdate() override;    ///< "reaction" after linear update, eventually update inversion
    std::string type2string() const override;
protected:
    cTopoObsSetDistParam();
    cTopoObsSetDistParam(cTopoObsSetDistParam const&) = delete;
    cTopoObsSetDistParam& operator=(cTopoObsSetDistParam const&) = delete;
    void createAllowedObsTypes() override;
    void createParams() override;
};

/**
 * @brief The cTopoObsSubFrame class represents a sub frame, where observations
 */
class cTopoObsSetSubFrame : public cTopoObsSet
{
    friend std::unique_ptr<cTopoObsSet> make_TopoObsSet<cTopoObsSetSubFrame>();
    std::string type2string() const override;
public:
    void OnUpdate() override;    ///< "reaction" after linear update, eventually update inversion
    std::vector<tREAL8> getRot() const;
protected:
    cTopoObsSetSubFrame();
    cTopoObsSetSubFrame(cTopoObsSetSubFrame const&) = delete;
    cTopoObsSetSubFrame& operator=(cTopoObsSetSubFrame const&) = delete;
    void createAllowedObsTypes() override;
    void createParams() override;
    cRotation3D<tREAL8> mRot; ///< the roation matrix, its small changes are the params
};

};
#endif // CTOPOOBSSET_H
