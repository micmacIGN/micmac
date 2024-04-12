#ifndef CTOPOPOINT_H
#define CTOPOPOINT_H

#include "../BundleAdjustment/BundleAdjustment.h"

//#define VERBOSE_TOPO

namespace MMVII
{

/**
 * @brief The cTopoPoint class represents a 3d point.
 * Its unknowns can be store in the cTopoPoint (for pure topo points),
 * in a GCP or an image pose. *
 *
 * they may have coord constraints (not isFree and sigmas)
 * vertical deflection can be given
 */
class cTopoPoint : public cMemCheck
{
    friend class cTopoData;
public:
    cTopoPoint(std::string name, const cPtxd<tREAL8,3> &_coord, bool _isFree, const cPtxd<tREAL8,3> &_sigmas={});
    cTopoPoint();
    ~cTopoPoint();
    void AddData(const  cAuxAr2007 & anAuxInit);

    void findOrMakeUK(const std::vector<cBA_GCP *> &vGCP, cPhotogrammetricProject *aPhProj, const cPt3dr &aCoordIfPureTopo); //< all params can be null
    cPt3dr* getPt() const;
    cObjWithUnkowns<tREAL8>* getUK() const;
    bool isReady() const { return mUK!=nullptr; } //< ready after findOrMakeUK. Can't use in equations if not ready
    bool doesOwnsUK() const { return mOwnsUK; }

    void setVertDefl(const cPtxd<tREAL8, 2> &_vertDefl);
    std::string toString();
    const cPtxd<tREAL8, 3> & getInitCoord() const {return mInitCoord;}
    std::string getName() {return mName;}
    std::vector<int> getIndices();
    void AddToSys(cSetInterUK_MultipeObj<tREAL8> & aSet);
    void makeConstraints(cResolSysNonLinear<tREAL8> & aSys);
protected:
    std::string mName;
    cPtxd<tREAL8, 3> mInitCoord; //< coord initialized only for pure topo points, may be also reference coordinates if not free for any type of point
    bool mIsFree;
    cPtxd<tREAL8, 3> mSigmas;
    std::optional<cPtxd<tREAL8, 2> > mVertDefl;
    // Unknowns part
    cObjWithUnkowns<tREAL8>* mUK; //< the unknowns are stored as ptr, to be owned or common with GPC or image
    cPt3dr* mPt;
    bool mOwnsUK;
};

///  Global function with standard interface required for serialization => just call member
void AddData(const cAuxAr2007 & anAux, cTopoPoint &aTopoPoint);

};
#endif // CTOPOPOINT_H
