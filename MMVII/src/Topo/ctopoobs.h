#ifndef CTOPOOBS_H
#define CTOPOOBS_H

#include "MMVII_AllClassDeclare.h"
#include "ctopopoint.h"

namespace MMVII
{
class cTopoObsSet;
class cTopoComp;

enum class TopoObsType
{
    dist=3,
    subFrame=11,
    distParam=22,
};

/**
 * @brief The cTopoObs class represents an observation between several points.
 * It exists only in a cTopoObsSet because the obs may share parameters with other obs.
 */
class cTopoObs
{
    friend class cTopoObsSet;
public:
    cTopoObs(cTopoObsSet* set, TopoObsType type, const std::vector<cTopoPoint*> & pts, const std::vector<tREAL8> & vals,  const cResidualWeighterExplicit<tREAL8> & aWeights);
    std::string toString() const;
    TopoObsType getType() const {return mType;}
    std::vector<int> getIndices() const;
    std::vector<tREAL8> getVals() const;
    cResidualWeighterExplicit<tREAL8>& getWeights();
    std::vector<tREAL8> getResiduals(const cTopoComp *comp) const;
    std::string type2string() const;
protected:
    cTopoObsSet* mSet;//the set containing the shared parameters
    TopoObsType mType;
    std::vector<cTopoPoint*> mPts;
    std::vector<tREAL8> mVals;
    cResidualWeighterExplicit<tREAL8> mWeights;
};


};
#endif // CTOPOOBS_H
