#ifndef CTOPOOBS_H
#define CTOPOOBS_H

#include "MMVII_AllClassDeclare.h"
#include "MMVII_enums.h"

namespace MMVII
{
class cTopoObsSet;
class cTopoPoint;
template <class Type> class cResidualWeighterExplicit;


/**
 * @brief The cTopoObs class represents an observation between several points.
 * It exists only in a cTopoObsSet because the obs may share parameters with other obs.
 */
class cTopoObs
{
    friend class cTopoObsSet;
public:
    void AddData(const  cAuxAr2007 & anAuxInit);
    std::string toString() const;
    eTopoObsType getType() const {return mType;}
    //std::vector<int> getIndices() const;
    std::vector<tREAL8> getVals() const;
    //cResidualWeighterExplicit<tREAL8>& getWeights();
    //std::vector<tREAL8> getResiduals(const cTopoComp *comp) const;
    std::string type2string() const;
protected:
    cTopoObs(cTopoObsSet* set, eTopoObsType type, const std::vector<cTopoPoint*> & pts, const std::vector<tREAL8> & vals,  const cResidualWeighterExplicit<tREAL8> & aWeights);
    cTopoObs(const cTopoObs &) = delete;
    cTopoObs& operator=(const cTopoObs &) = delete;
    cTopoObsSet* mSet;//the set containing the shared parameters
    eTopoObsType mType;
    std::vector<cTopoPoint*> mPts;
    std::vector<tREAL8> mVals;
    //cResidualWeighterExplicit<tREAL8> mWeights;
};

void AddData(const cAuxAr2007 & anAux, cTopoObs * aTopoObs);


};
#endif // CTOPOOBS_H
