#ifndef CTOPOOBS_H
#define CTOPOOBS_H

#include "MMVII_AllClassDeclare.h"
#include "MMVII_enums.h"
#include "MMVII_SysSurR.h"

namespace MMVII
{
class cTopoObsSet;
class cTopoPoint;
class cBA_Topo;
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
    std::vector<int> getIndices(cBA_Topo *aBA_Topo) const;
    std::vector<tREAL8> getVals() const;
    cResidualWeighterExplicit<tREAL8>& getWeights();
    //std::vector<tREAL8> getResiduals(const cTopoComp *comp) const;
protected:
    cTopoObs(cTopoObsSet* set, eTopoObsType type, const std::vector<std::string> & pts, const std::vector<tREAL8> & vals,  const cResidualWeighterExplicit<tREAL8> & aWeights);
    cTopoObs(const cTopoObs &) = delete;
    cTopoObs& operator=(const cTopoObs &) = delete;
    cTopoObsSet* mSet;//the set containing the shared parameters
    eTopoObsType mType;
    //std::vector<cTopoPoint*> mPts;
    std::vector<std::string> mPts;
    std::vector<tREAL8> mVals;
    cResidualWeighterExplicit<tREAL8> mWeights;
};

void AddData(const cAuxAr2007 & anAux, cTopoObs * aTopoObs);


};
#endif // CTOPOOBS_H
