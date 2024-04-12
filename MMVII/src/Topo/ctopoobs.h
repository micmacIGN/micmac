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
class cTopoObs : public cMemCheck
{
    friend class cTopoObsSet;
    friend class cTopoData;
public:
    //~cTopoObs() { std::cout<<"delete topo obs "<<toString()<<std::endl; }
    std::string toString() const;
    eTopoObsType getType() const {return mType;}
    std::vector<int> getIndices(cBA_Topo *aBA_Topo) const;
    std::vector<tREAL8> getVals() const; //< for least squares (with rotation matrix if needed
    std::vector<tREAL8> & getMeasures() { return mMeasures;} //< original measures
    std::vector<tREAL8> & getResiduals() { return mLastResiduals;} //< last residuals
    cResidualWeighterExplicit<tREAL8>& getWeights();
    const std::string & getPointName(size_t i) const { return mPtsNames.at(i); }
    //std::vector<tREAL8> getResiduals(const cTopoComp *comp) const;
protected:
    cTopoObs(cTopoObsSet* set, cBA_Topo * aBA_Topo, eTopoObsType type, const std::vector<std::string> & ptsNames, const std::vector<tREAL8> & measures,  const cResidualWeighterExplicit<tREAL8> & aWeights);
    cTopoObs(const cTopoObs &) = delete;
    cTopoObs& operator=(const cTopoObs &) = delete;
    cTopoObsSet* mSet;//the set containing the shared parameters
    cBA_Topo * mBA_Topo;
    eTopoObsType mType;
    //std::vector<cTopoPoint*> mPts;
    std::vector<std::string> mPtsNames;
    std::vector<tREAL8> mMeasures;
    cResidualWeighterExplicit<tREAL8> mWeights;
    std::vector<tREAL8> mLastResiduals;
};

};
#endif // CTOPOOBS_H
