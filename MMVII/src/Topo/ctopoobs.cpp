#include "ctopoobs.h"
#include "MMVII_PhgrDist.h"
#include "MMVII_2Include_Serial_Tpl.h"
#include <memory>
#include "MMVII_SysSurR.h"
#include "MMVII_Topo.h"

namespace MMVII
{


cTopoObs::cTopoObs(cTopoObsSet* set, cBA_Topo *aBA_Topo, eTopoObsType type, const std::vector<std::string> &ptsNames, const std::vector<tREAL8> & measures, const cResidualWeighterExplicit<tREAL8> &aWeights):
    mSet(set), mBA_Topo(aBA_Topo), mType(type),
    mPtsNames(ptsNames), mMeasures(measures), mWeights(aWeights),
    mLastResiduals(measures.size(), NAN)
{
    if (!mSet)
    {
        MMVII_INTERNAL_ERROR("Obs: no set given")
        return; //just to please the compiler
    }
    switch (mType) {
    case eTopoObsType::eHz:
    case eTopoObsType::eZen:
    case eTopoObsType::eDist:
    case eTopoObsType::eDX:
    case eTopoObsType::eDY:
    case eTopoObsType::eDZ:
        MMVII_INTERNAL_ASSERT_strong(mSet->getType()==eTopoObsSetType::eStation, "Obs: incorrect set type")
        MMVII_INTERNAL_ASSERT_strong(ptsNames.size()==2, "Obs: incorrect number of points")
        MMVII_INTERNAL_ASSERT_strong(measures.size()==1, "Obs: 1 value should be given")
        MMVII_INTERNAL_ASSERT_strong(aWeights.size()==1, "Obs: 1 weight should be given")
        break;
/*    case eTopoObsType::eDistParam:
        MMVII_INTERNAL_ASSERT_strong(mSet->getType()==eTopoObsSetType::eDistParam, "Obs: incorrect set type")
        MMVII_INTERNAL_ASSERT_strong(pts.size()==2, "Obs: incorrect number of points")
        MMVII_INTERNAL_ASSERT_strong(vals.empty(), "Obs: value should not be given")
        MMVII_INTERNAL_ASSERT_strong(aWeights.size()==1, "Obs: 1 weight should be given")
        break;*/
    case eTopoObsType::eNbVals:
        MMVII_INTERNAL_ERROR("unknown obs type")
    }

    // check values
    switch (mType) {
    case eTopoObsType::eHz:
    case eTopoObsType::eZen:
        for (auto &m:measures)
            MMVII_INTERNAL_ASSERT_strong(AssertRadAngleInOneRound(m, false),
                                         "Angle out of range for "+this->toString())
        break;
    default:
        break;
    }

}

std::string cTopoObs::toString() const
{
    std::ostringstream oss;
    oss<<"TopoObs "<<E2Str(mType)<<" ";
    for (auto & pt: mPtsNames)
        oss<<pt<<" ";
    oss<<"val: ";
    for (auto & val: mMeasures)
        oss<<val<<" ";
    oss<<"sigma: ";
    for (auto & val: mWeights.getSigmas())
        oss<<val<<" ";
    oss<<"prev res norm: ";
    for (unsigned int i=0; i<mLastResiduals.size(); ++i)
        oss<<mLastResiduals.at(i)/mWeights.getSigmas().at(i)<<" ";
    return oss.str();
}

std::vector<int> cTopoObs::getIndices() const
{
    std::vector<int> indices;
    switch (mSet->getType()) {
    case eTopoObsSetType::eStation:
    {
        cTopoObsSetStation* set = dynamic_cast<cTopoObsSetStation*>(mSet);
        if (!set)
        {
            MMVII_INTERNAL_ERROR("error set type")
            return {}; //just to please the compiler
        }
        if (!set->getPtOrigin())
        {
            MMVII_INTERNAL_ERROR("error set station has no origin")
            return {}; //just to please the compiler
        }
        set->getPtOrigin()->getUK()->PushIndexes(indices);
        indices.resize(3); // keep only the point part for cSensorImage UK // TODO: improve, how to get only the point part of UK?
        set->PushIndexes(indices, set->mParams.data(), 3);

        cObjWithUnkowns<tREAL8>* toUk = mBA_Topo->getPoint(mPtsNames[1]).getUK();
        int nbIndBefore = indices.size();
        toUk->PushIndexes(indices);
        indices.resize(nbIndBefore+3); // keep only the point part for cSensorImage UK // TODO: improve
        break;
    }
    case eTopoObsSetType::eNbVals:
        MMVII_INTERNAL_ERROR("unknown obs set type")
    }

/*#ifdef VERBOSE_TOPO
    std::cout<<indices.size()<<" indices:";
    for (auto &i:indices)
        std::cout<<i<<" ";
    std::cout<<std::endl;
#endif*/
    return indices;
}

std::vector<tREAL8> cTopoObs::getVals() const
{
    std::vector<tREAL8> vals;

    switch (mSet->getType()) {
    case eTopoObsSetType::eStation:
    {
        cTopoObsSetStation* set = dynamic_cast<cTopoObsSetStation*>(mSet);
        if (!set)
        {
            MMVII_INTERNAL_ERROR("error set type")
            return {}; //just to please the compiler
        }
        set->PushRotObs(vals);
        cPt3dr* aPtFrom = set->getPtOrigin()->getPt();
        cPt3dr* aPtTo = mBA_Topo->getPoint(mPtsNames[1]).getPt();
        if (mType==eTopoObsType::eZen)
        {
            tREAL8 ref_cor = 0.12 * mBA_Topo->getSysCo()->getDistHzApprox(*aPtFrom, *aPtTo)
                                  / (2*mBA_Topo->getSysCo()->getRadiusApprox(*aPtFrom));
            vals.push_back(ref_cor);
        }
        vals.insert(std::end(vals), std::begin(mMeasures), std::end(mMeasures));
        break;
    }
    case eTopoObsSetType::eNbVals:
        MMVII_INTERNAL_ERROR("unknown obs set type")
    }

/*#ifdef VERBOSE_TOPO
    std::cout<<vals.size()<<" values ";//<<std::endl;
    for (auto&v: vals)
        std::cout<<v<<" ";
    std::cout<<"\n";
#endif*/
    return vals;
}

cResidualWeighterExplicit<tREAL8> &cTopoObs::getWeights()
{
    return mWeights;
}

/*std::vector<tREAL8> cTopoObs::getResiduals(const cTopoComp *comp) const
{
    auto eq = comp->getEquation(getType());
    std::vector<int> indices = getIndices();
    std::vector<tREAL8> vals = getVals();
    std::vector<tREAL8> vUkVal;
    std::for_each(indices.begin(), indices.end(), [&](int i) { vUkVal.push_back(comp->getSys()->CurGlobSol()(i)); });
    std::vector<tREAL8> eval = eq->DoOneEval(vUkVal, vals);
    // select only residuals from formula eval
    std::vector<tREAL8> residuals;
    int valPerSubObs = eval.size() / mWeights.size();
    for (unsigned int i = 0; i < eval.size(); i += valPerSubObs)
        residuals.push_back(eval[i]);
    return residuals;
}*/

};
