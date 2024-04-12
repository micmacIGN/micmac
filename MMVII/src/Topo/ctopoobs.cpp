#include "ctopoobs.h"
#include "MMVII_PhgrDist.h"
#include "MMVII_2Include_Serial_Tpl.h"
#include <memory>
#include "ctopoobsset.h"
#include "ctopopoint.h"
#include "MMVII_SysSurR.h"
#include "Topo.h"

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
/*    case eTopoObsType::eSubFrame:
        MMVII_INTERNAL_ASSERT_strong(mSet->getType()==eTopoObsSetType::eSubFrame, "Obs: incorrect set type")
        MMVII_INTERNAL_ASSERT_strong(pts.size()==2, "Obs: incorrect number of points")
        MMVII_INTERNAL_ASSERT_strong(vals.size()==3, "Obs: 3 values should be given")
        MMVII_INTERNAL_ASSERT_strong(aWeights.size()==3, "Obs: 3 weights should be given")
        break;
    case eTopoObsType::eDistParam:
        MMVII_INTERNAL_ASSERT_strong(mSet->getType()==eTopoObsSetType::eDistParam, "Obs: incorrect set type")
        MMVII_INTERNAL_ASSERT_strong(pts.size()==2, "Obs: incorrect number of points")
        MMVII_INTERNAL_ASSERT_strong(vals.empty(), "Obs: value should not be given")
        MMVII_INTERNAL_ASSERT_strong(aWeights.size()==1, "Obs: 1 weight should be given")
        break;*/
    default:
        MMVII_INTERNAL_ERROR("unknown obs type")
    }
    //std::cout<<"DEBUG: create cTopoObs "<<toString()<<"\n";

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
    oss<<"res: ";
    for (auto & val: mLastResiduals)
        oss<<val<<" ";
    return oss.str();
}

std::vector<int> cTopoObs::getIndices(cBA_Topo *aBATopo) const
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
        set->getRotOmega().PushIndexes(indices);
        cObjWithUnkowns<tREAL8>* toUk = aBATopo->getPoint(mPtsNames[1]).getUK();
        int nbIndBefore = indices.size();
        toUk->PushIndexes(indices);
        indices.resize(nbIndBefore+3); // keep only the point part for cSensorImage UK // TODO: improve
        break;
    }
    default:
        MMVII_INTERNAL_ERROR("unknown obs set type")
    }
    /*switch (mType) {
    case eTopoObsType::eDist:
    case eTopoObsType::eDistParam:
    case eTopoObsType::eSubFrame:
        {
            // 2 points
            std::string nameFrom = mPts[0];
            std::string nameTo = mPts[1];
            auto & [ptFromUK, ptFrom3d] = aTopoData->getPointWithUK(nameFrom);
            auto & [ptToUK, ptTo3d] = aTopoData->getPointWithUK(nameTo);
            auto paramsIndices = mSet->getParamIndices();
            indices.insert(std::end(indices), std::begin(paramsIndices), std::end(paramsIndices)); // TODO: use PushIndexes
            ptFromUK->PushIndexes(indices, *ptFrom3d);
            ptToUK->PushIndexes(indices, *ptTo3d);
        }
        break;
    default:
        MMVII_INTERNAL_ERROR("unknown obs type")
    }*/
#ifdef VERBOSE_TOPO
    std::cout<<indices.size()<<" indices:";
    for (auto &i:indices)
        std::cout<<i<<" ";
    std::cout<<std::endl;
#endif
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
        vals.insert(std::end(vals), std::begin(mMeasures), std::end(mMeasures));
        break;
    }
    default:
        MMVII_INTERNAL_ERROR("unknown obs set type")
    }
    /*switch (mType) {
    case eTopoObsType::eDist:
    case eTopoObsType::eDistParam:
        //just send measurments
        vals = mVals;
        break;
    case eTopoObsType::eSubFrame:
    {
        //add rotations to measurments
        cTopoObsSetSubFrame* set = dynamic_cast<cTopoObsSetSubFrame*>(mSet);
        if (!set)
        {
            MMVII_INTERNAL_ERROR("error set type")
            return {}; //just to please the compiler
        }
        vals = set->getRot();
        vals.insert(std::end(vals), std::begin(mVals), std::end(mVals));
        break;
    }
    default:
        MMVII_INTERNAL_ERROR("unknown obs type")
    }*/
#ifdef VERBOSE_TOPO
    std::cout<<vals.size()<<" values ";//<<std::endl;
    for (auto&v: vals)
        std::cout<<v<<" ";
    std::cout<<"\n";
#endif
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
