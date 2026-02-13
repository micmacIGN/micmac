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
    case eTopoObsType::eDist:
    case eTopoObsType::eDH:
        MMVII_INTERNAL_ASSERT_strong(mSet->getType()==eTopoObsSetType::eSimple, "Obs: incorrect set type")
        MMVII_INTERNAL_ASSERT_strong(ptsNames.size()==2, "Obs: incorrect number of points")
        MMVII_INTERNAL_ASSERT_strong(measures.size()==1, "Obs: 1 value should be given")
        MMVII_INTERNAL_ASSERT_strong(aWeights.size()==1, "Obs: 1 weight should be given")
        break;
    case eTopoObsType::eHz:
    case eTopoObsType::eZen:
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
    case eTopoObsSetType::eSimple:
    {
        cTopoObsSetSimple* set = dynamic_cast<cTopoObsSetSimple*>(mSet);
        if (!set)
        {
            MMVII_INTERNAL_ERROR("error set type")
            return {}; //just to please the compiler
        }
        cObjWithUnkowns<tREAL8>* fromUk = mBA_Topo->getPoint(mPtsNames[0]).getUK();
        cObjWithUnkowns<tREAL8>* toUk = mBA_Topo->getPoint(mPtsNames[1]).getUK();
        fromUk->PushIndexes(indices);
        toUk->PushIndexes(indices);
        break;
    }
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
    case eTopoObsSetType::eSimple:
    {
        cTopoObsSetSimple* set = dynamic_cast<cTopoObsSetSimple*>(mSet);
        if (!set)
        {
            MMVII_INTERNAL_ERROR("error set type")
            return {}; //just to please the compiler
        }
        if (mType==eTopoObsType::eDH)
        {
            auto aSysCo = mBA_Topo->getSysCo();
            // RTL to GeoC transfo, as matrix + translation
            const tPoseR* aTranfo2GeoC = aSysCo->getTranfo2GeoC();
            aTranfo2GeoC->Rot().Mat().PushByLine(vals); // TODO: why by line?
            aTranfo2GeoC->Tr().PushInStdVector(vals);
            // a
            vals.push_back(aSysCo->getEllipsoid_a());
            // e2
            vals.push_back(aSysCo->getEllipsoid_e2());
            /*
            cPt3dr* aPtFrom = mBA_Topo->getPoint(mPtsNames[0]).getPt();
            cPt3dr* aPtTo = mBA_Topo->getPoint(mPtsNames[1]).getPt();
            //Phi_from
            auto aPtFromGeoG = aSysCo->toGeoG(*aPtFrom);
            auto aPhiFrom = aPtFromGeoG.y()/AngleFromRad(eTyUnitAngle::eUA_degree);
            vals.push_back(aPhiFrom);
            //M_from = a*sqrt(1-e*e*sin(phi)*sin(phi))
            auto aPtFromM = aSysCo->getEllipsoid_a()
                    *sqrt(1-aSysCo->getEllipsoid_e2()*sin(aPhiFrom)*sin(aPhiFrom));
            vals.push_back(aPtFromM);
            //Phi_to
            auto aPtToGeoG = aSysCo->toGeoG(*aPtTo);
            auto aPhiTo = aPtToGeoG.y()/AngleFromRad(eTyUnitAngle::eUA_degree);
            vals.push_back(aPhiTo);
            //M_To = a*sqrt(1-e*e*sin(phi)*sin(phi))
            auto aPtToM = aSysCo->getEllipsoid_a()
                    *sqrt(1-aSysCo->getEllipsoid_e2()*sin(aPhiTo)*sin(aPhiTo));
            vals.push_back(aPtToM);*/
        }
        vals.insert(std::end(vals), std::begin(mMeasures), std::end(mMeasures));
        break;
    }
    case eTopoObsSetType::eStation:
    {
        cTopoObsSetStation* set = dynamic_cast<cTopoObsSetStation*>(mSet);
        if (!set)
        {
            MMVII_INTERNAL_ERROR("error set type")
            return {}; //just to please the compiler
        }
        cPt3dr* aPtFrom = set->getPtOrigin()->getPt();
        cPt3dr* aPtTo = mBA_Topo->getPoint(mPtsNames[1]).getPt();
        set->PushRotObs(vals);
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
