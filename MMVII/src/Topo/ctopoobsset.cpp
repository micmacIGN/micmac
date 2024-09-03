#include "ctopoobsset.h"
#include "MMVII_PhgrDist.h"
#include "MMVII_2Include_Serial_Tpl.h"
#include "MMVII_PCSens.h"
#include "MMVII_Topo.h"
#include <memory>
namespace MMVII
{

cTopoObsSet::cTopoObsSet(cBA_Topo * aBA_Topo, eTopoObsSetType type):
    mType(type), 
    mBA_Topo(aBA_Topo), 
    mInit(false)
{
}

cTopoObsSet::~cTopoObsSet()
{
    std::for_each(mObs.begin(), mObs.end(), [](auto o){ delete o; });
}

void cTopoObsSet::create()
{
    createAllowedObsTypes();
}


void cTopoObsSet::PutUknowsInSetInterval()
{
    if (!mParams.empty())
        mSetInterv->AddOneInterv(mParams);
}

bool cTopoObsSet::addObs(eTopoObsType obsType, cBA_Topo *aBA_Topo, const std::vector<std::string> &pts, const std::vector<tREAL8> &vals, const cResidualWeighterExplicit<tREAL8> &aWeights)
{
    if (std::find(mAllowedObsTypes.begin(), mAllowedObsTypes.end(), obsType) == mAllowedObsTypes.end())
    {
        ErrOut() << "Error, " << E2Str(obsType)
                 << " obs type is not allowed in "
                 << E2Str(mType)<<" obs set!\n";
        return false;
    }
    mObs.push_back(new cTopoObs(this, aBA_Topo, obsType, pts, vals, aWeights));
    return true;
}

std::string cTopoObsSet::toString() const
{
    std::ostringstream oss;
    oss<<"TopoObsSet "<<E2Str(mType)<<":\n";
    for (auto & obs: mObs)
        oss<<"    - "<<obs->toString()<<"\n";
    if (!mParams.empty())
    {
        oss<<"    params: ";
        for (auto & param: mParams)
            oss<<param<<" ";
        oss<<"\n";
    }
    return  oss.str();
}

std::vector<int> cTopoObsSet::getParamIndices() const
{
    std::vector<int> indices;
    for (auto & param : mParams)
    {
        indices.push_back((int)IndOfVal(&param));
    }
    return indices;
}

//----------------------------------------------------------------
cTopoObsSetStation::cTopoObsSetStation(cBA_Topo *aBA_Topo) :
    cTopoObsSet(aBA_Topo, eTopoObsSetType::eStation), mOriStatus(eTopoStOriStat::eTopoStOriVert),
    mRotSysCo2Vert(tRot::Identity()), mRotVert2Instr(tRot::Identity()),
    mOriginName(""), mPtOrigin(nullptr)
{
    mParams = {0.,0.,0.}; // the rotation axiator unknown
}


void cTopoObsSetStation::createAllowedObsTypes()
{
    mAllowedObsTypes = {
        eTopoObsType::eDist,eTopoObsType::eHz,eTopoObsType::eZen,
        eTopoObsType::eDX,eTopoObsType::eDY,eTopoObsType::eDZ,
        eTopoObsType::eDH
    };
}


void cTopoObsSetStation::OnUpdate()
{
    auto aRotOmega = getRotOmega();
    aRotOmega = mRotVert2Instr.Inverse(aRotOmega); // see cPoseF comments

    // like cPoseWithUK::OnUpdate(), without -...
    mRotVert2Instr = mRotVert2Instr * cRotation3D<tREAL8>::RotFromAxiator(aRotOmega);

    // update mRotSysCo2Vert with new station position
    updateVertMat();

    //StdOut() << "  OnUpdate mRotOmega: "<<mRotOmega.Pt()<<"\n";

    // now this have modified rotation, the "delta" is void:
    resetRotOmega();
}

void cTopoObsSetStation::PushRotObs(std::vector<double> & aVObs) const
{
    // fill aPoseInstr2RTL
    getRotSysCo2Instr().Mat().PushByCol(aVObs);
}

void  cTopoObsSetStation::GetAdrInfoParam(cGetAdrInfoParam<tREAL8> & aGAIP)
{
    aGAIP.SetNameType("Station");
    aGAIP.SetIdObj(mOriginName);
    aGAIP.TestParam(this, &(mParams[0])    ,"Wx");
    aGAIP.TestParam(this, &(mParams[1])    ,"Wy");
    aGAIP.TestParam(this, &(mParams[2])    ,"Wz");
}

std::string cTopoObsSetStation::toString() const
{
    std::ostringstream oss;
    oss<<"   origin: "<<mOriginName;
    if (mPtOrigin)
        oss<<" "<<*mPtOrigin->getPt();
    oss<<"\n   Rot:\n";
    oss<<"      "<<mRotVert2Instr.AxeI()<<"\n";
    oss<<"      "<<mRotVert2Instr.AxeJ()<<"\n";
    oss<<"      "<<mRotVert2Instr.AxeK()<<"\n";
    oss<<"   "<<E2Str(mOriStatus)<<"\n";

    oss<<"\n   RotSysCo2Vert:\n";
    oss<<"      "<<mRotSysCo2Vert.AxeI()<<"\n";
    oss<<"      "<<mRotSysCo2Vert.AxeJ()<<"\n";
    oss<<"      "<<mRotSysCo2Vert.AxeK()<<"\n";

    return  cTopoObsSet::toString() + oss.str();
}


void cTopoObsSetStation::makeConstraints(cResolSysNonLinear<tREAL8> & aSys)
{
    resetRotOmega();
    switch (mOriStatus)
    {
    case(eTopoStOriStat::eTopoStOriContinue):
        // should not exist, Continue is only for obs files
        MMVII_INTERNAL_ASSERT_strong(false, "cTopoObsSetStation::makeConstraints: incorrect ori status")
        break;
    case(eTopoStOriStat::eTopoStOriFixed):
#ifdef VERBOSE_TOPO
        StdOut() << "Freeze rotation for "<<mParams.data()<<std::endl;
        StdOut() << "  rotation indices "<<IndUk0()<<"-"<<IndUk1()-1<<std::endl;
#endif
        aSys.SetFrozenVarCurVal(*this,mParams);
        break;
    case(eTopoStOriStat::eTopoStOriVert):
#ifdef VERBOSE_TOPO
        StdOut() << "Freeze bascule for "<<mParams.data()<<std::endl;
        StdOut() << "  rotation indices "<<IndUk0()<<"-"<<IndUk1()-2<<std::endl;
#endif
        aSys.SetFrozenVarCurVal(*this,mParams.data(), 2); // not z
        break;
    case(eTopoStOriStat::eTopoStOriBasc):
        // free rotation: nothing to constrain
        break;
    case(eTopoStOriStat::eNbVals):
        MMVII_INTERNAL_ASSERT_strong(false, "cTopoObsSetStation::makeConstraints: incorrect ori status")
    }
}


bool cTopoObsSetStation::initialize()
{
#ifdef VERBOSE_TOPO
    StdOut() <<"cTopoObsSetStation::initialize "<<mOriginName<<std::endl;
#endif

    // auto fix mStationIsOriented if has orientation obs
    bool hasOriObs = false;
    for (auto & obs: mObs)
    {
        switch (obs->getType()) {
        case eTopoObsType::eHz:
        case eTopoObsType::eDX:
        case eTopoObsType::eDY:
            hasOriObs = true;
            break;
        default:
            break;
        }
        if (hasOriObs)
            break;
    }

    if (!hasOriObs)
        mOriStatus = eTopoStOriStat::eTopoStOriFixed;

    // set origin
    std::string aOriginName;

    MMVII_INTERNAL_ASSERT_User(getAllObs().size()>0, eTyUEr::eUnClassedError, "Error: Obs Set without obs.")
    aOriginName = getObs(0)->getPointName(0);
    // check that every obs goes from the same point
    for (auto &aObs : getAllObs())
    {
        MMVII_INTERNAL_ASSERT_User(aObs->getPointName(0)==aOriginName, eTyUEr::eUnClassedError, "Error: Obs Set with several origins")
    }
    setOrigin(aOriginName); // use 1st from name as station name

    mInit = false;
    // initialize
    // mRotSysCo2Vert is initialized by setOrigin()
    switch (mOriStatus)
    {
    case(eTopoStOriStat::eTopoStOriContinue):
        // should not exist, Continue is only for obs files
        MMVII_INTERNAL_ASSERT_strong(false, "cTopoObsSetStation::initialize: incorrect ori status")
        break;
    case(eTopoStOriStat::eTopoStOriFixed):
        mInit = mPtOrigin->isInit();
        return true; // nothing to do
    case(eTopoStOriStat::eTopoStOriVert):
    {
        if (!mPtOrigin->isInit())
            return false;
        tREAL8 G0 = NAN;
        for (auto & obs: mObs)
        {
            auto & aPtTo = mBA_Topo->getPoint(obs->getPointName(1));
            if (!aPtTo.isInit())
                continue;
            if (obs->getType() == eTopoObsType::eHz)
            {
                cPt3dr aVectInstr = PtSysCo2Vert(aPtTo);
                G0 = atan2( aVectInstr.x(), aVectInstr.y()) - obs->getMeasures().at(0);
#ifdef VERBOSE_TOPO
                StdOut()<<"G0 hz: "<<*mPtOrigin->getPt()<<" -> "<<*aPtTo.getPt()<<" mes "<<obs->getMeasures().at(0)<<"\n";
#endif
                break;
            }
        }
        if (!std::isfinite(G0)) // try to init with DX and DY
        {
            std::map<std::string, std::pair<cTopoObs *,cTopoObs *> > aBigDxDyPerPt;
            for (auto & obs: mObs)
            {
                auto aPtTo = mBA_Topo->getPoint(obs->getPointName(1)).getName();
                if (obs->getType() == eTopoObsType::eDX)
                {
                    if (aBigDxDyPerPt.count(aPtTo)==0)
                        aBigDxDyPerPt[aPtTo] = {obs, nullptr};
                    else
                        if ((!aBigDxDyPerPt[aPtTo].first)
                            || (obs->getMeasures()[0] > fabs(aBigDxDyPerPt[aPtTo].first->getMeasures()[0])))
                                aBigDxDyPerPt[aPtTo].first = obs;
                }
                if (obs->getType() == eTopoObsType::eDY)
                {
                    if (aBigDxDyPerPt.count(aPtTo)==0)
                        aBigDxDyPerPt[aPtTo] = {nullptr, obs};
                    else
                        if ((!aBigDxDyPerPt[aPtTo].second)
                            || (obs->getMeasures()[0] > fabs(aBigDxDyPerPt[aPtTo].second->getMeasures()[0])))
                                aBigDxDyPerPt[aPtTo].second = obs;
                }
            }
            double aLongestObsDist2 = -1.;
            std::string aLongestObsName = "";
            for (auto & [aPtTo, aVal]: aBigDxDyPerPt)
            {
                auto & [aObsDX, aObsDY] = aVal;
                if (aObsDX && aObsDY)
                {
                    double dX = aObsDX->getMeasures()[0];
                    double dY = aObsDY->getMeasures()[0];
                    double aCurrDist2 = dX*dX + dY*dY;
                    if (aCurrDist2 > aLongestObsDist2)
                    {
                        aLongestObsDist2 = aCurrDist2;
                        aLongestObsName = aPtTo;
                    }
                }
            }
            if (aLongestObsDist2>0.) // compute G0 from DX DY if Hz not found
            {
                auto & [aObsDX, aObsDY] = aBigDxDyPerPt[aLongestObsName];
                auto & aPtTo = mBA_Topo->getPoint(aObsDX->getPointName(1));
                cPt3dr aVectInstr = PtSysCo2Vert(aPtTo);
                G0 = atan2( aVectInstr.x(), aVectInstr.y())
                            - atan2( aObsDX->getMeasures()[0], aObsDY->getMeasures()[0]);
    #ifdef VERBOSE_TOPO
                StdOut()<<"G0 dxy: "<<*mPtOrigin->getPt()<<" -> "<<*aPtTo.getPt()<<" mes "<<aObsDX->getMeasures()[0]<<" "<<aObsDY->getMeasures()[0]<<"\n";
    #endif
            }
        }
        if (std::isfinite(G0))
        {
            mRotVert2Instr = cRotation3D<tREAL8>::RotFromAxiator({0., 0., G0});
#ifdef VERBOSE_TOPO
            StdOut() << "Init G0: "<<G0<<std::endl;
            StdOut() << "Init mRotVert2Instr:\n";
            StdOut() << "    "<<mRotVert2Instr.AxeI()<<"\n";
            StdOut() << "    "<<mRotVert2Instr.AxeJ()<<"\n";
            StdOut() << "    "<<mRotVert2Instr.AxeK()<<"\n";
#endif
            mInit = true;
            return true;
        }
        return false;
    }
    case(eTopoStOriStat::eTopoStOriBasc):
        MMVII_DEV_WARNING("cTopoObsSetStation rotation initialization not ready.")
        // TODO: T-S?
        mInit = mPtOrigin->isInit();
        return true;
    case(eTopoStOriStat::eNbVals):
        MMVII_INTERNAL_ASSERT_strong(false, "cTopoObsSetStation::initialize: incorrect ori status")
        return false;
    }

    return false;
}


void cTopoObsSetStation::updateVertMat()
{
    if ((mOriStatus == eTopoStOriStat::eTopoStOriFixed) && (mBA_Topo->getSysCo()->getType()!=eSysCo::eRTL))
        mRotSysCo2Vert = tRot::Identity(); // do not seach for vertical if all fixed, to work will all SysCo
    else
        mRotSysCo2Vert = mBA_Topo->getSysCo()->getRot2Vertical(*mPtOrigin->getPt());
}

void cTopoObsSetStation::resetRotOmega()
{
    std::fill(mParams.begin(), mParams.end(), 0.); // makes sure to keep the same data address
}

void cTopoObsSetStation::setOrigin(std::string _OriginName)
{
#ifdef VERBOSE_TOPO
    StdOut() << "cTopoObsSetStation::setOrigin "<<_OriginName<<std::endl;
#endif
    mPtOrigin = &mBA_Topo->getPoint(_OriginName);
    mOriginName = _OriginName;

    mRotVert2Instr = tRot::Identity();
    resetRotOmega();
    if (mPtOrigin->isInit())
        updateVertMat();
}

tREAL8 cTopoObsSetStation::getG0() const
{
    return atan2(mRotVert2Instr.Mat().GetElem(0,1), mRotVert2Instr.Mat().GetElem(0,0));
}

cPt3dr cTopoObsSetStation::PtSysCo2Vert(const cTopoPoint & aPt) const
{
    return mRotSysCo2Vert.Value(*aPt.getPt() - *mPtOrigin->getPt());
}

cPt3dr cTopoObsSetStation::PtSysCo2Instr(const cTopoPoint &aPt) const
{
    return getRotSysCo2Instr().Value(*aPt.getPt() - *mPtOrigin->getPt());
}

cPt3dr cTopoObsSetStation::PtInstr2SysCo(const cPt3dr &aVect) const
{
    return getRotSysCo2Instr().Inverse(aVect) + *mPtOrigin->getPt();
}
/*
//----------------------------------------------------------------
cTopoObsSetDistParam::cTopoObsSetDistParam(cTopoData& aTopoData) :
    cTopoObsSet(aTopoData, eTopoObsSetType::eDistParam)
{
}

void cTopoObsSetDistParam::createAllowedObsTypes()
{
    mAllowedObsTypes = {eTopoObsType::eDistParam};
}

void cTopoObsSetDistParam::createParams()
{
    mParams.push_back(0.0); //the distance
}

void cTopoObsSetDistParam::OnUpdate()
{
    // nothing to do
}

void cTopoObsSetDistParam::makeConstraints(const cResolSysNonLinear<tREAL8> &aSys)
{
    // nothing to do
}

*/
}
