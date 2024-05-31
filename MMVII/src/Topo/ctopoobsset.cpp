#include "ctopoobsset.h"
#include "MMVII_PhgrDist.h"
#include "MMVII_2Include_Serial_Tpl.h"
#include "ctopoobs.h"
#include "ctopopoint.h"
#include "ctopodata.h"
#include "MMVII_PCSens.h"
#include "Topo.h"
#include <memory>
namespace MMVII
{

cTopoObsSet::cTopoObsSet(cBA_Topo * aBA_Topo, eTopoObsSetType type):
    mType(type), mBA_Topo(aBA_Topo)
{
}

cTopoObsSet::~cTopoObsSet()
{
    //std::cout<<"delete set"<<std::endl;
    std::for_each(mObs.begin(), mObs.end(), [](auto o){ delete o; });
}


void cTopoObsSet::AddToSys(cSetInterUK_MultipeObj<tREAL8> & aSet)
{
#ifdef VERBOSE_TOPO
    std::cout<<"AddToSys set "<<this<<std::endl;
#endif
    PutUknowsInSetInterval();
}

void cTopoObsSet::create()
{
    createAllowedObsTypes();
}


void cTopoObsSet::PutUknowsInSetInterval()
{
#ifdef VERBOSE_TOPO
    std::cout<<"PutUknowsInSetInterval set "<<this<<std::endl;
#endif
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
    mRotSysCo2Vert(tRot::Identity()), mRotVert2Instr(tRot::Identity()), mRotOmega({0.,0.,0.}),
    mOriginName(""), mPtOrigin(nullptr)
{
}

void cTopoObsSetStation::PutUknowsInSetInterval()
{
    cTopoObsSet::PutUknowsInSetInterval();

#ifdef VERBOSE_TOPO
    std::cout<<"PutUknowsInSetInterval setStation "<<this<<std::endl;
#endif
}


void cTopoObsSetStation::createAllowedObsTypes()
{
    mAllowedObsTypes = {
        eTopoObsType::eDist,eTopoObsType::eHz,eTopoObsType::eZen,
        eTopoObsType::eDX,eTopoObsType::eDY,eTopoObsType::eDZ
    };
}


void cTopoObsSetStation::AddToSys(cSetInterUK_MultipeObj<tREAL8> & aSet)
{
    cTopoObsSet::AddToSys(aSet);

#ifdef VERBOSE_TOPO
    std::cout<<"AddToSys SetStation mRotOmega "<<&mRotOmega<<std::endl;
#endif
    aSet.AddOneObj(&mRotOmega);
    aSet.AddOneObj(this); // to have OnUpdate called on SetVUnKnowns
}


void cTopoObsSetStation::OnUpdate()
{
    mRotOmega.Pt() = mRotVert2Instr.Inverse(mRotOmega.Pt()); // TODOJM: why ?????

    // like cPoseWithUK::OnUpdate(), without -...
    mRotVert2Instr = mRotVert2Instr * cRotation3D<tREAL8>::RotFromAxiator(mRotOmega.Pt());

    // update mRotSysCo2Vert with new station position
    mRotSysCo2Vert = mBA_Topo->getSysCo()->getVertical(*mPtOrigin->getPt());

    //std::cout<<"  OnUpdate mRotOmega: "<<mRotOmega.Pt()<<"\n";

    // now this have modified rotation, the "delta" is void:
    mRotOmega.Pt() = cPt3dr(0,0,0);
}

void cTopoObsSetStation::PushRotObs(std::vector<double> & aVObs) const
{
    (mRotSysCo2Vert * mRotVert2Instr).Mat().PushByCol(aVObs);
}

std::string cTopoObsSetStation::toString() const
{
    std::ostringstream oss;
    oss<<"   origin: "<<mOriginName;
    if (mPtOrigin)
        oss<<" "<<*mPtOrigin->getPt();
    oss<<"\n   RotOmega: "<<mRotOmega.Pt()<<"   ";
    oss<<"\n   Rot:\n";
    oss<<"      "<<mRotVert2Instr.AxeI()<<"\n";
    oss<<"      "<<mRotVert2Instr.AxeJ()<<"\n";
    oss<<"      "<<mRotVert2Instr.AxeK()<<"\n";
    oss<<"   "<<E2Str(mOriStatus)<<"\n";

    oss<<"\n   RotSysCo2Vert:\n";
    oss<<"      "<<mRotSysCo2Vert.AxeI()<<"\n";
    oss<<"      "<<mRotSysCo2Vert.AxeJ()<<"\n";
    oss<<"      "<<mRotSysCo2Vert.AxeK()<<"\n";

    /*std::cout<<"cTopoObsSetStation rot:\n";
    std::cout<<"   "<<mRotSysCo2Vert.AxeI()<<"       "<<mRotVert2Instr.AxeI()<<"       "<<(mRotSysCo2Vert * mRotVert2Instr).AxeI()<<"\n";
    std::cout<<"   "<<mRotSysCo2Vert.AxeJ()<<"   *   "<<mRotVert2Instr.AxeJ()<<"   =   "<<(mRotSysCo2Vert * mRotVert2Instr).AxeJ()<<"\n";
    std::cout<<"   "<<mRotSysCo2Vert.AxeK()<<"       "<<mRotVert2Instr.AxeK()<<"       "<<(mRotSysCo2Vert * mRotVert2Instr).AxeK()<<"\n";*/
    return  cTopoObsSet::toString() + oss.str();
}


void cTopoObsSetStation::makeConstraints(cResolSysNonLinear<tREAL8> & aSys)
{
    mRotOmega.Pt() = {0.,0.,0.};

    switch (mOriStatus)
    {
    case(eTopoStOriStat::eTopoStOriFixed):
        mRotOmega.Pt() = {0.,0.,0.};
#ifdef VERBOSE_TOPO
        std::cout<<"Freeze rotation for "<<&mRotOmega<<std::endl;
        std::cout<<"  rotation indices "<<mRotOmega.IndUk0()<<"-"<<mRotOmega.IndUk1()-1<<std::endl;
#endif
        aSys.SetFrozenVarCurVal(mRotOmega,mRotOmega.Pt());
        //for (int i=mRotOmega.IndUk0()+3;i<mRotOmega.IndUk1();++i)
        //    aSys.AddEqFixCurVar(i,0.001);
        break;
    case(eTopoStOriStat::eTopoStOriVert):
#ifdef VERBOSE_TOPO
        std::cout<<"Freeze bascule for "<<&mRotOmega<<std::endl;
        std::cout<<"  rotation indices "<<mRotOmega.IndUk0()<<"-"<<mRotOmega.IndUk1()-2<<std::endl;
#endif
        aSys.SetFrozenVarCurVal(mRotOmega,mRotOmega.Pt().PtRawData(), 2); // not z
        //for (int i=mRotOmega.IndUk0()+3;i<mRotOmega.IndUk1()-1;++i)
        //    aSys.AddEqFixCurVar(i,0.001);
        break;
    case(eTopoStOriStat::eTopoStOriBasc):
        // free rotation: nothing to constrain
        break;
    case(eTopoStOriStat::eNbVals):
        MMVII_INTERNAL_ASSERT_strong(false, "cTopoObsSetStation::makeConstraints: incorrect ori status")
    }
}


bool cTopoObsSetStation::initialize(const cTopoObsSetData * aData)
{
#ifdef VERBOSE_TOPO
    std::cout<<"cTopoObsSetStation::initialize "<<mOriginName<<std::endl;
#endif
    // set parameters
    if (aData)
        mOriStatus = aData->mStationOriStat.value_or(eTopoStOriStat::eTopoStOriVert);

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

    // initialize
    // mRotSysCo2Vert is initialized by setOrigin()
    switch (mOriStatus)
    {
    case(eTopoStOriStat::eTopoStOriFixed):
        return true; // nothing to do
    case(eTopoStOriStat::eTopoStOriVert):
    {
        cTopoObs * aObsDX = nullptr;
        cTopoObs * aObsDY = nullptr;
        tREAL8 G0 = NAN;
        for (auto & obs: mObs)
        {
            if (obs->getType() == eTopoObsType::eHz)
            {
                // TODO: use projection for init G0
                // TODO: check if points are init
                auto & aPtTo = mBA_Topo->getPoint(obs->getPointName(1));
                G0 = atan2( aPtTo.getPt()->x() - mPtOrigin->getPt()->x(),
                                   aPtTo.getPt()->y() - mPtOrigin->getPt()->y())
                            - obs->getMeasures().at(0);
                break;
            }
            if (obs->getType() == eTopoObsType::eDX && (!aObsDY || (aObsDY->getPointName(1)==obs->getPointName(1))))
                aObsDX = obs;
            if (obs->getType() == eTopoObsType::eDY && (!aObsDX || (aObsDX->getPointName(1)==obs->getPointName(1))))
                aObsDY = obs;
        }
        if (aObsDX && aObsDY) // compute G0 from DX DY if Hz not found
        {
            // TODO: use projection for init G0
            // TODO: check if points are init
            auto & aPtTo = mBA_Topo->getPoint(aObsDX->getPointName(1));
            G0 = atan2( aPtTo.getPt()->x() - mPtOrigin->getPt()->x(),
                               aPtTo.getPt()->y() - mPtOrigin->getPt()->y())
                        - atan2( aObsDX->getMeasures()[0], aObsDY->getMeasures()[0]);
        }
        if (std::isfinite(G0))
        {
            mRotVert2Instr = cRotation3D<tREAL8>::RotFromAxiator({0., 0., G0});
#ifdef VERBOSE_TOPO
            std::cout<<"Init G0: "<<G0<<std::endl;
            std::cout<<"Init mRotVert2Instr:\n";
            std::cout<<"    "<<mRotVert2Instr.AxeI()<<"\n";
            std::cout<<"    "<<mRotVert2Instr.AxeJ()<<"\n";
            std::cout<<"    "<<mRotVert2Instr.AxeK()<<"\n";
#endif
            return true;
        }
        return false;
    }
    case(eTopoStOriStat::eTopoStOriBasc):
        MMVII_DEV_WARNING("cTopoObsSetStation rotation initialization not ready.")
        // TODO: T-S?
        return true;
    case(eTopoStOriStat::eNbVals):
        MMVII_INTERNAL_ASSERT_strong(false, "cTopoObsSetStation::initialize: incorrect ori status")
        return false;
    }

    return false;
}


void cTopoObsSetStation::setOrigin(std::string _OriginName)
{
#ifdef VERBOSE_TOPO
    std::cout<<"cTopoObsSetStation::setOrigin "<<_OriginName<<std::endl;
#endif
    mPtOrigin = &mBA_Topo->getPoint(_OriginName);
    mOriginName = _OriginName;

    // automatic origin initialization
    if (!mPtOrigin->getPt()->IsValid())
    {
        MMVII_DEV_WARNING("cTopoObsSetStation origin initialization not ready.")
        *mPtOrigin->getPt() = {0.,0.,0.};
    }

    mRotVert2Instr = tRot::Identity();
    mRotOmega.Pt() = {0.,0.,0.};
    mRotSysCo2Vert = mBA_Topo->getSysCo()->getVertical(*mPtOrigin->getPt());
}

tREAL8 cTopoObsSetStation::getG0() const
{
    return atan2(mRotVert2Instr.Mat().GetElem(0,1), mRotVert2Instr.Mat().GetElem(0,0));
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
