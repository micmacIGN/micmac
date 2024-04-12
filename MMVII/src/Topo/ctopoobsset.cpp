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

void cTopoObsSet::init()
{
    createAllowedObsTypes();
    createParams();
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
    cTopoObsSet(aBA_Topo, eTopoObsSetType::eStation), mIsVericalized(true), mIsOriented(false),
    mRotVert2Instr(tRot::Identity()), mRotOmega({0.,0.,0.}), mOriginName(""),mPtOrigin(nullptr)
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

void cTopoObsSetStation::createParams()
{

}

void cTopoObsSetStation::OnUpdate()
{
    // like cPoseWithUK::OnUpdate(), without -...
    mRotVert2Instr = mRotVert2Instr * cRotation3D<tREAL8>::RotFromAxiator(mRotOmega.Pt());

    //std::cout<<"  OnUpdate mRotOmega: "<<mRotOmega.Pt()<<"\n";

    // now this have modify rotation, the "delta" is void :
    mRotOmega.Pt() = cPt3dr(0,0,0);
}

void cTopoObsSetStation::PushRotObs(std::vector<double> & aVObs) const
{
    // TODO: push rotLTF2Vert * mRotVert2Instr
    mRotVert2Instr.Mat().PushByCol(aVObs);
}

std::string cTopoObsSetStation::toString() const
{
    std::ostringstream oss;
    oss<<"   origin: "<<mOriginName;
    if (mPtOrigin)
        oss<<" "<<*mPtOrigin->getPt();
    oss<<"\n   mRotOmega: "<<mRotOmega.Pt()<<"   ";
    oss<<"\n   mRot:\n";
    oss<<"      "<<mRotVert2Instr.AxeI()<<"\n";
    oss<<"      "<<mRotVert2Instr.AxeJ()<<"\n";
    oss<<"      "<<mRotVert2Instr.AxeK()<<"\n";

    return  cTopoObsSet::toString() + oss.str();
}


void cTopoObsSetStation::makeConstraints(cResolSysNonLinear<tREAL8> & aSys)
{
    // TODO: depends on bubbling etc.
    if (mIsVericalized && mIsOriented)
    {
        mRotOmega.Pt() = {0.,0.,0.};
#ifdef VERBOSE_TOPO
        std::cout<<"Freeze rotation for "<<&mRotOmega<<std::endl;
        std::cout<<"  rotation indices "<<mRotOmega.IndUk0()<<"-"<<mRotOmega.IndUk1()-1<<std::endl;
#endif
        aSys.SetFrozenVarCurVal(mRotOmega,mRotOmega.Pt());
        //for (int i=mRotOmega.IndUk0()+3;i<mRotOmega.IndUk1();++i)
        //    aSys.AddEqFixCurVar(i,0.001);
    }
    else if (mIsVericalized)
    {
        mRotOmega.Pt() = {0.,0.,0.};
#ifdef VERBOSE_TOPO
        std::cout<<"Freeze bascule for "<<&mRotOmega<<std::endl;
        std::cout<<"  rotation indices "<<mRotOmega.IndUk0()<<"-"<<mRotOmega.IndUk1()-2<<std::endl;
#endif
        aSys.SetFrozenVarCurVal(mRotOmega,mRotOmega.Pt().PtRawData(), 2); // not z
        //for (int i=mRotOmega.IndUk0()+3;i<mRotOmega.IndUk1()-1;++i)
        //    aSys.AddEqFixCurVar(i,0.001);
    }
    else
    {
        MMVII_INTERNAL_ASSERT_strong(false,"Not fixed orientation station is forbidden for now");
    }
}


bool cTopoObsSetStation::initialize()
{
#ifdef VERBOSE_TOPO
    std::cout<<"cTopoObsSetStation::initialize "<<mOriginName<<std::endl;
#endif
    if (mIsVericalized && mIsOriented)
    {
        return true; // nothing to do
    }
    if (mIsVericalized) // compute initial G0
    {
        for (auto & obs: mObs)
            if (obs->getType() == eTopoObsType::eHz)
            {
                // TODO: use projection for init G0
                // TODO: check if points are init
                auto & aPtTo = mBA_Topo->getPoint(obs->getPointName(1));
                tREAL8 G0 = atan2( aPtTo.getPt()->x() - mPtOrigin->getPt()->x(),
                                   aPtTo.getPt()->y() - mPtOrigin->getPt()->y())
                            - obs->getMeasures().at(0);
                //std::cout<<"Init G0: "<<G0<<std::endl;
                mRotVert2Instr = mRotVert2Instr * cRotation3D<tREAL8>::RotFromAxiator({0., 0., G0});
                return true;
            }
        // if there is no Hz mes, fix ori (TODO: fail if Hz but targets not init)
        mIsOriented = true;
        return initialize();
    }
    MMVII_DEV_WARNING("cTopoObsSetStation initialization not ready for not vericalized stations.")
    return false;
}


void cTopoObsSetStation::setOrigin(std::string _OriginName, bool _IsVericalized)
{
#ifdef VERBOSE_TOPO
    std::cout<<"cTopoObsSetStation::setOrigin "<<_OriginName<<std::endl;
#endif
    mPtOrigin = &mBA_Topo->getPoint(_OriginName);
    mOriginName = _OriginName;
    mIsVericalized = _IsVericalized;
    mRotVert2Instr = tRot::Identity();
    mRotOmega.Pt() = {0.,0.,0.};

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


//----------------------------------------------------------------
cTopoObsSetSubFrame::cTopoObsSetSubFrame(cTopoData& aTopoData) :
    cTopoObsSet(aTopoData, eTopoObsSetType::eSubFrame), mRot(cRotation3D<tREAL8>::RotFromAxiator({0.,0.,0.}))
{
}

void cTopoObsSetSubFrame::createAllowedObsTypes()
{
    mAllowedObsTypes = {eTopoObsType::eSubFrame};
}

void cTopoObsSetSubFrame::createParams()
{
    mParams={0.,0.,0.}; //small rotation axiator
}

void cTopoObsSetSubFrame::OnUpdate()
{
    //update rotation
    mRot = mRot * cRotation3D<tREAL8>::RotFromAxiator(-cPt3dr(mParams[0],mParams[1],mParams[2]));
    mParams={0.,0.,0.};
}

std::vector<tREAL8> cTopoObsSetSubFrame::getRot() const
{
    return
    {
        mRot.Mat().GetElem(0,0), mRot.Mat().GetElem(0,1), mRot.Mat().GetElem(0,2),
        mRot.Mat().GetElem(1,0), mRot.Mat().GetElem(1,1), mRot.Mat().GetElem(1,2),
        mRot.Mat().GetElem(2,0), mRot.Mat().GetElem(2,1), mRot.Mat().GetElem(2,2),
    };
}

void cTopoObsSetSubFrame::makeConstraints(const cResolSysNonLinear<tREAL8> & aSys)
{
    // ?
}
*/
}
