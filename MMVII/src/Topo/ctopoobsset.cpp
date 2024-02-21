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

void cTopoObsSetData::AddData(const  cAuxAr2007 & anAuxInit)
{
    cAuxAr2007 anAux("TopoObsSetData",anAuxInit);
    //std::cout<<"Add data obs set '"<<toString()<<"'"<<std::endl;
    MMVII::EnumAddData(anAux,mType,"Type");
    MMVII::AddData(cAuxAr2007("AllObs",anAux),mObs);
}


void AddData(const cAuxAr2007 & anAux, cTopoObsSetData &aObsSet)
{
     aObsSet.AddData(anAux);
}


// ------------------------------------


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
    cTopoObsSet(aBA_Topo, eTopoObsSetType::eStation), mIsVericalized(true), mIsOriented(true),
    mPoseWithUK(), mOriginName("")
{
}

void cTopoObsSetStation::PutUknowsInSetInterval()
{
#ifdef VERBOSE_TOPO
    std::cout<<"PutUknowsInSetInterval setStation "<<this<<std::endl;
#endif
    mSetInterv->AddOneInterv(mPoseWithUK.Omega());
    mSetInterv->AddOneInterv(mPoseWithUK.Tr());

    if (!mParams.empty())
        mSetInterv->AddOneInterv(mParams);
}


void cTopoObsSetStation::createAllowedObsTypes()
{
    mAllowedObsTypes = {eTopoObsType::eDist,eTopoObsType::eHz,eTopoObsType::eZen};
}


void cTopoObsSetStation::AddToSys(cSetInterUK_MultipeObj<tREAL8> & aSet)
{
    if (!mParams.empty())
        aSet.AddOneInterv(mParams);
#ifdef VERBOSE_TOPO
    std::cout<<"AddToSys SetStation "<<&mPoseWithUK<<std::endl;
#endif
    aSet.AddOneObj(&mPoseWithUK);
}

void cTopoObsSetStation::createParams()
{

}

void cTopoObsSetStation::OnUpdate()
{
    // TODO: use pose update
}

std::string cTopoObsSetStation::toString() const
{
    std::ostringstream oss;
    oss<<"   origin: "<<mOriginName<<" "<<mPoseWithUK.Tr();
    return  cTopoObsSet::toString() + oss.str();
}


void cTopoObsSetStation::makeConstraints(cResolSysNonLinear<tREAL8> & aSys)
{
    // TODO: depends on bubbling etc.
    if (mIsVericalized && mIsOriented)
    {
        mPoseWithUK.Pose().SetRotation(cRotation3D<double>::Identity());
#ifdef VERBOSE_TOPO
        std::cout<<"Freeze rotation for "<<&mPoseWithUK<<" "<<&mPoseWithUK.Omega()<<std::endl;
        std::cout<<"  PoseWithUK indices "<<mPoseWithUK.IndUk0()<<"-"<<mPoseWithUK.IndUk1()-1<<std::endl;
#endif
        aSys.SetFrozenVarCurVal(mPoseWithUK,mPoseWithUK.Omega());
        //for (int i=mPoseWithUK.IndUk0()+3;i<mPoseWithUK.IndUk1();++i)
        //    aSys.AddEqFixCurVar(i,0.001);
    } else {
        MMVII_INTERNAL_ASSERT_strong(false,"Not fixed orientation station is forbidden");
    }
}

void cTopoObsSetStation::setOrigin(std::string _OriginName, bool _IsVericalized)
{
    mOriginName = _OriginName;
    mIsVericalized = _IsVericalized;
    //std::cout<<"setOrigin "<<_OriginName<<std::endl;
    const cTopoPoint & ptOrigin = mBA_Topo->getAllPts().at(mOriginName);
    mPoseWithUK.Tr() = *ptOrigin.getPt();
    mPoseWithUK.Pose().SetRotation(cRotation3D<double>::Identity());
#ifdef VERBOSE_TOPO
    std::cout<<"create mPoseWithUK: "<<&mPoseWithUK<<", rot "<<&mPoseWithUK.Pose().Rot()
            <<"  omega "<<&mPoseWithUK.Omega()<<std::endl;
#endif
}

bool cTopoObsSetStation::mergeUnknowns(cResolSysNonLinear<tREAL8> &aSys)
{
    // TODO: share and update unknowns with SetShared
    for (int aN=0; aN<3; aN++)
    {
        std::vector<int> indices;
        const cTopoPoint & ptOrigin = mBA_Topo->getAllPts().at(mOriginName);
        //tTopoPtUK& ptOrigin = mTopoData.getPointWithUK(mOriginName);
        mPoseWithUK.Tr() = *ptOrigin.getPt(); // be sure values are the same USEFUL??

        indices.push_back(ptOrigin.getUK()->IndUk0()+aN); // origin Uk index
        indices.push_back(mPoseWithUK.IndUk0()+aN); // pose Tr Uk index
#ifdef VERBOSE_TOPO
        std::cout<<"Merge unknowns ";
        for (auto &aI: indices)
            std::cout<<aI<<" ";
        std::cout<<std::endl;
#endif
        aSys.SetShared(indices);
        //cSparseVect<tREAL8> aVectCoef;
        //aVectCoef.AddIV(indices[0],1.);
        //aVectCoef.AddIV(indices[1],-1.);
        //aSys.AddObservationLinear(0.001, aVectCoef, 0.);
    }
    return true;
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
