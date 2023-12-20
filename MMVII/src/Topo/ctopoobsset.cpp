#include "ctopoobsset.h"
#include "MMVII_PhgrDist.h"
#include "MMVII_2Include_Serial_Tpl.h"
#include "ctopoobs.h"
#include <memory>
namespace MMVII
{

cTopoObsSet::cTopoObsSet(eTopoObsSetType type):
    mType(type)
{
}

void cTopoObsSet::AddData(const  cAuxAr2007 & anAuxInit)
{
    std::cout<<"Add data obs set '"<<toString()<<"'"<<std::endl;
    cAuxAr2007 anAux("TopoObsSet",anAuxInit);

    MMVII::EnumAddData(anAux,mType,"Type");
    MMVII::AddData(cAuxAr2007("AllObs",anAux),mObs);
}

void AddData(const cAuxAr2007 & anAux, std::unique_ptr<cTopoObsSet> &aObsSet)
{
     aObsSet->AddData(anAux);
}

void cTopoObsSet::init()
{
    createAllowedObsTypes();
    createParams();
}

void cTopoObsSet::PutUknowsInSetInterval()
{
    mSetInterv->AddOneInterv(mParams);
}

bool cTopoObsSet::addObs(cTopoObs * obs)
{
    if (std::find(mAllowedObsTypes.begin(), mAllowedObsTypes.end(), obs->getType()) == mAllowedObsTypes.end())
    {
        ErrOut() << "Error, " << obs.type2string()
                 << " obs type is not allowed in "
                 << type2string()<<" obs set!\n";
        return false;
    }
    mObs.push_back(obs);
    return true;
}

std::string cTopoObsSet::toString() const
{
    std::ostringstream oss;
    oss<<"TopoObsSet "<<type2string()<<":\n";
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
    //TODO: compute it in PutUknowsInSetInterval()?
    std::vector<int> indices;
    for (auto & param : mParams)
    {
        indices.push_back((int)IndOfVal(&param));
    }
    return indices;
}

//----------------------------------------------------------------
cTopoObsSetSimple::cTopoObsSetSimple() :
    cTopoObsSet(eTopoObsSetType::eSimple)
{
}

void cTopoObsSetSimple::createAllowedObsTypes()
{
    mAllowedObsTypes = {eTopoObsType::eDist};
}

void cTopoObsSetSimple::createParams()
{
    //no params
}

void cTopoObsSetSimple::OnUpdate()
{
    //nothing to do
}

std::string cTopoObsSetSimple::type2string() const
{
    return "simple";
}

//----------------------------------------------------------------
cTopoObsSetDistParam::cTopoObsSetDistParam() :
    cTopoObsSet(eTopoObsSetType::eDistParam)
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
    //nothing to do
}

std::string cTopoObsSetDistParam::type2string() const
{
    return "distParam";
}

//----------------------------------------------------------------
cTopoObsSetSubFrame::cTopoObsSetSubFrame() :
    cTopoObsSet(eTopoObsSetType::eSubFrame), mRot(cRotation3D<tREAL8>::RotFromAxiator({0.,0.,0.}))
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

std::string cTopoObsSetSubFrame::type2string() const
{
    return "subFrame";
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

}
