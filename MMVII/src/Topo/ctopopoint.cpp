#include "ctopopoint.h"
#include "MMVII_PhgrDist.h"
#include "MMVII_2Include_Serial_Tpl.h"
#include <memory>
namespace MMVII
{

cTopoPoint::cTopoPoint(std::string name, const cPtxd<tREAL8,3>& _coord, bool _isFree, const cPtxd<tREAL8, 3> &_sigmas) :
    mName(name), mInitCoord(_coord), mIsFree(_isFree), mSigmas(_sigmas), mVertDefl(std::nullopt),
    mUK(nullptr), mPt(nullptr), mOwnsUK(false)
{
}

cTopoPoint::cTopoPoint() :
    mName(""), mInitCoord(0,0,0), mIsFree(false), mSigmas(), mVertDefl(std::nullopt),
    mUK(nullptr), mPt(nullptr), mOwnsUK(false)
{
}

cTopoPoint::~cTopoPoint()
{
    if (mOwnsUK)
    {
        delete mUK;
    }
}

void cTopoPoint::AddData(const  cAuxAr2007 & anAuxInit)
{
     cAuxAr2007 anAux("TopoPoint",anAuxInit);

     MMVII::AddData(cAuxAr2007("Name",anAux),mName);
     MMVII::AddData(cAuxAr2007("InitCoord",anAux),mInitCoord);
     MMVII::AddData(cAuxAr2007("IsFree",anAux),mIsFree);
     MMVII::AddData(cAuxAr2007("Sigmas",anAux),mSigmas);
     AddOptData(anAux,"VertDefl",mVertDefl);
}

void AddData(const cAuxAr2007 & anAux, cTopoPoint &aTopoPoint)
{
     aTopoPoint.AddData(anAux);
}



void cTopoPoint::findOrMakeUK(const std::vector<cBA_GCP *> & vGCP, cPhotogrammetricProject *aPhProj, const cPt3dr & aCoordIfPureTopo)
{
#ifdef VERBOSE_TOPO
    std::cout<<"findOrMakeUK "<<mName<<": ";
#endif
    MMVII_INTERNAL_ASSERT_strong(!isReady(), "double cTopoPoint::findOrMakeUK for point "+mName);
    /*if (mOwnsUK)
    {
        delete mUK;
        mPt = nullptr;
        mUK = nullptr;
        mOwnsUK = false;
    }*/

    // search among GCP
    for (auto & gcp : vGCP)
    {
        //do not use Shurred GCP
        if (gcp->mGCP_UK.size() != gcp->mMesGCP->MesGCP().size())
            continue;

        for (unsigned int i=0; i<gcp->mMesGCP->MesGCP().size(); ++i )
        {
            if (mName ==gcp->mMesGCP->MesGCP()[i].mNamePt)
            {
                mUK = gcp->mGCP_UK.at(i);
                mPt = &gcp->mGCP_UK.at(i)->Pt();
                mOwnsUK = false;
#ifdef VERBOSE_TOPO
                std::cout<<"is a GCP\n";
#endif
                return;
            }
        }
    }


    // search among cameras
    if (aPhProj)
    {
        cSensorCamPC * aCam = aPhProj->ReadCamPC(mName, true, true);
        if (aCam)
        {
            mUK = aCam;
            mPt = &aCam->Center();
            mOwnsUK = false;
#ifdef VERBOSE_TOPO
            std::cout<<"is a camera\n";
#endif
            return;
        }
    }

    // create pure topo point
    cPt3dr_UK * aPt3dr_UK = new cPt3dr_UK(aCoordIfPureTopo);
    mUK = aPt3dr_UK;
    mPt = &aPt3dr_UK->Pt();
    mOwnsUK = true;
#ifdef VERBOSE_TOPO
    std::cout<<"is a pure topo point "<<*mPt<<"\n";
#endif
    return;
}

cPt3dr* cTopoPoint::getPt() const
{
    MMVII_INTERNAL_ASSERT_strong(isReady(), "Error: UK not ready for pt "+mName)
    return mPt;
}

cObjWithUnkowns<tREAL8>* cTopoPoint::getUK() const
{
    MMVII_INTERNAL_ASSERT_strong(isReady(), "Error: UK not ready for pt "+mName)
    return mUK;
}


void cTopoPoint::setVertDefl(const cPtxd<tREAL8, 2> &_vertDefl)
{
    mVertDefl = _vertDefl;
}

std::string cTopoPoint::toString()
{
    std::ostringstream oss;
    oss<<"TopoPoint "<<mName;
    if (isReady())
        oss<<" "<<*getPt();
    return  oss.str();
}

void cTopoPoint::makeConstraints(cResolSysNonLinear<tREAL8> & aSys)
{
    if (!mIsFree)
    {
#ifdef VERBOSE_TOPO
        std::cout<<"makeConstraints for point "<<mName<<" indices "<<
                   mUK->IndUk0()<<"-"<<mUK->IndUk1()-1<<std::endl;
#endif

        aSys.SetFrozenVarCurVal(*getUK(),*getPt());
        //for (int i=mUK->IndUk0(); i<mUK->IndUk1(); ++i)
        //    aSys.AddEqFixCurVar(i, 0.001);

    } else {
#ifdef VERBOSE_TOPO
        std::cout<<"no constraintes for point "<<mName<<" indices "<<
                   mUK->IndUk0()<<"-"<<mUK->IndUk1()-1<<std::endl;
#endif
    }
}


void cTopoPoint::AddToSys(cSetInterUK_MultipeObj<tREAL8> & aSet)
{
#ifdef VERBOSE_TOPO
    std::cout<<"AddToSys pt "<<mName<<" "<<this<<std::endl;
#endif
    if (doesOwnsUK())
        aSet.AddOneObj(getUK());
}

};
