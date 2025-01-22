#include "ctopopoint.h"
#include "MMVII_PhgrDist.h"
#include "MMVII_2Include_Serial_Tpl.h"
#include "MMVII_util_tpl.h"
#include <memory>
namespace MMVII
{

cTopoPoint::cTopoPoint(const std::string &name) :
    mName(name), mInitCoord(cPt3dr::Dummy()),
    mVertDefl(std::nullopt), mUK(nullptr), mPt(nullptr)
{
}


cTopoPoint::cTopoPoint() :
    mName(""), mInitCoord(cPt3dr::Dummy()), mVertDefl(std::nullopt),
    mUK(nullptr), mPt(nullptr)
{
}


void cTopoPoint::findUK(const cBA_GCP & aBA_GCP, cPhotogrammetricProject *aPhProj)
{
#ifdef VERBOSE_TOPO
    StdOut() << "findUK "<<mName<<": ";
#endif
    MMVII_INTERNAL_ASSERT_strong(!isReady(), "double cTopoPoint::findOrMakeUK for point "+mName);

    // search among GCP
    for (unsigned int i=0; i<aBA_GCP.getMesGCP().MesGCP().size(); ++i )
    {
        if (mName ==aBA_GCP.getMesGCP().MesGCP()[i].mNamePt)
        {
            mUK = aBA_GCP.getGCP_UK().at(i);
            MMVII_INTERNAL_ASSERT_strong(mUK, "cTopoPoint::findOrMakeUK with shurred GCP not accepted for now");
            mPt = &aBA_GCP.getGCP_UK().at(i)->Pt(); //< use existing unknown if available
            mInitCoord = *mPt;
    #ifdef VERBOSE_TOPO
            StdOut() << "is a GCP\n";
    #endif
            return;
        }
    }

    // search among cameras
    if (aPhProj && aPhProj->IsOriInDirInit())
    {
        cSensorCamPC * aCam = aPhProj->ReadCamPC(mName, true, true);
        if (aCam)
        {
            mUK = aCam;
            mPt = &aCam->Center();
            mInitCoord = *mPt;
#ifdef VERBOSE_TOPO
            StdOut() << "is a camera\n";
#endif
            return;
        }
    }

    MMVII_INTERNAL_ASSERT_strong(false, "cTopoPoint::findOrMakeUK topo point \""+mName+"\" not found in GCP or Ori");
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


};
