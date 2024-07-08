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

cTopoPoint::cTopoPoint(const std::string & name, const cPt3dr &aInitCoord, bool aIsFree, const cPt3dr &aSigmas) :
    mName(name), mInitCoord(aInitCoord),
    mVertDefl(std::nullopt), mUK(nullptr), mPt(nullptr)
{
}


cTopoPoint::cTopoPoint() :
    mName(""), mInitCoord(cPt3dr::Dummy()), mVertDefl(std::nullopt),
    mUK(nullptr), mPt(nullptr)
{
}


void cTopoPoint::findUK(const std::vector<cBA_GCP *> & vGCP, cPhotogrammetricProject *aPhProj, const cPt3dr & aCoordIfPureTopo)
{
#ifdef VERBOSE_TOPO
    StdOut() << "findUK "<<mName<<": ";
#endif
    MMVII_INTERNAL_ASSERT_strong(!isReady(), "double cTopoPoint::findOrMakeUK for point "+mName);

    // search among GCP
    for (auto & gcp : vGCP)
    {
        for (unsigned int i=0; i<gcp->mMesGCP->MesGCP().size(); ++i )
        {
            if (mName ==gcp->mMesGCP->MesGCP()[i].mNamePt)
            {
                if (gcp->mGCP_UK.size() == gcp->mMesGCP->MesGCP().size())
                {
                    mUK = gcp->mGCP_UK.at(i);
                    mPt = &gcp->mGCP_UK.at(i)->Pt(); //< use existing unknown if available
                    mInitCoord = *mPt;
    #ifdef VERBOSE_TOPO
                    StdOut() << "is a GCP with existing unknowns: "<<*mPt<<" "<<mUK<<"\n";
    #endif
                    return;
                } else {
                    MMVII_INTERNAL_ASSERT_strong(false, "cTopoPoint::findOrMakeUK with shurred GCP not accepted for now");
                    return;
                }
            }
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
