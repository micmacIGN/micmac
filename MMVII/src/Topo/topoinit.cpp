#include "MMVII_Topo.h"


namespace MMVII
{

//< try to initialize a point seen by 3 obs from one station
bool cBA_Topo::tryInit3Obs1Station(cTopoPoint & aPtToInit)
{
    for (auto& [aOriginPt, aStationVect] : mAllStations)
    {
        if (!aOriginPt->isInit())
            continue;
        for (auto & aStation: aStationVect)
        {
            if (!aStation->isInit())
                continue;
            auto a3DVect = aStation->obs2vector(aPtToInit.getName());
            if (a3DVect.IsValid())
            {
#ifdef VERBOSE_TOPO
                StdOut() << "Init from one station on " << aOriginPt->getName() << "\n";
#endif
                *aPtToInit.getPt() = aStation->PtInstr2SysCo(a3DVect);
                return true;
            }
        }
    }
    return false;
}

//< try to initialize a point seen from bearing and distance from verticalized stations
bool cBA_Topo::tryInitVertStations(cTopoPoint & aPtToInit)
{
    for (auto& [aOriginPt, aStationVect] : mAllStations)
    {
        tREAL8 az=NAN, zen=NAN, dist=NAN;
        cTopoObsSetStation * aStationHz = nullptr;
        if (!aOriginPt->isInit())
            continue;
        if (mAllSimpleObs.count(aOriginPt))
        {
            for (auto & aObs: mAllSimpleObs[aOriginPt])
            {
                if ( aObs->getPointName(0)==aPtToInit.getName()
                     || aObs->getPointName(1)==aPtToInit.getName() )
                {
                    switch (aObs->getType()) {
                    case eTopoObsType::eDist:
                        dist = aObs->getMeasures()[0];
                        break;
                    default:
                        break;
                    }
                }
            }
        }
        for (auto & aStation: aStationVect)
        {
            for (const auto & aObs: aStation->getAllObs())
            {
                if (aObs->getPointName(1)==aPtToInit.getName())
                {
                    // special case for verticalized stations: each obs can be in different stations
                    if (!std::isfinite(az))
                        if ((aObs->getType()==eTopoObsType::eHz) && (aStation->isInit())
                                && ((aStation->getOriStatus()==eTopoStOriStat::eTopoStOriFixed)
                                    || (aStation->getOriStatus()==eTopoStOriStat::eTopoStOriVert)))
                        {
                            az = aObs->getMeasures()[0];
                            aStationHz = aStation;
                        }
                    if (!std::isfinite(zen))
                        if ((aObs->getType()==eTopoObsType::eZen) && (
                                (aStation->getOriStatus()==eTopoStOriStat::eTopoStOriFixed)
                                || (aStation->getOriStatus()==eTopoStOriStat::eTopoStOriVert)) )
                            zen = aObs->getMeasures()[0];
                }
            }
        }
        if (aStationHz && std::isfinite(az) && std::isfinite(zen) && std::isfinite(dist))
        {
#ifdef VERBOSE_TOPO
            StdOut() << "Init as bearing and distance from " << aOriginPt->getName() << "\n";
#endif
            cPt3dr a3DVect;
            double d0 = dist*sin(zen);
            a3DVect.x() = d0*sin(az);
            a3DVect.y() = d0*cos(az);
            a3DVect.z() = dist*cos(zen);
            *aPtToInit.getPt() = aStationHz->PtInstr2SysCo(a3DVect);
            return true;
        }
    }
    return false;
}


// try resection



}
