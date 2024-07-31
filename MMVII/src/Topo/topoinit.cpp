#include "topoinit.h"

#include "ctopopoint.h"
#include "ctopoobs.h"
#include "ctopoobsset.h"


namespace MMVII
{

bool tryInit3Obs1Station(cTopoPoint & aPtToInit, tStationsMap &stationsMap) // try 3 obs from one station
{
    for (auto& [aOriginPt, aStationVect] : stationsMap)
    {
        if (!aOriginPt->isInit())
            continue;
        for (auto & aStation: aStationVect)
        {
            if (!aStation->isInit())
                continue;
            cTopoObs * obs_az = nullptr;
            cTopoObs * obs_zen = nullptr;
            cTopoObs * obs_dist = nullptr;
            cTopoObs * obs_dx = nullptr;
            cTopoObs * obs_dy = nullptr;
            cTopoObs * obs_dz = nullptr;
            for (const auto & aObs: aStation->getAllObs())
            {
                if (aObs->getPointName(1)==aPtToInit.getName())
                {
                    switch (aObs->getType()) {
                    case eTopoObsType::eHz:
                        obs_az = aObs;
                        break;
                    case eTopoObsType::eZen:
                        obs_zen = aObs;
                        break;
                    case eTopoObsType::eDist:
                        obs_dist = aObs;
                        break;
                    case eTopoObsType::eDX:
                        obs_dx = aObs;
                        break;
                    case eTopoObsType::eDY:
                        obs_dy = aObs;
                        break;
                    case eTopoObsType::eDZ:
                        obs_dz = aObs;
                        break;
                    default:
                        break;
                    }
                }
            }
            if (obs_az && obs_zen && obs_dist)
            {
#ifdef VERBOSE_TOPO
                StdOut() << "Init as angles and distance from one station on " << aOriginPt->getName() << "\n";
#endif
                cPt3dr a3DVect;
                double d0 = obs_dist->getMeasures()[0]*sin(obs_zen->getMeasures()[0]);
                a3DVect.x() = d0*sin(obs_az->getMeasures()[0]);
                a3DVect.y() = d0*cos(obs_az->getMeasures()[0]);
                a3DVect.z() = obs_dist->getMeasures()[0]*cos(obs_zen->getMeasures()[0]);
                *aPtToInit.getPt() =  *aOriginPt->getPt() +
                        (aStation->getRotSysCo2Vert() * aStation->getRotVert2Instr()).Inverse(a3DVect);
                return true;
            }
            if (obs_dx && obs_dy && obs_dz)
            {
#ifdef VERBOSE_TOPO
                StdOut() << "Init as dx dy dz from one station on " << aOriginPt->getName() << "\n";
#endif
                cPt3dr a3DVect( obs_dx->getMeasures()[0],
                                obs_dy->getMeasures()[0],
                                obs_dz->getMeasures()[0]
                        );
                *aPtToInit.getPt() =  *aOriginPt->getPt() +
                        (aStation->getRotSysCo2Vert() * aStation->getRotVert2Instr()).Inverse(a3DVect);
                return true;
            }
        }
    }
    return false;
}


bool tryInitVertStations(cTopoPoint & aPtToInit, tStationsMap &stationsMap) // try bearing and distance from verticalized stations
{
    for (auto& [aOriginPt, aStationVect] : stationsMap)
    {
        tREAL8 az=NAN, zen=NAN, dist=NAN;
        cTopoObsSetStation * aStationHz = nullptr;
        if (!aOriginPt->isInit())
            continue;
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
                    if (!std::isfinite(dist))
                        if (aObs->getType()==eTopoObsType::eDist)
                            dist = aObs->getMeasures()[0];

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
            *aPtToInit.getPt() =  *aOriginPt->getPt() +
                    (aStationHz->getRotSysCo2Vert() * aStationHz->getRotVert2Instr()).Inverse(a3DVect);
            return true;
        }
    }
    return false;
}


// try resection



}
