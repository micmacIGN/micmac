#ifndef TOPOINIT_H
#define TOPOINIT_H

#include "MMVII_Topo.h"

namespace MMVII
{

/** \file topoinit.h
    \brief 3D points initialization from topo measurements
*/

bool tryInit3Obs1Station(cTopoPoint & aPtToInit, tStationsMap &stationsMap, tSimpleObsMap &allSimpleObs);
bool tryInitVertStations(cTopoPoint & aPtToInit, tStationsMap &stationsMap, tSimpleObsMap &allSimpleObs);

}
#endif // TOPOINIT_H
