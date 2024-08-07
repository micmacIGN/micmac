#ifndef TOPOINIT_H
#define TOPOINIT_H

#include "Topo.h"

namespace MMVII
{

bool tryInit3Obs1Station(cTopoPoint & aPtToInit, tStationsMap &stationsMap);
bool tryInitVertStations(cTopoPoint & aPtToInit, tStationsMap &stationsMap);

}
#endif // TOPOINIT_H
