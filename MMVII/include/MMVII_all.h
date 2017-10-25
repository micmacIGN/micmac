#ifndef _MMVII_ALL_H_
#define _MMVII_ALL_H_

// Header standar c++
#include "memory.h"
#include <memory>
#include <iostream>
#include <fstream>
#include <string>
#include <typeinfo>
#include <vector>
// #include <algorithm> => put it in file requiring it as it seem to slow down compilation


namespace MMVII
{
// Header MMVII
// #include "TypeNum.h"
#include "MMVII_Sys.h"
#include "MMVII_Error.h"
#include "MMVII_memory.h"
#include "MMVII_util.h"

// Les class cPtxd, cPt1d, cPt2d
#include "MMVII_Ptxd.h"
#include "MMVII_Bench.h"
// 
#include "cMMVII_Appli.h"
};

using namespace MMVII;


#endif //  _MMVII_ALL_H_
