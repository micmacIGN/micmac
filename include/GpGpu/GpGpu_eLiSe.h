#ifndef _GPGPU_ELISE_INCLUDE_
#define _GPGPU_ELISE_INCLUDE_


#ifdef _WIN32
	#include <windows.h>
#endif

#include <string>
#include <vector>
#include <sstream>
#include <istream>
#include <ostream>
#include <iostream>
#include <list>
#include <map>
#include <cstring>
#include <sstream>
#include <limits>

using namespace std;

#include "general/opt_debug.h"
#include "general/sys_dep.h"
#include "general/util.h"
#include "general/ptxd.h"
#include "GpGpu/helper_math_extented.cuh"
#include "GpGpu/GpGpu_Defines.h"


SUPPRESS_NOT_USED_WARN static inline uint2 toUi2(Pt2di a){return make_uint2(a.x,a.y);}

SUPPRESS_NOT_USED_WARN static inline int2  toInt2(Pt2di a){return make_int2(a.x,a.y);}

SUPPRESS_NOT_USED_WARN static inline int2  toI2(Pt2dr a){return make_int2((int)a.x,(int)a.y);}

template<>
inline void dump_Type<Pt2di>(Pt2di var)
{
    printf("[%d,%d]\n",var.x,var.y);
}

template<>
inline void dump_Type<Pt2dr>(Pt2dr var)
{
    printf("[%f,%f]\n",var.x,var.y);
}

#endif // _GPGPU_ELISE_INCLUDE_
