#ifndef _GPGPU_ELISE_INCLUDE_
#define _GPGPU_ELISE_INCLUDE_

#include "general/opt_debug.h"
#include "general/sys_dep.h"
#include "general/util.h"
#include "general/ptxd.h"
#include "GpGpu/helper_math_extented.cuh"

SUPPRESS_NOT_USED_WARN static inline uint2 toUi2(Pt2di a){return make_uint2(a.x,a.y);}

SUPPRESS_NOT_USED_WARN static inline int2  toInt2(Pt2di a){return make_int2(a.x,a.y);}

SUPPRESS_NOT_USED_WARN static inline int2  toI2(Pt2dr a){return make_int2((int)a.x,(int)a.y);}

#endif // _GPGPU_ELISE_INCLUDE_
