#include "GpGpu/GpGpu_CommonHeader.h"
#include "GpGpu/GpGpu_Data.h"
#include "GpGpu/helper_math_extented.cuh"

#define __GPU_CONSTANT  __constant__
#define __GPU_GLOBAL
#define __GPU_KERNEL    __global__
#define __GPU_THREADX   threadIdx.x
