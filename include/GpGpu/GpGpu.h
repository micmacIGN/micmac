#ifndef __GPGPU_H__
#define __GPGPU_H__

#include "GpGpu/GpGpu_BuildOptions.h"

#if CUDA_ENABLED
#include "GpGpu/GpGpu_InterCorrel.h"
#include "GpGpu/GpGpu_InterOptimisation.h"
#include "GpGpu/GpGpu_Interface_CorMultiScale.h"
#include "GpGpu/GpGpu_eLiSe.h"
#include "GpGpu/GpGpu_Context.h"
#endif

#endif // __GPGPU_H__
