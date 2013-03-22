#ifndef __GPGPU_H__
#define __GPGPU_H__

#if defined __GNUC__
    #pragma GCC system_header
#elif defined __SUNPRO_CC
    #pragma disable_warn
#elif defined _MSC_VER
    #pragma warning(push, 1)
#endif

#include "GpGpu/GpGpuDefines.h"

#ifdef CUDA_ENABLED
#include "GpGpu/InterfaceMicMacGpGpu.h"
#endif

#endif // __GPGPU_H__
