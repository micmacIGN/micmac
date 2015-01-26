#ifndef GPGPU_COMMON_HEADER_H
#define GPGPU_COMMON_HEADER_H

/** @defgroup GpGpuDoc Documentation GpGpu
 *
 * Ce module parle de la GpGpu
 *
 */

#include <sstream>
#include <string>
#include <iostream>
#include <limits>

using namespace std;

#include "GpGpu/GpGpu_BuildOptions.h"

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <helper_cuda.h>
#include <helper_math.h>
#include <helper_functions.h>
#include "helper_math_extented.cuh"

#include <typeinfo>

#ifdef _WIN32
    #include <Lmcons.h>
#else
	#ifdef NVTOOLS
		#include "nvToolsExt.h"
	#endif
    #include <stdio.h>
    #include <unistd.h>
    #include <sys/types.h>
    #include <pwd.h>
    #include <cmath>
#endif

#include "GpGpu/GpGpu_Defines.h"


#endif  //GPGPU_COMMON_HEADER_H
