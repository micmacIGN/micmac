#include <iostream>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_math.h>
#include <helper_cuda.h>
#include "GpGpu/GpGpuOptimisation.h"
#include "GpGpu/GpGpuMultiThreadingCpu.h"

using namespace std;

int main()
{
    GpGpuMultiThreadingCpu< CuHostData3D<uint>, CuDeviceData3D<uint> >  JobCpuGpGpu;
    JobCpuGpGpu.createThread();

//    cout << "produced " << QQ.producer_count << " objects." << endl;
//    cout << "consumed " << QQ.consumerProducer_count << " objects." << endl;

    return 0;
}

