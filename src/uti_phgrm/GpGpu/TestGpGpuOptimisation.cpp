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
    GpGpuMultiThreadingCpu QQ;

    srand (time(NULL));
     cout << "boost::lockfree::queue 1 is ";

     if (!QQ.spsc_queue_1.is_lock_free())
         cout << "not ";

     cout << "lockfree" << endl;

     QQ.createThread();

     cout << "produced " << QQ.producer_count << " objects." << endl;
     cout << "consumed " << QQ.consumerProducer_count << " objects." << endl;

    return 0;
}

