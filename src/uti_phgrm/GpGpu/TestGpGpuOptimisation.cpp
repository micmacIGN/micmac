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
    /*
    //printf("TestGpGpu");
    // Cr�ation du contexte GPGPU
    cudaDeviceProp deviceProp;
    // Obtention de l'identifiant de la carte la plus puissante
    int devID = gpuGetMaxGflopsDeviceId();
    // Initialisation du contexte
    checkCudaErrors(cudaSetDevice(devID));
    // Obtention des propriétés de la carte
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));
    // Affichage des propriétés de la carte
//    printf("\n");
//    printf("GPU Device %d: \"%s\" with compute capability %d.%d\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);
//    printf("Maximum Threads Per Block : %d\n", deviceProp.maxThreadsPerBlock);
    Launch();
    */

    boost::signal<void()> mySignal;

    GpGpuMultiThreadingCpu myClass;
    mySignal.connect(boost::bind(&GpGpuMultiThreadingCpu::doSomething, boost::ref(myClass)));

    char caract = 0;

    printf("-------\n");

    // launches a thread and executes myClass.loop() there
    boost::thread t(boost::bind(&GpGpuMultiThreadingCpu::loop, boost::ref(myClass)));

    t.detach();

    // calls myClass.doSomething() in this thread, but loop() executes it in the other
    for(int i = 0 ; i < 3; i++)

    mySignal();

    std::cin >> caract;

    return 0;
}

