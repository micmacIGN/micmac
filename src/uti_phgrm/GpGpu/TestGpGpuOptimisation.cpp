#include <iostream>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_math.h>
#include <helper_cuda.h>
#include "GpGpu/GpGpuOptimisation.h"

using namespace std;

int main()
{
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

    return 0;
}

