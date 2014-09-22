#ifndef GPGPU_CONTEXT_H
#define GPGPU_CONTEXT_H

#include "StdAfx.h"

#ifdef __APPLE__
#include "OpenCL/opencl.h"
#else
#include "CL/cl.h"
#endif

enum GPGPUSDK {  CUDASDK
                ,OPENCLSDK
              };

template<int GPUSDK>
class CGpGpuContext
{
public:

    CGpGpuContext(){}

    void createContext(){}

    void deleteContext(){}

private:

    cl_context _contextOpenCL;
};

template <> inline
void CGpGpuContext<CUDASDK>::createContext() {

    //srand ((uint)time(NULL));
    // Creation du contexte GPGPU
    cudaDeviceProp deviceProp;
    // Obtention de l'identifiant de la carte la plus puissante
    int devID = gpuGetMaxGflopsDeviceId();

    ELISE_ASSERT(devID == 0 , "NO GRAPHIC CARD FOR USE CUDA");

    // Initialisation du contexte
    checkCudaErrors(cudaSetDevice(devID));
    // Obtention des proprietes de la carte
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));
    // Affichage des proprietes de la carte
    //printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);
}

template <> inline
void CGpGpuContext<OPENCLSDK>::createContext() {

    cl_uint platformIdCount = 0;
    clGetPlatformIDs (0, NULL, &platformIdCount);

    std::vector<cl_platform_id> platformIds (platformIdCount);
    clGetPlatformIDs (platformIdCount, platformIds.data (), NULL);

    cl_uint deviceIdCount = 0;
    clGetDeviceIDs (platformIds [0], CL_DEVICE_TYPE_ALL, 0, NULL,
            &deviceIdCount);

    std::vector<cl_device_id> deviceIds (deviceIdCount);
    clGetDeviceIDs (platformIds [0], CL_DEVICE_TYPE_ALL, deviceIdCount,
            deviceIds.data (), NULL);

    const cl_context_properties contextProperties [] =
    {
        CL_CONTEXT_PLATFORM,
        reinterpret_cast<cl_context_properties> (platformIds [0]),
        0, 0
    };

    cl_int error;

    _contextOpenCL = clCreateContext (
                contextProperties, deviceIdCount,
                deviceIds.data (), NULL,
                NULL, &error);

    if(error == CL_SUCCESS)
    {
        printf("CONTEXT OPENCL OK\n");
        printf("platform Id Count %d\n",platformIdCount);
        printf("device Id Count %d\n",deviceIdCount);
    }
}

template <> inline
void CGpGpuContext<CUDASDK>::deleteContext()
{

    checkCudaErrors( cudaDeviceReset() );
}

template <> inline
void CGpGpuContext<OPENCLSDK>::deleteContext()
{
    clReleaseContext(_contextOpenCL);
}

#endif // GPGPU_CONTEXT_H



