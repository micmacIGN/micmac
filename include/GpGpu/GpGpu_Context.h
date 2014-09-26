#ifndef GPGPU_CONTEXT_H
#define GPGPU_CONTEXT_H

//#include "StdAfx.h"
#include <list>
#include <map>

#include "GpGpu_CommonHeader.h"
#include "GpGpu_eLiSe.h"
#include "GpGpu_BuildOptions.h"


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

    static void createContext(){}

    static void deleteContext(){}

    static cl_context contextOpenCL(){return NULL;}

    static cl_command_queue commandQueue(){return NULL;}

    static  void OutputInfoGpuMemory(){}

    static  void check_Cuda(){}

private:

    static cl_context           _contextOpenCL;
    static cl_command_queue     _commandQueue;

};

template <> inline
void CGpGpuContext<CUDASDK>::OutputInfoGpuMemory()
{
    size_t free;
    size_t total;
    checkCudaErrors( cudaMemGetInfo(&free, &total));
    cout << "Memoire video       : " << (float)free / pow(2.0f,20) << " / " << (float)total / pow(2.0f,20) << "Mo" << endl;
}

template <> inline
void CGpGpuContext<CUDASDK>::check_Cuda()
{
    cout << "CUDA build enabled\n";

//    int apiVersion = 0;

//    cudaRuntimeGetVersion(&apiVersion);

            //DUMP_INT(apiVersion)

//	switch (__CUDA_API_VERSION)
//	{
//	case 0x3000:
//		cout << "3.0";
//		break;
//	case 0x3020:
//		cout << "3.2";
//		break;
//	case 0x4000:
//		cout << "4.0";
//		break;
//	case 0x5000:
//		cout << "5.0";
//		break;
//	case 0x5050:
//		cout << "5.5";
//		break;
//	case 0x6000:
//		cout << "6.0";
//		break;
//	}
//	cout << endl;

    int device_count = 0;

    checkCudaErrors(cudaGetDeviceCount(&device_count));

    if(device_count == 0)
        printf("NO NVIDIA GRAPHIC CARD FOR USE CUDA");
    else
    {

        // Creation du contexte GPGPU
        cudaDeviceProp deviceProp;
        // Obtention de l'identifiant de la carte la plus puissante
        int devID = gpuGetMaxGflopsDeviceId();

        // Initialisation du contexte
        checkCudaErrors(cudaSetDevice(devID));
        // Obtention des proprietes de la carte
        checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));
        // Affichage des proprietes de la carte
        printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);

    }
}


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
void CGpGpuContext<CUDASDK>::deleteContext()
{

    checkCudaErrors( cudaDeviceReset() );
}

#if OPENCL_ENABLED
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

    _commandQueue = clCreateCommandQueue(_contextOpenCL,deviceIds[0], 0, &error);

    if(error ==CL_SUCCESS)
        printf("Success create cqueue!!\n");
    else
        printf("Error create cqueue %d\n",error);

}

template <> inline
void CGpGpuContext<OPENCLSDK>::deleteContext()
{
    if(clReleaseContext(_contextOpenCL) == CL_SUCCESS)
        printf("CONTEXT OPENCL RELEASE\n");
}

template <> inline
cl_context CGpGpuContext<OPENCLSDK>::contextOpenCL()
{
    return _contextOpenCL;
}

template <> inline
cl_command_queue CGpGpuContext<OPENCLSDK>::commandQueue()
{
return _commandQueue;
}
#endif

#endif // GPGPU_CONTEXT_H








