#ifdef __APPLE__
#include "OpenCL/opencl.h"
#else
#include "CL/cl.h"
#endif

#include <vector>
#include <stdio.h>

int main()
{

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

    /*cl_context context =*/ clCreateContext (
    contextProperties, deviceIdCount,
    deviceIds.data (), NULL,
    NULL, &error);

    if(error == CL_SUCCESS)
    {
        printf("CONTEXT OPENCL OK\n");
        printf("platform Id Count %d\n",platformIdCount);
        printf("device Id Count %d\n",deviceIdCount);
    }

    return 0;

}
