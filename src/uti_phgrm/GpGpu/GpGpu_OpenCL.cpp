#ifdef __APPLE__
#include "OpenCL/opencl.h"
#else
#include "CL/cl.h"
#endif

#include "GpGpu/GpGpu_CommonHeader.h"
#include "GpGpu/GpGpu_Object.h"
#include "GpGpu/GpGpu_Data.h"
#include "GpGpu/GpGpu_Context.h"

#include <cstdarg>
template <>

cl_context  CGpGpuContext<OPENCLSDK>::_contextOpenCL = 0;

template <>
cl_command_queue  CGpGpuContext<OPENCLSDK>::_commandQueue = 0;

template <>
cl_kernel   CGpGpuContext<OPENCLSDK>::_kernel = 0;


void simple_printf(const char* fmt...)
{
    va_list args;
    va_start(args, fmt);

    while (*fmt != '\0') {
        if (*fmt == 'd') {
            int i = va_arg(args, int);
            std::cout << i << '\n';
        } else if (*fmt == 'c') {
            // note automatic conversion to integral type
            int c = va_arg(args, int);
            std::cout << static_cast<char>(c) << '\n';
        } else if (*fmt == 'f') {
            double d = va_arg(args, double);
            std::cout << d << '\n';
        }
        ++fmt;
    }

    va_end(args);
}




int main()
{

    CGpGpuContext<OPENCLSDK>::createContext();

    CuDeviceData2DOPENCL<int> buffer;
    CuHostData3D<int> bufferHost;

    uint2 sizeBuff = make_uint2(5,1);
    buffer.Malloc(sizeBuff);
    bufferHost.Malloc(sizeBuff,1);

    CGpGpuContext<OPENCLSDK>::createKernel("/home/gchoqueux/cuda-workspace/micmac/micmac-src/src/uti_phgrm/GpGpu/GpGpu_OpenCL_Kernel.cl");

    CGpGpuContext<OPENCLSDK>::addKernelArg(buffer);

    CGpGpuContext<OPENCLSDK>::launchKernel();

    buffer.CopyDevicetoHost(bufferHost.pData());

    bufferHost.OutputValues();

    CGpGpuContext<OPENCLSDK>::deleteContext();

    //simple_printf("dcff", 3, 'a', 1.999, 42.5);

    return 0;

}
