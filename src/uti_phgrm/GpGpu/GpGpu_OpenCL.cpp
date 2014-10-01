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

static std::vector<void*> stArgs;

#if OPENCL_ENABLED
template <> cl_context          CGpGpuContext<OPENCLSDK>::_contextOpenCL    = 0;
template <> cl_command_queue    CGpGpuContext<OPENCLSDK>::_commandQueue     = 0;
template <> cl_kernel           CGpGpuContext<OPENCLSDK>::_kernel           = 0;
template <> cl_context          CGpGpuContext<CUDASDK>::_contextOpenCL    = 0;
template <> cl_command_queue    CGpGpuContext<CUDASDK>::_commandQueue     = 0;
template <> cl_kernel           CGpGpuContext<CUDASDK>::_kernel           = 0;
#endif

template <> unsigned short      CGpGpuContext<CUDASDK>::_nbArg            = 0;
template <> std::vector<void*>  CGpGpuContext<CUDASDK>::_kernelArgs = stArgs;
template <> unsigned short      CGpGpuContext<OPENCLSDK>::_nbArg            = 0;
template <> std::vector<void*>  CGpGpuContext<OPENCLSDK>::_kernelArgs = stArgs;

extern void kMultTab();

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

template<int gpgskd>
void main_SDK()
{
    CGpGpuContext<gpgskd>::createContext();
#if OPENCL_ENABLED
    CuDeviceData2DOPENCL<int> buffer;
#endif
    CuDeviceData2D<int> bufferc;

    CuHostData3D<int> bufferHost;

    uint2 sizeBuff = make_uint2(5,1);

#if OPENCL_ENABLED
    if(gpgskd == OPENCLSDK)
        buffer.Malloc(sizeBuff);
    else if(gpgskd == CUDASDK)
        if(bufferc.Malloc(sizeBuff))
            printf("Success buffer device malloc DUDA\n");
#else
    bufferc.Malloc(sizeBuff);
#endif

    bufferHost.Malloc(sizeBuff,1);

    int factor = 100;
#ifdef _WIN32
    CGpGpuContext<gpgskd>::createKernel("D:\\MicMac\\src\\uti_phgrm\\GpGpu\\GpGpu_OpenCL_Kernel.cu","kMultTab");
#else
    CGpGpuContext<gpgskd>::createKernel("../src/uti_phgrm/GpGpu/GpGpu_OpenCL_Kernel.cu","kMultTab");
#endif

#if OPENCL_ENABLED
    if(gpgskd == OPENCLSDK)        
        CGpGpuContext<gpgskd>::addKernelArg(buffer);
    else if (gpgskd == CUDASDK)
        CGpGpuContext<gpgskd>::addKernelArg(bufferc);
#else
    CGpGpuContext<gpgskd>::addKernelArg(bufferc);
#endif

    CGpGpuContext<gpgskd>::addKernelArg(factor);

    CGpGpuContext<gpgskd>::launchKernel();



#if OPENCL_ENABLED
    if(gpgskd == OPENCLSDK)
        buffer.CopyDevicetoHost(bufferHost.pData());
    else if (gpgskd == CUDASDK)
        bufferc.CopyDevicetoHost(bufferHost.pData());
#else
    bufferc.CopyDevicetoHost(bufferHost.pData());
#endif


    bufferHost.OutputValues();

    CGpGpuContext<gpgskd>::deleteContext();
}

int main()
{
#if OPENCL_ENABLED
    main_SDK<OPENCLSDK>();
#endif
    main_SDK<CUDASDK>();
    return 0;
}
