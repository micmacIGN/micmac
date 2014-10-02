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

static std::vector<void*>   stArgs;
static cudaContext          sCCuda;
static openClContext        sCOpenCl;

#if OPENCL_ENABLED
cl_context                      openClContext::_contextOpenCL  = 0;
cl_command_queue                openClContext::_commandQueue   = 0;
cl_kernel                       openClContext::_kernel         = 0;
template <> cudaContext         CGpGpuContext<cudaContext>::_sContext          = sCCuda;
template <> openClContext       CGpGpuContext<openClContext>::_sContext        = sCOpenCl;
template <> unsigned short      vGpuContext<OPENCL_CONTEXT>::_nbArg            = 0;
template <> std::vector<void*>  vGpuContext<OPENCL_CONTEXT>::_kernelArgs       = stArgs;
#endif

template <> unsigned short      vGpuContext<CUDA_CONTEXT>::_nbArg      = 0;
template <> std::vector<void*>  vGpuContext<CUDA_CONTEXT>::_kernelArgs = stArgs;

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


#ifdef _WIN32
   #define PATHOPENCLFILE "D:\\MicMac\\src\\uti_phgrm\\GpGpu\\GpGpu_OpenCL_Kernel.cu"
#else
   #define PATHOPENCLFILE "../src/uti_phgrm/GpGpu/GpGpu_OpenCL_Kernel.cu"
#endif

template<class context>
void main_SDK()
{
    CGpGpuContext<context>::createContext();
#if OPENCL_ENABLED
    CuDeviceData2DOPENCL<int> buffer;
#endif
    CuDeviceData2D<int> bufferc;

    CuHostData3D<int> bufferHost;

    uint2 sizeBuff = make_uint2(5,1);

#if OPENCL_ENABLED
    if(CGpGpuContext<context>::typeContext() == OPENCL_CONTEXT )
        buffer.Malloc(sizeBuff);
    else if(CGpGpuContext<context>::typeContext() == CUDA_CONTEXT)
        if(bufferc.Malloc(sizeBuff))
            printf("Success buffer device malloc DUDA\n");
#else
    bufferc.Malloc(sizeBuff);
#endif

    bufferHost.Malloc(sizeBuff,1);

    int factor = 100;

    CGpGpuContext<context>::createKernel(PATHOPENCLFILE,"kMultTab");

#if OPENCL_ENABLED
    if(CGpGpuContext<context>::typeContext() == OPENCL_CONTEXT)
        CGpGpuContext<context>::addKernelArg(buffer);
    else if (CGpGpuContext<context>::typeContext() == CUDA_CONTEXT)
        CGpGpuContext<context>::addKernelArg(bufferc);
#else
    CGpGpuContext<gpgskd>::addKernelArg(bufferc);
#endif

    CGpGpuContext<context>::addKernelArg(factor);

    CGpGpuContext<context>::launchKernel();



#if OPENCL_ENABLED
    if(CGpGpuContext<context>::typeContext() == OPENCL_CONTEXT)    
    {
        buffer.CopyDevicetoHost(bufferHost.pData());        
        buffer.Dealloc();
    }
    else if (CGpGpuContext<context>::typeContext() == CUDA_CONTEXT)
    {
        bufferc.CopyDevicetoHost(bufferHost.pData());
        bufferc.Dealloc();
    }

#else
    bufferc.CopyDevicetoHost(bufferHost.pData());
#endif


    bufferHost.OutputValues();

    CGpGpuContext<context>::deleteContext();

    CGpGpuContext<context>::check_Cuda();
}

int main()
{
#if OPENCL_ENABLED
    main_SDK<openClContext>();
#endif
    main_SDK<cudaContext>();
    return 0;
}
