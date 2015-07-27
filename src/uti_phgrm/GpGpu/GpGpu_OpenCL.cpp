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


#if OPENCL_ENABLED
static cudaContext          sCCuda;
static openClContext        sCOpenCl;
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

    CuDeviceData2D<int,context> buffer;

    CuHostData3D<int> bufferHost;

    uint2 sizeBuff = make_uint2(5,1);

    buffer.Malloc(sizeBuff);
    bufferHost.Malloc(sizeBuff,1);

    int factor = 100;

    CGpGpuContext<context>::createKernel(PATHOPENCLFILE,"kMultTab");


    CGpGpuContext<context>::addKernelArg(buffer);


    CGpGpuContext<context>::addKernelArg(factor);

    CGpGpuContext<context>::launchKernel();

    buffer.CopyDevicetoHost(bufferHost.pData());
    buffer.Dealloc();

    bufferHost.OutputValues();

    CGpGpuContext<context>::deleteContext();

    CGpGpuContext<context>::check_Cuda();
}

void UnitTest___CPP11()
{
#ifndef NOCUDA_X11
    CStructure<3> testS;
#else
    CStructure<10> testS;
#endif

    DUMP(testS.getDimension())
            DUMP(testS.getNbLayer())
            DUMP_LINE

            testS.setDimension(3);

    DUMP(testS.getDimension())
            DUMP(testS.getNbLayer())
            DUMP_LINE

            testS.setDimension(20,5);

    DUMP(testS.getDimension())
            DUMP(testS.getNbLayer())
            DUMP_LINE

        #ifdef NOCUDA_X11
            testS.setDimension(3,5,5,8,223,4);
#endif

    DUMP(testS.getDimension())
            DUMP(testS.getNbLayer())
            DUMP_LINE

            testS.setDimension(2,5,88);

    DUMP(testS.getDimension())
            DUMP(testS.getNbLayer())

            DUMP(testS.getSize())
            DUMP_LINE

}

void UnitTest___MultiContext()
{
	CuHostData3D<int2> bufferHost;

	CuDeviceData3D<float3> bufferDevice(10,"fata");

	bufferDevice.OutputInfo();
	bufferHost.OutputInfo();

	bufferHost.Malloc(make_uint2(5,5),1);

	bufferHost.Fill(make_int2(5,8));

	bufferHost.OutputValues();

#if OPENCL_ENABLED
	main_SDK<openClContext>();
#endif
	main_SDK<cudaContext>();
}


void UnitTest___SGPU()
{
	DUMP(sgpu::__mult<32>(64))

	ushort dZ = 286;

	ushort _maxDz = sgpu::__mult<WARPSIZE>(sgpu::__iDivUp<WARPSIZE>(dZ));


	DUMP(_maxDz)

			_maxDz = iDivUp32(dZ) * WARPSIZE;

	DUMP(_maxDz)

			DUMP(sgpu::__div<32>(_maxDz))
			DUMP(sgpu::__multipleSup<32>(_maxDz))
			DUMP(sgpu::__multipleSup<34>(_maxDz))
}

int main()
{

	UnitTest___SGPU();

	//UnitTest___MultiContext()

	return 0;
}
