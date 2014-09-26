#ifdef __APPLE__
#include "OpenCL/opencl.h"
#else
#include "CL/cl.h"
#endif


#include "GpGpu/GpGpu_CommonHeader.h"
#include "GpGpu/GpGpu_Object.h"
#include "GpGpu/GpGpu_Data.h"
#include "GpGpu/GpGpu_Context.h"

template <>
cl_context  CGpGpuContext<OPENCLSDK>::_contextOpenCL = 0;

template <>
cl_command_queue  CGpGpuContext<OPENCLSDK>::_commandQueue = 0;

void errorOpencl(cl_int error,string erName)
{
    if(error ==CL_SUCCESS)
        printf("Success create %s\n",erName.c_str());
    else
        printf("Error create %s = %d\n",erName.c_str(),error);
}

int main()
{

    CGpGpuContext<OPENCLSDK>::createContext();

    CuDeviceData2DOPENCL<int> buffer;
    CuHostData3D<int> bufferHost;

    uint2 sizeBuff = make_uint2(5,1);
    buffer.Malloc(sizeBuff);
    bufferHost.Malloc(sizeBuff,1);

    std::ifstream file("/home/gchoqueux/cuda-workspace/micmac/micmac-src/src/uti_phgrm/GpGpu/GpGpu_OpenCL_Kernel.cl");

    std::string prog(std::istreambuf_iterator<char>(file),(std::istreambuf_iterator<char>()));

    if(file.is_open())
        printf("%s\n",prog.c_str());

    cl_int error = -1;

    const char* sourceProg = prog.c_str();
    size_t sourceSize[] = {strlen(prog.c_str())};
    cl_program program = clCreateProgramWithSource(CGpGpuContext<OPENCLSDK>::contextOpenCL(),1,&sourceProg,sourceSize,&error);

    errorOpencl(error,"Program");

    errorOpencl(clBuildProgram(program,0,NULL,NULL,NULL,NULL),"Build");

    cl_kernel kernel = clCreateKernel(program,"hello",&error);

    errorOpencl(error,"Kernel");

    cl_mem memBuffer = buffer.clMem();

    error = clSetKernelArg(kernel,0,sizeof(memBuffer),&memBuffer);

    errorOpencl(error,"Kernel Arg");

    size_t global_item_size = 5;
    size_t local_item_size  = 1;

    error  = clEnqueueNDRangeKernel(CGpGpuContext<OPENCLSDK>::commandQueue(),kernel,1,NULL,&global_item_size,&local_item_size,0,NULL,NULL);
    errorOpencl(error,"Enqueue");


    buffer.CopyDevicetoHost(bufferHost.pData());

    bufferHost.OutputValues();

    CGpGpuContext<OPENCLSDK>::deleteContext();

    return 0;

}
