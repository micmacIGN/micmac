#ifndef GPGPU_CONTEXT_H
#define GPGPU_CONTEXT_H

/** @addtogroup GpGpuDoc */
/*@{*/

//#include "StdAfx.h"
#include <list>
#include <map>
#include <iostream>
#include <string>
#include <vector>

#include "GpGpu_CommonHeader.h"
#include "GpGpu_eLiSe.h"
#include "GpGpu_BuildOptions.h"
#include "GpGpu_Data.h"

template<class T> class CData;

#ifdef OPENCL_ENABLED
#ifdef __APPLE__
#include "OpenCL/opencl.h"
#else
#include "CL/cl.h"
#endif
#endif

extern void kMultTab();

enum GPGPUCONTEXT
{
    CUDA_CONTEXT,
    OPENCL_CONTEXT
};

template<int typecontext>
class vGpuContext
{
public:
    int typeContext()
    {
        return typecontext;
    }
    static unsigned short       _nbArg;
    static std::vector<void*>   _kernelArgs;
};

class cudaContext : public vGpuContext<CUDA_CONTEXT>
{

};

#ifdef OPENCL_ENABLED
class openClContext : public vGpuContext<OPENCL_CONTEXT>
{
public:

    static cl_context           _contextOpenCL;
    static cl_command_queue     _commandQueue;
    static cl_kernel            _kernel;
};
#endif

/// \class CGpGpuContext
/// \brief le context GpGpu, OpenCl ou Cuda
template<class context>
class CGpGpuContext
{
public:

    CGpGpuContext(){}

    /// \brief créer le context en fonction de < class context > du template
    ///  si context = cudaContext alors un context cuda est créé
    ///  si context = openClContext alors un context openCl est créé
    static void createContext();

    /// \brief deleteContext Détruit le context créé
    static void deleteContext(){}

    /// \brief Sortie console de la memoire du device
    static  void OutputInfoGpuMemory(){}

    /// \brief Sortie console de la version Cuda utilisé dans cette compilation du logiciel
    static  void check_Cuda(){}

    /// \brief createKernel Créé le kernel OpenCL
    /// \param file Nom du fichier contenant le kernel OpenCL
    /// \param kernelName Non du kernel attaché
    static  void createKernel(string file,string kernelName){}

    ///
    /// \brief Lance le kernel OpenCL crér
    ///
    static  void launchKernel(){}


    template< class T, template< class O, class G > class U >
    ///
    /// \brief Ajoute un buffer pour le kernel OpenCL
    /// \param arg Buffer du kernel OpenCL
    ///
    static  void addKernelArg( U<T,context> &arg);

    template<class T>
    ///
    /// \brief Ajoute un argument pour le kernel OpenCL
    /// \param arg Argument du kernel OpenCL
    ///
    static  void addKernelArg(T &arg){}

    ///
    /// \brief Sortie console de l'erreur GpGpu
    /// \param tErr Type d'erreur
    /// \param erName
    /// \return
    ///
    static bool errorDump(int tErr, string erName);

#ifdef OPENCL_ENABLED

    ///
    /// \brief
    /// \return le context OpenCL
    ///
    static cl_context contextOpenCL(){return NULL;}

    ///
    /// \brief commandQueue
    /// \return le queue de commande OpenCL
    ///
    static cl_command_queue commandQueue(){return NULL;}

    ///
    /// \brief kernel
    /// \return le kernel courant
    ///
    static cl_kernel kernel(){return NULL;}

#endif

    static  void* arg(int id)
    {

        return _sContext._kernelArgs[id];
    }

    static int typeContext()
    {
        return _sContext.typeContext();
    }

private:

    template<class T>
    static  void addKernelArgSDK( CData<T> &arg){}

    static  context             _sContext;

    static  bool                errorGpGpu(int tErr,string erName, string gpuCon);
};


template<class context>
bool CGpGpuContext<context>::errorGpGpu(int tErr,string erName, string gpuCon)
{
    if(tErr != 0)
    {
        printf("Error %s %s \t: %d\n",erName.c_str(),gpuCon.c_str(),tErr);
        return false;
    }

    else
    {
#ifdef DEBUG_GPGPU
        printf("Success %s \t: %s\n",gpuCon.c_str(),erName.c_str());
#endif
        return true;
    }

}

template<class context>
bool CGpGpuContext<context>::errorDump(int tErr,string erName)
{
    return errorGpGpu(tErr,erName,"GpGpu");
}

template<> inline
bool CGpGpuContext<cudaContext>::errorDump(int tErr,string erName)
{
    string sErrorCuda(!tErr ? "" : _cudaGetErrorEnum((cudaError_t)tErr));
    return errorGpGpu(tErr,sErrorCuda + " " + erName,"Cuda");
}

#ifdef OPENCL_ENABLED
template<> inline
bool CGpGpuContext<openClContext>::errorDump(int tErr,string erName)
{
    return errorGpGpu(tErr,erName,"openCl");
}
#endif

template<class context>
void CGpGpuContext<context>::createContext(){}

template <> inline
void CGpGpuContext<cudaContext>::OutputInfoGpuMemory()
{
    size_t free;
    size_t total;
    checkCudaErrors( cudaMemGetInfo(&free, &total));
    cout << "Memoire video       : " << (float)free / pow(2.0f,20) << " / " << (float)total / pow(2.0f,20) << "Mo" << endl;
}

template <> inline
void CGpGpuContext<cudaContext>::check_Cuda()
{
    cout << "Cuda runtime version ";

    int apiVersion = 0;

    cudaRuntimeGetVersion(&apiVersion);

    int majVersion_CUDA = apiVersion/1000;
    int minVersion_CUDA = (apiVersion - majVersion_CUDA*1000)/10;

    printf("%d.%d\n",majVersion_CUDA,minVersion_CUDA);

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
void CGpGpuContext<cudaContext>::createContext() {

    //srand ((uint)time(NULL));
    // Creation du contexte GPGPU
    cudaDeviceProp deviceProp;
    // Obtention de l'identifiant de la carte la plus puissante
    int devID = gpuGetMaxGflopsDeviceId();

    //ELISE_ASSERT(devID == 0 , "NO GRAPHIC CARD FOR USE CUDA");
    if(devID != 0 )
        printf("NO GRAPHIC CARD FOR USE CUDA\n");

    // Initialisation du contexte
    checkCudaErrors(cudaSetDevice(devID));
    // Obtention des proprietes de la carte
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));
    // Affichage des proprietes de la carte
    //printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);
}


template <> inline
void CGpGpuContext<cudaContext>::deleteContext()
{
    checkCudaErrors( cudaDeviceReset() );
}

#if OPENCL_ENABLED
template <> inline
void CGpGpuContext<openClContext>::createContext() {

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

    openClContext::_contextOpenCL = clCreateContext (
                contextProperties, deviceIdCount,
                deviceIds.data (), NULL,
                NULL, &error);

    if(error == CL_SUCCESS)
    {
        printf("CONTEXT OPENCL OK\n");
        printf("platform Id Count %d\n",platformIdCount);
        printf("device Id Count %d\n",deviceIdCount);
    }

    openClContext::_commandQueue = clCreateCommandQueue(openClContext::_contextOpenCL,deviceIds[0], 0, &error);

    errorDump(error,"CommandQueue");

}

template <> inline
void CGpGpuContext<openClContext>::deleteContext()
{
    if(clReleaseContext(openClContext::_contextOpenCL) == CL_SUCCESS)
        printf("CONTEXT OPENCL RELEASE\n");
}

template <> inline
cl_context CGpGpuContext<openClContext>::contextOpenCL()
{
    return openClContext::_contextOpenCL;
}

template <> inline
cl_command_queue CGpGpuContext<openClContext>::commandQueue()
{
    return openClContext::_commandQueue;
}

template <> inline
cl_kernel CGpGpuContext<openClContext>::kernel()
{
    return openClContext::_kernel;
}

template <> inline
void CGpGpuContext<openClContext>::createKernel(string fileName,string kernelName)
{
    cl_int error = -1;

    char buffer[1024];
    char* path_end;

#ifndef _WIN32
    if (readlink ("/proc/self/exe", buffer, sizeof(buffer)) <= 0)
        return ;
#endif

    path_end = strrchr (buffer, '/');
    if (path_end == NULL)
        return ;

    ++path_end;

    *path_end = '\0';

    std::string binPath(buffer);

    fileName = binPath + fileName;

    std::ifstream   file(fileName.c_str());
    std::string     prog(std::istreambuf_iterator<char>(file),(std::istreambuf_iterator<char>()));
    std::string     fileNameDefine = binPath + "../src/uti_phgrm/GpGpu/GpGpu_OpenCL_Kernel_Define.cl";

    std::ifstream   fileDefine(fileNameDefine .c_str());
    std::string     progDefine(std::istreambuf_iterator<char>(fileDefine),(std::istreambuf_iterator<char>()));

    prog = progDefine + "//"  + prog;

    //    if(file.is_open())
    //        printf("%s\n",prog.c_str());

    const char* sourceProg = prog.c_str();
    size_t sourceSize[] = {strlen(prog.c_str())};
    cl_program program = clCreateProgramWithSource(contextOpenCL(),1,&sourceProg,sourceSize,&error);

    errorDump(error,"Program");

    errorDump(clBuildProgram(program,0,NULL,NULL,NULL,NULL),"Build");

    openClContext::_kernel = clCreateKernel(program,kernelName.c_str(),&error);

    errorDump(error,"Kernel");

}

template <> inline
void CGpGpuContext<openClContext>::launchKernel()
{

    cl_int error = -1;

    size_t global_item_size = 5;
    size_t local_item_size  = 1;

    error  = clEnqueueNDRangeKernel(openClContext::_commandQueue,openClContext::_kernel,1,NULL,&global_item_size,&local_item_size,0,NULL,NULL);
    errorDump(error,"Enqueue");

}

template <>
template <class T> inline
void CGpGpuContext<openClContext>::addKernelArg(T &arg)
{

    cl_int error = -1;

    error = clSetKernelArg(CGpGpuContext<openClContext>::kernel(),(cl_uint)_sContext._nbArg,sizeof(T),&arg);

    errorDump(error,"Kernel Arg");

    _sContext._nbArg++;

}

template<>
template <class T> inline
void CGpGpuContext<openClContext>::addKernelArgSDK( CData<T> &arg)
{
    cl_int error = -1;

    cl_mem memBuffer = arg.clMem();

    error = clSetKernelArg(CGpGpuContext<openClContext>::kernel(),(cl_uint)_sContext._nbArg,sizeof(memBuffer),&memBuffer);

    errorDump(error,"Kernel Arg buffer");

}

#endif

template<>
template <class T> inline
void CGpGpuContext<cudaContext>::addKernelArgSDK( CData<T> &arg)
{
    printf("Add cuda kernel argument buffer : %d\n",_sContext._nbArg);

    _sContext._kernelArgs.push_back((void*)&arg);
}

template <> inline
void CGpGpuContext<cudaContext>::launchKernel()
{
    kMultTab();
}
template <>
template <class T> inline
void CGpGpuContext<cudaContext>::addKernelArg(T &arg)
{

    printf("Add cuda kernel argument : %d\n",_sContext._nbArg);

    T* p = &arg;

    _sContext._kernelArgs.push_back((void*)p);

    _sContext._nbArg++;

}

template <class context>
template <class T , template< class O, class G > class U>
void CGpGpuContext<context>::addKernelArg(U <T,context > &arg)
{
    addKernelArgSDK(arg);

    _sContext._nbArg++;
}

/*@}*/

#endif // GPGPU_CONTEXT_H








