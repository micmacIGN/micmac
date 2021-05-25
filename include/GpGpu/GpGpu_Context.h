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


///
/// \brief kMultTab
/// Fonction Cuda test gpgpu générique
extern void kMultTab();

///
/// \brief The GPGPUCONTEXT enum
///	Enumération des types de contextes GpGpu
/// CUDA_CONTEXT Nvidia Cuda contexte
/// OpenCL_CONTEXT OpenCL contexte
enum GPGPUCONTEXT
{
    CUDA_CONTEXT,
    OPENCL_CONTEXT
};

///
/// \brief The vGpuContext class
/// Classe générique d'un contexte gpgpu
/// Chaque nouvelle définition de contexte doit héritée de cette classe
template<int typecontext>
class vGpuContext
{
public:
	///
	/// \brief typeContext
	/// \return Le type du contexte
	///
    int typeContext()
    {
        return typecontext;
    }
	///
	/// \brief _nbArg
	/// Nombre d'arguments du kernel
    static unsigned short       _nbArg;

	///
	/// \brief _kernelArgs
	/// Vecteur des arguments du kernel
    static std::vector<void*>   _kernelArgs;
};


///
/// \brief The cudaContext class
/// Définition du contexte Cuda
class cudaContext : public vGpuContext<CUDA_CONTEXT>
{

};

#ifdef OPENCL_ENABLED
///
/// \brief The openClContext class
/// Définition du contexte OpenCL
class openClContext : public vGpuContext<OPENCL_CONTEXT>
{
public:

	///
	/// \brief _contextOpenCL
	/// le vrai contexte OpenCL
    static cl_context           _contextOpenCL;

	///
	/// \brief _commandQueue
	///
    static cl_command_queue     _commandQueue;

	///
	/// \brief _kernel
	/// fonction kernel
    static cl_kernel            _kernel;
};
#endif

/// \class CGpGpuContext
/// \brief le context GpGpu, OpenCl ou Cuda
/// Permet de gérer génériquement le contexte
template<class context>
class CGpGpuContext
{
public:

    CGpGpuContext(){}



	/// \brief createContext créer le context en fonction de < class context > du template
	///  si context = cudaContext alors un context cuda est créé
	///  si context = openClContext alors un context openCl est créé
	static void createContext();

    /// \brief deleteContext Détruit le context créé
    static void deleteContext(){}

	///
	/// \brief OutputInfoGpuMemory Sortie console de la memoire du device
	///
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

	///
	/// \brief arg obtenir le argument
	/// \param id identifiant de l'argument
	/// \return pointeur de l'argument
	///
    static  void* arg(int id)
    {

        return _sContext._kernelArgs[id];
    }

	///
	/// \brief typeContext
	/// \return le type du contexte
	///
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
///
/// \brief CGpGpuContext<cudaContext>::errorDump retourne dans la console le type d'erreur CUDA
/// \param tErr
/// \param erName
/// \return true s'il n'y a pas d'erreur
///
bool CGpGpuContext<cudaContext>::errorDump(int tErr,string erName)
{
    string sErrorCuda(!tErr ? "" : _cudaGetErrorEnum((cudaError_t)tErr));
    return errorGpGpu(tErr,sErrorCuda + " " + erName,"Cuda");
}

#ifdef OPENCL_ENABLED
template<> inline
///
/// \brief CGpGpuContext<openClContext>::errorDump retourne dans la console le type d'erreur OpenCL
/// \param tErr
/// \param erName
/// \return true s'il n'y a pas d'erreur
///
bool CGpGpuContext<openClContext>::errorDump(int tErr,string erName)
{
    return errorGpGpu(tErr,erName,"openCl");
}
#endif
/// \cond
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
///
/// \brief CGpGpuContext<cudaContext>::check_Cuda renvoie dans la console le nom de la carte, sa capacité et la version de cuda
///
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
///
/// \brief CGpGpuContext<cudaContext>::createContext créer le contexte CUDA
///
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
///
/// \brief CGpGpuContext<cudaContext>::deleteContext
/// Supprime le contexte CUDA
void CGpGpuContext<cudaContext>::deleteContext()
{
    checkCudaErrors( cudaDeviceReset() );
}
/// \endcond
#if OPENCL_ENABLED

/// \cond
template <> inline
///
/// \brief CGpGpuContext::createContext
///  Créer le contexte OpenCL
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
///
/// \brief CGpGpuContext<openClContext>::deleteContext
/// Supprime le contexte OpenCL
void CGpGpuContext<openClContext>::deleteContext()
{
    if(clReleaseContext(openClContext::_contextOpenCL) == CL_SUCCESS)
        printf("CONTEXT OPENCL RELEASE\n");
}

/// \endcond
template <> inline
///
/// \brief CGpGpuContext<openClContext>::contextOpenCL
/// \return le contexte OpenCL
///
cl_context CGpGpuContext<openClContext>::contextOpenCL()
{
    return openClContext::_contextOpenCL;
}

template <> inline
///
/// \brief CGpGpuContext<openClContext>::commandQueue
/// \return la command Queue OpenCL
///
cl_command_queue CGpGpuContext<openClContext>::commandQueue()
{
    return openClContext::_commandQueue;
}

template <> inline
///
/// \brief CGpGpuContext<openClContext>::kernel
/// \return le kernel OPenCL
///
cl_kernel CGpGpuContext<openClContext>::kernel()
{
    return openClContext::_kernel;
}

template <> inline
///
/// \brief CGpGpuContext<openClContext>::createKernel Créer le contexte OpenCL
/// \param fileName Fichier contenant le code du kernel
/// \param kernelName non du kernel
///
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
/// \cond
/// \brief CGpGpuContext<openClContext>::launchKernel
/// Lance le Kernel OpenCL
///
void CGpGpuContext<openClContext>::launchKernel()
{

    cl_int error = -1;

    size_t global_item_size = 5;
    size_t local_item_size  = 1;

    error  = clEnqueueNDRangeKernel(openClContext::_commandQueue,openClContext::_kernel,1,NULL,&global_item_size,&local_item_size,0,NULL,NULL);
    errorDump(error,"Enqueue");

}
/// \endcond
template <>
template <class T> inline
///
/// \brief CGpGpuContext<openClContext>::addKernelArg Ajoute un argument
/// \param arg Argument à ajouter
///
void CGpGpuContext<openClContext>::addKernelArg(T &arg)
{

    cl_int error = -1;

    error = clSetKernelArg(CGpGpuContext<openClContext>::kernel(),(cl_uint)_sContext._nbArg,sizeof(T),&arg);

    errorDump(error,"Kernel Arg");

    _sContext._nbArg++;

}

template<>
template <class T> inline
///
/// \brief CGpGpuContext<openClContext>::addKernelArgSDK ajoute un argument de type CData<T> pour OpenCL
/// \param arg
///
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
///
/// \brief CGpGpuContext<cudaContext>::addKernelArgSDK ajoute un argument de type CData<T> pour Cuda
/// \param arg
///
void CGpGpuContext<cudaContext>::addKernelArgSDK( CData<T> &arg)
{
    printf("Add cuda kernel argument buffer : %d\n",_sContext._nbArg);

    _sContext._kernelArgs.push_back((void*)&arg);
}

template <> inline
/// \cond
/// \brief CGpGpuContext<cudaContext>::launchKernel
/// Lance le kernel Cuda
void CGpGpuContext<cudaContext>::launchKernel()
{
    kMultTab();
}
/// \endcond
template <>
template <class T> inline
///
/// \brief CGpGpuContext<cudaContext>::addKernelArg Ajoute un argument pour Cuda
/// \param arg
///
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








