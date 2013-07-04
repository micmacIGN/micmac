#ifndef __OPTIMISATION_H__
#define __OPTIMISATION_H__

#include "GpGpu/GpGpuTools.h"
#include "GpGpu/data2Optimize.h"
#include "GpGpu/GpGpuMultiThreadingCpu.h"

#include <boost/thread/thread.hpp>

template <class T>
void LaunchKernel();

#define HOST_Data2Opti Data2Optimiz<CuHostData3D,2>
#define DEVC_Data2Opti Data2Optimiz<CuDeviceData3D>

extern "C" void Launch(uint* value);
extern "C" void OptimisationOneDirection(DEVC_Data2Opti  &d2O);

/// \class InterfMicMacOptGpGpu
/// \brief Class qui permet a micmac de lancer les calculs d optimisations sur le Gpu
class InterfOptimizGpGpu : public GpGpuMultiThreadingCpu<CuHostData3D<uint>,CuDeviceData3D<uint> >
{
public:
    InterfOptimizGpGpu(bool UseMultiThreading = true);
    ~InterfOptimizGpGpu();


    HOST_Data2Opti& Data2Opt(){ return _H_data2Opt;}

    void            Dealloc();
    void            oneDirOptGpGpu();
    void            ReallocParam(uint size);

    void            createThreadOptGpGpu();
    void            deleteThreadOptGpGpu();

    void            SetCompute(bool compute);
    bool            GetCompute();

    void            SetDirToCopy(bool copy);
    bool            GetDirToCopy();

    void            SetPreCompNextDir(bool precompute);
    bool            GetPreCompNextDir();

    bool            UseMultiThreading();

private:

    virtual void    InitPrecompute(){}
    virtual void    Precompute(HOST_UINT3D* hostIn){}
    virtual void    GpuCompute(){}

    void            threadFuncOptimi();

    HOST_Data2Opti  _H_data2Opt;
    DEVC_Data2Opti  _D_data2Opt;
    boost::thread*  _gpGpuThreadOpti;

    boost::mutex    _mutexCompu;
    boost::mutex    _mutexCopy;
    boost::mutex    _mutexPreCompute;

    bool            _compute;
    bool            _copy;
    bool            _precompute;

    bool            _idbuf;
    uint            _idDir;
    bool            _multiThreading;

};


#endif
