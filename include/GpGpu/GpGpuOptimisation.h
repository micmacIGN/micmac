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
class InterfOptimizGpGpu : public CSimpleJobCpuGpu<bool>
{
public:
    InterfOptimizGpGpu();
    ~InterfOptimizGpGpu();


    HOST_Data2Opti& Data2Opt(){ return _H_data2Opt;}

    void            Dealloc();
    void            oneDirOptGpGpu();
    void            ReallocParam(uint size);

private:

    void            threadCompute();
    void            freezeCompute();

    HOST_Data2Opti  _H_data2Opt;
    DEVC_Data2Opti  _D_data2Opt;

    bool            _idbuf;
    uint            _idDir;


};


#endif
