#ifndef __OPTIMISATION_H__
#define __OPTIMISATION_H__

#include "GpGpu/SData2Optimize.h"
#include "GpGpu/GpGpu_MultiThreadingCpu.h"

template <class T>
void LaunchKernel();

extern "C" void Launch(uint* value);
extern "C" void OptimisationOneDirection(DEVC_Data2Opti  &d2O);
extern "C" void OptimisationOneDirectionZ_V01(DEVC_Data2Opti  &d2O);
extern "C" void OptimisationOneDirectionZ_V02(DEVC_Data2Opti  &d2O);

/// \class InterfOptimizGpGpu
/// \brief Class qui permet a micmac de lancer les calculs d optimisations sur le Gpu
///
class InterfOptimizGpGpu : public CSimpleJobCpuGpu<bool>
{

public:
    ///
    /// \brief InterfOptimizGpGpu
    ///
    InterfOptimizGpGpu();
    ~InterfOptimizGpGpu();

    ///
    /// \brief Data2Opt
    /// \return
    ///
    HOST_Data2Opti& HData2Opt(){ return _H_data2Opt;}

    DEVC_Data2Opti& DData2Opt(){ return _D_data2Opt;}

    ///
    /// \brief Dealloc
    ///
    void            Dealloc();

    ///
    /// \brief oneDirOptGpGpu
    ///

    ///
    /// \brief oneDirOptGpGpu
    ///
    void            oneDirOptGpGpu();

    ///
    /// \brief ReallocParam
    /// \param size
    ///
    void            Prepare(uint x,uint y);

    void            Prepare_V03(uint x,uint y);

    ///
    /// \brief freezeCompute
    ///
    void            freezeCompute();

    CuHostData3D<ushort*>   _preCostInit;
    CuHostData3D<uint*>     _preFinalCost;   
    CuHostData3D<short2>    _prePtZ;
    CuHostData3D<short>     _preDZ;

    CuHostData3D<ushort>    _preInitCost1D;
    CuHostData3D<uint>      _preFinalCost1D;
    CuHostData3D<uint>      _prePitTer;

private:

    void            threadCompute();

    HOST_Data2Opti  _H_data2Opt;
    DEVC_Data2Opti  _D_data2Opt;

};


#endif
