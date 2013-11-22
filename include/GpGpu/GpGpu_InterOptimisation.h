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

template <class T>
struct CuHostDaPo3D
{
    CuHostData3D<T>         _data1D;
    CuHostData3D<short2>    _ptZ;
    CuHostData3D<ushort>    _dZ;
    CuHostData3D<uint>      _pit;
    uint                    _size;

    void                    ReallocPt(uint2 dim)
    {
        _ptZ.ReallocIf(dim);
        _dZ.ReallocIf(dim);
        _pit.ReallocIf(dim);
        _size = 0;
    }

    void                    ReallocData()
    {
        _data1D.ReallocIf(_size);
    }

    void                    Dealloc()
    {
        _data1D.Dealloc();
        _ptZ.Dealloc();
        _dZ.Dealloc();
        _pit.Dealloc();
    }

    void                    PointIncre(uint2 pt,short2 ptZ)
    {
        ushort dZ   = abs(count(ptZ));
        _ptZ[pt]    = ptZ;
        _dZ[pt]     = dZ;
        _pit[pt]    = _size;
        _size      += dZ;
    }

    uint                    Pit(uint2 pt)
    {
        return _pit[pt];
    }

    short2                  PtZ(uint2 pt)
    {
        return _ptZ[pt];
    }

    ushort                  DZ(uint2 pt)
    {
        return _dZ[pt];
    }

    ushort                  DZ(uint ptX,uint ptY)
    {
        return _dZ[make_uint2(ptX,ptY)];
    }

    uint                    Size()
    {
        return _size;
    }

    T*                      operator[](uint2 pt)
    {
        return _data1D.pData() + _pit[pt];
    }

    T&                      operator[](int3 pt)
    {
        uint2 ptTer = make_uint2(pt.x,pt.y);
        return *(_data1D.pData() + _pit[ptTer] - _ptZ[ptTer].x + pt.z);
    }
};


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
    //void            Prepare(uint x,uint y);

    void            Prepare_V03(uint x,uint y);

    ///
    /// \brief freezeCompute
    ///
    void            freezeCompute();

//    CuHostData3D<ushort*>   _preCostInit;
//    CuHostData3D<uint*>     _preFinalCost;
//    CuHostData3D<short2>    _prePtZ;
//    CuHostData3D<ushort>    _preDZ;

//    CuHostData3D<ushort>    _preInitCost1D;
      CuHostData3D<uint>      _preFinalCost1D;
//    CuHostData3D<uint>      _prePitTer;

//    uint                    _pit;

    CuHostDaPo3D<ushort>      _poInitCost;

private:

    void            threadCompute();

    HOST_Data2Opti  _H_data2Opt;
    DEVC_Data2Opti  _D_data2Opt;

};


#endif
