#ifndef __OPTIMISATION_H__
#define __OPTIMISATION_H__

#include "GpGpu/SData2Optimize.h"
#include "GpGpu/GpGpu_MultiThreadingCpu.h"
#include "GpGpu/GpGpu_eLiSe.h"

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
    ushort                  _maxDz;

    CuHostDaPo3D():
        _maxDz(NAPPEMAX) // ATTENTION : NAPPE Dynamique
    {}

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

    void PointIncre(uint2 pt,short2 ptZ)
    {
        ushort dZ   = abs(count(ptZ));
        _ptZ[pt]    = ptZ;
        _dZ[pt]     = dZ; // PREDEFCOR : _dZ[pt]+1 reserved cell

    // ATTENTION : Nappe Dynamique!! _maxDz
    // NAPPEMAX
        if(_maxDz < dZ)
        {
            _maxDz = iDivUp32(dZ) * WARPSIZE;
            //DUMP_INT(_maxDz)
        }
        _pit[pt]    = _size;
        _size      += dZ;
    }

    uint                    Pit(uint2 pt)
    {
        return _pit[pt];
    }

    uint                    Pit(Pt2di pt)
    {
        return _pit[toUi2(pt)];
    }



    short2                  PtZ(uint2 pt)
    {
        return _ptZ[pt];
    }

    short2                  PtZ(Pt2di pt)
    {
        return _ptZ[toUi2(pt)];
    }

    ushort                  DZ(uint2 pt)
    {
        return _dZ[pt];
    }

    ushort                  DZ(Pt2di pt)
    {
        return _dZ[toUi2(pt)];
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

    T*                      operator[](Pt2di pt)
    {
        return _data1D.pData() + _pit[toUi2(pt)];
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

    void            Prepare(uint x, uint y, ushort penteMax, ushort NBDir, float zReg, float zRegQuad, ushort costDefMask,ushort costDefMaskTrans);

    ///
    /// \brief freezeCompute
    ///
    void            freezeCompute();

    void            simpleJob();

    CuHostData3D<uint>      _preFinalCost1D;
    CuHostData3D<uint>      _FinalDefCor;

    CuHostDaPo3D<ushort>    _poInitCost;

    void oneCompute();
    void optimisation();
private:

    void            threadCompute();

    HOST_Data2Opti  _H_data2Opt;
    DEVC_Data2Opti  _D_data2Opt;

};


#endif
