#ifndef __OPTIMISATION_H__
#define __OPTIMISATION_H__

#include "GpGpu/SData2Optimize.h"
#include "GpGpu/GpGpu_MultiThreadingCpu.h"
#include "GpGpu/GpGpu_eLiSe.h"

extern "C" void Gpu_OptimisationOneDirection(DEVC_Data2Opti  &d2O);

template <class T>
///
/// \brief The CuHostDaPo3D struct
/// Structure 1D des couts de corélation
struct sMatrixCellCost
{
    CuHostData3D<T>         _CostInit1D;
    CuHostData3D<short3>    _ptZ;
    CuHostData3D<ushort>    _dZ;
    CuHostData3D<uint>      _pit;
    uint                    _size;
    ushort                  _maxDz;

    sMatrixCellCost():
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
        _CostInit1D.ReallocIf(_size);

    }

    void                    fillCostInit(ushort val)
    {
        _CostInit1D.Fill(val);
    }

    void                    Dealloc()
    {
        _CostInit1D.Dealloc();

        _ptZ.Dealloc();
        _dZ.Dealloc();
        _pit.Dealloc();
    }        

    void PointIncre(uint2 pt,short2 ptZ)
    {
        ushort dZ   = abs(count(ptZ));
        _ptZ[pt]    = make_short3(ptZ.x,ptZ.y,0);
        _dZ[pt]     = dZ;

        // NAPPEMAX
        if(_maxDz < dZ) // Calcul de la taille de la Nappe Max pour le calcul Gpu
            _maxDz = iDivUp32(dZ) * WARPSIZE;

        _pit[pt]    = _size;
        _size      += dZ;
    }

    void                    setDefCor(uint2 pt,short defCor)
    {
        _ptZ[pt].z    = defCor;
    }

    uint                    Pit(uint2 pt)
    {
        return _pit[pt];
    }

    uint                    Pit(Pt2di pt)
    {
        return _pit[toUi2(pt)];
    }

    short3                  PtZ(uint2 pt)
    {
        return _ptZ[pt];
    }

    short3                  PtZ(Pt2di pt)
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
        return _CostInit1D.pData() + _pit[pt];
    }

    T*                      operator[](Pt2di pt)
    {
        return _CostInit1D.pData() + _pit[toUi2(pt)];
    }

    T&                      operator[](int3 pt)
    {
        uint2 ptTer = make_uint2(pt.x,pt.y);
        return *(_CostInit1D.pData() + _pit[ptTer] - _ptZ[ptTer].x + pt.z);
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

    void            Prepare(uint x, uint y, ushort penteMax, ushort NBDir, float zReg, float zRegQuad, ushort costDefMask,ushort costDefMaskTrans);

    ///
    /// \brief freezeCompute
    ///
    void            freezeCompute();

    CuHostData3D<uint>      _preFinalCost1D;
    CuHostData3D<uint>      _FinalDefCor;
    sMatrixCellCost<ushort>    _poInitCost;

    void            optimisation();

private:

    void            simpleWork();

    HOST_Data2Opti  _H_data2Opt;
    DEVC_Data2Opti  _D_data2Opt;

};


#endif
