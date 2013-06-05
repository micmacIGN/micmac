#ifndef __DATA2OPTIMIZ_H__
#define __DATA2OPTIMIZ_H__

#include "GpGpu/GpGpuTools.h"

//#define   NBUFFER   2

template<template<class T> class U, uint NBUFFER = 1 >
struct Data2Optimiz
{
    U<uint3>     _param[NBUFFER];
    U<uint>      _s_InitCostVol;
    U<uint>      _s_ForceCostVol;
    U<short2>    _s_Index;
    uint         _nbLines;
    bool         _idBuffer;

    Data2Optimiz():
        _s_InitCostVol((uint)1),
        _s_ForceCostVol((uint)1),
        _s_Index((uint)1),
        _idBuffer(false)
    {
        for(uint i = 0;i < NBUFFER;i++)
            _param[i].SetName("param");
        ReallocParam(1);
    }

    void Dealloc()
    {
        _s_InitCostVol  .Dealloc();
        _s_ForceCostVol .Dealloc();
        _s_Index        .Dealloc();
        _param[0]       .Dealloc();
    }

    void SwitchBuffer()
    {
        _idBuffer = !_idBuffer;
    }

    void ReallocParam(uint size)
    {
        for(uint i = 0;i < NBUFFER;i++)
            _param[i].Realloc(size);
    }

    void SetParamLine(uint id, uint pStr,uint pIdStr, uint lLine, uint idbuf = 0)
    {
        _param[idbuf][id] = make_uint3(pStr,pIdStr,lLine);
    }

    void ReallocIf(uint pStr,uint pIdStr)
    {
        _s_InitCostVol  .ReallocIf(pStr);
        _s_ForceCostVol .ReallocIf(pStr);
        _s_Index        .ReallocIf(pIdStr);
    }

    void ReallocInputIf(uint pStr,uint pIdStr)
    {
        _s_InitCostVol  .ReallocIf(pStr);
        _s_Index        .ReallocIf(pIdStr);

    }

    void ReallocOutputIf(uint pStr)
    {
        _s_ForceCostVol .ReallocIf(pStr);
    }

    void ReallocIf(Data2Optimiz<CuHostData3D,2> &d2o)
    {
        ReallocIf(d2o._s_InitCostVol.GetSize(),d2o._s_Index.GetSize());
    }

    void ReallocInputIf(Data2Optimiz<CuHostData3D,2> &d2o)
    {
        ReallocInputIf(d2o._s_InitCostVol.GetSize(),d2o._s_Index.GetSize());
    }

    void ReallocOutputIf(Data2Optimiz<CuHostData3D,2> &d2o)
    {
        ReallocOutputIf(d2o._s_InitCostVol.GetSize());
    }

    void SetNbLine(uint nbl)
    {
        _nbLines = nbl;
    }

    void CopyHostToDevice(Data2Optimiz<CuHostData3D,2> &d2o, uint idbuf = 0)
    {
        _s_InitCostVol.CopyHostToDevice(    d2o._s_InitCostVol .pData());
        _s_Index.CopyHostToDevice(          d2o._s_Index       .pData());
        _param[0].CopyHostToDevice(         d2o._param[idbuf]  .pData());
    }

    void CopyDevicetoHost(Data2Optimiz<CuHostData3D,2> &d2o)
    {
         _s_ForceCostVol.CopyDevicetoHost(d2o._s_ForceCostVol.pData());
    }

    void CopyDevicetoHost(uint *forceCost)
    {
         _s_ForceCostVol.CopyDevicetoHost(forceCost);
    }
};

#endif //__DATA2OPTIMIZ_H__
