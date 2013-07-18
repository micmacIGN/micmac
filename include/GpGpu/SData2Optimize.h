#ifndef __DATA2OPTIMIZ_H__
#define __DATA2OPTIMIZ_H__

#include "GpGpu/GpGpuTools.h"

template<template<class T> class U, uint NBUFFER = 1 >
struct Data2Optimiz
{
    U<uint3>     _param[NBUFFER];
    U<uint>      _s_InitCostVol;
    U<uint>      _s_ForceCostVol[NBUFFER];
    U<short2>    _s_Index;
    uint         _nbLines;
    bool         _idBuffer;

    Data2Optimiz():
        _idBuffer(false)
    {     
        for(uint i = 0;i < NBUFFER;i++)
        {
            _s_ForceCostVol[i].SetName("_s_ForceCostVol_0",i);
            _param[i].SetName("_param",i);
        }

        _s_InitCostVol.SetName("_s_InitCostVol");
        _s_Index.SetName("_s_Index");
        ReallocParam(1);
    }

    void Dealloc()
    {
        _s_InitCostVol  .Dealloc();        
        _s_Index        .Dealloc();

        for(uint i = 0;i < NBUFFER;i++)
        {
            _param[i]           .Dealloc();
            _s_ForceCostVol[i]  .Dealloc();
        }
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
        _s_ForceCostVol[0]  .ReallocIf(pStr);
        ReallocInputIf(pStr, pIdStr);
    }

    void ReallocInputIf(uint pStr,uint pIdStr)
    {
        _s_InitCostVol  .ReallocIf(pStr);
        _s_Index        .ReallocIf(pIdStr);
    }

    void ReallocOutputIf(uint pStr, uint idbuf = 0)    
    {
        _s_ForceCostVol[idbuf] .ReallocIf(pStr);
    }

    void ReallocIf(Data2Optimiz<CuHostData3D,2> &d2o)
    {
        ReallocIf(d2o._s_InitCostVol.GetSize(),d2o._s_Index.GetSize());
    }

    void ReallocInputIf(Data2Optimiz<CuHostData3D,2> &d2o)
    {
        ReallocInputIf(d2o._s_InitCostVol.GetSize(),d2o._s_Index.GetSize());
    }

    void ReallocOutputIf(Data2Optimiz<CuHostData3D,2> &d2o, uint idbuf = 0)
    {
        ReallocOutputIf(d2o._s_InitCostVol.GetSize(),idbuf);
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

    void CopyDevicetoHost(Data2Optimiz<CuHostData3D,2> &d2o, uint idbuf = 0)
    {
         _s_ForceCostVol[0].CopyDevicetoHost(d2o._s_ForceCostVol[idbuf].pData());
    }

};

#endif //__DATA2OPTIMIZ_H__
