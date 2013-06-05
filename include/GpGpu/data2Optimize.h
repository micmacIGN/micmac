#ifndef __DATA2OPTIMIZ_H__
#define __DATA2OPTIMIZ_H__

#include "GpGpu/GpGpuTools.h"

template<template<class T> class U>
struct Data2Optimiz
{
    U<uint3>     h__Param;
    U<uint>      hS_InitCostVol;
    U<uint>      hS_ForceCostVol;
    U<short2>    hS_Index;
    uint         nbLines;

    Data2Optimiz():
        h__Param((uint)1),
        hS_InitCostVol((uint)1),
        hS_ForceCostVol((uint)1),
        hS_Index((uint)1)
    {}

    void Dealloc()
    {
        hS_InitCostVol  .Dealloc();
        hS_ForceCostVol .Dealloc();
        hS_Index        .Dealloc();
        h__Param        .Dealloc();
    }

    void ReallocParam(uint size)
    {
        h__Param.Realloc(size);
    }

    void SetParamLine(uint id, uint pStr,uint pIdStr, uint lLine)
    {
        h__Param[id] = make_uint3(pStr,pIdStr,lLine);
    }

    void ReallocIf(uint pStr,uint pIdStr)
    {
        hS_InitCostVol  .ReallocIf(pStr);
        hS_ForceCostVol .ReallocIf(pStr);
        hS_Index        .ReallocIf(pIdStr);
    }

    void ReallocIf(Data2Optimiz<CuHostData3D> d2o)
    {
        ReallocIf(d2o.hS_InitCostVol.GetSize(),d2o.hS_Index.GetSize());
    }

    void SetNbLine(uint nbl)
    {
        nbLines = nbl;
    }

    void CopyHostToDevice(Data2Optimiz<CuHostData3D> d2o)
    {
        hS_InitCostVol.CopyHostToDevice(    d2o.hS_InitCostVol .pData());
        hS_Index.CopyHostToDevice(          d2o.hS_Index       .pData());
        h__Param.CopyHostToDevice(          d2o.h__Param       .pData());


    }


    void CopyDevicetoHost(Data2Optimiz<CuHostData3D> d2o)
    {
         hS_ForceCostVol.CopyDevicetoHost(d2o.hS_ForceCostVol.pData());
    }


};

#endif //__DATA2OPTIMIZ_H__
