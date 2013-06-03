#ifndef __OPTIMISATION_H__
#define __OPTIMISATION_H__

#include "GpGpu/GpGpuTools.h"

template <class T>
void LaunchKernel();


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
    void SetNbLine(uint nbl)
    {
        nbLines = nbl;
    }

};

//struct Data2Optimiz
//{
//    CuHostData3D<uint3>     h__Param;
//    CuHostData3D<uint>      hS_InitCostVol;
//    CuHostData3D<uint>      hS_ForceCostVol;
//    CuHostData3D<short2>    hS_Index;
//    uint                    nbLines;

//    Data2Optimiz():
//        h__Param((uint)1),
//        hS_InitCostVol((uint)1),
//        hS_ForceCostVol((uint)1),
//        hS_Index((uint)1)
//    {}

//    void Dealloc()
//    {
//        hS_InitCostVol  .Dealloc();
//        hS_ForceCostVol .Dealloc();
//        hS_Index        .Dealloc();
//        h__Param        .Dealloc();
//    }

//    void ReallocParam(uint size)
//    {
//        h__Param.Realloc(size);
//    }

//    void SetParamLine(uint id, uint pStr,uint pIdStr, uint lLine)
//    {
//        h__Param[id] = make_uint3(pStr,pIdStr,lLine);
//    }

//    void ReallocIf(uint pStr,uint pIdStr)
//    {
//        hS_InitCostVol  .ReallocIf(pStr);
//        hS_ForceCostVol .ReallocIf(pStr);
//        hS_Index        .ReallocIf(pIdStr);
//    }
//    void SetNbLine(uint nbl)
//    {
//        nbLines = nbl;
//    }
//};

extern "C" void Launch();
extern "C" void OptimisationOneDirection(Data2Optimiz<CuHostData3D> &d2O);

/// \class InterfMicMacOptGpGpu
/// \brief Class qui permet a micmac de lancer les calculs d optimisations sur le Gpu
class InterfMicMacOptGpGpu
{
public:
    InterfMicMacOptGpGpu();
    ~InterfMicMacOptGpGpu();

    /// \brief  Restructuration des donnes du volume de correlation
    ///         Pour le moment il lance egalement le calcul d optimisation
    //void StructureVolumeCost(CuHostData3D<float> &volumeCost, float defaultValue);

private:

    //CuHostData3D<int> _volumeCost;
};


#endif
