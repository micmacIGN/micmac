#ifndef __DATA2OPTIMIZ_H__
#define __DATA2OPTIMIZ_H__

#include "GpGpu/GpGpu_Data.h"

template<class T >
struct  buffer
{

};

struct st_line
{
    uint lenght;
    uint id;
    __device__ inline uint LOver()
    {
        return lenght - id;
    }
};

struct p_ReadLine
{
    ushort  ID_Bf_Icost;
    st_line line;
    st_line seg;
    bool    Id_Buf;
    const ushort    tid;
    const ushort    itid;
    short2 prev_Dz;
    ushort pente;

    __device__ p_ReadLine(ushort t,ushort ipente):
        Id_Buf(false),
        tid(t),
        itid(WARPSIZE - t - 1),
        pente(ipente)
    {
        line.id = 0;
        seg.id  = 1;
    }

    __device__ inline void swBuf()
    {
        Id_Buf = !Id_Buf;
    }

    __device__ void format()
    {
        const uint d  = line.lenght >> 5;
        const uint dL = d << 5;

        if(line.lenght-dL)
            line.lenght = dL + WARPSIZE;
    }

    __device__ inline void ouput()
    {
        if(!tid)
        {
            printf("seg.id       = %d\n",seg.id);
            printf("seg.lenght   = %d\n",seg.lenght);
            printf("line.id      = %d\n",line.id);
            printf("line.lenght  = %d\n",line.lenght);
            printf("ID_Bf_Icost  = %d\n",ID_Bf_Icost);
            printf("-----------------------------\n");
        }
    }

    __device__ inline void reverse(short2 *buffindex,ushort sizeBuff)
    {
        seg.id        = seg.lenght - 1;
        prev_Dz       = buffindex[seg.id];
        seg.id        = WARPSIZE  - seg.id;
        seg.lenght    = WARPSIZE;
        line.id       = 0;
        format();
        ID_Bf_Icost   = sizeBuff - ID_Bf_Icost + count(prev_Dz);
    }

    template<bool sens> __device__ inline ushort stid()
    {
        return sens ? tid : itid;
    }

};

template<template<class T> class U, uint NBUFFER = 1 >
struct Data2Optimiz
{
    Data2Optimiz();

    ~Data2Optimiz();

    void Dealloc();

    void ReallocParam(uint size);

    void SetParamLine(uint id, uint pStr,uint pIdStr, uint lLine, uint idbuf = 0);

    void ReallocIf(uint pStr,uint pIdStr);

    void ReallocInputIf(uint pStr,uint pIdStr);

    void ReallocOutputIf(uint pStr, uint idbuf = 0);

    void ReallocIf(Data2Optimiz<CuHostData3D,2> &d2o);

    void ReallocInputIf(Data2Optimiz<CuHostData3D,2> &d2o);

    void ReallocOutputIf(Data2Optimiz<CuHostData3D,2> &d2o, uint idbuf = 0);

    void SetNbLine(uint nbl);

    void CopyHostToDevice(Data2Optimiz<CuHostData3D,2> &d2o, uint idbuf = 0);

    void CopyDevicetoHost(Data2Optimiz<CuHostData3D,2> &d2o, uint idbuf = 0);

    uint NBlines(){return _nbLines;}

    ushort*     pInitCost(){    return _s_InitCostVol.pData();}
    short2*     pIndex(){       return _s_Index.pData();}
    uint*       pForceCostVol(){return _s_ForceCostVol[0].pData();}
    uint3*      pParam(){       return _param[0].pData();}

    U<uint3>     _param[NBUFFER];
    U<ushort>    _s_InitCostVol;
    U<uint>      _s_ForceCostVol[NBUFFER];
    U<short2>    _s_Index;

    uint         _nbLines;
    bool         _idBuffer;
    ushort       _m_DzMax;
};

TEMPLATE_D2OPTI
Data2Optimiz<U,NBUFFER>::~Data2Optimiz()
{
    Dealloc();
}

TEMPLATE_D2OPTI
Data2Optimiz<U,NBUFFER>::Data2Optimiz():
    _idBuffer(false),
    _m_DzMax(0)
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


TEMPLATE_D2OPTI
void Data2Optimiz<U,NBUFFER>::Dealloc()
{
    _s_InitCostVol  .Dealloc();
    _s_Index        .Dealloc();

    for(uint i = 0;i < NBUFFER;i++)
    {
        _param[i]           .Dealloc();
        _s_ForceCostVol[i]  .Dealloc();
    }
}

TEMPLATE_D2OPTI
void Data2Optimiz<U,NBUFFER>::ReallocParam(uint size)
{
    for(uint i = 0;i < NBUFFER;i++)
        _param[i].Dealloc();

    for(uint i = 0;i < NBUFFER;i++)
        _param[i].Realloc(size);
}

TEMPLATE_D2OPTI
void Data2Optimiz<U,NBUFFER>::SetParamLine(uint id, uint pStr, uint pIdStr, uint lLine, uint idbuf)
{
    _param[idbuf][id] = make_uint3(pStr,pIdStr,lLine);
}

TEMPLATE_D2OPTI
void Data2Optimiz<U,NBUFFER>::ReallocIf(uint pStr, uint pIdStr)
{
    _s_ForceCostVol[0]  .ReallocIf(pStr);
    ReallocInputIf(pStr, pIdStr);
}

TEMPLATE_D2OPTI
void Data2Optimiz<U,NBUFFER>::ReallocInputIf(uint pStr, uint pIdStr)
{
    _s_InitCostVol  .ReallocIf(pStr);
    _s_Index        .ReallocIf(pIdStr);

}

TEMPLATE_D2OPTI
void Data2Optimiz<U,NBUFFER>::ReallocOutputIf(uint pStr, uint idbuf)
{
    _s_ForceCostVol[idbuf] .ReallocIf(pStr);
}

TEMPLATE_D2OPTI
void Data2Optimiz<U,NBUFFER>::ReallocIf(Data2Optimiz<CuHostData3D,2> &d2o)
{
    ReallocIf(d2o._s_InitCostVol.GetSize(),d2o._s_Index.GetSize());
}

TEMPLATE_D2OPTI
void Data2Optimiz<U,NBUFFER>::ReallocInputIf(Data2Optimiz<CuHostData3D, 2> &d2o)
{
    ReallocInputIf(d2o._s_InitCostVol.GetSize(),d2o._s_Index.GetSize());
}

TEMPLATE_D2OPTI
void Data2Optimiz<U,NBUFFER>::ReallocOutputIf(Data2Optimiz<CuHostData3D, 2> &d2o, uint idbuf)
{
    ReallocOutputIf(d2o._s_InitCostVol.GetSize(),idbuf);
}

TEMPLATE_D2OPTI
void Data2Optimiz<U,NBUFFER>::SetNbLine(uint nbl)
{
    _nbLines = nbl;
}

TEMPLATE_D2OPTI
void Data2Optimiz<U,NBUFFER>::CopyHostToDevice(Data2Optimiz<CuHostData3D, 2> &d2o, uint idbuf)
{
    _s_InitCostVol.CopyHostToDevice(    d2o._s_InitCostVol .pData());
    _s_Index.CopyHostToDevice(          d2o._s_Index       .pData());    
    _param[0].CopyHostToDevice(         d2o._param[idbuf]  .pData());
}

TEMPLATE_D2OPTI
void Data2Optimiz<U,NBUFFER>::CopyDevicetoHost(Data2Optimiz<CuHostData3D, 2> &d2o, uint idbuf)
{
    _s_ForceCostVol[0].CopyDevicetoHost(d2o._s_ForceCostVol[idbuf]);
}

#endif //__DATA2OPTIMIZ_H__



