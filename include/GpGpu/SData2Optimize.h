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
    ushort          ID_Bf_Icost;
    st_line         line;
    st_line         seg;
    bool            Id_Buf;
    const ushort    tid;
    const ushort    itid;
    short2          prev_Dz;
    ushort          prevDefCor;
    ushort          pente;
    float           ZRegul;
    float           ZRegul_Quad;
    const ushort    costDefMask;
    const ushort    costTransDefMask;
    const bool      hasMaskauto;
    const ushort    sizeBuffer;

    __device__ p_ReadLine(ushort t,ushort ipente,float zReg,float zRegQuad,ushort pCostDefMask, ushort pCostTransDefMask,ushort pSizebuffer,bool automask):
        Id_Buf(false),
        tid(t),
        itid(WARPSIZE - t - 1),
        pente(ipente),
        ZRegul(zReg),
        ZRegul_Quad(zRegQuad),
        costDefMask(pCostDefMask),
        costTransDefMask(pCostTransDefMask),
        hasMaskauto(automask),
        sizeBuffer(pSizebuffer)
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

    __device__ inline void reverse(short3 *buffindex,ushort sizeBuff)
    {
        seg.id        = seg.lenght - 1;
        short3 tp     = buffindex[seg.id];
        prev_Dz       = make_short2(tp.x,tp.y);
        prevDefCor    = tp.z;
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
///
/// \brief The Data2Optimiz struct
///
struct Data2Optimiz
{
public:

    Data2Optimiz();

    ~Data2Optimiz();

    ///
    /// \brief Dealloc
    ///
    void Dealloc();

    void ReallocParam(uint size);

    void SetParamLine(uint id, uint pStr,uint pIdStr, uint lLine, uint idbuf = 0);

    void ReallocIf(uint pStr,uint pIdStr);

    void ReallocInputIf(uint pStr,uint pIdStr);

    void ReallocOutputIf(uint pStr, uint pIdStr,uint idbuf = 0);

    void ReallocIf(Data2Optimiz<CuHostData3D,2> &d2o);

    void ReallocInputIf(Data2Optimiz<CuHostData3D,2> &d2o);

    void ReallocOutputIf(Data2Optimiz<CuHostData3D,2> &d2o, uint idbuf = 0);

    void SetNbLine(uint nbl);

    void CopyHostToDevice(Data2Optimiz<CuHostData3D,2> &d2o, uint idbuf = 0);

    void CopyDevicetoHost(Data2Optimiz<CuHostData3D,2> &d2o, uint idbuf = 0);

    uint NBlines(){return _nbLines;}

    ushort*     pInitCost(){    return  _s_InitCostVol.pData();}
    short3*     pIndex(){       return  _s_Index.pData();}
    uint*       pDefCor(){      return  _s_DefCor[0].pData();}
    uint*       pForceCostVol(){return  _s_ForceCostVol[0].pData();}
    uint3*      pParam(){       return  _param[0].pData();}

    ushort      penteMax() const;
    void        setPenteMax(const ushort &penteMax);

    uint        nbLines() const;
    void        setNbLines(const uint &nbLines);

    U<ushort>   s_InitCostVol() const;

    U<uint3>    param(ushort i) const;

    U<short3>   s_Index() const;

    U<uint>     &s_ForceCostVol(ushort i);

    U<uint>     &s_DefCor(ushort i);

    ushort      DzMax() const;

    void        setDzMax(const ushort &m_DzMax);

    float       zReg() const;
    void        setZReg(float zReg);

    float       zRegQuad() const;
    void        setZRegQuad(float zRegQuad);

    ushort      CostDefMasked() const;
    void        setCostDefMasked(const ushort &CostDefMasked);

    ushort      CostTransMaskNoMask() const;
    void        setCostTransMaskNoMask(const ushort &CostTransMaskNoMask);

    bool        hasMaskAuto() const;
    void        setHasMaskAuto(const bool &hasMaskAuto);
private:

    U<uint3>     _param[NBUFFER];
    U<ushort>    _s_InitCostVol;
    U<uint>      _s_ForceCostVol[NBUFFER];
    U<short3>    _s_Index;
    U<uint>      _s_DefCor[NBUFFER];

    uint         _nbLines;
    bool         _idBuffer;
    ushort       _penteMax;
    ushort       _CostDefMasked;
    ushort       _CostTransMaskNoMask;
    float        _zReg;
    float        _zRegQuad;
    bool         _hasMaskAuto;

    ushort       _m_DzMax;
};

TEMPLATE_D2OPTI
Data2Optimiz<U,NBUFFER>::~Data2Optimiz()
{
    Dealloc();
}

TEMPLATE_D2OPTI
Data2Optimiz<U,NBUFFER>::Data2Optimiz():
    _nbLines(0),
    _idBuffer(false),
    _penteMax(0),
    _m_DzMax(NAPPEMAX)
{
    for(uint i = 0;i < NBUFFER;i++)
    {
        _s_ForceCostVol[i].SetName("_s_ForceCostVol_0",i);
        _s_DefCor[i].SetName("_s_DefCor_0",i);
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
        _s_DefCor[i]        .Dealloc();
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
    _s_DefCor[0]        .ReallocIf(pIdStr);
    ReallocInputIf(pStr, pIdStr);
}

TEMPLATE_D2OPTI
void Data2Optimiz<U,NBUFFER>::ReallocInputIf(uint pStr, uint pIdStr)
{
    _s_InitCostVol  .ReallocIf(pStr);
    _s_Index        .ReallocIf(pIdStr);

}

TEMPLATE_D2OPTI
void Data2Optimiz<U,NBUFFER>::ReallocOutputIf(uint pStr, uint pIdStr, uint idbuf)
{
    _s_ForceCostVol[idbuf] .ReallocIf(pStr);
    _s_DefCor[idbuf]       .ReallocIf(pIdStr);
}

TEMPLATE_D2OPTI
void Data2Optimiz<U,NBUFFER>::ReallocIf(Data2Optimiz<CuHostData3D,2> &d2o)
{
    ReallocIf(d2o.s_InitCostVol().GetSize(),d2o.s_Index().GetSize());
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
    _s_InitCostVol.CopyHostToDevice(    d2o.s_InitCostVol().pData());
    _s_Index.CopyHostToDevice(          d2o.s_Index()      .pData());
    _param[0].CopyHostToDevice(         d2o.param(idbuf)   .pData());
}

TEMPLATE_D2OPTI
void Data2Optimiz<U,NBUFFER>::CopyDevicetoHost(Data2Optimiz<CuHostData3D, 2> &h2o, uint idbuf)
{
    _s_ForceCostVol[0].CopyDevicetoHost(h2o.s_ForceCostVol(idbuf));
    _s_DefCor[0].CopyDevicetoHost(h2o.s_DefCor(idbuf));
}

TEMPLATE_D2OPTI
ushort Data2Optimiz<U,NBUFFER>::penteMax() const
{
    return _penteMax;
}

TEMPLATE_D2OPTI
void Data2Optimiz<U,NBUFFER>::setPenteMax(const ushort &penteMax)
{
    _penteMax = penteMax;
}

TEMPLATE_D2OPTI
uint Data2Optimiz<U,NBUFFER>::nbLines() const
{
    return _nbLines;
}
TEMPLATE_D2OPTI
void Data2Optimiz<U,NBUFFER>::setNbLines(const uint &nbLines)
{
    _nbLines = nbLines;
}

TEMPLATE_D2OPTI
U<ushort> Data2Optimiz<U,NBUFFER>::s_InitCostVol() const
{
    return _s_InitCostVol;
}

TEMPLATE_D2OPTI
U<uint3> Data2Optimiz<U,NBUFFER>::param(ushort i) const
{
    return _param[i];
}

TEMPLATE_D2OPTI
U<short3> Data2Optimiz<U,NBUFFER>::s_Index() const
{
    return _s_Index;
}

TEMPLATE_D2OPTI
U<uint> &Data2Optimiz<U,NBUFFER>::s_ForceCostVol(ushort i)
{
    return _s_ForceCostVol[i];
}

TEMPLATE_D2OPTI
U<uint> &Data2Optimiz<U,NBUFFER>::s_DefCor(ushort i)
{
    return _s_DefCor[i];
}

TEMPLATE_D2OPTI
ushort Data2Optimiz<U,NBUFFER>::DzMax() const
{
    return _m_DzMax;
}
TEMPLATE_D2OPTI
void Data2Optimiz<U,NBUFFER>::setDzMax(const ushort &m_DzMax)
{
    _m_DzMax = m_DzMax;
}
TEMPLATE_D2OPTI
float Data2Optimiz<U,NBUFFER>::zReg() const
{
    return _zReg;
}
TEMPLATE_D2OPTI
void Data2Optimiz<U,NBUFFER>::setZReg(float zReg)
{
    _zReg = zReg;
}
TEMPLATE_D2OPTI
float Data2Optimiz<U,NBUFFER>::zRegQuad() const
{
    return _zRegQuad;
}

TEMPLATE_D2OPTI
void Data2Optimiz<U,NBUFFER>::setZRegQuad(float zRegQuad)
{
    _zRegQuad = zRegQuad;
}

TEMPLATE_D2OPTI
ushort Data2Optimiz<U,NBUFFER>::CostDefMasked() const
{
    return _CostDefMasked;
}

TEMPLATE_D2OPTI
void Data2Optimiz<U,NBUFFER>::setCostDefMasked(const ushort &CostDefMasked)
{
    _CostDefMasked = CostDefMasked;
}

TEMPLATE_D2OPTI
ushort Data2Optimiz<U,NBUFFER>::CostTransMaskNoMask() const
{
    return _CostTransMaskNoMask;
}

TEMPLATE_D2OPTI
void Data2Optimiz<U,NBUFFER>::setCostTransMaskNoMask(const ushort &CostTransMaskNoMask)
{
    _CostTransMaskNoMask = CostTransMaskNoMask;
}


TEMPLATE_D2OPTI
bool Data2Optimiz<U,NBUFFER>::hasMaskAuto() const
{
    return _hasMaskAuto;
}


TEMPLATE_D2OPTI
void Data2Optimiz<U,NBUFFER>::setHasMaskAuto(const bool &hasMaskAuto)
{
    _hasMaskAuto = hasMaskAuto;
}

#endif //__DATA2OPTIMIZ_H__



