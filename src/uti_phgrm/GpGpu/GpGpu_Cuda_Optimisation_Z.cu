#ifndef _OPTIMISATION_KERNEL_Z_H_
#define _OPTIMISATION_KERNEL_Z_H_

#include "GpGpu/GpGpu_StreamData.cuh"
#include "GpGpu/SData2Optimize.h"

// On pourrait imaginer un buffer des tailles calculer en parallel
// SIZEBUFFER[threadIdx.x] = count(lI[threadIdx.x]);

#define sgn s<sens>

template< bool sens,class T>
__device__ inline T s(T v)
{
    return sens ? v : -v;
}

template<bool sens>
__device__ inline void readIndex(short2 *g_BuffIdex, short2 *s_BuffIdex, uint& id)
{
    const short tid  = sgn(threadIdx.x);
    *(s_BuffIdex + tid) = *(g_BuffIdex + tid + id);
    id += sgn(WARPSIZE);
}

template<bool sens>
__device__ inline void readInitCost(ushort *g__ICost, ushort* s__ICost, ushort& s_id, uint& g_id, ushort &sizeCopy)
{
    const short  tid  = sgn(threadIdx.x);
    const ushort rem  = sizeCopy - s_id;
    ushort* ts_Icost  = s__ICost + tid;
    ushort* mts_Icost = ts_Icost + sgn(s_id);
    ushort f = 0;

    while(rem - f > WARPSIZE)
    {
        *(ts_Icost + sgn(f)) = *(mts_Icost + sgn(f));
        f += WARPSIZE;
    }

    ushort* tls_Icost = ts_Icost + sgn(f);
    ushort* tg__Icost = g__ICost + tid + g_id;

    for(ushort i = 0; i < sgn(NAPPEMAX); i+= sgn(WARPSIZE))
        *(tls_Icost + i) = *(tg__Icost + i);

    g_id       += sgn(NAPPEMAX);
    s_id        = 0;
    sizeCopy    = NAPPEMAX + f;
}

template<class T, bool sens> __device__
void RunLine(short2 *g_BuffIdex,ushort *g_ICost,short2* S_BuffIndex,ushort *S_BuffICost,uint   &g_idIX, uint   &g_idCO,ushort &s_idCost,uint lLine)
{
    uint    idMain      = 0;
    ushort  sizeCopy    = 0;

    while(idMain < lLine)
    {
        readIndex<sens>(g_BuffIdex,S_BuffIndex,g_idIX);

        const uint segment = min(lLine-idMain,WARPSIZE);

        ushort l = 0;

        while(l < segment)
        {
            const short2 lI     = S_BuffIndex[l];
            const ushort sLI    = count(lI);

            if(s_idCost + sLI > NAPPEMAX)
                readInitCost<sens>(g_ICost,S_BuffICost,s_idCost,g_idCO,sizeCopy);

            l++;
        }

        idMain += WARPSIZE;
    }

}

template<class T> __device__
void Run(short2 *g_BuffIX,ushort *g_ICost)
{
    __shared__ short2 S_BuffIndex[WARPSIZE];
    __shared__ ushort S_BuffICost[2*NAPPEMAX];

    __shared__ uint   g_idIX;
    __shared__ uint   g_idICO;
    __shared__ ushort s_idICO;

    RunLine<T,true>(g_BuffIX,g_ICost,S_BuffIndex,S_BuffICost,g_idIX,g_idICO,s_idICO);

    g_idIX -= WARPSIZE;
    g_idICO-= NAPPEMAX;

    RunLine<T,false>(g_BuffIX,g_ICost,S_BuffIndex,S_BuffICost,g_idIX,g_idICO,s_idICO);
}


#endif //_OPTIMISATION_KERNEL_Z_H_
