#ifndef _OPTIMISATION_KERNEL_Z_H_
#define _OPTIMISATION_KERNEL_Z_H_

#include "GpGpu/GpGpu_StreamData.cuh"
#include "GpGpu/SData2Optimize.h"

// On pourrait imaginer un buffer des tailles calculer en parallel
// SIZEBUFFER[threadIdx.x] = count(lI[threadIdx.x]);

#define sgn s<sens>
#define max_cost 1e9

template< bool sens,class T>
__device__ inline T s(T v)
{
    return sens ? v : -v;
}

__device__ void GetConeZ(short2 & aDz, int aZ, int MaxDeltaZ, short2 aZ_Next, short2 aZ_Prev)
{
    aDz.x =   aZ_Prev.x-aZ;
    if (aZ != aZ_Next.x)
        aDz.x = max(aDz.x,-MaxDeltaZ);

    aDz.y = aZ_Prev.y-1-aZ;
    if (aZ != aZ_Next.y-1)
        aDz.y = min(aDz.y,MaxDeltaZ);

    if (aDz.x > aDz.y)
        if (aDz.y <0)
            aDz.x = aDz.y;
        else
            aDz.y = aDz.x;
}

template<bool sens>
__device__ inline void ReadIndex(short2 *g_BuffIdex, short2 *s_BuffIdex, uint& g_id,ushort* dZ)
{
    *(s_BuffIdex) = *(g_BuffIdex + g_id);
    *(dZ) = count(*(s_BuffIdex));
    g_id += sgn(WARPSIZE);
}

template<bool sens>
__device__ inline void ReadInitCost(ushort *g__ICst, ushort* s__ICst, ushort& s_id, uint& g_id)
{    
    for(ushort i = 0; i < sgn(NAPPEMAX); i+= sgn(WARPSIZE))
        *(s__ICst + i) = *(g__ICst + i);

    s_id  = 0;
    g_id += sgn(NAPPEMAX);
}

template<class T, bool sens> __device__
void RunLine(short2 *G_Stream_Index,ushort *G_Stream_ICost,short2* S_Bf_Index,ushort *S_Bf_ICost, uint* G_Stream_FCost, uint  &g_id_Index, uint   &g_id_ICost,ushort &s_id_ICost,uint lenghtLine)
{
    const ushort  tid       = threadIdx.x;
    const ushort penteMax   = 3;
    uint  id_Line           = 0;
    short2* ST_Bf_Index     = S_Bf_Index     + tid;
    short2* GT_Stream_Index = G_Stream_Index + tid;
    ushort* ST_Bf_ICost     = S_Bf_ICost     + tid;
    ushort* GT_Stream_ICost = G_Stream_ICost + tid;
    uint*   GT_Stream_FCost = G_Stream_FCost + tid;
    short2* ivS_Bf_Index    = S_Bf_Index     + WARPSIZE;

    __shared__ ushort s_dZ[WARPSIZE];

    short2 ConeZ;

    __shared__ uint S_FCost[2][NAPPEMAX];

    short2  prevIndex;

    bool sB = false;

    if(!sens) ReadInitCost<sens>(GT_Stream_ICost,ST_Bf_ICost,s_id_ICost,g_id_ICost);

    while(id_Line < lenghtLine)
    {
        ReadIndex<sens>(GT_Stream_Index,ST_Bf_Index,g_id_Index,s_dZ);

        uint segment = min(lenghtLine-id_Line,WARPSIZE);

        while(segment)
        {
            const short2 index  = *(ivS_Bf_Index + sgn(segment));
            const ushort dZ     = *(s_dZ + WARPSIZE  + sgn(segment));
            ushort       z      = 0;
            uint globMinFCost   = max_cost;
            ushort s_idCur_ICost;

            while(z < dZ)
            {
                s_idCur_ICost = s_idCur_ICost + z;

                if(s_idCur_ICost > NAPPEMAX)
                {
                    ReadInitCost<sens>(GT_Stream_ICost,ST_Bf_ICost,s_idCur_ICost,g_id_ICost);
                    GT_Stream_FCost  += sgn(NAPPEMAX);
                    s_idCur_ICost = z;
                }

                uint fCostMin           = max_cost;

                const ushort costInit   = ST_Bf_ICost[s_idCur_ICost];
                const ushort tZ         = z + tid;
                const short  Z          = index.x + tZ;

                GetConeZ(ConeZ,Z,penteMax,index,prevIndex);

                uint* prevFCost = S_FCost[sB] + Z - prevIndex.x;

                #pragma unroll
                for (int i = ConeZ.x; i < ConeZ.y; ++i)
                    fCostMin = min(fCostMin, costInit + *(prevFCost+i));

                const uint fcost                = fCostMin + sens * (GT_Stream_FCost[s_idCur_ICost] - costInit);
                S_FCost[!sB][tZ]                = fcost;
                GT_Stream_FCost[s_idCur_ICost]  = fcost;

                if(!sens) atomicMin(&globMinFCost,fcost);

                z += WARPSIZE;
            }

            if(!sens)
            {
                // retrancher le globMinFCost
            }

            prevIndex = index;

            segment--;
        }

        id_Line += WARPSIZE;
    }
}

template<class T> __device__
void Run(short2 *g_BuffIX,ushort *g_ICost)
{
    __shared__ short2 S_BuffIndex[WARPSIZE];
    __shared__ ushort S_BuffICost[NAPPEMAX];

    __shared__ uint   g_idIX;
    __shared__ uint   g_idICO;
    __shared__ ushort s_idICO;      


    RunLine<T,true>(g_BuffIX,g_ICost,S_BuffIndex,S_BuffICost,g_idIX,g_idICO,s_idICO);

    g_idIX -= WARPSIZE;
    g_idICO-= NAPPEMAX;

    RunLine<T,false>(g_BuffIX,g_ICost,S_BuffIndex,S_BuffICost,g_idIX,g_idICO,s_idICO);
}


#endif //_OPTIMISATION_KERNEL_Z_H_
