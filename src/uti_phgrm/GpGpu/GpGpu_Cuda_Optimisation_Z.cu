#ifndef _OPTIMISATION_KERNEL_Z_H_
#define _OPTIMISATION_KERNEL_Z_H_

#include "GpGpu/GpGpu_StreamData.cuh"
#include "GpGpu/SData2Optimize.h"

// On pourrait imaginer un buffer des tailles calculer en parallel
// SIZEBUFFER[threadIdx.x] = count(lI[threadIdx.x]);


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
void RunLine(SimpleStream<short2> &streamIndex, SimpleStream<uint> streamFCost, SimpleStream<ushort> &streamICost,short2* S_Bf_Index,ushort *ST_Bf_ICost, uint S_FCost[][NAPPEMAX + WARPSIZE], uint penteMax, uint lenghtLine,bool &idBuf)
{
    const ushort  tid       = threadIdx.x;
    short2* ST_Bf_Index     = S_Bf_Index + tid;

    __shared__ uint globMinFCost;

    short2 ConeZ;
    short2 prevIndex;

    streamICost.read<sens>(ST_Bf_ICost);

    for (ushort i = 0; i < NAPPEMAX; i+=WARPSIZE)
        S_FCost[idBuf][i +tid] = ST_Bf_ICost[i];

    streamIndex.read<sens>(ST_Bf_Index);
    ushort sId_ICost = count(S_Bf_Index[0]);

    uint  id_Line = 1;

    while(id_Line < lenghtLine)
    {

        const uint  segLine = min(lenghtLine-id_Line,WARPSIZE);
        ushort      idSeg   = 0;

        while(idSeg < segLine)
        {

            const short2 index  = S_Bf_Index[idSeg];
            const ushort dZ     = count(index); // creer buffer de count pre calculer en Multi threading lors de l'aquisition des index

            ushort       z      = 0;
            globMinFCost        = max_cost;

            while( z < dZ)
            {           

                if(sId_ICost > NAPPEMAX)
                {
                    if(z + NAPPEMAX < dZ )
                    {                 
                        streamICost.read<sens>(ST_Bf_ICost); /// ERREUR DE DEPASSEMENT!!!
                        streamFCost.incre<sens>();
                    }
                    sId_ICost = 0;
                }

                uint fCostMin           = max_cost;
                const ushort costInit   = ST_Bf_ICost[sId_ICost];
                const ushort tZ         = z + tid;
                const short  Z          = index.x + tZ;

                GetConeZ(ConeZ,Z,penteMax,index,prevIndex);

                uint* prevFCost = S_FCost[idBuf] + Z - prevIndex.x;

                #pragma unroll
                for (int i = ConeZ.x; i < ConeZ.y; ++i)
                    fCostMin = min(fCostMin, costInit + *(prevFCost+i));

                const uint fcost    =  fCostMin;// + sens * (streamFCost.GetValue(s_idCur_ICost) - costInit);

                S_FCost[!idBuf][tZ] = fcost;

                if(tZ < dZ)
                streamFCost.SetValue(sId_ICost, fcost);

                if(!sens)
                    atomicMin(&globMinFCost,fcost);

                z         += WARPSIZE;
                sId_ICost += WARPSIZE;
            }

            prevIndex = index;
            idSeg++;
            idBuf =!idBuf;
        }

        streamIndex.read<sens>(ST_Bf_Index);
        id_Line += segLine;
    }

 //   if(blockIdx.x == 35 && !tid)
//        printf(" Count : %d/%d", counter,compareCount);
}

template<class T> __global__
void Run(ushort* g_ICost, short2* g_Index, uint* g_FCost, uint3* g_RecStrParam, uint penteMax)
{

    __shared__ short2   S_BuffIndex[WARPSIZE];
    __shared__ ushort   S_BuffICost[NAPPEMAX + WARPSIZE];
    __shared__ uint     S_BuffFCost[2][NAPPEMAX + WARPSIZE];
    __shared__ uint     pit_Id;
    __shared__ uint     pit_Stream;
    __shared__ uint     lenghtLine;

    bool                idBuf       = false;

    if(!threadIdx.x)
    {
        uint3 recStrParam   = g_RecStrParam[blockIdx.x];
        pit_Stream          = recStrParam.x;
        pit_Id              = recStrParam.y;
        lenghtLine          = recStrParam.z;
    }

    __syncthreads();

    SimpleStream<ushort>    streamICost(g_ICost + pit_Stream,NAPPEMAX);
    SimpleStream<uint>      streamFCost(g_FCost + pit_Stream,NAPPEMAX);
    SimpleStream<short2>    streamIndex(g_Index + pit_Id    ,WARPSIZE);

    RunLine<T,true>(streamIndex,streamFCost,streamICost,S_BuffIndex,S_BuffICost + threadIdx.x,S_BuffFCost,penteMax,lenghtLine,idBuf);
    //RunLine<T,true>(streamIndex,streamFCost,streamICost,S_BuffIndex,S_BuffICost,S_BuffFCost,penteMax,lenghtLine,idBuf);

//    g_idIX -= WARPSIZE;
//    g_idICO-= NAPPEMAX;

   // RunLine<T,false>(streamIndex,streamFCost,streamICost,S_BuffIndex,S_BuffICost,S_BuffFCost,penteMax,lenghtLine,idBuf);
}

extern "C" void OptimisationOneDirectionZ(Data2Optimiz<CuDeviceData3D> &d2O)
{
    uint deltaMax = 3;
    dim3 Threads(WARPSIZE,1,1);
    dim3 Blocks(d2O.NBlines(),1,1);

    Run< uint ><<<Blocks,Threads>>>
                                    (
                                        d2O.pInitCost(),
                                        d2O.pIndex(),
                                        d2O.pForceCostVol(),
                                        d2O.pParam(),
                                        deltaMax
                                        );
    getLastCudaError("kernelOptiOneDirection failed");
}

#endif //_OPTIMISATION_KERNEL_Z_H_

