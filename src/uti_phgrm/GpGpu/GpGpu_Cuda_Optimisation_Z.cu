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

template<bool sens> __device__
void RunLine(   SimpleStream<short2>    &streamIndex,
                SimpleStream<uint>      &streamFCost,
                SimpleStream<ushort>    &streamICost,
                short2     *S_Bf_Index,
                ushort     *ST_Bf_ICost,
                uint       *S_FCost[2],
                ushort     &sId_ICost,
                uint        penteMax,
                uint        lenghtLine,
                short2     &prevIndex,
                int        &id_Line,
                ushort     &idSeg,
                bool       &idBuf)
{
    const ushort  tid   = threadIdx.x;
    short2* ST_Bf_Index = S_Bf_Index + sgn(tid);
    short2  ConeZ;
    uint    segLine     = 0;

    __shared__ uint globMinFCost;

    while(id_Line < lenghtLine)
    {

        uint  segLine = min(lenghtLine-id_Line,WARPSIZE);

        while(idSeg < segLine)
        {

            const short2 index  = S_Bf_Index[idSeg];
            const ushort dZ     = count(index); // creer buffer de count
            ushort       z      = 0;
            globMinFCost        = max_cost;

            while( z < dZ)
            {           
                if(sId_ICost > NAPPEMAX)
                {
                    streamICost.read<sens>(ST_Bf_ICost);
                    streamFCost.incre<sens>();
                    sId_ICost = 0;
                }

                uint fCostMin           = max_cost;
                const ushort costInit   = ST_Bf_ICost[sId_ICost];
                const ushort tZ         = z + tid;
                const short  Z          = index.x + tZ;
                const short prZ         = Z - prevIndex.x;

                GetConeZ(ConeZ,Z,penteMax,index,prevIndex);

                uint* prevFCost = S_FCost[idBuf] + prZ;

                ConeZ.y = min(NAPPEMAX - prZ,ConeZ.y );

                for (int i = ConeZ.x; i <= ConeZ.y; ++i)
                        fCostMin = min(fCostMin, costInit + prevFCost[i]);

                const uint fcost    =  fCostMin;// + sens * (streamFCost.GetValue(s_idCur_ICost) - costInit);

                if( tZ < NAPPEMAX)
                {
                    S_FCost[!idBuf][tZ] = fcost;
                    streamFCost.SetValue(sId_ICost, fcost);

                    if(!sens)
                        atomicMin(&globMinFCost,fcost);
                }

                const ushort pIdCost = sId_ICost;
                sId_ICost += min(dZ - z,WARPSIZE);
                z         += min(WARPSIZE,NAPPEMAX-pIdCost);
            }

            prevIndex = index;
            idSeg++;
            idBuf =!idBuf;
        }

        streamIndex.read<sens>(ST_Bf_Index);
        id_Line += segLine;
        idSeg   = 0;
    }

    idSeg = segLine - 1;
}

template<class T> __global__
void Run(ushort* g_ICost, short2* g_Index, uint* g_FCost, uint3* g_RecStrParam, uint penteMax)
{
    __shared__ short2   S_BuffIndex[WARPSIZE];
    __shared__ ushort   S_BuffICost0[NAPPEMAX + 2*WARPSIZE];
    __shared__ uint     S_BuffFCost0[NAPPEMAX + 2*WARPSIZE];
    __shared__ uint     S_BuffFCost1[NAPPEMAX + 2*WARPSIZE];
    __shared__ uint     pit_Id;
    __shared__ uint     pit_Stream;
    __shared__ uint     lenghtLine;

    const ushort    tid     = threadIdx.x;

    uint*    S_BuffFCost[2] = {S_BuffFCost0 + WARPSIZE,S_BuffFCost1 + WARPSIZE};
    ushort*  S_BuffICost    = S_BuffICost0 + WARPSIZE + tid;

    bool            idBuf   = false;
    ushort          s_id_Icost;

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

    streamICost.read<eAVANT>(S_BuffICost);

    uint*   locFCost = S_BuffFCost[idBuf] + tid;
    for (ushort i = 0; i < NAPPEMAX; i+=WARPSIZE)
        locFCost[i] = S_BuffICost[i];

    streamIndex.read<eAVANT>(S_BuffIndex + tid);

    short2  prevIndex   = S_BuffIndex[0];
    int     id_Line     = 0;
    ushort  idSeg       = 1;

    s_id_Icost   = count(prevIndex);

    RunLine<eAVANT>(streamIndex,streamFCost,streamICost,S_BuffIndex,S_BuffICost,S_BuffFCost,s_id_Icost,penteMax,lenghtLine,prevIndex,id_Line,idSeg,idBuf);

//    streamFCost.reverse<eARRIERE>();
//    streamIndex.reverse<eARRIERE>();
//    streamICost.reverse<eARRIERE>();

//    S_BuffFCost[0]  += NAPPEMAX;
//    S_BuffFCost[1]  += NAPPEMAX;
//    S_BuffICost     += NAPPEMAX - 2*tid;

//    streamICost.readFrom<eARRIERE>(S_BuffFCost[idBuf] - tid, s_id_Icost - NAPPEMAX);

//    streamICost.incre<eARRIERE>();
//    streamFCost.incre<eARRIERE>();
//    streamIndex.incre<eARRIERE>();

//    prevIndex       = S_BuffIndex[idSeg];
//    idSeg           = WARPSIZE - idSeg;
//    id_Line         = -idSeg;

//    const short nonRead   = count(prevIndex) - s_id_Icost;

//    if(count(prevIndex) < s_id_Icost)
//        s_id_Icost = NAPPEMAX + nonRead;
//    else
//    {
//        streamICost.read<eARRIERE>(S_BuffICost);
//        streamFCost.incre<eARRIERE>();
//        s_id_Icost = nonRead;
//    }

//    RunLine<eARRIERE>(  streamIndex,
//                        streamFCost,
//                        streamICost,
//                        S_BuffIndex + WARPSIZE,
//                        S_BuffICost - tid,
//                        S_BuffFCost,
//                        s_id_Icost,
//                        penteMax,
//                        lenghtLine,
//                        prevIndex,
//                        id_Line,
//                        idSeg,
//                        idBuf);

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

