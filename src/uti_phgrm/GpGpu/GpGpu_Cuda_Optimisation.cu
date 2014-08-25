#ifndef _OPTIMISATION_KERNEL_H_
#define _OPTIMISATION_KERNEL_H_

/// \file       GpGpuInterfaceOptimisation.cu
/// \brief      Kernel optimisation
/// \author     GC
/// \version    0.01
/// \date       Avril 2013

#include "GpGpu/GpGpu_StreamData.cuh"
#include "GpGpu/SData2Optimize.h"

/// brief Calcul le Z min et max.
__device__ void ComputeIntervaleDelta(short2 & aDz, int aZ, int MaxDeltaZ, short2 aZ_Next, short2 aZ_Prev)
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

template<class T, class S, bool sens > __device__ void ReadOneSens(CDeviceDataStream<T> &costStream, uint lenghtLine, T pData[][NAPPEMAX], bool& idBuffer, T* gData, ushort penteMax, uint3 dimBlockTer)
{
    const ushort    tid     = threadIdx.x;

    for(int idParLine = 0; idParLine < lenghtLine;idParLine++)
    {
        const short2 uZ = costStream.read<sens>(pData[0],tid,0);
        short z = uZ.x;

        while( z < uZ.y )
        {
            int Z       = z + tid - uZ.x;
            if(Z < NAPPEMAX )
                gData[idParLine * dimBlockTer.z + Z]    = pData[0][Z];
            z          += min(uZ.y - z,WARPSIZE);
        }
    }
}

template<class T, class S,bool sens > __device__
void ScanOneSens(
        CDeviceDataStream<T> &costStream,
        uint    lenghtLine,
        T*      bCostInit,
        S       pData[][NAPPEMAX],
        bool&   idBuf,
        S*      g_ForceCostVol,
        ushort  penteMax,
        int&    pitStrOut )
{

    const ushort    tid     = threadIdx.x;
    short2          uZ_Prev = costStream.read<sens>(pData[idBuf],tid, 0);
    short           Z       = uZ_Prev.x + tid;
    __shared__ S    globMinCost;

    if(sens)
        while( Z < uZ_Prev.y )
        {
            if(Z>>8) break; // ERREUR DEPASSEMENT A SIMPLIFIER , DEPASSEMENT SUR RAMSES
            int idGData        = Z - uZ_Prev.x;
            g_ForceCostVol[idGData]    = pData[idBuf][idGData];
            Z += min(uZ_Prev.y - Z,WARPSIZE);
        }

    for(int idLine = 1; idLine < lenghtLine;idLine++)//#pragma unroll
    {

        short2 uZ_Next  = costStream.read<sens>(bCostInit,tid,0);
        ushort nbZ_Next = count(uZ_Next);

        pitStrOut += sens ? count(uZ_Prev) : -nbZ_Next;

        short2 aDz;

        S* g_LFCV = g_ForceCostVol + pitStrOut;

        short   Z = uZ_Next.x + tid;

        if(!tid) // test Superflu
            globMinCost = 1e9;

        short Z_Id  = tid;

        while( Z < uZ_Next.y )
        {

            ComputeIntervaleDelta(aDz,Z,penteMax,uZ_Next,uZ_Prev);
            S costMin           = 1e9;

            const S costInit    = bCostInit[Z_Id];

            const short Z_P_Id  = Z - uZ_Prev.x;

            // ATTENTION DEBUG REV 1383 RALENTISSEMENT
            //aDz.y = min(aDz.y,(short)NAPPEMAX - 1 - Z_P_Id);// bug sur Bouhdha -> plantage mais resultat correct
            aDz.y = min(aDz.y,(short)NAPPEMAX - Z_P_Id);// bug sur Bouhdha -> pas de plantage mais resultat faux
            //

            #pragma unroll
            for(short i = aDz.x ; i <= aDz.y; i++)
                costMin = min(costMin, costInit + pData[idBuf][Z_P_Id + i]);

            pData[!idBuf][Z_Id] = costMin;

            const S cost        = sens ? costMin : costMin + g_LFCV[Z_Id] - costInit;

            g_LFCV[Z_Id]        = cost;

            if(!sens)
                atomicMin(&globMinCost,cost);

            Z += min(uZ_Next.y - Z,WARPSIZE);

            if((Z_Id  = Z - uZ_Next.x)>>8) break;

        }

        if(!sens)
        {
            Z = uZ_Next.x + tid;
            short Z_Id  = tid;
            S* g_LFCV = g_ForceCostVol + pitStrOut;

            while( Z < uZ_Next.y )
            {
                g_LFCV[Z_Id] -= globMinCost;
                Z += min(uZ_Next.y - Z,WARPSIZE);
                if((Z_Id  = Z - uZ_Next.x)>>8) break;
            }
        }

        idBuf    = !idBuf;
        uZ_Prev     = uZ_Next;
    }

}


template<class T,class S> __global__ void kernelOptiOneDirection(T* g_StrCostVol, short2* g_StrId, S* g_ForceCostVol, uint3* g_RecStrParam, uint penteMax)
{
    __shared__ T        bufferData[WARPSIZE];
    __shared__ short2   bufferIndex[WARPSIZE];
    __shared__ T        bCostInit[NAPPEMAX];
    __shared__ S        pdata[2][NAPPEMAX];
    __shared__ uint     pit_Id;
    __shared__ uint     pit_Stream;
    __shared__ uint     sizeLine;

    int                 pitStrOut   = 0;
    bool                idBuf       = false;

    if(!threadIdx.x)
    {
        uint3 recStrParam   = g_RecStrParam[blockIdx.x];
        pit_Stream          = recStrParam.x;
        pit_Id              = recStrParam.y;
        sizeLine            = recStrParam.z;
    }

    __syncthreads();

    CDeviceDataStream<T> costStream(bufferData, g_StrCostVol + pit_Stream,bufferIndex, g_StrId + pit_Id, sizeLine * NAPPEMAX, sizeLine);

    ScanOneSens<T,S,eAVANT>   (costStream, sizeLine, bCostInit, pdata,idBuf,g_ForceCostVol + pit_Stream,penteMax, pitStrOut);
    ScanOneSens<T,S,eARRIERE> (costStream, sizeLine, bCostInit, pdata,idBuf,g_ForceCostVol + pit_Stream,penteMax, pitStrOut);

}

extern "C" void OptimisationOneDirection(Data2Optimiz<CuDeviceData3D> &d2O)
{
    uint deltaMax = 3;
    dim3 Threads(WARPSIZE,1,1);
    dim3 Blocks(d2O.NBlines(),1,1);
	
    kernelOptiOneDirection<ushort,uint><<<Blocks,Threads>>>
                                                (
                                                    d2O.pInitCost(),
                                                    d2O.pIndex(),
                                                    d2O.pForceCostVol(),
                                                    d2O.pParam(),
                                                    deltaMax
                                                    );
    getLastCudaError("kernelOptiOneDirection failed");
}

__global__ void TestGpu(uint *value)
{    
    uint id = blockIdx.x * WARPSIZE + threadIdx.x;
    //for(int i = 0 ; i < 4096; i++)
        //value[id] = sqrt((float)value[id]) * sqrt((float)value[id]) + sqrt((float)value[id]);
        value[id]++;
}

/// \brief Appel exterieur du kernel
extern "C" void Launch(uint *value){

    dim3 Threads(WARPSIZE);
    dim3 Blocks(1);

    TestGpu<<<Blocks,Threads>>>(value);

}

#endif
