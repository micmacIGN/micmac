#ifndef _OPTIMISATION_KERNEL_H_
/// \brief ....
#define _OPTIMISATION_KERNEL_H_

/// \file       GpGpuOptimisation.cu
/// \brief      Kernel optimisation
/// \author     GC
/// \version    0.01
/// \date       Avril 2013

#include "GpGpu/GpGpuStreamData.cuh"
//#include "GpGpu/GpGpuOptimisation.h"
#include "GpGpu/data2Optimize.h"

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

template<class T, bool sens > __device__ void ReadOneSens(CDeviceDataStream<T> &costStream, uint lenghtLine, T pData[][NAPPEMAX], bool& idBuffer, T* gData, ushort penteMax, uint3 dimBlockTer)
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

template<class T, bool sens > __device__ void ScanOneSens(CDeviceDataStream<T> &costStream, uint lenghtLine, T pData[][NAPPEMAX], bool& idBuf, T* g_ForceCostVol, ushort penteMax, int& pitStrOut )
{
    const ushort    tid     = threadIdx.x;
    short2          uZ_Prev = costStream.read<sens>(pData[idBuf],tid, 0);
    short           Z       = uZ_Prev.x + tid;
    __shared__ T    globMinCost;

    if(sens)
        while( Z < uZ_Prev.y )
        {
            int idGData        = Z - uZ_Prev.x;
            g_ForceCostVol[idGData]    = pData[idBuf][idGData];
            Z += min(uZ_Prev.y - Z,WARPSIZE);
        }

    for(int idLine = 1; idLine < lenghtLine;idLine++)//#pragma unroll
    {

        short2 uZ_Next  = costStream.read<sens>(pData[2],tid,0);
        ushort nbZ_Next = count(uZ_Next);

        pitStrOut += sens ? count(uZ_Prev) : -nbZ_Next;

        short2 aDz;

        T* g_LFCV = g_ForceCostVol + pitStrOut;

        short   Z = uZ_Next.x + tid;
        if(!tid) globMinCost = 1e9;

        short Z_Id  = tid;

        while( Z < uZ_Next.y )
        {

            ComputeIntervaleDelta(aDz,Z,penteMax,uZ_Next,uZ_Prev);
            T costMin           = 1e9;
            const T costInit    = pData[2][Z_Id];
            const short Z_P_Id  = Z - uZ_Prev.x;

            aDz.y = min(aDz.y,(short)NAPPEMAX - Z_P_Id);

            for(short i = aDz.x ; i <= aDz.y; i++)
                costMin = min(costMin, costInit + pData[idBuf][Z_P_Id + i]);

            pData[!idBuf][Z_Id] = costMin;

            const T cost        = sens ? costMin : costMin + g_LFCV[Z_Id] - costInit;

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
            T* g_LFCV = g_ForceCostVol + pitStrOut;

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

template<class T> __global__ void kernelOptiOneDirection(T* g_StrCostVol, short2* g_StrId, T* g_ForceCostVol, uint3* g_RecStrParam, uint penteMax)
{
    __shared__ T        bufferData[WARPSIZE];
    __shared__ short2   bufferIndex[WARPSIZE];
    __shared__ T        pdata[3][NAPPEMAX];
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

    ScanOneSens<T,eAVANT>   (costStream, sizeLine, pdata,idBuf,g_ForceCostVol + pit_Stream,penteMax, pitStrOut);
    ScanOneSens<T,eARRIERE> (costStream, sizeLine, pdata,idBuf,g_ForceCostVol + pit_Stream,penteMax, pitStrOut);

}

extern "C" void OptimisationOneDirection(Data2Optimiz<CuHostData3D> &d2O)
{
    uint deltaMax = 3;
    dim3 Threads(WARPSIZE,1,1);
    dim3 Blocks(d2O._nbLines,1,1);

    kernelOptiOneDirection<uint><<<Blocks,Threads>>>
                                                (
                                                    d2O._s_InitCostVol  .pData(),
                                                    d2O._s_Index        .pData(),
                                                    d2O._s_ForceCostVol .pData(),
                                                    d2O._param[0]       .pData(),
                                                    deltaMax
                                                    );
}

/// \brief Appel exterieur du kernel
extern "C" void Launch(){}

#endif
