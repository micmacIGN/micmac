#ifndef _OPTIMISATION_KERNEL_H_
/// \brief ....
#define _OPTIMISATION_KERNEL_H_

/// \file       GpGpuOptimisation.cu
/// \brief      Kernel optimisation
/// \author     GC
/// \version    0.01
/// \date       Avril 2013

#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_math.h>
#include <helper_cuda.h>
#include "GpGpu/GpGpuTools.h"
#include "GpGpu/helper_math_extented.cuh"

using namespace std;

/// \brief Tableau des penalites pre-calculees
#define PENALITE 7
#define WARPSIZE 32
#define NAPPEMAX 256

#define eAVANT      true
#define eARRIERE    false

static __constant__ float   penalite[PENALITE];
static __constant__ ushort  dMapIndex[WARPSIZE];

/// \brief  Fonction Gpu d optimisation
template<class T> __global__ void kernelOptiOneDirection(T* g_idata,T* g_odata,int* g_oPath, uint2 dimPlanCost, uint2 delta, float defaultValue)
{
    __shared__ T    sdata[32];

    const int   tid = threadIdx.x;
    const uint  pit = blockIdx.x * blockDim.x;
    uint        i0  = pit + tid;
    sdata[tid]      = g_idata[i0];
    bool        defV= sdata[tid] == defaultValue;
    g_odata[i0]     = defV ? 0 : sdata[tid];
    g_oPath[i0]     = tid;

    T minCost, cost;

    for(int l=1;l<dimPlanCost.y;l++)
    {
        uint        i1   = i0 + dimPlanCost.x;
        int         iL   = tid;

        if(i1<size(dimPlanCost))
        {
            cost = g_idata[i1];

            if(cost!=defaultValue)

                minCost = defV ? cost : cost + sdata[tid] + penalite[0];

            __syncthreads();

            if(cost!=defaultValue)
                for(int t = -((int)(delta.x)); t < ((int)(delta.y));t++)
                {
                    int Tl = tid + t;
                    if( t!=0 && Tl >= 0 && Tl < blockDim.x && sdata[Tl] != defaultValue)
                    {
                        T Cost = cost + sdata[Tl] + penalite[abs(t)];
                        if(Cost < minCost || defV)
                        {
                            minCost = Cost;
                            iL      = Tl;
                        }
                    }
                }

            else
                minCost = defV ? 0 : sdata[tid];

            i0 = l * dimPlanCost.x + pit + tid;

            g_odata[i0] = minCost;
            sdata[tid]  = minCost;
            defV        = minCost == defaultValue;
            g_oPath[i0] = iL;
        }
    }
}

/// brief Calcul le Z min et max.

__device__ void ComputeIntervaleDelta
(
        short2 & aDz,
        int aZ,
        int MaxDeltaZ,
        short2 aZ1,
        short2 aZ0
        )
{
    aDz.x =   aZ0.x-aZ;
    if (aZ != aZ1.x)
        aDz.x = max(aDz.x,-MaxDeltaZ);

    aDz.y = aZ0.y-1-aZ;
    if (aZ != aZ1.y-1)
        aDz.y = min(aDz.y,MaxDeltaZ);

    if (aDz.x > aDz.y)
        if (aDz.y <0)
            aDz.x = aDz.y;
        else
            aDz.y = aDz.x;
}

template< class T >
class CDeviceStream
{
public:

    __device__ CDeviceStream(T* buf,T* stream):
        _bufferData(buf),
        _streamData(stream),
        _curStreamId(0),
        _curBufferId(WARPSIZE)
    {}

    __device__ virtual short getLengthToRead(short2 &index,bool sens)
    {
        index = make_short2(0,0);
        return 1;
    }

    __device__ short2 read(T* destData, ushort tid, bool sens, T def, bool waitSync = true)
    {
        short2  index;
        ushort  NbCopied = 0 , NbTotalToCopy = getLengthToRead(index, sens);
        short   PitSens = !sens * WARPSIZE;

        while(NbCopied < NbTotalToCopy)
        {
            ushort NbToCopy = GetNbToCopy(NbTotalToCopy,NbCopied, sens);

            if(!NbToCopy)
            {
                _bufferData[threadIdx.x] = _streamData[_curStreamId + threadIdx.x - 2 * PitSens];
                _curBufferId   = PitSens;
                _curStreamId   = _curStreamId  + vec(sens) * WARPSIZE;
                NbToCopy = GetNbToCopy(NbTotalToCopy,NbCopied, sens);
                __syncthreads();
            }

            ushort idDest = tid + (sens ? NbCopied : NbTotalToCopy - NbCopied - NbToCopy);

            destData[idDest] = (tid < NbToCopy && !(idDest >= NbTotalToCopy)) ? _bufferData[_curBufferId + tid - !sens * NbToCopy] : def ;

            if(waitSync) __syncthreads();

            _curBufferId  = _curBufferId + vec(sens) * NbToCopy;
            NbCopied     += NbToCopy;
        }
       return index;
    }

private:

    __device__ short vec(bool sens)
    {
        return 1 - 2 * !sens;
    }

    __device__ short GetNbToCopy(ushort nTotal,ushort nCopied, bool sens)
    {
        return min(nTotal - nCopied , MaxReadBuffer(sens));
    }

    __device__ ushort MaxReadBuffer(bool sens)
    {
        return sens ? ((ushort)WARPSIZE - _curBufferId) : _curBufferId;
    }

    T*                          _bufferData;
    T*                          _streamData;
    uint                        _curStreamId;
    ushort                      _curBufferId;
};

template< class T >
class CDeviceDataStream : public CDeviceStream<T>
{
public:

    __device__ CDeviceDataStream(T* buf,T* stream,short2* bufId,short2* streamId, ushort startIndex = 0):
        CDeviceStream<T>(buf,stream + startIndex),
        _streamIndex(bufId,streamId),
        _startIndex(startIndex)
    {}

    __device__ short getLengthToRead(short2 &index, bool sens)
    {
        _streamIndex.read(&index,0,sens,make_short2(0,0),false);
        const short leng = diffYX(index) + 1;        
        return leng;
    }

    __device__ ushort getStartIndex()
    {
        return _startIndex;
    }

private:
    CDeviceStream<short2>       _streamIndex;
    ushort                      _startIndex;
};

template<class T> __device__ void ScanOneSens(CDeviceDataStream<T> &costStream, bool sens, uint lenghtLine, T pData[][NAPPEMAX], bool& idBuffer, T* gData, ushort penteMax)
{
    const ushort    tid     =   threadIdx.x;

    short2 uZ_Prev = costStream.read(pData[idBuffer],tid, sens,0);

    for(int idCurLine = 1; idCurLine < lenghtLine;idCurLine++)
    {
        const short2 uZ_Next = costStream.read(pData[2],tid,sens,0);

        short2 aDz;
        short z = uZ_Next.x;

        while( z < uZ_Next.y )
        {
            int Z = z + tid;

            if( Z < uZ_Next.y)
            {
                ComputeIntervaleDelta(aDz,Z,penteMax,uZ_Next,uZ_Prev);
                int costMin = 1e9;
                for(int i = aDz.x ; i < aDz.y; i++)
                    costMin = min(costMin,pData[2][Z - uZ_Next.x] + pData[idBuffer][Z - uZ_Prev.x+ i]);

                pData[!idBuffer][Z - uZ_Next.x] = costMin;
                gData[costStream.getStartIndex() + idCurLine * WARPSIZE + Z - uZ_Next.x] = costMin;
            }

            z += min(uZ_Next.y - z,WARPSIZE);
        }

        idBuffer = !idBuffer;
        uZ_Prev = uZ_Next;
    }

}

template<class T> __global__ void kernelOptiOneDirection(T* gStream, short2* gStreamId, T* g_odata, uint3 dimBlockTer, uint penteMax)
{
    __shared__ T        bufferData[WARPSIZE];
    __shared__ short2   bufferIndex[WARPSIZE];
    __shared__ T        pdata[3][NAPPEMAX];

    const int       pit     =   blockIdx.x * dimBlockTer.y;
    const int       pitStr  =   pit * dimBlockTer.z;
    bool            idBuf   =   false;

    CDeviceDataStream<T> costStream(bufferData, gStream,bufferIndex, gStreamId + pit,pitStr);

    ScanOneSens<T>(costStream,eAVANT,dimBlockTer.y, pdata,idBuf,g_odata,penteMax);
    ScanOneSens<T>(costStream,eARRIERE,dimBlockTer.y, pdata,idBuf,g_odata,penteMax);
}

/// \brief Lance le kernel d optimisation pour une direction
template <class T> void LaunchKernelOptOneDirection(CuHostData3D<T> &hInputStream, CuHostData3D<short2> &hInputindex, uint3 dimVolCost)
{

    int     nBLine      =   dimVolCost.x;
    uint    deltaMax    =   3;
    uint    dimDeltaMax =   deltaMax * 2 + 1;
    dim3    Threads(32,1,1);
    dim3    Blocks(nBLine,1,1);

    float   hPen[PENALITE];
    ushort  hMapIndex[WARPSIZE];

    for(int i=0 ; i < WARPSIZE; i++)
        hMapIndex[i] = i / dimDeltaMax;

    for(int i=0;i<PENALITE;i++)
        hPen[i] = ((float)(1 / 10.0f));

    //---------------------- Copie des penalites dans le device --------------------------------------- //
                                                                                                        //
    checkCudaErrors(cudaMemcpyToSymbol(penalite, hPen, sizeof(float)*PENALITE));                        //
    checkCudaErrors(cudaMemcpyToSymbol(dMapIndex, hMapIndex, sizeof(ushort)*WARPSIZE));                 //
                                                                                                        //
    //------------------------------------------------------------------------------------------------- //

    uint2   sizeInput   =   make_uint2(dimVolCost.x * dimVolCost.z,dimVolCost.y);
    uint2   sizeIndex   =   make_uint2(dimVolCost.y,dimVolCost.x);

    //---------------------------- Declaration des variables Host -------------------------------------- //

    CuHostData3D<T> hOutputValue(sizeInput,1);
    hOutputValue.SetName("hOutputValue");

    //----------------- Variables Device -------------------------------------------------------------- //

    CuDeviceData3D<T>       dInputStream(sizeInput,1,"dInputStream");
    CuDeviceData3D<short2>     dInputIndex(sizeIndex,1,"dInputIndex");
    CuDeviceData3D<T>       dOutputData(sizeInput,1,"dOutputData");

    //--------- Initialisation des Variables Device ---------------------------------------------------- //

    dOutputData.Memset(0); //???

    //------- Copie du volume de couts dans le device  ------------------------------------------------- //

    dInputStream.CopyHostToDevice(hInputStream.pData());
    dInputIndex.CopyHostToDevice(hInputindex.pData());

    //------------------------------------------------------------------------------------------------- //

    kernelOptiOneDirection<T><<<Blocks,Threads>>>(dInputStream.pData(),dInputIndex.pData(),dOutputData.pData(),dimVolCost,deltaMax);
    getLastCudaError("kernelOptiOneDirection failed");

    //------------------------------------------------------------------------------------------------- //

    dOutputData.CopyDevicetoHost(hOutputValue.pData());
    //cudaDeviceSynchronize();
    hOutputValue.OutputValues(0,XY,NEGARECT,3,-1);
    dInputStream.Dealloc();
    dOutputData.Dealloc();
}

/// \brief Appel exterieur du kernel d optimisation
extern "C" void OptimisationOneDirection(CuHostData3D<float> &data, uint3 dimVolCost, float defaultValue)
{
    //LaunchKernelOptOneDirection(data,dimVolCost,defaultValue);
}

/// \brief Appel exterieur du kernel
extern "C" void Launch()
{
    uint3 dimVolCost  = make_uint3(1,10,32);

    CuHostData3D<int>       streamCost(make_uint2(dimVolCost.x * dimVolCost.z,dimVolCost.y));
    CuHostData3D<short2>    streamIndex(make_uint2(dimVolCost.y,dimVolCost.x));

    streamCost.SetName("streamCost");
    streamIndex.SetName("streamIndex");

    uint si = 0 , sizeStreamCost = 0;

    srand (time(NULL));

    for(int i = 0 ; i < dimVolCost.x ; i++)
    {
        int pit         = i * dimVolCost.y;
        int pitLine     = pit * dimVolCost.z;

        while (si < dimVolCost.y){

            int min                         =  -CData<int>::GetRandomValue(5,16);
            int max                         =   CData<int>::GetRandomValue(5,16);
            int dim                         =   max - min + 1;            
            streamIndex[pit + si]           =   make_short2(min,max);

            for(int i = 0 ; i < dim; i++)
                streamCost[pitLine + sizeStreamCost+i] =  CData<int>::GetRandomValue(16,128);

            si++;
            sizeStreamCost += dim;

        }
    }
    //streamCost.OutputValues();
    LaunchKernelOptOneDirection(streamCost,streamIndex,dimVolCost);

    streamCost.Dealloc();
    streamIndex.Dealloc();
}
#endif
