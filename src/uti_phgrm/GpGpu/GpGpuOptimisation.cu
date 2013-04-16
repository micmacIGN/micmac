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

#define eAVANT      1
#define eARRIERE   -1

static __constant__ float   penalite[PENALITE];
static __constant__ ushort  dMapIndex[WARPSIZE];

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type

/// \struct SharedMemory
/// \brief  Structure de donnees partagees pour un block.
///         Allocation dynamique de la memoire lors du lancement du kernel
template<class T>
struct SharedMemory
{
    /// \brief ...
    __device__ inline operator       T *()
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }

    /// \brief ...
    __device__ inline operator const T *() const
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }
};

/// \brief Opere une reduction d un tableau en Cpu
template<class T>
T reduceCPU(T *data, int size)
{
    T sum = data[0];
    T c = (T)0.0;

    for (int i = 1; i < size; i++)
    {
        T y = data[i] - c;
        T t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }

    return sum;
}

/// \brief Opere une reduction d un tableau en Gpu
template<class T> __global__ void kernelReduction(T* g_idata,T* g_odata,  int n)
{

    T *sdata = SharedMemory<T>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;

    T mySum = (i < n) ? g_idata[i] : 0;

    if (i + blockDim.x < n)
        mySum += g_idata[i+blockDim.x];

    sdata[tid] = mySum;
    __syncthreads();

    // do reduction in shared mem
    for (unsigned int s=blockDim.x/2; s>0; s>>=1)
    {
        if (tid < s)
        {
            sdata[tid] = mySum = mySum + sdata[tid + s];
        }

        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

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
        int & aDzMin,
        int & aDzMax,
        int aZ,
        int MaxDeltaZ,
        int aZ1Min,
        int aZ1Max,
        int aZ0Min,
        int aZ0Max
        )
{
    aDzMin =   aZ0Min-aZ;
    if (aZ != aZ1Min)
        aDzMin = max(aDzMin,-MaxDeltaZ);

    aDzMax = aZ0Max-1-aZ;
    if (aZ != aZ1Max-1)
        aDzMax = min(aDzMax,MaxDeltaZ);

    if (aDzMin > aDzMax)
        if (aDzMax <0)
            aDzMin = aDzMax;
        else
            aDzMax = aDzMin;
}


template< class T >
class CDeviceStream

{
public:

    __device__ CDeviceStream(T* buf,T* stream):
        _bufferData(buf),
        _streamData(stream),
        _curSteamId(0),
        _curBuffeId(WARPSIZE)
    {}

    __device__ virtual short getLengthRead(short2 &index)
    {
        index = make_short2(0,0);
        return 1;
    }

    __device__ short2 read(T* destData, ushort tid, short sens, T def, bool waitSync = true)
    {
        short2  index;
        ushort  NbCopied = 0 , NbTotalToCopy = getLengthRead(index);

        while(NbCopied < NbTotalToCopy)
        {
            ushort NbToCopy = min(NbTotalToCopy - NbCopied , WARPSIZE - _curBuffeId);

            if(NbToCopy == 0)
            {
                _bufferData[threadIdx.x] = _streamData[_curSteamId + threadIdx.x];
                _curBuffeId = 0;
                /*if (!tid)*/
                _curSteamId  += WARPSIZE;
                NbToCopy = min(NbTotalToCopy - NbCopied ,WARPSIZE);
                __syncthreads();
            }

            if (tid < NbToCopy)
                destData[NbCopied + tid] =  _bufferData[_curBuffeId + tid];
            else
                destData[NbCopied + tid] =  def;

            if(waitSync)
                __syncthreads();

            _curBuffeId += NbToCopy;
            NbCopied    += NbToCopy;
        }
       return index;
    }

private:

    T*                          _bufferData;
    T*                          _streamData;
    uint                        _curSteamId;
    ushort                      _curBuffeId;

};

template< class T >
class CDeviceDataStream : public CDeviceStream<T>
{
public:

    __device__ CDeviceDataStream(T* buf,T* stream,short2* bufId,short2* streamId):
        CDeviceStream<T>(buf,stream),
        _streamIndex(bufId,streamId)
    {}

    __device__ short getLengthRead(short2 &index)
    {
        _streamIndex.read(&index,0,0,make_short2(0,0),false);
        const short leng = diffYX(index) + 1;
        if(threadIdx.x ==0)printf("longueur %d\n",leng);
        return leng;
    }

private:
    CDeviceStream<short2>     _streamIndex;
};


template<class T> __device__ short2 readStream(T* destData,T* bufferData, short2* bufferIndex, T* streamData, short2* streamIndex, int tid, int& bufIdId, int& bufDaId, int& idCel, int& idStm, short sens)

{

    bufIdId ++;
    ushort elCopied = 0;

    if(bufIdId >= WARPSIZE)
    {
        int pit = idCel;
        bufferIndex[tid] = streamIndex[pit + tid];
        bufIdId = 0;
        __syncthreads();
    }

    const short2 Z      = bufferIndex[bufIdId];
    const ushort dimZ   = diffYX(Z);

    while(elCopied < dimZ)
    {
        ushort elToCopy = min(dimZ - elCopied , WARPSIZE - bufDaId);

        if(elToCopy == 0)
        {
            bufferData[tid] = streamData[idStm + tid];
            bufDaId = 0;

            if (!tid) idStm  += WARPSIZE;

            elToCopy = min(dimZ - elCopied ,WARPSIZE);
            __syncthreads();
        }

        destData[elCopied + tid] = (tid <= elToCopy) ? bufferData[bufDaId + tid] : -1;

        __syncthreads();
        bufDaId += elToCopy;
        elCopied  += elToCopy;
    }

    idCel++;

    return Z;
}

template<class T> __global__ void kernelOptiOneDirection2(T* gInputStream, short2* gInputIndex, T* g_odata, uint3 dimBlockTer, uint penteMax )

{

    __shared__ T        bufferData[WARPSIZE];
    __shared__ short2   bufferindex[WARPSIZE];
    __shared__ T        pdata[3][NAPPEMAX];

    int             idStm   =   0;
    const ushort    tid     =   threadIdx.x;
    int             bufIdId =   WARPSIZE;
    int             bufDaId =   WARPSIZE;
    int             idCel   =   0;
    bool            idBuf   =   0;
    const int       pit     =   blockIdx.x * dimBlockTer.y;
    const int       pitStr  =   pit * dimBlockTer.z;
    const int       pitId   =   pit * 2;

    short2 uZ_P = readStream(pdata[idBuf], bufferData , bufferindex , gInputStream + pitStr, gInputIndex + pitId, tid,bufIdId, bufDaId, idCel, idStm,1);

    g_odata[pitStr + tid] = pdata[idBuf][tid]; // ATTENTION  Faible bande passante

    for(int l=1;l<dimBlockTer.y;l++)
    {
        const short2 uZ_N = readStream(pdata[2],bufferData,bufferindex , gInputStream + pitStr, gInputIndex + pitId, tid,bufIdId, bufDaId, idCel, idStm,1);

        int aDzMin,aDzMax;
        short z = uZ_N.x;

        while( z < uZ_N.y )
        {
            int Z = z + tid;

            if( Z < uZ_N.y)
            {
                ComputeIntervaleDelta(aDzMin,aDzMax,Z,penteMax,uZ_N.x,uZ_N.y,uZ_P.x,uZ_P.y);
                int costMin = 1e9;
                for(int i = aDzMin ; i < aDzMax; i++)
                    costMin = min(costMin,pdata[2][Z - uZ_N.x] + pdata[idBuf][Z - uZ_P.x+ i]);

                pdata[!idBuf][Z - uZ_N.x]           = costMin;
                g_odata[pitStr + l*WARPSIZE + Z - uZ_N.x]    = costMin; // ATTENTION  Faible bande passante
            }

            z += min(uZ_N.y - z,WARPSIZE);
        }

        idBuf = !idBuf;
        uZ_P = uZ_N;
    }

}

template<class T> __global__ void kernelOptiOneDirection3(T* gStream, short2* gStreamId, T* g_odata, uint3 dimBlockTer, uint penteMax)
{

    __shared__ T        bufferData[WARPSIZE];
    __shared__ short2   bufferIndex[WARPSIZE];
    __shared__ T        pdata[3][NAPPEMAX];

    const ushort    tid     =   threadIdx.x;
    const int       pit     =   blockIdx.x * dimBlockTer.y;
    const int       pitStr  =   pit * dimBlockTer.z;
    bool            idBuf   =   false;

    CDeviceDataStream<T> costStream(bufferData, gStream + pitStr,bufferIndex, gStreamId + pit);

    short2 uZ_P = costStream.read(pdata[idBuf],tid, eAVANT,0);

    g_odata[tid] = pdata[idBuf][tid];

    __syncthreads();

    costStream.read(pdata[idBuf],tid,eAVANT,0);

    g_odata[WARPSIZE + tid] = pdata[idBuf][tid];

    __syncthreads();

    costStream.read(pdata[idBuf],tid,eAVANT,0);

    g_odata[WARPSIZE * 2+ tid] = pdata[idBuf][tid];

/*
    for(int l=1;l<dimBlockTer.y;l++)
    {
        const short2 uZ_N = costStream.read(pdata[2],tid,eAVANT);

        int aDzMin,aDzMax;
        short z = uZ_N.x;

        while( z < uZ_N.y )
        {
            int Z = z + tid;

            if( Z < uZ_N.y)
            {
                ComputeIntervaleDelta(aDzMin,aDzMax,Z,penteMax,uZ_N.x,uZ_N.y,uZ_P.x,uZ_P.y);
                int costMin = 1e9;
                for(int i = aDzMin ; i < aDzMax; i++)
                    costMin = min(costMin,pdata[2][Z - uZ_N.x] + pdata[idBuf][Z - uZ_P.x+ i]);

                pdata[!idBuf][Z - uZ_N.x]           = costMin;
                g_odata[pitStr + l*WARPSIZE + Z - uZ_N.x]    = costMin; // ATTENTION  Faible bande passante
            }

            z += min(uZ_N.y - z,WARPSIZE);
        }

        idBuf = !idBuf;
        uZ_P = uZ_N;
    }
*/
}

/// \brief Lance le kernel d optimisation pour une direction
template <class T> void LaunchKernelOptOneDirection2(CuHostData3D<T> &hInputStream, CuHostData3D<short2> &hInputindex, uint3 dimVolCost,float defaultValue, int sizeVolumeCost)
{

    int     nBLine      =   dimVolCost.x;
    int     si          =   dimVolCost.z * nBLine;
    int     dimLine     =   dimVolCost.y;
    uint2   diPlanCost  =   make_uint2(si,dimLine);
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

    //-------- Copie des penalites dans le device ----------

    checkCudaErrors(cudaMemcpyToSymbol(penalite, hPen, sizeof(float)*PENALITE));
    checkCudaErrors(cudaMemcpyToSymbol(dMapIndex, hMapIndex, sizeof(ushort)*WARPSIZE));

    //------------------------------------------------------

    uint2   sizeInput   =   make_uint2(dimVolCost.x * dimVolCost.z,dimVolCost.y);
    uint2   sizeIndex   =   make_uint2(dimVolCost.y,dimVolCost.x);

    //----------- Declaration des variables Host -----------

    CuHostData3D<T>         hOutputValue(sizeInput,1);
    hOutputValue.SetName("hOutputValue");

    //----------------- Variables Device -------------------

    CuDeviceData3D<T>       dInputStream(sizeInput,1,"dInputStream");
    CuDeviceData3D<short2>     dInputIndex(sizeIndex,1,"dInputIndex");
    CuDeviceData3D<T>       dOutputData(sizeInput,1,"dOutputData");

    //--------- Initialisation des Variables Device ---------

    dOutputData.Memset(0); //???

    //------- Copie du volume de couts dans le device  -------

    dInputStream.CopyHostToDevice(hInputStream.pData());
    dInputIndex.CopyHostToDevice(hInputindex.pData());

    kernelOptiOneDirection3<T><<<Blocks,Threads>>>(dInputStream.pData(),dInputIndex.pData(),dOutputData.pData(),dimVolCost,deltaMax);
    getLastCudaError("kernelOptiOneDirection failed");

    dOutputData.CopyDevicetoHost(hOutputValue.pData());
    cudaDeviceSynchronize();
    hOutputValue.OutputValues(0,XY,NEGARECT,3,-1);
//    hInputindex.OutputValues();

    dInputStream.Dealloc();
    dOutputData.Dealloc();

}


/// \brief Lance le kernel d optimisation pour une direction

template <class T> void LaunchKernelOptOneDirection(CuHostData3D<T> &hInputValue, uint3 dimVolCost,float defaultValue = 0)
{
    //nZ      = 32 doit etre en puissance de 2
    int     nBLine      =   dimVolCost.x;
    uint2   dimTer      =   make_uint2(dimVolCost.x,dimVolCost.y);
    int     si          =   dimVolCost.z * nBLine;
    int     dimLine     =   dimVolCost.y;
    uint2   diPlanCost  =   make_uint2(si,dimLine);
    uint2   delta       =   make_uint2(5);
    dim3    Threads(dimVolCost.z,1,1);
    dim3    Blocks(nBLine,1,1);

    float hPen[PENALITE];

    for(int i=0;i<PENALITE;i++)
        hPen[i] = ((float)(1 / 10.0f));

    //-------- Copie des penalites dans le device ----------

    checkCudaErrors(cudaMemcpyToSymbol(penalite, hPen, sizeof(float)*PENALITE));

    //----------- Declaration des variables Host -----------

    CuHostData3D<T>         hOutputValue(diPlanCost);
    CuHostData3D<int>       hPath(diPlanCost);
    CuHostData3D<float>     hMinCostId(dimTer);

    //----------------- Variables Device -------------------

    CuDeviceData3D<T>       dInputData(diPlanCost,1,"dInputData");
    CuDeviceData3D<T>       dOutputData(diPlanCost,1,"dOutputData");
    CuDeviceData3D<int>     dPath(diPlanCost,1,"dPath");
    CuDeviceData3D<float>   dMinCostId(make_uint2(dimVolCost.x,1),1,"minCostId");

    //--------- Initialisation des Variables Device ---------

    dOutputData.Memset(0);
    dPath.Memset(0);
    dMinCostId.Memset(0);

    //------- Copie du volume de couts dans le device  -------

    dInputData.CopyHostToDevice(hInputValue.pData());

    kernelOptiOneDirection<T><<<Blocks,Threads>>>(dInputData.pData(),dOutputData.pData(),dPath.pData(),diPlanCost, delta,/*dMinCostId.pData(),*/defaultValue);
    getLastCudaError("kernelOptimisation failed");

    dOutputData.CopyDevicetoHost(hOutputValue.pData());
    dPath.CopyDevicetoHost(hPath.pData());
    dMinCostId.CopyDevicetoHost(hMinCostId.pData());

/*

    uint2   ptTer;
    uint2   prev = make_uint2(0,1);
    for ( ptTer.x = 0; ptTer.x < dimTer.x; ptTer.x++)
        for(ptTer.y = 1; ptTer.y < dimTer.y ; ptTer.y++)
        {
            uint2 pt = make_uint2(ptTer.x * dimVolCost.z + (uint)hMinCostId[ptTer - prev],ptTer.y);
            hMinCostId[ptTer] =  (float)hPath[pt];
        }
    for (ptTer.x = 0; ptTer.x < dimTer.x; ptTer.x++)
        for(ptTer.y = 0; ptTer.y < dimTer.y ; ptTer.y++)
            if (defaultValue == hInputValue[ptTer])
                hMinCostId[ptTer] = 0.0f;
    hMinCostId.OutputValues();
    hInputValue.OutputValues(0,XY,Rect(0,0,32,dimVolCost.y));
    hPath.OutputValues(0,XY,Rect(0,0,dimVolCost.z,dimVolCost.y));
    hOutputValue.OutputValues(0,XY,Rect(0,0,dimVolCost.z,dimVolCost.y),4);
    GpGpuTools::Array1DtoImageFile(GpGpuTools::MultArray(hMinCostId.pData(),dimTer,1.0f/32.0f),"ZMap.pgm",dimTer);

*/

    hOutputValue.Dealloc();
    hPath.Dealloc();
    hMinCostId.Dealloc();
    dInputData.Dealloc();
    dOutputData.Dealloc();
    dPath.Dealloc();
    dMinCostId.Dealloc();

}

/// \brief Appel exterieur du kernel d optimisation
extern "C" void OptimisationOneDirection(CuHostData3D<float> &data, uint3 dimVolCost, float defaultValue)
{
    LaunchKernelOptOneDirection(data,dimVolCost,defaultValue);
}

/// \brief Appel exterieur du kernel
extern "C" void Launch()
{
    uint3 dimVolCost  = make_uint3(1,3,32);

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

            int min                         =  -CData<int>::GetRandomValue(10,16);
            int max                         =   CData<int>::GetRandomValue(10,16);
            int dim                         =   max - min + 1;
            printf("Dim cpu  : %d\n",dim);
            streamIndex[pit + si]           =   make_short2(min,max);

            for(int i = 0 ; i < dim; i++)
                streamCost[pitLine + sizeStreamCost+i] = sizeStreamCost+i;//CData<int>::GetRandomValue(4,10);

            si++;
            sizeStreamCost += dim;

        }
    }
    streamCost.OutputValues();

    LaunchKernelOptOneDirection2(streamCost,streamIndex,dimVolCost,5.0f, sizeStreamCost);

    streamCost.Dealloc();
    streamIndex.Dealloc();

}

#endif
