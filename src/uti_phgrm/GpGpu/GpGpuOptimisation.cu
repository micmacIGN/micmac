#ifndef _OPTIMISATION_KERNEL_H_
/// \brief ....
#define _OPTIMISATION_KERNEL_H_

/// \file       GpGpuOptimisation.cu
/// \brief      Kernel optimisation
/// \author     GC
/// \version    0.01
/// \date       Avril 2013

#include "GpGpu/GpGpuStreamData.cuh"

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
                    costMin = min(costMin, pData[2][Z - uZ_Next.x] + pData[idBuffer][Z - uZ_Prev.x+ i]);

                pData[!idBuffer][Z - uZ_Next.x] = costMin;
                gData[costStream.getStartIndex() + idCurLine * WARPSIZE + Z - uZ_Next.x] = costMin;
            }

            z += min(uZ_Next.y - z,WARPSIZE);
        }

        idBuffer = !idBuffer;
        uZ_Prev = uZ_Next;
    }

}

template<class T> __global__ void kernelOptiOneDirection(T* gStream, short2* gStreamId, T* g_odata_AV,T* g_odata_AR, uint3 dimBlockTer, uint penteMax)
{
    __shared__ T        bufferData[WARPSIZE];
    __shared__ short2   bufferIndex[WARPSIZE];
    __shared__ T        pdata[3][NAPPEMAX];

    const int       pit     =   blockIdx.x * dimBlockTer.y;
    const int       pitStr  =   pit * dimBlockTer.z;
    bool            idBuf   =   false;

    CDeviceDataStream<T> costStream(bufferData, gStream,bufferIndex, gStreamId + pit,pitStr);

    ScanOneSens<T>(costStream,eAVANT,dimBlockTer.y, pdata,idBuf,g_odata_AV,penteMax);
    ScanOneSens<T>(costStream,eARRIERE,dimBlockTer.y, pdata,idBuf,g_odata_AR,penteMax);
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

    //---------------------- Copie des penalites dans le device ---------------------------------------

    checkCudaErrors(cudaMemcpyToSymbol(penalite,    hPen,       sizeof(float)*PENALITE));
    checkCudaErrors(cudaMemcpyToSymbol(dMapIndex,   hMapIndex,  sizeof(ushort)*WARPSIZE));

    //-------------------------------------------------------------------------------------------------

    uint2   sizeInput   =   make_uint2(dimVolCost.x * dimVolCost.z,dimVolCost.y);
    uint2   sizeIndex   =   make_uint2(dimVolCost.y,dimVolCost.x);

    //    ---------------------- Declaration des variables Host ---------------------------------------

    CuHostData3D<T>         hOutputValue_AV(sizeInput,1);
    CuHostData3D<T>         hOutputValue_AR(sizeInput,1);
    hOutputValue_AV.SetName("hOutputValue");

    //---------------------- Variables Device ---------------------------------------------------------

    CuDeviceData3D<T>       dInputStream(   sizeInput,  1, "dInputStream");
    CuDeviceData3D<short2>  dInputIndex (   sizeIndex,  1, "dInputIndex");
    CuDeviceData3D<T>       dOutputData_AV( sizeInput,  1, "dOutputData");
    CuDeviceData3D<T>       dOutputData_AR( sizeInput,  1, "dOutputData");


    //  ------------------- Initialisation des Variables Device, est-il necessaire---------------------

    dOutputData_AV.Memset(0);

    //  ------------------- Copie du volume de couts dans le device  ----------------------------------

    dInputStream.CopyHostToDevice(  hInputStream.pData());
    dInputIndex.CopyHostToDevice(   hInputindex.pData());

    //-------------------------------------------------------------------------------------------------

    kernelOptiOneDirection<T><<<Blocks,Threads>>>
                                                (
                                                    dInputStream.pData(),
                                                    dInputIndex.pData(),
                                                    dOutputData_AV.pData(),
                                                    dOutputData_AR.pData(),
                                                    dimVolCost,deltaMax
                                                );

    getLastCudaError("kernelOptiOneDirection failed");

    //-------------------------------------------------------------------------------------------------

    dOutputData_AV.CopyDevicetoHost(hOutputValue_AV.pData());
    dOutputData_AR.CopyDevicetoHost(hOutputValue_AR.pData());

    dOutputData_AV.Dealloc();
    dOutputData_AR.Dealloc();
    dInputStream.Dealloc();
    dInputIndex.Dealloc();

    hOutputValue_AV.Dealloc();
    hOutputValue_AR.Dealloc();
}

/// \brief Appel exterieur du kernel d optimisation
extern "C" void OptimisationOneDirection(CuHostData3D<int> &data,CuHostData3D<short2> &index, uint3 dimVolCost)
{
    LaunchKernelOptOneDirection(data,index,dimVolCost);
}

/// \brief Appel exterieur du kernel
extern "C" void Launch()
{
    uint3 dimVolCost  = make_uint3(256,256,32);

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
