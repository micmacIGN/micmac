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

template<class T> __device__ void ReadOneSens(CDeviceDataStream<T> &costStream, bool sens, uint lenghtLine, T pData[][NAPPEMAX], bool& idBuffer, T* gData, ushort penteMax, uint3 dimBlockTer)
{
    const ushort    tid     = threadIdx.x;

    for(int idParLine = 0; idParLine < lenghtLine;idParLine++)
    {
        const short2 uZ = costStream.read(pData[0],tid,sens,0);
        short z = uZ.x;

        while( z < uZ.y )
        {
            int Z       = z + tid - uZ.x;
            gData[idParLine * dimBlockTer.z + Z]    = pData[0][Z];
            z          += min(uZ.y - z,WARPSIZE);
        }
    }
}

template<class T, bool sens > __device__ void DebugScanOneSens(CDeviceDataStream<T> &costStream, uint lenghtLine, T pData[][NAPPEMAX], bool& idBuffer, T* gCostVol, ushort penteMax, uint3 dimBlockTer)
{
    const ushort    tid     = threadIdx.x;
    short2          uZ_Prev = costStream.read(pData[idBuffer],tid, sens,0);
    short           z       = uZ_Prev.x;
    __shared__ T    minCost;

    if(sens)
        while( z < uZ_Prev.y )
        {
            int Z       = z + tid - uZ_Prev.x;
            gCostVol[Z]    = pData[idBuffer][Z];
            z += min(uZ_Prev.y - z,WARPSIZE);
        }

    for(int idParLine = 1; idParLine < lenghtLine;idParLine++)
    {

        //const short2 uZ_Next = make_short2(-122,123);
        const short2 uZ_Next = costStream.read(pData[2],tid,sens,0);

        short2 aDz;
        short z = uZ_Next.x;

        if(!tid) minCost = 1e9;

        while( z < uZ_Next.y )
        {
            int Z = z + tid;

            if( Z < uZ_Next.y)
            {
                ComputeIntervaleDelta(aDz,Z,penteMax,uZ_Next,uZ_Prev);
                T costMin   = 1e9;
                T costInit  = pData[2][Z - uZ_Next.x];

                for(short i = aDz.x ; i <= aDz.y; i++)
                    costMin = min(costMin, costInit + pData[idBuffer][Z - uZ_Prev.x + i]);

                pData[!idBuffer][Z - uZ_Next.x] = costMin;

                int idGData     = (sens ? idParLine : lenghtLine -  idParLine - 1) * dimBlockTer.z + Z - uZ_Next.x;
                int cost        = sens ? costMin : costMin + gCostVol[idGData] - costInit;

                gCostVol[idGData]  = cost;

                if(!sens)
                    atomicMin(&minCost,cost);

            }

            z += min(uZ_Next.y - z,WARPSIZE);
        }

        if(!sens)
        {
            z = uZ_Next.x;
            while( z < uZ_Next.y )
            {
                int Z = z + tid;
                int idGData     = (lenghtLine -  idParLine - 1) * dimBlockTer.z + Z - uZ_Next.x;
                gCostVol[idGData]  -= minCost;
                z += min(uZ_Next.y - z,WARPSIZE);
            }
        }

        idBuffer    = !idBuffer;
        uZ_Prev     = uZ_Next;
    }
}

template<class T, bool sens > __device__ void ScanOneSens(CDeviceDataStream<T> &costStream, uint lenghtLine, T pData[][NAPPEMAX], bool& idBuffer, T* gCostVol, ushort penteMax, uint3 dimBlockTer)
{
    const ushort    tid     = threadIdx.x;
    short2          uZ_Prev = costStream.read(pData[idBuffer],tid, sens,0);
    short           z       = uZ_Prev.x;
    __shared__ T    minCost;

    if(sens)
        while( z < uZ_Prev.y )
        {
            int Z       = z + tid - uZ_Prev.x;
            gCostVol[Z]    = pData[idBuffer][Z];
            z += min(uZ_Prev.y - z,WARPSIZE);
        }

    for(int idParLine = 1; idParLine < lenghtLine;idParLine++)
    {
        const short2 uZ_Next = costStream.read(pData[2],tid,sens,0);
        short2 aDz;
        short z = uZ_Next.x;

        if(!tid) minCost = 1e9;

        while( z < uZ_Next.y )
        {
            int Z = z + tid;

            if( Z < uZ_Next.y)
            {
                ComputeIntervaleDelta(aDz,Z,penteMax,uZ_Next,uZ_Prev);
                T costMin   = 1e9;
                T costInit  = pData[2][Z - uZ_Next.x];

                for(short i = aDz.x ; i <= aDz.y; i++)
                    costMin = min(costMin, costInit + pData[idBuffer][Z - uZ_Prev.x + i]);

                pData[!idBuffer][Z - uZ_Next.x] = costMin;

                int idGData     = (sens ? idParLine : lenghtLine -  idParLine - 1) * dimBlockTer.z + Z - uZ_Next.x;
                int cost        = sens ? costMin : costMin + gCostVol[idGData] - costInit;

                gCostVol[idGData]  = cost;

                if(!sens)
                    atomicMin(&minCost,cost);

            }


            z += min(uZ_Next.y - z,WARPSIZE);
        }

        if(!sens)
        {
            z = uZ_Next.x;
            while( z < uZ_Next.y )
            {
                int Z = z + tid;
                int idGData     = (lenghtLine -  idParLine - 1) * dimBlockTer.z + Z - uZ_Next.x;
                gCostVol[idGData]  -= minCost;
                z += min(uZ_Next.y - z,WARPSIZE);
            }
        }

        idBuffer    = !idBuffer;
        uZ_Prev     = uZ_Next;
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

    CDeviceDataStream<T> costStream(bufferData, gStream + pitStr,bufferIndex, gStreamId + pit);

   ScanOneSens<T,eAVANT>(costStream, dimBlockTer.y, pdata,idBuf,g_odata + pitStr,penteMax, dimBlockTer);
   // DebugScanOneSens<T,eAVANT>(costStream, dimBlockTer.y, pdata,idBuf,g_odata + pitStr,penteMax, dimBlockTer);
   ScanOneSens<T,eARRIERE>(costStream, dimBlockTer.y, pdata,idBuf,g_odata + pitStr,penteMax, dimBlockTer);

}

/// \brief Lance le kernel d optimisation pour une direction
template <class T> void LaunchKernelOptOneDirection(CuHostData3D<T> &hInputStream, CuHostData3D<short2> &hInputindex, uint3 dimVolCost, CuHostData3D<T> &h_ForceCostVol)
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

    checkCudaErrors(cudaMemcpyToSymbol(penalite,    hPen,       sizeof(float)   * PENALITE));
    checkCudaErrors(cudaMemcpyToSymbol(dMapIndex,   hMapIndex,  sizeof(ushort)  * WARPSIZE));

    //-------------------------------------------------------------------------------------------------

    uint2   sizeInput   =   make_uint2(dimVolCost.z, dimVolCost.y);
    uint2   sizeIndex   =   make_uint2(dimVolCost.y, dimVolCost.x);

    //---------------------- Variables Device ---------------------------------------------------------

    CuDeviceData3D<T>       d_InputStream    ( sizeInput,  dimVolCost.x, "d_InputStream"  );
    CuDeviceData3D<short2>  d_InputIndex     ( sizeIndex,  1,            "d_InputIndex"   );
    CuDeviceData3D<T>       d_ForceCostVol   ( sizeInput,  dimVolCost.x, "d_ForceCostVol");

    //  ------------------- Copie du volume de couts dans le device  ----------------------------------

    d_InputStream.CopyHostToDevice(  hInputStream.pData());
    d_InputIndex .CopyHostToDevice(  hInputindex .pData());

    //-------------------------------------------------------------------------------------------------


    kernelOptiOneDirection<T><<<Blocks,Threads>>>
                                                (
                                                    d_InputStream    .pData(),
                                                    d_InputIndex     .pData(),
                                                    d_ForceCostVol   .pData(),
                                                    dimVolCost,
                                                    deltaMax
                                                    );

    getLastCudaError("kernelOptiOneDirection failed");

    //-------------------------------------------------------------------------------------------------
    d_ForceCostVol.CopyDevicetoHost(h_ForceCostVol.pData());

    d_ForceCostVol   .Dealloc();;
    d_InputStream    .Dealloc();
    d_InputIndex     .Dealloc();

}

/// \brief Appel exterieur du kernel d optimisation
extern "C" void OptimisationOneDirection(CuHostData3D<uint> &data,CuHostData3D<short2> &index, uint3 dimVolCost, CuHostData3D<uint> & H_AV)
{
    LaunchKernelOptOneDirection(data,index,dimVolCost, H_AV);
}

/// \brief Appel exterieur du kernel
extern "C" void Launch()
{
    uint    prof        = 40;
    uint3   dimVolCost  = make_uint3(80,4,prof );

    CuHostData3D<uint>      H_StreamCost      ( NOPAGELOCKEDMEMORY, make_uint3( dimVolCost.z, dimVolCost.y, dimVolCost.x) );
    CuHostData3D<uint>      H_ForceCostVol    ( NOPAGELOCKEDMEMORY, make_uint3( dimVolCost.z, dimVolCost.y, dimVolCost.x) );
    CuHostData3D<short2>    H_StreamIndex     ( NOPAGELOCKEDMEMORY, make_uint2( dimVolCost.y, dimVolCost.x ));

    H_StreamCost  .SetName("streamCost");
    H_StreamIndex .SetName("streamIndex");

    uint si = 0 , sizeStreamCost = 0;

    srand (time(NULL));

    for(int i = 0 ; i < dimVolCost.x ; i++)
    {
        int pit         = i     * dimVolCost.y;
        int pitLine     = pit   * dimVolCost.z;
        si =  sizeStreamCost = 0;

        while (si < dimVolCost.y){

            int min                         =  -CData<int>::GetRandomValue(prof / 4,prof / 2 -1);
            int max                         =   CData<int>::GetRandomValue(prof / 4,prof / 2 -1);
            int dim                         =   max - min + 1;
            H_StreamIndex[pit + si]         =   make_short2(min,max);

            for(int i = 0 ; i < dim; i++)
                H_StreamCost[pitLine + sizeStreamCost + i] = i+1;//CData<uint>::GetRandomValue(16,128);

            si++;
            sizeStreamCost += dim;

        }
    }

    int id = 0;
    H_StreamCost.OutputValues(id);

    LaunchKernelOptOneDirection(H_StreamCost,H_StreamIndex,dimVolCost,H_ForceCostVol);

    H_ForceCostVol.OutputValues(id);

    H_StreamCost.Dealloc();
    H_StreamIndex.Dealloc();
}

#endif
