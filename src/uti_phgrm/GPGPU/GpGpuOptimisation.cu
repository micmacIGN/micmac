#ifndef _OPTIMISATION_KERNEL_H_
#define _OPTIMISATION_KERNEL_H_

#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_math.h>
#include <helper_cuda.h>
#include "GpGpu/GpGpuTools.h"
#include "GpGpu/helper_math_extented.cuh"

#define PENALITE 7
static __constant__ int penalite[PENALITE];

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template<class T>
struct SharedMemory
{
    __device__ inline operator       T *()
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }

    __device__ inline operator const T *() const
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }
};

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

template<class T> __global__ void kernelOptimisation(T* g_idata,T* g_odata,  int n)
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

template<class T> __global__ void kernelOptimisation2(T* g_idata,T* g_odata,int* path, uint2 dimLigne, uint2 delta, int* iMinCost )
{
    __shared__ T    sdata[32];
    __shared__ uint minCostC[1];

    uint tid     = threadIdx.x;
    uint i0      = blockIdx.x * blockDim.x * dimLigne.y + tid;
    sdata[tid]   = g_idata[i0];
    g_odata[i0]  = sdata[tid];
    path[i0]     = tid;
    if(tid == 0)
        minCostC[0]  = 1e9;

    T minCost, cost;

    for(int l=1;l<dimLigne.y;l++)
    {
        unsigned int i1 = i0 + dimLigne.x;


        if(i1<size(dimLigne))
        {
            cost    = g_idata[i1];
            minCost = cost + sdata[tid] + penalite[0];
            int iL  = tid;

            __syncthreads();

            for(int t = -((int)(delta.x)); t < ((int)(delta.y));t++)
            {
                int Tl = tid + t;
                if( t!=0 && Tl >= 0 && Tl < dimLigne.x)
                {
                    T Cost = cost + sdata[Tl] + penalite[abs(t)];
                    if(Cost < minCost)
                    {
                        minCost = Cost;
                        iL      = Tl;
                    }
                }
            }

            i0 = (blockIdx.x * blockDim.x + l) * dimLigne.y + tid;
            g_odata[i0] = minCost;
            sdata[tid]  = minCost;
            path[i0]    = iL;
        }
    }

    atomicMin(minCostC,minCost);
    __syncthreads();
    if(minCostC[0] == minCost)
        *iMinCost = tid;
}

template<class T>
void LaunchKernel()
{
    int warp    = 32;
    int nBLine  = 32;
    int si      = warp * nBLine;
    int longLine= 32;
    uint2 dA    = make_uint2(si,longLine);
    dim3 Threads(warp,1,1);
    dim3 Blocks(nBLine,1,1);

    int hPen[PENALITE] = {1,2,3,5,7,8,7};
    //int hPen[PENALITE] = {0,0,0,0,0,0,0};
    checkCudaErrors(cudaMemcpyToSymbol(penalite, hPen, sizeof(int)*PENALITE));

    int dZ          = 1;

    CuHostData3D<T> hostInputValue(dA,dZ);
    CuHostData3D<T> hostOutputValue(dA,dZ);
    CuHostData3D<T> hostPath(dA,dZ);

    int* hMinCostId = (int*)malloc(sizeof(int));

    hostInputValue.FillRandom(0,256);

    CuDeviceData3D<T> dInputData;
    CuDeviceData3D<T> dOutputData;
    CuDeviceData3D<int> dPath;
    CuDeviceData3D<int> minCostId;

    dInputData.SetName("dInputData");
    dOutputData.SetName("dOutputData");
    minCostId.SetName("minCostId");
    dPath.SetName("dPath");

    dInputData.Realloc(dA,dZ);
    dOutputData.Realloc(dA,dZ);
    dOutputData.Memset(0);
    dPath.Realloc(dA,dZ);
    dPath.Memset(0);
    minCostId.Realloc(make_uint2(1),1);
    minCostId.Memset(0);

    dInputData.CopyHostToDevice(hostInputValue.pData());

    uint2 delta = make_uint2(3,3);

    kernelOptimisation2<T><<<Blocks,Threads>>>(dInputData.pData(),dOutputData.pData(),dPath.pData(),dA, delta,minCostId.pData());
    getLastCudaError("kernelOptimisation failed");

    dOutputData.CopyDevicetoHost(hostOutputValue.pData());
    dPath.CopyDevicetoHost(hostPath.pData());
    minCostId.CopyDevicetoHost(hMinCostId);

//    GpGpuTools::OutputArray(hostInputValue.pData(),dA);
//    GpGpuTools::OutputArray(hostOutputValue.pData(),dA);
//    GpGpuTools::OutputArray(hostPath.pData(),dA);
    //printf("Index min Cost : %d\n",*hMinCostId);

    for(int i=0;i<longLine;i++)
    {

//        printf("%d, ",*hMinCostId);
        *hMinCostId = hostPath.pData()[(longLine - i -1)*warp + *hMinCostId];
    }

    printf("\n");
}

extern "C" void Launch()
{
    LaunchKernel<int>();
}

#endif
