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

static __constant__ float penalite[PENALITE];

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
template<class T> __global__ void kernelOptiOneDirection(T* g_idata,T* g_odata,int* g_oPath, uint2 dimPlanCost, uint2 delta, float* iMinCost, float defaultValue)
{
    __shared__ T    sdata[32];
    //__shared__ int  minCostC[1];

    const int   tid = threadIdx.x;
    const uint  pit = blockIdx.x * blockDim.x;
    uint        i0  = pit + tid;
    sdata[tid]      = g_idata[i0];
    bool        defV= sdata[tid] == defaultValue;
    g_odata[i0]     = defV ? 0 : sdata[tid];
    g_oPath[i0]     = tid;

    //if(tid == 0) minCostC[0]  = 1e9;

    T minCost, cost;

    for(int l=1;l<dimPlanCost.y;l++)
    {
        uint        i1   = i0 + dimPlanCost.x;
        int         iL   = tid;

        if(i1<size(dimPlanCost))
        {
            cost = g_idata[i1];

            if(cost!=defaultValue)
            {
                minCost = defV ? cost : cost + sdata[tid] + penalite[0];

                __syncthreads();

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

/*
    int intCost = (int)(minCost * 1e5f);
    atomicMin(minCostC,intCost);
    __syncthreads();
    if(minCostC[0] == intCost)
        iMinCost[blockIdx.x] = tid;
*/
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

    kernelOptiOneDirection<T><<<Blocks,Threads>>>(dInputData.pData(),dOutputData.pData(),dPath.pData(),diPlanCost, delta,dMinCostId.pData(),defaultValue);
    getLastCudaError("kernelOptimisation failed");

    dOutputData.CopyDevicetoHost(hOutputValue.pData());
    dPath.CopyDevicetoHost(hPath.pData());
    dMinCostId.CopyDevicetoHost(hMinCostId.pData());

//    uint2   ptTer;
//    uint2   prev = make_uint2(0,1);
//    for ( ptTer.x = 0; ptTer.x < dimTer.x; ptTer.x++)
//        for(ptTer.y = 1; ptTer.y < dimTer.y ; ptTer.y++)
//        {
//            uint2 pt = make_uint2(ptTer.x * dimVolCost.z + (uint)hMinCostId[ptTer - prev],ptTer.y);
//            hMinCostId[ptTer] =  (float)hPath[pt];
//        }
//    for (ptTer.x = 0; ptTer.x < dimTer.x; ptTer.x++)
//        for(ptTer.y = 0; ptTer.y < dimTer.y ; ptTer.y++)
//            if (defaultValue == hInputValue[ptTer])
//                hMinCostId[ptTer] = 0.0f;
//    hMinCostId.OutputValues();
//    hInputValue.OutputValues(0,XY,Rect(0,0,32,dimVolCost.y));
//    hPath.OutputValues(0,XY,Rect(0,0,dimVolCost.z,dimVolCost.y));
//    hOutputValue.OutputValues(0,XY,Rect(0,0,dimVolCost.z,dimVolCost.y),4);
//    GpGpuTools::Array1DtoImageFile(GpGpuTools::MultArray(hMinCostId.pData(),dimTer,1.0f/32.0f),"ZMap.pgm",dimTer);

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
    //    uint2 dim1  = make_uint2(4,32);
    //    int nZ1     = 32;
    //    CuHostData3D<float> data1(make_uint2(dim1.x * nZ1,dim1.y));
    //    data1.FillRandom(0,20);
    //    LaunchKernelOptOneDirection(data1,nZ1, dim1);
    //    data1.Dealloc();
    LaunchKernelOptOneDirection(data,dimVolCost,defaultValue);
}

/// \brief Apple exterieur du kernel
extern "C" void Launch()
{
    uint3 dimVolumeCost  = make_uint3(256,256,32);

    CuHostData3D<float> volumeCost(make_uint2(dimVolumeCost.x * dimVolumeCost.z,dimVolumeCost.y));
    volumeCost.FillRandom(0,2);
    LaunchKernelOptOneDirection(volumeCost,dimVolumeCost,5.0f);

//    data1.OutputValues();
//    data1.OutputValues(1,YZ);
//    data1.OutputValues(1,XZ);

    volumeCost.Dealloc();

    //LaunchKernel<int>();
}

#endif
