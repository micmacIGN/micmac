#include <iostream>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_math.h>
#include <helper_cuda.h>
#include "GpGpu/GpGpuOptimisation.h"

InterfOptimizGpGpu::InterfOptimizGpGpu()
{
    _gpGpuThreadOpti = new boost::thread(&InterfOptimizGpGpu::threadFuncOptimi,this);
    SetDirToCopy(false);
    SetCompute(false);
    SetPreCompNextDir(false);
}

InterfOptimizGpGpu::~InterfOptimizGpGpu(){

    _gpGpuThreadOpti->interrupt();
    delete _gpGpuThreadOpti;

}

void InterfOptimizGpGpu::Dealloc()
{
    _H_data2Opt.Dealloc();
    _D_data2Opt.Dealloc();
}

void InterfOptimizGpGpu::oneDirOptGpGpu()
{

    /*
        uint    dimDeltaMax =   deltaMax * 2 + 1;
        float   hPen[PENALITE];
        ushort  hMapIndex[WARPSIZE];

        for(int i=0 ; i < WARPSIZE; i++)
            hMapIndex[i] = i / dimDeltaMax;

        for(int i=0;i<PENALITE;i++)
            hPen[i] = ((float)(1 / 10.0f));

        //      Copie des penalites dans le device                              ---------------		-

        checkCudaErrors(cudaMemcpyToSymbol(penalite,    hPen,       sizeof(float)   * PENALITE));
        checkCudaErrors(cudaMemcpyToSymbol(dMapIndex,   hMapIndex,  sizeof(ushort)  * WARPSIZE));
    */

    _D_data2Opt.SetNbLine(_H_data2Opt._nbLines);
    _D_data2Opt.ReallocIf(_H_data2Opt);

    //      Copie du volume de couts dans le device                         ---------------		-
    _D_data2Opt.CopyHostToDevice(_H_data2Opt);

    //      Kernel optimisation                                             ---------------     -
    OptimisationOneDirection(_D_data2Opt);
    getLastCudaError("kernelOptiOneDirection failed");

    //      Copie des couts de passage forcé du device vers le host         ---------------     -
    _D_data2Opt.CopyDevicetoHost(_H_data2Opt);

}

void InterfOptimizGpGpu::ReallocParam(uint size)
{
    _H_data2Opt.ReallocParam(size);
    _D_data2Opt.ReallocParam(size);
}

void InterfOptimizGpGpu::SetCompute(bool compute)
{
    boost::lock_guard<boost::mutex> guard(_mutexCompu);
    _compute = compute;
    _mutexCompu.unlock();
}

bool InterfOptimizGpGpu::GetCompute()
{
    boost::lock_guard<boost::mutex> guard(_mutexCompu);
    bool compute = _compute;
    _mutexCompu.unlock();
    return compute;
}

void InterfOptimizGpGpu::SetDirToCopy(bool copy)
{
    boost::lock_guard<boost::mutex> guard(_mutexCopy);
    _copy = copy;
    _mutexCopy.unlock();
}

bool InterfOptimizGpGpu::GetDirToCopy()
{
    boost::lock_guard<boost::mutex> guard(_mutexCopy);
    bool copy = _copy;
    _mutexCopy.unlock();
    return copy;
}

bool InterfOptimizGpGpu::GetPreCompNextDir()
{
    boost::lock_guard<boost::mutex> guard(_mutexPreCompute);
    bool precompute = _precompute;
    _mutexPreCompute.unlock();
    return precompute;

}

void InterfOptimizGpGpu::SetPreCompNextDir(bool precompute)
{
    boost::lock_guard<boost::mutex> guard(_mutexPreCompute);
    _precompute = precompute;
    _mutexPreCompute.unlock();
}

void InterfOptimizGpGpu::threadFuncOptimi()
{
    bool idbuf   = false;
    uint idDir  = 0;
    while(true)
    {
        if(GetCompute() && !GetDirToCopy())
        {
            printf("compute[%d]      : %d\n",idbuf,idDir);
            SetCompute(false);

            _D_data2Opt.SetNbLine(_H_data2Opt._nbLines);
            _H_data2Opt.ReallocOutputIf(_H_data2Opt._s_InitCostVol.GetSize());
            _D_data2Opt.ReallocIf(_H_data2Opt);

            //      Transfert des données vers le device                         ---------------		-
            _D_data2Opt.CopyHostToDevice(_H_data2Opt,idbuf);

            SetPreCompNextDir(true);

            //      Kernel optimisation                                             ---------------     -
            OptimisationOneDirection(_D_data2Opt);
            getLastCudaError("kernelOptiOneDirection failed");

            //      Copie des couts de passage forcé du device vers le host         ---------------     -
            //_D_data2Opt.CopyDevicetoHost(_H_data2Opt);
            _D_data2Opt.CopyDevicetoHost(_H_data2Opt._s_ForceCostVol.pData());

            SetDirToCopy(true);
            idbuf =! idbuf;
            idDir++;
        }
    }
}

