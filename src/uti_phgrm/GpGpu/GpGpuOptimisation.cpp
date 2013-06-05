#include <iostream>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_math.h>
#include <helper_cuda.h>
#include "GpGpu/GpGpuOptimisation.h"

InterfOptimizGpGpu::InterfOptimizGpGpu()
{

//    SetDirToCompute(false);
//    SetDirToCopy(false);
//    SetPreCompNextDir(true);
//    _gpGpuThreadOpti = new boost::thread(&InterfOptimizGpGpu::threadFuncOptimi,this);

}

InterfOptimizGpGpu::~InterfOptimizGpGpu(){

//    _gpGpuThreadOpti->interrupt();
//    delete _gpGpuThreadOpti;

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

    _D_data2Opt.SetNbLine(_H_data2Opt.nbLines);
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

void InterfOptimizGpGpu::SetDirToCompute(bool compute)
{
    boost::lock_guard<boost::mutex> guard(_mutexCompu);
    _compute = compute;
}

bool InterfOptimizGpGpu::GetDirToCompute()
{
    boost::lock_guard<boost::mutex> guard(_mutexCompu);
    return _compute;
}

void InterfOptimizGpGpu::SetDirToCopy(bool copy)
{
    boost::lock_guard<boost::mutex> guard(_mutexCopy);
    _copy = copy;
}

bool InterfOptimizGpGpu::GetDirToCopy()
{
    boost::lock_guard<boost::mutex> guard(_mutexCopy);
    return _copy;
}

bool InterfOptimizGpGpu::GetPreCompNextDir()
{
    boost::lock_guard<boost::mutex> guard(_mutexPreCompute);
    return _precompute;

}

void InterfOptimizGpGpu::SetPreCompNextDir(bool precompute)
{

    boost::lock_guard<boost::mutex> guard(_mutexPreCompute);
    _precompute = precompute;
}

void InterfOptimizGpGpu::threadFuncOptimi()
{
    while(true)
    {


        if(GetDirToCompute())
        {

            SetDirToCompute(false);

            _D_data2Opt.SetNbLine(_H_data2Opt.nbLines);
            _D_data2Opt.ReallocIf(_H_data2Opt);

            //      Copie du volume de couts dans le device                         ---------------		-
            _D_data2Opt.CopyHostToDevice(_H_data2Opt);

            SetPreCompNextDir(true);

            //      Kernel optimisation                                             ---------------     -
            OptimisationOneDirection(_D_data2Opt);
            getLastCudaError("kernelOptiOneDirection failed");

            //      Copie des couts de passage forcé du device vers le host         ---------------     -
            _D_data2Opt.CopyDevicetoHost(_H_data2Opt);

            SetDirToCopy(true);

        }
    }
}

