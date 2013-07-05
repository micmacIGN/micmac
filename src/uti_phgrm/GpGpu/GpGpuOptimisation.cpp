#include "GpGpu/GpGpuOptimisation.h"

InterfOptimizGpGpu::InterfOptimizGpGpu(bool multiThreading):
    _idbuf(false),
    _multiThreading(multiThreading)
{

    if(UseMultiThreading())
        _gpGpuThreadOpti = new boost::thread(&InterfOptimizGpGpu::threadFuncOptimi,this);
    SetDirToCopy(false);
    SetCompute(false);
    SetPreCompNextDir(false);
}

InterfOptimizGpGpu::~InterfOptimizGpGpu(){

    if(UseMultiThreading())
    {
        _gpGpuThreadOpti->interrupt();
        //_gpGpuThreadOpti->join();
        delete _gpGpuThreadOpti;
    }
    _mutexCompu.unlock();
    _mutexCopy.unlock();
    _mutexPreCompute.unlock();
}

void InterfOptimizGpGpu::Dealloc()
{
    _H_data2Opt.Dealloc();
    _D_data2Opt.Dealloc();
}

void InterfOptimizGpGpu::oneDirOptGpGpu()
{
    _D_data2Opt.SetNbLine(_H_data2Opt._nbLines);
    _H_data2Opt.ReallocOutputIf(_H_data2Opt._s_InitCostVol.GetSize());
    _D_data2Opt.ReallocIf(_H_data2Opt);

    //      Transfert des données vers le device                            ---------------		-
    _D_data2Opt.CopyHostToDevice(_H_data2Opt);

    //      Kernel optimisation                                             ---------------     -
    OptimisationOneDirection(_D_data2Opt);
    getLastCudaError("kernelOptiOneDirection failed");

    //      Copie des couts de passage forcé du device vers le host         ---------------     -
    _D_data2Opt.CopyDevicetoHost(_H_data2Opt);

}

void InterfOptimizGpGpu::ReallocParam(uint size)
{
    _idbuf  = true;
    _idDir  = 0;
    _H_data2Opt.ReallocParam(size);
    _D_data2Opt.ReallocParam(size);
}

void InterfOptimizGpGpu::createThreadOptGpGpu()
{
    _gpGpuThreadOpti = new boost::thread(&InterfOptimizGpGpu::threadFuncOptimi,this);
}

void InterfOptimizGpGpu::deleteThreadOptGpGpu()
{
    _gpGpuThreadOpti->interrupt();
    delete _gpGpuThreadOpti;
}

void InterfOptimizGpGpu::SetCompute(bool compute)
{
    boost::lock_guard<boost::mutex> guard(_mutexCompu);
    _compute = compute;
}

bool InterfOptimizGpGpu::GetCompute()
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

bool InterfOptimizGpGpu::UseMultiThreading()
{
    return _multiThreading;
}

void InterfOptimizGpGpu::SetPreCompNextDir(bool precompute)
{
    boost::lock_guard<boost::mutex> guard(_mutexPreCompute);
    _precompute = precompute;
}

void InterfOptimizGpGpu::threadFuncOptimi()
{
    bool idbuf  = false;

    while(true)
    {
        boost::this_thread::sleep(boost::posix_time::microsec(1));
        if(/*!GetDirToCopy() && */GetCompute())
        {
            //printf("compute[%d]      : %d\n",idbuf,idDir);
            SetCompute(false);

            _D_data2Opt.SetNbLine(_H_data2Opt._nbLines);
            _H_data2Opt.ReallocOutputIf(_H_data2Opt._s_InitCostVol.GetSize(),idbuf);
            _D_data2Opt.ReallocIf(_H_data2Opt);

            //      Transfert des données vers le device                            ---------------		-
            _D_data2Opt.CopyHostToDevice(_H_data2Opt,idbuf);

            SetPreCompNextDir(true);

            //      Kernel optimisation                                             ---------------     -
            OptimisationOneDirection(_D_data2Opt);
            getLastCudaError("kernelOptiOneDirection failed");

            //      Copie des couts de passage forcé du device vers le host         ---------------     -
            _D_data2Opt.CopyDevicetoHost(_H_data2Opt,idbuf);

            while(GetDirToCopy());
            SetDirToCopy(true);
            idbuf =! idbuf;

        }
    }
}

