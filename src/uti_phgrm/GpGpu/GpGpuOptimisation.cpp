#include "GpGpu/GpGpuOptimisation.h"

InterfOptimizGpGpu::InterfOptimizGpGpu():
    _idbuf(false)
{

    if(UseMultiThreading())
    {
        setThread(new boost::thread(&InterfOptimizGpGpu::threadCompute,this));
        freezeCompute();
    }
}

InterfOptimizGpGpu::~InterfOptimizGpGpu(){}

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

void InterfOptimizGpGpu::threadCompute()
{
    bool idbuf  = false;

    while(true)
    {
        boost::this_thread::sleep(boost::posix_time::microsec(1));
        if(GetCompute())
        {
            SetCompute(false);
            _D_data2Opt.SetNbLine(_H_data2Opt._nbLines);
            _H_data2Opt.ReallocOutputIf(_H_data2Opt._s_InitCostVol.GetSize(),idbuf);
            _D_data2Opt.ReallocIf(_H_data2Opt);

            //      Transfert des données vers le device                            ---------------		-
            _D_data2Opt.CopyHostToDevice(_H_data2Opt,idbuf);

            SetPreComp(true);

            //      Kernel optimisation                                             ---------------     -
            OptimisationOneDirection(_D_data2Opt);

            //      Copie des couts de passage forcé du device vers le host         ---------------     -
            _D_data2Opt.CopyDevicetoHost(_H_data2Opt,idbuf);

            while(GetDataToCopy());
            SetDataToCopy(true);
            idbuf =! idbuf;

        }
    }
}

void InterfOptimizGpGpu::freezeCompute()
{
    SetDataToCopy(false);
    SetCompute(false);
    SetPreComp(false);
}

