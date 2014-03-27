#include "GpGpu/GpGpu_InterOptimisation.h"

InterfOptimizGpGpu::InterfOptimizGpGpu()
{
    CreateJob();
}

InterfOptimizGpGpu::~InterfOptimizGpGpu(){}

void InterfOptimizGpGpu::Dealloc()
{
    _H_data2Opt.Dealloc();
    _D_data2Opt.Dealloc();

    _preFinalCost1D.Dealloc();
    _poInitCost.Dealloc();
}

void InterfOptimizGpGpu::oneDirOptGpGpu()
{
    _D_data2Opt.SetNbLine(_H_data2Opt.NBlines());

    _H_data2Opt.ReallocOutputIf(_H_data2Opt.s_InitCostVol().GetSize());

    _D_data2Opt.ReallocIf(_H_data2Opt);

    //      Transfert des données vers le device                            ---------------		-
    _D_data2Opt.CopyHostToDevice(_H_data2Opt);

    //      Kernel optimisation                                             ---------------     -
    OptimisationOneDirection(_D_data2Opt);

    getLastCudaError("kernelOptiOneDirection failed");

    //      Copie des couts de passage forcé du device vers le host         ---------------     -
    _D_data2Opt.CopyDevicetoHost(_H_data2Opt);

}

void InterfOptimizGpGpu::Prepare(uint x, uint y, ushort penteMax, ushort NBDir)
{
    uint size = (uint)(1.5f*sqrt((float)x *x + y * y));

    ResetIdBuffer();

    SetProgress(NBDir);

    _H_data2Opt.ReallocParam(size);
    _D_data2Opt.ReallocParam(size);    
    _D_data2Opt.setPenteMax(penteMax);

//    _D_data2Opt._m_DzMax = iDivUp32(_poInitCost._maxDz) << 5;

//    _D_data2Opt._m_DzMax = _D_data2Opt._m_DzMax < NAPPEMAX ? NAPPEMAX : _D_data2Opt._m_DzMax;

//    if (_D_data2Opt._m_DzMax > 4 * NAPPEMAX) _D_data2Opt._m_DzMax = 4 * NAPPEMAX;

    //DUMP_UINT((uint)_D_data2Opt._m_DzMax)

 //   _D_data2Opt._m_DzMax = NAPPEMAX;

    _preFinalCost1D.Fill(0);

    SetPreComp(true);
}

void InterfOptimizGpGpu::threadCompute()
{   
    while(true)
    {
        if(GetCompute())
        {          
            SetCompute(false);

            _D_data2Opt.SetNbLine(_H_data2Opt.nbLines());

            _H_data2Opt.ReallocOutputIf(_H_data2Opt.s_InitCostVol().GetSize(),GetIdBuf());

            _D_data2Opt.ReallocIf(_H_data2Opt);

            //      Transfert des données vers le device                            ---------------		-
            _D_data2Opt.CopyHostToDevice(_H_data2Opt,GetIdBuf());

            SetPreComp(true);

            //      Kernel optimisation                                             ---------------     -
            OptimisationOneDirectionZ_V02(_D_data2Opt);

            //      Copie des couts de passage forcé du device vers le host         ---------------     -
            _D_data2Opt.CopyDevicetoHost(_H_data2Opt,GetIdBuf());

            SwitchIdBuffer();

            while(GetDataToCopy());

            IncProgress();

            SetDataToCopy(true);
        }
        else
            boost::this_thread::sleep(boost::posix_time::microsec(1));
    }
}

void InterfOptimizGpGpu::freezeCompute()
{
    SetDataToCopy(false);
    SetCompute(false);
    SetPreComp(false);
}
