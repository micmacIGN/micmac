#include "GpGpu/GpGpu_InterOptimisation.h"

InterfOptimizGpGpu::InterfOptimizGpGpu()
{
    //CreateJob();

    freezeCompute();
}

InterfOptimizGpGpu::~InterfOptimizGpGpu(){}

void InterfOptimizGpGpu::Dealloc()
{
    _H_data2Opt.Dealloc();
    _D_data2Opt.Dealloc();

    _preFinalCost1D.Dealloc();
    _poInitCost.Dealloc();
}


void InterfOptimizGpGpu::Prepare(uint x, uint y, ushort penteMax, ushort NBDir,float zReg,float zRegQuad, ushort costDefMask,ushort costDefMaskTrans, bool hasMaskAuto)
{
    uint size = (uint)(1.5f*sqrt((float)x *x + y * y));

    _H_data2Opt.setDzMax(_poInitCost._maxDz);
    _D_data2Opt.setDzMax(_poInitCost._maxDz);

    ResetIdBuffer();
    SetPreComp(true);

    SetProgress(NBDir);

    _H_data2Opt.ReallocParam(size);
    _D_data2Opt.ReallocParam(size);    
    _D_data2Opt.setPenteMax(penteMax);
    _D_data2Opt.setZReg(zReg);
    _D_data2Opt.setZRegQuad(zRegQuad);
    _D_data2Opt.setCostDefMasked(costDefMask);
    _D_data2Opt.setCostTransMaskNoMask(costDefMaskTrans);
    _D_data2Opt.setHasMaskAuto(hasMaskAuto);

    _FinalDefCor.Fill(0);
    _preFinalCost1D.Fill(0);

}

void InterfOptimizGpGpu::optimisation()
{
    _D_data2Opt.SetNbLine(_H_data2Opt.nbLines());

    _D_data2Opt.setPenteMax(_H_data2Opt.penteMax());

    _H_data2Opt.ReallocOutputIf(_H_data2Opt.s_InitCostVol().GetSize(),_H_data2Opt.s_Index().GetSize(),GetIdBuf());

    _D_data2Opt.ReallocIf(_H_data2Opt);

    //      Transfert des données vers le device                            ---------------		-
    _D_data2Opt.CopyHostToDevice(_H_data2Opt,GetIdBuf());

    SetPreComp(true);

    //      Kernel optimisation                                             ---------------     -
    Gpu_OptimisationOneDirection(_D_data2Opt);

    //      Copie des couts de passage forcé du device vers le host         ---------------     -
    _D_data2Opt.CopyDevicetoHost(_H_data2Opt,GetIdBuf());
}

void InterfOptimizGpGpu::simpleWork()
{
    optimisation();
}

void InterfOptimizGpGpu::freezeCompute()
{
    _H_data2Opt.setNbLines(0);
    _D_data2Opt.setNbLines(0);

    SetDataToCopy(false);
    SetCompute(false);
    SetPreComp(false);
}


