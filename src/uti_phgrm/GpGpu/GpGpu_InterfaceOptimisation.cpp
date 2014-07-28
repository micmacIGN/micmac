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

void InterfOptimizGpGpu::oneDirOptGpGpu()
{
    _D_data2Opt.SetNbLine(_H_data2Opt.NBlines());

    _H_data2Opt.ReallocOutputIf(_H_data2Opt.s_InitCostVol().GetSize(),_H_data2Opt.s_Index().GetSize());

    _D_data2Opt.ReallocIf(_H_data2Opt);

    //      Transfert des données vers le device                            ---------------		-
    _D_data2Opt.CopyHostToDevice(_H_data2Opt);

    //      Kernel optimisation                                             ---------------     -
    OptimisationOneDirection(_D_data2Opt);

    getLastCudaError("kernelOptiOneDirection failed");

    //      Copie des couts de passage forcé du device vers le host         ---------------     -
    _D_data2Opt.CopyDevicetoHost(_H_data2Opt);

}

void InterfOptimizGpGpu::Prepare(uint x, uint y, ushort penteMax, ushort NBDir,float zReg,float zRegQuad)
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
    _FinalDefCor.Fill(0);
    _preFinalCost1D.Fill(0);

}

void InterfOptimizGpGpu::optimisation()
{
    _D_data2Opt.SetNbLine(_H_data2Opt.nbLines());

    _H_data2Opt.ReallocOutputIf(_H_data2Opt.s_InitCostVol().GetSize(),_H_data2Opt.s_Index().GetSize(),GetIdBuf());

    _D_data2Opt.ReallocIf(_H_data2Opt);

    //      Transfert des données vers le device                            ---------------		-
    _D_data2Opt.CopyHostToDevice(_H_data2Opt,GetIdBuf());

    SetPreComp(true);

    //      Kernel optimisation                                             ---------------     -
    OptimisationOneDirectionZ_V02(_D_data2Opt);

    //      Copie des couts de passage forcé du device vers le host         ---------------     -
    _D_data2Opt.CopyDevicetoHost(_H_data2Opt,GetIdBuf());
}

void InterfOptimizGpGpu::oneCompute()
{
    //cout << "START OPTI :" << boost::this_thread::get_id() << endl;

    while(!GetCompute())
    {
        //printf("WAIT COMPUTE CORREL...\n");
        boost::this_thread::sleep(boost::posix_time::microsec(1));
    }

    SetCompute(false);

    optimisation();

    while(GetDataToCopy());
//    {
//        printf("WAIT DATA COPY CORREL...\n");
//        boost::this_thread::sleep(boost::posix_time::microsec(5));
//    }


    //IncProgress();
    SwitchIdBuffer();

    SetDataToCopy(true);

    SetCompute(true);

    //cout << "END OPTI   :" << boost::this_thread::get_id() << endl;


    //printf("END oneCompute\n");
}

void InterfOptimizGpGpu::threadCompute()
{   
    while(true)
    {
        if(GetCompute() /*&& !_H_data2Opt.nbLines()*/)
        {

            // TEMP : TENTATIVE DE DEBUGAGE THREAD
            while(!_H_data2Opt.nbLines())
                boost::this_thread::sleep(boost::posix_time::microsec(1));

            SetCompute(false);

            oneCompute();
        }
        else
            boost::this_thread::sleep(boost::posix_time::microsec(1));
    }
}

void InterfOptimizGpGpu::freezeCompute()
{
    _H_data2Opt.setNbLines(0);
    _D_data2Opt.setNbLines(0);

//    _preFinalCost1D
//    _poInitCost

    SetDataToCopy(false);
    SetCompute(false);
    SetPreComp(false);
}

void InterfOptimizGpGpu::simpleJob()
{
    boost::thread tOpti(&InterfOptimizGpGpu::oneCompute,this);
    tOpti.detach();
    //detached
}
