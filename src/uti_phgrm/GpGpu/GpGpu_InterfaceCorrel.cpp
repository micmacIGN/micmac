#include "GpGpu/GpGpu_InterCorrel.h"

/// \brief Constructeur GpGpuInterfaceCorrel
GpGpuInterfaceCorrel::GpGpuInterfaceCorrel():
    GlobalMaskVolume(0),
    ReduceMaskVolume(0)
{
    for (int s = 0;s<NSTREAM;s++)
        checkCudaErrors( cudaStreamCreate(GetStream(s)));

    CreateJob();
}

GpGpuInterfaceCorrel::~GpGpuInterfaceCorrel()
{
    for (int s = 0;s<NSTREAM;s++)
        checkCudaErrors( cudaStreamDestroy(*(GetStream(s))));

}

uint GpGpuInterfaceCorrel::InitCorrelJob(int Zmin, int Zmax)
{

    uint DZ     = abs(Zmin - Zmax);

    uint interZ = min(INTERZ, DZ);

    _param[0].SetZCInter(interZ);
    _param[1].SetZCInter(interZ);

    //CopyParamTodevice(_param[0]);

    _data2Cor.ReallocHostData(interZ,_param[0]);

    if(UseMultiThreading())
    {
        //CreateJob();
        ResetIdBuffer();
        SetPreComp(true);
    }

    return interZ;
}

/// \brief Initialisation des parametres constants
void GpGpuInterfaceCorrel::SetParameter(int nbLayer , uint2 dRVig , uint2 dimImg, float mAhEpsilon, uint samplingZ, int uvINTDef )
{
    _param[0].SetParamInva( dRVig * 2 + 1,dRVig, dimImg, mAhEpsilon, samplingZ, uvINTDef, nbLayer);
    _param[1].SetParamInva( dRVig * 2 + 1,dRVig, dimImg, mAhEpsilon, samplingZ, uvINTDef, nbLayer);
}

void GpGpuInterfaceCorrel::BasicCorrelation(uint ZInter)
{



    Param(GetIdBuf()).SetZCInter(ZInter);

    // Re-allocation les structures de données si elles ont été modifiées

    Data().ReallocDeviceData(Param(GetIdBuf()));

    // copie des donnees du host vers le device

    Data().copyHostToDevice(Param(GetIdBuf()));
    CopyParamTodevice(_param[GetIdBuf()]);
    // Indique que la copie est terminée pour le thread de calcul des projections


    SetPreComp(true);

    // Lancement du calcul de correlation

    CorrelationGpGpu();

    // relacher la texture de projection

    Data().UnBindTextureProj();

    // Lancement du calcul de multi-correlation

    MultiCorrelationGpGpu();

    // Copier les resultats de calcul des couts du device vers le host!

    Data().CopyDevicetoHost(GetIdBuf());

}

cudaStream_t* GpGpuInterfaceCorrel::GetStream( int stream )
{
    return &(_stream[stream]);
}

void GpGpuInterfaceCorrel::threadCompute()
{
    ResetIdBuffer();
    while (true)
    {
        if (GetCompute())
        {
            uint interZ = GetCompute();
            SetCompute(0);

            BasicCorrelation(interZ);

            SwitchIdBuffer();

            while(GetDataToCopy());

            SetDataToCopy(interZ);
        }
    }
}

void GpGpuInterfaceCorrel::freezeCompute()
{
    SetDataToCopy(0);
    SetCompute(0);
    SetPreComp(false);
    //KillJob();
}

void GpGpuInterfaceCorrel::IntervalZ(uint &interZ, int anZProjection, int aZMaxTer)
{
    uint intZ = (uint)abs(aZMaxTer - anZProjection );
    if (interZ >= intZ  &&  anZProjection != (aZMaxTer - 1) )
        interZ = intZ;
}

float *GpGpuInterfaceCorrel::VolumeCost()
{
    return UseMultiThreading() ? Data().HostVolumeCost(!GetIdBuf()) : Data().HostVolumeCost(0);
}

bool GpGpuInterfaceCorrel::TexturesAreLoaded()
{
    return _TexturesAreLoaded;
}

void GpGpuInterfaceCorrel::SetTexturesAreLoaded(bool load)
{
    _TexturesAreLoaded = load;
}

void GpGpuInterfaceCorrel::CorrelationGpGpu(const int s, ushort idBuf)
{
    LaunchKernelCorrelation(s, *(GetStream(s)),_param[idBuf], _data2Cor);
}

void GpGpuInterfaceCorrel::MultiCorrelationGpGpu(const int s, ushort idBuf)
{
    LaunchKernelMultiCorrelation( *(GetStream(s)),_param[idBuf],  _data2Cor);
}

pCorGpu& GpGpuInterfaceCorrel::Param(ushort idBuf)
{
    return _param[idBuf];
}

void GpGpuInterfaceCorrel::signalComputeCorrel(uint dZ)
{
    SetPreComp(false);
    SetCompute(dZ);
}
