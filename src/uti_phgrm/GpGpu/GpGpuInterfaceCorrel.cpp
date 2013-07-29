#include "GpGpu/GpGpuInterCorrel.h"

/// \brief Constructeur GpGpuInterfaceCorrel
GpGpuInterfaceCorrel::GpGpuInterfaceCorrel()
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


    uint interZ = min(INTERZ, abs(Zmin - Zmax));

    _param.SetZCInter(interZ);

    CopyParamTodevice(_param);

    _data2Cor.ReallocHostData(interZ,_param);

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
    _param.SetParamInva( dRVig * 2 + 1,dRVig, dimImg, mAhEpsilon, samplingZ, uvINTDef, nbLayer);
}

void GpGpuInterfaceCorrel::BasicCorrelation(uint ZInter)
{

    Param().SetZCInter(ZInter);

    // Re-allocation les structures de données si elles ont été modifiées

    Data().ReallocDeviceData(Param());

    // copie des donnees du host vers le device

    Data().copyHostToDevice(Param());

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
//        else
//            boost::this_thread::sleep(boost::posix_time::microsec(1));
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

void GpGpuInterfaceCorrel::CorrelationGpGpu(const int s)
{
    LaunchKernelCorrelation(s, *(GetStream(s)),_param, _data2Cor);
}

void GpGpuInterfaceCorrel::MultiCorrelationGpGpu(const int s)
{
    LaunchKernelMultiCorrelation( *(GetStream(s)),_param,  _data2Cor);
}

pCorGpu& GpGpuInterfaceCorrel::Param()
{
    return _param;
}

void GpGpuInterfaceCorrel::signalComputeCorrel(uint dZ)
{
    SetPreComp(false);
    SetCompute(dZ);
}
