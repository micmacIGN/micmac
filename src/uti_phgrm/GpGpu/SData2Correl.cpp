#include "GpGpu/SData2Correl.h"

SData2Correl::SData2Correl():
    _texMask(getMask()),
    _texMaskD(getMaskD()),
    _texImages(getImage()),
    _texProjections_00(getProjection(0)),
    _texProjections_01(getProjection(1)),
    _countAlloc(0)

{
    _d_volumeCost[0].SetName("_d_volumeCost");
    _d_volumeCach[0].SetName("_d_volumeCach");
    _d_volumeNIOk[0].SetName("_d_volumeNIOk");
    _dt_mask.CData2D::SetName("_dt_mask");
    _dt_LayeredImages.CData3D::SetName("_dt_LayeredImages");
    _dt_LayeredProjection->CData3D::SetName("_dt_LayeredProjection");

    // Parametres texture des projections
    for (int s = 0;s<NSTREAM;s++)
        GpGpuTools::SetParamterTexture(GetTeXProjection(s));

    // Parametres texture des Images
    GpGpuTools::SetParamterTexture(_texImages);

    for (int i = 0; i < SIZERING; ++i)
    {
        _hVolumeCost[i].SetName("_hVolumeCost_0",i);
        _hVolumeCost[i].SetPageLockedMemory(true);
    }
    _hVolumeProj.SetName("_hVolumeProj");
}

SData2Correl::~SData2Correl()
{
    DeallocDeviceData();
}

void SData2Correl::MallocInfo()
{
    std::cout << "Malloc Info GpGpu\n";
    GpGpuTools::OutputInfoGpuMemory();
    _d_volumeCost[0].MallocInfo();
    _d_volumeCach[0].MallocInfo();
    _d_volumeNIOk[0].MallocInfo();
    _dt_mask.CData2D::MallocInfo();
    _dt_LayeredImages.CData3D::MallocInfo();
    _dt_LayeredProjection[0].CData3D::MallocInfo();
}

float *SData2Correl::HostVolumeCost(uint id)
{
    return _hVolumeCost[id].pData();
}

float2 *SData2Correl::HostVolumeProj()
{
    return _hVolumeProj.pData();
}

void SData2Correl::DeallocHostData()
{
    for (int i = 0; i < SIZERING; ++i)
        if(!_hVolumeCost[i].GetSizeofMalloc())
            _hVolumeCost[i].Dealloc();

    if(!_hVolumeProj.GetSizeofMalloc())
        _hVolumeProj.Dealloc();
}

void SData2Correl::DeallocDeviceData()
{
    checkCudaErrors( cudaUnbindTexture(&_texImages) );
    checkCudaErrors( cudaUnbindTexture(&_texMask) );
    checkCudaErrors( cudaUnbindTexture(&_texMaskD) );

    for (int s = 0;s<NSTREAM;s++)
    {
        _d_volumeCach[s].Dealloc();
        _d_volumeCost[s].Dealloc();
        _d_volumeNIOk[s].Dealloc();
        _dt_LayeredProjection[s].Dealloc();
    }

    _dt_mask.Dealloc();
    _dt_LayeredImages.Dealloc();

}

textureReference &SData2Correl::GetTeXProjection(int TexSel)
{
    switch (TexSel)
    {
    case 0:
        return _texProjections_00;
    case 1:
        return _texProjections_01;
    default:
        return _texProjections_00;
    }
}

void SData2Correl::SetImages(float *dataImage, uint2 dimImage, int nbLayer)
{
    // Images vers Textures Gpu
    _dt_LayeredImages.CData3D::Realloc(dimImage,nbLayer);
    _dt_LayeredImages.copyHostToDevice(dataImage);
    _dt_LayeredImages.bindTexture(_texImages);

}

void SData2Correl::SetMask(pixel *dataMask, uint2 dimMask)
{
    _dt_mask.Dealloc();
    _dt_mask.InitImage(dimMask,dataMask);

    if(!_dt_mask.bindTexture(_texMask))
    {
        _dt_mask.OutputInfo();
        _dt_mask.CData2D::Name();
    }
}

void SData2Correl::copyHostToDevice(pCorGpu param,uint s)
{

    uint2 dimP = _dt_LayeredProjection[s].GetDimension();

    if(!aEq(param.dimSTer,dimP))
        _dt_LayeredProjection[s].Realloc(param.dimSTer,param.nbImages * param.ZLocInter);
    // Copier les projections du host --> device
    _dt_LayeredProjection[s].copyHostToDevice(_hVolumeProj.pData());
    // Lié de données de projections du device avec la texture de projections
    _dt_LayeredProjection[s].bindTexture(GetTeXProjection(s));
}

void SData2Correl::CopyDevicetoHost(uint idBuf, uint s)
{
    _d_volumeCost[s].CopyDevicetoHost(_hVolumeCost[idBuf].pData());
}

void SData2Correl::UnBindTextureProj(uint s)
{
    checkCudaErrors( cudaUnbindTexture(&(GetTeXProjection(s))));
}

void SData2Correl::ReallocHostData(uint zInter, pCorGpu param)
{
    for (int i = 0; i < SIZERING; ++i)
        _hVolumeCost[i].ReallocIf(param.dimTer,zInter);
    _hVolumeProj.ReallocIf(param.dimSTer,zInter*param.nbImages);
}

void SData2Correl::ReallocDeviceData(pCorGpu param)
{
    for (int s = 0;s<NSTREAM;s++)
    {
        ReallocDeviceData(s, param);

        _d_volumeCost[s].Memset(param.IntDefault);
        _d_volumeCach[s].Memset(param.IntDefault);
        _d_volumeNIOk[s].Memset(0);
    }
}

/*
void SData2Correl::ReallocDeviceArrayAsync(pCorGpu param, cudaStream_t *pstream, uint s)
{
    _d_volumeCost[s].SetDimension(param.dimTer,param.ZLocInter);
    _d_volumeCach[s].SetDimension(param.dimCach,param.nbImages * param.ZLocInter);
    _d_volumeNIOk[s].SetDimension(param.dimTer,param.ZLocInter);

    if (_d_volumeCost[s].GetSizeofMalloc() < _d_volumeCost[s].Sizeof() )
        ReallocDeviceArray(s, param);

    _d_volumeCost[s].MemsetAsync(param.IntDefault, *pstream);
    _d_volumeCach[s].MemsetAsync(param.IntDefault, *pstream);
    _d_volumeNIOk[s].MemsetAsync(0,*pstream);
}
*/

uint    *SData2Correl::DeviVolumeNOK(uint s){ return _d_volumeNIOk[s].pData();}

float   *SData2Correl::DeviVolumeCache(uint s){ return _d_volumeCach[s].pData();}

float   *SData2Correl::DeviVolumeCost(uint s){ return _d_volumeCost[s].pData();}

void SData2Correl::ReallocDeviceData(int nStream, pCorGpu param)
{
    //_countAlloc++;
    _d_volumeCost[nStream].ReallocIf(param.dimTer,     param.ZLocInter);
    _d_volumeCach[nStream].ReallocIf(param.dimCach,    param.nbImages * param.ZLocInter);
    _d_volumeNIOk[nStream].ReallocIf(param.dimTer,     param.ZLocInter);
}

void SData2Correl::MemsetHostVolumeProj(uint iDef)
{
    _hVolumeProj.Memset(iDef);
}
