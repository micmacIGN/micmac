#include "GpGpu/SData2Correl.h"

SData2Correl::SData2Correl():
    _texMaskGlobal(getMaskGlobal()),
    _texImages(getImage()),
    _texProjections_00(getProjection(0)),
    _texProjections_01(getProjection(1))
{
    _d_volumeCost[0].SetName("_d_volumeCost");
    _d_volumeCach[0].SetName("_d_volumeCach");
    _d_volumeNIOk[0].SetName("_d_volumeNIOk");    
    _dt_GlobalMask.SetNameImage("_dt_GlobalMask");

    _dt_LayeredImages.CData3D::SetName("_dt_LayeredImages");
    _dt_LayeredProjection->CData3D::SetName("_dt_LayeredProjection");

    // Parametres texture des projections
    for (int s = 0;s<NSTREAM;s++)
        GpGpuTools::SetParamterTexture(GetTeXProjection(s));

    // Parametres texture des Images
    GpGpuTools::SetParamterTexture(_texImages);

    _texMaskGlobal.addressMode[0]	= cudaAddressModeBorder;
    _texMaskGlobal.addressMode[1]	= cudaAddressModeBorder;
    _texMaskGlobal.filterMode       = cudaFilterModePoint; //cudaFilterModePoint cudaFilterModeLinear
    _texMaskGlobal.normalized       = false;

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
    CGpGpuContext<cudaContext>::OutputInfoGpuMemory();
    _d_volumeCost[0].MallocInfo();
    _d_volumeCach[0].MallocInfo();
    _d_volumeNIOk[0].MallocInfo();
    _dt_GlobalMask.DecoratorImage<cudaContext>::MallocInfo();
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

Rect *SData2Correl::HostRect()
{
    return _hRect.pData();
}

ushort2 *SData2Correl::HostClassEqui()
{
    return _hClassEqui.pData();
}

void SData2Correl::DeallocHostData()
{
    for (int i = 0; i < SIZERING; ++i)
            _hVolumeCost[i].Dealloc();

    _hVolumeProj.Dealloc();
    _hRect.Dealloc();
    _hClassEqui.Dealloc();
}

void SData2Correl::DeallocDeviceData()
{
    checkCudaErrors( cudaUnbindTexture(&_texImages) );    
    checkCudaErrors( cudaUnbindTexture(&_texMaskGlobal) );

    for (int s = 0;s<NSTREAM;s++)
    {
        _d_volumeCach[s].Dealloc();
        _d_volumeCost[s].Dealloc();
        _d_volumeNIOk[s].Dealloc();
        _dt_LayeredProjection[s].Dealloc();
    }

    _dt_GlobalMask.Dealloc();
    _dt_LayeredImages.Dealloc();

    _dRect.Dealloc();

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
#ifdef  NVTOOLS
    GpGpuTools::NvtxR_Push(__FUNCTION__,0xFF1A22B5);
#endif
    // Images vers Textures Gpu
    _dt_LayeredImages.CData3D::ReallocIfDim(dimImage,nbLayer);
    _dt_LayeredImages.copyHostToDevice(dataImage);
    _dt_LayeredImages.bindTexture(_texImages);
#ifdef  NVTOOLS
    nvtxRangePop();
#endif
}

void SData2Correl::SetGlobalMask(pixel *dataMask, uint2 dimMask)
{   
	#ifdef  NVTOOLS
    GpGpuTools::NvtxR_Push(__FUNCTION__,0xFF1A2B51);
    #endif
    //  TODO Verifier si le ReallocIfDim fonctionne.... s'il ne redimmension pas a chaque fois!!!
    _dt_GlobalMask.DecoratorImage<cudaContext>::ReallocIfDim(dimMask,1);
    _dt_GlobalMask.copyHostToDevice(dataMask);
    _dt_GlobalMask.bindTexture(_texMaskGlobal);
	#ifdef  NVTOOLS
    nvtxRangePop();
	#endif
}

void SData2Correl::copyHostToDevice(pCorGpu param,uint s)
{
	#ifdef  NVTOOLS
    GpGpuTools::NvtxR_Push(__FUNCTION__,0xFF292CB0);
	#endif


//    printf("STart Realloc _dt_LayeredProjection\n");
    _dt_LayeredProjection[s].ReallocIfDim(param.dimSTer,param.invPC.nbImages * param.ZCInter);
//    printf("START Realloc _dt_LayeredProjection\n");

    // Gestion des bords d'images
    _dRect.ReallocIfDim(make_uint2(1,1),param.invPC.nbImages * param.ZCInter);
    _dRect.CopyHostToDevice(_hRect.pData());
    //
    // Gestion des classes d'equivalences
    _dClassEqui.ReallocIfDim(make_uint2(1,1),param.invPC.nbImages);
    _dClassEqui.CopyHostToDevice(_hClassEqui.pData());


    // Copier les projections du host --> device
    _dt_LayeredProjection[s].copyHostToDevice(_hVolumeProj.pData());

    // Lié de données de projections du device avec la texture de projections
    _dt_LayeredProjection[s].bindTexture(GetTeXProjection(s));
	#ifdef  NVTOOLS
    nvtxRangePop();
	#endif
}

void SData2Correl::CopyDevicetoHost(uint idBuf, uint s)
{
    _d_volumeCost[s].CopyDevicetoHost(_hVolumeCost[idBuf]);
}

void SData2Correl::UnBindTextureProj(uint s)
{
    checkCudaErrors( cudaUnbindTexture(&(GetTeXProjection(s))));
}

void SData2Correl::ReallocHostClassEqui(uint nbImages)
{
    _hClassEqui.ReallocIfDim(make_uint2(1,1),nbImages);
}

void SData2Correl::ReallocHostData(uint zInter, pCorGpu param)
{
	#ifdef  NVTOOLS
    GpGpuTools::NvtxR_Push(__FUNCTION__,0xFFAA0000);
	#endif
    for (int i = 0; i < SIZERING; ++i)
        _hVolumeCost[i].ReallocIf(param.HdPc.dimTer,zInter);

    _hVolumeProj.ReallocIf(param.dimSTer,zInter*param.invPC.nbImages);
    _hRect.ReallocIfDim(make_uint2(1,1),zInter*param.invPC.nbImages);

	#ifdef  NVTOOLS
    nvtxRangePop();
	#endif
}

void SData2Correl::ReallocHostData(uint zInter, pCorGpu param, uint idBuff)
{
    _hVolumeCost[idBuff].ReallocIf(param.HdPc.dimTer,zInter);

    _hVolumeProj.ReallocIf(param.dimSTer,zInter*param.invPC.nbImages);
    _hRect.ReallocIfDim(make_uint2(1,1),zInter*param.invPC.nbImages);

}

void SData2Correl::ReallocDeviceData(pCorGpu &param)
{
	#ifdef NVTOOLS
    GpGpuTools::NvtxR_Push(__FUNCTION__,0xFF1A2BB5);
	#endif
    for (int s = 0;s<NSTREAM;s++)
    {
        ReallocDeviceData(s, param);

        DeviceMemset(param,s);
    }
	#ifdef NVTOOLS
    nvtxRangePop();
	#endif
}

void    SData2Correl::DeviceMemset(pCorGpu &param, uint s)
{
	#ifdef NVTOOLS
    GpGpuTools::NvtxR_Push(__FUNCTION__,0xFF1A2BB5);
	#endif
    _d_volumeCost[s].Memset(param.invPC.IntDefault);

    // A vERIFIER que le memset est inutile
    //_d_volumeCach[s].Memset(param.IntDefault);

    _d_volumeNIOk[s].Memset(0);
	#ifdef NVTOOLS
    nvtxRangePop();
	#endif
}

uint    *SData2Correl::DeviVolumeNOK(uint s){

    return _d_volumeNIOk[s].pData();

}

float   *SData2Correl::DeviVolumeCache(uint s){

    return _d_volumeCach[s].pData();

}

float   *SData2Correl::DeviVolumeCost(uint s){

    return _d_volumeCost[s].pData();

}

Rect *SData2Correl::DeviRect()
{
    return _dRect.pData();
}

ushort2 *SData2Correl::DeviClassEqui()
{
    return _dClassEqui.pData();
}

void SData2Correl::ReallocDeviceData(int nStream, pCorGpu param)
{

    _d_volumeCost[nStream].ReallocIf(param.HdPc.dimTer,     param.ZCInter);

    _d_volumeCach[nStream].ReallocIf(param.HdPc.dimCach,    param.invPC.nbImages * param.ZCInter);

    _d_volumeNIOk[nStream].ReallocIf(param.HdPc.dimTer,     param.ZCInter * param.invPC.nbClass);
}

void SData2Correl::MemsetHostVolumeProj(uint iDef)
{
    _hVolumeProj.Memset(iDef);
}
