#ifndef _SDATA2CORREL_H
#define _SDATA2CORREL_H

#include "GpGpu/GpGpuTools.h"
#include "GpGpu/cudaAppliMicMac.cuh"

extern "C" textureReference&    getMaskD();
extern "C" textureReference&	getMask();
extern "C" textureReference&	getImage();
extern "C" textureReference&	getProjection(int TexSel);

struct SData2Correl
{
    SData2Correl();

    ~SData2Correl();

    void    SetImages( float* dataImage, uint2 dimImage, int nbLayer );
    void    SetMask( pixel* dataMask, uint2 dimMask );

    void    ReallocDeviceData(int nStream, uint interZ,pCorGpu param);

    void    Realloc(pCorGpu param, uint oldSizeTer);

    void    MallocInfo();

    void    MemsetProj(uint iDef);

    float*  OuputCost(uint id);

    float2* InputProj();

    void    DeallocVolumes();

    void    DeallocMemory();

    textureReference& GetTeXProjection( int TexSel );

    void    copyHostToDevice(uint s);

    void    CopyDevicetoHost(uint idBuf, uint s);

    void    UnBindTextureProj(uint s);

    void    ReallocHostData(uint zInter, pCorGpu param);

    void    ReallocAllDeviceData(uint interZ, pCorGpu param);

    void    ReallocAllDeviceDataAsync(uint interZ, pCorGpu param, cudaStream_t* pstream, uint s );


    uint*   DVolumeNOK(uint s){ return _d_volumeNIOk[s].pData();}
    float*  DVolumeCache(uint s){ return _d_volumeCach[s].pData();}
    float*  DVolumeCost(uint s){ return _d_volumeCost[s].pData();}


    CuHostData3D<float>         _hVolumeCost[2];
    CuHostData3D<float2>        _hVolumeProj;

    CuDeviceData3D<float>       _d_volumeCost[NSTREAM];	// volume des couts
    CuDeviceData3D<float>       _d_volumeCach[NSTREAM];	// volume des calculs intermédiaires
    CuDeviceData3D<uint>        _d_volumeNIOk[NSTREAM];	// nombre d'image correct pour une vignette

    ImageCuda<pixel>            _dt_mask;
    ImageLayeredCuda<float>     _dt_LayeredImages;
    ImageLayeredCuda<float2>    _dt_LayeredProjection[NSTREAM];

    textureReference&           _texMask;
    textureReference&           _texMaskD;
    textureReference&           _texImages;
    //textureReference&         _texCache;
    textureReference&           _texProjections_00;
    textureReference&           _texProjections_01;
    textureReference&           _texProjections_02;
    textureReference&           _texProjections_03;
    textureReference&           _texProjections_04;
    textureReference&           _texProjections_05;
    textureReference&           _texProjections_06;
    textureReference&           _texProjections_07;
};

#endif



