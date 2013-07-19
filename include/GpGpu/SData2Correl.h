#ifndef _SDATA2CORREL_H
#define _SDATA2CORREL_H

#include "GpGpu/cudaAppliMicMac.cuh"

extern "C" textureReference&    getMaskD();
extern "C" textureReference&	getMask();
extern "C" textureReference&	getImage();
extern "C" textureReference&	getProjection(int TexSel);

#define SYNC false
#define ASYNC true

struct SData2Correl
{
    SData2Correl();

    ~SData2Correl();

    void    SetImages( float* dataImage, uint2 dimImage, int nbLayer );

    void    SetMask( pixel* dataMask, uint2 dimMask );

    void    MemsetHostVolumeProj(uint iDef);

    float*  HostVolumeCost(uint id);

    float2* HostVolumeProj();

    uint*   DeviVolumeNOK(uint s);

    float*  DeviVolumeCache(uint s);

    float*  DeviVolumeCost(uint s);

    void    copyHostToDevice(uint s, pCorGpu param);

    void    CopyDevicetoHost(uint idBuf, uint s);

    void    UnBindTextureProj(uint s);

    void    DeallocHostData();

    void    DeallocDeviceData();

    void    ReallocHostData(uint zInter, pCorGpu param);

    void    ReallocDeviceData(pCorGpu param);

private:

    void    ReallocDeviceData(int nStream, pCorGpu param);

    void    MallocInfo();

    textureReference& GetTeXProjection( int TexSel );

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
    textureReference&           _texProjections_00;
    textureReference&           _texProjections_01;
    int                         _countAlloc;

    //void    ReallocDeviceArrayAsync(pCorGpu param, cudaStream_t* pstream, uint s );
};

#endif
