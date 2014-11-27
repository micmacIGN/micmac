#ifndef GPGPU_INTERFACE_CENSUS_H
#define GPGPU_INTERFACE_CENSUS_H

#include "GpGpu/GpGpu.h"
#include "GpGpu/GpGpu_Data.h"
#include "GpGpu/GpGpu_MultiThreadingCpu.h"
#include <string>

#define SIZEWIN(rayonWin) (rayonWin*2+1)*(rayonWin*2+1)
#define NBSCALE 3

struct dataCorrelMS;
struct constantParameterCensus;

extern "C" textureReference&  texture_ImageEpi(int nEpi);
extern "C" textureReference* pTexture_ImageEpi(int nEpi);
extern "C" textureReference* ptexture_Masq_Erod(int nEpi);
extern "C" void LaunchKernelCorrelationCensusPreview(dataCorrelMS &data,constantParameterCensus &param);
extern "C" void paramCencus2Device( constantParameterCensus &param );
extern "C" void LaunchKernelCorrelationCensus(dataCorrelMS &data,constantParameterCensus &param);


struct constantParameterCensus
{
    //constantParameterCensus():_NBScale(NBSCALE){}

    ///
    /// \brief aVV
    /// parcours des vignettes
    short2  aVV[NBSCALE][SIZEWIN(NBSCALE)];

    ///
    /// \brief size_aVV
    /// taille des vignettes
    ushort  size_aVV[NBSCALE];

    ///
    /// \brief aVPds
    /// poid des vignettes
    float   aVPds[NBSCALE];

    ///
    /// \brief anOff0
    /// offset terrain image epipolaire 0
    int2    anOff0;

    ///
    /// \brief anOff1
    /// offset terrain image epipolaire 1
    int2    anOff1;

    Rect    _zoneTerrain;

    uint2   _dimTerrain;

    ushort  aNbScale;

    float   anEpsilon;

    float   mAhDefCost;

    ushort  mNbByPix;

    float   aStepPix;

    uint3   mDim3Cache;

    void transfertConstantCensus(const std::vector<std::vector<Pt2di> >  &aVV,
            const std::vector<double >              &aVPds,
            int2    offset0,
            int2    offset1,
            ushort  NbByPix,
            float   StepPix,
            ushort  nbscale = NBSCALE );

    void transfertTerrain(Rect    zoneTerrain);

    void dealloc();

//    __device__ uint3 dim3Cache()
//    {
//        return make_uint3(_dimTerrain.x,_dimTerrain.y,aNbScale);
//    }

};


#define NBEPIIMAGE 2
struct dataCorrelMS
{
    dataCorrelMS();

    ///
    /// \brief _HostImage
    /// Images epipolaires
    CuHostData3D<float>         _HostImage[NBEPIIMAGE];

    ///
    /// \brief _HostMaskErod
    /// Masque
    CuHostData3D<pixel>         _HostMaskErod[NBEPIIMAGE];

    /// \brief _HostInterval_Z
    /// Nappe des Z host
    //CuHostData3D<short2>        _HostInterval_Z;
    /// \brief _DeviceInterval_Z
    /// Nappe des Z device
    //CuDeviceData3D<short2>      _DeviceInterval_Z;
    CuUnifiedData3D<short2>        _uInterval_Z;

    CuUnifiedData3D<float>         _uCost;

    ImageGpGpu<pixel,cudaContext>           _dt_MaskErod[NBEPIIMAGE];
    ImageLayeredGpGpu<float,cudaContext>    _dt_Image[NBEPIIMAGE];

    textureReference*           _texImage[NBEPIIMAGE];
    textureReference*           _texMaskErod[NBEPIIMAGE];

    void    transfertImage(uint2 sizeImage, float ***dataImage , int id);

    void    transfertMask(uint2 dimMask0, uint2 dimMask1, pixel **mImMasqErod_0, pixel **mImMasqErod_1);

    void    transfertNappe(int  mX0Ter, int  mX1Ter, int  mY0Ter, int  mY1Ter, short **mTabZMin, short **mTabZMax);


//private:

    void    syncDeviceData();

    void    dealloc();

    uint    _maxDeltaZ;

};

class GpGpuInterfaceCensus : public CSimpleJobCpuGpu< bool>
{
public:

    GpGpuInterfaceCensus();
    ~GpGpuInterfaceCensus();

    virtual void    freezeCompute(){}

    void            jobMask();

    void transfertImageAndMask(uint2 sI0,uint2 sI1,float ***dataImg0,float ***dataImg1,pixel **mask0,pixel **mask1);

    void transfertParamCensus(Rect terrain,
                              const std::vector<std::vector<Pt2di> >  &aVV,
                              const std::vector<double >              &aVPds,
                              int2      offset0,
                              int2      offset1,
                              short   **mTabZMin,
                              short   **mTabZMax,
                              ushort    NbByPix,
                              float     StepPix,
                              ushort nbscale = NBSCALE );

private:

    virtual void    simpleWork(){}

    dataCorrelMS    _dataCMS;

    constantParameterCensus _cDataCMS;
};

#endif // GPGPU_INTERFACE_CENSUS_H
