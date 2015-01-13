#ifndef GPGPU_INTERFACE_CORMULTISCALE_H
#define GPGPU_INTERFACE_CORMULTISCALE_H

#include "GpGpu/GpGpu.h"
#include "GpGpu/GpGpu_Data.h"
#include "GpGpu/GpGpu_MultiThreadingCpu.h"
#include <string>

#define SIZEWIN(rayonWin) (rayonWin*2+1)*(rayonWin*2+1)
#define NBSCALE 3

struct dataCorrelMS;
struct const_Param_Cor_MS;

extern "C" textureReference&  texture_ImageEpi(int nEpi);
extern "C" textureReference* pTexture_ImageEpi(int nEpi);
extern "C" textureReference* ptexture_Masq_Erod(int nEpi);
extern "C" void LaunchKernelCorrelationMultiScalePreview(dataCorrelMS &data,const_Param_Cor_MS &param);
extern "C" void paramCorMultiScale2Device( const_Param_Cor_MS &param );
extern "C" void LaunchKernel__Correlation_MultiScale(dataCorrelMS &data, const_Param_Cor_MS &parCMS);


struct const_Param_Cor_MS
{
    //constantParameterCorMultiScale():_NBScale(NBSCALE){}

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

    uint    maxDeltaZ;

    ///
    /// \brief mNbByPix
    /// nombre de phase par pixel
    ushort  mNbByPix;

    ///
    /// \brief aStepPix
    /// Pas sub-pixelaire
    float   aStepPix;

    ///
    /// \brief mDim3Cache
    /// dimension du cache preparatoire au calcul de correlation multi-echelle
    uint3   mDim3Cache;

    void init(const std::vector<std::vector<Pt2di> >  &aVV,
            const std::vector<double >              &aVPds,
            int2    offset0,
            int2    offset1,
            ushort  NbByPix,
            float   StepPix,
            ushort  nbscale = NBSCALE );

    void setTerrain(Rect    zoneTerrain);

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


    ///
    /// \brief _uInterval_Z
    ///
    CuUnifiedData3D<short2>        _uInterval_Z;

    ///
    /// \brief _uCost
    ///
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


private:
    void unitT__CopyCoordInColor(uint2 sizeImage, float *dest);
};

class GpGpu_Interface_Cor_MS : public CSimpleJobCpuGpu< bool>
{
public:

    GpGpu_Interface_Cor_MS();
    ~GpGpu_Interface_Cor_MS();

    virtual void    freezeCompute(){}

    void            Job_Correlation_MultiScale();

    void transfertImageAndMask(uint2 sI0,uint2 sI1,float ***dataImg0,float ***dataImg1,pixel **mask0,pixel **mask1);

    void init(Rect terrain,
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

    const_Param_Cor_MS _cDataCMS;
};

#endif // GPGPU_INTERFACE_CORMULTISCALE_H
