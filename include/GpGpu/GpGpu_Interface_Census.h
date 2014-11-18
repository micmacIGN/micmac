#ifndef GPGPU_INTERFACE_CENSUS_H
#define GPGPU_INTERFACE_CENSUS_H

#include "GpGpu/GpGpu.h"
#include "GpGpu/GpGpu_Data.h"
#include "GpGpu/GpGpu_MultiThreadingCpu.h"
#include <string>

#define SIZEWIN(rayonWin) (rayonWin*2+1)*(rayonWin*2+1)
#define NBSCALE 3

extern "C" textureReference&  texture_ImageEpi(int nEpi);
extern "C" textureReference* pTexture_ImageEpi(int nEpi);
extern "C" textureReference& texture_Masq_Erod();
extern "C" void LaunchKernelCorrelationCensus();

struct constantParameterCensus
{
    ///
    /// \brief w
    /// parcours des vignettes
    short2  w[NBSCALE][SIZEWIN(NBSCALE)];

    ///
    /// \brief sizeW
    /// taille des vignettes
    ushort  sizeW[NBSCALE];

    ///
    /// \brief poids
    /// poid des vignettes
    float   poids[NBSCALE];

    ///
    /// \brief _offset0
    /// offset terrain image epiolaire 0
    int2    _offset0;


    ///
    /// \brief _offset1
    /// offset terrain image epiolaire 1
    int2    _offset1;

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
    CuHostData3D<pixel>         _HostMaskErod;

    /// \brief _HostInterval_Z
    /// Nappe des Z host
    CuHostData3D<short2>        _HostInterval_Z;
    /// \brief _DeviceInterval_Z
    /// Nappe des Z device
    CuDeviceData3D<short2>      _DeviceInterval_Z;


    ImageGpGpu<pixel,cudaContext>           _dt_MaskErod;
    ImageLayeredGpGpu<float,cudaContext>    _dt_Image[NBEPIIMAGE];

    textureReference*           _texImage[NBEPIIMAGE];
    textureReference&           _texMaskErod;


    void    transfertImage(uint2 sizeImage, float ***dataImage , int id);

private:

    void syncDeviceData();

};

class GpGpuInterfaceCensus //: public CSimpleJobCpuGpu< bool>
{
public:

//    virtual void    freezeCompute();

    dataCorrelMS    _dataCMS;

private:
//    virtual void    simpleWork(){}


};

#endif // GPGPU_INTERFACE_CENSUS_H
