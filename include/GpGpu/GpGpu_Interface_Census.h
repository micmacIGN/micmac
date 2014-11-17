#ifndef GPGPU_INTERFACE_CENSUS_H
#define GPGPU_INTERFACE_CENSUS_H


#include "GpGpu/GpGpu_MultiThreadingCpu.h"
#include "GpGpu/GpGpu_Data.h"

struct dataCorrelMS
{
    CuHostData3D<float>         _Image_00;
    CuHostData3D<float>         _Image_01;
    CuHostData3D<pixel>         _maskErod_01;
    CuHostData3D<short2>        _Interval_Z;
    CuHostData3D<short2>        _pVignette;

    int2                        _offset0;
    int2                        _offset1;
};

class GpGpuInterfaceCensus : public CSimpleJobCpuGpu< bool>
{

private:
    dataCorrelMS _dataCMS;
};

#endif // GPGPU_INTERFACE_CENSUS_H
