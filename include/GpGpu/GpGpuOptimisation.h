#ifndef __OPTIMISATION_H__
#define __OPTIMISATION_H__

#include "GpGpu/GpGpuTools.h"

extern "C" void Launch();
extern "C" void OptimisationOneDirection(CuHostData3D<float> data, int nZ, uint2 dim);

template <class T>
void LaunchKernel();



class InterfMicMacOptGpGpu
{
public:
    InterfMicMacOptGpGpu();
    ~InterfMicMacOptGpGpu();

    void StructureVolumeCost(CuHostData3D<float> &volumeCost);

private:

    CuHostData3D<float> _volumeCost;
};


#endif
