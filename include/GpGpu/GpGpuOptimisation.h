#ifndef __OPTIMISATION_H__
#define __OPTIMISATION_H__

#include "GpGpu/GpGpuTools.h"

extern "C" void Launch();
extern "C" void OptimisationOneDirection(CuHostData3D<float> data, int nZ, uint2 dim);

template <class T>
void LaunchKernel();


/// \class InterfMicMacOptGpGpu
/// \brief Class qui permet a micmac de lancer les calculs d optimisations sur le Gpu
class InterfMicMacOptGpGpu
{
public:
    InterfMicMacOptGpGpu();
    ~InterfMicMacOptGpGpu();

    /// \brief  Restructuration des donnes du volume de correlation
    ///         Pour le moment il lance egalement le calcul d optimisation
    void StructureVolumeCost(CuHostData3D<float> &volumeCost);

private:

    CuHostData3D<float> _volumeCost;
};


#endif
