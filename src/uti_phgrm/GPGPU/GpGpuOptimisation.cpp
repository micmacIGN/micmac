#include <iostream>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_math.h>
#include <helper_cuda.h>
#include "GpGpu/GpGpuOptimisation.h"

InterfMicMacOptGpGpu::InterfMicMacOptGpGpu(){}

InterfMicMacOptGpGpu::~InterfMicMacOptGpGpu(){}

void InterfMicMacOptGpGpu::StructureVolumeCost(CuHostData3D<float> &volumeCost)
{
    uint    nbZ         = volumeCost.GetNbLayer();
    uint2   dimVolCost  = volumeCost.GetDimension();
    uint2   dimRVolCost = make_uint2(dimVolCost.x*nbZ,dimVolCost.y);
    uint3   ptTer;

    _volumeCost.SetName("_volumeCost");
    _volumeCost.Realloc(dimRVolCost,1);

    for(ptTer.x = 0; ptTer.x < dimVolCost.x; ptTer.x++)
        for(ptTer.y = 0; ptTer.y < dimVolCost.y; ptTer.y++)
            for(ptTer.z = 0; ptTer.z < nbZ; ptTer.z++)
            {
                uint2 ptRTer = make_uint2(nbZ * ptTer.x + ptTer.z,ptTer.y);
                _volumeCost[ptRTer] = volumeCost[ptTer];
            }

    //_volumeCost.OutputValues(0,3,0,32);
    //volumeCost.OutputValues(0,3,0);

    OptimisationOneDirection(_volumeCost, nbZ, dimVolCost);

    //_volumeCost.Dealloc();
}
