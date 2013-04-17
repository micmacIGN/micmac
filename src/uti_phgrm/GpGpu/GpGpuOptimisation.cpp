#include <iostream>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_math.h>
#include <helper_cuda.h>
#include "GpGpu/GpGpuOptimisation.h"

InterfMicMacOptGpGpu::InterfMicMacOptGpGpu(){}

InterfMicMacOptGpGpu::~InterfMicMacOptGpGpu(){}

void InterfMicMacOptGpGpu::StructureVolumeCost(CuHostData3D<float> &volumeCost, float defaultValue)
{
    uint3   dimVolCost  = make_uint3(volumeCost.GetDimension().x,volumeCost.GetDimension().y,volumeCost.GetNbLayer());
    uint2   dimRVolCost = make_uint2(dimVolCost.x*dimVolCost.z,dimVolCost.y);
    uint3   ptTer;

    _volumeCost.SetName("_volumeCost");
    _volumeCost.Realloc(dimRVolCost,1);

    for(ptTer.x = 0; ptTer.x < dimVolCost.x; ptTer.x++)
        for(ptTer.y = 0; ptTer.y < dimVolCost.y; ptTer.y++)
            for(ptTer.z = 0; ptTer.z < dimVolCost.z; ptTer.z++)
                _volumeCost[make_uint2(dimVolCost.z * ptTer.x + ptTer.z,ptTer.y)] = volumeCost[ptTer];

/*
    volumeCost.OutputValues(0,XY,NEGARECT,3,defaultValue);
    volumeCost.OutputValues(0,XZ,NEGARECT,3,defaultValue);
    volumeCost.OutputValues(0,ZY,NEGARECT,3,defaultValue);
    _volumeCost.OutputValues(0,XY,Rect(0,0,32,dimVolCost.y),3,defaultValue);
*/
    OptimisationOneDirection(_volumeCost, dimVolCost, defaultValue);

    //_volumeCost.Dealloc();
}
