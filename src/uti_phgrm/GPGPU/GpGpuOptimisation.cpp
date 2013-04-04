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
    uint    pitchZ      = size(dimVolCost);
    uint2   ptTer       = make_uint2(0);

    _volumeCost.SetName("_volumeCost");
    _volumeCost.Realloc(dimRVolCost,1);

//    cudaError err = cudaErrorMemoryAllocation;
//    _volumeCost.ErrorOutput(err,"StructureVolumeCost");
//    volumeCost.ErrorOutput(err,"StructureVolumeCost");

    for(; ptTer.x < dimVolCost.x; ptTer.x++)
    {
        for(; ptTer.y < dimVolCost.y; ptTer.y++)
        {
            for(uint z = 0; z < nbZ; z++)
            {
                uint2 ptRTer = make_uint2(nbZ * ptTer.x + z,ptTer.y);
                uint idDest  = to1D(ptRTer,dimRVolCost);
                uint idSrc   = pitchZ * z + to1D(ptTer,dimVolCost);

                *(_volumeCost.pData()+ idDest) = abs(*(volumeCost.pData() + idSrc)-1);
            }
        }
    }

    GpGpuTools::Array1DtoImageFile(_volumeCost.pData(),"dd",dimVolCost);

    OptimisationOneDirection(_volumeCost, nbZ, dimVolCost);

    //GpGpuTools::OutputArray(_volumeCost.pData(),_volumeCost.GetDimension(),3,0,32);
    //_volumeCost.Dealloc();
}
