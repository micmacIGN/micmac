#ifndef _OPTIMISATION_KERNEL_Z_H_
#define _OPTIMISATION_KERNEL_Z_H_

#include "GpGpu/GpGpu_StreamData.cuh"
#include "GpGpu/SData2Optimize.h"

__device__ inline void readDepthCoor(short2 *GBuffer, short2 *SBuffer, uint& id)
{
    SBuffer[threadIdx.x] = GBuffer[threadIdx.x + id];
    // On pourrait imaginer un buffer des tailles calculer en parallel
    // SIZEBUFFER[threadIdx.x] = count(lI[threadIdx]);
    id += WARPSIZE;
}

template<class T, bool sens> __device__
void RunLine(short2 *GBuffer,ushort *GInitCost)
{

    const uint lLine    = 256;
    uint  idRun         = 0;
    ushort tid          = threadIdx.x;
    __shared__ short2 depthCoor[WARPSIZE];
    __shared__ ushort initCost[NAPPEMAX];

    uint   G_idIdex  = 0;
    uint   G_idCost  = 0;
    ushort S_idCost  = 0;

    while(idRun < lLine)
    {
        readDepthCoor(GBuffer,depthCoor,G_idIdex);

        const uint Z2Comp = min(lLine-idRun,WARPSIZE);

        ushort l = 0;

        while(l < Z2Comp)
        {
            const short2 lI     = GBuffer[l];
            const ushort sLI    = count(lI);

            if(S_idCost + sLI > NAPPEMAX)
            {
                initCost[threadIdx.x] = initCost[S_idCost + tid];
                S_idCost = 0;
                for(ushort i = 0;i<NAPPEMAX-WARPSIZE;i+=WARPSIZE)
                    initCost[i + tid] = GInitCost[G_idCost + i + tid];
            }

            l++;
        }

        idRun += WARPSIZE;
    }

}


#endif //_OPTIMISATION_KERNEL_Z_H_
