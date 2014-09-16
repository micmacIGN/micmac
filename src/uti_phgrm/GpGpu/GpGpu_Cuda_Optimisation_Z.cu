#ifndef _OPTIMISATION_KERNEL_Z_H_
#define _OPTIMISATION_KERNEL_Z_H_

#include "GpGpu/GpGpu_StreamData.cuh"
#include "GpGpu/SData2Optimize.h"

// On pourrait imaginer un buffer des tailles calculer en parallel
// SIZEBUFFER[threadIdx.x] = count(lI[threadIdx.x]);

__device__ void GetConeZ(short2 & aDz, int aZ, int MaxDeltaZ, short2 aZ_Next, short2 aZ_Prev)
{
    aDz.x =   aZ_Prev.x-aZ;
    if (aZ != aZ_Next.x)
        aDz.x = max(aDz.x,-MaxDeltaZ);

    aDz.y = aZ_Prev.y-1-aZ;
    if (aZ != aZ_Next.y-1)
        aDz.y = min(aDz.y,MaxDeltaZ);

    if (aDz.x > aDz.y)
        if (aDz.y <0)
            aDz.x = aDz.y;
        else
            aDz.y = aDz.x;
}

__device__ void BasicComputeIntervaleDelta
              (
                  short2 & aDz,
                  int aZ,
                  int MaxDeltaZ,
                  short2 aZ_Prev
              )
{
   aDz.x = max(-MaxDeltaZ,aZ_Prev.x-aZ);
   aDz.y = min(MaxDeltaZ,aZ_Prev.y-1-aZ);
}

inline __device__ uint minR(uint *sMin, uint &globalMin){ // TODO attention ajout de inline
    ushort  thread2;
    uint    temp;
    //

    int nTotalThreads = WARPSIZE;	// Total number of threads, rounded up to the next power of two

    while(nTotalThreads > 1)
    {
        int halfPoint = (nTotalThreads >> 1);	// divide by two
        // only the first half of the threads will be active.

        if (threadIdx.x < halfPoint)
        {
            thread2 = threadIdx.x + halfPoint;
            // Skipping the fictious threads blockDim.x ... blockDim_2-1
            if (thread2 < blockDim.x)
            {
                // Get the shared value stored by another thread
                temp = sMin[thread2];
                if (temp < sMin[threadIdx.x])
                    sMin[threadIdx.x] = temp;
            }
        }
        // Reducing the binary tree size by two:
        nTotalThreads = halfPoint;
    }

    const uint minus = sMin[0];

    if(minus < globalMin) globalMin = minus;

    return minus;
}

template<bool sens> __device__
void RunLine(   SimpleStream<short2>    &streamIndex,
                SimpleStream<uint>      &streamFCost,
                SimpleStream<ushort>    &streamICost,
                short2     *S_Bf_Index,
                ushort     *ST_Bf_ICost,
                uint       *S_FCost[2],
                ushort     &sId_ICost,
                uint        penteMax,
                int        lenghtLine,
                short2     &prevIndex,
                int        &id_Line,
                short      &idSeg,
                bool       &idBuf
)
{
    const ushort  tid   = threadIdx.x;
    short2* ST_Bf_Index = S_Bf_Index + tid + (sens ? 0 : -WARPSIZE + 1);
    short2  ConeZ;
    uint    segLine     = 0;

    __shared__ uint globMinFCost;

    while(id_Line < lenghtLine)
    {

        segLine = min(lenghtLine-id_Line,WARPSIZE);

        while(idSeg < segLine)
        {

            const short2 index  = S_Bf_Index[sgn(idSeg)];
            const ushort dZ     = count(index); // creer buffer de count
            ushort       z      = 0;
            globMinFCost        = max_cost;

            //if(sens)
                while( z < dZ)
                {
                    if(sId_ICost > NAPPEMAX)
                    {
                        streamICost.read<sens>(ST_Bf_ICost);
                        streamFCost.incre<sens>();
                        sId_ICost = 0;
                    }

                    uint fCostMin           = max_cost;
                    const ushort costInit   = ST_Bf_ICost[sgn(sId_ICost)];
                    const ushort tZ         = z + (sens ? tid : (WARPSIZE - tid));
                    const short  Z          = sens ? tZ + index.x : index.y - tZ;
                    const short pitPrZ      = sens ? Z - prevIndex.x : prevIndex.y - Z;

                    GetConeZ(ConeZ,Z,penteMax,index,prevIndex);

                    uint* prevFCost = S_FCost[idBuf] + sgn(pitPrZ);

                    ConeZ.y = min(NAPPEMAX - pitPrZ,ConeZ.y );

                    for (short i = ConeZ.x; i <= ConeZ.y; ++i)
                        fCostMin = min(fCostMin, costInit + prevFCost[i]); // ERROR

                    const uint fcost    =  fCostMin + sens * (streamFCost.GetValue(sId_ICost) - costInit);

                    if( tZ < NAPPEMAX)
                    {
                        S_FCost[!idBuf][sgn(tZ)] = fCostMin;
                        //if(sens)
                        streamFCost.SetValue(sgn(sId_ICost), fcost); // ERROR

                        if(!sens)
                            atomicMin(&globMinFCost,fcost);
                    }

                    const ushort pIdCost = sId_ICost;
                    sId_ICost += min(dZ - z,WARPSIZE);
                    z         += min(WARPSIZE,NAPPEMAX-pIdCost);
                }

            prevIndex = index;
            idSeg++;
            idBuf =!idBuf;
        }

        id_Line += segLine;

        if(id_Line < lenghtLine)
            streamIndex.read<sens>(ST_Bf_Index);

        idSeg   = 0;
    }

    idSeg = segLine - 1;
}

template<class T> __global__
void Run(ushort* g_ICost, short2* g_Index, uint* g_FCost, uint3* g_RecStrParam, uint penteMax)
{
    __shared__ short2   S_BuffIndex[WARPSIZE];
    __shared__ ushort   S_BuffICost0[NAPPEMAX + 2*WARPSIZE];
    __shared__ uint     S_BuffFCost0[NAPPEMAX + 2*WARPSIZE];
    __shared__ uint     S_BuffFCost1[NAPPEMAX + 2*WARPSIZE];
    __shared__ uint     pit_Id;
    __shared__ uint     pit_Stream;
    __shared__ int      lenghtLine;

    const ushort    tid     = threadIdx.x;

    uint*    S_BuffFCost[2] = {S_BuffFCost0 + WARPSIZE,S_BuffFCost1 + WARPSIZE};
    ushort*  S_BuffICost    = S_BuffICost0 + WARPSIZE + tid;

    bool            idBuf   = false;
    ushort          s_id_Icost;

    if(!threadIdx.x)
    {
        uint3 recStrParam   = g_RecStrParam[blockIdx.x];
        pit_Stream          = recStrParam.x;
        pit_Id              = recStrParam.y;
        lenghtLine          = recStrParam.z;
    }

    __syncthreads();

    SimpleStream<ushort>    streamICost(g_ICost + pit_Stream,NAPPEMAX);
    SimpleStream<uint>      streamFCost(g_FCost + pit_Stream,NAPPEMAX);
    SimpleStream<short2>    streamIndex(g_Index + pit_Id    ,WARPSIZE);

    streamICost.read<eAVANT>(S_BuffICost);

    uint*   locFCost = S_BuffFCost[idBuf] + tid;
    for (ushort i = 0; i < NAPPEMAX; i+=WARPSIZE)
        locFCost[i] = S_BuffICost[i];

    streamIndex.read<eAVANT>(S_BuffIndex + tid);

    short2  prevIndex   = S_BuffIndex[0];
    int     id_Line     = 0;
    short   idSeg       = 1;

    s_id_Icost   = count(prevIndex);

    RunLine<eAVANT>(streamIndex,streamFCost,streamICost,S_BuffIndex,S_BuffICost,S_BuffFCost,s_id_Icost,penteMax,lenghtLine,prevIndex,id_Line,idSeg,idBuf);

    streamIndex.reverse<eARRIERE>();
    streamFCost.reverse<eARRIERE>();

    S_BuffFCost[0]  += NAPPEMAX - WARPSIZE;
    S_BuffFCost[1]  += NAPPEMAX - WARPSIZE;
    S_BuffICost     += NAPPEMAX - WARPSIZE;

    streamICost.readFrom<eARRIERE>(S_BuffFCost[idBuf] + tid, NAPPEMAX - s_id_Icost);

    streamICost.reverse<eARRIERE>();

    prevIndex       = S_BuffIndex[idSeg];
    idSeg           = WARPSIZE - idSeg;
    id_Line         = 1;

    const short noRead   = count(prevIndex) - s_id_Icost;

    if(noRead < 0)
        s_id_Icost = NAPPEMAX + noRead;
    else
    {
        streamICost.read<eARRIERE>(S_BuffICost);
        streamFCost.incre<eARRIERE>();
        s_id_Icost = noRead;
    }

    RunLine<eARRIERE>(  streamIndex,
                        streamFCost,
                        streamICost,
                        S_BuffIndex + WARPSIZE - 1,
                        S_BuffICost,
                        S_BuffFCost,
                        s_id_Icost,
                        penteMax,
                        lenghtLine + idSeg,
                        prevIndex,
                        id_Line,
                        idSeg,
                        idBuf);

}

template<bool sens> __device__
void ReadLine(
                SimpleStream<short2>    &streamIndex,
                SimpleStream<uint>      &streamFCost,
                SimpleStream<ushort>    &streamICost,
                short2     *S_Bf_Index,
                ushort     *ST_Bf_ICost,
                uint       *S_FCost[2],
                p_ReadLine &p
)
{

    short2* ST_Bf_Index = S_Bf_Index + p.tid + (sens ? 0 : -WARPSIZE + 1);

    while(p.line.id < p.line.lenght)
    {
        while(p.seg.id < p.seg.lenght)
        {
            const short2 index  = S_Bf_Index[sgn(p.seg.id)];
            const ushort dZ     = count(index); // creer buffer de count
            ushort       z      = 0;

            while( z < dZ)
            {
                if(p.ID_Bf_Icost >= NAPPEMAX) // VERIFIER si > ou >=
                {
                    streamICost.read<sens>(ST_Bf_ICost);
                    streamFCost.incre<sens>();
                    p.ID_Bf_Icost = 0;
                }

                if(z + (sens ? p.tid : WARPSIZE - p.tid - 1)< dZ) // peut etre eviter en ajoutant une zone tampon
                {
                    const ushort costInit  = ST_Bf_ICost[sgn(p.ID_Bf_Icost)];
                    streamFCost.SetValue(sgn(p.ID_Bf_Icost),costInit);
                }

                const ushort pIdCost = p.ID_Bf_Icost;
                p.ID_Bf_Icost       += min(dZ - z           , WARPSIZE);
                z                   += min(NAPPEMAX-pIdCost , WARPSIZE);
            }

            p.prev_Dz = index;
            p.seg.id++;
            p.swBuf();
        }

        p.line.id += p.seg.lenght;

        if(p.line.id < p.line.lenght)
        {
            streamIndex.read<sens>(ST_Bf_Index);
            p.seg.lenght  = min(p.line.LOver(),WARPSIZE);
            p.seg.id      = 0;
        }
    }
}

template<bool sens> __device__
void ReadLine_V02(
                SimpleStream<short2>    &streamIndex,
                SimpleStream<uint>      &streamFCost,
                SimpleStream<ushort>    &streamICost,
                SimpleStream<uint>      &streamDefCor,
                short2     *S_Bf_Index,
                ushort     *ST_Bf_ICost,
                uint       *S_FCost[2],
                p_ReadLine &p
)
{

    short2* ST_Bf_Index = S_Bf_Index + p.tid + (sens ? 0 : -WARPSIZE + 1);

    __shared__ uint minCost[WARPSIZE];
    short2  ConeZ;
    uint globMinFCost;

    bool lined = p.line.id < p.line.lenght;

    const int regulZ  = (int)((float)10000.f*p.ZRegul);

    // Remarque
    // p.seg.id = 1 au premier passage, car simple copie des initcost

    #ifdef CUDA_DEFCOR
    //////////////////////////////////////////////////
    /// TODO!!!! : quel doit etre prevDefCor p.costTransDefMask + p.costDefMask ou p.costDefMask
    /////////////////////////////////////////////////
    uint         prevDefCor   =/* p.costTransDefMask + */p.costDefMask;
    const ushort idGline = p.line.id + p.seg.id;

    streamDefCor.SetOrAddValue<sens>(sens ? idGline : p.line.lenght  - idGline,prevDefCor);
    #endif

    uint         prevMinCost  = 0;

    while(lined)
    {
        while(p.seg.id < p.seg.lenght)
        {
            const short2 indexZ = S_Bf_Index[sgn(p.seg.id)];
            const ushort dZ     = count(indexZ); // creer buffer de count
            ushort       z      = 0;
            globMinFCost        = max_cost;
#ifdef CUDA_DEFCOR
            bool           mask = false;
#endif
            while( z < dZ)
            {                
                // Lecture du stream si le buffer est vide | TODO VERIFIER si > ou >=
                if(p.ID_Bf_Icost >= p.sizeBuffer)
                {
                    streamICost.read<sens>(ST_Bf_ICost);    //  Lecture des couts correlations
                    streamFCost.incre<sens>();              //  Pointage sur la sortie
                    p.ID_Bf_Icost = 0;                      //  Pointage la première valeur du buffer des couts correlations
                }

                uint    fCostMin        = max_cost;
                uint  costInit        = ST_Bf_ICost[sgn(p.ID_Bf_Icost)];
                const ushort tZ         = z + p.stid<sens>();
                const short  Z          = ((sens) ? tZ + indexZ.x : indexZ.y - tZ - 1);
                const short  pitPrZ     = ((sens) ? Z - p.prev_Dz.x : p.prev_Dz.y - Z - 1);

#ifdef CUDA_DEFCOR
                if(costInit < 55000)                
                    BasicComputeIntervaleDelta(ConeZ,Z,p.pente,p.prev_Dz);                
                else
                {                  
                    costInit = 500000;
                    mask = true;
                    BasicComputeIntervaleDelta(ConeZ,Z,0,p.prev_Dz);                    
                }
#else
                GetConeZ(ConeZ,Z,p.pente,indexZ,p.prev_Dz);
#endif
                uint* prevFCost = S_FCost[p.Id_Buf] + sgn(pitPrZ);

                ConeZ.y = min(p.sizeBuffer - pitPrZ,ConeZ.y );

                for (short i = ConeZ.x; i <= ConeZ.y; ++i)
                    fCostMin = min(fCostMin, costInit + prevFCost[i] + abs((int)i)*regulZ);


#ifdef CUDA_DEFCOR
                // NOTE DEFCOR
                // LES PROBLEMES
                    // les cellules dans la zone masquée
                    // les cellules dont la valeur le coef de corrélation n'a pas été calculé -> 1.01234 --> 10123
                    //  ces cellules contaminent les voisines en mode DEFCOR....
                if(!mask)
                    fCostMin = min(fCostMin, costInit + prevDefCor  + p.costTransDefMask );
                else
                    fCostMin = min(fCostMin, 20*costInit + prevDefCor  + p.costTransDefMask );
#endif

                if(tZ < dZ && p.ID_Bf_Icost +  p.stid<sens>() < p.sizeBuffer && tZ < p.sizeBuffer)
                {                    

                    fCostMin                    -= prevMinCost;
                    minCost[p.tid]               = fCostMin;
                    S_FCost[!p.Id_Buf][sgn(tZ)]  = fCostMin;

                    streamFCost.SetOrAddValue<sens>(sgn(p.ID_Bf_Icost),fCostMin,fCostMin - costInit);                    
                }
                else
                    minCost[p.tid] = max_cost;

                //if(!sens) // recherche du cout minimum
                {
                    //------------------------------- fcost remplacer par fCostMin
                    //                            |
                    //                            V
                    //minCost[p.tid] = inside ? fCostMin : max_cost;
                    minR(minCost,globMinFCost); // TODO verifier cette fonction elle peut lancer trop de fois..... Attentioncd ,inline en attendant
                }

                const ushort pIdCost = p.ID_Bf_Icost;
                p.ID_Bf_Icost       += min(dZ - z           , WARPSIZE);
                z                   += min(p.sizeBuffer-pIdCost , WARPSIZE);

            }

#ifdef CUDA_DEFCOR

            prevDefCor  = min(prevDefCor,prevMinCost + p.costTransDefMask )+ p.costDefMask - prevMinCost;
            prevMinCost = min(globMinFCost,prevDefCor);

            if(p.tid == 0)
            {
                const ushort idGline = p.line.id + p.seg.id;

                streamDefCor.SetOrAddValue<sens>(sens ? idGline : p.line.lenght  - idGline,prevDefCor);

            }
#else
            prevMinCost = globMinFCost;
#endif

            p.prev_Dz = indexZ;
            p.seg.id++;
            p.swBuf();

//            if(!sens) // retranche la cout minimum à toutes les cellules de même coordonnées terrain
//            {
//                const short piSFC = -p.ID_Bf_Icost + dZ ;
//                for (ushort i = 0; i < dZ - p.stid<sens>(); i+=WARPSIZE)
//                    streamFCost.SubValue(piSFC - i,globMinFCost);
//            }
        }

        p.line.id += p.seg.lenght;

        lined = p.line.id < p.line.lenght;

        if(lined)
        {
            streamIndex.read<sens>(ST_Bf_Index);
            p.seg.lenght  = min(p.line.LOver(),WARPSIZE);
            p.seg.id      = 0; // position dans le segment du stream index des Z
        }
    }
}

// TODO Passer les parametres en variable constante !!!!!!!!!!!

template<class T> __global__
void Run_V02(ushort* g_ICost, short2* g_Index, uint* g_FCost, uint* g_DefCor, uint3* g_RecStrParam, uint penteMax, float zReg,float zRegQuad, ushort costDefMask,ushort costTransDefMask,ushort sizeBuffer)
{

    extern __shared__ float sharedMemory[];

    ushort*   S_BuffICost0 = (ushort*)  sharedMemory;
    uint*     S_BuffFCost0 = (uint*)    &S_BuffICost0[sizeBuffer + 2*WARPSIZE];
    uint*     S_BuffFCost1 = (uint*)    &S_BuffFCost0[sizeBuffer + 2*WARPSIZE];
    short2*   S_BuffIndex  = (short2*)  &S_BuffFCost1[sizeBuffer + 2*WARPSIZE];
    uint*     pit_Id       = (uint*)    &S_BuffIndex[WARPSIZE];
    uint*     pit_Stream   = pit_Id + 1;

    p_ReadLine p(threadIdx.x,penteMax,zReg,zRegQuad,costDefMask,costTransDefMask,sizeBuffer);

    uint*    S_BuffFCost[2] = {S_BuffFCost0 + WARPSIZE,S_BuffFCost1 + WARPSIZE};
    ushort*  S_BuffICost    = S_BuffICost0 + WARPSIZE + p.tid;

    if(!threadIdx.x)
    {
        *pit_Stream          = g_RecStrParam[blockIdx.x].x;
        *pit_Id              = g_RecStrParam[blockIdx.x].y;
    }

    __syncthreads();

    p.line.lenght   = g_RecStrParam[blockIdx.x].z;
    p.seg.lenght    = min(p.line.LOver(),WARPSIZE);

    SimpleStream<ushort>    streamICost(    g_ICost     + *pit_Stream   ,sizeBuffer);
    SimpleStream<uint>      streamFCost(    g_FCost     + *pit_Stream   ,sizeBuffer);
    SimpleStream<short2>    streamIndex(    g_Index     + *pit_Id       ,WARPSIZE);
    SimpleStream<uint>      streamDefCor(   g_DefCor    + *pit_Id       ,WARPSIZE);

   if(p.tid == 0)
        streamDefCor.SetValue(0,0); // car la premiere ligne n'est calculer
                                    // Attention voir pour le retour arriere

    streamICost.read<eAVANT>(S_BuffICost);
    streamIndex.read<eAVANT>(S_BuffIndex + p.tid);

    p.prev_Dz       = S_BuffIndex[0];
    p.ID_Bf_Icost   = count(p.prev_Dz);

    for (ushort i = 0; i < p.ID_Bf_Icost - p.tid; i+=WARPSIZE)
    {
        S_BuffFCost[p.Id_Buf][i + p.tid] = S_BuffICost[i];
        streamFCost.SetValue(i,S_BuffICost[i]);
    }

    ReadLine_V02<eAVANT>(streamIndex,streamFCost,streamICost,streamDefCor,S_BuffIndex,S_BuffICost,S_BuffFCost,p);

    streamIndex.ReverseIncre<eARRIERE>();
    streamFCost.incre<eAVANT>();
    streamFCost.reverse<eARRIERE>();

    S_BuffFCost[0]  += sizeBuffer;
    S_BuffFCost[1]  += sizeBuffer;
    S_BuffICost     += sizeBuffer - WARPSIZE;

    streamICost.readFrom<eARRIERE>(S_BuffFCost[p.Id_Buf] + p.tid, sizeBuffer - p.ID_Bf_Icost);
    streamICost.ReverseIncre<eARRIERE>();

    p.reverse(S_BuffIndex,sizeBuffer);

    if(p.ID_Bf_Icost > sizeBuffer)
    {
        p.ID_Bf_Icost -= sizeBuffer;
        streamICost.read<eARRIERE>(S_BuffICost);
        streamFCost.incre<eARRIERE>();
    }

    uint* locFCost = S_BuffFCost[p.Id_Buf] - p.stid<eARRIERE>();

    for (ushort i = 0; i < sizeBuffer; i+=WARPSIZE)
        locFCost[-i] = S_BuffICost[-i];

    ReadLine_V02<eARRIERE>( streamIndex,streamFCost,streamICost,streamDefCor,S_BuffIndex + WARPSIZE - 1,S_BuffICost,S_BuffFCost,p);
}

extern "C" void OptimisationOneDirectionZ_V02(Data2Optimiz<CuDeviceData3D> &d2O)
{
    uint deltaMax   = d2O.penteMax();
    float zReg      = (float)d2O.zReg();
    float zRegQuad  = d2O.zRegQuad();
    ushort costDefMask = d2O.CostDefMasked();
    ushort costTransDefMask = d2O.CostTransMaskNoMask();

//    DUMP_FLOAT(zReg);
//    DUMP_FLOAT(zRegQuad);

    dim3 Threads(WARPSIZE,1,1);
    dim3 Blocks(d2O.NBlines(),1,1);


    ushort sizeBuff = d2O.DzMax();  //NAPPEMAX;
    ushort cacheLin = sizeBuff + 2 * WARPSIZE;

    uint   sizeSharedMemory =
            cacheLin * sizeof(ushort)   + // S_BuffICost0
            cacheLin * sizeof(uint)     + // S_BuffFCost0
            cacheLin * sizeof(uint)     + // S_BuffFCost1
            WARPSIZE * sizeof(short2)   + // S_BuffIndex
          //  WARPSIZE * sizeof(uint)     + // S_BuffDefCor
            sizeof(uint)                + // pit_Id
            sizeof(uint);                 // pit_Stream


    Run_V02< uint ><<<Blocks,Threads,sizeSharedMemory>>>
                                                       (
                                                           d2O.pInitCost(),
                                                           d2O.pIndex(),
                                                           d2O.pForceCostVol(),
                                                           d2O.pDefCor(),
                                                           d2O.pParam(),
                                                           deltaMax,
                                                           zReg,
                                                           zRegQuad,
                                                           costDefMask,
                                                           costTransDefMask,
                                                           sizeBuff
                                                           );

    cudaError_t err = cudaGetLastError();

    if (cudaSuccess != err)
    {
        // BUG : invalid configuration argument!!
        printf("Error CUDA OptimisationOneDirectionZ_V02\n");
        printf("%s",cudaGetErrorString(err));

        //DUMP_INT(WARPSIZE)
        //DUMP_INT(d2O.NBlines())
    }

    getLastCudaError("TestkernelOptiOneDirection failed");

}

extern "C" void OptimisationOneDirectionZ_V01(Data2Optimiz<CuDeviceData3D> &d2O)
{
    uint deltaMax = 3;
    dim3 Threads(WARPSIZE,1,1);
    dim3 Blocks(d2O.NBlines(),1,1);

    Run< uint ><<<Blocks,Threads>>>
                                  (
                                      d2O.pInitCost(),
                                      d2O.pIndex(),
                                      d2O.pForceCostVol(),
                                      d2O.pParam(),
                                      deltaMax
                                      );
    getLastCudaError("kernelOptiOneDirection failed");
}

#endif //_OPTIMISATION_KERNEL_Z_H_

