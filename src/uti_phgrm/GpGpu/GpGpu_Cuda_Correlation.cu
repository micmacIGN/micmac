#include "GpGpu/GpGpu_ParamCorrelation.cuh"
#include "GpGpu/GpGpu_TextureTools.cuh"
#include "GpGpu/GpGpu_TextureCorrelation.cuh"
#include "GpGpu/SData2Correl.h"


/// \file       GpGpuCudaCorrelation.cu
/// \brief      Kernel
/// \author     GC
/// \version    0.2
/// \date       mars 2013

static __constant__ invParamCorrel  invPc;

extern "C" void CopyParamInvTodevice( pCorGpu param )
{
  checkCudaErrors(cudaMemcpyToSymbol(invPc, &param.invPC, sizeof(invParamCorrel)));
}

template<int TexSel> __global__ void projectionImage( HDParamCorrel HdPc, float* projImages)
{
    //extern __shared__ float cacheImg[];

    const uint2 ptHTer = make_uint2(blockIdx) * BLOCKDIM + make_uint2(threadIdx);

    if (oSE(ptHTer,HdPc.dimDTer)) return;

    const ushort IdLayer = blockDim.z * blockIdx.z + threadIdx.z;

    const float2 ptProj  = GetProjection<TexSel>(ptHTer,invPc.sampProj, IdLayer);

    float* localImages = projImages + IdLayer * size(HdPc.dimDTer);

    localImages[to1D(ptHTer,HdPc.dimDTer)] = ptProj.x < 0 ? -1.f : GetImageValue(ptProj,threadIdx.z);
}

extern "C" void	 LaunchKernelprojectionImage(pCorGpu &param, CuDeviceData3D<float>  &DeviImagesProj)
{

    dim3	threads( BLOCKDIM, BLOCKDIM, param.invPC.nbImages);
    uint2	thd2D		= make_uint2(threads);
    uint2	block2D		= iDivUp(param.HdPc.dimDTer,thd2D);
    dim3	blocks(block2D.x , block2D.y, param.ZCInter);

    DeviImagesProj.ReallocIfDim(param.HdPc.dimDTer,param.ZCInter*param.invPC.nbImages);

//    CuHostData3D<float>  hostImagesProj;

//    hostImagesProj.ReallocIfDim(param.HdPc.dimDTer,param.ZCInter*param.invPC.nbImages);

//    projectionImage<0><<<blocks, threads, param.invPC.nbImages * BLOCKDIM * BLOCKDIM * sizeof(float), 0>>>(param.HdPc,DeviImagesProj.pData());

    projectionImage<0><<<blocks, threads>>>(param.HdPc,DeviImagesProj.pData());

//    DeviImagesProj.CopyDevicetoHost(hostImagesProj);

//    hostImagesProj.OutputValues();

//    GpGpuTools::Array1DtoImageFile(hostImagesProj.pData(),"toto.ppm",hostImagesProj.GetDimension());
//    hostImagesProj.Dealloc();

}

/// \fn template<int TexSel> __global__ void correlationKernel( uint *dev_NbImgOk, float* cachVig, uint2 nbActThrd)
/// \brief Kernel fonction GpGpu Cuda
/// Calcul les vignettes de correlation pour toutes les images
///
template<int TexSel> __global__ void correlationKernel( uint *dev_NbImgOk, float* cachVig, uint2 nbActThrd,HDParamCorrel HdPc)
{

  extern __shared__ float cacheImg[];

  // Coordonnées du terrain global avec bordure // __umul24!!!! A voir

  const uint2 ptHTer = make_uint2(blockIdx) * nbActThrd + make_uint2(threadIdx);

  // Si le processus est hors du terrain, nous sortons du kernel

  if (oSE(ptHTer,HdPc.dimDTer)) return;

  const float2 ptProj = GetProjection<TexSel>(ptHTer,invPc.sampProj,blockIdx.z);

  uint pitZ,modZ,piCa;

  if (oI(ptProj,0)) // retirer le 9 decembre 2013 à verifier
  {
      return;
  }
  else
  {
      pitZ  = blockIdx.z / invPc.nbImages;

      piCa  = pitZ * invPc.nbImages;

      modZ  = blockIdx.z - piCa;

      cacheImg[threadIdx.y*BLOCKDIM + threadIdx.x] = GetImageValue(ptProj,modZ);
  }

  __syncthreads();

  const int2 ptTer = make_int2(ptHTer) - make_int2(invPc.rayVig);

  // Nous traitons uniquement les points du terrain du bloque ou Si le processus est hors du terrain global, nous sortons du kernel

  // Simplifier!!!
  if (oSE(threadIdx, nbActThrd + invPc.rayVig) || oI(threadIdx , invPc.rayVig) || oSE( ptTer, HdPc.dimTer) || oI(ptTer, 0))
    return;


  // INCORRECT !!!
  if(tex2D(TexS_MaskGlobal, ptTer.x + HdPc.rTer.pt0.x , ptTer.y + HdPc.rTer.pt0.y) == 0) return;

  const short2 c0	= make_short2(threadIdx) - invPc.rayVig;
  const short2 c1	= make_short2(threadIdx) + invPc.rayVig;

  // Intialisation des valeurs de calcul
  float aSV = 0.0f, aSVV = 0.0f;
  short2 pt;

  #pragma unroll // ATTENTION PRAGMA FAIT AUGMENTER LA quantité MEMOIRE des registres!!!
  for (pt.y = c0.y ; pt.y <= c1.y; pt.y++)
  {
        //const int pic = pt.y*BLOCKDIM;
        float* cImg    = cacheImg +  pt.y*BLOCKDIM;
      #pragma unroll
      for (pt.x = c0.x ; pt.x <= c1.x; pt.x++)
      {
          const float val = cImg[pt.x];	// Valeur de l'image
          //        if (val ==  cH.floatDefault) return;
          aSV  += val;          // Somme des valeurs de l'image cte
          aSVV += (val*val);	// Somme des carrés des vals image cte
      }
  }

  aSV   = fdividef(aSV,(float)invPc.sizeVig );

  aSVV  = fdividef(aSVV,(float)invPc.sizeVig );

  aSVV -=	(aSV * aSV);

  if ( aSVV <= invPc.mAhEpsilon) return;

  aSVV =	rsqrtf(aSVV); // racine carre inverse

  const uint pitchCachY = ptTer.y * invPc.dimVig.y ;

  const int idN     = pitZ * HdPc.sizeTer + to1D(ptTer,HdPc.dimTer);

  const uint iCa    = atomicAdd( &dev_NbImgOk[idN], 1U) + piCa;

  float* cache      = cachVig + iCa * HdPc.sizeCach + ptTer.x * invPc.dimVig.x - c0.x + (pitchCachY - c0.y)* HdPc.dimCach.x;

#pragma unroll
  for ( pt.y = c0.y ; pt.y <= c1.y; pt.y++)
    {
      float* cImg = cacheImg + pt.y * BLOCKDIM;
      float* cVig = cache    + pt.y * HdPc.dimCach.x ;
#pragma unroll
      for ( pt.x = c0.x ; pt.x <= c1.x; pt.x++)
        cVig[ pt.x ] = (cImg[pt.x] -aSV)*aSVV;
    }
}

/// \brief Fonction qui lance les kernels de correlation
extern "C" void	 LaunchKernelCorrelation(const int s,cudaStream_t stream,pCorGpu &param,SData2Correl &data2cor)
{

    dim3	threads( BLOCKDIM, BLOCKDIM, 1);
    uint2	thd2D		= make_uint2(threads);
    uint2	nbActThrd	= thd2D - 2 * param.invPC.rayVig;
    uint2	block2D		= iDivUp(param.HdPc.dimDTer,nbActThrd);
    dim3	blocks(block2D.x , block2D.y, param.invPC.nbImages * param.ZCInter);

  switch (s)
    {
    case 0:      
      correlationKernel<0><<<blocks, threads, BLOCKDIM * BLOCKDIM * sizeof(float), stream>>>( data2cor.DeviVolumeNOK(0), data2cor.DeviVolumeCache(0), nbActThrd,param.HdPc);
      getLastCudaError("Basic Correlation kernel failed stream 0");
      break;
    case 1:
      correlationKernel<1><<<blocks, threads, BLOCKDIM * BLOCKDIM* sizeof(float), stream>>>( data2cor.DeviVolumeNOK(1), data2cor.DeviVolumeCache(1), nbActThrd,param.HdPc);
      getLastCudaError("Basic Correlation kernel failed stream 1");
      break;
    }
}


template<int TexSel> __global__ void correlationKernelZ( uint *dev_NbImgOk, float* cachVig, uint2 nbActThrd,float* imagesProj,HDParamCorrel HdPc)
{

    extern __shared__ float cacheImgLayered[];

    float* cacheImg = cacheImgLayered + threadIdx.z * BLOCKDIM * BLOCKDIM;

    // Coordonnées du terrain global avec bordure // __umul24!!!! A voir

    const uint2 ptHTer = make_uint2(blockIdx) * nbActThrd + make_uint2(threadIdx);

    // Si le processus est hors du terrain, nous sortons du kernel

    if (oSE(ptHTer,HdPc.dimDTer)) return;

    const ushort pitImages = blockIdx.z * invPc.nbImages;

    const float v = cacheImg[threadIdx.y*BLOCKDIM + threadIdx.x] = imagesProj[ ( pitImages + threadIdx.z) * size(HdPc.dimDTer) + to1D(ptHTer,HdPc.dimDTer) ];

    if(v < 0)
        return;

    __syncthreads();

    const int2 ptTer = make_int2(ptHTer) - make_int2(invPc.rayVig);

    // Nous traitons uniquement les points du terrain du bloque ou Si le processus est hors du terrain global, nous sortons du kernel

    // Simplifier!!!
    if (oSE(threadIdx, nbActThrd + invPc.rayVig) || oI(threadIdx , invPc.rayVig) || oSE( ptTer, HdPc.dimTer) || oI(ptTer, 0))
      return;


    // INCORRECT !!!
    if(tex2D(TexS_MaskGlobal, ptTer.x + HdPc.rTer.pt0.x , ptTer.y + HdPc.rTer.pt0.y) == 0) return;

    const short2 c0	= make_short2(threadIdx) - invPc.rayVig;
    const short2 c1	= make_short2(threadIdx) + invPc.rayVig;

    // Intialisation des valeurs de calcul
    float aSV = 0.0f, aSVV = 0.0f;
    short2 pt;

    #pragma unroll // ATTENTION PRAGMA FAIT AUGMENTER LA quantité MEMOIRE des registres!!!
    for (pt.y = c0.y ; pt.y <= c1.y; pt.y++)
    {
          //const int pic = pt.y*BLOCKDIM;
          float* cImg    = cacheImg +  pt.y*BLOCKDIM;
        #pragma unroll
        for (pt.x = c0.x ; pt.x <= c1.x; pt.x++)
        {
            const float val = cImg[pt.x];	// Valeur de l'image
            //        if (val ==  cH.floatDefault) return;
            aSV  += val;          // Somme des valeurs de l'image cte
            aSVV += (val*val);	// Somme des carrés des vals image cte
        }
    }

    aSV   = fdividef(aSV,(float)invPc.sizeVig );

    aSVV  = fdividef(aSVV,(float)invPc.sizeVig );

    aSVV -=	(aSV * aSV);

    if ( aSVV <= invPc.mAhEpsilon) return;

    aSVV =	rsqrtf(aSVV); // racine carre inverse

    const uint pitchCachY = ptTer.y * invPc.dimVig.y ;

    const int idN     = blockIdx.z * HdPc.sizeTer + to1D(ptTer,HdPc.dimTer);

    float* cache      = cachVig + (atomicAdd( &dev_NbImgOk[idN], 1U) + pitImages) * HdPc.sizeCach + ptTer.x * invPc.dimVig.x - c0.x + (pitchCachY - c0.y)* HdPc.dimCach.x;

  #pragma unroll
    for ( pt.y = c0.y ; pt.y <= c1.y; pt.y++)
      {
        float* cImg = cacheImg + pt.y * BLOCKDIM;
        float* cVig = cache    + pt.y * HdPc.dimCach.x ;
  #pragma unroll
        for ( pt.x = c0.x ; pt.x <= c1.x; pt.x++)
          cVig[ pt.x ] = (cImg[pt.x] -aSV)*aSVV;
      }
}

/// \brief Fonction qui lance les kernels de correlation
extern "C" void	 LaunchKernelCorrelationZ(const int s,pCorGpu &param,SData2Correl &data2cor)
{

    dim3	threads( BLOCKDIM, BLOCKDIM, param.invPC.nbImages);
    uint2	thd2D		= make_uint2(threads);
    uint2	nbActThrd	= thd2D - 2 * param.invPC.rayVig;
    uint2	block2D		= iDivUp(param.HdPc.dimDTer,nbActThrd);
    dim3	blocks(block2D.x , block2D.y, param.ZCInter);

    CuDeviceData3D<float>       DeviImagesProj;

    //const ushort HBLOCKDIM = BLOCKDIM + param.invPC.rayVig.x;

    LaunchKernelprojectionImage(param,DeviImagesProj);

    correlationKernelZ<0><<<blocks, threads, param.invPC.nbImages * BLOCKDIM * BLOCKDIM * sizeof(float), 0>>>(
                                                                                           data2cor.DeviVolumeNOK(0),
                                                                                           data2cor.DeviVolumeCache(0),
                                                                                           nbActThrd,
                                                                                           DeviImagesProj.pData(),
                                                                                           param.HdPc);
    getLastCudaError("Basic Correlation kernel failed stream 0");

}


/// \brief Kernel Calcul "rapide"  de la multi-correlation en utilisant la formule de Huygens n utilisant pas des fonctions atomiques
__global__ void multiCorrelationKernel(float *dTCost, float* cacheVign, uint* dev_NbImgOk, uint2 nbActThr,HDParamCorrel HdPc)
{

  __shared__ float aSV [ SBLOCKDIM ][ SBLOCKDIM ];          // Somme des valeurs
  __shared__ float aSVV[ SBLOCKDIM  ][ SBLOCKDIM ];         // Somme des carrés des valeurs
  __shared__ float resu[ SBLOCKDIM/2 ][ SBLOCKDIM/2 ];		// resultat
  //__shared__ ushort nbIm[ SBLOCKDIM/2][ SBLOCKDIM/2 ];		// nombre d'images correcte

  // coordonnées des threads
  const uint2 t = make_uint2(threadIdx);

  aSV [t.y][t.x]        = 0.0f;

  aSVV[t.y][t.x]        = 0.0f;

  resu[t.y/2][t.x/2]	= 0.0f;

  //nbIm[t.y/2][t.x/2]	= 0;

  if ( oSE( t, nbActThr))	return; // si le thread est inactif, il sort

  // Coordonnées 2D du cache vignette
  const uint2 ptCach = make_uint2(blockIdx) * nbActThr + t;

  // Si le thread est en dehors du cache
  if ( oSE(ptCach, HdPc.dimCach))	return;

  const uint2	ptTer	= ptCach / invPc.dimVig; // Coordonnées 2D du terrain

  if(!tex2D(TexS_MaskGlobal, ptTer.x + HdPc.rTer.pt0.x , ptTer.y + HdPc.rTer.pt0.y)) return;

  const uint	iTer	= blockIdx.z * HdPc.sizeTer + to1D(ptTer, HdPc.dimTer);     // Coordonnées 1D dans le terrain avec prise en compte des differents Z

  const uint2   thTer	= t / invPc.dimVig;                                        // Coordonnées 2D du terrain dans le repere des threads

  //if(aEq(t,thTer * cH.dimVig))
  //nbIm[thTer.y][thTer.x] = (ushort)dev_NbImgOk[iTer];/

  const ushort nImgOK = (ushort)dev_NbImgOk[iTer];

  //__syncthreads();

  if ( nImgOK  < 2) return;

  const uint pitLayerCache  = blockIdx.z * HdPc.sizeCachAll + to1D( ptCach, HdPc.dimCach );	// Taille du cache vignette pour une image
  //const uint pit  = blockIdx.z * cH.nbImages;

  float* caVi = cacheVign + pitLayerCache;

 #pragma unroll
  for(uint i = 0;i< nImgOK * HdPc.sizeCach ;i+=HdPc.sizeCach)
  //for(uint l = pit ;l< pit + cH.nbImages;l++)
    {
      const float val  = caVi[i];
      //const float val  = tex2DLayered( TexL_Cache,ptCach.x , ptCach.y,l);

      //if(val!= cH.floatDefault) A verifier si pas d'influence
        //{
          // Coordonnées 1D du cache vignette

          aSV[t.y][t.x]   += val;
          aSVV[t.y][t.x]  += val * val;
        //}
    }

  __syncthreads();

  atomicAdd(&(resu[thTer.y][thTer.x]),aSVV[t.y][t.x] - fdividef(aSV[t.y][t.x] * aSV[t.y][t.x],(float)nImgOK));

  if (!aEq(t - thTer*invPc.dimVig,0)) return;

  __syncthreads();

  // Normalisation pour le ramener a un equivalent de 1-Correl
  const float cost =  fdividef( resu[thTer.y][thTer.x], (float)( nImgOK -1.0f) * (invPc.sizeVig));

  dTCost[iTer] = 1.0f - max (-1.0, min(1.0f,1.0f - cost));

}


/// \brief Fonction qui lance les kernels de multi-Correlation n'utilisant pas des fonctions atomiques
extern "C" void LaunchKernelMultiCorrelation(cudaStream_t stream, pCorGpu &param, SData2Correl &dataCorrel)
{

    //-------------	calcul de dimension du kernel de multi-correlation NON ATOMIC ------------
    uint2	nbActThr	= SBLOCKDIM - make_uint2( SBLOCKDIM % param.invPC.dimVig.x, SBLOCKDIM % param.invPC.dimVig.y);
    dim3	threads(SBLOCKDIM, SBLOCKDIM, 1);
    uint2	block2D	= iDivUp(param.HdPc.dimCach,nbActThr);
    dim3	blocks(block2D.x,block2D.y,param.ZCInter);

    multiCorrelationKernel<<<blocks, threads, 0, stream>>>(dataCorrel.DeviVolumeCost(0), dataCorrel.DeviVolumeCache(0), dataCorrel.DeviVolumeNOK(0), nbActThr,param.HdPc);
    getLastCudaError("Multi-Correlation NON ATOMIC kernel failed");

}
