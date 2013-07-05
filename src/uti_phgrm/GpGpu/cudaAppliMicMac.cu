#include "GpGpu/cudaAppliMicMac.cuh"
#include "GpGpu/cudaTextureTools.cuh"
#include "GpGpu/CudaRefTexture.cuh"


/// \file       cudaAppliMicMac.cu
/// \brief      Kernel
/// \author     GC
/// \version    0.1
/// \date       mars 2013


static __constant__ pCorGpu cH;

/// \fn extern "C" void CopyParamTodevice( pCorGpu param )
/// \brief copie les paramteres du host vers le device
extern "C" void CopyParamTodevice( pCorGpu param )
{
  checkCudaErrors(cudaMemcpyToSymbol(cH, &param, sizeof(pCorGpu)));
}

/// \fn template<int TexSel> __global__ void correlationKernel( uint *dev_NbImgOk, float* cachVig, uint2 nbActThrd)
/// \brief Kernel fonction GpGpu Cuda
/// Calcul les vignettes de correlation pour toutes les images
template<int TexSel> __global__ void correlationKernel( uint *dev_NbImgOk, float* cachVig, uint2 nbActThrd)
{
  __shared__ float cacheImg[ BLOCKDIM ][ BLOCKDIM ];


  // Coordonnées du terrain global avec bordure // __umul24!!!! A voir
  const uint2 ptHTer = make_uint2(blockIdx) * nbActThrd + make_uint2(threadIdx);

  // Si le processus est hors du terrain, nous sortons du kernel
  if (oSE(ptHTer,cH.dimDTer)) return;

  const float2 ptProj = GetProjection<TexSel>(ptHTer,cH.sampProj,blockIdx.z);

  uint pitZ,modZ;

  if (oI(ptProj,0)) return;

  else
    {
      pitZ  = blockIdx.z / cH.nbImages;
      modZ  = blockIdx.z - pitZ * cH.nbImages;
      cacheImg[threadIdx.y][threadIdx.x] = GetImageValue(ptProj,modZ);
    }
  __syncthreads();

  const int2 ptTer = make_int2(ptHTer) - make_int2(cH.rayVig);
  // Nous traitons uniquement les points du terrain du bloque ou Si le processus est hors du terrain global, nous sortons du kernel
  if (oSE(threadIdx, nbActThrd + cH.rayVig) || oI(threadIdx , cH.rayVig) || oSE( ptTer, cH.dimTer) || oI(ptTer, 0))
    return;

  if(tex2D(TexS_MaskTer, ptTer.x, ptTer.y) == 0) return;

  const short2 c0	= make_short2(threadIdx) - cH.rayVig;
  const short2 c1	= make_short2(threadIdx) + cH.rayVig;

  // Intialisation des valeurs de calcul
  float aSV = 0.0f, aSVV = 0.0f;
  short2 pt;

#pragma unroll // ATTENTION PRAGMA FAIT AUGMENTER LA quantité MEMOIRE des registres!!!
  for (pt.y = c0.y ; pt.y <= c1.y; pt.y++)
#pragma unroll
    for (pt.x = c0.x ; pt.x <= c1.x; pt.x++)
      {
        const float val = cacheImg[pt.y][pt.x];	// Valeur de l'image
//        if (val ==  cH.floatDefault) return;
        aSV  += val;		// Somme des valeurs de l'image cte
        aSVV += (val*val);	// Somme des carrés des vals image cte
      }

#ifdef FLOATMATH
  aSV   = fdividef(aSV,(float)cH.sizeVig );
  aSVV  = fdividef(aSVV,(float)cH.sizeVig );
  aSVV -=	(aSV * aSV);
#else
  aSV	/=	cH.sizeVig;
  aSVV  /=	cH.sizeVig;
  aSVV  -=	(aSV * aSV);
#endif

  if ( aSVV <= cH.mAhEpsilon) return;

  aSVV =	rsqrtf(aSVV); // racine carre inverse

  const uint pitchCache = blockIdx.z * cH.sizeCach + ptTer.x * cH.dimVig.x;
  const uint pitchCachY = ptTer.y * cH.dimVig.y ;
#pragma unroll
  for ( pt.y = c0.y ; pt.y <= c1.y; pt.y++)
    {
      const int _py	= (pitchCachY + (pt.y - c0.y))* cH.dimCach.x;
#pragma unroll
      for ( pt.x = c0.x ; pt.x <= c1.x; pt.x++)
        cachVig[ pitchCache + _py  + (pt.x - c0.x)] = (cacheImg[pt.y][pt.x] -aSV)*aSVV;

    }

  const int ZPitch	= pitZ * cH.sizeTer;
  const int idN		= ZPitch + to1D(ptTer,cH.dimTer);
  atomicAdd( &dev_NbImgOk[idN], 1U);

}

/// \brief Fonction qui lance les kernels de correlation
extern "C" void	 KernelCorrelation(const int s,cudaStream_t stream, dim3 blocks, dim3 threads, uint *dev_NbImgOk, float* cachVig, uint2 nbActThrd)
{

  switch (s)
    {
    case 0:
      correlationKernel<0><<<blocks, threads, 0, stream>>>( dev_NbImgOk, cachVig, nbActThrd);
      getLastCudaError("Basic Correlation kernel failed stream 0");
      break;
    case 1:
      correlationKernel<1><<<blocks, threads, 0, stream>>>( dev_NbImgOk, cachVig, nbActThrd);
      getLastCudaError("Basic Correlation kernel failed stream 1");
      break;
    case 2:
      correlationKernel<2><<<blocks, threads, 0, stream>>>( dev_NbImgOk, cachVig, nbActThrd);
      getLastCudaError("Basic Correlation kernel failed stream 2");
      break;
    case 3:
      correlationKernel<3><<<blocks, threads, 0, stream>>>( dev_NbImgOk, cachVig, nbActThrd);
      getLastCudaError("Basic Correlation kernel failed stream 3");
      break;
    case 4:
      correlationKernel<4><<<blocks, threads, 0, stream>>>( dev_NbImgOk, cachVig, nbActThrd);
      getLastCudaError("Basic Correlation kernel failed stream 4");
      break;
    case 5:
      correlationKernel<5><<<blocks, threads, 0, stream>>>( dev_NbImgOk, cachVig, nbActThrd);
      getLastCudaError("Basic Correlation kernel failed stream 5");
      break;
    case 6:
      correlationKernel<6><<<blocks, threads, 0, stream>>>( dev_NbImgOk, cachVig, nbActThrd);
      getLastCudaError("Basic Correlation kernel failed stream 6");
      break;
    case 7:
      correlationKernel<7><<<blocks, threads, 0, stream>>>( dev_NbImgOk, cachVig, nbActThrd);
      getLastCudaError("Basic Correlation kernel failed stream 7");
      break;
    }

}
/// \brief Kernel Calcul "rapide"  de la multi-correlation en utilisant la formule de Huygens utilisant des fonctions atomiques
template<int sNbTh> __global__ void multiCorrelationKernel(float *dTCost, float* cacheVign, int* dev_NbImgOk, uint2 nbActThr)
{

  const ushort BB = ( 4 - sNbTh ) * SBLOCKDIM / 3;
  __shared__ float aSV [ BB ][ BB ];		// Somme des valeurs
  __shared__ float aSVV[ BB  ][ BB ];		// Somme des carrés des valeurs
  __shared__ float resu[ BB/2 ][ BB/2 ];		// resultat
  __shared__ ushort nbIm[ BB/2][ BB/2 ];		// nombre d'images correcte

  // coordonnées des threads
  const uint2 t = make_uint2(threadIdx);

  if ( threadIdx.z == 0)
    {
      aSV [t.y][t.x]		= 0.0f;
      aSVV[t.y][t.x]		= 0.0f;
      resu[t.y/2][t.x/2]	= 0.0f;
      nbIm[t.y/2][t.x/2]	= 0;
    }

  __syncthreads();

  if ( oSE( t, nbActThr))	return; // si le thread est inactif, il sort

  // Coordonnées 2D du cache vignette
  const uint2 ptCach = make_uint2(blockIdx) * nbActThr  + t;

  // Si le thread est en dehors du cache
  if ( oSE(ptCach, cH.dimCach))	return;

  const uint2	ptTer	= ptCach / cH.dimVig;						// Coordonnées 2D du terrain

  if(tex2D(TexS_MaskTer, ptTer.x, ptTer.y) == 0) return;

  const uint	iTer	= blockIdx.z * cH.sizeTer + to1D(ptTer, cH.dimTer);	// Coordonnées 1D dans le terrain
  const uint2   thTer	= t / cH.dimVig;									// Coordonnées 2D du terrain dans le repere des threads
  const bool	mThrd	= aEq(t - thTer*cH.dimVig,0) && threadIdx.z == 0;

  if (mThrd)
    nbIm[thTer.y][thTer.x] = (ushort)dev_NbImgOk[iTer];

  __syncthreads();

  if (nbIm[thTer.y][thTer.x] < 2) return;

  const uint sizLayer = (blockIdx.z * cH.nbImages + threadIdx.z) * cH.sizeCach;	// Taille du cache vignette pour une image

  const uint2 cc    = ptTer * cH.dimVig;					// coordonnées 2D 1er pixel de la vignette
  const int iCC     = sizLayer + to1D( cc, cH.dimCach );			// coordonnées 1D 1er pixel de la vignette

  if (cacheVign[iCC]== cH.floatDefault) return;					// sortir si la vignette incorrecte

  const uint iCach  = sizLayer + to1D( ptCach, cH.dimCach );		// Coordonnées 1D du cache vignette
  const float val   = cacheVign[iCach];

  atomicAdd( &(aSV[t.y][t.x]), val);
  atomicAdd(&(aSVV[t.y][t.x]), val * val);
  __syncthreads();

  if ( threadIdx.z != 0) return;

#ifdef FLOATMATH
  atomicAdd(&(resu[thTer.y][thTer.x]),aSVV[t.y][t.x] - fdividef(aSV[t.y][t.x] * aSV[t.y][t.x],(float)nbIm[thTer.y][thTer.x]));
#else
  atomicAdd(&(resu[thTer.y][thTer.x]),aSVV[t.y][t.x] - ((aSV[t.y][t.x] * aSV[t.y][t.x])/ nbIm[thTer.y][thTer.x]));
#endif

  if ( !mThrd ) return;
  __syncthreads();

  // Normalisation pour le ramener a un equivalent de 1-Correl
#ifdef FLOATMATH
  const float cost = fdividef( resu[thTer.y][thTer.x], (float)( nbIm[thTer.y][thTer.x] -1.0f) * (cH.sizeVig));
#else
  const float cost = resu[thTer.y][thTer.x]/ (( nbIm[thTer.y][thTer.x] -1.0f) * ((float)cH.sizeVig));
#endif

  dTCost[iTer] = 1.0f - max (-1.0, min(1.0f,1.0f - cost));
}

/// \brief Kernel Calcul "rapide"  de la multi-correlation en utilisant la formule de Huygens n utilisant pas des fonctions atomiques
__global__ void multiCorrelationKernelNA(float *dTCost, float* cacheVign, int* dev_NbImgOk, uint2 nbActThr)
{

  __shared__ float aSV [ SBLOCKDIM ][ SBLOCKDIM ];		// Somme des valeurs
  __shared__ float aSVV[ SBLOCKDIM  ][ SBLOCKDIM ];		// Somme des carrés des valeurs
  __shared__ float resu[ SBLOCKDIM/2 ][ SBLOCKDIM/2 ];		// resultat
  __shared__ ushort nbIm[ SBLOCKDIM/2][ SBLOCKDIM/2 ];		// nombre d'images correcte

  // coordonnées des threads
  const uint2 t = make_uint2(threadIdx);

  aSV [t.y][t.x]	= 0.0f;
  aSVV[t.y][t.x]	= 0.0f;
  resu[t.y/2][t.x/2]	= 0.0f;
  nbIm[t.y/2][t.x/2]	= 0;

  if ( oSE( t, nbActThr))	return; // si le thread est inactif, il sort

  // Coordonnées 2D du cache vignette
  const uint2 ptCach = make_uint2(blockIdx) * nbActThr + t;

  // Si le thread est en dehors du cache
  if ( oSE(ptCach, cH.dimCach))	return;

  const uint2	ptTer	= ptCach / cH.dimVig; // Coordonnées 2D du terrain

  if(tex2D(TexS_MaskTer, ptTer.x, ptTer.y) == 0) return;

  const uint	iTer	= blockIdx.z * cH.sizeTer + to1D(ptTer, cH.dimTer);     // Coordonnées 1D dans le terrain avec prise en compte des differents Z
  const uint2   thTer	= t / cH.dimVig;                                        // Coordonnées 2D du terrain dans le repere des threads

  //if(aEq(t,thTer * cH.dimVig))
  nbIm[thTer.y][thTer.x] = (ushort)dev_NbImgOk[iTer];

  //__syncthreads();

  if ( nbIm[thTer.y][thTer.x]  < 2) return;

  const uint pitLayerCache  = blockIdx.z * cH.sizeCachAll + to1D( ptCach, cH.dimCach );	// Taille du cache vignette pour une image
  //const uint pit  = blockIdx.z * cH.nbImages;

 #pragma unroll
  for(uint i = 0;i< cH.sizeCachAll;i+=cH.sizeCach)
  //for(uint l = pit ;l< pit + cH.nbImages;l++)
    {
      const uint iCach = pitLayerCache  + i;
      const float val  = cacheVign[iCach];
      //const float val  = tex2DLayered( TexL_Cache,ptCach.x , ptCach.y,l);

      if(val!= cH.floatDefault)
        {
          // Coordonnées 1D du cache vignette

          aSV[t.y][t.x]   += val;
          aSVV[t.y][t.x]  += val * val;
        }
    }

  __syncthreads();

  atomicAdd(&(resu[thTer.y][thTer.x]),aSVV[t.y][t.x] - fdividef(aSV[t.y][t.x] * aSV[t.y][t.x],(float)nbIm[thTer.y][thTer.x]));

  if (!aEq(t - thTer*cH.dimVig,0)) return;

  __syncthreads();

  // Normalisation pour le ramener a un equivalent de 1-Correl
  const float cost = fdividef( resu[thTer.y][thTer.x], (float)( nbIm[thTer.y][thTer.x] -1.0f) * (cH.sizeVig));

  dTCost[iTer] = 1.0f - max (-1.0, min(1.0f,1.0f - cost));

}

/// \brief Fonction qui lance les kernels de multi-Correlation utilisant des fonctions atomiques
extern "C" void KernelmultiCorrelation(cudaStream_t stream, dim3 blocks, dim3 threads, float *dTCost, float* cacheVign, int * dev_NbImgOk, uint2 nbActThr, ushort divideNThreads)
{

  switch (divideNThreads)
    {
    case 1:
      multiCorrelationKernel<1><<<blocks, threads, 0, stream>>>(dTCost, cacheVign, dev_NbImgOk, nbActThr);
      break;
    case 2:
      multiCorrelationKernel<2><<<blocks, threads, 0, stream>>>(dTCost, cacheVign, dev_NbImgOk, nbActThr);
      break;
    case 3:
      multiCorrelationKernel<3><<<blocks, threads, 0, stream>>>(dTCost, cacheVign, dev_NbImgOk, nbActThr);
      break;
    default :
      multiCorrelationKernel<3><<<blocks, threads, 0, stream>>>(dTCost, cacheVign, dev_NbImgOk, nbActThr);
    }

  getLastCudaError("Multi-Correlation kernel failed");

}

/// \brief Fonction qui lance les kernels de multi-Correlation n'utilisant pas des fonctions atomiques
extern "C" void KernelmultiCorrelationNA(cudaStream_t stream, dim3 blocks, dim3 threads, float *dTCost, float* cacheVign, int * dev_NbImgOk, uint2 nbActThr)
{
  multiCorrelationKernelNA<<<blocks, threads, 0, stream>>>(dTCost, cacheVign, dev_NbImgOk, nbActThr);
  getLastCudaError("Multi-Correlation NON ATOMIC kernel failed");

}
