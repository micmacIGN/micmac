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

template<int TexSel> __global__ void projectionImage( HDParamCorrel HdPc, float* projImages, Rect* pRect)
{
    //extern __shared__ float cacheImg[];

    const uint2 ptHTer = make_uint2(blockIdx) *  blockDim.x + make_uint2(threadIdx);

    if (oSE(ptHTer,HdPc.dimHaloTer)) return;

    const ushort IdLayer = blockDim.z * blockIdx.z + threadIdx.z;

    const float2 ptProj  = GetProjection<TexSel>(ptHTer,invPc.sampProj, IdLayer);

    const Rect  zoneImage = pRect[IdLayer];

    float* localImages = projImages + IdLayer * size(HdPc.dimHaloTer);

    localImages[to1D(ptHTer,HdPc.dimHaloTer)] = (oI(ptProj,0)|| oSE( ptHTer, make_uint2(zoneImage.pt1)) || oI(ptHTer,make_uint2(zoneImage.pt0))) ? 1.f : GetImageValue(ptProj,threadIdx.z) / 2048.f;

}

extern "C" void	 LaunchKernelprojectionImage(pCorGpu &param, CuDeviceData3D<float>  &DeviImagesProj, Rect* pRect)
{

    dim3	threads( BLOCKDIM / 2, BLOCKDIM /2, param.invPC.nbImages); // on divise par deux car on explose le nombre de threads par block
    uint2	thd2D		= make_uint2(threads);
    uint2	block2D		= iDivUp(param.HdPc.dimHaloTer,thd2D);
    dim3	blocks(block2D.x , block2D.y, param.ZCInter);

    DeviImagesProj.ReallocIfDim(param.HdPc.dimHaloTer,param.ZCInter*param.invPC.nbImages);

    CuHostData3D<float>  hostImagesProj;

    hostImagesProj.ReallocIfDim(param.HdPc.dimHaloTer,param.ZCInter*param.invPC.nbImages);


    projectionImage<0><<<blocks, threads>>>(param.HdPc,DeviImagesProj.pData(),pRect);

    getLastCudaError("Projection Image");

    DeviImagesProj.CopyDevicetoHost(hostImagesProj);

    //    hostImagesProj.OutputValues();

    for (int z = 0; z < (int)param.ZCInter; ++z)
    {
        for (int i = 0; i < (int)param.invPC.nbImages; ++i)
        {
            std::string nameFile = std::string(GpGpuTools::conca("IMAGES_0",(i+1) * 10 + z)) + std::string(".pgm");
            GpGpuTools::Array1DtoImageFile(hostImagesProj.pData() + (i  + z *  param.invPC.nbImages)* size(hostImagesProj.GetDimension()),nameFile.c_str(),hostImagesProj.GetDimension());
        }
    }

    hostImagesProj.Dealloc();
    DeviImagesProj.Dealloc();

}

/// \fn template<int TexSel> __global__ void correlationKernel( uint *dev_NbImgOk, float* cachVig, uint2 nbActThrd)
/// \brief Kernel fonction GpGpu Cuda
/// Calcul les vignettes de correlation pour toutes les images
///
template<int TexSel> __global__ void correlationKernel( uint *dev_NbImgOk, ushort2 *ClassEqui,float* cachVig, uint2* pRect, uint2 nbActThrd,HDParamCorrel HdPc)
{

  extern __shared__ float cacheImg[];

  // Coordonnées du terrain global avec bordure // __umul24!!!! A voir
  const uint2 ptHaloTer = make_uint2(blockIdx) * nbActThrd + make_uint2(threadIdx);

  // Si le point est hors du terrain, nous sortons du kernel
  if (oSE(ptHaloTer,HdPc.dimHaloTer)  ) return;

  // Obtenir la projection du point dans l'image
  const float2 ptProj   = GetProjection<TexSel>(ptHaloTer,invPc.sampProj,blockIdx.z);

  // Phase : obtention de la valeur dans l'image
  const uint	pitZ      = blockIdx.z / invPc.nbImages;
  const uint	pitCach   = pitZ * invPc.nbImages;
  const ushort	idImg     = blockIdx.z - pitCach; // ID image courante

  // DEBUT 2014 || taille de l'image
  const uint2  zoneImage = pRect[idImg];

  // Si la projection est en dehors de l'image on sort
  if (oI(ptProj,0) || ptProj.x >= (float)zoneImage.x || ptProj.y >= (float)zoneImage.y)
      return;

  cacheImg[sgpu::__mult<BLOCKDIM>(threadIdx.y) + threadIdx.x] = GetImageValue(ptProj,idImg);

  __syncthreads();

  // Point terrain local
  const int2 ptTer = make_int2(ptHaloTer) - make_int2(invPc.rayVig);

  // Nous traitons uniquement les points du terrain du bloque ou Si le processus est hors du terrain global, nous sortons du kernel

  // Sortir si threard inactif et si en dehors du terrain (à simplifier)
  if (oSE(threadIdx, nbActThrd + invPc.rayVig) || oI(threadIdx , invPc.rayVig) || oSE( ptTer, HdPc.dimTer) || oI(ptTer,0))
	return;

  // Faux mais fait le job! en limite d'image
  if ( oI( ptProj - invPc.rayVig.x-1, 0) || (ptProj.x + invPc.rayVig.x+1>= (float)zoneImage.x) || (ptProj.y + invPc.rayVig.x+1>= (float)zoneImage.y))
	  return;

  // Point terrain global
  int2 coorTer = ptTer + HdPc.rTer.pt0;

  if(tex2D(TexS_MaskGlobal, coorTer.x, coorTer.y) == 0) return;

  if(tex2DLayered(TexL_MaskImages, coorTer.x, coorTer.y,idImg) == 0) return;

  const short2 c0	= make_short2(threadIdx) - invPc.rayVig;
  const short2 c1	= make_short2(threadIdx) + invPc.rayVig;

  // Intialisation des valeurs de calcul
  float aSV = 0.0f, aSVV = 0.0f;
  short2 pt;

  #pragma unroll // ATTENTION PRAGMA FAIT AUGMENTER LA quantité MEMOIRE des registres!!!
  for (pt.y = c0.y ; pt.y <= c1.y; pt.y++)
  {

	  const float* cImg    = cacheImg +  sgpu::__mult<BLOCKDIM>(pt.y);
      #pragma unroll
      for (pt.x = c0.x ; pt.x <= c1.x; pt.x++)
      {
          const float val = cImg[pt.x];     // Valeur de l'image
          aSV  += val;                      // Somme des valeurs de l'image cte
          aSVV += (val*val);                // Somme des carrés des vals image cte
      }
  }

  aSV   = fdividef(aSV,(float)invPc.sizeVig );

  aSVV  = fdividef(aSVV,(float)invPc.sizeVig );

  aSVV -=	(aSV * aSV);

  if ( aSVV <= invPc.mAhEpsilon) return;

  aSVV =	rsqrtf(aSVV); // racine carre inverse

  const uint pitchCachY = ptTer.y * invPc.dimVig.y ;

  const ushort iCla = ClassEqui[idImg].x;

  const ushort pCla = ClassEqui[iCla].y;

  const int  idN    = (pitZ * invPc.nbClass + iCla ) * HdPc.sizeTer + to1D(ptTer,HdPc.dimTer);

  const uint iCa    = atomicAdd( &dev_NbImgOk[idN], 1U) + pitCach + pCla;

  float* cache      = cachVig + (iCa * HdPc.sizeCach) + ptTer.x * invPc.dimVig.x - c0.x + (pitchCachY - c0.y)* HdPc.dimCach.x;

#pragma unroll
  for ( pt.y = c0.y ; pt.y <= c1.y; pt.y++)
    {
	  const float* cImg = cacheImg +  sgpu::__mult<BLOCKDIM>(pt.y);
      float* cVig = cache    + pt.y * HdPc.dimCach.x ;
#pragma unroll
      for ( pt.x = c0.x ; pt.x <= c1.x; pt.x++)

          cVig[ pt.x ] = (cImg[pt.x] -aSV)*aSVV;

    }
}
__global__ void getValueImagesKernel(  ushort2 *ClassEqui, float* cuValImage, uint2* pRect, uint2 nbActThrd,HDParamCorrel HdPc)
{
    extern __shared__ float cacheImg[];

    // Coordonnées du terrain global avec bordure // __umul24!!!! A voir

    const uint2 ptHTer = make_uint2(blockIdx) * nbActThrd + make_uint2(threadIdx);

    // Si le processus est hors du terrain, nous sortons du kernel

    if (oSE(ptHTer,HdPc.dimHaloTer)  ) return;

    const float2 ptProj   = GetProjection<0>(ptHTer,invPc.sampProj,blockIdx.z);
  // DEBUT AJOUT 2014
	const uint2  zoneImage = pRect[blockIdx.z]; // TODO 2015 FAUX a remplacer pas idImg

    uint pitZ,idImg,piCa;

	if (oI(ptProj,0) || ptProj.x >= (float)zoneImage.x || ptProj.y >= (float)zoneImage.y)
    {
        cacheImg[threadIdx.y*BLOCKDIM + threadIdx.x] = -1;
        return;
    }
    else
    {
        pitZ  = blockIdx.z / invPc.nbImages;

        piCa  = pitZ * invPc.nbImages;

        idImg  = blockIdx.z - piCa;

        cacheImg[threadIdx.y*BLOCKDIM + threadIdx.x] = GetImageValue(ptProj,idImg);
    }

    const int2 ptTer = make_int2(ptHTer) - make_int2(invPc.rayVig);

    const int  idN    = pitZ  * size(HdPc.dimTer) + to1D(ptTer,HdPc.dimTer);

    int2 coorTer = ptTer + HdPc.rTer.pt0;


// tres bizarre.... surement faux!!
	if ( oI( ptProj - invPc.rayVig.x-1, 0) || (ptProj.x + invPc.rayVig.x+1>= (float)zoneImage.x) || (ptProj.y + invPc.rayVig.x+1>= (float)zoneImage.y))
    {
        cuValImage[idN]   = -1;
        return;
    }

    if(tex2DLayered(TexL_MaskImages, coorTer.x, coorTer.y,idImg) == 0 || oI(ptTer,0) || oSE( ptTer, HdPc.dimTer))
        cuValImage[idN]   = -1;
    else
        cuValImage[idN]   = cacheImg[threadIdx.y*BLOCKDIM + threadIdx.x];

}

extern "C" void	 LaunchKernelGetValueImages(pCorGpu &param,SData2Correl &data2cor)
{

    dim3	threads( BLOCKDIM, BLOCKDIM, 1);
    uint2	thd2D		= make_uint2(threads);
    uint2	nbActThrd	= thd2D - 2 * param.invPC.rayVig;
    uint2	block2D		= iDivUp(param.HdPc.dimHaloTer,nbActThrd);
    dim3	blocks(block2D.x , block2D.y, param.invPC.nbImages * param.ZCInter);

    CuHostData3D<float>     hoValImage;
    CuDeviceData3D<float>   cuValImage;

    cuValImage.Malloc(param.HdPc.dimTer,param.invPC.nbImages* param.ZCInter);
    hoValImage.Malloc(param.HdPc.dimTer,param.invPC.nbImages* param.ZCInter);

    getValueImagesKernel<<<blocks, threads, BLOCKDIM * BLOCKDIM * sizeof(float)>>>( data2cor.DeviClassEqui(),cuValImage.pData(),data2cor.DeviRect(), nbActThrd,param.HdPc);
    getLastCudaError("Basic getValue kernel failed stream 0");

    cuValImage.CopyDevicetoHost(hoValImage);

    hoValImage.OutputInfo();

    hoValImage.OutputValues(0,XY,/*NEGARECT*/Rect(0,0,20,hoValImage.GetDimension().y),8,-1);

    cuValImage.Dealloc();
    hoValImage.Dealloc();

}
/// \brief Fonction qui lance les kernels de correlation
extern "C" void	 LaunchKernelCorrelation(const int s,cudaStream_t stream,pCorGpu &param,SData2Correl &data2cor)
{

    dim3	threads( BLOCKDIM, BLOCKDIM, 1);
    uint2	thd2D		= make_uint2(threads);
    uint2	nbActThrd	= thd2D - 2 * param.invPC.rayVig;
    uint2	block2D		= iDivUp(param.HdPc.dimHaloTer,nbActThrd);
    dim3	blocks(block2D.x , block2D.y, param.invPC.nbImages * param.ZCInter);

//    CuDeviceData3D<float>       DeviImagesProj;
//    LaunchKernelprojectionImage(param,DeviImagesProj,data2cor.DeviRect());
//    DeviImagesProj.Dealloc();

    //LaunchKernelGetValueImages(param,data2cor);

    switch (s)
    {
    case 0:
        correlationKernel<0><<<blocks, threads, BLOCKDIM * BLOCKDIM * sizeof(float), stream>>>( data2cor.DeviVolumeNOK(0),data2cor.DeviClassEqui(), data2cor.DeviVolumeCache(0),data2cor.DeviRect(), nbActThrd,param.HdPc);
        getLastCudaError("Basic Correlation kernel failed stream 0");
        break;
    case 1:
        correlationKernel<1><<<blocks, threads, BLOCKDIM * BLOCKDIM* sizeof(float), stream>>>( data2cor.DeviVolumeNOK(1),data2cor.DeviClassEqui(), data2cor.DeviVolumeCache(1),data2cor.DeviRect(), nbActThrd,param.HdPc);
        getLastCudaError("Basic Correlation kernel failed stream 1");
        break;
    }
}



/// \brief Kernel Calcul "rapide"  de la multi-correlation en utilisant la formule de Huygens n utilisant pas des fonctions atomiques

template<ushort SIZE3VIGN > __global__ void multiCorrelationKernel(ushort2* classEqui,float *dTCost, float* cacheVign, uint* dev_NbImgOk, /*uint2 nbActThr,*/HDParamCorrel HdPc)
{

  __shared__ float aSV [ SIZE3VIGN   ][ SIZE3VIGN ];          // Somme des valeurs
  __shared__ float aSVV[ SIZE3VIGN   ][ SIZE3VIGN ];         // Somme des carrés des valeurs
  __shared__ float resu[ SIZE3VIGN>>1 ][ SIZE3VIGN>>1 ];		// resultat

  __shared__ float cResu[ SIZE3VIGN>>1][ SIZE3VIGN>>1 ];		// resultat
  __shared__ uint nbIm[ SIZE3VIGN>>1][ SIZE3VIGN>>1 ];		// nombre d'images correcte

  // coordonnées des threads // TODO uint2 to ushort2
  const uint2 t  = make_uint2(threadIdx);
  //const uint2 mt = make_uint2(t.x/2,t.y/2);

  // TODO : 2014 LE NOMBRE DE TREAD ACTIF peut etre nettement ameliorer par un template
  //if ( oSE( t, nbActThr))	return; // si le thread est inactif, il sort

  // Coordonnées 2D du cache vignette
  const uint2 ptCach = make_uint2(blockIdx) * SIZE3VIGN + t;

  // Si le thread est en dehors du cache // TODO 2014 à verifier ----
  if ( oSE(ptCach, HdPc.dimCach))	return;

  const uint2	ptTer	= ptCach / invPc.dimVig; // Coordonnées 2D du terrain

  // if(!tex2D(TexS_MaskGlobal, ptTer.x + HdPc.rTer.pt0.x , ptTer.y + HdPc.rTer.pt0.y)) return;// COM 6 mars 2014// TODO 2014 à verifier notamment quand il n'y a pas de cache!!!

  const uint    ter     = to1D(ptTer, HdPc.dimTer);            // Coordonnées 1D du terrain

  const uint	iTer	= blockIdx.z * HdPc.sizeTer + ter;     // Coordonnées 1D du terrain avec prise en compte des differents Z

  const uint2   thTer	= t / invPc.dimVig;                    // Coordonnées 2D du terrain dans le repere des threads

  const bool mainThread = aEq(t - thTer*invPc.dimVig,0);

  //if (!aEq(t - thTer*invPc.dimVig,0))
  //{
      resu[thTer.y][thTer.x]    = 0.0f;
      nbIm[thTer.y][thTer.x]    = 0;
  //}

  __syncthreads();

  for (ushort iCla = 0; iCla < invPc.nbClass; ++iCla)
  {

      const uint icTer    = (blockIdx.z* invPc.nbClass + iCla ) * HdPc.sizeTer + ter;

      const ushort nImgOK = (ushort)dev_NbImgOk[icTer];

      if ( nImgOK > 1)
      {
		  aSV [t.y][t.x]    = 0.0f;
		  aSVV[t.y][t.x]    = 0.0f;
		  cResu[thTer.y][thTer.x]	= 0.0f;

          const uint pitCla         = ((uint)classEqui[iCla].y) * HdPc.sizeCach;

          const uint pitLayerCache  = blockIdx.z  * HdPc.sizeCachAll + pitCla + to1D( ptCach, HdPc.dimCach );	// Taille du cache vignette pour une image

		  const float* caVi = cacheVign + pitLayerCache;

		  const uint limOK = nImgOK * HdPc.sizeCach;

 #pragma unroll
		  for(uint i =  0 ;i< limOK ;i+=HdPc.sizeCach)
          {
              const float val  = caVi[i];
              aSV[t.y][t.x]   += val;
              aSVV[t.y][t.x]  += val * val;
          }

          //__syncthreads();

          //atomicAdd(&(resu[thTer.y][thTer.x]),(aSVV[t.y][t.x] - fdividef(aSV[t.y][t.x] * aSV[t.y][t.x],(float)nImgOK)) * (nImgOK - 1));

		  atomicAdd(&(cResu[thTer.y][thTer.x]),(aSVV[t.y][t.x] - fdividef(aSV[t.y][t.x] * aSV[t.y][t.x],(float)nImgOK)));

          __syncthreads();

          if (mainThread)
          {
			  resu[thTer.y][thTer.x] += (float)(1.0f - max (-1.0, min(1.0f,1.0f - fdividef( cResu[thTer.y][thTer.x], ((float)(nImgOK - 1))* (invPc.sizeVig))))) * nImgOK;
              nbIm[thTer.y][thTer.x] += nImgOK;
          }
      }
  }

  __syncthreads();
  if( (nbIm[thTer.y][thTer.x] == 0) || (!mainThread) ) return;

  //__syncthreads();

  // Normalisation pour le ramener a un equivalent de 1-Correl
  //const float cost =  fdividef( resu[thTer.y][thTer.x], ((float)nImgOK -1.0f) * (invPc.sizeVig));

  //const float cost =  fdividef( resu[thTer.y][thTer.x], ((float)nbIm[thTer.y][thTer.x])* (invPc.sizeVig));

  //const float cost =  fdividef( resu[thTer.y][thTer.x], ((float)nbIm[thTer.y][thTer.x] -1)* (invPc.sizeVig));

  //dTCost[iTer] = 1.0f - max (-1.0, min(1.0f,1.0f - cost));

  dTCost[iTer] = fdividef(resu[thTer.y][thTer.x],(float)nbIm[thTer.y][thTer.x]);

}

template<ushort SIZE3VIGN > void LaunchKernelMultiCor(cudaStream_t stream, pCorGpu &param, SData2Correl &dataCorrel)
{
    //-------------	calcul de dimension du kernel de multi-correlation NON ATOMIC ------------
    //uint2	nbActThr	= SIZE3VIGN - make_uint2( SIZE3VIGN % param.invPC.dimVig.x, SIZE3VIGN % param.invPC.dimVig.y);
    dim3	threads(SIZE3VIGN, SIZE3VIGN, 1);
    uint2	block2D	= iDivUp(param.HdPc.dimCach,SIZE3VIGN);
    dim3	blocks(block2D.x,block2D.y,param.ZCInter);

    multiCorrelationKernel<SIZE3VIGN><<<blocks, threads, 0, stream>>>(dataCorrel.DeviClassEqui(),dataCorrel.DeviVolumeCost(0), dataCorrel.DeviVolumeCache(0), dataCorrel.DeviVolumeNOK(0),param.HdPc);
    getLastCudaError("Multi-Correlation NON ATOMIC kernel failed");
}

/// \brief Fonction qui lance les kernels de multi-Correlation n'utilisant pas des fonctions atomiques
extern "C" void LaunchKernelMultiCorrelation(cudaStream_t stream, pCorGpu &param, SData2Correl &dataCorrel)
{
    if(param.invPC.rayVig.x == 1 || param.invPC.rayVig.x == 2 )
        LaunchKernelMultiCor<SBLOCKDIM>(stream, param, dataCorrel);
    else if(param.invPC.rayVig.x == 3 )
        LaunchKernelMultiCor<7*2>(stream, param, dataCorrel);

}
