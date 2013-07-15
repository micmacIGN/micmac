#pragma once

#ifdef _WIN32
	#ifdef INT
		#undef INT 
	#endif
#endif

#include "GpGpu/GpGpu.h"

#ifdef CUDA_ENABLED

#ifndef BOOST_ALL_NO_LIB
    #define BOOST_ALL_NO_LIB
#endif

#include "GpGpu/cudaAppliMicMac.cuh"
#include <boost/thread/thread.hpp>
#include "GpGpu/GpGpuTools.h"
#include "GpGpu/GpGpuMultiThreadingCpu.h"

extern "C" void	CopyParamTodevice(pCorGpu h);
extern "C" void	KernelCorrelation(const int s,cudaStream_t stream, dim3 blocks, dim3 threads, uint *dev_NbImgOk, float* cachVig, uint2 nbActThrd);
extern "C" void	KernelmultiCorrelation(cudaStream_t stream, dim3 blocks, dim3 threads, float *dTCost, float* cacheVign, uint* dev_NbImgOk, uint2 nbActThr);

extern "C" void dilateKernel(pixel* HostDataOut, short r, uint2 dim);
extern "C" textureReference& getMaskD();

extern "C" textureReference&	getMask();
extern "C" textureReference&	getImage();
extern "C" textureReference&	getProjection(int TexSel);

/// \class InterfaceMicMacGpGpu
/// \brief Class qui lie micmac avec les outils de calculs GpGpu
class InterfaceMicMacGpGpu : public CSimpleJobCpuGpu< uint>
{

public:

  InterfaceMicMacGpGpu();
  ~InterfaceMicMacGpGpu();

  /// \brief    Initialise la taille du bloque terrain et nombre de Z a calculer sur le Gpu
  void          SetSizeBlock( uint Zinter, Rect Ter);
  /// \brief    Initialise le nombre de Z a calculer sur le Gpu
  void          SetSizeBlock( uint Zinter);
  /// \brief    Initialise les donnees du masque et sa dimension
  void          SetMask(pixel* dataMask, uint2 dimMask);
  /// \brief    Initialise les donnees des images, la dimension maximale d'une image et leur nombre
  void          SetImages(float* dataImage, uint2 dimImage, int nbLayer);
  /// \brief    Initialise les parametres de correlation
  void          SetParameter(Rect Ter, int nbLayer , uint2 dRVig , uint2 dimImg, float mAhEpsilon, uint samplingZ, int uvINTDef , uint interZ);
  /// \brief    Calcul de la correlation en Gpu
  void          BasicCorrelation(int nbLayer);
  /// \brief    Calcul asynchrone de la correlation en Gpu
  void          BasicCorrelationStream( float* hostVolumeCost, float2* hostVolumeProj,  int nbLayer, uint interZ );
  /// \brief    Renvoie les parametres de correlation
  pCorGpu       Param();

  /// \brief    Desalloue toutes la memoire globale alloué pour la correlation sur Gpu
  void          DeallocMemory();
  /// \brief    Affiche sur la console la memoire globale alloué pour la correlation sur Gpu
  void          MallocInfo();

  void          ReallocHost(uint zInter);

  void          MemsetProj();

  float*        OuputCost(uint id = 0);

  float2*       InputProj();

  void          DeallocVolumes();

  void          InitJob();
  void          freezeCompute();

   void         IntervalZ(uint &interZ, int anZProjection, int aZMaxTer);

private:

  void              ReallocAllDeviceData(uint interZ);
  void              ReallocAllDeviceDataAsync(uint interZ);
  void              ReallocDeviceData(int nStream, uint interZ);
  cudaStream_t*		GetStream(int stream);
  textureReference&	GetTeXProjection(int texSel);

  void              threadCompute();

  cudaStream_t  _stream[NSTREAM];
  pCorGpu		_param;

  CuDeviceData3D<float>     _d_volumeCost[NSTREAM];	// volume des couts
  CuDeviceData3D<float>     _d_volumeCach[NSTREAM];	// volume des calculs intermédiaires
  CuDeviceData3D<uint>      _d_volumeNIOk[NSTREAM];	// nombre d'image correct pour une vignette

  ImageCuda<pixel>          _dt_mask;
  ImageLayeredCuda<float>   _dt_LayeredImages;
  ImageLayeredCuda<float2>  _dt_LayeredProjection[NSTREAM];

  textureReference&         _texMask;
  textureReference&         _texMaskD;
  textureReference&         _texImages;
  //textureReference&       _texCache;
  textureReference&         _texProjections_00;
  textureReference&         _texProjections_01;
  textureReference&         _texProjections_02;
  textureReference&         _texProjections_03;
  textureReference&         _texProjections_04;
  textureReference&         _texProjections_05;
  textureReference&         _texProjections_06;
  textureReference&         _texProjections_07;

  CuHostData3D<float>		_hVolumeCost[2];
  CuHostData3D<float2>      _hVolumeProj;

#ifdef USEDILATEMASK	
public:
  void	dilateMask(uint2 dim);
  pixel*	GetDilateMask();
private:
  pixel	ValDilMask(int2 pt);
  pixel*				_dilateMask;
  uint2					_dimDilateMask;
#endif

};

#endif
