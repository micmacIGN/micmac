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

#include "GpGpu/GpGpuMultiThreadingCpu.h"
#include "GpGpu/SData2Correl.h"

extern "C" void	CopyParamTodevice(pCorGpu h);
extern "C" void	KernelCorrelation(const int s,cudaStream_t stream, dim3 blocks, dim3 threads,SData2Correl &dataCorrel, uint2 nbActThrd);
extern "C" void	KernelmultiCorrelation(cudaStream_t stream, dim3 blocks, dim3 threads, SData2Correl &dataCorrel, uint2 nbActThr);

extern "C" void dilateKernel(pixel* HostDataOut, short r, uint2 dim);


/// \class InterfaceMicMacGpGpu
/// \brief Class qui lie micmac avec les outils de calculs GpGpu
class InterfaceMicMacGpGpu : public CSimpleJobCpuGpu< uint>
{

public:

  InterfaceMicMacGpGpu();
  ~InterfaceMicMacGpGpu();

  /// \brief    Initialise les parametres de correlation
  void          SetParameter(int nbLayer , uint2 dRVig , uint2 dimImg, float mAhEpsilon, uint samplingZ, int uvINTDef);
  /// \brief    Calcul de la correlation en Gpu
  void          BasicCorrelation(int nbLayer);
  /// \brief    Calcul asynchrone de la correlation en Gpu
  void          BasicCorrelationStream( float* hostVolumeCost, float2* hostVolumeProj,  int nbLayer, uint interZ );
  /// \brief    Renvoie les parametres de correlation
  pCorGpu       &Param();

  void          signalComputeCorrel(uint dZ);

  void          InitJob(uint &interZ);

  void          freezeCompute();

  void          IntervalZ(uint &interZ, int anZProjection, int aZMaxTer);

  SData2Correl&  Data(){return _data2Cor;}

private:

  cudaStream_t*		GetStream(int stream);
  void              threadCompute();

  cudaStream_t      _stream[NSTREAM];
  pCorGpu           _param;

  SData2Correl      _data2Cor;

};

#endif
