#ifndef __GPGPU_INTERCORREL_H__
#define __GPGPU_INTERCORREL_H__

/** @addtogroup GpGpuDoc */
/*@{*/

#ifdef _WIN32
    #ifdef INT
        #undef INT
    #endif
#endif

#include "GpGpu/GpGpu.h"

#if CUDA_ENABLED


/// \cond
#ifndef BOOST_ALL_NO_LIB
    #define BOOST_ALL_NO_LIB
#endif
/// \endcond

#include "GpGpu/GpGpu_MultiThreadingCpu.h"
#include "GpGpu/SData2Correl.h"

/// \cond
extern "C" void	CopyParamTodevice(pCorGpu h);
extern "C" void CopyParamInvTodevice( pCorGpu param );
extern "C" void	LaunchKernelCorrelation(const int s,cudaStream_t stream,pCorGpu &param,SData2Correl &dataCorrel);
extern "C" void	LaunchKernelMultiCorrelation(cudaStream_t stream, pCorGpu &param, SData2Correl &dataCorrel);

extern "C" void dilateKernel(pixel* HostDataOut, short r, uint2 dim);
extern "C" void	LaunchKernelprojectionImage(pCorGpu &param,CuDeviceData3D<float>  &DeviImagesProj);
/// \endcond

/// \class GpGpuInterfaceCorrel
/// \brief Class qui lie micmac avec les outils de calculs GpGpu
class GpGpuInterfaceCorrel : public CSimpleJobCpuGpu< bool>
{

public:

  GpGpuInterfaceCorrel();
  ~GpGpuInterfaceCorrel();

  /// \brief    Initialise les parametres de correlation
  void          SetParameter(int nbLayer , ushort2 dRVig , uint2 dimImg, float mAhEpsilon, uint samplingZ, int uvINTDef, ushort nClass);

  /// \brief    Calcul de la correlation en Gpu
  void          BasicCorrelation();

  /// \brief    Renvoie les parametres de correlation

  pCorGpu&      Param(ushort idBuf);

  ///
  /// \brief signalComputeCorrel Signal le debut d'une corrélation
  /// \param dZ
  ///
  void          signalComputeCorrel(uint dZ);

  ///
  /// \brief InitCorrelJob initialise le job
  /// \param Zmin z minimum
  /// \param Zmax z minimum
  /// \return
  ///
  uint          InitCorrelJob(int Zmin, int Zmax);

  void          freezeCompute();

  ///
  /// \brief IntervalZ Calcul l'interval Z
  /// \param interZ
  /// \param anZProjection
  /// \param aZMaxTer
  ///
  void          IntervalZ(uint &interZ, int anZProjection, int aZMaxTer);

  ///
  /// \brief Data
  /// \return les données de corrélation
  ///
  SData2Correl& Data();

  ///
  /// \brief VolumeCost
  /// \param id
  /// \return Le volume de corrélation
  ///
  float*        VolumeCost(ushort id);

  ///
  /// \brief TexturesAreLoaded
  /// \return Vraie si les images sont chargées dans le device
  ///
  bool          TexturesAreLoaded();

  ///
  /// \brief SetTexturesAreLoaded Définir si les images sont chargées dans le device
  /// \param load
  ///
  void          SetTexturesAreLoaded(bool load);  

  ///
  /// \brief ReallocHostData Réallouer la mémoire dans l'hote
  /// \param interZ
  /// \param idBuff
  ///
  void          ReallocHostData(uint interZ, ushort idBuff);

  ///
  /// \brief DimTerrainGlob
  /// \return Les dimensions du terrain
  ///
  uint2&        DimTerrainGlob();

  ///
  /// \brief MaskVolumeBlock
  /// \return Un vecteur des cellules à corréler
  ///
  std::vector<cellules> &MaskVolumeBlock();

  ///
  /// \brief NoMasked
  /// Paramètre qui indique si les cellules doivent etre calculer
  bool              NoMasked;

private:

  void              CorrelationGpGpu(ushort idBuf = 0 , const int s = 0);

  void              MultiCorrelationGpGpu(ushort idBuf = 0,const int s = 0);

  cudaStream_t*		GetStream(int stream);

  void              simpleWork();

  cudaStream_t      _stream[NSTREAM];

  pCorGpu           _param[2];

  SData2Correl      _data2Cor;

  bool				_TexturesAreLoaded;

  uint2             _m_DimTerrainGlob;

  std::vector<cellules> _m_MaskVolumeBlock;

  bool              copyInvParam;


};

#endif
/*@}*/
#endif // __GPGPU_INTERCORREL_H__
