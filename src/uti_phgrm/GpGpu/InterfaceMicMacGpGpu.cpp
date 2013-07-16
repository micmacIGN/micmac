#include "GpGpu/InterfaceMicMacGpGpu.h"

/// \brief Constructeur InterfaceMicMacGpGpu
InterfaceMicMacGpGpu::InterfaceMicMacGpGpu()
{
  for (int s = 0;s<NSTREAM;s++)
    checkCudaErrors( cudaStreamCreate(GetStream(s)));

    CreateJob();
}

InterfaceMicMacGpGpu::~InterfaceMicMacGpGpu()
{
  for (int s = 0;s<NSTREAM;s++)
    checkCudaErrors( cudaStreamDestroy(*(GetStream(s))));

}

void InterfaceMicMacGpGpu::SetSizeBlock( uint Zinter, Rect Ter)
{
  _param.SetDimension(Ter,Zinter);

  _data2Cor.ReallocDeviceData(_param);
}

void InterfaceMicMacGpGpu::InitJob(uint interZ)
{
    CopyParamTodevice(_param);

    _data2Cor.ReallocHostData(interZ,_param);

    if(UseMultiThreading())
    {
        ResetIdBuffer();
        SetPreComp(true);
    }
}

/// \brief Initialisation des parametres constants
void InterfaceMicMacGpGpu::SetParameter(int nbLayer , uint2 dRVig , uint2 dimImg, float mAhEpsilon, uint samplingZ, int uvINTDef )
{ 
  _param.SetParamInva( dRVig * 2 + 1,dRVig, dimImg, mAhEpsilon, samplingZ, uvINTDef, nbLayer);
}

void InterfaceMicMacGpGpu::BasicCorrelation(int nbLayer)
{

  // Re-dimensionner les strucutres de données si elles ont été modifiées
  _data2Cor.ReallocDeviceArray(_param);

  // Calcul de dimension des threads des kernels
  //--------------- calcul de dimension du kernel de correlation ---------------
  dim3	threads( BLOCKDIM, BLOCKDIM, 1);
  uint2	thd2D		= make_uint2(threads);
  uint2	actiThsCo	= thd2D - 2 * _param.rayVig;
  uint2	block2D		= iDivUp(_param.dimDTer,actiThsCo);
  dim3	blocks(block2D.x , block2D.y, nbLayer * _param.ZLocInter);
  const uint s = 0;     // Affection du stream de calcul

  Data().copyHostToDevice(s);

  // Indique que la copie est terminée pour le thread de calcul des projections
  SetPreComp(true);

  // Lancement du calcul de correlation
  KernelCorrelation(s, *(GetStream(s)),blocks, threads, _data2Cor, actiThsCo);

  // Libérer la texture de projection
  Data().UnBindTextureProj(s);

  // Lancement du calcul de multi-correlation

  //-------------	calcul de dimension du kernel de multi-correlation NON ATOMIC ------------
  uint2	actiThs_NA	= SBLOCKDIM - make_uint2( SBLOCKDIM % _param.dimVig.x, SBLOCKDIM % _param.dimVig.y);
  dim3	threads_mC_NA(SBLOCKDIM, SBLOCKDIM, 1);
  uint2	block2D_mC_NA	= iDivUp(_param.dimCach,actiThs_NA);
  dim3	blocks_mC_NA(block2D_mC_NA.x,block2D_mC_NA.y,_param.ZLocInter);

  KernelmultiCorrelation( *(GetStream(s)),blocks_mC_NA, threads_mC_NA,  _data2Cor, actiThs_NA);

  // Copier les resultats de calcul des couts du device vers le host!
  Data().CopyDevicetoHost(GetIdBuf(),s);

}

void InterfaceMicMacGpGpu::BasicCorrelationStream( float* hostVolumeCost, float2* hostVolumeProj, int nbLayer, uint interZ )
{
/*
  _data2Cor.ReallocAllDeviceData(_param.ZLocInter,_param);
  uint Z = 0;

  //            calcul de dimension du kernel de correlation                ---------------

  dim3	threads( BLOCKDIM, BLOCKDIM, 1);
  uint2	thd2D		= make_uint2(threads);
  uint2	actiThsCo	= thd2D - 2 * _param.rayVig;
  uint2	block2D		= iDivUp(_param.dimDTer,actiThsCo);
  dim3	blocks(block2D.x , block2D.y, nbLayer * _param.ZLocInter);

  //            Calcul de dimension du kernel de multi-correlation          ---------------

  dim3    threads_mC,blocks_mC;
  uint2   actiThs,block2D_mC;

  //            Calcul de dimension du kernel de multi-correlation          ---------------


  //-------------	calcul de dimension du kernel de multi-correlation NON ATOMIC ------------
  actiThs     = SBLOCKDIM - make_uint2( SBLOCKDIM % _param.dimVig.x, SBLOCKDIM % _param.dimVig.y);
  threads_mC  = dim3(SBLOCKDIM, SBLOCKDIM, 1);


  block2D_mC  = iDivUp(_param.dimCach,actiThs);
  blocks_mC   = dim3(block2D_mC.x,block2D_mC.y,_param.ZLocInter);

  while(Z < interZ)
    {
      const uint nstream = (NSTREAM * _param.ZLocInter) >= (interZ - Z)  ? (interZ - Z) : NSTREAM;
      //const uint nstream = 1;

      for (uint s = 0;s<nstream;s++)
          _dt_LayeredProjection[s].copyHostToDeviceASync(hostVolumeProj + (Z  + s) *_dt_LayeredProjection->GetSize(),*(GetStream(s)));

      if (Z == 0) SetPreComp(true); // A faire quand toutes les copies asynchrones sont terminées!!

      for (uint s = 0;s<nstream;s++)
        {
          _dt_LayeredProjection[s].bindTexture(GetTeXProjection(s)); // peut etre qu'ici également
          KernelCorrelation(s, *(GetStream(s)),blocks, threads,  _d_volumeNIOk[s].pData(), _d_volumeCach[s].pData(), actiThsCo);

          KernelmultiCorrelation( *(GetStream(s)),blocks_mC, threads_mC,  _d_volumeCost[s].pData(), _d_volumeCach[s].pData(), _d_volumeNIOk[s].pData(), actiThs);
        }

      for (uint s = 0;s<nstream;s++)
          _d_volumeCost[s].CopyDevicetoHostASync(hostVolumeCost + (Z + s)*_d_volumeCost[s].GetSize(),*(GetStream(s)));

      for (uint s = 0;s<nstream;s++)
        checkCudaErrors( cudaUnbindTexture(&(GetTeXProjection(s))) ); // il faut mettre un signal pour unbinder...

      Z += nstream * _param.ZLocInter;
    }

    //SetComputeNextProj(true); // A faire quand toutes les copies asynchrones sont terminées!!

//  for (uint s = 0;s<NSTREAM;s++)
  //    cudaStreamSynchronize(*(GetStream(s));

    checkCudaErrors(cudaDeviceSynchronize());
    */
}

cudaStream_t* InterfaceMicMacGpGpu::GetStream( int stream )
{
  return &(_stream[stream]);
}

void InterfaceMicMacGpGpu::threadCompute()
{  
  ResetIdBuffer();
  while (true)
    {
      if (GetCompute())
        {          
          uint interZ = GetCompute();
          SetCompute(0);

          BasicCorrelation(_param.nbImages);

          SwitchIdBuffer();

          while(GetDataToCopy());          
          SetDataToCopy(interZ);
        }
    }
}

void InterfaceMicMacGpGpu::freezeCompute()
{
    SetDataToCopy(0);
    SetCompute(0);
    SetPreComp(false);
}

void InterfaceMicMacGpGpu::IntervalZ(uint &interZ, int anZProjection, int aZMaxTer)
{
    uint intZ = (uint)abs(aZMaxTer - anZProjection );
    if (interZ >= intZ  &&  anZProjection != (aZMaxTer - 1) )
        interZ = intZ;
}

pCorGpu InterfaceMicMacGpGpu::Param()
{
    return _param;
}

void InterfaceMicMacGpGpu::signalComputeCorrel(uint dZ)
{
    SetPreComp(false);
    SetCompute(dZ);
}




