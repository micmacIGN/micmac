#include "GpGpu/InterfaceMicMacGpGpu.h"

/// \brief Constructeur InterfaceMicMacGpGpu
InterfaceMicMacGpGpu::InterfaceMicMacGpGpu():
  _texMask(getMask()),
  _texMaskD(getMaskD()),
  _texImages(getImage()),
  _texProjections_00(getProjection(0)),
  _texProjections_01(getProjection(1)),
  _texProjections_02(getProjection(2)),
  _texProjections_03(getProjection(3)),
  _texProjections_04(getProjection(4)),
  _texProjections_05(getProjection(5)),
  _texProjections_06(getProjection(6)),
  _texProjections_07(getProjection(7))
{
  for (int s = 0;s<NSTREAM;s++)
    checkCudaErrors( cudaStreamCreate(GetStream(s)));

  CreateJob();

  _d_volumeCost->SetName("_volumeCost");
  _d_volumeCach->SetName("_volumeCach");
  _d_volumeNIOk->SetName("_volumeNIOk");
  _dt_mask.CData2D::SetName("_mask");
  _dt_LayeredImages.CData3D::SetName("_LayeredImages");
  _dt_LayeredProjection->CData3D::SetName("_LayeredProjection");

  // Parametres texture des projections
  for (int s = 0;s<NSTREAM;s++)
    {

      GetTeXProjection(s).addressMode[0]	= cudaAddressModeBorder;
      GetTeXProjection(s).addressMode[1]	= cudaAddressModeBorder;
      GetTeXProjection(s).filterMode		= cudaFilterModeLinear; //cudaFilterModePoint cudaFilterModeLinear
      GetTeXProjection(s).normalized		= false;

    }

  // Parametres texture des Images
  _texImages.addressMode[0]	= cudaAddressModeWrap;
  _texImages.addressMode[1]	= cudaAddressModeWrap;
  _texImages.filterMode		= cudaFilterModeLinear; //cudaFilterModeLinear cudaFilterModePoint
  _texImages.normalized		= false;

  _hVolumeCost[0].SetName("_hVolumeCost0");
  _hVolumeCost[1].SetName("_hVolumeCost1");
  _hVolumeCost[0].SetPageLockedMemory(true);
  _hVolumeCost[1].SetPageLockedMemory(true);

  _hVolumeProj.SetName("_hVolumeProj");

}

InterfaceMicMacGpGpu::~InterfaceMicMacGpGpu()
{
  for (int s = 0;s<NSTREAM;s++)
    checkCudaErrors( cudaStreamDestroy(*(GetStream(s))));

  DeallocMemory();
}

void InterfaceMicMacGpGpu::SetSizeBlock( uint Zinter )
{
  SetSizeBlock( Zinter,_param.RTer());
}

void InterfaceMicMacGpGpu::SetSizeBlock( uint Zinter, Rect Ter)
{

  uint oldSizeTer = _param.sizeDTer;

  _param.SetDimension(Ter,Zinter);

  //if(Param().MaskNoNULL())

  CopyParamTodevice(_param);

  for (int s = 0;s<NSTREAM;s++)
  {
      _dt_LayeredProjection[s].Realloc(_param.dimSTer,_param.nbImages * _param.ZLocInter);

      if (oldSizeTer < _param.sizeDTer)
          ReallocDeviceData(s,_param.ZLocInter);
  }

}

void InterfaceMicMacGpGpu::ReallocHost(uint zInter)
{
    for (int i = 0; i < SIZERING; ++i)
        _hVolumeCost[i].Realloc(Param().dimTer,zInter);
    _hVolumeProj.Realloc(Param().dimSTer,zInter*Param().nbImages);
}

void InterfaceMicMacGpGpu::SetParameter( Rect Ter, int nbLayer , uint2 dRVig , uint2 dimImg, float mAhEpsilon, uint samplingZ, int uvINTDef , uint interZ )
{

  // Initialisation des parametres constants
  _param.SetParamInva( dRVig * 2 + 1,dRVig, dimImg, mAhEpsilon, samplingZ, uvINTDef, nbLayer);
  // Initialisation des parametres de dimensions du terrain
  SetSizeBlock( interZ, Ter);

}

void InterfaceMicMacGpGpu::DeallocMemory()
{
  checkCudaErrors( cudaUnbindTexture(&_texImages) );
  checkCudaErrors( cudaUnbindTexture(&_texMask) );
  checkCudaErrors( cudaUnbindTexture(&_texMaskD) );

  for (int s = 0;s<NSTREAM;s++)
    {
      _d_volumeCach[s].Dealloc();
      _d_volumeCost[s].Dealloc();
      _d_volumeNIOk[s].Dealloc();
      _dt_LayeredProjection[s].Dealloc();
    }

  _dt_mask.Dealloc();
  _dt_LayeredImages.Dealloc();

#ifdef USEDILATEMASK
  delete [] _dilateMask;
#endif
}

void InterfaceMicMacGpGpu::SetMask( pixel* dataMask, uint2 dimMask )
{
  // Initialisation du masque utilisateur
  // le masque est passe en texture
  _dt_mask.Dealloc();
  _dt_mask.InitImage(dimMask,dataMask);
  if(!_dt_mask.bindTexture(_texMask))
    {
      _dt_mask.OutputInfo();
      _dt_mask.CData2D::Name();
    }

#ifdef USEDILATEMASK
  _mask.bindTexture(_texMaskD);
#endif

}

void InterfaceMicMacGpGpu::SetImages( float* dataImage, uint2 dimImage, int nbLayer )
{
  // Images vers Textures Gpu
  _dt_LayeredImages.CData3D::Realloc(dimImage,nbLayer);
  _dt_LayeredImages.copyHostToDevice(dataImage);
  _dt_LayeredImages.bindTexture(_texImages);
}

void InterfaceMicMacGpGpu::BasicCorrelation(int nbLayer)
{
  // Lancement des algo GPGPU

  // Re-dimensionner les strucutres de données si elles ont été modifiées
  ReallocAllDeviceData(_param.ZLocInter);

  // Calcul de dimension des threads des kernels
  //--------------- calcul de dimension du kernel de correlation ---------------
  dim3	threads( BLOCKDIM, BLOCKDIM, 1);
  uint2	thd2D		= make_uint2(threads);
  uint2	actiThsCo	= thd2D - 2 * _param.rayVig;
  uint2	block2D		= iDivUp(_param.dimDTer,actiThsCo);
  dim3	blocks(block2D.x , block2D.y, nbLayer * _param.ZLocInter);
  const uint s = 0;     // Affection du stream de calcul

  // Copier les projections du host --> device
  _dt_LayeredProjection[s].copyHostToDevice(_hVolumeProj.pData());

  // Indique que la copie est terminée pour le thread de calcul des projections
  SetPreComp(true);

  // Lié de données de projections du device avec la texture de projections
  _dt_LayeredProjection[s].bindTexture(GetTeXProjection(s));

  // Lancement du calcul de correlation
  KernelCorrelation(s, *(GetStream(s)),blocks, threads,  _d_volumeNIOk[s].pData(), _d_volumeCach[s].pData(), actiThsCo);

  // Libérer la texture de projection
  checkCudaErrors( cudaUnbindTexture(&(GetTeXProjection(s))));

  // Lancement du calcul de multi-correlation

  //-------------	calcul de dimension du kernel de multi-correlation NON ATOMIC ------------
  uint2	actiThs_NA	= SBLOCKDIM - make_uint2( SBLOCKDIM % _param.dimVig.x, SBLOCKDIM % _param.dimVig.y);
  dim3	threads_mC_NA(SBLOCKDIM, SBLOCKDIM, 1);
  uint2	block2D_mC_NA	= iDivUp(_param.dimCach,actiThs_NA);
  dim3	blocks_mC_NA(block2D_mC_NA.x,block2D_mC_NA.y,_param.ZLocInter);

  KernelmultiCorrelation( *(GetStream(s)),blocks_mC_NA, threads_mC_NA,  _d_volumeCost[s].pData(), _d_volumeCach[s].pData(), _d_volumeNIOk[s].pData(), actiThs_NA);

  // Copier les resultats de calcul des couts du device vers le host!
  _d_volumeCost[s].CopyDevicetoHost(_hVolumeCost[GetIdBuf()].pData());

}

void InterfaceMicMacGpGpu::BasicCorrelationStream( float* hostVolumeCost, float2* hostVolumeProj, int nbLayer, uint interZ )
{

  ReallocAllDeviceData(_param.ZLocInter);
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
}

void InterfaceMicMacGpGpu::ReallocDeviceData(int nStream, uint interZ)
{
    _d_volumeCost[nStream].Realloc(_param.dimTer,     interZ);
    _d_volumeCach[nStream].Realloc(_param.dimCach,    _param.nbImages * interZ);
    _d_volumeNIOk[nStream].Realloc(_param.dimTer,     interZ);
}

void InterfaceMicMacGpGpu::ReallocAllDeviceData(uint interZ )
{
  for (int s = 0;s<NSTREAM;s++)
    {
      _d_volumeCost[s].SetDimension(_param.dimTer,interZ);
      _d_volumeCach[s].SetDimension(_param.dimCach,_param.nbImages * interZ);
      _d_volumeNIOk[s].SetDimension(_param.dimTer,interZ);

      if (_d_volumeCost[s].GetSizeofMalloc() < _d_volumeCost[s].Sizeof() )
          ReallocDeviceData(s, interZ);

      _d_volumeCost[s].Memset(_param.IntDefault);
      _d_volumeCach[s].Memset(_param.IntDefault);
      _d_volumeNIOk[s].Memset(0);
  }
}

void InterfaceMicMacGpGpu::ReallocAllDeviceDataAsync(uint interZ)
{
  for (int s = 0;s<NSTREAM;s++)
    {
      _d_volumeCost[s].SetDimension(_param.dimTer,interZ);
      _d_volumeCach[s].SetDimension(_param.dimCach,_param.nbImages * interZ);
      _d_volumeNIOk[s].SetDimension(_param.dimTer,interZ);

      if (_d_volumeCost[s].GetSizeofMalloc() < _d_volumeCost[s].Sizeof() )
          ReallocDeviceData(s, interZ);


      _d_volumeCost[s].MemsetAsync(_param.IntDefault, *(GetStream(s)));
      _d_volumeCach[s].MemsetAsync(_param.IntDefault, *(GetStream(s)));
      _d_volumeNIOk[s].MemsetAsync(0, *(GetStream(s)));
  }
}

textureReference& InterfaceMicMacGpGpu::GetTeXProjection( int TexSel )
{
    switch (TexSel)
    {
    case 0:
        return _texProjections_00;
    case 1:
        return _texProjections_01;
    case 2:
        return _texProjections_02;
    case 3:
        return _texProjections_03;
    default:
        return _texProjections_00;
    }
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

void InterfaceMicMacGpGpu::MemsetProj()
{
    _hVolumeProj.Memset(Param().IntDefault);
}

float *InterfaceMicMacGpGpu::OuputCost(uint id)
{
    return _hVolumeCost[id].pData();
}

float2 *InterfaceMicMacGpGpu::InputProj()
{
    return _hVolumeProj.pData();
}

void InterfaceMicMacGpGpu::DeallocVolumes()
{
    for (int i = 0; i < SIZERING; ++i)
        if(!_hVolumeCost[i].GetSizeofMalloc())
            _hVolumeCost[i].Dealloc();

    if(!_hVolumeProj.GetSizeofMalloc())
        _hVolumeProj.Dealloc();
}

void InterfaceMicMacGpGpu::InitJob()
{
    if(UseMultiThreading())
    {
        ResetIdBuffer();
        SetPreComp(true);
    }
}

void InterfaceMicMacGpGpu::MallocInfo()
{
  std::cout << "Malloc Info GpGpu\n";
  GpGpuTools::OutputInfoGpuMemory();
  _d_volumeCost[0].MallocInfo();
  _d_volumeCach[0].MallocInfo();
  _d_volumeNIOk[0].MallocInfo();
  _dt_mask.CData2D::MallocInfo();
  _dt_LayeredImages.CData3D::MallocInfo();
  _dt_LayeredProjection[0].CData3D::MallocInfo();
}

#ifdef USEDILATEMASK
void InterfaceMicMacGpGpu::dilateMask(uint2 dim )
{

  _dimDilateMask = dim + 2*_param.rayVig.x;

  _dilateMask = new pixel[size(_dimDilateMask)];

  dilateKernel(_dilateMask, _param.rayVig.x,dim);

  //GpGpuTools::OutputArray(_dilateMask,_dimDilateMask);

}

pixel* InterfaceMicMacGpGpu::GetDilateMask()
{
  return _dilateMask;
}

pixel InterfaceMicMacGpGpu::ValDilMask(int2 pt)
{
  return (oI(pt,0) || oSE(pt,_dimDilateMask)) ? 0 : _dilateMask[to1D(pt,_dimDilateMask)];
}

#endif

pCorGpu InterfaceMicMacGpGpu::Param()
{
  return _param;
}


