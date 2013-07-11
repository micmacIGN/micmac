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

  _volumeCost->SetName("_volumeCost");
  _volumeCach->SetName("_volumeCach");
  _volumeNIOk->SetName("_volumeNIOk");
  _mask.CData2D::SetName("_mask");
  _LayeredImages.CData3D::SetName("_LayeredImages");
  _LayeredProjection->CData3D::SetName("_LayeredProjection");

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

  // Parametres texture des Caches
  //  _texCache.addressMode[0]	= cudaAddressModeWrap;
  //  _texCache.addressMode[1]	= cudaAddressModeWrap;
  //  _texCache.filterMode		= cudaFilterModePoint; //cudaFilterModeLinear cudaFilterModePoint
  //  _texCache.normalized		= false;

}

InterfaceMicMacGpGpu::~InterfaceMicMacGpGpu()
{
  for (int s = 0;s<NSTREAM;s++)
    checkCudaErrors( cudaStreamDestroy(*(GetStream(s))));

  DeallocMemory();
}

void InterfaceMicMacGpGpu::SetSizeBlock( uint Zinter, Rect Ter)
{

  uint oldSizeTer = _param.sizeDTer;

  _param.SetDimension(Ter,Zinter);

  //if(Param().MaskNoNULL())
  {
      CopyParamTodevice(_param);

      for (int s = 0;s<NSTREAM;s++)
        {
          _LayeredProjection[s].Realloc(_param.dimSTer,_param.nbImages * _param.ZLocInter);

          if (oldSizeTer < _param.sizeDTer)
            AllocMemory(s);
        }
  }
}

void InterfaceMicMacGpGpu::SetSizeBlock( uint Zinter )
{
  SetSizeBlock( Zinter,_param.RTer());
}

void InterfaceMicMacGpGpu::AllocMemory(int nStream)
{

  _volumeCost[nStream].Realloc(_param.dimTer,_param.ZLocInter);

  if(!_volumeCach[nStream].Realloc(_param.dimCach, _param.nbImages * _param.ZLocInter))
    _param.outConsole();

  _volumeNIOk[nStream].Realloc(_param.dimTer,_param.ZLocInter);
}

void InterfaceMicMacGpGpu::DeallocMemory()
{
  checkCudaErrors( cudaUnbindTexture(&_texImages) );
  checkCudaErrors( cudaUnbindTexture(&_texMask) );
  checkCudaErrors( cudaUnbindTexture(&_texMaskD) );

  for (int s = 0;s<NSTREAM;s++)
    {
      _volumeCach[s].Dealloc();
      _volumeCost[s].Dealloc();
      _volumeNIOk[s].Dealloc();
      _LayeredProjection[s].Dealloc();
    }

  _mask.Dealloc();
  _LayeredImages.Dealloc();



#ifdef USEDILATEMASK
  delete [] _dilateMask;
#endif
}

void InterfaceMicMacGpGpu::SetMask( pixel* dataMask, uint2 dimMask )
{
  // Initialisation du masque utilisateur
  // le masque est passe en texture
  _mask.Dealloc();
  _mask.InitImage(dimMask,dataMask);
  if(!_mask.bindTexture(_texMask))
    {
      _mask.OutputInfo();
      _mask.CData2D::Name();
    }

#ifdef USEDILATEMASK
  _mask.bindTexture(_texMaskD);
#endif

}

void InterfaceMicMacGpGpu::SetParameter( Rect Ter, int nbLayer , uint2 dRVig , uint2 dimImg, float mAhEpsilon, uint samplingZ, int uvINTDef , uint interZ )
{

  // Initialisation des parametres constants
  _param.SetParamInva( dRVig * 2 + 1,dRVig, dimImg, mAhEpsilon, samplingZ, uvINTDef, nbLayer);
  // Initialisation des parametres de dimensions du terrain
  SetSizeBlock( interZ, Ter);

  for (int s = 0;s<NSTREAM;s++)
    AllocMemory(s);
}

void InterfaceMicMacGpGpu::SetImages( float* dataImage, uint2 dimImage, int nbLayer )
{
  // Images vers Textures Gpu
  _LayeredImages.CData3D::Realloc(dimImage,nbLayer);
  _LayeredImages.copyHostToDevice(dataImage);
  _LayeredImages.bindTexture(_texImages);
}

void InterfaceMicMacGpGpu::BasicCorrelation(int nbLayer)
{
  // Lancement des algo GPGPU

  // Re-dimensionner les strucutres de données si elles ont été modifiées
  ResizeInputVolume(nbLayer,_param.ZLocInter);

  // Calcul de dimension des threads des kernels
  //--------------- calcul de dimension du kernel de correlation ---------------
  dim3	threads( BLOCKDIM, BLOCKDIM, 1);
  uint2	thd2D		= make_uint2(threads);
  uint2	actiThsCo	= thd2D - 2 * _param.rayVig;
  uint2	block2D		= iDivUp(_param.dimDTer,actiThsCo);
  dim3	blocks(block2D.x , block2D.y, nbLayer * _param.ZLocInter);
  const uint s = 0;     // Affection du stream de calcul

  // Copier les projections du host --> device
  _LayeredProjection[s].copyHostToDevice(_hVolumeProj.pData());

  // Indique que la copie est terminée pour le thread de calcul des projections
  SetPreComp(true);

  // Lié de données de projections du device avec la texture de projections
  _LayeredProjection[s].bindTexture(GetTeXProjection(s));




  // Lancement du calcul de correlation
  KernelCorrelation(s, *(GetStream(s)),blocks, threads,  _volumeNIOk[s].pData(), _volumeCach[s].pData(), actiThsCo);

  // Libérer la texture de projection
  checkCudaErrors( cudaUnbindTexture(&(GetTeXProjection(s))));

  // Lancement du calcul de multi-correlation

  //-------------	calcul de dimension du kernel de multi-correlation NON ATOMIC ------------
  uint2	actiThs_NA	= SBLOCKDIM - make_uint2( SBLOCKDIM % _param.dimVig.x, SBLOCKDIM % _param.dimVig.y);
  dim3	threads_mC_NA(SBLOCKDIM, SBLOCKDIM, 1);
  uint2	block2D_mC_NA	= iDivUp(_param.dimCach,actiThs_NA);
  dim3	blocks_mC_NA(block2D_mC_NA.x,block2D_mC_NA.y,_param.ZLocInter);

  KernelmultiCorrelation( *(GetStream(s)),blocks_mC_NA, threads_mC_NA,  _volumeCost[s].pData(), _volumeCach[s].pData(), _volumeNIOk[s].pData(), actiThs_NA);

  // Copier les resultats de calcul des couts du device vers le host!
  _volumeCost[s].CopyDevicetoHost(_hVolumeCost[GetIdBuf()].pData());

}

void InterfaceMicMacGpGpu::BasicCorrelationStream( float* hostVolumeCost, float2* hostVolumeProj, int nbLayer, uint interZ )
{

  ResizeInputVolume(nbLayer,_param.ZLocInter);
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
          _LayeredProjection[s].copyHostToDeviceASync(hostVolumeProj + (Z  + s) *_LayeredProjection->GetSize(),*(GetStream(s)));

      if (Z == 0) SetPreComp(true); // A faire quand toutes les copies asynchrones sont terminées!!

      for (uint s = 0;s<nstream;s++)
        {
          _LayeredProjection[s].bindTexture(GetTeXProjection(s)); // peut etre qu'ici également
          KernelCorrelation(s, *(GetStream(s)),blocks, threads,  _volumeNIOk[s].pData(), _volumeCach[s].pData(), actiThsCo);

          KernelmultiCorrelation( *(GetStream(s)),blocks_mC, threads_mC,  _volumeCost[s].pData(), _volumeCach[s].pData(), _volumeNIOk[s].pData(), actiThs);
        }

      for (uint s = 0;s<nstream;s++)
          _volumeCost[s].CopyDevicetoHostASync(hostVolumeCost + (Z + s)*_volumeCost[s].GetSize(),*(GetStream(s)));

      for (uint s = 0;s<nstream;s++)
        checkCudaErrors( cudaUnbindTexture(&(GetTeXProjection(s))) ); // il faut mettre un signal pour unbinder...

      Z += nstream * _param.ZLocInter;
    }

    //SetComputeNextProj(true); // A faire quand toutes les copies asynchrones sont terminées!!

//  for (uint s = 0;s<NSTREAM;s++)
  //    cudaStreamSynchronize(*(GetStream(s));

    checkCudaErrors(cudaDeviceSynchronize());
}

void InterfaceMicMacGpGpu::ResizeInputVolume( int nbLayer, uint interZ )
{
  for (int s = 0;s<NSTREAM;s++)
    {
      _volumeCost[s].SetDimension(_param.dimTer,interZ);
      _volumeCach[s].SetDimension(_param.dimCach,nbLayer * interZ);
      _volumeNIOk[s].SetDimension(_param.dimTer,interZ);

      if (_volumeCost[s].GetSizeofMalloc() < _volumeCost[s].Sizeof() )
        {
          //std::cout << "Realloc Device Data" << "\n";
          _volumeCost[s].Realloc(_param.dimTer,interZ);
          if (!_volumeCach[s].Realloc(_param.dimCach,nbLayer * interZ))
            _param.outConsole();
          _volumeNIOk[s].Realloc(_param.dimTer,interZ);
        }
      //----------------------------------------------------------------------------

      _volumeCost[s].Memset(_param.IntDefault);

      // FIN MODIFICATION TEST PERFORMANCE
      _volumeCach[s].Memset(_param.IntDefault);
      // FIN MODIFICATION TEST PERFORMANCE

      _volumeNIOk[s].Memset(0);
  }
}

void InterfaceMicMacGpGpu::ResizeVolumeAsync(int nbLayer, uint interZ)
{
  for (int s = 0;s<NSTREAM;s++)
    {
      _volumeCost[s].SetDimension(_param.dimTer,interZ);
      _volumeCach[s].SetDimension(_param.dimCach,nbLayer * interZ);
      _volumeNIOk[s].SetDimension(_param.dimTer,interZ);

      if (_volumeCost[s].GetSizeofMalloc() < _volumeCost[s].Sizeof() )
        {
          //std::cout << "Realloc Device Data" << "\n";
          _volumeCost[s].Realloc(_param.dimTer,interZ);
          if (!_volumeCach[s].Realloc(_param.dimCach,nbLayer * interZ))
            _param.outConsole();
          _volumeNIOk[s].Realloc(_param.dimTer,interZ);
        }
      //----------------------------------------------------------------------------

      _volumeCost[s].MemsetAsync(_param.IntDefault, *(GetStream(s)));
      _volumeCach[s].MemsetAsync(_param.IntDefault, *(GetStream(s)));
      _volumeNIOk[s].MemsetAsync(0, *(GetStream(s)));
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
  bool gpuThreadLoop    = true;
  ResetIdBuffer();

  while (gpuThreadLoop)
    {
      if (GetCompute()!=0)
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

void InterfaceMicMacGpGpu::ReallocInputProjection(uint2 dim, uint l)
{
    _hVolumeProj.Realloc(dim,l);
}

void InterfaceMicMacGpGpu::ReallocOutCost(uint2 dim, uint l)
{
    _hVolumeCost[0].Realloc(dim,l);
    _hVolumeCost[1].Realloc(dim,l);
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
    if(!_hVolumeCost[0].GetSizeofMalloc())
        _hVolumeCost[0].Dealloc();
    if(!_hVolumeCost[1].GetSizeofMalloc())
        _hVolumeCost[1].Dealloc();
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
  _volumeCost[0].MallocInfo();
  _volumeCach[0].MallocInfo();
  _volumeNIOk[0].MallocInfo();
  _mask.CData2D::MallocInfo();
  _LayeredImages.CData3D::MallocInfo();
  _LayeredProjection[0].CData3D::MallocInfo();
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


