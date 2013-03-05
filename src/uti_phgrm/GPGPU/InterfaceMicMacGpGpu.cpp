#include "GpGpu/InterfaceMicMacGpGpu.h"

InterfaceMicMacGpGpu::InterfaceMicMacGpGpu():
_texMask(getMask()),
_texMaskD(getMaskD()),
_texImages(getImage()),
_texProjections_00(getProjection(0)),
_texProjections_01(getProjection(1)),
_texProjections_02(getProjection(2)),
_texProjections_03(getProjection(3))
{
	for (int s = 0;s<NSTREAM;s++)
		checkCudaErrors( cudaStreamCreate(GetStream(s)));

	_gpuThread = new boost::thread(&InterfaceMicMacGpGpu::MTComputeCost,this);
	_gpuThread->detach();
	SetZCToCopy(0);
	SetZToCompute(0);

	_volumeCost->Name("_volumeCost");
	_volumeCach->Name("_volumeCach");
	_volumeNIOk->Name("_volumeNIOk");
// 	_mask.Name("_mask");
// 	_LayeredImages.Name("_LayeredImages");
// 	_LayeredProjection->Name("_LayeredProjection");
}

InterfaceMicMacGpGpu::~InterfaceMicMacGpGpu()
{
	for (int s = 0;s<NSTREAM;s++)
		checkCudaErrors( cudaStreamDestroy(*(GetStream(s))));
}

void InterfaceMicMacGpGpu::SetSizeBlock( Rect Ter, uint Zinter )
{

	uint oldSizeTer = _param.sizeTer;

	_param.SetDimension(Ter,Zinter);

	CopyParamTodevice(_param);

	for (int s = 0;s<NSTREAM;s++)
	{
		_LayeredProjection[s].Realloc(_param.dimSTer,_param.nbImages * _param.ZLocInter);

		if (oldSizeTer < _param.sizeTer)
			AllocMemory(s);
	}

}

void InterfaceMicMacGpGpu::SetSizeBlock( uint Zinter )
{
	SetSizeBlock( _param.GetRMask(), Zinter );
}

void InterfaceMicMacGpGpu::AllocMemory(int nStream)
{

	if(!((	_volumeCost[nStream].Realloc(_param.rDiTer,_param.ZLocInter)) &&
		(	_volumeCach[nStream].Realloc(_param.dimCach, _param.nbImages * _param.ZLocInter)) &&
		(	_volumeNIOk[nStream].Realloc(_param.rDiTer,_param.ZLocInter))))
		_param.outConsole();
	
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
	_mask.InitImage(dimMask,dataMask);
	_mask.bindTexture(_texMask);
	_mask.bindTexture(_texMaskD);
}

void InterfaceMicMacGpGpu::InitParam( Rect Ter, int nbLayer , uint2 dRVig , uint2 dimImg, float mAhEpsilon, uint samplingZ, int uvINTDef , uint interZ )
{
	
	// Parametres texture des projections
	for (int s = 0;s<NSTREAM;s++)
	{
		GetTeXProjection(s).addressMode[0]	= cudaAddressModeBorder;
		GetTeXProjection(s).addressMode[1]	= cudaAddressModeBorder;	
		GetTeXProjection(s).filterMode		= cudaFilterModeLinear; //cudaFilterModePoint cudaFilterModeLinear
		GetTeXProjection(s).normalized		= true;
	}
	// Parametres texture des Images
	_texImages.addressMode[0]	= cudaAddressModeWrap;
	_texImages.addressMode[1]	= cudaAddressModeWrap;
	_texImages.filterMode		= cudaFilterModeLinear; //cudaFilterModeLinear cudaFilterModePoint
	_texImages.normalized		= true;


	// Initialisation des parametres constants
	_param.SetParamInva( dRVig * 2 + 1,dRVig, dimImg, mAhEpsilon, samplingZ, uvINTDef, nbLayer);
	// Initialisation des parametres de dimensions du terrain
	SetSizeBlock( Ter, interZ );

	for (int s = 0;s<NSTREAM;s++)
		AllocMemory(s);
}

void InterfaceMicMacGpGpu::SetImages( float* dataImage, uint2 dimImage, int nbLayer )
{
	// Images vers Textures Gpu
	_LayeredImages.CData3D::Malloc(dimImage,nbLayer);
	_LayeredImages.copyHostToDevice(dataImage);
	_LayeredImages.bindTexture(_texImages);
}

void InterfaceMicMacGpGpu::BasicCorrelation( float* hostVolumeCost, float2* hostVolumeProj, int nbLayer, uint interZ )
{
	// Lancement des algo GPGPU

	// Re-dimensionner les strucutres de données si elles ont été modifiées
	ResizeVolume(nbLayer,_param.ZLocInter);
	

	// Cacul de dimension des threads des kernels
	/*--------------- calcul de dimension du kernel de correlation ---------------*/
	dim3	threads( BLOCKDIM, BLOCKDIM, 1);
	uint2	thd2D		= make_uint2(threads);
	uint2	actiThsCo	= thd2D - 2 * _param.rVig;
	uint2	block2D		= iDivUp(_param.dimTer,actiThsCo);
	dim3	blocks(block2D.x , block2D.y, nbLayer * _param.ZLocInter);

	/*-------------	calcul de dimension du kernel de multi-correlation ------------*/

	// Calcul pour eviter la limitaion de nomre de threads par block (ici 1024)
	ushort sNbTh	=  (ushort)((float)sqrt( (float)_param.nbImages / MAX_THREADS_PER_BLOCK) * SBLOCKDIM ) + 1;
	ushort	BLDIM	= ( 4 - sNbTh ) * SBLOCKDIM / 3 ;
	uint2	actiThs		= BLDIM - make_uint2( BLDIM % _param.dimVig.x, BLDIM % _param.dimVig.y);
	dim3	threads_mC(BLDIM, BLDIM, nbLayer);
	uint2	block2D_mC	= iDivUp(_param.dimCach,actiThs);
	dim3	blocks_mC(block2D_mC.x,block2D_mC.y,_param.ZLocInter);
	const uint s = 0; // Affection du stream de calcul

	// Copier les projections du host --> device
	_LayeredProjection[s].copyHostToDevice(hostVolumeProj);
	// Indique que la copie est terminée pour le thread de calcul des projections
	SetComputeNextProj(true);
	// Lié de données de projections du device avec la texture de projections
	_LayeredProjection[s].bindTexture(GetTeXProjection(s));

	// Lancement du calcul de correlation 
	KernelCorrelation(s, *(GetStream(s)),blocks, threads,  _volumeNIOk[s].pData(), _volumeCach[s].pData(), actiThsCo);
	// Lancement du calcul de multi-correlation
	KernelmultiCorrelation( *(GetStream(s)),blocks_mC, threads_mC,  _volumeCost[s].pData(), _volumeCach[s].pData(), _volumeNIOk[s].pData(), actiThs,sNbTh);

	// Libérer la texture de projection
	checkCudaErrors( cudaUnbindTexture(&(GetTeXProjection(s))));
	// Copier les resultats de calcul des couts du device vers le host!
	_volumeCost[s].CopyDevicetoHost(hostVolumeCost);	
	
}

void InterfaceMicMacGpGpu::BasicCorrelationStream( float* hostVolumeCost, float2* hostVolumeProj, int nbLayer, uint interZ )
{

	ResizeVolume(nbLayer,_param.ZLocInter);
	uint Z = 0;

	/*--------------- calcul de dimension du kernel de correlation ---------------*/

	dim3	threads( BLOCKDIM, BLOCKDIM, 1);
	uint2	thd2D		= make_uint2(threads);
	uint2	actiThsCo	= thd2D - 2 * _param.rVig;
	uint2	block2D		= iDivUp(_param.dimTer,actiThsCo);
	dim3	blocks(block2D.x , block2D.y, nbLayer * _param.ZLocInter);

	/*-------------	calcul de dimension du kernel de multi-correlation ------------*/

	

	uint2	actiThs		= SBLOCKDIM - make_uint2( SBLOCKDIM % _param.dimVig.x, SBLOCKDIM % _param.dimVig.y);
	dim3	threads_mC(SBLOCKDIM, SBLOCKDIM, nbLayer);
	uint2	block2D_mC	= iDivUp(_param.dimCach,actiThs);
	dim3	blocks_mC(block2D_mC.x,block2D_mC.y,_param.ZLocInter);

	while(Z < interZ)
	{
		//const uint nstream = (NSTREAM * _param.ZLocInter) >= (interZ - Z)  ? (interZ - Z) : NSTREAM;
		const uint nstream = 1;

		for (uint s = 0;s<nstream;s++)
		{
			//_LayeredProjection[s].copyHostToDevice(hostVolumeProj);
			_LayeredProjection[s].copyHostToDeviceASync(hostVolumeProj + (Z  + s) *_LayeredProjection->GetSize(),*(GetStream(s)));
			_LayeredProjection[s].bindTexture(GetTeXProjection(s));
		}

		for (uint s = 0;s<nstream;s++)
		{
			KernelCorrelation(s, *(GetStream(s)),blocks, threads,  _volumeNIOk[s].pData(), _volumeCach[s].pData(), actiThsCo);
			KernelmultiCorrelation( *(GetStream(s)),blocks_mC, threads_mC,  _volumeCost[s].pData(), _volumeCach[s].pData(), _volumeNIOk[s].pData(), actiThs,1);
		}

		for (uint s = 0;s<nstream;s++)
		{	
			checkCudaErrors( cudaUnbindTexture(&(GetTeXProjection(s))) );
			_volumeCost[s].CopyDevicetoHostASync(hostVolumeCost + (Z + s)*_volumeCost[s].GetSize(),*(GetStream(s)));	
		}
		Z += nstream * _param.ZLocInter;
	}

	checkCudaErrors(cudaDeviceSynchronize());
}

uint2 InterfaceMicMacGpGpu::GetDimensionTerrain()
{
	return _param.rDiTer;
}

bool InterfaceMicMacGpGpu::MaskNoNULL()
{
	return _param.MaskNoNULL();
}

uint InterfaceMicMacGpGpu::GetSample()
{
	return _param.sampTer;
}

float InterfaceMicMacGpGpu::GetDefaultVal()
{
	return _param.DefaultVal;
}

uint2 InterfaceMicMacGpGpu::GetSDimensionTerrain()
{
	return _param.dimSTer;
}

int InterfaceMicMacGpGpu::GetIntDefaultVal()
{
	return _param.IntDefault;
}

void InterfaceMicMacGpGpu::ResizeVolume( int nbLayer, uint interZ )
{
	for (int s = 0;s<NSTREAM;s++)
	{
		_volumeCost[s].SetDimension(_param.rDiTer,interZ);
		_volumeCach[s].SetDimension(_param.dimCach,nbLayer * interZ);
		_volumeNIOk[s].SetDimension(_param.rDiTer,interZ);

		if (_volumeCost[s].GetSizeofMalloc() < _volumeCost[s].Sizeof() )
		{
			//std::cout << "Realloc Device Data" << "\n";
			_volumeCost[s].Realloc(_param.rDiTer,interZ);
			_volumeCach[s].Realloc(_param.dimCach,nbLayer * interZ);
			_volumeNIOk[s].Realloc(_param.rDiTer,interZ);
		}
		//----------------------------------------------------------------------------

		_volumeCost[s].Memset(_param.IntDefault);
		_volumeCach[s].Memset(_param.IntDefault);
		_volumeNIOk[s].Memset(0);
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

void InterfaceMicMacGpGpu::MTComputeCost()
{
	bool gpuThreadLoop = true;

	while (gpuThreadLoop)
	{
		if (GetZToCompute()!=0 && GetZCtoCopy()==0)
		{
			uint interZ = GetZToCompute();
			SetZToCompute(0);
			BasicCorrelation(_vCost, _vProj, _param.nbImages, interZ);
			SetZCToCopy(interZ);
		}
	}
}

void InterfaceMicMacGpGpu::SetHostVolume( float* vCost, float2* vProj )
{
	_vCost = vCost;
	_vProj = vProj;	
}

uint InterfaceMicMacGpGpu::GetZToCompute()
{
	 boost::lock_guard<boost::mutex> guard(_mutex);
	 return _ZCompute;
}

void InterfaceMicMacGpGpu::SetZToCompute( uint Z )
{
	boost::lock_guard<boost::mutex> guard(_mutex);
	_ZCompute = Z;
}

uint InterfaceMicMacGpGpu::GetZCtoCopy()
{
	boost::lock_guard<boost::mutex> guard(_mutexC);
	return _ZCCopy;

}

void InterfaceMicMacGpGpu::SetZCToCopy( uint Z )
{
	boost::lock_guard<boost::mutex> guard(_mutexC);
	_ZCCopy = Z;
}

bool InterfaceMicMacGpGpu::GetComputeNextProj()
{
	boost::lock_guard<boost::mutex> guard(_mutexCompute);
	return _computeNextProj;
}

void InterfaceMicMacGpGpu::SetComputeNextProj( bool compute )
{
	boost::lock_guard<boost::mutex> guard(_mutexCompute);
	_computeNextProj = compute;
}

int InterfaceMicMacGpGpu::GetComputedZ()
{
	return _computedZ;
}

void InterfaceMicMacGpGpu::SetComputedZ( int computedZ )
{
	_computedZ = computedZ;
}

Rect InterfaceMicMacGpGpu::rMask()
{
	return _param.GetRMask();
}

Rect InterfaceMicMacGpGpu::rUTer()
{
	return _param.GetRUTer();
}

#ifdef USEDILATEMASK
void InterfaceMicMacGpGpu::dilateMask(uint2 dim )
{

	_dimDilateMask = dim + 2*_param.rVig.x;

	_dilateMask = new pixel[size(_dimDilateMask)];

	dilateKernel(_dilateMask, _param.rVig.x,dim);

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
