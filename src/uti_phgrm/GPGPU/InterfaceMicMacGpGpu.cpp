#include "GpGpu/InterfaceMicMacGpGpu.h"


NS_ParamMICMAC::InterfaceMicMacGpGpu::InterfaceMicMacGpGpu():
_texMask(getMask()),
_texImages(getImage()),
_texProjections(getProjection())
{
}

void NS_ParamMICMAC::InterfaceMicMacGpGpu::SetSizeBlock( uint2 ter0, uint2 ter1, uint Zinter )
{

	uint oldSizeTer = _param.sizeTer;

	_param.SetDimension(ter0,ter1,Zinter);

	CopyParamTodevice(_param);

	_LayeredProjection.Realloc(_param.dimSTer,_param.nbImages * _param.ZInter);

	if (oldSizeTer < _param.sizeTer)
		AllocMemory();

}

void NS_ParamMICMAC::InterfaceMicMacGpGpu::SetSizeBlock( uint Zinter )
{
	SetSizeBlock( make_uint2(_param.ptMask0), make_uint2(_param.ptMask1), Zinter );
}

void NS_ParamMICMAC::InterfaceMicMacGpGpu::AllocMemory()
{
	_volumeCost.Realloc(_param.rDiTer,_param.ZInter);
	_volumeCach.Realloc(_param.dimCach, _param.nbImages * _param.ZInter);
	_volumeNIOk.Realloc(_param.rDiTer,_param.ZInter);
}

void NS_ParamMICMAC::InterfaceMicMacGpGpu::DeallocMemory()
{
	checkCudaErrors( cudaUnbindTexture(&_texImages) );	
	checkCudaErrors( cudaUnbindTexture(&_texMask) );	

	_volumeCach.Dealloc();
	_volumeCost.Dealloc();
	_volumeNIOk.Dealloc();

	_mask.Dealloc();
	_LayeredImages.Dealloc();
	_LayeredProjection.Dealloc();
}

void NS_ParamMICMAC::InterfaceMicMacGpGpu::SetMask( pixel* dataMask, uint2 dimMask )
{
	_mask.InitImage(dimMask,dataMask);
	_mask.bindTexture(_texMask);
}

void NS_ParamMICMAC::InterfaceMicMacGpGpu::InitParam( uint2 ter0, uint2 ter1, int nbLayer , uint2 dRVig , uint2 dimImg, float mAhEpsilon, uint samplingZ, int uvINTDef , uint interZ )
{
	
	// Parametres texture des projections
	_texProjections.addressMode[0]	= cudaAddressModeClamp;
	_texProjections.addressMode[1]	= cudaAddressModeClamp;	
	_texProjections.filterMode		= cudaFilterModeLinear; //cudaFilterModePoint cudaFilterModeLinear
	_texProjections.normalized		= true;

	// Parametres texture des Images
	_texImages.addressMode[0]	= cudaAddressModeWrap;
	_texImages.addressMode[1]	= cudaAddressModeWrap;
	_texImages.filterMode		= cudaFilterModePoint; //cudaFilterModeLinear cudaFilterModePoint
	_texImages.normalized		= true;

	_param.SetParamInva( dRVig * 2 + 1,dRVig, dimImg, mAhEpsilon, samplingZ, uvINTDef, nbLayer);
	SetSizeBlock( ter0, ter1, interZ );
	AllocMemory();
	//return h;
}

void NS_ParamMICMAC::InterfaceMicMacGpGpu::SetImages( float* dataImage, uint2 dimImage, int nbLayer )
{
	_LayeredImages.CData3D::Malloc(dimImage,nbLayer);
	_LayeredImages.copyHostToDevice(dataImage);
	_LayeredImages.bindTexture(_texImages);
}

void NS_ParamMICMAC::InterfaceMicMacGpGpu::BasicCorrelation( float* hostVolumeCost, float2* hostVolumeProj, int nbLayer, uint interZ )
{
	
	_volumeCost.SetDimension(_param.rDiTer,interZ);
	_volumeCach.SetDimension(_param.dimCach,nbLayer * interZ);
	_volumeNIOk.SetDimension(_param.rDiTer,interZ);

	if (_volumeCost.GetSizeofMalloc() < _volumeCost.Sizeof() )
	{
		std::cout << "Realloc Device Data" << "\n";
		_volumeCost.Realloc(_param.rDiTer,interZ);
		_volumeCach.Realloc(_param.dimCach,nbLayer * interZ);
		_volumeNIOk.Realloc(_param.rDiTer,interZ);
	}
	//----------------------------------------------------------------------------

	_volumeCost.Memset(_param.IntDefault);
	_volumeCach.Memset(_param.IntDefault);
	_volumeNIOk.Memset(0);

	_LayeredProjection.copyHostToDevice(hostVolumeProj);
	_LayeredProjection.bindTexture(_texProjections);

	// --------------- calcul de dimension du kernel de correlation --------------

	dim3	threads( BLOCKDIM, BLOCKDIM, 1);
	uint2	thd2D		= make_uint2(threads);
	uint2	actiThsCo	= thd2D - 2 * _param.rVig;
	uint2	block2D		= iDivUp(_param.dimTer,actiThsCo);
	dim3	blocks(block2D.x , block2D.y, nbLayer * interZ);

	//-------------	calcul de dimension du kernel de multi-correlation ------------

	uint2	actiThs		= SBLOCKDIM - make_uint2( SBLOCKDIM % _param.dimVig.x, SBLOCKDIM % _param.dimVig.y);
	dim3	threads_mC(SBLOCKDIM, SBLOCKDIM, nbLayer);
	uint2	block2D_mC	= iDivUp(_param.dimCach,actiThs);
	dim3	blocks_mC(block2D_mC.x,block2D_mC.y,interZ);

	//-----------------------  KERNEL  Correlation  -------------------------------
	KernelCorrelation( blocks, threads, _volumeNIOk.pData(), _volumeCach.pData(), actiThsCo);

	//-------------------  KERNEL  Multi Correlation  ------------------------------
	KernelmultiCorrelation( blocks_mC, threads_mC, _volumeCost.pData(), _volumeCach.pData(), _volumeNIOk.pData(), actiThs);
	
	//----------------------------------------------------------------------------

	checkCudaErrors( cudaUnbindTexture(&_texProjections) );
	_volumeCost.CopyDevicetoHost(hostVolumeCost);
	
	//_volumeNIOk.CopyDevicetoHost(hostVolumeCost);
	//----------------------------------------------------------------------------
	//checkCudaErrors( cudaMemcpy( h_TabCost, dev_NbImgOk, costMemSize, cudaMemcpyDeviceToHost) );
	//checkCudaErrors( cudaMemcpy( host_Cache, dev_Cache,	  cac_MemSize, cudaMemcpyDeviceToHost) );
	
	//GpGpuTools::OutputArray(hostVolumeCost,_param.rDiTer,11.0f,_param.DefaultVal);
	//----------------------------------------------------------------------------
}

uint2 NS_ParamMICMAC::InterfaceMicMacGpGpu::GetDimensionTerrain()
{
	return _param.rDiTer;
}

bool NS_ParamMICMAC::InterfaceMicMacGpGpu::IsValid()
{
	return !(_param.ptMask0.x == - 1);
}

int2 NS_ParamMICMAC::InterfaceMicMacGpGpu::ptU1()
{
	return _param.pUTer1;
}

int2 NS_ParamMICMAC::InterfaceMicMacGpGpu::ptU0()
{
	return _param.pUTer0;
}

int2 NS_ParamMICMAC::InterfaceMicMacGpGpu::ptM0()
{
	return _param.ptMask0;
}

int2 NS_ParamMICMAC::InterfaceMicMacGpGpu::ptM1()
{
	return _param.ptMask1;
}

NS_ParamMICMAC::InterfaceMicMacGpGpu::~InterfaceMicMacGpGpu()
{
}

uint NS_ParamMICMAC::InterfaceMicMacGpGpu::GetSample()
{
	return _param.sampTer;
}

float NS_ParamMICMAC::InterfaceMicMacGpGpu::GetDefaultVal()
{
	return _param.DefaultVal;
}

uint2 NS_ParamMICMAC::InterfaceMicMacGpGpu::GetSDimensionTerrain()
{
	return _param.dimSTer;
}

int NS_ParamMICMAC::InterfaceMicMacGpGpu::GetIntDefaultVal()
{
	return _param.IntDefault;

}

