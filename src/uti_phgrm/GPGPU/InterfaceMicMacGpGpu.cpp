#include "GpGpu/InterfaceMicMacGpGpu.h"


InterfaceMicMacGpGpu::InterfaceMicMacGpGpu():
_texMask(getMask()),
_texImages(getImage()),
_texProjections_01(getProjection(1)),
_texProjections_02(getProjection(2))
{
	for (int s = 0;s<NSTREAM;s++)
		checkCudaErrors( cudaStreamCreate(GetStream(s)));
}

InterfaceMicMacGpGpu::~InterfaceMicMacGpGpu()
{
	for (int s = 0;s<NSTREAM;s++)
		checkCudaErrors( cudaStreamDestroy(*(GetStream(s))));
}

void InterfaceMicMacGpGpu::SetSizeBlock( uint2 ter0, uint2 ter1, uint Zinter )
{

	uint oldSizeTer = _param.sizeTer;

	_param.SetDimension(ter0,ter1,Zinter);

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
	SetSizeBlock( make_uint2(_param.ptMask0), make_uint2(_param.ptMask1), Zinter );
}

void InterfaceMicMacGpGpu::AllocMemory(int nStream)
{
	_volumeCost[nStream].Realloc(_param.rDiTer,_param.ZLocInter);
	_volumeCach[nStream].Realloc(_param.dimCach, _param.nbImages * _param.ZLocInter);
	_volumeNIOk[nStream].Realloc(_param.rDiTer,_param.ZLocInter);
}

void InterfaceMicMacGpGpu::DeallocMemory()
{
	checkCudaErrors( cudaUnbindTexture(&_texImages) );	
	checkCudaErrors( cudaUnbindTexture(&_texMask) );	

	for (int s = 0;s<NSTREAM;s++)
	{
		_volumeCach[s].Dealloc();
		_volumeCost[s].Dealloc();
		_volumeNIOk[s].Dealloc();
		_LayeredProjection[s].Dealloc();
	}

	_mask.Dealloc();
	_LayeredImages.Dealloc();
	
}

void InterfaceMicMacGpGpu::SetMask( pixel* dataMask, uint2 dimMask )
{
	_mask.InitImage(dimMask,dataMask);
	_mask.bindTexture(_texMask);
}

void InterfaceMicMacGpGpu::InitParam( uint2 ter0, uint2 ter1, int nbLayer , uint2 dRVig , uint2 dimImg, float mAhEpsilon, uint samplingZ, int uvINTDef , uint interZ )
{
	
	// Parametres texture des projections
	for (int s = 0;s<NSTREAM;s++)
	{
		GetTeXProjection(s).addressMode[0]	= cudaAddressModeClamp;
		GetTeXProjection(s).addressMode[1]	= cudaAddressModeClamp;	
		GetTeXProjection(s).filterMode		= cudaFilterModeLinear; //cudaFilterModePoint cudaFilterModeLinear
		GetTeXProjection(s).normalized		= true;
	}
	// Parametres texture des Images
	_texImages.addressMode[0]	= cudaAddressModeWrap;
	_texImages.addressMode[1]	= cudaAddressModeWrap;
	_texImages.filterMode		= cudaFilterModePoint; //cudaFilterModeLinear cudaFilterModePoint
	_texImages.normalized		= true;

	_param.SetParamInva( dRVig * 2 + 1,dRVig, dimImg, mAhEpsilon, samplingZ, uvINTDef, nbLayer);
	SetSizeBlock( ter0, ter1, interZ );

	for (int s = 0;s<NSTREAM;s++)
		AllocMemory(s);
}

void InterfaceMicMacGpGpu::SetImages( float* dataImage, uint2 dimImage, int nbLayer )
{
	_LayeredImages.CData3D::Malloc(dimImage,nbLayer);
	_LayeredImages.copyHostToDevice(dataImage);
	_LayeredImages.bindTexture(_texImages);
}

void InterfaceMicMacGpGpu::BasicCorrelation( float* hostVolumeCost, float2* hostVolumeProj, int nbLayer, uint interZ )
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

	//std::cout <<  interZ << "\n";

	while(Z < interZ)
	{
		//const uint nstream = ((NSTREAM * _param.ZLocInter) >= (interZ - Z)  &&  Z != (interZ - 1)) ? (interZ - Z) : NSTREAM;

		const uint nstream = NSTREAM;

		for (uint s = 0;s<nstream;s++)
		{
			//_LayeredProjection[s].copyHostToDevice(hostVolumeProj);
			_LayeredProjection[s].copyHostToDeviceASync(hostVolumeProj + (Z+s)*_LayeredProjection->GetSize() * _LayeredProjection->GetNbLayer(),*(GetStream(s)));
			_LayeredProjection[s].bindTexture(GetTeXProjection(s));
		}

		for (uint s = 0;s<nstream;s++)
		{
			KernelCorrelation( *(GetStream(s)),blocks, threads,  _volumeNIOk[s].pData(), _volumeCach[s].pData(), actiThsCo);
			KernelmultiCorrelation( *(GetStream(s)),blocks_mC, threads_mC,  _volumeCost[s].pData(), _volumeCach[s].pData(), _volumeNIOk[s].pData(), actiThs);
		}
	
		for (uint s = 0;s<nstream;s++)
		{
			checkCudaErrors( cudaUnbindTexture(&(GetTeXProjection(s))) );
			_volumeCost[s].CopyDevicetoHostASync(hostVolumeCost + (Z+s)*_volumeCost->GetSize()*_volumeCost->GetNbLayer(),*(GetStream(s)));
		}

		for (uint s = 0;s<NSTREAM;s++)
			checkCudaErrors( cudaStreamSynchronize(*(GetStream(s))));
		
		Z += nstream * _param.ZLocInter;
	}


	for (uint s = 0;s<NSTREAM;s++)
		checkCudaErrors( cudaStreamSynchronize(*(GetStream(s))));

	//_volumeNIOk.CopyDevicetoHost(hostVolumeCost);
	//checkCudaErrors( cudaMemcpy( h_TabCost, dev_NbImgOk, costMemSize, cudaMemcpyDeviceToHost) );	
	//GpGpuTools::OutputArray(hostVolumeCost,_param.rDiTer,11.0f,_param.DefaultVal);
}

uint2 InterfaceMicMacGpGpu::GetDimensionTerrain()
{
	return _param.rDiTer;
}

bool InterfaceMicMacGpGpu::IsValid()
{
	return !(_param.ptMask0.x == - 1);
}

int2 InterfaceMicMacGpGpu::ptU1()
{
	return _param.pUTer1;
}

int2 InterfaceMicMacGpGpu::ptU0()
{
	return _param.pUTer0;
}

int2 InterfaceMicMacGpGpu::ptM0()
{
	return _param.ptMask0;
}

int2 InterfaceMicMacGpGpu::ptM1()
{
	return _param.ptMask1;
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
	case 1:
		return _texProjections_01;
	case 2:
		return _texProjections_02;
	default:
		return _texProjections_01;
	}	
}

cudaStream_t* InterfaceMicMacGpGpu::GetStream( int stream )
{
	return &(_stream[stream]);
}

