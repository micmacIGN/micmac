#pragma once

#ifdef CUDA_ENABLED
		
		#include "StdAfx.h"
		#include "GpGpu/cudaAppliMicMac.cuh"

	namespace NS_ParamMICMAC
	{
		extern "C" void	CopyParamTodevice(paramMicMacGpGpu h);
		extern "C" void	KernelCorrelation(dim3 blocks, dim3 threads, float *dev_NbImgOk, float* cachVig, uint2 nbActThrd);
		extern "C" void	KernelmultiCorrelation(dim3 blocks, dim3 threads, float *dTCost, float* cacheVign, float * dev_NbImgOk, uint2 nbActThr);

		extern "C" textureReference&	getMask();
		extern "C" textureReference&	getImage();
		extern "C" textureReference&	getProjection();


		class InterfaceMicMacGpGpu
		{

			public:

				InterfaceMicMacGpGpu();
				~InterfaceMicMacGpGpu();

				void	SetSizeBlock( uint2 ter0, uint2 ter1, uint Zinter);
				void	SetSizeBlock( uint Zinter);
				void	AllocMemory();
				void	DeallocMemory();
				void	SetMask(pixel* dataMask, uint2 dimMask);
				void	SetImages(float* dataImage, uint2 dimImage, int nbLayer);
				void	InitParam(uint2 ter0, uint2 ter1, int nbLayer , uint2 dRVig , uint2 dimImg, float mAhEpsilon, uint samplingZ, int uvINTDef , uint interZ);
				void	BasicCorrelation( float* hostVolumeCost, float2* hostVolumeProj,  int nbLayer, uint interZ );
				uint2	GetDimensionTerrain();
				uint2	GetSDimensionTerrain();
				bool	IsValid();
				int2	ptU0();
				int2	ptU1();
				int2	ptM0();
				int2	ptM1();
				uint	GetSample();
				float	GetDefaultVal();
				int		GetIntDefaultVal();

			private:

				paramMicMacGpGpu		_param;

				CuDeviceData3D<float>	_volumeCost;	// volume des couts   
				CuDeviceData3D<float>	_volumeCach;	// volume des calculs intermédiaires
				CuDeviceData3D<float>	_volumeNIOk;	// nombre d'image correct pour une vignette

				ImageCuda<pixel>		_mask;
				ImageLayeredCuda<float>	_LayeredImages;
				ImageLayeredCuda<float2>_LayeredProjection;

				textureReference&		_texMask;
 				textureReference&		_texImages;
 				textureReference&		_texProjections;


		};
	}

#endif