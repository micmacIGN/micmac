#pragma once

#ifdef CUDA_ENABLED
		
		#include "StdAfx.h"
		#include "GpGpu/cudaAppliMicMac.cuh"

	namespace NS_ParamMICMAC
	{

		extern "C" void				freeGpuMemory();
		extern "C" void				basic_Correlation_GPU(  float* h_TabCorre, float2* hostVolumeProj, int nbLayer, uint interZ);
		extern "C" void				imagesToLayers(float *fdataImg1D, uint2 dimTer, int nbLayer);
		extern "C" paramMicMacGpGpu Init_Correlation_GPU( uint2 ter0, uint2 ter1, int nbLayer , uint2 dRVig , uint2 dimImg, float mAhEpsilon, uint samplingZ, int uvINTDef, uint interZ);
		extern "C" paramMicMacGpGpu updateSizeBlock( uint2 ter0, uint2 ter1, uint interZ );
		extern "C" void				SetMask(pixel* dataMask, uint2 dimMask);

		extern "C" void				CopyParamTodevice(paramMicMacGpGpu h);
		extern "C" void				KernelCorrelation(dim3 blocks, dim3 threads, float *dev_NbImgOk, float* cachVig, uint2 nbActThrd);
		extern "C" void				KernelmultiCorrelation(dim3 blocks, dim3 threads, float *dTCost, float* cacheVign, float * dev_NbImgOk, uint2 nbActThr);
		
		uint2 toUi2(Pt2di a){return make_uint2(a.x,a.y);};
		int2  toI2(Pt2dr a){return make_int2((int)a.x,(int)a.y);};
		paramMicMacGpGpu h;

		class InterfaceMicMacGpGpu
		{

			public:

				InterfaceMicMacGpGpu(textureReference* textureMask,paramMicMacGpGpu* h);
				~InterfaceMicMacGpGpu(){};

				void	SetSizeBlock( uint2 ter0, uint2 ter1, uint Zinter);
				void	AllocMemory();
				void	DeallocMemory();
				void	SetMask(pixel* dataMask, uint2 dimMask);
				void	SetImages(float* dataImage, uint2 dimImage, int nbLayer);
				void	InitParam(uint2 ter0, uint2 ter1, int nbLayer , uint2 dRVig , uint2 dimImg, float mAhEpsilon, uint samplingZ, int uvINTDef , uint interZ);
				void	BasicCorrelation( float* hostVolumeCost, float2* hostVolumeProj,  int nbLayer, uint interZ );

			private:

				paramMicMacGpGpu		_param;

				CuDeviceData3D<float>	_volumeCost;	// volume des couts   
				CuDeviceData3D<float>	_volumeCach;	// volume des calculs intermédiaires
				CuDeviceData3D<float>	_volumeNIOk;	// nombre d'image correct pour une vignette

				ImageCuda<pixel>		_mask;
				ImageLayeredCuda<float>	_LayeredImages;
				ImageLayeredCuda<float2>_LayeredProjection;

				textureReference*		_texMask;
				textureReference*		_texImages;
				textureReference*		_texProjections;

				//static __constant__ paramMicMacGpGpu* _cH;

		};
	}

#endif