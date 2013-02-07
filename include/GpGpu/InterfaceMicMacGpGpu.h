#pragma once

#ifdef CUDA_ENABLED
		
		#include "StdAfx.h"
		#include "GpGpu/cudaAppliMicMac.cuh"
		#include "GpGpu/helper_math_extented.cuh"
		#include "GpGpu/GpGpuTools.h"
	#ifdef _WIN32
		#include <Lmcons.h>
	#endif

	namespace NS_ParamMICMAC
	{

		extern "C" void		freeGpuMemory();
		extern "C" void		CopyProjToLayers(float2 *h_TabProj);
		extern "C" void		basic_Correlation_GPU(  float* h_TabCorre, int nbLayer, uint interZ);
		extern "C" void		imagesToLayers(float *fdataImg1D, uint2 dimTer, int nbLayer);
		extern "C" paramGPU Init_Correlation_GPU( uint2 ter0, uint2 ter1, int nbLayer , uint2 dRVig , uint2 dimImg, float mAhEpsilon, uint samplingZ, int uvINTDef, uint interZ);
		extern "C" paramGPU updateSizeBlock( uint2 ter0, uint2 ter1, uint interZ );
		extern "C" void		allocMemoryTabProj(uint2 dimTer, int nbLayer);
		extern "C" void		SetMask(pixel* dataMask, uint2 dimMask);

		uint2 toUi2(Pt2di a){return make_uint2(a.x,a.y);};
		int2  toI2(Pt2dr a){return make_int2((int)a.x,(int)a.y);};
		paramGPU h;
	}

#endif