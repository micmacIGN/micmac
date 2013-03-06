#pragma once

#ifdef CUDA_ENABLED
#define BOOST_ALL_NO_LIB 

#include "GpGpu/cudaAppliMicMac.cuh"
#include <boost/thread/thread.hpp>
#include "GpGpu/GpGpuTools.h"
//#include <boost\chrono\chrono.hpp>

extern "C" void	CopyParamTodevice(paramMicMacGpGpu h);
extern "C" void	KernelCorrelation(const int s,cudaStream_t stream, dim3 blocks, dim3 threads, uint *dev_NbImgOk, float* cachVig, uint2 nbActThrd);
extern "C" void	KernelmultiCorrelation(cudaStream_t stream, dim3 blocks, dim3 threads, float *dTCost, float* cacheVign, uint* dev_NbImgOk, uint2 nbActThr, ushort divideNThreads);

extern "C" void dilateKernel(pixel* HostDataOut, short r, uint2 dim);
extern "C" textureReference& getMaskD();

extern "C" textureReference&	getMask();
extern "C" textureReference&	getImage();
extern "C" textureReference&	getProjection(int TexSel);

#define NSTREAM 1

class InterfaceMicMacGpGpu
{

	public:

		InterfaceMicMacGpGpu();
		~InterfaceMicMacGpGpu();

		void	SetSizeBlock( Rect Ter, uint Zinter);
		void	SetSizeBlock( uint Zinter);
		void	SetMask(pixel* dataMask, uint2 dimMask);
		void	SetImages(float* dataImage, uint2 dimImage, int nbLayer);
		void	InitParam(Rect Ter, int nbLayer , uint2 dRVig , uint2 dimImg, float mAhEpsilon, uint samplingZ, int uvINTDef , uint interZ);
		void	BasicCorrelation( float* hostVolumeCost, float2* hostVolumeProj,  int nbLayer, uint interZ );
		void	BasicCorrelationStream( float* hostVolumeCost, float2* hostVolumeProj,  int nbLayer, uint interZ );
		uint2	GetDimensionTerrain();
		uint2	GetSDimensionTerrain();
		bool	MaskNoNULL();
		Rect	rUTer();
		Rect	rMask();
		uint	GetSample();
		float	GetDefaultVal();
		int		GetIntDefaultVal();
		void	DeallocMemory();
		void	SetHostVolume(float* vCost, float2* vProj);
		uint	GetZToCompute();
		void	SetZToCompute(uint Z);
		uint	GetZCtoCopy();
		void	SetZCToCopy(uint Z);
		bool	GetComputeNextProj();
		void	SetComputeNextProj(bool compute);
		int		GetComputedZ();
		void	SetComputedZ(int computedZ);
		void	MallocInfo();

#ifdef USEDILATEMASK
		void	dilateMask(uint2 dim);
		pixel*	GetDilateMask();
		pixel	ValDilMask(int2 pt);
#endif

	private:

		void					ResizeVolume(int nbLayer, uint interZ);
		void					AllocMemory(int nStream);
		cudaStream_t*			GetStream(int stream);
		textureReference&		GetTeXProjection(int texSel);
		void					MTComputeCost();

		cudaStream_t			_stream[NSTREAM];
		paramMicMacGpGpu		_param;

		CuDeviceData3D<float>	_volumeCost[NSTREAM];	// volume des couts   
		CuDeviceData3D<float>	_volumeCach[NSTREAM];	// volume des calculs intermédiaires
		CuDeviceData3D<uint>	_volumeNIOk[NSTREAM];	// nombre d'image correct pour une vignette

		ImageCuda<pixel>		_mask;
		ImageLayeredCuda<float>	_LayeredImages;
		ImageLayeredCuda<float2>_LayeredProjection[NSTREAM];

#ifdef USEDILATEMASK
		pixel*					_dilateMask;
		uint2					_dimDilateMask;
#endif
		textureReference&		_texMask;
		textureReference&		_texMaskD;
 		textureReference&		_texImages;
 		textureReference&		_texProjections_00;
		textureReference&		_texProjections_01;
		textureReference&		_texProjections_02;
		textureReference&		_texProjections_03;
		boost::thread*			_gpuThread;
		boost::mutex			_mutex;
		boost::mutex			_mutexC;
		boost::mutex			_mutexCompute;
		float*					_vCost;
		float2*					_vProj;	
		uint					_ZCompute;
		uint					_ZCCopy;
		bool					_computeNextProj;
		int						_computedZ;
};

#endif
