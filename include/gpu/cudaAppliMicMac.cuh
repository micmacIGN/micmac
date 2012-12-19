#pragma once
#include <cuda_runtime.h>
#include <helper_math.h>
#include <helper_functions.h>
#include <helper_cuda.h>

static __constant__ uint2 cDimTer;
static __constant__ uint2 cSDimTer;
static __constant__ uint2 cDimVig;
static __constant__ uint2 cDimImg;
static __constant__ uint2 cRVig;
static __constant__ float cMAhEpsilon;
static __constant__ uint  cSizeVig;
static __constant__ uint  cSizeTer;
static __constant__ uint  cSampTer;
static __constant__ uint  cSizeSTer;
static __constant__ float cUVDefValue;


struct paramGPU
{
	 uint2 DimTer;
	 uint2 SDimTer;
	 uint2 DimVig;
	 uint2 DimImg;
	 uint2 RVig;
	 uint  SizeVig;
	 uint  SizeTer;
	 uint  SizeSTer;
	 uint  SampTer;
	 float UVDefValue;
};



static int iDivUp(int a, int b)
{
	return (a % b != 0) ? (a / b + 1) : (a / b);
}

static uint2 iDivUp(uint2 a, uint b)
{
	return make_uint2(iDivUp(a.x,b),iDivUp(a.y,b));
}

inline __device__ __host__ uint size(uint2 v)
{
	return v.x * v.y;
}

inline __host__ __device__ uint2 operator/(uint2 a, uint2 b)
{
	return make_uint2(a.x / b.x, a.y / b.y);
}

inline __host__ __device__ uint2 operator/(uint2 a, int b)
{
	return make_uint2(a.x / b, a.y / b);
}
