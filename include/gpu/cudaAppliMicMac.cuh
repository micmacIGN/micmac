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
static __constant__ uint2 cDimCach;
static __constant__ uint  cSizeCach;
static __constant__ uint  cBadVignet;


struct paramGPU
{
	 int2 pUTer0;
	 int2 pUTer1;
	 uint2 dimTer;		// Dimension du bloque terrain
	 uint2 dimSTer;		// Dimension du bloque terrain sous echantilloné
	 uint2 dimVig;		// Dimension de la vignette
	 uint2 dimImg;		// Dimension des images
	 uint2 rVig;		// Rayon de la vignette
	 uint  sizeVig;		// Taille de la vignette en pixel 
	 uint  sizeTer;		// Taille du bloque terrain
	 uint  sizeSTer;	// Taille du bloque terrain sous echantilloné
	 uint  sampTer;		// Pas echantillonage du terrain
	 float UVDefValue;	// UV Terrain incorrect
	 uint2 dimCach;		// Dimension cache
	 uint  sizeCach;	// Taille du cache
	 uint nLayer;
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

inline __host__ __device__ uint2 make_uint2(dim3 a)
{
	return make_uint2((uint3)a);
}

inline __host__ __device__ int2 make_int2(dim3 a)
{
	return make_int2(a.x,a.y);
}

inline __host__ __device__ int2 operator-(const uint3 a, uint2 b)
{
	return make_int2(a.x - b.x, a.y - b.y);
}

inline __host__ __device__ int2 operator-(const int2 a, uint2 b)
{
	return make_int2(a.x - b.x, a.y - b.y);
}

inline __host__ __device__ int2 operator+(const uint3 a, uint2 b)
{
	return make_int2(a.x + b.x, a.y + b.y);
}