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
	 uint2 DimTer;		// Dimension du bloque terrain
	 uint2 SDimTer;		// Dimension du bloque terrain sous echantilloné
	 uint2 DimVig;		// Dimension de la vignette
	 uint2 DimImg;		// Dimension des images
	 uint2 RVig;		// Rayon de la vignette
	 uint  SizeVig;		// Taille de la vignette en pixel 
	 uint  SizeTer;		// Taille du bloque terrain
	 uint  SizeSTer;	// Taille du bloque terrain sous echantilloné
	 uint  SampTer;		// Pas echantillonage du terrain
	 float UVDefValue;	// UV Terrain incorrect
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
