#pragma once
#include <cuda_runtime.h>
#include <helper_math.h>
#include <helper_functions.h>
#include <helper_cuda.h>

struct paramGPU
{

	 int2	pUTer0;
	 int2	pUTer1;

	 uint2	rDiTer;		// Dimension du bloque terrain
	 uint2	dimTer;		// Dimension du bloque terrain + halo
	 uint2	dimSTer;	// Dimension du bloque terrain + halo sous echantilloné
	 uint2	dimVig;		// Dimension de la vignette
	 uint2	dimImg;		// Dimension des images
	 uint2	rVig;		// Rayon de la vignette
	 uint	sizeVig;	// Taille de la vignette en pixel
	 uint	sizeTer;	// Taille du bloque terrain + halo
	 uint	rSiTer;		// taille reel du terrain
	 uint	sizeSTer;	// Taille du bloque terrain + halo sous echantilloné
	 uint	sampTer;	// Pas echantillonage du terrain
	 float	UVDefValue;	// UV Terrain incorrect
	 uint2	dimCach;	// Dimension cache
	 uint	sizeCach;	// Taille du cache
	 uint	nLayer;		// Nombre d'images
	 int2	ptMask0;	// point debut du masque
	 int2	ptMask1;	// point fin du masque
	 float	badVig;		//
	 float	mAhEpsilon;
};

static __constant__ paramGPU cH;

static int iDivUp(int a, int b)
{
	return (a % b != 0) ? (a / b + 1) : (a / b);
}

static uint2 iDivUp(uint2 a, uint b)
{
	return make_uint2(iDivUp(a.x,b),iDivUp(a.y,b));
}

static int2 iDivUp(int2 a, uint b)
{
	return make_int2(iDivUp(a.x,b),iDivUp(a.y,b));
}

inline __device__ __host__ uint size(uint2 v)
{
	return v.x * v.y;
}

inline __device__ __host__ uint size(int2 v)
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

inline __host__ __device__ int2 operator/(int2 a, uint2 b)
{
	return make_int2(a.x / ((int)(b.x)), a.y / ((int)(b.y)));
}

inline __host__ __device__ uint2 operator*(int2 a, uint2 b)
{
	return make_uint2(a.x * b.x, a.y * b.y);
}

inline __host__ __device__ int2 operator+(int2 a, uint2 b)
{
	return make_int2(a.x + b.x, a.y + b.y);
}

inline __host__ __device__ uint2 make_uint2(dim3 a)
{
	return make_uint2((uint3)a);
}

inline __host__ __device__ int2 make_int2(dim3 a)
{
	return make_int2(a.x,a.y);
}

inline __host__ __device__ short2 make_short2(uint3 a)
{
	return make_short2((short)a.x,(short)a.y);
}

inline __host__ __device__ int2 operator-(const uint3 a, uint2 b)
{
	return make_int2(a.x - b.x, a.y - b.y);
}

inline __host__ __device__ int2 operator-(const int2 a, uint2 b)
{
	return make_int2(a.x - b.x, a.y - b.y);
}

inline __host__ __device__ short2 operator-(short2 a, uint2 b)
{
	return make_short2(a.x - b.x, a.y - b.y);
}


inline __host__ __device__ short2 operator+(short2 a, uint2 b)
{
	return make_short2(a.x + b.x, a.y + b.y);
}

inline __host__ __device__ int2 operator+(const uint3 a, uint2 b)
{
	return make_int2(a.x + b.x, a.y + b.y);
}

inline __host__ __device__ int mdlo(uint a, uint b)
{

	if (b == 0 ) return 0;
	return ((int)a) - (((int)a )/((int)b))*((int)b);

}

inline __host__ __device__ bool oSE(uint2 a, uint2 b)
{
	return ((a.x >= b.x) || (a.y >= b.y));
}

inline __host__ __device__ bool oSE(int2 a, uint2 b)
{
	return ((a.x >= (int)b.x) || (a.y >= (int)b.y));
}


inline __host__ __device__ bool oSE(uint3 a, uint2 b)
{
	return ((a.x >= b.x) || (a.y >= b.y));
}

inline __host__ __device__ bool oEq(float2 a, float b)
{
	return ((a.x == b) || (a.y == b));
}

inline __host__ __device__ bool oI(uint2 a, uint2 b)
{
	return ((a.x < b.x) || (a.y < b.y));
}

inline __host__ __device__ bool oI(int2 a, int b)
{
	return ((a.x < b) || (a.y < b));
}

inline __host__ __device__ bool oI(uint3 a, uint2 b)
{
	return ((a.x < b.x) || (a.y < b.y));
}

inline __host__ __device__ uint to1D( uint2 c2D, uint2 dim)
{
	return c2D.y * dim.x + c2D.x;
}

inline __host__ __device__ uint to1D( int2 c2D, uint2 dim)
{
	return c2D.y * dim.x + c2D.x;
}