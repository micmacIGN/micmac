#pragma once

#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_math.h>
#include <helper_cuda.h>

#ifdef __GNUC__
#define SUPPRESS_NOT_USED_WARN __attribute__ ((unused))
#else
#define SUPPRESS_NOT_USED_WARN
#endif

struct Rect
{
	int2 pt0;
	int2 pt1;

	__device__ __host__ Rect(){};
	Rect(int2 p0, int2 p1)
	{

		pt0 = p0;
		pt1 = p1;
	};


	Rect(uint2 p0, uint2 p1)
	{

		pt0 = make_int2(p0);
		pt1 = make_int2(p1);
	};

	__device__ __host__ Rect(int p0x,int p0y,int p1x,int p1y)
	{

		pt0 = make_int2(p0x,p0y);
		pt1 = make_int2(p1x,p1y);
	};

	uint2 dimension()
	{
		return make_uint2(pt1-pt0);
	};

#ifdef __cplusplus

	void out()
	{
		std::cout << "(" << pt0.x << "," <<  pt0.y << ")" << " -> (" << pt1.x << "," <<  pt1.y << ") ";
	}

#endif
};

static int iDivUp(int a, int b)
{
	return (a % b != 0) ? (a / b + 1) : (a / b);
}

SUPPRESS_NOT_USED_WARN static uint2 iDivUp(uint2 a, uint b)
{
	return make_uint2(iDivUp(a.x,b),iDivUp(a.y,b));
}

SUPPRESS_NOT_USED_WARN static uint2 iDivUp(uint2 a, uint2 b)
{
	return make_uint2(iDivUp(a.x,b.x),iDivUp(a.y,b.y));
}

SUPPRESS_NOT_USED_WARN static int2 iDivUp(int2 a, uint b)
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

inline __host__ __device__ int2 operator/(int2 a, uint b)
{
	return make_int2(a.x / b, a.y / b);
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

inline __host__ __device__ bool aSE(int2 a, int b)
{
	return ((a.x >= b) && (a.y >= b));
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

inline __host__ __device__ bool oI(float2 a, float b)
{
	return ((a.x < b) || (a.y < b));
}

inline __host__ __device__ bool oEq(uint2 a, int b)
{
	return ((a.x == (uint)b) || (a.y == (uint)b));
}


inline __host__ __device__ bool aEq(int2 a, int b)
{
	return ((a.x == b) && (a.y == b));
}

inline __host__ __device__ bool aEq(uint2 a, uint2 b)
{
	return ((a.x == b.x) && (a.y == b.y));
}

inline __host__ __device__ bool oI(uint2 a, uint2 b)
{
	return ((a.x < b.x) || (a.y < b.y));
}

inline __host__ __device__ bool oI(int2 a, int b)
{
	return ((a.x < b) || (a.y < b));
}

inline __host__ __device__ bool aI(int2 a, int2 b)
{
	return ((a.x < b.x) && (a.y < b.y));
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
