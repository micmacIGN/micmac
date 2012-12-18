#pragma once

#include <cuda_runtime.h>
#include <helper_math.h>
#include <helper_functions.h>
#include <helper_cuda.h>

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
