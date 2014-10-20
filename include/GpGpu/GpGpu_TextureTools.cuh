#ifndef GPGPU_TEXTURETOOLS_CUH
#define GPGPU_TEXTURETOOLS_CUH

#include "GpGpu_Defines.h"

// w0, w1, w2, and w3 are the four cubic B-spline basis functions
__host__ __device__
	float w0(float a)
{
	//    return (1.0f/6.0f)*(-a*a*a + 3.0f*a*a - 3.0f*a + 1.0f);
	return (1.0f/6.0f)*(a*(a*(-a + 3.0f) - 3.0f) + 1.0f);   // optimized
}

__host__ __device__
	float w1(float a)
{
	//    return (1.0f/6.0f)*(3.0f*a*a*a - 6.0f*a*a + 4.0f);
	return (1.0f/6.0f)*(a*a*(3.0f*a - 6.0f) + 4.0f);
}

__host__ __device__
	float w2(float a)
{
	//    return (1.0f/6.0f)*(-3.0f*a*a*a + 3.0f*a*a + 3.0f*a + 1.0f);
	return (1.0f/6.0f)*(a*(a*(-3.0f*a + 3.0f) + 3.0f) + 1.0f);
}

__host__ __device__
	float w3(float a)
{
	return (1.0f/6.0f)*(a*a*a);
}

// g0 and g1 are the two amplitude functions
__device__ float g0(float a)
{
	return w0(a) + w1(a);
} 

__device__ float g1(float a)
{
	return w2(a) + w3(a);
}

// h0 and h1 are the two offset functions
__device__ float h0(float a)
{
	// note +0.5 offset to compensate for CUDA linear filtering convention
	return -1.0f + w1(a) / (w0(a) + w1(a)) + 0.5f;
}

__device__ float h1(float a)
{
	return 1.0f + w3(a) / (w2(a) + w3(a)) + 0.5f;
}

// filter 4 values using cubic splines
template<class T>
__device__
T cubicFilter(float x, T c0, T c1, T c2, T c3)
{
    T r;
    r = c0 * w0(x);
    r += c1 * w1(x);
    r += c2 * w2(x);
    r += c3 * w3(x);
    return r;
}

template<class T>
inline __device__ T tex2DLayeredPt(texture<T, cudaTextureType2DLayered> t, uint2 pt, short sample, short layer)
{
    return tex2DLayered(t, (float)pt.x / sample + 0.5f, (float)pt.y / sample + 0.5f,layer) ;
}

template<class T>
inline __device__ T tex2DLayeredPt(texture<T, cudaTextureType2DLayered> t, float2 pt, short layer)
{
    return tex2DLayered(t, pt.x + 0.5f, pt.y + 0.5f,layer) ;
}

template<class T>
inline __device__ T tex2DLayeredPt(texture<T, cudaTextureType2DLayered> t, uint2 pt, short layer)
{
        return tex2DLayeredPt(t, make_float2(pt),layer) ;
}

/*
template<class T>
inline __device__ T tex2DLayeredPt(texture<T, cudaTextureType2DLayered, cudaReadModeElementType> t, uint2 pt, uint2 dim, short layer)
{
	return tex2DLayeredPt(t, make_float2(pt), dim, layer);
}
*/


// slow but precise bicubic lookup using 16 texture lookups
template<class T, class R>  // return type, texture type
__device__
R tex2DBicubic(const texture<T, 2, cudaReadModeNormalizedFloat> texref, float x, float y)
{
    x -= 0.5f;
    y -= 0.5f;
    float px = floor(x);
    float py = floor(y);
    float fx = x - px;
    float fy = y - py;

    return cubicFilter<R>(fy,
                          cubicFilter<R>(fx, tex2D(texref, px-1, py-1), tex2D(texref, px, py-1), tex2D(texref, px+1, py-1), tex2D(texref, px+2,py-1)),
                          cubicFilter<R>(fx, tex2D(texref, px-1, py),   tex2D(texref, px, py),   tex2D(texref, px+1, py),   tex2D(texref, px+2, py)),
                          cubicFilter<R>(fx, tex2D(texref, px-1, py+1), tex2D(texref, px, py+1), tex2D(texref, px+1, py+1), tex2D(texref, px+2, py+1)),
                          cubicFilter<R>(fx, tex2D(texref, px-1, py+2), tex2D(texref, px, py+2), tex2D(texref, px+1, py+2), tex2D(texref, px+2, py+2))
                          );
}

// slow but precise bicubic lookup using 16 texture lookups
template<class T, class R>  // return type, texture type
__device__
R tex2DBicubicLayered(const texture<T, cudaTextureType2DLayered> texref, float x, float y,int layer)
{

    x -= 0.5f;
    y -= 0.5f;


    float px = floor(x);
    float py = floor(y);
    float fx = x - px;
    float fy = y - py;


    return cubicFilter<R>(fy,
                          cubicFilter<R>(fx, tex2DLayered(texref, px-1, py-1,layer), tex2DLayered(texref, px, py-1,layer), tex2DLayered(texref, px+1, py-1,layer), tex2DLayered(texref, px+2,py-1,layer)),
                          cubicFilter<R>(fx, tex2DLayered(texref, px-1, py,layer),   tex2DLayered(texref, px, py,layer),   tex2DLayered(texref, px+1, py,layer),   tex2DLayered(texref, px+2, py,layer)),
                          cubicFilter<R>(fx, tex2DLayered(texref, px-1, py+1,layer), tex2DLayered(texref, px, py+1,layer), tex2DLayered(texref, px+1, py+1,layer), tex2DLayered(texref, px+2, py+1,layer)),
                          cubicFilter<R>(fx, tex2DLayered(texref, px-1, py+2,layer), tex2DLayered(texref, px, py+2,layer), tex2DLayered(texref, px+1, py+2,layer), tex2DLayered(texref, px+2, py+2,layer))
                          );
}

// fast bicubic texture lookup using 4 bilinear lookups
// assumes texture is set to non-normalized coordinates, point sampling
template<class T, class R>  // texture data type, return type
__device__ R tex2DFastBicubic(const texture<T, cudaTextureType2DLayered> texref, float x, float y, uint2 dim, int layer)
{
	x -= 0.5f;
	y -= 0.5f;
	float px = floor(x);
	float py = floor(y); 
	float fx = x - px;
	float fy = y - py;

	// note: we could store these functions in a lookup table texture, but maths is cheap
	float g0x = g0(fx);
	float g1x = g1(fx);
	float h0x = h0(fx);
	float h1x = h1(fx);
	float h0y = h0(fy);
	float h1y = h1(fy);

	R r = g0(fy) * (g0x * tex2DLayered(texref, (px + h0x + 0.5f)/dim.x, (py + h0y + 0.5f)/dim.y,layer)   +
		g1x * tex2DLayered(texref, (px + h1x + 0.5f)/dim.x, (py + h0y + 0.5f)/dim.y,layer)) +
		g1(fy) * (g0x * tex2DLayered(texref, (px + h0x+ 0.5f)/dim.x, (py + h1y+ 0.5f)/dim.y,layer)   +
		g1x * tex2DLayered(texref, (px + h1x+ 0.5f)/dim.x, (py + h1y+ 0.5f)/dim.y,layer));
	return r;
}

//template<class T, class R>  // texture data type, return type
//__device__ R tex2DFastBicubic(const texture<T, cudaTextureType2DLayered, cudaReadModeElementType> texref, float x, float y, int layer)
//{
//    x -= 0.5f;
//    y -= 0.5f;
//    float px = floor(x);
//    float py = floor(y);
//    float fx = x - px;
//    float fy = y - py;

//    // note: we could store these functions in a lookup table texture, but maths is cheap
//    float g0x = g0(fx);
//    float g1x = g1(fx);
//    float h0x = h0(fx);
//    float h1x = h1(fx);
//    float h0y = h0(fy);
//    float h1y = h1(fy);

//    R r = g0(fy) * (g0x * tex2DLayered(texref, (px + h0x + 0.5f), (py + h0y + 0.5f),layer)   +
//        g1x * tex2DLayered(texref, (px + h1x + 0.5f), (py + h0y + 0.5f),layer)) +
//        g1(fy) * (g0x * tex2DLayered(texref, (px + h0x+ 0.5f), (py + h1y+ 0.5f),layer)   +
//        g1x * tex2DLayered(texref, (px + h1x+ 0.5f), (py + h1y+ 0.5f),layer));
//    return r;
//}
// fast bicubic texture lookup using 4 bilinear lookups
template<class T, class R>  // return type, texture type
__device__
R tex2DFastBicubic(const texture<T,cudaTextureType2DLayered> texref, float x, float y, int layer)
{
    x -= 0.5f;
    y -= 0.5f;
    float px = floor(x);
    float py = floor(y);
    float fx = x - px;
    float fy = y - py;

    // note: we could store these functions in a lookup table texture, but maths is cheap
    float g0x = g0(fx);
    float g1x = g1(fx);
    float h0x = h0(fx);
    float h1x = h1(fx);
    float h0y = h0(fy);
    float h1y = h1(fy);

    R r = g0(fy) * ( g0x * tex2DLayered(texref, px + h0x, py + h0y,layer)   +
                     g1x * tex2DLayered(texref, px + h1x, py + h0y,layer) ) +
          g1(fy) * ( g0x * tex2DLayered(texref, px + h0x, py + h1y,layer)   +
                     g1x * tex2DLayered(texref, px + h1x, py + h1y,layer) );
    return r;
}

#endif //GPGPU_TEXTURETOOLS_CUH
