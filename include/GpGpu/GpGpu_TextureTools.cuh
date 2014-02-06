#ifndef GPGPU_TEXTURETOOLS_CUH
#define GPGPU_TEXTURETOOLS_CUH

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

template<class T>
inline __device__ T tex2DLayeredPt(texture<T, cudaTextureType2DLayered, cudaReadModeElementType> t, uint2 pt, short sample, short layer)
{
	return tex2DLayered(t, (float)pt.x / sample + 0.5f, (float)pt.y / sample + 0.5f,layer) ;
}

template<class T>
inline __device__ T tex2DLayeredPt(texture<T, cudaTextureType2DLayered, cudaReadModeElementType> t, float2 pt, short layer)
{
	return tex2DLayered(t, pt.x + 0.5f, pt.y + 0.5f,layer) ;
}

template<class T>
inline __device__ T tex2DLayeredPt(texture<T, cudaTextureType2DLayered, cudaReadModeElementType> t, uint2 pt, short layer)
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

// fast bicubic texture lookup using 4 bilinear lookups
// assumes texture is set to non-normalized coordinates, point sampling
template<class T, class R>  // texture data type, return type
__device__ R tex2DFastBicubic(const texture<T, cudaTextureType2DLayered, cudaReadModeElementType> texref, float x, float y, uint2 dim, int layer)
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
};

#endif //GPGPU_TEXTURETOOLS_CUH
