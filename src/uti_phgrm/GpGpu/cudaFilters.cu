#include "GpGpu/GpGpu_ParamCorrelation.cuh"

texture< pixel,	cudaTextureType2D >         TexS_MaskTerD;

__global__ void dilateKernel(pixel* dataOut, int r, uint2 dim, uint2 dimH)
{

	__shared__ pixel out[ BLOCKDIM ][ BLOCKDIM ];

	out[threadIdx.y][threadIdx.x] = 0;
	
	__syncthreads();

	const int2 ptH	= make_int2( blockIdx.x * (blockDim.x - 2*r) + threadIdx.x, blockIdx.y * (blockDim.y - 2*r) + threadIdx.y);
	const int2 pt = ptH - r ;

	if (threadIdx.x >= r  && threadIdx.y >=  r && threadIdx.x < blockDim.x - r && threadIdx.y < blockDim.y - r)
	{
		const pixel p = tex2D(TexS_MaskTerD, pt.x, pt.y);

		if (p == 1 && !(oI(pt,0) || oSE(pt,dim)))
		{
			#pragma unroll 
			for (int i = threadIdx.x - r; i < threadIdx.x ;i++)
				out[threadIdx.y][i] = 1;

			#pragma unroll 
			for (int i = threadIdx.x; i < threadIdx.x + r + 1;i++)
				out[threadIdx.y][i] = 1;
		}
	}

	__syncthreads();

	if (out[threadIdx.y][threadIdx.x] == 1 )
	{
		#pragma unroll 
		for (int j = threadIdx.y - r; j < threadIdx.y + r +1 ;j++)
			out[j][threadIdx.x] = 1;
	}

	__syncthreads();

	if (out[threadIdx.y][threadIdx.x] == 1 )
		dataOut[to1D(ptH,dimH)] = 1;
	
}

extern "C" void dilateKernel(pixel* HostDataOut, short r, uint2 dim)
{

	dim3	threads( BLOCKDIM, BLOCKDIM, 1);
	uint2	thd2D		= make_uint2(threads);
	uint2	actiThsCo	= thd2D - 2 * r;
	uint2	block2D		= iDivUp(dim,actiThsCo);
	dim3	blocks(block2D.x , block2D.y,1);

	CuDeviceData2D<pixel> deviceDataOut;

	uint2 dimDM = dim + 2*r;

	deviceDataOut.Realloc(dimDM);
	deviceDataOut.Memset(0);

	dilateKernel<<<blocks,threads>>>(deviceDataOut.pData(),r,dim,dimDM);
	getLastCudaError("DilateX kernel failed");

    deviceDataOut.DecoratorDeviceData::CopyDevicetoHost(HostDataOut);
	deviceDataOut.Dealloc();

}
