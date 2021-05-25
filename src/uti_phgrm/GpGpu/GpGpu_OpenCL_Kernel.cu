#include "GpGpu_CUDA_Define.cu" // DON'T MOVE THIS LINE!!!

#define FACTOR 4

__GPU_CONSTANT  int hw[] = {1,2,5,6,9};


__GPU_KERNEL void kMultTab(__GPU_GLOBAL int * out,  int t)
{
    size_t tid = __GPU_THREADX;

	//int2 dd = make_int2(50,20);

	//out[tid] = FACTOR*t*hw[tid] + dd.x;

	out[tid] = sgpu::__div<32>(tid*64);
}


#ifdef CUDA_ENABLED
extern void kMultTab()
{
    dim3	threads( 5, 1, 1);
    dim3	blocks(1, 1, 1);

    int* buffer     = ((CData<int>*)CGpGpuContext<cudaContext>::arg(0))->pData();
    int* pFactor    = ((int*)CGpGpuContext<cudaContext>::arg(1));

    kMultTab<<<blocks, threads>>>(buffer,*pFactor);
    getLastCudaError("kMultTab kernel failed");
}
#endif
