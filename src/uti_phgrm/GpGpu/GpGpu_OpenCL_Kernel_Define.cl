#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

#define __GPU_CONSTANT  __constant
#define __GPU_GLOBAL    __global
#define __GPU_KERNEL    __kernel
#define __GPU_THREADX   get_global_id(0)

#define make_int2(x,y) (int2)(x,y)


#define INCLUDE_SDK
