#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

__constant int hw[] = {1,2,5,6,8};

__kernel void hello(__global int * out)
{
    size_t tid = get_global_id(0);
    out[tid] = 1.5f*hw[tid];
}
