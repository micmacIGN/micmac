#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

__constant int hw[] = {1,2,5,6,8};

__kernel void kMultTab(__global int * out,  int t)
{
    size_t tid = get_global_id(0);
    out[tid] = t*hw[tid]+20;
}
