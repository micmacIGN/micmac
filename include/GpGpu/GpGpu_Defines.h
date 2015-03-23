#ifndef GPGPU_DEFINES_H
#define GPGPU_DEFINES_H


#include "GpGpu_BuildOptions.h"

typedef unsigned char pixel;
#include <string>

#define NOPAGLOCKMEM false
#define NOALIGNM128	 false
#define WARPSIZE    32
#define SIZERING    2
#define INTDEFAULT	-64
#define SAMPLETERR	4
#define INTERZ		8
#define NEAREST		0
#define LINEARINTER	1
#define BICUBIC		2
#define NSTREAM		1
#define DISPLAYOUTPUT

#define OPTIMZ      1
#define OPTIMX      0

#ifdef _DEBUG
    #define   BLOCKDIM	16
    #define   SBLOCKDIM 10
#else
    #define   BLOCKDIM	16  //#define   BLOCKDIM	32 moins rapide !!!!
    #define   SBLOCKDIM 15
#endif

#define NAPPEMAX WARPSIZE * 8

#define eAVANT      true
#define eARRIERE    false

#define TexFloat2Layered texture<float2,cudaTextureType2DLayered>
#define TEMPLATE_D2OPTI template<template<class T> class U, uint NBUFFER >

#define HOST_Data2Opti Data2Optimiz<CuHostData3D,2>
#define DEVC_Data2Opti Data2Optimiz<CuDeviceData3D>



template<class T>
__device__ __host__ inline void dump_Type(T var)
{

    printf("Warning dump var not define for this type\n");

}

template<>
__device__ __host__ inline void dump_Type<uint2>(uint2 var)
{
    printf("[%u,%u]",var.x,var.y);
}

template<>
__device__ __host__ inline void dump_Type<uint3>(uint3 var)
{
    printf("[%u,%u,%u]",var.x,var.y,var.z);
}


template<>
__device__ __host__ inline void dump_Type<dim3>(dim3 var)
{
    printf("[%u,%u,%u]",var.x,var.y,var.z);
}

template<>
__device__ __host__ inline void dump_Type<Rect>(Rect var)
{
    var.out();
}


template<>
__device__ __host__ inline void dump_Type<int2>(int2 var)
{
    printf("[%d,%d]",var.x,var.y);
}

template<>
__device__ __host__ inline void dump_Type<uint>(uint var)
{
    printf("%u",var);
}

template<>
__device__ __host__ inline void dump_Type<ushort>(ushort var)
{
    printf("%u",var);
}

template<>
__device__ __host__ inline void dump_Type<int>(int var)
{
    printf("%d",var);
}

template<>
__device__ __host__ inline void dump_Type<float2>(float2 var)
{
    printf("[%f,%f]",var.x,var.y);
}

template<>
__device__ __host__ inline void dump_Type<float>(float var)
{
    printf("%f",var);
}

template<>
__device__ __host__ inline void dump_Type<double>(double var)
{
    printf("%f",var);
}

template<>
__device__ __host__ inline void dump_Type<bool>(bool var)
{
    printf("%s",var ? "true" : "false");
}

template<>
__device__ __host__ inline void dump_Type<const char*>(const char* var)
{
    printf("%s",var);
}

template<>
__device__ __host__ inline void dump_Type<const std::string &>(const std::string &var)
{
    printf("%s",var.c_str());
}


template<class T> __device__ __host__ inline
void dump_variable(T var,const char* nameVariable)
{
    printf("%s\t= \t",nameVariable);
    dump_Type(var);
     printf("\n");
}

template<> __device__ __host__ inline
void dump_variable(const char* var,const char* nameVariable)
{
    dump_Type(var);
    printf("\n");
}

#define DUMP(varname)   dump_variable(varname,#varname);
#define DUMPI(varname)  dump_Type(varname);


//#define CUDA_DUMP_INT(varname) if(!threadIdx.x) printf("%s = %d\n", #varname, varname);
//#define CUDA_DUMP_INT_ALL(varname) printf("%s = %d\n", #varname, varname);

//#define DUMP_UINT(varname) printf("%s = %u\n", #varname, varname);
//#define DUMP_UINT2(varname) printf("%s = [%u,%u]\n", #varname, varname.x,varname.y);
//#define DUMP_INT2(varname) printf("%s = [%d,%d]\n", #varname, varname.x,varname.y);
//#define DUMP_INT(varname) printf("%s = %d\n", #varname, varname);
//#define DUMP_FLOAT2(varname) printf("%s = [%f,%f]\n", #varname, varname.x,varname.y);
//#define DUMP_FLOAT(varname) printf("%s = %f\n", #varname, varname);
//#define DUMP_POINTER(varname) printf("%s = %p\n", #varname, varname);
#define DUMP_LINE printf("-----------------------------------\n");
#define DUMP_END printf("\n");

/*
#define X(a) DUMP(a)

#define Paste(a,b) a ## b
#define XPASTE(a,b) Paste(a,b)

#define PP_NARG(...)    PP_NARG_(__VA_ARGS__,PP_RSEQ_N())
#define PP_NARG_(...)   PP_ARG_N(__VA_ARGS__)

#define PP_ARG_N( \
        _1, _2, _3, _4, _5, _6, _7, _8, _9,_10,  \
        _11,_12,_13,_14,_15,_16,_17,_18,_19,_20, \
        _21,_22,_23,_24,_25,_26,_27,_28,_29,_30, \
        _31,_32,_33,_34,_35,_36,_37,_38,_39,_40, \
        _41,_42,_43,_44,_45,_46,_47,_48,_49,_50, \
        _51,_52,_53,_54,_55,_56,_57,_58,_59,_60, \
        _61,_62,_63,N,...) N

#define PP_RSEQ_N() \
        63,62,61,60,                   \
        59,58,57,56,55,54,53,52,51,50, \
        49,48,47,46,45,44,43,42,41,40, \
        39,38,37,36,35,34,33,32,31,30, \
        29,28,27,26,25,24,23,22,21,20, \
        19,18,17,16,15,14,13,12,11,10, \
        9,8,7,6,5,4,3,2,1,0
*/



/* APPLYXn variadic X-Macro by M Joshua Ryan      */
/* Free for all uses. Don't be a jerk.            */
/* I got bored after typing 15 of these.          */
/* You could keep going upto 64 (PPNARG's limit). */
/*
#define APPLYX1(a)           X(a)
#define APPLYX2(a,b)         X(a) X(b)
#define APPLYX3(a,b,c)       X(a) X(b) X(c)
#define APPLYX4(a,b,c,d)     X(a) X(b) X(c) X(d)
#define APPLYX5(a,b,c,d,e)   X(a) X(b) X(c) X(d) X(e)
#define APPLYX6(a,b,c,d,e,f) X(a) X(b) X(c) X(d) X(e) X(f)
#define APPLYX7(a,b,c,d,e,f,g) \
    X(a) X(b) X(c) X(d) X(e) X(f) X(g)
#define APPLYX8(a,b,c,d,e,f,g,h) \
    X(a) X(b) X(c) X(d) X(e) X(f) X(g) X(h)
#define APPLYX9(a,b,c,d,e,f,g,h,i) \
    X(a) X(b) X(c) X(d) X(e) X(f) X(g) X(h) X(i)
#define APPLYX10(a,b,c,d,e,f,g,h,i,j) \
    X(a) X(b) X(c) X(d) X(e) X(f) X(g) X(h) X(i) X(j)
#define APPLYX11(a,b,c,d,e,f,g,h,i,j,k) \
    X(a) X(b) X(c) X(d) X(e) X(f) X(g) X(h) X(i) X(j) X(k)
#define APPLYX12(a,b,c,d,e,f,g,h,i,j,k,l) \
    X(a) X(b) X(c) X(d) X(e) X(f) X(g) X(h) X(i) X(j) X(k) X(l)
#define APPLYX13(a,b,c,d,e,f,g,h,i,j,k,l,m) \
    X(a) X(b) X(c) X(d) X(e) X(f) X(g) X(h) X(i) X(j) X(k) X(l) X(m)
#define APPLYX14(a,b,c,d,e,f,g,h,i,j,k,l,m,n) \
    X(a) X(b) X(c) X(d) X(e) X(f) X(g) X(h) X(i) X(j) X(k) X(l) X(m) X(n)
#define APPLYX15(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o) \
    X(a) X(b) X(c) X(d) X(e) X(f) X(g) X(h) X(i) X(j) X(k) X(l) X(m) X(n) X(o)

#define APPLYX_(M, ...) M(__VA_ARGS__)

#define APPLYXn(...) APPLYX_(XPASTE(APPLYX, PP_NARG(__VA_ARGS__)), __VA_ARGS__)

#define DUMPS(...) APPLYX_(XPASTE(APPLYX, PP_NARG(__VA_ARGS__)), __VA_ARGS__)
*/
//class sDebug
//{
//public:
//    sDebug operator<<(const int& obj)
//    {
//      DUMP(obj)
//      return *this;
//    }

//};

//static sDebug _debug;

//static sDebug gDebug()
//{
//    return _debug;
//}


#if USE_OPEN_MP
    #if ELISE_windows
        #define OMP_NT0 __pragma("omp parallel for num_threads(8)")
        #define OMP_NT1 __pragma("omp parallel for num_threads(4)")
        #define OMP_NT2 __pragma("omp parallel for num_threads(3)")
    #else
        #define OMP_NT0 _Pragma("omp parallel for num_threads(4)")
        #define OMP_NT1 _Pragma("omp parallel for num_threads(4)")
        #define OMP_NT2 _Pragma("omp parallel for num_threads(3)")
    #endif
#else
    #define OMP_NT0
    #define OMP_NT1
    #define OMP_NT2
#endif

enum Plans {XY,XZ,YZ,YX,ZX,ZY};


// UTILISER le define en dessous
inline std::string className(const std::string& prettyFunction)
{
    size_t colons = prettyFunction.find("::");
    if (colons == std::string::npos)
        return "::";
    size_t begin = prettyFunction.substr(0,colons).rfind(" ") + 1;
    size_t end = colons - begin;

    return prettyFunction.substr(begin,end);
}

//
#define __CLASS_NAME__ className(__PRETTY_FUNCTION__)

#endif //GPGPU_DEFINES_H
