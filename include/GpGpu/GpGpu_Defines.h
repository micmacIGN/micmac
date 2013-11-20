#ifndef GPGPU_DEFINES_H
#define GPGPU_DEFINES_H

typedef unsigned char pixel;

#define NOPAGLOCKMEM false
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

#define CUDA_DUMP_INT(varname) if(!threadIdx.x) printf("%s = %d\n", #varname, varname);
#define CUDA_DUMP_INT_ALL(varname) printf("%s = %d\n", #varname, varname);

enum Plans {XY,XZ,YZ,YX,ZX,ZY};

#endif //GPGPU_DEFINES_H
