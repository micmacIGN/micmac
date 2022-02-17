#ifndef __GAUSSIAN_CONVOLUTION_KERNEL_1D__
#define __GAUSSIAN_CONVOLUTION_KERNEL_1D__

#include "ConvolutionKernel1D.h"

#ifdef NO_ELISE
	#include "base_types.h"
#else
	#include "general/sys_dep.h"
#endif

//----------------------------------------------------------------------
// DigeoGaussianKernel
//----------------------------------------------------------------------

// Methode pour avoir la "meilleure" approximation entiere d'une image reelle
// avec somme imposee. Tres sous optimal, mais a priori utilise uniquement sur de
// toute petites images

void integralGaussianKernel( double aSigma, int aNbElements, int aSurEch, std::vector<REAL> &oKernel );

int integralGaussianKernelNbElements( double aSigma, double aResidu );

void integralGaussianKernelFromResidue( double aSigma, double aResidu, int aSurEch, ConvolutionKernel1D<REAL> &oKernel );

template <class T>
void integralGaussianKernel( double aSigma, int aNbShift, double aEpsilon, int aSurEch, ConvolutionKernel1D<T> &oKernel );


//----------------------------------------------------------------------
// SampledGaussianKernel
//----------------------------------------------------------------------

int sampledGaussianKernelNbElements( double aStandardDeviation );

void sampledGaussianKernel( double aSigma, std::vector<REAL> &oKernel );

template <class T>
void sampledGaussianKernel( double aSigma, int aNbShift, ConvolutionKernel1D<T> &oKernel );

//----------------------------------------------------------------------
// related functions
//----------------------------------------------------------------------

void normalize( REAL *aArray, size_t aSize );

#endif
