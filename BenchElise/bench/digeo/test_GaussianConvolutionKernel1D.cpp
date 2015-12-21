#include "GaussianConvolutionKernel1D.h"

#include <iostream>

using namespace std;

template <class T0, class T1>
REAL kernel_difference( const ConvolutionKernel1D<T0> &aKernel0, const ConvolutionKernel1D<T1> &aKernel1 )
{
	REAL diff = 0.,
	     scale0 = 1./(REAL)( 1<<aKernel0.nbShift() ),
	     scale1 = 1./(REAL)( 1<<aKernel1.nbShift() );
	const T0 *data0 = aKernel0.data();
	const T1 *data1 = aKernel1.data();
	int minBegin = min( aKernel0.begin(), aKernel1.begin() ),
	    maxEnd   = max( aKernel0.end(), aKernel1.end() );
	for ( int i=minBegin; i<=maxEnd; i++ )
	{
		REAL v0 = ( i>=aKernel0.begin() && i<=aKernel0.end() ) ? ( (REAL)data0[i] )*scale0 : 0.;
		REAL v1 = ( i>=aKernel1.begin() && i<=aKernel1.end() ) ? ( (REAL)data1[i] )*scale1 : 0.;
		REAL v = v0-v1;
		if ( v<0. ) v = -v;
		diff += v;
	}
	return diff;
}

void all_kernel_differences( double aSigma, int aShift, double aEpsilon, int aSurEch )
{
	ConvolutionKernel1D<INT> kernelSampledInt, kernelIntegralInt;
	ConvolutionKernel1D<REAL> kernelSampledReal, kernelIntegralReal;
	sampledGaussianKernel( aSigma, aShift, kernelSampledInt );
	sampledGaussianKernel( aSigma, aShift, kernelSampledReal );
	integralGaussianKernel( aSigma, aShift, aEpsilon, aSurEch, kernelIntegralInt );
	integralGaussianKernel( aSigma, aShift, aEpsilon, aSurEch, kernelIntegralReal );
	cout << "sampled,int / sampled,real  : " << kernel_difference( kernelSampledInt, kernelSampledReal ) << endl;
	cout << "integral,int / integral,real  : " << kernel_difference( kernelIntegralInt, kernelIntegralReal ) << endl;
	cout << "---" << endl;
	cout << "sampled,int / integral,int  : " << kernel_difference( kernelSampledInt, kernelIntegralInt ) << endl;
	cout << "sampled,real / integral,real  : " << kernel_difference( kernelSampledReal, kernelIntegralReal ) << endl;
	cout << "---" << endl;
	cout << "sampled,int / integral,real  : " << kernel_difference( kernelSampledInt, kernelIntegralInt ) << endl;
	cout << "sampled,real / integral,int  : " << kernel_difference( kernelSampledReal, kernelIntegralInt ) << endl;
}

int main( int argc, char **argv )
{
	all_kernel_differences( 2., 15, 1e-3, 10 ); // 2 = sigma, 15 = nbShift, 1e-3 = residue, 10 = surEch

	return EXIT_SUCCESS;
}
