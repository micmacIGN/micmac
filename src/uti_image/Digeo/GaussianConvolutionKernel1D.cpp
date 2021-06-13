#include "GaussianConvolutionKernel1D.h"
#ifdef NO_ELISE 
	#include "debug.h"
#else
	#include "StdAfx.h"
#endif

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <sstream>

#ifdef NO_ELISE
	#if ELISE_windows
		#define std_isnan _isnan
		#define std_isinf isinf
		#if _MSC_VER<_MSC_VER_2013
			double round( double aX );
		#endif
	#else
		#include <cmath>
		#define std_isnan std::isnan
		#define std_isinf std::isinf
	#endif
#endif

using namespace std;

// Methode pour avoir la "meilleure" approximation entiere d'une image reelle
// avec somme imposee. Tres sous optimal, mais a priori utilise uniquement sur de
// toute petites images

void normalize( REAL *aArray, size_t aSize )
{
	REAL aSom = 0;
	REAL *it = aArray;
	size_t i = aSize;
	while ( i-- ) aSom += *it++;

	while ( aSize-- ) *aArray++ /= aSom;
}

void ToIntegerKernel( vector<REAL> aSrcKernel, int aMul, bool aForceSym, vector<INT> &aDstKernel )
{
	int aSz = (int)aSrcKernel.size();
	ELISE_DEBUG_ERROR( aForceSym && (aSz%2==0), "ToIntegerKernel", "force symmetry on an odd sized kernel" );
	if ( aSz==0 )
	{
		ELISE_DEBUG_WARNING( true, "ToIntegerKernel", "empty kernel" );
		return;
	}

	normalize( aSrcKernel.data(), aSrcKernel.size() );
	aDstKernel.resize( aSrcKernel.size() );

	int aSom = 0, v;
	REAL *aDR = aSrcKernel.data();
	INT *aDI = aDstKernel.data();
	size_t i = aDstKernel.size();
	while ( i-- )
	{
		*aDI++ = v = (INT)round( (*aDR++)*aMul );
		aSom += v;
	}

	aDI = aDstKernel.data();
	aDR = aSrcKernel.data();
	const double aMul_d = (double)aMul;
	while ( aSom!=aMul )
	{
		int toAdd = aMul-aSom;
		int aSign = ( toAdd>0 )?1:-1;
		int aKBest;
		double aDeltaMin = 0.;

		if ( aForceSym && ( abs(toAdd)==1 ) )
			aKBest = aSz/2;
		else
		{
			aKBest = -1;
			for ( int aK=0; aK<aSz; aK++ )
			{
				if ( aDI[aK]+aSign>=0 )
				{
					aDeltaMin = ( ( aDI[0]/aMul_d )-aDR[0] )*aSign;
					aKBest = aK;
					break;
				}
			}
			if ( aKBest==-1 )
			{
				ELISE_DEBUG_WARNING( true, "ToIntegerKernel", "no suitable element found for adjustment" );
				return;
			}

			for ( int aK=aKBest; aK<aSz; aK++ )
			{
				double aDelta = ( ( aDI[aK]/aMul_d )-aDR[aK] )*aSign;
				if ( ( aDelta<aDeltaMin ) && ( aDI[aK]+aSign>=0 ) )
				{
					aDeltaMin = aDelta;
					aKBest= aK;
				}
			}
		}

		aDI[aKBest] += aSign;
		aSom += aSign;

		if ( aForceSym && (aSom!=aMul) )
		{
			int aKSym = aSz-aKBest-1;
			if ( aKSym!=aKBest )
			{
				aDI[aKSym] += aSign;
				aSom += aSign;
			}
		} 
	}
}

void ToOwnKernel( const vector<REAL> &aSrcKernel, int & aShift, bool aForceSym, vector<INT> &aDstKernel )
{
	ToIntegerKernel( aSrcKernel, 1<<aShift, aForceSym, aDstKernel );
}

/*
Im1D_REAL8 ToRealKernel( Im1D_INT4 aIK )
{
	Im1D_REAL8 aRK( aIK.tx() );
	ELISE_COPY( aIK.all_pts(), aIK.in(), aRK.out() );
	return MakeSom1(aRK);
}

Im1D_REAL8 ToRealKernel( Im1D_REAL8 aRK ) { return aRK; }

Im1D_REAL8 MakeSom( Im1D_REAL8 &i_vector, double i_dstSum )
{
	double aSomActuelle;
	Im1D_REAL8 aRes( i_vector.tx() );
	ELISE_COPY( i_vector.all_pts(), i_vector.in(), sigma(aSomActuelle) );
	ELISE_COPY( i_vector.all_pts(), i_vector.in()*( i_dstSum/aSomActuelle ), aRes.out() );
	return aRes;
}

Im1D_REAL8 MakeSom1( Im1D_REAL8 &i_vector ){ return MakeSom( i_vector, 1.0 ); }
*/

#ifdef __DEBUG_DIGEO_CONVOLUTIONS
static void check_is_valid( const string &aType, double aSigma, const vector<REAL> &aKernel )
{
	if ( aKernel.size()==0 ) return;
	bool isOk = true;
	for ( size_t i=0; i<aKernel.size(); i++ )
		if ( std_isnan(aKernel[i]) || std_isinf(aKernel[i]) )
		{
			isOk = false;
			break;
		}
	if ( isOk ) return;
	stringstream ss;
	ss << aKernel[0];
	for ( size_t i=1; i<aKernel.size(); i++ )
		ss << ' ' << aKernel[i];
	__elise_error( "invalid " << aType << " kernel for sigma = " << aSigma << " : [" << ss.str() << ']' );
}
#endif

void integralGaussianKernel( double aStandardDeviation, int aNbElements, int aSurEch, vector<REAL> &oKernel )
{
	oKernel.resize( 2*aNbElements+1 );

	const REAL d = REAL( 2*aSurEch+1 );
	for ( int aK=0; aK<=aNbElements; aK++ )
	{
		REAL aSom = 0;
		for ( int aKE=-aSurEch; aKE<=aSurEch; aKE++ )
		{
			REAL aX = aK - aNbElements + aKE/d;
			REAL x = aX/aStandardDeviation;
			REAL aG = exp( -(x*x)/2. );
			aSom += aG;
		}
		oKernel[aK] = oKernel[2*aNbElements-aK] = aSom;
	}

	normalize( oKernel.data(), oKernel.size() );

	#ifdef __DEBUG_DIGEO_CONVOLUTIONS
		check_is_valid( "integral", aStandardDeviation, oKernel );
	#endif
}

int integralGaussianKernelNbElements( double aSigma, double aResidu )
{
	return (int)ceil( sqrt(-2*log(aResidu))*aSigma );
}

void integralGaussianKernelFromResidue( double aSigma, double aResidu, int aSurEch, vector<REAL> &oKernel )
{
	return integralGaussianKernel( aSigma, integralGaussianKernelNbElements( aSigma, aResidu ), aSurEch, oKernel );
}

template <class T>
void integralGaussianKernel( double aSigma, int aNbShift, double aEpsilon, int aSurEch, ConvolutionKernel1D<T> &oKernel )
{
	vector<REAL> vKernelR;
	integralGaussianKernelFromResidue( aSigma, aEpsilon, aSurEch, vKernelR );
	vector<T> vKernel;
	ToOwnKernel( vKernelR, aNbShift, true, vKernel );
	oKernel.set( vKernel.data(), vKernel.size(), vKernel.size()/2, aNbShift );
}

template <> void integralGaussianKernel( double aSigma, int aNbShift, double aEpsilon, int aSurEch, ConvolutionKernel1D<REAL> &oKernel )
{
	vector<REAL> vKernel;
	integralGaussianKernelFromResidue( aSigma, aEpsilon, aSurEch, vKernel );
	oKernel.set( vKernel.data(), vKernel.size(), vKernel.size()/2, 0 );
}


//----------------------------------------------------------------------
// SampledGaussianKernel related functions
//----------------------------------------------------------------------

int sampledGaussianKernelNbElements( double aStandardDeviation ){ return (int)ceil( 4.*aStandardDeviation ); }

void sampledGaussianKernel( double aStandardDeviation, vector<REAL> &oKernel )
{
	int n = sampledGaussianKernelNbElements(aStandardDeviation),
	    N = 2*n+1;

	oKernel.resize(N);
	for ( int i=0; i<N; i++ )
		//accum += oKernel[i] = PixReal( std::exp( PixReal( -0.5 )*( i-n )*( i-n )/( PixReal( aStandardDeviation )*PixReal( aStandardDeviation ) ) ) );
		oKernel[i] = (REAL)std::exp( -0.5*( i-n )*( i-n )/( aStandardDeviation*aStandardDeviation ) );

	normalize( oKernel.data(), oKernel.size() );

	#ifdef __DEBUG_DIGEO_CONVOLUTIONS
		check_is_valid( "sampled", aStandardDeviation, oKernel );
	#endif
}

/*
void sampledGaussianKernel( double aSigma, vector<REAL> &oKernel )
{
	vector<float> kernel;
	createGaussianKernel_1d( aSigma, kernel );

	oKernel.resize( kernel.size() );
	const float *itSrc = kernel.data();
	REAL *itDst = oKernel.data();
	size_t i = kernel.size();
	while ( i-- ) *itDst++ = (REAL)( *itSrc++ );
}
*/

template <class T>
void sampledGaussianKernel( double aSigma, int aNbShift, ConvolutionKernel1D<T> &oKernel )
{
	vector<REAL> vKernelR;
	sampledGaussianKernel( aSigma, vKernelR );
	vector<T> vKernel;
	ToOwnKernel( vKernelR, aNbShift, true, vKernel );
	oKernel.set( vKernel.data(), vKernel.size(), vKernel.size()/2, aNbShift );
}

template <> void sampledGaussianKernel<REAL>( double aSigma, int /*aNbShift*/, ConvolutionKernel1D<REAL> &oKernel )
{
	vector<REAL> vKernel;
	sampledGaussianKernel( aSigma, vKernel );
	oKernel.set( vKernel.data(), vKernel.size(), vKernel.size()/2, 0 );
}

/*
template <> void sampledGaussianKernel( double aSigma, int aNbShift, ConvolutionKernel1D<float> &oKernel )
{
	vector<float> vKernel;
	createGaussianKernel_1d( aSigma, vKernel );
	oKernel.set( vKernel.data(), vKernel.size(), vKernel.size()/2, 0 );
}
*/

//----------------------------------------------------------------------
// instantiation
//----------------------------------------------------------------------

template void sampledGaussianKernel<INT>( double aSigma, int aNbShift, ConvolutionKernel1D<INT> &oKernel );
template void integralGaussianKernel<INT>( double aSigma, int aNbShift, double aEpsilon, int aSurEch, ConvolutionKernel1D<INT> &oKernel );

#if ELISE_windows && _MSC_VER>=_MSC_VER_2013
	template void integralGaussianKernel<REAL>( double aSigma, int aNbShift, double aEpsilon, int aSurEch, ConvolutionKernel1D<REAL> &oKernel );
	template void sampledGaussianKernel<REAL>( double aSigma, int aNbShift, ConvolutionKernel1D<REAL> &oKernel );
#endif
