#include "ConvolutionKernel1D.h"

#ifdef NO_ELISE
	#include "base_types.h"
#else
	#include "StdAfx.h"
#endif
#include "debug.h"

#include <cstring>
#include <cmath>

using namespace std;

//----------------------------------------------------------------------
// Kernel methods
//----------------------------------------------------------------------

template <class T>
inline ConvolutionKernel1D<T>::ConvolutionKernel1D( const ConvolutionKernel1D<T> &aKernel ){ set(aKernel); }

template <class T>
inline ConvolutionKernel1D<T>::ConvolutionKernel1D( const T *aCoefficients, size_t aSize, size_t aCenter, unsigned int aNbShift ){ set(aCoefficients,aSize,aCenter,aNbShift); }

template <class T>
inline ConvolutionKernel1D<T>::ConvolutionKernel1D( const T *aCoefficients, size_t aSize ){ set( aCoefficients, aSize, aSize/2, getNbShift(aCoefficients,aSize) ); }

template <class T>
inline ConvolutionKernel1D<T> & ConvolutionKernel1D<T>::operator =( const ConvolutionKernel1D<T> &aKernel )
{
	set(aKernel);
	return *this;
}

template <class T>
ConvolutionKernel1D<T>::ConvolutionKernel1D():
	mCoefficients( 1, T(1) ),
	mCenteredCoefficients( mCoefficients.data() ),
	mBegin(0),
	mEnd(0),
	mNbShift(0){}

template <class T>
inline ConvolutionKernel1D<T>::ConvolutionKernel1D( const std::vector<T> &aCoefficients, size_t aCenter, unsigned int aNbShift ){ set(aCoefficients,aCenter,aNbShift); }

template <class T>
inline void ConvolutionKernel1D<T>::set( const std::vector<T> &aCoefficients, size_t aCenter, unsigned int aNbShift ){ set(aCoefficients.data(),aCoefficients.size(),aCenter,aNbShift ); }

template <class T>
void ConvolutionKernel1D<T>::set( const T *aCoefficients, size_t aSize, size_t aCenter, unsigned int aNbShift )
{
	ELISE_DEBUG_ERROR( aSize==0, "ConvolutionKernel1D<T>::set", "aSize==0" );
	ELISE_DEBUG_ERROR( aCenter>=aSize, "ConvolutionKernel1D<T>::set", "aCenter = " << aCenter << ", out of range (max " << aSize-1 << ')' );

	// remove left zeros
	while ( *aCoefficients==0 && aSize>1 )
	{
		aCoefficients++;
		aSize--;
		aCenter--;
	}

	// remove right zeros
	const T *it = aCoefficients+aSize-1;
	while ( *it==0 && aSize>1 )
	{
		it--;
		aSize--;
	}

	mCoefficients.resize(aSize);
	memcpy( mCoefficients.data(), aCoefficients, aSize*sizeof(T) );
	mCenteredCoefficients = mCoefficients.data()+aCenter;
	mNbShift = aNbShift;
	mBegin = -(int)aCenter;
	mEnd = ( (int)aSize )-1+mBegin;

	// check for symmetry
	if ( aSize%2==0 )
		mIsSymmetric = false;
	else
	{
		mIsSymmetric = true;
		for ( int i=1; i<=mEnd; i++ )
			if ( mCenteredCoefficients[i]!=mCenteredCoefficients[-i] )
			{
				mIsSymmetric = false;
				break;
			}
	}
}

template <class T>
void ConvolutionKernel1D<T>::set( const ConvolutionKernel1D<T> &aKernel )
{
	mCoefficients = aKernel.mCoefficients;
	mBegin = aKernel.mBegin;
	mEnd = aKernel.mEnd;
	mNbShift = aKernel.mNbShift;
	mIsSymmetric = aKernel.mIsSymmetric;
	mCenteredCoefficients = mCoefficients.data()-mBegin;
}

template <class T>
inline T * ConvolutionKernel1D<T>::data(){ return mCenteredCoefficients; }

template <class T>
inline const T * ConvolutionKernel1D<T>::data() const{ return mCenteredCoefficients; }

template <class T>
inline const vector<T> & ConvolutionKernel1D<T>::coefficients() const{ return mCoefficients; }

template <class T>
inline int ConvolutionKernel1D<T>::begin() const{ return mBegin; }

template <class T>
inline int ConvolutionKernel1D<T>::end() const{ return mEnd; }

template <class T>
inline unsigned int ConvolutionKernel1D<T>::nbShift() const{ return mNbShift; }

template <class T>
bool ConvolutionKernel1D<T>::isSymmetric() const{ return mIsSymmetric; }

template <class T>
size_t ConvolutionKernel1D<T>::size() const { return mCoefficients.size(); }

template <class T>
T ConvolutionKernel1D<T>::sum() const
{
	T res = 0;
	const T *it = mCoefficients.data();
	size_t i = mCoefficients.size();
	while ( i-- ) res += *it++;
	return res;
}

template <class T>
void ConvolutionKernel1D<T>::dump( ostream &aStream ) const
{
	cout << '[';
	for ( int i=mBegin; i<0; i++ )
		aStream << mCenteredCoefficients[i] << ' ';
	aStream << '(' << mCenteredCoefficients[0] << ")";
	for ( int i=1; i<mEnd; i++ )
		aStream << ' ' << mCenteredCoefficients[i];
	if ( mEnd!=0 ) aStream << ' ' << mCenteredCoefficients[mEnd];
	aStream << ']';
	aStream << " sum=" << sum() << " shift=" << mNbShift << " 2^shift=" << (1<<mNbShift);
	aStream << (mIsSymmetric?" symmetric":" asymmetric") << endl;
}


//----------------------------------------------------------------------
// related functions
//----------------------------------------------------------------------

template <class T>
unsigned int getNbShift( const T *aKernel, size_t aSize )
{
	T sum = 0;
	while ( aSize-- ) sum += *aKernel++;
	unsigned int res = (unsigned int)( log((double)sum)/log(2.) );

	ELISE_DEBUG_WARNING( sum!=(1<<res), "getNbShift", "sum!=(1<<res)" );

	return res;
}


//----------------------------------------------------------------------
// instantiation
//----------------------------------------------------------------------

template class ConvolutionKernel1D<INT>;
template class ConvolutionKernel1D<REAL>;

template unsigned int getNbShift<INT>( const INT *aKernel, size_t aSize );
template unsigned int getNbShift<REAL>( const REAL *aKernel, size_t aSize );
