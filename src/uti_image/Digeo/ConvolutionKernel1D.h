#ifndef __CONVOLUTION_KERNEL_1D__
#define __CONVOLUTION_KERNEL_1D__

#if defined(__MINGW__) && !defined(__MSVCRT_VERSION__)
	#define __MSVCRT_VERSION__ 0x800
#endif

#include <vector>
#include <cstdlib>
#include <iostream>

template <class T>
class ConvolutionKernel1D
{
public:
	ConvolutionKernel1D();
	ConvolutionKernel1D( const T *aCoefficients, size_t aSize, size_t aCenter, unsigned int aNbShift );
	ConvolutionKernel1D( const T *aCoefficients, size_t aSize );
	ConvolutionKernel1D( const std::vector<T> &aCoefficients, size_t aCenter, unsigned int aNbShift );
	ConvolutionKernel1D( const ConvolutionKernel1D<T> &aKernel );
	ConvolutionKernel1D<T> & operator =( const ConvolutionKernel1D<T> &aKernel );
	void set( const std::vector<T> &aCoefficients, size_t aCenter, unsigned int aNbShift );
	void set( const T *aCoefficients, size_t aSize, size_t aCenter, unsigned int aNbShift );
	void set( const ConvolutionKernel1D<T> &aKernel );

	T                    * data();
	const T              * data() const;
	int                    begin() const;
	int                    end() const;
	unsigned int           nbShift() const;
	bool                   isSymmetric() const;
	size_t                 size() const;
	const std::vector<T> & coefficients() const;

	T sum() const;
	void dump( std::ostream &aStream=std::cout ) const;

private:
	std::vector<T> mCoefficients;
	T *mCenteredCoefficients;
	int mBegin, mEnd;
	unsigned int mNbShift;
	bool mIsSymmetric;
};

template <class T>
unsigned int getNbShift( const T *aKernel, size_t aSize );

#endif
