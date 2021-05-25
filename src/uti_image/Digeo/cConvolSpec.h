#ifndef __C_CONVOL_SPEC__
#define __C_CONVOL_SPEC__

#include "ConvolutionKernel1D.h"

#include <list>

#ifdef NO_ELISE
	#include "TypeTraits.h"
#else
	#include "StdAfx.h"
	#define TypeTraits El_CTypeTraits
#endif

template <class tData>
class cConvolSpec
{
public:
	virtual void Convol( tData *Out, const tData *In, int aK0, int aK1 ) const;
	virtual void ConvolCol( tData *Out, tData **In, int aX0, int aX1, int anYIn ) const;
	virtual bool IsCompiled() const;
	virtual cConvolSpec<tData> * duplicate() const;
	virtual ~cConvolSpec();

	cConvolSpec( const TBASE *aFilter, int aDeb, int aFin, int aNbShit );
	cConvolSpec( const ConvolutionKernel1D<TBASE> &aKernel );
	cConvolSpec( const cConvolSpec<tData> & );
	void set( const TBASE *aFilter, int aDeb, int aFin, int aNbShit );
	void set( const cConvolSpec<tData> &i_b );
	cConvolSpec<tData> & operator =( const cConvolSpec<tData> &i_b );

	int NbShift() const;
	int Deb() const;
	int Fin() const;
	const TBASE * DataCoeff() const;
	bool Sym() const;

	bool Match( const ConvolutionKernel1D<TBASE> &aKernel ) const;

private:
	int                mNbShift;
	int                mDeb;
	int                mFin;
	std::vector<TBASE> mCoeffs;
	TBASE *            mDataCoeff;
	bool               mSym;
};

template <class tData>
class ConvolutionHandler
{
private:
	std::list<cConvolSpec<tData>*> mConvolutions;

public:
	ConvolutionHandler();
	ConvolutionHandler( const ConvolutionHandler<tData> &aSrc );
	~ConvolutionHandler();

	void addCompiledKernels();

	void clear();
	void set( const ConvolutionHandler<tData> &i_b );

	ConvolutionHandler<tData> & operator =( const ConvolutionHandler<tData> &aSrc );

	cConvolSpec<tData> * getConvolution( const ConvolutionKernel1D<TBASE> &aKernel );
	cConvolSpec<tData> * getExistingConvolution( const ConvolutionKernel1D<TBASE> &aKernel );

	static std::string defaultCodeBasename();
	bool generateCode( const std::string &i_filename=defaultCodeBasename() ) const;

	size_t nbConvolutions() const;
	size_t nbConvolutionsNotCompiled() const;

};

template <class tData>
void convolution( const tData **aSrcData, const int aWidth, const int aHeight, const cConvolSpec<tData> &aConvolution1D, tData **aDstData );

template <class tData>
void legacy_convolution( const tData *aSrcData, const int aWidth, const int aHeight, tData *aTmpData, const ConvolutionKernel1D<TBASE> &aKernel, tData *aDstData );

template <class T>
int getNbShift( const T *aKernel, size_t aKernelSize );

template <class tData> 
tData ** new_data_lines( int i_width, int i_height );

template <class tData> 
void delete_data_lines( tData **i_data );

#endif
