#include <iostream>
#include <cstdlib>

#include "cConvolSpec.h"
#include "DiamondSquare.h"
#include "GaussianConvolutionKernel1D.h"
#include "Times.h"

#include <cstring>
#include <fstream>
#include <sstream>
#include <iomanip>

using namespace std;

const string digeoDirectory = "../../../src/uti_image/Digeo/";

U_INT2 generate_random_uint2(){ return (U_INT2)( rand()%65536 ); }

void write_pgm_header( ofstream &f, int w, int h, int vmax )
{
	f << "P5" << endl;
	f << w << ' ' << h << endl;
	f << vmax << endl;
}

void write_pgm( const string &filename, U_INT2 *data, int w, int h )
{
	ofstream f( filename.c_str(), ios::binary );
	write_pgm_header( f, w, h, 65535 );

	size_t nbBytes = (size_t)w*(size_t)h*2;
	char *tmp = new char[nbBytes];
	memcpy( tmp, data, nbBytes );

	char *itByte = (char*)tmp;
	size_t iByte = nbBytes;
	while ( iByte )
	{
		swap( itByte[0], itByte[1] );
		itByte += 2;
		iByte -= 2;
	}

	f.write( (char*)tmp, nbBytes );
	delete [] tmp;
}

void write_pgm( const string &filename, U_INT1 *data, int w, int h )
{
	ofstream f( filename.c_str(), ios::binary );
	write_pgm_header( f, w, h, 255 );
	f.write( (char*)data, w*h );
}

void write_pgm( const string &filename, REAL4 *data, int w, int h )
{
	size_t nbPix =size_t(w)*size_t(h);
	U_INT2 *data_ui2 = new U_INT2[nbPix];
	U_INT2 *itDst = data_ui2;
	size_t iPix = nbPix--;
	while ( iPix-- ) *itDst++ = (U_INT2)round(*data++*65535.+0.5);
	write_pgm( filename, data_ui2, w, h );
	delete [] data_ui2;
}

template <class T>
size_t nb_different_values( const T *data, int w, int h )
{
	size_t nbPix = (size_t)w*(size_t)h;

	const T nmin = TypeTraits<T>::nmin();
	size_t rangeSize = TypeTraits<T>::nmax()-nmin+1;
	cout << "rangeSize = " << rangeSize << endl;
	size_t *histo = new size_t[rangeSize];
	memset( histo, 0, rangeSize*sizeof(size_t) );

	while ( nbPix-- )
	{
		histo[(*data++)-nmin]++;
		data++;
	}

	size_t nbValues = 0;
	size_t *itHisto = histo;
	size_t iHisto = rangeSize;
	while ( iHisto-- )
		if ( *itHisto++!=0 ) nbValues++;

	delete [] histo;

	return nbValues;
}

template <class T>
size_t nb_different_values( const T *data0, int w, int h, const T *data1 )
{
	size_t iPix = (size_t)w*(size_t)h;
	size_t nbDifferentValues = 0;
	while ( iPix-- )
		if ( *data0++!=*data1++ ) nbDifferentValues++;
	return nbDifferentValues;
}

int main0( int argc, char **argv )
{
	ConvolutionHandler<U_INT1> convolutionHandler;

	cout << "nb compiled convolutions : " << convolutionHandler.nbConvolutions() << endl;

	ConvolutionKernel1D<INT> kernel;
	sampledGaussianKernel( 2., 15, kernel );
	cConvolSpec<U_INT1> *convolSpec = convolutionHandler.getConvolution(kernel);

	//srand(time(NULL));

	int width = 1024, height = 1024;

	/*
	U_INT2 **image = new_data_lines<U_INT2>(width,height);
	diamond_square<U_INT2>( image, width, height, generate_random_uint2(), generate_random_uint2(), generate_random_uint2(), generate_random_uint2() );
	*/

	U_INT1 **image = new_data_lines<U_INT1>(width,height);
	U_INT1 **image2 = new_data_lines<U_INT1>(width,height);
	U_INT1 **src = image, **dst = image2;
	diamond_square<U_INT1>( image, width, height, generate_random_uint2(), generate_random_uint2(), generate_random_uint2(), generate_random_uint2() );

	size_t nbIter = 0;
	{
		stringstream ss;
		ss << "diamond." << setw(3) << setfill('0') << nbIter++ << ".pgm";
		write_pgm( ss.str(), src[0], width, height );
	}

	memcpy( dst[0], src[0], (size_t)width*(size_t)height*sizeof(U_INT1) );
	dst[0][0]++;

	size_t nbValues;
	while ( ( nbValues=nb_different_values( src[0], width, height, dst[0] ) ) )
	{
		convolution( (const U_INT1 **)src, width, height, *convolSpec, dst );

		stringstream ss;
		ss << "diamond." << setw(3) << setfill('0') << nbIter << ".pgm";
		write_pgm( ss.str(), dst[0], width, height );

		swap( src, dst );
		nbIter++;
	}
	cout << "nbIter = " << nbIter-1 << endl;

	size_t nbNewConvolutions = convolutionHandler.nbConvolutionsNotCompiled();
	if ( nbNewConvolutions && convolutionHandler.generateCode( digeoDirectory+convolutionHandler.defaultCodeBasename() ) ) cout << "--- " << nbNewConvolutions << " new convolution" << (nbNewConvolutions>1?'s':'\0') << " generated" << endl;
	delete_data_lines(image);
	delete_data_lines(image2);

	return EXIT_SUCCESS;
}

template <class tData>
void compare_one_value( const cConvolSpec<tData> &aConvolution0, const cConvolSpec<tData> &aConvolution1 )
{
	// generate a simple test source
	size_t kernelSize = (size_t)( std::max<int>( aConvolution0.Fin()-aConvolution0.Deb(), aConvolution1.Fin()-aConvolution1.Deb() )+1 ),
	       center = kernelSize/2;
	tData *src = new tData[kernelSize];
	for ( size_t i=0; i<kernelSize; i++ )
		src[i] = (tData)i;

	tData dst0;
	aConvolution0.Convol( &dst0, src+center, 0, 1 );
	tData dst1;
	aConvolution1.Convol( &dst1, src+center, 0, 1 );

	delete [] src;

	if ( dst0!=dst1 ) __elise_warning( "compare_one_value<" << TypeTraits<tData>::Name() <<"> failed : " << (TBASE)dst0 << " != " << (TBASE)dst1 );
}

template <class tData>
void compare_compiled_not_compiled( int aWidth, int aHeight, unsigned int aNbIterations, bool aDoOutputImages )
{
	ConvolutionHandler<tData> convolutionHandler;
	ConvolutionKernel1D<TBASE> kernel;
	sampledGaussianKernel( 2., 15, kernel );
	cConvolSpec<tData> *convolSpec = convolutionHandler.getConvolution(kernel);

	if ( !convolSpec->IsCompiled() )
	{
		ELISE_WARNING( "convolution is not compiled, generating code and skiping 'compiled' VS 'not compiled' comparison" );
		convolutionHandler.generateCode( digeoDirectory+convolutionHandler.defaultCodeBasename() );
		return;
	}

	cConvolSpec<tData> convolSpecNotCompiled(*convolSpec);
	if ( convolSpecNotCompiled.IsCompiled() ) ELISE_ERROR_EXIT( "convolSpecNotCompiled.IsCompiled() = true" );

	/*
	disabled until there is a comparison between convolutions (ie. cConvolSpec inherits from ConvolutionKernel)

	if ( !convolSpec->Match(convolSpecNotCompiled) ) ELISE_WARNING( "!convolSpec->Match(convolSpecNotCompiled)" );
	{
		__elise_warning( "compiled and not compiled convolution do not Match" );
		for ( int i=convolSpec->Deb(); i<=convolSpec->Fin(); i++ )
			cout << convolSpec->DataCoeff()[i] << ' ';
		cout << endl;

		for ( int i=convolSpecNotCompiled.Deb(); i<=convolSpecNotCompiled.Fin(); i++ )
			cout << convolSpecNotCompiled.DataCoeff()[i] << ' ';
		cout << endl;
	}
	*/

	compare_one_value( *convolSpec, convolSpecNotCompiled );

	tData **src = new_data_lines<tData>(aWidth,aHeight);
	diamond_square<tData>( src, aWidth, aHeight, generate_random_uint2(), generate_random_uint2(), generate_random_uint2(), generate_random_uint2() );

	if ( aDoOutputImages ) write_pgm( "src.pgm", src[0], aWidth, aHeight );

	int iIteration;
	MapTimes times;
	tData **dst0 = new_data_lines<tData>(aWidth,aHeight);
	times.start();
		iIteration = aNbIterations;
		while ( iIteration-- )
			convolution( (const tData **)src, aWidth, aHeight, *convolSpec, dst0 );
	times.stop("compiled");

	if ( aDoOutputImages ) write_pgm( "dst0.pgm", dst0[0], aWidth, aHeight );

	tData **dst1 = new_data_lines<tData>(aWidth,aHeight);
	times.start();
		iIteration = aNbIterations;
		while ( iIteration-- )
			convolution( (const tData **)src, aWidth, aHeight, convolSpecNotCompiled, dst1 );
	times.stop("not compiled");

	if ( aDoOutputImages ) write_pgm( "dst1.pgm", dst1[0], aWidth, aHeight );

	size_t diff = nb_different_values( dst0[0], aWidth, aHeight, dst1[0] );

	delete_data_lines(src);
	delete_data_lines(dst0);
	delete_data_lines(dst1);

	if ( diff!=0 ) ELISE_ERROR_EXIT( "compare_compiled_not_compiled: diff = " << diff );
	cout << "time ratio : " << times.getRecordTime("not compiled")/times.getRecordTime("compiled") << endl;

	times.printTimes();
}

template <class tData>
class SetIterator
{
protected:
	double                     mSigma;
	ConvolutionHandler<tData>  mHandler;
	ConvolutionKernel1D<TBASE> mKernel;
	cConvolSpec<tData>         mNotCompiledConvolution;
	cConvolSpec<tData>       * mCompiledConvolution;
	tData                   ** mImageData;
	int                        mImageWidth, mImageHeight;
	int                        mNbIterations;

	void generateImage( int aWidth, int aHeight )
	{
		if ( aWidth==mImageWidth && aHeight==mImageHeight ) return;
		mImageWidth = aWidth;
		mImageHeight = aHeight;
		freeImage();
		mImageData = new_data_lines<tData>(mImageWidth,mImageHeight);
		diamond_square<tData>( mImageData, mImageWidth, mImageHeight, generate_random_uint2(), generate_random_uint2(), generate_random_uint2(), generate_random_uint2() );
	}

	void freeImage()
	{
		if ( mImageData!=NULL ) delete_data_lines<tData>(mImageData);
	}

	void updateFromSigma()
	{
		sampledGaussianKernel( mSigma, 15, mKernel );
		mCompiledConvolution = mHandler.getConvolution(mKernel);
		mNotCompiledConvolution.set(*mCompiledConvolution);
	}

	//virtual ~SetIterator(){ freeImage(); }

public:
	SetIterator( int aWidth, int aHeight, double aSigma, int aNbIterations ):
		mSigma(aSigma),
		mNotCompiledConvolution(mKernel),
		mCompiledConvolution(&mNotCompiledConvolution),
		mImageData(NULL),
		mImageWidth(0),
		mImageHeight(0),
		mNbIterations(aNbIterations)
	{
		if ( aWidth<=0 || aHeight<=0 ) ELISE_ERROR_EXIT( "SetIterator::SetIterator: invalid size " << aWidth << 'x' << aHeight );
		if ( aSigma<=0. ) ELISE_ERROR_EXIT( "SetIterator::SetIterator: invalid sigma " << aSigma );
		if ( aNbIterations<=0 ) ELISE_ERROR_EXIT( "SetIterator::SetIterator: invalid number of iterations " << aNbIterations );

		generateImage(aWidth,aHeight);
		updateFromSigma();
	}

	double                             sigma()                  const { return mSigma; }
	const ConvolutionHandler<tData>  & handler()                const { return mHandler; }
	const cConvolSpec<tData>         & notCompiledConvolution() const { return mNotCompiledConvolution; }
	const cConvolSpec<tData>         & compiledConvolution()    const { return *mCompiledConvolution; }
	const ConvolutionKernel1D<TBASE> & kernel()                 const { return mKernel; }
	int                                imageWidth()             const { return mImageWidth; }
	int                                imageHeight()            const { return mImageHeight; }
	const tData                     ** imageData()              const { return (const tData **)mImageData; }
	int                                nbIterations()           const { return mNbIterations; }

	virtual bool next() = 0;
};

template <class tData>
class SigmaSetIterator : public SetIterator<tData>
{
private:
	double mSigma0, mSigma1, mSigmaPace;

public:
	SigmaSetIterator( int aWidth, int aHeight, double aSigma0, double aSigma1, double aSigmaPace, int aNbIterations ):
		SetIterator<tData>(aWidth,aHeight,aSigma0,aNbIterations),
		mSigma0(aSigma0), mSigma1(aSigma1), mSigmaPace(aSigmaPace)
	{
		if ( aSigma0<=0. || aSigmaPace<=0. )
			ELISE_ERROR_EXIT("SigmaSetIterator::SigmaSetIterator: invalid parameters aSigma0="<<aSigma0 << " aSigma1=" << aSigma1 << " aSigmaPace=" << aSigmaPace);
		SetIterator<tData>::generateImage(SetIterator<tData>::mImageWidth,SetIterator<tData>::mImageHeight);
	}

	bool next()
	{
		double sigma = SetIterator<tData>::mSigma+mSigmaPace;
		if ( sigma>mSigma1 ) return false;
		SetIterator<tData>::mSigma = sigma;
		SetIterator<tData>::updateFromSigma();
		return true;
	}

	double sigma0() const{ return mSigma0; }
	double sigma1() const{ return mSigma1; }
	double sigmaPace() const{ return mSigmaPace; }
};

template <class tData>
class ImageSizeSetIterator : public SetIterator<tData>
{
private:
	int mWidth0, mHeight0, mWidth1, mHeight1, mWidthPace, mHeightPace;

public:
	ImageSizeSetIterator( int aWidth0, int aHeight0, double aSigma, int aWidthPace, int aHeightPace, int aWidth1, int aHeight1, int aNbIterations ):
		SetIterator<tData>(aWidth0,aHeight0,aSigma,aNbIterations),
		mWidth0(aWidth0), mHeight0(aHeight0), mWidth1(aWidth1), mHeight1(aHeight1), mWidthPace(aWidthPace), mHeightPace(aHeightPace)
	{
		if ( aWidthPace<=0 || aHeightPace<=0 )
			ELISE_ERROR_EXIT("ImageSizeSetIterator::ImageSizeSetIterator: invalid parameters aWidthPace=" << aWidthPace << " aHeightPace=" << aHeightPace);
		SetIterator<tData>::generateImage(SetIterator<tData>::mImageWidth,SetIterator<tData>::mImageHeight);
	}

	bool next()
	{
		int width = SetIterator<tData>::mImageWidth+mWidthPace, height = SetIterator<tData>::mImageHeight+mHeightPace;
		if ( width>mWidth1 && height>mHeight1 ) return false;
		SetIterator<tData>::generateImage(width,height);
		return true;
	}

	int width0() const { return mWidth0; }
	int height0() const { return mHeight0; }
	int width1() const { return mWidth1; }
	int height1() const { return mHeight1; }
	int widthPace() const { return mWidthPace; }
	int heightPace() const { return mHeightPace; }
};

typedef struct
{
	double sigma;
	int width, height;
	size_t kernelSize;
	double timeCompiled, timeNotCompiled, timeLegacy;
} TimeEntry;

template <class tData>
void times_along_set( string aBasename, SetIterator<tData> &aSetIterator )
{
	cout << "--- times_along_set<" << TypeTraits<tData>::Name() << '>' << endl;
	aBasename.append( std::string(".")+TypeTraits<tData>::Name()+".txt" );

	// generate src and dst image
	int dstWidth = 0, dstHeight = 0;
	tData *tmp = NULL;
	tData **dst = NULL;

	const int nbIterations = aSetIterator.nbIterations();
	const double nbIterationsD = (double)nbIterations;
	bool hasNotCompiledConvolutions = false;
	TimeEntry timeEntry;
	list<TimeEntry> timeEntries;
	list<size_t> kernelSizes;
	int iIteration;

	do
	{
		const int width  = aSetIterator.imageWidth();
		const int height = aSetIterator.imageHeight();
		const double sigma = aSetIterator.sigma();
		const tData **src = aSetIterator.imageData();
		const ConvolutionKernel1D<TBASE> &kernel = aSetIterator.kernel();
		const cConvolSpec<tData> &compiledConvolution = aSetIterator.compiledConvolution();
		const cConvolSpec<tData> &notCompiledConvolution = aSetIterator.notCompiledConvolution();

		if ( width==0 || height==0 ) ELISE_ERROR_EXIT( "times_along_set: invalid image size " << width << 'x' << height );

		cout << width << 'x' << height << ' ' << sigma << endl;

		// reallocate dst if needed
		if ( dstWidth!=width || dstHeight!=height )
		{
			if ( dst!=NULL ) delete_data_lines(dst);
			delete [] tmp;
			dstWidth = width;
			dstHeight = height;
			dst = new_data_lines<tData>(dstWidth,dstHeight);
			tmp = new tData[dstWidth*dstHeight];
		}

		if ( !aSetIterator.compiledConvolution().IsCompiled() ) hasNotCompiledConvolutions = true;

		kernelSizes.push_back( kernel.size() );

		Timer timer;
		iIteration = nbIterations;
		while ( iIteration-- ) convolution( src, width, height, compiledConvolution, dst );
		timeEntry.timeCompiled = timer.uval()/nbIterationsD;

		timer.reinit();
		iIteration = nbIterations;
		while ( iIteration-- ) convolution( src, width, height, notCompiledConvolution, dst );
		timeEntry.timeNotCompiled = timer.uval()/nbIterationsD;

		timer.reinit();
		iIteration = nbIterations;
		while ( iIteration-- )
			legacy_convolution( src[0], width, height, tmp, kernel, dst[0] );
		timeEntry.timeLegacy = timer.uval()/nbIterationsD;

		timeEntry.sigma = sigma;
		timeEntry.width = width;
		timeEntry.height = height;
		timeEntry.kernelSize = kernel.size();
		timeEntries.push_back(timeEntry);
	}
	while ( aSetIterator.next() );

	delete_data_lines(dst);
	delete [] tmp;

	ofstream f( aBasename.c_str() );
	f << "# " << TypeTraits<tData>::Name() << endl;
	f << "# sigma width height kernel_size compiled_time not_compiled_time legacy_time" << endl;
	list<TimeEntry>::const_iterator itEntry = timeEntries.begin();
	while ( itEntry!=timeEntries.end() )
	{
		const TimeEntry &entry = *itEntry++;
		f << entry.sigma << '\t' << entry.width << '\t' << entry.height << '\t' << entry.kernelSize
		  << '\t' << entry.timeCompiled << '\t' << entry.timeNotCompiled << '\t' << entry.timeLegacy << endl;
	}

	if ( hasNotCompiledConvolutions )
	{
		ELISE_WARNING( "some convolutions are not compiled: generating code (compiled times will not be relevant)" );
		aSetIterator.handler().generateCode( digeoDirectory+aSetIterator.handler().defaultCodeBasename() );
	}
}

int main( int argc, char **argv )
{
	const int width = 1024, height = 1024;
	const double sigma0 = 0.1, sigma1 = 10., sigmaPace = 0.1;
	unsigned int nbIterations = 30;
	const string outputBasename = "sigma_convolution_times";

	if ( argc>1 ) nbIterations = atoi( argv[1] );
	if ( nbIterations<1 ) ELISE_ERROR_EXIT( "invalid number of iterations : " << nbIterations );

	//const bool doOutputImages = false;

	//srand(time(NULL)); // for diamond_square generation

	/*
	compare_compiled_not_compiled<U_INT1>( width, height, nbIterations, doOutputImages );
	cout << "-------------------------" << endl;
	compare_compiled_not_compiled<U_INT2>( width, height, nbIterations, doOutputImages );
	cout << "-------------------------" << endl;
	compare_compiled_not_compiled<REAL4>( width, height, nbIterations, doOutputImages );
	*/

	cout << "*********************************** time along sigma ***********************************" << endl;
	SigmaSetIterator<U_INT1> sigmaSet_ui1( width, height, sigma0, sigma1, sigmaPace, nbIterations );
	times_along_set(outputBasename,sigmaSet_ui1);

	SigmaSetIterator<U_INT2> sigmaSet_ui2( width, height, sigma0, sigma1, sigmaPace, nbIterations );
	times_along_set(outputBasename,sigmaSet_ui2);

	SigmaSetIterator<REAL4> sigmaSet_r4( width, height, sigma0, sigma1, sigmaPace, nbIterations );
	times_along_set(outputBasename,sigmaSet_r4);

	cout << endl;
	cout << "********************************* time along image size ********************************" << endl;
	const double sigma = 2.;
	const int width0 = 128, height0 = 128, width1 = 6400, height1 = 6400, widthPace = 128, heightPace = 128;
	ImageSizeSetIterator<U_INT1> imageSizeSet( width0, height0, sigma, widthPace, heightPace, width1, height1, nbIterations );
	times_along_set("image_size_convolution_times.txt",imageSizeSet);

	return EXIT_SUCCESS;
}
