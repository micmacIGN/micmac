#define __DEBUG

#include "StdAfx.h"
#include "../src/uti_image/Digeo/Convolution.h"
#include "../src/uti_image/Digeo/GaussianConvolutionKernel.h"
#include "../src/uti_image/Digeo/MultiChannel.h"

using namespace std;

template <class T>
class TypeInfo
{
public:
	static T nmin(){ return numeric_limits<T>::min(); }
	static T nmax(){ return numeric_limits<T>::max(); }
};

// REAL4
template <> inline REAL4 TypeInfo<REAL4>::nmin(){ return (REAL4)0; }
template <> inline REAL4 TypeInfo<REAL4>::nmax(){ return (REAL4)1; }
// REAL8
template <> inline REAL8 TypeInfo<REAL8>::nmin(){ return (REAL8)0; }
template <> inline REAL8 TypeInfo<REAL8>::nmax(){ return (REAL8)1; }

class DiamondSquareRectangle
{
public:
	Pt2di p0, p1;
	U_INT2 v0, v1, v2, v3;

	DiamondSquareRectangle( const Pt2di &i_p0, const Pt2di &i_p1, INT i_v0, INT i_v1, INT i_v2, INT i_v3 ):
		p0(i_p0), p1(i_p1),
		v0(i_v0), v1(i_v1), v2(i_v2), v3(i_v3){}
};

INT add_error( INT i_v, double i_dimensionFactor )
{
	const int maxErrorPercentage_1 = 71; // actually max error percentage +1
	const INT dataMax = 65535;
	const double scale = i_dimensionFactor*( ( (double)dataMax )/100. );
	const double ek = 0.;

	INT e = round_ni( ( (double)(rand()%maxErrorPercentage_1) )*scale + ek );
	if ( rand()%2==0 ) e = -e;

	INT res = i_v+e;
	if ( res<0 ) res = 0;
	if ( res>dataMax ) res = dataMax;

	return res;
}

template <class tDataSrc, class tDataDst>
void ramp( const tDataSrc *i_src, int i_width, int i_height, const tDataSrc i_srcMin, const tDataSrc i_srcMax, const tDataDst i_dstMin, const tDataDst i_dstMax, tDataDst *o_dst )
{
	typedef typename El_CTypeTraits<tDataSrc>::tBase tBaseSrc;
	typedef typename El_CTypeTraits<tDataDst>::tBase tBaseDst;

	const double srcMin = (double)i_srcMin;
	const double dstMin = (double)i_dstMin;
	const double dstMax = (double)i_dstMax;
	const double scale = ( dstMax-dstMin )/( (double)i_srcMax-srcMin );
	int iPix = i_width*i_height;

	while ( iPix-- )
	{
		tBaseDst v = ( ( (double)(*i_src++) )-srcMin )*scale+dstMin;
		if ( v<i_dstMin ) v = i_dstMin;
		if ( v>i_dstMax ) v = i_dstMax;
		*o_dst++ = (tDataDst)v;
	}
}

template <class tData>
void diamond_square( tData **i_data, int i_width, int i_height, U_INT2 i_v0, U_INT2 i_v1, U_INT2 i_v2, U_INT2 i_v3 )
{
	if ( i_width==0 && i_height==0 ) return;

	Im2D<U_INT2,INT> tmp( i_width, i_height );
	diamond_square( tmp.data(), i_width, i_height, i_v0, i_v1, i_v2, i_v3 );
	ramp<U_INT2,tData>( tmp.data_lin(), i_width, i_height, (U_INT2)0, (U_INT2)65535, TypeInfo<tData>::nmin(), TypeInfo<tData>::nmax(), i_data[0] );
}

template <> void diamond_square( U_INT2 **i_data, int i_width, int i_height, U_INT2 i_v0, U_INT2 i_v1, U_INT2 i_v2, U_INT2 i_v3 )
{
	cout << i_v0 << ' ' << i_v1 << ' ' << i_v2 << ' ' << i_v3 << endl;

	list<DiamondSquareRectangle> rectangles;
	rectangles.push_back( DiamondSquareRectangle( Pt2di(0,0), Pt2di(i_width-1,i_height-1), i_v0, i_v1, i_v2, i_v3 ) );

	const double maxDimension = (double)min<int>( i_width, i_height );
	while ( rectangles.begin()!=rectangles.end() )
	{
		DiamondSquareRectangle rect = rectangles.front();
		rectangles.pop_front();

		#ifdef __DEBUG
			const INT vmax = 65535;
		#endif
		ELISE_DEBUG_ERROR( rect.v0<0 || rect.v0>vmax, "diamond_square", "rect.v0 = " << rect.v0 << " vmax = " << vmax );
		ELISE_DEBUG_ERROR( rect.v1<0 || rect.v1>vmax, "diamond_square", "rect.v1 = " << rect.v1 << " vmax = " << vmax );
		ELISE_DEBUG_ERROR( rect.v2<0 || rect.v2>vmax, "diamond_square", "rect.v2 = " << rect.v2 << " vmax = " << vmax );
		ELISE_DEBUG_ERROR( rect.v3<0 || rect.v3>vmax, "diamond_square", "rect.v3 = " << rect.v3 << " vmax = " << vmax );

		i_data[rect.p0.y][rect.p0.x] = (U_INT2)rect.v0;
		i_data[rect.p0.y][rect.p1.x] = (U_INT2)rect.v1;
		i_data[rect.p1.y][rect.p0.x] = (U_INT2)rect.v2;
		i_data[rect.p1.y][rect.p1.x] = (U_INT2)rect.v3;

		Pt2di rectSize = rect.p1-rect.p0+Pt2di(1,1);
		const double dimensionFactor = ( (double)min<int>( rectSize.x, rectSize.y ) )/maxDimension;
		INT iv0 = add_error( ( rect.v0+rect.v1 )/2, 0 ),
		    iv1 = add_error( ( rect.v0+rect.v2 )/2, 0 ),
		    iv2 = add_error( ( rect.v1+rect.v3 )/2, 0 ),
		    iv3 = add_error( ( rect.v2+rect.v3 )/2, 0 ),
		    iv4 = add_error( ( rect.v0+rect.v1+rect.v2+rect.v3 )/4, dimensionFactor );
		Pt2di ip = ( rect.p0+rect.p1 )/2;
		if ( rectSize.x>2 || rectSize.y>2 )
		{
			rectangles.push_back( DiamondSquareRectangle( rect.p0, ip, rect.v0, iv0, iv1, iv4 ) );
			rectangles.push_back( DiamondSquareRectangle( Pt2di(ip.x,rect.p0.y), Pt2di(rect.p1.x,ip.y), iv0, rect.v1, iv4, iv2 ) );
			rectangles.push_back( DiamondSquareRectangle( Pt2di(rect.p0.x,ip.y), Pt2di(ip.x,rect.p1.y), iv1, iv4, rect.v2, iv3 ) );
			rectangles.push_back( DiamondSquareRectangle( ip, rect.p1, iv4, iv2, iv3, rect.v3 ) );
		}
	}
}

string lower( const string &i_str );

template <class T>
string outputName( int i_nbConvolutions, const string &i_suffix )
{
	stringstream ss;
	ss << "test_convolution." << lower(El_CTypeTraits<T>::Name()) << '.' << setw(2) << setfill('0') << i_nbConvolutions << i_suffix;
	return ss.str(); 
}

template <class tData>
void save_to_tiff( Im2D<tData,typename El_CTypeTraits<tData>::tBase> &i_image, const string &i_filename )
{
	/*
	MultiChannel<tData> c;
	c.link(i_image);
	c.write_tiff(i_filename);
	*/

	// __DEL
	cout << "save_to_tiff<" << El_CTypeTraits<tData>::Name() << ">: image.sz() = " << i_image.sz() << endl;

	ELISE_COPY
	(
		i_image.all_pts(),
		i_image.in(),
		Tiff_Im(
			i_filename.c_str(),
			i_image.sz(),
			sizeof(tData)==1?GenIm::u_int1:GenIm::u_int2,
			Tiff_Im::No_Compr,
			Tiff_Im::BlackIsZero,
			Tiff_Im::Empty_ARG ).out()
	);
}

template <class tData>
void save_to_raw( Im2D<tData,typename El_CTypeTraits<tData>::tBase> &i_image, const string &i_filename )
{
	MultiChannel<tData> c;
	c.link(i_image);
	c.write_raw(i_filename);
}

template <class tData>
void save_to_pnm( Im2D<tData,typename El_CTypeTraits<tData>::tBase> &i_image, const string &i_filename )
{
	MultiChannel<tData> c;
	c.link(i_image);
	c.write_pnm(i_filename);
}

template <class tData, class tBase>
void get_min_max( Im2D<tData,tBase> &i_image, tData &o_min, tData &o_max )
{
	if ( i_image.tx()==0 || i_image.ty()==0 ) return;
	const tData *itPix = i_image.data_lin();
	o_min = o_max = *itPix++;
	int iPix = i_image.tx()*i_image.ty()-1;
	while ( iPix-- )
	{
		tData v = *itPix++;
		if ( v<o_min ) o_min = v;
		else if ( v>o_max ) o_max = v;
	}
}

template <> void save_to_tiff<REAL4>( Im2D<REAL4,REAL> &i_image, const string &i_filename )
{
	Im2D<U_INT2,INT> dst( i_image.tx(), i_image.ty() );
	const REAL4 *itSrc = i_image.data_lin();
	U_INT2 *itDst = dst.data_lin();
	size_t iPix = ( (size_t)i_image.tx() )*( (size_t)i_image.ty() );
	while ( iPix-- ) *itDst++ = (U_INT2)( ( *itSrc++ )*65535 );

	save_to_tiff( dst, i_filename );
}

U_INT2 generate_random_uint2(){ return (U_INT2)( rand()%65536 ); }

template <class tData>
void test_digeo_convolution( const tData **i_src, int i_width, int i_height, tBase *i_kernel, int i_kernelLength, tData **o_dst, int i_nbIteration )
{
	cConvolSpec<tData> *convolSpec = ToCompKer<tData>( i_kernel );
	while ( i_nbIterations )
	{
		if ( !convolution( (const tData **)src->data(), src->tx(), src->ty(), kernel, dst->data() ) )
		{
			usedSlowConvolution = true;
			ELISE_WARNING( "using slow convolution for type " << El_CTypeTraits<tData>::Name() << ", sigma = " << sigma );
		}

		//save_to_raw( *dst, outputName<tData>(nbConvolutions,".tif") );
		swap( src, dst );
	}
}

template <class tData>
void test_convolution()
{
	typedef typename El_CTypeTraits<tData>::tBase tBase;
 
	Im2D<tData,tBase> channel0( 256, 256, 1 ), channel1( channel0.tx(), channel0.ty() );
	Im2D<tData,tBase> *src = &channel0, *dst = &channel1;

	diamond_square<tData>( src->data(), src->tx(), src->ty(), generate_random_uint2(), generate_random_uint2(), generate_random_uint2(), generate_random_uint2() );

	DigeoConvolution<tData> convolution;
	vector<vector<tBase> > compiledKernels;
	convolution.getCompiledKernels(compiledKernels);
	cout << "nb compiled kernels : " << compiledKernels.size() << endl;

	Im1D<tBase,tBase> kernel = DigeoGaussianKernel<tBase>( sigma, 15, 0.001, 10 ); // 1.6 = sigma, 12 = nb shift, 0.01 = epsilon, 10 = surEch

	const int nbIterations = 5;
	const double sigma = 1.6;
	ElTimer chrono;
	test_digeo_convolution( channel0.data(), channel0.tx(), channel0.ty(), channel1.data(), nbIterations );
	bool usedSlowConvolution = false;
	for ( int nbConvolutions=1; nbConvolutions<=5; nbConvolutions++ )
	{

		if ( !convolution( (const tData **)src->data(), src->tx(), src->ty(), kernel, dst->data() ) )
		{
			usedSlowConvolution = true;
			ELISE_WARNING( "using slow convolution for type " << El_CTypeTraits<tData>::Name() << ", sigma = " << sigma );
		}

		//save_to_raw( *dst, outputName<tData>(nbConvolutions,".tif") );
		swap( src, dst );
	}
	cout << chrono.uval() << "ms" << endl;
	chrono.reinit();

	

	if ( usedSlowConvolution )
	{
		cout << "--- generating code for type " << El_CTypeTraits<tData>::Name() << endl;
		convolution.generateCode();
	}
}

int main( int argc, char **argv )
{
	srand( time(NULL) );

	test_convolution<U_INT1>();
	//test_convolution<U_INT2>();
	//test_convolution<REAL4>();

	return EXIT_SUCCESS;
}
