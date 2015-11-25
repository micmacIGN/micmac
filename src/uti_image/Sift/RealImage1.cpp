#include "Sift.h" // for Real_ definition

#include <iostream>
#include <cstring>
#include <fstream>
#include <algorithm>
#include <list>
#include <sstream>
#include <iomanip>
#include <limits.h>

#include "StdAfx.h"

#ifndef BYTE
    #define BYTE unsigned char
#endif

#ifndef UINT
    #define UINT unsigned int
#endif

using namespace std;

typedef RealImage1::Real PixReal;

void RealImage1::set( UINT i_width, UINT i_height, const BYTE *i_data )
{
    resize( i_width, i_height );
    UINT iPix = i_width*i_height;
    PixReal *itPix = m_data.data();
    const PixReal maxValue = (PixReal)255;
    while ( iPix-- )
        *itPix++ = PixReal( *i_data++ )/maxValue;
}

// return a downsampled copy of the image
// keeps one sample every 2^i_pacePower
// dim->dim/2^i_pacePower
void RealImage1::downsample( RealImage1 &o_res, UINT i_pacePower ) const
{
    int step = 1<<i_pacePower;
    UINT newWidth  = ( m_width>>i_pacePower ),
         newHeight = ( m_height>>i_pacePower );
    o_res.resize( newWidth, newHeight );

    PixReal *itDst = o_res.m_data.data();
    const PixReal *itSrc = m_data.data();
    UINT lineStep = m_width*( step-1 )-( newWidth*step-m_width ),
         x;
    while ( newHeight-- )
    {
        x = newWidth;
        while ( x-- )
        {
            *itDst++ = *itSrc;
            itSrc += step;
        }
        itSrc += lineStep;
    }
}

// return an upsampled copy of the image (dim->dim*2)
// new pixels are linearly interpolated
// dim->dim*2 if i_exactDouble is true (last line/column is duplicated)
// dim->dim*2-1 oherwise (more accurate)
void RealImage1::upsample( RealImage1 &o_res, bool i_exactDouble ) const
{
    #ifdef _DEBUG
        if ( ( m_width<=1 ) || ( m_height<=1 ) )
            cerr << "WARN: RealImage1::upsample trying to process an image of size : " << m_width << 'x' << m_height << endl;
    #endif

    if ( i_exactDouble )
        o_res.resize( m_width<<1, m_height<<1 );
    else
        o_res.resize( (m_width<<1)-1, (m_height<<1)-1 );

    // fist pass : process columns
    PixReal *itDst = o_res.m_data.data();
    const PixReal *itSrc = m_data.data();
    PixReal a, b=0;
    UINT y = m_height, x;
    while ( y-- )
    {
        a = *itSrc++;
        x = m_width-1;
        while ( x-- )
        {
            b = *itSrc++;
            *itDst++ = a;
            *itDst++ = (PixReal)( ( a+b )*0.5 );
            a = b;
        }
        *itDst++ = b;
        if ( i_exactDouble ) *itDst++ = b;
        itDst += o_res.m_width; // skip a line, it will be filled in the second pass
    }

    // second pass : process lines
    PixReal *itSrc_prev = o_res.m_data.data();
    itDst = itSrc_prev+o_res.m_width;
    PixReal *itSrc_next = itDst+o_res.m_width;
    y = m_height-1;
    while ( y-- )
    {
        x = o_res.m_width;
        while ( x-- )
            *itDst++ = (PixReal)( ( ( *itSrc_prev++ )+( *itSrc_next++ ) )*0.5 );
        itSrc_prev = itDst;
        itDst = itSrc_next;
        itSrc_next += o_res.m_width;
    }
    if ( i_exactDouble )
        memcpy( itDst, itSrc_prev, o_res.m_width*sizeof(PixReal) );
}

// *this is set to i_a-i_b
// i_a and i_b must have the same size
void RealImage1::difference( const RealImage1 &i_a, const RealImage1 &i_b )
{
    resize( i_a.m_width, i_a.m_height );
    const PixReal *itA = i_a.m_data.data(),
               *itB = i_b.m_data.data();
    PixReal *itPix = m_data.data();
    UINT iPix = m_width*m_height;
    while ( iPix-- )
        *itPix++ = (*itA++) - (*itB++);
}

// same as above but with absolute value of the difference
void RealImage1::absoluteDifference( const RealImage1 &i_a, const RealImage1 &i_b )
{
    #ifdef _DEBUG
        if ( i_a.m_width!=i_b.m_width || i_a.m_height!=i_b.m_height ){
            cerr << "ERROR: RealImage1::difference: i_a and i_b must have the same size: a=" << i_a.m_width << 'x' << i_a.m_height << " b=" << i_b.m_width << 'x' << i_b.m_height << endl;
            return;
        }
    #endif

    resize( i_a.m_width, i_a.m_height );
    const PixReal *itA = data(),
               *itB = i_b.data();
    PixReal *itPix = m_data.data(),
         diff;
    UINT iPix = m_width*m_height;
    while ( iPix-- ){
        diff = ( *itA++ )-( *itB++ );

        *itPix++ = ( diff<0?-diff:diff );
    }
}

// this method is just for a checking purpose
PixReal RealImage1::differenceAccumulation( const RealImage1 &i_b ) const
{
    const PixReal *itA = data(),
                  *itB = i_b.data();
    PixReal diff;
    UINT    minWidth  = min( m_width, i_b.m_width ),
            minHeight = min( m_height, i_b.m_height ),
            x, y;
    PixReal accu = 0;
    for ( y=0; y<minHeight; y++ )
    {
        itA = data()+y*m_width;
        itB = i_b.data()+y*i_b.m_width;
        for ( x=0; x<minWidth; x++ )
        {
            diff = ( *itA++ )-( *itB++ );

            if ( diff<0 )
                accu-=diff;
            else
                accu+=diff;
        }
    }
    return accu;
}

// *this becomes the difference i_a-i_b
// i_a and i_b must be of the same size
UINT RealImage1::binaryDifference( const RealImage1 &i_a, const RealImage1 &i_b )
{
    #ifdef _DEBUG
        if ( i_a.m_width!=i_b.m_width || i_a.m_height!=i_b.m_height ){
            cerr << "ERROR: RealImage1::binaryDifference: i_a and i_b must have the same size: a=" << i_a.m_width << 'x' << i_a.m_height << " b=" << i_b.m_width << 'x' << i_b.m_height << endl;
            return 0;
        }
    #endif

    resize( i_a.m_width, i_a.m_height );
    const PixReal *itA = i_a.data(),
                  *itB = i_b.data();
    PixReal *itPix = m_data.data();
    UINT iPix  = m_width*m_height,
         count = 0;
    while ( iPix-- ){
        if ( ( *itA++ )==( *itB++ ) )
            *itPix++ = 0;
        else{
            *itPix++ = 1;
            count++;
        }
    }
    return count;
}

// set every non-null value to 1
void RealImage1::binarize()
{
    PixReal *itPix = m_data.data();
    UINT iPix = m_width*m_height;
    while ( iPix-- ){
        if ( ( *itPix )!=0 ){ cout << "diff" << endl; *itPix=(PixReal)1; }
        itPix++;
    }
}

bool RealImage1::loadRaw( const string &i_filename )
{
    ifstream f( i_filename.c_str(), std::ios::binary );

    if ( !f ){
        #ifdef _DEBUG
            cerr << "RealImage1::loadRaw: unable to open file " << i_filename << endl;
        #endif
        return false;
    }

    U_INT2 width, height;
    f.read( (char*)&width, 2 );
    f.read( (char*)&height, 2 );
    resize( width, height );
    f.read( (char*)m_data.data(), m_width*m_height*sizeof( PixReal ) );

    return true;
}

bool RealImage1::saveRaw( const string &i_filename ) const
{
    ofstream f( i_filename.c_str(), std::ios::binary );

    if ( !f ) return false;

    U_INT2 width = (U_INT2)m_width,
	   heigth = (U_INT2)m_height;
    f.write( (char*)&width, 2 );
    f.write( (char*)&heigth, 2 );
    f.write( (char*)m_data.data(), m_width*m_height*sizeof( PixReal ) );

    return true;
}

// draw an image inside another image
// no clipping is done, user must ensure i_image will fit
void RealImage1::draw( int i_x0, int i_y0, const RealImage1 &i_image )
{
    #ifdef _DEBUG
        int x1 = i_x0+i_image.m_width-1,
            y1 = i_y0+i_image.m_height-1;
        if ( i_x0<0 || i_x0>=(int)m_width  ||
             i_y0<0 || i_y0>=(int)m_height ||
             x1<0   || x1>=(int)m_width  ||
             y1<0   || y1>=(int)m_height )
        {
            std::cerr << "ERROR: ImageReal::draw : trying to draw an image that does not fit entirely : " << std::endl;
            std::cerr << "\t canvas    : " << m_width << 'x' << m_height << std::endl;
            std::cerr << "\t thumbnail : " << i_x0 << ',' << i_y0 << ' ' << i_image.m_width << 'x' << i_image.m_height  << std::endl;
            return;
        }
    #endif

    int W = m_width,
        w = i_image.m_width;
    const PixReal *itSrcLine = i_image.m_data.data();
          PixReal *itDstLine = m_data.data()+i_x0+i_y0*W;
    int j = i_image.height();
    while ( j-- )
    {
        memcpy( itDstLine, itSrcLine, w*sizeof(PixReal) );
        itSrcLine += w;
        itDstLine += W;
    }
}

#ifdef _DEBUG
void is_inside( const string s, const PixReal *ptr, const vector<PixReal> &v ){
    if ( ptr<v.data() ){
        cerr << s << ": before range" << endl;
        exit( 0 );
    }
    if ( ptr>=( v.data()+v.size() ) ){
        cerr << s << ": after range" << endl;
        exit( 0 );
    }
}
#endif

inline int __skip_comments( istream &in )
{
    int commentsRead = 0;
    string str;
    while ( true )
    {
        if ( in.peek()!=35 ) return commentsRead; // next line does not begin by a '#'
        getline( in, str );
        //cout << "skipping ------> " << str << endl;
        commentsRead++;
    }
}

inline bool __read_uint( istream &in, UINT &o_int )
{
    int c;
    UINT iNextChar = 0;
    vector<char> str(10);
    while ( true )
    {
        c = in.peek();
        if ( c<48 || c>57 ) break;
        str[iNextChar++] = in.get();
        if ( iNextChar>=str.size() ) str.resize( str.size()+10 );
    }
    if ( iNextChar==0 ) return false;
    str[iNextChar++]='\0';
    o_int = atoi( str.data() );
    return true;
}

// reads only first image if it's a multi-image PGM
// values are scale between 0 and 1
// handle only raw format
bool RealImage1::loadPGM( const std::string &i_filename )
{
    ifstream f( i_filename.c_str(), ios::binary );
    if ( !f ){ cerr << "ImageTarga::loadPGM: unable to open file" << endl; return false; }

    int c, nbComments;
    string str;
    UINT width=0, height=0, maxValue=0;
    /*list<string> comments;
    read_comments( f, comments );*/
    __skip_comments( f );
    if ( f.get()!='P' ){ cerr << "ImageTarga::loadPGM: unable to read P" << endl; return false; }
    if ( f.get()!='5' ){ cerr << "ImageTarga::loadPGM: unable to read 5" << endl; return false; }
    f.get(); // separator character
    // read width
    nbComments = __skip_comments( f );
    if ( !__read_uint( f, width ) ){ cerr << "ImageTarga::loadPGM: unable to read width" << endl; return false; }
    // read height
    nbComments = __skip_comments( f );
    c = f.peek();
    if ( c==32 || c==9 || c==13 || c==10 ) f.get();
    else if ( nbComments==0 ){ cerr << "ImageTarga::loadPGM: unable to find a whitespace after width" << endl; return false; }
    if ( !__read_uint( f, height ) ){ cerr << "ImageTarga::loadPGM: unable to read height" << endl; return false; }
    // read maxvalue
    nbComments = __skip_comments( f );
    c = f.peek();
    if ( c==32 || c==9 || c==13 || c==10 ) f.get();
    else if ( nbComments==0 ){ cerr << "ImageTarga::loadPGM: unable to find a whitespace after height" << endl; return false; }
    if ( !__read_uint( f, maxValue ) ){ cerr << "ImageTarga::loadPGM: unable to read max value" << endl; return false; }
    f.get();

    // read data
    resize( width, height );

    unsigned int i = m_width*m_height;
    unsigned char *buffer = new unsigned char[i];
    f.read( (char*)buffer, i );

    unsigned char *itBuffer = buffer;
    PixReal       *itData   = m_data.data();
    const PixReal maxValue_r = 255;
    while ( i-- )
         ( *itData++ ) = PixReal( *itBuffer++ )/maxValue_r;
    delete [] buffer;

    if ( f.bad() || f.fail() ){ cerr << "ImageTarga::loadPGM: unable to read all pixel data" << endl; return false; }
    return true;
}

// write file in PGM raw format
// values are considered to be between 0 and 1
// handles only one-channel images
bool RealImage1::savePGM( const std::string &i_filename, bool i_adaptDynamic ) const
{
    ofstream f( i_filename.c_str(), ios::binary );
    if ( !f ) return false;

    const char LF_char = 10;
    char str[10];
    f.put( 'P' ); f.put( '5' ); f.put( LF_char );
    // write width
    sprintf( str, "%u\n", m_width );
    f.write( str, strlen(str) );
    // write height
    sprintf( str, "%u\n", m_height );
    f.write( str, strlen(str) );
    // write maxvalue
    sprintf( str, "%u\n", 255 );
    f.write( str, strlen(str) );
    // write data

    if ( m_width==0 || m_height==0 ) return true;
    unsigned int i;
    const PixReal *itData;
    PixReal img_min = 0, img_max=1;
    if ( i_adaptDynamic )
    {
       i = m_width*m_height;
       itData = m_data.data();
       img_min = img_max = *itData;
       while ( i-- )
       {
	   if ( *itData>img_max ) img_max=*itData;
	   if ( *itData<img_min ) img_min=*itData;
	   itData++;
       }
    }
    
    i = m_width*m_height;
    unsigned char *buffer   = new unsigned char[i],
                  *itBuffer = buffer;
    const PixReal maxValue = 255/(img_max-img_min);
    itData = m_data.data();
    while ( i-- )
        *itBuffer++ = ( unsigned char )( ( ( *itData++ )-img_min )*maxValue );

    f.write( (char*)buffer, m_width*m_height );
    delete [] buffer;

    return true;
}

// image is normalized between 0 and 1 using i_max as i_src's max value
template <class tData, class tBase>
void copyNormalized( Im2D<tData,tBase> &i_src, const PixReal i_max, RealImage1 &o_dst )
{
    o_dst.resize( i_src.sz().x, i_src.sz().y );

    U_INT iPix = o_dst.width()*o_dst.height();
    tData *itSrc = i_src.data_lin();
    PixReal *itDst = o_dst.data();
    while ( iPix-- )
        *itDst++ = (*itSrc++)/i_max;
}

// image is normalized between 0 and 1 using i_max as i_src's max value
template <class tData, class tBase>
void copyNormalized( Im2D<tData,tBase> &i_src, RealImage1 &o_dst )
{
    o_dst.resize( i_src.sz().x, i_src.sz().y );

    const U_INT nbPix = o_dst.width()*o_dst.height();
    if ( nbPix==0 ) return;

    tData *itSrc = i_src.data_lin();
    tData minv = itSrc[0], maxv = minv;
    U_INT iPix = nbPix;
    while ( iPix-- )
    {
        tData v = *itSrc++;
        if ( v<minv ) minv = v;
        if ( v>maxv ) maxv = v;
    }

    PixReal *itDst = o_dst.data();
    const PixReal scale = 1/( (PixReal)maxv-(PixReal)minv );
    itSrc = i_src.data_lin();
    iPix = nbPix;
    while ( iPix-- )
        *itDst++ = ( (*itSrc++)-minv )*scale;
}

bool RealImage1::load( const std::string &i_filename )
{
    Tiff_Im tiffHeader = Tiff_Im::BasicConvStd( i_filename.c_str() );
    Im2DGen im2d = tiffHeader.ReadIm();

    if ( tiffHeader.nb_chan()!=1 ){
        cerr << "RealImage1::load : invalid number of channels : " << tiffHeader.nb_chan() << endl;
        return false;
    }

    switch ( tiffHeader.type_el() )
    {
    case GenIm::u_int1: copyNormalized( *(Im2D<U_INT1,INT>*)&im2d, *this ); break;
    case GenIm::u_int2: copyNormalized( *(Im2D<U_INT2,INT>*)&im2d, *this ); break;
    default:
        cerr << "RealImage1::load : unhandled image base type" << endl;
        return false;
    }
    return true;
}

// convolutes *this by the 1d kernel i_kernel and store the transposed result in o_res
// the size of i_kernel is considered to of the form 2n+1, element n being the center of the filter
void RealImage1::convolution_transpose_1d( const std::vector<PixReal> &i_kernel, RealImage1 &o_res )
{
    o_res.resize( m_height, m_width );

    UINT n = (UINT)i_kernel.size();
    if ( n<3 || ( (n&1)==0 ) )
    {
        #ifdef _DEBUG
            cerr << "RealImage1::convolution_transpose_1d: invalid kernel size : " << n << endl;
        #endif
        return;
    }

    n = ( n-1 )/2;
    // process the n first columns where some elements of the kernel are out of image on the left (and the right if the kernel is bigger than the image is wide)
    UINT nbColumns = std::min( n, m_width ),
         iLine, iColumn, iKernelElement, nbValidElements;
    int srcStep;
    const PixReal *itKernelFirst = i_kernel.data()+n,
                  *itSrcKernel;
    PixReal *itDst = o_res.data(),
            *itSrc;
    PixReal  accum;
    for ( iColumn=0; iColumn<nbColumns; iColumn++ )
    {
        itSrc           = m_data.data();
        nbValidElements = std::min( iColumn+n+1, m_width );
        srcStep         = m_width-nbValidElements;
        iLine           = m_height;
        while ( iLine-- )
        {
            accum       = 0;
            itSrcKernel = itKernelFirst;
            iKernelElement = nbValidElements;
            while ( iKernelElement-- )
                accum += ( *itSrc++ )*( *itSrcKernel++ );

            *itDst++ = accum;
            itSrc += srcStep;
        }
        itKernelFirst--;
    }

    if ( m_width<=n ) return;
    // process full-kernel computed pixels
    UINT srcStepLine = 2*n;
    PixReal *itFirstDst = o_res.data()+n*m_height;
    nbValidElements = (UINT)i_kernel.size();
    itKernelFirst   = i_kernel.data();
    srcStep         = -nbValidElements+1;
    itSrc           = m_data.data();
    for ( iLine=0; iLine<m_height; iLine++ )
    {
        itDst          = itFirstDst+iLine;
        iColumn = m_width-2*n;
        while ( iColumn-- )
        {
            accum          = 0;
            itSrcKernel    = itKernelFirst;
            iKernelElement = nbValidElements;
            while ( iKernelElement-- )
                accum += ( *itSrc++ )*( *itSrcKernel++ );
            *itDst = accum;

            itDst += m_height; // which is o_res.m_width
            itSrc += srcStep;
        }
        itSrc += srcStepLine;
    }

    if ( m_width<=n+1 ) return;
    // process the n last columns where some elements of the kernel are out of image on the right
    nbValidElements = 2*n;
    srcStep         = m_width-nbValidElements;
    itKernelFirst   = i_kernel.data();
    itDst           = o_res.data()+m_height*( m_width-n );
    iColumn         = n;
    while ( iColumn-- )
    {
        itSrc = m_data.data()+m_width-nbValidElements;
        iLine = m_height;
        while ( iLine-- )
        {
            accum       = 0;
            itSrcKernel = itKernelFirst;
            iKernelElement = nbValidElements;
            while ( iKernelElement-- )
                accum += ( *itSrc++ )*( *itSrcKernel++ );

            *itDst++ = accum;
            itSrc += srcStep;
        }
        nbValidElements--;
        srcStep++;
    }
}

// convolutes *this by the 1d kernel i_kernel and store the transposed result in o_res
// the size of i_kernel is considered to of the form 2n+1, element n being the center of the filter
void RealImage1::convolution_transpose_1d_2( const std::vector<PixReal> &i_kernel, RealImage1 &o_res )
{
    o_res.resize( m_height, m_width );

    int N = (int)i_kernel.size(),
        n = ( N-1 )/2;
    if ( N<3 || ( (N&1)==0 ) )
    {
        #ifdef _DEBUG
            cerr << "RealImage1::convolution_transpose_1d_2: invalid kernel size : " << n << endl;
        #endif
        return;
    }

    UINT iLine, iColumn, iKernelElement;
    int nbValidElements, nbBeforeElements, nbAfterElements;
    int srcStep = -m_width*m_height;
    const PixReal *itSrcKernel;
    PixReal *itDst = o_res.data(),
         *itBaseSrc = m_data.data(),
         *itSrc;
    PixReal accum, v;

    for ( iColumn=0; iColumn<m_width; iColumn++ )
    {
        nbBeforeElements = std::max( 0, n-(int)iColumn );
        nbAfterElements  = std::max( 0, (int)iColumn+n+1-(int)m_width );
        nbValidElements  = N-nbBeforeElements-nbAfterElements;

        iLine = m_height;
        while ( iLine-- )
        {
            accum = 0;
            v = *( itSrc = itBaseSrc );
            itSrcKernel = i_kernel.data();

            iKernelElement = nbBeforeElements;
            while ( iKernelElement-- )
                accum += v*( *itSrcKernel++ );

            iKernelElement = nbValidElements;
            while ( iKernelElement-- )
                accum += ( *itSrc++ )*( *itSrcKernel++ );

            v = itSrc[-1];
            iKernelElement = nbAfterElements;
            while ( iKernelElement-- )
                accum += v*( *itSrcKernel++ );

            *itDst++ = accum;
            itBaseSrc += m_width;
        }
        itBaseSrc += srcStep;
        if ( nbBeforeElements==0 ) itBaseSrc++;
    }
}

void RealImage1::convolution_transpose_1d_3( const std::vector<PixReal> &i_kernel, RealImage1 &o_res )
{
    o_res.resize( m_height, m_width );

    // convolve along columns, save transpose
    // image is M by N
    // buffer is N by M
    // filter is (2*W+1) by 1
    const int N = (int)m_height,
              M = (int)m_width,
              W = (int)((i_kernel.size() - 1) /  2);
    const PixReal *filter_pt = i_kernel.data(),
                  *src_pt    = m_data.data();
    PixReal *dst_pt = o_res.m_data.data();
    for ( int j=0; j<N; ++j )
    {
        for ( int i=0; i<M; ++i )
        {
            PixReal acc = 0.0 ;
            const PixReal *g = filter_pt,
                          *start = src_pt + (i-W),
                          *stop;
            PixReal x;

            // beginning
            stop = src_pt ;
            x    = *stop ;
            while( start <= stop ) { acc += (*g++) * x ; start++ ; }

            // middle
            stop =  src_pt + std::min(M-1, i+W) ;
            while( start <  stop ) acc += (*g++) * (*start++) ;

            // end
            x  = *start ;
            stop = src_pt + (i+W) ;
            while( start <= stop ) { acc += (*g++) * x ; start++ ; }

            // save
            *dst_pt = acc ;
            dst_pt += N ;

            //assert( g - filter_pt == 2*W+1 ) ;
        }
        // next column
        src_pt += M ;
        dst_pt -= M*N - 1 ;
    }
}

//#define __DEBUG_OUTPUT_KERNELS

#ifdef __DEBUG_OUTPUT_KERNELS
	extern string __kernel_output_filename;
#endif

void RealImage1::gaussianFilter( Real_ i_standardDeviation, RealImage1 &o_res )
{
    static RealImage1 tmp_img;
    static vector<PixReal> kernel;

    createGaussianKernel_1d( i_standardDeviation, kernel );

	#ifdef __DEBUG_OUTPUT_KERNELS
		{
			ofstream f( __kernel_output_filename.c_str(), ios::binary|ios::app );
			// siftpp type
			f.put(1);
			// type name
			string typeName = El_CTypeTraits<PixReal>::Name();
			U_INT4 ui4 = (U_INT4)typeName.length();
			f.write( (char*)&ui4, 4 );
			f.write( typeName.c_str(), ui4 );
			// sigma
			REAL8 r8 = (REAL8)i_standardDeviation;
			f.write( (char*)&r8, 8 );
			// nb coefficients
			ui4 = (U_INT4)kernel.size();
			f.write( (char*)&ui4, 4 );
			// REAL8 coefficients
			for ( size_t i=0; i<kernel.size(); i++ ){
				double d = (double)kernel[i];
				f.write( (char*)(&d), 8 );
			}
		}
	#endif

    tmp_img.resize( m_width, m_height );
    convolution_transpose_1d_3( kernel, tmp_img );
    tmp_img.convolution_transpose_1d_3( kernel, o_res );
}

// set all data values to i_value
void RealImage1::set( PixReal i_value )
{
    PixReal *it = m_data.data();
    UINT i = m_width*m_height;
    while ( i-- )
        *it++ = i_value;
}

// copy i_b into *this
void RealImage1::copy( RealImage1 &i_b ) const
{
    i_b.resize( m_width, m_height );
    memcpy( i_b.m_data.data(), m_data.data(), m_width*m_height*sizeof(PixReal) );
}


// return the gradient a the image splited in two components : modulus and angle
// o_gradient is twice as wide as *this since there are two values for each pixel
// a border of 1 pixel is left undefined
void RealImage1::gradient( RealImage1 &o_gradient ) const
{
    o_gradient.resize( m_width*2, m_height );

    const int c1 = -m_width;
    int offset = m_width+1;
    const PixReal *src = m_data.data()+offset;
    PixReal *dst = o_gradient.m_data.data()+2*offset;
    Real_ gx, gy;
    int width_2 = m_width-2,
        y = m_height-2,
        x;
    while ( y-- )
    {
        x = width_2;
        while ( x-- )
        {
            gx = ( Real_ )( 0.5*( src[1]-src[-1] ) );
            gy = ( Real_ )( 0.5*( src[m_width]-src[c1] ) );
            //dst[0] = std::sqrt( gx*gx+gy*gy ); // __FAST_MATH
            dst[0] = ( PixReal )fast_maths::fast_sqrt( gx*gx+gy*gy ); // __FAST_MATH
            /*
            __FAST_MATH
            theta = std::atan2( gy, gx )+ PixReal( 2*M_PI );
            theta = std::fmod( theta, PixReal( 2*M_PI ) );
            if ( theta<0 ) theta+=2*M_PI;
            dst[1] = theta;
            */
            dst[1] = ( PixReal )fast_maths::fast_mod_2pi( fast_maths::fast_atan2( gy, gx ) + Real_( 2*M_PI ) );

            src++; dst+=2;
        }
        src+=2; dst+=4;
    }
}

void RealImage1::setHorizontalBorderToValue( int i_borderSize, PixReal i_value )
{
    PixReal *row = new PixReal[m_width],
         *it = row;

    // create a row filled with i_value
    int i=m_width;
    while ( i-- ) *it++=i_value;

    // copy that row to the i_borderSize-th first and last rows
    it = m_data.data();
    for ( i=0; i<i_borderSize; i++ )
    {
        memcpy( it+i*m_width, row, m_width*sizeof( PixReal ) );
        memcpy( it+( m_height-1-i )*m_width, row, m_width*sizeof( PixReal ) );
    }

    delete [] row;
}

void RealImage1::setVerticalBorderToValue( int i_borderSize, PixReal i_value )
{
    PixReal *it;
    PixReal *itEnd0 = m_data.data()+m_width-1,
         *itEnd;
    int i, j;

    // fill i_borderSize-th first and last columns
    for ( j=0; j<i_borderSize; j++ )
    {
        it = m_data.data()+j;
        itEnd = itEnd0-j;
        i = m_height;
        while ( i-- ){
            *it = *itEnd = i_value;
            it+=m_width; itEnd+=m_width;
        }
    }
}

// retrieve a sub part of an image
void RealImage1::getWindow( const RoiWindow_2d &i_window, RealImage1 &o_image ) const
{
    #ifdef _DEBUG
        if ( i_window.m_x0<0 ||
             i_window.m_y0<0 ||
             i_window.m_x1>=(int)width() ||
             i_window.m_y1>=(int)height() )
             cerr << "ERRROR: getWindow(): window out of image's range" << endl;
        if ( i_window.m_x0>=i_window.m_x1 ||
             i_window.m_y0>=i_window.m_y1 )
             cerr << "ERROR: getWindow(): inconsistent window " << i_window.m_x0 << ',' << i_window.m_y0 << ' '
                                                                << i_window.m_x1 << ',' << i_window.m_y1 << endl;
    #endif
    o_image.resize( i_window.m_x1-i_window.m_x0+1, i_window.m_y1-i_window.m_y0+1 );
    const PixReal *itSrc = m_data.data()+i_window.m_y0*m_width+i_window.m_x0;
    PixReal *itDst = o_image.m_data.data();
    int j = o_image.m_height,
        srcOffset = m_width-o_image.m_width,
        i;
    while ( j-- )
    {
        i = o_image.m_width;
        while ( i-- )
            ( *itDst++ ) = ( *itSrc++ );
        itSrc += srcOffset;
    }
}

// retrieve a sub part of an image
void RealImage1::drawWindow( int i_x, int i_y, const RealImage1 &i_image )
{
    #ifdef _DEBUG
        if ( i_x<0 || i_y<0 ||
             i_x>=(int)m_width || i_y>=(int)m_height )
             cerr << "ERRROR: drawWindow(): anchor (" << i_x << ',' << i_y << ") out of image's range " << m_width << 'x' << m_height << endl;
    #endif
    const PixReal *itSrc = i_image.m_data.data();
    PixReal *itDst = m_data.data()+i_y*m_width+i_x;
    int j = i_image.m_height,
        dstOffset = m_width-i_image.m_width,
        i;
    while ( j-- )
    {
        i = i_image.m_width;
        while ( i-- )
            ( *itDst++ ) = ( *itSrc++ );
        itDst += dstOffset;
    }
}

// RealImage1-related functions

Real_ loadKernel( const string &i_filename, vector<PixReal> &o_ker )
{
    ifstream f( i_filename.c_str(), ios::binary );

    #ifdef _DEBUG
        if ( !f ) cerr << "ERROR: loadKernel( " << i_filename << "): unable to open file" << endl;
    #endif

    int n, N;
    Real_ s;
    f.read( (char*)&s, sizeof( Real_ ) );
    f.read( (char*)&n, sizeof( int ) );

    N = 2*n+1;
    o_ker.resize( N );
    for ( int i=0; i<N; i++ )
        f.read( (char*)( &(o_ker[i]) ), sizeof( PixReal ) );
    return s;
}

Real_ saveKernel( const string &i_filename, Real_ s, const vector<PixReal> &o_ker )
{
    ofstream f( i_filename.c_str(), ios::binary );

    #ifdef _DEBUG
        if ( !f ) cerr << "ERROR: saveKernel( " << i_filename << "): unable to open file" << endl;
    #endif

    int N = (int)o_ker.size(),
        n = ( N-1 )/2;
    f.write( (char*)&s, sizeof( Real_ ) );
    f.write( (char*)&n, sizeof( int ) );
    for ( int i=0; i<N; i++ )
        f.write( (const char*)( &(o_ker[i]) ), sizeof( PixReal ) );
    return s;
}

bool compare_kernels( const vector<PixReal> i_ker0, const vector<PixReal> i_ker1 )
{
    if ( i_ker0.size()!=i_ker1.size() ){
        cout << "sizes: " << i_ker0.size() << " != " << i_ker1.size() << endl;
        return false;
    }
    bool ok = true;
    for ( unsigned int i=0; i<i_ker0.size(); i++ )
        if ( i_ker0[i]!=i_ker1[i] ){
            cout << "element " << i << ": " << i_ker0[i] << " != " << i_ker1[i] << endl;
            ok = false;
        }
    return ok;
}

void createGaussianKernel_1d( Real_ i_standardDeviation, vector<PixReal> &o_kernel )
{
    int n = getGaussianKernel_halfsize( i_standardDeviation ),
        N = 2*n+1;

    o_kernel.resize( N );
    PixReal accum = 0;
    for ( int i=0; i<N; i++ )
        #ifdef __ORIGINAL__
            accum += o_kernel[i] = PixReal( std::exp( PixReal( -0.5 )*( i-n )*( i-n )/( PixReal( i_standardDeviation )*PixReal( i_standardDeviation ) ) ) );
        #else
            accum += o_kernel[i] = PixReal( std::exp( Real_( -0.5*( i-n )*( i-n )/( i_standardDeviation*i_standardDeviation ) ) ) );
        #endif

    if ( accum==0 ){
        #ifdef _DEBUG
            cerr << "WARN: createGaussianKernel_1d( " << i_standardDeviation << ") unable to divide by a null norm" << endl;
        #endif
         return;
    }

    for ( int i=0; i<N; i++ )
        o_kernel[i] /= accum;
}

void clusterize_1d( int i_areaSize, int i_efficientSize, int i_overlap, std::vector<RoiWindow_1d> &o_cluster )
{
    #ifdef _DEBUG
        if ( i_efficientSize<=i_overlap )
            cerr << "WARN: clusterize_1d: efficientSize=" << i_efficientSize << " <= overlap=" << i_overlap << endl;
    #endif

    if ( i_areaSize<=i_efficientSize )
    {
        o_cluster.clear();
        o_cluster.push_back( RoiWindow_1d( 0, i_areaSize-1, 0, i_areaSize-1 ) );
        return;
    }
    int nbFullLength = i_areaSize/i_efficientSize,
        remaining    = i_areaSize%i_efficientSize;
    int nbWindows = ( remaining==0?nbFullLength:nbFullLength+1 );
    o_cluster.resize( nbWindows );
    RoiWindow_1d *itROI = o_cluster.data();
    // first window
    int i = std::min( i_efficientSize+i_overlap, i_areaSize ); // clip overlaping zone on the right if needed
    ( itROI++ )->set( 0, i-1, 0, i_efficientSize-1 );
    // middle windows (with overlapping on each side)
    int x0 = i_efficientSize-i_overlap;
    if ( nbWindows>2 )
    {
        int x1 = x0+i_efficientSize+2*i_overlap-1,
            roi_x1 = i_overlap+i_efficientSize-1;
        i = nbWindows-2;
        while ( i-- )
        {
            ( itROI++ )->set( x0, x1, i_overlap, roi_x1 );
            x0+=i_efficientSize; x1+=i_efficientSize;
        }
        // the last middle window's right overlapping zone might out-range the area, clip it then
        if ( remaining<i_overlap )
            itROI[-1].m_x1=i_areaSize-1;
    }
    // last window
    ( itROI++ )->set( x0, i_areaSize-1, i_overlap, i_overlap+remaining-1 );
}

void clusterize_2d( const ImageSize &i_areaSize, const ImageSize &i_efficientSize, const ImageSize &i_overlap, std::vector<RoiWindow_2d> &o_cluster )
{
    vector<RoiWindow_1d> clusterX, clusterY;
    clusterize_1d( i_areaSize.width(), i_efficientSize.width(), i_overlap.width(), clusterX );
    clusterize_1d( i_areaSize.height(), i_efficientSize.height(), i_overlap.height(), clusterY );

    o_cluster.resize( clusterX.size()*clusterY.size() );
    vector<RoiWindow_2d>::iterator it2d = o_cluster.begin();
    vector<RoiWindow_1d>::iterator itX = clusterX.begin(),
                                   itY;
    int x = (int)clusterX.size(),
        y;
    while ( x-- )
    {
        itY = clusterY.begin();
        y = (int)clusterY.size();
        while ( y-- )
        {
            it2d->set_along_y( *itY++ );
            ( it2d++ )->set_along_x( *itX );
        }
        itX++;
    }
}

// check i_win covers all i_img but not more
bool check_grid( const RealImage1 &i_img, const vector<RoiWindow_2d> &i_win )
{
    vector<RealImage1> subs( i_win.size() );

    // get all window
    int xmin = INT_MAX, xmax = INT_MIN,
        ymin = INT_MAX, ymax = INT_MIN;
    vector<RoiWindow_2d>::const_iterator itWin = i_win.begin();
    vector<RealImage1>::iterator itSub = subs.begin();
    int i = (int)i_win.size();
    while ( i-- ){
        if ( itWin->m_x0<xmin ) xmin=itWin->m_x0;
        if ( itWin->m_y0<ymin ) ymin=itWin->m_y0;
        if ( itWin->m_x1>xmax ) xmax=itWin->m_x1;
        if ( itWin->m_y1>ymax ) ymax=itWin->m_y1;
        i_img.getWindow( ( *itWin++ ), ( *itSub++ ) );
    }

    RealImage1 reconsImg( xmax-xmin+1, ymax-ymin+1 );
    if ( reconsImg.width()!=i_img.width() ||
         reconsImg.height()!=i_img.height() ){
         cout << "check_grid: different sizes, original=" << i_img.width() << 'x' << i_img.height() << " recons=" << reconsImg.width() << 'x' << reconsImg.height() << endl;
         return false;
    }

    // reconstruct image
    int nbTotalPix = 0;
    itWin = i_win.begin();
    itSub = subs.begin();
    i = (int)i_win.size();
    while ( i-- ){
        reconsImg.drawWindow( itWin->m_x0, itWin->m_y0, *itSub );
        nbTotalPix += itSub->width()*itSub->height();
        itWin++; itSub++;
    }

    if ( nbTotalPix!=(int)( i_img.width()*i_img.height() ) )
        cout << "check_grid: different number of pixels original=" << i_img.width()*i_img.height() << " recons=" << nbTotalPix << endl;

    // get a value of the difference between original and reconstructed images
    Real_ diff=i_img.differenceAccumulation( reconsImg );
    if ( diff==0 ){
        cout << "check_grid: OK" << endl;
        return true;
    }
    cout << "check_grid: difference accumulation = " << diff << endl;
    return false;
}
