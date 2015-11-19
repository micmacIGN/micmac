#ifndef __REALIMAGE1__
#define __REALIMAGE1__

#include <cstddef>
#include <vector>
#include <string>
#include <cmath>
#include <float.h>
#include <string.h>

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

#include "fast_maths.h"

#ifndef BYTE
    #define BYTE unsigned char
#endif

#ifndef UINT
    #define UINT unsigned int
#endif

class ImageSize
{
private:
    int m_width, m_height;

public:
    ImageSize(){}

    ImageSize( int i_width, int i_height ){ set( i_width, i_height ); }

    void set( int i_width, int i_height ){
        setWidth( i_width );
        setHeight( i_height );
    }

    void setWidth( int i_width ){
        #ifdef _DEBUG
            if ( i_width<0 ) std::cerr << "WARN: ImageSize::setWidth : negative value = " << i_width << std::endl;
        #endif
        m_width = i_width;
    }

    void setHeight( int i_height ){
        #ifdef _DEBUG
            if ( i_height<0 ) std::cerr << "WARN: ImageSize::setHeight : negative value = " << i_height << std::endl;
        #endif
        m_height = i_height;
    }

    int width() const { return m_width; }

    int height() const { return m_height; }
};

inline std::ostream &operator <<( std::ostream &o, const ImageSize &s ){
    return ( o << s.width() << 'x' << s.height() );
}

// a window with a region of interest in one dimension
class RoiWindow_1d
{
public:
    // window boundaries
    int m_x0, m_x1;
    // ROI boundaries
    int m_roi_x0, m_roi_x1;

    RoiWindow_1d(){}
    inline RoiWindow_1d( int i_x0, int i_x1, int i_roi_x0, int i_roi_x1 );
    inline void set( int i_x0, int i_x1, int i_roi_x0, int i_roi_x1 );

    #ifdef _DEBUG
        bool check() const;
    #endif
};

inline std::ostream &operator <<( std::ostream &o, const RoiWindow_1d &w ){
    return ( o << w.m_x0 << ',' << w.m_x1 << ' ' << w.m_roi_x0 << ',' << w.m_roi_x1 );
}

// a window with a region of interest in two dimensions
class RoiWindow_2d
{
public:
    // along x
    int m_x0, m_x1;
    int m_roi_x0, m_roi_x1;
    // along y
    int m_y0, m_y1;
    int m_roi_y0, m_roi_y1;

    RoiWindow_2d(){}
    inline void set_along_x( const RoiWindow_1d &window1d );
    inline void set_along_y( const RoiWindow_1d &window1d );

    #ifdef _DEBUG
        bool check() const;
    #endif
};

inline std::ostream &operator <<( std::ostream &o, const RoiWindow_2d &w ){
    return ( o << w.m_x0 << ',' << w.m_y0 << ' ' << w.m_x1 << ',' << w.m_y1 << " - "
               << w.m_roi_x0 << ',' << w.m_roi_y0 << ' ' << w.m_roi_x1 << ',' << w.m_roi_y1 );
}

void clusterize_1d( int i_areaSize, int i_efficientSize, int i_overlap, std::vector<RoiWindow_1d> &o_cluster );

void clusterize_2d( const ImageSize &i_areaSize, const ImageSize &i_efficientSize, const ImageSize &i_overlap, std::vector<RoiWindow_2d> &o_cluster );

// a one-channel real image
// values are considered between 0 and 1
class RealImage1
{
public:
    typedef float Real;
private:
    std::vector<Real> m_data;
    UINT              m_width;
    UINT              m_height;

public:
    RealImage1( UINT i_width=0, UINT i_height=0 );

    void resize( UINT i_width, UINT i_height );
      
    void set( UINT i_width, UINT i_height, const BYTE *i_data );

    Real       * data();
    const Real * data()   const;
    UINT         width()  const;
    UINT         height() const;

    // return a downsampled copy of the image
    // dim->dim/2^i_pacePower
    void downsample( RealImage1 &o_res, UINT i_pacePower ) const;

    // return an upsampled copy of the image (dim->dim*2-1)
    // new pixels are linearly interpolated
    // dim->dim*2 if i_exactDouble is true (last line/column is duplicated)
    // dim->dim*2-1 oherwise (more accurate)
    void upsample( RealImage1 &o_res, bool i_exactDouble=true ) const;

    void gaussianFilter( Real_ i_standardDeviation, RealImage1 &o_res );
    void gaussianFilter( Real_ i_standardDeviation );

    // convolutes *this by the 1d kernel i_kernel and store the transposed result in o_res
    // the size of i_kernel is considered to of the form 2n+1, element n being the center of the filter
    void convolution_transpose_1d( const std::vector<Real> &i_kernel, RealImage1 &o_res );

    // convolutes *this by the 1d kernel i_kernel and store the transposed result in o_res
    // the size of i_kernel is considered to of the form 2n+1, element n being the center of the filter
    void convolution_transpose_1d_2( const std::vector<Real> &i_kernel, RealImage1 &o_res );

    void convolution_transpose_1d_3( const std::vector<Real> &i_kernel, RealImage1 &o_res );

    // *this is set to i_a-i_b
    // i_a and i_b must have the same size
    void difference( const RealImage1 &i_a, const RealImage1 &i_b );
    // same as above but with absolute value of the difference
    void absoluteDifference( const RealImage1 &i_a, const RealImage1 &i_b );

    // create a difference image for images of the same size, values are 0 or 1
    UINT binaryDifference( const RealImage1 &i_a, const RealImage1 &i_b );

    Real differenceAccumulation( const RealImage1 &i_b ) const;

    // return the gradient a the image splited in two components : modulus and angle
    // o_gradient is twice as wide as *this since there are two values for each pixel
    // a border of 1 pixel is left undefined
    void gradient( RealImage1 &o_gradient ) const;

    bool loadPGM( const std::string &i_filename );
    bool savePGM( const std::string &i_filename, bool i_adaptDynamic=false ) const;

    bool load( const std::string &i_filename );

    bool loadRaw( const std::string &i_filename );
    bool saveRaw( const std::string &i_filename ) const;

    // draw an image inside another image
    // no clipping is done, user must ensure i_image will fit
    void draw( int i_x0, int i_y0, const RealImage1 &i_image );

    void binarize();

    // set all values in a border of size i_borderSize to value i_value
    void setHorizontalBorderToValue( int i_borderSize, Real i_value );
    void setVerticalBorderToValue( int i_borderSize, Real i_value );
    void setBorderToValue( int i_borderSize, Real i_value );

    // set all data values to i_value
    void set( Real i_value );
    // copy *this into o_res
    void copy( RealImage1 &o_res ) const;
    // low level swap : data buffers and dimensions are swaped
    void swap( RealImage1 &o_res  );

    // retrieve a sub part of an image
    void getWindow( const RoiWindow_2d &i_window, RealImage1 &o_image ) const;
    // draw i_image at coordinates (i_x,i_y) into (*this)
    // no clipping is done, i_image must fit
    void drawWindow( int i_x, int i_y, const RealImage1 &i_image );

    template <class T>
    RealImage1( UINT i_width, UINT i_height, const std::vector<T> &i_data );

    template <class T>
    void setFromArray( UINT i_width, UINT i_height, const T *i_data );

    template <class T>
    void toVector( std::vector<T> &o_data ) const;

    template <class T>
    void toArray( T *o_data ) const;
};

inline int getGaussianKernel_halfsize( Real_ i_standardDeviation ){ return int( ceil( Real_(4.0)*i_standardDeviation ) ); }

void createGaussianKernel_1d( Real_ i_standardDeviation, std::vector<RealImage1::Real> &o_kernel );

Real_ loadKernel( const std::string &i_filename, std::vector<RealImage1::Real> &o_ker );
Real_ saveKernel( const std::string &i_filename, float s, const std::vector<RealImage1::Real> &o_ker );
bool compare_kernels( const std::vector<RealImage1::Real> &i_ker0, const std::vector<RealImage1::Real> &i_ker1 );

bool check_grid( const RealImage1 &i_img, const std::vector<RoiWindow_2d> &i_win );

inline RealImage1::RealImage1( UINT i_width, UINT i_height ){ resize( i_width, i_height ); }

inline void RealImage1::resize( UINT i_width, UINT i_height ){ m_width=i_width; m_height=i_height; m_data.resize( m_width*m_height ); }

inline RealImage1::Real       * RealImage1::data()         { return m_data.data(); }
inline RealImage1::Real const * RealImage1::data()   const { return m_data.data(); }
inline UINT                     RealImage1::width()  const { return m_width; }
inline UINT                     RealImage1::height() const { return m_height; }

inline void RealImage1::gaussianFilter( Real_ i_standardDeviation ){
    gaussianFilter( i_standardDeviation, *this );
}

// set all values in a border of size i_borderSize to value i_value
inline void RealImage1::setBorderToValue( int i_borderSize, RealImage1::Real i_value ){
    setVerticalBorderToValue( i_borderSize, i_value );
    setHorizontalBorderToValue( i_borderSize, i_value );
}

// low level swap : data buffers and dimensions are swaped
inline void RealImage1::swap( RealImage1 &o_res  ){
    std::swap( m_width, o_res.m_width );
    std::swap( m_height, o_res.m_height );
    m_data.swap( o_res.m_data );
}

// RoiWindow_1d

inline RoiWindow_1d::RoiWindow_1d( int i_x0, int i_x1, int i_roi_x0, int i_roi_x1 ):
    m_x0( i_x0 ), m_x1( i_x1 ),
    m_roi_x0( i_roi_x0 ), m_roi_x1( i_roi_x1 ){
    #ifdef _DEBUG
        check();
    #endif
}

inline void RoiWindow_1d::set( int i_x0, int i_x1, int i_roi_x0, int i_roi_x1 ){
    m_x0=i_x0; m_x1=i_x1;
    m_roi_x0=i_roi_x0; m_roi_x1=i_roi_x1;
    #ifdef _DEBUG
        check();
    #endif
}

#ifdef _DEBUG
    // check ROI's size <= window's size
    inline bool RoiWindow_1d::check() const{
        return ( m_roi_x1-m_roi_x0 )<=( m_x1-m_x0 );
    }
#endif

// RoiWindow_2d

inline void RoiWindow_2d::set_along_x( const RoiWindow_1d &window1d ){
    memcpy( &m_x0, &window1d.m_x0, 4*sizeof(int) );
}

inline void RoiWindow_2d::set_along_y( const RoiWindow_1d &window1d ){
    memcpy( &m_y0, &window1d.m_x0, 4*sizeof(int) );
}

template <class T>
RealImage1::RealImage1( UINT i_width, UINT i_height, const std::vector<T> &i_data )
{
   resize( i_width, i_height );
   size_t i = i_width*i_height;
   i = std::min( i_data.size(), i );
   const T *itSrc = i_data.data();
   Real *itDst = m_data.data();
   while ( i-- )
   *itDst++ = (Real)( *itSrc++ );
}

template <class T>
void RealImage1::setFromArray( UINT i_width, UINT i_height, const T *i_data )
{
   resize( i_width, i_height );
   size_t i = i_width*i_height;
   const T *itSrc = i_data;
   Real *itDst = m_data.data();
   while ( i-- )
   *itDst++ = (Real)( *itSrc++ );
}

template <class T>
void RealImage1::toVector( std::vector<T> &o_data ) const
{
   size_t i = m_width*m_height;
   i = std::min( o_data.size(), i );
   const Real *itSrc = m_data.data();
   T *itDst = o_data.data();
   while ( i-- )
   *itDst++ = (T)( *itSrc++ );
}

template <class T>
void RealImage1::toArray( T *o_data ) const
{
   size_t i = m_width*m_height;
   const Real *itSrc = m_data.data();
   T *itDst = o_data;
   while ( i-- )
   *itDst++ = (T)( *itSrc++ );
}

#undef BYTE
#undef UINT

#endif // __REALIMAGE1__
