#include "Sift.h"

#include <iostream>
#include <algorithm>
#include <string.h>
#include <sstream>
#include <iomanip>
#include <limits>

#include "Gauss34.h"

using namespace std;

Siftator::Siftator( int i_nbOctaves, int i_nbLevels, int i_firstOctave ):
    m_strengthThreshold( default_strength_threshold ),
    m_onEdgeThreshold( default_onedge_threshold )
{
    scale_space_format( i_nbOctaves, i_nbLevels, i_firstOctave );
}

void Siftator::scale_space_format( int i_nbOctaves, int i_nbLevels, int i_firstOctave )
{
    #ifdef _DEBUG
        if ( i_nbOctaves<0 ) cerr << "WARN: scale_space_format: number of octaves = " << i_nbOctaves << " < 0" << endl;
        if ( i_nbLevels<0 ) cerr << "WARN: scale_space_format: number of levels = " << i_nbOctaves << " < 0" << endl;
    #endif

    m_nbOctaves         = i_nbOctaves;
    m_nbLevels          = i_nbLevels;
    m_nbStoredLevels    = m_nbLevels+3;

    // resize image vectors
    m_octaves.resize( m_nbOctaves );
    vector<vector<RealImage1> >::iterator itOctave=m_octaves.begin();
    UINT iOctave = m_octaves.size();
    // allocate space for two more levels per octave for continuity
    // first level and last level cross to lower octave and upper octave respectively
    while ( iOctave-- )
        ( itOctave++ )->resize( m_nbStoredLevels );
    m_nbDoG = m_nbLevels+2;
    m_DoG.resize( m_nbDoG );
    m_gradients.resize( m_nbLevels+2 );

    // compute scale value
    m_smax = m_nbLevels+1;
    m_sigma0  = float( 1.6*::powf( 2.0f, 1.0f/m_nbLevels ) );
    m_sigmak  = ::powf( 2.0f, 1.0/m_nbLevels );
    m_dsigma0 = m_sigma0*sqrt( Real_(1)-Real_(1)/( m_sigmak*m_sigmak ) );

    m_firstOctave = i_firstOctave;
    Real_ max_standard_deviation = m_dsigma0*powf( m_sigmak, m_smin+m_nbStoredLevels-1 );        // max standard deviation used by the gaussian filter
    m_max_neighbour_distance = getGaussianKernel_halfsize( max_standard_deviation );             // distance to the farest neighbour in subsampled image
    m_max_neighbour_distance = m_max_neighbour_distance*::pow( 2, m_firstOctave+m_nbOctaves-1 ); // distance to farest neighbour in original size image
}

void Siftator::print_parameters( std::ostream &o ) const
{
    o << "computation float size                      : " << sizeof(Real_) << std::endl;
    o << "image float size                            : " << sizeof(RealImage1::Real) << std::endl;
    #ifdef VL_USEFASTMATH
        o << "Fast maths                                  : enabled" << std::endl;
    #else
        o << "Fast maths                                  : disable" << std::endl;
    #endif
    o << "Nominal smoothing value of the image        : sigman = " << m_sigman << std::endl;
    o << "Base smoothing level                        : sigma0 = " << m_sigma0 << std::endl;
    o << "Number of octaves                           : O      = " << m_nbOctaves << std::endl;
    o << "Number of levels per octave                 : S      = " << m_nbLevels << std::endl;
    o << "First octave                                : omin   = " << m_firstOctave << std::endl;
    o << "last level of an octave                     : smax   = " << m_smax << std::endl;
    o << "First level in each octave                  : smin   = " << m_smin << std::endl;
    o << "number of DoG                               : " << m_smax-m_smin << std::endl;
    o << "descriptor's gaussian magnifying coeficient : " << m_magnify << std::endl;
    o << "Keypoint strength threshold                 : " << m_strengthThreshold << endl;
    o << "on-edge threshold                           : " << m_onEdgeThreshold << endl;
    o << std::endl;
}

// i_basename is image name withour extension
// save in PGM fileformat
void Siftator::save_gaussians( const std::string &i_basename, bool i_verbose ) const
{
    for ( int o=0; o<m_nbOctaves; o++ )
        for ( int l=0; l<m_nbLevels; l++ )
        {
            stringstream ss;
            ss << i_basename << '.' << o+m_firstOctave << '.' << l << ".pgm";
            if ( i_verbose ) cout << "saving gaussian to " << ss.str() << endl;
            m_octaves[o][l].savePGM( ss.str() );
        }
}

// fills the pyramid from i_image at octave 0
// a negative value for i_firstOctave is clipped to -1
void Siftator::compute_gaussians( const RealImage1 &i_image )
{
    if ( ( m_octaves.size()==0 ) || ( m_octaves[0].size()==0 ) )
    {
        #ifdef _DEBUG
            cerr << "WARN: Siftator::setImage: pyramid's size is null" << endl;
        #endif
        return;
    }

    // process the first scale of the first octave
    RealImage1 *firstImage = &m_octaves[0][0];
    if ( m_firstOctave==0 )
        i_image.copy( *firstImage );
    else if ( m_firstOctave<0 )
    {
        RealImage1 tmpImage;
        i_image.upsample( *firstImage );

        int i = -m_firstOctave-1;
        while ( i-- )
        {
            firstImage->upsample( tmpImage );
            firstImage->swap( tmpImage );
        }
    }
    else
        i_image.downsample( *firstImage, m_firstOctave ); // i_firstOctave is > 0

    Real_ sa    = m_sigma0*::powf( m_sigmak, m_smin ),
          sb    = m_sigman/::powf( 2.0f, m_firstOctave ),
          sigma;
    if( sa > sb ) // better have a positive square
    {
        sigma = ::sqrt( sa*sa-sb*sb );
        firstImage->gaussianFilter( sigma );
    }

    int o, l;
    for ( o=0; o<m_nbOctaves; o++ )
    {
        // process first scale level, which is a downsampled copy of the lower octave's last level
        if ( o!=0 )
            m_octaves[o-1][m_nbLevels].downsample( m_octaves[o][0], 1 );

        for ( l=1; l<m_nbStoredLevels; l++ )
        {
            sigma = m_dsigma0*powf( m_sigmak, l+m_smin );
            m_octaves[o][l-1].gaussianFilter( sigma, m_octaves[o][l] );
        }
    }
}

// compute the difference of gaussians for all scales of current octave
void Siftator::compute_differences_of_gaussians()
{
    int iDoG;
    RealImage1 *itGauss = m_octaves[m_iOctave].data(),
               *itDoG   = m_DoG.data();

    for ( iDoG=0; iDoG<m_nbDoG; iDoG++ )
    {
        ( itDoG++ )->difference( itGauss[1], itGauss[0] );
        itGauss++;
    }
}

// return a list of extrema in differences of gaussians for the set octave
void Siftator::getExtrema( list<Extremum> &o_extrema )
{
    RealImage1::Real *itScale0,
                     *itScale1,
                     *itScale2;
    RealImage1::Real v;
    Extremum extremum; extremum.o=m_iTrueOctave;
    int x, y, iDoG;

    bool isMax, isMin;
    int nbDoG_1  = m_nbDoG-1,
        width_1  = m_width-1,
        height_1 = m_height-1;

    for ( iDoG=1; iDoG<nbDoG_1; iDoG++ )
    {
        itScale0 = m_DoG[iDoG-1].data()+c8;
        itScale1 = m_DoG[iDoG].data()+c8;
        itScale2 = m_DoG[iDoG+1].data()+c8;
        for ( y=1; y<height_1; y++ )
        {
            for ( x=1; x<width_1; x++ )
            {
                v = *itScale1;

                // looking for minima
                isMin  = (  v<=-0.8*m_strengthThreshold &&
                            // lower level
                            v<itScale0[c0] && v<itScale0[c1] && v<itScale0[c2] &&
                            v<itScale0[-1] && v<itScale0[0]  && v<itScale0[1]  &&
                            v<itScale0[c6] && v<itScale0[m_width] && v<itScale0[c8] &&
                            // same level
                            v<itScale1[c0] && v<itScale1[c1] && v<itScale1[c2] &&
                            v<itScale1[-1] &&                   v<itScale1[1]  &&
                            v<itScale1[c6] && v<itScale1[m_width] && v<itScale1[c8] &&
                            // uppper level
                            v<itScale2[c0] && v<itScale2[c1] && v<itScale2[c2] &&
                            v<itScale2[-1] && v<itScale2[0]  && v<itScale2[1]  &&
                            v<itScale2[c6] && v<itScale2[m_width] && v<itScale2[c8] );
                // looking for maxima
                isMax = (   v>=0.8*m_strengthThreshold &&
                            // lower level
                            v>itScale0[c0] && v>itScale0[c1] && v>itScale0[c2] &&
                            v>itScale0[-1] && v>itScale0[0]  && v>itScale0[1]  &&
                            v>itScale0[c6] && v>itScale0[m_width] && v>itScale0[c8] &&
                            // same level
                            v>itScale1[c0] && v>itScale1[c1] && v>itScale1[c2] &&
                            v>itScale1[-1] &&                   v>itScale1[1]  &&
                            v>itScale1[c6] && v>itScale1[m_width] && v>itScale1[c8] &&
                            // uppper level
                            v>itScale2[c0] && v>itScale2[c1] && v>itScale2[c2] &&
                            v>itScale2[-1] && v>itScale2[0]  && v>itScale2[1]  &&
                            v>itScale2[c6] && v>itScale2[m_width] && v>itScale2[c8] );
                if ( isMax || isMin )
                {
                    extremum.x=x; extremum.y=y; extremum.s=iDoG-1; extremum.isMax=isMax;
                    o_extrema.push_back( extremum );
                }

                itScale0++; itScale1++; itScale2++;
            }
            itScale0+=2; itScale1+=2; itScale2+=2;
        }
    }
}

void Siftator::refinePoints( const list<Extremum> &i_extrema, list<RefinedPoint> &o_refinedPoints )
{
    list<Extremum>::const_iterator itExtremum = i_extrema.begin();
    RefinedPoint rp;
    while ( itExtremum!=i_extrema.end() )
    {
        if ( refinePoint( *itExtremum++, rp ) )
            o_refinedPoints.push_back( rp );
    }
}

bool Siftator::refinePoint( const Extremum &i_p, RefinedPoint &o_p )
{
    Real_ m[12];
    Real_ b[3];
    int x = i_p.x,
        y = i_p.y;
    int Dx=0, Dy=0;
    int iter;

    UINT offset;
    RealImage1::Real *itScale0, *itScale1, *itScale2;
    Real_ dx, dy, ds, dxx, dyy, dxy;

    // reiterate until variation is low
    for( iter=0; iter<5; iter++ )
    {
        x += Dx;
        y += Dy;

        offset = x+y*m_width;
        itScale0 = m_DoG[i_p.s].data()+offset;
        itScale1 = m_DoG[i_p.s+1].data()+offset;
        itScale2 = m_DoG[i_p.s+2].data()+offset;

        dx = 0.5*( itScale1[1]-itScale1[-1] );
        dy = 0.5*( itScale1[m_width]-itScale1[c1] );
        ds = 0.5*( itScale2[0]-itScale0[0] );
        m[3]  = b[0] = -dx;
        m[7]  = b[1] = -dy;
        m[11] = b[2] = -ds;

        m[0]  = dxx = itScale1[1]+itScale1[-1]-( 2.*itScale1[0] );  // dxx
        m[5]  = dyy = itScale1[m_width]+itScale1[c1]-( 2.*itScale1[0] ); // dyy
        m[10] = itScale2[0]+itScale0[0]-( 2.*itScale1[0] );   // dss

        m[1] = m[4] = dxy = 0.25*( itScale1[c8]+itScale1[c0]-itScale1[c6]-itScale1[c2] ); // dxy
        m[2] = m[8] = 0.25*( itScale2[1]+itScale0[-1]-itScale0[1]-itScale2[-1] );   // dxs
        m[6] = m[9] = 0.25*( itScale2[m_width]+itScale0[c1]-itScale0[m_width]-itScale2[c1] ); // dys

        siftpp__gauss33_invert_b( m, b );

        // shall we reiterate ?
        Dx=   ( ( ( b[0]>0.6 ) && ( x<m_width-2 ) )?1:0 )
            + ( ( ( b[0]<-0.6 ) && ( x>1 ) )?-1:0 );
        Dy=   ( ( ( b[1]>0.6 ) && ( y<m_height-2 ) )?1:0 )
            + ( ( ( b[1]<-0.6 ) && ( y>1 ) )?-1:0 );
        if( Dx == 0 && Dy == 0 ) break;
    }

    Real_ xn = x + b[0] ;
    Real_ yn = y + b[1] ;
    Real_ sn = i_p.s + b[2] ;

    memcpy( &o_p, &i_p, 4*sizeof(int)+sizeof(bool) );
    o_p.rx = xn*m_samplingPace;
    o_p.ry = yn*m_samplingPace;
    o_p.rs = m_sigma0*powf( 2.0f, i_p.o+( sn/m_nbLevels ) );

    // edge test
    Real_ val = itScale1[0] + 0.5*( dx*b[0] + dy*b[1] + ds*b[2] );
    Real_ score = dxx+dyy;
    score = ( score*score )/( dxx*dyy-dxy*dxy );
    return ( ( fast_maths::fast_abs(val)>m_strengthThreshold ) &&
             ( score<m_scoreMax ) && // m_scoreMax = ( ( m_onEdgeThreshold+1 )*( m_onEdgeThreshold+1 )/m_onEdgeThreshold)
             ( score>=0 ) &&
             ( fast_maths::fast_abs( b[0] )<1.5 ) &&
             ( fast_maths::fast_abs( b[1] )<1.5 ) &&
             ( fast_maths::fast_abs( b[2] )<1.5 ) &&
             ( xn>=0 ) &&
             ( xn<=( m_width-1 ) ) &&
             ( yn>=0 ) &&
             ( yn<=( m_height-1 ) ) &&
             ( sn>=m_smin ) &&
             ( sn<=m_smax ) );
}

void Siftator::compute_gradients()
{
    for ( int iLevel=1; iLevel<=m_nbLevels; iLevel++ )
        m_octaves[m_iOctave][iLevel].gradient( m_gradients[iLevel] );
}

int Siftator::orientations( RefinedPoint &i_p, Real_ o_angles[m_maxNbAngles] )
{
    // keypoint fractional geometry
    Real_ x     = i_p.rx/m_samplingPace;
    Real_ y     = i_p.ry/m_samplingPace;
    Real_ sigma = i_p.rs/m_samplingPace;

    // shall we use keypoints.ix,iy,is here?
    int xi = ((int) (x+0.5)) ;
    int yi = ((int) (y+0.5)) ;
    int si = int(i_p.s);

    const Real_ sigmaw = m_windowFactor*sigma;
    #ifdef __ORIGINAL__
        const int W = (int) ceil(3.0 * sigmaw);
    #else
        const int W = (int)floor( 3*sigmaw );
    #endif

    #ifdef _DEBUG
        // check the point is not out of range
        if ( xi < 0         ||
             xi >= m_width  ||
             yi < 0         ||
             yi >= m_height ||
             si < 0         ||
             si > m_smax-2 )
        {
            std::cerr << "WARN: orientations: try to find orientations of point out of range" << std::endl;
            return 0;
        }
    #endif

    // fill the SIFT histogram
    Real_ dx, dy, r2,
         wgt, mod, ang;
    int  offset;
    RealImage1::Real *p = m_gradients[si+1].data()+( xi+yi*m_width )*2;    
    std::fill( m_histo, m_histo+m_nbBins, 0 );
    for ( int ys=std::max( -W, 1-yi ); ys<=std::min( W, m_height-2-yi ); ys++ )
    {
        for ( int xs=std::max( -W, 1-xi ); xs<=std::min( W, m_width-2-xi ); xs++ )
        {
            dx = xi+xs-x;
            dy = yi+ys-y;
            r2 = dx*dx+dy*dy ;

            // limit to a circular window
            if ( r2>=W*W+0.5 ) continue;
    
            wgt    = fast_maths::fast_expn( r2/( 2*sigmaw*sigmaw ) );
            offset = ( xs+ys*m_width )*2;
            mod    = p[offset];
            ang    = p[offset+1];

            int bin = (int) floor( m_nbBins*ang/( 2*M_PI ) ) ;
            m_histo[bin] += mod*wgt ;

        }
    }
	
    Real_ prev;
    #ifdef __ORIGINAL__
        // smooth histogram  (Lowe style)
        for (int iter = 0; iter<6; iter++)
        {
            prev = m_histo[m_nbBins-1];
            for ( int i=0; i<m_nbBins; i++ )
            {
                Real_ newh = ( prev+m_histo[i]+m_histo[(i+1)%m_nbBins] )/3.0;
                prev = m_histo[i] ;
                m_histo[i] = newh ;
            }
        }
    #else
        // smooth histogram  (Vedaldi style)
        // mean of a bin and its two neighbour values (x6)
        Real_ *itHisto,
             first, mean;
        int iHisto,
            iIter = 6;
        while ( iIter-- )
        {
            itHisto = m_histo;
            iHisto  = m_nbBins-2;
            first = prev = *itHisto;
            *itHisto = ( m_histo[m_nbBins-1]+( *itHisto )+itHisto[1] )/3.; itHisto++;
            while ( iHisto-- ){
                mean = ( prev+(*itHisto)+itHisto[1] )/3.;
                prev = *itHisto;
                *itHisto++ = mean;
            }
            *itHisto = ( prev+( *itHisto )+first )/3.; itHisto++;
        }
    #endif
    
    // find histogram's peaks
    // peaks are values > 80% of histoMax and > to both its neighbours
    Real_ histoMax = 0.8*( *std::max_element( m_histo, m_histo+m_nbBins ) );
    Real_ v, next, di;
    int nbAngles = 0;
    for ( int i=0; i<m_nbBins; i++ )
    {
        v    = m_histo[i];
        prev = m_histo[ ( i==0 )?m_nbBins-1:i-1 ];
        next = m_histo[ ( i==( m_nbBins-1 ) )?0:i+1 ];
        if ( ( v>histoMax ) && ( v>prev ) && ( v>next ) )
        {
            // we found a peak
            // compute angle by quadratic interpolation
            di = -0.5*( next-prev )/( next+prev-2*v ) ;
            o_angles[nbAngles++] = 2*M_PI*( i+di+0.5 )/m_nbBins;
            if ( nbAngles==m_maxNbAngles ) return m_maxNbAngles;
        }
    }
    return nbAngles;
}

// o_descritpor must be of size m_descriptorSize
void Siftator::normalizeDescriptor( Real_ *o_descriptors )
{
    Real_  norm   = 0;
    int   i      = m_descriptorSize;
    Real_ *itDesc = o_descriptors;
    while ( i-- ){
        norm += ( *itDesc )*( *itDesc );
        itDesc++;
    }
    #ifdef __ORIGINAL__
        norm = fast_maths::fast_sqrt( norm );
    #else
        norm = std::numeric_limits<Real_>::epsilon()+fast_maths::fast_sqrt( norm );
    #endif

    i      = m_descriptorSize;
    itDesc = o_descriptors;
    while ( i-- ){
        *itDesc = ( *itDesc )/norm;
        itDesc++;
    }
}

// o_descritpor must be of size m_descriptorSize]
void Siftator::truncateDescriptor( Real_ *o_descriptors )
{
    int   i      = m_descriptorSize;
    Real_ *itDesc = o_descriptors;
    while ( i-- ){
        if ( ( *itDesc )>m_descriptorTreshold )
            ( *itDesc )=m_descriptorTreshold;
        itDesc++;
    }
}

// o_descritpor must be of size m_descriptorSize]
void Siftator::descriptor( RefinedPoint &i_p, Real_ i_angle, Real_ *o_descriptor )
{
    // keypoint fractional geometry
    Real_ x     = i_p.rx/m_samplingPace;
    Real_ y     = i_p.ry/m_samplingPace;
    Real_ sigma = i_p.rs/m_samplingPace;
    Real_ st0   = sinf( i_angle );
    Real_ ct0   = cosf( i_angle );

    // shall we use keypoints.ix,iy,is here?
    int xi = int( x+0.5 );
    int yi = int( y+0.5 );
    int si = i_p.s;

    const Real_ SBP = m_magnify*sigma;
    const int  W   = (int)floor( sqrt( 2.0 )*SBP*( m_NBP+1 )/2.0+0.5 );

    /* Offsets to move in the descriptor. */
    /* Use Lowe's convention. */
    const int binto = 1 ;
    const int binyo = m_NBO*m_NBP;
    const int binxo = m_NBO;

    #ifdef _DEBUG
        // check bounds
        if ( i_p.o <  m_firstOctave                       ||
             i_p.o >= m_firstOctave+m_nbOctaves           ||
             xi    <  0                                   ||
             xi    >  m_width-1                           ||
             yi    <  0                                   ||
             yi    >  m_height-1                          ||
             si    <  0                                   ||
             si    >  m_smax-2 )
        {
            std::cerr << "WARN: descriptors: try to find descriptors of a point out of range" << std::endl;
            return;
        }
    #endif

    std::fill( o_descriptor, o_descriptor+m_descriptorSize, 0 ) ;

    /* Center the scale space and the descriptor on the current keypoint.
    * Note that dpt is pointing to the bin of center (SBP/2,SBP/2,0).
    */
    const RealImage1::Real *p = m_gradients[si+1].data()+( xi+yi*m_width )*2;
    Real_ *dp = o_descriptor+( m_NBP/2 )*( binyo+binxo );

    #define atd(dbinx,dbiny,dbint) *(dp + (dbint)*binto + (dbiny)*binyo + (dbinx)*binxo)

    /*
    * Process pixels in the intersection of the image rectangle
    * (1,1)-(M-1,N-1) and the keypoint bounding box.
    */
    const Real_ wsigma = m_NBP/2 ;
    int  offset;
    Real_ mod, angle, theta;
    Real_ dx, dy;
    Real_ nx, ny, nt;
    int  binx, biny, bint;
    Real_ rbinx, rbiny, rbint;
    int dbinx, dbiny, dbint;
    Real_ weight;
    Real_ win;
    for ( int dyi=std::max( -W, 1-yi ); dyi<=std::min( W, m_height-2-yi ); dyi++ )
    {
        for ( int dxi=std::max( -W, 1-xi ); dxi<=std::min( W, m_width-2-xi ); dxi++ )
        {
            // retrieve
            offset = ( dxi+dyi*m_width )*2;
            mod    = p[ offset ];
            angle  = p[ offset+1 ];
            /*
            // __FAST_MATH
            theta  = -angle+i_angle;
            if ( theta>=0 )
                theta = std::fmod( theta, Real_( 2*M_PI ) );
            else
                theta = 2*M_PI+std::fmod( theta, Real_( 2*M_PI ) );
            */
            theta = fast_maths::fast_mod_2pi( -angle+i_angle );

            // fractional displacement
            dx = xi+dxi-x;
            dy = yi+dyi-y;

            // get the displacement normalized w.r.t. the keypoint
            // orientation and extension.
            nx = ( ct0*dx + st0*dy )/SBP ;
            ny = ( -st0*dx + ct0*dy )/SBP ;
            nt = m_NBO*theta/( 2*M_PI ) ;

            // Get the gaussian weight of the sample. The gaussian window
            // has a standard deviation equal to NBP/2. Note that dx and dy
            // are in the normalized frame, so that -NBP/2 <= dx <= NBP/2.
            win = fast_maths::fast_expn( ( nx*nx+ny*ny )/( 2.0*wsigma*wsigma ) );
            // win = std::exp( -( nx*nx+ny*ny )/( 2.0*wsigma*wsigma ) ); // __FAST_MATH

            // The sample will be distributed in 8 adjacent bins.
            // We start from the ``lower-left'' bin.
            binx = fast_maths::fast_floor( nx-0.5 );
            //binx = std::floor( nx-0.5 ); // __FAST_MATH
            biny = fast_maths::fast_floor( ny-0.5 );
            //biny = std::floor( ny-0.5 ); // __FAST_MATH
            bint = fast_maths::fast_floor( nt );
            //bint = std::floor( nt );     // __FAST_MATH
            rbinx = nx-( binx+0.5 );
            rbiny = ny-( biny+0.5 );
            rbint = nt-bint;

            // Distribute the current sample into the 8 adjacent bins
            for ( dbinx=0; dbinx<2; dbinx++ )
            {
                for ( dbiny=0; dbiny<2; dbiny++ )
                {
                    for ( dbint=0; dbint<2; dbint++ )
                    {
                        if ( ( ( binx+dbinx ) >= ( -(m_NBP/2)   ) ) &&
                             ( ( binx+dbinx ) <  ( m_NBP/2      ) ) &&
                             ( ( biny+dbiny ) >= ( -( m_NBP/2 ) ) ) &&
                             ( ( biny+dbiny ) <  ( m_NBP/2      ) ) )
                        {
                            weight = win*mod
                                    //*std::fabs( 1-dbinx-rbinx )  // __FAST_MATH
                                    *fast_maths::fast_abs( 1-dbinx-rbinx )
                                    //*std::fabs( 1-dbiny-rbiny )  // __FAST_MATH
                                    *fast_maths::fast_abs( 1-dbiny-rbiny )
                                    //*std::fabs( 1-dbint-rbint ); // __FAST_MATH
                                    *fast_maths::fast_abs( 1-dbint-rbint );

                            atd( binx+dbinx, biny+dbiny, ( bint+dbint )%m_NBO ) += weight ;
                        }
                    }
                }
            }
        }
    }
}
