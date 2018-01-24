#include "DescriptorExtractor.h"
#include "StdAfx.h"



template <class tData, class tComp>
DescriptorExtractor<tData,tComp>::DescriptorExtractor(Im2D<tData, tComp> Image)
{
    m_image=Image;
    this->gradient(1.0);
}


template <class tData, class tComp>
DescriptorExtractor<tData,tComp>::~DescriptorExtractor(){}


template <class tData, class tComp>
void DescriptorExtractor<tData,tComp>::gradient(REAL8 i_maxValue)
{
    m_gradim.Resize( Pt2di( m_image.tx()*2, m_image.ty() ) );

    const REAL8 coef = REAL8(0.5)/i_maxValue;
    const int c1 = -m_image.sz().x;
    int offset = m_image.sz().x+1;
    const tData *src = m_image.data_lin()+offset;
    REAL4 *dst = m_gradim.data_lin()+2*offset;
    REAL8 gx, gy, theta;
    int width_2 = m_image.sz().x-2,
        y = m_image.sz().y-2,
        x;
    while ( y-- )
    {
        x = width_2;
        while ( x-- )
        {
            gx = ( REAL8 )( coef*( REAL8(src[1])-REAL8(src[-1]) ) );
            gy = ( REAL8 )( coef*( REAL8(src[m_image.sz().x])-REAL8(src[c1]) ) );
            dst[0] = (REAL4)std::sqrt( gx*gx+gy*gy );

            theta = std::fmod( REAL8( std::atan2( gy, gx ) ), REAL8( 2*M_PI ) );
            if ( theta<0 ) theta+=2*M_PI;
            dst[1] = (REAL4)theta;

            src++; dst+=2;
        }
        src+=2; dst+=4;
    }
}



template <class tData, class tComp>
void DescriptorExtractor<tData,tComp>::describe(REAL8 i_x, REAL8 i_y, REAL8 i_localScale, REAL8 i_angle, REAL8 *o_descriptor )
    {
        REAL8 st0 = sinf( i_angle ),
              ct0 = cosf( i_angle );

        int xi = int( i_x+0.5 );
        int yi = int( i_y+0.5 );

        const REAL8 SBP = DIGEO_DESCRIBE_MAGNIFY*i_localScale;
        const int  W   = (int)ceil( sqrt( 2.0 )*SBP*( DIGEO_DESCRIBE_NBP+1 )/2.0+0.5 );

        /* Offsets to move in the descriptor. */
        /* Use Lowe's convention. */
        const int binto = 1 ;
        const int binyo = DIGEO_DESCRIBE_NBO*DIGEO_DESCRIBE_NBP;
        const int binxo = DIGEO_DESCRIBE_NBO;

        std::fill( o_descriptor, o_descriptor+DIGEO_DESCRIPTOR_SIZE, 0 ) ;

        /* Center the scale space and the descriptor on the current keypoint.
        * Note that dpt is pointing to the bin of center (SBP/2,SBP/2,0).
        */
        const INT width  = m_gradim.sz().x/2,
              height = m_gradim.sz().y;
        const REAL4 *p = m_gradim.data_lin()+( xi+yi*width )*2;
        REAL8 *dp = o_descriptor+( DIGEO_DESCRIBE_NBP/2 )*( binyo+binxo );

        #define atd(dbinx,dbiny,dbint) *(dp + (dbint)*binto + (dbiny)*binyo + (dbinx)*binxo)

        /*
        * Process pixels in the intersection of the image rectangle
        * (1,1)-(M-1,N-1) and the keypoint bounding box.
        */
        const REAL8 wsigma = DIGEO_DESCRIBE_NBP/2 ;
        int  offset;
        REAL8 mod, angle, theta,
              dx, dy,
              nx, ny, nt;
        int  binx, biny, bint;
        REAL8 rbinx, rbiny, rbint;
        int dbinx, dbiny, dbint;
        REAL weight, win;
        for ( int dyi=std::max( -W, 1-yi ); dyi<=std::min( W, height-2-yi ); dyi++ )
        {
            for ( int dxi=std::max( -W, 1-xi ); dxi<=std::min( W, width-2-xi ); dxi++ )
            {
                // retrieve
                offset = ( dxi+dyi*width )*2;
                mod    = p[ offset ];
                angle  = p[ offset+1 ];

                theta  = -angle+i_angle;
                if ( theta>=0 )
                    theta = std::fmod( theta, REAL8( 2*M_PI ) );
                else
                    theta = 2*M_PI+std::fmod( theta, REAL8( 2*M_PI ) );

                // fractional displacement
                dx = xi+dxi-i_x;
                dy = yi+dyi-i_y;

                // get the displacement normalized w.r.t. the keypoint
                // orientation and extension.
                nx = ( ct0*dx + st0*dy )/SBP ;
                ny = ( -st0*dx + ct0*dy )/SBP ;
                nt = DIGEO_DESCRIBE_NBO*theta/( 2*M_PI ) ;

                // Get the gaussian weight of the sample. The gaussian window
                // has a standard deviation equal to NBP/2. Note that dx and dy
                // are in the normalized frame, so that -NBP/2 <= dx <= NBP/2.
                 win = std::exp( -( nx*nx+ny*ny )/( 2.0*wsigma*wsigma ) );

                // The sample will be distributed in 8 adjacent bins.
                // We start from the ``lower-left'' bin.
                binx = std::floor( nx-0.5 );
                biny = std::floor( ny-0.5 );
                bint = std::floor( nt );
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
                            if ( ( ( binx+dbinx ) >= ( -(DIGEO_DESCRIBE_NBP/2)   ) ) &&
                                 ( ( binx+dbinx ) <  ( DIGEO_DESCRIBE_NBP/2      ) ) &&
                                 ( ( biny+dbiny ) >= ( -( DIGEO_DESCRIBE_NBP/2 ) ) ) &&
                                 ( ( biny+dbiny ) <  ( DIGEO_DESCRIBE_NBP/2      ) ) )
                            {
                                weight = win*mod
                                        *std::fabs( 1-dbinx-rbinx )
                                        *std::fabs( 1-dbiny-rbiny )
                                        *std::fabs( 1-dbint-rbint );

                                atd( binx+dbinx, biny+dbiny, ( bint+dbint )%DIGEO_DESCRIBE_NBO ) += weight ;
                            }
                        }
                    }
                }
            }
        }
    }


template <class tData, class tComp>
void DescriptorExtractor< tData, tComp >::normalizeDescriptor( REAL8 *io_descriptor )
{
    REAL8 norm    = 0;
    int   i       = DIGEO_DESCRIPTOR_SIZE;
    REAL8 *itDesc = io_descriptor;
    while ( i-- ){
        norm += ( *itDesc )*( *itDesc );
        itDesc++;
    }

    norm = std::sqrt( norm )+std::numeric_limits<REAL8>::epsilon();

    i      = DIGEO_DESCRIPTOR_SIZE;
    itDesc = io_descriptor;
    while ( i-- ){
        *itDesc = ( *itDesc )/norm;
        itDesc++;
    }
}


template <class tData, class tComp>
void DescriptorExtractor< tData, tComp >::truncateDescriptor( REAL8 *io_descriptor )
{
    int    i      = DIGEO_DESCRIPTOR_SIZE;
    REAL8 *itDesc = io_descriptor;
    while ( i-- ){
        if ( ( *itDesc )>DIGEO_DESCRIBE_THRESHOLD )
            ( *itDesc )=DIGEO_DESCRIBE_THRESHOLD;
        itDesc++;
    }
}



template <class tData, class tComp>
void DescriptorExtractor< tData, tComp >::normalize_and_truncate( REAL8 *io_descriptor )
{
    this->normalizeDescriptor( io_descriptor );
    this->truncateDescriptor( io_descriptor );
    this->normalizeDescriptor( io_descriptor );
}


template class DescriptorExtractor<U_INT1,INT>;
template class DescriptorExtractor<U_INT2,INT>;
template class DescriptorExtractor<REAL4,REAL8>;
