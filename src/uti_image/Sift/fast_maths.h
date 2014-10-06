#ifndef __FAST_MATHS__
#define __FAST_MATHS__

#include <iostream>
#include <cmath>
#include <stdint.h>

// this is from Andrea Vedaldi's siftpp (now VLFeat)

#ifndef VL_USEFASTMATH
    #define VL_USEFASTMATH
#endif

#ifndef M_PI
	#define M_PI 3.14159265358979323846
#endif

namespace fast_maths
{
    extern int  const expnTableSize;
    extern Real_ const expnTableMax;
    extern Real_ expnTable [];

    inline int32_t fast_floor( Real_ x )
    {
        #ifdef VL_USEFASTMATH
            return ( x>=0 )? int32_t(x) : int32_t(std::floor(x)) ;
        #else
            return int32_t( std::floor(x) ) ;
        #endif
    }

    inline Real_ fast_expn( Real_ x )
    {
        #ifdef _DEBUG
            if ( x<0 && x>expnTableMax ){
                std::cerr << "ERROR: fast_maths::fast_expn( " << x << " ) : this function is defined in [0," << expnTableMax << ']' << std::endl;
                return Real_(0);
            }
        #endif
        #ifdef VL_USEFASTMATH
            x *= expnTableSize/expnTableMax;
            int32_t i = fast_floor(x) ;
            Real_    r = x-i;
            Real_    a = expnTable[i] ;
            Real_    b = expnTable[i+1] ;
            return a+r*( b-a ) ;
        #else
            return ::exp( -x ) ;
        #endif
    }

    inline Real_ fast_mod_2pi( Real_ x )
    {
        #ifdef VL_USEFASTMATH
            while( x<Real_(0) ) x+=Real_( 2*M_PI );
            while ( x>Real_( 2*M_PI ) ) x-=Real_( 2*M_PI );
            return x ;
        #else
            return ( x>=0 ) ? std::fmod( x, Real_( 2*M_PI ) )
                            : 2*M_PI + std::fmod( x, Real_( 2*M_PI ) );
        #endif
    }

    inline Real_ fast_abs( Real_ x )
    {
        #ifdef VL_USEFASTMATH
            return ( x>=0 ) ? x : -x ;
        #else
            return std::fabs(x) ;
        #endif
    }

    inline Real_ fast_atan2( Real_ y, Real_ x )
    {
        #ifdef VL_USEFASTMATH
            /*
            The function f(r)=atan((1-r)/(1+r)) for r in [-1,1] is easier to
            approximate than atan(z) for z in [0,inf]. To approximate f(r) to
            the third degree we may solve the system

             f(+1) = c0 + c1 + c2 + c3 = atan(0) = 0
             f(-1) = c0 - c1 + c2 - c3 = atan(inf) = pi/2
             f(0)  = c0                = atan(1) = pi/4

            which constrains the polynomial to go through the end points and
            the middle point.

            We still miss a constrain, which might be simply a constarint on
            the derivative in 0. Instead we minimize the Linf error in the
            range [0,1] by searching for an optimal value of the free
            parameter. This turns out to correspond to the solution

             c0=pi/4, c1=-0.9675, c2=0, c3=0.1821

            which has maxerr = 0.0061 rad = 0.35 grad.
            */

            Real_ angle, r ;
            Real_ const c3 = 0.1821;
            Real_ const c1 = 0.9675;
            Real_ abs_y    = fast_abs(y) + Real_( 1e-10 );

            if (x >= 0)
            {
                r     = ( x-abs_y )/( x+abs_y );
                angle = Real_( M_PI/4.0 );
            }
            else
            {
                r = ( x+abs_y )/( abs_y-x );
                angle = Real_( 3*M_PI/4.0 );
            }
            angle += ( c3*r*r-c1 )*r;
            return ( y<0 )? -angle : angle;
        #else
            return std::atan2( y, x );
        #endif
    }

    inline float fast_resqrt( float x )
    {
        #ifdef VL_USEFASTMATH
            // Works if float is 32 bit ...
            union
            {
                float   x;
                int32_t i;
            } u;

            float xhalf = float( 0.5 )*x;
            u.x = x ;                                 // get bits for floating value
            u.i = 0x5f3759df - ( u.i>>1 );            // gives initial guess y0
            u.x = u.x*( float( 1.5 )-xhalf*u.x*u.x ); // Newton step (may repeat)
            u.x = u.x*( float( 1.5 )-xhalf*u.x*u.x ); // Newton step (may repeat)
            return u.x ;
        #else
            return float( 1.0 )/std::sqrt( x );
        #endif
    }

    inline double fast_resqrt( double x )
    {
        #ifdef VL_USEFASTMATH
            #ifdef __ORIGINAL__
                return (double)fast_resqrt((float)x);
            #else
                // Works if double is 64 bit ...
                union
                {
                    double  x;
                    int64_t i;
                } u;
                double xhalf = double( 0.5 )*x;
                u.x = x;                                   // get bits for floating value
                u.i = 0x5fe6ec85e7de30daLL - ( u.i>>1 );   // gives initial guess y0
                u.x = u.x*( double( 1.5 )-xhalf*u.x*u.x ); // Newton step (may repeat)
                u.x = u.x*( double( 1.5 )-xhalf*u.x*u.x ); // Newton step (may repeat)
                return u.x ;
            #endif
        #else
            return double( 1.0 )/std::sqrt( x );
        #endif
    }

    inline Real_ fast_sqrt( Real_ x )
    {
        #ifdef VL_USEFASTMATH
            return ( x<1e-8 ) ? 0 : x*fast_resqrt( x );
        #else
            return std::sqrt( x );
        #endif
    }
}

#endif
