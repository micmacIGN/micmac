// file:        sift.ipp
// author:      Andrea Vedaldi
// description: Sift inline members definition

// AUTORIGHTS
// Copyright (c) 2006 The Regents of the University of California
// All Rights Reserved.
// 
// Created by Andrea Vedaldi (UCLA VisionLab)
// 
// Permission to use, copy, modify, and distribute this software and its
// documentation for educational, research and non-profit purposes,
// without fee, and without a written agreement is hereby granted,
// provided that the above copyright notice, this paragraph and the
// following three paragraphs appear in all copies.
// 
// This software program and documentation are copyrighted by The Regents
// of the University of California. The software program and
// documentation are supplied "as is", without any accompanying services
// from The Regents. The Regents does not warrant that the operation of
// the program will be uninterrupted or error-free. The end-user
// understands that the program was developed for research purposes and
// is advised not to rely exclusively on the program for any reason.
// 
// This software embodies a method for which the following patent has
// been issued: "Method and apparatus for identifying scale invariant
// features in an image and use of same for locating an object in an
// image," David G. Lowe, US Patent 6,711,293 (March 23,
// 2004). Provisional application filed March 8, 1999. Asignee: The
// University of British Columbia.
// 
// IN NO EVENT SHALL THE UNIVERSITY OF CALIFORNIA BE LIABLE TO ANY PARTY
// FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES,
// INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND
// ITS DOCUMENTATION, EVEN IF THE UNIVERSITY OF CALIFORNIA HAS BEEN
// ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. THE UNIVERSITY OF
// CALIFORNIA SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE. THE SOFTWARE PROVIDED HEREUNDER IS ON AN "AS IS"
// BASIS, AND THE UNIVERSITY OF CALIFORNIA HAS NO OBLIGATIONS TO PROVIDE
// MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.


/** 
 ** @file
 ** @brief SIFT class - inline functions and members 
 **/
#include<iostream>
#include<cassert>

namespace VL
{

namespace Detail
{
extern int const expnTableSize ;
extern VL::float_t const expnTableMax ;
extern VL::float_t expnTable [] ;
} 

/** @brief Get width of source image
 ** @result width.
 **/
inline
int     
Sift::getWidth() const
{
  return width ;
}

/** @brief Get height of source image
 ** @result height.
 **/
inline
int
Sift::getHeight() const
{
  return height ;
}

/** @brief Get width of an octave
 ** @param o octave index.
 ** @result width of octave @a o.
 **/
inline
int     
Sift::getOctaveWidth(int o) const
{
  assert( omin <= o && o < omin + O ) ;
  return (o >= 0) ? (width >> o) : (width << -o) ;
}

/** @brief Get height of an octave
 ** @param o octave index.
 ** @result height of octave @a o.
 **/
inline
int
Sift::getOctaveHeight(int o) const
{
  assert( omin <= o && o < omin + O ) ;
  return (o >= 0) ? (height >> o) : (height << -o) ;
}

/** @brief Get octave
 ** @param o octave index.
 ** @return pointer to octave @a o.
 **/
inline
VL::pixel_t * 
Sift::getOctave(int o) 
{
  assert( omin <= o && o < omin + O ) ;
  return octaves[o-omin] ;
}
 
/** @brief Get level
 ** @param o octave index.
 ** @param s level index.
 ** @result pointer to level @c (o,s).
 **/
inline
VL::pixel_t * 
Sift::getLevel(int o, int s) 
{
  assert( omin <= o && o <  omin + O ) ;
  assert( smin <= s && s <= smax     ) ;  
  return octaves[o - omin] +
    getOctaveWidth(o)*getOctaveHeight(o) * (s-smin) ;
}

/** @brief Get octave sampling period
 ** @param o octave index.
 ** @result Octave sampling period (in pixels).
 **/
inline
VL::float_t
Sift::getOctaveSamplingPeriod(int o) const
{
  return (o >= 0) ? (1 << o) : 1.0f / (1 << -o) ;
}

/** @brief Convert index into scale
 ** @param o octave index.
 ** @param s scale index.
 ** @return scale.
 **/
inline
VL::float_t
Sift::getScaleFromIndex(VL::float_t o, VL::float_t s) const
{
  return sigma0 * powf( 2.0f, o + s / S ) ;
}

/** @brief Get keypoint list begin
 ** @return iterator to the beginning.
 **/
inline
Sift::KeypointsIter
Sift::keypointsBegin()
{
  return keypoints.begin() ;
}

/** @brief Get keypoint list end
 ** @return iterator to the end.
 **/
inline
Sift::KeypointsIter
Sift::keypointsEnd()
{
  return keypoints.end() ;
}

/** @brief Set normalize descriptor flag */
inline
void
Sift::setNormalizeDescriptor(bool flag)
{
  normalizeDescriptor = flag ;
}

/** @brief Get normalize descriptor flag */
inline
bool
Sift::getNormalizeDescriptor() const
{
  return normalizeDescriptor ;
}

/** @brief Set descriptor magnification */
inline
void
Sift::setMagnification(VL::float_t _magnif)
{
  magnif = _magnif ;
}

/** @brief Get descriptor magnification */
inline
VL::float_t
Sift::getMagnification() const
{
  return magnif ;
}

/** @brief Fast @ exp(-x)
 **
 ** The argument must be in the range 0-25.0 (bigger arguments may be
 ** truncated to zero).
 **
 ** @param x argument.
 ** @return @c exp(-x)
 **/
inline
VL::float_t
fast_expn(VL::float_t x)
{
  assert(VL::float_t(0) <= x && x <= Detail::expnTableMax) ;
#ifdef VL_USEFASTMATH
  x *= Detail::expnTableSize / Detail::expnTableMax ;
  VL::int32_t i = fast_floor(x) ;
  VL::float_t r = x - i ;
  VL::float_t a = VL::Detail::expnTable[i] ;
  VL::float_t b = VL::Detail::expnTable[i+1] ;
  return a + r * (b - a) ;
#else
  return exp(-x) ;
#endif
}

/** @brief Fast @c mod(x,2pi)
 **
 ** The function quickly computes the value @c mod(x,2pi).
 ** 
 ** @remark The computation is fast only for arguments @a x which are
 ** small in modulus.
 **
 ** @remark For negative arguments, the semantic of the function is
 ** not equivalent to the standard library @c fmod function.
 **
 ** @param x function argument.
 ** @return @c mod(x,2pi)
 **/
inline
VL::float_t 
fast_mod_2pi(VL::float_t x)
{
#ifdef VL_USEFASTMATH
  while(x < VL::float_t(0)      ) x += VL::float_t(2*M_PI) ;
  while(x > VL::float_t(2*M_PI) ) x -= VL::float_t(2*M_PI) ;
  return x ;
#else
  return (x>=0) ? std::fmod(x, VL::float_t(2*M_PI)) 
    : 2*M_PI + std::fmod(x, VL::float_t(2*M_PI)) ;
#endif
}

/** @brief Fast @c (int) floor(x)
 ** @param x argument.
 ** @return @c float(x)
 **/
inline
int32_t 
fast_floor(VL::float_t x)
{
#ifdef VL_USEFASTMATH
  return (x>=0)? int32_t(x) : std::floor(x) ;
  //  return int32_t( x - ((x>=0)?0:1) ) ; 
#else
  return int32_t( std::floor(x) ) ;
#endif
}

/** @brief Fast @c abs(x)
 ** @param x argument.
 ** @return @c abs(x)
 **/
inline
VL::float_t
fast_abs(VL::float_t x)
{
#ifdef VL_USEFASTMATH
  return (x >= 0) ? x : -x ;
#else
  return std::fabs(x) ; 
#endif
}

/** @brief Fast @c atan2
 ** @param x argument.
 ** @param y argument.
 ** @return Approximation of @c atan2(x).
 **/
inline
VL::float_t
fast_atan2(VL::float_t y, VL::float_t x)
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

  VL::float_t angle, r ;
  VL::float_t const c3 = 0.1821 ;
  VL::float_t const c1 = 0.9675 ;
  VL::float_t abs_y    = fast_abs(y) + VL::float_t(1e-10) ;

  if (x >= 0) {
    r = (x - abs_y) / (x + abs_y) ;
    angle = VL::float_t(M_PI/4.0) ;
  } else {
    r = (x + abs_y) / (abs_y - x) ;
    angle = VL::float_t(3*M_PI/4.0) ;
  } 
  angle += (c3*r*r - c1) * r ; 
  return (y < 0) ? -angle : angle ;
#else
  return std::atan2(y,x) ;
#endif
}

/** @brief Fast @c resqrt
 ** @param x argument.
 ** @return Approximation to @c resqrt(x).
 **/
inline
float
fast_resqrt(float x)
{
#ifdef VL_USEFASTMATH
  // Works if VL::float_t is 32 bit ...
  union {
    float x ;
    VL::int32_t i ;
  } u ;
  float xhalf = float(0.5) * x ;
  u.x = x ;                               // get bits for floating value
  u.i = 0x5f3759df - (u.i>>1);            // gives initial guess y0
  //u.i = 0xdf59375f - (u.i>>1);          // gives initial guess y0
  u.x = u.x*(float(1.5) - xhalf*u.x*u.x); // Newton step (may repeat)
  u.x = u.x*(float(1.5) - xhalf*u.x*u.x); // Newton step (may repeat)
  return u.x ;
#else
  return float(1.0) / std::sqrt(x) ;
#endif
}

/** @brief Fast @c resqrt
 ** @param x argument.
 ** @return Approximation to @c resqrt(x).
 **/
inline
double
fast_resqrt(double x)
{
#ifdef VL_USEFASTMATH
  // Works if double is 64 bit ...
  union {
    double x ;
    VL::int64_t i ;
  } u ;
  double xhalf = double(0.5) * x ;
  u.x = x ;                                // get bits for floating value
  u.i = 0x5fe6ec85e7de30daLL - (u.i>>1);   // gives initial guess y0
  u.x = u.x*(double(1.5) - xhalf*u.x*u.x); // Newton step (may repeat)
  u.x = u.x*(double(1.5) - xhalf*u.x*u.x); // Newton step (may repeat)
  return u.x ;
#else
  return double(1.0) / std::sqrt(x) ;
#endif
}

/** @brief Fast @c sqrt
 ** @param x argument.
 ** @return Approximation to @c sqrt(x).
 **/
inline
VL::float_t
fast_sqrt(VL::float_t x)
{
#ifdef VL_USEFASTMATH
  return (x < 1e-8) ? 0 : x * fast_resqrt(x) ;
#else
  return std::sqrt(x) ;
#endif
}

}

// Emacs:
// Local Variables:
// mode: C++
// End:
