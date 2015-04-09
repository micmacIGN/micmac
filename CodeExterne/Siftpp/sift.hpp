// file:        sift.hpp
// author:      Andrea Vedaldi
// description: Sift declaration

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

#ifndef VL_SIFT_HPP
#define VL_SIFT_HPP

#include<valarray>
#include<vector>
#include<ostream>
#include<cmath>
#include<limits>

#if defined (VL_USEFASTMATH)
#if defined (VL_MAC)
#define VL_FASTFLOAT float
#else
#define VL_FASTFLOAT double
#endif
#else
#define VL_FASTFLOAT float
#endif

#define VL_XEAS(x) #x
#define VL_EXPAND_AND_STRINGIFY(x) VL_XEAS(x)

/** @brief VisionLab namespace */
namespace VL {

/** @brief Pixel data type */
typedef float pixel_t ;

/** @brief Floating point data type 
 **
 ** Although floats are precise enough for this applicatgion, on Intel
 ** based architecture using doubles for floating point computations
 ** turns out to be much faster.
 **/
typedef VL_FASTFLOAT float_t ;

/** @brief 32-bit floating data type */
typedef float float32_t ;

/** @brief 64-bit floating data type */
typedef double float64_t ;

/** @brief 32-bit integer data type */
typedef int int32_t ;

/** @brief 64-bit integer data type */
typedef long long int int64_t ;

/** @brief 32-bit unsigned integer data type */
typedef int uint32_t ;

/** @brief 8-bit unsigned integer data type */
typedef char unsigned uint8_t ;

/** @name Fast math
 ** 
 ** We provide approximate mathematical functions. These are usually
 ** rather faster than the corresponding standard library functions.
 **/
/*@{*/
float   fast_resqrt(float x) ;
double  fast_resqrt(double x) ;
float_t fast_expn(float_t x) ;
float_t fast_abs(float_t x) ;
float_t fast_mod_2pi(float_t x) ;
float_t fast_atan2(float_t y, float_t x) ;
float_t fast_sqrt(float_t x) ;
int32_t fast_floor(float_t x) ;
/*@}*/

/** @brief Generic exception */
struct
__attribute__ ((__visibility__("default")))
 Exception
{
  /** @brief Build generic exception with message
   ** 
   ** The message can be accessed as the Exception::msg data member.
   **
   ** @param _msg message.
   **/
  Exception(std::string _msg) : msg(_msg) { }

  /** Exception message */
  std::string msg ; 
} ;

/** @brief Throw generic exception
 **
 ** The macro executes the stream operations @a x to obtain
 ** an error messages. The message is then wrapped in a
 ** generic exception VL::Exception and thrown.
 **
 ** @param x sequence of stream operations.
 **/
#define VL_THROW(x)                             \
  {                                             \
    std::ostringstream oss ;                    \
    oss << x ;                                  \
    throw VL::Exception(oss.str()) ;            \
  }

/** @name PGM input/output */
/*@{*/
/** @brief PGM buffer descriptor
 **
 ** The structure describes a gray scale image and it is used by the
 ** PGM input/output functions. The fileds are self-explanatory.
 **/
struct PgmBuffer
{
  int width ;     ///< Image width
  int height ;    ///< Image hegith
  pixel_t* data ; ///< Image data
} ;
std::ostream& insertPgm(std::ostream&, pixel_t const* im, int width, int height) ;
std::istream& extractPgm(std::istream&, PgmBuffer& buffer) ;
/*@}*/

/** @brief SIFT filter
 **
 ** This class is a filter computing the Scale Invariant Feature
 ** Transform (SIFT).
 **/
class Sift
{

public:
  
  /** @brief SIFT keypoint
   **
   ** A SIFT keypoint is charactedized by a location x,y and a scale
   ** @c sigma. The scale is obtained from the level index @c s and
   ** the octave index @c o through a simple formula (see the PDF
   ** documentation).
   **
   ** In addition to the location, scale indexes and scale, we also
   ** store the integer location and level. The integer location is
   ** unnormalized, i.e. relative to the resolution of the octave
   ** containing the keypoint (octaves are downsampled). 
   **/
  struct Keypoint
  {
    int o ;    ///< Keypoint octave index

    int ix ;   ///< Keypoint integer X coordinate (unnormalized)
    int iy ;   ///< Keypoint integer Y coordinate (unnormalized)
    int is ;   ///< Keypoint integer scale indiex

    float_t x  ;  ///< Keypoint fractional X coordinate
    float_t y  ;  ///< Keypoint fractional Y coordinate
    float_t s ;   ///< Keypoint fractional scale index

    float_t sigma ;  ///< Keypoint scale
  } ; 

  typedef std::vector<Keypoint>     Keypoints ;          ///< Keypoint list datatype
  typedef Keypoints::iterator       KeypointsIter ;      ///< Keypoint list iter datatype
  typedef Keypoints::const_iterator KeypointsConstIter ; ///< Keypoint list const iter datatype

  /** @brief Constructors and destructors */
  /*@{*/
  Sift(const pixel_t* _im_pt, int _width, int _height,
       float_t _sigman,
       float_t _sigma0,
       int _O, int _S,
       int _omin, int _smin, int _smax) ;
  ~Sift() ;
  /*@}*/

  void process(const pixel_t* _im_pt, int _width, int _height) ;

  /** @brief Querying the Gaussian scale space */
  /*@{*/
  VL::pixel_t* getOctave(int o) ;
  VL::pixel_t* getLevel(int o, int s) ;
  int          getWidth() const ;
  int          getHeight() const ;
  int          getOctaveWidth(int o) const ;
  int          getOctaveHeight(int o) const ;
  VL::float_t  getOctaveSamplingPeriod(int o) const ;
  VL::float_t  getScaleFromIndex(VL::float_t o, VL::float_t s) const ;
  Keypoint     getKeypoint(VL::float_t x, VL::float_t y, VL::float_t s) const ;
  /*@}*/

  /** @brief Descriptor parameters */
  /*@{*/
  bool getNormalizeDescriptor() const ;
  void setNormalizeDescriptor(bool) ;
  void setMagnification(VL::float_t) ;
  VL::float_t getMagnification() const ;  
  /*@}*/

  /** @brief Detector and descriptor */
  /*@{*/
  void detectKeypoints(VL::float_t threshold, VL::float_t edgeThreshold) ;
  int computeKeypointOrientations(VL::float_t angles [4], Keypoint keypoint) ; 
  void computeKeypointDescriptor(VL::float_t* descr_pt, Keypoint keypoint, VL::float_t angle) ;
  KeypointsIter keypointsBegin() ;
  KeypointsIter keypointsEnd() ;
  /*@}*/
    
private:
  void prepareBuffers() ;
  void freeBuffers() ;
  void smooth(VL::pixel_t       * dst, 
	      VL::pixel_t       * temp, 
              VL::pixel_t const * src, int width, int height, 
              VL::float_t s) ;

  void prepareGrad(int o) ;
  
  // scale space parameters
  VL::float_t sigman ;
  VL::float_t sigma0 ;
  VL::float_t sigmak ;

  int O ;
  int S ; 
  int omin ;
  int smin ; 
  int smax ;

  int width ;
  int height ;

  // descriptor parameters
  VL::float_t magnif ;
  bool        normalizeDescriptor ;

  // buffers
  VL::pixel_t*  temp ;
  int           tempReserved ;
  bool          tempIsGrad  ;
  int           tempOctave ;
  VL::pixel_t** octaves ;
  
  VL::pixel_t*  filter ;
  int           filterReserved ;

  Keypoints keypoints ;  
} ;


}

// Include inline functions definitions
#include<sift.ipp>

// VL_SIFT_HPP
#endif
