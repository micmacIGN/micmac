// file:        sift-conv.tpp
// author:      Andrea Vedaldi
// description: Sift template definitions

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

template<typename T>
void
normalize(T* filter, int W)
{
  T  acc  = 0 ; 
  T* iter = filter ;
  T* end  = filter + 2*W+1 ;
  while(iter != end) acc += *iter++ ;
  
  iter = filter ;
  while(iter != end) *iter++ /= acc ;
}

template<typename T>
void
convolve(T*       dst_pt, 
         const T* src_pt, int M, int N,
         const T* filter_pt, int W)
{
  typedef T const TC ;
  // convolve along columns, save transpose
  // image is M by N 
  // buffer is N by M 
  // filter is (2*W+1) by 1
  for(int j = 0 ; j < N ; ++j) {
    
    int i = 0 ;

    // top
    for(; i <= std::min(W-1, M-1) ; ++i) {
      TC* start = src_pt ;
      TC* stop  = src_pt    + std::min(i+W, M-1) + 1 ;
      TC* g     = filter_pt + W-i ;
      T   acc = 0.0 ;
      while(stop != start) acc += (*g++) * (*start++) ;
      *dst_pt = acc ;
      dst_pt += N ;
    }

    // middle
    // run this for W <= i <= M-1-W, only if M >= 2*W+1
    for(; i <= M-1-W ; ++i) {
      TC* start = src_pt    + i-W ;
      TC* stop  = src_pt    + i+W + 1 ;
      TC* g     = filter_pt ;
      T   acc = 0.0 ;
      while(stop != start) acc += (*g++) * (*start++) ;
      *dst_pt = acc ;
      dst_pt += N ;
    }

    // bottom
    // run this for M-W <= i <= M-1, only if M >= 2*W+1
    for(; i <= M-1 ; ++i) {
      TC* start = src_pt    + i-W ;
      TC* stop  = src_pt    + std::min(i+W, M-1) + 1 ;
      TC* g     = filter_pt ;
      T   acc   = 0.0 ;
      while(stop != start) acc += (*g++) * (*start++) ;
      *dst_pt = acc ;
      dst_pt += N ;
    }
    
    // next column
    src_pt += M ;
    dst_pt -= M*N - 1 ;
  }
}

// works with symmetric filters only
template<typename T>
void
nconvolve(T*       dst_pt, 
          const T* src_pt, int M, int N,
          const T* filter_pt, int W,
          T*       scratch_pt )
{
  typedef T const TC ;

  for(int i = 0 ; i <= W ; ++i) {
    T   acc = 0.0 ;
    TC* iter = filter_pt + std::max(W-i,  0) ;
    TC* stop = filter_pt + std::min(M-1-i,W) + W + 1 ;
    while(iter != stop) acc += *iter++ ;
    scratch_pt [i] = acc ;
  }

 for(int j = 0 ; j < N ; ++j) {
    
   int i = 0 ;
   // top margin
   for(; i <= std::min(W, M-1) ; ++i) {
     TC* start = src_pt ;
     TC* stop  = src_pt    + std::min(i+W, M-1) + 1 ;
     TC* g     = filter_pt + W-i ;
     T   acc = 0.0 ;
     while(stop != start) acc += (*g++) * (*start++) ;
     *dst_pt = acc / scratch_pt [i] ;
     dst_pt += N ;
   }
   
   // middle
   for(; i <= M-1-W ; ++i) {
     TC* start = src_pt    + i-W ;
     TC* stop  = src_pt    + i+W + 1 ;
     TC* g     = filter_pt ;
     T   acc = 0.0 ;
     while(stop != start) acc += (*g++) * (*start++) ;
     *dst_pt = acc ;
     dst_pt += N ;
   }

   // bottom
   for(; i <= M-1 ; ++i) {
     TC* start = src_pt    + i-W ;
     TC* stop  = src_pt    + std::min(i+W, M-1) + 1 ;
     TC* g     = filter_pt ;
     T   acc   = 0.0 ;
     while(stop != start) acc += (*g++) * (*start++) ;
     *dst_pt = acc / scratch_pt [M-1-i];
     dst_pt += N ;
   }
   
   // next column
   src_pt += M ;
   dst_pt -= M*N - 1 ;
 }
}

template<typename T>
void
econvolve(T*       dst_pt, 
	  const T* src_pt,    int M, int N,
	  const T* filter_pt, int W)
{
  typedef T const TC ;
  // convolve along columns, save transpose
  // image is M by N 
  // buffer is N by M 
  // filter is (2*W+1) by 1
  for(int j = 0 ; j < N ; ++j) {
    for(int i = 0 ; i < M ; ++i) {
      T   acc = 0.0 ;
      TC* g = filter_pt ;
      TC* start = src_pt + (i-W) ;
      TC* stop  ;
      T   x ;

      // beginning
      stop = src_pt + std::max(0, i-W) ;
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

      assert( g - filter_pt == 2*W+1 ) ;

    }
    // next column
    src_pt += M ;
    dst_pt -= M*N - 1 ;
  }
}



// Emacs:
// Local Variables:
// mode: C++
// End:
