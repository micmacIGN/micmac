/*! \file dpoint.hpp
    \brief d-dimensional point class
    
    A d-dimensional point class which is written carefully using templates. It allows for basic
    operations on points in any dimension. Orientation tests for 2 and 3 dimensional points are 
    supported using <a href="http://www.cs.berkeley.edu/~jrs/">Jonathan's</a> code. This class 
    forms the building block of other classes like dplane, dsphere etc.

    \author <a href="http://www.compgeom.com/~piyush">Piyush Kumar</a>
    \bug    No known bugs.
 */

#ifndef REVIVER_POINT_HPP
#define REVIVER_POINT_HPP

// mrkkrj
#ifndef TRPP_BUILD_SHARED
// mrkkrj - legacy usage
#include "tpp_assert.hpp"
#else
// mrkkrj - HACK:: we don't want export Assert function aout of DLL!
#include <cassert>
#define Assert(a, b) assert(a && b)
#endif

#include <iostream>
#include <valarray>
#include <stdio.h>
#include <limits>


//! The reviver namespace signifies the part of the code borrowed from reviver (dpoint.hpp). 
namespace reviver {


// Forward Declaration of the main Point Class
// Eucledian d-dimensional point. The distance is L_2

template<typename NumType, unsigned D>
class dpoint;


///////////////////////////////////////////////////////
// Internal number type traits for dpoint
///////////////////////////////////////////////////////

template<typename T> 
class InternalNumberType;

template<>
class InternalNumberType<float>{
public: 
  typedef double __INT;
};


template<>
class InternalNumberType<int>{
public: 
  typedef long long __INT;
};


template<>
class InternalNumberType<double>{
public: 
  typedef double __INT;
};

template<>
class InternalNumberType<long>{
public: 
  typedef long long __INT;
};


///////////////////////////////////////////////////////
// Origin of d-dimensional point
///////////////////////////////////////////////////////
template< typename NumType, unsigned D, unsigned I > struct origin
{
   static inline void eval( dpoint<NumType,D>& p )
   {
      p[I] = 0.0;
      origin< NumType, D, I-1 >::eval( p );
   }
};


// Partial Template Specialization
template <typename NumType, unsigned D> struct origin<NumType, D, 0>
{
   static inline void eval( dpoint<NumType,D>& p )
   {
      p[0] = 0.0;
   }
};


//! A structure to compute squared distances between points
/*!
    Uses unrolling of loops using templates.
*/
///////////////////////////////////////////////////////
// Squared Distance of d-dimensional point
///////////////////////////////////////////////////////
template< typename NumType, unsigned D, unsigned I > struct Distance
{
   static inline double eval( const dpoint<NumType,D>& p, const dpoint<NumType,D>& q )
   {
     double sum = ((double)p[I] -  (double)q[I] ) *( (double)p[I] - (double)q[I] );
      return sum + Distance< NumType, D, I-1 >::eval( p,q );
   }
};


//!  Partial Template Specialization for distance calculations
template <typename NumType, unsigned D> struct Distance<NumType, D, 0>
{
   static inline double eval( const dpoint<NumType,D>& p, const dpoint<NumType,D>& q )
   {
     return ((double) p[0] - (double)q[0] )*( (double)p[0] - (double)q[0] );
   }
};


//! A structure to compute dot product between two points or associated vectors
/*!
    Uses unrolling of loops using templates.
*/
///////////////////////////////////////////////////////
// Dot Product of two d-dimensional points
///////////////////////////////////////////////////////
template< typename __INT, typename NumType, unsigned D, unsigned I > struct DotProd
{
   static inline __INT eval( const dpoint<NumType,D>& p, const dpoint<NumType,D>& q )
   {
      __INT sum = ( ((__INT)p[I]) * ((__INT)q[I]) );
      return sum + DotProd< __INT, NumType, D, I-1 >::eval( p,q );
   }
};


//! Partial Template Specialization for dot product calculations
template < typename __INT, typename NumType, unsigned D> struct DotProd<__INT,NumType, D, 0>
{
   static inline __INT eval( const dpoint<NumType,D>& p, const dpoint<NumType,D>& q )
   {
      return ( ((__INT)p[0]) * ((__INT)q[0]) );
   }
};


///////////////////////////////////////////////////////
// Equality of two d-dimensional points
///////////////////////////////////////////////////////
template< typename NumType, unsigned D, unsigned I > struct IsEqual
{
   static inline bool eval( const dpoint<NumType,D>& p, const dpoint<NumType,D>& q )
   {
      if( p[I]  != q[I] ) return false;
      else return IsEqual< NumType, D, I-1 >::eval( p,q );
   }
};


// Partial Template Specialization
template <typename NumType, unsigned D> struct IsEqual<NumType, D, 0>
{
   static inline NumType eval( const dpoint<NumType,D>& p, const dpoint<NumType,D>& q )
   {
       return (p[0] == q[0])?1:0;
   }
};



//!   Equate two d-dimensional points. 
/*!
    Uses unrolling of loops using templates. 
    A class used to implement operator= for points. This class also helps in automatic type 
    conversions of points (with explicit calls for conversion).
*/
template< 
  typename NumType1, 
  typename NumType2, 
  unsigned D, 
  unsigned I 
  > struct Equate
{
   static inline void eval( dpoint<NumType1,D>& p,const dpoint<NumType2,D>& q )
   {
      p[I]  = q[I];
      Equate< NumType1, NumType2, D, I-1 >::eval( p,q );
   }
};


//! Partial Template Specialization for Equate
template <
  typename NumType1, 
  typename NumType2,
  unsigned D
  > struct Equate<NumType1,NumType2, D, 0>
{
   static inline void  eval( dpoint<NumType1,D>& p,const dpoint<NumType2,D>& q )
   {
       p[0] = q[0];
   }
};


//! A structure to add two points
/*!
    Uses unrolling of loops using templates.
*/
///////////////////////////////////////////////////////
// Add two d-dimensional points
///////////////////////////////////////////////////////
template< typename NumType, unsigned D, unsigned I > struct Add
{
   static inline void eval( dpoint<NumType,D>& result, const dpoint<NumType,D>& p, const dpoint<NumType,D>& q )
   {
      result[I] = p[I]  + q[I];
      Add< NumType, D, I-1 >::eval( result,p,q );
   }
};


//! Partial Template Specialization for Add structure
template <typename NumType, unsigned D> struct Add<NumType, D, 0>
{
   static inline void eval( dpoint<NumType,D>& result, const dpoint<NumType,D>& p, const dpoint<NumType,D>& q )
   {
       result[0] = p[0] + q[0];
   }
};


///////////////////////////////////////////////////////
// Subtract two d-dimensional points
///////////////////////////////////////////////////////
// Could actually be done using scalar multiplication and addition


// What about unsigned types?
template< typename NumType > 
inline NumType Subtract_nums(const NumType& x, const NumType& y) {
  if(!std::numeric_limits<NumType>::is_signed) {
      std::cerr << "Exception: Can't subtract unsigned types."; exit(1);
  }
  return x - y;
}


//!   Subtract two d-dimensional vectors
/*!
      Caution: Do not use on unsigned types.
*/
template< typename NumType, unsigned D, unsigned I > struct Subtract
{
   static inline void eval( dpoint<NumType,D>& result, const dpoint<NumType,D>& p, const dpoint<NumType,D>& q )
   {
     
          result[I] = Subtract_nums(p[I] , q[I]);
      Subtract< NumType, D, I-1 >::eval( result,p,q );
   }
};


//! Partial Template Specialization for subtraction of points (associated vectors)
template <typename NumType, unsigned D> struct Subtract<NumType, D, 0>
{
   static inline void eval( dpoint<NumType,D>& result, const dpoint<NumType,D>& p, const dpoint<NumType,D>& q )
   {
       result[0] = Subtract_nums(p[0] , q[0]);
   }
};





//!   Mutiply scalar with d-dimensional point
/*!
      Scalar mulipltication of d-dimensional point with a number using template unrolling.
*/
template< typename NumType, unsigned D, unsigned I > struct Multiply
{
   static inline void eval( dpoint<NumType,D>& result, const dpoint<NumType,D>& p, NumType k)
   {
      result[I] = p[I] * k;
      Multiply< NumType, D, I-1 >::eval( result,p,k );
   }
};


//! Partial Template Specialization for scalar multiplication
template <typename NumType, unsigned D> struct Multiply<NumType, D, 0>
{
   static inline void eval( dpoint<NumType,D>& result, const dpoint<NumType,D>& p, NumType k )
   {
       result[0] = p[0] * k;
   }
};



//!  Main d dimensional Point Class
/*!
    -  NumType = Floating Point Type
    -  D       = Dimension of Point
*/
template<typename NumType = double, unsigned D = 3>
class dpoint {

        // Makes Swap operation fast
        NumType  x[D];

public:
        typedef NumType NT;
        typedef typename InternalNumberType<NumType>::__INT __INT;

    // To be defined in a cpp file
    //  const MgcVector2 MgcVector2::ZERO(0,0);
    //  static const dpoint<NumType,D> Zero;

    inline void move2origin(){ origin<NumType, D, D-1>::eval(*this); };

    dpoint(){ 
        Assert( (D >= 1), "Dimension < 1 not allowed" ); 
        // move2origin(); 
    };

    //! 1 D Point
    dpoint(NumType x0){ x[0] = x0; };
    //! 2 D Point
    dpoint(NumType x0,NumType x1){ x[0] = x0;  x[1] = x1; };
    //! 3 D Point
    dpoint(NumType x0,NumType x1,NumType x2){  x[0] = x0;  x[1] = x1; x[2] = x2; };
    //! Array Initialization
    dpoint(NumType ax[]){ for(unsigned int i =0; i < D; ++i) x[i] = ax[i]; };
    //! Initialization from another point : Copy Constructor
        dpoint(const dpoint<NumType,D>& p){  Equate<NumType,NumType,D,D-1>::eval((*this),p);	};

         
    //! Automatic type conversions of points.
    //! Only allowed if the conversion is specified explicitly by the programmer.
    template<class OtherNumType>
        explicit dpoint(const dpoint<OtherNumType,D>& p){ Equate<NumType,OtherNumType,D,D-1>::eval((*this),p); };

    // Destructor
    ~dpoint(){};

    inline int      dim() const { return D; };
    inline double   sqr_dist(const dpoint<NumType,D> q) const ;
    inline double   distance(const dpoint<NumType,D> q) const ;
    inline __INT    dotprod (const dpoint<NumType,D> q) const ;
    inline __INT    sqr_length(void)  const;
    inline void     normalize (void);
    inline NumType& operator[](int i);
    inline NumType  operator[](int i) const;

    inline dpoint&  operator= (const dpoint<NumType,D>& q);

    template<typename NT, unsigned __DIM>
    friend dpoint<NT,__DIM>   operator- (const dpoint<NT,__DIM>& p, const dpoint<NT,__DIM>& q);

    template<typename NT, unsigned __DIM>
    friend dpoint<NT,__DIM>   operator+ (const dpoint<NT,__DIM>& p, const dpoint<NT,__DIM>& q);

    template<typename NT, unsigned __DIM>
    friend bool   operator== (const dpoint<NT,__DIM>& p, const dpoint<NT,__DIM>& q);

    template<typename NT, unsigned __DIM>
    friend bool   operator!= (const dpoint<NT,__DIM>& p, const dpoint<NT,__DIM>& q);


//	inline dpoint&  operator= (const valarray<NumType>& v);
//	inline operator valarray<NumType>() const;

    template<typename __NT,unsigned __DIM>
    friend void iswap(dpoint<__NT,__DIM>& p,dpoint<__NT,__DIM>& q);
};

template<typename NumType, unsigned D>
void dpoint<NumType,D>::normalize (void){
    double len = sqrt(sqr_length());
    if (len > 0.00001)
    for(int i = 0; i < D; ++i){
        x[i] /= len;
    }
}

/*
template<typename NumType, unsigned D>
dpoint<NumType,D>::operator valarray<NumType>() const{
    valarray<NumType> result((*this).x , D);
    return result;
}

//Warning : Valarray should be of size D
//TODO: Unwind this for loop into a template system
template<typename NumType, unsigned D>
dpoint<NumType,D>&
dpoint<NumType,D>::operator= (const valarray<NumType>& v){
    dpoint<NumType,D> result;
    for(int i = 0; i < D; i++) (*this).x[i] = v[i];
    return (*this);
}
*/

template<typename NT, unsigned __DIM>
dpoint<NT,__DIM>
operator+ (const dpoint<NT,__DIM>& p, const dpoint<NT,__DIM>& q){
    dpoint<NT,__DIM> result;
    Add<NT,__DIM,__DIM-1>::eval(result,p,q);	
    return result;
}

template<typename NT, unsigned __DIM>
dpoint<NT,__DIM>
operator- (const dpoint<NT,__DIM>& p, const dpoint<NT,__DIM>& q){
    dpoint<NT,__DIM> result;
    // cout << "Subtracting..." << p << " from " << q << " = ";
    Subtract<NT,__DIM,__DIM-1>::eval(result,p,q);	
    // cout << result << endl;	
    return result;
}

template<typename NT, unsigned __DIM>
bool
operator== (const dpoint<NT,__DIM>& p, const dpoint<NT,__DIM>& q){
    return IsEqual<NT,__DIM,__DIM-1>::eval(p,q);	
}

template<typename NT, unsigned __DIM>
bool
operator!= (const dpoint<NT,__DIM>& p, const dpoint<NT,__DIM>& q){
    return !(IsEqual<NT,__DIM,__DIM-1>::eval(p,q));	
}

template<typename NT, unsigned __DIM>
dpoint<NT,__DIM>
operator* (const dpoint<NT,__DIM>& p, const NT k){
    dpoint<NT,__DIM> result;
    Multiply<NT,__DIM,__DIM-1>::eval(result,p,k);	
    return result;
}

template<typename NT, unsigned __DIM>
dpoint<NT,__DIM>
operator/ (const dpoint<NT,__DIM>& p, const NT k){
    Assert( (k != 0), "Hell division by zero man...\n");
    dpoint<NT,__DIM> result;
    Multiply<NT,__DIM,__DIM-1>::eval(result,p,((double)1.0)/k);	
    return result;
}

template < typename NumType, unsigned D >
dpoint<NumType,D>&
dpoint<NumType,D>::operator=(const dpoint<NumType,D> &q)
{
  Assert((this != &q), "Error p = p");
  Equate<NumType,NumType,D,D-1>::eval(*this,q);	
  return *this;
}

template < typename NumType, unsigned D >
NumType
dpoint<NumType,D>::operator[](int i) const
{ return x[i]; }

template < typename NumType, unsigned D >
NumType&
dpoint<NumType,D>::operator[](int i)
{
  return x[i]; 
}


template<typename NumType, unsigned D>
double
dpoint<NumType,D>::sqr_dist (const dpoint<NumType,D> q) const {
    return Distance<NumType,D,D-1>::eval(*this,q);	
}

template<typename NumType, unsigned D>
double 
dpoint<NumType,D>::distance (const dpoint<NumType,D> q) const {
    return sqrt(static_cast<double>(Distance<NumType,D,D-1>::eval(*this,q)));	
}


template<typename NumType, unsigned D>
typename dpoint<NumType,D>::__INT
dpoint<NumType,D>::dotprod (const dpoint<NumType,D> q) const {
    return DotProd<__INT,NumType,D,D-1>::eval(*this,q);	
}

template<typename NumType, unsigned D>
typename dpoint<NumType,D>::__INT
dpoint<NumType,D>::sqr_length (void) const {
#ifdef _DEBUG	
    if( DotProd<__INT,NumType,D,D-1>::eval(*this,*this) < 0) {
          std::cerr << "Point that caused error: ";
      std::cerr << *this << std::endl;
      std::cerr << DotProd<__INT,NumType,D,D-1>::eval(*this,*this) << std::endl;
      std::cerr << "Fatal: Hell!\n"; exit(1);
    }
#endif
    return DotProd<__INT,NumType,D,D-1>::eval(*this,*this);	
    
}

template < class NumType, unsigned D >
std::ostream&
operator<<(std::ostream& os,const dpoint<NumType,D> &p)
{
     os << "Point (d=";
     os << D << ", (";
     for (unsigned int i=0; i<D-1; ++i)
        os << p[i] << ", ";
    return os << p[D-1] << "))";
    
};

template < class NumType, unsigned D >
std::istream&
operator>>(std::istream& is,dpoint<NumType,D> &p)
{
     for (int i=0; i<D; ++i)
         if(!(is >> p[i])){
             if(!is.eof()){
                std::cerr << "Error Reading Point:" 
#ifndef _WIN32
                      /*<< is*/ << std::endl;   // ---> OPEN TODO::: not compiling with gcc!
#else

                    // OPEN TODO::::
                      // << is << std::endl; <-- Microsoft OK!

                      << static_cast<bool>(is) << std::endl; // <-- MinGW!
#endif
                exit(1);
             }
         }
         
    return is;
    
};

/*
template<typename __NT,unsigned __DIM>
static inline void iswap(dpoint<__NT,__DIM>& p,dpoint<__NT,__DIM>& q){
    __NT *y;
    y = p.x;
    p.x = q.x;
    q.x = y;
}
*/



template < typename NumType, unsigned D >
dpoint<NumType, D> CrossProd(const dpoint<NumType, D>& vector1, 
                 const dpoint<NumType, D>& vector2) {
   Assert(D == 3, "Cross product only defined for 3d vectors");
   dpoint<NumType, D> vector;
   vector[0] = (vector1[1] * vector2[2]) - (vector2[1] * vector1[2]);
   vector[1] = (vector2[0] * vector1[2]) - (vector1[0] * vector2[2]);
   vector[2] = (vector1[0] * vector2[1]) - (vector2[0] * vector1[1]); 
   return vector;
}




template < typename __NT, unsigned __DIM >
int
orientation(const dpoint<__NT,__DIM> p[__DIM+1])
{
    int _sign = + 1;
    // To be implemented
    std::cerr << "Not yet implemented\n";
    exit(1);
    return _sign;
    
}


template < typename __NT >
inline __NT
orientation(
        const dpoint<__NT,2>& p,
        const dpoint<__NT,2>& q,
        const dpoint<__NT,2>& r
        )
{
   // 2D speaciliazation for orientation
    std::cout << "FATAL";
  exit(1);
  return ((p[0]-r[0])*(q[1]-r[1]))-((q[0]-r[0])*(p[1]-r[1]));
}


extern "C" double orient2d(double *p, double *q, double *r);

template < >
inline double
orientation<double>(
        const dpoint<double,2>& p,
        const dpoint<double,2>& q,
        const dpoint<double,2>& r
        )
{
   // 2D speaciliazation for orientation
  double pp[2] = { p[0], p[1] };
  double qq[2] = { q[0], q[1] };
  double rr[2] = { r[0], r[1] };
  return orient2d(pp,qq,rr);
}


template < >
inline float
orientation<float>(
        const dpoint<float,2>& p,
        const dpoint<float,2>& q,
        const dpoint<float,2>& r
        )
{
   // 2D speaciliazation for orientation
  double pp[2] = { p[0], p[1] };
  double qq[2] = { q[0], q[1] };
  double rr[2] = { r[0], r[1] };
  return (float)orient2d(pp,qq,rr);
}



};    // Namespace Ends here




#endif


