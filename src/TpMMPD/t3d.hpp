#ifndef _T3D_
#define _T3D_

#include <iostream>
#include <vector>
#include <cmath>

#ifdef _T2D_
template<class T> class T2D;
#endif

template<class T> 
class T3D{
protected:
  T _x;
  T _y;
  T _z;
  
public:
  //Constructeurs et destructeurs
  T3D():_x(0),_y(0),_z(0){};
  T3D(const T v):_x(v),_y(v),_z(v){};
  T3D(const T X, const T Y, const T Z):_x(X),_y(Y),_z(Z){};
  T3D(const T3D & src):_x(src.x()),_y(src.y()),_z(src.z()){}
#ifdef _T2D_
  T3D(const T2D<T> & XY, const T Z):_x(XY.x()),_y(XY.y()),_z(Z){}
#endif
  template<class V> T3D(const V X,const V Y, const V Z):_x((T)X),_y((T)Y),_z((T)Z){};
  template<class V> T3D(const T3D<V> & src):_x((T)src.x()),_y((T)src.y()),_z((T)src.z()){}
  template<class V> T3D(const std::vector<V> & src):_x((T)src[0]),_y((T)src[1]),_z((T)src[2]){}
  virtual ~T3D(){};
  
  const unsigned int dim() const{return 3;}
  const unsigned int size() const{return 3;}

  //typedef
  typedef T type;


  //Accesseurs
  inline const T x() const{return _x;} 
  inline T & x() {return _x;}
  inline const T y() const{return _y;}
  inline T & y() {return _y;}
  inline const T z() const{return _z;}
  inline T & z() {return _z;}

  inline const T p1() const{return _x;} 
  inline T & p1() {return _x;}
  inline const T p2() const{return _y;}
  inline T & p2() {return _y;}
  inline const T p3() const{return _z;}
  inline T & p3() {return _z;}

  const T operator [] (const int n) const {
    if (n==0) return x() ;
    if (n==1) return y() ;
    if (n==2) return z() ;
    return x() ;
  }
  T & operator [] (const int n) {
    if (n==0) return x() ;
    if (n==1) return y() ;
    if (n==2) return z() ;
    return x() ;
  }

  //Conserv� pour compatibilit�
  template<class V> inline void setx(const V v) { x()=(T)v;}
  template<class V> inline void sety(const V v) { y()=(T)v;}
  template<class V> inline void setz(const V v) { z()=(T)v;}
  inline void set (const T a, const T b, const T c) {x()=a; y()=b; z()=c;}
  template<class V> inline void set (const T3D<V> & p) {x()=p.x(); y()=p.y(); z()=p.z();}
  template<class V> inline void set (const V a, const V b, const V c) {x()=a; y()=b; z()=c;}
  template<class V> inline void set (const std::vector<V> & v) {x()=v[0]; y()=v[1]; z()=v[2];}
  
  
  //Pour travailler en double
  inline const double X() const{return (double)x();}
  inline const double Y() const{return (double)y();}
  inline const double Z() const{return (double)z();}
  inline const T3D<double> todouble() const{return T3D<double>(X(),Y(),Z());}
  inline const std::vector<T> tovector() const{std::vector<T> v(3); v[0]=x(); v[1]=y(); v[2]=z(); return v;}


#ifdef _T2D_
  inline const T2D<T> xy() const{return T2D<T>(_x,_y);} 
  inline const T2D<T> xz() const{return T2D<T>(_x,_z);} 
  inline const T2D<T> yz() const{return T2D<T>(_y,_z);} 
#endif

  //Pour travailler avec autre chose
  template<class V> void cast(T3D<V> & p) const{p.set(x(),y(),z());}
  template<class V> void cast(std::vector<V> & p) const{p.resize(3); p[0]=x(); p[1]=y(); p[2]=z();}
#ifdef _T2D_
  //  const T2D<T> T2D() const{return T2D<T>(x(),y());}
#endif


  //Produit scalaire
  const double operator * (const T3D & p) const {return X()*p.X()+Y()*p.Y()+Z()*p.Z();}
  // Produit vectoriel (d�terminant)
  const T3D<double> operator ^ (const T3D & p) const {
    return T3D<double>(Y()*p.Z()-p.Y()*Z(),p.X()*Z()-X()*p.Z(),X()*p.Y()-p.X()*Y());
  }

  const double det(const T3D<T> & A, const T3D<T> & B, const T3D<T> & C) const{
    return A*(B^C);
  }

  const T3D operator - () const {
    return T3D<T> (-_x, -_y, -_z) ;
  }
  const T3D operator - (const T3D & p) const {
    return T3D<T> (_x-p.x(),_y-p.y(),_z-p.z()) ;
  }
  const T3D operator + (const T3D & p) const {
    return T3D<T> (_x+p.x(),_y+p.y(),_z+p.z()) ;
  }
  const T3D operator / (const T p) const {
    return T3D<T> (_x/p,_y/p,_z/p) ;
  }
  const T3D operator * (const T p) const {
    return T3D<T> (_x*p,_y*p,_z*p) ;
  }

  const double norme() const{
    return sqrt(X()*X()+Y()*Y()+Z()*Z());
  }
  const double norme2() const{
    return X()*X()+Y()*Y()+Z()*Z();
  }

  void normaliser(){
    double d=norme();
    if(d!=0){
      x()=(T)((double)x()/d);
      y()=(T)((double)y()/d);
      z()=(T)((double)z()/d);
    }
  }
  void normalise(){normaliser();}

  const bool operator== (const T3D & pt) const{
    return _x==pt._x && _y==pt._y && _z==pt._z;
  }
  const bool operator!= (const T3D & pt) const{
    return !((*this)==pt);
  }
  const bool operator < (const T3D<T> & p) const { return ( (x()<p.x()) || ((x()==p.x())&&(y()<p.y())) || ((x()==p.x())&&(y()==p.y())&&(z()<p.z())) ) ; }
  const bool operator > (const T3D<T> & p) const { return ( (x()>p.x()) || ((x()==p.x())&&(y()>p.y())) || ((x()==p.x())&&(y()==p.y())&&(z()>p.z())) ) ; }


};

//-----------------------------------------------------------------------------
// Fonctions vectorielles
//-----------------------------------------------------------------------------
// Produit Scalaire
template<class T>
inline const double prodScal(const T3D<T> & A, const T3D<T> & B){
  return A.X()*B.X()+A.Y()*B.Y()+A.Z()*B.Z();
}

// Produit Vectoriel
template<class T>
inline const T3D<double> prodVect(const T3D<T> & A, const T3D<T> & B){
  double x = A.Y()*B.Z() - B.Y()*A.Z();
  double y = B.X()*A.Z() - A.X()*B.Z();
  double z = A.X()*B.Y() - B.X()*A.Y();
  return T3D<double>(x,y,z);
}

// Produit Mixte
template<class T>
inline const double prodMixt(const T3D<T> & A, const T3D<T> & B, const T3D<T> & C){
  return prodScal(A,prodVect(B,C));
}

template <class T>
inline std::ostream& operator << ( std::ostream & os, const T3D<T> & p ) {
  return (os <<"("<<p.x()<<","<<p.y()<<","<<p.z()<<")");
}

template <class T>
inline const T3D<T> operator * ( const T & c, const T3D<T> & p ) {
  return T3D<T> (c*p.X(),c*p.Y(),c*p.Z());
}

#ifdef __MATRICE_HPP__
const T3D<double> operator * (const Matrice & m, T3D<double> const & t){TPoint3D<double> p(t.x(),t.y(),t.z()); p=m*p; return T3D<double>(p.x,p.y,p.z);}
//const T2D<double> operator * (const Matrice & m, T2D<double> const & t){TPoint2D<double> p(t.x(),t.y()); p=m*p; return T2D<double>(p.x,p.y);}
#endif

#endif
