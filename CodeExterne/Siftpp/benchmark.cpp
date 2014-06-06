#include"sift.hpp"

#include<iostream>
#include<iomanip>
#include<sys/time.h>

#define FLOOR(x)   std::floor(x)
#define FMOD(x,y)  std::fmod(x,y)
#define SQRT(x)    std::sqrt(x)
#define ATAN2(y,x) std::atan2(y,x)
#define EXPN(x)    std::exp(-(x))

using namespace std ;
using namespace VL ;

long int
subtime(struct timeval const& a, struct timeval const& b)
{
  return (a.tv_sec - b.tv_sec) * 1000000 + a.tv_usec - b.tv_usec ;
}

int
main(int argc, char** argv)
{
  timeval time_start ;
  timeval time_stop ;
  
  // -----------------------------------------------------------------
  //                                                             speed
  // -----------------------------------------------------------------

  // compare fast_floor with floor
  gettimeofday(&time_start,0) ;
  for(VL::float_t x = - 100 ; x < 100 ; x += 1e-5) {
    VL::int32_t y = fast_floor( x ) ;
  }
  gettimeofday(&time_stop,0) ;  
  cout<<setw(20)<<"fast_floor : "<<subtime(time_stop,time_start)<<endl ;

  gettimeofday(&time_start,0) ;
  for(VL::float_t x = - 100 ; x < 100 ; x += 1e-5) {
    VL::int32_t y = VL::int32_t( FLOOR ( x )) ;
  }
  gettimeofday(&time_stop,0) ;  
  cout<<setw(20)<<"(int) floorf : "<<subtime(time_stop,time_start)<<endl ;

  // compare fast_mod_2pi
  gettimeofday(&time_start,0) ;
  for(VL::float_t x = - 4*M_PI; x < 2*M_PI ; x += 1e-5) {
    VL::float_t y = fast_mod_2pi(x) ;
  }
  gettimeofday(&time_stop,0) ;  
  cout<<setw(20)<<"fast_mod_2pi : "<<subtime(time_stop,time_start)<<endl ;

  gettimeofday(&time_start,0) ;
  for(VL::float_t x = - 4*M_PI; x < 2*M_PI ; x += 1e-5) {
    VL::float_t y = FMOD ( x, VL::float_t(2*M_PI) ) ;
  }
  gettimeofday(&time_stop,0) ;  
  cout<<setw(20)<<"fmod : "<<subtime(time_stop,time_start)<<endl ;
  
  // compare sqrt
  gettimeofday(&time_start,0) ;
  for(VL::float_t x = 0 ; x < 100 ; x += 1e-5) {
    VL::float_t y = fast_sqrt(x) ;
  }
  gettimeofday(&time_stop,0) ;  
  cout<<setw(20)<<"fast_sqrt : "<<subtime(time_stop,time_start)<<endl ;

  gettimeofday(&time_start,0) ;
  for(VL::float_t x = 0 ; x < 100 ; x += 1e-5) {
    VL::float_t y = SQRT ( x ) ;
  }
  gettimeofday(&time_stop,0) ;  
  cout<<setw(20)<<"sqrt : "<<subtime(time_stop,time_start)<<endl ;

  // compare atan2
  gettimeofday(&time_start,0) ;
  for(VL::float_t x = -100 ; x < 100 ; x += 1e-5) {
    VL::float_t y = fast_atan2(VL::float_t(1.0),x) ;
  }
  gettimeofday(&time_stop,0) ;  
  cout<<setw(20)<<"fast_atan2 : "<<subtime(time_stop,time_start)<<endl ;

  gettimeofday(&time_start,0) ;
  for(VL::float_t x = -100 ; x < 100 ; x += 1e-5) {
    VL::float_t y = ATAN2(VL::float_t(1.0),x) ;
  }
  gettimeofday(&time_stop,0) ;  
  cout<<setw(20)<<"atan2 : "<<subtime(time_stop,time_start)<<endl ;

  // compare epnx
  gettimeofday(&time_start,0) ;
  for(VL::float_t x = 0 ; x < 25.0 ; x += 1e-4) {
    VL::float_t y = fast_expn(x) ;
  }
  gettimeofday(&time_stop,0) ;  
  cout<<setw(20)<<"fast_epxn : "<<subtime(time_stop,time_start)<<endl ;

  gettimeofday(&time_start,0) ;
  for(VL::float_t x = 0 ; x < 25.0 ; x += 1e-4) {
    VL::float_t y = EXPN (x) ;
  }
  gettimeofday(&time_stop,0) ;  
  cout<<setw(20)<<"epx(-x) : "<<subtime(time_stop,time_start)<<endl ;


  // -----------------------------------------------------------------
  //                                                          accuracy
  // -----------------------------------------------------------------
  cout<<"fast_sqrt accuracy:" ;
  for(VL::float_t x = 0 ; x < 100 ; x += 5) {
    VL::float_t e = fast_abs( fast_sqrt(x) -  SQRT ( x ) ) ;
    cout<< " " << e / (x+1e-8) ;
  }
  cout<<endl ;
  
  cout<<"fast_atan2 accuracy:" ;
  for(VL::float_t x = -100 ; x < 100 ; x += 5) {
    VL::float_t e = fast_atan2(VL::float_t(1.0),x) - ATAN2 (VL::float_t(1.0),x) ;
    cout<< " " << e ;
  }
  cout<<endl ;
  cout<<"fast_atan2 accuracy:" ;
  for(VL::float_t x = -100 ; x < 100 ; x += 5) {
    VL::float_t e = fast_atan2(VL::float_t(-1.0),x) - ATAN2 (VL::float_t(-1.0),x) ;
    cout<< " " << e ;
  }
  cout<<endl ;

  cout<<"fast_expn accuracy:" ;
  for(VL::float_t x = 0 ; x < 25.0 ; x += 0.5) {
    VL::float_t e = fast_expn(x) - EXPN(x) ;
    cout<< " " << e ;
  }
  cout<<endl ;

  cout<<"fast_floor accuracy:" ;
  for(VL::float_t x = -2 ; x <= 2 ; x += 0.01) {
    VL::float_t e = fast_floor(x) - FLOOR(x) ;
    cout<< x<<" "<< e<<", " ;
  }
  cout<<endl ;

    
}
