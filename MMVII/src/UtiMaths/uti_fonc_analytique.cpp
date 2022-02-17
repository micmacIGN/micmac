#include "include/MMVII_all.h"

namespace MMVII
{

/*
template <typename Type> Type Fact(const Type & aTeta)
{
   return std::tgamma(aVal
}
*/
constexpr int Fact3 = 2 * 3;
constexpr int Fact5 = 2 * 3 * 4 * 5;  // Dont use Fact3, we dont control order of creation
constexpr int Fact7 = 2 * 3 * 4 * 5 * 6 * 7 ;
constexpr int Fact9 = 2 * 3 * 4 * 5 * 6 * 7 * 8 * 9;

template <typename Type> Type sinC(const Type & aTeta,const Type & aEps)
{
// x - x3/3! + x5/5! - x7/7! +

   if (std::abs(aTeta) > aEps)
      return std::sin(aTeta)/aTeta;
   Type aT2 = Square(aTeta);
   Type aT4 = Square(aT2);
   Type aT6 = aT4 * aT2;
   Type aT8 = Square(aT4);

   return 1.0 - aT2/Fact3 + aT4/Fact5 - aT6/Fact7 + aT8/Fact9;
}
template <typename Type> Type sinC(const Type & aTeta)
{
   return sinC(aTeta,tElemNumTrait<Type>::Accuracy());
}

template <typename Type> Type AtanXsY_sX(const Type & X,const Type & Y,const Type & aEps)
{
   if (std::abs(X) > aEps * std::abs(Y))
      return std::atan2(X,Y) / X;


   Type XsY2 = Square(X/Y);
   Type XsY4 = XsY2 * XsY2;
   Type XsY6 = XsY4 * XsY2;
   Type XsY8 = XsY4 * XsY4;

   return (1 -XsY2/3.0 + XsY4/5.0 -XsY6/7.0 + XsY8/9.0) / Y;
}
template <typename Type> Type AtanXsY_sX(const Type & X,const Type & Y)
{
    return AtanXsY_sX(X,Y,tElemNumTrait<Type>::Accuracy() );
}


template <typename Type> Type DerXAtanXsY_sX(const Type & X,const Type & Y,const Type & aEps)
{
   //  atan(x/y) /x  => -1/x2 atan(x/y) + 1/xy 1/(1+x/y^2)
   if (std::abs(X) > aEps * std::abs(Y))
      return -std::atan2(X,Y)/Square(X) +  Y/(X*(Square(X)+Square(Y)));

   Type XsY2 = Square(X/Y);
   Type XsY4 = XsY2 * XsY2;
   Type XsY6 = XsY4 * XsY2;
   Type XsY8 = XsY4 * XsY4;

   return (X/Cube(Y)) *( -(2.0/3.0) + (4.0/5.0)*XsY2 -(6.0/7.0)*XsY4 + (8.0/9.0) *XsY6 -(10.0/11.0)*XsY8);
}
template <typename Type> Type DerXAtanXsY_sX(const Type & X,const Type & Y)
{
   return DerXAtanXsY_sX(X,Y,tElemNumTrait<Type>::Accuracy());
}


template <typename Type> Type DerYAtanXsY_sX(const Type & X,const Type & Y)
{
    return -1/(Square(X)+Square(Y));
}

#define INSTATIATE_FUNC_ANALYTIQUE(TYPE)\
template  TYPE sinC(const TYPE & aTeta,const TYPE & aEps);\
template  TYPE sinC(const TYPE & aTeta);\
template  TYPE AtanXsY_sX(const TYPE & X,const TYPE & Y,const TYPE & aEps);\
template  TYPE AtanXsY_sX(const TYPE & X,const TYPE & Y);\
template  TYPE DerXAtanXsY_sX(const TYPE & X,const TYPE & Y,const TYPE & aEps);\
template  TYPE DerXAtanXsY_sX(const TYPE & X,const TYPE & Y);\
template  TYPE DerYAtanXsY_sX(const TYPE & X,const TYPE & Y);

INSTATIATE_FUNC_ANALYTIQUE(tREAL4)
INSTATIATE_FUNC_ANALYTIQUE(tREAL8)
INSTATIATE_FUNC_ANALYTIQUE(tREAL16)


};

