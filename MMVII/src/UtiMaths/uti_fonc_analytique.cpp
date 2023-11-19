
#include "MMVII_Ptxd.h"

namespace MMVII
{

	/*
template  TYPE Sqrt(const TYPE & aSin);
template <typename Type> Type Sqrt(const Type & aX)
{
     MMVII_INTERNAL_ASSERT_tiny((aX>=0),"Bad value for arcsinus");
     return std::sqrt(aX);
}
*/

template <typename Type> Type DerSqrt(const Type & aX)
{
     MMVII_INTERNAL_ASSERT_tiny((aX>0),"Bad value for arcsinus");
     return 1/(2*std::sqrt(aX));
}

template <typename Type> Type DerASin(const Type & aSin)
{
   Type UMS2 = 1-Square(aSin);
   MMVII_ASSERT_STRICT_POS_VALUE(UMS2);
   return 1 / std::sqrt(UMS2);
}

template <typename Type> Type ASin(const Type & aSin)
{
     MMVII_INTERNAL_ASSERT_tiny((aSin>=-1) && (aSin<=1),"Bad value for arcsinus");
     return std::asin(aSin);
}


constexpr int Fact3 = 2 * 3;
constexpr int Fact5 = 2 * 3 * 4 * 5;  // Dont use Fact3, we dont control order of creation
constexpr int Fact7 = 2 * 3 * 4 * 5 * 6 * 7 ;
constexpr int Fact9 = 2 * 3 * 4 * 5 * 6 * 7 * 8 * 9;
constexpr int Fact11 = 2 * 3 * 4 * 5 * 6 * 7 * 8 * 9;


template <typename Type> Type DerSinC(const Type & aTeta,const Type & aEps)
{
/*  sin(X)/X' =  (x cos(x) -sinx) / X^ 2 */
   if (std::abs(aTeta) > aEps)
      return (aTeta*std::cos(aTeta)-std::sin(aTeta))/Square(aTeta);

   Type aTeta2 = Square(aTeta);

   Type aT3 = aTeta * aTeta2;
   Type aT5 = aT3   * aTeta2;
   Type aT7 = aT5   * aTeta2;
   Type aT9 = aT7   * aTeta2;


   return  - aTeta * (static_cast<Type>(2.0)/static_cast<Type>(Fact3))
           + aT3 * (static_cast<Type>(4.0)/static_cast<Type>(Fact5))
           - aT5 * (static_cast<Type>(6.0)/static_cast<Type>(Fact7))
           + aT7 * (static_cast<Type>(8.0)/static_cast<Type>(Fact9))
           - aT9 * (static_cast<Type>(10.0)/static_cast<Type>(Fact11));
}
template <typename Type> Type DerSinC(const Type & aTeta)
{
   return DerSinC(aTeta,tElemNumTrait<Type>::Accuracy());
}

template <typename Type> Type sinC(const Type & aTeta,const Type & aEps)
{
// x - x3/3! + x5/5! - x7/7! +

   if (std::abs(aTeta) > aEps)
      return std::sin(aTeta)/aTeta;
   Type aT2 = Square(aTeta);
   Type aT4 = Square(aT2);
   Type aT6 = aT4 * aT2;
   Type aT8 = Square(aT4);

   return static_cast<Type>(1.0) - aT2/static_cast<Type>(Fact3) + aT4/static_cast<Type>(Fact5) - aT6/static_cast<Type>(Fact7) + aT8/static_cast<Type>(Fact9);
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

   return (1
           - static_cast<Type>(XsY2)/static_cast<Type>(3.0)
           + static_cast<Type>(XsY4)/static_cast<Type>(5.0)
           - static_cast<Type>(XsY6)/static_cast<Type>(7.0)
           + static_cast<Type>(XsY8)/static_cast<Type>(9.0) ) / static_cast<Type>(Y);
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

   return (X/Cube(Y)) * ( - static_cast<Type>(2.0/3.0)
                          + static_cast<Type>(4.0/5.0) * XsY2
                          - static_cast<Type>(6.0/7.0) * XsY4
                          + static_cast<Type>(8.0/9.0) * XsY6
                          - static_cast<Type>(10.0/11.0) * XsY8);
}
template <typename Type> Type DerXAtanXsY_sX(const Type & X,const Type & Y)
{
   return DerXAtanXsY_sX(X,Y,tElemNumTrait<Type>::Accuracy());
}


template <typename Type> Type DerYAtanXsY_sX(const Type & X,const Type & Y)
{
    return -1/(Square(X)+Square(Y));
}


template <typename Type> Type ATan2(const Type & aX,const Type & aY)
{
     MMVII_INTERNAL_ASSERT_tiny((aX!=0)||(aY!=0),"Bad value for arcsinus");

     return std::atan2(aX,aY);
}

template <typename Type> Type DerX_ATan2(const Type & aX,const Type & aY)
{
     MMVII_INTERNAL_ASSERT_tiny((aX!=0)||(aY!=0),"Bad value for arcsinus");
     return aY / (Square(aX)+Square(aY));
}

template <typename Type> Type DerY_ATan2(const Type & aX,const Type & aY)
{
     MMVII_INTERNAL_ASSERT_tiny((aX!=0)||(aY!=0),"Bad value for arcsinus");
     return  (- aX) / (Square(aX)+Square(aY));
}


template <typename Type> Type sinH(const Type & aX) {return std::exp(aX)-std::exp(-aX);}
template <typename Type> Type cosH(const Type & aX) {return std::exp(aX)+std::exp(-aX);}


#define INSTATIATE_FUNC_ANALYTIQUE(TYPE)\
template  TYPE DerSqrt(const TYPE & aSin);\
template  TYPE DerASin(const TYPE & aSin);\
template  TYPE ASin(const TYPE & aSin);\
template  TYPE sinC(const TYPE & aTeta,const TYPE & aEps);\
template  TYPE sinC(const TYPE & aTeta);\
template  TYPE DerSinC(const TYPE & aTeta,const TYPE & aEps);\
template  TYPE DerSinC(const TYPE & aTeta);\
template  TYPE ATan2(const TYPE & X,const TYPE & Y);\
template  TYPE DerX_ATan2(const TYPE & X,const TYPE & Y);\
template  TYPE DerY_ATan2(const TYPE & X,const TYPE & Y);\
template  TYPE AtanXsY_sX(const TYPE & X,const TYPE & Y,const TYPE & aEps);\
template  TYPE AtanXsY_sX(const TYPE & X,const TYPE & Y);\
template  TYPE DerXAtanXsY_sX(const TYPE & X,const TYPE & Y,const TYPE & aEps);\
template  TYPE DerXAtanXsY_sX(const TYPE & X,const TYPE & Y);\
template  TYPE DerYAtanXsY_sX(const TYPE & X,const TYPE & Y);\
template  TYPE sinH(const TYPE & );\
template  TYPE cosH(const TYPE & );

INSTATIATE_FUNC_ANALYTIQUE(tREAL4)
INSTATIATE_FUNC_ANALYTIQUE(tREAL8)
INSTATIATE_FUNC_ANALYTIQUE(tREAL16)


};

