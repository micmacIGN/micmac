#ifndef _COMMON_HEADER_SYMBDER_H_
#define _COMMON_HEADER_SYMBDER_H_

/** 
   \brief contain functionnality that are required for code gen and need micma lib
*/

#include "include/MMVII_all.h"
#include "include/SymbDer/SymbolicDerivatives.h"
#include <typeinfo>       // operator typeid

using namespace NS_SymbolicDerivative;


namespace MMVII
{
/// required so that we can define points on formula ...

template <> class tElemNumTrait<cFormula <tREAL8> >
{
    public :
        // For these type rounding mean something
        // static bool IsInt() {return true;}
        typedef cFormula<tREAL8>  tBase;
        typedef cFormula<tREAL8>  tBig;
};

template <class Type> Type SqNormL2V2(const Type & aX,const Type & aY)
{
    return Square(aX) + Square(aY);
}
template <class Type> Type SqNormL2V3(const Type & aX,const Type & aY,const Type & aZ)
{
    return Square(aX) + Square(aY) + Square(aZ);
}


template <class Type> Type NormL2V2(const Type & aX,const Type & aY)
{
    return sqrt(SqNormL2V2(aX,aY));
}
template <class Type> Type NormL2V3(const Type & aX,const Type & aY,const Type & aZ)
{
    return sqrt(SqNormL2V3(aX,aY,aZ));
}




template <class Type> Type NormL2Vec2(const std::vector<Type> & aVec)
{
    return NormL2V2(aVec.at(0),aVec.at(1));
}





};//  namespace MMVII

#endif // _COMMON_HEADER_SYMBDER_H_
