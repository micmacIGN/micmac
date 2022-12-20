#ifndef _COMMON_HEADER_SYMBDER_H_
#define _COMMON_HEADER_SYMBDER_H_

/** 
   \brief contain functionnality that are required for code gen and need micma lib
*/

#include "SymbDer/SymbolicDerivatives.h"
#include <typeinfo>       // operator typeid

using namespace NS_SymbolicDerivative;


namespace MMVII
{
/// required so that we can define points on formula ...

template <> class tNumTrait<cFormula <tREAL8> >
{
    public :
        // For these type rounding mean something
        // static bool IsInt() {return true;}
        typedef cFormula<tREAL8>  tBase;
        typedef cFormula<tREAL8>  tBig;
        typedef cFormula<tREAL8>  tFloatAssoc;
        static void AssertValueOk(const cFormula<double> & ) {}
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



template  <typename tScal> std::vector<tScal> ToVect(const cPtxd<tScal,3> & aPt)
{
     return  {aPt.x(),aPt.y(),aPt.z()};
}
template  <typename tScal> std::vector<tScal> ToVect(const cPtxd<tScal,2> & aPt)
{
     return {aPt.x(),aPt.y()};
}



template  <typename tScal> tScal PScal(const cPtxd<tScal,3> & aP1,const cPtxd<tScal,3> & aP2)
{
         return aP1.x()*aP2.x() + aP1.y() *aP2.y() + aP1.z() * aP2.z();
}
template  <typename tScal> cPtxd<tScal,3>  VtoP3(const  std::vector<tScal> & aV,size_t aInd=0)
{
        return cPtxd<tScal,3>(aV.at(aInd),aV.at(aInd+1),aV.at(aInd+2));
}
template  <typename tScal> cPtxd<tScal,2>  VtoP2(const  std::vector<tScal> & aV,size_t aInd=0)
{
        return cPtxd<tScal,2>(aV.at(aInd),aV.at(aInd+1));
}
template  <typename tScal> cPtxd<tScal,3>   MulMat(const std::vector<tScal> & aV,size_t aInd,const  cPtxd<tScal,3> & aP)
{
     cPtxd<tScal,3> aL1 =  VtoP3(aV,aInd);
     cPtxd<tScal,3> aL2 =  VtoP3(aV,aInd+3);
     cPtxd<tScal,3> aL3 =  VtoP3(aV,aInd+6);

     return cPtxd<tScal,3>(PScal(aP,aL1),PScal(aP,aL2),PScal(aP,aL3));
}



};//  namespace MMVII

#endif // _COMMON_HEADER_SYMBDER_H_
