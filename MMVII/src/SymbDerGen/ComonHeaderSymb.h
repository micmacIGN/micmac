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



};//  namespace MMVII

#endif // _COMMON_HEADER_SYMBDER_H_
