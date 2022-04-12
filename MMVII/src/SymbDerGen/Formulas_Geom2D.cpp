#include "include/MMVII_all.h"
#include "include/SymbDer/SymbolicDerivatives.h"
#include "include/SymbDer/SymbDer_MACRO.h"

using namespace NS_SymbolicDerivative;


namespace MMVII
{

template <class Type>
std::vector<Type> Dist2DConservation
                  (
                      const std::vector<Type> & aVUk,
                      const std::vector<Type> & aVObs
                  )
{
    const Type & x1 = aVUk[0];
    const Type & y1 = aVUk[1];
    const Type & x2 = aVUk[0];
    const Type & y2 = aVUk[1];

    const Type & d  = aVObs[0];  // Warn the data I got were in order y,x ..

    return { sqrt(square(x1-x2) + square(y1-y2)) - d } ;
}




};//  namespace MMVII

