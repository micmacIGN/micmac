#ifndef _SymbDer_Common_H_
#define _SymbDer_Common_H_

#include <iostream>
#include <assert.h>

namespace NS_SymbolicDerivative {

/* These functions are required if we want to have same operation on numbers double and formulas
   They are suposed to be optimized implementation of pow for integer low value
   of the exponent
*/
template <class Type> inline Type square(const Type & aV)  {return aV*aV;}
template <class Type> inline Type cube(const Type & aV)    {return aV*aV*aV;}
template <class Type> inline Type pow4(const Type & aV)    {return square(square(aV));}
template <class Type> inline Type pow5(const Type & aV)    {return aV *pow4(aV);}
template <class Type> inline Type pow6(const Type & aV)    {return square(cube(aV));}
template <class Type> inline Type pow7(const Type & aV)    {return aV *pow6(aV);}
template <class Type> inline Type pow8(const Type & aV)    {return square(pow4(aV));}
template <class Type> inline Type pow9(const Type & aV)    {return aV *pow8(aV);}

static inline void Error(const std::string & aMes,const std::string & aExplanation)
{
    std::cout << "In SymbolicDerivative a fatal error" << "\n";
    std::cout << "  Likely Source   ["<< aExplanation << "\n";
    std::cout << "  Message  ["<< aMes << "]\n";
    assert(false);
}
     ///    Error due probably to internal mistake
static inline void InternalError(const std::string & aMes)
{
   Error(aMes,"Internal Error of the Library");
}
     /// Error probably due to bas usage of the library (typically out limit vector access)
static inline void UserSError(const std::string & aMes)
{
   Error(aMes,"Probable error on user's side due to unapropriate usage of the library");
}

     /// Check equality in test, taking account numericall error
static inline void AssertAlmostEqual(const double & aV1,const double & aV2,const double & aEps)
{
   if ( (std::abs(aV1-aV2)> aEps*(std::abs(aV1)+std::abs(aV2))) )
      InternalError("Test equality failed");
}


} // namespace NS_SymbolicDerivative

#endif // _SymbDer_Common_H_
