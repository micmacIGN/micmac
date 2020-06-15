#ifndef _SymbDer_Common_H_
#define _SymbDer_Common_H_


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

} // namespace NS_SymbolicDerivative

#endif // _SymbDer_Common_H_
