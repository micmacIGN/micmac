#ifndef _V1ComLine
#define _V1ComLine

#include "MMVII_Ptxd.h"
#include "MMVII_Stringifier.h"

/** \file contains method when MMV1 is used by command (like any other external binary).

    In this cas we need few method to convert MMV2 object in string for MMV1 command, but no need for
  library.
*/


namespace MMVII
{


template <class Type> std::string ToStrComMMV1(const cPtxd<Type,2> & aP) {return "["+ToStr(aP.x()) + "," + ToStr(aP.y()) + "]";}
template <class Type> std::string ToStrComMMV1(const cTplBox<Type,2> & aBox) 
{
  return "["+ToStr(aBox.P0().x()) + "," + ToStr(aBox.P0().y()) + "," + ToStr(aBox.P1().x()) + "," + ToStr(aBox.P1().y()) +  "]";
}




};

#endif // _V1ComLine
