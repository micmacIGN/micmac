#include "StdAfx.h"
#include "include/MMVII_all.h"


namespace MMVII
{

template <class Type> Pt2d<Type>  ToMMV1(const cPtxd<Type,2> &  aP) {return  Pt2d<Type>(aP.x(),aP.y());}
template <class Type> cPtxd<Type,2> ToMMVII(const Pt2d<Type> &  aP) {return cPtxd<Type,2>(aP.x,aP.y);}

GenIm::type_el ToMMV1(eTyNums aV2);
eTyNums ToMMVII( GenIm::type_el );

};

