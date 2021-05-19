#include "include/MMVII_all.h"

namespace MMVII
{

/* ============================================= */
/*      cDataMapping<Type>                       */
/* ============================================= */

template <class cMapElem,class cIMapElem> 
   cInvertMappingFromElem<cMapElem,cIMapElem>::cInvertMappingFromElem
      (const cMapElem & aMap,const cIMapElem & aIMap) :
         mMap  (aMap),
         mIMap (aIMap)
{
}

template <class cMapElem,class cIMapElem> 
   cInvertMappingFromElem<cMapElem,cIMapElem>::cInvertMappingFromElem (const cMapElem & aMap) :
         cInvertMappingFromElem<cMapElem,cIMapElem>(aMap,aMap.MapInverse())
{
}


template <class cMapElem,class cIMapElem> 
  const typename  cInvertMappingFromElem<cMapElem,cIMapElem>::tVecPt &
                  cInvertMappingFromElem<cMapElem,cIMapElem>::Values(tVecPt & aRes,const tVecPt & aVIn ) const 
{
   for (const auto & aPtIn : aVIn)
       aRes.push_back(mMap.Value(aPtIn));
   return aRes;
}

template <class cMapElem,class cIMapElem> 
  typename  cInvertMappingFromElem<cMapElem,cIMapElem>::tPt 
            cInvertMappingFromElem<cMapElem,cIMapElem>::Value(const tPt & aPt) const 
{
   return mMap.Value(aPt);
}

template <class cMapElem,class cIMapElem> 
  const typename  cInvertMappingFromElem<cMapElem,cIMapElem>::tVecPt &
                  cInvertMappingFromElem<cMapElem,cIMapElem>::Inverses(tVecPt & aRes,const tVecPt & aVIn ) const 
{
   for (const auto & aPtIn : aVIn)
       aRes.push_back(mIMap.Value(aPtIn));
   return aRes;
}

template <class cMapElem,class cIMapElem> 
  typename  cInvertMappingFromElem<cMapElem,cIMapElem>::tPt 
            cInvertMappingFromElem<cMapElem,cIMapElem>::Inverse(const tPt & aPt) const 
{
   return mIMap.Value(aPt);
}







template  class cInvertMappingFromElem<cSim2D<tREAL8>,cSim2D<tREAL8>>;




};
