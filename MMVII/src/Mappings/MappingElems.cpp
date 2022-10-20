#include "MMVII_Mappings.h"
#include "MMVII_Geom2D.h"


namespace MMVII
{

/* ============================================= */
/*                cBijAffMapElem                 */
/* ============================================= */


template <class Type,const int Dim> 
  cBijAffMapElem<Type,Dim>::cBijAffMapElem(const tMat & aMat ,const tPt& aTr) :
    mMat     (aMat.Dup()),
    mTr      (aTr),
    mMatInv  (aMat.Inverse())
{
}

template <class Type,const int Dim>  cPtxd<Type,Dim> cBijAffMapElem<Type,Dim>::Value(const tPt & aP) const 
{
   return mTr + mMat * aP;
}
template <class Type,const int Dim>  cPtxd<Type,Dim> cBijAffMapElem<Type,Dim>::Inverse(const tPt & aP) const 
{
   return mMatInv*(aP-mTr);;
}
template <class Type,const int Dim>  cBijAffMapElem<Type,Dim> cBijAffMapElem<Type,Dim>::MapInverse() const 
{
   return cBijAffMapElem<Type,Dim>(mMatInv,-mTr*mMatInv);
}

template  class cBijAffMapElem<tREAL8,2>;
template  class cBijAffMapElem<tREAL8,3>;


/* ============================================= */
/*      cDataMapping<Type>                       */
/* ============================================= */

template <class cMapElem>
   cInvertMappingFromElem<cMapElem>::cInvertMappingFromElem
      (const cMapElem & aMap,const tMapInv & aIMap) :
         mMap  (aMap),
         mIMap (aIMap)
{
}

template <class cMapElem>
   cInvertMappingFromElem<cMapElem>::cInvertMappingFromElem (const cMapElem & aMap) :
         cInvertMappingFromElem<cMapElem>(aMap,aMap.MapInverse())
{
}


template <class cMapElem>
  const typename  cInvertMappingFromElem<cMapElem>::tVecPt &
                  cInvertMappingFromElem<cMapElem>::Values(tVecPt & aRes,const tVecPt & aVIn ) const 
{
   for (const auto & aPtIn : aVIn)
       aRes.push_back(mMap.Value(aPtIn));
   return aRes;
}

template <class cMapElem>
  typename  cInvertMappingFromElem<cMapElem>::tPt 
            cInvertMappingFromElem<cMapElem>::Value(const tPt & aPt) const 
{
   return mMap.Value(aPt);
}

template <class cMapElem>
  const typename  cInvertMappingFromElem<cMapElem>::tVecPt &
                  cInvertMappingFromElem<cMapElem>::Inverses(tVecPt & aRes,const tVecPt & aVIn ) const 
{
   for (const auto & aPtIn : aVIn)
       aRes.push_back(mIMap.Value(aPtIn));
   return aRes;
}

template <class cMapElem>
  typename  cInvertMappingFromElem<cMapElem>::tPt 
            cInvertMappingFromElem<cMapElem>::Inverse(const tPt & aPt) const 
{
   return mIMap.Value(aPt);
}


template <class tMapElem> class  cMapOfBox
{
     public :
	 static constexpr int    TheDim=tMapElem::TheDim;
         typedef  typename tMapElem::tTypeElem  tTypeElem;

         typedef  cTplBox<tTypeElem,TheDim> tBox;

	 static tBox ImageOfBox(const tMapElem& aMap,const tBox & aBox)
	 {
              cInvertMappingFromElem<tMapElem> aIMapFE(aMap);
              return aIMapFE.BoxOfCorners(aBox);
	 }
};

cBox2dr  ImageOfBox(const cAff2D_r & aAff,const cBox2dr & aBox)
{
   return cMapOfBox<cAff2D_r>::ImageOfBox(aAff,aBox);
}


template  class cInvertMappingFromElem<cAffin2D<tREAL8>>;
template  class cInvertMappingFromElem<cSim2D<tREAL8>>;
template  class cInvertMappingFromElem<cHomot2D<tREAL8>>;
template  class cInvertMappingFromElem<cBijAffMapElem<tREAL8,2>>;
template  class cInvertMappingFromElem<cBijAffMapElem<tREAL8,3>>;


};
