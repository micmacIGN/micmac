#include "MMVII_ImageMorphoMath.h"

namespace MMVII
{


/* ******************************************************** */
/*                                                          */
/*                       cFlag8Neigh                        */
/*                                                          */
/* ******************************************************** */

void  cFlag8Neigh::ComputeCCOfeNeigh(std::vector<std::vector<cCCOfOrdNeigh>> & aVCC)
{
     aVCC = std::vector<std::vector<cCCOfOrdNeigh>> (256);

     for (size_t aFlag=0 ; aFlag<256 ; aFlag++)  // parse all possible flag combinaisons
     {
         cFlag8Neigh aNeigh(aFlag);
         for (size_t aBit=0 ; aBit<8 ; aBit++)  // parse all bit
         {
              if (aNeigh.SafeIsIn(aBit) != aNeigh.SafeIsIn(aBit-1))  // select bit of transition
              {
                   int aBitNext = aBit+1;
                   while (aNeigh.SafeIsIn(aBit) == aNeigh.SafeIsIn(aBitNext)) // go to next transition
                         aBitNext++;

		   //  create a new connected component
                   cCCOfOrdNeigh aCC;
                   aCC.mIs1 = aNeigh.SafeIsIn(aBit);
                   aCC.mFirstBit = aBit;
                   aCC.mLastBit  = aBitNext;
		   // add it
                   aVCC.at(aFlag).push_back(aCC);
              }
         }
     }
}

/* ******************************************************** */
/*                                                          */
/*    :: Global func, based on value ordering & neigboors   */
/*                                                          */
/* ******************************************************** */

template <class Type>  cFlag8Neigh   FlagSup8Neigh(const cDataIm2D<Type> & aDIm,const cPt2di & aPt)
{
   Type aV0 = aDIm.GetV(aPt);
   cFlag8Neigh aResult ;
   //  for freeman 1,4  Y is positive, for freeman 0, X is positive
   for (int aK=0 ; aK<4 ; aK++)
   {
      if (aDIm.GetV(aPt+FreemanV8[aK]) >=  aV0)
              aResult.AddNeigh(aK);
   }
   for (int aK=4 ; aK<8 ; aK++)
   {
      if (aDIm.GetV(aPt+FreemanV8[aK]) >  aV0)
         aResult.AddNeigh(aK);
   }
   return aResult;
}

/*  For this "topo" criteria, we compute :
 *
 *      - aMaxOfMin = max for all connected component of the min of the connected component
 *        typically we should have 2 CC, they should be  both low
 *      - idem aMinOfMax
 *
 *    Value(Point) - aMaxOfMin : reflect it is a sadlle point for low value
 *    aMinOfMax - Value(Point) : idem
 *
 *    we select the worst criteria
 */



template <class Type>  tREAL8   CriterionTopoSadle(const cDataIm2D<Type> & aDIm,const cPt2di & aPixC)
{
    cFlag8Neigh  aNeigh = FlagSup8Neigh(aDIm,aPixC);
    const std::vector<cCCOfOrdNeigh> &   aVCC = aNeigh.ConComp() ;

    tREAL8  aMaxOfMin(-1e10);
    tREAL8  aMinOfMax(1e10);
    for (const  auto & aCC : aVCC)
    {
         cBoundVals<tREAL8> aBounds;
         for (int aBit = aCC.mFirstBit ; aBit<aCC.mLastBit ; aBit++)
         {
             aBounds.Add(aDIm.GetV(aPixC + FreemanV8[aBit%8]));
         }

         if ( aCC.mIs1)
            UpdateMin(aMinOfMax,aBounds.VMax());
         else
            UpdateMax(aMaxOfMin,aBounds.VMin());
    }

    tREAL8 aV0 =  aDIm.GetV(aPixC);

    return std::min(aMinOfMax-aV0, aV0-aMaxOfMin);
}

template cFlag8Neigh FlagSup8Neigh(const cDataIm2D<tREAL4> & aDIm,const cPt2di & aPt);
template tREAL8      CriterionTopoSadle(const cDataIm2D<tREAL4> & aDIm,const cPt2di & aPixC);



};
