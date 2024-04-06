#include "MMVII_Linear2DFiltering.h"
#include "MMVII_Geom2D.h"

#include "MMVII_ExtractLines.h"

namespace MMVII
{

/* ************************************************************************ */
/*                                                                          */
/*                       cImGradWithN                                       */
/*                                                                          */
/* ************************************************************************ */


template <class Type>   
  cImGradWithN<Type>::cImGradWithN(const cPt2di & aSz) :
     cImGrad<Type>  (aSz),
     mNormG         (aSz),
     mDataNG        (mNormG.DIm())
{
}

template <class Type>   
  cImGradWithN<Type>::cImGradWithN(const cDataIm2D<Type> & aImIn,Type aAlphaDeriche) :
	   cImGradWithN<Type>(aImIn.Sz())
{
    ComputeDericheAndNorm(*this,aImIn,aAlphaDeriche);
}


/*   The test is  more complicated than traditionnal local maxima :
 *
 *     - 1- On a given edge if we take all the 8-neighboors, one the neighbors in the edge will
 *         have higher value, that why we take the neigbours that are +- in the direction of grad
 *         See "NeighborsForMaxLoc()"
 *
 *     - 2- with thin line, the oposite contour may have value that may be higer than the contour itself,
 *      that's why we consider only the point which are orientd in the same direction (test "Scal > 0"),
 *      note this work for a dark or light line on a average back-ground
 */
template<class Type> bool  cImGradWithN<Type>::IsMaxLocDirGrad(const cPt2di& aPix,const std::vector<cPt2di> & aVP,tREAL8 aRatioXY) const
{
    //  [1] Compute unitary vector
    tREAL8 aN = mDataNG.GetV(aPix);
    if (aN==0) return false;  // avoid div by 0
    cPt2dr aDirGrad = ToR(this->Grad(aPix)) * (1.0/ aN);

    //[2]   A Basic test to reject point on integer neighbourhood in the two direction
    {
        cPt2di aIDirGrad = ToI(aDirGrad);
        for (int aSign : {-1,1})
        {
             cPt2di  aNeigh =  aPix + aIDirGrad * aSign;
             if ( (mDataNG.DefGetV(aNeigh,-1)>aN) && (Scal(aDirGrad,ToR(this->Grad(aNeigh)))>0) )
             {
                return false;
             }
        }
    }

    // [3]

    cPt2dr aConjDG =  conj(aDirGrad);
    for (const auto & aDeltaNeigh : aVP)
    {
        // cPt2dr aDirLoc = ToR(aDeltaNeigh) / aDirGrad;
        cPt2dr aDirLoc = ToR(aDeltaNeigh) * aConjDG;
        if (std::abs(aDirLoc.x()) >= std::abs(aDirLoc.y()*aRatioXY))
	{
            cPt2di aNeigh = aPix + aDeltaNeigh;

            if ( (mDataNG.DefGetV(aNeigh,-1)>aN) && (Scal(aDirGrad,ToR(this->Grad(aNeigh))) >0))
               return false;
	}
        /*cPt2dr aNeigh = ToR(aPix) + ToR(aDeltaNeigh) * aDirGrad;  
        if ( (mDataNG.DefGetVBL(aNeigh,-1)>aN) && (Scal(aDirGrad,ToR(this->GradBL(aNeigh))) >0))
           return false;
        */
    }

    return true;
}

//  return simply all the point except (0,0)
template<class Type> std::vector<cPt2di>   cImGradWithN<Type>::NeighborsForMaxLoc(tREAL8 aRay) 
{ 
     return SortedVectOfRadius(0.5,aRay); 
}

/*   Make one iteration of position computation :
 *
 *     -1- compute the value of gradient norm befor and after the point in gradient direction
 *     -2- extract the maximal by parabol fiting
 */

template<class Type> cPt2dr   cImGradWithN<Type>::OneRefinePos(const cPt2dr & aP1) const
{
     if (! mDataNG.InsideBL(aP1)) 
         return aP1;

     cPt2dr aGr = VUnit(ToR(this->GradBL(aP1)));

     // extract point "before"
     cPt2dr aP0 = aP1- aGr;
     if (! mDataNG.InsideBL(aP0)) 
         return aP1;

     // extract point "after"
     cPt2dr aP2 = aP1+ aGr;
     if (! mDataNG.InsideBL(aP2)) 
         return aP1;

     // extract abscisse of maxim of parabol
     tREAL8 aAbs = StableInterpoleExtr(mDataNG.GetVBL(aP0),mDataNG.GetVBL(aP1),mDataNG.GetVBL(aP2));

     return aP1 + aGr * aAbs;
}

//  Very basic just max two iterations
template<class Type> cPt2dr   cImGradWithN<Type>::RefinePos(const cPt2dr & aP1) const
{
    return OneRefinePos(OneRefinePos(aP1));
}

          /* ************************************************ */

template<class Type> void ComputeDericheAndNorm(cImGradWithN<Type> & aResGrad,const cDataIm2D<Type> & aImIn,double aAlpha) 
{
     ComputeDeriche(aResGrad,aImIn,aAlpha);

     auto & aDN =  aResGrad.NormG().DIm();
     for (const auto &  aPix : aDN)
     {
           aDN.SetV(aPix,Norm2(aResGrad.Grad(aPix)));
     }
}

template class cImGradWithN<tREAL4>;
template void ComputeDericheAndNorm(cImGradWithN<tREAL4> & aResGrad,const cDataIm2D<tREAL4> & aImIn,double aAlpha) ;

};

