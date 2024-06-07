#include "MMVII_Interpolators.h"
#include "MMVII_Mappings.h"
#include "MMVII_HeuristikOpt.h"
#include "MMVII_ExtractLines.h"


namespace MMVII
{

/* ************************************************************************ */
/*                                                                          */
/*                          cScoreTetaLine                                  */
/*                                                                          */
/* ************************************************************************ */

cScoreTetaLine::cScoreTetaLine(cDataIm2D<tREAL4> & aDIm,const cDiffInterpolator1D & anInt,tREAL8 aLength,tREAL8 aStep) :
        mDIm        (&aDIm),
        mLength     (aLength),
        mNb         (round_up(mLength/aStep)),
        mStepAbsc   (mLength/mNb),
        mStepTeta   (-1),
        mTabInt     (anInt,100)
{
}

void cScoreTetaLine::SetCenter(const cPt2dr & aC) { mC = aC; }

/**   For an "oriented" segemet S :
 *       - C  the center, Teta the direction, L the lengnt
 *    We return how the segement is one of the direction of the checkboard pattern. Let
 *       - T the tangent and N the normal
 *       - A be the absicsse of a point  -L < A < L
 *       - P =  C + A T  a point  of 
 *       - let G be the  gradient of image in P, G' = G/|G| the direction of G
 *    The score depends of two "signs" s1 and s2   (s1,s2 = "+-1")  :
 *       - s1 depends if we have black ofr white a the left of S (which is oriented)
 *       - s2 depends if A>0 or A<0 (because the left colour change when cross C)
 *    So the score is finally :
 *
 *        | G' - s1s2 N| 
 *
 */

cPt1dr  cScoreTetaLine::Value(const cPt1dr& aPtAngle) const
{
    tREAL8 aTeta = aPtAngle.x();
    cPt2dr aTgt = FromPolar(1.0,aTeta);
    cPt2dr aNormal = aTgt * cPt2dr(0,1.0);
    cWeightAv<tREAL8,tREAL8> aWA; // Average of difference

    for (int aKL=-mNb ; aKL<=mNb ; aKL++)  // samples all the point on the line
    {
         if (aKL)
         {
             tREAL8 aAbsc = mStepAbsc*aKL;
             cPt2dr aPt = mC + aTgt * aAbsc;
             auto [aVal,aGrad] = mDIm->GetValueAndGradInterpol(mTabInt,aPt);

             tREAL8 aN2 = Norm2(aGrad);
             if (aN2 > 0)
             {
                  // Orientation depend of Sign & position in the segment
                  aGrad = aGrad / (aN2 * mCurSign * (aKL>0 ? 1 : -1));
                  tREAL8 aDist = Norm2(aNormal-aGrad);
                  aWA.Add(aN2,aDist);
             }
         }
    }

    return cPt1dr(aWA.Average(1e10));  // 1e10 => def value in case no point
}


/**  Get initial guess of direction, simply parse all possible value at given step */

tREAL8  cScoreTetaLine::GetTetasInit(tREAL8 aStepPix,int aCurSign)
{
      mCurSign = aCurSign;
      cWhichMin<tREAL8,tREAL8> aWMin; // minimal score

      mStepTeta = aStepPix / mLength; // Step in radian to have given step in pixel
      int aNbTeta = round_up(M_PI / mStepTeta);
      mStepTeta  = M_PI / aNbTeta;

      for (int aKTeta=0 ; aKTeta<aNbTeta ; aKTeta ++) // samples all teta
      {
           tREAL8 aTeta = aKTeta * mStepTeta;
           tREAL8 aVal = Value(cPt1dr(aTeta)).x();
           aWMin.Add(aTeta,aVal);  //  update 
      }

      return aWMin.IndexExtre() ;
}

/**  Refine the existing value of mimimal using "cOptimByStep", suppose we are close enouh */
tREAL8  cScoreTetaLine::Refine(tREAL8 aTeta0,tREAL8 aStepPix,int aSign)
{
     mCurSign = aSign;
     cOptimByStep<1> aOpt(*this,true,10.0);  // 10.0 => DistMin
     cPt1dr  aRes = aOpt.Optim(cPt1dr(aTeta0),mStepTeta,aStepPix/mLength).second;

     return aRes.x();
}

std::pair<tREAL8,tREAL8> cScoreTetaLine::Tetas_CheckBoard(const cPt2dr & aC,tREAL8 aStepInit,tREAL8 aStepLim)
{
    std::vector<tREAL8> aVTeta;
    SetCenter(aC);

    for (const auto & aSign : {-1,1})
    {
        tREAL8  aTetaInit =  GetTetasInit(aStepInit,aSign);
        tREAL8  aTetaRefine = Refine(aTetaInit,aStepLim,aSign);

        // at this step the angles are define % pi, theoretically in [0,Pi], btw after due to optim it can be slightly outside
        aTetaRefine = mod_real(aTetaRefine,M_PI);
        aVTeta.push_back(aTetaRefine);
    }

    // the angle are defined %pi, to have a single representation we need that teta1/teta2
    // correpond to a direct repair
    if (aVTeta.at(0)> aVTeta.at(1))
       aVTeta.at(1) += M_PI;

    return std::pair<tREAL8,tREAL8>(aVTeta.at(0),aVTeta.at(1));
}

};
