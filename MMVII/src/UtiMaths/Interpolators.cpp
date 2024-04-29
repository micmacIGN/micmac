#include "MMVII_Interpolators.h"

namespace MMVII
{

class cInterpolator1D : public cMemCheck
{
      public :
        cInterpolator1D(const tREAL8 & aSzKernel);

        virtual tREAL8  Weight(const tREAL8 & anX) const = 0;
      protected :
	tREAL8 mSzKernel;
};



class cBilinInterpolator1D : public cInterpolator1D
{
      public :
        cBilinInterpolator1D();
        tREAL8  Weight(const tREAL8 & anX) const override ;
};




        // cInterpolator1D(const tREAL8 & aSzKernel);


tREAL8 CubAppGaussVal(const tREAL8& aV)
{
   tREAL8 aAbsV = std::abs(aV);
   if (aAbsV>1.0) return 0.0;

   tREAL8 aAV2 = Square(aAbsV);

   return 1.0 + 2.0*aAbsV*aAV2 - 3.0*aAV2;
}

/**   Compute the weighting for ressampling one pixel of an image with a mapping M.
 *  Formalisation :
 *
 *      - we have pixel out  Co
 *      - we have an image weighing arround Co  W(P) = BiCub((P-Co)/aSzK)
 *      - let S be the support of W(P) we compute the box of M-1(S)
 *
 */

cRessampleWeigth  cRessampleWeigth::GaussBiCub(const cPt2dr & aCenterOut,const cAff2D_r & aMapO2I, double aSzK)
{
     cRessampleWeigth aRes;

     // [1] compute the box in input image space 
     cPt2dr aSzW = cPt2dr::PCste(aSzK);
     cBox2dr aBoxOut(aCenterOut-aSzW,aCenterOut+aSzW);
     cBox2di aBoxIn =  ImageOfBox(aMapO2I,aBoxOut).Dilate(1).ToI();

     cAff2D_r  aMapI2O = aMapO2I.MapInverse();

     double aSomW = 0.0;
     for (const auto & aPixIn : cRect2(aBoxIn))
     {
         cPt2dr aPixOut = aMapI2O.Value(ToR(aPixIn));
         double aW =  CubAppGaussVal(Norm2(aPixOut-aCenterOut)/aSzK);
         if (aW>0)
         {
            aRes.mVPts.push_back(aPixIn);
            aRes.mVWeight.push_back(aW);
            aSomW += aW;
         }
     }

     // if not empty  , som W = 1
     if (aSomW>0)
     {
        for (auto & aW : aRes.mVWeight)
        {
            aW /= aSomW;
        }
     }

     return aRes;
}

};

