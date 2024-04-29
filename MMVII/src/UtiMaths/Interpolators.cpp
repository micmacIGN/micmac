#include "MMVII_Interpolators.h"

namespace MMVII
{

class cInterpolator1D : public cMemCheck
{
      public :
        cInterpolator1D(const tREAL8 & aSzKernel);
        virtual ~cInterpolator1D();

        virtual tREAL8  Weight(const tREAL8 & anX) const = 0;
	const tREAL8 SzKernel() const;  // accessor
      protected :
	tREAL8 mSzKernel;
};



class cBilinInterpolator1D : public cInterpolator1D
{
      public :
        cBilinInterpolator1D();
        tREAL8  Weight(const tREAL8 & anX) const override ;
};


// template <class Type> tREAL8 


/* *************************************************** */
/*                                                     */
/*           cInterpolator1D                           */
/*                                                     */
/* *************************************************** */

cInterpolator1D::cInterpolator1D(const tREAL8 & aSzKernel) :
	mSzKernel (aSzKernel)
{
}

cInterpolator1D::~cInterpolator1D()
{
}

const tREAL8 cInterpolator1D::SzKernel() const {return mSzKernel;}

/* *************************************************** */
/*                                                     */
/*           IM2D/IM2D                                 */
/*                                                     */
/* *************************************************** */

template <> inline  bool cPixBox<2>::InsideInterpolator(const cInterpolator1D & anInterpol,const cPtxd<double,2> & aP) const
{
    tREAL8 aSzK = anInterpol.SzKernel();
    tREAL8 aSzKm1 = aSzK-1.0;
    return   ( round_Uup(aP.x()-aSzKm1) >= tBox::mP0.x()) &&  (round_Ddown(aP.x()+aSzK) <  tBox::mP1.x())
          && ( round_Uup(aP.y()-aSzKm1) >= tBox::mP0.y()) &&  (round_Ddown(aP.y()+aSzK) <  tBox::mP1.y())
    ;
}

static std::vector<tREAL8>  TheBufCoeffX;
static std::vector<tREAL8>  TheBufCoeffY;
template <class Type>  tREAL8 cDataIm2D<Type>::GetValueInterpol(const cPt2dr & aP,const cInterpolator1D & anInterpol) const 
{
    tREAL8 aSzK = anInterpol.SzKernel();
    int aY0 = round_Uup(aP.y() -aSzK);  
    int aY1 = round_Uup(aP.y() +aSzK);
    int aX0 = round_Uup(aP.x() -aSzK); 
    int aX1 = round_Uup(aP.x() +aSzK);

    return 0.0;
}

template class cDataIm2D<tU_INT1>;
template class cDataIm2D<tREAL4>;

/* *************************************************** */
/*                                                     */
/*           cBilinInterpolator1D                      */
/*                                                     */
/* *************************************************** */

cBilinInterpolator1D::cBilinInterpolator1D() :
       cInterpolator1D (1.0)
{
}

tREAL8  cBilinInterpolator1D::Weight(const tREAL8 & anX) const 
{
      return std::max(0.0,1.0-std::abs(anX));
}


/* *************************************************** */
/*                                                     */
/*                ::                                   */
/*                                                     */
/* *************************************************** */


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

