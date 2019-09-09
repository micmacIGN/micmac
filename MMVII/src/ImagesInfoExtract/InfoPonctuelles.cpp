#include "include/MMVII_all.h"


namespace MMVII
{


template <class Type>
double CubGaussWeightStandardDev(const cDataIm2D<Type>  &anIm,const cPt2di& aC,double aRadius)
{
    cRect2 aDBox = cRect2::BoxWindow(round_up(aRadius));  //! Centerd box arround (0,0)
    double aSqRad = Square(aRadius);
    cComputeStdDev<1>  aCSD;

    for (const auto & aDP : aDBox)
    {
         double aN2 = SqN2(aDP);    // Square norm of displacement
         if (aN2 < aSqRad) // If in disk
         {
             cPt2di aP = aC + aDP;
             if (anIm.Inside(aP))
             {
                 double aRatio = std::sqrt(aN2) / aRadius;
                 double aWeight = CubAppGaussVal(aRatio);
                 double aVal =  anIm.GetV(aP);
                 aCSD.Add(&aVal,aWeight);
             }
         }
    }
    return aCSD.ComputeUnBiasedVar()[0];
}


#define MACRO_INSTANTIATE_INFO_PONCT(TYPE)\
template double CubGaussWeightStandardDev(const cDataIm2D<TYPE>  &anIm,const cPt2di&,double aRadius);


MACRO_INSTANTIATE_INFO_PONCT(tREAL4)
MACRO_INSTANTIATE_INFO_PONCT(tINT2)
};