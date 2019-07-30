#ifndef  _MMVII_NonLinear2DFiltering_H_
#define  _MMVII_NonLinear2DFiltering_H_
namespace MMVII
{

/** \file  MMVII_NonLinear2DFiltering.h
    \brief Declaration of non linear filters and (for now ?) some basic extraction
*/

cIm2D<tREAL4> CourbTgt(cIm2D<tREAL4> aImIn);
void SelfCourbTgt(cIm2D<tREAL4> aImIn);

cIm2D<tREAL4> Lapl(cIm2D<tREAL4> aImIn);
double  MoyAbs(cIm2D<tREAL4> aImIn);


struct cResultExtremum
{
     public :
         std::vector<cPt2di>  mPtsMin;
         std::vector<cPt2di>  mPtsMax;
         void Clear();

};

template <class Type> void ExtractExtremum1(const cDataIm2D<Type>  &anIm,cResultExtremum & aRes,double aRadius);

template <class Type> void ExtractExtremum3
                           (
                                const cDataIm2D<Type>  &anImUp,
                                const cDataIm2D<Type>  &anImC,
                                const cDataIm2D<Type>  &anImBot,
                                cResultExtremum & aRes,
                                double aRadius
                           );




};





#endif  //   _MMVII_NonLinear2DFiltering_H_
