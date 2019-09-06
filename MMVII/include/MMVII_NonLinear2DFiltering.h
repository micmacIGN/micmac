#ifndef  _MMVII_NonLinear2DFiltering_H_
#define  _MMVII_NonLinear2DFiltering_H_
namespace MMVII
{

/** \file  MMVII_NonLinear2DFiltering.h
    \brief Declaration of non linear filters and (for now ?) some basic extraction
*/


/* *********************************************** */
/*                                                 */
/*        Image Filtering                          */
/*                                                 */
/* *********************************************** */

cIm2D<tREAL4> CourbTgt(cIm2D<tREAL4> aImIn);
void SelfCourbTgt(cIm2D<tREAL4> aImIn);

cIm2D<tREAL4> Lapl(cIm2D<tREAL4> aImIn);


/* *********************************************** */
/*                                                 */
/*         Extractions                             */
/*                                                 */
/* *********************************************** */


double  MoyAbs(cIm2D<tREAL4> aImIn); ///< Compute  average of Abs of Image

/// Class to store results of extremum
struct cResultExtremum  
{
     public :
         std::vector<cPt2di>  mPtsMin;
         std::vector<cPt2di>  mPtsMax;
         void Clear();

};

/// compute extrema , ie points for wich I(X) is sup (inf) than any point in a circle of radius aRad
template <class Type> 
void ExtractExtremum1(const cDataIm2D<Type>  &anIm,cResultExtremum & aRes,double aRadius);

/** compute multi scaple extrema , ie points for wich central IC(X) is sup (inf) than any point 
    in a circle of radius aRad to IC , IUp and IDown
*/ 

template <class Type> 
   void ExtractExtremum3
        (
             const cDataIm2D<Type>  &anImUp,  ///< "Up" Image
             const cDataIm2D<Type>  &anImC,   ///<
             const cDataIm2D<Type>  &anImBot,
             cResultExtremum & aRes,
             double aRadius
        );



template <class Type> 
double CubGaussWeightStandardDev(const cDataIm2D<Type>  &anIm,const cPt2di&,double aRadius);

};





#endif  //   _MMVII_NonLinear2DFiltering_H_
