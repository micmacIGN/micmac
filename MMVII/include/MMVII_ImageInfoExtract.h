#ifndef  _MMVII_ImageInfoExtract_H_
#define  _MMVII_ImageInfoExtract_H_
namespace MMVII
{

/** \file MMVII_ImageInfoExtract.h 
    \brief Declaration of operation on image,  extracting information local or global
*/



/* *********************************************** */
/*                                                 */
/*         Extractions                             */
/*                                                 */
/* *********************************************** */


double  MoyAbs(cIm2D<tREAL4> aImIn); ///< Compute  average of Abs of Image
template <class Type> cPt2dr   ValExtre(cIm2D<Type> aImIn); ///< X -> Min, Y -> Max

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





#endif  //   _MMVII_ImageInfoExtract_H_
