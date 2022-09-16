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

/**  Given three consecutive value interpole the extremum assumming a parabol, no value if alligned
*/
std::optional<double>  InterpoleExtr(double V1,double V2,double V3);
/**  Given three consecutive value interpole the extremum assumming a parabol
*/
double  StableInterpoleExtr(double V1,double V2,double V3);



template <class Type> double  MoyAbs(cIm2D<Type> aImIn); ///< Compute  average of Abs of Image
template <class Type> cPt2dr   ValExtre(cIm2D<Type> aImIn); ///< X -> Min, Y -> Max

/// Class to store results of extremum
struct cResultExtremum  
{
     public :
         std::vector<cPt2di>  mPtsMin;
         std::vector<cPt2di>  mPtsMax;
         void Clear();

         cResultExtremum(bool DoMin=true,bool DoMax=true);
         bool mDoMin;
         bool mDoMax;
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

template <class Type> class cAffineExtremum
{
    public :
       cAffineExtremum(const cDataIm2D<Type>  &anIm,double aRadius);
       cPt2dr OneIter(const cPt2dr &);
       cPt2dr StdIter(const cPt2dr &,double Epsilon,int aNbIterMax); ///< aNbIterMax both res and val
       const cDataIm2D<Type>  &  Im() const;
    private :
       const cDataIm2D<Type>  &  mIm;
       double                    mRadius;
       double                    mSqRad;
       cRect2                    mBox;
       cLeasSqtAA<tREAL4>        mSysPol;
       cDenseVect<tREAL4>        mVectPol;
       cDenseMatrix<tREAL4>      mMatPt;
       cDenseVect<tREAL4>        mVectPt;
       int                       mNbIter;
       double                    mDistIter;
};


};





#endif  //   _MMVII_ImageInfoExtract_H_
