#ifndef  _MMVII_ImageInfoExtract_H_
#define  _MMVII_ImageInfoExtract_H_

#include "MMVII_SysSurR.h"
#include "MMVII_Linear2DFiltering.h"


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

/* *********************************************** */
/*                                                 */
/*         BW Target                               */
/*                                                 */
/* *********************************************** */


struct cParamBWTarget
{
    public :
      cParamBWTarget();

      int NbMaxPtsCC() const; ///<Max number of point (computed from MaxDiam)
      int NbMinPtsCC() const; ///<Min number of point (computed from MinDiam)

      double    mFactDeriche;   ///< Factor for gradient with deriche-method
      int       mD0BW;          ///< distance to border
      double    mValMinW;       ///< Min Value for white
      double    mValMaxB;       ///< Max value for black
      double    mRatioMaxBW;    ///< Max Ratio   Black/White
      double    mMinDiam;       ///< Minimal diameter
      double    mMaxDiam;       ///< Maximal diameter
      double    mPropFr;        ///< Minima prop of point wher frontier extraction suceeded
      int       mNbMinFront;    ///< Minimal number of point
};

struct cSeedBWTarget
{
    public :
       cPt2di mPixW;
       cPt2di mPixTop;

       cPt2di mPInf;
       cPt2di mPSup;

       tREAL4 mBlack;
       tREAL4 mWhite;
       bool   mOk;
       bool   mMarked4Test;

       cSeedBWTarget(const cPt2di & aPixW,const cPt2di & aPixTop,  tREAL4 mBlack,tREAL4 mWhite);
};
enum class eEEBW_Lab : tU_INT1
{
   eFree,
   eBorder,
   eTmp,
   eBadZ,
   eBadFr,
   eElNotOk,
   eBadEl,
   eAverEl,
   eBadTeta
};


class cExtract_BW_Target
{
   public :
        typedef tREAL4              tElemIm;
        typedef cDataIm2D<tElemIm>  tDataIm;
        typedef cIm2D<tElemIm>      tIm;
        typedef cImGrad<tElemIm>    tImGrad;

        typedef cIm2D<tU_INT1>      tImMarq;
        typedef cDataIm2D<tU_INT1>  tDImMarq;

        cExtract_BW_Target(tIm anIm,const cParamBWTarget & aPBWT,cIm2D<tU_INT1> aMasqTest);

        void ExtractAllSeed();
        const std::vector<cSeedBWTarget> & VSeeds() const;
        const tDImMarq&    DImMarq() const;
        const tDataIm &    DGx() const;
        const tDataIm &    DGy() const;

        void SetMarq(const cPt2di & aP,eEEBW_Lab aLab) {mDImMarq.SetV(aP,tU_INT1(aLab));}

        void CC_SetMarq(eEEBW_Lab aLab); ///< set marqer on all connected component


        eEEBW_Lab  GetMarq(const cPt2di & aP) {return eEEBW_Lab(mDImMarq.GetV(aP));}
        bool MarqEq(const cPt2di & aP,eEEBW_Lab aLab) const {return mDImMarq.GetV(aP) == tU_INT1(aLab);}
        bool MarqFree(const cPt2di & aP) const {return MarqEq(aP,eEEBW_Lab::eFree);}

        bool AnalyseOneConnectedComponents(cSeedBWTarget &);
        bool ComputeFrontier(cSeedBWTarget & aSeed);

  protected :

        /// Is the point a candidate for seed (+- local maxima)
        bool IsExtremalPoint(const cPt2di &) ;

        /// Update the data for connected component with a new point (centroid, bbox, heap...)
        void AddPtInCC(const cPt2di &);
        // Prolongat on the vertical, untill its a max or a min
        cPt2di Prolongate(cPt2di aPix,bool IsW,tElemIm & aMaxGy) const;

        /// Extract the accurate frontier point, essentially prepare data to call "cGetPts_ImInterp_FromValue"
        cPt2dr RefineFrontierPoint(const cSeedBWTarget & aSeed,const cPt2di & aP0,bool & Ok);

        tIm              mIm;      ///< Image to analyse
        tDataIm &        mDIm;     ///<  Data of Image
        cPt2di           mSz;      ///< Size of image
        tImMarq          mImMarq;    ///< Marqer used in cc exploration
        tDImMarq&        mDImMarq;   ///< Data of Marqer
        cParamBWTarget   mPBWT;      ///<  Copy of parameters
        tImGrad          mImGrad;    ///<  Structure for computing gradient
        tDataIm &        mDGx;       ///<  Access to x-grad
        tDataIm &        mDGy;       ///<  Access to y-grad

        std::vector<cSeedBWTarget> mVSeeds;

        std::vector<cPt2di>  mPtsCC;
        int                  mIndCurPts;  ///< index of point explored in connected component
        cPt2dr               mCentroid;   ///< Centroid of conected compoonent, used for direction & reduction of coordinates
        cIm2D<tU_INT1>       mMasqTest;   ///< Mask for "special" point where we want to make test (debug/visu ...)
        cDataIm2D<tU_INT1>&  mDMasqT;     ///< Data of Masq

        cPt2di               mPSup;  ///< For bounding box, Sup corner
        cPt2di               mPInf;  ///< For bounding box, Inf corner
        std::vector<cPt2dr>  mVFront;
};


};





#endif  //   _MMVII_ImageInfoExtract_H_
