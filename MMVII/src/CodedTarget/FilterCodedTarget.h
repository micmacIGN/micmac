#ifndef _FILTER_CODED_TARGET_H_
#define _FILTER_CODED_TARGET_H_

/** \file  FilterCodedTarget.h

    \brief contains image processing tools, developed 4 target detection, but
    that may be usefull to other stuff, and then exported later
*/

namespace MMVII
{
namespace  cNS_CodedTarget
{
    class cDCT;
};



/**  mother class of all filters used in detection; all these fiters share the same caracteristics :

        * their definition on the image is just the application to each individual pixel
        * the definition on each pixel is the aggregation of information on their neighboor

    So for definiing a new filter we essentially have to define what we do when a new neighboor
    "arrive"

    This class derive of cMemCheck :  thi allow to check (in debug mode) that all the allocated
    object are freed.
*/


template <class Type>  class  cFilterDCT : public cMemCheck
{
    public :
           typedef cIm2D<Type>     tIm;
           typedef cDataIm2D<Type> tDIm;
           typedef cImGrad<Type>   tImGr;


	   //  ===============  Allocator =============
           static cFilterDCT<Type> * AllocSym(tIm anIm,double aR0,double aR1,double aEpsilon);
           static cFilterDCT<Type> * AllocBin(tIm anIm,double aR0,double aR1);
           static cFilterDCT<Type> * AllocRad(const tImGr & aImGr,double aR0,double aR1,double aEpsilon);

	   virtual ~cFilterDCT();  ///<  X::~X() virtual as there is virtual methods


           virtual void Reset() = 0;
           virtual void Add(const Type & aWeight,const cPt2dr & aNeigh) = 0;
           virtual double Compute() =0;

           /// Some stuff computed by the filter may be of interest for followings
           virtual void UpdateSelected(cNS_CodedTarget::cDCT & aDC) const ;

           double ComputeVal(const cPt2dr & aP);
           tIm    ComputeIm();
           double ComputeValMaxCrown(const cPt2dr & aP,const double& aThreshold);
           tIm    ComputeImMaxCrown(const double& aThreshold);

           eDCTFilters  ModeF() const;

    protected  :
           cFilterDCT(bool IsCumul,eDCTFilters aMode,tIm anIm,bool IsSym,double aR0,double aR1,double aThickN=1.5);
           cFilterDCT (const cFilterDCT<Type> &) = delete;

           void IncrK0(int & aK0);
           void IncrK1(int & aK1,const int & aK0);

           bool                 mIsCumul;
           eDCTFilters          mModeF;
           tIm                  mIm;
           tDIm&                mDIm;
           cPt2di               mSz;
           bool                 mIsSym;
           double               mR0;
           double               mR1;
           double               mThickN;
           cPt2dr               mCurC;
           std::vector<cPt2di>  mIVois;
           std::vector<cPt2dr>  mRVois;
           double               mRhoEnd;
            
           /**   when we want to compute the filter in mode "min on all crown" we will use this
                 vector of pair index;  suppose we have made a computation, the current crown
                 is interval  [10,20[  , and next interval is [13,26[

                    we will have a loop  "supress" for K in  [10-13[  =>  Add(-1,Pix[K])
                    we will have a loop  "add" for K in  [20-26[  =>  Add(+1,Pix[K])
           */

           std::vector<cPt2di>  mVK0K1;
};
template<class TypeEl> cIm2D<TypeEl> ImSymetricity(bool DoCheck,cIm2D<TypeEl> anImIn,double aR0,double aR1,double Epsilon);


/** Class for fine extraction of parameters of target, begins by direction */
template <class Type>  class cExtractDir
{
     public :
         typedef cIm2D<Type>     tIm;   // shared pointer 
         typedef cDataIm2D<Type> tDIm;  // raw reference/pointer for manipulating object
         typedef cNS_CodedTarget::cDCT tDCT;
         typedef std::vector<cPt2dr> tVDir;

         cExtractDir(tIm anIm,double aRhoMin,double aRhoMax);

         /// try the computation of two directions of checkboard, mail fail , return true if sucess
         bool  CalcDir(tDCT &) ; 

         /// computes scores once the direction have been computed
         double ScoreRadiom(tDCT & aDCT) ;
     public :

         /// possible refinement of direction (not so usefull in fact ...)
         cPt2dr OptimScore(const cPt2dr & ,double aStepTeta);
         /// score used in OptimScore
         double Score(const cPt2dr & ,double aTeta); 

          tIm     mIm;  ///< smart pointer , will indicate that we need the object (no to free)
          tDIm&   mDIm;  ///< reference pour acces rapide
          float   mRhoMin;  ///<  Rho Min for correlation
          float   mRhoMax;  ///< Rho max for radiom & correl

          tResFlux                mPtsCrown;   ///< all points of crown
          std::vector<tResFlux>   mVCircles;   ///< vector of circle (1 pixel witdh/ brezenahm like)
          std::vector<tVDir>      mVDIrC;      ///< vector of direction
          float                   mVThrs ;     ///<  threshold Black/white
          tDCT *                  mPDCT;       ///< tested target
       // (SortedVectOfRadius(aR0,aR1,IsSym))
};
bool TestDirDCT(cNS_CodedTarget::cDCT & aDCT,cIm2D<tREAL4> anIm,double aRayCB);

};
#endif // _FILTER_CODED_TARGET_H_

