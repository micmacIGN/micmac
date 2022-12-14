#ifndef _FILTER_CODED_TARGET_H_
#define _FILTER_CODED_TARGET_H_

#include "MMVII_Linear2DFiltering.h"
#include "MMVII_Geom2D.h"
#include "MMVII_SysSurR.h"


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

class cParam1FilterDCT
{
     public :
         cParam1FilterDCT();  ///< default constructor for serialization

         virtual bool        IsSym() const =0;   ///< Being sym is a property constant in the derived class
         virtual bool        IsCumul() const =0;   ///< Being sym is a property constant in the derived class
         virtual eDCTFilters ModeF() const=0;    ///< Mode is a property constant in the derived class
         void AddData(const cAuxAr2007 & anAux); ///< serialization


         double   R0()     const;  ///< accessor
         double   R1()     const;  ///< accessor
         double   ThickN() const;  ///< accessor
     private :
         double   mR0;
         double   mR1;
         double   mThickN;
};


class cParamFilterDCT_Bin : public cParam1FilterDCT
{
     public :
         cParamFilterDCT_Bin();  ///< default constructor for serialization

         bool        IsSym() const override;     ///< No
         bool        IsCumul() const override;   ///< No
         eDCTFilters ModeF() const override;     ///<  eBin

         void AddData(const cAuxAr2007 & anAux); ///< serialization
         double   PropBW() const;  ///< accessor
     private :
         double  mPropBW; ///< prop for estimation of black & white
};

class cParamFilterDCT_Sym : public cParam1FilterDCT
{
     public :
         cParamFilterDCT_Sym();  ///< default constructor for serialization

         bool        IsSym() const override;     ///< true
         bool        IsCumul() const override;   ///< true
         eDCTFilters ModeF() const override;     ///<  eSym

         void AddData(const cAuxAr2007 & anAux); ///< serialization
	 double Epsilon() const;
     private :
	 double mEpsilon;
};

class cParamFilterDCT_Rad : public cParam1FilterDCT
{
     public :
         cParamFilterDCT_Rad();  ///< default constructor for serialization

         bool        IsSym() const override;     ///< false
         bool        IsCumul() const override;   ///< true
         eDCTFilters ModeF() const override;     ///<  eRad

         void AddData(const cAuxAr2007 & anAux); ///< serialization
	 double Epsilon() const;
     private :
	 double mEpsilon;
};



class cParamAllFilterDCT
{
     public :
        void AddData(const cAuxAr2007 & anAux); ///< serialization

        double   RGlob()     const;  ///< accessor
	const cParamFilterDCT_Bin & Bin() const ;
	const cParamFilterDCT_Sym & Sym() const ;
	const cParamFilterDCT_Rad & Rad() const ;

	cParamAllFilterDCT();
     private :
        cParamFilterDCT_Bin  mBin;  ///< default constructor for serialization
        cParamFilterDCT_Sym  mSym;  ///< default constructor for serialization
        cParamFilterDCT_Rad  mRad;  ///< default constructor for serialization
        double  mRGlob; ///< glob multiplier of all ray
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
           static cFilterDCT<Type> * AllocSym(tIm anIm,const cParamAllFilterDCT &);
           // static cFilterDCT<Type> * AllocBin(tIm anIm,double aR0,double aR1);
           static cFilterDCT<Type> * AllocRad(const tImGr & aImGr,const cParamAllFilterDCT &);
           static cFilterDCT<Type> * AllocBin(tIm anIm,const cParamAllFilterDCT &);

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
	   cFilterDCT(tIm anIm,const cParamAllFilterDCT & aGlob,const cParam1FilterDCT *);
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
           double               mThickN; ///< if filter is not cumulative, step of crown in ComputeValMaxCrown
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
         bool  CalcDir(tDCT &, std::vector<cPt2di> &) ;

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
bool TestDirDCT(cNS_CodedTarget::cDCT & aDCT,cIm2D<tREAL4> anIm,double aRayCB, double size_factor, std::vector<cPt2di> & vec2plot);


/* =====================================================================================================
 *    Class for computing saddle point,  in fact, more generally we fit the neigbouhoor of a pixel
 *    by a basis of function, by least square,
 *
 *      in this we fit   I(x,y) =  a + bx +cy + dx^2 ...
 ============================================================================================= */



/**  To be general we use this abstract class for descring a basis of function,
 *   for a given pixel it must return a vector of value for each element of the basis
 */
class cBasisFunc
{
     public :
        virtual  std::pair<tREAL8,std::vector<tREAL8>> WeightAndVals(const cPt2dr &) const = 0;
};

/** Specialisation to quadratic  case   x,y -> {x^2, xy , ... 1} */
class cBasisFuncQuad : public cBasisFunc
{
     public :
        std::pair<tREAL8,std::vector<tREAL8>> WeightAndVals(const cPt2dr &) const override;
};


/**  As typically we will use thousands of time the same neighbourhood, to be efficient 
 * we store as data the values of basis for all pixel of a neighb */

class cCompiledNeighBasisFunc
{
     public   :
         typedef std::vector<cPt2dr> tVNeigh;
         cCompiledNeighBasisFunc(const cBasisFunc & ,const tVNeigh &);

         /** Very basic implementation, slow but safe, will give a fast alternative later
	  *  Typically a faster implemantation will store for once the var/covar matrix and its inverse
	  *  (depend only of the neighbourhood */
         cDenseVect<tREAL8>   SlowCalc(const std::vector<tREAL8> & );
     private  :
         inline void AssertValsIsOk(const std::vector<tREAL8> & aVV)
         {
              MMVII_INTERNAL_ASSERT_tiny(aVV.size()==mNbNeigh,"Emty neigh cCompiledNeighBasisFunc");
         }

         tVNeigh                           mVNeigh;  ///< vector of neighboord
         size_t                            mNbNeigh; ///< number of neighboor, commodity
         std::vector<tREAL8>               mVWeight; ///< possible weight (all 1 for now)
         std::vector<cDenseVect<tREAL8>>   mVFuncs;  ///< for each neighboor contain value on the basis
         size_t                            mNbFunc;  ///< number of functions, commodity
         cLeasSqtAA<tREAL8>                mSys;     ///< least square system
};

/** Class specific to saddle point computation using a least square fitting of neighbourood by quadratic
 * functions */

class cCalcSaddle
{
        public :
             /// Radius of neighbourhood + step of discretization (to test if "aStep" influences accuracy)
             cCalcSaddle(double aRay,double aStep);

	     /// not use 4 now, compute criteria/eigen values
             tREAL8  CalcSaddleCrit(const std::vector<tREAL8> &,bool Show);

	     /// compute refined displacement a  saddle point using values of neihboor
             cPt2dr   RefineSadlPtFromVals(const std::vector<tREAL8> & aVVals,bool Show);
	     /// optimize position by iteration on  RefineSadlPtFromVals
             void RefineSadlePointFromIm(cIm2D<tREAL4> aIm,cNS_CodedTarget::cDCT & aDCT);

        private :
             double                   mRay;   ///< Radius of neighbourhood
             double                   mStep;  ///< Step for discretization
             std::vector<cPt2di>      mVINeigh;  ///<  integer neigh
             size_t                   mNbNeigh;  ///<  nb of neigh
             std::vector<cPt2dr>      mRNeigh;   ///< real neighboor => the on used
             cCompiledNeighBasisFunc  mCalcQuad;  ///< tabulated value on basis
             std::vector<tREAL8>      mVVals;     ///<  buffer for storing values of images
};

/**  Fast Computation of saddle point criteria, for preliminary computation, dont use
     linear decomp on a basis.  See the code for method description
     it returns two images :
 
       - first one has a dynamic proportionnal to images, so it favorise high constast, which is of interest for
         neighbooring filtering
       - firts one is the number of point that complies with the saddle point criteria, it allow a filter
         on absolute criteria
 */

std::pair<cIm2D<tREAL4>,cIm2D<tREAL4>> FastComputeSaddleCriterion(cIm2D<tREAL4>  aIm,double aRay);


};
#endif // _FILTER_CODED_TARGET_H_

