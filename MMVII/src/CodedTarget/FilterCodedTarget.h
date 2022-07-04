#ifndef _FILTER_CODED_TARGET_H_
#define _FILTER_CODED_TARGET_H_

namespace MMVII
{
template <class Type>  class  cFilterDCT;

namespace  cNS_CodedTarget
{
    class cDCT;
};


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

           std::vector<cPt2di>  mVK0K1;
};
template<class TypeEl> cIm2D<TypeEl> ImSymetricity(bool DoCheck,cIm2D<TypeEl> anImIn,double aR0,double aR1,double Epsilon);
};

#endif // _FILTER_CODED_TARGET_H_

