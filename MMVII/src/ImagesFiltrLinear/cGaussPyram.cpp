#include "include/MMVII_all.h"
#include "include/MMVII_Tpl_Images.h"


namespace MMVII
{

/* ==================================================== */
/*                                                      */
/*              cGP_OneImage                            */
/*                                                      */
/* ==================================================== */

template <class Type> cGP_OneImage<Type>::cGP_OneImage(tOct * anOct,int aNumInOct,tGPIm * anImUp) :
    mOct       (anOct),
    mPyr       (mOct->Pyram()),
    mNumInOct  (aNumInOct),
    mUp        (anImUp),
    mDown      (nullptr),
    mOverlapUp (nullptr),
    mImG       (mOct->SzIm()),
    mIsTopPyr  ((anImUp==nullptr) && (mOct->Up()==nullptr))
{
    if (mUp==nullptr)
    {
       mScaleAbs = anOct->Scale0Abs(); // initial value of octave
    }
    else
    {
       mUp->mDown = this; // create reciprocate link
       mScaleAbs =  mUp->mScaleAbs * mPyr->MulScale();  // geometric law
    }
    // Get possible image corresponding to same sigma in up octave
    if (mOct->Up() != nullptr)
    {
        mOverlapUp = mOct->Up()->ImageOfScaleAbs(mScaleAbs);
    }
    mScaleInO = mScaleAbs / anOct->Scale0Abs(); // This the relative sigma between different image

    mTargetSigmAbs = mPyr->SigmIm0() * mScaleAbs;
    mTargetSigmInO = mTargetSigmAbs / anOct->Scale0Abs();
}

template <class Type> void cGP_OneImage<Type>::ComputGaussianFilter()
{
   if (mIsTopPyr)  // Case top image do not need filtering, except if specified in ConvolIm0
   {
       double aCI0 = mPyr->Params().mConvolIm0;
       if (aCI0!=0.0)
       {           
          ExpFilterOfStdDev(mImG.DIm(),mPyr->Params().mNbIter1,aCI0);
       }
   }
   else if (mOverlapUp != nullptr)  // Case were there exist an homologous image, use decimation
   {
       mImG.DecimateInThis(2,mOverlapUp->mImG);
   }
   else if (mUp != nullptr)  // Other case comput from Up image
   {
        // To maximize the gaussian aspect, compute from already filtered image
        double aSig = DifSigm(mTargetSigmInO,mUp->mTargetSigmInO);
        // First convulotion use more iteration as we dont benefit from accumulation
        int aNbIter = mPyr->Params().mNbIter1 - mUp->mNumInOct;
        // Be sure to have a minimum
        aNbIter = std::max(aNbIter,mPyr->Params().mNbIterMin);
        ExpFilterOfStdDev(mImG.DIm(),mUp->mImG.DIm(),aNbIter,aSig);
   }
   else  
   {
        // This should never happen by construction
        MMVII_INTERNAL_ASSERT_strong(mUp!=nullptr,"Up Image in ComputGaussianFilter");
   }
}

template <class Type> void cGP_OneImage<Type>::MakeDiff(const tGPIm &  aImDif)
{
   MMVII_INTERNAL_ASSERT_strong(aImDif.mDown!=nullptr,"Down Image in MakeDiff");

   DiffImageInPlace(mImG.DIm(),aImDif.mImG.DIm(),aImDif.mDown->mImG.DIm());
}

template <class Type> void cGP_OneImage<Type>::SaveInFile() const
{
    const std::string & aPref = mPyr->Params().mPrefSave;
    if (aPref=="")
       return ;

    std::string aName = aPref + "-" + ShortId() + ".tif";
    mImG.DIm().ToFile(aName);
}



template <class Type> void cGP_OneImage<Type>::Show()  const
{
    StdOut() << "    "  << Id()  
             << " " << (mOverlapUp ? ("Ov:"+mOverlapUp->ShortId()) : "Ov:XXXXXX")  << "\n";
}

template <class Type> std::string cGP_OneImage<Type>::Id()  const
{
   return     
              " Oc:" + ToStr(mOct->NumInPyr()) 
          +   " Im:" + ToStr(mNumInOct) 
          +   " SOct:" + FixDigToStr(mScaleInO,4)   +  "/" + FixDigToStr(mTargetSigmInO,4)
          +   " SAbs:" + FixDigToStr(mScaleAbs,4)   +  "/" + FixDigToStr(mTargetSigmAbs,4)
          +   " T:"   + std::string(mIsTopPyr ? "1" : "0") 
   ;
}
template <class Type> std::string cGP_OneImage<Type>::ShortId()  const
{
   return     
              "o" + ToStr(mOct->NumInPyr()) +   "_i" + ToStr(mNumInOct)  ;
}

       // ==== Accessors ====
template <class Type> cIm2D<Type> cGP_OneImage<Type>::ImG() {return mImG;}
template <class Type> double  cGP_OneImage<Type>::ScaleAbs() const {return mScaleAbs;}


/* ==================================================== */
/*                                                      */
/*              cGP_OneOctave                           */
/*                                                      */
/* ==================================================== */

template <class Type> cGP_OneOctave<Type>::cGP_OneOctave(tPyr * aPyr,int aNum,tOct * anOctUp) :
   mPyram (aPyr),
   mUp    (anOctUp),
   mDown  (nullptr),
   mNumInPyr  (aNum),
   mSzIm      (-1,-1)  // defined later
{
    if (mUp==nullptr)
    {
        mSzIm = aPyr->SzIm0();        // Sz Max
        mScale0Abs  = aPyr->Scale0();  // If Top Octave, set sigma top image to s0
    }
    else
    {
       mScale0Abs  = mUp->Scale0Abs() * 2;  // geometric law
       mSzIm = mUp->mSzIm / 2;            // octave law
       mUp->mDown = this; // create reciprocate link
    }
    // StdOut() << "=== Oct === " << mNumInPyr << " Sz=" << mSzIm << "\n";
    tGPIm * aPrec = nullptr;
    for (int aKIm=0 ; aKIm<mPyram->NbImByOct()  ; aKIm++)
    {
        mVIms.push_back(tSP_GPIm(new tGPIm(this,aKIm,aPrec)));
        aPrec = mVIms.back().get();
    }
}


template <class Type> cGP_OneImage<Type>* cGP_OneOctave<Type>::ImageOfScaleAbs(double aScale,double aTolRel)
{
   for (auto & aPtr : mVIms)
       if (RelativeDifference(aPtr->ScaleAbs(),aScale)<aTolRel)
          return  aPtr.get();
  return nullptr;
}

template <class Type> void cGP_OneOctave<Type>::MakeDiff(const tOct & anOct)
{
    for (int aKIm=0 ; aKIm<int(mVIms.size()) ; aKIm++)
    {
         mVIms.at(aKIm)->MakeDiff(*anOct.mVIms.at(aKIm));
    }
}

template <class Type> void cGP_OneOctave<Type>::ComputGaussianFilter()
{
   for (auto & aPtrIm : mVIms)
       aPtrIm->ComputGaussianFilter();
}

template <class Type> void cGP_OneOctave<Type>::Show() const
{
    StdOut() << "   --- Oct --- " << mNumInPyr << " Sz=" << mSzIm << "\n";
    for (const auto & aPtrIm : mVIms)
       aPtrIm->Show();
}

template <class Type> void cGP_OneOctave<Type>::SaveInFile() const
{
   for (auto & aPtrIm : mVIms)
       aPtrIm->SaveInFile();
}



template <class Type> cIm2D<Type> cGP_OneOctave<Type>::ImTop() {return mVIms.at(0)->ImG();}
    //  Accessors

template <class Type> cGaussianPyramid<Type>* cGP_OneOctave<Type>::Pyram() const {return mPyram;}
template <class Type> const cPt2di & cGP_OneOctave<Type>::SzIm() const {return mSzIm;}
template <class Type> const double & cGP_OneOctave<Type>::Scale0Abs() const {return mScale0Abs;}
template <class Type> cGP_OneOctave<Type>* cGP_OneOctave<Type>::Up() const {return mUp;}
template <class Type> const int & cGP_OneOctave<Type>::NumInPyr() const {return mNumInPyr;}

/* ==================================================== */
/*                                                      */
/*              cGP_Params                              */
/*                                                      */
/* ==================================================== */

cGP_Params::cGP_Params(const cPt2di & aSzIm0,int aNbOct,int aNbLevByOct,int aOverlap) :
   mSzIm0        (aSzIm0),
   mNbOct        (aNbOct),
   mNbLevByOct   (aNbLevByOct),
   mNbOverlap    (aOverlap),
   mConvolIm0    (0.0),
   mNbIter1      (4),
   mNbIterMin    (2),
   mPrefSave     ("")
{
}


/* ==================================================== */
/*                                                      */
/*              cGaussianPyramid                        */
/*                                                      */
/* ==================================================== */

template <class Type> cGaussianPyramid<Type>::cGaussianPyramid(const cGP_Params & aParams) :
   mParams            (aParams),
   mMulScale          (pow(2.0,1/double(mParams.mNbLevByOct))),
   mScale0            (1.0),
   mEstimSigmInitIm0  (DefStdDevImWellSample),
   mSigmIm0           (SomSigm(mEstimSigmInitIm0,mParams.mConvolIm0))
{
   tOct * aPrec = nullptr;
   for (int aKOct=0 ; aKOct<mParams.mNbOct ; aKOct++)
   {
       mVOct.push_back(tSP_Oct(new tOct(this,aKOct,aPrec)));
       aPrec = mVOct.back().get();
   }
}


template <class Type>  std::shared_ptr<cGaussianPyramid<Type>>  
      cGaussianPyramid<Type>::Alloc(const cGP_Params & aParams)
{
   return  tSP_Pyr(new tPyr(aParams));
}

template <class Type> void cGaussianPyramid<Type>::Show() const
{
   StdOut() << "============ Gaussian Pyramid ==============\n";
   StdOut() << "     type elem: " <<  E2Str(tElemNumTrait<Type>:: TyNum())  << "\n";
   for (const auto & aPtrOct : mVOct)
       aPtrOct->Show();
}

template <class Type>  std::shared_ptr<cGaussianPyramid<Type>>  
      cGaussianPyramid<Type>::PyramDiff() const
{
    cGP_Params aParam = mParams;
    aParam.mNbOverlap--;
    // Not Sure thi would create a problem, but it would not be very coherent ?
    MMVII_INTERNAL_ASSERT_strong(aParam.mNbOverlap>=0,"No overlap for PyramDiff");

    tSP_Pyr aRes = Alloc(aParam);

    for (int aKo=0 ; aKo<int(mVOct.size()) ; aKo++)
    {
        aRes->mVOct.at(aKo)->MakeDiff(*mVOct.at(aKo));
    }


    return aRes;
}

template <class Type> void cGaussianPyramid<Type>::ComputGaussianFilter()
{
   for (const auto & aPtrOct : mVOct)
       aPtrOct->ComputGaussianFilter();
}

template <class Type> void cGaussianPyramid<Type>::SaveInFile() const
{
   for (auto & aPtrOct : mVOct)
       aPtrOct->SaveInFile();
}

template <class Type> void cGaussianPyramid<Type>::SetPrefSave(const std::string& aPref)
{
  mParams.mPrefSave = aPref;
}

template <class Type> cIm2D<Type> cGaussianPyramid<Type>::ImTop() {return mVOct.at(0)->ImTop();}

template <class Type> const cGP_Params & cGaussianPyramid<Type>::Params() const {return mParams;}
template <class Type> const double & cGaussianPyramid<Type>::MulScale() const {return mMulScale;}
template <class Type> int  cGaussianPyramid<Type>::NbImByOct() const {return  mParams.mNbLevByOct+mParams.mNbOverlap;}
template <class Type> const cPt2di & cGaussianPyramid<Type>::SzIm0() const {return  mParams.mSzIm0;}
template <class Type> const double & cGaussianPyramid<Type>::Scale0() const {return  mScale0;}
template <class Type> const double & cGaussianPyramid<Type>::SigmIm0() const {return mSigmIm0;}

/* ==================================================== */
/*              INSTANTIATION                           */
/* ==================================================== */

#define MACRO_INSTANTIATE_GaussPyram(Type) \
template class cGP_OneImage<Type>;\
template class cGP_OneOctave<Type>;\
template class cGaussianPyramid<Type>;\

MACRO_INSTANTIATE_GaussPyram(tREAL4)
MACRO_INSTANTIATE_GaussPyram(tINT2)



};
