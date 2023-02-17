#include "MMVII_Tpl_Images.h"
#include "V1VII.h"
#include "MMVII_2Include_Serial_Tpl.h"
#include "MMVII_Linear2DFiltering.h"
#include "MMVII_NonLinear2DFiltering.h"
#include "MMVII_Geom2D.h"
#include "MMVII_ImageInfoExtract.h"
#include "MMVII_AimeTieP.h"

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
    mIsTopPyr  ((anImUp==nullptr) && (mOct->Up()==nullptr)),
    mNumMaj    (-1),
    mBestEquiv (nullptr)
{
    mNameSave =  mPyr->Params().mAppli->NamePCarImage(mPyr->NameIm(),mPyr->TypePyr(),ShortId(),mPyr->Params().mNumTile);

/*
StdOut() << "SssSSSS=" << mNameSave << "\n";
    mNameSave =      mPyr->Params().mPrefixSave
                   + "-Ima-" 
                   + E2Str(mPyr->TypePyr()) 
                   + "-" + ShortId() 
                   // + "-" +  Prefix(mPyr->NameIm())
                   + "-" +  mPyr->Prefix()
                   + ".tif"
                ;
StdOut() << "XXXxxxX=" << mNameSave << "\n";
*/
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

    // This is the "targeted" sigma we want to reach in the "absolute" (no decimation)
    //  scaled image it suppose that we know the  the sigma of first image, then simply multiply by it
    mTargetSigmAbs = mPyr->SigmIm0() * mScaleAbs;
   // For the value in octave, just divide by decimation factor
    mTargetSigmInO = mTargetSigmAbs / anOct->Scale0Abs();
}


template <class Type> void cGP_OneImage<Type>::SetBestEquiv(int aKMaj,tGPIm * aBestEquiv)
{
   mNumMaj = aKMaj;
   mBestEquiv = aBestEquiv;
}


     //  === Image processing methods

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

template <class Type> void cGP_OneImage<Type>::MakeCorner()
{
    // Compute Curvature tangent to level lines
    SelfCourbTgt(mImG);
    // Normalise of scale; theory && experiment show a dependance in pow 3
    SelfMulImageCsteInPlace(mImG.DIm(),pow(mScaleInO,3));
}



template <class Type> void cGP_OneImage<Type>::MakeDiff(const tGPIm &  aImDif)
{
   MMVII_INTERNAL_ASSERT_strong(aImDif.mDown!=nullptr,"Down Image in MakeDiff");

   DiffImageInPlace(mImG.DIm(),aImDif.mImG.DIm(),aImDif.mDown->mImG.DIm());
}

template <class Type> void cGP_OneImage<Type>::MakeOrigNorm(const tGPIm &  aOriGPI)
{
    // Create a duplicata of original image
    tIm aImBlur = aOriGPI.mImG.Dup();

    // Put in  aImBlur the convolution with  Gaussian filter
    ExpFilterOfStdDev(aImBlur.DIm(),3,mTargetSigmInO*mPyr->Params().mScaleDirOrig);
    // Put In Res, diff between gaussian filter an Ori
    DiffImageInPlace(mImG.DIm(),aOriGPI.mImG.DIm(),aImBlur.DIm());
    // Multiply by ScaleInO
    SelfMulImageCsteInPlace(mImG.DIm(),pow(mScaleInO,1));
}

     //  === Export

template <class Type> cPt2dr cGP_OneImage<Type>::Im2File(const cPt2dr & aP) const
{
    return mOct->Oct2File(aP);
}

template <class Type> cPt2dr cGP_OneImage<Type>::File2Im(const cPt2dr & aP) const
{
    return mOct->File2Oct(aP);
}

template <class Type> void cGP_OneImage<Type>::SaveInFile() const
{
    mImG.DIm().ToFile(mNameSave);
    // MakeStdIm8BIts(mImG,mNameSave);
}

template <class Type> bool   cGP_OneImage<Type>::IsTopOct() const
{
    return  mUp == nullptr;
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
template <class Type> double  cGP_OneImage<Type>::ScaleInO() const {return mScaleInO;}

template <class Type> cGP_OneImage<Type> * cGP_OneImage<Type>::Up() const {return mUp;}
template <class Type> cGP_OneImage<Type> * cGP_OneImage<Type>::Down() const {return mDown;}
template <class Type> const std::string & cGP_OneImage<Type>::NameSave() const {return mNameSave;}
template <class Type> cGP_OneOctave<Type> * cGP_OneImage<Type>::Oct() const {return mOct;}

template <class Type> cGP_OneImage<Type> * cGP_OneImage<Type>::BestEquiv() const {return mBestEquiv;}

template <class Type> int cGP_OneImage<Type>::NumMaj() const {return mNumMaj;}

template <class Type> cGP_OneImage<Type> * cGP_OneImage<Type>::ImOriHom() 
{
   return mPyr->ImHomOri(this);
}

template <class Type> cGaussianPyramid<Type>& cGP_OneImage<Type>::Pyr() {return *mPyr;}

template <class Type> int  cGP_OneImage<Type>::NumInOct() const {return mNumInOct;}

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

template <class Type> void cGP_OneOctave<Type>::MakeOrigNorm(const tOct & anOct)
{
    for (int aKIm=0 ; aKIm<int(mVIms.size()) ; aKIm++)
    {
         mVIms.at(aKIm)->MakeOrigNorm(*anOct.mVIms.at(aKIm));
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

template <class Type> cPt2dr cGP_OneOctave<Type>::Oct2File(const cPt2dr & aP) const
{
    return mPyram->Pyr2File(aP*mScale0Abs);
}

template <class Type> cPt2dr cGP_OneOctave<Type>::File2Oct(const cPt2dr & aP) const
{
    return mPyram->File2Pyr(aP)/mScale0Abs;
}




template <class Type> cIm2D<Type> cGP_OneOctave<Type>::ImTop() const {return mVIms.at(0)->ImG();}
template <class Type> cGP_OneImage<Type>* cGP_OneOctave<Type>::GPImTop() const {return mVIms.at(0).get();}
    //  Accessors

template <class Type> cGaussianPyramid<Type>* cGP_OneOctave<Type>::Pyram() const {return mPyram;}
template <class Type> const cPt2di & cGP_OneOctave<Type>::SzIm() const {return mSzIm;}
template <class Type> const double & cGP_OneOctave<Type>::Scale0Abs() const {return mScale0Abs;}
template <class Type> cGP_OneOctave<Type>* cGP_OneOctave<Type>::Up() const {return mUp;}
template <class Type> const int & cGP_OneOctave<Type>::NumInPyr() const {return mNumInPyr;}
template <class Type> const  std::vector<std::shared_ptr<cGP_OneImage<Type>>> & cGP_OneOctave<Type>::VIms() const {return mVIms;}

/* ==================================================== */
/*                                                      */
/*              cFilterPCar                             */
/*                                                      */
/* ==================================================== */

typedef std::vector<double> tVD;


cFilterPCar::cFilterPCar(bool is4TieP) :
   mIsForTieP (is4TieP),
   mAutoC     ({0.9}),
   mPSF       ({35,3.0,0.2}),
   mEQsf      ({2,1,1}),
   mLPCirc    (is4TieP ? tVD({2.5,-1.0,2.0}) :  tVD({2.0,0.0,1.0})),
   mLPSample  (is4TieP ? tVD{16,8,32.0,1,0}  :  tVD({6,6,32.0,1,1})),
   mLPQuantif ({10.0,0.5})
{
}

bool cFilterPCar::IsForTieP() const
{
   return mIsForTieP;
}

cFilterPCar::~cFilterPCar() 
{
   Check();
}

void cFilterPCar::SetLPSample(const std::vector<double> & aVS)
{
   mLPSample = aVS;
}

void cFilterPCar::SetLPCirc(const std::vector<double> & aVC)
{
   mLPCirc = aVC;
}

void cFilterPCar::Check()
{
    if (The_MMVII_DebugLevel>=The_MMVII_DebugLevel_InternalError_strong)
    {
         MMVII_INTERNAL_ASSERT_strong(AutoC().size()   ==3,"Bad size for cFilterPCar::AutoC");
         MMVII_INTERNAL_ASSERT_strong(PSF().size()     ==3,"Bad size for cFilterPCar::PSF");
         MMVII_INTERNAL_ASSERT_strong(EQsf().size()    ==3,"Bad size for cFilterPCar::EQsf");
         MMVII_INTERNAL_ASSERT_strong(LPCirc().size()  ==3,"Bad size for cFilterPCar::LPCirc");
         MMVII_INTERNAL_ASSERT_strong(LPSample().size()==5,"Bad size for cFilterPCar::LPSample");
         MMVII_INTERNAL_ASSERT_strong(LPQuantif().size()==2,"Bad size for cFilterPCar::LPQuantif");

         LPC_DeltaI0();
         LPC_DeltaIm();
         LPS_NbTeta();
         LPS_NbRho();
         LPS_CensusMode();
         LPS_Interlaced();
    }
}

void cFilterPCar::FinishAC(double aVal)
{
    for (int aK=1 ; aK< 3 ; aK++)
    {
        if (int(mAutoC.size()) == aK)
           mAutoC.push_back(mAutoC.back()-aVal);
    }
// StdOut() << "HhhhhhhHHhhhhhhh " << mAutoC.size() << " :: " << mAutoC << "\n";
}

std::vector<double> &  cFilterPCar::AutoC() {return mAutoC;}
const double & cFilterPCar::AC_Threshold() const {return mAutoC.at(0);}
const double & cFilterPCar::AC_CutReal()   const  {return mAutoC.at(1);}
const double & cFilterPCar::AC_CutInt()    const {return mAutoC.at(2);}

std::vector<double> &  cFilterPCar::PSF() {return mPSF;}
const double & cFilterPCar::DistSF()       const {return mPSF.at(0);}
const double & cFilterPCar::MulDistSF()    const {return mPSF.at(1);}
const double & cFilterPCar::PropNoSF()     const {return mPSF.at(2);}

std::vector<double> &  cFilterPCar::EQsf() {return mEQsf;}
const double & cFilterPCar::PowAC()        const {return mEQsf.at(0);}
const double & cFilterPCar::PowVar()       const {return mEQsf.at(1);}
const double & cFilterPCar::PowScale()     const {return mEQsf.at(2);}


std::vector<double> &  cFilterPCar::LPCirc() {return mLPCirc;}
const double &  cFilterPCar::LPC_Rho0() const     {return mLPCirc.at(0);}
int             cFilterPCar::LPC_DeltaI0() const  {return EmbeddedIntVal(mLPCirc.at(1));}
int             cFilterPCar::LPC_DeltaIm() const  {return EmbeddedIntVal(mLPCirc.at(2));}

std::vector<double> &  cFilterPCar::LPSample() {return mLPSample;}
int  cFilterPCar::LPS_NbTeta()           const {return EmbeddedIntVal(mLPSample.at(0));}
int  cFilterPCar::LPS_NbRho()            const {return EmbeddedIntVal(mLPSample.at(1));}
const double &  cFilterPCar::LPS_Mult()  const {return mLPSample.at(2);}
bool  cFilterPCar::LPS_CensusMode()      const {return EmbeddedBoolVal(mLPSample.at(3));}
bool  cFilterPCar::LPS_Interlaced()      const {return EmbeddedBoolVal(mLPSample.at(4));}

std::vector<double> &  cFilterPCar::LPQuantif()        {return mLPQuantif;}
const double &         cFilterPCar::LPQ_Steep0() const {return mLPQuantif.at(0);}
const double &         cFilterPCar::LPQ_Exp()    const {return mLPQuantif.at(1);}



/*
   std::vector<double>  mLPSample;  ///< Sampling Mode for LogPol [NbTeta,NbRho,Multiplier,CensusNorm]
         int               LPS_NbTeta()     const;   ///< Number of sample in teta
         int               LPS_NbRho()      const;   ///< Number of sample in rho
         const double &    LPS_Mult()       const;   ///< Multiplier before making it integer
         bool              LPS_CensusMode() const;   ///< Do Normalization in census mode
*/

void cFilterPCar::InitDirTeta() const
{
   int aNbTeta = LPS_NbTeta();
   if (int(mVDirTeta0.size()) == aNbTeta)
      return;
    mVDirTeta0.clear();
    mVDirTeta1.clear();
 
   for (int aKTeta=0 ; aKTeta<aNbTeta ; aKTeta++)
   {
       mVDirTeta0.push_back(FromPolar(1.0,(M_PI*(2*aKTeta))/aNbTeta));
   }
   if (LPS_Interlaced())
   {
       for (int aKTeta=0 ; aKTeta<aNbTeta ; aKTeta++)
       {
           mVDirTeta1.push_back(FromPolar(1.0,(M_PI*(1+2*aKTeta))/aNbTeta));
       }
   }
   else
       mVDirTeta1 = mVDirTeta0;
}

const std::vector<cPt2dr> & cFilterPCar::VDirTeta0() const
{
   InitDirTeta();

   return mVDirTeta0;
}

const std::vector<cPt2dr> & cFilterPCar::VDirTeta1() const
{
   InitDirTeta();

   return mVDirTeta1;
}

void cFilterPCar::AddData(const cAuxAr2007 & anAux)
{
    MMVII::AddData(cAuxAr2007("Is4TieP",anAux)     , mIsForTieP);
    MMVII::AddData(cAuxAr2007("AutoC",anAux)       , mAutoC);
    MMVII::AddData(cAuxAr2007("SpaceFilter",anAux) , mPSF);
    MMVII::AddData(cAuxAr2007("ExpQuality",anAux)  , mEQsf);
    MMVII::AddData(cAuxAr2007("Circles",anAux)     , mLPCirc);
    MMVII::AddData(cAuxAr2007("Sampling",anAux)    , mLPSample);
    MMVII::AddData(cAuxAr2007("Quantif",anAux)     , mLPQuantif);
    // Don't save mVDirTetaX  as they are computed from others
}

/*
         std::vector<double>  mPSF; ///< Param Spatial Filtering  [Dist,MulRAy,PropNoFS]
         std::vector<double>  mEQsf; ///< Exposant for quality of point before spatial filter [AutoC,Var,Scale]
         std::vector<double>  mLPCirc;  ///< Circles of Log Pol param [Rho0,DeltaSI0,DeltaI]
         std::vector<double>  mLPSample;  ///< Sampling Mode for LogPol [NbTeta,NbRho,Multiplier,CensusNorm]
         std::vector<double>  mLPQuantif; 
*/

void AddData(const cAuxAr2007 & anAux, cFilterPCar &    aFPC)
{
    aFPC.AddData(anAux);
}



/* ==================================================== */
/*                                                      */
/*              cGP_Params                              */
/*                                                      */
/* ==================================================== */

cGP_Params::cGP_Params(const cPt2di & aSzIm0,int aNbOct,int aNbLevByOct,int aOverlap,const cMMVII_Appli * aPtrAppli,bool is4TieP) :
   mSzIm0        (aSzIm0),
   mNbOct        (aNbOct),
   mNbLevByOct   (aNbLevByOct),
   mNbOverlap    (aOverlap),
   mAppli        (aPtrAppli),
   mFPC          (is4TieP),
   mConvolIm0    (0.0),
   mNbIter1      (4),
   mNbIterMin    (2),
   mConvolC0     (1.0),
   mScaleDirOrig (10.0),
   mEstimSigmInitIm0  (DefStdDevImWellSample)
{
}


/* ==================================================== */
/*                                                      */
/*              cGaussianPyramid                        */
/*                                                      */
/* ==================================================== */


          //  - * - * - * - * - * - * - * - * - * - *
          //   Creators : Constructors + Allocators

template <class Type> cGaussianPyramid<Type>::cGaussianPyramid
                      (
                            const cGP_Params & aParams,
                            tPyr * aOrig,
                            eTyPyrTieP aType,
                            const std::string & aNameIm,
                            const cRect2 & aBIn,
                            const cRect2 & aBOut
                      ) :
   mPyrOrig           (aOrig ? aOrig : this),
   mTypePyr           (aType),
   mNameIm            (aNameIm),
   mBoxIn             (aBIn),
   mBoxOut            (aBOut),
   mParams            (aParams),
   mMulScale          (pow(2.0,1/double(mParams.mNbLevByOct))),
   mScale0            (1.0),
   mEstimSigmInitIm0  (aParams.mEstimSigmInitIm0),
   mSigmIm0           (SomSigm(mEstimSigmInitIm0,mParams.mConvolIm0))
{
   tOct * aPrec = nullptr;
   // Creates octaves who creates images ... 
   //  also create a vector of octaves and  images 
   for (int aKOct=0 ; aKOct<mParams.mNbOct ; aKOct++)
   {
       mVOcts.push_back(tSP_Oct(new tOct(this,aKOct,aPrec)));
       aPrec = mVOcts.back().get();
       std::copy(aPrec->VIms().begin(),aPrec->VIms().end(),std::back_inserter(mVAllIms));
   }

   // Images with same scale overlap (for computing difference and max loc) , but it is 
   // usefull to have a unique scale representation (for example when extracting LogPol image)
   {
       int aKMaj=0;
       for (int aK=0 ; aK<int(mVAllIms.size()) ; aK++)
       {
           tGPIm * aPIm = mVAllIms[aK].get(); // The image
           tOct  * aOct = aPIm->Oct();  // The octave
           tOct  * aOctUp = aOct->Up();  // The possible octave with better resol
           tGPIm * aPImUpEqui = (aOctUp ? aOctUp->ImageOfScaleAbs(aPIm->ScaleAbs()) : nullptr); // The possible im, same scale, better resol
           tGPIm * aPImBestEqui = (aPImUpEqui==nullptr) ? aPIm : aPImUpEqui;

           aPIm->SetBestEquiv(aKMaj,aPImBestEqui);

/*
           int aNumMaj = (aPImUpEqui==nullptr)? aKMaj : -1;
           StdOut() << "KKK " << aK 
                    <<  " O=" << aOct->NumInPyr() 
                    <<  " I=" << aPIm->NumInOct() 
                    <<  " NMaj=" << aNumMaj
                    <<  " RSA="  << aPIm->ScaleAbs() / aPImBestEqui->ScaleAbs() 
                    <<  " S=" << ((aPImUpEqui==nullptr) ? "--" : "**")
                    << "\n";
*/
           if (aPImUpEqui==nullptr)
           {
              aKMaj++;
              mVMajIm.push_back(aPIm);
           }
       }
       for (int aK=1 ; aK<int(mVMajIm.size()) ; aK++)
       {
           double aRatio = mVMajIm[aK]->ScaleAbs()/mVMajIm[aK-1]->ScaleAbs();
           //  StdOut() << " sSssSSs= " << aRatio << " MM=" <<  mMulScale     << "\n";
           MMVII_INTERNAL_ASSERT_strong(std::abs(aRatio-mMulScale)<1e-5,"Ratio in MajorVectImage");
       }
/*
       for (const auto & aPIm : mVMajIm)
          StdOut() << "MAJJJ " 
                    <<  " O=" << aPIm->Oct()->NumInPyr() 
                    <<  " I=" << aPIm->NumInOct() 
                   << " SA="<< aPIm->ScaleAbs() << "\n";
       getchar();
*/
   }
}


template <class Type>  std::shared_ptr<cGaussianPyramid<Type>>  
      cGaussianPyramid<Type>::Alloc(const cGP_Params & aParams,const std::string& aNameIm,const cRect2 & aBIn,const cRect2 & aBOut)
{
   return  tSP_Pyr(new tPyr(aParams,nullptr,eTyPyrTieP::eTPTP_Init,aNameIm,aBIn,aBOut));
}

template <class Type>  std::shared_ptr<cGaussianPyramid<Type>>  
      cGaussianPyramid<Type>::PyramDiff() 
{
    cGP_Params aParam = mParams;
    // Require one overlap less because  of diff
    aParam.mNbOverlap--;
    // Not Sure thi would create a problem, but it would not be very coherent ?
    MMVII_INTERNAL_ASSERT_strong(aParam.mNbOverlap>=0,"No overlap for PyramDiff");

    tSP_Pyr aRes(new tPyr(aParam,this,eTyPyrTieP::eTPTP_LaplG,mNameIm,mBoxIn,mBoxOut));

    for (int aKo=0 ; aKo<int(mVOcts.size()) ; aKo++)
    {
        aRes->mVOcts.at(aKo)->MakeDiff(*mVOcts.at(aKo));
    }

    return aRes;
}


template <class Type>  std::shared_ptr<cGaussianPyramid<Type>>
      cGaussianPyramid<Type>::PyramCorner() 
{
    cGP_Params aParam = mParams;
    aParam.mConvolIm0 = mParams.mConvolC0;
    aParam.mEstimSigmInitIm0 = mSigmIm0;

    //   mEstimSigmInitIm0  (aParam.mEstimSigmInitIm0),
    //   mSigmIm0           (SomSigm(mEstimSigmInitIm0,mParams.mConvolIm0))

    // Standard has created one overlap that we dont need
    aParam.mNbOverlap--;
    // Not Sure thi would create a problem, but it would not be very coherent ?
    MMVII_INTERNAL_ASSERT_strong(aParam.mNbOverlap>=0,"No overlap for PyramDiff");

    tSP_Pyr aRes(new tPyr(aParam,this,eTyPyrTieP::eTPTP_Corner,mNameIm,mBoxIn,mBoxOut));

    ImTop().DIm().DupIn(aRes->ImTop().DIm());
    aRes->ComputGaussianFilter();
    for (const auto & aSPIm : aRes->mVAllIms)
    {
        aSPIm->MakeCorner();
    }

    return aRes;
}

template <class Type>  std::shared_ptr<cGaussianPyramid<Type>>
      cGaussianPyramid<Type>::PyramOrigNormalize() 
{
    cGP_Params aParam = mParams;
    // Standard has created one overlap that we dont need
    aParam.mNbOverlap--;
    // Not Sure thi would create a problem, but it would not be very coherent ?
    MMVII_INTERNAL_ASSERT_strong(aParam.mNbOverlap>=0,"No overlap for PyramDiff");

    tSP_Pyr aRes(new tPyr(aParam,this,eTyPyrTieP::eTPTP_OriNorm,mNameIm,mBoxIn,mBoxOut));

    for (int aKo=0 ; aKo<int(mVOcts.size()) ; aKo++)
    {
        aRes->mVOcts.at(aKo)->MakeOrigNorm(*mVOcts.at(aKo));
    }

    return aRes;
}



          //  - * - * - * - * - * - * - * - * - * - *

template <class Type> void cGaussianPyramid<Type>::Show() const
{
   StdOut() << "============ Gaussian Pyramid ==============\n";
   StdOut() << "     type elem: " <<  E2Str(tElemNumTrait<Type>:: TyNum())  << "\n";
   for (const auto & aPtrOct : mVOcts)
       aPtrOct->Show();
}

template <class Type> void cGaussianPyramid<Type>::ComputGaussianFilter()
{
   for (const auto & aPtrOct : mVOcts)
       aPtrOct->ComputGaussianFilter();
}

template <class Type> cPt2dr cGaussianPyramid<Type>::Pyr2File(const cPt2dr & aP) const
{
    return ToR(mBoxIn.P0()) + aP;
}

template <class Type> cPt2dr cGaussianPyramid<Type>::File2Pyr(const cPt2dr & aP) const
{
    return aP- ToR(mBoxIn.P0()) ;
}

template <class Type> void ScaleAndAdd
                           (
                                cInterf_ExportAimeTiep<Type> * aIExp,
                                const std::vector<cPt2di> & aLoc,
                                cGP_OneImage<Type> * aPtrI
                           )
{
    cGP_OneImage<Type> * aBestI= aPtrI->BestEquiv();
    int aIRatio = round_ni(aBestI->ScaleInO() / aPtrI->ScaleInO());
    
    // StdOut()  << "NULL= " << (aBestI==nullptr) << " BI=" << (aBestI ==aPtrI) << " R=" << aRatio << "\n";
    cAutoTimerSegm aATS("CreatePCar");  // timing
    for (const auto & aPtImInit : aLoc)
    {
       // cProtoAimeTieP<Type>   aPATPCom(aPtrI,aPtImInit );
       cProtoAimeTieP<Type>   aPATPCom(aBestI,aPtImInit * aIRatio,aIRatio!=1);
       aIExp->AddAimeTieP(aPATPCom);
    }
}

template <class Type> void cGaussianPyramid<Type>::SaveInFile (int aPowSPr,bool ForInspect) const
{
   bool DoPrint = (aPowSPr>=0) && ForInspect;
   bool DoExport = (mTypePyr!= eTyPyrTieP::eTPTP_Init);

   std::unique_ptr<cInterf_ExportAimeTiep<Type>> aPtrExpMin(cInterf_ExportAimeTiep<Type>::Alloc(SzIm0(),false ,mTypePyr,E2Str(mTypePyr),ForInspect,mParams));
   std::unique_ptr<cInterf_ExportAimeTiep<Type>> aPtrExpMax(cInterf_ExportAimeTiep<Type>::Alloc(SzIm0(),true ,mTypePyr,E2Str(mTypePyr),ForInspect,mParams));

   if (DoPrint)
      StdOut() <<  "\n ######  STAT FOR " << E2Str(mTypePyr)  << " Pow " << aPowSPr << " ######\n";

   // std::vector<cPt2dr> aVGlobMin;
   // std::vector<cPt2dr> aVGlobMax;
   int aNbMinTot = 0;
   int aNbMaxTot = 0;

   for (auto & aPtrIm : mVAllIms)
   {
       if (ForInspect)
          aPtrIm->SaveInFile();
       if (DoExport)
       {
           if (DoPrint && aPtrIm->IsTopOct())
              StdOut() <<  " ===================================================\n";
           cIm2D<Type> aI = aPtrIm->ImG();
           double aML =  MoyAbs(aI);
           double aS =  aPtrIm->ScaleInO();

           double aRadiusMM = 3.0; // Radius for Min/Max
           int aNbE = 0;
           if (DoPrint)
           {
              cResultExtremum aResE;
              {
                 cAutoTimerSegm aATS("1Extremum");
                 ExtractExtremum1(aI.DIm(),aResE,aRadiusMM);
              }
              aNbE = aResE.mPtsMin.size() + aResE.mPtsMax.size();

              StdOut()  <<  " Scale " <<  FixDigToStr(aS  ,2,2)
                       << " M- " << FixDigToStr(aML * pow(aS,aPowSPr-1),3,8 )
                       << " M= " << FixDigToStr(aML * pow(aS,aPowSPr  ),3,8 )
                       << " M+ " << FixDigToStr(aML * pow(aS,aPowSPr+1),3,8 )
                       << " E= " << FixDigToStr(aNbE * pow(aPtrIm->ScaleAbs(),1),6,2) ;
           }
           if (aPtrIm->Up() && aPtrIm->Down())
           {
               cResultExtremum aResE3;
               {
                  cAutoTimerSegm aATS("3Extremum");
                  ExtractExtremum3
                  (
                      aPtrIm->Up()->ImG().DIm(),
                      aPtrIm->ImG().DIm(),
                      aPtrIm->Down()->ImG().DIm(),
                      aResE3,
                      aRadiusMM
                  );
               }
               int aNbE3 = aResE3.mPtsMin.size() + aResE3.mPtsMax.size();
               if (DoPrint)
                  StdOut()  << " PROP=" << aNbE3 << " " << aNbE3 / double(aNbE) ;
               
               // tOct * aOctH = mPyrOrig->OctHom(aPtrIm->Oct());
               // std::string aNameMaster = aOctH->GPImTop()->NameSave();

               ScaleAndAdd(aPtrExpMin.get(),aResE3.mPtsMin,aPtrIm.get());
               ScaleAndAdd(aPtrExpMax.get(),aResE3.mPtsMax,aPtrIm.get());

               aNbMinTot += aResE3.mPtsMin.size();
               aNbMaxTot += aResE3.mPtsMax.size();
           }
        
           if (DoPrint)
              StdOut()  << "\n";
       }
   }
   if (!DoExport)
      return;
   StdOut() << " ======  NbTot , Min " << aNbMinTot << " Max " << aNbMaxTot << "\n";

   
   // std::string aPref =     mParams.mPrefixSave + "-AimePCar-";
   // std::string aPost =   E2Str(TypePyr()) + "-" + std::string("Tiiiiiiiiiiiiiiiiiiiiil") +".dmp";

   aPtrExpMin->FiltrageSpatialPts();
   aPtrExpMax->FiltrageSpatialPts();

   // StdOut() <<  " ppppPPppp " <<  mParams.mAppli->NamePCar(mNameIm,eModeOutPCar::eMNO_PCarV1,TypePyr(),false,true,cPt2di(0,0)) << "\n";


   aPtrExpMin->Export(mNameIm,ForInspect);
   aPtrExpMax->Export(mNameIm,ForInspect);
}

template <class Type>  cGP_OneOctave<Type> * cGaussianPyramid<Type>::OctHom(tOct *anOct)
{
   tOct * aRes = mVOcts.at(anOct->NumInPyr()).get();

   MMVII_INTERNAL_ASSERT_strong(RelativeDifference(aRes->Scale0Abs(),anOct->Scale0Abs())<1e-5,"OctHom");

   return aRes;
}

template <class Type>  cGP_OneImage<Type> * cGaussianPyramid<Type>::ImHom(tGPIm *anIm)
{
    tOct *  anOct = OctHom(anIm->Oct());
    tGPIm * aRes = anOct->ImageOfScaleAbs(anIm->ScaleAbs(),1e-5);
    MMVII_INTERNAL_ASSERT_strong(aRes!=nullptr,"ImHom");

    return aRes;
}

template <class Type>  cGP_OneImage<Type> * cGaussianPyramid<Type>::ImHomOri(tGPIm *anIm)
{
   return mPyrOrig->ImHom(anIm);
}

template <class Type> cIm2D<Type> cGaussianPyramid<Type>::ImTop() const {return mVOcts.at(0)->ImTop();}
template <class Type> cGP_OneImage<Type>* cGaussianPyramid<Type>::GPImTop() const {return mVOcts.at(0)->GPImTop();}

template <class Type> const cGP_Params & cGaussianPyramid<Type>::Params() const {return mParams;}
template <class Type> const double & cGaussianPyramid<Type>::MulScale() const {return mMulScale;}
template <class Type> int  cGaussianPyramid<Type>::NbImByOct() const {return  mParams.mNbLevByOct+mParams.mNbOverlap;}
template <class Type> const cPt2di & cGaussianPyramid<Type>::SzIm0() const {return  mParams.mSzIm0;}
template <class Type> const double & cGaussianPyramid<Type>::Scale0() const {return  mScale0;}
template <class Type> const double & cGaussianPyramid<Type>::SigmIm0() const {return mSigmIm0;}

template <class Type> const  std::vector<std::shared_ptr<cGP_OneOctave<Type>>> & cGaussianPyramid<Type>::VOcts() const {return mVOcts;}
template <class Type> const  std::vector<std::shared_ptr<cGP_OneImage<Type>>> & cGaussianPyramid<Type>::VAllIms() const {return mVAllIms;}

template <class Type> const std::vector<cGP_OneImage<Type>*> &   cGaussianPyramid<Type>::VMajIm() const {return mVMajIm;}


template <class Type> const std::string & cGaussianPyramid<Type>::NameIm() const {return mNameIm;}
// template <class Type> const cPt2di & cGaussianPyramid<Type>::NumTile() const {return mNumTile;}
template <class Type> eTyPyrTieP cGaussianPyramid<Type>::TypePyr() const {return mTypePyr;}

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
