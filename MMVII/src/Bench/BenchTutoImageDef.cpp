#include "include/MMVII_all.h"
#include "include/MMVII_TplSymbImage.h"


namespace MMVII
{

class  cGaussF2D
{
     public :
           double  Val(const cPt2dr & aP) const;
           cGaussF2D(const cPt2dr & aC,double aSigm) ;
           cGaussF2D(const cGaussF2D & ,const  cHomot2D<tREAL8> & );
     private :
           cPt2dr  mC;
           double  mSigm;
};


cGaussF2D::cGaussF2D(const cPt2dr & aC,double aSigm) :
    mC    (aC),
    mSigm (aSigm)
{
}
double  cGaussF2D::Val(const cPt2dr & aP) const
{
   double aN2 = SqN2(aP-mC) / (2*Square(mSigm));
   return  exp(-aN2) ;
}

cGaussF2D::cGaussF2D(const cGaussF2D & aG ,const  cHomot2D<tREAL8> &  aH) : 
    cGaussF2D
    (
           (aG.mC -  aH.Tr()) / aH.Sc() ,
           aG.mSigm/aH.Sc()
    )
{
}

class cTestDeformIm
{
     public :
         cTestDeformIm();
     private :
         double  Modele(const cPt2dr & ) const;
         int                mSzGlob;
         cPt2di             mSzIm;
         cIm2D<tREAL8>      mIm;
         cDataIm2D<tREAL8>& mDIm;
         cPt2dr             mCenterIm;
         double             mSigmaIm;

         double             mAmplRad;
         double             mTrRad;
         double             mScRad;

         cHomot2D<tREAL8>   mGT_I2Mod;
         cHomot2D<tREAL8>   mGT_Mod2Im;

         cGaussF2D          mGaussIm;
         cGaussF2D          mGaussModel;

         std::vector<cPt2dr> mVPtsMod;
         std::vector<double> mValueMod;
};


cTestDeformIm::cTestDeformIm() :
   mSzGlob     ( 300),
   mSzIm       ( mSzGlob,mSzGlob), 
   mIm         ( mSzIm),
   mDIm        ( mIm.DIm()),
   mCenterIm   ( ToR(mSzIm)/2.0),
   mSigmaIm    (mSzGlob/10.0),
   mAmplRad    (255.0),
   mTrRad      (mAmplRad/20.0),
   mScRad      (0.7),
   mGT_I2Mod   (cPt2dr(10.0,15.0),2.0),
   mGT_Mod2Im  (mGT_I2Mod.MapInverse()),
   mGaussIm    (mCenterIm,mSigmaIm),
   mGaussModel (mGaussIm,mGT_Mod2Im)
{
    cGaussF2D aGIm(mCenterIm,mSigmaIm);
    cGaussF2D aGMod(aGIm,mGT_Mod2Im);
    
    for (const auto & aPixIm :  mDIm)
    {
         cPt2dr aPMod = mGT_I2Mod.Value(ToR(aPixIm));

         mVPtsMod.push_back(aPMod);
         mValueMod.push_back(mGaussModel.Val(aPMod)*mAmplRad);

         mDIm.SetV(aPixIm, mTrRad + mScRad*mValueMod.back());

      // little check on gaussian composed with homotethy
         double aV1 = mGaussModel.Val(aPMod);
         double aV2 = mGaussIm.Val(mGT_Mod2Im.Value(aPMod));
         MMVII_INTERNAL_ASSERT_bench(std::abs(aV1-aV2)<1e-6,"Gauss compos");
    }
    mDIm.ToFile("GaussDef.tif");
}

void BenchDeformIm(cParamExeBench & aParam)
{
    if (! aParam.NewBench("DeformIm")) return;

    cTestDeformIm aTDI;
    FakeUseIt(aTDI);

    aParam.EndBench();
}


};   // namespace MMVII
