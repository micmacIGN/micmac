#include "include/MMVII_all.h"
#include "LearnDM.h"
//#include "include/MMVII_Tpl_Images.h"

namespace MMVII
{


/* ************************************** */
/*                                        */
/*     cComputecVecCaracMatch             */
/*                                        */
/* ************************************** */

class cComputecVecCaracMatch
{
    public :
        cComputecVecCaracMatch
        (
                cVecCaracMatch&,
                float ScaleRho,float aGrayLev1,float aGrayLev2,
                const cAimePCar &,const cAimePCar &
        );
        tREAL4  StdDev(const cMatIner2Var<tREAL4> &) const;
    private :
        cVecCaracMatch  &    mVCM;
        const cAimePCar &    mAPC1;
        cIm2D<tU_INT1>       mIm1;
        cDataIm2D<tU_INT1>&  mDI1;
        const cAimePCar &    mAPC2;
        cIm2D<tU_INT1>       mIm2;
        cDataIm2D<tU_INT1>&  mDI2;
        cPt2di               mSz;
        int                  mNbTeta;
        int                  mNbRho;
        cIm1D<tREAL4>        mSomCQ;
        cDataIm1D<tREAL4>&   mDSomCQ;  // Som of CensuQ in a direction, weighted by rho
        cIm1D<tREAL4>        mConvSomCQ;
        cDataIm1D<tREAL4>&   mConvDSomCQ; // Convolution of mDSomCQ by hat 
        tREAL4               mGrL1;
        tREAL4               mGrL2;
};

tREAL4  cComputecVecCaracMatch::StdDev(const cMatIner2Var<tREAL4> & aM) const
{
   return std::min
          (
             aM.StdDev1()*mGrL1,
             aM.StdDev2()*mGrL2
          );
}

cComputecVecCaracMatch::cComputecVecCaracMatch
(
    cVecCaracMatch&  aVCM,
    float            aScaleRho,
    float            aGrayLev1,
    float            aGrayLev2,
    const cAimePCar & aAPC1,
    const cAimePCar & aAPC2
) :
    mVCM    (aVCM),
    mAPC1   (aAPC1),
    mIm1    (mAPC1.Desc().ILP()),
    mDI1    (mIm1.DIm()),
    mAPC2   (aAPC2),
    mIm2    (mAPC2.Desc().ILP()),
    mDI2    (mIm2.DIm()),
    mSz     (mDI1.Sz()),
    mNbTeta (mSz.x()),
    mNbRho  (mSz.y()),
    mSomCQ  (mNbTeta,nullptr,eModeInitImage::eMIA_Null),
    mDSomCQ (mSomCQ.DIm()),
    mConvSomCQ (mNbTeta,nullptr,eModeInitImage::eMIA_Null),
    mConvDSomCQ  (mConvSomCQ.DIm()),
    mGrL1        (aGrayLev1),
    mGrL2        (aGrayLev2)
{
    int aNbSample=0;
    // For standard measures
    tREAL4 aTotCQ=0.0;
    tREAL4 aTotCens=0.0;
    cMatIner2Var<tREAL4>  aTotMatI;
    aTotMatI.Add(1.0,128,128);

    // For Min/Max measures
    tREAL4  aBestCor = -1e20;
    tREAL4  aWorstCor = 1e20;
    tREAL4  aBestCQ =  -1e20;
    tREAL4  aWorstCQ = 1e20;

    // For weighted measures
    double aWeightRho = 1.0;
    double aSomWRho   = 0.0;
    cMatIner2Var<tREAL4>  aWMatI;
    aWMatI.Add(1.0,128,128);
    tREAL4 aWCQ = 0.0;
    tREAL4 aWCens = 0.0;

    for (int aKRho=0 ; aKRho<mNbRho ; aKRho++)
    {
        aWeightRho /= aScaleRho;
        aNbSample += mNbTeta;
        int aRhoCQ=0.0;
        int aRhoCens =0.0;
        cMatIner2Var<tREAL4>  aRhoMatI;

        for (int aKTeta=0 ; aKTeta<mNbTeta ; aKTeta++)
        {
             cPt2di aP(aKTeta,aKRho);
             int aV1 = mDI1.GetV(aP);
             int aV2 = mDI2.GetV(aP);
             tREAL4 aAbsDif = std::abs(aV1-aV2)/128.0;
             aRhoCQ += aAbsDif;
             aRhoCens += ((aV1>128) != (aV2>128));  // 128 correspond to 1.0, i.e values equal central values
             aRhoMatI.Add(1.0,aV1,aV2);
             mDSomCQ.AddV(aKTeta,aAbsDif*aWeightRho);
        }
        aTotCQ    += aRhoCQ;
        aTotCens  += aRhoCens ;
        aTotMatI.Add(aRhoMatI);
        
        tREAL4 anAvCQ = aTotCQ/aNbSample;
        tREAL4 anAvCens = aTotCens/aNbSample;
        tREAL4 aCorrel = 1-aTotMatI.Correl(1e-10);
        tREAL4 aStdDev = StdDev(aTotMatI);

        UpdateMin(aWorstCQ,anAvCQ);
        UpdateMax(aBestCQ,anAvCQ);
        UpdateMax(aBestCor,aCorrel);
        UpdateMin(aWorstCor,aCorrel);

        switch (aKRho) 
        {
           case 1 :
               mVCM.SetValue(eModeCaracMatch::eMCM_CQ1,anAvCQ);
               mVCM.SetValue(eModeCaracMatch::eMCM_Cen1,anAvCens);
               mVCM.SetValue(eModeCaracMatch::eMCM_Cor1,aCorrel);
               mVCM.SetValue(eModeCaracMatch::eMCM_MinStdDev1,aStdDev);
           break;

           case 2 :
               mVCM.SetValue(eModeCaracMatch::eMCM_CQ2,anAvCQ);
               mVCM.SetValue(eModeCaracMatch::eMCM_Cen2,anAvCens);
               mVCM.SetValue(eModeCaracMatch::eMCM_Cor2,aCorrel);
               mVCM.SetValue(eModeCaracMatch::eMCM_MinStdDev2,aStdDev);

               mVCM.SetValue(eModeCaracMatch::eMCM_BestCorrel2,aBestCor);
               mVCM.SetValue(eModeCaracMatch::eMCM_WorstCorrel2,aWorstCor);
               mVCM.SetValue(eModeCaracMatch::eMCM_WorstCQ2,aWorstCQ);
               mVCM.SetValue(eModeCaracMatch::eMCM_BestCQ2,aBestCQ);
           break;

           case 3 :
               mVCM.SetValue(eModeCaracMatch::eMCM_CQ3,anAvCQ);
               mVCM.SetValue(eModeCaracMatch::eMCM_Cen3,anAvCens);
               mVCM.SetValue(eModeCaracMatch::eMCM_Cor3,aCorrel);
               mVCM.SetValue(eModeCaracMatch::eMCM_MinStdDev3,aStdDev);

               mVCM.SetValue(eModeCaracMatch::eMCM_BestCorrel3,aBestCor);
               mVCM.SetValue(eModeCaracMatch::eMCM_WorstCorrel3,aWorstCor);
               mVCM.SetValue(eModeCaracMatch::eMCM_WorstCQ3,aWorstCQ);
               mVCM.SetValue(eModeCaracMatch::eMCM_BestCQ3,aBestCQ);
           break;
           case 4 :
               mVCM.SetValue(eModeCaracMatch::eMCM_CQ4,anAvCQ);
               mVCM.SetValue(eModeCaracMatch::eMCM_Cen4,anAvCens);
               mVCM.SetValue(eModeCaracMatch::eMCM_Cor4,aCorrel);
               mVCM.SetValue(eModeCaracMatch::eMCM_MinStdDev4,aStdDev);

               mVCM.SetValue(eModeCaracMatch::eMCM_BestCorrel4,aBestCor);
               mVCM.SetValue(eModeCaracMatch::eMCM_WorstCQ4,aWorstCQ);
           break;
           case 5 :
               mVCM.SetValue(eModeCaracMatch::eMCM_BestCorrel5,aBestCor);
               mVCM.SetValue(eModeCaracMatch::eMCM_WorstCQ5,aWorstCQ);
           break;
        }
        aWMatI.Add(aRhoMatI,aWeightRho);;
        aWCQ +=  aWeightRho* aRhoCQ;
        aWCens +=  aWeightRho * aRhoCens;
        aSomWRho   += aWeightRho;
    }
    mVCM.SetValue(eModeCaracMatch::eMCM_CQA,aTotCQ/aNbSample);
    mVCM.SetValue(eModeCaracMatch::eMCM_CenA,aTotCens/aNbSample);
    mVCM.SetValue(eModeCaracMatch::eMCM_CorA,1-aTotMatI.Correl(1e-10));

    aWCQ /= aSomWRho;
    aWCens /= aSomWRho;
    mVCM.SetValue(eModeCaracMatch::eMCM_CQW,aWCQ);
    mVCM.SetValue(eModeCaracMatch::eMCM_CenW,aWCens);
    mVCM.SetValue(eModeCaracMatch::eMCM_CorW,1-aWMatI.Correl(1e-10));

    // -------------- Compute  corners ----------------
          // Convolution
    int aIntTeta = (mNbTeta+1)/2;
    cWhitchMin<int,tREAL4> aWMin(0,1e10);
    for (int aKTeta1=0 ; aKTeta1<mNbTeta ; aKTeta1++)
    {
         tREAL4 aSom = 0;
         for (int aDTeta = -aIntTeta+1 ; aDTeta<aIntTeta ; aDTeta++)
         {
              aSom += mDSomCQ.CircGetV(aKTeta1+aDTeta) * (aIntTeta-std::abs(aDTeta));

         }
         mConvDSomCQ.SetV(aKTeta1,aSom); 
         aWMin.Add(aKTeta1,aSom);
    }
          // Extract minima sub pixellar
    int aITetaMin = aWMin.Index();
    double aDTetaMin =  StableInterpoleExtr
                        (
                            mConvDSomCQ.CircGetV(aITetaMin-1),
                            mConvDSomCQ.CircGetV(aITetaMin  ),
                            mConvDSomCQ.CircGetV(aITetaMin+1)
                           );
    double aRTetaMin = mod_real(aITetaMin+aDTetaMin,mNbTeta);
          // compute weighted criteria
    cWeightAv<tREAL4> aAv90;
    cWeightAv<tREAL4> aAv180;
    for (int aKTeta=0 ; aKTeta<mNbTeta ; aKTeta++)
    {
         tREAL4 aDifAng = (aKTeta-aRTetaMin) * ((2*M_PI)/mNbTeta);
         tREAL4 aWeight = (1+cos(aDifAng))/2.0;
         tREAL4 aVal    =  mDSomCQ.GetV(aKTeta);
         aAv180.Add(aWeight,aVal);
         aAv90.Add(Square(aWeight),aVal);
    }
    mVCM.SetValue(eModeCaracMatch::eMCM_CornW90,aAv90.Average()/aSomWRho);
    mVCM.SetValue(eModeCaracMatch::eMCM_CornW180,aAv180.Average()/aSomWRho);
          // Extract minima sub pixellar
    
    mVCM.SetValue(eModeCaracMatch::eMCM_DifGray,std::abs(aGrayLev1-aGrayLev2));
    mVCM.SetValue(eModeCaracMatch::eMCM_MinGray,std::min(aGrayLev1,aGrayLev2));

    mVCM.SetValue(eModeCaracMatch::eMCM_MinStdDevW,StdDev(aWMatI));

// StdOut() << "WWW " << aSomWRho << " " << int(eModeCaracMatch::eNbVals) <<  "\n";
}

/* ************************************** */
/*                                        */
/*         cVecCaracMatch                 */
/*                                        */
/* ************************************** */

const float & cVecCaracMatch::Value(eModeCaracMatch aCarac) const 
{
   const float & aVal =  AT_VECT(mVecCarac,int(aCarac));
   MMVII_INTERNAL_ASSERT_tiny(aVal!=UnDefVal,"Uninit val in cVecCaracMatch");
   return aVal;
}

cVecCaracMatch::cVecCaracMatch() :
    mVecCarac(int(eModeCaracMatch::eNbVals),UnDefVal)
{
}

cVecCaracMatch::cVecCaracMatch
(    
     float aScaleRho,float aGrayLev1,float aGrayLev2,
     const cAimePCar & aAPC1,const cAimePCar & aAPC2
)  :
   cVecCaracMatch()
{
   cComputecVecCaracMatch(*this,aScaleRho,aGrayLev1,aGrayLev2,aAPC1,aAPC2);
}




};
