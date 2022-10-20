
#include "LearnDM.h"
#include "MMVII_2Include_Serial_Tpl.h"
#include "MMVII_ImageInfoExtract.h"


namespace MMVII
{

/* ************************************** */
/*                ::MMVII                 */
/* ************************************** */

std::string  NameVecCar(const tVecCar & aVC)
{
   std::string aRes="";
   for (const auto & aLab : aVC)
   {
       std::string aStrLab = E2Str(aLab);
       if (aRes!="") aRes = aRes +"_";
       aRes = aRes + aStrLab;
   }
   return aRes;
}


/* ************************************** */
/*                                        */
/*     cComputecVecCaracMatch             */
/*                                        */
/* ************************************** */

class cComputecVecCaracMatch
{
    public :
        typedef cDataIm2D<tREAL4>   tDataIm;

        cComputecVecCaracMatch
        (
                cVecCaracMatch&, float ScaleRho,
                const tDataIm & aImInit1,const tDataIm & aImInit2,
                const tDataIm & aImNorm1,const tDataIm & aImNorm2,
                const cAimePCar &,const cAimePCar &
        );
        tREAL4  Min2StdDev(const cMatIner2Var<tREAL4> &) const;
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
        cPt2dr               mPt1;
        cPt2dr               mPt2;
        tREAL4               mGrNorm1;
        tREAL4               mGrNorm2;
};



tREAL4  cComputecVecCaracMatch::Min2StdDev(const cMatIner2Var<tREAL4> & aM) const
{
   return std::min
          (
             aM.StdDev1()*mGrNorm1,
             aM.StdDev2()*mGrNorm2
          );
}

tREAL4 AjdustStdCost(const tREAL4 aCost)
{
     return   std::max(tREAL4(0.0),std::min(tREAL4(1.0),aCost));
}

tREAL4 MakeStdCostOfCorrel(tREAL4 aCorrel)
{
   return AjdustStdCost((1-aCorrel)/2.0);
}

tREAL4 MakeStdCostOfCorrelNotC(tREAL4 aCorrel)
{
   double aCost = (1-aCorrel)/2.0;
   aCost = pow(std::max(0.0,aCost),1/2.0);
   return AjdustStdCost(aCost*10.0);
}

cComputecVecCaracMatch::cComputecVecCaracMatch
(
    cVecCaracMatch&  aVCM,
    float            aScaleRho,
    const tDataIm & aImInit1,const tDataIm & aImInit2,
    const tDataIm & aImNorm1,const tDataIm & aImNorm2,
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
    mPt1         (aAPC1.PtIm()),
    mPt2         (aAPC2.PtIm()),
    mGrNorm1     (aImNorm1.GetVBL(mPt1)),
    mGrNorm2     (aImNorm2.GetVBL(mPt2))
{
static int aCpt=0;
aCpt++;
bool BUG = 0&&(aCpt==476382);
{
if (BUG) StdOut() << "================== " << aCpt  << "\n";
aImNorm1.GetVBL(aAPC1.PtIm());
if (BUG) StdOut() << "LIINNNEEE " << __LINE__ << "\n";
aImNorm2.GetVBL(aAPC2.PtIm());
if (BUG) StdOut() << "LIINNNEEE " << __LINE__ << "\n";
}
/*
*/

// MULTI SCALE
{
    int aNbSample=0;
    // For standard measures
    tREAL4 aTotCQ=0.0;
    tREAL4 aTotCens=0.0;
    cMatIner2Var<tREAL4>  aTotMatI;
    aTotMatI.Add(1.0,1.0,1.0);  // Central Value

    // For Min/Max measures
    tREAL4  aBestCor = -1e20;
    tREAL4  aWorstCor = 1e20;
    tREAL4  aBestCQ =  -1e20;
    tREAL4  aWorstCQ = 1e20;

    // For weighted measures
    double aWeightRho = 1.0;
    double aSomWRho   = 0.0;
    cMatIner2Var<tREAL4>  aWMatI;
    aWMatI.Add(1.0,1.0,1.0);  // Central Value
    tREAL4 aWCQ = 0.0;
    tREAL4 aWCens = 0.0;

    for (int aKRho=0 ; aKRho<mNbRho ; aKRho++)
    {
        aWeightRho /= aScaleRho;
        aNbSample += mNbTeta;
        tREAL4 aRhoCQ= 0;
        int aRhoCens = 0;
        cMatIner2Var<tREAL4>  aRhoMatI;

        for (int aKTeta=0 ; aKTeta<mNbTeta ; aKTeta++)
        {
             cPt2di aP(aKTeta,aKRho);
             int aV1 = mDI1.GetV(aP);
             int aV2 = mDI2.GetV(aP);
             tREAL4 aAbsDif = std::abs(aV1-aV2)/256.0;
             aRhoCQ += aAbsDif;
             aRhoCens += ((aV1>128) != (aV2>128));  // 128 correspond to 1.0, i.e values equal central values
             aRhoMatI.Add(1.0,aV1/128.0,aV2/128.0);
             mDSomCQ.AddV(aKTeta,aAbsDif*aWeightRho);
        }
        aTotCQ    += aRhoCQ;
        aTotCens  += aRhoCens ;
        aTotMatI.Add(aRhoMatI);
        
        tREAL4 anAvCQ = aTotCQ/aNbSample;
        tREAL4 anAvCens = aTotCens/aNbSample;
        tREAL4 aCorrel = MakeStdCostOfCorrel(aTotMatI.Correl(1e-10));
        tREAL4 aStdDev = Min2StdDev(aTotMatI);

        UpdateMin(aWorstCQ,anAvCQ);
        UpdateMax(aBestCQ,anAvCQ);
        UpdateMax(aBestCor,aCorrel);
        UpdateMin(aWorstCor,aCorrel);

        switch (aKRho) 
        {
           case 1 :
               mVCM.SetValue(eModeCaracMatch::eMS_CQ1,anAvCQ);
               mVCM.SetValue(eModeCaracMatch::eMS_Cen1,anAvCens);
               mVCM.SetValue(eModeCaracMatch::eMS_Cor1,aCorrel);
               mVCM.SetValue(eModeCaracMatch::eMS_MinStdDev1,aStdDev);
           break;

           case 2 :
               mVCM.SetValue(eModeCaracMatch::eMS_CQ2,anAvCQ);
               mVCM.SetValue(eModeCaracMatch::eMS_Cen2,anAvCens);
               mVCM.SetValue(eModeCaracMatch::eMS_Cor2,aCorrel);
               mVCM.SetValue(eModeCaracMatch::eMS_MinStdDev2,aStdDev);

               mVCM.SetValue(eModeCaracMatch::eMS_BestCorrel2,aBestCor);
               mVCM.SetValue(eModeCaracMatch::eMS_WorstCorrel2,aWorstCor);
               mVCM.SetValue(eModeCaracMatch::eMS_WorstCQ2,aWorstCQ);
               mVCM.SetValue(eModeCaracMatch::eMS_BestCQ2,aBestCQ);
           break;

           case 3 :
               mVCM.SetValue(eModeCaracMatch::eMS_CQ3,anAvCQ);
               mVCM.SetValue(eModeCaracMatch::eMS_Cen3,anAvCens);
               mVCM.SetValue(eModeCaracMatch::eMS_Cor3,aCorrel);
               mVCM.SetValue(eModeCaracMatch::eMS_MinStdDev3,aStdDev);

               mVCM.SetValue(eModeCaracMatch::eMS_BestCorrel3,aBestCor);
               mVCM.SetValue(eModeCaracMatch::eMS_WorstCorrel3,aWorstCor);
               mVCM.SetValue(eModeCaracMatch::eMS_WorstCQ3,aWorstCQ);
               mVCM.SetValue(eModeCaracMatch::eMS_BestCQ3,aBestCQ);
           break;
           case 4 :
               mVCM.SetValue(eModeCaracMatch::eMS_CQ4,anAvCQ);
               mVCM.SetValue(eModeCaracMatch::eMS_Cen4,anAvCens);
               mVCM.SetValue(eModeCaracMatch::eMS_Cor4,aCorrel);
               mVCM.SetValue(eModeCaracMatch::eMS_MinStdDev4,aStdDev);

               mVCM.SetValue(eModeCaracMatch::eMS_BestCorrel4,aBestCor);
               mVCM.SetValue(eModeCaracMatch::eMS_WorstCQ4,aWorstCQ);
           break;
           case 5 :
               mVCM.SetValue(eModeCaracMatch::eMS_BestCorrel5,aBestCor);
               mVCM.SetValue(eModeCaracMatch::eMS_WorstCQ5,aWorstCQ);
           break;
        }
        aWMatI.Add(aRhoMatI,aWeightRho);;
        aWCQ +=  aWeightRho   * (aRhoCQ/mNbTeta);
        aWCens +=  aWeightRho * (aRhoCens/double(mNbTeta));
        aSomWRho   += aWeightRho;
    }
    mVCM.SetValue(eModeCaracMatch::eMS_CQA,aTotCQ/aNbSample);
    mVCM.SetValue(eModeCaracMatch::eMS_CenA,aTotCens/aNbSample);
    mVCM.SetValue(eModeCaracMatch::eMS_CorA,MakeStdCostOfCorrel(aTotMatI.Correl(1e-10)));


    aWCQ /= aSomWRho;
    aWCens /= aSomWRho;
    mVCM.SetValue(eModeCaracMatch::eMS_CQW,aWCQ);
    mVCM.SetValue(eModeCaracMatch::eMS_CenW,aWCens);
    mVCM.SetValue(eModeCaracMatch::eMS_CorW,MakeStdCostOfCorrel(aWMatI.Correl(1e-10)));

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
    int aITetaMin = aWMin.IndexExtre();
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
    mVCM.SetValue(eModeCaracMatch::eMS_CornW90,aAv90.Average()/aSomWRho);
    mVCM.SetValue(eModeCaracMatch::eMS_CornW180,aAv180.Average()/aSomWRho);
          // Extract minima sub pixellar
    
    mVCM.SetValue(eModeCaracMatch::eNI_DifGray,std::abs(mGrNorm1-mGrNorm2));
    mVCM.SetValue(eModeCaracMatch::eNI_MinGray,std::min(mGrNorm1,mGrNorm2));

    mVCM.SetValue(eModeCaracMatch::eMS_MinStdDevW,Min2StdDev(aWMatI));
}


if (BUG)
{
TPT(mPt1);
TPT(mPt2);

    StdOut() << "LIINNNEEE " << __LINE__ << "\n";
    StdOut() << "PTS=" << mPt1  << mPt2 << "\n";
    StdOut() << "SZI=" << aImInit1.Sz() <<  aImInit2.Sz() << "\n";
    StdOut() << "SZN=" << aImNorm1.Sz() <<  aImNorm2.Sz() << "\n";
}
// Normalize + Init
{
    const int aDMax = 8;
    const std::vector<std::vector<cPt2di> > & aVVNeigh = TabGrowNeigh<2>(aDMax);
    cMatIner2Var<tREAL4>  aMatInit;
    tREAL4 aCostCor     = 0;
    tREAL4 aCostCorNotC = 0;
    tREAL4 aCensus = 0;
    tREAL4 aCensusQ = 0;
    tREAL4 aSomCensus = 0;
    tREAL4 aSomCensusQ = 0;
    tREAL4 aSomDiffInit = 0;
    tREAL4 aSomDiffNorm = 0;

    tREAL4 aVInitC1 = aImInit1.GetVBL(mPt1);
    tREAL4 aVInitC2 = aImInit2.GetVBL(mPt2);
    int aNbPts = 0;
    for (int aDist=0 ; aDist<= aDMax ; aDist++)
    {
         const std::vector<cPt2di> & aVNeigh  =aVVNeigh.at(aDist);
         aNbPts += aVNeigh.size();
         for (const auto & aPVoisI : aVNeigh)
         {
             cPt2dr aPVoisR = ToR(aPVoisI);
             cPt2dr aPV1 = mPt1 + aPVoisR;
             cPt2dr aPV2 = mPt2 + aPVoisR;
             tREAL4 aVInit1 = aImInit1.GetVBL(aPV1);
             tREAL4 aVInit2 = aImInit2.GetVBL(aPV2);

             aMatInit.Add(1.0,aVInit1,aVInit2);

             aSomDiffInit += std::abs(aVInit1-aVInit2);
             aSomCensus += ((aVInit1>aVInitC1) != ((aVInit2>aVInitC2) ));
             tREAL4 aRatio1 = NormalisedRatioPos(aVInit1,aVInitC1);
             tREAL4 aRatio2 = NormalisedRatioPos(aVInit2,aVInitC2);
             aSomCensusQ += std::abs(aRatio1-aRatio2);

             tREAL4 aVNorm1 = aImNorm1.GetVBL(aPV1);
             tREAL4 aVNorm2 = aImNorm2.GetVBL(aPV2);
             aSomDiffNorm += std::abs(aVNorm1-aVNorm2);

// StdOut() << "VNNNNN " << aVNorm1 << " " << aVNorm2 << "\n";
    // tREAL4 aVNorm2 = aImInit1.GetVBL(mPt2);
// FakeUseIt(aPV1);
// FakeUseIt(aVC1);
// FakeUseIt(aPV2);
// FakeUseIt(aVC2);
         }
         tREAL4 aDiffInit = std::min(1.0,aSomDiffInit/(255.0*aNbPts));
         tREAL4 aDiffNorm = std::min(1.0,double(aSomDiffNorm/aNbPts));
         if ((aDist>0) && (aDist<=4))
         {
             aCostCor = MakeStdCostOfCorrel(aMatInit.Correl(1e-10));
             aCostCorNotC = MakeStdCostOfCorrelNotC(aMatInit.CorrelNotC(1e-10));
// StdOut() << "CORRR-NOTC " << aMatInit.Correl(1e-10) << " " << aMatInit.CorrelNotC(1e-10) << "\n";
         }
         if ((aDist>0) && (aDist%2))
         {
             aCensus = aSomCensus / aNbPts;
             aCensusQ = aSomCensusQ / (2.0*aNbPts);
         }
         //  ---------------------------------------------
         if (aDist==1) //4
         {
             mVCM.SetValue(eModeCaracMatch::eSTD_Cor1,aCostCor);
             mVCM.SetValue(eModeCaracMatch::eSTD_NCCor1,aCostCorNotC);
             mVCM.SetValue(eModeCaracMatch::eSTD_Diff1,aDiffInit);
             mVCM.SetValue(eModeCaracMatch::eNI_Diff1,aDiffNorm);
         }
         else if (aDist==2) // 4+ 6 = 10
         {
             mVCM.SetValue(eModeCaracMatch::eSTD_Cor2,aCostCor);
             mVCM.SetValue(eModeCaracMatch::eSTD_NCCor2,aCostCorNotC);
             mVCM.SetValue(eModeCaracMatch::eSTD_Diff2,aDiffInit);
             mVCM.SetValue(eModeCaracMatch::eNI_Diff2,aDiffNorm);

             mVCM.SetValue(eModeCaracMatch::eSTD_CQ2,aCensusQ);
             mVCM.SetValue(eModeCaracMatch::eSTD_Cen2,aCensus);
         }
         else if (aDist==3) // 10 + 4  = 14
         {
             mVCM.SetValue(eModeCaracMatch::eSTD_Cor3,aCostCor);
             mVCM.SetValue(eModeCaracMatch::eSTD_NCCor3,aCostCorNotC);
             mVCM.SetValue(eModeCaracMatch::eSTD_Diff3,aDiffInit);
             mVCM.SetValue(eModeCaracMatch::eNI_Diff3,aDiffNorm);
         }
         else if (aDist==4) // 14 + 4 = 18
         {
             mVCM.SetValue(eModeCaracMatch::eSTD_Cor4,aCostCor);
             mVCM.SetValue(eModeCaracMatch::eSTD_NCCor4,aCostCorNotC);

             mVCM.SetValue(eModeCaracMatch::eSTD_CQ4,aCensusQ);
             mVCM.SetValue(eModeCaracMatch::eSTD_Cen4,aCensus);
         }
         else if (aDist==5) // 18 + 2 = 20
         {
             mVCM.SetValue(eModeCaracMatch::eSTD_Diff5,aDiffInit);
             mVCM.SetValue(eModeCaracMatch::eNI_Diff5,aDiffNorm);
         }
         else if (aDist==6) // 20 +2 = 22
         {
             mVCM.SetValue(eModeCaracMatch::eSTD_CQ6,aCensusQ);
             mVCM.SetValue(eModeCaracMatch::eSTD_Cen6,aCensus);
         }
         else if (aDist==7) // 22 + 2 = 24
         {
             mVCM.SetValue(eModeCaracMatch::eSTD_Diff7,aDiffInit);
             mVCM.SetValue(eModeCaracMatch::eNI_Diff7,aDiffNorm);
         }
         else if (aDist==8) // 24 + 2 = 26
         {
             mVCM.SetValue(eModeCaracMatch::eSTD_CQ8,aCensusQ);
             mVCM.SetValue(eModeCaracMatch::eSTD_Cen8,aCensus);
         }

    }
}
    
// StdOut() << "WWW " << aSomWRho << " " << int(eModeCaracMatch::eNbVals) <<  "\n";
if (BUG) StdOut() << "LIINNNEEE " << __LINE__ << "\n";

}

/* ************************************** */
/*                                        */
/*         cVecCaracMatch                 */
/*                                        */
/* ************************************** */

const   cVecCaracMatch::tSaveValues &  cVecCaracMatch::Value(eModeCaracMatch aCarac) const 
{
   const tSaveValues & aVal =   mVecCarac[int(aCarac)];
   MMVII_INTERNAL_ASSERT_tiny(aVal!=TheUnDefVal,"Uninit val in cVecCaracMatch");
   return aVal;
}

void cVecCaracMatch::SetValue(eModeCaracMatch aCarac,const float & aVal) 
{
    bool Show=true;
    tREAL4 Eps = 1e-2;
    static tREAL4 aMinVal = 1e10;
    static tREAL4 aMaxVal = -1e10;

    if (aVal<aMinVal)
    {
       aMinVal = aVal;
       if (Show)   StdOut()  << "INTERVAL " << aMinVal << " " << aMaxVal << "\n";
    }
    if (aVal>aMaxVal)
    {
       aMaxVal = aVal;
       if (Show)   StdOut()  << "INTERVAL " << aMinVal << " " << aMaxVal << "\n";
    }
    if ((aVal<-Eps) || (aVal>1+Eps))
    {
       StdOut()  << "INTERVAL " << aMinVal << " " << aMaxVal  <<  " Type " <<  E2Str(aCarac) << "\n";
       MMVII_INTERNAL_ASSERT_always (false,"Value out interval [0,1]");
    }

    mVecCarac[int(aCarac)] = std::min(TheDyn4Save-1,round_down(TheDyn4Save * AjdustStdCost(aVal)));
}

cVecCaracMatch::cVecCaracMatch
(    
     float aScaleRho,
     const tDataIm & aImInit1,const tDataIm & aImInit2,
     const tDataIm & aImNorm1,const tDataIm & aImNorm2,
     const cAimePCar & aAPC1,const cAimePCar & aAPC2
)  :
   cVecCaracMatch()
{
   cComputecVecCaracMatch(*this,aScaleRho,aImInit1,aImInit2,aImNorm1,aImNorm2,aAPC1,aAPC2);
}

cVecCaracMatch::cVecCaracMatch
(
     const cPyr1ImLearnMatch  & aPyr1,const cPyr1ImLearnMatch  & aPyr2,
     const cAimePCar & aAPC1,const cAimePCar & aAPC2
)  :
    cVecCaracMatch
    (
         aPyr1.MulScale(),
         aPyr1.ImInit()     , aPyr2.ImInit(),
         aPyr1.ImFiltered() , aPyr2.ImFiltered(),
	 aAPC1,aAPC2
    )
{
}

/*
*/

cVecCaracMatch::cVecCaracMatch() 
{
    for (int aK=0 ; aK<TheNbVals; aK++)
       mVecCarac[aK] = TheUnDefVal;
}

void cVecCaracMatch::Show(tNameSelector aNameSel)
{
    for (int aK=0 ; aK<TheNbVals; aK++)
    {
        eModeCaracMatch aMode = eModeCaracMatch(aK);
        std::string aName =  E2Str(aMode);
        if (aNameSel.Match(aName))
        {
            StdOut() << "[" << aName << "] : " <<  mVecCarac[aK] /double(TheDyn4Save) << "\n";
        }
    }

}

void cVecCaracMatch::AddData(const cAuxAr2007 & anAux)
{
   cRawData4Serial aRDS = cRawData4Serial::Tpl(mVecCarac,TheNbVals);
   MMVII::AddData(cAuxAr2007("VCar",anAux),aRDS);
}

void AddData(const cAuxAr2007 & anAux, cVecCaracMatch &    aVCM) 
{
     aVCM.AddData(anAux);
}



void cVecCaracMatch::FillVect(cDenseVect<tINT4> & aVec,const tVecCar &  aVC) const
{
    for (int aK=0 ; aK<int(aVC.size()) ; aK++)
    {
        aVec(aK) = mVecCarac[(int)aVC[aK]];
        if (false  && DEBUG_LM)
        {
           StdOut() << " K="  << aK
		    <<  " LAB=" <<  E2Str(aVC[aK])
		    <<  " COST=" <<  aVec(aK) / double(TheDyn4Save)
		   << "\n";
        }
    }
}


/* ************************************** */
/*                                        */
/*      cFileVecCaracMatch                */
/*                                        */
/* ************************************** */


cFileVecCaracMatch::cFileVecCaracMatch(const cFilterPCar & aFPC,int aNb) :
   mNbVal  (cVecCaracMatch::TheNbVals),
   mFPC    (aFPC),
   mCheckRW ("XX")
{
   mVVCM.reserve(aNb);
}

cFileVecCaracMatch::cFileVecCaracMatch(const std::string & aNameFile) :
   mFPC(true),
   mCheckRW ("XX")
{
   ReadFromFile(*this,aNameFile);
   
   MMVII_INTERNAL_ASSERT_always( mNbVal == cVecCaracMatch::TheNbVals,"Changed Carac");
   MMVII_INTERNAL_ASSERT_always( mCheckRW == "CheckRW","Bad R/W for cFileVecCaracMatch");
}

const std::vector<cVecCaracMatch> & cFileVecCaracMatch::VVCM() const
{
   return mVVCM;
}



void cFileVecCaracMatch::AddCarac(const cVecCaracMatch & aVCM)
{
   mVVCM.push_back(aVCM);
}

void   cFileVecCaracMatch::AddData(const cAuxAr2007 & anAux)
{
    mCheckRW = "CheckRW";
    MMVII::AddData(cAuxAr2007("NbVal",anAux),mNbVal);
    MMVII::AddData(cAuxAr2007("FPC",anAux),mFPC);
    MMVII::AddData(cAuxAr2007("VCar",anAux), mVVCM);
    MMVII::AddData(cAuxAr2007("CheckRW",anAux), mCheckRW);
}

void AddData(const cAuxAr2007 & anAux, cFileVecCaracMatch &    aVCM)
{
   aVCM.AddData(anAux);
}



};
