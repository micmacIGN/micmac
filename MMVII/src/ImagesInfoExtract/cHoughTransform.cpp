#include "MMVII_Tpl_Images.h"
#include "MMVII_PCSens.h"
#include "MMVII_ImageInfoExtract.h"
#include "MMVII_ExtractLines.h"


namespace MMVII
{

/* ************************************************************************ */
/*                                                                          */
/*                       cHoughTransform                                    */
/*                                                                          */
/* ************************************************************************ */


cHoughTransform::cHoughTransform
(
     const cPt2dr  & aSzIn,
     const cPt2dr &  aMulTetaRho,
     const tREAL8 & aSigmTeta,
     cPerspCamIntrCalib * aCalib
) :
    mMiddle    (aSzIn / 2.0),
    mRhoMax    (Norm2(mMiddle)),
    mMulTeta   (aMulTetaRho.x()),
    mMulRho    (aMulTetaRho.y()),
    mSigmTeta  (aSigmTeta),
    mCalib     (aCalib),
    mNbTeta    (round_up(2*M_PI*mMulTeta * mRhoMax)),
    mFactI2T   ((2.0*M_PI)/mNbTeta),
    mNbRho     (2+round_up(2*mMulRho * mRhoMax)),
    mTabSin    (mNbTeta),
    mDTabSin   (mTabSin.DIm()),
    mTabCos    (mNbTeta),
    mDTabCos   (mTabCos.DIm()),
    mAccum     (cPt2di(mNbTeta,mNbRho),nullptr,eModeInitImage::eMIA_Null),
    mDAccum    (mAccum.DIm())
{

     // Just to avoid a Clang-Warning because "mCalib" is private and not used
     if (mCalib==nullptr) { delete mCalib; }

     //  Tabulate  "cos" and "sin"
     for (int aKTeta=0 ; aKTeta<mNbTeta ; aKTeta++)
     {
          mDTabSin.SetV(aKTeta,std::sin(Ind2Teta(aKTeta)));
          mDTabCos.SetV(aKTeta,std::cos(Ind2Teta(aKTeta)));
     }
}

cIm2D<tREAL4>      cHoughTransform::Accum() const {return mAccum;}
const tREAL8 & cHoughTransform::RhoMax() const {return mRhoMax;}

tREAL8  cHoughTransform::AvgS2() const
{
   return std::sqrt(DotProduct(mDAccum,mDAccum) / mDAccum.NbElem());
}

cHoughPS* cHoughTransform::PtToLine(const cPt3dr & aPt) const
{
   // (x,y) in L  <=> x cos(T) + y Sin(T) = R
   tREAL8  aTeta = RInd2Teta(aPt.x());
   tREAL8  aRho  = RInd2Rho(aPt.y());
   cPt2dr  aTgt(-sin(aTeta),cos(aTeta));
   cPt2dr  aP0 = mMiddle + cPt2dr(aRho*cos(aTeta),aRho*sin(aTeta));

   return new cHoughPS(this,cPt2dr(aTeta,aRho),aPt.z(),aP0-aTgt,aP0+aTgt);
}


void  cHoughTransform::AccumulatePtAndDir(const cPt2dr & aPt,tREAL8 aTetaC,tREAL8 aWeight)
{
      //  compute the index-interval, centered on aTetaC, of size mSigmTeta
      int  iTeta0 = round_down(Teta2RInd(aTetaC-mSigmTeta));
      int  iTeta1 = round_up(Teta2RInd(aTetaC+mSigmTeta));

      // Angle are defined %2PI, make interval > 0
      if (iTeta0<0)
      {
           iTeta0 += mNbTeta;
           iTeta1 += mNbTeta;
           aTetaC += 2 * M_PI;
      }
      //  Polar coordinate Rho-Teta, are defined arround point mMiddle
      tREAL8 aX = aPt.x() - mMiddle.x();
      tREAL8 aY = aPt.y() - mMiddle.y();

      for (int iTeta=iTeta0 ;  iTeta<=iTeta1 ; iTeta++)
      {
           tREAL8 aTeta = Ind2Teta(iTeta);
	   //  Compute a weighting function : 1 at aTetaC, 0 if dist > mSigmTeta 
           tREAL8 aWTot =    aWeight * ( 1 -    std::abs(aTeta-aTetaC) /mSigmTeta);
           if (aWTot>0) // (weitgh can be <0 because of rounding)
           {
               int aITetaOK = iTeta%mNbTeta;  // % 2PI => put indexe in interval
               //   equation :  (x,y) in L  <=> x cos(T) + y Sin(T) = R
               tREAL8 aRho =   aX * mDTabCos.GetV(aITetaOK) + aY*mDTabSin.GetV(aITetaOK) ;
               tREAL8  aRIndRho = Rho2RInd(aRho); // convert Real rho in its index inside accumulator
	       // Make bi-linear repartition of contribution as "aRIndRho" is  real
               int  aIRho0  = round_down(aRIndRho);
               int  aIRho1  = aIRho0 +1;
               tREAL8 aW0 = (aIRho1-aRIndRho);

	       //  Finally accumulate
               mDAccum.AddVal(cPt2di(aITetaOK,aIRho0),   aW0 *aWTot);
               mDAccum.AddVal(cPt2di(aITetaOK,aIRho1),(1-aW0)*aWTot);
           }
      }
}

tREAL8  cHoughTransform::GetValueBlob(cPt2di aPt,int aMaxNeigh) const
{
    tREAL8 aRes =  mDAccum.GetV(aPt);
    for (const int aSign : {-1,1})
    {
        int aNb =0;
	tREAL8 aLastV = mDAccum.GetV(aPt);
	cPt2di aPNext = aPt + cPt2di(0,aSign);

	while (mDAccum.Inside(aPNext) && (aNb<aMaxNeigh))
	{
            tREAL8 aNextV = mDAccum.GetV(aPNext);
	    if (aNextV<aLastV)
	    {
                aRes += aNextV;
	        aPNext += cPt2di(0,aSign);
		aLastV = aNextV;
	        aNb++;
	    }
	    else
	        aNb = aMaxNeigh;
	}
    }

    return aRes;
}


std::vector<cPt3dr>  cHoughTransform::ExtractLocalMax(size_t aNbMax,tREAL8 aDist,tREAL8 aThrAvg,tREAL8 aThrMax) const
{
    // [0]  Make a duplication as we will modify the accum
    cIm2D<tREAL4> anAccum = mAccum.Dup();
    cDataIm2D<tREAL4>& aDAccum = anAccum.DIm();

    // [1]  Compute average , max and  threshold
    //    [1.A]   : max & avg
    tREAL8 aAvg = 0;
    tREAL8 aVMax = 0;
    for (const auto & aPix : aDAccum)
    {
         tREAL8 aVal = aDAccum.GetV(aPix);
	 aAvg += aVal;
	 UpdateMax(aVMax,aVal);
    }
    aAvg /= aDAccum.NbElem();

    //    [1.B]    Threshlod is the stricter of both
    tREAL8 aThrHold = std::max(aAvg*aThrAvg,aVMax*aThrMax);


    // [2] Set to 0 point < aThrHold (to limitate number of pre-sel & accelerate IKthVal
    for (const auto & aPix : aDAccum)
    {
        if (aDAccum.GetV(aPix)<aThrHold)
        {
            aDAccum.SetV(aPix,0.0);
	}
    }

    // [3]  Extract the local maxima
    cResultExtremum aExtr(false,true);
    ExtractExtremum1(aDAccum,aExtr,aDist);

    // [4] Refine the point and give a value to the max

    std::vector<cPt3dr> aRes;
    cAffineExtremum<tREAL4>  aAffin(mAccum.DIm(),1.5);
    for (const auto aPt : aExtr.mPtsMax)
    {
         cPt2dr aPAff = aAffin.OneIter(ToR(aPt));
	 if ( mDAccum.InsideBL(aPAff))
            aRes.push_back(cPt3dr(aPAff.x(),aPAff.y(),GetValueBlob(aPt,7)));
    }

    // [5] Sort with highest value first, then select NbMax
    SortOnCriteria(aRes,[](const auto & aP) {return -aP.z();});
    while (aRes.size() > aNbMax)  
          aRes.pop_back();


    return aRes;
}

};
