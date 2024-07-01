#include "MMVII_Interpolators.h"
#include "MMVII_Mappings.h"
#include "MMVII_HeuristikOpt.h"
#include "MMVII_ExtractLines.h"


namespace MMVII
{

/* ************************************************************************ */
/*                                                                          */
/*                          cScoreTetaLine                                  */
/*                                                                          */
/* ************************************************************************ */

cScoreTetaLine::cScoreTetaLine(cDataIm2D<tREAL4> & aDIm,const cDiffInterpolator1D & anInt,tREAL8 aLength,tREAL8 aStep) :
        mDIm        (&aDIm),
        mLengthInit (aLength),
	mStepAInit  (aStep),
        mStepTeta   (-1),
        mTabInt     (anInt,100)
{
	SetLengthCur(mLengthInit);
}

void cScoreTetaLine::SetLengthCur(tREAL8 aL)
{
     mLengthCur = aL;
     mNb = round_up(mLengthCur/mStepAInit);
     mStepAbsc =  mLengthCur / mNb;
}

void cScoreTetaLine::SetCenter(const cPt2dr & aC) { mC = aC; }

/**   For an "oriented" segemet S :
 *       - C  the center, Teta the direction, L the lengnt
 *    We return how the segement is one of the direction of the checkboard pattern. Let
 *       - T the tangent and N the normal
 *       - A be the absicsse of a point  -L < A < L
 *       - P =  C + A T  a point  of 
 *       - let G be the  gradient of image in P, G' = G/|G| the direction of G
 *    The score depends of two "signs" s1 and s2   (s1,s2 = "+-1")  :
 *       - s1 depends if we have black ofr white a the left of S (which is oriented)
 *       - s2 depends if A>0 or A<0 (because the left colour change when cross C)
 *    So the score is finally :
 *
 *        | G' - s1s2 N| 
 *
 */

tREAL8   cScoreTetaLine::ScoreOfTeta(const tREAL8 & aTeta, tREAL8 aAbscMin,tREAL8 aSign) const
{
    cPt2dr aTgt = FromPolar(1.0,aTeta);
    cPt2dr aNormal = aTgt * cPt2dr(0,1.0);
    cWeightAv<tREAL8,tREAL8> aWA; // Average of difference

    for (int aKL=-mNb ; aKL<=mNb ; aKL++)  // samples all the point on the line
    {
         if (aKL)
         {
             tREAL8 aAbsc = mStepAbsc*aKL;
	     if ( std::abs(aAbsc)>aAbscMin )
	     {
                cPt2dr aPt = mC + aTgt * aAbsc;
                auto [aVal,aGrad] = mDIm->GetValueAndGradInterpol(mTabInt,aPt);

                tREAL8 aN2 = Norm2(aGrad);
                if (aN2 > 0)
                {
                  // Orientation depend of Sign & position in the segment
                   aGrad = aGrad / (aN2 * aSign * (aKL>0 ? 1 : -1));
                   tREAL8 aDist = Norm2(aNormal-aGrad);
                   aWA.Add(aN2,aDist);
                }
	     }
         }
    }

    return aWA.Average(1e10);  // 1e10 => def value in case no point
}

cPt1dr  cScoreTetaLine::Value(const cPt1dr& aPtAngle) const
{
    return cPt1dr(ScoreOfTeta(aPtAngle.x(),1.0,mCurSign));
}

tREAL8  cScoreTetaLine::Score2Teta(const std::pair<tREAL8,tREAL8> & aTeta, tREAL8 aAbscMin) const
{
     tREAL8 aScTeta0 = ScoreOfTeta(aTeta.first ,aAbscMin,-1);
     tREAL8 aScTeta1 = ScoreOfTeta(aTeta.second,aAbscMin,1);

     return std::max(aScTeta0,aScTeta1);
}



/**  Get initial guess of direction, simply parse all possible value at given step */

tREAL8  cScoreTetaLine::GetTetasInit(tREAL8 aStepPix,int aCurSign)
{
      mCurSign = aCurSign;
      cWhichMin<tREAL8,tREAL8> aWMin; // minimal score

      mStepTeta = aStepPix / mLengthCur; // Step in radian to have given step in pixel
      int aNbTeta = round_up(M_PI / mStepTeta);
      mStepTeta  = M_PI / aNbTeta;

      for (int aKTeta=0 ; aKTeta<aNbTeta ; aKTeta ++) // samples all teta
      {
           tREAL8 aTeta = aKTeta * mStepTeta;
           tREAL8 aVal = Value(cPt1dr(aTeta)).x();
           aWMin.Add(aTeta,aVal);  //  update 
				   //
      }

      return aWMin.IndexExtre() ;
}



/**  Refine the existing value of mimimal using "cOptimByStep", suppose we are close enouh */
tREAL8  cScoreTetaLine::Refine(tREAL8 aTeta0,tREAL8 aStepPix,int aSign)
{
     mCurSign = aSign;
     cOptimByStep<1> aOpt(*this,true,10.0);  // 10.0 => DistMin
     auto [aVal,aRes] = aOpt.Optim(cPt1dr(aTeta0),mStepTeta,aStepPix/mLengthCur);

     return aRes.x();
}

std::pair<tREAL8,tREAL8> cScoreTetaLine::Tetas_CheckBoard(const cPt2dr & aC,tREAL8 aStepInit,tREAL8 aStepLim)
{
    mStepTetaInit = aStepInit;
    mStepTetaLim  = aStepLim;
    SetLengthCur(mLengthInit);

    tREAL8 aVTeta[2];
    SetCenter(aC);

    for (const auto & aSign : {-1,1})
    {
        tREAL8  aTetaInit =  GetTetasInit(aStepInit,aSign);
        tREAL8  aTetaRefine = Refine(aTetaInit,aStepLim,aSign);
        aVTeta[(1+aSign)/2] = aTetaRefine;
    }
    NormalizeTeta(aVTeta);

    return std::pair<tREAL8,tREAL8>(aVTeta[0],aVTeta[1]);
}

extern bool DebugCB;

tREAL8 cScoreTetaLine::Prolongate(tREAL8 aLMax,std::pair<tREAL8,tREAL8> & aPairTeta,bool ReestimateTeta)
{
     std::vector<tREAL8>  aVTeta{aPairTeta.first,aPairTeta.second};
     std::vector<cPt2dr>  aVTgt;
     std::vector<cPt2dr>  aVNormBlack;
      std::vector<tREAL8> aVCurOrd;

     std::vector<tREAL8> aVBl;
     std::vector<tREAL8> aVWh;

     // [1]  Compute, for the existing "small" segment , the value of bl & white

     for (size_t aKTeta=0 ; aKTeta<2 ; aKTeta++)
     {
          for (const auto & aSign : {-1,1})
	  {
              cPt2dr aTgt = FromPolar(1.0,aVTeta.at(aKTeta))*double(aSign);
              aVTgt.push_back(aTgt);
	      cPt2dr aNormBlack = aTgt * cPt2dr(0, (aKTeta==0)?1:-1 );
	      aVNormBlack.push_back(aNormBlack);
              aVCurOrd.push_back(0.0);

	      int aNb = round_up(mLengthInit) ; // (2/pixel) + lenght/2 
	      for (int aKA=0 ; aKA<= aNb ; aKA++)
	      {
                   tREAL8 aAbsc= mLengthInit * (0.5 + aKA/(2.0*aNb)) ;
		   cPt2dr aPt = mC + aTgt *aAbsc;

		   tREAL8 aBl = mDIm->GetVBL(aPt + aNormBlack*(1.0));
		   tREAL8 aWh = mDIm->GetVBL(aPt - aNormBlack*(1.0));
                   aVBl.push_back(aBl);
                   aVWh.push_back(aWh);
	      }
	  }
     }
     tREAL8 aBl = NonConstMediane(aVBl);
     tREAL8 aWh = NonConstMediane(aVWh);

     tREAL8 aWBound = 0.25; //  0.5 => bound=avg,  0 = initial bound
     tREAL8 aMaxBl =  (1-aWBound) * aBl + aWBound * aWh ;
     tREAL8 aMinWh =  aWBound* aBl + (1-aWBound) * aWh ;

     bool aGoOn = true;
     tREAL8 aMargin = 2.0;
     tREAL8 aAbsc=mLengthCur ;

     tREAL8 aAngleFaisc = LineAngles(FromPolar(1.0,aVTeta.at(0)), FromPolar(1.0,aVTeta.at(1)));
     aAngleFaisc = std::min(0.1,aAngleFaisc/4.0);

     
     cWeightAv<tREAL8> aAvgD;
     while (aGoOn)
     {
           for (int aK=0 ; aK<4 ; aK++)
           {
                cPt2dr aTgt = aVTgt[aK];
		cPt2dr aNormBlack = aVNormBlack[aK];
                cPt2dr aPt = mC + aTgt *aAbsc + aNormBlack * aVCurOrd[aK];

		cWhichMax<tREAL8,tREAL8> aWMax;
		for (tREAL8 aKOrd =-1 ; aKOrd<=1 ; aKOrd++)
		{
	            tREAL8 aDeltaOrd = aKOrd * 0.2;
		    cPt2dr aNewPt = aPt + aNormBlack * aDeltaOrd;
                    auto [aVal,aGrad] = mDIm->GetValueAndGradInterpol(mTabInt,aNewPt);
		    aWMax.Add(aDeltaOrd,Norm2(aGrad));
		}
		aVCurOrd[aK] += aWMax.IndexExtre();
                aPt = mC + aTgt *aAbsc + aNormBlack * aVCurOrd[aK];

		// tREAL8 aOrdRad =  std::max(1.0, aAngleFaisc * aAbsc);
		tREAL8 aOrdRad =  1.0 + aAngleFaisc * aAbsc;

                tREAL8 aBl = mDIm->GetVBL(aPt + aNormBlack*(aOrdRad));
                tREAL8 aWh = mDIm->GetVBL(aPt - aNormBlack*(aOrdRad));
		if ((aBl>aMaxBl) || (aWh<aMinWh)) 
		{
                   if (DebugCB)
		   {
                       StdOut() << "Bound "  <<  aMaxBl << " " << aMinWh  << " VALS=" << aBl << " " << aWh << "\n"  ;
		   }
                   aGoOn = false;
		}

                auto [aVal,aGrad] = mDIm->GetValueAndGradInterpol(mTabInt,aPt);
		tREAL8 aN2 = Norm2(aGrad);
		if (aN2==0)
                   aGoOn = false;
		else
		{
                    aGrad = aGrad/aN2;
		    if (Norm2(aGrad+aNormBlack) > 0.5)
		    {
                       if (DebugCB)
                       {
                          StdOut() << "GRRRADD " << aGrad << " " << aNormBlack << " N2=" << Norm2(aGrad+aNormBlack) << "\n";
                       }
                       aGoOn = false;
		    }

                    aAvgD.Add(1.0,Norm2(aGrad+aNormBlack));
		}
           }

           if (aAbsc > aLMax+aMargin) aGoOn = false;
           if (aGoOn) 
              aAbsc += 0.5;
     }
     aAbsc = std::max(mLengthCur,aAbsc-aMargin);


     if (0) StdOut() << " -------------- ABSC= " << aAbsc << " AVGD=" << aAvgD.Average()  << "\n";

     return aAbsc;
}


void  cScoreTetaLine::NormalizeTeta(t2Teta & aVTeta)
{
   //  the angles are define % pi,  t2ddo have a unique representation
   for (int aK=0 ; aK<2 ; aK++)
        aVTeta[aK] = mod_real(aVTeta[aK],M_PI);

   //  we need that teta1/teta2  correpond to a direct repair
   if (aVTeta[0]> aVTeta[1])
       aVTeta[1] += M_PI;

}

const tREAL8 &  cScoreTetaLine::LengthCur() const {return mLengthCur;}
cDataIm2D<tREAL4> * cScoreTetaLine::DIm() const {return mDIm;}


};
