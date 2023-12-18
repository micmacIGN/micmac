#include "FilterCodedTarget.h"
#include "CodedTarget.h"
#include "MMVII_2Include_Serial_Tpl.h"
#include "MMVII_Tpl_Images.h"

/**  \file cSaddleFiter.cpp
 
     \brief file contains implemention of filter for computing sadle point
     Much a proto for now :
          * contains code for functionnal filtre : begin with  matching (by leastsq) the vignette
	    on a basis of function
 */


namespace MMVII
{

template <const int Dim> std::vector< cPtxd<tREAL8,Dim> > ToR(const std::vector< cPtxd<int,Dim> > &aVPtI,double aStep)
{
    // std::transform(aVPtI.begin(),aVPtI.end(),aRes.begin(),[](auto aPtI){return ToR<int>(aPtI);});  => core dump ??!!

    std::vector< cPtxd<tREAL8,Dim> > aRes;
    for (const auto & aPtI : aVPtI) 
        aRes.push_back(ToR(aPtI)*aStep);
    return aRes;
}


/* ******************************************** */
/*                                              */
/*               cBasisFuncQuad                 */
/*                                              */
/* ******************************************** */

std::pair<tREAL8,std::vector<tREAL8>> cBasisFuncQuad::WeightAndVals(const cPt2dr & aPt) const 
{
    std::pair<tREAL8,std::vector<tREAL8>> aRes;

    aRes.first = 1.0;
    aRes.second.push_back(Square(aPt.x()));
    aRes.second.push_back(2.0*aPt.x()*aPt.y());
    aRes.second.push_back(Square(aPt.y()));
    aRes.second.push_back(aPt.x());
    aRes.second.push_back(aPt.y());
    aRes.second.push_back(1);
    
    return aRes;
}

/* ******************************************** */
/*                                              */
/*               cCompiledNeighBasisFunc        */
/*                                              */
/* ******************************************** */


cCompiledNeighBasisFunc::cCompiledNeighBasisFunc(const cBasisFunc & aBaseF ,const tVNeigh & aVNeigh) :
    mVNeigh   (aVNeigh),
    mNbNeigh  (mVNeigh.size()),
    mSys      (1)

{
     MMVII_INTERNAL_ASSERT_tiny(!mVNeigh.empty(),"Emty neigh cCompiledNeighBasisFunc");
     for (const auto & aNeigh : mVNeigh)
     {
        auto [aWeight,aVals] = aBaseF.WeightAndVals(aNeigh);
        mVWeight.push_back(aWeight);
        mVFuncs.push_back(aVals);

        mNbFunc = mVFuncs[0].Sz();
        MMVII_INTERNAL_ASSERT_tiny(aVals.size()==mNbFunc,"Variable size in cCompiledNeighBasisFunc");
     }

     mSys = cLeasSqtAA<tREAL8> (mNbFunc);

}

cDenseVect<tREAL8>   cCompiledNeighBasisFunc::SlowCalc(const std::vector<tREAL8> & aVV) 
{
    AssertValsIsOk(aVV);
    mSys.Reset();

    for (size_t aK=0 ; aK<mNbNeigh ; aK++)
    {
         mSys.PublicAddObservation(mVWeight[aK],mVFuncs[aK],aVV[aK]);
    }

    return mSys.Solve();
}

/* ******************************************** */
/*                                              */
/*               cCalcSaddle                    */
/*                                              */
/* ******************************************** */

cCalcSaddle::cCalcSaddle(double aRay,double aStep) :
	mRay       (aRay),
        mStep      (aStep),
	mVINeigh   (SortedVectOfRadius(-1,aRay/aStep)),
	mNbNeigh   (mVINeigh.size()),
	mRNeigh    (ToR(mVINeigh,aStep)),
	mCalcQuad  (cBasisFuncQuad(),mRNeigh),
	mVVals     (mNbNeigh)
{
}


tREAL8  cCalcSaddle::CalcSaddleCrit(const std::vector<tREAL8> & aVVals,bool Show)
{
      cDenseVect<tREAL8>   aVect = mCalcQuad.SlowCalc(aVVals);

      tREAL8 aXX = aVect(0);
      tREAL8 aXY = aVect(1);
      tREAL8 aYY = aVect(2);

      if (Show) StdOut() << " dxx=" << aXX << " dxy=" << aXY << " dyy=" << aYY << std::endl;

       //               (A -X  B)
       //det(M-XI) =    (B  C-X)     =  (X-A) (X-C) - B^2 = X^2 - (A+C) X  + (-B^2 +AC) = 0
       //                R =  1/2  ((A+C)  +- SQRT((A+C)^ 2 + 4B2 -4AC)) = 1/2  ((A+C)  +- SQRT((A-C)^ 2 + 4B2 ))
      tREAL8 aLapl = aXX+aYY;
      tREAL8 aDiscr=  std::sqrt(Square(aXX-aYY) + 4*Square(aXY));

      tREAL8 aL1 = (aLapl+aDiscr)/2.0;
      tREAL8 aL2 = (aLapl-aDiscr)/2.0;

      if (Show) StdOut() << " L1=" << aL1 << " L2=" << aL2 << std::endl;
      if (aL1>0)
      {
         aL2 = -aL2;
      }
      else
      {
	 aL2 =-aL2;
	 std::swap(aL1,aL2);
      }

      return aL2;
}


cPt2dr   cCalcSaddle::RefineSadlPtFromVals(const std::vector<tREAL8> & aVVals,bool Show)
{
      cDenseVect<tREAL8>   aVect = mCalcQuad.SlowCalc(aVVals);

      tREAL8 dXX = aVect(0);
      tREAL8 dXY = aVect(1);
      tREAL8 dYY = aVect(2);

      tREAL8 dX = aVect(3);
      tREAL8 dY = aVect(4);

      /*    (dXX  dXY) -1                           (dYY -dXY)  (dX)
       *    (dXY  dYY)    =  1/(dXX *dYY - dXY) ^2  (-dXY dXX)  (dY)
      */

      tREAL8 aDet = dXX * dYY -Square(dXY);

      if (aDet==0) 
         return cPt2dr(0,0);

      return  -cPt2dr(dYY*dX -dXY*dY,-dXY*dX +dXX*dY) / aDet;
}

void cCalcSaddle::RefineSadlePointFromIm(cIm2D<tREAL4> aIm,cDCT & aDCT)
{
     double aThrDiv = 3.0;
     cPt2dr aP0 = aDCT.mPt;

     cDataIm2D<tREAL4> & aDIm = aIm.DIm();
     for (int aK=0 ; aK<4 ; aK++)
     {
          for (size_t aKNeigh=0; aKNeigh<mNbNeigh; aKNeigh++)
          {
              mVVals[aKNeigh] = aDIm.GetVBL(aDCT.mPt+mRNeigh[aKNeigh]);
	  }
          cPt2dr   aDPt =  RefineSadlPtFromVals(mVVals,false);

	  aDCT.mPt += aDPt;
	  if ((Norm2(aDCT.mPt-aP0) > aThrDiv) || (aDIm.Interiority(ToI(aDCT.mPt))<20))
	  {
              if (aDCT.mGT)  
	      {
		      StdOut() << "DIVG  "  << aDCT.mGT->mC << " DPT " << aDPt << std::endl;
	      }
              aDCT.mState =  eResDCT::Divg;
	      return;
	  }
	  if (Norm2(aDPt) < 1e-2)
             return;
     }
}

/* ******************************************** */
/*                                              */
/*                   ::                         */
/*                                              */
/* ******************************************** */

/*   Modelistaion, for a perfect saddle point  , let V be the sadle value,
 *   for each neighoor Q1 (x,y)
 *      * let  Q2(-y,x)
 *      * let  Q3(-x,-y)
 *      * let  Q4(y,-x)
 *
 *    C1=>  Q1 and Q3 must have the same relative position (both upper of lower V)
 *    C2=>  Q2 and Q4 must have the same relative position (both upper of lower V)
 *    C3=>  (Q1,Q3) and (Q2,Q4) will have "in general" opposite  position (not always depend of total curvature)               
 *
 *     So for each pixel:
 *        -  we compute V as an average in the neighboor
 *        -  we test  how often C1,C2,C3 are satisfied
 *                               
 */



std::pair<cIm2D<tREAL4>,cIm2D<tREAL4>> FastComputeSaddleCriterion(cIm2D<tREAL4>  aIm,double aRay)
{
     cDataIm2D<tREAL4> & aDIm = aIm.DIm();
     cIm2D<tREAL4> aIDif(aDIm.Sz(),nullptr,eModeInitImage::eMIA_Null);
     cIm2D<tREAL4> aICpt(aDIm.Sz(),nullptr,eModeInitImage::eMIA_Null);

     std::vector<cPt2di>    aVNeigh =  SortedVectOfRadius(0,aRay);
     size_t aNbNeigh = aVNeigh.size();

     // each of 4 vector will store one quarter
     std::vector<cPt2di>    aVQuatN1;
     std::vector<cPt2di>    aVQuatN2;
     std::vector<cPt2di>    aVQuatN3;
     std::vector<cPt2di>    aVQuatN4;

     for (auto aPt : aVNeigh)
     {
        if ((aPt.y()>=0) && (aPt.x()>0))
	{
            aVQuatN1.push_back(aPt);
            aVQuatN2.push_back(aPt*cPt2di(0,1));
            aVQuatN3.push_back(aPt*-1);
            aVQuatN4.push_back(aPt*cPt2di(0,-1));
	}
     }
     size_t aNbQN = aVQuatN1.size();

     cRect2 aBox (  aIm.DIm().Dilate(-round_up(aRay)));
     for (const auto & aPix : aBox)
     {
          // compute average
          tREAL8 aAvg=0;
          for (size_t aKNeigh=0; aKNeigh<aNbNeigh; aKNeigh++)
          {
               aAvg += aDIm.GetV(aPix+aVNeigh[aKNeigh]);
          }
	  aAvg /=aNbNeigh;
          tREAL8 aSomDif=0;
          tREAL8 aNbDif=0;
	  // now parse quarter
	  for (size_t aKQ=0 ; aKQ<aNbQN ; aKQ++)
	  {
              tREAL4 aD1 = aDIm.GetV(aPix+aVQuatN1[aKQ])-aAvg;
              tREAL4 aD3 = aDIm.GetV(aPix+aVQuatN3[aKQ])-aAvg;
	      // if D1 and D3 complies with condition C1
	      if ((aD1>0)==(aD3>0))
	      {
                  tREAL4 aD2 = aDIm.GetV(aPix+aVQuatN2[aKQ])-aAvg;
	          // if D1 and D2 complies with condition C3
		  if ((aD2>0) != (aD1>0))
		  {
                        tREAL4 aD4 = aDIm.GetV(aPix+aVQuatN4[aKQ])-aAvg;
	                // if D2 and D4 complies with condition C2
			if ((aD2>0) == (aD4>0))
			{
                         // then our criteria is satisfied, the difference with V quantify how much
			 // the criteria was satisfied
                            aSomDif += std::min
				        (
					     std::min(std::abs(aD1),std::abs(aD3)),
					     std::min(std::abs(aD2),std::abs(aD4))
					);
			    aNbDif ++; // non quantitativ criteria, for absol thresholds
			}
		  }
	      }
	  }

	  aIDif.DIm().SetV(aPix,aSomDif/aNbQN);
	  aICpt.DIm().SetV(aPix,aNbDif/double(aNbQN));
     }

     return std::pair<cIm2D<tREAL4>,cIm2D<tREAL4>>(aIDif,aICpt);
}



};
