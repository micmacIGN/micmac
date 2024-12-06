#include "MMVII_Tpl_Images.h"
#include "MMVII_PCSens.h"
#include "MMVII_ImageInfoExtract.h"
#include "MMVII_ExtractLines.h"
#include "MMVII_Linear2DFiltering.h"

namespace MMVII
{

/* ************************************************************************ */
/*                                                                          */
/*                          cParalLine                                      */
/*                                                                          */
/* ************************************************************************ */

cParalLine::cParalLine(const cHoughPS & aH1,const cHoughPS & aH2) :
    mMidleSeg   (aH1.Seg()),
    mRankMatch  (-1),
    mRadHom     (1e30)
{
    mVHS.push_back(aH1);
    mVHS.push_back(aH2);

    cPt2dr aP0 =   (aH1.Seg().PMil() +aH2.Seg().PMil() ) / 2.0;
    cPt2dr aTgt =  (aH1.Seg().Tgt() - aH2.Seg().Tgt() ) / 2.0;

    mMidleSeg =  tSeg2dr(aP0-aTgt,aP0+aTgt);
    mScoreMatch = std::min(aH1.Cumul(),aH2.Cumul());

    mAngleDif  = aH1.DistAnglAntiPar(aH2) * aH1.HT()->RhoMax() ;
    mWidth     = -( aH1.DY(aH2) + aH2.DY(aH1) ) / 2.0;
}


cOneLineAntiParal cParalLine::GetLAP(const cPerspCamIntrCalib & aCalib) const
{
   cOneLineAntiParal  aOLAP;

   aOLAP.mSeg  = aCalib.ExtenSegUndistIncluded(true,mMidleSeg);
   aOLAP.mAngDif = mAngleDif;
   aOLAP.mRadHom = mRadHom;
   aOLAP.mWidth  = mWidth;
   aOLAP.mCumul  = mScoreMatch;
   aOLAP.mSigmaLine = (mVHS.at(0).SigmaL() + mVHS.at(1).SigmaL()) / 2.0;
   
   return aOLAP;
}



const tREAL8& cParalLine::ScoreMatch() const {return mScoreMatch;}

size_t cParalLine::RankMatch() const  {return (size_t) mRankMatch;}
void cParalLine::SetRankMatch(size_t aRank)  
{ 
    mRankMatch=aRank;
}
const std::vector<cHoughPS> &   cParalLine::VHS() const {return mVHS;}


tREAL8 cParalLine::DistGt(const tSeg2dr & aSeg) const
{
	return std::max(mMidleSeg.Dist(aSeg.P1()),mMidleSeg.Dist(aSeg.P2()));
}


bool cParalLine::RejectByComparison(const cParalLine & aBetterOne) const
{
    if (aBetterOne.mScoreMatch<=mScoreMatch)
	return false;

    if (aBetterOne.mScoreMatch > (2.0 *mScoreMatch))
       return true;

    if (aBetterOne.mRadHom < (mRadHom/2.0))
       return true;

    if ((aBetterOne.mRadHom/aBetterOne.mScoreMatch ) < ((mRadHom/mScoreMatch)/ 2.0)  )
       return true;

     return false;
}



template <class Type> cIm1D<Type>  MedianFilter(const cDataIm1D<Type> & aDImIn,int aSz)
{
     int aNbX=aDImIn.Sz();

     cIm1D<Type> aRes(aNbX);
     for (int aKX = 0 ; aKX<aNbX ; aKX++)
     {
         int aKX0 = std::max(0,aKX-aSz);
         int aKX1 = std::min(aNbX-1,aKX+aSz);

	 std::vector<tREAL8>  aVN;
	 for (int aKN=aKX0 ; aKN<=aKX1 ; aKN++)
             aVN.push_back(aDImIn.GetV(aKN));

         aRes.DIm().SetV(aKX,NonConstMediane(aVN));
     }

    return aRes;
}


tREAL8 LeasSqModelRadiom(const cDataIm1D<tREAL8> &  anIm, int aDeg)
{
   int aNbX = anIm.Sz();
   cLeasSqtAA<tREAL8> aSys(aDeg+1);
   for (int aKX=0 ; aKX<aNbX ; aKX++)
   {
        tREAL8 aXN= (aKX-aNbX) / double(aNbX);
        tREAL8 aPowX=1.0;
        cDenseVect<tREAL8> aV(1+aDeg);
        for (int aD=0 ; aD<= aDeg; aD++)
        {
            aV(aD) = aPowX;
	    aPowX *= aXN;
        }
	aSys.PublicAddObservation(1.0,aV,anIm.GetV(aKX));
   }

   tREAL8 aSomRes = 0.0;
   cDenseVect<tREAL8> aSol = aSys.PublicSolve();
   for (int aKX=0 ; aKX<aNbX ; aKX++)
   {
        tREAL8 aXN= (aKX-aNbX) / double(aNbX);
        tREAL8 aPowX=1.0;
	tREAL8 aVCalc=0.0;
        for (int aD=0 ; aD<= aDeg; aD++)
        {
             aVCalc += aSol(aD) * aPowX;
	    aPowX *= aXN;
        }
	aSomRes += Square(aVCalc-anIm.GetV(aKX));
   }

   return  std::sqrt(aSomRes/aNbX);

}

void ShowImProfile(const cDataIm1D<tREAL8> &  anIm,const std::string & aName)
{
    tREAL8 aMin,aMax;
   
    int aNbX= anIm.Sz();
    GetBounds(aMin,aMax,anIm);
    cIm2D<tU_INT1> aRes(cPt2di(aNbX,256));

    for (int aX=0 ; aX<aNbX ; aX++)
    {
         tREAL8 aTrh = (anIm.GetV(aX) /aMax) * 255;
         for (int aY=0 ; aY<256; aY++)
         {
             aRes.DIm().SetV(cPt2di(aX,aY), ((255-aY) > aTrh ) * 255);
         }
    }
    aRes.DIm().ToFile(aName);
}

void  cParalLine::ComputeRadiomHomog(const cDataGenUnTypedIm<2> & anIm,cPerspCamIntrCalib * aCalib,const std::string & aNameFile) 
{
// StdOut() << "aNameFileaNameFileaNameFileaNameFile " << aNameFile << "\n";

    cSegment2DCompiled<tREAL8>  aSegFull = aCalib->ExtenSegUndistIncluded(false,mMidleSeg,0.05,1.0,5.0);

    // [1]   Compute a 1D  image of radiometry along line
    tREAL8 aN2 = aSegFull.N2();
    int aNbX = round_ni(aN2);
    cIm1D<tREAL8>  aIm1(aNbX+1);
    tREAL8 aSomV = 0;
    for (int aKX=0 ; aKX<=aNbX ; aKX++)
    {
         tREAL8 aXLoc = aN2 * (aKX/tREAL8(aNbX));
         cPt2dr aPt = aCalib->Redist(aSegFull.FromCoordLoc(cPt2dr(aXLoc,0.0)));
	 if (!anIm.InsideBL(aPt))
	 {
// StdOut() << "xxxxxxxxxxxx   aNameFileaNameFileaNameFileaNameFile " << aNameFile << "\n";
             mRadHom = 1e4;
             return;
	 }

	 tREAL8 aValue = anIm.GetVBL(aPt);

	 aIm1.DIm().SetV(aKX,aValue);
	aSomV += aValue;
    }
    aSomV   /= aNbX;

    // [2]  make median filter to remove high frequency noise
    cIm1D<tREAL8>  aImMed = MedianFilter(aIm1.DIm(),3);


    // [3]  compute a linear smoothing 
    cIm1D<tREAL8> aImLineFilt = aImMed.Dup();
    ExpFilterOfStdDev(aImLineFilt.DIm(),3,20.0);

    // [4]   compute  dif^2 between median and its smoothing
    tREAL8 aSom2  = 0.0;
    for (int aKX=0 ; aKX<=aNbX ; aKX++)
    {
         tREAL8 aDif = aImMed.DIm().GetV(aKX)-aImLineFilt.DIm().GetV(aKX);
	 aSom2 += Square(aDif);
    }
    aSom2 = std::sqrt(aSom2/aNbX);


    // [5]  eventualuy generate profile
    if ( aNameFile != "")
    {
        std::string aDir = DirOfPath(aNameFile,false);
        std::string aName = FileOfPath(aNameFile,false);
        ShowImProfile(aImMed.DIm(), aDir + "ProfilMed_" +  aName + "_" + ToStr(mRankMatch) + ".tif");
    }
    
    mRadHom = aSom2/aSomV;

    StdOut() 
	    << " FiltG2=" <<  aSom2/aSomV 
	    << " Cumul=" << ScoreMatch() << "\n";
}

/* ************************************************************************ */
/*                                                                          */
/*                          cHoughPS                                        */
/*                                                                          */
/* ************************************************************************ */

cHoughPS::cHoughPS(const cHoughTransform * aHT,const cPt2dr & aTR,tREAL8 aCumul,const cPt2dr & aP1,const cPt2dr & aP2) :
    mHT          (aHT),
    mTetaRho     (aTR),
    mCumul       (aCumul),
    mSigmaL      (-1),     // undef
    mSegE        (aP1,aP2),
    mCode        (eCodeHPS::Ok)
{
}

void cHoughPS::UpdateSegImage(const tSeg & aNewSeg,tREAL8 aNewCumul,tREAL8 aSigma)
{
    mCumul = aNewCumul;
    mSigmaL = aSigma;
    mSegE = aNewSeg;
    mTetaRho = mHT->Line2PtInit(aNewSeg);
}


tREAL8 cHoughPS::DistAnglAntiPar(const cHoughPS& aPS2) const
{
     return diff_circ(Teta()+M_PI,aPS2.Teta(),2*M_PI);
}


tREAL8 cHoughPS::DY(const cHoughPS & aHPS) const
{
   return mSegE.ToCoordLoc(aHPS.mSegE.PMil()) .y() ;
}

const cSegment2DCompiled<tREAL8> & cHoughPS::Seg() const {return mSegE;}
const cPt2dr & cHoughPS::TetaRho() const {return mTetaRho;}
const tREAL8 & cHoughPS::Teta()    const {return mTetaRho.x();}
const tREAL8 & cHoughPS::Rho()     const {return mTetaRho.y();}
const tREAL8 & cHoughPS::Cumul()     const {return mCumul;}
const tREAL8 & cHoughPS::SigmaL()     const {return mSigmaL;}
// cHoughPS * cHoughPS::Matched() const {return mMatched;}
eCodeHPS  cHoughPS::Code() const  {return mCode;}
void cHoughPS::SetCode(eCodeHPS aCode) {  mCode = aCode;}
const cHoughTransform *  cHoughPS::HT() const {return mHT;}


// const cHoughPS::tSeg&  cHoughPS::MidlSeg() const {return mMidleSeg;}


cPt2dr  cHoughPS::IndTetaRho() const
{
        return cPt2dr(mHT->Teta2RInd(Teta()),mHT->Rho2RInd(Rho()));
}

tREAL8 cHoughPS::Dist(const cHoughPS& aPS2,const tREAL8 &aFactTeta) const
{
     return   std::abs(Rho()-aPS2.Rho())
            + DistAnglAntiPar(aPS2) * (aFactTeta * mHT->RhoMax())
     ;
}

void cHoughPS::Test(const cHoughPS &  aHPS) const
{
        StdOut() << "LARG " << mSegE.ToCoordLoc(aHPS.mSegE.PMil()) .y()
                 << " RMDistAng " << DistAnglAntiPar(aHPS) * mHT->RhoMax()
                 << " DistAng " << DistAnglAntiPar(aHPS) 
                 // << " T1=" << Teta()<< " T2=" aHPS.Teta()  << " "<< diff_circ(Teta(),aHPS.Teta(),2*M_PI)
                 << "\n";
}

bool cHoughPS::Match(const cHoughPS & aPS2,bool IsLight,tREAL8 aMaxTeta,tREAL8 aDMin,tREAL8 aDMax) const
{
   tREAL8 aDY1 =      DY(aPS2 );
   tREAL8 aDY2 = aPS2.DY(*this);

   // Test the coherence of left/right position 
   if ( ((aDY1>0) ==IsLight) || ((aDY2>0) ==IsLight))  return false;

   if ( DistAnglAntiPar(aPS2) > aMaxTeta)  return false;

   tREAL8 aDYAbs1 = std::abs(aDY1);
   tREAL8 aDYAbs2 = std::abs(aDY2);

   return (aDYAbs1>aDMin) && (aDYAbs1<aDMax) && (aDYAbs2>aDMin) && (aDYAbs2<aDMax);
}


std::vector<std::pair<int,int>> cHoughPS::GetMatches(const std::vector<cHoughPS>&  aVPS,bool IsLight,tREAL8 aMaxTeta,tREAL8 aDMin,tREAL8 aDMax)
{
     std::vector<std::pair<int,int>>    aVMatches;
     std::vector<int>       aIndM( aVPS.size(),-1);
     std::vector<tREAL8>    aCostM( aVPS.size(),1e30);

     //  compute best match
     for (size_t aK1=0 ; aK1<aVPS.size() ; aK1++)
     {
          for (size_t aK2=aK1+1 ; aK2<aVPS.size() ; aK2++)
          {
               if (aVPS[aK1].Match(aVPS[aK2],IsLight,aMaxTeta,aDMin,aDMax))
               {
                  tREAL8 aD12 = aVPS[aK1].Dist(aVPS[aK2],2.0);
		  if (aD12<aCostM[aK1])
		  {
                     aCostM[aK1] = aD12;
		     aIndM[aK1]  = aK2;
		  }
		  if (aD12<aCostM[aK2])
		  {
                     aCostM[aK2] = aD12;
		     aIndM[aK2]  = aK1;
		  }
               }
          }
     }

     //  Test if reciproc match
     for (int aK=0 ; aK<(int)aIndM.size() ; aK++)
     {
          // test to get only one way && reciprocity
          if ((aIndM[aK] > aK) && (aIndM[aIndM[aK]] == aK))
          {
              aVMatches.push_back(std::pair<int,int>(aK,aIndM[aK]));
          }
     }
     return aVMatches;
}




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
    mMoreTeta  (5),
    mFactI2T   ((2.0*M_PI)/mNbTeta),
    mNbRho     (2+round_up(2*mMulRho * mRhoMax)),
    mTabSin    (mNbTeta),
    mDTabSin   (mTabSin.DIm()),
    mTabCos    (mNbTeta),
    mDTabCos   (mTabCos.DIm()),
    mAccum     (cPt2di(mNbTeta+2*mMoreTeta,mNbRho),nullptr,eModeInitImage::eMIA_Null),
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

cHoughPS cHoughTransform::PtToLine(const cPt3dr & aPt) const
{
   // (x,y) in L  <=> x cos(T) + y Sin(T) = R
   tREAL8  aTeta = RInd2Teta(aPt.x());
   tREAL8  aRho  = RInd2Rho(aPt.y());
   cPt2dr  aTgt(-sin(aTeta),cos(aTeta));
   cPt2dr  aP0 = mMiddle + cPt2dr(aRho*cos(aTeta),aRho*sin(aTeta));

   return  cHoughPS(this,cPt2dr(aTeta,aRho),aPt.z(),aP0-aTgt,aP0+aTgt);
}

cPt2dr cHoughTransform::Line2PtInit(const  tSeg2dr &aSeg) const
{
   // Grad = (C,S) 
   // T (-S,C)=  (C,S) * (0,-1) =  Grad * (-J)
   //  Grad =  T * J

   cPt2dr aGrad = VUnit(aSeg.V12() * cPt2dr(0,-1));
   tREAL8 aTeta = Teta(aGrad);
   tREAL8 aRho = Scal(aGrad,aSeg.PMil() - mMiddle);

   return cPt2dr(aTeta,aRho);
}

cPt2dr cHoughTransform::Line2PtPixel(const  tSeg2dr &aSeg) const
{
   cPt2dr aTR = Line2PtInit(aSeg);
   return cPt2dr(Teta2RInd(aTR.x()),Rho2RInd(aTR.y()));
}


// version quick

void  cHoughTransform::Quick_AccumulatePtAndDir(const cPt2dr & aPt,tREAL8 aTetaC,tREAL8 aWeight)
{
      //  compute the index-interval, centered on aTetaC, of size mSigmTeta
      int  iTeta0 = round_down(Teta2RInd(aTetaC-mSigmTeta));
      int  iTeta1 = round_up(Teta2RInd(aTetaC+mSigmTeta));

      {
          static bool First= true;
          if (First)
             StdOut() << "NBTETA IN HOUGH=" << iTeta1 - iTeta0 << "\n";
          First = false;
      }

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
          int aITetaOK = iTeta%mNbTeta;  // % 2PI => put indexe in interval
          //   equation :  (x,y) in L  <=> x cos(T) + y Sin(T) = R
          tREAL8 aRho =   aX * mDTabCos.GetV(aITetaOK) + aY*mDTabSin.GetV(aITetaOK) ;
          int  aIndRho = round_ni(Rho2RInd(aRho)); // convert Real rho in its index inside accumulator
	  //  Finally accumulate
          mDAccum.AddVal(cPt2di(aITetaOK,aIndRho),1.0);
      }
}

void  cHoughTransform::Accurate_AccumulatePtAndDir(const cPt2dr & aPt,tREAL8 aTetaC,tREAL8 aWeight)
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

void cHoughTransform::ExtendMoreTeta() const
{
    cDataIm2D<tREAL4>&  aNCAcc = const_cast<cDataIm2D<tREAL4>& > (mDAccum);

    for (int  aKTeta=0 ; aKTeta< 2* mMoreTeta ; aKTeta++)
    {
        for (int  aKRho=0 ; aKRho< mNbRho ; aKRho++)
		aNCAcc.SetV(cPt2di(aKTeta+mNbTeta,aKRho),mDAccum.GetV(cPt2di(aKTeta,aKRho)));
    }
}


std::vector<cPt3dr>  cHoughTransform::ExtractLocalMax(size_t aNbMax,tREAL8 aDist,tREAL8 aThrAvg,tREAL8 aThrMax) const
{
    ExtendMoreTeta();
    std::vector<cPt2di>  aVNeigh = SortedVectOfRadius(0.5,aDist,false);

    // [1]  Compute average , max and  threshold
    //    [1.A]   : max & avg
    tREAL8 aAvg = 0;
    tREAL8 aVMax = 0;
    for (const auto & aPix : mDAccum)
    {
         tREAL8 aVal = mDAccum.GetV(aPix);
	 aAvg += aVal;
	 UpdateMax(aVMax,aVal);
    }
    aAvg /= mDAccum.NbElem();

    //    [1.B]    Threshlod is the stricter of both
    tREAL8 aThrHold = std::max(aAvg*aThrAvg,aVMax*aThrMax);


    std::vector<cPt2di> aVPMaxLoc;
    // int aY0 = 1+round_up(aDist);
    // int aY1 =  mNbRho - aY0;

    // We parse only interoir to avoid Out in Rho and duplication in teta
    cRect2 aRecInt(mDAccum.Dilate (-mMoreTeta));

    for (const auto & aPix : aRecInt)
    {
        const tREAL4 & aVPix = mDAccum.GetV(aPix);
        if (      (aVPix>aThrHold) 
               &&  (aVPix >=  mDAccum.GetV(aPix-cPt2di(0,1)))
               &&  (aVPix >   mDAccum.GetV(aPix+cPt2di(0,1)))
	   )
        {
            bool IsMaxLoc=true;
            for (size_t aKNeigh=0 ; (aKNeigh<aVNeigh.size()) && IsMaxLoc ; aKNeigh++)
	    {
                 const cPt2di & aNeigh = aVNeigh.at(aKNeigh);
		 int aDX = aNeigh.x() ;
		 int aDY = aNeigh.y() ;
		 // use %, because may be out if Dist>mMoreTeta ...
		 const tREAL8 & aVNeigh =  mDAccum.GetV(cPt2di((aPix.x() + mNbTeta + aDX) % mNbTeta,aPix.y() + aDY));
		 if (aVNeigh>aVPix)
		 {
                     IsMaxLoc = false;
		 }
		 else if  (aVNeigh==aVPix)
		 {
                      if ((aDX>0) || ((aDX==0) && (aDY>0)))
                          IsMaxLoc = false;
		 }
	    }
	    if (IsMaxLoc)
               aVPMaxLoc.push_back(aPix);
	}
    }
    
    // [4] Refine the point and give a value to the max

    std::vector<cPt3dr> aRes;
    cAffineExtremum<tREAL4>  aAffin(mDAccum,1.5); // 1.5 => radius , 8-neighboorhood
    for (const auto aPt : aVPMaxLoc)
    {
         cPt2dr aPAff = aAffin.OneIter(ToR(aPt));
	 if ( mDAccum.InsideBL(aPAff))
	 {
            tREAL8 aTeta = aPAff.x();
	    if (aTeta>mNbTeta) 
               aTeta -= mNbTeta;
            aRes.push_back(cPt3dr(aTeta,aPAff.y(),GetValueBlob(aPt,7)));
	 }
    }

    // [5] Sort with highest value first, then select NbMax
    SortOnCriteria(aRes,[](const auto & aP) {return -aP.z();});
    while (aRes.size() > aNbMax)  
          aRes.pop_back();

    // StdOut() << "NBMAXLOC " << aRes  << "\n";

    return aRes;
}

};
