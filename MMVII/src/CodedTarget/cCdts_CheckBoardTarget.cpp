#include "cCheckBoardTargetExtract.h"
#include "FilterCodedTarget.h"

namespace MMVII
{

/* ********************************************* */
/*                                               */
/*           cOptimSymetryOnImage                */
/*                                               */
/* ********************************************* */

template <class Type> 
    cOptimSymetryOnImage<Type>::cOptimSymetryOnImage(const cPt2dr & aC0,const tDIm & aDIm,const cDiffInterpolator1D & anInt) :
         mC0     (aC0),
         mDIm    (aDIm),
	 mInterp (anInt)
{
}

template <class Type> cPt1dr cOptimSymetryOnImage<Type>::Value(const cPt2dr & aDelta ) const 
{
     cSymMeasure<tREAL8> aSymM; // Structure to compute symetry coeff
     cPt2dr a2NewC = (mC0 + aDelta) * 2.0;  // twice the center actualized

     for (const auto & aP1 : mPtsOpt)
     {
          cPt2dr aP2 = a2NewC - aP1;
	  if (mDIm.InsideInterpolator(mInterp,aP1) && mDIm.InsideInterpolator(mInterp,aP2))
             aSymM.Add(mDIm.GetValueInterpol(mInterp,aP1),mDIm.GetValueInterpol(mInterp,aP2));
     }

     return cPt1dr(aSymM.Sym(1e-5));
}

template <class Type> void cOptimSymetryOnImage<Type>::AddPts(const cPt2dr & aPt)
{
   mPtsOpt.push_back(aPt);
}

template <class Type> tREAL8 cOptimSymetryOnImage<Type>::OneIterLeastSqGrad()
{
     cLeasSqtAA<tREAL8>  aSys(2);

     cSymMeasure<tREAL8> aSymM; //
     for (const auto & aP1 : mPtsOpt)
     {
          cPt2dr aP2 = mC0 * 2.0 - aP1;
	  if (mDIm.InsideInterpolator(mInterp,aP1) && mDIm.InsideInterpolator(mInterp,aP2))
	  {
              auto [aV1,aG1] = mDIm.GetValueAndGradInterpol(mInterp,aP1);
              auto [aV2,aG2] = mDIm.GetValueAndGradInterpol(mInterp,aP2);

	      aSymM.Add(aV1,aV2);

	      cPt2dr a2G2= (aG2*2.0);
	      cDenseVect<tREAL8> aDV (a2G2.ToVect());

	      //  NewV2 = aV2 + 2 * aG2 . delta  = aV1
	      aSys.PublicAddObservation(1.0,aDV,aV1-aV2);


	      //  StdOut() << aG1 - aG2 << "\n";
	  }
     }

     cDenseVect<tREAL8> aSol = aSys.PublicSolve();
     cPt2dr aDelta = cPt2dr::FromVect(aSol);

     mC0 += aDelta;
     for (auto & aP1 : mPtsOpt)
         aP1 += aDelta;

     return  aSymM.Sym(1e-5);

     //  StdOut() << " Ggggg " << aDelta << " " << aSymM.Sym(1e-5) << "\n";

}

template <class Type> int cOptimSymetryOnImage<Type>::IterLeastSqGrad(tREAL8 aGainMin,int aNbMax)
{
   tREAL8 aLastScore =  OneIterLeastSqGrad();

   for (int aK= 1 ; aK<aNbMax ; aK++)
   {
       tREAL8 aNewScore = OneIterLeastSqGrad();
       if (aNewScore>aLastScore-aGainMin)
          return aK+1;
       aLastScore = aNewScore;
   }
   return aNbMax;
}

template class cOptimSymetryOnImage<tREAL4>;


namespace NS_CHKBRD_TARGET_EXTR { 


/* ================================================================= */


/* **************************************************** */
/*                     cCdSadle                         */
/* **************************************************** */

int cCdSadle::TheCptNum=0;
int cCdSadle::TheNum2Debug=-2;

cCdSadle::cCdSadle (const cPt2dr & aC,tREAL8 aCrit,bool isPTest) : 
    mC        (aC) , 
    mSadCrit  (aCrit) ,
    mIsPTest  (isPTest),
    mNum      (TheCptNum++)
{
}
cCdSadle::cCdSadle () :
    mNum  (-1)
{
}

bool cCdSadle::Is4Debug() const  {return  mNum == TheNum2Debug;}




/* ***************************************************** */
/*                                                       */
/*                    cCdRadiom                          */
/*                                                       */
/* ***************************************************** */

cCdRadiom::cCdRadiom
(
    const cAppliCheckBoardTargetExtract * anAppli,
    const cCdSym & aCdSym,
    const cDataIm2D<tREAL4> & aDIm,
    tREAL8 aTeta0,
    tREAL8 aTeta1,
    tREAL8 aLength,
    tREAL8 aThickness
) :
       cCdSym      (aCdSym),
       mAppli      (anAppli),
       mIsOk       (false),
       mDIm        (&aDIm),
       mTetas      {aTeta0,aTeta1},
       mLength     (aLength),
       mThickness  (aThickness),
       mCostCorrel (2.001),   // over maximal theoreticall value
       mRatioBW    (0)
{
    int aNbIn0=0,aNbIn1=0;

    cMatIner2Var<tREAL8> aCorGrayAll;
    cMatIner2Var<tREAL8> aCorGrayInside;

    cTmpCdRadiomPos  aCRC(*this,aThickness);

    std::vector<cPt2di> aVPixEllipse;
    ComputePtsOfEllipse(aVPixEllipse);
    for (const auto & aPImI : aVPixEllipse)
    {
	tREAL8 aValIm = aDIm.GetV(aPImI);
	cPt2dr aPImR = ToR(aPImI);

	auto [aState,aGrayTh] = aCRC.TheorRadiom(aPImR);

	if  (IsInside(aState))
	{
            aCorGrayInside.Add(aGrayTh,aValIm);
            aNbIn0 += (aState == eTPosCB::eInsideBlack);
            aNbIn1 += (aState == eTPosCB::eInsideWhite);
	}
	if  (IsOk(aState))
	{
            aCorGrayAll.Add(aGrayTh,aValIm);
	}
    }

    if ((aNbIn0==0) && (aNbIn1==0))
       return;

    mRatioBW = std::min(aNbIn0,aNbIn1) / (tREAL8) std::max(aNbIn0,aNbIn1);
    if (mRatioBW <0.05)
    {
       return ;
    }

    mIsOk = true;

    mCostCorrel = 1-aCorGrayAll.Correl();
    auto [a,b] = aCorGrayInside.FitLineDirect();
    mBlack = b ;
    mWhite = a+b;
}

tREAL8 cCdRadiom::Threshold(tREAL8 aWW) const 
{
	 return mBlack*(1-aWW) + mWhite *aWW;
}

void cCdRadiom::OptimSegIm(const cDataIm2D<tREAL4> & aDIm,tREAL8 aLength)
{
     // StdOut() <<  "TttTT=" << Threshold() << " " << mBlack << " " << mWhite << " " << mRatioBW << "\n";

     std::vector<cSegment2DCompiled<tREAL8>> aVSegOpt;
     for (int aKTeta=0 ; aKTeta<2 ; aKTeta++)
     {
         cPt2dr aTgt = FromPolar(aLength,mTetas[aKTeta]);
         tSeg2dr aSegInit(mC-aTgt,mC+aTgt);
         cOptimSeg_ValueIm<tREAL4>  aOSVI(aSegInit,0.5,aDIm,Threshold());
	 tSeg2dr  aSegOpt = aOSVI.OptimizeSeg(0.5,0.01,true,2.0);

	 aVSegOpt.push_back(aSegOpt);
	 mTetas[aKTeta] = Teta(aSegOpt.V12());
	 // mTetas[aKTeta] = aSegOpt.I//
     }

     cPt2dr aC = aVSegOpt.at(0).InterSeg(aVSegOpt.at(1));

     mC = aC;
     cScoreTetaLine::NormalizeTetaCheckBoard(mTetas);
}

void cCdRadiom::ComputePtsOfEllipse(std::vector<cPt2di> & aRes) const
{
	ComputePtsOfEllipse(aRes,mLength);
}


void cCdRadiom::ComputePtsOfEllipse(std::vector<cPt2di> & aRes,tREAL8 aLength) const
{
    aRes.clear();
    // [1]  Compute the affinity that goes from unity circle to ellipse
    //  ----  x,y ->   mC + x V0 + y V1  ------
    cPt2dr aV0 = FromPolar(aLength,mTetas[0]);
    cPt2dr aV1 = FromPolar(aLength,mTetas[1]);

    cAff2D_r aMapEll2Ori(mC,aV0,aV1);
    cAff2D_r aMapOri2Ell = aMapEll2Ori.MapInverse();

    // [2] Compute the bounding box containing the ellipse
    cTplBoxOfPts<tREAL8,2> aBox;
    int aNbTeta = 100;
    for (int aKTeta=0 ; aKTeta<aNbTeta ; aKTeta++) // sample the frontiers 
    {
         aBox.Add(aMapEll2Ori.Value(FromPolar(1.0, (2.0*M_PI * aKTeta) / aNbTeta)));
    }

    cBox2di aBoxI = aBox.CurBox().Dilate(2.0).ToI(); // add a bit of margin

    // [3]  Parse the bouding box and select point OK
    for (const auto & aPix : cRect2(aBoxI))
    {
         if (Norm2(aMapOri2Ell.Value(ToR(aPix))) < 1)
            aRes.push_back(aPix);
    }
}

bool cCdRadiom::PtIsOnLine(const cPt2dr & aPAbs,tREAL8 aTeta) const
{
    cSegment2DCompiled<tREAL8> aSeg(mC,mC+FromPolar(1.0,aTeta));

    cPt2dr aPLoc = aSeg.ToCoordLoc(aPAbs);

    if (std::abs(aPLoc.y()) <= 1.0 + std::abs(aPLoc.x()) /30.0)
       return true;

    return false;
}

bool cCdRadiom::PtIsOnEllAndRefine(cPt2dr & aPtAbs) const
{
    // point must be far enough of center (because close to, it's not easily separable)
    if  (Norm2(aPtAbs - mC)<3.0)
        return false;

    // point canot be a point of the line
    for (const auto & aTeta : mTetas )
        if (PtIsOnLine(aPtAbs,aTeta))
           return false;

    // extract the point that has the gray threshold (assuming gray starting point is bellow)
    cGetPts_ImInterp_FromValue<tREAL4> aGIFV(*mDIm,Threshold(),0.1,aPtAbs, VUnit(aPtAbs - mC));
    cPt2dr aNewP = aPtAbs;
    if (aGIFV.Ok())
    {
        // if interpoleted point is to far from initial : suscpicious
        aNewP = aGIFV.PRes();
	if (Norm2(aPtAbs-aNewP)>2.0)
           return false;

        cPt2dr aPGr =  Proj(mDIm->GetGradAndVBL(aNewP));
	// StdOut() << "PGR=== " << aPGr << "\n";
	tREAL8 aSc =  std::abs(CosWDef(aPGr,aNewP-mC,1.0));
	if (aSc<0.5)
           return false;

	aPtAbs = aNewP;
    }
    else
	  return false;


    return true;
}

void cCdRadiom::SelEllAndRefineFront(std::vector<cPt2dr> & aRes,const std::vector<cPt2di> & aFrontI) const
{
    aRes.clear();
    for (const auto & aPix : aFrontI)
    {
         cPt2dr aRPix = ToR(aPix);
	 if (PtIsOnEllAndRefine(aRPix))
            aRes.push_back(aRPix);
    }
}

bool cCdRadiom::FrontBlackCC(std::vector<cPt2di> & aVFront,cDataIm2D<tU_INT1> & aDMarq,int aNbMax) const
{
    std::vector<cPt2di> aRes;
    aVFront.clear();

    std::vector<cPt2di> aVPtsEll;
    ComputePtsOfEllipse(aVPtsEll,std::min(mLength,5.0));

    tREAL8 aThrs = Threshold();
    for (const auto & aPix : aVPtsEll)
    {
        if (mDIm->GetV(aPix)<aThrs)
	{
            aDMarq.SetV(aPix,1);
	    aRes.push_back(aPix);
	}
    }

    size_t aIndBot = 0;
    const std::vector<cPt2di> & aV4 = Alloc4Neighbourhood();

    cRect2  aImOk(aDMarq.Dilate(-10));
    bool isOk = true;

    while ((aIndBot != aRes.size())  && isOk)
    {
          for (const auto & aDelta : aV4)
          {
              cPt2di aPix = aRes.at(aIndBot) + aDelta;
	      if ((aDMarq.GetV(aPix)==0) && (mDIm->GetV(aPix)<aThrs) )
	      {
                 if (aImOk.Inside(aPix))
		 {
                    aDMarq.SetV(aPix,1);
		    aRes.push_back(aPix);
		    if ((int) aRes.size() == aNbMax)
                       isOk = false;
	         }
	         else
	         {
                    isOk = false;
	         }
	      }
          }
	  aIndBot++;
    }
    if (!isOk) 
    {
        for (const auto & aPix : aRes)
            aDMarq.SetV(aPix,0);
        return false;
    }

    const std::vector<cPt2di> & aV8 = Alloc8Neighbourhood();
    // compute frontier points
    for (const auto & aPix : aRes)
    {
        bool has8NeighWhite = false;
        for (const auto & aDelta : aV8)
	{
	     if (aDMarq.GetV(aPix+aDelta)==0) 
                has8NeighWhite = true;
	}

	if (has8NeighWhite)
	{
            aVFront.push_back(aPix);
	}
    }
    if (false && Is4Debug())
    {
	     StdOut()  << "--xxx--HASH " << HashValue(aVFront,true) << " SZCC=" << aRes.size() 
		       << " HELLL=" << HashValue(aVPtsEll,true)
		       << " HELLL=" << HashValue(aVPtsEll,true)
		       << " C=" << mC  << " Thr=" << aThrs  
		       <<  " L=" << mLength << "\n";
    }
    // StdOut() << "FFFF=" << aVFront << "\n";

    for (const auto & aPix : aRes)
        aDMarq.SetV(aPix,0);

    return isOk;
}



	// if (has8NeighWhite && PtIsOnEll(aRPix))

void  cCdRadiom::ShowDetail
      (
            int aCptMarq,
	    const cScoreTetaLine & aSTL,
	    const std::string & aNameIm,
	    cDataIm2D<tU_INT1> & aMarq,
	    cFullSpecifTarget *aSpec
       ) const
{
      cCdEllipse aCDE(*this,aMarq,-1,false);
      if (! aCDE.IsOk())
      {
         StdOut()    << "   @@@@@@@@@@@@@@@@@@@@@@@@@@@@ "  << aCptMarq << "\n";
         return;
      }

      std::pair<tREAL8,tREAL8> aPairTeta(mTetas[0],mTetas[1]);

      aCDE.DecodeByL2CP(2.0/3.0);
      StdOut()    << " CptMarq=" << aCptMarq 
	          << " NUM="     << mNum
		  << "  Corrrr=" <<  mCostCorrel 
                   << " Ratio=" <<  mRatioBW
		  << " V0="<< mBlack << " V1=" << mWhite 
		  << " ScTeta=" << aSTL.Score2Teta(aPairTeta,2.0)
		  << " ScSym=" << mSymCrit
		  << " LLL=" << mLength
		  << " ThickN=" << mThickness
		  << " CODE=[" <<   aCDE.Code() << "]"
		  << " C="   <<  mC
		  <<  " DELL=" << aCDE.MaxEllD()
		  <<  " OK=" << aCDE.mIsOk
		  <<  " OUTCB=" << aCDE.BOutCB()
		  << "\n";
}

/* ***************************************************** */
/*                                                       */
/*                    cCdEllipse                         */
/*                                                       */
/* ***************************************************** */

void cCdEllipse::GenImageFail(const std::string & aWhyFail)
{
     static int aCpt=0;
     cRGBImage  aIm = mAppli->GenImaRadiom(*this);
     StdOut()  << "Fail for Num=" << mNum << " Cpt=" << aCpt << " reason=" << aWhyFail << "\n";
     aIm.ToJpgFileDeZoom(mAppli->NameVisu("Failed"+ aWhyFail,ToStr(aCpt++)),1);
}


cCdEllipse::cCdEllipse(const cCdRadiom & aCdR,cDataIm2D<tU_INT1> & aMarq,int aNbMax,bool isCircle) :
     cCdRadiom (aCdR),
     mSpec     (mAppli->Specif()),
     mEll      (cPt2dr(0,0),0,1,1),
     mMaxEllD  (0.0),
     mCode     (nullptr),
     mBOutCB   (false),
     mIsCircle (isCircle)
{

     if (! mIsOk)
     {
         if(mIsPTest) StdOut() << "Ref cCdEllipse at L=" << __LINE__ << "\n" ;
         return;
     }
     mIsOk = true;
     std::vector<cPt2di> aIFront;
     mIsOk = FrontBlackCC(aIFront,aMarq,aNbMax);

     if (! mIsOk)
     {
        if(mIsPTest) StdOut() << "Ref cCdEllipse at L=" << __LINE__ << "\n" ;
	return;
     }

     {
         cTmpCdRadiomPos aCRP(*this,1.0);
         for (const auto & aPix : aIFront)
         {
             auto [aPos,aGray] = aCRP.TheorRadiom(ToR(aPix),2.0,1/20.0);
	     if (aPos == eTPosCB::eInsideWhite)
	     {
		     mBOutCB = true;
	     }
         }
	 if (mBOutCB)
	 {
             if(mIsPTest) StdOut() << "Ref cCdEllipse at L=" << __LINE__ << "\n" ;
             mIsOk = false;
             return;
	 }
     }

     std::vector<cPt2dr> aEllFr;
     SelEllAndRefineFront(aEllFr,aIFront);

     if ((int) aEllFr.size() < (mIsCircle ? 2 : mAppli->NbMinPtEllipse())  )
     {
        if  (mIsPTest) 
	    GenImageFail("NbEllipse");
        mIsOk = false;
        return;
     }

     mIsOk = true;

     cEllipse_Estimate anEE(mC,false,mIsCircle);
     for (const auto & aPixFr : aEllFr)
     {
         anEE.AddPt(aPixFr);
     }

     mEll = anEE.Compute();

     if  (!mEll.Ok())
     {
        if(mIsPTest) 
	{
	    GenImageFail("BadEll");
	}
        mIsOk = false;
        return;
     }

     tREAL8 aThrs = 0.6+ mEll.LGa()/40.0;
     for (const auto & aPixFr : aEllFr)
     {
         tREAL8 aD =  mEll.NonEuclidDist(aPixFr);
	 UpdateMax(mMaxEllD,aD);
	 if (aD>aThrs)
	 {
           if(mIsPTest) StdOut() << "Ref cCdEllipse at L=" << __LINE__ << "\n" ;
            mIsOk = false;
	    return;
	 }
     }


     // In specif the corner are for a white sector, with name of transition (B->W or W->B)
     // coming with trigonometric convention; here the angle have been comouted for a  black sector
     // The correction for angle have been made experimentally ...
     mCornerlEl_WB = mEll.InterSemiLine(mTetas[0]);
     mCornerlEl_BW = mEll.InterSemiLine(mTetas[1]+M_PI);



     cAff2D_r::tTabMin  aTabIm{mC,mCornerlEl_WB,mCornerlEl_BW};
     cAff2D_r::tTabMin  aTabMod{mSpec->Center(),mSpec->CornerlEl_WB(),mSpec->CornerlEl_BW()};

     mAffIm2Mod =  cAff2D_r::FromMinimalSamples(aTabIm,aTabMod);
     // mAffIm2Mod
}

bool cCdEllipse::BOutCB()   const {return mBOutCB;}
bool cCdEllipse::IsCircle() const {return mIsCircle;}

bool cCdEllipse::IsOk() const {return mIsOk;}
void cCdEllipse::AssertOk() const
{
    MMVII_INTERNAL_ASSERT_tiny(mIsOk,"No ellipse Ok in cCdEllipse");
}

cPt2dr  cCdEllipse::M2I(const cPt2dr & aPMod) const
{
    AssertOk();
    return mAffIm2Mod.Inverse(aPMod);
}

cPt2dr  cCdEllipse::I2M(const cPt2dr & aPMod) const
{
    AssertOk();
    return mAffIm2Mod.Value(aPMod);
}


const cOneEncoding * cCdEllipse::Code() const {return mCode;}
tREAL8 cCdEllipse::MaxEllD() const {return mMaxEllD;}

const cEllipse & cCdEllipse::Ell() const {AssertOk(); return mEll;}
const cPt2dr & cCdEllipse::CornerlEl_WB() const {AssertOk(); return mCornerlEl_WB;}
const cPt2dr & cCdEllipse::CornerlEl_BW() const {AssertOk(); return mCornerlEl_BW;}

// tREAL8 cCdEllipse::ScoreCodeBlack(const cPt2dr& aDir,tREAL8 aRho,

std::pair<tREAL8,cPt2dr>  cCdEllipse::Length2CodingPart(tREAL8 aWeighWhite,const cPt2dr & aModCenterBit) const
{
    // Rho end search, 1.5 theoretical end of code , highly over estimated (no risk ?)
    tREAL8 aRhoMaxRel = mSpec->Rho_2_EndCode() * 1.5;
    std::pair<tREAL8,cPt2dr>  aNoValue(-1,cPt2dr(0,0));

    // Not sur meaningful with other mode
    MMVII_INTERNAL_ASSERT_tiny(mSpec->Type()==eTyCodeTarget::eIGNIndoor,"Bad code in Length2CodingPart");


    cPt2dr aDirModel = VUnit(mSpec->Pix2Norm(aModCenterBit));  // -> normalize coord -> unitary vect
    // distance , ratio on the line mC-Pt, between a normalized model pixel, and it corresponding image pixel
    tREAL8  aMulN2I =  Norm2(M2I(mSpec->Norm2Pix(aDirModel))-mC);
							       //
    cPt2dr aDirIm = VUnit(M2I(aModCenterBit)-mC); // direction of line in image

    // Rho begin search in  image, at mid position between check board and begining of code
    tREAL8 aRho0 =   aMulN2I * ((mSpec->Rho_0_EndCCB()+mSpec->Rho_1_BeginCode()) / 2.0);

    // Rho end search, 1.5 theoretical end of code , highly over estimated (no risk ?)
    tREAL8 aRho1 =    aMulN2I * aRhoMaxRel;
    // step of research, overly small (no risk ?)
    tREAL8 aStepRho = 0.2;
    int aNbRho = round_up((aRho1-aRho0) / aStepRho);
    aStepRho  = (aRho1-aRho0) / aNbRho;

    // threshold for value computing the black code
    tREAL8 aThresh =  aWeighWhite * mWhite + (1-aWeighWhite) * mBlack;

    for (int aKRho = 0 ; aKRho<=aNbRho ; aKRho++)
    {
        tREAL8 aRho = aRho0 + aKRho * aStepRho;
	cPt2dr aPt =  mC + aDirIm * aRho;

        if (!mDIm->InsideBL(aPt))  // we are too far, end of game
	{
           return aNoValue;
	}
	else
	{
            auto [aVal,aGrad] = mDIm->GetPairGradAndVBL(aPt);
	    // we test also orientation of gradient, because with very small target, there is no clear separation , 
	    // and we dont want to accept the checkboard as a 
	    if ((aVal < aThresh) &&  (Scal(aGrad,aDirIm)<0))
	    {
               return std::pair<tREAL8,cPt2dr> (aRho/aMulN2I,aPt);
	    }
	}
    }
    std::pair<tREAL8,cPt2dr>  aDef(aRhoMaxRel,mC + aDirIm*aRhoMaxRel*aMulN2I);

    return aDef;
}

tREAL8 cCdEllipse::ComputeThresholdsByRLE(const std::vector<std::pair<int,tREAL8>> & aVBR) const
{
    // cPt2di MaxRunLength(tU_INT4 aVal,size_t aPow2);

   
   int aMaxWL = mSpec->Specs().mMaxRunL.x(); // Max run lenght for 0 (= white in general, 2 adapt ...)
   aMaxWL = std::min(aMaxWL,(int)mSpec->NbBits());

   size_t aFlag = 0;
   int aRunLE = mSpec->NbBits();
   size_t aKBit = 0;
   while ((aMaxWL<aRunLE)  && (aKBit<aVBR.size()))
   {
	aFlag |= ((size_t)1<<aVBR.at(aKBit).first);
        aKBit++;
	aRunLE = MaxRunLength(aFlag,(size_t)1<<mSpec->NbBits()).x();
   }

   return aVBR.at(aKBit-1).second * 1.2;

}

tREAL8 cCdEllipse::ComputeThresholdsMin(const std::vector<std::pair<int,tREAL8>> & aVBR) const
{
   tREAL8 aV0 = aVBR[0].second;

   return aV0 * 1.25;
}


void cCdEllipse::DecodeByL2CP(tREAL8 aWeighWhite) 
{
     std::vector<std::pair<int,tREAL8>  >  aVBR;  // Vector Bits/Rho
     const auto & aVC = mSpec->BitsCenters();     // Vector centers
     for (size_t aKBit=0 ; aKBit<aVC.size() ; aKBit++)
     {
         auto [aRho,aCenter] = Length2CodingPart(aWeighWhite,aVC[aKBit]);
	 // something bad, like out of image, occured
	 if (aRho<0)
            return ;
	 aVBR.push_back(std::pair<int,tREAL8>(aKBit,aRho));
     }

     SortOnCriteria(aVBR,[](const auto & aPair) {return aPair.second;});
     tREAL8 aRhoMin = aVBR.at(0).second;


     if (0)
     {
         std::vector<std::pair<int,tREAL8>  > aDup = aVBR;
         SortOnCriteria(aDup,[](const auto & aPair) {return aPair.first;});
	 for (const auto & [aBit,aRho] : aDup)
             StdOut() << " RRR=" << aRho/aRhoMin << "\n";
     }

     // tREAL8 aTreshold = ComputeThresholdsMin(aVBR);
     tREAL8 aTreshold = ComputeThresholdsByRLE(aVBR);

     cDecodeFromCoulBits aDec(mSpec);
     for (const auto & [aBit,aRho] : aVBR)
     {
         aDec.SetColBit(aRho<aTreshold , aBit);
     }

     mCode = aDec.Encoding();
}

const cOneEncoding *  cCdEllipse::BasicDecode(tREAL8 aWW)
{
    cDecodeFromCoulBits aDec(mSpec);
    const auto & aVC = mSpec->BitsCenters();
    tREAL8 aThrs = Threshold(aWW);

    for (size_t aKBit=0 ; aKBit<aVC.size() ; aKBit++)
    {
       cPt2dr aPIm = M2I(aVC.at(aKBit));
       if (mDIm->InsideBL(aPIm))
       {
	  aDec.SetColBit(mDIm->GetVBL(aPIm) < aThrs , aKBit);
       }
       else
	    return nullptr;
    }
    return aDec.Encoding();
}


/* ***************************************************** */
/*                                                       */
/*                    cTmpCdRadiomPos                    */
/*                                                       */
/* ***************************************************** */


cTmpCdRadiomPos::cTmpCdRadiomPos(const cCdRadiom & aCDR,tREAL8 aThickness) :
    cCdRadiom   (aCDR),
    mThickness  (aThickness),
    mSeg0       (mC,mC+FromPolar(1.0,mTetas[0])),
    mSeg1       (mC,mC+FromPolar(1.0,mTetas[1]))
{
}

std::pair<eTPosCB,tREAL8>  cTmpCdRadiomPos::TheorRadiom(const cPt2dr &aPt,tREAL8 aThickInit,tREAL8 aSteep) const
{
    eTPosCB aPos = eTPosCB::eUndef;
    tREAL8 aGrayTh = -1;

    // we compute locacl coordinates because the sign of y indicate if we are left/right of the oriented segment
    // and sign of x indicate if we are before/after the centre
    cPt2dr aLoc0 = mSeg0.ToCoordLoc(aPt);
    tREAL8  aY0 = aLoc0.y();
    tREAL8  aThick0 = aThickInit + aSteep * std::abs(aLoc0.x());

    cPt2dr aLoc1 = mSeg1.ToCoordLoc(aPt);
    tREAL8  aY1 = aLoc1.y();
    tREAL8  aThick1 = aThickInit + aSteep * std::abs(aLoc1.x());

    // compute if we are far enough of S0/S1 because the computation of gray will change
    //  black/white if far  enough, else interpolation
    bool FarS0 = std::abs(aY0)> aThick0; 
    bool FarS1 = std::abs(aY1)> aThick1;

    if ( FarS0 && FarS1)
    {
       if ((aY0>0)!=(aY1>0))
       {
           aPos = eTPosCB::eInsideBlack;
	   aGrayTh = 0.0;
       }
       else
       {
           aPos = eTPosCB::eInsideWhite;
	   aGrayTh = 1.0;
       }
    }
    else if  ((!FarS0) && FarS1)
    {
        // (! FarS0) => teta1
        // Red = teta1 , black on left on image, right on left in coord oriented
	 aPos = eTPosCB::eBorderRight;
         int aSignX = (aLoc0.x() >0) ? -1 : 1;
         aGrayTh = (aThick0+aSignX*aY0) / (2.0*aThick0);
    }
    else if  (FarS0 && (!FarS1))
    {
	 aPos = eTPosCB::eBorderLeft;
	 int aSignX = (aLoc1.x() <0) ? -1 : 1;
	 aGrayTh = (aThick1+aSignX*aY1) / (2.0 * aThick1);
    }

    return std::pair<eTPosCB,tREAL8>(aPos,aGrayTh);
}

std::pair<eTPosCB,tREAL8>  cTmpCdRadiomPos::TheorRadiom(const cPt2dr &aPt) const
{
	return TheorRadiom(aPt,mThickness,0.0);
}


/* ********************************************* */
/*                                               */
/*               cOptimPosCdM                    */
/*                                               */
/* ********************************************* */



class cOptimPosCdM : public cOptimSymetryOnImage<tREAL4>
{
	public :
           cOptimPosCdM(const cCdMerged & aCdM,const cDiffInterpolator1D & );

           // cPt1dr Value(const cPt2dr & ) const override;
	   typedef cSegment2DCompiled<tREAL8> tSeg;

	private :
	    void AddPts1Seg(const cPt2dr & aMaster,  const cPt2dr & aSecond,bool toAvoid2);
            const cCdMerged&        mCdM;
};



cOptimPosCdM::cOptimPosCdM(const cCdMerged & aCdM,const cDiffInterpolator1D & aInt)  :
        cOptimSymetryOnImage<tREAL4>(aCdM.mC0,(*aCdM.mDIm0),aInt),
	mCdM      (aCdM)
	// mCurInt   (aInt)
{
	AddPts1Seg(aCdM.CornerlEl_WB(), aCdM.CornerlEl_BW(),true);
	AddPts1Seg(aCdM.CornerlEl_BW(), aCdM.CornerlEl_WB(),false);
}

void cOptimPosCdM::AddPts1Seg(const cPt2dr & aSCorn1, const cPt2dr & aSCorn2,bool toAvoid2)
{
     cPt2dr  aCorn1 = aSCorn1  * mCdM.mScale;
     cPt2dr  aCorn2 = aSCorn2  * mCdM.mScale;

     tREAL8 aStep = 0.25;
     tREAL8 aWidth = 1.0;
     tREAL8 aL1 = std::min(10.0,Norm2(aCorn1-mCdM.mC0)-1.0);
     // cPt2dr  aCorn2 = aSCorn2  * mScale;

     int aNbX = round_up(aL1/aStep);
     tREAL8 aStepX = aL1 / aNbX;

     int aNbY = round_up(aWidth/aStep);
     tREAL8 aStepY = aWidth / aNbY;

     tSeg aSeg1(mCdM.mC0,aCorn1);
     tSeg aSeg2(mCdM.mC0,aCorn2);

     for (int aKX=-aNbX ; aKX<=aNbX ; aKX++)
     {
         for (int aKY=0 ; aKY<=aNbY ; aKY++)  // KY=0 : we take only one point /2 
	 {
             if ((aKY>0)  || (aKX>0))
             {
                  cPt2dr aPLoc(aKX*aStepX,aKY*aStepY);
	          cPt2dr aPAbs = aSeg1.FromCoordLoc(aPLoc);
	          if ((!toAvoid2)  ||  (aSeg2.DistLine(aPAbs) >aWidth))
		  {
                     mPtsOpt.push_back(aPAbs);
		     // StdOut() << "PAAAA " << aPAbs - mCdM.mC0  << "\n";
		  }
             }
	 }
     }
}

/* ********************************************* */
/*                                               */
/*                  cCdMerged                    */
/*                                               */
/* ********************************************* */

cCdMerged::cCdMerged(const cDataIm2D<tREAL4> * aDIm0,const cCdEllipse & aCDE,tREAL8 aScale) :
    cCdEllipse (aCDE),
    mScale     (aScale),
    mC0        (mC * mScale),
    mDIm0      (aDIm0)
{
}

void cCdMerged::GradOptimizePosition(const cDiffInterpolator1D & anInt,tREAL8 aStepEnd)
{
    cOptimPosCdM aCdGrad(*this,anInt);
         // StdOut() << "-----------  TEST GRAD ----------------  V0=" << aCdGrad.Value(cPt2dr(0,0)) << " \n";

    aCdGrad.IterLeastSqGrad(aStepEnd,5);
    mC0 = aCdGrad.C0();
}

void  cCdMerged::HeuristikOptimizePosition(const cDiffInterpolator1D & anInt,tREAL8 aStepEnd)
{
      cOptimPosCdM aCdtOpt(*this,anInt);
      cOptimByStep anOpt(aCdtOpt,true,1.0);


      // tREAL8  aV0 = aCdtOpt.Value(cPt2dr(0,0)).x();
      auto [aVal,aDelta] =   anOpt.Optim(cPt2dr(0,0),0.02,aStepEnd);

     //  StdOut() << "  HEURISTIK =" << aV0 << " => " <<  aVal << "\n";
      mC0 = mC0 + aDelta;
}



};  // ===================  NS_CHKBRD_TARGET_EXTR
};  // ===================  MMVII
