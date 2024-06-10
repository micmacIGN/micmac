#include "CodedTarget.h"
#include "MMVII_2Include_Serial_Tpl.h"
#include "MMVII_Tpl_Images.h"
#include "MMVII_Interpolators.h"
#include "MMVII_Linear2DFiltering.h"
#include "MMVII_ImageMorphoMath.h"
#include "MMVII_2Include_Tiling.h"
#include "MMVII_Sensor.h"
#include "MMVII_HeuristikOpt.h"
#include "MMVII_ExtractLines.h"
#include "MMVII_TplImage_PtsFromValue.h"


namespace MMVII
{


class cOptimPosSeg : public tFunc2DReal
{
       public :
            cOptimPosSeg(const tSeg2dr &);
	    cPt1dr Value(const cPt2dr &) const override;
            virtual tREAL8 CostOfSeg(const cSegment2DCompiled<tREAL8> &) const = 0;

	    cSegment2DCompiled<tREAL8>  ModifiedSeg(const cPt2dr & aPModif) const;

	    tSeg2dr OptimizeSeg(tREAL8 aStepInit,tREAL8 aStepLim,bool IsMin,tREAL8 aMaxDInfInit) const;

       private :
	    cSegment2DCompiled<tREAL8>  mSegInit;
	    cPt2dr                      mP0Loc;
	    cPt2dr                      mP1Loc;
};


cOptimPosSeg::cOptimPosSeg(const tSeg2dr & aSeg0) :
    mSegInit   (aSeg0),
    mP0Loc  (mSegInit.ToCoordLoc(mSegInit.P1())),
    mP1Loc  (mSegInit.ToCoordLoc(mSegInit.P2()))
{
}

cSegment2DCompiled<tREAL8>  cOptimPosSeg::ModifiedSeg(const cPt2dr & aPModif) const
{
    cPt2dr aP0Glob = mSegInit.FromCoordLoc(mP0Loc+cPt2dr(0,aPModif.x()));
    cPt2dr aP1Glob = mSegInit.FromCoordLoc(mP1Loc+cPt2dr(0,aPModif.y()));

    return cSegment2DCompiled<tREAL8>(aP0Glob,aP1Glob);
}

cPt1dr cOptimPosSeg::Value(const cPt2dr & aPt) const 
{
	return cPt1dr(CostOfSeg(ModifiedSeg(aPt)));
}

tSeg2dr cOptimPosSeg::OptimizeSeg(tREAL8 aStepInit,tREAL8 aStepLim,bool IsMin,tREAL8 aMaxDInfInit) const
{
    cOptimByStep<2>  aOpt(*this,true,IsMin,aMaxDInfInit);
    auto [aScore,aSol] = aOpt.Optim(cPt2dr(0,0),aStepInit,aStepLim);

    return ModifiedSeg(aSol);
}

class cOptimSeg_ValueIm : public cOptimPosSeg 
{
      public :
         cOptimSeg_ValueIm(const tSeg2dr &,tREAL8 aStepOnSeg,const cDataIm2D<tREAL4> & aDIm,tREAL8 aTargetValue);

         tREAL8 CostOfSeg(const cSegment2DCompiled<tREAL8> &) const override;
      private :
	 tREAL8                    mStepOnSeg;
	 int                       mNbOnSeg;
         const cDataIm2D<tREAL4> & mDataIm;
	 tREAL8                    mTargetValue;
};

cOptimSeg_ValueIm::cOptimSeg_ValueIm
(
     const tSeg2dr & aSegInit,
     tREAL8 aStepOnSeg,
     const cDataIm2D<tREAL4> & aDIm,
     tREAL8 aTargetValue
)  :
     cOptimPosSeg  (aSegInit),
     mStepOnSeg    (aStepOnSeg),
     mNbOnSeg      (round_up( Norm2(aSegInit.V12()) / mStepOnSeg )) ,
     mDataIm       (aDIm),
     mTargetValue  (aTargetValue)
{
}

tREAL8 cOptimSeg_ValueIm::CostOfSeg(const cSegment2DCompiled<tREAL8> & aSeg) const
{
     tREAL8 aSum=0;

     for (int aK=0 ; aK<= mNbOnSeg ; aK++)
     {
          cPt2dr aPt = Centroid(aK/double(mNbOnSeg),aSeg.P1(),aSeg.P2());
	  aSum += std::abs(mDataIm.GetVBL(aPt) - mTargetValue);
     }

     return aSum / mNbOnSeg;
}





static constexpr tU_INT1 eNone = 0 ;
static constexpr tU_INT1 eTopo0  = 1 ;
static constexpr tU_INT1 eTopoTmpCC  = 2 ;
static constexpr tU_INT1 eTopoMaxOfCC  = 3 ;
static constexpr tU_INT1 eTopoMaxLoc  = 4 ;
static constexpr tU_INT1 eFilterSym  = 5 ;
static constexpr tU_INT1 eFilterRadiom  = 6 ;

/// candidate that are pre-selected on the sadle-point criteria
class cCdSadle
{
    public :
        cCdSadle (const cPt2dr & aC,tREAL8 aCrit) : mC (aC) , mSadCrit (aCrit) {}
        cCdSadle (){}
        cPt2dr mC;

	///  Criterion of sadle point, obtain by fiting a quadric on gray level
        tREAL8 mSadCrit;
};

/// candidate that are pre-selected after symetry criterion
class cCdSym : public cCdSadle
{
    public :
        cCdSym(const cCdSadle &  aCdSad,tREAL8 aCrit) : cCdSadle  (aCdSad), mSymCrit  (aCrit) { }
	/// Criterion of symetry, obtain after optimisation of center
	tREAL8   mSymCrit;

};

/// candid obtain after radiometric modelization, 

class cCdRadiom : public cCdSym
{
      public :
          cCdRadiom(const cCdSym &,const cDataIm2D<tREAL4> & aDIm,tREAL8 aTeta1,tREAL8 aTeta2,tREAL8 aThickness);
	  cMatIner2Var<tREAL8> StatGray(const cDataIm2D<tREAL4> & aDIm,tREAL8 aThickness,bool IncludeSegs);

	  /// Once blac/white & teta are computed, refine seg using 
	  void OptimSegIm(const cDataIm2D<tREAL4> & aDIm,tREAL8 aLength);

//  cOptimSeg_ValueIm(const tSeg2dr &,tREAL8 aStepOnSeg,const cDataIm2D<tREAL8> & aDIm,tREAL8 aTargetValue);
	  tREAL8  mTetas[2];

	  tREAL8  mCostCorrel;  // 1-Correlation of model
	  tREAL8  mRatioBW;  // ratio min/max of BW
          tREAL8  mBlack;
          tREAL8  mWhite;
};
				
enum class eTPosCB
{
      eUndef,
      eInsideBlack,
      eInsideWhite,
      eBorderLeft,
      eBorderRight

};

inline bool IsInside(eTPosCB aState) {return (aState==eTPosCB::eInsideBlack)  ||   (aState==eTPosCB::eInsideWhite) ;}
inline bool IsOk(eTPosCB aState) {return aState!=eTPosCB::eUndef;}

///  Used temporary for compilation of radiom
class cCdRadiomCompiled : public cCdRadiom
{
	public :
          cCdRadiomCompiled(const cCdRadiom &,tREAL8 aThickness);

	  /// Theoreticall radiom of modelize checkboard + bool if was computed
	  std::pair<eTPosCB,tREAL8>  TheorRadiom(const cPt2dr &) const;

	  tREAL8                     mThickness;
          cSegment2DCompiled<tREAL8> mSeg0 ;
          cSegment2DCompiled<tREAL8> mSeg1 ;
};


cCdRadiomCompiled::cCdRadiomCompiled(const cCdRadiom & aCDR,tREAL8 aThickness) :
    cCdRadiom   (aCDR),
    mThickness  (aThickness),
    mSeg0       (mC,mC+FromPolar(1.0,mTetas[0])),
    mSeg1       (mC,mC+FromPolar(1.0,mTetas[1]))
{
}

std::pair<eTPosCB,tREAL8>  cCdRadiomCompiled::TheorRadiom(const cPt2dr &aPt) const
{
    eTPosCB aPos = eTPosCB::eUndef;
    tREAL8 aGrayTh = -1;

    // we compute locacl coordinates because the sign of y indicate if we are left/right of the oriented segment
    // and sign of x indicate if we are before/after the centre
    cPt2dr aLoc0 = mSeg0.ToCoordLoc(aPt);
    tREAL8  aY0 = aLoc0.y();

    cPt2dr aLoc1 = mSeg1.ToCoordLoc(aPt);
    tREAL8  aY1 = aLoc1.y();

    // compute if we are far enough of S0/S1 because the computation of gray will change
    //  black/white if far  enough, else interpolation
    bool FarS0 = std::abs(aY0)> mThickness; 
    bool FarS1 = std::abs(aY1)> mThickness;

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
         aGrayTh = (mThickness+aSignX*aY0) / (2.0*mThickness);
    }
    else if  (FarS0 && (!FarS1))
    {
	 aPos = eTPosCB::eBorderLeft;
	 int aSignX = (aLoc1.x() <0) ? -1 : 1;
	 aGrayTh = (mThickness+aSignX*aY1) / (2.0 * mThickness);
    }

    return std::pair<eTPosCB,tREAL8>(aPos,aGrayTh);
}

cCdRadiom::cCdRadiom(const cCdSym & aCdSym,const cDataIm2D<tREAL4> & aDIm,tREAL8 aTeta0,tREAL8 aTeta1,tREAL8 aThickness) :
       cCdSym      (aCdSym),
       mTetas      {aTeta0,aTeta1},
       mCostCorrel (2.001),   // over maximal theoreticall value
       mRatioBW    (0)
{
    static int aCpt=0 ; aCpt++;


    cSegment2DCompiled aSeg0 (mC,mC+FromPolar(1.0,mTetas[0]));
    cSegment2DCompiled aSeg1 (mC,mC+FromPolar(1.0,mTetas[1]));
    static std::vector<cPt2di>  aDisk = VectOfRadius(0.0,5);
    cStdStatRes aW0;
    cStdStatRes aW1;

    int aNbIn0=0,aNbIn1=0;

    cMatIner2Var<tREAL8> aCorGrayAll;
    cMatIner2Var<tREAL8> aCorGrayInside;

    cCdRadiomCompiled  aCRC(*this,aThickness);

    for (const auto & aDelta : aDisk)
    {
        cPt2di aPImI = aDelta + ToI(mC);
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

    mRatioBW = std::min(aNbIn0,aNbIn1) / (tREAL8) std::max(aNbIn0,aNbIn1);
    if (mRatioBW <0.05)
    {
       return ;
    }

     mCostCorrel = 1-aCorGrayAll.Correl();
     auto [a,b] = aCorGrayInside.FitLineDirect();
     mBlack = b ;
     mWhite = a+b;
}

void cCdRadiom::OptimSegIm(const cDataIm2D<tREAL4> & aDIm,tREAL8 aLength)
{
     std::vector<cSegment2DCompiled<tREAL8>> aVSegOpt;
     for (int aKTeta=0 ; aKTeta<2 ; aKTeta++)
     {
         cPt2dr aTgt = FromPolar(aLength,mTetas[aKTeta]);
         tSeg2dr aSegInit(mC-aTgt,mC+aTgt);
         cOptimSeg_ValueIm  aOSVI(aSegInit,0.5,aDIm,(mBlack+mWhite)/2.0);
	 tSeg2dr  aSegOpt = aOSVI.OptimizeSeg(0.5,0.01,true,2.0);

	 aVSegOpt.push_back(aSegOpt);
	 mTetas[aKTeta] = Teta(aSegOpt.V12());
	 // mTetas[aKTeta] = aSegOpt.I//
     }

     cPt2dr aC = aVSegOpt.at(0).InterSeg(aVSegOpt.at(1));

     mC = aC;
     cScoreTetaLine::NormalizeTeta(mTetas);
}

/*  *********************************************************** */
/*                                                              */
/*              cAppliCheckBoardTargetExtract                   */
/*                                                              */
/*  *********************************************************** */

class cScoreTetaLine;

class cAppliCheckBoardTargetExtract : public cMMVII_Appli
{
     public :
        typedef tREAL4            tElem;
        typedef cIm2D<tElem>      tIm;
        typedef cDataIm2D<tElem>  tDIm;
        typedef cAffin2D<tREAL8>  tAffMap;


        cAppliCheckBoardTargetExtract(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);

     private :
        // =========== overridding cMMVII_Appli::methods ============
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;

        bool IsPtTest(const cPt2dr & aPt) const;  ///< Is it a point marqed a test


	void DoOneImage() ;
	void MakeImageSaddlePoints(const tDIm &,const cDataIm2D<tU_INT1> & aDMasq) const;

	cPhotogrammetricProject     mPhProj;
	cTimerSegm                  mTimeSegm;

        cCdRadiom TestBinarization(cScoreTetaLine&,const cCdSym &,tREAL8 aThickness);

        // =========== Mandatory args ============

	std::string mNameIm;       ///< Name of background image
	std::string mNameSpecif;   ///< Name of specification file

        // =========== Optionnal args ============

                //  --

	tREAL8            mThickness;  ///<  used for fine estimation of radiom
        bool              mOptimSegByRadiom;  ///< Do we optimize the segment on average radiom     

        // =========== Internal param ============
        tIm                   mImIn;        ///< Input global image
        cPt2di                mSzIm;        ///< Size of image
	tDIm *                mDImIn;       ///< Data input image 
	bool                  mHasMasqTest; ///< Do we have a test image 4 debuf (with masq)
	cIm2D<tU_INT1>        mMasqTest;    ///< Possible image of mas 4 debug, print info ...
        cIm2D<tU_INT1>        mImLabel;     ///< Image storing labels of centers
	cDataIm2D<tU_INT1> *  mDImLabel;    ///< Data Image of label
};


/* *************************************************** */
/*                                                     */
/*              cAppliCheckBoardTargetExtract                  */
/*                                                     */
/* *************************************************** */

cAppliCheckBoardTargetExtract::cAppliCheckBoardTargetExtract(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli     (aVArgs,aSpec),
   mPhProj          (*this),
   mTimeSegm        (this),
   mThickness       (1.0),
   mOptimSegByRadiom (false),
   mImIn            (cPt2di(1,1)),
   mDImIn           (nullptr),
   mHasMasqTest     (false),
   mMasqTest        (cPt2di(1,1)),
   mImLabel         (cPt2di(1,1)),
   mDImLabel        (nullptr)
{
}



cCollecSpecArg2007 & cAppliCheckBoardTargetExtract::ArgObl(cCollecSpecArg2007 & anArgObl)
{
   return  anArgObl
             <<   Arg2007(mNameIm,"Name of image (first one)",{{eTA2007::MPatFile,"0"},eTA2007::FileImage})
             <<   Arg2007(mNameSpecif,"Name of target file")
   ;
}


cCollecSpecArg2007 & cAppliCheckBoardTargetExtract::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return
	        anArgOpt
             <<  mPhProj.DPMask().ArgDirInOpt("TestMask","Mask for selecting point used in detailed mesg/output")
             <<  AOpt2007(mThickness,"Thickness","Thickness for modelizaing line-blur in fine radiom model",{eTA2007::HDV})
             <<  AOpt2007(mOptimSegByRadiom,"OSBR","Optimize segement by radiometry",{eTA2007::HDV})
   ;
}


void cAppliCheckBoardTargetExtract::MakeImageSaddlePoints(const tDIm & aDIm,const cDataIm2D<tU_INT1> & aDMasq) const
{
    cRGBImage  aRGB = RGBImFromGray<tElem>(aDIm);

    for (const auto & aPix : cRect2(aDIm.Dilate(-1)))
    {
       if (aDMasq.GetV(aPix) >= (int) eTopoMaxLoc)
       {
          cPt3di  aCoul = cRGBImage::Yellow;
	  if (aDMasq.GetV(aPix)== eFilterSym) aCoul = cRGBImage::Green;
	  if (aDMasq.GetV(aPix)== eFilterRadiom) aCoul = cRGBImage::Red;
          aRGB.SetRGBPix(aPix,aCoul);
       }
    }
    aRGB.ToFile("Saddles.tif");
}

bool cAppliCheckBoardTargetExtract::IsPtTest(const cPt2dr & aPt) const
{
   return mHasMasqTest && (mMasqTest.DIm().GetV(ToI(aPt)) != 0);
}



cCdRadiom cAppliCheckBoardTargetExtract::TestBinarization(cScoreTetaLine & aSTL,const cCdSym & aCdSym,tREAL8 aThickness)
{

    auto [aTeta0,aTeta1] = aSTL.Tetas_CheckBoard(aCdSym.mC,0.5,1e-3);

    cCdRadiom aCdRadiom(aCdSym,*mDImIn,aTeta0,aTeta1,aThickness);

    if (mOptimSegByRadiom)
    {
       aCdRadiom.OptimSegIm(*(aSTL.DIm()),aSTL.Length());
    }

    if (IsPtTest(aCdSym.mC))
    {


          static int aCpt=0 ; aCpt++;
          StdOut() << " CPT=" << aCpt << "  Corrrr=" <<  aCdRadiom.mCostCorrel 
                   << " Ratio=" <<  aCdRadiom.mRatioBW
		  << " V0="<< aCdRadiom.mBlack << " V1=" << aCdRadiom.mWhite << "\n";
         // StdOut() << " CPT=" << aCpt << " TETASRef= " << aTeta0 << " " << aTeta1 << "\n";

	  int aZoom = 9;
	  cPt2di aSz(50,50);
	  cPt2di aDec =  ToI(aCdSym.mC) - aSz/2;
	  cPt2dr aCLoc = aCdRadiom.mC-ToR(aDec);

	  cRGBImage  aIm = cRGBImage:: FromFile(mNameIm,cBox2di(aDec,aDec+aSz),aZoom);
	  aIm.ResetGray();

	  if (0)
	  {
               cCdRadiomCompiled aCRC(aCdRadiom,2.0);
               for (const auto & aPix :  aIm.ImR().DIm())
               {
                    cPt2dr aPtR = ToR(aPix+aDec);
                    auto [aState,aGray] = aCRC.TheorRadiom(aPtR);
		    if (aState != eTPosCB::eUndef)
		    {
                        aIm.SetGrayPix(aPix,aGray*255);
		    }
               }
	  }

	  int aKT=0;
	  for (const auto & aTeta  : aCdRadiom.mTetas)
	  {
              for (int aK= -aZoom * 20 ; aK<=aZoom*20 ; aK++)
	      {
		  tREAL8 aAbsc= aK/ (2.0 * aZoom);
		  cPt2dr aPt = aCLoc + FromPolar(aAbsc,aTeta);
		  if (aK==6*aZoom)
	             aIm.DrawCircle((aKT==0) ?  cRGBImage::Blue : cRGBImage::Red,aPt,0.5);
	                 // aIm.SetRGBPoint(aPt, (aKT==0) ?  cRGBImage::Blue : cRGBImage::Red);
		  tREAL8 aSign = ((aAbsc>0) ? 1.0 : -1.0) * ((aKT==0) ? 1 : -1) ;
		  cPt2dr aNorm = FromPolar(1.0,aTeta + M_PI/2.0) * aSign;
	          aIm.SetRGBPoint(aPt,cRGBImage::Yellow);
		  if (std::abs(aAbsc) > 1.0)
		  {
                      tREAL8 aV = (aCdRadiom.mBlack+aCdRadiom.mWhite) /2.0;

                      cGetPts_ImInterp_FromValue<tREAL4> aGIFV(*mDImIn,aV,0.1,aPt+ToR(aDec)-aNorm, aNorm);
		      if (aGIFV.Ok())
		      {
	                  aIm.SetRGBPoint(aGIFV.PRes()-ToR(aDec),cRGBImage::Blue);
                      }
		      // StdOut() << "OKKK " << aGIFV.Ok()  << " K=" << aK << "\n";
		  }
	          // aIm.SetRGBPoint(aPt,cRGBImage::Green);
	      }
	      aKT++;
	  }

	  aIm.SetRGBPoint(aCLoc,cRGBImage::Red);
          aIm.ToFile("TestCenter_" + ToStr(aCpt) + ".tif");
// getchar();
    }

    return aCdRadiom;
}

void cAppliCheckBoardTargetExtract::DoOneImage() 
{ 
    int   mNbBlur1 = 4;  // Number of iteration of initial blurring
    tREAL8 mDistMaxLocSad = 10.0;  // for supressing sadle-point,  not max loc in a neighboorhoud
    int    mMaxNbMLS = 2000; //  Max number of point in best saddle points
    tREAL8 aRayCalcSadle = sqrt(4+1);  // limit point 2,1

    tREAL8 mThresholdSym     = 0.50;  // threshlod for symetry criteria
    tREAL8 mDistCalcSym0     = 8.0;   // distance for evaluating symetry criteria
    tREAL8 mDistDivSym       = 2.0;   // maximal distance to initial value in symetry opt
    
    tREAL8 mLengtSInit = 05.0;
    tREAL8 mStepSeg    = 0.5;
    tREAL8 mMaxCostCorrIm  = 0.1;

    //   computed threshold
    tINT8  mDistRectInt = 20; // to see later how we compute it


    /* [0]    Initialise : read image and mask */

    cAutoTimerSegm aTSInit(mTimeSegm,"Init");

	// [0.0]   read image
    mImIn =  tIm::FromFile(mNameIm);
    mDImIn = &mImIn.DIm() ;
    mSzIm = mDImIn->Sz();
    cRect2 aRectInt = mDImIn->Dilate(-mDistRectInt);

	// [0.1]   initialize labeling image 
    //mImLabel(mSzIm,nullptr,eModeInitImage::eMIA_Null);
    mDImLabel =  &(mImLabel.DIm());
    mDImLabel->Resize(mSzIm);
    mDImLabel->InitCste(eNone);


    mHasMasqTest = mPhProj.ImageHasMask(mNameIm);
    if (mHasMasqTest)
       mMasqTest =  mPhProj.MaskOfImage(mNameIm,*mDImIn);


    /* [1]   Compute a blurred image => less noise, less low level saddle */

    cAutoTimerSegm aTSBlur(mTimeSegm,"Blurr");

    tIm   aImBlur  = mImIn.Dup(); // create image blurred with less noise
    tDIm& aDImBlur = aImBlur.DIm();

    SquareAvgFilter(aDImBlur,mNbBlur1,1,1);



    /* [2]  Compute "topological" saddle point */

    cAutoTimerSegm aTSTopoSad(mTimeSegm,"TopoSad");

         // 2.1  point with criteria on conexity of point > in neighoor

    int aNbSaddle=0;
    int aNbTot=0;

    for (const auto & aPix : aRectInt)
    {
        if (FlagSup8Neigh(aDImBlur,aPix).NbConComp() >=4)
	{
            mDImLabel->SetV(aPix,eTopo0);
	    aNbSaddle++;
	}
        aNbTot++;
    }

    
         // 2.2  as often there 2 "touching" point with this criteria
	 // select 1 point in conected component

    cAutoTimerSegm aTSMaxCC(mTimeSegm,"MaxCCSad");
    int aNbCCSad=0;
    std::vector<cPt2di>  aVCC;
    const std::vector<cPt2di> & aV8 = Alloc8Neighbourhood();

    for (const auto& aPix : *mDImLabel)
    {
         if (mDImLabel->GetV(aPix)==eTopo0)
	 {
             aNbCCSad++;
             ConnectedComponent(aVCC,*mDImLabel,aV8,aPix,eTopo0,eTopoTmpCC);
	     cWhichMax<cPt2di,tREAL8> aBestPInCC;
	     for (const auto & aPixCC : aVCC)
	     {
                 aBestPInCC.Add(aPixCC,CriterionTopoSadle(aDImBlur,aPixCC));
	     }

	     cPt2di aPCC = aBestPInCC.IndexExtre();
	     mDImLabel->SetV(aPCC,eTopoMaxOfCC);
	 }
    }

    /* [3]  Compute point that are max local */


    std::vector<cCdSadle> aVCdtSad;
    cAutoTimerSegm aTSCritSad(mTimeSegm,"CritSad");

    cCalcSaddle  aCalcSBlur(aRayCalcSadle+0.001,1.0);

       // [3.1]  compute for each point the saddle criteria
    for (const auto& aPix : *mDImLabel)
    {
         if (mDImLabel->GetV(aPix)==eTopoMaxOfCC)
	 {
             tREAL8 aCritS = aCalcSBlur.CalcSaddleCrit(aDImBlur,aPix);
             aVCdtSad.push_back(cCdSadle(ToR(aPix),aCritS));
//  cPt3dr(aPix.x(),aPix.y(),aCritS));
	 }
    }
    int aNbSad0 = aVCdtSad.size();

    //   [3.2]  select KBest + MaxLocal
    cAutoTimerSegm aTSMaxLoc(mTimeSegm,"MaxLoc");

    //  sort by decreasing criteria of saddles => "-"  aCdt.mSadCrit
    SortOnCriteria(aVCdtSad,[](const auto & aCdt){return - aCdt.mSadCrit;});
    aVCdtSad = FilterMaxLoc((cPt2dr*)nullptr,aVCdtSad,[](const auto & aCdt) {return aCdt.mC;}, mDistMaxLocSad);
    int aNbSad1 = aVCdtSad.size();

    //  limit the number of point , a bit rough but first experiment show that sadle criterion is almost perfect on good images
    aVCdtSad.resize(std::min(aVCdtSad.size(),size_t(mMaxNbMLS)));

    for (const auto & aCdt : aVCdtSad)
        mDImLabel->SetV(ToI(aCdt.mC),eTopoMaxLoc);

    int aNbSad2 = aVCdtSad.size();
    StdOut() << "END MAXLOC \n";


    /* [4]  Calc Symetry criterion */

    std::vector<cCdSym> aVCdtSym;
    cAutoTimerSegm aTSSym(mTimeSegm,"SYM");
    {
       cCalcSaddle  aCalcSInit(1.5,1.0);
       cFilterDCT<tREAL4> * aFSym = cFilterDCT<tREAL4>::AllocSym(mImIn,0.0,mDistCalcSym0,1.0);
       cOptimByStep aOptimSym(*aFSym,true,mDistDivSym);

       for (auto & aCdtSad : aVCdtSad)
       {
	   auto [aValSym,aNewP] = aOptimSym.Optim(aCdtSad.mC,1.0,0.01);  // Pos Init, Step Init, Step Lim
	   aCdtSad.mC = aNewP;

	   if (aValSym < mThresholdSym)
	   {
	       aVCdtSym.push_back(cCdSym(aCdtSad,aValSym));
               mDImLabel->SetV(ToI(aNewP),eFilterSym);
	   }
       }

       delete aFSym;
    }
    int aNbSym = aVCdtSym.size();

    /* [5]  Compute lines, radiom model & correlation */
    std::vector<cCdRadiom> aVCdtRad;
    cAutoTimerSegm aTSRadiom(mTimeSegm,"Radiom");
    {
        cCubicInterpolator aCubI(-0.5);
        cScoreTetaLine  aSTL(*mDImIn,aCubI,mLengtSInit,mStepSeg);
        for (const auto & aCdtSym : aVCdtSym)
        {
            cCdRadiom aCdRad = TestBinarization(aSTL,aCdtSym,mThickness);
	    if (aCdRad.mCostCorrel <= mMaxCostCorrIm)
	    {
               aVCdtRad.push_back(aCdRad);
	        mDImLabel->SetV(ToI(aCdRad.mC),eFilterRadiom);
	    }
        }
    }



#if (0)


    if (1)
    {
       cFilterDCT<tREAL4> * aFSym = cFilterDCT<tREAL4>::AllocSym(mImIn,0.0,mDistCalcSym0,1.0);
       cOptimByStep aOptimSym(*aFSym,true,mDistDivSym);
// tPtR Optim(const tPtR & ,tREAL8 aStepInit,tREAL8 aStepLim,tREAL8 aMul=0.5);

       cCubicInterpolator aCubI(-0.5);
       cScoreTetaLine  aSTL(*mDImIn,aCubI,5.0,0.5);

       cStdStatRes aSSad1;
       cStdStatRes aSSad0;
       cStdStatRes aSSymInt_1;
       cStdStatRes aSSymInt_0;

       cStdStatRes aSSCor_0;
       cStdStatRes aSSCor_1;

       cCalcSaddle  aCalcSInit(1.5,1.0);
       for (const auto & aP3 : aVMaxLoc)
       {
	   cPt2dr aP0 = Proj(aP3);
           cPt2dr aP1 =  aP0;
	   aCalcSBlur.RefineSadlePointFromIm(aImBlur,aP1,true);
           cPt2dr aP2 =  aP1;
	   aCalcSInit.RefineSadlePointFromIm(aImBlur,aP2,true);


	   tREAL8 aSymInt = aOptimSym.Optim(aP0,1.0,0.05).first;
           tREAL8 aCorBin = TestBinarization(aSTL,Proj(aP3));

           bool  Ok = IsPtTest(Proj(aP3));

	   //StdOut() << "SYMM=" << aFSym->ComputeVal(aP0) << "\n";

	   if (Ok)
	   {
               aSSad1.Add(aP3.z());
	       aSSymInt_1.Add(aSymInt);
	       aSSCor_1.Add(aCorBin);
	   }
	   else
	   {
               aSSad0.Add(aP3.z());
	       aSSymInt_0.Add(aSymInt);
	       aSSCor_0.Add(aCorBin);
	   }
       }

       if (mHasMasqTest)
       {
          StdOut()  << " ================ STAT LOC CRITERIA ==================== \n";
          StdOut() << " * Saddle , #Ok  Min=" << aSSad1.Min()  
		   << "     #NotOk 90% " << aSSad0.ErrAtProp(0.9) 
		   << "   99%  " << aSSad0.ErrAtProp(0.99) 
		   << "   99.9%  " << aSSad0.ErrAtProp(0.999) 
		   << "\n";

          StdOut() << " * SYM , #Ok=" << aSSymInt_1.Max()   << " %75=" <<  aSSymInt_1.ErrAtProp(0.75)
		   << "  NotOk 50% " << aSSymInt_0.ErrAtProp(0.5) 
		   << "    10%  "    << aSSymInt_0.ErrAtProp(0.10) 
		   << "\n";

          StdOut() << " * CORR , #Max=" << aSSCor_1.Max()   << " %75=" <<  aSSCor_1.ErrAtProp(0.75)
		   << "  NotOk 50% " << aSSCor_0.ErrAtProp(0.5) 
		   << "    10%  "    << aSSCor_0.ErrAtProp(0.10) 
		   << "\n";
       }
       delete aFSym;
    }

    cAutoTimerSegm aTSMakeIm(mTimeSegm,"OTHERS");

    StdOut() << "NBS " << (100.0*aNbSaddle)/aNbTot << " " <<  (100.0*aNbCCSad)/aNbTot 
	    << " " <<  (100.0*aVMaxLoc.size())/aNbTot  << " NB=" << aVMaxLoc.size() << "\n";
    aDImBlur.ToFile("Blurred.tif");

#endif
    cAutoTimerSegm aTSMakeIm(mTimeSegm,"OTHERS");
    MakeImageSaddlePoints(*mDImIn,*mDImLabel);

    StdOut()  << "NB Cd,  SAD: " << aNbSad0 << " " << aNbSad1 <<  " " <<aNbSad2 
	      << " SYM:" << aNbSym <<  " Radiom:" << aVCdtRad.size() << "\n";
}

/*
 *  Dist= sqrt(5)
 *  T=6.3751
 *  2 sqrt->6.42483
 */


int  cAppliCheckBoardTargetExtract::Exe()
{
   mPhProj.FinishInit();

   if (RunMultiSet(0,0))
   {
       return ResultMultiSet();
   }

   DoOneImage();

   return EXIT_SUCCESS;
}

/* =============================================== */
/*                                                 */
/*                       ::                        */
/*                                                 */
/* =============================================== */

tMMVII_UnikPApli Alloc_CheckBoardCodedTarget(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppliCheckBoardTargetExtract(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpecExtractCheckBoardTarget
(
     "CodedTargetCheckBoardExtract",
      Alloc_CheckBoardCodedTarget,
      "Extract coded target from images",
      {eApF::CodedTarget,eApF::ImProc},
      {eApDT::Image,eApDT::Xml},
      {eApDT::Image,eApDT::Xml},
      __FILE__
);


};
