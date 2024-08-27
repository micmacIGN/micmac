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

extern bool DebugEll;
bool DebugCB = false;

namespace NS_CHKBRD_TARGET_EXTR { 

class cAppliCheckBoardTargetExtract ;

class cCdSadle;
class cCdSym;
class cCdRadiom;
class cTmpCdRadiomPos ;
class cCdEllipse ;

static constexpr tU_INT1 eNone = 0 ;
static constexpr tU_INT1 eTopo0  = 1 ;
static constexpr tU_INT1 eTopoTmpCC  = 2 ;
static constexpr tU_INT1 eTopoMaxOfCC  = 3 ;
static constexpr tU_INT1 eTopoMaxLoc  = 4 ;
static constexpr tU_INT1 eFilterSym  = 5 ;
static constexpr tU_INT1 eFilterRadiom  = 6 ;
static constexpr tU_INT1 eFilterEllipse  = 7 ;
static constexpr tU_INT1 eFilterCodedTarget  = 8 ;


/* **************************************************** */
/*                     cCdSadle                         */
/* **************************************************** */

/// candidate that are pre-selected on the sadle-point criteria
class cCdSadle
{
    public :
        cCdSadle (const cPt2dr & aC,tREAL8 aCrit,bool isPTest) ;
        cCdSadle ();

	bool Is4Debug() const;

	/// Center 
        cPt2dr mC;

	///  Criterion of sadle point, obtain by fiting a quadric on gray level
        tREAL8 mSadCrit;

	/// Is it a point marked 
        bool mIsPTest;

	///  Num : 4 debug
	int      mNum; 
	/// Use 2 compute mNum
	static int TheCptNum;
	/// Use to have breakpoint at creation
	static int TheNum2Debug;
};
//
/// candidate that are pre-selected after symetry criterion
class cCdSym : public cCdSadle
{
    public :
        cCdSym(const cCdSadle &  aCdSad,tREAL8 aCrit) : cCdSadle  (aCdSad), mSymCrit  (aCrit) { }
	/// Criterion of symetry, obtain after optimisation of center
	tREAL8   mSymCrit;

};

/// candidate obtain after radiometric modelization, 

class cCdRadiom : public cCdSym
{
      public :
          /// Cstr use the 2 direction + Thickness of transition between black & white
          cCdRadiom(const cAppliCheckBoardTargetExtract *,const cCdSym &,const cDataIm2D<tREAL4> & aDIm,tREAL8 aTeta1,tREAL8 aTeta2,tREAL8 aLength,tREAL8 aThickness);

          ///  Theoretical threshold
          tREAL8 Threshold(tREAL8 aWhite= 0.5 ) const ;
	  /// Once black/white & teta are computed, refine seg using 
	  void OptimSegIm(const cDataIm2D<tREAL4> & aDIm,tREAL8 aLength);

	  /// compute an ellipse that "safely" contains the points  of checkboard
          void ComputePtsOfEllipse(std::vector<cPt2di> & aRes,tREAL8 aLength) const;
	  ///  call previous with length
          void ComputePtsOfEllipse(std::vector<cPt2di> & aRes) const;

	  /// compute an ellipse that contain 
          bool FrontBlackCC(std::vector<cPt2di> & aRes,cDataIm2D<tU_INT1> & aMarq,int aNbMax) const;
	  ///  Select point of front that are on ellipse
          void SelEllAndRefineFront(std::vector<cPt2dr> & aRes,const std::vector<cPt2di> &) const;


	  /// Is the possibly the point on arc of black  ellipse
          bool PtIsOnEll(cPt2dr &) const;
	  ///  Is the point on the line for one of angles
          bool PtIsOnLine(const cPt2dr &,tREAL8 aTeta) const;

	  ///  Make a visualisation of geometry
	  void ShowDetail(int aCptMarq,const cScoreTetaLine & aSTL,const std::string &,cDataIm2D<tU_INT1> & aMarq, cFullSpecifTarget *) const;

	  const cAppliCheckBoardTargetExtract * mAppli;
          bool    mIsOk;
	  const cDataIm2D<tREAL4> * mDIm;

	  tREAL8  mTetas[2];
	  tREAL8  mLength;     ///< length of the lines
	  tREAL8  mThickness;  ///< thickness of the transition B/W

	  tREAL8  mCostCorrel;  // 1-Correlation of model
	  tREAL8  mRatioBW;  // ratio min/max of BW
	  tREAL8  mScoreTeta;  // ratio min/max of BW
          tREAL8  mBlack;
          tREAL8  mWhite;
	  cPt2di  mDec;  ///< Shit for visu
};
				

class cCdEllipse : public cCdRadiom
{
	public : 
           cCdEllipse(const cCdRadiom &,cDataIm2D<tU_INT1> & aMarq,int aNbMax,bool isCircle);
	   bool IsOk() const;
	   const cEllipse & Ell() const;
           const cPt2dr &   CornerlEl_WB() const;
           const cPt2dr &   CornerlEl_BW() const;
           cPt2dr  M2I(const cPt2dr & aPMod) const;
           cPt2dr  I2M(const cPt2dr & aPIM) const; 
	   bool  IsCircle() const;


           /** compute the "normalized" length to encoding part*/ 
	   std::pair<tREAL8,cPt2dr>  Length2CodingPart(tREAL8 aPropGray,const cPt2dr & aModCenterBit) const;

	   /** adapted to case where the geometric prediction
	    * is good on direction but relatively bad  on distance
	    */
           void  DecodeByL2CP(tREAL8 aPropGray) ;

	   /// Decode just by reading the value of predicted position of bits
	   const cOneEncoding *  BasicDecode(tREAL8 aWeightWhite);

	   /// Do the decoding, switch to one of the specialized method
	   const cOneEncoding *  Decode(cFullSpecifTarget *,tREAL8 aPropGray);


	   const cOneEncoding * Code() const; ///< Accessor
	   bool  BOutCB() const; ///< Accessor
	   tREAL8  MaxEllD() const;
	   tREAL8  ThrsEllD() const;


           void GenImageFail(const std::string & aWhyFail);

	private : 

	   /// Most basic method, return minimal of all lenght
           tREAL8 ComputeThresholdsMin(const std::vector<std::pair<int,tREAL8>> & aVBR) const;
	   /** "prefered" method, compute the minimal lenght under the constraint to have run length of white
	       over the value given specificiation */
           tREAL8 ComputeThresholdsByRLE(const std::vector<std::pair<int,tREAL8>> & aVBR) const;


           void AssertOk() const;

           const cFullSpecifTarget *  mSpec;
	   cEllipse             mEll;
	   cPt2dr               mCornerlEl_WB;
	   cPt2dr               mCornerlEl_BW;
	   cAff2D_r             mAffIm2Mod;
	   tREAL8               mMaxEllD; /// Maximal distance 2 ellipse (for frontier point, not on lines)
	   const cOneEncoding * mCode;
	   bool                 mBOutCB;  /// Is there black point outside the  check board
           bool                 mIsCircle;  ///< Was it obtained enforcing a circle
};

class cCdMerged : public  cCdEllipse
{
	public :
            cCdMerged(const cDataIm2D<tREAL4> *,const cCdEllipse & aCDE,tREAL8 aScale) ;

	    tREAL8 mScale;
	    cPt2dr mC0;  // center at initial image scale
            void  OptimizePosition(const cInterpolator1D &);

	    const cDataIm2D<tREAL4> * mDIm0;
};


class cOptimPosCdM : public cDataMapping<tREAL8,2,1>
{
	public :
           cOptimPosCdM(const cCdMerged & aCdM,const cInterpolator1D & );

           cPt1dr Value(const cPt2dr & ) const override;
	   typedef cSegment2DCompiled<tREAL8> tSeg;

	private :
	    void AddPts(const cPt2dr & aMaster,  const cPt2dr & aSecond,bool toAvoid2);

            const cCdMerged&        mCdM;
	    const cInterpolator1D & mCurInt;
	    std::vector<cPt2dr>     mPtsOpt;
};



void  cCdMerged::OptimizePosition(const cInterpolator1D & anInt)
{
     cOptimPosCdM aCdtOpt(*this,anInt);
     cOptimByStep anOpt(aCdtOpt,true,1.0);
     auto [aVal,aDelta] =   anOpt.Optim(cPt2dr(0,0),0.02,0.001);

     mC0 = mC0 + aDelta;
}

cPt1dr cOptimPosCdM::Value(const cPt2dr & aDelta ) const 
{
     cSymMeasure<tREAL8> aSymM; //
     cPt2dr a2NewC = (mCdM.mC0 + aDelta) * 2.0;

    const cDataIm2D<tREAL4> & aDIm = *(mCdM.mDIm0);
     for (const auto & aP1 : mPtsOpt)
     {
          cPt2dr aP2 = a2NewC - aP1;
	  if (aDIm.InsideInterpolator(mCurInt,aP1) && aDIm.InsideInterpolator(mCurInt,aP2))
             aSymM.Add(aDIm.GetValueInterpol(mCurInt,aP1),aDIm.GetValueInterpol(mCurInt,aP2));
     }

     return cPt1dr(aSymM.Sym(1e-5));
}


cOptimPosCdM::cOptimPosCdM(const cCdMerged & aCdM,const cInterpolator1D & aInt)  :
	mCdM      (aCdM),
	mCurInt   (aInt)
{
	AddPts(mCdM.CornerlEl_WB(), mCdM.CornerlEl_BW(),true);
	AddPts(mCdM.CornerlEl_BW(), mCdM.CornerlEl_WB(),false);
}

void cOptimPosCdM::AddPts(const cPt2dr & aSCorn1, const cPt2dr & aSCorn2,bool toAvoid2)
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

/*
*/

/*
	private :
	    std::vector<cPt2dr>  mPtsOpt;
	    */

//void cCdMerged::



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

///  Used temporary for theoreticall radiometric model compilation of radiom
class cTmpCdRadiomPos : public cCdRadiom
{
	public :
          cTmpCdRadiomPos(const cCdRadiom &,tREAL8 aThickness);

	  /// Theoreticall radiom of modelize checkboard + bool if was computed
	  std::pair<eTPosCB,tREAL8>  TheorRadiom(const cPt2dr &,tREAL8 aThick,tREAL8 aSteep) const;
	  std::pair<eTPosCB,tREAL8>  TheorRadiom(const cPt2dr &) const;

	  tREAL8                     mThickness;
          cSegment2DCompiled<tREAL8> mSeg0 ;
          cSegment2DCompiled<tREAL8> mSeg1 ;
};


/* *************************************************** */
/*                                                     */
/*              cAppliCheckBoardTargetExtract          */
/*                                                     */
/* *************************************************** */

class cAppliCheckBoardTargetExtract : public cMMVII_Appli
{
     public :
        typedef tREAL4            tElem;
        typedef cIm2D<tElem>      tIm;
        typedef cDataIm2D<tElem>  tDIm;
        typedef cAffin2D<tREAL8>  tAffMap;


        cAppliCheckBoardTargetExtract(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);


	///  Generate a small image, centred on decteted "cCdRadiom", showing the main carateristic detecte
        cRGBImage  GenImaRadiom(cCdRadiom &,int aSz) const;
	/// Call "GenImaRadiom" with default size
        cRGBImage  GenImaRadiom(cCdRadiom &) const;
	///  For generating visualizatio,
        std::string NameVisu(const std::string & aPref,const std::string aPost="") const;

	///  Add the information specfic to a "cCdEllipse" 
        void       ComplImaEllipse(cRGBImage &,const  cCdEllipse &) const;


        const cFullSpecifTarget *  Specif() const {return mSpecif;}    ///< Accessor
        int  NbMinPtEllipse() const {return mNbMinPtEllipse;}    ///< Accessor
     private :

	/// Memorize a detection as a label, if label image is init
	void SetLabel(const cPt2dr& aPt,tU_INT1 aLabel);

        // =========== overridding cMMVII_Appli::methods ============
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;

        void GenerateVisuFinal() const;
        void GenerateVisuDetail(std::vector<cCdEllipse> &) const;
        bool IsPtTest(const cPt2dr & aPt) const;  ///< Is it a point marqed a test

        /// Potentially Add a new detected 
	void  AddCdtE(const cCdEllipse & aCDE);

	/// Generate the xml-result
	void DoExport();


	/// Method called do each image
	void DoOneImage() ;
	/// Method called do each image
	void DoOneImageAndScale(tREAL8,const  tIm & anIm ) ;
	
	    ///  Read Image from files, init Label Image, init eventually masq 4 debug, compute blurred version of input image
           void ReadImagesAndBlurr();
	   /// compute points that are "topologicall" : memorized in  label image as  label "eTopoTmpCC"
           void ComputeTopoSadles();
	   /// start from topo point, compute the "saddle" criterion, and use it for filtering on relative  max
           void SaddleCritFiler() ;
	   /// start from saddle point, optimize position on symetry criterion then filter on sym thresholds
           void SymetryFiler() ;

	void MakeImageLabels(const std::string & aName,const tDIm &,const cDataIm2D<tU_INT1> & aDMasq) const;

	cPhotogrammetricProject     mPhProj;
	cTimerSegm                  mTimeSegm;

        cCdRadiom MakeCdtRadiom(cScoreTetaLine&,const cCdSym &,tREAL8 aThickness);

        // =========== Mandatory args ============

	std::string mNameIm;       ///< Name of background image
	std::string mNameSpecif;   ///< Name of specification file

        // =========== Optionnal args ============

                //  --

	tREAL8            mThickness;  ///<  used for fine estimation of radiom
        bool              mOptimSegByRadiom;  ///< Do we optimize the segment on average radiom     

        tREAL8            mLInitTeta;      ///<  = 05.0;
        tREAL8            mLInitProl;      ///<  = 03.0;
        tREAL8            mLengtProlong;    ///<  = 20.0;
        tREAL8            mStepSeg;         ///<  = 0.5;
        tREAL8            mMaxCostCorrIm;   ///<  = 0.1;
        int               mNbMaxBlackCB;    ///< Number max of point in black component of checkboard
        tREAL8            mPropGrayDCD;     ///< Proportion Black/White for extracting 
	int               mNbBlur1;         ///< = 4,  Number of initial blurring
	std::string       mStrShow;

	std::vector<tREAL8> mScales;        ///<  Different Scales at which computation is done def {1}, => 0.5 means biggers images
					    //
        // ---------------- Thresholds for Saddle point criteria --------------------
        tREAL8            mDistMaxLocSad ;  ///< =10.0, for supressing sadle-point,  not max loc in a neighboorhoud
        int               mDistRectInt;     ///< = 20,  insideness of points  for seed detection
        size_t            mMaxNbSP_ML0 ;   ///< = 30000  Max number of best point  saddle points, before MaxLoc
        size_t            mMaxNbSP_ML1  ;   ///< = 2000   Max number of best point  saddle points, after  MaxLoc
        cPt2di            mPtLimCalcSadle;  ///< =(2,1)  limit point for calc sadle neighbour , included, 


        // ---------------- Thresholds for Symetry  criteria --------------------
        tREAL8            mThresholdSym  ;  ///< = 0.5,  threshlod for symetry criteria
        tREAL8            mRayCalcSym0  ;  ///< = 8.0  distance for evaluating symetry criteria
        tREAL8            mDistDivSym    ;  ///< = 2.0  maximal distance to initial value in symetry opt

	int                   mNumDebugMT;
	int                   mNumDebugSaddle;

        // ---------------- Thresholds for Ellipse  criteria --------------------
        int                   mNbMinPtEllipse;
	bool                  mTryC;
	bool                  mRefinePos;
	
        // =========== Internal param ============

	int                   mZoomVisuDetec;  /// zoom Visu detail of detection
	int                   mDefSzVisDetec;  /// Default Sz Visu detection 
        cFullSpecifTarget *   mSpecif;

        tIm                   mImInCur;     ///< Input current image
        cPt2di                mSzImCur;     ///< Size of current image
	tDIm *                mDImInCur;    ///< Data ccurrent image 

        tIm                   mImIn0;       ///< Input file image
        cPt2di                mSzIm0;       ///< Size file image
	tDIm *                mDImIn0;      ///< Data input image 

					   
        tIm                   mImBlur;      ///< Blurred image, used in pre-detetction
	tDIm *                mDImBlur;     ///< Data input image 
	bool                  mHasMasqTest; ///< Do we have a test image 4 debuf (with masq)
	cIm2D<tU_INT1>        mMasqTest;    ///< Possible image of mas 4 debug, print info ...
        cIm2D<tU_INT1>        mImLabel;     ///< Image storing labels of centers
	cDataIm2D<tU_INT1> *  mDImLabel;    ///< Data Image of label
        cIm2D<tU_INT1>        mImTmp;       ///< Temporary image for connected components
	cDataIm2D<tU_INT1> *  mDImTmp;      ///< Data Image of "mImTmp"

        std::vector<cCdSadle> mVCdtSad;     ///< Candidate  that are selected as local max of saddle criteria
        std::vector<int>      mNbSads;      ///< For info, number of sadle points at different step
        std::vector<cCdSym>   mVCdtSym;     ///< Candidate that are selected on the symetry criteria
					    //
	std::vector<cCdMerged> mVCdtMerged; // Candidate merged form various scales
	tREAL8                mCurScale;    /// Memorize the current value of the scale
	bool                  mMainScale;   /// Is it the first/main scale 
	cInterpolator1D *     mInterpol;
};




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
    static int aCpt=0 ; aCpt++;

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

bool cCdRadiom::PtIsOnEll(cPt2dr & aPtAbs) const
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

   // cGetPts_ImInterp_FromValue<tREAL4> aGIFV(*mDIm,aV,0.1,aPt+ToR(aDec)-aNorm, aNorm);
    /*
    cPt2dr  aPGr =  Proj(mDIm->GetGradAndVBL(aPtAbs));
    if (IsNull(aPGr)) return false;

    cPt2dr aDir = (aPtAbs-mC) / aPGr;
    tREAL8 aTeta = 
    if (Norm2(aPGr) 

    */
     


    return true;
}

void cCdRadiom::SelEllAndRefineFront(std::vector<cPt2dr> & aRes,const std::vector<cPt2di> & aFrontI) const
{
    aRes.clear();
    for (const auto & aPix : aFrontI)
    {
         cPt2dr aRPix = ToR(aPix);
	 if (PtIsOnEll(aRPix))
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
// DebugEll=true;
	    GenImageFail("BadEll");
// DebugEll=false;
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
	aFlag |= (1<<aVBR.at(aKBit).first);
        aKBit++;
	aRunLE = MaxRunLength(aFlag,1<<mSpec->NbBits()).x();
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


/* *************************************************** */
/*                                                     */
/*              cAppliCheckBoardTargetExtract          */
/*                                                     */
/* *************************************************** */

     /* ------------------------------------------------- */
     /*      METHOD FOR CONSTRUCTION OF OBJECT            */
     /* ------------------------------------------------- */

cAppliCheckBoardTargetExtract::cAppliCheckBoardTargetExtract(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli     (aVArgs,aSpec),
   mPhProj          (*this),
   mTimeSegm        (this),
   mThickness       (1.0),
   mOptimSegByRadiom (false),
   mLInitTeta        (5.0),
   mLInitProl        (3.0),
   mLengtProlong     (20.0),
   mStepSeg          (0.5),
   mMaxCostCorrIm    (0.1),
   mNbMaxBlackCB     (10000),
   mPropGrayDCD      (2.0/3.0),
   mNbBlur1          (4),
   mStrShow          (""),
   mScales           {1.0},
   mDistMaxLocSad    (10.0),
   mDistRectInt      (20),
   mMaxNbSP_ML0      (30000),
   mMaxNbSP_ML1      (2000),
   mPtLimCalcSadle   (2,1),
   mThresholdSym     (0.5),
   mRayCalcSym0      (8.0),
   mDistDivSym       (2.0),
   mNumDebugMT       (-1),
   mNumDebugSaddle   (-1),
   mNbMinPtEllipse   (6),
   mTryC             (true),
   mRefinePos        (true),
   mZoomVisuDetec    (9),
   mDefSzVisDetec    (150),
   mSpecif           (nullptr),
   mImInCur          (cPt2di(1,1)),
   mDImInCur         (nullptr),
   mImIn0            (cPt2di(1,1)),
   mDImIn0           (nullptr),
   mImBlur           (cPt2di(1,1)),
   mDImBlur          (nullptr),
   mHasMasqTest      (false),
   mMasqTest         (cPt2di(1,1)),
   mImLabel          (cPt2di(1,1)),
   mDImLabel         (nullptr),
   mImTmp            (cPt2di(1,1)),
   mDImTmp           (nullptr),
   mCurScale         (false),
   mMainScale        (true),
   mInterpol         (nullptr)
{
}



cCollecSpecArg2007 & cAppliCheckBoardTargetExtract::ArgObl(cCollecSpecArg2007 & anArgObl)
{
   return  anArgObl
             <<   Arg2007(mNameIm,"Name of image (first one)",{{eTA2007::MPatFile,"0"},eTA2007::FileImage})
            //  <<   Arg2007(mNameSpecif,"Name of target file")
	     <<   Arg2007(mNameSpecif,"Xml/Json name for bit encoding struct",{{eTA2007::XmlOfTopTag,cFullSpecifTarget::TheMainTag}})

   ;
}


cCollecSpecArg2007 & cAppliCheckBoardTargetExtract::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return
	        anArgOpt
             <<  mPhProj.DPPointsMeasures().ArgDirOutOptWithDef("Std")
             <<  mPhProj.DPMask().ArgDirInOpt("TestMask","Mask for selecting point used in detailed mesg/output")
             <<  AOpt2007(mThickness,"Thickness","Thickness for modelizaing line-blur in fine radiom model",{eTA2007::HDV})
             <<  AOpt2007(mLInitTeta,"LSIT","Length Segment Init, for teta",{eTA2007::HDV})
             <<  AOpt2007(mNbBlur1,"NbB1","Number of blurr with sz1",{eTA2007::HDV})
             <<  AOpt2007(mStrShow,"StrV","String for generate Visu : G-lobal L-abels E-llipse N-ums",{eTA2007::HDV})
             <<  AOpt2007(mRayCalcSym0,"SymRay","Ray arround point for initial computation of symetry",{eTA2007::HDV}) 
             <<  AOpt2007(mLInitProl,"LSIP","Length Segment Init, for prolongation",{eTA2007::HDV})
             <<  AOpt2007(mNbMinPtEllipse,"NbMinPtEl","Number minimal of point for ellipse estimation",{eTA2007::HDV})
             <<  AOpt2007(mTryC,"TryC","Try also circle when ellipse fails",{eTA2007::HDV})
             <<  AOpt2007(mRefinePos,"RefinePos","Refine final position with SinC interpol & over sampling",{eTA2007::HDV})
	     <<  AOpt2007(mScales,"Scales","Diff scales of compute (! 0.5 means bigger)",{eTA2007::HDV})
             <<  AOpt2007(mOptimSegByRadiom,"OSBR","Optimize segement by radiometry",{eTA2007::HDV})
             <<  AOpt2007(mNbMaxBlackCB,"NbMaxBlackCB","Number max of point in black part of check-board ",{eTA2007::HDV})
             <<  AOpt2007(mPropGrayDCD,"PropGrayDCD","Proportion of gray for find coding part",{eTA2007::HDV})
             <<  AOpt2007(mNumDebugMT,"NumDebugMT","Num marq target for debug",{eTA2007::Tuning})
             <<  AOpt2007(mNumDebugSaddle,"NumDebugSaddle","Num Saddle point to debug",{eTA2007::Tuning})

   ;
}

     /* ------------------------------------------------- */
     /*      METHOD FOR VISUALIZATION OF RESULTS          */
     /* ------------------------------------------------- */

	// int                   mZoomVisuDetec;  /// zoom Visu detail of detection
	// int                   mDefSzVisDetec;  /// Default Sz Visu detection 

cRGBImage  cAppliCheckBoardTargetExtract::GenImaRadiom(cCdRadiom & aCdR) const
{
	return GenImaRadiom(aCdR,mDefSzVisDetec);
}

cRGBImage  cAppliCheckBoardTargetExtract::GenImaRadiom(cCdRadiom & aCdR,int aSzI) const
{
    bool  aLocalDyn=false; // if true generate the image with dynamic such "Black->0" , "White->255"
    bool  aTheorGray=false; // if true generate the "theoreticall" gray, inside ellipse
    cPt3di  aCoulCenter   = cRGBImage::Red; 
    cPt3di  aCoulEllFront = cRGBImage::Orange; 
    cPt3di  aCol_StrL = cRGBImage::Yellow;  // color for theoreticall straight line

    cPt2di aSz(aSzI,aSzI);
    aCdR.mDec =  ToI(aCdR.mC) - aSz/2;
    cPt2dr aCLoc = aCdR.mC-ToR(aCdR.mDec);

    // Read image from file using shift, and make of it a gray image
    // cRGBImage  aIm = cRGBImage::FromFile(mNameIm,cBox2di(aCdR.mDec,aCdR.mDec+aSz),mZoomVisuDetec);
    // aIm.ResetGray();

    cRGBImage aIm =  RGBImFromGray(*mDImInCur,cBox2di(aCdR.mDec,aCdR.mDec+aSz),1.0,mZoomVisuDetec);



    if (aTheorGray)   // generate the theoretical image + the area (ellipse) of gray modelization
    {
          cTmpCdRadiomPos aCRC(aCdR,mThickness);
	  std::vector<cPt2di> aVEllipse;
          aCdR.ComputePtsOfEllipse(aVEllipse);

          for (const auto & aPix :  aVEllipse)
          {
              auto [aState,aWeightWhite] = aCRC.TheorRadiom(ToR(aPix));
              if (aState != eTPosCB::eUndef)
              {
                 tREAL8 aGr = aCdR.mBlack+ aWeightWhite*(aCdR.mWhite-aCdR.mBlack);
		 cPt3di aCoul (128,aGr,aGr);
		 // cPt3di aCoul (aGr,aGr,aGr);
                 aIm.SetRGBPix(aPix-aCdR.mDec,aCoul);
              }
          }
    }

    // Parse all pixel, read radiom, and modify it to "maximize" dynamic
    if (aLocalDyn)   
    {
          const auto & aDImR = aIm.ImR().DIm();
          for (const auto & aPix : cRect2(cPt2di(0,0),aSz) )
	  {
              tREAL8 aGray = aDImR.GetV(aPix*mZoomVisuDetec);
	      aGray = 255.0 * (aGray-aCdR.mBlack) /(aCdR.mWhite-aCdR.mBlack);
	      aIm.SetGrayPix(aPix,round_ni(aGray));
	  }
    }
    

    // show the points that define the frontier (sub pixel detection)
    if (1)   
    {
	       // compute integer frontier point
          std::vector<cPt2di> aIFront;
          aCdR.FrontBlackCC(aIFront,*mDImTmp,10000);

	       // select frontier point that are not on line, then refine their position
          std::vector<cPt2dr> aEllFr;
	  aCdR.SelEllAndRefineFront(aEllFr,aIFront);
	       //  print the  frontier point
          for (const auto & aPt : aEllFr)
          {
              aIm.SetRGBPoint(aPt-ToR(aCdR.mDec),aCoulEllFront);
              aIm.DrawCircle(aCoulEllFront,aPt-ToR(aCdR.mDec),3.0/mZoomVisuDetec);
          }
    }

    // print the axes of the checkboard
    if (1)
    {
	  for (const auto & aTeta  : aCdR.mTetas)
	  {
              for (int aK= -mZoomVisuDetec * 20 ; aK<=mZoomVisuDetec*20 ; aK++)
	      {
		  tREAL8 aAbsc= aK/ (2.0 * mZoomVisuDetec);
		  cPt2dr aPt = aCLoc + FromPolar(aAbsc,aTeta);

	          aIm.SetRGBPoint(aPt,aCol_StrL); 
	      }
	  }
    }

    aIm.SetRGBPoint(aCLoc,aCoulCenter);

    return aIm;
}

void   cAppliCheckBoardTargetExtract::ComplImaEllipse(cRGBImage & aIm,const  cCdEllipse & aCDE) const
{
      cPt3di  aCol_CornBW = cRGBImage::Red;  // color for teta wih blakc on right (on visualization)
      cPt3di  aCol_CornWB = cRGBImage::Green;  // color for teta wih blakc on left (on visualization)
      cPt3di  aCoulBits_0 = cRGBImage::Blue; 
					      
      std::vector<tREAL8> aVRho = {mSpecif->Rho_0_EndCCB(),mSpecif->Rho_1_BeginCode(),mSpecif->Rho_2_EndCode(),2.0};
      int aNb= 20*aCDE.Ell().LGa() ; 
      for (int aK=0 ; aK<aNb ; aK++)
      {
          for (size_t aKRho=0 ; aKRho<aVRho.size() ; aKRho++)
          {
               tREAL8 aRho = aVRho.at(aKRho);
               cPt2dr aPt = aCDE.Ell().PtOfTeta((2*M_PI*aK)/aNb,aRho);
	       cPt3di aCoul = ((aKRho==1)||(aKRho==2)) ? cRGBImage::Red : cRGBImage::Green;
               aIm.SetRGBPoint(aPt-ToR(aCDE.mDec),aCoul);
          }
     }

     // Draw the 2 corner
     aIm.DrawCircle(aCol_CornWB,aCDE.CornerlEl_WB()-ToR(aCDE.mDec),0.5);
     aIm.DrawCircle(aCol_CornBW,aCDE.CornerlEl_BW()-ToR(aCDE.mDec),0.5);

     // Draw the bits position
     for (const auto & aPBit : mSpecif->BitsCenters())
     {
          auto [aVal,aPt] = aCDE.Length2CodingPart(2.0/3.0,aPBit);
          aIm.DrawCircle(aCoulBits_0,aPt-ToR(aCDE.mDec),0.3);
     }

}

std::string cAppliCheckBoardTargetExtract::NameVisu(const std::string & aPref,const std::string aPost) const
{
     std::string aRes = mPhProj.DirVisuAppli() +  aPref +"-" + LastPrefix(FileOfPath(mNameIm));
     if (aPost!="") aRes = aRes + "-"+aPost;
     return    aRes + ".tif";
}



void cAppliCheckBoardTargetExtract::MakeImageLabels(const std::string & aName,const tDIm & aDIm,const cDataIm2D<tU_INT1> & aDMasq) const
{
    cRGBImage  aRGB = RGBImFromGray<tElem>(aDIm);

    for (const auto & aPix : cRect2(aDIm.Dilate(-1)))
    {
       if (aDMasq.GetV(aPix) >= (int) eTopoMaxLoc)
       {
          cPt3di  aCoul = cRGBImage::Yellow;
	  if (aDMasq.GetV(aPix)== eFilterSym) aCoul = cRGBImage::Green;
	  if (aDMasq.GetV(aPix)== eFilterRadiom) aCoul = cRGBImage::Blue;
	  if (aDMasq.GetV(aPix)>= eFilterEllipse)
	  {
             aCoul = cRGBImage::Red;
	  }
          aRGB.SetRGBPix(aPix,aCoul);
       }
    }
    aRGB.ToJpgFileDeZoom(aName,1);
}

bool cAppliCheckBoardTargetExtract::IsPtTest(const cPt2dr & aPt) const
{
   return mHasMasqTest && (mMasqTest.DIm().DefGetV(ToI(aPt * mCurScale),0) != 0);
}

void cAppliCheckBoardTargetExtract::GenerateVisuFinal() const
{
      //  "G" => G-lobal image with rectangles : "green" : target with code OK, "red" : target shape but no code
      if (contains(mStrShow,'G') )
      {
         cRGBImage  aIm = cRGBImage::FromFile(mNameIm);
         aIm.ResetGray();
         for (auto & aCdt :  mVCdtMerged)
         {
             cPt3di aCoul =  aCdt.Code() ?  (aCdt.IsCircle() ? cRGBImage::Cyan : cRGBImage::Green)  : cRGBImage::Red;
	     aIm.SetRGBrectWithAlpha(ToI(aCdt.mC0),50,aCoul, 0.5);
	     if (aCdt.mScale!= 1.0)
	        aIm.SetRGBBorderRectWithAlpha(ToI(aCdt.mC0),60,10,cRGBImage::Blue, 0.5);

	 }
         aIm.ToJpgFileDeZoom(NameVisu("Glob"),1);
      }
}


void cAppliCheckBoardTargetExtract::GenerateVisuDetail(std::vector<cCdEllipse> & aVCdtEll) const
{
      if  (contains(mStrShow,'L'))
         MakeImageLabels(NameVisu("Label"),*mDImInCur,*mDImLabel);


      // "E" : show the ellipse for each decoded, "N": show nums of decoded (to used for debug)
      if (contains(mStrShow,'E') || contains(mStrShow,'N'))
      {
         int aCptIm = 0;
         for (auto & aCdt :  aVCdtEll)
         {
             if (contains(mStrShow,'E'))
             {
                cRGBImage aRGBIm = GenImaRadiom(aCdt,150);
                ComplImaEllipse(aRGBIm,aCdt);
                aRGBIm.ToJpgFileDeZoom(NameVisu( (aCdt.IsCircle() ? "Circle" : "Ellipse"), ToStr(aCptIm)),1);
             }

             if (contains(mStrShow,'N') )
	         StdOut() << "NumIm=" << aCptIm  <<  " NumDebug=" << aCdt.mNum << "\n";

	     aCptIm++;
         }
      }
      
}

void cAppliCheckBoardTargetExtract::SetLabel(const cPt2dr& aPt,tU_INT1 aLabel)
{
     mDImLabel->SetV(ToI(aPt),aLabel);
}

     /* ------------------------------------------------- */
     /*      METHOD FOR VISUALIZATION FOR COMUTATION      */
     /* ------------------------------------------------- */


/*  
 *
 *  (cos(T) U + sin(T) V)^2  =>  1 + 2 cos(T)sin(T) U.V = 1 + sin(2T) U.V, ValMin  -> 1 -U.V
 *
 */

cCdRadiom cAppliCheckBoardTargetExtract::MakeCdtRadiom(cScoreTetaLine & aSTL,const cCdSym & aCdSym,tREAL8 aThickness)
{
    bool IsMarqed = IsPtTest(aCdSym.mC);
    static int aCptGlob=0 ; aCptGlob++;
    static int aCptMarq=0 ; if (IsMarqed) aCptMarq++;
    DebugCB = (aCptMarq == mNumDebugMT) && IsMarqed;

    auto aPairTeta = aSTL.Tetas_CheckBoard(mLInitTeta,aCdSym.mC,0.1,1e-3);
    tREAL8 aLength = aSTL.Prolongate(mLInitProl,mLengtProlong,aPairTeta);

    // now restimate teta with a more appropriate lenght ?
    //  aPairTeta = aSTL.Tetas_CheckBoard(aLength,aCdSym.mC,0.1,1e-3);

    auto [aTeta0,aTeta1] = aPairTeta;

    cCdRadiom aCdRadiom(this,aCdSym,*mDImInCur,aTeta0,aTeta1,aLength,aThickness);

    if (! aCdRadiom.mIsOk) 
       return aCdRadiom;

    if (mOptimSegByRadiom)
    {
       aCdRadiom.OptimSegIm(*(aSTL.DIm()),aLength);
    }

    return aCdRadiom;
}


void cAppliCheckBoardTargetExtract::ReadImagesAndBlurr()
{
    /* [0]    Initialise : read image and mask */

    cAutoTimerSegm aTSInit(mTimeSegm,"0-Init");

	// [0.0]   read image

	// [0.1]   initialize labeling image 
    mDImLabel =  &(mImLabel.DIm());
    mDImLabel->Resize(mSzImCur);
    mDImLabel->InitCste(eNone);

    mDImTmp = &(mImTmp.DIm() );
    mDImTmp->Resize(mSzImCur);
    mDImTmp->InitCste(0);

    /* [1]   Compute a blurred image => less noise, less low level saddle */

    cAutoTimerSegm aTSBlur(mTimeSegm,"1-Blurr");

    mImBlur  = mImInCur.Dup(); // create image blurred with less noise
    mDImBlur = &(mImBlur.DIm());

    SquareAvgFilter(*mDImBlur,mNbBlur1,1,1); // 1,1 => Nbx,Nby
}

void cAppliCheckBoardTargetExtract::ComputeTopoSadles()
{
    cAutoTimerSegm aTSTopoSad(mTimeSegm,"2.0-TopoSad");
    cRect2 aRectInt = mDImInCur->Dilate(-mDistRectInt); // Rectangle excluding point too close to border

         // 2.1  point with criteria on conexity of point > in neighoor

    for (const auto & aPix : aRectInt)
    {
        if (FlagSup8Neigh(*mDImBlur,aPix).NbConComp() >=4)
	{
	    SetLabel(ToR(aPix),eTopo0);
	}
    }

         // 2.2  as often there 2 "touching" point with this criteria
	 // select 1 point in conected component

    cAutoTimerSegm aTSMaxCC(mTimeSegm,"2.1-MaxCCSad");
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
                 aBestPInCC.Add(aPixCC,CriterionTopoSadle(*mDImBlur,aPixCC));
	     }

	     cPt2di aPCC = aBestPInCC.IndexExtre();
	     SetLabel(ToR(aPCC),eTopoMaxOfCC);
	 }
    }
}

/**  The saddle criteria is defined by fitting a quadratic function on the image. Having computed the eigen value of quadratic function :
 *
 *      - this criteria is 0 if they have same sign
 *      - else it is the smallest eigen value
 *
 *   This fitting is done a smpothed version of the image :
 *      - it seem more "natural" for fitting a smooth model
 *      - it limits the effect of delocalization
 *      - it (should be) is not a problem as long as the kernel is smaller than the smallest checkbord we want to detect
 *
 *    As it is used on a purely relative criteria, we dont have to bother how it change the value.
 *     
 */
void cAppliCheckBoardTargetExtract::SaddleCritFiler() 
{
    cAutoTimerSegm aTSCritSad(mTimeSegm,"3.0-CritSad");

    cCalcSaddle  aCalcSBlur(Norm2(mPtLimCalcSadle)+0.001,1.0); // structure for computing saddle criteria

       // [3.1]  compute for each point the saddle criteria
    for (const auto& aPix : *mDImLabel)
    {
         if (mDImLabel->GetV(aPix)==eTopoMaxOfCC)
	 {
             tREAL8 aCritS = aCalcSBlur.CalcSaddleCrit(*mDImBlur,aPix);
             mVCdtSad.push_back(cCdSadle(ToR(aPix),aCritS,IsPtTest(ToR(aPix))) );
	 }
    }
    mNbSads.push_back(mVCdtSad.size()); // memo size for info 

    //   [3.2]    sort by decreasing criteria of saddles => "-"  aCdt.mSadCrit + limit size
    cAutoTimerSegm aTSMaxLoc(mTimeSegm,"3.1-MaxLoc");

    SortOnCriteria(mVCdtSad,[](const auto & aCdt){return - aCdt.mSadCrit;});
    ResizeDown(mVCdtSad,mMaxNbSP_ML0);   
    mNbSads.push_back(mVCdtSad.size()); // memo size for info 


    //   [3.3]  select  MaxLocal
    mVCdtSad = FilterMaxLoc((cPt2dr*)nullptr,mVCdtSad,[](const auto & aCdt) {return aCdt.mC;}, mDistMaxLocSad);
    mNbSads.push_back(mVCdtSad.size()); // memo size for info 

    //   [3.3]  select KBest + MaxLocal
    //  limit the number of point , a bit rough but first experiment show that sadle criterion is almost perfect on good images
    // mVCdtSad.resize(std::min(mVCdtSad.size(),size_t(mMaxNbMLS)));
    ResizeDown(mVCdtSad,mMaxNbSP_ML1);
    mNbSads.push_back(mVCdtSad.size()); // memo size for info 

    for (const auto & aCdt : mVCdtSad)
        SetLabel(aCdt.mC,eTopoMaxLoc);
}

void cAppliCheckBoardTargetExtract::SymetryFiler()
{
    cAutoTimerSegm aTSSym(mTimeSegm,"4-SYM");
    cFilterDCT<tREAL4> * aFSym = cFilterDCT<tREAL4>::AllocSym(mImInCur,0.0,mRayCalcSym0,1.0);
    cOptimByStep<2> aOptimSym(*aFSym,true,mDistDivSym);

    for (auto & aCdtSad : mVCdtSad)
    {
        auto [aValSym,aNewP] = aOptimSym.Optim(aCdtSad.mC,1.0,0.01);  // Pos Init, Step Init, Step Lim
        aCdtSad.mC = aNewP;

        if (aValSym < mThresholdSym)
        {
           mVCdtSym.push_back(cCdSym(aCdtSad,aValSym));
	   SetLabel(aNewP,eFilterSym);
        }
	else if (IsPtTest(aCdtSad.mC))
	{
           StdOut()  << "SYMREFUT,  C=" << aCdtSad.mC << " ValSym=" << aValSym << "\n";
	}
    }

    delete aFSym;
}

void  cAppliCheckBoardTargetExtract::AddCdtE(const cCdEllipse & aCDE)
{
     cCdMerged aNewCdM(mDImIn0,aCDE,mCurScale);

     for (auto & aCdM : mVCdtMerged)
     {
          tREAL8 aD = Norm2(aNewCdM.mC0-aCdM.mC0);

	  if (aD < 10.0)
          {
	      if (aNewCdM.Code() && (! aCdM.Code()) )
                 aCdM  = aNewCdM;
	      return;
	  }
     }

     mVCdtMerged.push_back(aNewCdM);
}

void  cAppliCheckBoardTargetExtract::DoExport()
{
     cSetMesPtOf1Im  aSetM(FileOfPath(mNameIm));
     for (const auto & aCdtM : mVCdtMerged)
     {
         if (aCdtM.Code())
         {
             std::string aCode = aCdtM.Code()->Name() ;
             aSetM.AddMeasure(cMesIm1Pt(aCdtM.mC0,aCode,1.0));
         }
     }

     mPhProj.SaveMeasureIm(aSetM);
}

void cAppliCheckBoardTargetExtract::DoOneImage() 
{
    mInterpol = new   cTabulatedDiffInterpolator(cSinCApodInterpolator(5.0,5.0));

    mSpecif = cFullSpecifTarget::CreateFromFile(mNameSpecif);

    mImIn0 =  tIm::FromFile(mNameIm);
    mDImIn0 = &mImIn0.DIm() ;
    mSzIm0 = mDImIn0->Sz();
    
    // [0.2]   Generate potential mask for test points
    mHasMasqTest = mPhProj.ImageHasMask(mNameIm);
    if (mHasMasqTest)
       mMasqTest =  mPhProj.MaskOfImage(mNameIm,*mDImIn0);


    for (const auto & aScale : mScales)
    {
        DoOneImageAndScale(aScale,mImIn0.Scale(aScale));
    }

    if (mRefinePos)
    {
        for (auto & aCdtM : mVCdtMerged)
            aCdtM.OptimizePosition(*mInterpol);
    }


    cAutoTimerSegm aTSMakeIm(mTimeSegm,"OTHERS");

    GenerateVisuFinal();
    DoExport();
    delete mSpecif;
    delete mInterpol;
}

void cAppliCheckBoardTargetExtract::DoOneImageAndScale(tREAL8 aScale,const  tIm & anIm ) 
{ 
    mVCdtSad.clear();
    mVCdtSym.clear();
    mCurScale     = aScale;

    mImInCur  = anIm;
    mDImInCur = &mImInCur.DIm();
    mSzImCur  = mDImInCur->Sz();

    if (IsInit(&mNumDebugSaddle))
       cCdSadle::TheNum2Debug= mNumDebugSaddle ;

    /* [0]    Initialise : read image ,  mask + Blurr */
    ReadImagesAndBlurr();
    /* [2]  Compute "topological" saddle point */
    ComputeTopoSadles();
    /* [3]  Compute point that are max local of  saddle point criteria */
    SaddleCritFiler();
    /* [4]  Calc Symetry criterion */
    SymetryFiler();

    /* [5]  Compute lines, radiom model & correlation */
    std::vector<cCdRadiom> aVCdtRad;
    cAutoTimerSegm aTSRadiom(mTimeSegm,"Radiom");
    {
        cCubicInterpolator aCubI(-0.5);
        cScoreTetaLine  aSTL(*mDImInCur,aCubI,mStepSeg);
        for (const auto & aCdtSym : mVCdtSym)
        {
            cCdRadiom aCdRad = MakeCdtRadiom(aSTL,aCdtSym,mThickness);
	    if (aCdRad.mCostCorrel <= mMaxCostCorrIm)
	    {
               aVCdtRad.push_back(aCdRad);
	       SetLabel(aCdRad.mC,eFilterRadiom);
	    }
        }
    }

    /* [6]  Compute model of geometry, ellipse & code */
    std::vector<cCdEllipse> aVCdtEll;
    int aNbEllWCode = 0;
    cAutoTimerSegm aTSEllipse(mTimeSegm,"Ellipse");
    {
        int aCpt=0;
        for (const auto & aCdtRad : aVCdtRad)
        {
           std::vector<bool>  TryCE = {false}; // Do we do the try in circle or ellipse mode
	   if (mTryC)  TryCE.push_back(true);
	   bool GotIt = false;
	   for (size_t aKC=0 ; (aKC<TryCE.size()) && (!GotIt) ; aKC++)
	   {
               cCdEllipse aCDE(aCdtRad,*mDImTmp,mNbMaxBlackCB,TryCE.at(aKC));
	       if (aCDE.IsOk())
	       {
	          SetLabel(aCDE.mC,eFilterEllipse);
                  aCDE.DecodeByL2CP(mPropGrayDCD);
                  aVCdtEll.push_back(aCDE);
	          if (aCDE.Code())
	          {
                     // StdOut() << "aCDE.mC,eFilterCodedTargetaCDE.mC,eFilterCodedTarget \n";
                     SetLabel(aCDE.mC,eFilterCodedTarget);
	             aNbEllWCode++;
		     GotIt = true;
	          }
		  AddCdtE(aCDE);
	       }
	   }
           aCpt++;
        }
    }

    cAutoTimerSegm aTSMakeIm(mTimeSegm,"OTHERS");
    if (mMainScale)
    {
      GenerateVisuDetail(aVCdtEll);
      StdOut()  << "NB Cd,  SAD: " << mNbSads
	      << " SYM:" << mVCdtSym.size() 
	      << " Radiom:" << aVCdtRad.size() 
	      << " Ellipse:" << aVCdtEll.size() 
	      << " Code:" << aNbEllWCode << "\n";
    }

    mMainScale = false;
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


};  // ===================  NS_CHKBRD_TARGET_EXTR

using namespace NS_CHKBRD_TARGET_EXTR;
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
