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

namespace NS_CHKBRD_TARGET_EXTR { 

class cAppliCheckBoardTargetExtract ;

class cCdSadle;
class cCdSym;
class cCdRadiom;
class cTmpCdRadiomPos ;
class cCdEllipse ;
class cCdMerged ;
class cOptimPosCdM;

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

	  /// compute frontier point of black connected component with center as seed
          bool FrontBlackCC(std::vector<cPt2di> & aRes,cDataIm2D<tU_INT1> & aMarq,int aNbMax) const;
	  ///  Select point of front that are on ellipse
          void SelEllAndRefineFront(std::vector<cPt2dr> & aRes,const std::vector<cPt2di> &) const;


	  /// Is the possibly the point on arc of black  ellipse, if yes refine it
          bool PtIsOnEllAndRefine(cPt2dr &) const;
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
           cCdEllipse(const cCdRadiom &,cDataIm2D<tU_INT1> & aMarq,int aNbMax,bool isCircle,std::vector<cPt2dr>& aEllFr);
	   bool IsOk() const;
	   const cEllipse & Ell() const;
           const cPt2dr &   CornerlEl_WB() const;
           const cPt2dr &   CornerlEl_BW() const;
           cPt2dr  M2I(const cPt2dr & aPMod) const;
           cPt2dr  I2M(const cPt2dr & aPIM) const; 
	   bool  IsCircle() const;

       const cAff2D_r   &  AffIm2Mod() const;



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

           /// once compute ellipse, use line to compute  affinity image <-> ref target
           void EstimateAffinity();
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
            void  HeuristikOptimizePosition(const cDiffInterpolator1D &,tREAL8 aStepEnd);

            void  GradOptimizePosition(const cDiffInterpolator1D &,tREAL8 aStepEnd);

	    const cDataIm2D<tREAL4> * mDIm0;
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

	tREAL8                mStepHeuristikRefinePos;
	tREAL8                mStepGradRefinePos;
	// bool                  mDoGradRefine;
	
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
	cDiffInterpolator1D * mInterpol;
        std::string           mIdExportCSV;  /// Identifier for handling result as CSV-files
};


}; // NS_CHKBRD_TARGET_EXTR
}; // MMVII
