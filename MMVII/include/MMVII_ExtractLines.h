#ifndef _MMVII_EXTRACT_LINES_H_
#define _MMVII_EXTRACT_LINES_H_

#include "MMVII_Image2D.h"
#include "MMVII_Interpolators.h"
#include "MMVII_Mappings.h"


namespace MMVII
{

class cHoughTransform;
template <class Type> class  cImGradWithN;
template <class Type> class  cExtractLines;

enum class eCodeHPS
{
      Ok,
      LowCumul
};

class cHoughPS // : public cMemCheck
{
     public :
         typedef cSegment2DCompiled<tREAL8> tSeg;

         cHoughPS(const cHoughTransform *,const cPt2dr & aPS,tREAL8 aCumul,const cPt2dr & aP1,const cPt2dr & aP2);
	 /// Angular distance for line anti parallel
	 tREAL8 DistAnglAntiPar(const cHoughPS& aPS2) const;
         tREAL8 DY(const cHoughPS&) const;
         tREAL8 Dist(const cHoughPS&,const tREAL8 &aFactTeta=1.0) const;

	 const cPt2dr & TetaRho() const; ///< Accessor
	 const tREAL8 & Teta() const;    ///< Accessor
	 const tREAL8 & Rho() const;     ///< Accessor
	 const tSeg & Seg() const ;      ///< Accessor
	 const tREAL8 & Cumul() const;   ///< Accessor
	 const tREAL8 & SigmaL() const;   ///< Accessor
	 eCodeHPS  Code() const ;        ///< Accessor
         void SetCode(eCodeHPS);         ///< Modifior
         const cHoughTransform *  HT() const; ///< Accessor

	 cPt2dr  IndTetaRho() const; ///< Teta/Rho in hough accum dynamic

	 void Test(const cHoughPS & ) const;
	 bool Match(const cHoughPS &,bool IsDark,tREAL8 aMaxTeta,tREAL8 aDMin,tREAL8 aDMax) const;
	 static std::vector<std::pair<int,int>> GetMatches(const std::vector<cHoughPS>&  mVPS,bool IsLight,tREAL8 aMaxTeta,tREAL8 aDMin,tREAL8 aDMax);

         void UpdateSegImage(const tSeg & aNewSeg,tREAL8 aNewCumul,tREAL8 aSigmaL);

     protected :
	 void UpdateMatch(cHoughPS *,tREAL8 aDist);

         const cHoughTransform *  mHT;
         cPt2dr                   mTetaRho;
         tREAL8                   mCumul;
         tREAL8                   mSigmaL;
         tSeg                     mSegE;
	 eCodeHPS                 mCode;
};

class cParalLine 
{
    public :
         typedef cSegment2DCompiled<tREAL8> tSeg;

         cParalLine(const cHoughPS & aS1,const cHoughPS & aS2);
	 const tREAL8 & ScoreMatch() const ;    ///< Accessor
	 cOneLineAntiParal GetLAP(const cPerspCamIntrCalib &) const;

         size_t RankMatch() const;       ///< Accessor
         void SetRankMatch(size_t);          ///< Modifior
	 const std::vector<cHoughPS> &   VHS() const; ///< Accessor

	 void  ComputeRadiomHomog(const cDataGenUnTypedIm<2> &,cPerspCamIntrCalib *,const std::string & aNameFile) ;

	 tREAL8 DistGt(const tSeg2dr &) const;

	 bool RejectByComparison(const cParalLine & aBetterOne) const;
    private :
	 std::vector<cHoughPS>     mVHS;
         tSeg         mMidleSeg;
	 tREAL8       mScoreMatch;
         int          mRankMatch;
	 tREAL8       mRadHom; // Radiometric homogeneity
	 tREAL8       mAngleDif;
	 tREAL8       mWidth;
};

/** cHoughTransform
                
       for a line  L (rho=R,teta=T) with , vector orthog to  (cos T,sin T) :
      the equation is  :

             (x,y) in L  <=> x cos(T) + y Sin(T) = R

      For now we consider oriented line,  typically when the point on a line comes from gradient,
    in this case we have :
 
           T in [0,2Pi]    R in [-RMax,+RMax]
*/
class cHoughTransform
{
    public :
         cHoughTransform
         (
	      const cPt2dr  & aSzIn,       // Sz of the space
	      const cPt2dr & aMulTetaRho,  // Multiplicator of Rho & Teta
	      const tREAL8 & aSigmTeta,    //  Incertitude on gradient direction
              cPerspCamIntrCalib * aCalib = nullptr  // possible internal calib for distorsion correction
          );

         ///  Add  a point with a given direction
         void  Quick_AccumulatePtAndDir(const cPt2dr & aPt,tREAL8 aTeta0,tREAL8 aWeight);
         void  Accurate_AccumulatePtAndDir(const cPt2dr & aPt,tREAL8 aTeta0,tREAL8 aWeight);

         cIm2D<tREAL4>      Accum() const; ///< Accessor

	 tREAL8  AvgS2() const; ///< Avg of square, possibly use to measure "compactness"
	 // tREAL8  Max() const;   ///< Max of val, possibly use to measure "compactness"


	 /// Extract the local maxima retur point/line in hough space + value of max
	 std::vector<cPt3dr> ExtractLocalMax
		             (
			           size_t aNbMax,  // bound to number of max-loc
				   tREAL8 aDist,   // min distance between maxima
				   tREAL8 aThrAvg, // threshold on Average, select if Accum > aThrAvg * Avg
				   tREAL8 aThrMax  // threshold on Max, select if Accum > Max * Avg
                             ) const;

	 /// max the conversion houg-point + value ->  euclidian line  + rho teta
	 cHoughPS  PtToLine(const cPt3dr &) const;

	 /// make the conversion seg (oriented)  -> hough point 
	 cPt2dr  Line2PtInit(const tSeg2dr &) const;
	 cPt2dr  Line2PtPixel(const tSeg2dr &) const;

	 const tREAL8 & RhoMax() const; ///<  Accessor
	 ///   return the angle teta, of a given  index/position in hough accumulator
         inline tREAL8      RInd2Teta(tREAL8 aIndTeta) const {return aIndTeta  *mFactI2T;}
	 ///   idem, return teta for "integer" index
         inline tREAL8      Ind2Teta(int aK) const {return RInd2Teta(aK);}
	 /// for a given teta, return  the index  (RInd2Teta o Teta2RInd = Identity)
         inline tREAL8      Teta2RInd(const tREAL8 & aTeta) const {return aTeta /mFactI2T;}

	 /// return the rho of a given index/position in hough accumulator
         inline tREAL8      RInd2Rho(const tREAL8 & aRInd) const { return (aRInd-1.0) / mMulRho - mRhoMax; }
	 /// return  the index of a given rho (Rho2RInd o RInd2Rho = Identity)
         inline tREAL8      Rho2RInd(const tREAL8 & aRho) const {return 1.0+ (aRho+mRhoMax) * mMulRho;}

	 tREAL8  GetValueBlob(cPt2di aP,int aMaxNeigh)  const;
    private :
	 void ExtendMoreTeta() const;

         // inline tREAL8      R2Teta(tREAL8 aIndTeta) const {return aIndTeta  *mFactI2T;}
	 
         cPt2dr             mMiddle;      ///< Middle point, use a origin of Rho
         tREAL8             mRhoMax;      ///< Max of distance to middle point
         tREAL8             mMulTeta;     ///< Teta multiplier, if =1 ,  1 pix teta ~ 1 pix init  (in worst case)
         tREAL8             mMulRho;      ///< Rho multiplier  , if =1, 1 pix-rho ~ 1 pix init
         tREAL8             mSigmTeta;    ///< incertitude on teta
	 cPerspCamIntrCalib* mCalib;      ///< Potential calibration for distorsion
         int                mNbTeta;      ///< Number of Teta for hough-accum
	 int                mMoreTeta;     ///< a bit more teta to handle topol struct
         tREAL8             mFactI2T ;    ///< Ratio Teta-Radian / Teta-Index
         int                mNbRho;       ///< Number of Rho for hough-accum

         cIm1D<tREAL8>      mTabSin;      ///< Tabulation of sinus for a given index of teta
         cDataIm1D<tREAL8>& mDTabSin;     ///< Data Image of "mTabSin"
         cIm1D<tREAL8>      mTabCos;      ///< Tabulation of co-sinus for a given index of teta
         cDataIm1D<tREAL8>& mDTabCos;     ///< Data Image of "mTabCos"
         cIm2D<tREAL4>      mAccum;       ///<  Accumulator of Hough
         cDataIm2D<tREAL4>& mDAccum;      ///< Data Image of "mAccum"
};

/**  Class for storing Grad + its norm
 */
template <class Type> class  cImGradWithN : public cImGrad<Type>
{
     public :
        ///  Constructor using size
        cImGradWithN(const cPt2di & aSz);
        ///  Constructor using image & Alpha-Deriche parameters
        void  SetDeriche(cDataIm2D<Type> & aDIm,Type aAlphaDeriche);

        ///  Compute sobel and norm with tabulation
        void  SetQuickSobel(cDataIm2D<Type> & aDIm,cTabulateGrad &,int aDiv);

	/// Is it a local-maxima in the direction of the gradient
        bool  IsMaxLocDirGrad(const cPt2di& aPix,const std::vector<cPt2di> &,tREAL8 aRatioXY = 1.0) const;

	/// Idem "IsMaxLocDirGrad" but used tabulated grad to accelerate
	bool  TabIsMaxLocDirGrad(const cPt2di& aPix,const cTabulateGrad &,bool isWhite) const;

	/// Allocat the neighbourhood use for computing local-maxima
        static  std::vector<cPt2di>  NeighborsForMaxLoc(tREAL8 aRay);
        cIm2D<Type>      NormG() {return mNormG;}  ///< Accessor
	cPt2dr   RefinePos(const cPt2dr &) const;  ///< Refine a sub-pixelar position of the contour

        cPt2dr  GradN(const cPt2di & aPix) {return ToR(this->Grad(aPix))/ tREAL8(mDataNG.GetV(aPix));}
        tREAL8  NormG(const cPt2di & aPix) {return  tREAL8(mDataNG.GetV(aPix));}
     private :
	cPt2dr   OneRefinePos(const cPt2dr &) const; ///< One iteration of refinement
 
        cIm2D<Type>       mNormG;   ///< Image of norm of gradient
        cDataIm2D<Type>&  mDataNG;  ///<  Data Image of "mNormG"
};
/// Compute the deriche + its norm
template<class Type> void ComputeDericheAndNorm(cImGradWithN<Type> & aResGrad,const cDataIm2D<Type> & aImIn,double aAlpha) ;


/**  Class for extracting line using gradient & hough transform*/

enum class eIsWhite
{
     Yes,
     No
};

enum class eIsQuick
{
     Yes,
     No
};

template <class Type> bool IsYes(const Type & aVal) {return aVal==Type::Yes;}

template <class Type> class  cExtractCurves
{
      public :
          typedef  cIm2D<Type>      tIm;
          typedef  cDataIm2D<Type>  tDIm;
          static const size_t  TheFlagLine=2;
          static const size_t  TheFlagSuprLine=  0xFFFFFFFF ^ TheFlagLine;

          cExtractCurves(tIm anIm);  ///< constructor , memorize image
          ~cExtractCurves();

          // isWhite is necessary for oriented test on max loc

	  /// initialize the gradient
          void SetSobelAndMasq(eIsWhite,tREAL8 aRayMaxLoc,int aBorder,bool Show=false);
          void SetDericheAndMasq(eIsWhite,tREAL8 aAlphaDerich,tREAL8 aRayMaxLoc,int aBorder,bool Show=false);


	  ///  Initialize the hough transform
          void SetHough(const cPt2dr & aMulTetaRho,tREAL8 aSigmTeta,cPerspCamIntrCalib *,bool Accurate,bool Show=false);

	  /// Generate an image for visualizing the contour,
          cRGBImage MakeImageMaxLoc(tREAL8 aAlphaTransparency);
          
          cHoughTransform &  Hough();  ///< Accessor
          cImGradWithN<Type> &  Grad(); ///< Acessor
          const std::vector<cPt2di>& PtsCont() const; ///< Accessor

          void  RefineLineInSpace(cHoughPS &); ///< refine the position of the line by matching on contour
          void MarqBorderMasq(size_t aFlag= TheFlagLine);  ///< write a flag on border
          void UnMarqBorderMasq(size_t aFlag= TheFlagSuprLine);  ///<  clear the flag on border
          cDataIm2D<tU_INT1>&   DImMasq(); ///< Accessor (for visu ?)


      protected :
          void SetGradAndMasq(eIsQuick Quick,eIsWhite isWhite,tREAL8 aRayMaxLoc,int aBorder,bool Show=false);

          cPt2di                mSz;        ///<  Size of the image
          tIm                   mIm;        ///< Memorize the image
          cIm2D<tU_INT1>        mImMasqCont;    ///<  Masq of point selected as contour
          cDataIm2D<tU_INT1>&   mDImMasq;    ///<  Masq of point selected as contour
          
          int                   mNbPtsCont;     ///<  Number of point detected as contour

          cImGradWithN<Type> *  mGrad;          ///< Structure allocated for computing gradient
          cTabulateGrad *       mTabG ;
          cHoughTransform    *  mHough;         ///< Structure allocatedf or computing hough
          cPerspCamIntrCalib *  mCalib;         ///< (Optional) calibration for distorsion correction
          std::vector<cPt2di>   mPtsCont;      ///< List of point in mImMasqCont
};



/** Class for extracting a line arround a point, for now its very specialized in the extraction of
 * direction in checkboard, may evolve to things more general ?
 */

class cScoreTetaLine : public tFunc1DReal // herit from tFunc1DReal for optimization in cOptimByStep
{
     public :
         ///  constructor  : aL length of the segment, aStep step of discretization in a segment
         cScoreTetaLine(cDataIm2D<tREAL4> &,const cDiffInterpolator1D & ,tREAL8 aStep);

	 /// extract the 2 angle of line in checkboar, aStepInit & aStepLim => used in cOptimByStep
         std::pair<tREAL8,tREAL8> Tetas_CheckBoard(tREAL8 aLength,const cPt2dr& aC,tREAL8 aStepInit,tREAL8 aStepLim);

	 typedef  tREAL8  t2Teta[2];
	 ///  Assure that teta1->teta2 is trigo and teta1<Pi
	 static void  NormalizeTetaCheckBoard(t2Teta &);

	 const tREAL8 &  LengthCur() const;  ///< Accessor
         cDataIm2D<tREAL4> * DIm() const; ///< Accessor

	 /// Current lenght can vary 
	 void SetLengthCur(tREAL8 aL);

	 tREAL8 Prolongate(tREAL8 aL0,tREAL8 aLMax,const std::pair<tREAL8,tREAL8> & aTeta) const;

         tREAL8  Score2Teta(const std::pair<tREAL8,tREAL8> & aTeta, tREAL8 aAbscMin) const;
     private :

	 ///  fix center of reusing the data (to avoid cost for cTabulatedDiffInterpolator)
         void SetCenter(const cPt2dr & aC);

	 /// Extract the initial values of tetas by testing all at a given step (in pixel)
         tREAL8  GetTetasInit(tREAL8 aStepPix,int aCurSign);
	 /// Refine the value with a limit step in pixel
         tREAL8  Refine(tREAL8 aTeta0,tREAL8 aStepPix,int aSign);

	 /// Value as tFunc1DReal
         cPt1dr  Value(const cPt1dr& aPt) const override;

         tREAL8  ScoreOfTeta(const tREAL8 & aTeta, tREAL8 aAbscMin,tREAL8 aSign) const;

         cDataIm2D<tREAL4> *         mDIm;

         tREAL8                      mStepAInit;
         tREAL8                      mLengthCur;
         int                         mNb;
         tREAL8                      mStepAbsc;
         tREAL8                      mStepTeta;
         cTabulatedDiffInterpolator  mTabInt;
         cPt2dr                      mC;
         tREAL8                      mCurSign;  /// used to specify an orientaion of segment
         tREAL8                      mStepTetaInit;
         tREAL8                      mStepTetaLim;
};

/**  Base class for optimizing  of a segment it inherits of tFunc2DReal so that it can be used by " cOptimByStep<2>"
 * in OptimizeSeg  . The function "CostOfSeg" must be defined.
*/

class cOptimPosSeg : public tFunc2DReal
{
       public :
            ///  constructor takes initial segment
            cOptimPosSeg(const tSeg2dr &);

            
            tSeg2dr OptimizeSeg(tREAL8 aStepInit,tREAL8 aStepLim,bool IsMin,tREAL8 aMaxDInfInit) const;
       private :
            cSegment2DCompiled<tREAL8>  ModifiedSeg(const cPt2dr & aPModif) const;
	    /// Inteface CostOfSeg to make it a function of tFunc2DReal
            cPt1dr Value(const cPt2dr &) const override;
	    ///  Method to define for indicating the cost to optimize
            virtual tREAL8 CostOfSeg(const cSegment2DCompiled<tREAL8> &) const = 0;

            cSegment2DCompiled<tREAL8>  mSegInit; ///< initial value of segment
            cPt2dr                      mP0Loc;   ///< First  point in local coordinates of segment (y=0)
            cPt2dr                      mP1Loc;   ///< Second point in local coordinates of segment (y=0)
};

/**   Class for refine the position of a segment where the objective is that, on a given image "I" , the points have a given value "V" :
 *
 *                -----
 *                \
 *                /      | I(m) - V|
 *   Cost(Seg) =  -----
 *                 p in Seg
 * */

template <class Type> class cOptimSeg_ValueIm : public cOptimPosSeg
{
      public :
         cOptimSeg_ValueIm(const tSeg2dr &,tREAL8 aStepOnSeg,const cDataIm2D<Type> & aDIm,tREAL8 aTargetValue);

         tREAL8 CostOfSeg(const cSegment2DCompiled<tREAL8> &) const override;
      private :
         tREAL8                    mStepOnSeg;   ///< Sampling step on the seg
         int                       mNbOnSeg;     ///< Number of points (computed from mStepOnSeg)
         const cDataIm2D<Type> & mDataIm;      ///<  Image on which it is computed
         tREAL8                    mTargetValue; ///<
};


};
#endif //  _MMVII_EXTRACT_LINES_H_
       //
