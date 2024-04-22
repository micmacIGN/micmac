#ifndef _MMVII_EXTRACT_LINES_H_
#define _MMVII_EXTRACT_LINES_H_
#include "MMVII_Image2D.h"


namespace MMVII
{

class cHoughTransform;
template <class Type> class  cImGradWithN;
template <class Type> class  cExtractLines;

enum class eCodeHPS
{
      Ok,
      LowCumul,
      NotFirst
};

class cHoughPS : public cMemCheck
{
     public :
         typedef cSegment2DCompiled<tREAL8> tSeg;

         cHoughPS(const cHoughTransform *,const cPt2dr & aPS,tREAL8 aCumul,const cPt2dr & aP1,const cPt2dr & aP2);
	 /// Angular distance for line anti parallel
	 tREAL8 DistAnglAntiPar(const cHoughPS& aPS2) const;
         tREAL8 DY(const cHoughPS&) const;
         tREAL8 Dist(const cHoughPS&,const tREAL8 &aFactTeta=1.0) const;
         tSeg2dr  SegMoyAntiParal(const cHoughPS& aPS2) const;

	 const cPt2dr & TetaRho() const; ///< Accessor
	 const tREAL8 & Teta() const;    ///< Accessor
	 const tREAL8 & Rho() const;     ///< Accessor
	 const tSeg & Seg() const ;      ///< Accessor
         cHoughPS * Matched() const;     ///< Accessor
	 const tREAL8 & Cumul() const;   ///< Accessor
	 eCodeHPS  Code() const ;        ///< Accessor
         void SetCode(eCodeHPS);         ///< Modifior
         bool IsBestMatch() const;       ///< Accessor
         void SetIsBestMatch();          ///< Modifior

	 cPt2dr  IndTetaRho() const; ///< Teta/Rho in hough accum dynamic

	 void Test(const cHoughPS & ) const;

	 bool Match(const cHoughPS &,bool IsDark,tREAL8 aMaxTeta,tREAL8 aDMin,tREAL8 aDMax) const;

	 static void SetMatch(std::vector<cHoughPS*>&  mVPS,bool IsLight,tREAL8 aMaxTeta,tREAL8 aDMin,tREAL8 aDMax);

         void UpdateSegImage(const tSeg & aNewSeg,tREAL8 aNewCumul);

     private :
	 void InitMatch();
	 void UpdateMatch(cHoughPS *,tREAL8 aDist);

         const cHoughTransform *  mHT;
         cPt2dr                   mTetaRho;
         tREAL8                   mCumul;
         tSeg                     mSegE;
         tSeg                     mOldSeg;
         cHoughPS *               mMatched;
         tREAL8                   mDistM;
	 eCodeHPS                 mCode;
         bool                     mIsBestMatch;
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
	 cHoughPS * PtToLine(const cPt3dr &) const;

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

template <class Type> class  cExtractLines
{
      public :
          typedef  cIm2D<Type>      tIm;
          typedef  cDataIm2D<Type>  tDIm;
          static const size_t  TheFlagLine=2;
          static const size_t  TheFlagSuprLine=  0xFFFFFFFF ^ TheFlagLine;

          cExtractLines(tIm anIm);  ///< constructor , memorize image
          ~cExtractLines();

	  /// initialize the gradient
          void SetDericheGradAndMasq(tREAL8 aAlphaDerich,tREAL8 aRayMaxLoc,int aBorder,bool Show=false);

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


      private :
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

};
#endif //  _MMVII_EXTRACT_LINES_H_
       //
