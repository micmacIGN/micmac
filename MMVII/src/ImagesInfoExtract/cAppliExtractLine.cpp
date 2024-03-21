#include "MMVII_Linear2DFiltering.h"
#include "MMVII_Tpl_Images.h"
#include "MMVII_Sensor.h"
#include "MMVII_PCSens.h"
#include "V1VII.h"
#include "MMVII_ImageInfoExtract.h"


namespace MMVII
{

class cHoughTransform;
template <class Type> class  cImGradWithN;
template <class Type> class  cExtractLines;

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
         void  AccumulatePtAndDir(const cPt2dr & aPt,tREAL8 aTeta0,tREAL8 aWeight);
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

	 /// max the conversion houg-point ->  euclidian line
	 cSegment<tREAL8,2> PtToLine(const cPt2dr &) const;
    private :
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

         // inline tREAL8      R2Teta(tREAL8 aIndTeta) const {return aIndTeta  *mFactI2T;}
	 
         cPt2dr             mMiddle;      ///< Middle point, use a origin of Rho
         tREAL8             mRhoMax;      ///< Max of distance to middle point
         tREAL8             mMulTeta;     ///< Teta multiplier, if =1 ,  1 pix teta ~ 1 pix init  (in worst case)
         tREAL8             mMulRho;      ///< Rho multiplier  , if =1, 1 pix-rho ~ 1 pix init
         tREAL8             mSigmTeta;    ///< incertitude on teta
	 cPerspCamIntrCalib* mCalib;      ///< Potential calibration for distorsion
         int                mNbTeta;      ///< Number of Teta for hough-accum
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
	/// Is it a local-maxima in the direction of the gradient
        bool  IsMaxLocDirGrad(const cPt2di& aPix,const std::vector<cPt2di> &) const;
	/// Allocat the neighbourhood use for computing local-maxima
        static  std::vector<cPt2di>  NeighborsForMaxLoc(tREAL8 aRay,tREAL8 aRatioXY = 1.0);
        cIm2D<Type>      NormG() {return mNormG;}  ///< Accessor
	cPt2dr   RefinePos(const cPt2dr &) const;  ///< Refine a sub-pixelar position of the contour
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

          cExtractLines(tIm anIm);  ///< constructor , memorize image
          ~cExtractLines();

	  /// initialize the gradient
          void SetDericheGradAndMasq(tREAL8 aAlphaDerich,tREAL8 aRayMaxLoc,int aBorder);
	  ///  Initialize the hough transform
          void SetHough(const cPt2dr & aMulTetaRho,tREAL8 aSigmTeta,cPerspCamIntrCalib *,bool AffineMax);

	  /// Generate an image for visualizing the contour,
          cRGBImage MakeImageMaxLoc(tREAL8 aAlphaTransparency);
          
          cHoughTransform &  Hough();  ///< Accessor
          cImGradWithN<Type> &  Grad(); ///< Acessor
      private :
          cPt2di                mSz;        ///<  Size of the image
          tIm                   mIm;        ///< Memorize the image
          cIm2D<tU_INT1>        mImMasqCont;    ///<  Masq of point selected as contour
          int                   mNbPtsCont;     ///<  Number of point detected as contour

          cImGradWithN<Type> *  mGrad;          ///< Structure allocated for computing gradient
          cHoughTransform    *  mHough;         ///< Structure allocatedf or computing hough
          cPerspCamIntrCalib *  mCalib;         ///< (Optional) calibration for distorsion correction
};

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
     //  Tabulate  "cos" and "sin"
     for (int aKTeta=0 ; aKTeta<mNbTeta ; aKTeta++)
     {
          mDTabSin.SetV(aKTeta,std::sin(Ind2Teta(aKTeta)));
          mDTabCos.SetV(aKTeta,std::cos(Ind2Teta(aKTeta)));
     }
}

cIm2D<tREAL4>      cHoughTransform::Accum() const {return mAccum;}

tREAL8  cHoughTransform::AvgS2() const
{
   return std::sqrt(DotProduct(mDAccum,mDAccum) / mDAccum.NbElem());
}

cSegment<tREAL8,2> cHoughTransform::PtToLine(const cPt2dr & aPt) const
{
   // (x,y) in L  <=> x cos(T) + y Sin(T) = R
   tREAL8  aTeta = RInd2Teta(aPt.x());
   tREAL8  aRho  = RInd2Rho(aPt.y());
   cPt2dr  aTgt(-sin(aTeta),cos(aTeta));
   cPt2dr  aP0 = mMiddle + cPt2dr(aRho*cos(aTeta),aRho*sin(aTeta));

   return cSegment<tREAL8,2>(aP0-aTgt,aP0+aTgt);
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
            aRes.push_back(cPt3dr(aPAff.x(),aPAff.y(),mDAccum.GetV(aPt)));
    }

    // [5] Sort with highest value first, then select NbMax
    SortOnCriteria(aRes,[](const auto & aP) {return -aP.z();});
    while (aRes.size() > aNbMax)  
          aRes.pop_back();


    return aRes;


    /*
    if (aVMaxI.size() > aNbMax)
    {
        std::vector<tREAL8> aVThrs;
        for (const auto & aPt : aExtr.mPtsMax)
        {
            aVThrs.push_back(-aDAccum.GetV(aPt));
        }
	tREAL8 aVThr = - IKthVal(aVThrs,aNbMax);
      
	aVMaxI.clear();
        for (const auto & aPt : aExtr.mPtsMax)
        {
            if (aDAccum.GetV(aPt) > aVThr)
               aVMaxI.push_back(aPt);
        }
	//KthVal
    }


    std::vector<cPt3dr> aRes;
    cAffineExtremum<tREAL4>  aAffin(mAccum.DIm(),1.5);
    for (const auto aPt : aVMaxI)
    {
         cPt2dr aPAff = aAffin.OneIter(ToR(aPt));
	 if ( mDAccum.InsideBL(aPAff))
            aRes.push_back(cPt3dr(aPAff.x(),aPAff.y(),mDAccum.GetVBL(aPAff)));
    }

    SortOnCriteria(aRes,[](const auto & aP) {return -aP.z();});
    return aRes;
    */
}

/* ************************************************************************ */
/*                                                                          */
/*                       cImGradWithN                                       */
/*                                                                          */
/* ************************************************************************ */







template <class Type>   
  cImGradWithN<Type>::cImGradWithN(const cPt2di & aSz) :
     cImGrad<Type>  (aSz),
     mNormG         (aSz),
     mDataNG        (mNormG.DIm())
{
}

template<class Type> bool  cImGradWithN<Type>::IsMaxLocDirGrad(const cPt2di& aPix,const std::vector<cPt2di> & aVP) const
{
    tREAL8 aN = mDataNG.GetV(aPix);

    if (aN==0) return false;

    cPt2dr aDirGrad = ToR(this->Grad(aPix)) * (1.0/ aN);

    //  A Basic test to reject point on integer neighbourhood
    {
        cPt2di aIDirGrad = ToI(aDirGrad);
        for (int aSign : {-1,1})
        {
             cPt2di  aNeigh =  aPix + aIDirGrad * aSign;
             if ( (mDataNG.DefGetV(aNeigh,-1)>aN) && (Scal(aDirGrad,ToR(this->Grad(aNeigh)))>0) )
             {
                return false;
             }
        }
    }


    for (const auto & aDeltaNeigh : aVP)
    {
        cPt2di aNeigh = aPix + ToI(ToR(aDeltaNeigh) * aDirGrad);
        if ( (mDataNG.DefGetV(aNeigh,-1)>aN) && (Scal(aDirGrad,ToR(this->Grad(aNeigh))) >0))
           return false;
       // Compute dir of Neigh in gradient dir

        /*cPt2dr aNeigh = ToR(aPix) + ToR(aDeltaNeigh) * aDirGrad;
        if ( (mDataNG.DefGetVBL(aNeigh,-1)>aN) && (Scal(aDirGrad,ToR(this->GradBL(aNeigh))) >0))
           return false;
        */
    }

    return true;
}

template<class Type> std::vector<cPt2di>   cImGradWithN<Type>::NeighborsForMaxLoc(tREAL8 aRay,tREAL8 aRatioXY)
{
   std::vector<cPt2di> aVec = SortedVectOfRadius(0.5,aRay);

   std::vector<cPt2di> aRes ;
   for (const auto & aPix : aVec)
      if (std::abs(aPix.x()) >= std::abs(aPix.y()*aRatioXY))
         aRes.push_back(aPix);

  return aRes;
}

template<class Type> cPt2dr   cImGradWithN<Type>::OneRefinePos(const cPt2dr & aP1) const
{
     if (! mDataNG.InsideBL(aP1)) 
         return aP1;

     cPt2dr aGr = VUnit(ToR(this->GradBL(aP1)));

     cPt2dr aP0 = aP1- aGr;
     if (! mDataNG.InsideBL(aP0)) 
         return aP1;

     cPt2dr aP2 = aP1+ aGr;
     if (! mDataNG.InsideBL(aP2)) 
         return aP1;

     tREAL8 aAbs = StableInterpoleExtr(mDataNG.GetVBL(aP0),mDataNG.GetVBL(aP1),mDataNG.GetVBL(aP2));

     return aP1 + aGr * aAbs;
}
template<class Type> cPt2dr   cImGradWithN<Type>::RefinePos(const cPt2dr & aP1) const
{
    return OneRefinePos(OneRefinePos(aP1));
}

          /* ************************************************ */

template<class Type> void ComputeDericheAndNorm(cImGradWithN<Type> & aResGrad,const cDataIm2D<Type> & aImIn,double aAlpha) 
{
     ComputeDeriche(aResGrad,aImIn,aAlpha);

     auto & aDN =  aResGrad.NormG().DIm();
     for (const auto &  aPix : aDN)
     {
           aDN.SetV(aPix,Norm2(aResGrad.Grad(aPix)));
     }
}




/* ************************************************************************ */
/*                                                                          */
/*                       cExtractLines                                      */
/*                                                                          */
/* ************************************************************************ */



template <class Type> cExtractLines<Type>::cExtractLines(tIm anIm) :
       mSz       (anIm.DIm().Sz()),
       mIm       (anIm),
       mImMasqCont   (mSz,nullptr,eModeInitImage::eMIA_Null),
       mGrad     (nullptr),
       mHough    (nullptr),
       mCalib    (nullptr)
{
}

template <class Type> cExtractLines<Type>::~cExtractLines()
{
    delete mGrad;
    delete mHough;
}

template <class Type> void cExtractLines<Type>::SetHough
                           (
                                const cPt2dr & aMulTetaRho,
                                tREAL8 aSigmTeta,
                                cPerspCamIntrCalib * aCalib,
                                bool AffineMax
                           )
{
     mCalib = aCalib;
     mHough = new cHoughTransform(ToR(mSz),aMulTetaRho,aSigmTeta,aCalib);

     tREAL8 aAvgIm=0;
     for (const auto & aPix :   mImMasqCont.DIm())
         aAvgIm += mGrad->NormG().DIm().GetV(aPix);
     aAvgIm /= mGrad->NormG().DIm().NbElem() ;
     StdOut() << "AVGGGGG " << aAvgIm  << "\n";
     
     tREAL8 aSomCorAff=0;  // sums the distance between point and its correction by dist
     tREAL8 aSomCorDist=0;  // sums the distance between point and its correction by dist
     tREAL8 aSomCorTeta=0;  // sums the distance between point and its correction by dist
     int aNbDone=0;



     for (const auto & aPix :   mImMasqCont.DIm())
     {
         if ( mImMasqCont.DIm().GetV(aPix))
         {
             if ((aNbDone%200000)==0) 
                StdOut() << "Remain to do " << mNbPtsCont-aNbDone << "\n";
             aNbDone++;

             cPt2dr aRPix0 = ToR(aPix);
	     cPt2dr aRPix = aRPix0;
	     if (AffineMax)
                aRPix = mGrad->RefinePos(aRPix0);
	     aSomCorAff += Norm2(aRPix0-aRPix);

             cPt2df aGrad =  mGrad->Grad(aPix);
             tREAL8 aTeta = Teta(aGrad);
 
             if (aCalib)
             {
            //  tPtOut Undist(const tPtOut &) const;
                 cPt2dr aCor = aCalib->Undist(aRPix);
            // StdOut() << " DDdDD " << aCalib->Redist(aCor) -aRPix << "\n";
                 cPt2dr aCor2 = aCalib->Undist(aRPix+ FromPolar(0.1,aTeta) );

                 tREAL8 aTetaCorr = Teta(aCor2-aCor);

                 aSomCorDist += Norm2(aCor-aRPix);
                 aSomCorTeta += std::abs(aTetaCorr-aTeta);
                 aRPix = aCor;
                 aTeta = aTetaCorr;
             }
           
	     tREAL8 aNorm = mGrad->NormG().DIm().GetV(aPix);
	     tREAL8 aW =  aNorm / (aNorm + (2.0*aAvgIm));
             if (mImMasqCont.DIm().InsideBL(aRPix))
                 mHough->AccumulatePtAndDir(aRPix,aTeta,aW);
         }
     }
     ExpFilterOfStdDev(mHough->Accum().DIm(),4,1.0);
     {
        StdOut()  
                  << " , Aff=" <<       (aSomCorAff/aNbDone) 
                  << " , Dist-Pt=" <<   (aSomCorDist/aNbDone) 
                  << " , Dist-Teta=" << (aSomCorTeta/aNbDone) 
                  << std::endl;
     }
}

// cHoughTransform::cHoughTransform(const cPt2dr  & aSzIn,const tREAL8 &  aMul,const tREAL8 & aSigmTeta) :
// void  cHoughTransform::AccumulatePtAndDir(const cPt2dr & aPt,tREAL8 aTetaC,tREAL8 aWeight)

template <class Type> void cExtractLines<Type>::SetDericheGradAndMasq(tREAL8 aAlpha,tREAL8 aRay,int aBorder)
{
     // Create the data for storing gradient
     mGrad = new cImGradWithN<Type>(mIm.DIm().Sz());
     //  compute the gradient & its norm using deriche method
     ComputeDericheAndNorm(*mGrad,mIm.DIm(),aAlpha);

     cRect2 aRect(mImMasqCont.DIm().Dilate(-aBorder));
     std::vector<cPt2di>  aVec = cImGradWithN<Type>::NeighborsForMaxLoc(aRay,1.1);

     mNbPtsCont = 0;
     int aNbPt =0;
     for (const auto & aPix :  aRect)
     {
         aNbPt++;
         if (mGrad->IsMaxLocDirGrad(aPix,aVec))
         {
            mImMasqCont.DIm().SetV(aPix,255);
            mNbPtsCont++;
         }
     }
     StdOut()<< " Prop Contour = " << mNbPtsCont / double(aNbPt) << "\n";
}

template <class Type> cRGBImage cExtractLines<Type>::MakeImageMaxLoc(tREAL8 aAlpha)
{
     cRGBImage aImV(mIm.DIm().Sz());
     for (const auto & aPix :  mImMasqCont.DIm())
     {
         aImV.SetGrayPix(aPix,mIm.DIm().GetV(aPix));
         if (mImMasqCont.DIm().GetV(aPix))
         {
            tREAL8 aAlpha= 0.5;
            aImV.SetRGBPixWithAlpha(aPix,cRGBImage::Red,cPt3dr(aAlpha,aAlpha,aAlpha));
         }
     }
     return aImV;
}



template <class Type> cHoughTransform & cExtractLines<Type>::Hough()   {return *mHough;}
template <class Type> cImGradWithN<Type> & cExtractLines<Type>::Grad() {return *mGrad;}

/* =============================================== */
/*                                                 */
/*                 cAppliExtractLine               */
/*                                                 */
/* =============================================== */

/**  An application for  testing the accuracy of a sensor : 
        - consistency of direct/inverse model
        - (optionnaly) comparison with a ground truth
 */

class cAppliExtractLine : public cMMVII_Appli
{
     public :
        cAppliExtractLine(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);
     private :
        typedef tREAL4 tIm;


        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;
	// std::vector<std::string>  Samples() const override;

        void  DoOneImage(const std::string & aNameIm) ;

        cPhotogrammetricProject  mPhProj;
        std::string              mPatImage;
        bool                     mShowSteps;
        cPerspCamIntrCalib *     mCalib;
	bool                     mAffineMax;
	std::vector<double>      mThreshCpt;
        tREAL8                   mAlphaContour;
};

/*
std::vector<std::string>  cAppliExtractLine::Samples() const
{
   return {
              "MMVII TestSensor SPOT_1B.tif SPOT_Init InPointsMeasure=XingB"
	};
}
*/

cAppliExtractLine::cAppliExtractLine(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
    cMMVII_Appli    (aVArgs,aSpec),
    mPhProj         (*this),
    mShowSteps      (true),
    mCalib          (nullptr),
    mAffineMax      (true),
    mThreshCpt      {100,200,400,600},
    mAlphaContour   (0.5)
{
}

cCollecSpecArg2007 & cAppliExtractLine::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
      return    anArgObl
             << Arg2007(mPatImage,"Name of input Image", {eTA2007::FileDirProj,{eTA2007::MPatFile,"0"}})
      ;
}

cCollecSpecArg2007 & cAppliExtractLine::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
    return anArgOpt
               << mPhProj.DPOrient().ArgDirInOpt("","Folder for calibration to integrate distorsion")
	       << AOpt2007(mAffineMax,"AffineMax","Affinate the local maxima",{eTA2007::HDV})
	       << AOpt2007(mShowSteps,"ShowSteps","Show detail of computation steps by steps",{eTA2007::HDV})
            ;
}



void  cAppliExtractLine::DoOneImage(const std::string & aNameIm)
{
    mCalib = nullptr;
    if (mPhProj.DPOrient().DirInIsInit())
       mCalib = mPhProj.InternalCalibFromImage(aNameIm);

    cIm2D<tIm> anIm = cIm2D<tIm>::FromFile(aNameIm);
    cExtractLines<tIm>  anExtrL(anIm);

    anExtrL.SetDericheGradAndMasq(2.0,10.0,2);
    anExtrL.SetHough(cPt2dr(1.0/M_PI,1.0),0.1,mCalib,mAffineMax);

    std::vector<cPt3dr> aVMaxLoc = anExtrL.Hough().ExtractLocalMax(10,4.0,10.0,0.1);

    StdOut() << "MMMAXVect:: " << aVMaxLoc << "\n";

    if (mShowSteps)
    {
        std::vector<int>  aVCpt(mThreshCpt.size(),0);
        const  cDataIm2D<tREAL4>& aDAccum =anExtrL.Hough().Accum().DIm();
	
	tREAL8 aVMax=0.0;
        for (const auto & aPix : aDAccum)
        {
            tREAL8 aV = aDAccum.GetV(aPix);
	    UpdateMax(aVMax,aV);
            for (size_t aK=0 ; aK<mThreshCpt.size() ; aK++)
            {
                if (aV>mThreshCpt.at(aK))
                   aVCpt.at(aK)++;
            }
        }
        std::string aNameTif = LastPrefix(aNameIm) + ".tif";

	StdOut() << "VMAX=" << aVMax << std::endl;
        for (size_t aK=0 ; aK<mThreshCpt.size() ; aK++)
            StdOut() << " Cpt=" << aVCpt.at(aK) << " for threshold " << mThreshCpt.at(aK) << std::endl;

	//  [1]  Visu selected max of gradient
	{
            cRGBImage aImV= anExtrL.MakeImageMaxLoc(mAlphaContour);
            aImV.ToJpgFileDeZoom(mPhProj.DirVisu() + "DetectL_"+ aNameTif,1);
	}

	//  [2]  Visu module of gradient
	{
	    std::string aNameGrad = mPhProj.DirVisu()+"Grad_" + aNameTif;
	    anExtrL.Grad().NormG().DIm().ToFile(aNameGrad);
	    Convert_JPG(aNameGrad,true,90,"jpg");
	}

	//  anExtrL.Grad().NormG().DIm().ToFile("toto.tif");
       
	// [3] Visu  the accum + local maximal
	{
            cRGBImage  aVisAccum =  RGBImFromGray(aDAccum,255.0/aVMax);
            for (const auto & aP : aVMaxLoc)
            {
                aVisAccum.SetRGBrectWithAlpha(cPt2di(round_ni(aP.x()),round_ni(aP.y())) ,15,cRGBImage::Red,0.5);
            }
	    aVisAccum.ToJpgFileDeZoom(mPhProj.DirVisu() + "Accum_" + aNameTif,1);
	}
	// [4]  Visu of Image + 
	{
            int aZoom = 1;
            cRGBImage  aVisIm =  cRGBImage::FromFile(aNameIm,aZoom);
	    const auto & aDIm = aVisIm.ImR().DIm();
            for (size_t aKH=0 ; aKH<aVMaxLoc.size() ; aKH++)
	    {
                cPt3dr aPHough = aVMaxLoc[aKH];
                cPt3di aCoul = cRGBImage::Red;
                if (aKH>=2)  aCoul = cRGBImage::Green;
                if (aKH>=4)  aCoul = cRGBImage::Blue;
                cSegment<tREAL8,2> aSeg =  anExtrL.Hough().PtToLine(cPt2dr(aPHough.x(),aPHough.y()));
                for (tREAL8 aSign : {-1.0,1.0})
                {
                    cPt2dr aPt = aSeg.PMil();
                    cPt2dr aTgt = VUnit(aSeg.V12()) * aSign;
		    cPt2dr  aQ = mCalib ? mCalib->Redist(aPt) : aPt;
                    while (aDIm.InsideBL(aQ))
                    {
                        aVisIm.SetRGBPoint(aQ,aCoul);
                        aPt = aPt+aTgt;
		        aQ = mCalib ? mCalib->Redist(aPt) : aPt;
                    }
                }
	    }
	    std::string aNameLine = mPhProj.DirVisu() + "Lines_" + aNameTif;
	    if (aZoom==1)
	        aVisIm.ToJpgFileDeZoom(mPhProj.DirVisu() + "Lines_" + aNameTif,1);
	    else
	        aVisIm.ToFile(aNameLine);

	}

	// cSegment<tREAL8,2> cHoughTransform::PtToLine(const cPt2dr & aPt) const
    }
}



int cAppliExtractLine::Exe()
{
    mPhProj.FinishInit();

    if (RunMultiSet(0,0))
    {
       return ResultMultiSet();
    }
    DoOneImage(UniqueStr(0));
    return EXIT_SUCCESS;
}


tMMVII_UnikPApli Alloc_AppliExtractLine(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
      return tMMVII_UnikPApli(new cAppliExtractLine(aVArgs,aSpec));
}


cSpecMMVII_Appli  TheSpecAppliExtractLine
(
     "ExtractLine",
      Alloc_AppliExtractLine,
      "Extraction of lines",
      {eApF::Ori},
      {eApDT::Ori,eApDT::GCP},
      {eApDT::Console},
      __FILE__
);

};
