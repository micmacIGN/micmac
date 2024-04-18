#include "MMVII_PCSens.h"
#include "MMVII_ImageInfoExtract.h"
#include "MMVII_ExtractLines.h"
#include "MMVII_TplGradImFilter.h"



namespace MMVII
{

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
                                bool isAccurate,
				bool Show
                           )
{
     // Memorize calbration and initialize hough structure
     mCalib = aCalib;  
     mHough = new cHoughTransform(ToR(mSz),aMulTetaRho,aSigmTeta,aCalib);

     // compute average for weighting
     tREAL8 aAvgIm=0;
     if (isAccurate)
     {
        for (const auto & aPix :   mImMasqCont.DIm())
            aAvgIm += mGrad->NormG().DIm().GetV(aPix);
        aAvgIm /= mGrad->NormG().DIm().NbElem() ;
     }
     
     // Three measure of correction (tuning) 
     tREAL8 aSomCorAff=0;   // sums the distance between point and its correction by refinement
     tREAL8 aSomCorDist=0;  // sums the distance between point and its correction by distorsion
     tREAL8 aSomCorTeta=0;  // sums the angulare distance of distorsion
     int aNbDone=0;

     // for (const auto & aPix :   mImMasqCont.DIm())
     for (const auto & aPix :   mPtsCont)
     {
             if (Show &&  ((aNbDone%200000)==0) )
                StdOut() << "Remain to do " << mNbPtsCont-aNbDone << "\n";
             aNbDone++;

	     // compute eventually refined position
             cPt2dr aRPix0 = ToR(aPix);
	     cPt2dr aRPix = aRPix0;
	     if (isAccurate)
                aRPix = mGrad->RefinePos(aRPix0);
	     aSomCorAff += Norm2(aRPix0-aRPix);

	     //  compute  Teta of grad, for limiting the  computation time
             tREAL8 aTeta = Teta(mGrad->Grad(aPix));
 
	     // if calibration, correct position an teta
             if (aCalib)
             {
                 cPt2dr aCor = aCalib->Undist(aRPix); // point correction
                 cPt2dr aCor2 = aCalib->Undist(aRPix+ FromPolar(0.1,aTeta) ); // point in gradient direction
                 tREAL8 aTetaCorr = Teta(aCor2-aCor); // new teta

                 aSomCorDist += Norm2(aCor-aRPix);  // sum  correction on point due to distorsion
                 aSomCorTeta += std::abs(aTetaCorr-aTeta);  // sum correction on angle due to distorsion
                 aRPix = aCor;  // set position to corrected
                 aTeta = aTetaCorr;  // set teta to corrected
             }
           
	     // compute a weighting, growing with norm, but limited to 1.0
	     tREAL8 aNorm = mGrad->NormG().DIm().GetV(aPix);
	     tREAL8 aW =  aNorm / (aNorm + (2.0*aAvgIm));

	     // finnaly add the point to hough-accumulator
             if (mImMasqCont.DIm().InsideBL(aRPix))
             {
                 if (isAccurate)
                    mHough->Accurate_AccumulatePtAndDir(aRPix,aTeta,aW);
                 else
                 {
                    mHough->Quick_AccumulatePtAndDir(aRPix,aTeta,aW);
                 }
             }
     }
     // make some filter, not sure usefull
     if (isAccurate)
         ExpFilterOfStdDev(mHough->Accum().DIm(),4,1.0);

     if (Show)
     {
        StdOut()  
                  << " , Aff=" <<       (aSomCorAff/aNbDone) 
                  << " , Dist-Pt=" <<   (aSomCorDist/aNbDone) 
                  << " , Dist-Teta=" << (aSomCorTeta/aNbDone) 
                  << std::endl;
     }
}

template <class Type> void cExtractLines<Type>::SetDericheGradAndMasq(tREAL8 aAlpha,tREAL8 aRay,int aBorder,bool Show)
{
     // Create the data for storing gradient & init gradient
     mGrad = new cImGradWithN<Type>(mIm.DIm().Sz());

     bool Quick = true;
     bool IsWhite = true;

     cTabulateGrad * aTabG =  nullptr;
     if (Quick)
     {
        aTabG =new cTabulateGrad(256);
	aTabG->TabulateNeighMaxLocGrad(63,1.7,aRay);  // 1.7=> maintain 8-neighboor,  63 ~
     }
         
     if (Quick)
     {
         mGrad->SetQuickSobel(mIm.DIm(),*aTabG,2);
     }
     else 
     {
         mGrad->SetDeriche(mIm.DIm(),aAlpha);
     }

     cRect2 aRect(mImMasqCont.DIm().Dilate(-aBorder)); // rect interior 
     std::vector<cPt2di>  aVecNeigh = cImGradWithN<Type>::NeighborsForMaxLoc(aRay); // neigbours for compute max

     //  count pts & pts of contour for stat
     mNbPtsCont = 0;
     int aNbPt =0;
     // Parse all points to set the masq if is local maxima in direction of gradient
     for (const auto & aPix :  aRect)
     {
         aNbPt++;
	 bool IsMaxLoc =  Quick                                             ?
		            mGrad->TabIsMaxLocDirGrad(aPix,*aTabG,IsWhite)  :
			    mGrad->IsMaxLocDirGrad(aPix,aVecNeigh,1.0)      ;
         if (IsMaxLoc)
         {
            mPtsCont.push_back(aPix);
            mImMasqCont.DIm().SetV(aPix,255);
            mNbPtsCont++;
         }
     }

     if (Show)
        StdOut()<< " Prop Contour = " << mNbPtsCont / double(aNbPt) << "\n";
     delete aTabG;
}

/* Generate a RGB-image :
 *     - background is initial image
 *     - point of contour are set to red with alpha transparency
 */
template <class Type> cRGBImage cExtractLines<Type>::MakeImageMaxLoc(tREAL8 aAlpha)
{
     cRGBImage aImV(mIm.DIm().Sz()); // init RGB with size
     for (const auto & aPix :  mImMasqCont.DIm())
     {
         aImV.SetGrayPix(aPix,mIm.DIm().GetV(aPix)); // transfer image
	 // set contour 
         if (mImMasqCont.DIm().GetV(aPix))
         {
            aImV.SetRGBPixWithAlpha(aPix,cRGBImage::Red,cPt3dr(aAlpha,aAlpha,aAlpha));
         }
     }
     return aImV;
}



template <class Type> cHoughTransform & cExtractLines<Type>::Hough()   {return *mHough;}
template <class Type> cImGradWithN<Type> & cExtractLines<Type>::Grad() {return *mGrad;}
template <class Type> const std::vector<cPt2di>& cExtractLines<Type>::PtsCont() const {return mPtsCont;}


// =========================  INSTANCIATION ===============

template class cExtractLines<tREAL4>;

};
