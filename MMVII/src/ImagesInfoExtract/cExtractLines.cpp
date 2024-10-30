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
       mSz         (anIm.DIm().Sz()),
       mIm         (anIm),
       mImMasqCont (mSz,nullptr,eModeInitImage::eMIA_Null),
       mDImMasq    (mImMasqCont.DIm()),
       mGrad       (nullptr),
       mTabG       (new cTabulateGrad(256)),
       mHough      (nullptr),
       mCalib      (nullptr)
{
}

template <class Type> cExtractLines<Type>::~cExtractLines()
{
    delete mHough;
    delete mTabG;
    delete mGrad;
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

template <class Type> void cExtractLines<Type>::SetSobelAndMasq(eIsWhite isWhite,tREAL8 aRayMaxLoc,int aBorder,bool Show)
{
     // Create the data for storing gradient & init gradient
     mGrad = new cImGradWithN<Type>(mIm.DIm().Sz());

     mTabG->TabulateNeighMaxLocGrad(63,1.7,aRayMaxLoc);  // 1.7=> maintain 8-neighboor,  63 number of direction
     mGrad->SetQuickSobel(mIm.DIm(),*mTabG,2);

     SetGradAndMasq(eIsQuick::Yes,isWhite, aRayMaxLoc,aBorder,Show);
}

/*  The behaviour is not coherent with "SetSobelAndMasq" , to modify later probably, for now comment 
     
template <class Type>  void cExtractLines<Type>::SetDericheAndMasq(eIsWhite isWhite,tREAL8 aAlphaDerich,tREAL8 aRayMaxLoc,int aBorder,bool Show)
{
     // Create the data for storing gradient & init gradient
     mGrad = new cImGradWithN<Type>(mIm.DIm().Sz());

     mGrad->SetDeriche(mIm.DIm(),aAlphaDerich);

     SetGradAndMasq(eIsQuick::No,isWhite, aRayMaxLoc,aBorder,Show);
}
*/

template <class Type> void cExtractLines<Type>::SetGradAndMasq(eIsQuick isQuick,eIsWhite isWhite,tREAL8 aRayMaxLoc,int aBorder,bool Show)

{
     // Create the data for storing gradient & init gradient
/*
     mGrad = new cImGradWithN<Type>(mIm.DIm().Sz());

     bool Quick = true;
     bool IsWhite = true;

     if (Quick)
     {
	 mTabG->TabulateNeighMaxLocGrad(63,1.7,aRay);  // 1.7=> maintain 8-neighboor,  63 ~
         mGrad->SetQuickSobel(mIm.DIm(),*mTabG,2);
     }
     else 
     {
         mGrad->SetDeriche(mIm.DIm(),aAlpha);
     }
*/

     cRect2 aRect(mImMasqCont.DIm().Dilate(-aBorder)); // rect interior 
     std::vector<cPt2di>  aVecNeigh = cImGradWithN<Type>::NeighborsForMaxLoc(aRayMaxLoc); // neigbours for compute max

     //  count pts & pts of contour for stat
     mNbPtsCont = 0;
     int aNbPt =0;
     // Parse all points to set the masq if is local maxima in direction of gradient
     for (const auto & aPix :  aRect)
     {
         aNbPt++;
	 bool IsMaxLoc =    IsYes(isQuick)                                  ?
		            mGrad->TabIsMaxLocDirGrad(aPix,*mTabG,IsYes(isWhite))  :
			    mGrad->IsMaxLocDirGrad(aPix,aVecNeigh,1.0)      ;
         if (IsMaxLoc)
         {
            mPtsCont.push_back(aPix);
            mImMasqCont.DIm().SetV(aPix,1);
            mNbPtsCont++;
         }
     }

     if (Show)
        StdOut()<< " Prop Contour = " << mNbPtsCont / double(aNbPt) << "\n";
}
/*
*/

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


template <class Type> void  cExtractLines<Type>::MarqBorderMasq(size_t aFlag)
{
    for (const auto & aPix : mDImMasq.Border(1))
       mDImMasq.GetReference_V(aPix) |= aFlag;
}

template <class Type> void  cExtractLines<Type>::UnMarqBorderMasq(size_t aFlag)
{
    for (const auto & aPix : mDImMasq.Border(1))
       mDImMasq.GetReference_V(aPix) &= aFlag;
}

template <class Type> cDataIm2D<tU_INT1>&   cExtractLines<Type>::DImMasq() {return mDImMasq;}


cPt2dr NewPtRefined(const cDenseVect<tREAL8> &aSol,const cSegment2DCompiled<tREAL8> & aSeg0,tREAL8 aDeltaAbs)
{
    tREAL8 aX =  aSeg0.N2() / 2.0 + aDeltaAbs;
    tREAL8 aY = aSol(0) + aSol(1) * aX;
    return aSeg0.FromCoordLoc(cPt2dr(aX,aY));
}

template <class Type> void  cExtractLines<Type>::RefineLineInSpace(cHoughPS & aHPS)
{
// static int aCpt=0 ; aCpt++;
// StdOut() << "\n\n RefineLineInSpacellll " << __LINE__ << " Cpt=" << aCpt << "\n";
    tREAL8 aMaxDL=2.0;  // Max Dist Line -> to parametrize
    MMVII_INTERNAL_ASSERT_strong(mCalib!=nullptr,"RefineLineInSpace w/o Calib stil to write");

// StdOut() << "RefineLineInSpacellll " << __LINE__ << "\n";
    //  ------  [1]  compute the point that are inside a "buffer" arround the undist line
    // --------      Use a connected component algorithm
    cSegment2DCompiled<tREAL8> aSegC (mCalib->ExtenSegUndistIncluded(false,aHPS.Seg()));

// StdOut() << " RefineLineInSpacellll " << __LINE__ << "\n";
         //  [1.1]  initialise the seed
    cPt2di  aSeed = ToI(mCalib->Redist(aSegC.PMil()));


    if ( (!mDImMasq.Inside(aSeed))  || ((mDImMasq.GetV(aSeed)&TheFlagLine) !=0))
       return;
    /*
    if ((!mDImMasq.Inside(aSeed))  |?| (  (mDImMasq.GetV(aSeed)&TheFlagLine) ==0))
       return;
       */
// StdOut() << "RefineLineInSpacellll " << __LINE__ << "\n";

    std::vector<cPt2di>  aBufLine;
    aBufLine.push_back(aSeed);
    mDImMasq.GetReference_V(aSeed) |=  TheFlagLine;
    size_t aCurInd = 0;
//  StdOut() << "RefineLineInSpacellll " << __LINE__ << "\n";

         //  [1.2]  iterate on neighboroud propagation
    std::vector<cPt2di>  aNeighoroud =  AllocNeighbourhood<2>(1);
    size_t aNbN = aNeighoroud.size();
    while( aCurInd != aBufLine.size())
    {
         cPt2di aCurP = aBufLine.at(aCurInd);
         for (size_t aKN=0 ; aKN<aNbN ; aKN++)
         {
             cPt2di aNeigh = aCurP+aNeighoroud.at(aKN);
             tU_INT1 & aMarq = mDImMasq.GetReference_V(aNeigh);
             if (  ((aMarq& TheFlagLine)==0) && (aSegC.Dist(mCalib->Undist(ToR(aNeigh))) < aMaxDL))
             {
                  aMarq |=  TheFlagLine;
                  aBufLine.push_back(aNeigh);
             }
         }
         aCurInd++;
    }
//  StdOut() << "RefineLineInSpacellll " << __LINE__ << "\n";

    int aNbIter = 2;
    std::vector<cStdStatRes>    aVStat(aNbIter+1);

    //-----[2]  extract point fitting the line, they must be
    //----- max loc of gradi and  with direction close to normal,
    //---- also refine their position
    tREAL8 aCosMin = std::cos(M_PI/10.0); // thresold to select good gradient direction
    std::vector<cPt2dr>  aVPInit;         // vector of point in "initial" coordinates (for check)

    for (const auto & aPix : aBufLine )
    {
        tU_INT1 & aMarq = mDImMasq.GetReference_V(aPix);
        aMarq &= TheFlagSuprLine; // clean the flag
        if (aMarq !=0)    // once flag cleaned, point marke are local directional max
        {
            cPt2dr aGrad = mGrad->GradN(aPix);  // extract grad normalized
            tREAL8 aCos = -Scal(aGrad,aSegC.Normal()); // cos : scal between unit vect
         
            if (aCos> aCosMin)  // if direction close enouh
            {
                cPt2dr aRPix =   mCalib->Undist(mGrad->RefinePos(ToR(aPix)));  // refine pose & undist
                aVPInit.push_back(aRPix);  // memorize position
                aVStat.at(0).Add(aSegC.Dist(aRPix));         // memorize residual4 stat
            }
        }
    }
//  StdOut() << "RefineLineInSpacellll " << __LINE__ << "\n";


    //-----[3]  Fit an adjusted line, in local coordinate just fit y = ax +b
    //---- also test to fit a parabol  optionnaly,  memorize the optimized segment
    tREAL8 aSigmaW= 0.2;  // sigma for weighting residual
    cSegment2DCompiled<tREAL8>  aNewSeg = aSegC;  // optimized segment

    std::vector<tREAL8>  aSumNumbering(aNbIter,0.0) ;  // sum of weight to count point weighted by "quality"

    for (int aKIter=0 ; aKIter<aNbIter ; aKIter++)  // if want to test with degre 2, try aKTest<2
    {
        cLeasSqtAA<tREAL8> aSys(2);
        for (const auto aPt : aVPInit)
        {
             cDenseVect<tREAL8> aV(2);  // vector of obs
             cPt2dr aPixL = aNewSeg.ToCoordLoc(aPt);
             aV(0) = 1.0; 
             aV(1) = aPixL.x();
             tREAL8 aW= 1/(1 + Square(aPixL.y()/aSigmaW));  // weight to reduce outlayer
             aSys.PublicAddObservation(aW,aV,aPixL.y());   // Add obs to system
        }
        cDenseVect<tREAL8> aSol = aSys.PublicSolve();

        // w/o parabol
        aNewSeg = cSegment2DCompiled<tREAL8>(NewPtRefined(aSol,aNewSeg,-1),NewPtRefined(aSol,aNewSeg,1));
        for (const auto & aPt : aVPInit)
        {
            tREAL8 aD = aNewSeg.Dist(aPt);
            aVStat.at(aKIter+1).Add(aD);
            aSumNumbering.at(aKIter) += 1/(1 + Square(aD/aSigmaW));
        }
    }
//  StdOut() << "RefineLineInSpacellll " << __LINE__ << "\n";

    if (0)
    {
         StdOut()  
              << " SSS=" << Scal(aNewSeg.Tgt(),aSegC.Tgt())
              << " NB=" << aVPInit.size()
              << " ERRINIT 0.5:" << aVStat.at(0).ErrAtProp(0.5)
              << " 0.75 :"       << aVStat.at(0).ErrAtProp(0.75)
              << "  #  " 
              << " ERRADJ 0.5:"  << aVStat.at(1).ErrAtProp(0.5)
              << " 0.75 :"       << aVStat.at(1).ErrAtProp(0.75)
              << "  #  " 
              << " ERRADJ 0.5:"  << aVStat.at(2).ErrAtProp(0.5)
              << " 0.75 :"       << aVStat.at(2).ErrAtProp(0.75)
              << " Wnb=" << aSumNumbering
              << " Cumul=" << aHPS.Cumul()
              << "\n";
    }
//  StdOut() << "RefineLineInSpacellll " << __LINE__ << "\n";

    aHPS.UpdateSegImage(aNewSeg,aSumNumbering.back(),aVStat.back().QuadAvg());
// StdOut() << "RESSSS " << aVStat.at(aKIter+1).QuadAvg() << "\n";
}



template <class Type> cHoughTransform & cExtractLines<Type>::Hough()   {return *mHough;}
template <class Type> cImGradWithN<Type> & cExtractLines<Type>::Grad() {return *mGrad;}
template <class Type> const std::vector<cPt2di>& cExtractLines<Type>::PtsCont() const {return mPtsCont;}


// =========================  INSTANCIATION ===============

template class cExtractLines<tREAL4>;

};
