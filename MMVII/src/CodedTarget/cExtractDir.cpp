#include "CodedTarget.h"
#include "include/MMVII_2Include_Serial_Tpl.h"
#include "include/MMVII_Tpl_Images.h"


namespace MMVII
{

template <class Type>  cExtractDir<Type>::cExtractDir(tIm anIm,double aRhoMin,double aRhoMax) :
     mIm        (anIm),  // memorize the shared pointer on image
     mDIm       (mIm.DIm()),  // memorize a faster acces to the raw image
     mRhoMin    (aRhoMin),
     mRhoMax    (aRhoMax),
     mPtsCrown  (SortedVectOfRadius(0.0,mRhoMax))  // compute vectot of neighboord sorted by norm
{
    // compute list of circles with a step of 1 pixel, and their direction
    for (double aRho = aRhoMin ; aRho<aRhoMax ; aRho++)
    {
         mVCircles.push_back(GetPts_Circle(cPt2dr(0,0),aRho,true));  // true=> 8 connexity
         mVDIrC.push_back(tVDir());

         // compute direction real that will be used for extracting axes of checkboard
         for (const auto& aPix :  mVCircles.back())
         {
               mVDIrC.back().push_back(VUnit(ToR(aPix)));
         }
    }
}

template <class Type>  double cExtractDir<Type>::Score(const cPt2dr & aDInit,double aDeltaTeta) 
{
    cPt2dr aDir = aDInit * FromPolar(1.0,aDeltaTeta);
    double aSomDiff = 0.0;
    double aStepRho = 1.0;
    int aNb=0;

    for (double aRho =mRhoMin/2 ; aRho<mRhoMax; aRho+=aStepRho)
    {
         float aV1 = mDIm.GetVBL(mPDCT->mPt+aRho* aDir);
         float aV2 = mDIm.GetVBL(mPDCT->mPt-aRho* aDir);

	 aSomDiff += std::abs(aV1-mVThrs) + std::abs(aV2-mVThrs);
	 aNb += 2;
    }

    return aSomDiff /aNb;
}

template <class Type>  cPt2dr cExtractDir<Type>::OptimScore(const cPt2dr & aDir,double aStepTeta)
{
    cWhitchMin<int,double>  aWMin(0,1e10);

    for (int aK=-1; aK<=1 ; aK++)
        aWMin.Add(aK,Score(aDir,aStepTeta*aK));

    aStepTeta *= aWMin.IndexExtre();

    if (aStepTeta==0) return aDir;

    double aScore = aWMin.ValExtre();
    double aScorePrec = 2*aScore;
    int aKTeta = 1;

    while(aScore < aScorePrec)
    {
	 aScorePrec = aScore;
	 aScore= Score(aDir,aStepTeta*(aKTeta+1));
	 aKTeta++;
    }
    aKTeta--;

    return aDir * FromPolar(1.0,aKTeta*aStepTeta);

}

double TestDir(const cNS_CodedTarget::cGeomSimDCT & aGT,const cNS_CodedTarget::cDCT  &aDCT)
{
    cPt2dr anEl1 = VUnit(aGT.mCornEl1-aGT.mC) ;
    cPt2dr anEl2 = VUnit(aGT.mCornEl2-aGT.mC) ;

    //StdOut()<<  (anEl1^anEl2) <<  " " <<  (aDCT.mDirC1 ^aDCT.mDirC2) << "\n";

    if (Scal(anEl2,aDCT.mDirC2) < 0)
    {
        anEl1 = - anEl1;
        anEl2 = - anEl2;
    }

    cPt2dr aD1 = anEl1 / aDCT.mDirC1;
    cPt2dr aD2 = anEl2 / aDCT.mDirC2;


    double aSc =  (std::abs(ToPolar(aD1).y()) + std::abs( ToPolar(aD2).y())) /  2.0 ;

    return aSc;
}


template <class Type>  bool cExtractDir<Type>::CalcDir(tDCT & aDCT) 
{
     mPDCT = & aDCT;  // memorize as internal variables
     std::vector<float>  aVVals;  // vectors of value of pixel , here to avoir reallocation
     std::vector<bool>   aVIsW;   // vector of boolean IsWhite ?  / IsBlack ?
     cPt2di aC= ToI(aDCT.mPt); // memorize integer center
     mVThrs = (aDCT.mVBlack+aDCT.mVWhite)/2.0;  // threshold for being black or white 

     
     cPt2dr aSomDir[2] = {{0,0},{0,0}};  // accumulate for black and white average direction

     //  Parse all circles
     for (int aKC=0 ; aKC<int(mVCircles.size()) ; aKC++)
     {
         const auto & aCircle  = mVCircles[aKC];
         const auto & aVDir    = mVDIrC[aKC];
         int aNbInC = aCircle.size();
         aVVals.clear();
         aVIsW.clear();
         //  parse the circle , for each pixel compute gray level and its thresholding 
         for (const auto & aPt : aCircle)
         {
             float aVal  = mDIm.GetV(aC+aPt);
             aVVals.push_back(aVal);
             aVIsW.push_back(aVal>mVThrs);
         }
         
         int aCpt = 0;
         // parse the value to detect black/white transitions
         for (int  aKp=0 ; aKp<aNbInC ; aKp++)
         {
             int aKp1 = (aKp+1)%aNbInC;  // next index, circulary speaking
             if (aVIsW[aKp] != aVIsW[aKp1])  // if we have a transition
             {
                 aCpt++;   // one more transition
                 cPt2dr aP1  = aVDir[aKp];  // unitary direction before transition
                 cPt2dr aP2  = aVDir[aKp1];  // unitary direction after transition
                 double aV1 = aVVals[aKp];   // value befor trans
                 double aV2 = aVVals[aKp1];  // value after trans
                 // make a weighted average of P1/P2 corresponding to linear interpolation with threshold
                 cPt2dr aDir =   (aP1 *(aV2-mVThrs) + aP2 * (mVThrs-aV1)) / (aV2-aV1);
                 if (SqN2(aDir)==0) return false;  // not interesting case
                 aDir = VUnit(aDir);  // reput to unitary 
                 aDir = aDir * aDir;  // make a tensor of it => double its angle, complexe-point multiplication 
                 aSomDir[aVIsW[aKp]] += aDir;  // acculate the direction in black or whit transition
             }
         }
         // if we dont have exactly 4 transition, there is someting wrong ...
         if (aCpt!=4 )  return false;
     }

     // now recover from the tensor one of its two vectors (we have no control one which)
     for (auto & aDir : aSomDir)
     {
         aDir = ToPolar(aDir,0.0);  // cartesian => polar  P= (Rho,Theta)  
         aDir = FromPolar(1.0,aDir.y()/2.0);  // polar=>cartesian  P.y() = theta
     }

     aDCT.mDirC1 = aSomDir[1];
     aDCT.mDirC2 = aSomDir[0];

     // As each directio is up to Pi,  and this arbirtray Pi is indepensant we may have 
     // an orientation problem  and Dir1,Dir2 being sometime a direct repair and sometime
     // an indirect one.  
     if ( (aDCT.mDirC1^aDCT.mDirC2) < 0)
     {
         aDCT.mDirC2 = -aDCT.mDirC2;
     }
     // Optimisation of direction, utility : uncertain
     aDCT.mDirC1 =  OptimScore(aDCT.mDirC1,1e-3);
     aDCT.mDirC2 =  OptimScore(aDCT.mDirC2,1e-3);

     return true;
}

/*  In this function we will establish a theoreticall model of the radiometry of the target,
    and compute a difference with the effective radiometry fund in the image
*/
template <class Type>  double cExtractDir<Type>::ScoreRadiom(tDCT & aDCT) 
{
     // compute the affine transformation that goes frome the canonical target repair to the image
     cAffin2D  aInit2Loc(aDCT.mPt,aDCT.mDirC1,aDCT.mDirC2);
     //  we need in fact the inverse, from image to reference target
     cAffin2D  aLoc2Init = aInit2Loc.MapInverse();

     //  we will need to compute the distance do the line to take care of transition pixel
     //  and weight them differently
     cSegment2DCompiled aSeg1(aDCT.mPt,aDCT.mPt+aDCT.mDirC1);
     cSegment2DCompiled aSeg2(aDCT.mPt,aDCT.mPt+aDCT.mDirC2);


     double aSomWEc     = 0.0; // sum of weighted difference
     double aSomWeight = 0.0;  // sum of weight

     cMatIner2Var<double>  aMat;  // use for computing correlation
     double aCorMin = 1.0;  // will compute min of correlation

     // parse all disc in increasing norm
     for (const auto & aPCr : mPtsCrown)
     {
          cPt2di aIPix = aPCr+aDCT.Pix();  // integral  pixel in image
          cPt2dr aRPix = ToR(aIPix);  //  real value
          cPt2dr aRPixInit = aLoc2Init.Value(aRPix); // correspond pixel in target

	  float aVal = mDIm.GetV(aIPix);  // value in image
          // the chekbord has 4 quarter, blac for x>0 and y>0 or inverse
	  bool isW = ((aRPixInit.x()>=0) != (aRPixInit.y()>=0) );
          // compute theoreticall value knwoing if corresponding pixel is black or white
	  float aValTheo = isW ?  aDCT.mVWhite : aDCT.mVBlack;

          // compute a weight to decrease the influence of transition pixel
	  double aWeight = 1.0;
	  double aD1 =  aSeg1.Dist(aRPix);  // distance to first line
	  double aD2 =  aSeg2.Dist(aRPix);  // distance to second
          // weight is 0 on the line,  1 if we are far enough, and proportional to line (closests)
          // in between;  what far enough means is controled by aMaxW
          double aMaxW= 1.0;
	  aWeight = std::min(   std::min(aD1,aMaxW ), std::min(aD2,aMaxW)) / aMaxW;

          //  accumulate for difference
	  aSomWeight += aWeight;
	  aSomWEc +=  aWeight * std::abs(aValTheo-aVal);
 
          // accumulate for correlation
          aMat.Add(aWeight,aVal,aValTheo);

          if  (Norm2(aPCr)>mRhoMin)
              UpdateMin(aCorMin,aMat.Correl());
     }

     aSomWEc /= aSomWeight;
     double aDev = aDCT.mVWhite - aDCT.mVBlack;
     aSomWEc /= aDev;

     aDCT.mScRadDir = aSomWEc;
     aDCT.mCorMinDir = aCorMin;

     if (aDCT.mGT)
     {
        StdOut() << (aDCT.mGT ? "++" : "--");
        StdOut() << "Difff=" <<    aSomWEc << " " << aCorMin   ;
        if (aDCT.mGT &&(aSomWEc>0.08))
           StdOut() <<  " *********************";
        if (aDCT.mGT &&(aCorMin<0.9))
           StdOut() <<  " ########################";
        
        StdOut() << "\n";
     }

     return aSomWEc;
}


template class cExtractDir<tREAL4>;

bool TestDirDCT(cNS_CodedTarget::cDCT & aDCT,cIm2D<tREAL4> anIm,double aRayCB)
{
    cExtractDir<tREAL4>  anED(anIm,aRayCB*0.4,aRayCB*0.8);
    bool Ok = anED.CalcDir(aDCT);
    if (!Ok) return false;

    anED.ScoreRadiom(aDCT) ;

    return (aDCT.mScRadDir <0.12) && (aDCT.mCorMinDir>0.85) ;

}



};
