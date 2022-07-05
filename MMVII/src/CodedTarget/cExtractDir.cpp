#include "CodedTarget.h"
#include "include/MMVII_2Include_Serial_Tpl.h"
#include "include/MMVII_Tpl_Images.h"


namespace MMVII
{


template <class Type>  class cExtractDir
{
     public :
         typedef cIm2D<Type>     tIm;
         typedef cDataIm2D<Type> tDIm;
	 typedef cNS_CodedTarget::cDCT tDCT;
         typedef std::vector<cPt2dr> tVDir;

         cExtractDir(tIm anIm,double aRhoMin,double aRhoMax);
         bool  CalcDir(tDCT &) ;
         double ScoreRadiom(tDCT & aDCT) ;
     public :

	 double Score(const cPt2dr & ,double aTeta); 
	 cPt2dr OptimScore(const cPt2dr & ,double aStepTeta); 


          tIm     mIm;
          tDIm&   mDIm;
          float   mRhoMin;
          float   mRhoMax;

	  tResFlux                mPtsCrown;
          std::vector<tResFlux>   mVCircles;
          std::vector<tVDir>      mVDIrC;
          float                   mVThrs ;
	  tDCT *                  mPDCT;
       // (SortedVectOfRadius(aR0,aR1,IsSym))
};

template <class Type>  cExtractDir<Type>::cExtractDir(tIm anIm,double aRhoMin,double aRhoMax) :
     mIm        (anIm),
     mDIm       (mIm.DIm()),
     mRhoMin      (aRhoMin),
     mRhoMax      (aRhoMax),
     mPtsCrown  (SortedVectOfRadius(0.0,mRhoMax))
{
    for (double aRho = aRhoMin ; aRho<aRhoMax ; aRho++)
    {
         mVCircles.push_back(GetPts_Circle(cPt2dr(0,0),aRho,true));
         mVDIrC.push_back(tVDir());

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
     mPDCT = & aDCT;
     std::vector<float>  aVVals;
     std::vector<bool>   aVIsW;
     cPt2di aC= ToI(aDCT.mPt);
     mVThrs = (aDCT.mVBlack+aDCT.mVWhite)/2.0;

     
     cPt2dr aSomDir[2] = {{0,0},{0,0}};

     for (int aKC=0 ; aKC<int(mVCircles.size()) ; aKC++)
     {
         const auto & aCircle  = mVCircles[aKC];
         const auto & aVDir    = mVDIrC[aKC];
         int aNbInC = aCircle.size();
         aVVals.clear();
         aVIsW.clear();
         for (const auto & aPt : aCircle)
         {
             float aVal  = mDIm.GetV(aC+aPt);
             aVVals.push_back(aVal);
             aVIsW.push_back(aVal>mVThrs);
         }
         int aCpt = 0;
         for (int  aKp=0 ; aKp<aNbInC ; aKp++)
         {
             int aKp1 = (aKp+1)%aNbInC;
             if (aVIsW[aKp] != aVIsW[aKp1])
             {
                 aCpt++;
                 cPt2dr aP1  = aVDir[aKp];
                 cPt2dr aP2  = aVDir[aKp1];
                 double aV1 = aVVals[aKp];
                 double aV2 = aVVals[aKp1];
                 cPt2dr aDir =   (aP1 *(aV2-mVThrs) + aP2 * (mVThrs-aV1)) / (aV2-aV1);
                 if (SqN2(aDir)==0) return false;
                 aDir = VUnit(aDir);
                 aDir = aDir * aDir;  // make a tensor of it => double its angle
                 aSomDir[aVIsW[aKp]] += aDir;
             }
         }
         if (aCpt!=4 )  return false;
     }

     for (auto & aDir : aSomDir)
     {
         aDir = ToPolar(aDir,0.0);
         aDir = FromPolar(1.0,aDir.y()/2.0);
     }
     aDCT.mDirC1 = aSomDir[1];
     aDCT.mDirC2 = aSomDir[0];

     // As each directio is up to Pi, make it oriented
     if ( (aDCT.mDirC1^aDCT.mDirC2) < 0)
     {
         aDCT.mDirC2 = -aDCT.mDirC2;
     }

     aDCT.mDirC1 =  OptimScore(aDCT.mDirC1,1e-3);
     aDCT.mDirC2 =  OptimScore(aDCT.mDirC2,1e-3);

     return true;
}

template <class Type>  double cExtractDir<Type>::ScoreRadiom(tDCT & aDCT) 
{
     cAffin2D  aInit2Loc(aDCT.mPt,aDCT.mDirC1,aDCT.mDirC2);
     cAffin2D  aLoc2Init = aInit2Loc.MapInverse();

     cSegment2DCompiled aSeg1(aDCT.mPt,aDCT.mPt+aDCT.mDirC1);
     cSegment2DCompiled aSeg2(aDCT.mPt,aDCT.mPt+aDCT.mDirC2);

     FakeUseIt(aLoc2Init);


     double aSomWeight = 0.0;
     double aSomWEc     = 0.0;

     cMatIner2Var<double>  aMat;
     double aCorMin = 1.0;

     for (const auto & aPCr : mPtsCrown)
     {
          cPt2di aIPix = aPCr+aDCT.Pix();
          cPt2dr aRPix = ToR(aIPix);
          cPt2dr aRPixInit = aLoc2Init.Value(aRPix);

	  float aVal = mDIm.GetV(aIPix);
	  bool isW = ((aRPixInit.x()>=0) != (aRPixInit.y()>=0) );
	  float aValTheo = isW ?  aDCT.mVWhite : aDCT.mVBlack;

	  double aWeight = 1.0;
	  double aD1 =  aSeg1.Dist(aRPix);
	  double aD2 =  aSeg2.Dist(aRPix);
          double aMaxW= 1.0;
	  aWeight = std::min(   std::min(aD1,aMaxW ), std::min(aD2,aMaxW)) / aMaxW;

	  aSomWeight += aWeight;
	  aSomWEc +=  aWeight * std::abs(aValTheo-aVal);
 
          aMat.Add(aWeight,aVal,aValTheo);

          if  (Norm2(aPCr)>mRhoMin)
              UpdateMin(aCorMin,aMat.Correl());
     }

     aSomWEc /= aSomWeight;
     double aDev = aDCT.mVWhite - aDCT.mVBlack;
     aSomWEc /= aDev;

     aDCT.mScRadDir = aSomWEc;
     aDCT.mCorMinDir = aCorMin;

     if (0) 
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
