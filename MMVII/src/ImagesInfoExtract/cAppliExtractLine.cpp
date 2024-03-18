#include "MMVII_Linear2DFiltering.h"
#include "MMVII_Tpl_Images.h"
#include "MMVII_Sensor.h"
#include "MMVII_PCSens.h"


namespace MMVII
{
/* ************************************************************************ */
/*                                                                          */
/*                       cHoughTransform                                    */
/*                                                                          */
/* ************************************************************************ */

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
         cHoughTransform(const cPt2dr  & aSzIn,const cPt2dr & aMulTetaRho,const tREAL8 & aSigmTeta);
         void  AddPt(const cPt2dr & aPt,tREAL8 aTeta0,tREAL8 aWeight);
         cIm2D<tREAL4>      Accum() const;

	 tREAL8  AvgS2() const;
	 tREAL8  Max() const;

    private :
         inline tREAL8      I2Teta(int aK) const {return aK  *mFactI2T;}
         inline tREAL8      Teta2RInd(const tREAL8 & aTeta) const {return aTeta /mFactI2T;}
         inline tREAL8      Rho2RInd(const tREAL8 & aRho) const {return 1+ (aRho+mRhoMax) * mMulRho;}

         cPt2dr             mMil;
         tREAL8             mRhoMax;
         tREAL8             mMulTeta;
         tREAL8             mScaleTeta;
         tREAL8             mMulRho;
         tREAL8             mScaleRho;
         tREAL8             mSigmTeta;
         int                mNbTeta;
         tREAL8             mFactI2T ;
         int                mNbRho;

         cIm1D<tREAL8>      mTabSin;
         cDataIm1D<tREAL8>& mDTabSin;
         cIm1D<tREAL8>      mTabCos;
         cDataIm1D<tREAL8>& mDTabCos;
         cIm2D<tREAL4>      mAccum;
         cDataIm2D<tREAL4>& mDAccum;
};

tREAL8  cHoughTransform::AvgS2() const
{
   return std::sqrt(DotProduct(mDAccum,mDAccum) / mDAccum.NbElem());
}




cHoughTransform::cHoughTransform(const cPt2dr  & aSzIn,const cPt2dr &  aMulTetaRho,const tREAL8 & aSigmTeta) :
    mMil       (aSzIn / 2.0),
    mRhoMax    (Norm2(mMil)),
    mMulTeta   (aMulTetaRho.x()),
    mScaleTeta (mMulTeta * mRhoMax),
    mMulRho    (aMulTetaRho.y()),
    mScaleRho  (mMulRho * mRhoMax),
    mSigmTeta  (aSigmTeta),
    mNbTeta    (round_up(2*mScaleTeta*M_PI)),
    mFactI2T   ((2.0*M_PI)/mNbTeta),
    mNbRho     (2+round_up(2*mScaleRho)),
    mTabSin    (mNbTeta),
    mDTabSin   (mTabSin.DIm()),
    mTabCos    (mNbTeta),
    mDTabCos   (mTabCos.DIm()),
    mAccum     (cPt2di(mNbTeta,mNbRho),nullptr,eModeInitImage::eMIA_Null),
    mDAccum    (mAccum.DIm())
{
     for (int aKTeta=0 ; aKTeta<mNbTeta ; aKTeta++)
     {
          mDTabSin.SetV(aKTeta,std::sin(I2Teta(aKTeta)));
          mDTabCos.SetV(aKTeta,std::cos(I2Teta(aKTeta)));
     }
}

cIm2D<tREAL4>      cHoughTransform::Accum() const {return mAccum;}


void  cHoughTransform::AddPt(const cPt2dr & aPt,tREAL8 aTetaC,tREAL8 aWeight)
{
      int  iTeta0 = round_down(Teta2RInd(aTetaC-mSigmTeta));
      int  iTeta1 = round_up(Teta2RInd(aTetaC+mSigmTeta));

      if (iTeta0<0)
      {
           iTeta0 += mNbTeta;
           iTeta1 += mNbTeta;
           aTetaC += 2 * M_PI;
      }
      tREAL8 aX = aPt.x() - mMil.x();
      tREAL8 aY = aPt.y() - mMil.y();

      for (int iTeta=iTeta0 ;  iTeta<=iTeta1 ; iTeta++)
      {
           tREAL8 aTeta = I2Teta(iTeta);
           tREAL8 aWTot =    aWeight * ( 1 -    std::abs(aTeta-aTetaC) /mSigmTeta);
           if (aWTot>0)
           {
               int aITetaOK = iTeta%mNbTeta;
               //  (x,y) in L  <=> x cos(T) + y Sin(T) = R
               tREAL8 aRho =   aX * mDTabCos.GetV(aITetaOK) + aY*mDTabSin.GetV(aITetaOK) ;
               tREAL8  aRIndRho = Rho2RInd(aRho);
               int  aIRho0  = round_down(aRIndRho);
               int  aIRho1  = aIRho0 +1;
               tREAL8 aW0 = (aIRho1-aRIndRho);

               mDAccum.AddVal(cPt2di(aITetaOK,aIRho0),   aW0 *aWTot);
               mDAccum.AddVal(cPt2di(aITetaOK,aIRho1),(1-aW0)*aWTot);
           }
      }
}

/* ************************************************************************ */
/*                                                                          */
/*                       cImGradWithN                                       */
/*                                                                          */
/* ************************************************************************ */



template <class Type> class  cImGradWithN : public cImGrad<Type>
{
     public :
        cImGradWithN(const cPt2di & aSz);


        bool  IsMaxLoc(const cPt2di& aPix,const std::vector<cPt2di> &) const;

        static  std::vector<cPt2di>  NeighborsForMaxLoc(tREAL8 aRay,tREAL8 aRatioXY = 1.0);

        cIm2D<Type>      NormG() {return mNormG;}

	cPt2dr   AffineOpt(const cPt2dr &) const;
     private :
	cPt2dr   OneAffineOpt(const cPt2dr &) const;
 
        cIm2D<Type>       mNormG;
        cDataIm2D<Type>&  mDataNG;
};





template <class Type>   
  cImGradWithN<Type>::cImGradWithN(const cPt2di & aSz) :
     cImGrad<Type>  (aSz),
     mNormG         (aSz),
     mDataNG        (mNormG.DIm())
{
}

template<class Type> bool  cImGradWithN<Type>::IsMaxLoc(const cPt2di& aPix,const std::vector<cPt2di> & aVP) const
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

template<class Type> cPt2dr   cImGradWithN<Type>::OneAffineOpt(const cPt2dr & aP1) const
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
template<class Type> cPt2dr   cImGradWithN<Type>::AffineOpt(const cPt2dr & aP1) const
{
    return OneAffineOpt(OneAffineOpt(aP1));
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

template <class Type> class  cExtractLines
{
      public :
          typedef  cIm2D<Type>      tIm;
          typedef  cDataIm2D<Type>  tDIm;

          cExtractLines(tIm anIm);
          ~cExtractLines();

          void SetDericheGradAndMasq(tREAL8 aAlpha,tREAL8 aRayMaxLoc,int aBorder);
          void SetHough(const cPt2dr & aMulTetaRho,tREAL8 aSigmTeta,cPerspCamIntrCalib *,bool AffineMax);
          void ShowDetect(const std::string & aName);
          
          cHoughTransform &  Hough();
          cImGradWithN<Type> &  Grad();
      private :
          cPt2di                mSz;
          tIm                   mIm;
          cIm2D<tU_INT1>        mImMasq;
          int                   mNbContour;

          cImGradWithN<Type> *  mGrad;
          cHoughTransform    *  mHough;
          cPerspCamIntrCalib *  mCalib;
};


template <class Type> cExtractLines<Type>::cExtractLines(tIm anIm) :
       mSz       (anIm.DIm().Sz()),
       mIm       (anIm),
       mImMasq   (mSz,nullptr,eModeInitImage::eMIA_Null),
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
     mHough = new cHoughTransform(ToR(mSz),aMulTetaRho,aSigmTeta);

     tREAL8 aAvgIm=0;
     for (const auto & aPix :   mImMasq.DIm())
         aAvgIm += mGrad->NormG().DIm().GetV(aPix);
     aAvgIm /= mGrad->NormG().DIm().NbElem() ;
     StdOut() << "AVGGGGG " << aAvgIm  << "\n";
     
     tREAL8 aSomCorAff=0;  // sums the distance between point and its correction by dist
     tREAL8 aSomCorDist=0;  // sums the distance between point and its correction by dist
     tREAL8 aSomCorTeta=0;  // sums the distance between point and its correction by dist
     int aNbDone=0;



     for (const auto & aPix :   mImMasq.DIm())
     {
         if ( mImMasq.DIm().GetV(aPix))
         {
             if ((aNbDone%200000)==0) 
                StdOut() << "Remain to do " << mNbContour-aNbDone << "\n";
             aNbDone++;

             cPt2dr aRPix0 = ToR(aPix);
	     cPt2dr aRPix = aRPix0;
	     if (AffineMax)
                aRPix = mGrad->AffineOpt(aRPix0);
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
             if (mImMasq.DIm().InsideBL(aRPix))
                 mHough->AddPt(aRPix,aTeta,aW);
         }
     }
     ExpFilterOfStdDev(mHough->Accum().DIm(),4,1.0);
     mHough->Accum().DIm().ToFile("Accum.tif");
     {
        StdOut()  
                  << " , Aff=" <<       (aSomCorAff/aNbDone) 
                  << " , Dist-Pt=" <<   (aSomCorDist/aNbDone) 
                  << " , Dist-Teta=" << (aSomCorTeta/aNbDone) 
                  << std::endl;
     }
}

// cHoughTransform::cHoughTransform(const cPt2dr  & aSzIn,const tREAL8 &  aMul,const tREAL8 & aSigmTeta) :
// void  cHoughTransform::AddPt(const cPt2dr & aPt,tREAL8 aTetaC,tREAL8 aWeight)

template <class Type> void cExtractLines<Type>::SetDericheGradAndMasq(tREAL8 aAlpha,tREAL8 aRay,int aBorder)
{
     // Create the data for storing gradient
     mGrad = new cImGradWithN<Type>(mIm.DIm().Sz());
     //  compute the gradient & its norm using deriche method
     ComputeDericheAndNorm(*mGrad,mIm.DIm(),aAlpha);

     cRect2 aRect(mImMasq.DIm().Dilate(-aBorder));
     std::vector<cPt2di>  aVec = cImGradWithN<Type>::NeighborsForMaxLoc(aRay,1.1);

     mNbContour = 0;
     int aNbPt =0;
     for (const auto & aPix :  aRect)
     {
         aNbPt++;
         if (mGrad->IsMaxLoc(aPix,aVec))
         {
            mImMasq.DIm().SetV(aPix,255);
            mNbContour++;
         }
     }
     StdOut()<< " Prop Contour = " << mNbContour / double(aNbPt) << "\n";
}

template <class Type> void cExtractLines<Type>::ShowDetect(const std::string & aName)
{
     cRGBImage aImV(mIm.DIm().Sz());
     for (const auto & aPix :  mImMasq.DIm())
     {
         aImV.SetGrayPix(aPix,mIm.DIm().GetV(aPix));
         if (mImMasq.DIm().GetV(aPix))
         {
            tREAL8 aAlpha= 0.5;
            aImV.SetRGBPixWithAlpha(aPix,cRGBImage::Red,cPt3dr(aAlpha,aAlpha,aAlpha));
         }
     }
     aImV.ToJpgFileDeZoom(aName,1);
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
    mThreshCpt      {100,200,400,600}
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
	StdOut() << "VMAX=" << aVMax << std::endl;
        for (size_t aK=0 ; aK<mThreshCpt.size() ; aK++)
            StdOut() << " Cpt=" << aVCpt.at(aK) << " for threshold " << mThreshCpt.at(aK) << std::endl;

        anExtrL.ShowDetect("tutu.tif");
	
	anExtrL.Grad().NormG().DIm().ToFile("Grad.tif");
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
