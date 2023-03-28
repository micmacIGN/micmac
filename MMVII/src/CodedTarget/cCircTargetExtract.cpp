#include "MMVII_Tpl_Images.h"
#include "MMVII_Linear2DFiltering.h"
#include "MMVII_Geom2D.h"
#include "MMVII_Sensor.h"
#include "MMVII_TplImage_PtsFromValue.h"
#include "MMVII_ImageInfoExtract.h"
#include "CodedTarget.h"
#include "CodedTarget_Tpl.h"
#include "MMVII_2Include_Serial_Tpl.h"

/*   Modularistion
 *   Code extern tel que ellipse
 *   Ellipse => avec centre
 *   Pas de continue
 */

namespace MMVII
{

/**   Class for vehiculing all the threshold parameters relative to circ target extraction
 */
struct cThresholdCircTarget
{
    public :
        cThresholdCircTarget();
	void SetMax4Inv(tREAL4 aMaxGray);

	//   Coding part
        tREAL8 mRatioStdDevGlob;   /// Ratio std-dev of individuel part / global std dev
        tREAL8 mRatioStdDevAmpl;   /// Ratio std-dev of individuel part / amplitude of Black&White
        tREAL8 mAngRadCode;        /// Angle of radial grad  on coding part
        tREAL8 mAngTanCode;        /// Angle of tangential radian on coding part
};

cThresholdCircTarget::cThresholdCircTarget() :
    mRatioStdDevGlob  (0.1),
    mRatioStdDevAmpl  (0.05),
    mAngRadCode       (0.15),
    mAngTanCode       (0.15)
{
}

using namespace cNS_CodedTarget;

namespace  cNS_CodedTarget
{


/* ********************************************* */
/*                                               */
/*                cCircTargExtr                  */
/*                                               */
/* ********************************************* */

/**    Store the result of a validated extracted circular target
 */

class cCircTargExtr : public cBaseTE
{
     public :
         cCircTargExtr(const cExtractedEllipse &);

         cEllipse         mEllipse;
	 // tREAL8           mVBlack;
	 // tREAL8           mVWhite;
	 bool             mMarked4Test;
	 bool             mWithCode;
	 cOneEncoding     mEncode;
};

cCircTargExtr::cCircTargExtr(const cExtractedEllipse & anEE)  :
	cBaseTE      (anEE.mEllipse.Center(),anEE.mSeed.mBlack,anEE.mSeed.mWhite),
	mEllipse     (anEE.mEllipse),
	// mVBlack       (anEE.mSeed.mBlack),
	// mVWhite       (anEE.mSeed.mWhite),
	mMarked4Test (anEE.mSeed.mMarked4Test),
	mWithCode    (false)
{
}


/* ********************************************* */
/*                                               */
/*                cCCDecode                      */
/*                                               */
/* ********************************************* */

/**  Class for computing the circular code: 

        *  make a polar representation , evaluate if it
        *  find the phase that maximize the standrd dev insid each interval
        *  decide code/non code regarding the average std dev
        *  compute the code

 */

class cCCDecode
{
    public :
         typedef cDataIm2D<tREAL4>  tDIm;
         typedef const tDIm   tCDIm;



         cCCDecode(cCircTargExtr & anEE,tCDIm & aDIm,tCDIm & aDGx , tCDIm & aDGy,const cFullSpecifTarget &,const cThresholdCircTarget &);

	 void Show(const std::string & aPrefix);

         /// Compute phase minimizing standard deviation, make a decision if its low enough
	 void ComputePhaseTeta() ;

         ///  Compute de binary flag, try to interpret as a code, eventually memorize in mEE
	 void ComputeCode();

    private :

	      //  Aggregation
	 tREAL8 StdDev(int aK1,int aK2) const;      ///< standard deviation of the interval
	 tREAL8 Avg(int aK1,int aK2) const;         ///< average  of the interval
	 tREAL8 TotalStdDevOfPhase(int aK0) const;  ///< Sum of standard dev, on all interval, for a given stard
         /// Used to compute the total deviation of black or white
         tREAL8 StdDevOfSumInterv(const std::vector<cPt2di> &);
         /// Add value of interval to dev structure
         void  AddStdDev(int aK1,int aK2,cComputeStdDev<tREAL8> & aCS) const;

         tREAL8 ScoreTransition(tREAL8 aPhase) const; ///< For a given angle how close are we to BW-average
         tREAL8 GlobalScorePhase(tREAL8 aPhase) const; ///< For a given phase, sum of score transition on all frontiers
         tREAL8 RefinePhase(tREAL8 aPhase0) const;  ///< Optimize phase to minimize GlobalScorePhase

	     // Geometric correspondances 
         tREAL8 K2Rho (int aK) const;   /// index of rho  2 real rho
         tREAL8 K2Teta(tREAL8 aK) const;   /// index of teta  2 real teta
         int Rho2K (tREAL8 aR) const;   ///  real rho 2 index of rho
	 cPt2dr  KTetaRho2Im(const cPt2di & aKTetaRho) const;   /// index rho-teta  2   cartesian coordinates 
         /// For weight in [0,1] return a rho corresponding to coding place
	 tREAL8 CodingRhoOfWeight(const tREAL8 &) const;

         int KBeginInterv(int aK0,int aNumBit) const;
         int KEndInterv(int aK0,int aNumBit) const;


         cCircTargExtr &           mEE;
         const cDataIm2D<tREAL4> & mDIm;
         const cDataIm2D<tREAL4> & mDGx;
         const cDataIm2D<tREAL4> & mDGy;
	 const cFullSpecifTarget & mSpec;
	 cThresholdCircTarget      mThresh;


	 bool                      mOK;
	 int                       mNbB;     ///< number of bits in code
	 int                       mTetaWanted;
	 const int                 mPixPerB; ///< number of pixel for each bit to decode
	 const int                 mNbRho;   ///< number of pixel for rho
	 int                       mNbTeta;  ///< number of pixel for each bit to decode
	 tREAL8                    mRho0;
	 tREAL8                    mRho1;
         cIm2D<tREAL4>             mImPolar;
         cDataIm2D<tREAL4> &       mDIP;
         cIm1D<tREAL4>             mAvg;
         cDataIm1D<tREAL4> &       mDAvg;
	 int                       mKR0;
	 int                       mKR1;
	 int                       mIPhase0;
	 tREAL8                    mRPhase0;
         std::vector<cPt2di>       mVInt0;
         std::vector<cPt2di>       mVInt1;
	 tREAL8                    mBlack;
	 tREAL8                    mWhite;
	 tREAL8                    mBWAmpl;
	 tREAL8                    mBWAvg;
	 const cOneEncoding *      mEnCode;
	 bool                      mOkGrad;
         bool                      mMarked4Test;
};

    // ==============   constructor ============================
   

//  mTetaWanted = mPixPerB * mNbB

cCCDecode::cCCDecode
(
   cCircTargExtr & anEE,
   tCDIm & aDIm,
   tCDIm & aDGx,
   tCDIm & aDGy,
   const cFullSpecifTarget & aSpec,
   const cThresholdCircTarget & aThresh
) :
	mEE          (anEE),
	mDIm         (aDIm),
        mDGx         (aDGx),
        mDGy         (aDGy),
	mSpec        (aSpec),
	mThresh      (aThresh),
	mOK          (true),
	mNbB         (mSpec.NbBits()),
	mTetaWanted  (round_up(2*M_PI* mEE.mEllipse.LGa())* 0.666 * (mSpec.Rho_1_BeginCode()/ mSpec.Rho_0_EndCCB())),
	mPixPerB     (std::max(6,DivSup(mTetaWanted,mNbB))),
	mNbRho       (20),
	mNbTeta      (mPixPerB * mNbB),
	mRho0        ((mSpec.Rho_0_EndCCB()+mSpec.Rho_1_BeginCode()) /2.0),
	mRho1        (mSpec.Rho_2_EndCode() +0.2),
	mImPolar     (cPt2di(mNbTeta,mNbRho)),
        mDIP         (mImPolar.DIm()),
	mAvg         ( mNbTeta,nullptr,eModeInitImage::eMIA_Null ),
	mDAvg        ( mAvg.DIm()),
	mKR0         ( Rho2K(CodingRhoOfWeight(0.25)) ) ,
	mKR1         ( Rho2K(CodingRhoOfWeight(0.75)) ) ,
	mIPhase0     (-1),
	mRPhase0     (-1),
	mBlack       (mEE.mVBlack),
	mWhite       (mEE.mVWhite),
	mBWAmpl      (mWhite-mBlack),
	mBWAvg       ((mBlack+mWhite)/2.0),
	mEnCode      (nullptr),
        mMarked4Test (anEE.mMarked4Test)
{
    if (mMarked4Test)  StdOut() << "ENTER MARKED " << mTetaWanted << "\n";
    //  compute a polar image
    for (int aKTeta=0 ; aKTeta < mNbTeta; aKTeta++)
    {
        for (int aKRho=0 ; aKRho < mNbRho; aKRho++)
        {
		cPt2dr aPt = KTetaRho2Im(cPt2di(aKTeta,aKRho));
		tREAL8 aVal = mDIm.DefGetVBL(aPt,-1);
		if (aVal<0)
		{
                   mOK=false;
                   return;
		}

		mDIP.SetV(cPt2di(aKTeta,aKRho),aVal);
        }
    }

    if (!mOK)
       return;

    // compute an image
    for (int aKTeta=0 ; aKTeta < mNbTeta; aKTeta++)
    {
        std::vector<tREAL8> aVGray;
        for (int aKRho=mKR0 ; aKRho <= mKR1; aKRho++)
	{
            aVGray.push_back(mDIP.GetV(cPt2di(aKTeta,aKRho)));
	}
        mDAvg.SetV(aKTeta,NonConstMediane(aVGray));
    }

    ComputePhaseTeta() ;
    if (!mOK) 
    {
        if (mMarked4Test)  StdOut() << "REFUTED AFTER ComputePhaseTeta\n";
        return;
    }

    ComputeCode();
    if (!mOK)
    {
        if (mMarked4Test)  StdOut() << "REFUTED AFTER ComputeCode\n";
        return;
    }
}

//  =============   Agregation on interval : StdDev , Avg, TotalStdDevOfPhase ====

void  cCCDecode::AddStdDev(int aK1,int aK2,cComputeStdDev<tREAL8> & aCS) const
{
    for (int aK=aK1 ; aK<aK2 ; aK++)
    {
         aCS.Add(mDAvg.GetV(aK%mNbTeta));
    }
}

tREAL8 cCCDecode::StdDev(int aK1,int aK2) const
{
    cComputeStdDev<tREAL8> aCS;
    AddStdDev(aK1,aK2,aCS);
    return aCS.StdDev(0);
}

tREAL8 cCCDecode::Avg(int aK1,int aK2) const
{
    tREAL8 aSom =0 ;
    for (int aK=aK1 ; aK<aK2 ; aK++)
    {
          aSom += mDAvg.GetV(aK%mNbTeta);
    }
    return aSom / (aK2-aK1);
}

int cCCDecode::KBeginInterv(int aK0,int aNumBit) const { return  aK0+aNumBit*mPixPerB +1 ; }
int cCCDecode::KEndInterv(int aK0,int aNumBit) const { return  aK0+aNumBit*mPixPerB -1 ; }

tREAL8  cCCDecode::StdDevOfSumInterv(const std::vector<cPt2di> & aVInterv)
{
   cComputeStdDev<tREAL8> aCS;
   for (const auto & anI : aVInterv)
       AddStdDev( KBeginInterv(mIPhase0,anI.x()), KEndInterv(mIPhase0,anI.y()), aCS);

    return aCS.StdDev(0);
}

tREAL8 cCCDecode::TotalStdDevOfPhase(int aK0) const
{
    tREAL8 aSum=0;
    for (int aKBit=0 ; aKBit<mNbB ; aKBit++)
    {
         aSum +=  StdDev(KBeginInterv(aK0,aKBit),KEndInterv(aK0,aKBit+1));
    }
    return aSum / mNbB;
}

// =====  Geometric correspondance between indexes, polar, cartesian ....

cPt2dr cCCDecode::KTetaRho2Im(const cPt2di & aKTR) const
{
     return mEE.mEllipse.PtOfTeta(K2Teta(aKTR.x()),K2Rho(aKTR.y()));
}

tREAL8 cCCDecode::K2Rho(const int aK)  const {return mRho0+ ((mRho1-mRho0)*aK) / mNbRho;}
tREAL8 cCCDecode::K2Teta(const tREAL8 aK) const {return  (2*M_PI*aK)/mNbTeta;}

int  cCCDecode::Rho2K(const tREAL8 aR)  const 
{
     return round_ni( ((aR-mRho0)/(mRho1-mRho0)) * mNbRho );
}

tREAL8 cCCDecode::CodingRhoOfWeight(const tREAL8 & aW) const
{
	return (1-aW) * mSpec.Rho_1_BeginCode() + aW * mSpec.Rho_2_EndCode();
}

/// Difference between interpolated value and theoreticall gray
tREAL8 cCCDecode::ScoreTransition(tREAL8 aXTrans) const
{
    return std::abs(mDAvg.GetVAndGradCircBL(aXTrans).x() - mBWAvg);
}

/// For a given phase, sum on all transition 0/1 or 1/0 of ScoreTransition
tREAL8 cCCDecode::GlobalScorePhase(tREAL8 aPhase) const
{
    tREAL8 aSum =0;
    for (size_t aKI=0 ; aKI<mVInt0.size() ; aKI++)
    {
        aSum +=  ScoreTransition(aPhase+mVInt0[aKI].x()*mPixPerB);
        aSum +=  ScoreTransition(aPhase+mVInt1[aKI].x()*mPixPerB);
    }
    return aSum / (2.0*mVInt0.size());
}

/// Optimize the phase to minize ScoreTransition, 
tREAL8 cCCDecode::RefinePhase(tREAL8 aPhase0) const
{
     for (tREAL8 aStep=1.0 ; aStep >0.05 ; aStep /= 1.666)  // sparse different scale
     {
         cWhichMin<tREAL8,tREAL8>  aWMin;
	 for (int aK=-1 ; aK<=1 ; aK++)  // for a given step "S"  test {-S,0,+S}
	 {
             tREAL8 aPhase = aPhase0 + aK*aStep;
             aWMin.Add(aPhase,GlobalScorePhase(aPhase));
	 }
	 aPhase0 = aWMin.IndexExtre();  // update phase with best one
     }
     return aPhase0;
}

//=================

void cCCDecode::ComputePhaseTeta() 
{

    // Extract phase minimizing the standard dev on all intervall
    cWhichMin<int,tREAL8> aMinDev;
    for (int aK0=0 ;aK0< mPixPerB ; aK0++)
	    aMinDev.Add(aK0,TotalStdDevOfPhase(aK0));
    mIPhase0 = aMinDev.IndexExtre();

    if (mMarked4Test)
    {
            StdOut() << "Ratio StdDev : Glob " << aMinDev.ValExtre() / StdDev(0,mNbTeta)
		     <<  " BW : " << aMinDev.ValExtre() /  mBWAmpl
		     << "\n";
    }

    //  decide if sufficiently homogeneous
    if (     (aMinDev.ValExtre() > mThresh.mRatioStdDevGlob * StdDev(0,mNbTeta))
          || (aMinDev.ValExtre() > mThresh.mRatioStdDevAmpl*  mBWAmpl)
       )
    {
        if (mMarked4Test)
	{
            StdOut() << "Bad ratio StdDev : Glob \n";
	}
        mOK = false;
	return;
    }
}

void cCCDecode::ComputeCode()
{
    // compute flag of bit
    size_t aFlag=0;
    for (int aKBit=0 ; aKBit<mNbB ; aKBit++)
    {
       tREAL8  aMoy = Avg(KBeginInterv(mIPhase0,aKBit), KEndInterv(mIPhase0,aKBit+1));

       if (mSpec.BitIs1(aMoy>mBWAvg))
           aFlag |= (1<<aKBit);
    }

    //  flag for coding must be eventually inverted, depending of orientation convention
    {
        size_t aFlagCode = aFlag;
        if (! mSpec.AntiClockWiseBit())
           aFlagCode = BitMirror(aFlag,1<<mSpec.NbBits());

        mEnCode = mSpec.EncodingFromCode(aFlagCode);

        if (! mEnCode) return;
    }

     // Make supplementary test 
    MaxRunLength(aFlag,1<<mNbB,mVInt0,mVInt1);

    // Test were made to compute the global deviation on black/white part, but not concluding as
    // on some scene there is a bias that creat smooth variation;  if want to use it modelize the bias ?
    if (0)
    {
         tREAL8 aDev0 = StdDevOfSumInterv(mVInt0);
         tREAL8 aDev1 = StdDevOfSumInterv(mVInt1);
         StdOut()  << mEnCode->Name() <<  " D0=" << aDev0/ mBWAmpl <<  " D1=" << aDev1/ mBWAmpl <<  "\n";
    }

    mRPhase0 = RefinePhase(mIPhase0);
    // StdOut() <<"PHASE "<< mRPhase0 -mIPhase0 <<" SC "<< GlobalScorePhase(mIPhase0)<<" => "<<GlobalScorePhase(mRPhase0)<<" "<< "\n";
    // tREAL8 cCCDecode::RefinePhase(tREAL8 aPhase0) const
    // tREAL8 cCCDecode::GlobalScorePhase(tREAL8 aPhase) const


    //  ==========  Verification by the gradient on coding part ===============
    {
        const std::vector<cPt2di> &  aVFG =  mSpec.ZeroIsBackGround() ? mVInt1  : mVInt0;
	//  WhiteBackGround

        tREAL8 aEpsTeta = 0.1;  // to keep away from corners
	cWeightAv<tREAL8>  aAvgRad;
	cWeightAv<tREAL8>  aAvgTan;
        mOkGrad=true;
        double  aThickCode = ( mSpec.Rho_2_EndCode() -mSpec.Rho_1_BeginCode()) * mEE.mEllipse.LSa();

	bool BugR = MatchRegex(mEnCode->Name(),"XXXXXXXXXXXXX");
	for (const auto & aI : aVFG)
	{
             MMVII_INTERNAL_ASSERT_tiny(aI.x()<aI.y(),"Bads assert in interval");
	     tREAL8 aTeta1 =  K2Teta(mRPhase0+aI.x()*mPixPerB);  // Teta begin coding slot (just on corners)
	     tREAL8 aTeta2 =  K2Teta(mRPhase0+aI.y()*mPixPerB);  // Teta end  coding slot (just on corners)

	     //=======  Tangential Part ==================================
	     for (bool aBeginTeta : {true,false})
	     {

	          int aNbRho  = round_ni(aThickCode/3.0);
		  aNbRho = std::max(2,std::min(10,aNbRho));
		  tREAL8 aTeta = aBeginTeta ? aTeta1 : aTeta2;
		  cPt2dr aDirRad  =  mEE.mEllipse.PtOfTeta(aTeta) -mEE.mEllipse.Center();
		  aDirRad = aDirRad *cPt2dr(0,aBeginTeta ? 1 : -1);

	          for (int aKRho=1 ; aKRho <= aNbRho-1   ; aKRho++)  // again avoid corner
	          {
	               tREAL8  aRho = CodingRhoOfWeight(aKRho/tREAL8(aNbRho));
                       cPt2dr aPIm = mEE.mEllipse.PtOfTeta(aTeta,aRho);
                       if (mDGx.InsideBL(aPIm))
                       {
                           cPt2dr aGRadIm(mDGx.GetVBL(aPIm),mDGy.GetVBL(aPIm)); // grad of image
		           tREAL8 aTeta = std::abs(ToPolar(aGRadIm/aDirRad).y());
			   aAvgTan.Add(1.0,aTeta);
			   if (BugR)  StdOut() << "TETAT= " << aTeta << "  GIM=" << aGRadIm   << " DRad=" << aDirRad  << "\n";
                       }
                       else
                       {
                          mOkGrad=false;
                       }
	          }
	     }

	     //=======  Radial Part ==================================
	     tREAL8 aEpsTetaLoc =  std::min(aEpsTeta,(aTeta2-aTeta1)/ 4.0); // adaptative precaution for corners

	     aTeta1 +=aEpsTetaLoc;   // Teta begin corrected
	     aTeta2 -=aEpsTetaLoc;   // Teta end corrected

	     int aNbTeta = std::max(5,round_up((aTeta2-aTeta1)/0.1));  // Magic formula

	     for (int aKTeta=1 ; aKTeta<= (aNbTeta-1)  ; aKTeta++)  // again avoid corner
	     {
                  tREAL8 aTeta = aTeta1 + (aTeta2-aTeta1)* (aKTeta/double(aNbTeta));  // sampling teta
                  cPt2dr  aGrad;
                  cPt2dr  aPIm = mEE.mEllipse.PtAndGradOfTeta(aTeta,aGrad,mSpec.Rho_1_BeginCode()); // theoreticall grad

                  if (mDGx.InsideBL(aPIm))
                  {
                      cPt2dr aGRadIm(mDGx.GetVBL(aPIm),mDGy.GetVBL(aPIm)); // grad of image
		      aAvgRad.Add(1.0,AbsAngleTrnk(aGRadIm,aGrad));
                  }
                  else
                  {
                     mOkGrad=false;
                  }
	     }
	}
	if ((aAvgRad.Average()>mThresh.mAngRadCode) || (aAvgTan.Average()>mThresh.mAngTanCode))
	{
           mOkGrad=false;
	}

        if (mMarked4Test)
	{
	       StdOut() << " "  << mEnCode->Name() 
		       << " PBB " << mPixPerB 
		       << " Rad:" <<  aAvgRad.Average()  
		       << " Tan:" << aAvgTan.Average() 
		       << " Th=" << aThickCode<< "\n"; 
	}
	// getchar();
    }

    if (mOkGrad)
    {
       mEE.mWithCode = true;
       mEE.mEncode = cOneEncoding(mEnCode->Num(),mEnCode->Code(),mEnCode->Name());
    }
}





void  cCCDecode::Show(const std::string & aPrefix)
{
    static int aCpt=0; aCpt++;

    cRGBImage  aIm = RGBImFromGray(mImPolar.DIm(),1.0,9);

    if (mIPhase0>=0)
    {
       for (int aKBit=0 ; aKBit<mNbB ; aKBit++)
       {
           tREAL8 aK1 = mIPhase0+aKBit*mPixPerB -0.5;

	   aIm.DrawLine(cPt2dr(aK1,0),cPt2dr(aK1,mNbTeta),cRGBImage::Red);

       }
    }

    aIm.ToFile(aPrefix + "_ImPolar_"+ToStr(aCpt)+".tif");

    StdOut() << "Adr=" << mEnCode << " ";
    if (mEnCode) 
    {
       StdOut() << " Name=" << mEnCode->Name()  
                << " Code=" <<  mEnCode->Code() 
                << " BF=" << StrOfBitFlag(mEnCode->Code(), 1<<mNbB);
    }
    StdOut() << "\n";
}


};

/*  *********************************************************** */
/*                                                              */
/*             cAppliExtractCodeTarget                          */
/*                                                              */
/*  *********************************************************** */

class cAppliExtractCircTarget : public cMMVII_Appli,
	                        public cAppliParseBoxIm<tREAL4>
{
     public :
        typedef tREAL4              tElemIm;
        typedef cDataIm2D<tElemIm>  tDataIm;
        typedef cImGrad<tElemIm>    tImGrad;


        cAppliExtractCircTarget(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);

     private :
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;

        int ExeOnParsedBox() override;

	void MakeImageLabel();
	void MakeImageFinalEllispe();

	void TestOnSimul();
	void DoExport();

	std::string         mNameSpec;
	cFullSpecifTarget * mSpec;
        bool                  mVisuLabel;
        bool                  mVisuElFinal;
        cExtract_BW_Ellipse * mExtrEll;
        cParamBWTarget  mPBWT;

        cIm2D<tU_INT1>  mImMarq;

        std::vector<cCircTargExtr*>  mVCTE;
	cPhotogrammetricProject     mPhProj;

	std::string                 mPrefixOut;
	bool                        mHasMask;
	std::string                 mNameMask;
	std::string                 mPatHihlight;
	bool                        mUseSimul; 
        cResSimul                   mResSimul;
	double                      mRatioDMML;
        cThresholdCircTarget        mThresh;

	std::vector<const cGeomSimDCT*>     mGTMissed;
	std::vector<const cCircTargExtr*>   mFalseExtr;

};



cAppliExtractCircTarget::cAppliExtractCircTarget
(
    const std::vector<std::string> & aVArgs,
    const cSpecMMVII_Appli & aSpec
) :
   cMMVII_Appli  (aVArgs,aSpec),
   cAppliParseBoxIm<tREAL4>(*this,true,cPt2di(20000,20000),cPt2di(300,300),false) ,
   mSpec         (nullptr),
   mVisuLabel    (false),
   mVisuElFinal  (false),
   mExtrEll      (nullptr),
   mImMarq       (cPt2di(1,1)),
   mPhProj       (*this),
   mPatHihlight  ("XXXXX"),
   mUseSimul     (false),
   mRatioDMML    (1.5)
{
}

        // cExtract_BW_Target * 
cCollecSpecArg2007 & cAppliExtractCircTarget::ArgObl(cCollecSpecArg2007 & anArgObl)
{
   // Standard use, we put args of  cAppliParseBoxIm first
   return
             APBI_ArgObl(anArgObl)
        <<   Arg2007(mNameSpec,"XML name for bit encoding struct",{{eTA2007::XmlOfTopTag,cFullSpecifTarget::TheMainTag}})
   ;
}

cCollecSpecArg2007 & cAppliExtractCircTarget::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return APBI_ArgOpt
          (
                anArgOpt
             << mPhProj.DPMask().ArgDirInOpt("TestMask","Mask for selecting point used in detailed mesg/output")
             << AOpt2007(mPBWT.mMinDiam,"DiamMin","Minimum diameters for ellipse",{eTA2007::HDV})
             << AOpt2007(mPBWT.mMaxDiam,"DiamMax","Maximum diameters for ellipse",{eTA2007::HDV})
             << AOpt2007(mRatioDMML,"RDMML","Ratio Distance minimal bewteen local max /Diam min ",{eTA2007::HDV})
             << AOpt2007(mVisuLabel,"VisuLabel","Make a visualisation of labeled image",{eTA2007::HDV})
             << AOpt2007(mVisuElFinal,"VisuEllipse","Make a visualisation extracted ellispe & target",{eTA2007::HDV})
             << AOpt2007(mPatHihlight,"PatHL","Pattern for highliting targets in visu",{eTA2007::HDV})
	     <<   mPhProj.DPPointsMeasures().ArgDirOutOptWithDef("Std")
          );
}

void cAppliExtractCircTarget::DoExport()
{
     cSetMesPtOf1Im  aSetM(FileOfPath(mNameIm));
     for (const auto & anEE : mVCTE)
     {
         if (anEE->mWithCode)  
         {
             aSetM.AddMeasure(cMesIm1Pt(anEE->mPt,anEE->mEncode.Name(),1.0));
         }
     }

     mPhProj.SaveMeasureIm(aSetM);
}

void cAppliExtractCircTarget::MakeImageFinalEllispe()
{
   cRGBImage   aImVisu=  cRGBImage::FromFile(mNameIm,CurBoxIn());

   cPt2dr  aSz(50,50);
   cPt3dr aAlpha(0.7,0.7,0.7);

   if (mUseSimul)
   {
      for (const auto & aGT :  mGTMissed)
      {
          if (aGT->mResExtr ==nullptr)
             aImVisu.FillRectangle(cRGBImage::Red,ToI(aGT->mC-aSz),ToI(aGT->mC+aSz),aAlpha);
      }
      for (const auto & anEE : mVCTE)
      {
          if ((anEE->mWithCode)  && (anEE->mGT ==nullptr))
          {
              aImVisu.FillRectangle(cRGBImage::Green,ToI(anEE->mPt-aSz),ToI(anEE->mPt+aSz),aAlpha);
          }
      }
   }

   for (const auto & anEE : mVCTE)
   {
        const cEllipse &   anEl  = anEE->mEllipse;
	bool doHL = MatchRegex(anEE->mEncode.Name(),mPatHihlight);
        for (tREAL8 aMul = 1.0; aMul < (doHL ? 4.0 : 1.5); aMul += (doHL ? 0.05 : 0.2))
        {
            aImVisu.DrawEllipse
            (
               cRGBImage::Blue ,  // anEE.mWithCode ? cRGBImage::Blue : cRGBImage::Red,
               anEl.Center(),
               anEl.LGa()*aMul , anEl.LSa()*aMul , anEl.TetaGa()
            );
        }
	if (anEE->mWithCode)
        {
             aImVisu.DrawString
             (
                  anEE->mEncode.Name(),cRGBImage::Red,
		  anEl.Center(),cPt2dr(0.5,0.5),
		  3
             );

	}
   }

    aImVisu.ToFile(mPrefixOut + "_VisuEllipses.tif");
}

void cAppliExtractCircTarget::MakeImageLabel()
{
    cRGBImage   aImVisuLabel =  cRGBImage::FromFile(mNameIm,CurBoxIn());
    const cExtract_BW_Target::tDImMarq&     aDMarq =  mExtrEll->DImMarq();
    for (const auto & aPix : aDMarq)
    {
         if (aDMarq.GetV(aPix)==tU_INT1(eEEBW_Lab::eTmp))
            aImVisuLabel.SetRGBPix(aPix,cRGBImage::Green);

         if (     (aDMarq.GetV(aPix)==tU_INT1(eEEBW_Lab::eBadZ))
               || (aDMarq.GetV(aPix)==tU_INT1(eEEBW_Lab::eElNotOk))
            )
            aImVisuLabel.SetRGBPix(aPix,cRGBImage::Blue);
         if (aDMarq.GetV(aPix)==tU_INT1(eEEBW_Lab::eBadFr))
            aImVisuLabel.SetRGBPix(aPix,cRGBImage::Cyan);
         if (aDMarq.GetV(aPix)==tU_INT1(eEEBW_Lab::eBadEl))
            aImVisuLabel.SetRGBPix(aPix,cRGBImage::Red);
         if (aDMarq.GetV(aPix)==tU_INT1(eEEBW_Lab::eAverEl))
            aImVisuLabel.SetRGBPix(aPix,cRGBImage::Orange);
         if (aDMarq.GetV(aPix)==tU_INT1(eEEBW_Lab::eBadTeta))
            aImVisuLabel.SetRGBPix(aPix,cRGBImage::Yellow);
    }

    for (const auto & aSeed : mExtrEll->VSeeds())
    {
        if (aSeed.mOk)
        {
           aImVisuLabel.SetRGBPix(aSeed.mPixW,cRGBImage::Red);
           aImVisuLabel.SetRGBPix(aSeed.mPixTop,cRGBImage::Yellow);
        }
        else
        {
           aImVisuLabel.SetRGBPix(aSeed.mPixW,cRGBImage::Yellow);
        }
    }
    aImVisuLabel.ToFile(mPrefixOut + "_Label.tif");
}


void cAppliExtractCircTarget::TestOnSimul()
{
      AllMatchOnGT(mResSimul,mVCTE,2.0,false,[](const auto& aPEE){return aPEE->mWithCode;});

     std::vector<tREAL8>  aVErr;
     // Analyse the ground truth to detect omited + prepare statistic on good match
     for (const auto & aRS : mResSimul.mVG)
     {
         if (aRS.mResExtr == nullptr)
	 {
             if (mGTMissed.empty())   
                StdOut () << "============= MISSED TARGET ============\n";
             StdOut () << "  * " << aRS.mEncod.Name() << "\n";
             mGTMissed.push_back(&aRS);
	 }
	 else
	 {
		 aVErr.push_back(Norm2(aRS.mResExtr->mPt-aRS.mC));
	 }
     }

     // analyse detected target to get false extracted
     for (const auto & anEE : mVCTE)
     {
         if ((anEE->mWithCode)  && (anEE->mGT ==nullptr))
         {
             if (mFalseExtr.empty())   
                StdOut () << "============= FALSE EXTRACTION ============\n";
             StdOut () << "  * " << anEE->mEncode.Name() << "\n";
             mFalseExtr.push_back(anEE);
         }
     }

     if (aVErr.size())
     {
         StdOut()  <<  "==============  ERROR SATISTICS ===================\n";
         StdOut()  <<  "AVERAGE = " << Average(aVErr) << "\n";

         for (const auto & aProp : {0.5,0.75,0.9})
             StdOut()  << "  * Er at " << aProp << " = " << Cst_KthVal(aVErr,aProp)  << "\n";
     }	
     else
     {
         StdOut()  <<  "  ==============  NOT ANY MATCH !!!! ===================\n";
     }
}


int cAppliExtractCircTarget::ExeOnParsedBox()
{
   mPBWT.mDistMinMaxLoc =  mPBWT.mMinDiam * mRatioDMML;
   // All the process has been devloppe/tested using target with black background, rather than revisiting
   // all the process to see where the varaiant black/white has to be adressed, I do it "quick and (not so) dirty",
   // by inverting the image at the beging of process if necessary
   if (mSpec->WhiteBackGround())
   {
      mSpec->SetWhiteBackGround(false);
      tREAL4 aVMin,aVMax;

      tDataIm & aDIm = APBI_DIm();
      GetBounds(aVMin,aVMax,aDIm);

      for (const auto & aPix : aDIm)
      {
          aDIm.SetV(aPix,aVMax-aDIm.GetV(aPix));
      }

      mPBWT.SetMax4Inv(aVMax);
   }
   double aT0 = SecFromT0();

   mExtrEll = new cExtract_BW_Ellipse(APBI_Im(),mPBWT,mPhProj.MaskWithDef(mNameIm,CurBoxIn(),false));

   double aT1 = SecFromT0();
   mExtrEll->ExtractAllSeed();
   double aT2 = SecFromT0();
   mExtrEll->AnalyseAllConnectedComponents(mNameIm);
   double aT3 = SecFromT0();

   if (mVisuElFinal)
   {
       StdOut() << "TIME-INIT " << aT1-aT0 << "\n";
       StdOut() << "TIME-SEED " << aT2-aT1 << "\n";
       StdOut() << "TIME-CC   " << aT3-aT2 << "\n";
   }

   for (const auto & anEE : mExtrEll->ListExtEl() )
   {
       if (anEE.mSeed.mMarked4Test)
          anEE.ShowOnFile(mNameIm,21,mPrefixOut);
       if (anEE.mValidated  || anEE.mSeed.mMarked4Test)
       {
	  mVCTE.push_back(new cCircTargExtr(anEE));
       }
   }

   for (auto & anEE : mVCTE)
   {
       cCCDecode aCCD(*anEE,APBI_DIm(),mExtrEll->DGx(),mExtrEll->DGy(),*mSpec,mThresh);
       if (anEE->mMarked4Test)
       {
	     aCCD.Show(mPrefixOut);
       }
   }

   if (mUseSimul)
   {
      TestOnSimul();
   }

   if (mVisuLabel)
   {
      MakeImageLabel();
   }

   if (mVisuElFinal)
   {
      MakeImageFinalEllispe();
   }

   DoExport();


   delete mExtrEll;

   return EXIT_SUCCESS;
}



int  cAppliExtractCircTarget::Exe()
{
   mPhProj.FinishInit();


    mPhProj.GetMetaData(mNameIm);


   if (RunMultiSet(0,0))  // If a pattern was used, run in // by a recall to itself  0->Param 0->Set
   {
      return ResultMultiSet();
   }

   mPrefixOut = "CircTarget_" +  LastPrefix(APBI_NameIm());

   if (! IsInit(&mUseSimul))
   {
       mUseSimul = MatchRegex(mNameIm,ThePrefixSimulTarget+".*");
   }
   if (mUseSimul)
   {
       std::string aNameResSim = LastPrefix(APBI_NameIm()) + ThePostfixGTSimulTarget;
       mResSimul = cResSimul::FromFile(aNameResSim);
   }

   mSpec = cFullSpecifTarget::CreateFromFile(mNameSpec);

   // mPhProj.FinishInit();

   mHasMask =  mPhProj.ImageHasMask(APBI_NameIm()) ;
   if (mHasMask)
   {
      mNameMask =  mPhProj.NameMaskOfImage(APBI_NameIm());
      StdOut() << "MAK=== " <<   mHasMask << " " << mNameMask  << " UseSim=" << mUseSimul << "\n";
   }


   APBI_ExecAll();  // run the parse file  SIMPL


   delete mSpec;
   DeleteAllAndClear(mVCTE);
   return EXIT_SUCCESS;
}

/* =============================================== */
/*                                                 */
/*                       ::                        */
/*                                                 */
/* =============================================== */

tMMVII_UnikPApli Alloc_ExtractCircTarget(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppliExtractCircTarget(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpecExtractCircTarget
(
     "CodedTargetCircExtract",
      Alloc_ExtractCircTarget,
      "Extract coded target from images",
      {eApF::ImProc,eApF::CodedTarget},
      {eApDT::Image,eApDT::Xml},
      {eApDT::Xml},
      __FILE__
);


};

