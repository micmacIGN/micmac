#include "MMVII_Tpl_Images.h"
#include "MMVII_Linear2DFiltering.h"
#include "MMVII_Geom2D.h"
#include "MMVII_Sensor.h"
#include "MMVII_TplImage_PtsFromValue.h"
#include "MMVII_ImageInfoExtract.h"
#include "CodedTarget.h"

/*   Modularistion
 *   Code extern tel que ellipse
 *   Ellipse => avec centre
 *   Pas de continue
 */

namespace MMVII
{

using namespace cNS_CodedTarget;

/* ********************************************* */
/*                                               */
/*                cCircTargExtr                  */
/*                                               */
/* ********************************************* */

/**    Store the result of a validated extracted circular target
 */

class cCircTargExtr
{
     public :
         cCircTargExtr(const cExtractedEllipse &);

         cEllipse         mEllipse;
	 tREAL8           mBlack;
	 tREAL8           mWhite;
	 bool             mMarked4Test;
	 bool             mWithCode;
	 cOneEncoding     mEncode;
};

cCircTargExtr::cCircTargExtr(const cExtractedEllipse & anEE)  :
	mEllipse     (anEE.mEllipse),
	mBlack       (anEE.mSeed.mBlack),
	mWhite       (anEE.mSeed.mWhite),
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
         cCCDecode(cCircTargExtr & anEE,const cDataIm2D<tREAL4> & aDIm,const cFullSpecifTarget &);

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

	     // Geometric correspondances 
         tREAL8 K2Rho (int aK) const;   /// index of rho  2 real rho
         tREAL8 K2Teta(int aK) const;   /// index of teta  2 real teta
         int Rho2K (tREAL8 aR) const;   ///  real rho 2 index of rho
	 cPt2dr  KTetaRho2Im(const cPt2di & aKTetaRho) const;   /// index rho-teta  2   cartesian coordinates 
         /// For weight in [0,1] return a rho corresponding to coding place
	 tREAL8 CodingRhoOfWeight(const tREAL8 &) const;

         int KBeginInterv(int aK0,int aNumBit) const;
         int KEndInterv(int aK0,int aNumBit) const;


         cCircTargExtr &           mEE;
         const cDataIm2D<tREAL4> & mDIm;
	 const cFullSpecifTarget & mSpec;


	 bool                      mOK;
	 const int                 mPixPerB; ///< number of pixel for each bit to decode
	 const int                 mNbRho;   ///< number of pixel for rho
	 int                       mNbB;     ///< number of bits in code
	 int                       mNbTeta;  ///< number of pixel for each bit to decode
	 tREAL8                    mRho0;
	 tREAL8                    mRho1;
         cIm2D<tREAL4>             mImPolar;
         cDataIm2D<tREAL4> &       mDIP;
         cIm1D<tREAL4>             mAvg;
         cDataIm1D<tREAL4> &       mDAvg;
	 int                       mKR0;
	 int                       mKR1;
	 int                       mPhase0;
	 tREAL8                    mBlack;
	 tREAL8                    mWhite;
	 tREAL8                    mBWAmpl;
	 tREAL8                    mBWAvg;
	 const cOneEncoding *      mEnCode;
};

    // ==============   constructor ============================

cCCDecode::cCCDecode(cCircTargExtr & anEE,const cDataIm2D<tREAL4> & aDIm,const cFullSpecifTarget & aSpec) :
	mEE        (anEE),
	mDIm       (aDIm),
	mSpec      (aSpec),
	mOK        (true),
	mPixPerB   (10),
	mNbRho     (20),
	mNbB       (mSpec.NbBits()),
	mNbTeta    (mPixPerB * mNbB),
	mRho0      ((mSpec.Rho_0_EndCCB()+mSpec.Rho_1_BeginCode()) /2.0),
	mRho1      (mSpec.Rho_2_EndCode() +0.2),
	mImPolar   (cPt2di(mNbTeta,mNbRho)),
        mDIP       (mImPolar.DIm()),
	mAvg       ( mNbTeta,nullptr,eModeInitImage::eMIA_Null ),
	mDAvg      ( mAvg.DIm()),
	mKR0       ( Rho2K(CodingRhoOfWeight(0.25)) ) ,
	mKR1       ( Rho2K(CodingRhoOfWeight(0.75)) ) ,
	mPhase0    (-1),
	mBlack     (mEE.mBlack),
	mWhite     (mEE.mWhite),
	mBWAmpl    (mWhite-mBlack),
	mBWAvg     ((mBlack+mWhite)/2.0),
	mEnCode    (nullptr)
{

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
    if (!mOK) return;

    ComputeCode();
    if (!mOK) return;
}

//  =============   Agregation on interval : StdDev , Avg, TotalStdDevOfPhase ====

void  cCCDecode::AddStdDev(int aK1,int aK2,cComputeStdDev<tREAL8> & aCS) const
{
    for (int aK=aK1 ; aK<aK2 ; aK++)
    {
         aCS.Add(mDAvg.GetV(aK%mNbTeta));
    }
}
         // aSum +=  (KBeginInterv(aK0,aKBit),KEndInterv(aK0,aKBit+1));

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
       AddStdDev( KBeginInterv(mPhase0,anI.x()), KEndInterv(mPhase0,anI.y()), aCS);

    return aCS.StdDev(0);
}

tREAL8 cCCDecode::TotalStdDevOfPhase(int aK0) const
{
    tREAL8 aSum=0;
    for (int aKBit=0 ; aKBit<mNbB ; aKBit++)
    {
        // int aK1 = aK0+aKBit*mPixPerB;
        // aSum +=  StdDev(aK1+1,aK1+mPixPerB-1);
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
tREAL8 cCCDecode::K2Teta(const int aK) const {return  (2*M_PI*aK)/mNbTeta;}

int  cCCDecode::Rho2K(const tREAL8 aR)  const 
{
     return round_ni( ((aR-mRho0)/(mRho1-mRho0)) * mNbRho );
}

tREAL8 cCCDecode::CodingRhoOfWeight(const tREAL8 & aW) const
{
	return (1-aW) * mSpec.Rho_1_BeginCode() + aW * mSpec.Rho_2_EndCode();
}


//=================

void cCCDecode::ComputePhaseTeta() 
{
    // Extract phase minimizing the standard dev on all intervall
    cWhichMin<int,tREAL8> aMinDev;
    for (int aK0=0 ;aK0< mPixPerB ; aK0++)
	    aMinDev.Add(aK0,TotalStdDevOfPhase(aK0));
    mPhase0 = aMinDev.IndexExtre();

    //  decide if sufficiently homogeneous
    if (     (aMinDev.ValExtre() > 0.1 * StdDev(0,mNbTeta))
          || (aMinDev.ValExtre() > 0.05 *  mBWAmpl)
       )
    {
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
        int aK1 = mPhase0+aKBit*mPixPerB;
        tREAL8 aMoy =  Avg(aK1+1,aK1+mPixPerB-1);

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
    std::vector<cPt2di> aV0;
    std::vector<cPt2di> aV1;
    MaxRunLength(aFlag,1<<mNbB,aV0,aV1);

    // Test were made to compute the global deviation on black/white part, but not concluding as
    if (0)
    {
         tREAL8 aDev0 = StdDevOfSumInterv(aV0);
         tREAL8 aDev1 = StdDevOfSumInterv(aV1);
         StdOut()  << mEnCode->Name() <<  " D0=" << aDev0/ mBWAmpl <<  " D1=" << aDev1/ mBWAmpl <<  "\n";
    }



    mEE.mWithCode = true;
    mEE.mEncode = cOneEncoding(mEnCode->Num(),mEnCode->Code(),mEnCode->Name());
}





void  cCCDecode::Show(const std::string & aPrefix)
{
    static int aCpt=0; aCpt++;

    cRGBImage  aIm = RGBImFromGray(mImPolar.DIm(),1.0,9);

    if (mPhase0>=0)
    {
       for (int aKBit=0 ; aKBit<mNbB ; aKBit++)
       {
           tREAL8 aK1 = mPhase0+aKBit*mPixPerB -0.5;

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

	std::string         mNameSpec;
	cFullSpecifTarget * mSpec;
        bool                  mVisuLabel;
        bool                  mVisuElFinal;
        cExtract_BW_Ellipse * mExtrEll;
        cParamBWTarget  mPBWT;

        tImGrad         mImGrad;
        cIm2D<tU_INT1>  mImMarq;

        std::vector<cCircTargExtr>  mVCTE;
	cPhotogrammetricProject     mPhProj;

	std::string                 mPrefixOut;
	bool                        mHasMask;
	std::string                 mNameMask;
};



cAppliExtractCircTarget::cAppliExtractCircTarget
(
    const std::vector<std::string> & aVArgs,
    const cSpecMMVII_Appli & aSpec
) :
   cMMVII_Appli  (aVArgs,aSpec),
   cAppliParseBoxIm<tREAL4>(*this,true,cPt2di(10000,10000),cPt2di(300,300),false) ,
   mSpec         (nullptr),
   mVisuLabel    (false),
   mVisuElFinal (true),
   mExtrEll      (nullptr),
   mImGrad       (cPt2di(1,1)),
   mImMarq       (cPt2di(1,1)),
   mPhProj       (*this)

{
}

        // cExtract_BW_Target * 
cCollecSpecArg2007 & cAppliExtractCircTarget::ArgObl(cCollecSpecArg2007 & anArgObl)
{
   // Standard use, we put args of  cAppliParseBoxIm first
   return
             APBI_ArgObl(anArgObl)
        <<   Arg2007(mNameSpec,"XML name for bit encoding struct")

                   //  << AOpt2007(mDiamMinD, "DMD","Diam min for detect",{eTA2007::HDV})
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
             << AOpt2007(mVisuLabel,"VisuLabel","Make a visualisation of labeled image",{eTA2007::HDV})
          );
}

void cAppliExtractCircTarget::MakeImageFinalEllispe()
{
   cRGBImage   aImVisu=  cRGBImage::FromFile(mNameIm,CurBoxIn());

   for (const auto & anEE : mVCTE)
   {
        const cEllipse &   anEl  = anEE.mEllipse;
        for (const auto & aMul : {1.0,1.2,1.4})
        {
            aImVisu.DrawEllipse
            (
               cRGBImage::Blue ,  // anEE.mWithCode ? cRGBImage::Blue : cRGBImage::Red,
               anEl.Center(),
               anEl.LGa()*aMul , anEl.LSa()*aMul , anEl.TetaGa()
            );
        }
	if (anEE.mWithCode)
        {
             aImVisu.DrawString
             (
                  anEE.mEncode.Name(),cRGBImage::Red,
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




int cAppliExtractCircTarget::ExeOnParsedBox()
{
   double aT0 = SecFromT0();

   mExtrEll = new cExtract_BW_Ellipse(APBI_Im(),mPBWT,mPhProj.MaskWithDef(mNameIm,CurBoxIn(),false));

   double aT1 = SecFromT0();
   mExtrEll->ExtractAllSeed();
   double aT2 = SecFromT0();
   mExtrEll->AnalyseAllConnectedComponents(mNameIm);
   double aT3 = SecFromT0();

   StdOut() << "TIME-INIT " << aT1-aT0 << "\n";
   StdOut() << "TIME-SEED " << aT2-aT1 << "\n";
   StdOut() << "TIME-CC   " << aT3-aT2 << "\n";

   for (const auto & anEE : mExtrEll->ListExtEl() )
   {
       if (anEE.mSeed.mMarked4Test)
          anEE.ShowOnFile(mNameIm,21,mPrefixOut);
       if (anEE.mValidated  || anEE.mSeed.mMarked4Test)
       {
	  cCircTargExtr aCTE(anEE);
	  mVCTE.push_back(aCTE);
       }
   }

   for (auto & anEE : mVCTE)
   {
       cCCDecode aCCD(anEE,APBI_DIm(),*mSpec);
       if (anEE.mMarked4Test)
       {
	     aCCD.Show(mPrefixOut);
       }
   }

   if (mVisuLabel)
      MakeImageLabel();

   if (mVisuElFinal)
      MakeImageFinalEllispe();

   delete mExtrEll;

   return EXIT_SUCCESS;
}



int  cAppliExtractCircTarget::Exe()
{
   mPrefixOut = "CircTarget_" +  Prefix(APBI_NameIm());

   mSpec = cFullSpecifTarget::CreateFromFile(mNameSpec);

   mPhProj.FinishInit();

   mHasMask =  mPhProj.ImageHasMask(APBI_NameIm()) ;
   if (mHasMask)
      mNameMask =  mPhProj.NameMaskOfImage(APBI_NameIm());


   APBI_ExecAll();  // run the parse file  SIMPL

   StdOut() << "MAK=== " <<   mHasMask << " " << mNameMask  << "\n";

   delete mSpec;
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

