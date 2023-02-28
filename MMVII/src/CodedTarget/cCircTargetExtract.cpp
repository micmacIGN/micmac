#include "MMVII_Tpl_Images.h"
#include "MMVII_Linear2DFiltering.h"
#include "MMVII_Geom2D.h"
#include "MMVII_Sensor.h"
#include "MMVII_TplImage_PtsFromValue.h"
#include "MMVII_ImageInfoExtract.h"

/*   Modularistion
 *   Code extern tel que ellipse
 *   Ellipse => avec centre
 *   Pas de continue
 */

namespace MMVII
{


	/*
class cCCDecode
{
    public :
         cCCDecode(const cExtractedEllipse & anEE,const cDataIm2D<tREAL4> & anIm, int aNbBits);
    private :
};

*/



bool  ShowCode(const cExtractedEllipse & anEE,const cDataIm2D<tREAL4> & anIm, int aNbBits)
{
    static int aCpt=0; aCpt++;

    int aMulB   = 10;
    int aNbTeta = aNbBits * aMulB;
    int aNbRho  = 20;

    double aRho0 = 1.5;
    double aRho1 = 3.1;

    cIm2D<tU_INT1> aImPolar(cPt2di(aNbTeta,aNbRho));

    for (int aKTeta=0 ; aKTeta < aNbTeta; aKTeta++)
    {
        for (int aKRho=0 ; aKRho < aNbRho; aKRho++)
        {
		tREAL8 aTeta = (2*M_PI*aKTeta)/aNbTeta;
		tREAL8 aRho  = aRho0 + ((aRho1-aRho0) *aKRho) / aNbRho;
		cPt2dr aPt = anEE.mEllipse.PtOfTeta(aTeta,aRho);
		tREAL8 aVal = anIm.DefGetVBL(aPt,-1);
		if (aVal<0) return false;

		aImPolar.DIm().SetV(cPt2di(aKTeta,aKRho),aVal);
        }
    }

    aImPolar.DIm().ToFile("ImPolar_"+ToStr(aCpt)+".tif");

    return true;
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

        bool            mVisu;
        cExtract_BW_Ellipse * mExtrEll;
        cParamBWTarget  mPBWT;

        tImGrad         mImGrad;
        cRGBImage       mImVisu;
        cIm2D<tU_INT1>  mImMarq;

        std::vector<cSeedBWTarget> mVSeeds;
	cPhotogrammetricProject     mPhProj;

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
   mVisu    (true),
   mExtrEll (nullptr),
   mImGrad  (cPt2di(1,1)),
   mImVisu  (cPt2di(1,1)),
   mImMarq  (cPt2di(1,1)),
   mPhProj  (*this)

{
}

        // cExtract_BW_Target * 
cCollecSpecArg2007 & cAppliExtractCircTarget::ArgObl(cCollecSpecArg2007 & anArgObl)
{
   // Standard use, we put args of  cAppliParseBoxIm first
   return
         APBI_ArgObl(anArgObl)
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
          );
}



int cAppliExtractCircTarget::ExeOnParsedBox()
{
   double aT0 = SecFromT0();

   mExtrEll = new cExtract_BW_Ellipse(APBI_Im(),mPBWT,mPhProj.MaskWithDef(mNameIm,CurBoxIn(),false));
   if (mVisu)
      mImVisu =  cRGBImage::FromFile(mNameIm,CurBoxIn());

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
       {
          ShowEllipse(anEE,mNameIm);
          ShowCode(anEE,APBI_DIm(),14);
       }
   }

   if (mVisu)
   {
       const cExtract_BW_Target::tDImMarq&     aDMarq =  mExtrEll->DImMarq();
       for (const auto & aPix : aDMarq)
       {
            if (aDMarq.GetV(aPix)==tU_INT1(eEEBW_Lab::eTmp))
               mImVisu.SetRGBPix(aPix,cRGBImage::Green);

            if (     (aDMarq.GetV(aPix)==tU_INT1(eEEBW_Lab::eBadZ))
                  || (aDMarq.GetV(aPix)==tU_INT1(eEEBW_Lab::eElNotOk))
	       )
               mImVisu.SetRGBPix(aPix,cRGBImage::Blue);
            if (aDMarq.GetV(aPix)==tU_INT1(eEEBW_Lab::eBadFr))
               mImVisu.SetRGBPix(aPix,cRGBImage::Cyan);
            if (aDMarq.GetV(aPix)==tU_INT1(eEEBW_Lab::eBadEl))
               mImVisu.SetRGBPix(aPix,cRGBImage::Red);
            if (aDMarq.GetV(aPix)==tU_INT1(eEEBW_Lab::eAverEl))
               mImVisu.SetRGBPix(aPix,cRGBImage::Orange);
            if (aDMarq.GetV(aPix)==tU_INT1(eEEBW_Lab::eBadTeta))
               mImVisu.SetRGBPix(aPix,cRGBImage::Yellow);
	       /*
	       */

	       /*
	    cPt3di aRGB =  mImVisu.GetRGBPix(aPix);
	    if ( mExtrEll->DGx().GetV(aPix) >0)
	        aRGB[2] =    aRGB[2] * 0.9;
	    else
	        aRGB[2] =   255 - (255- aRGB[2]) * 0.9;
            mImVisu.SetRGBPix(aPix,aRGB);
	    */

	    // if (mGrad.mGy.DIm().GetV(aPix) == 0)
       }

       for (const auto & aSeed : mExtrEll->VSeeds())
       {
           if (aSeed.mOk)
	   {
              mImVisu.SetRGBPix(aSeed.mPixW,cRGBImage::Red);
              mImVisu.SetRGBPix(aSeed.mPixTop,cRGBImage::Yellow);
	   }
	   else
	   {
              mImVisu.SetRGBPix(aSeed.mPixW,cRGBImage::Yellow);
	   }
       }
   }


   delete mExtrEll;
   if (mVisu)
      mImVisu.ToFile("TTTT.tif");

   return EXIT_SUCCESS;
}



int  cAppliExtractCircTarget::Exe()
{
   mPhProj.FinishInit();

   mHasMask =  mPhProj.ImageHasMask(APBI_NameIm()) ;
   if (mHasMask)
      mNameMask =  mPhProj.NameMaskOfImage(APBI_NameIm());


   APBI_ExecAll();  // run the parse file  SIMPL

   StdOut() << "MAK=== " <<   mHasMask << " " << mNameMask  << "\n";

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

