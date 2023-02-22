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


/*  *********************************************************** */
/*                                                              */
/*               cExtract_BW_Ellipse                            */
/*                                                              */
/*  *********************************************************** */

struct cExtracteEllipse
{
     public :
        cSeedBWTarget    mSeed;
	cEllipse         mEllipse;

	cExtracteEllipse(const cSeedBWTarget& aSeed,const cEllipse & anEllipse);

	tREAL8               mDist;
	tREAL8               mDistPond;
	tREAL8               mEcartAng;
	bool                 mValidated;
	std::vector<cPt2dr>  mVFront;
};

cExtracteEllipse::cExtracteEllipse(const cSeedBWTarget& aSeed,const cEllipse & anEllipse) :
    mSeed      (aSeed),
    mEllipse   (anEllipse),
    mDist      (10.0),
    mDistPond  (10.0),
    mEcartAng  (10.0),
    mValidated (false)
{
}


/*  *********************************************************** */
/*                                                              */
/*               cExtract_BW_Ellipse                            */
/*                                                              */
/*  *********************************************************** */

class cExtract_BW_Ellipse  : public cExtract_BW_Target
{
	public :
             cExtract_BW_Ellipse(tIm anIm,const cParamBWTarget & aPBWT,cIm2D<tU_INT1> aMasqTest);

             void AnalyseAllConnectedComponents(const std::string & aNameIm);
             bool AnalyseEllipse(cSeedBWTarget & aSeed,const std::string & aNameIm);

	     const std::list<cExtracteEllipse> & ListExtEl() const;  ///< Accessor
	private :

	     std::list<cExtracteEllipse> mListExtEl;
};

cExtract_BW_Ellipse::cExtract_BW_Ellipse(tIm anIm,const cParamBWTarget & aPBWT,cIm2D<tU_INT1> aMasqTest) :
	cExtract_BW_Target(anIm,aPBWT,aMasqTest)
{
}

void cExtract_BW_Ellipse::AnalyseAllConnectedComponents(const std::string & aNameIm)
{
    for (auto & aSeed : mVSeeds)
    {
        if (AnalyseOneConnectedComponents(aSeed))
        {
            if (ComputeFrontier(aSeed))
	    {
                AnalyseEllipse(aSeed,aNameIm);
	    }
        }
    }
}


bool  cExtract_BW_Ellipse::AnalyseEllipse(cSeedBWTarget & aSeed,const std::string & aNameIm)
{
     cEllipse_Estimate anEEst(mCentroid);
     for (const auto  & aPFr : mVFront)
         anEEst.AddPt(aPFr);
     cEllipse anEl = anEEst.Compute();
     if (! anEl.Ok())
     {
        CC_SetMarq(eEEBW_Lab::eElNotOk); 
        return false;
     }
     double aSomD = 0;
     double aSomRad = 0;
     tREAL8 aGrFr = (aSeed.mBlack+aSeed.mWhite)/2.0;
     for (const auto  & aPFr : mVFront)
     {
         aSomD += std::abs(anEl.ApproxSigneDist(aPFr));
	 aSomRad += std::abs(mDIm.GetVBL(aPFr)-aGrFr);
     }

     aSomD /= mVFront.size();

     tREAL8 aSomDPond =  aSomD / (1+anEl.RayMoy()/50.0);

     int aNbPts = round_ni(4*(anEl.LGa()+anEl.LSa()));
     tREAL8 aSomTeta = 0.0;
     for (int aK=0 ; aK<aNbPts ; aK++)
     {
            double aTeta = (aK * 2.0 * M_PI) / aNbPts;
	    cPt2dr aGradTh;
	    cPt2dr aPt = anEl.PtAndGradOfTeta(aTeta,aGradTh);

	    if (! mDGx.InsideBL(aPt))
	    {
                  CC_SetMarq(eEEBW_Lab::eElNotOk); 
                  return false;
	    }
            cPt2dr aGradIm (mDGx.GetVBL(aPt),mDGy.GetVBL(aPt));
	    aSomTeta += std::abs(ToPolar(aGradIm/-aGradTh).y());
     }
     aSomTeta /= aNbPts;

     if (aSomDPond>0.2)
     {
        CC_SetMarq(eEEBW_Lab::eBadEl); 
	return false;
     }
     

     cExtracteEllipse  anEE(aSeed,anEl);
     anEE.mDist      = aSomD;
     anEE.mDistPond  = aSomDPond;
     anEE.mEcartAng  = aSomTeta;
     anEE.mVFront    = mVFront;

     if (aSomDPond>0.1)
     {
         CC_SetMarq(eEEBW_Lab::eAverEl); 
     }
     else
     {
         if (aSomTeta>0.05)
	 {
            CC_SetMarq(eEEBW_Lab::eBadTeta); 
	 }
	 else
	 {
            anEE.mValidated = true;
	 }
     }

     mListExtEl.push_back(anEE);
     return true;
}

const std::list<cExtracteEllipse> & cExtract_BW_Ellipse::ListExtEl() const {return mListExtEl;}

void  ShowEllipse(const cExtracteEllipse & anEE,const std::string & aNameIm)
{
    
    static int aCptIm = 0;
    aCptIm++;
    const cSeedBWTarget &  aSeed = anEE.mSeed;
    const cEllipse &       anEl  = anEE.mEllipse;

     StdOut() << "SOMDDd=" << anEE.mDist << " DP=" << anEE.mDistPond << " " << aSeed.mPixTop 
		 << " NORM =" << anEl.Norm()  << " RayM=" << anEl.RayMoy()
		 << "\n";

     /*
        std::vector<cPt3dr> aVF3;
        for (const auto  & aP : mVFront)
	{
             aVF3.push_back(cPt3dr(aP.x(),aP.y(),ToPolar(aP-mCentroid).y()));
	}
        std::sort
        (
	    aVF3.begin(),
	    aVF3.end(),
            [](const cPt3dr  aP1,const cPt3dr &  aP2) {return aP1.z() >aP2.z();}
        );
	*/

        StdOut() << " BOX " << aSeed.mPInf << " " << aSeed.mPSup << "\n";

	cPt2di  aPMargin(6,6);
	cBox2di aBox(aSeed.mPInf-aPMargin,aSeed.mPSup+aPMargin);

	int aZoom = 21;
	cRGBImage aRGBIm = cRGBImage::FromFile(aNameIm,aBox,aZoom);  ///< Allocate and init from file
	cPt2dr aPOfs = ToR(aBox.P0());
	/*
	int aNbPts = 5000;
	for (int aK=0 ; aK<aNbPts ; aK++)
	{
            double aTeta = (aK * 2.0 * M_PI) / aNbPts;
	    cPt2dr aP1 = anEl.PtOfTeta(aTeta);
	    tREAL8 aEps=1e-3;
	    cPt2dr aP0 = anEl.PtOfTeta(aTeta - aEps );
	    cPt2dr aP2 = anEl.PtOfTeta(aTeta + aEps );

	    cPt2dr aGrad = VUnit((aP2-aP0) / aEps)/cPt2dr(0,1);
            //  aRGBIm.SetRGBPoint(aPT-aPOfs,cRGBImage::Green);
	    cPt2dr aG3;
            cPt2dr aP3 =   anEl.PtAndGradOfTeta(aTeta,aG3);

	    // StdOut() << " Pt" << Norm2(aP1-aP3)  << " Gd "<< Norm2(aGrad-aG3)  << aGrad << aG3<< "\n";
	}
	*/

	cPt2dr aCenter = anEl.Center() - aPOfs;
	aCenter = aRGBIm.PointToRPix(aCenter);
	std::vector<cPt2di> aVPts;
	GetPts_Ellipse(aVPts,aCenter,anEl.LGa()*aZoom,anEl.LSa()*aZoom,anEl.TetaGa(),true);
	for (const auto & aPix : aVPts)
	{
            aRGBIm.RawSetPoint(aPix,cRGBImage::Blue);
	}

        for (const auto  & aPFr : anEE.mVFront)
	{
            aRGBIm.SetRGBPoint(aPFr-aPOfs,cRGBImage::Red);
	    //StdOut() <<  "DDDD " <<  anEl.ApproxSigneDist(aPFr) << "\n";
	}

	aRGBIm.ToFile("VisuDetectCircle_"+ ToStr(aCptIm) + ".tif");
#if (0)
#endif
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
          ShowEllipse(anEE,mNameIm);
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
      {eApF::ImProc},
      {eApDT::Image,eApDT::Xml},
      {eApDT::Xml},
      __FILE__
);


};

