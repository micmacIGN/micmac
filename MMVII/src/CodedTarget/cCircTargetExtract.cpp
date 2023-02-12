#include "MMVII_Tpl_Images.h"
#include "MMVII_Linear2DFiltering.h"

/*   Modularistion
 *   Code extern tel que ellipse
 *   Ellipse => avec centre
 *   Pas de continue
 */

namespace MMVII
{



/*  *********************************************************** */
/*                                                              */
/*                   cParamCircTarg                             */
/*                                                              */
/*  *********************************************************** */

struct cParamCircTarg
{
    public :
      cParamCircTarg();

      double mFactDeriche;
      int    mD0BW;
      double mValMinW;
      double mValMaxB;
      double mRatioMaxBW;
};

cParamCircTarg::cParamCircTarg() :
    mFactDeriche (2.0),
    mD0BW        (2),
    mValMinW     (20), 
    mValMaxB     (100),
    mRatioMaxBW  (1/3.0)
{
}

struct cSeedCircTarg
{
    public :
       cPt2di mPix;
       tREAL4 mBlack;
       tREAL4 mWhite;

       cSeedCircTarg(const cPt2di & aPix,  tREAL4 mBlack,tREAL4 mWhite);
};

cSeedCircTarg::cSeedCircTarg(const cPt2di & aPix,  tREAL4 aBlack,tREAL4 aWhite):
   mPix   (aPix),
   mBlack (aBlack),
   mWhite (aWhite)
{
}

/*  *********************************************************** */
/*                                                              */
/*               cExtract_BW_Ellipse                            */
/*                                                              */
/*  *********************************************************** */

class cExtract_BW_Ellipse
{
   public :
        typedef tREAL4              tElemIm;
        typedef cDataIm2D<tElemIm>  tDataIm;
        typedef cIm2D<tElemIm>      tIm;
        typedef cImGrad<tElemIm>    tImGrad;

        cExtract_BW_Ellipse(tIm anIm,const cParamCircTarg & aPCT);

        void ExtractAllSeed();
        const std::vector<cSeedCircTarg> & VSeeds();
   private :

        bool IsCandidateTopOfEllipse(const cPt2di &) ;

        tIm              mIm;
        tDataIm &        mDIm;
        cParamCircTarg   mPCT;
        tImGrad          mImGrad;
        std::vector<cSeedCircTarg> mVSeeds;
};

cExtract_BW_Ellipse::cExtract_BW_Ellipse(tIm anIm,const cParamCircTarg & aPCT) :
   mIm      (anIm),
   mDIm     (mIm.DIm()),
   mPCT     (aPCT),
   mImGrad  (Deriche( mDIm,mPCT.mFactDeriche))
{
}

/*
        0 0 0 0 0
      L 0 0 1 0 0  R 
        0 1 1 1 0
     
       R => negative gradient on x
       L => positive gradient on x
*/
         
bool cExtract_BW_Ellipse::IsCandidateTopOfEllipse(const cPt2di & aPix) 
{
   // is it a point where gradient cross vertical line
   const tDataIm & aDGx = mImGrad.mGx.DIm();
   if ( (aDGx.GetV(aPix)>0) ||  (aDGx.GetV(aPix+cPt2di(-1,0)) <=0) )
      return false;

   const tDataIm & aDGy = mImGrad.mGy.DIm();

   tElemIm aGy =  aDGy.GetV(aPix);
   // At top , grady must be positive
   if (  aGy<=0) 
      return false;

   // Now test that grad is a local maxima

   if (    (aGy  <   aDGy.GetV(aPix+cPt2di(0, 1)))
        || (aGy  <=  aDGy.GetV(aPix+cPt2di(0,-1)))
      )
      return false;


   tElemIm aVBlack =  mDIm.GetV(aPix+cPt2di(0,-mPCT.mD0BW));
   tElemIm aVWhite =  mDIm.GetV(aPix+cPt2di(0,mPCT.mD0BW));
   
   if (aVWhite < mPCT.mValMinW)
      return false;

   if (aVBlack > mPCT.mValMaxB)
      return false;

    if ((aVBlack/double(aVWhite)) > mPCT.mRatioMaxBW)
      return false;

   mVSeeds.push_back(cSeedCircTarg(aPix,aVBlack,aVWhite));

   return true;
}

void cExtract_BW_Ellipse::ExtractAllSeed()
{
   const cBox2di &  aFullBox = mDIm;
   cRect2  aBoxInt (aFullBox.Dilate(-mPCT.mD0BW));
   int aNb=0;
   int aNbOk=0;
   for (const auto & aPix : aBoxInt)
   {
       aNb++;
       if (IsCandidateTopOfEllipse(aPix))
       {
           aNbOk++;
       }
   }
   std::sort
   (
      mVSeeds.begin(),
      mVSeeds.end(),
      [](const cSeedCircTarg &  aS1,const cSeedCircTarg &  aS2) {return aS1.mWhite>aS2.mWhite;}
   );
   StdOut() << " PPPP="  << aNbOk / double(aNb) << "\n";
}

const std::vector<cSeedCircTarg> & cExtract_BW_Ellipse::VSeeds()
{
   return mVSeeds;
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

        bool IsCandidateTopOfEllipse(const cPt2di &) ;

        bool            mVisu;
        cExtract_BW_Ellipse * mExtrEll;
        cParamCircTarg  mPCT;

        tImGrad         mImGrad;
        cRGBImage       mImVisu;
        cIm2D<tU_INT1>  mImMarq;

        std::vector<cSeedCircTarg> mVSeeds;

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
   mImMarq  (cPt2di(1,1))
{
}

        // cExtract_BW_Ellipse * 
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
                   //  << AOpt2007(mDiamMinD, "DMD","Diam min for detect",{eTA2007::HDV})
	  );
   ;
}






int cAppliExtractCircTarget::ExeOnParsedBox()
{
   mExtrEll = new cExtract_BW_Ellipse(APBI_Im(),mPCT);

   if (mVisu)
      mImVisu =  cRGBImage::FromFile(mNameIm,CurBoxIn());

   mExtrEll->ExtractAllSeed();
   if (mVisu)
   {
       for (const auto & aSeed : mExtrEll->VSeeds())
           mImVisu.SetRGBPix(aSeed.mPix,cRGBImage::Red);
   }


   delete mExtrEll;
   if (mVisu)
      mImVisu.ToFile("TTTT.tif");

   return EXIT_SUCCESS;
}



int  cAppliExtractCircTarget::Exe()
{
   APBI_ExecAll();  // run the parse file  SIMPL



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

