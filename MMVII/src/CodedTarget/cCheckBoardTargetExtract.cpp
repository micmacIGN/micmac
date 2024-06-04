#include "CodedTarget.h"
#include "MMVII_2Include_Serial_Tpl.h"
#include "MMVII_Tpl_Images.h"
#include "MMVII_Interpolators.h"
#include "MMVII_Linear2DFiltering.h"
#include "MMVII_ImageMorphoMath.h"
#include "MMVII_2Include_Tiling.h"
#include "MMVII_Sensor.h"
#include "MMVII_HeuristikOpt.h"


// Test git branch

namespace MMVII
{


static constexpr tU_INT1 eNone = 0 ;
static constexpr tU_INT1 eTopo0  = 1 ;
static constexpr tU_INT1 eTopoTmpCC  = 2 ;
static constexpr tU_INT1 eTopoMaxOfCC  = 3 ;
static constexpr tU_INT1 eTopoMaxLoc  = 4 ;
static constexpr tU_INT1 eFilterSym  = 5 ;

/*  *********************************************************** */
/*                                                              */
/*              cAppliCheckBoardTargetExtract                   */
/*                                                              */
/*  *********************************************************** */


class cAppliCheckBoardTargetExtract : public cMMVII_Appli
{
     public :
        typedef tREAL4            tElem;
        typedef cIm2D<tElem>      tIm;
        typedef cDataIm2D<tElem>  tDIm;
        typedef cAffin2D<tREAL8>  tAffMap;


        cAppliCheckBoardTargetExtract(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);

     private :
        // =========== overridding cMMVII_Appli::methods ============
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;

        bool IsPtTest(const cPt2dr & aPt) const;  ///< Is it a point marqed a test


	void DoOneImage() ;
	void MakeImageSaddlePoints(const tDIm &,const cDataIm2D<tU_INT1> & aDMasq) const;

        cWhichMin<cPt2dr,tREAL8>   OptimFilter(cFilterDCT<tREAL4> *,const cPt2dr &) const;

	cPhotogrammetricProject     mPhProj;
	cTimerSegm                  mTimeSegm;

        // =========== Mandatory args ============

	std::string mNameIm;       ///< Name of background image
	std::string mNameSpecif;   ///< Name of specification file

        // =========== Optionnal args ============

                //  --

	std::vector<int>  mShowQuickSaddleF;

        // =========== Internal param ============
        tIm                   mImIn;        ///< Input global image
        cPt2di                mSzIm;        ///< Size of image
	tDIm *                mDImIn;       ///< Data input image 
	bool                  mHasMasqTest; ///< Do we have a test image 4 debuf (with masq)
	cIm2D<tU_INT1>        mMasqTest;    ///< Possible image of mas 4 debug, print info ...
        cIm2D<tU_INT1>        mImLabel;     ///< Image storing labels of centers
	cDataIm2D<tU_INT1> *  mDImLabel;    ///< Data Image of label
};


/* *************************************************** */
/*                                                     */
/*              cAppliCheckBoardTargetExtract                  */
/*                                                     */
/* *************************************************** */

cAppliCheckBoardTargetExtract::cAppliCheckBoardTargetExtract(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli     (aVArgs,aSpec),
   mPhProj          (*this),
   mTimeSegm        (this),
   mImIn            (cPt2di(1,1)),
   mDImIn           (nullptr),
   mHasMasqTest     (false),
   mMasqTest        (cPt2di(1,1)),
   mImLabel         (cPt2di(1,1)),
   mDImLabel        (nullptr)
{
}



cCollecSpecArg2007 & cAppliCheckBoardTargetExtract::ArgObl(cCollecSpecArg2007 & anArgObl)
{
   return  anArgObl
             <<   Arg2007(mNameIm,"Name of image (first one)",{{eTA2007::MPatFile,"0"},eTA2007::FileImage})
             <<   Arg2007(mNameSpecif,"Name of target file")
   ;
}


cCollecSpecArg2007 & cAppliCheckBoardTargetExtract::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return
	        anArgOpt
             <<  mPhProj.DPMask().ArgDirInOpt("TestMask","Mask for selecting point used in detailed mesg/output")
             <<  AOpt2007(mShowQuickSaddleF,"ShowQSF","Vector to show quick saddle filters")
   ;
}


void cAppliCheckBoardTargetExtract::MakeImageSaddlePoints(const tDIm & aDIm,const cDataIm2D<tU_INT1> & aDMasq) const
{
    cRGBImage  aRGB = RGBImFromGray<tElem>(aDIm);

    for (const auto & aPix : cRect2(aDIm.Dilate(-1)))
    {
       if (aDMasq.GetV(aPix) >= (int) eTopoMaxLoc)
       {
          aRGB.SetRGBPix(aPix,(aDMasq.GetV(aPix)==eFilterSym) ? cRGBImage::Red : cRGBImage::Green );
       }
    }
    aRGB.ToFile("Saddles.tif");
}

bool cAppliCheckBoardTargetExtract::IsPtTest(const cPt2dr & aPt) const
{
   return mHasMasqTest && (mMasqTest.DIm().GetV(ToI(aPt)) != 0);
}

cWhichMin<cPt2dr,tREAL8>  cAppliCheckBoardTargetExtract::OptimFilter(cFilterDCT<tREAL4> * aFilter,const cPt2dr & aP0) const
{
     cWhichMin<cPt2dr,tREAL8> aRes;
     for (tREAL8 aDx=-1.0 ; aDx<1.0 ; aDx += 0.1)
     {
         for (tREAL8 aDy=-1.0 ; aDy<1.0 ; aDy += 0.1)
	 {
             cPt2dr aPt = aP0+cPt2dr(aDx,aDy);
	     // tREAL8 aVal = aFilter->ComputeValMaxCrown(aPt,1e10);
	     tREAL8 aVal = aFilter->ComputeVal(aPt);
	     aRes.Add(aPt,aVal);
             // UpdateMin(aMinSym,aFSym->ComputeValMaxCrown(aP2+cPt2dr(aDx,aDy),10));
	 }
     }
     return aRes;
}

void cAppliCheckBoardTargetExtract::DoOneImage() 
{ 
    int   mNbBlur1 = 4;  // Number of iteration of initial blurring
    tREAL8 mDistMaxLocSad = 10.0;  // for supressing sadle-point,  not max loc in a neighboorhoud
    int    mMaxNbMLS = 2000; //  Max number of point in best saddle points
    tREAL8 aRayCalcSadle = sqrt(4+1);  // limit point 2,1

    tREAL8 mThresholdSym     = 0.50;
    tREAL8 mDistCalcSym0     = 8.0;
    tREAL8 mDistDivSym       = 2.0;
    

    //   computed threshold
    tINT8  mDistRectInt = 20; // to see later how we compute it


    /* [0]    Initialise : read image and mask */

    cAutoTimerSegm aTSInit(mTimeSegm,"Init");

	// [0.0]   read image
    mImIn =  tIm::FromFile(mNameIm);
    mDImIn = &mImIn.DIm() ;
    mSzIm = mDImIn->Sz();
    cRect2 aRectInt = mDImIn->Dilate(-mDistRectInt);

	// [0.1]   initialize labeling image 
    //mImLabel(mSzIm,nullptr,eModeInitImage::eMIA_Null);
    mDImLabel =  &(mImLabel.DIm());
    mDImLabel->Resize(mSzIm);
    mDImLabel->InitCste(eNone);


    mHasMasqTest = mPhProj.ImageHasMask(mNameIm);
    if (mHasMasqTest)
       mMasqTest =  mPhProj.MaskOfImage(mNameIm,*mDImIn);


    /* [1]   Compute a blurred image => less noise, less low level saddle */

    cAutoTimerSegm aTSBlur(mTimeSegm,"Blurr");

    tIm   aImBlur  = mImIn.Dup(); // create image blurred with less noise
    tDIm& aDImBlur = aImBlur.DIm();

    SquareAvgFilter(aDImBlur,mNbBlur1,1,1);



    /* [2]  Compute "topological" saddle point */

    cAutoTimerSegm aTSTopoSad(mTimeSegm,"TopoSad");

         // 2.1  point with criteria on conexity of point > in neighoor

    int aNbSaddle=0;
    int aNbTot=0;

    for (const auto & aPix : aRectInt)
    {
        if (FlagSup8Neigh(aDImBlur,aPix).NbConComp() >=4)
	{
            mDImLabel->SetV(aPix,eTopo0);
	    aNbSaddle++;
	}
        aNbTot++;
    }

    
         // 2.2  as often there 2 "touching" point with this criteria
	 // select 1 point in conected component

    cAutoTimerSegm aTSMaxCC(mTimeSegm,"MaxCCSad");
    int aNbCCSad=0;
    std::vector<cPt2di>  aVCC;
    const std::vector<cPt2di> & aV8 = Alloc8Neighbourhood();

    for (const auto& aPix : *mDImLabel)
    {
         if (mDImLabel->GetV(aPix)==eTopo0)
	 {
             aNbCCSad++;
             ConnectedComponent(aVCC,*mDImLabel,aV8,aPix,eTopo0,eTopoTmpCC);
	     cWhichMax<cPt2di,tREAL8> aBestPInCC;
	     for (const auto & aPixCC : aVCC)
	     {
                 aBestPInCC.Add(aPixCC,CriterionTopoSadle(aDImBlur,aPixCC));
	     }

	     cPt2di aPCC = aBestPInCC.IndexExtre();
	     mDImLabel->SetV(aPCC,eTopoMaxOfCC);
	 }
    }

    /* [3]  Compute point that are max local */

    cAutoTimerSegm aTSCritSad(mTimeSegm,"CritSad");

    std::vector<cPt3dr> aVSad0;
    cCalcSaddle  aCalcSBlur(aRayCalcSadle+0.001,1.0);

    for (const auto& aPix : *mDImLabel)
    {
         if (mDImLabel->GetV(aPix)==eTopoMaxOfCC)
	 {
             tREAL8 aCritS = aCalcSBlur.CalcSaddleCrit(aDImBlur,aPix);
             aVSad0.push_back(cPt3dr(aPix.x(),aPix.y(),aCritS));
	 }
    }

    //   [3.2]  select KBest + MaxLocal
    cAutoTimerSegm aTSMaxLoc(mTimeSegm,"MaxLoc");

    SortOnCriteria(aVSad0,[](const auto & aPt){return - aPt.z();} );
    std::vector<cPt3dr> aVMaxLoc = FilterMaxLoc((cPt2dr*)nullptr,aVSad0,[](const auto & aP) {return Proj(aP);}, mDistMaxLocSad);

    //  limit the number of point , a bit rough but first experiment show that sadle criterion is almost perfect on good images
    aVMaxLoc.resize(std::min(aVMaxLoc.size(),size_t(mMaxNbMLS)));

    for (const auto & aP3 : aVMaxLoc)
        mDImLabel->SetV(ToI(Proj(aP3)),eTopoMaxLoc);

    StdOut() << "END MAXLOC \n";



    /* [4]  Calc Symetry criterion */

    cAutoTimerSegm aTSSym(mTimeSegm,"SYM");
    if (1)
    {
       cCalcSaddle  aCalcSInit(1.5,1.0);
       std::vector<cPt3dr> aNewP3;
       cFilterDCT<tREAL4> * aFSym = cFilterDCT<tREAL4>::AllocSym(mImIn,0.0,mDistCalcSym0,1.0);
       cOptimByStep aOptimSym(*aFSym,true,mDistDivSym);
       for (const auto & aP3 : aVMaxLoc)
       {
	   cPt2dr aP0 = Proj(aP3);

	   auto [aValSym,aNewP] = aOptimSym.Optim(aP0,1.0,0.01);

	   if (aValSym< mThresholdSym)
	   {
               aNewP3.push_back(cPt3dr(aNewP.x(),aNewP.y(),aValSym));
               mDImLabel->SetV(ToI(aNewP),eFilterSym);
	   }
       }

       delete aFSym;
       aVMaxLoc = aNewP3;
    }

    if (0)
    {
       cFilterDCT<tREAL4> * aFSym = cFilterDCT<tREAL4>::AllocSym(mImIn,0.0,mDistCalcSym0,1.0);
       cOptimByStep aOptimSym(*aFSym,true,mDistDivSym);
// tPtR Optim(const tPtR & ,tREAL8 aStepInit,tREAL8 aStepLim,tREAL8 aMul=0.5);


       cStdStatRes aSSad1;
       cStdStatRes aSSad0;
       cStdStatRes aSSymInt_1;
       cStdStatRes aSSymInt_0;

       cCalcSaddle  aCalcSInit(1.5,1.0);
       for (const auto & aP3 : aVMaxLoc)
       {
	   cPt2dr aP0 = Proj(aP3);
           cPt2dr aP1 =  aP0;
	   aCalcSBlur.RefineSadlePointFromIm(aImBlur,aP1,true);
           cPt2dr aP2 =  aP1;
	   aCalcSInit.RefineSadlePointFromIm(aImBlur,aP2,true);

	   // tREAL8 aSymInt = aFSym->ComputeValMaxCrown(aP2,10);
           // tREAL8  aSymInt = OptimFilter(aFSym,aP2).ValExtre();

	   tREAL8 aSymInt = aOptimSym.Optim(aP0,1.0,0.05).first;

           bool  Ok = IsPtTest(Proj(aP3));

	   //StdOut() << "SYMM=" << aFSym->ComputeVal(aP0) << "\n";

	   if (Ok)
	   {
               aSSad1.Add(aP3.z());
	       aSSymInt_1.Add(aSymInt);
	   }
	   else
	   {
               aSSad0.Add(aP3.z());
	       aSSymInt_0.Add(aSymInt);
	   }
       }

       if (mHasMasqTest)
       {
          StdOut()  << " ================ STAT LOC CRITERIA ==================== \n";
          StdOut() << " * Saddle , #Ok  Min=" << aSSad1.Min()  
		   << "     #NotOk 90% " << aSSad0.ErrAtProp(0.9) 
		   << "   99%  " << aSSad0.ErrAtProp(0.99) 
		   << "   99.9%  " << aSSad0.ErrAtProp(0.999) 
		   << "\n";

          StdOut() << " * SYM , #Ok=" << aSSymInt_1.Max()   << " %75=" <<  aSSymInt_1.ErrAtProp(0.75)
		   << "  NotOk 50% " << aSSymInt_0.ErrAtProp(0.5) 
		   << "    10%  "    << aSSymInt_0.ErrAtProp(0.10) 
		   << "\n";
       }
       delete aFSym;
    }

    cAutoTimerSegm aTSMakeIm(mTimeSegm,"OTHERS");

    MakeImageSaddlePoints(*mDImIn,*mDImLabel);
    StdOut() << "NBS " << (100.0*aNbSaddle)/aNbTot << " " <<  (100.0*aNbCCSad)/aNbTot 
	    << " " <<  (100.0*aVMaxLoc.size())/aNbTot  << " NB=" << aVMaxLoc.size() << "\n";
    aDImBlur.ToFile("Blurred.tif");

}

/*
 *  Dist= sqrt(5)
 *  T=6.3751
 *  2 sqrt->6.42483
 */


int  cAppliCheckBoardTargetExtract::Exe()
{
   mPhProj.FinishInit();

   if (RunMultiSet(0,0))
   {
       return ResultMultiSet();
   }

   DoOneImage();

   return EXIT_SUCCESS;
}

/* =============================================== */
/*                                                 */
/*                       ::                        */
/*                                                 */
/* =============================================== */

tMMVII_UnikPApli Alloc_CheckBoardCodedTarget(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppliCheckBoardTargetExtract(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpecExtractCheckBoardTarget
(
     "CodedTargetCheckBoardExtract",
      Alloc_CheckBoardCodedTarget,
      "Extract coded target from images",
      {eApF::CodedTarget,eApF::ImProc},
      {eApDT::Image,eApDT::Xml},
      {eApDT::Image,eApDT::Xml},
      __FILE__
);


};
