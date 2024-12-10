/*
#include "CodedTarget.h"
#include "MMVII_2Include_Serial_Tpl.h"
#include "MMVII_Tpl_Images.h"
#include "MMVII_Interpolators.h"
#include "MMVII_Linear2DFiltering.h"
#include "MMVII_ImageMorphoMath.h"
#include "MMVII_2Include_Tiling.h"
#include "MMVII_Sensor.h"
#include "MMVII_HeuristikOpt.h"
#include "MMVII_ExtractLines.h"
#include "MMVII_TplImage_PtsFromValue.h"
*/

#include "cCheckBoardTargetExtract.h"


namespace MMVII
{
bool DebugCB = false;


namespace NS_CHKBRD_TARGET_EXTR { 

/* *************************************************** */
/*                                                     */
/*              cAppliCheckBoardTargetExtract          */
/*                                                     */
/* *************************************************** */

     /* ------------------------------------------------- */
     /*      METHOD FOR CONSTRUCTION OF OBJECT            */
     /* ------------------------------------------------- */

cAppliCheckBoardTargetExtract::cAppliCheckBoardTargetExtract(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli     (aVArgs,aSpec),
   mPhProj          (*this),
   mTimeSegm        (this),
   mThickness       (1.0),
   mOptimSegByRadiom (false),
   mLInitTeta        (5.0),
   mLInitProl        (3.0),
   mLengtProlong     (20.0),
   mStepSeg          (0.5),
   mMaxCostCorrIm    (0.1),
   mNbMaxBlackCB     (10000),
   mPropGrayDCD      (2.0/3.0),
   mNbBlur1          (1),
   mStrShow          (""),
   mScales           {1.0},
   mDistMaxLocSad    (10.0),
   mDistRectInt      (20),
   mMaxNbSP_ML0      (30000),
   mMaxNbSP_ML1      (2000),
   mPtLimCalcSadle   (2,1),
   mThresholdSym     (0.5),
   mRayCalcSym0      (4.0),
   mDistDivSym       (2.0),
   mNumDebugMT       (-1),
   mNumDebugSaddle   (-1),
   mNbMinPtEllipse   (6),
   mTryC             (true),
   mStepHeuristikRefinePos    (-1),
   mStepGradRefinePos (1e-4),
   mZoomVisuDetec    (9),
   mDefSzVisDetec    (150),
   mSpecif           (nullptr),
   mImInCur          (cPt2di(1,1)),
   mDImInCur         (nullptr),
   mImIn0            (cPt2di(1,1)),
   mDImIn0           (nullptr),
   mImBlur           (cPt2di(1,1)),
   mDImBlur          (nullptr),
   mHasMasqTest      (false),
   mMasqTest         (cPt2di(1,1)),
   mImLabel          (cPt2di(1,1)),
   mDImLabel         (nullptr),
   mImTmp            (cPt2di(1,1)),
   mDImTmp           (nullptr),
   mCurScale         (false),
   mMainScale        (true),
   mInterpol         (nullptr)
{
}



cCollecSpecArg2007 & cAppliCheckBoardTargetExtract::ArgObl(cCollecSpecArg2007 & anArgObl)
{
   return  anArgObl
             <<   Arg2007(mNameIm,"Name of image (first one)",{{eTA2007::MPatFile,"0"},eTA2007::FileImage})
            //  <<   Arg2007(mNameSpecif,"Name of target file")
	     <<   Arg2007(mNameSpecif,"Xml/Json name for bit encoding struct",{{eTA2007::XmlOfTopTag,cFullSpecifTarget::TheMainTag}})

   ;
}


cCollecSpecArg2007 & cAppliCheckBoardTargetExtract::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return
	        anArgOpt
             <<  mPhProj.DPPointsMeasures().ArgDirOutOptWithDef("Std")
             <<  mPhProj.DPMask().ArgDirInOpt("TestMask","Mask for selecting point used in detailed mesg/output")
             <<  AOpt2007(mThickness,"Thickness","Thickness for modelizaing line-blur in fine radiom model",{eTA2007::HDV})
             <<  AOpt2007(mLInitTeta,"LSIT","Length Segment Init, for teta",{eTA2007::HDV})
             <<  AOpt2007(mNbBlur1,"NbB1","Number of blurr with sz1",{eTA2007::HDV})
             <<  AOpt2007(mStrShow,"StrV","String for generate Visu : G-lobal L-abels E-llipse N-ums",{eTA2007::HDV})
             <<  AOpt2007(mRayCalcSym0,"SymRay","Ray arround point for initial computation of symetry",{eTA2007::HDV}) 
             <<  AOpt2007(mLInitProl,"LSIP","Length Segment Init, for prolongation",{eTA2007::HDV})
             <<  AOpt2007(mNbMinPtEllipse,"NbMinPtEl","Number minimal of point for ellipse estimation",{eTA2007::HDV})
             <<  AOpt2007(mTryC,"TryC","Try also circle when ellipse fails",{eTA2007::HDV})
             <<  AOpt2007(mStepHeuristikRefinePos,"HeuristikStepRefinePos","Step Gradient-Refine final position with SinC interpol & over sampling (<0 : no refine)",{eTA2007::HDV,eTA2007::Tuning})

             <<  AOpt2007(mStepGradRefinePos,"GradStepRefinePos","Step Gradient-Refine final position with SinC interpol & over sampling (<0 : no refine)",{eTA2007::HDV})
	     <<  AOpt2007(mScales,"Scales","Diff scales of compute (! 0.5 means bigger)",{eTA2007::HDV})
             <<  AOpt2007(mOptimSegByRadiom,"OSBR","Optimize segement by radiometry",{eTA2007::HDV})
             <<  AOpt2007(mNbMaxBlackCB,"NbMaxBlackCB","Number max of point in black part of check-board ",{eTA2007::HDV})
             <<  AOpt2007(mPropGrayDCD,"PropGrayDCD","Proportion of gray for find coding part",{eTA2007::HDV})
             <<  AOpt2007(mNumDebugMT,"NumDebugMT","Num marq target for debug",{eTA2007::Tuning})
             <<  AOpt2007(mNumDebugSaddle,"NumDebugSaddle","Num Saddle point to debug",{eTA2007::Tuning})

   ;
}

     /* ------------------------------------------------- */
     /*      METHOD FOR VISUALIZATION OF RESULTS          */
     /* ------------------------------------------------- */

	// int                   mZoomVisuDetec;  /// zoom Visu detail of detection
	// int                   mDefSzVisDetec;  /// Default Sz Visu detection 

cRGBImage  cAppliCheckBoardTargetExtract::GenImaRadiom(cCdRadiom & aCdR) const
{
	return GenImaRadiom(aCdR,mDefSzVisDetec);
}

cRGBImage  cAppliCheckBoardTargetExtract::GenImaRadiom(cCdRadiom & aCdR,int aSzI) const
{
    bool  aLocalDyn=false; // if true generate the image with dynamic such "Black->0" , "White->255"
    bool  aTheorGray=false; // if true generate the "theoreticall" gray, inside ellipse
    cPt3di  aCoulCenter   = cRGBImage::Red; 
    cPt3di  aCoulEllFront = cRGBImage::Orange; 
    cPt3di  aCol_StrL = cRGBImage::Yellow;  // color for theoreticall straight line

    cPt2di aSz(aSzI,aSzI);
    aCdR.mDec =  ToI(aCdR.mC) - aSz/2;
    cPt2dr aCLoc = aCdR.mC-ToR(aCdR.mDec);

    // Read image from file using shift, and make of it a gray image
    // cRGBImage  aIm = cRGBImage::FromFile(mNameIm,cBox2di(aCdR.mDec,aCdR.mDec+aSz),mZoomVisuDetec);
    // aIm.ResetGray();

    cRGBImage aIm =  RGBImFromGray(*mDImInCur,cBox2di(aCdR.mDec,aCdR.mDec+aSz),1.0,mZoomVisuDetec);



    if (aTheorGray)   // generate the theoretical image + the area (ellipse) of gray modelization
    {
          cTmpCdRadiomPos aCRC(aCdR,mThickness);
	  std::vector<cPt2di> aVEllipse;
          aCdR.ComputePtsOfEllipse(aVEllipse);

          for (const auto & aPix :  aVEllipse)
          {
              auto [aState,aWeightWhite] = aCRC.TheorRadiom(ToR(aPix));
              if (aState != eTPosCB::eUndef)
              {
                 tREAL8 aGr = aCdR.mBlack+ aWeightWhite*(aCdR.mWhite-aCdR.mBlack);
		 cPt3di aCoul (128,aGr,aGr);
		 // cPt3di aCoul (aGr,aGr,aGr);
                 aIm.SetRGBPix(aPix-aCdR.mDec,aCoul);
              }
          }
    }

    // Parse all pixel, read radiom, and modify it to "maximize" dynamic
    if (aLocalDyn)   
    {
          const auto & aDImR = aIm.ImR().DIm();
          for (const auto & aPix : cRect2(cPt2di(0,0),aSz) )
	  {
              tREAL8 aGray = aDImR.GetV(aPix*mZoomVisuDetec);
	      aGray = 255.0 * (aGray-aCdR.mBlack) /(aCdR.mWhite-aCdR.mBlack);
	      aIm.SetGrayPix(aPix,round_ni(aGray));
	  }
    }
    

    // show the points that define the frontier (sub pixel detection)
    if (1)   
    {
	       // compute integer frontier point
          std::vector<cPt2di> aIFront;
          aCdR.FrontBlackCC(aIFront,*mDImTmp,10000);

	       // select frontier point that are not on line, then refine their position
          std::vector<cPt2dr> aEllFr;
	  aCdR.SelEllAndRefineFront(aEllFr,aIFront);
	       //  print the  frontier point
          for (const auto & aPt : aEllFr)
          {
              aIm.SetRGBPoint(aPt-ToR(aCdR.mDec),aCoulEllFront);
              aIm.DrawCircle(aCoulEllFront,aPt-ToR(aCdR.mDec),3.0/mZoomVisuDetec);
          }
    }

    // print the axes of the checkboard
    if (1)
    {
	  for (const auto & aTeta  : aCdR.mTetas)
	  {
              for (int aK= -mZoomVisuDetec * 20 ; aK<=mZoomVisuDetec*20 ; aK++)
	      {
		  tREAL8 aAbsc= aK/ (2.0 * mZoomVisuDetec);
		  cPt2dr aPt = aCLoc + FromPolar(aAbsc,aTeta);

	          aIm.SetRGBPoint(aPt,aCol_StrL); 
	      }
	  }
    }

    aIm.SetRGBPoint(aCLoc,aCoulCenter);

    return aIm;
}

void   cAppliCheckBoardTargetExtract::ComplImaEllipse(cRGBImage & aIm,const  cCdEllipse & aCDE) const
{
      cPt3di  aCol_CornBW = cRGBImage::Red;  // color for teta wih blakc on right (on visualization)
      cPt3di  aCol_CornWB = cRGBImage::Green;  // color for teta wih blakc on left (on visualization)
      cPt3di  aCoulBits_0 = cRGBImage::Blue; 
					      
      std::vector<tREAL8> aVRho = {mSpecif->Rho_0_EndCCB(),mSpecif->Rho_1_BeginCode(),mSpecif->Rho_2_EndCode(),2.0};
      int aNb= 20*aCDE.Ell().LGa() ; 
      for (int aK=0 ; aK<aNb ; aK++)
      {
          for (size_t aKRho=0 ; aKRho<aVRho.size() ; aKRho++)
          {
               tREAL8 aRho = aVRho.at(aKRho);
               cPt2dr aPt = aCDE.Ell().PtOfTeta((2*M_PI*aK)/aNb,aRho);
	       cPt3di aCoul = ((aKRho==1)||(aKRho==2)) ? cRGBImage::Red : cRGBImage::Green;
               aIm.SetRGBPoint(aPt-ToR(aCDE.mDec),aCoul);
          }
     }

     // Draw the 2 corner
     aIm.DrawCircle(aCol_CornWB,aCDE.CornerlEl_WB()-ToR(aCDE.mDec),0.5);
     aIm.DrawCircle(aCol_CornBW,aCDE.CornerlEl_BW()-ToR(aCDE.mDec),0.5);

     // Draw the bits position
     for (const auto & aPBit : mSpecif->BitsCenters())
     {
          auto [aVal,aPt] = aCDE.Length2CodingPart(2.0/3.0,aPBit);
          aIm.DrawCircle(aCoulBits_0,aPt-ToR(aCDE.mDec),0.3);
     }

}

std::string cAppliCheckBoardTargetExtract::NameVisu(const std::string & aPref,const std::string aPost) const
{
     std::string aRes = mPhProj.DirVisuAppli() +  aPref +"-" + LastPrefix(FileOfPath(mNameIm));
     if (aPost!="") aRes = aRes + "-"+aPost;
     return    aRes + ".tif";
}



void cAppliCheckBoardTargetExtract::MakeImageLabels(const std::string & aName,const tDIm & aDIm,const cDataIm2D<tU_INT1> & aDMasq) const
{
    cRGBImage  aRGB = RGBImFromGray<tElem>(aDIm);

    for (const auto & aPix : cRect2(aDIm.Dilate(-1)))
    {
       if (aDMasq.GetV(aPix) >= (int) eTopoMaxLoc)
       {
          cPt3di  aCoul = cRGBImage::Yellow;
	  if (aDMasq.GetV(aPix)== eFilterSym) aCoul = cRGBImage::Green;
	  if (aDMasq.GetV(aPix)== eFilterRadiom) aCoul = cRGBImage::Blue;
	  if (aDMasq.GetV(aPix)>= eFilterEllipse)
	  {
             aCoul = cRGBImage::Red;
	  }
          aRGB.SetRGBPix(aPix,aCoul);
       }
    }
    aRGB.ToJpgFileDeZoom(aName,1);
}

bool cAppliCheckBoardTargetExtract::IsPtTest(const cPt2dr & aPt) const
{
   return mHasMasqTest && (mMasqTest.DIm().DefGetV(ToI(aPt * mCurScale),0) != 0);
}

void cAppliCheckBoardTargetExtract::GenerateVisuFinal() const
{
      //  "G" => G-lobal image with rectangles : "green" : target with code OK, "red" : target shape but no code
      if (contains(mStrShow,'G') )
      {
         cRGBImage  aIm = cRGBImage::FromFile(mNameIm);
         aIm.ResetGray();
         for (auto & aCdt :  mVCdtMerged)
         {
             cPt3di aCoul =  aCdt.Code() ?  (aCdt.IsCircle() ? cRGBImage::Cyan : cRGBImage::Green)  : cRGBImage::Red;
	     aIm.SetRGBrectWithAlpha(ToI(aCdt.mC0),50,aCoul, 0.5);
	     if (aCdt.mScale!= 1.0)
	        aIm.SetRGBBorderRectWithAlpha(ToI(aCdt.mC0),60,10,cRGBImage::Blue, 0.5);

	 }
         aIm.ToJpgFileDeZoom(NameVisu("Glob"),1);
      }
}


void cAppliCheckBoardTargetExtract::GenerateVisuDetail(std::vector<cCdEllipse> & aVCdtEll) const
{
      if  (contains(mStrShow,'L'))
         MakeImageLabels(NameVisu("Label"),*mDImInCur,*mDImLabel);


      // "E" : show the ellipse for each decoded, "N": show nums of decoded (to used for debug)
      if (contains(mStrShow,'E') || contains(mStrShow,'N'))
      {
         int aCptIm = 0;
         for (auto & aCdt :  aVCdtEll)
         {
             if (contains(mStrShow,'E'))
             {
                cRGBImage aRGBIm = GenImaRadiom(aCdt,150);
                ComplImaEllipse(aRGBIm,aCdt);
                aRGBIm.ToJpgFileDeZoom(NameVisu( (aCdt.IsCircle() ? "Circle" : "Ellipse"), ToStr(aCptIm)),1);
             }

             if (contains(mStrShow,'N') )
	         StdOut() << "NumIm=" << aCptIm  <<  " NumDebug=" << aCdt.mNum << "\n";

	     aCptIm++;
         }
      }
      
}

void cAppliCheckBoardTargetExtract::SetLabel(const cPt2dr& aPt,tU_INT1 aLabel)
{
     mDImLabel->SetV(ToI(aPt),aLabel);
}

     /* ------------------------------------------------- */
     /*      METHOD FOR VISUALIZATION FOR COMUTATION      */
     /* ------------------------------------------------- */


/*  
 *
 *  (cos(T) U + sin(T) V)^2  =>  1 + 2 cos(T)sin(T) U.V = 1 + sin(2T) U.V, ValMin  -> 1 -U.V
 *
 */

cCdRadiom cAppliCheckBoardTargetExtract::MakeCdtRadiom(cScoreTetaLine & aSTL,const cCdSym & aCdSym,tREAL8 aThickness)
{
    bool IsMarqed = IsPtTest(aCdSym.mC);
//    static int aCptGlob=0 ; aCptGlob++;
    static int aCptMarq=0 ; if (IsMarqed) aCptMarq++;
    DebugCB = (aCptMarq == mNumDebugMT) && IsMarqed;

    auto aPairTeta = aSTL.Tetas_CheckBoard(mLInitTeta,aCdSym.mC,0.1,1e-3);
    tREAL8 aLength = aSTL.Prolongate(mLInitProl,mLengtProlong,aPairTeta);

    // now restimate teta with a more appropriate lenght ?
    //  aPairTeta = aSTL.Tetas_CheckBoard(aLength,aCdSym.mC,0.1,1e-3);

    auto [aTeta0,aTeta1] = aPairTeta;

    cCdRadiom aCdRadiom(this,aCdSym,*mDImInCur,aTeta0,aTeta1,aLength,aThickness);

    if (! aCdRadiom.mIsOk) 
       return aCdRadiom;

    if (mOptimSegByRadiom)
    {
       aCdRadiom.OptimSegIm(*(aSTL.DIm()),aLength);
    }

    return aCdRadiom;
}


void cAppliCheckBoardTargetExtract::ReadImagesAndBlurr()
{
    /* [0]    Initialise : read image and mask */

    cAutoTimerSegm aTSInit(mTimeSegm,"0-Init");

	// [0.0]   read image

	// [0.1]   initialize labeling image 
    mDImLabel =  &(mImLabel.DIm());
    mDImLabel->Resize(mSzImCur);
    mDImLabel->InitCste(eNone);

    mDImTmp = &(mImTmp.DIm() );
    mDImTmp->Resize(mSzImCur);
    mDImTmp->InitCste(0);

    /* [1]   Compute a blurred image => less noise, less low level saddle */

    cAutoTimerSegm aTSBlur(mTimeSegm,"1-Blurr");

    mImBlur  = mImInCur.Dup(); // create image blurred with less noise
    mDImBlur = &(mImBlur.DIm());

    SquareAvgFilter(*mDImBlur,mNbBlur1,1,1); // 1,1 => Nbx,Nby
}

void cAppliCheckBoardTargetExtract::ComputeTopoSadles()
{
    cAutoTimerSegm aTSTopoSad(mTimeSegm,"2.0-TopoSad");
    cRect2 aRectInt = mDImInCur->Dilate(-mDistRectInt); // Rectangle excluding point too close to border

         // 2.1  point with criteria on conexity of point > in neighoor

    for (const auto & aPix : aRectInt)
    {
        if (FlagSup8Neigh(*mDImBlur,aPix).NbConComp() >=4)
	{
	    SetLabel(ToR(aPix),eTopo0);
	}
    }

         // 2.2  as often there 2 "touching" point with this criteria
	 // select 1 point in conected component

    cAutoTimerSegm aTSMaxCC(mTimeSegm,"2.1-MaxCCSad");
    std::vector<cPt2di>  aVCC;
    const std::vector<cPt2di> & aV8 = Alloc8Neighbourhood();

    for (const auto& aPix : *mDImLabel)
    {
         if (mDImLabel->GetV(aPix)==eTopo0)
	 {
             ConnectedComponent(aVCC,*mDImLabel,aV8,aPix,eTopo0,eTopoTmpCC);
	     cWhichMax<cPt2di,tREAL8> aBestPInCC;
	     for (const auto & aPixCC : aVCC)
	     {
                 aBestPInCC.Add(aPixCC,CriterionTopoSadle(*mDImBlur,aPixCC));
	     }

	     cPt2di aPCC = aBestPInCC.IndexExtre();
	     SetLabel(ToR(aPCC),eTopoMaxOfCC);
	 }
    }
}

/**  The saddle criteria is defined by fitting a quadratic function on the image. Having computed the eigen value of quadratic function :
 *
 *      - this criteria is 0 if they have same sign
 *      - else it is the smallest eigen value
 *
 *   This fitting is done a smpothed version of the image :
 *      - it seem more "natural" for fitting a smooth model
 *      - it limits the effect of delocalization
 *      - it (should be) is not a problem as long as the kernel is smaller than the smallest checkbord we want to detect
 *
 *    As it is used on a purely relative criteria, we dont have to bother how it change the value.
 *     
 */
void cAppliCheckBoardTargetExtract::SaddleCritFiler() 
{
    cAutoTimerSegm aTSCritSad(mTimeSegm,"3.0-CritSad");

    cCalcSaddle  aCalcSBlur(Norm2(mPtLimCalcSadle)+0.001,1.0); // structure for computing saddle criteria

       // [3.1]  compute for each point the saddle criteria
    for (const auto& aPix : *mDImLabel)
    {
         if (mDImLabel->GetV(aPix)==eTopoMaxOfCC)
	 {
             tREAL8 aCritS = aCalcSBlur.CalcSaddleCrit(*mDImBlur,aPix);
             mVCdtSad.push_back(cCdSadle(ToR(aPix),aCritS,IsPtTest(ToR(aPix))) );
	 }
    }
    mNbSads.push_back(mVCdtSad.size()); // memo size for info 

    //   [3.2]    sort by decreasing criteria of saddles => "-"  aCdt.mSadCrit + limit size
    cAutoTimerSegm aTSMaxLoc(mTimeSegm,"3.1-MaxLoc");

    SortOnCriteria(mVCdtSad,[](const auto & aCdt){return - aCdt.mSadCrit;});
    ResizeDown(mVCdtSad,mMaxNbSP_ML0);   
    mNbSads.push_back(mVCdtSad.size()); // memo size for info 


    //   [3.3]  select  MaxLocal
    mVCdtSad = FilterMaxLoc((cPt2dr*)nullptr,mVCdtSad,[](const auto & aCdt) {return aCdt.mC;}, mDistMaxLocSad);
    mNbSads.push_back(mVCdtSad.size()); // memo size for info 

    //   [3.3]  select KBest + MaxLocal
    //  limit the number of point , a bit rough but first experiment show that sadle criterion is almost perfect on good images
    // mVCdtSad.resize(std::min(mVCdtSad.size(),size_t(mMaxNbMLS)));
    ResizeDown(mVCdtSad,mMaxNbSP_ML1);
    mNbSads.push_back(mVCdtSad.size()); // memo size for info 

    for (const auto & aCdt : mVCdtSad)
        SetLabel(aCdt.mC,eTopoMaxLoc);
}

void cAppliCheckBoardTargetExtract::SymetryFiler()
{
    cAutoTimerSegm aTSSym(mTimeSegm,"4-SYM");
    cFilterDCT<tREAL4> * aFSym = cFilterDCT<tREAL4>::AllocSym(mImInCur,0.0,mRayCalcSym0,1.0);
    cOptimByStep<2> aOptimSym(*aFSym,true,mDistDivSym);

    for (auto & aCdtSad : mVCdtSad)
    {
        auto [aValSym,aNewP] = aOptimSym.Optim(aCdtSad.mC,1.0,0.01);  // Pos Init, Step Init, Step Lim
        aCdtSad.mC = aNewP;

        if (aValSym < mThresholdSym)
        {
           mVCdtSym.push_back(cCdSym(aCdtSad,aValSym));
	   SetLabel(aNewP,eFilterSym);
        }
	else if (IsPtTest(aCdtSad.mC))
	{
           StdOut()  << "SYMREFUT,  C=" << aCdtSad.mC << " ValSym=" << aValSym << "\n";
	}
    }

    delete aFSym;
}

void  cAppliCheckBoardTargetExtract::AddCdtE(const cCdEllipse & aCDE)
{
     cCdMerged aNewCdM(mDImIn0,aCDE,mCurScale);

     for (auto & aCdM : mVCdtMerged)
     {
          tREAL8 aD = Norm2(aNewCdM.mC0-aCdM.mC0);

	  if (aD < 10.0)
          {
	      if (aNewCdM.Code() && (! aCdM.Code()) )
                 aCdM  = aNewCdM;
	      return;
	  }
     }

     mVCdtMerged.push_back(aNewCdM);
}

void  cAppliCheckBoardTargetExtract::DoExport()
{
     cSetMesPtOf1Im  aSetM(FileOfPath(mNameIm));
     for (const auto & aCdtM : mVCdtMerged)
     {
         if (aCdtM.Code())
         {
             std::string aCode = aCdtM.Code()->Name() ;
             cMesIm1Pt aMesIm(aCdtM.mC0,aCode,1.0);
             aSetM.AddMeasure(aMesIm);
             Tpl_AddOneObjReportCSV(*this,mIdExportCSV,aMesIm);
         }
     }

     aSetM.SortMes();
     mPhProj.SaveMeasureIm(aSetM);
}

void cAppliCheckBoardTargetExtract::DoOneImage() 
{
   mIdExportCSV       = "CheckBoardCodedTarget-" + mNameIm;
   //  Create a report with header computed from type
   Tpl_AddHeaderReportCSV<cMesIm1Pt>(*this,mIdExportCSV,false);
   // Redirect the reports on folder of result
   SetReportRedir(mIdExportCSV,mPhProj.DPPointsMeasures().FullDirOut());



    mInterpol = new   cTabulatedDiffInterpolator(cSinCApodInterpolator(5.0,5.0));

    mSpecif = cFullSpecifTarget::CreateFromFile(mNameSpecif);

    mImIn0 =  tIm::FromFile(mNameIm);
    mDImIn0 = &mImIn0.DIm() ;
    mSzIm0 = mDImIn0->Sz();
    
    // [0.2]   Generate potential mask for test points
    mHasMasqTest = mPhProj.ImageHasMask(mNameIm);
    if (mHasMasqTest)
       mMasqTest =  mPhProj.MaskOfImage(mNameIm,*mDImIn0);


    for (const auto & aScale : mScales)
    {
        cAutoTimerSegm aTSRefine(mTimeSegm,"00-Scaling");
        auto aScIm = mImIn0.Scale(aScale);
        DoOneImageAndScale(aScale,aScIm);
    }

    if (mStepHeuristikRefinePos>0)
    {
        cAutoTimerSegm aTSRefine(mTimeSegm,"RefineHeuristik");
        for (auto & aCdtM : mVCdtMerged)
            aCdtM.HeuristikOptimizePosition(*mInterpol,mStepHeuristikRefinePos);
    }
    if (mStepGradRefinePos>0)
    {
        cAutoTimerSegm aTSRefine(mTimeSegm,"RefineGradient");
        for (auto & aCdtM : mVCdtMerged)
            aCdtM.GradOptimizePosition(*mInterpol,mStepGradRefinePos);
    }


    cAutoTimerSegm aTSMakeIm(mTimeSegm,"OTHERS");

    GenerateVisuFinal();
    DoExport();
    delete mSpecif;
    delete mInterpol;
}

void cAppliCheckBoardTargetExtract::DoOneImageAndScale(tREAL8 aScale,const  tIm & anIm ) 
{ 
    mVCdtSad.clear();
    mVCdtSym.clear();
    mCurScale     = aScale;

    mImInCur  = anIm;
    mDImInCur = &mImInCur.DIm();
    mSzImCur  = mDImInCur->Sz();

    if (IsInit(&mNumDebugSaddle))
       cCdSadle::TheNum2Debug= mNumDebugSaddle ;

    /* [0]    Initialise : read image ,  mask + Blurr */
    ReadImagesAndBlurr();
    /* [2]  Compute "topological" saddle point */
    ComputeTopoSadles();
    /* [3]  Compute point that are max local of  saddle point criteria */
    SaddleCritFiler();
    /* [4]  Calc Symetry criterion */
    SymetryFiler();

    /* [5]  Compute lines, radiom model & correlation */
    std::vector<cCdRadiom> aVCdtRad;
    cAutoTimerSegm aTSRadiom(mTimeSegm,"Radiom");
    {
        cCubicInterpolator aCubI(-0.5);
        cScoreTetaLine  aSTL(*mDImInCur,aCubI,mStepSeg);
        for (const auto & aCdtSym : mVCdtSym)
        {
            cCdRadiom aCdRad = MakeCdtRadiom(aSTL,aCdtSym,mThickness);
	    if (aCdRad.mCostCorrel <= mMaxCostCorrIm)
	    {
               aVCdtRad.push_back(aCdRad);
	       SetLabel(aCdRad.mC,eFilterRadiom);
	    }
        }
    }

    /* [6]  Compute model of geometry, ellipse & code */
    std::vector<cCdEllipse> aVCdtEll;
    int aNbEllWCode = 0;
    cAutoTimerSegm aTSEllipse(mTimeSegm,"Ellipse");
    {
        for (const auto & aCdtRad : aVCdtRad)
        {
           std::vector<bool>  TryCE = {false}; // Do we do the try in circle or ellipse mode
	   if (mTryC)  TryCE.push_back(true);
	   bool GotIt = false;
	   for (size_t aKC=0 ; (aKC<TryCE.size()) && (!GotIt) ; aKC++)
	   {
               cCdEllipse aCDE(aCdtRad,*mDImTmp,mNbMaxBlackCB,TryCE.at(aKC));
	       if (aCDE.IsOk())
	       {
	          SetLabel(aCDE.mC,eFilterEllipse);
                  aCDE.DecodeByL2CP(mPropGrayDCD);
                  aVCdtEll.push_back(aCDE);
	          if (aCDE.Code())
	          {
                     // StdOut() << "aCDE.mC,eFilterCodedTargetaCDE.mC,eFilterCodedTarget \n";
                     SetLabel(aCDE.mC,eFilterCodedTarget);
	             aNbEllWCode++;
		     GotIt = true;
	          }
		  AddCdtE(aCDE);
	       }
	   }
        }
    }

    cAutoTimerSegm aTSMakeIm(mTimeSegm,"OTHERS");
    if (mMainScale)
    {
      GenerateVisuDetail(aVCdtEll);
      StdOut()  << "NB Cd,  SAD: " << mNbSads
	      << " SYM:" << mVCdtSym.size() 
	      << " Radiom:" << aVCdtRad.size() 
	      << " Ellipse:" << aVCdtEll.size() 
	      << " Code:" << aNbEllWCode << "\n";
    }

    mMainScale = false;
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


};  // ===================  NS_CHKBRD_TARGET_EXTR

using namespace NS_CHKBRD_TARGET_EXTR;
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
