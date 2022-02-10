#include "CodedTarget.h"
#include "include/MMVII_2Include_Serial_Tpl.h"


namespace MMVII
{

namespace  cNS_CodedTarget
{


/* ****    cSetCodeOf1Circle  ***** */
  
cSetCodeOf1Circle::cSetCodeOf1Circle(double aRho0,int aK,int aN):
   mRho0   (aRho0),
   mK      (aK),
   mN      (aN),
   mVSet   (SubKAmongN<tBinCodeTarg>(aK,aN))
{
}
int  cSetCodeOf1Circle::NbSub() const {return mVSet.size();}

const tBinCodeTarg & cSetCodeOf1Circle::CodeOfNum(int aNum) const
{
    return  mVSet.at(aNum);
}

int cSetCodeOf1Circle::N() const {return mN;}
int cSetCodeOf1Circle::K() const {return mK;}

/* ****    cCodesOf1Target   ***** */

cCodesOf1Target::cCodesOf1Target(int aNum) :
   mNum   (aNum)
{
}

void cCodesOf1Target::AddOneCode(const tBinCodeTarg & aCode)
{
    mCodes.push_back(aCode);
}

void  cCodesOf1Target::Show()
{
    StdOut()  << "Num=" << mNum << " Codes=";
    for (const auto & aCode : mCodes)
         StdOut()  << aCode;
    StdOut()  << "\n";
}

const tBinCodeTarg & cCodesOf1Target::CodeOfNumC(int aK) const
{
    return  mCodes.at(aK);
}

int cCodesOf1Target::Num() const {return mNum;}


/**************************************************/
/*                                                */
/*           cParamCodedTarget                    */
/*                                                */
/**************************************************/

void cParamCodedTarget::AddData(const cAuxAr2007 & anAux)
{
    MMVII::AddData(cAuxAr2007("NbRedond",anAux),mNbRedond);
    MMVII::AddData(cAuxAr2007("RatioBar",anAux),mRatioBar);
    MMVII::AddData(cAuxAr2007("RhoTargetC",anAux),mRhoTargetC);
    MMVII::AddData(cAuxAr2007("NbC",anAux),mNbCircle);
    MMVII::AddData(cAuxAr2007("ThC",anAux),mThCircle);
    MMVII::AddData(cAuxAr2007("DistFM",anAux),mDistMarkFid);
    MMVII::AddData(cAuxAr2007("BorderB",anAux),mBorderB);
    MMVII::AddData(cAuxAr2007("BorderW",anAux),mBorderW);
    MMVII::AddData(cAuxAr2007("RadiusFM",anAux),mRadiusFidMark);
    MMVII::AddData(cAuxAr2007("Teta0FM",anAux),mTetaCenterFid);
    MMVII::AddData(cAuxAr2007("NbPaqFM",anAux),mNbPaqFid);
    MMVII::AddData(cAuxAr2007("NbByPaqFM",anAux),mNbFidByPaq);
    MMVII::AddData(cAuxAr2007("NbGapFM",anAux),mGapFid);
    MMVII::AddData(cAuxAr2007("ScaleTopo",anAux),mScaleTopo);
    MMVII::AddData(cAuxAr2007("NbPixBin",anAux),mNbPixelBin);
}

void AddData(const  cAuxAr2007 & anAux,cParamCodedTarget & aPCT)
{
   aPCT.AddData(anAux);
}

cParamCodedTarget::cParamCodedTarget() :
   mNbRedond      (2),
   mRhoTargetC    (1.0),
   mRatioBar      (0.8),
   mNbCircle      (1),
   mThCircle      (4),
   mDistMarkFid   (0.5),
   mBorderB       (0.5),
   mBorderW       (0.1),
   mRadiusFidMark (0.6),
   mTetaCenterFid (M_PI/4.0),
   mNbPaqFid      (0),  //  -1 Marqer of No Init  , 0 None
   mNbFidByPaq    (3),
   mGapFid        (1.0),
   mScaleTopo     (0.5),
   mNbPixelBin    (1800),
   mDecP          ({1,1})  // "Fake" init 4 now
{
}

cPt2dr cParamCodedTarget::Pix2Norm(const cPt2di & aPix) const
{
   return (ToR(aPix)-mMidle) / mScale;
}
cPt2dr cParamCodedTarget::Norm2PixR(const cPt2dr & aP) const
{
    return mMidle + aP * mScale;
}
cPt2di cParamCodedTarget::Norm2PixI(const cPt2dr & aP) const
{
    return ToI(Norm2PixR(aP));
}

int&    cParamCodedTarget::NbRedond() {return mNbRedond;}
int&    cParamCodedTarget::NbCircle() {return mNbCircle;}
double& cParamCodedTarget::RatioBar() {return mRatioBar;}

void cParamCodedTarget::Finish()
{
  if (mNbPaqFid<0)
     mNbPaqFid = mNbRedond ;
  mSzBin = cPt2di(mNbPixelBin,mNbPixelBin);

  mRhoCodage1  = mRhoTargetC + mNbCircle * mThCircle;
  mRhoFidMark  = mRhoCodage1  + mDistMarkFid;
  mRhoBlackB =  mRhoFidMark  + mBorderB;
  mRhoEnd      = mRhoBlackB + mBorderW;

  mMidle = ToR(mSzBin) / 2.0;
  mScale = mNbPixelBin / (2.0 * mRhoEnd);

  std::vector<int> aVNbSub;
  for (int aKCirc = 0 ; aKCirc< mNbCircle ; aKCirc++)
  {
      double aRho0 = mRhoTargetC + aKCirc * mThCircle;
      double aRho1 = aRho0 + mThCircle;
      double aRhoM = (aRho0 + aRho1) / 2.0;
      int aNb = std::max(2,round_up((mRatioBar* 2*M_PI*aRhoM)/mNbRedond));
      /*
      double aProp = (mNbCircle-aKCirc) /  mRhoCodage1;
      int aK = round_down(aProp*aNb);
      if (mNbCircle==1)
      */
      int aK = aNb/2;
      aK =std::max(1,std::min(aNb-1,aK));

      mVecSetOfCode.push_back(cSetCodeOf1Circle(aRho0,aK,aNb));
      aVNbSub.push_back( mVecSetOfCode.back().NbSub());
      StdOut()  << " aK=" << aK << " N=" << aNb  << " R=" << aRhoM << " C(k,n)=" <<  aVNbSub.back() << "\n";
  }
  mDecP = cDecomposPAdikVar(aVNbSub);
  StdOut()  << " NbTarget="   << NbCodeAvalaible() << "\n";


  if (mNbPaqFid)
  {
      for (int aK=0 ; aK< mNbFidByPaq ; aK++)
      {
          double aAmpl = mNbFidByPaq +  mGapFid;
          double aInd = (aK+0.5- mNbFidByPaq /2.0) / aAmpl;

          mTetasQ.push_back(mTetaCenterFid+aInd*((2*M_PI)/mNbPaqFid));
      }
  }
}

cCodesOf1Target cParamCodedTarget::CodesOfNum(int aNum)
{
   cCodesOf1Target aRes(aNum);
   std::vector<int>  aVDec = mDecP.DecomposSizeBase(aNum);

   for (int aKCirc=0 ; aKCirc<mNbCircle ; aKCirc ++)
   {
       int aDec = aVDec.at(aKCirc);
       aRes.AddOneCode(mVecSetOfCode.at(aKCirc).CodeOfNum(aDec));
   }
   return aRes;
}

int cParamCodedTarget::NbCodeAvalaible() const
{
   return  mDecP.MulBase();
}

tImTarget  cParamCodedTarget::MakeIm(const cCodesOf1Target & aSetCodesOfT)
{
     tImTarget aImT(mSzBin);
     tDataImT  & aDImT = aImT.DIm();

     int aDW = mBorderW * mScale;
     int aDB = (mBorderW+mBorderB) * mScale;

     //  Parse all pixels of image
     for (const auto & aPix : aDImT)
     {
         cPt2dr  aPixN =  Pix2Norm(aPix);     // "Nomalized" coordinate
         cPt2dr  aRT  = ToPolar(aPixN,0.0);   // Polar then Rho teta
	 double  aRho = aRT.x();
         double  aTeta = aRT.y();

	 bool IsW = true;  // Default is white

         if (aRho< mRhoCodage1)  
         {
             // Generate the stars
	     if (aRho>=mRhoTargetC)
	     {
		if (aTeta < 0)
                   aTeta += 2 *M_PI;
                int aIndRho = std::max(0,std::min(mNbCircle-1,round_down((aRho-mRhoTargetC)/mThCircle)));
		const cSetCodeOf1Circle & aSet1C =  mVecSetOfCode.at(aIndRho);
		int aN  = aSet1C.N();

		int aIndTeta = round_down((aTeta*aN*mNbRedond)/(2*M_PI));
		aIndTeta = aIndTeta % aN;
                const tBinCodeTarg & aCodeBin = aSetCodesOfT.CodeOfNumC(aIndRho);
                if (aCodeBin.IsInside(aIndTeta))
                   IsW = false;
	     }
	     else
	     {
                  // double  mScaleTopo
                  double aRatio = mRhoTargetC / std::max(1.0/mScale,aRho);  // 1/mScale => smallest pixel
		  double aRInd = log(aRatio) / log(mScaleTopo);
		  int aInd = round_down(aRInd);
		  IsW = ((aInd%2) == 0);
	     }
         }
         else
         {
              // Outside => border and fid marks (done after)
	      int aDInter = aDImT.Interiority(aPix);
	      IsW = (aDInter<aDW) || (aDInter>=aDB);
         }

         int aVal = IsW ? 255 : 0;
         aDImT.SetV(aPix,aVal);
     }

     {
         std::string aStr = ToStr(aSetCodesOfT.Num(),2);
         cIm2D<tU_INT1> aImStr = ImageOfString(aStr,1);
         cDataIm2D<tU_INT1> & aDImStr = aImStr.DIm();
	 cPt2di aNbPixStr = aDImStr.Sz();
	 double mHString = 0.7;
	 double  aScaleStr =  (mHString/aNbPixStr.y()) * mScale;
         // StdOut() << "STR=[" << aStr <<  "] ScSt " << aScaleStr << "\n";

	 cPt2dr aSzStr = ToR(aNbPixStr) * aScaleStr;
	 // cPt2di aP0 = ToI(aMidStr-aSzStr/2.0);
	 cPt2di aP0(aDB,aDB);
	 cPt2di aP1 = aP0 + ToI(aSzStr);

	 cRect2 aBox(aP0,aP1);

	 for (const auto & aPix : aBox)
	 {
             cPt2di aPixSym = mSzBin-aPix-cPt2di(1,1);

             cPt2di aPStr =  ToI(ToR(aPix-aP0)/aScaleStr);
	     int IsCar = aDImStr.DefGetV(aPStr,0);
             int aCoul = IsCar ? 255 : 0;
             aDImT.SetV(aPix,aCoul);
             aDImT.SetV(aPixSym,aCoul);
	 }

	 // StdOut() << " MMM " << aMidStr << aP0 << aP1 << "\n";


     }

     for (int aKQ=0 ; aKQ<mNbPaqFid ; aKQ++)
     {
         for (const auto & aDTeta :mTetasQ)
	 {
             double aTeta = aKQ * (2*M_PI/mNbPaqFid) + aDTeta;
             cPt2dr  aCenterN = FromPolar(mRhoFidMark,aTeta);
             cPt2dr  aCenterP = Norm2PixR(aCenterN);
	     double  aRadiusPix  = mRadiusFidMark * mScale;
	     cPt2dr aPRad(aRadiusPix,aRadiusPix);
	     cRect2 aBoxP(Pt_round_down(aCenterP-aPRad),Pt_round_up(aCenterP+aPRad));

             for (const auto & aPix  : aBoxP)
             {
                 double  aDist  = Norm2(ToR(aPix)-aCenterP);
		 if (aDist<=aRadiusPix)
                    aDImT.SetV(aPix,0);
             }
	 }
     }

     aImT = aImT.GaussDeZoom(3);
     return aImT;
}

/*  *********************************************************** */
/*                                                              */
/*             cAppliGenCodedTarget                             */
/*                                                              */
/*  *********************************************************** */

class cAppliGenCodedTarget : public cMMVII_Appli
{
     public :
        cAppliGenCodedTarget(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);

     private :


        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;


	int                mPerGen;  // Pattern of numbers
	cParamCodedTarget  mPCT;
};


/* *************************************************** */
/*                                                     */
/*              cAppliGenCodedTarget                   */
/*                                                     */
/* *************************************************** */


cAppliGenCodedTarget::cAppliGenCodedTarget(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli  (aVArgs,aSpec)
{
}

cCollecSpecArg2007 & cAppliGenCodedTarget::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
 return
      anArgObl
          <<   Arg2007(mPerGen,"Periode of generated, give 1 to generate all")
   ;
}

cCollecSpecArg2007 & cAppliGenCodedTarget::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return anArgOpt
          << AOpt2007(mPCT.NbRedond(), "Redund","Number of repetition inside a circle",{eTA2007::HDV})
          << AOpt2007(mPCT.NbCircle(), "NbC","Number of circles",{eTA2007::HDV})
          << AOpt2007(mPCT.RatioBar(), "RatioBar","Ratio Sz of coding bar",{eTA2007::HDV})
   ;
}


int  cAppliGenCodedTarget::Exe()
{
   mPCT.Finish();

   for (int aNum=0 ; aNum<mPCT.NbCodeAvalaible() ; aNum+=mPerGen)
   {
      cCodesOf1Target aCodes = mPCT.CodesOfNum(aNum);
      aCodes.Show();
      tImTarget aImT= mPCT.MakeIm(aCodes);
      
      std::string aName = "Target_" + ToStr(aNum) + ".tif";
      aImT.DIm().ToFile(aName);
      // FakeUseIt(aCodes);
   }

   SaveInFile(mPCT,"Target_Spec.xml");


   return EXIT_SUCCESS;
}
/*
*/
};


/* =============================================== */
/*                                                 */
/*                       ::                        */
/*                                                 */
/* =============================================== */
using namespace  cNS_CodedTarget;

tMMVII_UnikPApli Alloc_GenCodedTarget(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppliGenCodedTarget(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpecGenCodedTarget
(
     "CodedTargetGenerate",
      Alloc_GenCodedTarget,
      "Generate images for coded target",
      {eApF::CodedTarget},
      {eApDT::Console},
      {eApDT::Image},
      __FILE__
);


};
