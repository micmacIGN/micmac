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

const tBinCodeTarg & cCodesOf1Target::CodeOfNumC(int aNum) const
{
    return  mCodes.at(aNum);
}

/**************************************************/
/*                                                */
/*           cParamCodedTarget                    */
/*                                                */
/**************************************************/

void cParamCodedTarget::AddData(const cAuxAr2007 & anAux)
{
    MMVII::AddData(cAuxAr2007("NbRedond",anAux),mNbRedond);
    MMVII::AddData(cAuxAr2007("RatioBar",anAux),mRatioBar);
    MMVII::AddData(cAuxAr2007("RW0",anAux),mRhoWhite0);
    MMVII::AddData(cAuxAr2007("RB0",anAux),mRhoBlack0);
    MMVII::AddData(cAuxAr2007("NbC",anAux),mNbCircle);
    MMVII::AddData(cAuxAr2007("ThC",anAux),mThCircle);
    MMVII::AddData(cAuxAr2007("DistFM",anAux),mDistMarkFid);
    MMVII::AddData(cAuxAr2007("DistBorderFM",anAux),mBorderMarkFid);
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
   mRatioBar      (1.2),
   mRhoWhite0     (0),// (1.5),
   mRhoBlack0     (0),// (2.5),
   mNbCircle      (1),
   mThCircle      (4),
   mDistMarkFid   (2.5),
   mBorderMarkFid (1.5),
   mRadiusFidMark (0.6),
   mTetaCenterFid (M_PI/4.0),
   mNbPaqFid      (-1),  // Marqer of No Init
   mNbFidByPaq    (3),
   mGapFid        (1.0),
   mScaleTopo     (0.25),
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
  if (mNbPaqFid<=0)
     mNbPaqFid = mNbRedond ;
  mSzBin = cPt2di(mNbPixelBin,mNbPixelBin);
  mRhoCodage0  = mRhoWhite0 + mRhoBlack0;

  mRhoCodage1  = mRhoCodage0 + mNbCircle * mThCircle;
  mRhoFidMark  = mRhoCodage1  + mDistMarkFid;
  mRhoEnd      = mRhoFidMark  + mBorderMarkFid;

  mRho_00_TopoB   = mRhoWhite0 * mScaleTopo;
  mRho_000_TopoW  = mRho_00_TopoB * mScaleTopo;
  mRho_0000_TopoB = mRho_000_TopoW * mScaleTopo;

  mMidle = ToR(mSzBin) / 2.0;
  mScale = mNbPixelBin / (2.0 * mRhoEnd);

  std::vector<int> aVNbSub;
  for (int aKCirc = 0 ; aKCirc< mNbCircle ; aKCirc++)
  {
      double aRho0 = mRhoCodage0 + aKCirc * mThCircle;
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


  for (int aK=0 ; aK< mNbFidByPaq ; aK++)
  {
      double aAmpl = mNbFidByPaq +  mGapFid;
      double aInd = (aK+0.5- mNbFidByPaq /2.0) / aAmpl;

      mTetasQ.push_back(mTetaCenterFid+aInd*((2*M_PI)/mNbPaqFid));
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

     for (const auto & aPix : aDImT)
     {
         cPt2dr  aPixN =  Pix2Norm(aPix);
         cPt2dr  aRT  = ToPolar(aPixN,0.0);
	 double  aRho = aRT.x();

	 bool IsW = true;

         if (aRho< mRhoCodage1)  
         {
	     if (aRho>=mRhoCodage0)
	     {
	        double  aTeta = aRT.y();
		if (aTeta < 0)
                   aTeta += 2 *M_PI;
                int aIndRho = std::max(0,std::min(mNbCircle-1,round_down((aRho-mRhoCodage0)/mThCircle)));
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
		  if (aRho<mRho_0000_TopoB)
		  {
                      IsW= false;
		  }
		  else if (aRho<mRho_000_TopoW)
		  {
                      IsW= true;
		  }
		  else if (aRho<mRho_00_TopoB)
		  {
                      IsW= false;
		  }
		  else if (aRho<mRhoWhite0)
		  {
                      IsW= true;
		  }
		  else
		  {
                      IsW= false;
		  }
	     }
         }
         else
         {
              // Outside => Only fid marks, done after
         }

         int aVal = IsW ? 255 : 0;
         aDImT.SetV(aPix,aVal);
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
