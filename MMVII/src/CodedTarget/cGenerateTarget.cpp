#include "CodedTarget.h"
#include "include/MMVII_2Include_Serial_Tpl.h"

/*
    1313  966
    1188  936  6
*/


namespace MMVII
{

static constexpr double ExtRatioW1   = 1.25;
static constexpr double ExtRatioCode = 1.6;
static constexpr double ExtRatioBrd =  1.65;

static constexpr double ExtTextMargin =  0.3;  // Margin if we want text not sticked to coding

namespace  cNS_CodedTarget
{


/* ****    cSetCodeOf1Circle  ***** */
  
cSetCodeOf1Circle::cSetCodeOf1Circle(const std::vector<int> & aVCard,int aN):
   mVCards    (aVCard),
   mN         (aN)
{
   for (const auto &aCard : aVCard)
       AppendIn(mVSet,SubKAmongN<tBinCodeTarg>(aCard,aN));
}



int  cSetCodeOf1Circle::NbSub() const {return mVSet.size();}

const tBinCodeTarg & cSetCodeOf1Circle::CodeOfNum(int aNum) const
{
    return  mVSet.at(aNum);
}

int cSetCodeOf1Circle::N() const {return mN;}
// int cSetCodeOf1Circle::K() const {return mK;}

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
    MMVII::AddData(cAuxAr2007("NbC",anAux),mNbCircle);


    MMVII::AddData(cAuxAr2007("ThicknessTargetC",anAux),mThTargetC);
    MMVII::AddData(cAuxAr2007("ThcknessStars",anAux),mThStars);
    MMVII::AddData(cAuxAr2007("ThBrdWhiteInt",anAux),mThBrdWhiteInt);
    MMVII::AddData(cAuxAr2007("ThBorderB",anAux),mThBrdBlack);
    MMVII::AddData(cAuxAr2007("ThBorderW",anAux),mThBrdWhiteExt);

    MMVII::AddData(cAuxAr2007("ScaleTopo",anAux),mScaleTopo);
    MMVII::AddData(cAuxAr2007("NbPixBin",anAux),mNbPixelBin);
}

void AddData(const  cAuxAr2007 & anAux,cParamCodedTarget & aPCT)
{
   aPCT.AddData(anAux);
}

void cParamCodedTarget::InitFromFile(const std::string & aNameFile)
{
    ReadFromFile(*this,aNameFile);
    Finish();
}


cParamCodedTarget::cParamCodedTarget() :
   mCodeExt       (true),
   mNbRedond      (2),
   mNbCircle      (1),
   mThTargetC     (0.0), // (0.8),
   mThStars       (4),
   mThBlCircExt   (0.0), // (0.5),
   mThBrdWhiteInt (1.0),
   mThBrdBlack    (mCodeExt ? 0 : 0.7),
   mThBrdWhiteExt (mCodeExt ? 0 : 0.1),
   mThTxt         (1.5),
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

void cParamCodedTarget::Finish()
{
  mThRing = mThStars / mNbCircle;
  mRhoEndTargetC     = mThTargetC;
  mRhoEndStar        = mRhoEndTargetC     +  mThStars;
  mRhoEndBlackCircle = mRhoEndStar        +  mThBlCircExt;
  mRhoEnBrdWhiteInt  = mRhoEndBlackCircle +  mThBrdWhiteInt;
  mRhoEndBrdBlack    = mRhoEnBrdWhiteInt  +  mThBrdBlack;
  mRhoEndBrdWhiteExt = mRhoEndBrdBlack    +  mThBrdWhiteExt;
  mRhoEndTxt    =  mRhoEndBrdWhiteExt + mThTxt ;

  int aWidthBin =   mNbPixelBin * (mNbPixelBin ? 1 : (mRhoEndTxt/mRhoEndBrdWhiteExt));
  mSzBin = cPt2di(aWidthBin,aWidthBin);

  mMidle = ToR(mSzBin) / 2.0;
  mScale = mNbPixelBin / (2.0 * mRhoEndBrdWhiteExt);
  if (mCodeExt)
  {
      mScale = mNbPixelBin / (2.0 * ExtRatioBrd * mRhoEndStar);
  }

  std::vector<int> aVNbSub;
  for (int aKCirc = 0 ; aKCirc< mNbCircle ; aKCirc++)
  {
      // double aRho0 = mRhoEndTargetC + aKCirc * mThRing;
      int  aNb = mCodeExt ? 9 :8;

      std::vector<int>  aVK;
      if (mCodeExt)
      {
          for (int aK=1 ; aK<aNb ; aK+=2 )
	      aVK.push_back(aK);       
      }
      else
         aVK.push_back(aNb/2);

      mVecSetOfCode.push_back(cSetCodeOf1Circle(aVK,aNb));
      aVNbSub.push_back( mVecSetOfCode.back().NbSub());
      StdOut()  << " aK=" << aVK << " N=" << aNb  <<  " C(k,n)=" <<  aVNbSub.back() << "\n";
  }
  mDecP = cDecomposPAdikVar(aVNbSub);
  StdOut()  << " NbModelTarget="   << NbCodeAvalaible() << "\n";

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

int cParamCodedTarget::BaseForNum() const
{
    int aNBC = NbCodeAvalaible();
    if (aNBC < 100)
	    return 10;  // decimal number on 2 dig
    if (aNBC < 256)
	    return 16;  // hexadecimal on 2 dig
    if (aNBC < 1296)
	    return 36;  // alpha num   on 2 dig
    MMVII_INTERNAL_ERROR("Too big number of code for current system");
    return -1;
}

bool cParamCodedTarget::CodeBinOfPts(double aRho,double aTeta,const cCodesOf1Target & aSetCodesOfT,double aRho0,double aThRho)
{
     int aIndRho = round_down((aRho-aRho0)/aThRho);  // indexe o
     aIndRho = std::max(0,std::min(mNbCircle-1,aIndRho));
     const cSetCodeOf1Circle & aSet1C =  mVecSetOfCode.at(aIndRho);
     int aN  = aSet1C.N();

     int aIndTeta = round_down((aTeta*aN*mNbRedond)/(2*M_PI));
     aIndTeta = aIndTeta % aN;
     const tBinCodeTarg & aCodeBin = aSetCodesOfT.CodeOfNumC(aIndRho);
     return aCodeBin.IsInside(aIndTeta);
}

tImTarget  cParamCodedTarget::MakeImCodeExt(const cCodesOf1Target & aSetCodesOfT)
{
     tImTarget aImT(mSzBin);
     tDataImT  & aDImT = aImT.DIm();

     double mRhoWhit1 = mRhoEndStar * ExtRatioW1;
     double mRhoCode  = mRhoEndStar * ExtRatioCode;

     for (const auto & aPix : aDImT)
     {
         cPt2dr  aPixN =  Pix2Norm(aPix);     // "Nomalized" coordinate
         cPt2dr  aRT  = ToPolar(aPixN,0.0);   // Polar then Rho teta
	 double  aRho = aRT.x();
         double  aTeta = aRT.y();
         if (aTeta < 0)
            aTeta += 2 *M_PI;

	 bool IsW = true;  // Default is white

         if (aRho < mRhoEndStar)  
         {
               
               int aIndTeta = round_down((aTeta/(M_PI/2.0)));
               IsW = (aIndTeta%2)==0;
         }
         else if (aRho<mRhoWhit1)
	 {
             IsW = true;
	     /*
             int aIndRho = round_down((aRho-mRhoWhit1)/(mRhoCode-mRhoWhit1));
             aIndRho = std::max(0,std::min(mNbCircle-1,aIndRho));
	     const cSetCodeOf1Circle & aSet1C =  mVecSetOfCode.at(aIndRho);
		    int aN  = aSet1C.N();

		    int aIndTeta = round_down((aTeta*aN*mNbRedond)/(2*M_PI));
		    aIndTeta = aIndTeta % aN;
                    const tBinCodeTarg & aCodeBin = aSetCodesOfT.CodeOfNumC(aIndRho);
                    if (aCodeBin.IsInside(aIndTeta))
                       IsW = false;
		*/
	 }
         else if (aRho<mRhoCode)
         {
             IsW = ! CodeBinOfPts(aRho,aTeta,aSetCodesOfT,mRhoWhit1,mRhoCode-mRhoWhit1);
         }
/*
         else
         {
              // Outside => border and fid marks (done after)
	      int aDInter = aRectHT.Interiority(aPix);
	      IsW = (aDInter<aDW) || (aDInter>=aDB);
         }
*/

         int aVal = IsW ? 255 : 0;
         aDImT.SetV(aPix,aVal);
     }

     ///compute string 
     int aNum = aSetCodesOfT.Num();
     int aBase = BaseForNum();
     for (int aK=0 ; aK<2 ;aK++)
     {
          int aDigit = (aK==0) ? (aNum%aBase) : (aNum/aBase);
	  char aCar = (aDigit < 10) ? ('0'+aDigit) : ('A'+(aDigit-10));
	  std::string aStr;
	  aStr.push_back(aCar);

          cIm2D<tU_INT1> aImStr = ImageOfString_DCT(aStr,1);
          cDataIm2D<tU_INT1>&  aDataImStr = aImStr.DIm();

	  double aPropFree =    ( (sqrt(2)-1)/(sqrt(2))) *  ExtRatioCode / ExtRatioBrd   // Prop free on diag inside coding
		              + (ExtRatioBrd-ExtRatioCode)  / ExtRatioBrd;               // Prop free due to the border

	  aPropFree *= (1-ExtTextMargin);

	  int  aNbTarget =  round_ni((mNbPixelBin/2)  * aPropFree);
	  
	  cPt2di aSzTarget (aNbTarget,aNbTarget);

	  cPt2di aOfs((aK==0)*(mSzBin.x()-aNbTarget),0);

	  cPt2dr aRatio = DivCByC(ToR(aDataImStr.Sz()),ToR(aSzTarget));

	  for (const auto & aPix : cRect2(cPt2di(0,0),aSzTarget))
          {
	      cPt2di aPixIm = aPix+aOfs;
              cPt2di aPixSym = mSzBin-aPixIm-cPt2di(1,1);

	      cPt2di aPixStr = ToI(MulCByC(ToR(aPix),aRatio));
	      int aVal = aDataImStr.DefGetV(aPixStr,0) ? 255 : 0;

              aDImT.SetV(aPixIm,aVal);
              aDImT.SetV(aPixSym,aVal);
          }
     }

     // MMVII_INTERNAL_ASSERT_User(aN<256,"For
     aImT = aImT.GaussDeZoom(3);
     return aImT;
}


tImTarget  cParamCodedTarget::MakeIm(const cCodesOf1Target & aSetCodesOfT)
{
     if (mCodeExt)
        return MakeImCodeExt(aSetCodesOfT);

     cRect2 aRectHT = cRect2::BoxWindow(ToI(mMidle),mNbPixelBin/2);

     tImTarget aImT(mSzBin);
     tDataImT  & aDImT = aImT.DIm();

     int aDW = (mRhoEndBrdWhiteExt-mRhoEndBrdBlack) * mScale;
     int aDB = (mRhoEndBrdWhiteExt-mRhoEnBrdWhiteInt ) * mScale;


     //  Parse all pixels of image
     for (const auto & aPix : aDImT)
     {
         cPt2dr  aPixN =  Pix2Norm(aPix);     // "Nomalized" coordinate
         cPt2dr  aRT  = ToPolar(aPixN,0.0);   // Polar then Rho teta
	 double  aRho = aRT.x();
         double  aTeta = aRT.y();

	 bool IsW = true;  // Default is white

         if (aRho < mRhoEndStar)  
         {
             // Generate the stars
	     if (aRho>=mRhoEndTargetC)
	     {
		if (aTeta < 0)
                   aTeta += 2 *M_PI;
                IsW = ! CodeBinOfPts(aRho,aTeta,aSetCodesOfT,mRhoEndTargetC,mThRing);
	     }
	     else
	     {
                  // Gennerate the intenal disks
                  // double  mScaleTopo
                  double aRatio = mRhoEndTargetC / std::max(1.0/mScale,aRho);  // 1/mScale => smallest pixel
		  double aRInd = log(aRatio) / log(mScaleTopo);
		  int aInd = round_down(aRInd+0.6666);
		  IsW = ((aInd%2) != 0);
	     }
         }
         else if (aRho<mRhoEndBlackCircle)
	 {
             IsW = false;
	 }
         else
         {
              // Outside => border and fid marks (done after)
	      int aDInter = aRectHT.Interiority(aPix);
	      IsW = (aDInter<aDW) || (aDInter>=aDB);
         }

         int aVal = IsW ? 255 : 0;
         aDImT.SetV(aPix,aVal);
     }

     // Print the string of number
     {
          std::string aStrCode = ToStr(aSetCodesOfT.Num(),2);
          cIm2D<tU_INT1> aImStr = ImageOfString_10x8(aStrCode,mCodeExt ? 1 : -1);
          cDataIm2D<tU_INT1> & aDImStr = aImStr.DIm();
          cPt2di aNbPixStr = aDImStr.Sz();
          // Ratio between pix of bin image and pix of string
          double  aScaleStr =  (mThTxt/(mCodeExt ? aNbPixStr.y() : aNbPixStr.x())) * mScale; 

         // StdOut() << "STR=[" << aStr <<  "] ScSt " << aScaleStr << "\n";

	 cPt2dr aSzStr = ToR(aNbPixStr) * aScaleStr;
	 // cPt2di aP0 = ToI(aMidStr-aSzStr/2.0);
	 // cPt2di aP0(aDB,aDB);
	 cPt2di aP0 = Pt_round_up(cPt2dr(mThBrdWhiteExt,mThBrdWhiteExt)*mScale);
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
