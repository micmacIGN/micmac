#include "CodedTarget.h"
#include "include/MMVII_2Include_Serial_Tpl.h"


namespace MMVII
{

static constexpr int SzGaussDeZoom =  3;  // De Zoom to have a gray value target

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

int cCodesOf1Target::getCodeLength(){
    return mCodes.size();
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
    MMVII::AddData(cAuxAr2007("SzF",anAux),mSzF);
    MMVII::AddData(cAuxAr2007("CenterF",anAux),mCenterF);
    MMVII::AddData(cAuxAr2007("CornEl1",anAux),mCornEl1);
    MMVII::AddData(cAuxAr2007("CornEl2",anAux),mCornEl2);
    MMVII::AddData(cAuxAr2007("SzCCB",anAux),mSz_CCB);
    MMVII::AddData(cAuxAr2007("ThickN_WhiteInt",anAux),mThickN_WInt);
    MMVII::AddData(cAuxAr2007("ThickN_Code",anAux),mThickN_Code);
    MMVII::AddData(cAuxAr2007("ThickN_WhiteExt",anAux),mThickN_WExt);
    MMVII::AddData(cAuxAr2007("ThickN_Car",anAux),mThickN_Car);
    MMVII::AddData(cAuxAr2007("ChessBoardAngle",anAux),mChessboardAng);


    /*
    std::vector<std::string> TARGET_NAMES;
    std::vector<std::vector<int>> TARGET_CODES;

    for (int aNum=0 ; aNum<NbCodeAvalaible(); aNum++){
        TARGET_NAMES.push_back(NameOfNum(aNum));
        cCodesOf1Target code = CodesOfNum(aNum);
        TARGET_CODES.push_back(code.CodeOfNumC(0).ToVect());
    }

    MMVII::AddData(cAuxAr2007("TargetNames",anAux), TARGET_NAMES);
    MMVII::AddData(cAuxAr2007("TargetCodes",anAux), TARGET_CODES);
    */

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
   mNbBit         (9),
   mWithParity    (true),
   mNbRedond      (2),
   mNbCircle      (1),
   mNbPixelBin    (round_to(1800,2*SzGaussDeZoom)), // make size a multiple of 2 * zoom, to have final center at 1/2 pix
   mSz_CCB        (1),
   mThickN_WInt   (0.35),
   mThickN_Code   (0.35),
   mThickN_WExt   (0.2),
   mThickN_Car    (0.8),
   mThickN_BExt   (0.05),
   mChessboardAng (M_PI/4.0),
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
  MMVII_INTERNAL_ASSERT_strong(((mNbPixelBin%2)==0),"Require odd pixel 4 binary image");
  mSzBin = cPt2di(mNbPixelBin,mNbPixelBin);

  double aCumulThick = 1.0;
  mRho_0_EndCCB = mSz_CCB     * aCumulThick;

  aCumulThick += mThickN_WInt;
  mRho_1_BeginCode =  mSz_CCB   * aCumulThick;

  aCumulThick += mThickN_Code;
  mRho_2_EndCode =  mSz_CCB   * aCumulThick;

  aCumulThick += mThickN_WExt;
  mRho_3_BeginCar =  mSz_CCB   * aCumulThick;

  mRho_4_EndCar = std::max
                  (
                        mRho_3_BeginCar,
                        mRho_3_BeginCar/sqrt(2) + mThickN_Car
                  );



  // mMidle = ToR(mSzBin-cPt2di(1,1)) / 2.0 - cPt2dr(1,1) ; //  pixel center model,suppose sz=2,  pixel 0 and 1 => center is 0.5
  mMidle = ToR(mSzBin-cPt2di(SzGaussDeZoom,SzGaussDeZoom)) / 2.0  ; //  pixel center model,suppose sz=2,  pixel 0 and 1 => center is 0.5
  mScale = mNbPixelBin / (2.0 * mRho_4_EndCar);

  std::vector<int> aVNbSub;
  for (int aKCirc = 0 ; aKCirc< mNbCircle ; aKCirc++)
  {
      int  aNb =  mNbBit ;
      int aStep = mWithParity ? 2 : 1;
      int aN0 = mWithParity ? 1 : 0;

      std::vector<int>  aVK;
      for (int aK=aN0 ; aK<=aNb ; aK+= aStep)
          aVK.push_back(aK);

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
    if (aNBC <= 256)
	    return 16;  // hexadecimal on 2 dig
    if (aNBC < 1296)
	    return 36;  // alpha num   on 2 dig
    MMVII_INTERNAL_ERROR("Too big number of code for current system");
    return -1;
}

std::string cParamCodedTarget::NameOfNum(int aNum) const
{
  int aBase = BaseForNum();
  std::string aRes;
  for (int aK=0 ; aK<2 ;aK++)
  {
      int aDigit = (aNum%aBase) ;
      aNum /= aBase;
      char aCar = (aDigit < 10) ? ('0'+aDigit) : ('A'+(aDigit-10));
      aRes.push_back(aCar);
  }
  std::reverse(aRes.begin(),aRes.end());

   return aRes;
}

std::string cParamCodedTarget::NameFileOfNum(int aNum) const
{
   return "Target_" + NameOfNum(aNum) + ".tif";
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

tImTarget  cParamCodedTarget::MakeImCircle(const cCodesOf1Target & aSetCodesOfT)
{
     tImTarget aImT(mSzBin);
     tDataImT  & aDImT = aImT.DIm();

     int aBrdBlack =  (mThickN_BExt/mRho_4_EndCar) * (mNbPixelBin/2);

     for (const auto & aPix : aDImT)
     {
         cPt2dr  aPixN =  Pix2Norm(aPix);     // "Nomalized" coordinate
         cPt2dr  aRT  = ToPolar(aPixN,0.0);   // Polar then Rho teta
	 double  aRho = aRT.x();
         double  aTeta = aRT.y();
         if (aTeta < 0)
            aTeta += 2 *M_PI;

	 bool IsW = true;  // Default is white


         if (aRho < mRho_0_EndCCB)  // if we are inside the square bord circle
         {
            double PIsur2 = M_PI/2.0;
            double OrigineTeta = this->mChessboardAng;     // Origine angle of chessboard pattern; // 0; // Pi/4

            int aIndTeta = round_down((aTeta+OrigineTeta)/PIsur2);
            IsW = (aIndTeta%2)==0;
         }
         else if (aRho<mRho_1_BeginCode)
	     {
             IsW = true;
         }
         else if (aRho<mRho_2_EndCode)
         {
             IsW = ! CodeBinOfPts(aRho,aTeta,aSetCodesOfT,mRho_1_BeginCode,mRho_2_EndCode-mRho_1_BeginCode);
         }
         else
         {
              // Outside => border and fid marks (done after)
	      int aDInter = aDImT.Interiority(aPix);
              if  (aDInter <aBrdBlack)
	          IsW = false;
         }

         int aVal = IsW ? 255 : 0;
         aDImT.SetV(aPix,aVal);
     }

     ///compute string
     int aNum = aSetCodesOfT.Num();
     std::string aName = NameOfNum(aNum);
     for (int aK=0 ; aK<2 ;aK++)
     {
	  std::string aStr;
	  aStr.push_back(aName[aK]);

           cIm2D<tU_INT1> aImStr =   (BaseForNum() <= 16)       ?
                                     ImageOfString_DCT(aStr,1)  :
                                     ImageOfString_10x8(aStr,1) ;
          cDataIm2D<tU_INT1>&  aDataImStr = aImStr.DIm();


	  int  aNbTarget =  round_ni((mNbPixelBin/2)  * (mThickN_Car /mRho_4_EndCar));

	  cPt2di aSzTarget (aNbTarget,aNbTarget);


	  cPt2dr aRatio = DivCByC(ToR(aDataImStr.Sz()),ToR(aSzTarget));
	  cPt2di aOfsGlob((aK!=0)*(mSzBin.x()-aNbTarget),0);
          cPt2di aOfPix((aK!=0)*-3,0);

	  for (const auto & aPix : cRect2(cPt2di(0,0),aSzTarget))
          {
	      cPt2di aPixIm = aPix+aOfsGlob;
              cPt2di aPixSym = mSzBin-aPixIm-cPt2di(1,1);

	      cPt2di aPixStr = ToI(MulCByC(ToR(aPix),aRatio));
	      int aVal = aDataImStr.DefGetV(aPixStr+aOfPix,0) ? 255 : 0;

              aDImT.SetV(aPixIm,aVal);
              aDImT.SetV(aPixSym,aVal);
          }
     }

     // MMVII_INTERNAL_ASSERT_User(aN<256,"For

     aImT = aImT.GaussDeZoom(SzGaussDeZoom);
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
          << AOpt2007(mPCT.mNbBit,"NbBit","Nb Bit printed",{eTA2007::HDV})
          << AOpt2007(mPCT.mWithParity,"WPar","With parity bit",{eTA2007::HDV})

          << AOpt2007(mPCT.mThickN_WInt,"ThW0","Thickness of interior white circle",{eTA2007::HDV})
          << AOpt2007(mPCT.mThickN_Code,"ThCod","Thickness of bin-coding black circle",{eTA2007::HDV})
          << AOpt2007(mPCT.mThickN_WExt,"ThSepCar","Thickness of sep bin-code / ahpha code",{eTA2007::HDV})
          << AOpt2007(mPCT.mThickN_Car,"ThCar","Thickness of separation alpha ccode ",{eTA2007::HDV})
          << AOpt2007(mPCT.mThickN_BExt,"ThBExt","Thickness of black border ",{eTA2007::HDV})
          << AOpt2007(mPCT.mChessboardAng,"Theta","Origin angle of chessboard pattern ",{eTA2007::HDV})


/*  For now dont confuse user with these values probably unused
          << AOpt2007(mPCT.NbRedond(), "Redund","Number of repetition inside a circle",{eTA2007::HDV})
          << AOpt2007(mPCT.NbCircle(), "NbC","Number of circles",{eTA2007::HDV})
*/
   ;
}


int  cAppliGenCodedTarget::Exe()
{
   mPCT.Finish();

   for (int aNum=0 ; aNum<mPCT.NbCodeAvalaible() ; aNum+=mPerGen)
   {
      cCodesOf1Target aCodes = mPCT.CodesOfNum(aNum);
      aCodes.Show();

	  tImTarget aImT= mPCT.MakeImCircle(aCodes);

      // std::string aName = "Target_" + mPCT.NameOfNum(aNum) + ".tif";
      // FakeUseIt(aCodes);
      mPCT.mSzF = aImT.DIm().Sz();
      mPCT.mCenterF = mPCT.mMidle / double(SzGaussDeZoom);

StdOut() << "mPCT.mCenterF  " << mPCT.mCenterF  << mPCT.mMidle << "\n";

      double aRhoChB = ((mPCT.mRho_0_EndCCB/mPCT.mRho_4_EndCar) * (mPCT.mNbPixelBin /2.0)  )/SzGaussDeZoom;
      mPCT.mCornEl1 = mPCT.mCenterF+FromPolar(aRhoChB,M_PI/4.0);
      mPCT.mCornEl2 = mPCT.mCenterF+FromPolar(aRhoChB,3.0*(M_PI/4.0));

      if (0)  // Marking point specific, do it only for tuning
      {
         for (const auto & aDec : cRect2::BoxWindow(3))
         {
              aImT.DIm().SetV(ToI(mPCT.mCornEl1)+aDec,128);
              aImT.DIm().SetV(ToI(mPCT.mCornEl2)+aDec,128);
         }
      }

      aImT.DIm().ToFile(mPCT.NameFileOfNum(aNum));
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


/*  ===   Old target, was coding on all the stars, maybe it will be reused later ...

*/
/*
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

     // Print th
e string of number
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

     aImT = aImT.GaussDeZoom(SzGaussDeZoom);
     return aImT;
}
*/

