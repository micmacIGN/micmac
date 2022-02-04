#include "include/MMVII_all.h"
#include "include/MMVII_SetITpl.h"
#include "include/MMVII_2Include_Serial_Tpl.h"


namespace MMVII
{

namespace  cNS_CodedTarget
{

typedef cSetISingleFixed<tU_INT4>  tBinCodeTarg;
typedef std::vector<tBinCodeTarg> tVSetICT;
typedef cIm2D<tU_INT1>     tImTarget;
typedef cDataIm2D<tU_INT1> tDataImT;


/*  *********************************************************** */
/*                                                              */
/*                      cParamCodedTarget                       */
/*                                                              */
/*  *********************************************************** */

class cSetCodeOf1Circle
{
    public :
      cSetCodeOf1Circle(double aRho0,int aK,int aN);
      int  NbSub() const;
      const tBinCodeTarg & CodeOfNum(int aNum) const;
      int N() const;
      int K() const;
    private :
      double   mRho0;
      int      mK;
      int      mN;
      tVSetICT mVSet ;
};


class cCodesOf1Target
{
   public :
      cCodesOf1Target(int aNum);

      void AddOneCode(const tBinCodeTarg &);
      void  Show();
      const tBinCodeTarg & CodeOfNumC(int) const;
   private :
      int                        mNum;
      std::vector<tBinCodeTarg>  mCodes;
};




class cParamCodedTarget
{
    public :
       cParamCodedTarget();
       int &     NbRedond();  // Redundancy = number of repetition of a pattern in a circle
       int &     NbCircle();  // Redundancy = number of repetition of a pattern in a circle
       double &  RatioBar();  // Ratio on codin bar
       void      Finish();

       int NbCodeAvalaible() const;         // Number of different code we can generate
       cCodesOf1Target CodesOfNum(int);     // One combinaison of binary code
       tImTarget  MakeIm(const cCodesOf1Target &);  // Generate the image of 1 combinaison

       void AddData(const cAuxAr2007 & anAux);


    private :

       cPt2dr    Pix2Norm(const cPt2di &) const;
       cPt2dr    Norm2PixR(const cPt2dr &) const;
       cPt2di    Norm2PixI(const cPt2dr &) const;

       int       mNbRedond;  // Redundancy = number of repetition of a pattern in a circle
       double    mRatioBar;  // Ratio H/l on coding rect, def 1, 2-> mean more bar
       double    mRhoWhite0;  // Central circle, used to compute affinity
       double    mRhoBlack0;  // Black circle, used for detection
       int       mNbCircle;  // Number of circles encoding information
       double    mDistMarkFid;    // Dist between Fid mark &  codage
       double    mBorderMarkFid;  // Dist between Bord & FidMark
       double    mRadiusFidMark;  // Radius of Fid Mark
       double    mTetaCenterFid;   // Teta init 
       int       mNbPaqFid;        // Number of group in "Fid Mark" By defaut==mNbRedond

       int       mNbFidByPaq;        // Number of Fiducial by quarter
       double    mGapFid;            // Size of gab in fiducial repeat
       double    mScaleTopo;         // Scale used to create identifiable center 4 toto
       int       mNbPixelBin;        // Number of pixel  Binary image


       std::vector<double> mTetasQ;  // Tetas of first quarter

       double    mRhoCodage0;   // Rho when begin binarie code
       double    mRhoCodage1;   // Rho when ends binarie code
       double    mRhoFidMark;   // Rho where are located Fid Mark
       double    mRhoEnd ;      // Rho where are finish the target


       double mRho_00_TopoB   ;  // Circle for topo ident
       double mRho_000_TopoW  ;  // Circle for topo ident
       double mRho_0000_TopoB ;  // Circle for topo ident

       cPt2di    mSzBin;
       cPt2dr    mMidle;
       double    mScale;  // Sz of Pixel in normal coord

       std::vector<cSetCodeOf1Circle>     mVecSetOfCode;
       cDecomposPAdikVar                  mDecP;
};

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
    MMVII::AddData(cAuxAr2007("DistFM",anAux),mDistMarkFid);
    MMVII::AddData(cAuxAr2007("DistBorderFM",anAux),mBorderMarkFid);
    MMVII::AddData(cAuxAr2007("RadiusFM",anAux),mRadiusFidMark);
}

void AddData(const  cAuxAr2007 & anAux,cParamCodedTarget & aPCT)
{
   aPCT.AddData(anAux);
}

cParamCodedTarget::cParamCodedTarget() :
   mNbRedond      (2),
   mRatioBar      (0.7),
   mRhoWhite0     (1.5),
   mRhoBlack0     (2.5),
   mNbCircle      (1),
   mDistMarkFid   (2.5),
   mBorderMarkFid (1.5),
   mRadiusFidMark (0.4),
   mTetaCenterFid (M_PI/4.0),
   mNbPaqFid      (-1),  // Marqer of No Init
   mNbFidByPaq    (5),
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

  mRhoCodage1  = mRhoCodage0 + mNbCircle;
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
      double aRho0 = mRhoCodage0 + aKCirc;
      double aRho1 = aRho0 +1;
      double aRhoM = (aRho0 + aRho1) / 2.0;
      int aNb = round_up((mRatioBar* 2*M_PI*aRhoM)/mNbRedond);
      double aProp = (mNbCircle-aKCirc) /  mRhoCodage1;
      int aK = round_down(aProp*aNb);
      if (mNbCircle==1)
         aK = aNb/2;
      aK =std::max(1,std::min(aNb-1,aK));

      mVecSetOfCode.push_back(cSetCodeOf1Circle(aRho0,aK,aNb));
      aVNbSub.push_back( mVecSetOfCode.back().NbSub());
      StdOut()  << " P="   << aProp << " aK=" << aK << " N=" << aNb  << " R=" << aRhoM << " C(k,n)=" <<  aVNbSub.back() << "\n";
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
                int aIndRho = std::max(0,std::min(mNbCircle-1,round_down(aRho-mRhoCodage0)));
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


	std::string        mPatNum;  // Pattern of numbers
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
          <<   Arg2007(mPatNum,"Pattern of numbers to generate")
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

   for (int aNum=0 ; aNum<mPCT.NbCodeAvalaible() ; aNum+=11)
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
