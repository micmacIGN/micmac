#include <bitset>
#include "CodedTarget.h"
#include "include/MMVII_2Include_Serial_Tpl.h"


namespace MMVII
{


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

int cCodesOf1Target::getCodeLength() const{
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

/*
template <class eType> void EnumAddData(const cAuxAr2007 & anAux,const std::string & aTag,eType & aType)
{
    std::string aName= E2Str(aType);
    MMVII::AddData(cAuxAr2007(aTag,anAux),aName);
    if (anAux.Input())
       aType = Str2E<eProjPC>(aName);
}
*/


void cParamCodedTarget::AddData(const cAuxAr2007 & anAux)
{
    MMVII::EnumAddData(anAux,mType,"Type");
    MMVII::AddData(cAuxAr2007("NbBits",anAux),mNbBit);
    MMVII::AddData(cAuxAr2007("SzF",anAux),mSzF);
    MMVII::AddData(cAuxAr2007("CenterF",anAux),mCenterF);
    MMVII::AddData(cAuxAr2007("CornEl1",anAux),mCornEl1);
    MMVII::AddData(cAuxAr2007("CornEl2",anAux),mCornEl2);
    MMVII::AddData(cAuxAr2007("SzCCB",anAux),mSz_CCB);
    MMVII::AddData(cAuxAr2007("ThickN_WhiteInt",anAux),mThickN_WInt);
    MMVII::AddData(cAuxAr2007("ThickN_Code",anAux),mThickN_Code);
    MMVII::AddData(cAuxAr2007("ThickN_WhiteExt",anAux),mThickN_WExt);
    MMVII::AddData(cAuxAr2007("ThickN_BlackExt",anAux),mThickN_BExt);
    MMVII::AddData(cAuxAr2007("ThickN_Car",anAux),mThickN_Car);
    MMVII::AddData(cAuxAr2007("ChessBoardAngle",anAux),mChessboardAng);
    MMVII::AddData(cAuxAr2007("ModeFlight",anAux),mModeFlight);
    MMVII::AddData(cAuxAr2007("WithChessBoard",anAux),mWithChessboard);
    MMVII::AddData(cAuxAr2007("WhiteBackGround",anAux),mWhiteBackGround);


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
   mType             (eTyCodeTarget::eIGNIndoor),  // eNbVals create problem in reading  eIGNIndoor
   mNbBit            (9),
   mWithParity       (true),
   mNbRedond         (2),
   mNbCircle         (1),
   mSzGaussDeZoom    (3),
   mNbPixelBin       (round_to(1800,2*mSzGaussDeZoom)), // make size a multiple of 2 * zoom, to have final center at 1/2 pix
   mSz_CCB           (1),
   mThickN_WInt      (0.35),
   mThickN_Code      (0.35),
   mThickN_WExt      (0.2),
   mThickN_Car       (0.8),
   mThickN_BExt      (0.05),
   mChessboardAng    (M_PI/4.0),
   mWithChessboard   (true),
   mWhiteBackGround  (true),
   mModeFlight       (false),  // MPD => def value was not initialized ?
   mDecP             ({1,1})  // "Fake" init 4 now
{
}

void cParamCodedTarget::FinishInitOfType(eTyCodeTarget aType)
{
   mType = aType;
   cMMVII_Appli & anAppli = cMMVII_Appli::CurrentAppli();

   if (aType==eTyCodeTarget::eIGNIndoor)
   {
         // Nothingto do all default value have been setled for this case
   }
   else if (aType==eTyCodeTarget::eIGNDrone)
   {
	   anAppli.SetIfNotInit(mModeFlight,true);
   }
   else if (aType==eTyCodeTarget::eCERN)
   {
       anAppli.SetIfNotInit(mNbBit,20);
       anAppli.SetIfNotInit(mWithParity,false);
       anAppli.SetIfNotInit(mNbRedond,1);
       anAppli.SetIfNotInit(mThickN_WInt,1.0);
       anAppli.SetIfNotInit(mThickN_Code,1.0);
       anAppli.SetIfNotInit(mThickN_WExt,1.0);
       anAppli.SetIfNotInit(mThickN_BExt,0.0);

       anAppli.SetIfNotInit(mWithChessboard,false);
       anAppli.SetIfNotInit(mWhiteBackGround,false);

   // mThickN_WExt      (0.2),
   // mThickN_Car       (0.8),
   // mThickN_BExt      (0.05),
   }
}

//	SetIfNotInit

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

  if (mModeFlight){
    mSzBin = cPt2di(0.95*0.707*mNbPixelBin,0.95*mNbPixelBin);
    mThickN_WInt = 0.1;
    mThickN_Code = 0;
  } else{
    mSzBin = cPt2di(mNbPixelBin,mNbPixelBin);
  }

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
  mMidle = ToR(mSzBin-cPt2di(mSzGaussDeZoom,mSzGaussDeZoom)) / 2.0  ; //  pixel center model,suppose sz=2,  pixel 0 and 1 => center is 0.5
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


  cHamingCoder aHCTest(mNbBit-1*mWithParity);




  StdOut() << "-------------------------------------------------------------------\n";
  StdOut() << "Number of targets: "   << NbCodeAvalaible() << "\n";
  if (mModeFlight){
    StdOut() << "Code pattern: " << ceil(((double)aHCTest.NbBitsOut())/2.0) << " x 2 " << "\n";
  }
  StdOut() << "-------------------------------------------------------------------\n";

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

    {
        MMVII_DEV_WARNING("Too big number of code for current system");
        return 36;
    }
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

tImTarget  cParamCodedTarget::MakeImCircle(const cCodesOf1Target & aSetCodesOfT, bool modeFlight)
{
     tImTarget aImT(mSzBin);
     tDataImT  & aDImT = aImT.DIm();

     int aBrdBlack =  (mThickN_BExt/mRho_4_EndCar) * (mNbPixelBin/2);

     for (const auto & aPix : aDImT){
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
	    if (!mWithChessboard)
               IsW = false;
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
	          IsW = false || mModeFlight;
         }

         if (!mWhiteBackGround) 
            IsW = ! IsW;
         int aVal = IsW ? 255 : 0;
         cPt2di aNewPx = cPt2di(aPix.x(), aPix.y()-250);

         if (mModeFlight){
             if ((aNewPx.y() > 0)){
                aDImT.SetV(aNewPx, aVal);
             }

             if ((aDImT.Sz().y()-aPix.y() < 250)){
                 aDImT.SetV(aPix, 255);
             }
         }else{
            aDImT.SetV(aPix, aVal);
         }

     }



    // -------------------------------------------------------------
    // Orientation mark for flight mode
    // -------------------------------------------------------------
    if (mModeFlight){
        int sz = 90;

        double cx = aDImT.Sz().x()/2.0;
        double cy = aDImT.Sz().y()/2-250;

        double dx = 3.5/10.0*aDImT.Sz().x();
        double dy = 0;

        double theta = mChessboardAng-0.785398;
        double px = cx + dx*cos(theta) + dy*sin(theta);
        double py = cy - dx*sin(theta) + dy*cos(theta);

        for (int i=-sz; i<=sz; i++){
            for (int j=-sz; j<=sz; j++){
                if (abs(i)+abs(j) > sz) continue;
                aDImT.SetV(cPt2di(px+i, py+j), 0);
            }
        }
    }

    // -------------------------------------------------------------
    // Hamming code for flight mode
    // -------------------------------------------------------------
    if (mModeFlight){
        cHamingCoder aHC(mNbBit-1);
        tU_INT4 hammingCode = aHC.Coding(aSetCodesOfT.Num());

        // 21 bits for maximal code size of 16
        std::bitset<21> hammingBinaryCode = std::bitset<21>(hammingCode);
        StdOut() << "Hamming code: ";

        int NbCols = ceil(((double)aHC.NbBitsOut())/2.0);

        int sq_vt = 180;
        int sq_sz = 900/NbCols;
        int idl, idc;

        for (int k=0; k<aHC.NbBitsOut(); k++){
            idc = k % NbCols;
            idl = (k>=NbCols)*1;
            //StdOut() << "idc = " << idc << " idl = " << idl << " bit = ";
            for (int px=150 + idc*sq_sz; px<150 + (idc+1)*sq_sz; px++){
                for (int py=1250 + idl*sq_vt; py<1250 + sq_vt + idl*sq_vt; py++){
                    aDImT.SetV(cPt2di(px, py), 255*(1-hammingBinaryCode[k]));
                }
            }
            StdOut() << hammingBinaryCode[k];
        }

        StdOut() << "   "  <<  aSetCodesOfT.Num() << "   " ;
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

	  if (modeFlight){
         aNbTarget /= 3;
	  }


	//  aNbTarget /= 3;
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
	    if (modeFlight) aVal = 255 - aVal;
              aDImT.SetV(aPixIm,aVal);
              aDImT.SetV(aPixSym,aVal);
          }
     }





     // MMVII_INTERNAL_ASSERT_User(aN<256,"For

     aImT = aImT.GaussDeZoom(mSzGaussDeZoom);
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
	eTyCodeTarget      Type();

     private :


        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;


	int                mPerGen;  // Pattern of numbers
	std::string        mNameBE;
	cBitEncoding       mBE;

	cParamCodedTarget  mPCT;
};

eTyCodeTarget cAppliGenCodedTarget::Type() {return mBE.Specs().mType ;}


/* *************************************************** */
/*                                                     */
/*              cAppliGenCodedTarget                   */
/*                                                     */
/* *************************************************** */

typedef cParamCodedTarget cParamGeomTarget;

class cFullSpecifTarget
{
      public :
         cFullSpecifTarget(const cBitEncoding&,const cParamGeomTarget&);
         const cBitEncoding     & mBE;
         const cParamGeomTarget & mGeom;
};

class cNormPix2Bit
{
    public :

	 virtual cPt2dr  Transfo(const cPt2dr & aPt) const = 0;
	 virtual bool PNormIsCodin(const cPt2dr & aPt) const = 0;
	 virtual int  BitsOfNorm    (const cPt2dr & aPt) const = 0;
    protected :
};


	/*
class cCircNP2B :  cNormPix2Bit
{
     public :
         cCircNP2B(const cFullSpecifTarget & aSpecif);

	 cPt2dr  Transfo(const cPt2dr & aPt)        const   override;
	 bool    PNormIsCodin(const cPt2dr & aPt)   const   override;
	 int     BitsOfNorm    (const cPt2dr & aPt) const   override;
     private :
	 tREAL8 mRho0;
	 tREAL8 mRho1;
	 tREAL8 mTeta0;
	 int    mNbBits;
};

cCircNP2B::cCircNP2B(const cFullSpecifTarget & aSpecif) :
   mRho0  (aSpecif.mGeom.mRho_1_BeginCode),
   mRho1  (aSpecif.mGeom.mRho_2_EndCode)
{
}

cPt2dr  cCircNP2B::Transfo(const cPt2dr & aPt)   const 
{
    return ToPolar(aPt);
}
bool  cCircNP2B::PNormIsCodin(const cPt2dr & aPt) const 
{
    tREAL8 aRho = aPt.x();
    return   (aRho>=mRho0)  && (aRho<mRho1) ;
}

int cCircNP2B::BitsOfNorm(const cPt2dr & aPt) const 
{
     tREAL8mSpecif.mGeom.mRho_1_BeginCode aTeta = aPt.y() -mChessboardAng;
     tREAL8 aIndex = (aTeta / (2*M_PI)) 
     aIndex = mod_real
     aTeta = (aTeta -mChessboardAng) / (2*M_PI)
     return round_ni 
}
*/	

class cCodedTargetPatternIm
{
     public :
          typedef tU_INT1            tElem;
          typedef cIm2D<tElem>       tIm;
          typedef cDataIm2D<tElem>   tDataIm;

          cCodedTargetPatternIm(const cBitEncoding & aBE,const cParamGeomTarget &);
     private :
	  cBitEncoding      mBE;
          cParamGeomTarget  mPGeom;
	  tIm               mIm;
};


cCodedTargetPatternIm::cCodedTargetPatternIm(const cBitEncoding & aBE,const cParamGeomTarget & aParamGeom) :
     mBE     (aBE),
     mPGeom  (aParamGeom),
     mIm     (mPGeom.mSzBin)
{
}




cAppliGenCodedTarget::cAppliGenCodedTarget(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli  (aVArgs,aSpec),
   mPerGen       (10)
{
}

cCollecSpecArg2007 & cAppliGenCodedTarget::ArgObl(cCollecSpecArg2007 & anArgObl)
{
 return
      anArgObl
          <<   Arg2007(mNameBE,"XML name for bit encoding struct")
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
          << AOpt2007(mPCT.mModeFlight,"ModeFlight","Special mode for Patricio ",{eTA2007::HDV})


/*  For now dont confuse user with these values probably unused
          << AOpt2007(mPCT.NbRedond(), "Redund","Number of repetition inside a circle",{eTA2007::HDV})
          << AOpt2007(mPCT.NbCircle(), "NbC","Number of circles",{eTA2007::HDV})
*/
   ;
}


int  cAppliGenCodedTarget::Exe()
{
   ReadFromFile(mBE,mNameBE);

   mPCT.FinishInitOfType(Type());
   mPCT.Finish();

   for (int aNum=0 ; aNum<mPCT.NbCodeAvalaible() ; aNum+=mPerGen)
   {

      cCodesOf1Target aCodes = mPCT.CodesOfNum(aNum);

      StdOut() << "[" << mPCT.NameOfNum(aNum) << "]  ";

	  tImTarget aImT= mPCT.MakeImCircle(aCodes, mPCT.mModeFlight);

      // std::string aName = "Target_" + mPCT.NameOfNum(aNum) + ".tif";
      // FakeUseIt(aCodes);
      mPCT.mSzF = aImT.DIm().Sz();
      mPCT.mCenterF = mPCT.mMidle / double(mPCT.mSzGaussDeZoom);

      //StdOut() << "mPCT.mCenterF  " << mPCT.mCenterF  << mPCT.mMidle << "\n";

      double aRhoChB = ((mPCT.mRho_0_EndCCB/mPCT.mRho_4_EndCar) * (mPCT.mNbPixelBin /2.0)  )/mPCT.mSzGaussDeZoom;
      mPCT.mCornEl1 = mPCT.mCenterF+FromPolar(aRhoChB,M_PI/4.0);
      mPCT.mCornEl2 = mPCT.mCenterF+FromPolar(aRhoChB,3.0*(M_PI/4.0));
       
#if 0  // Marking point specific, do it only for tuning
      {
         for (const auto & aDec : cRect2::BoxWindow(3))
         {
              aImT.DIm().SetV(ToI(mPCT.mCornEl1)+aDec,128);
              aImT.DIm().SetV(ToI(mPCT.mCornEl2)+aDec,128);
         }
      }
#endif
       
      StdOut() << mPCT.NameFileOfNum(aNum) << " created\n";
      aImT.DIm().ToFile(mPCT.NameFileOfNum(aNum));
   }

   StdOut() << "-------------------------------------------------------------------\n";
   SaveInFile(mPCT,"Target_Spec.xml");
   StdOut() << "File " << "Target_Spec.xml" << " created\n";
   StdOut() << "-------------------------------------------------------------------\n";


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

