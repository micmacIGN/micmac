#include <bitset>
#include "CodedTarget.h"
#include "include/MMVII_2Include_Serial_Tpl.h"


namespace MMVII
{


namespace  cNS_CodedTarget
{

/**************************************************/
/*                                                */
/*           cParamCodedTarget                    */
/*                                                */
/**************************************************/


std::string  cParamCodedTarget::NameOfBinCode(int aNum) const
{
    MMVII_INTERNAL_ASSERT_strong(false,"NameOfBinCode");

    return "-1111111";
}


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

int cParamCodedTarget::ToMultiple_2DeZoom(int aSz) const
{
    return round_to(aSz,2*mSzGaussDeZoom);
}

cPt2di cParamCodedTarget::ToMultiple_2DeZoom(const cPt2di &aSz) const
{
	return cPt2di(ToMultiple_2DeZoom(aSz.x()),ToMultiple_2DeZoom(aSz.y()));
}


cParamCodedTarget::cParamCodedTarget() :
   mType             (eTyCodeTarget::eIGNIndoor),  // eNbVals create problem in reading  eIGNIndoor
   mNbBit            (9),
   mWithParity       (true),
   mNbRedond         (2),
   mNbCircle         (1),
   mSzGaussDeZoom    (3),
   mNbPixelBin       (ToMultiple_2DeZoom(1800)), // make size a multiple of 2 * zoom, to have final center at 1/2 pix
   mSz_CCB           (1),
   mThickN_WInt      (0.35),
   mThickN_Code      (0.35),
   mThickN_WExt      (0.2),
   mThickN_Car       (0.8),
   mThickN_BExt      (0.05),
   mChessboardAng    (0.0),
   mWithChessboard   (true),
   mWhiteBackGround  (true),
   mZeroIsBackGround (true),
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
       anAppli.SetIfNotInit(mThickN_WInt,0.1);
       anAppli.SetIfNotInit(mThickN_Code,0.0);
       anAppli.SetIfNotInit(mThickN_WExt,0.0);
       anAppli.SetIfNotInit(mThickN_Car,0.3);
       anAppli.SetIfNotInit(mChessboardAng,M_PI/4.0);
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
   }
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

  if (mModeFlight){
    // mSzBin = cPt2di(0.95*0.707*mNbPixelBin,0.95*mNbPixelBin);

    mSzBin = ToMultiple_2DeZoom(cPt2di(mNbPixelBin,mNbPixelBin*sqrt(2)));

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




/* *************************************************** */
/*                                                     */
/*                cFullSpecifTarget                    */
/*                                                     */
/* *************************************************** */

cFullSpecifTarget::cFullSpecifTarget(const cBitEncoding& aBE,const cParamRenderingTarget& aRender) :
   mBE      (aBE),
   mRender  (aRender),
   mCEC     (nullptr)
{
}

cFullSpecifTarget::~cFullSpecifTarget()
{
    delete mCEC;
}

const std::vector<cOneEncoding> &  cFullSpecifTarget::Encodings() const {return mBE.Encodings();}
const  cSpecBitEncoding &          cFullSpecifTarget::Specs()     const {return mBE.Specs();}
const  cParamRenderingTarget &     cFullSpecifTarget::Render()    const {return mRender;}
const  std::string &               cFullSpecifTarget::PostFix()   const {return Specs().mPostFix;}


std::string cFullSpecifTarget::NameOfImPattern() const
{
   return  PostFix() + "_Pattern"+ ".tif"; 
}

std::string cFullSpecifTarget::NameOfEncode(const cOneEncoding & anEncode) const
{
    return   PostFix()  +  "_Target_"+  anEncode.Name() +   ".tif";
}




/* *************************************************** */
/*                                                     */
/*             cNormPix2Bit                            */
/*                                                     */
/* *************************************************** */


class cNormPix2Bit
{
    public :
	 virtual bool    PNormIsCoding(const cPt2dr & aPt)   const = 0;
	 virtual int     BitsOfNorm    (const cPt2dr & aPt)  const = 0;

         static cNormPix2Bit * Alloc(const cFullSpecifTarget & aSpecif);

	 virtual ~cNormPix2Bit() {}
    protected :
};

/* *************************************************** */
/*                                                     */
/*             cCircNP2B                               */
/*                                                     */
/* *************************************************** */


class cCircNP2B : public  cNormPix2Bit
{
     public :
         cCircNP2B(const cFullSpecifTarget & aSpecif);
	 bool    PNormIsCoding(const cPt2dr & aPt)      const   override;
	 int     BitsOfNorm    (const cPt2dr & aPt)     const   override;

     private :
	 cPt2dr  PreProcessCoord(const cPt2dr & aPt)    const;

	 tREAL8 mRho0;
	 tREAL8 mRho1;
	 tREAL8 mTeta0;
	 int    mNbBits;
};



cCircNP2B::cCircNP2B(const cFullSpecifTarget & aSpecif) :
   mRho0   (aSpecif.Render().mRho_1_BeginCode),
   mRho1   (aSpecif.Render().mRho_2_EndCode),
   mTeta0  (aSpecif.Render().mChessboardAng),
   mNbBits (aSpecif.Specs().mNbBits)
{
}

cPt2dr  cCircNP2B::PreProcessCoord(const cPt2dr & aPt)   const 
{
    return ToPolar(aPt);
}
bool  cCircNP2B::PNormIsCoding(const cPt2dr & aPt) const 
{
    tREAL8 aRho = PreProcessCoord(aPt).x();
    // StdOut() << " RRR " << aRho  << " "<< mRho0 << " " << mRho1 <<"\n";
    return   (aRho>=mRho0)  && (aRho<mRho1) ;
}

int cCircNP2B::BitsOfNorm(const cPt2dr & aPt) const 
{
     tREAL8 aTeta = PreProcessCoord(aPt).y() +mTeta0;
     tREAL8 aIndex = mNbBits * (aTeta / (2*M_PI)) ;
     aIndex = mod_real(aIndex,mNbBits);
     return round_down (aIndex);
}

/* *************************************************** */
/*                                                     */
/*             cStraightNP2B                           */
/*                                                     */
/* *************************************************** */

class cStraightNP2B : public  cNormPix2Bit
{
     public :
         cStraightNP2B(const cFullSpecifTarget & aSpecif);
	 bool    PNormIsCoding(const cPt2dr & aPt)      const   override;
	 int     BitsOfNorm    (const cPt2dr & aPt)     const   override;

     private :
	 tREAL8   mRho1;
         int      mNbBits;
         int      mNbBS2;
};

cStraightNP2B::cStraightNP2B(const cFullSpecifTarget & aSpecif) :
   mRho1   (aSpecif.Render().mRho_2_EndCode),
   mNbBits (aSpecif.Specs().mNbBits),
   mNbBS2  (mNbBits /2)
{
    MMVII_INTERNAL_ASSERT_tiny((mNbBS2*2)==mNbBits,"Odd nbbits in cStraightNP2B");
}

bool    cStraightNP2B::PNormIsCoding(const cPt2dr & aPt) const   
{
    return std::abs(aPt.y()) > mRho1;
}

int   cStraightNP2B::BitsOfNorm(const cPt2dr & aPt) const
{
    return    round_down((aPt.x()+mRho1)/(2*mRho1) *mNbBS2)  
	    + mNbBS2*(aPt.y()<0);
}

/* *************************************************** */
/*                                                     */
/*             cNormPix2Bit                            */
/*                                                     */
/* *************************************************** */

cNormPix2Bit * cNormPix2Bit::Alloc(const cFullSpecifTarget & aSpecif)
{
   switch (aSpecif.Specs().mType)
   {
         case eTyCodeTarget::eCERN :
         case eTyCodeTarget::eIGNIndoor:
	       return new cCircNP2B(aSpecif);
         case eTyCodeTarget::eIGNDrone:
	       return new cStraightNP2B(aSpecif);

         case eTyCodeTarget::eNbVals:
              return nullptr;
   }

   return nullptr;
}

/* *************************************************** */
/*                                                     */
/*                cCodedTargetPatternIm                */
/*                                                     */
/* *************************************************** */

enum class eLPT  // Label Pattern Target
           {
              eBackGround,
              eForeGround,
              eChar,
              eNumB0   // num first bit
           };

class cCodedTargetPatternIm 
{
     public :
          typedef tU_INT1            tElem;
          typedef cIm2D<tElem>       tIm;
          typedef cDataIm2D<tElem>   tDataIm;

          cCodedTargetPatternIm(const cFullSpecifTarget &);

	  tIm  ImCoding() const;

	  tIm MakeOneImTarget(const cOneEncoding & aCode);
     private :
	  const cFullSpecifTarget & mSpec;
	  tIm               mImCoding;
	  tDataIm &         mDIC;

	  tIm               mImTarget;
	  tDataIm &         mDIT;

	  tREAL8            mTeta0;
	  tREAL8            mRhoC;
	  tREAL8            mRho2C;
};

cCodedTargetPatternIm::cCodedTargetPatternIm(const cFullSpecifTarget & aSpec) :
     mSpec       (aSpec),
     mImCoding   (mSpec.Render().mSzBin),
     mDIC        (mImCoding.DIm()),
     mImTarget   (mSpec.Render().mSzBin),
     mDIT        (mImTarget.DIm()),
     mTeta0      (mSpec.Render().mChessboardAng),
     mRhoC       (mSpec.Render().mRho_0_EndCCB),
     mRho2C      (Square(mRhoC))
{
    mDIC.InitCste(tElem(eLPT::eBackGround));

    // std::vector<cPt2dr> & aVBC = mSpec.Render().mBitsCenters;


    // std::unique_ptr<cNormPix2Bit>  aP2B (new cCircNP2B (mSpec));
    std::unique_ptr<cNormPix2Bit>  aP2B (cNormPix2Bit::Alloc(aSpec));
    for (const auto & aPix : mDIC)
    {
       cPt2dr aPN = mSpec.Render().Pix2Norm(aPix);
       //  ============  1  Generate the bit coding =======================
       if (aP2B->PNormIsCoding(aPN))
       {
           int aNumB =  aP2B->BitsOfNorm(aPN);
	   mDIC.SetV(aPix,int(eLPT::eNumB0)+aNumB);
       }
       //  ============  2  Generate the central circle =======================
       else if (SqN2(aPN) <mRho2C)
       {
           eLPT aLab = eLPT::eForeGround;
           if (mSpec.Render().mWithChessboard)
	   {
               double PIsur2 = M_PI/2.0;
	       tREAL8 aTeta = ToPolar(aPN).y();
               int aIndTeta = round_down((aTeta+mTeta0)/PIsur2);
               if ((aIndTeta%2)==0)
                   aLab = eLPT::eBackGround;
	   }
	   mDIC.SetV(aPix,int(aLab));
       }
    }
}

cCodedTargetPatternIm::tIm cCodedTargetPatternIm::ImCoding() const {return mImCoding;}

cCodedTargetPatternIm::tIm cCodedTargetPatternIm::MakeOneImTarget(const cOneEncoding & anEnCode)
{
   int aBG_Coul = mSpec.Render().mWhiteBackGround ? 255 : 0;
   int aFG_Coul =  255-aBG_Coul;

   mDIT.InitCste(aBG_Coul);
   size_t aCode = anEnCode.Code();
   bool  BGIs_0 = mSpec.Render().mZeroIsBackGround;
   for (const auto & aPix : mDIC)
   {
       eLPT aLab =  eLPT(mDIC.GetV(aPix));
       if (aLab!=eLPT::eBackGround)
       {
           bool isBG = true;
           if (aLab==eLPT::eForeGround)
           {
               isBG = false;
           }
           else if (aLab>=eLPT::eNumB0)
           {
                bool BitIs_1 =  (aCode & (1<<(int(aLab)-int(eLPT::eNumB0)))) != 0;
                isBG = BitIs_1 !=  BGIs_0;
           }

	   if (!isBG)
	   {
               mDIT.SetV(aPix,aFG_Coul);
	   }
       }
   }

   return mImTarget.GaussDeZoom(mSpec.Render().mSzGaussDeZoom);
}

/* *************************************************** */
/*                                                     */
/*                cAppliGenCodedTarget                 */
/*                                                     */
/* *************************************************** */

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
   ;
}


int  cAppliGenCodedTarget::Exe()
{
   ReadFromFile(mBE,mNameBE);

   mPCT.FinishInitOfType(Type());
   mPCT.Finish();

   cFullSpecifTarget  aFullSpec(mBE,mPCT);
   cCodedTargetPatternIm  aCTPI(aFullSpec);

   aCTPI.ImCoding().DIm().ToFile(aFullSpec.NameOfImPattern());

   for (const auto & anEncode : aFullSpec.Encodings())
   {
       cCodedTargetPatternIm::tIm anIm = aCTPI.MakeOneImTarget(anEncode);

       std::string aName = aFullSpec.NameOfEncode(anEncode);
       anIm.DIm().ToFile(aName);
       StdOut() << aName << "\n";
   }


   return EXIT_SUCCESS;
}
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


