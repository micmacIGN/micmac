#include <bitset>
#include "CodedTarget.h"
#include "include/MMVII_2Include_Serial_Tpl.h"

//  49 : 387 =  256 + 128 + 2 +1
//              00000110000011
//
//  50 : 389 = 256 + 128 + 4 +1
//              00000110000101

namespace MMVII
{
using namespace cNS_CodedTarget;

/**  Generate a visualistion of target, made an external-non-friend  function
 * to test the usability.
 *
 */
void TestReloadAndShow_cFullSpecifTarget(const std::string & aName,int aZoom)
{
    // -1- ----------- Create the object from file ------------------
    std::unique_ptr<cFullSpecifTarget> aFullSpec (cFullSpecifTarget::CreateFromFile(aName));

    // -2- ------------ generate a "standard" target image -----------
        // -2.1- generate an encoding with maximum of transition
    cOneEncoding aEnc(0,Str2BitFlag("010101010101010101010101010101"));
    aEnc.SetName("xx");

       // -2.2- generate a "standard" target image
    cFullSpecifTarget::tIm anIm = aFullSpec->OneImTarget(aEnc);
    anIm.DIm().ToFile("TestTarget_"+aFullSpec->Prefix()+".tif");

    // -3-  ------------------generate a high resolution to visualize
    cRGBImage aImZoom =   RGBImFromGray(anIm.DIm(),1.0,aZoom);

         // -3.1- visualize centers and corners
    aImZoom.DrawCircle(cRGBImage::Blue,aFullSpec->Center(),1.0);
    aImZoom.DrawCircle(cRGBImage::Red,aFullSpec->CornerlEl_BW(),1.0);
    aImZoom.DrawCircle(cRGBImage::Green,aFullSpec->CornerlEl_WB(),1.0);

         // -3.2- visualize the bits
    for (const auto & aC : aFullSpec->BitsCenters())
    {
        for (const auto & aR : {1.0,3.0,5.0})
            aImZoom.DrawCircle(cRGBImage::Cyan,aC,aR);
    }

         // -3.3- write the file
    aImZoom.ToFile("TestZoom_"+aFullSpec->Prefix()+".tif");
}

void Bench_Target_Encoding()
{
    // -1-  Create the spec
    std::string aName =  cMMVII_Appli::InputDirTestMMVII()
	                 + "Targets" +  StringDirSeparator()  + "IGNIndoor_Nbb12_Freq2_Hamm3_Run2_3_FullSpecif.xml";

    std::unique_ptr<cFullSpecifTarget> aFullSpec (cFullSpecifTarget::CreateFromFile(aName));

    // -2-   check some elementary values
    MMVII_INTERNAL_ASSERT_bench(aFullSpec->NbBits()==12,"Bench_Target_Encoding");
    MMVII_INTERNAL_ASSERT_bench(aFullSpec->MinHammingD()==3,"Bench_Target_Encoding");
    MMVII_INTERNAL_ASSERT_bench(aFullSpec->EncodingFromName("toto")==nullptr,"Bench_Target_Encoding");
    MMVII_INTERNAL_ASSERT_bench(aFullSpec->EncodingFromCode((1<<12))==nullptr,"Bench_Target_Encoding"); // too big

    // -3-   check access to encoding from name/code 
    for (const auto & aNameT : std::vector<std::string>({"14","26","03"}))
    {
        const cOneEncoding * anEncod = aFullSpec->EncodingFromName(aNameT);
        MMVII_INTERNAL_ASSERT_bench(anEncod!=nullptr,"Bench_Target_Encoding");

        size_t aBitFlag = anEncod->Code();
        const cOneEncoding * anEncod2 = aFullSpec->EncodingFromCode(aBitFlag);
        MMVII_INTERNAL_ASSERT_bench(anEncod==anEncod2,"Bench_Target_Encoding");

        anEncod2 = aFullSpec->EncodingFromCode(aBitFlag+1);  // should ==0, due to hamming constraint
        MMVII_INTERNAL_ASSERT_bench(anEncod2==nullptr,"Bench_Target_Encoding");

        size_t anEquiCode  = N_LeftBitsCircPerm(aBitFlag,1<<12,6);
        anEncod2 = aFullSpec->EncodingFromCode(anEquiCode);  // should ==0, due to hamming constraint

        MMVII_INTERNAL_ASSERT_bench(anEncod==anEncod2,"Bench_Target_Encoding");
    }
}


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
    //  MMVII::AddData(cAuxAr2007("SzF",anAux),mSzF);
    MMVII::AddData(cAuxAr2007("CenterF",anAux),mCenterF);
    MMVII::AddData(cAuxAr2007("CornEl1",anAux),mCornEl1);
    MMVII::AddData(cAuxAr2007("CornEl2",anAux),mCornEl2);
    MMVII::AddData(cAuxAr2007("SzCCB",anAux),mSz_CCB);
    MMVII::AddData(cAuxAr2007("ThickN_WhiteInt",anAux),mThickN_WInt);
    MMVII::AddData(cAuxAr2007("ThickN_Code",anAux),mThickN_Code);
    MMVII::AddData(cAuxAr2007("ThickN_WhiteExt",anAux),mThickN_WExt);
    MMVII::AddData(cAuxAr2007("ThickN_BorderExt",anAux),mThickN_BorderExt);
    MMVII::AddData(cAuxAr2007("ThickN_Car",anAux),mThickN_Car);
    MMVII::AddData(cAuxAr2007("ChessBoardAngle",anAux),mChessboardAng);
    MMVII::AddData(cAuxAr2007("ModeFlight",anAux),mModeFlight);
    MMVII::AddData(cAuxAr2007("CBAtTop",anAux),mCBAtTop);
    MMVII::AddData(cAuxAr2007("WithChessBoard",anAux),mWithChessboard);
    MMVII::AddData(cAuxAr2007("WhiteBackGround",anAux),mWhiteBackGround);
    MMVII::AddData(cAuxAr2007("ZeroIsBackGround",anAux),mZeroIsBackGround);
    MMVII::AddData(cAuxAr2007("AntiClockWiseBit,",anAux),mAntiClockWiseBit);

    MMVII::AddData(cAuxAr2007("RayOrientTablet",anAux),mRayOrientTablet);
    MMVII::AddData(cAuxAr2007("CenterOrientTablet",anAux),mCenterOrientTablet);
    MMVII::AddData(cAuxAr2007("RayCenterMiniTarget",anAux),mRayCenterMiniTarget);

     if (anAux.Input())
	Finish();
}

void AddData(const  cAuxAr2007 & anAux,cParamCodedTarget & aPCT)
{
   aPCT.AddData(anAux);
}

void cParamCodedTarget::InitFromFile(const std::string & aNameFile)
{
    ReadFromFile(*this,aNameFile);
    // Finish();
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
   mType             (eTyCodeTarget::eIGNIndoor),  // used to create problem in reading serial, solved, but maintain fake-init
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
   mThickN_Car       (0.5),
   mThickN_BorderExt (0.05),
   mChessboardAng    (0.0),
   mWithChessboard   (true),
   mWhiteBackGround  (true),
   mZeroIsBackGround (true),
   mAntiClockWiseBit (true),
   mRayOrientTablet     (-1),
   mCenterOrientTablet  (0,0),
   mRayCenterMiniTarget (-1),
   mModeFlight       (false),  // MPD => def value was not initialized ?
   mCBAtTop          (false),//
   mDecP             ({1,1})  // "Fake" init 4 now
{
}

void cParamCodedTarget::FinishInitOfSpec(const cSpecBitEncoding & aSpec)
{
   mType = aSpec.mType;
   cMMVII_Appli & anAppli = cMMVII_Appli::CurrentAppli();

   if (aSpec.mType==eTyCodeTarget::eIGNIndoor)
   {
         // Nothingto do all default value have been setled for this case
   }
   else if ((aSpec.mType==eTyCodeTarget::eIGNDroneSym) || (aSpec.mType==eTyCodeTarget::eIGNDroneTop))
   {
       anAppli.SetIfNotInit(mModeFlight,true);
       anAppli.SetIfNotInit(mCBAtTop,(aSpec.mType==eTyCodeTarget::eIGNDroneTop));
       anAppli.SetIfNotInit(mThickN_WInt,0.1);
       anAppli.SetIfNotInit(mThickN_Code,0.0);
       anAppli.SetIfNotInit(mThickN_WExt,0.0);
       anAppli.SetIfNotInit(mThickN_Car,0.3);
       anAppli.SetIfNotInit(mChessboardAng,-M_PI/4.0);

       anAppli.SetIfNotInit(mRayOrientTablet,0.1);
       anAppli.SetIfNotInit(mCenterOrientTablet,cPt2dr(0.7,0));
   }
   else if (aSpec.mType==eTyCodeTarget::eCERN)
   {
       anAppli.SetIfNotInit(mNbBit,20);
       anAppli.SetIfNotInit(mWithParity,false);
       anAppli.SetIfNotInit(mNbRedond,1);
       anAppli.SetIfNotInit(mThickN_WInt,1.0);
       anAppli.SetIfNotInit(mThickN_Code,1.0);
       anAppli.SetIfNotInit(mThickN_WExt,0.0);
       anAppli.SetIfNotInit(mWithChessboard,false);
       anAppli.SetIfNotInit(mWhiteBackGround,false);
       anAppli.SetIfNotInit(mAntiClockWiseBit,true);
   }
   mThickN_Car *= (aSpec.mNbDigit+1)/2;
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


  aCumulThick = mRho_4_EndCar;
  aCumulThick += mThickN_BorderExt;
  mRho_EndIm = mSz_CCB * aCumulThick; 


  // mMidle = ToR(mSzBin-cPt2di(1,1)) / 2.0 - cPt2dr(1,1) ; //  pixel center model,suppose sz=2,  pixel 0 and 1 => center is 0.5
  cPt2di aSz4Mid =  mCBAtTop ? cPt2di(mNbPixelBin,mNbPixelBin) : mSzBin ;
  mMidle = ToR(aSz4Mid-cPt2di(mSzGaussDeZoom,mSzGaussDeZoom)) / 2.0  ; //  pixel center model,suppose sz=2,  pixel 0 and 1 => center is 0.5
								     
  mScale = mNbPixelBin / (2.0 * mRho_EndIm);


  // mCenterF = mMidle / double(mSzGaussDeZoom);
  mCenterF = Norm2PixR(FromPolar(0.0,0.0))  / double(mSzGaussDeZoom);
  mSignAngle = (mAntiClockWiseBit ? 1 :-1);

  mCornEl1 = Norm2PixR(FromPolar(mRho_0_EndCCB,mChessboardAng))  / double(mSzGaussDeZoom);
  mCornEl2 = Norm2PixR(FromPolar(mRho_0_EndCCB,mChessboardAng+M_PI/2.0))  / double(mSzGaussDeZoom);

  //mCornEl1 = mCenterF + FromPolar




  if (false)
  {
      StdOut() <<  "r0 : " << mRho_0_EndCCB << "\n"
               <<  "r1 : " << mRho_1_BeginCode << "\n"
               <<  "r2 : " << mRho_2_EndCode << "\n"
               <<  "r3 : " << mRho_3_BeginCar << "\n"
               <<  "r4 : " << mRho_4_EndCar << "\n"
	       <<  "r5 : " << mRho_EndIm << "\n";

      getchar();
  }
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
	 tREAL8 mSignT;
};



cCircNP2B::cCircNP2B(const cFullSpecifTarget & aSpecif) :
   mRho0   (aSpecif.Render().mRho_1_BeginCode),
   mRho1   (aSpecif.Render().mRho_2_EndCode),
   mTeta0  (aSpecif.Render().mChessboardAng),
   mNbBits (aSpecif.Specs().mNbBits),
   mSignT  (aSpecif.Render().mSignAngle)
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
     tREAL8 aTeta = (PreProcessCoord(aPt).y() -mTeta0) * mSignT;
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
         cStraightNP2B(const cFullSpecifTarget & aSpecif,bool IsSym);
	 bool    PNormIsCoding(const cPt2dr & aPt)      const   override;
	 int     BitsOfNorm    (const cPt2dr & aPt)     const   override;

     private :
         bool     mIsSym;
	 tREAL8   mRho1;
         int      mNbBits;
         int      mNbBS2;
	 tREAL8   mSep2L;
};

cStraightNP2B::cStraightNP2B(const cFullSpecifTarget & aSpecif,bool IsSym) :
   mIsSym  (IsSym),
   mRho1   (aSpecif.Render().mRho_EndIm),
   mNbBits (aSpecif.Specs().mNbBits),
   mNbBS2  (mNbBits /2),
   mSep2L  ( IsSym ? 0 : (mRho1 * (1+sqrt(2)/4)))
   
{
    MMVII_INTERNAL_ASSERT_tiny((mNbBS2*2)==mNbBits,"Odd nbbits in cStraightNP2B");
     // StdOut() << "r1=" << mRho1 << " I=" << aSpecif.Render().mRho_EndIm  << " p2n:" << aSpecif.Render().Pix2Norm(cPt2di(0,0)) << "\n";
}

bool    cStraightNP2B::PNormIsCoding(const cPt2dr & aPt) const   
{
    return  mIsSym                      ?
	    (std::abs(aPt.y()) > mRho1) : 
	    (aPt.y() >  mRho1)        ;
}

int   cStraightNP2B::BitsOfNorm(const cPt2dr & aPt) const
{

    int aRes = round_down((aPt.x()+mRho1)/(2*mRho1) *mNbBS2)  ;
    aRes = std::max(0,std::min(aRes,mNbBS2-1));


    bool  isLine2 =   (aPt.y()>mSep2L)  ;

   aRes =  aRes +  mNbBS2* isLine2;

   // StdOut()  << "rrr = " << aRes << " " << (aPt.x()+mRho1)/(2*mRho1) << "\n";
   return aRes;
}

/* *************************************************** */
/*                                                     */
/*             cNormPix2Bit                            */
/*                                                     */
/* *************************************************** */

cNormPix2Bit * cNormPix2Bit::Alloc(const cFullSpecifTarget & aSpecif)
{
   switch (aSpecif.Type())
   {
         case eTyCodeTarget::eCERN :
         case eTyCodeTarget::eIGNIndoor:
	       return new cCircNP2B(aSpecif);
         case eTyCodeTarget::eIGNDroneSym:
	       return new cStraightNP2B(aSpecif,true);

         case eTyCodeTarget::eIGNDroneTop:
	       return new cStraightNP2B(aSpecif,false);
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

          cCodedTargetPatternIm(cFullSpecifTarget &);

	  tIm  ImCoding() const;

	  tIm MakeOneImTarget(const cOneEncoding & aCode,bool doMarkC = false);
     private :
	  cCodedTargetPatternIm(const cCodedTargetPatternIm &) = delete;
	  cPt2di  PDiag(tREAL8 aRhoNorm) const;


	  cFullSpecifTarget &     mSpec;
	  const cParamRenderingTarget&  mRender;
	  cPt2di                        mSzBin;
	  tIm                     mImCoding;
	  tDataIm &               mDIC;

	  tIm               mImTarget;
	  tDataIm &         mDIT;

	  tREAL8            mTeta0;
	  tREAL8            mRhoC;
	  tREAL8            mRho2C;
          bool              mWOriTab;
          tREAL8            mRayOT;
          tPt2dr            mCenterOT;
          tREAL8            mRayCT;
          tREAL8            mRay2CT;
};

cPt2di  cCodedTargetPatternIm::PDiag(tREAL8 aRhoNorm) const
{
       tREAL8 r =  aRhoNorm ;
       cPt2dr aP3r(-r,-r);
       cPt2di  aP3I = mRender.Norm2PixI(aP3r);

       return  mDIC.Proj(aP3I);
}


cCodedTargetPatternIm::cCodedTargetPatternIm(cFullSpecifTarget & aSpec) :
     mSpec       (aSpec),
     mRender     (mSpec.Render()),
     mSzBin      (mRender.mSzBin),
     mImCoding   (mSzBin),
     mDIC        (mImCoding.DIm()),
     mImTarget   (mSzBin),
     mDIT        (mImTarget.DIm()),
     mTeta0      (mRender.mChessboardAng),
     mRhoC       (mRender.mRho_0_EndCCB),
     mRho2C      (Square(mRhoC)),
     mWOriTab    (mRender.mRayOrientTablet >0),
     mRayOT      (mRender.mRayOrientTablet),
     mCenterOT   (mRender.mCenterOrientTablet),
     mRayCT      (mRender.mRayCenterMiniTarget),
     mRay2CT     (Square(mRayCT))
{
    mDIC.InitCste(tElem(eLPT::eBackGround));

    // Structures for computing center of bits
    std::vector<cPt2dr>  aVCenters =  mSpec.BitsCenters();
    std::vector<tREAL8>  aVWeight(mSpec.NbBits(),0.0);

    // structure specifying bits location
    std::unique_ptr<cNormPix2Bit>  aP2B (cNormPix2Bit::Alloc(aSpec));

    for (const auto & aPix : mDIC)
    {
       cPt2dr aPN = mSpec.Render().Pix2Norm(aPix);
       //  ============  1  Generate the bit coding =======================
       if (aP2B->PNormIsCoding(aPN))  // if point belong to bit-coding space
       {
           int aNumB =  aP2B->BitsOfNorm(aPN);  // get value of bit
	   mDIC.SetV(aPix,int(eLPT::eNumB0)+aNumB);  // marq the image with num

	   aVWeight.at(aNumB) += 1 ;  // increment number of point for this bit
	   aVCenters.at(aNumB) += ToR(aPix);  // accumulate for centroid
       }
       //  ============  2  Generate the central circle =======================
       else if (SqN2(aPN) <mRho2C)
       {
           eLPT aLab = eLPT::eForeGround;  // a priori mar circle
           if (mSpec.Render().mWithChessboard)
	   {
               // computation to separate the plane in 4 quadrant
               double PIsur2 = M_PI/2.0;
	       tREAL8 aTeta = ToPolar(aPN).y();
               int aIndTeta = round_down((aTeta-mTeta0)/PIsur2);
               if ((aIndTeta%2)==0)
                   aLab = eLPT::eBackGround;
	   }
           if (mWOriTab && (Norm1(aPN-mCenterOT) < mRayOT))
           {
		   // StdOut() <<  "RRRoot " << aPix << aPN  << mCenterOT<< " \n";
              aLab = eLPT::eForeGround;
           }

	   if ((mRayCT>0) && (SqN2(aPN)<mRay2CT))
	   {
               tREAL8 aLogR = (std::log(std::max(1.0/mRender.mScale,Norm2(aPN))/mRayCT))/std::log(2) ;
	       int aILogR = round_down(aLogR);
               aLab = ((aILogR % 2)!=0) ?  eLPT::eBackGround : eLPT::eForeGround;
	   }
	   mDIC.SetV(aPix,int(aLab));
       }
    }

    // compute and memorize the center
    for (size_t aB=0 ; aB< aVWeight.size() ; aB++)
    {
       mSpec.SetBitCenter(aB,aVCenters.at(aB) / tREAL8(aVWeight.at(aB) * mSpec.DeZoomIm() ));
    }
}

cCodedTargetPatternIm::tIm cCodedTargetPatternIm::ImCoding() const {return mImCoding;}

cCodedTargetPatternIm::tIm cCodedTargetPatternIm::MakeOneImTarget(const cOneEncoding & anEnCode,bool doMarkC)
{
   // compute gray level for background & foreground
   int aBG_Coul = mSpec.Render().mWhiteBackGround ? 255 : 0;
   int aFG_Coul =  255-aBG_Coul;

   // by default all is backgrounf
   mDIT.InitCste(aBG_Coul);

   size_t aCode = anEnCode.Code();
   bool  BGIs_0 = mSpec.Render().mZeroIsBackGround;  // true mean bits 0 are considered background
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

   // ---------------  caracter insertion  --------------------

   {
        //cPt2di  aSzFt(10,10);
        std::string aFullName = anEnCode.Name();
	size_t aIndSplit = (aFullName.size() +1)/2;

	// Corners of string, 
        cPt2di  aP00 = PDiag(mRender.mRho_4_EndCar);
	cPt2di  aP11 = PDiag(mRender.mRho_4_EndCar-mRender.mThickN_Car) ; // for a 1 length caracr

	// udate highth of string, to adapt to length (aIndSplit is maximal legnt of 2 substrings)
	int aHigth = (aP11.y()-aP00.y()) / aIndSplit;
	aP11 =  cPt2di(aP11.x(),aP00.y() + aHigth);

	// loop for processing the 2 subsrt string 
	for (const auto & aLeftSS : {true,false} )
	{
             int aFG_StrCoul = aFG_Coul;
             // extract one of the 2 substr
             std::string aName = aFullName.substr((aLeftSS?0:aIndSplit) , (aLeftSS?aIndSplit:std::string::npos));
	     // get image of string
	     cIm2D<tU_INT1>  aImStr = ImageOfString_10x8(aName,1);
	     cDataIm2D<tU_INT1>&  aDImStr =  aImStr.DIm();

	     // compute the size of each pixel to fill the hights
	     tREAL8 aSzPixStr = aHigth / tREAL8(aDImStr.Sz().y());

	     cPt2di aPOri = aP00;
	     if (! aLeftSS)
	     {
	          int aWitdhStr = round_ni(aDImStr.Sz().x())*aSzPixStr;  // Width of string
	          //  Off + WidtStr/2 =  SzBin.x - P4.x
	          int OffsX  = mSzBin.x()  - aP00.x() -  aWitdhStr;
		  aPOri = cPt2di(OffsX,aP00.y());
	     }

	     cPt2di  aP4Sym = ToI(mSpec.Render().mMidle * 2.0) - cPt2di(1,1);

	     for (const auto & aPixStr : aDImStr)
	     {
                  if (aDImStr.GetV(aPixStr))
	          {
                       cPt2di aP0 = aPOri+ToI(ToR(aPixStr)*aSzPixStr);
                       cPt2di aP1 = aPOri+ToI(ToR(aPixStr+cPt2di(1,1))*aSzPixStr);

		       for (const auto& aPixIm : cRect2(aP0,aP1))
		       {
                           mDIT.SetV(aPixIm,aFG_StrCoul);
                           mDIT.SetV(aP4Sym-aPixIm,aFG_StrCoul);
		       }
	          }
	     }

	}
   }


   tIm aRes = mImTarget.GaussDeZoom(mSpec.DeZoomIm());

   // in debug mode, marq with one pixel the center
   if (doMarkC)
   {
      for (const auto & aC : mSpec.BitsCenters())
      {
	      aRes.DIm().SetV(ToI(aC),128);
      }
   }

   return aRes;
}

/* *************************************************** */
/*                                                     */
/*                cFullSpecifTarget                    */
/*                                                     */
/* *************************************************** */

cFullSpecifTarget::cFullSpecifTarget() :
   mCEC         (nullptr),
   mCTPI        (nullptr)
{
}

cFullSpecifTarget::cFullSpecifTarget(const cBitEncoding& aBE,const cParamRenderingTarget& aRender) :
   //cFullSpecifTarget(),
   mCEC         (nullptr),
   mCTPI        (nullptr),
   mBE          (aBE),
   mRender      (aRender),
   mBitsCenters (NbBits(),cPt2dr(0,0))
{
}

cCodedTargetPatternIm * cFullSpecifTarget::AllocCTPI()
{
    if (mCTPI==nullptr)
       mCTPI = new  cCodedTargetPatternIm(*this);

    return mCTPI;
}

cCompEquiCodes *   cFullSpecifTarget::CEC() const
{
    if (mCEC==nullptr)
    {
       size_t aPer = NbBits()/Specs().mFreqCircEq;
       mCEC = cCompEquiCodes::Alloc(NbBits(),aPer);
    }

    return mCEC;
}


cFullSpecifTarget::tIm   cFullSpecifTarget::ImagePattern()
{
	return AllocCTPI()->ImCoding();
}
cFullSpecifTarget::tIm   cFullSpecifTarget::OneImTarget(const cOneEncoding & aCode)
{
	return AllocCTPI()->MakeOneImTarget(aCode);
}



cFullSpecifTarget::~cFullSpecifTarget()
{
    delete mCTPI;
    delete mCEC;
}

const std::vector<cOneEncoding> & cFullSpecifTarget::Encodings()   const {return mBE.Encodings();}
const cSpecBitEncoding &          cFullSpecifTarget::Specs()       const {return mBE.Specs();}
const cParamRenderingTarget &     cFullSpecifTarget::Render()      const {return mRender;}
const std::string &               cFullSpecifTarget::Prefix()      const {return Specs().mPrefix;}
const std::vector<cPt2dr>&        cFullSpecifTarget::BitsCenters() const {return mBitsCenters;}
size_t                            cFullSpecifTarget::NbBits()      const {return Specs().mNbBits;}
int                               cFullSpecifTarget::DeZoomIm()    const {return mRender.mSzGaussDeZoom;}
eTyCodeTarget                     cFullSpecifTarget::Type()        const {return Specs().mType;}
size_t                            cFullSpecifTarget::MinHammingD() const {return Specs().mMinHammingD;}

tREAL8 cFullSpecifTarget::Rho_0_EndCCB() const    {return mRender.mRho_0_EndCCB;}
tREAL8 cFullSpecifTarget::Rho_1_BeginCode() const {return mRender.mRho_1_BeginCode;}
tREAL8 cFullSpecifTarget::Rho_2_EndCode() const   {return mRender.mRho_2_EndCode;}


const cPt2dr & cFullSpecifTarget::Center() const {return mRender.mCenterF;}
const cPt2dr & cFullSpecifTarget::CornerlEl_BW() const {return mRender.mCornEl1;}
const cPt2dr & cFullSpecifTarget::CornerlEl_WB() const {return mRender.mCornEl2;}
bool  cFullSpecifTarget::AntiClockWiseBit() const { return mRender.mAntiClockWiseBit; }

bool cFullSpecifTarget::BitIs1(bool IsWhite) const
{
            //  For example 3D ICON with white pixel
	    //     true          false                     true       
	    //      !(T^ F ^T)  => 1   and that 's the case, Youpi !!
    return  !(   (IsWhite ^  mRender.mWhiteBackGround)  ^ mRender.mZeroIsBackGround );
}




void  cFullSpecifTarget::SetBitCenter(size_t aBit,const cPt2dr& aC)
{
      mBitsCenters.at(aBit) = aC;
}


std::string cFullSpecifTarget::NameOfImPattern() const
{
   return  Prefix() + "_Pattern"+ ".tif"; 
}

std::string cFullSpecifTarget::NameOfEncode(const cOneEncoding & anEncode) const
{
    return   Prefix()  +  "_Target_"+  anEncode.Name() +   ".tif";
}

void cFullSpecifTarget::AddData(const  cAuxAr2007 & anAux)
{
     mBE.AddData(cAuxAr2007("BitEncoding",anAux));
     mRender.AddData(cAuxAr2007("Geometry",anAux));
     StdContAddData(cAuxAr2007("Centers",anAux),mBitsCenters);
}

void AddData(const  cAuxAr2007 & anAux,cFullSpecifTarget & aSpecif)
{
   aSpecif.AddData(anAux);
}

cFullSpecifTarget *  cFullSpecifTarget::CreateFromFile(const std::string & aName)
{
    cFullSpecifTarget * aRes = new cFullSpecifTarget;
    ReadFromFile(*aRes,aName);

    return aRes;
}

const cOneEncoding * cFullSpecifTarget::EncodingFromCode(size_t aBitFlag) const
{
   // 1 Extract the low-standard value, if it exist
   const cCelCC * aCell = CEC()->CellOfCode(aBitFlag);
   if (aCell==nullptr) return nullptr;

   aBitFlag = aCell->mLowCode;

   // 2 Extract the encoding with adequat code
   for (const auto & anEncod : Encodings())
   {
       if (anEncod.Code() == aBitFlag)
          return &anEncod;
   }

   //  none was found
   return nullptr;
}

const cOneEncoding * cFullSpecifTarget::EncodingFromName(const std::string &aName) const
{
   //  Extract the encoding with adequate name
   for (const auto & anEncod : Encodings())
       if (anEncod.Name() == aName)
          return &anEncod;

   //  none was found
   return nullptr;
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
	int                mZoomShow;
	std::string        mNameBE;
	cBitEncoding       mBE;

	cParamCodedTarget  mPCT;
	bool               mDoMarkC;
	std::string        mPatternDoImage;
};

eTyCodeTarget cAppliGenCodedTarget::Type() {return mBE.Specs().mType ;}


cAppliGenCodedTarget::cAppliGenCodedTarget(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli  (aVArgs,aSpec),
   mPerGen       (10),
   mDoMarkC      (false)
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
          << AOpt2007(mPatternDoImage,"PatIm","Pattern for generating image (def no generation)")
          << AOpt2007(mPCT.mRayCenterMiniTarget,"RayMCT","Rayon \"mini\" center target (for topo)",{eTA2007::HDV})
          << AOpt2007(mPCT.mNbBit,"NbBit","Nb Bit printed",{eTA2007::HDV})
          << AOpt2007(mPCT.mWithParity,"WPar","With parity bit",{eTA2007::HDV})
          << AOpt2007(mPCT.mThickN_WInt,"ThW0","Thickness of interior white circle",{eTA2007::HDV})
          << AOpt2007(mPCT.mThickN_Code,"ThCod","Thickness of bin-coding black circle",{eTA2007::HDV})
          << AOpt2007(mPCT.mThickN_WExt,"ThSepCar","Thickness of sep bin-code / ahpha code",{eTA2007::HDV})
          << AOpt2007(mPCT.mThickN_Car,"ThCar","Thickness of separation alpha ccode ",{eTA2007::HDV})
          << AOpt2007(mPCT.mThickN_BorderExt,"ThBExt","Thickness of border exterior",{eTA2007::HDV})
          << AOpt2007(mPCT.mChessboardAng,"Theta","Origin angle of chessboard pattern ",{eTA2007::HDV})
          << AOpt2007(mPCT.mModeFlight,"ModeFlight","Special mode for Patricio ",{eTA2007::HDV})
          << AOpt2007(mDoMarkC,"MarkC","Mark center of bits, just for verif ",{eTA2007::HDV,eTA2007::Tuning})
          << AOpt2007(mZoomShow,"ZoomShow","Zoom to generate a high resolution check images",{eTA2007::Tuning})
   ;
}


int  cAppliGenCodedTarget::Exe()
{
    //  Bench_Target_Encoding();


   ReadFromFile(mBE,mNameBE);

   mPCT.FinishInitOfSpec(mBE.Specs());
   mPCT.Finish();

   cFullSpecifTarget  aFullSpec(mBE,mPCT);

   // Activate the computaion of centers
   aFullSpec.ImagePattern();

   if (IsInit(&mPatternDoImage))
   {
      //  generate the pattern image
      aFullSpec.ImagePattern().DIm().ToFile(aFullSpec.NameOfImPattern());

      // parse all encodings
      for (const auto & anEncode : aFullSpec.Encodings())
      {
          if (MatchRegex(anEncode.Name(),mPatternDoImage))
	  {
             cCodedTargetPatternIm::tIm anIm = aFullSpec.OneImTarget(anEncode);

             std::string aName = aFullSpec.NameOfEncode(anEncode);
             anIm.DIm().ToFile(aName);
             StdOut() << aName << "\n";
	  }
      }
   }

   std::string aName = aFullSpec.Prefix()+"_FullSpecif.xml";
   SaveInFile(aFullSpec, aName);

   if (IsInit(&mZoomShow))
   {
      TestReloadAndShow_cFullSpecifTarget(aName,mZoomShow);
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


